"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Exports Diorama's two perception networks to ONNX, in the form every jetlink
backend runs (TensorRT, onnxruntime with CoreML, onnxruntime's CPU provider):

- diorama_seg.onnx: Mask2Former (Swin-L, Mapillary Vistas panoptic).
- diorama_dd3d.onnx: DD3D (VoVNet-99, KITTI), monocular 3D boxes, read for
  the heading of each car.

Both take the frame exactly as the comma sends it in an INFER_REQ: the narrow
camera warped to the driving model's 512x256 frame and packed into six
128x256 planes (four of Y, then U and V), as float16 0..255. The YUV unpack,
the colour conversion and the post-processing are in the graph, so a server
copies the bytes in and the numbers out, and every output is float32 with a
fixed shape:

    seg   qmap    (1, 128, 256)  query index per pixel at half resolution, 255 for none
          label   (200,)         each query's Mapillary class
          score   (200,)         its score, 0 when it was dropped
    dd3d  boxes   (100, 16)      the 100 best detections, see DD3D_COLUMNS

The thresholds are baked in at export and written to diorama.json next to the
graphs, with the class names and the camera the boxes were decoded for.

    python -m diorama export --out ~/diorama-models --dd3d-weights model_final.pth
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SEG_MODEL = 'facebook/mask2former-swin-large-mapillary-vistas-panoptic'
DD3D_WEIGHTS_URL = 'https://tri-ml-public.s3.amazonaws.com/github/dd3d/experiments/4elbgev2-20210825_201852/model_final.pth'

# The driving model's narrow frame: openpilot's medmodel intrinsics at 512x256.
FRAME_W, FRAME_H = 512, 256
MEDMODEL_FL, MEDMODEL_CY = 910.0, 47.6
PACKED_SHAPE = (1, 6, FRAME_H // 2, FRAME_W // 2)

# Mask2Former: a thing (car, person...) query needs THING_SCORE, stuff (road,
# paint...) STUFF_SCORE; a pixel needs ASSIGN of its best query's mask.
THING_SCORE, STUFF_SCORE, ASSIGN = 0.8, 0.5, 0.3
SEG_DS = 2  # the query map is at half the frame's resolution

# Mapillary classes the road map draws (1 road, 2 lane paint, 3 crosswalk, 4 curb, 5 sidewalk)...
ROAD_CLASS = {'Road': 1, 'Bike Lane': 1, 'Service Lane': 1, 'Lane Marking - General': 2, 'Lane Marking - Crosswalk': 3,
              'Crosswalk - Plain': 3, 'Curb': 4, 'Curb Cut': 4, 'Sidewalk': 5}
# ...and the things it places, by the name the renderer knows them by.
THING = {'Car': 'car', 'Truck': 'truck', 'Bus': 'bus', 'Caravan': 'truck', 'Trailer': 'truck', 'Other Vehicle': 'car',
         'Motorcycle': 'motorcycle', 'Bicycle': 'bicycle', 'Person': 'person'}

DD3D_TOPK = 100
DD3D_PRE_NMS = 0.05
# One row per detection: the 2D box in frame pixels, class, 2D score, 3D score,
# centre (camera frame: x right, y down, z forward, metres), the object's
# forward axis in that frame, and its size (width, length, height).
DD3D_COLUMNS = ['x1', 'y1', 'x2', 'y2', 'cls', 'score', 'score3d', 'cx', 'cy', 'cz', 'fx', 'fy', 'fz', 'w', 'l', 'h']


def frame_intrinsics() -> np.ndarray:
  return np.array([[MEDMODEL_FL, 0, FRAME_W / 2], [0, MEDMODEL_FL, MEDMODEL_CY], [0, 0, 1]], np.float64)


def _torch():
  import torch
  from torch import nn
  return torch, nn


def unpack_rgb(packed):
  """(1, 6, 128, 256) packed YUV 0..255 -> (1, 3, 256, 512) RGB 0..255, as cv2's I420 conversion (BT.601, video range)."""
  torch, _ = _torch()
  # the four Y planes are Y[0::2, 0::2], Y[1::2, 0::2], Y[0::2, 1::2], Y[1::2, 1::2]; pixel_shuffle wants
  # channel 2 * row parity + column parity
  y = torch.nn.functional.pixel_shuffle(packed[:, [0, 2, 1, 3]], 2)
  uv = torch.nn.functional.interpolate(packed[:, 4:6], scale_factor=2.0, mode='nearest')
  c = (y - 16.0) * 1.164
  u, v = uv[:, 0:1] - 128.0, uv[:, 1:2] - 128.0
  r = c + 1.596 * v
  g = c - 0.391 * u - 0.813 * v
  b = c + 2.018 * u
  return torch.cat([r, g, b], 1).clamp(0.0, 255.0)


def _float32_reference_points(net):
  """The pixel decoder's reference points as float32 constants. The exporter turns torch.linspace into float64
  arithmetic, and a float64 grid reaches GridSample, which neither onnxruntime's providers nor CoreML run. The
  frame's size is fixed, so the points are too (no padding: every valid ratio is 1)."""
  torch, _ = _torch()
  encoder = net.model.pixel_level_module.decoder.encoder

  def reference_points(spatial_shapes_list, valid_ratios, device):
    points = []
    for height, width in spatial_shapes_list:
      ys, xs = np.meshgrid((np.arange(height) + 0.5) / height, (np.arange(width) + 0.5) / width, indexing='ij')
      points.append(np.stack([xs.ravel(), ys.ravel()], -1))
    ref = torch.from_numpy(np.concatenate(points).astype(np.float32)).to(device)[None]
    return ref[:, :, None].expand(valid_ratios.shape[0], -1, len(spatial_shapes_list), 2)

  encoder.get_reference_points = reference_points


def _rank5_swin_windows():
  """Swin's window partition and reverse without a 6-D tensor: the batch and
  the row of windows sit side by side, so they merge into one axis and the
  result is the same. onnxruntime's CoreML provider takes no Reshape past
  rank 5, and every 6-D one cut the graph, 85 pieces with a CPU hop between
  each, so the segmentation ran no faster on the GPU than on the CPU."""
  from transformers.models.swin import modeling_swin

  def window_partition(x, window_size):
    batch_size, height, width, num_channels = x.shape
    x = x.reshape(batch_size * (height // window_size), window_size, width // window_size, window_size, num_channels)
    return x.transpose(1, 2).contiguous().view(-1, window_size, window_size, num_channels)

  def window_reverse(windows, window_size, height, width):
    num_channels = windows.shape[-1]
    windows = windows.view(-1, width // window_size, window_size, window_size, num_channels)
    return windows.transpose(1, 2).contiguous().view(-1, height, width, num_channels)

  modeling_swin.window_partition = window_partition
  modeling_swin.window_reverse = window_reverse


def _rank5_deformable_attention():
  """The pixel decoder's deformable attention without a 6-D tensor, for the
  same reason as the Swin windows: its sampling offsets and locations were
  (batch, query, head, level, point, 2). Batch and query sit side by side
  and merge, and the arithmetic is unchanged. Only 2-D reference points,
  which `_float32_reference_points` makes them."""
  torch, nn = _torch()
  from transformers.models.mask2former import modeling_mask2former

  def forward(self, hidden_states, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None,
              position_embeddings=None, reference_points=None, spatial_shapes_list=None, level_start_index=None,
              output_attentions=False):
    if position_embeddings is not None:
      hidden_states = self.with_pos_embed(hidden_states, position_embeddings)
    batch_size, num_queries, _ = hidden_states.shape
    _, sequence_length, _ = encoder_hidden_states.shape
    heads, levels, points = self.n_heads, self.n_levels, self.n_points
    hidden_dim = self.d_model // heads
    value = self.value_proj(encoder_hidden_states)
    if attention_mask is not None:
      value = value.masked_fill(attention_mask[..., None], float(0))
    value = value.view(batch_size, sequence_length, heads, hidden_dim)
    offsets = self.sampling_offsets(hidden_states).view(batch_size * num_queries, heads, levels, points, 2)
    weights = nn.functional.softmax(self.attention_weights(hidden_states).view(batch_size, num_queries, heads, levels * points), -1)
    normalizer = torch.tensor([[w, h] for h, w in spatial_shapes_list], dtype=torch.float32, device=offsets.device)
    grids = 2 * (reference_points.reshape(batch_size * num_queries, 1, levels, 1, 2) + offsets / normalizer.view(1, 1, levels, 1, 2)) - 1

    values = value.split([h * w for h, w in spatial_shapes_list], dim=1)
    sampled = []
    for level, (height, width) in enumerate(spatial_shapes_list):
      value_l = values[level].flatten(2).transpose(1, 2).reshape(batch_size * heads, hidden_dim, height, width)
      # a slice, not an index: an index exports as a Gather, which CoreML does not take
      grid_l = grids[:, :, level:level + 1].reshape(batch_size, num_queries, heads, points, 2).transpose(1, 2).flatten(0, 1)
      sampled.append(nn.functional.grid_sample(value_l, grid_l, mode='bilinear', padding_mode='zeros', align_corners=False))
    weights_bh = weights.transpose(1, 2).reshape(batch_size * heads, 1, num_queries, levels * points)
    output = (torch.stack(sampled, dim=-2).flatten(-2) * weights_bh).sum(-1).view(batch_size, heads * hidden_dim, num_queries)
    output = self.output_proj(output.transpose(1, 2).contiguous())
    return output, weights.view(batch_size, num_queries, heads, levels, points)

  modeling_mask2former.Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention.forward = forward


def seg_module(model_id: str = SEG_MODEL):
  torch, nn = _torch()
  from transformers import Mask2FormerForUniversalSegmentation

  _rank5_swin_windows()
  _rank5_deformable_attention()
  net = Mask2FormerForUniversalSegmentation.from_pretrained(model_id).eval()
  _float32_reference_points(net)
  labels = {int(k): v for k, v in net.config.id2label.items()}
  is_thing = torch.tensor([labels[i] in THING for i in range(len(labels))])

  class Seg(nn.Module):
    def __init__(self):
      super().__init__()
      self.net = net
      self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255)
      self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255)
      self.register_buffer('is_thing', is_thing)

    def forward(self, packed):
      x = (unpack_rgb(packed.float()) - self.mean) / self.std
      o = self.net(pixel_values=x)
      score, label = o.class_queries_logits[0].float().softmax(-1)[:, :-1].max(-1)
      thresh = torch.where(self.is_thing[label], torch.tensor(THING_SCORE), torch.tensor(STUFF_SCORE))
      score = torch.where(score > thresh, score, torch.zeros_like(score))
      size = (FRAME_H // SEG_DS, FRAME_W // SEG_DS)
      m = torch.nn.functional.interpolate(o.masks_queries_logits.float(), size, mode='bilinear', align_corners=False)[0]
      m = m.sigmoid() * score[:, None, None]
      best, q = m.max(0)
      qmap = torch.where(best > ASSIGN, q.float(), torch.full_like(best, 255.0))
      return qmap[None], label.float(), score

  meta = {
    'model': model_id, 'labels': [labels[i] for i in range(len(labels))],
    'road_class': {name: ROAD_CLASS[name] for name in labels.values() if name in ROAD_CLASS},
    'things': {name: THING[name] for name in labels.values() if name in THING},
    'thing_score': THING_SCORE, 'stuff_score': STUFF_SCORE, 'assign': ASSIGN, 'ds': SEG_DS,
  }
  return Seg().eval(), meta


def dd3d_module(weights: str):
  torch, nn = _torch()
  from diorama.dd3d import CLASSES, DD3D, STRIDES, quat_to_matrix, DEPTH_FOCAL_FACTOR, MIN_DEPTH, MAX_DEPTH

  net = DD3D.from_checkpoint(weights, device='cpu')
  K = torch.tensor(frame_intrinsics(), dtype=torch.float32)
  Kinv = torch.linalg.inv(K)
  pixel_size = float(torch.linalg.norm(torch.stack([Kinv[0, 0], Kinv[1, 1]])))
  nc = net.nc

  class Boxes(nn.Module):
    def __init__(self):
      super().__init__()
      self.net = net
      self.register_buffer('Kinv', Kinv)
      # every location's centre, level by level, in the order the heads' outputs are flattened
      locs = []
      for s in STRIDES:
        h, w = -(-FRAME_H // s), -(-FRAME_W // s)
        ys, xs = torch.meshgrid(torch.arange(h) * s, torch.arange(w) * s, indexing='ij')
        locs.append(torch.stack([xs.reshape(-1), ys.reshape(-1)], 1).float())
      self.register_buffer('loc', torch.cat(locs))

    def forward(self, packed):
      rgb = unpack_rgb(packed.float())
      x = (rgb[:, [2, 1, 0]] - self.net.pixel_mean) / self.net.pixel_std  # the network was trained on BGR
      feats = self.net.backbone(x)
      out2d, out3d = self.net.fcos2d_head(feats), self.net.fcos3d_head(feats)
      flat = lambda t, c: t[0].permute(1, 2, 0).reshape(-1, c)  # noqa: E731
      cat = lambda i, c, heads: torch.cat([flat(h[i], c) for h in heads])  # noqa: E731
      scores = cat(0, nc, out2d).sigmoid() * cat(2, 1, out2d).sigmoid()
      best, cls = scores.max(1)
      reg = cat(1, 4, out2d)
      quat, pctr, depth = cat(0, 4 * nc, out3d), cat(1, 2 * nc, out3d), cat(2, nc, out3d)
      size, conf = cat(3, 3 * nc, out3d), cat(4, nc, out3d)

      def pick(t, k):  # the predicted class's k values of each location
        n = t.shape[0]
        return torch.gather(t.view(n, k, nc), 2, cls.view(n, 1, 1).expand(n, k, 1))[:, :, 0]

      score2d = best.sqrt()
      score3d = score2d * pick(conf, 1)[:, 0].sigmoid()
      top = torch.topk(torch.where(best > DD3D_PRE_NMS, score3d, torch.zeros_like(score3d)), DD3D_TOPK).indices
      loc, r = self.loc[top], reg[top]
      box = torch.stack([loc[:, 0] - r[:, 0], loc[:, 1] - r[:, 1], loc[:, 0] + r[:, 2], loc[:, 1] + r[:, 3]], 1)
      c = cls[top]
      q = pick(quat, 4)[top]
      q = q / q.norm(dim=1, keepdim=True).clamp(min=1e-7)
      pc = pick(pctr, 2)[top] + loc
      dp = (pick(depth, 1)[top, 0] / (pixel_size * DEPTH_FOCAL_FACTOR)).clamp(MIN_DEPTH, MAX_DEPTH)
      ray = (self.Kinv @ torch.cat([pc, torch.ones_like(pc[:, :1])], 1).T).T
      zax = ray / ray.norm(dim=1, keepdim=True)
      yax = torch.tensor([0.0, 1.0, 0.0]) - zax[:, 1:2] * zax
      yax = yax / yax.norm(dim=1, keepdim=True)
      xax = torch.cross(yax, zax, dim=1)
      R = torch.stack([xax, yax, zax], -1) @ quat_to_matrix(q)
      wlh = (pick(size, 3)[top].tanh() + 1.0) * self.net.canon[c]
      return torch.cat([box, c[:, None].float(), score2d[top, None], score3d[top, None], ray * dp[:, None], R[:, :, 0], wlh], 1)

  meta = {'classes': list(CLASSES), 'topk': DD3D_TOPK, 'columns': DD3D_COLUMNS, 'weights': DD3D_WEIGHTS_URL}
  return Boxes().eval(), meta


def export_onnx(module, path: Path, output_names: list[str]):
  import time

  import onnx
  torch, _ = _torch()
  x = torch.full(PACKED_SHAPE, 128.0, dtype=torch.float16)
  with torch.no_grad():
    torch.onnx.export(module, (x,), str(path), input_names=['frame'], output_names=output_names, opset_version=17,
                      do_constant_folding=True, dynamo=False)
  # CoreML keeps its compile of a graph under this key: a new export must not reuse an old one's
  model = onnx.load(str(path))
  key = 'diorama' + path.stem.split('_')[-1] + time.strftime('%Y%m%d%H%M%S')
  model.metadata_props.add(key='COREML_CACHE_KEY', value=key)
  onnx.save(model, str(path))


def export(out: Path, dd3d_weights: str | None, seg_model: str = SEG_MODEL, skip_seg: bool = False):
  out.mkdir(parents=True, exist_ok=True)
  meta = {
    'version': 1,
    'frame': {'w': FRAME_W, 'h': FRAME_H, 'f': MEDMODEL_FL, 'cx': FRAME_W / 2, 'cy': MEDMODEL_CY},
    'input': {'name': 'frame', 'shape': list(PACKED_SHAPE), 'type': 'float16'},
  }
  if not skip_seg:
    seg, meta['seg'] = seg_module(seg_model)
    export_onnx(seg, out / 'diorama_seg.onnx', ['qmap', 'label', 'score'])
    meta['seg']['file'] = 'diorama_seg.onnx'
  if dd3d_weights:
    boxes, meta['dd3d'] = dd3d_module(dd3d_weights)
    export_onnx(boxes, out / 'diorama_dd3d.onnx', ['boxes'])
    meta['dd3d']['file'] = 'diorama_dd3d.onnx'
  elif (out / 'diorama.json').exists():
    meta.setdefault('dd3d', json.loads((out / 'diorama.json').read_text()).get('dd3d'))
  (out / 'diorama.json').write_text(json.dumps(meta, indent=1) + '\n')
  return meta

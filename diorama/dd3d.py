"""Standalone inference port of DD3D (monocular 3D detection) for current PyTorch, without detectron2/pytorch3d.

Port of "Is Pseudo-Lidar needed for Monocular 3D Object detection?" (Park et al., ICCV 2021),
https://github.com/TRI-ML/dd3d -- MIT License, Copyright 2021 Toyota Research Institute.
VoVNet backbone after Youngwan Lee (ETRI); FCOS heads after AdelaiDet.

Only inference is ported: VoVNet-99-eSE + FPN (P2-P6) + FCOS2D head + FCOS3D head, decoding and 2D NMS.
Module names mirror the released checkpoints so they load with strict=True.

Camera frame of the outputs: x right, y down, z forward (OpenCV). Released weights are KITTI-trained,
classes Car, Pedestrian, Cyclist, Van, Truck.
"""
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn

CLASSES = ('Car', 'Pedestrian', 'Cyclist', 'Van', 'Truck')
NUM_CLASSES = len(CLASSES)
# (width, length, height) priors per class; predicted sizes are (tanh + 1) * prior
CANONICAL_WLH = [[1.61876949, 3.89154523, 1.52969237], [0.62806586, 0.82038497, 1.76784787],
                 [0.56898187, 1.77149234, 1.7237099], [1.9134491, 5.15499603, 2.18998422],
                 [2.61168401, 9.22692319, 3.36492722]]
MEAN_DEPTH = [32.594, 15.178, 8.424, 5.004, 4.662]
STD_DEPTH = [14.682, 7.139, 4.345, 2.399, 2.587]
STRIDES = [4, 8, 16, 32, 64]
PIXEL_MEAN = [103.530, 116.280, 123.675]  # BGR
PIXEL_STD = [57.375, 57.120, 58.395]
DEPTH_FOCAL_FACTOR = 500.0
MIN_DEPTH, MAX_DEPTH = 0.1, 80.0
V99 = {'stem': [64, 64, 128], 'stage_ch': [128, 160, 192, 224], 'out_ch': [256, 512, 768, 1024], 'layers': 5, 'blocks': [1, 3, 9, 3]}


class FrozenBN(nn.Module):
  def __init__(self, n, eps=1e-5):
    super().__init__()
    self.eps = eps
    for name, v in (('weight', 1.0), ('bias', 0.0), ('running_mean', 0.0), ('running_var', 1.0)):
      self.register_buffer(name, torch.full((n,), v))

  def forward(self, x):
    scale = self.weight * (self.running_var + self.eps).rsqrt()
    return x * scale[None, :, None, None] + (self.bias - self.running_mean * scale)[None, :, None, None]


class NormConv(nn.Conv2d):
  """Conv (no bias) followed by a norm stored as `.norm` (detectron2 Conv2d layout). With per_level norms,
  `.norm` is a ModuleList indexed by the FPN level passed to forward."""
  def __init__(self, cin, cout, k, norm, relu=False, stride=1):
    super().__init__(cin, cout, k, stride=stride, padding=k // 2, bias=False)
    self.norm = norm
    self.relu = relu

  def forward(self, x, level=None):
    x = self._conv_forward(x, self.weight, None)
    x = self.norm[level](x) if isinstance(self.norm, nn.ModuleList) else self.norm(x)
    return F.relu(x) if self.relu else x


# ---------------- VoVNet-99-eSE ----------------
def _cbr(cin, cout, name, postfix, stride=1, k=3):
  return [(f'{name}_{postfix}/conv', nn.Conv2d(cin, cout, k, stride, k // 2, bias=False)),
          (f'{name}_{postfix}/norm', FrozenBN(cout)), (f'{name}_{postfix}/relu', nn.ReLU(inplace=True))]


class ESE(nn.Module):
  def __init__(self, ch):
    super().__init__()
    self.fc = nn.Conv2d(ch, ch, 1)

  def forward(self, x):
    return x * F.hardsigmoid(self.fc(F.adaptive_avg_pool2d(x, 1)))  # relu6(x + 3) / 6


class OSA(nn.Module):
  def __init__(self, cin, stage_ch, out_ch, n_layers, name, identity):
    super().__init__()
    self.identity = identity
    self.layers = nn.ModuleList()
    c = cin
    for i in range(n_layers):
      self.layers.append(nn.Sequential(OrderedDict(_cbr(c, stage_ch, name, i))))
      c = stage_ch
    self.concat = nn.Sequential(OrderedDict(_cbr(cin + n_layers * stage_ch, out_ch, name, 'concat', k=1)))
    self.ese = ESE(out_ch)

  def forward(self, x):
    outs = [x]
    y = x
    for layer in self.layers:
      y = layer(y)
      outs.append(y)
    y = self.ese(self.concat(torch.cat(outs, 1)))
    return y + x if self.identity else y


class VoVNet(nn.Module):
  def __init__(self, spec=V99):
    super().__init__()
    s = spec['stem']
    self.stem = nn.Sequential(OrderedDict(_cbr(3, s[0], 'stem', '1', 2) + _cbr(s[0], s[1], 'stem', '2', 1) +
                                          _cbr(s[1], s[2], 'stem', '3', 2)))
    cins = [s[2]] + spec['out_ch'][:-1]
    for i in range(4):
      n = i + 2
      stage = nn.Sequential()
      if n != 2:
        stage.add_module('Pooling', nn.MaxPool2d(3, 2, ceil_mode=True))
      for b in range(spec['blocks'][i]):
        cin = cins[i] if b == 0 else spec['out_ch'][i]
        stage.add_module(f'OSA{n}_{b + 1}', OSA(cin, spec['stage_ch'][i], spec['out_ch'][i], spec['layers'],
                                                f'OSA{n}_{b + 1}', identity=b > 0))
      self.add_module(f'stage{n}', stage)

  def forward(self, x):
    x = self.stem(x)
    feats = []
    for n in range(2, 6):
      x = getattr(self, f'stage{n}')(x)
      feats.append(x)
    return feats


class TopBlock(nn.Module):
  def __init__(self, ch):
    super().__init__()
    self.p6 = nn.Conv2d(ch, ch, 3, 2, 1)


class Backbone(nn.Module):
  """VoVNet + FPN (P2-P5 top-down, P6 from P5)."""
  def __init__(self, ch=256):
    super().__init__()
    self.bottom_up = VoVNet()
    for i, cin in zip(range(2, 6), V99['out_ch'], strict=True):
      self.add_module(f'fpn_lateral{i}', NormConv(cin, ch, 1, FrozenBN(ch)))
      self.add_module(f'fpn_output{i}', NormConv(ch, ch, 3, FrozenBN(ch)))
    self.top_block = TopBlock(ch)

  def forward(self, x):
    c = self.bottom_up(x)
    prev = self.fpn_lateral5(c[3])
    outs = [self.fpn_output5(prev)]
    for i in (4, 3, 2):
      prev = getattr(self, f'fpn_lateral{i}')(c[i - 2]) + F.interpolate(prev, scale_factor=2.0, mode='nearest')
      outs.insert(0, getattr(self, f'fpn_output{i}')(prev))
    outs.append(self.top_block.p6(outs[-1]))
    return outs  # P2..P6, strides 4..64


# ---------------- heads ----------------
class Scale(nn.Module):
  def __init__(self, v=1.0):
    super().__init__()
    self.scale = nn.Parameter(torch.tensor([float(v)]))


class Offset(nn.Module):
  def __init__(self, v=0.0):
    super().__init__()
    self.bias = nn.Parameter(torch.tensor([float(v)]))


def _tower(ch, n, levels, frozen):
  norm = (lambda: FrozenBN(ch)) if frozen else (lambda: nn.BatchNorm2d(ch))
  return nn.ModuleList([NormConv(ch, ch, 3, nn.ModuleList([norm() for _ in range(levels)]), relu=True) for _ in range(n)])


def _run(tower, x, level):
  for conv in tower:
    x = conv(x, level)
  return x


class FCOS2DHead(nn.Module):
  def __init__(self, nc, ch=256, levels=5):
    super().__init__()
    self.cls_tower = _tower(ch, 4, levels, frozen=False)
    self.box2d_tower = _tower(ch, 4, levels, frozen=False)
    self.cls_logits = nn.Conv2d(ch, nc, 3, 1, 1)
    self.box2d_reg = nn.Conv2d(ch, 4, 3, 1, 1)
    self.centerness = nn.Conv2d(ch, 1, 3, 1, 1)
    self.scales_box2d_reg = nn.ModuleList([Scale() for _ in range(levels)])

  def forward(self, feats):
    out = []
    for lvl, f in enumerate(feats):
      c, b = _run(self.cls_tower, f, lvl), _run(self.box2d_tower, f, lvl)
      out.append((self.cls_logits(c), F.relu(self.box2d_reg(b) * self.scales_box2d_reg[lvl].scale), self.centerness(b)))
    return out


class FCOS3DHead(nn.Module):
  def __init__(self, nc, ch=256, levels=5):
    super().__init__()
    self.box3d_tower = _tower(ch, 4, levels, frozen=True)
    conv = lambda k, bias=True: nn.ModuleList([nn.Conv2d(ch, k * nc, 3, 1, 1, bias=bias)])  # noqa: E731
    self.box3d_quat, self.box3d_ctr, self.box3d_depth = conv(4), conv(2), conv(1, bias=False)
    self.box3d_size, self.box3d_conf = conv(3), conv(1)
    self.register_buffer('mean_depth_per_level', torch.tensor(MEAN_DEPTH))
    self.register_buffer('std_depth_per_level', torch.tensor(STD_DEPTH))
    self.scales_proj_ctr = nn.ModuleList([Scale() for _ in range(levels)])
    self.scales_size = nn.ModuleList([Scale() for _ in range(levels)])
    self.scales_conf = nn.ModuleList([Scale() for _ in range(levels)])
    self.scales_depth = nn.ModuleList([Scale() for _ in range(levels)])
    self.offsets_depth = nn.ModuleList([Offset() for _ in range(levels)])

  def forward(self, feats):
    out = []
    for lvl, f in enumerate(feats):
      t = _run(self.box3d_tower, f, lvl)
      out.append((self.box3d_quat[0](t), self.box3d_ctr[0](t) * self.scales_proj_ctr[lvl].scale,
                  self.box3d_depth[0](t) * self.scales_depth[lvl].scale + self.offsets_depth[lvl].bias,
                  self.box3d_size[0](t) * self.scales_size[lvl].scale, self.box3d_conf[0](t) * self.scales_conf[lvl].scale))
    return out


def quat_to_matrix(q):
  """(N,4) unit quaternions, w x y z -> (N,3,3)."""
  w, x, y, z = q.unbind(-1)
  return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                      2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                      2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)


class DD3D(nn.Module):
  def __init__(self, num_classes=NUM_CLASSES):
    super().__init__()
    self.nc = num_classes
    self.backbone = Backbone()
    self.fcos2d_head = FCOS2DHead(num_classes)
    self.fcos3d_head = FCOS3DHead(num_classes)
    self.register_buffer('pixel_mean', torch.tensor(PIXEL_MEAN).view(3, 1, 1))
    self.register_buffer('pixel_std', torch.tensor(PIXEL_STD).view(3, 1, 1))
    self.register_buffer('canon', torch.tensor(CANONICAL_WLH))

  @classmethod
  def from_checkpoint(cls, path, device='cuda'):
    m = cls()
    sd = torch.load(path, map_location='cpu', weights_only=False)
    sd = sd.get('model', sd)
    sd = {k: v for k, v in sd.items() if not k.endswith('num_batches_tracked')}
    sd['canon'] = m.canon
    missing, unexpected = m.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.endswith('num_batches_tracked')]
    assert not missing and not unexpected, (missing, unexpected)
    return m.to(device).eval()

  def heads(self, x):
    """Model part (fixed shapes for a fixed input size: can be replaced by a CUDA-graph replay)."""
    with torch.autocast('cuda', dtype=torch.float16, enabled=x.device.type == 'cuda'):
      feats = self.backbone(x)
      return self.fcos2d_head(feats), self.fcos3d_head(feats)

  @torch.no_grad()
  def forward(self, bgr, K, score_thresh=0.2, pre_nms_thresh=0.05, pre_nms_topk=1000, nms_thresh=0.75, max_det=100):
    """bgr: (H, W, 3) uint8 image (already resized); K: (3,3) intrinsics of that image.

    Returns dict of numpy arrays: box2d (N,4) xyxy, score (2D), score3d, cls, center (N,3) camera frame,
    size (N,3) (W, L, H) metres, R (N,3,3) object->camera rotation (column 0 = forward, 1 = left/right, 2 = up/down).
    """
    dev = self.pixel_mean.device
    H, W = bgr.shape[:2]
    x = torch.from_numpy(np.ascontiguousarray(bgr)).to(dev).permute(2, 0, 1).float()
    x = (x - self.pixel_mean) / self.pixel_std
    ph, pw = (-H) % 64, (-W) % 64  # FPN with P6 needs multiples of 64; padding is zero after normalisation
    x = F.pad(x, (0, pw, 0, ph))[None]
    out2d, out3d = self.heads(x)

    Kt = torch.as_tensor(K, dtype=torch.float32, device=dev)
    Kinv = torch.linalg.inv(Kt)
    pixel_size = torch.linalg.norm(torch.stack([Kinv[0, 0], Kinv[1, 1]]))
    rows = []
    for lvl, ((logits, reg, ctr), (quat, pctr, depth, size, conf)) in enumerate(zip(out2d, out3d, strict=True)):
      s = STRIDES[lvl]
      h, w = logits.shape[-2:]
      ys, xs = torch.meshgrid(torch.arange(h, device=dev) * s, torch.arange(w, device=dev) * s, indexing='ij')
      loc = torch.stack([xs.reshape(-1), ys.reshape(-1)], 1).float()
      flat = lambda t, c: t[0].float().permute(1, 2, 0).reshape(-1, c)  # noqa: E731
      scores = flat(logits, self.nc).sigmoid() * flat(ctr, 1).sigmoid()  # thresholded with centerness
      cand = (scores > pre_nms_thresh).nonzero()
      if len(cand) > pre_nms_topk:
        cand = cand[scores[cand[:, 0], cand[:, 1]].topk(pre_nms_topk).indices]
      if not len(cand):
        continue
      p, c = cand[:, 0], cand[:, 1]
      sc = scores[p, c].sqrt()
      r = flat(reg, 4)[p]
      l_ = loc[p]
      box = torch.stack([l_[:, 0] - r[:, 0], l_[:, 1] - r[:, 1], l_[:, 0] + r[:, 2], l_[:, 1] + r[:, 3]], 1)
      # class-specific 3D predictions
      q = flat(quat, 4 * self.nc)[p].view(-1, 4, self.nc)[torch.arange(len(p)), :, c]
      pc = flat(pctr, 2 * self.nc)[p].view(-1, 2, self.nc)[torch.arange(len(p)), :, c]
      dp = flat(depth, self.nc)[p, c]
      sz = flat(size, 3 * self.nc)[p].view(-1, 3, self.nc)[torch.arange(len(p)), :, c]
      cf = flat(conf, self.nc)[p, c].sigmoid()
      rows.append((box, sc, c, q, pc + l_, dp, sz, cf))
    if not rows:
      return {k: np.zeros((0,) + sh) for k, sh in (('box2d', (4,)), ('score', ()), ('score3d', ()), ('cls', ()),
                                                   ('center', (3,)), ('size', (3,)), ('R', (3, 3)))}
    box, sc, c, q, pc, dp, sz, cf = (torch.cat(t) for t in zip(*rows, strict=True))
    score3d = sc * cf
    keep = torchvision.ops.batched_nms(box, score3d, c, nms_thresh)[:max_det]
    keep = keep[sc[keep] > score_thresh]
    box, sc, c, q, pc, dp, sz, score3d = (t[keep] for t in (box, sc, c, q, pc, dp, sz, score3d))

    # decode: depth (focal-length aware), egocentric rotation from the allocentric quaternion, size from priors
    dp = (dp / (pixel_size * DEPTH_FOCAL_FACTOR)).clamp(MIN_DEPTH, MAX_DEPTH)
    q = q / q.norm(dim=1, keepdim=True).clamp(min=1e-7)
    ray = (Kinv @ torch.cat([pc, torch.ones_like(pc[:, :1])], 1).T).T
    zax = ray / ray.norm(dim=1, keepdim=True)
    yax = torch.tensor([0.0, 1.0, 0.0], device=dev) - zax[:, 1:2] * zax
    yax = yax / yax.norm(dim=1, keepdim=True)
    xax = torch.cross(yax, zax, dim=1)
    R = torch.stack([xax, yax, zax], -1) @ quat_to_matrix(q)
    center = ray * dp[:, None]
    wlh = (sz.tanh() + 1.0) * self.canon[c]
    to = lambda t: t.float().cpu().numpy()  # noqa: E731
    return {'box2d': to(box), 'score': to(sc), 'score3d': to(score3d), 'cls': to(c).astype(int), 'center': to(center),
            'size': to(wlh), 'R': to(R)}

"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Mask2Former R50, Mapillary Vistas semantic (the Mask2Former model zoo's
49189528_1, a Detectron2 checkpoint), as transformers' Mask2Former with a
ResNet-50 backbone: what ernestchu/jetlink-diorama keeps under
r50-semantic/mask2former, which `python -m diorama export` takes. Every
weight of the checkpoint goes somewhere, and every one of the model's is
filled. The decoders' names follow transformers' own converter for the
original checkpoints; the backbone is Detectron2's ResNet (its stride in
the 3x3 convolution, torchvision's) as transformers' ResNetBackbone.

    curl -LO https://dl.fbaipublicfiles.com/maskformer/mask2former/mapillary_vistas/semantic/maskformer_R50_bs16_300k/model_final_6c66d0.pkl
    python -m diorama.convert_r50 model_final_6c66d0.pkl --out mask2former

Needs torch and transformers.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

# the same decoders, and the class names, as the Swin-L semantic model transformers publishes
BASE = 'facebook/mask2former-swin-large-mapillary-vistas-semantic'


def convert(pkl: Path, out: Path) -> int:
  import torch
  from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation, ResNetConfig

  with open(pkl, 'rb') as fh:
    src = {k: torch.from_numpy(np.asarray(v)) for k, v in pickle.load(fh, encoding='latin1')['model'].items()}
  base = Mask2FormerConfig.from_pretrained(BASE).to_dict()
  backbone = ResNetConfig(depths=[3, 4, 6, 3], hidden_sizes=[256, 512, 1024, 2048], embedding_size=64, layer_type='bottleneck',
                          downsample_in_bottleneck=False, out_features=['stage1', 'stage2', 'stage3', 'stage4'])
  keep = {k: v for k, v in base.items() if k not in ('backbone_config', 'architectures', '_commit_hash', 'transformers_version', 'model_type')}
  config = Mask2FormerConfig(backbone_config=backbone, **keep)
  model = Mask2FormerForUniversalSegmentation(config).eval()
  dst = model.state_dict()
  mapped = {}

  def take(s, d, t=None):
    t = src.pop(s) if t is None else t
    assert d in dst and t.shape == dst[d].shape, (s, d)
    mapped[d] = t

  def conv_bn(s, d):
    take(f'{s}.weight', f'{d}.convolution.weight')
    for p in ('weight', 'bias', 'running_mean', 'running_var'):
      take(f'{s}.norm.{p}', f'{d}.normalization.{p}')

  E = 'model.pixel_level_module.encoder'
  conv_bn('backbone.stem.conv1', f'{E}.embedder.embedder')
  for stage, blocks in enumerate([3, 4, 6, 3]):
    for b in range(blocks):
      s, d = f'backbone.res{stage + 2}.{b}', f'{E}.encoder.stages.{stage}.layers.{b}'
      if b == 0:
        conv_bn(f'{s}.shortcut', f'{d}.shortcut')
      for i in range(3):
        conv_bn(f'{s}.conv{i + 1}', f'{d}.layer.{i}')

  P, D = 'sem_seg_head.pixel_decoder', 'model.pixel_level_module.decoder'
  for name in ('adapter_1', 'layer_1'):
    take(f'{P}.{name}.weight', f'{D}.{name}.0.weight')
    take(f'{P}.{name}.norm.weight', f'{D}.{name}.1.weight')
    take(f'{P}.{name}.norm.bias', f'{D}.{name}.1.bias')
  for i in range(3):
    for j in range(2):
      for p in ('weight', 'bias'):
        take(f'{P}.input_proj.{i}.{j}.{p}', f'{D}.input_projections.{i}.{j}.{p}')
  take(f'{P}.transformer.level_embed', f'{D}.level_embed')
  for layer in range(config.encoder_layers):
    s, d = f'{P}.transformer.encoder.layers.{layer}', f'{D}.encoder.layers.{layer}'
    for a, b in [('linear1', 'fc1'), ('linear2', 'fc2'), ('norm1', 'self_attn_layer_norm'), ('norm2', 'final_layer_norm'),
                 ('self_attn.attention_weights', 'self_attn.attention_weights'), ('self_attn.output_proj', 'self_attn.output_proj'),
                 ('self_attn.sampling_offsets', 'self_attn.sampling_offsets'), ('self_attn.value_proj', 'self_attn.value_proj')]:
      for p in ('weight', 'bias'):
        take(f'{s}.{a}.{p}', f'{d}.{b}.{p}')
  for p in ('weight', 'bias'):
    take(f'{P}.mask_features.{p}', f'{D}.mask_projection.{p}')

  Q, T = 'sem_seg_head.predictor', 'model.transformer_module'
  hidden = config.hidden_dim
  for i in range(config.decoder_layers - 1):
    d = f'{T}.decoder.layers.{i}'
    for a, b in [(f'transformer_self_attention_layers.{i}.self_attn.out_proj', 'self_attn.out_proj'),
                 (f'transformer_self_attention_layers.{i}.norm', 'self_attn_layer_norm'),
                 (f'transformer_cross_attention_layers.{i}.multihead_attn.out_proj', 'cross_attn.out_proj'),
                 (f'transformer_cross_attention_layers.{i}.norm', 'cross_attn_layer_norm'),
                 (f'transformer_ffn_layers.{i}.linear1', 'fc1'), (f'transformer_ffn_layers.{i}.linear2', 'fc2'),
                 (f'transformer_ffn_layers.{i}.norm', 'final_layer_norm')]:
      for p in ('weight', 'bias'):
        take(f'{Q}.{a}.{p}', f'{d}.{b}.{p}')
    take(f'{Q}.transformer_cross_attention_layers.{i}.multihead_attn.in_proj_weight', f'{d}.cross_attn.in_proj_weight')
    take(f'{Q}.transformer_cross_attention_layers.{i}.multihead_attn.in_proj_bias', f'{d}.cross_attn.in_proj_bias')
    # self-attention's packed q, k, v projection, split
    w = src.pop(f'{Q}.transformer_self_attention_layers.{i}.self_attn.in_proj_weight')
    bias = src.pop(f'{Q}.transformer_self_attention_layers.{i}.self_attn.in_proj_bias')
    for n, k in enumerate('qkv'):
      take('', f'{d}.self_attn.{k}_proj.weight', w[n * hidden:(n + 1) * hidden])
      take('', f'{d}.self_attn.{k}_proj.bias', bias[n * hidden:(n + 1) * hidden])
  for p in ('weight', 'bias'):
    take(f'{Q}.decoder_norm.{p}', f'{T}.decoder.layernorm.{p}')
    for i in range(3):
      take(f'{Q}.mask_embed.layers.{i}.{p}', f'{T}.decoder.mask_predictor.mask_embedder.{i}.0.{p}')
    take(f'{Q}.class_embed.{p}', f'class_predictor.{p}')
  take(f'{Q}.query_embed.weight', f'{T}.queries_embedder.weight')
  take(f'{Q}.static_query.weight', f'{T}.queries_features.weight')  # query_feat in later checkpoints
  take(f'{Q}.level_embed.weight', f'{T}.level_embed.weight')
  take('criterion.empty_weight', 'criterion.empty_weight')  # the loss's no-object weight

  assert not src, f'unused: {sorted(src)[:20]}'
  # BatchNorm's counters are bookkeeping, not weights
  left = [k for k in dst if k not in mapped and not k.endswith('num_batches_tracked')]
  assert not left, f'unfilled: {left[:20]}'
  model.load_state_dict(mapped | {k: dst[k] for k in dst if k not in mapped}, strict=True)
  model.save_pretrained(out)
  return len(mapped)


if __name__ == '__main__':
  ap = argparse.ArgumentParser(prog='python -m diorama.convert_r50', description=__doc__.split('\n\n')[1])
  ap.add_argument('pkl', type=Path, help="the checkpoint, model_final_6c66d0.pkl")
  ap.add_argument('--out', type=Path, required=True)
  args = ap.parse_args()
  print(f'converted {convert(args.pkl, args.out)} tensors into {args.out}')

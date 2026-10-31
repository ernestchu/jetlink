"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

A driving model run over a recorded drive the way a jetlink server runs it
for the comma: jetlink's ModelSpec for the layout, PolicyQueues for a queued
graph's history, the state loop for a stateful one, and the packed scalars a
parked US car would send (no desire, right-hand traffic).
"""
from __future__ import annotations

import numpy as np

from jetlink.queues import PolicyQueues
from jetlink.spec import DRIVING_OUTPUT, spec_from_onnx

# openpilot's lat_action_t and long_action_t with its default delays.
ACTION_T = (0.275, 0.525)


def providers(device: str) -> list[str]:
  import onnxruntime as ort
  have = ort.get_available_providers()
  want = {'cuda': ['CUDAExecutionProvider'], 'coreml': ['CoreMLExecutionProvider'], 'cpu': []}.get(device)
  if want is None:
    want = [p for p in ('CUDAExecutionProvider', 'CoreMLExecutionProvider') if p in have]
  missing = [p for p in want if p not in have]
  if missing:
    raise SystemExit(f"onnxruntime here has no {', '.join(missing)} (it has {', '.join(have)})")
  if 'CUDAExecutionProvider' in want and hasattr(ort, 'preload_dlls'):
    ort.preload_dlls()  # CUDA and cuDNN from their pip wheels (torch's), which onnxruntime does not look for by itself
  return [*want, 'CPUExecutionProvider']


class DrivingModel:
  def __init__(self, onnx_path: str, device: str = 'auto', rhd: bool = False):
    import onnxruntime as ort
    self.spec = spec_from_onnx(onnx_path)
    self.sess = ort.InferenceSession(onnx_path, providers=providers(device))
    self.slices: dict[str, slice] = self.spec.output_slices
    self.dtypes = {i.name: np.float16 if 'float16' in i.type else np.uint8 if 'uint8' in i.type else np.float32 for i in self.sess.get_inputs()}
    self.shapes = {i.name: tuple(i.shape) for i in self.sess.get_inputs()}
    self.names = [o.name for o in self.sess.get_outputs()]
    packed = np.zeros(self.spec.packed_nelem, np.float32)
    for name, (at, _shape) in self.spec.packed_layout.items():
      if name == 'traffic_convention':
        packed[at] = [0.0, 1.0] if rhd else [1.0, 0.0]
      elif name == 'action_t':
        packed[at] = ACTION_T[:packed[at].size]
    self.packed = packed
    self.reset()

  def reset(self):
    if self.spec.stateful:
      self.state = {name: np.zeros(self.shapes[name], self.dtypes[name]) for name in self.spec.state_pairs}
    else:
      self.queues = PolicyQueues(self.spec)
      self.queues.reset()

  def step(self, warped: np.ndarray) -> np.ndarray:
    """One 20 Hz frame: (2, 6, 128, 256) uint8 in, the flat output vector out."""
    if self.spec.stateful:
      feed = {'new_img': warped, **self.state, **{name: self.packed[at] for name, (at, _) in self.spec.packed_layout.items()}}
      feed = {k: np.ascontiguousarray(v, self.dtypes[k]).reshape(self.shapes[k]) for k, v in feed.items()}
      outs = dict(zip(self.names, self.sess.run(None, feed), strict=True))
      self.state = {name: outs[nxt] for name, nxt in self.spec.state_pairs.items()}
      return np.asarray(outs[DRIVING_OUTPUT], np.float32).reshape(-1)
    feed = self.queues.step(warped, self.packed)
    feed = {k: np.ascontiguousarray(v, self.dtypes[k]) for k, v in feed.items()}
    out = np.asarray(self.sess.run(None, feed)[0], np.float32).reshape(-1)
    self.queues.after_run({DRIVING_OUTPUT: out})
    return out

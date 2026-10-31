"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The slice of openpilot's parse_model_outputs Diorama reads, over the flat
`outputs` vector and the ONNX's output_slices, without openpilot. diorama.js
parses the same heads the same way.
"""
from __future__ import annotations

import numpy as np

IDX_N = 33
X_IDXS = 192.0 * (np.arange(IDX_N) / 32) ** 2
PLAN_WIDTH, POSE_WIDTH = 15, 6
LEAD_MHP_N, LEAD_MHP_SELECTION, LEAD_TRAJ_LEN, LEAD_WIDTH = 2, 3, 6, 4

# The heads Diorama draws from, in the order a drive packet carries them, and
# the ones the offline calibration reads besides.
DRIVE_FIELDS = ('plan', 'lane_lines', 'lane_lines_prob', 'road_edges', 'lead', 'lead_prob', 'pose')
CALIB_FIELDS = ('pose', 'road_transform', 'wide_from_device_euler')


def _exp(x):
  return np.exp(np.clip(x, -np.inf, 11))


def _sigmoid(x):
  return 1.0 / (1.0 + _exp(-x))


def _softmax(x, axis=-1):
  e = _exp(x - x.max(axis=axis, keepdims=True))
  return e / e.sum(axis=axis, keepdims=True)


def _mdn(raw, shape):
  n = raw.size // 2
  return raw[:n].reshape(shape), _exp(raw[n:2 * n]).reshape(shape)


def lead_is_mhp(n: int) -> bool:
  return n != 2 * LEAD_MHP_SELECTION * LEAD_TRAJ_LEN * LEAD_WIDTH


def parse(vector: np.ndarray, slices: dict[str, slice], names=DRIVE_FIELDS) -> dict[str, np.ndarray]:
  """Means and stds of the named heads, shaped as openpilot's Parser leaves them less the batch axis."""
  raw = {k: np.asarray(vector[slices[k]], np.float32) for k in names if k in slices}
  out = {}
  for name, shape in (('plan', (IDX_N, PLAN_WIDTH)), ('pose', (POSE_WIDTH,)), ('road_transform', (POSE_WIDTH,)),
                      ('wide_from_device_euler', (3,)), ('lane_lines', (4, IDX_N, 2)), ('road_edges', (2, IDX_N, 2))):
    if name in raw:
      out[name], out[name + '_stds'] = _mdn(raw[name], shape)
  for name in ('lane_lines_prob', 'lead_prob'):
    if name in raw:
      out[name] = _sigmoid(raw[name])
  if 'lead' in raw:
    r = raw['lead']
    if lead_is_mhp(r.size):
      r = r.reshape(LEAD_MHP_N, -1)
      n = (r.shape[1] - LEAD_MHP_SELECTION) // 2
      weights = _softmax(r[:, -LEAD_MHP_SELECTION:], axis=0)
      best = weights.argmax(0)  # per selection, the most likely hypothesis
      shape = (LEAD_TRAJ_LEN, LEAD_WIDTH)
      out['lead'] = np.stack([r[best[s], :n].reshape(shape) for s in range(LEAD_MHP_SELECTION)])
      out['lead_stds'] = np.stack([_exp(r[best[s], n:2 * n]).reshape(shape) for s in range(LEAD_MHP_SELECTION)])
    else:
      out['lead'], out['lead_stds'] = _mdn(r, (LEAD_MHP_SELECTION, LEAD_TRAJ_LEN, LEAD_WIDTH))
  return out

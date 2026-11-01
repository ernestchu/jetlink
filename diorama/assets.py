"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Diorama's car: a procedural electric crossover (an original design, no external
models), the one vehicle the page draws, as the lead and as the ego car, and
the binary file the page loads it from (`write_library`). After changing it:

    python -m diorama assets

The body is one smooth surface: every cross-section is a single rounded outline
from the underside, up the side to the shoulder and on over the glass to the
roof, lofted from the tail to the nose, with both ends rounded off. The glass,
lamps and trim are colours on it, so nothing sits proud of it.

Local frame: x forward, y right, z UP, origin on the ground under the car's centre, units in metres.
Each mesh is a dict of arrays: pos (N,3), nrm (N,3), col (N,3) in 0..1, emis (N,) and idx (M,3).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

RESOURCES = Path(__file__).resolve().parent.parent / 'JetlinkKit' / 'Sources' / 'JetlinkDiorama' / 'Resources'

BODY = np.array([0.62, 0.62, 0.64])  # the paint as stored; the page colours it (emission -1 marks it)
GLASS = np.array([0.33, 0.33, 0.35])
TRIM = np.array([0.30, 0.30, 0.32])
TIRE = np.array([0.22, 0.22, 0.23])
RIM = np.array([0.40, 0.40, 0.42])
TAIL = np.array([0.80, 0.10, 0.10])
HEAD = np.array([0.90, 0.91, 0.93])

L, W, H = 4.60, 1.88, 1.68  # length, width without the mirrors, height
FRONT, REAR = L / 2, -L / 2
AXLES, WHEEL_R, WHEEL_W = (1.38, -1.32), 0.37, 0.25
ARCH = WHEEL_R + 0.06
END_R = (0.16, 0.24)  # how far the tail and the nose round off over


def smoothstep(e0, e1, x):
  t = np.clip((np.asarray(x, float) - e0) / (e1 - e0), 0.0, 1.0)
  return t * t * (3 - 2 * t)


def merge(*meshes):
  pos, nrm, col, emis, idx, n = [], [], [], [], [], 0
  for m in meshes:
    pos.append(m['pos'])
    nrm.append(m['nrm'])
    col.append(m['col'])
    emis.append(m['emis'])
    idx.append(m['idx'] + n)
    n += len(m['pos'])
  return {'pos': np.concatenate(pos).astype(np.float32), 'nrm': np.concatenate(nrm).astype(np.float32),
          'col': np.concatenate(col).astype(np.float32), 'emis': np.concatenate(emis).astype(np.float32),
          'idx': np.concatenate(idx).astype(np.uint32)}


def grid_mesh(P, color_fn, caps=True):
  """P: (N stations, K ring points, 3) closed rings -> a smooth mesh, with flat caps closing the ends."""
  N, K, _ = P.shape
  n = np.cross(np.gradient(P, axis=0), np.roll(P, -1, axis=1) - np.roll(P, 1, axis=1))
  flip = np.sum(n * (P - P.mean(axis=1, keepdims=True)), axis=-1) < 0
  n[flip] *= -1
  n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
  pos, nrm = P.reshape(-1, 3), n.reshape(-1, 3)
  col, emis = color_fn(pos, nrm)
  i, k = np.meshgrid(np.arange(N - 1), np.arange(K), indexing='ij')
  a, b = i * K + k, i * K + (k + 1) % K
  idx = np.concatenate([np.stack([a, b, a + K], -1).reshape(-1, 3), np.stack([b, b + K, a + K], -1).reshape(-1, 3)])
  meshes = [{'pos': pos, 'nrm': nrm, 'col': col, 'emis': emis, 'idx': idx}]
  for end in ((0, N - 1) if caps else ()):
    ring = P[end]
    ax = (P[min(end + 1, N - 1)] - P[max(end - 1, 0)]).mean(0)
    ax = ax / max(np.linalg.norm(ax), 1e-9) * (-1 if end == 0 else 1)
    cp = np.vstack([ring.mean(0, keepdims=True), ring])
    cn = np.repeat(ax[None], K + 1, 0)
    cc, ce = color_fn(cp, cn)
    ring_i = np.arange(1, K + 1)
    meshes.append({'pos': cp, 'nrm': cn, 'col': cc, 'emis': ce, 'idx': np.stack([np.zeros(K, int), ring_i, np.roll(ring_i, -1)], 1)})
  return merge(*meshes)


def chaikin(pts, rounds=4):
  """An open polyline's corners cut `rounds` times: a smooth curve through the same ends."""
  for _ in range(rounds):
    a, b = pts[:-1], pts[1:]
    pts = np.vstack([pts[:1], np.stack([0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b], 1).reshape(-1, 2), pts[-1:]])
  return pts


# ---------------- the body's outline along its length ----------------
def lower_top(x):
  """The top of the lower body: the tailgate's lower edge, the shoulder along the cabin, a short hood falling to a blunt nose."""
  return np.interp(x, [REAR, REAR + 0.30, -0.50, 0.70, 1.00, 1.50, 1.95, FRONT], [1.06, 1.10, 1.06, 1.04, 1.12, 1.07, 1.00, 0.90])


def roof(x):
  """The top of the section: a steep tailgate from its lower edge up to a short spoiler, a long roof held almost level,
  then the windshield down to its base and the hood."""
  cabin = np.interp(x, [REAR, REAR + 0.06, REAR + 0.20, REAR + 0.34, REAR + 0.42, -1.20, -0.30, 0.15, 0.40, 0.70, 1.00],
                    [1.06, 1.30, 1.54, 1.64, H - 0.01, H, H - 0.005, 1.63, 1.52, 1.33, 1.12])
  return np.where(x < 1.00, np.maximum(cabin, lower_top(x)), lower_top(x))


def underside(x):
  zb = np.interp(x, [REAR, REAR + 0.50, FRONT - 0.50, FRONT], [0.34, 0.22, 0.21, 0.30])
  for ax in AXLES:  # the arches: the underside rises over each wheel
    zb = np.maximum(zb, WHEEL_R + np.sqrt(np.clip(ARCH ** 2 - (x - ax) ** 2, 0, None)) * (np.abs(x - ax) < ARCH))
  return zb


def half_width(x):
  return np.interp(x, [REAR, REAR + 0.35, REAR + 0.90, FRONT - 0.90, FRONT - 0.45, FRONT], [0.84, 0.92, 0.95, 0.95, 0.93, 0.80]) * W / 1.9


# rows of each half outline, from the middle of the underside to the middle of the roof: underside, side up to the
# shoulder, the glass up to the roof's edge, the roof. A colour's edge is then always one row, a clean line along the car.
ROWS = {'under': 8, 'side': 16, 'shoulder': 3, 'glass': 9, 'rim': 2, 'roof': 6}


def section(x):
  """The half outline at x as rows (see ROWS), each tagged with its part: the middle of the underside, up the side,
  over the shoulder and the glass, to the middle of the roof."""
  hw, zb, belt, top = half_width(x), underside(x), lower_top(x), roof(x)
  cabin = smoothstep(0.03, 0.30, top - belt)  # 0 where there is no cabin over the lower body (hood, tail)
  wtop = (1 - cabin) * hw * 0.86 + cabin * hw * 0.64  # the roof's (or the hood's) half width
  rise = max(top - belt, 0.0)
  keys = {  # the outline's control points, and which part each begins
    'under': [[0.0, zb], [hw * 0.55, zb]],
    'side': [[hw * 0.90, zb + 0.02], [hw, zb + 0.14], [hw * 1.005, belt - 0.30], [hw * 0.99, belt - 0.08]],
    'shoulder': [[hw * (0.955 - 0.015 * cabin), belt + 0.01]],
    'glass': [[hw * (0.94 - 0.01 * cabin), belt + 0.035 * cabin + 0.01], [wtop + (hw * 0.94 - wtop) * 0.35, belt + rise * 0.55]],
    'rim': [[wtop, top - 0.05 - 0.04 * cabin]],
    'roof': [[wtop * 0.75, top - 0.02], [wtop * 0.40, top - 0.004], [0.0, top]],
  }
  pts = np.array([q for part in ROWS for q in keys[part]])
  starts = np.cumsum([0] + [len(keys[part]) for part in ROWS])[:-1]
  curve = chaikin(pts, 4)
  s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))])
  # where each part begins on the smooth curve: the curve's nearest point to its first control point
  at = [s[np.argmin(np.linalg.norm(curve - pts[i], axis=1))] for i in starts] + [s[-1]]
  u = np.concatenate([np.linspace(at[j], at[j + 1], n, endpoint=False) for j, n in enumerate(ROWS.values())] + [[s[-1]]])
  part = np.concatenate([[j] * n for j, n in enumerate(ROWS.values())] + [[len(ROWS) - 1]])
  return np.interp(u, s, curve[:, 0]), np.interp(u, s, curve[:, 1]), part


def body_mesh(N=160):
  # stations closer together toward both ends, where the shape turns
  t = np.linspace(0, 1, N)
  xs = REAR + L * (0.5 * t + 0.5 * (0.5 - 0.5 * np.cos(np.pi * t)))
  half = sum(ROWS.values()) + 1
  K = 2 * half - 2
  P, part = np.zeros((N, K, 3)), np.zeros((N, K), int)
  for i, x in enumerate(xs):
    y, z, pr = section(x)
    y, z, pr = np.concatenate([y, -y[-2:0:-1]]), np.concatenate([z, z[-2:0:-1]]), np.concatenate([pr, pr[-2:0:-1]])
    # the ends round off like a rounded box: full almost to the end, then the section closes in on the lamps' height,
    # so the lamps run straight across
    r = END_R[0] if x < 0 else END_R[1]
    d = np.clip(max((REAR + r) - x, x - (FRONT - r)), 0, None)
    s = np.clip(1 - (d / r) ** 4, 0.0, 1) ** 0.25
    zc = float(lower_top(x)) - 0.04
    P[i, :, 0] = x
    P[i, :, 1] = y * (0.004 + 0.996 * s)
    P[i, :, 2] = zc + (z - zc) * (0.006 + 0.994 * s)
    part[i] = pr
  names = list(ROWS)
  m = grid_mesh(P, lambda p, n: (np.zeros((len(p), 3)), np.zeros(len(p))))
  n_grid = N * K
  pos, nrm = m['pos'], m['nrm']
  x, z = pos[:, 0], pos[:, 2]
  row = np.concatenate([part.reshape(-1), np.full(len(pos) - n_grid, names.index('side'))])  # the caps: paint
  k = np.arange(K)
  ring = np.concatenate([np.tile(np.minimum(k, K - k), N), np.zeros(len(pos) - n_grid, int)])  # the row, counted from the underside
  station = np.concatenate([np.repeat(xs, K), np.full(len(pos) - n_grid, xs[0])])
  cabin = (roof(station) - lower_top(station)) > 0.15
  col = np.repeat(BODY[None], len(pos), 0)
  emis = np.full(len(pos), -1.0)  # the paint, for the page to colour
  glass = cabin & ((row == names.index('glass')) | (row == names.index('roof')))
  glass &= (station > REAR + 0.12) & (station < 0.98)
  col[glass], emis[glass] = GLASS, 0.0
  # the lamps: the shoulder's rows and the side's top two, where they face back (a band across the tail) or forward
  # (the headlamps, at the nose's corners)
  lamp = (row == names.index('shoulder')) | ((row == names.index('side')) & (ring >= ROWS['under'] + ROWS['side'] - 1))
  tail = lamp & (station < REAR + END_R[0] + 0.25) & (nrm[:, 0] < -0.2) & (np.abs(pos[:, 1]) > 0.30)  # at the corners
  # and the high stop lamp, in the middle of the spoiler's edge
  tail |= (station < REAR + 0.40) & (nrm[:, 0] < -0.3) & (np.abs(pos[:, 1]) < 0.20) & (np.abs(z - (H - 0.12)) < 0.014)
  head = lamp & (station > FRONT - END_R[1] - 0.15) & (nrm[:, 0] > 0.2) & (np.abs(pos[:, 1]) > 0.30)
  col[tail], emis[tail] = TAIL, 0.7
  col[head], emis[head] = HEAD, 0.5
  # the black underside, the arch liners and a slim rocker
  dark = nrm[:, 2] < -0.40
  for ax in AXLES:
    dark |= (np.abs(np.hypot(x - ax, z - WHEEL_R) - ARCH) < 0.035) & (np.abs(nrm[:, 1]) > 0.3)
  col[dark], emis[dark] = TRIM, 0.0
  m['col'] = col * (0.82 + 0.18 * smoothstep(0.2, 0.9, z))[:, None]
  m['emis'] = emis
  return m


def wheel(K=36):
  """A tyre with rounded shoulders and a dark rim face on both sides, its axis along y."""
  th = np.linspace(0, 2 * np.pi, K, endpoint=False)
  r, w = WHEEL_R, WHEEL_W
  prof = [(-w / 2, 0.0), (-w / 2, 0.66), (-w / 2 + 0.012, 0.72), (-w / 2 + 0.004, 0.80), (-w / 2 + 0.02, 0.95), (-w / 2 + 0.06, 1.0),
          (w / 2 - 0.06, 1.0), (w / 2 - 0.02, 0.95), (w / 2 - 0.004, 0.80), (w / 2 - 0.012, 0.72), (w / 2, 0.66), (w / 2, 0.0)]
  P = np.array([np.stack([k * r * np.cos(th), np.full(K, y), k * r * np.sin(th)], 1) for y, k in prof])

  def color_fn(pos, nrm):
    rim = (np.abs(nrm[:, 1]) > 0.6) & (np.hypot(pos[:, 0], pos[:, 2]) < 0.75 * r)
    return np.where(rim[:, None], RIM, TIRE), np.zeros(len(pos))

  return grid_mesh(P, color_fn, caps=False)


def mirror(side):
  """A small smooth mirror at the base of the windshield, its housing in the paint."""
  u, v = np.meshgrid(np.linspace(0, np.pi, 10), np.linspace(0, 2 * np.pi, 16, endpoint=False), indexing='ij')
  P = np.stack([0.09 * np.sin(u) * np.cos(v), 0.08 * np.cos(u), 0.055 * np.sin(u) * np.sin(v)], -1)
  m = grid_mesh(P, lambda p, n: (np.repeat(BODY[None], len(p), 0), np.full(len(p), -1.0)), caps=False)
  m['pos'] = m['pos'] + np.array([0.86, side * (W / 2 + 0.07), 1.10])
  return m


def car():
  parts = [body_mesh(), mirror(-1), mirror(1)]
  for ax in AXLES:
    for side in (-1, 1):
      w = wheel()
      w['pos'] = w['pos'] + np.array([ax, side * (W / 2 - WHEEL_W / 2 - 0.03), WHEEL_R])
      parts.append(w)
  return merge(*parts)


def library():
  """The one car, the lead and the ego car alike, with its nominal (L, W, H)."""
  return {'car': (car(), (L, W, H))}


def write_library(path=RESOURCES / 'diorama-assets.bin') -> int:
  """diorama-assets.bin: u32 header length, the JSON header (per mesh: dims, vertex and index counts and offsets),
  then every mesh's vertices as float32 (x, y, z, nx, ny, nz, r, g, b, emission) and its triangles as uint16."""
  import json
  import struct
  header, verts, idx, v_at, i_at = {}, [], [], 0, 0
  for name, (m, dims) in library().items():
    n, k = len(m['pos']), m['idx'].size
    assert n < 65536, name
    header[name] = {'dims': [float(d) for d in dims], 'vertices': n, 'indices': k, 'vertexOffset': v_at, 'indexOffset': i_at}
    verts.append(np.hstack([m['pos'], m['nrm'], m['col'], m['emis'][:, None]]).astype('<f4'))
    idx.append(m['idx'].reshape(-1).astype('<u2'))
    v_at += n
    i_at += k
  head = json.dumps(header, separators=(',', ':')).encode()
  head += b' ' * (-len(head) % 4)
  body = np.concatenate(verts).tobytes() + np.concatenate(idx).tobytes()
  with open(path, 'wb') as fh:
    fh.write(struct.pack('<I', len(head)) + head + body)
  return 4 + len(head) + len(body)

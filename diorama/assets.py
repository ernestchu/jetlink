"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Diorama's models: a procedural low-poly-but-smooth "clay" sedan (an original
design, no external models), the one vehicle the page draws, and the binary
file the page loads it from (`write_library`).

Local frame: x forward, y right, z UP, origin on the ground under the object's centre, units in metres.
Each mesh is a dict of float32 arrays: pos (N,3), nrm (N,3), col (N,3) in 0..1, emis (N,) and idx (M,3) uint32.
"""
import numpy as np

BODY = np.array([0.62, 0.62, 0.64])
GLASS = np.array([0.17, 0.18, 0.20])
TIRE = np.array([0.10, 0.10, 0.11])
HUB = np.array([0.50, 0.51, 0.53])


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


def transform(m, R=None, t=None, scale=None):
  R = np.eye(3) if R is None else R
  t = np.zeros(3) if t is None else t
  scale = np.ones(3) if scale is None else scale
  out = dict(m)
  out['pos'] = (m['pos'] * scale) @ R.T + t
  n = (m['nrm'] / scale) @ R.T
  out['nrm'] = n / np.linalg.norm(n, axis=1, keepdims=True)
  return out


def grid_mesh(P, color_fn, cap_start=True, cap_end=True, center_fn=None):
  """P: (N stations, K ring points, 3) closed rings -> smooth mesh with optional end caps."""
  N, K, _ = P.shape
  dt = np.gradient(P, axis=0)
  dk = np.roll(P, -1, axis=1) - np.roll(P, 1, axis=1)
  n = np.cross(dt, dk)
  centre = P.mean(axis=1, keepdims=True) if center_fn is None else center_fn(P)
  flip = np.sum(n * (P - centre), axis=-1) < 0
  n[flip] *= -1
  n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
  pos, nrm = P.reshape(-1, 3), n.reshape(-1, 3)
  col, emis = color_fn(pos, nrm)
  i, k = np.meshgrid(np.arange(N - 1), np.arange(K), indexing='ij')
  a, b = i * K + k, i * K + (k + 1) % K
  c, d = a + K, b + K
  idx = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([b, d, c], -1).reshape(-1, 3)])
  meshes = [{'pos': pos, 'nrm': nrm, 'col': col, 'emis': emis, 'idx': idx}]
  for end, use in ((0, cap_start), (N - 1, cap_end)):
    if not use:
      continue
    ring = P[end]
    ax = P[min(end + 1, N - 1)] - P[max(end - 1, 0)]
    ax = ax.mean(0)
    ax = ax / max(np.linalg.norm(ax), 1e-9) * (-1 if end == 0 else 1)
    cp = np.vstack([ring.mean(0, keepdims=True), ring])
    cn = np.repeat(ax[None], K + 1, 0)
    cc, ce = color_fn(cp, cn)
    ring_i = np.arange(1, K + 1)
    tri = np.stack([np.zeros(K, int), ring_i, np.roll(ring_i, -1)], 1)
    meshes.append({'pos': cp, 'nrm': cn, 'col': cc, 'emis': ce, 'idx': tri})
  return merge(*meshes)


def superellipse(K, p):
  th = np.linspace(0, 2 * np.pi, K, endpoint=False)
  c, s = np.cos(th), np.sin(th)
  return np.sign(c) * np.abs(c) ** (2 / p), np.sign(s) * np.abs(s) ** (2 / p)


# ---------------- car bodies ----------------
def car_body(L, W, H, profile, body=BODY, glass=GLASS, p=4.2, N=56, K=40, tail_lights=False, head_lights=False):
  """Loft along x. profile(t) -> (half_width_frac, bottom, belt, top) with t in 0 (rear) .. 1 (front), heights in m."""
  ts = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, N))  # denser at the ends
  yy, zz = superellipse(K, p)
  P = np.zeros((N, K, 3))
  belts, tops = np.zeros(N), np.zeros(N)
  for i, t in enumerate(ts):
    wf, zb, belt, top = profile(t)
    z = zb + (zz + 1) / 2 * (top - zb)
    above = np.clip((z - belt) / max(top - belt, 1e-3), 0, 1)
    scale = np.where(z > belt, 0.96 - 0.24 * above, 1.0)  # tumblehome: cabin narrower towards the roof
    P[i, :, 0] = (t - 0.5) * L
    P[i, :, 1] = yy * wf * W / 2 * scale
    P[i, :, 2] = z
    belts[i], tops[i] = belt, top
  belt_at = lambda x: np.interp(x, (ts - 0.5) * L, belts)  # noqa: E731
  top_at = lambda x: np.interp(x, (ts - 0.5) * L, tops)  # noqa: E731

  def color_fn(pos, nrm):
    x, z = pos[:, 0], pos[:, 2]
    is_glass = (z > belt_at(x) + 0.05) & (z < top_at(x) - 0.06) & (top_at(x) - belt_at(x) > 0.25)
    col = np.where(is_glass[:, None], glass, body)
    ao = 0.68 + 0.32 * smoothstep(0.1, 0.75, z / H)  # baked darkening towards the bottom
    col = col * ao[:, None]
    emis = np.zeros(len(pos))
    if tail_lights:
      tl = (x < -L / 2 + 0.10) & (np.abs(z - belt_at(x) + 0.07) < 0.035) & (np.abs(pos[:, 1]) > 0.28 * W)
      col[tl] = [0.55, 0.05, 0.05]
      emis[tl] = 0.6
    if head_lights:
      hl = (x > L / 2 - 0.25) & (np.abs(z - belt_at(x) + 0.08) < 0.05) & (np.abs(pos[:, 1]) > 0.22 * W)
      col[hl] = [0.95, 0.95, 0.92]
      emis[hl] = 0.5
    return col, emis

  return grid_mesh(P, color_fn, center_fn=lambda P: np.stack([P[..., 0].mean(1, keepdims=True),
                                                              np.zeros((P.shape[0], 1)),
                                                              P[..., 2].mean(1, keepdims=True)], -1))


def wheel(r, w, K=28):
  """Tyre (cylinder along y) with a lighter hub face on the outside."""
  th = np.linspace(0, 2 * np.pi, K, endpoint=False)
  rings = []
  for y, rr in ((-w / 2, r * 0.8), (-w / 2, r), (w / 2, r), (w / 2, r * 0.8)):
    rings.append(np.stack([rr * np.cos(th), np.full(K, y), rr * np.sin(th)], 1))
  P = np.array(rings)

  def color_fn(pos, nrm):
    hub = (np.abs(nrm[:, 1]) > 0.7) & (np.hypot(pos[:, 0], pos[:, 2]) < 0.82 * r)
    return np.where(hub[:, None], HUB, TIRE), np.zeros(len(pos))

  return grid_mesh(P, color_fn, center_fn=lambda P: np.zeros_like(P[:, :1]) + np.array([0, 0, 0]))


def with_wheels(body, L, W, r, axles, w=0.24):
  parts = [body]
  for ax in axles:
    for side in (-1, 1):
      parts.append(transform(wheel(r, w), t=np.array([ax, side * (W / 2 - w / 2 + 0.03), r])))
  return merge(*parts)


def sedan(L=4.7, W=1.85, H=1.45, body=BODY, ego=False):
  def prof(t):
    s = 2 * t - 1
    e = 4.5 if s > 0 else 7.0
    wf = (1 - np.abs(s) ** e) ** (1 / e)
    zb = 0.16 + 0.16 * smoothstep(0.86, 1, t) + 0.14 * smoothstep(0.12, 0, t)
    belt = 0.80 - 0.10 * smoothstep(0.55, 0.92, t) - 0.18 * smoothstep(0.90, 1, t) - 0.10 * smoothstep(0.08, 0, t)
    cabin = smoothstep(0.14, 0.36, t) * smoothstep(0.80, 0.58, t)
    return max(wf, 0.05), zb, belt, belt + (H - belt) * cabin
  b = car_body(L, W, H, prof, body=body, tail_lights=True, head_lights=ego)
  return with_wheels(b, L, W, 0.34, (-0.31 * L, 0.33 * L))


def library():
  """The one vehicle Diorama draws, the lead, and the ego car, each with its nominal (L, W, H)."""
  return {
    'sedan': (sedan(), (4.7, 1.85, 1.45)),
    'ego': (sedan(body=np.array([0.93, 0.93, 0.94]), ego=True), (4.7, 1.85, 1.45)),
  }


def write_library(path) -> int:
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

"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Diorama's models: procedural low-poly-but-smooth "clay" vehicles, people and
traffic lights (original designs, no external models), and the binary file the
page loads them from (`python -m diorama assets` writes it).

Local frame: x forward, y right, z UP, origin on the ground under the object's centre, units in metres.
Each mesh is a dict of float32 arrays: pos (N,3), nrm (N,3), col (N,3) in 0..1, emis (N,) and idx (M,3) uint32.
"""
import numpy as np

BODY = np.array([0.62, 0.62, 0.64])
GLASS = np.array([0.17, 0.18, 0.20])
TIRE = np.array([0.10, 0.10, 0.11])
HUB = np.array([0.50, 0.51, 0.53])
SKIN = np.array([0.80, 0.80, 0.82])


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


def rot_z(a):
  c, s = np.cos(a), np.sin(a)
  return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rot_y(a):
  c, s = np.cos(a), np.sin(a)
  return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


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


def suv(L=4.9, W=1.95, H=1.75, body=BODY):
  def prof(t):
    s = 2 * t - 1
    e = 5.0 if s > 0 else 9.0
    wf = (1 - np.abs(s) ** e) ** (1 / e)
    zb = 0.24 + 0.16 * smoothstep(0.88, 1, t) + 0.10 * smoothstep(0.08, 0, t)
    belt = 0.98 - 0.12 * smoothstep(0.62, 0.92, t) - 0.20 * smoothstep(0.92, 1, t)
    cabin = smoothstep(0.0, 0.07, t) * smoothstep(0.84, 0.64, t)
    return max(wf, 0.05), zb, belt, belt + (H - belt) * cabin
  b = car_body(L, W, H, prof, body=body, tail_lights=True, p=3.6)
  return with_wheels(b, L, W, 0.38, (-0.31 * L, 0.32 * L))


def hatchback(L=4.3, W=1.8, H=1.48, body=BODY):
  def prof(t):
    s = 2 * t - 1
    e = 4.5 if s > 0 else 9.0
    wf = (1 - np.abs(s) ** e) ** (1 / e)
    zb = 0.17 + 0.15 * smoothstep(0.86, 1, t) + 0.10 * smoothstep(0.08, 0, t)
    belt = 0.84 - 0.10 * smoothstep(0.58, 0.92, t) - 0.18 * smoothstep(0.90, 1, t)
    cabin = smoothstep(0.0, 0.10, t) * smoothstep(0.80, 0.60, t)
    return max(wf, 0.05), zb, belt, belt + (H - belt) * cabin
  b = car_body(L, W, H, prof, body=body, tail_lights=True, p=4.0)
  return with_wheels(b, L, W, 0.33, (-0.32 * L, 0.33 * L))


def van(L=5.2, W=2.0, H=2.05, body=BODY):
  def prof(t):
    s = 2 * t - 1
    e = 6.0 if s > 0 else 12.0
    wf = (1 - np.abs(s) ** e) ** (1 / e)
    zb = 0.24 + 0.12 * smoothstep(0.90, 1, t) + 0.08 * smoothstep(0.06, 0, t)
    belt = 1.08 - 0.12 * smoothstep(0.80, 0.95, t) - 0.18 * smoothstep(0.94, 1, t)
    cabin = smoothstep(0.0, 0.05, t) * smoothstep(0.94, 0.80, t)
    return max(wf, 0.05), zb, belt, belt + (H - belt) * cabin
  b = car_body(L, W, H, prof, body=body, tail_lights=True, p=5.0)
  return with_wheels(b, L, W, 0.37, (-0.32 * L, 0.34 * L))


def pickup(L=5.6, W=2.0, H=1.85, body=BODY):
  def prof(t):
    s = 2 * t - 1
    wf = (1 - np.abs(s) ** (3.0 if s > 0 else 10.0)) ** (1 / (3.0 if s > 0 else 10.0))
    zb = 0.30 + 0.08 * smoothstep(0.9, 1, t)
    belt = 1.08 - 0.10 * smoothstep(0.7, 0.95, t) - 0.18 * smoothstep(0.94, 1, t)
    cabin = smoothstep(0.40, 0.45, t) * smoothstep(0.80, 0.68, t)
    return max(wf, 0.05), zb, belt, belt + (H - belt) * cabin
  b = car_body(L, W, H, prof, body=body, tail_lights=True, p=4.0)
  return with_wheels(b, L, W, 0.4, (-0.30 * L, 0.32 * L), w=0.28)


def box_truck(L=7.5, W=2.4, H=3.2, body=BODY):
  def cab_prof(t):
    wf = (1 - np.abs(2 * t - 1) ** 8) ** (1 / 8)
    return max(wf, 0.05), 0.45, 1.4, 1.4 + (2.5 - 1.4) * smoothstep(0.0, 0.3, t) * smoothstep(1.0, 0.75, t)
  def box_prof(t):
    wf = (1 - np.abs(2 * t - 1) ** 14) ** (1 / 14)
    return max(wf, 0.05), 0.75, H, H
  cab = transform(car_body(1.9, W * 0.96, 2.5, cab_prof, body=body, p=6), t=np.array([L / 2 - 0.95, 0, 0]))
  box = transform(car_body(L - 2.1, W, H, box_prof, body=body * 1.05, p=10, glass=body * 1.05), t=np.array([-1.05, 0, 0]))
  return with_wheels(merge(cab, box), L, W, 0.5, (-0.30 * L, 0.34 * L), w=0.32)


def semi(L=16.0, W=2.55, H=4.0, body=BODY):
  def tractor_prof(t):
    wf = (1 - np.abs(2 * t - 1) ** 6) ** (1 / 6)
    hood = 1.7 + (3.3 - 1.7) * smoothstep(0.55, 0.62, 1 - t)
    return max(wf, 0.05), 0.5, 1.7, hood if t < 0.62 else 1.7 + 0.5 * smoothstep(1.0, 0.7, t)
  def trailer_prof(t):
    wf = (1 - np.abs(2 * t - 1) ** 16) ** (1 / 16)
    return max(wf, 0.05), 1.2, H, H
  tractor = transform(car_body(5.5, W * 0.95, 3.3, tractor_prof, body=body, p=6), t=np.array([L / 2 - 2.75, 0, 0]))
  trailer = transform(car_body(L - 6.2, W, H, trailer_prof, body=body * 1.08, glass=body * 1.08, p=12), t=np.array([-3.1, 0, 0]))
  return with_wheels(merge(tractor, trailer), L, W, 0.52, (L / 2 - 1.2, L / 2 - 4.6, -L / 2 + 2.2, -L / 2 + 3.6), w=0.34)


def bus(L=12.0, W=2.55, H=3.2, body=BODY):
  def prof(t):
    wf = (1 - np.abs(2 * t - 1) ** 14) ** (1 / 14)
    return max(wf, 0.05), 0.35, 1.25, H
  b = car_body(L, W, H, prof, body=body, p=9)
  # window band along the sides
  z, n = b['pos'][:, 2], b['nrm']
  band = (z > 1.35) & (z < 2.65) & (np.abs(n[:, 1]) > 0.6)
  b['col'][band] = GLASS
  return with_wheels(b, L, W, 0.5, (-0.28 * L, 0.30 * L), w=0.32)


# ---------------- people / two-wheelers ----------------
def lathe(zs, rs, K=24, depth=0.62, color=SKIN):
  """Surface of revolution about z (flattened front-to-back by `depth`)."""
  th = np.linspace(0, 2 * np.pi, K, endpoint=False)
  P = np.stack([np.stack([r * depth * np.cos(th), r * np.sin(th), np.full(K, z)], 1) for z, r in zip(zs, rs, strict=True)])
  return grid_mesh(P, lambda pos, nrm: (np.tile(color, (len(pos), 1)) * (0.75 + 0.25 * smoothstep(0, 1.6, pos[:, 2]))[:, None],
                                        np.zeros(len(pos))),
                   center_fn=lambda P: np.stack([np.zeros((P.shape[0], 1)), np.zeros((P.shape[0], 1)), P[..., 2].mean(1, keepdims=True)], -1))


def mannequin(h=1.75, color=SKIN):
  """Smooth standing figure: two legs, torso, arms, neck, head."""
  k = h / 1.75
  z = np.array([0.47, 0.50, 0.55, 0.62, 0.70, 0.78, 0.81, 0.83, 0.85, 0.875, 0.90, 0.935, 0.97, 0.99, 1.0]) * h
  r = np.array([0.15, 0.18, 0.17, 0.16, 0.18, 0.21, 0.22, 0.17, 0.065, 0.06, 0.09, 0.105, 0.10, 0.07, 0.01]) * k
  torso = lathe(z, r, color=color)
  legs = [tube([0, s * 0.09 * k, 0.50 * h], [0, s * 0.10 * k, 0.03 * h], 0.075 * k, color) for s in (-1, 1)]
  feet = [tube([-0.03, s * 0.10 * k, 0.03 * h], [0.14 * k, s * 0.10 * k, 0.02 * h], 0.05 * k, color) for s in (-1, 1)]
  arms = [tube([0, s * 0.22 * k, 0.80 * h], [0.02, s * 0.25 * k, 0.48 * h], 0.05 * k, color) for s in (-1, 1)]
  return merge(torso, *legs, *feet, *arms)


def tube(p0, p1, r, color, K=10):
  p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
  d = p1 - p0
  L = np.linalg.norm(d)
  th = np.linspace(0, 2 * np.pi, K, endpoint=False)
  ring = np.stack([np.zeros(K), r * np.cos(th), r * np.sin(th)], 1)
  P = np.stack([ring, ring + [L, 0, 0]])
  m = grid_mesh(P, lambda pos, nrm: (np.tile(color, (len(pos), 1)), np.zeros(len(pos))))
  a = d / L
  v = np.cross([1, 0, 0], a)
  s, c = np.linalg.norm(v), np.dot([1, 0, 0], a)
  if s < 1e-9:
    R = np.eye(3) if c > 0 else np.diag([-1, -1, 1])
  else:
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + vx + vx @ vx * (1 - c) / s ** 2
  return transform(m, R=R, t=p0)


def rider(seat_z, color=SKIN):
  torso = lathe(np.array([0.0, 0.08, 0.30, 0.48, 0.52, 0.55, 0.60, 0.67, 0.70]),
                np.array([0.15, 0.17, 0.16, 0.22, 0.07, 0.065, 0.10, 0.10, 0.01]), color=color)
  torso = transform(torso, R=rot_y(0.35), t=np.array([-0.1, 0, seat_z]))  # leaning forward
  arms = [tube([0.05, s * 0.2, seat_z + 0.42], [0.42, s * 0.25, seat_z + 0.2], 0.045, color) for s in (-1, 1)]
  legs = [tube([-0.05, s * 0.13, seat_z + 0.02], [0.25, s * 0.16, seat_z - 0.35], 0.06, color) for s in (-1, 1)]
  return merge(torso, *arms, *legs)


def motorcycle(body=BODY):
  frame = lathe(np.linspace(-0.85, 0.85, 9), np.array([0.06, 0.16, 0.22, 0.24, 0.24, 0.23, 0.2, 0.14, 0.05]), depth=1.0, color=body)
  frame = transform(frame, R=rot_y(np.pi / 2), t=np.array([0, 0, 0.62]))
  wheels = [transform(wheel(0.32, 0.14), t=np.array([x, 0, 0.32])) for x in (-0.72, 0.72)]
  return merge(frame, *wheels, rider(0.82))


def bicycle(body=BODY):
  ring = lambda x: transform(wheel(0.34, 0.05, K=24), t=np.array([x, 0, 0.34]))  # noqa: E731
  frame_pts = [([-0.52, 0, 0.34], [0.0, 0, 0.75]), ([0.0, 0, 0.75], [0.48, 0, 0.8]), ([0.48, 0, 0.8], [0.55, 0, 0.34]),
               ([-0.52, 0, 0.34], [0.05, 0, 0.36]), ([0.05, 0, 0.36], [0.0, 0, 0.75]), ([0.05, 0, 0.36], [0.48, 0, 0.8])]
  frame = [tube(a, b, 0.025, body * 0.8) for a, b in frame_pts]
  return merge(ring(-0.55), ring(0.55), *frame, rider(0.85))


def traffic_light(height=4.5):
  """Pole + 3-lamp housing facing -x (towards the viewer when yawed to face the ego car). Lamps get emissive later."""
  pole = tube([0, 0, 0], [0, 0, height - 0.5], 0.06, np.array([0.45, 0.46, 0.48]))
  def prof(t):
    return 1.0, height - 0.5, height + 0.5, height + 0.5
  housing = car_body(0.32, 0.36, height + 0.5, prof, body=np.array([0.16, 0.16, 0.18]), glass=np.array([0.16, 0.16, 0.18]), p=8, N=8, K=24)
  # car_body puts the loft along x from bottom..top in z already; housing spans z height-0.5 .. height+0.5
  lamps = []
  for i, zc in enumerate((height + 0.3, height, height - 0.3)):
    disc = transform(wheel(0.11, 0.04, K=20), R=rot_z(np.pi / 2), t=np.array([-0.18, 0, zc]))
    disc['col'][:] = 0.12
    disc['emis'][:] = -1 - i  # marker: lamp index (red, yellow, green), resolved in the shader
    lamps.append(disc)
  return merge(pole, housing, *lamps)


def library():
  """Canonical assets keyed by display type, with their nominal (L, W, H)."""
  return {
    'sedan': (sedan(), (4.7, 1.85, 1.45)),
    'suv': (suv(), (4.9, 1.95, 1.75)),
    'hatchback': (hatchback(), (4.3, 1.8, 1.48)),
    'van': (van(), (5.2, 2.0, 2.05)),
    'pickup': (pickup(), (5.6, 2.0, 1.85)),
    'box_truck': (box_truck(), (7.5, 2.4, 3.2)),
    'semi': (semi(), (16.0, 2.55, 4.0)),
    'bus': (bus(), (12.0, 2.55, 3.2)),
    'motorcycle': (motorcycle(), (2.1, 0.8, 1.5)),
    'bicycle': (bicycle(), (1.8, 0.6, 1.7)),
    'person': (mannequin(), (0.5, 0.5, 1.75)),
    'traffic_light': (traffic_light(), (0.4, 0.4, 5.0)),
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

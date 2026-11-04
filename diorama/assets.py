"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Diorama's car: the one vehicle the page draws, as the lead and as the ego car,
and the binary file the page loads it from (`write_library`).

It is "2021 Tesla Model Y" by tonielpro520
(https://sketchfab.com/3d-models/2021-tesla-model-y-59e2ead369984b1a85c800ff6cf6789d),
licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Changed
for Diorama: recoloured in the page's flat colours by part, and its badges,
interior, brakes, the tail lamps' clear and red lenses and the frames hidden
behind its body left out. Its triangles are otherwise the model's own, at full
resolution.

To rebuild diorama-assets.bin, download the model's OBJ from that page, then (with scipy installed):

    python -m diorama assets ~/Downloads/2021-tesla-model-y/source/<model>.obj

Local frame: x forward, y right, z UP, origin on the ground under the car's centre, units in metres.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

RESOURCES = Path(__file__).resolve().parent.parent / 'JetlinkKit' / 'Sources' / 'JetlinkDiorama' / 'Resources'
DIMS = (4.76, 1.87, 1.66)  # length, width without the mirrors, height
SPAN = 4.0  # positions are stored as 16-bit fractions of this many metres either way

BODY = np.array([0.62, 0.62, 0.64])  # the paint as stored; the page colours it (emission -1 marks it)
GLASS = np.array([0.33, 0.33, 0.35])
TRIM = np.array([0.30, 0.30, 0.32])
TIRE = np.array([0.22, 0.22, 0.23])
RIM = np.array([0.40, 0.40, 0.42])
TAIL = np.array([0.80, 0.10, 0.10])  # the tail lamps' light strips
TAIL_DARK = TAIL * 0.42  # the dark red inside the lamps
HOUSING = TAIL * 0.38  # the lamps' housing, a smoked red that glows when braking, so a far lead's brake lights show
HEAD = np.array([0.88, 0.89, 0.91])

# the model's materials, and what each is in the page's colours
PARTS = {
  'Carro_Pintura': 'body', 'Carro_Vidros2': 'glass', 'Carro_Plastico_Brilho_1': 'glass', 'Carro_Espelhos': 'glass',
  'Carro_Plastico_Brilho': 'trim', 'Carro_Plastico': 'trim', 'Carro_Metal_Preto': 'trim', 'Carro_Pneu': 'tire',
  'Carro_Metal_Lanterna_Traseira': 'housing', 'Carro_Metal_Vermelho': 'tail', 'Carro_Metal_Vermelho_1': 'tail',
  'Carro_Metal_Vermelho_3': 'tail_dark', 'Carro_Metal_Vermelho_2': 'reflector',
  'Carro_Metal_Farol': 'head', 'Carro_Metal_Farol_1': 'head',
}  # left out: Carro_Cromado (the badges), Carro_Interno, Carro_Freio and Carro_Freio_Disco (unseen behind the glass and
# wheels), and Carro_Vidro_Vermelho2, the tail lamps' red lens: the page has no see-through, and drawn solid it hides them
AXLES, WHEEL_R, TRACK = (1.47, -1.42), 0.40, 0.773  # where the model's wheels are


def read_obj(path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """An OBJ's vertices, its triangles (polygons as fans) and each triangle's material."""
  verts, tris, mats, mat = [], [], [], ''
  with open(path, errors='replace') as fh:
    for line in fh:
      if line.startswith('v '):
        verts.append([float(t) for t in line.split()[1:4]])
      elif line.startswith('usemtl'):
        mat = line.split(None, 1)[1].strip()
      elif line.startswith('f '):
        idx = [int(t.split('/')[0]) - 1 for t in line.split()[1:]]
        for k in range(1, len(idx) - 1):
          tris.append((idx[0], idx[k], idx[k + 1]))
          mats.append(mat)
  return np.array(verts), np.array(tris), np.array(mats)


def kept(P, tris, mat):
  """The triangles of a material that can be seen."""
  c = P[tris].mean(axis=1)
  if mat == 'Carro_Plastico':  # of the matte black, the arch liners and lower trim; above, frames behind the body and glass
    return tris[P[tris, 2].max(axis=1) < 0.8]
  if mat == 'Carro_Plastico_Brilho_1':  # the glass roof; a strip of it flush on the tailgate would flicker against the paint
    return tris[P[tris, 2].min(axis=1) > 1.2]
  if mat == 'Carro_Metal_Preto':  # the wheels and the wipers, not the radiator behind the bumper
    wheel = np.zeros(len(tris), bool)
    for ax in AXLES:
      wheel |= (np.hypot(c[:, 0] - ax, c[:, 2] - WHEEL_R) < WHEEL_R) & (np.abs(c[:, 1]) > TRACK - 0.20)
    return tris[wheel | (c[:, 2] > 0.9)]
  return tris


def lenses(P, tris):
  """The window glass's material also makes the lamps' clear lenses, over the lamps: its pieces' roles by where each
  sits, the two at the tail's corners the tail lamps' lenses, those at the nose the headlamps'; the rest are windows."""
  from scipy.sparse import coo_matrix
  from scipy.sparse.csgraph import connected_components
  _, at = np.unique(np.round(P[tris.ravel()] / 0.001).astype(np.int64), axis=0, return_inverse=True)  # a piece's panels joined
  t = at.reshape(-1, 3)
  n = t.max() + 1
  edges = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
  _, piece = connected_components(coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)), directed=False)
  piece = piece[t[:, 0]]
  role = np.full(len(tris), 'glass', dtype=object)
  for i in np.unique(piece):
    c = P[tris[piece == i]].reshape(-1, 3).mean(axis=0)
    if c[0] < -1.8 and c[2] < 1.2:
      role[piece == i] = 'tail'
    elif c[0] > 1.6 and c[2] < 1.0:
      role[piece == i] = 'head'
  return role


def normals(p, t):
  """Angle-weighted vertex normals."""
  e = [p[t[:, (j + 1) % 3]] - p[t[:, j]] for j in range(3)]
  fn = np.cross(e[0], -e[2])
  fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
  n = np.zeros_like(p)
  for j in range(3):
    a, b = e[j], -e[(j + 2) % 3]
    cos = np.sum(a * b, axis=1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)
    np.add.at(n, t[:, j], fn * np.arccos(np.clip(cos, -1, 1))[:, None])
  return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def car(obj):
  """The model in the page's colours, a list of meshes (dicts of pos, nrm, col, emis, idx), one per material. The tail
  lamps glow (the brake lights the page brightens) and the headlamps too; the paint is marked with emission -1, for
  the page to colour: grey for the lead, and for the ego car."""
  V, tris, mats = read_obj(obj)
  P = np.c_[V[:, 2], -V[:, 0], V[:, 1]]  # the model is y up, its headlamps toward +z
  meshes = []
  pieces = []
  for mat, role in PARTS.items():
    mine = kept(P, tris[mats == mat], mat)
    if mat == 'Carro_Vidros2':
      roles = lenses(P, mine)  # the tail lamps' clear lenses left out, as their red one is
      pieces += [(mine[roles == r], r) for r in ('glass', 'head')]
    else:
      pieces.append((mine, role))
  for mine, role in pieces:
    if not len(mine):
      continue
    used, inv = np.unique(mine.ravel(), return_inverse=True)
    p, t = P[used], inv.reshape(-1, 3)
    col = np.repeat({'body': BODY, 'glass': GLASS, 'trim': TRIM, 'tire': TIRE, 'tail': TAIL, 'tail_dark': TAIL_DARK,
                     'housing': HOUSING, 'reflector': TAIL * 0.8, 'head': HEAD}[role][None], len(p), 0)
    if role == 'trim':  # the dark metal inside a wheel is its rim
      rim = np.zeros(len(p), bool)
      for ax in AXLES:
        rim |= (np.hypot(p[:, 0] - ax, p[:, 2] - WHEEL_R) < 0.8 * WHEEL_R) & (np.abs(p[:, 1]) > TRACK - 0.18)
      col[rim] = RIM
    emis = np.full(len(p), {'body': -1.0, 'tail': 1.0, 'tail_dark': 0.5, 'housing': 0.6, 'head': 0.5}.get(role, 0.0))
    meshes.append({'pos': p, 'nrm': normals(p, t), 'col': col, 'emis': emis, 'idx': t})
  return meshes


def parts(meshes, limit=65535):
  """The meshes packed into as few parts as fit 16-bit indices (each under `limit` vertices)."""
  out, cur = [], []
  for m in meshes:
    if cur and sum(len(c['pos']) for c in cur) + len(m['pos']) > limit:
      out.append(cur)
      cur = []
    cur.append(m)
  out.append(cur)
  packed = []
  for group in out:
    n, idx = 0, []
    for m in group:
      idx.append(m['idx'] + n)
      n += len(m['pos'])
    packed.append({k: np.concatenate([m[k] for m in group]) for k in ('pos', 'nrm', 'col', 'emis')} | {'idx': np.concatenate(idx)})
  return packed


def write_library(obj, path=RESOURCES / 'diorama-assets.bin') -> int:
  """diorama-assets.bin: u32 header length, the JSON header, then the vertices and the triangles.

  The header names each asset with its dims (L, W, H), `span` and its parts, each part's vertex and index counts and
  offsets. A vertex is 16 bytes: x, y, z as int16 fractions of `span` metres, the normal as three int8 (/127), the
  emission as int8 (/127: -1 marks the paint), r, g, b as uint8 (/255) and a byte of padding. Triangles are uint16,
  indices into their part's vertices."""
  import json
  import struct
  verts, idx, header, v_at, i_at = [], [], {}, 0, 0
  entry = {'dims': [float(d) for d in DIMS], 'span': SPAN, 'parts': []}
  for part in parts(car(obj)):
    n, k = len(part['pos']), part['idx'].size
    v = np.zeros((n, 16), np.uint8)
    v[:, 0:6] = np.round(np.clip(part['pos'] / SPAN, -1, 1) * 32767).astype('<i2').view(np.uint8).reshape(n, 6)
    v[:, 6:9] = np.round(np.clip(part['nrm'], -1, 1) * 127).astype(np.int8).view(np.uint8)
    v[:, 9] = np.round(np.clip(part['emis'], -1, 1) * 127).astype(np.int8).view(np.uint8)
    v[:, 10:13] = np.round(np.clip(part['col'], 0, 1) * 255).astype(np.uint8)
    entry['parts'].append({'vertices': n, 'indices': k, 'vertexOffset': v_at, 'indexOffset': i_at})
    verts.append(v.reshape(-1))
    idx.append(part['idx'].reshape(-1).astype('<u2'))
    v_at += n
    i_at += k
  header['car'] = entry
  head = json.dumps(header, separators=(',', ':')).encode()
  head += b' ' * (-len(head) % 4)
  body = np.concatenate(verts).tobytes() + np.concatenate(idx).tobytes()
  with open(path, 'wb') as fh:
    fh.write(struct.pack('<I', len(head)) + head + body)
  return 4 + len(head) + len(body)

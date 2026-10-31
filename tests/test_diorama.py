"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Diorama's offline half: the messages, the output parsing, the calibration
maths, and the page's engine (diorama-engine.js, under node) on a drive made
up here.
"""
from __future__ import annotations

import json
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from diorama import packets
from diorama.outputs import IDX_N, LEAD_MHP_N, LEAD_MHP_SELECTION, LEAD_TRAJ_LEN, LEAD_WIDTH, X_IDXS, parse
from diorama.route import euler_from_rot, rot_from_euler, warp_matrix

ROOT = Path(__file__).resolve().parent.parent
ENGINE = ROOT / 'JetlinkKit/Sources/JetlinkDiorama/Resources/diorama-engine.js'


def unrle(data: bytes, w: int, h: int) -> np.ndarray:
  (runs,) = struct.unpack_from('<I', data)
  pairs = np.frombuffer(data, np.uint8, 2 * runs, 4).reshape(-1, 2)
  return np.repeat(pairs[:, 0], pairs[:, 1].astype(int) + 1).reshape(h, w)


def test_seg_message_round_trips():
  rng = np.random.default_rng(0)
  qmap = np.full((128, 256), 255.0, np.float32)
  qmap[40:90, 10:200] = rng.integers(0, 4, (50, 190))
  qmap[100:, :] = 7  # runs longer than 256
  label = np.arange(200, dtype=np.float32) % 65
  score = np.where(np.arange(200) % 3 == 0, 0.9, 0.0).astype(np.float32)
  msg = packets.seg(42, 12.5, qmap[None], label, score)
  assert msg[:1] == packets.SEG
  frame, ms, w, h, nq = struct.unpack_from('<IfHHH', msg, 1)
  assert (frame, ms, w, h, nq) == (42, 12.5, 256, 128, 200)
  labels = np.frombuffer(msg, np.uint8, nq, 15)
  assert (labels[score == 0] == 255).all() and (labels[score > 0] == label[score > 0]).all()
  assert (unrle(msg[15 + 2 * nq:], w, h) == qmap).all()


def test_boxes_keep_the_scored_rows():
  rows = np.zeros((100, 16), np.float32)
  rows[:3, 6] = [0.5, 0.05, 0.2]
  msg = packets.boxes(8, 3.0, rows)
  assert struct.unpack_from('<IfH', msg, 1) == (8, 3.0, 2)
  assert len(msg) == 11 + 2 * 16 * 4


def test_file_framing():
  import io
  fh = io.BytesIO()
  for m in (packets.meta({'a': 1}), packets.drive(1, 0.05, 2.0, np.ones(3))):
    packets.write(fh, m)
  got = list(packets.read(fh.getvalue()))
  assert json.loads(got[0][1:]) == {'a': 1}
  assert struct.unpack_from('<Ifff', got[1], 1)[2:] == (2.0, 1.0)


def test_parse_heads():
  sizes = {'plan': 2 * IDX_N * 15, 'pose': 12, 'lane_lines_prob': 8,
           'lead': LEAD_MHP_N * (2 * LEAD_MHP_SELECTION * LEAD_TRAJ_LEN * LEAD_WIDTH // LEAD_MHP_SELECTION + LEAD_MHP_SELECTION)}
  slices, at = {}, 0
  for name, n in sizes.items():
    slices[name] = slice(at, at + n)
    at += n
  v = np.zeros(at, np.float32)
  v[slices['plan']][:IDX_N * 15] = np.arange(IDX_N * 15)
  v[slices['pose'].start + 6:slices['pose'].stop] = np.log(2.0)
  # the lead's second hypothesis is the likelier one for every selection, and says x = 30
  per = sizes['lead'] // LEAD_MHP_N
  lead = v[slices['lead']].reshape(LEAD_MHP_N, per)
  lead[1, :per - LEAD_MHP_SELECTION] = 30.0
  lead[1, -LEAD_MHP_SELECTION:] = 5.0
  v[slices['lead']] = lead.reshape(-1)
  out = parse(v, slices)
  assert out['plan'].shape == (IDX_N, 15) and out['plan'][1, 0] == 15
  assert np.allclose(out['pose_stds'], 2.0)
  assert out['lead'].shape == (LEAD_MHP_SELECTION, LEAD_TRAJ_LEN, LEAD_WIDTH)
  assert (out['lead'] == 30).all()
  assert np.allclose(out['lane_lines_prob'], 0.5)


def test_euler_round_trip_and_the_warp():
  rpy = np.array([0.01, -0.03, 0.02])
  assert np.allclose(euler_from_rot(rot_from_euler(rpy)), rpy)
  # uncalibrated, the narrow frame's principal point is the camera's centre row and column
  M = warp_matrix(np.zeros(3), 2648.0, (1928, 1208), big=False)
  p = M @ np.array([256.0, 47.6, 1.0])
  assert np.allclose(p[:2] / p[2], [964, 604])


def synthetic_drive(path: Path, frames: int = 40) -> None:
  """A straight road at 10 m/s, and one car 20 m ahead, seen every other frame."""
  fields = [('plan', 2 * IDX_N * 15), ('lane_lines', 2 * 4 * IDX_N * 2), ('lane_lines_prob', 8), ('road_edges', 2 * 2 * IDX_N * 2),
            ('lead', 2 * 3 * 6 * 4), ('lead_prob', 3), ('pose', 12)]
  meta = {
    'version': 1, 'fps': 20, 'height': 1.22, 'frame': {'w': 512, 'h': 256, 'f': 910.0, 'cx': 256.0, 'cy': 47.6},
    'drive': [list(f) for f in fields],
    'seg': {'labels': ['Road', 'Car'], 'road_class': {'Road': 1}, 'things': {'Car': 'car'}, 'ds': 2}, 'dd3d': None,
  }
  raw = {}
  plan = np.zeros((2, IDX_N, 15), np.float32)
  plan[0, :, 0] = X_IDXS
  plan[0, :, 3] = 10.0
  raw['plan'] = plan
  lanes = np.zeros((2, 4, IDX_N, 2), np.float32)
  for i, y in enumerate((-5.4, -1.8, 1.8, 5.4)):
    lanes[0, i, :, 0] = y
  raw['lane_lines'] = lanes
  raw['lane_lines_prob'] = np.full(8, 3.0, np.float32)
  edges = np.zeros((2, 2, IDX_N, 2), np.float32)
  edges[0, 0, :, 0], edges[0, 1, :, 0] = -7.0, 7.0
  raw['road_edges'] = edges
  raw['lead'] = np.zeros(2 * 3 * 6 * 4, np.float32)
  raw['lead_prob'] = np.full(3, -5.0, np.float32)
  pose = np.zeros(12, np.float32)
  pose[0] = 10.0
  raw['pose'] = pose
  values = np.concatenate([raw[n].reshape(-1) for n, _ in fields])
  # the car: a blob whose bottom row meets the road 20 m ahead (half-resolution query map)
  qmap = np.full((128, 256), 0.0, np.float32)
  v_bot = int((47.6 + 910 * 1.22 / 20) / 2)
  qmap[:, :] = 255
  qmap[v_bot:, :] = 0
  qmap[v_bot - 12:v_bot, 120:136] = 1
  label = np.full(200, 0, np.float32)
  label[1] = 1
  score = np.zeros(200, np.float32)
  score[0], score[1] = 0.9, 0.95
  with open(path, 'wb') as fh:
    packets.write(fh, packets.meta(meta))
    for k in range(frames):
      packets.write(fh, packets.drive(k, k / 20, 4.0, values))
      if k % 2 == 0:
        packets.write(fh, packets.seg(k, 20.0, qmap, label, score))


@pytest.mark.skipif(shutil.which('node') is None, reason='node runs the page engine')
def test_page_engine_tracks_the_car(tmp_path):
  drive = tmp_path / 'drive.diorama'
  synthetic_drive(drive)
  script = f"""
    const fs = require('fs');
    const E = require({json.dumps(str(ENGINE))});
    const buf = fs.readFileSync({json.dumps(str(drive))});
    const eng = new E.Engine();
    for (const m of E.readFile(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength))) eng.push(m);
    const s = eng.state;
    console.log(JSON.stringify({{frame: s.frame, view: s.view, v: s.hud.v, tracks: s.tracks.map((t) => [t.name, t.x[0], t.x[1], t.heading])}}));
  """
  out = json.loads(subprocess.run(['node', '-e', script], capture_output=True, text=True, check=True).stdout)
  assert out['frame'] == 39 and abs(out['v'] - 10) < 0.5
  assert out['view'] == 0  # 22 mph: the road view
  assert len(out['tracks']) == 1
  name, x, y, heading = out['tracks'][0]
  # the car stays 20 m ahead as both move at 10 m/s: the tracker started it at the ego car's speed
  assert name == 'car' and abs(x - 20) < 1.5 and abs(y) < 0.5
  assert abs(heading) < 0.2


def test_export_unpack_matches_opencv():
  torch = pytest.importorskip('torch')
  cv2 = pytest.importorskip('cv2')
  from diorama.export import unpack_rgb
  rng = np.random.default_rng(1)
  w6 = rng.integers(16, 236, (6, 128, 256)).astype(np.uint8)
  Y = np.zeros((256, 512), np.uint8)
  Y[0::2, 0::2], Y[1::2, 0::2], Y[0::2, 1::2], Y[1::2, 1::2] = w6[:4]
  ref = cv2.cvtColor(np.concatenate([Y.ravel(), w6[4].ravel(), w6[5].ravel()]).reshape(384, 512), cv2.COLOR_YUV2RGB_I420)
  got = unpack_rgb(torch.from_numpy(w6[None].astype(np.float32)))[0].permute(1, 2, 0).numpy()
  assert np.abs(got - ref).max() <= 1.0

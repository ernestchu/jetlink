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


def synthetic_drive(path: Path, frames: int = 40, car_at: float | None = 20.0) -> None:
  """A straight road at 10 m/s, seen every other frame, with a car on it `car_at` metres ahead (None: no
  segmentation at all); the model's lead 0.1 m further each frame, 20 m to 23.9 m."""
  fields = [('plan', 2 * IDX_N * 15), ('lane_lines', 2 * 4 * IDX_N * 2), ('lane_lines_prob', 8), ('road_edges', 2 * 2 * IDX_N * 2),
            ('lead', 2 * 3 * 6 * 4), ('lead_prob', 3), ('pose', 12)]
  meta = {
    'version': 1, 'fps': 20, 'height': 1.22, 'frame': {'w': 512, 'h': 256, 'f': 910.0, 'cx': 256.0, 'cy': 47.6},
    'drive': [list(f) for f in fields],
    'seg': {'labels': ['Road', 'Car'], 'road_class': {'Road': 1}, 'things': {'Car': 'car'}, 'ds': 2},
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
  raw['lead_prob'] = np.full(3, 5.0, np.float32)
  pose = np.zeros(12, np.float32)
  pose[0] = 10.0
  raw['pose'] = pose

  def values(k):
    # selection 0, now: x, y (the means of the (3, 6, 4) lead, then its log stds)
    raw['lead'][0], raw['lead'][1] = 20 + 0.1 * k, 0.3
    return np.concatenate([raw[n].reshape(-1) for n, _ in fields])
  # the car: a blob whose bottom row meets the road `car_at` ahead (half-resolution query map)
  qmap = np.full((128, 256), 255.0, np.float32)
  v_bot = int((47.6 + 910 * 1.22 / (car_at or 20)) / 2)
  qmap[v_bot:, :] = 0
  qmap[v_bot - 12:v_bot, 120:136] = 1
  label = np.full(200, 0, np.float32)
  label[1] = 1
  score = np.zeros(200, np.float32)
  score[0], score[1] = 0.9, 0.95
  with open(path, 'wb') as fh:
    packets.write(fh, packets.meta(meta))
    for k in range(frames):
      packets.write(fh, packets.drive(k, k / 20, 4.0, values(k)))
      if k % 2 == 0 and car_at is not None:
        packets.write(fh, packets.seg(k, 20.0, qmap, label, score))


def engine_state(tmp_path: Path, **drive) -> dict:
  """The page engine's state after a synthetic drive."""
  path = tmp_path / 'drive.diorama'
  synthetic_drive(path, **drive)
  script = f"""
    const fs = require('fs');
    const E = require({json.dumps(str(ENGINE))});
    const buf = fs.readFileSync({json.dumps(str(path))});
    const eng = new E.Engine();
    for (const m of E.readFile(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength))) eng.push(m);
    const s = eng.state;
    console.log(JSON.stringify({{frame: s.frame, view: s.view, v: s.hud.v, lead: s.lead}}));
  """
  return json.loads(subprocess.run(['node', '-e', script], capture_output=True, text=True, check=True).stdout)


@pytest.mark.skipif(shutil.which('node') is None, reason='node runs the page engine')
def test_page_engine_draws_each_frames_lead(tmp_path):
  out = engine_state(tmp_path)
  assert out['frame'] == 39 and abs(out['v'] - 10) < 0.5
  assert out['view'] == 0  # 22 mph: the road view
  # the last frame's own lead, not smoothed or tracked across frames, facing along the straight lane: the
  # segmentation's car stands at 20 m, within 25% of it
  lead = out['lead']
  assert abs(lead['x'] - 23.9) < 1e-4 and abs(lead['y'] - 0.3) < 1e-6
  assert lead['heading'] == 0 and lead['braking'] is False


@pytest.mark.skipif(shutil.which('node') is None, reason='node runs the page engine')
@pytest.mark.parametrize('car_at', [None, 40.0], ids=['no segmentation', 'the car seen is further on'])
def test_page_engine_hides_a_lead_the_segmentation_does_not_see(tmp_path, car_at):
  assert engine_state(tmp_path, car_at=car_at)['lead'] is None


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


def chooser(script: str) -> dict:
  """The page engine's ViewChooser, run by `script` under node, which prints JSON."""
  head = f"const E = require({json.dumps(str(ENGINE))});\n"
  return json.loads(subprocess.run(['node', '-e', head + script], capture_output=True, text=True, check=True).stdout)


@pytest.mark.skipif(shutil.which('node') is None, reason='node runs the page engine')
def test_view_pans_down_for_a_cue_held_a_moment_and_back_only_after_calm_at_speed():
  out = chooser("""
    const c = new E.ViewChooser({}), mps = 30 / E.MPH, seen = [];
    const run = (cue, seconds, v = mps) => { for (let i = 0; i < seconds * 20; i++) seen.push(c.step(cue, v)); return seen[seen.length - 1]; };
    console.log(JSON.stringify({
      cruising: run('', 4),
      glimpse: run('turn', 0.3),  // shorter than `enter`: no pan
      turn: run('turn', 0.2),
      after: run('', 2.0),  // within `leave` (and `hold`)
      calm: run('', 1.2),
      slowCalm: (run('turn', 4), run('', 5, 10 / E.MPH)),  // calm, but below roadAbove: stays
      why: c.why,
    }));
  """)
  assert out == {'cruising': 0, 'glimpse': 0, 'turn': 1, 'after': 1, 'calm': 0, 'slowCalm': 1, 'why': 'turn'}


@pytest.mark.skipif(shutil.which('node') is None, reason='node runs the page engine')
def test_view_cues():
  out = chooser("""
    const c = new E.ViewChooser({}), mph = (m) => m / E.MPH;
    const xs = Array.from({length: 33}, (_, i) => i * 3), straight = xs.map(() => 0);
    const bend = xs.map((x) => (x > 15 ? (x - 15) ** 2 / 60 : 0));  // curving off to one side past 15 m
    const ramp = xs.map((x) => (x > 15 ? -((x - 15) ** 2) / 80 : 0));  // a sweeping bend, about 30 degrees by 40 m
    const pose = [100, 50, 0], ahead = (x) => [[[100 + x, 48], [100 + x, 52], [101 + x, 52], [101 + x, 48]]];
    const at = (m) => xs.map(() => mph(m)), slowing = (from, to) => xs.map((_, i) => mph(from + (to - from) * i / 32));
    console.log(JSON.stringify({
      stopped: c.cue(mph(1), xs, straight, at(1), [], pose, null),
      cruise: c.cue(mph(30), xs, straight, at(30), [], pose, null),
      turn: c.cue(mph(30), xs, bend, slowing(30, 10), [], pose, null),  // sharp, and slowing for it
      ramp: c.cue(mph(30), xs, ramp, at(30), [], pose, null),  // taken at speed: not a turn
      rampSlowing: c.cue(mph(30), xs, ramp, slowing(30, 5), [], pose, null),  // 19 mph by 40 m
      turnFast: c.cue(mph(60), xs, bend, at(60), [], pose, null),
      crossing: c.cue(mph(15), xs, straight, at(15), ahead(15), pose, null),
      crossingStopping: c.cue(mph(28), xs, straight, xs.map((_, i) => mph(Math.max(0, 28 - 3 * i))), ahead(20), pose, null),
      crossingOver: c.cue(mph(28), xs, straight, at(28), ahead(20), pose, null),  // driven straight over: not one
      crossingFar: c.cue(mph(15), xs, straight, at(15), ahead(60), pose, null),
      traffic: c.cue(mph(20), xs, straight, at(20), [], pose, {x: 9}),
      slow: c.cue(mph(8), xs, straight, at(8), [], pose, null),
    }));
  """)
  assert out == {'stopped': 'stopped', 'cruise': '', 'turn': 'turn', 'ramp': '', 'rampSlowing': 'turn', 'turnFast': '',
                 'crossing': 'crossing', 'crossingStopping': 'crossing', 'crossingOver': '', 'crossingFar': '',
                 'traffic': 'traffic', 'slow': 'slow'}

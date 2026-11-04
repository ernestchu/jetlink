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
from diorama.route import RLOG_SCHEMA, read_calibration, rot_from_euler, warp_matrix

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


def test_the_warp():
  R = rot_from_euler(np.array([0.01, -0.03, 0.02]))
  assert np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1)
  # at the default calibration (0 0 0, as mounted), the narrow frame's principal point is the camera's centre
  M = warp_matrix(np.zeros(3), 2648.0, (1928, 1208), big=False)
  p = M @ np.array([256.0, 47.6, 1.0])
  assert np.allclose(p[:2] / p[2], [964, 604])


def synthetic_drive(path: Path, frames: int = 40, car_at: float | None = 20.0, wide: bool = False, lead_prob: float = 5.0) -> None:
  """A straight road at 10 m/s, seen every other frame, with a car 1.8 m wide on it `car_at` metres ahead (None:
  no segmentation at all); the model's lead 0.1 m further each frame, 20 m to 23.9 m, at `lead_prob` (a logit).
  `wide`: the segmentations are the wide frame's."""
  fields = [('plan', 2 * IDX_N * 15), ('lane_lines', 2 * 4 * IDX_N * 2), ('lane_lines_prob', 8), ('road_edges', 2 * 2 * IDX_N * 2),
            ('lead', 2 * 3 * 6 * 4), ('lead_prob', 3), ('pose', 12)]
  meta = {
    'version': 1, 'fps': 20, 'height': 1.22, 'frame': {'w': 512, 'h': 256, 'f': 910.0, 'cx': 256.0, 'cy': 47.6},
    'wide': {'w': 512, 'h': 256, 'f': 455.0, 'cx': 256.0, 'cy': 151.8, 'from_device': [0.0, 0.0, 0.0]},
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
  raw['lead_prob'] = np.full(3, lead_prob, np.float32)
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
  half = round(910 * 0.9 / (car_at or 20) / 2)
  qmap[v_bot - 12:v_bot, 128 - half:128 + half] = 1
  label = np.full(200, 0, np.float32)
  label[1] = 1
  score = np.zeros(200, np.float32)
  score[0], score[1] = 0.9, 0.95
  with open(path, 'wb') as fh:
    packets.write(fh, packets.meta(meta))
    for k in range(frames):
      packets.write(fh, packets.drive(k, k / 20, 4.0, values(k)))
      if k % 2 == 0 and car_at is not None:
        packets.write(fh, packets.seg(k, 20.0, qmap, label, score, wide=wide))


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
  assert lead['heading'] == 0 and lead['braking'] is False and lead['seg'] is False


@pytest.mark.skipif(shutil.which('node') is None, reason='node runs the page engine')
@pytest.mark.parametrize('car_at', [None, 40.0], ids=['no segmentation', 'the car seen is further on'])
def test_page_engine_hides_a_lead_the_segmentation_does_not_see(tmp_path, car_at):
  assert engine_state(tmp_path, car_at=car_at)['lead'] is None


@pytest.mark.skipif(shutil.which('node') is None, reason='node runs the page engine')
@pytest.mark.parametrize('lead_prob', [-5.0, 5.0], ids=['the model has no lead', 'the model has one further on'])
def test_page_engine_draws_a_close_car_the_segmentation_alone_sees(tmp_path, lead_prob):
  # the car stands 10 m ahead: the lead is where it meets the road, in the lane's centre, plain and never braking
  lead = engine_state(tmp_path, car_at=10.0, lead_prob=lead_prob)['lead']
  assert abs(lead['x'] - 10) < 0.3 and lead['y'] == 0 and lead['heading'] == 0
  assert lead['seg'] is True and lead['braking'] is False


@pytest.mark.skipif(shutil.which('node') is None, reason='node runs the page engine')
def test_page_engine_draws_no_car_the_segmentation_alone_sees_beyond_15_m(tmp_path):
  assert engine_state(tmp_path, car_at=16.0, lead_prob=-5.0)['lead'] is None


@pytest.mark.skipif(shutil.which('node') is None, reason='node runs the page engine')
def test_page_engine_uses_only_the_narrow_frames_segmentation_in_the_road_view(tmp_path):
  out = engine_state(tmp_path, wide=True)
  assert out['view'] == 0 and out['lead'] is None


def test_a_wide_seg_message_is_a_seg_of_another_kind():
  qmap, label, score = np.zeros((128, 256), np.float32), np.zeros(4, np.float32), np.ones(4, np.float32)
  narrow, wide = packets.seg(2, 1.0, qmap, label, score), packets.seg(2, 1.0, qmap, label, score, wide=True)
  assert narrow[:1] == packets.SEG and wide[:1] == packets.WIDE_SEG and narrow[1:] == wide[1:]
  assert packets.wide_frame({'w': 512, 'h': 256, 'f': 910.0, 'cx': 256.0, 'cy': 47.6}) == \
    {'w': 512, 'h': 256, 'f': 455.0, 'cx': 256.0, 'cy': 151.8}


def rlog(segment: Path, *calibrations: tuple[str, list, list, list], other: int = 3) -> None:
  """A segment's rlog.zst: `other` events of members the schema leaves out, then each (status, rpyCalib,
  wideFromDeviceEuler, height) as a calibration event, as calibrationd logs them."""
  capnp = pytest.importorskip('capnp')
  zstandard = pytest.importorskip('zstandard')
  capnp.remove_import_hook()
  log = capnp.load(str(RLOG_SCHEMA))
  out = b''
  for _ in range(other):
    out += log.Event.new_message(logMonoTime=1, reserved60=None).to_bytes()
  for status, rpy, wide, height in calibrations:
    ev = log.Event.new_message(logMonoTime=2)
    cal = ev.init('calibration')
    cal.calStatus, cal.rpyCalib, cal.wideFromDeviceEuler, cal.height = status, rpy, wide, height
    out += ev.to_bytes()
  segment.mkdir(parents=True, exist_ok=True)
  (segment / 'rlog.zst').write_bytes(zstandard.ZstdCompressor().compress(out))


def test_the_calibration_is_the_first_calibrated_one_the_rlog_has(tmp_path):
  seg = tmp_path / 'r--0'
  rlog(seg, ('uncalibrated', [0, 0, 0], [0, 0, 0], [1.22]), ('calibrated', [0.0001, 0.0433, 0.0146], [0, 0.0217, -0.0161], [1.375]),
       ('calibrated', [0, 0.05, 0], [0, 0.03, 0], [1.4]))
  cal = read_calibration(seg)
  assert np.allclose(cal.rpy, (0.0001, 0.0433, 0.0146)) and np.allclose(cal.wide_from_device, (0, 0.0217, -0.0161))
  assert abs(cal.height - 1.375) < 1e-6


def test_a_segment_without_a_calibrated_rlog_is_refused(tmp_path):
  with pytest.raises(SystemExit, match='no rlog.zst'):
    read_calibration(tmp_path / 'none')
  rlog(tmp_path / 'never', ('uncalibrated', [0, 0, 0], [0, 0, 0], [1.22]))
  with pytest.raises(SystemExit, match='no calibrated'):
    read_calibration(tmp_path / 'never')
  (tmp_path / 'broken').mkdir()
  (tmp_path / 'broken' / 'rlog.zst').write_bytes(b'not zstd')
  pytest.importorskip('zstandard')
  with pytest.raises(SystemExit, match='cannot read'):
    read_calibration(tmp_path / 'broken')


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


RENDER = ENGINE.with_name('diorama-render.js')


def renderer(script: str) -> dict:
  """The page renderer's camera, run by `script` under node with R its exports, which prints JSON."""
  head = f'require({json.dumps(str(ENGINE))}); require({json.dumps(str(RENDER))}); const R = globalThis.DioramaRender;\n'
  return json.loads(subprocess.run(['node', '-e', head + script], capture_output=True, text=True, check=True).stdout)


def test_a_camera_file_is_any_views_numbers_with_no_limits_but_a_zoom_above_0():
  from diorama import camera
  assert camera.check({}) == {}
  wild = {'road': {'zoom': 40, 'pitch': 170, 'forward': -500, 'rise': 300, 'shift': 12}, 'bird': {'zoom': 0.01, 'pitch': -15}}
  assert camera.check(wild) == wild
  for bad, why in [
    ([], 'not a JSON object'),
    ({'side': {}}, '"side" is not a view'),
    ({'road': 1}, 'road is not an object'),
    ({'road': {'tilt': 1}}, 'road.tilt is not a camera number'),
    ({'road': {'zoom': 0}}, 'road.zoom is 0, not above 0'),
    ({'road': {'zoom': -1}}, 'road.zoom is -1, not above 0'),
    ({'bird': {'forward': '5'}}, 'bird.forward is "5", not a number'),
    ({'bird': {'shift': float('inf')}}, 'bird.shift is Infinity'),
    ({'bird': {'pitch': True}}, 'bird.pitch is true'),
    ({'bird': {'pitch': float('nan')}}, 'bird.pitch is NaN'),
  ]:
    with pytest.raises(ValueError, match=why):
      camera.check(bad)


def test_the_camera_file_saves_whole_and_loads_back(tmp_path, monkeypatch):
  from diorama import camera
  monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path))
  path = camera.default_path()
  assert path == tmp_path / 'jetlink' / 'diorama-camera.json'
  assert camera.load(path) is None
  camera.save(path, {'road': {'zoom': 0.7, 'pitch': 25, 'forward': 3}})
  assert camera.load(path) == {'road': {'zoom': 0.7, 'pitch': 25, 'forward': 3}}
  with pytest.raises(ValueError, match='road.zoom'):
    camera.save(path, {'road': {'zoom': 0}})
  assert camera.load(path)['road']['zoom'] == 0.7  # a bad camera leaves the saved one be
  path.write_text('{"road": ')
  with pytest.raises(ValueError, match=str(path)):
    camera.load(path)


def test_the_viewer_saves_the_setup_pages_camera_and_serves_it(tmp_path):
  import functools
  import http.server
  import threading
  import urllib.error
  import urllib.request

  from diorama.viewer import Handler

  file = tmp_path / 'cam' / 'diorama-camera.json'

  def server(setup: bool) -> str:
    s = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Handler, drive=tmp_path, camera=file, setup=setup))
    threading.Thread(target=s.serve_forever, daemon=True).start()
    servers.append(s)
    return f'http://127.0.0.1:{s.server_address[1]}/diorama-camera.json'

  def post(url: str, body: bytes, kind: str = 'application/json') -> tuple[int, str]:
    try:
      with urllib.request.urlopen(urllib.request.Request(url, body, {'Content-Type': kind})) as r:
        return r.status, r.read().decode().strip()
    except urllib.error.HTTPError as e:
      return e.code, e.read().decode().strip()

  servers = []
  try:
    setup, view = server(True), server(False)
    with pytest.raises(urllib.error.HTTPError, match='404'):
      urllib.request.urlopen(view)
    no_redirect = urllib.request.build_opener(type('NoRedirect', (urllib.request.HTTPRedirectHandler,), {'redirect_request': lambda *a: None}))
    with pytest.raises(urllib.error.HTTPError) as found:
      no_redirect.open(setup.replace('diorama-camera.json', ''))
    assert found.value.headers['Location'] == '/diorama.html?camera=setup'  # no drive, no footage
    assert post(setup, b'{"road": {"zoom": 0.8}}', 'text/plain')[0] == 415  # what a form elsewhere could send
    assert post(setup, b'{"road": {"zoom": -8}}') == (400, 'road.zoom is -8, not above 0')
    assert post(setup, b'{"road": ')[0] == 400
    assert not file.exists()
    assert post(view, b'{"road": {"zoom": 0.8}}')[0] == 404  # only the setup page saves
    assert post(setup, b'{"road": {"zoom": 0.8}}') == (200, str(file))
    with urllib.request.urlopen(view) as r:
      assert json.loads(r.read()) == {'road': {'zoom': 0.8}}
  finally:
    for s in servers:
      s.shutdown()


@pytest.mark.skipif(shutil.which('node') is None, reason='node runs the page renderer')
def test_the_renderer_reads_a_camera_file_as_the_viewer_checks_it():
  from diorama import camera
  got = renderer("""
    const out = { numbers: R.CAMERA_NUMBERS, over: R.readCamera({ bird: { pitch: 50 } }), base: R.defaultCamera(), bad: [] };
    out.wild = R.readCamera({ road: { zoom: 40, pitch: 170, forward: -500 } }).road;
    for (const f of [[], { side: {} }, { road: { tilt: 1 } }, { road: { zoom: 0 } }, { road: { zoom: -1 } }, { bird: { forward: '5' } }]) {
      try { R.readCamera(f); out.bad.push(null); } catch (e) { out.bad.push(e.message); }
    }
    console.log(JSON.stringify(out));
  """)
  assert tuple(got['numbers']) == camera.NUMBERS
  assert got['wild']['zoom'] == 40 and got['wild']['pitch'] == 170 and got['wild']['forward'] == -500  # no limits
  assert set(got['over']) == set(camera.VIEWS)
  assert got['over'] == {**got['base'], 'bird': {**got['base']['bird'], 'pitch': 50}}  # over the defaults
  assert all(got['bad'])


@pytest.mark.skipif(shutil.which('node') is None, reason='node runs the page renderer')
def test_pitch_and_zoom_keep_the_ego_car_where_it_is_wherever_forward_moved_the_camera():
  got = renderer("""
    const px = (n, s) => {  // where the ego car's middle on the road lands, or null behind the camera
      const c = R.Renderer.prototype.camera.call({ views: n }, s, 1.22, 1280, 720), v = [-0.75 - c.C[0], -c.C[1], 1.22 - c.C[2]];
      const d = [0, 1, 2].map((i) => c.M[3 * i] * v[0] + c.M[3 * i + 1] * v[1] + c.M[3 * i + 2] * v[2]);
      return d[0] > 0.5 ? [c.cx + c.f * d[1] / d[0], c.cy + c.f * d[2] / d[0]] : null;
    };
    const copy = (n) => JSON.parse(JSON.stringify(n)), out = { pitch: [], zoom: [] };
    for (const [view, s] of [['road', 0], ['road', 0.4], ['bird', 1]]) for (const moved of [0, 30, -60]) {  // pixels of the gesture
      const base = R.defaultCamera();
      R.moveForward(base, view, moved);
      for (const pitch of [-15, 0, 5, 45, 70, 89, 120, 170]) {
        const pitched = copy(base);
        pitched[view].pitch = pitch;
        if (s === 0 || s === 1) out.pitch.push({ at: px(base, s), now: px(pitched, s) });  // within a view; the pan draws the car its own way
        for (const ratio of [0.3, 0.8, 1.5, 2.5]) {
          const zoomed = copy(pitched);
          R.zoomView(zoomed, view, ratio);
          out.zoom.push({ at: px(pitched, s), now: px(zoomed, s) });
        }
      }
    }
    console.log(JSON.stringify(out));
  """)
  for kind, cases in got.items():
    kept = [v for v in cases if v['at'] and v['now']]
    assert len(kept) > len(cases) / 2, kind
    for v in kept:
      assert v['now'] == pytest.approx(v['at'], abs=1e-6), kind


@pytest.mark.skipif(shutil.which('node') is None, reason='node runs the page renderer')
def test_forward_slides_the_camera_along_the_road_level_at_any_pitch():
  got = renderer("""
    const out = [];
    for (const view of ['road', 'bird']) for (const pitch of [5, 30, 60, 85]) {
      const n = R.defaultCamera(), s = view === 'road' ? 0 : 1, cam = (n) => R.Renderer.prototype.camera.call({ views: n }, s, 1.22, 1280, 720);
      n[view].pitch = pitch;
      const before = cam(n);
      R.moveForward(n, view, 40);
      const after = cam(n);
      out.push({ C: [before.C, after.C], M: [[...before.M], [...after.M]], picture: [[before.f, before.cx, before.cy], [after.f, after.cx, after.cy]] });
    }
    console.log(JSON.stringify(out));
  """)
  for v in got:
    (b, a), (bm, am) = v['C'], v['M']
    assert a[0] > b[0] and a[1:] == pytest.approx(b[1:])  # forward, keeping its height
    assert am == pytest.approx(bm)  # and its aim
    assert v['picture'][0] == v['picture'][1]  # and the picture is not moved: nothing but the camera moves


@pytest.mark.skipif(shutil.which('node') is None, reason='node runs the page renderer')
def test_the_camera_is_set_over_a_still_scene_of_its_own():
  got = renderer("""
    const { state, engine } = R.previewScene(), E = globalThis.DioramaEngine;
    const shown = [0, 1, 2, 3].filter((i) => state.out.lane_lines_prob[2 * i + 1] > 0.15);
    const ys = shown.map((i) => [...new Set(E.X_IDXS.map((_, j) => E.lane(state.out.lane_lines, i, j, 0)))]);
    console.log(JSON.stringify({ lead: state.lead, gap: state.pathGap, front: E.EGO_FRONT, stripes: state.stripes.length, map: engine.roadmap, ys }));
  """)
  assert got['lead']['x'] == pytest.approx(got['front'] + 10) and not got['lead']['seg']  # the model's lead, 10 m ahead
  assert got['lead']['x'] - got['gap'] == pytest.approx(got['front'] + 8)  # the path stops 2 m short of it
  assert got['stripes'] == 0 and got['map'] is None  # nothing from the segmentation
  assert got['ys'] == [[-1.8], [1.8]]  # the ego lane's two lines, straight


@pytest.mark.skipif(shutil.which('node') is None, reason='node runs the page renderer')
def test_the_default_cameras_are_the_first_ones_further_back_with_a_longer_lens():
  got = renderer("""
    const k = R.CAMERA_BACK, P = [(globalThis.DioramaEngine.EGO_FRONT + globalThis.DioramaEngine.EGO_REAR) / 2, 0, 1.22];
    // the first cameras: the defaults with the eye moved back to the first one, at the first zoom
    const first = R.defaultCamera();
    for (const [name, v] of Object.entries(R.CAMERA_VIEWS)) {
      first[name].forward = v.first.eye[0] - v.eye[0];
      first[name].rise = v.first.eye[1] - v.eye[1];
      first[name].zoom = v.first.zoom;
      first[name].shift = 0;
    }
    const cam = (n, s) => R.Renderer.prototype.camera.call({ views: n }, s, 1.22, 1280, 720);
    const px = (c) => {  // where the ego car's middle on the road lands
      const v = P.map((p, i) => p - c.C[i]), d = [0, 1, 2].map((i) => c.M[3 * i] * v[0] + c.M[3 * i + 1] * v[1] + c.M[3 * i + 2] * v[2]);
      return [c.cx + c.f * d[1] / d[0], c.cy + c.f * d[2] / d[0]];
    };
    const out = [];
    for (const s of [0, 0.3, 0.7, 1]) {
      const a = cam(first, s), b = cam(R.defaultCamera(), s);
      out.push({ s, M: [[...a.M], [...b.M]], C: [a.C.map((v, i) => P[i] + k * (v - P[i])), b.C], f: [k * a.f, b.f], px: [px(a), px(b)] });
    }
    console.log(JSON.stringify({ k, out }));
  """)
  assert got['k'] > 1
  for v in got['out']:
    # further back, looking the same way, and in each view the car as big and where it was (the pan draws it its own way)
    for kind in ('M', 'C', 'f') + (('px',) if v['s'] in (0, 1) else ()):
      assert v[kind][1] == pytest.approx(v[kind][0], abs=1e-9), kind


@pytest.mark.skipif(shutil.which('node') is None, reason='node runs the page renderer')
def test_between_the_views_the_car_goes_straight_from_one_to_the_other():
  got = renderer("""
    const E = globalThis.DioramaEngine, P = [(E.EGO_FRONT + E.EGO_REAR) / 2, 0, 1.22], out = [];
    const px = (n, s) => {
      const c = R.Renderer.prototype.camera.call({ views: n }, s, 1.22, 1280, 720), v = P.map((p, i) => p - c.C[i]);
      const d = [0, 1, 2].map((i) => c.M[3 * i] * v[0] + c.M[3 * i + 1] * v[1] + c.M[3 * i + 2] * v[2]);
      return [c.cx + c.f * d[1] / d[0], c.cy + c.f * d[2] / d[0]];
    };
    const set = R.defaultCamera();
    set.road.pitch = 30; R.zoomView(set, 'road', 0.7); R.moveForward(set, 'road', -40);
    set.bird.pitch = 50; R.zoomView(set, 'bird', 1.4); R.moveForward(set, 'bird', 10);
    for (const n of [R.defaultCamera(), set]) for (const s of [0.1, 0.3, 0.5, 0.7, 0.9]) out.push({ s, a: px(n, 0), b: px(n, 1), at: px(n, s) });
    console.log(JSON.stringify(out));
  """)
  for v in got:
    a, b, at, s = v['a'], v['b'], v['at'], v['s']
    assert at == pytest.approx([a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s], abs=1e-6)
    assert 0 <= at[0] <= 1280 and 0 <= at[1] <= 720  # in the picture all the way


@pytest.mark.skipif(shutil.which('node') is None, reason='node runs the page renderer')
def test_a_touch_drag_pulls_the_road_along_under_the_fingers():
  got = renderer("""
    const hgt = 1.22, w = 1280, h = 720, out = [];
    const row = (n, s, G) => {  // where road point G lands
      const c = R.Renderer.prototype.camera.call({ views: n }, s, hgt, w, h), v = G.map((g, i) => g - c.C[i]);
      const d = [0, 1, 2].map((i) => c.M[3 * i] * v[0] + c.M[3 * i + 1] * v[1] + c.M[3 * i + 2] * v[2]);
      return c.cy + c.f * d[2] / d[0];
    };
    for (const [view, s] of [['road', 0], ['bird', 1]]) for (const pitch of [10, 40, 75]) for (const fwd of [-20, 0, 15])
      for (const [x, y] of [[640, 650], [300, 600], [900, 500]]) for (const dy of [-40, -3, 3, 40]) {
        const n = R.defaultCamera();
        n[view].pitch = pitch;
        R.moveForwardBy(n, view, fwd);
        const c = R.Renderer.prototype.camera.call({ views: n }, s, hgt, w, h);
        const m = R.forwardFollowing(c, hgt, x, y, dy);
        if (m === null) { out.push(null); continue; }
        // the road point under (x, y): the ray through it to the road
        const a = (x - c.cx) / c.f, b = (y - c.cy) / c.f, M = c.M;
        const dir = [M[0] + a * M[3] + b * M[6], M[1] + a * M[4] + b * M[7], M[2] + a * M[5] + b * M[8]], t = (hgt - c.C[2]) / dir[2];
        const G = c.C.map((v, i) => v + t * dir[i]);
        R.moveForwardBy(n, view, m);
        out.push({ want: y + dy, got: row(n, s, G) });
      }
    console.log(JSON.stringify(out));
  """)
  followed = [v for v in got if v]
  assert len(followed) > len(got) / 2
  for v in followed:
    assert v['got'] == pytest.approx(v['want'], abs=1e-6)


@pytest.mark.skipif(shutil.which('node') is None, reason='node runs the page renderer')
def test_the_picture_keeps_its_bottom_edge_whatever_its_height():
  got = renderer("""
    const E = globalThis.DioramaEngine, P = [(E.EGO_FRONT + E.EGO_REAR) / 2, 0, 1.22], out = [];
    const gap = (n, s, w, h) => {  // the car's middle above the bottom edge
      const c = R.Renderer.prototype.camera.call({ views: n }, s, 1.22, w, h), v = P.map((p, i) => p - c.C[i]);
      const d = [0, 1, 2].map((i) => c.M[3 * i] * v[0] + c.M[3 * i + 1] * v[1] + c.M[3 * i + 2] * v[2]);
      return h - (c.cy + c.f * d[2] / d[0]);
    };
    const set = R.defaultCamera();
    set.road.pitch = 30; R.zoomView(set, 'road', 0.8); R.moveForward(set, 'road', -30);
    set.bird.pitch = 55; R.zoomView(set, 'bird', 1.3);
    for (const n of [R.defaultCamera(), set]) for (const w of [402, 1280]) for (const s of [0, 0.4, 1])
      out.push([0.3, 0.5625, 1, 2.2].map((aspect) => gap(n, s, w, w * aspect)));
    console.log(JSON.stringify(out));
  """)
  for gaps in got:
    assert gaps == pytest.approx([gaps[1]] * len(gaps), abs=1e-6)  # the same gap at every height

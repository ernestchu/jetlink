"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

A recorded drive through the driving model and Diorama's models, into a
folder the Diorama page plays beside the drive's own footage:

    drive.diorama   the messages a live server would have streamed (packets.py)
    footage.mp4     the road camera over the same frames, for the side-by-side

The frames are what a comma would send a jetlink server, warped and packed the
same way, and the models see what they would see live, so a drive tried
here looks as it would from the car.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np

from diorama import packets
from diorama.driving import DrivingModel, providers
from diorama.outputs import CALIB_FIELDS, DRIVE_FIELDS, parse
from diorama.route import FPS, Route, euler_from_rot, rot_from_euler

CAMERA_HEIGHT = 1.22  # openpilot's, when the model never says
WARMUP = 100  # frames of history the driving model gets before the first one kept


def calibrate(route: Route, model: DrivingModel, frames: int, passes: int = 2) -> tuple[np.ndarray, float]:
  """calibrationd's estimate from the model's own pose, batch style: frames driven straight at speed say where
  straight ahead is in the camera. Each pass warps with the last pass's answer. Also the camera's height from
  road_transform where the model is sure of it."""
  rpy, height = np.zeros(3), CAMERA_HEIGHT
  for i in range(passes):
    model.reset()
    rpys, heights = [], []
    for _k, _f, warped in route.warped(rpy, 0, frames):
      out = parse(model.step(warped), model.slices, CALIB_FIELDS)
      trans, rot, trans_std = out['pose'][:3], out['pose'][3:], out['pose_stds'][:3]
      if trans[0] > 15 * 0.44704 and abs(rot[2]) < np.radians(2) and np.arctan2(trans_std[1], trans[0]) < np.radians(0.25):
        obs = np.array([0.0, -np.arctan2(trans[2], trans[0]), np.arctan2(trans[1], trans[0])])
        rpys.append(euler_from_rot(rot_from_euler(rpy) @ rot_from_euler(obs)))
        if 'road_transform' in out and out['road_transform_stds'][2] < np.exp(-3.5):
          heights.append(out['road_transform'][2])
    if not rpys:
      print('calibration: never straight at speed in the first '
            f'{frames / FPS:.0f} s; using the camera as mounted (pass --rpy to set it)')
      return rpy, height
    rpy = np.mean(rpys, axis=0)
    height = float(np.mean(heights)) if heights else CAMERA_HEIGHT
    print(f'calibration pass {i + 1}: rpy {np.degrees(rpy).round(2).tolist()} deg, height {height:.2f} m, '
          f'{len(rpys)} frames straight at speed')
  return rpy, height


class Models:
  """Diorama's segmentation on onnxruntime, as a server runs it: the packed narrow frame in as float16."""

  def __init__(self, folder: Path, device: str):
    import onnxruntime as ort
    self.meta = json.loads((folder / 'diorama.json').read_text())
    prov = providers(device)
    self.seg = ort.InferenceSession(str(folder / self.meta['seg']['file']), providers=prov)

  def segment(self, narrow: np.ndarray):
    return self.seg.run(None, {'frame': narrow[None].astype(np.float16)})


def drive_meta(model: DrivingModel, models: Models, height: float, rpy, route: Route, start: int) -> dict:
  m = models.meta
  return {
    'version': 1, 'fps': FPS, 'frame': m['frame'], 'height': height, 'live': False,
    'drive': [[name, model.slices[name].stop - model.slices[name].start] for name in DRIVE_FIELDS if name in model.slices],
    'seg': {k: m['seg'][k] for k in ('labels', 'road_class', 'things', 'ds')},
    'source': {'route': route.segments[0].name.rsplit('--', 1)[0], 'start': start, 'rpy': list(map(float, rpy)),
               'camera': list(route.size)},
  }


def footage_encoder(path: Path, size: tuple[int, int], width: int = 960) -> subprocess.Popen:
  w, h = size
  return subprocess.Popen(
    ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'yuv420p', '-s', f'{w}x{h}', '-r', str(FPS), '-i', '-',
     '-vf', f'scale={width}:-2', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '26', '-pix_fmt', 'yuv420p',
     '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE)


def run(paths: list[Path], onnx: str, models_dir: Path, out: Path, start_s: float = 0.0, seconds: float | None = None,
        device: str = 'auto', rpy: list[float] | None = None, calib_s: float = 60.0, seg_every: int = 2,
        rhd: bool = False) -> Path:
  route = Route.open(paths)
  print(f'{len(route.segments)} segments of {route.segments[0].name.rsplit("--", 1)[0]}, camera {route.size[0]}x{route.size[1]}')
  model = DrivingModel(onnx, device, rhd)
  models = Models(models_dir, device)
  if rpy is not None:
    rpy, height = np.asarray(rpy, float), CAMERA_HEIGHT
  else:
    rpy, height = calibrate(route, model, int(calib_s * FPS))
  start = int(round(start_s * FPS))
  count = None if seconds is None else int(round(seconds * FPS))
  first = max(start - WARMUP, 0)

  out.mkdir(parents=True, exist_ok=True)
  enc = footage_encoder(out / 'footage.mp4', route.size)
  model.reset()
  timing = {'driving': 0.0, 'seg': 0.0}
  n = 0
  t0 = time.perf_counter()
  with open(out / 'drive.diorama', 'wb') as fh:
    packets.write(fh, packets.meta(drive_meta(model, models, height, rpy, route, start)))
    for k, narrow_yuv, warped in route.warped(rpy, first, None if count is None else start - first + count):
      t = time.perf_counter()
      vector = model.step(warped)
      model_s = time.perf_counter() - t
      timing['driving'] += model_s
      if k < start:
        continue
      frame = k - start
      enc.stdin.write(narrow_yuv.tobytes())
      values = np.concatenate([vector[model.slices[name]] for name in DRIVE_FIELDS if name in model.slices])
      packets.write(fh, packets.drive(frame, frame / FPS, 1000 * model_s, values))
      if frame % seg_every == 0:
        t = time.perf_counter()
        qmap, label, score = models.segment(warped[0])
        seg_s = time.perf_counter() - t
        timing['seg'] += seg_s
        packets.write(fh, packets.seg(frame, 1000 * seg_s, qmap, label, score))
      n += 1
      if n % 200 == 0:
        print(f'{n / FPS:.0f} s of drive in {time.perf_counter() - t0:.0f} s', flush=True)
  enc.stdin.close()
  enc.wait()
  if not n:
    raise SystemExit(f'the drive has no frames from {start_s} s on')
  print(f'{n} frames ({n / FPS:.1f} s): driving model {1000 * timing["driving"] / n:.1f} ms a frame, '
        f'seg {1000 * timing["seg"] / max(n // seg_every, 1):.1f} ms')
  print(f'wrote {out}/drive.diorama and footage.mp4; play it with: python -m diorama view {out}')
  return out

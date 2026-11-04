"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

A recorded drive through the driving model and Diorama's models, into a
folder the Diorama page plays beside the drive's own footage:

    drive.diorama   the messages a live server would have streamed (packets.py)
    footage.mp4     the road camera over the same frames, for the side-by-side

The frames are what a comma would send a jetlink server, warped and packed the
same way with the calibration the comma logged, and the models see what they
would see live, so a drive tried here looks as it would from the car. Live,
the server segments the narrow frame in the road view and the wide frame in
bird's-eye; here both are segmented, and the page takes the one for its view.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np

from diorama import packets
from diorama.driving import DrivingModel, providers
from diorama.outputs import DRIVE_FIELDS
from diorama.route import FPS, Route

WARMUP = 100  # frames of history the driving model gets before the first one kept


class Models:
  """Diorama's segmentation on onnxruntime, as a server runs it: the packed narrow frame in as float16."""

  def __init__(self, folder: Path, device: str):
    import onnxruntime as ort
    self.meta = json.loads((folder / 'diorama.json').read_text())
    prov = providers(device)
    self.seg = ort.InferenceSession(str(folder / self.meta['seg']['file']), providers=prov)

  def segment(self, frame: np.ndarray):
    """One packed frame, narrow or wide."""
    return self.seg.run(None, {'frame': frame[None].astype(np.float16)})


def drive_meta(model: DrivingModel, models: Models, route: Route, start: int) -> dict:
  m, cal = models.meta, route.calibration
  return {
    'version': 1, 'fps': FPS, 'frame': m['frame'], 'wide': dict(packets.wide_frame(m['frame']), from_device=list(cal.wide_from_device)),
    'height': cal.height, 'live': False,
    'drive': [[name, model.slices[name].stop - model.slices[name].start] for name in DRIVE_FIELDS if name in model.slices],
    'seg': {k: m['seg'][k] for k in ('labels', 'road_class', 'things', 'ds')},
    'source': {'route': route.segments[0].name.rsplit('--', 1)[0], 'start': start, 'rpy': list(cal.rpy), 'camera': list(route.size)},
  }


def footage_encoder(path: Path, size: tuple[int, int], width: int = 960) -> subprocess.Popen:
  w, h = size
  return subprocess.Popen(
    ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'yuv420p', '-s', f'{w}x{h}', '-r', str(FPS), '-i', '-',
     '-vf', f'scale={width}:-2', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '26', '-pix_fmt', 'yuv420p',
     '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE)


def run(paths: list[Path], onnx: str, models_dir: Path, out: Path, start_s: float = 0.0, seconds: float | None = None,
        device: str = 'auto', seg_every: int = 2, rhd: bool = False) -> Path:
  route = Route.open(paths)  # with the calibration the comma logged, or refused
  print(f'{len(route.segments)} segments of {route.segments[0].name.rsplit("--", 1)[0]}, camera {route.size[0]}x{route.size[1]}')
  model = DrivingModel(onnx, device, rhd)
  models = Models(models_dir, device)
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
    packets.write(fh, packets.meta(drive_meta(model, models, route, start)))
    for k, narrow_yuv, warped in route.warped(first, None if count is None else start - first + count):
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
        for camera in (0, 1):  # narrow, wide
          t = time.perf_counter()
          qmap, label, score = models.segment(warped[camera])
          seg_s = time.perf_counter() - t
          timing['seg'] += seg_s / 2
          packets.write(fh, packets.seg(frame, 1000 * seg_s, qmap, label, score, wide=camera == 1))
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

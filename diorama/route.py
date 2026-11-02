"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

A recorded drive as the comma's modeld would see it: the segments' road and
wide camera files decoded in step, and each frame warped to the driving
model's two 512x256 frames and packed into the six planes an INFER_REQ
carries. The warp is openpilot's (common/transformations), with the camera's
calibration as given (0 0 0, the camera as mounted, by default), so a route
needs only its video files.
"""
from __future__ import annotations

import re
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FPS = 20
# Focal lengths by sensor resolution (openpilot's DEVICE_CAMERAS): comma 3X's
# AR0231 and OX03C10, and comma 4's OS04C10 binned.
FOCALS = {(1928, 1208): (2648.0, 567.0), (1344, 760): (1141.5, 425.25)}

view_frame_from_device_frame = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
MEDMODEL_FL, MEDMODEL_CY = 910.0, 47.6
SBIGMODEL_FL, SBIGMODEL_CY = 455.0, 0.5 * (256 + 47.6)


def rot_from_euler(rpy) -> np.ndarray:
  r, p, y = rpy
  cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
  rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
  ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
  rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
  return rz @ ry @ rx


def _model_frame(fl, cy):
  return np.array([[fl, 0.0, 256.0], [0.0, fl, cy], [0.0, 0.0, 1.0]]) @ view_frame_from_device_frame


def warp_matrix(rpy, fl: float, size: tuple[int, int], big: bool) -> np.ndarray:
  """Model-frame pixel -> camera pixel."""
  w, h = size
  K = np.array([[fl, 0.0, w / 2], [0.0, fl, h / 2], [0.0, 0.0, 1.0]])
  calib_from_model = np.linalg.inv(_model_frame(SBIGMODEL_FL, SBIGMODEL_CY) if big else _model_frame(MEDMODEL_FL, MEDMODEL_CY))
  return K @ view_frame_from_device_frame @ rot_from_euler(rpy) @ calib_from_model


def pack(yuv: np.ndarray, M: np.ndarray, size: tuple[int, int]) -> np.ndarray:
  """modeld's warp: nearest neighbour to 512x256, packed into (6, 128, 256): four Y planes, U, V."""
  import cv2
  w, h = size
  y = yuv[:w * h].reshape(h, w)
  u = yuv[w * h:w * h * 5 // 4].reshape(h // 2, w // 2)
  v = yuv[w * h * 5 // 4:].reshape(h // 2, w // 2)
  flags = cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP
  wy = cv2.warpPerspective(y, M, (512, 256), flags=flags, borderMode=cv2.BORDER_REPLICATE)
  M_uv = np.diag([0.5, 0.5, 1.0]) @ M @ np.diag([2.0, 2.0, 1.0])
  wu = cv2.warpPerspective(u, M_uv, (256, 128), flags=flags, borderMode=cv2.BORDER_REPLICATE)
  wv = cv2.warpPerspective(v, M_uv, (256, 128), flags=flags, borderMode=cv2.BORDER_REPLICATE)
  return np.stack([wy[0::2, 0::2], wy[1::2, 0::2], wy[0::2, 1::2], wy[1::2, 1::2], wu, wv])


def probe_size(path: Path) -> tuple[int, int]:
  out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'csv=p=0', str(path)],
                       capture_output=True, text=True, check=True).stdout.strip()
  w, h = (int(v) for v in out.split(',')[:2])
  return w, h


def read_yuv(path: Path, size: tuple[int, int]) -> Iterator[np.ndarray]:
  w, h = size
  p = subprocess.Popen(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'rawvideo', '-pix_fmt', 'yuv420p', '-'], stdout=subprocess.PIPE)
  n = w * h * 3 // 2
  try:
    while True:
      buf = p.stdout.read(n)
      if len(buf) < n:
        return
      yield np.frombuffer(buf, np.uint8)
  finally:
    p.kill()
    p.wait()


@dataclass
class Route:
  """The segments of one drive, in order."""
  segments: list[Path]
  size: tuple[int, int]
  narrow_fl: float
  wide_fl: float

  @classmethod
  def open(cls, paths: list[Path]) -> Route:
    """Segment directories, or a directory of them (a comma's /data/media/0/realdata, or a copy of one route)."""
    segs: list[Path] = []
    for p in paths:
      if (p / 'fcamera.hevc').exists():
        segs.append(p)
      else:
        segs.extend(d for d in p.iterdir() if (d / 'fcamera.hevc').exists())
    if not segs:
      raise SystemExit(f"no segments (directories with fcamera.hevc and ecamera.hevc) in {', '.join(map(str, paths))}")
    routes = {re.sub(r'--\d+$', '', s.name) for s in segs}
    if len(routes) > 1:
      raise SystemExit(f"those segments are from {len(routes)} drives ({', '.join(sorted(routes))}); pick one")
    segs.sort(key=lambda s: int(m.group(1)) if (m := re.search(r'--(\d+)$', s.name)) else 0)
    for s in segs:
      if not (s / 'ecamera.hevc').exists():
        raise SystemExit(f"{s} has no ecamera.hevc: the driving model needs the wide camera too")
    size = probe_size(segs[0] / 'fcamera.hevc')
    if size not in FOCALS:
      raise SystemExit(f"unknown camera {size[0]}x{size[1]}; known: {', '.join(f'{w}x{h}' for w, h in FOCALS)}")
    return cls(segs, size, *FOCALS[size])

  def frames(self, start: int = 0, count: int | None = None) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """(frame index from the route's start, narrow yuv, wide yuv)."""
    k = 0
    for seg in self.segments:
      for f, e in zip(read_yuv(seg / 'fcamera.hevc', self.size), read_yuv(seg / 'ecamera.hevc', self.size), strict=False):
        if count is not None and k >= start + count:
          return
        if k >= start:
          yield k, f, e
        k += 1

  def warped(self, rpy, start: int = 0, count: int | None = None) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """(frame index, narrow yuv, (2, 6, 128, 256) warped narrow and wide) as an INFER_REQ carries them."""
    Mn = warp_matrix(rpy, self.narrow_fl, self.size, big=False)
    Mw = warp_matrix(rpy, self.wide_fl, self.size, big=True)
    for k, f, e in self.frames(start, count):
      yield k, f, np.stack([pack(f, Mn, self.size), pack(e, Mw, self.size)])

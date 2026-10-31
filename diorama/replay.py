"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

A recorded drive played into a running jetlink server over TCP, frame by
frame at 20 Hz, as a comma would send it: the live Diorama page on that
server then draws it, with no car. The server needs --listen, --status-port
and --diorama.

    jetlink-server --listen --status-port 8080 --diorama ~/diorama-models
    python -m diorama replay <segments> --model big.onnx --host 127.0.0.1
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from jetlink.client import JetlinkClient
from jetlink.spec import sha256_file

from diorama.driving import ACTION_T, DrivingModel
from diorama.offline import calibrate
from diorama.route import FPS, Route


def packed_scalars(spec, rhd: bool = False) -> np.ndarray:
  """desire none, the traffic side, openpilot's default action times."""
  packed = np.zeros(spec.packed_nelem, np.float32)
  for name, (at, _) in spec.packed_layout.items():
    if name == 'traffic_convention':
      packed[at] = [0.0, 1.0] if rhd else [1.0, 0.0]
    elif name == 'action_t':
      packed[at] = ACTION_T[:packed[at].size]
  return packed


def replay(paths: list[Path], onnx: str, host: str, port: int, start_s: float = 0.0, seconds: float | None = None,
           rpy: list[float] | None = None, calib_s: float = 60.0, device: str = 'auto', rhd: bool = False, loop: bool = False) -> None:
  route = Route.open(paths)
  if rpy is None:
    print('estimating the calibration with the model here first (pass --rpy to skip)')
    rpy, _ = calibrate(route, DrivingModel(onnx, device, rhd), int(calib_s * FPS))
  rpy = np.asarray(rpy, float)
  client = JetlinkClient.open_tcp(host, port)
  hello = client.hello()
  print(f"server: {hello.get('backend')} {hello.get('runtime_version')} on {hello.get('device')}")
  sha256, nbytes = sha256_file(onnx)
  spec = client.ensure_engine(sha256, nbytes, onnx_path=onnx, progress=lambda s, f, m: print(f'  {s:<7} {f * 100:5.1f}%  {m}'))
  packed = packed_scalars(spec, rhd)
  start = int(round(start_s * FPS))
  count = None if seconds is None else int(round(seconds * FPS))
  lat: list[float] = []
  try:
    while True:
      next_t = time.perf_counter()
      for i, (_k, _f, warped) in enumerate(route.warped(rpy, start, count)):
        now = time.perf_counter()
        if next_t > now:
          time.sleep(next_t - now)
        next_t += 1.0 / FPS
        t = time.perf_counter()
        client.infer(warped, packed, frame_id=i, reset=i == 0)
        lat.append(1000 * (time.perf_counter() - t))
        if len(lat) == 200:
          a = np.array(lat)
          print(f'{(i + 1) / FPS:.0f} s: round trip p50 {np.percentile(a, 50):.1f} ms, p99 {np.percentile(a, 99):.1f} ms')
          lat.clear()
      if not loop:
        break
  finally:
    client.leave('stopped')
    client.close()

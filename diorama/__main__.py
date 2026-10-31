"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

    python -m diorama export --out ~/diorama-models
    python -m diorama run /data/media/0/realdata/<route>--* --model big.onnx --nets ~/diorama-models --out drive/
    python -m diorama view drive/
    python -m diorama replay /data/media/0/realdata/<route>--* --model big.onnx --host 127.0.0.1   # into a live server
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
  ap = argparse.ArgumentParser(prog='python -m diorama', description='Diorama on a recorded drive: export the networks, run a drive, play it.')
  sub = ap.add_subparsers(dest='cmd', required=True)

  e = sub.add_parser('export', help="export Diorama's networks to ONNX (needs torch and transformers)")
  e.add_argument('--out', type=Path, required=True, help='folder for diorama_seg.onnx, diorama_dd3d.onnx and diorama.json')
  e.add_argument('--dd3d-weights', help='DD3D V2-99 KITTI checkpoint (downloaded when omitted)')
  e.add_argument('--no-dd3d', action='store_true', help='segmentation only: cars face along their lane')

  r = sub.add_parser('run', help='run a recorded drive through the driving model and the networks')
  r.add_argument('segments', type=Path, nargs='+', help="the drive's segment folders (fcamera.hevc, ecamera.hevc), or a folder of them")
  r.add_argument('--model', required=True, help='the driving model ONNX, as jetlink serves it')
  r.add_argument('--nets', type=Path, required=True, help='the folder `export` wrote')
  r.add_argument('--out', type=Path, required=True)
  r.add_argument('--start', type=float, default=0.0, help='seconds into the drive')
  r.add_argument('--seconds', type=float, help='how much of it (default: to the end)')
  r.add_argument('--device', choices=('auto', 'cuda', 'coreml', 'cpu'), default='auto')
  r.add_argument('--rpy', type=float, nargs=3, metavar=('ROLL', 'PITCH', 'YAW'), help='calibration in radians, instead of estimating it')
  r.add_argument('--calib-seconds', type=float, default=60.0, help='how much of the drive the calibration estimate reads')
  r.add_argument('--rhd', action='store_true', help='left-hand traffic (the driver sits on the right)')

  p = sub.add_parser('replay', help="play a recorded drive into a running jetlink server, as a comma would, for its live page")
  p.add_argument('segments', type=Path, nargs='+')
  p.add_argument('--model', required=True, help='the driving model ONNX; uploaded to the server if it lacks it')
  p.add_argument('--host', default='127.0.0.1')
  p.add_argument('--port', type=int, default=5599)
  p.add_argument('--start', type=float, default=0.0)
  p.add_argument('--seconds', type=float)
  p.add_argument('--rpy', type=float, nargs=3, metavar=('ROLL', 'PITCH', 'YAW'))
  p.add_argument('--calib-seconds', type=float, default=60.0)
  p.add_argument('--device', choices=('auto', 'cuda', 'coreml', 'cpu'), default='auto', help='for the calibration estimate')
  p.add_argument('--rhd', action='store_true')
  p.add_argument('--loop', action='store_true', help='start over at the end')

  v = sub.add_parser('view', help='play a drive `run` wrote, in the browser')
  v.add_argument('drive', type=Path)
  v.add_argument('--port', type=int, default=8642)
  v.add_argument('--no-browser', action='store_true')

  args = ap.parse_args(argv)
  if args.cmd == 'export':
    from diorama.export import DD3D_WEIGHTS_URL, export
    weights = None if args.no_dd3d else args.dd3d_weights
    if not args.no_dd3d and weights is None:
      import torch
      weights = torch.hub.load_state_dict_from_url(DD3D_WEIGHTS_URL, map_location='cpu', progress=True, file_name='dd3d_kitti_v99.pth')
      weights = str(Path(torch.hub.get_dir()) / 'checkpoints' / 'dd3d_kitti_v99.pth')
    export(args.out, weights)
    print(f'wrote {args.out}')
  elif args.cmd == 'run':
    from diorama.offline import run
    run(args.segments, args.model, args.nets, args.out, args.start, args.seconds, args.device, args.rpy, args.calib_seconds, rhd=args.rhd)
  elif args.cmd == 'replay':
    from diorama.replay import replay
    replay(args.segments, args.model, args.host, args.port, args.start, args.seconds, args.rpy, args.calib_seconds, args.device,
           args.rhd, args.loop)
  else:
    from diorama.viewer import serve
    serve(args.drive, args.port, open_browser=not args.no_browser)
  return 0


if __name__ == '__main__':
  sys.exit(main())

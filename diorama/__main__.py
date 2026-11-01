"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

    python -m diorama export --out ~/diorama-models
    python -m diorama run /data/media/0/realdata/<route>--* --model big.onnx --models ~/diorama-models --out drive/
    python -m diorama view drive/
    python -m diorama replay /data/media/0/realdata/<route>--* --model big.onnx --host 127.0.0.1   # into a live server
    python -m diorama assets   # rebuild the page's car (diorama/assets.py)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
  ap = argparse.ArgumentParser(prog='python -m diorama', description='Diorama on a recorded drive: export the models, run a drive, play it.')
  sub = ap.add_subparsers(dest='cmd', required=True)

  e = sub.add_parser('export', help="export Diorama's models to ONNX (needs torch and transformers)")
  e.add_argument('--out', type=Path, required=True, help='folder for diorama_seg.onnx and diorama.json')
  e.add_argument('--coreml', action='store_true', help="also native CoreML packages, which the Mac app runs (needs coremltools)")

  r = sub.add_parser('run', help="run a recorded drive through the driving model and Diorama's models")
  r.add_argument('segments', type=Path, nargs='+', help="the drive's segment folders (fcamera.hevc, ecamera.hevc), or a folder of them")
  r.add_argument('--model', required=True, help='the driving model ONNX, as jetlink serves it')
  r.add_argument('--models', '--nets', dest='models', type=Path, required=True, help='the folder `export` wrote')
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

  a = sub.add_parser('assets', help="rebuild the page's car, diorama-assets.bin, after changing diorama/assets.py")
  a.add_argument('--out', type=Path, help='default: the page resources in JetlinkKit')

  args = ap.parse_args(argv)
  if args.cmd == 'export':
    from diorama.export import export
    export(args.out, coreml=args.coreml)
    print(f'wrote {args.out}')
  elif args.cmd == 'run':
    from diorama.offline import run
    run(args.segments, args.model, args.models, args.out, args.start, args.seconds, args.device, args.rpy, args.calib_seconds, rhd=args.rhd)
  elif args.cmd == 'replay':
    from diorama.replay import replay
    replay(args.segments, args.model, args.host, args.port, args.start, args.seconds, args.rpy, args.calib_seconds, args.device,
           args.rhd, args.loop)
  elif args.cmd == 'assets':
    from diorama.assets import RESOURCES, write_library
    out = args.out or RESOURCES / 'diorama-assets.bin'
    print(f'wrote {out}, {write_library(out)} bytes')
  else:
    from diorama.viewer import serve
    serve(args.drive, args.port, open_browser=not args.no_browser)
  return 0


if __name__ == '__main__':
  sys.exit(main())

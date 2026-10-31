"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Plays a drive `python -m diorama run` wrote: a local web server for the
Diorama page (the same files jetlink-server serves live) and the drive's
folder, on 127.0.0.1 only.
"""
from __future__ import annotations

import functools
import http.server
import webbrowser
from pathlib import Path

RESOURCES = Path(__file__).resolve().parent.parent / 'JetlinkKit' / 'Sources' / 'JetlinkDiorama' / 'Resources'
PAGE_FILES = {'diorama.html', 'diorama-engine.js', 'diorama-render.js', 'diorama-assets.bin'}
DRIVE_FILES = {'drive.diorama', 'footage.mp4'}


class Handler(http.server.SimpleHTTPRequestHandler):
  def __init__(self, *args, drive: Path, **kwargs):
    self.drive = drive
    super().__init__(*args, directory=str(RESOURCES), **kwargs)

  def translate_path(self, path: str) -> str:
    name = path.split('?', 1)[0].lstrip('/')
    if name in DRIVE_FILES:
      return str(self.drive / name)
    if name in PAGE_FILES:
      return str(RESOURCES / name)
    return str(RESOURCES / '__missing__')

  def do_GET(self):
    if self.path in ('/', '/index.html'):
      footage = '&footage=footage.mp4' if (self.drive / 'footage.mp4').exists() else ''
      self.send_response(302)
      self.send_header('Location', f'/diorama.html?drive=drive.diorama{footage}')
      self.end_headers()
      return
    rng = self.headers.get('Range', '')
    if rng.startswith('bytes='):
      self.send_range(Path(self.translate_path(self.path)), rng[6:])
      return
    super().do_GET()

  def send_range(self, path: Path, spec: str):
    """One byte range: a browser seeks in the footage with these."""
    if not path.is_file():
      self.send_error(404)
      return
    size = path.stat().st_size
    first, _, last = spec.split(',', maxsplit=1)[0].strip().partition('-')
    start = int(first) if first else max(size - int(last), 0)
    end = min(int(last), size - 1) if first and last else size - 1
    if start >= size or start > end:
      self.send_response(416)
      self.send_header('Content-Range', f'bytes */{size}')
      self.end_headers()
      return
    self.send_response(206)
    self.send_header('Content-Type', self.guess_type(str(path)))
    self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
    self.send_header('Content-Length', str(end - start + 1))
    self.send_header('Accept-Ranges', 'bytes')
    self.end_headers()
    with open(path, 'rb') as fh:
      fh.seek(start)
      left = end - start + 1
      while left > 0:
        chunk = fh.read(min(left, 1 << 16))
        if not chunk:
          break
        self.wfile.write(chunk)
        left -= len(chunk)

  def end_headers(self):
    self.send_header('Cache-Control', 'no-store')
    super().end_headers()

  def log_message(self, *args):
    pass


def serve(drive: Path, port: int = 8642, open_browser: bool = True) -> None:
  drive = drive.resolve()
  if not (drive / 'drive.diorama').exists():
    raise SystemExit(f'{drive} has no drive.diorama; write one with: python -m diorama run ... --out {drive}')
  server = http.server.ThreadingHTTPServer(('127.0.0.1', port), functools.partial(Handler, drive=drive))
  url = f'http://127.0.0.1:{server.server_address[1]}/'
  print(f'Diorama: {url}  (Ctrl-C to stop)')
  if open_browser:
    webbrowser.open(url)
  try:
    server.serve_forever()
  except KeyboardInterrupt:
    pass

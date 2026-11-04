"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Plays a drive `python -m diorama run` wrote: a local web server for the
Diorama page (the same files jetlink-server serves live) and the drive's
folder, on 127.0.0.1 only. The camera file, when there is one, is the page's
diorama-camera.json; `python -m diorama camera` opens the page to set it over a
still scene of its own, no drive needed, and saves what the page posts back.
"""
from __future__ import annotations

import functools
import http.server
import json
import webbrowser
from pathlib import Path

from diorama import camera as cam

RESOURCES = Path(__file__).resolve().parent.parent / 'JetlinkKit' / 'Sources' / 'JetlinkDiorama' / 'Resources'
PAGE_FILES = {'diorama.html', 'diorama-engine.js', 'diorama-render.js', 'diorama-assets.bin'}
DRIVE_FILES = {'drive.diorama', 'footage.mp4'}
CAMERA = 'diorama-camera.json'


class Handler(http.server.SimpleHTTPRequestHandler):
  def __init__(self, *args, drive: Path | None, camera: Path | None = None, setup: bool = False, **kwargs):
    self.drive = drive
    self.camera = camera
    self.setting = setup
    super().__init__(*args, directory=str(RESOURCES), **kwargs)

  def translate_path(self, path: str) -> str:
    name = path.split('?', 1)[0].lstrip('/')
    if name in DRIVE_FILES and self.drive:
      return str(self.drive / name)
    if name == CAMERA and self.camera:
      return str(self.camera)
    if name in PAGE_FILES:
      return str(RESOURCES / name)
    return str(RESOURCES / '__missing__')

  def do_GET(self):
    if self.path in ('/', '/index.html'):
      if self.setting:
        page = '/diorama.html?camera=setup'
      else:
        footage = '&footage=footage.mp4' if (self.drive / 'footage.mp4').exists() else ''
        page = f'/diorama.html?drive=drive.diorama{footage}'
      self.send_response(302)
      self.send_header('Location', page)
      self.end_headers()
      return
    rng = self.headers.get('Range', '')
    if rng.startswith('bytes='):
      self.send_range(Path(self.translate_path(self.path)), rng[6:])
      return
    super().do_GET()

  def do_POST(self):
    """The setup page's camera, saved; JSON only, which a page elsewhere cannot post without asking first."""
    if not self.setting or self.path.split('?', 1)[0] != f'/{CAMERA}':
      self.send_error(404)
      return
    if self.headers.get('Content-Type', '').split(';')[0].strip() != 'application/json':
      self.reply(415, 'the camera comes as application/json')
      return
    try:
      length = int(self.headers.get('Content-Length', ''))
      if not 0 < length <= 1 << 16:
        raise ValueError(f'{length} bytes is no camera')
      cam.save(self.camera, json.loads(self.rfile.read(length)))
    except ValueError as e:  # JSONDecodeError is one
      self.reply(400, f'{e}')
      return
    except OSError as e:
      self.reply(500, f'cannot write {self.camera}: {e.strerror}')
      return
    self.reply(200, str(self.camera))

  def reply(self, status: int, text: str):
    body = (text + '\n').encode()
    self.send_response(status)
    self.send_header('Content-Type', 'text/plain; charset=utf-8')
    self.send_header('Content-Length', str(len(body)))
    self.end_headers()
    self.wfile.write(body)

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


def serve(drive: Path | None, port: int = 8642, open_browser: bool = True, camera: Path | None = None, setup: bool = False) -> None:
  """The drive's page, with `camera` as its camera file; `setup`, with no drive, opens the page to set the camera
  over its still scene, and saves it there."""
  if drive:
    drive = drive.resolve()
    if not (drive / 'drive.diorama').exists():
      raise SystemExit(f'{drive} has no drive.diorama; write one with: python -m diorama run ... --out {drive}')
  try:
    if camera:
      cam.load(camera)
  except ValueError as e:
    if not setup:
      raise SystemExit(f'{e}\nfix it, or set it again with: python -m diorama camera') from None
    print(f'{e}; the page starts from its own camera, and Save writes over it')
  server = http.server.ThreadingHTTPServer(('127.0.0.1', port), functools.partial(Handler, drive=drive, camera=camera, setup=setup))
  url = f'http://127.0.0.1:{server.server_address[1]}/'
  print(f'Diorama: {url}  (Ctrl-C to stop)')
  if setup:
    print(f'Set each view, then Save: it goes to {camera}')
  if open_browser:
    webbrowser.open(url)
  try:
    server.serve_forever()
  except KeyboardInterrupt:
    pass

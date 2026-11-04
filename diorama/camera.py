"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The page's camera file, diorama-camera.json: each view's zoom, pitch, forward, rise and shift,
which `python -m diorama camera` sets in the browser and the page reads over
its own defaults. diorama-render.js says what the numbers mean and holds the
defaults. There are no limits: any finite numbers, but a zoom above 0.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

VIEWS = ('road', 'bird')
NUMBERS = ('zoom', 'pitch', 'forward', 'rise', 'shift')


def default_path() -> Path:
  """${XDG_CONFIG_HOME:-~/.config}/jetlink/diorama-camera.json."""
  base = os.environ.get('XDG_CONFIG_HOME') or str(Path.home() / '.config')
  return Path(base) / 'jetlink' / 'diorama-camera.json'


def check(camera: object) -> dict:
  """`camera` if it is a camera file's contents, else ValueError saying what is wrong."""
  if not isinstance(camera, dict):
    raise ValueError('it is not a JSON object')
  for view, nums in camera.items():
    if view not in VIEWS:
      raise ValueError(f'"{view}" is not a view: road or bird')
    if not isinstance(nums, dict):
      raise ValueError(f'{view} is not an object')
    for k, v in nums.items():
      if k not in NUMBERS:
        raise ValueError(f'{view}.{k} is not a camera number: zoom, pitch, forward, rise or shift')
      if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f'{view}.{k} is {json.dumps(v)}, not a number')
      if k == 'zoom' and not v > 0:
        raise ValueError(f'{view}.zoom is {v:g}, not above 0')
  return camera


def load(path: Path) -> dict | None:
  """The file's camera, None when there is no file, or ValueError naming the file."""
  try:
    text = path.read_text()
  except FileNotFoundError:
    return None
  try:
    return check(json.loads(text))
  except ValueError as e:  # JSONDecodeError is one
    raise ValueError(f'{path}: {e}') from None


def save(path: Path, camera: object) -> None:
  """Checks `camera` and writes it whole: a reader never sees half a file."""
  text = json.dumps(check(camera), indent=2) + '\n'
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_name(path.name + '.tmp')
  tmp.write_text(text)
  tmp.replace(path)

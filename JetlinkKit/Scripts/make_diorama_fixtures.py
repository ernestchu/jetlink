#!/usr/bin/env python3
"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Writes JetlinkDioramaTests' fixture: diorama/packets.py's bytes for fixed
inputs, which DioramaPackets.swift must reproduce exactly.

    python3 JetlinkKit/Scripts/make_diorama_fixtures.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from diorama import packets  # noqa: E402


def inputs():
  """The same numbers DioramaPacketsTests builds."""
  values = np.array([0.5, -1.25, 3.0, 1e-3, 250.0], np.float32)
  qmap = np.full((3, 400), 255.0, np.float32)
  qmap[1, 10:20] = 7
  qmap[2, :] = 3  # a run longer than 256
  label = np.array([12, 40, 7, 3, 0], np.float32)
  score = np.array([0.0, 0.5, 0.9, 1.0, 0.25], np.float32)
  rows = np.arange(3 * 16, dtype=np.float32).reshape(3, 16) / 10
  rows[:, 6] = [0.05, 0.5, 0.1]
  return values, qmap, label, score, rows


def main():
  values, qmap, label, score, rows = inputs()
  out = ROOT / 'JetlinkKit/Tests/JetlinkDioramaTests/Fixtures/packets.bin'
  with open(out, 'wb') as fh:
    packets.write(fh, packets.drive(7, 0.35, 3.5, values))
    packets.write(fh, packets.seg(8, 21.5, qmap, label, score))
    packets.write(fh, packets.boxes(8, 12.25, rows))
  print(f'wrote {out}')


if __name__ == '__main__':
  main()

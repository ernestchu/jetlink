"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Diorama's messages, as the server streams them to a page and a recorded drive
stores them: one byte of kind, then its payload, little-endian throughout.
JetlinkDiorama's DioramaPackets.swift writes the same bytes and diorama.js
reads them; docs/diorama.md has the table.

    M  meta    JSON: the frame's camera, the drive fields and their float counts, the networks' classes
    D  drive   u32 frame, f32 seconds, f32 model ms, f32[] the drive fields' raw outputs, in meta's order
    S  seg     u32 frame, f32 ms, u16 width, u16 height, u16 queries, u8[queries] label (255: dropped),
               u8[queries] score * 255, u32 runs, runs x (u8 query, u8 length - 1): the query map, row by row
    B  boxes   u32 frame, f32 ms, u16 rows, f32[rows * 16] DD3D rows (export.DD3D_COLUMNS) with score3d >= BOX_MIN_SCORE

`frame` counts the comma's frames (20 a second); a seg or boxes message names
the frame its image came from. The milliseconds are how long the driving model
took for that frame, and the network for that image.

A file is the messages one after another, each after its u32 length.
"""
from __future__ import annotations

import json
import struct

import numpy as np

META, DRIVE, SEG, BOXES = b'M', b'D', b'S', b'B'
BOX_MIN_SCORE = 0.1


def meta(info: dict) -> bytes:
  return META + json.dumps(info, separators=(',', ':')).encode()


def drive(frame: int, seconds: float, model_ms: float, values: np.ndarray) -> bytes:
  return DRIVE + struct.pack('<Iff', frame, seconds, model_ms) + np.ascontiguousarray(values, '<f4').tobytes()


def rle(qmap: np.ndarray) -> bytes:
  """Runs of equal bytes, row-major, each at most 256 long."""
  flat = qmap.reshape(-1)
  edges = np.flatnonzero(np.diff(flat)) + 1
  starts = np.concatenate([[0], edges])
  lengths = np.diff(np.concatenate([starts, [flat.size]]))
  values, counts = [], []
  for v, n in zip(flat[starts], lengths, strict=True):
    while n > 0:
      take = min(int(n), 256)
      values.append(v)
      counts.append(take - 1)
      n -= take
  runs = np.empty(2 * len(values), np.uint8)
  runs[0::2], runs[1::2] = values, counts
  return struct.pack('<I', len(values)) + runs.tobytes()


def seg(frame: int, ms: float, qmap: np.ndarray, label: np.ndarray, score: np.ndarray) -> bytes:
  """The seg graph's float outputs as a message: the query map as bytes, and each query's class and score."""
  q = np.asarray(qmap, np.float32).reshape(qmap.shape[-2:])
  h, w = q.shape
  keep = np.asarray(score) > 0
  labels = np.where(keep, np.asarray(label), 255).astype(np.uint8)
  scores = np.round(np.clip(score, 0, 1) * 255).astype(np.uint8)
  head = struct.pack('<IfHHH', frame, ms, w, h, len(labels))
  return SEG + head + labels.tobytes() + scores.tobytes() + rle(q.astype(np.uint8))


def boxes(frame: int, ms: float, rows: np.ndarray) -> bytes:
  rows = np.asarray(rows, np.float32)
  rows = rows[rows[:, 6] >= BOX_MIN_SCORE]
  return BOXES + struct.pack('<IfH', frame, ms, len(rows)) + np.ascontiguousarray(rows, '<f4').tobytes()


def write(fh, message: bytes) -> None:
  fh.write(struct.pack('<I', len(message)))
  fh.write(message)


def read(data: bytes):
  """The messages of a file, in order."""
  at = 0
  while at + 4 <= len(data):
    (n,) = struct.unpack_from('<I', data, at)
    yield data[at + 4:at + 4 + n]
    at += 4 + n

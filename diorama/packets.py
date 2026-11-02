"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Diorama's messages, as the server streams them to a page and a recorded drive
stores them: one byte of kind, then its payload, little-endian throughout.
JetlinkDiorama's DioramaPackets.swift writes the same bytes and diorama.js
reads them; docs/diorama-internals.md has the table.

    M  meta    JSON: the frame's camera and the wide frame's (with the wide camera's angles from the device),
               the camera's height, the drive fields and their float counts, the models' classes
    D  drive   u32 frame, f32 seconds, f32 model ms, f32[] the drive fields' raw outputs, in meta's order
    S  seg     u32 frame, f32 ms, u16 width, u16 height, u16 queries, u8[queries] label (255: dropped),
               u8[queries] score * 255, u32 runs, runs x (u8 query, u8 length - 1): the query map, row by row
    W  seg     as S, of the wide frame: the page uses S in the road view and W in bird's-eye

`frame` counts the comma's frames (20 a second); a seg message names
the frame its image came from. The milliseconds are how long the driving model
took for that frame, and the model for that image.

A file is the messages one after another, each after its u32 length.
"""
from __future__ import annotations

import json
import struct

import numpy as np

META, DRIVE, SEG, WIDE_SEG = b'M', b'D', b'S', b'W'


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


def wide_frame(frame: dict) -> dict:
  """The wide frame as the comma warps it, openpilot's sbigmodel: half the narrow frame's focal length, its horizon
  halfway down from the narrow's. DioramaStage.swift makes the same of diorama.json's frame."""
  return dict(frame, f=frame['f'] / 2, cy=(frame['h'] + frame['cy']) / 2)


def seg(frame: int, ms: float, qmap: np.ndarray, label: np.ndarray, score: np.ndarray, wide: bool = False) -> bytes:
  """The seg graph's float outputs as a message: the query map as bytes, and each query's class and score; of the
  narrow frame, or the wide."""
  q = np.asarray(qmap, np.float32).reshape(qmap.shape[-2:])
  h, w = q.shape
  keep = np.asarray(score) > 0
  labels = np.where(keep, np.asarray(label), 255).astype(np.uint8)
  scores = np.round(np.clip(score, 0, 1) * 255).astype(np.uint8)
  head = struct.pack('<IfHHH', frame, ms, w, h, len(labels))
  return (WIDE_SEG if wide else SEG) + head + labels.tobytes() + scores.tobytes() + rle(q.astype(np.uint8))


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

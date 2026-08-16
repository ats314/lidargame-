"""Minimal PNG encoder (8-bit RGB / RGBA / grey), zlib only.

Baked textures have to be real image files -- glTF, a browser and every engine
importer all expect PNG -- but pulling in Pillow for the handful of buffers this
project writes is not worth the dependency. This is the whole spec surface we
need: IHDR, IDAT with filter 0, IEND.
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

_COLOR_TYPE = {1: 0, 2: 4, 3: 2, 4: 6}      # channels -> PNG colour type


def _chunk(tag: bytes, payload: bytes) -> bytes:
    return (struct.pack(">I", len(payload)) + tag + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))


def encode(image: np.ndarray, *, level: int = 6) -> bytes:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0 if arr.dtype.kind == "f" else arr, 0, 255).astype(np.uint8)
    height, width, channels = arr.shape
    if channels not in _COLOR_TYPE:
        raise ValueError(f"unsupported channel count {channels}")

    # Filter byte 0 (None) in front of every scanline.
    raw = np.concatenate(
        [np.zeros((height, 1), np.uint8), arr.reshape(height, width * channels)], axis=1)
    return b"".join([
        b"\x89PNG\r\n\x1a\n",
        _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, _COLOR_TYPE[channels], 0, 0, 0)),
        _chunk(b"IDAT", zlib.compress(raw.tobytes(), level)),
        _chunk(b"IEND", b""),
    ])


def write(path: str | Path, image: np.ndarray, **kwargs) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode(image, **kwargs))
    return path

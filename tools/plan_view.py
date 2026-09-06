#!/usr/bin/env python3
"""Draw a baked NOCTIS-7 city texture as a plan, so a bake can be checked.

The renderer is the real test, but it needs a browser and, without a GPU, four
seconds a frame. This needs neither: it decodes the texture's own planes and
paints them from above.

    python tools/plan_view.py build/ams/city.png -o build/ams/plan.png

It exists because the failure modes of a bake are geometric and invisible to
every metric. A transposed terrain plane, a mirrored grid, a footprint fill
that leaks, an axis swapped between the seed and the tile -- all of them report
perfectly good counts and none of them survive one look from above. The plan
that caught this one showed the Amstel bending through the canal belt, which is
the only evidence that says the numbers describe Amsterdam rather than
something Amsterdam-shaped.
"""
from __future__ import annotations

import argparse
import struct
import zlib
from pathlib import Path

import numpy as np

ROAD, PARK, WATER = 1, 8, 16

# Deliberately flat and unbranded: this is a diagnostic, not a render. Water
# reads first because a canal in the wrong place is the most obvious tell.
GROUND = (18, 18, 22)
CARRIAGEWAY = (70, 70, 78)
CANAL = (40, 90, 150)
CANOPY = (50, 110, 60)


def read_png(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a PNG")
    pos, idat, width, height = 8, b"", 0, 0
    while pos < len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        tag = data[pos + 4:pos + 8]
        if tag == b"IHDR":
            width, height, depth, colour = struct.unpack(">IIBB", data[pos + 8:pos + 18])
            if (depth, colour) != (8, 6):
                raise ValueError("expected 8-bit RGBA")
        elif tag == b"IDAT":
            idat += data[pos + 8:pos + 8 + length]
        pos += 12 + length
    raw = zlib.decompress(idat)
    stride = width * 4
    rows = []
    for y in range(height):
        row = raw[y * (stride + 1):(y + 1) * (stride + 1)]
        if row[0] != 0:
            raise ValueError("only filter type 0 is supported")
        rows.append(np.frombuffer(row[1:], dtype=np.uint8).reshape(width, 4))
    return np.stack(rows)


def write_png(rgba: np.ndarray, path: Path) -> int:
    height, width, _ = rgba.shape
    raw = bytearray()
    for row in rgba:
        raw.append(0)
        raw += row.tobytes()

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + tag + payload
                + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))

    blob = (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
            + chunk(b"IEND", b""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return len(blob)


def plan(planes: np.ndarray, *, scale: int = 2) -> np.ndarray:
    grid = planes.shape[1]
    if planes.shape[0] < grid * 2:
        raise ValueError("texture has no flag plane; is this a NOCTIS-7 bake?")
    height = planes[:grid, :, 0].astype(float)
    flags = planes[grid:grid * 2, :, 0]

    out = np.empty((grid, grid, 4), dtype=np.uint8)
    out[...] = (*GROUND, 255)
    out[(flags & ROAD).astype(bool)] = (*CARRIAGEWAY, 255)
    out[(flags & PARK).astype(bool)] = (*CANOPY, 255)
    out[(flags & WATER).astype(bool)] = (*CANAL, 255)

    built = height > 0
    if built.any():
        shade = np.clip(90 + height / height.max() * 165, 0, 255).astype(np.uint8)
        for channel in range(3):
            out[..., channel][built] = shade[built]

    # Row 0 of the texture is the *southern* edge, because the baker indexes
    # `[iy, ix]` from the tile's lower-left corner. Flip so north is up, or
    # every plan comes out mirrored against the map you compare it to.
    out = out[::-1]
    return np.repeat(np.repeat(out, scale, axis=0), scale, axis=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("texture", help="a city.png written by `lidarworld noctis`")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--scale", type=int, default=2)
    args = ap.parse_args()

    texture = Path(args.texture)
    planes = read_png(texture)
    image = plan(planes, scale=args.scale)
    out = Path(args.out) if args.out else texture.with_name(texture.stem + ".plan.png")
    size = write_png(image, out)

    grid = planes.shape[1]
    flags = planes[grid:grid * 2, :, 0]
    built = int((planes[:grid, :, 0] > 0).sum())
    print(f"{out}  {image.shape[1]}x{image.shape[0]}  {size / 1024:.0f} KB")
    print(f"  {built:,} built cells, "
          f"{int((flags & WATER).astype(bool).sum()):,} water, "
          f"{int((flags & ROAD).astype(bool).sum()):,} road, "
          f"{int((flags & PARK).astype(bool).sum()):,} canopy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

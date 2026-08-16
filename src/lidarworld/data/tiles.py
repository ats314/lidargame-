"""Lazy tile index over a directory of LiDAR tiles.

Until now `compile` took a list of files and loaded all of them. That is fine
for one tile and impossible for a city: Denver metro is thousands of 65 MB LAZ
tiles, and the question is never "load everything", it is "give me the points
inside this area".

A tile index answers that without opening a file it does not need. Headers are
cheap -- a LAS header carries the tile's bounds in its first few hundred bytes
-- so the index is built by reading headers only, cached to disk, and queried
spatially. Loading happens per tile, on demand.

`pointcloudset` (MIT) is used for orchestration when installed: it is built for
exactly this, handles datasets over time, and does lazy parallel evaluation.
The index works without it, because the header scan and the spatial query are
the parts that matter and neither needs a dependency.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


@dataclass
class TileRecord:
    path: str
    minx: float
    miny: float
    maxx: float
    maxy: float
    minz: float
    maxz: float
    points: int
    crs: str = ""

    @property
    def area(self) -> float:
        return (self.maxx - self.minx) * (self.maxy - self.miny)

    @property
    def density(self) -> float:
        return self.points / max(self.area, 1e-9)

    def intersects(self, bbox) -> bool:
        minx, miny, maxx, maxy = bbox
        return not (self.minx > maxx or self.maxx < minx
                    or self.miny > maxy or self.maxy < miny)


def read_header(path: Path) -> TileRecord | None:
    """Bounds and count from a LAS/LAZ header, without decoding any points."""
    try:
        with open(path, "rb") as fh:
            raw = fh.read(512)
        if raw[:4] != b"LASF":
            return None
        version = (raw[24], raw[25])
        header_size = struct.unpack_from("<H", raw, 94)[0]
        count = struct.unpack_from("<I", raw, 107)[0]
        if version >= (1, 4) and header_size >= 375:
            with open(path, "rb") as fh:
                fh.seek(247)
                big = struct.unpack("<Q", fh.read(8))[0]
            count = big or count
        maxx, minx, maxy, miny, maxz, minz = struct.unpack_from("<6d", raw, 179)
        return TileRecord(str(path), minx, miny, maxx, maxy, minz, maxz, int(count))
    except (OSError, struct.error):
        return None


class TileIndex:
    """Spatial index over tiles, built from headers and cached."""

    CACHE = "tile_index.json"

    def __init__(self, tiles: list[TileRecord], root: Path | None = None):
        self.tiles = tiles
        self.root = root

    def __len__(self) -> int:
        return len(self.tiles)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        if not self.tiles:
            return (0.0, 0.0, 0.0, 0.0)
        return (min(t.minx for t in self.tiles), min(t.miny for t in self.tiles),
                max(t.maxx for t in self.tiles), max(t.maxy for t in self.tiles))

    @property
    def total_points(self) -> int:
        return sum(t.points for t in self.tiles)

    @classmethod
    def build(cls, root: str | Path, *, patterns=("*.laz", "*.las"),
              use_cache: bool = True) -> "TileIndex":
        root = Path(root)
        cache = root / cls.CACHE
        if use_cache and cache.exists():
            try:
                data = json.loads(cache.read_text())
                return cls([TileRecord(**t) for t in data["tiles"]], root)
            except (json.JSONDecodeError, TypeError, KeyError):
                pass

        tiles = []
        for pattern in patterns:
            for path in sorted(root.rglob(pattern)):
                record = read_header(path)
                if record is not None:
                    tiles.append(record)
        index = cls(tiles, root)
        if use_cache and tiles:
            cache.write_text(json.dumps({"tiles": [asdict(t) for t in tiles]}, indent=1))
        return index

    def query(self, bbox) -> list[TileRecord]:
        """Tiles overlapping (minx, miny, maxx, maxy), densest first."""
        hits = [t for t in self.tiles if t.intersects(bbox)]
        return sorted(hits, key=lambda t: -t.density)

    def around(self, x: float, y: float, size: float) -> list[TileRecord]:
        half = size / 2
        return self.query((x - half, y - half, x + half, y + half))

    def coverage(self, bbox) -> float:
        """Fraction of `bbox` any tile covers -- catches gaps before compiling."""
        minx, miny, maxx, maxy = bbox
        hits = self.query(bbox)
        if not hits:
            return 0.0
        cells = 64
        gx = np.linspace(minx, maxx, cells)
        gy = np.linspace(miny, maxy, cells)
        covered = np.zeros((cells, cells), dtype=bool)
        for t in hits:
            covered |= ((gx[:, None] >= t.minx) & (gx[:, None] <= t.maxx)
                        & (gy[None, :] >= t.miny) & (gy[None, :] <= t.maxy))
        return float(covered.mean())

    def summary(self) -> dict:
        lo_x, lo_y, hi_x, hi_y = self.bounds
        return {
            "tiles": len(self.tiles),
            "points": self.total_points,
            "extent_km2": round((hi_x - lo_x) * (hi_y - lo_y) / 1e6, 2),
            "mean_density": round(float(np.mean([t.density for t in self.tiles])), 2)
                            if self.tiles else 0.0,
            "bounds": [lo_x, lo_y, hi_x, hi_y],
        }

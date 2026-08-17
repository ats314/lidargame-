"""Remote tile index: resolve an area of interest to download URLs.

`data/tiles.py` indexes tiles by reading their LAS headers, which is exact and
only works for tiles already on disk. That is the wrong way round for
acquisition: to decide what to download you need the extents *before* you have
the files.

The USGS acquisitions publish that. Each DRCOG 2020 block has a GeoJSON of
per-tile footprints carrying a direct rockyweb LAZ URL, so an area of interest
resolves to a download list without fetching a single point. Denver's LiDAR is
6,505 tiles across three blocks; a city block needs one or two of them, and
this is the difference between knowing which and guessing.

    remote index (extents + URLs)  ->  fetch the 1-2 tiles an AOI needs
    local index  (LAS headers)     ->  what is on disk, exactly

Both produce the same query surface, so `compile --area` does not care which
answered. Note the published extents are approximate tile bounds in WGS84,
while the header index is the tile's true extent in its own CRS -- the remote
index picks candidates, the local one is authoritative once they land.
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

USER_AGENT = "lidarworld/0.1 (+https://github.com/ats314/lidargame-)"

#: Published per-tile extents for the DRCOG 2020 acquisition, one per block.
#: Licence is "Other (Public Domain)" -- these are USGS 3DEP products.
ACQUISITIONS: dict[str, dict] = {
    "co_drcog_2020_b1": {
        "project": "CO_DRCOG_2020_B20/CO_DRCOG_1_2020",
        "extents": "https://wifire-data.sdsc.edu/dataset/095c357b-fdec-4a6e-968d-a56786f9ce56"
                   "/resource/7d3a91e3-eafd-4e24-9399-c85f7e53781e/download/"
                   "spatial_extents_co_drcog_1_2020.json",
        "tiles": 875,
    },
    "co_drcog_2020_b2": {
        "project": "CO_DRCOG_2020_B20/CO_DRCOG_2_2020",
        "extents": "https://wifire-data.sdsc.edu/dataset/5211b6f6-877e-4e06-99a7-0d9389b10c32"
                   "/resource/8eeb2e31-a615-453e-bca4-92498d8f4c0f/download/"
                   "spatial_extents_co_drcog_2_2020.json",
        "tiles": 3451,
        "notes": "Denver proper. This is the block a Denver AOI resolves into.",
    },
    "co_drcog_2020_b3": {
        "project": "CO_DRCOG_2020_B20/CO_DRCOG_3_2020",
        "extents": "https://wifire-data.sdsc.edu/dataset/9e61a570-8a5a-45a3-82df-1a21a7416271"
                   "/resource/e11119cf-2660-4976-af7a-589c5f4c2533/download/"
                   "spatial_extents_co_drcog_3_2020.json",
        "tiles": 2179,
    },
}

LICENSE = "Public domain (US Government work, 17 U.S.C. §105) -- USGS 3DEP"


@dataclass(frozen=True)
class RemoteTile:
    id: str
    url: str
    west: float
    south: float
    east: float
    north: float
    acquisition: str
    start: str = ""
    end: str = ""

    def intersects(self, bbox) -> bool:
        west, south, east, north = bbox
        return not (self.west > east or self.east < west
                    or self.south > north or self.north < south)

    @property
    def name(self) -> str:
        return self.url.rsplit("/", 1)[-1]


def _bbox_of(geometry: dict) -> tuple[float, float, float, float]:
    coords = geometry["coordinates"]
    while coords and isinstance(coords[0][0], list):
        coords = coords[0]
    array = np.asarray(coords, dtype=float)
    return (float(array[:, 0].min()), float(array[:, 1].min()),
            float(array[:, 0].max()), float(array[:, 1].max()))


class RemoteIndex:
    """Per-tile extents and URLs for one or more published acquisitions."""

    def __init__(self, tiles: list[RemoteTile]):
        self.tiles = tiles

    def __len__(self) -> int:
        return len(self.tiles)

    @classmethod
    def load(cls, acquisition: str, *, cache_dir: str | Path | None = None,
             timeout: int = 180) -> "RemoteIndex":
        if acquisition not in ACQUISITIONS:
            raise KeyError(f"unknown acquisition {acquisition!r}; "
                           f"have {sorted(ACQUISITIONS)}")
        entry = ACQUISITIONS[acquisition]
        geojson = _fetch_json(entry["extents"], acquisition, cache_dir, timeout)

        tiles = []
        for feature in geojson.get("features", []):
            props = feature.get("properties") or {}
            url = props.get("url")
            geometry = feature.get("geometry")
            if not url or not geometry:
                continue
            west, south, east, north = _bbox_of(geometry)
            temporal = props.get("temporal") or {}
            tiles.append(RemoteTile(
                id=props.get("description_id") or url.rsplit("/", 1)[-1],
                url=url, west=west, south=south, east=east, north=north,
                acquisition=acquisition,
                start=str(temporal.get("startTime", "")),
                end=str(temporal.get("endTime", "")),
            ))
        return cls(tiles)

    @classmethod
    def load_all(cls, acquisitions=None, **kwargs) -> "RemoteIndex":
        names = acquisitions or list(ACQUISITIONS)
        tiles: list[RemoteTile] = []
        for name in names:
            tiles.extend(cls.load(name, **kwargs).tiles)
        return cls(tiles)

    def query(self, bbox_wgs84) -> list[RemoteTile]:
        return [t for t in self.tiles if t.intersects(bbox_wgs84)]

    def around(self, lon: float, lat: float, size_deg: float) -> list[RemoteTile]:
        half = size_deg / 2
        return self.query((lon - half, lat - half, lon + half, lat + half))

    def summary(self) -> dict:
        if not self.tiles:
            return {"tiles": 0, "acquisitions": []}
        return {
            "tiles": len(self.tiles),
            "acquisitions": sorted({t.acquisition for t in self.tiles}),
            "bounds": [min(t.west for t in self.tiles), min(t.south for t in self.tiles),
                       max(t.east for t in self.tiles), max(t.north for t in self.tiles)],
        }


def _fetch_json(url: str, key: str, cache_dir, timeout: int) -> dict:
    """Fetch the extents, caching them: they are static and a few MB."""
    cache = Path(cache_dir) / f"extents_{key}.json" if cache_dir else None
    if cache is not None and cache.exists():
        try:
            return json.loads(cache.read_text())
        except json.JSONDecodeError:
            pass
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(payload)
    return json.loads(payload)

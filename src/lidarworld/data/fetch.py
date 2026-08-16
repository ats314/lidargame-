"""Resolve an area of interest into tile URLs, and download them.

Bulk LiDAR is fetched rather than vendored: one 3DEP tile is ~65 MB and a city
is thousands of them. The repository carries addresses; this module turns an
address into bytes on disk.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

from .catalog import PLACES, describe

TNM_API = "https://tnmaccess.nationalmap.gov/api/v1/products"
USER_AGENT = "lidarworld/0.2 (+https://github.com/ats314/lidargame-)"


def resolve_tiles(bbox_wgs84, *, dataset: str = "Lidar Point Cloud (LPC)",
                  limit: int = 50, prefer_project: str | None = None) -> list[dict]:
    """Ask The National Map which 3DEP tiles cover a WGS84 bbox.

    Returns newest-first, because 3DEP has overlapping surveys from different
    years and the recent one is almost always the one you want.
    """
    query = urllib.parse.urlencode({
        "datasets": dataset,
        "bbox": ",".join(str(v) for v in bbox_wgs84),
        "outputFormat": "JSON",
        "max": limit,
    })
    request = urllib.request.Request(f"{TNM_API}?{query}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.load(response)

    tiles = [{
        "title": item.get("title", ""),
        "url": item.get("downloadURL", ""),
        "published": item.get("publicationDate", ""),
        "bytes": item.get("sizeInBytes") or 0,
        "format": item.get("format", ""),
    } for item in payload.get("items", []) if item.get("downloadURL")]

    if prefer_project:
        preferred = [t for t in tiles if prefer_project in t["title"]]
        if preferred:
            tiles = preferred
    return sorted(tiles, key=lambda t: t["published"], reverse=True)


def download(url: str, dest: Path, *, chunk: int = 1 << 20, progress=None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    partial = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(request, timeout=600) as response, open(partial, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while True:
            block = response.read(chunk)
            if not block:
                break
            out.write(block)
            done += len(block)
            if progress:
                progress(done, total)
    partial.rename(dest)
    return dest


def fetch_place(place_id: str, out_dir: str | Path, *, max_tiles: int = 1,
                progress=None) -> list[Path]:
    """Download the tiles covering a named place from the catalogue."""
    if place_id not in PLACES:
        raise KeyError(f"unknown place {place_id!r}; have {sorted(PLACES)}")
    place = PLACES[place_id]
    source = describe(place["source"])
    if not source.commercial:
        raise ValueError(f"{source.id} is not cleared for commercial use")

    tiles = resolve_tiles(place["bbox_wgs84"], prefer_project=place.get("project"))
    if not tiles:
        raise RuntimeError(f"no tiles returned for {place_id}")

    out_dir = Path(out_dir)
    paths = []
    for tile in tiles[:max_tiles]:
        name = Path(urllib.parse.urlparse(tile["url"]).path).name
        paths.append(download(tile["url"], out_dir / name, progress=progress))
    return paths

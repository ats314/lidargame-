"""Resolve an area of interest into tile URLs, and download them.

Bulk LiDAR is fetched rather than vendored: one 3DEP tile is ~65 MB and a city
is thousands of them. The repository carries addresses; this module turns an
address into bytes on disk.
"""
from __future__ import annotations

import json
import time
import urllib.error
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


#: A tile is hundreds of megabytes from a public host with no SLA, so a reset
#: part-way through is ordinary rather than exceptional. Four attempts at
#: 2/4/8 s covers the transient case without turning a genuine 404 into a
#: two-minute wait.
DOWNLOAD_ATTEMPTS = 4


def download(url: str, dest: Path, *, chunk: int = 1 << 20, progress=None,
             attempts: int = DOWNLOAD_ATTEMPTS, sleep=time.sleep) -> Path:
    """Fetch `url` to `dest`, retrying a dropped connection.

    Without this a single `Connection reset by peer` fails the whole command --
    and it took down a viewer deploy on `main` for a docs-only commit, because
    the deploy fetches a real tile. The retry is on the transport, not on the
    result: an HTTP error is the server answering, so it is raised immediately
    rather than hammered.

    The partial file is discarded between attempts. Resuming with a Range
    request would be better for a 300 MB tile, but only if the server honours
    it; starting over is at least always correct.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    partial = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(1, max(1, attempts) + 1):
        try:
            with urllib.request.urlopen(request, timeout=600) as response, \
                    open(partial, "wb") as out:
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
                if total and done < total:
                    raise OSError(
                        f"truncated: {done} of {total} bytes from {url}")
        except urllib.error.HTTPError:
            # The server answered. Retrying a 404 or a 403 just repeats it.
            partial.unlink(missing_ok=True)
            raise
        except (urllib.error.URLError, OSError, TimeoutError):
            partial.unlink(missing_ok=True)
            if attempt == max(1, attempts):
                raise
            sleep(2 ** attempt)
            continue
        partial.rename(dest)
        return dest
    raise AssertionError("unreachable")


def fetch_place(place_id: str, out_dir: str | Path, *, max_tiles: int = 1,
                progress=None) -> list[Path]:
    """Download the tiles covering a named place from the catalogue."""
    if place_id not in PLACES:
        raise KeyError(f"unknown place {place_id!r}; have {sorted(PLACES)}")
    place = PLACES[place_id]
    source = describe(place["source"])
    if not source.commercial:
        raise ValueError(f"{source.id} is not cleared for commercial use")

    tiles = resolve_place_tiles(place, place_id)

    out_dir = Path(out_dir)
    paths = []
    for tile in tiles[:max_tiles]:
        name = Path(urllib.parse.urlparse(tile["url"]).path).name
        paths.append(download(tile["url"], out_dir / name, progress=progress))
    return paths


def resolve_place_tiles(place: dict, place_id: str) -> list[dict]:
    """Tiles covering a place, deterministically where the extents are published.

    The National Map answers a bbox query with whatever overlapping tiles it
    holds, newest first, and "newest" is a tie among tiles from one flight. That
    makes `fetch <place>` return a different tile between runs, which is fine
    until a build pins an area inside the tile it happened to get.

    The acquisition's published extents remove the ambiguity: they carry an
    exact bbox per tile, so the tile whose centre is nearest the place is a
    stable answer. TNM stays as the fallback for places with no index.
    """
    if place.get("grid") == "ahn":
        from . import ahn

        crop = place.get("suggested_crop")
        if crop:
            west, south, east, north = crop
        else:
            raise ValueError("an AHN place needs a suggested_crop in RD metres "
                             "to resolve tiles; its WGS84 bbox is documentation")
        tiles = ahn.tiles_for((west, south, east, north),
                              version=place.get("version", "ahn5"))
        return [{"title": t.id, "url": t.url, "published": place.get("version", ""),
                 "bytes": 0, "format": "LAZ"} for t in tiles]

    acquisition = place.get("acquisition")
    if acquisition:
        try:
            from .catalog_index import RemoteIndex

            index = RemoteIndex.load(acquisition)
            west, south, east, north = place["bbox_wgs84"]
            hits = index.query((west, south, east, north))
            if hits:
                cx, cy = (west + east) / 2, (south + north) / 2
                hits.sort(key=lambda t: (((t.west + t.east) / 2 - cx) ** 2
                                         + ((t.south + t.north) / 2 - cy) ** 2))
                return [{"title": t.id, "url": t.url, "published": t.end,
                         "bytes": 0, "format": "LAZ"} for t in hits]
        except Exception as exc:                       # network, parse, anything
            print(f"  published extents unavailable ({type(exc).__name__}), "
                  f"falling back to The National Map", flush=True)

    tiles = resolve_tiles(place["bbox_wgs84"], prefer_project=place.get("project"))
    if not tiles:
        raise RuntimeError(f"no tiles returned for {place_id}")
    return tiles

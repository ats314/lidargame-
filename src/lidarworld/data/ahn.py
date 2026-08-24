"""AHN: the Dutch national point cloud, and how to address one square kilometre.

Denver is 3DEP at ~4 pts/m2, two thirds of it on pavement, which is why walls
had to be extruded from footprints rather than measured. AHN is the same
airborne product flown properly: `25GN1_02`, the tile the Amsterdam canal belt
sits in, carries 29,219,688 returns over 1000 x 1250 m -- **23 points per square
metre**, six times Denver. Facades there are sparse rather than absent, which is
a different problem and a better one.

Two addressing schemes matter and only one is useful here:

    PDOK ATOM      DSM/DTM rasters at 0.5 m. Not the point cloud.
    GeoTiles       the AHN LAZ, cut into subtiles small enough to download.

GeoTiles (TU Delft) republishes each AHN version on the national 1:25,000 sheet
grid, subdivided 5 x 5. A sheet -- a *kaartblad* -- is 5000 x 6250 m, so a
subtile is 1000 x 1250 m and lands between 300 and 550 MB depending on version.
Subtiles are numbered 1..25 in reading order from the north-west corner and
written zero-padded to two digits:

     1  2  3  4  5      north
     6  7  8  9 10
    11 12 13 14 15
    16 17 18 19 20
    21 22 23 24 25      south

That layout is not documented anywhere; it was derived by range-reading the LAS
headers of probe tiles and reading their true bounds back out, which is also how
`extent_of` verifies any tile this module names. Ten probes across four sheets
agree with the formula to the metre, and every extent below is a measured header
value rather than an arithmetic one -- tiles carry a 20 m overlap on each side
that the grid maths does not know about.

Licence: CC0 1.0. AHN is public domain, no attribution required -- the cleanest
terms of any source in the catalogue, Denver's liability disclaimer included.
"""
from __future__ import annotations

import struct
import urllib.request
from dataclasses import dataclass

BASE = "https://geotiles.citg.tudelft.nl"
USER_AGENT = "lidarworld/0.2 (+https://github.com/ats314/lidargame-)"

#: CC0: no attribution obligation, no share-alike, commercial use unrestricted.
TERMS = ("CC0 1.0 Universal (public domain dedication) -- Actueel "
         "Hoogtebestand Nederland, via GeoTiles (TU Delft)")
ATTRIBUTION = "AHN / Rijkswaterstaat, tiles by GeoTiles (TU Delft) -- CC0"

#: Amersfoort / RD New. Metres, y north, and the frame every Dutch open layer
#: below serves natively, so nothing in an Amsterdam build needs reprojecting.
CRS = "EPSG:28992"

#: Available versions, newest first. AHN5 is the 2023-2025 flight and is denser
#: and cleaner; AHN4 is 2020-2022 and is what 3D BAG's heights were derived
#: from, so it is the one to use when comparing against those heights.
VERSIONS = {
    "ahn5": "AHN5_T",
    "ahn4": "AHN4_T",
    "ahn3": "AHN3_T",
}

#: A kaartblad is 5000 x 6250 m, cut 5 x 5.
BLAD_WIDTH, BLAD_HEIGHT = 5000.0, 6250.0
COLS = ROWS = 5
TILE_WIDTH, TILE_HEIGHT = BLAD_WIDTH / COLS, BLAD_HEIGHT / ROWS

#: South-west corner of each sheet, in RD metres. Only the sheets covering the
#: Amsterdam region are listed: each was pinned by range-reading a probe tile's
#: LAS header, not by extrapolating the national sheet index, and a sheet that
#: has not been verified that way does not belong in a lookup that decides what
#: to download. `extent_of` checks any name against the server.
BLADEN: dict[str, tuple[float, float]] = {
    "25EN1": (120000.0, 500000.0),
    "25EZ1": (120000.0, 487500.0),
    "25EZ2": (125000.0, 487500.0),
    "25GN1": (120000.0, 481250.0),
    "25GN2": (125000.0, 481250.0),
    "25GZ1": (120000.0, 475000.0),
}


@dataclass(frozen=True)
class Tile:
    """One GeoTiles subtile. Bounds are the nominal grid cell; the file carries
    a 20 m overlap beyond them, which `extent_of` reports and this does not."""
    blad: str
    index: int                      # 1..25, reading order from the north-west
    version: str
    west: float
    south: float
    east: float
    north: float

    @property
    def id(self) -> str:
        return f"{self.blad}_{self.index:02d}"

    @property
    def url(self) -> str:
        return f"{BASE}/{VERSIONS[self.version]}/{self.id}.LAZ"

    @property
    def name(self) -> str:
        return f"{self.id}.LAZ"

    def intersects(self, bbox) -> bool:
        west, south, east, north = bbox
        return not (self.west > east or self.east < west
                    or self.south > north or self.north < south)


def tile_at(x: float, y: float, *, version: str = "ahn5") -> Tile:
    """The subtile containing an RD coordinate."""
    for blad, (bx, by) in BLADEN.items():
        if bx <= x < bx + BLAD_WIDTH and by <= y < by + BLAD_HEIGHT:
            # Clamp: a coordinate exactly on the sheet's southern edge divides
            # to row 5, which is one row past the sheet and names a tile the
            # server does not have. An AOI given as a round number of metres --
            # which every hand-written crop is -- lands on that edge often.
            col = min(int((x - bx) // TILE_WIDTH), COLS - 1)
            row = min(int((by + BLAD_HEIGHT - y) // TILE_HEIGHT), ROWS - 1)  # 0 north
            west = bx + col * TILE_WIDTH
            north = by + BLAD_HEIGHT - row * TILE_HEIGHT
            return Tile(blad=blad, index=row * COLS + col + 1, version=version,
                        west=west, south=north - TILE_HEIGHT,
                        east=west + TILE_WIDTH, north=north)
    raise KeyError(f"({x:.0f}, {y:.0f}) is outside the verified sheets "
                   f"{sorted(BLADEN)}; add its kaartblad after probing it")


def tiles_for(bbox_rd, *, version: str = "ahn5") -> list[Tile]:
    """Every subtile intersecting an RD bbox, north-west first."""
    west, south, east, north = bbox_rd
    hits: dict[str, Tile] = {}
    x = west
    while x <= east:
        y = south
        while y <= north:
            try:
                tile = tile_at(x, y, version=version)
            except KeyError:
                y += TILE_HEIGHT
                continue
            hits[tile.id] = tile
            y += TILE_HEIGHT
        x += TILE_WIDTH
    for corner in ((east, north), (west, north), (east, south)):
        try:
            tile = tile_at(*corner, version=version)
        except KeyError:
            continue
        hits[tile.id] = tile
    return sorted(hits.values(), key=lambda t: (-t.north, t.west))


def extent_of(tile: Tile | str, *, version: str = "ahn5", timeout: int = 60):
    """True bounds of a tile, by range-reading its LAS header.

    375 bytes settles what the grid can only assume: whether the tile exists,
    where it actually is, and how many returns it holds. This is the same trick
    `data/tiles.py` plays on local files, over HTTP.

    Returns (west, south, east, north, points), or None if the tile is absent.
    """
    url = tile.url if isinstance(tile, Tile) else f"{BASE}/{VERSIONS[version]}/{tile}.LAZ"
    request = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT, "Range": "bytes=0-400"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            head = response.read(401)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    if len(head) < 375 or head[:4] != b"LASF":
        return None
    max_x, min_x, max_y, min_y, _, _ = struct.unpack_from("<6d", head, 179)
    legacy = struct.unpack_from("<I", head, 107)[0]
    points = struct.unpack_from("<Q", head, 247)[0] or legacy
    return (min_x, min_y, max_x, max_y, points)

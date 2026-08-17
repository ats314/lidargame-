"""Helsinki 3D+: the only city publishing a semantic model and a reality mesh.

Hamburg gives finished LoD3 buildings whose walls are flat polygons -- every
window is paint, and measured over its own best block the aerial texture only
covers a median 17% of a wall's height, all of it at the top. Vienna gives
survey-grade street capture over one 250 m suburban street. Helsinki gives
something neither has: a photogrammetric triangle mesh of the whole city, in
which a balcony is a balcony and a window reveal has depth, published beside a
semantic CityGML model of the same place.

    kolmioverkkomalli   reality mesh, 2017, 122 tiles of 2 km, OBJ + textures
    kaupunkitietomalli  semantic CityGML with textures and a WFS

Both CC BY 4.0, both direct download, no form.

The whole mesh is 190 GB, which is not a thing to acquire casually and is not
what a demo needs. Tile names decode to kilometre coordinates, so an area of
interest resolves to a file rather than a search -- see `tile_for`.

CRS is ETRS-GK25 (EPSG:3879), heights N2000. Note the easting carries the zone:
a Helsinki easting is 25,497,363, and the tile name drops the leading 25.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

#: Directory listing of the whole-city mesh, 2 km tiles with 250 m subtiles.
MESH_BASE = ("https://3d.hel.ninja/data/mesh/"
             "Helsinki3D-MESH_2017_OBJ_2km-250m_ZIP")

#: The showcase district, delivered in five formats including two that Unreal
#: reads natively. Each is 10-13 GB, so it is a deliberate choice, not a default.
KALASATAMA = {
    "3d_tiles": "https://3d.hel.ninja/data/mesh/Kalasatama/"
                "Helsinki3D_MESH_Kalasatama_2017_3D_Tiles_ZIP.zip",
    "obj": "https://3d.hel.ninja/data/mesh/Kalasatama/"
           "Helsinki3D_MESH_Kalasatama_2017_OBJ_ZIP.zip",
    "fbx": "https://3d.hel.ninja/data/mesh/Kalasatama/"
           "Helsinki3D_MESH_Kalasatama_2017_FBX_ZIP.zip",
    "dae": "https://3d.hel.ninja/data/mesh/Kalasatama/"
           "Helsinki3D_MESH_Kalasatama_2017_DAE_ZIP.zip",
    "3mx_3sm": "https://3d.hel.ninja/data/mesh/Kalasatama/"
               "Helsinki3D_MESH_Kalasatama_2017_3MX_3SM_ZIP.zip",
}

CITYGML_KALASATAMA = ("https://3d.hel.ninja/data/citygml/"
                      "Helsinki3D_CityGML_Kalasatama_20190326.zip")
PHOTOS_360 = "https://3d.hel.ninja/data/360/Helsinki3D_360Photos.zip"
VIDEOS_360 = "https://3d.hel.ninja/data/360/Helsinki3D_360Videos.zip"

#: The semantic model's live interface, so a building's attributes can be had
#: without downloading a city.
CITYDB_WFS = ("https://kartta.hel.fi/3d/citydb-wfs/wfs"
              "?SERVICE=WFS&VERSION=2.0.0&REQUEST=GetCapabilities")

TERMS = ("Helsingin kaupunki / City of Helsinki -- CC BY 4.0, commercial use "
         "permitted, attribution required")
ATTRIBUTION = "© City of Helsinki"

CRS = "EPSG:3879"          # ETRS-GK25
VERTICAL = "N2000"

#: Tiles are 2 km and the name is <northing_km><easting_km>, each three digits,
#: with the UTM-style zone prefix dropped from the easting.
TILE_M = 2000
EASTING_PREFIX = 25_000_000


def tile_for(x: float, y: float) -> str:
    """Mesh tile code covering a point in EPSG:3879.

    Senate Square (25497363, 6672958) -> '672496'. Verified against the
    published listing, not against the convention: Esplanadi, Kamppi and Senate
    Square all resolve to the same tile, which is what a 2 km tile over the
    historic core should do.
    """
    east = int(x - EASTING_PREFIX) if x > EASTING_PREFIX else int(x)
    return f"{int(y // TILE_M * 2) % 1000:03d}{int(east // TILE_M * 2) % 1000:03d}"


def mesh_url(code: str) -> str:
    return f"{MESH_BASE}/Helsinki3D_2017_OBJ_{code}x2.zip"


@dataclass(frozen=True)
class Tile:
    code: str
    label: str
    bytes: int | None = None

    @property
    def url(self) -> str:
        return mesh_url(self.code)

    @property
    def origin(self) -> tuple[int, int]:
        """South-west corner in EPSG:3879 metres."""
        north = int(self.code[:3])
        east = int(self.code[3:])
        return EASTING_PREFIX + east * 1000, 6_000_000 + north * 1000


#: The nine tiles covering central Helsinki, 6 x 6 km, 13.3 GB. Sizes are the
#: server's own, measured 2026-08-17.
CENTRAL: dict[str, Tile] = {t.code: t for t in [
    Tile("672496", "historic core: Senate Square, Esplanadi, Kamppi", 1_940_000_000),
    Tile("674496", "Kruununhaka and Hakaniemi", 2_170_000_000),
    Tile("674494", "Töölö", 2_330_000_000),
    Tile("672494", "west core", 1_530_000_000),
    Tile("676498", "north", 1_540_000_000),
    Tile("674498", "Kalasatama", 1_310_000_000),
    Tile("670496", "south, Kaivopuisto", 1_090_000_000),
    Tile("670494", "south-west", 730_000_000),
    Tile("672498", "east core", 670_000_000),
]}


def acquisition_plan() -> list[dict]:
    """What to fetch, best-demo-first.

    The historic core leads because it is the tile a person means by "Helsinki",
    and the small extras come next because they cost nothing and answer
    questions the mesh cannot -- whether the semantic model carries facade
    openings, and what the published 360 imagery actually is.
    """
    plan = [{"key": "citygml_kalasatama", "kind": "semantic",
             "url": CITYGML_KALASATAMA, "bytes": 23_546_638,
             "name": "CityGML Kalasatama (semantic model sample)"},
            {"key": "photos_360", "kind": "imagery", "url": PHOTOS_360,
             "bytes": 47_302_273, "name": "Helsinki3D 360 photos"}]
    plan += [{"key": f"mesh_{t.code}", "kind": "mesh", "url": t.url,
              "bytes": t.bytes, "name": f"{t.code} — {t.label}"}
             for t in CENTRAL.values()]
    return plan


def subtiles(archive: str | Path) -> list[str]:
    """The 250 m OBJ subtiles inside a 2 km mesh tile."""
    import zipfile
    with zipfile.ZipFile(archive) as handle:
        return sorted(n for n in handle.namelist() if n.lower().endswith(".obj"))


def subtile_origin(name: str) -> tuple[int, int] | None:
    """South-west corner of a 250 m subtile, from its filename.

    Returns None rather than guessing when the name does not carry coordinates,
    because a wrongly placed subtile is a building in the wrong street and
    nothing about it raises.
    """
    digits = re.findall(r"(\d{6,7})", Path(name).stem)
    if len(digits) < 2:
        return None
    north, east = int(digits[0]), int(digits[1])
    if east < EASTING_PREFIX:
        east += EASTING_PREFIX
    return east, north

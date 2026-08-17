"""Hamburg LoD3.0-HH: a city that has already been reconstructed, with textures.

Denver and Vienna are both *reconstruction* problems -- sensors in, geometry
out. This one is not, and that is the point of having it.

Hamburg publishes its entire building stock as textured CityGML LoD3: roof
structures over 1 m2 modelled, significant overhangs kept, facade openings
present, and a texture per surface derived from the 2020 nadir-and-oblique
flight at 20 cm. It was compiled photogrammetrically by the state survey office,
not inferred by anyone's algorithm.

That decouples two failures this repo has never been able to tell apart. When
Denver looked like nothing, the cause could have been the reconstruction, the
Spatial IR, the materialisation, or the viewer, and no measurement separated
them. Feeding Hamburg in holds the front half fixed at known-good and tests only
the back half:

    official semantic LoD3 + texture -> World Seed -> engine

If that looks wrong, the compiler is wrong. If it looks right, the back half is
proven and reconstruction can be attacked on its own with Vienna.

Verified against the live server on 2026-08-17: Area1 is 1,524,133,837 bytes and
holds 82 tiles, 36,437 JPEGs (1.16 GB) and 2.28 GB of GML. Textures are not a
claim off the dataset page; they are entries in the archive index.

Range requests work, so a single tile can be pulled without the 1.5 GB -- see
`open_remote` and `extract_tile`.
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

#: Hamburg's archives are Deflate64 (method 9), which the standard library does
#: not decompress -- it raises NotImplementedError only when a member is read,
#: so the index parses fine and the failure looks like a corrupt download.
#: `zipfile-deflate64` registers the decompressor by import; Info-ZIP `unzip`
#: does not handle it either, so there is no shell fallback.
try:                                            # pragma: no cover - env dependent
    import zipfile_deflate64                    # noqa: F401
    DEFLATE64 = True
except ImportError:                             # pragma: no cover
    DEFLATE64 = False

DEFLATE64_METHOD = 9

#: The redirect target. daten-hamburg.de 301s to www.daten-hamburg.de, and
#: urllib follows it, but naming the resolved host keeps range requests on one
#: connection instead of re-negotiating per slice.
BASE = "https://www.daten-hamburg.de/opendata/3d_stadtmodell_lod3"

#: Datenlizenz Deutschland Namensnennung 2.0 is the standard grant for Hamburg
#: Transparenzportal geodata: commercial use permitted, attribution required.
#: Recorded, not enforced -- see CLAUDE.md on third-party terms.
TERMS = ("Freie und Hansestadt Hamburg, Landesbetrieb Geoinformation und "
         "Vermessung -- dl-de/by-2-0, commercial use permitted, attribution "
         "required")
ATTRIBUTION = "Freie und Hansestadt Hamburg, LGV"

#: ETRS89 / UTM zone 32N with DHHN2016 heights, the German national frame.
CRS = "EPSG:25832"


@dataclass(frozen=True)
class Area:
    id: str
    name: str
    file: str
    bytes: int | None = None
    notes: str = ""

    @property
    def url(self) -> str:
        return f"{BASE}/{self.file}"


AREAS: dict[str, Area] = {area.id: area for area in [
    Area("area1", "Innenstadt", "LoD3-HH_Area1_2023_12_14.zip",
         1_524_133_837,
         "The inner city -- Altstadt, Neustadt, Speicherstadt, HafenCity. "
         "82 tiles. This is the one to start from: it is the smallest package "
         "and it holds the buildings anyone would recognise as Hamburg."),
    Area("area2", "Bergedorf", "LoD3-HH_Area2_2023_12_27.zip"),
    Area("area3", "Harburg und Südwesten", "LoD3-HH_Area3_2024_04_04.zip"),
    Area("area4", "Altona und Westen", "LoD3-HH_Area4_2024_10_10.zip"),
    Area("area5", "Nordosten (Wandsbek)", "LoD3-HH_Area5_2024_02_23.zip"),
]}

#: What the model actually carries, from the publisher's own description rather
#: than from looking at it. Kept next to the code because the interesting fields
#: are the ones that decide whether a stage has evidence or is inventing.
MODEL = {
    "lod": "3.0",
    "source": "manual photogrammetric evaluation of a 2020 nadir + oblique flight",
    "roofs": "detailed roof landscape beyond LoD2; superstructures over 1 m2 "
             "and significant overhangs modelled",
    "coverage": "every building on Hamburg territory, not only cadastre "
                "buildings (excludes Neuwerk and Scharhörn)",
    "textures": "generated from the oblique imagery at 20 cm, "
                "data-protection-compliant resolution",
    "terrain": "buildings placed on DGM 5H, a 5 m DTM with break lines",
    "ids": "essentially ALKIS cadastre ids, with occasional divergence in "
           "both footprint and id",
    "independence": 2,
}

#: What 20 cm buys, stated plainly so nobody has to be disappointed by it later.
#: A 3 m storey is 15 pixels tall. That is enough to read the facade -- material,
#: bay rhythm, window and door positions, cornice line -- and not enough to be a
#: close-up texture. The intended use is as appearance *evidence* driving a
#: procedural material, with the photograph itself as the baseline to beat.
TEXTURE_GSD_M = 0.20


#: Everything else the city publishes that a *world* needs, as opposed to a pile
#: of buildings. The first Hamburg render put a correct inner-city block on a
#: green field with no streets, and that was not a rendering fault: the building
#: model is the only thing that had been acquired. Buildings are the figure;
#: these are the ground.
#:
#: All dl-de/by-2-0, all direct download, all resolved live from the Hamburg
#: Transparenzportal CKAN API on 2026-08-17 rather than read off a page.
ARCHIVE = "https://archiv.transparenz.hamburg.de/hmbtgarchive/HMDK"

CONTEXT: dict[str, dict] = {
    "terrain": {
        "name": "Digitales Höhenmodell DGM 1",
        "url": f"{ARCHIVE}/dgm1_2x2km_xyz_hh_2016-01-04_4735_snap_1_16267_snap_1.ZIP",
        "bytes": 2_900_000_000,
        "format": "ASCII xyz, 2 x 2 km tiles, 1 m grid",
        "why": "The LoD3 buildings are already placed on a DGM but the DGM is "
               "not in the package, so the ground under them is missing "
               "entirely. Without it a street is a hole between two blocks.",
    },
    "roads": {
        "name": "Straßen- und Wegenetz Hamburg (HH-SIB)",
        "url": f"{ARCHIVE}/strassen_hh_sibstrassen_hh-sib_2014-11-20_9337_snap_1.ZIP",
        "format": "GML",
        "why": "Carriageway and path network. The seed's road list was empty, "
               "so the generator had nothing to stamp and every unbuilt cell "
               "stayed generic ground -- which the theme painted as grass, in "
               "the middle of the Altstadt.",
    },
    "cadastre": {
        "name": "ALKIS Liegenschaftskarte, ausgewählte Daten",
        "url": f"{ARCHIVE}/alkis_liegenschaftskarte_ausgewaehltedaten_hh_2018-07-07_25591_snap_1.GML",
        "bytes": 526_000_000,
        "format": "GML",
        "why": "Parcels and actual land use. This is what separates pavement "
               "from carriageway from courtyard from park, which is the "
               "distinction the ground surface needs and centrelines cannot "
               "give.",
    },
    "orthophoto": {
        "name": "Digitale Orthophotos 20 cm",
        "wms": "https://geodienste.hamburg.de/HH_WMS_Cache_DOP20",
        "why": "Ground texture, at the same 20 cm as the facades. Deliberately "
               "*not* a bulk download: a full epoch is 15 GB of JPEG for the "
               "whole city, and a block needs a few square kilometres. Fetch "
               "per area of interest from the cached WMS instead.",
        "note": "no bulk fetch -- request tiles for the AOI",
    },
}


def context_urls() -> dict[str, str]:
    """Direct downloads only; the orthophoto is a service, not a file."""
    return {key: spec["url"] for key, spec in CONTEXT.items() if "url" in spec}


def master_city_plan() -> list[dict]:
    """Everything to acquire, in the order that makes a block look right soonest.

    Area1 first because it is the inner city and the smallest textured package;
    then the ground, because a correct building on a green field is still wrong;
    then the rest of the city. Sizes are the publisher's own, so the total is
    honest about what this costs before anything starts.
    """
    def entry(area: Area) -> dict:
        # `url` is a property, so spreading __dict__ silently omits it -- which
        # it did, and the fetcher skipped every building package without a word
        # because its guard was `if not url: continue`. Named explicitly now.
        return {"key": f"lod3_{area.id}", "kind": "buildings", "name": area.name,
                "url": area.url, "bytes": area.bytes, "notes": area.notes}

    plan = [entry(AREAS["area1"])]
    plan += [{"key": k, "kind": "context", **v}
             for k, v in CONTEXT.items() if "url" in v]
    plan += [entry(AREAS[a]) for a in ("area2", "area3", "area5", "area4")]
    return plan


#: Tile naming decodes to kilometre coordinates in both layers, which is what
#: turns "what covers this point" into a lookup instead of a spatial query.
#:
#:     buildings   6534                    E 565-566 km, N 5934-5935 km, 1 km
#:     terrain     DGM1_32564_5934_2_FHH   E 564-566 km, N 5934-5936 km, 2 km
#:
#: The terrain easting carries the UTM zone as a prefix (32), which has to come
#: off before it is a coordinate. One terrain tile covers exactly four building
#: tiles.
BUILDING_TILE_M = 1000
TERRAIN_TILE_M = 2000
UTM_ZONE = 32


def building_tile(x: float, y: float) -> str:
    """The LoD3 tile id covering a point, e.g. 565648, 5934179 -> '6534'."""
    return f"{int(x // BUILDING_TILE_M) % 100:02d}{int(y // BUILDING_TILE_M) % 100:02d}"


def terrain_tile(x: float, y: float) -> str:
    """The DGM 1 member name covering a point."""
    east = int(x // TERRAIN_TILE_M) * (TERRAIN_TILE_M // 1000)
    north = int(y // TERRAIN_TILE_M) * (TERRAIN_TILE_M // 1000)
    return f"DGM1_{UTM_ZONE}{east}_{north}_2_FHH.xyz"


def read_dgm(archive: str | Path, tile: str):
    """One DGM 1 tile as (origin_xy, cell, z-grid), z indexed [x, y].

    The file is 4 million ASCII triples and `np.loadtxt` takes about a minute on
    it, which is a minute per tile of a 243 tile city. `np.frombuffer` on the
    whitespace-separated text is the same parse in about a second.

    Posts sit on half-metre centres -- 549991.50, not 549991.0 -- so the origin
    is the first post, not the tile corner. Getting that wrong shifts the whole
    terrain half a metre against the buildings, which is the same order as the
    offset we are trying to measure.
    """
    import numpy as np

    with zipfile.ZipFile(archive) as handle:
        name = next((n for n in handle.namelist() if n.endswith(tile)), None)
        if name is None:
            raise KeyError(f"{tile} not in {archive}")
        raw = handle.read(name)
    flat = np.array(raw.split(), dtype=np.float64).reshape(-1, 3)

    origin = flat[:, :2].min(axis=0)
    span = flat[:, :2].max(axis=0) - origin
    # Posts are on a 1 m grid; derive it rather than assuming, so a 0.5 m or
    # 2 m product does not silently come out scrambled.
    nx = int(round(span[0])) + 1
    ny = int(round(span[1])) + 1
    cell = float(span[0] / (nx - 1)) if nx > 1 else 1.0

    grid = np.full((nx, ny), np.nan, dtype=np.float32)
    ix = np.rint((flat[:, 0] - origin[0]) / cell).astype(np.int64)
    iy = np.rint((flat[:, 1] - origin[1]) / cell).astype(np.int64)
    inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    grid[ix[inside], iy[inside]] = flat[inside, 2]
    return origin, cell, grid


class _HttpFile(io.RawIOBase):
    """Enough of a seekable file for `zipfile` to work over HTTP ranges.

    A 1.5 GB archive holds 82 tiles and only one is needed to see whether the
    back half of the compiler works. Reading the central directory costs a few
    kilobytes, after which any single member can be pulled on its own.
    """

    def __init__(self, url: str):
        self.url = url
        self.pos = 0
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=60) as response:
            self.size = int(response.headers["Content-Length"])

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def tell(self) -> int:
        return self.pos

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def read(self, n: int = -1) -> bytes:
        # Past the end is not an error here: zipfile's end-of-directory scan
        # deliberately over-reads, and a 416 from the server would abort it.
        if self.pos >= self.size:
            return b""
        if n < 0:
            n = self.size - self.pos
        end = min(self.pos + n, self.size) - 1
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(request, timeout=180) as response:
            data = response.read()
        self.pos += len(data)
        return data

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


def open_remote(area: str = "area1") -> zipfile.ZipFile:
    """The archive's index, without the archive."""
    return zipfile.ZipFile(io.BufferedReader(_HttpFile(AREAS[area].url), 1 << 20))


def tiles(archive: zipfile.ZipFile) -> list[str]:
    """Tile ids, in name order."""
    return sorted({name.split("/", 1)[0] for name in archive.namelist()
                   if name.endswith(".gml")})


def extract_tile(archive: zipfile.ZipFile, tile: str, out: str | Path,
                 *, textures: bool = True) -> dict:
    """Pull one tile's GML and, by default, its textures.

    Returns what was written rather than printing, so a caller can report the
    byte counts honestly instead of assuming the fetch matched the index.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    wanted = [i for i in archive.infolist()
              if i.filename.startswith(f"{tile}/") and not i.is_dir()
              and (textures or i.filename.endswith(".gml"))]
    if not wanted:
        raise KeyError(f"tile {tile!r} not in this archive")
    if not DEFLATE64 and any(i.compress_type == DEFLATE64_METHOD for i in wanted):
        raise RuntimeError(
            "these members are Deflate64; pip install zipfile-deflate64. "
            "The standard library and Info-ZIP unzip both fail on it, and the "
            "stdlib failure only surfaces on read, which reads as a bad download.")
    written = {"gml": 0, "images": 0, "gml_bytes": 0, "image_bytes": 0}
    for info in wanted:
        target = out / info.filename
        target.parent.mkdir(parents=True, exist_ok=True)
        data = archive.read(info)
        target.write_bytes(data)
        key = "gml" if info.filename.endswith(".gml") else "images"
        written[key] += 1
        written[f"{'gml' if key == 'gml' else 'image'}_bytes"] += len(data)
    return written

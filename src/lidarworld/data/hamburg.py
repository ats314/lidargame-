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

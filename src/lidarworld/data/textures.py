"""CC0 material libraries, and what their terms actually are.

The measured macro cannot supply high frequency. That is settled and measured:
this mesh is over-smoothed to 4.4 cm locally, the median of 48 bays carries half
the detail of one bay, and no recombination of those bays recovers it. So the
sharpness has to come from somewhere else, and there are two candidates.

A procedural generator, which is what `themes/procedural.py` does. It is
seamless, infinitely variable and free, and it is noise that resembles masonry
rather than masonry -- `stone_block` produces a plausible coursing and nothing
that was ever a wall.

A photographed material, which is what this module fetches. A Poly Haven brick
scan *is* a wall: it carries the chipping, the efflorescence, the mortar that was
struck badly on one course, all the correlated detail a generator has no way to
invent. And the measured facade still supplies the identity -- colour, coursing
period, material class -- so the pairing is the same macro x micro architecture
with a photograph in the micro slot.

Poly Haven matters especially because it publishes each texture's real-world
dimensions in millimetres. Metric UV1 needs exactly that number, and without it a
library texture has to be scaled by eye, which is how one course of brick ends up
spanning a shed and a warehouse identically.

Terms are recorded, not enforced. Both libraries below are CC0, which is as
permissive as it gets, and the catalogue entry says so with its source so the
owner can check rather than take this file's word for it.
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

#: Poly Haven's own API rejects a default Python user agent with 403 while
#: serving the identical request to curl. Not a policy denial -- a UA filter.
AGENT = "lidarworld/0.1 (+internal research tool)"

TIMEOUT = 60


@dataclass(frozen=True)
class Library:
    key: str
    name: str
    licence: str
    commercial: bool
    attribution: str
    api: str
    terms_url: str
    scale_published: bool          # does it state real-world size in metres?


LIBRARIES: dict[str, Library] = {
    "polyhaven": Library(
        key="polyhaven", name="Poly Haven",
        licence="CC0 1.0 Universal (public domain dedication)",
        commercial=True, attribution="not required; Poly Haven appreciated",
        api="https://api.polyhaven.com",
        terms_url="https://polyhaven.com/license",
        scale_published=True),
    "ambientcg": Library(
        key="ambientcg", name="ambientCG",
        licence="CC0 1.0 Universal (public domain dedication)",
        commercial=True, attribution="not required",
        api="https://ambientcg.com/api/v2",
        terms_url="https://ambientcg.com/license",
        scale_published=True),
}

#: Poly Haven categories that describe a wall. Roofs and floors are excluded --
#: not because they are useless but because matching a facade against a floor
#: tile is how a building ends up paved.
WALL_CATEGORIES = ("brick", "plaster", "concrete", "stone", "wall")

#: Categories and tags that mean the scan is of something walked on. Poly Haven
#: files pavements and floor tiles under "brick" alongside walls, and they are the
#: majority of that category.
UNDERFOOT = {"floor", "floors", "pavement", "ground", "road", "path", "tiles",
             "cobblestone", "crosswalk", "gravel", "terrain", "outdoor floor"}


def _get(url: str) -> dict | list:
    request = urllib.request.Request(url, headers={"User-Agent": AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return json.loads(response.read())


def _download(url: str, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        return out
    request = urllib.request.Request(url, headers={"User-Agent": AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        out.write_bytes(response.read())
    return out


@dataclass
class Material:
    """One library texture, with the real-world scale a metric UV needs."""
    key: str
    library: str
    name: str
    metres: float                  # width of one tile in the real world
    categories: tuple = ()
    tags: tuple = ()
    albedo: Path | None = None

    def to_record(self) -> dict:
        return {
            "key": self.key, "library": self.library, "name": self.name,
            "metres": round(self.metres, 3),
            "categories": list(self.categories),
            "licence": LIBRARIES[self.library].licence,
            "commercial": LIBRARIES[self.library].commercial,
            "epistemic": "measured elsewhere; a photograph of a different wall",
        }


def polyhaven_walls(limit: int = 40) -> list[Material]:
    """Wall materials from Poly Haven, with their published real-world size.

    `dimensions` comes back in millimetres and is the honest reason to prefer
    this library: a texture whose real size is unknown cannot be placed on a
    metric UV without guessing, and guessing the scale of masonry is the one
    error that makes a building read as a doll's house or a car park.
    """
    # Round-robin across the categories, not one at a time. Draining `brick`
    # first filled a 20-slot budget with twenty bricks and never reached plaster,
    # stone or concrete -- so a rendered Helsinki block could only ever be matched
    # to brick, and the "nearest" material was chosen from a category that had
    # already been decided by the fetch order.
    pools: list[list[tuple[str, dict]]] = []
    for category in WALL_CATEGORIES:
        try:
            assets = _get(f"{LIBRARIES['polyhaven'].api}"
                          f"/assets?t=textures&c={category}")
        except Exception:                                   # noqa: BLE001
            continue
        pools.append(sorted(assets.items()))

    out: list[Material] = []
    seen: set[str] = set()
    for depth in range(max((len(p) for p in pools), default=0)):
        for pool in pools:
            if depth >= len(pool) or len(out) >= limit:
                continue
            key, spec = pool[depth]
            dimensions = spec.get("dimensions")
            if key in seen or not dimensions:
                continue
            categories = tuple(spec.get("categories", ()))
            tags = tuple(spec.get("tags", ()))
            # A wall is not a floor. Matching a facade against a pavement scan is
            # how a building ends up paved, and the first run of this chose
            # "Brick Floor" for a six-storey frontage.
            if UNDERFOOT & {c.lower() for c in categories + tags}:
                continue
            seen.add(key)
            out.append(Material(
                key=key, library="polyhaven", name=spec.get("name", key),
                metres=float(dimensions[0]) / 1000.0,
                categories=categories, tags=tags))
        if len(out) >= limit:
            break
    return out


def fetch_albedo(material: Material, root: str | Path,
                 resolution: str = "1k") -> Material | None:
    """Download one material's colour map. Returns it with `albedo` filled in."""
    root = Path(root)
    try:
        files = _get(f"{LIBRARIES['polyhaven'].api}/files/{material.key}")
    except Exception:                                       # noqa: BLE001
        return None
    colour = files.get("Diffuse") or files.get("diffuse") or files.get("Color")
    if not colour:
        return None
    entry = colour.get(resolution) or next(iter(colour.values()))
    spec = entry.get("jpg") or entry.get("png") or next(iter(entry.values()))
    url = spec.get("url") if isinstance(spec, dict) else None
    if not url:
        return None
    suffix = Path(url).suffix or ".jpg"
    try:
        path = _download(url, root / f"{material.key}{suffix}")
    except Exception:                                       # noqa: BLE001
        return None
    material.albedo = path
    return material


def describe() -> list[dict]:
    """Every library and its terms, for the source catalogue."""
    return [{
        "key": lib.key, "name": lib.name, "licence": lib.licence,
        "commercial": lib.commercial, "attribution": lib.attribution,
        "terms": lib.terms_url, "real_world_scale_published": lib.scale_published,
    } for lib in LIBRARIES.values()]

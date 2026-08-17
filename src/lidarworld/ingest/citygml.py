"""Read textured CityGML LoD3 into surfaces, openings and texture bindings.

This is the one ingest path that does not start from returns. Everything else in
`ingest/` reads a sensor; this reads somebody else's finished reconstruction --
Hamburg's LoD3.0-HH, compiled photogrammetrically by the state survey office and
shipped with 20 cm textures.

The point is diagnostic. Denver could look like nothing for five different
reasons at once and no measurement separated them. Holding the geometry fixed at
known-good isolates the back half of the compiler: if a Hamburg block still
looks wrong after this, the fault is in the Spatial IR, the materialisation or
the viewer, and not in reconstruction.

Three things are extracted, and the third is the one Denver never had:

    boundary surfaces   wall / roof / ground, already classified
    openings            windows and doors as real geometry, not invented
    texture bindings    polygon id -> image + UV, so a facade has appearance
                        evidence rather than a guessed material

Epistemically these are `derived`, not `observed`. The survey office measured
imagery; the wall is their interpretation of it. Nothing here may be written
into the IR as a sensor observation -- that invariant already caught this repo
once, when reconstructed walls were marked `observed`.

Parsing is streaming. An inner-city tile is 142 MB of XML and holding a DOM for
it is pointless when the whole file is read once.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

#: CityGML has shipped as 1.0, 2.0 and 3.0 with different namespace URIs, and
#: Hamburg's tiles are not the only thing this will ever read. Matching on the
#: local name sidesteps the whole question; nothing here is ambiguous without
#: the namespace.
def _tag(element) -> str:
    tag = element.tag
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _attr(element, name: str):
    """An attribute by local name, whatever namespace it arrived in.

    `gml:id` and `xlink:href` are the two that matter and both are namespaced.
    """
    for key, value in element.attrib.items():
        if key.rsplit("}", 1)[-1] == name:
            return value
    return None


#: CityGML boundary surface types, mapped to the vocabulary the rest of the
#: compiler already speaks. ClosureSurface is deliberately absent: it is a
#: virtual lid over an opening in the building shell, not a real surface, and
#: materialising it puts glass where a passage should be.
SURFACE_KIND = {
    "WallSurface": "wall",
    "RoofSurface": "roof",
    "GroundSurface": "ground",
    "OuterCeilingSurface": "roof",
    "OuterFloorSurface": "ground",
}

OPENING_KIND = {"Window": "window", "Door": "door"}


@dataclass
class Polygon:
    """One ring set in world coordinates. Interiors are holes, not separate faces."""
    gml_id: str | None
    exterior: np.ndarray                       # (N, 3)
    interiors: list[np.ndarray] = field(default_factory=list)

    @property
    def area(self) -> float:
        """Planar area via the Newell vector, which needs no projection choice."""
        ring = self.exterior
        if len(ring) < 3:
            return 0.0
        normal = np.zeros(3)
        for i in range(len(ring)):
            a, b = ring[i], ring[(i + 1) % len(ring)]
            normal += np.cross(a, b)
        return float(np.linalg.norm(normal) / 2.0)


@dataclass
class Surface:
    kind: str                                  # wall | roof | ground
    gml_id: str | None
    polygons: list[Polygon] = field(default_factory=list)
    openings: list["Opening"] = field(default_factory=list)


@dataclass
class Opening:
    kind: str                                  # window | door
    gml_id: str | None
    polygons: list[Polygon] = field(default_factory=list)


@dataclass
class Building:
    gml_id: str | None
    attributes: dict = field(default_factory=dict)
    surfaces: list[Surface] = field(default_factory=list)

    def of(self, kind: str) -> list[Surface]:
        return [s for s in self.surfaces if s.kind == kind]

    @property
    def openings(self) -> list[Opening]:
        return [o for s in self.surfaces for o in s.openings]


@dataclass
class TextureBinding:
    """Which image paints which polygon, and where.

    CityGML binds a texture to a polygon by gml:id, so the link survives the
    surface hierarchy being flattened. `uv` is per-ring, matching the polygon's
    exterior-then-interiors order.
    """
    image: str
    target: str                                # polygon gml:id, without the '#'
    uv: list[np.ndarray] = field(default_factory=list)


def _poslist(element) -> np.ndarray | None:
    """Coordinates out of a gml:posList or a run of gml:pos."""
    for child in element.iter():
        if _tag(child) == "posList" and child.text:
            dim = int(child.get("srsDimension", 3))
            values = np.fromstring(child.text.strip(), sep=" ", dtype=float)
            if len(values) < dim * 3:
                return None
            return values.reshape(-1, dim)[:, :3]
    points = [np.fromstring(c.text.strip(), sep=" ", dtype=float)
              for c in element.iter() if _tag(c) == "pos" and c.text]
    if len(points) >= 3:
        return np.asarray(points)[:, :3]
    return None


def _polygons(element) -> list[Polygon]:
    """Every gml:Polygon under `element`, exteriors and holes kept apart."""
    out: list[Polygon] = []
    for polygon in element.iter():
        if _tag(polygon) != "Polygon":
            continue
        exterior, interiors = None, []
        for part in polygon:
            local = _tag(part)
            if local not in ("exterior", "interior"):
                continue
            ring = _poslist(part)
            if ring is None:
                continue
            if local == "exterior":
                exterior = ring
            else:
                interiors.append(ring)
        if exterior is not None:
            out.append(Polygon(_attr(polygon, "id"), exterior, interiors))
    return out


def _openings(surface_element) -> list[Opening]:
    out: list[Opening] = []
    for element in surface_element.iter():
        kind = OPENING_KIND.get(_tag(element))
        if kind is None:
            continue
        out.append(Opening(kind, _attr(element, "id"), _polygons(element)))
    return out


#: Attributes worth carrying forward. Generic string attributes in CityGML are
#: unbounded and most of a Hamburg building's are administrative; these are the
#: ones a generator or a scorer can act on.
KEEP_ATTRIBUTES = {"function", "class", "usage", "roofType", "yearOfConstruction",
                   "measuredHeight", "storeysAboveGround", "storeysBelowGround"}


def _attributes(element) -> dict:
    out: dict = {}
    for child in element:
        local = _tag(child)
        if local in KEEP_ATTRIBUTES and child.text:
            out[local] = child.text.strip()
        elif local in ("stringAttribute", "doubleAttribute", "intAttribute"):
            name = child.get("name")
            value = next((c.text for c in child if _tag(c) == "value"), None)
            if name and value:
                out[name] = value.strip()
    return out


def read_buildings(path: str | Path, *, limit: int | None = None) -> list[Building]:
    """Buildings with their LoD3 boundary surfaces and openings.

    Streaming: each building is released as soon as it is parsed, so a 142 MB
    tile costs its own size rather than a DOM's worth of it.
    """
    buildings: list[Building] = []
    for _, element in ElementTree.iterparse(str(path), events=("end",)):
        if _tag(element) not in ("Building", "BuildingPart"):
            continue
        surfaces: list[Surface] = []
        for candidate in element.iter():
            kind = SURFACE_KIND.get(_tag(candidate))
            if kind is None:
                continue
            surface = Surface(kind, _attr(candidate, "id"))
            # An opening's geometry sits inside the wall's subtree, so collect
            # it first and then exclude it from the wall's own polygons -- a
            # window counted as part of its wall doubles the facade area.
            surface.openings = _openings(candidate)
            opening_ids = {p.gml_id for o in surface.openings for p in o.polygons}
            surface.polygons = [p for p in _polygons(candidate)
                                if p.gml_id not in opening_ids or p.gml_id is None]
            if surface.polygons or surface.openings:
                surfaces.append(surface)
        if surfaces:
            buildings.append(Building(_attr(element, "id"),
                                      _attributes(element), surfaces))
        element.clear()
        if limit is not None and len(buildings) >= limit:
            break
    return buildings


def read_textures(path: str | Path) -> list[TextureBinding]:
    """Every ParameterizedTexture binding in the file.

    Hamburg puts appearances alongside the geometry rather than in a separate
    file, so one pass finds them. The UVs come back per ring in the polygon's
    own order; a target with no coordinates is kept anyway, because knowing
    which image was meant is still evidence.
    """
    bindings: list[TextureBinding] = []
    for _, element in ElementTree.iterparse(str(path), events=("end",)):
        if _tag(element) != "ParameterizedTexture":
            continue
        image = next((c.text.strip() for c in element
                      if _tag(c) == "imageURI" and c.text), None)
        if image:
            for target in element.iter():
                if _tag(target) != "target":
                    continue
                uri = _attr(target, "uri") or target.get("uri") or ""
                uv = [np.fromstring(c.text.strip(), sep=" ", dtype=float).reshape(-1, 2)
                      for c in target.iter()
                      if _tag(c) == "textureCoordinates" and c.text]
                bindings.append(TextureBinding(image, uri.lstrip("#"), uv))
        element.clear()
    return bindings


def summarise(buildings: list[Building], textures: list[TextureBinding]) -> dict:
    """What actually came out, in the terms the repo argues about.

    Deliberately reports openings and texture coverage rather than a triangle
    count: those are the two things Denver could not supply at all, and they are
    what makes this tile worth reading.
    """
    walls = [s for b in buildings for s in b.of("wall")]
    roofs = [s for b in buildings for s in b.of("roof")]
    openings = [o for b in buildings for o in b.openings]
    wall_area = sum(p.area for s in walls for p in s.polygons)
    targets = {t.target for t in textures}
    polygons = [p for b in buildings for s in b.surfaces for p in s.polygons]
    textured = sum(1 for p in polygons if p.gml_id in targets)
    return {
        "buildings": len(buildings),
        "walls": len(walls),
        "roofs": len(roofs),
        "wall_area_m2": round(wall_area, 1),
        "openings": len(openings),
        "windows": sum(1 for o in openings if o.kind == "window"),
        "doors": sum(1 for o in openings if o.kind == "door"),
        "buildings_with_openings": sum(1 for b in buildings if b.openings),
        "texture_bindings": len(textures),
        "distinct_images": len({t.image for t in textures}),
        "polygons": len(polygons),
        "polygons_textured": textured,
        "texture_coverage": round(textured / len(polygons), 4) if polygons else 0.0,
    }

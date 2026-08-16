"""Geometric-role taxonomy.

A role answers "what does this piece of the world *act* like", independently of
what it will eventually look like. Roles are the vocabulary the theme compiler
speaks: a theme pack maps roles (plus context flags) to materials, so the same
Spatial IR can be re-skinned without touching geometry.

Roles are dotted strings so they form a hierarchy and can be matched by prefix:
``surface.wall.vertical`` matches a rule for ``surface.wall`` or ``surface``.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Role:
    id: str
    label: str
    description: str
    #: Which reconstruction strategy claims points with this role.
    reconstruct: str = "surface"
    #: Sensible defaults so an unthemed world is still viewable.
    debug_color: tuple[int, int, int] = (160, 160, 160)
    tags: tuple[str, ...] = ()


def _r(*args, **kwargs) -> Role:
    return Role(*args, **kwargs)


ROLES: dict[str, Role] = {r.id: r for r in [
    # ---- terrain ---------------------------------------------------------
    _r("terrain.ground", "Ground", "Bare-earth surface below everything else.",
       "heightfield", (110, 96, 74), ("walkable",)),
    _r("terrain.road", "Road surface", "Ground that is paved and part of a carriageway.",
       "heightfield", (70, 70, 76), ("walkable", "drivable")),
    _r("terrain.road.edge", "Road edge", "Kerb line where the carriageway ends.",
       "spline", (200, 180, 90), ("linear",)),
    _r("terrain.water", "Water", "Flat, near-zero-return surface at a constant level.",
       "heightfield", (48, 84, 120), ("liquid",)),

    # ---- planar surfaces -------------------------------------------------
    _r("surface.wall.vertical", "Wall", "Vertical planar patch on a structure.",
       "tiled_plane", (196, 184, 164), ("facade",)),
    _r("surface.roof.flat", "Flat roof", "Near-horizontal planar patch above ground level.",
       "tiled_plane", (140, 120, 110), ("roof",)),
    _r("surface.roof.pitched", "Pitched roof", "Sloped planar patch capping a structure.",
       "tiled_plane", (150, 96, 84), ("roof",)),
    _r("surface.slab", "Slab", "Horizontal patch that is neither ground nor roof (deck, balcony).",
       "tiled_plane", (170, 170, 175), ()),

    # ---- creases and corners: the "relative index" family ----------------
    _r("edge.crease.convex", "Convex crease", "Outward fold between two surfaces (building corner).",
       "trim", (240, 190, 90), ("edge",)),
    _r("edge.crease.concave", "Concave crease", "Inward fold between two surfaces (wall meets wall).",
       "trim", (200, 140, 70), ("edge",)),
    _r("edge.boundary.free", "Free boundary", "Patch border with nothing on the other side.",
       "trim", (230, 120, 120), ("edge",)),
    _r("edge.eave", "Eave", "Junction where a roof plane overhangs a wall plane.",
       "trim", (220, 160, 100), ("edge", "roof")),
    _r("edge.ridge", "Ridge", "Junction between two pitched roof planes.",
       "trim", (235, 175, 110), ("edge", "roof")),
    _r("edge.ground_contact", "Ground contact", "Where a wall meets the terrain.",
       "trim", (150, 130, 90), ("edge",)),
    _r("corner.trihedral", "Corner", "Point where three or more surfaces meet.",
       "trim", (255, 230, 120), ("edge", "corner")),

    # ---- openings --------------------------------------------------------
    _r("opening.window", "Window", "Hole in a wall plane, clear of the ground.",
       "opening", (120, 200, 240), ("glazed",)),
    _r("opening.door", "Door", "Hole in a wall plane that reaches the ground.",
       "opening", (170, 130, 90), ("passable",)),
    _r("opening.unknown", "Opening", "Hole in a surface that could not be typed.",
       "opening", (140, 170, 190), ()),

    # ---- volumes and instances -------------------------------------------
    _r("volume.vegetation.high", "Tree", "Scattered volumetric return cluster with a trunk.",
       "instance", (70, 140, 70), ("vegetation",)),
    _r("volume.vegetation.low", "Shrub", "Low scattered vegetation with no clear trunk.",
       "instance", (100, 150, 80), ("vegetation",)),
    _r("volume.building", "Building", "Aggregate node owning walls, roofs and openings.",
       "aggregate", (190, 175, 160), ("structure",)),
    _r("instance.vehicle", "Vehicle", "Compact object cluster on a drivable surface.",
       "instance", (120, 140, 200), ("dynamic",)),
    _r("instance.prop", "Prop", "Small unclassified object cluster.",
       "instance", (150, 150, 160), ()),

    # ---- linear ----------------------------------------------------------
    _r("linear.pole", "Pole", "Thin vertical linear structure.",
       "spline", (180, 180, 190), ("linear",)),
    _r("linear.wire", "Wire", "Thin catenary linear structure above the ground.",
       "spline", (200, 200, 210), ("linear",)),
    _r("linear.fence", "Fence", "Low linear barrier.",
       "spline", (140, 120, 100), ("linear",)),

    # ---- fallback --------------------------------------------------------
    _r("unknown", "Unclassified", "Points no rule could claim.",
       "points", (110, 110, 110), ()),
]}


# ---------------------------------------------------------------------------
# Context bitmask -- per-tile "relative index".
#
# This is what makes "corner wall" and "wall next to a window" addressable by a
# theme rule. Every tile of every reconstructed surface carries this mask, so a
# rule can ask for `surface.wall.vertical` + CORNER_CONVEX + NEAR_OPENING and
# get exactly the tiles that sit on a building corner beside a window.
# ---------------------------------------------------------------------------
class Ctx:
    OCCUPIED = 1 << 0
    EDGE_U_MIN = 1 << 1
    EDGE_U_MAX = 1 << 2
    EDGE_V_MIN = 1 << 3
    EDGE_V_MAX = 1 << 4
    CORNER_CONVEX = 1 << 5
    CORNER_CONCAVE = 1 << 6
    NEAR_OPENING = 1 << 7
    OPENING_BOUNDARY = 1 << 8
    TOP = 1 << 9            # highest run of the patch in world-up
    BOTTOM = 1 << 10        # lowest run of the patch in world-up
    GROUND_CONTACT = 1 << 11
    ADJ_PERPENDICULAR = 1 << 12
    ADJ_ROOF = 1 << 13
    ADJ_COPLANAR = 1 << 14
    INTERIOR = 1 << 15      # "centre of mass" tiles, far from every boundary
    STREET_FACING = 1 << 16
    SHELTERED = 1 << 17     # under an overhang -> weathering rules differ
    SPARSE_EVIDENCE = 1 << 18   # reconstructed from few points; low confidence
    OCCLUDED = 1 << 19      # inferred, never directly observed by the sensor

    NAMES = {
        1 << 0: "occupied", 1 << 1: "edge_u_min", 1 << 2: "edge_u_max",
        1 << 3: "edge_v_min", 1 << 4: "edge_v_max", 1 << 5: "corner_convex",
        1 << 6: "corner_concave", 1 << 7: "near_opening", 1 << 8: "opening_boundary",
        1 << 9: "top", 1 << 10: "bottom", 1 << 11: "ground_contact",
        1 << 12: "adj_perpendicular", 1 << 13: "adj_roof", 1 << 14: "adj_coplanar",
        1 << 15: "interior", 1 << 16: "street_facing", 1 << 17: "sheltered",
        1 << 18: "sparse_evidence", 1 << 19: "occluded",
    }

    BY_NAME = {v: k for k, v in NAMES.items()}

    EDGE_ANY = EDGE_U_MIN | EDGE_U_MAX | EDGE_V_MIN | EDGE_V_MAX

    @classmethod
    def decode(cls, mask: int) -> list[str]:
        return [name for bit, name in sorted(cls.NAMES.items()) if mask & bit]

    @classmethod
    def encode(cls, names) -> int:
        mask = 0
        for n in names:
            if n not in cls.BY_NAME:
                raise KeyError(f"unknown context flag {n!r}")
            mask |= cls.BY_NAME[n]
        return mask


def role_matches(role_id: str, pattern: str) -> bool:
    """Prefix match on the dotted hierarchy. ``*`` matches everything."""
    if pattern in ("*", ""):
        return True
    return role_id == pattern or role_id.startswith(pattern + ".")


def role_or_unknown(role_id: str) -> Role:
    return ROLES.get(role_id, ROLES["unknown"])


ROLE_IDS: list[str] = list(ROLES)
ROLE_INDEX: dict[str, int] = {r: i for i, r in enumerate(ROLE_IDS)}


# ---------------------------------------------------------------------------
# CityGML 3.0 alignment.
#
# CityGML is the closest existing standard to this IR: a platform-independent
# semantic model for 3D urban objects with geometry, hierarchy and LoD. Roles
# are deliberately finer-grained (CityGML has no notion of "the convex crease
# between two walls"), so this maps the coarse part of the taxonomy onto its
# vocabulary and lets the CityJSON backend emit conformant surface semantics.
# See docs/PRIOR_ART.md.
# ---------------------------------------------------------------------------
CITYGML_SURFACE = {
    "surface.wall.vertical": "WallSurface",
    "surface.roof.flat": "RoofSurface",
    "surface.roof.pitched": "RoofSurface",
    "surface.slab": "OuterFloorSurface",
    "terrain.ground": "GroundSurface",
    "terrain.road": "GroundSurface",
    "terrain.water": "WaterSurface",
    "opening.window": "Window",
    "opening.door": "Door",
    "opening.unknown": "Window",
    # CityGML has no vocabulary for creases and corners; they are part of the
    # boundary surface they trim, which is how a conformant reader expects them.
    "edge.crease.convex": "WallSurface",
    "edge.crease.concave": "WallSurface",
    "edge.boundary.free": "WallSurface",
    "edge.ground_contact": "WallSurface",
    "edge.eave": "OuterCeilingSurface",
    "edge.ridge": "RoofSurface",
    "corner.trihedral": "WallSurface",
}

CITYGML_OBJECT = {
    "volume.building": "Building",
    "volume.vegetation.high": "SolitaryVegetationObject",
    "volume.vegetation.low": "SolitaryVegetationObject",
    "linear.pole": "CityFurniture",
    "linear.wire": "CityFurniture",
    "linear.fence": "CityFurniture",
    "instance.vehicle": "GenericCityObject",
    "instance.prop": "GenericCityObject",
    "terrain.ground": "TINRelief",
    "terrain.road": "Road",
    "terrain.water": "WaterBody",
}


def citygml_type(role_id: str, *, surface: bool = True) -> str:
    """Best-matching CityGML class for a role, falling back up the hierarchy."""
    table = CITYGML_SURFACE if surface else CITYGML_OBJECT
    if role_id in table:
        return table[role_id]
    parts = role_id.split(".")
    while len(parts) > 1:
        parts.pop()
        prefix = ".".join(parts)
        if prefix in table:
            return table[prefix]
    return "GenericCityObject"

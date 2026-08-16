"""Per-point role assignment.

Semantics say *what a thing is*; roles say *what a piece of it does*. A building
point is not just "building" -- it is wall interior, or the crease where two
walls meet, or the free rim of a roof. That distinction is the whole point of
the design: theme rules bind to roles, so a re-skin can put weathered quoin
stones on corner tiles without knowing anything about the building.

Roles assigned here are hints at point resolution. Segmentation and topology
later promote them to node roles, where the evidence is much stronger (a patch
knows its own area, slope and neighbours; a single point does not).
"""
from __future__ import annotations

import numpy as np

from ..types import SEMANTIC_INDEX, PointCloud
from .taxonomy import ROLE_IDS, ROLE_INDEX

S = SEMANTIC_INDEX

#: Tuned against the sample tiles; exposed so a caller can retune per dataset.
THRESHOLDS = {
    "crease": 0.30,
    "corner": 0.28,
    "boundary": 0.55,
    "wall_verticality": 0.62,
    "roof_verticality": 0.40,
    "pitched_slope_deg": 12.0,
    "planar": 0.35,
}


def classify(cloud: PointCloud, thresholds: dict | None = None) -> PointCloud:
    cloud.require("semantic", "hag", "planarity", "verticality",
                  "crease_score", "corner_score", "boundary", "normal")
    th = {**THRESHOLDS, **(thresholds or {})}
    n = len(cloud)

    semantic = cloud["semantic"]
    hag = cloud["hag"]
    planarity = cloud["planarity"]
    verticality = cloud["verticality"]
    crease = cloud["crease_score"]
    corner = cloud["corner_score"]
    boundary = cloud["boundary"]
    normal = cloud["normal"]

    role = np.full(n, ROLE_INDEX["unknown"], dtype=np.uint8)
    conf = np.full(n, 0.4, dtype=np.float32)

    def put(mask, role_id, c):
        idx = ROLE_INDEX[role_id]
        take = mask & (role == ROLE_INDEX["unknown"])
        role[take] = idx
        conf[take] = c

    is_structure = np.isin(semantic, [S["building"], S["bridge"]])
    slope_deg = np.degrees(np.arccos(np.clip(np.abs(normal[:, 2]), 0, 1)))

    # --- creases and corners win over the surfaces they separate -----------
    put(is_structure & (corner > th["corner"]), "corner.trihedral", 0.6)
    put(is_structure & (crease > th["crease"]) & (verticality > th["wall_verticality"]),
        "edge.crease.convex", 0.55)
    put(is_structure & (crease > th["crease"]), "edge.ridge", 0.5)
    put(is_structure & (boundary > th["boundary"]) & (verticality < th["roof_verticality"]),
        "edge.eave", 0.5)
    put(is_structure & (boundary > th["boundary"]), "edge.boundary.free", 0.45)

    # --- planar surfaces ---------------------------------------------------
    planar = planarity > th["planar"]
    put(is_structure & planar & (verticality > th["wall_verticality"]) & (hag > 0.8),
        "surface.wall.vertical", 0.75)
    put(is_structure & planar & (verticality < th["roof_verticality"])
        & (slope_deg > th["pitched_slope_deg"]) & (hag > 2.0), "surface.roof.pitched", 0.7)
    put(is_structure & planar & (verticality < th["roof_verticality"]) & (hag > 2.0),
        "surface.roof.flat", 0.7)
    put(is_structure & planar & (verticality < th["roof_verticality"]), "surface.slab", 0.5)
    put(is_structure, "surface.wall.vertical", 0.4)

    # --- terrain -----------------------------------------------------------
    put(semantic == S["road"], "terrain.road", 0.8)
    put(semantic == S["water"], "terrain.water", 0.7)
    put(semantic == S["ground"], "terrain.ground", 0.8)

    # --- volumes, linears, instances --------------------------------------
    put(semantic == S["vegetation_high"], "volume.vegetation.high", 0.7)
    put(semantic == S["vegetation_low"], "volume.vegetation.low", 0.65)
    put(semantic == S["pole"], "linear.pole", 0.7)
    put(semantic == S["wire"], "linear.wire", 0.65)
    put(semantic == S["fence"], "linear.fence", 0.6)
    put(semantic == S["vehicle"], "instance.vehicle", 0.6)
    put(semantic == S["person"], "instance.prop", 0.5)

    if "semantic_confidence" in cloud:
        conf = (conf * (0.5 + 0.5 * cloud["semantic_confidence"])).astype(np.float32)

    cloud["role"] = role
    cloud["role_confidence"] = conf
    return cloud


def role_histogram(cloud: PointCloud) -> dict[str, int]:
    role = cloud.get("role")
    if role is None:
        return {}
    counts = np.bincount(role, minlength=len(ROLE_IDS))
    return {name: int(c) for name, c in zip(ROLE_IDS, counts) if c}

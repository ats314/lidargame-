"""Simple tree geometry for generated blocks.

A tree is a trunk (box) and a crown (faceted cone or sphere approximation).
Good enough to break a street's sight line and cast a shadow; not good enough
to mistake for a scan. That is exactly the fidelity budget a seed-to-world
pipeline should spend on a tree: the silhouette matters, the bark does not.
"""
from __future__ import annotations

import numpy as np


def _circle(n: int, radius: float, z: float, centre=(0.0, 0.0)) -> np.ndarray:
    """N points on a circle at height z."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([
        centre[0] + radius * np.cos(angles),
        centre[1] + radius * np.sin(angles),
        np.full(n, z),
    ])


def tree_geometry(x: float, y: float, ground_z: float, *,
                  height: float = 10.0, crown_radius: float = 3.0,
                  trunk_radius: float = 0.25, crown_start: float = 0.4,
                  facets: int = 8) -> tuple[list[np.ndarray], list[str]]:
    """Quads and kinds for one tree at (x, y, ground_z).

    Returns (quads, kinds) in the same format `elevation.Elevation` uses,
    so they can be appended directly to a face list.

    `crown_start` is the fraction of total height where the crown begins.
    """
    quads, kinds = [], []
    crown_base_z = ground_z + height * crown_start
    apex_z = ground_z + height

    # Trunk: a faceted cylinder
    lo = _circle(facets, trunk_radius, ground_z, (x, y))
    hi = _circle(facets, trunk_radius, crown_base_z, (x, y))
    for i in range(facets):
        j = (i + 1) % facets
        quads.append(np.array([lo[i], lo[j], hi[j], hi[i]]))
        kinds.append("trunk")

    # Crown: a faceted cone
    crown_lo = _circle(facets, crown_radius, crown_base_z, (x, y))
    apex = np.array([x, y, apex_z])
    # Lower ring of the crown — wider at the base
    mid_z = crown_base_z + (apex_z - crown_base_z) * 0.35
    crown_mid = _circle(facets, crown_radius * 0.85, mid_z, (x, y))

    # Crown side panels: base to mid
    for i in range(facets):
        j = (i + 1) % facets
        quads.append(np.array([crown_lo[i], crown_lo[j], crown_mid[j], crown_mid[i]]))
        kinds.append("foliage")

    # Crown side panels: mid to apex (triangles as degenerate quads)
    for i in range(facets):
        j = (i + 1) % facets
        quads.append(np.array([crown_mid[i], crown_mid[j], apex, apex]))
        kinds.append("foliage")

    # Crown bottom cap
    for i in range(facets):
        j = (i + 1) % facets
        centre = np.array([x, y, crown_base_z])
        quads.append(np.array([crown_lo[j], crown_lo[i], centre, centre]))
        kinds.append("foliage")

    return quads, kinds


def street_trees(footprint_rings: list[np.ndarray], ground_z: float, *,
                 spacing: float = 12.0, setback: float = 6.0,
                 height_range: tuple[float, float] = (8.0, 14.0),
                 crown_range: tuple[float, float] = (2.5, 4.0),
                 seed: int = 0) -> list[dict]:
    """Place trees along the longest edges of a block's footprints.

    Returns a list of tree specs that `tree_geometry` can expand. Trees are
    placed `setback` metres from the building face, which puts them roughly
    at the kerb on a Helsinki street.
    """
    rng = np.random.default_rng(seed)
    trees = []

    for ring in footprint_rings:
        ring = np.asarray(ring, dtype=float)
        if len(ring) < 3:
            continue
        # Find the two longest edges (the street frontages)
        closed = np.vstack([ring, ring[:1]])
        edges = [(i, float(np.hypot(*(closed[i+1] - closed[i])[:2])))
                 for i in range(len(closed) - 1)]
        edges.sort(key=lambda e: -e[1])

        for edge_idx, length in edges[:2]:
            if length < spacing * 2:
                continue
            a = closed[edge_idx][:2]
            b = closed[edge_idx + 1][:2]
            along = b - a
            along_n = along / max(np.hypot(*along), 1e-9)
            # Normal pointing outward (to the right of travel direction)
            outward = np.array([along_n[1], -along_n[0]])

            n_trees = max(1, int(length / spacing))
            for k in range(n_trees):
                t = (k + 0.5) / n_trees
                pos = a + along * t + outward * setback
                h = rng.uniform(*height_range)
                cr = rng.uniform(*crown_range)
                trees.append({
                    "x": float(pos[0]), "y": float(pos[1]),
                    "ground_z": ground_z,
                    "height": round(h, 1),
                    "crown_radius": round(cr, 1),
                })
    return trees

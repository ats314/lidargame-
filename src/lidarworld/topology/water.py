"""Canals, which airborne LiDAR describes as an absence.

Water is specular at 1064 nm. A pulse that hits a canal at anything but nadir
leaves and does not come back, so the returns describe Amsterdam as a city with
holes in it -- 400 m of the canal belt compiled with not one point classified
water, and the canals came out as VOID cells the mesher correctly refused to
surface. Correct, and unwalkable: the ground simply stops.

The surveyed polygon says the hole is a canal, which is the same move
`--streets` makes for the carriageway: the compiler does not invent surface, it
accepts an authoritative statement about what the surface *is* and fills only
where its own returns already gave it a level to fill to.

Two things are measured and one is inferred, and they are kept apart:

    measured    the polygon (BGT, surveyed) and the quay height around it,
                which is terrain the scan did see
    inferred    the water surface itself, placed a drop below the lowest quay
                because no return came back from it

That last number is the honest weakness here. A canal's water level is not
derivable from a scan that never saw the water, so it is a constant offset from
the measured bank, recorded as inferred and carried into the seed as such. It
is right to about the depth of a step, which is enough to walk beside and not
enough to boat on.
"""
from __future__ import annotations

import numpy as np

#: Metres below the lowest measured bank cell. Amsterdam's canals sit roughly
#: 0.4 m below NAP with a quay about a metre above that; this is deliberately
#: shallow, because a water plane set too low reads as a trench and a plane set
#: too high floods the street.
QUAY_DROP = 0.9


def polygons(geojson: dict) -> list[np.ndarray]:
    """Exterior rings of every (Multi)Polygon, as (N,2) arrays."""
    rings = []
    for feature in geojson.get("features", []):
        geometry = feature.get("geometry") or {}
        kind = geometry.get("type")
        if kind == "Polygon":
            rings.append(np.asarray(geometry["coordinates"][0], dtype=np.float64)[:, :2])
        elif kind == "MultiPolygon":
            for part in geometry["coordinates"]:
                rings.append(np.asarray(part[0], dtype=np.float64)[:, :2])
    return [ring for ring in rings if len(ring) >= 4]


def rasterise(rings, raster) -> np.ndarray:
    """Boolean mask of cells whose centre falls inside any ring."""
    from ..data.gis import point_in_polygon

    mask = np.zeros(raster.shape, dtype=bool)
    if not rings:
        return mask
    gx, gy = raster.cell_centers()
    xx, yy = np.meshgrid(gx, gy, indexing="ij")
    centres = np.column_stack([xx.ravel(), yy.ravel()])
    for ring in rings:
        lo, hi = ring.min(axis=0), ring.max(axis=0)
        # Skip the whole ring on a bbox miss: a canal network over a block is a
        # few dozen polygons and most of them do not reach most of the raster.
        if (lo[0] > gx[-1] or hi[0] < gx[0] or lo[1] > gy[-1] or hi[1] < gy[0]):
            continue
        closed = ring if np.allclose(ring[0], ring[-1]) else np.vstack([ring, ring[:1]])
        mask |= point_in_polygon(centres, closed).reshape(raster.shape)
    return mask


def bank_level(dtm: np.ndarray, mask: np.ndarray, *, percentile: float = 10.0):
    """Height of the bank around a water body, from measured terrain only.

    The ring of cells just outside the polygon is quay, and the scan saw it.
    Taking a low percentile rather than the median keeps a bridge deck or a
    moored barge from lifting the whole canal.
    """
    if not mask.any():
        return None
    ring = np.zeros_like(mask)
    for axis in (0, 1):
        for shift in (-1, 1):
            ring |= np.roll(mask, shift, axis=axis)
    ring &= ~mask
    values = dtm[ring]
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    return float(np.percentile(values, percentile))


def apply(class_raster: np.ndarray, dtm: np.ndarray, mask: np.ndarray, *,
          water: int = 2, void: int = 255, drop: float = QUAY_DROP) -> dict:
    """Fill the canals: class the masked cells water and give them a level.

    Only VOID and already-water cells are taken. A cell inside the polygon that
    carries measured ground is a bridge, a houseboat or a quay the polygon
    overshoots, and overwriting it would drown real surface.
    """
    level = bank_level(dtm, mask)
    eligible = mask & ((class_raster == void) | (class_raster == water))
    if level is None or not eligible.any():
        return {"cells": 0, "level_m": None,
                "polygon_cells": int(mask.sum()),
                "cells_left_as_measured": int((mask & ~eligible).sum())}

    surface = level - drop
    class_raster[eligible] = water
    dtm[eligible] = surface
    return {
        "cells": int(eligible.sum()),
        "polygon_cells": int(mask.sum()),
        "cells_left_as_measured": int((mask & ~eligible).sum()),
        "bank_level_m": round(level, 2),
        "level_m": round(surface, 2),
        "drop_m": drop,
        "epistemic": "surface inferred: no return came back from the water",
    }

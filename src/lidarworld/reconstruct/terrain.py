"""Terrain surface classification.

The DTM says how high the ground is; this says what the ground *is*. Road,
bare ground and water each get their own cells so the theme compiler can put
asphalt on one, grass on another and a water shader on the third -- and so the
boundary between them can carry a kerb.
"""
from __future__ import annotations

import numpy as np

from ..spatial.grid import Raster2D, box_blur
from ..types import SEMANTIC_INDEX, PointCloud

S = SEMANTIC_INDEX

#: Terrain cell codes used in the class raster.
GROUND, ROAD, WATER, VOID = 0, 1, 2, 255

ROLE_LOOKUP = {GROUND: "terrain.ground", ROAD: "terrain.road", WATER: "terrain.water"}


def classify_cells(cloud: PointCloud, raster: Raster2D, dtm: np.ndarray, *,
                   min_votes: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Majority-vote each terrain cell from the points that landed on it.

    Returns ``(class_raster, coverage)`` where coverage counts near-ground
    points per cell -- cells with none are marked VOID and left unmeshed rather
    than invented.
    """
    hag = cloud.get("hag")
    semantic = cloud.get("semantic")
    near_ground = np.ones(len(cloud), dtype=bool) if hag is None else (hag < 0.6)

    shape = (raster.nx, raster.ny)
    votes = {code: np.zeros(shape) for code in (GROUND, ROAD, WATER)}
    for code, classes in ((ROAD, (S["road"],)), (WATER, (S["water"],)),
                          (GROUND, (S["ground"], S["vegetation_low"]))):
        mask = near_ground & np.isin(semantic, classes) if semantic is not None else near_ground
        if not mask.any():
            continue
        votes[code] = np.nan_to_num(
            raster.accumulate(cloud.xyz[mask], None, how="count"), nan=0.0)

    coverage = sum(votes.values())
    stacked = np.stack([votes[GROUND], votes[ROAD], votes[WATER]])
    winner = np.argmax(stacked, axis=0).astype(np.uint8)
    class_raster = np.where(coverage >= min_votes, winner, VOID).astype(np.uint8)

    # Fill isolated voids that are surrounded by real terrain: a scan shadow
    # under a car should still be road, but the empty half of the tile should
    # not become invented ground.
    class_raster = _fill_small_voids(class_raster, coverage, max_radius=2)
    class_raster = majority_filter(class_raster)
    return class_raster, coverage


def majority_filter(class_raster: np.ndarray, passes: int = 1) -> np.ndarray:
    """3x3 mode filter.

    Per-cell voting is noisy at surface boundaries, and every stray cell inside
    a carriageway becomes a spurious kerb once EDGE_ANY is derived from class
    changes. Smoothing the classification first keeps kerbs where kerbs are.
    """
    out = class_raster.copy()
    codes = [c for c in np.unique(out) if c != VOID]
    if len(codes) < 2:
        return out
    for _ in range(passes):
        counts = []
        for code in codes:
            m = (out == code).astype(np.int16)
            padded = np.pad(m, 1, mode="edge")
            acc = np.zeros_like(m)
            for di in range(3):
                for dj in range(3):
                    acc += padded[di:di + m.shape[0], dj:dj + m.shape[1]]
            counts.append(acc)
        stacked = np.stack(counts)
        winner = np.take(np.asarray(codes, dtype=np.uint8), np.argmax(stacked, axis=0))
        out = np.where(out == VOID, VOID, winner)
    return out


def _fill_small_voids(class_raster: np.ndarray, coverage: np.ndarray, max_radius: int = 2) -> np.ndarray:
    out = class_raster.copy()
    for _ in range(max_radius):
        void = out == VOID
        if not void.any():
            break
        neighbours = []
        for du, dv in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rolled = np.roll(out, (du, dv), axis=(0, 1))
            neighbours.append(np.where(rolled == VOID, -1, rolled.astype(np.int16)))
        stack = np.stack(neighbours)
        valid = stack >= 0
        filled = np.where(valid.sum(axis=0) >= 3,
                          np.max(np.where(valid, stack, -1), axis=0), -1)
        out = np.where(void & (filled >= 0), filled.astype(np.uint8), out)
    return out


def smooth_terrain(dtm: np.ndarray, class_raster: np.ndarray, *, road_passes: int = 2) -> np.ndarray:
    """Flatten carriageways slightly -- real roads are smoother than a min-raster."""
    out = dtm.copy()
    road = class_raster == ROAD
    if road.any() and road_passes:
        blurred = box_blur(np.nan_to_num(out, nan=float(np.nanmedian(out))), 1, road_passes)
        out[road] = blurred[road]
    return out


def road_mask(class_raster: np.ndarray) -> np.ndarray:
    return class_raster == ROAD

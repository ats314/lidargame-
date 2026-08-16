"""Bare-earth extraction and height above ground.

Height above ground (HAG) is the single most discriminative channel in the whole
pipeline -- it is what separates a road from a roof that happens to be equally
flat, and a shrub from a tree canopy. Everything downstream leans on it, so it
runs before role inference.

Two paths:

* If the source already labels ground (ASPRS class 2/11, SemanticKITTI road and
  sidewalk), those points define the terrain directly. Airborne tiles almost
  always take this path, and it is far more reliable than any filter.
* Otherwise a progressive morphological filter (Zhang et al. 2003) runs on the
  minimum-elevation raster: opening with a growing window removes objects
  smaller than the window, and anything that drops by more than the allowed
  slope over that window is not terrain.
"""
from __future__ import annotations

import numpy as np

from ..spatial.grid import Raster2D, box_blur, fill_holes
from ..types import SEMANTIC_INDEX, PointCloud

try:                                        # optional, ~10x faster morphology
    from scipy import ndimage as _ndi
except ImportError:                         # pragma: no cover
    _ndi = None

GROUNDISH = (SEMANTIC_INDEX["ground"], SEMANTIC_INDEX["road"], SEMANTIC_INDEX["water"])


def _erode(field: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return field
    if _ndi is not None:
        return _ndi.grey_erosion(field, size=2 * radius + 1, mode="nearest")
    out = field.copy()
    for axis in (0, 1):
        pad = np.pad(out, radius, mode="edge")
        stack = [np.roll(pad, k, axis=axis) for k in range(-radius, radius + 1)]
        out = np.min(np.stack(stack), axis=0)[radius:-radius, radius:-radius]
    return out


def _dilate(field: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return field
    if _ndi is not None:
        return _ndi.grey_dilation(field, size=2 * radius + 1, mode="nearest")
    out = field.copy()
    for axis in (0, 1):
        pad = np.pad(out, radius, mode="edge")
        stack = [np.roll(pad, k, axis=axis) for k in range(-radius, radius + 1)]
        out = np.max(np.stack(stack), axis=0)[radius:-radius, radius:-radius]
    return out


def progressive_morphological(zmin: np.ndarray, cell: float, *, max_window: float = 24.0,
                              slope: float = 0.25, initial_dh: float = 0.35,
                              max_dh: float = 4.0) -> np.ndarray:
    """Iteratively open the surface until only terrain survives."""
    surface = fill_holes(zmin)
    terrain = surface.copy()
    window = 1
    while window * cell <= max_window:
        opened = _dilate(_erode(terrain, window), window)
        dh = min(initial_dh + slope * (2 * window * cell), max_dh)
        removed = (terrain - opened) > dh
        terrain = np.where(removed, opened, terrain)
        window *= 2
    return terrain


def estimate(cloud: PointCloud, *, cell: float = 1.0, smooth: int = 1,
             prefer_labels: bool = True) -> tuple[Raster2D, np.ndarray]:
    """Build a digital terrain model and attach the ``hag`` channel.

    Returns ``(raster, dtm)``; ``cloud['hag']`` is set as a side effect.
    """
    lo, hi = cloud.bounds
    raster = Raster2D(lo, hi, cell)

    semantic = cloud.get("semantic")
    labelled = None
    if prefer_labels and semantic is not None:
        mask = np.isin(semantic, GROUNDISH)
        if mask.sum() >= max(64, 0.01 * len(cloud)):
            labelled = mask

    if labelled is not None:
        dtm = raster.accumulate(cloud.xyz[labelled], cloud.xyz[labelled, 2], how="min")
        dtm = fill_holes(dtm)
        method = "labelled ground points"
    else:
        zmin = raster.accumulate(cloud.xyz, cloud.z, how="min")
        dtm = progressive_morphological(zmin, cell)
        method = "progressive morphological filter"

    if smooth:
        dtm = box_blur(dtm, radius=1, passes=smooth)

    hag = cloud.z - raster.sample_bilinear(dtm, cloud.xyz)
    hag = np.nan_to_num(hag, nan=0.0)
    cloud["hag"] = hag.astype(np.float32)
    cloud.meta["dtm_method"] = method
    cloud.meta["dtm_cell"] = float(cell)
    return raster, dtm


def canopy_height_model(cloud: PointCloud, raster: Raster2D, dtm: np.ndarray) -> np.ndarray:
    """Max height above terrain per cell -- the input to tree instancing."""
    zmax = raster.accumulate(cloud.xyz, cloud.z, how="max")
    return np.where(np.isfinite(zmax), zmax - dtm, 0.0)

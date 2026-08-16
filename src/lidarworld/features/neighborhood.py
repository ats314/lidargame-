"""Multiscale neighbourhood descriptors.

This is the stage that turns undifferentiated dots into *roles*. Every point
gets, at several spatial scales, the shape of the neighbourhood around it --
and, more importantly, where it sits within that shape: deep in a flat region,
on a crease between two flat regions, on a corner where three meet, or on the
free rim of a surface where the scan simply ran out.

Two tensors do the work:

* **Position covariance** gives the classic Weinmann/Demantke dimensionality
  descriptors (linearity / planarity / sphericity / curvature) and the normal.
* **Normal tensor** ``T = sum(n n^T)`` over a neighbourhood is sign-invariant,
  which matters because eigenvector normals have arbitrary sign. Its spectrum
  separates the three cases cleanly: one dominant eigenvalue means one surface,
  two means a crease, three means a corner.

Descriptors are computed per voxel and broadcast to the points inside, which
keeps the whole thing linear in point count.
"""
from __future__ import annotations

import numpy as np

from ..spatial.grid import (build_voxel_index, covariance_from_moments,
                            eigen_sorted, voxel_moments)
from ..types import PointCloud

#: Fine / structural / contextual. Roughly: surface detail, wall-vs-crease,
#: which object this belongs to.
DEFAULT_SCALES = (0.5, 1.5, 4.0)
EPS = 1e-12


def _dimensionality(vals: np.ndarray) -> dict[str, np.ndarray]:
    l0 = np.maximum(vals[:, 0], EPS)
    l1, l2 = vals[:, 1], vals[:, 2]
    total = np.maximum(vals.sum(axis=1), EPS)
    return {
        "linearity": (l0 - l1) / l0,
        "planarity": (l1 - l2) / l0,
        "sphericity": l2 / l0,
        "anisotropy": (l0 - l2) / l0,
        "curvature": l2 / total,
    }


def scale_features(xyz: np.ndarray, scale: float) -> dict[str, np.ndarray]:
    """Per-point descriptors at one scale."""
    vi = build_voxel_index(xyz, scale)
    moments = voxel_moments(xyz, vi)
    block = vi.neighbor_sum(moments)                 # 3x3x3 -> ~1.5*scale radius
    mean, cov, count = covariance_from_moments(block)
    vals, vecs = eigen_sorted(cov)

    normals = vecs[:, :, 2].copy()                   # smallest-eigenvalue direction
    flip = normals[:, 2] < 0                         # canonical hemisphere (+z)
    normals[flip] *= -1

    feats = _dimensionality(vals)
    feats["normal"] = normals
    feats["verticality"] = 1.0 - np.abs(normals[:, 2])
    feats["density"] = count / (scale ** 3 * 27.0)

    # Boundary: on the rim of a surface the neighbourhood centroid is pulled
    # inward, so the offset from the voxel centre approaches the search radius.
    centers = vi.centers()
    offset = np.linalg.norm(mean - centers, axis=1)
    feats["boundary"] = np.clip(offset / (1.5 * scale), 0.0, 1.0)

    # Normal tensor over the same block -> surface / crease / corner split.
    weights = vi.counts.astype(np.float64)
    nw = normals * weights[:, None]
    tensor_cols = np.column_stack([
        np.zeros(len(vi.keys)),                      # placeholder for count column
        nw[:, 0] * normals[:, 0], nw[:, 0] * normals[:, 1], nw[:, 0] * normals[:, 2],
        nw[:, 1] * normals[:, 1], nw[:, 1] * normals[:, 2], nw[:, 2] * normals[:, 2],
    ])
    tensor_cols[:, 0] = weights
    block_t = vi.neighbor_sum(tensor_cols)
    w = np.maximum(block_t[:, 0], EPS)
    T = np.empty((len(w), 3, 3))
    T[:, 0, 0] = block_t[:, 1] / w
    T[:, 0, 1] = T[:, 1, 0] = block_t[:, 2] / w
    T[:, 0, 2] = T[:, 2, 0] = block_t[:, 3] / w
    T[:, 1, 1] = block_t[:, 4] / w
    T[:, 1, 2] = T[:, 2, 1] = block_t[:, 5] / w
    T[:, 2, 2] = block_t[:, 6] / w
    tv, _ = eigen_sorted(T)
    tsum = np.maximum(tv.sum(axis=1), EPS)
    feats["surface_score"] = tv[:, 0] / tsum
    feats["crease_score"] = np.clip(3.0 * tv[:, 1] / tsum, 0.0, 1.0)
    feats["corner_score"] = np.clip(6.0 * tv[:, 2] / tsum, 0.0, 1.0)

    return {k: v[vi.point_voxel] for k, v in feats.items()}


def compute(cloud: PointCloud, scales=DEFAULT_SCALES, primary: int = 1) -> PointCloud:
    """Attach multiscale descriptors to `cloud`. Returns the same cloud.

    Channels named without a suffix come from the *primary* scale (index 1 by
    default, the structural scale). Per-scale copies are kept with a ``@`` suffix
    so cross-scale contrasts stay available to later stages and to debugging.
    """
    xyz = cloud.xyz
    per_scale = []
    for s in scales:
        feats = scale_features(xyz, float(s))
        per_scale.append(feats)
        tag = f"@{s:g}"
        for name in ("planarity", "linearity", "sphericity", "curvature",
                     "verticality", "density", "boundary", "crease_score",
                     "corner_score", "surface_score"):
            cloud[name + tag] = feats[name].astype(np.float32)

    base = per_scale[min(primary, len(per_scale) - 1)]
    for name, value in base.items():
        cloud[name] = value.astype(np.float32)

    # Cross-scale contrast: structure that is planar when you zoom out but not
    # when you zoom in is detail sitting on a flat surface (a sill, a balcony,
    # a parked car against a facade) rather than a genuine crease.
    if len(per_scale) >= 2:
        fine, coarse = per_scale[0], per_scale[-1]
        cloud["scale_contrast"] = (coarse["planarity"] - fine["planarity"]).astype(np.float32)
        cloud["local_detail"] = np.clip(
            fine["sphericity"] - coarse["sphericity"] + 0.5, 0, 1).astype(np.float32)

    cloud.meta["feature_scales"] = list(map(float, scales))
    cloud.meta["primary_scale"] = float(scales[min(primary, len(scales) - 1)])
    return cloud

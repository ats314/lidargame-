"""Object instancing: trees, vehicles, poles, props.

Volumetric and linear things are never reconstructed literally. A tree is a few
thousand scattered returns and no amount of meshing makes that look like a tree
-- but the returns do tell you where the trunk is, how tall the crown is and how
wide it spreads, and those three numbers drive a procedural or authored asset
that looks right in any theme. Same for poles and vehicles.

So this stage answers "how many, where, how big", and leaves "what does it look
like" entirely to the theme compiler.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..spatial.grid import Raster2D, build_voxel_index
from ..types import SEMANTIC_INDEX, PointCloud

S = SEMANTIC_INDEX


@dataclass
class Instance:
    id: int
    role: str
    center: np.ndarray            # x, y, base z
    size: np.ndarray              # radius_x, radius_y, height
    support: int
    confidence: float = 0.5
    axis: np.ndarray | None = None
    point_idx: np.ndarray | None = None
    attrs: dict = field(default_factory=dict)


def _connected_components(vi, min_voxels: int) -> np.ndarray:
    """Label occupied voxels by 26-connectivity. Returns per-voxel label."""
    m = vi.n_voxels
    offsets = [o for o in vi.neighbor_offsets(1) if o.any()]
    neighbors = np.full((m, len(offsets)), -1, dtype=np.int64)
    for k, off in enumerate(offsets):
        neighbors[:, k] = vi.lookup(vi.ijk + off)

    label = np.full(m, -1, dtype=np.int32)
    next_label = 0
    for start in range(m):
        if label[start] >= 0:
            continue
        stack = [start]
        label[start] = next_label
        size = 0
        while stack:
            cur = stack.pop()
            size += 1
            for nb in neighbors[cur]:
                if nb >= 0 and label[nb] < 0:
                    label[nb] = next_label
                    stack.append(int(nb))
        if size < min_voxels:
            label[label == next_label] = -2          # too small: drop
        else:
            next_label += 1
    return label


def cluster(cloud: PointCloud, classes: tuple[str, ...], role: str, *,
            voxel: float = 0.8, min_points: int = 24, min_voxels: int = 3,
            start_id: int = 0) -> list[Instance]:
    """Euclidean clustering of one semantic group into object instances."""
    semantic = cloud["semantic"]
    mask = np.isin(semantic, [S[c] for c in classes])
    idx = np.flatnonzero(mask)
    if idx.size < min_points:
        return []

    xyz = cloud.xyz[idx]
    vi = build_voxel_index(xyz, voxel)
    labels = _connected_components(vi, min_voxels)
    per_point = labels[vi.point_voxel]

    out: list[Instance] = []
    for lab in np.unique(per_point):
        if lab < 0:
            continue
        sel = per_point == lab
        if sel.sum() < min_points:
            continue
        pts = xyz[sel]
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        center = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
        size = np.array([(hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2, hi[2] - lo[2]])
        support = int(sel.sum())
        out.append(Instance(
            id=start_id + len(out), role=role, center=center, size=size, support=support,
            confidence=float(np.clip(0.3 + 0.5 * min(1.0, support / 250.0), 0.1, 0.95)),
            point_idx=idx[sel],
        ))
    return out


def trees(cloud: PointCloud, raster: Raster2D, chm: np.ndarray, *,
          min_height: float = 2.5, start_id: int = 0) -> list[Instance]:
    """Individual trees from canopy-height local maxima.

    Plain clustering merges a whole treeline into one blob. Local maxima on the
    canopy height model split it back into stems, which is what an instanced
    renderer needs. Crown radius follows the usual allometric rule of thumb
    (roughly a quarter of height), clamped by the local canopy footprint.
    """
    semantic = cloud["semantic"]
    mask = semantic == S["vegetation_high"]
    idx = np.flatnonzero(mask)
    if idx.size < 30:
        return []

    smooth = chm.copy()
    pad = np.pad(smooth, 1, mode="edge")
    stack = np.stack([pad[di:di + smooth.shape[0], dj:dj + smooth.shape[1]]
                      for di in range(3) for dj in range(3)])
    local_max = stack.max(axis=0)
    is_peak = (smooth >= local_max - 1e-6) & (smooth > min_height)

    peaks = np.argwhere(is_peak)
    if peaks.size == 0:
        return []
    gx, gy = raster.cell_centers()
    xy = cloud.xyz[idx, :2]

    out: list[Instance] = []
    for pi, pj in peaks:
        height = float(smooth[pi, pj])
        cx, cy = float(gx[pi]), float(gy[pj])
        radius = float(np.clip(height * 0.26, 0.9, 6.0))
        near = np.flatnonzero(((xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2) < radius ** 2)
        if near.size < 20:
            continue
        pts = cloud.xyz[idx[near]]
        base = float(np.percentile(pts[:, 2], 5))
        out.append(Instance(
            id=start_id + len(out), role="volume.vegetation.high",
            center=np.array([cx, cy, base]),
            size=np.array([radius, radius, height]),
            support=int(near.size),
            confidence=float(np.clip(0.35 + 0.5 * min(1.0, near.size / 300.0), 0.15, 0.95)),
            point_idx=idx[near],
            attrs={"crown_radius": radius, "canopy_height": height},
        ))
    return out


def poles(cloud: PointCloud, *, start_id: int = 0) -> list[Instance]:
    """Vertical linear structures, fitted as an axis plus a height."""
    found = cluster(cloud, ("pole",), "linear.pole", voxel=0.7, min_points=12,
                    min_voxels=2, start_id=start_id)
    for inst in found:
        inst.axis = np.array([0.0, 0.0, 1.0])
        inst.attrs["radius"] = float(max(0.08, min(inst.size[0], inst.size[1])))
    return found

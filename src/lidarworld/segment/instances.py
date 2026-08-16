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

#: A candidate this much lower than a stem within this multiple of its crown is
#: the same crown's flank, not a neighbouring tree.
SHOULDER_REACH = 1.6
SHOULDER_RATIO = 0.66


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


def crown_radius(height, *, slope: float = 0.16, intercept: float = 0.6,
                 lo: float = 1.2, hi: float = 7.0):
    """Allometric crown radius from canopy height, in metres.

    Roughly the Popescu/Wynne relation for mixed urban canopy: crowns widen with
    height but sub-linearly. The clamps stop a noise spike from claiming a 14 m
    crown, and stop a real oak from being searched with a 1 m window.
    """
    return np.clip(slope * np.asarray(height, dtype=float) + intercept, lo, hi)


def _smooth_chm(chm: np.ndarray, sigma_cells: float) -> np.ndarray:
    """Separable Gaussian blur. Raw CHM noise is what invents phantom stems."""
    if sigma_cells <= 0.05:
        return chm
    radius = max(1, int(round(sigma_cells * 2)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma_cells) ** 2)
    kernel /= kernel.sum()
    out = chm.astype(float)
    for axis in (0, 1):
        pad = [(radius, radius) if a == axis else (0, 0) for a in (0, 1)]
        out = np.apply_along_axis(lambda line: np.convolve(line, kernel, mode="valid"),
                                  axis, np.pad(out, pad, mode="edge"))
    return out


def _stems(smooth: np.ndarray, raster: Raster2D, min_height: float, max_height: float):
    """Variable-window canopy maxima with non-maximum suppression.

    A fixed 3x3 window invents a tree per raster cell: canopy noise makes most
    cells a local maximum of their neighbours, and on a flat plateau *every*
    tied cell qualifies. Instead a cell must win against everything inside its
    own allometric crown, and accepted stems suppress later ones within it.
    """
    # `max_height` is a physical prior, not a tuning knob: no tree is 74 m tall
    # in a city, so a canopy maximum that high is a building the semantics got
    # wrong. Excluding only the over-tall cells is not enough -- the flanks of a
    # tall structure fall back under the cap and become rim trees around it --
    # so the rejection is grown by a crown's width first.
    too_tall = smooth >= max_height
    if too_tall.any():
        reach = int(np.ceil(crown_radius(max_height) / max(raster.cell, 1e-6)))
        pad = np.pad(too_tall, reach, mode="constant", constant_values=False)
        span = 2 * reach + 1
        grown = np.zeros_like(too_tall)
        for di in range(span):
            for dj in range(span):
                grown |= pad[di:di + too_tall.shape[0], dj:dj + too_tall.shape[1]]
        too_tall = grown

    tall = np.argwhere((smooth > min_height) & ~too_tall)
    if tall.size == 0:
        return np.empty((0, 2)), np.empty(0), np.empty(0)

    # Tallest first, so a stem is only ever suppressed by something at least as high.
    heights = smooth[tall[:, 0], tall[:, 1]]
    order = np.argsort(-heights, kind="stable")
    tall, heights = tall[order], heights[order]
    radii = crown_radius(heights)

    gx, gy = raster.cell_centers()
    candidates = np.column_stack([gx[tall[:, 0]], gy[tall[:, 1]]])

    keep_xy, keep_h, keep_r = [], [], []
    for k in range(len(candidates)):
        if keep_xy:
            d2 = ((np.asarray(keep_xy) - candidates[k]) ** 2).sum(axis=1)
            kr, kh = np.asarray(keep_r), np.asarray(keep_h)
            # Two stems whose crowns would overlap cannot be told apart from
            # above, so they are one tree. Comparing against the larger radius
            # alone lets crowns interpenetrate and splits one canopy into
            # several -- the sum is the geometric statement that matters.
            overlapping = d2 < (kr + radii[k]) ** 2
            # And a candidate far lower than a nearby stem is that stem's
            # shoulder, not a neighbour: real crowns are wider than allometry
            # where the tree is old or open-grown, and their flanks otherwise
            # ring the true peak with phantom stems.
            shoulder = (d2 < (kr * SHOULDER_REACH) ** 2) & (heights[k] < SHOULDER_RATIO * kh)
            if np.any(overlapping | shoulder):
                continue
        keep_xy.append(candidates[k])
        keep_h.append(float(heights[k]))
        keep_r.append(float(radii[k]))
    return np.asarray(keep_xy), np.asarray(keep_h), np.asarray(keep_r)


def _nearest_stem(xy: np.ndarray, stems: np.ndarray):
    """Index of and squared distance to the closest stem, chunked to bound memory."""
    nearest = np.zeros(len(xy), np.int64)
    best = np.full(len(xy), np.inf)
    for s, stem in enumerate(stems):
        dist = ((xy - stem) ** 2).sum(axis=1)
        closer = dist < best
        best[closer] = dist[closer]
        nearest[closer] = s
    return nearest, best


def trees(cloud: PointCloud, raster: Raster2D, chm: np.ndarray, *,
          min_height: float = 2.5, max_height: float = 40.0, min_points: int = 20,
          smooth_sigma: float = 0.8, start_id: int = 0) -> list[Instance]:
    """Individual trees from canopy-height maxima, one stem per crown.

    Plain clustering merges a whole treeline into one blob, so stems come from
    maxima on the canopy height model -- but the CHM is blurred first and the
    search window scales with height (see `_stems`), because a fixed window over
    a noisy CHM produces a tree per cell rather than a tree per tree.

    Points are partitioned to the *nearest* stem, so every return belongs to
    exactly one tree instead of being double-counted by overlapping radius
    queries around neighbouring peaks.
    """
    semantic = cloud["semantic"]
    idx = np.flatnonzero(semantic == S["vegetation_high"])
    if idx.size < 30:
        return []

    sigma_cells = smooth_sigma / max(raster.cell, 1e-6)
    smooth = _smooth_chm(np.nan_to_num(np.asarray(chm, float), nan=0.0), sigma_cells)
    stems, stem_h, stem_r = _stems(smooth, raster, min_height, max_height)
    if len(stems) == 0:
        return []

    xy = cloud.xyz[idx, :2]
    nearest, best = _nearest_stem(xy, stems)
    # A return past 1.3x the crown belongs to no stem rather than to the closest.
    nearest[best > (stem_r[nearest] * 1.3) ** 2] = -1

    out: list[Instance] = []
    order = np.argsort(nearest, kind="stable")
    grouped = nearest[order]
    bounds = np.searchsorted(grouped, np.arange(len(stems) + 1))
    for s in range(len(stems)):
        sel = order[bounds[s]:bounds[s + 1]]
        if sel.size < min_points:
            continue
        pts = cloud.xyz[idx[sel]]
        base = float(np.percentile(pts[:, 2], 5))
        # Prefer the observed spread over the allometric guess when it is tighter.
        spread = float(np.percentile(np.linalg.norm(pts[:, :2] - stems[s], axis=1), 90))
        radius = float(np.clip(spread, 0.9, stem_r[s]))
        out.append(Instance(
            id=start_id + len(out), role="volume.vegetation.high",
            center=np.array([stems[s][0], stems[s][1], base]),
            size=np.array([radius, radius, stem_h[s]]),
            support=int(sel.size),
            confidence=float(np.clip(0.35 + 0.5 * min(1.0, sel.size / 300.0), 0.15, 0.95)),
            point_idx=idx[sel],
            attrs={"crown_radius": radius, "canopy_height": float(stem_h[s])},
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

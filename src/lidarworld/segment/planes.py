"""Planar patch extraction.

Walls, roofs and slabs are recovered as explicit plane-bounded regions rather
than left as loose points, because a patch is the thing a theme can be applied
to: it has an orientation, an extent, a local 2D frame, and -- once the tile
lattice is laid over it -- a place for every context flag to live.

Region growing runs on the sparse voxel graph, not on points. A city block is
millions of points but only tens of thousands of occupied voxels, and the voxel
normal (from the sign-invariant normal tensor) is far steadier than any single
point normal.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..spatial.grid import build_voxel_index, eigen_sorted
from ..types import SEMANTIC_INDEX, PointCloud

S = SEMANTIC_INDEX
UP = np.array([0.0, 0.0, 1.0])


@dataclass
class PlanarPatch:
    id: int
    normal: np.ndarray
    offset: float                    # plane is n.x + offset = 0
    centroid: np.ndarray
    u: np.ndarray                    # in-plane axis, horizontal for walls
    v: np.ndarray                    # in-plane axis, up-slope
    point_idx: np.ndarray
    support: int = 0
    area: float = 0.0
    extent: tuple[float, float] = (0.0, 0.0)
    rms: float = 0.0
    role: str = "surface.wall.vertical"
    confidence: float = 0.5
    attrs: dict = field(default_factory=dict)

    @property
    def slope_deg(self) -> float:
        return float(np.degrees(np.arccos(min(1.0, abs(float(self.normal[2]))))))

    def project(self, xyz: np.ndarray) -> np.ndarray:
        rel = xyz - self.centroid
        return np.column_stack([rel @ self.u, rel @ self.v])

    def unproject(self, uv: np.ndarray) -> np.ndarray:
        return self.centroid + uv[:, :1] * self.u + uv[:, 1:2] * self.v


def plane_frame(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """In-plane axes with `u` horizontal wherever the plane is not itself flat.

    Keeping `u` on the horizon and `v` pointing up-slope is what lets the tile
    lattice carry meaningful TOP / BOTTOM / GROUND_CONTACT flags.
    """
    n = normal / max(np.linalg.norm(normal), 1e-12)
    if abs(float(n @ UP)) > 0.95:
        u = np.array([1.0, 0.0, 0.0])
        u = u - n * (u @ n)
    else:
        u = np.cross(UP, n)
    u /= max(np.linalg.norm(u), 1e-12)
    v = np.cross(n, u)
    v /= max(np.linalg.norm(v), 1e-12)
    if v @ UP < 0:
        v = -v
        u = np.cross(v, n)
        u /= max(np.linalg.norm(u), 1e-12)
    return u, v


def _voxel_normals(cloud: PointCloud, vi, idx: np.ndarray) -> np.ndarray:
    """Dominant normal per voxel via the sign-invariant normal tensor."""
    normals = cloud["normal"][idx]
    pv = vi.point_voxel
    m = vi.n_voxels
    T = np.zeros((m, 3, 3))
    for a in range(3):
        for b in range(a, 3):
            acc = np.bincount(pv, weights=normals[:, a] * normals[:, b], minlength=m)
            T[:, a, b] = acc
            if a != b:
                T[:, b, a] = acc
    _, vecs = eigen_sorted(T)
    out = vecs[:, :, 0]
    flip = out[:, 2] < 0
    out[flip] *= -1
    return out


def extract(cloud: PointCloud, *, voxel: float = 0.6, angle_deg: float = 18.0,
            dist: float = 0.32, min_voxels: int = 12,
            max_patches: int = 4000) -> list[PlanarPatch]:
    """Grow planar patches over structural points."""
    cloud.require("semantic", "normal", "planarity")
    semantic = cloud["semantic"]
    structural = np.isin(semantic, [S["building"], S["bridge"]])
    idx = np.flatnonzero(structural)
    if idx.size < min_voxels:
        return []

    xyz = cloud.xyz[idx]
    vi = build_voxel_index(xyz, voxel)
    m = vi.n_voxels
    centers = np.zeros((m, 3))
    for a in range(3):
        centers[:, a] = np.bincount(vi.point_voxel, weights=xyz[:, a], minlength=m)
    centers /= np.maximum(vi.counts, 1)[:, None]
    normals = _voxel_normals(cloud, vi, idx)
    planarity = np.bincount(vi.point_voxel, weights=cloud["planarity"][idx].astype(np.float64),
                            minlength=m) / np.maximum(vi.counts, 1)

    # 26-neighbourhood adjacency, resolved once, vectorised.
    offsets = [o for o in vi.neighbor_offsets(1) if o.any()]
    neighbors = np.full((m, len(offsets)), -1, dtype=np.int64)
    for k, off in enumerate(offsets):
        neighbors[:, k] = vi.lookup(vi.ijk + off)

    cos_thresh = np.cos(np.radians(angle_deg))
    assigned = np.full(m, -1, dtype=np.int32)
    order = np.argsort(-planarity)
    patches: list[PlanarPatch] = []

    for seed in order:
        if assigned[seed] >= 0 or planarity[seed] < 0.25 or len(patches) >= max_patches:
            continue
        pid = len(patches)
        n_hat = normals[seed].copy()
        origin = centers[seed].copy()
        offset = -float(n_hat @ origin)

        members = [int(seed)]
        assigned[seed] = pid
        stack = [int(seed)]
        # Incremental moments so the plane can be refit as the region grows.
        acc_n = 0.0
        acc_sum = np.zeros(3)
        acc_outer = np.zeros((3, 3))
        refit_at = 24

        def absorb(v):
            nonlocal acc_n, acc_sum, acc_outer
            w = float(vi.counts[v])
            c = centers[v]
            acc_n += w
            acc_sum += w * c
            acc_outer += w * np.outer(c, c)

        absorb(seed)

        while stack:
            cur = stack.pop()
            for nb in neighbors[cur]:
                if nb < 0 or assigned[nb] >= 0:
                    continue
                if abs(float(normals[nb] @ n_hat)) < cos_thresh:
                    continue
                if abs(float(n_hat @ centers[nb]) + offset) > dist:
                    continue
                assigned[nb] = pid
                members.append(int(nb))
                absorb(nb)
                stack.append(int(nb))
                if len(members) >= refit_at:
                    refit_at = int(len(members) * 1.6) + 8
                    mean = acc_sum / acc_n
                    cov = acc_outer / acc_n - np.outer(mean, mean)
                    vals, vecs = eigen_sorted(cov[None])
                    cand = vecs[0, :, 2]
                    if abs(float(cand @ n_hat)) > 0.7:      # keep orientation stable
                        n_hat = cand * np.sign(float(cand @ n_hat))
                        origin = mean
                        offset = -float(n_hat @ mean)

        if len(members) < min_voxels:
            for v in members:
                assigned[v] = -1
            continue

        member_arr = np.asarray(members, dtype=np.int64)
        pt_mask = np.isin(vi.point_voxel, member_arr)
        pts = xyz[pt_mask]
        mean = pts.mean(axis=0)
        cov = np.cov(pts.T) if len(pts) > 3 else np.eye(3) * 1e-6
        vals, vecs = eigen_sorted(cov[None])
        n_final = vecs[0, :, 2]
        if n_final[2] < 0 or (abs(n_final[2]) < 0.2 and n_final[0] < 0):
            n_final = -n_final
        u, v_axis = plane_frame(n_final)
        rel = pts - mean
        uu, vv = rel @ u, rel @ v_axis
        residual = rel @ n_final

        patch = PlanarPatch(
            id=pid, normal=n_final, offset=-float(n_final @ mean), centroid=mean,
            u=u, v=v_axis, point_idx=idx[pt_mask], support=int(pt_mask.sum()),
            extent=(float(uu.max() - uu.min()), float(vv.max() - vv.min())),
            rms=float(np.sqrt(np.mean(residual ** 2))),
        )
        patch.area = patch.extent[0] * patch.extent[1]
        patch.role = classify_patch(patch, cloud, pt_mask, idx)
        patch.confidence = float(np.clip(
            0.35 + 0.4 * min(1.0, patch.support / 400.0) + 0.25 * (1.0 - min(1.0, patch.rms / dist)),
            0.1, 0.99))
        patches.append(patch)

    return patches


def classify_patch(patch: PlanarPatch, cloud: PointCloud, mask: np.ndarray,
                   idx: np.ndarray) -> str:
    """Promote per-point role hints to a decision for the whole patch."""
    hag = cloud["hag"][idx[mask]]
    slope = patch.slope_deg
    median_hag = float(np.median(hag))
    if slope > 55:
        return "surface.wall.vertical"
    if median_hag < 1.0:
        return "surface.slab"
    if slope > 12:
        return "surface.roof.pitched"
    return "surface.roof.flat" if median_hag > 2.0 else "surface.slab"


def assign_patch_channel(cloud: PointCloud, patches: list[PlanarPatch]) -> np.ndarray:
    """Write a `patch` channel (-1 = unassigned) and return it."""
    out = np.full(len(cloud), -1, dtype=np.int32)
    for p in patches:
        out[p.point_idx] = p.id
    cloud["patch"] = out
    return out

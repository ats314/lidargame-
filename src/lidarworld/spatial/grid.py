"""Sparse voxel indexing.

The design answer said "sparse voxels/octrees only for fast spatial reasoning",
and that is exactly what this is: never the output representation, always the
accelerator underneath neighbourhood statistics, region growing and adjacency.

The trick used throughout is moment aggregation. Instead of gathering k nearest
neighbours per point (a kd-tree query per point, and a different neighbourhood
per scale), we accumulate the ten second-order moments per voxel with
``np.bincount``, then sum a 3x3x3 block of voxels to get the covariance of a
whole neighbourhood in one shot. Cost is linear in points and, crucially,
linear in *voxels* for the expensive part -- a few million points stay
interactive in pure numpy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VoxelIndex:
    """Sparse voxel partition of a point set."""
    size: float
    ijk: np.ndarray          # (M,3) int64 integer coordinates of occupied voxels
    keys: np.ndarray         # (M,)  encoded keys, sorted ascending
    point_voxel: np.ndarray  # (N,)  index into ijk/keys for every point
    counts: np.ndarray       # (M,)  points per voxel
    origin: np.ndarray       # (3,)  world position of integer coordinate (0,0,0)
    dims: np.ndarray         # (3,)  integer extent, used for key encoding

    @property
    def n_voxels(self) -> int:
        return len(self.keys)

    def encode(self, ijk: np.ndarray) -> np.ndarray:
        nx, ny = int(self.dims[0]), int(self.dims[1])
        ijk = np.asarray(ijk, dtype=np.int64)
        return ijk[:, 0] + nx * (ijk[:, 1] + ny * ijk[:, 2])

    def lookup(self, ijk: np.ndarray) -> np.ndarray:
        """Voxel index for integer coordinates; -1 where the voxel is empty."""
        ijk = np.asarray(ijk, dtype=np.int64)
        inside = np.all((ijk >= 0) & (ijk < self.dims), axis=1)
        out = np.full(len(ijk), -1, dtype=np.int64)
        if not inside.any():
            return out
        q = self.encode(ijk[inside])
        pos = np.searchsorted(self.keys, q)
        pos_clipped = np.clip(pos, 0, len(self.keys) - 1)
        hit = self.keys[pos_clipped] == q
        idx = np.where(inside)[0][hit]
        out[idx] = pos_clipped[hit]
        return out

    def centers(self) -> np.ndarray:
        return self.origin + (self.ijk.astype(np.float64) + 0.5) * self.size

    def neighbor_offsets(self, radius: int = 1) -> np.ndarray:
        r = np.arange(-radius, radius + 1)
        gx, gy, gz = np.meshgrid(r, r, r, indexing="ij")
        return np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()]).astype(np.int64)

    def neighbor_sum(self, values: np.ndarray, radius: int = 1) -> np.ndarray:
        """Sum per-voxel `values` over the (2r+1)^3 block around every voxel."""
        values = np.asarray(values)
        out = np.zeros_like(values, dtype=np.float64)
        for off in self.neighbor_offsets(radius):
            nb = self.lookup(self.ijk + off)
            ok = nb >= 0
            out[ok] += values[nb[ok]]
        return out

    def voxel_of(self, xyz: np.ndarray) -> np.ndarray:
        """Voxel index for arbitrary world positions; -1 if unoccupied."""
        ijk = np.floor((np.asarray(xyz, dtype=np.float64) - self.origin) / self.size).astype(np.int64)
        return self.lookup(ijk)


def build_voxel_index(xyz: np.ndarray, size: float, origin: np.ndarray | None = None) -> VoxelIndex:
    xyz = np.asarray(xyz, dtype=np.float64)
    if origin is None:
        origin = xyz.min(axis=0) - size
    ijk_all = np.floor((xyz - origin) / size).astype(np.int64)
    dims = ijk_all.max(axis=0) + 2
    nx, ny = int(dims[0]), int(dims[1])
    keys_all = ijk_all[:, 0] + nx * (ijk_all[:, 1] + ny * ijk_all[:, 2])
    keys, point_voxel, counts = np.unique(keys_all, return_inverse=True, return_counts=True)
    order = np.argsort(keys, kind="stable")
    if not np.array_equal(order, np.arange(len(keys))):     # np.unique already sorts
        keys, counts = keys[order], counts[order]
    kz = keys // (nx * ny)
    rem = keys - kz * (nx * ny)
    ky = rem // nx
    kx = rem - ky * nx
    ijk = np.column_stack([kx, ky, kz]).astype(np.int64)
    return VoxelIndex(size=size, ijk=ijk, keys=keys, point_voxel=point_voxel.astype(np.int64),
                      counts=counts, origin=np.asarray(origin, dtype=np.float64), dims=dims)


def voxel_moments(xyz: np.ndarray, index: VoxelIndex) -> np.ndarray:
    """Per-voxel (n, sum_xyz, sum_outer_products) -> (M, 10) moment table."""
    m = index.n_voxels
    pv = index.point_voxel
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    out = np.empty((m, 10), dtype=np.float64)
    out[:, 0] = np.bincount(pv, minlength=m)
    for col, vals in enumerate((x, y, z, x * x, x * y, x * z, y * y, y * z, z * z), start=1):
        out[:, col] = np.bincount(pv, weights=vals, minlength=m)
    return out


def covariance_from_moments(moments: np.ndarray, min_points: int = 4):
    """(M,10) moments -> (mean (M,3), covariance (M,3,3), count (M,))."""
    n = moments[:, 0]
    safe = np.maximum(n, 1.0)
    mean = moments[:, 1:4] / safe[:, None]
    cov = np.empty((len(n), 3, 3), dtype=np.float64)
    sxx, sxy, sxz, syy, syz, szz = (moments[:, i] for i in range(4, 10))
    cov[:, 0, 0] = sxx / safe - mean[:, 0] ** 2
    cov[:, 1, 1] = syy / safe - mean[:, 1] ** 2
    cov[:, 2, 2] = szz / safe - mean[:, 2] ** 2
    cov[:, 0, 1] = cov[:, 1, 0] = sxy / safe - mean[:, 0] * mean[:, 1]
    cov[:, 0, 2] = cov[:, 2, 0] = sxz / safe - mean[:, 0] * mean[:, 2]
    cov[:, 1, 2] = cov[:, 2, 1] = syz / safe - mean[:, 1] * mean[:, 2]
    degenerate = n < min_points
    cov[degenerate] = np.eye(3) * 1e-9
    return mean, cov, n


def eigen_sorted(cov: np.ndarray):
    """Batched symmetric eigendecomposition, eigenvalues descending.

    Returns (values (M,3) descending, vectors (M,3,3) with column j matching
    value j). ``np.linalg.eigh`` returns ascending, so both are reversed.
    """
    vals, vecs = np.linalg.eigh(cov)
    vals = vals[:, ::-1]
    vecs = vecs[:, :, ::-1]
    return np.maximum(vals, 0.0), vecs


class Raster2D:
    """Regular 2D lattice over the XY footprint -- terrain, footprints, masks."""

    def __init__(self, bounds_min, bounds_max, cell: float, pad: int = 1):
        self.cell = float(cell)
        self.origin = np.asarray(bounds_min, dtype=np.float64)[:2] - pad * cell
        far = np.asarray(bounds_max, dtype=np.float64)[:2] + pad * cell
        self.shape = tuple(np.maximum(np.ceil((far - self.origin) / cell).astype(int), 1))

    @property
    def nx(self) -> int: return int(self.shape[0])

    @property
    def ny(self) -> int: return int(self.shape[1])

    def to_cell(self, xy: np.ndarray) -> np.ndarray:
        ij = np.floor((np.asarray(xy, dtype=np.float64)[:, :2] - self.origin) / self.cell).astype(np.int64)
        np.clip(ij, 0, np.array(self.shape) - 1, out=ij)
        return ij

    def flat(self, ij: np.ndarray) -> np.ndarray:
        return ij[:, 0] * self.ny + ij[:, 1]

    def cell_centers(self):
        gx = self.origin[0] + (np.arange(self.nx) + 0.5) * self.cell
        gy = self.origin[1] + (np.arange(self.ny) + 0.5) * self.cell
        return gx, gy

    def accumulate(self, xy: np.ndarray, values: np.ndarray | None = None, how: str = "min"):
        """Rasterise a scattered field. Empty cells come back as NaN (min/max/mean)."""
        ij = self.to_cell(xy)
        flat = self.flat(ij)
        size = self.nx * self.ny
        counts = np.bincount(flat, minlength=size).astype(np.float64)
        if values is None or how == "count":
            return counts.reshape(self.shape)
        if how == "mean":
            total = np.bincount(flat, weights=values, minlength=size)
            out = np.where(counts > 0, total / np.maximum(counts, 1), np.nan)
            return out.reshape(self.shape)
        fill = np.inf if how == "min" else -np.inf
        out = np.full(size, fill, dtype=np.float64)
        np.minimum.at(out, flat, values) if how == "min" else np.maximum.at(out, flat, values)
        out[~np.isfinite(out)] = np.nan
        return out.reshape(self.shape)

    def sample_bilinear(self, field: np.ndarray, xy: np.ndarray) -> np.ndarray:
        """Bilinear lookup at arbitrary world XY (NaNs treated as holes)."""
        p = (np.asarray(xy, dtype=np.float64)[:, :2] - self.origin) / self.cell - 0.5
        i0 = np.floor(p[:, 0]).astype(np.int64)
        j0 = np.floor(p[:, 1]).astype(np.int64)
        fx = p[:, 0] - i0
        fy = p[:, 1] - j0
        i0 = np.clip(i0, 0, self.nx - 2) if self.nx > 1 else np.zeros_like(i0)
        j0 = np.clip(j0, 0, self.ny - 2) if self.ny > 1 else np.zeros_like(j0)
        i1 = np.minimum(i0 + 1, self.nx - 1)
        j1 = np.minimum(j0 + 1, self.ny - 1)
        f = np.nan_to_num(field, nan=0.0)
        w = np.isfinite(field).astype(np.float64)
        num = (f[i0, j0] * (1 - fx) * (1 - fy) + f[i1, j0] * fx * (1 - fy)
               + f[i0, j1] * (1 - fx) * fy + f[i1, j1] * fx * fy)
        den = (w[i0, j0] * (1 - fx) * (1 - fy) + w[i1, j0] * fx * (1 - fy)
               + w[i0, j1] * (1 - fx) * fy + w[i1, j1] * fx * fy)
        return np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nan)


def fill_holes(field: np.ndarray, iterations: int = 64) -> np.ndarray:
    """Flood NaNs with an iterative 4-neighbour mean -- a cheap Laplacian fill."""
    out = field.copy()
    holes = ~np.isfinite(out)
    if not holes.any():
        return out
    out[holes] = np.nanmean(field) if np.isfinite(field).any() else 0.0
    for _ in range(iterations):
        if not holes.any():
            break
        s = np.zeros_like(out)
        s[:-1] += out[1:]; s[1:] += out[:-1]
        s[:, :-1] += out[:, 1:]; s[:, 1:] += out[:, :-1]
        w = np.zeros_like(out)
        w[:-1] += 1; w[1:] += 1; w[:, :-1] += 1; w[:, 1:] += 1
        out[holes] = (s / np.maximum(w, 1))[holes]
    return out


def box_blur(field: np.ndarray, radius: int = 1, passes: int = 1) -> np.ndarray:
    out = field.astype(np.float64, copy=True)
    for _ in range(passes):
        pad = np.pad(out, radius, mode="edge")
        acc = np.zeros_like(out)
        k = 2 * radius + 1
        for di in range(k):
            for dj in range(k):
                acc += pad[di:di + out.shape[0], dj:dj + out.shape[1]]
        out = acc / (k * k)
    return out

"""Reject invented geometry that the observations contradict.

The compiler synthesises surfaces where the sensor saw nothing -- walls extruded
down from a footprint, lattice cells filled across a gap. That is the right
thing to do, because airborne LiDAR barely sees facades at all, but it is also
where the compiler is free to be wrong, and forward validation says it is: 5,421
rays hit geometry that is not there.

Hydra++ (Lim et al., IROS 2026) makes the same observation about learned shape
estimates and answers it with a reprojection consistency check -- estimate the
shape, reproject it against the observation, reject the ones that disagree.
This is the same idea applied at the surface level, and the reason it can be
cheap is that airborne acquisition is nearly nadir: a return at height z proves
the beam travelled down through everything above z in that column. Any
synthesised surface sitting above the highest return in its column is standing
in space the beam demonstrably crossed.

So no sensor pose is needed and no ray casting: the check is a column maximum,
which is a `np.maximum.at` over the cloud. That matters because the gate has to
run inside reconstruction, once per candidate cell, not as an after-the-fact
report over the finished world.

The check is deliberately one-sided. It can say "this surface is in free space";
it cannot say "this surface is real". Only synthesised geometry is ever gated --
measured surfaces stand on their own returns and are never second-guessed here.
"""
from __future__ import annotations

import numpy as np

from ..spatial.grid import Raster2D


class FreeSpace:
    """Highest observed return per ground column, and the queries over it.

    `clearance` is how far above the highest return a surface has to reach
    before it counts as contradicted. It absorbs the two things that make the
    nadir approximation imperfect: off-nadir scan angle, which lets a beam pass
    beside a column rather than through it, and the fact that a column with no
    return at all is unobserved rather than empty.
    """

    def __init__(self, xyz: np.ndarray, raster: Raster2D, *, clearance: float = 1.5):
        self.raster = raster
        self.clearance = float(clearance)
        size = raster.nx * raster.ny
        ceiling = np.full(size, -np.inf)
        flat = raster.flat(raster.to_cell(xyz))
        np.maximum.at(ceiling, flat, xyz[:, 2])
        self.ceiling = ceiling.reshape(raster.shape)
        self.observed = np.isfinite(self.ceiling)

    @property
    def observed_fraction(self) -> float:
        return float(self.observed.mean())

    def conflict(self, points: np.ndarray) -> np.ndarray:
        """True where a point sits above the highest return in its column.

        Columns with no return are `False` -- unobserved is not empty, and
        treating it as empty would delete exactly the geometry synthesis exists
        to supply.
        """
        points = np.atleast_2d(np.asarray(points, dtype=float))
        if not len(points):
            return np.zeros(0, bool)
        ij = self.raster.to_cell(points)
        ceiling = self.ceiling[ij[:, 0], ij[:, 1]]
        return np.isfinite(ceiling) & (points[:, 2] > ceiling + self.clearance)

    def conflict_fraction(self, points: np.ndarray) -> float:
        points = np.atleast_2d(np.asarray(points, dtype=float))
        if not len(points):
            return 0.0
        return float(self.conflict(points).mean())


def gate_lattice(lattice, patch, free: FreeSpace, *, max_conflict: float = 0.5):
    """Clear synthesised cells that stand in observed free space.

    Returns (cells_cleared, was_rejected). A patch that loses most of itself is
    reported as rejected so the caller can drop the surface outright rather than
    keep a shredded remnant of it.
    """
    solid = lattice.occupancy.astype(bool)
    if not solid.any():
        return 0, True

    iu, iv = np.nonzero(solid)
    uv = lattice.cell_uv(iu, iv)
    world = patch_points(patch, uv)
    bad = free.conflict(world)
    if not bad.any():
        return 0, False

    lattice.occupancy[iu[bad], iv[bad]] = 0
    lattice.evidence[iu[bad], iv[bad]] = 0
    cleared = int(bad.sum())
    remaining = int(lattice.occupancy.sum())
    rejected = remaining == 0 or cleared / max(cleared + remaining, 1) > max_conflict
    return cleared, rejected


def patch_points(patch, uv: np.ndarray) -> np.ndarray:
    """Lift patch-local (u, v) coordinates back into world space.

    Inverse of `PlanarPatch.project`, which measures uv from the centroid along
    the patch's own in-plane axes.
    """
    return (np.asarray(patch.centroid, dtype=float)[None, :]
            + uv[:, 0:1] * np.asarray(patch.u, dtype=float)[None, :]
            + uv[:, 1:2] * np.asarray(patch.v, dtype=float)[None, :])

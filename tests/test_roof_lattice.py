"""Sizing a roof's lattice to what its returns can actually fill.

The speckle was not a meshing bug. It was a 0.25 m cell -- a facade number --
applied to a surface sampled at a few points per metre from an aircraft, so
81% of roof cells were empty and the holes reached the patch border, where
nothing fills them.
"""
import math

import numpy as np

from lidarworld.reconstruct import lattice


class FlatPatch:
    """Just enough patch to project points into a plane."""

    def project(self, xyz):
        return np.asarray(xyz)[:, :2]


def _poisson(density, size, seed):
    rng = np.random.default_rng(seed)
    n = rng.poisson(density * size * size)
    return np.column_stack([rng.uniform(0, size, n), rng.uniform(0, size, n),
                            np.zeros(n)])


def test_cell_scales_with_the_inverse_root_of_density():
    """Four times the returns should halve the cell: the sampling length
    scale is 1/sqrt(density), and the cell has to follow it."""
    patch = FlatPatch()
    sparse = lattice.sampling_cell(patch, _poisson(4.0, 60, 1), floor=0.01, ceiling=99)
    dense = lattice.sampling_cell(patch, _poisson(16.0, 60, 2), floor=0.01, ceiling=99)
    assert 1.7 < sparse / dense < 2.3, (sparse, dense)


def test_the_chosen_cell_hits_the_occupancy_it_aimed_for():
    """The Poisson model is the whole justification, so check it predicts."""
    patch = FlatPatch()
    for density in (1.0, 3.6, 12.0):
        xyz = _poisson(density, 80, int(density * 10))
        cell = lattice.sampling_cell(patch, xyz, floor=0.01, ceiling=99, target=0.6)
        iu = (xyz[:, 0] / cell).astype(int)
        iv = (xyz[:, 1] / cell).astype(int)
        occupied = len(set(zip(iu.tolist(), iv.tolist())))
        total = math.ceil(80 / cell) ** 2
        assert 0.5 < occupied / total < 0.7, (density, cell, occupied / total)


def test_bounds_are_respected():
    patch = FlatPatch()
    thin = _poisson(0.05, 60, 3)
    assert lattice.sampling_cell(patch, thin, floor=0.25, ceiling=1.0) == 1.0
    thick = _poisson(400.0, 20, 4)
    assert lattice.sampling_cell(patch, thick, floor=0.25, ceiling=1.0) == 0.25


def test_a_patch_with_almost_no_returns_does_not_divide_by_zero():
    patch = FlatPatch()
    for n in (0, 1, 3):
        cell = lattice.sampling_cell(patch, np.zeros((n, 3)), floor=0.25, ceiling=1.0)
        assert cell == 1.0


def test_density_uses_occupied_area_not_the_bounding_box():
    """An L-shaped roof fills half its box. Measuring density over the box
    halves it, and a halved density coarsens the cell by root two for no
    reason -- which is the blur this function exists to avoid."""
    patch = FlatPatch()
    full = _poisson(4.0, 60, 5)
    ell = full[(full[:, 0] < 30) | (full[:, 1] < 30)]
    assert 0.6 < len(ell) / len(full) < 0.9, "expected a genuinely L-shaped subset"
    a = lattice.sampling_cell(patch, full, floor=0.01, ceiling=99)
    b = lattice.sampling_cell(patch, ell, floor=0.01, ceiling=99)
    assert abs(a - b) / a < 0.1, (a, b)


def test_a_sparse_roof_lattice_is_mostly_holes_at_the_facade_cell():
    """The measurement the fix is built on, as a test: at 0.25 m an airborne
    roof is ~20% occupied, and at the chosen cell it is ~60%."""
    patch = FlatPatch()
    xyz = _poisson(3.6, 80, 7)

    def occupancy(cell):
        iu = (xyz[:, 0] / cell).astype(int)
        iv = (xyz[:, 1] / cell).astype(int)
        return len(set(zip(iu.tolist(), iv.tolist()))) / math.ceil(80 / cell) ** 2

    assert occupancy(0.25) < 0.3
    assert occupancy(lattice.sampling_cell(patch, xyz, floor=0.25, ceiling=1.0)) > 0.5

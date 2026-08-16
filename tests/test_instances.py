"""Tree instancing.

The failure this guards against is over-segmentation: a fixed local-maximum
window over a noisy canopy model finds a tree per raster cell, which is how a
300 m Denver block came out with 1,197 "trees" in it.
"""
from __future__ import annotations

import numpy as np
import pytest

from lidarworld.segment.instances import _stems, crown_radius, trees
from lidarworld.spatial.grid import Raster2D
from lidarworld.types import SEMANTIC_INDEX, PointCloud

S = SEMANTIC_INDEX


def raster_for(size=40.0, cell=1.0):
    return Raster2D([0.0, 0.0, 0.0], [size, size, 0.0], cell, pad=0)


def chm_with(raster, peaks, base=0.0):
    """Canopy model with a cone at each (x, y, height, radius)."""
    gx, gy = raster.cell_centers()
    X, Y = np.meshgrid(gx, gy, indexing="ij")
    chm = np.full(X.shape, base)
    for (px, py, height, radius) in peaks:
        d = np.hypot(X - px, Y - py)
        cone = np.clip(height * (1 - d / radius), 0, None)
        chm = np.maximum(chm, cone)
    return chm


def cloud_under(chm, raster, rng, per_cell=6):
    """Vegetation returns scattered through the volume the CHM describes."""
    gx, gy = raster.cell_centers()
    pts = []
    for i in range(chm.shape[0]):
        for j in range(chm.shape[1]):
            if chm[i, j] < 2.5:
                continue
            n = per_cell
            pts.append(np.column_stack([
                gx[i] + rng.uniform(-0.5, 0.5, n),
                gy[j] + rng.uniform(-0.5, 0.5, n),
                chm[i, j] * rng.uniform(0.45, 1.0, n),
            ]))
    xyz = np.concatenate(pts)
    return PointCloud(xyz, semantic=np.full(len(xyz), S["vegetation_high"], np.uint8))


def test_crown_radius_grows_with_height_and_stays_sane():
    assert crown_radius(3.0) < crown_radius(12.0) < crown_radius(25.0)
    assert crown_radius(0.0) == pytest.approx(1.2)      # clamped low
    assert crown_radius(500.0) == pytest.approx(7.0)    # clamped high
    assert np.all(crown_radius(np.array([2.0, 10.0, 30.0])) <= 7.0)


def test_a_flat_plateau_is_one_stem_not_one_per_cell():
    """Every cell of a plateau ties for local maximum. Only one may survive."""
    raster = raster_for()
    chm = np.zeros((raster.nx, raster.ny))
    chm[10:14, 10:14] = 12.0                            # 4x4 m flat canopy
    stems, heights, radii = _stems(chm, raster, 2.5, 40.0)
    assert len(stems) == 1, "every cell of a plateau ties; only one is a tree"
    assert heights[0] == pytest.approx(12.0)


def test_two_separated_trees_stay_two():
    raster = raster_for()
    chm = chm_with(raster, [(10.0, 10.0, 12.0, 3.0), (30.0, 30.0, 9.0, 2.5)])
    stems, heights, _ = _stems(chm, raster, 2.5, 40.0)
    assert len(stems) == 2
    # Cell centres miss the cone apex, so heights are sampled low -- what must
    # hold is that the taller cone stays the taller stem.
    order = np.argsort(-heights)
    assert np.linalg.norm(stems[order[0]] - [10.0, 10.0]) < 2.0
    # ... and each stem sits on its own cone, not between them.
    for want in ([10.0, 10.0], [30.0, 30.0]):
        assert np.min(np.linalg.norm(stems - np.array(want), axis=1)) < 2.0


def test_one_broad_crown_is_not_split_into_four():
    """A wide crown spans several cells; a fixed 3x3 window splits it."""
    raster = raster_for()
    chm = chm_with(raster, [(20.0, 20.0, 22.0, 6.0)])
    stems, _, _ = _stems(chm, raster, 2.5, 40.0)
    assert len(stems) == 1


def test_noise_does_not_manufacture_stems():
    rng = np.random.default_rng(7)
    raster = raster_for()
    chm = chm_with(raster, [(20.0, 20.0, 14.0, 4.0)])
    noisy = chm + rng.normal(0, 0.35, chm.shape)
    smooth_count = len(_stems(chm, raster, 2.5, 40.0)[0])
    noisy_count = len(trees(cloud_under(noisy, raster, rng), raster, noisy))
    assert smooth_count == 1
    assert noisy_count <= 3, f"canopy noise produced {noisy_count} stems from one tree"


def test_a_seventy_metre_tree_is_rejected_as_a_building():
    raster = raster_for()
    chm = chm_with(raster, [(20.0, 20.0, 74.0, 5.0)])
    assert len(_stems(chm, raster, 2.5, 40.0)[0]) == 0
    assert len(_stems(chm, raster, 2.5, 90.0)[0]) == 1, "the cap must be the only reason"


def test_every_return_belongs_to_exactly_one_tree():
    """Overlapping radius queries double-count points between neighbours."""
    rng = np.random.default_rng(11)
    raster = raster_for()
    chm = chm_with(raster, [(14.0, 20.0, 13.0, 3.5), (24.0, 20.0, 11.0, 3.0)])
    cloud = cloud_under(chm, raster, rng)
    found = trees(cloud, raster, chm)
    assert len(found) >= 2
    owned = np.concatenate([t.point_idx for t in found])
    assert len(owned) == len(np.unique(owned)), "a point was claimed by two trees"
    for tree in found:
        assert tree.support == len(tree.point_idx)


def test_trees_reports_nothing_when_there_is_no_canopy():
    rng = np.random.default_rng(3)
    raster = raster_for()
    flat = np.zeros((raster.nx, raster.ny))
    xyz = rng.random((200, 3)) * [40, 40, 0.2]
    cloud = PointCloud(xyz, semantic=np.full(200, S["ground"], np.uint8))
    assert trees(cloud, raster, flat) == []

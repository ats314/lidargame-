"""The descriptors have to mean what they claim.

These are the tests that matter most: every role decision and every theme rule
downstream is built on planarity, verticality, crease and corner scores being
right, so they are checked against geometry with a known answer.
"""
from __future__ import annotations

import numpy as np

from lidarworld.features import neighborhood
from lidarworld.features.ground import estimate
from lidarworld.spatial.grid import build_voxel_index, covariance_from_moments, eigen_sorted, voxel_moments
from lidarworld.types import PointCloud


def _plane(rng, n=4000, size=8.0, noise=0.01):
    xy = rng.random((n, 2)) * size
    return np.column_stack([xy, rng.normal(0, noise, n)])


def test_voxel_moments_match_direct_covariance():
    rng = np.random.default_rng(0)
    pts = rng.normal(0, 3, (2000, 3))
    index = build_voxel_index(pts, 50.0)          # one voxel holds everything
    assert index.n_voxels == 1
    _, cov, count = covariance_from_moments(voxel_moments(pts, index))
    assert count[0] == 2000
    assert np.allclose(cov[0], np.cov(pts.T, bias=True), atol=1e-8)


def test_eigen_values_are_descending():
    rng = np.random.default_rng(1)
    cov = np.stack([np.cov(rng.normal(0, s, (200, 3)).T) for s in (1.0, 4.0)])
    vals, vecs = eigen_sorted(cov)
    assert np.all(np.diff(vals, axis=1) <= 1e-9)
    assert vecs.shape == (2, 3, 3)


def test_plane_is_planar_and_line_is_linear():
    rng = np.random.default_rng(2)
    plane = neighborhood.scale_features(_plane(rng), 1.0)
    assert plane["planarity"].mean() > 0.4
    # A finite square patch is never perfectly isotropic in-plane, so
    # linearity sits well above zero; what matters is the ordering.
    assert plane["linearity"].mean() < 0.4
    assert plane["planarity"].mean() > plane["linearity"].mean() * 2
    # A horizontal plane has a vertical normal, so verticality must be ~0.
    assert plane["verticality"].mean() < 0.1

    t = np.linspace(0, 10, 2000)
    line = np.column_stack([t, np.zeros_like(t), np.zeros_like(t)]) + rng.normal(0, 0.01, (2000, 3))
    feats = neighborhood.scale_features(line, 1.0)
    assert feats["linearity"].mean() > 0.85
    assert feats["planarity"].mean() < 0.2


def test_vertical_wall_reads_as_vertical():
    rng = np.random.default_rng(3)
    n = 4000
    wall = np.column_stack([rng.random(n) * 8, rng.normal(0, 0.01, n), rng.random(n) * 6])
    feats = neighborhood.scale_features(wall, 1.0)
    assert feats["verticality"].mean() > 0.9
    assert feats["planarity"].mean() > 0.4


def test_crease_and_corner_scores_localise():
    """Two perpendicular walls: crease high on the shared edge, low on the faces."""
    rng = np.random.default_rng(4)
    n = 6000
    a = np.column_stack([rng.random(n) * 6, np.zeros(n), rng.random(n) * 6])
    b = np.column_stack([np.zeros(n), rng.random(n) * 6, rng.random(n) * 6])
    pts = np.concatenate([a, b]) + rng.normal(0, 0.008, (2 * n, 3))

    feats = neighborhood.scale_features(pts, 0.8)
    distance_to_crease = np.hypot(pts[:, 0], pts[:, 1])
    on_crease = distance_to_crease < 0.5
    on_face = distance_to_crease > 2.5

    assert feats["crease_score"][on_crease].mean() > feats["crease_score"][on_face].mean() * 2
    assert feats["surface_score"][on_face].mean() > 0.9


def test_boundary_score_finds_the_rim():
    rng = np.random.default_rng(5)
    pts = _plane(rng, n=6000, size=10.0)
    feats = neighborhood.scale_features(pts, 1.0)
    edge = (pts[:, 0] < 0.6) | (pts[:, 0] > 9.4) | (pts[:, 1] < 0.6) | (pts[:, 1] > 9.4)
    middle = (np.abs(pts[:, 0] - 5) < 2) & (np.abs(pts[:, 1] - 5) < 2)
    # Random sampling puts a floor under the metric even in the interior,
    # so assert a clear separation rather than a large ratio.
    assert feats["boundary"][edge].mean() > feats["boundary"][middle].mean() * 1.3
    assert feats["boundary"][edge].mean() > 0.4


def test_multiscale_attaches_per_scale_channels(tiny_cloud):
    neighborhood.compute(tiny_cloud, scales=(0.4, 1.2))
    for name in ("planarity", "crease_score", "normal", "boundary"):
        assert name in tiny_cloud
    assert "planarity@0.4" in tiny_cloud and "planarity@1.2" in tiny_cloud
    assert tiny_cloud["normal"].shape == (len(tiny_cloud), 3)
    assert np.allclose(np.linalg.norm(tiny_cloud["normal"], axis=1), 1.0, atol=1e-5)


def test_height_above_ground_uses_labelled_ground(tiny_cloud):
    raster, dtm = estimate(tiny_cloud, cell=1.0)
    assert tiny_cloud.meta["dtm_method"] == "labelled ground points"
    hag = tiny_cloud["hag"]
    ground = tiny_cloud["semantic"] == 1
    assert abs(float(np.median(hag[ground]))) < 0.3
    # The roof of the tiny scene sits at z = 6.
    assert float(hag.max()) > 5.0


def test_ground_falls_back_to_morphological_filter():
    """With no labels at all, the filter still has to find the ground."""
    rng = np.random.default_rng(6)
    ground = np.column_stack([rng.random(6000) * 30, rng.random(6000) * 30, np.zeros(6000)])
    ground[:, 2] = 0.4 * np.sin(ground[:, 0] / 8)
    block = np.column_stack([10 + rng.random(2000) * 6, 10 + rng.random(2000) * 6,
                             4 + rng.random(2000) * 4])
    cloud = PointCloud(np.concatenate([ground, block]))
    estimate(cloud, cell=1.0)
    assert cloud.meta["dtm_method"] == "progressive morphological filter"
    hag = cloud["hag"]
    assert abs(float(np.median(hag[:6000]))) < 0.5
    assert float(np.median(hag[6000:])) > 3.0

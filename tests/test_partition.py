"""The partitioner seam.

What matters is the contract, not the backend: whatever computes segments must
leave the cloud carrying the same descriptors, so roles/segment/reconstruct
stay portable across a numpy partitioner and a GPU one.
"""
from __future__ import annotations

import numpy as np
import pytest

from lidarworld.features.partition import BACKENDS, REQUIRED, Partition, partition


def test_voxel_backend_satisfies_the_contract(tiny_cloud):
    result = partition(tiny_cloud, "voxel")
    assert isinstance(result, Partition)
    assert result.backend == "voxel"
    assert result.levels == 1
    assert result.segments.shape == (len(tiny_cloud.xyz),)
    assert result.count > 1
    for name in REQUIRED:
        assert name in tiny_cloud, f"{name} missing after partitioning"


def test_segments_are_spatially_coherent(tiny_cloud):
    """Points sharing a segment must be closer together than random pairs."""
    result = partition(tiny_cloud, "voxel", scales=(0.5, 1.0))
    xyz = tiny_cloud.xyz
    segments = result.segments

    order = np.argsort(segments, kind="stable")
    grouped = segments[order]
    starts = np.flatnonzero(np.diff(grouped)) + 1
    spreads = []
    for chunk in np.split(order, starts):
        if len(chunk) > 2:
            spreads.append(float(np.ptp(xyz[chunk], axis=0).max()))
    assert spreads, "expected multi-point segments"

    rng = np.random.default_rng(3)
    pairs = rng.integers(0, len(xyz), (400, 2))
    scatter = float(np.median(np.abs(xyz[pairs[:, 0]] - xyz[pairs[:, 1]]).max(axis=1)))
    assert np.median(spreads) < scatter / 4


def test_partition_is_deterministic(tiny_cloud):
    first = partition(tiny_cloud, "voxel").segments
    second = partition(tiny_cloud, "voxel").segments
    assert np.array_equal(first, second)


def test_unknown_backend_names_the_ones_that_exist(tiny_cloud):
    with pytest.raises(KeyError) as excinfo:
        partition(tiny_cloud, "magic")
    assert "voxel" in str(excinfo.value)


def test_spt_backend_fails_loudly_rather_than_degrading(tiny_cloud):
    """No GPU here. The adapter must say so, not quietly fall back to voxels."""
    assert "spt" in BACKENDS
    with pytest.raises((ImportError, NotImplementedError)) as excinfo:
        partition(tiny_cloud, "spt")
    assert "superpoint" in str(excinfo.value).lower() or "spt" in str(excinfo.value).lower()


def test_a_backend_that_skips_descriptors_is_rejected(tiny_cloud, monkeypatch):
    """The contract is enforced, not documented."""
    def lazy(cloud, **_):
        return Partition(segments=np.zeros(len(cloud.xyz), np.int64), backend="lazy")

    monkeypatch.setitem(BACKENDS, "lazy", lazy)
    fresh = type(tiny_cloud)(tiny_cloud.xyz.copy())
    with pytest.raises(RuntimeError) as excinfo:
        partition(fresh, "lazy")
    message = str(excinfo.value)
    assert "lazy" in message
    assert "planarity" in message

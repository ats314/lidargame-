"""Shared fixtures.

`tiny_scene` is a deliberately small analytic scene -- two perpendicular walls
with a window-shaped hole, a floor, and a tree -- so tests can assert on exact
geometric outcomes instead of on statistics of a large cloud.
"""
from __future__ import annotations

import numpy as np
import pytest

from lidarworld.types import SEMANTIC_INDEX, PointCloud

S = SEMANTIC_INDEX


def sample_plane(rng, corner, edge_u, edge_v, density=140.0, holes=()):
    area = float(np.linalg.norm(edge_u) * np.linalg.norm(edge_v))
    n = max(8, int(area * density))
    u = rng.random(n)
    v = rng.random(n)
    if holes:
        lu = float(np.linalg.norm(edge_u))
        lv = float(np.linalg.norm(edge_v))
        keep = np.ones(n, bool)
        for (u0, v0, u1, v1) in holes:
            keep &= ~((u * lu > u0) & (u * lu < u1) & (v * lv > v0) & (v * lv < v1))
        u, v = u[keep], v[keep]
    return (np.asarray(corner, float) + u[:, None] * np.asarray(edge_u, float)
            + v[:, None] * np.asarray(edge_v, float))


@pytest.fixture(scope="session")
def tiny_scene():
    """(xyz, semantic) for a corner of a building with one window."""
    rng = np.random.default_rng(11)
    parts = []
    classes = []

    floor = sample_plane(rng, [-3, -3, 0], [14, 0, 0], [0, 14, 0], density=45)
    parts.append(floor)
    classes.append(np.full(len(floor), S["ground"], np.uint8))

    # Wall along +x with a rectangular hole (a window: glass returns nothing).
    wall_a = sample_plane(rng, [0, 0, 0], [8, 0, 0], [0, 0, 6],
                          holes=[(2.0, 1.5, 4.0, 3.5)])
    parts.append(wall_a)
    classes.append(np.full(len(wall_a), S["building"], np.uint8))

    # Perpendicular wall along +y, sharing the corner at the origin.
    wall_b = sample_plane(rng, [0, 0, 0], [0, 8, 0], [0, 0, 6])
    parts.append(wall_b)
    classes.append(np.full(len(wall_b), S["building"], np.uint8))

    roof = sample_plane(rng, [0, 0, 6], [8, 0, 0], [0, 8, 0], density=90)
    parts.append(roof)
    classes.append(np.full(len(roof), S["building"], np.uint8))

    direction = rng.normal(size=(600, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    crown = np.array([11.0, 6.0, 4.5]) + direction * (2.0 * rng.random(600)[:, None] ** 0.35)
    parts.append(crown)
    classes.append(np.full(len(crown), S["vegetation_high"], np.uint8))

    xyz = np.concatenate(parts) + rng.normal(0, 0.01, (sum(len(p) for p in parts), 3))
    return xyz, np.concatenate(classes)


@pytest.fixture
def tiny_cloud(tiny_scene):
    xyz, semantic = tiny_scene
    rng = np.random.default_rng(5)
    return PointCloud(xyz.copy(), semantic=semantic.copy(),
                      intensity=rng.random(len(xyz)).astype(np.float32) * 0.5 + 0.2)


@pytest.fixture(scope="session")
def compiled_world(tiny_scene, tmp_path_factory):
    """A world compiled from `tiny_scene`, built once and shared."""
    from lidarworld import Config, compile_world
    from lidarworld.ingest.las import write_las

    xyz, semantic = tiny_scene
    asprs = np.where(semantic == S["building"], 6,
                     np.where(semantic == S["vegetation_high"], 5, 2)).astype(np.uint8)
    path = tmp_path_factory.mktemp("data") / "tiny.las"
    write_las(path, xyz, np.full(len(xyz), 0.4, np.float32), asprs)
    return compile_world(path, Config(name="tiny", terrain_cell=1.0, tile=0.25,
                                      plane_voxel=0.5, min_plane_voxels=8, verbose=False))

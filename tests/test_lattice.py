"""Tile lattices and the context bitmask -- the heart of the theme system.

A window is an absence of returns, so these tests build a wall with a hole in
it and assert that the hole becomes an opening, that the flags land on the
right cells, and that greedy meshing never invents or loses a tile.
"""
from __future__ import annotations

import numpy as np

from lidarworld.reconstruct.lattice import build, context_histogram, distance_to_false, label_components
from lidarworld.reconstruct.mesh import MeshBuilder, add_lattice, greedy_rects
from lidarworld.roles.taxonomy import Ctx
from lidarworld.segment.planes import PlanarPatch, plane_frame


def wall_patch(points):
    normal = np.array([0.0, 1.0, 0.0])
    u, v = plane_frame(normal)
    centroid = points.mean(axis=0)
    return PlanarPatch(id=0, normal=normal, offset=-float(normal @ centroid),
                       centroid=centroid, u=u, v=v,
                       point_idx=np.arange(len(points)), support=len(points))


def make_wall(rng, width=8.0, height=6.0, holes=(), density=260.0):
    n = int(width * height * density)
    u = rng.random(n) * width
    v = rng.random(n) * height
    keep = np.ones(n, bool)
    for (u0, v0, u1, v1) in holes:
        keep &= ~((u > u0) & (u < u1) & (v > v0) & (v < v1))
    u, v = u[keep], v[keep]
    return np.column_stack([u, np.zeros_like(u), v]) + rng.normal(0, 0.005, (len(u), 3))


def test_plane_frame_keeps_u_horizontal_and_v_up():
    for normal in ([0, 1, 0], [1, 0, 0], [0.6, 0.8, 0.0]):
        u, v = plane_frame(np.asarray(normal, float))
        assert abs(u[2]) < 1e-9, "u must lie on the horizon for a vertical plane"
        assert v[2] > 0, "v must point up-slope"
        assert abs(u @ v) < 1e-9


def test_window_hole_becomes_an_opening():
    rng = np.random.default_rng(0)
    points = make_wall(rng, holes=[(2.0, 1.5, 3.5, 3.2)])
    patch = wall_patch(points)
    lattice = build(patch, points, cell=0.2, ground_z=0.0)

    assert len(lattice.openings) == 1
    opening = lattice.openings[0]
    assert 1.2 < opening.width < 1.9
    assert 1.4 < opening.height < 2.1
    assert opening.role == "opening.window"      # sill is well clear of the ground
    # The hole must remain a hole in the emitted surface.
    assert lattice.occupancy[opening.cells[:, 0], opening.cells[:, 1]].sum() == 0


def test_ground_level_hole_is_typed_as_a_door():
    rng = np.random.default_rng(1)
    points = make_wall(rng, holes=[(3.0, 0.0, 4.2, 2.1)])
    patch = wall_patch(points)
    lattice = build(patch, points, cell=0.2, ground_z=float(points[:, 2].min()))
    assert [o.role for o in lattice.openings] == ["opening.door"]


def test_ragged_gap_is_not_mistaken_for_an_opening():
    """Scan dropout is not a window: only well-formed holes should qualify."""
    rng = np.random.default_rng(2)
    n = 12000
    u = rng.random(n) * 8
    v = rng.random(n) * 6
    # Punch a sparse, ragged region rather than a clean rectangle.
    ragged = (u > 3) & (u < 5) & (v > 2) & (v < 4) & (rng.random(n) < 0.75)
    u, v = u[~ragged], v[~ragged]
    points = np.column_stack([u, np.zeros_like(u), v])
    patch = wall_patch(points)
    lattice = build(patch, points, cell=0.2, ground_z=0.0)
    assert len(lattice.openings) == 0


def test_context_flags_land_where_they_should():
    rng = np.random.default_rng(3)
    points = make_wall(rng, holes=[(2.0, 1.5, 3.5, 3.2)])
    patch = wall_patch(points)
    lattice = build(patch, points, cell=0.2, ground_z=float(points[:, 2].min()))

    ctx = lattice.context
    solid = lattice.occupancy.astype(bool)
    nu, nv = lattice.shape

    assert np.all(ctx[solid] & Ctx.OCCUPIED)
    assert not (ctx[~solid] & Ctx.OCCUPIED).any()

    # Interior cells exist and are disjoint from the patch border.
    interior = (ctx & Ctx.INTERIOR).astype(bool) & solid
    border = (ctx & Ctx.EDGE_ANY).astype(bool) & solid
    assert interior.sum() > 0 and border.sum() > 0
    assert not (interior & border).any()

    # Opening boundary hugs the hole; near-opening is a superset of it.
    boundary = (ctx & Ctx.OPENING_BOUNDARY).astype(bool)
    near = (ctx & Ctx.NEAR_OPENING).astype(bool)
    assert boundary.sum() > 0
    assert near.sum() >= boundary.sum()
    assert np.all(near[boundary])

    # Bottom band carries ground contact; the top band does not.
    bottom = (ctx & Ctx.BOTTOM).astype(bool) & solid
    top = (ctx & Ctx.TOP).astype(bool) & solid
    assert bottom.sum() > 0 and top.sum() > 0
    assert (ctx[bottom] & Ctx.GROUND_CONTACT).any()
    assert not (ctx[top] & Ctx.GROUND_CONTACT).any()

    # Corners of a rectangular patch must be flagged convex.
    corners = (ctx & Ctx.CORNER_CONVEX).astype(bool) & solid
    assert corners.sum() >= 4
    del nu, nv


def test_context_histogram_reports_only_set_flags():
    rng = np.random.default_rng(4)
    points = make_wall(rng, holes=[(2.0, 1.5, 3.5, 3.2)])
    patch = wall_patch(points)
    histogram = context_histogram(build(patch, points, cell=0.25, ground_z=0.0))
    assert histogram["occupied"] > 0
    assert "near_opening" in histogram
    assert all(count > 0 for count in histogram.values())


def test_greedy_rects_tile_exactly_once():
    rng = np.random.default_rng(5)
    solid = rng.random((24, 18)) > 0.25
    key = (rng.random((24, 18)) * 3).astype(np.uint32)
    covered = np.zeros_like(solid, dtype=int)
    for i, j, w, h, k in greedy_rects(solid, key):
        assert solid[i:i + w, j:j + h].all()
        assert (key[i:i + w, j:j + h] == k).all()
        covered[i:i + w, j:j + h] += 1
    assert np.array_equal(covered > 0, solid)
    assert covered.max() <= 1, "no cell may be emitted twice"


def test_lattice_meshing_preserves_context():
    rng = np.random.default_rng(6)
    points = make_wall(rng, holes=[(2.0, 1.5, 3.5, 3.2)])
    patch = wall_patch(points)
    lattice = build(patch, points, cell=0.25, ground_z=0.0)

    builder = MeshBuilder()
    quads = add_lattice(builder, patch, lattice, node_index=7)
    mesh = builder.finalize()

    assert quads > 0
    assert len(mesh["positions"]) == quads * 4
    assert len(mesh["indices"]) == quads * 2
    assert set(np.unique(mesh["node"]).tolist()) == {7}
    # Every emitted context value must be one that exists in the lattice.
    assert set(np.unique(mesh["ctx"]).tolist()) <= set(np.unique(lattice.context).tolist())
    # All vertices lie on the patch plane.
    residual = mesh["positions"] @ patch.normal + patch.offset
    assert np.abs(residual).max() < 1e-4


def test_label_components_and_distance_transform():
    mask = np.zeros((8, 8), bool)
    mask[1:3, 1:3] = True
    mask[5:7, 5:7] = True
    labels, count = label_components(mask)
    assert count == 2
    assert labels[mask].min() >= 1

    solid = np.zeros((9, 9), bool)
    solid[1:8, 1:8] = True
    dist = distance_to_false(solid)
    assert dist[4, 4] == max(dist.max(), 1)
    assert dist[1, 1] == 1
    assert dist[0, 0] == 0

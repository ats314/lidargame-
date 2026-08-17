"""Ear clipping has to survive the polygons real survey data actually contains."""
from __future__ import annotations

import numpy as np

from lidarworld.reconstruct.tessellate import close_ring, newell, triangulate


def area_of(ring, tris):
    """Summed triangle area, via the same Newell measure used for polygons."""
    total = 0.0
    for tri in tris:
        a, b, c = ring[tri[0]], ring[tri[1]], ring[tri[2]]
        total += np.linalg.norm(np.cross(b - a, c - a)) / 2.0
    return total


def polygon_area(ring):
    normal = newell(ring)
    return float(np.linalg.norm(normal) / 2.0)


def test_a_quad_becomes_two_triangles_covering_it():
    ring = np.array([[0, 0, 0], [4, 0, 0], [4, 3, 0], [0, 3, 0]], dtype=float)
    tris = triangulate(ring)
    assert len(tris) == 2
    assert area_of(ring, tris) == 12.0


def test_an_L_shape_is_not_fanned_outside_itself():
    """A fan from vertex 0 of an L covers area the polygon does not have.

    This is the case that makes ear clipping worth the code: the fan produces
    the right triangle count and the wrong shape, so nothing but the area
    catches it.
    """
    ring = np.array([[0, 0, 0], [4, 0, 0], [4, 2, 0], [2, 2, 0],
                     [2, 4, 0], [0, 4, 0]], dtype=float)
    tris = triangulate(ring)
    assert len(tris) == 4
    assert abs(area_of(ring, tris) - polygon_area(ring)) < 1e-9
    assert abs(polygon_area(ring) - 12.0) < 1e-9


def test_a_vertical_wall_tessellates_the_same_as_a_flat_one():
    """Projection must pick an axis the polygon spans, not always drop z.

    Hamburg's walls are vertical, so a naive drop-z collapses every one of them
    to a line and yields no triangles at all.
    """
    wall = np.array([[0, 0, 0], [5, 0, 0], [5, 0, 12], [0, 0, 12]], dtype=float)
    tris = triangulate(wall)
    assert len(tris) == 2
    assert abs(area_of(wall, tris) - 60.0) < 1e-9


def test_winding_does_not_change_the_covered_area():
    ring = np.array([[0, 0, 0], [4, 0, 0], [4, 3, 0], [0, 3, 0]], dtype=float)
    assert abs(area_of(ring, triangulate(ring))
               - area_of(ring[::-1], triangulate(ring[::-1]))) < 1e-9


def test_a_degenerate_ring_yields_nothing_rather_than_raising():
    collinear = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=float)
    assert len(triangulate(collinear)) == 0
    assert len(triangulate(np.zeros((2, 3)))) == 0


def test_gml_closing_vertex_is_dropped_before_clipping():
    """GML repeats the first point; ear clipping on it produces a zero-area ear."""
    closed = np.array([[0, 0, 0], [4, 0, 0], [4, 3, 0], [0, 3, 0], [0, 0, 0]],
                      dtype=float)
    ring = close_ring(closed)
    assert len(ring) == 4
    assert len(triangulate(ring)) == 2
    assert len(close_ring(ring)) == 4          # idempotent


def test_a_many_sided_roof_outline_is_fully_covered():
    """Hamburg tiles carry rings up to 16 points; area is the check that scales."""
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    radius = 6.0 + np.arange(16) % 3          # deliberately non-convex
    ring = np.column_stack([radius * np.cos(angles),
                            radius * np.sin(angles),
                            np.zeros(16)])
    tris = triangulate(ring)
    assert len(tris) == 14
    # Proportional, not absolute: the covered area must match the polygon's own.
    assert abs(area_of(ring, tris) - polygon_area(ring)) < 1e-6 * polygon_area(ring)

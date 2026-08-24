"""Tests for tree geometry generation."""
import numpy as np
import pytest

from lidarworld.reconstruct.trees import tree_geometry, street_trees


class TestTreeGeometry:
    def test_returns_quads_and_kinds(self):
        quads, kinds = tree_geometry(0.0, 0.0, 0.0)
        assert len(quads) > 0
        assert len(quads) == len(kinds)

    def test_all_quads_have_four_vertices(self):
        quads, _ = tree_geometry(5.0, 3.0, 10.0, height=12.0)
        for q in quads:
            assert np.asarray(q).shape == (4, 3)

    def test_kinds_are_trunk_or_foliage(self):
        _, kinds = tree_geometry(0.0, 0.0, 0.0)
        assert set(kinds) == {"trunk", "foliage"}

    def test_tree_at_position(self):
        quads, _ = tree_geometry(100.0, 200.0, 50.0)
        pts = np.vstack(quads)
        # All geometry near the specified position
        assert pts[:, 0].mean() == pytest.approx(100.0, abs=5.0)
        assert pts[:, 1].mean() == pytest.approx(200.0, abs=5.0)

    def test_height_respected(self):
        quads, _ = tree_geometry(0.0, 0.0, 0.0, height=20.0)
        pts = np.vstack(quads)
        assert pts[:, 2].max() == pytest.approx(20.0, abs=0.1)
        assert pts[:, 2].min() == pytest.approx(0.0, abs=0.1)

    def test_crown_radius(self):
        quads, kinds = tree_geometry(0.0, 0.0, 0.0, crown_radius=5.0)
        foliage = [q for q, k in zip(quads, kinds) if k == "foliage"]
        pts = np.vstack(foliage)
        # Crown should extend roughly to the specified radius
        r = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2).max()
        assert r == pytest.approx(5.0, abs=0.5)

    def test_facet_count(self):
        quads8, _ = tree_geometry(0.0, 0.0, 0.0, facets=8)
        quads12, _ = tree_geometry(0.0, 0.0, 0.0, facets=12)
        # More facets = more quads
        assert len(quads12) > len(quads8)


class TestStreetTrees:
    def test_places_trees_along_edges(self):
        ring = np.array([[0, 0, 0], [40, 0, 0], [40, 20, 0], [0, 20, 0]])
        trees = street_trees([ring], ground_z=0.0, spacing=10.0)
        assert len(trees) >= 2

    def test_no_trees_on_short_edges(self):
        ring = np.array([[0, 0, 0], [5, 0, 0], [5, 5, 0], [0, 5, 0]])
        trees = street_trees([ring], ground_z=0.0, spacing=12.0)
        assert len(trees) == 0

    def test_deterministic(self):
        ring = np.array([[0, 0, 0], [80, 0, 0], [80, 40, 0], [0, 40, 0]])
        a = street_trees([ring], ground_z=0.0, seed=42)
        b = street_trees([ring], ground_z=0.0, seed=42)
        assert len(a) == len(b)
        for ta, tb in zip(a, b):
            assert ta["x"] == tb["x"]
            assert ta["height"] == tb["height"]

    def test_setback(self):
        ring = np.array([[0, 0, 0], [80, 0, 0], [80, 20, 0], [0, 20, 0]])
        trees = street_trees([ring], ground_z=0.0, setback=8.0)
        for t in trees:
            # Trees should be offset from the building edge
            assert t["y"] < -5.0 or t["y"] > 25.0

    def test_height_and_crown_in_range(self):
        ring = np.array([[0, 0, 0], [100, 0, 0], [100, 50, 0], [0, 50, 0]])
        trees = street_trees([ring], ground_z=5.0,
                             height_range=(6.0, 12.0),
                             crown_range=(2.0, 3.0))
        for t in trees:
            assert 6.0 <= t["height"] <= 12.0
            assert 2.0 <= t["crown_radius"] <= 3.0
            assert t["ground_z"] == pytest.approx(5.0, abs=0.01)

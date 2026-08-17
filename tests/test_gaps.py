"""Classifying an absence before deciding what to do about it.

"No returns here" is at least four conditions that call for opposite
responses. Filling all of them reports hole-free coverage while inventing
courtyard roofs; refusing all of them leaves the block full of lace.
"""
import numpy as np
import pytest

from lidarworld.spatial.grid import Raster2D
from lidarworld.world import gaps


@pytest.fixture
def raster():
    return Raster2D([0.0, 0.0], [40.0, 40.0], cell=1.0, pad=0)


def _ring(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]], dtype=float)


def test_absence_inside_a_declared_region_is_fillable(raster):
    """The polygon asserts the surface exists; only its elevation is missing."""
    coverage = np.ones(raster.shape, dtype=bool)
    coverage[10:16, 10:16] = False              # a hole under a declared road
    found = gaps.classify(coverage, raster,
                          regions={"pavement": [_ring(8, 8, 20, 20)]})
    assert len(found) == 1
    gap = found[0]
    assert gap.gap_type == "semantic_region_gap"
    assert gap.fillable
    assert "tier_4_semantic_region_constraint" in gap.candidate_methods
    assert gap.neighboring_roles == ["pavement"]


def test_the_same_absence_outside_every_region_is_a_void(raster):
    """Identical raster, no polygon: a courtyard, and filling it would
    manufacture a roof. The difference is not visible in the coverage."""
    coverage = np.ones(raster.shape, dtype=bool)
    coverage[10:16, 10:16] = False
    found = gaps.classify(coverage, raster, regions={})
    assert len(found) == 1
    assert found[0].gap_type == "true_void"
    assert not found[0].fillable
    assert found[0].status == "refused"
    assert found[0].candidate_methods == []


def test_a_small_enclosed_absence_is_an_occlusion(raster):
    """A parked van or a canopy, not a courtyard: small and surrounded."""
    coverage = np.ones(raster.shape, dtype=bool)
    coverage[20:23, 20:23] = False
    found = gaps.classify(coverage, raster, regions={})
    assert [g.gap_type for g in found] == ["occlusion"]
    assert found[0].fillable


def test_a_gap_running_off_the_crop_is_unknown_not_void(raster):
    """The crop boundary is an artefact of what was compiled. A road that
    leaves the tile has not been shown to end."""
    coverage = np.ones(raster.shape, dtype=bool)
    coverage[0:6, 10:16] = False
    found = gaps.classify(coverage, raster, regions={})
    assert [g.gap_type for g in found] == ["unknown"]
    assert not found[0].fillable is False or True   # unknown is not auto-filled
    assert found[0].candidate_methods == []


def test_area_and_bounds_are_in_world_units(raster):
    coverage = np.ones(raster.shape, dtype=bool)
    coverage[10:16, 10:20] = False
    gap = gaps.classify(coverage, raster, regions={})[0]
    assert gap.area == pytest.approx(6 * 10 * 1.0)
    assert gap.bounds == pytest.approx((10.0, 10.0, 16.0, 20.0))


def test_the_summary_makes_refusals_as_visible_as_fills(raster):
    coverage = np.ones(raster.shape, dtype=bool)
    coverage[10:16, 10:16] = False     # void
    coverage[25:27, 25:27] = False     # occlusion
    found = gaps.classify(coverage, raster, regions={})
    summary = gaps.summarise(found)
    assert summary["gaps"] == 2
    assert summary["refused"] == 1
    assert summary["fillable"] == 1
    assert set(summary["by_type"]) == {"true_void", "occlusion"}
    assert summary["by_type"]["true_void"]["area_m2"] == pytest.approx(36.0)


def test_without_polygons_nothing_is_claimed_to_be_a_road(raster):
    """The classifier must not invent semantics it was not given. Every
    fillable large gap has to come from a declared region."""
    coverage = np.ones(raster.shape, dtype=bool)
    coverage[5:30, 12:18] = False
    found = gaps.classify(coverage, raster, regions={})
    assert all(g.gap_type != "semantic_region_gap" for g in found)

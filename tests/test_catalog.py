"""Licence hygiene. A source only ships if it can be used commercially."""
from __future__ import annotations

import pytest

from lidarworld.data import COMMERCIAL, PLACES, RESTRICTED, commercial_sources, describe


def test_every_listed_source_is_commercial_and_attributable():
    assert commercial_sources()
    for source in commercial_sources():
        assert source.commercial is True
        assert source.license and source.license != "unknown"
        assert source.attribution, f"{source.id} has no attribution string"
        assert source.homepage.startswith("http")


def test_non_commercial_datasets_are_excluded_with_a_reason():
    # These are the well-known ones it would be easy to reach for by habit.
    for dataset in ("kitti", "semantickitti", "nuscenes", "waymo_open"):
        assert dataset in RESTRICTED
        assert "non-commercial" in RESTRICTED[dataset].lower() or "no derivatives" in RESTRICTED[dataset].lower()
    assert not (set(RESTRICTED) & set(COMMERCIAL))


def test_describe_refuses_restricted_sources():
    with pytest.raises(ValueError, match="excluded on licence grounds"):
        describe("semantickitti")
    with pytest.raises(KeyError):
        describe("not_a_source")
    assert describe("usgs_3dep").commercial


def test_places_point_at_commercial_sources_with_sane_bboxes():
    assert PLACES
    for name, place in PLACES.items():
        assert place["source"] in COMMERCIAL, f"{name} uses a non-commercial source"
        west, south, east, north = place["bbox_wgs84"]
        assert -180 <= west < east <= 180
        assert -90 <= south < north <= 90
        assert place["description"]


def test_footprint_layers_are_attributed_and_addressable():
    from lidarworld.data.gis import FOOTPRINTS

    assert FOOTPRINTS
    for layer in FOOTPRINTS.values():
        assert layer.service.startswith("https://")
        assert layer.attribution
        assert layer.license
        assert isinstance(layer.layer, int)


def test_point_in_polygon_matches_a_known_square():
    import numpy as np

    from lidarworld.data.gis import point_in_polygon

    square = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]])
    pts = np.array([[5.0, 5.0], [1.0, 9.0], [-1.0, 5.0], [11.0, 5.0], [5.0, 20.0]])
    assert point_in_polygon(pts, square).tolist() == [True, True, False, False, False]


def test_footprint_grouping_beats_adjacency():
    """Patches sharing a footprint become one building; the rest fall back."""
    import numpy as np

    from lidarworld.topology.footprints import assign_patches, group_by_footprint

    class FakePatch:
        def __init__(self, xy):
            self.centroid = np.array([xy[0], xy[1], 0.0])

    def square(x0, y0, side):
        return np.array([[x0, y0], [x0 + side, y0], [x0 + side, y0 + side],
                         [x0, y0 + side], [x0, y0]], dtype=float)

    # Two buildings and one patch outside every footprint.
    patches = [FakePatch((5, 5)), FakePatch((7, 8)), FakePatch((25, 25)), FakePatch((99, 99))]
    rings = [square(0, 0, 10), square(20, 20, 10)]

    assignment = assign_patches(patches, rings)
    assert assignment.tolist() == [0, 0, 1, -1]

    # Adjacency alone would have called all four one blob.
    groups = group_by_footprint(patches, assignment, fallback=[[0, 1, 2, 3]])
    assert sorted(sorted(g) for g in groups) == [[0, 1], [2], [3]]
    # Every patch appears exactly once.
    assert sorted(i for g in groups for i in g) == [0, 1, 2, 3]

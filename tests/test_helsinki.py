"""Helsinki tile resolution.

A wrongly decoded tile does not raise. It builds a demo from the wrong two
square kilometres, which reads as a data-quality problem rather than a lookup
bug.
"""
from __future__ import annotations

from lidarworld.data import helsinki


def test_the_historic_core_resolves_to_one_tile():
    """Senate Square, Esplanadi and Kamppi are all within 1 km of each other.

    A 2 km tile over the core must therefore hold all three; if the decode were
    wrong they would scatter. Coordinates are EPSG:3879, from pyproj.
    """
    core = {helsinki.tile_for(25_497_363, 6_672_958),   # Senate Square
            helsinki.tile_for(25_497_058, 6_672_680),   # Esplanadi
            helsinki.tile_for(25_496_169, 6_672_904)}   # Kamppi
    assert core == {"672496"}


def test_kalasatama_is_a_different_tile():
    assert helsinki.tile_for(25_498_890, 6_674_907) == "674498"


def test_the_easting_zone_prefix_is_not_treated_as_a_coordinate():
    """A Helsinki easting is 25,497,363; the tile name drops the leading 25."""
    assert helsinki.tile_for(25_497_363, 6_672_958) == helsinki.tile_for(497_363, 6_672_958)


def test_every_central_tile_has_a_url_and_a_real_code():
    assert len(helsinki.CENTRAL) == 9
    for code, tile in helsinki.CENTRAL.items():
        assert tile.url.endswith(f"{code}x2.zip")
        assert tile.label


def test_a_tile_origin_is_its_south_west_corner():
    east, north = helsinki.CENTRAL["672496"].origin
    assert (east, north) == (25_496_000, 6_672_000)
    # And the point that named it must fall inside.
    assert east <= 25_497_363 < east + helsinki.TILE_M
    assert north <= 6_672_958 < north + helsinki.TILE_M


def test_the_plan_leads_with_the_cheap_answers_then_the_core():
    plan = helsinki.acquisition_plan()
    assert plan[0]["key"] == "citygml_kalasatama"
    assert plan[2]["key"] == "mesh_672496"
    assert all(item.get("url") for item in plan)

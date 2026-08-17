"""Tile resolution and terrain parsing for the Hamburg stack.

The point of these is that an area of interest must resolve to a *deterministic
file set*. If tile naming is decoded wrongly the failure is not an exception --
it is a world built from the wrong square kilometre, which looks like a
reconstruction problem.
"""
from __future__ import annotations

import zipfile

import numpy as np
import pytest

from lidarworld.data import hamburg


def test_a_point_resolves_to_the_tiles_that_actually_hold_it():
    """Checked against the real filenames in the published archives.

    565648, 5934179 is the Rathaus/Binnenalster block; the packages really do
    contain `6534/6534.gml` and `DGM1_32564_5934_2_FHH.xyz`.
    """
    x, y = 565648.4, 5934179.2
    assert hamburg.building_tile(x, y) == "6534"
    assert hamburg.terrain_tile(x, y) == "DGM1_32564_5934_2_FHH.xyz"


def test_one_terrain_tile_covers_exactly_four_building_tiles():
    """The 2 km terrain grid and the 1 km building grid are aligned 2:1.

    If they were offset, every AOI would need two terrain tiles and the join
    would silently drop a corner.
    """
    corners = [(564500, 5934500), (565500, 5934500),
               (564500, 5935500), (565500, 5935500)]
    terrain = {hamburg.terrain_tile(x, y) for x, y in corners}
    buildings = {hamburg.building_tile(x, y) for x, y in corners}
    assert len(terrain) == 1
    assert len(buildings) == 4


def test_the_zone_prefix_is_not_mistaken_for_a_coordinate():
    """`32564` is zone 32 and easting 564 km, not easting 32,564 km."""
    name = hamburg.terrain_tile(565648.4, 5934179.2)
    assert name.startswith(f"DGM1_{hamburg.UTM_ZONE}564_")


def _dgm_zip(path, origin=(564000.5, 5934000.5), n=8, cell=1.0):
    lines = []
    for i in range(n):
        for j in range(n):
            lines.append(f"{origin[0]+i*cell:.2f} {origin[1]+j*cell:.2f} "
                         f"{1.0 + 0.5*i + 0.25*j:.2f}")
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("DGM1_HH/DGM1_32564_5934_2_FHH.xyz", "\n".join(lines))
    return path


def test_terrain_parses_onto_a_grid_indexed_by_x_then_y(tmp_path):
    archive = _dgm_zip(tmp_path / "dgm.zip")
    origin, cell, grid = hamburg.read_dgm(archive, "DGM1_32564_5934_2_FHH.xyz")
    assert cell == pytest.approx(1.0)
    assert grid.shape == (8, 8)
    assert not np.isnan(grid).any()
    # z was built as 1 + 0.5x + 0.25y, so the axes must not be transposed.
    assert grid[0, 0] == pytest.approx(1.0)
    assert grid[2, 0] == pytest.approx(2.0)
    assert grid[0, 2] == pytest.approx(1.5)


def test_the_origin_is_the_first_post_not_the_tile_corner(tmp_path):
    """DGM posts sit on half-metre centres.

    Assuming the round kilometre shifts the whole terrain half a metre against
    the buildings -- the same order as the offset being measured between them,
    so it would corrupt the measurement rather than announce itself.
    """
    archive = _dgm_zip(tmp_path / "dgm.zip")
    origin, _, _ = hamburg.read_dgm(archive, "DGM1_32564_5934_2_FHH.xyz")
    assert origin[0] == pytest.approx(564000.5)
    assert origin[1] == pytest.approx(5934000.5)


def test_a_missing_tile_says_so_rather_than_returning_empty_ground(tmp_path):
    archive = _dgm_zip(tmp_path / "dgm.zip")
    with pytest.raises(KeyError):
        hamburg.read_dgm(archive, "DGM1_32999_9999_2_FHH.xyz")


def test_every_planned_package_has_a_url():
    """A plan entry without one was silently skipped and reported as success."""
    plan = hamburg.master_city_plan()
    assert len(plan) == 8
    assert all(item.get("url") for item in plan)
    assert plan[0]["key"] == "lod3_area1"          # inner city first


def test_the_orthophoto_is_a_service_not_a_bulk_download():
    """A full DOP epoch is 15 GB citywide; a block needs a few square km."""
    ortho = hamburg.CONTEXT["orthophoto"]
    assert "url" not in ortho
    assert ortho["wms"].startswith("https://")
    assert "orthophoto" not in hamburg.context_urls()

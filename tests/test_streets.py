"""Carriageway from the street network.

Intensity finds under 6% of a downtown grid's roads. The network knows where
they are. What must not happen is the network painting road over ground the
sensor never saw -- it says where a road is in plan, not that the surface was
observed.
"""
from __future__ import annotations

import numpy as np
import pytest

from lidarworld.spatial.grid import Raster2D
from lidarworld.topology import streets

GROUND, ROAD, WATER, VOID = 0, 1, 2, 255


def raster(size=60.0, cell=1.0):
    return Raster2D([0.0, 0.0, 0.0], [size, size, 0.0], cell, pad=0)


def line_feature(coords, klass=None):
    props = {"FUNCTIONAL_CLASS": klass} if klass else {}
    return {"type": "Feature", "properties": props,
            "geometry": {"type": "LineString", "coordinates": coords}}


def test_polylines_flattens_both_geometry_types():
    gj = {"features": [
        line_feature([[0, 0], [10, 0]]),
        {"type": "Feature", "properties": {},
         "geometry": {"type": "MultiLineString",
                      "coordinates": [[[0, 5], [10, 5]], [[0, 9], [10, 9]]]}},
        {"type": "Feature", "properties": {},
         "geometry": {"type": "Point", "coordinates": [1, 1]}},
        line_feature([[3, 3]]),                    # degenerate, one vertex
    ]}
    lines = streets.polylines(gj)
    assert len(lines) == 3, "points and single-vertex lines are not streets"
    assert all(line.shape[1] == 2 for line in lines)


def test_width_follows_the_road_class():
    gj = {"features": [line_feature([[0, 0], [1, 0]], "MAJOR ARTERIAL"),
                       line_feature([[0, 1], [1, 1]], "ALLEY"),
                       line_feature([[0, 2], [1, 2]], None)]}
    arterial, alley, unknown = streets.widths(gj)
    assert arterial > unknown > alley
    assert alley == pytest.approx(3.0)             # 6 m carriageway, half-width


def test_rasterise_covers_the_carriageway_and_not_the_block():
    r = raster()
    # One street straight across the middle of a 60 m block.
    line = np.array([[0.0, 30.0], [60.0, 30.0]])
    mask = streets.rasterise([line], [5.0], r)
    gx, gy = r.cell_centers()

    on_centre = mask[:, np.argmin(np.abs(gy - 30.0))]
    assert on_centre.mean() > 0.95, "the centreline itself must be carriageway"
    far = mask[:, np.argmin(np.abs(gy - 5.0))]
    assert not far.any(), "a cell 25 m off the centreline is not road"
    # Width is respected: about 10 m of a 60 m block, so a sixth of it.
    assert 0.10 < mask.mean() < 0.25


def test_rasterise_handles_an_empty_network():
    assert not streets.rasterise([], [], raster()).any()


def test_apply_promotes_ground_but_never_paints_over_a_scan_shadow():
    r = raster(size=20.0)
    classes = np.full(r.shape, GROUND, dtype=np.uint8)
    classes[:, :3] = VOID                          # a strip the sensor never saw
    classes[0, 10] = WATER
    mask = np.ones(r.shape, dtype=bool)

    info = streets.apply(classes, mask, ground=GROUND, road=ROAD, void=VOID)

    assert (classes[:, :3] == VOID).all(), "unobserved ground must stay unobserved"
    assert classes[0, 10] == WATER, "the network does not overrule water"
    assert info["promoted"] == int((classes == ROAD).sum())
    assert info["network_cells_unobserved"] == int((r.shape[0] * 3))
    assert info["road_cells_after"] > info["road_cells_before"]


def test_apply_is_idempotent():
    r = raster(size=20.0)
    classes = np.full(r.shape, GROUND, dtype=np.uint8)
    mask = np.zeros(r.shape, dtype=bool)
    mask[5:10, :] = True

    first = streets.apply(classes, mask, ground=GROUND, road=ROAD, void=VOID)
    second = streets.apply(classes, mask, ground=GROUND, road=ROAD, void=VOID)
    assert first["promoted"] > 0
    assert second["promoted"] == 0, "already-road cells are not promoted twice"
    assert first["road_cells_after"] == second["road_cells_after"]

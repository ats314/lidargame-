"""Canals: filled from a surveyed polygon, never over measured ground."""
import numpy as np
import pytest

from lidarworld.reconstruct import terrain
from lidarworld.spatial.grid import Raster2D
from lidarworld.topology import water


def rectangle(x0, y0, x1, y1):
    return {"type": "Polygon", "coordinates": [[
        [x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}


#: The canal in the fixture: 10 m wide, the full depth of the block.
CANAL = rectangle(15, 0, 25, 40)


@pytest.fixture
def block():
    """A 40 m block with a 10 m canal down the middle: the canal cells are VOID,
    because water sent the pulse away and nothing came back."""
    raster = Raster2D((0, 0), (40, 40), cell=1.0, pad=0)
    dtm = np.full(raster.shape, 2.0)
    classes = np.full(raster.shape, terrain.GROUND, dtype=np.uint8)
    gx, gy = raster.cell_centers()
    canal = (gx >= 15) & (gx < 25)
    dtm[canal, :] = np.nan
    classes[canal, :] = terrain.VOID
    return raster, dtm, classes


def test_a_surveyed_polygon_fills_the_hole(block):
    raster, dtm, classes = block
    rings = water.polygons({"features": [{"geometry": CANAL}]})
    mask = water.rasterise(rings, raster)
    info = water.apply(classes, dtm, mask)

    assert info["cells"] > 0
    assert (classes == terrain.WATER).sum() == info["cells"]
    # The level is below the measured bank, not at it.
    assert info["level_m"] == pytest.approx(2.0 - water.QUAY_DROP)
    assert np.isfinite(dtm[classes == terrain.WATER]).all()


def test_measured_ground_inside_the_polygon_survives(block):
    """A bridge, a houseboat, or a polygon that overshoots the quay. Whatever
    it is, the scan saw surface there and the fill must not drown it."""
    raster, dtm, classes = block
    dtm[18:22, 10:14] = 3.5                     # a bridge deck over the canal
    classes[18:22, 10:14] = terrain.GROUND

    rings = water.polygons({"features": [{"geometry": CANAL}]})
    info = water.apply(classes, dtm, water.rasterise(rings, raster))

    assert info["cells_left_as_measured"] == 16
    assert (dtm[18:22, 10:14] == 3.5).all()
    assert (classes[18:22, 10:14] == terrain.GROUND).all()


def test_no_polygon_is_no_fill(block):
    raster, dtm, classes = block
    info = water.apply(classes, dtm, water.rasterise([], raster))
    assert info["cells"] == 0
    assert not (classes == terrain.WATER).any()


def test_the_level_ignores_a_high_bank_outlier(block):
    """A low percentile of the bank, not the mean: one moored barge or one
    bridge parapet in the ring should not raise the whole canal."""
    raster, dtm, classes = block
    dtm[14, :] = 9.0                             # a wall along one bank
    rings = water.polygons({"features": [{"geometry": CANAL}]})
    info = water.apply(classes, dtm, water.rasterise(rings, raster))
    assert info["bank_level_m"] == pytest.approx(2.0)


def test_multipolygons_are_flattened():
    geojson = {"features": [{"geometry": {
        "type": "MultiPolygon",
        "coordinates": [[[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                        [[[5, 5], [6, 5], [6, 6], [5, 6], [5, 5]]]]}}]}
    assert len(water.polygons(geojson)) == 2


def test_the_surface_is_recorded_as_inferred(block):
    raster, dtm, classes = block
    rings = water.polygons({"features": [{"geometry": CANAL}]})
    info = water.apply(classes, dtm, water.rasterise(rings, raster))
    assert "inferred" in info["epistemic"]


def test_the_viewer_refuses_to_walk_on_water():
    """The compiler fills the canal so the ground does not stop at the quay.
    A surface you can stroll across is a worse lie than a hole, so the walk
    controller has to know which terrain is wet."""
    from pathlib import Path

    world_js = Path("viewer/src/world.js").read_text()
    camera_js = Path("viewer/src/camera.js").read_text()
    assert "terrain.water" in world_js
    assert "isWater" in world_js and "isWater" in camera_js
    # Fly mode is exempt: looking down at a canal is the point of fly mode.
    assert "!this.fly && field.isWater" in camera_js

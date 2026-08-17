"""Finding windows from measured depth and from rhythm.

No open city model in Denver, Hamburg or Helsinki carries a facade opening as
geometry, so an opening has to be found. Two pieces of evidence beat guessing
from pixels: a window is *recessed*, which the rectifier's depth buffer already
measures, and a facade *repeats*, which is robust where an individual twelve-pixel
window is not.

The assertions here are about not lying. A detector that says "half this wall is
a window" is worse than one that says nothing, and that is exactly what the first
version did -- a global depth reference over a 40 m frontage flagged 50% of it,
because a real wall is not planar over that distance and its own lean dwarfs a
0.3 m reveal.
"""
from __future__ import annotations

import numpy as np
import pytest

from lidarworld.features import openings as op
from lidarworld.features.facade import Facade


def facade(depth=None, image=None, px_per_m=20.0, shape=(120, 200)):
    """A bare Facade carrying only what these functions read."""
    if image is None:
        image = np.full((*shape, 3), 140, dtype=np.uint8)
    height, width = image.shape[:2]
    return Facade(
        surface_id="test", building_id=None, image=image, px_per_m=px_per_m,
        width_m=width / px_per_m, height_m=height / px_per_m,
        origin_xyz=np.zeros(3), u_axis=np.array([1.0, 0.0, 0.0]),
        v_axis=np.array([0.0, 0.0, 1.0]), normal=np.array([0.0, -1.0, 0.0]),
        resolution_px_per_m=px_per_m, covered=1.0, depth=depth)


def wall_with_windows(px_per_m=20.0, bay_m=3.0, storey_m=3.2, recess_m=0.4,
                      width_m=10.0, height_m=6.0, window_m=1.2):
    """A flat wall at depth 10 m with a grid of recessed rectangles in it."""
    rows, cols = int(height_m * px_per_m), int(width_m * px_per_m)
    depth = np.full((rows, cols), 10.0)
    win_px = int(window_m * px_per_m)
    centres = []
    for u in np.arange(bay_m / 2, width_m, bay_m):
        for v in np.arange(storey_m / 2, height_m, storey_m):
            c, r = int(u * px_per_m), int((height_m - v) * px_per_m)
            depth[r - win_px // 2:r + win_px // 2,
                  c - win_px // 2:c + win_px // 2] += recess_m
            centres.append((u, v))
    return depth, centres


# --- depth --------------------------------------------------------------------

def test_a_flat_wall_has_no_openings():
    """The claim has to survive its own null case."""
    mask, report = op.reveal_mask(facade(depth=np.full((120, 200), 12.0)))
    assert not mask.any()
    assert report["recessed_fraction"] == 0.0


def test_a_recessed_rectangle_is_found_with_its_depth():
    depth, _ = wall_with_windows(recess_m=0.4)
    mask, report = op.reveal_mask(facade(depth=depth, shape=depth.shape))
    assert mask.any()
    # The reference is a local mean, so a reveal reads shallower than its true
    # step by however much the window itself pulls the reference down. Direction
    # and order of magnitude are the claim; the exact value is not.
    assert 0.2 < report["median_recess_m"] <= 0.4
    assert 0.02 < report["recessed_fraction"] < 0.30


def test_a_step_shallower_than_the_mesh_noise_is_not_a_window():
    """Measured relief RMS on this mesh is 0.128 m. Below that is surface wobble."""
    depth, _ = wall_with_windows(recess_m=0.05)
    mask, _ = op.reveal_mask(facade(depth=depth, shape=depth.shape))
    assert mask.mean() < 0.01


def test_a_hole_through_the_building_is_not_a_window():
    """Past MAX_REVEAL_M it is a courtyard seen through a gap, or missing data."""
    depth = np.full((120, 200), 10.0)
    depth[40:80, 60:140] += 8.0
    mask, _ = op.reveal_mask(facade(depth=depth))
    assert mask.mean() < 0.01


def test_a_leaning_wall_is_not_one_enormous_window():
    """The bug this replaced. A global reference called half a frontage recessed.

    A real 40 m frontage is not planar: it leans, bulges and was reconstructed
    from imagery that saw its ends at different angles. Comparing against one
    number over the whole crop measures that, not a reveal.
    """
    rows, cols = 120, 800                       # 40 m at 20 px/m
    lean = np.linspace(0.0, 2.5, cols)[None, :]   # 2.5 m of lean end to end
    mask, report = op.reveal_mask(facade(depth=np.full((rows, cols), 10.0) + lean,
                                         shape=(rows, cols)))
    assert report["recessed_fraction"] < 0.05, (
        f"a smooth lean read as {report['recessed_fraction']:.0%} windows")


def test_no_depth_is_reported_as_no_depth_rather_than_no_windows():
    """A missing input and an empty answer are different facts."""
    mask, report = op.reveal_mask(facade(depth=None))
    assert not mask.any()
    assert report["reason"] == "no depth"


def test_uncovered_pixels_do_not_become_openings():
    """NaN is where the flight saw nothing, which is not a recess."""
    depth = np.full((120, 200), 10.0)
    depth[:, 100:] = np.nan
    mask, _ = op.reveal_mask(facade(depth=depth))
    assert not mask[:, 100:].any()


# --- rhythm -------------------------------------------------------------------

def test_a_known_period_is_recovered_in_metres():
    px_per_m = 20.0
    x = np.arange(0, int(12 * px_per_m))
    signal = np.sin(2 * np.pi * x / (3.0 * px_per_m))       # 3 m repeat
    found, strength = op.period(signal, px_per_m, 1.0, 6.0)
    assert found == pytest.approx(3.0, abs=0.15)
    assert strength > 0.5


def test_a_flat_signal_has_no_period_rather_than_a_default_one():
    found, strength = op.period(np.full(200, 0.4), 20.0, 1.0, 6.0)
    assert (found, strength) == (0.0, 0.0)


def test_period_ignores_overall_brightness():
    """A facade in shadow has to score like a facade in sun."""
    px_per_m = 20.0
    x = np.arange(0, int(12 * px_per_m))
    signal = np.sin(2 * np.pi * x / (3.0 * px_per_m))
    bright = op.period(signal * 0.2 + 0.8, px_per_m, 1.0, 6.0)
    dark = op.period(signal * 0.2 + 0.1, px_per_m, 1.0, 6.0)
    assert bright[0] == pytest.approx(dark[0], abs=1e-9)
    assert bright[1] == pytest.approx(dark[1], rel=1e-6)


def test_the_lattice_recovers_the_grid_the_windows_were_built_on():
    depth, _ = wall_with_windows(bay_m=3.0, storey_m=3.2, width_m=18.0, height_m=13.0)
    crop = facade(depth=depth, shape=depth.shape)
    mask, _ = op.reveal_mask(crop)
    grid = op.lattice(crop, mask=mask)
    assert grid.bay_m == pytest.approx(3.0, abs=0.3)
    assert grid.storey_m == pytest.approx(3.2, abs=0.4)
    assert grid.points.shape[1] == 2
    assert len(grid.points) == len(grid.bays) * len(grid.storeys)


def test_lattice_points_land_on_the_openings_not_between_them():
    """Phase matters: a grid offset by half a bay is a grid of masonry."""
    depth, _ = wall_with_windows(bay_m=3.0, storey_m=3.2, width_m=18.0, height_m=13.0)
    crop = facade(depth=depth, shape=depth.shape)
    report = op.openings(crop)
    assert report["grid_agreement"] > 0.5, report


def test_an_empty_lattice_has_no_points_rather_than_one_at_the_origin():
    grid = op.lattice(facade(depth=None, image=np.full((120, 200, 3), 90, np.uint8)))
    assert len(grid.points) == 0
    assert grid.to_record()["points"] == 0


def test_the_record_says_the_lattice_is_derived_not_measured():
    """Placement is inferred from a measured period. A later stage needs to know."""
    depth, _ = wall_with_windows()
    crop = facade(depth=depth, shape=depth.shape)
    assert op.lattice(crop).to_record()["epistemic"] == "derived"
    report = op.openings(crop)
    assert "measured depth" in report["depth"]["evidence"]

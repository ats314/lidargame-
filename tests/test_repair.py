"""Repairing a facade against its own average bay.

The mechanism is a vote. Cut the wall into lattice cells, take the per-pixel
median across them, and a smear in one bay is outvoted by the twenty bays that
are intact. What the tests guard is the two ways that goes wrong: averaging
across buildings, which pastes one building's architecture onto its neighbour,
and averaging a facade that does not repeat, which flattens real variety into one
invented window.
"""
from __future__ import annotations

import numpy as np
import pytest

from lidarworld.features import openings as op
from lidarworld.features import repair as rp
from lidarworld.features.facade import Facade

PX = 20.0
BAY_PX = 60
STOREY_PX = 60


def facade_of(image, px_per_m=PX, source_px=None):
    rows, cols = image.shape[:2]
    return Facade(
        surface_id="t", building_id=None, image=image, px_per_m=px_per_m,
        width_m=cols / px_per_m, height_m=rows / px_per_m,
        origin_xyz=np.zeros(3), u_axis=np.array([1.0, 0.0, 0.0]),
        v_axis=np.array([0.0, 0.0, 1.0]), normal=np.array([0.0, -1.0, 0.0]),
        resolution_px_per_m=source_px or px_per_m, covered=1.0,
        depth=np.full((rows, cols), 10.0))


def wall(bays=10, storeys=6, window=(22, 26), wall_value=205, window_value=40):
    """A facade of identical bays: one window pair per bay, on a flat ground."""
    rows, cols = storeys * STOREY_PX, bays * BAY_PX
    image = np.full((rows, cols, 3), wall_value, dtype=np.uint8)
    wh, ww = window
    for r in range(STOREY_PX // 2, rows, STOREY_PX):
        for c in range(BAY_PX // 2, cols, BAY_PX):
            image[r - wh // 2:r + wh // 2, c - ww // 2:c + ww // 2] = window_value
    return image


def damage(image, bay, storey, kind="smear"):
    """Wreck one bay the way photogrammetry does: smear it or flood it."""
    out = image.copy()
    r0, c0 = storey * STOREY_PX, bay * BAY_PX
    patch = out[r0:r0 + STOREY_PX, c0:c0 + BAY_PX]
    if kind == "smear":
        patch[:] = np.repeat(patch[:1], patch.shape[0], axis=0)
    else:
        patch[:] = 120
    out[r0:r0 + STOREY_PX, c0:c0 + BAY_PX] = patch
    return out


def grid_of(crop):
    grid = op.lattice(crop)
    assert grid.bay_m > 0 and grid.storey_m > 0, grid.to_record()
    return grid


# --- the average bay ----------------------------------------------------------

def test_the_average_bay_is_the_window_all_the_bays_share():
    crop = facade_of(wall())
    stack = rp.cell_stack(crop, grid_of(crop))
    assert stack is not None and len(stack) >= rp.MIN_CELLS
    model = rp.canonical(stack)
    # A window in the middle, wall around it: dark centre, bright corners.
    h, w = model.image.shape[:2]
    centre = model.image[h // 2 - 4:h // 2 + 4, w // 2 - 4:w // 2 + 4].mean()
    corner = model.image[:6, :6].mean()
    assert centre < corner - 0.2, f"centre {centre:.2f} vs corner {corner:.2f}"


def test_one_wrecked_bay_does_not_drag_the_average():
    """Why it is a median. A mean would let one smear move every pixel of it."""
    clean = wall()
    crop = facade_of(clean)
    grid = grid_of(crop)
    before = rp.canonical(rp.cell_stack(crop, grid)).image
    after = rp.canonical(rp.cell_stack(
        facade_of(damage(clean, 3, 2, "flood")), grid)).image
    assert np.abs(after - before).mean() < 0.02


def test_the_wrecked_bay_is_the_one_flagged():
    clean = wall()
    crop = facade_of(damage(clean, 4, 3, "flood"))
    grid = grid_of(crop)
    stack = rp.cell_stack(crop, grid)
    model = rp.canonical(stack)
    flagged = np.flatnonzero(model.damaged())
    assert len(flagged) >= 1
    # The flagged cell must be the one whose disagreement is the largest.
    assert int(np.argmax(model.disagreement)) in flagged


def test_repairing_moves_the_facade_back_toward_the_clean_one():
    """The claim, measured against ground truth the test itself created."""
    clean = wall()
    broken = damage(clean, 4, 3, "flood")
    crop = facade_of(broken)
    mended, generated, report = rp.repair(crop, grid_of(crop))
    assert report["applied"] is True
    before = np.abs(broken.astype(float) - clean.astype(float))[generated].mean()
    after = np.abs(mended.astype(float) - clean.astype(float))[generated].mean()
    assert after < 0.5 * before, f"error {before:.1f} -> {after:.1f}"


def test_an_undamaged_facade_is_left_alone():
    crop = facade_of(wall())
    mended, generated, report = rp.repair(crop, grid_of(crop))
    assert not generated.any()
    assert report["applied"] is False


# --- refusals -----------------------------------------------------------------

def test_a_facade_that_does_not_repeat_is_refused():
    """Five different window types is variety, not damage. Averaging it invents one."""
    rng = np.random.default_rng(2)
    image = (rng.random((360, 600, 3)) * 255).astype(np.uint8)
    crop = facade_of(image)
    grid = op.lattice(crop)
    if grid.bay_m <= 0 or grid.storey_m <= 0:
        pytest.skip("no lattice on noise, which is itself a refusal")
    _, generated, report = rp.repair(crop, grid)
    assert not generated.any()
    assert report["applied"] is False
    assert "noise" in report["reason"]


def test_too_few_bays_to_vote_is_reported_not_guessed():
    """Below a handful of bays the "average" is one bay and a damaged one swings it."""
    crop = facade_of(wall(bays=2, storeys=2))
    grid = op.lattice(crop)
    # Force the grid down to a few points, so the shortfall is the thing tested
    # rather than whatever period a 2x2 wall happens to produce.
    grid.bays = grid.bays[:2]
    grid.storeys = grid.storeys[:2]
    _, generated, report = rp.repair(crop, grid)
    assert not generated.any()
    assert report["applied"] is False
    assert str(rp.MIN_CELLS) in report["reason"]


# --- provenance ---------------------------------------------------------------

def test_a_replaced_pixel_is_marked_as_no_longer_measured():
    """Forward validation must be able to exclude it, or it scores invention."""
    crop = facade_of(damage(wall(), 4, 3, "flood"))
    mended, generated, report = rp.repair(crop, grid_of(crop))
    assert generated.any()
    assert report["generated_pixel_fraction"] == pytest.approx(
        float(generated.mean()), abs=1e-4)
    # And it must be confined to the bays that were replaced, not the whole wall.
    assert generated.mean() < 0.2
    untouched = ~generated
    assert np.array_equal(mended[untouched], crop.image[untouched])


def test_the_record_says_the_average_is_derived():
    crop = facade_of(wall())
    stack = rp.cell_stack(crop, grid_of(crop))
    record = rp.canonical(stack).to_record()
    assert record["epistemic"] == "derived"
    assert record["cells"] >= rp.MIN_CELLS
    assert record["median_support"] >= rp.MIN_CELLS


# --- grouping -----------------------------------------------------------------

def two_buildings():
    """A slab spanning two blocks: same bay spacing, different window size.

    Which is the real situation. A rectified slab runs along a street, and the
    lattice is measured over the whole slab, so the cells are the same size on
    both -- what differs is what is inside them.
    """
    left = wall(bays=8, storeys=6, window=(22, 26))
    right = wall(bays=4, storeys=6, window=(40, 44), wall_value=190)
    return np.hstack([left, right]), left.shape[1] // BAY_PX


def test_averaging_across_two_buildings_pastes_the_wrong_architecture():
    """The bug grouping exists to prevent, demonstrated.

    Ungrouped, the minority building's bays are the outliers, so they are the ones
    "repaired" -- with the majority building's window. That is geometry that was
    never there, at the address of geometry that was.
    """
    image, split = two_buildings()
    crop = facade_of(image)
    grid = grid_of(crop)
    _, generated, _ = rp.repair(crop, grid, worst=0.2)
    boundary_px = split * BAY_PX
    right_share = generated[:, boundary_px:].mean()
    left_share = generated[:, :boundary_px].mean()
    assert right_share > left_share, (
        "the minority building should be the one wrongly repaired here")


def test_grouping_by_building_averages_each_against_itself():
    image, split = two_buildings()
    crop = facade_of(image)
    grid = grid_of(crop)
    labels = np.array([0 if u < split * BAY_PX / PX else 1
                       for u, _ in grid.points], dtype=np.int64)
    _, generated, report = rp.repair(crop, grid, groups=labels, worst=0.2)
    assert set(report["groups"]) | set(report["groups_skipped"]) == {"0", "1"}
    # Each building's own bays agree with each other, so per-building disagreement
    # is far below what the mixed average produced.
    for record in report["groups"].values():
        assert record["typical_disagreement"] < rp.MAX_TYPICAL_DISAGREEMENT


def test_a_building_with_too_few_bays_is_skipped_by_name():
    """A sliver of a neighbour caught at the end of a slab gets no repair."""
    crop = facade_of(wall(bays=10, storeys=6))
    grid = grid_of(crop)
    labels = np.zeros(len(grid.points), dtype=np.int64)
    labels[-3:] = 1                      # three bays of the building next door
    _, _, report = rp.repair(crop, grid, groups=labels)
    assert "1" in report["groups_skipped"]
    assert "3 bays" in report["groups_skipped"]["1"]


def test_the_worst_fraction_policy_replaces_that_fraction():
    """A clearer statement of intent than a sigma: fix the worst 10%."""
    crop = facade_of(wall(bays=10, storeys=6))
    grid = grid_of(crop)
    total = len(grid.points)
    _, _, report = rp.repair(crop, grid, worst=0.1)
    assert report["cells_replaced"] == pytest.approx(round(0.1 * total), abs=1)
    assert report["policy"] == "worst 10%"

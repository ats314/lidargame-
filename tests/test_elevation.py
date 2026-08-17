"""Building a facade from measured numbers rather than repairing a scan.

The reality mesh's own surface is not shippable and every attempt to make it so is
recorded in `features/repair.py`. This is the way through: the scan supplies
numbers, the footprint supplies a straight line, and the geometry is constructed.
A plane cannot droop, so the tests here are about the geometry being correct and
about the one number that is assumed staying labelled as assumed.
"""
from __future__ import annotations

import numpy as np
import pytest

from lidarworld.reconstruct import elevation as el


def dna(**kwargs):
    base = dict(bay_m=3.77, storey_m=3.31, storeys=7, window_w_m=1.6,
                window_h_m=1.9, sill_m=-1.4, base_z=2.7, top_z=29.0)
    base.update(kwargs)
    return el.FacadeDNA(**base)


def average_bay(px_per_m=48.0, bay_m=3.77, storey_m=3.31,
                window=(1.6, 1.9), wall=0.62, glass=0.14):
    """A synthetic average bay: one window in a field of masonry."""
    h, w = int(storey_m * px_per_m), int(bay_m * px_per_m)
    cell = np.full((h, w, 3), wall)
    wh, ww = int(window[1] * px_per_m), int(window[0] * px_per_m)
    r0, c0 = (h - wh) // 2, (w - ww) // 2
    cell[r0:r0 + wh, c0:c0 + ww] = glass
    return cell


# --- measurement --------------------------------------------------------------

def test_the_window_extent_comes_back_in_metres():
    """A blurred edge still crosses the midpoint where the sharp edge was."""
    cell = average_bay()
    width, height, _, report = el.window_box(cell, 48.0)
    assert width == pytest.approx(1.6, abs=0.1)
    assert height == pytest.approx(1.9, abs=0.1)
    assert report["wall_level"] > report["window_level"]


def test_a_blurred_window_keeps_its_extent():
    """The whole reason extent is usable when detail is not."""
    from lidarworld.features.frequency import box_blur
    cell = average_bay()
    sharp = el.window_box(cell, 48.0)[:2]
    soft = el.window_box(box_blur(cell, 8, wrap=False), 48.0)[:2]
    assert soft[0] == pytest.approx(sharp[0], abs=0.25)
    assert soft[1] == pytest.approx(sharp[1], abs=0.25)


def test_a_flat_bay_reports_no_window_rather_than_a_guess():
    flat = np.full((160, 180, 3), 0.5)
    width, height, _, report = el.window_box(flat, 48.0)
    assert (width, height) == (0.0, 0.0)
    assert "contrast" in report["reason"]


def test_the_reveal_depth_is_labelled_assumed():
    """Airborne data does not contain it, and the record has to say so."""
    from lidarworld.features.openings import Lattice
    grid = Lattice(bay_m=3.77, storey_m=3.31, bay_strength=0.44,
                   storey_strength=0.40)
    measured = el.measure(_facade(average_bay()), grid, average_bay(),
                          base_z=2.7, top_z=29.0)
    assert "ASSUMED" in measured.provenance["reveal_m"]
    assert measured.provenance["base_z"] == "measured"
    assert "measured" in measured.provenance["bay_m"]


def _facade(cell):
    from lidarworld.features.facade import Facade
    return Facade(surface_id="t", building_id=None, image=(cell * 255).astype(np.uint8),
                  px_per_m=48.0, width_m=cell.shape[1] / 48.0,
                  height_m=cell.shape[0] / 48.0, origin_xyz=np.zeros(3),
                  u_axis=np.array([1.0, 0.0, 0.0]), v_axis=np.array([0.0, 0.0, 1.0]),
                  normal=np.array([0.0, -1.0, 0.0]), resolution_px_per_m=13.2,
                  covered=1.0)


# --- geometry -----------------------------------------------------------------

def test_openings_are_punched_not_painted():
    """A hole is a hole: no wall quad may overlap an opening."""
    wall, openings = el.punch(12.0, 10.0, [(1.0, 2.6), (5.0, 6.6)],
                              [(3.0, 4.9), (6.3, 8.2)])
    assert len(openings) == 4
    for u0, u1, v0, v1 in wall:
        for a0, a1, b0, b1 in openings:
            # Clamp each axis at zero first: two disjoint intervals give a
            # negative span on both axes, and the product of two negatives is a
            # confident report of an overlap that is not there.
            du = max(0.0, min(u1, a1) - max(u0, a0))
            dv = max(0.0, min(v1, b1) - max(v0, b0))
            assert du * dv <= 1e-9, f"wall overlaps an opening by {du * dv}"


def test_the_wall_is_covered_exactly_once():
    """Quads must tile the rectangle: no gaps, no double-covered strips."""
    width, height = 12.0, 10.0
    wall, openings = el.punch(width, height, [(1.0, 2.6)], [(3.0, 4.9)])
    area = sum((u1 - u0) * (v1 - v0) for u0, u1, v0, v1 in wall + openings)
    assert area == pytest.approx(width * height, rel=1e-9)


def test_a_wall_gets_glass_and_a_reveal_for_every_opening():
    """Four reveal faces per opening is what makes it a hole with thickness."""
    built = el.build_wall(np.array([0.0, 0.0, 0.0]), np.array([20.0, 0.0, 0.0]),
                          dna())
    counts = {k: built.kinds.count(k) for k in ("wall", "glass", "reveal")}
    assert counts["glass"] == built.report["openings"]
    assert counts["reveal"] == 4 * counts["glass"]
    assert counts["wall"] > 0


def test_the_glass_sits_behind_the_wall_plane():
    built = el.build_wall(np.array([0.0, 0.0, 0.0]), np.array([20.0, 0.0, 0.0]),
                          dna())
    glass = built.quads[[i for i, k in enumerate(built.kinds) if k == "glass"]]
    wall = built.quads[[i for i, k in enumerate(built.kinds) if k == "wall"]]
    # The wall runs along +x, so its outward normal is -y and the glass is at +y.
    assert glass[:, :, 1].mean() > wall[:, :, 1].mean()
    assert abs(glass[:, :, 1].mean() - wall[:, :, 1].mean()) == pytest.approx(
        el.REVEAL_M, abs=1e-6)


def test_openings_land_on_the_measured_period():
    built = el.build_wall(np.array([0.0, 0.0, 0.0]), np.array([40.0, 0.0, 0.0]),
                          dna(bay_m=4.0, window_w_m=1.6))
    glass = built.quads[[i for i, k in enumerate(built.kinds) if k == "glass"]]
    centres = np.unique(np.round(glass[:, :, 0].mean(axis=1), 2))
    gaps = np.diff(np.sort(centres))
    assert np.allclose(gaps, 4.0, atol=0.05), centres


def test_a_short_edge_is_refused_rather_than_filled():
    built = el.build_wall(np.array([0.0, 0.0, 0.0]), np.array([0.2, 0.0, 0.0]),
                          dna())
    assert not len(built.quads)
    assert "too short" in built.report["reason"]


def test_a_whole_footprint_builds_every_long_edge():
    ring = np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0],
                     [20.0, 14.0, 0.0], [0.0, 14.0, 0.0]])
    built = el.build(ring, dna())
    assert built.report["edges"] == 4
    assert built.report["openings"] > 0
    assert len(built.quads) == len(built.kinds)


def test_nothing_is_built_above_the_roof():
    built = el.build_wall(np.array([0.0, 0.0, 0.0]), np.array([20.0, 0.0, 0.0]),
                          dna(base_z=0.0, top_z=12.0))
    assert built.quads[:, :, 2].max() <= 12.0 + 1e-9
    glass = built.quads[[i for i, k in enumerate(built.kinds) if k == "glass"]]
    assert glass[:, :, 2].max() < 12.0

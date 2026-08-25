"""A footprint vertex is not automatically a corner.

Every extruded wall used to flag both of its ends CORNER_CONVEX, so the
victorian pack -- which puts stone quoins on convex corners, exactly as
designed -- painted a stone stripe at every vertex of every footprint. A
register's footprint carries vertices that are not corners: collinear points
from digitising, and a degree or two of survey noise along a straight
frontage. A LoDo block came out striped every few metres.

This was invisible in every metric and obvious in the first render.
"""
from __future__ import annotations

import numpy as np

from lidarworld.reconstruct.extrude import CORNER_TURN_DEG, walls_from_footprint
from lidarworld.reconstruct.lattice import build_solid
from lidarworld.roles.taxonomy import Ctx


def square(size: float = 20.0) -> np.ndarray:
    """Counter-clockwise, closed. Four real corners."""
    return np.array([[0, 0], [size, 0], [size, size], [0, size], [0, 0]], dtype=float)


def split_frontage(size: float = 20.0) -> np.ndarray:
    """The same square with the south edge split by a collinear vertex."""
    return np.array([[0, 0], [size / 2, 0], [size, 0], [size, size], [0, size], [0, 0]],
                    dtype=float)


def corner_flags(patch):
    lattice = build_solid(patch, patch.extent[0], patch.extent[1], cell=0.5)
    context = lattice.context
    return (bool(context[0, 0] & Ctx.CORNER_CONVEX),
            bool(context[-1, 0] & Ctx.CORNER_CONVEX))


def test_a_real_square_still_has_four_corners():
    walls = walls_from_footprint(square(), 0.0, 12.0)
    assert len(walls) == 4
    for wall in walls:
        assert corner_flags(wall) == (True, True), "a right angle is a corner"


def test_a_collinear_vertex_is_not_a_corner():
    """The bug, stated as a test: the split frontage must not grow a quoin."""
    walls = walls_from_footprint(split_frontage(), 0.0, 12.0)
    assert len(walls) == 5

    flagged = sum(sum(corner_flags(w)) for w in walls)
    # Four genuine corners, each shared by two walls, so eight flagged ends --
    # and not ten, which is what flagging every end gave.
    assert flagged == 8, f"{flagged} corner ends for a shape with four corners"


def test_a_reflex_corner_is_concave_not_convex():
    """An L-plan's inside corner is not a quoin.

    Counter-clockwise L. The notch vertex turns the other way, and painting
    outside dressings on the inside of a corner reads as a mistake.
    """
    ring = np.array([[0, 0], [20, 0], [20, 20], [12, 20], [12, 8], [0, 8], [0, 0]],
                    dtype=float)
    walls = walls_from_footprint(ring, 0.0, 12.0)

    kinds = [w.attrs["corner_u_min"] for w in walls] + [w.attrs["corner_u_max"] for w in walls]
    assert "concave" in kinds, "the notch has to register as a reflex corner"

    concave = [w for w in walls
               if "concave" in (w.attrs["corner_u_min"], w.attrs["corner_u_max"])]
    lattice = build_solid(concave[0], *concave[0].extent, cell=0.5)
    end = 0 if concave[0].attrs["corner_u_min"] == "concave" else -1
    assert lattice.context[end, 0] & Ctx.CORNER_CONCAVE
    assert not lattice.context[end, 0] & Ctx.CORNER_CONVEX


def test_the_threshold_is_the_thing_being_asserted():
    """A shallow bend is a frontage; a sharp one is a corner.

    Proportional either side of the constant rather than pinned to a number,
    so tightening or loosening the threshold keeps the test meaningful.
    """
    def bend(degrees: float) -> str:
        angle = np.radians(degrees)
        ring = np.array([[0, 0], [20, 0],
                         [20 + 15 * np.cos(angle), 15 * np.sin(angle)],
                         [10, 40], [0, 0]], dtype=float)
        walls = walls_from_footprint(ring, 0.0, 12.0)
        return walls[0].attrs["corner_u_max"] if walls else "flat"

    assert bend(CORNER_TURN_DEG * 0.4) == "flat"
    assert bend(CORNER_TURN_DEG * 2.5) == "convex"


def test_a_measured_surface_is_unaffected():
    """Only extruded walls carry the hint; anything else keeps both corners."""
    walls = walls_from_footprint(square(), 0.0, 12.0)
    patch = walls[0]
    patch.attrs.pop("corner_u_min")
    patch.attrs.pop("corner_u_max")
    assert corner_flags(patch) == (True, True)

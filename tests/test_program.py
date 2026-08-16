"""World programs: keep the parameters, and measure what they fail to explain.

The value of a program is that it can be re-executed and that its cost can be
counted. A program with a small cost and an unmeasured residual is a guess, so
these tests care about both halves.
"""
from __future__ import annotations

import numpy as np
import pytest

from lidarworld.ir import program as pr


def square(x0=0.0, y0=0.0, side=10.0):
    return np.array([[x0, y0], [x0 + side, y0], [x0 + side, y0 + side],
                     [x0, y0 + side], [x0, y0]], dtype=float)


def test_cost_counts_free_parameters_not_bytes():
    p = pr.extrusion("b1", square(), 1580.0, 1592.0)
    # A 5-vertex closed ring is 10 numbers, plus ground and eave.
    assert p.cost == 12
    # Strings and flags describe, they do not parameterise.
    labelled = pr.extrusion("b2", square(), 1580.0, 1592.0, roof="flat", source="gis")
    assert labelled.cost == p.cost


def test_a_program_is_smaller_than_the_geometry_it_makes():
    """The whole argument for keeping programs, stated as an assertion."""
    p = pr.extrusion("b1", square(side=40.0), 0.0, 20.0)
    walls = pr.execute(p)
    assert len(walls) == 4
    corners = 4 * 4                       # every wall quad, before any tiling
    assert p.cost < corners * 3


def test_execute_regenerates_the_same_walls():
    """Completion by execution: the geometry follows from the parameters."""
    p = pr.extrusion("b1", square(side=12.0), 100.0, 110.0)
    first, second = pr.execute(p), pr.execute(p)
    assert len(first) == len(second) == 4
    for a, b in zip(first, second):
        assert np.allclose(a.centroid, b.centroid)
        assert np.allclose(a.normal, b.normal)
    # And it lands where the parameters said.
    heights = {round(w.extent[1], 3) for w in first}
    assert heights == {10.0}
    assert all(abs(w.centroid[2] - 105.0) < 1e-6 for w in first)


def test_an_unknown_generator_refuses_rather_than_guessing():
    p = pr.Program(id="x", kind="gable", params={"pitch": 0.5})
    with pytest.raises(NotImplementedError, match="no generator for 'gable'"):
        pr.execute(p)
    # It still costs what it costs, so it can be compared.
    assert p.cost == 1


def test_residual_is_optional_and_reported_as_such():
    p = pr.extrusion("b1", square(), 0.0, 10.0)
    assert p.residual is None
    assert "residual" not in p.to_json()
    p.residual = 0.25
    assert p.to_json()["residual"] == 0.25


def test_summarise_separates_measured_from_merely_stored():
    programs = [pr.extrusion(f"b{i}", square(), 0.0, 10.0) for i in range(4)]
    programs[0].residual = 0.1
    programs[1].residual = 0.3
    s = pr.summarise(programs)
    assert s["programs"] == 4
    assert s["measured"] == 2, "unmeasured programs must not be counted as scored"
    assert s["residual_mean"] == pytest.approx(0.2)
    assert s["residual_max"] == pytest.approx(0.3)
    assert s["parameters"] == sum(p.cost for p in programs)
    assert pr.summarise([]) == {"programs": 0, "parameters": 0}


def test_json_round_trips_the_ring():
    p = pr.extrusion("b1", square(side=7.0), 1.5, 9.25)
    j = p.to_json()
    assert j["kind"] == "extrude"
    assert j["params"]["ground_z"] == 1.5
    assert j["params"]["eave_z"] == 9.25
    ring = np.asarray(j["params"]["footprint"])
    assert ring.shape == (5, 2)
    assert np.allclose(ring[0], ring[-1]), "footprint must stay closed"


def test_the_compiler_emits_a_program_per_extruded_building(tiny_scene, tmp_path):
    """End to end: extrusion records what it executed, not only the result."""
    from lidarworld.reconstruct import extrude

    class Raster:
        def sample_bilinear(self, dtm, xy):
            return np.zeros(len(xy))

    class Patch:
        role = "surface.roof.flat"
        point_idx = np.arange(4)

    class Cloud:
        xyz = np.array([[0, 0, 12.0], [1, 0, 12.0], [0, 1, 12.0], [1, 1, 12.0]])

    walls, programs = extrude.build([square(side=20.0)], np.array([0]), [Patch()],
                                    Cloud(), Raster(), None)
    assert walls and len(programs) == 1
    program = programs[0]
    assert program.kind == "extrude"
    assert program.params["eave_z"] == pytest.approx(12.0)
    # Every wall points back at the program that produced it.
    assert {w.attrs["program"] for w in walls} == {program.id}
    assert len(pr.execute(program)) == len(walls)

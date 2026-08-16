"""World Seed extraction.

The seed is lossy on purpose, so these tests care about two things: that what
survives is enough to regenerate the place, and that what is thrown away is
thrown away honestly rather than silently.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from lidarworld.ir import seed as seed_ir


def test_simplify_keeps_the_shape_and_drops_the_noise():
    # A straight run of collinear points is one edge, however it was surveyed.
    line = np.array([[0.0, 0.0], [1.0, 0.001], [2.0, 0.0], [3.0, 0.002], [4.0, 0.0]])
    assert len(seed_ir._simplify(line, 0.5)) == 2

    # A real corner survives.
    corner = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0]])
    assert len(seed_ir._simplify(corner, 0.5)) == 3


def test_downsample_keeps_the_shape_of_the_ground():
    ramp = np.tile(np.linspace(0.0, 40.0, 40)[:, None], (1, 40))
    coarse = seed_ir._downsample(ramp, 4)
    assert coarse.shape == (10, 10)
    assert coarse[0, 0] < coarse[-1, 0]
    assert coarse.max() == pytest.approx(ramp.max(), rel=0.1)
    assert seed_ir._downsample(ramp, 1).shape == ramp.shape


def test_a_seed_round_trips_through_disk(tmp_path):
    seed = seed_ir.WorldSeed(
        name="block", crs="EPSG:26913", origin=[500000.0, 4400000.0, 0.0],
        buildings=[{"id": "b1", "footprint": [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]],
                    "ground_z": 1580.0, "height": 9.0, "roof": "flat", "residual": 0.02}],
        roads=[{"line": [[0, 5], [40, 5]], "half_width": 5.5}],
        vegetation=[{"xy": [20.0, 20.0], "base_z": 1580.0, "crown_r": 3.0, "height": 11.0}],
        terrain={"shape": [2, 2], "step_m": 4, "z": [[1580.0, 1580.2], [1580.1, 1580.3]]},
    )
    info = seed_ir.write(seed, tmp_path / "s.json")
    assert info["buildings"] == 1 and info["roads"] == 1 and info["trees"] == 1
    assert info["bytes"] > 0

    back = seed_ir.read(tmp_path / "s.json")
    assert back.name == "block"
    assert back.crs == "EPSG:26913"
    assert back.buildings == seed.buildings
    assert back.roads == seed.roads
    assert back.vegetation == seed.vegetation


def test_the_seed_carries_no_material_theme_or_engine(tmp_path):
    """Same invariant as the rest of the IR: materialisation is the backend's."""
    seed = seed_ir.WorldSeed(name="b", buildings=[
        {"id": "b1", "footprint": [[0, 0], [1, 0], [1, 1], [0, 0]],
         "ground_z": 0.0, "height": 4.0, "roof": "flat", "residual": None}])
    text = json.dumps(seed.to_json()).lower()
    for banned in ("material", "texture", "shader", "brick", "albedo", "theme"):
        assert banned not in text, f"the seed leaked {banned!r}"


def test_compression_is_reported_only_when_it_was_measured(tmp_path):
    seed = seed_ir.WorldSeed(name="b")
    info = seed_ir.write(seed, tmp_path / "a.json")
    assert "ratio" not in info, "no ratio without a source to compare against"

    source = tmp_path / "bundle.bin"
    source.write_bytes(b"x" * 500_000)
    seed.provenance["compressed_from_bytes"] = seed_ir._bundle_bytes(source)
    info = seed_ir.write(seed, tmp_path / "b.json")
    assert info["ratio"] > 1


def test_provenance_says_what_is_not_recoverable():
    seed = seed_ir.WorldSeed(name="b")
    seed.provenance["note"] = seed_ir.extract.__doc__ and ""
    got = seed_ir.WorldSeed(name="b")
    # extract() always writes the caveat; assert on the real one.
    class FakeWorld:
        name, crs, origin, bounds = "b", "", np.zeros(3), np.zeros((2, 3))
        arrays, nodes, sources, programs = {}, {}, [], []
        notes: dict = {}
    real = seed_ir.extract(FakeWorld())
    assert "not recoverable" in real.provenance["note"]
    assert real.buildings == [] and real.vegetation == []


def test_extract_pulls_buildings_from_programs_not_the_mesh():
    from lidarworld.ir import program as pr

    ring = np.array([[0.0, 0.0], [12.0, 0.0], [12.0, 8.0], [0.0, 8.0], [0.0, 0.0]])
    program = pr.extrusion("bldg.0001", ring, 1580.0, 1591.0)
    program.residual = 0.07

    class FakeWorld:
        name, crs = "block", "EPSG:26913"
        origin = np.array([500000.0, 4400000.0, 0.0])
        bounds = np.zeros((2, 3))
        arrays: dict = {}
        nodes: dict = {}
        sources: list = []
        notes: dict = {}
        programs = [program]

    seed = seed_ir.extract(FakeWorld())
    assert len(seed.buildings) == 1
    building = seed.buildings[0]
    assert building["id"] == "bldg.0001"
    assert building["height"] == pytest.approx(11.0)
    assert building["ground_z"] == pytest.approx(1580.0)
    assert building["residual"] == 0.07
    # A rectangle stays a rectangle: five vertices, closed.
    assert len(building["footprint"]) == 5
    assert building["footprint"][0] == building["footprint"][-1]

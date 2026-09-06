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


def _seed_from_world_with_a_tree(origin, tree_xy):
    """A minimal World carrying one instanced tree, at an absolute position.

    The pipeline shifts the cloud to a local origin and then adds that origin
    back when it writes instance nodes, so a tree node's frame is in absolute
    CRS metres while `world.bounds` is local. That asymmetry is the trap.
    """
    from lidarworld.types import Geometry, Node, World

    world = World(name="frame", crs="EPSG:28992")
    world.origin = np.asarray(origin, dtype=float)
    world.bounds = np.array([[0.0, 0.0, 0.0], [400.0, 400.0, 40.0]])
    world.add(Node(
        id="tree.0000", role="volume.vegetation.high", semantic="vegetation_high",
        kind="vegetation", confidence=0.5, stage="segment",
        geometry=Geometry("instance", {}, {"position": list(tree_xy) + [1.0],
                                           "size": [3.0, 3.0, 9.0]}),
        attrs={"crown_radius": 3.0, "canopy_height": 9.0}))
    return seed_ir.extract(world)


def test_every_section_of_a_seed_is_in_one_frame():
    """The bug this asserts against put a 400 m block's trees 121 km away.

    Buildings, roads, water and `bounds` are local to `origin`; vegetation was
    written straight from the instance node, which is absolute. Nothing raised
    -- a seed does not declare a frame per section, so no consumer could tell,
    and the trees simply vanished off the far edge of every tile baked from it.
    """
    origin = [121300.0, 486500.0, 0.0]
    seed = _seed_from_world_with_a_tree(origin, [121350.0, 486600.0])

    assert seed.origin == origin
    lo, hi = np.asarray(seed.bounds[0][:2]), np.asarray(seed.bounds[1][:2])
    xy = np.asarray(seed.vegetation[0]["xy"], dtype=float)

    # Local to `origin`, like everything else in the file.
    assert xy.tolist() == [50.0, 100.0]
    assert (xy >= lo).all() and (xy <= hi).all(), (
        f"tree at {xy.tolist()} is outside bounds {lo.tolist()}..{hi.tolist()}; "
        "it was written in a different frame from the rest of the seed")


def test_the_expander_places_trees_where_the_seed_put_them():
    """`expand()` read `position`/`crown_radius`; the seed writes `xy`/`crown_r`.

    Every tree therefore fell back to the default and stacked on the origin at
    2.5 m. `.get` with a default cannot distinguish a missing key from an absent
    tree, so a whole canopy collapsed to a point without a single error.
    """
    from lidarworld.world import generate

    seed = {
        "seed": "lidarworld/0.1", "name": "trees", "crs": "EPSG:28992",
        "origin": [0.0, 0.0, 0.0], "bounds": [[0, 0, 0], [40, 40, 20]],
        "terrain": {"shape": [10, 10], "step_m": 4.0,
                    "z": [[0.0] * 10 for _ in range(10)]},
        "buildings": [], "roads": [], "water": [],
        "vegetation": [
            {"xy": [8.0, 30.0], "base_z": 0.0, "crown_r": 3.5, "height": 11.0},
            {"xy": [31.0, 12.0], "base_z": 0.0, "crown_r": 2.0, "height": 7.0},
        ],
    }
    world = generate.expand(seed)
    trees = [n for n in world.nodes.values()
             if n.role == "volume.vegetation.high"]
    assert len(trees) == 2

    placed = sorted([n.attrs["center"][:2] for n in trees])
    assert placed == [[8.0, 30.0], [31.0, 12.0]]
    assert sorted(n.attrs["size"][0] for n in trees) == [2.0, 3.5]

    # The failure mode was every tree on top of the origin, so assert they are
    # apart rather than only that they moved.
    first, second = (np.asarray(n.attrs["center"][:2]) for n in trees)
    assert np.linalg.norm(first - second) > 10.0

"""End-to-end: a known scene must compile into the structure it obviously has,
survive a round trip through the IR, and export through every backend.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from lidarworld.backends import cityjson as cityjson_backend
from lidarworld.backends import gltf as gltf_backend
from lidarworld.backends import web as web_backend
from lidarworld.ir import inspect, read_world, write_world
from lidarworld.roles.taxonomy import Ctx
from lidarworld.themes import load_pack


def test_compiles_the_structure_that_is_there(compiled_world):
    world = compiled_world
    summary = world.summary()

    assert summary["nodes"] > 5
    assert world.by_role("volume.building"), "the two walls and roof must form a building"
    assert world.by_role("surface.wall"), "vertical patches must be recognised as walls"
    assert "terrain" in world.nodes

    # Opening detection itself is covered directly in test_lattice.py; here we
    # only require that anything it does find is dimensionally sane, because a
    # two-wall toy scene is too small to guarantee a patch survives segmentation.
    for opening in world.by_role("opening"):
        assert 0.3 < opening.attrs.get("width", 0) < 6.0
        assert 0.3 < opening.attrs.get("height", 0) < 6.0


def test_every_stage_is_recorded(compiled_world):
    names = [stage.name for stage in compiled_world.stages]
    for expected in ("ingest", "datum", "terrain", "features", "semantics", "roles",
                     "segment.planes", "lattice", "topology", "reconstruct"):
        assert expected in names
    assert all(stage.seconds >= 0 for stage in compiled_world.stages)


def test_provenance_reaches_the_nodes(compiled_world):
    assert compiled_world.sources
    source = compiled_world.sources[0]
    assert source.adapter == "las"
    assert source.point_count > 0
    surfaces = compiled_world.by_role("surface")
    assert surfaces and all(s.sources for s in surfaces)
    assert all(0.0 < s.confidence <= 1.0 for s in surfaces)


def test_mesh_carries_context_and_is_well_formed(compiled_world):
    arrays = compiled_world.arrays
    positions = arrays["mesh/positions"]
    indices = arrays["mesh/indices"]
    ctx = arrays["mesh/ctx"]

    assert len(positions) > 0
    assert indices.max() < len(positions), "index out of range"
    assert len(ctx) == len(positions)
    assert np.all(ctx & Ctx.OCCUPIED), "every emitted vertex must be a solid tile"
    # Some cells must be interior and some on a corner, or the mask is useless.
    assert (ctx & Ctx.INTERIOR).any()
    assert (ctx & Ctx.CORNER_CONVEX).any()


def test_ir_round_trip(compiled_world, tmp_path):
    path = write_world(compiled_world, tmp_path / "w.lwir")
    assert inspect(path)["magic"] == "lidarworld/spatial-ir"

    reloaded = read_world(path)
    assert set(reloaded.nodes) == set(compiled_world.nodes)
    assert len(reloaded.edges) == len(compiled_world.edges)
    assert np.allclose(reloaded.origin, compiled_world.origin)
    assert np.allclose(reloaded.arrays["mesh/positions"], compiled_world.arrays["mesh/positions"])

    node_id = next(iter(compiled_world.nodes))
    assert reloaded.nodes[node_id].role == compiled_world.nodes[node_id].role
    assert reloaded.points is not None


def test_lazy_arrays_do_not_load_everything(compiled_world, tmp_path):
    path = write_world(compiled_world, tmp_path / "w.lwir")
    reloaded = read_world(path, load_points=False)
    assert reloaded.points is None
    assert "mesh/positions" in reloaded.arrays
    assert reloaded.arrays["mesh/positions"].shape[1] == 3


def test_web_backend_bundle(compiled_world, tmp_path):
    info = web_backend.export(compiled_world, tmp_path, themes=["survey"])
    header = json.loads((tmp_path / "world.json").read_text())
    blob = (tmp_path / "world.bin").read_bytes()

    assert header["format"] == "lidarworld/web"
    assert header["mesh"]["vertexCount"] == info["vertices"]
    assert len(blob) == info["bytes"]
    # The vertex layout must describe exactly the stride that was written.
    assert sum(f["components"] * 4 for f in header["vertexLayout"]) == header["vertexStride"]
    assert header["contextFlags"]["corner_convex"] == Ctx.CORNER_CONVEX
    assert header["mesh"]["indexOffset"] >= info["vertices"] * header["vertexStride"]


def test_slot_names_line_up_with_the_mesh(compiled_world):
    names = web_backend.mesh_slot_names(compiled_world)
    used = np.unique(compiled_world.arrays["mesh/node"])
    assert used.max() < len(names)
    assert names[0] == "terrain"
    for slot in used:
        assert names[slot] == "terrain" or names[slot] in compiled_world.nodes


def test_gltf_export_is_materialised(compiled_world, tmp_path):
    pack = load_pack("victorian")
    info = gltf_backend.export(compiled_world, pack, tmp_path, name="t")
    document = json.loads((tmp_path / "t.gltf").read_text())

    assert document["asset"]["version"] == "2.0"
    assert len(document["materials"]) == info["materials"] >= 2
    assert document["meshes"][0]["primitives"]
    for material in document["materials"]:
        assert material["extras"]["license"]
    total = sum(document["accessors"][p["indices"]]["count"] for p in document["meshes"][0]["primitives"])
    assert total == info["triangles"] * 3, "every triangle must land in exactly one primitive"


def test_gltf_splits_by_resolved_material(compiled_world):
    """Different themes must partition the same mesh differently."""
    a = gltf_backend.resolve_vertex_materials(compiled_world, load_pack("victorian"))[0]
    b = gltf_backend.resolve_vertex_materials(compiled_world, load_pack("neon"))[0]
    assert len(a) == len(b) == len(compiled_world.arrays["mesh/positions"])
    assert len(set(a.tolist())) > 1


def test_cityjson_export_conforms(compiled_world, tmp_path):
    info = cityjson_backend.export(compiled_world, tmp_path / "w.city.json")
    document = json.loads((tmp_path / "w.city.json").read_text())

    assert document["type"] == "CityJSON"
    assert document["version"] == "1.1"
    assert len(document["vertices"]) == info["vertices"]
    assert document["CityObjects"]

    known = {"Building", "BuildingPart", "TINRelief", "SolitaryVegetationObject",
             "CityFurniture", "GenericCityObject", "WaterBody", "Road", "Window", "Door"}
    for key, obj in document["CityObjects"].items():
        assert obj["type"] in known, f"{key} has non-CityGML type {obj['type']}"
        for geometry in obj.get("geometry", []):
            assert geometry["type"] == "MultiSurface"
            assert len(geometry["semantics"]["values"]) == len(geometry["boundaries"])
            for boundary in geometry["boundaries"]:
                for ring in boundary:
                    assert max(ring) < len(document["vertices"])


def test_cityjson_keeps_context_that_citygml_cannot_express(compiled_world, tmp_path):
    cityjson_backend.export(compiled_world, tmp_path / "w.city.json")
    document = json.loads((tmp_path / "w.city.json").read_text())
    with_context = [o for o in document["CityObjects"].values()
                    if o["attributes"].get("+lidarworld_context")]
    assert with_context, "context summaries must survive the CityGML mapping"

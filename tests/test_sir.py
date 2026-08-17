"""The compiler must emit documents that satisfy the vendored SIR v0.1 spec.

This is the contract between the two halves of the project: `spec/` is the
normative schema and round-trip benchmark, and `lidarworld` is the thing that
produces worlds for it to score. If this test fails, they have diverged.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from lidarworld.ir.sir import build_document, write_document

SPEC = Path(__file__).resolve().parent.parent / "spec"
pytestmark = pytest.mark.skipif(not SPEC.exists(), reason="spec/ not vendored")


@pytest.fixture(scope="module")
def sir_document(compiled_world):
    return build_document(compiled_world)


def test_validates_against_the_normative_schema(sir_document):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((SPEC / "schema" / "spatial_ir_v0_1.schema.json").read_text())
    jsonschema.Draft202012Validator(schema).validate(sir_document)


def test_satisfies_the_reference_invariants(sir_document):
    pytest.importorskip("jsonschema")
    sys.path.insert(0, str(SPEC))
    from sir.reference import validate_invariants

    validate_invariants(sir_document)


def test_entities_carry_citygml_classes(sir_document):
    classes = {e["class"] for e in sir_document["entities"]}
    assert "Building" in classes
    assert classes & {"WallSurface", "RoofSurface"}
    assert all(e["kind"] in {"region", "space", "boundary", "opening", "object", "terrain",
                             "transport", "vegetation", "water", "sensor", "logical"}
               for e in sir_document["entities"])


def test_epistemic_state_is_derived_not_asserted(sir_document):
    """Surfaces must be graded by how much of them was actually measured."""
    surfaces = [e for e in sir_document["entities"] if e["kind"] == "boundary"]
    assert surfaces
    states = {e["epistemic_state"] for e in surfaces}
    # v0.3 vocabulary. `hybrid` is gone: a surface mostly backed by returns is
    # `derived`, because the fit follows deterministically from them, which is
    # a sharper claim than "a composition of states".
    assert states <= {"observed", "derived", "inferred"}
    assert "hybrid" not in states
    # Geometry confidence is the measured fraction, so it must vary between
    # surfaces rather than being a constant stamped on everything.
    measured = {e["attributes"]["measured_fraction"] for e in surfaces}
    assert len(measured) > 1, "measured_fraction is not being computed per surface"
    for e in surfaces:
        fraction = e["attributes"]["measured_fraction"]
        if e["epistemic_state"] == "observed":
            assert fraction >= 0.85
        elif e["epistemic_state"] == "inferred":
            assert fraction < 0.35


def test_provenance_records_the_stage_that_made_each_entity(sir_document):
    modes = {p["mode"] for e in sir_document["entities"] for p in e["provenance"]}
    assert modes <= {"sensor_observation", "geometric_inference", "semantic_inference",
                     "procedural_generation", "manual_authoring", "structured_import",
                     "fusion", "other"}
    assert "geometric_inference" in modes
    for entity in sir_document["entities"]:
        assert entity["provenance"][0]["algorithm"].startswith("lidarworld.")


def test_relations_reference_real_entities(sir_document):
    ids = {e["id"] for e in sir_document["entities"]}
    assert sir_document["relations"]
    for relation in sir_document["relations"]:
        assert relation["source"] in ids
        assert relation["target"] in ids
    assert any(r["type"] == "part_of" for r in sir_document["relations"])


def test_observations_carry_licence_and_frame(sir_document):
    assert sir_document["observations"]
    for observation in sir_document["observations"]:
        assert observation["modality"] == "lidar"
        assert observation["frame"]
        assert observation["metadata"]["license"]


def test_write_document_round_trips(compiled_world, tmp_path):
    info = write_document(compiled_world, tmp_path / "w.sir.json")
    document = json.loads((tmp_path / "w.sir.json").read_text())
    assert document["sir_version"] == "0.1.0"
    assert len(document["entities"]) == info["entities"]
    assert document["world"]["up_axis"] == "Z"
    assert document["world"]["linear_unit"] == "m"


def test_derived_must_cite_what_it_was_derived_from(sir_document):
    """`derived` claims a deterministic consequence of evidence. Without a
    citation the state means whatever the producer wanted, which is what
    `hybrid` had quietly become."""
    for entity in sir_document["entities"]:
        if entity["epistemic_state"] != "derived":
            continue
        modes = {p["mode"] for p in entity["provenance"]}
        for geometry in entity["geometry"]:
            modes |= {p["mode"] for p in geometry.get("provenance", [])}
        assert modes & {"sensor_observation", "structured_import",
                        "geometric_inference"}, entity["id"]


def test_the_superseded_vocabulary_is_gone_everywhere(sir_document):
    """manual and imported are origin, not epistemic state, and belong in
    provenance mode -- where they already live as manual_authoring and
    structured_import. Carrying them in both let an entity be `imported`
    without ever saying whether the import was measured or invented."""
    retired = {"hybrid", "manual", "imported", "fusion"}
    states = {e["epistemic_state"] for e in sir_document["entities"]}
    assert not (states & retired), states & retired
    assert states <= {"observed", "derived", "inferred", "resolved",
                      "generated", "unknown"}

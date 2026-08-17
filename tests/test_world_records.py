"""The records that make a repaired world auditable.

An inferred wall and a measured one are the same triangles. The only thing
that can tell them apart afterwards is a record written when the decision was
made, so these guard the properties that make such a record worth keeping.
"""
import pytest

from lidarworld.world import (BoundarySeed, GapRecord, RepairLog, RepairRecord)


def test_a_repair_can_never_produce_an_observation():
    """The one state a repair cannot output. A repair is by definition
    something the sensor did not supply, and letting one claim otherwise is
    how generated geometry becomes validation truth."""
    with pytest.raises(ValueError, match="never observations"):
        RepairRecord(id="r1", pass_name="building_closure",
                     operation="extrude", target_entity_id="b1",
                     epistemic_output_state="observed")


def test_unknown_vocabulary_is_rejected_rather_than_stored():
    with pytest.raises(ValueError, match="unknown gap_type"):
        GapRecord(id="g1", gap_type="hole", bounds=(0, 0, 1, 1))
    with pytest.raises(ValueError, match="unknown pass"):
        RepairRecord(id="r1", pass_name="magic", operation="x",
                     target_entity_id="b1", epistemic_output_state="inferred")
    with pytest.raises(ValueError, match="unknown boundary_type"):
        BoundarySeed(id="b1", left_entity="a", right_entity="b",
                     boundary_type="glued")


def test_a_true_void_is_not_fillable():
    """Filling one manufactures surface where the world has none, and it is the
    commonest way a hole-free metric is met dishonestly."""
    assert not GapRecord(id="g", gap_type="true_void", bounds=(0, 0, 1, 1)).fillable
    assert not GapRecord(id="g", gap_type="procedural_freedom",
                         bounds=(0, 0, 1, 1)).fillable
    assert GapRecord(id="g", gap_type="occlusion", bounds=(0, 0, 1, 1)).fillable
    assert GapRecord(id="g", gap_type="missing_observation",
                     bounds=(0, 0, 1, 1)).fillable


def test_a_boundary_needs_two_different_sides():
    with pytest.raises(ValueError, match="same entity on both sides"):
        BoundarySeed(id="b", left_entity="road.1", right_entity="road.1",
                     boundary_type="curb_profile")


def test_the_parameter_hash_is_stable_across_key_order():
    """The record is worthless for reproducing a build if the hash moves when
    a dict happens to iterate differently."""
    a = RepairRecord(id="r", pass_name="road_continuity", operation="x",
                     target_entity_id="t", epistemic_output_state="derived",
                     parameters={"alpha": 1, "beta": [2, 3], "gamma": "z"})
    b = RepairRecord(id="r", pass_name="road_continuity", operation="x",
                     target_entity_id="t", epistemic_output_state="derived",
                     parameters={"gamma": "z", "beta": [2, 3], "alpha": 1})
    assert a.parameter_hash == b.parameter_hash
    c = RepairRecord(id="r", pass_name="road_continuity", operation="x",
                     target_entity_id="t", epistemic_output_state="derived",
                     parameters={"alpha": 2})
    assert c.parameter_hash != a.parameter_hash


def test_the_log_counts_what_it_refused_as_well_as_what_it_did():
    """A world that filled everything and a world that refused half of it are
    different worlds, and the summary has to distinguish them."""
    log = RepairLog()
    log.gap(gap_type="occlusion", bounds=(0, 0, 1, 1), status="filled")
    log.gap(gap_type="true_void", bounds=(2, 2, 3, 3), status="refused")
    log.gap(gap_type="conflict", bounds=(4, 4, 5, 5), status="refused")
    log.repair(pass_name="building_closure", operation="extrude",
               target_entity_id="b1", epistemic_output_state="inferred")
    summary = log.summary()
    assert summary["gaps"] == 3
    assert summary["refused"] == 2
    assert summary["repairs_by_state"] == {"inferred": 1}
    assert summary["gaps_by_type"] == {"conflict": 1, "occlusion": 1, "true_void": 1}


def test_records_serialise_with_the_spec_field_names():
    log = RepairLog()
    record = log.repair(pass_name="building_closure", tier=3,
                        operation="extrude_envelope_from_footprint",
                        target_entity_id="bldg.0001",
                        epistemic_output_state="inferred",
                        parameters={"base_z": 1.0})
    payload = record.to_dict()
    assert payload["pass"] == "building_closure"   # spec spells it `pass`
    assert "pass_name" not in payload
    assert payload["parameter_hash"]
    assert payload["tier"] == 3
    assert log.to_dict()["repairs"] == [payload]


def test_a_resolved_world_can_say_why_a_part_of_it_exists():
    """A procedural city can produce a building. Only a world that recorded
    its decisions can say which parts of that building are defensible."""
    log = RepairLog()
    log.repair(pass_name="building_closure", tier=3,
               operation="extrude_envelope_from_footprint",
               target_entity_id="bldg.0182", epistemic_output_state="inferred",
               evidence_ids=["footprint.0182", "patch.44"],
               confidence=0.75, max_displacement=12.9,
               reason="airborne returns do not describe these facades")
    proof = log.proof_for("bldg.0182")
    assert proof["method"] == "tier_3_geometric_constraint"
    assert proof["epistemic_state"] == "inferred"
    assert proof["confidence"] == 0.75
    assert proof["evidence"] == ["footprint.0182", "patch.44"]
    assert proof["parameter_hash"]
    assert log.proof_for("bldg.9999") is None, (
        "an entity nothing repaired has no proof, which is itself an answer")


def test_the_proof_reports_the_weakest_step_not_the_strongest():
    """A wall geometrically closed and then procedurally detailed is only as
    defensible as its weakest step. Reporting the best one would let a
    generated facade inherit the confidence of a measured roof."""
    log = RepairLog()
    log.repair(pass_name="building_closure", tier=3, operation="close",
               target_entity_id="b", epistemic_output_state="derived",
               confidence=0.94)
    log.repair(pass_name="building_closure", tier=7, operation="fenestration",
               target_entity_id="b", epistemic_output_state="generated",
               confidence=0.2)
    proof = log.proof_for("b")
    assert proof["epistemic_state"] == "generated"
    assert proof["method"] == "tier_7_procedural_generation"
    assert proof["operations"] == 2

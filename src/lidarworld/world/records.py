"""The three canonical records world building has to keep.

A repaired world is only worth having if you can ask it, afterwards, which
parts were measured and which were supplied by the compiler. Geometry does not
answer that -- an inferred wall and a measured one are the same triangles. So
the answer lives in records written at the moment of the decision, not
reconstructed later from the result.

    GapRecord      a missing region, classified before anything fills it
    RepairRecord   one completion operation, and what justified it
    BoundarySeed   a shared edge two domains must agree about

The rule these exist to enforce is that a generated object must never silently
become validation truth. A record carries the epistemic state its operation
produced, so a wall that was invented stays inferred through target
compilation, and a forward-validation score can exclude it or count it
knowingly rather than by accident.

There is deliberately no `fill_holes()` here. A gap is classified first, and
the classification decides which completion tier may touch it -- filling a
scan shadow and filling a true void are different claims about the world.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

#: Why a region is empty. The taxonomy is the point: `TRUE_VOID` must not be
#: filled at all, `MISSING_OBSERVATION` may be, and telling them apart is a
#: judgement about the world rather than about the raster.
GAP_TYPES = (
    "true_void",             # no surface should exist here
    "missing_observation",   # a surface likely exists but was not sampled
    "occlusion",             # hidden behind another object or a blind spot
    "low_density",           # sampled, too sparsely to materialise directly
    "boundary_break",        # a linear feature enters and leaves the region
    "structural_gap",        # a component a coherent object requires is absent
    "semantic_region_gap",   # a known region polygon lacks geometric support
    "conflict",              # sources disagree beyond tolerance
    "unknown",               # no supported resolution
    "procedural_freedom",    # constrained enough; the detail is intentionally free
)

#: The completion tiers, least speculative first. A gap is resolved by the
#: lowest-numbered tier whose support clears its threshold.
COMPLETION_TIERS = {
    1: ("same_surface_interpolation", "derived"),
    2: ("topological_continuity", "inferred"),
    3: ("geometric_constraint", "derived"),
    4: ("semantic_region_constraint", "derived"),
    5: ("contextual_inference", "inferred"),
    6: ("prototype_family", "inferred"),
    7: ("procedural_generation", "generated"),
}

REPAIR_PASSES = ("ground_continuity", "building_closure", "road_continuity",
                 "sidewalk_continuity", "vegetation_integrity",
                 "junction_reconciliation", "collision_compilation")

#: How two domains agree about the edge they share. `NO_WELD` is a real answer:
#: a kerb is meant to be discontinuous, and welding it flat is a defect.
BOUNDARY_TYPES = ("c0_shared_position", "c1_smooth", "step_discontinuity",
                  "curb_profile", "waterline", "building_ground_contact",
                  "no_weld")

EPISTEMIC_STATES = ("observed", "derived", "inferred", "resolved", "generated",
                    "unknown")


def _hash(params: dict) -> str:
    """A stable digest of the parameters an operation ran with.

    Sorted and separator-pinned so the same parameters hash the same across
    runs and machines -- the record is worthless for reproducing a build if the
    hash moves when a dict happens to iterate differently.
    """
    blob = json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _clean(record: dict) -> dict:
    return {k: v for k, v in record.items() if v not in (None, [], {}, "")}


@dataclass
class GapRecord:
    """A missing region, classified before anything is allowed to fill it."""
    id: str
    gap_type: str
    bounds: tuple[float, ...]            # (minx, miny, maxx, maxy[, minz, maxz])
    dimensionality: int = 2              # 1 linear, 2 surface, 3 volume
    parent_entity_id: str | None = None
    neighboring_entities: list[str] = field(default_factory=list)
    neighboring_roles: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    candidate_methods: list[str] = field(default_factory=list)
    selected_method: str | None = None
    max_extrapolation_distance: float | None = None
    status: str = "open"                 # open | filled | refused | unknown
    area: float | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.gap_type not in GAP_TYPES:
            raise ValueError(f"unknown gap_type {self.gap_type!r}; have {GAP_TYPES}")
        if self.status not in ("open", "filled", "refused", "unknown"):
            raise ValueError(f"unknown status {self.status!r}")

    @property
    def fillable(self) -> bool:
        """Whether any tier may touch it.

        A true void is not a defect to be repaired, it is the answer. Filling
        one manufactures surface where the world has none, and it is the
        commonest way a 'hole-free' metric is met dishonestly.
        """
        return self.gap_type not in ("true_void", "procedural_freedom")

    def to_dict(self) -> dict:
        return _clean(asdict(self))


@dataclass
class RepairRecord:
    """One completion operation, and what justified it.

    Every field here answers a question someone will ask of a finished world:
    what changed, why it was allowed to, how far it moved anything, and what
    kind of claim the result is.
    """
    id: str
    pass_name: str
    operation: str
    target_entity_id: str
    epistemic_output_state: str
    tier: int | None = None
    gap_id: str | None = None
    input_geometry_ids: list[str] = field(default_factory=list)
    output_geometry_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    reason: str = ""
    confidence: float = 0.0
    max_displacement: float | None = None
    algorithm: str = ""
    version: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    timestamp: str | None = None

    def __post_init__(self) -> None:
        if self.pass_name not in REPAIR_PASSES:
            raise ValueError(f"unknown pass {self.pass_name!r}; have {REPAIR_PASSES}")
        if self.epistemic_output_state not in EPISTEMIC_STATES:
            raise ValueError(f"unknown epistemic state "
                             f"{self.epistemic_output_state!r}")
        if self.epistemic_output_state == "observed":
            # The one state a repair can never produce. A repair is by
            # definition something the sensor did not supply.
            raise ValueError(f"repair {self.id} claims to have produced an "
                             f"observation; repairs are never observations")
        if self.tier is not None and self.tier not in COMPLETION_TIERS:
            raise ValueError(f"tier {self.tier} is not one of {sorted(COMPLETION_TIERS)}")

    @property
    def parameter_hash(self) -> str:
        return _hash(self.parameters)

    def to_dict(self) -> dict:
        record = asdict(self)
        record["pass"] = record.pop("pass_name")
        record["parameter_hash"] = self.parameter_hash
        return _clean(record)


@dataclass
class BoundarySeed:
    """An edge two domains share, owned once instead of generated twice.

    Independently meshing a road and its pavement produces two nearly
    coincident vertex rows, which is where cracks, z-fighting and doubled
    colliders come from. Both sides referencing one boundary makes the
    agreement structural rather than a tolerance.
    """
    id: str
    left_entity: str
    right_entity: str
    boundary_type: str
    geometry_id: str | None = None
    continuity_rule: str | None = None
    elevation_rule: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.boundary_type not in BOUNDARY_TYPES:
            raise ValueError(f"unknown boundary_type {self.boundary_type!r}; "
                             f"have {BOUNDARY_TYPES}")
        if self.left_entity == self.right_entity:
            raise ValueError(f"boundary {self.id} has the same entity on both sides")

    def to_dict(self) -> dict:
        return _clean(asdict(self))


class RepairLog:
    """Every gap and repair for one build, in the order they happened."""

    def __init__(self) -> None:
        self.gaps: list[GapRecord] = []
        self.repairs: list[RepairRecord] = []
        self.boundaries: list[BoundarySeed] = []

    def gap(self, **kwargs) -> GapRecord:
        record = GapRecord(id=kwargs.pop("id", f"gap.{len(self.gaps):05d}"), **kwargs)
        self.gaps.append(record)
        return record

    def repair(self, **kwargs) -> RepairRecord:
        record = RepairRecord(id=kwargs.pop("id", f"repair.{len(self.repairs):05d}"),
                              **kwargs)
        self.repairs.append(record)
        return record

    def boundary(self, **kwargs) -> BoundarySeed:
        record = BoundarySeed(id=kwargs.pop("id", f"bound.{len(self.boundaries):05d}"),
                              **kwargs)
        self.boundaries.append(record)
        return record

    def proof_for(self, entity_id: str) -> dict | None:
        """Why this entity exists, and what kind of claim it is.

        The question a resolved world has to be able to answer about any part
        of itself. It is a lookup rather than an analysis precisely because the
        answer was recorded when the decision was made -- reconstructing it
        afterwards from the geometry is the thing that cannot be done.

        Returns None when nothing repaired this entity, which is itself an
        answer: the entity is whatever the evidence made it, with no completion
        operation in its history.
        """
        records = [r for r in self.repairs if r.target_entity_id == entity_id]
        if not records:
            return None
        # Most speculative operation wins the summary: a wall that was
        # geometrically closed and then procedurally detailed is only as
        # defensible as its weakest step.
        rank = {"derived": 0, "resolved": 1, "inferred": 2, "generated": 3,
                "unknown": 4}
        worst = max(records, key=lambda r: (rank.get(r.epistemic_output_state, 9),
                                            r.tier or 0))
        return {
            "entity": entity_id,
            "method": (f"tier_{worst.tier}_{COMPLETION_TIERS[worst.tier][0]}"
                       if worst.tier else worst.operation),
            "operation": worst.operation,
            "epistemic_state": worst.epistemic_output_state,
            "confidence": worst.confidence,
            "reason": worst.reason,
            "evidence": sorted({e for r in records for e in r.evidence_ids}),
            "max_displacement_m": worst.max_displacement,
            "parameter_hash": worst.parameter_hash,
            "operations": len(records),
            "repair_ids": [r.id for r in records],
        }

    def summary(self) -> dict:
        by_state: dict[str, int] = {}
        by_pass: dict[str, int] = {}
        for record in self.repairs:
            by_state[record.epistemic_output_state] = \
                by_state.get(record.epistemic_output_state, 0) + 1
            by_pass[record.pass_name] = by_pass.get(record.pass_name, 0) + 1
        by_gap: dict[str, int] = {}
        for record in self.gaps:
            by_gap[record.gap_type] = by_gap.get(record.gap_type, 0) + 1
        return {"gaps": len(self.gaps), "repairs": len(self.repairs),
                "boundaries": len(self.boundaries),
                "gaps_by_type": dict(sorted(by_gap.items())),
                "repairs_by_pass": dict(sorted(by_pass.items())),
                "repairs_by_state": dict(sorted(by_state.items())),
                "refused": sum(1 for g in self.gaps if g.status == "refused")}

    def to_dict(self) -> dict:
        return {"summary": self.summary(),
                "gaps": [g.to_dict() for g in self.gaps],
                "repairs": [r.to_dict() for r in self.repairs],
                "boundaries": [b.to_dict() for b in self.boundaries]}

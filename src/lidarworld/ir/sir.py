"""Export a compiled :class:`World` as a Spatial IR v0.1 document.

`spec/` carries the normative SIR v0.1 schema and the synthetic round-trip
benchmark. This module is the bridge: it makes the compiler *emit that spec*,
so a world reconstructed from real LiDAR can be scored by exactly the same
metrics as a reconstruction of a known synthetic scene.

Two things the mapping has to get right, because they are the parts of SIR that
have no equivalent in a mesh format:

**Epistemic state.** SIR refuses to let inferred detail masquerade as measured
reality, and it is strict: reconstructed geometry is never `observed`, however
dense the evidence behind it -- only the point cloud itself was observed. What
the compiler can say honestly is how much of a surface it actually saw, since
it tracks this per tile (`SPARSE_EVIDENCE` for cells filled by morphological
closing, `OCCLUDED` for cells the sensor never saw). A facade whose cells are
mostly backed by returns is `derived` -- the fit and its clipping follow
deterministically from those returns -- and one mostly invented is `inferred`.
The quantity lives in `measured_fraction`.

**Provenance mode.** Which pass created a node is recorded during compilation,
so it maps directly onto SIR's provenance modes rather than being asserted.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..roles.taxonomy import Ctx, citygml_type
from ..types import World

SIR_VERSION = "0.1.0"

#: role prefix -> SIR entity kind
KIND = [
    ("volume.building", "object"),
    ("volume.vegetation", "vegetation"),
    ("surface", "boundary"),
    ("edge", "boundary"),
    ("corner", "boundary"),
    ("opening", "opening"),
    ("terrain.road", "transport"),
    ("terrain.water", "water"),
    ("terrain", "terrain"),
    ("linear", "object"),
    ("instance", "object"),
]

#: role prefix -> SIR semantic resolution
RESOLUTION = [
    ("terrain", "macro"),
    ("volume.building", "object"),
    ("volume.vegetation", "object"),
    ("instance", "object"),
    ("linear", "object"),
    ("surface", "part"),
    ("opening", "component"),
    ("edge", "detail"),
    ("corner", "detail"),
]

#: compiler stage -> SIR provenance mode
PROVENANCE_MODE = {
    "ingest": "sensor_observation",
    "datum": "sensor_observation",
    "terrain": "geometric_inference",
    "features": "geometric_inference",
    "semantics": "semantic_inference",
    "roles": "semantic_inference",
    "segment.planes": "geometric_inference",
    "segment.instances": "geometric_inference",
    "lattice": "geometric_inference",
    "topology": "geometric_inference",
    "reconstruct": "geometric_inference",
}

#: SIR uses part_of for containment; the rest of the compiler's relations are
#: already the vocabulary SIR expects.
RELATION_TYPE = {"contains": "part_of", "opening_in": "opening_in"}


def _first(table, role: str, default: str) -> str:
    for prefix, value in table:
        if role == prefix or role.startswith(prefix + "."):
            return value
    return default


def _epistemic_state(world: World, node) -> tuple[str, float]:
    """(state, measured_fraction) from the node's own tile evidence."""
    geometry = node.geometry
    if geometry is None or geometry.kind != "tiled_plane":
        # Instances and terrain are fitted to measured points but are not
        # themselves measurements.
        return ("inferred", float(node.confidence))

    context_key = geometry.arrays.get("context")
    occupancy_key = geometry.arrays.get("occupancy")
    if not context_key or context_key not in world.arrays:
        return ("inferred", float(node.confidence))

    context = np.asarray(world.arrays[context_key])
    occupancy = np.asarray(world.arrays[occupancy_key]).astype(bool)
    solid = context[occupancy]
    if solid.size == 0:
        return ("inferred", 0.0)
    invented = ((solid & Ctx.SPARSE_EVIDENCE) | (solid & Ctx.OCCLUDED)).astype(bool)
    measured = 1.0 - float(invented.mean())
    # Never `observed`. SPEC.md is explicit and correct about this: a wall
    # reconstructed from a point cloud is not a measurement however dense the
    # evidence -- only the point cloud itself was observed.
    #
    # A surface whose cells are mostly backed by returns is `derived`: the plane
    # fit and its clipping are a deterministic consequence of those returns, and
    # v0.3 has a word for exactly that. This replaces `hybrid`, which said only
    # "a composition of states" -- a vaguer claim about a better-understood
    # thing. The quantitative story stays in measured_fraction, which is where
    # it belongs; the state says which kind of claim is being made, not how
    # much of it.
    if measured >= 0.35:
        return ("derived", measured)
    return ("inferred", measured)


def _confidence(node, measured: float) -> dict:
    overall = float(np.clip(node.confidence, 0.0, 1.0))
    return {
        "overall": round(overall, 4),
        "geometry": round(float(np.clip(measured, 0.0, 1.0)), 4),
        "semantics": round(overall, 4),
        "topology": round(overall, 4),
    }


def _provenance(node, sources: list[str], parameters_hash: str) -> list[dict]:
    record = {
        "mode": PROVENANCE_MODE.get(node.stage, "geometric_inference"),
        "algorithm": f"lidarworld.{node.stage or 'reconstruct'}",
        "version": SIR_VERSION,
        "parameters_hash": parameters_hash,
    }
    if sources:
        record["source_ids"] = sources
    return [record]


def _bbox_geometry(node, bounds, parameters_hash: str) -> dict:
    lo, hi = bounds
    return {
        "id": f"g_{node.id}",
        "representation": "bbox",
        "role": "extent",
        "bbox": {"min": [round(float(v), 4) for v in lo],
                 "max": [round(float(v), 4) for v in hi]},
        "provenance": [{"mode": PROVENANCE_MODE.get(node.stage, "geometric_inference"),
                        "algorithm": f"lidarworld.{node.stage or 'reconstruct'}",
                        "version": SIR_VERSION,
                        "parameters_hash": parameters_hash}],
        "metadata": {},
    }


def _node_bounds(world: World, node, slot_bounds: dict) -> tuple | None:
    if node.geometry and node.geometry.bounds:
        b = node.geometry.bounds
        return (np.asarray(b[:3]) + world.origin, np.asarray(b[3:]) + world.origin)
    if node.geometry and node.geometry.kind == "instance":
        frame = node.geometry.frame
        centre = np.asarray(frame.get("position", [0, 0, 0]), dtype=float)
        size = np.asarray(frame.get("size", [1, 1, 1]), dtype=float)
        half = np.array([abs(size[0]), abs(size[1]), 0.0])
        return (centre - half, centre + half + np.array([0, 0, abs(size[2])]))
    if node.id in slot_bounds:
        lo, hi = slot_bounds[node.id]
        return (lo + world.origin, hi + world.origin)
    return None


def _mesh_bounds_by_node(world: World) -> dict:
    """AABB per surface node, taken from the vertices it owns."""
    from ..backends.web import mesh_slot_names

    if "mesh/positions" not in world.arrays:
        return {}
    positions = np.asarray(world.arrays["mesh/positions"], dtype=np.float64)
    slots = np.asarray(world.arrays["mesh/node"], dtype=np.int64)
    names = mesh_slot_names(world)
    out = {}
    for slot in np.unique(slots):
        if slot >= len(names):
            continue
        sel = slots == slot
        out[names[slot]] = (positions[sel].min(axis=0), positions[sel].max(axis=0))
    return out


def build_document(world: World, *, world_id: str | None = None) -> dict:
    """Convert a compiled world into a SIR v0.1 document (as a dict)."""
    parameters_hash = hashlib.sha256(
        json.dumps([s.to_json() for s in world.stages], sort_keys=True).encode()
    ).hexdigest()[:16]

    source_ids = [f"obs_{s.id}" for s in world.sources]
    slot_bounds = _mesh_bounds_by_node(world)

    entities = []
    for node in world.nodes.values():
        state, measured = _epistemic_state(world, node)
        bounds = _node_bounds(world, node, slot_bounds)

        entity = {
            "id": node.id.replace("/", "."),
            "name": node.id,
            "kind": _first(KIND, node.role, "object"),
            "class": citygml_type(node.role, surface=node.kind in ("surface", "opening", "terrain")),
            "semantic_resolution": _first(RESOLUTION, node.role, "object"),
            "epistemic_state": state,
            "confidence": _confidence(node, measured),
            "geometry": [],
            "provenance": _provenance(node, source_ids, parameters_hash),
            "attributes": {
                "lidarworld_role": node.role,
                "semantic": node.semantic,
                "support_points": int(node.support),
                "measured_fraction": round(float(measured), 4),
                **{k: v for k, v in (node.attrs or {}).items()
                   if isinstance(v, (int, float, str, bool))},
            },
        }
        if node.tags:
            entity["attributes"]["tags"] = list(node.tags)
        if bounds is not None:
            entity["geometry"].append(_bbox_geometry(node, bounds, parameters_hash))
        entities.append(entity)

    known = {e["id"] for e in entities}
    relations = []
    for node in world.nodes.values():
        if not node.parent:
            continue
        a, b = node.id.replace("/", "."), node.parent.replace("/", ".")
        if a in known and b in known:
            relations.append({
                "id": f"r_part_{a}",
                "type": "part_of",
                "source": a, "target": b,
                "confidence": round(float(node.confidence), 4),
                "provenance": _provenance(node, source_ids, parameters_hash),
                "attributes": {},
            })
    for i, edge in enumerate(world.edges):
        a, b = edge.a.replace("/", "."), edge.b.replace("/", ".")
        if a not in known or b not in known:
            continue
        relations.append({
            "id": f"r_{i}_{edge.relation}",
            "type": RELATION_TYPE.get(edge.relation, edge.relation),
            "source": a, "target": b,
            "confidence": round(float(np.clip(edge.confidence, 0, 1)), 4),
            "provenance": [{"mode": "geometric_inference",
                            "algorithm": "lidarworld.topology",
                            "version": SIR_VERSION,
                            "parameters_hash": parameters_hash}],
            "attributes": {k: v for k, v in (edge.attrs or {}).items()
                           if isinstance(v, (int, float, str, bool))},
        })

    observations = [{
        "id": f"obs_{s.id}",
        "modality": "lidar",
        "uri": s.uri or f"{s.id}.las",
        "frame": s.crs or "local_ENU",
        "sensor_id": s.sensor or None,
        "timestamp": s.acquired or None,
        "sha256": None,
        "calibration": {"adapter": s.adapter, "crs": s.crs},
        "preprocessing": [{"stage": st.name, "seconds": st.seconds} for st in world.stages],
        "metadata": {"license": s.license, "attribution": s.attribution,
                     "point_count": int(s.point_count), "notes": s.notes},
    } for s in world.sources]

    return {
        "sir_version": SIR_VERSION,
        "world": {
            "id": world_id or f"world_{world.name}",
            "name": world.name,
            "linear_unit": "m",
            "up_axis": "Z",
            "coordinate_frame": world.crs or "local_ENU",
            "crs": world.crs or None,
            "origin": [float(v) for v in world.origin],
            "metadata": {
                "compiler": "lidarworld",
                "spatial_ir_schema": world.schema,
                "bounds": [[float(v) for v in row] for row in world.bounds],
            },
        },
        "entities": entities,
        "relations": relations,
        "observations": observations,
        "alternatives": [],
        # What the compiler supplied rather than measured, and why it was
        # allowed to. Without these the finished world cannot answer "was this
        # part measured?" -- an inferred wall and a measured one are the same
        # triangles, so the answer has to be written when the decision is made.
        "repairs": [r.to_dict() for r in world.repairs.repairs],
        "gaps": [g.to_dict() for g in world.repairs.gaps],
        "metadata": {
            "generator": "lidarworld.ir.sir",
            "stages": [s.to_json() for s in world.stages],
            "repair_summary": world.repairs.summary(),
            "note": "Context bitmasks and theme rules have no SIR v0.1 equivalent; "
                    "epistemic_state is derived from per-tile measured evidence.",
        },
    }


def write_document(world: World, path: str | Path, *, world_id: str | None = None) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = build_document(world, world_id=world_id)
    path.write_text(json.dumps(document, indent=1))
    from collections import Counter
    states = Counter(e["epistemic_state"] for e in document["entities"])
    return {"path": str(path), "entities": len(document["entities"]),
            "relations": len(document["relations"]),
            "observations": len(document["observations"]),
            "epistemic": dict(states), "bytes": path.stat().st_size}

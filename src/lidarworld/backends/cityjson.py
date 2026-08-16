"""CityJSON 1.1 backend -- the geospatial target.

CityGML 3.0 is the closest existing standard to this project's Spatial IR: a
platform-independent semantic model for 3D urban objects with hierarchy,
multiple levels of detail and extensible attributes. Exporting to CityJSON (its
compact JSON encoding) is therefore both a practical interoperability win --
QGIS, FME, 3dfier, azul, ninja and the 3D BAG tooling all read it -- and a
sanity check that the IR's vocabulary maps onto an established ontology.

What survives the trip: the building/surface hierarchy, CityGML boundary-surface
semantics, openings, vegetation and city furniture, plus every node's confidence
and provenance as attributes.

What does not: the context bitmask. CityGML has no place for "this tile is on a
convex corner beside a window", which is precisely the information this project
adds on top -- so it is carried in `+lidarworld` extension attributes rather
than silently dropped.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..roles.taxonomy import Ctx, ROLE_IDS, citygml_type
from ..types import World
from .web import mesh_slot_names

SCALE = 0.001


def export(world: World, out_path: str | Path, *, lod: str = "2",
           include_context: bool = True) -> dict:
    """Write a CityJSON 1.1 file. Returns a small summary."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    positions = np.asarray(world.arrays["mesh/positions"], dtype=np.float64)
    indices = np.asarray(world.arrays["mesh/indices"], dtype=np.int64).reshape(-1, 3)
    node_attr = np.asarray(world.arrays["mesh/node"], dtype=np.int64)
    ctx_attr = np.asarray(world.arrays["mesh/ctx"], dtype=np.uint32)
    role_attr = np.asarray(world.arrays["mesh/role"], dtype=np.int64)

    # CityJSON stores quantised integer vertices plus a transform.
    world_positions = positions + world.origin
    translate = world_positions.min(axis=0)
    vertices = np.round((world_positions - translate) / SCALE).astype(np.int64)

    slot_names = mesh_slot_names(world)
    tri_slot = node_attr[indices[:, 0]]

    city_objects: dict[str, dict] = {}

    # --- parents: buildings, vegetation, furniture -------------------------
    for node in world.nodes.values():
        if node.kind not in ("object", "instance", "terrain"):
            continue
        obj_type = citygml_type(node.role, surface=False)
        entry = {
            "type": obj_type,
            "attributes": _attributes(node),
            "geometry": [],
        }
        children = [c for c in node.children if c in world.nodes]
        if children:
            entry["children"] = children
        if node.kind == "instance" and node.geometry and node.geometry.frame:
            frame = node.geometry.frame
            entry["attributes"].update({
                "height": round(float(frame.get("size", [0, 0, 0])[2]), 2),
                "position": [round(float(v), 3) for v in frame.get("position", [0, 0, 0])],
            })
        city_objects[node.id] = entry

    # --- boundary surfaces --------------------------------------------------
    for slot, name in enumerate(slot_names):
        sel = tri_slot == slot
        if not sel.any():
            continue
        node = world.nodes.get(name)
        role = ROLE_IDS[min(int(role_attr[indices[sel][0, 0]]), len(ROLE_IDS) - 1)] \
            if node is None else node.role
        boundaries = [[[int(a), int(b), int(c)]] for a, b, c in indices[sel]]

        surface_type = citygml_type(role, surface=True)
        geometry = {
            "type": "MultiSurface",
            "lod": lod,
            "boundaries": boundaries,
            "semantics": {
                "surfaces": [{"type": surface_type}],
                "values": [0] * len(boundaries),
            },
        }

        if name in city_objects:                       # terrain
            city_objects[name]["geometry"].append(geometry)
            continue

        attributes = _attributes(node) if node is not None else {"role": role}
        if include_context:
            attributes["+lidarworld_context"] = _context_summary(ctx_attr[indices[sel][:, 0]])
        entry = {
            "type": _boundary_object_type(surface_type),
            "attributes": attributes,
            "geometry": [geometry],
        }
        parent = node.parent if node is not None else None
        if parent and parent in city_objects:
            entry["parents"] = [parent]
        city_objects[name] = entry

    # --- openings as child objects of their surface ------------------------
    for node in world.nodes.values():
        if node.kind != "opening" or not node.geometry:
            continue
        frame = node.geometry.frame
        city_objects[node.id] = {
            "type": citygml_type(node.role, surface=True),
            "attributes": {
                **_attributes(node),
                "width": node.attrs.get("width"),
                "height": node.attrs.get("height"),
                "position": [round(float(v), 3) for v in frame.get("position", [0, 0, 0])],
            },
            "parents": [node.parent] if node.parent in city_objects else [],
        }

    document = {
        "type": "CityJSON",
        "version": "1.1",
        "transform": {"scale": [SCALE] * 3, "translate": translate.tolist()},
        "CityObjects": city_objects,
        "vertices": vertices.tolist(),
        "metadata": {
            "geographicalExtent": [
                *(world_positions.min(axis=0)).tolist(),
                *(world_positions.max(axis=0)).tolist()],
            "referenceSystem": world.crs or "",
            "title": world.name,
        },
        "extensions": {},
        "+lidarworld": {
            "schema": world.schema,
            "note": "Compiled from a theme-independent Spatial IR. Context bitmasks "
                    "and theme rules have no CityGML equivalent and are summarised "
                    "in +lidarworld_context attributes.",
            "contextFlags": {name: bit for bit, name in sorted(Ctx.NAMES.items())},
            "sources": [s.to_json() for s in world.sources],
            "stages": [s.to_json() for s in world.stages],
        },
    }

    out_path.write_text(json.dumps(document, separators=(",", ":")))
    return {"path": str(out_path), "cityObjects": len(city_objects),
            "vertices": len(vertices), "bytes": out_path.stat().st_size}


def _boundary_object_type(surface_type: str) -> str:
    """CityJSON needs a city-object type for a standalone boundary surface."""
    return {
        "WallSurface": "BuildingPart", "RoofSurface": "BuildingPart",
        "OuterFloorSurface": "BuildingPart", "OuterCeilingSurface": "BuildingPart",
        "GroundSurface": "TINRelief", "WaterSurface": "WaterBody",
    }.get(surface_type, "GenericCityObject")


def _attributes(node) -> dict:
    if node is None:
        return {}
    attrs = {
        "role": node.role,
        "semantic": node.semantic,
        "confidence": round(float(node.confidence), 3),
        "support": int(node.support),
    }
    if node.stage:
        attrs["+lidarworld_stage"] = node.stage
    if node.sources:
        attrs["+lidarworld_sources"] = node.sources
    for key, value in (node.attrs or {}).items():
        if key == "context":
            continue
        if isinstance(value, (int, float, str, bool)):
            attrs[key] = value
    return attrs


def _context_summary(masks: np.ndarray) -> dict:
    """How many triangles of this surface carry each context flag."""
    out = {}
    for bit, name in sorted(Ctx.NAMES.items()):
        count = int((masks & bit).astype(bool).sum())
        if count:
            out[name] = count
    return out

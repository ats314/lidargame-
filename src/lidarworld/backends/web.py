"""Web backend: emit what the bundled viewer streams.

The split is deliberate and is the whole architecture in miniature:

    world.bin    positions / normals / uv / **context** / role / node -- geometry
                 only, no material anywhere in it
    world.json   header, node graph, instances, provenance
    themes/*/    rule tables and baked textures

The viewer loads `world.bin` once and re-evaluates theme rules against the
per-vertex context attribute whenever you switch theme, so a re-skin costs one
small JSON fetch and zero geometry work.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..roles.taxonomy import Ctx, ROLES, ROLE_IDS
from ..types import World

FORMAT_VERSION = 3

#: Interleaved vertex layout, in order. (name, components, dtype)
VERTEX_LAYOUT = [
    ("position", 3, "f4"),
    ("normal", 3, "f4"),
    ("uv", 2, "f4"),
    ("ctx", 1, "u4"),
    ("role", 1, "u4"),
    ("node", 1, "u4"),
]
VERTEX_STRIDE = sum(c * 4 for _, c, _ in VERTEX_LAYOUT)


def mesh_slot_names(world: World) -> list[str]:
    """Node id for each value of the mesh `node` attribute.

    The reconstruct stage assigns slot 0 to terrain and then one slot per tiled
    surface in node order. Every consumer that needs to get from a triangle back
    to a graph node -- this exporter, CityJSON, forward validation -- reads the
    order from here rather than reimplementing it.
    """
    names = ["terrain"]
    for node in world.nodes.values():
        if node.kind == "surface" and node.geometry and node.geometry.kind == "tiled_plane":
            names.append(node.id)
    return names


def export(world: World, out_dir: str | Path, *, themes: list[str] | None = None,
           include_points: bool = True, max_points: int = 400_000) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh = world.arrays.get("mesh/positions")
    if mesh is None:
        raise ValueError("world has no mesh arrays -- run the reconstruct stage first")

    positions = np.asarray(world.arrays["mesh/positions"], dtype=np.float32)
    normals = np.asarray(world.arrays["mesh/normals"], dtype=np.float32)
    uv = np.asarray(world.arrays["mesh/uv"], dtype=np.float32)
    ctx = np.asarray(world.arrays["mesh/ctx"], dtype=np.uint32)
    role = np.asarray(world.arrays["mesh/role"], dtype=np.uint32)
    node = np.asarray(world.arrays["mesh/node"], dtype=np.uint32)
    indices = np.asarray(world.arrays["mesh/indices"], dtype=np.uint32).reshape(-1)

    n = len(positions)
    interleaved = np.zeros((n, VERTEX_STRIDE // 4), dtype=np.float32)
    interleaved[:, 0:3] = positions
    interleaved[:, 3:6] = normals
    interleaved[:, 6:8] = uv
    view = interleaved.view(np.uint32)
    view[:, 8] = ctx
    view[:, 9] = role
    view[:, 10] = node

    blob = bytearray()
    vertex_offset = 0
    blob += interleaved.tobytes()
    index_offset = len(blob)
    blob += indices.astype(np.uint32).tobytes()

    point_offset = point_count = 0
    if include_points and world.points is not None and len(world.points):
        pc = world.points
        step = max(1, len(pc) // max_points)
        xyz = pc.xyz[::step].astype(np.float32)
        cls = pc.get("semantic")
        rl = pc.get("role")
        conf = pc.get("role_confidence")
        point_count = len(xyz)
        packed = np.zeros((point_count, 4), dtype=np.float32)
        packed[:, 0:3] = xyz
        aux = packed.view(np.uint32)
        aux[:, 3] = ((np.asarray(cls[::step] if cls is not None else np.zeros(point_count), np.uint32) & 0xFF)
                     | ((np.asarray(rl[::step] if rl is not None else np.zeros(point_count), np.uint32) & 0xFF) << 8)
                     | ((np.clip(np.asarray(conf[::step] if conf is not None else np.ones(point_count)) * 255, 0, 255).astype(np.uint32) & 0xFF) << 16))
        point_offset = len(blob)
        blob += packed.tobytes()

    (out_dir / "world.bin").write_bytes(bytes(blob))

    instances = []
    for node_obj in world.nodes.values():
        if node_obj.geometry and node_obj.geometry.kind == "instance":
            frame = node_obj.geometry.frame
            instances.append({
                "id": node_obj.id, "role": node_obj.role,
                "position": frame.get("position", [0, 0, 0]),
                "size": frame.get("size", [1, 1, 1]),
                "yaw": frame.get("yaw", 0.0),
                "confidence": round(node_obj.confidence, 3),
                "attrs": node_obj.attrs,
            })

    header = {
        "format": "lidarworld/web",
        "version": FORMAT_VERSION,
        "name": world.name,
        "crs": world.crs,
        "units": world.units,
        "origin": [float(v) for v in world.origin],
        "bounds": [[float(v) for v in row] for row in world.bounds],
        "vertexLayout": [{"name": n, "components": c, "type": t} for n, c, t in VERTEX_LAYOUT],
        "vertexStride": VERTEX_STRIDE,
        "mesh": {
            "vertexOffset": vertex_offset, "vertexCount": n,
            "indexOffset": index_offset, "indexCount": int(len(indices)),
        },
        "points": {"offset": point_offset, "count": point_count, "stride": 16},
        "instances": instances,
        "roles": [{"id": r, "label": ROLES[r].label, "color": list(ROLES[r].debug_color),
                   "reconstruct": ROLES[r].reconstruct} for r in ROLE_IDS],
        "contextFlags": {name: bit for bit, name in sorted(Ctx.NAMES.items())},
        "themes": themes or [],
        "graph": {
            "nodes": [n.to_json() for n in world.nodes.values() if n.kind != "surface_tiles"],
            "edges": [e.to_json() for e in world.edges],
        },
        "sources": [s.to_json() for s in world.sources],
        "stages": [s.to_json() for s in world.stages],
        "summary": world.summary(),
        # Promoted out of notes so the viewer can say, in its header, whether
        # this world was measured or generated. Same triangles either way.
        "generated_from": world.notes.get("generated_from"),
        "notes": world.notes,
    }
    (out_dir / "world.json").write_text(json.dumps(header, indent=1))
    return {"vertices": n, "triangles": int(len(indices) // 3), "points": point_count,
            "instances": len(instances), "bytes": len(blob), "dir": str(out_dir)}

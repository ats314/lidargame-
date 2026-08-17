"""glTF 2.0 backend -- the universal target.

glTF has no concept of a theme, so this is where a world stops being
theme-independent: the pack is resolved once, per vertex, and the mesh is split
into primitives by material. Everything that reads glTF (Blender, Godot, Unity,
Unreal, three.js, USD via converters, most GIS viewers) then reads the world.

Point being: the *compiler* stays engine-agnostic, and materialisation happens
in the backend, once, at the boundary.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from ..roles.taxonomy import ROLE_IDS
from ..themes import ThemePack, procedural
from ..themes.png import write as write_png
from ..themes.request import MaterialRequest
from ..types import World

# Z-up (survey) -> Y-up (glTF), column-major.
Z_UP_TO_Y_UP = [1, 0, 0, 0,  0, 0, -1, 0,  0, 1, 0, 0,  0, 0, 0, 1]


def _placed(rotation: list, origin) -> list:
    """`rotation` with a translation that puts recentred vertices back.

    glTF matrices are column-major, so the translation occupies elements 12-14.
    The offset is the origin carried through the same axis swap the vertices
    went through, otherwise the world lands rotated about the wrong point.
    """
    matrix = list(rotation)
    basis = np.asarray(rotation, dtype=float).reshape(4, 4).T[:3, :3]
    matrix[12], matrix[13], matrix[14] = (basis @ np.asarray(origin, dtype=float)).tolist()
    return [float(v) for v in matrix]


def resolve_vertex_materials(world: World, pack: ThemePack) -> tuple[np.ndarray, list]:
    """Material index per vertex, resolving each distinct (role, ctx) pair once."""
    ctx = np.asarray(world.arrays["mesh/ctx"], dtype=np.uint32)
    role = np.asarray(world.arrays["mesh/role"], dtype=np.uint32)
    combined = (role.astype(np.uint64) << 32) | ctx.astype(np.uint64)
    unique, inverse = np.unique(combined, return_inverse=True)

    specs: list = []
    spec_index: dict[str, int] = {}
    lut = np.zeros(len(unique), dtype=np.int32)
    for i, packed in enumerate(unique):
        role_id = ROLE_IDS[min(int(packed >> 32), len(ROLE_IDS) - 1)]
        spec, _ = pack.resolve(MaterialRequest(role=role_id, context=int(packed & 0xFFFFFFFF)))
        if spec.id not in spec_index:
            spec_index[spec.id] = len(specs)
            specs.append(spec)
        lut[i] = spec_index[spec.id]
    return lut[inverse], specs


def export(world: World, pack: ThemePack, out_dir: str | Path, *,
           name: str = "world", bake_textures: bool = True) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Recentre before the cast, or projected coordinates lose the world.
    #
    # A UTM northing is about 5.93e6. float32 has a 24-bit mantissa, so it
    # resolves 0.5 m there -- and 0.0625 m at a 5.7e5 easting. Exporting raw
    # therefore snapped every vertex to a half-metre grid in north only, giving
    # a stair-stepped, anisotropically warped mesh that no triangle count or
    # residual could show. Measured on a Hamburg block: the smallest distinct
    # northing step in the exported buffer was exactly 0.5 m.
    #
    # Vertices go out local to `geo_origin` and the node matrix carries the
    # offset, so absolute placement survives and float32 gets its full range.
    raw = np.asarray(world.arrays["mesh/positions"], dtype=np.float64)
    geo_origin = ((raw.min(axis=0) + raw.max(axis=0)) / 2.0
                  if len(raw) else np.zeros(3))
    positions = (raw - geo_origin).astype(np.float32)
    normals = np.asarray(world.arrays["mesh/normals"], dtype=np.float32)
    uv = np.asarray(world.arrays["mesh/uv"], dtype=np.float32)
    indices = np.asarray(world.arrays["mesh/indices"], dtype=np.uint32).reshape(-1, 3)

    vertex_material, specs = resolve_vertex_materials(world, pack)
    tri_material = vertex_material[indices[:, 0]]

    # UVs are in world metres; glTF has no per-material UV scale, so bake the
    # per-material tiling into the coordinates at export time.
    scaled_uv = uv.copy()
    for i, spec in enumerate(specs):
        sel = vertex_material == i
        if sel.any():
            scaled_uv[sel] /= max(spec.scale_m, 1e-3)

    buffer = bytearray()

    def push(array: np.ndarray) -> tuple[int, int]:
        while len(buffer) % 4:
            buffer.append(0)
        offset = len(buffer)
        buffer.extend(np.ascontiguousarray(array).tobytes())
        return offset, len(buffer) - offset

    pos_off, pos_len = push(positions)
    nrm_off, nrm_len = push(normals)
    uv_off, uv_len = push(scaled_uv)

    buffer_views = [
        {"buffer": 0, "byteOffset": pos_off, "byteLength": pos_len, "target": 34962},
        {"buffer": 0, "byteOffset": nrm_off, "byteLength": nrm_len, "target": 34962},
        {"buffer": 0, "byteOffset": uv_off, "byteLength": uv_len, "target": 34962},
    ]
    accessors = [
        {"bufferView": 0, "componentType": 5126, "count": len(positions), "type": "VEC3",
         "min": positions.min(axis=0).tolist(), "max": positions.max(axis=0).tolist()},
        {"bufferView": 1, "componentType": 5126, "count": len(normals), "type": "VEC3"},
        {"bufferView": 2, "componentType": 5126, "count": len(scaled_uv), "type": "VEC2"},
    ]

    primitives = []
    for i, spec in enumerate(specs):
        sel = tri_material == i
        if not sel.any():
            continue
        tri = indices[sel].reshape(-1).astype(np.uint32)
        off, length = push(tri)
        buffer_views.append({"buffer": 0, "byteOffset": off, "byteLength": length, "target": 34963})
        accessors.append({"bufferView": len(buffer_views) - 1, "componentType": 5125,
                          "count": len(tri), "type": "SCALAR"})
        primitives.append({
            "attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2},
            "indices": len(accessors) - 1,
            "material": i,
        })

    images, textures, materials = [], [], []
    for i, spec in enumerate(specs):
        pbr = {
            "baseColorFactor": [*spec.base_color, spec.opacity],
            "metallicFactor": float(spec.metallic),
            "roughnessFactor": float(spec.roughness),
        }
        if bake_textures and spec.kind == "procedural":
            maps = procedural.bake(spec)
            rel = f"tex/{spec.id}_albedo.png"
            write_png(out_dir / rel, maps["albedo"])
            images.append({"uri": rel})
            textures.append({"source": len(images) - 1, "sampler": 0})
            pbr["baseColorTexture"] = {"index": len(textures) - 1}
            pbr["baseColorFactor"] = [1, 1, 1, spec.opacity]
        material = {
            "name": spec.id,
            "pbrMetallicRoughness": pbr,
            "doubleSided": True,
            "extras": {"license": spec.license, "source": spec.source,
                       "era": list(spec.era), "tags": list(spec.tags),
                       "worldScaleMetres": spec.scale_m},
        }
        if any(spec.emissive):
            material["emissiveFactor"] = list(spec.emissive)
        if spec.opacity < 1.0:
            material["alphaMode"] = "BLEND"
        materials.append(material)

    bin_name = f"{name}.bin"
    (out_dir / bin_name).write_bytes(bytes(buffer))

    gltf = {
        "asset": {"version": "2.0", "generator": "lidarworld",
                  "copyright": "; ".join(f"{s.id}: {s.license}" for s in world.sources)},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": world.name, "mesh": 0,
                   "matrix": _placed(Z_UP_TO_Y_UP, geo_origin),
                   "extras": {"geoOrigin": [float(v) for v in geo_origin],
                              "crs": world.crs}}],
        "meshes": [{"name": world.name, "primitives": primitives}],
        "buffers": [{"uri": bin_name, "byteLength": len(buffer)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
        "materials": materials,
        "samplers": [{"wrapS": 10497, "wrapT": 10497, "magFilter": 9729, "minFilter": 9987}],
        "extras": {
            "theme": pack.id,
            "spatialIRSchema": world.schema,
            "sources": [s.to_json() for s in world.sources],
            "note": "Materialised from a theme-independent Spatial IR; "
                    "re-export with a different pack to re-skin.",
        },
    }
    if images:
        gltf["images"] = images
        gltf["textures"] = textures

    path = out_dir / f"{name}.gltf"
    path.write_text(json.dumps(gltf, indent=1))
    return {"path": str(path), "primitives": len(primitives), "materials": len(materials),
            "triangles": int(len(indices)), "bytes": len(buffer)}

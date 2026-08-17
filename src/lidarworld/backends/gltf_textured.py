"""Write measured appearance straight to glTF, as the baseline a theme must beat.

`gltf.py` materialises a themed world: the Spatial IR carries no appearance, and
a theme pack decides what a wall looks like. This module does the opposite and
is deliberately kept separate from it. It takes surfaces that already carry
photographic evidence -- a texture image and UVs measured by somebody's
photogrammetry -- and writes them out unchanged.

Two reasons that is worth having.

It is an upper baseline. Once a Hamburg block exists with its own 20 cm
photographs on it, "does the procedural material look better or worse than the
photograph" becomes a comparison instead of an opinion. Without it, every theme
is judged against memory.

It is the Unreal path. glTF imports natively; the same file is the deliverable
whether the appearance came from a photograph or from a theme.

The invariant still holds. Nothing before `backends/` names a material or a
shader: what ingest carries is an image reference and UVs, which are evidence in
the same sense that RGB on a return is evidence. The decision that the image
becomes a PBR base-colour map is made here and nowhere earlier.

Output is binary glTF, self-contained, JPEGs embedded rather than referenced.
"""
from __future__ import annotations

import base64
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..reconstruct.tessellate import close_ring, triangulate, wall_frame

#: 20 cm imagery is coarse enough that a bilinear magnify is the honest choice:
#: nearest turns a facade into visible blocks at walking distance, which reads as
#: a rendering bug rather than as the resolution limit it actually is.
LINEAR = 9729
LINEAR_MIPMAP_LINEAR = 9987
REPEAT = 10497


@dataclass
class Face:
    """One planar polygon with optional measured appearance.

    `image` is a path to a source image, not a material name. `uv` is per
    exterior-ring vertex in the source's own convention; the v-flip to glTF
    happens here.

    `surface_id` and `building_id` are carried through because the appearance
    pipeline needs them: a mask detected in UV0 has to be mappable back to a
    named 3D surface, and a material instance has to be attributable to a
    building. Dropping them at the conversion boundary makes that impossible
    afterwards.
    """
    ring: np.ndarray                 # (N, 3) world coordinates
    uv: np.ndarray | None = None     # (N, 2)
    image: str | None = None
    kind: str = ""                   # wall | roof | ground, for the fallback colour
    surface_id: str | None = None
    building_id: str | None = None


#: Where a surface has no texture binding, the fallback names the *class*, not a
#: style. These are neutral greys chosen so an untextured surface is obviously
#: untextured in a render rather than blending into a textured one.
FALLBACK = {
    "wall":   (0.72, 0.70, 0.67, 1.0),
    "roof":   (0.45, 0.42, 0.40, 1.0),
    "ground": (0.55, 0.55, 0.53, 1.0),
    "":       (0.60, 0.60, 0.60, 1.0),
}


def _pad(buffer: bytearray, alignment: int = 4, fill: bytes = b"\x00") -> None:
    while len(buffer) % alignment:
        buffer += fill


def export(faces: list[Face], out_path: str | Path, *,
           image_root: str | Path = ".", origin: np.ndarray | None = None,
           y_up: bool = True) -> dict:
    """Write `faces` to a self-contained .glb. Returns what was actually written.

    Coordinates are recentred before being cast to float32. A Hamburg easting is
    566,000 and a northing 5,934,000; float32 resolves about 0.5 m there, so
    exporting raw makes every wall visibly jitter. `origin` defaults to the
    centre of the data and is returned so the caller can georeference again.
    """
    image_root = Path(image_root)
    if origin is None:
        allpts = np.vstack([f.ring for f in faces]) if faces else np.zeros((1, 3))
        origin = allpts.mean(axis=0)
    origin = np.asarray(origin, dtype=float)

    # One primitive per source image keeps the draw-call count at the number of
    # distinct textures rather than the number of surfaces.
    groups: dict[str | None, list[Face]] = {}
    for face in faces:
        groups.setdefault(face.image if face.uv is not None else None,
                          []).append(face)

    buffer = bytearray()
    views: list[dict] = []
    accessors: list[dict] = []
    primitives: list[dict] = []
    materials: list[dict] = []
    textures: list[dict] = []
    images: list[dict] = []
    skipped = {"degenerate": 0, "no_uv_match": 0}

    def add_view(data: bytes, target: int | None = None) -> int:
        _pad(buffer)
        offset = len(buffer)
        buffer.extend(data)
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        if target is not None:
            view["bufferViewTarget"] = target
        views.append(view)
        return len(views) - 1

    def add_accessor(data: np.ndarray, kind: str, component: int) -> int:
        view = add_view(data.tobytes())
        accessor = {"bufferView": view, "componentType": component,
                    "count": int(len(data)), "type": kind}
        if kind in ("VEC3", "VEC2"):
            accessor["min"] = data.min(axis=0).tolist()
            accessor["max"] = data.max(axis=0).tolist()
        else:
            accessor["min"] = [int(data.min())]
            accessor["max"] = [int(data.max())]
        accessors.append(accessor)
        return len(accessors) - 1

    surfaces: list[dict] = []
    for image_name, members in groups.items():
        positions: list[np.ndarray] = []
        uvs: list[np.ndarray] = []
        metric: list[np.ndarray] = []
        indices: list[np.ndarray] = []
        base = 0
        for face in members:
            ring = close_ring(np.asarray(face.ring, dtype=float))
            tris = triangulate(ring)
            if not len(tris):
                skipped["degenerate"] += 1
                continue
            local = ring - origin

            # UV1: wall-local coordinates in *metres*, not normalised.
            #
            # Normalising by wall width and height was the obvious thing and it
            # is wrong: it makes one brick course span a garden shed and a
            # warehouse identically. The micro material has a real repeat -- a
            # brick is 240 mm -- so the shader needs a metric coordinate and
            # divides by its own tile size. `wall_frame` also keeps courses
            # running along the wall and stacking toward the sky, so masonry
            # does not rotate at a corner.
            u_axis, v_axis, _ = wall_frame(ring)
            su = local @ u_axis
            sv = local @ v_axis
            # glTF's v runs down the image; negating keeps world-up pointing up
            # the texture, so a directional material is not upside down.
            metric.append(np.column_stack([su, -sv]).astype(np.float32))

            if y_up:
                # Z-up right-handed -> glTF's Y-up right-handed.
                local = np.column_stack([local[:, 0], local[:, 2], -local[:, 1]])
            positions.append(local.astype(np.float32))
            if image_name is not None:
                uv = close_ring(np.asarray(face.uv, dtype=float))
                if len(uv) != len(ring):
                    skipped["no_uv_match"] += 1
                    uv = np.zeros((len(ring), 2))
                # CityGML puts (0,0) at the lower left; glTF at the upper left.
                # This is the identity channel and is passed through untouched
                # apart from that flip -- it is what makes the building itself.
                uvs.append(np.column_stack([uv[:, 0], 1.0 - uv[:, 1]]).astype(np.float32))
            indices.append(tris + base)
            base += len(ring)

            surfaces.append({
                "surface_id": face.surface_id, "building_id": face.building_id,
                "kind": face.kind, "image": image_name,
                "width_m": round(float(su.max() - su.min()), 3),
                "height_m": round(float(sv.max() - sv.min()), 3),
                "u_axis": [round(float(v), 6) for v in u_axis],
                "v_axis": [round(float(v), 6) for v in v_axis],
                "origin_xyz": [round(float(v), 3) for v in ring[0]],
            })

        if not positions:
            continue
        position = np.vstack(positions)
        index = np.vstack(indices).reshape(-1).astype(np.uint32)
        attributes = {"POSITION": add_accessor(position, "VEC3", 5126)}
        if uvs:
            attributes["TEXCOORD_0"] = add_accessor(np.vstack(uvs), "VEC2", 5126)
        if metric:
            attributes["TEXCOORD_1"] = add_accessor(np.vstack(metric), "VEC2", 5126)
        primitive = {"attributes": attributes,
                     "indices": add_accessor(index, "SCALAR", 5125),
                     "material": len(materials)}

        if image_name is not None:
            path = image_root / str(image_name).replace("\\", "/")
            data = path.read_bytes()
            mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
            images.append({"bufferView": add_view(data), "mimeType": mime})
            textures.append({"source": len(images) - 1, "sampler": 0})
            materials.append({
                "name": path.stem,
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": len(textures) - 1},
                    "metallicFactor": 0.0, "roughnessFactor": 0.9},
                "doubleSided": True})
        else:
            kinds = {f.kind for f in members}
            kind = kinds.pop() if len(kinds) == 1 else ""
            materials.append({
                "name": f"untextured_{kind or 'surface'}",
                "pbrMetallicRoughness": {
                    "baseColorFactor": list(FALLBACK.get(kind, FALLBACK[""])),
                    "metallicFactor": 0.0, "roughnessFactor": 0.95},
                "doubleSided": True})
        primitives.append(primitive)

    _pad(buffer)
    gltf = {
        "asset": {"version": "2.0", "generator": "lidarworld gltf_textured"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "measured"}],
        "meshes": [{"primitives": primitives}],
        "buffers": [{"byteLength": len(buffer)}],
        "bufferViews": views,
        "accessors": accessors,
        "materials": materials,
    }
    if textures:
        gltf["textures"] = textures
        gltf["images"] = images
        gltf["samplers"] = [{"magFilter": LINEAR, "minFilter": LINEAR_MIPMAP_LINEAR,
                             "wrapS": REPEAT, "wrapT": REPEAT}]

    payload = json.dumps(gltf, separators=(",", ":")).encode()
    payload += b" " * ((4 - len(payload) % 4) % 4)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as handle:
        handle.write(struct.pack("<III", 0x46546C67, 2,
                                 12 + 8 + len(payload) + 8 + len(buffer)))
        handle.write(struct.pack("<II", len(payload), 0x4E4F534A))
        handle.write(payload)
        handle.write(struct.pack("<II", len(buffer), 0x004E4942))
        handle.write(bytes(buffer))

    # The surface index is the piece that makes appearance work possible later.
    # A mask detected in UV0 is a point in image space; turning it back into a
    # place on a named wall needs that wall's frame and metric extent, and none
    # of that survives inside a glTF. Written beside the model rather than into
    # it, because it is metadata for a pipeline, not something an engine reads.
    index_path = out_path.with_suffix(".surfaces.json")
    index_path.write_text(json.dumps({
        "model": out_path.name,
        "origin": origin.tolist(),
        "uv0": "source appearance atlas, passed through unchanged",
        "uv1": "wall-local metres; divide by the material repeat in metres",
        "surfaces": surfaces,
    }, separators=(",", ":")))

    return {"path": str(out_path), "bytes": out_path.stat().st_size,
            "primitives": len(primitives), "materials": len(materials),
            "textures": len(textures),
            "triangles": sum(a["count"] for a in accessors
                             if a["type"] == "SCALAR") // 3,
            "origin": origin.tolist(), "skipped": skipped,
            "surfaces": len(surfaces), "surface_index": str(index_path)}


def export_mesh(positions: np.ndarray, uvs: np.ndarray,
                groups: list, out_path: str | Path, *,
                origin: np.ndarray | None = None, y_up: bool = True,
                max_texture_px: int | None = None) -> dict:
    """Write an already-triangulated mesh with per-group textures to .glb.

    `export` above takes polygons and tessellates them, which is right for
    CityGML surfaces and wrong for a photogrammetric mesh: a Helsinki 250 m
    subtile is millions of triangles that are already triangles, and routing
    them through ear clipping would be slow and would achieve nothing.

    `groups` is any sequence of objects with `.material`, `.image` and `.faces`
    -- the shape `ingest.objmesh.Group` has. One glTF primitive is emitted per
    group, so draw calls follow the texture count.

    Same discipline as `export`: coordinates are recentred before the float32
    cast, because these meshes are georeferenced and float32 cannot hold a
    projected coordinate without quantising it.
    """
    from collections import OrderedDict

    out_path = Path(out_path)
    positions = np.asarray(positions, dtype=np.float64)

    # Keep only vertices some triangle actually uses, and reindex.
    #
    # A cropped mesh arrives with the whole tile's vertex array and a subset of
    # its faces. Writing that out is not merely wasteful -- it makes the file
    # lie about its own extent, because POSITION min/max then describe the
    # uncropped tile. Anything that frames a camera from the accessor bounds
    # aims at empty space, which is exactly what happened: pedestrian cameras
    # were placed in the 250 m subtile's outer ring while only the central
    # 140 m had geometry, and rendered 0% coverage.
    used = np.zeros(len(positions), dtype=bool)
    for group in groups:
        faces = np.asarray(group.faces).reshape(-1)
        if len(faces):
            used[faces] = True
    if used.any() and not used.all():
        remap = np.full(len(positions), -1, dtype=np.int64)
        remap[used] = np.arange(int(used.sum()))
        positions = positions[used]
        uvs = np.asarray(uvs)
        if len(uvs) == len(used):
            uvs = uvs[used]
        groups = [type(g)(g.material, g.image,
                          remap[np.asarray(g.faces).reshape(-1, 3)])
                  for g in groups if len(np.asarray(g.faces))]

    if origin is None:
        origin = ((positions.min(axis=0) + positions.max(axis=0)) / 2.0
                  if len(positions) else np.zeros(3))
    origin = np.asarray(origin, dtype=float)

    local = positions - origin
    if y_up:
        local = np.column_stack([local[:, 0], local[:, 2], -local[:, 1]])
    local = local.astype(np.float32)
    uvs = np.asarray(uvs, dtype=np.float32)
    if len(uvs) != len(positions):
        uvs = np.zeros((len(positions), 2), dtype=np.float32)
    # OBJ's v runs up from the bottom-left; glTF's runs down from the top-left.
    flipped = np.column_stack([uvs[:, 0], 1.0 - uvs[:, 1]]).astype(np.float32)

    buffer = bytearray()
    views: list[dict] = []
    accessors: list[dict] = []

    def add_view(data: bytes) -> int:
        _pad(buffer)
        offset = len(buffer)
        buffer.extend(data)
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(data)})
        return len(views) - 1

    def add_accessor(data: np.ndarray, kind: str, component: int) -> int:
        view = add_view(data.tobytes())
        spec = {"bufferView": view, "componentType": component,
                "count": int(len(data)), "type": kind}
        if kind in ("VEC3", "VEC2"):
            spec["min"] = data.min(axis=0).tolist()
            spec["max"] = data.max(axis=0).tolist()
        else:
            spec["min"] = [int(data.min())] if len(data) else [0]
            spec["max"] = [int(data.max())] if len(data) else [0]
        accessors.append(spec)
        return len(accessors) - 1

    position_accessor = add_accessor(local, "VEC3", 5126)
    uv_accessor = add_accessor(flipped, "VEC2", 5126)

    images: list[dict] = []
    textures: list[dict] = []
    materials: list[dict] = []
    primitives: list[dict] = []
    image_slot: "OrderedDict[str, int]" = OrderedDict()
    embedded_bytes = 0

    for group in groups:
        faces = np.asarray(group.faces, dtype=np.uint32).reshape(-1)
        if not len(faces):
            continue
        primitive = {"attributes": {"POSITION": position_accessor,
                                    "TEXCOORD_0": uv_accessor},
                     "indices": add_accessor(faces, "SCALAR", 5125),
                     "material": len(materials)}
        if group.image is not None:
            key = str(group.image)
            if key not in image_slot:
                payload = Path(group.image).read_bytes()
                mime = "image/png" if key.lower().endswith(".png") else "image/jpeg"
                if max_texture_px:
                    payload, mime = _downscale(payload, max_texture_px, mime)
                images.append({"bufferView": add_view(payload), "mimeType": mime})
                textures.append({"source": len(images) - 1, "sampler": 0})
                image_slot[key] = len(textures) - 1
                embedded_bytes += len(payload)
            materials.append({
                "name": group.material,
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": image_slot[key]},
                    "metallicFactor": 0.0, "roughnessFactor": 0.95},
                "doubleSided": True})
        else:
            materials.append({
                "name": group.material or "untextured",
                "pbrMetallicRoughness": {
                    "baseColorFactor": list(FALLBACK[""]),
                    "metallicFactor": 0.0, "roughnessFactor": 0.95},
                "doubleSided": True})
        primitives.append(primitive)

    _pad(buffer)
    gltf = {
        "asset": {"version": "2.0", "generator": "lidarworld gltf_textured mesh"},
        "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "mesh",
                   "extras": {"geoOrigin": origin.tolist()}}],
        "meshes": [{"primitives": primitives}],
        "buffers": [{"byteLength": len(buffer)}],
        "bufferViews": views, "accessors": accessors, "materials": materials,
    }
    if textures:
        gltf["textures"] = textures
        gltf["images"] = images
        gltf["samplers"] = [{"magFilter": LINEAR,
                             "minFilter": LINEAR_MIPMAP_LINEAR,
                             "wrapS": REPEAT, "wrapT": REPEAT}]

    payload = json.dumps(gltf, separators=(",", ":")).encode()
    payload += b" " * ((4 - len(payload) % 4) % 4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as handle:
        handle.write(struct.pack("<III", 0x46546C67, 2,
                                 12 + 8 + len(payload) + 8 + len(buffer)))
        handle.write(struct.pack("<II", len(payload), 0x4E4F534A))
        handle.write(payload)
        handle.write(struct.pack("<II", len(buffer), 0x004E4942))
        handle.write(bytes(buffer))

    return {"path": str(out_path), "bytes": out_path.stat().st_size,
            "vertices": int(len(local)),
            "triangles": int(sum(len(np.asarray(g.faces).reshape(-1, 3))
                                 for g in groups)),
            "primitives": len(primitives), "textures": len(textures),
            "texture_bytes": embedded_bytes, "origin": origin.tolist()}


def _downscale(payload: bytes, max_px: int, mime: str) -> tuple[bytes, str]:
    """Shrink an embedded texture, for when a tile's images dwarf its geometry."""
    import io as _io

    from PIL import Image

    image = Image.open(_io.BytesIO(payload))
    if max(image.size) <= max_px:
        return payload, mime
    scale = max_px / max(image.size)
    image = image.convert("RGB").resize(
        (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
        Image.LANCZOS)
    out = _io.BytesIO()
    image.save(out, format="JPEG", quality=88)
    return out.getvalue(), "image/jpeg"


def data_uri(path: str | Path) -> str:
    """A .glb as a data: URI, for dropping into a self-contained page."""
    raw = Path(path).read_bytes()
    return "data:model/gltf-binary;base64," + base64.b64encode(raw).decode()

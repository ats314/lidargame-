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

from ..reconstruct.tessellate import close_ring, triangulate

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
    """
    ring: np.ndarray                 # (N, 3) world coordinates
    uv: np.ndarray | None = None     # (N, 2)
    image: str | None = None
    kind: str = ""                   # wall | roof | ground, for the fallback colour


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

    for image_name, members in groups.items():
        positions: list[np.ndarray] = []
        uvs: list[np.ndarray] = []
        indices: list[np.ndarray] = []
        base = 0
        for face in members:
            ring = close_ring(np.asarray(face.ring, dtype=float))
            tris = triangulate(ring)
            if not len(tris):
                skipped["degenerate"] += 1
                continue
            local = ring - origin
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
                uvs.append(np.column_stack([uv[:, 0], 1.0 - uv[:, 1]]).astype(np.float32))
            indices.append(tris + base)
            base += len(ring)

        if not positions:
            continue
        position = np.vstack(positions)
        index = np.vstack(indices).reshape(-1).astype(np.uint32)
        attributes = {"POSITION": add_accessor(position, "VEC3", 5126)}
        if uvs:
            attributes["TEXCOORD_0"] = add_accessor(np.vstack(uvs), "VEC2", 5126)
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

    return {"path": str(out_path), "bytes": out_path.stat().st_size,
            "primitives": len(primitives), "materials": len(materials),
            "textures": len(textures),
            "triangles": sum(a["count"] for a in accessors
                             if a["type"] == "SCALAR") // 3,
            "origin": origin.tolist(), "skipped": skipped}


def data_uri(path: str | Path) -> str:
    """A .glb as a data: URI, for dropping into a self-contained page."""
    raw = Path(path).read_bytes()
    return "data:model/gltf-binary;base64," + base64.b64encode(raw).decode()

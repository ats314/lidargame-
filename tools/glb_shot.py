"""Render a .glb to PNG without a browser or a GPU, so the world gets looked at.

`tools/shoot.py` drives headless Chromium over the viewer, which is the right
tool once geometry is in the viewer's own format. This one is for the step
before that: a .glb straight out of a backend, checked on its own before
anything downstream is blamed. It is also a round-trip test -- if this cannot
read the file, no engine will either.

A software rasteriser is enough. The point is not fidelity, it is catching the
failures that no metric shows: a facade texture applied upside down, a wall in
the wrong place, a building at the origin instead of its footprint, a UV
convention off by a flip. Every one of those has happened in this repo and none
of them moved a number.

    python tools/glb_shot.py build/hamburg/rathaus.glb --out build/shots

Perspective-correct interpolation, a z-buffer, and no lighting model beyond a
fixed headlamp: shading that hides a texture error would defeat the purpose.
"""
from __future__ import annotations

import argparse
import io
import json
import struct
from pathlib import Path
from urllib.parse import unquote

import numpy as np

COMPONENT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16,
             5125: np.uint32, 5126: np.float32}
COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def read_glb(path: str | Path) -> tuple[dict, bytes]:
    """Load .glb or .gltf. Returns the document and its single binary blob.

    Both forms are here because both backends are: `gltf_textured` writes a
    self-contained .glb, and the themed `gltf` backend writes .gltf with a
    sidecar .bin and a tex/ directory. A renderer that only read one of them
    could not compare them, which is the entire point of the round trip.
    """
    path = Path(path)
    raw = path.read_bytes()
    if raw[:4] != b"glTF":
        gltf = json.loads(raw)
        buffers = gltf.get("buffers", [])
        uri = buffers[0].get("uri") if buffers else None
        if uri is None:
            return gltf, b""
        if uri.startswith("data:"):
            import base64
            return gltf, base64.b64decode(uri.split(",", 1)[1])
        return gltf, (path.parent / unquote(uri)).read_bytes()

    magic, version, _ = struct.unpack_from("<III", raw, 0)
    if magic != 0x46546C67:
        raise ValueError(f"{path} is not a glb (magic {magic:#x})")
    offset, gltf, binary = 12, None, b""
    while offset < len(raw):
        length, kind = struct.unpack_from("<II", raw, offset)
        chunk = raw[offset + 8: offset + 8 + length]
        if kind == 0x4E4F534A:
            gltf = json.loads(chunk)
        elif kind == 0x004E4942:
            binary = chunk
        offset += 8 + length + ((4 - length % 4) % 4 if length % 4 else 0)
    if gltf is None:
        raise ValueError("no JSON chunk")
    return gltf, binary


def accessor(gltf: dict, binary: bytes, index: int) -> np.ndarray:
    spec = gltf["accessors"][index]
    view = gltf["bufferViews"][spec["bufferView"]]
    start = view.get("byteOffset", 0) + spec.get("byteOffset", 0)
    dtype = COMPONENT[spec["componentType"]]
    n = spec["count"] * COUNT[spec["type"]]
    values = np.frombuffer(binary, dtype=dtype, count=n, offset=start)
    return values.reshape(spec["count"], COUNT[spec["type"]])


def node_transforms(gltf: dict) -> dict[int, np.ndarray]:
    """Mesh index -> world matrix, walking the scene graph.

    Ignoring this is not a cosmetic shortcut. The themed backend keeps its
    geometry Z-up and puts the axis swap and the georeference offset in the node
    matrix, so a renderer that skips it draws the world on its side and in the
    wrong place -- and then the round-trip comparison measures the renderer
    rather than the compiler.
    """
    out: dict[int, np.ndarray] = {}

    def local(node: dict) -> np.ndarray:
        if "matrix" in node:
            return np.asarray(node["matrix"], dtype=float).reshape(4, 4).T
        matrix = np.eye(4)
        if "scale" in node:
            matrix[:3, :3] = np.diag(node["scale"])
        if "rotation" in node:
            x, y, z, w = node["rotation"]
            matrix[:3, :3] = np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]) @ matrix[:3, :3]
        if "translation" in node:
            matrix[:3, 3] = node["translation"]
        return matrix

    def walk(index: int, parent: np.ndarray) -> None:
        node = gltf["nodes"][index]
        here = parent @ local(node)
        if "mesh" in node:
            out[node["mesh"]] = here
        for child in node.get("children", []):
            walk(child, here)

    scene = gltf.get("scenes", [{}])[gltf.get("scene", 0)]
    for root in scene.get("nodes", range(len(gltf.get("nodes", [])))):
        walk(root, np.eye(4))
    return out


def _look_at(eye, target, up=(0, 1, 0)) -> np.ndarray:
    eye, target, up = (np.asarray(v, dtype=float) for v in (eye, target, up))
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-9:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    matrix = np.eye(4)
    matrix[:3, :3] = np.vstack([right, true_up, -forward])
    matrix[:3, 3] = -matrix[:3, :3] @ eye
    return matrix


def render(path: str | Path, *, eye, target, width=1280, height=720, fov=60.0,
           near=0.5, far=6000.0, sky=(24, 28, 34)) -> np.ndarray:
    from PIL import Image

    gltf, binary = read_glb(path)
    view = _look_at(eye, target)
    aspect = width / height
    focal = 1.0 / np.tan(np.radians(fov) / 2.0)

    colour = np.zeros((height, width, 3), dtype=np.float32)
    colour[:] = np.asarray(sky, dtype=np.float32) / 255.0
    depth = np.full((height, width), np.inf, dtype=np.float32)

    placement = node_transforms(gltf)
    for mesh_index, mesh in enumerate(gltf.get("meshes", [])):
        model = placement.get(mesh_index, np.eye(4))
        for primitive in mesh["primitives"]:
            position = accessor(gltf, binary, primitive["attributes"]["POSITION"])
            position = (model @ np.column_stack(
                [position, np.ones(len(position))]).T).T[:, :3]
            index = accessor(gltf, binary, primitive["indices"]).reshape(-1)
            material = gltf["materials"][primitive["material"]]
            pbr = material.get("pbrMetallicRoughness", {})

            texture = None
            uv = None
            if "baseColorTexture" in pbr and "TEXCOORD_0" in primitive["attributes"]:
                source = gltf["textures"][pbr["baseColorTexture"]["index"]]["source"]
                spec = gltf["images"][source]
                if "bufferView" in spec:
                    # Named `image_view`, not `view`: `view` is the camera
                    # matrix built once at the top of this function, and
                    # shadowing it here replaced the whole view transform with
                    # a bufferView dict the moment a texture was embedded in
                    # the buffer -- which is every .glb `gltf_textured` writes.
                    image_view = gltf["bufferViews"][spec["bufferView"]]
                    start = image_view.get("byteOffset", 0)
                    handle = io.BytesIO(
                        binary[start: start + image_view["byteLength"]])
                else:
                    # Themed exports keep their baked textures in a sidecar
                    # directory rather than in the buffer.
                    handle = Path(path).parent / unquote(spec["uri"])
                texture = np.asarray(Image.open(handle).convert("RGB"),
                                     dtype=np.float32) / 255.0
                uv = accessor(gltf, binary, primitive["attributes"]["TEXCOORD_0"])
            base = np.asarray(pbr.get("baseColorFactor", [0.7, 0.7, 0.7, 1])[:3],
                              dtype=np.float32)

            camera = (view @ np.column_stack(
                [position, np.ones(len(position))]).T).T[:, :3]
            # Behind the near plane there is no valid projection; dropping the
            # whole triangle is wrong at the frame edge but never wrong inside it.
            tri = index.reshape(-1, 3)
            visible = (camera[tri][:, :, 2] < -near).all(axis=1)
            tri = tri[visible]
            if not len(tri):
                continue

            w = -camera[:, 2]
            ndc = np.column_stack([focal / aspect * camera[:, 0] / w,
                                   focal * camera[:, 1] / w])
            screen = np.column_stack([(ndc[:, 0] + 1) * 0.5 * width,
                                      (1 - ndc[:, 1]) * 0.5 * height])

            for face in tri:
                p = screen[face]
                zw = w[face]
                lo = np.floor(p.min(axis=0)).astype(int)
                hi = np.ceil(p.max(axis=0)).astype(int)
                x0, y0 = max(lo[0], 0), max(lo[1], 0)
                x1, y1 = min(hi[0] + 1, width), min(hi[1] + 1, height)
                if x1 <= x0 or y1 <= y0:
                    continue
                area = ((p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1])
                        - (p[1, 1] - p[0, 1]) * (p[2, 0] - p[0, 0]))
                if abs(area) < 1e-9:
                    continue
                ys, xs = np.mgrid[y0:y1, x0:x1]
                px, py = xs + 0.5, ys + 0.5
                w0 = ((p[1, 0] - p[0, 0]) * (py - p[0, 1])
                      - (p[1, 1] - p[0, 1]) * (px - p[0, 0])) / area
                w1 = ((p[2, 0] - p[1, 0]) * (py - p[1, 1])
                      - (p[2, 1] - p[1, 1]) * (px - p[1, 0])) / area
                w2 = 1.0 - w0 - w1
                # w0 opposes vertex 2 and w1 opposes vertex 0 in this form.
                l0, l1, l2 = w1, w2, w0
                inside = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
                if not inside.any():
                    continue
                inv = l0 / zw[0] + l1 / zw[1] + l2 / zw[2]
                with np.errstate(divide="ignore", invalid="ignore"):
                    z = 1.0 / inv
                near_enough = inside & (z > near) & (z < far)
                if not near_enough.any():
                    continue
                sub = depth[y0:y1, x0:x1]
                win = near_enough & (z < sub)
                if not win.any():
                    continue
                if texture is not None:
                    tu = (l0 * uv[face[0], 0] / zw[0] + l1 * uv[face[1], 0] / zw[1]
                          + l2 * uv[face[2], 0] / zw[2]) * z
                    tv = (l0 * uv[face[0], 1] / zw[0] + l1 * uv[face[1], 1] / zw[1]
                          + l2 * uv[face[2], 1] / zw[2]) * z
                    th, tw = texture.shape[:2]
                    ix = np.clip((tu[win] * tw).astype(int), 0, tw - 1)
                    iy = np.clip((tv[win] * th).astype(int), 0, th - 1)
                    shade = texture[iy, ix]
                else:
                    shade = np.broadcast_to(base, (int(win.sum()), 3))
                sub[win] = z[win]
                colour[y0:y1, x0:x1][win] = shade

    return (np.clip(colour, 0, 1) * 255).astype(np.uint8)


def bounds(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """World-space extent, with node placement applied.

    Accessor min/max are in the mesh's own frame. Framing a camera on those
    while the node rotates and translates the mesh points the camera at empty
    space, which looks exactly like an empty world.
    """
    gltf, _ = read_glb(path)
    placement = node_transforms(gltf)
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for mesh_index, mesh in enumerate(gltf.get("meshes", [])):
        model = placement.get(mesh_index, np.eye(4))
        for primitive in mesh["primitives"]:
            spec = gltf["accessors"][primitive["attributes"]["POSITION"]]
            box = np.array([spec["min"], spec["max"]], dtype=float)
            corners = np.array(np.meshgrid(*box.T.tolist(), indexing="ij")
                               ).reshape(3, -1).T
            placed = (model @ np.column_stack(
                [corners, np.ones(len(corners))]).T).T[:, :3]
            lo = np.minimum(lo, placed.min(axis=0))
            hi = np.maximum(hi, placed.max(axis=0))
    return lo, hi


def main() -> int:
    from PIL import Image

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("glb")
    ap.add_argument("--out", default="build/shots")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    lo, hi = bounds(args.glb)
    centre = (lo + hi) / 2.0
    span = float(np.max(hi - lo))
    print(f"bounds {lo.round(1)} .. {hi.round(1)}  span {span:.0f} m")

    # glTF is Y-up: y is height, x/z are the ground plane.
    ground = float(lo[1])
    views = {
        "overview": (centre + np.array([span * 0.55, span * 0.5, span * 0.55]), centre),
        "street":   (np.array([centre[0] - span * 0.30, ground + 1.7,
                               centre[2] - span * 0.30]),
                     np.array([centre[0], ground + 12.0, centre[2]])),
        "roofline": (np.array([centre[0] - span * 0.35, ground + span * 0.28,
                               centre[2] - span * 0.35]),
                     np.array([centre[0], ground + 10.0, centre[2]])),
        "facade":   (np.array([centre[0] - span * 0.12, ground + 8.0,
                               centre[2] - span * 0.12]),
                     np.array([centre[0], ground + 14.0, centre[2]])),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, (eye, target) in views.items():
        pixels = render(args.glb, eye=eye, target=target,
                        width=args.width, height=args.height)
        path = out / f"{name}.png"
        Image.fromarray(pixels).save(path)
        sky = (pixels.reshape(-1, 3) == [24, 28, 34]).all(axis=1).mean()
        print(f"  {path}  {100*(1-sky):.1f}% covered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

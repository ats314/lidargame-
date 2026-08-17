"""The measured-appearance backend, checked by reading back what it wrote.

Every bug this file guards against has the same shape: the export succeeds, the
counts look right, and the render is wrong. So the assertions are on what a
consumer decodes, not on what the writer intended.
"""
from __future__ import annotations

import io
import json
import struct

import numpy as np
import pytest

from lidarworld.backends.gltf_textured import FALLBACK, Face, export


def read_back(path):
    raw = path.read_bytes()
    magic, version, length = struct.unpack_from("<III", raw, 0)
    assert magic == 0x46546C67
    assert version == 2
    assert length == len(raw)
    offset, gltf, binary = 12, None, None
    while offset < len(raw):
        size, kind = struct.unpack_from("<II", raw, offset)
        chunk = raw[offset + 8: offset + 8 + size]
        if kind == 0x4E4F534A:
            gltf = json.loads(chunk)
        elif kind == 0x004E4942:
            binary = chunk
        offset += 8 + size
    return gltf, binary


def wall(x=0.0):
    return np.array([[x, 0, 0], [x + 5, 0, 0], [x + 5, 0, 10], [x, 0, 10]],
                    dtype=float)


def jpeg_bytes():
    Image = pytest.importorskip("PIL.Image")
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 40, 40)).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_an_untextured_face_still_exports_with_a_class_colour(tmp_path):
    out = tmp_path / "plain.glb"
    result = export([Face(ring=wall(), kind="wall")], out)
    gltf, _ = read_back(out)
    assert result["triangles"] == 2
    assert result["textures"] == 0
    colour = gltf["materials"][0]["pbrMetallicRoughness"]["baseColorFactor"]
    assert colour == list(FALLBACK["wall"])


def test_coordinates_are_recentred_so_float32_can_hold_them(tmp_path):
    """A Hamburg easting is 566,000, where float32 resolves about 0.5 m.

    Exporting raw makes every wall jitter, which looks like bad reconstruction.
    The check is that the stored vertices are small and the origin carries the
    offset, so georeferencing is recoverable.
    """
    far = wall() + np.array([565_660.0, 5_934_160.0, 0.0])
    out = tmp_path / "far.glb"
    result = export([Face(ring=far, kind="wall")], out)
    gltf, _ = read_back(out)
    extent = np.array(gltf["accessors"][0]["max"]) - np.array(gltf["accessors"][0]["min"])
    assert np.all(np.abs(gltf["accessors"][0]["max"]) < 100)
    assert np.allclose(sorted(extent), [0.0, 5.0, 10.0])
    assert result["origin"][0] == pytest.approx(565_662.5, abs=1e-3)


def test_z_up_becomes_y_up(tmp_path):
    """A 10 m wall must be 10 m tall in glTF's y, not in its z."""
    out = tmp_path / "up.glb"
    export([Face(ring=wall(), kind="wall")], out, origin=np.zeros(3))
    gltf, _ = read_back(out)
    lo = np.array(gltf["accessors"][0]["min"])
    hi = np.array(gltf["accessors"][0]["max"])
    assert (hi - lo)[1] == pytest.approx(10.0)
    assert (hi - lo)[2] == pytest.approx(0.0)


def test_texture_v_is_flipped_for_gltf(tmp_path, monkeypatch):
    """CityGML puts (0,0) at the lower left and glTF at the upper left.

    Getting this wrong renders every facade upside down, which no count shows.
    """
    image = tmp_path / "t.jpg"
    image.write_bytes(jpeg_bytes())
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    out = tmp_path / "tex.glb"
    export([Face(ring=wall(), uv=uv, image="t.jpg", kind="wall")],
           out, image_root=tmp_path, origin=np.zeros(3))
    gltf, binary = read_back(out)
    spec = gltf["accessors"][gltf["meshes"][0]["primitives"][0]
                             ["attributes"]["TEXCOORD_0"]]
    view = gltf["bufferViews"][spec["bufferView"]]
    values = np.frombuffer(binary, dtype=np.float32, count=spec["count"] * 2,
                           offset=view["byteOffset"]).reshape(-1, 2)
    # The vertex that was at v=0 must come back at v=1.
    assert values[0].tolist() == [0.0, 1.0]
    assert values[2].tolist() == [1.0, 0.0]


def test_the_image_is_embedded_not_referenced(tmp_path):
    """A .glb that points at a neighbouring file is not portable to an engine."""
    image = tmp_path / "t.jpg"
    payload = jpeg_bytes()
    image.write_bytes(payload)
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    out = tmp_path / "tex.glb"
    result = export([Face(ring=wall(), uv=uv, image="t.jpg", kind="wall")],
                    out, image_root=tmp_path, origin=np.zeros(3))
    gltf, binary = read_back(out)
    assert result["textures"] == 1
    assert "uri" not in gltf["images"][0]
    assert gltf["images"][0]["mimeType"] == "image/jpeg"
    view = gltf["bufferViews"][gltf["images"][0]["bufferView"]]
    stored = binary[view["byteOffset"]: view["byteOffset"] + view["byteLength"]]
    assert stored == payload


def test_faces_sharing_an_image_share_one_primitive(tmp_path):
    """Draw calls should follow the texture count, not the surface count."""
    image = tmp_path / "t.jpg"
    image.write_bytes(jpeg_bytes())
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    faces = [Face(ring=wall(x), uv=uv, image="t.jpg", kind="wall")
             for x in (0.0, 20.0, 40.0)]
    out = tmp_path / "shared.glb"
    result = export(faces, out, image_root=tmp_path)
    assert result["primitives"] == 1
    assert result["textures"] == 1
    assert result["triangles"] == 6


def test_a_degenerate_face_is_counted_not_silently_dropped(tmp_path):
    out = tmp_path / "bad.glb"
    flat = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
    result = export([Face(ring=wall(), kind="wall"), Face(ring=flat, kind="wall")],
                    out)
    assert result["skipped"]["degenerate"] == 1
    assert result["triangles"] == 2

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


def test_uv1_is_metric_not_normalised(tmp_path):
    """A brick is 240 mm on a shed and on a warehouse.

    Normalising UV1 by wall width and height was the obvious choice and it is
    wrong: it makes one course span whatever the wall happens to be. The micro
    channel must come out in metres so the shader divides by a real repeat.
    """
    small = np.array([[0, 0, 0], [4, 0, 0], [4, 0, 3], [0, 0, 3]], dtype=float)
    big = np.array([[0, 20, 0], [40, 20, 0], [40, 20, 30], [0, 20, 30]], dtype=float)
    out = tmp_path / "metric.glb"
    export([Face(ring=small, kind="wall"), Face(ring=big, kind="wall")],
           out, origin=np.zeros(3))
    gltf, binary = read_back(out)
    spec = gltf["accessors"][gltf["meshes"][0]["primitives"][0]
                             ["attributes"]["TEXCOORD_1"]]
    span = np.array(spec["max"]) - np.array(spec["min"])
    # Both walls in one primitive: 40 m of width and 30 m of height total.
    assert span[0] == pytest.approx(40.0, abs=1e-3)
    assert span[1] == pytest.approx(30.0, abs=1e-3)


def test_uv1_v_axis_is_world_up_for_a_wall(tmp_path):
    """Masonry stacks toward the sky; if v is not up, brick courses run sideways."""
    wall_ring = np.array([[0, 0, 0], [6, 0, 0], [6, 0, 9], [0, 0, 9]], dtype=float)
    out = tmp_path / "up.glb"
    export([Face(ring=wall_ring, kind="wall")], out, origin=np.zeros(3))
    gltf, binary = read_back(out)
    spec = gltf["accessors"][gltf["meshes"][0]["primitives"][0]
                             ["attributes"]["TEXCOORD_1"]]
    view = gltf["bufferViews"][spec["bufferView"]]
    uv = np.frombuffer(binary, dtype=np.float32, count=spec["count"] * 2,
                       offset=view["byteOffset"]).reshape(-1, 2)
    # Vertices 0 and 2 differ by 9 m of height and 6 m of width.
    assert abs(uv[2, 1] - uv[0, 1]) == pytest.approx(9.0, abs=1e-3)
    assert abs(uv[1, 0] - uv[0, 0]) == pytest.approx(6.0, abs=1e-3)


def test_masonry_does_not_rotate_at_a_corner(tmp_path):
    """Two walls meeting at 90 degrees must both keep v pointing up.

    This is the triplanar failure the guidance calls out: a world-axis
    projection flips or rotates a directional material across a corner.
    """
    north = np.array([[0, 0, 0], [10, 0, 0], [10, 0, 8], [0, 0, 8]], dtype=float)
    east = np.array([[10, 0, 0], [10, 10, 0], [10, 10, 8], [10, 0, 8]], dtype=float)
    from lidarworld.backends.gltf_textured import wall_frame
    for ring in (north, east):
        u, v, _ = wall_frame(ring)
        assert v.tolist() == [0.0, 0.0, 1.0]
        assert abs(u[2]) < 1e-9              # u stays horizontal


def test_the_surface_index_locates_a_wall_in_three_dimensions(tmp_path):
    """A mask found in UV0 has to be mappable back onto a named wall."""
    out = tmp_path / "idx.glb"
    ring = np.array([[0, 0, 0], [6, 0, 0], [6, 0, 9], [0, 0, 9]], dtype=float)
    result = export([Face(ring=ring, kind="wall", surface_id="surf-1",
                          building_id="bldg-9")], out, origin=np.zeros(3))
    assert result["surfaces"] == 1
    index = json.loads((out.with_suffix(".surfaces.json")).read_text())
    entry = index["surfaces"][0]
    assert entry["surface_id"] == "surf-1"
    assert entry["building_id"] == "bldg-9"
    assert entry["width_m"] == pytest.approx(6.0, abs=1e-3)
    assert entry["height_m"] == pytest.approx(9.0, abs=1e-3)
    assert entry["v_axis"] == [0.0, 0.0, 1.0]


def _group(material, image, faces):
    from lidarworld.ingest.objmesh import Group
    return Group(material=material, image=image, faces=np.asarray(faces))


def test_a_triangulated_mesh_exports_without_going_through_ear_clipping(tmp_path):
    """A photogrammetric tile is millions of triangles that already are triangles."""
    from lidarworld.backends.gltf_textured import export_mesh
    positions = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], dtype=float)
    uvs = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    out = tmp_path / "mesh.glb"
    result = export_mesh(positions, uvs, [_group("m", None, [[0, 1, 2], [0, 2, 3]])],
                         out, origin=np.zeros(3))
    assert result["triangles"] == 2
    assert result["vertices"] == 4
    gltf, _ = read_back(out)
    assert len(gltf["meshes"][0]["primitives"]) == 1


def test_mesh_export_recentres_before_the_float32_cast(tmp_path):
    """Helsinki eastings are 25,497,363 - float32 there resolves about 2 m."""
    from lidarworld.backends.gltf_textured import export_mesh
    far = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1]], dtype=float) + \
        np.array([25_497_363.0, 6_672_958.0, 0.0])
    out = tmp_path / "far.glb"
    result = export_mesh(far, np.zeros((3, 2), np.float32),
                         [_group("m", None, [[0, 1, 2]])], out)
    gltf, _ = read_back(out)
    assert np.all(np.abs(gltf["accessors"][0]["max"]) < 10)
    assert result["origin"][0] == pytest.approx(25_497_363.5, abs=1e-2)


def test_mesh_export_shares_one_texture_between_groups_that_use_it(tmp_path):
    from lidarworld.backends.gltf_textured import export_mesh
    image = tmp_path / "t.jpg"
    image.write_bytes(jpeg_bytes())
    positions = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], dtype=float)
    uvs = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    groups = [_group("a", image, [[0, 1, 2]]), _group("b", image, [[0, 2, 3]])]
    result = export_mesh(positions, uvs, groups, tmp_path / "shared.glb",
                         origin=np.zeros(3))
    assert result["primitives"] == 2      # two material runs
    assert result["textures"] == 1        # but one embedded image


def test_mesh_export_flips_v_for_gltf(tmp_path):
    """OBJ's v runs up from the bottom-left; glTF's runs down from the top-left."""
    from lidarworld.backends.gltf_textured import export_mesh
    positions = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1]], dtype=float)
    uvs = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.float32)
    out = tmp_path / "flip.glb"
    export_mesh(positions, uvs, [_group("m", None, [[0, 1, 2]])], out,
                origin=np.zeros(3))
    gltf, binary = read_back(out)
    spec = gltf["accessors"][gltf["meshes"][0]["primitives"][0]
                             ["attributes"]["TEXCOORD_0"]]
    view = gltf["bufferViews"][spec["bufferView"]]
    values = np.frombuffer(binary, dtype=np.float32, count=spec["count"] * 2,
                           offset=view["byteOffset"]).reshape(-1, 2)
    assert values[0].tolist() == [0.0, 1.0]

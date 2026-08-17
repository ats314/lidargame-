"""Reading a photogrammetric city mesh out of OBJ.

The failure this file mostly guards against is the quiet one: an OBJ that reads
without error and comes out with the wrong corners welded, the wrong UVs, or
triangles silently dropped. None of that raises and all of it looks like a
rendering problem later.
"""
from __future__ import annotations

import numpy as np
import pytest

from lidarworld.ingest import objmesh

OBJ_SIMPLE = """\
mtllib m.mtl
v 0 0 0
v 1 0 0
v 1 0 1
v 0 0 1
vt 0 0
vt 1 0
vt 1 1
vt 0 1
usemtl brick
f 1/1 2/2 3/3
f 1/1 3/3 4/4
"""

MTL_SIMPLE = """\
newmtl brick
Kd 1 1 1
map_Kd wall.jpg
newmtl bare
Kd 0.5 0.5 0.5
"""


def write(tmp_path, obj=OBJ_SIMPLE, mtl=MTL_SIMPLE, image=True):
    (tmp_path / "m.mtl").write_text(mtl)
    if image:
        pytest.importorskip("PIL.Image")
        from PIL import Image
        Image.new("RGB", (8, 8), (200, 40, 40)).save(tmp_path / "wall.jpg")
    path = tmp_path / "m.obj"
    path.write_text(obj)
    return path


def test_a_textured_quad_reads_as_two_triangles(tmp_path):
    mesh = objmesh.read_obj(write(tmp_path))
    assert mesh.triangles == 2
    assert len(mesh.groups) == 1
    assert mesh.groups[0].material == "brick"
    assert mesh.groups[0].image is not None
    assert mesh.textured == 2


def test_a_material_without_a_map_has_no_image(tmp_path):
    obj = OBJ_SIMPLE.replace("usemtl brick", "usemtl bare")
    mesh = objmesh.read_obj(write(tmp_path, obj=obj))
    assert mesh.groups[0].image is None
    assert mesh.textured == 0


def test_a_missing_texture_file_is_none_not_a_dangling_path(tmp_path):
    """A path that does not resolve must not reach the exporter as if it did."""
    mesh = objmesh.read_obj(write(tmp_path, image=False))
    assert mesh.groups[0].image is None


def test_corners_sharing_a_position_but_not_a_uv_are_split(tmp_path):
    """OBJ indexes v and vt separately; glTF has one index buffer.

    Welding on position alone would give the second triangle the first's UVs,
    which renders as a smeared texture and never raises.
    """
    obj = """\
v 0 0 0
v 1 0 0
v 1 0 1
vt 0 0
vt 1 0
vt 1 1
vt 0.5 0.5
usemtl m
f 1/1 2/2 3/3
f 1/4 2/2 3/3
"""
    (tmp_path / "m.obj").write_text(obj)
    mesh = objmesh.read_obj(tmp_path / "m.obj")
    assert mesh.triangles == 2
    # Vertex 1 appears with two different UVs, so it must exist twice.
    assert len(mesh.positions) == 4
    at_origin = mesh.uvs[np.all(mesh.positions == 0, axis=1)]
    assert len(at_origin) == 2
    assert not np.allclose(at_origin[0], at_origin[1])


def test_a_polygon_face_is_fanned_into_triangles(tmp_path):
    obj = """\
v 0 0 0
v 1 0 0
v 1 0 1
v 0 0 1
usemtl m
f 1 2 3 4
"""
    (tmp_path / "m.obj").write_text(obj)
    assert objmesh.read_obj(tmp_path / "m.obj").triangles == 2


def test_negative_indices_count_back_from_what_has_been_read(tmp_path):
    obj = """\
v 0 0 0
v 1 0 0
v 1 0 1
usemtl m
f -3 -2 -1
"""
    (tmp_path / "m.obj").write_text(obj)
    mesh = objmesh.read_obj(tmp_path / "m.obj")
    assert mesh.triangles == 1
    assert len(mesh.positions) == 3


def test_merge_keeps_groups_separate_and_reindexes(tmp_path):
    """Two chunks merged must not have the second pointing at the first's vertices."""
    a = objmesh.read_obj(write(tmp_path))
    second = tmp_path / "two"
    second.mkdir()
    b = objmesh.read_obj(write(second))
    merged = objmesh.merge([a, b])
    assert merged.triangles == 4
    assert len(merged.groups) == 2
    assert len(merged.positions) == len(a.positions) + len(b.positions)
    assert merged.groups[1].faces.min() >= len(a.positions)
    assert merged.groups[1].faces.max() < len(merged.positions)


def test_crop_assigns_a_straddling_triangle_to_exactly_one_side(tmp_path):
    """Centroid membership, so a seam does not duplicate geometry."""
    obj = """\
v 0 0 0
v 2 0 0
v 2 0 1
v 8 0 0
v 10 0 0
v 10 0 1
usemtl m
f 1 2 3
f 4 5 6
"""
    (tmp_path / "m.obj").write_text(obj)
    mesh = objmesh.read_obj(tmp_path / "m.obj")
    left = objmesh.crop(mesh, [-1, -1], [5, 5])
    right = objmesh.crop(mesh, [5, -1], [11, 5])
    assert left.triangles == 1
    assert right.triangles == 1
    assert left.triangles + right.triangles == mesh.triangles


def test_an_empty_crop_is_empty_rather_than_everything(tmp_path):
    mesh = objmesh.read_obj(write(tmp_path))
    assert objmesh.crop(mesh, [100, 100], [200, 200]).triangles == 0


def test_bounds_are_local_and_reported_as_read(tmp_path):
    """Helsinki vertices are local; inventing a georeference would be worse."""
    mesh = objmesh.read_obj(write(tmp_path))
    lo, hi = mesh.bounds
    assert np.allclose(lo, [0, 0, 0])
    assert np.allclose(hi, [1, 0, 1])


def webbed(tmp_path):
    """Two ordinary triangles and one bridge thrown across a street."""
    obj = """\
v 0 0 0
v 1 0 0
v 1 0 1
v 0 0 1
v 60 0 0
v 60 0 30
usemtl m
f 1 2 3
f 1 3 4
f 1 5 6
"""
    (tmp_path / "m.obj").write_text(obj)
    return objmesh.read_obj(tmp_path / "m.obj")


def test_webbing_is_dropped_and_reported(tmp_path):
    """A bridge is textured like everything else, so nothing flags it but area.

    Measured on the Helsinki historic core: 3.5% of triangles carried 60% of all
    surface area, and in a render they are flat grey membranes over the street.
    """
    mesh = webbed(tmp_path)
    clean, report = objmesh.drop_webbing(mesh)
    assert mesh.triangles == 3
    assert clean.triangles == 2
    assert report["dropped_triangles"] == 1
    assert report["dropped_area_fraction"] > 0.9
    assert report["kept_area_m2"] == pytest.approx(1.0, abs=1e-6)


def test_dropping_webbing_leaves_a_hole_rather_than_a_wall(tmp_path):
    """The gap it bridged was a street. A hole is true; a membrane is not."""
    mesh = webbed(tmp_path)
    clean, _ = objmesh.drop_webbing(mesh)
    remaining = np.vstack([np.asarray(g.faces).reshape(-1, 3) for g in clean.groups])
    # No surviving triangle may reach the far side of the street.
    assert clean.positions[remaining].reshape(-1, 3)[:, 0].max() <= 1.0


def test_a_generous_threshold_keeps_everything(tmp_path):
    mesh = webbed(tmp_path)
    clean, report = objmesh.drop_webbing(mesh, max_area=10_000.0)
    assert clean.triangles == mesh.triangles
    assert report["dropped_triangles"] == 0
    assert report["dropped_area_fraction"] == 0.0

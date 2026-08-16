"""Adapters must round-trip and must normalise, not just parse."""
from __future__ import annotations

import numpy as np
import pytest

from lidarworld import ingest
from lidarworld.ingest.kitti import write_kitti, write_labels
from lidarworld.ingest.las import write_las, _read_native
from lidarworld.types import SEMANTIC_INDEX

S = SEMANTIC_INDEX


def test_las_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    xyz = rng.random((500, 3)) * np.array([40.0, 40.0, 12.0])
    intensity = rng.random(500).astype(np.float32)
    classification = rng.choice([2, 5, 6, 11], 500).astype(np.uint8)

    path = write_las(tmp_path / "t.las", xyz, intensity, classification)
    result = ingest.load(path)

    assert len(result.cloud) == 500
    # Written at millimetre scale, so positions survive to well under a cm.
    assert np.allclose(result.cloud.xyz, xyz, atol=2e-3)
    assert result.source.adapter == "las"
    assert result.source.point_count == 500

    semantic = result.cloud["semantic"]
    assert semantic[classification == 6].tolist() == [S["building"]] * int((classification == 6).sum())
    assert semantic[classification == 11].tolist() == [S["road"]] * int((classification == 11).sum())


def test_las_builtin_reader_matches_laspy(tmp_path):
    """The dependency-free reader must agree with laspy where laspy exists."""
    laspy = pytest.importorskip("laspy")
    rng = np.random.default_rng(1)
    xyz = rng.random((200, 3)) * 25
    path = write_las(tmp_path / "t.las", xyz, np.full(200, 0.5, np.float32),
                     np.full(200, 2, np.uint8))

    native_xyz, _, native_class, header = _read_native(path)
    with laspy.open(str(path)) as fh:
        las = fh.read()
    assert np.allclose(native_xyz[:, 0], np.asarray(las.x), atol=1e-6)
    assert np.array_equal(native_class, np.asarray(las.classification))
    assert header["point_format"] == 3


def test_las_drops_noise_class(tmp_path):
    xyz = np.zeros((10, 3))
    xyz[:, 0] = np.arange(10)
    classification = np.array([7, 2, 2, 2, 18, 2, 2, 2, 2, 2], np.uint8)
    path = write_las(tmp_path / "n.las", xyz, np.zeros(10, np.float32), classification)
    assert len(ingest.load(path).cloud) == 8
    assert len(ingest.load(path, keep_noise=True).cloud) == 10


def test_kitti_round_trip_with_labels(tmp_path):
    rng = np.random.default_rng(2)
    xyz = rng.normal(0, 10, (300, 3))
    xyz[:, 2] = rng.random(300) * 4 - 1.7
    intensity = rng.random(300).astype(np.float32)
    write_kitti(tmp_path / "s.bin", xyz, intensity)
    # 40 = road, 50 = building, 70 = vegetation in the SemanticKITTI vocabulary
    raw_ids = rng.choice([40, 50, 70], 300)
    write_labels(tmp_path / "s.label", raw_ids)

    result = ingest.load(tmp_path / "s.bin")
    cloud = result.cloud
    assert len(cloud) == 300
    assert cloud["semantic"][raw_ids == 50].tolist() == [S["building"]] * int((raw_ids == 50).sum())
    # Re-datumed so the ground sits near z=0 rather than the sensor.
    assert "datum_shift_z" in cloud.meta
    assert result.source.sensor_origin is not None


def test_kitti_rejects_wrong_stride(tmp_path):
    (tmp_path / "bad.bin").write_bytes(np.arange(7, dtype=np.float32).tobytes())
    with pytest.raises(ValueError, match="multiple of 4"):
        ingest.load(tmp_path / "bad.bin")


def test_pcd_ascii(tmp_path):
    rng = np.random.default_rng(3)
    xyz = rng.random((120, 3)) * 5
    lines = ["VERSION 0.7", "FIELDS x y z intensity", "SIZE 4 4 4 4", "TYPE F F F F",
             "COUNT 1 1 1 1", f"WIDTH {len(xyz)}", "HEIGHT 1", f"POINTS {len(xyz)}", "DATA ascii"]
    lines += [f"{x} {y} {z} 0.5" for x, y, z in xyz]
    path = tmp_path / "c.pcd"
    path.write_text("\n".join(lines) + "\n")

    cloud = ingest.load(path).cloud
    assert len(cloud) == 120
    assert np.allclose(cloud.xyz, xyz, atol=1e-5)
    assert np.allclose(cloud["intensity"], 0.5)


def test_pcd_binary(tmp_path):
    rng = np.random.default_rng(4)
    xyz = (rng.random((64, 3)) * 3).astype(np.float32)
    header = ("VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
              f"WIDTH {len(xyz)}\nHEIGHT 1\nPOINTS {len(xyz)}\nDATA binary\n")
    path = tmp_path / "b.pcd"
    path.write_bytes(header.encode() + xyz.tobytes())
    cloud = ingest.load(path).cloud
    assert np.allclose(cloud.xyz, xyz, atol=1e-5)


def test_ply_ascii(tmp_path):
    xyz = np.array([[0, 0, 0], [1, 2, 3], [4, 5, 6]], float)
    body = "\n".join(f"{x} {y} {z}" for x, y, z in xyz)
    (tmp_path / "m.ply").write_text(
        "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\n"
        f"property float z\nend_header\n{body}\n")
    assert np.allclose(ingest.load(tmp_path / "m.ply").cloud.xyz, xyz)


def test_xyz_with_class_column(tmp_path):
    (tmp_path / "p.xyz").write_text("0 0 0 0.1 2\n1 1 1 0.2 6\n2 2 2 0.3 11\n")
    cloud = ingest.load(tmp_path / "p.xyz").cloud
    assert len(cloud) == 3
    assert cloud["semantic"].tolist() == [S["ground"], S["building"], S["road"]]


def test_unknown_extension_lists_adapters(tmp_path):
    (tmp_path / "x.e57").write_bytes(b"nope")
    with pytest.raises(ValueError, match="no adapter handles"):
        ingest.load(tmp_path / "x.e57")


def test_every_adapter_is_registered():
    names = set(ingest.adapters())
    assert {"las", "kitti", "pcd", "ply", "xyz"} <= names

"""LAS / LAZ adapter -- the airborne path (USGS 3DEP, national mapping tiles).

Airborne tiles are the highest-leverage public source: they are already
semantically classified with ASPRS codes, they cover entire countries, and they
are in projected metres so a tile drops straight into a world.

``laspy`` is used when installed (and is required for LAZ). A minimal built-in
reader handles uncompressed LAS 1.0-1.4 point formats 0-3 and 6-8 so the core
package has no hard dependency beyond numpy.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from ..semantics.vocab import ASPRS
from ..types import PointCloud, Source
from .base import IngestResult, register, remap

_POINT_RECORD_LENGTHS = {0: 20, 1: 28, 2: 26, 3: 34, 4: 57, 5: 63, 6: 30, 7: 36, 8: 38, 9: 59, 10: 67}


def _read_native(path: Path):
    """Uncompressed-LAS reader. Returns (xyz, intensity, classification, header)."""
    raw = path.read_bytes()
    if raw[:4] != b"LASF":
        raise ValueError(f"{path} is not a LAS file (bad signature)")
    version = (raw[24], raw[25])
    header_size = struct.unpack_from("<H", raw, 94)[0]
    offset_to_data = struct.unpack_from("<I", raw, 96)[0]
    point_format = raw[104] & 0b00111111
    record_len = struct.unpack_from("<H", raw, 105)[0]
    legacy_count = struct.unpack_from("<I", raw, 107)[0]
    scale = np.array(struct.unpack_from("<3d", raw, 131))
    offset = np.array(struct.unpack_from("<3d", raw, 155))
    count = legacy_count
    if version >= (1, 4) and header_size >= 375:
        count = struct.unpack_from("<Q", raw, 247)[0] or legacy_count
    if point_format not in _POINT_RECORD_LENGTHS:
        raise ValueError(f"unsupported LAS point format {point_format}")

    body = np.frombuffer(raw, dtype=np.uint8, count=count * record_len, offset=offset_to_data)
    body = body.reshape(count, record_len)

    xyz_i = body[:, :12].copy().view(np.int32).reshape(count, 3)
    xyz = xyz_i.astype(np.float64) * scale + offset
    intensity = body[:, 12:14].copy().view(np.uint16).ravel().astype(np.float32) / 65535.0
    if point_format <= 5:
        classification = (body[:, 15] & 0b00011111).astype(np.uint8)
        # Byte 14 packs return number (bits 0-2) and return count (bits 3-5).
        returns = (body[:, 14] & 0b111, (body[:, 14] >> 3) & 0b111)
    else:                                    # 1.4 formats widened both fields
        classification = body[:, 16].astype(np.uint8)
        returns = (body[:, 14] & 0b1111, (body[:, 14] >> 4) & 0b1111)
    header = {"version": f"{version[0]}.{version[1]}", "point_format": point_format,
              "count": int(count), "scale": scale.tolist(), "offset": offset.tolist()}
    return xyz, intensity, classification, returns, header


@register("las", (".las", ".laz"), "LAS/LAZ airborne or terrestrial tile with ASPRS classes")
def load_las(path: Path, options: dict) -> IngestResult:
    crs = ""
    try:
        import laspy  # type: ignore

        with laspy.open(str(path)) as fh:
            las = fh.read()
        xyz = np.column_stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)]).astype(np.float64)
        intensity = np.asarray(las.intensity, dtype=np.float32) / 65535.0
        classification = np.asarray(las.classification, dtype=np.uint8)
        returns = (np.asarray(las.return_number, dtype=np.uint8),
                   np.asarray(las.number_of_returns, dtype=np.uint8))
        header = {"version": str(las.header.version), "point_format": las.header.point_format.id,
                  "count": int(las.header.point_count)}
        try:
            crs_obj = las.header.parse_crs()
            crs = crs_obj.to_string() if crs_obj else ""
        except Exception:
            crs = ""
        reader = "laspy"
    except ImportError:
        if path.suffix.lower() == ".laz":
            raise ImportError(
                "reading .laz needs laspy with a decompression backend: "
                "pip install 'laspy[lazrs]'") from None
        xyz, intensity, classification, returns, header = _read_native(path)
        reader = "builtin"

    keep_noise = options.get("keep_noise", False)
    semantic = remap(classification, ASPRS)
    # Return structure is the strongest vegetation evidence airborne LiDAR
    # carries: a pulse through a canopy comes back several times, a pulse off a
    # roof comes back once. Dropping it at the door throws that away.
    return_number, num_returns = (np.asarray(r, dtype=np.uint8) for r in returns)
    cloud = PointCloud(xyz, intensity=intensity, semantic=semantic,
                       source_class=classification.astype(np.uint8),
                       return_number=return_number, num_returns=num_returns)
    if not keep_noise:
        from ..types import SEMANTIC_INDEX
        mask = semantic != SEMANTIC_INDEX["noise"]
        if not mask.all():
            cloud = cloud.subset(mask)
    cloud.meta.update(header)
    cloud.meta["reader"] = reader

    labelled = float((cloud["semantic"] != 0).mean()) if len(cloud) else 0.0
    source = Source(
        id=options.get("source_id", path.stem),
        uri=str(path),
        license=options.get("license", "unknown -- check the tile's provider"),
        attribution=options.get("attribution", ""),
        sensor=options.get("sensor", "airborne lidar"),
        crs=options.get("crs", crs),
        notes=f"LAS {header.get('version')} pf{header.get('point_format')} via {reader}; "
              f"{labelled:.0%} of points carry a usable ASPRS class",
    )
    return IngestResult(cloud, source)


def write_las(path: Path, xyz: np.ndarray, intensity: np.ndarray, classification: np.ndarray,
              *, scale: float = 0.001) -> Path:
    """Minimal LAS 1.2 point-format-3 writer (used to bake sample tiles)."""
    path = Path(path)
    n = len(xyz)
    offset = xyz.min(axis=0)
    xyz_i = np.round((xyz - offset) / scale).astype(np.int32)
    header = bytearray(227)
    header[0:4] = b"LASF"
    struct.pack_into("<H", header, 24, 0)          # file source / global encoding
    header[24], header[25] = 1, 2                  # version 1.2
    header[26:58] = b"lidarworld".ljust(32, b"\0")
    header[58:90] = b"lidarworld sample baker".ljust(32, b"\0")
    struct.pack_into("<HH", header, 90, 1, 2026)
    struct.pack_into("<H", header, 94, 227)
    struct.pack_into("<I", header, 96, 227)
    struct.pack_into("<I", header, 100, 0)
    header[104] = 3
    struct.pack_into("<H", header, 105, _POINT_RECORD_LENGTHS[3])
    struct.pack_into("<I", header, 107, n)
    struct.pack_into("<3d", header, 131, scale, scale, scale)
    struct.pack_into("<3d", header, 155, *offset)
    hi, lo = xyz.max(axis=0), xyz.min(axis=0)
    struct.pack_into("<6d", header, 179, hi[0], lo[0], hi[1], lo[1], hi[2], lo[2])

    rec = np.zeros((n, _POINT_RECORD_LENGTHS[3]), dtype=np.uint8)
    rec[:, :12] = xyz_i.view(np.uint8).reshape(n, 12)
    inten = np.clip(intensity * 65535, 0, 65535).astype(np.uint16)
    rec[:, 12:14] = inten.view(np.uint8).reshape(n, 2)
    rec[:, 14] = 0b00001001                        # return 1 of 1 (bits 0-2, 3-5)
    rec[:, 15] = classification.astype(np.uint8)
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(rec.tobytes())
    return path

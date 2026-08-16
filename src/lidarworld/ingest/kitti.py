"""KITTI / SemanticKITTI adapter -- the street-level path.

A KITTI Velodyne scan is a raw float32 stream of (x, y, z, intensity) in the
sensor frame. SemanticKITTI ships a sidecar ``.label`` file with one uint32 per
point: the low 16 bits are the semantic id, the high 16 bits the instance id --
which is exactly the object grouping the segmentation stage would otherwise have
to guess, so it is carried straight through into the world graph.

Sensor-frame scans are z-up already, but the origin sits at the sensor, ~1.7 m
above the road. `lift` re-datums them so ground lands near z=0 like an airborne
tile, which is what makes both sources composable behind one interface.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

#: SemanticKITTI raw ids live in semantics.vocab alongside every other
#: benchmark's table; re-exported here for callers that expect them here.
from ..semantics.vocab import SEMANTIC_KITTI
from ..types import PointCloud, Source
from .base import IngestResult, register, remap

MOVING_IDS = frozenset(range(252, 260))


def _find_labels(path: Path, explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit)
    candidates = [
        path.with_suffix(".label"),
        path.parent.parent / "labels" / (path.stem + ".label"),
    ]
    return next((c for c in candidates if c.exists()), None)


@register("kitti", (".bin",), "KITTI Velodyne scan (float32 xyzi) + optional SemanticKITTI labels")
def load_kitti(path: Path, options: dict) -> IngestResult:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 4:
        raise ValueError(f"{path}: {raw.size} floats is not a multiple of 4 -- "
                         "not a KITTI xyzi scan?")
    pts = raw.reshape(-1, 4)
    xyz = pts[:, :3].astype(np.float64)
    intensity = pts[:, 3].astype(np.float32)

    n = len(xyz)
    semantic = np.zeros(n, dtype=np.uint8)
    instance = np.zeros(n, dtype=np.int32)
    moving = np.zeros(n, dtype=bool)
    notes = "unlabelled scan -- semantics will be inferred from geometry"

    label_path = _find_labels(path, options.get("labels"))
    if label_path and label_path.exists():
        labels = np.fromfile(label_path, dtype=np.uint32)
        if labels.size != n:
            raise ValueError(f"{label_path} has {labels.size} labels for {n} points")
        sem_ids = (labels & 0xFFFF).astype(np.int32)
        instance = (labels >> 16).astype(np.int32)
        semantic = remap(sem_ids, SEMANTIC_KITTI)
        moving = np.isin(sem_ids, list(MOVING_IDS))
        notes = f"SemanticKITTI labels from {label_path.name}; {moving.sum()} moving points"

    cloud = PointCloud(xyz, intensity=intensity, semantic=semantic)
    if label_path and label_path.exists():
        cloud["instance_hint"] = instance
        cloud["moving"] = moving

    # Re-datum: put the road at z~0 rather than the sensor at z=0.
    lift = options.get("lift", "auto")
    if lift == "auto":
        ground_mask = semantic == 1 if (semantic == 1).sum() > 100 else None
        z = xyz[ground_mask, 2] if ground_mask is not None else xyz[:, 2]
        dz = float(np.percentile(z, 5))
    else:
        dz = float(lift or 0.0)
    cloud.xyz[:, 2] -= dz
    cloud.meta["datum_shift_z"] = dz

    source = Source(
        id=options.get("source_id", path.stem),
        uri=str(path),
        license=options.get("license", "check the dataset terms (KITTI: CC BY-NC-SA 3.0)"),
        attribution=options.get("attribution", ""),
        sensor=options.get("sensor", "Velodyne HDL-64E"),
        crs="sensor-local (metres)",
        sensor_origin=[0.0, 0.0, -dz],
        notes=notes,
    )
    return IngestResult(cloud, source)


def write_kitti(path: Path, xyz: np.ndarray, intensity: np.ndarray) -> Path:
    """Write a KITTI-format scan (used by the sample baker and round-trip tests)."""
    arr = np.empty((len(xyz), 4), dtype=np.float32)
    arr[:, :3] = xyz
    arr[:, 3] = intensity
    arr.tofile(path)
    return Path(path)


def write_labels(path: Path, semantic_ids: np.ndarray, instance_ids: np.ndarray | None = None) -> Path:
    """Write a SemanticKITTI ``.label`` sidecar."""
    sem = np.asarray(semantic_ids, dtype=np.uint32) & 0xFFFF
    inst = np.zeros_like(sem) if instance_ids is None else np.asarray(instance_ids, dtype=np.uint32)
    ((inst << 16) | sem).astype(np.uint32).tofile(path)
    return Path(path)

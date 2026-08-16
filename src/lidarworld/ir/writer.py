"""Serialise a :class:`World` to a ``.lwir`` archive.

Layout (a plain zip -- ``unzip -l world.lwir`` works, and so does streaming a
single array out of it without parsing the rest):

    manifest.json     schema, CRS, bounds, sources, stage log, array index
    graph.json        nodes + edges
    points.json       per-point channel index (optional layer)
    arrays/<key>.bin  little-endian raw buffers, dtype+shape in the manifest

JSON for structure, raw binary for bulk. No custom container format to
reimplement in an engine backend, and every field is inspectable by hand.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np

from ..types import World

MAGIC = "lidarworld/spatial-ir"


def _array_entry(arr: np.ndarray) -> dict:
    if arr.dtype.byteorder == ">":
        arr = arr.astype(arr.dtype.newbyteorder("<"))
    return {
        "dtype": np.dtype(arr.dtype).str.replace(">", "<"),
        "shape": list(arr.shape),
        "bytes": int(arr.nbytes),
    }


def write_world(world: World, path: str | Path, *, compress: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = dict(world.arrays)
    points_index: dict | None = None
    if world.points is not None:
        pc = world.points
        arrays["points/xyz"] = pc.xyz.astype(np.float32)
        channels = {}
        for name, values in pc.channels.items():
            key = f"points/{name}"
            arrays[key] = np.ascontiguousarray(values)
            channels[name] = key
        points_index = {
            "count": len(pc),
            "xyz": "points/xyz",
            "channels": channels,
            "source_id": pc.source_id,
        }

    manifest = {
        "magic": MAGIC,
        "schema": world.schema,
        "name": world.name,
        "crs": world.crs,
        "up": world.up,
        "units": world.units,
        "created": world.created,
        "origin": [float(v) for v in world.origin],
        "bounds": [[float(v) for v in row] for row in world.bounds],
        "sources": [s.to_json() for s in world.sources],
        "stages": [s.to_json() for s in world.stages],
        "arrays": {k: _array_entry(v) for k, v in arrays.items()},
        "notes": world.notes,
        "summary": world.summary(),
    }
    graph = {
        "nodes": [n.to_json() for n in world.nodes.values()],
        "edges": [e.to_json() for e in world.edges],
    }

    mode = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    with zipfile.ZipFile(path, "w", mode) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=1))
        zf.writestr("graph.json", json.dumps(graph, separators=(",", ":")))
        if points_index is not None:
            zf.writestr("points.json", json.dumps(points_index, indent=1))
        for key, arr in arrays.items():
            arr = np.ascontiguousarray(arr)
            if arr.dtype.byteorder == ">":
                arr = arr.astype(arr.dtype.newbyteorder("<"))
            zf.writestr(f"arrays/{key}.bin", arr.tobytes())
    return path


def write_world_dir(world: World, path: str | Path) -> Path:
    """Same content, exploded into a directory. Handy for diffing in git."""
    path = Path(path)
    (path / "arrays").mkdir(parents=True, exist_ok=True)
    tmp = path / "_tmp.lwir"
    write_world(world, tmp, compress=False)
    with zipfile.ZipFile(tmp) as zf:
        zf.extractall(path)
    tmp.unlink()
    return path

"""Read a ``.lwir`` archive back into a :class:`World`.

Arrays are lazy by default: the manifest and graph load immediately, buffers are
pulled from the zip on first access. A backend that only needs terrain never
pays for the facade lattices.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np

from ..types import Edge, Node, PointCloud, Source, StageRecord, World
from .writer import MAGIC


class LazyArrays(dict):
    """dict-like view over arrays/*.bin inside the archive."""

    def __init__(self, path: Path, index: dict):
        super().__init__()
        self._path = path
        self._index = index
        self._cache: dict[str, np.ndarray] = {}

    def __contains__(self, key) -> bool:
        return key in self._index

    def __iter__(self):
        return iter(self._index)

    def __len__(self) -> int:
        return len(self._index)

    def keys(self):
        return self._index.keys()

    def items(self):
        for k in self._index:
            yield k, self[k]

    def __getitem__(self, key: str) -> np.ndarray:
        if key in self._cache:
            return self._cache[key]
        meta = self._index[key]
        with zipfile.ZipFile(self._path) as zf:
            raw = zf.read(f"arrays/{key}.bin")
        arr = np.frombuffer(raw, dtype=np.dtype(meta["dtype"])).reshape(meta["shape"])
        self._cache[key] = arr
        return arr

    def get(self, key, default=None):
        return self[key] if key in self._index else default


def read_world(path: str | Path, *, load_points: bool = True, eager: bool = False) -> World:
    path = Path(path)
    with zipfile.ZipFile(path) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        graph = json.loads(zf.read("graph.json"))
        points_index = json.loads(zf.read("points.json")) if "points.json" in zf.namelist() else None

    if manifest.get("magic") != MAGIC:
        raise ValueError(f"{path} is not a lidarworld Spatial IR archive")

    w = World(manifest["name"], manifest.get("crs", ""), manifest.get("up", "z"))
    w.schema = manifest["schema"]
    w.units = manifest.get("units", "m")
    w.created = manifest.get("created", "")
    w.origin = np.asarray(manifest.get("origin", [0, 0, 0]), dtype=np.float64)
    w.bounds = np.asarray(manifest.get("bounds", [[0, 0, 0], [0, 0, 0]]), dtype=np.float64)
    w.sources = [Source(**s) for s in manifest.get("sources", [])]
    w.stages = [StageRecord(**s) for s in manifest.get("stages", [])]
    w.notes = manifest.get("notes", {})

    for nd in graph["nodes"]:
        node = Node.from_json(nd)
        w.nodes[node.id] = node
    w.edges = [Edge.from_json(ed) for ed in graph["edges"]]

    lazy = LazyArrays(path, manifest.get("arrays", {}))
    w.arrays = dict(lazy.items()) if eager else lazy

    if load_points and points_index is not None:
        xyz = lazy[points_index["xyz"]].astype(np.float64)
        pc = PointCloud(xyz, source_id=points_index.get("source_id", "src0"))
        for name, key in points_index.get("channels", {}).items():
            pc[name] = lazy[key]
        w.points = pc
    return w


def inspect(path: str | Path) -> dict:
    """Cheap header read -- manifest only, no arrays touched."""
    with zipfile.ZipFile(Path(path)) as zf:
        return json.loads(zf.read("manifest.json"))

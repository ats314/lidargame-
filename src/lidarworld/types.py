"""Core data model shared by every stage of the compiler.

Two things live here:

* :class:`PointCloud` -- the working representation while the cloud is still a
  cloud (positions plus whatever per-point channels a stage has attached).
* :class:`World` -- the Spatial IR: a hierarchical graph of semantic objects
  with geometry, topology, confidence and provenance. It is deliberately
  theme-independent; nothing in here names a texture or a shader.

See docs/SPATIAL_IR.md for the on-disk schema.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable

import numpy as np

SCHEMA_VERSION = "0.2.0"

# Semantic classes. Deliberately small and source-agnostic: every ingest adapter
# maps its own vocabulary (ASPRS codes, SemanticKITTI ids) onto this set.
SEMANTIC_CLASSES = [
    "unclassified",   # 0
    "ground",         # 1
    "road",           # 2
    "building",       # 3
    "vegetation_low", # 4
    "vegetation_high",# 5
    "water",          # 6
    "vehicle",        # 7
    "pole",           # 8
    "wire",           # 9
    "fence",          # 10
    "bridge",         # 11
    "noise",          # 12
    "person",         # 13
]
SEMANTIC_INDEX = {name: i for i, name in enumerate(SEMANTIC_CLASSES)}


@dataclass
class Source:
    """Where a piece of the world came from, and under what terms."""
    id: str
    uri: str = ""
    adapter: str = ""
    license: str = "unknown"
    attribution: str = ""
    sensor: str = ""
    acquired: str = ""
    crs: str = ""
    point_count: int = 0
    #: Where the sensor stood, in the source's own frame. Forward validation
    #: needs this; adapters fill it in when the format implies it.
    sensor_origin: list | None = None
    notes: str = ""

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class StageRecord:
    """One compiler pass, recorded so a world can explain how it was built."""
    name: str
    version: str = "0"
    params: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0
    notes: str = ""

    def to_json(self) -> dict:
        return asdict(self)


class PointCloud:
    """Positions plus named per-point channels, all row-aligned.

    Channels are plain numpy arrays of length N. Stages add channels rather than
    replacing the cloud, so the provenance of every derived value stays visible:
    ``intensity``, ``semantic``, ``role``, ``hag`` (height above ground),
    ``planarity``, ``normal``, ``patch``, ``instance``, ``confidence``.
    """

    __slots__ = ("xyz", "channels", "source_id", "meta")

    def __init__(self, xyz: np.ndarray, *, source_id: str = "src0", **channels):
        xyz = np.asarray(xyz, dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"xyz must be (N,3), got {xyz.shape}")
        self.xyz = xyz
        self.channels: dict[str, np.ndarray] = {}
        self.source_id = source_id
        self.meta: dict[str, Any] = {}
        for k, v in channels.items():
            if v is not None:
                self[k] = v

    # -- container protocol -------------------------------------------------
    def __len__(self) -> int:
        return self.xyz.shape[0]

    def __contains__(self, key: str) -> bool:
        return key in self.channels

    def __getitem__(self, key: str) -> np.ndarray:
        return self.channels[key]

    def __setitem__(self, key: str, value: np.ndarray) -> None:
        value = np.asarray(value)
        if value.shape[0] != len(self):
            raise ValueError(
                f"channel {key!r} has {value.shape[0]} rows, cloud has {len(self)}")
        self.channels[key] = value

    def get(self, key: str, default=None):
        return self.channels.get(key, default)

    def require(self, *keys: str) -> None:
        missing = [k for k in keys if k not in self.channels]
        if missing:
            raise KeyError(f"cloud is missing required channels: {missing}. "
                           "Run the stage that produces them first.")

    # -- convenience --------------------------------------------------------
    @property
    def x(self) -> np.ndarray: return self.xyz[:, 0]

    @property
    def y(self) -> np.ndarray: return self.xyz[:, 1]

    @property
    def z(self) -> np.ndarray: return self.xyz[:, 2]

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.xyz.min(axis=0), self.xyz.max(axis=0)

    def subset(self, idx: np.ndarray) -> "PointCloud":
        out = PointCloud(self.xyz[idx], source_id=self.source_id)
        for k, v in self.channels.items():
            out[k] = v[idx]
        out.meta = dict(self.meta)
        return out

    def __repr__(self) -> str:
        lo, hi = self.bounds
        span = np.round(hi - lo, 1)
        return (f"PointCloud({len(self):,} pts, extent {span[0]}x{span[1]}x{span[2]} m, "
                f"channels={sorted(self.channels)})")


@dataclass
class Geometry:
    """Reference to the geometric payload of a node.

    ``kind`` picks the interpretation; array payloads are stored out-of-line in
    the IR and referenced by name so a backend can stream just what it needs.

    kind          arrays
    ------------- --------------------------------------------------------
    tiled_plane   occupancy(uint8), context(uint32) on a plane-local lattice
    heightfield   height(float32), mask(uint8) on a world-axis lattice
    mesh          positions(float32 N,3), indices(uint32 M,3), normals, uv
    polyline      positions(float32 N,3)
    instance      transform only (see attrs)
    voxels        ijk(int32 N,3)
    """
    kind: str
    arrays: dict[str, str] = field(default_factory=dict)
    #: plane/lattice frame: origin + two in-plane axes + cell size
    frame: dict[str, Any] = field(default_factory=dict)
    bounds: list[float] | None = None

    def to_json(self) -> dict:
        d = {"kind": self.kind, "arrays": self.arrays, "frame": self.frame}
        if self.bounds is not None:
            d["bounds"] = self.bounds
        return d

    @staticmethod
    def from_json(d: dict) -> "Geometry":
        return Geometry(d["kind"], d.get("arrays", {}), d.get("frame", {}), d.get("bounds"))


@dataclass
class Node:
    """A semantic object, surface, opening or instance in the world graph."""
    id: str
    role: str
    semantic: str = "unclassified"
    kind: str = "surface"
    parent: str | None = None
    children: list[str] = field(default_factory=list)
    confidence: float = 1.0
    support: int = 0                       # how many source points back this up
    sources: list[str] = field(default_factory=list)
    stage: str = ""
    geometry: Geometry | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        d = {
            "id": self.id, "role": self.role, "semantic": self.semantic,
            "kind": self.kind, "confidence": round(float(self.confidence), 4),
            "support": int(self.support),
        }
        if self.parent: d["parent"] = self.parent
        if self.children: d["children"] = self.children
        if self.sources: d["sources"] = self.sources
        if self.stage: d["stage"] = self.stage
        if self.geometry: d["geometry"] = self.geometry.to_json()
        if self.attrs: d["attrs"] = _jsonable(self.attrs)
        if self.tags: d["tags"] = self.tags
        return d

    @staticmethod
    def from_json(d: dict) -> "Node":
        g = d.get("geometry")
        return Node(
            id=d["id"], role=d["role"], semantic=d.get("semantic", "unclassified"),
            kind=d.get("kind", "surface"), parent=d.get("parent"),
            children=list(d.get("children", [])), confidence=d.get("confidence", 1.0),
            support=d.get("support", 0), sources=list(d.get("sources", [])),
            stage=d.get("stage", ""), geometry=Geometry.from_json(g) if g else None,
            attrs=d.get("attrs", {}), tags=list(d.get("tags", [])),
        )


@dataclass
class Edge:
    """A typed, weighted relation between two nodes."""
    a: str
    b: str
    relation: str
    confidence: float = 1.0
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict:
        d = {"a": self.a, "b": self.b, "rel": self.relation,
             "confidence": round(float(self.confidence), 4)}
        if self.attrs: d["attrs"] = _jsonable(self.attrs)
        return d

    @staticmethod
    def from_json(d: dict) -> "Edge":
        return Edge(d["a"], d["b"], d["rel"], d.get("confidence", 1.0), d.get("attrs", {}))


RELATIONS = (
    "contains", "adjacent_to", "coplanar_with", "perpendicular_to", "opening_in",
    "supports", "borders", "above", "connects", "parallel_to", "occludes",
)


class World:
    """The Spatial IR. Theme-independent, engine-independent, inspectable."""

    def __init__(self, name: str = "world", crs: str = "", up: str = "z"):
        self.schema = SCHEMA_VERSION
        self.name = name
        self.crs = crs
        self.up = up
        self.units = "m"
        self.created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.origin = np.zeros(3)          # world was shifted by this on ingest
        self.bounds = np.zeros((2, 3))
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self.sources: list[Source] = []
        #: Generative programs whose execution produced part of this world.
        #: The parameters, not the geometry -- see ir/program.py.
        self.programs: list[Any] = []
        self.stages: list[StageRecord] = []
        self.arrays: dict[str, np.ndarray] = {}
        self.points: PointCloud | None = None
        self.notes: dict[str, Any] = {}

    # -- graph --------------------------------------------------------------
    def add(self, node: Node) -> Node:
        if node.id in self.nodes:
            raise KeyError(f"duplicate node id {node.id!r}")
        self.nodes[node.id] = node
        if node.parent and node.parent in self.nodes:
            parent = self.nodes[node.parent]
            if node.id not in parent.children:
                parent.children.append(node.id)
        return node

    def link(self, a: str, b: str, relation: str, confidence: float = 1.0, **attrs) -> Edge:
        if relation not in RELATIONS:
            raise ValueError(f"unknown relation {relation!r}; expected one of {RELATIONS}")
        e = Edge(a, b, relation, confidence, attrs)
        self.edges.append(e)
        return e

    def put_array(self, key: str, arr: np.ndarray) -> str:
        self.arrays[key] = np.ascontiguousarray(arr)
        return key

    def by_role(self, pattern: str) -> list[Node]:
        from .roles.taxonomy import role_matches
        return [n for n in self.nodes.values() if role_matches(n.role, pattern)]

    def children_of(self, node_id: str) -> list[Node]:
        return [self.nodes[c] for c in self.nodes[node_id].children if c in self.nodes]

    def neighbors(self, node_id: str, relation: str | None = None) -> list[tuple[str, Edge]]:
        out = []
        for e in self.edges:
            if relation and e.relation != relation:
                continue
            if e.a == node_id: out.append((e.b, e))
            elif e.b == node_id: out.append((e.a, e))
        return out

    def stage(self, name: str, **params) -> "_StageTimer":
        return _StageTimer(self, name, params)

    def summary(self) -> dict[str, Any]:
        from collections import Counter
        roles = Counter(n.role for n in self.nodes.values())
        rels = Counter(e.relation for e in self.edges)
        return {
            "name": self.name, "schema": self.schema,
            "nodes": len(self.nodes), "edges": len(self.edges),
            "arrays": len(self.arrays),
            "points": len(self.points) if self.points is not None else 0,
            "roles": dict(roles.most_common()),
            "relations": dict(rels.most_common()),
            "bounds": self.bounds.tolist(),
        }

    def __repr__(self) -> str:
        return f"World({self.name!r}, {len(self.nodes)} nodes, {len(self.edges)} edges)"


class _StageTimer:
    def __init__(self, world: World, name: str, params: dict):
        self.world, self.name, self.params = world, name, params
        self.record = StageRecord(name=name, params=params)

    def __enter__(self) -> StageRecord:
        self._t0 = time.perf_counter()
        return self.record

    def __exit__(self, *exc) -> bool:
        self.record.seconds = round(time.perf_counter() - self._t0, 3)
        self.world.stages.append(self.record)
        return False


def _jsonable(obj: Any) -> Any:
    """numpy -> plain python, recursively, so dataclasses serialise cleanly."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, float)):
        return round(float(obj), 6)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj

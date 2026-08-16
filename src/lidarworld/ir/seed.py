"""World Seed: the smallest description a generator needs to rebuild the place.

The compiler's output is a faithful, ugly reconstruction -- 53 MB of tiles that
records every HVAC unit, every parapet wobble and every scan shadow the sensor
happened to produce. For a digital twin that is the point. For a world you walk
around in, most of it is noise the game does not want and the player will never
identify.

So this is deliberately lossy, in the direction of usefulness. It keeps the
things that make a place recognisably itself -- where the ground is, where the
streets run, where the buildings stand and how tall they are, where the trees
are -- and discards the measured surface entirely. What comes back out is not
the building that was scanned. It is *a* building on that footprint, at that
height, facing that street, which is what a game needs and what reconstruction
cannot give you anyway on airborne data that never saw the facade.

    reality -> semantic lossy compression -> seed -> generative expansion -> world

The compression number is the honest headline and it is measured, not claimed:
`extract()` reports the seed's size against the bundle it came from. Unlike an
image codec the decoder is generative, so the expansion does not have to
resemble the original in detail -- only in structure. Which means the same seed
expands into 1890 Denver or a neon one without re-measuring anything.

The seed carries no materials, no theme and no engine. That is the same
invariant the rest of the IR keeps: materialisation happens at the backend.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class WorldSeed:
    """Terrain, roads, buildings, vegetation, regions. Nothing else."""
    name: str
    crs: str = ""
    origin: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    bounds: list = field(default_factory=list)
    terrain: dict = field(default_factory=dict)
    roads: list = field(default_factory=list)
    buildings: list = field(default_factory=list)
    vegetation: list = field(default_factory=list)
    regions: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"seed": "lidarworld/0.1", "name": self.name, "crs": self.crs,
                "origin": self.origin, "bounds": self.bounds,
                "terrain": self.terrain, "roads": self.roads,
                "buildings": self.buildings, "vegetation": self.vegetation,
                "regions": self.regions, "provenance": self.provenance}

    @property
    def counts(self) -> dict:
        return {"buildings": len(self.buildings), "roads": len(self.roads),
                "trees": len(self.vegetation),
                "terrain_cells": int(np.prod(self.terrain.get("shape", [0, 0])))}


def _downsample(field_2d: np.ndarray, factor: int) -> np.ndarray:
    """Block-mean, so a coarse heightfield keeps the shape of the ground."""
    if factor <= 1:
        return field_2d
    nx = (field_2d.shape[0] // factor) * factor
    ny = (field_2d.shape[1] // factor) * factor
    trimmed = field_2d[:nx, :ny]
    return np.nanmean(trimmed.reshape(nx // factor, factor, ny // factor, factor),
                      axis=(1, 3))


def _simplify(ring: np.ndarray, tolerance: float) -> np.ndarray:
    """Ramer-Douglas-Peucker. A footprint is a shape, not a vertex list."""
    if len(ring) < 3:
        return ring
    start, end = ring[0], ring[-1]
    line = end - start
    length = float(np.hypot(line[0], line[1]))
    if length < 1e-9:
        distances = np.hypot(*(ring - start).T)
    else:
        normal = np.array([-line[1], line[0]]) / length
        distances = np.abs((ring - start) @ normal)
    index = int(np.argmax(distances))
    if distances[index] <= tolerance:
        return np.array([start, end])
    left = _simplify(ring[: index + 1], tolerance)
    right = _simplify(ring[index:], tolerance)
    return np.vstack([left[:-1], right])


def extract(world, *, terrain_step: int = 4, simplify: float = 0.5,
            bundle: str | Path | None = None) -> WorldSeed:
    """Reduce a compiled world to the description a generator can expand.

    `terrain_step` coarsens the height grid; 4 turns a 1 m DTM into 4 m, which
    is finer than any street gradient a player perceives. `simplify` is the
    footprint tolerance in metres -- half a metre keeps a building's shape and
    drops the survey's vertex noise.
    """
    seed = WorldSeed(name=world.name, crs=world.crs,
                     origin=[float(v) for v in world.origin],
                     bounds=np.round(world.bounds, 2).tolist())

    dtm = world.arrays.get("terrain/dtm")
    if dtm is not None:
        coarse = _downsample(np.asarray(dtm, dtype=float), terrain_step)
        seed.terrain = {
            "shape": list(coarse.shape), "step_m": terrain_step,
            "z": np.round(np.nan_to_num(coarse, nan=float(np.nanmin(coarse))), 2).tolist(),
        }

    classes = world.arrays.get("terrain/class")
    if classes is not None:
        coarse = np.asarray(classes)[::terrain_step, ::terrain_step]
        seed.regions = {"shape": list(coarse.shape), "step_m": terrain_step,
                        "legend": {"0": "ground", "1": "road", "2": "water",
                                   "255": "unobserved"},
                        "class": coarse.astype(int).tolist()}

    # Buildings come from the programs, because that is already the description
    # -- footprint and two heights. Nothing needs deriving from the mesh.
    for program in getattr(world, "programs", []):
        if program.kind != "extrude":
            continue
        ring = np.asarray(program.params["footprint"], dtype=float)
        ring = _simplify(ring, simplify)
        ground = float(program.params["ground_z"])
        eave = float(program.params["eave_z"])
        seed.buildings.append({
            "id": program.id,
            "footprint": np.round(ring, 2).tolist(),
            "ground_z": round(ground, 2),
            "height": round(eave - ground, 2),
            "roof": program.params.get("roof", "flat"),
            "residual": None if program.residual is None else round(program.residual, 3),
        })

    for node in world.nodes.values():
        if node.role != "volume.vegetation.high" or node.geometry is None:
            continue
        frame = node.geometry.frame or {}
        position = frame.get("position")
        if position is None:
            continue
        size = frame.get("size") or [2.0, 2.0, 6.0]
        seed.vegetation.append({
            "xy": [round(float(position[0]), 2), round(float(position[1]), 2)],
            "base_z": round(float(position[2]), 2),
            "crown_r": round(float(node.attrs.get("crown_radius", size[0])), 2),
            "height": round(float(node.attrs.get("canopy_height", size[2])), 2),
        })

    seed.roads = list(world.notes.get("road_network", []))
    seed.provenance = {
        "sources": [s.id for s in world.sources],
        "crs": world.crs,
        "note": "Lossy by design. Building facades, roof detail and surface "
                "texture are not described here and are not recoverable from "
                "it -- the generator invents them within these constraints.",
    }
    if bundle is not None:
        seed.provenance["compressed_from_bytes"] = _bundle_bytes(bundle)
    return seed


def _bundle_bytes(bundle: str | Path) -> int:
    path = Path(bundle)
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return path.stat().st_size if path.exists() else 0


def write(seed: WorldSeed, path: str | Path) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(seed.to_json(), separators=(",", ":")))
    size = path.stat().st_size
    info = {"path": str(path), "bytes": size, **seed.counts}
    source = seed.provenance.get("compressed_from_bytes")
    if source:
        info["compressed_from_bytes"] = source
        info["ratio"] = round(source / max(size, 1), 1)
    return info


def read(path: str | Path) -> WorldSeed:
    data = json.loads(Path(path).read_text())
    return WorldSeed(name=data.get("name", "world"), crs=data.get("crs", ""),
                     origin=data.get("origin", [0, 0, 0]),
                     bounds=data.get("bounds", []), terrain=data.get("terrain", {}),
                     roads=data.get("roads", []), buildings=data.get("buildings", []),
                     vegetation=data.get("vegetation", []),
                     regions=data.get("regions", {}),
                     provenance=data.get("provenance", {}))

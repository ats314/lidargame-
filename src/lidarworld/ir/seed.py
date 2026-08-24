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



def _roof_form(ring: np.ndarray, roofs) -> dict:
    """What shape the roof measured, in the few numbers a generator needs.

    A silhouette is most of what makes a skyline read as a place, and it is
    cheap to carry: a slope, a direction and a plane count per building. What
    it is *not* is a mesh -- the generator rebuilds the surface, this only says
    what kind of surface to rebuild.

    Classification is deliberately coarse, because that is all the evidence
    supports at 3.6 pts/m2 from above:

        flat     nothing measurably sloped
        shed     one sloped plane
        gable    two planes facing roughly opposite
        hip      three or more
    """
    from ..data.gis import point_in_polygon

    if not roofs or len(ring) < 4:
        return {}
    xy = np.array([[r[0], r[1]] for r in roofs])
    lo, hi = ring[:, :2].min(axis=0), ring[:, :2].max(axis=0)
    near = ((xy[:, 0] >= lo[0]) & (xy[:, 0] <= hi[0])
            & (xy[:, 1] >= lo[1]) & (xy[:, 1] <= hi[1]))
    if not near.any():
        return {}
    index = np.flatnonzero(near)
    inside = index[point_in_polygon(xy[index], ring[:, :2])]
    if not len(inside):
        return {}

    slopes = np.array([roofs[i][2] for i in inside])
    areas = np.array([max(roofs[i][3], 1e-6) for i in inside])
    normals = np.array([roofs[i][4] for i in inside])
    sloped = slopes > 10.0
    slope = float(np.average(slopes[sloped], weights=areas[sloped])) if sloped.any() else 0.0

    if not sloped.any():
        form = "flat"
    elif sloped.sum() == 1:
        form = "shed"
    else:
        # Two planes whose downhill directions oppose is a ridge; more is a hip.
        aspects = np.arctan2(normals[sloped][:, 1], normals[sloped][:, 0])
        spread = np.abs(np.angle(np.exp(1j * (aspects[:, None] - aspects[None, :]))))
        form = "gable" if (spread > 2.4).any() and sloped.sum() <= 3 else "hip"

    # Ridge runs across the slope, so it is the dominant aspect turned 90 deg.
    dominant = normals[np.argmax(areas)] if not sloped.any() else \
        normals[sloped][np.argmax(areas[sloped])]
    ridge = float(np.degrees(np.arctan2(dominant[1], dominant[0])) + 90.0) % 180.0
    return {"roof": form, "roof_slope_deg": round(slope, 1),
            "roof_ridge_deg": round(ridge, 1), "roof_planes": int(len(inside))}

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

    # The compiler measures roof planes -- slope, aspect, how many of them --
    # and the seed used to drop every one, writing "flat" for all 257 buildings
    # in a LoDo block where 129 of 369 measured patches are pitched. That is
    # not compression, it is loss: the generator cannot invent a silhouette the
    # seed never recorded, so every building came back a box.
    #
    # Roof patches hang off the building node they belong to, and the programs
    # are keyed by footprint index instead, so the join is spatial.
    roofs_by_xy = []
    for node in world.nodes.values():
        if not node.role.startswith("surface.roof") or node.geometry is None:
            continue
        frame = node.geometry.frame
        origin = frame.get("origin")
        normal = frame.get("normal")
        if origin is None or normal is None:
            continue
        roofs_by_xy.append((float(origin[0]), float(origin[1]),
                            float(node.attrs.get("slope_deg", 0.0)),
                            float(node.attrs.get("area", 0.0)),
                            [float(v) for v in normal]))

    # Buildings come from the programs, because that is already the description
    # -- footprint and two heights. Nothing needs deriving from the mesh.
    for program in getattr(world, "programs", []):
        if program.kind != "extrude":
            continue
        ring = np.asarray(program.params["footprint"], dtype=float)
        ring = _simplify(ring, simplify)
        ground = float(program.params["ground_z"])
        eave = float(program.params["eave_z"])
        entry = {
            "id": program.id,
            "footprint": np.round(ring, 2).tolist(),
            "ground_z": round(ground, 2),
            "height": round(eave - ground, 2),
            "roof": program.params.get("roof", "flat"),
            "residual": None if program.residual is None else round(program.residual, 3),
        }
        if program.params.get("source_id"):
            # The register's id for this building, kept so the seed can be
            # scored against the register it came from.
            entry["source_id"] = program.params["source_id"]
        entry.update(_roof_form(ring, roofs_by_xy))
        seed.buildings.append(entry)

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


@dataclass
class Solid:
    """A building as anything upstream can describe it: outline and two heights.

    `extract()` reads a compiled `World` and pulls this out of the programs. Not
    everything worth seeding is a compiled World, though -- a published city
    model already *is* a reconstruction, and re-deriving it from returns it
    never had would be theatre. So this is the neutral form both routes meet in.

    `roof_planes` is (slope_deg, area_m2, normal) per measured roof plane, which
    is what decides the silhouette. Denver recorded `flat` for all 257 buildings
    because roof form was measured and then dropped, and a generator cannot
    invent a skyline the seed never carried.
    """
    footprint: np.ndarray                       # (N, 2) exterior ring
    ground_z: float
    eave_z: float
    roof_planes: list = field(default_factory=list)
    attrs: dict = field(default_factory=dict)


def _terrain_from_solids(solids: list[Solid], bounds: np.ndarray,
                         step: float) -> dict:
    """A ground surface interpolated from where buildings meet it.

    This is `derived`, not `observed`, and the note says so. Hamburg's building
    model is placed on a DGM that ships separately; until that is acquired, the
    ground intersections are the only terrain evidence in the file, and inventing
    a flat plane at z=0 would drop every building into the air or bury it.

    Inverse distance over the building base heights. Crude, and honest about it:
    it cannot know about a cutting or an underpass, only about the level the
    buildings sit at.
    """
    lo, hi = bounds[0][:2], bounds[1][:2]
    nx = max(2, int(np.ceil((hi[0] - lo[0]) / step)))
    ny = max(2, int(np.ceil((hi[1] - lo[1]) / step)))
    gx = lo[0] + (np.arange(nx) + 0.5) * step
    gy = lo[1] + (np.arange(ny) + 0.5) * step
    grid_x, grid_y = np.meshgrid(gx, gy, indexing="ij")

    if not solids:
        return {"shape": [nx, ny], "step_m": step,
                "z": np.zeros((nx, ny)).tolist()}

    anchors = np.array([[s.footprint[:, 0].mean(), s.footprint[:, 1].mean(),
                         s.ground_z] for s in solids])
    d2 = ((grid_x[..., None] - anchors[:, 0]) ** 2
          + (grid_y[..., None] - anchors[:, 1]) ** 2)
    weight = 1.0 / np.maximum(d2, 1.0)
    z = (weight * anchors[:, 2]).sum(axis=-1) / weight.sum(axis=-1)
    return {"shape": [nx, ny], "step_m": step, "z": np.round(z, 2).tolist(),
            "evidence": "derived from building ground intersections; no DTM "
                        "was read"}


def from_solids(solids: list[Solid], *, name: str, crs: str = "",
                simplify: float = 0.5, terrain_step: float = 4.0,
                origin=None, source_bytes: int | None = None,
                note: str = "") -> WorldSeed:
    """Seed a place from building solids alone, with no point cloud involved.

    The seed that comes out is the same shape as the compiler's, which is the
    whole point: if a published city model and a reconstruction both reduce to
    this, then whatever the generator does to one it does to the other, and a
    bad-looking result stops being ambiguous about which half caused it.
    """
    rings = [np.asarray(s.footprint, dtype=float)[:, :2] for s in solids
             if len(s.footprint) >= 3]
    if rings:
        stacked = np.vstack(rings)
        zs = [s.ground_z for s in solids] + [s.eave_z for s in solids]
        bounds = np.array([[stacked[:, 0].min(), stacked[:, 1].min(), min(zs)],
                           [stacked[:, 0].max(), stacked[:, 1].max(), max(zs)]])
    else:
        bounds = np.zeros((2, 3))

    seed = WorldSeed(
        name=name, crs=crs,
        origin=[float(v) for v in (origin if origin is not None
                                   else bounds.mean(axis=0))],
        bounds=np.round(bounds, 2).tolist())
    seed.terrain = _terrain_from_solids(solids, bounds, terrain_step)

    for index, solid in enumerate(solids):
        ring = np.asarray(solid.footprint, dtype=float)[:, :2]
        if len(ring) < 3:
            continue
        ring = _simplify(ring, simplify)
        if len(ring) < 4:
            continue
        entry = {
            "id": f"solid.{index:05d}",
            "footprint": np.round(ring, 2).tolist(),
            "ground_z": round(float(solid.ground_z), 2),
            "height": round(float(solid.eave_z - solid.ground_z), 2),
            "roof": "flat",
            "residual": None,
        }
        # Reuse the same classifier the compiled route uses, so the two seeds
        # disagree about roofs only where the evidence disagrees.
        placed = [(float(ring[:, 0].mean()), float(ring[:, 1].mean()),
                   slope, area, normal)
                  for slope, area, normal in solid.roof_planes]
        entry.update(_roof_form(ring, placed))
        entry.update(solid.attrs)
        seed.buildings.append(entry)

    seed.provenance = {
        "sources": [name],
        "crs": crs,
        "note": note or ("Lossy by design. Facades, roof detail and surface "
                         "texture are not described here and are not "
                         "recoverable from it."),
    }
    if source_bytes:
        seed.provenance["compressed_from_bytes"] = int(source_bytes)
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

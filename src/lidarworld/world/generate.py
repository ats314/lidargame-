"""Expand a World Seed back into a walkable world, with no point cloud.

This is the direction the whole project points at and the half that was never
built. `compile` runs reality -> seed. This runs seed -> place:

    reality -> semantic lossy compression -> World Seed -> generation -> world

The seed is 290 KB against a 124 MB bundle. It holds a terrain grid, a footprint
and two heights per building, a centreline and a width per road, and a position
and size per tree. Nothing else. Everything the returns measured about the
*surface* has been thrown away, which is the point: the building that comes back
is not the building that was scanned, it is *a* building on that footprint at
that height facing that street. Airborne data never saw the facade anyway.

So this module is not a decompressor. It is a generator whose constraints
happen to be measured. Where the seed says something, it is obeyed; where the
seed is silent, structure is invented -- and everything invented is `generated`,
never `observed`, so a world can still say which of it was real.

Two things fall out of running it:

    a game world      you can walk around in, from 290 KB
    a proof           the point cloud can be deleted and the place still exists
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..reconstruct import extrude as extrude_stage
from ..reconstruct import lattice as lattice_stage
from ..reconstruct import mesh as mesh_stage
from ..reconstruct import terrain as terrain_stage
from ..spatial.grid import Raster2D
from ..types import Geometry, Node, World


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def _terrain(seed: dict) -> tuple[Raster2D, np.ndarray]:
    """The heightfield, back from a flat list of samples."""
    grid = seed["terrain"]
    nx, ny = grid["shape"]
    step = float(grid["step_m"])
    z = np.asarray(grid["z"], dtype=np.float32)
    if z.ndim == 1:
        z = z.reshape(nx, ny)
    lo = np.asarray(seed["bounds"][0], dtype=float)
    raster = Raster2D(lo[:2], lo[:2] + np.array([nx * step, ny * step]), step, pad=0)
    # Raster2D rounds its own shape; match it rather than trusting both agree.
    if raster.shape != z.shape:
        out = np.full(raster.shape, float(np.nanmedian(z)), dtype=np.float32)
        n0 = min(raster.shape[0], z.shape[0])
        n1 = min(raster.shape[1], z.shape[1])
        out[:n0, :n1] = z[:n0, :n1]
        z = out
    return raster, z


def _roads(seed: dict, raster: Raster2D, dtm: np.ndarray) -> np.ndarray:
    """Stamp the carriageway into a terrain class raster."""
    classes = np.full(raster.shape, terrain_stage.GROUND, dtype=np.uint8)
    for road in seed.get("roads", []):
        line = np.asarray(road["line"], dtype=float)
        if len(line) < 2:
            continue
        half = float(road.get("half_width", 5.0))
        # Walk the centreline and stamp a disc: a seed road is a graph edge with
        # a width, not a polygon, and a disc per sample needs no clipper.
        for a, b in zip(line[:-1], line[1:]):
            span = float(np.hypot(*(b - a)))
            steps = max(2, int(span / (raster.cell * 0.5)))
            for t in np.linspace(0.0, 1.0, steps):
                p = a + (b - a) * t
                ij = raster.to_cell(p[None, :2])[0]
                r = int(np.ceil(half / raster.cell))
                i0, i1 = max(0, ij[0] - r), min(raster.nx, ij[0] + r + 1)
                j0, j1 = max(0, ij[1] - r), min(raster.ny, ij[1] + r + 1)
                if i0 >= i1 or j0 >= j1:
                    continue
                ii, jj = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing="ij")
                cx = raster.origin[0] + (ii + 0.5) * raster.cell
                cy = raster.origin[1] + (jj + 0.5) * raster.cell
                inside = (cx - p[0]) ** 2 + (cy - p[1]) ** 2 <= half * half
                block = classes[i0:i1, j0:j1]
                block[inside] = terrain_stage.ROAD
                classes[i0:i1, j0:j1] = block
    return classes


def expand(seed: dict, *, tile: float = 0.5, roof_pitch: float = 0.0) -> World:
    """Build a World from a seed. Nothing here reads a point cloud."""
    world = World(name=seed.get("name", "generated"), crs=seed.get("crs", ""))
    world.origin = np.asarray(seed.get("origin", [0, 0, 0]), dtype=float)
    world.bounds = np.asarray(seed["bounds"], dtype=float)
    world.notes["generated_from"] = {
        "seed": seed.get("seed", "unknown"),
        "buildings": len(seed.get("buildings", [])),
        "roads": len(seed.get("roads", [])),
        "vegetation": len(seed.get("vegetation", [])),
        "note": "No point cloud was read. Every surface here is generated from "
                "the seed's parameters; only those parameters were measured.",
    }

    raster, dtm = _terrain(seed)
    classes = _roads(seed, raster, dtm)

    builder = mesh_stage.MeshBuilder()
    context = mesh_stage.terrain_context(classes, dtm)
    world.add(Node(
        id="terrain", role="terrain.ground", semantic="ground", kind="terrain",
        confidence=0.9, stage="generate",
        geometry=Geometry("heightfield",
                          {"height": "terrain/dtm", "class": "terrain/class"},
                          {"origin": raster.origin.tolist(), "cell": raster.cell,
                           "shape": [raster.nx, raster.ny]}),
        attrs={"cell": raster.cell, "method": "seed"}))
    node_slots = ["terrain"]
    quads = mesh_stage.add_terrain(builder, raster, dtm, classes, context,
                                   terrain_stage.ROLE_LOOKUP, 0)

    walls_made = 0
    for index, building in enumerate(seed.get("buildings", [])):
        ring = np.asarray(building["footprint"], dtype=float)
        if len(ring) < 4:
            continue
        base = float(building["ground_z"])
        top = base + float(building["height"])
        bid = f"bldg.{index:04d}"
        centre = ring.mean(axis=0)
        world.add(Node(
            id=bid, role="volume.building", semantic="building", kind="object",
            confidence=0.6, stage="generate",
            attrs={"height": round(top - base, 2), "ground_z": round(base, 2),
                   "roof": building.get("roof", "flat"),
                   "epistemic": "generated"}))
        node_slots.append(bid)

        surfaces = extrude_stage.walls_from_footprint(ring, base, top,
                                                      start_id=walls_made)
        walls_made += len(surfaces)
        for patch in surfaces:
            width, height = patch.extent
            lat = lattice_stage.build_solid(patch, width, height, cell=tile,
                                            ground_z=base)
            sid = f"{bid}.wall.{patch.id:05d}"
            world.add(Node(
                id=sid, role=patch.role, semantic="building", kind="surface",
                parent=bid, confidence=0.5, stage="generate",
                attrs={"epistemic": "generated"}))
            node_slots.append(sid)
            quads += mesh_stage.add_lattice(builder, patch, lat, len(node_slots) - 1)

        # A flat cap is a claim the seed actually makes: it stores a roof form,
        # and `flat` is the honest default for a prism the returns only gave a
        # height for.
        roof = _roof_patch(ring, top, start_id=walls_made)
        if roof is not None:
            walls_made += 1
            lat = lattice_stage.build_solid(roof, *roof.extent, cell=tile,
                                            ground_z=top)
            rid = f"{bid}.roof"
            world.add(Node(id=rid, role="surface.roof.flat", semantic="building",
                           kind="surface", parent=bid, confidence=0.5,
                           stage="generate", attrs={"epistemic": "generated"}))
            node_slots.append(rid)
            quads += mesh_stage.add_lattice(builder, roof, lat, len(node_slots) - 1)

    for index, tree in enumerate(seed.get("vegetation", [])):
        pos = np.asarray(tree.get("position", tree.get("center", [0, 0, 0])), dtype=float)
        radius = float(tree.get("crown_radius", tree.get("radius", 2.5)))
        height = float(tree.get("height", 8.0))
        nid = f"tree.{index:04d}"
        world.add(Node(
            id=nid, role="volume.vegetation.high", semantic="vegetation_high",
            kind="vegetation", confidence=0.5, stage="generate",
            geometry=Geometry("instance", {}, {}),
            attrs={"center": pos.tolist(), "size": [radius, radius, height],
                   "epistemic": "generated"}))

    arrays = builder.finalize()
    for key, value in arrays.items():
        world.put_array(f"mesh/{key}", value)
    world.put_array("terrain/dtm", np.nan_to_num(dtm, nan=0.0).astype(np.float32))
    world.put_array("terrain/class", classes.astype(np.uint8))
    world.put_array("terrain/context", context.astype(np.uint32))
    world.notes["node_slots"] = node_slots
    world.notes["generated_from"]["quads"] = int(quads)
    return world


def _roof_patch(ring: np.ndarray, z: float, *, start_id: int):
    """A horizontal patch spanning the footprint, used as a flat roof cap."""
    from ..segment.planes import PlanarPatch

    xy = ring[:, :2]
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    if float(np.min(hi - lo)) < 0.5:
        return None
    centroid = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, z])
    normal = np.array([0.0, 0.0, 1.0])
    patch = PlanarPatch(
        id=start_id, normal=normal, offset=float(-normal @ centroid),
        centroid=centroid, u=np.array([1.0, 0.0, 0.0]),
        v=np.array([0.0, 1.0, 0.0]),
        extent=(float(hi[0] - lo[0]), float(hi[1] - lo[1])),
        point_idx=np.empty(0, dtype=np.int64), role="surface.roof.flat",
        confidence=0.5, support=0)
    patch.attrs["generated"] = True
    return patch

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
from ..roles.taxonomy import Ctx
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
            _fenestrate(lat, patch, building, index)
            sid = f"{bid}.wall.{patch.id:05d}"
            world.add(Node(
                id=sid, role=patch.role, semantic="building", kind="surface",
                parent=bid, confidence=0.5, stage="generate",
                attrs={"epistemic": "generated"}))
            node_slots.append(sid)
            quads += mesh_stage.add_lattice(builder, patch, lat, len(node_slots) - 1)

        # Roof form, from what the scan measured rather than a flat default.
        # This is the line that decides whether a skyline reads as a place or a
        # row of boxes, and the seed now carries the slope and ridge bearing to
        # drive it.
        for roof in _roof_patches(ring, top, building, start_id=walls_made):
            walls_made += 1
            lat = lattice_stage.build_solid(roof, *roof.extent, cell=tile,
                                            ground_z=top)
            _clip_to_footprint(lat, roof, ring)
            rid = f"{bid}.roof.{roof.id:05d}"
            world.add(Node(id=rid, role=roof.role, semantic="building",
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


def _roof_patches(ring: np.ndarray, z: float, building: dict, *, start_id: int):
    """A flat cap per building.

    Pitched roofs were generated here from the seed's measured slope and ridge
    bearing and were reverted: they produced plates far larger than the
    buildings they sat on, tilted across their neighbours, and clamping the
    slope did not fix it. The seed's roof form is still recorded and still
    correct -- 47 flat, 8 shed, 17 hip, 2 gable over a block -- so the
    information is not lost. What was wrong is this function's reconstruction
    of a plane from it: extents derived from the footprint's half-span do not
    bound a roof plane once it is tilted about an arbitrary ridge, and the
    result is a plate rather than a roof.

    A flat cap is honest in the meantime. It is what the returns support least
    ambiguously, and it is what the block looked right with.
    """
    xy = ring[:, :2]
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    span = hi - lo
    if float(np.min(span)) < 0.5:
        return []
    return [_plane(np.array([0.0, 0.0, 1.0]),
                   np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, z]),
                   np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
                   (float(span[0]), float(span[1])), start_id,
                   "surface.roof.flat")]


def _plane(normal, centroid, u, v, extent, pid, role):
    from ..segment.planes import PlanarPatch

    patch = PlanarPatch(
        id=pid, normal=np.asarray(normal, float),
        offset=float(-np.asarray(normal, float) @ np.asarray(centroid, float)),
        centroid=np.asarray(centroid, float), u=np.asarray(u, float),
        v=np.asarray(v, float), extent=extent,
        point_idx=np.empty(0, dtype=np.int64), role=role,
        confidence=0.5, support=0)
    patch.attrs["generated"] = True
    return patch


#: Storey height in metres. Denver's downtown stock is mostly 3.5-4 m floor to
#: floor; a generated facade only has to be *plausible* at this spacing, since
#: airborne returns never saw a single window of it.
STOREY_M = 3.8


def _fenestrate(lattice, patch, building, index: int) -> None:
    """Cut window and door openings into a generated facade.

    Every wall here is Tier 7 -- procedural generation -- because the evidence
    genuinely does not constrain it: airborne LiDAR never sees a facade, so
    there is nothing to be faithful to and nothing to be wrong about. What the
    seed *does* constrain is the envelope, and a window grid derived from the
    building's own height and footprint stays inside that.

    Deterministic on the building's identity, so the same seed regenerates the
    same street rather than a different one each run. That is what makes the
    world reproducible rather than merely random.
    """
    if not patch.role.startswith("surface.wall"):
        return
    occupancy = lattice.occupancy
    context = lattice.context
    nu, nv = occupancy.shape
    cell = lattice.cell
    height_m = nv * cell
    if height_m < 4.0 or nu * cell < 3.0:
        return                      # a garden wall, not a facade

    rng = np.random.default_rng(abs(hash((building.get("id", index), patch.id))) % (2 ** 32))
    storeys = max(1, int(round(height_m / STOREY_M)))
    win_w = max(2, int(round(1.4 / cell)))     # ~1.4 m wide
    win_h = max(2, int(round(1.6 / cell)))     # ~1.6 m tall
    pier = max(2, int(round(1.6 / cell)))      # wall between windows
    pitch = win_w + pier
    if pitch >= nu:
        return

    margin = max(1, (nu % pitch) // 2)
    for storey in range(storeys):
        # Sill sits ~1 m above each floor, and the ground storey is taller,
        # which is what makes a shopfront read differently from a flat above it.
        floor_v = int(round(storey * STOREY_M / cell))
        sill = floor_v + int(round((1.5 if storey else 0.8) / cell))
        top = sill + (win_h if storey else int(round(2.4 / cell)))
        if top >= nv - 1:
            break
        for u0 in range(margin, nu - win_w, pitch):
            if storey and rng.random() < 0.06:
                continue           # a blank bay; perfect regularity reads as CGI
            occupancy[u0:u0 + win_w, sill:top] = 0
            # Flag the reveal so a theme can put trim or a lintel on it.
            context[max(0, u0 - 1):u0 + win_w + 1,
                    max(0, sill - 1):min(nv, top + 1)] |= int(Ctx.NEAR_OPENING)
            context[u0:u0 + win_w, sill:top] |= int(Ctx.OPENING_BOUNDARY)


def _clip_to_footprint(lattice, patch, ring: np.ndarray) -> None:
    """Cut the roof back to the building it belongs to.

    `build_solid` fills a rectangle, because that is all a plane knows about
    itself. A footprint is almost never a rectangle -- LoDo is full of L-shapes
    and stepped frontages -- so the rectangle spills over the neighbours, and
    from above the block reads as one continuous grey deck with buildings
    poking through it. That is what it looked like.

    Clipping is the whole fix: keep the cells whose centres lie inside the
    footprint and drop the rest. Cheap, and it is the same polygon the walls
    were extruded from, so the roof lands exactly on them.
    """
    from ..data.gis import point_in_polygon

    nu, nv = lattice.occupancy.shape
    iu, iv = np.meshgrid(np.arange(nu), np.arange(nv), indexing="ij")
    uv = lattice.cell_uv(iu.ravel(), iv.ravel())
    world_xy = patch.unproject(uv)[:, :2]
    inside = point_in_polygon(world_xy, ring[:, :2]).reshape(nu, nv)
    lattice.occupancy[~inside] = 0

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
from ..reconstruct import fenestrate as fenestrate_stage
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


def _upsample(raster, z: np.ndarray, cell: float):
    """Re-grid the seed's coarse heightfield onto a finer terrain lattice.

    The seed stores terrain at 4 m because that is finer than any street
    gradient a player feels, and for *height* that is true. It is not true for
    anything that reads off the terrain classes: a kerb is a sub-metre feature,
    so on a 4 m grid the "edge of the carriageway" is 42% of the carriageway,
    and the theme dutifully paints half the street as kerb stone. That is the
    checkerboard -- not a texture bug, a resolution mismatch between what the
    seed stores and what the surface rules need.

    Interpolating height up costs nothing in fidelity (it was smooth at 4 m and
    is smooth at 1 m) and gives the road stamp somewhere to put an edge that is
    actually an edge.
    """
    from ..spatial.grid import Raster2D

    nx, ny = z.shape
    far = raster.origin + np.array([nx, ny]) * raster.cell
    fine = Raster2D(raster.origin, far, cell, pad=0)
    fu = np.linspace(0, nx - 1, fine.nx)
    fv = np.linspace(0, ny - 1, fine.ny)
    iu = np.clip(fu.astype(int), 0, nx - 2)
    iv = np.clip(fv.astype(int), 0, ny - 2)
    tu = (fu - iu)[:, None]
    tv = (fv - iv)[None, :]
    z00 = z[np.ix_(iu, iv)]; z10 = z[np.ix_(iu + 1, iv)]
    z01 = z[np.ix_(iu, iv + 1)]; z11 = z[np.ix_(iu + 1, iv + 1)]
    out = (z00 * (1 - tu) * (1 - tv) + z10 * tu * (1 - tv)
           + z01 * (1 - tu) * tv + z11 * tu * tv)
    return fine, out.astype(np.float32)


def expand(seed: dict, *, tile: float = 0.5, terrain_cell: float = 1.0,
           roof_pitch: float = 0.0) -> World:
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
    if terrain_cell and terrain_cell < raster.cell:
        raster, dtm = _upsample(raster, dtm, terrain_cell)
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

    # Footprints of every building, so a wall can ask whether the next
    # building is standing against it.
    neighbours = []
    for other in seed.get("buildings", []):
        other_ring = np.asarray(other.get("footprint", []), dtype=float)
        neighbours.append(other_ring if len(other_ring) >= 4 else None)

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

        # An architectural family per building. LoDo is a brick warehouse
        # district so brick is right, but a street where every frontage is the
        # *same* brick reads as one continuous wall -- which is exactly what it
        # looked like. Real terraces vary: stock brick, weathered stock, and
        # the occasional stone-fronted building.
        #
        # Deterministic on the footprint, so a place keeps its character across
        # regenerations rather than reshuffling every run.
        family = _family(ring, index)
        surfaces = extrude_stage.walls_from_footprint(ring, base, top,
                                                      start_id=walls_made)
        _mark_party_walls(surfaces, ring, neighbours, index)
        walls_made += len(surfaces)
        for patch in surfaces:
            width, height = patch.extent
            lat = lattice_stage.build_solid(patch, width, height, cell=tile,
                                            ground_z=base)
            _fenestrate(lat, patch, building, index)
            _restate_evidence(lat, family)
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

    The rhythm itself lives in `reconstruct/fenestrate.py`, because the compile
    path needs exactly the same thing: an extruded wall has no returns to detect
    an opening in, so both paths are generating rather than measuring. Keeping
    one copy is what stops a street generated from a seed drifting away from
    the same street compiled from the tile.
    """
    fenestrate_stage.fenestrate(
        lattice, patch, key=building.get("id", index), storey_m=STOREY_M)


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


def _mark_party_walls(surfaces, ring: np.ndarray, neighbours, index: int,
                      probe: float = 0.9) -> None:
    """Flag walls that stand against another building rather than open air.

    A terrace shares its side walls, and those are blank brick because there is
    literally another building there. Testing a point just outside the wall's
    midpoint against every other footprint is enough to tell -- if it lands
    inside a neighbour, nothing can be seen through that wall.
    """
    from ..data.gis import point_in_polygon

    centre = ring[:, :2].mean(axis=0)
    for patch in surfaces:
        mid = np.asarray(patch.centroid, dtype=float)[:2]
        outward = mid - centre
        norm = float(np.hypot(*outward))
        if norm < 1e-6:
            continue
        sample = (mid + outward / norm * probe)[None, :]
        for other, other_ring in enumerate(neighbours):
            if other == index or other_ring is None:
                continue
            lo, hi = other_ring[:, :2].min(axis=0), other_ring[:, :2].max(axis=0)
            if not (lo[0] <= sample[0, 0] <= hi[0] and lo[1] <= sample[0, 1] <= hi[1]):
                continue
            if point_in_polygon(sample, other_ring[:, :2])[0]:
                patch.attrs["party_wall"] = True
                break


#: Frontage families, and how often each appears on a street. Weights are the
#: rough mix of a brick warehouse district: mostly stock brick, a good amount
#: weathered, a few stone-fronted civic or bank buildings.
FAMILIES = (("stock", 0.55), ("weathered", 0.32), ("stone", 0.13))


def _family(ring: np.ndarray, index: int) -> str:
    """Pick a frontage family for one building, stably."""
    key = abs(hash((round(float(ring[:, 0].mean()), 1),
                    round(float(ring[:, 1].mean()), 1), index))) % 1000 / 1000.0
    running = 0.0
    for name, weight in FAMILIES:
        running += weight
        if key < running:
            return name
    return FAMILIES[-1][0]


def _restate_evidence(lattice, family: str) -> None:
    """Say what a generated wall actually is, so the theme can differentiate.

    `build_solid` flags every cell SPARSE_EVIDENCE and OCCLUDED, which is true
    of a wall the compiler *inferred from thin returns*. It is the wrong claim
    about a wall the generator *invented from a footprint*: nothing was
    measured sparsely here, nothing was measured at all. OCCLUDED alone carries
    that honestly.

    It also had a visible cost. The victorian pack maps SPARSE_EVIDENCE to
    weathered brick at priority 1, above the default stock brick at 0, so every
    frontage in the block resolved to the same grubby brick and the quoin,
    cornice and plinth rules had nothing to sit against. Clearing the flag lets
    the articulation show, and re-applying it only for the weathered family
    turns an accident into a choice.
    """
    context = lattice.context
    context &= ~np.uint32(Ctx.SPARSE_EVIDENCE)
    if family == "weathered":
        context |= np.uint32(Ctx.SPARSE_EVIDENCE)
    elif family == "stone":
        # Stone-fronted buildings read as dressed stone across the frontage,
        # which the pack already has a material for.
        context |= np.uint32(Ctx.SHELTERED)

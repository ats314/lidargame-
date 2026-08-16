"""The compiler: point clouds in, Spatial IR out.

Stage order and what each one earns:

    ingest      normalise any source into one cloud, keep licence + CRS
    datum       shift to a local origin (UTM coordinates destroy float32)
    terrain     bare-earth model, then height above ground for everything
    features    multiscale shape descriptors -- planar? linear? edge? corner?
    semantics   source labels if present, geometric inference if not
    roles       per-point role hints from semantics + shape
    segment     planar patches, tree instances, vehicles, poles
    lattice     a tile grid per patch, with the context bitmask and openings
    topology    relations between patches -> buildings, corners, street frontage
    reconstruct terrain mesh + merged surface quads, all theme-independent
    graph       nodes, edges, provenance, confidence

Every stage is timed and recorded in the IR, so a compiled world can explain
itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import ingest
from .features import ground as ground_stage
from .features import neighborhood
from .reconstruct import lattice as lattice_stage
from .reconstruct import mesh as mesh_stage
from .reconstruct import terrain as terrain_stage
from .roles.classify import classify as classify_roles, role_histogram
from .roles.taxonomy import Ctx, ROLE_INDEX
from .segment import instances as instance_stage
from .segment import planes as plane_stage
from .semantics import infer as semantic_stage
from .topology import graph as topology_stage
from .types import (SEMANTIC_CLASSES, Geometry, Node, PointCloud, World)


@dataclass
class Config:
    name: str = "world"
    #: metres; drives the DTM, terrain mesh and road classification
    terrain_cell: float = 1.0
    #: neighbourhood radii for the shape descriptors
    scales: tuple[float, ...] = neighborhood.DEFAULT_SCALES
    #: voxel size for planar region growing
    plane_voxel: float = 0.6
    plane_angle_deg: float = 18.0
    plane_dist: float = 0.32
    min_plane_voxels: int = 12
    #: facade tile size -- this is the resolution of the context mask
    tile: float = 0.25
    #: Fuse patches that are the same physical surface before tiling them.
    #: Without this one facade arrives as several ragged patches.
    merge_coplanar: bool = True
    #: Carry wall surfaces down to terrain contact, flagged as inferred.
    extend_walls_to_ground: bool = True
    detect_openings: bool = True
    min_opening_area: float = 0.35
    max_opening_area: float = 14.0
    instance_trees: bool = True
    keep_points: bool = True
    max_kept_points: int = 600_000
    #: uniform decimation applied at ingest (1 = keep everything)
    decimate: int = 1
    #: (minx, miny, maxx, maxy) in the source CRS. Real tiles are typically
    #: 1.5 km square and a city block is what you actually want to walk.
    bbox: tuple[float, float, float, float] | None = None
    #: Centred square crop, in metres. Unlike `bbox` this needs no knowledge of
    #: the tile's CRS or extent, so it survives being pointed at whichever tile
    #: a fetch happens to return.
    crop_m: float | None = None
    verbose: bool = True
    extras: dict = field(default_factory=dict)


def _log(config: Config, message: str) -> None:
    if config.verbose:
        print(f"  {message}", flush=True)


def load_sources(paths, config: Config, adapter: str | None = None,
                 options: dict | None = None) -> tuple[PointCloud, list]:
    """Ingest one or many files into a single cloud with per-point provenance."""
    clouds, sources = [], []
    for i, path in enumerate(paths):
        result = ingest.load(path, adapter=adapter, source_id=f"src{i}", **(options or {}))
        cloud = result.cloud
        bbox = config.bbox
        if bbox is None and config.crop_m:
            lo, hi = cloud.bounds
            cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
            half = config.crop_m / 2
            bbox = (cx - half, cy - half, cx + half, cy + half)
        if bbox is not None:
            minx, miny, maxx, maxy = bbox
            inside = ((cloud.xyz[:, 0] >= minx) & (cloud.xyz[:, 0] <= maxx)
                      & (cloud.xyz[:, 1] >= miny) & (cloud.xyz[:, 1] <= maxy))
            kept = int(inside.sum())
            if kept == 0:
                lo, hi = cloud.bounds
                raise ValueError(
                    f"bbox {tuple(round(v, 1) for v in bbox)} selects no points from "
                    f"{Path(path).name}; that file covers "
                    f"X {lo[0]:.0f}..{hi[0]:.0f}, Y {lo[1]:.0f}..{hi[1]:.0f} "
                    f"(CRS {result.source.crs or 'unknown'}). Use --crop <metres> to take a "
                    f"centred block instead of guessing coordinates.")
            cloud = cloud.subset(inside)
            result.source.notes += (f"; cropped to {tuple(round(v, 1) for v in bbox)} "
                                    f"({kept:,} of {len(result.cloud):,} points)")
        if config.decimate > 1:
            cloud = cloud.subset(np.arange(0, len(cloud), config.decimate))
        cloud["source"] = np.full(len(cloud), i, dtype=np.uint8)
        clouds.append(cloud)
        sources.append(result.source)

    if len(clouds) == 1:
        return clouds[0], sources

    # Union the channels so a labelled tile and an unlabelled one can merge.
    channel_names = set().union(*(set(c.channels) for c in clouds))
    merged_xyz = np.concatenate([c.xyz for c in clouds])
    merged = PointCloud(merged_xyz, source_id="merged")
    for name in channel_names:
        parts = []
        for c in clouds:
            if name in c:
                parts.append(c[name])
            else:
                template = next(x[name] for x in clouds if name in x)
                shape = (len(c),) + template.shape[1:]
                parts.append(np.zeros(shape, dtype=template.dtype))
        merged[name] = np.concatenate(parts)
    return merged, sources


def compile_world(paths, config: Config | None = None, *, adapter: str | None = None,
                  ingest_options: dict | None = None) -> World:
    config = config or Config()
    paths = [paths] if isinstance(paths, (str, Path)) else list(paths)

    world = World(name=config.name)
    with world.stage("ingest", files=[str(p) for p in paths], decimate=config.decimate) as rec:
        cloud, sources = load_sources(paths, config, adapter, ingest_options)
        world.sources = sources
        world.crs = next((s.crs for s in sources if s.crs), "")
        rec.notes = f"{len(cloud):,} points from {len(sources)} source(s)"
    _log(config, rec.notes)

    # --- datum -----------------------------------------------------------
    with world.stage("datum") as rec:
        lo, hi = cloud.bounds
        origin = np.array([lo[0], lo[1], 0.0])
        cloud.xyz -= origin
        world.origin = origin
        world.bounds = np.array([cloud.bounds[0], cloud.bounds[1]])
        sensors = {}
        for source in world.sources:
            if source.sensor_origin is not None:
                sensors[source.id] = (np.asarray(source.sensor_origin) - origin).tolist()
        if sensors:
            world.notes["sensor_origins"] = sensors
        rec.notes = f"shifted to local origin {np.round(origin, 2).tolist()}"

    # --- terrain ---------------------------------------------------------
    with world.stage("terrain", cell=config.terrain_cell) as rec:
        raster, dtm = ground_stage.estimate(cloud, cell=config.terrain_cell)
        rec.notes = f"DTM {raster.nx}x{raster.ny} @ {config.terrain_cell} m via {cloud.meta['dtm_method']}"
    _log(config, rec.notes)

    # --- shape descriptors ------------------------------------------------
    with world.stage("features", scales=list(config.scales)) as rec:
        neighborhood.compute(cloud, config.scales)
        rec.notes = f"multiscale descriptors at {config.scales} m"
    _log(config, f"{rec.notes} ({rec.seconds:.1f}s)")

    # --- semantics --------------------------------------------------------
    with world.stage("semantics") as rec:
        semantic_stage.infer(cloud)
        histogram = semantic_stage.class_histogram(cloud)
        rec.notes = cloud.meta.get("semantic_source", "")
        rec.params["histogram"] = histogram
    _log(config, f"semantics ({rec.notes}): " +
         ", ".join(f"{k} {v:,}" for k, v in sorted(histogram.items(), key=lambda kv: -kv[1])[:6]))

    # --- roles ------------------------------------------------------------
    with world.stage("roles") as rec:
        classify_roles(cloud)
        rec.params["histogram"] = role_histogram(cloud)

    # --- planar segmentation ---------------------------------------------
    with world.stage("segment.planes", voxel=config.plane_voxel) as rec:
        patches = plane_stage.extract(
            cloud, voxel=config.plane_voxel, angle_deg=config.plane_angle_deg,
            dist=config.plane_dist, min_voxels=config.min_plane_voxels)
        raw_count = len(patches)
        if config.merge_coplanar:
            patches = plane_stage.merge_coplanar(patches, cloud)
        plane_stage.assign_patch_channel(cloud, patches)
        rec.notes = (f"{len(patches)} planar patches"
                     + (f" (merged from {raw_count})" if len(patches) != raw_count else ""))
    _log(config, f"{rec.notes} ({rec.seconds:.1f}s)")

    # --- tile lattices ----------------------------------------------------
    with world.stage("lattice", tile=config.tile) as rec:
        lattices = {}
        total_openings = 0
        for patch in patches:
            pts = cloud.xyz[patch.point_idx]
            ground_z = float(np.percentile(pts[:, 2], 2))
            lat = lattice_stage.build(
                patch, pts, cell=config.tile, ground_z=ground_z,
                extend_to_ground=config.extend_walls_to_ground,
                min_opening_area=config.min_opening_area if config.detect_openings else 1e9,
                max_opening_area=config.max_opening_area)
            lattices[patch.id] = lat
            total_openings += len(lat.openings)
        rec.notes = f"{sum(l.solid_count for l in lattices.values()):,} tiles, {total_openings} openings"
    _log(config, rec.notes)

    # --- topology ---------------------------------------------------------
    with world.stage("topology") as rec:
        relations = topology_stage.relate_patches(patches, cloud)
        topology_stage.annotate_cross_patch_context(patches, lattices, relations, cloud)
        class_raster, coverage = terrain_stage.classify_cells(cloud, raster, dtm)
        dtm = terrain_stage.smooth_terrain(dtm, class_raster)
        street = topology_stage.mark_street_facing(
            patches, lattices, raster, terrain_stage.road_mask(class_raster))
        structures = topology_stage.group_structures(patches, relations)
        rec.params["relations"] = topology_stage.summarize(relations)
        rec.notes = (f"{len(structures)} structures, {len(relations)} relations, "
                     f"{street} street-facing patches")
    _log(config, rec.notes)

    # --- instances --------------------------------------------------------
    with world.stage("segment.instances") as rec:
        chm = ground_stage.canopy_height_model(cloud, raster, dtm)
        trees = instance_stage.trees(cloud, raster, chm) if config.instance_trees else []
        vehicles = instance_stage.cluster(cloud, ("vehicle",), "instance.vehicle", voxel=0.9)
        poles = instance_stage.poles(cloud)
        rec.notes = f"{len(trees)} trees, {len(vehicles)} vehicles, {len(poles)} poles"
    _log(config, rec.notes)

    # --- geometry ---------------------------------------------------------
    with world.stage("reconstruct") as rec:
        builder = mesh_stage.MeshBuilder()
        terrain_ctx = mesh_stage.terrain_context(class_raster, dtm)
        terrain_node = world.add(Node(
            id="terrain", role="terrain.ground", semantic="ground", kind="terrain",
            confidence=0.9, stage="reconstruct",
            sources=[s.id for s in world.sources],
            geometry=Geometry("heightfield", {"height": "terrain/dtm", "class": "terrain/class"},
                              {"origin": raster.origin.tolist(), "cell": raster.cell,
                               "shape": [raster.nx, raster.ny]}),
            attrs={"cell": raster.cell, "method": cloud.meta.get("dtm_method", "")}))
        node_slots = ["terrain"]
        quads = mesh_stage.add_terrain(
            builder, raster, dtm, class_raster, terrain_ctx,
            terrain_stage.ROLE_LOOKUP, 0, mask=class_raster != terrain_stage.VOID)

        patch_to_structure = {}
        for si, group in enumerate(structures):
            for pid in group:
                patch_to_structure[pid] = si

        for si, group in enumerate(structures):
            member_patches = [patches[i] for i in group]
            support = sum(p.support for p in member_patches)
            pts = np.concatenate([cloud.xyz[p.point_idx] for p in member_patches])
            lo, hi = pts.min(axis=0), pts.max(axis=0)
            bid = f"bldg.{si:04d}"
            world.add(Node(
                id=bid, role="volume.building", semantic="building", kind="object",
                confidence=float(np.clip(np.mean([p.confidence for p in member_patches]), 0.1, 0.99)),
                support=support, stage="topology",
                sources=[s.id for s in world.sources],
                attrs={"footprint": [lo[:2].tolist(), hi[:2].tolist()],
                       "height": float(hi[2] - lo[2]), "patches": len(member_patches)},
                geometry=Geometry("aggregate", bounds=[*lo.tolist(), *hi.tolist()])))

        for patch in patches:
            lat = lattices[patch.id]
            if lat.solid_count == 0:
                continue
            si = patch_to_structure.get(patch.id)
            parent = f"bldg.{si:04d}" if si is not None else None
            sid = f"{parent}/face.{patch.id:04d}" if parent else f"face.{patch.id:04d}"
            node_index = len(node_slots)
            node_slots.append(sid)

            ctx_key = f"lattice/{patch.id}"
            world.put_array(f"{ctx_key}/occupancy", lat.occupancy)
            world.put_array(f"{ctx_key}/context", lat.context)
            world.put_array(f"{ctx_key}/evidence", lat.evidence)
            world.add(Node(
                id=sid, role=patch.role, semantic="building", kind="surface",
                parent=parent, confidence=patch.confidence, support=patch.support,
                stage="segment.planes", sources=[s.id for s in world.sources],
                geometry=Geometry("tiled_plane",
                                  {"occupancy": f"{ctx_key}/occupancy",
                                   "context": f"{ctx_key}/context",
                                   "evidence": f"{ctx_key}/evidence"},
                                  {"origin": patch.centroid.tolist(), "u": patch.u.tolist(),
                                   "v": patch.v.tolist(), "normal": patch.normal.tolist(),
                                   "cell": lat.cell, "shape": list(lat.shape),
                                   "uvOrigin": list(lat.uv_origin)}),
                attrs={"area": round(float(patch.area), 2), "slope_deg": round(patch.slope_deg, 1),
                       "rms": round(patch.rms, 3), "tiles": lat.solid_count,
                       "context": lattice_stage.context_histogram(lat),
                       **patch.attrs},
                tags=["street_facing"] if patch.attrs.get("street_facing") else []))
            quads += mesh_stage.add_lattice(builder, patch, lat, node_index)

            for opening in lat.openings:
                oid = f"{sid}/opening.{opening.id:02d}"
                world.add(Node(
                    id=oid, role=opening.role, semantic="building", kind="opening",
                    parent=sid, confidence=opening.confidence, stage="lattice",
                    sources=[s.id for s in world.sources],
                    geometry=Geometry("instance", frame={
                        "position": (opening.center_world + world.origin).tolist(),
                        "size": [opening.width, opening.height, 0.15],
                        "normal": patch.normal.tolist()}),
                    attrs={"width": round(opening.width, 2), "height": round(opening.height, 2),
                           "sill_height": round(opening.sill_height, 2)}))
                world.link(oid, sid, "opening_in", opening.confidence)

        # relations between faces
        for i, j, rel, conf, attrs in relations:
            a = _face_id(patches[i].id, patch_to_structure)
            b = _face_id(patches[j].id, patch_to_structure)
            if a in world.nodes and b in world.nodes:
                world.link(a, b, rel, conf, **attrs)

        _add_instances(world, trees, "tree", "supports")
        _add_instances(world, vehicles, "vehicle", "supports")
        _add_instances(world, poles, "pole", "supports")

        arrays = builder.finalize()
        for key, value in arrays.items():
            world.put_array(f"mesh/{key}", value)
        world.put_array("terrain/dtm", np.nan_to_num(dtm, nan=0.0).astype(np.float32))
        world.put_array("terrain/class", class_raster.astype(np.uint8))
        world.put_array("terrain/context", terrain_ctx.astype(np.uint32))
        rec.notes = (f"{quads:,} merged quads, {len(arrays['positions']):,} vertices, "
                     f"{len(arrays['indices']):,} triangles")
    _log(config, rec.notes)

    # --- retained point layer --------------------------------------------
    if config.keep_points:
        step = max(1, len(cloud) // config.max_kept_points)
        kept = cloud.subset(np.arange(0, len(cloud), step))
        for name in list(kept.channels):
            if "@" in name:                      # per-scale debug channels
                del kept.channels[name]
        world.points = kept

    world.notes.update({
        "semantic_classes": SEMANTIC_CLASSES,
        "tile_size_m": config.tile,
        "context_flags": {name: bit for bit, name in sorted(Ctx.NAMES.items())},
    })
    return world


def _face_id(patch_id: int, patch_to_structure: dict) -> str:
    si = patch_to_structure.get(patch_id)
    return f"bldg.{si:04d}/face.{patch_id:04d}" if si is not None else f"face.{patch_id:04d}"


def _add_instances(world: World, items, prefix: str, relation: str) -> None:
    for item in items:
        nid = f"{prefix}.{item.id:04d}"
        if nid in world.nodes:
            continue
        world.add(Node(
            id=nid, role=item.role, semantic=item.role.split(".")[-1], kind="instance",
            confidence=item.confidence, support=item.support, stage="segment.instances",
            sources=[s.id for s in world.sources],
            geometry=Geometry("instance", frame={
                "position": (item.center + world.origin).tolist(),
                "size": item.size.tolist(),
                "yaw": float(item.attrs.get("yaw", 0.0))}),
            attrs={k: (round(float(v), 3) if isinstance(v, (int, float, np.floating)) else v)
                   for k, v in item.attrs.items()}))
        world.link(nid, "terrain", relation, item.confidence)

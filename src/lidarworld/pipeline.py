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
from .reconstruct import extrude as extrude_stage
from .reconstruct import freespace
from .reconstruct import lattice as lattice_stage
from .reconstruct import mesh as mesh_stage
from .reconstruct import terrain as terrain_stage
from .roles.classify import classify as classify_roles, role_histogram
from .roles.taxonomy import Ctx, ROLE_INDEX
from .segment import instances as instance_stage
from .segment import planes as plane_stage
from .semantics import infer as semantic_stage
from .topology import footprints as footprint_stage
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
    #: Synthesise walls by extruding footprints to measured roof height.
    #: Airborne LiDAR does not contain facades; this is how a city gets them.
    extrude_walls: bool = True
    #: Promote terrain under the published street network to carriageway.
    #: Intensity finds under 6% of a downtown grid; the network is
    #: authoritative and Denver publishes it.
    streets: str | None = None
    #: Reject synthesised surfaces standing above the highest return in
    #: their column -- space the beam demonstrably crossed.
    gate_free_space: bool = True
    free_clearance: float = 1.5
    #: Authoritative building footprints: a path to GeoJSON, or a layer id from
    #: lidarworld.data.gis.FOOTPRINTS to fetch for the compiled extent.
    footprints: str | None = None
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
    shared_bbox = config.bbox
    kept_total = 0
    for i, path in enumerate(paths):
        result = ingest.load(path, adapter=adapter, source_id=f"src{i}", **(options or {}))
        cloud = result.cloud
        # The crop box is decided once, from the first source, and reused for
        # every other one. Centring per file would carve a separate block out
        # of each tile and leave them scattered across the world with empty
        # space between -- which is exactly what happens when a glob picks up
        # more tiles than you meant.
        if shared_bbox is None and config.crop_m:
            lo, hi = cloud.bounds
            cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
            half = config.crop_m / 2
            shared_bbox = (cx - half, cy - half, cx + half, cy + half)
        bbox = shared_bbox
        if bbox is not None:
            minx, miny, maxx, maxy = bbox
            inside = ((cloud.xyz[:, 0] >= minx) & (cloud.xyz[:, 0] <= maxx)
                      & (cloud.xyz[:, 1] >= miny) & (cloud.xyz[:, 1] <= maxy))
            kept = int(inside.sum())
            kept_total += kept
            if kept == 0 and len(paths) > 1:
                # A neighbouring tile that does not reach the crop box is
                # normal, not an error. Only an empty result overall is.
                continue
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

    if not clouds:
        raise ValueError(
            f"crop selected no points from any of {len(paths)} source(s). "
            "Check that they overlap the area you asked for.")
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

    # --- footprints (before segmentation: they constrain merging) ---------
    footprint_rings, footprint_attrs = [], []
    if config.footprints:
        with world.stage("footprints", source=config.footprints) as rec:
            footprint_rings, footprint_attrs = _load_footprints(config.footprints, world, cloud)
            footprint_rings = [r - world.origin[:2] for r in footprint_rings]
            rec.notes = f"{len(footprint_rings)} footprint polygons"
        _log(config, rec.notes or "no footprints returned")

    # --- planar segmentation ---------------------------------------------
    with world.stage("segment.planes", voxel=config.plane_voxel) as rec:
        patches = plane_stage.extract(
            cloud, voxel=config.plane_voxel, angle_deg=config.plane_angle_deg,
            dist=config.plane_dist, min_voxels=config.min_plane_voxels)
        raw_count = len(patches)
        if config.merge_coplanar:
            confine = (footprint_stage.assign_patches(patches, footprint_rings)
                       if footprint_rings else None)
            patches = plane_stage.merge_coplanar(patches, cloud, groups=confine)
        plane_stage.assign_patch_channel(cloud, patches)
        rec.notes = (f"{len(patches)} planar patches"
                     + (f" (merged from {raw_count})" if len(patches) != raw_count else ""))
    _log(config, f"{rec.notes} ({rec.seconds:.1f}s)")

    # --- extruded walls ---------------------------------------------------
    if config.extrude_walls and footprint_rings:
        with world.stage("extrude", source=config.footprints) as rec:
            assignment = footprint_stage.assign_patches(patches, footprint_rings)
            new_walls, programs = extrude_stage.build(
                footprint_rings, assignment, patches, cloud, raster, dtm,
                start_id=len(patches), attrs=footprint_attrs)
            patches.extend(new_walls)
            world.programs.extend(programs)
            params = sum(p.cost for p in programs)
            rec.params["programs"] = {"count": len(programs), "parameters": params}
            rec.notes = (f"{len(new_walls)} walls extruded from "
                         f"{len(footprint_rings)} footprints "
                         f"({len(programs)} programs, {params} parameters)")
        _log(config, rec.notes)

    # --- tile lattices ----------------------------------------------------
    with world.stage("lattice", tile=config.tile) as rec:
        lattices = {}
        total_openings = 0
        # Synthesised surfaces are the compiler's own guesses, so they get
        # checked against the returns before they are allowed to exist.
        free = (freespace.FreeSpace(cloud.xyz, raster, clearance=config.free_clearance)
                if config.gate_free_space else None)
        gated_cells = 0
        gated_patches: set[int] = set()
        program_residual: dict[str, list[int]] = {}
        for patch in patches:
            if patch.attrs.get("extruded"):
                lat = lattice_stage.build_solid(
                    patch, patch.extent[0], patch.extent[1], cell=config.tile,
                    ground_z=patch.attrs.get("base_z"))
                if free is not None:
                    cleared, rejected = freespace.gate_lattice(lat, patch, free)
                    gated_cells += cleared
                    # The residual belongs to the program, not the wall: it is
                    # the D(O, S(E(W))) term for the parameters that made it.
                    key = patch.attrs.get("program")
                    if key is not None:
                        before, after = program_residual.setdefault(key, [0, 0])
                        program_residual[key] = [before + cleared,
                                                 after + cleared + lat.solid_count]
                    if rejected:
                        gated_patches.add(patch.id)
                        continue
                    if cleared:
                        patch.attrs["free_space_cleared"] = cleared
                lattices[patch.id] = lat
                continue
            pts = cloud.xyz[patch.point_idx]
            ground_z = float(np.percentile(pts[:, 2], 2))
            lat = lattice_stage.build(
                patch, pts, cell=config.tile, ground_z=ground_z,
                extend_to_ground=config.extend_walls_to_ground,
                # Openings are a facade phenomenon: glass returns nothing at
                # 905 nm, so an enclosed hole in a wall's returns is a window.
                # An enclosed hole in a *roof* is a scan shadow or a rooftop
                # plant occlusion, and calling it a window speckles every roof
                # in the block with window trim.
                min_opening_area=(config.min_opening_area
                                  if config.detect_openings and patch.role.startswith(
                                      "surface.wall") else 1e9),
                max_opening_area=config.max_opening_area)
            lattices[patch.id] = lat
            total_openings += len(lat.openings)
        rec.notes = f"{sum(l.solid_count for l in lattices.values()):,} tiles, {total_openings} openings"
        if free is not None and (gated_cells or gated_patches):
            rec.params["free_space"] = {"cells_cleared": gated_cells,
                                        "patches_rejected": len(gated_patches),
                                        "observed_columns": round(free.observed_fraction, 3)}
            rec.notes += (f" ({gated_cells:,} synthesised cells and "
                          f"{len(gated_patches)} surfaces rejected as free space)")
            patches = [p for p in patches if p.id not in gated_patches]
        for program in world.programs:
            bad, total = program_residual.get(program.id, [0, 0])
            if total:
                program.residual = bad / total
    _log(config, rec.notes)

    # --- topology ---------------------------------------------------------
    with world.stage("topology") as rec:
        relations = topology_stage.relate_patches(patches, cloud)
        topology_stage.annotate_cross_patch_context(patches, lattices, relations, cloud)
        class_raster, coverage = terrain_stage.classify_cells(cloud, raster, dtm)
        if config.streets:
            info = _apply_streets(config.streets, world, cloud, raster, class_raster)
            rec.params["streets"] = info
            _log(config, f"street network promoted {info['promoted']:,} terrain cells "
                         f"to carriageway ({info['road_cells_before']:,} -> "
                         f"{info['road_cells_after']:,})")
        dtm = terrain_stage.smooth_terrain(dtm, class_raster)
        street = topology_stage.mark_street_facing(
            patches, lattices, raster, terrain_stage.road_mask(class_raster))
        adjacency_groups = topology_stage.group_structures(patches, relations)
        structures = adjacency_groups
        footprint_info = ""
        if footprint_rings:
            rings = footprint_rings
            if True:
                assignment = footprint_stage.assign_patches(patches, rings)
                structures = footprint_stage.group_by_footprint(
                    patches, assignment, adjacency_groups)
                matched = int((assignment >= 0).sum())
                footprint_info = (f"; {len(rings)} footprints matched {matched}/"
                                  f"{len(patches)} patches")
                rec.params["footprints"] = {"polygons": len(rings), "matched": matched}
                world.notes["footprint_source"] = config.footprints
        rec.params["relations"] = topology_stage.summarize(relations)
        rec.notes = (f"{len(structures)} structures, {len(relations)} relations, "
                     f"{street} street-facing patches{footprint_info}")
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
            member_pts = [cloud.xyz[p.point_idx] for p in member_patches
                          if len(p.point_idx)]
            member_pts += [topology_stage.patch_corners(p) for p in member_patches
                           if not len(p.point_idx)]
            pts = np.concatenate(member_pts)
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


def _load_footprints(spec: str, world: World, cloud: PointCloud):
    """Resolve `--footprints` to (rings, attributes) in the source CRS."""
    import json

    from .data.gis import FOOTPRINTS, attributes, fetch_footprints, polygons

    path = Path(spec)
    if path.exists():
        geojson = json.loads(path.read_text())
        layer = FOOTPRINTS.get("denver")
        return polygons(geojson), (attributes(geojson, layer) if layer else [])

    if spec not in FOOTPRINTS:
        raise ValueError(f"unknown footprint source {spec!r}; "
                         f"have {sorted(FOOTPRINTS)} or a path to GeoJSON")
    layer = FOOTPRINTS[spec]

    try:
        from pyproj import Transformer
    except ImportError:
        raise ImportError(
            "fetching footprints for an extent needs pyproj to convert the "
            "cloud's bounds to WGS84: pip install pyproj. Alternatively pass a "
            "path to a GeoJSON file you have already downloaded.") from None
    if not world.crs:
        raise ValueError("the source has no CRS, so its extent cannot be converted "
                         "to WGS84 to request footprints. Pass a GeoJSON path instead.")

    lo, hi = cloud.bounds
    lo = lo[:2] + world.origin[:2]
    hi = hi[:2] + world.origin[:2]
    to_wgs = Transformer.from_crs(world.crs, "EPSG:4326", always_xy=True)
    west, south = to_wgs.transform(lo[0], lo[1])
    east, north = to_wgs.transform(hi[0], hi[1])

    # Request WGS84 and reproject here rather than scraping an EPSG code out of
    # the WKT: a compound CRS ends with its *vertical* datum, so asking the
    # server for "the last EPSG code" quietly returns nothing.
    geojson = fetch_footprints(layer, (west, south, east, north), out_crs="4326")
    from_wgs = Transformer.from_crs("EPSG:4326", world.crs, always_xy=True)
    rings = []
    for ring in polygons(geojson):
        x, y = from_wgs.transform(ring[:, 0], ring[:, 1])
        rings.append(np.column_stack([x, y]))
    return rings, attributes(geojson, layer)


def _apply_streets(spec: str, world, cloud, raster, class_raster) -> dict:
    """Rasterise a street network over the terrain classes. Same fetch path as
    footprints: a local GeoJSON, or a named municipal service."""
    import json

    from .data import denver as denver_data
    from .topology import streets as street_stage

    path = Path(spec)
    if path.exists():
        geojson = json.loads(path.read_text())
    else:
        if spec not in ("denver",):
            raise ValueError(f"unknown street source {spec!r}; have 'denver' or "
                             "a path to GeoJSON")
        from pyproj import Transformer
        lo, hi = cloud.bounds
        lo = lo[:2] + world.origin[:2]
        hi = hi[:2] + world.origin[:2]
        to_wgs = Transformer.from_crs(world.crs, "EPSG:4326", always_xy=True)
        west, south = to_wgs.transform(lo[0], lo[1])
        east, north = to_wgs.transform(hi[0], hi[1])
        import urllib.request
        url = denver_data.query_url(denver_data.LAYERS["street_centerlines"],
                                    (west, south, east, north), out_crs="4326")
        request = urllib.request.Request(url, headers={"User-Agent": "lidarworld/0.1"})
        with urllib.request.urlopen(request, timeout=180) as response:
            geojson = json.load(response)

    lines = street_stage.polylines(geojson)
    half = street_stage.widths(geojson)
    if len(half) != len(lines):
        half = [5.5] * len(lines)
    if not path.exists():
        from pyproj import Transformer
        from_wgs = Transformer.from_crs("EPSG:4326", world.crs, always_xy=True)
        converted = []
        for line in lines:
            x, y = from_wgs.transform(line[:, 0], line[:, 1])
            converted.append(np.column_stack([x, y]))
        lines = converted
    lines = [line - world.origin[:2] for line in lines]

    # The seed wants the network itself, not the raster: a centreline and a
    # width regenerate a road, a class raster only recolours one.
    world.notes["road_network"] = [
        {"line": [[round(float(x), 2), round(float(y), 2)] for x, y in line],
         "half_width": round(float(w), 2)}
        for line, w in zip(lines, half)]

    mask = street_stage.rasterise(lines, half, raster)
    info = street_stage.apply(class_raster, mask,
                              ground=terrain_stage.GROUND, road=terrain_stage.ROAD,
                              void=terrain_stage.VOID)
    info["segments"] = len(lines)
    world.notes["street_source"] = spec
    return info

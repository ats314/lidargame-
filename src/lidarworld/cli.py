"""Command line interface.

    lidarworld compile tile.las -o build/world --theme victorian --theme neon
    lidarworld inspect build/world/world.lwir
    lidarworld themes
    lidarworld explain --theme victorian --role surface.wall.vertical \\
                       --context corner_convex,street_facing
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse_area(text: str) -> tuple[float, float, float, float]:
    parts = [v for v in text.split(",") if v.strip()]
    if len(parts) != 3:
        raise SystemExit("--area takes x,y,size in the source CRS, "
                         "e.g. --area 500250,4399600,350")
    try:
        x, y, size = (float(v) for v in parts)
    except ValueError:
        raise SystemExit(f"--area {text!r} is not three numbers") from None
    half = size / 2.0
    return (x - half, y - half, x + half, y + half)


def _resolve_inputs(args) -> tuple[list[str], tuple | None]:
    """Turn directories and `--area` into the explicit list of tiles to open.

    Passing files loads exactly those files, as before. Passing a directory
    builds a header-only index over it and opens only the tiles that overlap
    the requested area -- the difference between compiling a block and
    decoding a metro area to throw 99% of it away.
    """
    from .data.tiles import TileIndex

    area = _parse_area(args.area) if args.area else None
    roots = [Path(p) for p in args.inputs]
    directories = [p for p in roots if p.is_dir()]
    files = [p for p in roots if not p.is_dir()]

    missing = [p for p in files if not p.exists()]
    if missing:
        raise SystemExit("no such file: " + ", ".join(str(p) for p in missing))
    selected = [str(p) for p in files]

    for root in directories:
        index = TileIndex.build(root)
        if not len(index):
            raise SystemExit(f"no LAS/LAZ tiles under {root}")
        if area is None:
            print(f"  index       {root}: {len(index)} tiles, "
                  f"{index.total_points:,} points (no --area, taking all)")
            selected.extend(t.path for t in index.tiles)
            continue
        hits = index.query(area)
        if not hits:
            lo_x, lo_y, hi_x, hi_y = index.bounds
            raise SystemExit(
                f"--area selects no tiles from {root}. The index covers "
                f"{lo_x:.0f},{lo_y:.0f} .. {hi_x:.0f},{hi_y:.0f}; the area asked for "
                f"{area[0]:.0f},{area[1]:.0f} .. {area[2]:.0f},{area[3]:.0f}. "
                "Coordinates are in the source CRS, not lat/lon.")
        covered = index.coverage(area)
        print(f"  index       {root}: {len(hits)}/{len(index)} tiles overlap, "
              f"{covered:.0%} of the area covered")
        if covered < 0.999:
            print(f"  warning     {1 - covered:.0%} of the requested area has no tile "
                  "behind it -- the world will have a hole there, not empty ground")
        selected.extend(t.path for t in hits)

    return selected, area


def _cmd_compile(args) -> int:
    from . import Config, compile_world
    from .backends import gltf as gltf_backend
    from .backends import web as web_backend
    from .ir import write_world
    from .themes import compile_theme, load_pack

    inputs, area = _resolve_inputs(args)
    bbox = tuple(float(v) for v in args.bbox.split(",")) if args.bbox else area

    config = Config(
        name=args.name or Path(args.inputs[0]).stem,
        terrain_cell=args.terrain_cell,
        tile=args.tile,
        plane_voxel=args.plane_voxel,
        decimate=args.decimate,
        bbox=bbox,
        crop_m=args.crop,
        footprints=args.footprints,
        detect_openings=not args.no_openings,
        keep_points=not args.no_points,
        verbose=not args.quiet,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"compiling {len(inputs)} source(s) -> {out}")
    world = compile_world(inputs, config, adapter=args.adapter)

    ir_path = write_world(world, out / f"{config.name}.lwir")
    print(f"  spatial IR  {ir_path} ({ir_path.stat().st_size / 1e6:.1f} MB)")

    theme_ids = args.theme or ["survey"]
    for theme_id in theme_ids:
        pack = load_pack(theme_id)
        info = compile_theme(pack, out / "themes" / pack.id, bake=not args.no_textures)
        print(f"  theme       {pack.id}: {info['materials']} materials, "
              f"{info['rules']} rules, {info['baked']} baked")

    web_info = web_backend.export(world, out, themes=theme_ids,
                                  include_points=not args.no_points)
    print(f"  web bundle  {web_info['vertices']:,} verts, {web_info['triangles']:,} tris, "
          f"{web_info['points']:,} points, {web_info['bytes'] / 1e6:.1f} MB")

    if args.sir:
        from .ir.sir import write_document
        info = write_document(world, out / f"{config.name}.sir.json")
        print(f"  spatial IR  {info['path']} (SIR v0.1: {info['entities']} entities, "
              f"{info['relations']} relations, epistemic {info['epistemic']})")

    if args.cityjson:
        from .backends import cityjson as cityjson_backend
        info = cityjson_backend.export(world, out / f"{config.name}.city.json")
        print(f"  cityjson    {info['path']} ({info['cityObjects']} CityObjects, "
              f"{info['bytes'] / 1e6:.1f} MB)")

    if args.gltf:
        pack = load_pack(theme_ids[0])
        info = gltf_backend.export(world, pack, out / "gltf", name=config.name,
                                   bake_textures=not args.no_textures)
        print(f"  gltf        {info['path']} ({info['primitives']} primitives, "
              f"{info['materials']} materials)")

    summary = world.summary()
    print(f"\n{summary['nodes']} nodes / {summary['edges']} edges")
    for role, count in list(summary["roles"].items())[:10]:
        print(f"  {role:32s} {count:>6}")
    return 0


def _cmd_validate(args) -> int:
    """Forward validation: re-scan the reconstruction and score it."""
    import numpy as np

    from . import ingest
    from .ir import read_world, write_world
    from .validate import apply_to_world, simulate

    world = read_world(args.world, load_points=False)
    result = ingest.load(args.scan)
    points = result.cloud.xyz - world.origin

    if args.sensor:
        sensor = np.array([float(v) for v in args.sensor.split(",")], dtype=float)
    elif result.source.sensor_origin is not None:
        sensor = np.asarray(result.source.sensor_origin, dtype=float) - world.origin
    elif world.notes.get("sensor_origins"):
        sensor = np.asarray(next(iter(world.notes["sensor_origins"].values())), dtype=float)
    else:
        raise SystemExit(
            "no sensor position known for this scan. Pass --sensor x,y,z in world "
            "coordinates -- comparing a scan against a viewpoint it was not taken "
            "from does not measure anything.")

    print(f"simulating {min(len(points), args.max_rays):,} rays from "
          f"{np.round(sensor, 2).tolist()} against the reconstruction")
    report = simulate(world, points, sensor, resolution=args.resolution,
                      tolerance=args.tolerance, max_rays=args.max_rays,
                      backend=args.backend)
    print(f"  {report.summary()}")

    ranked = sorted(report.per_node.items(), key=lambda kv: kv[1]["fraction"])
    if ranked:
        print("\nleast consistent surfaces:")
        for name, stats in ranked[: args.limit]:
            print(f"  {name:44s} {stats['fraction']:6.1%} of {stats['rays']:>5} rays"
                  f"  bias {stats['bias']:+.2f} m")

    if args.write_back:
        updated = apply_to_world(world, report)
        write_world(world, args.world)
        print(f"\nfolded measured consistency into {updated} node confidences -> {args.world}")
    return 0


def _cmd_inspect(args) -> int:
    from .ir import inspect, read_world

    path = Path(args.path)
    manifest = inspect(path)
    print(json.dumps({k: v for k, v in manifest.items() if k != "arrays"}, indent=1))
    print(f"\narrays ({len(manifest['arrays'])}):")
    for key, meta in sorted(manifest["arrays"].items())[: args.limit]:
        print(f"  {key:44s} {meta['dtype']:>6} {str(meta['shape']):>16} "
              f"{meta['bytes'] / 1024:>9.1f} KiB")
    if args.graph:
        world = read_world(path, load_points=False)
        print(f"\nnodes ({len(world.nodes)}):")
        for node in list(world.nodes.values())[: args.limit]:
            print(f"  {node.id:38s} {node.role:26s} conf={node.confidence:.2f} "
                  f"support={node.support}")
    return 0


def _cmd_themes(args) -> int:
    from .themes import available_packs, load_pack

    for pack_id in available_packs():
        pack = load_pack(pack_id)
        print(f"{pack.id:12s} {pack.name:22s} era={pack.era or '-':12s} "
              f"{len(pack.materials):>3} materials {len(pack.rules):>3} rules")
        if args.verbose:
            print(f"             {pack.description}")
            for rule in pack.rules:
                bits = "+".join(rule.ctx_all) or "-"
                print(f"               {rule.role:28s} [{bits:>28s}] -> {rule.material}")
    return 0


def _cmd_explain(args) -> int:
    from .roles.taxonomy import Ctx
    from .themes import explain, load_pack
    from .themes.request import MaterialRequest

    pack = load_pack(args.theme)
    flags = [f.strip() for f in (args.context or "").split(",") if f.strip()]
    request = MaterialRequest(role=args.role, context=Ctx.encode(flags))
    for row in explain(pack, [request]):
        print(json.dumps(row, indent=1))
    return 0


def _cmd_roles(args) -> int:
    from .roles.taxonomy import Ctx, ROLES

    print("roles:")
    for role in ROLES.values():
        print(f"  {role.id:28s} {role.reconstruct:12s} {role.description}")
    print("\ncontext flags:")
    for bit, name in sorted(Ctx.NAMES.items()):
        print(f"  {name:20s} 1<<{bit.bit_length() - 1:<2d}")
    return 0


def _cmd_sources(args) -> int:
    from .data import PLACES, RESTRICTED, commercial_sources

    print("cleared for commercial use:")
    for s in commercial_sources():
        print(f"  {s.id:16s} {s.license}")
        print(f"  {'':16s} {s.coverage}")
        print(f"  {'':16s} attribute as: {s.attribution}")
        if s.notes:
            print(f"  {'':16s} {s.notes}")
        print()
    print("excluded on licence grounds (do not wire these in):")
    for name, reason in RESTRICTED.items():
        print(f"  {name:16s} {reason}")
    print("\nnamed places (lidarworld fetch <id>):")
    for name, place in PLACES.items():
        print(f"  {name:20s} {place['description']}")
    return 0


def _cmd_fetch(args) -> int:
    from .data import PLACES, fetch_place
    from .data.fetch import resolve_tiles

    if args.place not in PLACES:
        raise SystemExit(f"unknown place {args.place!r}; try: {', '.join(PLACES)}")
    place = PLACES[args.place]
    tiles = resolve_tiles(place["bbox_wgs84"], prefer_project=place.get("project"))
    print(f"{args.place}: {place['description']}")
    print(f"{len(tiles)} tiles cover it; taking the {args.max_tiles} newest\n")
    for tile in tiles[: args.max_tiles]:
        print(f"  {tile['published']}  {tile['bytes'] / 1e6:7.1f} MB  {tile['title']}")
    if args.list_only:
        return 0

    def progress(done, total):
        if total:
            print(f"\r    {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)

    paths = fetch_place(args.place, args.out, max_tiles=args.max_tiles, progress=progress)
    print()
    for path in paths:
        print(f"  -> {path} ({path.stat().st_size / 1e6:.1f} MB)")
    crop = place.get("suggested_crop")
    if crop:
        print(f"\ncompile a block of it:\n  lidarworld compile {paths[0]} -o build/{args.place} "
              f"\\\n    --bbox {','.join(str(v) for v in crop)} --theme victorian --theme neon --sir")
    return 0


def _cmd_tiles(args) -> int:
    """What is on disk, and does it actually cover the area you want?"""
    from .data.tiles import TileIndex

    index = TileIndex.build(args.root, use_cache=not args.reindex)
    if not len(index):
        print(f"no LAS/LAZ tiles under {args.root}")
        return 1
    s = index.summary()
    lo_x, lo_y, hi_x, hi_y = index.bounds
    print(f"{s['tiles']} tiles / {s['points']:,} points / {s['extent_km2']} km2 / "
          f"{s['mean_density']:.2f} pts per m2")
    print(f"bounds  {lo_x:.1f},{lo_y:.1f} .. {hi_x:.1f},{hi_y:.1f}")

    if args.area:
        area = _parse_area(args.area)
        hits = index.query(area)
        covered = index.coverage(area)
        print(f"\narea    {area[0]:.1f},{area[1]:.1f} .. {area[2]:.1f},{area[3]:.1f}")
        print(f"        {len(hits)} tile(s), {covered:.0%} covered, "
              f"{sum(t.points for t in hits):,} points to read")
        if covered < 0.999:
            print(f"        {1 - covered:.0%} of it has no tile behind it")
        tiles = hits
    else:
        tiles = sorted(index.tiles, key=lambda t: -t.density)

    print()
    for tile in tiles[: args.limit]:
        print(f"  {Path(tile.path).name:38s} {tile.points:>11,} pts  "
              f"{tile.density:6.2f}/m2  at {tile.minx:.0f},{tile.miny:.0f}")
    if len(tiles) > args.limit:
        print(f"  ... {len(tiles) - args.limit} more")
    return 0


def _cmd_adapters(args) -> int:
    from .ingest import adapters

    for name, (exts, _, description) in sorted(adapters().items()):
        print(f"  {name:8s} {'/'.join(exts):28s} {description}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lidarworld",
        description="Compile LiDAR point clouds into themeable, engine-agnostic worlds.")
    sub = parser.add_subparsers(dest="command", required=True)

    c = sub.add_parser("compile", help="point cloud -> Spatial IR + backend output")
    c.add_argument("inputs", nargs="+",
                   help="las/laz/bin/pcd/ply/xyz files, or a directory to index")
    c.add_argument("-o", "--out", default="build/world")
    c.add_argument("-n", "--name", default=None)
    c.add_argument("-t", "--theme", action="append",
                   help="theme pack id or path (repeatable; first is used for glTF)")
    c.add_argument("--adapter", default=None, help="force an ingest adapter")
    c.add_argument("--tile", type=float, default=0.25, help="facade tile size in metres")
    c.add_argument("--terrain-cell", type=float, default=1.0)
    c.add_argument("--plane-voxel", type=float, default=0.6)
    c.add_argument("--decimate", type=int, default=1)
    c.add_argument("--footprints", default=None,
                   help="authoritative building footprints: a GeoJSON path, or a "
                        "layer id (e.g. 'denver') fetched for the compiled extent")
    c.add_argument("--crop", type=float, default=None,
                   help="take a centred square of this many metres; needs no CRS "
                        "knowledge, so it works on whatever tile you were given")
    c.add_argument("--area", default=None,
                   help="x,y,size in the source CRS: pick tiles out of an indexed "
                        "directory and crop to that square. Overridden by --bbox.")
    c.add_argument("--bbox", default=None,
                   help="crop to minx,miny,maxx,maxy in the source CRS "
                        "(a real tile is ~1.5 km square; a block is ~400 m)")
    c.add_argument("--gltf", action="store_true", help="also export materialised glTF")
    c.add_argument("--cityjson", action="store_true",
                   help="also export CityJSON 1.1 (CityGML semantics)")
    c.add_argument("--sir", action="store_true",
                   help="also export Spatial IR v0.1 (spec/schema), with "
                        "epistemic state derived from measured evidence")
    c.add_argument("--no-openings", action="store_true")
    c.add_argument("--no-textures", action="store_true")
    c.add_argument("--no-points", action="store_true")
    c.add_argument("-q", "--quiet", action="store_true")
    c.set_defaults(func=_cmd_compile)

    v = sub.add_parser("validate",
                       help="re-simulate a scan against the reconstruction and score it")
    v.add_argument("world", help="path to a .lwir archive")
    v.add_argument("--scan", required=True, help="the observed scan to compare against")
    v.add_argument("--sensor", default=None, help="sensor position x,y,z in world coordinates")
    v.add_argument("--resolution", type=float, default=0.25,
                   help="voxel size for the fallback backend only")
    v.add_argument("--backend", default="auto", choices=("auto", "embree", "voxel"),
                   help="embree is exact (needs Open3D); voxel is the numpy fallback")
    v.add_argument("--tolerance", type=float, default=0.35, help="range agreement in metres")
    v.add_argument("--max-rays", type=int, default=40000)
    v.add_argument("--limit", type=int, default=10)
    v.add_argument("--write-back", action="store_true",
                   help="fold measured consistency into node confidence and re-save")
    v.set_defaults(func=_cmd_validate)

    i = sub.add_parser("inspect", help="show a .lwir archive's manifest and graph")
    i.add_argument("path")
    i.add_argument("--graph", action="store_true")
    i.add_argument("--limit", type=int, default=25)
    i.set_defaults(func=_cmd_inspect)

    t = sub.add_parser("themes", help="list theme packs")
    t.add_argument("-v", "--verbose", action="store_true")
    t.set_defaults(func=_cmd_themes)

    e = sub.add_parser("explain", help="trace how a material request resolves")
    e.add_argument("--theme", default="victorian")
    e.add_argument("--role", default="surface.wall.vertical")
    e.add_argument("--context", default="", help="comma-separated context flags")
    e.set_defaults(func=_cmd_explain)

    r = sub.add_parser("roles", help="print the role taxonomy and context flags")
    r.set_defaults(func=_cmd_roles)

    so = sub.add_parser("sources", help="list LiDAR sources cleared for commercial use")
    so.set_defaults(func=_cmd_sources)

    f = sub.add_parser("fetch", help="download public-domain tiles covering a named place")
    f.add_argument("place", help="a place id from `lidarworld sources`")
    f.add_argument("-o", "--out", default="data/real")
    f.add_argument("--max-tiles", type=int, default=1)
    f.add_argument("--list-only", action="store_true")
    f.set_defaults(func=_cmd_fetch)

    ti = sub.add_parser("tiles", help="index a directory of tiles and query it")
    ti.add_argument("root", help="directory to scan for las/laz tiles")
    ti.add_argument("--area", default=None,
                    help="x,y,size in the source CRS: report which tiles cover it")
    ti.add_argument("--limit", type=int, default=20)
    ti.add_argument("--reindex", action="store_true", help="ignore the cached index")
    ti.set_defaults(func=_cmd_tiles)

    a = sub.add_parser("adapters", help="list ingest adapters")
    a.set_defaults(func=_cmd_adapters)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

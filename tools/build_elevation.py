"""Measure a Helsinki block, then build a clean one from the measurements.

    python tools/build_elevation.py --subtile data/helsinki/mesh/672496/673497d1

Not a repair and not a match. The reality mesh supplies numbers -- storey height,
bay width, window size, colour, roof height -- and the CityGML footprint supplies
a straight survey line to build on. The geometry is then constructed: quads with
openings punched through them and real reveals around the openings, which is depth
the source mesh never contained. A plane cannot droop, so there is nothing to
de-warp.

With `--match-material` (the default) the wall's measured colour and coursing are
matched against a CC0 photographed-material library (Poly Haven), and the winner
is applied instead of the procedural generator. The macro supplies identity; the
library texture supplies structure. Without `--no-match` the pipeline falls back
to procedural `stone_block`.

With `--vary` (the default) each building gets its own jittered DNA so the block
reads as a neighbourhood rather than a stamp. The variation budget is the
measurement's own uncertainty. `--no-vary` stamps every building identically.

Writes a glTF and a JSON record in which every number carries its provenance,
including the one that is assumed rather than measured.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidarworld.backends import gltf_textured                # noqa: E402
from lidarworld.data import helsinki                          # noqa: E402
from lidarworld.features import facade as facade_mod          # noqa: E402
from lidarworld.features import openings as openings_mod      # noqa: E402
from lidarworld.features import repair as repair_mod          # noqa: E402
from lidarworld.ingest import citygml, objmesh                # noqa: E402
from lidarworld.reconstruct import elevation                  # noqa: E402
from lidarworld.reconstruct import vary as vary_mod           # noqa: E402
from lidarworld.semantics import transfer                     # noqa: E402

from facade_repair import cell_buildings                      # noqa: E402
from textured_wall import bake, uv_for                        # noqa: E402

#: Surface kind -> (procedural generator, real-world repeat in metres, how the
#: building's own measured wall colour is tinted for it). Relief only reads if
#: the faces that catch light differ from the ones that do not, and a cornice is
#: the same render as the wall -- it is just in the light.
SURFACES = {
    "wall":    ("stone_block", 0.55, 1.00),
    "plinth":  ("stone_block", 0.85, 0.72),
    "string":  ("stone_block", 0.55, 1.05),
    "cornice": ("stone_block", 0.55, 1.08),
    "reveal":  ("stone_block", 0.55, 0.82),
    "sill":    ("concrete",    0.50, 1.14),
    "frame":   ("wood_plank",  0.40, 1.25),
    "roof":    ("roof_tile",   0.60, 0.55),
}
GLAZED = {"glass": ("glass", 1.2, 1.10), "door": ("wood_plank", 0.6, 0.90)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subtile", default="data/helsinki/mesh/672496/673497d1")
    ap.add_argument("--gml", default="data/helsinki/citygml/"
                                     "Helsinki3D_CityGML_Kalasatama_20190326.gml")
    ap.add_argument("--px-per-m", type=float, default=48.0)
    ap.add_argument("--buildings", type=int, default=6,
                    help="how many of the block's buildings to build")
    ap.add_argument("--tile-px-per-m", type=float, default=300.0)
    ap.add_argument("--match-material", dest="match", action="store_true",
                    default=True, help="match walls to CC0 photographed textures")
    ap.add_argument("--no-match", dest="match", action="store_false",
                    help="use procedural textures only")
    ap.add_argument("--texture-library", default="data/textures/polyhaven")
    ap.add_argument("--texture-limit", type=int, default=24)
    ap.add_argument("--vary", dest="vary", action="store_true", default=True,
                    help="jitter each building's DNA for variety")
    ap.add_argument("--no-vary", dest="vary", action="store_false")
    ap.add_argument("--vary-strength", type=float, default=1.0)
    ap.add_argument("-o", "--out", default="build/helsinki/elevation.gltf")
    args = ap.parse_args()

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    subtile = Path(args.subtile)
    tile = "".join(c for c in subtile.name if c.isdigit())[:6]
    print(f"reading {subtile}", flush=True)
    mesh, _ = objmesh.drop_webbing(
        objmesh.merge(objmesh.read_directory(subtile)))
    lo, _ = mesh.bounds
    offset = np.array(helsinki.local_offset(tile, lo))

    print(f"reading {Path(args.gml).name}", flush=True)
    buildings = citygml.read_buildings(args.gml)
    index = transfer.index_footprints(buildings)
    registration = transfer.register(mesh, index, proposed=offset)
    print(f"  registered: peak {registration.score:.3f} vs "
          f"{registration.runner_up:.3f}, residual {registration.residual_m} m, "
          f"unique={registration.unique}", flush=True)
    joined = transfer.transfer(mesh, index, registration)

    # The facade the block's rhythm is measured from: the one with the most.
    best, score = None, -1.0
    for slab in facade_mod.facade_slabs(mesh)[:8]:
        crop = facade_mod.rectify_mesh(mesh, slab, px_per_m=args.px_per_m)
        if crop is None or crop.covered < 0.45:
            continue
        value = float(np.max(facade_mod.rhythm_profile(
            crop.image, crop.px_per_m))) * crop.covered
        if value > score:
            best, score = crop, value
    if best is None:
        raise SystemExit("no usable facade to measure")
    grid = openings_mod.lattice(best)
    straightened, _ = openings_mod.dewarp(best, grid)
    flat = facade_mod.Facade(**{**best.__dict__, "image": straightened})
    print(f"measured on {best.width_m:.0f} x {best.height_m:.0f} m: "
          f"bay {grid.bay_m:.2f} m ({grid.bay_strength:.2f}), "
          f"storey {grid.storey_m:.2f} m ({grid.storey_strength:.2f})", flush=True)

    groups = cell_buildings(flat, grid, index, registration.offset)
    stack = repair_mod.cell_stack(flat, grid)
    counts = {int(g): int((groups == g).sum()) for g in np.unique(groups)}
    donor = max((g for g in counts if g >= 0), key=lambda g: counts[g], default=None)
    if donor is None:
        raise SystemExit("no lattice point landed on a building")
    model = repair_mod.canonical(
        repair_mod._subset(stack, np.flatnonzero(groups == donor)))
    print(f"  average bay from building {donor}, {counts[donor]} bays", flush=True)

    # --- CC0 material match ------------------------------------------------
    matched_tile = None
    match_record = None
    if args.match:
        from lidarworld.data import textures as tex_mod                  # noqa
        from lidarworld.features import match as match_mod               # noqa
        from lidarworld.features import openings as openings_mod_2       # noqa
        # Build a masonry patch from the wall between windows.
        from texture_match import masonry_patch                          # noqa
        patch, patch_m = masonry_patch(flat, grid)
        wall_desc = match_mod.describe(patch, metres_across=patch_m,
                                       source="helsinki",
                                       source_px_per_m=flat.resolution_px_per_m)
        if not wall_desc.resolves_coursing:
            print(f"  coursing NOT resolved ({wall_desc.px_per_course:.1f} px); "
                  f"matching on colour and roughness", flush=True)

        catalogue = tex_mod.polyhaven_walls(limit=args.texture_limit)
        print(f"matching against {len(catalogue)} CC0 materials", flush=True)
        from PIL import Image as PILImage
        library = []
        for material in catalogue:
            got = tex_mod.fetch_albedo(material, args.texture_library)
            if got is None or got.albedo is None:
                continue
            img = np.asarray(PILImage.open(got.albedo).convert("RGB"))
            desc = match_mod.describe(img, metres_across=got.metres,
                                      source=got.key, delight=False)
            library.append((got.key, desc))
        if library:
            ranked = match_mod.rank(wall_desc, library)
            winner_key, winner_dist, winner_desc = ranked[0]
            winner = next(m for m in catalogue if m.key == winner_key)
            print(f"  matched: {winner.name} ({winner_key}), "
                  f"distance {winner_dist:.3f}", flush=True)
            chosen_img = np.asarray(PILImage.open(winner.albedo).convert("RGB"))
            tinted = match_mod.recolour(chosen_img, wall_desc)
            tile_path = out_dir / "matched_tile.png"
            PILImage.fromarray((tinted * 255).astype(np.uint8)).save(tile_path)
            matched_tile = (tile_path, winner.metres)
            match_record = {"key": winner_key, "name": winner.name,
                            "distance": round(winner_dist, 4),
                            "tile_m": winner.metres,
                            "ranking": [{"key": k, "distance": round(d, 4)}
                                        for k, d, _ in ranked[:5]]}
        else:
            print("  no CC0 material could be fetched; using procedural", flush=True)

    # Every building the mesh actually touched, tallest first: those are the ones
    # a walker sees.
    touched = [b for b in np.unique(joined["building"]) if b >= 0]
    ordered = sorted(touched, key=lambda b: -(index.roof_z[b] - index.ground_z[b])
                     if np.isfinite(index.roof_z[b]) else 0.0)

    faces, records = [], []
    for slot in ordered[:args.buildings]:
        base_z = float(index.ground_z[slot])
        top_z = float(index.roof_z[slot])
        if not (np.isfinite(base_z) and np.isfinite(top_z)) or top_z - base_z < 6.0:
            continue
        rings = transfer._footprint_rings(index.buildings[slot])
        if not rings:
            continue
        dna = elevation.measure(flat, grid, model.image, base_z=base_z,
                                top_z=top_z, support=model.support > 0)
        if args.vary:
            dna = vary_mod.vary(dna, int(slot), strength=args.vary_strength)
        built = elevation.build_detailed(max(rings, key=len), dna)
        if not len(built.quads):
            continue

        # Tiles per building. When a CC0 match succeeded, the wall, plinth,
        # string, cornice and reveal all use the recoloured photograph; only
        # glazing, sills, frames and roofs stay procedural. Otherwise
        # everything is procedural as before.
        w = np.asarray(dna.wall_rgb, dtype=float)
        g = np.asarray(dna.window_rgb, dtype=float)
        tiles = {}

        if matched_tile is not None:
            # The matched tile is already recoloured to the block's measured
            # colour. Per-building variation only needs a slight tint shift.
            from PIL import Image as PILImage                        # noqa
            from lidarworld.features import match as match_mod_2     # noqa
            base_img = np.asarray(PILImage.open(matched_tile[0]).convert("RGB"))
            base_img = base_img.astype(np.float64) / 255.0
            tile_m = matched_tile[1]
            for kind in ("wall", "plinth", "string", "cornice", "reveal"):
                _, _, tint_factor = SURFACES[kind]
                tinted = np.clip(base_img * tint_factor, 0.0, 1.0)
                path = out_dir / f"tile_{slot}_{kind}.png"
                PILImage.fromarray((tinted * 255).astype(np.uint8)).save(path)
                tiles[kind] = (path, tile_m)

        # Procedural tiles for surfaces the photograph does not cover, or all
        # surfaces when no match was made.
        for kind, (material, tile_m, tint) in {**SURFACES, **GLAZED}.items():
            if kind in tiles:
                continue                          # already filled by CC0 match
            base_colour = g if kind in GLAZED else w
            path = out_dir / f"tile_{slot}_{kind}.png"
            bake(material, tile_m, tuple(np.clip(base_colour * tint, 0, 1)),
                 px_per_m=args.tile_px_per_m, seed=int(slot) % 97, out=path)
            tiles[kind] = (path, tile_m)

        for quad, kind in zip(built.quads, built.kinds):
            path, tile_m = tiles.get(kind, tiles["wall"])
            ring = np.asarray(quad, dtype=float)
            faces.append(gltf_textured.Face(
                ring=ring, uv=uv_for(ring, tile_m), image=path.name, kind=kind,
                surface_id=f"b{slot}_{kind}", building_id=str(slot)))
        records.append({"building": int(slot),
                        "gml_id": index.buildings[slot].gml_id,
                        "varied": args.vary,
                        **built.report})
        print(f"  built {slot}: {top_z - base_z:.1f} m, "
              f"{built.report['quads']} quads, "
              f"{built.report['openings']} openings"
              f"{' (varied)' if args.vary else ''}", flush=True)

    # Ground, so the block is standing on something rather than in space.
    if faces:
        pts = np.vstack([f.ring for f in faces])
        pad = 22.0
        lo2, hi2 = pts[:, :2].min(axis=0) - pad, pts[:, :2].max(axis=0) + pad
        z = float(np.percentile(pts[:, 2], 1)) - 0.15
        faces.append(gltf_textured.Face(
            ring=np.array([[lo2[0], lo2[1], z], [hi2[0], lo2[1], z],
                           [hi2[0], hi2[1], z], [lo2[0], hi2[1], z]]),
            kind="ground", surface_id="ground"))
        gltf_textured.FALLBACK["ground"] = (0.30, 0.30, 0.31, 1.0)

    if not faces:
        raise SystemExit("nothing was built")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    gltf_textured.export(faces, out, image_root=out_dir)
    summary = {
        "registration": registration.to_record(),
        "measured_on": {"facade_m": [round(best.width_m, 1), round(best.height_m, 1)],
                        "source_px_per_m": round(best.resolution_px_per_m, 1)},
        "lattice": grid.to_record(),
        "average_bay": model.to_record(),
        "donor_building": int(donor),
        "material_match": match_record,
        "varied": args.vary,
        "vary_strength": args.vary_strength if args.vary else None,
        "buildings": records,
        "faces": len(faces),
    }
    out.with_suffix(".json").write_text(json.dumps(summary, indent=1, default=float))
    print(f"\n{out}\n{out.with_suffix('.json')}")
    print(f"{len(faces)} faces across {len(records)} buildings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

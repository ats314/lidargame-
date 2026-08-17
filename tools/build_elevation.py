"""Measure a Helsinki block, then build a clean one from the measurements.

    python tools/build_elevation.py --subtile data/helsinki/mesh/672496/673497d1

Not a repair and not a match. The reality mesh supplies numbers -- storey height,
bay width, window size, colour, roof height -- and the CityGML footprint supplies
a straight survey line to build on. The geometry is then constructed: quads with
openings punched through them and real reveals around the openings, which is depth
the source mesh never contained. A plane cannot droop, so there is nothing to
de-warp.

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
        built = elevation.build_detailed(max(rings, key=len), dna)
        if not len(built.quads):
            continue
        # Tiles per building, so six buildings are not one material. The
        # building's own de-lit measured colour tints every surface it has.
        w = np.asarray(dna.wall_rgb, dtype=float)
        g = np.asarray(dna.window_rgb, dtype=float)
        tiles = {}
        for kind, (material, tile_m, tint) in {**SURFACES, **GLAZED}.items():
            base = g if kind in GLAZED else w
            path = out_dir / f"tile_{slot}_{kind}.png"
            bake(material, tile_m, tuple(np.clip(base * tint, 0, 1)),
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
                        **built.report})
        print(f"  built {slot}: {top_z - base_z:.1f} m, "
              f"{built.report['quads']} quads, "
              f"{built.report['openings']} openings", flush=True)

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
        "buildings": records,
        "faces": len(faces),
    }
    out.with_suffix(".json").write_text(json.dumps(summary, indent=1, default=float))
    print(f"\n{out}\n{out.with_suffix('.json')}")
    print(f"{len(faces)} faces across {len(records)} buildings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

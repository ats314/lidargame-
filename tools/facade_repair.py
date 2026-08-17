"""Fix a warped facade against its own average bay, per building.

    python tools/facade_repair.py --subtile data/helsinki/mesh/672496/673497d1

Five panels:

    macro        the photograph as the reality mesh supplies it
    de-warped    the global droop removed, where removing it measurably helps
    canonical    the average bay of each building, assembled by median vote
    repaired     the disagreeing bays replaced by their building's average
    provenance   which pixels are no longer a measurement of that wall

The reason this needs the CityGML join and not just the lattice: a rectified slab
runs along a street and a street is several buildings. Averaging across the slab
produces the majority building's window, and repairing with it pastes one
building's architecture onto its neighbour. The join labels every lattice point
with the building it stands on, so each building is averaged against itself.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidarworld.data import helsinki                       # noqa: E402
from lidarworld.features import facade as facade_mod        # noqa: E402
from lidarworld.features import openings as openings_mod    # noqa: E402
from lidarworld.features import repair as repair_mod        # noqa: E402
from lidarworld.ingest import citygml, objmesh              # noqa: E402
from lidarworld.semantics import transfer                   # noqa: E402

from facade_openings import label                           # noqa: E402


def cell_buildings(facade, grid, index, offset) -> np.ndarray:
    """Which CityGML building each lattice point stands on.

    Probed one metre behind the wall plane rather than on it: a photogrammetric
    wall bulges outward past the survey line, so a point taken on the surface can
    land in the street and come back with no building at all.
    """
    labels = []
    for u_m, v_m in grid.points:
        world = (facade.origin_xyz + facade.u_axis * u_m + facade.v_axis * v_m
                 + offset)
        probe = world[:2] - facade.normal[:2] * 1.0
        labels.append(int(index.lookup(probe[None, :])[0]))
    return np.asarray(labels, dtype=np.int64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subtile", default="data/helsinki/mesh/672496/673497d1")
    ap.add_argument("--gml", default="data/helsinki/citygml/"
                                     "Helsinki3D_CityGML_Kalasatama_20190326.gml")
    ap.add_argument("--px-per-m", type=float, default=48.0)
    ap.add_argument("--worst", type=float, default=None,
                    help="repair this fraction of the least-agreeing bays, "
                         "e.g. 0.1 for the worst 10%%")
    ap.add_argument("--no-join", action="store_true",
                    help="average the whole slab together, which is wrong on a "
                         "street of several buildings; here to show the difference")
    ap.add_argument("-o", "--out", default="build/helsinki/repair.png")
    args = ap.parse_args()

    subtile = Path(args.subtile)
    tile = "".join(c for c in subtile.name if c.isdigit())[:6]
    print(f"reading {subtile}", flush=True)
    mesh, _ = objmesh.drop_webbing(
        objmesh.merge(objmesh.read_directory(subtile)))

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
        raise SystemExit("no usable facade in this subtile")

    grid = openings_mod.lattice(best)
    print(f"facade {best.width_m:.0f} x {best.height_m:.0f} m, "
          f"{best.resolution_px_per_m:.1f} px/m source", flush=True)
    print(f"  lattice {grid.to_record()}", flush=True)

    straightened, warp = openings_mod.dewarp(best, grid)
    print(f"  de-warp {json.dumps({k: v for k, v in warp.items() if isinstance(v, dict)})}",
          flush=True)
    flat = facade_mod.Facade(**{**best.__dict__, "image": straightened})

    groups = None
    if not args.no_join:
        lo, _ = mesh.bounds
        offset = np.array(helsinki.local_offset(tile, lo))
        print(f"reading city model {Path(args.gml).name}", flush=True)
        buildings = citygml.read_buildings(args.gml)
        index = transfer.index_footprints(buildings)
        groups = cell_buildings(flat, grid, index, offset)
        found = {int(g): int((groups == g).sum()) for g in np.unique(groups)}
        print(f"  lattice points by building: {found}", flush=True)

    mended, generated, report = repair_mod.repair(
        flat, grid, groups=groups, worst=args.worst)
    print(json.dumps(report, indent=1), flush=True)

    stack = repair_mod.cell_stack(flat, grid)
    tiles = []
    if groups is None:
        tiles = [repair_mod.canonical(stack).image]
    else:
        for value in np.unique(groups):
            members = np.flatnonzero(groups == value)
            if len(members) >= repair_mod.MIN_CELLS:
                tiles.append(repair_mod.canonical(
                    repair_mod._subset(stack, members)).image)
    # The average bays, tiled up the panel so they are visible next to a facade
    # forty metres wide.
    board = np.zeros_like(mended, dtype=np.float64)
    if tiles:
        strip = np.vstack([np.tile(t, (max(1, board.shape[0] // (t.shape[0] * len(tiles))) + 1,
                                       1, 1)) for t in tiles])
        reps = board.shape[1] // strip.shape[1] + 1
        wide = np.tile(strip, (1, reps, 1))[:board.shape[0], :board.shape[1]]
        board = wide * 255.0

    from PIL import Image
    provenance = np.zeros((*generated.shape, 3), dtype=np.uint8)
    provenance[..., 1] = (best.image.mean(axis=2) * 0.6).astype(np.uint8)
    provenance[generated] = (255, 90, 40)

    rows = best.image.shape[0]
    band = slice(int(0.08 * rows), int(0.70 * rows))
    panels = [
        label(best.image[band], "macro", "as the mesh supplies it"),
        label(straightened[band], "de-warped",
              f"vertical {warp['vertical'].get('rms_m')} m  ·  "
              f"strength {warp['vertical'].get('strength_before')} -> "
              f"{warp['vertical'].get('strength_after')}"),
        label(np.clip(board[band], 0, 255).astype(np.uint8), "average bay",
              f"{len(tiles)} building(s), median vote over "
              f"{report.get('cells', 0)} bays"),
        label(mended[band], "repaired",
              f"{report.get('cells_replaced', 0)} of {report.get('cells', 0)} bays  ·  "
              f"{100 * report.get('generated_pixel_fraction', 0):.1f}% of pixels"),
        label(provenance[band], "provenance",
              "orange = median of other bays, no longer measured"),
    ]
    height = min(p.shape[0] for p in panels)
    sheet = np.hstack([p[:height] for p in panels])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet).save(out)
    out.with_suffix(".json").write_text(json.dumps(
        {"lattice": grid.to_record(), "dewarp": warp, "repair": report},
        indent=1, default=float))
    print(f"\n{out}  ({sheet.shape[1]}x{sheet.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

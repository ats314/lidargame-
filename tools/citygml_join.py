"""Georeference a reality mesh against a city model, and stamp it with semantics.

    python tools/citygml_join.py --subtile data/helsinki/mesh/672496/672496a1

A reality mesh has the surface and none of the meaning: 42 cm triangles, 7.7 cm
texels, no building ids, no surface classes, and local coordinates whose origin
is not written down anywhere. A semantic city model has the opposite problem.
Helsinki publishes both for the same city, so this joins them.

The registration is derived and then checked, not asserted. The tile name implies
a translation; a search around it scores each candidate by the fraction of mesh
*wall* cells that land inside a CityGML footprint, and reports the peak, the
runner-up and how far the search had to move. A confident join is a sharp peak at
zero residual. Anything else is visible as a number rather than as a city quietly
sitting a few metres into the harbour.

What comes out per triangle is a building id and a surface class, and per
building a height residual against the city model -- the one cross-model check
this data supports.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidarworld.data import helsinki                      # noqa: E402
from lidarworld.features import facade as facade_mod        # noqa: E402
from lidarworld.features import openings as openings_mod    # noqa: E402
from lidarworld.ingest import citygml, objmesh              # noqa: E402
from lidarworld.semantics import transfer                   # noqa: E402


def storey_check(joined: dict, index: transfer.FootprintIndex, mesh,
                 *, px_per_m: float = 48.0, max_facades: int = 6) -> dict:
    """Does the detected storey period match the city model's storey count?

    This is the payoff of the join for opening detection, and the only
    independent evidence available about it. `openings.lattice` measures a
    vertical repeat from the mesh; the CityGML carries `Kerroksia`, a surveyed
    storey count. Height over count gives an expected storey height, and the two
    can be compared without either being ground truth for the other.

    The detected period is the thing under suspicion: on a Helsinki frontage it
    came out 5.35 m, which is nobody's storey and much more likely a double
    period from a facade whose windows alternate.
    """
    out = []
    for slab in facade_mod.facade_slabs(mesh)[:max_facades]:
        crop = facade_mod.rectify_mesh(mesh, slab, px_per_m=px_per_m)
        if crop is None or crop.covered < 0.45:
            continue
        mask, _ = openings_mod.reveal_mask(crop)
        grid = openings_mod.lattice(crop, mask=mask)
        if grid.storey_m <= 0:
            continue
        # Which building this facade belongs to: the crop's own origin, pushed
        # slightly inward along the wall normal so a bulging wall still lands on
        # the footprint it came from.
        centre = (crop.origin_xyz
                  + crop.u_axis * crop.width_m / 2.0
                  + crop.v_axis * crop.height_m / 2.0
                  + joined["report"]["_offset"])
        probe = centre[:2] - crop.normal[:2] * 1.0
        slot = int(index.lookup(probe[None, :])[0])
        record = {"facade_m": [round(crop.width_m, 1), round(crop.height_m, 1)],
                  "detected_storey_m": round(grid.storey_m, 2),
                  "storey_strength": round(grid.storey_strength, 3),
                  "building": slot}
        if slot >= 0:
            count = index.storeys[slot]
            ground = index.ground_z[slot]
            eaves, ridge = index.eaves_z[slot], index.roof_z[slot]
            if np.isfinite(count) and count >= 1 and np.isfinite(ground):
                record["model_storeys"] = int(count)
                # Two denominators, because the choice of one is most of the
                # answer. Ridge over storey count charges the whole attic to the
                # storeys; eaves over storey count is the top of the topmost
                # floor, which is what a storey height means.
                for name, top in (("to_eaves", eaves), ("to_ridge", ridge)):
                    if np.isfinite(top) and top > ground:
                        expected = (top - ground) / count
                        record[f"model_storey_m_{name}"] = round(float(expected), 2)
                        record[f"ratio_{name}"] = round(
                            float(grid.storey_m / expected), 2)
        out.append(record)
    matched = [r for r in out if "ratio_to_eaves" in r]
    summary = {"facades": out}
    if matched:
        ratios = np.array([r["ratio_to_eaves"] for r in matched])
        summary["median_ratio_to_eaves"] = round(float(np.median(ratios)), 2)
        ridge = np.array([r["ratio_to_ridge"] for r in matched
                          if "ratio_to_ridge" in r])
        if len(ridge):
            summary["median_ratio_to_ridge"] = round(float(np.median(ridge)), 2)
        summary["doubled"] = int((np.abs(ratios - 2.0) < 0.35).sum())
        summary["matched"] = int((np.abs(ratios - 1.0) < 0.25).sum())
        summary["note"] = ("ratio ~2 means the detector found every other floor; "
                           "~1 means it found the storey; the eaves ratio is the "
                           "one to read")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subtile", default="data/helsinki/mesh/672496/672496a1")
    ap.add_argument("--gml", default="data/helsinki/citygml/"
                                     "Helsinki3D_CityGML_Kalasatama_20190326.gml")
    ap.add_argument("--tile", default=None,
                    help="2 km tile code; inferred from the subtile path")
    ap.add_argument("--cell-m", type=float, default=transfer.CELL_M)
    ap.add_argument("--search-m", type=float, default=6.0)
    ap.add_argument("--storeys", action="store_true",
                    help="also check detected storey periods against the model")
    ap.add_argument("-o", "--out", default="build/helsinki/join.json")
    args = ap.parse_args()

    subtile = Path(args.subtile)
    tile = args.tile or "".join(c for c in subtile.name if c.isdigit())[:6]

    print(f"reading mesh {subtile}", flush=True)
    mesh, webbing = objmesh.drop_webbing(
        objmesh.merge(objmesh.read_directory(subtile)))
    lo, hi = mesh.bounds
    offset = np.array(helsinki.local_offset(tile, lo))
    print(f"  {mesh.triangles} triangles, local {np.round(lo[:2], 0)}.."
          f"{np.round(hi[:2], 0)}", flush=True)
    print(f"  tile {tile} implies offset {np.round(offset[:2], 0)}  ->  absolute "
          f"{np.round((lo + offset)[:2], 0)}..{np.round((hi + offset)[:2], 0)}",
          flush=True)

    print(f"reading city model {Path(args.gml).name}", flush=True)
    buildings = citygml.read_buildings(args.gml)
    index = transfer.index_footprints(buildings, cell_m=args.cell_m)
    print(f"  {len(buildings)} buildings, footprint raster "
          f"{index.ids.shape} at {args.cell_m} m", flush=True)

    print("registering", flush=True)
    registration = transfer.register(mesh, index, proposed=offset,
                                     search_m=args.search_m)
    r = registration.to_record()
    print(f"  peak {r['wall_cells_on_an_outline']:.3f} vs runner-up "
          f"{r['runner_up']:.3f}  residual {r['residual_m']} m  "
          f"unique={r['unique_peak']}  ({r['offsets_searched']} offsets)",
          flush=True)
    at_zero = registration.field.get((0.0, 0.0))
    if at_zero is not None:
        print(f"  score at the tile name's own offset: {at_zero:.3f}", flush=True)

    joined = transfer.transfer(mesh, index, registration)
    joined["report"]["_offset"] = registration.offset
    report = joined["report"]
    report["webbing"] = webbing
    report["subtile"] = str(subtile)
    report["tile"] = tile

    print(f"  stamped {100 * report['stamped_fraction']:.1f}% of triangles, "
          f"{100 * report['stamped_area_fraction']:.1f}% of area, "
          f"{report['buildings_touched']} buildings", flush=True)
    print(f"  classes {report['by_class']}", flush=True)
    print(f"  area    {report['area_by_class_m2']}", flush=True)
    print(f"  height agreement {report['height_agreement']}", flush=True)

    if args.storeys:
        print("checking storey periods against the model", flush=True)
        report["storeys"] = storey_check(joined, index, mesh)
        for row in report["storeys"]["facades"]:
            print(f"  {row}", flush=True)
        if "median_ratio_to_eaves" in report["storeys"]:
            print(f"  median ratio to eaves "
                  f"{report['storeys']['median_ratio_to_eaves']}, to ridge "
                  f"{report['storeys'].get('median_ratio_to_ridge')} "
                  f"(doubled {report['storeys']['doubled']}, "
                  f"matched {report['storeys']['matched']})", flush=True)

    groups = transfer.stamp_groups(mesh, joined)
    report["semantic_groups"] = len(groups)
    report["material_groups"] = len(mesh.groups)
    print(f"  {len(mesh.groups)} material groups -> {len(groups)} "
          f"building x class groups", flush=True)

    report.pop("_offset", None)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=float))
    print(f"\n{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

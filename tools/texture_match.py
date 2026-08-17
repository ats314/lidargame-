"""Fetch a CC0 material library, match it to a measured wall, apply the winner.

    python tools/texture_match.py

Ingest, match, apply. The library is Poly Haven because it publishes each
texture's real-world size, so its coursing can be measured in metres and compared
against a facade measured in metres. The wall is the same Helsinki frontage
everything else in this work was measured on.

Prints the ranking with the numbers behind it, because "this brick looks right"
is not checkable and "this brick courses at 0.072 m against the wall's 0.068 m"
is.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lidarworld.data import textures                        # noqa: E402
from lidarworld.features import facade as facade_mod        # noqa: E402
from lidarworld.features import match as match_mod          # noqa: E402
from lidarworld.features import openings as openings_mod    # noqa: E402
from lidarworld.ingest import objmesh                       # noqa: E402


def measured_wall(subtile: str, px_per_m: float = 48.0):
    """The rectified Helsinki frontage, de-warped, as the thing to match."""
    mesh, _ = objmesh.drop_webbing(
        objmesh.merge(objmesh.read_directory(subtile)))
    best, score = None, -1.0
    for slab in facade_mod.facade_slabs(mesh)[:8]:
        crop = facade_mod.rectify_mesh(mesh, slab, px_per_m=px_per_m)
        if crop is None or crop.covered < 0.45:
            continue
        value = float(np.max(facade_mod.rhythm_profile(
            crop.image, crop.px_per_m))) * crop.covered
        if value > score:
            best, score = crop, value
    if best is None:
        raise SystemExit("no usable facade")
    grid = openings_mod.lattice(best)
    straight, _ = openings_mod.dewarp(best, grid)
    return facade_mod.Facade(**{**best.__dict__, "image": straight}), grid


def masonry_patch(crop, grid) -> tuple[np.ndarray, float]:
    """A piece of blank wall between windows: masonry, not architecture.

    Matching against the whole frontage would match the window rhythm, which is
    3.77 m and belongs to the building rather than to its material. The material
    lives in the pier between two windows.
    """
    top, bottom = grid.band if grid.band[1] > grid.band[0] else (0, crop.image.shape[0])
    band = crop.image[top:bottom]
    px = crop.px_per_m
    width_px = max(32, int(round(min(grid.bay_m, 2.0) * px)))
    grey = band.astype(np.float64).mean(axis=2)
    valid = band.sum(axis=2) > 0
    # The flattest fully-covered window in the crop: least architecture in it.
    best, best_score = 0, np.inf
    for x in range(0, max(1, band.shape[1] - width_px), width_px // 2):
        piece = grey[:, x:x + width_px]
        if valid[:, x:x + width_px].mean() < 0.98:
            continue
        score = float(np.std(piece))
        if score < best_score:
            best_score, best = score, x
    patch = band[:, best:best + width_px]
    return patch, width_px / px


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subtile", default="data/helsinki/mesh/672496/673497d1")
    ap.add_argument("--library", default="data/textures/polyhaven")
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("-o", "--out", default="build/helsinki/match")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print("measuring the wall", flush=True)
    crop, grid = measured_wall(args.subtile)
    patch, patch_m = masonry_patch(crop, grid)
    wall = match_mod.describe(patch, metres_across=patch_m, source="helsinki",
                              source_px_per_m=crop.resolution_px_per_m)
    print(f"  masonry patch {patch_m:.2f} m across at "
          f"{crop.resolution_px_per_m:.1f} px/m source", flush=True)
    print(f"  {json.dumps(wall.to_record())}", flush=True)
    if not wall.resolves_coursing:
        print(f"  coursing NOT resolved: {wall.px_per_course:.1f} source px per "
              f"course, under {match_mod.MIN_COURSE_PX}. Matching on colour and "
              f"roughness only.", flush=True)

    print(f"fetching up to {args.limit} CC0 wall materials", flush=True)
    catalogue = textures.polyhaven_walls(limit=args.limit)
    print(f"  {len(catalogue)} candidates listed", flush=True)

    from PIL import Image
    library, records = [], []
    for material in catalogue:
        got = textures.fetch_albedo(material, args.library)
        if got is None or got.albedo is None:
            continue
        image = np.asarray(Image.open(got.albedo).convert("RGB"))
        descriptor = match_mod.describe(image, metres_across=got.metres,
                                        source=got.key, delight=False)
        library.append((got.key, descriptor))
        records.append({**got.to_record(), **descriptor.to_record()})
        print(f"  {got.key:34s} {got.metres:5.2f} m  "
              f"course {descriptor.course_h_m:.3f} x {descriptor.course_v_m:.3f} m  "
              f"rough {descriptor.roughness:.4f}", flush=True)
    if not library:
        raise SystemExit("no material could be fetched")

    ranked = match_mod.rank(wall, library)
    print("\nnearest first:", flush=True)
    for key, score, descriptor in ranked[:8]:
        print(f"  {score:6.3f}  {key:34s} "
              f"course {descriptor.course_h_m:.3f} m  "
              f"rough {descriptor.roughness:.4f}", flush=True)

    winner = next(m for m in catalogue if m.key == ranked[0][0])
    chosen = np.asarray(Image.open(winner.albedo).convert("RGB"))
    tinted = match_mod.recolour(chosen, wall)
    Image.fromarray((tinted * 255).astype(np.uint8)).save(
        out.parent / "matched_tile.png")

    report = {"wall": wall.to_record(),
              "patch_m": round(patch_m, 3),
              "library": textures.describe(),
              "candidates": records,
              "ranking": [{"key": k, "distance": round(s, 4)}
                          for k, s, _ in ranked],
              "chosen": {**winner.to_record(),
                         "tile_png": "matched_tile.png"}}
    (out.parent / "match.json").write_text(json.dumps(report, indent=1))
    print(f"\nchosen: {winner.name} ({winner.key}), {winner.metres:.2f} m tile, "
          f"{textures.LIBRARIES[winner.library].licence}", flush=True)
    print(f"{out.parent / 'matched_tile.png'}\n{out.parent / 'match.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

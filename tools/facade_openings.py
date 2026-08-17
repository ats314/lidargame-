"""Draw what the opening detector found on top of the facade it found it on.

    python tools/facade_openings.py --subtile data/helsinki/mesh/672496/673497d1

Four panels for one rectified facade:

    macro       the photograph, as the mesh supplies it
    depth       the rectifier's depth buffer, which was being thrown away
    reveals     where depth steps back far enough to be an opening
    lattice     the detected bay and storey grid, on the macro

This exists because a storey period cannot be checked by a number alone. The
join against Helsinki's city model says a seven-storey block is 3.85 m per
storey; the detector says 2.4-3.1 m, and a consistent 0.67 ratio across five
buildings is either a register that undercounts floors or a detector locking
onto a sub-storey harmonic. Both are plausible and no metric in the repo
separates them, which is exactly the situation the render exists for -- the fog
drowning a block at 63% opacity and 1,315 phantom roof windows both showed up
here and in no measurement.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidarworld.features import facade as facade_mod       # noqa: E402
from lidarworld.features import frequency                   # noqa: E402
from lidarworld.features import openings as openings_mod    # noqa: E402
from lidarworld.ingest import objmesh                       # noqa: E402


def label(image: np.ndarray, text: str, sub: str = "") -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont
    pil = Image.fromarray(image).convert("RGB")
    pad = 46 if sub else 30
    out = Image.new("RGB", (pil.width, pil.height + pad), (14, 17, 22))
    out.paste(pil, (0, pad))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default(size=17)
        small = ImageFont.load_default(size=13)
    except TypeError:
        font = small = ImageFont.load_default()
    draw.text((10, 6), text, fill=(232, 234, 236), font=font)
    if sub:
        draw.text((10, 26), sub, fill=(139, 149, 161), font=small)
    return np.asarray(out)


def ramp(field: np.ndarray) -> np.ndarray:
    """A depth buffer as an image, with the uncovered pixels obviously uncovered."""
    valid = np.isfinite(field)
    out = np.zeros((*field.shape, 3), dtype=np.uint8)
    if valid.any():
        # Stretched over the middle of the distribution, not the full range: the
        # pavement and the roof are metres away from the wall plane and would
        # compress the whole facade into two grey levels.
        lo, hi = np.percentile(field[valid], [15, 85])
        scaled = np.zeros(field.shape)
        scaled[valid] = np.clip((field[valid] - lo) / max(hi - lo, 1e-6), 0, 1)
        grey = (scaled * 235 + 20).astype(np.uint8)
        out[valid] = np.stack([grey, grey, grey], axis=-1)[valid]
    out[~valid] = (40, 12, 12)                      # never seen, not "near"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subtile", default="data/helsinki/mesh/672496/673497d1")
    ap.add_argument("--px-per-m", type=float, default=48.0)
    ap.add_argument("--pick", type=int, default=None,
                    help="facade index; default is the one with the most rhythm")
    ap.add_argument("--delight", action="store_true",
                    help="show the de-lit albedo instead of the raw photograph")
    ap.add_argument("-o", "--out", default="build/helsinki/openings.png")
    args = ap.parse_args()

    print(f"reading {args.subtile}", flush=True)
    mesh, _ = objmesh.drop_webbing(
        objmesh.merge(objmesh.read_directory(args.subtile)))

    crops = []
    for slab in facade_mod.facade_slabs(mesh)[:8]:
        crop = facade_mod.rectify_mesh(mesh, slab, px_per_m=args.px_per_m)
        if crop is None or crop.covered < 0.45:
            continue
        rhythm = float(np.max(facade_mod.rhythm_profile(crop.image, crop.px_per_m)))
        crops.append((rhythm * crop.covered, rhythm, crop))
        print(f"  candidate {len(crops) - 1}: {crop.width_m:.0f}x{crop.height_m:.0f} m "
              f"covered {100 * crop.covered:.0f}%  rhythm {rhythm:.2f}", flush=True)
    if not crops:
        raise SystemExit("no usable facade in this subtile")
    crop = (crops[args.pick][2] if args.pick is not None
            else max(crops, key=lambda c: c[0])[2])

    mask, depth_report = openings_mod.reveal_mask(crop)
    grid = openings_mod.lattice(crop, mask=mask)
    print(f"chosen {crop.width_m:.1f} x {crop.height_m:.1f} m, "
          f"{crop.resolution_px_per_m:.1f} px/m source", flush=True)
    print(f"  depth   {depth_report}", flush=True)
    print(f"  lattice {grid.to_record()}", flush=True)

    macro = crop.image.astype(np.float64) / 255.0
    if args.delight:
        macro, _, _ = frequency.delight(macro, px_per_m=crop.px_per_m)
    base = (np.clip(macro, 0, 1) * 255).astype(np.uint8)

    overlay = base.copy()
    overlay[mask] = (overlay[mask] * 0.35 + np.array([0, 210, 255]) * 0.65
                     ).astype(np.uint8)

    lines = base.copy()
    for u in grid.bays:
        col = int(round(u * crop.px_per_m))
        if 0 <= col < lines.shape[1]:
            lines[:, col] = (255, 190, 60)
    for v in grid.storeys:
        row = int(round((crop.height_m - v) * crop.px_per_m))
        if 0 <= row < lines.shape[0]:
            lines[row, :] = (120, 255, 140)

    from PIL import Image
    panels = [
        label(base, f"macro  ·  {crop.width_m:.0f} x {crop.height_m:.0f} m",
              f"{crop.resolution_px_per_m:.1f} px/m source  ·  "
              f"{100 * crop.covered:.0f}% covered"
              + ("  ·  de-lit" if args.delight else "")),
        label(ramp(np.asarray(crop.depth)), "depth from the mesh",
              "dark red = never seen by the flight"),
        label(overlay, "reveals: measured recess",
              f"{100 * depth_report.get('recessed_fraction', 0):.1f}% of the wall  ·  "
              f"median {depth_report.get('median_recess_m')} m"),
        label(lines, "lattice: derived period",
              f"bay {grid.bay_m:.2f} m ({grid.bay_strength:.2f})  ·  "
              f"storey {grid.storey_m:.2f} m ({grid.storey_strength:.2f})  ·  "
              f"{len(grid.storeys)} rows"),
    ]
    height = min(p.shape[0] for p in panels)
    sheet = np.hstack([p[:height] for p in panels])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet).save(out)
    print(f"\n{out}  ({sheet.shape[1]}x{sheet.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

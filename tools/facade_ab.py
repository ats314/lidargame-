"""Does frequency separation rescue a 77 cm mesh? A/B it before building a shader.

    python tools/facade_ab.py --material plaster --tile-m 0.9

Takes one real Helsinki facade out of the reality mesh and composites it the way
M_MasterFacade specifies -- measured photograph as the low-frequency identity
layer, neutralised procedural detail multiplied on top, micro normal and
roughness lighting the result -- then renders the raw and composited versions at
the magnification a person actually sees from 1 m and from 5 m.

The point is to find out whether the architecture works on this data before
anyone builds it in an engine. The comparison is deliberately unflattering:
magnification is nearest-neighbour, so the macro's real limits stay visible
instead of being blurred into looking better than they are.

What each column is:

    macro only     the photograph, magnified. This is what a raw reality mesh
                   gives at that distance, and what melts.
    + detail       macro x neutralised micro albedo. Colour is untouched by
                   construction -- the detail field averages 1.0 -- so any
                   change is added frequency, not a tint.
    + relief       the same, lit through the micro normal and roughness. This is
                   where the gain should be, because what is missing at 1 m is
                   not colour variation but surface.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidarworld.features import facade as facade_mod       # noqa: E402
from lidarworld.features import frequency                   # noqa: E402
from lidarworld.ingest import objmesh                       # noqa: E402

#: Frame geometry for the "what would a person see" magnification. A 60 degree
#: horizontal field at distance d shows 2*d*tan(30) metres across the frame.
FOV_DEG = 60.0
FRAME_PX = 620


def visible_width_m(distance_m: float) -> float:
    return 2.0 * distance_m * np.tan(np.radians(FOV_DEG) / 2.0)


def best_macro(subtile: Path, *, px_per_m: float, min_rhythm: float) -> facade_mod.Facade:
    """The facade in this subtile with the most real architectural rhythm.

    Largest-by-area is the wrong criterion: the biggest wall in the tile is a
    courtyard flank the flight barely saw, which rectifies into a smooth grey
    sheet. Horizontal repetition is what distinguishes a frontage with windows
    from a blank side, and it is already measured for exactly this purpose.
    """
    mesh, _ = objmesh.drop_webbing(
        objmesh.merge(objmesh.read_directory(subtile)))
    best, best_score = None, -1.0
    for slab in facade_mod.facade_slabs(mesh)[:6]:
        crop = facade_mod.rectify_mesh(mesh, slab, px_per_m=px_per_m)
        if crop is None or crop.covered < 0.45:
            continue
        profile = facade_mod.rhythm_profile(crop.image, crop.px_per_m)
        score = float(np.max(profile)) * crop.covered
        print(f"  candidate {crop.width_m:.0f}x{crop.height_m:.0f} m  "
              f"covered {100*crop.covered:.0f}%  rhythm {np.max(profile):.2f}  "
              f"score {score:.3f}", flush=True)
        if score > best_score:
            best, best_score = crop, score
    if best is None:
        raise SystemExit("no facade in this subtile was usable")
    if best_score < min_rhythm:
        print(f"  warning: best rhythm score {best_score:.3f} is weak; the crop "
              f"may be a blank flank rather than a frontage", flush=True)
    return best


def strongest_window(crop: facade_mod.Facade, width_px: int) -> tuple[int, int]:
    """The column band with the most rhythm, so the A/B looks at real facade.

    A rectified mesh facade is part frontage and part whatever the camera could
    not see. Scoring columns rather than taking the middle keeps the comparison
    on the half that has windows in it.
    """
    grey = crop.image.astype(np.float64).mean(axis=2)
    step = max(16, width_px // 8)
    best, best_score = 0, -1.0
    for x in range(0, max(1, grey.shape[1] - width_px), step):
        band = crop.image[:, x:x + width_px]
        if (band.sum(axis=2) > 0).mean() < 0.85:
            continue
        score = float(np.max(facade_mod.rhythm_profile(band, crop.px_per_m)))
        if score > best_score:
            best_score, best = score, x
    return best, best_score


def high_frequency(image: np.ndarray) -> float:
    """RMS energy above a 4 px low-pass: how much detail is actually present.

    The A/B has to be answerable by a number, not by squinting. The first run of
    this used `plaster` as the micro material and looked identical in all three
    columns -- because that generator contributes 0.003 of high-frequency energy
    against brick's 0.16. Without this measurement that reads as "frequency
    separation does not work" rather than "wrong material".
    """
    grey = np.asarray(image, dtype=np.float64)
    if grey.ndim == 3:
        grey = grey.mean(axis=2)
    if grey.max() > 1.5:
        grey = grey / 255.0
    low = frequency.box_blur(grey[:, :, None], 4)[:, :, 0]
    return float(np.sqrt(np.mean((grey - low) ** 2)))


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
    except TypeError:                       # older Pillow: fixed-size bitmap
        font = small = ImageFont.load_default()
    draw.text((10, 6), text, fill=(232, 234, 236), font=font)
    if sub:
        draw.text((10, 26), sub, fill=(139, 149, 161), font=small)
    return np.asarray(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subtile", default="data/helsinki/mesh/672496/673497c1")
    ap.add_argument("--material", default="stone_block",
                    help="micro material: plaster, brick, stone_block, concrete")
    ap.add_argument("--tile-m", type=float, default=0.45,
                    help="real-world repeat of the micro material, metres")
    ap.add_argument("--detail", type=float, default=1.0)
    ap.add_argument("--ambient", type=float, default=0.55,
                    help="lower lets the micro normal do more work")
    ap.add_argument("--distances", default="1,5")
    ap.add_argument("--px-per-m", type=float, default=48.0)
    ap.add_argument("--min-rhythm", type=float, default=0.15)
    ap.add_argument("-o", "--out", default="build/helsinki/facade_ab.png")
    args = ap.parse_args()

    print(f"reading {args.subtile}", flush=True)
    crop = best_macro(Path(args.subtile), px_per_m=args.px_per_m,
                      min_rhythm=args.min_rhythm)
    print(f"chosen facade {crop.width_m:.1f} x {crop.height_m:.1f} m, "
          f"source {crop.resolution_px_per_m:.1f} px/m", flush=True)

    from PIL import Image
    rows, report = [], {"facade": crop.to_dna(), "columns": [], "rows": []}

    # Pick the piece of wall ONCE, at a span wide enough for rhythm to be
    # measurable, and show that same piece at every distance. Choosing per
    # distance meant the 1 m and 5 m panels looked at different buildings, which
    # is not a comparison. At 1.15 m across there is no room for a 1.5 m bay, so
    # rhythm cannot be scored there at all.
    anchor_m = 6.0
    anchor_px = max(16, int(round(anchor_m * crop.px_per_m)))
    x0, rhythm = strongest_window(crop, anchor_px)
    rows_kept = np.flatnonzero(
        (crop.image[:, x0:x0 + anchor_px].sum(axis=2) > 0).mean(axis=1) > 0.7)
    if not len(rows_kept):
        raise SystemExit("the chosen window has no covered rows")
    y_top, y_bottom = int(rows_kept[0]), int(rows_kept[-1])
    print(f"window at u = {x0 / crop.px_per_m:.1f} m, rhythm {rhythm:.2f}, "
          f"rows {y_top}-{y_bottom}", flush=True)
    report["window"] = {"u_m": round(x0 / crop.px_per_m, 2),
                        "rhythm": round(rhythm, 3),
                        "anchor_m": anchor_m}

    frame_h = int(round(FRAME_PX * 9 / 16))
    for distance in [float(v) for v in args.distances.split(",")]:
        span_m = visible_width_m(distance)
        display_px_per_m = FRAME_PX / span_m
        patch_w = max(8, int(round(span_m * crop.px_per_m)))
        patch_h = max(8, int(round(span_m * (9 / 16) * crop.px_per_m)))
        # Centre inside the chosen window, and inside its covered rows, so a
        # close crop stays on facade rather than sliding onto sky or pavement.
        cx = x0 + anchor_px // 2
        cy = (y_top + y_bottom) // 2
        px0 = int(np.clip(cx - patch_w // 2, 0, max(0, crop.image.shape[1] - patch_w)))
        py0 = int(np.clip(cy - patch_h // 2, y_top,
                          max(y_top, y_bottom - patch_h)))
        band = crop.image[py0:py0 + patch_h, px0:px0 + patch_w]
        if band.shape[0] < 4 or band.shape[1] < 4:
            print(f"  {distance:g} m: patch too small, skipped", flush=True)
            continue
        macro_small = np.asarray(Image.fromarray(band).resize(
            (FRAME_PX, frame_h), Image.NEAREST), dtype=np.float64) / 255.0

        composite = frequency.compose(
            macro_small, material=args.material, px_per_m=display_px_per_m,
            tile_m=args.tile_m, detail=args.detail)

        raw = (np.clip(macro_small, 0, 1) * 255).astype(np.uint8)
        albedo_only = (np.clip(composite.albedo, 0, 1) * 255).astype(np.uint8)
        lit = frequency.shade(composite, ambient=args.ambient)

        magnify = display_px_per_m / max(crop.resolution_px_per_m, 1e-6)
        hf_raw = high_frequency(raw)
        hf_albedo = high_frequency(albedo_only)
        hf_lit = high_frequency(lit)
        panels = [
            label(raw, f"macro only  ·  {distance:g} m",
                  f"{magnify:.0f}x over {crop.resolution_px_per_m:.1f} px/m  ·  "
                  f"detail {hf_raw:.3f}"),
            label(albedo_only, "+ micro albedo",
                  f"{args.material} at {args.tile_m:g} m  ·  detail {hf_albedo:.3f}  "
                  f"({hf_albedo / max(hf_raw, 1e-6):.1f}x)"),
            label(lit, "+ micro relief",
                  f"normal + roughness  ·  detail {hf_lit:.3f}  "
                  f"({hf_lit / max(hf_raw, 1e-6):.1f}x)"),
        ]
        rows.append(np.hstack(panels))
        report["rows"].append({
            "distance_m": distance,
            "visible_width_m": round(span_m, 2),
            "display_px_per_m": round(display_px_per_m, 1),
            "magnification_from_source": round(magnify, 1),
            "high_frequency": {"macro": round(hf_raw, 5),
                               "with_albedo": round(hf_albedo, 5),
                               "with_relief": round(hf_lit, 5),
                               "gain_albedo": round(hf_albedo / max(hf_raw, 1e-6), 2),
                               "gain_relief": round(hf_lit / max(hf_raw, 1e-6), 2)},
            **composite.to_record(),
        })
        print(f"  {distance:g} m: {span_m:.2f} m across, {magnify:.0f}x  |  "
              f"detail {hf_raw:.4f} -> {hf_albedo:.4f} -> {hf_lit:.4f}  "
              f"({hf_lit / max(hf_raw, 1e-6):.1f}x)", flush=True)

    width = min(r.shape[1] for r in rows)
    sheet = np.vstack([r[:, :width] for r in rows])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet).save(out)
    out.with_suffix(".json").write_text(json.dumps(report, indent=1))
    print(f"\n{out}  ({sheet.shape[1]}x{sheet.shape[0]})")
    print(f"{out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

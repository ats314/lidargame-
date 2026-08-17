"""Cut one Hamburg wall out of a tile as a self-contained Unreal prototype.

    python tools/facade_prototype.py data/hamburg/area1/6534/6534.gml \
        --area 565660,5934160,220 -o build/facade

The first prototype is one wall, one macro texture, one brick PBR set and one
Material Instance -- not a system. This produces the half of that which Unreal
cannot: the wall as a *wall*, front on and upright at a known metres-per-pixel,
plus the record that maps a detection in it back to a place in 3D.

Emitted per wall:

    <id>.png            rectified macro crop, ready to be the base colour
    <id>.glb            the wall alone, UV0 = [0,1] over the crop, UV1 = metres
    facade_dna.json     geometry, frame, and the measured source resolution

The glb's UV0 is regenerated here on purpose, and it is the one place that is
allowed. Everywhere else UV0 is Hamburg's atlas mapping and is passed through
untouched; here the crop *is* the texture, so the wall's UV0 is the unit square
by construction. That is what makes the asset self-contained.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidarworld.backends.gltf_textured import Face, export      # noqa: E402
from lidarworld.features import facade as facade_mod            # noqa: E402
from lidarworld.ingest import citygml                           # noqa: E402
from lidarworld.reconstruct.tessellate import close_ring, wall_frame  # noqa: E402


def score(polygon, binding) -> float:
    """How good a first prototype this wall is.

    Big and well-observed, because a 3 m2 wall the camera barely saw proves
    nothing either way about a material system.
    """
    found = binding.get(polygon.gml_id)
    if found is None or not found.uv:
        return -1.0
    return polygon.area


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gml")
    ap.add_argument("--area", help="x,y,size in the file's own CRS")
    ap.add_argument("-o", "--out", default="build/facade")
    ap.add_argument("--count", type=int, default=6, help="how many walls to cut")
    ap.add_argument("--px-per-m", type=float, default=facade_mod.DEFAULT_PX_PER_M)
    args = ap.parse_args()

    gml = Path(args.gml)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    buildings = citygml.read_buildings(gml)
    textures = citygml.read_textures(gml)
    if args.area:
        x, y, size = (float(v) for v in args.area.split(","))
        half = size / 2.0
        buildings = [b for b in buildings
                     if abs(np.vstack([p.exterior for s in b.surfaces
                                       for p in s.polygons])[:, 0].mean() - x) <= half
                     and abs(np.vstack([p.exterior for s in b.surfaces
                                        for p in s.polygons])[:, 1].mean() - y) <= half]
    binding = {t.target: t for t in textures}

    candidates = []
    for building in buildings:
        for surface in building.of("wall"):
            for polygon in surface.polygons:
                value = score(polygon, binding)
                if value > 0:
                    candidates.append((value, polygon, surface, building))
    candidates.sort(key=lambda c: -c[0])
    print(f"{len(candidates)} textured wall polygons; taking the {args.count} largest",
          flush=True)

    atlases: dict[str, np.ndarray] = {}
    records = []
    for area, polygon, surface, building in candidates[:args.count]:
        found = binding[polygon.gml_id]
        image_path = gml.parent / str(found.image).replace("\\", "/")
        if str(image_path) not in atlases:
            atlases[str(image_path)] = facade_mod.load_atlas(image_path)
        atlas = atlases[str(image_path)]

        result = facade_mod.rectify(
            polygon.exterior, found.uv[0], atlas, px_per_m=args.px_per_m,
            surface_id=polygon.gml_id, building_id=building.gml_id)
        if result is None:
            print(f"  [skip] {polygon.gml_id}: degenerate", flush=True)
            continue

        name = (polygon.gml_id or "wall").replace("UUID_", "")[:24]
        facade_mod.save(result, out / f"{name}.png")

        # The completed macro and its confidence travel beside the measured one.
        # The raw crop is kept unchanged for reference and A/B, exactly as the
        # guidance asks: the source stays the truth layer.
        completed, confidence, fill = facade_mod.complete(result)
        from PIL import Image
        Image.fromarray(completed).save(out / f"{name}.macro.png")
        Image.fromarray((confidence * 255).astype(np.uint8)).save(
            out / f"{name}.confidence.png")

        # UV0 over the crop is the unit square: u along the wall, v down the
        # image to match glTF, which `export` flips back.
        ring = close_ring(np.asarray(polygon.exterior, dtype=float))
        u_axis, v_axis, _ = wall_frame(ring)
        su, sv = ring @ u_axis, ring @ v_axis
        # np.ptp as a method was removed in numpy 2; the free function stays.
        unit = np.column_stack([(su - su.min()) / max(np.ptp(su), 1e-9),
                                (sv - sv.min()) / max(np.ptp(sv), 1e-9)])
        export([Face(ring=ring, uv=unit, image=f"{name}.png", kind="wall",
                     surface_id=polygon.gml_id, building_id=building.gml_id)],
               out / f"{name}.glb", image_root=out)

        record = result.to_dna()
        record["source_texture"] = f"{name}.png"
        record["macro_texture"] = f"{name}.macro.png"
        record["confidence_map"] = f"{name}.confidence.png"
        record["completion"] = fill
        record["model"] = f"{name}.glb"
        record["source_atlas"] = image_path.name
        records.append(record)
        print(f"  {name}  {result.width_m:.1f} x {result.height_m:.1f} m  "
              f"crop {result.image.shape[1]}x{result.image.shape[0]}  "
              f"source {result.resolution_px_per_m:.1f} px/m  "
              f"covered {100*result.covered:.0f}%", flush=True)

    (out / "facade_dna.json").write_text(json.dumps({
        "tile": gml.name,
        "crs": "EPSG:25832",
        "uv0": "unit square over the rectified crop (regenerated: the crop is "
               "the texture)",
        "uv1": "wall-local metres; divide by the material repeat in metres",
        "colour_space": {"macro_texture": "sRGB", "masks": "linear/non-sRGB"},
        "facades": records,
    }, indent=1))
    print(f"\n{out/'facade_dna.json'}: {len(records)} facades")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

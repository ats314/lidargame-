"""One wall, built and textured, rendered close enough to judge.

    python tools/textured_wall.py

Every render before this was flat colour: the material system existed, was
measured, and was never applied to the geometry. This does the wiring. It bakes a
seamless tile from the procedural micro material over the building's own de-lit
measured colour, gives every face a UV0 in wall-metres over the tile size, and
points the material at it.

The tile has the micro relief lit into it. That is a stand-in for a normal map,
not a substitute: the software renderer shades flat per face, so without it a
2 mm mortar joint contributes nothing at all. In an engine the normal and
roughness maps would do this and the albedo would stay unlit. Recorded here so
nobody reads the baked shading as measured.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidarworld.backends import gltf_textured                 # noqa: E402
from lidarworld.features import frequency                     # noqa: E402
from lidarworld.reconstruct import elevation                  # noqa: E402


def bake(material: str, tile_m: float, colour, *, px_per_m: float = 420.0,
         seed: int = 3, ambient: float = 0.62, out: Path) -> dict:
    """A seamless tile of `material` in `colour`, with its own relief lit in."""
    size = max(64, int(round(tile_m * px_per_m)))
    macro = np.zeros((size, size, 3))
    macro[:, :] = np.asarray(colour, dtype=float)
    composite = frequency.compose(macro, material=material, px_per_m=px_per_m,
                                  tile_m=tile_m, seed=seed)
    lit = frequency.shade(composite, ambient=ambient)

    from PIL import Image
    Image.fromarray(lit).save(out)

    def hf(image):
        grey = np.asarray(image, dtype=np.float64)
        if grey.ndim == 3:
            grey = grey.mean(axis=2)
        if grey.max() > 1.5:
            grey = grey / 255.0
        low = frequency.box_blur(grey[:, :, None], 4, wrap=False)[:, :, 0]
        return float(np.sqrt(np.mean((grey - low) ** 2)))

    return {"material": material, "tile_m": tile_m, "px": size,
            "px_per_m": px_per_m,
            "colour_in": [round(float(v), 3) for v in colour],
            "colour_out": [round(float(v), 3) for v in
                           (lit.reshape(-1, 3).mean(0) / 255.0)],
            "high_frequency": round(hf(lit), 4),
            "flat_reference": round(hf(macro), 4),
            "note": "micro relief is lit into the tile as a stand-in for a "
                    "normal map; in an engine the albedo stays unlit"}


def uv_for(ring: np.ndarray, tile_m: float) -> np.ndarray:
    """UV0 in wall-local metres over the tile size, so masonry has a real repeat.

    Not normalised by wall size, which is the mistake that makes one brick course
    span a shed and a warehouse identically. `wall_frame` also keeps courses
    running along the wall and stacking toward the sky, so the coursing does not
    rotate at a corner.
    """
    from lidarworld.reconstruct.tessellate import wall_frame
    u_axis, v_axis, _ = wall_frame(ring)
    origin = ring[0]
    local = ring - origin
    return np.column_stack([(local @ u_axis) / tile_m,
                            (local @ v_axis) / tile_m])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--material", default="stone_block")
    ap.add_argument("--tile-m", type=float, default=0.55)
    ap.add_argument("--width-m", type=float, default=26.0)
    ap.add_argument("--from-json", default="build/helsinki/elevation.json",
                    help="take the measured DNA from a previous build")
    ap.add_argument("-o", "--out", default="build/helsinki/wall")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    record = json.loads(Path(args.from_json).read_text())
    measured = record["buildings"][0]["dna"]
    dna = elevation.FacadeDNA(
        bay_m=measured["bay_m"], storey_m=measured["storey_m"],
        storeys=measured["storeys"], window_w_m=measured["window_m"][0],
        window_h_m=measured["window_m"][1], sill_m=measured["sill_m"],
        base_z=0.0, top_z=min(measured["height_m"], 22.0),
        wall_rgb=tuple(measured["wall_rgb"]),
        window_rgb=tuple(measured["window_rgb"]))
    print(f"measured: bay {dna.bay_m:.2f} m, storey {dna.storey_m:.2f} m, "
          f"window {dna.window_w_m:.2f} x {dna.window_h_m:.2f} m, "
          f"wall {[round(v, 2) for v in dna.wall_rgb]}", flush=True)

    tiles = {}
    for kind, material, tile_m, tint in (
            ("wall", args.material, args.tile_m, dna.wall_rgb),
            ("plinth", "stone_block", 0.85,
             tuple(np.asarray(dna.wall_rgb) * 0.72)),
            ("cornice", args.material, args.tile_m,
             tuple(np.clip(np.asarray(dna.wall_rgb) * 1.08, 0, 1))),
            ("string", args.material, args.tile_m,
             tuple(np.clip(np.asarray(dna.wall_rgb) * 1.05, 0, 1))),
            ("sill", "concrete", 0.5,
             tuple(np.clip(np.asarray(dna.wall_rgb) * 1.14, 0, 1))),
            ("reveal", args.material, args.tile_m,
             tuple(np.asarray(dna.wall_rgb) * 0.82)),
            ("glass", "glass", 1.2, tuple(np.asarray(dna.window_rgb) * 1.1)),
            ("frame", "wood_plank", 0.4,
             tuple(np.clip(np.asarray(dna.wall_rgb) * 1.25, 0, 1))),
            ("door", "wood_plank", 0.6,
             tuple(np.asarray(dna.window_rgb) * 0.9)),
            ("roof", "roof_tile", 0.6,
             tuple(np.asarray(dna.wall_rgb) * 0.55))):
        path = out.parent / f"tile_{kind}.png"
        tiles[kind] = (path, tile_m,
                       bake(material, tile_m, tint, out=path))
        info = tiles[kind][2]
        print(f"  {kind:8s} {material:12s} {tile_m:.2f} m  "
              f"detail {info['flat_reference']:.4f} -> {info['high_frequency']:.4f}",
              flush=True)

    a = np.array([0.0, 0.0, 0.0])
    b = np.array([args.width_m, 0.0, 0.0])
    built = elevation.build_wall_detailed(a, b, dna, door=True)
    print(f"wall {args.width_m:.0f} x {dna.height_m:.0f} m: "
          f"{len(built.quads)} quads, {built.report['openings']} openings",
          flush=True)

    faces = []
    for quad, kind in zip(built.quads, built.kinds):
        path, tile_m, _ = tiles.get(kind, tiles["wall"])
        ring = np.asarray(quad, dtype=float)
        faces.append(gltf_textured.Face(
            ring=ring, uv=uv_for(ring, tile_m), image=path.name, kind=kind,
            surface_id=f"wall_{kind}"))
    # A strip of ground so the wall meets something.
    # In FRONT of the wall: the wall runs along +x at y = 0 with its outward
    # normal at -y, so the street is at negative y.
    faces.append(gltf_textured.Face(
        ring=np.array([[-8.0, -30.0, -0.02], [args.width_m + 8, -30.0, -0.02],
                       [args.width_m + 8, 1.0, -0.02], [-8.0, 1.0, -0.02]]),
        kind="ground", surface_id="ground"))
    gltf_textured.FALLBACK["ground"] = (0.29, 0.29, 0.30, 1.0)

    model = out.with_suffix(".gltf")
    gltf_textured.export(faces, model, image_root=out.parent)
    (out.parent / "wall_tiles.json").write_text(json.dumps(
        {k: v[2] for k, v in tiles.items()}, indent=1))
    print(f"\n{model}", flush=True)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from glb_shot import bounds, render
    from PIL import Image
    # The exporter converts Z-up to glTF's Y-up, so in the FILE's frame height is
    # y and the street is +z. Aiming in the source frame put the camera under the
    # pavement looking at the underside of it -- which rendered as a grey wedge
    # and read as "the geometry is broken" rather than "the camera is."
    lo, hi = bounds(model)
    centre = (lo + hi) / 2
    shots = []
    for name, dist, height, target_y in (("street", 19.0, 1.7, 10.0),
                                         ("close", 6.5, 2.2, 5.0)):
        eye = np.array([centre[0] - 2.0, lo[1] + height, hi[2] + dist])
        image = render(model, eye=eye,
                       target=np.array([centre[0], lo[1] + target_y, centre[2]]),
                       width=1180, height=680, sky=(158, 172, 188))
        Image.fromarray(image).save(out.parent / f"wall_{name}.png")
        shots.append(image)
        print(f"  {out.parent / f'wall_{name}.png'}", flush=True)
    Image.fromarray(np.vstack(shots)).save(out.parent / "wall_pair.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

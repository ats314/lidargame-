"""Build a Helsinki demo from the reality mesh, and measure what it is worth.

    python tools/helsinki_demo.py --tile 672496 --size 140 -o build/helsinki

Helsinki's mesh is the opposite trade from Hamburg's LoD3. Hamburg gives flat
polygon walls with a photograph stretched over them; every window is paint, and
below about five metres there is no photograph at all. Helsinki gives real
geometry -- a balcony is a balcony, a window reveal has depth -- but it is
photogrammetry, so it melts at close range instead of going missing.

Neither is better in the abstract. What decides it is where the camera lives, so
this measures the things that differ with distance rather than reporting a
triangle count:

    texels per metre on a facade   the same measure used for Hamburg's atlas,
                                   so the two numbers are comparable
    triangles per square metre     geometric detail actually present
    facade relief RMS              deviation from a fitted plane. A LoD3 box
                                   reads ~0. Anything above a few centimetres is
                                   real modelled depth.
    detail by height band          0-3 m, 3-10 m, 10 m+. An aerial product
                                   degrades toward the ground, and the ground is
                                   where a player stands.
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidarworld.backends.gltf_textured import export_mesh          # noqa: E402
from lidarworld.data import helsinki                                # noqa: E402
from lidarworld.ingest import objmesh                               # noqa: E402


def extract_subtiles(archive: Path, out: Path, limit: int | None = None) -> list[Path]:
    """Unpack the 250 m subtile directories from a 2 km tile archive."""
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        names = handle.namelist()
        dirs: dict[str, list[str]] = {}
        for name in names:
            if name.endswith("/"):
                continue
            parts = name.split("/")
            if len(parts) < 2:
                continue
            dirs.setdefault(parts[-2], []).append(name)
        chosen = sorted(dirs)[:limit] if limit else sorted(dirs)
        written = []
        for key in chosen:
            target = out / key
            if target.exists() and any(target.glob("*.obj")):
                written.append(target)
                continue
            target.mkdir(parents=True, exist_ok=True)
            for name in dirs[key]:
                data = handle.read(name)
                (target / Path(name).name).write_bytes(data)
            written.append(target)
    return written


def facade_metrics(mesh: objmesh.Mesh, *, vertical_cos: float = 0.35) -> dict:
    """Texel density, relief and detail-by-height for the near-vertical surfaces.

    Only near-vertical triangles count as facade: a roof is textured from
    straight above and would flatter every number here.
    """
    positions = mesh.positions
    out: dict = {}
    tex_area_px = 0.0
    metric_area = 0.0
    tri_total = 0
    band_counts = {"0-3m": 0, "3-10m": 0, "10m+": 0}
    band_area = {"0-3m": 0.0, "3-10m": 0.0, "10m+": 0.0}
    ground = float(np.percentile(positions[:, 2], 1)) if len(positions) else 0.0
    normals_sample = []
    plane_points = []

    from PIL import Image
    image_size: dict[str, tuple[int, int]] = {}

    for group in mesh.groups:
        faces = np.asarray(group.faces).reshape(-1, 3)
        if not len(faces):
            continue
        a, b, c = (positions[faces[:, i]] for i in range(3))
        cross = np.cross(b - a, c - a)
        areas = np.linalg.norm(cross, axis=1) / 2.0
        with np.errstate(invalid="ignore", divide="ignore"):
            nz = np.abs(cross[:, 2]) / np.maximum(np.linalg.norm(cross, axis=1), 1e-12)
        vertical = nz < vertical_cos
        if not vertical.any():
            continue
        tri_total += int(vertical.sum())
        metric_area += float(areas[vertical].sum())

        heights = ((a + b + c) / 3.0)[:, 2] - ground
        for label, mask in (("0-3m", heights < 3.0),
                            ("3-10m", (heights >= 3.0) & (heights < 10.0)),
                            ("10m+", heights >= 10.0)):
            sel = vertical & mask
            band_counts[label] += int(sel.sum())
            band_area[label] += float(areas[sel].sum())

        if group.image is not None and len(mesh.uvs) == len(positions):
            key = str(group.image)
            if key not in image_size:
                with Image.open(group.image) as im:
                    image_size[key] = im.size
            width, height = image_size[key]
            ua, ub, uc = (mesh.uvs[faces[:, i]] * np.array([width, height])
                          for i in range(3))
            px = np.abs(np.cross(ub - ua, uc - ua)) / 2.0
            tex_area_px += float(px[vertical].sum())

        # Sample a few facade patches for relief.
        if len(plane_points) < 24 and vertical.any():
            index = np.flatnonzero(vertical)[:400]
            plane_points.append(positions[faces[index]].reshape(-1, 3))
            normals_sample.append(cross[index] / np.maximum(
                np.linalg.norm(cross[index], axis=1, keepdims=True), 1e-12))

    out["facade_triangles"] = tri_total
    out["facade_area_m2"] = round(metric_area, 1)
    out["tris_per_m2"] = round(tri_total / metric_area, 2) if metric_area else 0.0
    out["texels_per_m"] = (round(float(np.sqrt(tex_area_px / metric_area)), 2)
                           if metric_area and tex_area_px else 0.0)
    out["cm_per_texel"] = (round(100.0 / out["texels_per_m"], 2)
                           if out["texels_per_m"] else None)
    out["by_height_band"] = {
        k: {"triangles": band_counts[k], "area_m2": round(band_area[k], 1),
            "tris_per_m2": round(band_counts[k] / band_area[k], 2)
            if band_area[k] else 0.0}
        for k in band_counts}

    # Relief: RMS distance of facade vertices from their own best-fit plane.
    reliefs = []
    for patch in plane_points[:24]:
        if len(patch) < 30:
            continue
        centre = patch.mean(axis=0)
        centred = patch - centre
        # Smallest singular vector is the plane normal.
        _, _, vh = np.linalg.svd(centred, full_matrices=False)
        normal = vh[-1]
        reliefs.append(float(np.sqrt(np.mean((centred @ normal) ** 2))))
    out["facade_relief_rms_m"] = round(float(np.median(reliefs)), 4) if reliefs else None
    out["relief_patches"] = len(reliefs)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tile", default="672496")
    ap.add_argument("--archive")
    ap.add_argument("--size", type=float, default=140.0)
    ap.add_argument("--subtiles", type=int, default=None,
                    help="limit how many 250 m subtiles to unpack")
    ap.add_argument("-o", "--out", default="build/helsinki")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    archive = Path(args.archive) if args.archive else Path(
        f"data/helsinki/Helsinki3D_2017_OBJ_{args.tile}x2.zip")
    if not archive.exists():
        print(f"missing {archive}", file=sys.stderr)
        return 1

    work = Path("data/helsinki/mesh") / args.tile
    dirs = extract_subtiles(archive, work, limit=args.subtiles)
    print(f"{len(dirs)} subtile directories under {work}", flush=True)

    # Pick the subtile with the most near-vertical surface: that is street wall,
    # which is what a pedestrian demo is about. A subtile of park or water has
    # plenty of triangles and no facade.
    best, best_score, best_mesh = None, -1.0, None
    for directory in dirs:
        mesh = objmesh.merge(objmesh.read_directory(directory))
        if not mesh.triangles:
            continue
        metrics = facade_metrics(mesh)
        score = metrics["facade_area_m2"]
        print(f"  {directory.name}: {mesh.triangles:,} tris, "
              f"facade {metrics['facade_area_m2']:.0f} m2", flush=True)
        if score > best_score:
            best, best_score, best_mesh = directory, score, mesh
    if best_mesh is None:
        print("no subtile had any facade", file=sys.stderr)
        return 1
    print(f"\nchosen: {best.name} ({best_score:.0f} m2 of facade)", flush=True)

    lo, hi = best_mesh.bounds
    centre = (lo + hi) / 2.0
    cropped = objmesh.crop(best_mesh, centre[:2] - args.size / 2,
                           centre[:2] + args.size / 2)
    metrics = facade_metrics(cropped)
    result = export_mesh(cropped.positions, cropped.uvs, cropped.groups,
                         out / "helsinki.glb")
    report = {"tile": args.tile, "subtile": best.name,
              "crop_m": args.size, "export": result, "facade": metrics,
              "local_origin_note": "OBJ vertices are local; georeference is the "
                                   "tile origin and is not in the file"}
    (out / "helsinki_demo.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Textured CityGML tile -> .glb, so it can be looked at and put into an engine.

    python tools/citygml_glb.py data/hamburg/area1/6534/6534.gml \
        --area 565660,5934160,220 -o build/hamburg/rathaus.glb

Cropping matters. An inner-city Hamburg tile is 500 buildings, 126,504 polygons
and 65 MB of texture; exported whole it is a ~90 MB file nobody will open twice
while iterating. A block is the unit anyone actually judges a world by.

The crop keeps a building whole if any part of it is inside the box, rather than
clipping geometry, because a half-building reads as a reconstruction failure
when it is only a viewport.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidarworld.backends.gltf_textured import Face, export      # noqa: E402
from lidarworld.ingest import citygml                            # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gml")
    ap.add_argument("-o", "--out", default="build/citygml/tile.glb")
    ap.add_argument("--area", help="x,y,size in the file's own CRS")
    ap.add_argument("--limit", type=int, help="stop after N buildings")
    args = ap.parse_args()

    gml = Path(args.gml)
    print(f"reading {gml} ({gml.stat().st_size/1e6:.0f} MB)", flush=True)
    buildings = citygml.read_buildings(gml, limit=args.limit)
    textures = citygml.read_textures(gml)
    print(json.dumps(citygml.summarise(buildings, textures), indent=1), flush=True)

    if args.area:
        x, y, size = (float(v) for v in args.area.split(","))
        half = size / 2.0
        lo = np.array([x - half, y - half])
        hi = np.array([x + half, y + half])
        kept = []
        for building in buildings:
            points = np.vstack([p.exterior for s in building.surfaces
                                for p in s.polygons] or [np.zeros((1, 3))])
            centre = points[:, :2].mean(axis=0)
            if np.all(centre >= lo) and np.all(centre <= hi):
                kept.append(building)
        print(f"crop {size:g} m at {x:g},{y:g}: {len(kept)}/{len(buildings)} buildings",
              flush=True)
        buildings = kept

    binding = {t.target: t for t in textures}
    faces: list[Face] = []
    for building in buildings:
        for surface in building.surfaces:
            for polygon in surface.polygons:
                found = binding.get(polygon.gml_id)
                faces.append(Face(
                    ring=polygon.exterior,
                    uv=found.uv[0] if found and found.uv else None,
                    image=found.image if found and found.uv else None,
                    kind=surface.kind,
                    # Carried, not dropped: the appearance pipeline has to be
                    # able to name the wall a detected mask belongs to, and
                    # these ids are essentially the ALKIS cadastre's.
                    surface_id=polygon.gml_id or surface.gml_id,
                    building_id=building.gml_id))
    print(f"{len(faces)} faces, {sum(1 for f in faces if f.image)} textured",
          flush=True)

    result = export(faces, args.out, image_root=gml.parent)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""The seed round trip, on geometry nobody can blame the reconstruction for.

    CityGML -> World Seed -> [source unloaded] -> generated world -> render

This is the experiment the project has never been able to run cleanly. When a
Denver block looked like nothing, the cause could have been the reconstruction,
the seed, the generator, the materialisation or the viewer, and no measurement
separated them. Hamburg's published model holds the first of those fixed at
known-good, so whatever comes out the far end is the back half's doing and
nobody else's.

    python tools/citygml_roundtrip.py data/hamburg/area1/6534/6534.gml \
        --area 565660,5934160,220 --theme victorian -o build/hamburg

Three artefacts, in the order they answer questions:

    measured.glb    the source's own textures, mapped directly. The upper
                    baseline: whatever the generator produces is being compared
                    against this, not against memory.
    <name>.seed.json  everything that survives the compression.
    generated/      the world rebuilt from the seed alone. The source is closed
                    before this runs -- not as a gesture, the file handle is
                    gone and the buildings list is dropped.

The comparison that matters is not "are they identical". They cannot be: the
seed throws the measured surface away on purpose. It is whether the generated
block still reads as the same place -- same massing, same street walls, same
skyline.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidarworld.backends import gltf                                  # noqa: E402
from lidarworld.backends.gltf_textured import Face                    # noqa: E402
from lidarworld.backends.gltf_textured import export as export_measured  # noqa: E402
from lidarworld.ingest import citygml                                 # noqa: E402
from lidarworld.ir import seed as seed_module                         # noqa: E402
from lidarworld.reconstruct.tessellate import close_ring, newell      # noqa: E402
from lidarworld.themes.pack import load_pack                          # noqa: E402
from lidarworld.world import generate                                 # noqa: E402


def _slope_and_area(ring: np.ndarray) -> tuple[float, float, list]:
    """Roof plane geometry: how steep, how big, facing where."""
    normal = newell(ring)
    length = float(np.linalg.norm(normal))
    if length < 1e-12:
        return 0.0, 0.0, [0.0, 0.0, 1.0]
    unit = normal / length
    if unit[2] < 0:
        unit = -unit
    slope = float(np.degrees(np.arccos(np.clip(unit[2], -1.0, 1.0))))
    return slope, length / 2.0, [float(v) for v in unit]


#: Below this a "building" is a survey artefact -- a canopy sliver, a light well
#: lid -- and extruding it puts a spike in the block. Measured: the crop's
#: smallest footprint was 1 m2.
MIN_FOOTPRINT_M2 = 10.0


def _area_weighted_z(polygons, fraction: float) -> float:
    """The height below which `fraction` of the roof area sits."""
    z = np.array([p.exterior[:, 2].mean() for p in polygons])
    area = np.array([max(p.area, 1e-9) for p in polygons])
    order = np.argsort(z)
    z, area = z[order], area[order]
    cumulative = np.cumsum(area) / area.sum()
    return float(z[min(np.searchsorted(cumulative, fraction), len(z) - 1)])


def _ring_area(ring: np.ndarray) -> float:
    x, y = ring[:, 0], ring[:, 1]
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) / 2.0


def to_solids(buildings) -> list[seed_module.Solid]:
    """Buildings -> the neutral form the seed extractor takes.

    The footprint is the ground surface where there is one. Where there is not
    -- Hamburg has one building in five hundred without -- the wall bases are
    the fallback, because a building dropped for want of a ground polygon is a
    hole in the block that looks like a reconstruction failure.
    """
    solids: list[seed_module.Solid] = []
    for building in buildings:
        ground = building.of("ground")
        roofs = building.of("roof")
        walls = building.of("wall")

        footprint = None
        if ground:
            biggest = max((p for s in ground for p in s.polygons),
                          key=lambda p: p.area, default=None)
            if biggest is not None:
                footprint = close_ring(biggest.exterior)
        if footprint is None and walls:
            base = np.vstack([p.exterior for s in walls for p in s.polygons])
            floor = base[:, 2].min()
            low = base[np.abs(base[:, 2] - floor) < 0.5]
            if len(low) >= 3:
                # Convex hull of the wall feet. Coarser than a ground polygon,
                # and it keeps the building in the block.
                centre = low[:, :2].mean(axis=0)
                order = np.argsort(np.arctan2(low[:, 1] - centre[1],
                                              low[:, 0] - centre[0]))
                footprint = low[order]
        if footprint is None or len(footprint) < 3:
            continue
        if _ring_area(footprint[:, :2]) < MIN_FOOTPRINT_M2:
            continue

        ground_z = float(footprint[:, 2].mean()) if footprint.shape[1] > 2 else 0.0
        roof_polygons = [p for s in roofs for p in s.polygons]
        if roof_polygons:
            # The eave is where most of the roof *area* is, not the lowest roof
            # surface anywhere on the building.
            #
            # Measured on this block: taking the minimum gives a median height
            # of 10.3 m, and the area-weighted median gives 25.8 m. The real
            # buildings are 25-30 m. The minimum is catching courtyard wings,
            # rear extensions and the low sides of dormers -- every one a real
            # roof, none of them the roofline -- so it halved the whole block
            # and turned an inner-city street wall into bungalows.
            eave_z = _area_weighted_z(roof_polygons, 0.5)
            if eave_z - ground_z < 2.0:
                eave_z = float(np.vstack([p.exterior for p in roof_polygons])[:, 2].max())
        elif walls:
            eave_z = float(np.vstack([p.exterior for s in walls
                                      for p in s.polygons])[:, 2].max())
        else:
            continue

        planes = []
        for surface in roofs:
            for polygon in surface.polygons:
                slope, area, normal = _slope_and_area(close_ring(polygon.exterior))
                if area > 1.0:
                    planes.append((slope, area, normal))

        solids.append(seed_module.Solid(
            footprint=footprint[:, :2], ground_z=ground_z, eave_z=eave_z,
            roof_planes=planes))
    return solids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gml")
    ap.add_argument("--area", help="x,y,size in the file's own CRS")
    ap.add_argument("--theme", default="victorian")
    ap.add_argument("-o", "--out", default="build/roundtrip")
    ap.add_argument("--name", default="hamburg")
    args = ap.parse_args()

    gml = Path(args.gml)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"source": str(gml), "source_bytes": gml.stat().st_size}

    buildings = citygml.read_buildings(gml)
    textures = citygml.read_textures(gml)
    report["tile"] = citygml.summarise(buildings, textures)

    if args.area:
        x, y, size = (float(v) for v in args.area.split(","))
        half = size / 2.0
        keep = []
        for building in buildings:
            points = np.vstack([p.exterior for s in building.surfaces
                                for p in s.polygons] or [np.zeros((1, 3))])
            centre = points[:, :2].mean(axis=0)
            if abs(centre[0] - x) <= half and abs(centre[1] - y) <= half:
                keep.append(building)
        buildings = keep
    report["buildings_in_crop"] = len(buildings)

    # ---- 1. the upper baseline: the source's own photographs -----------------
    binding = {t.target: t for t in textures}
    faces = [Face(ring=p.exterior,
                  uv=binding[p.gml_id].uv[0]
                  if p.gml_id in binding and binding[p.gml_id].uv else None,
                  image=binding[p.gml_id].image
                  if p.gml_id in binding and binding[p.gml_id].uv else None,
                  kind=s.kind)
             for b in buildings for s in b.surfaces for p in s.polygons]
    measured = export_measured(faces, out / "measured.glb", image_root=gml.parent)
    report["measured"] = measured
    print(f"measured: {measured['triangles']:,} triangles, "
          f"{measured['textures']} textures, {measured['bytes']/1e6:.1f} MB",
          flush=True)

    # ---- 2. compress to a seed ----------------------------------------------
    solids = to_solids(buildings)
    tile_bytes = sum(f.stat().st_size for f in gml.parent.rglob("*") if f.is_file())
    world_seed = seed_module.from_solids(
        solids, name=args.name, crs="EPSG:25832", source_bytes=tile_bytes,
        note="Compressed from a published CityGML model, not from returns. "
             "Facade openings were never in the source either -- Hamburg's "
             "LoD3 is roof detail, and its windows live only in the texture, "
             "which this does not carry.")
    seed_path = out / f"{args.name}.seed.json"
    written = seed_module.write(world_seed, seed_path)
    report["seed"] = written
    print(f"seed: {written['bytes']:,} B for {written['buildings']} buildings"
          + (f", {written['ratio']}x vs {written['compressed_from_bytes']/1e6:.0f} MB"
             if "ratio" in written else ""), flush=True)

    # ---- 3. drop the source, then rebuild ------------------------------------
    # Not a gesture. If anything downstream still reaches for a polygon or a
    # texture, it fails here rather than quietly making the result look better
    # than the seed can justify.
    del buildings, textures, binding, faces, solids, world_seed
    gc.collect()

    reloaded = json.loads(seed_path.read_text())
    world = generate.expand(reloaded)
    pack = load_pack(args.theme)
    result = gltf.export(world, pack, out / "generated", name=args.name)
    report["generated"] = {k: v for k, v in result.items()
                           if isinstance(v, (int, float, str))}
    report["generated"]["note"] = world.notes["generated_from"]["note"]
    print(f"generated: {json.dumps(report['generated'])}", flush=True)

    (out / "roundtrip.json").write_text(json.dumps(report, indent=1))
    print(f"\n{out/'roundtrip.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

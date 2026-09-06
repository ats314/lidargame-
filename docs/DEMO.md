# The demo: one measured block, two renderers, one file between them

A 400 m block of the Amsterdam canal belt, compiled once and walked in an ASCII
megacity that has never heard of this compiler. It exists to test the claim in
`docs/MASTER.md` — that the World Seed is a real contract and not a diagram —
and the claim did not survive first contact intact. What it found is the more
useful half of this document.

## Reproduce it

```bash
lidarworld fetch amsterdam_grachtengordel -o data/real
lidarworld compile data/real/25GN1_02.LAZ -o build/ams -n amsterdam_grachtengordel \
    --bbox 121300,486500,121700,486900 \
    --footprints amsterdam --streets amsterdam --water amsterdam --seed --tile 0.3
lidarworld noctis build/ams/amsterdam_grachtengordel.seed.json \
    -o build/ams/city.png --meta build/ams/city.json --grid 256 --extent 400
python tools/plan_view.py build/ams/city.png          # look at it from above
```

Then paste `city.png` into `CityBuilder`'s `LIDAR_CITIES` as a data URI with the
fields from `city.json`, and walk it.

## The data

**AHN5 via GeoTiles (TU Delft), CC0 1.0.** The best source in the catalogue:
23 pts/m² against 3DEP's 4, a public-domain dedication rather than an
attribution licence, with surveyed footprints, street centrelines and canal
outlines available as side inputs. The block is the Rembrandtplein/Amstel bend,
which is the case footprint-driven grouping should be best at — canal frontage
on three sides.

No synthetic data appears anywhere in this demo.

## What it costs

| stage | size | what it holds |
|---|---|---|
| source tile | 240 MB LAZ | 29.2 M points, 1000 × 1250 m |
| the 400 m block | 3.66 M points | what the crop keeps |
| Spatial IR | 67.6 MB `.lwir` | 10,270 nodes, 26,362 edges |
| web bundle | 128 MB | 2.36 M verts, 1.18 M tris |
| **World Seed** | **285 KB** | 349 buildings, 98 roads, 20 water bodies, 1,240 trees |
| NOCTIS-7 texture | 39 KB | 256 × 256 cells at 1.5625 m |

**The seed is 740× smaller than the bundle it came from**, and the ASCII city is
39 KB — four bytes per cell, no triangles, no materials, no scene graph.

## What compiled

```
349 footprint polygons          2,772 walls extruded from them
3,919 planar patches            merged from 4,000
4,795,869 tiles                 1,336 openings
558 structures                  349 footprints matched 5,004/6,555 patches
1,240 trees, 67 vehicles, 513 poles
water surface filled 24,257 void cells at -0.71 m, 0.9 m below a measured
  bank at 0.19 m (inferred: nothing came back off the water)
```

In the bake: 27,347 building cells, 12,507 road, 12,012 water, 4,897 canopy.

## The bad numbers

**37 of 349 buildings are cropped.** The tile is 400 m and the seed's bounds are
400 m, but footprints straddle the crop edge, so 312 are placed and the rest
fall outside. `buildingsCropped` is in the meta rather than discovered later by
wondering where the edge of the city went.

**The cell is 1.5625 m, not the 4 m of the other measured cities.** The
renderer's world cell is a fixed unit, so this block reads about 2.5× larger
in-game than Manhattan does. The alternative was a 400 m block sitting as a
small island in a kilometre of empty plaza. The finer cell is the better demo
and the worse comparison, and the number is in the entry rather than hidden.

**1,240 trees is still 6.8× more than BGT surveys.** Known weakness #6 is
unchanged by this work; the trees are now in the right *place*, which is a
different problem from there being too many of them.

## What the demo found

Four defects, all silent, none caught by the 417 tests that existed. Three are
fixed here; the fourth is recorded.

**1. Vegetation was written in the wrong coordinate frame.** The pipeline shifts
the cloud to a local origin and adds that origin back when it writes instance
nodes, so a tree node's position is absolute CRS metres while `bounds`,
footprints, road centrelines and canal rings are local. `extract()` passed
vegetation straight through, which put this block's trees 121 km away. The
symptom was a city with **0 canopy cells from a seed carrying 1,240 trees**.
Nothing raised: a seed did not declare a frame per section, so no consumer could
tell which one it held. Fixed, and `WorldSeed` now states the rule — one frame
per file, metres local to `origin`.

**2. The expander read keys the seed does not write.** `generate.expand()` looked
for `position` / `crown_radius`; the seed writes `xy` / `crown_r`. Every tree
fell back to its default, so all 1,240 stacked on `[0, 0, 0]` at 2.5 m. `.get`
with a default cannot distinguish a missing key from an absent tree. Fixed, with
the old names kept as fallbacks.

**3. A source's terms were dropped at three boundaries.** `fetch` called
`describe()` to check commercial use and threw the answer away; a LAS header
records no publisher, so ingest reported "unknown — check the tile's provider";
the seed carried bare internal ids; and the renderer put `src0` on screen where
the attribution belonged. Fixed end to end: `fetch` writes `<tile>.source.json`,
ingest reads it, the seed carries licence and attribution per source, and the
bake shows

> AHN / Rijkswaterstaat; tiling by GeoTiles, TU Delft · CC0 1.0 (public domain dedication)

CC0 is the forgiving case. The same path would have silently stripped a CC BY
notice.

**4. Programs are never written to `.lwir`.** Not fixed. `extract()` reads
`getattr(world, "programs", [])` and the reader restores none, so extracting a
seed from a re-read archive silently yields **zero buildings**. Nothing is
blocked, because the seed is written during the compile — but a seed cannot be
regenerated from an archive, and every correction above cost a six-minute
recompile instead of a three-second re-extract.

## Why the plan view exists

`tools/plan_view.py` draws the baked texture from above. The failure modes of a
bake are geometric and invisible to every metric: a transposed terrain plane, a
mirrored grid, an axis swapped between seed and tile all report perfectly good
counts. The plan is what says these numbers describe *Amsterdam* rather than
something Amsterdam-shaped — the Amstel bends through it, the canals branch, the
blocks have canal frontage on three sides.

That is the same rule as known weakness #9, applied one layer further out.

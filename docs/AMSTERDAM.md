# Amsterdam

The block: 400 m at Rembrandtplein and the Amstel bend, RD 121300,486500 to
121700,486900, out of AHN5 tile `25GN1_02`.

```bash
lidarworld fetch amsterdam_grachtengordel -o data/real
lidarworld compile data/real --bbox 121300,486500,121700,486900 -o build/ams \
    --footprints amsterdam --streets amsterdam --seed --sir --gltf
python tools/amsterdam_check.py build/ams/real.seed.json
python tools/shoot.py --out build/shots
```

## Why here

Denver is the worst city this compiler has, and it was the first one. Amsterdam
is better on the three axes that were actually blocking:

| | Denver (3DEP) | Amsterdam (AHN5) |
|---|---|---|
| density | ~4 pts/m2 | **23 pts/m2** (29,219,688 returns in one 1000 x 1250 m tile) |
| terms | liability disclaimer, no grant | **CC0** |
| independent height | aerial stereo (level 2) | 3D BAG (level 2) |
| roof form | published nowhere, all 257 seeds said `flat` | `b3_dak_type` states flat or slanted |
| tree ground truth | none | **BGT surveys every public tree as a point** |

What Amsterdam does not fix: airborne LiDAR still never sees a facade. A canal
frontage at 23 pts/m2 is sparse rather than absent, which is better than Denver's
nothing and still not a measured elevation -- see `GENERATED_FACADES.md`. Vienna
remains the city with oblique imagery.

## What compiled

```
3,659,659 points in the block          3919 planar patches (merged from 4000)
349 footprints                         2772 walls extruded, 1663 openings
1450 pitched roof surfaces             1043 flat
98 road segments                       1243 "trees", 67 vehicles, 513 poles
seed 270 KB against a 154 MB bundle    1002x
```

Pitched against flat is worth pausing on. Every Denver building came back flat,
because roof form was measured and then discarded and a fitted slope produced
plates larger than the building. Here the majority of roof surface is pitched and
it survives into the seed, which is most of what makes a Dutch street look Dutch.

## The two scores

`tools/amsterdam_check.py` compares the seed against layers the build never saw.

### Heights, against 3D BAG

```
compared              349 buildings, joined on the BAG id the seed now carries
median abs error      0.64 m          (Denver against aerial stereo: 1.18 m)
median bias          +0.47 m          LiDAR reads taller
within 2 m            86.2%           (Denver: 72%)
within 5 m            97.7%           (Denver: 90%)
```

The bias has the same sign and roughly the same size as Denver's +0.40 m, and
the same explanation: the returns land on the parapet, and a modelled height
digitises past it.

**This is a weaker check than Denver's.** 3D BAG derives its heights from the
same AHN returns -- a different team and a different pipeline, but not a
different sensor -- so a shared acquisition artefact would not show up as
disagreement. It is recorded at independence 2 in `data/amsterdam.py` and should
not be quoted as if it were an independent survey.

### Trees, against the BGT -- and this one is bad

```
surveyed (BGT)        183
reconstructed        1243              6.8x
within 5 m of a surveyed tree  257
beyond 15 m of any surveyed tree  790
median distance to the nearest surveyed tree  24.35 m
```

Known weakness 6 says the tree count "is still probably high; there is no
airborne ground truth in the repo to check it against". There is now, and it is
6.8x high.

The honest caveat, stated up front so it is not mistaken for a defence: BGT
surveys the public realm. A tree in a courtyard or behind a canal house is
canopy in the returns and absent from the layer, and the canal belt is full of
them. So 6.8x is an upper bound on the error rather than the error. It is not
explained by private gardens alone -- 790 of 1243 are more than 15 m from any
surveyed tree, which is further than a garden is deep, and the median
reconstructed "tree" is 24 m from the nearest real one.

The layer is held as `hidden_truth` in the manifest. Feeding it in would turn
the tree count into a copy of the answer, and the point of having it is that it
is a score.

## The tile grid

AHN's own PDOK service publishes 0.5 m rasters, not the point cloud. GeoTiles
(TU Delft) publishes the LAZ, cut on the national 1:25,000 sheet grid: a sheet
is 5000 x 6250 m, cut 5 x 5, so a subtile is 1000 x 1250 m and 300-550 MB.
Subtiles are numbered in reading order from the north-west and written
zero-padded:

```
 1  2  3  4  5     north
 6  7  8  9 10
11 12 13 14 15
16 17 18 19 20
21 22 23 24 25     south
```

None of that is documented. It was derived by range-reading 400 bytes of the LAS
header off ten probe tiles across four sheets and reading their true bounds back
out; `ahn.extent_of` does the same thing for any tile, which is how to check one
before committing to a download. Sheet origins in `data/ahn.py` cover the
Amsterdam region only -- a sheet nobody has probed is not in the table, because a
lookup that decides what to download should not contain a guess.

## Three bugs this found

- `published_height` multiplied by `FOOT` unconditionally, which is correct for
  Denver and wrong for everyone else. A 3D BAG metre read as a foot made a 16 m
  canal house 5 m tall. Units now normalise in `data.gis.attributes`.
- A WFS bbox in URN-form `EPSG:4326` is read latitude-first, so `west, south,
  east, north` asked for a square in the Indian Ocean. The server answered 200
  with an empty collection and the build printed `0 footprint polygons`. Extents
  are now requested in the layer's own projected frame.
- `TileIndex` globbed `*.laz` case-sensitively; GeoTiles serves `.LAZ`. A
  directory with a 240 MB tile in it indexed as empty.

## Not wired

- **The canals.** `BGT waterdeel` has them as polygons. LiDAR over water is a
  hole -- the pulse leaves and does not come back -- so the returns describe a
  city with gaps and the layer says the gaps are canals. The World Seed has no
  water field, which is the actual blocker, and in Amsterdam that omission is
  not a detail.
- **BGT wegdeel**, the carriageway as a surveyed polygon with its surface
  material. Strictly better evidence than a centreline and an assumed width; the
  street stage rasterises lines, so swapping it is its own change.
- **Amsterdam's own tree register**, which carries species and a height class
  per tree. That would score tree *heights*, not just counts.
- **AHN4 against AHN5.** Same city, two flights three years apart. The tiles are
  addressable today (`--version ahn4`); nothing consumes the pair yet.

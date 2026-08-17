# Uniting the Hamburg layers into one world

Six datasets, six publishers' conventions, one world. This is how they join, and
— more usefully — where they *don't*, with the disagreement measured rather than
assumed.

Nothing here is a file-format exercise. Format conversion is the easy part and
takes an afternoon. The work is reconciliation: two authorities describing the
same square metre of Hamburg and disagreeing about it.

## 1. What each layer is authoritative for

The rule is that exactly one layer owns each question. Where two could answer,
the more specific and more recently measured one wins, and the loser is kept for
cross-checking rather than discarded.

| question | authority | epoch | why it wins |
|---|---|---|---|
| where the ground is | DGM 1 | 2016 | 1 m grid, the only bare-earth surface here |
| what the ground *is* | ALKIS land use | 2018 | carriageway vs pavement vs courtyard vs water |
| where streets run | HH-SIB network | 2014 | a routable graph with class and width |
| what the ground looks like | DOP20 | current | 20 cm, same resolution as the facades |
| building massing and roofs | LoD3.0-HH | 2023–24 | photogrammetric, detailed roof landscape |
| what a facade looks like | LoD3 textures | 2020 flight | the only facade evidence that exists |

Note the epochs. They span a decade. That is not a defect to hide — it is a
temporal conflation problem, and it is why the road network cannot be trusted to
agree with the 2023 buildings about a street that has been rebuilt since.

## 2. One spatial frame

Everything is **EPSG:25832** (ETRS89 / UTM zone 32N). Verified, not assumed:

- LoD3 tiles declare `srsName="epsg:25832"` in `gml:boundedBy`.
- DGM 1 posts read `549991.50 5934267.50 3.23` — eastings in the 5.5×10⁵ range,
  northings 5.93×10⁶. Same frame, 1 m grid on half-metre cell centres.

That is the easy half. The hard half is that **projected coordinates do not fit
in float32**, which every engine and every vertex buffer uses. At a Hamburg
northing, float32 resolves 0.5 m; at the easting, 0.0625 m. Exporting raw snaps
every vertex to that grid, anisotropically, and sawtooths any wall not aligned
to the axes.

So the frame has a second component: **a georeference origin per delivered
tile**. Vertices go out local to it, the origin travels as node translation and
metadata, and float32 gets a range it can actually hold. This was a live bug
until this week — distinct northing values in an exported block went from 512 to
6,731 once it was fixed.

## 3. One tile grid

The tile names decode, which makes the join a lookup rather than a spatial query:

```
buildings   6534                      E 565–566 km, N 5934–5935 km   1 km
terrain     DGM1_32564_5934_2_FHH     E 564–566 km, N 5934–5936 km   2 km
```

One DGM tile covers exactly four building tiles. Roads, cadastre and imagery are
citywide and get indexed into the same grid on ingest.

An area of interest therefore resolves to a deterministic file set: one building
tile, one terrain tile, and a clipped extract from each citywide layer. No
layer is ever loaded whole to build a block.

## 4. Joins are by identity where identity exists

Hamburg's building ids are *essentially the ALKIS cadastre's own*. That is a
real join key, not a spatial guess, and it is the thing Denver never had — two
Denver publishers shared 461 of 503 ids by luck of lineage rather than design.

```
LoD3 building  --id-->  ALKIS parcel  --id-->  address
     |
     +-- surface id --> texture image + UV0     (already carried)
     +-- surface id --> wall frame + metric extent  (surfaces.json)
```

Only where no identity exists does the join fall back to geometry: terrain
sampled under a footprint, land use polygons intersected with the ground plane,
road centrelines buffered to their width.

## 5. The seams, measured

### Buildings sit ~0.55 m below the terrain surface

The most important number here, and it is bad news:

| sampling | median (building ground − DGM 1) | sd | within 0.5 m | within 1 m |
|---|---|---|---|---|
| footprint centroid | −0.62 m | 0.98 | 44% | 66% |
| 2 m outside footprint | −0.54 m | 1.01 | 45% | 63% |

Measured over 499 buildings in the Rathaus/Binnenalster tile.

The two sampling methods agree, which rules out the obvious explanation. A
bare-earth model has no measurement *under* a building, so a centroid sample
reads the interpolator rather than the ground; if that were the cause, sampling
outside the footprint would have moved the number. It did not.

So it is a genuine systematic offset, and the likely cause is stated in the
publisher's own description: the LoD3 buildings were placed on **DGM 5H**, a 5 m
model with break lines, and this is **DGM 1** from a different epoch. Two
terrain models, both official, disagreeing by half a metre.

**Consequence if ignored:** drop the buildings straight onto DGM 1 and the median
building floats 0.55 m above the pavement, with a 1 m spread — some sunk, some
hovering. At street level that is the single most visible possible defect.

**How it gets handled:** the building's own ground intersection wins locally, and
the terrain is warped to meet it. The LoD3 model carries
`lod3TerrainIntersection` per building — the surveyed line where that building
meets grade — so the DGM becomes a *far-field* surface, conformed to the
buildings near them. The residual after conforming is the number to report, and
it replaces this one as the scoreboard.

### Other seams, not yet measured

- **Road width vs ALKIS carriageway polygons.** HH-SIB is centrelines with a
  class; ALKIS has the actual surface. Where they disagree the polygon should
  win, but the 2014-vs-2018 gap means some streets have been rebuilt between.
- **Orthophoto registration.** Denver's imagery registration was never resolved
  and the same check is owed here before ground texture is trusted.
- **Facade texture vs geometry.** Already visible: in narrow streets some walls
  carry pavement imagery, because the oblique camera never saw into the alley.

## 6. What actually reaches Unreal

Per delivered tile, one self-contained `.glb` plus a sidecar:

```
UV0   source appearance atlas, passed through untouched   the building's identity
UV1   wall-local metres                                   micro material scale
```

UV1 is **metric, not normalised**. Normalising by wall width and height was the
obvious choice and it is wrong: it makes one brick course span a garden shed and
a warehouse identically. The shader divides by a real material repeat instead.

The frame is *oriented*, not projected: v is world up, u runs horizontally along
the face. Masonry is directional, and a triplanar projection rotates brick
courses at a corner, which reads as fake immediately.

The sidecar `.surfaces.json` carries, per face: surface id, building id, wall
frame axes, metric extent, source image. That is what makes a mask detected in
UV0 mappable back onto a named wall in 3D — none of it survives inside a glTF,
and dropping the ids at the conversion boundary makes the appearance work
impossible afterwards.

## 7. Order of assembly

1. **Terrain first.** Nothing else can be placed until there is a ground.
2. **Conform terrain to buildings** using `lod3TerrainIntersection`; report the
   residual.
3. **Classify the ground** from ALKIS, with HH-SIB as the fallback where the
   cadastre is silent.
4. **Place buildings** on the conformed surface.
5. **Texture the ground** from DOP20 for the AOI.
6. **Emit per-tile glb + surface index.**

Ground before buildings, because the first Hamburg render put a correct
inner-city block on a green field and that was not a rendering fault — buildings
were the only thing that had been acquired.

## 8. What is on disk

| layer | size | status |
|---|---|---|
| LoD3 textured Area 1 (Innenstadt) | 1.52 GB | ✅ |
| LoD3 textured Area 2 (Bergedorf) | 1.56 GB | released |
| LoD3 textured Area 3 (Harburg) | 1.87 GB | released |
| LoD3 textured Area 5 (Wandsbek) | 3.48 GB | released |
| LoD3 textured Area 4 (Altona/west) | 4.50 GB | ✅ |
| DGM 1 terrain | 2.95 GB | ✅ |
| ALKIS cadastre | 0.54 GB | ✅ |
| HH-SIB roads | 0.01 GB | ✅ |
| DOP20 orthophoto | — | per-AOI via cached WMS |

**Acquired complete at 15.4 GB, then trimmed to 9.5 GB.** Area 4 was verified
after download — 190 tiles, 166,443 textures.

Areas 2, 3 and 5 were released once the working area narrowed to a street
stretch in Area 1. That is a disk decision, not a data decision: the archive
server measured 89 MB/s, so the whole 12.9 GB building set came down in minutes
and restoring any area is `python tools/fetch_hamburg.py --only lod3_area3`.
Holding 6.9 GB of outer districts to avoid a few minutes of re-download is the
wrong trade on a fixed allowance.

Room was made by deleting Denver's orthophoto imagery (3.8 GB, 10 GeoTIFF tiles
plus world files), on instruction. It is re-downloadable from the county, and no
point cloud was touched: Denver's LAZ tiles remain in `data/real`.

5.3 GB free. Extracting all five areas at once would not fit — an area expands
to roughly twice its packed size — so extraction stays per-tile and on demand,
which `hamburg.extract_tile` already does. That is the reason the remote-zip
path exists rather than a convenience.

## 9. Gotchas worth not rediscovering

- **The ALKIS `.GML` is a ZIP.** It starts with `PK`, contains 258
  `ALKIS_HH_*.xml` members, and the extension lies.
- **Hamburg's archives are Deflate64.** Python's `zipfile` and Info-ZIP `unzip`
  both refuse it, and the stdlib fails only when a member is *read*, so the index
  parses fine and the failure looks like a corrupt download.
- **daten-hamburg.de 301s to www.daten-hamburg.de.** Range requests need the
  resolved host or every slice renegotiates.
- **"LoD3.0-HH" does not mean facade openings.** There is no `bldg:opening`,
  `Window` or `Door` anywhere in the model. It means a detailed roof landscape —
  7,403 building installations in one tile. The windows exist only in the
  texture.

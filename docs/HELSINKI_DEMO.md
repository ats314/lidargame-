# Helsinki: the demo, and why it wins

Measured on the historic core — tile `672496`, subtile `673497c1`, a 140 m crop
of the block between Senate Square and Kamppi. Every number below comes from the
data, not from a dataset page.

## The result

| | Hamburg LoD3 | Helsinki reality mesh |
|---|---|---|
| facade texture | 4.1 px/m → **24 cm per texel** | 10.8 texels/m → **9.3 cm per texel** |
| facade geometry | flat polygons, relief **0 by construction** | relief RMS **0.128 m**, 4,598 patches |
| windows as geometry | none | none, but *recessed* in the mesh |
| usable wall height | median **17%**, top only | **full height** |
| detail at 0–3 m | no photograph at all | **2.10 tris/m² — its densest band** |
| triangles | 51,140 over a 220 m block | **346,971** over a 140 m crop |
| demo size | 10.5 MB | 18.5 MB |

**Helsinki is better on every axis that was measured.** Not marginally: 2.6× the
texture resolution, real geometric depth where Hamburg has none, and — the line
that decides it — *more* detail at pedestrian height than above it.

## The line that decides it

```
height band     triangles per m² of facade
0–3 m               2.10      <- densest
3–10 m              1.51
10 m+               1.82
```

Hamburg's aerial oblique sees the top of a wall and not the bottom: measured
over its own best block, the photograph covered a median 17% of wall height, all
of it at the top, and six of twenty-four walls had nothing trustworthy at all.
The macro layer was strongest exactly where a player never is.

Helsinki's mesh does not do that. The 0–3 m band is its *densest*, because the
mesh is built from imagery flown to see the street, and the ground plane is
where the most views overlap. That inverts the whole problem: the lower storeys
stop being something to invent.

## What it still cannot do

**Median facade triangle is 0.296 m² — a 77 cm edge.** At 20 m and beyond the
block is convincing. At 1 m it melts: the classic photogrammetric candle-wax,
where a window reveal becomes a smooth dimple and a downpipe becomes a ripple in
the wall. Real relief, wrong resolution.

So the frequency-separation architecture is still the answer, and now it has a
better base layer to work from: the macro channel is 9.3 cm rather than 24 cm,
it exists all the way to the pavement, and it sits on geometry that already has
depth. The procedural micro layer supplies what is missing below ~70 cm rather
than inventing whole storeys.

**No open city model anywhere gives facade openings as geometry.** Checked
across three now:

| model | windows | doors | openings |
|---|---|---|---|
| Hamburg LoD3.0-HH | 0 | 0 | 0 |
| Helsinki CityGML (LoD2, Kalasatama) | 0 | 0 | 0 |
| Denver | n/a — nothing was ever modelled | | |

Helsinki's semantic model carries 2,980 buildings, 34,260 wall surfaces and
47,730 texture bindings, and not one `bldg:Window`. Windows are always paint or
absent. The reality mesh is the only source of facade depth in open data, which
is the strongest single argument for Helsinki.

## Street width is a real constraint

The demo cameras stand in open ground and look along the street. In the historic
core the **furthest any pedestrian position gets from a building is 15 m** — the
streets are 15 m wide. The 20 m and 50 m standoffs in the distance series both
clamp to it.

That is not a defect in the data, it is Helsinki. It also means the canyon view
already contains the whole distance series: looking down the street puts facades
at 1 m, 5 m, 20 m and 50 m in one frame, which is a fairer test of a material
system than four separate shots.

## Webbing, reported rather than hidden

A photogrammetric mesh bridges gaps it should leave open — long near-vertical
triangles thrown across a street or between buildings. On this crop:

- 5,557 triangles
- carrying **56.9%** of the raw near-vertical area
- largest single triangle **889 m²**

They are excluded from every density figure above and counted separately. The
first version of this measurement did not exclude them and reported 245,852 m²
of facade in a 140 m crop, which is impossible and looked merely large.

## What is on disk

Nine tiles, 6 × 6 km of central Helsinki, **13.3 GB**, every archive
CRC-verified:

| tile | area | size |
|---|---|---|
| 672496 | **historic core — Senate Square, Esplanadi, Kamppi** | 1.94 GB |
| 674496 | Kruununhaka, Hakaniemi | 2.17 GB |
| 674494 | Töölö | 2.33 GB |
| 672494 | west core | 1.53 GB |
| 676498 | north | 1.54 GB |
| 674498 | Kalasatama | 1.31 GB |
| 670496 | south, Kaivopuisto | 1.09 GB |
| 670494 | south-west | 0.73 GB |
| 672498 | east core | 0.67 GB |

Plus the semantic CityGML for Kalasatama (23 MB, LoD2, textured) and the
published 360 photos (47 MB).

The full reality mesh is **190 GB over 122 tiles**, which is why the manifest
takes nine and says so. Tile names decode — three digits of northing km, three
of easting km, with the `25` zone prefix dropped — so an area of interest
resolves to a file rather than a search. Senate Square, Esplanadi and Kamppi all
land in `672496`, which is the check that the decode is right rather than merely
plausible.

## Coordinates

- Mesh: **EPSG:3879** (ETRS-GK25), heights N2000, but OBJ vertices are **local**
  and the file carries no georeference. This is recorded rather than guessed —
  a mesh on the wrong offset is a building in the wrong street and nothing about
  it raises.
- Semantic CityGML: same CRS, **absolute** (`2.5497485E7`). So the semantic model
  can georeference the mesh, which is the obvious next step.

## Gotchas

- **Vertices must be compacted on crop.** A cropped mesh arrives with the whole
  tile's vertex array; writing it out makes the file lie about its own extent,
  because POSITION min/max then describe the uncropped tile. Pedestrian cameras
  framed from those bounds landed in the empty outer ring and rendered 0%
  coverage.
- **A mesh's minimum height is not the pavement.** It is a basement or a void
  triangle, and its centroid is masonry — so eye-height cameras have to be
  placed by finding open ground and a wall to face, not by offsetting from the
  bounding box.
- **OBJ indexes position and texture coordinate separately.** A corner shared by
  two triangles with different UVs is one `v` and two `vt`; welding on position
  alone smears the texture and raises nothing.

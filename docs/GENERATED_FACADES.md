# Generated facades: what the scan can and cannot supply

This records a negative result and the pipeline built on the back of it. The
short version: **the reality mesh's own surface is not shippable, cannot be made
shippable, and does not need to be.** It is a measuring instrument. The geometry
is built from what it measures.

## The measurement that closed the other path

Helsinki's mesh is 7.6 cm/texel and 42 cm triangles — dense sampling, and dense
sampling of an over-smoothed surface is still over-smoothed. Against a local 3 m
reference the wall is flat to **0.044 m**, and a window sits **0.05 m** off its
own wall. The reveals are below the noise floor. A global comparison makes them
look present (window pixels read 0.65 m further than masonry) but that is the
facade's own lean correlating with which parts of it are in shade.

So `reveal_mask` refuses when the local noise cannot support a reveal, and
reports the noise floor it measured. It finds building-scale recesses — a light
well, a gap between blocks — and no windows.

### Three ways round it, all measured, two of them dead ends

High-frequency energy at one scale on a 48-bay building:

| | |
|---|---|
| one measured bay | 0.0173 |
| median of 48 bays | **0.0084** — the vote *halves* it |
| sharpest single bay | 0.0247 |

- **`repair`** replaces bays that disagree with the median. This works: eight of
  sixty-six visibly improved. It fixes damage that differs bay to bay — one bad
  look, a sign, an occluding tree.
- **`align`** re-cuts cells registered to the median first, on the theory that
  the blur was registration error. Measured median shift: **0 px**. Theory
  disposed of.
- **`restore`** gives every bay the median's detail while keeping its own colour.
  It *softens* the wall, because the reference it copies from is itself soft.
- **`exemplar`** picks the sharpest typical bay — lucky imaging. Reports a 3.6×
  gain and is not usable: the bay it picks is sharp because it has venetian
  blinds behind the glass. High-frequency energy is not a proxy for a better view
  of a wall.

The smearing is **correlated across bays** — every bay reconstructed from the
same oblique looks — and correlated damage survives any vote or selection over
the same population.

## What the scan does supply

| | source | confidence |
|---|---|---|
| footprint, corners | CityGML | survey-grade |
| base and roof height | mesh | agrees with the city model to **0.20 m** at the median |
| storey height | mesh, 3.31 m | correlation 0.40; register's storey count agrees to 3% |
| bay width | mesh, 3.77 m | correlation 0.44 |
| window extent | average bay | soft but well bounded — a blurred edge still crosses the midpoint where the sharp edge was |
| wall and window colour | mesh, de-lit | low frequency, which is what survives smearing |
| reveal depth | **assumed, 0.15 m** | airborne data does not contain it at any resolution |

## The registration that makes it possible

`semantics/transfer.py` registers the local-coordinate mesh against absolute
EPSG:3879 footprints. The tile name implies a translation; a 625-point search
around it scores candidates, and **the winner is the tile name's own offset** —
the search confirms the derivation rather than rescuing it.

Scoring footprint *interiors* does not work: on a block that is half building,
half the walls land inside some footprint whatever the shift, so the surface came
out flat at 0.465 against a runner-up of 0.461. Distance to the footprint
*outline* is sharp — peak 0.435 against runner-up 0.308, residual 0.5 m, unique.

Roof-height residual by percentile, mesh against model:

| p50 | p75 | p90 | p95 | p99 |
|---|---|---|---|---|
| −0.20 m | +0.64 m | +1.40 m | +1.72 m | +2.56 m |

At the median the two models agree to 0.20 m. The +1.72 m at p95 is chimneys,
plant rooms and railings. Independence is **unverified** — both are Helsinki
products and may share aerial campaigns, so this is consistency between two
models, not accuracy of either.

## Building instead of matching

`reconstruct/elevation.py` builds quads with openings punched through them and
real reveals around the openings — jamb, head and sill — which is depth the
source mesh never contained. Plus plinth, string course, cornice, projecting
sills, mullion and transom bars, a door per street frontage, and a roof cap.

On one 250 m block: **eight buildings, 26,432 faces, 1,299 openings.**

## Material

Two sources of high frequency, and the second is better.

`themes/procedural.py` generates seamless material. It is noise that resembles
masonry. `data/textures.py` fetches CC0 photographed material from **Poly Haven**
— preferred over ambientCG because it publishes each texture's real size in
millimetres, which is exactly what metric UV1 needs.

`features/match.py` reduces a wall and a texture to four descriptors, all in
metres so a 40 m frontage at 13 px/m and a 1 m tile at 1024 px are comparable:
de-lit colour, horizontal course, vertical course, roughness. Coursing compares
as a **log ratio** — 20 mm on a 70 mm brick is a different material, the same
20 mm on a 450 mm ashlar block is the same one cut differently.

Result on the historic core: **Blue Plaster Wall**, distance 0.130, with the next
three also plaster and smooth concrete. Helsinki's core is rendered masonry in
blue-grey.

It was only right after the descriptor learned to refuse. The wall's measured
coursing is 0.083 m in both axes, which at 13.2 px/m is **1.1 source pixels** —
the strongest lag in noise. Weighted at 3.0 it *was* the score, and it chose an
asbestos sheet for a Jugendstil block. The distance now drops coursing entirely
when the source never resolved a course, so this data matches on colour and
roughness and says so.

## UV convention

UV0 is **wall-metres over the tile's real repeat**, never normalised by wall
size. Normalising makes one course of stone span a shed and a warehouse
identically. `wall_frame` keeps coursing running along the wall and stacking
toward the sky, so it does not rotate at a corner.

## Reproduce

```bash
python tools/citygml_join.py --subtile data/helsinki/mesh/672496/673497d1 --storeys
python tools/facade_openings.py     # macro, depth, reveals, lattice
python tools/facade_repair.py --worst 0.12
python tools/build_elevation.py --buildings 8
python tools/textured_wall.py       # one wall, close enough to judge
python tools/texture_match.py       # CC0 ingest, match, apply
```

## Not done

- The matched CC0 material is not yet wired into `build_elevation.py`, which
  still bakes procedural `stone_block`.
- The ground plane exports but does not draw, so the block floats.
- Hipped roof caps are degenerate quads standing in for triangles.
- The software renderer point-samples with no mipmap, so masonry moirés at
  distance. This is the renderer, not the pipeline, and it is currently the
  largest single source of "it looks bad".

# The master idea: one seed, five repositories

Internal. This is the note that says what the five repositories are together,
which of them is the product, and where the seam between them runs. It is
written here because this repository owns the format the seam is made of.

## The five, in one line each

| repository | what it is | licence |
|---|---|---|
| `lidargame-` (**lidarworld**) | a compiler: LiDAR in, a themeable semantic world out | proprietary |
| `Kalasatama` | Helsinki's photogrammetric reality mesh + LoD2 semantics: the one measured source that has walls | proprietary (no `LICENSE` file yet) |
| `CityBuilder` (**NOCTIS-7**) | a walkable ASCII megacity in one HTML file; four bytes per 4 m cell, no geometry at all | MIT |
| `GODOT-GAME` | a Godot 4 project over ~10 vendored MIT/CC0 codebases, plus a designed game (ACCRETE) | MIT |
| `react-native-game-engine` | a fork of bberak's component-entity-system loop for React Native | MIT (upstream) |

## The idea

There is one product across all five, and it is already named in this
repository's own invariant:

> The Spatial IR is theme-independent and engine-independent. Renderers and
> engines are targets.

The other four repositories *are* those targets, plus one unclaimed input. The
consolidation is not a merge of source. It is naming the contract, and then
making every repository either produce it or consume it.

```
  PRODUCERS  (measured reality in)          THE CONTRACT            TARGETS  (worlds out)

  lidargame- / lidarworld ────compiles────►                  ────► CityBuilder / NOCTIS-7
    airborne LiDAR + footprint and                                   web, ASCII, 256 KB, no triangles
    street registers → Spatial IR              World Seed
                                                (.seed.json)   ────► GODOT-GAME
  Kalasatama ─────────────────supplies────►    + Spatial IR            desktop, Steam, the vetted shell
    photogrammetry + CityGML LoD2:              (.lwir)
    the facade detail airborne data                            ────► react-native-game-engine
    physically cannot see                                              mobile, entity/system loop
```

Everything left of the contract measures a place. Everything right of it builds
one. Nothing crosses in source form.

## Why this is the real shape and not a retrofit

Three pieces of evidence, none of them arranged after the fact.

**Two repositories reached the same rule independently.** `lidargame-` says the
seed "keeps what makes a place recognisably itself and throws away the measured
surface." `Kalasatama` says "the measured layer is the reference, never the
shipping asset. Get that backwards in either direction and it fails — reproduce
the mesh faithfully and you ship melted mush; ignore it and you have invented a
city that happens to be named Kalasatama." That is the same sentence about the
same seam, written twice without coordination.

**CityBuilder already ingests measured reality — through its own private
path.** `tools/bake_lidar_city.py` reads USGS 3DEP from S3, subtracts DTM from
DSM, and writes the renderer's texture. It reimplements ingest, ground
modelling, height-above-ground, vegetation discrimination and classification,
none of which it should own, and it is the second place in the estate where
"what is a building" gets decided. Two answers to that question is one too
many.

**The licences forbid the alternative.** `lidarworld` and `Kalasatama` are
proprietary; `CityBuilder`, `GODOT-GAME` and `react-native-game-engine` are MIT,
the last one under an upstream copyright that is not ours. A monorepo, or
vendoring the compiler into a target, either relicenses proprietary work by
accident or contaminates an MIT repository with code that cannot be MIT. The
only join that is legally clean is a **data contract**: a seed file is output,
and output carries the terms of its inputs, not of its compiler.

So the shape is forced, which is the best kind of architecture decision.

## The contract

**The World Seed** (`src/lidarworld/ir/seed.py`, `seed: "lidarworld/0.1"`) is
the interchange format. It is a JSON document holding:

- `bounds`, `crs`, `origin` — where on Earth this is, and the local frame
- `terrain` — a ground grid, `z[ix][iy]` from `bounds[0]` at `step_m`
- `buildings` — a footprint ring, `ground_z`, `height`, roof form
- `roads` — a centreline and a half-width
- `water` — an outline (measured) and a level (inferred)
- `vegetation` — a position, a crown radius, a height
- `regions`, `provenance` — class raster, sources, and what was compressed

It holds **no materials, no theme, no engine, and no measured surface**. A
320 m block of Amsterdam is a few hundred KB against a bundle of tens of MB.

Three properties make it the right seam rather than a convenient one:

1. **It is lossy in the direction of usefulness.** What comes back is not the
   building that was scanned; it is *a* building on that footprint, at that
   height, facing that street. Airborne data never saw the facade, so nothing
   was lost that the sensor had.
2. **The decoder is generative.** A target does not have to resemble the
   original in detail, only in structure. This is why the same seed can drive
   glTF, CityJSON, an ASCII raycaster and (next) a Godot scene without any of
   them agreeing about anything except metres.
3. **It carries its own provenance and its own bad news.** `residual` per
   building, `evidence` on derived terrain, `surface: inferred` on water. A
   target that wants to say what is measured and what is invented can.

The Spatial IR (`.lwir`) remains the richer contract for anything that needs
per-tile context bitmasks and confidence — currently the `web` backend and
forward validation. **Targets should consume the seed; only tooling that needs
the measured surface should consume the IR.**

### Version rule

`seed: "lidarworld/0.1"` is in every file. A target must refuse a major version
it does not know rather than guess. The schema is owned here; `spec/` holds the
normative SIR schema, and when a target and the spec disagree, the target
changes.

## What each repository becomes

### `lidargame-` — the hub, and the only producer of the contract

Unchanged in purpose. It gains one responsibility: the seed is now a published
interface with consumers outside this repository, so a breaking change to it is
a breaking change to three other repositories, and the version field has to
start meaning something.

It also gains the backend that proves the point — see below.

### `Kalasatama` — the facade supplier, not a competing world

This is the answer to known weakness #10 and the reason for known weakness #5.
Airborne LiDAR is ~4 pts/m² from above and two thirds of that is pavement, so a
wall is edge-on and absent rather than sparse; no reconstruction fixes an input
that never saw the surface. Photogrammetry did see it: 7.6 cm/texel, a balcony
is a balcony, a reveal has depth.

Its role in the master idea is **not** to be a second walkable world. It is to
supply, into the seed's expansion side, the things airborne data cannot:
measured storey heights, bay rhythm, opening proportions and colour, per
building type — the `features/repair.py` and `features/match.py` seam. A seed
says "a 24 m building on this footprint facing this street"; Kalasatama says
what 24 m of Helsinki wall actually looks like at 0.05 m reveal depth.

Its own honest limits carry across: the mesh is 2017 and the CityGML 2019, so
some buildings do not register; LoD2 has no openings, so every window comes from
the mesh; there are no interiors.

### `CityBuilder` — the target that proves engine-independence

The most valuable property of NOCTIS-7 is that it is *not* a 3D engine. No
triangles, no materials, no scene graph, no meshes — a world reaches it as a
256 KB texture and a raycast per character cell. `web`, `gltf` and `cityjson`
are three ways of writing down the same triangles, so between them they cannot
tell you whether the IR is engine-independent or merely glTF-shaped. This one
can.

**Built now:** `src/lidarworld/backends/noctis.py` bakes a World Seed into the
renderer's exact texture format, with tests. See "State" below.

**Still owed:** `tools/bake_lidar_city.py` should become a thin wrapper over
`lidarworld fetch` + `compile --seed` + `noctis`, so "what is a building" is
decided in one place. Until then the two paths will drift, and the ASCII city
and the glTF city will disagree about the same block.

### `GODOT-GAME` — the shipping engine, and a game that is not this pipeline

Two things live here and they should not be confused.

The **infrastructure** is directly in the master idea: ~10 vendored MIT/CC0
codebases surveyed in `docs/GODOT_CODE_SURVEY.md`, a complete game shell
(menus, options, pause, threaded loads, saves), Terrain3D, phantom-camera,
beehave. A Godot backend is the shortest route from a seed to something
shippable, and the licence policy here (MIT/Apache/CC0 only, art and code
audited separately, every import recorded) is the estate's best one and should
be the model for the others.

**ACCRETE is orthogonal and should be said so plainly.** It is a 2D
action-incremental about accreting mass onto a star. It shares no geometry, no
data and no pipeline stage with a measured city. It is a good separate product
with a real market case, and pretending it is part of the city pipeline would
be the kind of retrofitting this document is meant to avoid. What it does
contribute is the proof that the shell in this repository can carry a finished
game to a store page.

**Still owed:** a `godot` backend — a GDScript `WorldSeed` loader plus a scene
builder that stands generated geometry at measured coordinates, using the
Starter-Kit-City-Builder placement patterns already vendored here.

### `react-native-game-engine` — the mobile target and the entity contract

The odd one out, and honest about it: it is an upstream MIT fork of bberak's
library, not our code. Its role is a target and a pattern, not a dependency.

The pattern matters more than the platform. RNGE's contract is
`entities × systems × time → entities`: a world is a dictionary of entities and
a list of pure functions over them. That is the same shape a seed expansion
wants — buildings, roads and trees are entities; theming, LOD and traffic are
systems. A seed → entities adapter is the smallest possible fourth target and
would be a strong check that the contract is not secretly 3D-shaped.

**Constraint:** never merge our proprietary code into this repository. It
carries a third party's copyright. Anything we write for it goes in as a
separate, clearly-owned module or, better, in a repository of ours.

## State: what is true today

Built and tested in this repository as part of this consolidation:

- `src/lidarworld/backends/noctis.py` — World Seed → NOCTIS-7 city texture.
  Registered in `BACKENDS`, exposed as `lidarworld noctis`, 10 tests in
  `tests/test_noctis.py` including a from-scratch PNG decoder (the file has to
  be readable by something that is not our writer).
- Style is derived from measured quantities only — height and roof form — and
  `street_facing` is computed on the 4 m grid the same way the IR defines it.
  A test asserts that no material, theme or engine name reaches the output.

Smoke run, on **synthetic** data from `tools/make_sample_data.py` (labelled
synthetic because it is: no real place was measured here):

```
127,538 points  →  6.7 MB .lwir  →  4.4 KB seed  →  3.5 KB city texture
                                     6 buildings, 25 trees, 1 road
                                     7,203 building cells, 5,584 road, 1,051 park
```

The one bad number from that run, reported rather than smoothed: compiling the
synthetic block **without** a footprint register yields *zero* buildings in the
seed, because seed buildings come from `extrude` programs and those come from
footprints. The 6 above were reconstructed from the compiler's own building
bounding boxes. Airborne data alone still does not produce buildings without a
register — that is known weakness #5 and it reaches all the way to the last
target in the chain.

## Not built

In build order, cheapest and most load-bearing first:

1. **`CityBuilder` stops ingesting LiDAR itself.** Make `bake_lidar_city.py`
   call the shared path. One definition of "building", one of "ground".
2. **A `godot` backend.** Seed → GDScript scene. Highest value per hour: it is
   the only target on the list that can reach a store page.
3. **Kalasatama's facade library feeds the expansion.** Measured storey
   heights, bay rhythm and opening proportions into `features/repair.py` and
   `features/match.py`. This is the only work that improves how every target
   looks at once.
4. **A seed → RNGE entities adapter.** Smallest fourth target; checks the
   contract is not 3D-shaped.
5. **A conformance test for the contract itself,** in `spec/`, that a target in
   any language can run: given this seed, these are the buildings, roads and
   ground levels you must agree about.

## The numbers this idea has to live with

Carried here rather than left in each repository's own notes, because a
consolidation that hides its worst inputs is a brochure.

| measurement | value | what it means |
|---|---|---|
| forward validation, explained returns | **~28%** | 5,421 rays hit geometry that is not there; 5,010 pass through geometry that should be |
| building height vs Denver aerial stereo | median **1.18 m** (72% within 2 m) | the only level-2 independent check the compiler has |
| vegetation over-segmentation, Amsterdam | **6.8×** | 1,243 trees compiled where BGT surveyed 183 |
| topology grouping | 1,411 patches → **1,184** structures | almost no merging; airborne roof patches rarely touch |
| Manhattan height model, transferred | **R² = −0.065** | worse than predicting the mean; the fit does not transfer down in scale |
| Helsinki facade high frequency | median of 48 bays carries **half** the detail of one bay | smearing is correlated across bays; a vote cannot recover it |

None of these are blockers for the master idea, and one of them is an argument
*for* it: if a facade reconstructed to the centimetre is worth nothing a
generator cannot invent, then the seam belongs exactly where the seed puts it.

## Rules that follow from all this

1. **The seed is the only thing that crosses a repository boundary.** Not
   source, not meshes, not textures. If a target needs something the seed does
   not carry, add it to the seed and version it — do not reach past it.
2. **No target names a material before the backend boundary.** This survives
   into NOCTIS-7: the ASCII renderer picks its own palette from a style index
   and a height byte.
3. **Proprietary source never enters an MIT repository, and MIT-upstream code
   never absorbs ours.** `react-native-game-engine` in particular is a third
   party's copyright.
4. **Measured and generated never blur, in any repository.** Kalasatama's rule
   is the estate's rule.
5. **`Kalasatama` needs a `LICENSE` file.** Its `CLAUDE.md` states
   proprietary, all rights reserved; there is no artifact saying so in the
   repository. Stated policy without a file is not a licence.

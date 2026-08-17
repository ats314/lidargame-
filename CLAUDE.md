# Working notes for Claude

Read this before starting anything in this repo.

## What this is

A compiler: LiDAR point clouds in, a themeable semantic world out. The Spatial
IR in the middle is the product; renderers and engines are targets.

**This is a product, not a paper.** Optimise for the pipeline working on messy
real input without hand-holding. Reconstruction quality, error handling,
performance on large tiles, install and onboarding, and the viewer feeling good
all beat rigour theatre. Benchmarks exist as regression tests, not as results
to publish.

**Audience is internal.** Docs stay functional, not persuasive. Effort that
would have gone into making the README convincing goes into making the
reconstruction better instead.

## Working agreement

**Push constantly.** Commit and push at every checkpoint where tests are green
— minimum every ~20 minutes of real work. Never hold work locally to make it
presentable first. An hour of invisible progress is a failure regardless of
what the code does.

**Merge without asking.** Open a PR, merge it yourself once CI is green. No
approval needed. `main` is unprotected on purpose. Stop and ask only for:
licence changes, deleting data, force-pushing `main`, or anything that costs
money.

**Don't go dark.** A one-line status at every push. Batch questions; ask only
when the answer changes the architecture. Never block waiting for an answer if
there is other work that can proceed.

**Use `AskUserQuestion`.** When there is a real decision, use the tool with
concrete options, not a wall of prose.

**Keep replies short.** Lead with the result and the link. Three sentences
unless depth is requested. Detail belongs in the PR body and in `docs/`.

## Ground rules

**Licence: proprietary, all rights reserved.** Never add a permissive licence.
Never publish to PyPI, npm or any registry. That is about *our* licence; other
people's terms are recorded, not enforced -- `describe()` hands back every
source with its `license` line and `commercial` flag, and refuses nothing.

**Third-party licences are the owner's call, not yours.** Record what a source's
terms are in `src/lidarworld/data/catalog.py` and move on — do not gate work on
them, do not ask, do not refuse a dataset over licence ambiguity. Use the data
you have unless there is a hard technical reason not to.

**Synthetic data is scaffolding only.** `tools/make_sample_data.py` exists so
tests and CI run without downloads. It must be labelled as synthetic in every
artifact it touches. Any demo a human will actually look at uses real
public-domain data (`lidarworld fetch <place>`). Never let a generated scene
stand in for a result.

**Report the bad number.** Keep "measured" and "inferred" separate everywhere,
surface the failing metric prominently, and never round a weakness away. If
forward validation says 18% of returns are explained, that is the headline, not
a footnote.

## Definition of done

Tests pass → CI green → merged to `main` → deployed. All four, or it is not
done.

Never merge red CI. Never skip, disable or loosen a test to get green — if a
test is wrong, fix the test and say why in the commit. If a check is genuinely
blocked, say so plainly rather than routing around it.

## What we are actually optimising

Not a perfect Denver. Enough structure from Denver to generate a coherent
Denver-*like* world that someone can walk around in.

    reality -> semantic lossy compression -> World Seed -> generation -> game world

The seed keeps what makes a place recognisably itself -- ground, streets,
footprints, heights, tree positions -- and throws away the measured surface.
A 320 m block of LoDo is 121 KB against a 92 MB bundle. The building that comes
back is not the building that was scanned; it is *a* building on that footprint
at that height facing that street, which is what a game needs and what airborne
data cannot supply anyway, having never seen the facade.

This is why reconstruction accuracy is a means, not the goal. Forward validation
still matters -- it is what stops the skeleton being wrong -- but a facade
reconstructed to the centimetre is worth nothing the generator cannot invent.

## The invariant everything hangs off

**The Spatial IR is theme-independent and engine-independent.** No stage before
`src/lidarworld/backends/` may name a material, a shader, a texture or an
engine. Materialisation happens at the backend boundary, once.

This is what makes a theme swap a lookup instead of a recompile. If a change
would violate it, stop and flag it — do not work around it.

## Spec authority

`spec/` holds the normative SIR v0.1 schema and round-trip benchmark.
`src/lidarworld/` is the compiler that feeds it.

**When they disagree, the compiler changes, not the spec.** This has already
caught a real error: reconstructed walls were being marked `observed`, and the
invariant correctly rejected it — only the point cloud was observed.

## Repo map

```
src/lidarworld/
  ingest/       LAS/LAZ, KITTI, PCD, PLY, XYZ behind one adapter interface
  spatial/      sparse voxel indexing, moment aggregation
  features/     multiscale PCA descriptors, terrain, height above ground,
                pluggable partitioner seam (voxel default, SPT backend);
                frequency.py: macro x micro separation and de-lighting;
                openings.py: reveals from depth, bay/storey lattice;
                repair.py: the average bay, and what a vote cannot recover;
                match.py: measured wall -> CC0 material, in metres
  semantics/    transfer.py: register a local-coordinate reality mesh against
                absolute CityGML footprints and stamp every triangle with a
                building id and surface class;
                vocab.py: one label table per public benchmark (ASPRS,
                SemanticKITTI, DALES, Toronto-3D, Paris-Lille-3D, nuScenes);
                infer.py: rule-based inference where labels are absent
  roles/        role taxonomy, context bitmask, CityGML alignment
  segment/      planar region growing, tree/vehicle/pole instancing
  reconstruct/  tile lattices, opening detection, meshing;
                elevation.py: build a clean facade from measured numbers --
                openings with real reveals, plinth, cornice, sills, roof
  topology/     relations, building grouping, street frontage
  themes/       packs, resolver, procedural texture backend
  backends/     web, glTF, CityJSON  <- the only place materials exist
  ir/           .lwir reader/writer, SIR v0.1 exporter,
                program.py: generative programs + their measured residual,
                seed.py: the World Seed a generator expands into a place
  data/         source catalogue, tile fetcher, header-only tile index,
                textures.py: CC0 material libraries with their terms and each
                texture's published real-world size,
                catalog_index.py: published extents -> download URLs,
                denver.py: acquisition manifest with independence levels
  validate.py   forward LiDAR simulation, consistency scoring
spec/           normative SIR v0.1 schema + benchmark (authoritative)
docs/GENERATED_FACADES.md
                why the scan cannot be the surface, and what is built instead
viewer/         dependency-free WebGL2 walkthrough
```

## Commands

```bash
pip install -e ".[dev,laz]"
python -m pytest tests/ -q                    # 375 tests, ~10s
python spec/benchmark/smoke_test.py           # spec conformance

lidarworld sources                            # what is commercial-use clear
lidarworld fetch denver_lodo -o data/real     # pull a real 3DEP tile
lidarworld tiles data/real --area x,y,size    # header-only index; what covers this?
lidarworld tiles . --remote --area=lon,lat,deg # what to DOWNLOAD; 6,505 Denver tiles
lidarworld compile data/real --area x,y,size -o build/x \
  --footprints denver --streets denver --theme victorian --theme neon --sir
python tools/shoot.py --out build/shots        # render it and LOOK at it

# Helsinki: measure a real block, build a clean one, texture it
python tools/citygml_join.py --storeys         # register mesh to CityGML, stamp it
python tools/build_elevation.py --buildings 8  # measured numbers -> clean geometry
python tools/textured_wall.py                  # one wall, close enough to judge
python tools/texture_match.py                  # CC0 ingest, match, apply
lidarworld validate build/x/x.lwir --scan scan.bin
```

## Known weaknesses — do not rediscover these

Ranked by how much they hurt, which is roughly the order to fix them.

1. ~~Facades come out as swiss cheese.~~ Fixed: coplanar patches merge per
   footprint before tiling.
2. ~~Buildings do not meet the ground.~~ Fixed: wall columns extend to terrain,
   and footprints are extruded into walls where airborne data has none.
3. **Building heights agree with Denver's aerial stereo to a median 1.18 m**
   (72% within 2 m, 90% within 5 m, median bias +0.40 m: LiDAR reads slightly
   taller, consistent with seeing parapets photogrammetry digitises past). This
   is the *only* level-2 independent check the compiler has -- different sensor,
   different failure modes -- and it is cheap. Add more of these before trusting
   any self-consistency number.
4. **Forward validation explains ~28% of returns** (Embree backend). 5,421 rays
   hit geometry that is not there and 5,010 pass through geometry that should
   be. Both are real -- swapping the voxel raycaster for exact ray-triangle
   intersection collapsed the inconclusive-grazing bucket from 10,307 to 2,333
   and raised explained from 18.5%, which proved most of the old over-occlusion
   was measurement artefact but left this residue as genuine. Symptom of 1 and
   2. This number is the scoreboard; move it.
5. ~~Airborne data barely produces walls.~~ Addressed by extrusion: 3DEP is
   ~4 pts/m2 from above, two thirds of it on pavement, so facades are absent
   rather than sparse. `--footprints <id>` extrudes them from the footprint to
   the measured roof height. Denver went from 50 walls to 872.
6. ~~**Vegetation over-segments.**~~ Fixed. Was 1197 "trees" in a 300 m Denver
   block with a 74 m maximum; now 307 with a 28.9 m maximum. Three causes, all
   real: return number was discarded at ingest (it is *the* canopy discriminator
   -- roof 9.3% multi-return vs scatter 70.7%), the CHM was unsmoothed with a
   fixed 3x3 window, and there was no suppression between neighbouring peaks.
   The remaining count is still probably high; there is no airborne ground truth
   in the repo to check it against, which is what DALES is for.
7. **Topology barely groups.** 1411 patches became 1184 "structures" — almost
   no merging. Airborne roof patches rarely touch, so `relate_patches` finds
   little. Needs footprint-based grouping, not just patch adjacency.
8. **Highlighting terrain tints the whole world**, because terrain is a single
   node. Scope the viewer's highlight to a face.
9. **Look at the render before believing a metric.** Every number can be fine
   while the world looks like nothing. `tools/shoot.py` drives headless Chromium
   over the viewer and writes PNGs; it is how the fog was found drowning the
   block at 63% opacity by 100 m, and how 1,315 phantom roof "windows" were
   spotted as pink speckle. Neither showed up in any metric. Since then it also
   caught a reveal mask that was three bands of crop edge and no windows, a
   lattice phase sitting on every pier instead of every window, and a "sharpest
   bay" that was sharp because it had venetian blinds in it.
10. **A photogrammetric facade cannot supply high frequency, and no
    recombination of it can.** Helsinki is 7.6 cm/texel and 42 cm triangles, and
    the wall is flat to 0.044 m locally against a 0.05 m window reveal. The
    median of 48 bays carries *half* the detail of one bay -- the smearing is
    correlated across bays because they share look angles, and a vote only
    removes independent damage. `docs/GENERATED_FACADES.md` has the numbers and
    the three failed attempts. The macro supplies identity (colour, extent,
    rhythm, height); structure is generated or matched from a CC0 library.
11. **The software renderer is currently the largest source of "it looks bad".**
    `tools/glb_shot.py` point-samples with no mipmap, so masonry moirés at
    distance, and it had no lighting model at all until a flat Lambert was added
    -- which meant cornices, reveals and projecting sills returned base colour
    and a relieved facade rendered as a paper cut-out. Before blaming geometry
    for a bad render, check what the renderer is capable of showing.

## Things that wasted time before

- Building on synthetic data for an hour without saying it was synthetic.
- Adding an MIT licence nobody asked for.
- Absolute thresholds in tests that pass locally and fail on another numpy
  version — make assertions proportional.
- Backticks in `git commit -m` heredocs trigger shell substitution. Use
  `git commit -F file`.
- Reading every `.obj` in a ContextCapture subtile. It is a quadtree written out
  in full: 216 files across nine levels, 64 of them leaves, the rest the same
  ground at coarser sampling. Merging them stacked 3.4x the triangles a few
  centimetres apart and silently corrupted every Helsinki number for weeks --
  including the claim that 60% of the mesh was webbing, which was mostly coarse
  LOD copies. Real webbing is 2.6%. `read_directory` defaults to leaves now.
- Aiming a camera in the source frame when the exporter converts Z-up to glTF's
  Y-up. It puts the camera under the pavement looking at the underside of it,
  which renders as a grey wedge and reads as broken geometry.
- The integration token has no `actions: write`, so workflows cannot be
  dispatched directly. Trigger them with a push.

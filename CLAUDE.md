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
                pluggable partitioner seam (voxel default, SPT backend)
  semantics/    vocab.py: one label table per public benchmark (ASPRS,
                SemanticKITTI, DALES, Toronto-3D, Paris-Lille-3D, nuScenes);
                infer.py: rule-based inference where labels are absent
  roles/        role taxonomy, context bitmask, CityGML alignment
  segment/      planar region growing, tree/vehicle/pole instancing
  reconstruct/  tile lattices, opening detection, meshing
  topology/     relations, building grouping, street frontage
  themes/       packs, resolver, procedural texture backend
  backends/     web, glTF, CityJSON  <- the only place materials exist
  ir/           .lwir reader/writer, SIR v0.1 exporter
  data/         licence catalogue, tile fetcher, header-only tile index
  validate.py   forward LiDAR simulation, consistency scoring
spec/           normative SIR v0.1 schema + benchmark (authoritative)
viewer/         dependency-free WebGL2 walkthrough
```

## Commands

```bash
pip install -e ".[dev,laz]"
python -m pytest tests/ -q                    # 115 tests, ~3s
python spec/benchmark/smoke_test.py           # spec conformance

lidarworld sources                            # what is commercial-use clear
lidarworld fetch denver_lodo -o data/real     # pull a real 3DEP tile
lidarworld tiles data/real --area x,y,size    # header-only index; what covers this?
lidarworld compile data/real/*.laz -o build/x --bbox minx,miny,maxx,maxy \
  --theme victorian --theme neon --sir
lidarworld validate build/x/x.lwir --scan scan.bin
```

## Known weaknesses — do not rediscover these

Ranked by how much they hurt, which is roughly the order to fix them.

1. ~~Facades come out as swiss cheese.~~ Fixed: coplanar patches merge per
   footprint before tiling.
2. ~~Buildings do not meet the ground.~~ Fixed: wall columns extend to terrain,
   and footprints are extruded into walls where airborne data has none.
3. **Forward validation explains ~28% of returns** (Embree backend). 5,421 rays
   hit geometry that is not there and 5,010 pass through geometry that should
   be. Both are real -- swapping the voxel raycaster for exact ray-triangle
   intersection collapsed the inconclusive-grazing bucket from 10,307 to 2,333
   and raised explained from 18.5%, which proved most of the old over-occlusion
   was measurement artefact but left this residue as genuine. Symptom of 1 and
   2. This number is the scoreboard; move it.
4. ~~Airborne data barely produces walls.~~ Addressed by extrusion: 3DEP is
   ~4 pts/m2 from above, two thirds of it on pavement, so facades are absent
   rather than sparse. `--footprints <id>` extrudes them from the footprint to
   the measured roof height. Denver went from 50 walls to 872.
5. ~~**Vegetation over-segments.**~~ Fixed. Was 1197 "trees" in a 300 m Denver
   block with a 74 m maximum; now 307 with a 28.9 m maximum. Three causes, all
   real: return number was discarded at ingest (it is *the* canopy discriminator
   -- roof 9.3% multi-return vs scatter 70.7%), the CHM was unsmoothed with a
   fixed 3x3 window, and there was no suppression between neighbouring peaks.
   The remaining count is still probably high; there is no airborne ground truth
   in the repo to check it against, which is what DALES is for.
6. **Topology barely groups.** 1411 patches became 1184 "structures" — almost
   no merging. Airborne roof patches rarely touch, so `relate_patches` finds
   little. Needs footprint-based grouping, not just patch adjacency.
7. **Highlighting terrain tints the whole world**, because terrain is a single
   node. Scope the viewer's highlight to a face.

## Things that wasted time before

- Building on synthetic data for an hour without saying it was synthetic.
- Adding an MIT licence nobody asked for.
- Absolute thresholds in tests that pass locally and fail on another numpy
  version — make assertions proportional.
- Backticks in `git commit -m` heredocs trigger shell substitution. Use
  `git commit -F file`.
- The integration token has no `actions: write`, so workflows cannot be
  dispatched directly. Trigger them with a push.

# Roadmap

Stages, in order. Each stage names what it delivers, the acceptance test that
can *fail*, and the way it is most likely to fool you. Nothing counts as done
without a green test, green CI and a merge to `main`.

Written to be picked up cold. If you are starting a stage, read
[ARCHITECTURE.md](ARCHITECTURE.md) first, then this stage only.

## Rules for every stage

- **Synthetic data is scaffolding.** A synthetic multi-view rig makes fusion
  developable and CI runnable, and it must be labelled synthetic in every
  artifact it touches. It never stands in as a result. A result is real
  phones around a real table.
- **Look at the render before believing a metric.** The city compiler learned
  this expensively — every number can be fine while the model looks like
  nothing. Every stage that produces geometry produces a picture of it.
- **Report the bad number.** Keep measured and inferred separate, surface the
  failing metric first, never round a weakness away.
- **The invariant holds.** No stage before `backends/` names a material, a
  shader, a texture or an engine.
- **Push at every green checkpoint.** Roughly every 20 minutes of real work.

## Stage 0 — the synthetic rig

**Deliver.** A generator that places N virtual phones around a virtual table
holding a few objects, and emits exactly the observation record from
ARCHITECTURE.md Layer 0: RGB, depth, confidence, intrinsics, pose — with
ground-truth geometry and ground-truth `T_table<-i` retained separately.
Deliberate occlusion: at least one object fully hidden from at least one phone.

Model the sensor's real failures or the rig is a lie: depth noise growing
with range, edge smear at depth discontinuities, dropouts at grazing
incidence and on dark/glossy surfaces, pose noise, and slow pose drift.

**Accept.** Ground-truth-posed depth reprojects to within a tolerance of
ground-truth geometry, and the occlusion is genuinely there — the hidden
object appears in zero of phone A's returns and a useful number of phone B's.

**Fools you.** A rig with no sensor noise makes every later stage look
excellent and predicts nothing about a real table.

## Stage 1 — frames and registration

**Deliver.** `T_table<-i` estimation. Fiducial path first (a marker board of
known geometry), then a geometric path (coarse alignment, then point-to-plane
ICP against the fused model), then drift correction on a pose graph.

**Accept.** With Stage 0 pose noise applied, recovered `T_table<-i` is within
a stated translation and rotation tolerance of ground truth, for every phone.
State the tolerance in the test as a number and justify it against the voxel
size — registration error below one voxel is the bar.

**Fools you.** Testing registration only on the same synthetic scene it was
tuned on. And symmetry: a square table with symmetric contents has ICP basins
that are confidently, catastrophically wrong.

## Stage 2 — weighted TSDF fusion

**Deliver.** The core. Voxel grid in the table frame, weighted integration,
three-state occupancy with genuine free-space carving, Marching Cubes gated
on accumulated weight. Weighting per ARCHITECTURE.md Layer 2, each factor
individually switchable so its contribution is measurable.

**Accept.** Two tests, both of which can fail:
1. **Surface accuracy** — fused surface against Stage 0 ground truth, as a
   distribution, not a mean. Report the median and the tail.
2. **The occlusion test, which is the point** — the object hidden from phone
   A is reconstructed to comparable accuracy as one visible to all phones.
   If it is not, the multi-phone premise is not paying and everything after
   this is premature.

Plus an ablation: fusion with each weighting factor disabled. If disabling a
factor does not move the number, delete the factor.

**Fools you.** Unobserved space quietly becoming confident surface. Check the
weight field, not just the mesh — and render it.

## Stage 3 — the scoreboard

**Deliver.** Forward depth simulation. Render the fused model from a phone's
pose, compare with that phone's measured depth, report fraction of returns
explained, split into: explained, hit-nothing, passed-through-geometry.

Then the strong version: **hold a phone out entirely.** Fuse from N-1,
predict the held-out phone's depth. Level-2 independent — a different
viewpoint with different failure modes.

**Accept.** Both metrics run and report on the synthetic rig. There is no
target number yet; the first honest measurement *sets* the baseline, and it
is quoted in the docs whatever it is.

**Fools you.** Validating against the same views you fused from. Self-
consistency is nearly free and nearly meaningless; the held-out number is
the one that matters.

## Stage 4 — objects and motion

**Deliver.** Instance segmentation of the fused field (table plane, then
connected components above it), frame-to-frame tracking, and the object graph
with per-object pose, extent, visibility and confidence over time.

**Accept.** Move an object in the synthetic rig between frames. The object's
pose updates, its identity persists, and **no ghost remains at the old
position**. Assert on the absence of the ghost explicitly — it is the failure
mode this stage exists to prevent, and a static TSDF has no native defence
against it.

**Fools you.** Tracking that works when one object moves slowly and collapses
when two similar miniatures swap places. Test that case.

## Stage 5 — real phones

**Deliver.** The capture contract on a real device: stream posed RGB + depth
+ confidence off the phone in the Stage 0 record format. Then the same
pipeline, unchanged, on a real table.

**Accept.** A real table, at least three phones, held-out-phone validation
run and reported. And a picture — several, from angles nobody fused from.

**Fools you.** Everything. This is where the synthetic rig's optimism gets
priced. Expect the number to be much worse than Stage 3 and report it
plainly; the gap between synthetic and real *is* the finding.

## Stage 6 — appearance

**Deliver.** Colour on the geometry. Projected RGB first, since it is simple
and reveals registration error immediately as ghosted texture. Then evaluate
the LiDAR-constrained Gaussian route for appearance while geometry stays
metric.

**Accept.** Side-by-side renders. This stage is judged by eye and says so.

**Fools you.** Blaming geometry for a bad render before checking what the
renderer can show. The city compiler lost real time to a software renderer
with no mipmaps and no lighting model — cornices returned base colour and a
relieved facade rendered as a paper cut-out. Check the renderer first.

## Before Stage 0

One afternoon, not more: the literature sweep at the end of
[PRIOR_ART.md](PRIOR_ART.md). Multi-device cooperative RGB-D reconstruction,
distributed depth fusion, multi-camera occlusion resolution. Find out whether
the seam is occupied before building into it.

# Prior art

What already exists, what each piece contributes, and where the gap is.
Links checked to resolve; the assessments are ours.

## Papers

| Paper | Why it matters |
|---|---|
| [High-resolution topographic surveying and change detection with the iPhone LiDAR](https://www.nature.com/articles/s41596-024-01024-9) — Nature Protocols, 2024 | What Apple mobile LiDAR can actually measure, with acquisition and reconstruction workflow. Read it for the error budget before designing around the sensor. |
| [Mobile Phone Based Indoor Mapping](https://isprs-archives.copernicus.org/articles/XLVIII-2-2024/415/2024/index.html) — Strecha, Rehak, Cucci, 2024 | The most directly relevant. iPhone LiDAR + imagery, handles drift explicitly, and combines **multiple independent scans** into one representation. Our `T_table<-i` problem, one phone at a time. |
| [Accurate and complete neural implicit surface reconstruction in street scenes using images and LiDAR point clouds](https://www.sciencedirect.com/science/article/pii/S0924271624004829) — Shi et al., 2025 | RGB + LiDAR jointly optimising an implicit SDF rather than triangulating points. Reports ~7 points of F-score over a prior image/LiDAR neural method. Wrong scale, right architecture. |
| [SHINE-Mapping](https://arxiv.org/abs/2210.02299) — ICRA 2023 | LiDAR into a sparse hierarchical learned SDF. The argument for a continuously refined field instead of a mesh regenerated per frame. |
| [SDFConnect](https://openaccess.thecvf.com/content/CVPR2024W/DLGC/html/Jignasu_SDFConnect_Neural_Implicit_Surface_Reconstruction_of_a_Sparse_Point_Cloud_CVPRW_2024_paper.html) — CVPRW 2024 | Topological constraints so surface reconstruction does not fall apart on incomplete observation. Directly aimed at our occlusion case. |
| [NeuralIndicator](https://proceedings.mlr.press/v235/huang24b.html) — ICML 2024 | Built for incomplete, noisy clouds with complex topology. The candidate for filling occlusion gaps *honestly* — note that anything filled is inferred, never observed. |
| [OffsetOPT: Explicit Surface Reconstruction without Normals](https://openaccess.thecvf.com/content/CVPR2025/html/Lei_OffsetOPT_Explicit_Surface_Reconstruction_without_Normals_CVPR_2025_paper.html) — CVPR 2025 | Points straight to triangles without reliable normals. Consumer depth normals are poor; this sidesteps them. |
| [LI-GS](https://arxiv.org/abs/2409.12899) | LiDAR geometry constraining Gaussian splats, with mesh extraction after. The appearance layer. |
| [Structured-Li-GS](https://arxiv.org/abs/2606.27509) — 2026 | LiDAR-inertial-visual SLAM to colourised cloud to a geometry-anchored Gaussian representation. Closest to a modern full-stack scanner architecture. |

## Code

**[Open3D](https://github.com/isl-org/Open3D)** — the foundation to start from.
Point clouds, filtering, registration, ICP, pose graphs, RGB-D integration,
TSDF voxel grids with voxel hashing, GPU paths, visualisation. Most of Layers
1 and 2 exist here already. Its TSDF integration is approximately where the
prototype should begin; writing our own before we have measured a problem
with theirs is exactly the rigour theatre we do not do.

**[R3LIVE](https://github.com/hku-mars/r3live)** — LiDAR + IMU + RGB into a
real-time coloured 3D map, with offline mesh and texture tooling. Its
architecture is the separation we want: LiDAR carries geometry, camera
carries appearance, IMU assists pose. That maps onto phones cleanly.

**[FAST-LIVO2](https://github.com/hku-mars/FAST-LIVO2)** — direct
LiDAR-inertial-visual odometry and mapping, positioned for real-time
reconstruction. ARKit already supplies much of the equivalent pose
estimation, so we likely do not reimplement its estimator — but the
data-fusion architecture is worth studying.

**[FAST-LIO2](https://github.com/hku-mars/FAST_LIO)** — incremental
scan-to-map registration against a continually updated map. The single-sensor
form of our problem; ours is harder because each phone `i` needs
`T_table<-i` as well as `T_i(t)`.

**[SHINE-Mapping](https://github.com/PRBonn/SHINE_mapping)** — the implicit
alternative to a voxel grid: store sparse hierarchical features whose decoder
predicts signed distance, surface is the zero set. Keep in reserve behind a
TSDF-first prototype.

**[StreetRecon](https://github.com/SCH1001/StreetRecon)** — code for the 2025
image+LiDAR neural reconstruction paper: adaptive RGB weighting, hierarchical
B-spline hash encoding, spatial-hash SDF. Street scale, transferable maths —
the adaptive RGB weighting in particular is the closest published thing to
the multi-view weighting in ARCHITECTURE.md.

## The four generations

1. **Point clouds** — LiDAR to `{x,y,z}`. Simple, fast, noisy. Open3D, PCL.
2. **TSDF / voxel fusion** — frames into a truncated SDF volume, Marching
   Cubes out. Still the best engineering starting point, and where we start.
3. **Neural SDF** — RGB + LiDAR into a continuous learned field. Better
   completion and continuity. SHINE, StreetRecon, NeuralIndicator.
4. **LiDAR-constrained Gaussians** — very high visual quality retaining
   metric LiDAR geometry. LI-GS, Structured-Li-GS.

We do not have to pick one. Geometry at generation 2 (metric, debuggable,
simulatable), appearance at generation 4 (looks right), with generation 3
held in reserve for occlusion filling.

## The gap

Every component of the tabletop twin is published and mature somewhere. What
appears not to be published is the combination:

> several consumer LiDAR phones simultaneously observing one compact
> interactive scene from a perimeter, resolving each other's occlusions
> through a disagreement-aware weighting, and maintaining persistent semantic
> object state as things move.

Three specifics the precedents do not do together:

1. **The scanners are simultaneous and co-located.** The mobile-mapping
   literature moves one sensor through a scene over time. Here N sensors
   observe the same small scene at the same instant, so voxel disagreement is
   a live signal rather than a registration artefact.
2. **Occlusion is resolvable, not just fillable.** The neural-completion work
   infers what it cannot see. Here a neighbour usually *can* see it, so the
   right answer is a correct three-state occupancy (occupied / empty /
   unobserved) rather than a plausible hallucination.
3. **The scene moves and the state is the product.** Reconstruction papers
   target a static scene and score a mesh. Here the deliverable is a live
   object graph, and the scoreboard has to include tracking through movement,
   not just surface accuracy.

Point 3 is also where it stops being 3D scanning and becomes a maintained
digital twin — which is the thing worth building.

## Unclaimed, and what to do about it

Before writing a line of fusion code, run the follow-up sweep this document
did not: **multi-device cooperative RGB-D reconstruction, distributed depth
fusion, multi-camera occlusion resolution**. If the seam above is already
occupied, better to know in an afternoon than in a month. If it is not, the
weighting function and the held-out-phone validation are the contributions.

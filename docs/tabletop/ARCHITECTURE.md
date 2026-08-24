# Architecture

The shape of the system, the maths that has to be right, and the decisions
still open. Design only — no part of this has been measured yet.

## The one-line version

    N posed RGB-D streams -> common table frame -> weighted TSDF -> surface
                                                -> object graph -> temporal state

Do *not* build `phone -> point cloud -> mesh`. That discards confidence,
incidence angle, colour agreement and the fact that two phones disagreeing
about a voxel is information rather than noise.

## Layer 0: what a phone actually gives you

Per phone `i`, per frame `t`, an observation:

    O_i(t) = { I_i(t), D_i(t), C_i(t), K_i, T_i(t) }

- `I_i` — RGB image
- `D_i` — depth map, metres, from the LiDAR + fused stereo
- `C_i` — per-pixel confidence, if the platform exposes it (ARKit does, as
  three levels; treat it as ordinal, not metric)
- `K_i` — intrinsics
- `T_i(t)` — pose in that phone's *own* session frame

The pose is the trap. `T_i(t)` is in phone `i`'s arbitrary session frame,
which drifts, and which has no relationship to phone `j`'s. Everything
downstream needs

    X = T_table<-i * T_i(t) * D_i(p) * K_i^-1 * p~

and `T_table<-i` is the whole ballgame. See **Registration** below.

Known properties of the sensor that the fusion has to respect, not wish away:

- range accuracy degrades with distance and with grazing incidence
- dark, glossy and transparent surfaces return nothing or return garbage —
  painted miniatures with gloss varnish are a real hazard
- depth is upsampled from a sparse dot pattern, so edges are smeared;
  a depth discontinuity is where the sensor is least trustworthy
- the phone is a thermal and battery-bound device streaming continuously

## Layer 1: registration into the table frame

Three candidate mechanisms, in increasing order of how much we should trust
them and decreasing order of how easy they are:

1. **A fiducial on the table.** A printed marker board of known size defines
   the table frame outright; every phone that sees it solves `T_table<-i`
   directly. Cheapest, most robust, needs a physical object in the scene.
2. **Geometric cross-registration.** Estimate `T_table<-i` by aligning phone
   `i`'s accumulated cloud to the fused model (ICP / point-to-plane, or a
   feature-based coarse pass then ICP). No props, but needs enough shared
   observed surface and can converge to a wrong basin on a symmetric table.
3. **Drift correction over time.** Whatever supplies the initial alignment,
   `T_table<-i` is not constant — ARKit's frame drifts. Treat it as a slowly
   varying transform re-estimated on a pose-graph, with the fused model as the
   anchor.

**Decision, open:** start at (1) for the prototype so that fusion can be
developed against a known-good alignment, and treat (2) as the thing that
makes it a product. Do not let a registration bug masquerade as a fusion bug —
the acceptance tests in ROADMAP.md separate them deliberately.

The dominant error term is here. A 5 mm registration error across phones
smears every surface by 5 mm no matter how good the fusion is, and at
tabletop scale 5 mm is a visible fraction of a miniature.

## Layer 2: fusion into a signed distance field

Maintain, over a voxel grid in the table frame, a truncated signed distance

    D(x) = sum_i w_i(x) d_i(x) / sum_i w_i(x)
    W(x) = sum_i w_i(x)

where `d_i(x)` is the truncated signed distance implied by phone `i`'s depth
along the ray through `x`, and `w_i(x)` is that measurement's weight. This is
standard TSDF fusion; what is not standard is that `w` has to carry the
multi-view physics:

    w_i = f(range, incidence angle, sensor confidence, motion, colour agreement)

Concretely, as a starting form — each factor in [0,1], multiplied:

- **range** — falls off past the depth sensor's reliable band
- **incidence** — `max(0, n . v)`, near zero at grazing; grazing returns are
  where the depth sensor lies
- **confidence** — the platform's own per-pixel confidence
- **motion** — down-weight during fast phone motion (rolling shutter, pose
  interpolation error) and near pixels that moved between frames
- **colour agreement** — if two phones see the same voxel and their RGB
  disagrees after white balance, one of them is looking at a specular
  highlight or the surface is not where it is thought to be

**This weighting is the research seam.** The single-sensor literature has
little use for it because there is only one opinion per voxel. With N phones
around a perimeter, disagreement is the signal.

Surface extraction is Marching Cubes over the zero level set, gated on
`W(x) > w_min` so unobserved space stays unobserved rather than becoming
confident flat geometry.

### Why the perimeter matters

Phone A's view of miniature 2 is blocked by miniature 1. Phone B, ninety
degrees round, sees it. Naively the reconstruction is the union

    S = union_i S_i

but that is wrong in a specific and important way: it forgets that A's
*non*-observation of that region was caused by occlusion, not by absence.
A correct fusion carries three states per voxel, not two:

- **observed occupied** — some phone returned a surface here
- **observed empty** — some phone's ray passed *through* here and terminated
  beyond it
- **unobserved** — every phone's ray was blocked short of here

Free-space carving from the second state is what removes the phantom
geometry behind occluders. Collapsing it into "no data" is the single most
common way this class of system produces a bloated, blobby model.

## Layer 3: objects

The field is not the product; a table with things on it is. Above the
geometry sits an object graph:

    miniature_001  pose(t)  extent  visibility(t)  confidence(t)
    terrain_014    pose(t)  ...
    die_003        pose(t)  face_up(t)  ...

Instances come from segmenting the fused field — connected components above
the table plane, then per-instance tracking frame to frame. An object that
moves should update its pose, not be re-fused as new geometry in a new place
while a ghost of it persists in the old one. **Ghosting on movement is the
second most common failure mode of this class of system**, and the SDF has
no native defence against it: a static TSDF only ever accumulates.

Mitigation, in order of ambition:

1. per-voxel temporal decay, so stale evidence fades
2. explicit free-space carving (an object that left is now observed empty)
3. object-level: detach the instance's voxels, transform them, re-attach

**Decision, open.** (2) is probably necessary and sufficient for the
prototype; (3) is what makes the twin feel alive.

## Layer 4: appearance

Geometry and appearance separate cleanly, and should:

- **metric geometry** — the SDF. Answers *where is the surface*.
- **visual surface** — RGB projected onto that geometry, or a
  LiDAR-constrained Gaussian representation. Answers *what does it look like*.

The Gaussian-splatting route (see PRIOR_ART.md: LI-GS, Structured-Li-GS)
gives markedly better appearance at the cost of a representation that is not
a mesh and does not trivially collide, physics or export. Hence keeping them
separate: geometry stays metric and simulatable, appearance is swappable.

**The invariant applies here.** Nothing before the backend boundary names a
material, a shader or an engine. The fused model records what was measured —
colour, roughness proxy, extent. Materialisation happens once, at the
backend, exactly as in the city compiler.

## Module layout, when it lands

    src/tabletop/
      capture/     platform-side contract: what a phone must send
      frames/      pose bookkeeping, T_table<-i, drift, the pose graph
      fuse/        weighted TSDF, free-space carving, surface extraction
      objects/     instance segmentation, tracking, the object graph
      state/       temporal state, decay, confidence
      backends/    the only place materials exist
      validate.py  forward depth simulation, per-phone

Nothing shared with `src/lidarworld/` at import time. If a piece turns out to
be genuinely common, extract it deliberately later — do not couple now.

## The scoreboard

Same discipline as the city compiler: a forward sensor simulation, held out.

Render the fused model from phone `i`'s pose, compare the simulated depth to
what phone `i` actually measured, and report the fraction of returns
explained. Then the stronger version — **hold a phone out entirely**, fuse
from the other N-1, and predict the held-out phone's depth. That is a level-2
independent check of exactly the kind the city compiler has only one of.

Report the bad number. If held-out prediction explains 40% of returns, that
is the headline.

## Open decisions

Answer these with experiments, not arguments:

1. Fiducial or markerless registration for v1?
2. Voxel size — a compromise between miniature detail (sub-millimetre
   features exist) and a grid that fits in memory and updates at rate.
3. Where does fusion run? On one phone, on a host machine, or split? This
   determines the wire format and is hard to change later.
4. Rate. Does the table update at 30 Hz, or does geometry update slowly while
   object poses update fast? The second is almost certainly right.
5. TSDF versus neural SDF. TSDF first — it is debuggable and has no training
   loop. Neural SDF earns its place only if hole-filling under occlusion
   demonstrably needs it.
6. How many phones before returns diminish? Suspect 3-4 covers most tabletop
   occlusion; this is measurable and worth measuring early.

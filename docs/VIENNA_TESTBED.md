# Vienna testbed — execution plan

From the v1.0 handoff, recorded here so the sequence survives a session change.
The directive is narrow on purpose: **do not redesign SWC or add architecture
before this experiment works.**

## Why the move

Denver mixed a hard urban morphology with a weak façade-observation stack, and
the two confounded each other. Vienna removes the confound: it publishes street
panoramas, mobile LiDAR, camera orientation, four-way obliques, orthophoto,
terrain and an authoritative municipal map, all mutually georeferenced.

The measured Denver limits this answers:

| Denver limit | Measured | Vienna |
|---|---|---|
| No façade evidence at all | opening detector reports **0 openings**; walls are extruded, not observed | 360° panoramas ~every 3 m with published pose |
| No per-tree truth | tree count "probably still high" for the life of the repo | `BAUMKATOGD`, every street tree with species and planting year |
| No second independent check | roofprints and outlines are one stereocompilation, **461/503 shared ids** | LoD1.4 / LoD2.1 / LoD2.4, held out |
| Roof form discarded | seed said `flat` for all 257 buildings | `DACHTYP` declared per address |
| No architectural family | every generated building an identical box | `BAUTYP_TXT` typology per building |

## Epoch

Use **2020 throughout** — Kappazunder 2020, 2020 obliques, 2020 orthophoto.
A common epoch removes false disagreement from construction, demolition,
vegetation growth and temporary objects. 2023 is a second-phase cross-epoch
robustness test, not the first target.

## Coordinate frame

Stay in the evidence frame until fusion is done; do not move into a render
frame early.

    horizontal  EPSG:31256  (MGI / Austria GK M34)
    vertical    EPSG:8881   (Wiener Null)
    units       metres

Derive a local origin for rendering only at materialisation, and keep the full
world-to-local transform in the World Seed.

## Three tracks, kept apart

- **Track A, sensor-first.** Kappazunder LAZ + panoramas + pose, obliques,
  orthophoto, DGM. Minimal 2D footprints only if ownership seeding needs them.
  **Held out:** LoD1.4 bodies, LoD2.1 roofs, LoD2.4 detailed roofs, and any
  other finished geometry that supplies the answer.
- **Track B, production fusion.** Adds MZK and Flächen-MZK once A works.
  Everything legally available, provenance preserved.
- **Track C, ablation.** Remove one source at a time and recompile. This is
  what turns Vienna from a demo into an evidence-efficiency benchmark.

## Phases, and the gate

0. **Intake audit.** Enumerate LAZ files, point counts, bounds, attributes,
   RGB presence, coordinate ranges. Enumerate panoramas, verify dimensions.
   Parse trajectory and metadata into a table linking image id, timestamp,
   pose and cloud segment. Verify CRS and height reference. Render a preview
   coloured by RGB and height. **Reconstruct nothing.**
1. **Prove image-to-3D association.** Project 10–20 façade points into their
   panorama using the published orientation, overlay, and measure the pixel
   residual.

   > **Gate 1 — do not proceed to semantic reconstruction until projected 3D
   > points land on the correct physical objects in the source images.**

   This gate is the whole reason for moving here. Denver's registration was
   left `unresolved` on a 2% correlation improvement, and everything built on
   top of an unproven registration inherits the error silently.
2. **One building, not one city.** Ownership, wall and roof evidence, a
   minimal closed shell, material family from observation, openings only at
   high confidence. Materialise it without the point cloud.
3. **One block.** Only after a single building is valid.

## Safety rules, each one a Denver failure

Every one of these is something that actually went wrong, not a hypothetical:

- Every wall and roof surface has exactly one owning entity.
- No surface escapes its footprint without evidence for the overhang.
  *(Denver: flat caps spanned the footprint bounding box and roofed the
  neighbours; survived a whole session behind a street-level camera.)*
- No floating roof plane is committed.
  *(Denver: pitched roofs from a fitted slope became plates larger than their
  buildings.)*
- Roof proposals pass connectivity, support and footprint compatibility, or
  fall back to a simple valid roof.
- Party walls are one shared boundary, not two coincident walls.
- Ground domains stay distinct unless evidence merges them.
- **UNKNOWN is a valid output.** Do not invent structure to avoid uncertainty.

Propose → judge → commit. Highest-supported *valid* hypothesis, else the
simple fallback, else UNKNOWN.

## Acceptance

The benchmark punishes topology errors harder than geometric ones: a visually
elaborate wrong model is a failure.

| metric | target |
|---|---|
| ownership | 100% of committed surfaces have one valid owner |
| floating geometry | 0 committed |
| closed shell | ≥95%, and 100% for the first ten |
| roof support | every non-fallback roof sensor-supported and footprint-compatible |
| ground semantics | no systemic road/rail/sidewalk confusion |
| party walls | represented once, referenced twice |
| appearance | every material names its evidence or is marked generated |

## Acquisition status

`data/vienna.py` records eight WFS layers, all probed live and returning
features over the historic core. Those are fetchable now.

**The mobile mapping is fetchable too.** An earlier version of this section said
Kappazunder could only be obtained by submitting a request form, and called that
a blocker that stopped Phase 0. That was read off the product page and was
wrong. The test datasets are published on data.gv.at under CC BY 4.0 with direct
download URLs — DCAT record `ed24cfff-1361-48d5-a071-31e4c697b844`, mirrored in
`vienna.TEST_DATASETS`. The form route exists for arbitrary areas of the city;
it was never the only route.

Recording the error rather than quietly fixing it, because the failure mode is
the recurring one in this repo: asserting from a page instead of querying the
catalogue. The cost was real — it is what put Amsterdam below into the document
as a "replacement" for a blocker that did not exist.

---

## Candidate: Amsterdam

Kept because the comparison is still worth having, not because Vienna is
blocked. The criterion: **street-level imagery with published pose, downloadable
without a form.** Probed 2026-08-17.

**Amsterdam clears it.** `api.data.amsterdam.nl/panorama` is open, unauthenticated
and returns **8,109,172 panoramas**, each with `heading`, `pitch` and `roll`,
and image URLs that resolve directly on `t1.data.amsterdam.nl` in
equirectangular and cubic form. That is the capability the Vienna move was made
for, available immediately instead of behind a form.

The wider Dutch stack is strong for the same reasons Vienna was: AHN national
LiDAR, and 3D BAG as authoritative building models to hold out. `api.3dbag.nl`
responded to an unauthenticated request.

**Not yet verified, and not to be assumed:**

- Panorama **position** came back null in the list view. Heading/pitch/roll are
  present but a pose without a position is useless for projection, so the
  detail endpoint has to be checked before anything is built on this.
- AHN density and download route over an Amsterdam AOI.
- 3D BAG field structure — the response parsed but the sampled feature exposed
  no properties through the query used.
- Whether the panorama frames are calibrated **measuring** images with an inner
  orientation, or only georeferenced with an exterior heading. Vienna publishes
  inner *and* outer orientation; if Amsterdam publishes only the latter, Gate 1
  needs a pose refinement step against LiDAR rather than a direct projection.

That last point is the one that decides whether Amsterdam is genuinely
equivalent or merely close, and it should be settled before any acquisition.

Denmark remains the other candidate: nationwide LiDAR, GeoDanmark and
nationwide four-way obliques. Its street-level position is unprobed.

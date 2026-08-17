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

## Acquisition status — blocked, and by what

`data/vienna.py` records eight WFS layers, all probed live and returning
features over the historic core. Those are fetchable now.

**The mobile mapping is not.** Kappazunder is obtained by submitting a request
form to the City, not by direct download, and the 5.5 GB test dataset is not
linked from the product page. That is an outward-facing request in a named
person's identity, so it is the owner's action to take, not something to
automate around. See `vienna.KAPPAZUNDER_ORDER`.

Until it lands, Phase 0 cannot start and neither can Gate 1 — which means the
one capability the whole testbed change rests on is untested. Everything else
here is preparation.

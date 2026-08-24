# Tabletop twin

A second product, not a second front end for the city compiler.

Several consumer LiDAR phones stand around a physical tabletop — miniatures,
terrain, dice — and each one streams posed RGB + depth. The system fuses those
views into one persistent, semantic, metric model of the table that survives
occlusion and tracks objects as they move.

Three documents:

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — what gets built and in what order
  the data flows. Coordinate frames, fusion maths, the four layers of the
  world model, and every place a decision is still open.
- **[PRIOR_ART.md](PRIOR_ART.md)** — the literature and the code that already
  exists, what each contributes, and the seam that appears genuinely unclaimed.
- **[ROADMAP.md](ROADMAP.md)** — the staged build, each stage with an
  acceptance test that can fail.

## Why it is a separate product

The city compiler takes airborne LiDAR of a real place and emits a themeable
semantic world. Its defining constraints — ~4 pts/m2 from above, facades never
observed at all, a tile compiled once offline, geometry deliberately thrown
away in favour of a World Seed — are the exact opposite of this problem.

| | City compiler | Tabletop twin |
|---|---|---|
| Sensor geometry | one pass, from above | N sensors, from the perimeter, continuously |
| Density | ~4 pts/m2 | ~10^4-10^5 pts/m2 |
| Missing data | systematic (no facades) | incidental (occlusion, resolvable by a neighbour) |
| Time | static tile | the scene moves; state is the point |
| Output | a seed a generator expands | a twin that must stay faithful to *this* table |
| Fidelity goal | recognisably Denver-*like* | recognisably *that die, at that position* |

The city compiler is lossy on purpose. Here loss is the failure mode. Sharing
a codebase would mean two products fighting over one set of defaults.

## What is shared, and what is not

Shared, by copy or by extraction later — never by coupling now:

- the `.lwir` container and the reader/writer discipline around it
- the provenance idea: `observed` means a sensor saw it, and nothing else
- the invariant. **No stage before a backend may name a material, a shader,
  a texture or an engine.** It holds here for the same reason it holds there.
- the habit of a forward sensor simulation as the scoreboard

Not shared: the ingest adapters, the airborne feature stack, terrain and
height-above-ground, footprint extrusion, street topology, the World Seed.
None of those mean anything on a table.

## Status

**Design only. Nothing here is measured.** Every number in these documents is
either cited from published work or marked as a target. When the prototype
starts producing numbers, they replace the targets in place, and the bad ones
get reported as loudly as the good ones.

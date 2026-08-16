# Spatial IR + Synthetic LiDAR Round-Trip Benchmark v0.1

This repository is a minimum executable specification for a provenance-aware Spatial Intermediate Representation (SIR) and a synthetic LiDAR closure benchmark.

The architecture is:

```
known world -> USD -> RTX LiDAR observations -> reconstructor -> predicted SIR -> recompiled USD -> RTX LiDAR -> closure metrics
```

The benchmark intentionally separates four questions:

1. **Geometry** — did the reconstruction recover shape and placement?
2. **Semantics** — did it recover what the objects are?
3. **Topology** — did it recover part/adjacency/boundary relationships?
4. **Sensor closure** — does the reconstructed world generate observations consistent with the original observations?

A fifth axis, **provenance**, checks whether the system preserves the distinction between observed, inferred, imported, manual, and generated information instead of flattening all information into a mesh.

## Repository layout

- `SPEC.md` — normative v0.1 design notes and invariants.
- `schema/spatial_ir_v0_1.schema.json` — JSON Schema for SIR documents.
- `examples/reference_scene.sir.json` — small semantic building scene used by the smoke test and USD compiler.
- `sir/reference.py` — loader, schema validation, hashing, and basic invariants.
- `benchmark/metrics.py` — entity matching, geometry/semantic/topology/provenance metrics and point-cloud closure metrics.
- `benchmark/make_reference_scene_usd.py` — compiles the example SIR into OpenUSD primitives.
- `benchmark/capture_isaacsim_6_0_1.py` — Isaac Sim 6.0.1 RTX LiDAR capture script using the current experimental RTX sensor API.
- `benchmark/evaluate_roundtrip.py` — compares a ground-truth SIR, predicted SIR, and optional LiDAR point clouds.
- `benchmark/mock_reconstructor.py` — creates a controlled imperfect reconstruction for pipeline testing only.
- `benchmark/smoke_test.py` — runs local non-Isaac validation and metric checks.
- `SOURCES.md` — technical source notes.

## Quick local smoke test

Requires Python 3.10+ with `numpy`, `scipy`, and `jsonschema`.

```bash
python benchmark/smoke_test.py
```

Expected behavior:

- the exact-copy reconstruction scores essentially perfect geometry, semantics and topology;
- the perturbed mock reconstruction scores lower;
- the schema and provenance invariants pass.

## Isaac Sim round-trip

The capture script targets Isaac Sim **6.0.1** and the `isaacsim.sensors.experimental.rtx` API.

From the Isaac Sim installation root:

```bash
./python.sh /absolute/path/to/benchmark/make_reference_scene_usd.py \
  --sir /absolute/path/to/examples/reference_scene.sir.json \
  --out /absolute/path/to/reference_scene.usda

./python.sh /absolute/path/to/benchmark/capture_isaacsim_6_0_1.py \
  --usd /absolute/path/to/reference_scene.usda \
  --out /absolute/path/to/scan_reference.npz \
  --poses /absolute/path/to/examples/lidar_poses.json
```

Then run your reconstruction system:

```text
scan_reference.npz -> YOUR RECONSTRUCTOR -> predicted.sir.json
```

Compile `predicted.sir.json` to USD, capture it with the same pose set, then evaluate:

```bash
python benchmark/evaluate_roundtrip.py \
  --gt-sir examples/reference_scene.sir.json \
  --pred-sir predicted.sir.json \
  --gt-points scan_reference.npz \
  --pred-points scan_reconstruction.npz
```

## Benchmark contract

A reconstruction system is not allowed to read hidden ground-truth entity IDs or USD prim metadata during the inverse pass. Only the declared observation package may be consumed. Ground-truth object IDs emitted by the simulator are reserved for evaluation and debugging.

The minimum publishable experiment should report each axis separately rather than only one composite score:

- mean matched AABB IoU;
- centroid RMSE;
- class accuracy;
- relation precision/recall/F1;
- confidence Brier score;
- provenance violation rate;
- symmetric LiDAR Chamfer-L1 distance;
- LiDAR p95 nearest-neighbor distance;
- coverage within fixed tolerances.

## What v0.1 deliberately does not solve

This is an interface and falsification harness, not a full city reconstructor. v0.1 does not yet include learned segmentation, plane extraction, roof reconstruction, image fusion, GIS ingestion, CityGML import/export, or procedural style generation. Those modules are expected to compile into and out of SIR without changing the core benchmark contract.

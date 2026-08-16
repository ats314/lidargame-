# First experiment runbook

## Goal

Demonstrate a complete inverse/forward closure cycle without letting the inverse system access hidden simulator structure.

## Phase A — create hidden world

1. Validate `examples/reference_scene.sir.json`.
2. Compile it to `reference_scene.usda` with `benchmark/make_reference_scene_usd.py`.
3. Treat the SIR and USD scene graph as hidden ground truth after this point.

## Phase B — generate observations

Run the reference scene through Isaac Sim 6.0.1 RTX LiDAR using `benchmark/capture_isaacsim_6_0_1.py` and the fixed `examples/lidar_poses.json` trajectory.

The inverse model may receive:

- XYZ returns;
- public sensor poses/calibration;
- optional return intensity if a benchmark track declares it.

It may not receive:

- SIR entity IDs;
- USD prim paths;
- RTX stable object IDs;
- semantic labels from the hidden scene;
- hidden topology.

## Phase C — inverse reconstruction

Produce `predicted.sir.json`.

For the first real baseline, use a deliberately conservative pipeline:

1. ground removal;
2. vertical/horizontal plane extraction;
3. connected-component grouping;
4. wall/roof candidate inference;
5. opening candidates only when supported by a return discontinuity or secondary evidence;
6. relation inference from geometric incidence;
7. confidence from residual/error statistics;
8. provenance pointing to the observation IDs used by each inference.

Do not add procedural facade detail in the baseline. That would contaminate the reconstruction benchmark with style generation.

## Phase D — forward compilation

Compile `predicted.sir.json` to `predicted.usda` with the same SIR-to-USD compiler. Generated renderer-only proxies may be used, but they must retain `generated` provenance and may not alter the predicted semantic graph.

## Phase E — rescan

Capture `predicted.usda` with exactly the same LiDAR configuration and pose set to produce `scan_reconstruction.npz`.

## Phase F — score

Run:

```bash
python benchmark/evaluate_roundtrip.py \
  --gt-sir examples/reference_scene.sir.json \
  --pred-sir predicted.sir.json \
  --gt-points scan_reference.npz \
  --pred-points scan_reconstruction.npz \
  --out metrics.json
```

## Required first-paper plots/tables

Report at least:

- entity precision/recall;
- mean AABB IoU and centroid RMSE;
- class accuracy;
- relation F1;
- confidence Brier score;
- provenance violation rate;
- LiDAR Chamfer-L1;
- LiDAR p95 nearest-neighbor distance;
- coverage at 5 cm, 10 cm and 25 cm;
- ablation: geometry-only reconstruction vs geometry+semantics vs geometry+semantics+topology;
- ablation: with and without forward sensor-closure optimization.

## Falsification criterion

The architecture is not justified merely because it can reconstruct a scene. The central hypothesis is weakened if a mesh-only baseline achieves the same sensor closure and downstream task performance while SIR semantics/topology/provenance provide no measurable benefit.

The strongest positive result would be a case where two reconstructions have similar geometric or sensor closure but SIR identifies the semantically/topologically correct reconstruction and improves a downstream task such as simulation, editing, navigation, code analysis, or semantic LOD streaming.

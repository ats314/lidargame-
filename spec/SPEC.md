# Spatial Intermediate Representation (SIR) v0.1

Status: experimental minimum specification.

## 1. Design objective

SIR is a persistent representation of a physical or synthetic world that separates:

- identity;
- semantic class;
- spatial representation;
- relations/topology;
- uncertainty;
- provenance;
- epistemic state;
- rendering/simulation representation.

The core invariant is that **the world model is not the mesh**. Meshes, point clouds, analytic primitives, polygons, solids, and bounding volumes are replaceable representations attached to stable semantic entities.

## 2. Entity model

Every entity has:

- a stable `id` inside the document;
- a coarse `kind`;
- a semantic `class`;
- zero or more geometry representations;
- an `epistemic_state`;
- confidence values;
- provenance records;
- open-ended attributes.

### 2.1 Kinds

v0.1 defines these broad kinds:

- `region`
- `space`
- `boundary`
- `opening`
- `object`
- `terrain`
- `transport`
- `vegetation`
- `water`
- `sensor`
- `logical`

The `class` string remains extensible. Example classes include `Building`, `Room`, `WallSurface`, `RoofSurface`, `Window`, `Door`, `Road`, and `Tree`.

## 3. Epistemic state

`epistemic_state` is one of:

- `observed` — direct sensor data or directly imported authoritative geometry;
- `inferred` — reconstructed or semantically inferred from evidence;
- `generated` — procedurally or artistically created;
- `manual` — explicitly authored by a human;
- `imported` — imported from an external structured source whose own epistemic status is not reduced further;
- `hybrid` — composition of multiple states.

A reconstructed wall inferred from a LiDAR point cloud is normally `inferred`, while the point-cloud observation itself is `observed`.

This distinction is normative because it prevents generated or inferred detail from being silently presented as measured reality.

## 4. Provenance

Each provenance record should specify:

- `mode`;
- source observation/entity IDs where applicable;
- evidence IDs where applicable;
- algorithm name/version;
- parameter hash where reproducibility matters;
- timestamp where available;
- optional notes.

A geometry representation may also carry its own provenance records if different representations of one entity come from different sources.

## 5. Geometry

Geometry is a list because one entity can have multiple simultaneous representations.

v0.1 representations:

- `primitive`
- `mesh`
- `polygon`
- `solid`
- `polyline`
- `point`
- `point_cloud`
- `bbox`
- `external`

Each representation has a `role`:

- `render`
- `extent`
- `collision`
- `source`
- `proxy`
- `analysis`

This permits, for example, a building to have a coarse extent box, a detailed render mesh, and an observed source point cloud without conflating them.

## 6. Semantic resolution

SIR does not equate level of detail with polygon count. `semantic_resolution` is categorical:

- `macro` — city, district, terrain region;
- `object` — building, road, tree, vehicle;
- `part` — wall, roof, room, road segment;
- `component` — door, window, beam, curb, sign;
- `detail` — trim, hardware, joints, small surface features.

Geometry fidelity is represented separately through explicit geometry and optional `geometric_tolerance_m`.

Therefore a runtime may reduce semantic complexity by omitting `detail` and `component` entities while retaining the same high-level world identity and topology.

## 7. Relations

Relations are first-class directed edges with their own confidence and provenance.

Core relation vocabulary:

- `part_of`
- `contains`
- `bounds`
- `bounded_by`
- `adjacent_to`
- `connected_to`
- `supports`
- `intersects`
- `inside`
- `derived_from`
- `corresponds_to`

Custom relation strings are allowed, but benchmark relations should use the core vocabulary when possible.

## 8. Observations

Observations are separate from entities. An observation may contain:

- a sensor identity and modality;
- a timestamp or interval;
- sensor pose/reference frame;
- an external file URI/path;
- calibration metadata;
- preprocessing history;
- hash/checksum.

A reconstructed entity refers back to observations through provenance, not by embedding raw sensor data inside the entity.

## 9. Alternatives and uncertainty

A world reconstruction may contain mutually competing hypotheses. v0.1 includes an `alternatives` array that can group candidate entity IDs with probabilities or scores.

This avoids forcing ambiguous evidence into one prematurely certain geometry.

## 10. Spatial closure benchmark

For a known scene `S`, simulator `F`, inverse model `R`, and compiler `C`:

```
P = F(S)
S_hat = R(P)
P_hat = F(C(S_hat))
```

The benchmark evaluates both latent-world fidelity and sensor closure:

```
E_geometry(S, S_hat)
E_semantics(S, S_hat)
E_topology(S, S_hat)
E_provenance(S_hat)
D_sensor(P, P_hat)
```

A system can therefore fail in distinguishable ways. Two reconstructions may have similar point-cloud closure while one has incorrect semantics or topology.

## 11. Required anti-cheating rule

Ground-truth simulator metadata, stable prim IDs, hidden scene graph information, and source SIR IDs are evaluation-only unless a benchmark track explicitly declares them as input.

The standard inverse track receives only declared sensor observations and public calibration.

## 12. v0.1 acceptance criteria

A conforming SIR document must:

1. validate against the JSON Schema;
2. have unique entity IDs;
3. have unique relation IDs;
4. reference only existing entity IDs in relations;
5. reference only existing observation/entity IDs in provenance when those references are internal;
6. not label derived analytic geometry as `observed` unless the geometry itself is a direct observation/import;
7. use meters as the benchmark linear unit unless a track explicitly states otherwise.

## 13. Compatibility direction

SIR is intended to interoperate with rather than replace established formats:

- CityGML-like semantic spaces and boundaries can map into SIR entities and relations;
- OpenUSD can serve as a rendering/simulation target;
- 3D Tiles can serve as a streaming target;
- BIM/GIS sources can be represented as imported provenance-bearing entities.

The distinctive contract is the stable provenance-aware intermediate world model and the inverse/forward closure benchmark around it.

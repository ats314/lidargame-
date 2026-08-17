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

- `observed` — directly supported by declared evidence;
- `derived` — deterministically computed from observed evidence, such as a
  plane fit through measured returns, a plane intersection, or a surface
  clipped to a polygon boundary;
- `inferred` — structurally or semantically implied but not directly measured;
- `resolved` — the canonical World Seed interpretation selected after
  reconciling competing sources;
- `generated` — deliberately synthesised target detail, not claimed as measured
  reality;
- `unknown` — evidence is insufficient or conflicting, and no canonical
  physical claim should be made.

A reconstructed wall inferred from a LiDAR point cloud is normally `inferred`,
while the point-cloud observation itself is `observed`.

`manual`, `imported`, `fusion` and similar concepts are *origin*, not
epistemic state, and belong in provenance `mode` — where they already live.
Carrying them in both places let an entity be `imported` without ever saying
whether the import was measured or invented.

A geometry representation carries its own epistemic state. An observed point
set and an inferred fitted wall can belong to the same semantic entity, so the
state is a property of the representation and not only of the entity.

`unknown` is a valid and sometimes required output. The compiler must refuse to
fill when evidence conflicts materially, when extrapolation would exceed its
configured support, or when several hypotheses remain equally plausible.

This distinction is normative because it prevents generated or inferred detail
from being silently presented as measured reality.

> Superseded v0.2, recorded so the change is legible: the earlier set was
> `observed | inferred | generated | manual | imported | hybrid`. `hybrid` meant
> "a composition of states" and was used for a surface part measured and part
> hole-filled. That is `derived` — the fit is a deterministic consequence of the
> returns — and saying so is a sharper claim than `hybrid`, not a weaker one.

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

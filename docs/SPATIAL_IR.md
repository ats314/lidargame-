# Spatial IR (`.lwir`) — schema v0.2.0

The intermediate representation between "a scan" and "a world". It is the
canonical output of the compiler; every backend is a consumer of it.

Three properties define it:

- **Theme-independent.** Nothing in the IR names a texture, a shader or a
  colour. Materials are decided later, from roles and context flags.
- **Engine-independent.** JSON for structure, raw little-endian binary for bulk.
  No custom container to reimplement.
- **Provenance-carrying.** Every source's licence and CRS, every stage's
  parameters and timings, and every node's confidence and support survive to
  the far end.

## On disk

A `.lwir` is a zip. `unzip -l world.lwir` works, and a backend can stream a
single array out of it without parsing the rest.

```
manifest.json      schema, CRS, origin, bounds, sources, stage log, array index
graph.json         nodes + edges
points.json        per-point channel index (optional layer)
arrays/<key>.bin   raw buffers; dtype and shape declared in the manifest
```

`lidarworld inspect world.lwir --graph` prints all of it.

## Coordinates

Z-up, metres, right-handed — what survey and airborne LiDAR use.

Ingest subtracts a local `origin` (recorded in the manifest) because UTM
easting/northing destroy float32 precision. Add `origin` back to recover the
source CRS. Backends that need Y-up (glTF) apply the swap at their boundary,
never upstream.

## Nodes

```jsonc
{
  "id": "bldg.0003/face.0017",
  "role": "surface.wall.vertical",   // taxonomy id, see docs/ROLES via `lidarworld roles`
  "semantic": "building",            // canonical class
  "kind": "surface",                 // object | surface | opening | instance | terrain
  "parent": "bldg.0003",
  "children": ["bldg.0003/face.0017/opening.00"],
  "confidence": 0.82,                // measured where validation has run
  "support": 4213,                   // source points backing this up
  "sources": ["src0"],
  "stage": "segment.planes",         // which pass created it
  "geometry": { … },
  "attrs": { "area": 48.2, "slope_deg": 88.4, "rms": 0.021, "tiles": 771 },
  "tags": ["street_facing"]
}
```

Ids are hierarchical and stable within a compile: `bldg.NNNN`,
`bldg.NNNN/face.NNNN`, `bldg.NNNN/face.NNNN/opening.NN`, `tree.NNNN`,
`vehicle.NNNN`, `pole.NNNN`, `terrain`.

## Geometry kinds

| kind | payload | used by |
|---|---|---|
| `tiled_plane` | `occupancy` (uint8), `context` (uint32), `evidence` (uint16) on a plane-local lattice, plus the frame | walls, roofs, slabs |
| `heightfield` | `height` (float32), `class` (uint8) on a world-axis lattice | terrain |
| `instance` | frame only: position, size, yaw | trees, poles, vehicles, openings |
| `aggregate` | bounds only; geometry lives in the children | buildings |
| `mesh` | positions, indices, normals, uv | reserved |
| `polyline` | positions | reserved |

A `tiled_plane` frame carries `origin`, `u`, `v`, `normal`, `cell`, `shape` and
`uvOrigin`, which is everything needed to place cell *(i, j)* in world space:

```
world = origin + (uvOrigin.u + (i + 0.5) * cell) * u
               + (uvOrigin.v + (j + 0.5) * cell) * v
```

## The context bitmask

Per tile, stored in the `context` array and shipped to runtimes as a vertex
attribute. This is the part that has no equivalent in CityGML, glTF or a mesh
format, and it is what makes "corner wall" and "wall next to a window"
addressable by a rule.

| bit | flag | meaning |
|---|---|---|
| 0 | `occupied` | this cell is solid surface |
| 1–4 | `edge_u_min` … `edge_v_max` | the patch ends on this side |
| 5 | `corner_convex` | outside corner in the lattice |
| 6 | `corner_concave` | inside corner, confirmed by a perpendicular neighbour |
| 7 | `near_opening` | within ~0.6 m of a window or door |
| 8 | `opening_boundary` | directly adjacent to one |
| 9 / 10 | `top` / `bottom` | extreme band along the up-slope axis |
| 11 | `ground_contact` | this cell meets the terrain |
| 12–14 | `adj_perpendicular`, `adj_roof`, `adj_coplanar` | what the neighbouring patch is |
| 15 | `interior` | ≥3 cells from every boundary — the "centre of mass" tiles |
| 16 | `street_facing` | the outward normal looks onto a carriageway |
| 17 | `sheltered` | under an overhang |
| 18 | `sparse_evidence` | filled by closing or hull, not measured |
| 19 | `occluded` | inferred; the sensor never saw it |

`lidarworld roles` prints the live list. Bits 18 and 19 matter for honesty: a
renderer can choose to show only what was actually observed.

## Edges

```jsonc
{ "a": "bldg.0003/face.0017", "b": "bldg.0003/face.0018",
  "rel": "perpendicular_to", "confidence": 0.71, "attrs": { "cos": 0.02 } }
```

Relations: `contains`, `adjacent_to`, `coplanar_with`, `perpendicular_to`,
`opening_in`, `supports`, `borders`, `above`, `connects`, `parallel_to`,
`occludes`.

## Point layer

Optional and decimated. Carries the working channels — `semantic`, `role`,
`hag`, `planarity`, `verticality`, `crease_score`, `corner_score`, `boundary`,
`normal`, `patch`, `role_confidence` — so a world can be re-segmented or
re-inspected without going back to the source file.

## Reading it

```python
from lidarworld.ir import read_world, inspect

inspect("world.lwir")                 # manifest only, no arrays touched
world = read_world("world.lwir")      # arrays stay lazy until indexed

for wall in world.by_role("surface.wall"):
    ctx = world.arrays[wall.geometry.arrays["context"]]
    print(wall.id, wall.confidence, ctx.shape)
```

## Compatibility

`schema` is semver. Within a minor version, consumers may see new node kinds,
new roles and new context bits — so match roles by prefix and ignore unknown
bits rather than switching exhaustively.

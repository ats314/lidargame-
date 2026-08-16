# Writing a backend

The compiler is the product; the engine is a target. Nothing upstream of
`src/lidarworld/backends/` knows that a renderer exists, so adding one is a
single function.

## Contract

```python
def export(world: World, out_path, **options) -> dict:
    """Consume the Spatial IR, write something, return a small summary."""
```

That is the whole interface. Register it in `backends/__init__.py`:

```python
BACKENDS["usd"] = {
    "module": usd,
    "needs_theme": True,           # does it have to materialise?
    "description": "OpenUSD stage for Omniverse / Isaac Sim",
}
```

## The decision every backend makes

**Does the target understand "theme" as a runtime concept?**

- **No** (glTF, CityJSON, USD, most engines) — resolve the pack once at the
  boundary and bake materials in. `gltf.resolve_vertex_materials()` does this:
  it resolves each distinct `(role, context)` pair once, not each vertex, then
  splits the mesh into primitives by resolved material.
- **Yes** (the bundled web viewer) — ship the context bitmask as a vertex
  attribute and the rule table as data, and let the runtime resolve. That is
  what makes theme switching cost one JSON fetch instead of a recompile.

Either way the IR stays theme-independent. Materialisation happens *at the
boundary*, once.

## What is available

```python
world.arrays["mesh/positions"]   # (N,3) float32, world space, origin-shifted
world.arrays["mesh/normals"]     # (N,3) float32
world.arrays["mesh/uv"]          # (N,2) float32, in metres
world.arrays["mesh/ctx"]         # (N,)  uint32  context bitmask
world.arrays["mesh/role"]        # (N,)  uint32  index into ROLE_IDS
world.arrays["mesh/node"]        # (N,)  uint32  slot -> mesh_slot_names(world)
world.arrays["mesh/indices"]     # (M,3) uint32
world.arrays["terrain/dtm"]      # heightfield
world.nodes / world.edges        # the graph
world.sources / world.stages     # provenance
world.origin                     # add back to recover the source CRS
```

`mesh_slot_names(world)` (in `backends/web.py`) is the one definition of how the
`node` attribute maps back to graph node ids. Use it rather than reimplementing
the traversal — CityJSON and forward validation both do.

## Conventions worth keeping

- **Up axis.** The IR is Z-up. If the target is Y-up, apply the swap in the
  backend (glTF does it with a root node matrix), never upstream.
- **Georeferencing.** Add `world.origin` back before writing coordinates that
  claim to be in a CRS. CityJSON does this and records the CRS in metadata.
- **Provenance.** Carry `license`, `source` and `confidence` through. glTF puts
  them in `material.extras` and `asset.copyright`; CityJSON uses
  `+lidarworld_*` attributes.
- **Say what you drop.** CityGML has no place for a context bitmask, so the
  CityJSON backend summarises it into an extension attribute rather than
  silently discarding it. Do the same for anything a target cannot express.

## Targets worth adding

| target | why | notes |
|---|---|---|
| **OpenUSD** | Omniverse, Isaac Sim, and the RTX LiDAR sensor for closed-loop validation against a known scene | `UsdGeom.Mesh` per node, `UsdShade` from the theme; hierarchy maps directly onto node parents |
| **Cesium 3D Tiles** | planetary-scale streaming with semantic metadata per feature | node graph → tileset tree; `EXT_structural_metadata` carries roles and confidence |
| **Godot / Unity / Unreal** | play it | glTF already imports; a native importer buys runtime theme swapping |
| **Bevy / wgpu** | a GPU-driven runtime over the IR directly | the web bundle format is already close to what a GPU-driven renderer wants |

## Testing one

`tests/test_pipeline.py` shows the shape: export the shared `compiled_world`
fixture and check *internal consistency* rather than golden files — every
triangle lands in exactly one glTF primitive, every CityJSON ring indexes a
real vertex, the declared vertex stride matches the bytes written.

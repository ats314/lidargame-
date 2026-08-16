# lidarworld

**Compile LiDAR point clouds into semantic worlds you can walk through — and re-skin into any era without recompiling.**

A public LiDAR scan is a few million dots. This turns those dots into a structured, engine-agnostic **Spatial IR**: a hierarchical world graph of buildings, walls, roofs, openings, roads, terrain and trees, each carrying geometry, topology, confidence and provenance — and deliberately carrying *no materials at all*.

Materials come afterwards, from a **theme pack** that binds to what a surface *means* rather than what it is called:

```
surface.wall.vertical + corner_convex           -> quoin stones
surface.wall.vertical + opening_boundary        -> dressed stone surround
surface.wall.vertical + ground_contact          -> plinth
surface.wall.vertical + street_facing           -> shopfront / signage
```

Swap the pack and the same scan becomes a Victorian street, a neon sprawl, or a flat diagnostic render. No stage of the compiler re-runs, no geometry is rebuilt, no vertex moves.

---

## The idea

Every tile of every reconstructed surface carries a **context bitmask** — its position relative to everything around it:

| | |
|---|---|
| `corner_convex` / `corner_concave` | outside and inside building corners |
| `opening_boundary` / `near_opening` | the reveal around a window or door |
| `top` / `bottom` / `ground_contact` | cornice band, plinth, where wall meets pavement |
| `interior` | deep in the middle of a surface, far from any edge |
| `street_facing` | this facade looks onto a carriageway |
| `sheltered` | under an overhang, so it weathers differently |
| `sparse_evidence` / `occluded` | inferred geometry the sensor never actually saw |

That mask is what makes "corner wall" and "wall close to a window" addressable. It is computed once, stored in the IR, and shipped to the runtime as a vertex attribute — which is why re-theming costs one small JSON fetch instead of a recompile.

## Pipeline

```
 .las .laz .bin .pcd .ply .xyz
            │
   ingest   │  normalise: metres, canonical semantics, licence + CRS preserved
   terrain  │  bare-earth model -> height above ground
   features │  multiscale PCA: planar? linear? crease? corner? boundary?
   semantics│  source labels where they exist, inference where they do not
   roles    │  what each point *does*, not just what it is
   segment  │  planar patches, tree instances, vehicles, poles
   lattice  │  tile grid per surface + context bitmask + openings
   topology │  relations -> buildings, corners, street frontage
 reconstruct│  terrain mesh + merged surface quads (theme-independent)
            ▼
      Spatial IR  (.lwir)
            │
    ┌───────┼────────┬──────────────┐
    ▼       ▼        ▼              ▼
   web    glTF   CityJSON     forward validation
 viewer  engines   GIS       re-scan and score it
```

Openings are found the way a sensor finds them: glass returns almost nothing at 905 nm, so a window is a **hole in the returns enclosed by solid returns**. It is kept as a hole, not patched over.

## Forward validation

The last stage runs the pipeline backwards. It puts a virtual sensor back where the real one stood, casts the same beams at the *reconstructed* geometry, and compares simulated ranges against measured ones:

```
observed scan -> compiler -> world -> simulated scan -> compare
```

That turns confidence from a heuristic into a measurement, per node:

```
$ lidarworld validate build/street/street.lwir --scan data/samples/street.bin
  4,660/25,248 returns explained (18.5%), range RMSE 16.9 cm,
  2,711 unexplained, 6,272 over-occluded, 10,307 grazing (inconclusive)

least consistent surfaces:
  bldg.0013/face.0037    4.5% of 3409 rays  bias -11.43 m
```

A negative bias means the compiler put a surface *in front of* where the beam actually stopped — geometry it invented. That is exactly the failure a mesh-only pipeline cannot tell you about.

## Quick start

```bash
pip install -e ".[dev]"

python tools/make_sample_data.py --out data/samples      # deterministic sample scans
lidarworld compile data/samples/townblock.las \
    -o viewer/world -n townblock \
    --theme survey --theme victorian --theme neon

python -m http.server 8000 --directory viewer            # then open localhost:8000
```

Walk with `WASD`, look with the mouse, press `Tab` to cycle themes, `E` to inspect the surface under the crosshair. The inspector shows the node, its confidence and support, its context flags, and **which theme rule chose its material and why**.

### Other commands

```bash
lidarworld adapters                       # what it can read
lidarworld roles                          # role taxonomy + context flags
lidarworld themes -v                      # theme packs and their rules
lidarworld explain --theme victorian \
    --role surface.wall.vertical --context corner_convex,street_facing
lidarworld inspect build/town/townblock.lwir --graph
lidarworld compile tile.las -o out --gltf --cityjson
```

## Using real data

The sample tiles are synthetic so the pipeline runs with no downloads, but nothing in it is tuned to them:

- **Airborne** — [USGS 3DEP](https://www.usgs.gov/3d-elevation-program) / [OpenTopography](https://opentopography.org/) LAZ tiles arrive pre-classified with ASPRS codes and in projected metres. Best data-to-world ratio available.
- **Street level** — [SemanticKITTI](https://semantic-kitti.org/), [nuScenes](https://www.nuscenes.org/), [Paris-Lille-3D](https://npm3d.fr/paris-lille-3d). Dense facades, real per-point labels, known sensor pose for validation.
- **Anything else** — PCD, PLY, XYZ from a robot or CloudCompare. No labels needed; the inference stage earns them from geometry and records its own confidence.

See [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) for licences and download recipes.

## Repository layout

```
src/lidarworld/
  ingest/       adapters: LAS/LAZ, KITTI + SemanticKITTI, PCD, PLY, XYZ
  spatial/      sparse voxel indexing and moment aggregation
  features/     multiscale descriptors, terrain, height above ground
  semantics/    ASPRS + SemanticKITTI mappings, rule-based inference
  roles/        role taxonomy, context bitmask, CityGML alignment
  segment/      planar region growing, tree/vehicle/pole instancing
  reconstruct/  tile lattices, openings, terrain and surface meshing
  topology/     relations, building grouping, street frontage
  themes/       theme packs, material resolver, procedural texture backend
  backends/     web, glTF 2.0, CityJSON 1.1
  ir/           Spatial IR reader/writer
  validate.py   forward LiDAR simulation and consistency scoring
viewer/         dependency-free WebGL2 first-person viewer
tools/          sample-data baker, resource-index generator
docs/           IR schema, architecture, themes, prior art, data sources
```

## Design notes

**The compiler is the product; the engine is a target.** Nothing upstream of `backends/` knows a renderer exists. Adding USD, Bevy, Cesium 3D Tiles or a native runtime is one function — see [docs/BACKENDS.md](docs/BACKENDS.md).

**Materials are requested by meaning.** A tile asks for "a wall, on a convex corner, in a Victorian world" and a resolver satisfies it. Procedural generation is the zero-dependency default (no asset licences, a new era is a parameter block), and image-based, CC0, photogrammetry or engine-native packs override by material id.

**Provenance survives to the far end.** Every source's licence and CRS, every stage's parameters and timings, and every node's confidence and support travel in the IR and out through the exporters.

**Dependencies:** numpy only. `scipy` accelerates morphology, `laspy` reads LAZ; both optional. The viewer has no dependencies at all.

Related work this builds on and where it sits — MIT SPARK's Hydra, TU Delft's City3D, OGC CityGML, Esri CityEngine, NVIDIA RTX LiDAR, Cesium 3D Tiles — is mapped in [docs/PRIOR_ART.md](docs/PRIOR_ART.md).

The sensor, dataset, algorithm and tool index the viewer ships with is generated from [szenergy/awesome-lidar](https://github.com/szenergy/awesome-lidar); see [docs/RESOURCES.md](docs/RESOURCES.md).

## Licence

MIT. Sample data is generated, not derived from any dataset. Procedural textures are CC0.

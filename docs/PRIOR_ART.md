# Prior art, and where this sits

Every stage of this pipeline has a mature precedent somewhere. What is
conspicuously unclaimed is the connective tissue: an engine-agnostic,
provenance-aware semantic IR that stays stable while geometry, materials,
themes, engines and sensor models are swapped around it — and that can be
checked by simulating the sensor back against its own reconstruction.

## The map

| Layer | Closest existing work | Relationship |
|---|---|---|
| Semantic/topological world graph | **MIT SPARK — Hydra / Spark-DSG** ([repo](https://github.com/MIT-SPARK/Hydra)) | Closest thing to the IR itself: builds 3D scene graphs of objects, places, rooms and buildings incrementally from observations. Robotics-scale and online; no materialisation layer. `World`/`Node`/`Edge` here is the same shape of idea at tile scale, offline, with provenance. |
| Airborne → structured buildings | **TU Delft — City3D / 3D BAG / 3dfier** ([City3D](https://github.com/tudelft3d/City3D)) | The mature front end for exactly the input this targets. City3D produces LoD2 building models; this produces surfaces plus context and topology. A City3D-style reconstruction backend would slot in behind the `reconstruct` stage. |
| Schema / ontology | **OGC CityGML 3.0 & CityJSON** ([standard](https://www.ogc.org/standards/citygml/)) | The strongest schema precedent: platform-independent semantic model for 3D urban objects with hierarchy and LoD. Roles map onto its boundary surfaces (`citygml_type()`), and `backends/cityjson.py` exports conformant CityJSON. What CityGML has no place for is the per-tile context mask. |
| Theme / geometry compiler | **Esri CityEngine + CGA** ([product](https://www.esri.com/en-us/arcgis/products/arcgis-cityengine/overview)) | The clearest analogue to the theme layer: a grammar turning structured spatial description into architectural geometry and style. CGA *generates* geometry from rules; theme packs here only *materialise* geometry that was measured, which is why a re-skin is free. |
| Streaming / runtime | **Cesium 3D Tiles** ([overview](https://cesium.com/why-cesium/3d-tiles/3d-tiles-essentials/)) | The delivery precedent for planetary scale with per-feature semantic metadata. A 3D Tiles backend is the natural next target (see BACKENDS.md). |
| Forward sensor simulation | **NVIDIA RTX LiDAR / Omniverse** ([docs](https://docs.isaacsim.omniverse.nvidia.com/)), **LiDARsim** ([paper](https://arxiv.org/abs/2006.09348)) | RTX LiDAR returns object ids, material ids, normals and echo data from a known USD scene — the strongest forward operator available. `validate.py` is the same arrow, at much lower fidelity, closing the loop on the compiler's own output. |
| Learned reconstruction | **ETH Zurich — Point2Building** ([paper](https://arxiv.org/abs/2403.02136)) | A learned alternative to explicit geometric reconstruction. The `reconstruct` stage is deliberately interchangeable so a learned backend can replace region growing without touching the IR. |
| Scene-graph precedent | **Stanford 3D Scene Graph** ([project](https://3dscenegraph.stanford.edu/)) | Earlier unified semantic-spatial graph; establishes the objects-plus-relations framing. |
| Geometric inference | **INRIA TITANE / Lafarge** | Planar-constraint meshing and urban reconstruction; relevant to improving patch extraction and boundary fitting. |
| Processing primitives | **PDAL, Open3D, CloudCompare, Potree** | The plumbing layer. This project reimplements a thin slice in numpy to stay dependency-light; any of these can replace a stage. |
| Open-vocabulary semantics | **CitySeg** ([paper](https://arxiv.org/html/2508.09470v1)) | A learned semantic front end that could supply the `semantic` channel directly, replacing the rule cascade. |

## The gap this occupies

Reconstruction, semantic segmentation, scene graphs, procedural city grammars,
synthetic LiDAR and streamed semantic 3D data all exist at high maturity — but
not combined into one system that reconstructs a hierarchical semantic world
from LiDAR, materialises alternative themed worlds from it, and uses a LiDAR
forward simulator as an explicit reconstruction-consistency test.

Concretely, three choices here that the precedents do not make together:

1. **The IR is theme-independent, and that is enforced.** No stage before
   `backends/` may name a material. This is what makes runtime re-skinning a
   lookup instead of a rebuild.
2. **Context is a first-class, stored quantity.** Not "this is a wall" but
   "this is a wall tile on a convex corner, one cell from a window, that the
   sensor never actually observed". Rules bind to that; renderers can hide it.
3. **The loop closes.** `observed → compiler → world → simulated → compare`
   turns confidence into a measurement, per node, and makes invented geometry
   visible rather than plausible.

## Honest limitations

- Region growing is a 2003-era technique. Learned reconstruction (Point2Building,
  City3D's optimisation) produces cleaner building models. The IR is designed so
  those can be swapped in.
- The rule-based semantic cascade is far weaker than RandLA-Net or KPConv on
  unlabelled data. It exists so the pipeline runs with no weights and reports
  its own low confidence honestly, not because it is competitive.
- Forward validation uses a voxelised scene and a lockstep march, so
  near-tangent hits are inconclusive and reported separately. A proper
  ray-triangle test with a BVH would tighten this considerably.
- Nothing here registers clouds. Multi-source input must already be in a common
  frame.

## Reading order

If you want to go deeper, roughly: CityGML 3.0 for the schema, Hydra for the
graph, City3D for the reconstruction front end, CityEngine/CGA for the
materialisation grammar, RTX LiDAR for the forward operator, 3D Tiles for
delivery.

# Technical source notes

Checked 2026-08-16.

## OGC CityGML 3.0

- Standard overview: https://www.ogc.org/standards/citygml/
- Conceptual Model Standard: https://docs.ogc.org/is/20-010/20-010.html

Relevant concepts reused at a high level: semantic entities, geometry independent of semantics, spaces and space boundaries, relations, multiple levels of detail, and platform-independent modeling.

## OpenUSD

- `UsdRelationship`: https://openusd.org/release/api/class_usd_relationship.html
- `UsdGeomMesh`: https://openusd.org/release/api/class_usd_geom_mesh.html

SIR uses USD as a target representation, not as the canonical world model. USD relationships and custom metadata are used by the reference compiler to preserve SIR IDs/classes for simulation/debugging.

## NVIDIA Isaac Sim 6.0.1

- RTX LiDAR: https://docs.isaacsim.omniverse.nvidia.com/6.0.1/sensors/isaacsim_sensors_rtx_lidar.html
- RTX sensor annotators: https://docs.isaacsim.omniverse.nvidia.com/6.0.0/sensors/isaacsim_sensors_rtx_annotators.html
- Standalone Python: https://docs.isaacsim.omniverse.nvidia.com/latest/python_scripting/manual_standalone_python.html

The current 6.0.x API uses `isaacsim.sensors.experimental.rtx`. `Lidar` authors/wraps the OmniLidar prim and `LidarSensor` handles runtime data collection. The GenericModelOutput buffer exposes point returns and optional auxiliary data. Stable 128-bit object IDs can be enabled for evaluation/debugging, but they are prohibited as reconstructor inputs in the standard inverse track.

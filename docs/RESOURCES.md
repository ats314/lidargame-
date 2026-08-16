# The awesome-lidar index

The viewer ships a browsable index of every entry in
[szenergy/awesome-lidar](https://github.com/szenergy/awesome-lidar) — 124
manufacturers, datasets, libraries, frameworks, algorithms, simulators and
tools, with links straight to the primary resource.

It is generated, not hand-maintained:

```bash
node tools/build_codex.mjs          # fetches the upstream README
node tools/build_codex.mjs path/to/README.md   # or from a local copy
```

Output lands in `viewer/src/data/codex.js` as `CODEX` / `CODEX_GROUPS`. Every
top-level bullet becomes one entry `{k: kind, g: group, n: name, u: url, d: description}`;
sub-bullets (YouTube, ROS drivers, papers) are deliberately dropped so the index
points at the primary resource and defers the rest upstream. The generator
fails loudly if it parses fewer than 50 entries, so an upstream format change
cannot silently empty it.

## What this project actually uses from that list

| awesome-lidar entry | Where it shows up here |
|---|---|
| **Datasets** — KITTI, SemanticKITTI | `ingest/kitti.py`: the `.bin` layout and the SemanticKITTI label vocabulary, including the moving-object duplicate ids (252–259) |
| **Datasets** — the airborne/national tiles this list points at | `ingest/las.py`: ASPRS class mapping, LAS 1.0–1.4 reader |
| **Libraries** — PCL, Open3D, LAStools | Format compatibility: PCD (ascii, binary, `binary_compressed` with an LZF decoder), PLY, LAS. Voxel-grid downsampling and Euclidean clustering are reimplemented in numpy rather than depended on |
| **Algorithms / Ground segmentation** — Patchwork, LineFit, plane fitting | `features/ground.py`: the progressive morphological filter, and the labelled-ground fast path |
| **Algorithms / Semantic segmentation** — RandLA-Net, RangeNet++, KPConv, SPT | `semantics/infer.py` is the honest fallback for unlabelled clouds. Any of these can supply the `semantic` channel directly and the stage skips itself |
| **Algorithms / Basic matching** — ICP, KISS-ICP | Not implemented. Multi-source input must already be registered; see DATA_SOURCES.md |
| **Algorithms / Object detection** — clustering and tracking work | `segment/instances.py`: connected-component instancing for vehicles and poles, canopy-maxima instancing for trees |
| **Simulators** — CARLA, AWSIM, Gazebo, and NVIDIA's RTX LiDAR | `validate.py` runs the same forward direction: virtual sensor, cast beams, compare returns. See PRIOR_ART.md |
| **Others** — CloudCompare, MeshLab, Foxglove, Rerun, Potree | Interchange targets. glTF and CityJSON open in most of them; the point layer in the IR keeps the original dots inspectable |
| **Manufacturers** — Velodyne, Ouster, Livox, Hokuyo, SICK, Riegl, Aeva… | Context for the sensor model. Wavelength matters directly: opening detection depends on glass returning almost nothing at 905 nm |

## Attribution

The index content — names, links, descriptions — is from awesome-lidar, which
is maintained by [szenergy](https://github.com/szenergy) and contributors. Only
the generator and the presentation are part of this repository. Re-run
`tools/build_codex.mjs` to pick up upstream additions.

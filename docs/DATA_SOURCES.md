# Where to get real LiDAR

The bundled samples are synthetic so the pipeline runs with no downloads, but
nothing in it is tuned to them. Check each dataset's own terms before
redistributing anything derived from it — the ingest adapters record whatever
licence string you pass and carry it through to every export.

## Airborne — the best data-to-world ratio

Tiles arrive already classified with ASPRS codes and in projected metres, so
the semantics stage has real labels to work from and a tile drops straight into
a world.

**USGS 3DEP** (public domain, most of the United States)

```bash
# Browse and download: https://apps.nationalmap.gov/downloader/
lidarworld compile USGS_LPC_tile.laz -o build/here \
  --theme survey --theme victorian
```

**OpenTopography** (global aggregator, per-dataset licences) —
<https://opentopography.org/>. Good for finding non-US coverage.

**National providers** — AHN (Netherlands, CC BY), England Environment Agency
(OGL), Denmark, Switzerland, Spain and others publish nationwide LiDAR. Most are
open with attribution.

Reading `.laz` needs a decompression backend:

```bash
pip install "laspy[lazrs]"
```

Uncompressed `.las` works with no dependencies at all — the built-in reader
handles LAS 1.0–1.4, point formats 0–3 and 6–8.

### What to expect

Airborne gives you excellent roofs, terrain and vegetation, and thin facades —
the sensor sees walls at a glancing angle from above, so wall patches are
sparse and the opening detector will find few windows. That is honest: the data
does not contain them. Compile with a larger `--tile` (0.4–0.5 m) so facade
lattices are not mostly `sparse_evidence`.

## Street level — facades and detail

**SemanticKITTI** (CC BY-NC-SA 4.0) — <https://semantic-kitti.org/>. Per-point
labels and a known sensor pose, which is what forward validation needs.

```bash
lidarworld compile sequences/00/velodyne/000000.bin -o build/street --theme neon
lidarworld validate build/street/street.lwir --scan sequences/00/velodyne/000000.bin
```

Labels are found automatically at `../labels/<stem>.label`, or pass
`--adapter kitti` with an explicit path.

**nuScenes** (CC BY-NC-SA 4.0), **Paris-Lille-3D** (CC BY-NC-SA 4.0),
**Toronto-3D**, **DALES** — all street or aerial MLS with labels.

**Note on single sweeps.** One sweep is one viewpoint: everything behind a
parked car is a scan shadow, and the compiler will either leave a hole or fill
it and flag the result `sparse_evidence` / `occluded`. Forward validation makes
this visible rather than letting it pass as geometry. Aggregate several
consecutive sweeps into one file for a fuller world.

## Anything else

PCD, PLY, XYZ/CSV from a robot, a terrestrial scanner or CloudCompare. No labels
required — the semantics stage infers them from height above ground and the
multiscale descriptors, and records a per-point confidence saying how much to
trust it.

```bash
lidarworld compile scan.pcd -o build/room --theme survey
```

## Multiple sources at once

Files are merged behind one adapter interface, channels unioned, per-point
source recorded:

```bash
lidarworld compile airborne_tile.laz street_scan.bin -o build/combined
```

Both must be in the same coordinate frame. Airborne tiles are in a projected
CRS; sensor-frame scans are not, so georeference the scan first (a pose from
the dataset, or an ICP alignment) — the compiler does not register clouds for
you, and it will not pretend to.

## Sizing

| tile | points | compile time | notes |
|---|---|---|---|
| bundled sample | 128k | ~5 s | |
| 500 m × 500 m airborne @ 8 pts/m² | ~2M | ~1–2 min | comfortable |
| 1 km² @ 20 pts/m² | ~20M | slow, memory-hungry | use `--decimate 2` or tile it |

Time is dominated by the descriptor and segmentation stages. `scipy` roughly
halves the morphology and connected-component cost:

```bash
pip install "lidarworld[fast]"
```

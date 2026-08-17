"""Download the Denver AOI bundle: every GIS layer, clipped, with provenance.

    python tools/acquire_denver.py --out data/gis
    python tools/acquire_denver.py --place denver_capitol --mode generation

This was a standalone script carrying its own copy of the layer table, its own
pagination and its own provenance format. That copy has been deleted: it had
already drifted from `lidarworld.data.denver` -- different services for the same
layer, different roles for the same data, two spellings of the licence -- and a
second layer table is a second thing to keep true about the world.

What it does that the module could not is now in the module: the layers it had
and the manifest did not, and the field-level findings that were the real value
in it. Those are recorded on the layers themselves, in `denver.LAYERS`, where
anything consuming a layer will see them.

The one behavioural change worth knowing about: output is laid out by whether a
layer is admissible rather than dropped in one directory. The old bundle wrote
everything side by side and added a note asking the reader not to feed hidden
truth to the compiler. Now `input/`, `prior/` and `truth/` are separate
directories and `load()` reads the first two, so the request is enforced by the
path instead of being asked for in a manifest nobody re-reads.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lidarworld.data.acquire import acquire            # noqa: E402
from lidarworld.data.catalog import PLACES             # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="data/gis")
    ap.add_argument("--place", default="denver_lodo",
                    help=f"an area of interest from the catalogue: "
                         f"{', '.join(sorted(PLACES))}")
    ap.add_argument("--aoi", default=None,
                    help="west,south,east,north in WGS84; overrides --place")
    ap.add_argument("--epoch", default="2020", help="which LiDAR epoch is being reconstructed")
    ap.add_argument("--mode", default="reconstruction",
                    choices=("reconstruction", "generation"))
    ap.add_argument("--crs", default="26913", help="EPSG code to reproject into")
    ap.add_argument("--overwrite", action="store_true",
                    help="refetch layers already on disk")
    args = ap.parse_args()

    if args.aoi:
        aoi = tuple(float(v) for v in args.aoi.split(","))
    elif args.place in PLACES:
        aoi = tuple(PLACES[args.place]["bbox_wgs84"])
    else:
        ap.error(f"unknown place {args.place!r}; have {sorted(PLACES)}")

    seen: set[str] = set()

    def progress(layer, record):
        if layer.id not in seen:
            seen.add(layer.id)
            print(f"  {layer.id:26s} [{layer.role}/L{layer.independence}] ...",
                  end="", flush=True)
        elif record:
            where = record["file"].split("/")[0]
            print(f" {record['features']:>6,} features  "
                  f"{record['bytes']/1024:>8.0f} KB  {record['fetch_seconds']:4.1f}s"
                  f"  -> {where}"
                  f"{'  TRUNCATED' if record['truncated'] else ''}", flush=True)
        else:
            print("  FAILED", flush=True)

    summary = acquire(aoi, args.out, epoch=args.epoch, mode=args.mode,
                      out_crs=args.crs, overwrite=args.overwrite,
                      progress=progress)

    print(f"\n{len(summary['acquired'])} layers, {summary['features_total']:,} "
          f"features, {summary['bytes_total']/1024/1024:.1f} MB -> {args.out}")
    withheld = [r for r in summary["acquired"] if r["withheld"]]
    print(f"{len(withheld)} withheld into truth/ and unreadable by load(): "
          f"{', '.join(r['id'] for r in withheld)}")
    for failure in summary["failed"]:
        print(f"  FAILED {failure['id']}: {failure['error'][:120]}")
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

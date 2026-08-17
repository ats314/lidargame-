"""Acquire the Hamburg master-city stack, in the order that pays off soonest.

    python tools/fetch_hamburg.py --out data/hamburg --budget-gb 15

Buildings alone are not a world. The first Hamburg render put a correct inner-
city block on a green field with no streets, which was not a rendering fault --
the building model was the only thing acquired. So this pulls the ground too:
terrain, the road network, the cadastral land use, and it points at the
orthophoto service rather than downloading 15 GB of it.

Ordering is deliberate. Area1 is the inner city and the smallest textured
package, so it is first; the ground layers come next because a correct building
standing in a field is still wrong; the outer areas come last.

Disk is a fixed allowance here and the full stack is about 15 GB, so `--budget-gb`
stops before the allowance runs out rather than after. A refused item is
reported, not skipped silently.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_parallel import fetch, total_size          # noqa: E402
from lidarworld.data import hamburg                    # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/hamburg")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--budget-gb", type=float, default=15.0,
                    help="stop before exceeding this much new data")
    ap.add_argument("--reserve-gb", type=float, default=2.0,
                    help="free space to leave on the volume")
    ap.add_argument("--only", help="comma-separated keys to fetch")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plan = hamburg.master_city_plan()
    if args.only:
        wanted = {k.strip() for k in args.only.split(",")}
        plan = [item for item in plan if item["key"] in wanted]

    spent = 0
    results = []
    for item in plan:
        url = item.get("url")
        if not url:
            # Never a quiet skip. A plan entry with no URL is a manifest bug,
            # and the last time this passed silently it dropped every building
            # package from the run while reporting success.
            print(f"[skip] {item['key']}: no download url in the manifest",
                  flush=True)
            results.append({"key": item["key"], "status": "no_url"})
            continue
        name = url.rsplit("/", 1)[-1]
        target = out / name
        try:
            size = total_size(url)
        except Exception as exc:                        # noqa: BLE001
            print(f"[skip] {item['key']}: {exc}", flush=True)
            results.append({"key": item["key"], "status": "unreachable",
                            "error": str(exc)})
            continue

        free = shutil.disk_usage(out).free
        if free - size < args.reserve_gb * 1e9:
            print(f"[stop] {item['key']} needs {size/1e9:.1f} GB, "
                  f"{free/1e9:.1f} GB free, reserve {args.reserve_gb} GB",
                  flush=True)
            results.append({"key": item["key"], "status": "no_space",
                            "needs_gb": round(size / 1e9, 2)})
            continue
        if (spent + size) / 1e9 > args.budget_gb:
            print(f"[stop] {item['key']} would exceed the "
                  f"{args.budget_gb} GB budget", flush=True)
            results.append({"key": item["key"], "status": "over_budget",
                            "needs_gb": round(size / 1e9, 2)})
            continue

        print(f"\n== {item['key']}: {item.get('name', name)} "
              f"({size/1e9:.2f} GB)", flush=True)
        started = time.time()
        try:
            fetch(url, str(target), workers=args.workers, quiet=False)
        except Exception as exc:                        # noqa: BLE001
            print(f"[fail] {item['key']}: {exc}", flush=True)
            results.append({"key": item["key"], "status": "failed",
                            "error": str(exc)})
            continue
        spent += size
        results.append({"key": item["key"], "status": "ok",
                        "path": str(target), "gb": round(size / 1e9, 2),
                        "seconds": round(time.time() - started)})
        print(f"   done in {time.time()-started:.0f}s", flush=True)

    print("\n---- summary ----")
    for row in results:
        print(f"  {row['status']:12s} {row['key']}"
              + (f"  {row.get('gb')} GB" if row.get("gb") else "")
              + (f"  ({row.get('error','')[:60]})" if row.get("error") else ""))
    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\n{ok}/{len(results)} acquired, {spent/1e9:.1f} GB")
    print("orthophoto: not bulk-downloaded -- "
          f"{hamburg.CONTEXT['orthophoto']['wms']} per AOI")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

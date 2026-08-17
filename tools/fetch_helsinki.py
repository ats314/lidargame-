"""Acquire the Helsinki 3D+ stack, best-demo-first, with CRC verification.

    python tools/fetch_helsinki.py --out data/helsinki --budget-gb 20

The whole reality mesh is 190 GB across 122 tiles, which is not what a demo
needs and not what a fixed disk allowance can hold. This takes the nine tiles
covering central Helsinki -- 6 x 6 km, 13.3 GB -- historic core first.

Every zip is CRC-verified after download. That is not belt and braces: a ranged
parallel download can produce a file of exactly the right length that is wrong
in the middle, and the size, the central directory and the member listing all
look fine. It happened here on a 4.2 GB archive and cost a whole comparison.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_parallel import fetch, total_size, verify_zip      # noqa: E402
from lidarworld.data import helsinki                          # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/helsinki")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--budget-gb", type=float, default=20.0)
    ap.add_argument("--reserve-gb", type=float, default=3.0)
    ap.add_argument("--only", help="comma-separated keys")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plan = helsinki.acquisition_plan()
    if args.only:
        wanted = {k.strip() for k in args.only.split(",")}
        plan = [item for item in plan if item["key"] in wanted]

    spent, results = 0, []
    for item in plan:
        url = item["url"]
        target = out / url.rsplit("/", 1)[-1]

        if target.exists():
            bad = verify_zip(str(target))
            if not bad:
                print(f"[have] {item['key']} ({target.stat().st_size/1e9:.2f} GB, CRC ok)",
                      flush=True)
                results.append({"key": item["key"], "status": "already_present",
                                "gb": round(target.stat().st_size / 1e9, 2)})
                continue
            print(f"[redo] {item['key']}: {len(bad)} members fail CRC", flush=True)
            target.unlink()

        try:
            size = total_size(url)
        except Exception as exc:                       # noqa: BLE001
            print(f"[skip] {item['key']}: {exc}", flush=True)
            results.append({"key": item["key"], "status": "unreachable",
                            "error": str(exc)[:160]})
            continue

        free = shutil.disk_usage(out).free
        if free - size < args.reserve_gb * 1e9:
            print(f"[stop] {item['key']} needs {size/1e9:.2f} GB, "
                  f"{free/1e9:.1f} GB free", flush=True)
            results.append({"key": item["key"], "status": "no_space",
                            "needs_gb": round(size / 1e9, 2)})
            continue
        if (spent + size) / 1e9 > args.budget_gb:
            print(f"[stop] {item['key']} exceeds the {args.budget_gb} GB budget",
                  flush=True)
            results.append({"key": item["key"], "status": "over_budget",
                            "needs_gb": round(size / 1e9, 2)})
            continue

        print(f"\n== {item['key']}: {item['name']} ({size/1e9:.2f} GB)", flush=True)
        started = time.time()
        try:
            fetch(url, str(target), workers=args.workers, quiet=False)
        except Exception as exc:                       # noqa: BLE001
            print(f"[fail] {item['key']}: {exc}", flush=True)
            results.append({"key": item["key"], "status": "failed",
                            "error": str(exc)[:200]})
            continue
        spent += size
        results.append({"key": item["key"], "status": "ok",
                        "path": str(target), "gb": round(size / 1e9, 2),
                        "seconds": round(time.time() - started)})
        print(f"   done in {time.time()-started:.0f}s, CRC verified", flush=True)

    print("\n---- summary ----")
    for row in results:
        print(f"  {row['status']:16s} {row['key']}"
              + (f"  {row.get('gb')} GB" if row.get("gb") else ""))
    ok = sum(1 for r in results if r["status"] in ("ok", "already_present"))
    print(f"\n{ok}/{len(results)} present, {spent/1e9:.1f} GB fetched")
    (Path("build") / "helsinki_acquisition.json").parent.mkdir(exist_ok=True)
    Path("build/helsinki_acquisition.json").write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

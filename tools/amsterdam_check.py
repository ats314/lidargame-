"""Score an Amsterdam build against layers it was never given.

Two numbers this repo has never been able to produce, both cheap here:

  trees    The canopy segmentation has only ever been checked by eye. Denver
           produced 1197 "trees" in a 300 m block, then 307 after three real
           fixes, and nobody could say which was right. The BGT surveys
           vegetation objects as points, so the count and the positions are
           answerable.

           Read the ratio as an upper bound on the error, not the error. BGT
           surveys the public realm: a tree in a courtyard or a back garden --
           and the canal belt is full of them, behind the houses where nobody
           photographs -- is canopy in the returns and absent from the layer.
           A reconstructed tree far from every surveyed one is therefore
           either over-segmentation or a private tree, and this cannot tell
           them apart. What it can say is that a 6.8x ratio is not explained
           by private gardens alone.

  heights  3D BAG states a roof height per building, derived from the same AHN
           returns by a different team with a different pipeline. That is a
           level-2 check: independent in method, not in sensor. Denver's aerial
           stereo remains the stronger comparison; this one is weaker and free.

Neither layer is fed into the build. Run this after `lidarworld compile ...
--seed`, and report what it says even when it is bad.

    python tools/amsterdam_check.py build/ams/real.seed.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lidarworld.data import amsterdam                      # noqa: E402
from lidarworld.data.gis import FOOTPRINTS, fetch_footprints  # noqa: E402

USER_AGENT = "lidarworld/0.2 (+https://github.com/ats314/lidargame-)"


def _get(url: str, timeout: int = 180) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _points(geojson: dict) -> list[tuple[float, float]]:
    out = []
    for feature in geojson.get("features", []):
        geometry = feature.get("geometry") or {}
        if geometry.get("type") == "Point":
            x, y, *_ = geometry["coordinates"]
            out.append((float(x), float(y)))
    return out


def _seed_bbox(seed: dict) -> tuple[float, float, float, float]:
    """The compiled extent in RD. `bounds` is local to the seed's origin."""
    origin = seed.get("origin") or [0.0, 0.0, 0.0]
    (west, south, _), (east, north, _) = seed["bounds"]
    return (west + origin[0], south + origin[1],
            east + origin[0], north + origin[1])


def check_trees(seed: dict, bbox) -> dict:
    surveyed = _points(_get(amsterdam.bgt_items_url("vegetatieobject_punt", bbox)))
    west, south, east, north = bbox
    inside = [p for p in surveyed if west <= p[0] <= east and south <= p[1] <= north]
    found = seed.get("vegetation", [])

    nearest = []
    for tree in found:
        position = tree.get("xy")
        if not position or not inside:
            continue
        x, y = float(position[0]), float(position[1])
        nearest.append(min(((x - px) ** 2 + (y - py) ** 2) ** 0.5 for px, py in inside))

    return {
        "surveyed": len(inside),
        "reconstructed": len(found),
        "ratio": round(len(found) / len(inside), 2) if inside else None,
        "within_5m": sum(1 for d in nearest if d <= 5.0),
        "beyond_15m": sum(1 for d in nearest if d > 15.0),
        "median_nearest_m": round(statistics.median(nearest), 2) if nearest else None,
    }


def check_heights(seed: dict, bbox) -> dict:
    layer = FOOTPRINTS["amsterdam"]
    geojson = fetch_footprints(layer, bbox, in_crs="28992", out_crs="28992")
    stated: dict[str, float] = {}
    for feature in geojson.get("features", []):
        props = feature.get("properties") or {}
        roof, ground = props.get("b3_h_70p"), props.get("b3_h_maaiveld")
        key = props.get("identificatie")
        if key and roof is not None and ground is not None:
            stated[str(key)] = float(roof) - float(ground)

    deltas = []
    for building in seed.get("buildings", []):
        key = str(building.get("source_id") or "")
        height = building.get("height")
        if key in stated and height:
            deltas.append(float(height) - stated[key])

    if not deltas:
        return {"compared": 0, "stated": len(stated),
                "note": "no building in the seed carries a source id that 3D BAG "
                        "also states a height for, so there is nothing to join on"}
    absolute = sorted(abs(d) for d in deltas)
    return {
        "compared": len(deltas),
        "median_abs_error_m": round(statistics.median(absolute), 2),
        "median_bias_m": round(statistics.median(deltas), 2),
        "within_2m": round(100 * sum(d <= 2 for d in absolute) / len(absolute), 1),
        "within_5m": round(100 * sum(d <= 5 for d in absolute) / len(absolute), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", help="path to a *.seed.json from `compile --seed`")
    args = parser.parse_args()

    seed = json.loads(Path(args.seed).read_text())
    bbox = _seed_bbox(seed)
    print(f"area {bbox[0]:.0f},{bbox[1]:.0f} -> {bbox[2]:.0f},{bbox[3]:.0f} (RD)")

    trees = check_trees(seed, bbox)
    print("\ntrees vs BGT vegetatieobject_punt (surveyed, never fed in)")
    for key, value in trees.items():
        print(f"  {key:20s} {value}")

    heights = check_heights(seed, bbox)
    print("\nheights vs 3D BAG b3_h_70p (independence 2: same returns, other pipeline)")
    for key, value in heights.items():
        print(f"  {key:20s} {value}")


if __name__ == "__main__":
    main()

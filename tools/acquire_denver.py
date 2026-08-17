"""Download the Denver AOI bundle: every GIS layer, clipped, with provenance.

Not a library. A script that puts bytes on disk and writes down where each one
came from, so a demo built on them is reproducible and nobody has to guess later
whether a layer was hidden truth or input.

    python tools/acquire_denver.py --out data/denver01

Every asset records: source URL, licence text as published, retrieval time,
AOI, feature count, and SHA-256 of the bytes written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

UA = {"User-Agent": "lidarworld/0.1 (+https://github.com/ats314/lidargame-)"}
FS = "https://services1.arcgis.com/zdB7qR0BtYrg0Xpl/ArcGIS/rest/services"

# LoDo / Union Station. WGS84 minx,miny,maxx,maxy.
AOI = (-105.002, 39.740, -104.985, 39.755)

DENVER_TERMS = (
    "City and County of Denver open data. Published terms are a liability "
    "disclaimer with no explicit copyright grant, marked NOT FOR ENGINEERING "
    "PURPOSES. No commercial restriction is stated."
)

# id, service path, layer, role, independence, note
LAYERS = [
    ("roofprints",        "ODC_PROP_BUILDINGOUTLINES_A", 111, "hidden_truth", 2,
     "BLDG_HEIGH + GROUND_ELE (US survey feet) + BUILDING_I grouping"),
    ("parcels",           "ODC_PROP_PARCELS_A",          245, "prior", 3,
     "year built, floor area, structure type. 82% are condo rows sharing a footprint"),
    ("zoning",            "CCD_Zoning",                   27, "prior", 3,
     "HEIGHT_STORIES null on 63%, and null for every Downtown district here"),
    ("streets",           "ODC_TRANS_STREET_L",          145, "prior", 3,
     "VOLCLASS is the reliable class; FUNCLASS disagrees with it and must not be used"),
    ("sidewalks_2020",    "ODC_TRANS_SIDEWALKS2020_L",   333, "hidden_truth", 2,
     "compiled from 2020 DRAPP imagery, contemporaneous with the LiDAR. No width field"),
    ("sidewalks_current", "ODC_TRANS_SIDEWALKS_L",       143, "runtime", 3,
     "maintained by DOTI; present day, not the 2020 epoch"),
    ("parking_2022",      "ODC_TRANS_PARKINGLOTS_A",     139, "later_epoch", 2,
     "TYPE is Impervious/Gravel; interior rings matter"),
    ("parkland",          "DPR_Parkland_2026",             0, "prior", 3,
     "PARK_TYPE, PARK_CLASS, FACILITIES prop manifest, GIS_ACRES"),
    ("playgrounds",       "ODC_PARK_PLAYGROUNDS_A",       91, "prior", 3,
     "0 features in LoDo; kept so a park AOI picks them up"),
    ("athletic_fields",   "ODC_PARK_ATHLETICFIELDS_A",    82, "prior", 3,
     "0 features in LoDo; kept so a park AOI picks them up"),
    ("curb_ramps_2022",   "ODC_TRANS_CURBRAMPS_P",       228, "later_epoch", 3,
     "implies kerb lines and crossing geometry"),
    ("landuse_2020",      "ODC_PLAN_EXISTINGLANDUSE2020_A", 319, "prior", 2,
     "contemporaneous with the LiDAR"),
    ("tree_canopy_2020",  "ODC_ENV_TREECANOPY2020_A",    348, "prior", 2,
     "neighbourhood aggregate statistics, NOT canopy polygons"),
    ("alleys",            "ODC_PWTRN_TRN_ALLEY_L",       116, "prior", 3,
     "Denver's alley grid is structural and barely changes"),
    ("subdivisions",      "ODC_ENG_SRVSUBDIVISIONS_A",    54, "prior", 3,
     "groups buildings built together to one pattern"),
]


def get(url: str, timeout: int = 180) -> bytes:
    return urllib.request.urlopen(
        urllib.request.Request(url, headers=UA), timeout=timeout).read()


def query_url(path: str, layer: int, aoi, *, offset: int = 0,
              count: int = 1000, out_sr: str = "4326") -> str:
    q = urllib.parse.urlencode({
        "where": "1=1",
        "geometry": ",".join(str(v) for v in aoi),
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "outSR": out_sr,
        "resultOffset": offset,
        "resultRecordCount": count,
        "f": "geojson",
    })
    return f"{FS}/{path}/FeatureServer/{layer}/query?{q}"


def fetch_layer(spec, aoi, out_dir: Path) -> dict:
    """Page through a layer for the AOI and write one GeoJSON."""
    lid, path, layer, role, independence, note = spec
    features, offset, pages = [], 0, 0
    while True:
        url = query_url(path, layer, aoi, offset=offset)
        payload = json.loads(get(url))
        if "error" in payload:
            raise RuntimeError(payload["error"].get("message", "server error"))
        batch = payload.get("features", [])
        features.extend(batch)
        pages += 1
        # Page until the server stops saying there is more, not until a batch
        # looks short: a layer can return exactly the page size and be done.
        if not payload.get("properties", {}).get("exceededTransferLimit") and \
           not payload.get("exceededTransferLimit"):
            break
        if not batch or pages > 60:
            break
        offset += len(batch)

    collection = {"type": "FeatureCollection", "features": features}
    blob = json.dumps(collection, separators=(",", ":")).encode()
    target = out_dir / f"{lid}.geojson"
    target.write_bytes(blob)

    return {
        "id": lid, "file": target.name, "features": len(features), "pages": pages,
        "url": query_url(path, layer, aoi),
        "service": f"{FS}/{path}/FeatureServer/{layer}",
        "role": role, "independence": independence, "note": note,
        "license": DENVER_TERMS,
        "attribution": "City and County of Denver, Department of Technology Services",
        "retrieved": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/denver01")
    ap.add_argument("--aoi", default=",".join(str(v) for v in AOI))
    args = ap.parse_args()

    aoi = tuple(float(v) for v in args.aoi.split(","))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    assets, failures = [], []
    for spec in LAYERS:
        started = time.time()
        try:
            info = fetch_layer(spec, aoi, out_dir)
            assets.append(info)
            print(f"  {info['id']:20s} {info['features']:>6,} features  "
                  f"{info['bytes']/1024:>8.0f} KB  {time.time()-started:4.1f}s  "
                  f"[{info['role']}/L{info['independence']}]", flush=True)
        except Exception as exc:
            # One dead layer must not lose the bundle.
            failures.append({"id": spec[0], "error": f"{type(exc).__name__}: {exc}"})
            print(f"  {spec[0]:20s} FAILED {type(exc).__name__}: {str(exc)[:70]}",
                  flush=True)

    manifest = {
        "bundle": "denver01",
        "aoi_wgs84": list(aoi),
        "crs": "EPSG:4326",
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "assets": assets,
        "failures": failures,
        "note": "Roles are not advisory. A layer marked hidden_truth must not be "
                "fed to the compiler in a run whose output is then scored "
                "against it.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    total = sum(a["bytes"] for a in assets)
    print(f"\n{len(assets)} layers, {sum(a['features'] for a in assets):,} features, "
          f"{total/1024/1024:.1f} MB -> {out_dir}/manifest.json")
    if failures:
        print(f"{len(failures)} failed: {[f['id'] for f in failures]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

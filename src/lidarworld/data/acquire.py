"""Pull an acquisition manifest to disk, with provenance, and keep truth apart.

`denver.py` says what to fetch and what each layer is *for*. This turns that
into bytes on disk without losing the second half, which is the part that
usually gets lost: a directory of GeoJSON files does not remember that
`building_outlines` is the yardstick and `parcels` is an admissible prior, and
once it is forgotten someone scores a reconstruction against a layer the
compiler was handed.

So the role is enforced by the filesystem rather than documented in a README:

    <out>/input/     evidence the compiler may read
    <out>/prior/     context that may inform completion
    <out>/truth/     withheld -- hidden_truth and later_epoch layers
    <out>/manifest.json

`load()` reads back only what a mode admits, so the compiler cannot open the
truth directory by accident. Feeding truth in is not prevented by anything
clever; it is prevented by the input path not containing it.

Every layer lands with a `.provenance.json` sidecar recording where it came
from, when, under what terms, how many features arrived and -- the field that
has already caught a mistake -- which attribute names the response *actually*
carried. Layers get renamed and re-schema'd upstream without warning; a
manifest asserting `BLDG_HEIGH` proves nothing about what arrived today.

Pagination is not optional. ArcGIS silently truncates at the service's
`maxRecordCount` and sets a flag; a layer that comes back suspiciously round
(1000, 2000) and unflagged is the normal way to lose half an area of interest.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .denver import INDEPENDENCE, LAYERS, Layer, manifest

USER_AGENT = "lidarworld/0.2 (+https://github.com/ats314/lidargame-)"

#: Which directory a role lands in *when the manifest admitted it*. Role is not
#: the whole story: `manifest()` also withholds on epoch, so a plain `prior`
#: surveyed after the scan is inadmissible evidence despite its role. Anything
#: withheld for any reason goes to `truth` -- see `_target_dir`.
ROLE_DIR = {
    "input": "input",
    "prior": "prior",
    "runtime": "prior",
    "hidden_truth": "truth",
    "later_epoch": "truth",
}

WITHHELD_DIR = "truth"


def _target_dir(layer: Layer, withheld: bool) -> str:
    """Withheld beats role. A 2026 parkland layer is a prior and still not
    evidence for a 2020 scan, and routing it by role alone put it in `prior/`
    where `load()` reads it -- which is the exact leak this module exists to
    stop."""
    return WITHHELD_DIR if withheld else ROLE_DIR[layer.role]


class AcquisitionError(RuntimeError):
    """A layer could not be fetched, or came back as an ArcGIS error object."""


def _query_url(layer: Layer, bbox_wgs84, *, out_crs: str, offset: int,
               page: int, precision: int) -> str:
    query = urllib.parse.urlencode({
        "where": "1=1",
        "geometry": ",".join(str(v) for v in bbox_wgs84),
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "outSR": out_crs,
        "f": "geojson",
        "resultOffset": offset,
        "resultRecordCount": page,
        # Coordinates come back in metres, and the default is ~15 significant
        # digits of it. Denver's parcels are 4,886 polygons over the AOI and
        # that ran a single page to 41 MB, which is how the first attempt died
        # mid-stream. Three decimals is a millimetre -- far below the survey's
        # own accuracy, so this discards no information anyone has.
        "geometryPrecision": precision,
    })
    return f"{layer.url}/query?{query}"


#: Everything a flaky ArcGIS endpoint throws mid-transfer. `IncompleteRead` is
#: an HTTPException rather than a URLError, so a retry tuple built around
#: urllib alone misses the commonest large-response failure.
_TRANSIENT = (urllib.error.URLError, http.client.HTTPException,
              json.JSONDecodeError, TimeoutError, ConnectionError, OSError)


def _get_json(url: str, *, timeout: int, retries: int = 5) -> dict:
    """GET with backoff. ArcGIS answers overload with 5xx, HTML, and truncation."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
            return json.loads(payload)
        except _TRANSIENT as exc:
            last = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise AcquisitionError(f"{type(last).__name__}: {last}") from last


def fetch_layer(layer: Layer, bbox_wgs84, *, out_crs: str = "26913",
                page: int = 500, timeout: int = 300, precision: int = 3,
                max_pages: int = 200) -> dict:
    """Fetch one layer over a bbox as GeoJSON, following pagination to the end.

    ArcGIS caps a response at the service's `maxRecordCount` and reports the cap
    with `exceededTransferLimit`. Requesting a page larger than the cap does not
    raise -- it just returns the cap -- so the only safe termination condition
    is an empty page or an unflagged short one.
    """
    features: list[dict] = []
    crs = None
    truncated = False
    for pages in range(max_pages):
        payload = _get_json(
            _query_url(layer, bbox_wgs84, out_crs=out_crs,
                       offset=len(features), page=page, precision=precision),
            timeout=timeout)
        if "error" in payload:
            error = payload["error"]
            raise AcquisitionError(
                f"{layer.id}: ArcGIS error {error.get('code')} "
                f"{error.get('message')} {'; '.join(error.get('details') or [])}")
        batch = payload.get("features") or []
        crs = crs or payload.get("crs")
        features.extend(batch)
        exceeded = bool(payload.get("exceededTransferLimit")
                        or (payload.get("properties") or {}).get("exceededTransferLimit"))
        if not batch or not exceeded:
            break
    else:
        truncated = True

    result = {"type": "FeatureCollection", "features": features}
    if crs:
        result["crs"] = crs
    if truncated:
        result["lidarworld_truncated"] = True
    return result


def _fields_present(features: list[dict]) -> list[str]:
    """Attribute names that actually arrived, not the ones the manifest claims."""
    seen: set[str] = set()
    for feature in features[:500]:
        seen.update((feature.get("properties") or {}).keys())
    return sorted(seen)


def _geometry_types(features: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for feature in features:
        kind = (feature.get("geometry") or {}).get("type") or "null"
        counts[kind] = counts.get(kind, 0) + 1
    return dict(sorted(counts.items()))


def acquire_layer(layer: Layer, bbox_wgs84, out_dir: str | Path, *,
                  out_crs: str = "26913", overwrite: bool = False,
                  withheld: bool = False, withheld_reason: str = "",
                  **kwargs) -> dict:
    """Fetch one layer into `<out_dir>/<role dir>/`, with a provenance sidecar."""
    root = Path(out_dir) / _target_dir(layer, withheld)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{layer.id}.geojson"
    sidecar = root / f"{layer.id}.provenance.json"

    if path.exists() and sidecar.exists() and not overwrite:
        return json.loads(sidecar.read_text())

    started = time.time()
    geojson = fetch_layer(layer, bbox_wgs84, out_crs=out_crs, **kwargs)
    payload = json.dumps(geojson).encode()
    path.write_bytes(payload)

    features = geojson["features"]
    provenance = {
        "id": layer.id,
        "name": layer.name,
        "url": layer.url,
        "role": layer.role,
        "epoch": layer.epoch,
        "withheld": withheld,
        "withheld_reason": withheld_reason,
        "independence": layer.independence,
        "independence_note": INDEPENDENCE[layer.independence],
        "geometry_declared": layer.geometry,
        "geometry_returned": _geometry_types(features),
        "license": layer.license,
        "attribution": layer.attribution,
        "notes": layer.notes,
        "aoi_wgs84": list(bbox_wgs84),
        "out_crs": f"EPSG:{out_crs}",
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fetch_seconds": round(time.time() - started, 1),
        "features": len(features),
        # The manifest's field names are a claim about upstream, not a fact.
        "fields_present": _fields_present(features),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "file": str(path.relative_to(Path(out_dir))),
        "truncated": bool(geojson.get("lidarworld_truncated")),
    }
    sidecar.write_text(json.dumps(provenance, indent=2))
    return provenance


def acquire(bbox_wgs84, out_dir: str | Path, *, epoch: str = "2020",
            mode: str = "reconstruction", out_crs: str = "26913",
            include_withheld: bool = True, overwrite: bool = False,
            progress=None, **kwargs) -> dict:
    """Pull every layer the manifest names, admitted and withheld alike.

    Withheld layers are fetched by default and written to `truth/`. Downloading
    them is not the mistake -- you cannot score against a layer you do not have.
    The mistake is reading them back as input, which is what `load()` is for.
    """
    plan = manifest(bbox_wgs84, epoch=epoch, mode=mode)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted = [(LAYERS[entry["id"]], False, "") for entry in plan["layers"]]
    if include_withheld:
        wanted += [(LAYERS[entry["id"]], True, entry["reason"])
                   for entry in plan["withheld"]]

    acquired, failed = [], []
    for layer, withheld, reason in wanted:
        if progress:
            progress(layer, None)
        try:
            record = acquire_layer(layer, bbox_wgs84, out_dir, out_crs=out_crs,
                                   overwrite=overwrite, withheld=withheld,
                                   withheld_reason=reason, **kwargs)
            acquired.append(record)
        except AcquisitionError as exc:
            failed.append({"id": layer.id, "url": layer.url, "error": str(exc)})
        if progress:
            progress(layer, acquired[-1] if acquired and acquired[-1]["id"] == layer.id
                     else None)

    summary = {
        **{k: v for k, v in plan.items() if k != "layers"},
        "out_dir": str(out_dir),
        "acquired": acquired,
        "failed": failed,
        "features_total": sum(record["features"] for record in acquired),
        "bytes_total": sum(record["bytes"] for record in acquired),
    }
    (out_dir / "manifest.json").write_text(json.dumps(summary, indent=2))
    return summary


def load(out_dir: str | Path, *, mode: str = "reconstruction") -> dict[str, dict]:
    """Read back the layers a mode admits. `truth/` is never in the result.

    Scoring code wants the withheld layers and should say so by reading
    `truth/` directly -- an explicit path, so it appears in a diff.
    """
    out_dir = Path(out_dir)
    dirs = ["input", "prior"] if mode != "generation" else ["input", "prior"]
    layers: dict[str, dict] = {}
    for name in dirs:
        for sidecar in sorted((out_dir / name).glob("*.provenance.json")):
            record = json.loads(sidecar.read_text())
            path = out_dir / record["file"]
            layers[record["id"]] = {
                "provenance": record,
                "geojson": json.loads(path.read_text()),
            }
    return layers

"""GIS side inputs: building footprints and other vector layers.

The master build spec asks for optional GIS alongside the point cloud, and it
is the single highest-leverage extra input available. Airborne LiDAR sees roofs
well and walls badly, so grouping roof planes into buildings from patch
adjacency alone is guesswork -- a Denver block produced 1184 "structures" from
1411 patches, which is barely any grouping at all. An authoritative footprint
polygon answers that question outright, and usually carries a height too.

Municipal portals serve this through ArcGIS FeatureServer endpoints, which
accept a bbox and will reproject on the way out -- so footprints can be
requested directly in the point cloud's own CRS and used without a transform.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass

import numpy as np

USER_AGENT = "lidarworld/0.2 (+https://github.com/ats314/lidargame-)"


@dataclass
class FootprintLayer:
    id: str
    name: str
    service: str
    layer: int
    license: str
    attribution: str
    height_field: str | None = None
    ground_field: str | None = None
    notes: str = ""


#: Municipal footprint services, keyed by the place ids in catalog.PLACES.
FOOTPRINTS: dict[str, FootprintLayer] = {
    "denver": FootprintLayer(
        id="denver",
        name="Denver Building Outlines 2022",
        service="https://services1.arcgis.com/zdB7qR0BtYrg0Xpl/arcgis/rest/services/"
                "ODC_PROP_BUILDINGOUTLINES_A/FeatureServer",
        layer=111,
        license="City and County of Denver open data. The published terms are a "
                "liability disclaimer rather than a copyright restriction, and no "
                "explicit grant is stated -- treat commercial use as probable but "
                "unconfirmed, and check with the city before shipping.",
        attribution="City and County of Denver, Department of Technology Services",
        height_field="BLDG_HEIGH",
        ground_field="GROUND_ELE",
        notes="Carries per-building height and ground elevation, which is enough "
              "to validate reconstructed building heights independently.",
    ),
}


def fetch_footprints(layer: FootprintLayer, bbox_wgs84, *, out_crs: str = "26913",
                     max_records: int = 4000) -> dict:
    """Request footprints for a bbox, reprojected into `out_crs`.

    Returns GeoJSON. The server does the reprojection, so the polygons come back
    in the same frame as the point cloud and need no transform here.
    """
    query = urllib.parse.urlencode({
        "where": "1=1",
        "geometry": ",".join(str(v) for v in bbox_wgs84),
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "outSR": out_crs,
        "f": "geojson",
        "resultRecordCount": max_records,
    })
    url = f"{layer.service}/{layer.layer}/query?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def polygons(geojson: dict) -> list[np.ndarray]:
    """Flatten GeoJSON polygons to a list of (N,2) exterior rings."""
    rings = []
    for feature in geojson.get("features", []):
        geometry = feature.get("geometry") or {}
        kind = geometry.get("type")
        if kind == "Polygon":
            rings.append(np.asarray(geometry["coordinates"][0], dtype=np.float64))
        elif kind == "MultiPolygon":
            for part in geometry["coordinates"]:
                rings.append(np.asarray(part[0], dtype=np.float64))
    return rings


def attributes(geojson: dict, layer: FootprintLayer) -> list[dict]:
    out = []
    for feature in geojson.get("features", []):
        props = feature.get("properties") or {}
        out.append({
            "height": props.get(layer.height_field) if layer.height_field else None,
            "ground": props.get(layer.ground_field) if layer.ground_field else None,
            "source_id": props.get("BUILDING_I") or props.get("OBJECTID"),
        })
    return out


def point_in_polygon(points: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Vectorised even-odd test. `points` is (N,2), `ring` is a closed (M,2)."""
    x, y = points[:, 0], points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    x1, y1 = ring[:-1, 0], ring[:-1, 1]
    x2, y2 = ring[1:, 0], ring[1:, 1]
    for a, b, c, d in zip(x1, y1, x2, y2):
        crosses = ((b > y) != (d > y))
        if not crosses.any():
            continue
        t = (c - a) * (y - b) / np.where(d != b, d - b, 1e-12) + a
        inside ^= crosses & (x < t)
    return inside

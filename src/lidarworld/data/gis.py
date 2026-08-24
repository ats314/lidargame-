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
    """One footprint service. `kind` picks the protocol: municipal ArcGIS
    FeatureServers and OGC WFS both answer a bbox and both reproject on the way
    out, but they spell it differently.

    `height_units` is the unit the layer states heights in, and `height_datum`
    says what they are measured *from*: Denver publishes a height above the
    building's own ground in US survey feet, 3D BAG publishes an absolute NAP
    elevation in metres, and subtracting its ground field is what turns the
    second into the first. Normalising here rather than downstream is what stops
    `published_height` carrying one city's units as a constant.
    """
    id: str
    name: str
    service: str
    layer: int | str
    license: str
    attribution: str
    height_field: str | None = None
    ground_field: str | None = None
    kind: str = "arcgis"                    # "arcgis" | "wfs"
    height_units: str = "ft"                # "ft" (US survey foot) | "m"
    height_datum: str = "above_ground"      # "above_ground" | "absolute"
    id_fields: tuple[str, ...] = ("BUILDING_I", "OBJECTID")
    extra_fields: tuple[str, ...] = ()
    default_crs: str = "4326"
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
    "amsterdam": FootprintLayer(
        id="amsterdam",
        name="3D BAG (LoD1.2) over the BAG building register",
        service="https://data.3dbag.nl/api/BAG3D/wfs",
        layer="BAG3D:lod12",
        kind="wfs",
        license="CC BY 4.0 -- 3D BAG, TU Delft 3D geoinformation group, over "
                "Kadaster BAG (public task data). Commercial use permitted, "
                "attribution required.",
        attribution="3D BAG by the TU Delft 3D geoinformation group (CC BY 4.0), "
                    "built from BAG and AHN",
        height_field="b3_h_70p",
        ground_field="b3_h_maaiveld",
        height_units="m",
        height_datum="absolute",
        id_fields=("identificatie", "fid"),
        extra_fields=("b3_dak_type", "b3_h_50p", "b3_h_max", "b3_h_min",
                      "b3_h_nok", "b3_nodata_fractie_ahn5"),
        default_crs="28992",
        notes="Heights are NAP elevations in metres, derived per building from "
              "AHN by a third party rather than by this compiler -- so they are "
              "a level-2 independent check on reconstructed height, the same "
              "role Denver's aerial stereo plays and the reason Amsterdam was "
              "worth wiring. `b3_dak_type` states flat or slanted outright, "
              "which is the roof form Denver had to guess at and got wrong. "
              "`b3_h_70p` is the 70th roof percentile, 3D BAG's own choice for "
              "a LoD1.2 extrusion: the median runs low on a pitched roof and "
              "the max runs to the chimney.",
    ),
}


def fetch_footprints(layer: FootprintLayer, bbox, *, out_crs: str = "26913",
                     max_records: int = 4000, in_crs: str = "4326") -> dict:
    """Request footprints for a bbox, reprojected into `out_crs`.

    Returns GeoJSON. The server does the reprojection, so the polygons come back
    in the same frame as the point cloud and need no transform here.
    """
    if layer.kind == "wfs":
        return _fetch_wfs(layer, bbox, out_crs=out_crs, max_records=max_records,
                          in_crs=in_crs)
    query = urllib.parse.urlencode({
        "where": "1=1",
        "geometry": ",".join(str(v) for v in bbox),
        "geometryType": "esriGeometryEnvelope",
        "inSR": in_crs,
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


def _fetch_wfs(layer: FootprintLayer, bbox, *, out_crs: str, max_records: int,
               in_crs: str) -> dict:
    """WFS 2.0 flavour of the same request.

    The bbox carries its own CRS as a URN suffix -- without it the server
    assumes the layer's native frame and silently returns nothing when the
    numbers are degrees. `srsName` asks for the output frame, which for a Dutch
    build is RD New and therefore no transform at all.
    """
    query = urllib.parse.urlencode({
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeNames": layer.layer,
        "bbox": ",".join(str(v) for v in bbox) + f",urn:ogc:def:crs:EPSG::{in_crs}",
        "srsName": f"urn:ogc:def:crs:EPSG::{out_crs}",
        "outputFormat": "application/json",
        "count": max_records,
    })
    url = f"{layer.service}?{query}"
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


#: US survey foot. Denver states building height in whole feet.
FOOT = 0.3048


def attributes(geojson: dict, layer: FootprintLayer) -> list[dict]:
    """Per-footprint attributes, with `height` normalised to metres above the
    building's own ground whatever the layer publishes."""
    out = []
    for feature in geojson.get("features", []):
        props = feature.get("properties") or {}
        raw = props.get(layer.height_field) if layer.height_field else None
        ground = props.get(layer.ground_field) if layer.ground_field else None
        record = {
            "height": _height_metres(raw, ground, layer),
            "ground": _number(ground),
            "source_id": next((props[f] for f in layer.id_fields if props.get(f)), None),
        }
        for field in layer.extra_fields:
            if field in props:
                record[field] = props[field]
        out.append(record)
    return out


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _height_metres(raw, ground, layer: FootprintLayer) -> float | None:
    """Height above the building's own ground, in metres."""
    value = _number(raw)
    if value is None:
        return None
    if layer.height_units == "ft":
        value *= FOOT
    if layer.height_datum == "absolute":
        base = _number(ground)
        if base is None:
            return None
        value -= base
    return value


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


@dataclass
class StreetLayer:
    """A street centreline network. Same two protocols as the footprints, and
    the same reason for existing: intensity finds under 6% of the carriageway,
    a centreline and a width find all of it."""
    id: str
    name: str
    service: str
    layer: int | str
    license: str
    attribution: str
    kind: str = "arcgis"
    default_crs: str = "4326"
    notes: str = ""


STREETS: dict[str, StreetLayer] = {
    "denver": StreetLayer(
        id="denver",
        name="Denver Street Centerlines",
        service="https://services1.arcgis.com/zdB7qR0BtYrg0Xpl/arcgis/rest/services/"
                "ODC_TRANS_STREET_L/FeatureServer",
        layer=145,
        license=FOOTPRINTS["denver"].license,
        attribution=FOOTPRINTS["denver"].attribution,
        notes="Class lives in VOLCLASS, not FUNCLASS -- see data/denver.py.",
    ),
    "amsterdam": StreetLayer(
        id="amsterdam",
        name="NWB Wegvakken (Nationaal Wegenbestand)",
        service="https://service.pdok.nl/rws/nwbwegen/wfs/v1_0",
        layer="wegvakken",
        kind="wfs",
        default_crs="28992",
        license="CC BY 4.0 -- Rijkswaterstaat, via PDOK",
        attribution="Nationaal Wegenbestand, Rijkswaterstaat (CC BY 4.0)",
        notes="Every carriageway in the country as a centreline in RD, with a "
              "street name and the authority that maintains it. There is no "
              "width and no functional class: `wegbehsrt` (R/P/G/W -- state, "
              "province, municipality, water board) is the only hierarchy on "
              "offer, and inside a city centre it is flat, so the widths come "
              "out near-uniform. Amsterdam's canal streets genuinely are, which "
              "is why this is tolerable here and would not be in Denver.",
    ),
}


def fetch_streets(layer: StreetLayer, bbox, *, out_crs: str, in_crs: str = "4326",
                  max_records: int = 4000) -> dict:
    """Centrelines over a bbox, as GeoJSON in `out_crs`."""
    shim = FootprintLayer(
        id=layer.id, name=layer.name, service=layer.service, layer=layer.layer,
        license=layer.license, attribution=layer.attribution, kind=layer.kind,
        default_crs=layer.default_crs)
    return fetch_footprints(shim, bbox, out_crs=out_crs, in_crs=in_crs,
                            max_records=max_records)


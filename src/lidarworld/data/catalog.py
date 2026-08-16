"""Catalogue of LiDAR sources that are free to use commercially.

Every entry here has been chosen on licence first. The rule is strict: if the
terms do not permit commercial use and redistribution of derived works, it does
not go in this file -- it goes in ``RESTRICTED`` with the reason, so nobody
wires it in by accident.

Bulk data is fetched, never vendored. A single 3DEP tile is ~65 MB and a city
is thousands of them, so the repository carries the *addresses* and a fetcher;
`lidarworld fetch` resolves an area of interest into tile URLs and pulls them.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Source:
    id: str
    name: str
    license: str
    commercial: bool
    attribution: str
    coverage: str
    classified: str
    api: str = ""
    notes: str = ""
    homepage: str = ""


#: Cleared for commercial use, including derived works.
COMMERCIAL: dict[str, Source] = {s.id: s for s in [
    Source(
        id="usgs_3dep",
        name="USGS 3D Elevation Program (3DEP)",
        license="Public domain (US Government work, 17 U.S.C. §105)",
        commercial=True,
        attribution="U.S. Geological Survey, 3D Elevation Program",
        coverage="United States, ~100% at QL2 or better",
        classified="ASPRS classes; most projects label ground/noise reliably, "
                   "buildings and vegetation vary by project",
        api="https://tnmaccess.nationalmap.gov/api/v1/products",
        homepage="https://www.usgs.gov/3d-elevation-program",
        notes="No use restrictions of any kind. The single best bulk source.",
    ),
    Source(
        id="noaa_coastal",
        name="NOAA Digital Coast LiDAR",
        license="Public domain (US Government work)",
        commercial=True,
        attribution="NOAA Office for Coastal Management",
        coverage="US coastline, Great Lakes, territories",
        classified="ASPRS classes, bathymetric where surveyed",
        api="https://coast.noaa.gov/dataviewer/",
        homepage="https://coast.noaa.gov/digitalcoast/",
    ),
    Source(
        id="ahn_netherlands",
        name="AHN (Actueel Hoogtebestand Nederland)",
        license="CC BY 4.0",
        commercial=True,
        attribution="AHN, Rijkswaterstaat / Waterschappen (CC BY 4.0)",
        coverage="Entire Netherlands, 8-10 pts/m²",
        classified="ASPRS classes, high quality",
        homepage="https://www.ahn.nl/",
        notes="Among the densest and cleanest national datasets anywhere. "
              "Pairs with the 3D BAG building models.",
    ),
    Source(
        id="ea_england",
        name="Environment Agency National LiDAR Programme",
        license="Open Government Licence v3",
        commercial=True,
        attribution="© Environment Agency copyright and/or database right, "
                    "Open Government Licence v3",
        coverage="England, ~100% at 1 m or better",
        classified="Ground/non-ground; DSM and DTM products",
        homepage="https://environment.data.gov.uk/survey",
    ),
    Source(
        id="swisstopo",
        name="swissSURFACE3D",
        license="Open Government Data (free use, attribution)",
        commercial=True,
        attribution="© swisstopo",
        coverage="Switzerland, 15-20 pts/m²",
        classified="ASPRS classes including buildings and vegetation",
        homepage="https://www.swisstopo.admin.ch/en/height-model-swisssurface3d",
    ),
    Source(
        id="denmark_dhm",
        name="Danmarks Højdemodel (DHM/Punktsky)",
        license="Public sector open data (free, attribution)",
        commercial=True,
        attribution="Styrelsen for Dataforsyning og Infrastruktur",
        coverage="Denmark, 4-5 pts/m²",
        classified="ASPRS classes",
        homepage="https://dataforsyningen.dk/data/930",
    ),
    Source(
        id="spain_pnoa",
        name="PNOA LiDAR",
        license="CC BY 4.0",
        commercial=True,
        attribution="© Instituto Geográfico Nacional de España (CC BY 4.0)",
        coverage="Spain, 0.5-14 pts/m² by coverage generation",
        classified="ASPRS classes",
        homepage="https://pnoa.ign.es/pnoa-lidar",
    ),
    Source(
        id="co_hazard_mapping",
        name="Colorado Hazard Mapping Program LiDAR",
        license="State of Colorado open data (free download, no stated restriction)",
        commercial=True,
        attribution="Colorado Water Conservation Board / Colorado Hazard Mapping Program",
        coverage="Colorado, flood-risk focused watersheds and Front Range",
        classified="ASPRS classes; DEM and DSM derivatives alongside the point cloud",
        homepage="https://coloradohazardmapping.com/lidarDownload",
        notes="Denser and often more recent than 3DEP over its coverage. "
              "Complements CO_DRCOG_2020 rather than replacing it.",
    ),
    Source(
        id="drcog_open_data",
        name="DRCOG Regional Data Catalog",
        license="Varies by dataset; most are open with attribution",
        commercial=True,
        attribution="Denver Regional Council of Governments",
        coverage="Denver metro: built environment, land use, transportation",
        classified="Vector GIS, not point clouds -- side input for the compiler",
        homepage="https://data.drcog.org/",
        notes="Read each dataset's own terms. Useful as GIS side input "
              "(footprints, land use, road centrelines) rather than as LiDAR.",
    ),
    Source(
        id="denver_open_data",
        name="Denver Open Data Catalog (geospatial)",
        license="Liability disclaimer, no explicit copyright grant -- commercial "
                "use probable but UNCONFIRMED; verify with the city before shipping",
        commercial=True,
        attribution="City and County of Denver, Department of Technology Services",
        coverage="City and County of Denver",
        classified="Vector GIS. Building Outlines 2022 carries height and ground "
                   "elevation per building",
        homepage="https://opendata-geospatialdenver.hub.arcgis.com",
        notes="Building footprints are the highest-leverage side input available: "
              "they answer building grouping outright, which airborne LiDAR alone "
              "cannot. See lidarworld.data.gis. Licence is the weakest of any "
              "source listed here -- flagged rather than assumed.",
    ),
    Source(
        id="opentopography",
        name="OpenTopography",
        license="Per-dataset; many CC BY or public domain",
        commercial=True,
        attribution="Check the individual dataset's citation block",
        coverage="Global aggregator, strong outside the US",
        classified="Varies by dataset",
        api="https://portal.opentopography.org/API/",
        homepage="https://opentopography.org/",
        notes="Licence varies PER DATASET. Read the citation before shipping "
              "anything derived from one.",
    ),
]}


#: Deliberately excluded. Do not wire these in without re-reading the terms.
RESTRICTED: dict[str, str] = {
    "semantickitti": "CC BY-NC-SA 4.0 -- non-commercial only.",
    "kitti": "CC BY-NC-SA 3.0 -- non-commercial only.",
    "nuscenes": "CC BY-NC-SA 4.0 -- non-commercial for the full dataset.",
    "paris_lille_3d": "CC BY-NC-SA 4.0 -- non-commercial only.",
    "waymo_open": "Custom licence, non-commercial research use.",
    "argoverse": "CC BY-NC-SA 4.0 -- non-commercial only.",
    "a2d2": "CC BY-ND 4.0 -- no derivatives, which a compiler necessarily makes.",
    "toronto_3d": "CC BY-NC 4.0 -- non-commercial only.",
    "dales": "CC BY-NC-SA 4.0 -- non-commercial only.",
}


def commercial_sources() -> list[Source]:
    return list(COMMERCIAL.values())


def describe(source_id: str) -> Source:
    if source_id in COMMERCIAL:
        return COMMERCIAL[source_id]
    if source_id in RESTRICTED:
        raise ValueError(
            f"{source_id} is excluded on licence grounds: {RESTRICTED[source_id]}")
    raise KeyError(f"unknown source {source_id!r}; have {sorted(COMMERCIAL)}")


#: Pre-resolved areas of interest, so `lidarworld fetch denver_lodo` just works.
#: bbox is (west, south, east, north) in WGS84 degrees.
PLACES: dict[str, dict] = {
    "denver_lodo": {
        "source": "usgs_3dep",
        "bbox_wgs84": (-105.002, 39.740, -104.985, 39.755),
        "description": "Denver LoDo / Union Station, CO. Dense downtown grid.",
        "project": "CO_DRCOG_2020_B20",
        "crs": "EPSG:26913",
        "suggested_crop": (499800, 4400100, 500300, 4400600),
    },
    "denver_capitol": {
        "source": "usgs_3dep",
        "bbox_wgs84": (-104.990, 39.735, -104.980, 39.745),
        "description": "Colorado State Capitol and Civic Center, Denver.",
        "project": "CO_DRCOG_2020_B20",
        "crs": "EPSG:26913",
        "suggested_crop": (501000, 4398500, 501600, 4399100),
    },
    "manhattan_midtown": {
        "source": "usgs_3dep",
        "bbox_wgs84": (-73.990, 40.750, -73.975, 40.762),
        "description": "Midtown Manhattan. Extreme vertical relief, deep street canyons.",
        "crs": "EPSG:26918",
    },
    "sf_downtown": {
        "source": "usgs_3dep",
        "bbox_wgs84": (-122.408, 37.788, -122.393, 37.798),
        "description": "San Francisco Financial District. Steep terrain plus towers.",
        "crs": "EPSG:26910",
    },
    "dc_mall": {
        "source": "usgs_3dep",
        "bbox_wgs84": (-77.040, 38.887, -77.020, 38.895),
        "description": "National Mall, Washington DC. Low-rise, monumental, open ground.",
        "crs": "EPSG:26918",
    },
}

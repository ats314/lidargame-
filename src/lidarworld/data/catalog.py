"""Catalogue of LiDAR sources the compiler can read.

Every entry records what its terms actually are, in the `license` field, and
whether those terms clear commercial use, in the `commercial` flag. That record
is the deliverable -- it is not a gate. Licence decisions belong to whoever owns
the project, so nothing here refuses to hand back a source, and no stage further
down asks. Read `license` before shipping something derived from a source.

Bulk data is fetched, never vendored. A single 3DEP tile is ~65 MB and a city
is thousands of them, so the repository carries the *addresses* and a fetcher;
`lidarworld fetch` resolves an area of interest into tile URLs and pulls them.

The annotated benchmarks are here for a specific reason. This compiler infers
semantics from geometry and has no way to check itself: on the Denver tile the
only published classes are ground and noise, so a wrong vegetation rule is
invisible. DALES is airborne and labelled, which makes it the one source that
can score this pipeline's own inference rather than a car's.
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
    #: Key into `lidarworld.semantics.vocab.VOCABULARIES`, when labelled.
    vocabulary: str = ""
    #: What `lidarworld.ingest` adapter reads it, when the format is known.
    fmt: str = ""


#: Terms clear commercial use and derived works.
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


#: Readable, but the terms restrict commercial use or derived works. Recorded
#: so the constraint is visible at the point of use, not enforced here.
NONCOMMERCIAL: dict[str, Source] = {s.id: s for s in [
    Source(
        id="dales",
        name="DALES (Dayton Annotated LiDAR Earth Scan)",
        license="CC BY-NC-SA 4.0 -- non-commercial, share-alike",
        commercial=False,
        attribution="Varney, Asari & Graehling, University of Dayton",
        coverage="10 km² of Surrey, BC. 505M points, aerial, 8 classes",
        classified="Hand-labelled: ground, vegetation, cars, trucks, power lines, "
                   "fences, poles, buildings",
        homepage="https://udayton.edu/engineering/research/centers/vision_lab/"
                 "research/was_data_analysis_and_processing/dale.php",
        vocabulary="dales",
        fmt="ply",
        notes="The only *airborne* labelled benchmark here, and therefore the only "
              "one that can score this compiler's semantic inference on the kind of "
              "data it actually compiles. Everything else is a car's point of view. "
              "Highest-value source in this file for that reason alone.",
    ),
    Source(
        id="toronto_3d",
        name="Toronto-3D",
        license="CC BY-NC 4.0 -- non-commercial",
        commercial=False,
        attribution="Tan et al., University of Waterloo",
        coverage="1 km of Avenue Road, Toronto. 78M points, MLS, 8 classes",
        classified="road, road markings, natural, building, utility line, pole, car, fence",
        homepage="https://github.com/WeikaiTan/Toronto-3D",
        vocabulary="toronto_3d",
        fmt="ply",
        notes="Street level, so facades are seen head-on rather than at the glancing "
              "angle airborne data gives. Directly relevant to the wall reconstruction "
              "weaknesses.",
    ),
    Source(
        id="paris_lille_3d",
        name="Paris-Lille-3D",
        license="CC BY-NC-SA 4.0 -- non-commercial, share-alike",
        commercial=False,
        attribution="Roynard, Deschaud & Goulette, Mines ParisTech",
        coverage="2 km of Paris and Lille. 143M points, MLS",
        classified="Coarse and fine class hierarchy; ground, building, pole, "
                   "bollard, barrier, pedestrian, car, natural",
        homepage="https://npm3d.fr/paris-lille-3d",
        vocabulary="paris_lille_3d",
        fmt="ply",
        notes="Dense facades with per-point labels. European street geometry, which "
              "stresses the theme packs differently from a US grid.",
    ),
    Source(
        id="semantickitti",
        name="SemanticKITTI",
        license="CC BY-NC-SA 4.0 -- non-commercial, share-alike",
        commercial=False,
        attribution="Behley et al., University of Bonn",
        coverage="Karlsruhe, 43,552 labelled scans, 28 classes",
        classified="Full per-point labels including moving/static distinction",
        homepage="https://semantic-kitti.org/",
        vocabulary="semantickitti",
        fmt="kitti",
        notes="Known sensor pose per scan, which is what forward validation needs to "
              "mean anything -- comparing a scan against a viewpoint it was not taken "
              "from measures nothing.",
    ),
    Source(
        id="kitti",
        name="KITTI raw / odometry",
        license="CC BY-NC-SA 3.0 -- non-commercial, share-alike",
        commercial=False,
        attribution="Geiger, Lenz & Urtasun, KIT and Toyota TI Chicago",
        coverage="Karlsruhe. Velodyne HDL-64E sweeps with pose",
        classified="Unlabelled; SemanticKITTI supplies the labels",
        homepage="https://www.cvlibs.net/datasets/kitti/",
        fmt="kitti",
    ),
    Source(
        id="nuscenes",
        name="nuScenes / nuScenes-lidarseg",
        license="CC BY-NC-SA 4.0 for the full dataset -- non-commercial",
        commercial=False,
        attribution="Motional (nuTonomy)",
        coverage="Boston and Singapore, 1000 scenes, 32-beam",
        classified="lidarseg: 32 raw classes over 1.4B points",
        homepage="https://www.nuscenes.org/",
        vocabulary="nuscenes",
        fmt="nuscenes",
        notes="Sparser than KITTI (32 beams vs 64) but two cities and both traffic "
              "handednesses.",
    ),
    Source(
        id="a2d2",
        name="Audi A2D2",
        license="CC BY-ND 4.0 -- no derivative works",
        commercial=False,
        attribution="Audi AG",
        coverage="Germany, 41k labelled frames",
        classified="Semantic labels projected from camera, 38 classes",
        homepage="https://www.a2d2.audi/a2d2/en.html",
        notes="ND is the awkward one: a compiler produces a derivative by definition, "
              "so this is the least usable of the set regardless of intent. Labels "
              "come from image projection rather than direct 3D annotation.",
    ),
    Source(
        id="waymo_open",
        name="Waymo Open Dataset",
        license="Custom Waymo licence -- non-commercial research use",
        commercial=False,
        attribution="Waymo LLC",
        coverage="US cities, 1150 scenes, 5 sensors",
        classified="3D semantic segmentation on a labelled subset, 23 classes",
        homepage="https://waymo.com/open/",
        notes="TFRecord/protobuf, so reading it needs tensorflow -- a heavier "
              "dependency than everything else in this project combined. Catalogued "
              "for completeness; no adapter.",
    ),
    Source(
        id="argoverse",
        name="Argoverse 2",
        license="CC BY-NC-SA 4.0 -- non-commercial, share-alike",
        commercial=False,
        attribution="Argo AI",
        coverage="Six US cities, 1000 scenes",
        classified="3D object annotations; LiDAR is unlabelled per-point",
        homepage="https://www.argoverse.org/av2.html",
        notes="Apache Feather/parquet, so reading it needs pyarrow. Object boxes "
              "rather than per-point labels, which is the wrong shape for this "
              "compiler's semantics stage.",
    ),
]}

#: Every source, whatever the terms. This is the lookup everything else uses.
SOURCES: dict[str, Source] = {**COMMERCIAL, **NONCOMMERCIAL}

#: Backwards-compatible view: id -> the licence line, for the restricted set.
RESTRICTED: dict[str, str] = {k: v.license for k, v in NONCOMMERCIAL.items()}


def commercial_sources() -> list[Source]:
    return list(COMMERCIAL.values())


def all_sources() -> list[Source]:
    return list(SOURCES.values())


def describe(source_id: str) -> Source:
    """Look up a source. Never refuses -- read `.license` and `.commercial`."""
    if source_id in SOURCES:
        return SOURCES[source_id]
    raise KeyError(f"unknown source {source_id!r}; have {sorted(SOURCES)}")


#: Pre-resolved areas of interest, so `lidarworld fetch denver_lodo` just works.
#: bbox is (west, south, east, north) in WGS84 degrees.
PLACES: dict[str, dict] = {
    "denver_lodo": {
        "source": "usgs_3dep",
        # The published extents index resolves this to an exact tile. Without
        # it, The National Map answers a bbox query with any overlapping tile
        # and the answer moves between runs -- which broke the Pages deploy
        # when CI got a different tile than the one the demo was tuned on.
        "acquisition": "co_drcog_2020_b2",
        "bbox_wgs84": (-105.002, 39.740, -104.985, 39.755),
        "description": "Denver LoDo / Union Station, CO. Dense downtown grid.",
        "project": "CO_DRCOG_2020_B20",
        "crs": "EPSG:26913",
        "suggested_crop": (499800, 4400100, 500300, 4400600),
    },
    "denver_capitol": {
        "source": "usgs_3dep",
        "acquisition": "co_drcog_2020_b2",
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

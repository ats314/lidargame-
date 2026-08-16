"""Denver acquisition manifest: what to pull, and what each layer is *for*.

Denver is the best available laboratory for this compiler, and not because the
LiDAR is special. It is because several organisations independently described
the same physical city in the same period: DRCOG flew QL2 LiDAR in 2020, and the
City stereocompiled building outlines, sidewalks, curb ramps, land use and tree
canopy from contemporaneous imagery. Two descriptions of one city is what lets
you hide one and check whether the compiler recovers it.

Point clouds and polygons mix freely -- footprints already drive building
grouping and wall extrusion, and that is the point of the thing. `role` is not
about keeping them apart. It is about not marking your own homework: a layer the
compiler consumed cannot also be the yardstick it is measured against. In
generation mode, where the goal is a believable city rather than a faithful one,
everything is admitted.

Which means the interesting field here is not the URL, it is `role`:

``input``         evidence the compiler is allowed to see.
``prior``         context that may inform completion but is not observation.
``hidden_truth``  independently surveyed geometry, withheld and used to *score*
                  a reconstruction. Feeding this in is the one mistake that
                  invalidates every number downstream.
``later_epoch``   the same feature surveyed after the LiDAR. Useful for change
                  detection, invalid as truth for the earlier epoch.
``runtime``       current-day layers for building a playable city, where being
                  faithful to a 2020 reconstruction is not the goal.

`manifest()` enforces that separation rather than documenting it: ask for a
reconstruction manifest and hidden-truth and later-epoch layers are not in it.

Verified against the live catalogue on 2026-08-16 -- 341 datasets, of which
exactly one mentions LiDAR ("Tree Canopy 2014", a derived product). Denver
publishes no point cloud of its own. Everything here is vector GIS, which is
precisely why it is useful as prior and truth rather than as input.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

#: Every layer is served from the City's one ArcGIS Online organisation.
SERVICE = "https://services1.arcgis.com/zdB7qR0BtYrg0Xpl/arcgis/rest/services"

#: What the catalogue actually says. Not a CC licence -- a liability disclaimer
#: with no explicit grant, ending "NOT FOR ENGINEERING PURPOSES". Recorded
#: verbatim in spirit so nobody assumes more than is written.
DENVER_TERMS = ("City and County of Denver open data: liability disclaimer, no "
                "explicit copyright grant, marked NOT FOR ENGINEERING PURPOSES")
DENVER_ATTRIBUTION = "City and County of Denver, Department of Technology Services"

ROLES = ("input", "prior", "hidden_truth", "later_epoch", "runtime")

#: How independent a layer is from the LiDAR it would be used to check. This is
#: the axis that decides whether a validation number means anything, and it is
#: not the same as `role`: a layer can be withheld from the compiler and still
#: be worthless as truth, because it was made *from* the same returns.
#:
#: 0  the returns themselves
#: 1  same-sensor derivative -- contours, DEM/DSM, the LAZ's own classes.
#:    Scoring a reconstruction against these tests the source against its own
#:    derivative and inflates the result. Constraints, never validation.
#: 2  independent modality -- stereocompiled from imagery. Different sensor,
#:    different failure modes, so agreement is real corroboration.
#: 3  external semantic record -- parcels, zoning, inventories. Surveyed on the
#:    ground by someone with no remote-sensing pipeline in the loop.
#: 4  this compiler's own inference. Never evidence for itself.
INDEPENDENCE = {0: "raw returns", 1: "same-sensor derivative",
                2: "independent modality", 3: "external record",
                4: "our inference"}


@dataclass(frozen=True)
class Layer:
    id: str
    name: str
    path: str                   # service path, appended to SERVICE
    layer: int
    role: str
    epoch: str                  # acquisition/compilation epoch, or "current"
    geometry: str               # polygon | polyline | point
    #: See INDEPENDENCE. Only level >= 2 may be used to score a reconstruction.
    independence: int = 2
    license: str = DENVER_TERMS
    attribution: str = DENVER_ATTRIBUTION
    notes: str = ""

    @property
    def service(self) -> str:
        return f"{SERVICE}/{self.path}/FeatureServer"

    @property
    def url(self) -> str:
        return f"{self.service}/{self.layer}"


LAYERS: dict[str, Layer] = {layer.id: layer for layer in [
    Layer(
        id="building_outlines",
        name="Building Outlines 2022",
        path="ODC_PROP_BUILDINGOUTLINES_A", layer=111,
        role="hidden_truth", epoch="2022", geometry="polygon", independence=2,
        notes="Stereocompiled from DRAPP imagery, carrying height and ground "
              "elevation per building. Independently surveyed, so it is the "
              "measuring stick for footprint and height recovery -- which is "
              "exactly why the compiler must not be handed it while being "
              "scored. The pipeline's --footprints flag deliberately promotes "
              "it to 'input'; that is world-generation mode, not reconstruction.",
    ),
    Layer(
        id="parcels",
        name="Parcels",
        path="ODC_PROP_PARCELS_A", layer=245,
        role="prior", epoch="current", geometry="polygon", independence=3,
        notes="Year built, above-grade area, commercial structure type, unit "
              "count, land area. The architectural prior: it says what kind of "
              "building should be there when the returns are too thin to say. "
              "Continuously edited, so it describes today, not 2020.",
    ),
    Layer(
        id="zoning",
        name="Zoning",
        path="ODC_ZONE_ZONING_A", layer=209,
        role="prior", epoch="current", geometry="polygon", independence=3,
        notes="Form-based zoning encodes expected massing and height directly, "
              "which is a stronger completion prior than land use.",
    ),
    Layer(
        id="landuse_2020",
        name="Existing Landuse 2020",
        path="ODC_PLAN_EXISTINGLANDUSE2020_A", layer=319,
        role="prior", epoch="2020", geometry="polygon", independence=2,
        notes="Contemporaneous with the LiDAR, unlike parcels and zoning. The "
              "one land-use prior admissible in reconstruction mode.",
    ),
    Layer(
        id="sidewalks_2020",
        name="Sidewalk Line 2020",
        path="ODC_TRANS_SIDEWALKS2020_L", layer=333,
        role="hidden_truth", epoch="2020", geometry="polyline", independence=2,
        notes="Stereocompiled from the 2020 DRAPP 4-band RGBIR acquisition at "
              "0.5 ft GSD -- the same year as the LiDAR. Truth for road-edge "
              "and kerb recovery, which airborne returns describe poorly.",
    ),
    Layer(
        id="street_centerlines",
        name="Street Centerlines",
        path="ODC_TRANS_STREET_L", layer=145,
        role="prior", epoch="current", geometry="polyline", independence=3,
        notes="Topology of the road graph. Where the returns lose a stretch of "
              "carriageway, the centreline says whether the road continued.",
    ),
    Layer(
        id="alleys",
        name="Alleys",
        path="ODC_PWTRN_TRN_ALLEY_L", layer=116,
        role="prior", epoch="2011", geometry="polyline", independence=3,
        notes="Denver's alley grid is a structural feature of the block layout "
              "and barely changes.",
    ),
    Layer(
        id="tree_canopy_2020",
        name="Tree Canopy 2020",
        path="ODC_ENV_TREECANOPY2020_A", layer=348,
        role="prior", epoch="2020", geometry="polygon", independence=2,
        notes="NOT canopy polygons. Queried live: 7 features over the LoDo AOI, "
              "carrying CANOPY_ACRES / PCT_CANOPY / NBHD_NAME -- neighbourhood "
              "aggregate statistics, one row per neighbourhood. Useful as a "
              "density prior and useless as per-tree truth, so it cannot check "
              "this compiler's tree segmentation however much that is wanted. "
              "Denver publishes no per-tree canopy geometry.",
    ),
    Layer(
        id="curb_ramps_2022",
        name="Curb Ramps 2022",
        path="ODC_TRANS_CURBRAMPS_P", layer=228,
        role="later_epoch", epoch="2022", geometry="point", independence=3,
        notes="Kerb ramp positions imply kerb lines and crossing geometry.",
    ),
    Layer(
        id="sidewalks_current",
        name="Sidewalks - Current",
        path="ODC_TRANS_SIDEWALKS_L", layer=143,
        role="runtime", epoch="current", geometry="polyline", independence=3,
        notes="Maintained by DOTI with widths. For building a present-day "
              "playable city, not for scoring a 2020 reconstruction.",
    ),
    Layer(
        id="subdivisions",
        name="Subdivisions",
        path="ODC_ENG_SRVSUBDIVISIONS_A", layer=54,
        role="prior", epoch="current", geometry="polygon", independence=3,
        notes="Subdivision boundaries group buildings built together, to the "
              "same pattern. The natural unit for learning an architectural "
              "family rather than guessing per building.",
    ),
]}

#: The LiDAR epochs flown over Denver. All airborne -- no ground-level or
#: mobile point cloud is published for this city by anyone.
LIDAR_EPOCHS: dict[str, dict] = {
    "2020": {"project": "CO_DRCOG_2020_B20", "quality": "QL2",
             "source": "usgs_3dep", "crs": "EPSG:26913",
             "notes": "DRCOG Block 20, flown with USGS support, released 2022. "
                      "The default epoch: contemporaneous with the 2020 "
                      "imagery-derived layers, which is what makes those usable "
                      "as truth."},
    "2013": {"project": "CO_SoPlatteRiver_Lidar_2013", "quality": "QL2",
             "source": "usgs_3dep", "crs": "EPSG:26913",
             "notes": "South Platte River and Denver post-flood."},
    "2011": {"project": "CO_DenverArea_2011", "quality": "QL2",
             "source": "usgs_3dep", "crs": "EPSG:26913",
             "notes": "Earliest Denver-area acquisition."},
}


#: Probed 2026-08-16 so the next person does not repeat the hunt. Boulder is
#: often suggested as an instance-level vegetation laboratory -- segmented
#: trees, a municipal tree inventory with species and mature height, a LiDAR
#: tile index. None of it was reachable.
BOULDER_PROBE = {
    "service_root": "https://maps.bouldercolorado.gov/arcgis/rest/services",
    "found": {
        "raster/AP2020Cached3inWM": "2020 aerial photography, 3 inch. Genuine "
            "independent modality (level 2) and contemporaneous with the LiDAR.",
        "general/Contours": "LiDAR-derived contours. Level 1: a constraint on "
            "terrain completion, never a check on the scan that produced it.",
        "raster/CHM2013": "canopy height model, listed but returns HTTP 500.",
    },
    "not_found": [
        "segmented trees / tree instances",
        "municipal tree inventory with species and mature height",
        "LiDAR tile index with per-tile classification percentages",
        "tree canopy polygons",
    ],
    "dead_hosts": ["opendata-bouldercolorado.hub.arcgis.com (404)",
                   "opendata.bouldercolorado.gov (522)",
                   "gis.bouldercounty.org (502)"],
    "conclusion": "No instance-level vegetation truth is published here. The "
                  "compiler's tree segmentation still has nothing to be scored "
                  "against, which remains the open problem DALES exists to solve.",
}


def validators(min_independence: int = 2) -> list[Layer]:
    """Layers admissible as a yardstick, i.e. not made from the same returns."""
    return [l for l in LAYERS.values() if l.independence >= min_independence]


def layers_for(role: str) -> list[Layer]:
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}; have {list(ROLES)}")
    return [layer for layer in LAYERS.values() if layer.role == role]


def _epoch_year(value: str) -> int | None:
    return int(value) if value.isdigit() else None


def manifest(bbox_wgs84, *, epoch: str = "2020", mode: str = "reconstruction",
             crs: str = "EPSG:26913") -> dict:
    """Everything to pull for one area of interest, with each layer's role.

    ``mode='reconstruction'`` admits only evidence that existed at `epoch`:
    hidden truth and later surveys are listed separately, under `withheld`, so a
    score computed against them means something. ``mode='generation'`` admits
    everything, because the goal there is a believable city rather than a
    faithful one.
    """
    if mode not in ("reconstruction", "generation"):
        raise ValueError(f"mode must be reconstruction or generation, not {mode!r}")
    if epoch not in LIDAR_EPOCHS:
        raise ValueError(f"no Denver LiDAR epoch {epoch!r}; have {sorted(LIDAR_EPOCHS)}")

    year = _epoch_year(epoch)
    admitted, withheld = [], []
    for layer in LAYERS.values():
        reason = None
        if layer.role in ("hidden_truth", "later_epoch"):
            reason = f"{layer.role}: withheld so a score against it is honest"
        elif mode == "reconstruction" and layer.role == "runtime":
            reason = "runtime layer: describes today, not the reconstructed epoch"
        else:
            layer_year = _epoch_year(layer.epoch)
            if (mode == "reconstruction" and year is not None
                    and layer_year is not None and layer_year > year):
                reason = f"surveyed {layer.epoch}, after the {epoch} observation"
        if mode == "generation" or reason is None:
            admitted.append(layer)
        else:
            withheld.append((layer, reason))

    lidar = LIDAR_EPOCHS[epoch]
    return {
        "aoi_wgs84": list(bbox_wgs84),
        "crs": crs,
        "epoch": epoch,
        "mode": mode,
        "generated": date.today().isoformat(),
        "lidar": {**lidar, "role": "input", "independence": 0},
        "layers": [
            {"id": l.id, "name": l.name, "role": l.role, "epoch": l.epoch,
             "geometry": l.geometry, "independence": l.independence,
             "independence_note": INDEPENDENCE[l.independence],
             "url": l.url, "license": l.license,
             "attribution": l.attribution, "notes": l.notes}
            for l in admitted
        ],
        "withheld": [
            {"id": l.id, "name": l.name, "role": l.role, "epoch": l.epoch,
             "independence": l.independence, "url": l.url, "reason": reason}
            for l, reason in withheld
        ],
    }


def query_url(layer: Layer, bbox_wgs84, *, out_crs: str = "26913",
              max_records: int = 4000) -> str:
    """A ready-to-fetch GeoJSON query for one layer over the area of interest."""
    import urllib.parse

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
    return f"{layer.url}/query?{query}"

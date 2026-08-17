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

#: DRCOG's Regional Data Catalog. The portal is a static front end over an
#: authenticated GraphQL API, so the published terms are not machine-readable
#: from it; the data itself is served openly and unauthenticated from the ARCGIS
#: server below. Recorded as found, not as assumed -- and per the ground rules,
#: not a reason to hold up work.
DRCOG_ARCGIS = "https://gis.drcog.org/server/rest/services/RDC"
DRCOG_TERMS = ("DRCOG Regional Data Catalog: served open and unauthenticated; "
               "explicit terms not retrievable from the portal (JS front end "
               "over an authenticated API) as probed 2026-08-17")
DRCOG_ATTRIBUTION = "Denver Regional Council of Governments (DRCOG)"

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
    path: str                   # service path, appended to `root`
    layer: int
    role: str
    epoch: str                  # acquisition/compilation epoch, or "current"
    geometry: str               # polygon | polyline | point
    #: See INDEPENDENCE. Only level >= 2 may be used to score a reconstruction.
    independence: int = 2
    license: str = DENVER_TERMS
    attribution: str = DENVER_ATTRIBUTION
    notes: str = ""
    #: Publisher plumbing. The City is one ArcGIS Online org serving
    #: FeatureServers; DRCOG is a self-hosted ArcGIS Server serving MapServers.
    #: Both answer the same /query, so nothing downstream needs to know which.
    root: str = SERVICE
    server: str = "FeatureServer"

    @property
    def service(self) -> str:
        return f"{self.root}/{self.path}/{self.server}"

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
              "Continuously edited, so it describes today, not 2020. Counted "
              "over the LoDo AOI: 3,890 of 4,886 rows (80%) are condominium "
              "records, so a parcel is not a building -- a tower is one "
              "footprint and hundreds of stacked parcels, and anything "
              "counting parcels to count buildings is off by two orders of "
              "magnitude here.",
    ),
    Layer(
        id="zoning",
        name="Zoning",
        path="ODC_ZONE_ZONING_A", layer=209,
        role="prior", epoch="current", geometry="polygon", independence=3,
        notes="Form-based zoning encodes expected massing and height directly, "
              "which is a stronger completion prior than land use -- except "
              "here it mostly does not. Counted over the LoDo AOI: "
              "HEIGHT_STORIES is null or zero on 34 of 55 districts (62%), "
              "including 9 of the 11 downtown ones. The field is real and the "
              "prior is unusable exactly where the buildings are tallest, so "
              "treat a present HEIGHT_STORIES as a bonus and never as the "
              "expected case. CCD_Zoning/27 is the same layer under another "
              "name (54 features, same schema); no reason to prefer it.",
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
              "and kerb recovery, which airborne returns describe poorly. "
              "Centrelines with no width field, so it fixes where a pavement "
              "runs and not how wide it is; DRCOG's 2024 sidewalk polygons "
              "are the layer that answers width.",
    ),
    Layer(
        id="street_centerlines",
        name="Street Centerlines",
        path="ODC_TRANS_STREET_L", layer=145,
        role="prior", epoch="current", geometry="polyline", independence=3,
        notes="Topology of the road graph. Where the returns lose a stretch of "
              "carriageway, the centreline says whether the road continued. "
              "Use VOLCLASS for road class, not FUNCLASS: over the LoDo AOI "
              "they flatly contradict each other, VOLCLASS calling 268 of 400 "
              "segments ARTERIAL while FUNCLASS calls 296 of them Local-Urban "
              "and files another 53 as 'Not in HUTF Inventory'. VOLCLASS is "
              "the one that matches a downtown grid.",
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
    Layer(
        id="survey_lots",
        name="Survey Lots",
        path="ODC_ENG_SRVLOTS_A", layer=46,
        role="prior", epoch="current", geometry="polygon", independence=3,
        notes="The lot grid inside each subdivision. Finer than parcels, which "
              "get merged and split by ownership -- lots keep the platted "
              "rhythm that decides how wide a building on this street can be.",
    ),
    Layer(
        id="parking",
        name="Parking",
        path="ODC_TRANS_PARKING_A", layer=138,
        role="prior", epoch="current", geometry="polygon", independence=3,
        notes="Parking areas as polygons. Directly useful: a parking polygon is "
              "a positive assertion that the ground there is open and paved, "
              "which is the one thing airborne returns cannot distinguish from "
              "a flat roof at grade or a demolished lot.",
    ),
    Layer(
        id="parking_lots",
        name="Parking Lots",
        path="ODC_TRANS_PARKINGLOTS_A", layer=139,
        role="prior", epoch="current", geometry="polygon", independence=3,
        notes="Surface lots specifically. LoDo is full of them and they are the "
              "commonest false positive for 'flat low building' -- having them "
              "enumerated stops the compiler inventing a one-storey box on "
              "every asphalt rectangle. TYPE separates Impervious from Gravel, "
              "which a theme can use. Interior rings are load-bearing: a lot "
              "wrapped around a building is a polygon with a hole, and reading "
              "only exterior rings paves the building.",
    ),
    Layer(
        id="parkland",
        name="DPR Parkland 2026",
        path="DPR_Parkland_2026", layer=0,
        role="prior", epoch="2026", geometry="polygon", independence=3,
        notes="Parks and Recreation's parkland boundaries. The vegetation prior: "
              "inside a park, tall returns are trees; outside one, on a footprint, "
              "they are a building. Surveyed 2026, so `manifest()` withholds it "
              "from a 2020 reconstruction on epoch grounds -- it is a "
              "generation-mode input, and parkland boundaries barely move. "
              "Carries PARK_TYPE, PARK_CLASS, GIS_ACRES and a FACILITIES list, "
              "which is a prop manifest in all but name.",
    ),
    Layer(
        id="playgrounds",
        name="Playgrounds",
        path="ODC_PARK_PLAYGROUNDS_A", layer=91,
        role="prior", epoch="current", geometry="polygon", independence=3,
        notes="Zero features over the LoDo AOI, confirmed by query rather than "
              "assumed. Catalogued anyway because the AOI is the variable here "
              "and a park-adjacent block picks them up -- a layer absent from "
              "one downtown crop is not a layer that does not exist.",
    ),
    Layer(
        id="athletic_fields",
        name="Athletic Fields",
        path="ODC_PARK_ATHLETICFIELDS_A", layer=82,
        role="prior", epoch="current", geometry="polygon", independence=3,
        notes="Zero over LoDo, same reasoning as playgrounds. Pitch and court "
              "outlines are flat ground that must not become building.",
    ),

    # ---- DRCOG regional planimetrics -------------------------------------
    # The 2024 regional stereocompilation, on DRCOG's own ArcGIS Server. Every
    # one of these is epoch 2024 and therefore withheld from a 2020
    # reconstruction automatically -- they are generation-mode inputs, and a
    # change-detection set against the 2020 scan.
    Layer(
        id="roofprints_2024",
        name="DRCOG Building Roofprints 2024",
        path="PLANIMETRICS_2024_BUILDING_ROOFPRINTS_TOTAL", layer=0,
        role="prior", epoch="2024", geometry="polygon", independence=2,
        root=DRCOG_ARCGIS, server="MapServer",
        license=DRCOG_TERMS, attribution=DRCOG_ATTRIBUTION,
        notes="Roofprints, not footprints, which is the distinction that "
              "matters here: a roofprint is the roof edge seen from above, "
              "and that is exactly the outline airborne returns actually "
              "describe. A ground footprint differs from it by the overhang, "
              "so matching returns to a footprint charges the compiler for a "
              "discrepancy the sensor could never have resolved. Carries "
              "Bldg_Height and Ground_Elevation, same schema family as the "
              "City's Building Outlines. NOT INDEPENDENT OF THEM: measured, "
              "not assumed -- 461 of 503 Building_IDs over the AOI are shared "
              "and the matched polygons have an area ratio of 1.0000. One "
              "stereocompilation, republished twice. Agreement between them "
              "corroborates nothing, and only one may be used to score. See "
              "PUBLISHER_OFFSET.",
    ),
    Layer(
        id="sidewalk_polygons_2024",
        name="DRCOG Sidewalk Polygons 2024",
        path="PLANIMETRICS_2024_POLYGON_SIDEWALKS_TOTAL", layer=0,
        role="prior", epoch="2024", geometry="polygon", independence=2,
        root=DRCOG_ARCGIS, server="MapServer",
        license=DRCOG_TERMS, attribution=DRCOG_ATTRIBUTION,
        notes="Sidewalks as polygons rather than centrelines, so the width is "
              "surveyed instead of assumed. The pavement/kerb/carriageway "
              "split is the part of a street a game world is actually walked "
              "on, and airborne returns describe it poorly.",
    ),
    Layer(
        id="edge_pavement_polygon_2024",
        name="DRCOG Edge of Pavement Polygons 2024",
        path="PLANIMETRICS_2024_EDGE_PAVEMENT_POLYGON_TOTAL", layer=0,
        role="prior", epoch="2024", geometry="polygon", independence=2,
        root=DRCOG_ARCGIS, server="MapServer",
        license=DRCOG_TERMS, attribution=DRCOG_ATTRIBUTION,
        notes="The carriageway as a filled surface, with Surface and Type. The "
              "strongest street geometry available: centrelines say a road "
              "exists, this says where its edges are.",
    ),
    Layer(
        id="edge_pavement_line_2024",
        name="DRCOG Edge of Pavement Lines 2024",
        path="PLANIMETRICS_2024_EDGE_PAVEMENT_LINE_TOTAL", layer=0,
        role="prior", epoch="2024", geometry="polyline", independence=2,
        root=DRCOG_ARCGIS, server="MapServer",
        license=DRCOG_TERMS, attribution=DRCOG_ATTRIBUTION,
        notes="The same edges as lines, carrying a Curb flag -- which is the "
              "one attribute that says whether the pavement steps up here or "
              "runs flat, and there is no way to read that off 4 pts/m2 from "
              "above.",
    ),
    Layer(
        id="paved_parking_2024",
        name="DRCOG Paved Parking 2024",
        path="PLANIMETRICS_2024_PAVED_PARKING_TOTAL", layer=0,
        role="prior", epoch="2024", geometry="polygon", independence=2,
        root=DRCOG_ARCGIS, server="MapServer",
        license=DRCOG_TERMS, attribution=DRCOG_ATTRIBUTION,
        notes="Surveyed rather than inferred parking surface. Same job as the "
              "City's parking layers and stereocompiled instead of "
              "administrative, so the two disagree in useful ways.",
    ),
    Layer(
        id="driveways_2024",
        name="DRCOG Driveway Polygons 2024",
        path="PLANIMETRICS_2024_POLYGON_DRIVEWAYS_TOTAL", layer=0,
        role="prior", epoch="2024", geometry="polygon", independence=2,
        root=DRCOG_ARCGIS, server="MapServer",
        license=DRCOG_TERMS, attribution=DRCOG_ATTRIBUTION,
        notes="Where a building meets the street surface. Thin in LoDo (5 over "
              "the AOI, it being a downtown grid) and central anywhere "
              "residential.",
    ),
    Layer(
        id="ramps_2024",
        name="DRCOG Ramps 2024",
        path="PLANIMETRICS_2024_RAMPS_TOTAL", layer=0,
        role="prior", epoch="2024", geometry="point", independence=2,
        root=DRCOG_ARCGIS, server="MapServer",
        license=DRCOG_TERMS, attribution=DRCOG_ATTRIBUTION,
        notes="Kerb ramps, stereocompiled. The regional counterpart to the "
              "City's 2022 curb ramps, and a crossing-geometry cue.",
    ),
    Layer(
        id="tip_polygons_2024_2027",
        name="DRCOG TIP Polygons 2024-2027",
        path="TIP_POLYGONS_2024_2027", layer=0,
        role="later_epoch", epoch="2024", geometry="polygon", independence=3,
        root=DRCOG_ARCGIS, server="MapServer",
        license=DRCOG_TERMS, attribution=DRCOG_ATTRIBUTION,
        notes="Funded transport projects for 2024-2027: not a survey of "
              "anything that existed when the scan was flown, but a record of "
              "what is about to change. Useful for explaining a disagreement "
              "between the 2020 returns and the 2024 planimetrics, and never "
              "evidence about 2020. One feature over the AOI.",
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


#: The City and DRCOG publish the same stereocompiled buildings in frames that
#: differ by a rigid 1.12 m. Measured over the LoDo AOI on 2026-08-17 by
#: matching the 461 shared Building_IDs: the offset's interquartile range is
#: 0.4 mm, so this is a datum realisation difference, not survey disagreement.
#:
#: It matters because it is comparable to the errors being chased. The height
#: agreement this repo quotes is a median 1.18 m; a 1.12 m horizontal shift is
#: the same size, and it lands on every footprint the compiler extrudes.
#:
#: Which frame the returns prefer is NOT settled. Scoring in-polygon returns by
#: whether they sit within 2 m of the stated roof gives 46.5% for the City's
#: outlines and 45.0% for DRCOG's, and a coarse shift sweep peaks at 47.1%
#: around (-0.5, +1.0) -- nudged towards DRCOG, but by less than the metric can
#: resolve. The 3DEP tiles are EPSG:6342, NAD83(2011); both vector layers were
#: requested as EPSG:26913, plain NAD83, and that is the likeliest source of
#: the shift. Do not "fix" it by applying this vector until a sharper test says
#: which end is wrong.
PUBLISHER_OFFSET = {
    "measured": "2026-08-17",
    "aoi": "denver_lodo",
    "matched_buildings": 461,
    "drcog_minus_denver_m": (-0.8662, +0.7126),
    "magnitude_m": 1.1217,
    "iqr_m": 0.0004,
    "lidar_prefers": "unresolved",
    "roof_hit_rate": {"denver_2022": 0.465, "drcog_2024": 0.450,
                      "best_shift": 0.471, "best_shift_dxdy": (-0.5, 1.0)},
    "suspected_cause": "vectors requested in EPSG:26913 (NAD83) against 3DEP "
                       "tiles in EPSG:6342 (NAD83(2011))",
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

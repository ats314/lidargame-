"""Vienna acquisition manifest. The testbed Denver should have been.

Denver was chosen for one good reason -- two organisations independently
described the same city in the same period -- and it ran out of road on the
things that actually block a game world. Airborne LiDAR never sees a facade, so
every window in a Denver build is invented; the city publishes no per-tree
geometry, so the tree segmentation has nothing to be scored against; and its
"two independent descriptions" turned out to be one stereocompilation
republished twice.

Vienna fixes all three, and the reason is that the city observed itself from
every direction rather than only from above:

    nadir aerial          roofs
    four-way oblique      all four elevations of every building
    360 street panorama   what a pedestrian sees, every ~3 m
    mobile LiDAR          the same street at +/- 2 cm
    airborne LiDAR/DSM    terrain and roofs

The consequence for this compiler is specific. Denver forced Tier 7 -- pure
procedural invention -- for every facade, because there was no evidence at all.
Vienna has evidence for facades, so a window can be *observed* rather than
generated, and the epistemic states stop being a polite fiction on the walls.

What makes it better still is the held-out truth. Vienna publishes its own
building-body model with roof and wall surfaces separated. Withhold it, rebuild
from sensors alone, and compare -- which is a real score, unlike looking at a
render and deciding the roof seems about right.

Verified against the live WFS on 2026-08-17; every layer below returned
features over the historic core.
"""
from __future__ import annotations

from dataclasses import dataclass

#: One WFS endpoint serves the whole open geodata catalogue -- 377 feature
#: types. Layer names are the `ogdwien:` prefixed ones.
WFS = "https://data.wien.gv.at/daten/geo"

#: Vienna publishes its open geodata under CC BY 4.0, which is a real licence
#: grant rather than Denver's liability disclaimer. Attribution is required and
#: commercial use is permitted.
VIENNA_TERMS = ("Stadt Wien, data.wien.gv.at, CC BY 4.0 -- explicit grant, "
                "commercial use permitted, attribution required")
VIENNA_ATTRIBUTION = "Stadt Wien – data.wien.gv.at"

#: Vienna's own local grid. The WFS reprojects on request, so this is what to
#: ask for rather than something to transform into afterwards.
CRS = "EPSG:31256"          # MGI / Austria GK East


@dataclass(frozen=True)
class Layer:
    id: str
    name: str
    typename: str               # WFS typeName, without the ogdwien: prefix
    role: str                   # see data/denver.py ROLES
    geometry: str
    independence: int = 3
    fields: tuple[str, ...] = ()
    license: str = VIENNA_TERMS
    attribution: str = VIENNA_ATTRIBUTION
    notes: str = ""

    @property
    def url(self) -> str:
        return f"{WFS}?service=WFS&version=1.1.0&typeName=ogdwien:{self.typename}"


LAYERS: dict[str, Layer] = {layer.id: layer for layer in [
    Layer(
        id="building_bodies",
        name="Baukörpermodell",
        typename="FMZKBKMOGD", role="hidden_truth", geometry="polygon",
        independence=2,
        fields=("FMZK_ID", "F_KLASSE", "H_KLASSE", "O_KOTE", "U_KOTE"),
        notes="The city's own building-body model, carrying upper and lower "
              "elevation per body. This is the yardstick: withhold it, rebuild "
              "from sensors, and the comparison is a score rather than an "
              "opinion about a render. O_KOTE and U_KOTE give eave and ground "
              "directly, which in Denver had to be inferred from returns.",
    ),
    Layer(
        id="footprints",
        name="FMZK Gebäude",
        typename="FMZKGEBOGD", role="prior", geometry="polygon", independence=3,
        fields=("FMZK_ID", "F_KLASSE", "BW_GEB_ID", "BW_BRK_ID"),
        notes="Building footprints from the Mehrzweckkarte, the city's precise "
              "vector base map. Carries a stable building id, which is the "
              "conflation key Denver never had -- two Denver publishers shared "
              "461 of 503 ids by luck of lineage rather than by design.",
    ),
    Layer(
        id="typology",
        name="Bautypologien",
        typename="GEBAEUDETYPOGD", role="prior", geometry="polygon",
        independence=3,
        fields=("BAUTYP", "BAUTYP_TXT", "OBJ_STR", "OBJ_STR_TXT"),
        notes="Architectural typology per building -- Gründerzeit, interwar "
              "municipal, postwar, and so on. This is the single most valuable "
              "layer here for generation: it is the architectural family the "
              "World Seed has no field for, and its absence is why every "
              "generated Denver building came back the same brick box. A "
              "typology drives storey height, facade rhythm, roof form and "
              "material palette at once.",
    ),
    Layer(
        id="roof_cadastre",
        name="Dachbodenkataster",
        typename="DACHKATASTEROBJEKTEOGD", role="prior", geometry="point",
        independence=3,
        fields=("ADR_ID", "ADRESSE", "DACHTYP", "BEZIRK"),
        notes="Roof type per address. Denver's seed recorded `flat` for all 257 "
              "buildings because roof form was measured and then discarded, and "
              "generating pitched roofs from a fitted slope produced plates "
              "larger than the buildings. A declared roof type sidesteps the "
              "fit entirely.",
    ),
    Layer(
        id="tree_cadastre",
        name="Baumkataster",
        typename="BAUMKATOGD", role="hidden_truth", geometry="point",
        independence=3,
        fields=("BAUM_ID", "GATTUNG_ART", "PFLANZJAHR", "BEZIRK"),
        notes="Every street tree, individually, with species and planting year. "
              "This closes the oldest open question in the repo: Denver "
              "publishes no per-tree geometry, so the tree segmentation had "
              "nothing to be scored against and its count was only ever "
              "'probably still high'. Withheld, this is instance-level "
              "vegetation truth -- position, count and, via species and age, a "
              "plausible crown size.",
    ),
    Layer(
        id="traffic_surfaces",
        name="FMZK Verkehr",
        typename="FMZKVERKEHR1OGD", role="prior", geometry="polygon",
        independence=3,
        fields=("FMZK_ID", "F_KLASSE", "KLASSE_SUB"),
        notes="Carriageway, pavement and tram surfaces as polygons rather than "
              "centrelines with an assumed width. F_KLASSE separates the "
              "surface types, which is what a kerb needs in order to be a kerb "
              "instead of 42% of the street.",
    ),
    Layer(
        id="green_surfaces",
        name="FMZK Grün",
        typename="FMZKGRUEN2OGD", role="prior", geometry="polygon",
        independence=3,
        notes="Surveyed green surface. FMZKGRUEN1OGD returned a null geometry "
              "on probe and FMZKGRUEN2OGD is the one to use; recorded so the "
              "next person does not repeat that.",
    ),
    Layer(
        id="building_info",
        name="Gebäudeinformationen",
        typename="GEBAEUDEINFOOGD", role="prior", geometry="point",
        independence=3,
        fields=("ACD", "STRCD", "STRNAML", "VONN", "BISN"),
        notes="Address points with street name and house number ranges. The "
              "join key between typology, roof cadastre and footprint.",
    ),
]}


#: The mobile mapping campaign, and the reason to move here at all.
#:
#: Kappazunder 2020 covers 4,600 km of street with 250 MP 360 panoramas about
#: every 3 m, georeferenced to ~10 cm, alongside mobile LiDAR at ~2 cm out to
#: 50 m -- and it ships the camera's inner and outer orientation. That last
#: item is what makes true PointPainting possible: the projection
#:
#:     (u, v) = K [R | t] p
#:
#: is computable rather than approximated. Every facade claim in a Denver build
#: is Tier 7 invention because nothing observed a wall; here a wall is observed.
#:
#: A predefined test area is published under CC BY 4.0, which is the right
#: place to start rather than requesting the city.
#: Confirmed from the product page on 2026-08-17, not inferred.
#:
#: CORRECTED 2026-08-17. An earlier note here said Kappazunder could only be
#: obtained by submitting a request form, and recorded that as a blocker. That
#: was read off the product page and was wrong. The test datasets are published
#: on data.gv.at under CC BY 4.0 with direct download URLs and no form at all --
#: see TEST_DATASETS below. The form route exists for arbitrary areas of the
#: city; it is not the only route, and it never blocked getting started.
KAPPAZUNDER_ORDER = {
    "product_page": ("https://www.wien.gv.at/stadtplanung/"
                     "mobile-mapping-befahrungsdaten-produktinformation"),
    "interface_spec_en": "https://www.wien.gv.at/pdf/ma41/datainterface-kappazunder-en.pdf",
    "route": "arbitrary areas: request form on wien.gv.at formularserver. "
             "Test datasets: direct download, no form (see TEST_DATASETS).",
    "acquired": "driven from May 2020",
    "coverage": "the whole Vienna street network, plus the Donauinsel, "
                "selected parks and the urban motorways",
    "panoramas": "360 degree, delivered as cubemap",
    "point_clouds": "LAZ; one trajectory may span several files",
    "point_attributes": "intensity and RGB, first echoes only",
    "metadata": "ASCII, carrying inner and outer camera orientation and "
                "timestamps per image and per cloud",
}

KAPPAZUNDER = {
    "name": "Kappazunder 2020 mobile mapping",
    "coverage_km": 4600,
    "panorama": "250 MP 360 degree, roughly every 3 m",
    "geolocation": "95% within 10 cm",
    "lidar": "mobile, ~2 cm accuracy to 50 m",
    "ships": ["JPG panoramas", "LAZ point clouds", "vehicle trajectory",
              "inner and outer camera orientation"],
    "test_dataset_gb": 5.5,
    "license": "CC BY 4.0",
    "why": "Camera pose is published, so image-to-LiDAR correspondence is "
           "computed rather than assumed. This is the capability Denver never "
           "had and the reason facades there are wholly invented.",
}

#: Calibrated oblique aerial imagery, flown 2020 and 2023. Each station takes
#: one nadir plus four 45 degree views facing N/E/S/W, so every building is
#: seen on all four elevations from above as well as from the street.
OBLIQUE = {
    "epochs": ("2020", "2023"),
    "resolution": "150 MP per image, ~8 cm GSD in the central region (2023)",
    "images_2023": 36000,
    "geometry": "1 nadir + 4 oblique at 45 degrees, N/E/S/W",
    "license": "CC BY 4.0",
    "why": "Two epochs three years apart is also a change-detection corpus, "
           "which is what the temporal conflation question needs and what the "
           "Denver 2020-vs-2024 disagreement could never settle.",
}


def layers_for(role: str) -> list[Layer]:
    return [layer for layer in LAYERS.values() if layer.role == role]


def withheld() -> list[Layer]:
    """What must not reach the compiler in a scored run.

    The building bodies and the tree cadastre are the two things worth scoring
    against, and both are exactly the kind of layer it is tempting to feed in
    because it would make the output better.
    """
    return [layer for layer in LAYERS.values() if layer.role == "hidden_truth"]


#: Directly downloadable, CC BY 4.0, no request form. Found via the data.gv.at
#: DCAT record ed24cfff-1361-48d5-a071-31e4c697b844, which carries the URLs the
#: product page does not.
#:
#: This is the 2023 epoch rather than the 2020 one the handoff prefers. 2020 is
#: the temporally coherent stack -- 2020 mobile, 2020 obliques, 2020 ortho --
#: and remains the right target for a scored run. But 2023 is the same
#: interface, it is here now, and Phase 0 is an intake audit: reading the
#: structure does not need the epoch to match anything.
TEST_BASE = "https://www.wien.gv.at/ma41datenviewer/downloads/Wien/Testdaten"
TEST_DATASETS = {
    "info": {"file": "01-Info_2023.zip", "bytes": 3_558_148,
             "note": "documentation; read before assuming the archive layout"},
    "gis": {"file": "02-GIS-Daten_2023.zip", "bytes": 39_600,
            "note": "the AOI's vector context"},
    "kappazunder": {"file": "03_Kappazunder_Testdatensatz_2023.zip",
                    "note": "the mobile mapping itself -- LAZ, panoramas, "
                            "trajectory, and the orientation metadata that "
                            "makes projection arithmetic rather than a fit"},
}
TEST_LICENSE = "CC BY 4.0"


def test_dataset_urls() -> dict[str, str]:
    return {key: f"{TEST_BASE}/{spec['file']}"
            for key, spec in TEST_DATASETS.items()}

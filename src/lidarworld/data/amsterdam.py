"""Amsterdam acquisition manifest: dense returns, free terms, an independent check.

Denver was the first city and it is the worst one available. 3DEP gives ~4
points per square metre from above, two thirds of them on pavement, so every
facade is invented; the city's terms are a liability disclaimer with no
copyright grant; and its one independent height check turned out to be a single
stereocompilation republished twice.

Amsterdam is better on all three at once, and none of it required a special
arrangement -- the Dutch national geo stack is open by default:

    AHN5 point cloud      23 pts/m2, CC0, 1 km tiles      data/ahn.py
    3D BAG                per-building height + roof type, CC BY 4.0
    BAG                   footprint + construction year, public task data
    BGT vegetatieobject   *every street tree as a point*, CC BY 4.0
    NWB wegvakken         street centrelines in RD, CC BY 4.0

Two of those change what the compiler can claim.

**3D BAG is a level-2 independent check.** It states a roof height per building,
derived from AHN by the TU Delft 3D geoinformation group with their own
pipeline. Comparing a reconstructed height against it is not self-consistency:
different code, different failure modes. Denver's aerial-stereo comparison is
the only check of that kind in the repo, and this is a second one -- with the
caveat that 3D BAG's input is the same AHN flight, so it is independent in
*method* and not in *sensor*, which is a weaker claim than Denver's and must be
recorded as such.

**BGT publishes the trees.** The tree segmentation has never been scored: the
Denver block produced 1197 "trees" and then 307 after three real fixes, and
nobody could say which was right because there is no airborne ground truth in
the repo. The BGT carries individual vegetation objects as points, and Amsterdam
additionally publishes its own street-tree register with species and trunk
diameter. That turns known weakness 6 from an opinion into a number.

What Amsterdam does *not* fix: airborne LiDAR still never sees a facade. A
canal-house front at 23 pts/m2 is sparse rather than absent, which is a real
improvement over Denver's nothing, but the elevation still has to be generated
(`docs/GENERATED_FACADES.md`). Vienna remains the city with oblique imagery.

Verified against the live services on 2026-08-24: the 3D BAG WFS returned
LoD1.2 features with `b3_h_70p` and `b3_dak_type` over the canal belt, the BGT
OGC API listed `vegetatieobject_punt`, NWB returned named wegvakken in RD, and
`AHN5_T/25GN1_02.LAZ` range-read as 29,219,688 returns over 1000 x 1250 m.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import ahn

CRS = ahn.CRS                       # EPSG:28992, Amersfoort / RD New

#: OGC API Features over the Basisregistratie Grootschalige Topografie: the
#: large-scale base map, 49 collections, everything on the ground plane.
BGT = "https://api.pdok.nl/lv/bgt/ogc/v1"

#: Roles, as in data/denver.py: what a layer is allowed to be used *as*.
#:   sensor       measured returns
#:   prior        authoritative geometry the compiler may lean on
#:   hidden_truth withheld from the build, used only to score it
ROLES = ("sensor", "prior", "hidden_truth")


@dataclass(frozen=True)
class Layer:
    id: str
    name: str
    url: str
    role: str
    geometry: str
    license: str
    attribution: str
    #: 1 = same sensor and same pipeline, 3 = wholly independent survey.
    independence: int = 3
    fields: tuple[str, ...] = ()
    notes: str = ""


LAYERS: dict[str, Layer] = {layer.id: layer for layer in [
    Layer(
        id="ahn5",
        name="AHN5 point cloud (2023-2025)",
        url=f"{ahn.BASE}/{ahn.VERSIONS['ahn5']}/",
        role="sensor", geometry="pointcloud", independence=3,
        license=ahn.TERMS, attribution=ahn.ATTRIBUTION,
        notes="23 points per square metre over the canal belt against Denver's "
              "4. Classified ground/building/vegetation/water at source, and "
              "carrying return number -- which is *the* canopy discriminator "
              "and was being discarded at ingest until it was found to be the "
              "cause of 1197 phantom Denver trees.",
    ),
    Layer(
        id="ahn4",
        name="AHN4 point cloud (2020-2022)",
        url=f"{ahn.BASE}/{ahn.VERSIONS['ahn4']}/",
        role="sensor", geometry="pointcloud", independence=3,
        license=ahn.TERMS, attribution=ahn.ATTRIBUTION,
        notes="The previous flight. Two epochs of the same city three years "
              "apart is a change-detection pair, and more usefully a way to "
              "ask which parts of a reconstruction are stable under a "
              "different survey of the same buildings.",
    ),
    Layer(
        id="footprints_3dbag",
        name="3D BAG LoD1.2",
        url="https://data.3dbag.nl/api/BAG3D/wfs",
        role="prior", geometry="polygon", independence=2,
        license="CC BY 4.0 -- TU Delft 3D geoinformation group",
        attribution="3D BAG (TU Delft), from BAG and AHN",
        fields=("identificatie", "b3_h_70p", "b3_h_50p", "b3_h_max", "b3_h_nok",
                "b3_h_maaiveld", "b3_dak_type", "b3_nodata_fractie_ahn5"),
        notes="Footprint, ground level, roof height percentiles and a declared "
              "roof type per building. Independence 2, not 3: a different team "
              "and a different pipeline, but the same AHN returns underneath, "
              "so a shared sensor artefact would not show up as disagreement.",
    ),
    Layer(
        id="footprints_bgt",
        name="BGT pand",
        url=f"{BGT}/collections/pand/items",
        role="prior", geometry="polygon", independence=3,
        license="CC BY 4.0 -- Kadaster / PDOK",
        attribution="BGT, Kadaster (CC BY 4.0)",
        notes="The surveyed outline rather than the registered one. Where BGT "
              "and BAG disagree about a wall, BGT is the one measured on the "
              "ground.",
    ),
    Layer(
        id="trees",
        name="BGT vegetatieobject (punt)",
        url=f"{BGT}/collections/vegetatieobject_punt/items",
        role="hidden_truth", geometry="point", independence=3,
        license="CC BY 4.0 -- Kadaster / PDOK",
        attribution="BGT, Kadaster (CC BY 4.0)",
        notes="One point per tree, surveyed. This is the layer that ends the "
              "guessing on known weakness 6: withhold it, run the canopy "
              "segmentation, and count. Hold it as hidden truth -- feeding it "
              "in would make the tree count a copy rather than a measurement.",
    ),
    Layer(
        id="roads",
        name="NWB wegvakken",
        url="https://service.pdok.nl/rws/nwbwegen/wfs/v1_0",
        role="prior", geometry="polyline", independence=3,
        license="CC BY 4.0 -- Rijkswaterstaat",
        attribution="Nationaal Wegenbestand, Rijkswaterstaat (CC BY 4.0)",
        fields=("sttNaam", "wegbehsrt", "bstCode", "gmeNaam"),
        notes="Centrelines with street names. No width and no functional "
              "class, so widths come from the maintaining authority and a "
              "special-carriageway code -- see topology/streets.py.",
    ),
    Layer(
        id="road_surface",
        name="BGT wegdeel",
        url=f"{BGT}/collections/wegdeel/items",
        role="prior", geometry="polygon", independence=3,
        license="CC BY 4.0 -- Kadaster / PDOK",
        attribution="BGT, Kadaster (CC BY 4.0)",
        notes="The carriageway as a surveyed polygon with its surface material, "
              "which is strictly better evidence than a centreline plus an "
              "assumed width. Not wired: the street stage rasterises lines, and "
              "swapping it for a polygon clipper is its own change.",
    ),
    Layer(
        id="water",
        name="BGT waterdeel",
        url=f"{BGT}/collections/waterdeel/items",
        role="prior", geometry="polygon", independence=3,
        license="CC BY 4.0 -- Kadaster / PDOK",
        attribution="BGT, Kadaster (CC BY 4.0)",
        notes="The canals, which is most of what makes Amsterdam Amsterdam. "
              "LiDAR over water is a hole -- the pulse leaves and does not come "
              "back -- so the returns describe a city with gaps in it and this "
              "layer says the gaps are canals. Not wired yet; the World Seed "
              "has no water field.",
    ),
]}


def bgt_items_url(collection: str, bbox_rd, *, limit: int = 1000) -> str:
    """OGC API Features query for one BGT collection over an RD bbox."""
    import urllib.parse

    query = urllib.parse.urlencode({
        "bbox": ",".join(str(round(float(v), 2)) for v in bbox_rd),
        "bbox-crs": "http://www.opengis.net/def/crs/EPSG/0/28992",
        "crs": "http://www.opengis.net/def/crs/EPSG/0/28992",
        "limit": limit,
        "f": "json",
    })
    return f"{BGT}/collections/{collection}/items?{query}"

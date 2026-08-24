"""Amsterdam wiring: the AHN tile grid, and the unit trap under a footprint.

Everything here is offline. The grid maths was derived from live LAS headers
(see data/ahn.py) and the numbers below are those measured extents, so a change
that quietly renames or reshapes a tile fails here rather than three hundred
megabytes into a download.
"""
import pytest

from lidarworld.data import ahn, amsterdam
from lidarworld.data.catalog import PLACES, describe
from lidarworld.data.gis import FOOTPRINTS, STREETS, attributes


def test_tile_at_matches_the_measured_header():
    # 25GN1_02 holds the canal belt: header bounds 120980..122020 x
    # 486230..487520, which is the nominal cell plus a 20 m overlap.
    tile = ahn.tile_at(121500, 486800)
    assert tile.id == "25GN1_02"
    assert (tile.west, tile.south, tile.east, tile.north) == (
        121000.0, 486250.0, 122000.0, 487500.0)
    assert tile.url.endswith("/AHN5_T/25GN1_02.LAZ")


def test_a_coordinate_on_the_sheet_edge_stays_inside_the_sheet():
    # A crop written in round metres lands on a sheet edge often, and the
    # unclamped division named tile 27 of a sheet that has 25.
    tile = ahn.tile_at(121300, 487500)
    assert tile.index <= ahn.COLS * ahn.ROWS
    assert tile.id == "25EZ1_22"


def test_version_selects_the_directory_not_the_grid():
    a5 = ahn.tile_at(121500, 486800, version="ahn5")
    a4 = ahn.tile_at(121500, 486800, version="ahn4")
    assert a4.id == a5.id
    assert "/AHN4_T/" in a4.url and "/AHN5_T/" in a5.url


def test_an_area_can_span_sheets():
    ids = [t.id for t in ahn.tiles_for((121300, 487000, 122300, 487900))]
    assert "25GN1_02" in ids          # south of the sheet boundary
    assert "25EZ1_22" in ids          # north of it
    assert len(set(ids)) == len(ids)


def test_a_coordinate_outside_the_verified_sheets_is_refused():
    # Better an error naming the gap than a confidently wrong URL.
    with pytest.raises(KeyError):
        ahn.tile_at(50000, 400000)


def test_places_resolve_to_tiles_without_the_network():
    from lidarworld.data.fetch import resolve_place_tiles

    for place_id in ("amsterdam_grachtengordel", "amsterdam_centraal"):
        place = PLACES[place_id]
        assert place["crs"] == "EPSG:28992"
        tiles = resolve_place_tiles(place, place_id)
        assert tiles and all(t["url"].endswith(".LAZ") for t in tiles)


def test_the_sources_are_catalogued_with_their_terms():
    assert describe("ahn_geotiles").commercial
    assert "CC0" in describe("ahn_geotiles").license
    for source_id in ("bag3d", "bgt", "nwb"):
        assert describe(source_id).attribution


def test_metres_are_not_read_as_feet():
    """The unit trap: 3D BAG states an absolute NAP elevation in metres and
    Denver states a height above ground in feet. Reading one as the other made
    a 15 m canal house 4.6 m tall."""
    geojson = {"features": [{"geometry": None, "properties": {
        "identificatie": "NL.IMBAG.Pand.0363100012175601",
        "b3_h_70p": 16.84, "b3_h_maaiveld": 0.47, "b3_dak_type": "slanted"}}]}
    record = attributes(geojson, FOOTPRINTS["amsterdam"])[0]
    assert record["height"] == pytest.approx(16.37, abs=0.01)
    assert record["ground"] == pytest.approx(0.47)
    assert record["source_id"].startswith("NL.IMBAG")
    assert record["b3_dak_type"] == "slanted"

    denver = {"features": [{"geometry": None, "properties": {
        "BLDG_HEIGH": 50, "GROUND_ELE": 5280, "BUILDING_I": 7}}]}
    assert attributes(denver, FOOTPRINTS["denver"])[0]["height"] == pytest.approx(15.24)


def test_a_footprint_with_no_ground_level_states_no_height():
    """An absolute elevation with nothing to subtract is not a height, and
    guessing zero would put the building on the NAP datum."""
    geojson = {"features": [{"geometry": None, "properties": {"b3_h_70p": 16.8}}]}
    assert attributes(geojson, FOOTPRINTS["amsterdam"])[0]["height"] is None


def test_street_widths_come_from_the_nwb_authority_code():
    from lidarworld.topology import streets

    network = {"features": [
        {"geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 1]]},
         "properties": {"wegbehsrt": "G", "sttNaam": "Herengracht"}},
        {"geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 1]]},
         "properties": {"wegbehsrt": "R", "sttNaam": "A10"}},
        {"geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 1]]},
         "properties": {"wegbehsrt": "G", "bstCode": "FP"}},
    ]}
    half = streets.widths(network)
    assert half[0] == 5.0                    # a municipal street
    assert half[1] > half[0]                 # a motorway is wider
    assert half[2] < half[0]                 # a cycle path is not a street


def test_the_manifest_keeps_the_trees_out_of_the_build():
    """The tree layer is the score, so feeding it in would make the tree count
    a copy of the answer rather than a measurement."""
    assert amsterdam.LAYERS["trees"].role == "hidden_truth"
    assert amsterdam.LAYERS["ahn5"].role == "sensor"
    for layer in amsterdam.LAYERS.values():
        assert layer.role in amsterdam.ROLES
        assert layer.license and layer.attribution
        assert layer.url.startswith("https://")


def test_the_height_check_is_not_claimed_as_fully_independent():
    """3D BAG derives its heights from the same AHN returns, so it is
    independent in method and not in sensor. Recording that as level 2 is the
    difference between a check and a coincidence."""
    assert amsterdam.LAYERS["footprints_3dbag"].independence == 2
    assert amsterdam.LAYERS["trees"].independence == 3


def test_the_amsterdam_layers_are_addressable():
    assert FOOTPRINTS["amsterdam"].kind == "wfs"
    assert FOOTPRINTS["amsterdam"].default_crs == "28992"
    assert STREETS["amsterdam"].kind == "wfs"
    url = amsterdam.bgt_items_url("vegetatieobject_punt", (121300, 486500, 121700, 486900))
    assert "bbox-crs" in url and "28992" in url

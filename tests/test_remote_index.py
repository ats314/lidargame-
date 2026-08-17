"""Remote tile index.

The local index reads LAS headers, which is exact and only works for files you
already have. To decide what to *download* you need the extents first, which is
what the published per-tile GeoJSON gives. These tests use a fixture rather than
the network, so they say nothing about whether USGS is up -- only that an area
of interest resolves to the right download URLs.
"""
from __future__ import annotations

import json

import pytest

from lidarworld.data.catalog_index import ACQUISITIONS, RemoteIndex, RemoteTile


def tile_feature(ident, west, south, east, north, block="CO_DRCOG_2_2020"):
    return {
        "type": "Feature",
        "properties": {
            "description_id": ident,
            "url": ("https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/"
                    f"LPC/Projects/CO_DRCOG_2020_B20/{block}/LAZ/"
                    f"USGS_LPC_CO_DRCOG_2020_B20_{ident}.laz"),
            "temporal": {"startTime": "20200526", "endTime": "20200612"},
        },
        "geometry": {"type": "Polygon", "coordinates": [[
            [west, north], [east, north], [east, south], [west, south], [west, north]]]},
    }


@pytest.fixture
def index(tmp_path):
    """A 2x2 grid of tiles, cached so no network call is made."""
    features = [
        tile_feature("w0499n4398", -105.01, 39.73, -105.00, 39.74),
        tile_feature("w0499n4399", -105.01, 39.74, -105.00, 39.75),
        tile_feature("w0501n4398", -105.00, 39.73, -104.99, 39.74),
        tile_feature("w0501n4399", -105.00, 39.74, -104.99, 39.75),
    ]
    cache = tmp_path / "extents_co_drcog_2020_b2.json"
    cache.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return RemoteIndex.load("co_drcog_2020_b2", cache_dir=tmp_path)


def test_the_published_acquisitions_are_addressable():
    assert set(ACQUISITIONS) == {"co_drcog_2020_b1", "co_drcog_2020_b2", "co_drcog_2020_b3"}
    for entry in ACQUISITIONS.values():
        assert entry["extents"].startswith("https://")
        assert entry["tiles"] > 0
    # Denver is block 2, and that is recorded rather than folklore.
    assert "Denver" in ACQUISITIONS["co_drcog_2020_b2"].get("notes", "")


def test_an_index_carries_urls_and_flight_dates(index):
    assert len(index) == 4
    tile = index.tiles[0]
    assert tile.url.endswith(".laz")
    assert "/LPC/Projects/" in tile.url, "the LAZ lives under LPC, not OPR"
    assert tile.name.startswith("USGS_LPC_")
    assert tile.start == "20200526" and tile.end == "20200612"
    assert index.summary()["tiles"] == 4


def test_query_selects_only_the_tiles_an_area_touches(index):
    # A small area wholly inside one tile.
    inside = index.query((-105.008, 39.735, -105.004, 39.738))
    assert [t.id for t in inside] == ["w0499n4398"]

    # An area straddling the shared corner touches all four.
    corner = index.query((-105.002, 39.738, -104.998, 39.742))
    assert len(corner) == 4

    assert index.query((-100.0, 30.0, -99.0, 31.0)) == []


def test_around_is_a_centred_box(index):
    hits = index.around(-105.005, 39.735, 0.004)
    assert [t.id for t in hits] == ["w0499n4398"]


def test_intersects_is_inclusive_of_a_shared_edge():
    tile = RemoteTile("t", "http://x/t.laz", -105.0, 39.0, -104.0, 40.0, "a")
    assert tile.intersects((-104.0, 39.5, -103.0, 39.6))    # touching edge
    assert tile.intersects((-106.0, 38.0, -103.0, 41.0))    # containing
    assert not tile.intersects((-103.9, 39.5, -103.0, 39.6))


def test_an_unknown_acquisition_names_the_ones_that_exist():
    with pytest.raises(KeyError) as excinfo:
        RemoteIndex.load("co_drcog_2020_b9")
    assert "co_drcog_2020_b2" in str(excinfo.value)


def test_features_without_a_url_are_skipped(tmp_path):
    """A malformed row must not become a tile that cannot be downloaded."""
    features = [
        tile_feature("good", -105.01, 39.73, -105.00, 39.74),
        {"type": "Feature", "properties": {"description_id": "nourl"},
         "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}},
        {"type": "Feature", "properties": {"url": "http://x/y.laz"}, "geometry": None},
    ]
    cache = tmp_path / "extents_co_drcog_2020_b2.json"
    cache.write_text(json.dumps({"features": features}))
    index = RemoteIndex.load("co_drcog_2020_b2", cache_dir=tmp_path)
    assert len(index) == 1
    assert index.tiles[0].id == "good"


def test_an_empty_index_summarises_without_dividing_by_zero(tmp_path):
    cache = tmp_path / "extents_co_drcog_2020_b2.json"
    cache.write_text(json.dumps({"features": []}))
    index = RemoteIndex.load("co_drcog_2020_b2", cache_dir=tmp_path)
    assert len(index) == 0
    assert index.summary() == {"tiles": 0, "acquisitions": []}
    assert index.query((-105.0, 39.0, -104.0, 40.0)) == []

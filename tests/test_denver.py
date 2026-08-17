"""The Denver manifest, and the one invariant it exists to enforce.

Point clouds and polygons mix freely; that is the whole design. What must not
happen is a layer being fed to the compiler *and* used to score it, because the
resulting number measures nothing.
"""
from __future__ import annotations

import pytest

from lidarworld.data import denver

LODO = (-105.002, 39.740, -104.985, 39.755)


def test_every_layer_is_addressable_and_has_a_declared_role():
    assert denver.LAYERS
    for layer in denver.LAYERS.values():
        assert layer.role in denver.ROLES, f"{layer.id} has role {layer.role!r}"
        assert layer.url.startswith("https://services1.arcgis.com/")
        assert layer.url.endswith(f"/{layer.layer}")
        assert layer.geometry in ("polygon", "polyline", "point")
        assert layer.license and layer.attribution
        assert layer.notes, f"{layer.id} does not say what it is for"


def test_the_terms_are_recorded_as_written_not_as_hoped():
    """Denver publishes a disclaimer, not a CC licence. Do not upgrade it."""
    assert "no explicit copyright grant" in denver.DENVER_TERMS
    assert "NOT FOR ENGINEERING PURPOSES" in denver.DENVER_TERMS


def test_reconstruction_withholds_truth_and_admits_priors():
    m = denver.manifest(LODO, epoch="2020")
    admitted = {l["id"] for l in m["layers"]}
    withheld = {l["id"] for l in m["withheld"]}

    assert "building_outlines" in withheld, "footprints must not score their own output"
    assert "sidewalks_2020" in withheld
    assert admitted & withheld == set()
    assert admitted | withheld == set(denver.LAYERS)
    for entry in m["withheld"]:
        assert entry["reason"]
    assert m["lidar"]["project"] == "CO_DRCOG_2020_B20"
    assert m["lidar"]["role"] == "input"


def test_no_admitted_layer_postdates_the_observation():
    m = denver.manifest(LODO, epoch="2013")
    for entry in m["layers"]:
        if entry["epoch"].isdigit():
            assert int(entry["epoch"]) <= 2013, f"{entry['id']} is from the future"
    # The 2020 land-use layer is admissible against 2020 and not against 2013.
    ids_2013 = {l["id"] for l in m["layers"]}
    ids_2020 = {l["id"] for l in denver.manifest(LODO, epoch="2020")["layers"]}
    assert "landuse_2020" in ids_2020
    assert "landuse_2020" not in ids_2013


def test_generation_mode_admits_everything():
    """Mixing is the point; the split is only about scoring."""
    m = denver.manifest(LODO, mode="generation")
    assert len(m["layers"]) == len(denver.LAYERS)
    assert m["withheld"] == []


def test_tree_canopy_is_not_claimed_as_per_tree_truth():
    """Measured live: 7 rows of neighbourhood statistics, not canopy polygons."""
    canopy = denver.LAYERS["tree_canopy_2020"]
    assert canopy.role == "prior"
    assert "aggregate" in canopy.notes


def test_every_lidar_epoch_over_denver_is_airborne():
    assert set(denver.LIDAR_EPOCHS) == {"2011", "2013", "2020"}
    for epoch in denver.LIDAR_EPOCHS.values():
        assert epoch["source"] == "usgs_3dep"
        assert epoch["crs"] == "EPSG:26913"


def test_bad_arguments_are_rejected():
    with pytest.raises(ValueError, match="mode must be"):
        denver.manifest(LODO, mode="whatever")
    with pytest.raises(ValueError, match="no Denver LiDAR epoch"):
        denver.manifest(LODO, epoch="1999")
    with pytest.raises(ValueError, match="unknown role"):
        denver.layers_for("truthy")


def test_query_url_asks_for_the_point_clouds_crs():
    url = denver.query_url(denver.LAYERS["parcels"], LODO)
    assert "outSR=26913" in url, "polygons must arrive in the LiDAR's frame"
    assert "f=geojson" in url
    assert "inSR=4326" in url
    assert "-105.002" in url


def test_independence_is_recorded_for_every_layer():
    """A layer can be withheld and still be worthless as truth."""
    for layer in denver.LAYERS.values():
        assert layer.independence in denver.INDEPENDENCE, layer.id
        # Nothing catalogued here is a derivative of the scan it would check.
        assert layer.independence >= 2, (
            f"{layer.id} is level {layer.independence}: same-sensor derivatives "
            "are constraints, never validation")


def test_validators_exclude_same_sensor_derivatives():
    """The threshold is the assertion; the roster is not.

    This used to spell out all seven level-3 ids, so adding a layer failed here
    rather than anywhere meaningful. What matters is that `validators(n)` is
    exactly the layers at or above n -- and that the imagery-derived layers,
    which are the ones it would be tempting to score against, are the ones a
    level-3 request drops.
    """
    for level in (2, 3, 4):
        assert {l.id for l in denver.validators(level)} == {
            l.id for l in denver.LAYERS.values() if l.independence >= level}

    stereocompiled = {"building_outlines", "sidewalks_2020", "landuse_2020",
                      "tree_canopy_2020"}
    level3 = {l.id for l in denver.validators(3)}
    assert not (level3 & stereocompiled), (
        "level 3 is the external-record tier; imagery-derived layers are level 2")
    assert stereocompiled <= {l.id for l in denver.validators(2)}

    assert len(denver.validators(2)) == len(denver.LAYERS)
    assert denver.validators(4) == []


def test_every_layer_resolves_to_a_distinct_endpoint():
    """A copy-pasted path or layer id silently fetches the wrong thing."""
    urls = [l.url for l in denver.LAYERS.values()]
    assert len(set(urls)) == len(urls), "two layers point at one endpoint"
    for layer in denver.LAYERS.values():
        assert layer.geometry in ("polygon", "polyline", "point"), layer.id


def test_the_boulder_dead_end_is_written_down():
    """Probed live and absent. Recording it stops the next hunt."""
    probe = denver.BOULDER_PROBE
    assert any("segmented trees" in miss for miss in probe["not_found"])
    assert any("tree inventory" in miss for miss in probe["not_found"])
    assert probe["dead_hosts"]
    assert "AP2020Cached3inWM" in " ".join(probe["found"])
    # The contours are present, and must not be mistaken for validation.
    assert "Level 1" in probe["found"]["general/Contours"]


def test_roles_partition_the_catalogue():
    counted = sum(len(denver.layers_for(role)) for role in denver.ROLES)
    assert counted == len(denver.LAYERS)
    assert {l.id for l in denver.layers_for("hidden_truth")} == {
        "building_outlines", "sidewalks_2020"}

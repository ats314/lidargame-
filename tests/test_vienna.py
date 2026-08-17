"""The Vienna manifest, and the reasons it replaces Denver.

These are cheap assertions about a table, but the table encodes the decision to
move testbed, and the reasons should not quietly rot out of it.
"""
from lidarworld.data import vienna


def test_every_layer_is_addressable():
    assert vienna.LAYERS
    for layer in vienna.LAYERS.values():
        assert layer.url.startswith("https://data.wien.gv.at/")
        assert f"ogdwien:{layer.typename}" in layer.url
        assert layer.geometry in ("polygon", "polyline", "point")
        assert layer.license and layer.attribution
        assert layer.notes, f"{layer.id} does not say what it is for"


def test_the_licence_is_recorded_as_the_grant_it_is():
    """Unlike Denver's, this one is a real grant. Do not downgrade it, and do
    not upgrade Denver's to match."""
    assert "CC BY 4.0" in vienna.VIENNA_TERMS
    assert "commercial use permitted" in vienna.VIENNA_TERMS


def test_the_two_scoring_layers_are_withheld():
    """Building bodies and the tree cadastre are the yardsticks, and both are
    exactly the layers it is tempting to feed in because the output improves."""
    withheld = {layer.id for layer in vienna.withheld()}
    assert withheld == {"building_bodies", "tree_cadastre"}


def test_the_layers_that_answer_denver_s_open_problems_are_present():
    """Each of these closes something Denver could not."""
    # Known weakness 6: no per-tree ground truth anywhere in Denver.
    trees = vienna.LAYERS["tree_cadastre"]
    assert "GATTUNG_ART" in trees.fields, "species is what makes crown size plausible"
    # The seed has no architectural family, which is why generated buildings
    # were identical boxes.
    assert "BAUTYP_TXT" in vienna.LAYERS["typology"].fields
    # Roof form was measured and discarded, then generated wrongly.
    assert "DACHTYP" in vienna.LAYERS["roof_cadastre"].fields
    # Eave and ground came from fitted returns in Denver; here they are stated.
    assert {"O_KOTE", "U_KOTE"} <= set(vienna.LAYERS["building_bodies"].fields)


def test_the_facade_capability_is_recorded_with_its_reason():
    """The whole point of the move: camera pose is published, so image to
    LiDAR correspondence is computed rather than approximated."""
    assert vienna.KAPPAZUNDER["license"] == "CC BY 4.0"
    assert any("orientation" in item for item in vienna.KAPPAZUNDER["ships"])
    assert vienna.OBLIQUE["epochs"] == ("2020", "2023")


def test_the_test_datasets_are_directly_downloadable():
    """An earlier note recorded Kappazunder as blocked behind a request form.
    That was read off the product page and was wrong: the test datasets are on
    data.gv.at under CC BY 4.0 with direct URLs. The form is for arbitrary
    areas, not for getting started."""
    urls = vienna.test_dataset_urls()
    assert set(urls) == {"info", "gis", "kappazunder"}
    for url in urls.values():
        assert url.startswith("https://www.wien.gv.at/ma41datenviewer/")
        assert url.endswith(".zip")
    assert vienna.TEST_LICENSE == "CC BY 4.0"


def test_the_ordering_route_still_distinguishes_the_two_paths():
    order = vienna.KAPPAZUNDER_ORDER
    assert "Test datasets: direct download" in order["route"]
    assert order["product_page"].startswith("https://www.wien.gv.at/")
    # The capability the whole move depends on, stated where it can be checked.
    assert "orientation" in order["metadata"]
    assert "first echoes" in order["point_attributes"]

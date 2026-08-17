"""The acquisition module, tested without a network.

The property worth protecting here is not that the download works -- that is the
service's job -- but that a layer's role survives the trip to disk. Truth that
lands in the input directory is invisible until it has already inflated a score.
"""
import json

import pytest

from lidarworld.data import acquire, denver


def _feature(oid, **props):
    return {"type": "Feature",
            "geometry": {"type": "Polygon",
                         "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
            "properties": {"OBJECTID": oid, **props}}


class FakeService:
    """Stands in for ArcGIS, including its habit of truncating silently."""

    def __init__(self, counts, max_records=2):
        self.counts = counts
        self.max_records = max_records
        self.calls = []

    def __call__(self, url, *, timeout, retries=5):
        self.calls.append(url)
        layer_id = next(l.id for l in denver.LAYERS.values() if l.url in url)
        offset = int(url.split("resultOffset=")[1].split("&")[0])
        total = self.counts.get(layer_id, 1)
        batch = [_feature(i, NAME=f"{layer_id}-{i}")
                 for i in range(offset, min(offset + self.max_records, total))]
        return {"type": "FeatureCollection", "features": batch,
                "exceededTransferLimit": offset + len(batch) < total}


@pytest.fixture
def service(monkeypatch):
    fake = FakeService({"parcels": 5, "building_outlines": 3})
    monkeypatch.setattr(acquire, "_get_json", fake)
    return fake


BBOX = (-105.002, 39.740, -104.985, 39.755)


def test_pagination_runs_to_the_end(service, tmp_path):
    """A short page is only the end if the service did not flag a cap."""
    parcels = denver.LAYERS["parcels"]
    result = acquire.fetch_layer(parcels, BBOX, page=2)
    assert len(result["features"]) == 5          # 2 + 2 + 1, not 2
    assert not result.get("lidarworld_truncated")


def test_truncation_is_recorded_not_swallowed(service, tmp_path):
    service.counts["parcels"] = 100
    result = acquire.fetch_layer(denver.LAYERS["parcels"], BBOX, page=2,
                                 max_pages=3)
    assert result["lidarworld_truncated"] is True


def test_truth_never_lands_where_the_compiler_reads(service, tmp_path):
    summary = acquire.acquire(BBOX, tmp_path, progress=None)
    assert not summary["failed"], summary["failed"]

    # Whatever the manifest withheld, for whatever reason -- not just the two
    # withholding roles. This is the assertion that failed before: a `prior`
    # surveyed after the scan is withheld on epoch grounds and was still being
    # routed to prior/ by its role.
    plan = denver.manifest(BBOX, epoch="2020", mode="reconstruction")
    withheld = {entry["id"] for entry in plan["withheld"]}
    admitted = {entry["id"] for entry in plan["layers"]}
    assert withheld and admitted

    epoch_only = withheld - {l.id for l in denver.LAYERS.values()
                             if l.role in ("hidden_truth", "later_epoch")}
    assert epoch_only, ("expected at least one layer withheld purely on epoch; "
                        "without one this test cannot see the bug it exists for")

    on_disk = {name: {p.name.split(".")[0] for p in (tmp_path / name).glob("*.geojson")}
               for name in ("input", "prior", "truth")}
    assert withheld <= on_disk["truth"]
    assert not (withheld & on_disk["input"])
    assert not (withheld & on_disk["prior"])
    assert admitted == on_disk["input"] | on_disk["prior"]

    # ...and reading it back cannot surface them either.
    loaded = acquire.load(tmp_path)
    assert not (withheld & set(loaded))
    assert set(loaded) == admitted


def test_withholding_reason_travels_with_the_file(service, tmp_path):
    acquire.acquire(BBOX, tmp_path, progress=None)
    record = json.loads(
        (tmp_path / "truth" / "parkland.provenance.json").read_text())
    assert record["withheld"] is True
    assert "after the 2020 observation" in record["withheld_reason"]
    assert record["role"] == "prior", "role is unchanged; only the routing moved"


def test_provenance_records_what_arrived_not_what_was_claimed(service, tmp_path):
    record = acquire.acquire_layer(denver.LAYERS["parcels"], BBOX, tmp_path)
    assert record["features"] == 5
    assert record["fields_present"] == ["NAME", "OBJECTID"]
    assert record["independence"] == 3
    assert record["role"] == "prior"
    assert record["sha256"] and record["bytes"] > 0
    assert record["geometry_returned"] == {"Polygon": 5}
    assert "Denver" in record["attribution"]

    sidecar = json.loads((tmp_path / "prior" / "parcels.provenance.json").read_text())
    assert sidecar == record


def test_an_arcgis_error_object_is_not_an_empty_layer(monkeypatch, tmp_path):
    """HTTP 200 with an error body is the failure mode that looks like success."""
    monkeypatch.setattr(acquire, "_get_json", lambda url, **kw: {
        "error": {"code": 400, "message": "Invalid or missing input parameters",
                  "details": []}})
    with pytest.raises(acquire.AcquisitionError, match="400"):
        acquire.fetch_layer(denver.LAYERS["parcels"], BBOX)


def test_every_role_has_a_directory():
    assert set(acquire.ROLE_DIR) == set(denver.ROLES)


def test_refetch_is_free(service, tmp_path):
    first = acquire.acquire_layer(denver.LAYERS["parcels"], BBOX, tmp_path)
    calls = len(service.calls)
    again = acquire.acquire_layer(denver.LAYERS["parcels"], BBOX, tmp_path)
    assert again == first
    assert len(service.calls) == calls, "cached layer was refetched"

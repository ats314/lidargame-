"""Tile index: headers only, spatial query, coverage.

The point of the index is that it never decodes a point, so the tests assert
that too -- a header read must be cheap and must survive a truncated file.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from lidarworld.data.tiles import TileIndex, TileRecord, read_header
from lidarworld.ingest.las import write_las


def bake(path, minx, miny, *, n=400, z=(0.0, 10.0), size=100.0):
    rng = np.random.default_rng(abs(hash((minx, miny))) % 2**31)
    xyz = np.column_stack([
        minx + rng.random(n) * size,
        miny + rng.random(n) * size,
        rng.uniform(z[0], z[1], n),
    ])
    # Pin the corners so the header bounds are exactly the tile bounds.
    xyz[0] = [minx, miny, z[0]]
    xyz[1] = [minx + size, miny + size, z[1]]
    return write_las(path, xyz, np.full(n, 0.3, np.float32),
                     np.full(n, 2, np.uint8))


@pytest.fixture
def grid(tmp_path):
    """A 2x2 grid of 100 m tiles with the top-right one missing."""
    for (i, j) in [(0, 0), (1, 0), (0, 1)]:
        bake(tmp_path / f"tile_{i}_{j}.las", i * 100.0, j * 100.0)
    return tmp_path


def test_read_header_reports_bounds_without_decoding(tmp_path):
    path = bake(tmp_path / "one.las", 500.0, 700.0, n=250)
    record = read_header(path)
    assert record is not None
    assert record.points == 250
    assert record.minx == pytest.approx(500.0, abs=1e-3)
    assert record.miny == pytest.approx(700.0, abs=1e-3)
    assert record.maxx == pytest.approx(600.0, abs=1e-3)
    assert record.maxy == pytest.approx(800.0, abs=1e-3)
    assert record.area == pytest.approx(1e4, rel=1e-3)
    assert record.density == pytest.approx(250 / 1e4, rel=1e-3)

    # Only the header is needed: truncating every point still reads.
    head = path.read_bytes()[:512]
    stub = tmp_path / "headeronly.las"
    stub.write_bytes(head)
    truncated = read_header(stub)
    assert truncated is not None
    assert truncated.maxx == pytest.approx(record.maxx, abs=1e-6)


def test_read_header_rejects_non_las(tmp_path):
    junk = tmp_path / "notes.txt"
    junk.write_text("this is not a point cloud")
    assert read_header(junk) is None
    assert read_header(tmp_path / "missing.las") is None


def test_intersects_is_inclusive_of_shared_edges():
    t = TileRecord("x", 0, 0, 10, 10, 0, 5, points=10)
    assert t.intersects((5, 5, 15, 15))
    assert t.intersects((-5, -5, 0, 0))       # touching corner
    assert t.intersects((-5, -5, 50, 50))     # fully contains
    assert not t.intersects((10.5, 0, 20, 10))
    assert not t.intersects((0, -20, 10, -0.5))


def test_build_indexes_every_tile_and_caches(grid):
    index = TileIndex.build(grid)
    assert len(index) == 3
    assert index.total_points == 1200
    assert index.bounds == pytest.approx((0.0, 0.0, 200.0, 200.0), abs=1e-3)

    cache = grid / TileIndex.CACHE
    assert cache.exists()
    assert len(json.loads(cache.read_text())["tiles"]) == 3

    # A second build must come from the cache, not from the files.
    for path in grid.glob("*.las"):
        path.unlink()
    again = TileIndex.build(grid)
    assert len(again) == 3
    assert again.summary() == index.summary()


def test_build_ignores_a_corrupt_cache(grid):
    (grid / TileIndex.CACHE).write_text("{not json")
    assert len(TileIndex.build(grid)) == 3


def test_query_returns_only_overlapping_tiles_densest_first(grid):
    bake(grid / "dense.las", 0.0, 0.0, n=4000)
    index = TileIndex.build(grid, use_cache=False)

    hits = index.query((10, 10, 40, 40))
    assert {h.points for h in hits} == {400, 4000}
    assert hits[0].points == 4000, "densest tile should sort first"

    assert index.query((900, 900, 950, 950)) == []
    assert len(index.around(50.0, 50.0, 20.0)) == 2


def test_coverage_detects_the_hole_in_the_grid(grid):
    index = TileIndex.build(grid, use_cache=False)
    assert index.coverage((10, 10, 90, 90)) == pytest.approx(1.0)
    assert index.coverage((110, 110, 190, 190)) == 0.0
    # The full 200x200 extent is three quarters tiles, one quarter hole.
    assert index.coverage((0, 0, 200, 200)) == pytest.approx(0.75, abs=0.02)


class Args:
    def __init__(self, inputs, area=None):
        self.inputs = [str(p) for p in inputs]
        self.area = area


def test_cli_area_is_a_centred_square():
    from lidarworld.cli import _parse_area

    assert _parse_area("100,200,50") == (75.0, 175.0, 125.0, 225.0)
    for bad in ("1,2", "1,2,3,4", "a,b,c"):
        with pytest.raises(SystemExit):
            _parse_area(bad)


def test_cli_opens_only_the_tiles_the_area_touches(grid):
    from lidarworld.cli import _resolve_inputs

    inputs, area = _resolve_inputs(Args([grid], area="20,20,20"))
    assert [p.rsplit("/", 1)[-1] for p in inputs] == ["tile_0_0.las"]
    assert area == (10.0, 10.0, 30.0, 30.0)

    everything, no_area = _resolve_inputs(Args([grid]))
    assert len(everything) == 3 and no_area is None

    # Explicit files stay explicit -- no index, no filtering.
    one = grid / "tile_1_0.las"
    files, _ = _resolve_inputs(Args([one], area="20,20,20"))
    assert files == [str(one)]


def test_cli_refuses_an_area_outside_the_data(grid):
    from lidarworld.cli import _resolve_inputs

    with pytest.raises(SystemExit) as excinfo:
        _resolve_inputs(Args([grid], area="9000,9000,50"))
    assert "selects no tiles" in str(excinfo.value)

    with pytest.raises(SystemExit) as excinfo:
        _resolve_inputs(Args([grid / "typo.las"], area=None))
    assert "no such file" in str(excinfo.value)

    (grid / "empty").mkdir()
    with pytest.raises(SystemExit) as excinfo:
        _resolve_inputs(Args([grid / "empty"], area=None))
    assert "no LAS/LAZ tiles" in str(excinfo.value)


def test_summary_of_an_empty_index_does_not_divide_by_zero(tmp_path):
    index = TileIndex.build(tmp_path)
    assert len(index) == 0
    assert index.summary() == {"tiles": 0, "points": 0, "extent_km2": 0.0,
                               "mean_density": 0.0, "bounds": [0.0, 0.0, 0.0, 0.0]}
    assert index.query((0, 0, 1, 1)) == []
    assert index.coverage((0, 0, 1, 1)) == 0.0

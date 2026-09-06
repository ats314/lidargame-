"""A World Seed driving a renderer that has no geometry in it.

The other backends all emit triangles, so between them they cannot tell you
whether the IR is engine-independent or merely glTF-shaped. This one emits four
bytes per 4 m cell and nothing else, which is the sharper test: if a seed drives
an ASCII raycaster as well as it drives glTF, the invariant is real.

The PNG is decoded here rather than trusted, because the whole claim is that a
program which knows nothing about the compiler can read the file.
"""
from __future__ import annotations

import json
import struct
import zlib

import numpy as np
import pytest

from lidarworld.backends import noctis


def _decode_png(path):
    """Minimal PNG reader: enough to prove the file is really a PNG.

    Deliberately not Pillow -- the point is that the bytes are conformant, not
    that a library we also wrote the writer against agrees with itself.
    """
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat, width, height = 8, b"", 0, 0
    while pos < len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        tag = data[pos + 4:pos + 8]
        payload = data[pos + 8:pos + 8 + length]
        crc = struct.unpack(">I", data[pos + 8 + length:pos + 12 + length])[0]
        assert crc == zlib.crc32(tag + payload) & 0xFFFFFFFF, f"bad CRC on {tag}"
        if tag == b"IHDR":
            width, height, depth, colour = struct.unpack(">IIBB", payload[:10])
            assert (depth, colour) == (8, 6), "expected 8-bit RGBA"
        elif tag == b"IDAT":
            idat += payload
        pos += 12 + length

    raw = zlib.decompress(idat)
    stride = width * 4
    out = np.zeros((height, width, 4), dtype=np.uint8)
    for y in range(height):
        row = raw[y * (stride + 1):(y + 1) * (stride + 1)]
        assert row[0] == 0, "writer only emits filter type 0"
        out[y] = np.frombuffer(row[1:], dtype=np.uint8).reshape(width, 4)
    return out


@pytest.fixture
def block():
    """A 200 m block: two buildings, a street between them, a canal, a tree."""
    return {
        "seed": "lidarworld/0.1",
        "name": "test_block",
        "crs": "EPSG:28992",
        "origin": [0.0, 0.0, 0.0],
        "bounds": [[0.0, 0.0, 0.0], [200.0, 200.0, 80.0]],
        # The seed stores ground as z[ix][iy]; the renderer wants [row=y][col=x].
        # Ramping both axes at different rates makes a transpose detectable.
        "terrain": {"shape": [4, 4], "step_m": 50.0,
                    "z": [[ix * 1.0 + iy * 0.5 for iy in range(4)]
                          for ix in range(4)]},
        "buildings": [
            {"id": "b.low", "ground_z": 0.0, "height": 12.0, "roof": "gabled",
             "footprint": [[20, 20], [60, 20], [60, 60], [20, 60], [20, 20]]},
            {"id": "b.tower", "ground_z": 0.0, "height": 72.0, "roof": "flat",
             "footprint": [[120, 20], [170, 20], [170, 70], [120, 70], [120, 20]]},
        ],
        "roads": [{"line": [[0, 100], [200, 100]], "half_width": 6.0}],
        "water": [{"ring": [[0, 170], [200, 170], [200, 195], [0, 195], [0, 170]],
                   "level_z": 0.0, "surface": "inferred"}],
        "vegetation": [{"xy": [90.0, 140.0], "base_z": 0.0,
                        "crown_r": 6.0, "height": 9.0}],
        "provenance": {"sources": ["synthetic"], "crs": "EPSG:28992"},
    }


def _planes(block, **kw):
    kw.setdefault("grid", 64)
    kw.setdefault("extent_m", 200.0)
    planes, meta = noctis.bake(block, **kw)
    grid = meta["grid"]
    return planes[:grid], planes[grid:grid * 2], planes[grid * 2:], meta


def test_the_texture_has_the_shape_the_renderer_reads(block):
    planes, meta = noctis.bake(block, grid=64, extent_m=200.0)
    # N wide, 3N tall: grid plane, flag plane, terrain plane.
    assert planes.shape == (192, 64, 4)
    assert (planes[..., 3] == 255).all(), "alpha carries no data; canvas is premultiplied"
    assert meta["cellM"] == pytest.approx(200.0 / 64)


def test_buildings_land_where_the_seed_put_them_at_the_height_it_recorded(block):
    top, _, _, meta = _planes(block)
    cell = meta["cellM"]

    # Cell coordinates are (x - lo) / cell with the tile centred on the bounds,
    # which for a 200 m tile over 200 m of seed is the identity.
    def at(x, y):
        return top[int(y / cell), int(x / cell)]

    assert at(40, 40)[0] == round(12.0 / noctis.HSTEP)     # the low block
    assert at(145, 45)[0] == round(72.0 / noctis.HSTEP)    # the tower
    assert at(100, 40)[0] == 0                             # the gap between them

    # Style is derived from measured quantities, and the two buildings differ
    # because their heights and roof forms do.
    assert (at(40, 40)[1] >> 4) == noctis.STYLE_UNIFORM    # pitched roof
    assert (at(145, 45)[1] >> 4) == noctis.STYLE_GLAZED    # >= 60 m


def test_roads_water_and_canopy_reach_the_flag_plane(block):
    _, flags, _, meta = _planes(block)
    cell = meta["cellM"]

    def at(x, y):
        return int(flags[int(y / cell), int(x / cell)][0])

    street = at(100, 100)
    assert street & noctis.ROAD
    assert street & noctis.ROADX and not street & noctis.ROADZ   # runs along x
    assert at(100, 180) & noctis.WATER
    assert at(90, 140) & noctis.PARK
    assert at(145, 45) & noctis.BEACON, "a 72 m building is lit for aircraft"


def test_street_facing_is_derived_not_invented(block):
    """The one context flag the renderer uses, computed on the 4 m grid."""
    building = dict(block["buildings"][1])
    building["footprint"] = [[120, 92], [170, 92], [170, 96], [120, 96], [120, 92]]
    fronting = dict(block, buildings=[building])
    _, flags, _, meta = _planes(fronting)
    cell = meta["cellM"]
    assert int(flags[int(94 / cell), int(145 / cell)][0]) & noctis.SIGNSTRIP

    # Move the same building away from the carriageway and the flag goes away.
    building = dict(building,
                    footprint=[[120, 20], [170, 20], [170, 24], [120, 24], [120, 20]])
    _, flags, _, _ = _planes(dict(block, buildings=[building], roads=[]))
    assert not int(flags[int(22 / cell), int(145 / cell)][0]) & noctis.SIGNSTRIP


def test_terrain_is_quantised_against_its_own_relief(block):
    _, _, terrain, meta = _planes(block)
    # 3 m of fall west to east, 1.5 m south to north. Both must climb, and the
    # x gradient must be the steeper one -- swap the axes and this inverts.
    south_west = int(terrain[4, 4][0])
    south_east = int(terrain[4, 60][0])
    north_west = int(terrain[60, 4][0])
    assert south_east > south_west
    assert north_west > south_west
    assert south_east - south_west > north_west - south_west

    assert meta["relief"] == pytest.approx(4.5, abs=0.01)
    assert terrain.max() * meta["terrStep"] == pytest.approx(4.5, abs=0.05)


def test_a_seed_that_overflows_the_tile_says_so_instead_of_losing_buildings(block):
    # 64 cells at 4 m is 256 m across; push a building to 500 m out and it is
    # cropped. Silently dropping it is how half a city goes missing unnoticed.
    far = dict(block["buildings"][0], id="b.far",
               footprint=[[500, 500], [540, 500], [540, 540], [500, 540], [500, 500]])
    _, meta = noctis.bake(dict(block, buildings=block["buildings"] + [far]),
                          grid=64, cell_m=4.0)
    assert meta["seedBuildings"] == 3
    assert meta["buildingsPlaced"] == 2
    assert meta["buildingsCropped"] == 1


def test_the_bake_names_no_material_theme_or_engine(block):
    """The invariant, at the last stage that could break it.

    Materialisation happens in the renderer, from four bytes that mean height,
    style index, window density and context -- never a texture name.
    """
    _, meta = noctis.bake(block, grid=32, extent_m=200.0)
    blob = json.dumps(meta).lower()
    for word in ("brick", "stucco", "material", "texture", "shader", "theme",
                 "victorian", "neon", "albedo", "godot", "gltf"):
        assert word not in blob, f"{word!r} leaked into a seed-derived bake"


def test_the_same_seed_bakes_to_the_same_bytes(block, tmp_path):
    first = noctis.export(block, tmp_path / "a.png", grid=64, extent_m=200.0)
    second = noctis.export(block, tmp_path / "b.png", grid=64, extent_m=200.0)
    assert (tmp_path / "a.png").read_bytes() == (tmp_path / "b.png").read_bytes()
    assert first["bytes"] == second["bytes"]


def test_the_file_is_a_png_a_stranger_can_read(block, tmp_path):
    meta = noctis.export(block, tmp_path / "city.png", grid=64, extent_m=200.0,
                         meta_path=tmp_path / "city.json")
    decoded = _decode_png(tmp_path / "city.png")
    assert decoded.shape == (192, 64, 4)

    planes, _ = noctis.bake(block, grid=64, extent_m=200.0)
    assert np.array_equal(decoded, planes), "what was written is what was baked"

    written = json.loads((tmp_path / "city.json").read_text())
    assert written["grid"] == 64 and written["path"].endswith("city.png")
    assert meta["credit"] == "synthetic", "provenance survives into the target"


def test_a_city_costs_a_quarter_of_a_megabyte_at_the_renderer_s_size(block, tmp_path):
    """256 cells is 1 km across, and the renderer's budget for it is one texture."""
    meta = noctis.export(block, tmp_path / "city.png", grid=256)
    assert meta["extentM"] == pytest.approx(1024.0)
    assert meta["bytes"] < 256 * 1024


def test_the_credit_is_an_attribution_not_an_internal_id(block):
    """Terms travel with the data, or the target credits nobody.

    Seed provenance used to carry only source ids -- "src0" -- so a renderer
    consuming a real survey put an internal label on screen where the
    provider's attribution line belonged.
    """
    licensed = dict(block, provenance={"sources": [
        {"id": "src0", "license": "CC0 1.0",
         "attribution": "AHN / Rijkswaterstaat; tiling by GeoTiles, TU Delft"}]})
    _, meta = noctis.bake(licensed, grid=32, extent_m=200.0)
    assert meta["credit"] == (
        "AHN / Rijkswaterstaat; tiling by GeoTiles, TU Delft (CC0 1.0)")

    # An older seed that carries bare ids still renders something.
    legacy = dict(block, provenance={"sources": ["usgs_3dep"]})
    assert noctis.bake(legacy, grid=32, extent_m=200.0)[1]["credit"] == "usgs_3dep"

    # And silence is never mistaken for permission.
    bare = dict(block, provenance={})
    assert noctis.bake(bare, grid=32, extent_m=200.0)[1]["credit"] == (
        "source and terms unrecorded")

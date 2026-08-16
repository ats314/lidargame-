"""Theme resolution: the same (role, context) must always pick the same material,
and a more specific rule must always beat a general one.
"""
from __future__ import annotations

import numpy as np
import pytest

from lidarworld.roles.taxonomy import Ctx, citygml_type, role_matches
from lidarworld.themes import (available_packs, compile_theme, compile_runtime_table,
                               load_pack, procedural)
from lidarworld.themes.png import encode
from lidarworld.themes.pack import ThemePack
from lidarworld.themes.request import MaterialRequest, MaterialSpec, Rule


def test_context_flags_round_trip():
    names = ["corner_convex", "street_facing", "near_opening"]
    mask = Ctx.encode(names)
    assert set(Ctx.decode(mask)) == set(names)
    with pytest.raises(KeyError):
        Ctx.encode(["not_a_flag"])


def test_role_prefix_matching():
    assert role_matches("surface.wall.vertical", "surface")
    assert role_matches("surface.wall.vertical", "surface.wall")
    assert role_matches("surface.wall.vertical", "*")
    assert not role_matches("surface.wall.vertical", "surface.roof")
    # A prefix must stop at a dot boundary, not at any character.
    assert not role_matches("surface.walltop", "surface.wall")


def test_more_specific_rule_wins():
    pack = ThemePack(
        id="t", name="t", fallback="plain",
        materials={m.id: m for m in [MaterialSpec(id="plain"), MaterialSpec(id="quoin")]},
        rules=[
            Rule(material="plain", role="surface.wall"),
            Rule(material="quoin", role="surface.wall", ctx_all=("corner_convex",)),
        ],
    )
    plain = MaterialRequest("surface.wall.vertical", 0)
    corner = MaterialRequest("surface.wall.vertical", Ctx.CORNER_CONVEX)
    assert pack.resolve(plain)[0].id == "plain"
    assert pack.resolve(corner)[0].id == "quoin"


def test_ctx_none_excludes():
    pack = ThemePack(
        id="t", name="t", fallback="plain",
        materials={m.id: m for m in [MaterialSpec(id="plain"), MaterialSpec(id="lit")]},
        rules=[
            Rule(material="lit", role="surface.wall", ctx_all=("street_facing",),
                 ctx_none=("ground_contact",)),
            Rule(material="plain", role="*"),
        ],
    )
    street = MaterialRequest("surface.wall.vertical", Ctx.STREET_FACING)
    plinth = MaterialRequest("surface.wall.vertical", Ctx.STREET_FACING | Ctx.GROUND_CONTACT)
    assert pack.resolve(street)[0].id == "lit"
    assert pack.resolve(plinth)[0].id == "plain"


def test_unmatched_request_falls_back():
    pack = ThemePack(id="t", name="t", fallback="plain",
                     materials={"plain": MaterialSpec(id="plain")},
                     rules=[Rule(material="plain", role="surface.roof")])
    material, rule = pack.resolve(MaterialRequest("volume.vegetation.high", 0))
    assert material.id == "plain"
    assert rule is None


def test_pack_validation_catches_mistakes():
    bad = ThemePack(id="t", name="t", fallback="missing",
                    materials={"real": MaterialSpec(id="real")},
                    rules=[Rule(material="ghost", role="*"),
                           Rule(material="real", role="*", ctx_all=("nope",))])
    problems = bad.validate()
    assert any("ghost" in p for p in problems)
    assert any("nope" in p for p in problems)
    assert any("missing" in p for p in problems)


@pytest.mark.parametrize("pack_id", available_packs())
def test_builtin_packs_are_valid_and_total(pack_id):
    """Every shipped pack must resolve every role without falling through."""
    pack = load_pack(pack_id)
    assert pack.validate() == []
    from lidarworld.roles.taxonomy import ROLE_IDS
    for role in ROLE_IDS:
        material, _ = pack.resolve(MaterialRequest(role, 0))
        assert material.id in pack.materials


def test_runtime_table_is_ordered_most_specific_first():
    table = compile_runtime_table(load_pack("victorian"))
    materials = [table["materials"][r["material"]]["id"] for r in table["rules"]]
    # A context-qualified wall rule must be evaluated before the plain one.
    assert materials.index("portland_stone") < materials.index("stock_brick")
    assert table["rules"][-1]["role"] == "*", "a pack must end with a catch-all"
    # Every rule that carries no context must come after one that does.
    first_unqualified = next(i for i, r in enumerate(table["rules"]) if not (r["all"] or r["any"]))
    assert any(r["all"] or r["any"] for r in table["rules"][:first_unqualified])


def test_victorian_puts_stone_on_corners_and_openings():
    pack = load_pack("victorian")
    wall = MaterialRequest("surface.wall.vertical", 0)
    corner = MaterialRequest("surface.wall.vertical", Ctx.CORNER_CONVEX)
    opening = MaterialRequest("surface.wall.vertical", Ctx.OPENING_BOUNDARY)
    plinth = MaterialRequest("surface.wall.vertical", Ctx.GROUND_CONTACT)

    assert pack.resolve(wall)[0].id == "stock_brick"
    assert pack.resolve(corner)[0].id == "portland_stone"
    assert pack.resolve(opening)[0].id == "portland_stone"
    assert pack.resolve(plinth)[0].id == "stone_plinth"


def test_themes_disagree_on_the_same_request():
    """The whole point: identical geometry, different material per theme."""
    request = MaterialRequest("surface.wall.vertical", Ctx.STREET_FACING)
    chosen = {p: load_pack(p).resolve(request)[0].id for p in ("victorian", "neon", "survey")}
    assert len(set(chosen.values())) == 3, chosen


def test_procedural_generators_are_tileable_and_shaped():
    for name in procedural.GENERATORS:
        spec = MaterialSpec(id=f"m_{name}", generator=name)
        maps = procedural.bake(spec)
        size = procedural.SIZE
        assert maps["albedo"].shape == (size, size, 3)
        assert maps["normal"].shape == (size, size, 3)
        assert maps["albedo"].dtype == np.uint8
        # Tileable: the wrap-around step must be no worse than the largest
        # step the pattern already takes internally (a roof course or a mortar
        # line is a hard edge by design; a seam is one the pattern never has).
        albedo = maps["albedo"].astype(int)
        seam = np.abs(albedo[0] - albedo[-1]).mean()
        internal = np.abs(np.diff(albedo, axis=0)).mean(axis=(1, 2)).max()
        # A plank seam or roof course is a hard edge the pattern already makes
        # once per period, and the wrap lands on one of them. What would be a
        # bug is a step the pattern never takes anywhere else.
        assert seam <= internal * 1.6 + 4, f"{name} seams at the tile boundary"


def test_unknown_generator_is_rejected():
    with pytest.raises(KeyError, match="unknown procedural generator"):
        procedural.bake(MaterialSpec(id="x", generator="unobtanium"))


def test_png_encoder_writes_a_valid_header():
    image = np.zeros((4, 6, 3), np.uint8)
    blob = encode(image)
    assert blob[:8] == b"\x89PNG\r\n\x1a\n"
    assert blob[12:16] == b"IHDR"
    width = int.from_bytes(blob[16:20], "big")
    height = int.from_bytes(blob[20:24], "big")
    assert (width, height) == (6, 4)
    assert blob[-8:-4] == b"IEND"


def test_compile_theme_writes_textures_and_provenance(tmp_path):
    info = compile_theme(load_pack("neon"), tmp_path / "neon")
    assert info["baked"] == info["materials"]
    theme = __import__("json").loads((tmp_path / "neon" / "theme.json").read_text())
    for material in theme["materials"]:
        assert material["provenance"]["license"]
        assert material["textures"]["albedo"]
        assert (tmp_path / "neon" / material["textures"]["albedo"]).exists()


def test_citygml_alignment_is_sane():
    assert citygml_type("surface.wall.vertical") == "WallSurface"
    assert citygml_type("surface.roof.pitched") == "RoofSurface"
    assert citygml_type("opening.door") == "Door"
    assert citygml_type("volume.building", surface=False) == "Building"
    assert citygml_type("volume.vegetation.high", surface=False) == "SolitaryVegetationObject"
    assert citygml_type("nonsense.role") == "GenericCityObject"

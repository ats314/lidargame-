"""Frequency-separated facade composition.

The property that matters is non-destructiveness: adding high-frequency detail
must not move the measured colour. Every bug found while building this broke that
property while looking fine, so the assertions are on colour and brightness
rather than on appearance.
"""
from __future__ import annotations

import numpy as np
import pytest

from lidarworld.features import frequency as fq


def flat(value=0.55, size=96):
    return np.full((size, size, 3), value)


def pink(size=96):
    out = np.zeros((size, size, 3))
    out[:, :, 0], out[:, :, 1], out[:, :, 2] = 0.62, 0.45, 0.42
    return out


def high_frequency(image):
    grey = np.asarray(image, dtype=np.float64)
    if grey.ndim == 3:
        grey = grey.mean(axis=2)
    if grey.max() > 1.5:
        grey = grey / 255.0
    return float(np.sqrt(np.mean((grey - fq.box_blur(grey[:, :, None], 4)[:, :, 0]) ** 2)))


def test_box_blur_conserves_total_and_wraps():
    """Wrap-around, because the micro tile has to stay seamless.

    Clamping at the edge would put a visible seam exactly where the tile
    repeats, which is the one place a tileable texture must not have one.
    """
    spike = np.zeros((16, 16, 1))
    spike[0, 0, 0] = 16.0                       # on the corner, so wrap matters
    blurred = fq.box_blur(spike, 2)
    assert blurred.sum() == pytest.approx(16.0, rel=1e-9)
    # Energy must appear on the opposite edges, not be lost off the side.
    assert blurred[-1, -1, 0] > 0


def test_a_neutralised_field_averages_one():
    """Without this the composite darkens the facade by the micro's own mean."""
    from lidarworld.themes import procedural
    for material in ("brick", "stone_block", "concrete", "plaster"):
        albedo, _, _ = procedural.GENERATORS[material](seed=3)
        field = fq.neutralise(albedo)
        assert field.ndim == 2, "the detail field must be scalar, not per-channel"
        assert float(field.mean()) == pytest.approx(1.0, abs=1e-6)


def test_detail_is_luminance_so_hue_cannot_shift():
    """Per-channel neutralisation repainted a grey wall as red brick.

    The raw ratios do average one per channel, but the tails have to be clipped
    and brick's mortar is bright against its dark blue base, so blue lost far
    more to the clip than red. Post-clip means came out [0.997, 0.941, 0.884]
    and a flat grey macro acquired a 0.09 blue deficit. A scalar field cannot do
    that however strong the detail is.
    """
    composite = fq.compose(flat(), material="brick", px_per_m=500.0, tile_m=0.3,
                           seed=3)
    shift = composite.albedo.reshape(-1, 3).mean(0) - flat().reshape(-1, 3).mean(0)
    assert np.allclose(shift, shift[0], atol=1e-9), "channels must move together"


def test_a_coloured_facade_keeps_its_colour():
    """The photograph exists to supply identity; the micro must not repaint it.

    The patch has to span several tiles. At 96 px and 500 px/m it covers 0.19 m
    of wall against a 0.45 m tile, so the sample lands inside a single course and
    its mean says nothing about the material.
    """
    macro = pink(size=480)                       # 0.96 m at 500 px/m, ~2 tiles
    composite = fq.compose(macro, material="stone_block", px_per_m=500.0,
                           tile_m=0.45, seed=3)
    before = macro.reshape(-1, 3).mean(0)
    after = composite.albedo.reshape(-1, 3).mean(0)
    assert after[0] / after[2] == pytest.approx(before[0] / before[2], rel=0.02)
    assert after.mean() == pytest.approx(before.mean(), abs=0.01)


def test_zero_detail_returns_the_photograph_exactly():
    """The w = 0 case has to be the identity, or the injection is not auditable."""
    macro = pink()
    composite = fq.compose(macro, material="brick", px_per_m=500.0, tile_m=0.3,
                           detail=0.0, seed=3)
    assert np.allclose(composite.albedo, macro, atol=1e-12)


def test_detail_actually_adds_high_frequency():
    """The whole claim. A flat macro has none; the composite must have some.

    Also the reason the measurement exists: the first A/B used `plaster`, which
    contributes 0.003 against brick's 0.16, and read as "frequency separation
    does not work" rather than "wrong material".
    """
    macro = flat()
    assert high_frequency(macro) == pytest.approx(0.0, abs=1e-9)
    composite = fq.compose(macro, material="brick", px_per_m=500.0, tile_m=0.3,
                           seed=3)
    assert high_frequency(composite.albedo) > 0.05


def test_the_micro_is_placed_in_metres_not_normalised():
    """A brick is 240 mm on a shed and on a warehouse.

    The invariant is about *wall*, not pixels: a fixed extent of wall shows the
    same number of courses at any sampling. Asserting that more pixels per metre
    means more courses is backwards -- at a fixed pixel size, raising px_per_m
    covers less wall and shows fewer.
    """
    def courses(px_per_m, wall_m=1.6, tile_m=0.4):
        size = int(round(wall_m * px_per_m))
        composite = fq.compose(flat(size=size), material="brick",
                               px_per_m=px_per_m, tile_m=tile_m, seed=3)
        column = composite.albedo[:, size // 2].mean(axis=1)
        crossings = np.diff((column > column.mean()).astype(int))
        return int((crossings != 0).sum())

    coarse, fine = courses(150.0), courses(300.0)
    assert coarse == pytest.approx(fine, rel=0.25), (
        f"{coarse} courses at 150 px/m vs {fine} at 300 px/m over the same wall")

    # And doubling the wall at fixed sampling must double the courses.
    assert courses(200.0, wall_m=3.2) > 1.6 * courses(200.0, wall_m=1.6)


def test_relief_shading_uses_the_micro_normal():
    """A facade at 1 m is missing surface, not colour, so normals must count."""
    composite = fq.compose(flat(), material="stone_block", px_per_m=500.0,
                           tile_m=0.45, seed=3)
    with_normal = fq.shade(composite, ambient=0.45)
    without = fq.shade(composite, ambient=0.45, use_micro_normal=False)
    assert high_frequency(with_normal) > high_frequency(without)


def test_an_unknown_material_is_refused_by_name():
    with pytest.raises(KeyError, match="unknown micro material"):
        fq.compose(flat(), material="marzipan")


def test_the_record_keeps_measured_and_generated_apart():
    composite = fq.compose(flat(), material="brick", px_per_m=500.0, tile_m=0.3)
    record = composite.to_record()
    assert record["macro_epistemic"] == "derived"     # somebody's photogrammetry
    assert record["micro_epistemic"] == "generated"   # synthesised here
    assert record["micro_tile_m"] == 0.3

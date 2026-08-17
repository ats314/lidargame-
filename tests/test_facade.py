"""Rectifying a wall out of a texture atlas, and judging what came out.

The failures guarded against here are all silent: a crop that is upside down, a
crop that is mirrored, a resolution figure that reads zero because two arrays
disagreed about a closing vertex. None of them raise, and all of them look like
facade data until someone looks.
"""
from __future__ import annotations

import numpy as np
import pytest

from lidarworld.features import facade as F


def wall(width=8.0, height=6.0):
    """A wall in the x-z plane, counter-clockwise seen from -y."""
    return np.array([[0, 0, 0], [width, 0, 0], [width, 0, height], [0, 0, height]],
                    dtype=float)


def unit_uv():
    """UV0 in CityGML's convention: v increases upward."""
    return np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


def gradient_atlas(size=64):
    """An atlas whose top rows are bright, so an upside-down crop is obvious."""
    atlas = np.zeros((size, size, 3), dtype=np.uint8)
    atlas[:, :, :] = np.linspace(255, 0, size, dtype=np.uint8)[:, None, None]
    return atlas


def test_a_wall_rectifies_to_its_metric_shape():
    result = F.rectify(wall(8.0, 6.0), unit_uv(), gradient_atlas(),
                       px_per_m=16.0, surface_id="s", building_id="b")
    assert result is not None
    assert result.width_m == pytest.approx(8.0)
    assert result.height_m == pytest.approx(6.0)
    assert result.image.shape[:2] == (96, 128)          # 6 m x 8 m at 16 px/m
    assert result.covered > 0.99


def test_the_crop_is_not_upside_down():
    """CityGML's v origin is the lower left; an image's row 0 is the top.

    Getting this wrong flips every facade and no count changes -- the crop is
    still the right size, still fully covered, still the right colours.
    """
    atlas = gradient_atlas()                            # bright at row 0
    result = F.rectify(wall(), unit_uv(), atlas, px_per_m=8.0)
    top = result.image[:4].mean()
    bottom = result.image[-4:].mean()
    # v=1 is the top of the wall and maps to the atlas's bright row 0.
    assert top > bottom + 100


def test_a_pixel_maps_back_to_a_place_on_the_wall():
    """The whole point of keeping the frame: a detection becomes a position."""
    result = F.rectify(wall(8.0, 6.0), unit_uv(), gradient_atlas(), px_per_m=16.0)
    bottom_left = result.to_world(0.0, result.image.shape[0])
    top_right = result.to_world(result.image.shape[1], 0.0)
    assert np.allclose(bottom_left, [0, 0, 0], atol=1e-6)
    assert np.allclose(top_right, [8, 0, 6], atol=1e-6)


def test_source_resolution_reports_what_the_atlas_actually_gave():
    """A 64 px atlas edge over an 8 m wall is 8 px/m, whatever the crop is."""
    result = F.rectify(wall(8.0, 8.0), unit_uv(), gradient_atlas(64),
                       px_per_m=32.0)
    assert result.resolution_px_per_m == pytest.approx(8.0, rel=1e-3)
    # The crop oversamples; that must not be confused with source detail.
    assert result.px_per_m == 32.0


def test_a_uv_ring_that_does_not_close_identically_still_measures():
    """Six of twenty-four Hamburg walls hit this and silently read 0 px/m.

    `close_ring` drops a repeated final vertex. When the ring repeats it and the
    UVs do not -- or repeat it to a different last bit -- closing the two
    independently makes their lengths disagree, and the length guard returned
    zero rather than a resolution.
    """
    ring = np.vstack([wall(8.0, 8.0), [[0, 0, 0]]])     # explicitly closed
    uv = np.vstack([unit_uv(), [[0.0, 1e-9]]])          # closed, but not exactly
    assert F.source_resolution(ring, uv, (64, 64)) > 0.0


def test_a_degenerate_wall_is_refused_rather_than_returned_blank():
    sliver = np.array([[0, 0, 0], [0.01, 0, 0], [0.01, 0, 0.01], [0, 0, 0.01]],
                      dtype=float)
    assert F.rectify(sliver, unit_uv(), gradient_atlas()) is None


def test_rhythm_separates_a_repeating_facade_from_a_flat_one():
    """Window bays repeat horizontally; a roof photographed obliquely does not."""
    px = 32.0
    height, width = int(7 * px), int(20 * px)
    bays = np.zeros((height, width, 3), dtype=np.uint8)
    bays[:, :, :] = 180
    for x in range(0, width, int(1.5 * px)):            # the measured 1.5 m bay
        bays[:, x:x + int(0.6 * px)] = 40
    flat = np.full((height, width, 3), 150, dtype=np.uint8)

    assert F.rhythm_profile(bays, px).max() > F.RHYTHM_THRESHOLD
    assert F.rhythm_profile(flat, px).max() < F.RHYTHM_THRESHOLD


def test_the_usable_band_finds_where_the_facade_stops():
    """An aerial oblique sees the top of a wall and not the bottom.

    Built as a crop whose upper half repeats and whose lower half is noise,
    which is the shape of a real Hamburg wall with roof bleed underneath it.
    """
    px = 32.0
    height, width = int(12 * px), int(20 * px)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, :] = 180
    for x in range(0, width, int(1.5 * px)):
        image[:height // 2, x:x + int(0.6 * px)] = 40
    rng = np.random.default_rng(0)
    image[height // 2:] = rng.integers(90, 170, (height - height // 2, width, 3),
                                       dtype=np.uint8)
    top, bottom = F.usable_band(image, px)
    assert top == 0.0
    # 12 m at 3.5 m storeys is three whole bands. Bays run to 6 m, so band 2
    # (3.5-7 m) is 71% facade and belongs inside; band 3 is pure noise and must
    # not. The boundary lands on a storey, which is the resolution of the
    # measurement and the unit anyone would act on.
    assert 0.5 <= bottom < 1.0


def test_quality_travels_in_the_dna_record():
    result = F.rectify(wall(8.0, 6.0), unit_uv(), gradient_atlas(), px_per_m=16.0)
    dna = result.to_dna()
    for key in ("usable_fraction", "usable_height_m", "macro_trustworthy",
                "source_px_per_m", "source_usable", "u_axis", "origin_xyz"):
        assert key in dna
    # The record must not decide a material; that is a later, separable stage.
    assert "material_family" not in dna

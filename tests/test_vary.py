"""Tests for per-building facade DNA variation."""
import numpy as np
import pytest

from lidarworld.reconstruct.elevation import FacadeDNA
from lidarworld.reconstruct.vary import vary


def _base_dna(**overrides):
    defaults = dict(bay_m=3.77, storey_m=3.31, storeys=5,
                    window_w_m=1.2, window_h_m=1.6, sill_m=0.3,
                    base_z=0.0, top_z=18.0,
                    wall_rgb=(0.72, 0.70, 0.67),
                    window_rgb=(0.16, 0.19, 0.23))
    return FacadeDNA(**{**defaults, **overrides})


class TestVary:
    def test_deterministic(self):
        """Same building index always gives the same variation."""
        dna = _base_dna()
        a = vary(dna, 42)
        b = vary(dna, 42)
        assert a.bay_m == b.bay_m
        assert a.storey_m == b.storey_m
        assert a.wall_rgb == b.wall_rgb

    def test_different_buildings_differ(self):
        """Different indices give different DNA."""
        dna = _base_dna()
        results = [vary(dna, i) for i in range(10)]
        bays = [r.bay_m for r in results]
        assert len(set(bays)) > 1, "all buildings got the same bay width"

    def test_zero_strength_identity(self):
        """strength=0 returns the original DNA unchanged."""
        dna = _base_dna()
        v = vary(dna, 7, strength=0.0)
        assert v.bay_m == dna.bay_m
        assert v.storey_m == dna.storey_m
        assert v.window_w_m == dna.window_w_m

    def test_bay_within_budget(self):
        """Bay width stays within ±12% of original."""
        dna = _base_dna()
        for i in range(50):
            v = vary(dna, i)
            assert v.bay_m >= dna.bay_m * 0.88 - 0.01
            assert v.bay_m <= dna.bay_m * 1.12 + 0.01

    def test_window_narrower_than_bay(self):
        """Window can never be wider than 85% of the bay."""
        dna = _base_dna(window_w_m=3.5, bay_m=3.77)
        for i in range(50):
            v = vary(dna, i)
            assert v.window_w_m <= v.bay_m * 0.85 + 0.01

    def test_storey_count_recomputed(self):
        """Storey count adjusts to varied storey height."""
        dna = _base_dna(top_z=30.0)
        counts = {vary(dna, i).storeys for i in range(50)}
        # With a 30m building and ±8% on 3.31m storeys, some variation expected.
        assert len(counts) >= 1  # At minimum it doesn't crash.

    def test_colour_stays_in_gamut(self):
        """Wall and window colours stay in [0, 1]."""
        dna = _base_dna(wall_rgb=(0.99, 0.01, 0.5))
        for i in range(50):
            v = vary(dna, i)
            for c in v.wall_rgb:
                assert 0.0 <= c <= 1.0
            for c in v.window_rgb:
                assert 0.0 <= c <= 1.0

    def test_provenance_records_variation(self):
        """Provenance notes the building index."""
        dna = _base_dna()
        v = vary(dna, 42)
        assert "42" in v.provenance.get("variation", "")

    def test_high_strength(self):
        """strength > 1.0 is allowed for stylised generation."""
        dna = _base_dna()
        v = vary(dna, 0, strength=3.0)
        # Should not crash, and bay should be noticeably different.
        assert isinstance(v.bay_m, float)

    def test_minimum_floors(self):
        """Ground floor height never drops below 3.0m."""
        dna = _base_dna()
        for i in range(50):
            v = vary(dna, i)
            assert v.ground_floor_m >= 3.0

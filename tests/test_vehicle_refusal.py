"""Vehicle inference is refused on sparse data, deliberately.

The rule required `density > 1.0` on data whose maximum is 0.98, so it could
never fire and nothing said so. Fixing that silent failure does not produce
vehicles -- it produces kerbs labelled as vehicles -- so the rule now refuses
explicitly, with the measurement behind the refusal.
"""
import numpy as np

from lidarworld.semantics.infer import (VEHICLE_MIN_DENSITY,
                                        _density_supports_vehicles)


def test_airborne_density_does_not_support_vehicles():
    """3DEP over Denver peaks at 0.98 in the vehicle band."""
    hag = np.full(500, 1.2)
    density = np.random.default_rng(0).uniform(0.2, 0.98, 500)
    assert not _density_supports_vehicles(density, hag)


def test_terrestrial_density_does():
    """The original threshold is right for the scans it was written for."""
    hag = np.full(500, 1.2)
    density = np.random.default_rng(1).uniform(5.0, 40.0, 500)
    assert _density_supports_vehicles(density, hag)


def test_a_cloud_with_nothing_in_the_band_refuses():
    hag = np.full(100, 9.0)
    assert not _density_supports_vehicles(np.full(100, 99.0), hag)


def test_the_decision_uses_the_band_not_the_whole_cloud():
    """Dense canopy or facade returns must not vouch for the street."""
    hag = np.concatenate([np.full(400, 12.0), np.full(100, 1.2)])
    density = np.concatenate([np.full(400, 50.0), np.full(100, 0.4)])
    assert not _density_supports_vehicles(density, hag)


def test_the_threshold_is_the_documented_one():
    assert VEHICLE_MIN_DENSITY == 1.0

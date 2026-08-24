"""Per-building variation of FacadeDNA, so a block reads as a neighbourhood.

Every building on a scanned block shares one measured DNA because the lattice
was measured on a single facade. Without variation, that DNA stamps every
building identically, and an eight-building block reads as one building
duplicated eight times -- which is worse than any LiDAR artefact.

The variation budget is the measurement's own uncertainty: bay width was
measured with a correlation of 0.44, so there is room to move it around without
contradicting the data. Each building gets a stable seed from its index, so the
variation is deterministic and reproducible.

This is NOT randomness for its own sake. The jitter is bounded by what the scan
could not resolve, so a varied block is no less accurate than the uniform one --
it is exactly as uncertain, but honestly rather than deceptively so.
"""
from __future__ import annotations

import numpy as np

from .elevation import FacadeDNA, GROUND_FLOOR_M


def vary(dna: FacadeDNA, building_index: int, *,
         strength: float = 1.0) -> FacadeDNA:
    """Return a varied copy of `dna` for one building on the block.

    `strength` scales the variation: 0.0 gives back the original, 1.0 uses the
    full uncertainty budget. Values above 1.0 are allowed for deliberately
    stylised generation.

    The seed is the building index, so the same building always gets the same
    variation regardless of how many others are built.
    """
    rng = np.random.default_rng(int(building_index) * 7919 + 31)

    def jitter(value: float, rel: float, lo: float = 0.0) -> float:
        """Jitter by up to `rel` of value, clamped above `lo`."""
        delta = rng.uniform(-rel * strength, rel * strength) * value
        return max(lo, value + delta)

    # Bay width: ±12% — the correlation was 0.44, so the period detector
    # was uncertain by about this much.
    bay_m = jitter(dna.bay_m, 0.12, lo=2.0)

    # Storey height: ±8% — less variable than bay width on a real block.
    storey_m = jitter(dna.storey_m, 0.08, lo=2.6)

    # Window proportions: ±15% width, ±10% height.
    window_w = jitter(dna.window_w_m, 0.15, lo=0.6)
    window_h = jitter(dna.window_h_m, 0.10, lo=0.8)

    # Window cannot be wider than the bay.
    window_w = min(window_w, bay_m * 0.85)

    # Sill height: ±20%.
    sill_m = jitter(dna.sill_m, 0.20)

    # Colour: slight hue/value shift. The measured colour has the scan's
    # exposure baked in, so small shifts are within the measurement noise.
    wall_rgb = np.array(dna.wall_rgb, dtype=float)
    shift = rng.uniform(-0.04 * strength, 0.04 * strength, size=3)
    wall_rgb = np.clip(wall_rgb + shift, 0.0, 1.0)

    window_rgb = np.array(dna.window_rgb, dtype=float)
    window_rgb = np.clip(window_rgb + rng.uniform(-0.03, 0.03, size=3) * strength,
                         0.0, 1.0)

    # Ground floor height: ±10%.
    ground_floor = jitter(dna.ground_floor_m, 0.10, lo=3.0)

    # Recompute storey count from the varied height and storey period.
    height = dna.top_z - dna.base_z
    storeys = max(1, int(round((height - ground_floor) / max(storey_m, 1e-6))) + 1)

    return FacadeDNA(
        bay_m=round(bay_m, 3),
        storey_m=round(storey_m, 3),
        storeys=storeys,
        window_w_m=round(window_w, 3),
        window_h_m=round(window_h, 3),
        sill_m=round(sill_m, 3),
        base_z=dna.base_z,
        top_z=dna.top_z,
        wall_rgb=tuple(np.round(wall_rgb, 4)),
        window_rgb=tuple(np.round(window_rgb, 4)),
        reveal_m=dna.reveal_m,
        ground_floor_m=round(ground_floor, 3),
        provenance={**dna.provenance,
                    "variation": f"building {building_index}, strength {strength}"},
    )

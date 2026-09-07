"""Generated windows on walls that were never measured.

Denver's facades are 100% extruded: 3DEP sees a wall edge-on from an aircraft
and gets almost nothing, so the opening *detector* -- which reads holes in the
returns -- found 2 openings across 2,794,075 tiles. A city of windowless brick
slabs is not more honest than a fenestrated one; both are inventions, and only
one looks like a building. What matters is that the invention is labelled.
"""
from __future__ import annotations

import numpy as np

from lidarworld.reconstruct.extrude import walls_from_footprint
from lidarworld.reconstruct.fenestrate import _seed_of, fenestrate
from lidarworld.reconstruct.lattice import build_solid
from lidarworld.roles.taxonomy import Ctx


def wall(width: float = 24.0, height: float = 16.0, cell: float = 0.3):
    ring = np.array([[0, 0], [width, 0], [width, width], [0, width], [0, 0]], dtype=float)
    patch = walls_from_footprint(ring, 0.0, height)[0]
    return patch, build_solid(patch, patch.extent[0], patch.extent[1], cell=cell)


def test_a_blank_extruded_wall_gets_windows():
    patch, lattice = wall()
    solid_before = lattice.solid_count

    added = fenestrate(lattice, patch)

    assert added > 0
    assert len(lattice.openings) == added
    assert lattice.solid_count < solid_before, "occupancy must actually be cut"
    assert (lattice.context & int(Ctx.OPENING_BOUNDARY)).any()
    assert (lattice.context & int(Ctx.NEAR_OPENING)).any(), "no reveal for a theme to trim"


def test_every_generated_opening_says_it_was_generated():
    """The whole licence for inventing these is that they are labelled."""
    patch, lattice = wall()
    fenestrate(lattice, patch)

    assert lattice.openings
    for opening in lattice.openings:
        assert opening.generated is True
        assert opening.confidence < 0.5, "nothing was measured; do not claim otherwise"


def test_the_detector_never_claims_generated():
    """A detected opening and a generated one must stay distinguishable."""
    from lidarworld.reconstruct.lattice import Opening
    detected = Opening(id=0, role="opening.window", cells=np.zeros((1, 2), dtype=np.int64),
                       uv_min=(0, 0), uv_max=(1, 1), center_world=np.zeros(3),
                       width=1.0, height=1.0, sill_height=1.0, confidence=0.8)
    assert detected.generated is False


def test_the_same_building_regenerates_the_same_facade():
    """Across processes, not just within one.

    The previous implementation seeded from `hash()`, which is salted per
    interpreter for strings, so a building keyed by its register id got a
    different facade on every run -- and each run looked equally plausible,
    which is how it went unnoticed.
    """
    assert _seed_of("NL.IMBAG.0363100012169587", 4) == _seed_of("NL.IMBAG.0363100012169587", 4)
    assert _seed_of("a", 1) != _seed_of("b", 1)

    def facade(key):
        patch, lattice = wall()
        fenestrate(lattice, patch, key=key)
        return lattice.occupancy.copy()

    assert np.array_equal(facade("bag-1"), facade("bag-1"))
    assert not np.array_equal(facade("bag-1"), facade("bag-2")), \
        "every building the same is a spreadsheet, not a street"


def test_a_party_wall_stays_blank():
    """You cannot put a window where the next building is."""
    patch, lattice = wall()
    patch.attrs["party_wall"] = True
    assert fenestrate(lattice, patch) == 0
    assert lattice.solid_count == lattice.occupancy.size


def test_a_garden_wall_is_not_a_facade():
    patch, lattice = wall(width=24.0, height=2.5)
    assert fenestrate(lattice, patch) == 0


def test_a_roof_is_never_fenestrated():
    patch, lattice = wall()
    patch.role = "surface.roof.flat"
    assert fenestrate(lattice, patch) == 0


def test_openings_stay_inside_the_wall():
    """An opening whose cells fall outside the lattice is a crash downstream."""
    patch, lattice = wall()
    fenestrate(lattice, patch)
    nu, nv = lattice.occupancy.shape

    for opening in lattice.openings:
        assert opening.cells[:, 0].min() >= 0 and opening.cells[:, 0].max() < nu
        assert opening.cells[:, 1].min() >= 0 and opening.cells[:, 1].max() < nv
        assert opening.width > 0 and opening.height > 0
        # Nothing may be cut below the pavement or above the parapet.
        assert 0 <= opening.sill_height <= nv * lattice.cell


def test_a_taller_building_gets_more_storeys_of_windows():
    """The rhythm has to come from the measured envelope, not a constant."""
    def storey_span(height):
        patch, lattice = wall(height=height)
        fenestrate(lattice, patch, key="same")
        return {round(o.sill_height, 1) for o in lattice.openings}

    assert len(storey_span(30.0)) > len(storey_span(10.0))

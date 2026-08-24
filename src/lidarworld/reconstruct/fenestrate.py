"""Windows on a facade nobody measured.

There are two ways an opening gets into the IR and they must not be confused.

`lattice.build` *detects* one: glass returns nothing at 905 nm, so an enclosed
hole in a wall's returns is a window, and the evidence for it is the absence
itself. That opening is observed.

This module *generates* one. An extruded wall exists because a footprint and a
roof height were measured and the facade between them was not -- airborne LiDAR
sees a wall edge-on from an aircraft and gets almost nothing. There is no
evidence to be faithful to, and none to contradict. A blank extruded wall is
not more honest than a fenestrated one, it is just blanker: both are inventions,
and only one of them looks like a building.

So everything here is marked generated, never observed, and the wall it lands on
is already flagged OCCLUDED and SPARSE_EVIDENCE. Forward validation can exclude
it, the survey theme can paint it as unmeasured, and `describe()` can say which
of the city was real.

The rhythm comes from the building's own measured envelope -- its height decides
the storey count, its width the bay count -- so what is invented stays inside
what was measured.
"""
from __future__ import annotations

import zlib

import numpy as np

from ..roles.taxonomy import Ctx
from .lattice import Opening, TileLattice

#: Floor to floor. Not a measurement of anything -- airborne returns never saw a
#: single storey line -- but the spacing a generated facade has to be plausible
#: at. Denver's warehouse stock runs taller than a house, which is what this is.
STOREY_M = 3.8

#: A facade smaller than this is a garden wall, a parapet return or a lift
#: overrun. Glazing them is what makes a block look like a doll's house.
MIN_FACADE_M = (3.0, 4.0)           # width, height


def _seed_of(*parts) -> int:
    """A seed that survives the process it was made in.

    `hash()` on a str is salted per interpreter unless PYTHONHASHSEED is set,
    so seeding an RNG with it gives a *different* street on every run -- which
    is the opposite of the reproducibility the seed exists to provide, and
    silently so: each run looks equally plausible.
    """
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0xFFFFFFFF


def fenestrate(lattice: TileLattice, patch, *, key=None,
               storey_m: float = STOREY_M) -> int:
    """Cut generated windows and doors into one wall's lattice.

    Returns the number of openings added. They are appended to
    `lattice.openings` with `generated=True`, so they become real IR nodes and
    are counted like any other -- an opening that only edits occupancy is
    invisible to every consumer downstream.
    """
    if not patch.role.startswith("surface.wall"):
        return 0
    if patch.attrs.get("party_wall"):
        # Built hard against the neighbour. In a terrace these are blank by
        # construction -- there is no outside for a window to look at -- and
        # glazing them makes a block read as a free-standing office park.
        return 0

    occupancy, context = lattice.occupancy, lattice.context
    nu, nv = occupancy.shape
    cell = lattice.cell
    width_m, height_m = nu * cell, nv * cell
    if width_m < MIN_FACADE_M[0] or height_m < MIN_FACADE_M[1]:
        return 0

    rng = np.random.default_rng(_seed_of(key if key is not None else patch.id, patch.id))
    storeys = max(1, int(round(height_m / storey_m)))

    # A window every 3 m on every wall of every building reads as a
    # spreadsheet, not a street. Varying the bay per building is most of what
    # stops a row of blocks looking stamped.
    bay_m = float(rng.uniform(4.2, 7.0))
    glass_fraction = float(rng.uniform(0.30, 0.45))
    win_w = max(2, int(round(bay_m * glass_fraction / cell)))
    win_h = max(2, int(round(rng.uniform(1.5, 2.1) / cell)))
    pitch = max(win_w + 2, int(round(bay_m / cell)))
    if pitch >= nu:
        return 0

    uv_min = lattice.uv_origin
    margin = max(1, (nu % pitch) // 2)
    added = 0

    for storey in range(storeys):
        floor_v = int(round(storey * storey_m / cell))
        # The ground storey is taller and its opening starts lower, which is
        # what makes a shopfront read differently from a flat above it.
        sill_v = floor_v + int(round((1.5 if storey else 0.8) / cell))
        top_v = sill_v + (win_h if storey else int(round(2.4 / cell)))
        if top_v >= nv - 1:
            break

        for u0 in range(margin, nu - win_w, pitch):
            if storey and rng.random() < 0.10:
                continue            # a blank bay; perfect regularity reads as CGI
            u1, v0, v1 = u0 + win_w, sill_v, top_v
            occupancy[u0:u1, v0:v1] = 0
            # The reveal, so a theme can put a surround or a lintel on it.
            context[max(0, u0 - 1):min(nu, u1 + 1),
                    max(0, v0 - 1):min(nv, v1 + 1)] |= int(Ctx.NEAR_OPENING)
            context[u0:u1, v0:v1] |= int(Ctx.OPENING_BOUNDARY)

            cells = np.array([(u, v) for u in range(u0, u1) for v in range(v0, v1)],
                             dtype=np.int64)
            centre_uv = np.array([[uv_min[0] + (u0 + u1) / 2 * cell,
                                   uv_min[1] + (v0 + v1) / 2 * cell]])
            sill_m = v0 * cell
            opening = Opening(
                id=len(lattice.openings),
                role="opening.door" if sill_m < 0.5 else "opening.window",
                cells=cells,
                uv_min=(float(uv_min[0] + u0 * cell), float(uv_min[1] + v0 * cell)),
                uv_max=(float(uv_min[0] + u1 * cell), float(uv_min[1] + v1 * cell)),
                center_world=patch.unproject(centre_uv)[0],
                width=float(win_w * cell), height=float((v1 - v0) * cell),
                sill_height=float(sill_m),
                # Not the detector's fill-ratio confidence. Nothing was
                # measured, so this is the confidence that a building of this
                # envelope has windows at all -- high -- and says nothing about
                # these windows being in the right place.
                confidence=0.2,
                generated=True,
            )
            lattice.openings.append(opening)
            added += 1

    return added

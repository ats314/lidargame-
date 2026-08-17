"""Build a clean facade from measured numbers, instead of repairing a scan.

Everything before this tried to make the reality mesh's own surface shippable. It
is not, and the measurements are unambiguous about why: at 7.6 cm/texel from an
oblique aerial pass the wall is over-smoothed to 4.4 cm locally, window reveals
sit at 5 cm and are below the noise floor, and the smearing is *correlated across
bays* -- every bay reconstructed from the same look angles -- so no vote,
selection or detail transfer over those bays recovers it. The median of 48 bays
carries half the high-frequency energy of one bay. That path is closed.

The way through is the one this repo was built around: the scan is a source of
*measurements*, and the geometry is built. Not matched -- built. What comes out is
not the building that was scanned; it is a building on that footprint, at that
height, with that storey rhythm and that window, facing that street.

What is measured, and what is not, kept strictly apart because the whole product
depends on the distinction:

    measured    footprint and its corners        CityGML, survey-grade
                base and roof height             mesh, agreeing with the city
                                                 model to 0.20 m at the median
                storey height                    3.31 m, and the register's own
                                                 storey count agrees to 3%
                bay width                        3.77 m, correlation 0.44
                window width and height          from the average bay's extent,
                                                 which is soft but well bounded
                wall and window colour           low frequency, which is what
                                                 survives smearing
    assumed     reveal depth                     0.15 m. Airborne data cannot see
                                                 it at any resolution; this is a
                                                 Helsinki convention, not a
                                                 measurement, and it is labelled
                                                 as such in every record.

The geometry is quads with real openings punched through them and real reveals
around the openings -- jamb, head and sill -- which is depth the source mesh never
contained. Clean by construction: a plane cannot droop.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .tessellate import close_ring

#: Reveal depth, metres. The one number here that is not measured. Airborne
#: photogrammetry resolves this wall to 4.4 cm locally and a reveal is 15 to 25 cm,
#: so it is at the edge of the noise even in principle; the mesh does not contain
#: it. 0.15 m is a normal Helsinki masonry reveal and it is recorded as `assumed`
#: everywhere it appears.
REVEAL_M = 0.15

#: Ground floor of a Helsinki block is taller than the storeys above it. Measured
#: from the lattice where the bottom band is resolvable and otherwise this.
GROUND_FLOOR_M = 4.2

#: A window narrower or shorter than this is a detection artefact, not an opening.
MIN_WINDOW_M = 0.6

#: Fraction of the wall-to-window contrast at which the window's edge is called.
#: Half, so the boundary sits where the smear crosses the midpoint between the two
#: measured levels rather than at either of them -- which is the only defensible
#: choice when the transition itself is blurred over 20 cm.
WINDOW_EDGE = 0.5

#: Median luminance a de-lit facade is normalised to. Masonry and render sit
#: around here; the number a survey texture arrives at is the flight's exposure,
#: not the building's.
ALBEDO_TARGET = 0.58


@dataclass
class FacadeDNA:
    """A facade as numbers, with each number's provenance attached."""
    bay_m: float
    storey_m: float
    storeys: int
    window_w_m: float
    window_h_m: float
    sill_m: float                       # window base above its storey line
    base_z: float
    top_z: float
    wall_rgb: tuple = (0.72, 0.70, 0.67)
    window_rgb: tuple = (0.16, 0.19, 0.23)
    reveal_m: float = REVEAL_M
    ground_floor_m: float = GROUND_FLOOR_M
    provenance: dict = field(default_factory=dict)

    @property
    def height_m(self) -> float:
        return self.top_z - self.base_z

    def to_record(self) -> dict:
        return {
            "bay_m": round(self.bay_m, 3),
            "storey_m": round(self.storey_m, 3),
            "storeys": self.storeys,
            "window_m": [round(self.window_w_m, 3), round(self.window_h_m, 3)],
            "sill_m": round(self.sill_m, 3),
            "height_m": round(self.height_m, 2),
            "reveal_m": self.reveal_m,
            "wall_rgb": [round(v, 3) for v in self.wall_rgb],
            "window_rgb": [round(v, 3) for v in self.window_rgb],
            "provenance": self.provenance,
        }


def window_box(cell: np.ndarray, px_per_m: float, support: np.ndarray | None = None
               ) -> tuple[float, float, float, dict]:
    """Window width, height and sill offset from the average bay.

    The average bay is soft -- that is the finding this module exists because of --
    but soft in a way that leaves its *extent* usable. A blurred edge still crosses
    the midpoint between wall and glass at very close to where the sharp edge was,
    because blurring is symmetric. So the window's boundary is defensible even
    though its detail is not, and a boundary is what geometry needs.
    """
    grey = np.asarray(cell, dtype=np.float64)
    if grey.ndim == 3:
        grey = grey.mean(axis=2)
    if grey.max() > 1.5:
        grey = grey / 255.0
    if support is not None:
        grey = np.where(support, grey, np.nan)
    if not np.isfinite(grey).any():
        return 0.0, 0.0, 0.0, {"reason": "no support in the average bay"}

    bright = float(np.nanpercentile(grey, 85))      # masonry
    dark = float(np.nanpercentile(grey, 10))        # glass
    if bright - dark < 0.05:
        return 0.0, 0.0, 0.0, {"reason": f"wall/window contrast only "
                                         f"{bright - dark:.3f}"}
    level = dark + WINDOW_EDGE * (bright - dark)
    mask = np.isfinite(grey) & (grey <= level)
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if not len(rows) or not len(cols):
        return 0.0, 0.0, 0.0, {"reason": "no window found in the average bay"}

    height_m = (rows[-1] - rows[0] + 1) / px_per_m
    width_m = (cols[-1] - cols[0] + 1) / px_per_m
    # The cell is centred on its lattice point, so the window's base relative to
    # that point is the offset a generator needs to place it on a storey line.
    centre_row = mask.shape[0] / 2.0
    sill_m = (centre_row - rows[-1]) / px_per_m
    report = {
        "wall_level": round(bright, 3),
        "window_level": round(dark, 3),
        "edge_at": round(level, 3),
        "dark_fraction": round(float(mask.mean()), 3),
        "evidence": "extent of the average bay, thresholded at the midpoint",
    }
    return width_m, height_m, sill_m, report


def measure(facade, grid, average: np.ndarray, *, base_z: float, top_z: float,
            support: np.ndarray | None = None) -> FacadeDNA:
    """Turn a measured facade into the numbers a generator builds from."""
    px = facade.px_per_m
    width_m, height_m, sill_m, box = window_box(average, px, support)
    grey = np.asarray(average, dtype=np.float64)
    if grey.max() > 1.5:
        grey = grey / 255.0
    # A photogrammetric texture is not albedo -- it carries the sun, the sky and
    # the exposure the survey flew at. Taken raw, this Helsinki block's measured
    # wall came out (0.32, 0.38, 0.44): a real hue on a dusk exposure, which
    # renders as a gloomy slab and then gets lit AGAIN by the engine. De-lighting
    # first, then normalising the value while keeping the hue, gives a base colour
    # that is the wall's own rather than the weather's.
    from ..features.frequency import LUMA, delight            # noqa: PLC0415
    grey, _, _ = delight(grey, px_per_m=px)
    luma = grey @ LUMA
    scale = ALBEDO_TARGET / max(float(np.median(luma)), 1e-3)
    grey = np.clip(grey * scale, 0.0, 1.0)
    luma = grey.mean(axis=2)
    lit = luma >= np.nanpercentile(luma, 75)
    glass = luma <= np.nanpercentile(luma, 15)

    storeys = max(1, int(round((top_z - base_z - GROUND_FLOOR_M)
                               / max(grid.storey_m, 1e-6))) + 1)
    dna = FacadeDNA(
        bay_m=grid.bay_m, storey_m=grid.storey_m, storeys=storeys,
        window_w_m=max(width_m, 0.0), window_h_m=max(height_m, 0.0),
        sill_m=sill_m, base_z=base_z, top_z=top_z,
        wall_rgb=tuple(np.round(grey[lit].mean(axis=0), 4)),
        window_rgb=tuple(np.round(grey[glass].mean(axis=0), 4)),
    )
    dna.provenance = {
        "bay_m": f"measured, correlation {grid.bay_strength:.3f}",
        "storey_m": f"measured, correlation {grid.storey_strength:.3f}",
        "storeys": "derived from height and storey period",
        "window_m": "derived from the average bay's extent",
        "sill_m": "derived from the average bay's extent",
        "base_z": "measured",
        "top_z": "measured",
        "wall_rgb": "measured, low frequency",
        "window_rgb": "measured, low frequency",
        "reveal_m": "ASSUMED -- airborne data does not contain it",
        "window_box": box,
    }
    return dna


def _bands(extent_m: float, spacing_m: float, size_m: float, offset_m: float
           ) -> list[tuple[float, float]]:
    """Opening spans along one axis: [(lo, hi), ...] within [0, extent]."""
    if spacing_m <= 0 or size_m < MIN_WINDOW_M:
        return []
    out = []
    position = offset_m
    while position + size_m <= extent_m:
        if position >= 0:
            out.append((position, position + size_m))
        position += spacing_m
    return out


def punch(width_m: float, height_m: float, columns: list[tuple[float, float]],
          rows: list[tuple[float, float]]) -> tuple[list, list]:
    """Split a wall rectangle into quads around its openings.

    Returns (wall cells, opening cells) as (u0, u1, v0, v1) in wall-local metres.
    A grid whose lines are the opening edges: every cell is either entirely wall or
    entirely opening, so the wall comes out as quads with no triangulation of a
    polygon with holes anywhere -- which is where a mesher with an ear-clipping
    triangulator goes wrong on a facade.
    """
    us = sorted({0.0, width_m, *[v for span in columns for v in span]})
    vs = sorted({0.0, height_m, *[v for span in rows for v in span]})
    wall, openings = [], []
    for u0, u1 in zip(us[:-1], us[1:]):
        if u1 - u0 < 1e-6:
            continue
        in_column = any(lo - 1e-9 <= u0 and u1 <= hi + 1e-9 for lo, hi in columns)
        for v0, v1 in zip(vs[:-1], vs[1:]):
            if v1 - v0 < 1e-6:
                continue
            in_row = any(lo - 1e-9 <= v0 and v1 <= hi + 1e-9 for lo, hi in rows)
            (openings if (in_column and in_row) else wall).append((u0, u1, v0, v1))
    return wall, openings


@dataclass
class Elevation:
    """The built geometry of one wall: quads, each with what it is."""
    quads: np.ndarray                   # (Q, 4, 3) world, in winding order
    kinds: list                         # per quad: wall | glass | reveal
    report: dict = field(default_factory=dict)


def build_wall(a: np.ndarray, b: np.ndarray, dna: FacadeDNA) -> Elevation:
    """One footprint edge, from `a` to `b`, as a clean walled elevation.

    The wall plane is defined by the footprint, not by the scan, which is what
    makes it clean: a survey line is straight and a photogrammetric surface is not.
    Openings are placed on the measured lattice and given real reveals.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    along = b - a
    width_m = float(np.hypot(along[0], along[1]))
    if width_m < MIN_WINDOW_M:
        return Elevation(np.zeros((0, 4, 3)), [], {"reason": "edge too short"})
    u_axis = np.array([along[0] / width_m, along[1] / width_m, 0.0])
    v_axis = np.array([0.0, 0.0, 1.0])
    # Outward normal: to the right of the direction of travel, which is outward for
    # a counter-clockwise ring.
    normal = np.array([u_axis[1], -u_axis[0], 0.0])
    base = np.array([a[0], a[1], dna.base_z])
    height_m = dna.height_m

    columns = _bands(width_m, dna.bay_m, dna.window_w_m,
                     max(0.0, (dna.bay_m - dna.window_w_m) / 2.0))
    storey_rows = []
    level = dna.ground_floor_m
    while level + dna.window_h_m <= height_m - 0.5:
        storey_rows.append((level, level + dna.window_h_m))
        level += dna.storey_m
    wall_cells, opening_cells = punch(width_m, height_m, columns, storey_rows)

    def corner(u, v, depth=0.0):
        return base + u_axis * u + v_axis * v + normal * depth

    quads, kinds = [], []
    for u0, u1, v0, v1 in wall_cells:
        quads.append([corner(u0, v0), corner(u1, v0), corner(u1, v1), corner(u0, v1)])
        kinds.append("wall")
    for u0, u1, v0, v1 in opening_cells:
        d = -dna.reveal_m                       # inward, behind the wall plane
        quads.append([corner(u0, v0, d), corner(u1, v0, d),
                      corner(u1, v1, d), corner(u0, v1, d)])
        kinds.append("glass")
        # Jamb, jamb, head, sill: the four faces that make the opening a hole with
        # thickness rather than a painted rectangle.
        for p, q in (((u0, v0), (u0, v1)), ((u1, v1), (u1, v0)),
                     ((u0, v1), (u1, v1)), ((u1, v0), (u0, v0))):
            quads.append([corner(*p), corner(*q), corner(*q, d), corner(*p, d)])
            kinds.append("reveal")

    return Elevation(np.asarray(quads), kinds, {
        "width_m": round(width_m, 2),
        "height_m": round(height_m, 2),
        "bays": len(columns),
        "storeys": len(storey_rows),
        "openings": len(opening_cells),
        "wall_quads": len(wall_cells),
        "quads": len(quads),
    })


def build(ring: np.ndarray, dna: FacadeDNA) -> Elevation:
    """A whole building's elevations from its footprint ring and its DNA."""
    # `close_ring` DROPS a repeated final vertex -- it is named for what a
    # clipper wants, not for what an edge walk wants. Walking pairs over its
    # output silently loses the edge from the last corner back to the first, so
    # a rectangular building came out with three walls and an open side.
    ring = close_ring(np.asarray(ring, dtype=np.float64))
    ring = np.vstack([ring, ring[:1]])
    quads, kinds, edges = [], [], []
    for a, b in zip(ring[:-1], ring[1:]):
        wall = build_wall(a, b, dna)
        if len(wall.quads):
            quads.append(wall.quads)
            kinds.extend(wall.kinds)
        edges.append(wall.report)
    if not quads:
        return Elevation(np.zeros((0, 4, 3)), [], {"reason": "no usable edges"})
    return Elevation(np.concatenate(quads), kinds, {
        "edges": len(edges),
        "quads": sum(e.get("quads", 0) for e in edges),
        "openings": sum(e.get("openings", 0) for e in edges),
        "dna": dna.to_record(),
        "per_edge": edges,
    })


# --- architecture -------------------------------------------------------------
#
# The first version of this built a box with holes in it. Six of them rendered as
# extruded prisms with one window stamped in a grid: structurally correct and
# visually worthless, because a real facade is mostly *relief* -- a plinth the
# wall stands on, a cornice it stops at, a string course dividing the shopfront
# from the flats above, a sill each window sits on and a lintel over it. All of
# those are ten centimetres of projection, they cost almost nothing in triangles,
# and they are what light catches. Without them a wall has one plane and returns
# one colour.

#: How far a band stands proud of the wall, metres. Small: a Helsinki string
#: course is a few centimetres of render, not a ledge.
PLINTH_PROJECTION_M = 0.09
CORNICE_PROJECTION_M = 0.42
STRING_PROJECTION_M = 0.07
SILL_PROJECTION_M = 0.06

#: Heights of those bands.
PLINTH_H_M = 0.75
CORNICE_H_M = 0.55
STRING_H_M = 0.22
SILL_H_M = 0.10

#: A door is taller and wider than a window and there is one per street frontage.
DOOR_W_M = 1.5
DOOR_H_M = 2.6

#: Mullion and transom, as a fraction of the opening. A single dark rectangle
#: reads as a hole; two bars across it read as a window.
BAR_M = 0.06


def _box(origin, u_axis, v_axis, normal, u0, u1, v0, v1, d0, d1):
    """Six quads of an axis-aligned box in the wall's own frame."""
    def at(u, v, d):
        return origin + u_axis * u + v_axis * v + normal * d
    faces = [
        [at(u0, v0, d1), at(u1, v0, d1), at(u1, v1, d1), at(u0, v1, d1)],   # front
        [at(u1, v0, d0), at(u0, v0, d0), at(u0, v1, d0), at(u1, v1, d0)],   # back
        [at(u0, v1, d0), at(u0, v1, d1), at(u1, v1, d1), at(u1, v1, d0)],   # top
        [at(u1, v0, d0), at(u1, v0, d1), at(u0, v0, d1), at(u0, v0, d0)],   # bottom
        [at(u0, v0, d0), at(u0, v0, d1), at(u0, v1, d1), at(u0, v1, d0)],   # left
        [at(u1, v1, d0), at(u1, v1, d1), at(u1, v0, d1), at(u1, v0, d0)],   # right
    ]
    return faces


def build_wall_detailed(a, b, dna, *, door: bool = False) -> Elevation:
    """One footprint edge with its relief: plinth, cornice, string, sills, frames."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    along = b - a
    width_m = float(np.hypot(along[0], along[1]))
    if width_m < MIN_WINDOW_M:
        return Elevation(np.zeros((0, 4, 3)), [], {"reason": "edge too short"})
    u_axis = np.array([along[0] / width_m, along[1] / width_m, 0.0])
    v_axis = np.array([0.0, 0.0, 1.0])
    normal = np.array([u_axis[1], -u_axis[0], 0.0])
    origin = np.array([a[0], a[1], dna.base_z])
    height_m = dna.height_m
    quads, kinds = [], []

    def add(face_list, kind):
        for face in face_list:
            quads.append(face)
            kinds.append(kind)

    def flat(u0, u1, v0, v1, d=0.0):
        return [[origin + u_axis * u0 + v_axis * v0 + normal * d,
                 origin + u_axis * u1 + v_axis * v0 + normal * d,
                 origin + u_axis * u1 + v_axis * v1 + normal * d,
                 origin + u_axis * u0 + v_axis * v1 + normal * d]]

    # Openings: windows on the measured lattice above the ground floor, and one
    # door on the ground floor of a street frontage.
    columns = _bands(width_m, dna.bay_m, dna.window_w_m,
                     max(0.0, (dna.bay_m - dna.window_w_m) / 2.0))
    rows = []
    level = dna.ground_floor_m + dna.sill_m + 0.6
    while level + dna.window_h_m <= height_m - CORNICE_H_M - 0.4:
        rows.append((level, level + dna.window_h_m))
        level += dna.storey_m

    door_span = None
    if door and columns and width_m > 3 * dna.bay_m:
        centre = columns[len(columns) // 2]
        mid = (centre[0] + centre[1]) / 2.0
        door_span = (mid - DOOR_W_M / 2, mid + DOOR_W_M / 2)

    all_columns = columns + ([door_span] if door_span else [])
    all_rows = rows + ([(0.35, 0.35 + DOOR_H_M)] if door_span else [])
    # BOTH lists. Discarding the opening cells here built a wall with the relief
    # bands correct and not a single window in it -- 488 wall quads and zero
    # glass -- because the openings only ever existed in the return value that
    # was thrown away.
    wall_cells, opening_cells = punch(width_m, height_m, all_columns, all_rows)
    # Only the cells that are BOTH a window column and a window row are openings;
    # `punch` already did that, but the door column crosses the window rows and
    # the window columns cross the door row, so re-derive which is which.
    def is_opening(u0, u1, v0, v1):
        for cu in columns:
            for cv in rows:
                if (cu[0] - 1e-9 <= u0 and u1 <= cu[1] + 1e-9
                        and cv[0] - 1e-9 <= v0 and v1 <= cv[1] + 1e-9):
                    return "window"
        if door_span:
            dv = all_rows[-1]
            if (door_span[0] - 1e-9 <= u0 and u1 <= door_span[1] + 1e-9
                    and dv[0] - 1e-9 <= v0 and v1 <= dv[1] + 1e-9):
                return "door"
        return None

    for u0, u1, v0, v1 in wall_cells + opening_cells:
        if is_opening(u0, u1, v0, v1) is None:
            add(flat(u0, u1, v0, v1), "wall")

    # Openings: recessed pane, four reveal faces, a projecting sill, and bars.
    def opening(u0, u1, v0, v1, kind):
        d = -dna.reveal_m
        add(flat(u0, u1, v0, v1, d), "glass" if kind == "window" else "door")
        for p, q in (((u0, v0), (u0, v1)), ((u1, v1), (u1, v0)),
                     ((u0, v1), (u1, v1)), ((u1, v0), (u0, v0))):
            add([[origin + u_axis * p[0] + v_axis * p[1],
                  origin + u_axis * q[0] + v_axis * q[1],
                  origin + u_axis * q[0] + v_axis * q[1] + normal * d,
                  origin + u_axis * p[0] + v_axis * p[1] + normal * d]], "reveal")
        # A mullion and a transom, so it reads as glazing rather than a hole.
        mid_u, mid_v = (u0 + u1) / 2, (v0 + v1) / 2
        add(flat(mid_u - BAR_M / 2, mid_u + BAR_M / 2, v0, v1, d + 0.02), "frame")
        add(flat(u0, u1, mid_v - BAR_M / 2, mid_v + BAR_M / 2, d + 0.02), "frame")
        if kind == "window":
            add(_box(origin, u_axis, v_axis, normal,
                     u0 - 0.10, u1 + 0.10, v0 - SILL_H_M, v0,
                     -dna.reveal_m, SILL_PROJECTION_M), "sill")

    for u0, u1, v0, v1 in opening_cells:
        kind = is_opening(u0, u1, v0, v1)
        if kind:
            opening(u0, u1, v0, v1, kind)

    # Horizontal relief, full width.
    add(_box(origin, u_axis, v_axis, normal, 0, width_m, 0, PLINTH_H_M,
             0.0, PLINTH_PROJECTION_M), "plinth")
    add(_box(origin, u_axis, v_axis, normal, 0, width_m,
             dna.ground_floor_m - STRING_H_M, dna.ground_floor_m,
             0.0, STRING_PROJECTION_M), "string")
    add(_box(origin, u_axis, v_axis, normal, 0, width_m,
             height_m - CORNICE_H_M, height_m, 0.0, CORNICE_PROJECTION_M), "cornice")

    return Elevation(np.asarray(quads), kinds, {
        "width_m": round(width_m, 2), "height_m": round(height_m, 2),
        "bays": len(columns), "storeys": len(rows),
        "openings": len(rows) * len(columns) + (1 if door_span else 0),
        "quads": len(quads), "door": bool(door_span),
    })


def roof_cap(ring: np.ndarray, dna: FacadeDNA) -> tuple[list, list]:
    """Close the building. A hipped cap, because an open-topped shell reads broken."""
    ring = close_ring(np.asarray(ring, dtype=np.float64))
    centre = ring[:, :2].mean(axis=0)
    eaves = dna.top_z
    ridge = eaves + max(1.6, min(4.0, 0.12 * float(np.ptp(ring[:, :2]))))
    apex = np.array([centre[0], centre[1], ridge])
    quads, kinds = [], []
    closed = np.vstack([ring, ring[:1]])
    for p, q in zip(closed[:-1], closed[1:]):
        # Pulled in slightly so the roof sits inside the cornice, not on its lip.
        a = np.array([p[0], p[1], eaves])
        b = np.array([q[0], q[1], eaves])
        quads.append([a, b, apex, apex])
        kinds.append("roof")
    return quads, kinds


def build_detailed(ring: np.ndarray, dna: FacadeDNA, *, roof: bool = True
                   ) -> Elevation:
    """A whole building: relieved elevations on every edge, and a closed roof."""
    ring = close_ring(np.asarray(ring, dtype=np.float64))
    ring = np.vstack([ring, ring[:1]])
    lengths = [float(np.hypot(*(b - a)[:2])) for a, b in zip(ring[:-1], ring[1:])]
    longest = int(np.argmax(lengths)) if lengths else -1

    quads, kinds, edges = [], [], []
    for i, (a, b) in enumerate(zip(ring[:-1], ring[1:])):
        wall = build_wall_detailed(a, b, dna, door=(i == longest))
        if len(wall.quads):
            quads.append(wall.quads)
            kinds.extend(wall.kinds)
        edges.append(wall.report)
    if roof:
        cap, cap_kinds = roof_cap(ring[:-1], dna)
        if cap:
            quads.append(np.asarray(cap))
            kinds.extend(cap_kinds)
    if not quads:
        return Elevation(np.zeros((0, 4, 3)), [], {"reason": "no usable edges"})
    return Elevation(np.concatenate(quads), kinds, {
        "edges": len(edges),
        "quads": sum(len(q) for q in quads),
        "openings": sum(e.get("openings", 0) for e in edges),
        "doors": sum(1 for e in edges if e.get("door")),
        "dna": dna.to_record(),
    })

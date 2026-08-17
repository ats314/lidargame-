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

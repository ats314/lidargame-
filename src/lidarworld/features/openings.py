"""Find windows on a facade from depth and from rhythm, not from a segmenter.

Three cities in, no open city model anywhere carries a facade opening as
geometry. Hamburg's LoD3 has zero `bldg:Window`; Helsinki's CityGML has 34,260
wall surfaces and zero windows. So a window has to be found, and there are two
pieces of evidence that beat guessing from pixels.

**Depth.** A photogrammetric mesh resolves relief to about 13 cm RMS, and a
window is *recessed*. The rectifier already builds a depth buffer to resolve
overlaps, so a reveal is a measured step in that buffer rather than an inference
from a 10 px/m photograph in which a 1.2 m window is twelve pixels wide.

**Rhythm.** A facade repeats. On a Helsinki Kontorhaus the bay spacing measures
1.50 m with harmonics at 2.16, 2.59 and 3.03, and storeys repeat vertically the
same way. Individually a window at twelve pixels is unreliable; the lattice it
sits on is robust. So the grid is found first and openings are placed on it,
which is also exactly the point set a procedural facade system consumes.

Both are deterministic and inspectable, and neither needs a model. Anything they
cannot settle is left as `unknown` rather than filled in.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .facade import BAY_MAX_M, BAY_MIN_M, rhythm_profile

#: A reveal shallower than this is inside the mesh's own noise. Measured facade
#: relief RMS on the Helsinki historic core is 0.128 m, so a step has to clear
#: that to be a window rather than surface wobble.
MIN_REVEAL_M = 0.18

#: And deeper than this is not a reveal -- it is a courtyard seen through a gap,
#: or a hole the reconstruction left open.
MAX_REVEAL_M = 1.6

#: Storey heights worth searching for. Below 2.4 m is not a storey; above 6 m is
#: a hall or a mis-detection at half the true period.
STOREY_MIN_M = 2.4
STOREY_MAX_M = 6.0

#: Width of the local wall reference, in metres. Wide enough to span a bay so a
#: window is compared against the masonry either side of it, narrow enough that
#: a frontage's own curvature over tens of metres does not enter the comparison.
REFERENCE_SPAN_M = 3.0


def reveal_mask(facade) -> tuple[np.ndarray, dict]:
    """Where the facade steps back far enough to be an opening.

    Depth is measured from the facade plane, so "recessed" means further from
    the camera than the local wall surface. The reference is a wide median of
    the depth itself rather than the plane's nominal offset: a real frontage is
    bumpy and sits at its own distance, and comparing against the fitted plane
    would call one whole leaning wall a window.
    """
    if facade.depth is None:
        return np.zeros(facade.image.shape[:2], dtype=bool), {"reason": "no depth"}
    depth = np.asarray(facade.depth, dtype=np.float64)
    valid = np.isfinite(depth)
    if valid.sum() < 64:
        return np.zeros(depth.shape, dtype=bool), {"reason": "too little depth"}

    # The reference has to be LOCAL. A single global percentile over a 40 m
    # frontage flagged half the facade as recessed, because a real wall is not
    # planar over that distance and its own curvature dwarfs a 0.3 m reveal.
    # Comparing against a wide blur of the depth itself measures a step relative
    # to the wall beside it, which is what a reveal actually is.
    from .frequency import box_blur

    filled = np.where(valid, depth, 0.0)
    weight = valid.astype(np.float64)
    radius = max(2, int(round(REFERENCE_SPAN_M * facade.px_per_m / 2.0)))
    # Uncovered pixels carry no depth, so the reference is a weighted mean over
    # whatever the flight actually saw. No wrap either: the left edge of a
    # frontage is not adjacent to its right edge.
    total = box_blur(filled[:, :, None], radius, wrap=False)[:, :, 0]
    count = box_blur(weight[:, :, None], radius, wrap=False)[:, :, 0]
    reference = np.where(count > 1e-6, total / np.maximum(count, 1e-6), np.nan)
    recess = depth - reference
    mask = valid & np.isfinite(reference) & (recess > MIN_REVEAL_M) & (recess < MAX_REVEAL_M)

    report = {
        "reference_span_m": REFERENCE_SPAN_M,
        "recessed_fraction": round(float(mask.sum() / max(valid.sum(), 1)), 4),
        "median_recess_m": round(float(np.median(recess[mask])), 3) if mask.any() else None,
        "p90_recess_m": round(float(np.percentile(recess[mask], 90)), 3) if mask.any() else None,
        "min_reveal_m": MIN_REVEAL_M,
        "evidence": "measured depth from the mesh, not inferred from pixels",
    }
    return mask, report


def period(signal: np.ndarray, px_per_m: float, lo_m: float, hi_m: float
           ) -> tuple[float, float]:
    """Dominant repeat of a 1-D signal, in metres, with its correlation strength.

    Differenced before correlating, so the answer is about where edges recur
    rather than about overall brightness -- a facade in shadow scores like a
    facade in sun.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if len(signal) < 8:
        return 0.0, 0.0
    diff = np.diff(signal)
    diff = diff - diff.mean()
    energy = float(diff @ diff)
    if energy < 1e-9:
        return 0.0, 0.0
    lo = max(2, int(lo_m * px_per_m))
    hi = min(len(diff) // 2, int(hi_m * px_per_m))
    if hi <= lo:
        return 0.0, 0.0
    best_lag, best = lo, -1.0
    for lag in range(lo, hi):
        score = float(diff[:-lag] @ diff[lag:]) / energy
        if score > best:
            best, best_lag = score, lag
    return best_lag / px_per_m, max(0.0, best)


@dataclass
class Lattice:
    """The bay and storey grid a facade repeats on."""
    bay_m: float
    storey_m: float
    bay_strength: float
    storey_strength: float
    bays: np.ndarray = field(default_factory=lambda: np.zeros(0))    # u, metres
    storeys: np.ndarray = field(default_factory=lambda: np.zeros(0))  # v, metres

    @property
    def points(self) -> np.ndarray:
        """(N, 2) bay/storey intersections in wall-local metres.

        This is the set a procedural facade system consumes: one point per
        opening position, typed later. Placing openings on a lattice rather than
        detecting each one matters because at 10 px/m a 1.2 m window is twelve
        pixels and unreliable, while the period it repeats on is robust.
        """
        if not len(self.bays) or not len(self.storeys):
            return np.zeros((0, 2))
        uu, vv = np.meshgrid(self.bays, self.storeys, indexing="xy")
        return np.column_stack([uu.ravel(), vv.ravel()])

    def to_record(self) -> dict:
        return {
            "bay_m": round(self.bay_m, 3),
            "storey_m": round(self.storey_m, 3),
            "bay_strength": round(self.bay_strength, 3),
            "storey_strength": round(self.storey_strength, 3),
            "bays": len(self.bays),
            "storeys": len(self.storeys),
            "points": int(len(self.points)),
            "epistemic": "derived",     # measured periods, inferred placement
        }


def lattice(facade, *, mask: np.ndarray | None = None) -> Lattice:
    """Bay and storey periods, and the grid they imply.

    The horizontal period comes from the column signal and the vertical from the
    row signal, both over the same crop. Where a reveal mask is supplied it is
    used as the signal instead of luminance -- a measured recess is a far
    cleaner periodic signal than a photograph in which the windows are twelve
    pixels of dark grey.
    """
    if mask is not None and mask.any():
        field_2d = mask.astype(np.float64)
    else:
        field_2d = facade.image.astype(np.float64).mean(axis=2) / 255.0

    bay_m, bay_strength = period(field_2d.mean(axis=0), facade.px_per_m,
                                 BAY_MIN_M, BAY_MAX_M)
    storey_m, storey_strength = period(field_2d.mean(axis=1), facade.px_per_m,
                                       STOREY_MIN_M, STOREY_MAX_M)

    bays = np.zeros(0)
    storeys = np.zeros(0)
    if bay_m > 0:
        # Phase from the strongest correlation position, so the grid sits on the
        # openings rather than between them.
        column = field_2d.mean(axis=0)
        step = bay_m * facade.px_per_m
        offsets = np.arange(0, step)
        scores = [float(column[int(o)::int(max(step, 2))].mean()) for o in offsets]
        phase = float(offsets[int(np.argmax(scores))]) / facade.px_per_m
        bays = np.arange(phase, facade.width_m, bay_m)
    if storey_m > 0:
        row = field_2d.mean(axis=1)
        step = storey_m * facade.px_per_m
        offsets = np.arange(0, step)
        scores = [float(row[int(o)::int(max(step, 2))].mean()) for o in offsets]
        phase = float(offsets[int(np.argmax(scores))]) / facade.px_per_m
        # v runs up from the base, and the crop's rows run down from the top.
        storeys = facade.height_m - np.arange(phase, facade.height_m, storey_m)

    return Lattice(bay_m=bay_m, storey_m=storey_m, bay_strength=bay_strength,
                   storey_strength=storey_strength, bays=bays, storeys=storeys)


def openings(facade) -> dict:
    """Everything this module can say about a facade's openings, with evidence.

    Deliberately reports the two sources separately. Depth is measured and
    rhythm is derived, and a stage that later places a window frame needs to
    know which of the two put it there.
    """
    mask, depth_report = reveal_mask(facade)
    grid = lattice(facade, mask=mask)
    on_grid = 0
    if grid.points.size and mask.any():
        rows = np.clip(((facade.height_m - grid.points[:, 1]) * facade.px_per_m
                        ).astype(int), 0, mask.shape[0] - 1)
        cols = np.clip((grid.points[:, 0] * facade.px_per_m).astype(int),
                       0, mask.shape[1] - 1)
        on_grid = int(mask[rows, cols].sum())
    return {
        "depth": depth_report,
        "lattice": grid.to_record(),
        "grid_points_on_a_recess": on_grid,
        "grid_agreement": round(on_grid / max(len(grid.points), 1), 3),
        "note": "depth is measured; the lattice is derived; placement is inferred",
    }

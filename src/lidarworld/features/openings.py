"""Find windows on a facade from depth and from rhythm, not from a segmenter.

Three cities in, no open city model anywhere carries a facade opening as
geometry. Hamburg's LoD3 has zero `bldg:Window`; Helsinki's CityGML has 34,260
wall surfaces and zero windows. So a window has to be found, and there are two
pieces of evidence that beat guessing from pixels.

**Depth**, which does not work on this data, and the measurement saying so is
the useful part. The rectifier builds a depth buffer to resolve overlaps, so a
reveal ought to be a measured step in it rather than an inference from a 13 px/m
photograph in which a 1.2 m window is sixteen pixels. Measured on a Helsinki
frontage: against a local 3 m reference the wall is flat to 0.044 m and a window
sits 0.05 m off it. The reveals are below resolution. A global comparison makes
them look present -- window pixels read 0.65 m further than masonry -- but that
is the facade's own lean correlating with which parts of it are in shade, not
depth. So `reveal_mask` refuses when the wall's local noise cannot support a
reveal, and what it finds when it does answer is building-scale: a light well, a
gap between blocks, a deep loggia. Not a window.

The depth buffer is still worth keeping, for the job it is good at. A pavement
is metres from the wall plane and a roof slopes away, so depth separates wall
from not-wall cleanly, which is what `wall_band` uses. That mattered more than
the reveals did: restricting the rhythm to the wall took the storey correlation
from 0.107 to 0.404.

**Rhythm**, which does work. A facade repeats, and the period survives what an
individual sixteen-pixel window does not. On the same frontage: bay 3.77 m at
strength 0.44, storey 3.31 m at 0.40, against a city model that records seven
storeys over about 23.9 m of wall -- 3.41 m each, so the detected period agrees
to 3%. The grid is found first and openings are placed on it, which is also
exactly the point set a procedural facade system consumes.

Both are deterministic and inspectable, and neither needs a model. Anything they
cannot settle is left as `unknown` rather than filled in -- including, on this
mesh, where the windows are.
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

#: How much the depth may vary within the reference window, metres, for the
#: window's own centre to be judged. This is the criterion that makes the mask
#: about facade rather than about the crop.
#:
#: A rectified slab runs over the roof and down onto the pavement, and at a corner
#: it catches the return wall. Those regions are not ramps -- they are plateaux
#: metres from the wall plane -- so a local reference sitting inside one of them
#: reports no recess at all. What it does report is a band the width of the
#: reference either side of every boundary, where the mean is a blend of wall and
#: not-wall and half of it therefore looks recessed. On a Helsinki frontage that
#: put 8.6% of the wall in the mask with essentially all of it in three bands at
#: the crop edges and almost none on the windows, which then fed the lattice a
#: signal made of crop edges and produced a 1.60 m bay on a facade whose bays
#: measure 2.9 m.
#:
#: A window reveal is a small step inside an otherwise flat neighbourhood, so its
#: local spread stays well under a metre. A wall/pavement boundary is metres. The
#: cost is real and worth stating: a deep balcony recess also raises the local
#: spread and is excluded with the pavement.
MAX_LOCAL_SPREAD_M = 0.8

#: How far above the depth map's own local noise a reveal has to stand for the
#: detector to answer at all. Three is the conventional detection margin and it is
#: not tuned: at 3, the Helsinki historic core's 0.044 m noise floor cannot
#: support a 0.18 m reveal, and the detector says so instead of guessing.
REVEAL_SNR = 3.0

#: A row belongs to the wall band if this fraction of the busiest row's planar
#: pixels are planar in it too. Relative rather than absolute, because a frontage
#: seen obliquely is never fully covered and an absolute floor would reject the
#: whole crop.
WALL_ROW_FRACTION = 0.75


@dataclass
class DepthField:
    """A facade's depth, reduced to the three things every consumer of it wants."""
    recess: np.ndarray                  # depth minus its own local reference
    interior: np.ndarray                # bool: planar wall, clear of every edge
    noise_m: float                      # what the wall's own depth resolves


def depth_field(facade) -> DepthField | None:
    """Local recess, wall interior and noise floor, from the rectifier's depth.

    One function because `reveal_mask` and `wall_band` were disagreeing about what
    counts as wall, and two definitions of that in one module is how a detector
    ends up measuring the crop instead of the building.

    The reference is LOCAL. A single global percentile over a 40 m frontage
    flagged half the facade as recessed, because a real wall is not planar over
    that distance and its own lean dwarfs a 0.3 m reveal. Comparing against a wide
    weighted mean of the depth itself measures a step relative to the wall beside
    it, which is what a reveal actually is.
    """
    if facade.depth is None:
        return None
    depth = np.asarray(facade.depth, dtype=np.float64)
    valid = np.isfinite(depth)
    if valid.sum() < 64:
        return None

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

    # Local spread over the same window, from E[d^2] - E[d]^2. Cheap, because the
    # running sums are already the machinery in use.
    squares = box_blur((filled ** 2)[:, :, None], radius, wrap=False)[:, :, 0]
    variance = np.where(count > 1e-6, squares / np.maximum(count, 1e-6), np.nan) \
        - reference ** 2
    spread = np.sqrt(np.maximum(variance, 0.0))
    facade_like = np.isfinite(spread) & (spread <= MAX_LOCAL_SPREAD_M) & valid

    # The interior is the facade-like region eroded by the reference radius, so a
    # surviving pixel's whole reference window was facade. Without it the band
    # either side of every wall/pavement boundary qualifies, because there the
    # reference is a blend of the two and half of it reads as recessed.
    coverage = box_blur(facade_like.astype(np.float64)[:, :, None], radius,
                        wrap=False)[:, :, 0]
    interior = coverage > 0.999

    residual = recess[interior & np.isfinite(recess)]
    noise = float(np.std(residual)) if residual.size > 64 else float("nan")
    return DepthField(recess=recess, interior=interior, noise_m=noise)


def reveal_mask(facade) -> tuple[np.ndarray, dict]:
    """Where the facade steps back far enough to be an opening.

    On the Helsinki reality mesh the answer is nowhere, and the point of this
    function is now to say so. See the module docstring: locally this depth map is
    flat to 4-10 cm and a window sits 5 cm off its own wall, so the reveals are
    below resolution and everything that clears a 0.18 m threshold is a
    building-scale recess -- a light well, a gap between blocks -- rather than an
    opening. The refusal is the result, not a failure to try.
    """
    field_ = depth_field(facade)
    if field_ is None:
        empty = np.zeros(facade.image.shape[:2], dtype=bool)
        return empty, {"reason": "no usable depth", "resolves_reveals": False}
    recess, interior, noise = field_.recess, field_.interior, field_.noise_m
    valid = np.isfinite(np.asarray(facade.depth, dtype=np.float64))

    mask = (interior & np.isfinite(recess)
            & (recess > MIN_REVEAL_M) & (recess < MAX_REVEAL_M))

    report = {
        "reference_span_m": REFERENCE_SPAN_M,
        "min_reveal_m": MIN_REVEAL_M,
        "interior_fraction": round(float(interior.sum() / max(valid.sum(), 1)), 4),
        "max_local_spread_m": MAX_LOCAL_SPREAD_M,
        "local_noise_m": round(noise, 4) if np.isfinite(noise) else None,
        "evidence": "measured depth from the mesh, not inferred from pixels",
    }

    # Refuse rather than answer. If the wall's own depth never varies by as much
    # as a reveal, this mesh does not resolve reveals and the honest output is
    # nothing -- an empty mask with a reason beats a mask made of crop edges,
    # which is what the previous version returned and what then produced a 1.60 m
    # bay on a facade whose bays measure 3.8 m.
    if np.isfinite(noise) and noise * REVEAL_SNR < MIN_REVEAL_M:
        report["recessed_fraction"] = 0.0
        report["reason"] = (
            f"depth resolves {noise:.3f} m locally; a {MIN_REVEAL_M} m reveal "
            f"needs {MIN_REVEAL_M / REVEAL_SNR:.3f} m to clear noise")
        report["resolves_reveals"] = False
        return np.zeros(recess.shape, dtype=bool), report

    report["resolves_reveals"] = True
    report["recessed_fraction"] = round(float(mask.sum() / max(valid.sum(), 1)), 4)
    report["median_recess_m"] = (round(float(np.median(recess[mask])), 3)
                                 if mask.any() else None)
    report["p90_recess_m"] = (round(float(np.percentile(recess[mask], 90)), 3)
                              if mask.any() else None)
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


def wall_band(facade) -> tuple[int, int]:
    """The contiguous rows of the crop that are actually wall.

    A rectified slab runs from the pavement to the roof ridge, and the rows at
    either end are neither. Including them is what kept the storey period weak:
    on a Helsinki frontage the vertical correlation strength went from 0.107 over
    the whole crop to 0.415 over the wall alone, because the roof and the road
    contribute rows that repeat at nothing.

    This is the job the depth buffer is good for. It cannot resolve a 15 cm window
    reveal, but a pavement is metres from the wall plane and a roof slopes away, so
    depth separates wall from not-wall easily. Where there is no depth, the covered
    rows are the best available answer.
    """
    rows = facade.image.shape[0]
    field_ = depth_field(facade)
    covered = (field_.interior.mean(axis=1) if field_ is not None
               else (facade.image.sum(axis=2) > 0).mean(axis=1))
    if not covered.any():
        return 0, rows
    # The longest run of rows that are mostly wall, not every such row: a gutter
    # or a doorway can be planar too, and a band with holes in it is not a band.
    keep = covered >= WALL_ROW_FRACTION * float(covered.max())
    best, run_start, best_span = (0, rows), None, 0
    for row in range(rows + 1):
        if row < rows and keep[row]:
            run_start = row if run_start is None else run_start
        elif run_start is not None:
            if row - run_start > best_span:
                best, best_span = (run_start, row), row - run_start
            run_start = None
    return best


@dataclass
class Lattice:
    """The bay and storey grid a facade repeats on."""
    bay_m: float
    storey_m: float
    bay_strength: float
    storey_strength: float
    bays: np.ndarray = field(default_factory=lambda: np.zeros(0))    # u, metres
    storeys: np.ndarray = field(default_factory=lambda: np.zeros(0))  # v, metres
    bay_source: str = "luminance"
    storey_source: str = "luminance"
    band: tuple[int, int] = (0, 0)

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
            "bay_source": self.bay_source,
            "storey_source": self.storey_source,
            "wall_band_rows": list(self.band),
            "bays": len(self.bays),
            "storeys": len(self.storeys),
            "points": int(len(self.points)),
            "epistemic": "derived",     # measured periods, inferred placement
        }


def _phase(signal: np.ndarray, period_m: float, px_per_m: float,
           extent_m: float) -> np.ndarray:
    """Where to start the grid, so it sits on the openings rather than between.

    `signal` must be an *openness* signal -- larger where an opening is more
    likely. That is what a reveal mask already is and the opposite of luminance,
    where a window is the dark part, so the caller inverts. Getting this backwards
    puts every grid point on the masonry pier between two windows and on the
    spandrel band between two storeys, with the period still exactly right, which
    looks convincing in a render until you notice the lines avoid every window.
    """
    step = int(max(period_m * px_per_m, 2))
    offsets = np.arange(0, step)
    scores = [float(signal[int(o)::step].mean()) for o in offsets]
    return np.arange(float(offsets[int(np.argmax(scores))]) / px_per_m,
                     extent_m, period_m)


def lattice(facade, *, mask: np.ndarray | None = None) -> Lattice:
    """Bay and storey periods, and the grid they imply.

    Two candidate signals per axis and the stronger wins, per axis, reported. The
    first version preferred a reveal mask whenever one existed, on the reasoning
    that a measured recess beats a photograph. That reasoning is right and the
    premise was wrong: this mesh does not resolve window reveals, so the mask was
    made of building-scale recesses, and preferring it produced a 1.60 m bay at
    strength 0.05 where luminance gives 3.77 m at 0.42 on a facade whose bays
    measure about 3.8 m. Measuring both and choosing is the fix -- on a survey
    that does resolve reveals the mask will win on its own merits.
    """
    top, bottom = wall_band(facade)
    if bottom - top < 8:
        top, bottom = 0, facade.image.shape[0]
    band_m = (bottom - top) / facade.px_per_m

    # Both signals point the same way: larger means more likely an opening. A
    # window is the DARK part of a photograph, so luminance is inverted; a reveal
    # mask is already an openness field.
    luma = facade.image.astype(np.float64).mean(axis=2)[top:bottom] / 255.0
    signals = {"luminance": 1.0 - luma}
    if mask is not None and mask.any():
        signals["reveal-depth"] = mask.astype(np.float64)[top:bottom]

    def best(axis: int, lo: float, hi: float) -> tuple[float, float, str, np.ndarray]:
        found = {name: period(field_2d.mean(axis=axis), facade.px_per_m, lo, hi)
                 for name, field_2d in signals.items()}
        source = max(found, key=lambda k: found[k][1])
        return (*found[source], source, signals[source].mean(axis=axis))

    bay_m, bay_strength, bay_source, bay_signal = best(0, BAY_MIN_M, BAY_MAX_M)
    storey_m, storey_strength, storey_source, storey_signal = best(
        1, STOREY_MIN_M, STOREY_MAX_M)

    bays = (_phase(bay_signal, bay_m, facade.px_per_m, facade.width_m)
            if bay_m > 0 else np.zeros(0))
    storeys = np.zeros(0)
    if storey_m > 0:
        # v runs up from the base of the crop; the band's rows run down from its
        # own top, so the grid is placed within the band and then lifted to the
        # crop's frame.
        within = _phase(storey_signal, storey_m, facade.px_per_m, band_m)
        storeys = facade.height_m - (top / facade.px_per_m + within)

    return Lattice(bay_m=bay_m, storey_m=storey_m, bay_strength=bay_strength,
                   storey_strength=storey_strength, bays=bays, storeys=storeys,
                   bay_source=bay_source, storey_source=storey_source,
                   band=(top, bottom))


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

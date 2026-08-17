"""Repair a facade against its own average bay, and record what was invented.

The warping in a photogrammetric wall is not global. Measured on a Helsinki
frontage, the storey phase holds to about 8 cm across most of the wall, so the
cornices are essentially straight and the de-warp in `openings` has little to
remove. What is left is *local*: a cornice smeared where its own overhang
occluded the wall below it, a bay melted where the flight had one bad look at it,
a corner slid where two buildings meet. Those defects are not a distortion of the
facade, they are holes in it wearing the facade's colours.

A facade repeats, though, and that is the way in. Cut the wall into its lattice
cells -- one bay wide, one storey tall, aligned on the measured grid -- and most
of them are the same window. Take the per-pixel median across all of them and the
result is that window with the damage voted out, because a smear in one cell is
outvoted by the twenty cells that are intact. Then any cell that disagrees badly
with the median is the damaged one, and can be replaced by it.

Three things this is careful about.

**It is a vote, not a filter.** The median is over cells, so the canonical window
is assembled entirely from measured pixels -- just not from the pixels of the cell
being repaired. Nothing is hallucinated, and the support (how many cells voted) is
reported per repair.

**A replaced pixel is no longer measured.** `repair` returns a provenance mask
alongside the image. A pixel taken from the median is `generated` even though its
ingredients were measured, because it is no longer evidence about *that* part of
the wall. Downstream stages that care -- forward validation especially -- need to
be able to exclude it.

**It refuses when the facade does not repeat.** A frontage with a shop front, an
arch and five different window types has no canonical cell, and pasting a median
over it would flatten real variety into one invented window. The agreement
distribution decides, and a facade whose cells do not agree is returned untouched
with the numbers that say why.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: A cell is damaged when its disagreement with the canonical exceeds the median
#: disagreement by this many robust deviations. Robust rather than absolute,
#: because how much a facade's bays differ is a property of the building: a
#: rhythmic Helsinki block sits far tighter than a terrace of five different
#: houses, and one threshold cannot serve both.
DAMAGE_SIGMA = 2.5

#: Cells must outnumber this for a median to mean anything. Below it the "average
#: bay" is one or two bays and a single damaged one swings it.
MIN_CELLS = 6

#: A facade whose typical cell disagrees with the canonical by more than this
#: fraction of full range does not repeat, and there is nothing to repair against.
MAX_TYPICAL_DISAGREEMENT = 0.16

#: And the lattice itself has to be real. Correlation strength below this is not a
#: period, it is the strongest lag in noise, and cells cut on it are arbitrary
#: rectangles whose median is a grey smudge. Random noise reaches 0.11 and passes
#: the disagreement gate above -- bilinear resampling smooths it enough to look
#: consistent -- so the disagreement test alone is not sufficient. Measured on
#: three real Helsinki frontages: 0.435, 0.295 and 0.181.
MIN_LATTICE_STRENGTH = 0.15

#: Width of the blend at a replaced cell's edge, as a fraction of the cell. A hard
#: paste leaves a rectangle visible at exactly the spacing of the lattice, which
#: reads as a grid of patches rather than as a wall.
FEATHER = 0.18


def _sample(image: np.ndarray, rows: np.ndarray, cols: np.ndarray
           ) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear gather at float positions. Returns (samples, inside).

    Float positions matter: a bay is 3.77 m at 48 px/m, so cells cut at integer
    pixels drift by a pixel every few bays and the stack blurs itself. Sampling at
    the true grid positions keeps the windows registered.
    """
    height, width = image.shape[:2]
    r0 = np.floor(rows).astype(np.int64)
    c0 = np.floor(cols).astype(np.int64)
    fr = (rows - r0)[..., None]
    fc = (cols - c0)[..., None]
    inside = (r0 >= 0) & (r0 < height - 1) & (c0 >= 0) & (c0 < width - 1)
    r0c = np.clip(r0, 0, height - 2)
    c0c = np.clip(c0, 0, width - 2)
    top = image[r0c, c0c] * (1 - fc) + image[r0c, c0c + 1] * fc
    bottom = image[r0c + 1, c0c] * (1 - fc) + image[r0c + 1, c0c + 1] * fc
    return top * (1 - fr) + bottom * fr, inside


@dataclass
class CellStack:
    """Every lattice cell of a facade, resampled onto a common frame."""
    cells: np.ndarray                   # (N, h, w, 3) float, 0..1
    valid: np.ndarray                   # (N, h, w) bool
    rows: np.ndarray                    # (N, h, w) source row per sample
    cols: np.ndarray                    # (N, h, w) source col per sample
    shape: tuple[int, int]              # (h, w)

    def __len__(self) -> int:
        return len(self.cells)


def cell_stack(facade, grid) -> CellStack | None:
    """Cut the wall band into lattice cells, one per bay-storey intersection.

    Each cell is one period wide and one period tall, centred on its grid point,
    so the window sits in the middle of the cell rather than across a seam.
    """
    if grid.bay_m <= 0 or grid.storey_m <= 0:
        return None
    points = grid.points
    if len(points) < MIN_CELLS:
        return None

    px = facade.px_per_m
    height_px = max(4, int(round(grid.storey_m * px)))
    width_px = max(4, int(round(grid.bay_m * px)))
    # Cell-local sample grid, in pixels, centred on the grid point.
    dv = np.linspace(-grid.storey_m * px / 2, grid.storey_m * px / 2, height_px)
    du = np.linspace(-grid.bay_m * px / 2, grid.bay_m * px / 2, width_px)
    ddv, ddu = np.meshgrid(dv, du, indexing="ij")

    image = np.asarray(facade.image, dtype=np.float64)
    if image.max() > 1.5:
        image = image / 255.0

    rows_all, cols_all = [], []
    for u_m, v_m in points:
        centre_row = (facade.height_m - v_m) * px
        centre_col = u_m * px
        rows_all.append(centre_row + ddv)
        cols_all.append(centre_col + ddu)
    rows = np.stack(rows_all)
    cols = np.stack(cols_all)
    samples, inside = _sample(image, rows, cols)
    # A black pixel is one the flight never saw, not a very dark window.
    valid = inside & (samples.sum(axis=-1) > 0)
    return CellStack(cells=samples, valid=valid, rows=rows, cols=cols,
                     shape=(height_px, width_px))


@dataclass
class Canonical:
    """The average bay of a facade, and how well each bay matches it."""
    image: np.ndarray                   # (h, w, 3) float 0..1
    support: np.ndarray                 # (h, w) how many cells voted per pixel
    disagreement: np.ndarray            # (N,) mean absolute error per cell
    typical: float                      # median disagreement
    spread: float                       # robust deviation of it

    @property
    def agreement(self) -> np.ndarray:
        """Per cell, 1 at the median cell and falling as it disagrees."""
        return np.clip(1.0 - self.disagreement, 0.0, 1.0)

    def damaged(self, sigma: float = DAMAGE_SIGMA) -> np.ndarray:
        """Which cells disagree with the average by more than the facade's own scatter."""
        return self.disagreement > self.typical + sigma * max(self.spread, 1e-6)

    def to_record(self) -> dict:
        return {
            "cells": int(len(self.disagreement)),
            "cell_px": list(self.image.shape[:2]),
            "median_support": int(np.median(self.support)),
            "typical_disagreement": round(self.typical, 4),
            "disagreement_spread": round(self.spread, 4),
            "cells_agreeing": int((~self.damaged()).sum()),
            "agreeing_fraction": round(float((~self.damaged()).mean()), 3),
            "epistemic": "derived",   # a median over measured cells
        }


def canonical(stack: CellStack) -> Canonical:
    """The per-pixel median cell, and each cell's distance from it.

    Median rather than mean, and that is the whole mechanism. A mean would let one
    smeared bay drag every pixel of the average toward the smear; a median lets the
    intact bays outvote it entirely. Which is also why the support map is reported:
    a pixel that only three cells could see is a weak vote, and a repair that leans
    on it should be visible as such.
    """
    cells = stack.cells
    valid = stack.valid
    filled = np.where(valid[..., None], cells, np.nan)
    with np.errstate(invalid="ignore"):
        median = np.nanmedian(filled, axis=0)
    support = valid.sum(axis=0)
    median = np.where(np.isfinite(median), median, 0.0)

    both = valid & (support > 0)[None, :, :]
    error = np.abs(cells - median[None]).mean(axis=-1)
    counts = both.sum(axis=(1, 2))
    disagreement = np.where(
        counts > 0, (error * both).sum(axis=(1, 2)) / np.maximum(counts, 1), 1.0)
    typical = float(np.median(disagreement))
    # Median absolute deviation, scaled to a standard deviation. Robust, because
    # the damaged cells are exactly the outliers a standard deviation would
    # inflate itself with.
    spread = 1.4826 * float(np.median(np.abs(disagreement - typical)))
    return Canonical(image=median, support=support, disagreement=disagreement,
                     typical=typical, spread=spread)


def _feather(shape: tuple[int, int], fraction: float = FEATHER) -> np.ndarray:
    """1 in the middle of a cell, falling to 0 at its edge."""
    def ramp(n):
        edge = max(1, int(round(n * fraction)))
        w = np.ones(n)
        taper = np.linspace(0.0, 1.0, edge + 2)[1:-1]
        w[:edge] = taper
        w[-edge:] = taper[::-1]
        return w
    return ramp(shape[0])[:, None] * ramp(shape[1])[None, :]


def _subset(stack: CellStack, members: np.ndarray) -> CellStack:
    return CellStack(cells=stack.cells[members], valid=stack.valid[members],
                     rows=stack.rows[members], cols=stack.cols[members],
                     shape=stack.shape)


def repair(facade, grid, *, sigma: float = DAMAGE_SIGMA,
           groups: np.ndarray | None = None, worst: float | None = None
           ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Replace the bays that disagree with the average bay. Returns (image, generated, report).

    `groups` labels each lattice point with the building it belongs to, and it is
    not optional in practice. A rectified slab runs along a street, and a street is
    several buildings: on a Helsinki frontage one slab spanned two blocks whose bays
    differ, and averaging them together produced a canonical window belonging to the
    majority building. Repairing with it would paste one building's architecture onto
    its neighbour -- geometry that was never there, at the address of geometry that
    was. Grouping keeps each building averaged against itself, and a building with
    too few bays to vote is left alone and said so.

    `worst` repairs a fixed fraction of the least-agreeing cells instead of using
    the scatter-based threshold. Use it when the facade is known to repeat and the
    damage is known to be a minority -- "97% of these bays are the same window, fix
    the other 3%" -- which is a clearer statement of intent than a sigma.

    `generated` is the provenance mask: True where a pixel came from the median
    rather than from that part of the wall. It is not a detail -- forward validation
    compares the model against returned sensor data, and scoring a synthesised
    window against a measured return would credit the pipeline for inventing well.
    """
    image = np.asarray(facade.image, dtype=np.float64)
    scale = 255.0 if image.max() > 1.5 else 1.0
    out = image / scale
    generated = np.zeros(out.shape[:2], dtype=bool)

    stack = cell_stack(facade, grid)
    if stack is None:
        return facade.image, generated, {
            "applied": False,
            "reason": f"fewer than {MIN_CELLS} lattice cells to average over"}

    strength = max(grid.bay_strength, grid.storey_strength)
    if strength < MIN_LATTICE_STRENGTH:
        return facade.image, generated, {
            "applied": False,
            "lattice_strength": round(float(strength), 3),
            "reason": (f"lattice correlates at {strength:.3f}, under the "
                       f"{MIN_LATTICE_STRENGTH} at which it is a period rather "
                       f"than the strongest lag in noise")}

    if groups is None:
        labels = np.zeros(len(stack), dtype=np.int64)
    else:
        labels = np.asarray(groups, dtype=np.int64)[:len(stack)]

    weight = _feather(stack.shape)
    per_group, replaced, skipped = {}, 0, {}
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        if len(members) < MIN_CELLS:
            skipped[str(label)] = f"only {len(members)} bays; needs {MIN_CELLS}"
            continue
        subset = _subset(stack, members)
        model = canonical(subset)
        per_group[str(label)] = model.to_record()
        if model.typical > MAX_TYPICAL_DISAGREEMENT:
            skipped[str(label)] = (
                f"bays disagree by {model.typical:.3f}, over the "
                f"{MAX_TYPICAL_DISAGREEMENT} at which a facade counts as repeating")
            continue

        if worst is not None:
            count = int(round(worst * len(members)))
            order = np.argsort(model.disagreement)[::-1]
            damaged = np.zeros(len(members), dtype=bool)
            damaged[order[:count]] = True
        else:
            damaged = model.damaged(sigma)

        for local in np.flatnonzero(damaged):
            index = members[local]
            r = np.rint(stack.rows[index]).astype(np.int64)
            c = np.rint(stack.cols[index]).astype(np.int64)
            inside = ((r >= 0) & (r < out.shape[0]) & (c >= 0) & (c < out.shape[1])
                      & (model.support > 0))
            if not inside.any():
                continue
            blend = (weight * inside)[inside][:, None]
            out[r[inside], c[inside]] = (out[r[inside], c[inside]] * (1 - blend)
                                        + model.image[inside] * blend)
            # Any pixel the blend touched at all. Marking only the ones over half
            # taken from the median left the feathered rim modified but labelled
            # measured, which is the worst of both: changed pixels that forward
            # validation would still score as evidence.
            generated[r[inside], c[inside]] = True
            replaced += 1

    record = {
        "applied": replaced > 0,
        "groups": per_group,
        "groups_skipped": skipped,
        "cells": len(stack),
        "cells_replaced": replaced,
        "replaced_fraction": round(float(replaced / max(len(stack), 1)), 3),
        "generated_pixel_fraction": round(float(generated.mean()), 4),
        "policy": (f"worst {worst:.0%}" if worst is not None
                   else f"{sigma} robust deviations"),
        "feather": FEATHER,
        "note": "replaced pixels are a median of other bays: derived, not measured",
    }
    if not replaced and not skipped:
        record["reason"] = "no bay disagreed with the average by enough to replace"
    return (np.clip(out * scale, 0, scale).astype(facade.image.dtype)
            if scale == 255.0 else np.clip(out, 0, 1)), generated, record

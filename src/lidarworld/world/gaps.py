"""Classify a missing region before anything is allowed to fill it.

There is deliberately no `fill_holes()` in this compiler. "No returns here" is
not one condition, it is at least four, and they call for opposite responses:

    a courtyard with no roof        must NOT be filled -- there is no surface
    a roof behind a taller building must be filled    -- the surface is there
    a road under a parked van       must be filled    -- the road continues
    a river                         must be filled    -- but from a boundary,
                                                         not from returns

Filling all four is how a pipeline reports hole-free coverage while inventing
courtyard roofs. Refusing all four leaves the block full of lace. The
difference between them is not visible in the raster -- every one of them is
simply an absence -- so it has to come from somewhere else, and the only
honest source is a declared semantic region.

That is what makes the acquired polygons load-bearing rather than decorative.
Inside a pavement polygon, absence means the sensor missed a road that is
certainly there. Outside every declared region, absence is the answer.

    with polygons     absence is classified, and Tier 4 can fill it
    without them      everything unenclosed is `unknown`, and stays a hole

The output is GapRecords, not geometry. Deciding what a hole *is* and deciding
what to do about it are separate steps, and keeping them separate is what lets
the second one be audited.
"""
from __future__ import annotations

import numpy as np

from ..data.gis import point_in_polygon
from .records import GapRecord

try:
    from scipy import ndimage as _ndi
except ImportError:                      # pragma: no cover
    _ndi = None


def _label(mask: np.ndarray) -> tuple[np.ndarray, int]:
    if _ndi is not None:
        labels, count = _ndi.label(mask)
        return labels.astype(np.int32), int(count)
    from ..reconstruct.lattice import label_components
    return label_components(mask)


def _touches_border(labels: np.ndarray, index: int) -> bool:
    return bool(np.any(labels[0, :] == index) or np.any(labels[-1, :] == index)
                or np.any(labels[:, 0] == index) or np.any(labels[:, -1] == index))


def region_masks(raster, regions: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    """Rasterise each declared semantic region onto the terrain grid.

    A cell is in the region if its centre is inside any of the region's rings.
    Centres rather than overlap: a boundary cell belonging to both a road and a
    verge is a reconciliation question (see BoundarySeed), not something to
    settle by rounding here.
    """
    nx, ny = raster.shape
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    centres = np.column_stack([
        raster.origin[0] + (ii.ravel() + 0.5) * raster.cell,
        raster.origin[1] + (jj.ravel() + 0.5) * raster.cell,
    ])
    masks = {}
    for name, rings in regions.items():
        inside = np.zeros(len(centres), dtype=bool)
        for ring in rings:
            if len(ring) < 4:
                continue
            lo = ring.min(axis=0)
            hi = ring.max(axis=0)
            # Only test cells in the ring's bounding box. A downtown block has
            # thousands of rings and hundreds of thousands of cells; the full
            # cross product is the difference between a second and an hour.
            near = ((centres[:, 0] >= lo[0]) & (centres[:, 0] <= hi[0])
                    & (centres[:, 1] >= lo[1]) & (centres[:, 1] <= hi[1]))
            if not near.any():
                continue
            index = np.flatnonzero(near)
            inside[index] |= point_in_polygon(centres[index], ring[:, :2])
        masks[name] = inside.reshape(nx, ny)
    return masks


#: Largest absence that a passing obstruction plausibly explains, in square
#: metres. A car is about 8, a delivery truck 20, a bus 30. Above that the
#: thing casting the shadow would have to be the size of a building, and a
#: building-sized absence in the middle of a block is usually a courtyard.
#:
#: In metres rather than cells on purpose: a threshold in cells reclassifies
#: the same courtyard when the terrain resolution changes, which is how a
#: number that looked tuned starts inventing roofs at a finer grid.
OCCLUSION_MAX_AREA = 30.0


def classify(coverage: np.ndarray, raster, *, regions=None,
             min_cells: int = 2, occlusion_max_area: float = OCCLUSION_MAX_AREA,
             log=None) -> list[GapRecord]:
    """Classify every no-data component of `coverage` into a GapRecord.

    `coverage` is True where returns landed. `regions` maps a semantic name to
    a list of polygon rings in world coordinates -- pavement, sidewalk,
    roofprint, parkland. Absence inside one of those is a
    `semantic_region_gap`; absence outside all of them is a `true_void` and is
    not fillable.

    Components touching the raster border are `unknown` rather than void: the
    crop boundary is an artefact of what was compiled, and a road that leaves
    the tile has not been shown to end.
    """
    regions = regions or {}
    masks = region_masks(raster, regions) if regions else {}

    empty = ~coverage.astype(bool)
    labels, count = _label(empty)
    cell_area = float(raster.cell) ** 2
    gaps: list[GapRecord] = []

    for index in range(1, count + 1):
        cells = np.argwhere(labels == index)
        if len(cells) < min_cells:
            continue
        i0, j0 = cells.min(axis=0)
        i1, j1 = cells.max(axis=0)
        bounds = (float(raster.origin[0] + i0 * raster.cell),
                  float(raster.origin[1] + j0 * raster.cell),
                  float(raster.origin[0] + (i1 + 1) * raster.cell),
                  float(raster.origin[1] + (j1 + 1) * raster.cell))

        member = np.zeros(labels.shape, dtype=bool)
        member[labels == index] = True
        containing = [name for name, mask in masks.items()
                      if mask[member].any()]

        if containing:
            # A declared surface with no returns over it. The polygon asserts
            # the surface exists; only its elevation is missing, which is what
            # Tier 4 is for.
            gap_type = "semantic_region_gap"
            methods = ["tier_4_semantic_region_constraint"]
        elif _touches_border(labels, index):
            # Runs off the edge of what was compiled. Nothing has been shown
            # about it either way, and calling it void would be a claim the
            # crop box is not entitled to make.
            gap_type = "unknown"
            methods = []
        elif len(cells) * cell_area <= occlusion_max_area:
            # Small, enclosed by returns, and not in any declared region. A
            # parked vehicle, a canopy, a sensor shadow.
            gap_type = "occlusion"
            methods = ["tier_1_same_surface_interpolation",
                       "tier_2_topological_continuity"]
        else:
            # Large, enclosed, and no source says a surface belongs here.
            # Courtyards and light wells live in this bucket, and filling them
            # is how hole-free coverage gets reported dishonestly.
            gap_type = "true_void"
            methods = []

        record = GapRecord(
            id=f"gap.{len(gaps):05d}", gap_type=gap_type, bounds=bounds,
            dimensionality=2,
            neighboring_roles=sorted(containing),
            candidate_methods=methods,
            area=round(len(cells) * cell_area, 2),
            confidence=0.8 if containing else 0.5,
            status="open" if methods else "refused",
            notes=(f"{len(cells)} cells at {raster.cell:.2f} m"
                   + (f", inside {'+'.join(sorted(containing))}" if containing else "")),
        )
        gaps.append(record)
        if log is not None:
            log.gaps.append(record)
    return gaps


def summarise(gaps: list[GapRecord]) -> dict:
    """Counts and area by type, so the refusals are as visible as the fills."""
    by_type: dict[str, dict] = {}
    for gap in gaps:
        entry = by_type.setdefault(gap.gap_type, {"count": 0, "area_m2": 0.0})
        entry["count"] += 1
        entry["area_m2"] = round(entry["area_m2"] + (gap.area or 0.0), 1)
    fillable = sum(1 for g in gaps if g.fillable)
    return {"gaps": len(gaps), "fillable": fillable,
            "refused": len(gaps) - fillable,
            "by_type": dict(sorted(by_type.items()))}

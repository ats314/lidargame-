"""Synthesise walls that airborne LiDAR cannot contain.

A 3DEP tile is about 4 points per square metre measured from directly above.
Two thirds of that lands on pavement and roughly a quarter on roofs; facades
get almost nothing, because from an aircraft a wall is edge-on. The walls are
not sparse in the data -- they are absent from it. No amount of better
segmentation recovers them.

The standard answer, and what City3D and 3D BAG do, is to stop trying: take the
authoritative footprint, take the roof height that *was* measured, and extrude
the wall between them. The result is real geometry in the right place, derived
from two measurements, and it is the difference between floating slabs and a
city.

Every tile produced here is flagged OCCLUDED and SPARSE_EVIDENCE, so the IR
records these as inferred, the survey theme paints them as unmeasured, and
forward validation can exclude them. Synthesising geometry is fine. Claiming it
was observed is not.
"""
from __future__ import annotations

import numpy as np

from ..ir import program as program_ir
from ..segment.planes import PlanarPatch, plane_frame

UP = np.array([0.0, 0.0, 1.0])


def _ring_is_clockwise(ring: np.ndarray) -> bool:
    """Shoelace sign; decides which way is 'outside'."""
    x, y = ring[:, 0], ring[:, 1]
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]))) > 0


def walls_from_footprint(ring: np.ndarray, base_z: float, top_z: float, *,
                         start_id: int = 0, min_edge: float = 1.0,
                         max_edges: int = 64) -> list[PlanarPatch]:
    """One wall patch per footprint edge, spanning base_z to top_z."""
    if top_z - base_z < 2.0 or len(ring) < 4:
        return []

    clockwise = _ring_is_clockwise(ring)
    height = top_z - base_z
    mid_z = (base_z + top_z) / 2
    walls: list[PlanarPatch] = []

    for a, b in zip(ring[:-1], ring[1:]):
        edge = b - a
        length = float(np.hypot(edge[0], edge[1]))
        if length < min_edge or len(walls) >= max_edges:
            continue
        direction = edge / length
        # Outward normal: rotate the edge direction into the exterior.
        # For a counter-clockwise ring the interior is to the left of each
        # edge, so the outward normal is (dy, -dx); clockwise flips it.
        normal = (np.array([-direction[1], direction[0], 0.0]) if clockwise
                  else np.array([direction[1], -direction[0], 0.0]))

        centroid = np.array([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, mid_z])
        u, v = plane_frame(normal)
        patch = PlanarPatch(
            id=start_id + len(walls), normal=normal,
            offset=-float(normal @ centroid), centroid=centroid, u=u, v=v,
            point_idx=np.zeros(0, dtype=np.int64), support=0,
            extent=(length, height), rms=0.0,
            role="surface.wall.vertical",
            # Low by construction: nothing here was measured directly. The
            # footprint and the roof height were; this wall is their product.
            confidence=0.35,
        )
        patch.area = length * height
        patch.attrs["extruded"] = True
        patch.attrs["base_z"] = round(base_z, 2)
        patch.attrs["top_z"] = round(top_z, 2)
        walls.append(patch)
    return walls


def roof_height(patches, group: list[int], cloud) -> float | None:
    """Top of a building, from the roof planes that were actually measured."""
    tops = [float(cloud.xyz[patches[i].point_idx][:, 2].max())
            for i in group
            if patches[i].role.startswith("surface.roof") and len(patches[i].point_idx)]
    if not tops:
        return None
    # The tallest roof plane, not the mean: a building is as tall as its top.
    return max(tops)


def build(rings, assignment: np.ndarray, patches, cloud, raster, dtm, *,
          min_height: float = 2.5, start_id: int = 0):
    """Extrude every footprint that has a measured roof above it.

    Returns (walls, programs). The programs are the generative description
    the walls came out of -- a ring and two heights each -- kept so the
    envelope can be re-executed rather than only rendered.
    """
    if not len(rings):
        return [], []

    by_footprint: dict[int, list[int]] = {}
    for i, f in enumerate(assignment):
        if f >= 0:
            by_footprint.setdefault(int(f), []).append(i)

    walls: list[PlanarPatch] = []
    programs = []
    for f, group in by_footprint.items():
        top = roof_height(patches, group, cloud)
        if top is None:
            continue
        ring = rings[f]
        centre = ring.mean(axis=0)[None, :]
        base = float(np.nan_to_num(raster.sample_bilinear(dtm, centre)[0]))
        if top - base < min_height:
            continue
        new = walls_from_footprint(ring, base, top, start_id=start_id + len(walls))
        # The parameters are the point. Executing them produced `new`, and
        # keeping them is what lets a wall lost to a crop or an occlusion be
        # regenerated instead of predicted.
        program = program_ir.extrusion(f"bldg.{f:04d}", ring, base, top,
                                       roof="flat", source="footprint")
        program.notes = (f"{len(new)} walls from {program.cost} parameters; "
                         "roof form not inferred, so the envelope is a prism")
        for patch in new:
            patch.attrs["program"] = program.id
        programs.append(program)
        walls.extend(new)
    return walls, programs

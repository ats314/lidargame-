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


#: How far a footprint has to turn at a vertex before that vertex is a corner
#: somebody would call a corner. A register's footprint carries vertices that
#: are not corners at all -- collinear points from digitising, and one or two
#: degrees of survey noise along a straight frontage. Below this the wall is
#: continuing, not turning.
CORNER_TURN_DEG = 25.0


def _turns(ring: np.ndarray, clockwise: bool) -> np.ndarray:
    """Signed turn at each vertex of a closed ring, in degrees.

    Positive is convex -- the building turning outward, the kind of corner a
    quoin belongs on. Negative is a reflex corner, the inside of an L. Near
    zero is a straight frontage that merely has a vertex in it.
    """
    points = np.asarray(ring, dtype=float)[:, :2]
    if len(points) > 1 and np.allclose(points[0], points[-1]):
        points = points[:-1]
    if len(points) < 3:
        return np.zeros(len(points))

    incoming = points - np.roll(points, 1, axis=0)
    outgoing = np.roll(points, -1, axis=0) - points
    cross = incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0]
    dot = (incoming * outgoing).sum(axis=1)
    turn = np.degrees(np.arctan2(cross, dot))
    # For a clockwise ring the exterior is on the other side, so an outward
    # turn has the opposite sign.
    return -turn if clockwise else turn


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

    turn = _turns(ring, clockwise)
    vertices = len(turn)

    def corner_kind(index: int) -> str:
        """What a theme should paint at this footprint vertex."""
        if not vertices:
            return "flat"
        degrees = float(turn[index % vertices])
        if degrees >= CORNER_TURN_DEG:
            return "convex"
        if degrees <= -CORNER_TURN_DEG:
            return "concave"
        return "flat"

    for index, (a, b) in enumerate(zip(ring[:-1], ring[1:])):
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

        # Which end of *this wall's own u axis* each footprint vertex lands on.
        # plane_frame is free to point u either way along the edge, so deciding
        # by projection is the only thing that cannot get it backwards -- and a
        # quoin painted on the wrong end is invisible in a test and obvious in
        # a render.
        start = np.array([a[0], a[1], mid_z]) - centroid
        end = np.array([b[0], b[1], mid_z]) - centroid
        at_start, at_end = corner_kind(index), corner_kind(index + 1)
        if float(start @ u) <= float(end @ u):
            patch.attrs["corner_u_min"], patch.attrs["corner_u_max"] = at_start, at_end
        else:
            patch.attrs["corner_u_min"], patch.attrs["corner_u_max"] = at_end, at_start
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


#: US survey feet to metres. Denver publishes BLDG_HEIGH and GROUND_ELE in feet.
FOOT = 0.3048


def published_height(attrs, index: int) -> float | None:
    """Building height in metres from the footprint layer, if it carries one.

    Always metres above the building's own lowest ground point, whatever the
    layer publishes -- `data.gis.attributes` normalises feet to metres and an
    absolute NAP elevation to a height. That conversion used to live here as a
    Denver-shaped constant, which read a 3D BAG metre as a foot and made every
    Amsterdam building a third of its height.

    It is returned as a height rather than an elevation because converting an
    elevation would drag in the vertical datum, while a height rides on the DTM
    the compiler already trusts.

    Caveat worth knowing: the outline is digitised at the eave, and for a
    pitched roof the height is measured to the ridge. Extruding straight to it
    runs a pitched building tall. Downtown stock is overwhelmingly flat-roofed
    so this barely shows here; it will bite in residential tiles.
    """
    if not attrs or index >= len(attrs):
        return None
    value = attrs[index].get("height")
    if value is None:
        return None
    try:
        metres = float(value)
    except (TypeError, ValueError):
        return None
    # A zero height is a Foundation/Ruin record, not a building.
    return metres if 2.0 < metres < 700.0 else None


def build(rings, assignment: np.ndarray, patches, cloud, raster, dtm, *,
          min_height: float = 2.5, start_id: int = 0, attrs=None, log=None):
    """Extrude every footprint whose roof height is known, measured or published.

    Returns (walls, programs). The programs are the generative description
    the walls came out of -- a ring and two heights each -- kept so the
    envelope can be re-executed rather than only rendered.

    A measured roof wins when there is one: it is in the compiler's own datum by
    construction, and using it keeps the extrusion answerable to the returns.
    The published height is the fallback, and it is what lets a footprint with
    no roof returns at all become a building instead of nothing.
    """
    if not len(rings):
        return [], []

    by_footprint: dict[int, list[int]] = {}
    for i, f in enumerate(assignment):
        if f >= 0:
            by_footprint.setdefault(int(f), []).append(i)

    walls: list[PlanarPatch] = []
    programs = []
    # Every footprint is a candidate, not only the ones a patch landed on.
    for f in range(len(rings)):
        group = by_footprint.get(f, [])
        ring = rings[f]
        centre = ring.mean(axis=0)[None, :]
        base = float(np.nan_to_num(raster.sample_bilinear(dtm, centre)[0]))

        top = roof_height(patches, group, cloud) if group else None
        source = "measured"
        stated = published_height(attrs, f)
        if top is None:
            if stated is None:
                continue
            top, source = base + stated, "published"
        if top - base < min_height:
            continue
        new = walls_from_footprint(ring, base, top, start_id=start_id + len(walls))
        # The parameters are the point. Executing them produced `new`, and
        # keeping them is what lets a wall lost to a crop or an occlusion be
        # regenerated instead of predicted.
        program = program_ir.extrusion(f"bldg.{f:04d}", ring, base, top,
                                       roof="flat", source=source)
        # The publisher's own building id, carried through to the seed. Without
        # it a later comparison against the same register has to join on
        # geometry and guess, which is a worse answer to a question the data
        # already answers -- 3D BAG states a height per BAG id, and so does this.
        published_id = (attrs[f].get("source_id")
                        if attrs and f < len(attrs) else None)
        if published_id:
            program.params["source_id"] = str(published_id)
        agree = ""
        if source == "measured" and stated is not None:
            # Both numbers exist: report the disagreement rather than hiding it.
            delta = (top - base) - stated
            program.params["published_height"] = round(stated, 2)
            program.params["height_delta"] = round(delta, 2)
            agree = f"; published height differs by {delta:+.1f} m"
        program.notes = (f"{len(new)} walls from {program.cost} parameters, "
                         f"height {source}{agree}; roof form not inferred, so "
                         "the envelope is a prism")
        for patch in new:
            patch.attrs["program"] = program.id
        programs.append(program)
        walls.extend(new)

        if log is not None:
            # This pass is Pass B, building envelope closure, and it has always
            # been a repair -- airborne returns do not describe these walls, the
            # footprint and a roof height do. Recording it is what stops an
            # invented wall being counted as evidence downstream.
            #
            # Tier 3: the wall is a geometric consequence of an authoritative
            # boundary and a height, not a guess from context. The output is
            # still `inferred` rather than `derived`, because the *evidence* for
            # a facade the sensor never saw is the footprint's existence, not a
            # measurement of the facade.
            log.repair(
                pass_name="building_closure", tier=3,
                operation="extrude_envelope_from_footprint",
                target_entity_id=program.id,
                epistemic_output_state="inferred",
                output_geometry_ids=[f"patch.{p.id}" for p in new],
                evidence_ids=([f"footprint.{f:04d}"]
                              + [f"patch.{patches[i].id}" for i in group]),
                reason=(f"airborne returns do not describe these facades; "
                        f"{len(new)} walls closed from the footprint to a "
                        f"{source} roof height"),
                confidence=0.75 if source == "measured" else 0.45,
                max_displacement=round(float(top - base), 3),
                algorithm="lidarworld.reconstruct.extrude.build",
                parameters={"ring_vertices": int(len(ring)), "base_z": round(base, 3),
                            "top_z": round(float(top), 3), "height_source": source,
                            "min_height": min_height,
                            "published_height": program.params.get("published_height")},
            )
    return walls, programs

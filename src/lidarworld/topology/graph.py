"""World topology: how the pieces relate.

Relations are the part a mesh cannot carry. "This wall is perpendicular to that
wall" is what turns two flat rectangles into a building corner; "this wall
borders a road" is what tells a theme to put a shopfront on one side and a
service door on the other. The graph is also where confidence and provenance
live, so a backend can choose to render only what was actually observed.

Cross-patch context flags are written back into the tile lattices here, because
a patch cannot know it has an inside corner until it knows its neighbours.
"""
from __future__ import annotations

import numpy as np

from ..roles.taxonomy import Ctx

PERPENDICULAR_COS = 0.30      # |n_a . n_b| below this -> treat as perpendicular
PARALLEL_COS = 0.94


def _aabb(patch, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return points.min(axis=0), points.max(axis=0)


def relate_patches(patches, cloud, *, tol: float = 0.8, max_probe: int = 900):
    """Pairwise relations between planar patches.

    Returns a list of ``(i, j, relation, confidence, attrs)``. Candidate pairs
    are filtered by axis-aligned bounds first; the expensive test (does B have
    points lying on A's plane, inside A's extent) only runs on survivors.
    """
    n = len(patches)
    if n == 0:
        return []

    boxes = []
    probes = []
    for p in patches:
        pts = cloud.xyz[p.point_idx]
        boxes.append(_aabb(p, pts))
        step = max(1, len(pts) // max_probe)
        probes.append(pts[::step])

    out = []
    for i in range(n):
        lo_i, hi_i = boxes[i]
        for j in range(i + 1, n):
            lo_j, hi_j = boxes[j]
            if np.any(lo_i - tol > hi_j) or np.any(lo_j - tol > hi_i):
                continue

            a, b = patches[i], patches[j]
            cos = abs(float(a.normal @ b.normal))

            # Does either patch have points sitting on the other's plane?
            dist_ba = np.abs(probes[j] @ a.normal + a.offset)
            dist_ab = np.abs(probes[i] @ b.normal + b.offset)
            touch_ba = int((dist_ba < tol).sum())
            touch_ab = int((dist_ab < tol).sum())
            if touch_ba == 0 and touch_ab == 0:
                continue
            strength = max(touch_ba / len(probes[j]), touch_ab / len(probes[i]))
            conf = float(np.clip(0.25 + 1.5 * strength, 0.1, 0.95))

            if cos < PERPENDICULAR_COS:
                rel = "perpendicular_to"
            elif cos > PARALLEL_COS:
                gap = abs(float(a.offset - b.offset)) if float(a.normal @ b.normal) > 0 \
                    else abs(float(a.offset + b.offset))
                rel = "coplanar_with" if gap < tol else "parallel_to"
            else:
                rel = "adjacent_to"
            out.append((i, j, rel, conf, {"cos": round(cos, 3), "contact": strength}))
    return out


def group_structures(patches, relations, *, min_patches: int = 1) -> list[list[int]]:
    """Connected components over touching patches -- one component per building."""
    n = len(patches)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j, rel, conf, _ in relations:
        if rel in ("perpendicular_to", "adjacent_to", "coplanar_with") and conf > 0.3:
            ra, rb = find(i), find(j)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [g for g in groups.values() if len(g) >= min_patches]


def annotate_cross_patch_context(patches, lattices, relations, cloud, *,
                                 band: float = 0.6, max_probe: int = 1200) -> None:
    """Write ADJ_* flags into each lattice using its neighbours' geometry.

    For every related pair, the neighbour's points that lie close to this
    patch's plane are projected into this patch's lattice, and the cells they
    land on (plus a band around them) get flagged. That gives inside corners,
    eave bands and coplanar seams without any explicit line-intersection maths.
    """
    for i, j, rel, conf, _ in relations:
        for src, dst in ((i, j), (j, i)):
            patch = patches[dst]
            lattice = lattices.get(dst)
            if lattice is None:
                continue
            other = patches[src]
            pts = cloud.xyz[other.point_idx]
            step = max(1, len(pts) // max_probe)
            pts = pts[::step]
            near = np.abs(pts @ patch.normal + patch.offset) < band * 1.6
            if not near.any():
                continue
            uv = patch.project(pts[near])
            iu = ((uv[:, 0] - lattice.uv_origin[0]) / lattice.cell).astype(np.int64)
            iv = ((uv[:, 1] - lattice.uv_origin[1]) / lattice.cell).astype(np.int64)
            nu, nv = lattice.shape
            keep = (iu >= 0) & (iv >= 0) & (iu < nu) & (iv < nv)
            if not keep.any():
                continue
            mask = np.zeros((nu, nv), dtype=bool)
            mask[iu[keep], iv[keep]] = True
            from ..reconstruct.lattice import _dilate
            mask = _dilate(mask, max(1, int(round(band / lattice.cell))))
            solid = lattice.occupancy.astype(bool)

            if rel == "perpendicular_to":
                flag = Ctx.ADJ_PERPENDICULAR
                # A perpendicular neighbour makes these cells a real corner, not
                # merely a lattice boundary.
                lattice.context[mask & solid] |= Ctx.CORNER_CONCAVE
            elif rel == "coplanar_with":
                flag = Ctx.ADJ_COPLANAR
            else:
                flag = Ctx.ADJ_ROOF if other.role.startswith("surface.roof") else Ctx.ADJ_PERPENDICULAR
            lattice.context[mask & solid] |= flag

            if other.role.startswith("surface.roof") and patch.role.startswith("surface.wall"):
                lattice.context[mask & solid] |= Ctx.SHELTERED


def mark_street_facing(patches, lattices, raster, road_mask, *, probe: float = 6.0) -> int:
    """Flag wall patches whose outward normal looks onto a road surface."""
    if road_mask is None or not road_mask.any():
        return 0
    marked = 0
    for i, patch in enumerate(patches):
        if not patch.role.startswith("surface.wall"):
            continue
        lattice = lattices.get(i)
        if lattice is None:
            continue
        outward = patch.normal.copy()
        outward[2] = 0.0
        norm = np.linalg.norm(outward)
        if norm < 1e-6:
            continue
        outward /= norm
        hit = False
        for sign in (1.0, -1.0):
            for dist in (2.0, probe * 0.5, probe):
                probe_xy = patch.centroid[:2] + sign * outward[:2] * dist
                ij = raster.to_cell(probe_xy[None, :])
                if road_mask[ij[0, 0], ij[0, 1]]:
                    hit = True
                    if sign < 0:
                        patch.normal = -patch.normal
                        patch.offset = -patch.offset
                    break
            if hit:
                break
        if hit:
            solid = lattice.occupancy.astype(bool)
            lattice.context[solid] |= Ctx.STREET_FACING
            patch.attrs["street_facing"] = True
            marked += 1
    return marked


def summarize(relations) -> dict[str, int]:
    from collections import Counter
    return dict(Counter(rel for _, _, rel, _, _ in relations))

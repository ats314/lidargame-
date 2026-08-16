"""Tile lattices: where a patch of dots becomes an addressable surface.

Every planar patch gets a 2D lattice in its own plane. Each cell records whether
it is solid, how it was arrived at (measured / closed / inferred), and -- the
important part -- a **context bitmask** describing where the cell sits relative
to everything around it: on the free edge, on a convex corner, one cell from a
window, in the deep interior, touching the ground.

That mask is the "relative index". A theme rule can ask for
``surface.wall.vertical`` + ``corner_convex`` and get quoin stones on exactly
the right cells, in any theme, without knowing anything about this building.

Openings fall out of the same structure for free: a window is a hole in the
returns that is enclosed by solid cells, because glass does not send a pulse
back. Rather than patching that hole, we keep it -- it is real geometry.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..roles.taxonomy import Ctx

try:
    from scipy import ndimage as _ndi
except ImportError:                      # pragma: no cover
    _ndi = None


@dataclass
class Opening:
    id: int
    role: str                        # opening.window | opening.door | opening.unknown
    cells: np.ndarray                # (K,2) lattice coordinates
    uv_min: tuple[float, float]
    uv_max: tuple[float, float]
    center_world: np.ndarray
    width: float
    height: float
    sill_height: float               # metres above the patch's lowest edge
    confidence: float


@dataclass
class TileLattice:
    cell: float
    shape: tuple[int, int]           # (nu, nv)
    uv_origin: tuple[float, float]   # patch-local uv of cell (0,0) corner
    occupancy: np.ndarray            # (nu,nv) uint8, 1 = solid surface
    context: np.ndarray              # (nu,nv) uint32 bitmask
    evidence: np.ndarray             # (nu,nv) uint16, source points per cell
    openings: list[Opening] = field(default_factory=list)

    @property
    def solid_count(self) -> int:
        return int(self.occupancy.sum())

    def cell_uv(self, iu: np.ndarray, iv: np.ndarray) -> np.ndarray:
        return np.column_stack([
            self.uv_origin[0] + (iu + 0.5) * self.cell,
            self.uv_origin[1] + (iv + 0.5) * self.cell,
        ])


# --- small morphology helpers (scipy when present, numpy when not) ---------

def _dilate(mask: np.ndarray, r: int = 1, diagonal: bool = True) -> np.ndarray:
    if _ndi is not None:
        structure = np.ones((3, 3), bool) if diagonal else None
        return _ndi.binary_dilation(mask, structure=structure, iterations=r)
    out = mask.copy()
    for _ in range(r):
        acc = out.copy()
        acc[1:] |= out[:-1]; acc[:-1] |= out[1:]
        acc[:, 1:] |= out[:, :-1]; acc[:, :-1] |= out[:, 1:]
        if diagonal:
            acc[1:, 1:] |= out[:-1, :-1]; acc[:-1, :-1] |= out[1:, 1:]
            acc[1:, :-1] |= out[:-1, 1:]; acc[:-1, 1:] |= out[1:, :-1]
        out = acc
    return out


def _erode(mask: np.ndarray, r: int = 1) -> np.ndarray:
    return ~_dilate(~mask, r)


def label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """4-connected labelling of True cells. Returns (labels, count)."""
    if _ndi is not None:
        labels, count = _ndi.label(mask)
        return labels.astype(np.int32), int(count)
    nu, nv = mask.shape
    labels = np.zeros((nu, nv), dtype=np.int32)
    parent: list[int] = [0]

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    nxt = 1
    for i in range(nu):
        for j in range(nv):
            if not mask[i, j]:
                continue
            up = labels[i - 1, j] if i > 0 else 0
            left = labels[i, j - 1] if j > 0 else 0
            if up and left:
                labels[i, j] = min(up, left)
                union(up, left)
            elif up or left:
                labels[i, j] = up or left
            else:
                labels[i, j] = nxt
                parent.append(nxt)
                nxt += 1
    remap: dict[int, int] = {}
    for i in range(nu):
        for j in range(nv):
            if labels[i, j]:
                root = find(labels[i, j])
                labels[i, j] = remap.setdefault(root, len(remap) + 1)
    return labels, len(remap)


def distance_to_false(mask: np.ndarray, limit: int = 15) -> np.ndarray:
    """Chebyshev distance from each True cell to the nearest False cell.

    Successive erosions: a cell that survives k of them is k steps from the
    outside. Bounded by `limit` because the only consumer is an "am I deep in
    the interior" test.
    """
    if _ndi is not None:
        return np.minimum(
            _ndi.distance_transform_cdt(mask, metric="chessboard"), limit).astype(np.uint8)
    dist = np.zeros(mask.shape, dtype=np.uint8)
    current = mask.copy()
    for d in range(1, limit + 1):
        if not current.any():
            break
        dist[current] = d
        current = _erode(current, 1)
    return dist


def build(patch, xyz: np.ndarray, *, cell: float = 0.25, close_radius: int = 1,
          min_opening_area: float = 0.35, max_opening_area: float = 14.0,
          ground_z: float | None = None, interior_depth: int = 3) -> TileLattice:
    """Lay a lattice over `patch` using the world points `xyz` that belong to it."""
    uv = patch.project(xyz)
    uv_min = uv.min(axis=0) - cell
    uv_max = uv.max(axis=0) + cell
    nu = max(1, int(np.ceil((uv_max[0] - uv_min[0]) / cell)))
    nv = max(1, int(np.ceil((uv_max[1] - uv_min[1]) / cell)))

    iu = np.clip(((uv[:, 0] - uv_min[0]) / cell).astype(np.int64), 0, nu - 1)
    iv = np.clip(((uv[:, 1] - uv_min[1]) / cell).astype(np.int64), 0, nv - 1)
    evidence = np.bincount(iu * nv + iv, minlength=nu * nv).reshape(nu, nv)
    measured = evidence > 0

    # Close 1-cell scan gaps, then fill anything fully enclosed. The difference
    # between the two is the set of candidate openings.
    closed = _erode(_dilate(measured, close_radius), close_radius) | measured
    empty_labels, n_empty = label_components(~closed)
    border = set(np.unique(np.concatenate([
        empty_labels[0, :], empty_labels[-1, :], empty_labels[:, 0], empty_labels[:, -1]])))
    border.discard(0)
    enclosed = (empty_labels > 0) & ~np.isin(empty_labels, list(border))

    solid = closed | enclosed
    occupancy = solid.astype(np.uint8)

    # --- openings -------------------------------------------------------
    openings: list[Opening] = []
    if n_empty:
        hole_labels, n_holes = label_components(enclosed)
        cell_area = cell * cell
        patch_bottom_v = None
        for lab in range(1, n_holes + 1):
            cells = np.argwhere(hole_labels == lab)
            area = len(cells) * cell_area
            if area < min_opening_area or area > max_opening_area:
                continue
            u0, v0 = cells.min(axis=0)
            u1, v1 = cells.max(axis=0)
            width = (u1 - u0 + 1) * cell
            height = (v1 - v0 + 1) * cell
            fill_ratio = len(cells) / max(1, (u1 - u0 + 1) * (v1 - v0 + 1))
            aspect = max(width, height) / max(1e-6, min(width, height))
            if fill_ratio < 0.55 or aspect > 6.0:
                continue                       # ragged blob, not an opening
            if patch_bottom_v is None:
                occupied_v = np.argwhere(solid)[:, 1]
                patch_bottom_v = int(occupied_v.min()) if occupied_v.size else 0
            sill = (v0 - patch_bottom_v) * cell
            center_uv = np.array([[uv_min[0] + (u0 + u1 + 1) / 2 * cell,
                                   uv_min[1] + (v0 + v1 + 1) / 2 * cell]])
            center_world = patch.unproject(center_uv)[0]
            role = "opening.door" if sill < 0.5 and height > 1.4 else "opening.window"
            if height < 0.5 or width < 0.35:
                role = "opening.unknown"
            openings.append(Opening(
                id=len(openings), role=role, cells=cells,
                uv_min=(float(uv_min[0] + u0 * cell), float(uv_min[1] + v0 * cell)),
                uv_max=(float(uv_min[0] + (u1 + 1) * cell), float(uv_min[1] + (v1 + 1) * cell)),
                center_world=center_world, width=float(width), height=float(height),
                sill_height=float(sill),
                confidence=float(np.clip(0.3 + 0.5 * fill_ratio, 0.15, 0.9)),
            ))
            occupancy[hole_labels == lab] = 0     # the hole is real geometry

    solid = occupancy.astype(bool)

    # --- context bitmask -------------------------------------------------
    ctx = np.zeros((nu, nv), dtype=np.uint32)
    ctx[solid] |= Ctx.OCCUPIED

    def shifted(mask, du, dv):
        out = np.zeros_like(mask)
        src_u = slice(max(0, -du), nu - max(0, du))
        dst_u = slice(max(0, du), nu - max(0, -du))
        src_v = slice(max(0, -dv), nv - max(0, dv))
        dst_v = slice(max(0, dv), nv - max(0, -dv))
        out[dst_u, dst_v] = mask[src_u, src_v]
        return out

    left = shifted(solid, 1, 0)      # neighbour at u-1
    right = shifted(solid, -1, 0)
    down = shifted(solid, 0, 1)
    up = shifted(solid, 0, -1)

    ctx[solid & ~left] |= Ctx.EDGE_U_MIN
    ctx[solid & ~right] |= Ctx.EDGE_U_MAX
    ctx[solid & ~down] |= Ctx.EDGE_V_MIN
    ctx[solid & ~up] |= Ctx.EDGE_V_MAX

    horiz_edge = (~left | ~right)
    vert_edge = (~down | ~up)
    ctx[solid & horiz_edge & vert_edge] |= Ctx.CORNER_CONVEX

    depth = distance_to_false(solid)
    ctx[solid & (depth >= interior_depth)] |= Ctx.INTERIOR

    # v is the up-slope axis, so the extreme rows are the top and bottom bands.
    if solid.any():
        vs = np.argwhere(solid)[:, 1]
        v_lo, v_hi = int(vs.min()), int(vs.max())
        band = max(1, int(round(0.35 / cell)))
        ctx[solid & (np.arange(nv)[None, :] <= v_lo + band - 1)] |= Ctx.BOTTOM
        ctx[solid & (np.arange(nv)[None, :] >= v_hi - band + 1)] |= Ctx.TOP

    if openings:
        opening_mask = np.zeros((nu, nv), dtype=bool)
        for o in openings:
            opening_mask[o.cells[:, 0], o.cells[:, 1]] = True
        boundary = _dilate(opening_mask, 1) & solid
        near = _dilate(opening_mask, max(1, int(round(0.6 / cell)))) & solid
        ctx[boundary] |= Ctx.OPENING_BOUNDARY
        ctx[near] |= Ctx.NEAR_OPENING

    inferred = solid & ~measured
    ctx[inferred] |= Ctx.SPARSE_EVIDENCE
    ctx[solid & enclosed & ~measured] |= Ctx.OCCLUDED

    if ground_z is not None:
        centers_uv = np.stack(np.meshgrid(np.arange(nu), np.arange(nv), indexing="ij"), axis=-1)
        flat = centers_uv.reshape(-1, 2)
        world = patch.unproject(np.column_stack([
            uv_min[0] + (flat[:, 0] + 0.5) * cell,
            uv_min[1] + (flat[:, 1] + 0.5) * cell]))
        touching = (world[:, 2] - ground_z < 0.45).reshape(nu, nv)
        ctx[solid & touching] |= Ctx.GROUND_CONTACT

    return TileLattice(cell=cell, shape=(nu, nv), uv_origin=(float(uv_min[0]), float(uv_min[1])),
                       occupancy=occupancy, context=ctx,
                       evidence=np.minimum(evidence, 65535).astype(np.uint16),
                       openings=openings)


def context_histogram(lattice: TileLattice) -> dict[str, int]:
    solid = lattice.occupancy.astype(bool)
    ctx = lattice.context[solid]
    return {name: int((ctx & bit).astype(bool).sum())
            for bit, name in sorted(Ctx.NAMES.items()) if (ctx & bit).any()}

"""Geometry assembly.

The output mesh is **theme-independent**. Every vertex carries its role and its
context bitmask as attributes, and no material is baked in anywhere. Swapping a
theme is then a pure lookup -- (role, context) -> material -- which a viewer can
redo at runtime without re-running a single stage of the compiler. That property
is the reason the context mask is an attribute rather than a submesh split.

Quads are merged greedily, and only cells whose context masks are *identical*
merge together, so the attribute stays exact after merging.
"""
from __future__ import annotations

import numpy as np

from ..roles.taxonomy import Ctx, ROLE_INDEX


class MeshBuilder:
    """Accumulates a single attribute-interleaved mesh."""

    def __init__(self):
        self.positions: list[np.ndarray] = []
        self.normals: list[np.ndarray] = []
        self.uvs: list[np.ndarray] = []
        self.ctx: list[np.ndarray] = []
        self.roles: list[np.ndarray] = []
        self.nodes: list[np.ndarray] = []
        self.indices: list[np.ndarray] = []
        self._vertex_count = 0
        self.quad_count = 0

    def add_quads(self, corners: np.ndarray, normal: np.ndarray, uv: np.ndarray,
                  ctx: np.ndarray, role_id: int, node_id: int) -> None:
        """corners: (Q,4,3) in winding order; uv: (Q,4,2); ctx: (Q,)."""
        q = len(corners)
        if q == 0:
            return
        base = self._vertex_count
        # float64, deliberately, and the only array here that is.
        #
        # These are projected coordinates: a UTM northing is about 5.93e6, where
        # float32 resolves 0.5 m, and an easting about 5.7e5, where it resolves
        # 0.0625 m. Casting on the way in snapped every vertex to that grid --
        # measured on a Hamburg block, the smallest distinct northing in the
        # exported buffer was exactly 0.5 m against 0.0625 m in easting. The
        # result is an anisotropic sawtooth on any wall not aligned to the axes,
        # which is every wall in a rotated street grid, and it moved no metric.
        #
        # Precision has to be kept until something recentres. Backends do that
        # and cast there, where the local range is metres and float32 is ample.
        self.positions.append(corners.reshape(-1, 3).astype(np.float64))
        if normal.ndim == 1:
            self.normals.append(np.repeat(normal[None, :], q * 4, axis=0).astype(np.float32))
        else:
            self.normals.append(np.repeat(normal, 4, axis=0).astype(np.float32))
        self.uvs.append(uv.reshape(-1, 2).astype(np.float32))
        self.ctx.append(np.repeat(ctx, 4).astype(np.uint32))
        self.roles.append(np.full(q * 4, role_id, dtype=np.uint8))
        self.nodes.append(np.full(q * 4, node_id, dtype=np.uint32))
        offsets = base + 4 * np.arange(q, dtype=np.uint32)[:, None]
        tris = np.concatenate([
            offsets + np.array([0, 1, 2], dtype=np.uint32),
            offsets + np.array([0, 2, 3], dtype=np.uint32),
        ])
        self.indices.append(tris)
        self._vertex_count += q * 4
        self.quad_count += q

    def finalize(self) -> dict[str, np.ndarray]:
        if not self.positions:
            return {
                "positions": np.zeros((0, 3), np.float64), "normals": np.zeros((0, 3), np.float32),
                "uv": np.zeros((0, 2), np.float32), "ctx": np.zeros(0, np.uint32),
                "role": np.zeros(0, np.uint8), "node": np.zeros(0, np.uint32),
                "indices": np.zeros((0, 3), np.uint32),
            }
        return {
            "positions": np.concatenate(self.positions),
            "normals": np.concatenate(self.normals),
            "uv": np.concatenate(self.uvs),
            "ctx": np.concatenate(self.ctx),
            "role": np.concatenate(self.roles),
            "node": np.concatenate(self.nodes),
            "indices": np.concatenate(self.indices),
        }


def greedy_rects(solid: np.ndarray, key: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """Merge equal-key cells into maximal rectangles.

    Returns ``(u0, v0, width, height, key)`` tuples covering every solid cell
    exactly once. Standard greedy meshing: extend along u while the key holds,
    then extend that whole run along v.
    """
    nu, nv = solid.shape
    used = np.zeros_like(solid, dtype=bool)
    out = []
    for i in range(nu):
        j = 0
        while j < nv:
            if not solid[i, j] or used[i, j]:
                j += 1
                continue
            k = int(key[i, j])
            # extend along v
            j2 = j
            while j2 + 1 < nv and solid[i, j2 + 1] and not used[i, j2 + 1] and key[i, j2 + 1] == k:
                j2 += 1
            height = j2 - j + 1
            # extend along u while the whole column run matches
            i2 = i
            while i2 + 1 < nu:
                row = slice(j, j2 + 1)
                if (solid[i2 + 1, row].all() and not used[i2 + 1, row].any()
                        and (key[i2 + 1, row] == k).all()):
                    i2 += 1
                else:
                    break
            used[i:i2 + 1, j:j2 + 1] = True
            out.append((i, j, i2 - i + 1, height, k))
            j = j2 + 1
    return out


def add_lattice(builder: MeshBuilder, patch, lattice, node_index: int,
                *, double_sided: bool = False) -> int:
    """Emit merged quads for one tiled plane. Returns the quad count."""
    solid = lattice.occupancy.astype(bool)
    if not solid.any():
        return 0
    rects = greedy_rects(solid, lattice.context)
    if not rects:
        return 0

    cell = lattice.cell
    u0 = np.array([r[0] for r in rects], dtype=np.float64)
    v0 = np.array([r[1] for r in rects], dtype=np.float64)
    w = np.array([r[2] for r in rects], dtype=np.float64)
    h = np.array([r[3] for r in rects], dtype=np.float64)
    keys = np.array([r[4] for r in rects], dtype=np.uint32)

    ou, ov = lattice.uv_origin
    a_u = ou + u0 * cell
    a_v = ov + v0 * cell
    b_u = a_u + w * cell
    b_v = a_v + h * cell

    uv_corners = np.stack([
        np.column_stack([a_u, a_v]), np.column_stack([b_u, a_v]),
        np.column_stack([b_u, b_v]), np.column_stack([a_u, b_v]),
    ], axis=1)                                   # (Q,4,2)

    flat = uv_corners.reshape(-1, 2)
    world = patch.unproject(flat).reshape(-1, 4, 3)
    # World-scale UVs so a texture tiles at true metres in any theme.
    tex_uv = flat.reshape(-1, 4, 2)

    builder.add_quads(world, patch.normal, tex_uv, keys,
                      ROLE_INDEX.get(patch.role, ROLE_INDEX["unknown"]), node_index)
    if double_sided:
        builder.add_quads(world[:, ::-1], -patch.normal, tex_uv[:, ::-1], keys,
                          ROLE_INDEX.get(patch.role, ROLE_INDEX["unknown"]), node_index)
    return len(rects)


#: How far behind the wall plane the glass sits. A window is not flush with the
#: brick; the reveal is what makes an opening read as depth rather than as a
#: hole cut in cardboard, and it is the cheapest depth cue on a facade because
#: the surround casts a shadow onto it whenever the sun is off-axis.
GLAZING_INSET_M = 0.10


def add_glazing(builder: MeshBuilder, patch, lattice, node_index: int,
                *, inset: float = GLAZING_INSET_M) -> int:
    """Fill each opening with a pane, set back from the wall. Returns the count.

    Until this existed, an opening was a hole and nothing else: `add_lattice`
    skipped its cells, so a viewer looked straight through the building to the
    sky behind it. Every theme pack already carries a rule binding
    `opening.window` to a glass material -- the rule simply had no triangle to
    resolve against, because openings had never produced geometry.

    The pane carries the opening's own role, not a material. What glass *is*
    remains a backend decision, exactly as for every other surface.
    """
    if not lattice.openings:
        return 0

    corners, keys, roles = [], [], []
    for opening in lattice.openings:
        (u0, v0), (u1, v1) = opening.uv_min, opening.uv_max
        if u1 <= u0 or v1 <= v0:
            continue
        corners.append([[u0, v0], [u1, v0], [u1, v1], [u0, v1]])
        # OCCUPIED because a pane is real surface -- the flag is geometric,
        # and provenance lives in the two beside it. OCCLUDED because glass is
        # never observed by this sensor even when the opening was detected:
        # the detector infers a window from an *absence* of returns, which is
        # precisely a surface the LiDAR could not see.
        keys.append(int(Ctx.OCCUPIED | Ctx.OPENING_BOUNDARY
                        | Ctx.OCCLUDED | Ctx.SPARSE_EVIDENCE))
        roles.append(ROLE_INDEX.get(opening.role, ROLE_INDEX["unknown"]))

    if not corners:
        return 0

    uv = np.asarray(corners, dtype=np.float64)                  # (Q,4,2)
    world = patch.unproject(uv.reshape(-1, 2)).reshape(-1, 4, 3)
    world = world - patch.normal * inset

    key_array = np.asarray(keys, dtype=np.uint32)
    # One call per role so a door and a window do not share a role id. There
    # are at most three, so the draw-call cost is nil.
    for role_id in sorted(set(roles)):
        pick = np.array([r == role_id for r in roles])
        builder.add_quads(world[pick], patch.normal, uv[pick], key_array[pick],
                          role_id, node_index)
    return len(corners)


def corner_heights(dtm: np.ndarray) -> np.ndarray:
    """Cell-centred heights -> (nx+1, ny+1) corner heights.

    Terrain quads have to share corners or the mesh cracks along every cell
    boundary, so heights move to the lattice corners once, here, and every quad
    reads them rather than sampling the cell centre.
    """
    filled = np.nan_to_num(dtm, nan=float(np.nanmedian(dtm)) if np.isfinite(dtm).any() else 0.0)
    padded = np.pad(filled, 1, mode="edge")
    return 0.25 * (padded[:-1, :-1] + padded[1:, :-1] + padded[:-1, 1:] + padded[1:, 1:])


def add_terrain(builder: MeshBuilder, raster, dtm: np.ndarray, class_raster: np.ndarray,
                ctx_raster: np.ndarray, role_lookup: dict[int, str], node_index: int,
                *, mask: np.ndarray | None = None, height_bucket: float = 0.12) -> int:
    """Emit the ground, split by surface role.

    Cells merge into larger quads only when they share a context mask *and* sit
    at the same height: a quad is planar by construction, so merging across a
    slope would flatten the terrain between its corners. Quantising height into
    buckets keeps flat carriageways cheap while preserving relief.
    """
    nx, ny = raster.nx, raster.ny
    gx, gy = raster.cell_centers()
    half = raster.cell / 2.0
    corners_z = corner_heights(dtm)
    total = 0

    valid = np.isfinite(dtm) if mask is None else (np.isfinite(dtm) & mask)
    buckets = np.round(np.nan_to_num(dtm, nan=0.0) / height_bucket).astype(np.int64)
    buckets -= buckets.min(initial=0)
    merge_key = (ctx_raster.astype(np.int64) << 20) | np.minimum(buckets, (1 << 20) - 1)

    for class_id, role_id in role_lookup.items():
        sel = valid & (class_raster == class_id)
        if not sel.any():
            continue
        rects = greedy_rects(sel, merge_key)
        if not rects:
            continue
        corners = np.empty((len(rects), 4, 3), dtype=np.float64)
        uvs = np.empty((len(rects), 4, 2), dtype=np.float64)
        keys = np.empty(len(rects), dtype=np.uint32)
        for q, (i, j, w, h, _) in enumerate(rects):
            x0, x1 = gx[i] - half, gx[i + w - 1] + half
            y0, y1 = gy[j] - half, gy[j + h - 1] + half
            z00 = float(corners_z[i, j])
            z10 = float(corners_z[i + w, j])
            z11 = float(corners_z[i + w, j + h])
            z01 = float(corners_z[i, j + h])
            corners[q] = [[x0, y0, z00], [x1, y0, z10], [x1, y1, z11], [x0, y1, z01]]
            uvs[q] = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
            keys[q] = ctx_raster[i, j]
        normals = _quad_normals(corners)
        builder.add_quads(corners, normals, uvs, keys,
                          ROLE_INDEX.get(role_id, ROLE_INDEX["unknown"]), node_index)
        total += len(rects)
    return total


def _quad_normals(corners: np.ndarray) -> np.ndarray:
    e1 = corners[:, 1] - corners[:, 0]
    e2 = corners[:, 3] - corners[:, 0]
    n = np.cross(e1, e2)
    length = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, np.maximum(length, 1e-9))
    flip = n[:, 2] < 0
    n[flip] *= -1
    return n


def terrain_context(class_raster: np.ndarray, dtm: np.ndarray) -> np.ndarray:
    """Context flags for terrain cells: edges of a surface type, slope, holes."""
    ctx = np.zeros(class_raster.shape, dtype=np.uint32)
    ctx |= Ctx.OCCUPIED

    def neighbour_differs(axis, shift):
        rolled = np.roll(class_raster, shift, axis=axis)
        if axis == 0:
            if shift > 0: rolled[0, :] = class_raster[0, :]
            else: rolled[-1, :] = class_raster[-1, :]
        else:
            if shift > 0: rolled[:, 0] = class_raster[:, 0]
            else: rolled[:, -1] = class_raster[:, -1]
        return rolled != class_raster

    edge = (neighbour_differs(0, 1) | neighbour_differs(0, -1)
            | neighbour_differs(1, 1) | neighbour_differs(1, -1))
    # On terrain, EDGE_ANY means "boundary between two surface types" -- the
    # kerb line where road meets ground, which is what a trim rule binds to.
    ctx[edge] |= Ctx.EDGE_ANY
    ctx[~edge] |= Ctx.INTERIOR

    gy, gx = np.gradient(np.nan_to_num(dtm, nan=0.0))
    slope = np.hypot(gx, gy)
    ctx[slope > 0.35] |= Ctx.CORNER_CONVEX
    ctx[~np.isfinite(dtm)] |= Ctx.SPARSE_EVIDENCE
    return ctx

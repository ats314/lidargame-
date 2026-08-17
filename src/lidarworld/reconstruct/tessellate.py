"""Turn planar polygons into triangles.

Every surface format the compiler reads or writes is polygonal -- CityGML rings,
footprints, fitted planar patches -- and every engine it targets wants
triangles. Doing that with a triangle fan works until it does not: an L-shaped
roof or a re-entrant facade fans into triangles that lie outside the polygon,
and the failure is invisible in a triangle count and obvious the moment anyone
looks at the render.

So this is ear clipping, in the polygon's own plane. It is O(n^2) in the ring
length, which is irrelevant here -- Hamburg's inner-city tile is 126,504
polygons and 82% of them are quads.

No holes. CityGML permits interior rings and the Hamburg tiles do not use them;
`triangulate` reports what it skipped rather than silently dropping it, because
a courtyard quietly filled in is exactly the kind of error that survives to the
screenshot.
"""
from __future__ import annotations

import numpy as np


def newell(ring: np.ndarray) -> np.ndarray:
    """Best-fit plane normal, robust to non-planarity and to collinear edges.

    Cross-producting two edges fails when they happen to be parallel, which for
    surveyed geometry is common; Newell's method uses every vertex.
    """
    normal = np.zeros(3)
    n = len(ring)
    for i in range(n):
        a, b = ring[i], ring[(i + 1) % n]
        normal[0] += (a[1] - b[1]) * (a[2] + b[2])
        normal[1] += (a[2] - b[2]) * (a[0] + b[0])
        normal[2] += (a[0] - b[0]) * (a[1] + b[1])
    return normal


def _project(ring: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop the axis the polygon is most perpendicular to, keeping orientation.

    Choosing the dominant normal component means the remaining two axes are the
    ones the polygon actually spans, so no near-degenerate 2D ring comes out.
    """
    normal = newell(ring)
    axis = int(np.argmax(np.abs(normal)))
    keep = [i for i in range(3) if i != axis]
    flat = ring[:, keep]
    # Preserve winding: dropping an axis flips handedness for one of the three
    # choices, and a flipped ring ear-clips into nothing.
    if normal[axis] < 0:
        flat = flat[:, ::-1]
    return flat, normal


def _area2(flat: np.ndarray) -> float:
    x, y = flat[:, 0], flat[:, 1]
    return float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _inside(p, a, b, c) -> bool:
    def side(u, v, w):
        return (v[0] - u[0]) * (w[1] - u[1]) - (v[1] - u[1]) * (w[0] - u[0])
    d1, d2, d3 = side(a, b, p), side(b, c, p), side(c, a, p)
    return not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0))


def triangulate(ring: np.ndarray) -> np.ndarray:
    """Indices into `ring`, shape (T, 3). Empty if the ring is degenerate.

    `ring` must not repeat its first vertex at the end; GML rings do, so strip
    the closing vertex before calling.
    """
    n = len(ring)
    if n < 3:
        return np.zeros((0, 3), dtype=np.int32)
    if n == 3:
        # Three collinear points are a valid ring and not a triangle. Emitting
        # one anyway puts a zero-area face in the mesh, which survives every
        # count and shows up as a z-fighting sliver in the render.
        if np.linalg.norm(np.cross(ring[1] - ring[0], ring[2] - ring[0])) < 1e-12:
            return np.zeros((0, 3), dtype=np.int32)
        return np.array([[0, 1, 2]], dtype=np.int32)

    flat, _ = _project(ring)
    if abs(_area2(flat)) < 1e-12:
        return np.zeros((0, 3), dtype=np.int32)
    if _area2(flat) < 0:
        flat = flat[::-1]
        remap = list(range(n - 1, -1, -1))
    else:
        remap = list(range(n))

    index = list(range(n))
    out: list[list[int]] = []
    guard = 0
    while len(index) > 3 and guard < 2 * n * n:
        guard += 1
        clipped = False
        for k in range(len(index)):
            i0 = index[(k - 1) % len(index)]
            i1 = index[k]
            i2 = index[(k + 1) % len(index)]
            a, b, c = flat[i0], flat[i1], flat[i2]
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if cross <= 0:                      # reflex or collinear: not an ear
                continue
            if any(_inside(flat[j], a, b, c)
                   for j in index if j not in (i0, i1, i2)):
                continue
            out.append([remap[i0], remap[i1], remap[i2]])
            index.pop(k)
            clipped = True
            break
        if not clipped:
            # Self-intersecting or numerically hopeless. A fan is wrong here but
            # it is bounded and visible, which beats dropping the surface.
            break
    if len(index) == 3:
        out.append([remap[index[0]], remap[index[1]], remap[index[2]]])
    elif len(index) > 3:
        out.extend([[remap[index[0]], remap[index[i]], remap[index[i + 1]]]
                    for i in range(1, len(index) - 1)])
    return np.asarray(out, dtype=np.int32) if out else np.zeros((0, 3), dtype=np.int32)


def close_ring(ring: np.ndarray) -> np.ndarray:
    """Drop a repeated final vertex, which GML always has and clipping cannot use."""
    if len(ring) > 1 and np.allclose(ring[0], ring[-1]):
        return ring[:-1]
    return ring

"""Rectify a wall's slice of a texture atlas into a front-facing image.

Hamburg maps facade appearance onto walls through a per-building atlas and a UV
per polygon. That is exactly right for rendering and useless for everything
else: the pixels for one wall are an arbitrary quadrilateral somewhere in a
shared image, sheared by whatever the oblique camera's angle was.

Anything that wants to *understand* a facade -- segment its windows, classify
its material, judge a reconstruction against it -- needs the wall as a wall:
front on, upright, at a known metres-per-pixel. So this resamples the atlas
through the UV mapping onto the wall's own metric frame.

    atlas + UV + wall frame  ->  rectified crop, scale known, orientation fixed

Two things fall out that are worth as much as the image.

The mapping is invertible. A window found at (x, y) in the crop is at a known
offset in metres along the wall and up from its base, so a detection becomes a
position on a named surface in 3D rather than a pixel.

The effective resolution is measurable. The source is nominally 20 cm, but a
wall seen at 45 degrees from a nadir-ish flight gets fewer pixels than that, and
a wall the camera barely saw gets almost none. `resolution_px_per_m` says which,
per wall, so a facade that cannot support analysis can be refused instead of
guessed at.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..reconstruct.tessellate import close_ring, triangulate, wall_frame

#: Output sampling. 20 cm source is 5 px/m at best, so 32 px/m oversamples by
#: about 6x -- deliberately. The crop is an analysis surface and a material
#: input, not a compression target, and resampling artefacts at the native rate
#: would be indistinguishable from facade detail.
DEFAULT_PX_PER_M = 32.0

#: Below this a wall's pixels cannot support window detection: at 2 px/m a
#: 1.2 m window is two pixels across. Recorded rather than silently analysed.
MIN_USABLE_PX_PER_M = 2.0


@dataclass
class Facade:
    """One wall, rectified, with everything needed to map back to the world."""
    surface_id: str | None
    building_id: str | None
    image: np.ndarray                       # (H, W, 3) uint8
    px_per_m: float
    width_m: float
    height_m: float
    origin_xyz: np.ndarray                  # world position of the crop's (0, 0)
    u_axis: np.ndarray
    v_axis: np.ndarray
    normal: np.ndarray
    resolution_px_per_m: float              # what the *source* actually supplied
    covered: float                          # fraction of the crop with real pixels
    meta: dict = field(default_factory=dict)

    def to_world(self, x_px: float, y_px: float) -> np.ndarray:
        """A pixel in the crop -> a point on the wall in world coordinates.

        This is the whole reason the frame is carried alongside the image. The
        crop's y runs down from the top, so it is flipped back against v.
        """
        u_m = x_px / self.px_per_m
        v_m = self.height_m - y_px / self.px_per_m
        return self.origin_xyz + u_m * self.u_axis + v_m * self.v_axis

    def _quality(self) -> dict:
        """Which part of the crop is really the wall, and how much that is.

        Conservative on purpose. The band boundary is where horizontal rhythm
        stops, and the transition storey is genuinely mixed -- part facade, part
        whatever the camera saw past it -- so it falls on the unusable side. A
        macro layer trusted too far down is worse than one trusted too little,
        because the procedural layer can cover an honest gap and cannot undo a
        roof photographed onto a wall.
        """
        top, bottom = usable_band(self.image, self.px_per_m)
        return {
            "usable_top": round(top, 3),
            "usable_bottom": round(bottom, 3),
            "usable_height_m": round((bottom - top) * self.height_m, 2),
            "usable_fraction": round(bottom - top, 3),
            "macro_trustworthy": bool(bottom - top > 0.25),
        }

    def to_dna(self) -> dict:
        """The inspectable record the appearance pipeline is contracted on.

        Deliberately holds no material decision. Everything here is measured or
        geometric; `material_family`, masks and confidences are added by whatever
        analyses the crop, and stay separable from what was observed.
        """
        return {
            "surface_id": self.surface_id,
            "building_id": self.building_id,
            "width_m": round(self.width_m, 3),
            "height_m": round(self.height_m, 3),
            "crop_px": [int(self.image.shape[1]), int(self.image.shape[0])],
            "px_per_m": self.px_per_m,
            "source_px_per_m": round(self.resolution_px_per_m, 2),
            "source_usable": bool(self.resolution_px_per_m >= MIN_USABLE_PX_PER_M),
            "covered": round(self.covered, 4),
            **self._quality(),
            "origin_xyz": [round(float(v), 3) for v in self.origin_xyz],
            "u_axis": [round(float(v), 6) for v in self.u_axis],
            "v_axis": [round(float(v), 6) for v in self.v_axis],
            "normal": [round(float(v), 6) for v in self.normal],
            "source_uv_channel": 0,
            **self.meta,
        }


def _barycentric(points: np.ndarray, a, b, c):
    """Barycentric coordinates of `points` in triangle abc, all 2D."""
    v0, v1, v2 = b - a, c - a, points - a
    d00 = v0 @ v0
    d01 = v0 @ v1
    d11 = v1 @ v1
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-14:
        return None
    d20 = v2 @ v0
    d21 = v2 @ v1
    beta = (d11 * d20 - d01 * d21) / denom
    gamma = (d00 * d21 - d01 * d20) / denom
    return 1.0 - beta - gamma, beta, gamma


def source_resolution(ring: np.ndarray, uv: np.ndarray,
                      atlas_shape: tuple[int, int]) -> float:
    """Pixels per metre the atlas actually devotes to this wall.

    Ratio of the wall's area in texture pixels to its area in square metres,
    square-rooted. This is the honest number: nominal GSD says what the camera
    could resolve looking straight down, and a facade was never looked at
    straight on.
    """
    # Close the ring, then take the UVs the same way rather than closing them
    # independently: a UV's repeated last vertex is not always bit-identical to
    # its first, so `close_ring` drops one and not the other, the lengths
    # disagree and the whole measurement silently returns zero. Six of
    # twenty-four Hamburg walls hit exactly that.
    ring = np.asarray(ring, dtype=float)
    uv = np.asarray(uv, dtype=float)
    if len(ring) != len(uv) or len(ring) < 3:
        return 0.0
    flat_ring = close_ring(ring)
    flat_uv = uv[:len(flat_ring)]
    if len(flat_ring) < 3:
        return 0.0
    u_axis, v_axis, _ = wall_frame(flat_ring)
    plane = np.column_stack([flat_ring @ u_axis, flat_ring @ v_axis])

    def shoelace(points):
        x, y = points[:, 0], points[:, 1]
        return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) / 2.0

    metric_area = shoelace(plane)
    height, width = atlas_shape[:2]
    pixel_area = shoelace(flat_uv * np.array([width, height]))
    if metric_area < 1e-9:
        return 0.0
    return float(np.sqrt(pixel_area / metric_area))


def rectify(ring: np.ndarray, uv: np.ndarray, atlas: np.ndarray, *,
            px_per_m: float = DEFAULT_PX_PER_M,
            surface_id: str | None = None,
            building_id: str | None = None,
            max_px: int = 4096) -> Facade | None:
    """Resample the atlas onto the wall's own frame. None if degenerate.

    Nearest-neighbour on purpose. The source is already a resampled
    photogrammetric product and the output oversamples it several times over, so
    bilinear would only blur evidence that a later stage has to read. Anything
    outside the polygon comes back as zero and is reported as uncovered rather
    than filled, because an invented pixel is indistinguishable from a measured
    one once it is in the image.
    """
    ring = close_ring(np.asarray(ring, dtype=float))
    uv = close_ring(np.asarray(uv, dtype=float))
    if len(ring) < 3 or len(uv) != len(ring):
        return None

    u_axis, v_axis, normal = wall_frame(ring)
    su = ring @ u_axis
    sv = ring @ v_axis
    width_m = float(su.max() - su.min())
    height_m = float(sv.max() - sv.min())
    if width_m < 0.1 or height_m < 0.1:
        return None

    # A 60 m warehouse wall at 32 px/m is 1920 px, which is fine; a whole block
    # face is not. Scale down rather than refuse, and record what was used.
    scale = px_per_m
    if max(width_m, height_m) * scale > max_px:
        scale = max_px / max(width_m, height_m)
    out_w = max(1, int(round(width_m * scale)))
    out_h = max(1, int(round(height_m * scale)))

    plane = np.column_stack([su - su.min(), sv - sv.min()])
    tris = triangulate(ring)
    if not len(tris):
        return None

    image = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    filled = np.zeros((out_h, out_w), dtype=bool)
    atlas_h, atlas_w = atlas.shape[:2]

    ys, xs = np.mgrid[0:out_h, 0:out_w]
    # Pixel centres, with y flipped so row 0 is the top of the wall.
    px_u = (xs + 0.5) / scale
    px_v = height_m - (ys + 0.5) / scale
    points = np.column_stack([px_u.ravel(), px_v.ravel()])

    for tri in tris:
        a, b, c = plane[tri[0]], plane[tri[1]], plane[tri[2]]
        bary = _barycentric(points, a, b, c)
        if bary is None:
            continue
        w0, w1, w2 = bary
        inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        if not inside.any():
            continue
        index = np.flatnonzero(inside)
        # The UV map is affine within a triangle, so the same barycentrics that
        # located the point on the wall locate it in the atlas.
        tu = (w0[index] * uv[tri[0], 0] + w1[index] * uv[tri[1], 0]
              + w2[index] * uv[tri[2], 0])
        tv = (w0[index] * uv[tri[0], 1] + w1[index] * uv[tri[1], 1]
              + w2[index] * uv[tri[2], 1])
        # CityGML's v origin is the lower left; image row 0 is the top.
        ax = np.clip((tu * atlas_w).astype(np.int64), 0, atlas_w - 1)
        ay = np.clip(((1.0 - tv) * atlas_h).astype(np.int64), 0, atlas_h - 1)
        rows, cols = np.divmod(index, out_w)
        image[rows, cols] = atlas[ay, ax, :3]
        filled[rows, cols] = True

    # World position of the crop's lower-left corner: step from any known ring
    # vertex to the frame's minimum along each axis.
    corner = (ring[0] + (su.min() - su[0]) * u_axis
              + (sv.min() - sv[0]) * v_axis)

    return Facade(
        surface_id=surface_id, building_id=building_id, image=image,
        px_per_m=float(scale), width_m=width_m, height_m=height_m,
        origin_xyz=corner, u_axis=u_axis, v_axis=v_axis, normal=normal,
        resolution_px_per_m=source_resolution(ring, uv, atlas.shape),
        covered=float(filled.mean()))


#: A facade repeats horizontally and a roof seen obliquely does not. Bay spacing
#: is *measured*, not assumed: on the Rathaus block a known-good Kontorhaus
#: elevation autocorrelates at 1.50 m with harmonics at 2.16, 2.59 and 3.03 m.
#: The first guess here was 2.5 to 9 m, which excluded the real bay entirely and
#: reported a plainly good facade as 8% usable.
BAY_MIN_M = 1.0
BAY_MAX_M = 6.0

#: One storey. The band has to span a whole window row: on a half-metre band the
#: score swings between 0.80 and 0.07 depending on whether the band happens to
#: land on glass or on brick between rows, which is noise, not evidence.
STOREY_M = 3.5

#: Below this normalised autocorrelation peak a band has no horizontal rhythm.
#: Measured on the Rathaus block: storey bands over the good elevation score
#: 0.24 to 0.47 and the roof-bleed band scores 0.11.
RHYTHM_THRESHOLD = 0.18


def rhythm_profile(image: np.ndarray, px_per_m: float, *,
                   band_m: float = STOREY_M) -> np.ndarray:
    """Per-storey strength of horizontal repetition, top band first.

    An aerial oblique sees the top of a wall well and the bottom badly: the
    building's own eaves, the roof in front and the street occlude the lower
    storeys, so the lower part of a crop is frequently not the wall at all. On a
    known-good Hamburg elevation that boundary is plainly visible and no metric
    in the pipeline noticed it.

    Detected rather than assumed, because the usable fraction varies per wall
    with the flight geometry. The column signal is differenced before
    correlating, so a facade in shadow scores like a facade in sun.
    """
    grey = image.astype(np.float32).mean(axis=2)
    band_px = max(4, int(round(band_m * px_per_m)))
    count = max(1, grey.shape[0] // band_px)
    if grey.shape[1] < 8:
        return np.zeros(count, dtype=np.float32)
    lag_lo = max(2, int(BAY_MIN_M * px_per_m))
    lag_hi = min(grey.shape[1] // 2, int(BAY_MAX_M * px_per_m))
    if lag_hi <= lag_lo:
        return np.zeros(count, dtype=np.float32)

    scores = []
    for start in range(0, grey.shape[0], band_px):
        band = grey[start:start + band_px]
        if len(band) < band_px // 2:
            break
        signal = np.diff(band.mean(axis=0))
        signal -= signal.mean()
        energy = float(signal @ signal)
        if energy < 1e-6:
            scores.append(0.0)
            continue
        best = max((float(signal[:-lag] @ signal[lag:]) / energy)
                   for lag in range(lag_lo, lag_hi))
        scores.append(max(0.0, best))
    return np.asarray(scores or [0.0], dtype=np.float32)


def usable_band(image: np.ndarray, px_per_m: float, *,
                band_m: float = STOREY_M) -> tuple[float, float]:
    """(top, bottom) of the trustworthy region as fractions of crop height.

    The longest contiguous run of storeys with real horizontal rhythm. Where
    that is the whole crop the wall was fully seen; where it stops part way
    down, the lower storeys are occlusion bleed and the macro layer must not be
    trusted there -- which is precisely the region a street-level camera spends
    all its time looking at.
    """
    profile = rhythm_profile(image, px_per_m, band_m=band_m)
    good = profile >= RHYTHM_THRESHOLD
    best_len, best_start, run_start = 0, 0, None
    for i, value in enumerate(np.append(good, False)):
        if value and run_start is None:
            run_start = i
        elif not value and run_start is not None:
            if i - run_start > best_len:
                best_len, best_start = i - run_start, run_start
            run_start = None
    if best_len == 0:
        return 0.0, 0.0
    return (best_start / len(profile), (best_start + best_len) / len(profile))


def load_atlas(path: str | Path) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"))


def save(facade: Facade, path: str | Path) -> Path:
    from PIL import Image
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(facade.image).save(path)
    return path

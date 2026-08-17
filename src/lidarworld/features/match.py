"""Match a measured wall to a photographed material, in the units of the wall.

The measured facade knows what it is and not what it looks like up close; a
library texture knows what a wall looks like up close and nothing about this one.
The join is a small set of descriptors both can be reduced to, all of them in
real-world units so a 40 m frontage at 13 px/m and a 1 m texture tile at 1024 px
are comparable at all:

    colour          de-lit, so the survey flight's exposure is not being matched
    coursing        the horizontal repeat in METRES, from the same period finder
                    the bay lattice uses, run at masonry scale instead
    course height   the vertical repeat in metres -- a brick course is 70 mm and
                    an ashlar block 300, and that difference is most of what
                    distinguishes the two materials
    roughness       high-frequency energy, which separates smooth render from
                    rubble even where colour and period agree

Scale is the reason this can be done at all, and the reason Poly Haven is worth
preferring: it publishes each texture's real size, so its coursing measures in
metres rather than in pixels. Against a library that does not, every match is a
guess about magnification, and guessing that is how masonry ends up doll-sized.

Colour is matched but not copied. The chosen texture is recoloured toward the
building's own measured colour by the same luminance-only route `frequency`
uses, so what is borrowed is structure and what is kept is identity. A Helsinki
block does not become a Tuscan one because the nearest scan was warm.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frequency import LUMA, box_blur
from .openings import period

#: Masonry repeats to search for, metres. A brick course is 0.07 m and a large
#: ashlar block 0.45; below 0.03 is noise and above 0.7 is architecture, not
#: material.
COURSE_MIN_M = 0.03
COURSE_MAX_M = 0.70

#: How the four descriptors are weighted. Coursing dominates because it is the
#: one a person reads instantly -- a wall with the wrong brick size is wrong at a
#: glance, where a wall with a slightly wrong hue is not.
WEIGHTS = {"course_h": 3.0, "course_v": 3.0, "roughness": 1.5, "colour": 1.0}

#: A coursing measurement is only real if the source resolved it. Below this many
#: source pixels per course it is the strongest lag in noise, and on the Helsinki
#: mesh at 13.2 px/m that is exactly what came back: 0.083 m in both axes, which
#: is 1.1 pixels. Ranking on it picked an asbestos sheet for a plastered
#: Jugendstil block, because the dominant term in the score carried no
#: information at all.
MIN_COURSE_PX = 4.0


@dataclass
class Descriptor:
    """What a wall or a texture reduces to, all in real-world units."""
    course_h_m: float
    course_v_m: float
    course_h_strength: float
    course_v_strength: float
    roughness: float
    colour: np.ndarray             # (3,) de-lit, normalised to unit luminance
    luminance: float
    source: str = ""
    #: Whether the source resolved masonry at all. False means the coursing
    #: fields are noise and must not be matched on.
    resolves_coursing: bool = True
    px_per_course: float = 0.0

    def to_record(self) -> dict:
        return {
            "course_h_m": round(self.course_h_m, 4),
            "course_v_m": round(self.course_v_m, 4),
            "course_strength": [round(self.course_h_strength, 3),
                                round(self.course_v_strength, 3)],
            "roughness": round(self.roughness, 4),
            "colour": [round(float(v), 3) for v in self.colour],
            "luminance": round(self.luminance, 3),
            "source": self.source,
            "resolves_coursing": self.resolves_coursing,
            "px_per_course": round(self.px_per_course, 2),
        }


def describe(image: np.ndarray, *, metres_across: float, source: str = "",
             delight: bool = True, source_px_per_m: float = 0.0) -> Descriptor:
    """Reduce an image of a wall to its descriptors, given how wide it really is.

    `metres_across` is what makes this comparable across sources, and it is not
    optional: without it the coursing comes out in pixels and a 1024 px brick
    scan matches a 1024 px ashlar scan exactly.
    """
    array = np.asarray(image, dtype=np.float64)
    if array.max() > 1.5:
        array = array / 255.0
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    rows, cols = array.shape[:2]
    px_per_m = cols / max(metres_across, 1e-6)

    if delight:
        from .frequency import delight as delight_fn
        array, _, _ = delight_fn(array, px_per_m=px_per_m)

    luma = array @ LUMA
    course_h, strength_h = period(luma.mean(axis=0), px_per_m,
                                  COURSE_MIN_M, COURSE_MAX_M)
    # The vertical axis carries the same real scale only if the image is square
    # in the world; textures are, and a facade crop is by construction.
    px_per_m_v = rows / max(metres_across * rows / max(cols, 1), 1e-6)
    course_v, strength_v = period(luma.mean(axis=1), px_per_m_v,
                                  COURSE_MIN_M, COURSE_MAX_M)

    low = box_blur(luma[:, :, None], 4, wrap=False)[:, :, 0]
    roughness = float(np.sqrt(np.mean((luma - low) ** 2)))

    mean = array.reshape(-1, 3).mean(axis=0)
    brightness = float(mean @ LUMA)
    # Against the SOURCE's resolution, not the crop's. A facade upsampled from
    # 13 px/m to 48 has plenty of pixels per course and no information in them.
    native = source_px_per_m if source_px_per_m else px_per_m
    per_course = min(course_h, course_v) * native if min(course_h, course_v) > 0 else 0.0
    return Descriptor(
        course_h_m=course_h, course_v_m=course_v,
        course_h_strength=strength_h, course_v_strength=strength_v,
        roughness=roughness,
        colour=mean / max(brightness, 1e-6), luminance=brightness,
        source=source, px_per_course=per_course,
        resolves_coursing=per_course >= MIN_COURSE_PX)


def distance(wall: Descriptor, material: Descriptor) -> float:
    """How unlike the material is, in the wall's own terms. Lower is nearer.

    Coursing is compared as a LOG ratio, not a difference. A 20 mm error on a
    70 mm brick course is a different material; the same 20 mm on a 450 mm ashlar
    block is the same material cut slightly differently, and a linear metric
    cannot tell those apart.
    """
    def ratio(a: float, b: float) -> float:
        if a <= 1e-6 or b <= 1e-6:
            return 1.0                       # one side has no coursing to match
        return abs(np.log(a / b))

    colour = float(np.linalg.norm(wall.colour - material.colour))
    score = (WEIGHTS["roughness"] * abs(wall.roughness - material.roughness) * 8.0
             + WEIGHTS["colour"] * colour)
    # Drop the coursing terms entirely when the wall never resolved a course.
    # Weighting a noise measurement at 3.0 makes it the whole answer, and the
    # answer is then arbitrary rather than merely uncertain.
    if wall.resolves_coursing:
        score += (WEIGHTS["course_h"] * ratio(wall.course_h_m, material.course_h_m)
                  + WEIGHTS["course_v"] * ratio(wall.course_v_m, material.course_v_m))
    return score


def rank(wall: Descriptor, library: list[tuple[str, Descriptor]]
         ) -> list[tuple[str, float, Descriptor]]:
    """Every candidate, nearest first."""
    scored = [(key, distance(wall, d), d) for key, d in library]
    return sorted(scored, key=lambda row: row[1])


def recolour(texture: np.ndarray, wall: Descriptor) -> np.ndarray:
    """Move a library texture onto the building's own measured colour.

    Luminance-only, and per-channel scaling of the *mean* rather than of each
    pixel, so the texture keeps its own variation -- the darker header brick, the
    lighter mortar -- and only its overall cast moves. Dividing each channel by
    its own low-pass is the version that repaints a grey wall in the source's
    hue, which is the one thing the measured photograph exists to prevent.
    """
    array = np.asarray(texture, dtype=np.float64)
    if array.max() > 1.5:
        array = array / 255.0
    mean = array.reshape(-1, 3).mean(axis=0)
    brightness = float(mean @ LUMA)
    target = wall.colour * wall.luminance
    scale = target / np.maximum(mean, 1e-4)
    out = array * scale[None, None, :]
    # Preserve the texture's own contrast: scaling can clip the highlights, and a
    # clipped brick reads as a flat one.
    peak = float(out.max())
    if peak > 1.0:
        out = out / peak * min(1.0, brightness * 1.9 + 0.25)
    return np.clip(out, 0.0, 1.0)

"""Frequency-separated facade appearance: keep the photograph, add what it lost.

A photogrammetric facade carries the building's identity and none of its
surface. Helsinki's mesh gives 9.3 cm per texel and a 77 cm median triangle, so
at arm's length there is nothing there: no brick, no mortar, no plaster grain, no
reveal shadow. Stretching the photograph over that gap is what makes a reality
mesh look like melted wax.

The fix is not a sharper photograph. It is to treat the capture as the *low
frequency* channel -- colour, staining, window layout, architectural identity,
all of it real and none of it inventable -- and to synthesise only the high
frequencies it could never resolve.

    macro (measured)   what the building is
    micro (generated)  what any surface of that material does at 1 cm

Two things make this work rather than merely darken the image.

**The micro albedo is neutralised before use.** A tileable brick texture averages
around 0.4, so multiplying by it darkens the facade by 60% and throws away the
measured colour. Dividing the micro by its own low-pass leaves a field whose
average is 1.0 and whose variation is pure detail:

    D = C_micro / (lowpass(C_micro) + eps)
    C_final = C_macro * lerp(1, D, w)

At w = 0 the result is exactly the photograph. Nothing about the measured colour
moves; only detail is injected.

**The micro is placed in metres, not in UV.** A brick is 240 mm on a shed and on
a warehouse. Normalising the micro to the patch would make one course span
whatever the wall happens to be, which is the single most common way this
technique looks fake.

Everything generated here is `generated`, never `observed`. The composite carries
a weight map so a later stage can still say which frequencies were measured.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Real repeat of the micro material, in metres. A tile of `procedural.SIZE`
#: pixels represents this much wall, which is what turns a UV into a metric
#: coordinate. Values are the physical thing: a brick course with its mortar is
#: about 240 x 71 mm, so a 256 px tile covering 1.2 m puts a brick at ~51 px.
DEFAULT_TILE_M = 1.2

#: How much detail to inject. 1.0 uses the neutralised micro at full strength;
#: the macro's own colour is preserved at any value because the field averages
#: one. Above about 1.4 the variation starts reading as noise rather than
#: material.
DEFAULT_DETAIL = 1.0

#: Low-pass radius used to neutralise the micro, as a fraction of the tile. Has
#: to be wide enough to average over several bricks -- a narrow blur leaves the
#: course rhythm in the "low frequency" estimate and then divides it out, which
#: removes the very structure the layer exists to add.
NEUTRALISE_FRACTION = 0.25


def _window_sums(image: np.ndarray, radius: int, wrap: bool) -> np.ndarray:
    """Separable running sum over a (2r+1)^2 window. Outside is wrapped or zero."""
    out = np.asarray(image, dtype=np.float64)
    window = 2 * radius + 1
    for axis in (0, 1):
        if wrap:
            edges = (np.take(out, range(-radius, 0), axis=axis),
                     np.take(out, range(0, radius), axis=axis))
        else:
            shape = list(out.shape)
            shape[axis] = radius
            edges = (np.zeros(shape), np.zeros(shape))
        padded = np.concatenate([edges[0], out, edges[1]], axis=axis)
        # Leading zero so a window sum is one subtraction: sum(i .. i+w-1) is
        # cumulative[i+w] - cumulative[i]. Without it the top index runs one
        # past the end of the cumulative array.
        shape = list(padded.shape)
        shape[axis] = 1
        cumulative = np.cumsum(
            np.concatenate([np.zeros(shape), padded], axis=axis), axis=axis)
        n = out.shape[axis]
        lo = np.take(cumulative, range(0, n), axis=axis)
        hi = np.take(cumulative, range(window, window + n), axis=axis)
        out = hi - lo
    return out


def box_blur(image: np.ndarray, radius: int, *, wrap: bool = True) -> np.ndarray:
    """Separable box blur. Wraps by default, so a tileable input stays tileable.

    Reflecting or clamping at the edge would leave a seam exactly where the micro
    tile repeats, which is the one place it must not.

    `wrap=False` is for the other case, and it is not cosmetic. A rectified
    facade crop does not repeat: wrapping makes the reference at the sunlit end
    of a frontage include the shaded end, which is precisely the gradient the
    de-lighting exists to measure. Outside the crop is then treated as absent
    rather than as zero -- the window shrinks at the border -- so an edge column
    is compared against its own neighbourhood and not against a black margin.
    """
    if radius < 1:
        return np.asarray(image, dtype=np.float64)
    total = _window_sums(image, radius, wrap)
    if wrap:
        return total / (2 * radius + 1) ** 2
    count = _window_sums(np.ones_like(total), radius, False)
    return total / count


#: Rec. 709 luminance. The micro layer contributes brightness detail only.
LUMA = np.array([0.2126, 0.7152, 0.0722])

#: How wide a low-pass estimates the baked illumination, in metres of wall. Wide
#: enough to be lighting rather than architecture: a 4 m blur averages across a
#: whole bay, so a window's own darkness is not mistaken for shadow, while the
#: sunlit-to-shaded gradient across a frontage survives.
DELIGHT_SPAN_M = 4.0

#: Below this the de-light divisor is untrustworthy -- a deep shadow divided by
#: near-zero explodes -- so the result blends back toward the source and the
#: confidence map says so.
DELIGHT_FLOOR = 0.06


def delight(macro: np.ndarray, *, px_per_m: float,
            span_m: float = DELIGHT_SPAN_M,
            strength: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove the survey flight's lighting from a facade. Returns (albedo, light, confidence).

    A photogrammetric texture is not albedo. It contains the sun, the sky, every
    self-shadow and the exposure the aircraft flew at:

        C_photo ~ A_intrinsic * L_baked

    Feed it to a renderer as base colour and light it again and the recesses --
    exactly the features that carry a facade's depth -- get darkened twice. The
    composite in this module was compensating for that with a raised ambient
    term, which is a defect being papered over rather than a surface being lit.

    This is the same operation as `neutralise`, pointed at the macro instead of
    the micro: divide by a wide low-pass of luminance. It is an approximation and
    not an inverse-rendering solution, so the confidence map records where the
    divisor was too dark to trust and the caller can blend back toward the
    original there. Nothing here moves a window, alters geometry or invents
    detail -- it only flattens illumination.
    """
    macro = np.clip(np.asarray(macro, dtype=np.float64), 0.0, 1.0)
    if macro.ndim == 2:
        macro = np.repeat(macro[:, :, None], 3, axis=2)
    radius = max(1, int(round(span_m * px_per_m / 2.0)))
    luma = macro[:, :, :3] @ LUMA
    # No wrap: a facade crop is not a tile, and wrapping averages the sunlit end
    # of a frontage into the shaded end -- the exact gradient being removed.
    light = box_blur(luma[:, :, None], radius, wrap=False)[:, :, 0]

    confidence = np.clip((light - DELIGHT_FLOOR) / max(DELIGHT_FLOOR, 1e-6), 0.0, 1.0)
    safe = np.maximum(light, DELIGHT_FLOOR)
    # Preserve mean brightness exactly. Scaling by the mean of the divisor is
    # close but not equal -- the division brightens shadow and darkens sun
    # asymmetrically, and on a real facade that came out 24% darker overall,
    # which then reads as the de-light "not working" rather than as a missing
    # normalisation.
    flattened = macro / safe[:, :, None]
    blended = macro + (flattened - macro) * (strength * confidence)[:, :, None]
    before, after = float(macro.mean()), float(blended.mean())
    if after > 1e-9:
        blended = blended * (before / after)
    return np.clip(blended, 0.0, 1.0), light, confidence


def neutralise(micro: np.ndarray, *, fraction: float = NEUTRALISE_FRACTION,
               eps: float = 1e-3) -> np.ndarray:
    """Micro albedo -> a single-channel detail field averaging 1.0.

    Two things have to be true for the composite to be non-destructive, and the
    first attempt at this got the second wrong.

    **Mean one.** Without it the naive `macro * micro` darkens a facade by
    whatever the micro happens to average, and the measured colour -- the entire
    reason for using the photograph -- is gone.

    **Luminance only.** The field must be scalar, not per-channel. Dividing each
    channel by its own low-pass looks like it preserves hue, and the raw ratios
    do: measured on the brick generator they average [0.997, 0.994, 0.992].
    But the tails have to be clipped, and brick's mortar is bright relative to
    its dark blue base, so blue's ratio reaches 3.2 and loses far more to the
    clip than red does. Post-clip means come out [0.997, 0.941, 0.884], which
    turns a flat grey macro into red brick -- a 0.09 shift in blue. The facade's
    identity, which is the one thing the photograph is for, gets repainted.

    So the micro is desaturated first and the same scalar multiplies every
    channel. Mortar then reads as lighter and brick as darker, which is
    physically what they are, and the building keeps its own colour.
    """
    micro = np.asarray(micro, dtype=np.float64)
    if micro.ndim == 2:
        luma = micro
    else:
        luma = micro[:, :, :3] @ LUMA
    radius = max(1, int(round(min(luma.shape[:2]) * fraction)))
    low = box_blur(luma[:, :, None], radius)[:, :, 0]
    # Range chosen so the product cannot clip a bright macro. A pale stucco
    # facade at 0.62 red times 2.2 exceeds one and clips, while its 0.42 blue
    # does not -- which shifts the ratio between them even though the field
    # itself is colour-neutral. Measured: R/B went 1.476 to 1.344 at the wide
    # range, and holds at 1.47 here.
    detail = np.clip(luma / (low + eps), 0.55, 1.55)
    # Restore the mean the clip just moved, so the guard cannot darken or
    # brighten the facade it is protecting.
    return detail / max(float(detail.mean()), 1e-6)


def tile_to(field: np.ndarray, height: int, width: int, *,
            px_per_m: float, tile_m: float,
            offset_px: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """Repeat `field` across an (height, width) patch at a real-world scale.

    `px_per_m` is the output's sampling and `tile_m` the material's physical
    repeat, so the number of output pixels per tile is `tile_m * px_per_m` --
    independent of the patch's size. That is the whole difference between a
    metric micro layer and a normalised one.
    """
    field = np.asarray(field, dtype=np.float64)
    if field.ndim == 2:
        field = field[:, :, None]
    tile_px = max(2.0, tile_m * px_per_m)
    scale = field.shape[0] / tile_px
    rows = ((np.arange(height) + offset_px[0]) * scale).astype(np.int64) % field.shape[0]
    cols = ((np.arange(width) + offset_px[1]) * scale).astype(np.int64) % field.shape[1]
    return field[np.ix_(rows, cols)]


@dataclass
class Composite:
    """A facade patch with its measured and generated frequencies kept apart."""
    macro: np.ndarray               # (H, W, 3) float, the photograph
    albedo: np.ndarray              # (H, W, 3) float, macro x detail
    normal: np.ndarray              # (H, W, 3) float, unit vectors in wall frame
    roughness: np.ndarray           # (H, W) float
    px_per_m: float
    tile_m: float
    detail: float
    material: str

    @property
    def shape(self) -> tuple[int, int]:
        return self.albedo.shape[0], self.albedo.shape[1]

    def to_record(self) -> dict:
        return {
            "material_family": self.material,
            "micro_tile_m": self.tile_m,
            "detail_strength": self.detail,
            "px_per_m": self.px_per_m,
            "crop_px": [int(self.shape[1]), int(self.shape[0])],
            "macro_epistemic": "derived",     # somebody's photogrammetry
            "micro_epistemic": "generated",   # synthesised here
            "macro_mean": [round(float(v), 4) for v in self.macro.reshape(-1, 3).mean(0)],
            "albedo_mean": [round(float(v), 4) for v in self.albedo.reshape(-1, 3).mean(0)],
        }


def compose(macro: np.ndarray, *, material: str = "brick",
            px_per_m: float = 32.0, tile_m: float = DEFAULT_TILE_M,
            detail: float = DEFAULT_DETAIL, seed: int = 7,
            normal_strength: float = 2.0, **params) -> Composite:
    """Add synthesised high frequencies to a measured facade crop.

    `macro` is float 0..1 RGB. The micro material comes from the repo's own
    procedural generators, so no image library is involved and the scale is
    whatever `tile_m` says it is.
    """
    from ..themes import procedural

    generator = procedural.GENERATORS.get(material)
    if generator is None:
        raise KeyError(f"unknown micro material {material!r}; "
                       f"have {sorted(procedural.GENERATORS)}")

    macro = np.clip(np.asarray(macro, dtype=np.float64), 0.0, 1.0)
    if macro.ndim == 2:
        macro = np.repeat(macro[:, :, None], 3, axis=2)
    height, width = macro.shape[:2]

    micro_albedo, micro_height, micro_rough = generator(seed=seed, **params)
    field = neutralise(micro_albedo)
    normal_tile = procedural.normal_map(micro_height, strength=normal_strength) * 2.0 - 1.0

    tiled_detail = tile_to(field[:, :, None], height, width,
                           px_per_m=px_per_m, tile_m=tile_m)
    tiled_normal = tile_to(normal_tile, height, width, px_per_m=px_per_m, tile_m=tile_m)
    tiled_rough = tile_to(micro_rough, height, width,
                          px_per_m=px_per_m, tile_m=tile_m)[:, :, 0]

    # lerp(1, D, w): at w = 0 this is exactly the photograph, so the measured
    # colour is recoverable and the injection is provably non-destructive. The
    # field is scalar, so it scales all three channels equally and cannot shift
    # hue however strong the detail is.
    weighted = 1.0 + (tiled_detail - 1.0) * detail
    albedo = np.clip(macro * weighted, 0.0, 1.0)

    norm = np.linalg.norm(tiled_normal, axis=2, keepdims=True)
    normal = tiled_normal / np.maximum(norm, 1e-9)

    return Composite(macro=macro, albedo=albedo, normal=normal,
                     roughness=np.clip(tiled_rough, 0.04, 1.0),
                     px_per_m=px_per_m, tile_m=tile_m, detail=detail,
                     material=material)


#: Light direction in the wall's own frame: up and to the left, slightly toward
#: the viewer. Chosen so relief reads without the shading fighting whatever the
#: photograph already contains.
LIGHT = np.array([-0.45, 0.62, 0.65])


def shade(composite: Composite, *, light=LIGHT, ambient: float = 0.72,
          specular: float = 0.16, use_micro_normal: bool = True) -> np.ndarray:
    """Light a composite in its own wall frame. Returns uint8 RGB.

    Deliberately restrained. The macro already contains the sun, the sky and
    every self-shadow the survey flight captured, so a full relight would
    double-darken exactly the recesses that carry a facade's depth. What the
    micro normal is allowed to do is modulate around that: a shallow diffuse
    term plus a small specular, so mortar reads as recessed and brick as
    slightly proud, without inventing a second sun.
    """
    light = np.asarray(light, dtype=np.float64)
    light = light / np.linalg.norm(light)
    if use_micro_normal:
        lambert = np.clip((composite.normal * light).sum(axis=2), 0.0, 1.0)
    else:
        lambert = np.full(composite.shape, float(np.clip(light[2], 0.0, 1.0)))
    diffuse = ambient + (1.0 - ambient) * lambert

    view = np.array([0.0, 0.0, 1.0])
    half = light + view
    half /= np.linalg.norm(half)
    if use_micro_normal:
        gloss = np.clip((composite.normal * half).sum(axis=2), 0.0, 1.0)
    else:
        gloss = np.full(composite.shape, float(half[2]))
    sharpness = 2.0 + 60.0 * (1.0 - composite.roughness) ** 2
    highlight = specular * (1.0 - composite.roughness) * gloss ** sharpness

    lit = composite.albedo * diffuse[:, :, None] + highlight[:, :, None]
    return (np.clip(lit, 0.0, 1.0) * 255).astype(np.uint8)


def magnify(image: np.ndarray, factor: float) -> np.ndarray:
    """Nearest-neighbour magnify, to show a source at a viewing distance.

    Nearest rather than smooth on purpose: bilinear would blur the macro's
    limits away and flatter it. What a person sees standing at 1 m from a wall
    rendered off a 10.8 texel/m source is blocks, and the comparison is only
    honest if the blocks are visible.
    """
    if factor == 1.0:
        return image
    height, width = image.shape[:2]
    rows = np.clip((np.arange(int(height * factor)) / factor).astype(int), 0, height - 1)
    cols = np.clip((np.arange(int(width * factor)) / factor).astype(int), 0, width - 1)
    return image[np.ix_(rows, cols)]

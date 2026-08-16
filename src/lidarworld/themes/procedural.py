"""Procedural material backend -- the zero-dependency default.

Every texture is synthesised from code, so the repo ships no image assets, there
is no licence to audit, and a new era is a parameter block rather than a
gigabyte download. Higher-fidelity backends (CC0 libraries, photogrammetry,
authored packs) override these by material id; see docs/THEMES.md.

Everything generated here is tileable: the noise lattice wraps, so a texture can
repeat across a facade without a visible seam.
"""
from __future__ import annotations

import numpy as np

SIZE = 256


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed & 0x7FFFFFFF)


def value_noise(size: int, period: int, seed: int) -> np.ndarray:
    """Tileable bilinear value noise with `period` lattice cells per edge."""
    rng = _rng(seed)
    lattice = rng.random((period, period))
    coords = np.linspace(0, period, size, endpoint=False)
    i0 = np.floor(coords).astype(int) % period
    i1 = (i0 + 1) % period
    frac = coords - np.floor(coords)
    smooth = frac * frac * (3 - 2 * frac)
    a = lattice[np.ix_(i0, i0)] * (1 - smooth)[:, None] + lattice[np.ix_(i1, i0)] * smooth[:, None]
    b = lattice[np.ix_(i0, i1)] * (1 - smooth)[:, None] + lattice[np.ix_(i1, i1)] * smooth[:, None]
    return a * (1 - smooth)[None, :] + b * smooth[None, :]


def fbm(size: int, seed: int, octaves: int = 5, base_period: int = 4) -> np.ndarray:
    total = np.zeros((size, size))
    amplitude = 1.0
    norm = 0.0
    for o in range(octaves):
        period = base_period * (2 ** o)
        if period > size:
            break
        total += amplitude * value_noise(size, period, seed + o * 7919)
        norm += amplitude
        amplitude *= 0.5
    return total / max(norm, 1e-6)


def _tint(height: np.ndarray, color, variation: np.ndarray, strength: float = 0.35):
    base = np.asarray(color, dtype=np.float64).reshape(1, 1, 3)
    shade = (1.0 - strength) + strength * (0.5 + variation - variation.mean())
    rgb = base * shade[:, :, None] * (0.85 + 0.3 * height)[:, :, None]
    return np.clip(rgb, 0, 1)


def _grid_mask(size: int, cols: int, rows: int, mortar: float, offset_rows: bool = True):
    """Running-bond course pattern. Returns (mortar_mask, brick_id)."""
    y = np.arange(size)[:, None] / size * rows
    x = np.arange(size)[None, :] / size * cols
    row = np.floor(y)
    if offset_rows:
        x = x + 0.5 * (row % 2)
    col = np.floor(x)
    fx = x - col
    fy = y - row
    m = mortar
    mortar_mask = (fx < m) | (fx > 1 - m) | (fy < m) | (fy > 1 - m)
    brick_id = (row * 131 + col * 17).astype(np.int64)
    return mortar_mask, brick_id


# --- generators ------------------------------------------------------------
# Each returns (albedo float[0..1] HxWx3, height float[0..1] HxW, roughness HxW)

def plaster(seed=0, color=(0.80, 0.77, 0.70), wear=0.3, **_):
    n = fbm(SIZE, seed, 6, 3)
    grime = np.clip(fbm(SIZE, seed + 11, 4, 2) - 0.45, 0, 1) * wear
    albedo = _tint(n * 0.4, color, n, 0.18) * (1 - 0.45 * grime)[:, :, None]
    return np.clip(albedo, 0, 1), n * 0.25, np.full((SIZE, SIZE), 0.9) - 0.1 * n


def brick(seed=0, color=(0.55, 0.24, 0.18), mortar_color=(0.78, 0.76, 0.72),
          cols=8, rows=16, wear=0.35, **_):
    mortar_mask, brick_id = _grid_mask(SIZE, cols, rows, 0.045)
    variation = _rng(seed).random(int(brick_id.max()) + 2)[brick_id % (int(brick_id.max()) + 1)]
    grain = fbm(SIZE, seed + 3, 5, 8)
    body = _tint(grain * 0.5, color, variation, 0.42)
    mortar = _tint(grain * 0.3, mortar_color, grain, 0.15)
    albedo = np.where(mortar_mask[:, :, None], mortar, body)
    weather = np.clip(fbm(SIZE, seed + 21, 3, 2) - 0.5, 0, 1) * wear * 2
    albedo = np.clip(albedo * (1 - 0.4 * weather)[:, :, None], 0, 1)
    height = np.where(mortar_mask, 0.25, 0.75) - 0.15 * grain
    return albedo, height, np.where(mortar_mask, 0.95, 0.78)


def stone_block(seed=0, color=(0.62, 0.60, 0.55), cols=5, rows=8, **_):
    mortar_mask, block_id = _grid_mask(SIZE, cols, rows, 0.03)
    variation = _rng(seed).random(int(block_id.max()) + 2)[block_id % (int(block_id.max()) + 1)]
    grain = fbm(SIZE, seed + 5, 6, 6)
    albedo = _tint(grain * 0.6, color, variation, 0.3)
    albedo = np.where(mortar_mask[:, :, None], albedo * 0.72, albedo)
    height = np.where(mortar_mask, 0.2, 0.8) - 0.2 * grain
    return np.clip(albedo, 0, 1), height, np.full((SIZE, SIZE), 0.88)


def concrete(seed=0, color=(0.63, 0.63, 0.62), wear=0.4, **_):
    grain = fbm(SIZE, seed, 6, 4)
    pits = (value_noise(SIZE, 64, seed + 2) > 0.86).astype(float)
    streak = np.clip(fbm(SIZE, seed + 9, 3, 2) - 0.5, 0, 1) * wear
    albedo = _tint(grain * 0.4, color, grain, 0.14)
    albedo = np.clip(albedo * (1 - 0.3 * streak - 0.25 * pits)[:, :, None], 0, 1)
    return albedo, grain * 0.3 - 0.3 * pits, np.full((SIZE, SIZE), 0.92)


def asphalt(seed=0, color=(0.19, 0.19, 0.20), **_):
    grain = fbm(SIZE, seed, 6, 16)
    aggregate = (value_noise(SIZE, 96, seed + 4) > 0.72).astype(float)
    albedo = _tint(grain * 0.5, color, grain, 0.3)
    albedo = np.clip(albedo + 0.10 * aggregate[:, :, None], 0, 1)
    return albedo, grain * 0.2, np.full((SIZE, SIZE), 0.96)


def cobble(seed=0, color=(0.42, 0.40, 0.38), cols=10, rows=10, **_):
    rng = _rng(seed)
    y, x = np.mgrid[0:SIZE, 0:SIZE] / SIZE
    jitter_u = value_noise(SIZE, cols, seed + 1) * 0.35
    jitter_v = value_noise(SIZE, rows, seed + 2) * 0.35
    u = (x * cols + jitter_u) % cols
    v = (y * rows + jitter_v) % rows
    fu, fv = u - np.floor(u), v - np.floor(v)
    dome = 1 - np.clip(np.hypot(fu - 0.5, fv - 0.5) * 2.1, 0, 1)
    stone_id = (np.floor(u) * 37 + np.floor(v) * 91).astype(np.int64)
    variation = rng.random(int(stone_id.max()) + 2)[stone_id % (int(stone_id.max()) + 1)]
    albedo = _tint(dome * 0.6, color, variation, 0.4)
    joint = dome < 0.08
    albedo = np.where(joint[:, :, None], albedo * 0.45, albedo)
    return np.clip(albedo, 0, 1), dome, np.where(joint, 0.98, 0.82)


def roof_tile(seed=0, color=(0.45, 0.22, 0.16), rows=22, **_):
    y = np.arange(SIZE)[:, None] / SIZE * rows
    x = np.arange(SIZE)[None, :] / SIZE * rows * 0.75
    fy = y - np.floor(y)
    ridge = np.abs(np.sin(x * np.pi)) ** 0.7
    course = np.clip(1.4 * (1 - fy), 0, 1)
    grain = fbm(SIZE, seed, 5, 8)
    shade = 0.55 + 0.45 * ridge * course
    albedo = _tint(grain * 0.4, color, grain, 0.22) * shade[:, :, None]
    return np.clip(albedo, 0, 1), ridge * course, np.full((SIZE, SIZE), 0.8)


def metal_panel(seed=0, color=(0.55, 0.57, 0.60), cols=4, rows=4, rust=0.0, **_):
    mortar_mask, panel_id = _grid_mask(SIZE, cols, rows, 0.02, offset_rows=False)
    brushed = fbm(SIZE, seed, 4, 64)
    variation = _rng(seed).random(int(panel_id.max()) + 2)[panel_id % (int(panel_id.max()) + 1)]
    albedo = _tint(brushed * 0.3, color, variation, 0.12)
    if rust:
        blotch = np.clip(fbm(SIZE, seed + 13, 5, 3) - 0.45, 0, 1) * rust * 3
        albedo = albedo * (1 - blotch)[:, :, None] + np.array([0.42, 0.18, 0.07]) * blotch[:, :, None]
    albedo = np.where(mortar_mask[:, :, None], albedo * 0.7, albedo)
    return np.clip(albedo, 0, 1), np.where(mortar_mask, 0.3, 0.7), np.full((SIZE, SIZE), 0.35 + 0.4 * rust)


def glass(seed=0, color=(0.32, 0.44, 0.52), **_):
    sheen = fbm(SIZE, seed, 3, 2)
    y = np.linspace(0, 1, SIZE)[:, None]
    albedo = _tint(sheen * 0.2, color, sheen, 0.1) * (0.7 + 0.5 * y)
    return np.clip(albedo, 0, 1), np.zeros((SIZE, SIZE)), np.full((SIZE, SIZE), 0.08)


def wood_plank(seed=0, color=(0.45, 0.30, 0.17), rows=8, **_):
    y = np.arange(SIZE)[:, None] / SIZE * rows
    plank = np.floor(y).astype(np.int64)
    variation = _rng(seed).random(rows + 1)[plank % rows]
    rings = np.sin((np.arange(SIZE)[None, :] / SIZE * 18 + fbm(SIZE, seed, 4, 4) * 6) * np.pi)
    grain = 0.5 + 0.5 * rings
    albedo = _tint(grain * 0.5, color, variation, 0.3)
    seam = (y - plank) < 0.04
    albedo = np.where(seam[:, :, None], albedo * 0.5, albedo)
    return np.clip(albedo, 0, 1), grain * 0.3, np.full((SIZE, SIZE), 0.7)


def foliage(seed=0, color=(0.20, 0.38, 0.16), **_):
    leaves = fbm(SIZE, seed, 6, 12)
    speckle = value_noise(SIZE, 128, seed + 6)
    albedo = _tint(leaves * 0.7, color, speckle, 0.5)
    return np.clip(albedo, 0, 1), leaves * 0.4, np.full((SIZE, SIZE), 0.85)


def grass(seed=0, color=(0.28, 0.42, 0.20), **_):
    blades = fbm(SIZE, seed, 6, 24)
    patch = fbm(SIZE, seed + 8, 3, 3)
    albedo = _tint(blades * 0.6, color, patch, 0.35)
    return np.clip(albedo, 0, 1), blades * 0.2, np.full((SIZE, SIZE), 0.93)


def gravel(seed=0, color=(0.48, 0.45, 0.41), **_):
    stones = value_noise(SIZE, 72, seed)
    fine = fbm(SIZE, seed + 3, 5, 24)
    albedo = _tint(fine * 0.5, color, stones, 0.45)
    return np.clip(albedo, 0, 1), stones * 0.5, np.full((SIZE, SIZE), 0.95)


def water(seed=0, color=(0.10, 0.24, 0.34), **_):
    ripple = fbm(SIZE, seed, 5, 6)
    albedo = _tint(ripple * 0.3, color, ripple, 0.15)
    return np.clip(albedo, 0, 1), ripple * 0.1, np.full((SIZE, SIZE), 0.12)


def neon_panel(seed=0, color=(0.06, 0.07, 0.11), accent=(0.10, 0.95, 0.85),
               cols=6, rows=10, **_):
    mortar_mask, panel_id = _grid_mask(SIZE, cols, rows, 0.03, offset_rows=False)
    lit = _rng(seed).random(int(panel_id.max()) + 2)[panel_id % (int(panel_id.max()) + 1)] > 0.72
    grain = fbm(SIZE, seed + 2, 4, 8)
    albedo = _tint(grain * 0.3, color, grain, 0.2)
    accent_arr = np.asarray(accent).reshape(1, 1, 3)
    albedo = np.where((mortar_mask | lit)[:, :, None], albedo * 0.4 + accent_arr * 0.85, albedo)
    return np.clip(albedo, 0, 1), np.where(mortar_mask, 0.3, 0.6), np.full((SIZE, SIZE), 0.3)


def thatch(seed=0, color=(0.58, 0.45, 0.22), **_):
    straw = fbm(SIZE, seed, 6, 40)
    course = 0.5 + 0.5 * np.sin(np.arange(SIZE)[:, None] / SIZE * 14 * np.pi)
    albedo = _tint(straw * 0.6, color, straw, 0.4) * (0.7 + 0.4 * course)
    return np.clip(albedo, 0, 1), straw * 0.5, np.full((SIZE, SIZE), 0.95)


GENERATORS = {
    "plaster": plaster, "brick": brick, "stone_block": stone_block, "concrete": concrete,
    "asphalt": asphalt, "cobble": cobble, "roof_tile": roof_tile, "metal_panel": metal_panel,
    "glass": glass, "wood_plank": wood_plank, "foliage": foliage, "grass": grass,
    "gravel": gravel, "water": water, "neon_panel": neon_panel, "thatch": thatch,
}


def normal_map(height: np.ndarray, strength: float = 2.0) -> np.ndarray:
    """Tangent-space normal map from a height field (wrapping, so it tiles)."""
    dx = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) * strength
    dy = (np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)) * strength
    nz = np.ones_like(height)
    length = np.sqrt(dx * dx + dy * dy + nz * nz)
    return np.stack([(-dx / length + 1) / 2, (-dy / length + 1) / 2, (nz / length + 1) / 2], axis=-1)


def bake(spec) -> dict[str, np.ndarray]:
    """Render a MaterialSpec into uint8 PBR channel images."""
    generator = GENERATORS.get(spec.generator)
    if generator is None:
        raise KeyError(f"unknown procedural generator {spec.generator!r}; "
                       f"have {sorted(GENERATORS)}")
    params = dict(spec.params)
    params.setdefault("color", tuple(spec.base_color))
    params.setdefault("seed", abs(hash(spec.id)) % 100000)
    albedo, height, roughness = generator(**params)
    return {
        "albedo": (np.clip(albedo, 0, 1) * 255).astype(np.uint8),
        "normal": (normal_map(height) * 255).astype(np.uint8),
        "orm": (np.stack([                                  # occlusion/rough/metal
            np.clip(0.55 + 0.45 * height, 0, 1),
            np.clip(roughness * spec.roughness, 0, 1),
            np.full_like(height, spec.metallic),
        ], axis=-1) * 255).astype(np.uint8),
    }

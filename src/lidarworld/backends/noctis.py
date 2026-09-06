"""NOCTIS-7 backend: a World Seed as a walkable ASCII city.

The other backends emit geometry -- triangles, materials, a scene graph. This
one emits no geometry at all. It emits a 256 KB texture, and a renderer that
has never heard of the compiler raycasts a height field out of it and prints
the result as text characters.

That is the point of it. `web`, `gltf` and `cityjson` are three ways of writing
down the same triangles, so they cannot tell you whether the IR is really
engine-independent or merely glTF-shaped. NOCTIS-7 has no triangles, no
materials, no meshes and no scene graph; a world reaches it as four bytes per
4 m cell. If the seed drives that too, the claim holds:

    reality -> seed -> {glTF world, CityJSON model, ASCII megacity}

The encoding is the renderer's, documented in CityBuilder's index.html and in
docs/MASTER.md. An N-wide, 3N-tall opaque RGBA8 PNG, three stacked planes:

    rows      0..N    R height in 1.6 m steps   G palette | style<<4   B window density
    rows     N..2N    R flag bits (road / park / water / sign / beacon / plaza)
    rows    2N..3N    R terrain elevation in `terrain_step_m` steps

Lossy, and lossier than the seed already is: a facade becomes one nibble, a
building becomes a column of cells, and anything under 4 m across disappears.
Nothing here is recoverable back into the seed, which is fine -- this is a
target, and targets are the end of the road.

Style is derived from measured quantities only (height, roof form, whether the
cell fronts a carriageway). No material, texture or theme is named, so the
invariant survives the trip: materialisation happens in the renderer, from the
same four bytes any other city hands it.
"""
from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import numpy as np

#: Renderer metres per height byte. A 255-byte column is 408 m.
HSTEP = 1.6

# Flag bits, in the renderer's order.
ROAD, ROADX, ROADZ, PARK, WATER, SIGNSTRIP, BEACON, PLAZA = (
    1, 2, 4, 8, 16, 32, 64, 128)

# Facade styles, in the high nibble of G. The renderer picks a window pattern
# from these; they are indices into its table, not materials.
STYLE_UNIFORM, STYLE_RIBBON, STYLE_GLAZED, STYLE_STAGGERED, STYLE_PLANTED = (
    0, 1, 2, 3, 7)

#: A building at least this tall gets a rooftop beacon, as aviation lighting
#: rules would put one there.
BEACON_M = 55.0


def _write_png(rgba: np.ndarray, path: Path) -> int:
    """Write an RGBA8 array as a PNG. No Pillow: numpy is the only dependency.

    Alpha is forced opaque by the caller, not here -- a canvas backing store is
    premultiplied, so a data byte in alpha silently destroys RGB wherever it is
    small, and flag bits are mostly small. That bug is why the format stacks
    planes vertically instead of using the fourth channel.
    """
    height, width, _ = rgba.shape
    raw = bytearray()
    for row in rgba:
        raw.append(0)                       # filter type 0 (None)
        raw += row.tobytes()

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + tag + payload
                + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
           + chunk(b"IEND", b""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)
    return len(png)


def _ring_mask(ring: np.ndarray, grid: int):
    """Even-odd fill of a closed ring given in cell coordinates.

    Returns ``(y0, x0, mask)`` over the ring's bounding box, or ``None`` when it
    falls outside the grid entirely. Bounded by the box rather than the grid
    because a tile holds thousands of footprints and almost all of them are a
    few cells across.
    """
    if len(ring) < 3:
        return None
    if not np.allclose(ring[0], ring[-1]):
        ring = np.vstack([ring, ring[:1]])

    x0 = max(0, int(np.floor(ring[:, 0].min())))
    x1 = min(grid - 1, int(np.ceil(ring[:, 0].max())))
    y0 = max(0, int(np.floor(ring[:, 1].min())))
    y1 = min(grid - 1, int(np.ceil(ring[:, 1].max())))
    if x1 < x0 or y1 < y0:
        return None

    px = (np.arange(x0, x1 + 1) + 0.5)[None, :]
    py = (np.arange(y0, y1 + 1) + 0.5)[:, None]
    inside = np.zeros((py.size, px.size), dtype=bool)

    ax, ay = ring[:-1, 0], ring[:-1, 1]
    bx, by = ring[1:, 0], ring[1:, 1]
    for i in range(len(ax)):
        dy = by[i] - ay[i]
        if dy == 0.0:                        # horizontal edge crosses nothing
            continue
        straddles = (ay[i] > py) != (by[i] > py)
        cut = ax[i] + (py - ay[i]) * (bx[i] - ax[i]) / dy
        inside ^= straddles & (px < cut)
    return y0, x0, inside


def _stamp(field: np.ndarray, y0: int, x0: int, mask: np.ndarray, value) -> None:
    """Raise `field` to `value` where `mask` is set. Overlaps take the taller."""
    window = field[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]]
    np.maximum(window, np.where(mask, value, 0), out=window)


def _disc(grid: int, cx: float, cy: float, radius: float):
    """A filled circle in cell coordinates, bounded by its box."""
    x0 = max(0, int(np.floor(cx - radius)))
    x1 = min(grid - 1, int(np.ceil(cx + radius)))
    y0 = max(0, int(np.floor(cy - radius)))
    y1 = min(grid - 1, int(np.ceil(cy + radius)))
    if x1 < x0 or y1 < y0:
        return None
    px = (np.arange(x0, x1 + 1) + 0.5)[None, :]
    py = (np.arange(y0, y1 + 1) + 0.5)[:, None]
    return y0, x0, ((px - cx) ** 2 + (py - cy) ** 2) <= max(radius, 0.5) ** 2


def _bounds(seed: dict) -> np.ndarray:
    """The seed's bounds, or the footprints' if it did not record any."""
    bounds = np.asarray(seed.get("bounds") or [], dtype=float)
    if bounds.shape == (2, 3) and np.ptp(bounds[:, :2]) > 0:
        return bounds
    rings = [np.asarray(b["footprint"], dtype=float)
             for b in seed.get("buildings", []) if len(b.get("footprint", [])) >= 3]
    if not rings:
        raise ValueError("seed has neither bounds nor footprints; nothing to bake")
    stacked = np.vstack(rings)
    zs = [b.get("ground_z", 0.0) for b in seed.get("buildings", [])]
    hi = [b.get("ground_z", 0.0) + b.get("height", 0.0)
          for b in seed.get("buildings", [])]
    return np.array([[stacked[:, 0].min(), stacked[:, 1].min(), min(zs)],
                     [stacked[:, 0].max(), stacked[:, 1].max(), max(hi)]])


def _terrain_plane(seed: dict, grid: int, lo: np.ndarray, cell: float):
    """Resample the seed's ground grid onto the renderer's cells.

    The seed stores terrain as ``z[ix][iy]`` from ``bounds[0]`` at ``step_m``;
    the renderer wants a byte per cell against its own quantisation. Nearest
    sample, because the renderer's cell is 4 m and the seed's step is typically
    4 m too -- interpolating between two samples of the same size invents a
    gradient that was never measured.
    """
    terrain = seed.get("terrain") or {}
    if not terrain.get("z"):
        return np.zeros((grid, grid), dtype=np.uint8), 1.0, 0.0, 0.0

    nx, ny = terrain["shape"]
    step = float(terrain["step_m"])
    z = np.asarray(terrain["z"], dtype=float)
    if z.ndim == 1:
        z = z.reshape(nx, ny)
    t_lo = np.asarray(seed["bounds"][0], dtype=float)[:2]

    centres = lo[:, None] + (np.arange(grid) + 0.5) * cell        # (2, grid)
    ix = np.clip(((centres[0] - t_lo[0]) / step).astype(int), 0, z.shape[0] - 1)
    iy = np.clip(((centres[1] - t_lo[1]) / step).astype(int), 0, z.shape[1] - 1)
    ground = z[np.ix_(ix, iy)].T                                  # -> [iy, ix]

    ground = np.nan_to_num(ground, nan=float(np.nanmedian(z)))
    base = float(np.nanmin(ground))
    relief = float(np.nanmax(ground)) - base
    tstep = max(relief, 1e-6) / 255.0
    plane = np.clip((ground - base) / tstep, 0, 255).astype(np.uint8)
    return plane, tstep, base, relief


def _credit(seed: dict) -> str:
    """What the renderer should put on screen under the city's name.

    A source's `attribution` is the line its provider asks for; its id is an
    internal label. Showing the id credits nobody, and an empty credit on a
    real survey is worse than saying the terms are unrecorded.
    """
    lines = []
    for source in seed.get("provenance", {}).get("sources", []):
        if isinstance(source, str):        # seeds written before terms travelled
            lines.append(source)
            continue
        name = source.get("attribution") or ""
        terms = source.get("license") or ""
        if name and terms:
            # Separator, not brackets: licence strings carry their own
            # parentheses ("CC0 1.0 (public domain dedication)") and nesting
            # them reads like a typo on screen.
            lines.append(f"{name} \u00b7 {terms}")
        else:
            lines.append(name or terms or source.get("id", "unrecorded source"))
    return " / ".join(dict.fromkeys(lines)) or "source and terms unrecorded"


def bake(seed: dict, *, grid: int = 256, cell_m: float = 4.0,
         extent_m: float | None = None) -> tuple:
    """World Seed -> the renderer's four bytes per cell.

    Returns ``(planes, meta)`` where `planes` is ``(3 * grid, grid, 4)`` uint8,
    ready to write, and `meta` is what the renderer needs to interpret it: the
    cell size, the terrain quantisation, and the counts, so a bad bake is
    visible in the numbers before anyone opens it.

    The cell defaults to 4 m because that is what the renderer's existing
    measured cities were baked at, and its world cell is a fixed unit: change
    the metres per cell and the same building comes out a different size next
    to a city baked at the default. Fit the bounds with `extent_m` only when
    apparent scale does not matter.
    """
    bounds = _bounds(seed)
    centre = (bounds[0][:2] + bounds[1][:2]) / 2.0
    span = float(max(bounds[1][0] - bounds[0][0], bounds[1][1] - bounds[0][1]))
    extent = float(extent_m) if extent_m else grid * float(cell_m)
    cell = extent / grid
    lo = centre - extent / 2.0

    def to_cell(xy: np.ndarray) -> np.ndarray:
        return (np.asarray(xy, dtype=float)[..., :2] - lo) / cell

    height = np.zeros((grid, grid), dtype=float)       # metres above ground
    style = np.zeros((grid, grid), dtype=np.uint8)
    flags = np.zeros((grid, grid), dtype=np.uint8)
    building = np.zeros((grid, grid), dtype=bool)
    canopy = np.zeros((grid, grid), dtype=bool)

    # ── roads ────────────────────────────────────────────────────────────
    # The carriageway is stamped first so a footprint that overlaps it wins:
    # the seed's roads are centrelines with a half-width, not surveyed kerbs,
    # and a metre of overlap is inside their error.
    for road in seed.get("roads", []):
        line = to_cell(np.asarray(road.get("line", []), dtype=float))
        if len(line) < 2:
            continue
        half = max(float(road.get("half_width", 4.0)) / cell, 0.5)
        for (ax, ay), (bx, by) in zip(line[:-1], line[1:]):
            dx, dy = bx - ax, by - ay
            length = float(np.hypot(dx, dy))
            if length == 0.0:
                continue
            steps = int(length * 2) + 2
            for t in np.linspace(0.0, 1.0, steps):
                disc = _disc(grid, ax + dx * t, ay + dy * t, half)
                if disc is None:
                    continue
                y0, x0, mask = disc
                axis = ROADZ if abs(dy) > abs(dx) else ROADX
                window = flags[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]]
                window |= np.where(mask, ROAD | axis, 0).astype(np.uint8)

    # ── water ────────────────────────────────────────────────────────────
    for body in seed.get("water", []):
        hit = _ring_mask(to_cell(np.asarray(body.get("ring", []), dtype=float)), grid)
        if hit is None:
            continue
        y0, x0, mask = hit
        window = flags[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]]
        window &= np.where(mask, ~np.uint8(ROAD | ROADX | ROADZ), 0xFF).astype(np.uint8)
        window |= np.where(mask, WATER, 0).astype(np.uint8)

    # ── buildings ────────────────────────────────────────────────────────
    for entry in seed.get("buildings", []):
        ring = to_cell(np.asarray(entry.get("footprint", []), dtype=float))
        hit = _ring_mask(ring, grid)
        if hit is None:
            continue
        y0, x0, mask = hit
        metres = float(entry.get("height", 0.0))
        if metres <= 0.0:
            continue
        _stamp(height, y0, x0, mask, metres)
        window = building[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]]
        window |= mask

        # Style from measured quantities only: how tall it is and what the
        # roof planes said. A pitched roof reads as older stock, a tall flat
        # one as a curtain wall. Nothing here names a material.
        roof = str(entry.get("roof", "flat"))
        if metres >= 60.0:
            kind = STYLE_GLAZED
        elif roof != "flat":
            kind = STYLE_UNIFORM
        elif metres >= 24.0:
            kind = STYLE_RIBBON
        else:
            kind = STYLE_STAGGERED
        _stamp(style, y0, x0, mask, np.uint8(kind))

        if metres >= BEACON_M:
            window = flags[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]]
            window |= np.where(mask, BEACON, 0).astype(np.uint8)

    # ── vegetation ───────────────────────────────────────────────────────
    # A surveyed tree is a position and a crown, so it is a disc. It writes
    # height where no building already stands -- a tree overhanging a roof is
    # one cell of canopy the renderer would draw as a taller building.
    for tree in seed.get("vegetation", []):
        xy = to_cell(np.asarray(tree.get("xy", [0.0, 0.0]), dtype=float))
        disc = _disc(grid, float(xy[0]), float(xy[1]),
                     max(float(tree.get("crown_r", 2.0)) / cell, 0.5))
        if disc is None:
            continue
        y0, x0, mask = disc
        free = mask & ~building[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]]
        _stamp(height, y0, x0, free, float(tree.get("height", 6.0)))
        _stamp(style, y0, x0, free, np.uint8(STYLE_PLANTED))
        canopy[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]] |= free

    # ── derived context ──────────────────────────────────────────────────
    # `street_facing` is the one context flag the renderer has a use for: it
    # puts signage on the frontage. Same definition as the IR's -- a building
    # cell orthogonally adjacent to a carriageway -- computed here rather than
    # carried, because the seed does not record it and a 4 m grid can.
    road = (flags & ROAD).astype(bool)
    near_road = np.zeros_like(road)
    near_road[1:, :] |= road[:-1, :]
    near_road[:-1, :] |= road[1:, :]
    near_road[:, 1:] |= road[:, :-1]
    near_road[:, :-1] |= road[:, 1:]
    flags |= np.where(building & near_road & (height >= 8.0),
                      SIGNSTRIP, 0).astype(np.uint8)
    flags |= np.where(canopy & ~building, PARK, 0).astype(np.uint8)
    empty = (height <= 0.0) & (flags & (ROAD | WATER | PARK) == 0)
    flags |= np.where(empty, PLAZA, 0).astype(np.uint8)

    # ── quantise ─────────────────────────────────────────────────────────
    hbyte = np.clip(np.round(height / HSTEP), 0, 255).astype(np.uint8)

    # The low nibble is a palette index. The renderer re-ranks it per tile for
    # measured cities -- a low-rise place would otherwise only ever use the
    # dark end of the ramp -- so this is a sane default it is free to replace.
    tallest = max(float(height.max()), 1.0)
    palette = np.clip((height / tallest * 13.0).astype(np.uint8), 0, 13)

    density = np.clip(0.22 + height / 260.0, 0.15, 0.72)

    terrain_plane, tstep, base_z, relief = _terrain_plane(seed, grid, lo, cell)

    planes = np.zeros((grid * 3, grid, 4), dtype=np.uint8)
    planes[:grid, :, 0] = hbyte
    planes[:grid, :, 1] = (palette & 15) | ((style & 15) << 4)
    planes[:grid, :, 2] = (density * 255).astype(np.uint8)
    planes[grid:grid * 2, :, 0] = flags
    planes[grid * 2:, :, 0] = terrain_plane
    planes[..., 3] = 255                      # opaque; the planes carry the data

    # A tile smaller than the seed crops it, and that has to be visible rather
    # than discovered later by wondering where half the city went.
    centroids = np.array([np.asarray(b["footprint"], dtype=float)[:, :2].mean(axis=0)
                          for b in seed.get("buildings", [])
                          if len(b.get("footprint", [])) >= 3])
    if len(centroids):
        inside = to_cell(centroids)
        kept = int(((inside >= 0) & (inside < grid)).all(axis=1).sum())
    else:
        kept = 0

    built = int(building.sum())
    meta = {
        "name": str(seed.get("name", "world")).upper().replace("_", " "),
        "credit": _credit(seed),
        "crs": seed.get("crs", ""),
        "grid": grid,
        "cellM": round(cell, 4),
        "extentM": round(extent, 1),
        "terrStep": round(tstep, 6),
        "relief": round(relief, 1),
        "groundMinZ": round(base_z, 2),
        "buildingCells": built,
        "roadCells": int(road.sum()),
        "waterCells": int((flags & WATER).astype(bool).sum()),
        "parkCells": int((flags & PARK).astype(bool).sum()),
        "tallestM": round(float(height.max()), 1),
        "meanBuildingM": round(float(height[building].mean()), 1) if built else 0.0,
        "seedBuildings": len(seed.get("buildings", [])),
        "seedRoads": len(seed.get("roads", [])),
        "seedTrees": len(seed.get("vegetation", [])),
        "seedSpanM": round(span, 1),
        "buildingsPlaced": kept,
        "buildingsCropped": len(centroids) - kept,
        "note": "Baked from a World Seed. Lossy: below the cell size nothing "
                "survives, and this cannot be read back into a seed.",
    }
    return planes, meta


def export(seed: dict, path: str | Path, *, grid: int = 256, cell_m: float = 4.0,
           extent_m: float | None = None, meta_path: str | Path | None = None) -> dict:
    """Bake a seed and write the PNG. Returns the meta, plus what it cost."""
    planes, meta = bake(seed, grid=grid, cell_m=cell_m, extent_m=extent_m)
    path = Path(path)
    meta["bytes"] = _write_png(planes, path)
    meta["path"] = str(path)
    if meta_path is not None:
        meta_path = Path(meta_path)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=1))
    return meta

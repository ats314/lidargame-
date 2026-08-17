"""Carriageway from the street network, because intensity does not find it.

Roads are inferred from return intensity: asphalt reflects poorly at 905 nm, so
a flat low-intensity ground cell is called road. On the Denver block that
recovers 3,536 of 60,716 terrain vertices -- under 6% -- for a downtown grid
where the streets are nearer a third of the open ground. Intensity is not
calibrated between flight lines, concrete and asphalt overlap, and a wet or
freshly sealed surface moves the threshold. The result is a city whose ground
is one undifferentiated field.

The street network says where the carriageway is, authoritatively, and Denver
publishes it. Rasterising the centrelines with a width buffer turns a guess into
a lookup, and the kerb line -- the boundary between road and ground, which is
where the context mask puts EDGE_ANY and where a theme hangs its kerbstones --
becomes a real edge rather than an intensity artefact.

The centrelines are geometry, not elevation: they say where the road is in plan
and nothing about height. Terrain height still comes entirely from the returns,
so this adds evidence without inventing surface.
"""
from __future__ import annotations

import numpy as np


def polylines(geojson: dict) -> list[np.ndarray]:
    """Flatten GeoJSON LineString/MultiLineString into (N,2) arrays."""
    out = []
    for feature in geojson.get("features", []):
        geometry = feature.get("geometry") or {}
        kind = geometry.get("type")
        if kind == "LineString":
            out.append(np.asarray(geometry["coordinates"], dtype=np.float64)[:, :2])
        elif kind == "MultiLineString":
            for part in geometry["coordinates"]:
                out.append(np.asarray(part, dtype=np.float64)[:, :2])
    return [line for line in out if len(line) >= 2]


#: Carriageway width in metres by road class. Denver's centrelines carry a class
#: rather than a width, so this is the lookup. Wrong by a metre is invisible;
#: one width for every street is a city with no hierarchy.
CARRIAGEWAY = {
    "FREEWAY": 24.0, "EXPRESSWAY": 22.0, "MAJOR ARTERIAL": 18.0,
    "ARTERIAL": 16.0, "MINOR ARTERIAL": 15.0, "COLLECTOR": 13.0,
    "LOCAL": 10.0, "RESIDENTIAL": 10.0, "RAMP": 9.0, "ALLEY": 6.0,
}

#: Fields that actually carry the class, most reliable first. VOLCLASS is the
#: one to trust on Denver's network: FUNCLASS exists and is populated but
#: disagrees with it -- segments VOLCLASS calls ARTERIAL are labelled
#: "Local-Urban" by FUNCLASS -- so reading FUNCLASS gives a flat, wrong city.
CLASS_FIELDS = ("VOLCLASS", "FUNCTIONAL_CLASS", "ROADCLASS", "STREETTYPE", "CLASS")


def widths(geojson: dict, *, default: float = 11.0) -> list[float]:
    """Half-width per line, from the network's own class attribute."""
    out = []
    for feature in geojson.get("features", []):
        props = feature.get("properties") or {}
        name = ""
        for key in CLASS_FIELDS:
            if props.get(key):
                name = str(props[key]).upper()
                break
        # Longest key first, so "MAJOR ARTERIAL" is not matched by "ARTERIAL".
        width = next((w for key, w in sorted(CARRIAGEWAY.items(),
                                             key=lambda kv: -len(kv[0]))
                      if key in name), default)
        geometry = feature.get("geometry") or {}
        parts = 1 if geometry.get("type") == "LineString" else len(
            geometry.get("coordinates", []) or [1])
        out.extend([width / 2.0] * max(parts, 1))
    return out


def rasterise(lines, half_widths, raster, *, samples_per_metre: float = 2.0) -> np.ndarray:
    """Boolean mask of cells within `half_width` of any centreline.

    Stamps a disc per sample rather than rasterising a polygon: the network is a
    few hundred segments over a block, the discs overlap into a continuous
    carriageway, and it needs no polygon clipper.
    """
    mask = np.zeros(raster.shape, dtype=bool)
    if not len(lines):
        return mask
    cell = raster.cell
    for line, half in zip(lines, half_widths):
        segments = np.diff(line, axis=0)
        lengths = np.hypot(segments[:, 0], segments[:, 1])
        for (a, b), length in zip(zip(line[:-1], line[1:]), lengths):
            if length <= 0:
                continue
            n = max(2, int(length * samples_per_metre))
            t = np.linspace(0.0, 1.0, n)[:, None]
            points = a[None, :] * (1 - t) + b[None, :] * t
            _stamp(mask, points, half, raster, cell)
    return mask


def _stamp(mask, points, half, raster, cell) -> None:
    reach = max(1, int(np.ceil(half / cell)))
    ij = raster.to_cell(points)
    offsets = np.arange(-reach, reach + 1)
    du, dv = np.meshgrid(offsets, offsets, indexing="ij")
    inside = (du * cell) ** 2 + (dv * cell) ** 2 <= half ** 2
    du, dv = du[inside], dv[inside]
    for oi, oj in zip(du.ravel(), dv.ravel()):
        i = np.clip(ij[:, 0] + oi, 0, raster.nx - 1)
        j = np.clip(ij[:, 1] + oj, 0, raster.ny - 1)
        mask[i, j] = True


def apply(class_raster: np.ndarray, road_mask: np.ndarray, *,
          ground: int = 0, road: int = 1, void: int = 255) -> dict:
    """Promote ground cells under the network to road. Never invents surface.

    A cell with no returns stays VOID: the network says where a road *is*, not
    that the ground was observed there, and painting unobserved cells would put
    carriageway over a scan shadow.
    """
    eligible = road_mask & (class_raster == ground)
    before = int((class_raster == road).sum())
    class_raster[eligible] = road
    return {
        "promoted": int(eligible.sum()),
        "road_cells_before": before,
        "road_cells_after": int((class_raster == road).sum()),
        "network_cells_unobserved": int((road_mask & (class_raster == void)).sum()),
    }

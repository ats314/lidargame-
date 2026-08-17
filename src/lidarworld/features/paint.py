"""Paint returns with orthophoto and polygon evidence, then throw the paint away.

Classical PointPainting projects a segmented camera image onto LiDAR using the
camera pose. We have no calibrated perspective imagery, and pretending
otherwise was a mistake. What we do have is better suited to the problem
anyway: a georeferenced orthophoto and surveyed polygons, both of which share a
coordinate system with the returns. So the projection is a lookup, not a pose
estimate:

    (u, v) = T_geo->image(x, y)

That works for anything the camera saw from above -- roofs, carriageway,
pavement, plaza, rail, lawn, canopy -- and says nothing whatever about walls,
because a nadir image cannot see them. Which is the right trade for the failure
in front of us: the compiler currently cannot reliably tell a building from a
rail corridor from a street, and none of that needs a facade.

The output is per-point evidence, and it is deliberately temporary. Fifty
million permanently coloured points is not a world. The points are the medium
in which surfaces get separated; once they are, the surface carries the
appearance and the points can go.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..data.gis import point_in_polygon


def rings_of(path: str | Path) -> list[np.ndarray]:
    """Exterior rings from a GeoJSON polygon layer."""
    data = json.loads(Path(path).read_text())
    out: list[np.ndarray] = []
    for feature in data.get("features", []):
        geometry = feature.get("geometry") or {}
        kind = geometry.get("type")
        if kind == "Polygon":
            parts = [geometry["coordinates"][0]]
        elif kind == "MultiPolygon":
            parts = [part[0] for part in geometry["coordinates"]]
        else:
            continue
        for part in parts:
            ring = np.asarray(part, dtype=float)
            if ring.ndim == 2 and len(ring) >= 4:
                out.append(ring[:, :2])
    return out


def inside_any(xy: np.ndarray, rings: list[np.ndarray]) -> np.ndarray:
    """Which of `xy` fall inside any ring. Bounding boxes first, or a downtown
    block against thousands of polygons is an hour instead of a second."""
    hit = np.zeros(len(xy), dtype=bool)
    if not len(xy):
        return hit
    for ring in rings:
        lo, hi = ring.min(axis=0), ring.max(axis=0)
        near = ((xy[:, 0] >= lo[0]) & (xy[:, 0] <= hi[0])
                & (xy[:, 1] >= lo[1]) & (xy[:, 1] <= hi[1]) & ~hit)
        if not near.any():
            continue
        index = np.flatnonzero(near)
        hit[index] |= point_in_polygon(xy[index], ring)
    return hit


class Orthophoto:
    """A georeferenced image tile set, sampled by world coordinate.

    World files give an affine, so a point's pixel is arithmetic. Tiles are
    opened lazily and one at a time: a 3 inch tile is 10560 x 10560 x 4 bytes,
    and holding nine of them is 4 GB of resident memory for no reason.
    """

    def __init__(self, directory: str | Path, *, transformer=None):
        self.directory = Path(directory)
        self.transformer = transformer
        self.tiles = []
        for world_file in sorted(self.directory.glob("*.tfw")):
            image = world_file.with_suffix(".tif")
            if not image.exists():
                continue
            a, _, _, e, cx, fy = [float(v) for v in world_file.read_text().split()]
            self.tiles.append({"tif": image, "a": a, "e": e, "cx": cx, "fy": fy})

    def __len__(self) -> int:
        return len(self.tiles)

    def sample(self, xy: np.ndarray) -> np.ndarray:
        """(N, 4) uint8 of whatever bands the imagery carries; 0 where uncovered.

        `xy` is in the point cloud's own frame; `transformer` converts it to the
        imagery's. Points outside every tile come back zero rather than raising,
        because an AOI legitimately runs off the edge of an acquisition.
        """
        import tifffile

        if self.transformer is not None:
            ix, iy = self.transformer.transform(xy[:, 0], xy[:, 1])
        else:
            ix, iy = xy[:, 0], xy[:, 1]
        out = np.zeros((len(xy), 4), dtype=np.uint8)
        remaining = np.ones(len(xy), dtype=bool)

        for tile in self.tiles:
            if not remaining.any():
                break
            with tifffile.TiffFile(tile["tif"]) as handle:
                page = handle.pages[0]
                height, width = page.shape[0], page.shape[1]
                px = ((ix - tile["cx"]) / tile["a"]).astype(np.int64)
                py = ((iy - tile["fy"]) / tile["e"]).astype(np.int64)
                here = remaining & (px >= 0) & (px < width) & (py >= 0) & (py < height)
                if not here.any():
                    continue
                array = page.asarray()
            index = np.flatnonzero(here)
            bands = min(4, array.shape[2] if array.ndim == 3 else 1)
            out[index, :bands] = array[py[index], px[index], :bands]
            remaining[index] = False
        return out


def paint(cloud, *, ortho: Orthophoto | None = None, regions=None,
          transformer=None) -> dict:
    """Attach appearance and region-membership channels to a cloud.

    Adds `rgb` (and `nir` where the imagery has a fourth band) plus one boolean
    channel per named region. Returns a summary of what stuck, because a
    silently uncovered AOI looks exactly like a correctly painted one.
    """
    xy = cloud.xyz[:, :2]
    summary: dict = {"points": int(len(cloud))}

    if ortho is not None and len(ortho):
        pixels = ortho.sample(xy)
        covered = pixels.any(axis=1)
        cloud["rgb"] = pixels[:, :3]
        cloud["nir"] = pixels[:, 3]
        cloud["painted"] = covered
        summary["imagery"] = {"tiles": len(ortho),
                              "covered": float(covered.mean())}

    for name, rings in (regions or {}).items():
        inside = inside_any(xy, rings)
        cloud[f"in_{name}"] = inside
        summary.setdefault("regions", {})[name] = {
            "polygons": len(rings), "points_inside": int(inside.sum()),
            "fraction": round(float(inside.mean()), 4)}
    return summary


#: Ground surfaces, most specific first. Where two surveyed polygons cover the
#: same ground, the narrower claim wins: a parking bay is a more particular
#: statement than "pavement", and a pavement polygon that also contains a
#: sidewalk polygon is the looser of the two.
GROUND_PRECEDENCE = ("parking", "sidewalk", "pavement")

#: A point this far above local ground is not standing on a ground surface,
#: whatever polygon it happens to sit over. Chosen well above kerb, step and
#: ramp height so real ground detail is never reassigned.
GROUND_BAND_M = 2.0


def reconcile(cloud, *, ground_band: float = GROUND_BAND_M,
              hag_channel: str = "hag") -> dict:
    """Give every point one owning surface instead of several.

    Region membership overlaps badly -- on a LoDo crop 48.9% of returns are
    inside a roofprint, 40.0% inside pavement and 33.0% inside a sidewalk,
    which is 128% of the cloud. A point cannot be three surfaces.

    Almost all of it is one mistake: testing membership in XY while ignoring Z.
    A roofprint overhangs the pavement that runs underneath it, so a roof point
    14 m up is genuinely inside both polygons in plan. Measured on that crop,
    94% of roof/pavement conflicts sit above 2 m and their median height is
    14.0 m -- they are roofs, and nothing about them was ever ambiguous.

    So height decides first and the polygons only arbitrate among things
    actually on the ground:

        above the band, inside a roofprint   -> roof
        above the band, outside one          -> not a surveyed surface at all
        on the ground                        -> most specific polygon wins

    Writes an `owner` channel and returns what changed. Points owned by nothing
    stay unowned rather than being assigned a nearest guess: an unclaimed
    surface is a real answer and the compiler has a state for it.
    """
    n = len(cloud)
    owner = np.full(n, "", dtype=object)
    hag = cloud.get(hag_channel)
    if hag is None:
        raise KeyError(f"reconcile needs a {hag_channel!r} channel; run the "
                       "terrain stage first")
    elevated = np.asarray(hag) > ground_band

    roof = cloud.get("in_roofprint")
    if roof is not None:
        owner[elevated & np.asarray(roof)] = "roof"

    for name in GROUND_PRECEDENCE:
        mask = cloud.get(f"in_{name}")
        if mask is None:
            continue
        claim = (~elevated) & np.asarray(mask) & (owner == "")
        owner[claim] = name

    if roof is not None:
        # A footprint's own ground: inside the building, under the roof.
        owner[(~elevated) & np.asarray(roof) & (owner == "")] = "building_ground"

    cloud["owner"] = owner
    counts: dict = {}
    for value in owner:
        counts[value or "unowned"] = counts.get(value or "unowned", 0) + 1
    return {"points": int(n), "owned": int((owner != "").sum()),
            "by_owner": dict(sorted(counts.items(), key=lambda kv: -kv[1]))}

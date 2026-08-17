"""Stamp a reality mesh with the semantics of a city model, and georeference it.

The reality mesh is the honest input and the useless one. It has 42 cm triangles
and 7.7 cm texels -- enough surface for a person to stand in front of -- and no
idea what any of it is. No building ids, no surface classes, not even a
coordinate system: a Helsinki chunk's vertices read like (6187, 4043, 24), which
is neither ETRS-GK25 nor a tile corner. That is why a reality mesh cannot be a
compiler target on its own. A scan is not a world.

Helsinki happens to publish both halves for the same city. The CityGML LoD2
model is in absolute EPSG:3879 and carries a building id, a surface class and a
storey count on every wall; the mesh carries the geometry that CityGML does not
have. So the join does two jobs with one registration:

    georeference   the mesh gets absolute coordinates, derived and then checked
                   against footprints rather than asserted from a filename
    semantics      every mesh triangle gets a building id and a surface class,
                   from an independent model instead of from a segmenter

What comes out is not a scan any more. It is the mesh's measured surface indexed
by the city model's own objects, which is what every later stage in this repo
wants: per-building material instances, per-surface UV frames, a storey count to
check a detected lattice against.

Two things this deliberately does not do. It does not move the mesh to fit --
registration is a pure translation, found by search and reported with its score,
so a bad fit shows up as a bad number rather than as a warped city. And it does
not promote anything to `observed`. Both sides are somebody's photogrammetry;
the join is `derived` from two `derived` sources, and the agreement between them
is evidence about the pair, not proof of either.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..reconstruct.tessellate import HORIZONTAL_NZ

#: Grid pitch of the footprint raster, metres. A CityGML footprint edge is a
#: straight line and the mesh wall that belongs to it wanders either side of it
#: by a photogrammetric error, so the raster only has to be finer than that
#: wander. 0.5 m keeps a 2,900-building model under 100 MB of int32.
CELL_M = 0.5

#: How far outside a footprint a wall triangle may sit and still belong to it.
#: Photogrammetric walls bulge: the mesh surface is the plaster, the render and
#: the drainpipe, while the CityGML footprint is the survey line. Measured
#: offsets on Kalasatama run to a metre.
WALL_TOLERANCE_M = 1.5

#: A horizontal surface this far above the footprint's own ground level is a
#: roof; below it is a courtyard, a plinth or the street.
ROOF_MIN_M = 2.0


#: How near a footprint's outline a mesh wall cell has to fall to count as
#: registered. The two models disagree about a wall's position by the plaster,
#: the render and the survey convention, which on Kalasatama runs under a metre.
EDGE_TOLERANCE_M = 1.5


@dataclass
class Registration:
    """A translation from mesh-local to absolute, with the evidence for it."""
    offset: np.ndarray                          # (3,) metres, local + offset = absolute
    score: float                                # fraction of wall cells on an outline
    proposed: np.ndarray                        # what the tile name implied
    residual_m: float                           # how far the search moved it
    searched: int = 0
    runner_up: float = 0.0
    field: dict = field(default_factory=dict)   # (dx, dy) -> score, for inspection

    @property
    def unique(self) -> bool:
        """Whether the winning offset actually stands out from its neighbours.

        A flat scoring surface means the mesh landed on *something* whichever way
        it was shifted. The first version of this scored footprint *interiors* and
        came out 0.465 against a runner-up of 0.461 on a dense block -- because
        half that block is building, so any shift lands half the walls on some
        building. The peak has to beat the best distinct position by a margin, and
        only a score that measures alignment rather than occupancy can produce one.
        """
        return self.score > 1.15 * self.runner_up

    def to_record(self) -> dict:
        return {
            "offset": [round(float(v), 1) for v in self.offset],
            "proposed": [round(float(v), 1) for v in self.proposed],
            "residual_m": round(self.residual_m, 2),
            "wall_cells_on_an_outline": round(self.score, 4),
            "runner_up": round(self.runner_up, 4),
            "unique_peak": self.unique,
            "offsets_searched": self.searched,
            "edge_tolerance_m": EDGE_TOLERANCE_M,
            "epistemic": "derived",         # a search result, not a survey datum
        }


@dataclass
class FootprintIndex:
    """CityGML building footprints rasterised, so a lookup is an array index."""
    ids: np.ndarray                             # (rows, cols) int32, -1 = no building
    lo: np.ndarray                              # (2,) absolute easting/northing
    cell_m: float
    buildings: list = field(default_factory=list)
    ground_z: np.ndarray = field(default_factory=lambda: np.zeros(0))
    roof_z: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: Lowest roof point -- the eaves. This is the top of the topmost storey,
    #: where `roof_z` is the top of the ridge, and the difference is attic. Using
    #: the ridge as the denominator of a storey height inflates it by the whole
    #: roof: a 7-storey block came out at 4.11 m per storey that way.
    eaves_z: np.ndarray = field(default_factory=lambda: np.zeros(0))
    storeys: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def lookup(self, xy: np.ndarray) -> np.ndarray:
        """Building index per absolute (x, y), or -1 outside every footprint."""
        cols = np.floor((xy[:, 0] - self.lo[0]) / self.cell_m).astype(np.int64)
        rows = np.floor((xy[:, 1] - self.lo[1]) / self.cell_m).astype(np.int64)
        inside = ((rows >= 0) & (rows < self.ids.shape[0]) &
                  (cols >= 0) & (cols < self.ids.shape[1]))
        out = np.full(len(xy), -1, dtype=np.int64)
        out[inside] = self.ids[rows[inside], cols[inside]]
        return out

    def outline_distance(self) -> np.ndarray:
        """Metres from each cell to the nearest footprint outline.

        The outline, not the interior. A mesh wall belongs on the survey line, so
        distance to that line is a sharp function of misregistration where
        containment is a blunt one. Courtyard rims count: the mesh sees those too.
        """
        from scipy import ndimage                 # noqa: PLC0415  (optional dep)
        built = self.ids >= 0
        rim = built & ~ndimage.binary_erosion(built, np.ones((3, 3), bool))
        if not rim.any():
            return np.full(self.ids.shape, np.inf)
        return ndimage.distance_transform_edt(~rim, sampling=self.cell_m)

    def dilated(self, metres: float) -> "FootprintIndex":
        """The same index with footprints grown, for walls that bulge outward.

        Grown by nearest-neighbour rather than by a morphological max, so a cell
        between two buildings goes to the closer one instead of to whichever has
        the larger id.
        """
        from scipy import ndimage                 # noqa: PLC0415  (optional dep)
        empty = self.ids < 0
        _, (rows, cols) = ndimage.distance_transform_edt(
            empty, sampling=self.cell_m, return_indices=True)
        distance = ndimage.distance_transform_edt(empty, sampling=self.cell_m)
        grown = self.ids[rows, cols]
        grown[distance > metres] = -1
        grown[~empty] = self.ids[~empty]
        return FootprintIndex(grown, self.lo, self.cell_m, self.buildings,
                              self.ground_z, self.roof_z, self.eaves_z,
                              self.storeys)


def _footprint_rings(building) -> list[np.ndarray]:
    """A building's ground polygons, or its wall bases if it has no ground surface."""
    rings = [p.exterior for s in building.of("ground") for p in s.polygons]
    if rings:
        return rings
    return [p.exterior for s in building.of("wall") for p in s.polygons]


def _number(building, *keys, default=np.nan) -> float:
    for key in keys:
        value = building.attributes.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except ValueError:
            continue
    return default


def index_footprints(buildings: list, *, cell_m: float = CELL_M,
                     bounds: tuple[np.ndarray, np.ndarray] | None = None
                     ) -> FootprintIndex:
    """Rasterise footprints to a building-index grid over `bounds`.

    Rasterising rather than testing point-in-polygon per triangle is not just
    speed: it makes the answer for a corner deterministic. Two adjacent
    footprints in a terrace share an edge, and a floating-point containment test
    on that edge answers differently depending on which polygon is asked first.
    """
    from PIL import Image, ImageDraw            # noqa: PLC0415

    rings_per_building = [_footprint_rings(b) for b in buildings]
    points = [r[:, :2] for rings in rings_per_building for r in rings if len(r)]
    if not points:
        raise ValueError("no footprint geometry in these buildings")
    stacked = np.vstack(points)
    lo = stacked.min(axis=0) if bounds is None else np.asarray(bounds[0], float)[:2]
    hi = stacked.max(axis=0) if bounds is None else np.asarray(bounds[1], float)[:2]
    shape = (max(1, int(np.ceil((hi[1] - lo[1]) / cell_m))),
             max(1, int(np.ceil((hi[0] - lo[0]) / cell_m))))

    # Painted as index+1 so 0 can mean "no building"; PIL's I mode is signed
    # int32 and its fill argument does not accept -1 cleanly.
    canvas = Image.new("I", (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(canvas)
    for slot, rings in enumerate(rings_per_building):
        for ring in rings:
            if len(ring) < 3:
                continue
            xy = [(float((x - lo[0]) / cell_m), float((y - lo[1]) / cell_m))
                  for x, y in ring[:, :2]]
            draw.polygon(xy, fill=slot + 1)
    ids = np.asarray(canvas, dtype=np.int32) - 1

    ground = np.array([_number(b, "GroundLevel") for b in buildings])
    roof = np.array([_number(b, "HighestRoof", "BREC_BuildingHeightNN")
                     for b in buildings])
    eaves = np.array([_number(b, "LowestRoof") for b in buildings])
    storeys = np.array([_number(b, "Kerroksia", "storeysAboveGround")
                        for b in buildings])
    # Fall back to the geometry where an attribute is missing: a ground surface's
    # own z is a ground level, whatever the model chose to record.
    for slot, rings in enumerate(rings_per_building):
        if np.isnan(ground[slot]) and rings:
            ground[slot] = float(np.min(np.vstack(rings)[:, 2]))
    return FootprintIndex(ids, lo, cell_m, buildings, ground, roof, eaves,
                          storeys)


def triangle_frames(mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centroids, unit normals and areas for every triangle in the mesh."""
    faces = [np.asarray(g.faces).reshape(-1, 3) for g in mesh.groups
             if len(np.asarray(g.faces))]
    if not faces:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    faces = np.vstack(faces)
    a, b, c = (mesh.positions[faces[:, i]] for i in range(3))
    cross = np.cross(b - a, c - a)
    length = np.linalg.norm(cross, axis=1)
    normals = cross / np.maximum(length, 1e-12)[:, None]
    return (a + b + c) / 3.0, normals, length / 2.0


#: Wall triangles below this height above the mesh's own local ground are not
#: building outline: they are kerbs, parked cars, bollards, street trees and the
#: reconstruction's own noise where the pavement meets a step. Excluding them is
#: what makes the score about buildings.
WALL_MIN_HEIGHT_M = 4.0

#: Neighbourhood for the local ground estimate, metres. Wide enough that a
#: courtyard block does not become its own ground level, narrow enough to follow
#: a city that slopes.
GROUND_SPAN_M = 40.0


def wall_samples(mesh, *, cell_m: float = CELL_M,
                 min_height_m: float = WALL_MIN_HEIGHT_M) -> np.ndarray:
    """One local (x, y) per occupied cell of the mesh's building walls.

    One sample per cell, so a densely triangulated frontage does not outvote a
    coarse one and the score is about how much outline is covered rather than how
    many triangles cover it.
    """
    centroids, normals, _ = triangle_frames(mesh)
    if not len(centroids):
        raise ValueError("mesh has no triangles to register")
    vertical = np.abs(normals[:, 2]) < HORIZONTAL_NZ

    # Local ground from the mesh alone -- the registration is not known yet, so
    # the city model's own GroundLevel cannot be used here without circularity.
    span = max(1, int(round(GROUND_SPAN_M / cell_m)))
    keys = np.floor(centroids[:, :2] / (span * cell_m)).astype(np.int64)
    ground: dict[tuple[int, int], float] = {}
    for key, z in zip(map(tuple, keys), centroids[:, 2]):
        if z < ground.get(key, np.inf):
            ground[key] = float(z)
    local_ground = np.array([ground[k] for k in map(tuple, keys)])
    tall = vertical & (centroids[:, 2] - local_ground >= min_height_m)
    if tall.sum() < 100:
        raise ValueError(f"only {int(tall.sum())} tall vertical triangles; "
                         "registration needs building walls")
    cells = np.unique(np.floor(centroids[tall][:, :2] / cell_m).astype(np.int64),
                      axis=0)
    return (cells + 0.5) * cell_m


def register(mesh, index: FootprintIndex, *, proposed: np.ndarray | None = None,
             search_m: float = 6.0, step_m: float = 0.5,
             tolerance_m: float = EDGE_TOLERANCE_M) -> Registration:
    """Find the translation that puts the mesh on the city model.

    The score is the fraction of mesh wall cells that land within `tolerance_m`
    of a footprint *outline*. Scoring containment instead was the obvious thing
    and it does not work: on a block that is half building, half the walls land
    inside some footprint whatever the shift, so the scoring surface came out flat
    at 0.46 and the peak meant nothing. Distance to the outline is sharp, because a
    wall moved two metres is off the line it belongs to whether or not it is still
    over a building.

    A pure translation, deliberately. Two reconstructions of the same city do not
    differ by a rotation, and allowing one would let a bad registration hide as a
    slightly warped city instead of as a low number.
    """
    sample = wall_samples(mesh, cell_m=index.cell_m)
    distance = index.outline_distance()

    proposed = (np.zeros(3) if proposed is None
                else np.asarray(proposed, dtype=float))
    offsets = np.arange(-search_m, search_m + step_m / 2, step_m)
    scores: dict[tuple[float, float], float] = {}
    for dx in offsets:
        for dy in offsets:
            shifted = sample + proposed[:2] + np.array([dx, dy])
            cols = np.floor((shifted[:, 0] - index.lo[0]) / index.cell_m).astype(np.int64)
            rows = np.floor((shifted[:, 1] - index.lo[1]) / index.cell_m).astype(np.int64)
            inside = ((rows >= 0) & (rows < distance.shape[0]) &
                      (cols >= 0) & (cols < distance.shape[1]))
            near = np.zeros(len(shifted), dtype=bool)
            near[inside] = distance[rows[inside], cols[inside]] <= tolerance_m
            scores[(round(float(dx), 3), round(float(dy), 3))] = float(near.mean())

    (best_dx, best_dy), best = max(scores.items(), key=lambda kv: kv[1])
    # The runner-up has to be a *different* position, not the neighbouring cell
    # of the same peak, or every peak looks non-unique. One tolerance away is far
    # enough that the two candidates cannot both be right.
    far = [v for (dx, dy), v in scores.items()
           if max(abs(dx - best_dx), abs(dy - best_dy)) > 2.0 * tolerance_m]
    offset = proposed + np.array([best_dx, best_dy, 0.0])
    return Registration(offset=offset, score=best, proposed=proposed,
                        residual_m=float(np.hypot(best_dx, best_dy)),
                        searched=len(scores),
                        runner_up=max(far) if far else 0.0, field=scores)


#: Surface classes this transfers. Deliberately the same words `ingest/citygml`
#: already uses, so a stamped mesh triangle and a CityGML surface are comparable
#: without a translation table.
CLASSES = ("wall", "roof", "ground", "unknown")


def transfer(mesh, index: FootprintIndex, registration: Registration, *,
             tolerance_m: float = WALL_TOLERANCE_M) -> dict:
    """Per-triangle building id and surface class, plus what the join measured.

    Classification is by geometry against the city model, not by the city model
    alone: CityGML says where a building is and how high its ground and roof sit,
    the mesh triangle's own normal and height say what part of it this is. That
    ordering matters, because the mesh contains things CityGML does not model at
    all -- trees, cars, the street -- and those must come out `unknown` or
    `ground` rather than being forced onto the nearest building.
    """
    centroids, normals, areas = triangle_frames(mesh)
    absolute = centroids + registration.offset
    vertical = np.abs(normals[:, 2]) < HORIZONTAL_NZ

    building = index.lookup(absolute[:, :2])
    # Walls bulge outward past the survey line, so a vertical triangle that
    # missed gets a second chance against the grown index. Roofs do not get one:
    # a horizontal triangle just outside a footprint is much more likely to be
    # the pavement beside the building than its eaves.
    outside = vertical & (building < 0)
    if outside.any() and tolerance_m > 0:
        try:
            grown = index.dilated(tolerance_m)
        except ImportError:
            grown = None
        if grown is not None:
            building[outside] = grown.lookup(absolute[outside][:, :2])
    recovered = int((outside & (building >= 0)).sum())

    has_building = building >= 0
    ground_z = np.where(has_building, index.ground_z[np.clip(building, 0, None)],
                        np.nan)
    height = absolute[:, 2] - ground_z

    surface = np.full(len(centroids), "unknown", dtype=object)
    surface[vertical & has_building] = "wall"
    surface[~vertical & has_building & (height >= ROOF_MIN_M)] = "roof"
    surface[~vertical & has_building & (height < ROOF_MIN_M)] = "ground"
    surface[~vertical & ~has_building] = "ground"       # street, park, water

    hit = {b for b in np.unique(building) if b >= 0}
    stamped_area = float(areas[has_building].sum())
    report = {
        "registration": registration.to_record(),
        "triangles": int(len(centroids)),
        "surface_area_m2": round(float(areas.sum()), 1),
        "by_class": {name: int((surface == name).sum()) for name in CLASSES},
        "area_by_class_m2": {name: round(float(areas[surface == name].sum()), 1)
                             for name in CLASSES},
        "stamped_fraction": round(float(has_building.mean()), 4),
        "stamped_area_fraction": round(stamped_area / max(areas.sum(), 1e-9), 4),
        "walls_recovered_by_tolerance": recovered,
        "wall_tolerance_m": tolerance_m,
        "buildings_touched": len(hit),
        "buildings_in_index": len(index.buildings),
        "epistemic": "derived from two derived sources; neither is observed",
    }
    report.update(_agreement(absolute, areas, vertical, building, index))
    return {"building": building, "surface": surface, "height": height,
            "absolute": absolute, "normals": normals, "areas": areas,
            "report": report}


def _agreement(absolute, areas, vertical, building, index) -> dict:
    """How far the two models disagree about height, per building.

    This is the cheap cross-model check, and the one number here that is not
    self-consistency. Denver has exactly one of these -- building heights against
    the city's aerial stereo, median 1.18 m -- and it caught real errors that no
    amount of internal scoring did.

    It is NOT a level-2 independent check, and saying so matters. Helsinki's
    CityGML and Helsinki's reality mesh may well descend from overlapping aerial
    campaigns, in which case a shared error cancels here and the agreement
    flatters both. Reported as a cross-model residual, not as validation.
    """
    roof = (~vertical) & (building >= 0)
    if roof.sum() < 20:
        return {"height_agreement": {"reason": "too few roof triangles"}}
    slots = building[roof]
    heights = absolute[roof][:, 2]
    # Several percentiles, not one. The mesh's roof height is a choice, and the
    # choice is exactly what the disagreement is about: photogrammetry sees the
    # chimney, the plant room, the railing and the parapet, while an LoD2 model
    # carries a modelled ridge. If the bias collapses as the percentile drops,
    # the disagreement is superstructure rather than datum, and that distinction
    # is the whole value of the check.
    percentiles = (50, 75, 90, 95, 99)
    rows: dict[int, list] = {p: [] for p in percentiles}
    counts = 0
    for slot in np.unique(slots):
        model_roof = index.roof_z[slot]
        if not np.isfinite(model_roof):
            continue
        mine = heights[slots == slot]
        if len(mine) < 8:
            continue
        for p in percentiles:
            rows[p].append(float(np.percentile(mine, p) - model_roof))
        counts += 1
    if not counts:
        return {"height_agreement": {"reason": "no comparable roof heights"}}
    headline = np.asarray(rows[95])
    return {"height_agreement": {
        "buildings_compared": counts,
        "mesh_percentile": 95,
        "median_abs_m": round(float(np.median(np.abs(headline))), 3),
        "median_bias_m": round(float(np.median(headline)), 3),
        "within_2m": round(float((np.abs(headline) <= 2.0).mean()), 3),
        "within_5m": round(float((np.abs(headline) <= 5.0).mean()), 3),
        "bias_by_percentile_m": {p: round(float(np.median(rows[p])), 3)
                                 for p in percentiles},
        "independence": "unverified; both models may share aerial campaigns",
    }}


def stamp_groups(mesh, joined: dict) -> list:
    """Split the mesh's material groups by building and surface class.

    An engine wants one draw call per texture; a facade material system wants one
    surface per wall. Splitting here rather than at the backend keeps the
    material grouping the mesh already has -- the texture atlas is what it is --
    while adding the semantic axis on top, so a downstream exporter can pick
    either or both.
    """
    from ..ingest.objmesh import Group           # noqa: PLC0415

    # A dict, not np.searchsorted: CLASSES is in a meaningful order rather than
    # a sorted one, and searchsorted on an unsorted table silently returns the
    # insertion point -- which for "wall" is 4, one past the end.
    slot_of = {name: i for i, name in enumerate(CLASSES)}
    out, cursor = [], 0
    for group in mesh.groups:
        faces = np.asarray(group.faces).reshape(-1, 3)
        if not len(faces):
            continue
        span = slice(cursor, cursor + len(faces))
        cursor += len(faces)
        classes = np.array([slot_of[str(s)] for s in joined["surface"][span]])
        key = np.stack([joined["building"][span], classes], axis=1)
        for unique in np.unique(key, axis=0):
            picked = (key == unique).all(axis=1)
            if not picked.any():
                continue
            new = Group(group.material, group.image, faces[picked])
            new.building = int(unique[0])                     # type: ignore[attr-defined]
            new.surface = CLASSES[int(unique[1])]              # type: ignore[attr-defined]
            out.append(new)
    return out


def join(mesh, gml_path: str | Path, *, proposed: np.ndarray | None = None,
         cell_m: float = CELL_M, search_m: float = 6.0,
         limit: int | None = None) -> dict:
    """Read a city model, register the mesh against it, and stamp the triangles."""
    from ..ingest import citygml                 # noqa: PLC0415

    buildings = citygml.read_buildings(gml_path, limit=limit)
    index = index_footprints(buildings, cell_m=cell_m)
    registration = register(mesh, index, proposed=proposed, search_m=search_m)
    return transfer(mesh, index, registration)

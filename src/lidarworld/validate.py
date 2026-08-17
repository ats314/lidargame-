"""Forward validation: does the reconstructed world explain the scan it came from?

Every stage before this one is inverse -- observations to structure. This stage
runs the arrow the other way: put a virtual sensor back where the real one
stood, cast the same beams at the *reconstructed* geometry, and compare the
ranges it returns against the ranges that were actually measured.

    observed scan  ->  compiler  ->  world  ->  simulated scan  ->  compare

That closes the loop, and it converts confidence from a heuristic into a
measurement. A wall that reproduces its returns to within a few centimetres is
real; a wall the compiler invented shows up immediately as simulated returns
with nothing behind them, and geometry it failed to reconstruct shows up as
observed returns that the simulation passes straight through.

Attribution is per node, so the result says *which* surfaces are trustworthy
rather than issuing one number for the whole tile.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .backends.web import mesh_slot_names
from .types import World


@dataclass
class Consistency:
    """Reconstruction-consistency report for one simulated viewpoint."""
    rays: int
    explained: int
    missing: int              # observed a return, simulation saw nothing
    early: int                # simulation hit something in front of the return
    late: int                 # simulation hit something behind the return
    grazing: int              # disagreement at near-tangent incidence (unreliable)
    backend: str              # which ray caster produced this
    rmse: float
    tolerance: float
    sensor: list[float]
    per_node: dict[str, dict] = field(default_factory=dict)

    @property
    def explained_fraction(self) -> float:
        return self.explained / max(self.rays, 1)

    def to_json(self) -> dict:
        return {
            "rays": self.rays, "explained": self.explained, "missing": self.missing,
            "early": self.early, "late": self.late, "grazing": self.grazing,
            "backend": self.backend,
            "explainedFraction": round(self.explained_fraction, 4),
            "rangeRmse": round(self.rmse, 4), "tolerance": self.tolerance,
            "sensor": [round(v, 3) for v in self.sensor],
            "perNode": self.per_node,
        }

    def summary(self) -> str:
        return (f"[{self.backend}] {self.explained:,}/{self.rays:,} returns explained "
                f"({self.explained_fraction:.1%}), range RMSE {self.rmse * 100:.1f} cm, "
                f"{self.missing:,} unexplained, {self.early:,} over-occluded, "
                f"{self.grazing:,} grazing (inconclusive)")


class EmbreeScene:
    """Exact ray-triangle intersection via Open3D's Embree-backed raycaster.

    This is the accurate backend and the default when Open3D is installed. It
    matters for two reasons the voxel version cannot fix:

    * **No thickening.** A voxel scene inflates every surface by up to half a
      cell, which at grazing incidence down a street occludes rays that would
      really have passed by -- inventing over-occlusion that is an artefact of
      the measurement, not of the reconstruction.
    * **True normals.** Embree returns the struck triangle's own normal, so the
      incidence angle is exact rather than quantised to a voxel face. That is
      what makes the grazing classification trustworthy instead of a guess.

    It is also CPU-only and vendor-neutral, so the closure loop does not depend
    on any particular GPU.
    """

    backend = "embree"

    def __init__(self, world: World, **_ignored):
        import open3d as o3d

        # Embree is float32 throughout, so the mesh has to be recentred before
        # it goes in: at a UTM northing float32 resolves half a metre, and a
        # scoreboard built on geometry snapped to a 0.5 m grid measures the
        # cast, not the reconstruction. Ray origins are shifted by the same
        # offset in `march`.
        world_positions = np.asarray(world.arrays["mesh/positions"], dtype=np.float64)
        self.geo_origin = ((world_positions.min(axis=0) + world_positions.max(axis=0)) / 2.0
                           if len(world_positions) else np.zeros(3))
        positions = (world_positions - self.geo_origin).astype(np.float32)
        indices = np.asarray(world.arrays["mesh/indices"], dtype=np.uint32).reshape(-1, 3)
        node_attr = np.asarray(world.arrays["mesh/node"], dtype=np.int64)

        self._o3d = o3d
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.core.Tensor(positions), o3d.core.Tensor(indices))
        # Every vertex of a quad carries the same node slot, so the first
        # index of each triangle identifies its owner.
        self.triangle_node = node_attr[indices[:, 0].astype(np.int64)]
        self.resolution = 0.0

    def march(self, origin: np.ndarray, directions: np.ndarray, max_range: float,
              **_ignored):
        n = len(directions)
        rays = np.empty((n, 6), dtype=np.float32)
        rays[:, 0:3] = (np.asarray(origin, dtype=np.float64)
                        - self.geo_origin).astype(np.float32)
        rays[:, 3:6] = directions.astype(np.float32)
        result = self.scene.cast_rays(self._o3d.core.Tensor(rays))

        t_hit = result["t_hit"].numpy().astype(np.float64)
        primitive = result["primitive_ids"].numpy().astype(np.int64)
        normals = result["primitive_normals"].numpy().astype(np.float64)

        missed = ~np.isfinite(t_hit) | (t_hit > max_range)
        t_hit[missed] = np.inf
        hit_node = np.where(missed, -1, self.triangle_node[np.clip(primitive, 0, len(self.triangle_node) - 1)])
        normals[missed] = 0.0
        return t_hit, hit_node.astype(np.int32), normals


class VoxelScene:
    """Dense occupancy + node attribution -- the numpy-only fallback.

    Used when Open3D is not installed. Surfaces are thickened to the voxel
    size and normals are quantised, so treat its over-occlusion and grazing
    counts as upper bounds rather than measurements. `EmbreeScene` is exact.
    """

    backend = "voxel"

    def __init__(self, world: World, resolution: float = 0.25, max_cells: int = 40_000_000):
        positions = np.asarray(world.arrays["mesh/positions"], dtype=np.float64)
        indices = np.asarray(world.arrays["mesh/indices"], dtype=np.int64).reshape(-1, 3)
        node_attr = np.asarray(world.arrays["mesh/node"], dtype=np.int64)

        lo = positions.min(axis=0) - resolution
        hi = positions.max(axis=0) + resolution
        dims = np.maximum(np.ceil((hi - lo) / resolution).astype(int) + 1, 1)
        while int(np.prod(dims)) > max_cells:
            resolution *= 1.5
            dims = np.maximum(np.ceil((hi - lo) / resolution).astype(int) + 1, 1)

        self.resolution = float(resolution)
        self.origin = lo
        self.dims = dims
        # 0 = empty, otherwise node slot + 1
        self.grid = np.zeros(int(np.prod(dims)), dtype=np.int32)
        # Surface orientation per voxel, so a hit can report its incidence
        # angle. Near-tangent hits are the one case a voxel scene cannot
        # adjudicate -- half a voxel of thickness swings the range by metres.
        self.normals = np.zeros((int(np.prod(dims)), 3), dtype=np.int8)

        self._rasterise(positions, indices, node_attr)

    def _rasterise(self, positions, indices, node_attr) -> None:
        """Barycentric point-sampling of every triangle at sub-voxel spacing."""
        a = positions[indices[:, 0]]
        b = positions[indices[:, 1]]
        c = positions[indices[:, 2]]
        nodes = node_attr[indices[:, 0]] + 1

        edge1 = b - a
        edge2 = c - a
        face_normals = np.cross(edge1, edge2)
        face_normals /= np.maximum(np.linalg.norm(face_normals, axis=1, keepdims=True), 1e-9)
        longest = np.maximum(np.linalg.norm(edge1, axis=1), np.linalg.norm(edge2, axis=1))
        steps = np.clip(np.ceil(longest / (self.resolution * 0.6)).astype(int) + 1, 2, 96)

        for n in np.unique(steps):
            sel = steps == n
            u = np.linspace(0, 1, n)
            uu, vv = np.meshgrid(u, u, indexing="ij")
            keep = (uu + vv) <= 1.0
            uu = uu[keep][None, :]
            vv = vv[keep][None, :]
            pts = (a[sel][:, None, :] + uu[..., None] * edge1[sel][:, None, :]
                   + vv[..., None] * edge2[sel][:, None, :])
            flat = pts.reshape(-1, 3)
            owner = np.repeat(nodes[sel], uu.shape[1])
            normal = np.repeat(face_normals[sel], uu.shape[1], axis=0)
            self._stamp(flat, owner, normal)

    def _stamp(self, points: np.ndarray, owner: np.ndarray, normal: np.ndarray) -> None:
        ijk = np.floor((points - self.origin) / self.resolution).astype(np.int64)
        inside = np.all((ijk >= 0) & (ijk < self.dims), axis=1)
        if not inside.any():
            return
        ijk = ijk[inside]
        flat = (ijk[:, 2] * self.dims[1] + ijk[:, 1]) * self.dims[0] + ijk[:, 0]
        self.grid[flat] = owner[inside].astype(np.int32)
        self.normals[flat] = np.clip(normal[inside] * 127, -127, 127).astype(np.int8)

    def march(self, origin: np.ndarray, directions: np.ndarray, max_range: float,
              step_scale: float = 0.5, near: float = 0.6):
        """Step every ray in lockstep; return (hit_distance, node_slot).

        Distance is inf and slot -1 where a ray left the grid without hitting.
        """
        step = self.resolution * step_scale
        steps = int(np.ceil((max_range - near) / step))
        n = len(directions)
        hit_t = np.full(n, np.inf)
        hit_node = np.full(n, -1, dtype=np.int32)
        hit_normal = np.zeros((n, 3))
        active = np.ones(n, dtype=bool)

        for s in range(steps):
            if not active.any():
                break
            t = near + s * step
            pts = origin[None, :] + directions[active] * t
            ijk = np.floor((pts - self.origin) / self.resolution).astype(np.int64)
            inside = np.all((ijk >= 0) & (ijk < self.dims), axis=1)
            idx = np.flatnonzero(active)

            # Rays that left the grid are done.
            outside_idx = idx[~inside]
            if outside_idx.size:
                # Only retire once they are past the far side, not on entry.
                beyond = np.linalg.norm(pts[~inside] - self.origin - self.dims * self.resolution / 2,
                                        axis=1) > np.linalg.norm(self.dims * self.resolution)
                active[outside_idx[beyond]] = False

            if inside.any():
                ii = ijk[inside]
                flat = (ii[:, 2] * self.dims[1] + ii[:, 1]) * self.dims[0] + ii[:, 0]
                occupancy = self.grid[flat]
                struck = occupancy > 0
                if struck.any():
                    target = idx[inside][struck]
                    hit_t[target] = t
                    hit_node[target] = occupancy[struck] - 1
                    hit_normal[target] = self.normals[flat[struck]] / 127.0
                    active[target] = False
        return hit_t, hit_node, hit_normal


def build_scene(world: World, *, resolution: float = 0.25, backend: str = "auto"):
    """Pick a ray-casting backend. Exact where possible, numpy where not."""
    if backend in ("auto", "embree"):
        try:
            return EmbreeScene(world)
        except ImportError:
            if backend == "embree":
                raise ImportError(
                    "the embree backend needs Open3D: pip install 'lidarworld[exact]'") from None
    return VoxelScene(world, resolution)


def simulate(world: World, points: np.ndarray, sensor: np.ndarray, *,
             resolution: float = 0.25, tolerance: float = 0.35,
             max_rays: int = 40_000, max_range: float = 120.0,
             backend: str = "auto") -> Consistency:
    """Re-scan the reconstruction from `sensor` and score it against `points`.

    `points` must be observations made *from that sensor position* -- comparing
    a scan against a viewpoint it was not taken from measures nothing.
    """
    points = np.asarray(points, dtype=np.float64)
    sensor = np.asarray(sensor, dtype=np.float64)
    if len(points) > max_rays:
        points = points[:: int(np.ceil(len(points) / max_rays))]

    rel = points - sensor
    observed = np.linalg.norm(rel, axis=1)
    keep = observed > 1.0
    rel, observed = rel[keep], observed[keep]
    directions = rel / observed[:, None]

    scene = build_scene(world, resolution=resolution, backend=backend)
    simulated, hit_node, hit_normal = scene.march(sensor, directions, max_range)

    delta = simulated - observed
    finite = np.isfinite(simulated)
    explained = finite & (np.abs(delta) <= tolerance)
    # At near-tangent incidence a voxel scene cannot decide whether the beam
    # would have grazed past or struck: half a voxel of thickness moves the
    # range by many metres. Those rays are reported, not counted as errors.
    incidence = np.abs(np.einsum("ij,ij->i", directions, hit_normal))
    # With exact hits the tangent band can be tight; the voxel fallback needs a
    # wider one because half a cell of thickening moves the range by metres.
    grazing_cos = 0.08 if scene.backend == "embree" else 0.2
    grazing = finite & ~explained & (incidence < grazing_cos)
    early = finite & ~explained & ~grazing & (delta < -tolerance)
    late = finite & ~explained & ~grazing & (delta > tolerance)
    missing = ~finite

    rmse = float(np.sqrt(np.mean(delta[explained] ** 2))) if explained.any() else float("nan")

    node_names = mesh_slot_names(world)
    per_node: dict[str, dict] = {}
    for slot in np.unique(hit_node[hit_node >= 0]):
        sel = (hit_node == slot) & ~grazing
        name = node_names[slot] if slot < len(node_names) else f"slot{slot}"
        agree = int(explained[sel].sum())
        total = int(sel.sum())
        if total < 4:
            continue
        per_node[name] = {
            "rays": total,
            "explained": agree,
            "fraction": round(agree / max(total, 1), 3),
            "bias": round(float(np.median(delta[sel & finite])) if (sel & finite).any() else 0.0, 3),
        }

    return Consistency(
        rays=int(len(observed)), explained=int(explained.sum()), missing=int(missing.sum()),
        early=int(early.sum()), late=int(late.sum()), grazing=int(grazing.sum()),
        backend=scene.backend, rmse=rmse, tolerance=tolerance,
        sensor=sensor.tolist(), per_node=per_node)


def apply_to_world(world: World, report: Consistency, *, weight: float = 0.5) -> int:
    """Fold measured consistency back into node confidence.

    A node whose surfaces reproduce the scan earns confidence; one that does not
    loses it. The original estimate is kept in `attrs` so the adjustment is
    never silent.
    """
    updated = 0
    for node_id, stats in report.per_node.items():
        node = world.nodes.get(node_id)
        if node is None or stats["rays"] < 8:
            continue
        node.attrs["validation"] = stats
        prior = node.confidence
        node.attrs["confidence_prior"] = round(float(prior), 3)
        node.confidence = float(np.clip((1 - weight) * prior + weight * stats["fraction"], 0.05, 0.99))
        updated += 1
    world.notes["validation"] = report.to_json()
    return updated

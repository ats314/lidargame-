#!/usr/bin/env python3
"""Bake deterministic sample scans so the pipeline can be run with no downloads.

One synthetic town block is generated, then written out twice:

  townblock.las   airborne/mobile-mapping style tile with ASPRS class codes
  street.bin      a single street-level sweep of the same block, KITTI layout
  street.label    SemanticKITTI sidecar for that sweep
  room.pcd        a tiny ASCII PCD, used by the format round-trip tests

The street sweep is produced by keeping the nearest point per (azimuth,
elevation) bin from a sensor standing in the road, which reproduces the two
things that matter downstream: ring structure and occlusion shadows.

Windows are *absences*, not markers. Glass returns almost nothing at 905 nm, so
the generator simply does not emit points there -- which is exactly the signal
the opening detector looks for.

    python tools/make_sample_data.py --out data/samples
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lidarworld.ingest.kitti import write_kitti, write_labels  # noqa: E402
from lidarworld.ingest.las import write_las  # noqa: E402

# ASPRS codes
GROUND, VEG_LOW, VEG_HIGH, BUILDING, WATER, ROAD, TOWER, UNCLASSIFIED = 2, 3, 5, 6, 9, 11, 15, 1

#: ASPRS -> SemanticKITTI raw id, for the street sweep's label sidecar.
TO_KITTI = {GROUND: 72, VEG_LOW: 72, VEG_HIGH: 70, BUILDING: 50, WATER: 40,
            ROAD: 40, TOWER: 80, UNCLASSIFIED: 10}


class Scene:
    def __init__(self, seed: int = 7):
        self.rng = np.random.default_rng(seed)
        self.xyz: list[np.ndarray] = []
        self.intensity: list[np.ndarray] = []
        self.classification: list[np.ndarray] = []

    def add(self, xyz, intensity, cls):
        n = len(xyz)
        if n == 0:
            return
        self.xyz.append(np.asarray(xyz, dtype=np.float64))
        self.intensity.append(np.full(n, intensity, dtype=np.float32)
                              if np.isscalar(intensity) else np.asarray(intensity, np.float32))
        self.classification.append(np.full(n, cls, dtype=np.uint8))

    def finish(self, jitter=0.02, dropout=0.02):
        xyz = np.concatenate(self.xyz)
        intensity = np.concatenate(self.intensity)
        cls = np.concatenate(self.classification)
        xyz = xyz + self.rng.normal(0, jitter, xyz.shape)
        intensity = np.clip(intensity + self.rng.normal(0, 0.04, len(intensity)), 0, 1)
        keep = self.rng.random(len(xyz)) > dropout
        return xyz[keep], intensity[keep], cls[keep]


def sample_rect(rng, corner, edge_u, edge_v, density, holes=()):
    """Uniformly sample a parallelogram, skipping rectangular holes in (u,v)."""
    area = np.linalg.norm(edge_u) * np.linalg.norm(edge_v)
    n = max(4, int(area * density))
    u = rng.random(n)
    v = rng.random(n)
    if holes:
        lu, lv = np.linalg.norm(edge_u), np.linalg.norm(edge_v)
        keep = np.ones(n, bool)
        for (u0, v0, u1, v1) in holes:
            keep &= ~((u * lu > u0) & (u * lu < u1) & (v * lv > v0) & (v * lv < v1))
        u, v = u[keep], v[keep]
    return np.asarray(corner) + u[:, None] * np.asarray(edge_u) + v[:, None] * np.asarray(edge_v)


def window_holes(rng, width, height, *, sill=1.1, win_w=1.25, win_h=1.55,
                 spacing=3.1, floor_height=3.2, door=True):
    """Lay out a regular window grid, plus a door on the ground floor."""
    holes = []
    columns = max(1, int((width - 1.4) // spacing))
    floors = max(1, int((height - 1.0) // floor_height))
    margin = (width - columns * spacing) / 2 + (spacing - win_w) / 2
    for f in range(floors):
        base = sill + f * floor_height
        if base + win_h > height - 0.4:
            break
        for c in range(columns):
            u0 = margin + c * spacing
            if rng.random() < 0.08:            # a few boarded-up openings
                continue
            holes.append((u0, base, u0 + win_w, base + win_h))
    if door and columns >= 1:
        u0 = margin + (columns // 2) * spacing - 0.1
        holes.append((u0, 0.0, u0 + 1.15, 2.15))
    return holes


def add_building(scene, x0, y0, w, d, height, *, pitched=False, density=22.0, ground_z=0.0):
    rng = scene.rng
    corners = [
        (np.array([x0, y0, ground_z]), np.array([w, 0, 0]), np.array([0, 0, height])),
        (np.array([x0 + w, y0, ground_z]), np.array([0, d, 0]), np.array([0, 0, height])),
        (np.array([x0 + w, y0 + d, ground_z]), np.array([-w, 0, 0]), np.array([0, 0, height])),
        (np.array([x0, y0 + d, ground_z]), np.array([0, -d, 0]), np.array([0, 0, height])),
    ]
    for corner, edge_u, edge_v in corners:
        span = float(np.linalg.norm(edge_u))
        holes = window_holes(rng, span, height, door=rng.random() < 0.5)
        pts = sample_rect(rng, corner, edge_u, edge_v, density, holes)
        scene.add(pts, 0.42, BUILDING)

    if pitched:
        ridge_h = height + min(w, d) * 0.32
        mid = y0 + d / 2
        for sign in (1, -1):
            base_y = y0 if sign > 0 else y0 + d
            scene.add(sample_rect(
                rng, np.array([x0, base_y, ground_z + height]),
                np.array([w, 0, 0]), np.array([0, sign * d / 2, ridge_h - height]),
                density * 0.9), 0.5, BUILDING)
        del mid
    else:
        scene.add(sample_rect(rng, np.array([x0, y0, ground_z + height]),
                              np.array([w, 0, 0]), np.array([0, d, 0]), density * 0.8),
                  0.5, BUILDING)
        # parapet ring
        for corner, edge_u, edge_v in corners:
            scene.add(sample_rect(rng, corner + np.array([0, 0, height]), edge_u,
                                  np.array([0, 0, 0.55]), density * 0.7), 0.45, BUILDING)


def add_tree(scene, x, y, z, height=8.0, radius=2.8, points=520):
    rng = scene.rng
    trunk_h = height * 0.38
    t = rng.random(points // 6)
    scene.add(np.column_stack([
        x + rng.normal(0, 0.09, len(t)), y + rng.normal(0, 0.09, len(t)), z + t * trunk_h]),
        0.3, VEG_HIGH)
    n = points
    direction = rng.normal(size=(n, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    r = radius * rng.random(n) ** 0.34
    crown = np.column_stack([x + direction[:, 0] * r, y + direction[:, 1] * r,
                             z + trunk_h + (height - trunk_h) / 2
                             + direction[:, 2] * r * 0.95])
    scene.add(crown, 0.6, VEG_HIGH)


def add_pole(scene, x, y, z, height=7.5, points=140):
    rng = scene.rng
    t = rng.random(points)
    scene.add(np.column_stack([x + rng.normal(0, 0.05, points),
                               y + rng.normal(0, 0.05, points), z + t * height]),
              0.55, TOWER)


def add_vehicle(scene, x, y, z, yaw=0.0, points=380):
    rng = scene.rng
    length, width, height = 4.4, 1.85, 1.5
    u = (rng.random(points) - 0.5) * length
    v = (rng.random(points) - 0.5) * width
    shell = 1 - (2 * u / length) ** 2 * 0.35
    w = z + height * np.clip(shell, 0.2, 1) * (0.55 + 0.45 * rng.random(points))
    c, s = np.cos(yaw), np.sin(yaw)
    scene.add(np.column_stack([x + u * c - v * s, y + u * s + v * c, w]), 0.35, UNCLASSIFIED)


def build_town(seed=7, size=62.0, density=9.0):
    scene = Scene(seed)
    rng = scene.rng

    # Terrain: gentle bowl so the DTM has something to do.
    n = int(size * size * density)
    gx = rng.random(n) * size
    gy = rng.random(n) * size
    gz = 0.35 * np.sin(gx / 17.0) + 0.28 * np.cos(gy / 21.0)

    road_a = (np.abs(gy - size * 0.5) < 5.0)
    road_b = (np.abs(gx - size * 0.42) < 4.5)
    on_road = road_a | road_b
    scene.add(np.column_stack([gx[on_road], gy[on_road], gz[on_road] - 0.06]), 0.12, ROAD)
    scene.add(np.column_stack([gx[~on_road], gy[~on_road], gz[~on_road]]), 0.34, GROUND)

    def terrain_z(x, y):
        return 0.35 * np.sin(x / 17.0) + 0.28 * np.cos(y / 21.0)

    blocks = [
        (5.0, 5.0, 14.0, 13.0, 11.0, False),
        (23.0, 6.0, 11.0, 12.0, 8.0, True),
        (40.0, 5.5, 15.0, 14.0, 14.5, False),
        (6.0, 33.0, 12.0, 15.0, 9.5, True),
        (22.0, 34.0, 10.0, 11.0, 7.0, False),
        (38.0, 32.0, 16.0, 17.0, 12.0, False),
    ]
    for x0, y0, w, d, h, pitched in blocks:
        add_building(scene, x0, y0, w, d, h, pitched=pitched,
                     ground_z=float(terrain_z(x0 + w / 2, y0 + d / 2)))

    for x, y in [(19.5, 27.5), (19.5, 14.0), (19.5, 44.0), (36.0, 27.5),
                 (52.0, 27.0), (10.0, 27.0), (30.0, 52.0), (48.0, 50.0)]:
        add_tree(scene, x, y, float(terrain_z(x, y)),
                 height=float(rng.uniform(6.0, 11.0)), radius=float(rng.uniform(2.2, 3.6)))

    for x, y in [(17.0, 24.0), (17.0, 36.0), (37.5, 24.0), (37.5, 36.0), (56.0, 30.0)]:
        add_pole(scene, x, y, float(terrain_z(x, y)))

    for x, y, yaw in [(24.0, 29.5, 0.0), (33.0, 32.0, np.pi), (44.0, 29.0, 0.05),
                      (18.5, 40.0, 1.55), (18.5, 18.0, 1.58)]:
        add_vehicle(scene, x, y, float(terrain_z(x, y)), yaw=yaw)

    return scene.finish()


def simulate_sweep(xyz, intensity, cls, origin, *, beams=64, vfov=(-24.9, 2.0),
                   az_bins=1800, max_range=80.0):
    """Nearest return per (azimuth, elevation) bin -- ring structure + occlusion."""
    rel = xyz - np.asarray(origin)
    dist = np.linalg.norm(rel, axis=1)
    keep = (dist > 1.2) & (dist < max_range)
    rel, dist = rel[keep], dist[keep]
    idx = np.flatnonzero(keep)

    azimuth = np.arctan2(rel[:, 1], rel[:, 0])
    elevation = np.degrees(np.arcsin(np.clip(rel[:, 2] / dist, -1, 1)))
    ring = np.round((elevation - vfov[0]) / (vfov[1] - vfov[0]) * (beams - 1)).astype(int)
    inside = (ring >= 0) & (ring < beams)
    az_bin = ((azimuth + np.pi) / (2 * np.pi) * az_bins).astype(int) % az_bins
    key = ring * az_bins + az_bin

    rel, dist, idx, key, inside = rel[inside], dist[inside], idx[inside], key[inside], inside[inside]
    order = np.lexsort((dist, key))
    key_sorted = key[order]
    first = np.ones(len(order), bool)
    first[1:] = key_sorted[1:] != key_sorted[:-1]
    chosen = order[first]

    return rel[chosen], intensity[idx[chosen]], cls[idx[chosen]]


def write_room_pcd(path: Path, seed: int = 3) -> Path:
    """Small ASCII PCD: a corner of a room with a window-shaped hole."""
    rng = np.random.default_rng(seed)
    parts = []
    parts.append(sample_rect(rng, [0, 0, 0], [4, 0, 0], [0, 4, 0], 30))          # floor
    parts.append(sample_rect(rng, [0, 0, 0], [4, 0, 0], [0, 0, 2.6], 30,
                             holes=[(1.2, 0.9, 2.6, 2.0)]))                      # wall + window
    parts.append(sample_rect(rng, [0, 0, 0], [0, 4, 0], [0, 0, 2.6], 30))         # side wall
    xyz = np.concatenate(parts) + rng.normal(0, 0.008, (sum(len(p) for p in parts), 3))
    intensity = np.clip(rng.normal(0.4, 0.08, len(xyz)), 0, 1)
    lines = [
        "# .PCD v0.7 - Point Cloud Data file format",
        "VERSION 0.7", "FIELDS x y z intensity", "SIZE 4 4 4 4", "TYPE F F F F",
        "COUNT 1 1 1 1", f"WIDTH {len(xyz)}", "HEIGHT 1", "VIEWPOINT 0 0 0 1 0 0 0",
        f"POINTS {len(xyz)}", "DATA ascii",
    ]
    lines += [f"{x:.3f} {y:.3f} {z:.3f} {i:.3f}" for (x, y, z), i in zip(xyz, intensity)]
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/samples")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--density", type=float, default=9.0,
                        help="ground points per square metre")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    xyz, intensity, cls = build_town(args.seed, density=args.density)
    las_path = write_las(out / "townblock.las", xyz, intensity, cls)
    print(f"{las_path}  {len(xyz):,} points  {las_path.stat().st_size / 1e6:.1f} MB")

    sensor = np.array([26.0, 31.0, float(0.35 * np.sin(26 / 17.0) + 0.28 * np.cos(31 / 21.0)) + 1.73])
    sweep_xyz, sweep_i, sweep_cls = simulate_sweep(xyz, intensity, cls, sensor)
    bin_path = write_kitti(out / "street.bin", sweep_xyz, sweep_i)
    kitti_ids = np.array([TO_KITTI.get(int(c), 0) for c in sweep_cls], dtype=np.uint32)
    label_path = write_labels(out / "street.label", kitti_ids)
    print(f"{bin_path}  {len(sweep_xyz):,} points (single sweep from {sensor.round(1).tolist()})")
    print(f"{label_path}  SemanticKITTI ids")

    pcd_path = write_room_pcd(out / "room.pcd")
    print(f"{pcd_path}  ascii PCD")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

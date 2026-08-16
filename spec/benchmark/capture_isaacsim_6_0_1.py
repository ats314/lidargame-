#!/usr/bin/env python3
"""Capture a multi-pose RTX LiDAR observation package in Isaac Sim 6.0.1.

Run with Isaac Sim's Python, e.g.:
  ./python.sh capture_isaacsim_6_0_1.py --usd scene.usda --poses lidar_poses.json --out scan.npz

Ground-truth object IDs are recorded only when available and are explicitly evaluation-only.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaacsim import SimulationApp


def quat_rotate(q, pts):
    import numpy as np
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)],
    ], dtype=np.float64)
    return pts @ R.T


def enum_name(x):
    s = str(x)
    return s.split(".")[-1].upper()


def gmo_points_sensor_frame(gmo):
    import numpy as np
    x = np.asarray(gmo.x, dtype=np.float64)
    y = np.asarray(gmo.y, dtype=np.float64)
    z = np.asarray(gmo.z, dtype=np.float64)
    if len(x) == 0:
        return np.empty((0, 3), dtype=np.float64)
    coords = enum_name(gmo.elementsCoordsType)
    if "SPHERICAL" in coords:
        az = np.deg2rad(x)
        el = np.deg2rad(y)
        r = z
        ce = np.cos(el)
        pts = np.column_stack([r * ce * np.cos(az), r * ce * np.sin(az), r * np.sin(el)])
    else:
        pts = np.column_stack([x, y, z])
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd", required=True)
    ap.add_argument("--poses", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--headless", action="store_true", default=True)
    ap.add_argument("--config", default="Example_Rotary")
    ap.add_argument("--scan-rate", type=float, default=10.0)
    ap.add_argument("--warmup-frames", type=int, default=20)
    ap.add_argument("--frames-per-pose", type=int, default=20,
                    help="Full application updates after moving to each pose; increase if scans are incomplete.")
    args = ap.parse_args()

    sim = SimulationApp({
        "headless": args.headless,
        "multi_gpu": False,
        "extra_args": [
            "--/rtx-transient/stableIds/enabled=true",
            "--/app/sensors/nv/lidar/outputBufferOnGPU=false",
        ],
    })

    import carb
    import numpy as np
    import omni.timeline
    import omni.usd
    from isaacsim.sensors.experimental.rtx import Lidar, LidarSensor, parse_generic_model_output_data

    ctx = omni.usd.get_context()
    if not ctx.open_stage(str(Path(args.usd).resolve())):
        sim.close()
        raise RuntimeError(f"could not open USD stage {args.usd}")
    for _ in range(10):
        sim.update()

    with open(args.poses, "r", encoding="utf-8") as f:
        pose_doc = json.load(f)
    poses = pose_doc["poses"]
    if not poses:
        raise ValueError("pose file contains no poses")

    p0 = poses[0]
    lidar = Lidar.create(
        path="/World/BenchmarkLidar",
        config=args.config,
        tick_rate=args.scan_rate,
        accumulate_outputs=True,
        aux_output_level="EXTRA",
        positions=np.asarray([p0["position"]], dtype=np.float32),
        orientations=np.asarray([p0["orientation_wxyz"]], dtype=np.float32),
        attributes={"omni:sensor:Core:scanRateBaseHz": args.scan_rate},
    )
    sensor = LidarSensor(lidar, annotators=["generic-model-output", "stable-id-map"])

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(args.warmup_frames):
        sim.update()

    all_pts = []
    all_pose_idx = []
    all_obj = []
    scan_meta = []

    for i, pose in enumerate(poses):
        pos = np.asarray([pose["position"]], dtype=np.float32)
        quat = np.asarray([pose["orientation_wxyz"]], dtype=np.float32)
        lidar.set_world_poses(positions=pos, orientations=quat)
        for _ in range(args.frames_per_pose):
            sim.update()

        data, info = sensor.get_data("generic-model-output")
        gmo = parse_generic_model_output_data(data)
        pts = gmo_points_sensor_frame(gmo)
        frame_name = enum_name(gmo.frameOfReference)

        # Standard benchmark wants world-frame points. Default GMO is SENSOR frame.
        if "WORLD" not in frame_name:
            pts = quat_rotate(pose["orientation_wxyz"], pts) + np.asarray(pose["position"], dtype=np.float64)

        valid = np.isfinite(pts).all(axis=1)
        pts = pts[valid]
        all_pts.append(pts.astype(np.float32))
        all_pose_idx.append(np.full(len(pts), i, dtype=np.int32))

        obj_ids = None
        try:
            # In 6.0.x, object IDs are exposed from FULL/EXTRA auxiliary data when stable IDs are enabled.
            # Python binding details may vary; preserve raw bytes if exposed.
            candidate = getattr(gmo, "objectId", None)
            if candidate is None and hasattr(gmo, "lidarAuxiliaryData"):
                candidate = getattr(gmo.lidarAuxiliaryData, "objectId", None)
            if candidate is not None:
                arr = np.asarray(candidate)
                if arr.size:
                    obj_ids = arr.copy()
        except Exception:
            obj_ids = None
        if obj_ids is not None:
            all_obj.append(obj_ids)

        scan_meta.append({
            "pose_index": i,
            "num_points": int(len(pts)),
            "frame_of_reference": frame_name,
            "coords_type": enum_name(gmo.elementsCoordsType),
            "frame_id": int(gmo.frameId),
            "timestamp_ns": int(gmo.timestampNs),
        })
        print(f"pose={i} points={len(pts)} frame={frame_name} coords={scan_meta[-1]['coords_type']}")

    timeline.stop()
    points = np.concatenate(all_pts, axis=0) if all_pts else np.empty((0,3), np.float32)
    pose_index = np.concatenate(all_pose_idx, axis=0) if all_pose_idx else np.empty((0,), np.int32)

    payload = {
        "points": points,
        "pose_index": pose_index,
        "poses_json": np.asarray(json.dumps(pose_doc)),
        "scan_meta_json": np.asarray(json.dumps(scan_meta)),
        "simulator": np.asarray("Isaac Sim 6.0.1"),
        "lidar_config": np.asarray(args.config),
        "ground_truth_metadata_policy": np.asarray("evaluation_only"),
    }
    # Object-id layout is intentionally not normalized here; if present, preserve as a side channel.
    if all_obj:
        try:
            payload["object_id_raw"] = np.concatenate(all_obj, axis=0)
        except Exception:
            payload["object_id_raw_json"] = np.asarray(json.dumps([np.asarray(x).tolist() for x in all_obj]))

    np.savez_compressed(args.out, **payload)
    print(f"WROTE {args.out} points={len(points)}")
    sim.close()


if __name__ == "__main__":
    main()

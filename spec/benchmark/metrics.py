from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


def _as3(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(3)


def geometry_aabb(entity: Dict[str, Any]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return a union AABB from SIR geometry that can be bounded analytically."""
    mins, maxs = [], []
    for g in entity.get("geometry", []):
        rep = g.get("representation")
        if rep == "bbox":
            mn = _as3(g["bbox"]["min"])
            mx = _as3(g["bbox"]["max"])
        elif rep == "primitive":
            p = g["primitive"]
            c = _as3(p["center"])
            typ = p["type"]
            if typ == "box":
                half = 0.5 * _as3(p["size"])
            elif typ == "sphere":
                half = np.full(3, float(p["radius"]))
            elif typ == "cylinder":
                r, h = float(p["radius"]), float(p["height"])
                axis = p.get("axis", "Z")
                half = np.array([r, r, r], dtype=np.float64)
                half[{"X": 0, "Y": 1, "Z": 2}[axis]] = h / 2.0
            elif typ == "plane":
                # For v0.1, a plane needs size to produce a finite AABB.
                if "size" not in p:
                    continue
                half = 0.5 * _as3(p["size"])
            else:
                continue
            mn, mx = c - half, c + half
        else:
            continue
        mins.append(mn)
        maxs.append(mx)
    if not mins:
        return None
    return np.min(np.stack(mins), axis=0), np.max(np.stack(maxs), axis=0)


def aabb_iou(a: Tuple[np.ndarray, np.ndarray], b: Tuple[np.ndarray, np.ndarray]) -> float:
    amin, amax = a
    bmin, bmax = b
    inter = np.maximum(0.0, np.minimum(amax, bmax) - np.maximum(amin, bmin))
    inter_v = float(np.prod(inter))
    va = float(np.prod(np.maximum(0.0, amax - amin)))
    vb = float(np.prod(np.maximum(0.0, bmax - bmin)))
    denom = va + vb - inter_v
    return 0.0 if denom <= 0 else inter_v / denom


def aabb_center_extent(a: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    mn, mx = a
    return (mn + mx) / 2.0, mx - mn


def match_entities(
    gt_doc: Dict[str, Any],
    pred_doc: Dict[str, Any],
    max_center_distance_m: float = 5.0,
    min_iou: float = 0.01,
) -> Dict[str, Any]:
    """Geometry-only Hungarian matching so semantic scoring is not biased by class labels."""
    gt = [(e, geometry_aabb(e)) for e in gt_doc.get("entities", [])]
    pr = [(e, geometry_aabb(e)) for e in pred_doc.get("entities", [])]
    gt = [(e, b) for e, b in gt if b is not None]
    pr = [(e, b) for e, b in pr if b is not None]
    if not gt or not pr:
        return {"pairs": [], "unmatched_gt": [e[0]["id"] for e in gt], "unmatched_pred": [e[0]["id"] for e in pr]}

    cost = np.full((len(gt), len(pr)), 1e6, dtype=np.float64)
    for i, (_, ga) in enumerate(gt):
        gc, ge = aabb_center_extent(ga)
        scale = max(float(np.linalg.norm(ge)), 1.0)
        for j, (_, pa) in enumerate(pr):
            pc, _ = aabb_center_extent(pa)
            dist = float(np.linalg.norm(gc - pc))
            iou = aabb_iou(ga, pa)
            if dist <= max_center_distance_m or iou >= min_iou:
                cost[i, j] = (1.0 - iou) + 0.25 * min(dist / scale, 4.0)

    rows, cols = linear_sum_assignment(cost)
    pairs = []
    used_g, used_p = set(), set()
    for i, j in zip(rows, cols):
        if cost[i, j] >= 1e5:
            continue
        ge, ga = gt[i]
        pe, pa = pr[j]
        gc, gx = aabb_center_extent(ga)
        pc, px = aabb_center_extent(pa)
        iou = aabb_iou(ga, pa)
        dist = float(np.linalg.norm(gc - pc))
        if dist > max_center_distance_m and iou < min_iou:
            continue
        pairs.append({
            "gt_id": ge["id"], "pred_id": pe["id"], "iou": iou,
            "center_distance_m": dist,
            "extent_abs_error_m": np.abs(gx - px).tolist(),
        })
        used_g.add(i); used_p.add(j)

    return {
        "pairs": pairs,
        "unmatched_gt": [gt[i][0]["id"] for i in range(len(gt)) if i not in used_g],
        "unmatched_pred": [pr[j][0]["id"] for j in range(len(pr)) if j not in used_p],
    }


def score_sir(gt_doc: Dict[str, Any], pred_doc: Dict[str, Any]) -> Dict[str, Any]:
    m = match_entities(gt_doc, pred_doc)
    gt_idx = {e["id"]: e for e in gt_doc.get("entities", [])}
    pr_idx = {e["id"]: e for e in pred_doc.get("entities", [])}
    pairs = m["pairs"]

    if pairs:
        mean_iou = float(np.mean([p["iou"] for p in pairs]))
        center_rmse = float(np.sqrt(np.mean([p["center_distance_m"] ** 2 for p in pairs])))
        extent_mae = float(np.mean([np.mean(p["extent_abs_error_m"]) for p in pairs]))
        class_acc = float(np.mean([gt_idx[p["gt_id"]]["class"] == pr_idx[p["pred_id"]]["class"] for p in pairs]))
        kind_acc = float(np.mean([gt_idx[p["gt_id"]]["kind"] == pr_idx[p["pred_id"]]["kind"] for p in pairs]))
        semres_acc = float(np.mean([
            gt_idx[p["gt_id"]]["semantic_resolution"] == pr_idx[p["pred_id"]]["semantic_resolution"] for p in pairs
        ]))
    else:
        mean_iou = center_rmse = extent_mae = class_acc = kind_acc = semres_acc = 0.0

    # Relation scoring after geometry-derived entity mapping.
    pred_to_gt = {p["pred_id"]: p["gt_id"] for p in pairs}
    gt_rel = {(r["type"], r["source"], r["target"]) for r in gt_doc.get("relations", [])}
    pred_rel = set()
    for r in pred_doc.get("relations", []):
        if r["source"] in pred_to_gt and r["target"] in pred_to_gt:
            pred_rel.add((r["type"], pred_to_gt[r["source"]], pred_to_gt[r["target"]]))
    tp = len(gt_rel & pred_rel)
    fp = len(pred_rel - gt_rel)
    fn = len(gt_rel - pred_rel)
    precision = tp / (tp + fp) if tp + fp else 1.0 if not gt_rel else 0.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    rel_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    # Confidence calibration: target is correct class AND IoU >= 0.5.
    briers = []
    for p in pairs:
        pe = pr_idx[p["pred_id"]]
        conf = float(pe.get("confidence", {}).get("overall", 0.5))
        correct = float(p["iou"] >= 0.5 and gt_idx[p["gt_id"]]["class"] == pe["class"])
        briers.append((conf - correct) ** 2)
    confidence_brier = float(np.mean(briers)) if briers else 1.0

    # Provenance consistency: derived analytic geometry marked observed without direct evidence is a violation.
    violations = 0
    analytic = 0
    for e in pred_doc.get("entities", []):
        reps = {g.get("representation") for g in e.get("geometry", [])}
        if reps & {"primitive", "mesh", "polygon", "solid", "bbox"}:
            analytic += 1
            if e.get("epistemic_state") == "observed":
                modes = {p.get("mode") for p in e.get("provenance", [])}
                for g in e.get("geometry", []):
                    modes |= {p.get("mode") for p in g.get("provenance", [])}
                if "sensor_observation" not in modes and "structured_import" not in modes:
                    violations += 1
    prov_rate = violations / analytic if analytic else 0.0

    gt_n = max(len(gt_doc.get("entities", [])), 1)
    pred_n = max(len(pred_doc.get("entities", [])), 1)
    entity_recall = len(pairs) / gt_n
    entity_precision = len(pairs) / pred_n

    return {
        "matching": m,
        "geometry": {
            "mean_aabb_iou": mean_iou,
            "centroid_rmse_m": center_rmse,
            "extent_mae_m": extent_mae,
            "entity_precision": entity_precision,
            "entity_recall": entity_recall,
        },
        "semantics": {
            "class_accuracy_matched": class_acc,
            "kind_accuracy_matched": kind_acc,
            "semantic_resolution_accuracy_matched": semres_acc,
        },
        "topology": {
            "relation_precision": precision,
            "relation_recall": recall,
            "relation_f1": rel_f1,
            "tp": tp, "fp": fp, "fn": fn,
        },
        "uncertainty": {"confidence_brier": confidence_brier},
        "provenance": {"analytic_observed_violation_rate": prov_rate, "violations": violations, "analytic_entities": analytic},
    }


def _load_points(path: str) -> np.ndarray:
    x = np.load(path)
    if isinstance(x, np.lib.npyio.NpzFile):
        if "points" not in x.files:
            raise KeyError(f"{path} must contain an array named 'points'")
        pts = x["points"]
    else:
        pts = x
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("point cloud must have shape (N,3)")
    return pts[np.isfinite(pts).all(axis=1)]


def point_cloud_closure(gt_points_path: str, pred_points_path: str, max_points: int = 250000) -> Dict[str, float]:
    a = _load_points(gt_points_path)
    b = _load_points(pred_points_path)
    if len(a) == 0 or len(b) == 0:
        raise ValueError("empty point cloud")
    # Deterministic thinning if required.
    if len(a) > max_points:
        a = a[np.linspace(0, len(a)-1, max_points, dtype=int)]
    if len(b) > max_points:
        b = b[np.linspace(0, len(b)-1, max_points, dtype=int)]
    ta, tb = cKDTree(a), cKDTree(b)
    d_ab = tb.query(a, k=1, workers=-1)[0]
    d_ba = ta.query(b, k=1, workers=-1)[0]
    both = np.concatenate([d_ab, d_ba])
    return {
        "chamfer_l1_m": float(0.5 * (d_ab.mean() + d_ba.mean())),
        "rmse_nn_m": float(np.sqrt(np.mean(both ** 2))),
        "p95_nn_m": float(np.quantile(both, 0.95)),
        "approx_hausdorff_m": float(np.max(both)),
        "coverage_0p05m": float(np.mean(both <= 0.05)),
        "coverage_0p10m": float(np.mean(both <= 0.10)),
        "coverage_0p25m": float(np.mean(both <= 0.25)),
        "gt_points": int(len(a)),
        "pred_points": int(len(b)),
    }

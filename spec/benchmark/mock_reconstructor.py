#!/usr/bin/env python3
"""Controlled perturbation baseline. This is NOT a reconstruction algorithm.

It exists only to prove that the benchmark responds to geometry, semantic,
relation, confidence, and provenance errors in the expected direction.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path


def perturb(doc, seed=7, shift_sigma=0.12, drop_relation_rate=0.12, semantic_error_rate=0.10):
    rng = random.Random(seed)
    out = copy.deepcopy(doc)
    idmap = {}
    for i, e in enumerate(out["entities"]):
        old = e["id"]
        new = f"pred_{i:03d}"
        idmap[old] = new
        e["id"] = new
        e["epistemic_state"] = "inferred"
        e["confidence"]["overall"] = 0.78
        e["confidence"]["geometry"] = 0.80
        e["confidence"]["semantics"] = 0.82
        e["confidence"]["topology"] = 0.75
        e["provenance"] = [{
            "mode":"geometric_inference",
            "source_ids":["obs_lidar_reference"],
            "algorithm":"mock_reconstructor",
            "version":"0.1.0",
            "parameters_hash":f"seed-{seed}"
        }]
        if rng.random() < semantic_error_rate:
            e["class"] = "UnknownStructure"
        for g in e.get("geometry", []):
            p = g.get("primitive")
            if p and "center" in p:
                p["center"] = [float(v + rng.gauss(0, shift_sigma)) for v in p["center"]]
            if p and p.get("type") == "box" and "size" in p:
                p["size"] = [max(0.02, float(v * (1 + rng.gauss(0, 0.02)))) for v in p["size"]]
            g["provenance"] = [{
                "mode":"geometric_inference",
                "source_ids":["obs_lidar_reference"],
                "algorithm":"mock_reconstructor",
                "version":"0.1.0"
            }]

    new_rel = []
    for i, r in enumerate(out["relations"]):
        if rng.random() < drop_relation_rate:
            continue
        r["id"] = f"pred_r_{i:03d}"
        r["source"] = idmap[r["source"]]
        r["target"] = idmap[r["target"]]
        r["confidence"] = 0.72
        r["provenance"] = [{"mode":"semantic_inference","source_ids":["obs_lidar_reference"],"algorithm":"mock_reconstructor","version":"0.1.0"}]
        new_rel.append(r)
    out["relations"] = new_rel
    out["metadata"] = {**out.get("metadata", {}), "warning":"mock perturbation baseline; not a reconstructor"}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    doc = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    pred = perturb(doc, seed=args.seed)
    Path(args.out).write_text(json.dumps(pred, indent=2)+"\n", encoding="utf-8")
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()

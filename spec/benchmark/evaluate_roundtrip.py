#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from sir.reference import load_json, validate_document
from benchmark.metrics import point_cloud_closure, score_sir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-sir", required=True)
    ap.add_argument("--pred-sir", required=True)
    ap.add_argument("--gt-points")
    ap.add_argument("--pred-points")
    ap.add_argument("--out")
    args = ap.parse_args()

    schema = ROOT / "schema" / "spatial_ir_v0_1.schema.json"
    gt = validate_document(args.gt_sir, schema)
    pred = validate_document(args.pred_sir, schema)
    result = {"sir": score_sir(gt, pred)}
    if bool(args.gt_points) != bool(args.pred_points):
        raise ValueError("provide both --gt-points and --pred-points, or neither")
    if args.gt_points:
        result["sensor_closure"] = point_cloud_closure(args.gt_points, args.pred_points)

    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

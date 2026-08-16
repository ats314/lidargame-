#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sir.reference import validate_document, validate_invariants, validate_schema, load_json
from benchmark.metrics import score_sir
from benchmark.mock_reconstructor import perturb


def main():
    schema = ROOT / "schema" / "spatial_ir_v0_1.schema.json"
    gt_path = ROOT / "examples" / "reference_scene.sir.json"
    gt = validate_document(gt_path, schema)

    exact = copy.deepcopy(gt)
    exact_score = score_sir(gt, exact)
    assert exact_score["geometry"]["mean_aabb_iou"] > 0.999999
    assert exact_score["semantics"]["class_accuracy_matched"] == 1.0
    assert exact_score["topology"]["relation_f1"] == 1.0

    pred = perturb(gt, seed=7)
    validate_schema(pred, schema)
    validate_invariants(pred)
    noisy = score_sir(gt, pred)
    assert noisy["geometry"]["mean_aabb_iou"] < exact_score["geometry"]["mean_aabb_iou"]
    assert noisy["topology"]["relation_f1"] <= exact_score["topology"]["relation_f1"]

    print("SIR v0.1 smoke test: PASS")
    print(json.dumps({"exact": exact_score, "perturbed": noisy}, indent=2))


if __name__ == "__main__":
    main()

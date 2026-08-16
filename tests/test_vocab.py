"""Label vocabularies and labelled PLY ingest.

Every benchmark numbered its classes independently and several ranges collide
outright -- DALES and Toronto-3D are both 0-8 and disagree about what 3 means.
Guessing wrong silently relabels an entire dataset, so the failure mode these
tests guard is a confident wrong answer, not a missing one.
"""
from __future__ import annotations

import numpy as np
import pytest

from lidarworld import ingest
from lidarworld.semantics.vocab import VOCABULARIES, coverage, detect
from lidarworld.types import SEMANTIC_CLASSES, SEMANTIC_INDEX

S = SEMANTIC_INDEX


def write_ply(path, xyz, labels=None, label_name="scalar_label"):
    lines = ["ply", "format ascii 1.0", f"element vertex {len(xyz)}",
             "property float x", "property float y", "property float z"]
    if labels is not None:
        lines.append(f"property int {label_name}")
    lines.append("end_header")
    for i, p in enumerate(xyz):
        row = f"{p[0]} {p[1]} {p[2]}"
        if labels is not None:
            row += f" {int(labels[i])}"
        lines.append(row)
    path.write_text("\n".join(lines) + "\n")
    return path


def test_every_vocabulary_maps_onto_the_canonical_class_list():
    assert VOCABULARIES
    for name, table in VOCABULARIES.items():
        assert table, name
        for code, label in table.items():
            assert isinstance(code, int), f"{name}: {code!r} is not an int id"
            assert label in SEMANTIC_CLASSES, f"{name}[{code}] = {label!r} is not canonical"


def test_the_airborne_and_street_vocabularies_disagree_about_id_three():
    """The exact collision that makes silent detection dangerous."""
    assert VOCABULARIES["dales"][3] == "vehicle"
    assert VOCABULARIES["toronto_3d"][3] == "vegetation_high"


def test_coverage_reports_how_much_a_vocabulary_explains():
    assert coverage("dales", np.arange(9)) == pytest.approx(1.0)
    assert coverage("dales", np.array([100, 101])) == 0.0
    assert coverage("dales", np.array([1, 2, 999])) == pytest.approx(2 / 3)
    assert coverage("dales", np.array([])) == 0.0


def test_detect_refuses_the_ambiguous_small_ranges():
    """0-8 could be DALES or Toronto-3D. Neither answer is defensible."""
    guess, _ = detect(np.arange(9))
    assert guess is None


def test_detect_recognises_a_distinctive_range():
    kitti_ids = np.array([0, 10, 30, 40, 50, 70, 80, 252])
    guess, score = detect(kitti_ids)
    assert guess == "semantickitti"
    assert score == pytest.approx(1.0)

    assert detect(np.array([]))[0] is None
    assert detect(np.array([900, 901, 902]))[0] is None


def test_ply_loads_labels_when_told_which_vocabulary(tmp_path):
    xyz = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], float)
    path = write_ply(tmp_path / "dales.ply", xyz, [1, 2, 8, 3])
    result = ingest.load(path, vocab="dales")
    assert result.cloud["semantic"].tolist() == [
        S["ground"], S["vegetation_high"], S["building"], S["vehicle"]]
    assert result.cloud["source_class"].tolist() == [1, 2, 8, 3]
    assert "dales" in result.source.notes


def test_the_same_file_read_as_toronto_means_something_else(tmp_path):
    """Same integers, different dataset, different world. Hence no guessing."""
    xyz = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], float)
    path = write_ply(tmp_path / "either.ply", xyz, [1, 2, 8, 3])
    dales = ingest.load(path, vocab="dales").cloud["semantic"]
    toronto = ingest.load(path, vocab="toronto_3d").cloud["semantic"]
    assert dales.tolist() != toronto.tolist()
    assert toronto.tolist()[3] == S["vegetation_high"]


def test_an_ambiguous_labelled_ply_is_left_for_inference(tmp_path):
    xyz = np.zeros((9, 3))
    xyz[:, 0] = np.arange(9)
    path = write_ply(tmp_path / "unknown.ply", xyz, np.arange(9))
    result = ingest.load(path)
    # Raw ids are preserved so nothing is lost, but no semantics are claimed.
    assert result.cloud["source_class"].tolist() == list(range(9))
    assert "semantic" not in result.cloud
    assert "--vocab" in result.source.notes


def test_an_unknown_vocabulary_name_is_an_error(tmp_path):
    xyz = np.zeros((3, 3))
    path = write_ply(tmp_path / "x.ply", xyz, [1, 2, 3])
    with pytest.raises(ValueError, match="unknown label vocabulary"):
        ingest.load(path, vocab="not_a_dataset")


def test_a_wrong_vocabulary_says_so_rather_than_silently_mismapping(tmp_path):
    xyz = np.zeros((3, 3))
    xyz[:, 0] = np.arange(3)
    path = write_ply(tmp_path / "kitti_ids.ply", xyz, [40, 50, 70])
    result = ingest.load(path, vocab="dales")
    assert "only 0%" in result.source.notes


@pytest.mark.parametrize("column", ["label", "class", "scalar_label", "semantic"])
def test_the_usual_label_column_names_are_all_recognised(tmp_path, column):
    xyz = np.zeros((2, 3))
    xyz[:, 0] = [0.0, 1.0]
    path = write_ply(tmp_path / f"{column}.ply", xyz, [1, 8], label_name=column)
    result = ingest.load(path, vocab="dales")
    assert result.cloud["semantic"].tolist() == [S["ground"], S["building"]]


def test_an_unlabelled_ply_still_loads(tmp_path):
    xyz = np.random.default_rng(0).random((20, 3))
    result = ingest.load(write_ply(tmp_path / "plain.ply", xyz))
    assert len(result.cloud) == 20
    assert "semantic" not in result.cloud
    assert "unlabelled" in result.source.notes

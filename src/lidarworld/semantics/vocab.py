"""Label vocabularies for the public annotated LiDAR datasets.

Every benchmark invented its own class list, so a file full of integers means
nothing without knowing which one produced it. These tables map each dataset's
raw ids onto the compiler's canonical semantics, which is the only vocabulary
anything downstream sees.

Mapping is lossy on purpose and in one direction only. The canonical list is
deliberately small -- it exists to drive reconstruction, not to win a
segmentation benchmark -- so `bus`, `truck` and `car` all land on `vehicle`,
and a dataset's finer distinctions are dropped rather than smuggled through.
Where a dataset's class has no canonical home (road markings, trash cans) it
maps to the nearest structural thing, or to `unclassified` when there is none.

Registry is `VOCABULARIES`. `detect()` guesses from the id range when the
caller has not said which dataset a file came from, and says so when it cannot.
"""
from __future__ import annotations

import numpy as np

#: DALES: aerial LiDAR, 8 classes, 505M points over 10 km2.
#: The only *airborne* benchmark here, which makes it the one that can measure
#: this compiler's own semantic inference rather than a car's.
DALES = {
    0: "unclassified",
    1: "ground",
    2: "vegetation_high",
    3: "vehicle",          # cars
    4: "vehicle",          # trucks
    5: "wire",             # power lines
    6: "fence",
    7: "pole",
    8: "building",
}

#: Toronto-3D: mobile laser scanning along Avenue Road, 8 classes.
TORONTO_3D = {
    0: "unclassified",
    1: "road",
    2: "road",             # road markings -- same surface, different paint
    3: "vegetation_high",  # "natural", i.e. trees
    4: "building",
    5: "wire",             # utility lines
    6: "pole",
    7: "vehicle",
    8: "fence",
}

#: Paris-Lille-3D: mobile laser scanning, coarse class level.
PARIS_LILLE_3D = {
    0: "unclassified",
    1: "ground",
    2: "building",
    3: "pole",
    4: "pole",             # bollards are short poles
    5: "unclassified",     # trash cans
    6: "fence",            # barriers
    7: "person",
    8: "vehicle",
    9: "vegetation_high",  # "natural"
}

#: nuScenes-lidarseg raw ids (0-31) before the usual 16-class merge.
NUSCENES = {
    0: "noise", 1: "unclassified",
    2: "person", 3: "person", 4: "person", 5: "person", 6: "person",
    7: "person", 8: "person",
    9: "fence",            # movable barrier
    10: "unclassified", 11: "unclassified",
    12: "pole",            # traffic cone
    13: "unclassified",
    14: "vehicle", 15: "vehicle", 16: "vehicle", 17: "vehicle", 18: "vehicle",
    19: "vehicle", 20: "vehicle", 21: "vehicle", 22: "vehicle", 23: "vehicle",
    24: "road",            # driveable surface
    25: "ground",          # other flat
    26: "road",            # sidewalk
    27: "ground",          # terrain
    28: "building",        # static manmade
    29: "unclassified",
    30: "vegetation_high",
    31: "noise",           # ego vehicle returns
}

#: SemanticKITTI / KITTI, shared with the .bin adapter.
SEMANTIC_KITTI = {
    0: "unclassified", 1: "noise",
    10: "vehicle", 11: "vehicle", 13: "vehicle", 15: "vehicle", 16: "vehicle",
    18: "vehicle", 20: "vehicle",
    30: "person", 31: "person", 32: "person",
    40: "road", 44: "road", 48: "road", 49: "ground",
    50: "building", 51: "fence", 52: "unclassified",
    60: "road", 70: "vegetation_high", 71: "vegetation_high", 72: "vegetation_low",
    80: "pole", 81: "unclassified", 99: "unclassified",
    252: "vehicle", 253: "person", 254: "person", 255: "vehicle",
    256: "vehicle", 257: "vehicle", 258: "vehicle", 259: "vehicle",
}

#: ASPRS standard codes, shared with the LAS adapter.
ASPRS = {
    2: "ground", 3: "vegetation_low", 4: "vegetation_low", 5: "vegetation_high",
    6: "building", 7: "noise", 9: "water", 10: "ground", 11: "road",
    13: "wire", 14: "wire", 15: "pole", 16: "wire", 17: "bridge", 18: "noise",
}

VOCABULARIES: dict[str, dict[int, str]] = {
    "dales": DALES,
    "toronto_3d": TORONTO_3D,
    "paris_lille_3d": PARIS_LILLE_3D,
    "nuscenes": NUSCENES,
    "semantickitti": SEMANTIC_KITTI,
    "asprs": ASPRS,
}


def coverage(vocab: str, ids: np.ndarray) -> float:
    """Fraction of `ids` this vocabulary has a mapping for. 1.0 is a good sign."""
    table = VOCABULARIES[vocab]
    known = np.fromiter(table.keys(), dtype=np.int64)
    return float(np.isin(np.asarray(ids, dtype=np.int64), known).mean()) if len(ids) else 0.0


def detect(ids: np.ndarray) -> tuple[str | None, float]:
    """Guess the vocabulary from the observed ids. Returns (name, coverage).

    Ranges overlap -- a file whose labels are all 0-8 could be DALES or
    Toronto-3D and no amount of counting separates them -- so this is a
    convenience for the unambiguous cases, not a substitute for being told.
    Callers should treat a tie as unknown and require an explicit vocabulary.
    """
    ids = np.asarray(ids)
    if not len(ids):
        return None, 0.0
    unique = np.unique(ids)
    scored = sorted(((coverage(name, unique), name) for name in VOCABULARIES), reverse=True)
    best_score, best_name = scored[0]
    if best_score < 0.9:
        return None, best_score
    # Ambiguous when several vocabularies explain the ids equally well.
    tied = [name for score, name in scored if score >= best_score - 1e-9]
    if len(tied) > 1 and int(unique.max()) <= 9:
        return None, best_score
    return best_name, best_score

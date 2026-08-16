"""Semantic classification: source labels where available, inference where not.

The label vocabularies live in `vocab` -- one table per public dataset, mapping
that benchmark's ids onto the canonical class list. The ingest adapters import
them from here rather than the other way round, so nothing in `ingest` owns a
class table of its own.
"""
from . import infer, vocab  # noqa: F401
from .infer import class_histogram  # noqa: F401
from .vocab import (ASPRS, DALES, NUSCENES, PARIS_LILLE_3D, SEMANTIC_KITTI,  # noqa: F401
                    TORONTO_3D, VOCABULARIES, coverage, detect)

__all__ = ["infer", "vocab", "class_histogram", "VOCABULARIES", "coverage", "detect",
           "ASPRS", "SEMANTIC_KITTI", "DALES", "TORONTO_3D", "PARIS_LILLE_3D", "NUSCENES"]

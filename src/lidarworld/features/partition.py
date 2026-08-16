"""Pluggable partitioner: points -> segments with geometric descriptors.

The compiler's own partitioner is multiscale PCA over a sparse voxel grid
(`features/neighborhood.py`). It is honest numpy, it runs anywhere, and it is
a weaker version of something that already exists: Superpoint Transformer and
its EZ-SP partitioner compute the same segment features -- planarity,
linearity, scattering, verticality, curvature, normals -- but produce a
*hierarchical* superpoint graph rather than flat per-point values, and do it on
GPU at roughly two orders of magnitude more throughput.

Rather than reimplement that, this is the seam it plugs into. A backend takes a
cloud and returns segment ids plus per-point descriptors; everything downstream
(roles, segmentation, lattices) consumes that contract and does not care which
backend produced it.

Backends:
  ``voxel``  multiscale PCA, numpy only, the default. Works everywhere.
  ``spt``    Superpoint Transformer / EZ-SP. Needs torch + torch-geometric and
             a GPU to be worth it. NOT EXERCISED in this repository's CI --
             there is no GPU here, so the adapter is written against the
             published interface and marked unverified rather than claimed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..types import PointCloud

#: Descriptors every backend must produce, so downstream stages are portable.
REQUIRED = ("planarity", "linearity", "sphericity", "verticality", "curvature",
            "normal", "boundary", "crease_score", "corner_score", "surface_score")


@dataclass
class Partition:
    """Segment assignment plus the descriptors the segments were judged on."""
    segments: np.ndarray                  # (N,) segment id per point, -1 = none
    backend: str
    levels: int = 1                       # >1 when the backend is hierarchical
    parents: dict[int, np.ndarray] = field(default_factory=dict)
    stats: dict = field(default_factory=dict)

    @property
    def count(self) -> int:
        return int(self.segments.max()) + 1 if len(self.segments) else 0


def voxel_backend(cloud: PointCloud, *, scales=None, **_) -> Partition:
    """Multiscale PCA over a sparse voxel grid. numpy only."""
    from ..spatial.grid import build_voxel_index
    from . import neighborhood

    scales = scales or neighborhood.DEFAULT_SCALES
    neighborhood.compute(cloud, scales)
    index = build_voxel_index(cloud.xyz, float(scales[0]))
    return Partition(
        segments=index.point_voxel.astype(np.int64),
        backend="voxel",
        levels=1,
        stats={"scales": list(map(float, scales)), "segments": index.n_voxels},
    )


def spt_backend(cloud: PointCloud, *, device: str = "cuda", **_) -> Partition:
    """Superpoint Transformer / EZ-SP partitioning.

    Produces a hierarchical superpoint graph rather than flat voxels, which is
    strictly more information than the voxel backend: the hierarchy is the
    thing the world graph wants and currently has to infer.

    Unverified here -- this container has no GPU and CI does not install torch,
    so the adapter is written against the published interface and fails loudly
    rather than silently degrading.
    """
    try:
        import torch  # noqa: F401
        from src.datasets import instantiate_datamodule_transforms  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "the spt backend needs Superpoint Transformer installed and importable "
            "(torch, torch-geometric, and the SPT repository on PYTHONPATH). "
            "See https://github.com/drprojects/superpoint_transformer. "
            "Use backend='voxel' for the dependency-free path."
        ) from exc

    raise NotImplementedError(
        "The SPT adapter is a declared seam, not a working integration. It needs "
        "a GPU to be worth running and neither this container nor CI has one, so "
        "writing an unexercised implementation here would be claiming something "
        "that has never executed. The contract it must satisfy is Partition("
        "segments, backend='spt', levels=n, parents={level: array}) with the "
        "REQUIRED descriptors attached to the cloud.")


BACKENDS = {"voxel": voxel_backend, "spt": spt_backend}


def partition(cloud: PointCloud, backend: str = "voxel", **options) -> Partition:
    if backend not in BACKENDS:
        raise KeyError(f"unknown partition backend {backend!r}; have {sorted(BACKENDS)}")
    result = BACKENDS[backend](cloud, **options)
    missing = [name for name in REQUIRED if name not in cloud]
    if missing:
        raise RuntimeError(
            f"backend {backend!r} did not produce required descriptors: {missing}. "
            "Every backend must leave the cloud in the same state so downstream "
            "stages stay portable.")
    return result

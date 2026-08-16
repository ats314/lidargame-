"""Ingest adapter contract and registry.

Every public LiDAR source speaks a different dialect: airborne tiles arrive as
LAS/LAZ with ASPRS class codes in projected metre CRS, street-level scans as raw
float32 blobs with a sidecar label file in a sensor-local frame. An adapter's
job is to erase that difference -- positions in metres, a `semantic` channel
using :data:`lidarworld.types.SEMANTIC_CLASSES`, and a :class:`Source` record
carrying the licence so attribution survives all the way to the exported world.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from ..types import PointCloud, Source, SEMANTIC_INDEX


@dataclass
class IngestResult:
    cloud: PointCloud
    source: Source


AdapterFn = Callable[[Path, dict], IngestResult]

_ADAPTERS: dict[str, tuple[tuple[str, ...], AdapterFn, str]] = {}


def register(name: str, extensions: Iterable[str], description: str = ""):
    """Decorator registering an adapter for a set of file extensions."""
    def deco(fn: AdapterFn) -> AdapterFn:
        _ADAPTERS[name] = (tuple(e.lower() for e in extensions), fn, description)
        return fn
    return deco


def adapters() -> dict[str, tuple[tuple[str, ...], AdapterFn, str]]:
    return dict(_ADAPTERS)


def adapter_for(path: str | Path, name: str | None = None) -> tuple[str, AdapterFn]:
    path = Path(path)
    if name:
        if name not in _ADAPTERS:
            raise KeyError(f"unknown adapter {name!r}; have {sorted(_ADAPTERS)}")
        return name, _ADAPTERS[name][1]
    ext = path.suffix.lower()
    for adapter_name, (exts, fn, _) in _ADAPTERS.items():
        if ext in exts:
            return adapter_name, fn
    raise ValueError(
        f"no adapter handles {ext!r}. Registered: "
        + ", ".join(f"{n} ({'/'.join(e)})" for n, (e, _, _) in _ADAPTERS.items()))


def load(path: str | Path, adapter: str | None = None, **options) -> IngestResult:
    """Load any supported file into the canonical representation."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    name, fn = adapter_for(path, adapter)
    result = fn(path, options)
    result.source.adapter = name
    result.source.point_count = len(result.cloud)
    if not result.source.uri:
        result.source.uri = str(path)
    result.cloud.source_id = result.source.id
    return result


def semantic_array(n: int, fill: str = "unclassified") -> np.ndarray:
    return np.full(n, SEMANTIC_INDEX[fill], dtype=np.uint8)


def remap(codes: np.ndarray, table: dict[int, str], default: str = "unclassified") -> np.ndarray:
    """Vectorised source-code -> canonical-semantic-index mapping."""
    codes = np.asarray(codes)
    lut = np.full(int(codes.max(initial=0)) + 1, SEMANTIC_INDEX[default], dtype=np.uint8)
    for code, name in table.items():
        if code < lut.size:
            lut[code] = SEMANTIC_INDEX[name]
    return lut[np.clip(codes, 0, lut.size - 1)]

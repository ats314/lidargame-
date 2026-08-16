"""Ingest adapters. Importing this module registers every built-in adapter."""
from .base import IngestResult, adapter_for, adapters, load, register, remap  # noqa: F401
from . import las, kitti, generic  # noqa: F401  (registration side effects)

__all__ = ["load", "adapters", "adapter_for", "register", "remap", "IngestResult",
           "las", "kitti", "generic"]

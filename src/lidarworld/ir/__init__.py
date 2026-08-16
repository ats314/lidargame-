"""Spatial IR serialisation. See docs/SPATIAL_IR.md for the on-disk schema."""
from .reader import inspect, read_world  # noqa: F401
from .writer import write_world, write_world_dir  # noqa: F401

__all__ = ["write_world", "write_world_dir", "read_world", "inspect"]

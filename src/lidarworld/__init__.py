"""lidarworld -- compile LiDAR point clouds into themeable, playable worlds.

    from lidarworld import compile_world, Config
    world = compile_world("tile.las", Config(name="downtown"))
    world.summary()

The compiler turns dots into a **Spatial IR**: a hierarchical world graph of
semantic objects carrying geometry, topology, confidence and provenance -- and
deliberately carrying no materials at all. Themes bind to roles and context
flags afterwards, so one compile can be re-skinned into any era or style, and
any engine backend can consume the same IR.
"""
from .pipeline import Config, compile_world, load_sources
from .types import (Edge, Geometry, Node, PointCloud, SEMANTIC_CLASSES, Source,
                    World, SCHEMA_VERSION)

__version__ = "0.2.0"

__all__ = [
    "compile_world", "Config", "load_sources",
    "World", "Node", "Edge", "Geometry", "PointCloud", "Source",
    "SEMANTIC_CLASSES", "SCHEMA_VERSION", "__version__",
]


def __getattr__(name):
    """Lazy submodule access so `import lidarworld` stays cheap."""
    import importlib
    if name in {"ingest", "features", "semantics", "roles", "segment", "topology",
                "reconstruct", "themes", "backends", "ir", "spatial"}:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

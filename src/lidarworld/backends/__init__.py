"""Engine backends.

A backend consumes the Spatial IR and emits something a specific runtime can
open. Nothing upstream of here knows a backend exists, which is the point: the
compiler is the product, the engine is a target.

Built in:
  ``web``    geometry + per-vertex context for the bundled viewer (themes stay
             swappable at runtime)
  ``gltf``   materialised glTF 2.0 for Blender / Godot / Unity / Unreal / USD
             converters
  ``cityjson`` CityJSON 1.1 with CityGML boundary-surface semantics, for QGIS,
             FME, azul, ninja and the 3D BAG toolchain

Writing another (USD, Bevy, a native engine) means one function with this
signature; see docs/BACKENDS.md.
"""
from __future__ import annotations

from . import cityjson, gltf, web  # noqa: F401

BACKENDS = {
    "web": {
        "module": web,
        "needs_theme": False,
        "description": "viewer bundle: world.bin + world.json, themes applied at runtime",
    },
    "gltf": {
        "module": gltf,
        "needs_theme": True,
        "description": "glTF 2.0 with materials resolved from a theme pack",
    },
    "cityjson": {
        "module": cityjson,
        "needs_theme": False,
        "description": "CityJSON 1.1 (CityGML 3.0 encoding) for GIS tooling",
    },
}

__all__ = ["BACKENDS", "web", "gltf", "cityjson"]

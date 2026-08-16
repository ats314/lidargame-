"""Theme compilation: semantic material requests -> concrete materials.

The Spatial IR never names a texture. Surfaces carry a role and a context mask;
a theme pack maps those to materials; a resolver backend produces the actual
pixels. Swap the pack and the same world becomes a different era, with no stage
of the compiler re-run.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import png, procedural
from .pack import (PACK_DIR, ThemePack, available_packs, compile_runtime_table,
                   explain, load_pack, save_pack)
from .request import MaterialRequest, MaterialSpec, Rule

__all__ = ["ThemePack", "MaterialRequest", "MaterialSpec", "Rule", "load_pack",
           "save_pack", "available_packs", "compile_runtime_table", "explain",
           "compile_theme", "procedural", "png", "PACK_DIR"]


def compile_theme(pack: ThemePack, out_dir: str | Path, *, bake: bool = True,
                  channels: tuple[str, ...] = ("albedo", "normal", "orm")) -> dict:
    """Write a runtime-ready theme directory.

        <out_dir>/theme.json        rules + material table + texture refs
        <out_dir>/tex/<id>_*.png    baked PBR channels (procedural materials)

    Image-backed and engine-backed materials are passed through untouched: the
    resolver records where they came from, and the consumer fetches them.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = compile_runtime_table(pack)

    baked = 0
    for entry, spec in zip(table["materials"], pack.materials.values()):
        entry["textures"] = {}
        if not bake or spec.kind != "procedural":
            entry["provenance"] = {"kind": spec.kind, "source": spec.source,
                                   "license": spec.license, "baked": False}
            continue
        maps = procedural.bake(spec)
        for channel in channels:
            if channel not in maps:
                continue
            rel = f"tex/{spec.id}_{channel}.png"
            png.write(out_dir / rel, maps[channel])
            entry["textures"][channel] = rel
        entry["provenance"] = {
            "kind": "procedural", "generator": spec.generator, "params": spec.params,
            "source": spec.source, "license": spec.license, "baked": True,
        }
        baked += 1

    table["textureSize"] = procedural.SIZE
    (out_dir / "theme.json").write_text(json.dumps(table, indent=1))
    return {"theme": table["id"], "materials": len(table["materials"]),
            "rules": len(table["rules"]), "baked": baked, "dir": str(out_dir)}

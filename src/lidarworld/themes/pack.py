"""Theme packs and the material resolver.

A pack is data, not code: a list of rules over (role, context) and a table of
material specs. Resolution walks the rules most-specific-first and returns the
winner along with *why* it won, which is what makes an unexpected look
debuggable instead of mysterious.

Nothing here knows about a renderer. The compiled output is a small JSON table
that a viewer, an exporter or an engine importer can evaluate identically.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from ..roles.taxonomy import Ctx
from .request import MaterialRequest, MaterialSpec, Rule

PACK_DIR = Path(__file__).parent / "packs"


@dataclass
class ThemePack:
    id: str
    name: str
    description: str = ""
    era: str = ""
    author: str = ""
    license: str = "CC0-1.0"
    fallback: str = "default"
    rules: list[Rule] = field(default_factory=list)
    materials: dict[str, MaterialSpec] = field(default_factory=dict)
    environment: dict = field(default_factory=dict)

    def __post_init__(self):
        # Most specific first, ties broken by explicit priority then order.
        self._ordered = sorted(
            enumerate(self.rules),
            key=lambda pair: (-pair[1].priority, -pair[1].specificity, pair[0]))

    def resolve(self, request: MaterialRequest) -> tuple[MaterialSpec, Rule | None]:
        for _, rule in self._ordered:
            if rule.matches(request) and rule.material in self.materials:
                return self.materials[rule.material], rule
        fallback = self.materials.get(self.fallback)
        if fallback is None:
            fallback = next(iter(self.materials.values()), MaterialSpec(id="missing"))
        return fallback, None

    def to_json(self) -> dict:
        return {
            "id": self.id, "name": self.name, "description": self.description,
            "era": self.era, "author": self.author, "license": self.license,
            "fallback": self.fallback,
            "environment": self.environment,
            "materials": [m.to_json() for m in self.materials.values()],
            "rules": [r.to_json() for r in self.rules],
        }

    @staticmethod
    def from_json(d: dict) -> "ThemePack":
        materials = {m["id"]: MaterialSpec.from_json(m) for m in d.get("materials", [])}
        rules = [Rule.from_json(r) for r in d.get("rules", [])]
        return ThemePack(
            id=d["id"], name=d.get("name", d["id"]), description=d.get("description", ""),
            era=d.get("era", ""), author=d.get("author", ""), license=d.get("license", "unknown"),
            fallback=d.get("fallback", "default"), rules=rules, materials=materials,
            environment=d.get("environment", {}),
        )

    def validate(self) -> list[str]:
        problems = []
        for rule in self.rules:
            if rule.material not in self.materials:
                problems.append(f"rule -> unknown material {rule.material!r}")
            for group in (rule.ctx_all, rule.ctx_any, rule.ctx_none):
                for flag in group:
                    if flag not in Ctx.BY_NAME:
                        problems.append(f"rule {rule.material!r} -> unknown context flag {flag!r}")
        if self.fallback not in self.materials:
            problems.append(f"fallback material {self.fallback!r} is not defined")
        return problems


def load_pack(source: str | Path) -> ThemePack:
    """Load by pack id (built-in) or from a path to a JSON file."""
    path = Path(source)
    if not path.exists():
        candidate = PACK_DIR / f"{source}.json"
        if not candidate.exists():
            raise FileNotFoundError(
                f"no theme pack {source!r}; built-ins: {', '.join(available_packs())}")
        path = candidate
    pack = ThemePack.from_json(json.loads(path.read_text()))
    problems = pack.validate()
    if problems:
        raise ValueError(f"theme pack {pack.id!r} is invalid:\n  " + "\n  ".join(problems))
    return pack


def available_packs() -> list[str]:
    return sorted(p.stem for p in PACK_DIR.glob("*.json"))


def save_pack(pack: ThemePack, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(pack.to_json(), indent=1))
    return path


def compile_runtime_table(pack: ThemePack) -> dict:
    """Flatten a pack into the table a runtime evaluates per vertex.

    Rules keep their predicate form (roles and bitmasks), because the set of
    (role, context) pairs a world actually contains is not known here -- and
    resolving lazily in the viewer is what allows a theme swap with no
    geometry rebuild.
    """
    materials = list(pack.materials.values())
    index = {m.id: i for i, m in enumerate(materials)}
    rules = []
    for _, rule in pack._ordered:
        if rule.material not in index:
            continue
        rules.append({
            "role": rule.role,
            "all": Ctx.encode(rule.ctx_all),
            "any": Ctx.encode(rule.ctx_any),
            "none": Ctx.encode(rule.ctx_none),
            "semantic": rule.semantic or "",
            "material": index[rule.material],
            "note": rule.note,
        })
    return {
        "id": pack.id, "name": pack.name, "era": pack.era,
        "description": pack.description, "license": pack.license,
        "environment": pack.environment,
        "fallback": index.get(pack.fallback, 0),
        "materials": [m.to_json() for m in materials],
        "rules": rules,
    }


def explain(pack: ThemePack, requests: Iterable[MaterialRequest]) -> list[dict]:
    """Trace resolution for a set of requests -- the debugging surface."""
    out = []
    for request in requests:
        material, rule = pack.resolve(request)
        out.append({
            "request": request.describe(),
            "material": material.id,
            "matched_rule": rule.to_json() if rule else None,
            "source": material.source,
            "license": material.license,
        })
    return out

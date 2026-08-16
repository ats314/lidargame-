"""Material requests: asking for a surface by *meaning*, never by filename.

A reconstructed tile does not ask for ``brick_07.png``. It asks for
"a wall, on a convex corner, street-facing, in a Victorian era world" and lets
a resolver decide what satisfies that. Swapping the resolver swaps the entire
look of the world without touching a single vertex.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..roles.taxonomy import Ctx, role_matches


@dataclass(frozen=True)
class MaterialRequest:
    role: str
    context: int = 0
    semantic: str = "unclassified"
    tags: tuple[str, ...] = ()
    confidence: float = 1.0

    def has(self, *flags: str) -> bool:
        return all(self.context & Ctx.BY_NAME[f] for f in flags)

    def describe(self) -> str:
        ctx = ",".join(Ctx.decode(self.context)) or "-"
        return f"{self.role}[{ctx}]"


@dataclass
class Rule:
    """One line of a theme pack: a predicate over (role, context) -> material."""
    material: str
    role: str = "*"
    ctx_all: tuple[str, ...] = ()
    ctx_any: tuple[str, ...] = ()
    ctx_none: tuple[str, ...] = ()
    semantic: str | None = None
    priority: int = 0
    note: str = ""

    def __post_init__(self):
        # Unknown flags must not crash construction: a pack should be loadable
        # so that validate() can report every problem at once.
        self.unknown_flags = tuple(
            f for group in (self.ctx_all, self.ctx_any, self.ctx_none)
            for f in group if f not in Ctx.BY_NAME)
        known = lambda group: [f for f in group if f in Ctx.BY_NAME]  # noqa: E731
        self._all = Ctx.encode(known(self.ctx_all))
        self._any = Ctx.encode(known(self.ctx_any))
        self._none = Ctx.encode(known(self.ctx_none))
        # More specific rules win: each required flag is worth a point.
        self._specificity = (len(self.ctx_all) * 2 + len(self.ctx_any)
                             + len(self.ctx_none) + (2 if self.role != "*" else 0)
                             + self.role.count("."))

    @property
    def specificity(self) -> int:
        return self._specificity

    def matches(self, request: MaterialRequest) -> bool:
        if not role_matches(request.role, self.role):
            return False
        if self.semantic and request.semantic != self.semantic:
            return False
        ctx = request.context
        if self._all and (ctx & self._all) != self._all:
            return False
        if self._any and not (ctx & self._any):
            return False
        if self._none and (ctx & self._none):
            return False
        return True

    def to_json(self) -> dict[str, Any]:
        d: dict[str, Any] = {"material": self.material, "role": self.role}
        if self.ctx_all: d["ctx_all"] = list(self.ctx_all)
        if self.ctx_any: d["ctx_any"] = list(self.ctx_any)
        if self.ctx_none: d["ctx_none"] = list(self.ctx_none)
        if self.semantic: d["semantic"] = self.semantic
        if self.priority: d["priority"] = self.priority
        if self.note: d["note"] = self.note
        return d

    @staticmethod
    def from_json(d: dict) -> "Rule":
        return Rule(
            material=d["material"], role=d.get("role", "*"),
            ctx_all=tuple(d.get("ctx_all", ())), ctx_any=tuple(d.get("ctx_any", ())),
            ctx_none=tuple(d.get("ctx_none", ())), semantic=d.get("semantic"),
            priority=int(d.get("priority", 0)), note=d.get("note", ""),
        )


@dataclass
class MaterialSpec:
    """What a material *is*, independent of who provides it."""
    id: str
    kind: str = "procedural"            # procedural | image | engine
    generator: str = "plaster"
    params: dict[str, Any] = field(default_factory=dict)
    base_color: tuple[float, float, float] = (0.7, 0.7, 0.7)
    roughness: float = 0.85
    metallic: float = 0.0
    emissive: tuple[float, float, float] = (0.0, 0.0, 0.0)
    opacity: float = 1.0
    scale_m: float = 1.0                # world metres per texture tile
    license: str = "Proprietary - All Rights Reserved"
    source: str = "procedural (lidarworld)"
    era: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id, "kind": self.kind, "generator": self.generator,
            "params": self.params, "baseColor": list(self.base_color),
            "roughness": self.roughness, "metallic": self.metallic,
            "emissive": list(self.emissive), "opacity": self.opacity,
            "scale": self.scale_m, "license": self.license, "source": self.source,
            "era": list(self.era), "tags": list(self.tags),
        }

    @staticmethod
    def from_json(d: dict) -> "MaterialSpec":
        return MaterialSpec(
            id=d["id"], kind=d.get("kind", "procedural"),
            generator=d.get("generator", "plaster"), params=d.get("params", {}),
            base_color=tuple(d.get("baseColor", (0.7, 0.7, 0.7))),
            roughness=d.get("roughness", 0.85), metallic=d.get("metallic", 0.0),
            emissive=tuple(d.get("emissive", (0, 0, 0))), opacity=d.get("opacity", 1.0),
            scale_m=d.get("scale", 1.0), license=d.get("license", "unknown"),
            source=d.get("source", ""), era=tuple(d.get("era", ())), tags=tuple(d.get("tags", ())),
        )

"""World programs: keep the parameters, not just the geometry they produced.

The compiler already executes generative programs; it just discards them. A
building comes out of `Extrude(footprint, ground_z, eave_z)` -- a closed ring
and two scalars -- and what gets stored is the several thousand tiles that
executing it produced. The description was smaller than the output and was
thrown away at the door.

Keeping it makes three things possible that geometry alone cannot:

*Completion by execution.* A missing wall is not something to predict. If the
program says the building is an extrusion of a closed ring, the wall follows
from re-executing it. Prediction is only needed while the *program* is
uncertain, not once it is known.

*A complexity term.* `cost()` counts the parameters a program carries, which is
the `C(W)` in `argmin_W [D(O, S(E(W))) + λC(W)]`. Without it, a million
independent triangles reproduce the returns perfectly and win, which is the
degenerate answer that objective exists to rule out.

*An honest residual.* `Program.residual` records how much of the observation the
program failed to explain, measured -- not asserted. A program with a low
complexity and a low residual is a real finding. A program with a low
complexity and an unmeasured residual is a guess wearing a suit.

This module is deliberately one primitive wide. `Extrude` is the only generator
the compiler currently runs, and a grammar of speculative primitives that have
never executed would be exactly the rigour theatre this project is supposed to
avoid. Gable, hip, sweep and catenary belong here when something emits them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Program:
    """A generator plus its parameters, and what it cost to be wrong.

    ``kind``       which generator to execute.
    ``params``     everything needed to re-execute it, and nothing else.
    ``residual``   fraction of this program's own output the observations
                   contradict, or None when it has not been measured.
    """
    id: str
    kind: str
    params: dict = field(default_factory=dict)
    residual: float | None = None
    notes: str = ""

    @property
    def cost(self) -> int:
        """Free parameters, which is this program's contribution to C(W)."""
        return _count(self.params)

    def to_json(self) -> dict:
        out = {"id": self.id, "kind": self.kind, "cost": self.cost,
               "params": _jsonable(self.params)}
        if self.residual is not None:
            out["residual"] = round(float(self.residual), 4)
        if self.notes:
            out["notes"] = self.notes
        return out


def _count(value) -> int:
    """Scalars are one parameter; an (n, 2) ring is 2n. Strings are free."""
    if isinstance(value, dict):
        return sum(_count(v) for v in value.values())
    if isinstance(value, np.ndarray):
        return int(value.size)
    if isinstance(value, (list, tuple)):
        return sum(_count(v) for v in value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return 0
    return 1


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return np.round(value, 3).tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def extrusion(id: str, ring: np.ndarray, ground_z: float, eave_z: float,
              **extra) -> Program:
    """The one generator the compiler actually runs today."""
    ring = np.asarray(ring, dtype=float)[:, :2]
    return Program(
        id=id, kind="extrude",
        params={"footprint": ring, "ground_z": round(float(ground_z), 3),
                "eave_z": round(float(eave_z), 3), **extra},
    )


def execute(program: Program):
    """Re-run a program. Returns whatever its generator returns.

    This is the `E` in the objective, and the reason a program is worth storing:
    geometry lost to a crop, a decimation or a missing wall is one call away as
    long as the parameters survived.
    """
    if program.kind != "extrude":
        raise NotImplementedError(
            f"no generator for {program.kind!r}. Only 'extrude' is implemented, "
            "because it is the only one the compiler emits -- add a generator "
            "here when a stage actually produces that primitive.")
    from ..reconstruct.extrude import walls_from_footprint

    p = program.params
    return walls_from_footprint(np.asarray(p["footprint"], dtype=float),
                                float(p["ground_z"]), float(p["eave_z"]))


def summarise(programs) -> dict:
    """Complexity of a world's program set, against the geometry it stands for.

    `compression` is the honest headline: how many surface tiles one parameter
    accounts for. It is only meaningful next to the residual, which says how
    much of the observation those parameters failed to explain.
    """
    programs = list(programs)
    if not programs:
        return {"programs": 0, "parameters": 0}
    measured = [p.residual for p in programs if p.residual is not None]
    out = {
        "programs": len(programs),
        "parameters": sum(p.cost for p in programs),
        "kinds": sorted({p.kind for p in programs}),
        "measured": len(measured),
    }
    if measured:
        out["residual_mean"] = round(float(np.mean(measured)), 4)
        out["residual_max"] = round(float(np.max(measured)), 4)
    return out

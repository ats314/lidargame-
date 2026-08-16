#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_name(s: str) -> str:
    out = "".join(c if c.isalnum() or c == "_" else "_" for c in s)
    if not out or out[0].isdigit():
        out = "e_" + out
    return out


def set_common_metadata(prim, entity, geom=None):
    prim.SetCustomDataByKey("sir_id", entity["id"])
    prim.SetCustomDataByKey("sir_class", entity["class"])
    prim.SetCustomDataByKey("sir_kind", entity["kind"])
    prim.SetCustomDataByKey("sir_semantic_resolution", entity["semantic_resolution"])
    prim.SetCustomDataByKey("sir_epistemic_state", entity["epistemic_state"])
    if geom is not None:
        prim.SetCustomDataByKey("sir_geometry_id", geom["id"])
        prim.SetCustomDataByKey("sir_geometry_role", geom["role"])


def add_primitive(stage, path: str, entity, geom):
    p = geom["primitive"]
    typ = p["type"]
    center = Gf.Vec3d(*map(float, p["center"]))

    if typ == "box":
        obj = UsdGeom.Cube.Define(stage, path)
        obj.CreateSizeAttr(1.0)
        prim = obj.GetPrim()
        xf = UsdGeom.Xformable(prim)
        xf.AddTranslateOp().Set(center)
        xf.AddScaleOp().Set(Gf.Vec3f(*map(float, p["size"])))
    elif typ == "sphere":
        obj = UsdGeom.Sphere.Define(stage, path)
        obj.CreateRadiusAttr(float(p["radius"]))
        prim = obj.GetPrim()
        UsdGeom.Xformable(prim).AddTranslateOp().Set(center)
    elif typ == "cylinder":
        obj = UsdGeom.Cylinder.Define(stage, path)
        obj.CreateRadiusAttr(float(p["radius"]))
        obj.CreateHeightAttr(float(p["height"]))
        obj.CreateAxisAttr(p.get("axis", "Z"))
        prim = obj.GetPrim()
        UsdGeom.Xformable(prim).AddTranslateOp().Set(center)
    elif typ == "plane":
        if "size" not in p:
            raise ValueError(f"plane {geom['id']} requires size in v0.1 USD compiler")
        obj = UsdGeom.Cube.Define(stage, path)
        obj.CreateSizeAttr(1.0)
        prim = obj.GetPrim()
        xf = UsdGeom.Xformable(prim)
        xf.AddTranslateOp().Set(center)
        xf.AddScaleOp().Set(Gf.Vec3f(*map(float, p["size"])))
    else:
        raise ValueError(f"unsupported primitive type {typ}")

    set_common_metadata(prim, entity, geom)
    return prim


def compile_sir_to_usd(sir_path: str, out_path: str):
    doc = load(sir_path)
    stage = Usd.Stage.CreateNew(str(Path(out_path).resolve()))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    sir_root = UsdGeom.Xform.Define(stage, "/World/SIR").GetPrim()
    sir_root.SetCustomDataByKey("sir_version", doc["sir_version"])
    sir_root.SetCustomDataByKey("sir_world_id", doc["world"]["id"])

    entity_paths = {}
    for e in doc["entities"]:
        ep = f"/World/SIR/{safe_name(e['id'])}"
        eprim = UsdGeom.Xform.Define(stage, ep).GetPrim()
        set_common_metadata(eprim, e)
        entity_paths[e["id"]] = ep

        n_render = 0
        for g in e.get("geometry", []):
            if g.get("role") != "render":
                continue
            if g.get("representation") != "primitive":
                continue
            gp = f"{ep}/{safe_name(g['id'])}"
            add_primitive(stage, gp, e, g)
            n_render += 1
        eprim.SetCustomDataByKey("sir_render_geometry_count", n_render)

    # Preserve semantic graph in USD relationships for debugging/evaluation.
    by_source = {}
    for r in doc.get("relations", []):
        by_source.setdefault((r["source"], r["type"]), []).append(r["target"])
    for (source, rtype), targets in by_source.items():
        if source not in entity_paths:
            continue
        sprim = stage.GetPrimAtPath(entity_paths[source])
        rel = sprim.CreateRelationship(f"sir:rel:{safe_name(rtype)}", custom=True)
        rel.SetTargets([Sdf.Path(entity_paths[t]) for t in targets if t in entity_paths])

    stage.SetDefaultPrim(world)
    stage.GetRootLayer().Save()
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    compile_sir_to_usd(args.sir, args.out)
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()

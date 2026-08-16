from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def canonical_sha256(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_schema(doc: Dict[str, Any], schema_path: str | Path) -> None:
    try:
        import jsonschema
    except ImportError as e:
        raise RuntimeError("jsonschema is required for schema validation") from e
    schema = load_json(schema_path)
    jsonschema.Draft202012Validator(schema).validate(doc)


def index_entities(doc: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {e["id"]: e for e in doc.get("entities", [])}


def index_observations(doc: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {o["id"]: o for o in doc.get("observations", [])}


def validate_invariants(doc: Dict[str, Any]) -> None:
    entities = doc.get("entities", [])
    relations = doc.get("relations", [])
    observations = doc.get("observations", [])

    eids = [e["id"] for e in entities]
    rids = [r["id"] for r in relations]
    oids = [o["id"] for o in observations]

    if len(eids) != len(set(eids)):
        raise ValueError("entity ids must be unique")
    if len(rids) != len(set(rids)):
        raise ValueError("relation ids must be unique")
    if len(oids) != len(set(oids)):
        raise ValueError("observation ids must be unique")

    entity_set = set(eids)
    obs_set = set(oids)

    for r in relations:
        if r["source"] not in entity_set or r["target"] not in entity_set:
            raise ValueError(f"relation {r['id']} references a missing entity")

    internal_ids = entity_set | obs_set
    for e in entities:
        for p in e.get("provenance", []):
            for key in ("source_ids", "evidence_ids"):
                for ref in p.get(key, []):
                    # External URIs/identifiers are allowed when namespaced with ':' or '/'.
                    if ref not in internal_ids and ":" not in ref and "/" not in ref:
                        raise ValueError(f"entity {e['id']} provenance references missing internal id {ref}")
        # Epistemic consistency guard: analytic reconstruction geometry should not be called observed.
        if e.get("epistemic_state") == "observed":
            for g in e.get("geometry", []):
                if g.get("representation") in {"primitive", "mesh", "polygon", "solid", "bbox"}:
                    modes = {p.get("mode") for p in g.get("provenance", [])} | {p.get("mode") for p in e.get("provenance", [])}
                    if "sensor_observation" not in modes and "structured_import" not in modes:
                        raise ValueError(
                            f"entity {e['id']} marks derived analytic geometry as observed without sensor/import provenance"
                        )


def validate_document(doc_path: str | Path, schema_path: str | Path) -> Dict[str, Any]:
    doc = load_json(doc_path)
    validate_schema(doc, schema_path)
    validate_invariants(doc)
    return doc


def iter_geometry(doc: Dict[str, Any]) -> Iterable[Tuple[Dict[str, Any], Dict[str, Any]]]:
    for entity in doc.get("entities", []):
        for geom in entity.get("geometry", []):
            yield entity, geom

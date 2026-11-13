"""Schema utilities.

The output schema is *inferred from the gold annotations* during data prep and
saved to artifacts/schema.json, so field names are never hard-coded. Every other
component (training, eval, serving) loads that file.

A "struct" looks like:
    {
      "invoice":  {"type": "object", "keys": ["invoice_number", "invoice_date", ...]},
      "items":    {"type": "array",  "keys": ["description", "quantity", "total_price"]},
      "subtotal": {"type": "object", "keys": ["tax", "discount", "total"]},
    }
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

Struct = dict[str, dict[str, Any]]
DEFAULT_SCHEMA_PATH = Path("artifacts/schema.json")


def infer_struct(annotations: list[dict]) -> Struct:
    struct: Struct = {}
    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        for section, value in ann.items():
            if value is None:
                continue
            if isinstance(value, dict):
                entry = struct.setdefault(section, {"type": "object", "keys": []})
                keys = list(value.keys())
            elif isinstance(value, list):
                entry = struct.setdefault(section, {"type": "array", "keys": []})
                keys = [k for item in value if isinstance(item, dict) for k in item]
            else:
                entry = struct.setdefault(section, {"type": "scalar"})
                keys = []
            if "keys" in entry:
                for k in keys:
                    if k not in entry["keys"]:
                        entry["keys"].append(k)
    return struct


def save_struct(struct: Struct, path: Path | str = DEFAULT_SCHEMA_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(struct, indent=2))


def load_struct(path: Path | str = DEFAULT_SCHEMA_PATH) -> Struct:
    return json.loads(Path(path).read_text())


def skeleton(struct: Struct) -> dict:
    """Empty template shown to the model in the prompt."""
    out: dict[str, Any] = {}
    for section, spec in struct.items():
        if spec["type"] == "object":
            out[section] = {k: "" for k in spec["keys"]}
        elif spec["type"] == "array":
            out[section] = [{k: "" for k in spec["keys"]}]
        else:
            out[section] = ""
    return out


def _object_schema(keys: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {k: {"type": "string"} for k in keys},
        "required": list(keys),
        "additionalProperties": False,
    }


def json_schema(struct: Struct) -> dict:
    """JSON Schema used for guided (constrained) decoding in vLLM."""
    props: dict[str, Any] = {}
    for section, spec in struct.items():
        if spec["type"] == "object":
            props[section] = _object_schema(spec["keys"])
        elif spec["type"] == "array":
            props[section] = {"type": "array", "items": _object_schema(spec["keys"])}
        else:
            props[section] = {"type": "string"}
    return {
        "type": "object",
        "properties": props,
        "required": list(struct.keys()),
        "additionalProperties": False,
    }


def _to_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def conform(data: Any, struct: Struct) -> dict:
    """Force any dict into the exact schema: missing keys -> "", unknown keys dropped,
    all leaf values stringified."""
    data = data if isinstance(data, dict) else {}
    out: dict[str, Any] = {}
    for section, spec in struct.items():
        val = data.get(section)
        if spec["type"] == "object":
            val = val if isinstance(val, dict) else {}
            out[section] = {k: _to_str(val.get(k)) for k in spec["keys"]}
        elif spec["type"] == "array":
            items = val if isinstance(val, list) else []
            out[section] = [
                {k: _to_str(it.get(k)) for k in spec["keys"]}
                for it in items
                if isinstance(it, dict)
            ]
        else:
            out[section] = _to_str(val)
    return out


def flatten(data: dict, struct: Struct) -> list[tuple[str, str]]:
    """(path, raw_value) pairs. Array items share a path (e.g. "items.description"),
    so line items are compared as a multiset, independent of order."""
    pairs: list[tuple[str, str]] = []
    data = conform(data, struct)
    for section, spec in struct.items():
        val = data[section]
        if spec["type"] == "object":
            pairs += [(f"{section}.{k}", v) for k, v in val.items()]
        elif spec["type"] == "array":
            for item in val:
                pairs += [(f"{section}.{k}", v) for k, v in item.items()]
        else:
            pairs.append((section, val))
    return pairs

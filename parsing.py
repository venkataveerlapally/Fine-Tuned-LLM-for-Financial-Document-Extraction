"""Robustly pull a JSON object out of raw model output."""
from __future__ import annotations

import json
import re

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def extract_json(text: str | None) -> dict | None:
    if not text:
        return None
    t = _FENCE.sub("", text.strip())
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        start, end = t.find("{"), t.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            obj = json.loads(t[start : end + 1])
        except json.JSONDecodeError:
            return None
    return obj if isinstance(obj, dict) else None

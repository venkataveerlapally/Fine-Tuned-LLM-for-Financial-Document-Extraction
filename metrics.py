"""Field-level precision / recall / F1.

Each document becomes a multiset of (field_path, normalized_value) pairs for its
non-empty fields. Then:
  TP = pairs in both gold and prediction
  FP = predicted pairs not in gold   (wrong or hallucinated values)
  FN = gold pairs not in prediction  (missed or wrong values)
Micro-averaged over all documents. Invalid JSON counts as an empty prediction,
so every gold field becomes a false negative.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import datetime

from .schema import Struct, flatten

_NUM_RE = re.compile(r"^[-+]?[$€£]?\s*[-+]?[\d,]*\.?\d+\s*%?$")
_NUM_STRIP = re.compile(r"[$€£,%\s]")
_DATE_FORMATS = (
    "%m/%d/%Y", "%Y-%m-%d", "%m-%d-%Y", "%d.%m.%Y",
    "%b %d, %Y", "%B %d, %Y", "%d %b %Y", "%d %B %Y",
)


def normalize(value) -> str:
    """Make formatting differences not count as errors:
    '$1,250.0' == '1250.00', '03/14/2025' == '2025-03-14', case/whitespace ignored."""
    s = " ".join(str(value).split()).lower()
    if not s:
        return ""
    if _NUM_RE.match(s):
        try:
            num = float(_NUM_STRIP.sub("", s))
            return f"{num:.2f}"
        except ValueError:
            pass
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    return s.rstrip(".,;:")


def field_counter(data: dict | None, struct: Struct) -> Counter:
    if data is None:
        return Counter()
    c: Counter = Counter()
    for path, raw in flatten(data, struct):
        v = normalize(raw)
        if v:
            c[(path, v)] += 1
    return c


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}


def score(golds: list[dict], preds: list[dict | None], struct: Struct) -> dict:
    assert len(golds) == len(preds), "golds and preds must align"
    per = defaultdict(lambda: [0, 0, 0])  # path -> [tp, fp, fn]
    invalid = exact = 0
    for gold, pred in zip(golds, preds):
        if pred is None:
            invalid += 1
        gc, pc = field_counter(gold, struct), field_counter(pred, struct)
        if gc == pc:
            exact += 1
        for (path, _), n in (gc & pc).items():
            per[path][0] += n
        for (path, _), n in (pc - gc).items():
            per[path][1] += n
        for (path, _), n in (gc - pc).items():
            per[path][2] += n

    tp = sum(v[0] for v in per.values())
    fp = sum(v[1] for v in per.values())
    fn = sum(v[2] for v in per.values())
    n = len(golds)
    return {
        **_prf(tp, fp, fn),
        "n_docs": n,
        "json_valid_rate": round(1 - invalid / n, 4) if n else 0.0,
        "doc_exact_match": round(exact / n, 4) if n else 0.0,
        "per_field": {
            path: {**_prf(*v), "support": v[0] + v[2]} for path, v in sorted(per.items())
        },
    }

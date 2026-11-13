"""Step 1b (optional): grow the training set with teacher-labeled invoices.

Only ~1.5k invoices in the dataset have gold labels. To reach ~3,000 training
documents, this OCRs unlabeled invoices with Tesseract and labels them with a
larger "teacher" model behind any OpenAI-compatible endpoint (e.g. a vLLM server
running Qwen2.5-72B-Instruct-AWQ, or a hosted API). Labels are kept only if
they pass grounding checks (values must actually appear in the OCR text).

Pseudo-labels go into TRAINING ONLY. The test set stays 100% gold.

    # teacher on another GPU box:
    #   vllm serve Qwen/Qwen2.5-72B-Instruct-AWQ --max-model-len 8192
    python scripts/pseudo_label.py --endpoint http://TEACHER:8000/v1 \
        --model Qwen/Qwen2.5-72B-Instruct-AWQ --limit 1700
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ie.data import DEFAULT_REPO, load_records  # noqa: E402
from ie.metrics import normalize  # noqa: E402
from ie.parsing import extract_json  # noqa: E402
from ie.prompt import build_messages  # noqa: E402
from ie.schema import conform, flatten, json_schema, load_struct, skeleton  # noqa: E402


def grounded(data: dict, struct: dict, ocr: str, min_fill: float, min_grounded: float) -> bool:
    """Reject labels that are mostly empty or contain values not found in the OCR text."""
    pairs = flatten(data, struct)
    filled = [v for _, v in pairs if v.strip()]
    if not pairs or len(filled) / len(pairs) < min_fill:
        return False
    hay = " ".join(ocr.lower().split())
    hay_digits = "".join(ch for ch in hay if ch.isdigit() or ch == ".")
    hits = 0
    for v in filled:
        v_low = " ".join(v.lower().split())
        v_digits = "".join(ch for ch in v_low if ch.isdigit() or ch == ".")
        if v_low in hay or (v_digits and v_digits in hay_digits) or normalize(v) in hay:
            hits += 1
    return hits / len(filled) >= min_grounded


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--endpoint", required=True, help="OpenAI-compatible base URL, e.g. http://host:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=1700)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--gold-ids", default="data/gold_ids.txt")
    ap.add_argument("--out", default="data/pseudo.jsonl")
    ap.add_argument("--min-fill", type=float, default=0.6)
    ap.add_argument("--min-grounded", type=float, default=0.9)
    args = ap.parse_args()

    import pytesseract
    from openai import OpenAI

    struct = load_struct()
    skel, schema = skeleton(struct), json_schema(struct)
    gold_ids = set(Path(args.gold_ids).read_text().split()) if Path(args.gold_ids).exists() else set()

    records, get_image = load_records(args.repo)
    pool = [r for r in records if not r["annotation"] and r["id"] not in gold_ids][: args.limit]
    print(f"[pseudo] labeling {len(pool)} unlabeled invoices with {args.model}")

    client = OpenAI(base_url=args.endpoint, api_key=os.getenv("TEACHER_API_KEY", "EMPTY"))

    def work(rec: dict) -> dict | None:
        ocr = rec["ocr_text"].strip() or pytesseract.image_to_string(get_image(rec))
        if len(ocr.strip()) < 50:
            return None
        resp = client.chat.completions.create(
            model=args.model,
            messages=build_messages(ocr, skel),
            temperature=0,
            max_tokens=2048,
            response_format={"type": "json_schema",
                             "json_schema": {"name": "invoice", "schema": schema, "strict": True}},
        )
        data = extract_json(resp.choices[0].message.content)
        if data is None:
            return None
        data = conform(data, struct)
        if not grounded(data, struct, ocr, args.min_fill, args.min_grounded):
            return None
        return {"id": f"pseudo-{rec['id']}", "ocr_text": ocr, "target": data, "source": "pseudo"}

    kept = rejected = 0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f, ThreadPoolExecutor(args.concurrency) as ex:
        futures = [ex.submit(work, r) for r in pool]
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                row = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"[pseudo] error: {e}")
                row = None
            if row:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
            else:
                rejected += 1
            if i % 50 == 0:
                print(f"[pseudo] {i}/{len(pool)} kept={kept} rejected={rejected}")
    print(f"[pseudo] done: kept={kept} rejected={rejected} -> {args.out}")
    print("[pseudo] Spot-check ~50 rows by hand before training on them.")


if __name__ == "__main__":
    main()

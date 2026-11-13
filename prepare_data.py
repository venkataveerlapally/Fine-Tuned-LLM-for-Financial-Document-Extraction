"""Step 1: build train/val/test JSONL from the labeled invoices.

Output rows: {"id", "ocr_text", "target", "source"}
Also writes artifacts/schema.json (inferred from the gold labels).

The test split is gold-only and fixed by seed; never add pseudo-labels to it.

    python scripts/prepare_data.py --out data
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ie.data import DEFAULT_REPO, load_records  # noqa: E402
from ie.schema import conform, infer_struct, save_struct  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--out", default="data")
    ap.add_argument("--schema-out", default="artifacts/schema.json")
    ap.add_argument("--test-size", type=int, default=200)
    ap.add_argument("--val-size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    records, _ = load_records(args.repo)
    labeled = [r for r in records if r["annotation"] and r["ocr_text"].strip()]
    print(f"[prep] {len(labeled)} labeled docs with OCR text "
          f"(of {len(records)} total; {len(records) - len(labeled)} unlabeled)")
    if len(labeled) < args.test_size + args.val_size + 100:
        sys.exit("[prep] not enough labeled docs for the requested splits")

    struct = infer_struct([r["annotation"] for r in labeled])
    save_struct(struct, args.schema_out)
    print(f"[prep] schema -> {args.schema_out}:\n{json.dumps(struct, indent=2)}")

    rows = [
        {"id": r["id"], "ocr_text": r["ocr_text"],
         "target": conform(r["annotation"], struct), "source": "gold"}
        for r in labeled
    ]
    random.Random(args.seed).shuffle(rows)
    test = rows[: args.test_size]
    val = rows[args.test_size : args.test_size + args.val_size]
    train = rows[args.test_size + args.val_size :]

    out = Path(args.out)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "val.jsonl", val)
    write_jsonl(out / "test.jsonl", test)
    # IDs of every gold doc, so pseudo-labeling can never touch them.
    (out / "gold_ids.txt").write_text("\n".join(r["id"] for r in rows))
    print(f"[prep] train={len(train)} val={len(val)} test={len(test)} -> {out}/")


if __name__ == "__main__":
    main()

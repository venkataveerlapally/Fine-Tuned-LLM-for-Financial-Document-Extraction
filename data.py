"""Load the invoice dataset from the Hugging Face Hub.

The dataset is a FiftyOne port of the Kaggle "High Quality Invoice Images for OCR"
set. Per sample it has `json_annotation` (a JSON string, ~1.5k samples) and
`ocr_text`; the rest are unlabeled images.

Tries plain `datasets` first; falls back to FiftyOne if the parquet view does not
expose the annotation columns.
"""
from __future__ import annotations

import json
from typing import Any, Callable

DEFAULT_REPO = "Voxel51/high-quality-invoice-images-for-ocr"


def _parse_ann(raw: Any) -> dict | None:
    if raw is None or raw == "":
        return None
    if isinstance(raw, dict):
        return raw
    try:
        obj = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    return obj if isinstance(obj, dict) else None


def load_records(repo: str = DEFAULT_REPO) -> tuple[list[dict], Callable[[dict], Any]]:
    """Returns (records, get_image). Each record: {id, annotation, ocr_text, ...}.
    get_image(record) returns a PIL image (only needed for OCR of unlabeled docs)."""
    try:
        from datasets import load_dataset

        ds = load_dataset(repo, split="train")
        if "json_annotation" in ds.column_names:
            cols = [c for c in ("json_annotation", "ocr_text") if c in ds.column_names]
            meta = ds.select_columns(cols)  # avoids decoding images
            records = [
                {
                    "id": f"hf-{i}",
                    "index": i,
                    "annotation": _parse_ann(row.get("json_annotation")),
                    "ocr_text": row.get("ocr_text") or "",
                }
                for i, row in enumerate(meta)
            ]
            image_col = next((c for c in ds.column_names if c in ("image", "img")), None)

            def get_image(rec: dict):
                if image_col is None:
                    raise RuntimeError("No image column in this dataset view")
                return ds[rec["index"]][image_col].convert("RGB")

            print(f"[data] loaded {len(records)} rows via `datasets`")
            return records, get_image
        print("[data] parquet view has no annotation columns; falling back to FiftyOne")
    except Exception as e:  # noqa: BLE001
        print(f"[data] `datasets` load failed ({e}); falling back to FiftyOne")

    from fiftyone.utils.huggingface import load_from_hub
    from PIL import Image

    fo_ds = load_from_hub(repo)
    anns, ocrs, paths, ids = fo_ds.values(["json_annotation", "ocr_text", "filepath", "id"])
    records = [
        {"id": f"fo-{sid}", "annotation": _parse_ann(a), "ocr_text": o or "", "filepath": p}
        for a, o, p, sid in zip(anns, ocrs, paths, ids)
    ]
    print(f"[data] loaded {len(records)} samples via FiftyOne")
    return records, lambda rec: Image.open(rec["filepath"]).convert("RGB")

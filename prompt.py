"""Prompt construction shared by training, evaluation and serving.

Keeping one prompt format everywhere matters: the fine-tuned model only works
well on the exact format it was trained on.
"""
from __future__ import annotations

import json

SYSTEM_PROMPT = (
    "You are a financial document extraction engine. Given the OCR text of an "
    "invoice, fill in the JSON template. Rules: return ONLY valid JSON with exactly "
    "the template's keys; copy values as they appear in the document; use an empty "
    "string for any value that is not present; include one object per line item."
)

MAX_OCR_CHARS = 8000


def dumps(obj) -> str:
    """Compact JSON: fewer output tokens -> lower latency."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def build_messages(ocr_text: str, skel: dict) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Template:\n{dumps(skel)}\n\nOCR text:\n{ocr_text[:MAX_OCR_CHARS]}",
        },
    ]


def build_prompt_text(tokenizer, ocr_text: str, skel: dict) -> str:
    return tokenizer.apply_chat_template(
        build_messages(ocr_text, skel), tokenize=False, add_generation_prompt=True
    )

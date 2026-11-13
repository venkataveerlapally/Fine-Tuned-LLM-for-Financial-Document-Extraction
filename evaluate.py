"""Step 4: measure field-level F1 on the held-out gold test set with vLLM.

Run twice -- base model (baseline) and fine-tuned model -- with identical prompts:

    python scripts/evaluate.py --model Qwen/Qwen2.5-7B-Instruct --label base
    python scripts/evaluate.py --model outputs/merged           --label finetuned
    python scripts/compare.py results/base_metrics.json results/finetuned_metrics.json

By default no constrained decoding is used, so JSON validity reflects the model
itself. Add --guided to measure with JSON-schema-constrained decoding.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ie.metrics import score  # noqa: E402
from ie.parsing import extract_json  # noqa: E402
from ie.prompt import build_prompt_text  # noqa: E402
from ie.schema import conform, json_schema, load_struct, skeleton  # noqa: E402


def guided_kwargs(schema: dict) -> dict:
    try:  # newer vLLM
        from vllm.sampling_params import StructuredOutputsParams
        return {"structured_outputs": StructuredOutputsParams(json=schema)}
    except ImportError:  # older vLLM
        from vllm.sampling_params import GuidedDecodingParams
        return {"guided_decoding": GuidedDecodingParams(json=schema)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--test", default="data/test.jsonl")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--guided", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    struct = load_struct()
    skel = skeleton(struct)
    rows = [json.loads(l) for l in open(args.test) if l.strip()][: args.limit]

    llm = LLM(model=args.model, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem, dtype="bfloat16", seed=0)
    tok = llm.get_tokenizer()
    prompts = [build_prompt_text(tok, r["ocr_text"], skel) for r in rows]
    sp = dict(temperature=0.0, max_tokens=args.max_tokens)
    if args.guided:
        sp.update(guided_kwargs(json_schema(struct)))

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, SamplingParams(**sp))
    elapsed = time.perf_counter() - t0

    preds, out_tokens = [], []
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{args.label}_predictions.jsonl", "w") as f:
        for row, o in zip(rows, outputs):
            text = o.outputs[0].text
            out_tokens.append(len(o.outputs[0].token_ids))
            parsed = extract_json(text)
            pred = conform(parsed, struct) if parsed is not None else None
            preds.append(pred)
            f.write(json.dumps({"id": row["id"], "raw": text, "pred": pred, "gold": row["target"]},
                               ensure_ascii=False) + "\n")

    metrics = score([r["target"] for r in rows], preds, struct)
    metrics.update({
        "label": args.label,
        "model": args.model,
        "guided": args.guided,
        "offline_docs_per_sec": round(len(rows) / elapsed, 2),
        "avg_output_tokens": round(sum(out_tokens) / len(out_tokens), 1),
    })
    (out_dir / f"{args.label}_metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\n=== {args.label} ({args.model}) on {len(rows)} test docs ===")
    for k in ("precision", "recall", "f1", "json_valid_rate", "doc_exact_match",
              "avg_output_tokens", "offline_docs_per_sec"):
        print(f"{k:>22}: {metrics[k]}")
    worst = sorted(metrics["per_field"].items(), key=lambda kv: kv[1]["f1"])[:5]
    print("  weakest fields:", ", ".join(f"{p}={m['f1']}" for p, m in worst))


if __name__ == "__main__":
    main()

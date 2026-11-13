"""Step 3: merge the LoRA adapter into the bf16 base model for vLLM serving.

    python scripts/merge_lora.py --adapter outputs/qlora/adapter --out outputs/merged
"""
from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter", default="outputs/qlora/adapter")
    ap.add_argument("--out", default="outputs/merged")
    args = ap.parse_args()

    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map="cpu")
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    merged.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(args.out)
    print(f"[merge] merged model -> {args.out}")


if __name__ == "__main__":
    main()

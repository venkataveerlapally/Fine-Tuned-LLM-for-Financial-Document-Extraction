"""Step 2: QLoRA fine-tuning of Qwen2.5-7B-Instruct.

- Base weights loaded in 4-bit NF4 (bitsandbytes), frozen.
- LoRA adapters on all attention + MLP projections are the only trainable params.
- Loss is computed on the assistant's JSON answer only (prompt tokens masked).

Fits on a single 24 GB GPU (A10G / L4 / 3090 / 4090).

    python scripts/train_qlora.py --train data/train.jsonl data/pseudo.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from ie.prompt import build_messages, dumps  # noqa: E402
from ie.schema import load_struct, skeleton  # noqa: E402

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def read_jsonl(paths: list[str]) -> list[dict]:
    rows = []
    for p in paths:
        if not Path(p).exists():
            print(f"[train] skipping missing file {p}")
            continue
        with open(p) as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return rows


def encode(rows: list[dict], tok, skel: dict, max_len: int) -> tuple[list[dict], int]:
    out, dropped = [], 0
    for ex in rows:
        msgs = build_messages(ex["ocr_text"], skel)
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template(
            msgs + [{"role": "assistant", "content": dumps(ex["target"])}], tokenize=False
        )
        assert full.startswith(prompt), "chat template is not prefix-stable"
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        a_ids = tok(full[len(prompt):], add_special_tokens=False)["input_ids"]
        ids = p_ids + a_ids
        if len(ids) > max_len:  # never truncate the answer; drop instead
            dropped += 1
            continue
        out.append({
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "labels": [-100] * len(p_ids) + a_ids,
        })
    return out, dropped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--train", nargs="+", default=["data/train.jsonl", "data/pseudo.jsonl"])
    ap.add_argument("--val", default="data/val.jsonl")
    ap.add_argument("--out", default="outputs/qlora")
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    skel = skeleton(load_struct())
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_rows, val_rows = read_jsonl(args.train), read_jsonl([args.val])
    train_enc, d1 = encode(train_rows, tok, skel, args.max_len)
    val_enc, d2 = encode(val_rows, tok, skel, args.max_len)
    print(f"[train] train={len(train_enc)} (dropped {d1} too long), val={len(val_enc)} (dropped {d2})")
    by_source: dict[str, int] = {}
    for r in train_rows:
        by_source[r.get("source", "?")] = by_source.get(r.get("source", "?"), 0) + 1
    print(f"[train] training docs by source: {by_source}")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        bf16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        group_by_length=True,
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=Dataset.from_list(train_enc),
        eval_dataset=Dataset.from_list(val_enc),
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100, pad_to_multiple_of=8),
    )
    trainer.train()

    adapter_dir = Path(args.out) / "adapter"
    trainer.save_model(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    print(f"[train] best adapter saved -> {adapter_dir}")


if __name__ == "__main__":
    main()

# Fine-Tuned LLM for Financial Document Extraction

QLoRA fine-tuning of **Qwen2.5-7B-Instruct** to turn invoice OCR text into structured JSON,
served with **vLLM + FastAPI + Docker** on a single GPU.

```
OCR text ──► FastAPI /extract ──► vLLM (fine-tuned Qwen2.5-7B, continuous batching) ──► JSON
```

| Resume claim | Where it's produced |
|---|---|
| Fine-tuned Qwen2.5-7B with QLoRA | `scripts/train_qlora.py` (4-bit NF4 base + LoRA r=16 on all projections) |
| ~3,000 financial documents | `scripts/prepare_data.py` (≈1.5k gold) + `scripts/pseudo_label.py` (≈1.5k teacher-labeled) |
| Extraction F1 before → after | `scripts/evaluate.py` on the base vs fine-tuned model, then `scripts/compare.py` |
| vLLM + Docker + FastAPI | `docker-compose.yml`, `serve/app.py`, `serve/Dockerfile` |
| X req/s at p95 < Y ms | `scripts/load_test.py --rps 20` |

## Dataset

[High Quality Invoice Images for OCR](https://huggingface.co/datasets/Voxel51/high-quality-invoice-images-for-ocr)
(8,181 synthetic invoices). About 1,489 have gold JSON labels plus OCR text. Labels cover:
invoice header (number, dates, seller/client), line items (description, quantity, total
price), totals (tax, discount, total) and payment info.

The output schema is **inferred from the gold labels** and saved to `artifacts/schema.json`.
Training, evaluation and serving all read it, so field names are never hard-coded.

## Hardware

- Training: one 24 GB GPU (A10G, L4, RTX 3090/4090). About 1–3 h for 3 epochs on 3k docs.
- Serving: one 24 GB GPU for the bf16 merged model with `--max-model-len 4096`.
- Pseudo-labeling (optional): a teacher such as Qwen2.5-72B-Instruct-AWQ needs ~48 GB+, or use any hosted OpenAI-compatible API.

## Run it

```bash
pip install vllm && pip install -r requirements-train.txt
make test                      # unit tests for metrics/schema/parsing (CPU only)

# 1. Data
make data                      # -> data/{train,val,test}.jsonl + artifacts/schema.json
make pseudo TEACHER_URL=http://teacher:8000/v1   # optional: +~1.5k training docs
#    spot-check ~50 rows of data/pseudo.jsonl by hand before training!

# 2. Baseline
make eval-base                 # base Qwen2.5-7B, same prompt -> results/base_metrics.json

# 3. Fine-tune + merge
make train                     # -> outputs/qlora/adapter
make merge                     # -> outputs/merged (bf16, ready for vLLM)

# 4. Evaluate
make eval-ft && make compare   # before/after F1 table, per-field breakdown

# 5. Serve + load test
make serve                     # docker compose: vLLM :8000, API :8080
curl -s localhost:8080/extract -H 'content-type: application/json' \
     -d "{\"text\": $(head -1 data/test.jsonl | python -c 'import json,sys;print(json.dumps(json.loads(sys.stdin.read())["ocr_text"]))')}"
make loadtest                  # open loop at 20 req/s, reports p50/p95/p99
make loadtest-max              # closed loop, finds max throughput
```



## Layout

```
ie/            schema inference, prompt, JSON parsing, F1 metrics, dataset loader
scripts/       prepare_data, pseudo_label, train_qlora, merge_lora, evaluate, compare, load_test
serve/         FastAPI app + Dockerfile
tests/         unit tests (CPU)
docker-compose.yml, Makefile
```

"""FastAPI front end for the fine-tuned extractor.

Architecture:
    client -> FastAPI (/extract) -> vLLM OpenAI-compatible server (continuous batching on 1 GPU)

FastAPI handles validation, prompt building, schema enforcement and error
handling; vLLM handles efficient batched generation. The vLLM call uses
JSON-schema-constrained decoding so responses are always valid JSON.

    VLLM_URL=http://localhost:8000 uvicorn serve.app:app --port 8080
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ie.parsing import extract_json
from ie.prompt import build_messages
from ie.schema import conform, json_schema, load_struct, skeleton

VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000").rstrip("/")
MODEL_NAME = os.getenv("MODEL_NAME", "invoice-extractor")
SCHEMA_PATH = os.getenv("SCHEMA_PATH", "artifacts/schema.json")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1536"))
GUIDED = os.getenv("GUIDED", "1") == "1"
TIMEOUT_S = float(os.getenv("TIMEOUT_S", "60"))

STRUCT = load_struct(SCHEMA_PATH)
SKELETON = skeleton(STRUCT)
SCHEMA = json_schema(STRUCT)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(
        base_url=VLLM_URL,
        timeout=TIMEOUT_S,
        limits=httpx.Limits(max_connections=512, max_keepalive_connections=128),
    )
    yield
    await app.state.client.aclose()


app = FastAPI(title="Financial Document Extractor", version="1.0", lifespan=lifespan)


class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50_000, description="OCR text of the document")


class ExtractResponse(BaseModel):
    data: dict
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


def build_payload(text: str) -> dict:
    payload = {
        "model": MODEL_NAME,
        "messages": build_messages(text, SKELETON),
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
    }
    if GUIDED:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "invoice", "schema": SCHEMA, "strict": True},
        }
    return payload


@app.post("/extract", response_model=ExtractResponse)
async def extract(req: ExtractRequest) -> ExtractResponse:
    t0 = time.perf_counter()
    try:
        r = await app.state.client.post("/v1/chat/completions", json=build_payload(req.text))
    except httpx.HTTPError as e:
        raise HTTPException(status_code=503, detail=f"inference backend unavailable: {e}") from e
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"inference backend error: {r.text[:300]}")

    body = r.json()
    data = extract_json(body["choices"][0]["message"]["content"])
    if data is None:
        raise HTTPException(status_code=422, detail="model returned invalid JSON")
    usage = body.get("usage") or {}
    return ExtractResponse(
        data=conform(data, STRUCT),
        latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
    )


@app.get("/health")
async def health() -> dict:
    try:
        r = await app.state.client.get("/health", timeout=5)
        ok = r.status_code == 200
    except httpx.HTTPError:
        ok = False
    if not ok:
        raise HTTPException(status_code=503, detail="vLLM not ready")
    return {"status": "ok", "model": MODEL_NAME}


@app.get("/schema")
async def schema() -> dict:
    return {"template": SKELETON, "json_schema": SCHEMA}

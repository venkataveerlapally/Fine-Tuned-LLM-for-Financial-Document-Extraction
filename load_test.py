"""Step 5: measure throughput and tail latency of the running API.

Two modes:
  --rps 20         open loop: send exactly 20 req/s, report latency at that load
                   (this is the honest way to claim "20 req/s at p95 < X ms")
  --concurrency 32 closed loop: N requests in flight, report max throughput

    python scripts/load_test.py --rps 20 --duration 60
    python scripts/load_test.py --concurrency 32 --requests 500
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time

import httpx


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


async def send(client: httpx.AsyncClient, url: str, text: str, results: list) -> None:
    t0 = time.perf_counter()
    try:
        r = await client.post(url, json={"text": text})
        ok = r.status_code == 200
    except httpx.HTTPError:
        ok = False
    results.append((time.perf_counter() - t0, ok))


async def run(args) -> None:
    texts = [json.loads(l)["ocr_text"] for l in open(args.data) if l.strip()]
    rng = random.Random(0)
    limits = httpx.Limits(max_connections=2000, max_keepalive_connections=500)
    async with httpx.AsyncClient(timeout=120, limits=limits) as client:
        print(f"[load] warmup ({args.warmup} requests)...")
        await asyncio.gather(*(send(client, args.url, rng.choice(texts), []) for _ in range(args.warmup)))

        results: list = []
        t_start = time.perf_counter()
        if args.rps:
            n = int(args.rps * args.duration)
            print(f"[load] open loop: {args.rps} req/s for {args.duration}s ({n} requests)")
            tasks = []
            for i in range(n):
                delay = t_start + i / args.rps - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
                tasks.append(asyncio.create_task(send(client, args.url, rng.choice(texts), results)))
            await asyncio.gather(*tasks)
        else:
            print(f"[load] closed loop: concurrency={args.concurrency}, {args.requests} requests")
            sem = asyncio.Semaphore(args.concurrency)

            async def bounded():
                async with sem:
                    await send(client, args.url, rng.choice(texts), results)

            await asyncio.gather(*(bounded() for _ in range(args.requests)))
        wall = time.perf_counter() - t_start

    lat_ms = [d * 1000 for d, ok in results if ok]
    errors = sum(1 for _, ok in results if not ok)
    p95 = pct(lat_ms, 0.95)
    print("\n=== load test results ===")
    print(f"  requests       : {len(results)}  (errors: {errors}, {errors / max(len(results), 1):.1%})")
    print(f"  throughput     : {len(lat_ms) / wall:.1f} successful req/s")
    print(f"  latency p50    : {pct(lat_ms, 0.50):.0f} ms")
    print(f"  latency p95    : {p95:.0f} ms")
    print(f"  latency p99    : {pct(lat_ms, 0.99):.0f} ms")
    if args.rps:
        verdict = "PASS" if p95 < args.target_p95_ms and errors == 0 else "FAIL"
        print(f"  target         : {args.rps} req/s with p95 < {args.target_p95_ms} ms -> {verdict}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080/extract")
    ap.add_argument("--data", default="data/test.jsonl")
    ap.add_argument("--rps", type=float, default=None)
    ap.add_argument("--duration", type=float, default=60)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--requests", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--target-p95-ms", type=float, default=800)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()

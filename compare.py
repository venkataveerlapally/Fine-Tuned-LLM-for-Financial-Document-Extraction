"""Print a before/after table from two metrics files.

    python scripts/compare.py results/base_metrics.json results/finetuned_metrics.json
"""
import json
import sys

KEYS = ["precision", "recall", "f1", "json_valid_rate", "doc_exact_match", "avg_output_tokens"]

a, b = (json.load(open(p)) for p in sys.argv[1:3])
print(f"{'metric':>20} | {a['label']:>10} | {b['label']:>10}")
print("-" * 47)
for k in KEYS:
    print(f"{k:>20} | {a[k]:>10} | {b[k]:>10}")
print("\nPer-field F1:")
for field in sorted(set(a["per_field"]) | set(b["per_field"])):
    fa = a["per_field"].get(field, {}).get("f1", 0)
    fb = b["per_field"].get(field, {}).get("f1", 0)
    print(f"{field:>32}: {fa:.3f} -> {fb:.3f}")

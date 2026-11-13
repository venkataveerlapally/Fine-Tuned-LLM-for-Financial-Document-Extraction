from ie.metrics import normalize, score
from ie.parsing import extract_json
from ie.schema import conform, flatten, infer_struct, json_schema, skeleton

GOLD = {
    "invoice": {"invoice_number": "INV-001", "invoice_date": "03/14/2025",
                "seller_name": "Acme Corp", "client_name": "Bob LLC"},
    "items": [
        {"description": "Widget", "quantity": "2.00", "total_price": "20.00"},
        {"description": "Gadget", "quantity": "1.00", "total_price": "15.50"},
    ],
    "subtotal": {"tax": "3.55", "discount": "", "total": "$39.05"},
}
STRUCT = infer_struct([GOLD])


def test_infer_struct():
    assert STRUCT["invoice"]["type"] == "object"
    assert STRUCT["items"] == {"type": "array", "keys": ["description", "quantity", "total_price"]}
    assert STRUCT["subtotal"]["keys"] == ["tax", "discount", "total"]


def test_infer_struct_skips_null_and_merges_keys():
    s = infer_struct([{"a": None}, {"a": {"x": "1"}}, {"a": {"y": "2"}}])
    assert s["a"] == {"type": "object", "keys": ["x", "y"]}


def test_skeleton_and_schema():
    sk = skeleton(STRUCT)
    assert sk["items"] == [{"description": "", "quantity": "", "total_price": ""}]
    js = json_schema(STRUCT)
    assert js["properties"]["items"]["type"] == "array"
    assert js["properties"]["subtotal"]["additionalProperties"] is False


def test_conform_fills_drops_and_stringifies():
    out = conform({"invoice": {"invoice_number": 7, "junk": "x"}, "items": "bad"}, STRUCT)
    assert out["invoice"]["invoice_number"] == "7"
    assert "junk" not in out["invoice"]
    assert out["invoice"]["seller_name"] == ""
    assert out["items"] == []
    assert out["subtotal"] == {"tax": "", "discount": "", "total": ""}


def test_flatten_items_share_path():
    paths = [p for p, _ in flatten(GOLD, STRUCT)]
    assert paths.count("items.description") == 2


def test_normalize():
    assert normalize("$1,250.0") == "1250.00"
    assert normalize("1250") == "1250.00"
    assert normalize("03/14/2025") == normalize("2025-03-14") == "2025-03-14"
    assert normalize("Mar 14, 2025") == "2025-03-14"
    assert normalize("  Acme   CORP. ") == "acme corp"
    assert normalize("") == ""


def test_extract_json():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! {"a": {"b": 2}} hope this helps') == {"a": {"b": 2}}
    assert extract_json("not json") is None
    assert extract_json("[1,2]") is None
    assert extract_json(None) is None


def test_score_perfect_with_formatting_noise():
    pred = conform(GOLD, STRUCT)
    pred["subtotal"]["total"] = "39.05"           # no $ sign
    pred["invoice"]["invoice_date"] = "2025-03-14"  # different date format
    pred["items"] = list(reversed(pred["items"]))   # order shouldn't matter
    m = score([GOLD], [pred], STRUCT)
    assert m["f1"] == 1.0 and m["doc_exact_match"] == 1.0


def test_score_partial():
    pred = conform(GOLD, STRUCT)
    pred["subtotal"]["total"] = "40.00"   # wrong -> 1 FP + 1 FN
    pred["invoice"]["client_name"] = ""   # missing -> 1 FN
    m = score([GOLD], [pred], STRUCT)
    n_gold = 12  # non-empty gold fields
    tp = n_gold - 2
    assert m["precision"] == round(tp / (tp + 1), 4)
    assert m["recall"] == round(tp / n_gold, 4)
    assert m["per_field"]["subtotal.total"]["f1"] == 0.0


def test_score_invalid_json_counts_as_misses():
    m = score([GOLD, GOLD], [None, conform(GOLD, STRUCT)], STRUCT)
    assert m["json_valid_rate"] == 0.5
    assert m["precision"] == 1.0
    assert m["recall"] == 0.5

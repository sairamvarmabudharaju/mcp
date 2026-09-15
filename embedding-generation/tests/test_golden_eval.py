import json
from pathlib import Path

import pytest

from evals.golden.evaluate_golden import (
    best_rank,
    compare,
    load_golden,
    matches,
    metrics,
    normalize_url,
    url_base,
)

INTR = "https://developer.arm.com/architectures/instruction-sets/intrinsics/"
DASH = "https://www.arm.com/developer-hub/ecosystem-dashboard/"
LP = "https://learn.arm.com/learning-paths/"


def test_normalize_url_strips_tracking_but_keeps_query_and_fragment():
    assert normalize_url(f"{DASH}?package=numpy&utm_source=arm-mcp") == f"{DASH}?package=numpy"
    assert normalize_url(f"{INTR}?utm_source=arm-mcp#q=vaddq_f32") == f"{INTR}#q=vaddq_f32"
    assert normalize_url(None) == ""


def test_url_base_drops_query_fragment_and_trailing_slash():
    assert url_base(f"{LP}cross-platform/docker/buildx/?utm_source=x#frag") == f"{LP}cross-platform/docker/buildx"


@pytest.mark.parametrize(
    "result,expected,ok",
    [
        (f"{LP}cross-platform/docker/buildx?utm_source=arm-mcp", {"url": f"{LP}cross-platform/docker/buildx", "match": "page"}, True),
        (f"{LP}cross-platform/docker/buildx", {"url": f"{LP}cross-platform/docker", "match": "prefix"}, True),
        (f"{LP}cross-platform/docker", {"url": f"{LP}cross-platform/docker", "match": "prefix"}, True),
        (f"{LP}cross-platform/docker-build-cloud", {"url": f"{LP}cross-platform/docker", "match": "prefix"}, False),
        (f"{INTR}?utm_source=arm-mcp#q=vclzq_u32", {"url": f"{INTR}#q=vclz", "match": "startswith"}, True),
        (f"{INTR}?utm_source=arm-mcp#q=vclsq_s8", {"url": f"{INTR}#q=vclz", "match": "startswith"}, False),
        (f"{DASH}?package=numpy&utm_source=arm-mcp", {"url": f"{DASH}?package=numpy", "match": "exact"}, True),
        (f"{DASH}?package=numpy-stl&utm_source=arm-mcp", {"url": f"{DASH}?package=numpy", "match": "exact"}, False),
        (f"{INTR}#q=vzipq_f16", {"url": r"#q=vzip[12]?q?_", "match": "regex"}, True),
        (None, {"url": f"{LP}x", "match": "page"}, False),
    ],
)
def test_match_rules(result, expected, ok):
    assert matches(result, expected) is ok


def test_best_rank_respects_grade_threshold():
    expected = [{"url": f"{LP}a", "match": "page", "grade": 2}, {"url": f"{LP}b", "match": "page", "grade": 1}]
    ranked = [f"{LP}b", f"{LP}c", f"{LP}a"]
    assert best_rank(ranked, expected, 1) == 1
    assert best_rank(ranked, expected, 2) == 3
    assert best_rank([f"{LP}c"], expected, 1) is None


def test_metrics_and_compare():
    rows = [
        {"id": "x", "rank_lenient": 1, "rank_strict": 1, "distinct_pages": 5},
        {"id": "y", "rank_lenient": 4, "rank_strict": None, "distinct_pages": 3},
        {"id": "z", "rank_lenient": None, "rank_strict": None, "distinct_pages": 1},
    ]
    m = metrics(rows, 5)
    assert m["hit@1"] == pytest.approx(1 / 3)
    assert m["hit@5"] == pytest.approx(2 / 3)
    assert m["strict_hit@5"] == pytest.approx(1 / 3)
    assert m["mrr"] == pytest.approx((1 + 0.25) / 3)
    baseline = [dict(r, rank_lenient=None if r["id"] == "x" else 2) for r in rows]
    for r in rows:
        r["query"] = r["id"]
    diff = compare(rows, baseline, 5)
    assert [d[0] for d in diff["improved"]] == ["x"]
    assert [d[0] for d in diff["regressed"]] == ["y", "z"]
    assert 0.0 <= diff["p_value"] <= 1.0


def test_shipped_golden_set_is_valid():
    doc = load_golden(
        Path(__file__).resolve().parents[1] / "evals" / "golden" / "eval_golden.json"
    )
    assert len(doc["items"]) >= 150
    splits = {i["split"] for i in doc["items"]}
    assert splits == {"dev", "holdout"}
    holdout = sum(1 for i in doc["items"] if i["split"] == "holdout")
    assert 0.15 * len(doc["items"]) <= holdout <= 0.35 * len(doc["items"])


def test_golden_set_rejects_items_without_a_grade_two_answer(tmp_path):
    doc = load_golden(
        Path(__file__).resolve().parents[1] / "evals" / "golden" / "eval_golden.json"
    )
    bad = dict(doc)
    bad["items"] = [dict(doc["items"][0], expected=[{"url": "https://x", "match": "page", "grade": 1}])]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="grade-2"):
        load_golden(path)

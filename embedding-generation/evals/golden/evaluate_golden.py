"""Evaluate knowledge_base_search against the golden query set (eval_golden.json).

Compared with evaluate_retrieval.py this runner supports per-URL match rules
(page / prefix / startswith / exact / regex), graded relevance (strict = grade 2
only, lenient = any grade), per-stratum breakdowns, a bootstrap confidence
interval on the headline metric, and a paired comparison against a saved run.

Example:
    python evaluate_golden.py --model-path .cache/embedding-model --output golden_run.json
    python evaluate_golden.py --model-path .cache/embedding-model --baseline golden_run.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from collections.abc import Callable
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MATCH_TYPES = ("page", "prefix", "startswith", "exact", "regex")
STRATA = ("form", "intent", "area", "difficulty", "split")
TRACKING_PARAMS = {"utm_source"}


def normalize_url(url: str | None) -> str:
    """Drop tracking query parameters; keep path, remaining query and fragment."""
    if not url:
        return ""
    parsed = urlparse(url)
    params = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True) if k not in TRACKING_PARAMS]
    return urlunparse(parsed._replace(query=urlencode(params)))


def url_base(url: str | None) -> str:
    parsed = urlparse(url or "")
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/') or '/'}"


def matches(result_url: str | None, expected: dict) -> bool:
    kind = expected["match"]
    target = expected["url"]
    full = normalize_url(result_url)
    if kind == "page":
        return url_base(full) == url_base(target)
    if kind == "prefix":
        base, want = url_base(full), url_base(target)
        return base == want or base.startswith(want + "/")
    if kind == "startswith":
        return full.startswith(target)
    if kind == "exact":
        return full == normalize_url(target)
    if kind == "regex":
        return re.search(target, full) is not None
    raise ValueError(f"unknown match type {kind!r}")


def best_rank(ranked_urls: list[str | None], expected: list[dict], min_grade: int) -> int | None:
    wanted = [e for e in expected if e["grade"] >= min_grade]
    for rank, url in enumerate(ranked_urls, start=1):
        if any(matches(url, e) for e in wanted):
            return rank
    return None


def load_golden(path: Path) -> dict:
    with path.open() as file:
        doc = json.load(file)
    schema = doc["schema"]
    seen: set[str] = set()
    for item in doc["items"]:
        if item["id"] in seen:
            raise ValueError(f"duplicate id {item['id']}")
        seen.add(item["id"])
        for field in ("form", "intent", "difficulty"):
            if item[field] not in schema[field]:
                raise ValueError(f"{item['id']}: bad {field} {item[field]!r}")
        if not any(e["grade"] == 2 for e in item["expected"]):
            raise ValueError(f"{item['id']}: needs at least one grade-2 expected url")
        for e in item["expected"]:
            if e["match"] not in MATCH_TYPES:
                raise ValueError(f"{item['id']}: bad match {e['match']!r}")
            if e["match"] == "regex":
                re.compile(e["url"])
    return doc


def evaluate(items: list[dict], retrieve: Callable[[str, int], list[str | None]], top_k: int) -> list[dict]:
    rows = []
    for item in items:
        error = None
        try:
            urls = list(retrieve(item["query"], top_k))[:top_k]
        except Exception as exc:  # keep going; report the failure per item
            urls, error = [], str(exc)
        rows.append(
            {
                "id": item["id"],
                "query": item["query"],
                "ranked_urls": urls,
                "rank_lenient": best_rank(urls, item["expected"], 1),
                "rank_strict": best_rank(urls, item["expected"], 2),
                "distinct_pages": len({url_base(normalize_url(u)) for u in urls}),
                "error": error,
            }
        )
    return rows


def hit(rank: int | None, k: int) -> float:
    return 1.0 if rank is not None and rank <= k else 0.0


def metrics(rows: list[dict], top_k: int) -> dict:
    if not rows:
        return {}
    n = len(rows)
    lenient = [r["rank_lenient"] for r in rows]
    strict = [r["rank_strict"] for r in rows]
    return {
        "n": n,
        "hit@1": sum(hit(r, 1) for r in lenient) / n,
        "hit@3": sum(hit(r, 3) for r in lenient) / n,
        f"hit@{top_k}": sum(hit(r, top_k) for r in lenient) / n,
        f"strict_hit@{top_k}": sum(hit(r, top_k) for r in strict) / n,
        "mrr": sum(1 / r for r in lenient if r) / n,
        "distinct_pages": statistics.mean(r["distinct_pages"] for r in rows),
    }


def bootstrap_ci(values: list[float], iterations: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    means = sorted(statistics.mean(rng.choice(values) for _ in range(n)) for _ in range(iterations))
    return means[int(0.025 * iterations)], means[int(0.975 * iterations)]


def permutation_p_value(deltas: list[float], iterations: int = 5000, seed: int = 0) -> float:
    """Paired sign-flip test on per-item differences (H0: no difference)."""
    nonzero = [d for d in deltas if d != 0]
    if not nonzero:
        return 1.0
    rng = random.Random(seed)
    observed = abs(sum(nonzero))
    hits = sum(1 for _ in range(iterations) if abs(sum(d if rng.random() < 0.5 else -d for d in nonzero)) >= observed)
    return hits / iterations


def compare(rows: list[dict], baseline_rows: list[dict], top_k: int) -> dict:
    base = {r["id"]: r for r in baseline_rows}
    improved, regressed, deltas = [], [], []
    for row in rows:
        old = base.get(row["id"])
        if old is None:
            continue
        a, b = old["rank_lenient"], row["rank_lenient"]
        deltas.append(hit(b, top_k) - hit(a, top_k))
        if (b or 99) < (a or 99):
            improved.append((row["id"], a, b, row["query"]))
        elif (b or 99) > (a or 99):
            regressed.append((row["id"], a, b, row["query"]))
    return {"improved": improved, "regressed": regressed, "p_value": permutation_p_value(deltas), "compared": len(deltas)}


def print_report(doc: dict, rows: list[dict], top_k: int, show_misses: int, baseline: list[dict] | None) -> None:
    items = {item["id"]: item for item in doc["items"]}
    overall = metrics(rows, top_k)
    lo, hi = bootstrap_ci([hit(r["rank_lenient"], top_k) for r in rows])
    print(f"Golden set v{doc['version']} ({overall['n']} queries, corpus: {doc['corpus']})")
    print(
        f"hit@1 {overall['hit@1']:.1%}  hit@3 {overall['hit@3']:.1%}  hit@{top_k} {overall[f'hit@{top_k}']:.1%} "
        f"[95% CI {lo:.1%}-{hi:.1%}]  strict hit@{top_k} {overall[f'strict_hit@{top_k}']:.1%}  "
        f"MRR {overall['mrr']:.3f}  distinct pages in top {top_k}: {overall['distinct_pages']:.2f}"
    )
    errors = [r for r in rows if r["error"]]
    if errors:
        print(f"Errors: {len(errors)} (first: {errors[0]['id']}: {errors[0]['error']})")
    for stratum in STRATA:
        print(f"\nby {stratum}:")
        groups: dict[str, list[dict]] = {}
        for row in rows:
            groups.setdefault(items[row["id"]][stratum], []).append(row)
        for name, group in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            m = metrics(group, top_k)
            print(f"  {name:20} n={m['n']:3d}  hit@1 {m['hit@1']:6.1%}  hit@{top_k} {m[f'hit@{top_k}']:6.1%}  strict {m[f'strict_hit@{top_k}']:6.1%}  mrr {m['mrr']:.3f}")
    misses = [r for r in rows if r["rank_lenient"] is None]
    print(f"\nmisses (not in top {top_k}): {len(misses)}")
    for row in misses[:show_misses]:
        print(f"  [{row['id']}] {row['query']}")
        print(f"      got: {[normalize_url(u).replace('https://', '')[:80] for u in row['ranked_urls'][:3]]}")
    if baseline is not None:
        diff = compare(rows, baseline, top_k)
        print(f"\nvs baseline ({diff['compared']} paired): improved {len(diff['improved'])}, regressed {len(diff['regressed'])}, sign-flip p={diff['p_value']:.3f}")
        for id_, a, b, q in diff["improved"]:
            print(f"  + [{id_}] {a}->{b}  {q[:80]}")
        for id_, a, b, q in diff["regressed"]:
            print(f"  - [{id_}] {a}->{b}  {q[:80]}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval against the golden query set.")
    parser.add_argument("--index-path", default="usearch_index.bin")
    parser.add_argument("--metadata-path", default="metadata.json")
    parser.add_argument("--eval-path", default="eval_golden.json")
    parser.add_argument("--model-path", default=None, help="Local model directory (offline). Omit to use --model-name.")
    parser.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--split", choices=("all", "dev", "holdout"), default="all")
    parser.add_argument("--output", type=Path, default=None, help="Write per-query results as JSON.")
    parser.add_argument("--baseline", type=Path, default=None, help="Earlier --output file to compare against.")
    parser.add_argument("--show-misses", type=int, default=15)
    args = parser.parse_args()

    from arm_kb_search import load_search_resources, search  # noqa: E402  (slow import)

    doc = load_golden(Path(args.eval_path))
    items = [i for i in doc["items"] if args.split == "all" or i["split"] == args.split]
    resources = load_search_resources(
        metadata_path=args.metadata_path,
        usearch_index_path=args.index_path,
        model_name=args.model_name,
        model_path=args.model_path,
        include_disclaimers=False,
    )

    def retrieve(query: str, top_k: int) -> list[str | None]:
        return [item.get("url") for item in search(query, resources, k=top_k)]

    rows = evaluate(items, retrieve, args.top_k)
    baseline = None
    if args.baseline:
        with args.baseline.open() as file:
            baseline = json.load(file)["rows"]
    print_report(doc, rows, args.top_k, args.show_misses, baseline)
    if args.output:
        with args.output.open("w") as file:
            json.dump({"eval_version": doc["version"], "top_k": args.top_k, "split": args.split, "rows": rows}, file, indent=1)
        print(f"\nwrote {args.output}")
    return 1 if any(r["error"] for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())

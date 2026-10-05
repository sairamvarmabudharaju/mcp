"""One retrieval evaluator for PR smoke checks and weekly benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arm_kb_search import load_search_resources, search  # noqa: E402
from arm_kb_search.evaluation import (  # noqa: E402
    evaluate_retrieval,
    load_eval_rows,
    suite_url_matches,
)


def valid_url(value):
    return (
        isinstance(value, str)
        and urlparse(value).scheme in ("http", "https")
        and bool(urlparse(value).netloc)
    )


def validate_rows(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("Evaluation suite must be a nonempty JSON array")
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each evaluation question must be an object")
        # Legacy --eval-path files used question text as their identity.
        row.setdefault("id", row.get("question"))
        if any(
            not isinstance(row.get(key), str) or not row[key].strip()
            for key in ("id", "question")
        ):
            raise ValueError("Each question needs a nonempty ID and question")
        if row["id"] in seen:
            raise ValueError(f"Duplicate question ID: {row['id']}")
        seen.add(row["id"])
        urls = row.get("expected_urls")
        if (
            not isinstance(urls, list)
            or not urls
            or not all(valid_url(url) for url in urls)
        ):
            raise ValueError(
                f"{row['id']}: expected_urls must contain absolute HTTP(S) URLs"
            )
        if len(urls) != len(set(urls)):
            raise ValueError(f"{row['id']}: duplicate expected URL")
        for field in ("area", "topic", "intent"):
            if field in row and (
                not isinstance(row[field], str) or not row[field].strip()
            ):
                raise ValueError(f"{row['id']}: invalid {field}")
    return rows


def git(*args):
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def select_rows(rows, eval_path, ids=None, changed_since=None):
    if ids:
        unknown = set(ids) - {row["id"] for row in rows}
        if unknown:
            raise ValueError(f"Unknown question IDs: {', '.join(sorted(unknown))}")
        return [row for row in rows if row["id"] in ids], []
    if not changed_since:
        return rows, []
    base = git("merge-base", changed_since, "HEAD")
    relative = eval_path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    previous = (
        validate_rows(json.loads(git("show", f"{base}:{relative}")))
        if git("ls-tree", "--name-only", base, "--", relative)
        else []
    )
    old = {row["id"]: row for row in previous}
    removed = sorted(old.keys() - {row["id"] for row in rows})
    return [row for row in rows if old.get(row["id"]) != row], removed


def summarize(cases, top_k):
    total = len(cases)
    errors = sum(case["error"] is not None for case in cases)
    hits = sum(case["match_rank"] is not None for case in cases)
    return {
        "total": total,
        "hits": hits,
        "misses": total - hits - errors,
        "errors": errors,
        **{
            f"hit_at_{k}": (
                sum(
                    case["match_rank"] is not None and case["match_rank"] <= k
                    for case in cases
                )
                / total
                if total
                else None
            )
            if top_k >= k
            else None
            for k in (1, 3, 5)
        },
        "mrr": sum(case["reciprocal_rank"] for case in cases) / total
        if total
        else None,
    }


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_report(output, report):
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


def evaluate(
    index_path,
    metadata_path,
    eval_path,
    model_path,
    top_k,
    *,
    suite="benchmark",
    ids=None,
    changed_since=None,
    output=None,
    baseline=None,
):
    if top_k < 1:
        raise ValueError("--top-k must be positive")
    if output and output.exists():
        raise ValueError(f"Output already exists; choose a new report path: {output}")
    rows = validate_rows(load_eval_rows(eval_path))
    selected, removed = select_rows(rows, eval_path, ids, changed_since)
    print(f"{suite}: selected {len(selected)}/{len(rows)} questions; top-k={top_k}")
    if removed:
        print(f"Removed IDs: {', '.join(removed)}")
    fingerprint = hashlib.sha256(
        json.dumps(sorted(selected, key=lambda row: row["id"]), sort_keys=True).encode()
    ).hexdigest()
    report = {
        "format_version": 1,
        "suite": suite,
        "policy": f"{suite}-v1",
        "top_k": top_k,
        "input_fingerprint": fingerprint,
        "suite_total": len(rows),
        "removed_ids": removed,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
    }
    if not selected:
        print("No changed questions; retrieval was not started.")
        report["status"] = "no_changed_questions"
        write_report(output, report)
        return 0
    previous = None
    if baseline:
        previous = json.loads(baseline.read_text())
        keys = ("format_version", "suite", "policy", "top_k", "input_fingerprint")
        if (
            not isinstance(previous, dict)
            or any(previous.get(key) != report[key] for key in keys)
            or previous.get("status") != "complete"
            or not isinstance(previous.get("summary"), dict)
            or previous["summary"].get("errors") != 0
            or not isinstance(previous.get("cases"), list)
        ):
            raise ValueError(
                "Incompatible or failed baseline; run a fresh baseline with the same questions, policy and depth"
            )
        for case in previous["cases"]:
            if (
                not isinstance(case, dict)
                or not isinstance(case.get("question_id"), str)
                or "match_rank" not in case
                or case.get("error") is not None
            ):
                raise ValueError("Invalid baseline case")
            rank = case["match_rank"]
            if rank is not None and (type(rank) is not int or not 1 <= rank <= top_k):
                raise ValueError("Invalid baseline rank")
            if case.get("reciprocal_rank") != (1 / rank if rank else 0):
                raise ValueError("Invalid baseline reciprocal rank")
        if {case.get("question_id") for case in previous["cases"]} != {
            row["id"] for row in selected
        }:
            raise ValueError("Baseline cases do not match selected questions")
        if len(previous["cases"]) != len(selected) or previous["summary"] != summarize(
            previous["cases"], top_k
        ):
            raise ValueError("Invalid baseline summary or duplicate cases")
    try:
        model_files = sorted(path for path in model_path.rglob("*") if path.is_file())
        if not model_files:
            raise ValueError(f"Model directory missing or empty: {model_path}")
        report["corpus"] = {
            "metadata_sha256": file_hash(metadata_path),
            "index_sha256": file_hash(index_path),
            "model_sha256": hashlib.sha256(
                json.dumps(
                    [
                        (path.relative_to(model_path).as_posix(), file_hash(path))
                        for path in model_files
                    ]
                ).encode()
            ).hexdigest(),
        }
        report["target"] = os.environ.get("EVAL_TARGET")
        source_root = Path(__file__).resolve().parents[1]
        report["code_sha256"] = hashlib.sha256(
            json.dumps(
                [
                    (path.relative_to(source_root).as_posix(), file_hash(path))
                    for path in [
                        Path(__file__).resolve(),
                        *sorted((source_root / "arm_kb_search").glob("*.py")),
                    ]
                ]
            ).encode()
        ).hexdigest()
        report["source_revision"] = os.environ.get("GITHUB_SHA")
        if not report["source_revision"]:
            try:
                report["source_revision"] = git("rev-parse", "HEAD")
            except (OSError, subprocess.CalledProcessError):
                pass
        resources = load_search_resources(
            metadata_path=str(metadata_path),
            usearch_index_path=str(index_path),
            model_path=str(model_path),
        )
        if (
            not resources.metadata
            or resources.usearch_index is None
            or len(resources.usearch_index) != len(resources.metadata)
        ):
            raise ValueError(
                "Corpus must have nonempty metadata and a matching, nonempty vector index"
            )
    except Exception as exc:
        report.update(status="error", error=str(exc))
        write_report(output, report)
        print(f"Evaluation setup failed: {exc}", file=sys.stderr)
        return 2

    def retrieve_urls(question, k):
        results = search(question, resources, k=k)
        if not isinstance(results, list) or any(
            not isinstance(item, dict) or not valid_url(item.get("url"))
            for item in results
        ):
            raise ValueError("Retrieval returned malformed results or missing URLs")
        return [item["url"] for item in results]

    result = evaluate_retrieval(
        selected,
        retrieve_urls,
        top_k,
        url_matcher=lambda actual, expected: suite_url_matches(actual, expected, suite),
    )
    cases = [asdict(case) for case in result.cases]
    report.update(
        status="error" if result.errors else "complete",
        cases=cases,
        summary=summarize(cases, top_k),
        finished_at=datetime.now(timezone.utc).isoformat(),
    )
    for field in ("area", "topic", "intent"):
        groups = sorted({row[field] for row in selected if field in row})
        if groups:
            report[f"by_{field}"] = {
                group: summarize(
                    [
                        case
                        for row, case in zip(selected, cases)
                        if row.get(field) == group
                    ],
                    top_k,
                )
                for group in groups
            }
    for metric, value in report["summary"].items():
        print(f"{metric}: {value}")
    for case in cases:
        if case["match_rank"] is None:
            print(
                f"{case['question_id']}: {case['error'] or 'MISS'}; expected={case['expected_urls']}; got={case['ranked_urls']}"
            )
    if previous and not result.errors:
        old = {case["question_id"]: case for case in previous["cases"]}
        report["comparison"] = {
            "baseline": str(baseline),
            "baseline_corpus": previous.get("corpus"),
            "regressions": [
                c["question_id"]
                for c in cases
                if old[c["question_id"]]["match_rank"] is not None
                and c["match_rank"] is None
            ],
            "recoveries": [
                c["question_id"]
                for c in cases
                if old[c["question_id"]]["match_rank"] is None
                and c["match_rank"] is not None
            ],
            "delta": {
                key: report["summary"][key] - previous["summary"][key]
                for key in ("hit_at_1", "hit_at_3", "hit_at_5", "mrr")
                if report["summary"][key] is not None
            },
        }
        print("Comparison:", json.dumps(report["comparison"]))
    elif not previous:
        print("No baseline supplied; regression comparison unavailable.")
    write_report(output, report)
    return 2 if result.errors else int(suite == "smoke" and bool(result.misses))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("smoke", "benchmark"), default="benchmark")
    parser.add_argument("--index-path", type=Path, default=Path("usearch_index.bin"))
    parser.add_argument("--metadata-path", type=Path, default=Path("metadata.json"))
    parser.add_argument(
        "--eval-path", type=Path, help="Override the selected suite's JSON file"
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--id",
        dest="ids",
        action="append",
        help="Select an ID; repeat for multiple questions",
    )
    selection.add_argument(
        "--changed-since",
        help="Select added/edited questions since the merge base with REF",
    )
    parser.add_argument("--output", type=Path, help="Write a new JSON report")
    parser.add_argument(
        "--baseline", type=Path, help="Compare with a compatible JSON report"
    )
    args = parser.parse_args(argv)
    args.eval_path = args.eval_path or REPO_ROOT / "evals" / f"{args.suite}.json"
    try:
        return evaluate(**vars(args))
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        if args.output and not args.output.exists():
            try:
                write_report(
                    args.output,
                    {
                        "format_version": 1,
                        "suite": args.suite,
                        "status": "error",
                        "error": str(exc),
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
            except OSError as report_error:
                print(f"Could not write error report: {report_error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

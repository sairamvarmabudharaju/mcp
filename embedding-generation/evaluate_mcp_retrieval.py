#!/usr/bin/env python3
"""Evaluate one or more Arm MCP images through real sequential stdio traffic."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import TextIO

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evals.golden.evaluate_golden import (  # noqa: E402
    STRATA,
    best_rank,
    bootstrap_ci,
    compare,
    hit,
    load_golden,
    metrics,
    normalize_url,
    url_base,
)

DEFAULT_GOLDEN = SCRIPT_DIR / "evals" / "golden" / "eval_golden.json"
DEFAULT_HIVEMIND = SCRIPT_DIR / "evals" / "hivemind" / "eval_hivemind.json"
TRANSPORT = "MCP stdio: initialize -> notifications/initialized -> tools/call"


def load_hivemind(path: Path) -> dict:
    with path.open() as file:
        document = json.load(file)
    seen: set[str] = set()
    for item in document["items"]:
        if item["id"] in seen:
            raise ValueError(f"duplicate hive-mind id {item['id']}")
        seen.add(item["id"])
        if not item.get("expected"):
            raise ValueError(f"{item['id']}: expected must not be empty")
    return document


def selected_cases(
    golden_path: Path,
    hivemind_path: Path,
    suites: list[str],
    split: str,
) -> tuple[list[dict], dict]:
    cases: list[dict] = []
    metadata: dict = {}
    if "golden" in suites:
        golden = load_golden(golden_path)
        metadata["golden"] = {
            key: golden[key]
            for key in ("version", "created", "review_status", "corpus", "source")
        }
        for item in golden["items"]:
            if split != "all" and item["split"] != split:
                continue
            cases.append({**item, "suite": "golden", "group": item["area"]})
    if "hivemind" in suites:
        hivemind = load_hivemind(hivemind_path)
        metadata["hivemind"] = {
            key: hivemind[key]
            for key in ("version", "created", "review_status", "source")
        }
        cases.extend({**item, "suite": "hivemind"} for item in hivemind["items"])
    duplicate_ids = {
        item["id"] for item in cases if sum(other["id"] == item["id"] for other in cases) > 1
    }
    if duplicate_ids:
        raise ValueError(f"duplicate ids across suites: {sorted(duplicate_ids)}")
    return cases, metadata


def _read_response(process: subprocess.Popen[str], expected_id: int) -> dict:
    assert process.stdout is not None
    while True:
        line = process.stdout.readline()
        if not line:
            raise RuntimeError(
                f"MCP server exited before response {expected_id}: status={process.poll()}"
            )
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("id") == expected_id:
            return payload


class McpClient:
    """Minimal sequential JSON-RPC client for an MCP Docker image."""

    def __init__(self, docker: str, image: str, server_log: TextIO):
        self.process = subprocess.Popen(
            [docker, "run", "--rm", "-i", image],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=server_log,
            text=True,
            bufsize=1,
        )
        self.responses: list[dict] = []
        initialize = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "arm-mcp-retrieval-eval", "version": "1.0"},
            },
        }
        self._send(initialize)
        self.responses.append(_read_response(self.process, 1))
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _send(self, payload: dict) -> None:
        if self.process.stdin is None:
            raise RuntimeError("MCP server stdin is unavailable")
        self.process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self.process.stdin.flush()

    def search(self, request_id: int, query: str) -> dict:
        self._send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "tools/call",
                "params": {
                    "name": "knowledge_base_search",
                    "arguments": {
                        "query": query,
                        "invocation_reason": "Real-MCP retrieval evaluation",
                    },
                },
            }
        )
        response = _read_response(self.process, request_id)
        self.responses.append(response)
        return response

    def close(self) -> None:
        if self.process.stdin:
            self.process.stdin.close()
        try:
            return_code = self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            return_code = self.process.wait(timeout=10)
        if return_code:
            raise RuntimeError(f"MCP server exited with status {return_code}")


def response_urls(response: dict) -> tuple[list[str | None], object | None]:
    if response.get("error"):
        return [], response["error"]
    result = response.get("result", {})
    if result.get("isError"):
        return [], result
    structured = result.get("structuredContent", {}).get("result")
    if structured is None:
        for content in result.get("content", []):
            if content.get("type") != "text":
                continue
            try:
                structured = json.loads(content["text"])
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            break
    if not isinstance(structured, list):
        return [], "MCP response did not contain a structured result list"
    return [item.get("url") for item in structured], None


def evaluate_image(
    docker: str,
    image: str,
    cases: list[dict],
    top_k: int,
    output_dir: Path,
) -> dict:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", image)
    server_log_path = output_dir / f"{slug}.server.log"
    rows: list[dict] = []
    client: McpClient | None = None
    with server_log_path.open("w") as server_log:
        try:
            client = McpClient(docker, image, server_log)
            for offset, case in enumerate(cases, start=100):
                error = None
                try:
                    response = client.search(offset, case["query"])
                    urls, error = response_urls(response)
                except Exception as exc:
                    urls, error = [], str(exc)
                urls = urls[:top_k]
                rows.append(
                    {
                        "id": case["id"],
                        "suite": case["suite"],
                        "group": case["group"],
                        "query": case["query"],
                        "ranked_urls": urls,
                        "rank_lenient": best_rank(urls, case["expected"], 1),
                        "rank_strict": best_rank(urls, case["expected"], 2),
                        "distinct_pages": len(
                            {url_base(normalize_url(url)) for url in urls}
                        ),
                        "error": error,
                        **{
                            key: case[key]
                            for key in (*STRATA, "confidence")
                            if key in case
                        },
                    }
                )
        finally:
            if client:
                client.close()
    raw_path = output_dir / f"{slug}.responses.ndjson"
    raw_path.write_text(
        "".join(
            json.dumps(response, separators=(",", ":")) + "\n"
            for response in (client.responses if client else [])
        )
    )
    return {
        "image": image,
        "transport": TRANSPORT,
        "top_k": top_k,
        "summary": summarize(rows, top_k),
        "rows": rows,
    }


def metric_summary(rows: list[dict], top_k: int) -> dict:
    result = metrics(rows, top_k)
    if not result:
        return {}
    lo, hi = bootstrap_ci([hit(row["rank_lenient"], top_k) for row in rows])
    return {**result, f"hit@{top_k}_95%_ci": [lo, hi]}


def summarize(rows: list[dict], top_k: int) -> dict:
    summary: dict = {
        "overall": metric_summary(rows, top_k),
        "errors": sum(row["error"] is not None for row in rows),
        "by_suite": {},
    }
    for suite in sorted({row["suite"] for row in rows}):
        selected = [row for row in rows if row["suite"] == suite]
        summary["by_suite"][suite] = metric_summary(selected, top_k)
    golden = [row for row in rows if row["suite"] == "golden"]
    if golden:
        summary["golden_strata"] = {}
        for stratum in STRATA:
            grouped: dict[str, list[dict]] = defaultdict(list)
            for row in golden:
                grouped[row[stratum]].append(row)
            summary["golden_strata"][stratum] = {
                key: metric_summary(value, top_k)
                for key, value in sorted(grouped.items())
            }
    hivemind = [row for row in rows if row["suite"] == "hivemind"]
    if hivemind:
        grouped = defaultdict(list)
        for row in hivemind:
            grouped[row["group"]].append(row)
        summary["hivemind_groups"] = {
            key: metric_summary(value, top_k)
            for key, value in sorted(grouped.items())
        }
    return summary


def compare_runs(baseline: dict, candidate: dict, top_k: int) -> dict:
    by_suite: dict = {}
    for suite in sorted({row["suite"] for row in candidate["rows"]}):
        old = [row for row in baseline["rows"] if row["suite"] == suite]
        new = [row for row in candidate["rows"] if row["suite"] == suite]
        diff = compare(new, old, top_k)
        old_by_id = {row["id"]: row for row in old}
        hit_gained = []
        hit_lost = []
        for row in new:
            old_rank = old_by_id[row["id"]]["rank_lenient"]
            new_rank = row["rank_lenient"]
            old_hit = old_rank is not None and old_rank <= top_k
            new_hit = new_rank is not None and new_rank <= top_k
            detail = (row["id"], old_rank, new_rank, row["query"])
            if new_hit and not old_hit:
                hit_gained.append(detail)
            elif old_hit and not new_hit:
                hit_lost.append(detail)
        by_suite[suite] = {
            "compared": diff["compared"],
            f"hit@{top_k}_gained": len(hit_gained),
            f"hit@{top_k}_lost": len(hit_lost),
            "rank_improved": len(diff["improved"]),
            "rank_regressed": len(diff["regressed"]),
            "p_value": diff["p_value"],
            f"hit@{top_k}_gained_cases": hit_gained,
            f"hit@{top_k}_lost_cases": hit_lost,
            "rank_improved_cases": diff["improved"],
            "rank_regressed_cases": diff["regressed"],
        }
    return {
        "baseline": baseline["image"],
        "candidate": candidate["image"],
        "by_suite": by_suite,
    }


def pct(value: float) -> str:
    return f"{value:.1%}"


def write_report(runs: list[dict], comparisons: list[dict], path: Path) -> None:
    top_k = runs[0]["top_k"]
    lines = [
        "# Real-MCP retrieval evaluation",
        "",
        f"Transport: {TRANSPORT}.",
        "",
        f"| Image | Suite | n | H@1 | H@3 | H@{top_k} | strict H@{top_k} | MRR |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        for suite, values in run["summary"]["by_suite"].items():
            lines.append(
                f"| `{run['image']}` | {suite} | {values['n']} | "
                f"{pct(values['hit@1'])} | {pct(values['hit@3'])} | "
                f"{pct(values[f'hit@{top_k}'])} | "
                f"{pct(values[f'strict_hit@{top_k}'])} | "
                f"{values['mrr']:.3f} |"
            )
    for comparison in comparisons:
        lines.extend(
            [
                "",
                f"## {comparison['candidate']} vs {comparison['baseline']}",
                "",
                f"| Suite | Paired | H@{top_k} gained | H@{top_k} lost | "
                "Rank better | Rank worse | sign-flip p |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for suite, values in comparison["by_suite"].items():
            lines.append(
                f"| {suite} | {values['compared']} | "
                f"{values[f'hit@{top_k}_gained']} | "
                f"{values[f'hit@{top_k}_lost']} | "
                f"{values['rank_improved']} | {values['rank_regressed']} | "
                f"{values['p_value']:.3f} |"
            )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run broad and targeted retrieval suites through real MCP stdio."
    )
    parser.add_argument("--image", action="append", required=True)
    parser.add_argument(
        "--suite",
        action="append",
        choices=("golden", "hivemind"),
        help="Suite to run; repeat to select both (default: both).",
    )
    parser.add_argument("--golden-path", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--hivemind-path", type=Path, default=DEFAULT_HIVEMIND)
    parser.add_argument("--split", choices=("all", "dev", "holdout"), default="all")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--docker", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    docker = args.docker or shutil.which("docker")
    if not docker:
        raise SystemExit("docker executable not found; use --docker")
    suites = list(dict.fromkeys(args.suite or ["golden", "hivemind"]))
    cases, eval_metadata = selected_cases(
        args.golden_path, args.hivemind_path, suites, args.split
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for image in args.image:
        print(f"Evaluating {image}: {len(cases)} queries over real MCP stdio")
        run = evaluate_image(docker, image, cases, args.top_k, args.output_dir)
        runs.append(run)
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", image)
        (args.output_dir / f"{slug}.evaluated.json").write_text(
            json.dumps({**run, "eval_metadata": eval_metadata}, indent=2) + "\n"
        )
        for suite, values in run["summary"]["by_suite"].items():
            print(
                f"  {suite}: n={values['n']} H@1={values['hit@1']:.1%} "
                f"H@3={values['hit@3']:.1%} "
                f"H@{args.top_k}={values[f'hit@{args.top_k}']:.1%} "
                f"strict={values[f'strict_hit@{args.top_k}']:.1%} "
                f"MRR={values['mrr']:.3f}"
            )
    comparisons = [
        compare_runs(runs[0], candidate, args.top_k) for candidate in runs[1:]
    ]
    comparison_document = {
        "transport": TRANSPORT,
        "top_k": args.top_k,
        "eval_metadata": eval_metadata,
        "comparisons": comparisons,
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison_document, indent=2) + "\n"
    )
    write_report(runs, comparisons, args.output_dir / "comparison.md")
    return 1 if any(run["summary"]["errors"] for run in runs) else 0


if __name__ == "__main__":
    raise SystemExit(main())

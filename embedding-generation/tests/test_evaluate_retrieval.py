"""Exercise the shared runner without loading a real embedding model."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from arm_kb_search.evaluation import evaluate_retrieval, url_base

import evaluate_retrieval as runner


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    rows = [
        {
            "id": "Q1",
            "question": "first",
            "topic": "cloud",
            "intent": "setup",
            "expected_urls": ["https://example.com/guide#answer"],
        },
        {
            "id": "Q2",
            "question": "second",
            "topic": "cloud",
            "intent": "reference",
            "expected_urls": ["https://example.com/second"],
        },
    ]
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps(rows))
    metadata = tmp_path / "metadata.json"
    metadata.write_text("[]")
    index = tmp_path / "index.bin"
    index.write_bytes(b"index")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    # These two external operations load a model and perform inference; all
    # selection, validation, scoring, comparison, reporting and exit logic is real.
    monkeypatch.setattr(
        runner,
        "load_search_resources",
        lambda **kwargs: SimpleNamespace(metadata=[{}], usearch_index=[0]),
    )
    monkeypatch.setattr(
        runner,
        "search",
        lambda question, resources, k: [
            {
                "url": "https://example.com/guide#answer"
                if question == "first"
                else "https://example.com/wrong"
            }
        ],
    )
    args = [
        "--eval-path",
        str(suite),
        "--metadata-path",
        str(metadata),
        "--index-path",
        str(index),
        "--model-path",
        str(model),
    ]
    return rows, suite, args


@pytest.mark.parametrize("suite,expected_exit", [("smoke", 1), ("benchmark", 0)])
def test_same_miss_gates_only_smoke(inputs, tmp_path, suite, expected_exit):
    _, _, args = inputs
    output = tmp_path / "result.json"
    assert (
        runner.main([*args, "--suite", suite, "--output", str(output)]) == expected_exit
    )
    report = json.loads(output.read_text())
    assert report["summary"]["hit_at_5"] == 0.5
    assert report["summary"]["mrr"] == 0.5
    assert report["summary"]["misses"] == 1
    assert report["summary"]["errors"] == 0
    assert report["by_topic"]["cloud"]["total"] == 2


@pytest.mark.parametrize("suite", ["smoke", "benchmark"])
def test_query_errors_fail_both_suites(inputs, monkeypatch, tmp_path, suite):
    def failed_search(*args, **kwargs):
        raise RuntimeError("inference failed")

    monkeypatch.setattr(runner, "search", failed_search)
    output = tmp_path / "error.json"
    assert runner.main([*inputs[2], "--suite", suite, "--output", str(output)]) == 2
    report = json.loads(output.read_text())
    assert report["summary"]["errors"] == 2
    assert report["summary"]["misses"] == 0
    assert report["summary"]["hit_at_5"] == 0


@pytest.mark.parametrize(
    "suite,actual,expected,hit",
    [
        (
            "smoke",
            "https://example.com/guide/chapter#x",
            "https://example.com/guide",
            True,
        ),
        (
            "smoke",
            "https://example.com/guide-other",
            "https://example.com/guide",
            False,
        ),
        ("smoke", "https://elsewhere.com/guide", "https://example.com/guide", False),
        (
            "benchmark",
            "https://EXAMPLE.com/guide/?b=2&a=1&utm_source=mcp#answer",
            "https://example.com/guide?a=1&b=2#answer",
            True,
        ),
        (
            "benchmark",
            "https://example.com/guide#other",
            "https://example.com/guide#answer",
            False,
        ),
        (
            "benchmark",
            "https://example.com/?package=redis",
            "https://example.com/?package=mongo",
            False,
        ),
        (
            "benchmark",
            "https://example.com/#q=svcntb",
            "https://example.com/#q=svcntw",
            False,
        ),
        (
            "benchmark",
            "https://example.com/windows/redis",
            "https://example.com/linux/redis",
            False,
        ),
        (
            "benchmark",
            "https://example.com/guide/chapter",
            "https://example.com/guide",
            False,
        ),
    ],
)
def test_suite_matching(inputs, monkeypatch, tmp_path, suite, actual, expected, hit):
    rows, path, args = inputs
    rows[0]["expected_urls"] = [expected]
    path.write_text(json.dumps(rows[:1]))
    monkeypatch.setattr(runner, "search", lambda *args, **kwargs: [{"url": actual}])
    out = tmp_path / "match.json"
    assert runner.main([*args, "--suite", suite, "--output", str(out)]) == (
        1 if suite == "smoke" and not hit else 0
    )
    assert json.loads(out.read_text())["summary"]["hit_at_5"] == int(hit)


def test_legacy_scorer_default_is_unchanged():
    rows = [{"question": "q", "expected_urls": ["https://example.com/p#expected"]}]
    result = evaluate_retrieval(rows, lambda q, k: ["https://example.com/p#other"], 5)
    assert result.hit_at_1 == 1
    assert url_base("https://example.com/p?q=1#anchor") == "https://example.com/p"


@pytest.mark.parametrize(
    "returned,expected",
    [([], 0), (["wrong"] * 4 + ["answer"], 1), (["wrong"] * 5 + ["answer"], 0)],
)
def test_rank_five_boundary(inputs, monkeypatch, tmp_path, returned, expected):
    rows, path, args = inputs
    rows[0]["expected_urls"] = ["https://example.com/answer"]
    path.write_text(json.dumps(rows[:1]))
    monkeypatch.setattr(
        runner,
        "search",
        lambda *a, **kw: [{"url": "https://example.com/" + u} for u in returned],
    )
    out = tmp_path / "rank.json"
    assert runner.main([*args, "--output", str(out)]) == 0
    assert json.loads(out.read_text())["summary"]["hit_at_5"] == expected


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [{}],
        [{"id": "Q", "question": "q", "expected_urls": "https://example.com"}],
        [{"id": "Q", "question": "q", "expected_urls": ["relative"]}],
        [{"id": "Q", "question": "q", "expected_urls": ["https://example.com"]}] * 2,
    ],
)
def test_invalid_inputs_fail_before_model_loading(inputs, monkeypatch, rows):
    inputs[1].write_text(json.dumps(rows))
    monkeypatch.setattr(
        runner,
        "load_search_resources",
        lambda **kw: pytest.fail("loaded invalid suite"),
    )
    assert runner.main(inputs[2]) == 2


def test_selection_and_unknown_ids(inputs, tmp_path, monkeypatch):
    out = tmp_path / "selected.json"
    assert (
        runner.main(
            [
                *inputs[2],
                "--suite",
                "smoke",
                "--id",
                "Q1",
                "--id",
                "Q1",
                "--output",
                str(out),
            ]
        )
        == 0
    )
    report = json.loads(out.read_text())
    assert report["summary"]["total"] == 1
    assert report["suite_total"] == 2
    monkeypatch.setattr(
        runner, "load_search_resources", lambda **kw: pytest.fail("loaded unknown ID")
    )
    assert runner.main([*inputs[2], "--id", "missing"]) == 2


def test_broken_model_writes_failure_report(inputs, monkeypatch, tmp_path):
    def broken(**kwargs):
        raise RuntimeError("cannot load model")

    monkeypatch.setattr(runner, "load_search_resources", broken)
    out = tmp_path / "broken.json"
    assert runner.main([*inputs[2], "--output", str(out)]) == 2
    assert json.loads(out.read_text())["status"] == "error"


def test_malformed_result_is_error_not_a_rank_improvement(
    inputs, monkeypatch, tmp_path
):
    monkeypatch.setattr(
        runner,
        "search",
        lambda *a, **kw: [{}, {"url": "https://example.com/guide#answer"}],
    )
    out = tmp_path / "malformed.json"
    assert runner.main([*inputs[2], "--output", str(out)]) == 2
    assert json.loads(out.read_text())["summary"]["errors"] == 2


def test_baseline_reports_regression_and_recovery(inputs, monkeypatch, tmp_path):
    before, after = tmp_path / "before.json", tmp_path / "after.json"
    assert runner.main([*inputs[2], "--output", str(before)]) == 0
    monkeypatch.setattr(
        runner, "search", lambda q, *a, **kw: [{"url": "https://example.com/second"}]
    )
    assert (
        runner.main([*inputs[2], "--baseline", str(before), "--output", str(after)])
        == 0
    )
    comparison = json.loads(after.read_text())["comparison"]
    assert comparison["regressions"] == ["Q1"]
    assert comparison["recoveries"] == ["Q2"]
    assert comparison["delta"]["hit_at_5"] == 0
    # Edits are different evaluation inputs, not a retrieval regression.
    rows, path, args = inputs
    rows[0]["question"] = "edited"
    path.write_text(json.dumps(rows))
    assert runner.main([*args, "--baseline", str(before)]) == 2


def test_small_depth_does_not_claim_hit_at_five(inputs, tmp_path):
    out = tmp_path / "depth.json"
    assert runner.main([*inputs[2], "--top-k", "1", "--output", str(out)]) == 0
    assert json.loads(out.read_text())["summary"]["hit_at_5"] is None
    assert runner.main([*inputs[2], "--top-k", "0"]) == 2


def test_changed_since_uses_merge_base_and_working_tree(
    inputs, tmp_path, monkeypatch, capsys
):
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "-b", "main")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.com")
    path = repo / "suite.json"
    rows = inputs[0]
    path.write_text(json.dumps(rows))
    git("add", ".")
    git("commit", "-m", "base")
    git("checkout", "-b", "feature")
    monkeypatch.setattr(runner, "REPO_ROOT", repo)
    args = inputs[2].copy()
    args[1] = str(path)
    # Formatting and row order alone are not changes.
    path.write_text(json.dumps(list(reversed(rows)), indent=4))
    assert runner.main([*args, "--changed-since", "main"]) == 0
    assert "No changed questions" in capsys.readouterr().out
    # Uncommitted edit + new record; a removed ID is reported, not queried.
    rows[0]["question"] = "edited"
    path.write_text(json.dumps([rows[0], {**rows[1], "id": "Q3"}]))
    out = tmp_path / "changed.json"
    assert runner.main([*args, "--changed-since", "main", "--output", str(out)]) == 0
    report = json.loads(out.read_text())
    assert [c["question_id"] for c in report["cases"]] == ["Q1", "Q3"]
    assert report["removed_ids"] == ["Q2"]
    assert runner.main([*args, "--changed-since", "not-a-ref"]) == 2


@pytest.mark.parametrize(
    "mutation", ["summary", "case", "rank", "policy", "depth", "error"]
)
def test_invalid_baseline_fails_before_loading(inputs, tmp_path, monkeypatch, mutation):
    baseline = tmp_path / "baseline.json"
    assert runner.main([*inputs[2], "--output", str(baseline)]) == 0
    report = json.loads(baseline.read_text())
    if mutation == "summary":
        report["summary"] = "bad"
    elif mutation == "case":
        report["cases"][0] = "bad"
    elif mutation == "rank":
        report["cases"][0]["match_rank"] = -1
    elif mutation == "policy":
        report["policy"] = "different-v1"
    elif mutation == "depth":
        report["top_k"] = 10
    else:
        report["summary"]["errors"] = 1
    baseline.write_text(json.dumps(report))
    monkeypatch.setattr(
        runner,
        "load_search_resources",
        lambda **kw: pytest.fail("loaded malformed baseline"),
    )
    assert runner.main([*inputs[2], "--baseline", str(baseline)]) == 2


def test_wrapper_forwards_selection_and_report_options(tmp_path):
    import os
    import shutil

    wrapper = tmp_path / "run-question-eval.sh"
    shutil.copy(Path(runner.__file__).with_name("run-question-eval.sh"), wrapper)
    (tmp_path / "vector-db-sources.csv").write_text("source\n")
    suite = tmp_path / "suite.json"
    suite.write_text("[]")
    log = tmp_path / "calls.log"
    fake_python = tmp_path / "python"
    fake_python.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$CALL_LOG"\n')
    fake_python.chmod(0o755)
    completed = subprocess.run(
        [
            "bash",
            str(wrapper),
            "--skip-intrinsic-copy",
            "--eval",
            str(suite),
            "--suite",
            "smoke",
            "--id",
            "Q1",
            "--output",
            str(tmp_path / "report.json"),
        ],
        env={**os.environ, "PYTHON": str(fake_python), "CALL_LOG": str(log)},
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    last = log.read_text().splitlines()[-1]
    assert "evaluate_retrieval.py" in last
    assert "--suite smoke --id Q1 --output" in last
    assert f"--eval-path {suite}" in last


@pytest.mark.parametrize(
    "metadata,index", [([], [0]), ([{}], None), ([{}], []), ([{}, {}], [0])]
)
def test_unusable_corpus_fails_setup(inputs, monkeypatch, tmp_path, metadata, index):
    from types import SimpleNamespace

    monkeypatch.setattr(
        runner,
        "load_search_resources",
        lambda **kw: SimpleNamespace(metadata=metadata, usearch_index=index),
    )
    out = tmp_path / "unusable.json"
    assert runner.main([*inputs[2], "--output", str(out)]) == 2
    report = json.loads(out.read_text())
    assert report["status"] == "error"
    assert "cases" not in report


def test_real_empty_index_is_not_a_successful_benchmark(inputs, monkeypatch, tmp_path):
    from types import SimpleNamespace

    import arm_kb_search.resources as resources
    from usearch.index import Index

    index = Path(inputs[2][inputs[2].index("--index-path") + 1])
    Index(ndim=3).save(str(index))
    monkeypatch.setattr(
        resources,
        "load_embedding_model",
        lambda *a, **kw: SimpleNamespace(get_sentence_embedding_dimension=lambda: 3),
    )
    monkeypatch.setattr(
        runner, "load_search_resources", resources.load_search_resources
    )
    monkeypatch.setattr(runner, "search", resources.search)
    out = tmp_path / "empty-corpus.json"
    assert runner.main([*inputs[2], "--output", str(out)]) == 2
    assert json.loads(out.read_text())["status"] == "error"


@pytest.mark.parametrize("invalid", ["suite", "id", "ref", "baseline"])
def test_preflight_failure_preserves_error_artifact(inputs, tmp_path, invalid):
    args = inputs[2].copy()
    if invalid == "suite":
        inputs[1].write_text("[]")
    elif invalid == "id":
        args += ["--id", "no-such-id"]
    elif invalid == "ref":
        args += ["--changed-since", "no-such-ref"]
    else:
        broken = tmp_path / "bad-baseline.json"
        broken.write_text("{}")
        args += ["--baseline", str(broken)]
    out = tmp_path / "invalid.json"
    assert runner.main([*args, "--output", str(out)]) == 2
    assert json.loads(out.read_text())["status"] == "error"


def test_existing_report_is_never_overwritten(inputs, tmp_path):
    out = tmp_path / "existing.json"
    out.write_text("original evidence")
    assert runner.main([*inputs[2], "--output", str(out)]) == 2
    assert out.read_text() == "original evidence"

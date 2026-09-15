# Copyright © 2026, Arm Limited and Contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import evaluate_mcp_retrieval
from evaluate_mcp_retrieval import (
    load_hivemind,
    response_urls,
    selected_cases,
    summarize,
)


def test_shipped_eval_suites_are_valid():
    cases, metadata = selected_cases(
        evaluate_mcp_retrieval.DEFAULT_GOLDEN,
        evaluate_mcp_retrieval.DEFAULT_HIVEMIND,
        ["golden", "hivemind"],
        "all",
    )
    assert len(cases) == 213
    assert {case["suite"] for case in cases} == {"golden", "hivemind"}
    assert metadata["golden"]["review_status"] == "draft"
    assert len(load_hivemind(evaluate_mcp_retrieval.DEFAULT_HIVEMIND)["items"]) == 30


def test_response_urls_prefers_structured_content():
    response = {
        "result": {
            "structuredContent": {
                "result": [{"url": "https://example.com/a"}, {"url": None}]
            }
        }
    }
    assert response_urls(response) == (["https://example.com/a", None], None)


def test_response_urls_falls_back_to_text_content():
    response = {
        "result": {
            "content": [
                {"type": "text", "text": '[{"url":"https://example.com/fallback"}]'}
            ]
        }
    }
    assert response_urls(response) == (["https://example.com/fallback"], None)


def test_summary_separates_broad_and_targeted_suites():
    rows = [
        {
            "id": "G1",
            "suite": "golden",
            "group": "area",
            "form": "agent_query",
            "intent": "howto",
            "area": "area",
            "difficulty": "easy",
            "split": "dev",
            "rank_lenient": 1,
            "rank_strict": 1,
            "distinct_pages": 5,
            "error": None,
        },
        {
            "id": "HM1",
            "suite": "hivemind",
            "group": "targeted",
            "rank_lenient": None,
            "rank_strict": None,
            "distinct_pages": 4,
            "error": None,
        },
    ]
    summary = summarize(rows, 5)
    assert summary["by_suite"]["golden"]["hit@5"] == 1
    assert summary["by_suite"]["hivemind"]["hit@5"] == 0
    assert summary["hivemind_groups"]["targeted"]["n"] == 1

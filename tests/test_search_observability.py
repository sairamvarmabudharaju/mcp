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

import importlib

import numpy as np
import pytest

import arm_kb_search.resources as resources_module
from arm_kb_search.resources import SearchResources

search_module = importlib.import_module("arm_kb_search.search")


def test_url_tokenization_adds_compact_separator_alias():
    tokens = search_module.tokenize_url_for_search("https://example.com/aws-cli/")

    assert "awscli" in tokens


def test_url_content_tokenization_excludes_hostname_boilerplate():
    tokens = search_module.tokenize_url_content_for_search(
        "https://learn.arm.com/install-guides/aws-cli/#install"
    )

    assert "arm" not in tokens
    assert "awscli" in tokens


def _metadata() -> dict:
    return {
        "chunk_uuid": "numpy-arm64",
        "url": "https://example.com/numpy-arm64",
        "title": "NumPy Arm64 compatibility",
        "heading": "Install native wheels",
        "search_text": "Python NumPy supports native Arm64 wheels and source builds.",
        "original_text": "Install the native NumPy wheel.",
        "doc_type": "tutorial",
        "product": "NumPy",
    }


def test_hybrid_search_records_retrieval_ranks_and_score_contributions(monkeypatch):
    metadata = _metadata()
    monkeypatch.setattr(
        search_module,
        "lexical_prepass_search",
        lambda *args, **kwargs: [
            {
                "metadata": metadata,
                "bm25_score": 8.0,
                "lexical_prepass_score": 1.5,
                "lexical_prepass_rank": 1,
                "pinned_lexical": True,
            }
        ],
    )
    monkeypatch.setattr(
        search_module,
        "embedding_search",
        lambda *args, **kwargs: [{"metadata": metadata, "rank": 2, "distance": 0.25}],
    )
    monkeypatch.setattr(
        search_module,
        "bm25_search",
        lambda *args, **kwargs: [{"metadata": metadata, "rank": 3, "bm25_score": 8.0}],
    )

    result = search_module.hybrid_search(
        "NumPy Arm64 compatibility",
        usearch_index=None,
        metadata=[metadata],
        embedding_model=None,
        bm25_index=None,
        k=1,
        candidate_depth=25,
    )[0]

    debug = result["score_debug"]
    assert debug["retrieval"]["dense"]["rank"] == 2
    assert debug["retrieval"]["bm25"]["rank"] == 3
    assert debug["retrieval"]["lexical_prepass"]["rank"] == 1
    assert set(debug["retrieval"]["rrf_contributions"]) == {
        "lexical_prepass",
        "dense",
        "bm25",
    }
    assert debug["pipeline"] == {
        "candidate_depth": 25,
        "lexical_candidates": 1,
        "dense_candidates": 1,
        "bm25_candidates": 1,
        "fused_candidates": 1,
    }
    assert debug["final_score"] == pytest.approx(sum(debug["contributions"].values()))


def test_hybrid_search_fuses_windows_from_the_same_parent(monkeypatch):
    dense_window = {
        **_metadata(),
        "chunk_uuid": "parent__window_1_of_2",
        "parent_chunk_uuid": "parent",
    }
    lexical_window = {
        **_metadata(),
        "chunk_uuid": "parent__window_2_of_2",
        "parent_chunk_uuid": "parent",
    }
    monkeypatch.setattr(
        search_module,
        "lexical_prepass_search",
        lambda *args, **kwargs: [{
            "metadata": lexical_window,
            "lexical_prepass_score": 1.0,
            "lexical_prepass_rank": 1,
            "pinned_lexical": True,
        }],
    )
    monkeypatch.setattr(
        search_module,
        "embedding_search",
        lambda *args, **kwargs: [{"metadata": dense_window, "rank": 1, "distance": 0.2}],
    )
    monkeypatch.setattr(search_module, "bm25_search", lambda *args, **kwargs: [])

    results = search_module.hybrid_search(
        "NumPy Arm64 compatibility",
        usearch_index=None,
        metadata=[dense_window, lexical_window],
        embedding_model=None,
        bm25_index=None,
        k=5,
    )

    assert len(results) == 1
    assert set(results[0]["score_debug"]["retrieval"]["rrf_contributions"]) == {
        "lexical_prepass",
        "dense",
    }


def test_bm25_search_collapses_sibling_windows_and_preserves_raw_rank():
    metadata = [
        {**_metadata(), "chunk_uuid": "parent-a-1", "parent_chunk_uuid": "parent-a"},
        {**_metadata(), "chunk_uuid": "parent-a-2", "parent_chunk_uuid": "parent-a"},
        {**_metadata(), "chunk_uuid": "parent-b-1", "parent_chunk_uuid": "parent-b"},
    ]

    class FakeBm25:
        def get_scores(self, tokens):
            return np.array([3.0, 2.0, 1.0])

    results = search_module.bm25_search("numpy", metadata, FakeBm25(), k=2)

    assert [item["metadata"]["parent_chunk_uuid"] for item in results] == ["parent-a", "parent-b"]
    assert [item["rank"] for item in results] == [1, 2]
    assert [item["raw_rank"] for item in results] == [1, 3]


def test_bm25_index_has_one_document_per_parent():
    metadata = [
        {
            **_metadata(),
            "chunk_uuid": "parent-a-1",
            "parent_chunk_uuid": "parent-a",
            "search_text": "numpy complete parent text",
        },
        {
            **_metadata(),
            "chunk_uuid": "parent-a-2",
            "parent_chunk_uuid": "parent-a",
            "search_text": "sibling-only-noise",
        },
        {
            **_metadata(),
            "chunk_uuid": "parent-b-1",
            "parent_chunk_uuid": "parent-b",
            "search_text": "unrelated second parent",
        },
    ]

    index = search_module.build_bm25_index(metadata)

    assert index is not None
    assert index.metadata_indices.tolist() == [0, 2]
    assert search_module.bm25_search("sibling-only-noise", metadata, index, k=5) == []


def test_bm25_index_adds_compact_url_identifier_aliases():
    expected = {
        **_metadata(),
        "chunk_uuid": "aws-cli",
        "url": "https://example.com/install-guides/aws-cli/",
        "title": "AWS CLI",
        "search_text": "command line installation guide",
    }
    metadata = [
        expected,
        {**_metadata(), "chunk_uuid": "other-1", "title": "Docker Engine", "search_text": "containers"},
        {**_metadata(), "chunk_uuid": "other-2", "title": "Visual Studio", "search_text": "compiler"},
        {**_metadata(), "chunk_uuid": "other-3", "title": "Python Runtime", "search_text": "language"},
    ]

    index = search_module.build_bm25_index(metadata)
    results = search_module.bm25_search("awscli", metadata, index, k=5)

    assert results[0]["metadata"]["chunk_uuid"] == "aws-cli"


def test_public_search_only_returns_debug_details_when_requested(monkeypatch):
    metadata = _metadata()
    candidate = {
        "metadata": metadata,
        "distance": 0.25,
        "rerank_score": 0.75,
        "score_debug": {
            "scoring_profile": "short_query",
            "contributions": {"reciprocal_rank_fusion": 0.75},
            "final_score": 0.75,
            "pre_dedup_rank": 1,
        },
    }
    monkeypatch.setattr(resources_module, "hybrid_search", lambda *args, **kwargs: [candidate])
    resources = SearchResources(
        metadata=[metadata],
        embedding_model=None,
        usearch_index=None,
        bm25_index=None,
        default_k=5,
        include_disclaimers=False,
    )

    normal_result = resources_module.search("NumPy Arm64 compatibility", resources)
    debug_result = resources_module.search(
        "NumPy Arm64 compatibility",
        resources,
        include_debug=True,
    )

    assert "debug" not in normal_result[0]
    assert debug_result[0]["debug"]["result_rank"] == 1
    assert debug_result[0]["debug"]["requested_results"] == 5
    assert debug_result[0]["debug"]["final_score"] == debug_result[0]["score"]


def test_deduplicate_urls_groups_fragments_but_preserves_query_parameters():
    results = [
        {"metadata": {"url": "https://example.com/page/#first"}},
        {"metadata": {"url": "https://example.com/page/#second"}},
        {"metadata": {"url": "https://example.com/page/?package=numpy#details"}},
        {"metadata": {"url": "https://example.com/page?utm_source=mcp&package=numpy#other"}},
        {"metadata": {"url": "https://example.com/page/?package=scipy#details"}},
    ]

    deduplicated = search_module.deduplicate_urls(results)

    assert [item["metadata"]["url"] for item in deduplicated] == [
        "https://example.com/page/#first",
        "https://example.com/page/?package=numpy#details",
        "https://example.com/page/?package=scipy#details",
    ]


def test_deduplicate_urls_preserves_semantic_query_fragments():
    results = [
        {"metadata": {"url": "https://example.com/intrinsics/#q=vaddq_f32"}},
        {"metadata": {"url": "https://example.com/intrinsics/#q=vsubq_f32"}},
    ]

    deduplicated = search_module.deduplicate_urls(results)

    assert deduplicated == results


def test_install_guides_receive_install_intent_bonus():
    base_candidate = {
        "rrf_score": 0.0,
        "distance": None,
        "bm25_score": 0.0,
        "lexical_prepass_score": 0.0,
    }
    install_guide = {
        **base_candidate,
        "metadata": {
            **_metadata(),
            "chunk_uuid": "docker-guide",
            "url": "https://learn.arm.com/install-guides/docker/",
            "title": "Docker",
            "doc_type": "Install Guides",
        },
    }
    tutorial = {
        **base_candidate,
        "metadata": {
            **_metadata(),
            "chunk_uuid": "docker-tutorial",
            "url": "https://example.com/docker/",
            "title": "Docker",
            "doc_type": "Tutorial",
        },
    }

    ranked = search_module.rerank_candidates(
        "How do I install Docker?",
        [tutorial, install_guide],
    )

    assert ranked[0]["metadata"]["chunk_uuid"] == "docker-guide"
    assert ranked[0]["score_debug"]["contributions"]["document_type"] == pytest.approx(0.35)


def test_compatibility_query_boosts_exact_dashboard_package():
    dashboard = {
        "metadata": {
            **_metadata(),
            "chunk_uuid": "numpy-dashboard",
            "url": "https://www.arm.com/developer-hub/ecosystem-dashboard/?package=numpy",
            "title": "Ecosystem Dashboard - Numpy",
            "doc_type": "Ecosystem Dashboard",
        },
        "rrf_score": 0.0,
        "distance": None,
        "bm25_score": 0.0,
        "lexical_prepass_score": 0.0,
    }

    ranked = search_module.rerank_candidates(
        "python numpy arm64 compatibility",
        [dashboard],
    )

    assert ranked[0]["score_debug"]["contributions"]["dashboard_package_match"] == pytest.approx(0.30)


def test_compatibility_query_does_not_boost_different_dashboard_package():
    dashboard = {
        "metadata": {
            **_metadata(),
            "chunk_uuid": "scipy-dashboard",
            "url": "https://www.arm.com/developer-hub/ecosystem-dashboard/?package=scipy",
            "title": "Ecosystem Dashboard - Scipy",
            "doc_type": "Ecosystem Dashboard",
        },
        "rrf_score": 0.0,
        "distance": None,
        "bm25_score": 0.0,
        "lexical_prepass_score": 0.0,
    }

    ranked = search_module.rerank_candidates(
        "python numpy arm64 compatibility",
        [dashboard],
    )

    assert ranked[0]["score_debug"]["contributions"]["dashboard_package_match"] == 0.0

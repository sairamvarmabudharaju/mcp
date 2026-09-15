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


def _windows() -> list[dict]:
    """Two embedding windows of one parent chunk; window 1 carries the full text."""
    window_one = {
        **_metadata(),
        "chunk_uuid": "parent__window_1_of_2",
        "parent_chunk_uuid": "parent",
        "chunk_index": 1,
        "chunk_count": 2,
        "original_text": "Document Title: NumPy\n\nFull parent text about NumPy wheels and source builds.",
        "search_text": "NumPy full parent text about NumPy wheels and source builds",
    }
    window_two = {
        **_metadata(),
        "chunk_uuid": "parent__window_2_of_2",
        "parent_chunk_uuid": "parent",
        "chunk_index": 2,
        "chunk_count": 2,
        "original_text": "source builds.",
        "search_text": "NumPy source builds",
    }
    return [window_one, window_two]


def _candidate(metadata: dict) -> dict:
    return {
        "metadata": metadata,
        "rrf_score": 0.0,
        "distance": None,
        "bm25_score": 0.0,
        "lexical_prepass_score": 0.0,
    }


def _patch_retrievers(monkeypatch, lexical=(), dense=(), sparse=()):
    monkeypatch.setattr(search_module, "lexical_prepass_search", lambda *a, **k: list(lexical))
    monkeypatch.setattr(search_module, "embedding_search", lambda *a, **k: list(dense))
    monkeypatch.setattr(search_module, "bm25_search", lambda *a, **k: list(sparse))


# --- tokenisation -----------------------------------------------------------


def test_url_tokenization_adds_compact_separator_alias():
    assert "awscli" in search_module.tokenize_url_for_search("https://example.com/aws-cli/")


def test_url_content_tokenization_excludes_hostname_boilerplate():
    tokens = search_module.tokenize_url_content_for_search("https://learn.arm.com/install-guides/aws-cli/#install")

    assert "arm" not in tokens
    assert "awscli" in tokens


def test_query_normalization_splits_camel_case_identifiers():
    normalized = search_module.normalize_query_for_search("GoogleChrome installation guide")

    assert normalized == "googlechrome google chrome installation guide"


def test_query_normalization_preserves_compact_technical_identifiers():
    normalized = search_module.normalize_query_for_search("TensorFlow int8x16_t")

    # The exact identifiers stay; word parts are added; single characters (the
    # ``t`` of ``int8x16_t``) are not, because they would match everything.
    assert normalized == "tensorflow tensor flow int8x16_t int8x16"


def test_query_normalization_preserves_leading_underscore_identifiers():
    assert search_module.normalize_query_for_search("_mm_shuffle_epi8") == "_mm_shuffle_epi8 mm shuffle epi8"
    assert search_module.normalize_query_for_search("__crc32b") == "__crc32b crc32b"


def test_query_normalization_canonicalizes_general_setup_phrase():
    normalized = search_module.normalize_query_for_search("Set up Docker on an Arm-based instance")

    assert normalized == "setup docker on an arm based instance"


def test_google_provider_detection_requires_a_provider_phrase():
    assert not search_module._is_google_provider_query({"google", "chrome"})
    assert not search_module._is_google_provider_query({"docker", "engine"})
    assert search_module._is_google_provider_query({"google", "cloud"})
    assert search_module._is_google_provider_query({"compute", "engine"})
    assert search_module._is_google_provider_query({"gke"})



def test_long_query_exact_bonus_requires_a_non_generic_entity():
    generic = {
        **_metadata(),
        "title": "Run a web app on an instance",
        "url": "https://example.com/web-app/",
        "doc_type": "learning path",
    }

    result = search_module.rerank_candidates(
        "run my web app on an instance", [_candidate(generic)]
    )[0]

    assert result["score_debug"]["query_analysis"]["entity_tokens"] == []
    assert result["score_debug"]["contributions"].get("exact_entity", 0.0) == 0.0


def test_direct_phrase_scoring_prefers_a_matching_heading_anchor():
    base = {
        **_metadata(),
        "title": "AWS CLI",
        "search_text": "AWS command line interface installation guide",
        "doc_type": "install guide",
    }
    overview = {
        **base,
        "chunk_uuid": "aws-overview",
        "url": "https://example.com/install-guides/aws-cli/#about",
        "heading": "About AWS CLI",
    }
    installation = {
        **base,
        "chunk_uuid": "aws-installation",
        "url": "https://example.com/install-guides/aws-cli/#install",
        "heading": "How do I download and install AWS CLI version 2?",
    }

    results = search_module.rerank_candidates(
        "install aws cli", [_candidate(overview), _candidate(installation)]
    )

    assert results[0]["metadata"]["chunk_uuid"] == "aws-installation"


# --- fusion and parent resolution ------------------------------------------


def test_hybrid_search_records_retrieval_ranks_and_score_contributions(monkeypatch):
    metadata = _metadata()
    _patch_retrievers(
        monkeypatch,
        lexical=[{
            "metadata": metadata,
            "bm25_score": 8.0,
            "lexical_prepass_score": 1.5,
            "lexical_prepass_rank": 1,
            "pinned_lexical": True,
        }],
        dense=[{"metadata": metadata, "rank": 2, "distance": 0.25}],
        sparse=[{"metadata": metadata, "rank": 3, "bm25_score": 8.0}],
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
    assert set(debug["retrieval"]["rrf_contributions"]) == {"lexical_prepass", "dense", "bm25"}
    assert debug["pipeline"] == {
        "candidate_depth": 25,
        "lexical_candidates": 1,
        "dense_candidates": 1,
        "bm25_candidates": 1,
        "fused_candidates": 1,
    }
    assert debug["final_score"] == pytest.approx(sum(debug["contributions"].values()))


def test_rerank_lexical_exactness_ignores_cached_prepass_score():
    metadata = {
        **_metadata(),
        "chunk_uuid": "docker-guide",
        "url": "https://learn.arm.com/install-guides/docker/",
        "title": "Docker",
        "search_text": "Install Docker Engine on Arm Linux",
    }
    unpinned = search_module.rerank_candidates(
        "setup docker on arm", [_candidate(metadata)]
    )[0]
    cached_candidate = _candidate(metadata)
    cached_candidate["lexical_prepass_score"] = 99.0
    cached = search_module.rerank_candidates(
        "setup docker on arm", [cached_candidate]
    )[0]

    unpinned_bonus = unpinned["score_debug"]["contributions"]["lexical_prepass"]
    cached_bonus = cached["score_debug"]["contributions"]["lexical_prepass"]
    assert cached_bonus == pytest.approx(unpinned_bonus)


def test_hybrid_search_fuses_windows_from_the_same_parent(monkeypatch):
    window_one, window_two = _windows()
    _patch_retrievers(
        monkeypatch,
        lexical=[{"metadata": window_one, "lexical_prepass_score": 1.0, "lexical_prepass_rank": 1, "pinned_lexical": True}],
        dense=[{"metadata": window_two, "rank": 1, "distance": 0.2}],
    )

    results = search_module.hybrid_search(
        "NumPy Arm64 compatibility",
        usearch_index=None,
        metadata=[window_one, window_two],
        embedding_model=None,
        bm25_index=None,
        k=5,
    )

    assert len(results) == 1
    assert set(results[0]["score_debug"]["retrieval"]["rrf_contributions"]) == {"lexical_prepass", "dense"}


def test_hybrid_search_scores_every_window_through_its_parent(monkeypatch):
    """A parent found through window 2 must score and display exactly like one found through window 1."""
    window_one, window_two = _windows()
    metadata = [window_one, window_two]
    parent_index = search_module.build_parent_index(metadata)

    def run(found_window):
        _patch_retrievers(monkeypatch, dense=[{"metadata": found_window, "rank": 1, "distance": 0.2}])
        return search_module.hybrid_search(
            "numpy wheels source builds",
            usearch_index=None,
            metadata=metadata,
            embedding_model=None,
            bm25_index=None,
            k=5,
            parent_index=parent_index,
        )[0]

    via_window_two = run(window_two)
    via_window_one = run(window_one)

    assert via_window_two["metadata"] is window_one
    assert via_window_two["rerank_score"] == pytest.approx(via_window_one["rerank_score"])
    assert via_window_two["score_debug"]["retrieval"]["matched_window"]["chunk_index"] == 2
    assert via_window_one["score_debug"]["retrieval"]["matched_window"] is None


def test_build_parent_index_prefers_the_first_window():
    window_one, window_two = _windows()

    parent_index = search_module.build_parent_index([window_two, window_one, _metadata()])

    assert parent_index["parent"] is window_one
    assert parent_index["numpy-arm64"]["chunk_uuid"] == "numpy-arm64"


# --- BM25 --------------------------------------------------------------------


def test_bm25_index_has_one_document_per_parent():
    metadata = [
        {**_metadata(), "chunk_uuid": "parent-a-1", "parent_chunk_uuid": "parent-a", "search_text": "numpy complete parent text"},
        {**_metadata(), "chunk_uuid": "parent-a-2", "parent_chunk_uuid": "parent-a", "search_text": "sibling-only-noise"},
        {**_metadata(), "chunk_uuid": "parent-b-1", "parent_chunk_uuid": "parent-b", "search_text": "unrelated second parent"},
    ]

    index = search_module.build_bm25_index(metadata)

    assert index is not None
    assert index.metadata_indices.tolist() == [0, 2]
    assert search_module.bm25_search("sibling-only-noise", metadata, index, k=5) == []


def test_bm25_index_uses_first_window_when_metadata_is_reordered():
    window_one, window_two = _windows()

    index = search_module.build_bm25_index([window_two, window_one])

    assert index is not None
    assert index.metadata_indices.tolist() == [1]
    assert "full" in index.index.doc_freqs[0]
    assert "continuation" not in index.index.doc_freqs[0]


def test_parent_aware_bm25_scatters_scores_to_metadata_positions():
    metadata = [
        {**_metadata(), "chunk_uuid": "parent-a-1", "parent_chunk_uuid": "parent-a", "search_text": "numpy wheels"},
        {**_metadata(), "chunk_uuid": "parent-a-2", "parent_chunk_uuid": "parent-a", "search_text": "numpy wheels"},
        {**_metadata(), "chunk_uuid": "parent-b-1", "parent_chunk_uuid": "parent-b", "search_text": "docker engine"},
        {**_metadata(), "chunk_uuid": "parent-c-1", "parent_chunk_uuid": "parent-c", "search_text": "visual studio"},
        {**_metadata(), "chunk_uuid": "parent-d-1", "parent_chunk_uuid": "parent-d", "search_text": "python runtime"},
    ]

    scores = search_module.build_bm25_index(metadata).get_scores(["numpy"])

    assert scores.shape == (5,)
    assert scores[0] > 0
    assert scores[1:].tolist() == [0, 0, 0, 0]


def test_bm25_search_ranks_by_score_and_stops_at_zero():
    metadata = [_metadata(), {**_metadata(), "chunk_uuid": "b"}, {**_metadata(), "chunk_uuid": "c"}]

    class FakeBm25:
        def get_scores(self, tokens):
            return np.array([1.0, 3.0, 0.0])

    results = search_module.bm25_search("numpy", metadata, FakeBm25(), k=3)

    assert [item["metadata"]["chunk_uuid"] for item in results] == ["b", "numpy-arm64"]
    assert [item["rank"] for item in results] == [1, 2]


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


# --- dense retrieval ---------------------------------------------------------


def test_embedding_search_deepens_until_k_unique_parents():
    metadata = [{**_metadata(), "chunk_uuid": f"a-{i}", "parent_chunk_uuid": "parent-a"} for i in range(4)]
    metadata.append({**_metadata(), "chunk_uuid": "b-1", "parent_chunk_uuid": "parent-b"})
    metadata.append({**_metadata(), "chunk_uuid": "c-1", "parent_chunk_uuid": "parent-c"})
    requested_depths = []

    class Matches:
        def __init__(self, keys, distances):
            self.keys = np.array(keys)
            self.distances = np.array(distances)

    class FakeIndex:
        def search(self, embedding, depth, exact=False):
            requested_depths.append(depth)
            keys = list(range(len(metadata)))[:depth]
            return Matches(keys, [0.1 * (position + 1) for position in range(len(keys))])

    class FakeModel:
        def encode(self, texts):
            return np.zeros((len(texts), 1))

    results = search_module.embedding_search("numpy", FakeIndex(), metadata, FakeModel(), k=2)

    assert requested_depths == [4, 6]
    assert [item["metadata"]["parent_chunk_uuid"] for item in results] == ["parent-a", "parent-b"]
    assert [item["rank"] for item in results] == [1, 2]
    assert [item["raw_rank"] for item in results] == [1, 5]


# --- public search -----------------------------------------------------------


def _resources(metadata: list[dict]) -> SearchResources:
    return SearchResources(
        metadata=metadata,
        embedding_model=None,
        usearch_index=None,
        bm25_index=None,
        default_k=5,
        include_disclaimers=False,
        parent_index=search_module.build_parent_index(metadata),
    )


def test_public_search_only_returns_debug_details_when_requested(monkeypatch):
    metadata = _metadata()
    candidate = {
        "metadata": metadata,
        "distance": 0.25,
        "rerank_score": 0.75,
        "score_debug": {"scoring_profile": "short_query", "contributions": {"reciprocal_rank_fusion": 0.75}, "final_score": 0.75, "pre_dedup_rank": 1},
    }
    monkeypatch.setattr(resources_module, "hybrid_search", lambda *args, **kwargs: [candidate])
    resources = _resources([metadata])

    normal_result = resources_module.search("NumPy Arm64 compatibility", resources)
    debug_result = resources_module.search("NumPy Arm64 compatibility", resources, include_debug=True)

    assert "debug" not in normal_result[0]
    assert debug_result[0]["debug"]["result_rank"] == 1
    assert debug_result[0]["debug"]["requested_results"] == 5
    assert debug_result[0]["debug"]["final_score"] == debug_result[0]["score"]


def test_public_search_normalizes_identifiers_in_the_query(monkeypatch):
    metadata = {
        **_metadata(),
        "chunk_uuid": "aws-cli",
        "url": "https://example.com/install-guides/aws-cli/",
        "title": "AWS CLI",
        "keywords": "AWS CLI; install",
    }
    candidate = {
        "metadata": metadata,
        "distance": 0.25,
        "rerank_score": 0.75,
        "score_debug": {},
    }
    observed_queries = []

    def fake_hybrid_search(query, *args, **kwargs):
        observed_queries.append(query)
        return [candidate]

    monkeypatch.setattr(resources_module, "hybrid_search", fake_hybrid_search)
    resources = _resources([metadata])

    result = resources_module.search(
        "AWSCLI ARM64 setup", resources, include_debug=True
    )[0]

    assert observed_queries == ["awscli arm64 setup"]
    assert result["debug"]["query_normalization"] == {
        "original": "AWSCLI ARM64 setup",
        "normalized": "awscli arm64 setup",
    }


def test_public_search_returns_the_full_parent_text_for_a_window_hit(monkeypatch):
    window_one, window_two = _windows()
    _patch_retrievers(monkeypatch, dense=[{"metadata": window_two, "rank": 1, "distance": 0.2}])

    result = resources_module.search("numpy wheels", _resources([window_one, window_two]), include_debug=True)[0]

    assert result["snippet"] == window_one["original_text"]
    assert result["debug"]["retrieval"]["matched_window"]["chunk_index"] == 2


def test_public_search_widens_the_pool_when_dedup_collapses_it(monkeypatch):
    same_page = [
        {"metadata": {**_metadata(), "chunk_uuid": f"same-{i}", "url": f"https://example.com/page/#s{i}"}, "rerank_score": 1.0, "score_debug": {}}
        for i in range(50)
    ]
    other_pages = [
        {"metadata": {**_metadata(), "chunk_uuid": f"other-{i}", "url": f"https://example.com/other-{i}"}, "rerank_score": 0.5, "score_debug": {}}
        for i in range(4)
    ]
    requested = []

    def fake_hybrid_search(*args, **kwargs):
        requested.append(kwargs["k"])
        return same_page + other_pages if kwargs["k"] > 50 else same_page

    monkeypatch.setattr(resources_module, "hybrid_search", fake_hybrid_search)

    results = resources_module.search("numpy", _resources([]))

    assert requested == [50, 200]
    assert len(results) == 5


# --- deduplication -----------------------------------------------------------


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

    assert search_module.deduplicate_urls(results) == results


def test_deduplicate_urls_groups_by_resolved_url():
    results = [
        {"metadata": {"url": "https://example.com/old-name/", "resolved_url": "https://example.com/page/"}},
        {"metadata": {"url": "https://example.com/page/#section", "resolved_url": "https://example.com/page/#section"}},
    ]

    assert len(search_module.deduplicate_urls(results)) == 1


# --- reranking bonuses -------------------------------------------------------


def _install_guide(slug: str, title: str) -> dict:
    return _candidate({
        **_metadata(),
        "chunk_uuid": f"{slug}-guide",
        "url": f"https://learn.arm.com/install-guides/{slug}/",
        "title": title,
        "doc_type": "Install Guides",
        "keywords": "",
        "product": "",
    })


def test_install_guides_receive_install_intent_bonus():
    tutorial = _candidate({**_metadata(), "chunk_uuid": "docker-tutorial", "url": "https://example.com/docker/", "title": "Docker", "doc_type": "Tutorial"})

    ranked = search_module.rerank_candidates("How do I install Docker?", [tutorial, _install_guide("docker", "Docker")])

    assert ranked[0]["metadata"]["chunk_uuid"] == "docker-guide"
    assert ranked[0]["score_debug"]["contributions"]["document_type"] == pytest.approx(0.35)


def test_install_bonus_requires_the_named_product():
    ranked = search_module.rerank_candidates(
        "How do I install Docker on Ubuntu?",
        [_install_guide("gfortran", "GFortran"), _install_guide("docker", "Docker")],
    )
    by_uuid = {item["metadata"]["chunk_uuid"]: item for item in ranked}

    assert by_uuid["docker-guide"]["score_debug"]["contributions"]["document_type"] == pytest.approx(0.35)
    assert by_uuid["gfortran-guide"]["score_debug"]["contributions"]["document_type"] == pytest.approx(0.10)


def test_setup_intent_receives_the_same_product_gated_install_bonus():
    ranked = search_module.rerank_candidates(
        "setup docker on an arm instance",
        [_install_guide("gfortran", "GFortran"), _install_guide("docker", "Docker")],
    )
    by_uuid = {item["metadata"]["chunk_uuid"]: item for item in ranked}

    assert by_uuid["docker-guide"]["score_debug"]["contributions"]["document_type"] == pytest.approx(0.35)
    assert by_uuid["gfortran-guide"]["score_debug"]["contributions"]["document_type"] == pytest.approx(0.10)


def test_install_bonus_matches_headings_but_not_platform_words():
    java_guide = _install_guide("java", "Java")
    java_guide["metadata"]["heading"] = "Install the Microsoft Build of OpenJDK"
    compiler_guide = _install_guide("acfl", "Arm Compiler for Linux")

    ranked = search_module.rerank_candidates("openjdk install arm linux", [compiler_guide, java_guide])
    by_uuid = {item["metadata"]["chunk_uuid"]: item for item in ranked}

    assert by_uuid["java-guide"]["score_debug"]["contributions"]["document_type"] == pytest.approx(0.35)
    assert by_uuid["acfl-guide"]["score_debug"]["contributions"]["document_type"] == pytest.approx(0.10)


def _dashboard(package: str) -> dict:
    return _candidate({
        **_metadata(),
        "chunk_uuid": f"{package}-dashboard",
        "url": f"https://www.arm.com/developer-hub/ecosystem-dashboard/?package={package}",
        "title": f"Ecosystem Dashboard - {package}",
        "doc_type": "Ecosystem Dashboard",
    })


def test_compatibility_query_boosts_exact_dashboard_package():
    ranked = search_module.rerank_candidates("python numpy arm64 compatibility", [_dashboard("numpy")])

    assert ranked[0]["score_debug"]["contributions"]["dashboard_package_match"] == pytest.approx(0.30)


def test_compatibility_query_does_not_boost_different_dashboard_package():
    ranked = search_module.rerank_candidates("python numpy arm64 compatibility", [_dashboard("scipy")])

    assert ranked[0]["score_debug"]["contributions"]["dashboard_package_match"] == 0.0


def test_dashboard_bonus_ignores_generic_query_tokens():
    ranked = search_module.rerank_candidates("geekbench on arm server", [_dashboard("mongodb-enterprise-server")])

    assert ranked[0]["score_debug"]["contributions"]["dashboard_package_match"] == 0.0


def test_bm25_scores_are_computed_once_and_shared():
    rows = [
        {**_metadata(), "chunk_uuid": "a", "search_text": "install docker on ubuntu arm64"},
        {**_metadata(), "chunk_uuid": "b", "search_text": "docker compose on graviton"},
        {**_metadata(), "chunk_uuid": "c", "search_text": "postgresql tuning guide"},
        {**_metadata(), "chunk_uuid": "d", "search_text": "mysql benchmark on neoverse"},
        {**_metadata(), "chunk_uuid": "e", "search_text": "kubernetes cluster autoscaling"},
    ]
    bm25_index = search_module.build_bm25_index(rows)
    scores = search_module.bm25_query_scores("install docker", bm25_index)

    shared = search_module.bm25_search("install docker", rows, bm25_index, k=3, scores=scores)
    recomputed = search_module.bm25_search("install docker", rows, bm25_index, k=3)
    assert [r["metadata"]["chunk_uuid"] for r in shared] == [r["metadata"]["chunk_uuid"] for r in recomputed] == ["a", "b"]
    prepass_shared = search_module.lexical_prepass_search("install docker", rows, bm25_index, k=2, bm25_scores=scores)
    prepass_recomputed = search_module.lexical_prepass_search("install docker", rows, bm25_index, k=2)
    assert [r["lexical_prepass_score"] for r in prepass_shared] == [r["lexical_prepass_score"] for r in prepass_recomputed]
    assert all("lexical_exactness_score" in r for r in prepass_shared)


@pytest.mark.parametrize("query", ["", "   ", "!!!"])
def test_public_search_rejects_empty_queries_before_retrieval(monkeypatch, query):
    def unexpected_retrieval(*args, **kwargs):
        pytest.fail("Empty queries must not retrieve arbitrary pages")

    monkeypatch.setattr(resources_module, "hybrid_search", unexpected_retrieval)
    assert resources_module.search(query, _resources([_metadata()])) == []


def test_manually_constructed_resources_resolve_compact_window_hits(monkeypatch):
    parent, window = _windows()
    compact_window = {key: window[key] for key in (
        "chunk_uuid", "parent_chunk_uuid", "chunk_index", "chunk_count"
    )}
    _patch_retrievers(monkeypatch, dense=[{
        "metadata": compact_window, "rank": 1, "distance": 0.2
    }])
    resources = SearchResources(
        metadata=[compact_window, parent], embedding_model=None,
        usearch_index=None, bm25_index=None, include_disclaimers=False,
    )

    results = resources_module.search("numpy source builds", resources)

    assert len(results) == 1
    assert results[0]["url"] == parent["url"]
    assert results[0]["snippet"] == parent["original_text"]

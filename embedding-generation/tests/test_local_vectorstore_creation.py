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

import re

import pytest

from local_vectorstore_creation import (
    _fit_embedding_text,
    _window_spans,
    load_local_yaml_files,
    prepare_embedding_records,
)


class WhitespaceTokenizer:
    """One token per whitespace-separated word."""

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False, truncation=False, padding=False):
        del truncation, padding
        matches = list(re.finditer(r"\S+", text))
        result = {"input_ids": list(range(len(matches) + (2 if add_special_tokens else 0)))}
        if return_offsets_mapping:
            result["offset_mapping"] = [match.span() for match in matches]
        return result


class PieceTokenizer(WhitespaceTokenizer):
    """Splits every word into three-character pieces, like a WordPiece vocabulary would."""

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False, truncation=False, padding=False):
        del truncation, padding
        offsets = []
        for match in re.finditer(r"\S+", text):
            for start in range(match.start(), match.end(), 3):
                offsets.append((start, min(start + 3, match.end())))
        result = {"input_ids": list(range(len(offsets) + (2 if add_special_tokens else 0)))}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


class SeamTokenizer(WhitespaceTokenizer):
    """Counts extra tokens once context and window are joined, to simulate seam re-tokenisation."""

    def __call__(self, text, add_special_tokens=True, **kwargs):
        result = super().__call__(text, add_special_tokens=add_special_tokens, **kwargs)
        if add_special_tokens and "\n\n" in text:
            result["input_ids"] = result["input_ids"] + [0, 0, 0, 0]
        return result


class NonlinearSeamTokenizer(WhitespaceTokenizer):
    """Adds a join penalty large enough that subtracting overflow once is insufficient."""

    def __call__(self, text, add_special_tokens=True, **kwargs):
        result = super().__call__(text, add_special_tokens=add_special_tokens, **kwargs)
        if add_special_tokens and "\n\n" in text:
            result["input_ids"] += [0] * (len(result["input_ids"]) // 2)
        return result


def _source(body: str, **overrides) -> dict:
    return {
        "uuid": "source",
        "chunk_uuid": "chunk",
        "url": "https://example.com/page/#section",
        "title": "Example",
        "heading": "Section",
        "heading_path": ["Section"],
        "keywords": "example",
        "content": f"Document Title: Example\nHeading Path: Section\n\n{body}",
        **overrides,
    }


def test_load_local_yaml_files_requires_intrinsic_chunks(tmp_path, monkeypatch):
    (tmp_path / "yaml_data").mkdir()
    (tmp_path / "yaml_data" / "example.yaml").write_text("content: example\n")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError, match="No intrinsic chunk YAML files found"):
        load_local_yaml_files()


def test_prepare_embedding_records_preserves_body_with_overlap():
    body = " ".join(f"word{index}" for index in range(20))

    contents, metadata = prepare_embedding_records([_source(body)], WhitespaceTokenizer(), max_seq_length=10, overlap_tokens=2)

    assert len(contents) > 1
    assert all(item["embedding_token_count"] <= 10 for item in metadata)
    assert all(item["parent_chunk_uuid"] == "chunk" for item in metadata)
    assert metadata[0]["content_start_char"] == 0
    assert metadata[-1]["content_end_char"] == len(body)
    for index in range(len(metadata) - 1):
        assert metadata[index]["content_end_char"] > metadata[index + 1]["content_start_char"]
    embedded_bodies = " ".join(content.split("\n\n", 1)[-1] for content in contents)
    assert all(word in embedded_bodies for word in body.split())


def test_first_window_carries_the_full_source_text():
    body = " ".join(f"word{index}" for index in range(20))
    source = _source(body)

    _, metadata = prepare_embedding_records([source], WhitespaceTokenizer(), max_seq_length=10, overlap_tokens=2)

    assert metadata[0]["chunk_index"] == 1
    assert metadata[0]["original_text"] == source["content"]
    assert body in metadata[0]["search_text"]
    assert "original_text" not in metadata[1]
    assert "search_text" not in metadata[1]
    assert "url" not in metadata[1]
    assert metadata[1]["parent_chunk_uuid"] == "chunk"


def test_prepare_embedding_records_keeps_short_source_as_one_record():
    source = _source("short body")
    source["content"] = "short body"

    contents, metadata = prepare_embedding_records([source], WhitespaceTokenizer(), max_seq_length=10, overlap_tokens=2)

    assert len(contents) == 1
    assert metadata[0]["chunk_uuid"] == "chunk"
    assert metadata[0]["original_text"] == "short body"
    assert metadata[0]["chunk_count"] == 1


def test_windows_start_and_end_on_word_boundaries():
    text = " ".join("abcdefgh" for _ in range(12))  # every word is three pieces

    spans = _window_spans(PieceTokenizer(), text, max_tokens=8, overlap_tokens=2)

    assert len(spans) > 1
    for start, end in spans:
        assert start == 0 or text[start - 1] == " "
        assert end == len(text) or text[end - 1] == " "
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    for (_, end), (next_start, _) in zip(spans, spans[1:]):
        assert next_start < end


def test_single_oversized_word_is_split_without_stalling():
    text = "x" * 60

    spans = _window_spans(PieceTokenizer(), text, max_tokens=5, overlap_tokens=2)

    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    assert all(later[0] > earlier[0] for earlier, later in zip(spans, spans[1:]))


def test_every_window_extends_coverage_past_a_long_word_boundary():
    text = " ".join([*("prefix" for _ in range(45)), "x" * 600, "suffix"])

    spans = _window_spans(PieceTokenizer(), text, max_tokens=32, overlap_tokens=8)

    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    assert all(end > previous_end for (_, previous_end), (_, end) in zip(spans, spans[1:]))
    assert all(text[start:end] for start, end in spans)


def test_embedding_context_includes_the_full_heading_path():
    source = _source("short body", heading_path=["Getting started", "Install"])

    contents, _ = prepare_embedding_records([source], WhitespaceTokenizer(), max_seq_length=40, overlap_tokens=2)

    assert contents[0].startswith("Example\nGetting started > Install\n\n")


def test_overflowing_window_retries_with_a_lossless_budget():
    body = " ".join(f"word{index}" for index in range(30))

    _, metadata = prepare_embedding_records([_source(body)], SeamTokenizer(), max_seq_length=12, overlap_tokens=2)

    assert all(item["embedding_token_count"] <= 12 for item in metadata)
    assert metadata[0]["content_start_char"] == 0
    assert metadata[-1]["content_end_char"] == len(body)
    assert all(
        current["content_start_char"] <= previous["content_end_char"]
        for previous, current in zip(metadata, metadata[1:])
    )


def test_embedding_fit_rechecks_nonlinear_seam_overflow():
    embedding_text, token_count, fitted_body, trimmed = _fit_embedding_text(
        NonlinearSeamTokenizer(),
        "Example",
        "\n\n",
        " ".join(f"word{index}" for index in range(20)),
        max_seq_length=12,
    )

    assert trimmed
    assert fitted_body
    assert fitted_body in embedding_text
    assert token_count <= 12

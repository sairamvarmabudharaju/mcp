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

import pytest

from local_vectorstore_creation import load_local_yaml_files, prepare_embedding_records


class WhitespaceTokenizer:
    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(
        self,
        text,
        add_special_tokens=True,
        return_offsets_mapping=False,
        truncation=False,
        padding=False,
    ):
        del truncation, padding
        matches = list(__import__("re").finditer(r"\S+", text))
        input_ids = list(range(len(matches) + (2 if add_special_tokens else 0)))
        result = {"input_ids": input_ids}
        if return_offsets_mapping:
            result["offset_mapping"] = [match.span() for match in matches]
        return result


def test_load_local_yaml_files_requires_intrinsic_chunks(tmp_path, monkeypatch):
    intrinsic_dir = tmp_path / "intrinsic_chunks"
    intrinsic_dir.mkdir()
    monkeypatch.setenv("INTRINSIC_CHUNKS_DIR", str(intrinsic_dir))
    monkeypatch.setenv("YAML_DATA_DIR", str(tmp_path / "yaml_data"))

    with pytest.raises(FileNotFoundError, match="No intrinsic chunk YAML files found"):
        load_local_yaml_files()


def test_prepare_embedding_records_preserves_body_with_overlap():
    body = " ".join(f"word{index}" for index in range(20))
    source = {
        "uuid": "source",
        "chunk_uuid": "chunk",
        "url": "https://example.com/page/#section",
        "title": "Example",
        "heading": "Section",
        "heading_path": ["Section"],
        "keywords": "example",
        "content": f"Document Title: Example\nHeading Path: Section\n\n{body}",
    }

    contents, metadata = prepare_embedding_records(
        [source],
        WhitespaceTokenizer(),
        max_seq_length=10,
        overlap_tokens=2,
    )

    assert len(contents) > 1
    assert all(item["embedding_token_count"] <= 10 for item in metadata)
    assert all(item["parent_chunk_uuid"] == "chunk" for item in metadata)
    assert metadata[0]["content_start_char"] == 0
    assert metadata[-1]["content_end_char"] == len(body)
    assert body in metadata[0]["search_text"]
    assert body not in metadata[1]["search_text"]
    for index in range(len(metadata) - 1):
        assert metadata[index]["content_end_char"] > metadata[index + 1]["content_start_char"]
    assert all(word in " ".join(item["original_text"] for item in metadata) for word in body.split())


def test_prepare_embedding_records_keeps_short_source_as_one_record():
    source = {
        "uuid": "source",
        "chunk_uuid": "chunk",
        "url": "https://example.com/page/",
        "title": "Example",
        "keywords": "example",
        "content": "short body",
    }

    contents, metadata = prepare_embedding_records(
        [source],
        WhitespaceTokenizer(),
        max_seq_length=10,
        overlap_tokens=2,
    )

    assert len(contents) == 1
    assert metadata[0]["chunk_uuid"] == "chunk"
    assert metadata[0]["original_text"] == "short body"
    assert metadata[0]["chunk_count"] == 1

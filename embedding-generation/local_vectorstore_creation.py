# Copyright © 2025, Arm Limited and Contributors. All rights reserved.
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

import argparse
import datetime
import glob
import json
import os
import re
from typing import Any

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer
from usearch.index import Index

EMBEDDING_WINDOW_OVERLAP_TOKENS = 32
MAX_EMBEDDING_CONTEXT_TOKENS = 64
EMBEDDING_TOKEN_SAFETY_MARGIN = 2
CONTENT_PREFIX_PATTERN = re.compile(
    r"^Document Title:\s*(.*?)\nHeading Path:\s*(.*?)\n\n",
    re.DOTALL,
)


def load_local_yaml_files() -> list[dict]:
    """Load locally stored YAML files and return their contents as a list of dictionaries."""
    print("Loading local YAML files")
    yaml_contents = []
    intrinsic_dir = os.getenv("INTRINSIC_CHUNKS_DIR", "intrinsic_chunks")
    yaml_dir = os.getenv("YAML_DATA_DIR", "yaml_data")

    intrinsic_files = glob.glob(os.path.join(intrinsic_dir, "*.yaml"))
    print(f"Found {len(intrinsic_files)} YAML files in {intrinsic_dir} directory")
    if not intrinsic_files:
        raise FileNotFoundError(
            f"No intrinsic chunk YAML files found in '{intrinsic_dir}'. "
            "Supply the intrinsic chunks before creating the vector store."
        )

    yaml_data_files = glob.glob(os.path.join(yaml_dir, "*.yaml"))
    print(f"Found {len(yaml_data_files)} YAML files in {yaml_dir} directory")

    # Combine all files
    all_files = intrinsic_files + yaml_data_files
    total_files = len(all_files)
    print(f"Total files to process: {total_files}")

    for i, file_path in enumerate(all_files, 1):
        if i <= 10 or i % 1000 == 0 or i == total_files:
            print(f"Loading file {i}/{total_files}: {file_path}")

        # Extract chunk identifier based on file location
        if os.path.normpath(file_path).startswith(os.path.normpath(intrinsic_dir)):
            chunk_uuid = f"intrinsic_{os.path.basename(file_path).replace('.yaml', '')}"
        elif os.path.normpath(file_path).startswith(os.path.normpath(yaml_dir)):
            chunk_uuid = f"yaml_data_{os.path.basename(file_path).replace('.yaml', '')}"
        else:
            chunk_uuid = file_path.replace("chunk_", "").replace(".yaml", "")

        try:
            with open(file_path, "r") as f:
                yaml_content = yaml.safe_load(f)
                yaml_content["chunk_uuid"] = chunk_uuid
                yaml_contents.append(yaml_content)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

    print(f"Successfully loaded {len(yaml_contents)} YAML files")
    return yaml_contents


def load_embedding_model(model_path: str) -> SentenceTransformer:
    """Load the local SentenceTransformer used for splitting and embedding."""
    return SentenceTransformer(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
    )


def _token_offsets(tokenizer: Any, text: str) -> list[tuple[int, int]]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
        padding=False,
    )
    return [tuple(offset) for offset in encoded["offset_mapping"] if tuple(offset) != (0, 0)]


def _truncate_to_tokens(tokenizer: Any, text: str, max_tokens: int) -> str:
    offsets = _token_offsets(tokenizer, text)
    if len(offsets) <= max_tokens:
        return text
    return text[: offsets[max_tokens - 1][1]]


def _window_spans(
    tokenizer: Any,
    text: str,
    max_tokens: int,
    overlap_tokens: int,
) -> list[tuple[int, int]]:
    if max_tokens <= 0:
        raise ValueError("Embedding body token budget must be positive")
    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("Embedding overlap must be smaller than the body token budget")
    offsets = _token_offsets(tokenizer, text)
    if len(offsets) <= max_tokens:
        return [(0, len(text))]

    spans = []
    start_token = 0
    while start_token < len(offsets):
        end_token = min(len(offsets), start_token + max_tokens)
        start_char = 0 if start_token == 0 else offsets[start_token][0]
        end_char = len(text) if end_token == len(offsets) else offsets[end_token][0]
        spans.append((start_char, end_char))
        if end_token == len(offsets):
            break
        start_token = end_token - overlap_tokens
    return spans


def _embedding_context(yaml_content: dict, tokenizer: Any) -> str:
    heading_path = yaml_content.get("heading_path", []) or []
    values = [yaml_content.get("title", "")]
    if heading_path:
        values.append(str(heading_path[-1]))
    elif yaml_content.get("heading"):
        values.append(str(yaml_content["heading"]))
    unique_values = []
    for value in values:
        normalized = str(value).strip()
        if normalized and normalized not in unique_values:
            unique_values.append(normalized)
    return _truncate_to_tokens(
        tokenizer,
        "\n".join(unique_values),
        MAX_EMBEDDING_CONTEXT_TOKENS,
    )


def _content_body(content: str) -> str:
    match = CONTENT_PREFIX_PATTERN.match(content)
    return content[match.end():] if match else content


def _metadata_for_window(
    yaml_content: dict,
    original_text: str,
    search_text_content: str,
    chunk_uuid: str,
    parent_chunk_uuid: str,
    chunk_index: int,
    chunk_count: int,
    content_start_char: int,
    content_end_char: int,
    parent_content_length: int,
    embedding_token_count: int,
) -> dict:
    heading_path = yaml_content.get("heading_path", []) or []
    search_text = " ".join(
        str(value)
        for value in [
            yaml_content.get("title", ""),
            " ".join(heading_path),
            yaml_content.get("heading", ""),
            yaml_content.get("doc_type", ""),
            yaml_content.get("product", ""),
            yaml_content.get("version", ""),
            yaml_content.get("keywords", ""),
            search_text_content,
        ]
        if value
    )
    return {
        "uuid": yaml_content["uuid"],
        "url": yaml_content["url"],
        "resolved_url": yaml_content.get("resolved_url", yaml_content["url"]),
        "original_text": original_text,
        "title": yaml_content["title"],
        "keywords": yaml_content["keywords"],
        "chunk_uuid": chunk_uuid,
        "parent_chunk_uuid": parent_chunk_uuid,
        "chunk_index": chunk_index,
        "chunk_count": chunk_count,
        "content_start_char": content_start_char,
        "content_end_char": content_end_char,
        "parent_content_length": parent_content_length,
        "embedding_token_count": embedding_token_count,
        "heading": yaml_content.get("heading", ""),
        "heading_path": heading_path,
        "doc_type": yaml_content.get("doc_type", ""),
        "product": yaml_content.get("product", ""),
        "version": yaml_content.get("version", ""),
        "content_type": yaml_content.get("content_type", ""),
        "search_text": search_text,
    }


def prepare_embedding_records(
    yaml_contents: list[dict],
    tokenizer: Any,
    max_seq_length: int,
    overlap_tokens: int = EMBEDDING_WINDOW_OVERLAP_TOKENS,
) -> tuple[list[str], list[dict]]:
    """Create lossless overlapping windows that fit the embedding model."""
    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    contents = []
    metadata = []
    for yaml_content in yaml_contents:
        source_content = yaml_content["content"]
        body = _content_body(source_content)
        context = _embedding_context(yaml_content, tokenizer)
        separator = "\n\n" if context and body else ""
        context_token_count = len(
            tokenizer(
                f"{context}{separator}",
                add_special_tokens=False,
                truncation=False,
                padding=False,
            )["input_ids"]
        )
        body_budget = (
            max_seq_length
            - special_tokens
            - context_token_count
            - EMBEDDING_TOKEN_SAFETY_MARGIN
        )
        spans = _window_spans(tokenizer, body, body_budget, overlap_tokens)
        parent_chunk_uuid = yaml_content["chunk_uuid"]
        chunk_count = len(spans)
        for chunk_index, (start_char, end_char) in enumerate(spans, start=1):
            body_window = body[start_char:end_char]
            embedding_text = f"{context}{separator}{body_window}"
            embedding_token_count = len(
                tokenizer(
                    embedding_text,
                    add_special_tokens=True,
                    truncation=False,
                    padding=False,
                )["input_ids"]
            )
            if embedding_token_count > max_seq_length:
                raise ValueError(
                    f"Embedding window exceeds model limit: {embedding_token_count} > "
                    f"{max_seq_length} for {parent_chunk_uuid}"
                )
            chunk_uuid = parent_chunk_uuid
            if chunk_count > 1:
                chunk_uuid = f"{parent_chunk_uuid}__window_{chunk_index}_of_{chunk_count}"
            original_text = source_content if chunk_count == 1 else body_window
            search_text_content = source_content if chunk_index == 1 else body_window
            contents.append(embedding_text)
            metadata.append(
                _metadata_for_window(
                    yaml_content,
                    original_text,
                    search_text_content,
                    chunk_uuid,
                    parent_chunk_uuid,
                    chunk_index,
                    chunk_count,
                    start_char,
                    end_char,
                    len(body),
                    embedding_token_count,
                )
            )
    return contents, metadata


def create_embeddings(
    contents: list[str],
    model: SentenceTransformer,
) -> np.ndarray:
    """Create embeddings for the given contents using SentenceTransformers."""
    embeddings = model.encode(contents, show_progress_bar=True, convert_to_numpy=True)
    print(f"Created embeddings with shape: {embeddings.shape}")
    return embeddings


def create_usearch_index(
    embeddings: np.ndarray, metadata: list[dict]
) -> tuple[Index, list[dict]]:
    """Create a USearch index with the given embeddings and metadata."""
    print("Creating USearch index")
    print(f"Embeddings shape: {embeddings.shape}")

    dimension = embeddings.shape[1]
    num_vectors = embeddings.shape[0]

    # Create USearch index
    index = Index(
        ndim=dimension,
        metric="l2sq",
        dtype="f32",
        connectivity=16,
        expansion_add=128,
        expansion_search=64,
    )

    # Add vectors to the index
    print(f"Adding {num_vectors} vectors to the index")
    for i, embedding in enumerate(embeddings):
        index.add(i, embedding)

    print(f"Added {len(index)} vectors to the index")
    return index, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the USearch datastore from local chunks and a local embedding model."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the embedding model created by acquire-model.py.",
    )
    parser.add_argument(
        "--skip-embeddings-text",
        action="store_true",
        help="Do not write the optional plain-text embedding matrix.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("Starting the USearch datastore creation process")

    # Load local YAML files
    yaml_contents = load_local_yaml_files()

    print(f"Loading embedding model: {args.model_path}")
    model = load_embedding_model(args.model_path)

    print("Creating tokenizer-aligned embedding windows")
    contents, metadata = prepare_embedding_records(
        yaml_contents,
        model.tokenizer,
        model.max_seq_length,
    )
    split_parents = len({item["parent_chunk_uuid"] for item in metadata if item["chunk_count"] > 1})
    print(
        f"Prepared {len(contents)} embedding windows from {len(yaml_contents)} source chunks; "
        f"split {split_parents} oversized chunks"
    )

    # Create embeddings
    embeddings = create_embeddings(contents, model)

    if not args.skip_embeddings_text:
        print("Saving embeddings to file")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"embeddings_{timestamp}.txt"
        np.savetxt(filename, embeddings)

    # Create USearch index
    print("Creating USearch index")
    index, metadata = create_usearch_index(embeddings, metadata)

    # Save the USearch index
    index_filename = os.getenv("USEARCH_INDEX_FILENAME", "usearch_index.bin")
    print(f"Saving USearch index to {index_filename}")
    index.save(index_filename)

    # Save metadata
    metadata_filename = os.getenv("METADATA_FILENAME", "metadata.json")
    print(f"Saving metadata to {metadata_filename}")
    with open(metadata_filename, "w") as f:
        json.dump(metadata, f, indent=2)

    print("USearch index and metadata have been created and saved.")
    print(f"Total documents processed: {len(contents)}")
    print(f"USearch index saved to: {os.path.abspath(index_filename)}")
    print(f"Metadata saved to: {os.path.abspath(metadata_filename)}")


if __name__ == "__main__":
    main()

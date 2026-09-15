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

"""Intrinsic lookup: detect the intent, then match the structured fields.

The vocabulary (Arm's category phrases and their synonyms) is read from the
index at load time, so the server never carries its own copy of the taxonomy
and a rebuilt index automatically brings new categories along.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Arm intrinsic names (vaddq_f32, svld1[_u32], vqtbl1q_u8), x86 names
# (_mm_shuffle_epi8, _mm256_add_ps) and compiler builtins (__crc32b).
SYMBOL_PATTERN = re.compile(
    r"(?<![a-z0-9])(?:_?mm(?:256|512)?_[a-z0-9_]+|__[a-z0-9_]+|"
    r"sv[a-z0-9_\[\]]{3,}|v[a-z0-9\[\]]{2,}_[a-z0-9_\[\]]+)(?![a-z0-9])",
    re.IGNORECASE,
)
TYPE_PATTERN = re.compile(
    r"\b(?:sv)?(?:(?:u?int|float|bfloat|poly)\d+(?:x\d+(?:x\d+)?)?|bool)_t\b",
    re.IGNORECASE,
)
ELEMENT_PATTERN = re.compile(
    r"\b(?:sv)?(?P<kind>u?int|float|bfloat|poly)(?P<bits>8|16|32|64)(?:x(?P<lanes>\d+)(?:x\d+)?)?(?:_t)?\b",
    re.IGNORECASE,
)
LANE_COUNT_PATTERN = re.compile(
    r"\b(?P<lanes>\d+)\s+(?:lanes?|elements?|floats?|doubles?|bytes?|integers?|ints?)\b",
    re.IGNORECASE,
)
X86_PATTERN = re.compile(
    r"_?mm(?P<width>256|512)?_[a-z0-9_]*?(?:(?:epi|epu)(?P<int_bits>8|16|32|64)|(?P<float_kind>ps|pd))\b",
    re.IGNORECASE,
)
ISA_NAMES = ("sme2", "sme", "sve2", "sve", "neon", "aarch64")
_SIGNEDNESS = {
    "uint": "unsigned", "int": "signed", "float": "floating-point",
    "bfloat": "floating-point", "poly": "polynomial",
}
# How much a perfect intrinsic match adds in the rerank; comparable to the
# direct-match bonus, so it decides between intrinsics without burying pages.
INTRINSIC_BONUS_WEIGHT = 0.5


def normalize_phrase(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[_\-\[\](),.:;/]+", " ", str(text).lower())).strip()


@dataclass
class IntrinsicIndex:
    """Intrinsic parent rows plus the category vocabulary they carry."""

    rows: list[dict[str, Any]]
    terms: list[str] = field(default_factory=list)
    taxonomy_versions: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        vocabulary: set[str] = set()
        for row in self.rows:
            vocabulary.update(normalize_phrase(term) for term in row.get("intrinsic_search_terms", []) or [])
            self.taxonomy_versions.add(str(row.get("intrinsic_taxonomy_version", "")))
        # Longest phrases first so "maximum across vector" wins over "maximum".
        self.terms = sorted((term for term in vocabulary if term), key=len, reverse=True)


def is_intrinsic_metadata(metadata: dict[str, Any]) -> bool:
    return (
        str(metadata.get("doc_type", "")).strip().lower() == "intrinsic"
        and bool(metadata.get("intrinsic_name"))
    )


def build_intrinsic_index(metadata: list[dict[str, Any]]) -> IntrinsicIndex | None:
    rows = [
        item for item in metadata
        if item.get("chunk_index", 1) == 1 and is_intrinsic_metadata(item)
    ]
    if not rows:
        return None
    index = IntrinsicIndex(rows)
    if not index.terms:
        print("Intrinsic rows carry no search terms; rebuild the vectorstore with the current toolchain.")
    return index


def _matched_terms(normalized_query: str, terms: list[str]) -> list[str]:
    """Vocabulary phrases present in the query; the longest phrase claims its words first."""
    matched: list[str] = []
    remaining = normalized_query
    for term in terms:
        pattern = rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])"
        if re.search(pattern, remaining):
            matched.append(term)
            remaining = re.sub(pattern, " ", remaining)
    return matched


def analyze_intrinsic_query(query: str, intrinsic_index: IntrinsicIndex | None) -> dict[str, Any]:
    """What the query says about the intrinsic it wants, and whether it wants one at all."""
    lower = query.lower()
    normalized = normalize_phrase(query)
    words = set(normalized.split())
    symbols = [match.group(0).lower() for match in SYMBOL_PATTERN.finditer(lower)]
    arm_symbols = [symbol for symbol in symbols if not symbol.lstrip("_").startswith("mm")]
    types = [match.group(0).lower() for match in TYPE_PATTERN.finditer(lower)]
    element_bits: set[int] = set()
    signedness: set[str] = set()
    lane_counts: set[int] = set()
    for match in ELEMENT_PATTERN.finditer(lower):
        element_bits.add(int(match.group("bits")))
        signedness.add(_SIGNEDNESS[match.group("kind").lower()])
        if match.group("lanes"):
            lane_counts.add(int(match.group("lanes")))
    for match in LANE_COUNT_PATTERN.finditer(lower):
        lane_counts.add(int(match.group("lanes")))
    if re.search(r"\bfloats?\b", normalized):
        element_bits.add(32)
        signedness.add("floating-point")
    if re.search(r"\bdoubles?\b", normalized):
        element_bits.add(64)
        signedness.add("floating-point")
    x86_symbols = []
    for symbol in symbols:
        x86 = X86_PATTERN.match(symbol)
        if not x86:
            continue
        x86_symbols.append(symbol)
        width = int(x86.group("width") or 128)
        bits = int(x86.group("int_bits")) if x86.group("int_bits") else 32 if x86.group("float_kind") == "ps" else 64
        element_bits.add(bits)
        lane_counts.add(width // bits)
    isa = [name for name in ISA_NAMES if re.search(rf"\b{name}\b", lower)]
    terms = _matched_terms(normalized, intrinsic_index.terms) if intrinsic_index else []
    mentions_intrinsic = bool(words & {"intrinsic", "intrinsics"})
    # A lookup needs something specific: a symbol, a C type, or a category
    # phrase together with the word "intrinsic" or an element width. ISA words
    # plus generic vocabulary ("SVE vector length") are not enough.
    intent = bool(
        arm_symbols
        or x86_symbols
        or types
        or (terms and (mentions_intrinsic or element_bits or lane_counts))
        or (mentions_intrinsic and isa and (element_bits or lane_counts))
    )
    return {
        "intent": intent,
        "symbols": arm_symbols,
        "x86_symbols": x86_symbols,
        "types": types,
        "element_bits": sorted(element_bits),
        "signedness": sorted(signedness),
        "lane_counts": sorted(lane_counts),
        "isa": isa,
        "terms": terms,
    }


def _symbol_score(analysis: dict[str, Any], name: str) -> float:
    for symbol in analysis["symbols"]:
        if name == symbol:
            return 1.0
        if name.startswith(f"{symbol}[") or name.startswith(f"{symbol}_") or symbol.startswith(f"{name}["):
            return 0.8
    return 0.0


def _agreement(expected: set, actual: set) -> float | None:
    if not expected:
        return None
    return len(expected & actual) / len(expected)


def intrinsic_match_score(analysis: dict[str, Any], metadata: dict[str, Any]) -> float:
    """0..1: how well one intrinsic row fits the query's symbol, categories, types and ISA."""
    if not analysis.get("intent") or not is_intrinsic_metadata(metadata):
        return 0.0
    name = str(metadata.get("intrinsic_name", "")).lower()
    symbol = _symbol_score(analysis, name)
    row_terms = {normalize_phrase(term) for term in metadata.get("intrinsic_search_terms", []) or []}
    category = _agreement(set(analysis["terms"]), row_terms)
    type_parts = [
        _agreement(set(analysis["types"]), {str(t).lower() for t in [*(metadata.get("intrinsic_input_types") or []), metadata.get("intrinsic_output_type", "")] if t}),
        _agreement(set(analysis["element_bits"]), set(metadata.get("intrinsic_element_bits") or [])),
        _agreement(set(analysis["signedness"]), set(metadata.get("intrinsic_signedness") or [])),
        _agreement(set(analysis["lane_counts"]), set(metadata.get("intrinsic_lane_counts") or [])),
    ]
    type_parts = [part for part in type_parts if part is not None]
    types = sum(type_parts) / len(type_parts) if type_parts else None
    isa = None
    if analysis["isa"]:
        row_isa = str(metadata.get("intrinsic_isa", "")).lower()
        isa = 1.0 if row_isa in analysis["isa"] else 0.5 if any(
            row_isa.startswith(value) or value.startswith(row_isa) for value in analysis["isa"] if row_isa
        ) else 0.0
    if symbol:
        return symbol
    # ISA alone never scores: it would promote every intrinsic of that ISA.
    if category is None and types is None:
        return 0.0
    weights = [(0.6, category), (0.3, types), (0.1, isa)]
    total = sum(weight for weight, value in weights if value is not None)
    return sum(weight * value for weight, value in weights if value is not None) / total


def intrinsic_candidate_search(
    analysis: dict[str, Any],
    intrinsic_index: IntrinsicIndex | None,
    k: int,
) -> list[dict[str, Any]]:
    """Intrinsic rows ordered by structured fit; empty unless the query wants an intrinsic."""
    if intrinsic_index is None or not analysis.get("intent"):
        return []
    scored = []
    for row in intrinsic_index.rows:
        score = intrinsic_match_score(analysis, row)
        if score > 0:
            scored.append((score, str(row.get("intrinsic_name", "")), row))
    scored.sort(key=lambda value: (-value[0], value[1]))
    return [
        {"metadata": row, "intrinsic_rank": rank, "intrinsic_score": score}
        for rank, (score, _, row) in enumerate(scored[:k], start=1)
    ]

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

"""Structured fields for Arm intrinsic records, derived from Arm's own metadata.

Every intrinsic record from developer.arm.com carries Arm's instruction
categories in its keywords ("vector arithmetic, maximum across vector",
"load, gather", "table lookup", ...). Those categories are the vocabulary we
index and search on. A short synonym table maps common natural-language
phrasings ("horizontal add", "popcount", "broadcast") onto them. Everything
else here is read from the intrinsic's C signature.
"""

from __future__ import annotations

import re
from urllib.parse import unquote

# Bump when the derived fields or the synonym table change, so an index can be
# told apart from one built with an older taxonomy.
INTRINSIC_TAXONOMY_VERSION = "2.0.0"

# Natural-language phrase -> Arm category phrase(s) it should match. Keys are
# matched as whole phrases in a normalised (lower-case, punctuation-free) query.
CATEGORY_SYNONYMS: dict[str, tuple[str, ...]] = {
    "horizontal add": ("addition across vector",),
    "horizontal sum": ("addition across vector",),
    "sum across lanes": ("addition across vector",),
    "add across lanes": ("addition across vector",),
    "add across vector": ("addition across vector",),
    "sum of all lanes": ("addition across vector",),
    "reduce add": ("addition across vector",),
    "horizontal max": ("maximum across vector",),
    "maximum element": ("maximum across vector",),
    "largest lane": ("maximum across vector",),
    "max across lanes": ("maximum across vector",),
    "horizontal min": ("minimum across vector",),
    "minimum element": ("minimum across vector",),
    "smallest lane": ("minimum across vector",),
    "min across lanes": ("minimum across vector",),
    "fma": ("fused multiply-accumulate",),
    "fused multiply add": ("fused multiply-accumulate",),
    "multiply add": ("multiply-accumulate",),
    "multiply and accumulate": ("multiply-accumulate",),
    "mla": ("multiply-accumulate",),
    "multiply subtract": ("multiply-accumulate",),
    "mls": ("multiply-accumulate",),
    "bitcast": ("reinterpret casts",),
    "bit cast": ("reinterpret casts",),
    "reinterpret": ("reinterpret casts",),
    "broadcast": ("set all lanes to the same value",),
    "splat": ("set all lanes to the same value",),
    "duplicate": ("set all lanes to the same value",),
    "load and broadcast": ("load and replicate",),
    "load and duplicate": ("load and replicate",),
    "interleave": ("zip elements",),
    "zip": ("zip elements",),
    "deinterleave": ("unzip elements",),
    "unzip": ("unzip elements",),
    "shuffle": ("table lookup", "table lookups"),
    "permute": ("table lookup", "table lookups"),
    "lookup table": ("table lookup", "table lookups"),
    "popcount": ("population count",),
    "count set bits": ("population count",),
    "count bits": ("population count",),
    "clz": ("count leading zeros",),
    "leading zeros": ("count leading zeros",),
    "leading sign bits": ("count leading sign bits",),
    "select bits": ("bitwise select", "select between two vectors"),
    "bitwise select": ("bitwise select", "select between two vectors"),
    "select using a mask": ("bitwise select", "select between two vectors"),
    "select between two vectors": ("bitwise select", "select between two vectors"),
    "predicated select": ("select between two vectors",),
    "blend": ("bitwise select", "select between two vectors"),
    "compact": ("concatenate active elements",),
    "pack the active": ("concatenate active elements",),
    "pack active lanes": ("concatenate active elements",),
    "last active element": ("extract last active element",),
    "last active lane": ("extract last active element",),
    "convert": ("data type conversion", "conversions"),
    "conversion": ("data type conversion", "conversions"),
    "float to int": ("data type conversion", "conversions"),
    "to integer": ("data type conversion", "conversions"),
    "to float": ("data type conversion", "conversions"),
    "shift right": ("vector shift right", "right"),
    "shift left": ("vector shift left", "left"),
    "reverse": ("reverse elements",),
    "abs diff": ("absolute difference",),
    "absolute difference": ("absolute difference",),
    "matrix multiply": ("matrix multiply",),
    "mmla": ("matrix multiply",),
    "dot product": ("dot product",),
    "all true predicate": ("initialize to pattern",),
    "all lanes true": ("initialize to pattern",),
    "ptrue": ("initialize to pattern",),
    "while less than": ("while counter meets condition (forward)",),
    "loop predicate": ("while counter meets condition (forward)",),
    "first fault": ("first-faulting loads",),
    "first faulting": ("first-faulting loads",),
    "non faulting": ("non-faulting loads",),
    "saturating add": ("saturating addition",),
    "saturating subtract": ("saturating subtract",),
    "halving add": ("narrowing addition",),
    "rounding halving add": ("narrowing addition",),
    "pairwise add": ("pairwise addition",),
    "pairwise max": ("pairwise maximum",),
    "pairwise min": ("pairwise minimum",),
    "widening multiply": ("widening multiplication",),
    "gather load": ("gather",),
    "gather from memory": ("gather",),
    "scatter store": ("scatter",),
    "scatter to memory": ("scatter",),
    "square root": ("square root",),
    "sqrt": ("square root",),
    "reciprocal estimate": ("reciprocal estimate",),
    "keeps lanes before the first true": ("break before first true condition",),
    "lanes before the first true": ("break before first true condition",),
    "break before": ("break before first true condition",),
    "break after": ("break after first true condition",),
    "crc": ("crc32",),
    "cyclic redundancy check": ("crc32",),
    "aes encrypt": ("aes",),
    "load interleaved": ("stride", "load"),
    "interleaved load": ("stride", "load"),
    "structure load": ("stride", "load"),
    "extract lane": ("extract one element from vector",),
    "get lane": ("extract one element from vector",),
    "set lane": ("set vector lane",),
    "insert lane": ("set vector lane",),
}

# Keyword entries that are not categories.
_NOT_A_CATEGORY = {
    "intrinsic", "sse", "avx", "streaming simd extension",
    "neon", "sve", "sve2", "sme", "sme2", "aarch64",
}
_ISA_LABELS = {"neon": "Neon", "sve": "SVE", "sve2": "SVE2", "sme": "SME", "sme2": "SME2", "aarch64": "AArch64"}

INTRINSIC_URL_PATTERN = re.compile(r"[?#&]q=([^&#]+)", re.IGNORECASE)
ISA_PATTERN = re.compile(
    r"part of the\s+(neon|sve2|sve|sme2|sme|aarch64)\s+instruction set architecture",
    re.IGNORECASE,
)
SIGNATURE_PATTERN = re.compile(r"`([^`\n]+\([^`\n]*\);?)`")
# C types in a signature: Neon fixed vectors (uint8x16_t), SVE scalable vectors
# (svuint8_t, svbool_t) and scalars (uint8_t, float32_t).
TYPE_PATTERN = re.compile(
    r"\b(?:sv)?(?:(?:u?int|float|bfloat|poly)\d+(?:x\d+(?:x\d+)?)?|bool)_t\b|\bvoid\b",
    re.IGNORECASE,
)
ELEMENT_PATTERN = re.compile(
    r"\b(?:sv)?(?P<kind>u?int|float|bfloat|poly)(?P<bits>\d+)(?:x(?P<lanes>\d+)(?:x\d+)?)?_t\b",
    re.IGNORECASE,
)
_SIGNEDNESS = {
    "uint": "unsigned", "int": "signed", "float": "floating-point",
    "bfloat": "floating-point", "poly": "polynomial",
}


def is_intrinsic_record(record: dict) -> bool:
    """Return whether a source or metadata row is an Arm intrinsic."""
    return bool(
        str(record.get("doc_type", "")).strip().lower() == "intrinsic"
        or str(record.get("chunk_uuid", "")).startswith("intrinsic_")
        or "/intrinsics/#q=" in str(record.get("url", "")).lower()
    )


def normalize_phrase(text: str) -> str:
    """Lower-case and strip punctuation so phrases compare by words only."""
    return re.sub(r"\s+", " ", re.sub(r"[_\-\[\](),.:;/]+", " ", str(text).lower())).strip()


def _content(record: dict) -> str:
    return str(record.get("content") or record.get("original_text") or "")


def _signature(record: dict) -> str:
    for match in SIGNATURE_PATTERN.finditer(_content(record)):
        if "#include" not in match.group(1) and "(" in match.group(1):
            return match.group(1).rstrip(";").strip()
    return ""


def intrinsic_name(record: dict) -> str:
    url_match = INTRINSIC_URL_PATTERN.search(str(record.get("url", "")))
    if url_match:
        return unquote(url_match.group(1))
    title = str(record.get("title", ""))
    if " - " in title:
        return title.rsplit(" - ", 1)[-1].strip()
    name_match = re.search(r"([A-Za-z_][A-Za-z0-9_\[\]]*)\s*\(", _signature(record))
    return name_match.group(1) if name_match else title.strip()


def intrinsic_isa(record: dict) -> str:
    match = ISA_PATTERN.search(_content(record))
    if match:
        return _ISA_LABELS[match.group(1).lower()]
    for keyword in _keyword_entries(record):
        if keyword in _ISA_LABELS:
            return _ISA_LABELS[keyword]
    return ""


def _keyword_entries(record: dict) -> list[str]:
    keywords = record.get("keywords", "")
    if isinstance(keywords, list):
        entries = [str(value) for value in keywords]
    else:
        entries = str(keywords).split(",")
    return [entry.strip().lower() for entry in entries if entry.strip()]


def intrinsic_categories(record: dict) -> list[str]:
    """Arm's category phrases from the record keywords, in their original order."""
    name = intrinsic_name(record).lower()
    categories: list[str] = []
    for entry in _keyword_entries(record):
        if entry in _NOT_A_CATEGORY or entry == name or entry in categories:
            continue
        # The first keyword is the intrinsic name itself, possibly in bracket form.
        if re.fullmatch(r"(?:sv|v)[a-z0-9_\[\]]+", entry):
            continue
        categories.append(entry)
    return categories


def search_terms_for(categories: list[str]) -> list[str]:
    """Categories plus every synonym that maps onto one of them."""
    terms = [normalize_phrase(category) for category in categories]
    normalized = set(terms)
    for phrase, targets in CATEGORY_SYNONYMS.items():
        if any(normalize_phrase(target) in normalized for target in targets) and phrase not in normalized:
            terms.append(phrase)
    return terms


def _description(record: dict) -> str:
    content = _content(record)
    match = re.search(
        r"brief intrinsic description:\s*(.*?)(?:\n\nThe signature|The signature)",
        content,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return ""
    description = re.sub(r"\s+", " ", match.group(1)).strip(" `")
    sentences = re.split(r"(?<=[.!?])\s+", description)
    return " ".join(sentences[:2])[:400].strip()


def signature_types(signature: str) -> dict:
    """Input/output C types and the element sizes, signedness and lane counts they imply."""
    types = TYPE_PATTERN.findall(signature)
    output_type = types[0] if types else ""
    input_types = types[1:] if len(types) > 1 else []
    element_bits: set[int] = set()
    signedness: set[str] = set()
    lane_counts: set[int] = set()
    for match in ELEMENT_PATTERN.finditer(signature):
        element_bits.add(int(match.group("bits")))
        signedness.add(_SIGNEDNESS[match.group("kind").lower()])
        if match.group("lanes"):
            lane_counts.add(int(match.group("lanes")))
    return {
        "intrinsic_input_types": input_types,
        "intrinsic_output_type": output_type,
        "intrinsic_element_bits": sorted(element_bits),
        "intrinsic_signedness": sorted(signedness),
        "intrinsic_lane_counts": sorted(lane_counts),
    }


def enrich_intrinsic_record(record: dict) -> dict:
    """Return the structured fields and the compact text to embed for one intrinsic."""
    name = intrinsic_name(record)
    isa = intrinsic_isa(record)
    categories = intrinsic_categories(record)
    terms = search_terms_for(categories)
    description = _description(record)
    types = signature_types(_signature(record))
    fields = {
        "doc_type": "Intrinsic",
        "intrinsic_taxonomy_version": INTRINSIC_TAXONOMY_VERSION,
        "intrinsic_name": name,
        "intrinsic_isa": isa,
        "intrinsic_categories": categories,
        "intrinsic_search_terms": terms,
        "intrinsic_description": description,
        **types,
    }
    io_text = ""
    if types["intrinsic_input_types"] or types["intrinsic_output_type"]:
        io_text = (
            f" Takes {', '.join(types['intrinsic_input_types']) or 'no arguments'}"
            f" and returns {types['intrinsic_output_type'] or 'nothing'}."
        )
    fields["intrinsic_embedding_text"] = (
        f"{name}: {description or ', '.join(categories)}."
        f" {isa + ' ' if isa else ''}intrinsic for {'; '.join(categories) or 'vector operations'}."
        f"{' Also called ' + ', '.join(term for term in terms if term not in categories) + '.' if len(terms) > len(categories) else ''}"
        f"{io_text}"
    )
    # Lexical search sees the categories, synonyms and description before the body.
    fields["intrinsic_search_text_prefix"] = " ".join((name, isa, "; ".join(terms), description)).strip()
    return fields

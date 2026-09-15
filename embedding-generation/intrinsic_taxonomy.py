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

"""Structured, versioned enrichment for Arm intrinsic source records."""

from __future__ import annotations

import re
from urllib.parse import unquote

INTRINSIC_TAXONOMY_VERSION = "1.0.0"

# Specific operations precede their general components. These are vocabulary
# categories, not eval-query rewrites: they enrich all current and future data.
OPERATION_TAXONOMY: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("gather_load", ("gather load", "gather from memory", "gather")),
    ("scatter_store", ("scatter store", "scatter to memory", "scatter")),
    (
        "reduction_add",
        (
            "horizontal add", "horizontal sum", "add across vector",
            "add across lanes", "addition across vector", "sum lanes",
            "sum vector elements", "vector reduction add",
            "add every vector element",
        ),
    ),
    (
        "fused_multiply_add",
        ("fused multiply add", "fused multiply-add", "fma", "multiply-add to accumulator"),
    ),
    (
        "multiply_accumulate",
        ("multiply accumulate", "multiply-accumulate", "multiply and accumulate", "mla"),
    ),
    (
        "multiply_subtract",
        ("multiply subtract", "multiply-subtract", "multiply and subtract", "mls"),
    ),
    ("table_lookup", ("table lookup", "lookup table", "byte shuffle", "vector shuffle", "shuffle bytes")),
    ("predicate_all_true", ("all true predicate", "all-true predicate", "every element is true", "all lanes true")),
    ("count_leading_zeros", ("count leading zeros", "leading zero count", "clz")),
    ("reinterpret", ("reinterpret", "bit cast", "bitcast", "change signedness without conversion")),
    ("widen", ("widening", "widen", "long operation", "sign extend", "zero extend")),
    ("narrow", ("narrowing", "narrow", "saturating narrow")),
    ("zip", ("zip interleave", "zip vectors", "interleave two vectors", "interleave")),
    ("duplicate", ("load and replicate", "load and duplicate", "broadcast", "splat", "duplicate", "dup")),
    ("compare", ("comparison", "compare", "equal to", "greater than", "less than")),
    ("predicate", ("predicate", "predication", "while less than", "active lanes")),
    ("store", ("store to memory", "write to memory", "store")),
    ("load", ("load from memory", "read from memory", "load")),
    ("multiply", ("vector multiply", "multiply vectors", "multiplication", "multiply")),
    ("subtract", ("vector subtract", "subtraction", "subtract")),
    ("add", ("vector add", "addition", "add")),
    ("crc", ("cyclic redundancy check", "crc32", "crc")),
)

INTRINSIC_URL_PATTERN = re.compile(r"[?#&]q=([^&#]+)", re.IGNORECASE)
ISA_PATTERN = re.compile(
    r"part of the\s+(neon|sve2|sve|sme2|sme|aarch64)\s+instruction set architecture",
    re.IGNORECASE,
)
SIGNATURE_PATTERN = re.compile(r"`([^`\n]+\([^`\n]*\);?)`")
TYPE_PATTERN = re.compile(
    r"\b(?:sv(?:u?int|float|bfloat)\d+_t|svbool_t|(?:u?int|float|bfloat|poly)\d+"
    r"(?:x\d+(?:x\d+)?)?_t|void)\b",
    re.IGNORECASE,
)
FIXED_VECTOR_PATTERN = re.compile(
    r"\b(?P<kind>u?int|float|bfloat|poly)(?P<bits>\d+)x(?P<lanes>\d+)(?:x(?P<tuple>\d+))?_t\b",
    re.IGNORECASE,
)
ELEMENT_PATTERN = re.compile(
    r"\b(?P<kind>u?int|float|bfloat|poly)(?P<bits>\d+)(?:x\d+(?:x\d+)?)?_t\b",
    re.IGNORECASE,
)


def is_intrinsic_record(record: dict) -> bool:
    """Return whether a source or metadata row represents an intrinsic."""
    return bool(
        str(record.get("doc_type", "")).strip().lower() == "intrinsic"
        or str(record.get("chunk_uuid", "")).startswith("intrinsic_")
        or "/intrinsics/#q=" in str(record.get("url", "")).lower()
    )


def _signature(record: dict) -> str:
    content = str(record.get("content") or record.get("original_text") or "")
    for match in SIGNATURE_PATTERN.finditer(content):
        if "#include" not in match.group(1) and "(" in match.group(1):
            return match.group(1).rstrip(";").strip()
    return ""


def _intrinsic_name(record: dict) -> str:
    url_match = INTRINSIC_URL_PATTERN.search(str(record.get("url", "")))
    if url_match:
        return unquote(url_match.group(1))
    title = str(record.get("title", ""))
    if " - " in title:
        return title.rsplit(" - ", 1)[-1].strip()
    name_match = re.search(r"([A-Za-z_][A-Za-z0-9_\[\]]*)\s*\(", _signature(record))
    return name_match.group(1) if name_match else title.strip()


def _description(record: dict) -> str:
    content = str(record.get("content") or record.get("original_text") or "")
    match = re.search(
        r"brief intrinsic description:\s*(.*?)(?:\n\nThe signature|The signature)",
        content,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        match = re.search(
            r"(?:pseudo|sudo)code for how .*? operates:\s*(.*)",
            content,
            re.IGNORECASE | re.DOTALL,
        )
    if not match:
        return ""
    description = re.sub(r"\s+", " ", match.group(1)).strip(" `")
    sentences = re.split(r"(?<=[.!?])\s+", description)
    return " ".join(sentences[:2])[:420].strip()


def _canonical_operation(record: dict, description: str) -> str:
    text = " ".join(str(record.get(field, "")) for field in ("title", "keywords"))
    normalized = re.sub(r"[_\[\](),.:;/]+", " ", f"{text} {description}".lower())
    normalized = re.sub(r"\s+", " ", normalized)
    for operation, aliases in OPERATION_TAXONOMY:
        if any(
            re.search(
                rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])",
                normalized,
            )
            for alias in aliases
        ):
            return operation
    return "other"


def _isa(record: dict) -> str:
    content = str(record.get("content") or record.get("original_text") or "")
    match = ISA_PATTERN.search(content)
    if match:
        return match.group(1).upper() if match.group(1).lower() != "neon" else "Neon"
    keywords = str(record.get("keywords", "")).lower()
    for isa in ("sme2", "sme", "sve2", "sve", "neon", "aarch64"):
        if re.search(rf"\b{isa}\b", keywords):
            return isa.upper() if isa != "neon" else "Neon"
    return "Unknown"


def _type_details(signature: str) -> dict:
    types = TYPE_PATTERN.findall(signature)
    output_type = types[0] if types else ""
    input_types = types[1:] if len(types) > 1 else []
    fixed_vectors = list(FIXED_VECTOR_PATTERN.finditer(signature))
    lane_counts = sorted({int(match.group("lanes")) for match in fixed_vectors})
    vector_widths = sorted({
        int(match.group("bits")) * int(match.group("lanes"))
        for match in fixed_vectors
    })
    elements = list(ELEMENT_PATTERN.finditer(signature))
    element_bits = sorted({int(match.group("bits")) for match in elements})
    signedness = set()
    names = {
        "uint": "unsigned", "int": "signed", "float": "floating-point",
        "bfloat": "floating-point", "poly": "polynomial",
    }
    for match in elements:
        signedness.add(names[match.group("kind").lower()])
    output_lower = output_type.lower()
    if output_lower == "void":
        output_shape = "void"
    elif output_lower == "svbool_t":
        output_shape = "predicate"
    elif FIXED_VECTOR_PATTERN.search(output_type) or output_lower.startswith("sv"):
        output_shape = (
            "vector tuple"
            if re.search(r"x\d+x\d+_t$", output_lower)
            else "vector"
        )
    else:
        output_shape = "scalar"
    return {
        "intrinsic_input_types": input_types,
        "intrinsic_output_type": output_type,
        "intrinsic_output_shape": output_shape,
        "intrinsic_lane_counts": lane_counts,
        "intrinsic_vector_width_bits": vector_widths,
        "intrinsic_element_bits": element_bits,
        "intrinsic_signedness": sorted(signedness),
    }


def _predication(name: str, signature: str, operation: str) -> str:
    if operation == "predicate_all_true":
        return "Creates an all-true predicate."
    if "svbool_t" not in signature:
        return "Unpredicated."
    if re.search(r"_m(?:\W|$)", name):
        return "Predicated; inactive lanes merge from the first input."
    if re.search(r"_z(?:\W|$)", name):
        return "Predicated; inactive lanes are zeroed."
    if re.search(r"_x(?:\W|$)", name):
        return "Predicated; inactive lanes are unspecified."
    return "Predicated; operates only on active lanes."


def _memory_behavior(operation: str) -> str:
    return {
        "gather_load": "Reads noncontiguous elements from memory.",
        "scatter_store": "Writes noncontiguous elements to memory.",
        "load": "Reads contiguous data from memory.",
        "store": "Writes contiguous data to memory.",
        "duplicate": "May replicate a scalar or loaded element across lanes.",
    }.get(operation, "No memory access.")


def enrich_intrinsic_record(record: dict) -> dict:
    """Extract stable structured fields and a compact dense-embedding document."""
    name = _intrinsic_name(record)
    signature = _signature(record)
    description = _description(record)
    operation = _canonical_operation(record, description)
    aliases = next(
        (list(values) for key, values in OPERATION_TAXONOMY if key == operation),
        [],
    )
    details = {
        "doc_type": "Intrinsic",
        "intrinsic_taxonomy_version": INTRINSIC_TAXONOMY_VERSION,
        "intrinsic_name": name,
        "intrinsic_isa": _isa(record),
        "intrinsic_operation": operation,
        "intrinsic_aliases": aliases,
        **_type_details(signature),
        "intrinsic_predication": _predication(name, signature, operation),
        "intrinsic_memory_behavior": _memory_behavior(operation),
        "intrinsic_description": description,
    }
    input_text = ", ".join(details["intrinsic_input_types"]) or "none"
    vectors = []
    if details["intrinsic_vector_width_bits"]:
        vectors.append(
            "width "
            + "/".join(map(str, details["intrinsic_vector_width_bits"]))
            + " bits"
        )
    elif details["intrinsic_isa"].lower().startswith(("sve", "sme")):
        vectors.append("scalable vector length")
    if details["intrinsic_lane_counts"]:
        vectors.append(
            "/".join(map(str, details["intrinsic_lane_counts"])) + " lanes"
        )
    if details["intrinsic_element_bits"]:
        vectors.append(
            "/".join(map(str, details["intrinsic_element_bits"]))
            + "-bit elements"
        )
    if details["intrinsic_signedness"]:
        vectors.append("/".join(details["intrinsic_signedness"]))
    details["intrinsic_embedding_text"] = "\n".join((
        f"Name: {name}",
        f"ISA: {details['intrinsic_isa']}",
        f"Operation: {operation.replace('_', ' ')}",
        f"Aliases: {'; '.join(aliases) or operation.replace('_', ' ')}",
        f"Input: {input_text}",
        f"Output: {details['intrinsic_output_type'] or 'unknown'} ({details['intrinsic_output_shape']})",
        f"Vector: {'; '.join(vectors) or 'not specified'}",
        f"Predication: {details['intrinsic_predication']}",
        f"Memory: {details['intrinsic_memory_behavior']}",
        f"Taxonomy: {INTRINSIC_TAXONOMY_VERSION}",
        f"Behavior: {description or operation.replace('_', ' ')}",
    ))
    return details

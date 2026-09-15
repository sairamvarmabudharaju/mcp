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

"""Intrinsic-intent analysis and structured retrieval scoring."""

from __future__ import annotations

import re
from typing import Any

QUERY_OPERATION_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("gather_load", ("gather load", "gather from memory", "gather")),
    ("scatter_store", ("scatter store", "scatter to memory", "scatter")),
    (
        "reduction_add",
        (
            "horizontal add", "horizontal sum", "add across vector",
            "add across lanes", "sum lanes", "sum vector elements",
            "vector reduction add",
        ),
    ),
    (
        "fused_multiply_add",
        ("fused multiply add", "fma", "multiply add to accumulator"),
    ),
    (
        "multiply_accumulate",
        ("multiply accumulate", "multiply and accumulate", "mla"),
    ),
    (
        "multiply_subtract",
        ("multiply subtract", "multiply and subtract", "mls"),
    ),
    (
        "table_lookup",
        ("table lookup", "lookup table", "byte shuffle", "vector shuffle", "shuffle bytes", "shuffle"),
    ),
    (
        "predicate_all_true",
        ("all true predicate", "every element true", "all lanes true"),
    ),
    ("count_leading_zeros", ("count leading zeros", "leading zero count", "clz")),
    ("reinterpret", ("reinterpret", "bit cast", "bitcast")),
    ("widen", ("widening", "widen", "sign extend", "zero extend")),
    ("narrow", ("narrowing", "narrow")),
    ("zip", ("zip interleave", "zip vectors", "interleave two vectors", "interleave")),
    ("duplicate", ("load and replicate", "load and duplicate", "broadcast", "splat", "duplicate", "dup")),
    ("compare", ("comparison", "compare", "equal to", "greater than", "less than")),
    ("predicate", ("predicate", "predication", "while less than")),
    ("store", ("store to memory", "write to memory", "store")),
    ("load", ("load from memory", "read from memory", "load")),
    ("multiply", ("vector multiply", "multiply vectors", "multiplication", "multiply")),
    ("subtract", ("vector subtract", "subtraction", "subtract")),
    ("add", ("vector add", "addition", "add")),
    ("crc", ("cyclic redundancy check", "crc32", "crc")),
)

SYMBOL_PATTERN = re.compile(
    r"(?<![a-z0-9])(?:_?mm(?:256|512)?_[a-z0-9_]+|__[a-z0-9_]+|"
    r"sv[a-z0-9_\[\]]{3,}|v[a-z0-9\[\]]{2,}_[a-z0-9_\[\]]+)(?![a-z0-9])",
    re.IGNORECASE,
)
TYPE_PATTERN = re.compile(
    r"\b(?:sv(?:u?int|float|bfloat)\d+_t|svbool_t|(?:u?int|float|bfloat|poly)\d+"
    r"(?:x\d+(?:x\d+)?)?_t)\b",
    re.IGNORECASE,
)
FIXED_VECTOR_PATTERN = re.compile(
    r"\b(?P<kind>u?int|float|bfloat|poly)(?P<bits>\d+)x(?P<lanes>\d+)(?:x\d+)?_t\b",
    re.IGNORECASE,
)
SCALAR_KIND_PATTERN = re.compile(
    r"\b(?P<kind>unsigned|signed|uint|int|float|bfloat|poly)(?P<bits>8|16|32|64)\b",
    re.IGNORECASE,
)
VECTOR_INTENT_TERMS = {
    "lane", "lanes", "vector", "vectors", "register", "intrinsic",
    "predicate", "predication", "neon", "sve", "sve2", "sme", "sme2",
}


def _normalized_phrase_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[_\-\[\](),.:;/]+", " ", text.lower())).strip()


def _operation(text: str) -> str:
    normalized = _normalized_phrase_text(text)
    for operation, aliases in QUERY_OPERATION_ALIASES:
        if any(
            re.search(
                rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])",
                normalized,
            )
            for alias in aliases
        ):
            return operation
    return ""


def _type_query_details(query: str) -> dict[str, Any]:
    types = [match.group(0).lower() for match in TYPE_PATTERN.finditer(query)]
    lanes = set()
    widths = set()
    element_bits = set()
    signedness = set()
    for match in FIXED_VECTOR_PATTERN.finditer(query):
        bits = int(match.group("bits"))
        lane_count = int(match.group("lanes"))
        lanes.add(lane_count)
        widths.add(bits * lane_count)
        element_bits.add(bits)
        kind = match.group("kind").lower()
        signedness.add({
            "uint": "unsigned", "int": "signed", "float": "floating-point",
            "bfloat": "floating-point", "poly": "polynomial",
        }[kind])
    for match in SCALAR_KIND_PATTERN.finditer(_normalized_phrase_text(query)):
        element_bits.add(int(match.group("bits")))
        kind = match.group("kind").lower()
        if kind in {"uint", "unsigned"}:
            signedness.add("unsigned")
        elif kind in {"int", "signed"}:
            signedness.add("signed")
        elif kind in {"float", "bfloat"}:
            signedness.add("floating-point")
        else:
            signedness.add("polynomial")
    normalized_query = _normalized_phrase_text(query)
    if re.search(r"\bfloats?\b", normalized_query):
        element_bits.add(32)
        signedness.add("floating-point")
    if re.search(r"\bdoubles?\b", normalized_query):
        element_bits.add(64)
        signedness.add("floating-point")
    for match in re.finditer(
        r"\b(\d+)\s+(?:lanes?|elements?|floats?|doubles?|bytes?|integers?|ints?)\b",
        query,
        re.IGNORECASE,
    ):
        lanes.add(int(match.group(1)))
    for symbol in SYMBOL_PATTERN.findall(query):
        x86 = re.match(
            r"_?mm(?P<width>256|512)?_[a-z0-9_]*"
            r"(?:(?:epi|epu)(?P<int_bits>8|16|32|64)|(?P<float_kind>ps|pd))",
            symbol,
            re.IGNORECASE,
        )
        if not x86:
            continue
        width = int(x86.group("width") or 128)
        bits = (
            int(x86.group("int_bits"))
            if x86.group("int_bits")
            else 32 if x86.group("float_kind") == "ps" else 64
        )
        widths.add(width)
        element_bits.add(bits)
        lanes.add(width // bits)
        if not x86.group("float_kind"):
            # Packed integer shuffle operations treat lanes as raw bit patterns.
            signedness.add("unsigned")
        signedness.add(
            "floating-point" if x86.group("float_kind") else "signed"
        )
    if len(lanes) == 1 and len(element_bits) == 1:
        widths.add(next(iter(lanes)) * next(iter(element_bits)))
    return {
        "types": types,
        "lane_counts": sorted(lanes),
        "vector_width_bits": sorted(widths),
        "element_bits": sorted(element_bits),
        "signedness": sorted(signedness),
    }


def analyze_intrinsic_query(query: str) -> dict[str, Any]:
    """Extract intrinsic intent plus operation, ISA, type and shape constraints."""
    lower = query.lower()
    symbols = [match.group(0).lower() for match in SYMBOL_PATTERN.finditer(lower)]
    isa = []
    for name in ("sme2", "sme", "sve2", "sve", "neon", "aarch64"):
        if re.search(rf"\b{name}\b", lower):
            isa.append(name)
    operation = _operation(query)
    type_details = _type_query_details(query)
    normalized_words = set(_normalized_phrase_text(query).split())
    signal_count = (
        (2 if symbols else 0)
        + (2 if type_details["types"] else 0)
        + (1 if isa else 0)
        + (1 if "intrinsic" in normalized_words else 0)
        + (1 if operation else 0)
        + (1 if normalized_words & VECTOR_INTENT_TERMS else 0)
    )
    is_x86_mapping = any(
        symbol.lstrip("_").startswith("mm") for symbol in symbols
    )
    intent = bool(symbols or is_x86_mapping or signal_count >= 2)
    output_shape = ""
    if operation == "reduction_add":
        output_shape = "scalar"
    elif operation in {"predicate", "predicate_all_true", "compare"}:
        output_shape = "predicate"
    memory_behavior = ""
    if operation in {"load", "gather_load"}:
        memory_behavior = "read"
    elif operation in {"store", "scatter_store"}:
        memory_behavior = "write"
    return {
        "intent": intent,
        "symbols": symbols,
        "x86_mapping": is_x86_mapping,
        "operation": operation,
        "isa": isa,
        "output_shape": output_shape,
        "memory_behavior": memory_behavior,
        **type_details,
    }


def is_intrinsic_metadata(metadata: dict[str, Any]) -> bool:
    return (
        str(metadata.get("doc_type", "")).strip().lower() == "intrinsic"
        and bool(metadata.get("intrinsic_name"))
    )


def _candidate_dimension_values(
    metadata: dict[str, Any], candidate_field: str
) -> set[Any]:
    """Prefer output-vector dimensions over mixed input/output aggregates."""
    output_type = str(metadata.get("intrinsic_output_type", ""))
    output_vector = FIXED_VECTOR_PATTERN.search(output_type)
    if not output_vector:
        return set(metadata.get(candidate_field, []) or [])
    bits = int(output_vector.group("bits"))
    lanes = int(output_vector.group("lanes"))
    kind = output_vector.group("kind").lower()
    if candidate_field == "intrinsic_lane_counts":
        return {lanes}
    if candidate_field == "intrinsic_vector_width_bits":
        return {bits * lanes}
    if candidate_field == "intrinsic_element_bits":
        return {bits}
    if candidate_field == "intrinsic_signedness":
        return {{
            "uint": "unsigned",
            "int": "signed",
            "float": "floating-point",
            "bfloat": "floating-point",
            "poly": "polynomial",
        }[kind]}
    return set(metadata.get(candidate_field, []) or [])


def intrinsic_score_components(
    analysis: dict[str, Any], metadata: dict[str, Any]
) -> dict[str, float]:
    """Score independent structured agreements and explicit contradictions."""
    if not analysis.get("intent") or not is_intrinsic_metadata(metadata):
        return {}
    components: dict[str, float] = {}
    name = str(metadata.get("intrinsic_name", "")).lower()
    arm_symbols = [
        symbol
        for symbol in analysis["symbols"]
        if not symbol.lstrip("_").startswith("mm")
    ]
    if name and name in arm_symbols:
        components["exact_symbol"] = 3.0
    elif name and any(
        name.startswith(f"{symbol}[")
        or name.startswith(f"{symbol}_")
        or symbol.startswith(f"{name}[")
        for symbol in arm_symbols
    ):
        components["symbol_family"] = 2.5

    query_operation = analysis.get("operation", "")
    candidate_operation = str(metadata.get("intrinsic_operation", ""))
    if query_operation:
        components["operation"] = (
            1.4 if query_operation == candidate_operation else -1.25
        )
        mnemonic = re.sub(r"^sv|^v", "", name)
        if query_operation == candidate_operation:
            operation_stems = {
                "multiply_accumulate": (("mla", 0.65), ("mad", 0.35)),
                "fused_multiply_add": (("fma", 0.50),),
                "multiply_subtract": (("mls", 0.50),),
            }.get(query_operation, ())
            for stem, weight in operation_stems:
                if mnemonic.startswith(stem):
                    components["operation_mnemonic"] = weight
                    break

    query_isa = set(analysis.get("isa", []))
    candidate_isa = str(metadata.get("intrinsic_isa", "")).lower()
    if query_isa:
        if candidate_isa in query_isa:
            components["isa"] = 0.65
        elif any(
            candidate_isa.startswith(value) or value.startswith(candidate_isa)
            for value in query_isa
        ):
            components["isa"] = 0.20
        else:
            components["isa"] = -0.45

    query_types = set(analysis.get("types", []))
    candidate_types = {
        str(value).lower()
        for value in (
            list(metadata.get("intrinsic_input_types", []) or [])
            + [metadata.get("intrinsic_output_type", "")]
        )
        if value
    }
    if query_types:
        components["datatype"] = (
            1.1 * len(query_types & candidate_types) / len(query_types)
            if query_types & candidate_types
            else -0.55
        )

    dimensions = (
        ("lane_count", "lane_counts", "intrinsic_lane_counts", 0.65),
        ("vector_width", "vector_width_bits", "intrinsic_vector_width_bits", 0.45),
        ("element_size", "element_bits", "intrinsic_element_bits", 0.45),
        ("signedness", "signedness", "intrinsic_signedness", 0.45),
    )
    for label, query_field, candidate_field, weight in dimensions:
        expected = set(analysis.get(query_field, []) or [])
        actual = _candidate_dimension_values(metadata, candidate_field)
        if expected:
            components[label] = weight if expected & actual else -weight * 0.6

    output_shape = analysis.get("output_shape")
    if output_shape:
        components["output_shape"] = (
            0.55
            if output_shape == metadata.get("intrinsic_output_shape")
            else -0.45
        )

    memory_behavior = analysis.get("memory_behavior")
    if memory_behavior:
        candidate_memory = str(metadata.get("intrinsic_memory_behavior", "")).lower()
        components["memory_behavior"] = (
            0.45 if memory_behavior in candidate_memory else -0.35
        )

    if analysis.get("x86_mapping"):
        aliases = " ".join(metadata.get("intrinsic_aliases", []) or []).lower()
        if query_operation and (
            query_operation == candidate_operation
            or query_operation.replace("_", " ") in aliases
        ):
            components["x86_operation_mapping"] = 0.50
    return components


def intrinsic_candidate_search(
    query: str,
    metadata: list[dict[str, Any]],
    k: int,
) -> list[dict[str, Any]]:
    """Retrieve intrinsic parents separately using structured metadata."""
    analysis = analyze_intrinsic_query(query)
    if not analysis["intent"]:
        return []
    scored = []
    for item in metadata:
        if item.get("chunk_index", 1) != 1:
            continue
        components = intrinsic_score_components(analysis, item)
        score = sum(components.values())
        if not components or score <= 0:
            continue
        scored.append((score, str(item.get("intrinsic_name", "")), item, components))
    scored.sort(key=lambda value: (-value[0], value[1]))
    return [
        {
            "metadata": item,
            "intrinsic_rank": rank,
            "intrinsic_score": score,
            "intrinsic_components": components,
        }
        for rank, (score, _, item, components) in enumerate(scored[:k], start=1)
    ]

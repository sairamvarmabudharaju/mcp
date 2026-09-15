# Intrinsic structured retrieval evaluation — 2026-09-14

## Result

Structured intrinsic enrichment and intrinsic-aware ranking meet the acceptance
targets through the real sequential MCP stdio protocol.

| Metric | Previous candidate | Structured compact candidate |
|---|---:|---:|
| Intrinsic H@1 | 50.0% | 81.3% |
| Intrinsic H@3 | 50.0% | 93.8% |
| Intrinsic H@5 | 50.0% (8/16) | 100.0% (16/16) |
| Intrinsic strict H@5 | 50.0% | 93.8% (15/16) |
| Intrinsic MRR | 0.500 | 0.880 |
| Broad golden H@5 | 89.6% | 92.3% |
| Broad golden strict H@5 | 83.6% | 86.3% |
| Hive-mind H@5 | 100.0% | 100.0% |

The complete broad metrics were:

| Image | Suite | n | H@1 | H@3 | H@5 | strict H@5 | MRR |
|---|---|---:|---:|---:|---:|---:|---:|
| `arm-mcp:hivemind-selective-intrinsic` | golden | 183 | 75.4% | 85.8% | 89.6% | 83.6% | 0.809 |
| `arm-mcp:intrinsic-structured-compact` | golden | 183 | 77.0% | 87.4% | 92.3% | 86.3% | 0.828 |
| `arm-mcp:hivemind-selective-intrinsic` | hive-mind | 30 | 93.3% | 100.0% | 100.0% | 100.0% | 0.967 |
| `arm-mcp:intrinsic-structured-compact` | hive-mind | 30 | 93.3% | 100.0% | 100.0% | 100.0% | 0.967 |

The paired broad comparison had eight H@5 gains and three losses. Hive-mind
had no rank or hit changes.

## Candidate construction

The candidate used:

- compact, taxonomy-versioned dense documents for all 10,384 intrinsics;
- explicit `doc_type: Intrinsic` and structured metadata for operation, ISA,
  types, vector shape, signedness, predication, and memory behavior;
- a separate intrinsic candidate pass fused with lexical, dense, and BM25
  retrieval;
- structured agreement bonuses and contradictory-operation penalties;
- unchanged dense vectors for all general documentation.

The intrinsic vectors replaced the previous intrinsic vectors at the same index
keys. The index therefore remained at 48,092 vectors. No additional intrinsic
vectors are justified by the measured result.

Across all 10,384 intrinsic records, the largest compact document was 209 tokens
with the production tokenizer. No compact intrinsic document exceeded the
256-token model limit.

## Reproduction artifacts

The comparison artifacts are under:

```text
/private/tmp/intrinsic-structured-compact-real-mcp-20260914
```

The local-only candidate image was:

```text
arm-mcp:intrinsic-structured-compact
```

The evaluator command was:

```bash
python3 embedding-generation/evaluate_mcp_retrieval.py \
  --image arm-mcp:hivemind-selective-intrinsic \
  --image arm-mcp:intrinsic-structured-compact \
  --output-dir /private/tmp/intrinsic-structured-compact-real-mcp-20260914
```

Focused regression tests passed 62/62. The broader non-Docker repository suite
passed 202 tests and 2 subtests. The separate real-MCP run exercised server
startup, initialization, and `knowledge_base_search` over 426 protocol calls
across the two images.

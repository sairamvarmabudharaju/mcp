# knowledge_base_search retrieval comparison — 2026-09-14

Direct calls to `arm_kb_search.search` (k=5) with the branch code and an index
built by the branch's own `local_vectorstore_creation.py`, against `main`'s code
on the index its builder produces from the same chunks. Two corpora:

- **snapshot**: the 2026-08-23 chunk set (34,993 chunks); the golden set was
  grounded on it, so it is the fairest golden comparison.
- **fresh**: a full acquisition on 2026-09-14 with this branch's
  `generate-chunks.py` (27,370 chunks + 10,384 intrinsics); it contains the
  install-guide and learning-path rows this branch adds to the catalog.

Suites: `evals/golden/eval_golden.json` (183 graded queries, dev/holdout),
`evals/hivemind/eval_hivemind.json` (30 acceptance queries), the 35-query core
draft, and the 570-question `eval_questions.json` regression gate.

## Results

| suite | main, snapshot | **branch, snapshot** | main, fresh | **branch, fresh** |
|---|---|---|---|---|
| golden hit@5 / strict hit@5 / hit@1 | 86.9 / 77.6 / 77.6 | **92.9 / 88.5 / 79.2** | 85.8 / 75.4 / 73.2 | **91.8 / 85.2 / 76.0** |
| golden holdout hit@5 (57) | 82.5 | **91.2** | 82.5 | **91.2** |
| core draft hit@5 / hit@1 (35) | 62.9 / 45.7 | **77.1 / 65.7** | 65.7 / 48.6 | **80.0 / 65.7** |
| hive-mind hit@5 / hit@1 (30) | 13.3 / 10.0 | 56.7 / 26.7 ¹ | 63.3 / 53.3 | **100 / 90.0** |
| regression-570 hit@5 / hit@1 | **97.9** / 86.5 | 96.8 / 85.1 ² | 97.0 / 85.1 | **97.2** / 84.9 |
| query latency mean / p95 ms (80 golden queries, warm) | 135 / 228 | 159 / 253 | 135 / — | 166 / 266 |
| index rows / metadata / vectors | 34,993 / 105 MB / 59 MB | 49,493 / 120 MB / 80 MB | 37,754 / 109 MB / 61 MB | 53,132 / 135 MB / 85 MB |

¹ 13 of the 30 hive-mind queries (Whisper on Android ×10, Chrome ×3) have no
answer page in the 2026-08-23 snapshot; 17/30 is the ceiling there and the
branch reaches it. On the fresh corpus, which contains those pages, it is
30/30 (28 at rank 1).

² Seven questions leave the top 5 on the snapshot (R018, R020, R059, R104,
R131, R532, R570); on the fresh corpus the gate is above main. In every case
the expected page is at rank 6–9 and the pages above it are learning-path
sub-pages or newer learning paths on the same topic outranking an edX video or
a learning-path root (for example the five `loop-reflowing` autovectorization
sub-pages above the `loop-reflowing/` root). The same code on the *old*
one-vector snapshot index scores the same 96.8, so this is ranking, not the
new index.

Twenty intrinsic paraphrases that appear in no suite ("population count of
each byte in a neon vector", "sve predicated select between two vectors", …):
18/20 in the top 5 (main's ranking on the same index: 12/20). Conceptual
SVE/SME questions ("difference between neon and sve", "what is the sve vector
length") return no intrinsic pages.

## Where the golden gains come from (snapshot, hit@5 by area, main → branch)

install-guides 20 → 100, ecosystem-dashboard strict 35.7 → 85.7, intrinsics
62.5 → 93.8, ampere 80 → 100, servers-cloud 90.6 → 92.2. Lost: G01 "zephyr on
cortex-m55 with ethos-u55" (root learning path 1 → 7).

## Reproduce

```bash
# regression gate
./run-question-eval.sh
# golden set on a local index
python evals/golden/evaluate_golden.py --model-path .cache/embedding-model --output golden.json
# both suites through the real MCP protocol, two local images
python evaluate_mcp_retrieval.py --image armlimited/arm-mcp:latest --image arm-mcp:candidate --output-dir reports/retrieval
```

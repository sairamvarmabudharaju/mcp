# Knowledge-base retrieval evaluations

This directory keeps two complementary query suites:

- `golden/eval_golden.json`: 183 broad, graded queries split into development
  and holdout sets. The labels are still marked `draft` pending a human review.
- `hivemind/eval_hivemind.json`: 30 targeted acceptance and consistency
  queries for Docker installation, Android Whisper, AWS CLI, Chrome, and
  Graviton web-app retrieval.

The golden set supports exact, page, prefix, starts-with, and regular-expression
URL matching. Grade 2 is a direct answer; grade 1 is an acceptable partial
answer. See `golden/EVAL_GOLDEN.md` for the authoring and review methodology.

## Real MCP evaluation

Run both suites against one or more local images. Images are evaluated in one
sequential stdio session each using the actual MCP protocol and
`knowledge_base_search` tool call:

```bash
python evaluate_mcp_retrieval.py \
  --image armlimited/arm-mcp:latest \
  --image arm-mcp:candidate \
  --output-dir reports/retrieval-comparison
```

The first image is the comparison baseline. The runner writes per-image scored
rows, raw MCP responses, server logs, `comparison.json`, and a compact
`comparison.md`. Use `--suite golden`, `--suite hivemind`, or
`--split dev|holdout` to select a slice.

The direct local-index evaluator remains available for golden-set development:

```bash
python evals/golden/evaluate_golden.py \
  --model-path .cache/embedding-model \
  --output reports/golden-local.json
```

Keep the original 570-question `eval_questions.json` as the saturated
regression gate; use the golden set to measure broader capability and the
hive-mind set to measure the requested acceptance area.

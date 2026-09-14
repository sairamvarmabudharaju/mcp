# Golden query set for `knowledge_base_search`

`eval_golden.json` is a hand-built set of queries phrased the way people and coding
agents actually use the Arm MCP server, each grounded to the page(s) in the corpus
that answer it. It exists because the 570-question `eval_questions.json` is
saturated (about 98% hit@5): most of its questions are the page title with
"How do I" in front, so it cannot tell a good ranker from a title index.

Run it with `evaluate_golden.py`; keep `eval_questions.json` as the regression gate.

## Design principles (and where they come from)

| Principle | Applied here | Source |
|---|---|---|
| Queries must mirror real traffic, not paraphrase the documents | Three forms: terse agent tool calls, natural user questions, pasted errors. 57 of 183 items use vocabulary the target page does not. | Anthropic, *Demystifying evals for AI agents* (2026); OpenAI evaluation best practices; Chroma, *Generative Benchmarking* (2025) |
| A saturated eval becomes a regression suite; build a harder one for capability | This set targets 50-80% hit@5 on the current ranker; the old set stays as the ~100% gate. | Anthropic (2026); BRIGHT (Su et al., 2024) |
| Small, high-quality, human-reviewed beats large synthetic | 183 expert-written items with notes and confidence, no generated questions. | Husain & Shankar, *AI Evals FAQ* (2025) |
| Multiple acceptable answers, graded | Every item lists one or more grade-2 ("answers it") URLs and optional grade-1 partials; report strict and lenient. | FreshStack (Thakur et al., 2025); TREC RAG |
| Avoid pooling bias | Gold pages were found by title/URL/keyword search over the corpus, not by looking at what the ranker returns. | Buckley et al., *Bias and the limits of pooling* (2007) |
| Hold out part of the set | `split=holdout` (md5 of id, ~31%) is never used for tuning. Report dev and holdout separately. | RTEB (MTEB, 2025); FreshStack |
| Report per stratum with uncertainty | Breakdown by form, intent, area, difficulty and split; bootstrap 95% CI; paired sign-flip test against a baseline run. | Webber, Moffat & Zobel (2008); Smucker et al. (2007) |

Statistical note: with 183 paired queries the minimum detectable difference in
hit@5 at the usual thresholds is roughly 7-10 points. Smaller effects need more
queries (about 300 for 5 points). Treat per-stratum numbers (n of 10-60) as
directional.

## Schema

```json
{
  "id": "B03",
  "query": "Is redis available for arm?",
  "form": "user_question",          // agent_query | user_question | error_paste
  "intent": "compat",               // howto | troubleshoot | compare | compat | tuning | migrate | intrinsic_lookup | concept | provider
  "area": "ecosystem-dashboard",
  "difficulty": "vocab_shift",      // easy | vocab_shift | near_neighbor | ambiguous | symbol
  "expected": [
    {"url": "https://www.arm.com/developer-hub/ecosystem-dashboard/?package=redis", "match": "exact", "grade": 2},
    {"url": "https://learn.arm.com/learning-paths/servers-and-cloud-computing/redis", "match": "prefix", "grade": 1}
  ],
  "notes": "...", "confidence": "high", "split": "dev",
  "expected_urls": ["..."]          // flat page-level list for the legacy evaluate_retrieval.py
}
```

Match rules (tracking parameters such as `utm_source` are stripped first):

- `page`: result URL without query/fragment equals the expected page.
- `prefix`: result page equals the expected page or sits under it (`.../docker` matches `.../docker/buildx`, not `.../docker-build-cloud`). Use for learning paths where any step answers the question.
- `startswith`: full URL starts with the expected string. Use for intrinsic families (`#q=vclz` matches `vclzq_u32`).
- `exact`: full URL equals the expected one. Required for dashboard rows and single intrinsics, because the legacy page-level match collapses all 1,168 dashboard packages (and all 10,384 intrinsics) into one page.
- `regex`: `re.search` on the full URL.

## Running

```bash
# same inputs as evaluate_retrieval.py
python evaluate_golden.py --model-path .cache/embedding-model --output runs/golden_$(git rev-parse --short HEAD).json

# compare a change against a saved run (paired: improved / regressed / p-value)
python evaluate_golden.py --model-path .cache/embedding-model --baseline runs/golden_<base>.json

# tune on dev only, then confirm on holdout
python evaluate_golden.py ... --split dev
python evaluate_golden.py ... --split holdout
```

Headline metric: lenient hit@5 (any grade), with strict hit@5 (grade 2 only),
hit@1 and MRR as diagnostics. `distinct pages in top 5` exposes the fragment-level
duplicate problem (the same page returned several times).

## Adding or reviewing items

1. Write the query first, in the voice of the user or agent, without looking at search results.
2. Find the answer page(s) by searching titles, URLs and keywords in `metadata.json`, not by running the ranker. Record why in `notes`.
3. Give at least one grade-2 URL. Add grade-1 URLs for pages a reasonable person would also accept.
4. Pick the narrowest match rule that is still fair (`page` for a specific step, `prefix` for a whole learning path, `exact` for dashboard/intrinsics).
5. Tag form, intent, area and difficulty. Set `confidence: medium` when the page only implicitly answers the query, and say so in `notes`.
6. Do not tune on holdout items. Do not change an item's `id` (the split derives from it).
7. Review protocol for the draft labels: a second reviewer checks a 20% sample and every `medium` item; disagreements are resolved in `notes`. Record reviewer and date in the file header.

## Known limits of v1

- Labels were drafted by an LLM-assisted process against the 2026-08-23 corpus snapshot and are marked `review_status: draft` until a human pass is done.
- No real traffic yet: the server's `mcp-traffic.jsonl` is only written inside the container unless a workspace is mounted. Once logs exist, mine query strings into the `agent_query` stratum and re-weight by traffic.
- Out-of-corpus queries (nothing in the knowledge base answers them) are not scored; the server has no relevance threshold to test against.
- Corpus drift: pages move or disappear. Re-validate expected URLs against the current `metadata.json` when the eval starts reporting errors or unexplained misses.

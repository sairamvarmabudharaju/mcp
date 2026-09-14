# Golden-set research for `knowledge_base_search` (2026-09-09)

Compiled from primary sources by a research pass on 2026-09-09. Used to design `eval_golden.json`.

## 1. Building golden sets / eval methodology

- **AI Evals FAQ** — Husain & Shankar, 2025, https://hamel.dev/blog/posts/evals-faq/ — Do error analysis on real traces before writing evaluators; binary pass/fail plus a written critique; label ~100-200 examples per failure mode; for RAG, score retrieval separately with Recall@k / MRR; add each production failure to the CI set as a regression case; avoid unstructured "give me test queries" prompting.
- **A Field Guide to Rapidly Improving AI Products** — Husain, O'Reilly, Apr 2025, https://hamel.dev/blog/posts/field-guide/ — Generate synthetic inputs along explicit dimensions (features × scenarios × personas), ground them in real system data, verify scenario coverage.
- **Demystifying evals for AI agents** — Anthropic, Jan 2026, https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents — Start with 20-50 tasks from real failures; a good task is one two domain experts would grade the same way; when an eval saturates, graduate it to a regression suite and build a harder capability eval.
- **Writing effective tools for agents** — Anthropic, 2025, https://www.anthropic.com/engineering/writing-tools-for-agents — Eval prompts inspired by real-world uses; record accuracy, tool-call count, tokens, errors; verifiers must not reject correct answers over spurious differences.
- **Create strong empirical evaluations** — Anthropic platform docs, https://platform.claude.com/docs/en/test-and-evaluate/develop-tests — Mirror the real task distribution; include edge cases; grade with a different model than the one under test.
- **Evaluation best practices** — OpenAI docs, https://developers.openai.com/api/docs/guides/evaluation-best-practices — Mix production data with expert-curated cases; anti-pattern: eval sets that do not reproduce production traffic; calibrate automated graders against human labels.
- **Rankers, Judges, and Assistants** — Balog, Metzler, Qin (Google), SIGIR 2025, https://arxiv.org/abs/2503.19092 — LLM judges are biased toward LLM-based rankers and struggle with subtle system differences.
- **LLM-Evaluation Tropes** — Dietz et al., 2025, https://arxiv.org/abs/2504.19076 — Circularity when one LLM generates queries, judges, and sits in the pipeline.
- **Variations in relevance judgments** — Voorhees, IP&M 2000 — Assessor disagreement shifts absolute scores but leaves system rankings stable; compare systems relatively.
- **Bias and the limits of pooling** — Buckley, Dimmick, Soboroff, Voorhees, IR Journal 2007 — Small pools favor documents containing query words; unjudged documents penalize new systems.

## 2. Realistic technical-documentation retrieval benchmarks

- **FreshStack** — Thakur et al. (Waterloo/Databricks), NeurIPS 2025 D&B, https://arxiv.org/abs/2504.13128 — Real Stack Overflow questions on niche technical topics; pooled candidates from BM25 + three dense models; nugget-level binary support judgments; fusion beats every single model. Metrics: alpha-nDCG@10, Coverage@20, Recall@50.
- **BRIGHT** — Su et al., 2024 (ICLR 2025), https://arxiv.org/abs/2407.12883 — 1,384 real queries with low lexical overlap; positives confirmed by multiple experts; BM25 14.5 vs best dense 22.5 nDCG@10; contamination-robust.
- **BRIGHT-Pro** — Zhao et al., ACL 2026, https://arxiv.org/abs/2605.04018 — Multi-aspect gold evidence; static and agentic evaluation protocols.
- **Direct Corpus Interaction** — Li et al., May 2026, https://arxiv.org/abs/2605.05242 — Agents do better with grep-like lexical access than a fixed similarity interface; evaluate the interface, not only the embedder.
- **CORE-Bench** — Zhang et al., EMNLP 2026, https://arxiv.org/abs/2606.11864 — Embedding models drop sharply from traditional code search to agentic code retrieval.
- **CodeRAG-Bench** — Wang et al., NAACL Findings 2025, https://arxiv.org/abs/2406.14497 — Retrievers often fail to fetch useful documentation context.
- **HumanMCP** — Laddha et al., Feb 2026, https://arxiv.org/abs/2602.23367 — Persona-driven queries for MCP tools; homogeneous synthetic queries inflate reliability.
- **BrowseComp-Plus** — Aug 2025, https://arxiv.org/abs/2508.06600 — Fixed corpus with human-verified evidence and mined hard negatives.
- **TREC 2025 RAG overview** — https://arxiv.org/abs/2603.09891 — Graded 0-4 relevance; LLM ensemble judging with human calibration; reports nDCG, Recall@100, nugget support.
- **AutoNuggetizer** — Pradeep et al., 2024, https://arxiv.org/abs/2411.09607; **Support Evaluation: Human vs LLM Judges** — Thakur et al., SIGIR 2025, https://arxiv.org/abs/2504.15205 — Human post-editing of LLM labels is the efficient protocol (agreement 56% from scratch vs 72% post-edited).
- **UMBRELA** — Upadhyay et al., 2024, https://arxiv.org/abs/2406.06519; **Benchmarking LLM relevance-judgment methods** — Arabzadeh & Clarke, 2025, https://arxiv.org/abs/2504.12558.
- **BEIR** — Thakur et al., NeurIPS 2021, https://arxiv.org/abs/2104.08663; **RTEB** — MTEB maintainers, Oct 2025, https://huggingface.co/blog/rteb — public + private held-out splits; a gap between them signals overfitting.
- **Contextual Retrieval** — Anthropic, Sept 2024, https://www.anthropic.com/engineering/contextual-retrieval — Reports 1 − recall@20 as failure rate; hybrid BM25 + embeddings + rerank.

## 3. Synthetic query generation

- **InPars** (2022, https://arxiv.org/abs/2202.05144), **Promptagator** (2022, https://arxiv.org/abs/2209.11755), **Doc2Query--** (ECIR 2023, https://arxiv.org/abs/2301.03266) — few-shot generation with round-trip and hallucination filtering.
- **Generative Benchmarking** — Chroma, Apr 2025, https://www.trychroma.com/research/generative-benchmarking — Naive generated queries are more specific to the target document than real ones and inflate every model's score; MTEB rank order failed to predict production rank.
- **Synthetic Test Collections** — Rahmani et al., SIGIR 2024, https://arxiv.org/abs/2405.07767; **Bias in Synthetic Data for Evaluation** — CIKM 2025, https://arxiv.org/abs/2506.10301 — trust relative, not absolute, numbers.
- **PersonaHub** — Ge et al., 2024, https://arxiv.org/abs/2406.20094; **Leakage-free RAG benchmarks** — Liu et al., May 2026, https://arxiv.org/abs/2605.08838; **Systematically Improving RAG** — Jason Liu, 2024, https://jxnl.co/writing/2024/05/22/systematically-improving-your-rag/ — per-chunk synthetic questions hit ~97% recall and are easier than real questions.

## 4. Metrics and statistics

- **Statistical power in retrieval experimentation** — Webber, Moffat, Zobel, CIKM 2008 — 50 topics detect only ~0.06-0.08 AP deltas; ~150 topics for δ≈0.033.
- **Topic set size design** — Sakai, IRJ 2016. **Some common mistakes in IR evaluation** — Fuhr, SIGIR Forum 2017 — MRR is not interval-scale; report effect sizes and CIs. **Smucker, Allan, Carterette**, CIKM 2007 — paired t-test, bootstrap, or permutation.
- Calculation for this project: paired comparison, α=0.05, power 0.8, 10-20% discordant queries → minimum detectable difference ≈ 9-13 points at n=100, 6-9 at n=200, 5-7 at n=300.

## Concrete recommendations applied to `eval_golden.json`

1. 200-300 hard queries plus the frozen 570 as a regression suite (v1 shipped 183).
2. Target 50-80% hit@5 on the current system so the set can rank changes.
3. Sourcing mix: real tool-call logs, expert-written intents, controlled synthetic (v1 is expert-written; logs need a mounted workspace).
4. Strata tagged on every query: form, intent, area, difficulty.
5. Lexical-overlap guard against title-reworded questions.
6. Multiple acceptable answers, graded 2/1, with per-URL match rules.
7. LLM pre-label, human post-edit, second reviewer on a 20% sample.
8. Primary metric lenient hit@5 with strict hit@5, hit@1, MRR as diagnostics.
9. Per-stratum tables with bootstrap CIs; paired permutation test against a pinned baseline.
10. Held-out split never used for tuning (v1: 31%).

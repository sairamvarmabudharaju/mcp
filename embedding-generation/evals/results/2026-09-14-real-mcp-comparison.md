# Real-MCP retrieval comparison — 2026-09-14

## Outcome

The initial production candidate's improvement was concentrated in the targeted
hive-mind area. A follow-up ranking fix preserves the 100% targeted hit@5 while
raising broad draft-golden hit@5 to 89.6%, above both the initial candidate
(81.4%) and the published baseline (86.3%).

The expanded-corpus, old-vectorization isolation ties the initial windowed
candidate on both hit@5 headlines. The lossless overlapping windows therefore
provide no measurable benefit on these 213 queries. Expanded source coverage
plus current ranking is sufficient for all 30 targeted queries. Lossless
identifier handling and result diversity recover the broad ranking regression.

## Method

- 183 broad, graded golden queries plus 30 targeted hive-mind queries.
- One real sequential MCP stdio session per image:
  `initialize` → `notifications/initialized` →
  `knowledge_base_search` `tools/call`.
- No direct invocation of the Python search function for acceptance scoring.
- All retained runs completed without MCP errors.
- The old-vector isolation uses the same 35,259 expanded parent chunks and the
  same embedding model as the candidate, with the base branch's one-vector-per-
  chunk behavior. It contains no parent/window metadata.

## Results

| Configuration | Golden H@1 | Golden H@3 | Golden H@5 | Golden strict H@5 | Targeted H@1 | Targeted H@3/H@5 |
|---|---:|---:|---:|---:|---:|---:|
| Published baseline | 77.6% | 82.5% | 86.3% | 77.0% | 10.0% | 13.3% |
| Current ranking + published vector store | 66.7% | 78.1% | 82.0% | 77.0% | 33.3% | 56.7% |
| Current ranking + expanded corpus + old vectorization | 66.7% | 76.5% | 81.4% | 76.5% | 93.3% | 100% |
| Current ranking + expanded corpus + overlapping windows | 67.2% | 77.6% | 81.4% | 76.0% | 93.3% | 100% |
| Ranking fix + expanded corpus + overlapping windows | 75.4% | 85.8% | 89.6% | 83.6% | 93.3% | 100% |

The ranking fix's bootstrap 95% interval for broad H@5 is 85.2–94.0%. Against
the initial candidate it gains 16 and loses one golden hit@5 case, with 25 rank
improvements and six regressions (sign-flip `p=0.0002`). Against the published
baseline it gains nine and loses three golden hit@5 cases; that smaller net
difference is not statistically conclusive (`p=0.146`). It retains all 30
targeted hit@5 cases.

## What fixed the broad regression

- Query normalization now retains exact technical identifiers while adding
  split or corpus-derived forms, for example `tensorflow tensor flow` and
  `int8x16_t int8x16 t`.
- The sparse index and title/heading reranker add the same safe identifier
  variants, avoiding query-only vocabulary rewriting.
- Explicit platform agreement is rewarded and clear server/mobile or
  server/embedded conflicts are penalized.
- At most two results from one learning-path family can occupy the final list,
  preventing one new path from taking all five positions.

Eight of the ten identifier-related hit@5 failures identified in the initial
candidate are recovered. The server-specific Whisper query is recovered at
rank two while all ten Android Whisper targeted queries remain hits within the
top three.

An ablation that kept the original user string exclusively for dense retrieval
scored 89.1% broad hit@5 and lost the SVE runtime-detection hit at rank five. It
provided no targeted benefit and was not retained.

## What the isolation says

| Step | Targeted H@5 change | Golden H@5 change |
|---|---:|---:|
| Current ranking on the old store | +43.4 points | -4.3 points |
| Expanded/current corpus with old vectorization | +43.3 points | -0.6 points |
| Overlapping lossless windows | 0.0 points | 0.0 points |

Most of the broad regression is already present when current ranking runs on
the published vector store. Corpus expansion and overlapping windows are not
the primary cause.

The old-vector and windowed candidates have identical targeted ranks: 30/30 in
the top three and 28/30 at rank one. On the broad set, windowing produces one
hit@5 gain and one loss, with five rank improvements and five regressions.

The largest broad H@5 movements from published baseline to the initial windowed
candidate are directional because some strata are small:

- ecosystem dashboard: 78.6% → 57.1% (`n=14`)
- servers/cloud: 90.6% → 82.8% (`n=64`)
- intrinsics: 56.3% → 50.0% (`n=16`)
- install guides: 20.0% → 40.0% (`n=5`)

The two new broad hits are the GCC AArch64 cross-toolchain install query and
NumPy compatibility. The eleven lost hits include four ecosystem-dashboard
packages, five servers/cloud cases, one intrinsic, and one embedded-MCU case.
One of the servers/cloud losses is an Android Whisper near-neighbor crowding
out a query explicitly asking for an Arm server.

Those counts describe the initial windowed candidate. After the ranking fix,
the published-baseline comparison has nine gained and three lost golden hit@5
cases. The remaining losses are the ISV-list query, the `int8x16_t` reinterpret
intrinsic query, and the Zephyr Cortex-M55/Ethos-U55 query.

## Reproducibility

| Image | Local image ID |
|---|---|
| Published baseline | `sha256:c6d3c802da84d7f81a176598daa7c405728e31a3d9c2aae77fb2bd1d8d3c5429` |
| Current ranking + old store | `sha256:347dff9d100a2167f87bfd6929e1a35dddd77400019437b719dee0a3eb19be10` |
| Expanded old-vector isolation | `sha256:16a36b6415eefbb7520e1524e26420621f13c560750aa0e67c6fcd3af830308d` |
| Windowed candidate | `sha256:83ebc56b87e48882febc847b839e5ca2899996a1af158f5e2bb4c3188b8b7e32` |
| Ranking-fix candidate | `sha256:c282d7292065bee48bf6f91133633078d056188c15baec44823fd76749af14fb` |

The isolation and initial candidate have identical `search.py`, `resources.py`,
and `server.py` hashes. Their raw scored rows, responses, and server logs are in
`/private/tmp/hivemind-unified-real-mcp-20260914`. Ranking-fix artifacts are in
`/private/tmp/hivemind-ranking-fix-real-mcp-20260914`.

## Caveats

- The golden labels are draft and await human review.
- The golden set was grounded against the 2026-08-23 corpus snapshot, while the
  expanded candidate contains newer pages.
- The expanded source data came from the retained synchronized corpus. A fresh
  internet acquisition did not complete and is not claimed here.
- This result shows that overlapping windows are unnecessary for the measured
  suites; it does not prove they cannot help long documents or unrepresented
  queries.

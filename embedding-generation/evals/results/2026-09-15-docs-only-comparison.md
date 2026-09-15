# Documentation and general retrieval split — 15 September 2026

This change retains documentation/source acquisition, lossless document embedding
windows, parent-aware retrieval, query normalization, URL deduplication and search
diagnostics from `kb-search-lossless-chunking-v3` (`b8703a6`). It defers the dedicated
intrinsic candidate channel, intrinsic intent/ranking bonuses, taxonomy extraction,
and enriched intrinsic embedding/lexical inputs. Intrinsics retain their existing
raw input and one-vector encoding, including the model's prior truncation behavior.

The previous full implementation and its historical results remain available at
commit `b8703a6`. This update stays on `kb-search-lossless-chunking-v3` and adds a
new commit to its existing history. This report replaces the earlier full-version
results document because those numbers are not measurements of this split.

## Measured comparison

All 918 frozen cases were replayed through the split image's real MCP stdio tool;
there were no tool errors and every scored response returned five results. The
reference is the full branch with its rebuilt index on the **same 35,027-parent
published corpus**, using the same cached model/runtime and four-CPU limit.
Success means an expected URL appears in the top five, not answer correctness.

The non-intrinsic subset was fixed before the run: 892 cases. Only the golden
intrinsic area (16 cases) and core intrinsic lookups (10 cases) are excluded;
implementation-guide queries that mention intrinsics remain included.

| Non-intrinsic suite | Full branch hit@5 | Split hit@5 | Full strict hit@5 | Split strict hit@5 |
| --- | ---: | ---: | ---: | ---: |
| Regression, 570 | 554/570 | 554/570 | 554/570 | 554/570 |
| General golden, 167 | 158/167 | 157/167 | 150/167 | 149/167 |
| Hivemind, 30 | 17/30 | 17/30 | 17/30 | 17/30 |
| General core, 25 | 21/25 | 23/25 | 18/25 | 20/25 |
| Real-chat, 100 | 67/100 | 67/100 | 67/100 | 67/100 |

Hivemind's Android Whisper and Chrome targets are absent from this frozen corpus,
so 17/30 is the maximum answerable result here. All 17 successes and their expected
ranks are retained. The new catalog/acquisition changes remain in scope, but this
run does **not** revalidate the full branch's historical fresh-corpus 30/30 claim.

For visibility, including intrinsic cases: golden is 170/183 -> 166/183
(strict 161/183 -> 158/183); core is 26/35 -> 25/35
(strict 21/35 -> 21/35). Expected intrinsic-specific losses are not acceptance
criteria for this split.

## Cases requiring review

The same aggregate regression count hides one loss offset by one gain. Do not
interpret the table as preservation of every individual general-query result.

- R528, "How is Streaming SVE used in SME code?": expected Streaming SVE page
  moves from rank 5 to rank 6. Different candidate contributions let a SIMD-loop
  implementation page take its top-five slot.
- Golden B12, "how to detect at runtime whether the CPU supports SVE in C code":
  expected CPU-feature guide moves from rank 5 to rank 6. Legacy intrinsic vectors
  crowd it out of the dense candidate pool, leaving its lexical contribution.

Three formerly missed general cases enter the top five: R532 (int8 SME
matrix-by-vector, rank 5), core SME06 (SME2 C implementation guide, rank 1), and
core SVE02 (SVE vector-length portability, rank 5). Two existing real-chat hits
move down one position: "arm mcp server?" 2 -> 3 and "What is Graviton" 3 -> 4.
Two other expected ranks improve. No labels were changed to obtain these numbers.

## Validation and limitations

- 140 search, chunking/acquisition, evaluator and package-version tests passed,
  plus two subtests; the remaining CI unit-test selection passed 50 tests.
- Embedding-generation's configured Ruff check passed; `git -c core.whitespace=cr-at-eol diff --check` passed (the source CSV
  retains main's CRLF line endings).
- The CI unit job now uses `--group test` so runtime/package dependencies are
  included alongside pytest. The obsolete taxonomy build trigger/assertion and
  Docker COPY input are absent from this split.
- Focused tests cover empty queries and restoring full parent text when
  `SearchResources` is manually constructed with compact window metadata.
- Reduced builder preparation reproduces all 39,142 documentation inputs and
  metadata rows from the full rebuilt branch exactly. Those document vectors
  were reused; all 10,384 intrinsic inputs, lexical text and vectors were restored
  to main. Serialized vectors were checked for exact equality. Both artifacts
  have 49,526 rows. This avoids redundant embedding work without changing inputs.
- These are comparison images over published `armlimited/arm-mcp:3.0.0` (digest
  `sha256:861496487eb057b7e16230db3b1e9381730a1d22c44944a265b53d82394e2257`),
  not a validation of the locked release build. The published runtime uses
  sentence-transformers 5.7.0 while the project declares >=6. Publishing/pinning
  the intended new vector-store artifact and checking the canonical release
  build remain separate release work. No full-stack integration run is claimed.

The split is prepared for code review with the two general retrieval misses
explicitly recorded. Intrinsic work remains deferred; additional ranking changes
are outside this commit.

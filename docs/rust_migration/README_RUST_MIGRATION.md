# Rust Migration Workspace

This directory is a copy of the repository `src` tree for incremental Rust
migration work. The original `src` package is intentionally left unchanged.

## Current Rust Slice

The first migrated surfaces are:

- LM2011 text tokenization and dictionary counting.
- LM2011 dictionary-token normalization for dictionary/list inputs.
- LM2011 dictionary materialization helpers for cell normalization, active
  extended-dictionary membership checks, and order-preserving word-list
  de-duplication.
- LM2011 document-stat preparation for uncleaned and Python-cleaned text
  feature batches, including column-oriented handoff, metadata row projection,
  token counts, recognized-token counts, matched-token counts, and IDF inputs.
- LM2011 streaming text-feature pass-1 row construction for uncleaned and
  Python-cleaned text batches, including column-oriented handoff, compact
  matched-count JSON payloads, and document frequency inputs.
- LM2011 streaming text-feature pass-2 feature-row construction from pass-1
  shards, including column-oriented handoff, signal proportions, TF-IDF
  weights, and cleaning-policy metadata.
- LM2011 streaming writer incremental `doc_id` duplicate validation for staged
  text-source shards.
- LM2011 batch token-count expression for Polars text columns, preserving
  null propagation while batching non-null text through Rust.
- LM2011 count-only token counting for visible-prefix aggregation paths.
- FinBERT visible-prefix LM2011 token-count batches for original and retained
  sentence text.
- FinBERT bucket-length tuning summary row assembly for fixed bucket ordering
  and empty-bucket fill rows, including a column-oriented handoff before the
  row-dict compatibility bridge.
- LM2011 eager text-feature microbatch span planning for memory-bounded
  production text scoring.
- LM2011 text feature-row construction for token totals, recognized-token
  totals, signal proportions, and paper-formula TF-IDF weights.
- LM2011 SEC/CCM form normalization helpers used by scalar wrappers and
  Polars batch expressions.
- LM2011 SEC/CCM batch form normalization helpers used by LazyFrame
  `map_batches`, replacing row-wise Python callbacks in sample-backbone form
  gates.
- LM2011 text-feature base `normalized_form` construction now uses the
  Rust-backed batch form expression for eager and LazyFrame full-10-K/MD&A
  feature inputs instead of scalar `map_elements`.
- LM2011 post-Refinitiv FinBERT-visible prefix source construction now uses
  the Rust-backed LM token-count batch expression instead of scalar
  `map_elements` for original and visible-prefix document text counts.
- LM2011 post-Refinitiv extension CSV companions now batch JSON serialization
  for signal-input list columns through the Rust extension with Python
  fallback, replacing the final scalar `map_elements` callback in the copied
  package tree.
- LM2011 FF48 SIC mapping text parsing for cached industry-range lookup.
- CCM daily/SEC bridge form-match token normalization used before filtering
  filings by keep-form sets.
- SEC regime form-type normalization used by historical 10-K item-regime
  lookup.
- SEC filing identifier utility helpers for digit-only accession normalization,
  10-digit CIK formatting, and canonical `doc_id` construction.
- SEC utility newline normalization, narrow date parsing, Roman numeral
  parsing, default item-part inference, and bullet-prefix classification used
  by filing cleanup, regime lookup, extraction date handling, and item
  metadata/boundary heuristics.
- SEC extraction EDGAR metadata stripping for ASCII filing bodies, including
  complete `<SEC-HEADER>`/`<Header>`/`<FileStats>`/`<XML_Chars>` block removal,
  synthetic leading-header suffix stripping, truncated leading-header guards,
  and Python fallback for non-ASCII text.
- SEC extraction Part-marker scanning for 10-K/10-Q part heading detection,
  including sparse-layout midline detection, 10-Q form-header rescue,
  TOC-mask skipping, ordered Part I/II filtering, and fallback through the
  existing native/Python chain.
- SEC extraction high-confidence item-text truncation for empty-section stubs,
  successor Item headings, and confirmed later-Part restarts while preserving
  cross-reference and TOC-window guards.
- SEC source filename parsing for raw filing metadata extracted from archive
  member names.
- SEC pipeline completed-year checkpoint JSON decoding for per-year batch
  merge resume state.
- SEC pipeline no-item stats weighted aggregation for full-sample document-type
  and total rows, including a column-oriented Rust handoff while keeping CSV
  I/O in Python.
- Parquet I/O stream-copy and quick magic-byte probing used by verified
  artifact promotion and staged concat validation while keeping PyArrow
  quick/full validation in Python.
- SEC Parquet filing-text stream batch matching by `doc_id`, used by
  multi-surface and suspicious-boundary diagnostics to fetch escalated full
  filings without materializing unmatched `full_text` values.
- SEC filing boundary-authority status classification and public boundary
  payload projection used when converting extraction diagnostics into stable
  item-boundary metadata.
- SEC HTML-audit scalar rendering helpers for safe slugs, boolean/integer
  parsing, item ordering, and quartile bucketing.
- SEC HTML-audit filing status classification for pass/warning/fail report
  grouping and sampling.
- SEC HTML-audit deterministic status/stratified filing sampling, including
  weighted pass/warning/fail allocation, item-count quartile strata, missing
  core-item strata, and Python fallback with exact seeded parity.
- Sentence split quality audit classifiers for small row-wise text scans
  (`numeric_only`, `one_word`, `separator_only`, `table_like`, and terminal
  punctuation checks).
- Sentence split quality residual-fragment classifiers for short, very-short,
  and lowercase fragments.
- Sentence split quality generic-reference continuation classifier used by
  reference-stitch diagnostics.
- Sentence dataset chunk-row expansion for long cleaned item scopes, including
  split-audit row construction.
- Item-text cleaning LM2011 token-count batches for cleaned scope audit and
  drop-rule inputs.
- Item-text cleaning row-audit base payload, review-status, and production
  eligibility helpers used by cleaned-scope audit assembly.
- Item-text cleaning batch audit/cleaned-row finalization for cleaned scope
  artifacts, including drop/review/production metadata and audit snippets.
- Item-text cleaning fused single-text cleaning orchestration for FinBERT
  cleaned scope preprocessing, covering newline normalization, layout-line
  removal, TOC-prefix trimming, item-aware tail truncation, reference-stub
  detection, table-like block removal, blank-run collapse, and non-body text
  detection in one Rust call with Python fallback.
- Item-text cleaning cleaned-scope preparation for FinBERT preprocessing,
  including item-code scope normalization, source newline normalization,
  fused cleaning dispatch, character-count/removal-ratio calculation, and
  prepared-row payload assembly before LM token-count and finalization batches.
- Item-text cleaning manual-audit sample selection, including deterministic
  per-scope/period caps, manual-audit-priority ordering, sample-reason
  projection, and manual-review placeholder columns.
- Item-text cleaning scope-diagnostics/manual-audit batch classification for
  activation status and calendar-year audit period, replacing remaining
  `map_elements` scalar callbacks in those dataframe assembly paths.
- Item-text cleaning manual-audit fallback sampling now also uses the batch
  audit-period expression, removing the fallback-only scalar `map_elements`
  callback while preserving output parity.
- FinBERT section-universe document-type normalization now uses Rust-backed
  SEC raw-form and canonical LM2011 form batch expressions instead of scalar
  `map_elements` callbacks.
- FinBERT sentence provenance benchmark item-code to text-scope normalization
  now uses a Rust-backed batch expression instead of scalar `map_elements`.
- Item 7 LM token floor-sweep threshold summary-row reduction, including
  dropped-row totals, floor/reference-stub counts, and confirmed
  false-positive share metrics.
- Sentence split quality batch flag classification for short/very-short,
  one-word, numeric-only, separator-only, citation stubs, generic-reference
  endings/continuations, citation-prefix-only lines, header-like lines,
  table-like, lowercase-fragment, and missing-terminal-punctuation flags.
- Sentence split quality report item-code ordering for metadata payloads.
- LM2011 Refinitiv document-ownership `KYPERMNO` normalization used by
  row-wise request/universe preparation.
- LM2011 Refinitiv document-ownership date normalization for row-wise workbook
  and API response parsing.
- LM2011 Refinitiv document-ownership target quarter-end derivation used by
  row-wise ownership request construction.
- LM2011 Refinitiv document-ownership target-effective-date and request-bound
  date-window clipping used by ownership request construction.
- LM2011 Refinitiv document-ownership request row construction, including
  filing-date target windows, authority exception matching, request-bound
  clipping, and exclusion reasons.
- LM2011 Refinitiv document-ownership universe-diagnostics summary reduction,
  including a column-oriented detail-frame handoff before the row-dict
  compatibility bridge, doc-set overlap counts, final-status counts, mismatch
  CIK top counts, and overlap rates.
- LM2011 Refinitiv document-ownership membership-prep batch normalization for
  diagnostic `doc_id`, CIK, form/status text, and `KYPERMNO` columns.
- LM2011 Refinitiv document-ownership doc-filing artifact reader `KYPERMNO`
  normalization, replacing the remaining row-wise `map_elements` callback with
  the existing Rust-backed batch normalizer and Python batch fallback.
- LM2011 Refinitiv document-ownership retrieval-sheet matching used when
  parsing filled exact/fallback workbooks.
- LM2011 Refinitiv document-ownership exact institutional hit selection used
  by fallback-request construction and final output assembly.
- LM2011 Refinitiv document-ownership fallback institutional hit selection,
  including latest-date selection, fallback-window cutoff filtering, and
  conflict detection.
- LM2011 Refinitiv document-ownership fallback-request row selection after
  exact-hit and exact-conflict screening.
- Refinitiv ownership/analyst numeric response normalization and institutional
  ownership percentage capping.
- Refinitiv analyst-estimate integer count normalization used for
  `TR.EPSNumberofEstimates` response parsing.
- Refinitiv analyst actuals/estimates fiscal-period label normalization inside
  LSEG response assembly loops.
- Refinitiv analyst month-end shift helper used by forecast revision cutoff
  selection.
- Refinitiv analyst latest-date lookup helper used when selecting forecast
  snapshots on or before cutoff dates.
- Refinitiv analyst estimate date-index freezing used before forecast snapshot
  lookup and revision-base selection.
- Refinitiv analyst estimate snapshot canonicalization for duplicate-conflict
  detection and source request-group/RIC consolidation.
- Refinitiv analyst actual event canonicalization for duplicate actual-EPS
  conflict detection and source request-group/RIC consolidation.
- Refinitiv analyst normalized-event row construction, including selected
  forecast snapshot lookup, missing/nonunique fiscal-period derivation
  rejections, forecast revision calculations, and duplicate normalized-event
  conflict detection.
- Refinitiv analyst source request-group list normalization used by actual
  event and estimate snapshot canonicalization.
- Refinitiv analyst LSEG actuals and estimates request-item row construction
  from request-group windows.
- Refinitiv analyst LSEG actuals and estimates batch response row
  normalization.
- Refinitiv ownership LSEG universe, document-exact, and document-fallback
  request-item row construction from handoff/request frames.
- Refinitiv ownership LSEG universe, document-exact, and document-fallback
  batch response row normalization.
- Refinitiv LSEG document-recovery unresolved request masking, including
  duplicate returned `doc_id` handling, null returned-ID skipping, null request
  `doc_id` exclusion, retrieval-eligibility gating, and request-row order
  preservation.
- LM2011 Refinitiv document-ownership returned-category normalization and
  institutional-category classification.
- Refinitiv ownership LSEG request month-boundary normalization.
- Refinitiv analyst LSEG item-window date parsing used before interval
  batching.
- Refinitiv ownership LSEG item-window date parsing used before interval
  batching.
- Refinitiv LSEG interval span-day calculation used by interval batch
  planning and candidate scoring.
- Refinitiv LSEG interval-batch candidate evaluation arithmetic used by
  greedy interval batch planning.
- Refinitiv analyst request-group ID hashing for the row-wise request-group
  builder.
- Refinitiv analyst batched request-group ID hashing for eligible authority
  rows in the request-group builder.
- Refinitiv ownership authority scalar helpers for date-span counts,
  tolerance-based value matching, distinct ownership-value sorting,
  ownership candidate-key normalization, source-family labels, and component
  IDs.
- Refinitiv ownership authority allowlist key normalization used by reviewed
  ticker authority decisions, including a column-oriented Rust handoff before
  the row-dict compatibility bridge.
- Refinitiv ownership authority interval merging for date-window
  consolidation.
- Refinitiv ownership authority candidate metric construction, including
  request/result grouping, ownership observation sets, per-PERMNO coverage
  shares, and downstream metadata payloads.
- Refinitiv ownership authority final-panel row reduction from ownership
  result frame columns and selected assignment rows, including selected-RIC
  priority, same-date/category conflict detection, and row-Rust/Python
  fallbacks.
- Refinitiv ownership authority review-required table assembly from decision
  and candidate-metric frame columns, preserving sorted conventional/ticker
  candidate lists and Python-visible table columns.
- Refinitiv ownership authority pairwise alias diagnostics for candidate-RIC
  overlap, value-conflict, benign-alias, regime-split, and review flags.
- Refinitiv ownership authority conventional-component grouping and metadata
  reduction, including a direct candidate/pair metadata handoff before the
  row-dict compatibility bridge, benign-alias union-find collapse, canonical
  member selection, bridge-window merging, ownership union maps, and review
  flags.
- Refinitiv LSEG `stable_hash_id` fast path for scalar and simple nested JSON
  payloads used by request IDs, batch IDs, and plan fingerprints.
- Refinitiv LSEG request-signature hashing for simple field/parameter payloads
  used by batch grouping and interval planners.
- Refinitiv LSEG simple batch item grouping, item-id sorting, chunking, and
  unique-instrument-limit batching.
- Refinitiv LSEG split-batch child metadata construction for retry
  subdivision.
- Refinitiv LSEG error classification for unresolved identifiers, session
  failures, timeouts, and backend overload markers.
- Refinitiv LSEG session-not-opened message detection used by provider retry
  decisions.
- Refinitiv LSEG identifier-list parsing for unresolved-identifier diagnostics.
- Refinitiv LSEG unresolved-identifier message parsing for direct provider
  helper calls.
- Refinitiv LSEG provider integer coercion for response status and byte
  metadata.
- Refinitiv LSEG provider UTF-8 record coercion used by the pandas-to-Polars
  fallback path.
- Refinitiv LSEG provider response-header sanitization for metadata capture.
- Refinitiv LSEG provider response-frame fingerprint hashing for API response
  metadata, including a column-oriented Rust handoff before the row-dict
  compatibility bridge.
- Refinitiv LSEG API execution response row-count aggregation by `item_id`.
- Refinitiv LSEG API execution response row-count aggregation now passes
  plain `item_id` value lists to Rust instead of one-column row dictionaries.
- Refinitiv LSEG API execution item-result detail row construction for
  per-item success/requeue diagnostics.
- Refinitiv LSEG API execution mixed zero/positive ownership-universe
  requeue decision.
- Refinitiv LSEG API execution singleton and split child-batch row planning
  for retry subdivision.
- Refinitiv LSEG batch retry overload classification used by API error policy.
- Refinitiv LSEG daily-limit header/message detection used by API retry and
  stage-defer policy.
- Refinitiv LSEG empty-result classification for final-attempt unresolved
  identifier handling.
- Refinitiv LSEG deterministic batch error-policy classification for
  unresolved identifiers, daily limits, session failures, overloads, fatal
  exceptions, and max-attempt exhaustion.
- Refinitiv LSEG lookup request-item row construction from lookup snapshots.
- Refinitiv LSEG lookup batch response normalization used by lookup API
  staging.
- Refinitiv LSEG stage-audit metadata scalar/list/mapping normalization for
  fetch and stage manifests.
- LM2011 event-panel previous-month-end date bucketing and quarterly
  Fama-MacBeth quarter-start bucketing.
- LM2011 previous-month-end and quarter-start batch date expressions used by
  event-panel price joins, Phase 0 validation SUE diagnostics, and quarterly
  Fama-MacBeth grouping, replacing remaining scalar `map_elements` callbacks.
- LM2011 quarterly Fama-MacBeth weighted-mean and Newey-West standard-error
  numeric helpers.
- LM2011 event-window scalar validation and derived event/post-event day
  offsets used by event-panel builders.
- LM2011 fixed-shape FF3 OLS alpha/RMSE calculation used by event-window
  regression metric builders.
- LM2011 event-window per-document regression metric row construction with
  column-oriented handoff.
- LM2011 event-window daily row expansion for event-screen and event-return
  panel builders, replacing the eager range join with a Rust row-index pair
  selector while preserving Polars schema assembly and downstream semantics.
- LM2011 fixed-shape FF4 OLS coefficient, standard-error, t-statistic, and
  R2 calculation used by monthly strategy factor-loading summaries.
- LM2011 monthly strategy factor-loading row grouping, column-oriented
  preparation, and summary-row construction.
- LM2011 regression-output finite-float sanitizer used before writing model
  fit diagnostics.
- LM2011 Phase 0 validation-audit legacy Appendix-style tokenization used in
  audit packet comparisons.
- LM2011 Phase 0 validation-audit marker/snippet scanning, whitespace
  truncation, truthy-row counting, and series counting used by packet summaries.
- LM2011 Phase 0 validation-audit raw form normalization/count aggregation used
  by broad year-merged corpus distribution checks.
- LM2011 Phase 0 validation-audit form-count row finalization used by corpus
  form distribution comparisons.
- LM2011 Phase 0 validation-audit representative-term document count updates
  used by corpus coverage comparisons.
- LM2011 Phase 0 validation-audit Packet B event-panel attrition row
  aggregation used by backbone-to-event coverage checks.
- LM2011 Phase 0 validation-audit Packet C unit-audit row construction for
  non-null counts and mean absolute value checks.
- LM2011 Phase 0 validation-audit Packet D coverage-reconciliation row
  construction for reported/actual FinBERT backbone denominator diagnostics.
- LM2011 Phase 0 validation-audit Packet A MD&A row payload construction,
  including current/Appendix token counts, recognized-word counts, threshold
  flags, and audit snippets.
- LM2011 Phase 0 validation-audit Packet A token-delta summary aggregation by
  full-10-K and MD&A scope.
- LM2011 Phase 0 validation-audit Packet A summary-frame per-year and overall
  aggregation for marker, threshold, and cleaning diagnostics.
- LM2011 Phase 0 validation-audit Packet A strip-comparison per-year and
  overall aggregation for residual marker and truncation diagnostics.
- LM2011 Phase 0 validation-audit Packet A example-row filtering and priority
  ordering for manual audit excerpts.
- LM2011 extension text-scope alias normalization used by extension
  specification and feature builders.
- LM2011 extension normal-approximation two-sided p-value calculation for
  extension result tables.
- LM2011 extension result-row conversion for quarterly regression outputs,
  including `n_obs` derivation and normal-approximation p-value calculation,
  now with a column-oriented handoff before the row-dict compatibility bridge.
- LM2011 extension quarterly fit-row construction for common-sample
  fit-comparison diagnostics, including a column-oriented handoff before the
  row-dict compatibility bridge.
- LM2011 extension skipped-quarter diagnostic row construction for
  fit-comparison artifacts, including a column-oriented handoff before the
  row-dict compatibility bridge.
- LM2011 extension quarterly fit-difference row construction for common-sample
  specification comparisons.
- LM2011 extension common-quarter fit-summary aggregation for
  fit-comparison diagnostics.
- LM2011 extension common-quarter fit-comparison aggregation, including
  weighted/equal R2 deltas, Newey-West SEs, t-statistics, and p-values.
- FinBERT robustness-runner fit-comparison row assembly now delegates to the
  Rust-backed LM2011 extension quarterly-fit, skipped-quarter, summary,
  difference, and comparison-row helpers.
- LM2011 Table IA.II monthly strategy result-row expansion from FF4 summary
  rows, including column-oriented handoff.
- LM2011 quarterly Fama-MacBeth final result-row aggregation from retained
  quarter coefficient time series.
- LM2011 cross-sectional OLS design-row and column-oriented preparation for
  quarterly Fama-MacBeth regressions.
- Refinitiv bridge workbook scalar normalization used when reading workbook
  headers and row values.
- Refinitiv bridge extended-workbook boolean cell normalization used for
  diagnostic workbook parsing.
- Refinitiv bridge lookup-text normalization used across handoff and
  diagnostics assembly.
- Refinitiv bridge lookup-result filtering, normalized lookup-result matching,
  and bridge-row LIID parsing.
- Refinitiv bridge ownership-result date normalization for filled ownership
  validation workbook parsing.
- Refinitiv bridge ownership-universe request-date and date-text rendering
  used by handoff and review-row assembly.
- Refinitiv bridge identity-candidate agreement checks used by RIC resolution
  conflict and extension policy.
- Refinitiv bridge material identity-conflict checks for ISIN/CUSIP mismatch,
  RIC-only disagreement, and partial-identity ticker diagnostics.
- Refinitiv Excel ownership-universe retrieval block payload construction,
  including repeated-block input values, cell references, and Workspace
  ownership formula strings while keeping workbook writing in Python.
- Refinitiv Excel LM2011 document-ownership retrieval block payload
  construction, including exact/fallback formula strings, repeated-block cell
  references, and date-preserving input values while keeping workbook writing
  in Python.
- Refinitiv Excel resolution-diagnostic retrieval block payload construction,
  including repeated-block input values, `_xll.RDP.Data` formula strings, and
  diagnostic lookup date references while keeping workbook writing in Python.
- Refinitiv Excel ownership-smoke retrieval block payload construction,
  including repeated-block smoke-test input values while keeping workbook
  writing in Python.
- Refinitiv Excel ownership-validation sheet payload construction, including
  case/sheet grouping, repeated-slot input values, and Workspace ownership
  formula strings while keeping workbook writing in Python.
- Refinitiv Excel extended RIC lookup summary formula payload construction,
  including lookup-range COUNTIF/SUMPRODUCT formulas while keeping workbook
  writing in Python.
- Refinitiv Excel extended RIC lookup formula payload construction, including
  per-identifier lookup, success, pairwise identity, and all-successful
  consistency formulas while keeping workbook writing in Python.
- Refinitiv bridge lookup/accepted candidate extraction used by accepted
  resolution and adjacent identity extension policy.
- Refinitiv bridge accepted-resolution derivation for conventional
  ISIN/CUSIP agreement, conflict, and ticker-candidate diagnostics.
- Refinitiv bridge effective-resolution field derivation for accepted,
  extended, and unresolved RIC rows.
- Refinitiv bridge Step 1 RIC resolution-frame assembly now has
  column-oriented and row-Rust entrypoints before the Python fallback,
  preserving accepted/extended/effective RIC status behavior and output
  column ordering.
- Refinitiv bridge resolution-diagnostic handoff row projection, including
  normalized lookup inputs, request date text, candidate RIC fields, and
  adjacent effective-RIC context.
- Refinitiv bridge accepted-source matching and adjacent-extension source
  choice used by RIC extension policy.
- Refinitiv bridge resolution-diagnostic classifiers for target class,
  conventional-source detection, support scope, and block-reason labels.
- Refinitiv bridge summary reducers for normalized value counts and strict
  `True` record counts used by diagnostic manifests.
- Refinitiv bridge batched strict-`True` field counting used by resolution
  diagnostic manifests.
- Refinitiv bridge resolution-diagnostic class-summary reducer, including
  target grouping, truthy adjacency/support counts, and normalized
  support-scope/block-reason counts.
- Refinitiv bridge ownership-validation retrieval-role sort key used when
  ordering validation workbook blocks and retrieval summaries.
- Refinitiv bridge ownership-validation handoff row planning, including case
  grouping, retrieval-role ordering, workbook sheet assignment, and block slot
  metadata, with a column-oriented handoff before the row-Rust/Python
  fallback chain.
- Refinitiv bridge ownership-validation pairwise result comparison, including
  overlap counts, returned-RIC/category agreement, value-difference summaries,
  corroboration, same-identity RIC-variant support, and conflict flags.
- Refinitiv bridge ownership-validation case-summary row construction,
  including candidate/adjacent ownership-data flags, pair-support counts, and
  validation-bucket assignment, with a column-oriented handoff before the
  row-Rust/Python fallback chain.
- Refinitiv bridge ownership-universe handoff row construction, including
  effective-RIC, conventional-conflict, ticker-fallback, and nonretrievable
  branches, now reading source resolution columns directly in Rust before the
  row-Rust/Python fallback chain.
- Refinitiv bridge failed-lookup classification used by null-RIC rescue
  diagnostics.
- Refinitiv bridge batched failed-lookup classification used by null-RIC
  rescue candidate construction.
- Refinitiv bridge ownership-universe candidate-key normalization used when
  matching filled ownership retrieval workbooks back to handoff rows.
- Refinitiv bridge ownership-smoke lookup-input priority resolution used when
  building null-RIC rescue review samples.
- Refinitiv bridge ownership-smoke sample-row projection, including normalized
  request fields, date text rendering, truthy flags, and count coercion for
  null-RIC rescue review samples.
- Refinitiv bridge alternative-identifier selection used by null-RIC rescue
  diagnostics.
- Refinitiv bridge batched alternative-identifier selection used by null-RIC
  rescue candidate construction.
- Refinitiv bridge ownership-validation result text and numeric value
  normalization.
- Refinitiv document-ownership final-row construction, including exact-hit,
  fallback-hit, conflict, ineligible-review, and no-authority retrieval status
  assignment.
- Multi-surface audit whitespace normalization used in case construction.
- Multi-surface audit boundary snippet-risk detection used in item-boundary
  review case construction.
- Multi-surface audit stable integer sort-key fast path for digit-bearing case
  and sentence identifiers.
- Multi-surface audit ASCII normalized-text index construction used for
  full-report context matching, with Python fallback for non-ASCII text.
- Multi-surface audit ASCII normalized match-bound lookup used by full-report
  context matching, with Python fallback for non-ASCII text.
- Multi-surface audit escalation scoring, cap enforcement, stable ordering,
  and full-report review reason projection for audit-pack case rows.
- Multi-surface audit review-record interleaving and chunk index planning for
  audit-pack chunk files.
- FinBERT sentence confusion-review sample row ID and order assignment before
  neighbor-context attachment and labeling chunk export, including a
  column-oriented handoff from the sorted sample frame before the row-dict
  compatibility bridge.
- FinBERT sentence confusion-review neighbor-target row construction for
  previous/next sentence context joins.
- FinBERT sentence confusion-review allocation row finalization, including
  sample weights, population fractions, and allocation-mode labels.
- FinBERT sentence confusion-review labeling JSONL payload row construction
  for human review and LLM pass chunks.
- FinBERT sentence confusion-review round-robin chunk index planning for
  review and LLM-pass JSONL shards.
- FinBERT sentence confusion-review CSV-safe reviewed-row export preparation
  for nested list/dict values.
- FinBERT sentence confusion-review reviewed-case row construction, including
  review-label joins, final gold-label selection, and confusion-cell
  assignment.
- FinBERT sentence confusion-review examples-by-cell markdown rendering for
  per-cell reviewed sentence examples.
- FinBERT high-confidence sentence-example sample-candidate row finalization,
  including item/sentiment ordering, candidate sorting, and filing-date export
  normalization.
- FinBERT high-confidence sentence-example item/sentiment and
  year/item/sentiment count-row finalization from accumulator dictionaries.
- FinBERT high-confidence sentence-example markdown report rendering for
  ordered item/sentiment sample sections, now with a column-oriented handoff
  before the row-dict compatibility bridge.
- Manifest-contract stable string fingerprint normalization, including
  iterable handling, null skipping, de-duplication, sorting, and SHA-256
  hashing.
- Manifest-contract semantic reuse-guard mismatch detection for version,
  payload, and fingerprint drift checks.
- FinBERT token-bucket batch assignment used after token-length annotation.
- Refinitiv bridge resolution summary scalar counts for accepted, extended,
  effective, unresolved, blocked, ticker-candidate, and identity-conflict
  rows.
- Refinitiv bridge resolution-diagnostic class-summary reducer for grouped
  target diagnostics and extension support/block counts.
- Refinitiv bridge batched strict-`True` field counts for resolution
  diagnostic target, candidate-match, and identity-match summaries.
- Refinitiv bridge ownership-validation case-summary row construction for
  candidate data flags, adjacent support flags, pair counts, and bucket labels.
- Refinitiv bridge ownership-universe handoff row construction for
  universe-effective, ISIN/CUSIP conflict candidate, ticker fallback, and
  nonretrievable cases.
- Refinitiv bridge batched failed-lookup classification for null-RIC rescue
  candidate construction.
- Refinitiv bridge ownership-smoke sample-row projection for normalized
  lookup fields, request dates, truthy flags, and count fields.
- Refinitiv bridge batched alternative-identifier selection for null-RIC
  rescue candidate construction.
- Item-text cleaning layout classifiers for page markers, running report
  headers, structural residue lines, table-like numeric rows, and table
  header/context rows.
- Item-text cleaning full-text newline normalization before cleaned-scope row
  construction.
- Item-text cleaning blank-run collapse after newline normalization and table
  filtering.
- Item-text cleaning early TOC-like line detection used by clustered-prefix
  trimming.
- Item-text cleaning early TOC-prefix trimming for clustered table-of-contents
  leakage before item body text.
- Item-text cleaning item-aware tail-bleed marker scanning used before final
  item text collapse.
- Item-text cleaning reference-only stub detection.
- Item-text cleaning drop-reason and manual-audit reason construction for
  cleaned-scope row audit payloads.
- Item-text cleaning benchmark item-code to text-scope mapping, activation
  status, and audit-period labeling.
- Item-text cleaning batch audit/cleaned-row finalization for cleaned scope
  artifacts, including drop/review/production metadata and audit snippets.
- Item-text cleaning layout-only line removal for page markers, running report
  headers, and structural residue lines.
- Item-text cleaning table-like block removal, including table headers,
  support lines, single-line-with-header handling, and header-context drops.
- Item-text cleaning fused single-text cleaning orchestration for FinBERT
  cleaned scope preprocessing, covering the same staged cleaning semantics as
  the Python implementation while avoiding multiple Python/Rust crossings per
  section row.
- LM2011 paper-style full-10-K exhibit-tail detection used before full-text
  dictionary scoring.
- FinBERT sentence chunk-end selection for oversized section chunking before
  sentencizer execution.
- FinBERT sentence postprocessing classifiers for reference-stitching and
  artifact-line cleanup.
- FinBERT sentence postprocessing shared sentence-key whitespace normalization.
- FinBERT sentence postprocessing list transformation for artifact cleanup and
  reference-stitch policy application.
- FinBERT token-bucket assignment for per-row token length annotation.
- FinBERT visible-prefix retained-end calculation from fast-tokenizer offset
  mappings.
- FinBERT model-label normalization used when resolving sentiment label maps.
- FinBERT benchmark label-mapping resolution from model `id2label`/`label2id`
  config dictionaries, including normalized label validation and duplicate
  detection.
- FinBERT benchmark fallback softmax row and batch probability calculation for
  logits when a native Torch softmax is unavailable.
- FinBERT benchmark per-sentence probability-column and predicted-label
  construction after model logits are converted to probabilities.
- FinBERT benchmark median timing summary helper used by benchmark and staged
  inference summaries.
- FinBERT benchmark stage-summary aggregation for rows, median-seconds totals,
  throughput, and peak VRAM reporting in run summaries.
- FinBERT benchmark bucket max-length selection used by benchmark and staged
  inference batch loops.
- FinBERT benchmark CUDA device-index parsing used by runtime environment
  summaries.
- FinBERT benchmark AMP dtype resolution used by runtime/autocast setup.
- FinBERT staged-inference tokenizer bucket-summary aggregation for
  per-year/per-bucket row counts and token-count mean/median statistics.
- FinBERT item-analysis coverage-report reduction for per-backbone-document
  item coverage flags and summary counts.
- FinBERT sentence-preprocessing token-bucket count reduction used by yearly
  preprocessing summary rows.
- FinBERT sentence-preprocessing manifest count aggregation for processed,
  reused, sentence, oversize, chunking, and cleaning row totals, including a
  column-oriented Rust handoff before the row-dict compatibility bridge.
- FinBERT sentence-preprocessing split-audit metric reduction for chunked
  section counts, warning-boundary counts, and max original character length.
- FinBERT sentence-preprocessing fallback split-boundary warning payload
  reduction for affected section counts and sorted split-reason counts.
- FinBERT sentence-example sample-text normalization for ASCII candidate rows,
  with Python `casefold()` fallback for non-ASCII text.
- FinBERT sentence-length report item-code ordering and JSON-ready record
  serialization for summary payloads, including a column-oriented frame-record
  Rust handoff before the row-dict compatibility bridge.
- FinBERT shared contract year-filter normalization for run configuration
  dataclasses.
- FinBERT bucket-length tuning item-code/year filter normalization and
  sentence-example item-code filter normalization.
- FinBERT bucket-length tuning short/medium/long scalar selection and
  round-up-to-multiple recommendation helpers.
- FinBERT sentence-example deterministic BLAKE2b sample-key calculation for
  candidate sampling order.
- FinBERT sentence-example batched BLAKE2b sample-key calculation and ASCII
  sample-text normalization used before per-group candidate reduction.
- FinBERT sentence-example per-group candidate replacement and de-duplication
  during deterministic sample accumulation.
- FinBERT sentence-example batch accumulator updates for count dictionaries,
  document-id sets, and deterministic sample selections.
- FinBERT sentence-example item-code ordering for candidate count/sample
  report metadata.
- FinBERT sentence-confusion review binary-label normalization for human/LLM
  gold-label aggregation.
- FinBERT sentence-confusion review deterministic sample seed derivation from
  SHA-256 prefixes.
- FinBERT sentence-confusion review candidate-threshold and confusion-cell
  helpers used by exact sampling and reviewed-label metrics.
- FinBERT sentence-confusion review metric-payload calculation for
  reviewed-label accuracy, precision, recall, specificity, and error rates.
- FinBERT sentence-confusion review confusion-cell weighted/unweighted count
  reduction and uncertain-row metric-bound calculation used in review
  summaries.
- FinBERT sentence-confusion review balanced and proportional sample-count
  allocation for deterministic stratum review sampling.
- FinBERT sentence-confusion review streaming sample target-position
  generation, including a column-oriented Rust handoff before the row-dict
  compatibility bridge.
- FinBERT sentence-confusion review majority-bucket metric row aggregation used
  by review summary CSVs, including a column-oriented Rust handoff before the
  row-dict compatibility bridge.
- FinBERT benchmark dataset SHA-256 selection-key and selected-text hashing.
- FinBERT benchmark constrained Hamilton apportionment used by year and
  year-item sample allocation.
- FinBERT benchmark dataset year-level allocation row construction, including
  capacity caps and selected/eligible share columns.
- FinBERT benchmark dataset year-item allocation orchestration, capacity-map
  reducers, and positive allocation-target extraction used by sample
  planning/selection.
- FinBERT benchmark dataset audit-share row construction for selected versus
  eligible universe summaries after Polars grouping and sorting, now with a
  column-oriented handoff before the row-dict compatibility bridge.
- FinBERT benchmark dataset ranked section selection by per-stratum quotas
  after deterministic selection-key sorting.
- Sample item-cleaning sentence diagnostics doc-id sampler, preserving
  item-scope coverage seeding and deterministic fallback fill ordering.
- Manifest canonical JSON SHA-256 and stable string-set fingerprint hashing.
- Manifest semantic file SHA-256 fingerprinting with a Rust streaming hasher
  that reads files in bounded chunks.
- FinBERT tail-feature text-scope alias normalization used before tail
  aggregation.
- FinBERT tail document-surface eager row reduction for materialized
  sentence-score frames, preserving the Polars LazyFrame builder as the core
  large-data path and retaining Python fallback behavior, now with a
  column-oriented Rust handoff before the row-dict compatibility bridge.
- Multi-surface audit boundary-snippet risk and snippet-delta risk batch
  construction used by item-boundary case scoring, replacing scalar struct
  `map_elements` callbacks while preserving Python fallback behavior.
- Multi-surface audit control-case peer-group membership now uses native
  Polars struct membership instead of a scalar `map_elements` callback.
- Multi-surface audit deterministic stable sort-key batch construction used
  by case sampling, replacing the scalar `map_elements` callback while
  preserving nullable sort-key behavior and Python fallback.
- Copied runner bootstrap paths now prefer `src_rust_migration` when scripts are
  executed from the migration copy, so they import the Rust-accelerated package
  tree instead of the original `src` tree.
- SEC/CCM pre-merge markdown scalar value formatting and table rendering for
  bounded run reports, preserving float/date formatting, pipe escaping, and
  truncation notes.
- Refinitiv ownership LSEG request-log event counting for fetch manifests and
  daily-limit diagnostics.
- Refinitiv LSEG ledger item-ID JSON decoding for completed-batch state
  updates.
- Refinitiv LSEG ledger string-array JSON decoding for item and batch field
  lists.
- Refinitiv LSEG stage-audit item-ID JSON decoding for legacy backfill
  tolerance checks.

- Python fallback and metrics:
  `thesis_pkg/core/sec/lm2011_text.py`
  `thesis_pkg/core/sec/lm2011_dictionary.py`
  `thesis_pkg/core/sec/filing_text.py`
  `thesis_pkg/core/sec/html_audit.py`
  `thesis_pkg/core/sec/regime.py`
  `thesis_pkg/core/ccm/lm2011.py`
  `thesis_pkg/core/ccm/transforms.py`
  `thesis_pkg/benchmarking/contracts.py`
  `thesis_pkg/benchmarking/sentence_split_quality_assessment.py`
  `thesis_pkg/benchmarking/item_text_cleaning.py`
  `thesis_pkg/core/sec/lm2011_cleaning.py`
  `thesis_pkg/benchmarking/sentences.py`
  `thesis_pkg/benchmarking/token_lengths.py`
  `thesis_pkg/benchmarking/finbert_visible_prefix.py`
  `thesis_pkg/benchmarking/finbert_benchmark.py`
  `thesis_pkg/benchmarking/finbert_bucket_length_tuning.py`
  `thesis_pkg/benchmarking/finbert_dataset.py`
  `thesis_pkg/benchmarking/finbert_sentence_confusion_review.py`
  `thesis_pkg/benchmarking/finbert_sentence_examples.py`
  `thesis_pkg/benchmarking/finbert_sentence_preprocessing.py`
  `thesis_pkg/benchmarking/finbert_staged_inference.py`
  `thesis_pkg/benchmarking/finbert_tail_features.py`
  `thesis_pkg/benchmarking/manifest_contracts.py`
  `thesis_pkg/benchmarking/multisurface_audit.py`
  `thesis_pkg/pipelines/lm2011_extension.py`
  `thesis_pkg/pipelines/lm2011_pipeline.py`
  `thesis_pkg/pipelines/lm2011_regressions.py`
  `thesis_pkg/pipelines/lm2011_validation_audit.py`
  `thesis_pkg/pipelines/refinitiv/analyst.py`
  `thesis_pkg/pipelines/refinitiv/authority.py`
  `thesis_pkg/pipelines/refinitiv/doc_ownership.py`
  `thesis_pkg/pipelines/refinitiv/lseg_api_common.py`
  `thesis_pkg/pipelines/refinitiv/lseg_api_execution.py`
  `thesis_pkg/pipelines/refinitiv/lseg_analyst_api.py`
  `thesis_pkg/pipelines/refinitiv/lseg_lookup_api.py`
  `thesis_pkg/pipelines/refinitiv/lseg_ownership_api.py`
  `thesis_pkg/pipelines/refinitiv/lseg_recovery.py`
  `thesis_pkg/pipelines/refinitiv/lseg_stage_audit.py`
  `thesis_pkg/pipelines/refinitiv_bridge_pipeline.py`
  `thesis_pkg/pipelines/sec_ccm_pipeline.py`
- Private PyO3 extension module:
  `thesis_native._lm2011_rust`
- Legacy compatibility shim:
  `thesis_pkg.core.sec._lm2011_rust` is aliased to the same native module.
- Rust source:
  `rust/lm2011_rust/src/lib.rs` now owns only implementation module
  declarations/re-exports and the stable PyO3 module entrypoint. The
  `#[pymodule]` body delegates to `rust/lm2011_rust/src/py_exports/`, where
  focused registration modules keep the Python-visible export surface grouped
  by domain: LM2011 core, LM2011 analysis, SEC, Refinitiv/LSEG, text
  processing/item cleaning, FinBERT, and common I/O/audit helpers. Domain
  implementation code remains split across sibling Rust modules under
  `rust/lm2011_rust/src/`.

The Python code imports `_lm2011_rust` opportunistically through the neutral
`thesis_native` boundary. If the Rust extension is unavailable or fails at
runtime, scoring and form normalization fall back to the existing Python
implementations. Runtime counters are exposed through
`get_lm2011_rust_accel_metrics()` and
`get_lm2011_form_rust_accel_metrics()`. CCM transform form-token counters are
exposed through `get_ccm_transforms_rust_accel_metrics()`.
LM2011 dictionary materialization counters are exposed through
`get_lm2011_dictionary_rust_accel_metrics()`.
Sentence-quality counters are exposed through
`get_sentence_quality_rust_accel_metrics()`. Document-ownership
normalizer counters are exposed through
`get_doc_ownership_rust_accel_metrics()`.
Item-cleaning layout counters are exposed through
`get_item_cleaning_rust_accel_metrics()`.
LM2011 full-text cleaning counters are exposed through
`get_lm2011_cleaning_rust_accel_metrics()`.
Sentence postprocessing counters are exposed through
`get_sentence_postprocess_rust_accel_metrics()`.
FinBERT shared contract counters are exposed through
`get_contracts_rust_accel_metrics()`.
FinBERT token-bucket counters are exposed through
`get_finbert_token_rust_accel_metrics()`.
FinBERT visible-prefix counters are exposed through
`get_finbert_visible_prefix_rust_accel_metrics()`.
SEC regime form-normalizer counters are exposed through
`get_regime_rust_accel_metrics()`.
FinBERT bucket-length tuning counters are exposed through
`get_bucket_length_tuning_rust_accel_metrics()`.
FinBERT sentence-example counters are exposed through
`get_sentence_examples_rust_accel_metrics()`.
FinBERT sentence-confusion review counters are exposed through
`get_confusion_review_rust_accel_metrics()`.
FinBERT benchmark inference-helper counters are exposed through
`get_finbert_benchmark_rust_accel_metrics()`.
FinBERT staged-inference tokenizer-summary counters are exposed through
`get_finbert_staged_rust_accel_metrics()`.
FinBERT sentence-preprocessing bucket-count counters are exposed through
`get_finbert_preprocessing_rust_accel_metrics()`.
FinBERT benchmark dataset hashing and apportionment counters are exposed through
`get_finbert_dataset_rust_accel_metrics()`.
LM2011 extension scope-normalizer and p-value counters are exposed through
`get_lm2011_extension_rust_accel_metrics()`.
Manifest fingerprint counters are exposed through
`get_manifest_contracts_rust_accel_metrics()`.
FinBERT tail-feature scope-normalizer and eager document-surface reducer
counters are exposed through `get_finbert_tail_rust_accel_metrics()`.
Multi-surface audit counters are exposed through
`get_multisurface_audit_rust_accel_metrics()`.
LM2011 validation-audit legacy-tokenizer and packet-scalar helper counters are exposed through
`get_lm2011_validation_audit_rust_accel_metrics()`.
Refinitiv LSEG stage-audit counters are exposed through
`get_lseg_stage_audit_rust_accel_metrics()`.
Refinitiv LSEG batching counters are exposed through
`get_lseg_batching_rust_accel_metrics()`.
Refinitiv LSEG API execution counters are exposed through
`get_lseg_api_execution_rust_accel_metrics()`.
Refinitiv LSEG lookup API counters are exposed through
`get_lseg_lookup_api_rust_accel_metrics()`.
Refinitiv LSEG ledger counters are exposed through
`get_lseg_ledger_rust_accel_metrics()`.
Refinitiv ownership authority counters are exposed through
`get_refinitiv_authority_rust_accel_metrics()`.
Refinitiv analyst normalization counters are exposed through
`get_refinitiv_analyst_rust_accel_metrics()`.
Refinitiv LSEG analyst API counters are exposed through
`get_lseg_analyst_api_rust_accel_metrics()`.
Refinitiv LSEG ownership API counters are exposed through
`get_lseg_ownership_rust_accel_metrics()`.
Refinitiv LSEG recovery counters are exposed through
`get_lseg_recovery_rust_accel_metrics()`.
Refinitiv bridge pipeline counters are exposed through
`get_refinitiv_bridge_rust_accel_metrics()`.
SEC HTML-audit helper counters are exposed through
`get_html_audit_rust_accel_metrics()`.
SEC filing-text filename parser counters are exposed through
`get_filing_text_rust_accel_metrics()`.
SEC extraction preprocessing and scanner counters are exposed through
`get_extraction_fastpath_metrics()`.
SEC/CCM pre-merge report counters are exposed through
`get_sec_ccm_pipeline_rust_accel_metrics()`.

## Local Build

This workspace has been checked with `cargo check` and built in place on the
local Windows/Python 3.10 environment. On a machine with Rust and MSVC Build
Tools installed:

```powershell
cd src_rust_migration
$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
python -m pip install setuptools-rust
python setup.py build_ext --inplace
```

The local `setup.py` sets `debug=False`, so `build_ext` uses Cargo's optimized
release profile for the Rust extension.

Then run the lightweight parity tests from the repository root:

```powershell
$env:PYTHONPATH = "src_rust_migration"
pytest tests/test_lm2011_rust_accel.py tests/test_lm2011_pipeline.py::test_tokenize_lm2011_text_matches_appendix_contract tests/test_lm2011_pipeline.py::test_lm2011_text_features_use_total_token_denominators_and_match_hyphenated_dictionary_entries
```

The current broader small-regression command also includes focused regime,
item-cleaning, FinBERT tokenization, Refinitiv ownership, and FinBERT label-map
checks, plus LM2011 dictionary materialization fixtures,
Refinitiv batching/analyst/provider, multi-surface audit, LSEG
ownership smoke tests, FinBERT bucket-tuning and sentence-example synthetic
tests, a Phase 0 packet-A audit smoke test, the Refinitiv analyst LSEG API
synthetic suite, and two FinBERT sentence chunking caller tests, and last
passed with `327 passed` before the Phase 0 packet-scalar helper, FinBERT
apportionment, multi-surface match-bound, sentence-postprocessing,
layout-only line-removal, table-block line-removal, SEC filename parser,
Refinitiv lookup batch-response, Refinitiv analyst group-list, and Refinitiv
lookup item-builder slices.
The focused Rust acceleration suite last passed with `299 passed` after adding
the Refinitiv LSEG lookup request-item row builder fast path. The targeted
sentence-postprocessing policy fixtures passed with `12 passed`; the targeted
packet-A audit smoke test, FinBERT allocation capacity fixture, and
multi-surface audit-pack smoke test each passed with `1 passed`.
After adding Refinitiv analyst LSEG actuals/estimates request-item builders,
the focused Rust acceleration suite passed with `302 passed`, and the targeted
actuals and estimates API batching fixtures each passed with `1 passed`.
After adding Refinitiv ownership LSEG universe/document request-item builders,
the focused Rust acceleration suite passed with `305 passed`, and the targeted
ownership universe batching plus document ownership exact/fallback API fixtures
passed with `2 passed`.
After adding Refinitiv ownership LSEG universe/document batch response
normalization, the focused Rust acceleration suite passed with `308 passed`,
and the targeted ownership response/API fixtures passed with `3 passed`.
After adding Refinitiv analyst LSEG actuals/estimates batch response
normalization, the focused Rust acceleration suite passed with `311 passed`,
and the targeted analyst API fixtures passed with `3 passed`.
After adding Refinitiv LSEG simple `batch_items` grouping/chunking, the
focused acceleration plus standalone batching suites passed with `322 passed`,
and a targeted lookup API smoke fixture passed with `1 passed`.
After adding Refinitiv LSEG provider response-frame fingerprint hashing, the
focused Rust acceleration suite passed with `317 passed`.
After adding Refinitiv ownership authority allowlist key normalization, the
focused Rust acceleration suite passed with `320 passed`, and the targeted
authority allowlist pipeline fixture passed with `1 passed`.
After adding Refinitiv document-ownership retrieval-sheet matching, the
focused Rust acceleration suite passed with `323 passed`, and the targeted
document-ownership finalize fixture passed with `1 passed`.
After adding Refinitiv document-ownership exact-hit selection, the focused
Rust acceleration suite passed with `326 passed`, and the targeted
document-ownership finalize fixture passed with `1 passed`.
After adding Refinitiv document-ownership fallback-hit selection, the focused
Rust acceleration suite passed with `329 passed`, and the targeted
document-ownership finalize fixture passed with `1 passed`.
After adding Refinitiv document-ownership fallback-request row selection, the
focused Rust acceleration suite passed with `332 passed`, and the targeted
document-ownership finalize fixture passed with `1 passed`.
After adding and expanding sentence split quality batch flag classification,
the focused Rust acceleration suite passed with `334 passed`, and the targeted
sentence split quality analysis fixture passed with `1 passed`.
After adding Refinitiv LSEG API execution row-count aggregation, the focused
Rust acceleration suite passed with `337 passed`, and a targeted LSEG API
batch hardening fixture passed with `1 passed`.
After adding Refinitiv LSEG API execution mixed zero/positive requeue
decision, the focused Rust acceleration suite passed with `340 passed`, and
the targeted LSEG API batch hardening fixture passed with `1 passed`.
After adding Refinitiv analyst estimate date-index freezing plus actual-event
and estimate-snapshot canonicalization, the focused Rust acceleration suite
passed with `349 passed`, and the targeted analyst normalized-output fixtures
passed with `6 passed`.
After adding Refinitiv analyst batched request-group ID hashing, the focused
Rust acceleration suite passed with `352 passed`, and the targeted analyst
request-group builder fixtures passed with `4 passed`.
After switching Refinitiv LSEG API execution row-count aggregation from
one-column row dictionaries to plain `item_id` value lists, the focused
row-count tests passed with `3 passed`, a targeted LSEG API batch hardening
fixture passed with `1 passed`, and the focused Rust acceleration suite still
passed with `352 passed`.
After adding FinBERT sentence-confusion balanced and proportional sample-count
allocation, the focused allocation tests passed with `6 passed`, the standalone
FinBERT sentence-confusion review suite passed with `4 passed`, and the
focused Rust acceleration suite passed with `358 passed`.
After adding FinBERT sentence-example batched sample-key and normalized-text
helpers, the focused batch helper tests passed with `6 passed`, the standalone
FinBERT sentence-example suite passed with `2 passed`, and the focused Rust
acceleration suite passed with `364 passed`.
After adding FinBERT benchmark stage-summary aggregation, the focused
stage-summary tests passed with `3 passed`, the copied FinBERT benchmark runner
and sweep smoke tests passed with `4 passed`, and the focused Rust acceleration
suite passed with `367 passed`.
After adding FinBERT staged-inference tokenizer bucket-summary aggregation, the
focused staged bucket-summary tests passed with `3 passed`, the standalone
staged-inference fixture suite passed with `5 passed`, and the focused Rust
acceleration suite passed with `370 passed`.
After adding FinBERT item-analysis coverage-report reduction, the focused
coverage-report tests passed with `3 passed, 679 deselected`, the standalone
FinBERT item-analysis suite passed with `14 passed`, and the focused Rust
acceleration suite passed with `682 passed`.
After adding FinBERT sentence-preprocessing token-bucket count reduction, the
focused preprocessing bucket-count tests passed with `3 passed`, the targeted
sentence-preprocessing fixture subset passed with `8 passed`, and the focused
Rust acceleration suite passed with `373 passed`.
After adding FinBERT sentence-preprocessing manifest count aggregation, the
focused preprocessing manifest-count tests passed with `3 passed, 645
deselected`, the focused Rust acceleration suite passed with `648 passed`, and
the targeted sentence-preprocessing by-year artifact fixture passed with
`1 passed, 22 deselected`.
After adding FinBERT sentence-preprocessing split-audit metric reduction, the
focused split-metric tests passed with `3 passed, 676 deselected`, the
standalone synthetic FinBERT sentence dataset suite passed with `23 passed`,
and the focused Rust acceleration suite passed with `679 passed`.
After adding FinBERT sentence-preprocessing fallback split-boundary warning
payload reduction, the focused fallback split-warning tests passed with
`3 passed, 682 deselected`, the standalone synthetic FinBERT sentence dataset
suite passed with `23 passed`, and the focused Rust acceleration suite passed
with `685 passed`.
After adding FinBERT benchmark dataset capacity/target reducers, the focused
capacity-and-target reducer tests passed with `3 passed`, the standalone
synthetic FinBERT benchmark dataset suite passed with `8 passed`, and the
focused Rust acceleration suite passed with `376 passed`.
After adding FinBERT benchmark dataset year-item allocation orchestration, the
focused year-item allocation tests passed with `3 passed`, the standalone
synthetic FinBERT benchmark dataset suite passed with `8 passed`, and the
focused Rust acceleration suite passed with `379 passed`.
After adding FinBERT sentence-confusion review count and uncertain-bound
reducers, the focused reducer tests passed with `3 passed`, the standalone
FinBERT sentence-confusion review suite passed with `4 passed`, and the
focused Rust acceleration suite passed with `382 passed`.
After adding FinBERT sentence-confusion review bucket-metric row aggregation,
the focused bucket-metric tests passed with `3 passed`, the standalone FinBERT
sentence-confusion review suite passed with `4 passed`, and the focused Rust
acceleration suite passed with `385 passed`.
After adding LM2011 Phase 0 validation-audit Packet A token-delta summary
aggregation, the focused Packet A delta tests passed with `3 passed`, the
targeted Packet A audit fixture passed with `1 passed`, and the focused Rust
acceleration suite passed with `388 passed`.
After adding LM2011 Phase 0 validation-audit Packet A summary-frame aggregation,
the focused Packet A summary-frame tests passed with `3 passed`, the targeted
Packet A audit fixture passed with `1 passed`, and the focused Rust
acceleration suite passed with `391 passed`.
After adding LM2011 Phase 0 validation-audit Packet A strip-comparison
aggregation, the focused Packet A strip-comparison tests passed with `3 passed`,
the targeted Packet A audit fixture passed with `1 passed`, and the focused
Rust acceleration suite passed with `394 passed`.
After adding LM2011 Phase 0 validation-audit Packet A example-row selection and
ordering, the focused Packet A examples-frame tests passed with `3 passed`, the
targeted Packet A audit fixture passed with `1 passed`, and the focused Rust
acceleration suite passed with `397 passed`.
After adding LM2011 extension result-row conversion, the focused extension row
converter tests passed with `3 passed`, the targeted synthetic extension
estimation scaffold passed with `2 passed`, and the focused Rust acceleration
suite passed with `400 passed`.
After adding LM2011 extension quarterly fit-row construction, the focused
quarterly-fit tests passed with `3 passed`, the targeted synthetic
fit-comparison scaffold passed with `3 passed`, and the focused Rust
acceleration suite passed with `403 passed`.
After adding LM2011 extension skipped-quarter diagnostic row construction, the
focused skipped-quarter tests passed with `3 passed`, the targeted synthetic
fit-comparison scaffold passed with `3 passed`, and the focused Rust
acceleration suite passed with `406 passed`.
After adding LM2011 extension quarterly fit-difference row construction, the
focused quarterly-difference tests passed with `3 passed`, the targeted
synthetic fit-comparison scaffold passed with `3 passed`, and the focused Rust
acceleration suite passed with `409 passed`.
After adding LM2011 extension common-quarter fit-summary aggregation, the
focused fit-summary row tests passed with `3 passed`, the targeted synthetic
fit-comparison scaffold passed with `3 passed`, and the focused Rust
acceleration suite passed with `412 passed`.
After adding LM2011 extension common-quarter fit-comparison aggregation, the
focused fit-comparison row tests passed with `3 passed`, the targeted
synthetic fit-comparison scaffold passed with `3 passed`, and the focused Rust
acceleration suite passed with `415 passed`.
After adding LM2011 Table IA.II monthly strategy result-row expansion, the
focused IA.II row tests passed with `3 passed`, the targeted synthetic IA.II
regression fixtures passed with `2 passed`, and the focused Rust acceleration
suite passed with `418 passed`.
After adding LM2011 quarterly Fama-MacBeth final result-row aggregation, the
focused Fama-MacBeth row tests passed with `3 passed`, the targeted synthetic
quarterly regression fixtures passed with `3 passed`, and the focused Rust
acceleration suite passed with `421 passed`.
After adding LM2011 cross-sectional OLS design-row preparation, the focused
design-row tests passed with `3 passed`, the targeted synthetic quarterly
regression fixtures passed with `3 passed`, and the focused Rust acceleration
suite passed with `424 passed`.
After adding LM2011 text feature-row construction, the focused feature-row
tests passed with `3 passed`, the targeted text-feature contract tests passed
with `3 passed`, and the focused Rust acceleration suite passed with
`427 passed`.
After adding LM2011 fixed-shape FF3 OLS alpha/RMSE calculation, the focused
OLS tests passed with `3 passed`, and the targeted deterministic/rank-deficient
FF3 fixture tests passed with `2 passed`; the focused Rust acceleration suite
passed with `430 passed`.
After adding LM2011 dictionary-token normalization, the focused dictionary-token
tests plus the text-feature builder fixture passed with `4 passed`, and the
targeted text-feature contract tests passed with `3 passed`; the focused Rust
acceleration suite passed with `433 passed`.
After adding LM2011 event-window regression metric row construction, the
focused regression-window tests passed with `3 passed`, and the targeted
small event-window fixtures passed with `3 passed`; the focused Rust
acceleration suite passed with `436 passed`.
After adding LM2011 fixed-shape FF4 OLS coefficient calculation, the focused
FF4 coefficient tests passed with `3 passed`, and the deterministic FF4
pipeline fixture passed with `1 passed`; the focused Rust acceleration suite
passed with `439 passed`.
After adding LM2011 monthly strategy factor-loading row construction, the
focused strategy factor-loading tests passed with `3 passed`, and the targeted
trading-strategy FF4 fixture passed with `1 passed`; the focused Rust
acceleration suite passed with `442 passed`.
After adding LM2011 document-stat preparation for uncleaned text batches, the
focused document-stat tests plus text-feature builder fixture passed with
`4 passed`, and the targeted text-feature contract tests passed with
`3 passed`; the focused Rust acceleration suite passed with `445 passed`.
After adding LM2011 streaming pass-1 row construction for uncleaned text
batches, the focused pass-1 tests plus text-feature builder fixture passed
with `4 passed`, and the targeted streaming writer fixtures passed with
`2 passed`; the focused Rust acceleration suite passed with `448 passed`.
After adding LM2011 streaming pass-2 feature-row construction from pass-1
shards, the focused pass-1 feature-row tests passed with `3 passed`, the
text-feature builder fixture passed with `1 passed`, and the targeted
streaming writer fixtures passed with `2 passed`; the focused Rust
acceleration suite passed with `451 passed`.
After extending LM2011 document-stat and streaming pass-1 preparation to use
batch Rust counting after Python text cleaning, the focused cleaned-text tests
passed with `2 passed`, the document/pass-1 focused tests passed with
`8 passed`, and the targeted streaming writer fixtures passed with `2 passed`;
the focused Rust acceleration suite passed with `453 passed`.
After adding FinBERT visible-prefix LM2011 token-count batching, the focused
visible-prefix Rust tests passed with `6 passed`, the targeted visible-prefix
tokenization fixtures passed with `2 passed`, and the focused Rust
acceleration suite passed with `456 passed`.
After adding FinBERT bucket-length tuning summary row assembly, the focused
bucket-length summary tests passed with `3 passed`, the broader bucket-length
helper selection passed with `10 passed`, and the focused Rust acceleration
suite passed with `459 passed`.
After adding sentence dataset chunk-row expansion, the focused chunk tests
passed with `6 passed`, the targeted synthetic sentence-dataset chunk fixture
passed with `1 passed`, and the focused Rust acceleration suite passed with
`462 passed`.
After adding item-text cleaning LM2011 token-count batching for cleaned scope
audit inputs, the focused item-cleaning token-count tests passed with
`4 passed`, the item-cleaning focused selection passed with `26 passed`, and
the focused Rust acceleration suite passed with `466 passed`.
After adding multi-surface audit review-record chunk planning, the focused
chunk tests passed with `3 passed`, and the focused Rust acceleration suite
passed with `469 passed`.
After adding FinBERT sentence confusion-review sample ID row construction, the
focused sample-ID tests passed with `3 passed`, and the focused Rust
acceleration suite passed with `472 passed`.
After adding FinBERT sentence confusion-review neighbor-target row
construction, the focused neighbor-target tests passed with `3 passed`, and
the focused Rust acceleration suite passed with `475 passed`.
After adding FinBERT sentence confusion-review allocation-row finalization, the
focused allocation-row tests passed with `3 passed`, and the focused Rust
acceleration suite passed with `478 passed`.
After adding FinBERT sentence confusion-review labeling-record construction,
the focused labeling-record tests passed with `3 passed`, and the focused Rust
acceleration suite passed with `481 passed`.
After adding FinBERT sentence confusion-review round-robin chunk planning, the
focused chunk-row tests passed with `3 passed`, and the focused Rust
acceleration suite passed with `484 passed`.
After adding FinBERT sentence confusion-review CSV-safe row preparation, the
focused CSV-safe row tests passed with `3 passed`, and the focused Rust
acceleration suite passed with `487 passed`.
After adding FinBERT sentence confusion-review reviewed-case row construction,
the focused reviewed-row tests passed with `3 passed`, and the focused Rust
acceleration suite passed with `490 passed`.
After adding FinBERT sentence confusion-review examples-by-cell markdown
rendering, the focused examples-by-cell tests passed with
`3 passed, 688 deselected`, the standalone FinBERT sentence-confusion review
suite passed with `4 passed`, and the focused Rust acceleration suite passed
with `691 passed`.
After adding FinBERT high-confidence sentence-example sample-candidate row
finalization, the focused sample-candidate row tests passed with `3 passed`,
and the focused Rust acceleration suite passed with `493 passed`.
After adding FinBERT high-confidence sentence-example count-row finalization,
the focused sentence-example count-row tests passed with `3 passed`, and the
focused Rust acceleration suite passed with `496 passed`.
After adding FinBERT high-confidence sentence-example markdown rendering, the
focused markdown-renderer tests passed with `3 passed, 670 deselected`, the
standalone sentence-example fixture suite passed with `2 passed`, and the
focused Rust acceleration suite passed with `673 passed`.
After moving manifest-contract stable string fingerprint normalization into
Rust, the focused manifest-contract tests passed with `5 passed`, and the
focused Rust acceleration suite passed with `496 passed`.
After adding FinBERT token-bucket batch assignment, the focused token-bucket
tests passed with `7 passed`, and the focused Rust acceleration suite passed
with `500 passed`.
After adding Refinitiv bridge resolution summary scalar counts, the focused
resolution-summary count tests passed with `3 passed`, and the focused Rust
acceleration suite passed with `503 passed`.
After adding Refinitiv bridge resolution-diagnostic class-summary reduction,
the focused class-summary tests passed with `3 passed`, and the focused Rust
acceleration suite passed with `506 passed`.
After adding Refinitiv bridge batched strict-`True` field counts, the focused
summary-reducer tests passed with `3 passed`, and the focused Rust acceleration
suite passed with `506 passed`.
After adding Refinitiv bridge ownership-validation case-row construction, the
focused case-row tests passed with `3 passed`, and the focused Rust
acceleration suite passed with `509 passed`.
After adding Refinitiv bridge ownership-universe handoff row construction, the
focused handoff-row tests passed with `3 passed`, and the focused Rust
acceleration suite passed with `512 passed`.
After adding Refinitiv bridge ownership-smoke sample-row projection, the
focused smoke-sample row tests passed with `3 passed`, and the focused Rust
acceleration suite passed with `515 passed`.
After adding Refinitiv bridge batched failed-lookup classification, the
focused failed-lookup record tests passed with `3 passed`, and the focused
Rust acceleration suite passed with `518 passed`.
After adding Refinitiv bridge batched alternative-identifier selection, the
focused alternative-identifier batch tests passed with `3 passed`, and the
focused Rust acceleration suite passed with `521 passed`.
After exposing the Refinitiv LSEG unresolved-identifier message parser
directly through Rust, the focused provider identifier parser tests passed with
`3 passed`, and the focused Rust acceleration suite passed with `521 passed`.
After adding Refinitiv LSEG provider UTF-8 record coercion, the focused UTF-8
record tests passed with `3 passed`, and the focused Rust acceleration suite
passed with `524 passed`.
After adding Refinitiv LSEG provider response-header sanitization, the focused
header sanitization tests passed with `3 passed`, and the focused Rust
acceleration suite passed with `527 passed`.
After adding Refinitiv LSEG session-not-opened message detection, the focused
session-message tests passed with `3 passed`, and the focused Rust acceleration
suite passed with `530 passed`.
After adding Refinitiv LSEG stage-audit optional mapping normalization, the
focused stage-audit normalizer tests passed with `3 passed`, and the focused
Rust acceleration suite passed with `530 passed`.
After adding SEC utility newline normalization, date parsing, Roman numeral
parsing, default item-part inference, and bullet-prefix classification, the
focused SEC utility tests passed with `3 passed`, and the focused Rust
acceleration suite passed with `530 passed`.
After adding SEC filing boundary-authority status classification and public
boundary payload projection, the focused boundary/filename tests passed with
`9 passed`, and the focused Rust acceleration suite passed with `536 passed`.
After adding the sample item-cleaning sentence diagnostics doc-id sampler, the
focused sampler tests passed with `3 passed`, and the existing deterministic
diagnostics sampler test passed with `1 passed`; the focused Rust acceleration
suite passed with `539 passed`.
After adding FinBERT sentence-length report item ordering and JSON-ready record
serialization, the focused sentence-length helper tests passed with `3 passed`,
and the standalone sentence-length visualization fixture passed with
`1 passed`; the focused Rust acceleration suite passed with `542 passed`.
After wiring sentence split-quality report item ordering to the native ordering
helper, the focused quality ordering tests passed with `3 passed`, and the
standalone split-quality fixture passed with `1 passed`; the focused Rust
acceleration suite passed with `545 passed`.
After wiring FinBERT sentence-example item ordering to the same native helper,
the focused ordering tests passed with `3 passed`, and the existing
sentence-example sample/count Rust tests passed with `6 passed`; the focused
Rust acceleration suite passed with `548 passed`.
After adding SEC/CCM pre-merge markdown table rendering, the focused markdown
tests passed with `3 passed`, and the synthetic SEC/CCM end-to-end artifact
fixture passed with `1 passed`; the focused Rust acceleration suite passed with
`551 passed`.
After exposing the SEC/CCM markdown scalar value formatter through Rust and
routing the copied Python wrapper through it, the focused SEC/CCM markdown
tests passed with `6 passed, 688 deselected`, and the synthetic Rust
acceleration suite passed with `694 passed`.
After adding Refinitiv ownership LSEG request-log event counting, the focused
request-log tests passed with `3 passed`, and the focused Rust acceleration
suite passed with `554 passed`.
After adding Refinitiv LSEG ledger item-ID JSON decoding, the focused ledger
tests passed with `4 passed`, and the focused Rust acceleration suite passed
with `558 passed`; the existing request-ledger claim/stale-requeue fixture
passed with `1 passed`.
After wiring Refinitiv LSEG stage-audit legacy-gap item-ID JSON decoding to the
same native helper, the focused stage-audit tests passed with `3 passed`, and
the focused Rust acceleration suite passed with `561 passed`.
After adding Refinitiv LSEG ledger string-array JSON decoding for `fields_json`,
the focused ledger JSON tests passed with `8 passed`, and the focused Rust
acceleration suite passed with `565 passed`.
After adding LM2011 FF48 SIC mapping text parsing, the focused FF48 tests
passed with `3 passed`, and the focused Rust acceleration suite passed with
`568 passed`.
After adding SEC pipeline completed-year checkpoint JSON decoding, the focused
checkpoint tests passed with `3 passed`, and the focused Rust acceleration
suite passed with `571 passed`.
After adding SEC pipeline no-item stats weighted aggregation, the focused
no-item aggregation tests passed with `3 passed, 685 deselected`, the existing
filing-text aggregation fixture passed with `1 passed, 54 deselected`, and the
focused Rust acceleration suite passed with `688 passed`.
After adding Refinitiv bridge ownership-result date normalization, the focused
ownership-result helper tests passed with `3 passed, 568 deselected`, and the
focused Rust acceleration suite passed with `571 passed`.
After adding Refinitiv bridge ownership-universe request-date and date-text
rendering, the focused bridge date-text helper tests passed with `3 passed,
571 deselected`, and the focused Rust acceleration suite passed with
`574 passed`.
After adding item-text cleaning row-audit base payload, review-status, and
production eligibility helpers, the focused row-audit helper tests passed with
`3 passed, 574 deselected`, and the focused Rust acceleration suite passed
with `577 passed`.
After adding FinBERT sentence-example per-group candidate selection, the
focused sample-selection tests passed with `3 passed, 577 deselected`, and the
focused Rust acceleration suite passed with `580 passed`.
After adding Refinitiv LSEG API execution singleton/split child-batch row
planning, the focused child-batch tests passed with `3 passed, 580 deselected`,
and the focused Rust acceleration suite passed with `583 passed`.
After adding Refinitiv LSEG API execution item-result detail row construction,
the focused item-result detail tests passed with `3 passed, 583 deselected`,
and the focused Rust acceleration suite passed with `586 passed`.
After adding Refinitiv bridge accepted-resolution derivation, the focused
accepted-resolution tests passed with `3 passed, 606 deselected`, and the
focused Rust acceleration suite passed with `609 passed`.
After adding Refinitiv bridge lookup/accepted candidate extraction, the focused
candidate-extractor and accepted-resolution tests passed with `6 passed, 606
deselected`, and the focused Rust acceleration suite passed with `612 passed`.
After adding Refinitiv bridge effective-resolution field derivation, the
focused resolver tests passed with `9 passed, 606 deselected`, and the focused
Rust acceleration suite passed with `615 passed`; targeted Refinitiv bridge
resolver fixtures passed with `2 passed, 11 deselected`.
After adding Refinitiv bridge resolution-diagnostic handoff row projection, the
focused handoff tests passed with `3 passed, 615 deselected`, the focused Rust
acceleration suite passed with `618 passed`, and the targeted resolution
diagnostic fixture passed with `1 passed, 12 deselected`.
After adding Refinitiv bridge ownership-validation handoff row planning, the
focused ownership-validation handoff tests passed with `3 passed, 618
deselected`, the focused Rust acceleration suite passed with `621 passed`, and
the targeted bridge handoff/diagnostic fixture passed with `1 passed, 12
deselected`.
After adding Refinitiv document-ownership final-row construction, the focused
doc-ownership final-row tests passed with `3 passed, 621 deselected`, the
focused Rust acceleration suite passed with `624 passed`, and the standalone
Refinitiv document-ownership pipeline fixture suite passed with `13 passed`.
After adding FinBERT benchmark label-mapping resolution, the focused
label-mapping tests passed with `3 passed, 624 deselected`, the focused Rust
acceleration suite passed with `627 passed`, and the targeted FinBERT
item-analysis mapping fixture passed with `1 passed, 13 deselected`.
After adding FinBERT benchmark probability-column and predicted-label
construction, the focused probability-label tests passed with `4 passed, 651
deselected`, the focused Rust acceleration suite passed with `655 passed`, the
targeted FinBERT item-analysis score fixture passed with `1 passed, 13
deselected`, and the targeted staged-inference fake-model fixture passed with
`1 passed, 4 deselected`.
After adding item-text cleaning batch audit/cleaned-row finalization, the
focused finalize-row tests passed with `3 passed, 655 deselected`, the focused
Rust acceleration suite passed with `658 passed`, and the standalone
item-text cleaning fixture suite passed with `15 passed`.
After adding SEC HTML-audit filing status classification, the focused
HTML-audit scalar helper tests passed with `3 passed, 655 deselected`, the
small HTML-audit output fixture passed with `1 passed`, and the focused Rust
acceleration suite passed with `658 passed`.
After adding LM2011 streaming writer incremental `doc_id` duplicate
validation, the focused doc-id validation tests passed with `3 passed, 658
deselected`, the targeted duplicate-writer fixtures passed with `2 passed, 81
deselected`, and the focused Rust acceleration suite passed with `661 passed`.
After adding Refinitiv document-ownership target-effective-date and
request-bound date-window clipping, the focused date-window helper tests passed
with `3 passed, 661 deselected`, the targeted request-bound fixtures passed
with `6 passed, 7 deselected`, and the focused Rust acceleration suite passed
with `664 passed`.
After adding FinBERT benchmark dataset audit-share row construction, the
focused share-row tests passed with `3 passed, 664 deselected`, the standalone
synthetic FinBERT benchmark dataset suite passed with `8 passed`, and the
focused Rust acceleration suite passed with `667 passed`.
After adding FinBERT benchmark dataset year-level allocation row construction,
the focused year-allocation tests passed with `3 passed, 667 deselected`, the
standalone synthetic FinBERT benchmark dataset suite passed with `8 passed`,
and the focused Rust acceleration suite passed with `670 passed`.
After adding FinBERT benchmark dataset ranked section selection, the focused
ranked-selection tests passed with `3 passed, 673 deselected`, the standalone
synthetic FinBERT benchmark dataset suite passed with `8 passed`, and the
focused Rust acceleration suite passed with `676 passed`.
After adding LM2011 Phase 0 validation-audit form-count row finalization, the
focused form-count tests passed with `3 passed, 627 deselected`, the focused
Rust acceleration suite passed with `630 passed`, and the targeted Phase 0
validation-audit subset passed with `2 passed, 7 deselected`.
After adding LM2011 Phase 0 validation-audit representative-term count updates,
the focused term-count tests passed with `3 passed, 630 deselected`, the
focused Rust acceleration suite passed with `633 passed`, and the targeted
Phase 0 validation-audit subset passed with `2 passed, 7 deselected`.
After adding LM2011 Phase 0 validation-audit raw normalized-form count
aggregation, the focused normalized-form tests passed with `3 passed, 633
deselected`, the focused Rust acceleration suite passed with `636 passed`, and
the targeted Phase 0 validation-audit subset passed with `2 passed, 7
deselected`.
After adding LM2011 Phase 0 validation-audit Packet B event-attrition row
aggregation, the focused event-attrition tests passed with `3 passed, 636
deselected`, the focused Rust acceleration suite passed with `639 passed`, and
the targeted Phase 0 validation-audit subset passed with `2 passed, 7
deselected`.
After adding LM2011 Phase 0 validation-audit Packet C unit-audit row
construction, the focused units-row tests passed with `3 passed, 639
deselected`, the focused Rust acceleration suite passed with `642 passed`, and
the targeted Packet C validation-audit fixture passed with `1 passed, 8
deselected`.
After adding LM2011 Phase 0 validation-audit Packet A MD&A row payload
construction, the focused MD&A row tests passed with `3 passed, 642
deselected`, the focused Rust acceleration suite passed with `645 passed`, and
the targeted Packet A validation-audit fixture passed with `1 passed, 8
deselected`.
After adding LM2011 Phase 0 validation-audit Packet D coverage-reconciliation
row construction, the focused Packet D coverage-row tests passed with `3
passed, 648 deselected`, the focused Rust acceleration suite passed with `651
passed`, and the targeted Packet D validation-audit fixture passed with `1
passed, 8 deselected`.
The two targeted widened-interval consumer fixtures last passed with `1 passed`
each.
The small HTML audit fixture suite last passed with `3 passed`.
The standalone Refinitiv LSEG batching fixture suite last passed with
`8 passed`.
The standalone item-cleaning fixture suite last passed with `15 passed`.
The targeted Refinitiv bridge resolution policy fixture last passed with
`1 passed`.
The standalone Refinitiv bridge pipeline fixture suite last passed with
`13 passed`.
The standalone Refinitiv document-ownership fixture suite last passed with
`13 passed`.
The standalone Refinitiv analyst fixture suite last passed with `13 passed`.
After adding Refinitiv bridge ownership-validation pairwise result comparison,
the focused ownership-comparison tests passed with `4 passed, 694 deselected`,
and the focused Rust acceleration suite passed with `698 passed`.
After adding Refinitiv ownership authority pairwise alias diagnostics, the
focused alias-diagnostics tests passed with `4 passed, 698 deselected`, the
focused Rust acceleration suite passed with `702 passed`, and the standalone
Refinitiv authority fixture suite passed with `2 passed`.
After adding Refinitiv ownership authority candidate metrics, the focused
candidate-metrics tests passed with `3 passed, 702 deselected`, the focused
Rust acceleration suite passed with `705 passed`, and the standalone Refinitiv
authority fixture suite passed with `2 passed`.
After adding Refinitiv analyst normalized-event row construction, the focused
normalized-event tests passed with `3 passed, 705 deselected`, the focused
Rust acceleration suite passed with `708 passed`, and the standalone Refinitiv
analyst fixture suite passed with `13 passed`.
After adding Refinitiv bridge material identity-conflict checks, the focused
material-conflict tests passed with `3 passed, 708 deselected`, the focused
bridge identity/accepted-resolution selection passed with
`9 passed, 702 deselected`, the standalone Refinitiv bridge fixture suite
passed with `13 passed`, and the focused Rust acceleration suite passed with
`711 passed`.
After adding the Item 7 LM floor-sweep threshold summary-row reducer, the
focused floor-sweep Rust tests passed with `3 passed, 711 deselected`, the
standalone floor-sweep fixture passed with `1 passed`, and the focused Rust
acceleration suite passed with `714 passed`.
After adding Refinitiv Excel ownership-universe block payload construction,
the focused Excel payload/workbook tests passed with
`4 passed, 714 deselected`, and the focused Rust acceleration suite passed
with `718 passed`.
After adding Refinitiv Excel LM2011 document-ownership block payload
construction, the focused doc-ownership Excel tests passed with
`4 passed, 718 deselected`, the combined Excel payload/workbook tests passed
with `8 passed, 714 deselected`, the standalone doc-ownership workbook formula
fixture passed with `1 passed, 12 deselected`, and the focused Rust
acceleration suite passed with `722 passed`.
After adding Refinitiv Excel resolution-diagnostic block payload construction,
the focused resolution-diagnostic Excel tests passed with
`4 passed, 722 deselected`, the combined Excel payload/workbook tests passed
with `12 passed, 714 deselected`, and the focused Rust acceleration suite
passed with `726 passed`.
After adding Refinitiv Excel ownership-smoke block payload construction, the
focused smoke Excel tests passed with `4 passed, 726 deselected`, the combined
Excel payload/workbook tests passed with `16 passed, 714 deselected`, and the
focused Rust acceleration suite passed with `730 passed`.
After adding Refinitiv Excel ownership-validation sheet payload construction,
the focused ownership-validation Excel tests passed with
`4 passed, 730 deselected`, the combined Excel payload/workbook tests passed
with `20 passed, 714 deselected`, and the focused Rust acceleration suite
passed with `734 passed`.
After adding Refinitiv Excel extended summary formula payload construction, the
focused extended-summary Excel tests passed with `4 passed, 734 deselected`,
the combined Excel payload/workbook tests passed with `24 passed, 714
deselected`, and the focused Rust acceleration suite passed with `738 passed`.
After adding Refinitiv Excel extended lookup formula payload construction, the
focused extended-lookup Excel tests passed with `4 passed, 738 deselected`,
the combined Excel payload/workbook tests passed with `28 passed, 714
deselected`, and the focused Rust acceleration suite passed with `742 passed`.
After adding LM2011 document-ownership request row construction, the focused
request-row tests passed with `3 passed, 742 deselected`, the focused
doc-ownership Rust tests passed with `40 passed, 705 deselected`, and the
focused Rust acceleration suite passed with `745 passed`.
After adding LM2011 document-ownership universe-diagnostics summary reduction,
the focused universe-summary tests passed with `3 passed, 745 deselected`, the
focused doc-ownership Rust tests passed with `43 passed, 705 deselected`, the
standalone universe-diagnostics pipeline tests passed with `2 passed, 11
deselected`, and the focused Rust acceleration suite passed with `748 passed`.
After adding LM2011 SEC/CCM batch form normalization for LazyFrame expression
paths, the focused LM2011 form tests passed with `6 passed, 744 deselected`,
the small CCM suite passed with `7 passed`, and the focused Rust acceleration
suite passed with `750 passed`.
After adding LM2011 document-ownership membership-prep batch normalization,
the focused batch/membership diagnostics tests passed with `6 passed, 747
deselected`, the focused doc-ownership Rust tests passed with `46 passed, 707
deselected`, the standalone universe-diagnostics pipeline tests passed with
`2 passed, 11 deselected`, and the focused Rust acceleration suite passed with
`753 passed`.

Import-only smoke checks confirmed that these copied runners resolve `SRC` to
`src_rust_migration`: `finbert_item_analysis_runner.py`,
`lm2011_phase0_validation_audit.py`, `lm2011_finbert_robustness_runner.py`,
`lm2011_sample_post_refinitiv_runner.py`, and `sec_ccm_unified_runner.py`.

Avoid broad tests or full runner tests unless explicitly using small samples;
many pipeline tests can materialize larger data surfaces.

The copied FinBERT tail document-surface contract now includes
`top_5_sentences_neg_mean` alongside the existing top-decile and top-quintile
signals. The eager `build_finbert_tail_doc_surface` wrapper now attempts the
Rust `finbert_tail_doc_surface_rows` reducer for materialized sentence-score
frames, while `build_finbert_tail_doc_surface_lf` remains the Polars LazyFrame
implementation for large pipeline surfaces. The targeted Rust tail selector
passed with `6 passed, 750 deselected`; the combined standalone
tail/robustness probe passed with `9 passed`; and the focused Rust acceleration
suite passed with `756 passed`. The standalone
`tests/test_finbert_tail_features.py` suite explicitly forces imports from
`src_rust_migration`.

The FinBERT robustness runner now reuses the Rust-backed LM2011 extension
fit-comparison row helpers instead of duplicating Python row construction
inside the runner. The standalone `tests/test_lm2011_finbert_robustness_runner.py`
suite explicitly forces imports from `src_rust_migration` and passed with
`5 passed`. `cargo fmt --check`, `cargo check`, and the focused Rust
acceleration suite also passed for this slice, with `756 passed` in
`tests/test_lm2011_rust_accel.py`.

Parquet promotion now uses Rust for stream-copy and quick magic-byte probing
when the extension is available, while PyArrow validation remains in Python.
The SEC Parquet filing-text streamer now uses Rust to select matched `doc_id`
row indices within each PyArrow batch before materializing `full_text`. Both
slices preserve Python fallback behavior and passed their focused
`tests/test_lm2011_rust_accel.py` selectors plus the full migration
acceleration suite.

Refinitiv LSEG document-recovery artifacts now use Rust for the unresolved
request mask over request `doc_id`, retrieval eligibility, and returned raw
`doc_id` values. The migration-only tests cover duplicate returned IDs, null
returned IDs, null request IDs, retrieval-ineligible rows, row-order
preservation, real-extension dispatch, and fallback after extension absence or
failure. `cargo fmt --check`, `cargo check`, `python setup.py build_ext
--inplace`, the focused recovery selector (`4 passed, 768 deselected`), and
the full Rust acceleration suite (`772 passed`) passed for this slice.

LM2011 event-window row expansion now uses Rust to find matching document and
daily-market row pairs by `KYPERMNO` and inclusive relative trade-index window.
The Python wrapper keeps the existing Polars schema construction, daily column
aliases, `relative_day` calculation, sort order, and fallback implementation.
The migration-only tests cover multiple PERMNOs, multiple docs per PERMNO,
duplicate daily rows, alternate event windows, null market values, empty
no-match output, real-extension dispatch, and extension-unavailable fallback.
`cargo fmt --check`, `cargo check`, `python setup.py build_ext --inplace`,
the focused window-row selector (`4 passed, 772 deselected`), the migration-
forced LM2011 pipeline event-window gate (`3 passed, 80 deselected`), and the
full Rust acceleration suite (`776 passed`) passed for this slice.

Item-text cleaning now has a fused Rust fast path for `clean_item_text`, the
single-section helper called by FinBERT cleaned-scope preprocessing. The
wrapper preserves the pure Python fallback and returns the existing
`CleanedTextResult` contract; non-ASCII inputs fall back to the Python
implementation. The migration-only tests cover enabled and disabled cleaning,
layout artifact removal, clustered TOC trimming, table-like block removal,
item-aware tail truncation, reference-stub/non-body handling, table-scope
allowlist skip behavior, non-ASCII fallback, real-extension dispatch, and
extension-unavailable fallback. `cargo fmt --check`, `cargo check`,
`python setup.py build_ext --inplace`, the focused clean-text selector (`3
passed, 776 deselected`), the broader item-cleaning Rust selector (`35 passed,
744 deselected`), the migration-forced standalone item-text-cleaning suite
(`15 passed`), and the full Rust acceleration suite (`779 passed`) passed for
this slice.

Item-text cleaning cleaned-scope preparation now uses Rust to assemble
per-section prepared records before the existing LM token-count and
finalization batches. The wrapper preserves Python fallback behavior and falls
back for non-ASCII batches. `cargo fmt --check`, `cargo check`, `python
setup.py build_ext --inplace`, the focused prepare-row selector (`3 passed,
779 deselected`), the broader item-cleaning Rust selector (`38 passed, 744
deselected`), the migration-forced standalone item-text-cleaning suite (`15
passed`), and the full Rust acceleration suite (`782 passed`) passed for this
slice.

SEC HTML-audit sampling now has Rust fast paths for stratified row sampling and
status-weighted filing sampling. The wrappers preserve the existing Python
fallbacks and use the same seeded Python RNG behavior for exact sample parity.
`cargo fmt --check`, `cargo check`, `python setup.py build_ext --inplace`, the
focused sampling selector (`3 passed, 782 deselected`), the broader HTML-audit
selector (`6 passed, 779 deselected`), the full Rust acceleration suite (`785
passed`), and the small HTML-audit output fixture suite (`3 passed`) passed for
this slice.

Item-text cleaning manual-audit sample construction now uses Rust for the
post-sort per-scope/period cap and review-row projection. The wrapper keeps the
existing Polars preparation, schema alignment, and Python fallback. `cargo fmt
--check`, `cargo check`, `python setup.py build_ext --inplace`, the focused
manual-audit sample selector (`3 passed, 785 deselected`), the broader
item-cleaning Rust selector (`41 passed, 747 deselected`), the
migration-forced standalone item-text-cleaning suite (`15 passed`), and the
full Rust acceleration suite (`788 passed`) passed for this slice.

SEC extraction Part-marker scanning now uses the Rust extension before the
existing Cython/Python fallback chain for ASCII filing-line batches. The
wrapper preserves the prior Cython dispatch behavior for monkeypatched/test
fast-path stubs and falls back for non-ASCII batches. `python -m py_compile`,
`cargo fmt --check`, `cargo check`, `python setup.py build_ext --inplace`, the
focused extraction Part-marker selector (`3 passed, 788 deselected`), the
combined SEC utility/extraction selector (`6 passed, 785 deselected`), the
full Rust acceleration suite (`791 passed`), focused extraction dispatch
fixtures (`9 passed`), focused 10-Q/extraction logic fixtures (`10 passed`),
the filing-text suite (`55 passed`), and suspicious-boundary diagnostics on
the local validation parquet sample (`6` filings, `18` items, `0` flagged
items) passed for this slice. The Hypothesis-based extraction fuzz selector
could not run because `hypothesis` is not installed in this environment.

Multi-surface audit escalation marking now uses Rust for case scoring, stable
priority ordering, escalation-cap enforcement, and full-report reason
projection. The wrapper keeps the prior Python implementation as fallback and
returns copied row dictionaries with the same audit-pack schema contract.
`python -m py_compile`, `cargo fmt --check`, `cargo check`, `python setup.py
build_ext --inplace`, the focused escalation selector (`3 passed, 791
deselected`), the broader multi-surface Rust selector (`17 passed, 777
deselected`), the audit-pack fixture
(`tests/test_multisurface_finbert_audit.py::test_build_multisurface_audit_pack_creates_cases_and_chunks`,
`1 passed`), and the full Rust acceleration suite (`794 passed`) passed for
this slice.

SEC extraction high-confidence truncation now uses Rust for ASCII item text
sections before falling back to the Python helper for non-ASCII input or native
failures. The wrapper preserves empty-section stub handling, successor Item
heading cuts, cross-reference suffix protection, TOC-window Part restart
skips, and confirmed later-Part restart truncation. `python -m py_compile`,
`cargo fmt --check`, `cargo check`, `python setup.py build_ext --inplace`, the
focused truncation selector (`3 passed, 794 deselected`), the combined
extraction selector (`6 passed, 791 deselected`), and the full Rust
acceleration suite (`797 passed`) passed. Suspicious-boundary diagnostics run
through the explicit migration entrypoint on `full_data_run/year_merged` with
`--max-files 1` processed `229` filings, extracted `1372` items, flagged `225`
items, and reported `2` embedded fail items.

Refinitiv ownership authority conventional-component grouping now uses Rust for
the per-PERMNO reducer after candidate metrics and pairwise alias diagnostics.
The wrapper preserves Python fallback behavior and reconstructs the existing
nested component-map/component-meta contract, including `union_value_sets` and
merged bridge-window payloads. `python -m py_compile`, `cargo fmt --check`,
`cargo check`, `python setup.py build_ext --inplace`, the focused conventional
component selector (`3 passed, 797 deselected`), the broader Refinitiv authority
Rust selector (`19 passed, 781 deselected`), the standalone authority pipeline
fixture (`2 passed`), and the full Rust acceleration suite (`800 passed`) passed
for this slice.

SEC embedded-heading GIJ asset-backed context detection now uses Rust for the
per-filing General Instruction J omission/substitution scan before falling back
to the Python helper. The wrapper preserves the existing dictionary contract,
including omit/substitute ranges, omitted Item IDs as a set, and the GIJ reason
string. `python -m py_compile`, `cargo fmt --check`, `cargo check`, `python
setup.py build_ext --inplace`, the focused GIJ selector (`3 passed, 800
deselected`), the migration-forced GIJ/stub fixture (`1 passed`), and the full
Rust acceleration suite (`803 passed`) passed. Suspicious-boundary diagnostics
run through the explicit migration entrypoint on `full_data_run/year_merged`
with `--max-files 1` processed `229` filings, extracted `1372` items, flagged
`225` items, and reported `2` embedded fail items.

SEC embedded-heading hit classification now uses Rust for ASCII item-text
sections before falling back to the Python verifier for non-ASCII input or
native failures. The wrapper preserves the `EmbeddedHeadingHit` dataclass
contract, warning/failure classifications, same-item continuation handling,
TOC-row guards, glued-title markers, successor-overlap promotion, part-restart
checks, and `max_hits` counting semantics. `python -m py_compile`, `cargo
fmt --check`, `cargo check`, `python setup.py build_ext --inplace`, the
focused embedded-heading selector (`3 passed, 803 deselected`), the
migration-forced embedded-heading verifier suite (`13 passed`), and the full
Rust acceleration suite (`806 passed`) passed. Suspicious-boundary diagnostics
run through the explicit migration entrypoint on `full_data_run/year_merged`
with `--max-files 1` processed `229` filings, extracted `1372` items, flagged
`225` items, and reported `2` embedded fail items.

FinBERT sentence-confusion review streaming sample target-position generation
now uses Rust for allocation-row validation, stable per-stratum seed expansion,
target ordinal/rank row construction, and target-summary rows. To preserve the
existing review-sample contract exactly, the Rust path delegates the actual
`random.Random(...).sample(...)` call to Python with the same SHA-256-derived
seed as the fallback. The wrapper keeps the Python implementation as fallback
and leaves the downstream Polars streaming selection unchanged. `python -m
py_compile`, `cargo fmt --check`, `cargo check`, `python setup.py build_ext
--inplace`, the focused target-position selector (`4 passed, 806 deselected`),
the migration-forced standalone confusion-review suite (`4 passed`), and the
full Rust acceleration suite (`810 passed`) passed for this slice.

SEC heuristic TOC line-range detection now uses Rust for the line-level
Table-of-Contents/Summary range scanner before falling back to the Python
helper. The wrapper preserves inline multi-`ITEM` rows, explicit/split TOC
markers, dense heading clusters, prose-gap splitting, page/dot-leader guards,
and the `max_lines` boundary. `python -m py_compile`, `cargo fmt --check`,
`cargo check`, `python setup.py build_ext --inplace`, the focused TOC selector
(`3 passed, 810 deselected`), the nearby SEC extraction/heuristics selector
(`9 passed, 804 deselected`), and the full Rust acceleration suite (`813
passed`) passed. Suspicious-boundary diagnostics run through the explicit
migration entrypoint on `full_data_run/year_merged` with `--max-files 1`
processed `229` filings, extracted `1372` items, flagged `225` items, and
reported `2` embedded fail items.

SEC heuristic character-level TOC end-position inference now uses Rust for
ASCII filing bodies before falling back to the Python regex helper. The wrapper
preserves the `int | None` cutoff contract, the `max_chars` window, four-valid
entry threshold, item-number normalization, page-number filtering, and
non-ASCII Python fallback for character-index safety. `python -m py_compile`,
`cargo fmt --check`, `cargo check`, `python setup.py build_ext --inplace`, the
focused TOC end-position selector (`3 passed, 813 deselected`), the nearby SEC
extraction/heuristics selector (`12 passed, 804 deselected`), and the full Rust
acceleration suite (`816 passed`) passed. Suspicious-boundary diagnostics run
through the explicit migration entrypoint on `full_data_run/year_merged` with
`--max-files 1` processed `229` filings, extracted `1372` items, flagged `225`
items, and reported `2` embedded fail items.

SEC pagination artifact removal now uses Rust for ASCII extracted item chunks
before falling back to the Python helper. The wrapper preserves newline
normalization, whitespace-only blank-line normalization, TOC header/page marker
removal, roman/numeric page removal only around blank separators, blank-run
collapse, final `strip()`, and non-ASCII Python fallback. `python -m
py_compile`, `cargo fmt --check`, `cargo check`, `python setup.py build_ext
--inplace`, the focused pagination selector (`3 passed, 816 deselected`), the
nearby SEC extraction/heuristics selector (`15 passed, 804 deselected`), and
the full Rust acceleration suite (`819 passed`) passed. A real-shard parity
probe over `full_data_run/year_merged/1995.parquet` checked `274` ASCII filings
with zero Rust/Python pagination mismatches. Suspicious-boundary diagnostics run
through the explicit migration entrypoint on `full_data_run/year_merged` with
`--max-files 1` processed `229` filings, extracted `1372` items, flagged `225`
items, and reported `2` embedded fail items.

SEC trailing Part-marker trimming now uses Rust for ASCII extracted item chunks
before falling back to the Python helper. The wrapper preserves trailing blank
line removal, bare `PART I`/`II`/`III`/`IV` marker removal when the suffix is
only spaces/tabs/colon/hyphen, non-marker tail preservation, all-blank input
behavior, final `rstrip()`, and non-ASCII Python fallback. `python -m
py_compile`, `cargo fmt --check`, `cargo check`, `python setup.py build_ext
--inplace`, the focused trim selector (`3 passed, 819 deselected`), the nearby
SEC extraction/heuristics selector (`18 passed, 804 deselected`), and the full
Rust acceleration suite (`822 passed`) passed. A real-shard parity probe over
`full_data_run/year_merged/1995.parquet` checked `274` ASCII filings with zero
Rust/Python trim mismatches. Suspicious-boundary diagnostics run through the
explicit migration entrypoint on `full_data_run/year_merged` with
`--max-files 1` processed `229` filings, extracted `1372` items, flagged `225`
items, and reported `2` embedded fail items.

SEC reserved-stub end detection now uses Rust for ASCII extracted item chunks
before falling back to the Python helper. The wrapper preserves `int | None`
offset semantics, blank/probe-only line skipping, first-nonempty-line
short-circuiting, the existing `EMPTY_ITEM_PATTERN` reserved-only behavior,
common `splitlines(keepends=True)` newline offsets, and non-ASCII Python
fallback for character-index safety. `python -m py_compile`, `cargo
fmt --check`, `cargo check`, `python setup.py build_ext --inplace`, the focused
reserved-stub selector (`3 passed, 822 deselected`), the nearby SEC
extraction/heuristics selector (`21 passed, 804 deselected`), and the full Rust
acceleration suite (`825 passed`) passed. A real-shard parity probe over
`full_data_run/year_merged/1995.parquet` checked `274` ASCII filings with zero
Rust/Python reserved-stub mismatches. Suspicious-boundary diagnostics run
through the explicit migration entrypoint on `full_data_run/year_merged` with
`--max-files 1` processed `229` filings, extracted `1372` items, flagged `225`
items, and reported `2` embedded fail items.

SEC line-start Item heading matching now uses Rust for ASCII heading/probe
lines before falling back to the Python helper. The wrapper preserves
`(item_id | None, match_end | None)` semantics, direct `ITEM` matches, optional
`PART ... ITEM ...` prefixes, slash/colon/hyphen Part separators, max-item
filtering, punctuation-sensitive end offsets, and non-ASCII Python fallback for
offset safety. `python -m py_compile`, `cargo fmt --check`, `cargo check`,
`python setup.py build_ext --inplace`, the focused line-start selector (`3
passed, 825 deselected`), the nearby SEC extraction/heuristics selector (`24
passed, 804 deselected`), and the full Rust acceleration suite (`828 passed`)
passed. A real-shard parity probe over `full_data_run/year_merged/1995.parquet`
checked `105410` ASCII lines with zero Rust/Python line-start mismatches.
Suspicious-boundary diagnostics run through the explicit migration entrypoint on
`full_data_run/year_merged` with `--max-files 1` processed `229` filings,
extracted `1372` items, flagged `225` items, and reported `2` embedded fail
items.

SEC cross-reference prefix detection now uses Rust for ASCII prefixes before
falling back to the Python helper. The wrapper preserves bool semantics, the
80-character tail window, cross-reference lead phrases, flexible
`in\s+part ...` whitespace, ASCII word-boundary behavior, and non-ASCII Python
fallback for character handling. `python -m py_compile`, `cargo fmt --check`,
`cargo check`, `python setup.py build_ext --inplace`, the focused prefix
selector (`3 passed, 828 deselected`), the nearby SEC extraction/heuristics
selector (`6 passed, 825 deselected`), and the full Rust acceleration suite
(`831 passed`) passed. A real-shard parity probe over
`full_data_run/year_merged/1995.parquet` checked `648` ASCII prefixes with zero
Rust/Python cross-reference prefix mismatches. Suspicious-boundary diagnostics
run through the explicit migration entrypoint on `full_data_run/year_merged`
with `--max-files 1` processed `229` filings, extracted `1372` items, flagged
`225` items, and reported `2` embedded fail items.

SEC compound item-line detection now uses Rust for ASCII heading lines before
falling back to the Python regex helper. The extraction wrapper and suspicious
boundary diagnostics share the same accelerated helper, preserving bool
semantics, `ITEM` word boundaries, flexible whitespace, digit or roman item
tokens with an optional single-letter suffix, final word-boundary rejection, and
non-ASCII Python fallback. `python -m py_compile`, `cargo fmt --check`, `cargo
check`, `python setup.py build_ext --inplace`, the focused compound-item
selector (`3 passed, 831 deselected`), the nearby SEC extraction/heuristics
selector (`9 passed, 825 deselected`), and the full Rust acceleration suite
(`834 passed`) passed. A real-shard parity probe over
`full_data_run/year_merged/1995.parquet` checked `495031` ASCII lines with zero
Rust/Python compound-item mismatches. Suspicious-boundary diagnostics run
through the explicit migration entrypoint on `full_data_run/year_merged` with
`--max-files 1` processed `229` filings, extracted `1372` items, flagged `225`
items, and reported `2` embedded fail items.

SEC heading-suffix prose detection now uses Rust for ASCII item-candidate
suffixes before falling back to the Python regex helper. The wrapper preserves
bool semantics, ASCII whitespace trim behavior, leading `space/tab/colon/hyphen`
stripping, the 160-character head window, anchored cross-reference suffix
phrases, `ITEM` and `PART` word-boundary checks, sentence-break detection,
lowercase-initial word ratio, comma/word-count heuristic, and non-ASCII Python
fallback. `python -m py_compile`, `cargo fmt --check`, `cargo check`, `python
setup.py build_ext --inplace`, the focused heading-suffix selector (`3 passed,
834 deselected`), the nearby SEC extraction/heuristics selector (`12 passed,
825 deselected`), and the full Rust acceleration suite (`837 passed`) passed.
A real-shard parity probe over `full_data_run/year_merged/1995.parquet` checked
`2956` ASCII item-candidate suffixes with zero Rust/Python heading-suffix
mismatches. Suspicious-boundary diagnostics run through the explicit migration
entrypoint on `full_data_run/year_merged` with `--max-files 1` processed `229`
filings, extracted `1372` items, flagged `225` items, and reported `2` embedded
fail items.

SEC Part-marker heading classification now uses Rust for ASCII lines before
falling back to the Python helper. The wrapper preserves the existing
`re.Match`-offset contract, prefix bullet/non-bullet rejection, combined
`PART ... ITEM ...` handling, comma/alphanumeric separator rejection,
10-character item-start threshold, punctuation and 80-character suffix
rejection, uppercase-letter ratio, short non-cross-reference suffix acceptance,
and non-ASCII Python fallback. `python -m py_compile`, `cargo fmt --check`,
`cargo check`, `python setup.py build_ext --inplace`, the focused Part-marker
heading selector (`3 passed, 837 deselected`), the nearby SEC
extraction/heuristics selector (`15 passed, 825 deselected`), and the full Rust
acceleration suite (`840 passed`) passed. A real-shard parity probe over
`full_data_run/year_merged/1995.parquet` checked `1213` ASCII Part-marker
matches with zero Rust/Python classification mismatches. Suspicious-boundary
diagnostics run through the explicit migration entrypoint on
`full_data_run/year_merged` with `--max-files 1` processed `229` filings,
extracted `1372` items, flagged `225` items, and reported `2` embedded fail
items.

SEC pageish-line detection now uses Rust for ASCII lines before falling back to
the Python regex helper. The wrapper preserves bool semantics, Python
`strip()`-then-match behavior for ASCII text, 1-4 digit page numbers,
hyphen-wrapped page numbers, 1-6 roman page markers, `page N` / `page N of M`
forms, rejection of `page1` without whitespace after `page`, and non-ASCII
Python fallback. `python -m py_compile`, `cargo fmt --check`, `cargo check`,
`python setup.py build_ext --inplace`, the focused pageish selector (`3 passed,
840 deselected`), the nearby SEC extraction/heuristics selector (`18 passed,
825 deselected`), and the full Rust acceleration suite (`843 passed`) passed.
A real-shard parity probe over `full_data_run/year_merged/1995.parquet` checked
`495031` ASCII lines with zero Rust/Python pageish-line mismatches.
Suspicious-boundary diagnostics run through the explicit migration entrypoint on
`full_data_run/year_merged` with `--max-files 1` processed `229` filings,
extracted `1372` items, flagged `225` items, and reported `2` embedded fail
items.

SEC Part-prefix helper detection now uses Rust for ASCII candidate prefixes
before falling back to the Python regex helpers. The wrapper preserves
`_prefix_is_part_only` bool semantics, `_prefix_part_tail` `str | None`
semantics with uppercase roman output, leading/trailing ASCII whitespace,
optional colon/hyphen for part-only prefixes, tail-only matching after start or
whitespace/`:;,.'` separators, rejection of invalid roman parts and suffixed
tails, and non-ASCII Python fallback. `python -m py_compile`, `cargo
fmt --check`, `cargo check`, `python setup.py build_ext --inplace`, the
focused Part-prefix selector (`3 passed, 843 deselected`), the nearby SEC
extraction/heuristics selector (`21 passed, 825 deselected`), and the full Rust
acceleration suite (`846 passed`) passed. A real-shard parity probe over
`full_data_run/year_merged/1995.parquet` checked `4879` ASCII item-candidate
prefixes with zero Rust/Python mismatches across both helpers.
Suspicious-boundary diagnostics run through the explicit migration entrypoint on
`full_data_run/year_merged` with `--max-files 1` processed `229` filings,
extracted `1372` items, flagged `225` items, and reported `2` embedded fail
items.

FinBERT bucket-length tuning conservative edge recommendation now uses Rust for
the scalar policy arithmetic in `finbert_bucket_length_tuning._recommend_edge`
before falling back to the Python helper. The wrapper preserves the
`(edge, reason)` tuple contract, `keep_current` and `no_rows` reasons,
quantile ceiling plus safety-margin arithmetic, round-up-to-multiple
validation, current-edge capping, lower-bound enforcement, and the existing
outer Polars aggregation/rebucketing data contracts. `python -m py_compile`,
`cargo fmt`, `cargo fmt --check`, `cargo check`, `python setup.py build_ext
--inplace`, the focused bucket-length selector (`14 passed, 836 deselected`),
and the full Rust acceleration suite (`850 passed`) passed. SEC
suspicious-boundary diagnostics were not applicable because this slice does not
touch SEC extraction or boundary heuristics.

SEC EDGAR metadata stripping now uses Rust for ASCII filing bodies in
`extraction._strip_edgar_metadata` before falling back to the original Python
regex helper. The wrapper preserves complete EDGAR metadata block removal for
`SEC-HEADER`, `Header`, `FileStats`/`File-Stats`, and
`XML_Chars`/`XML-Chars`, mixed-case closing tags, exact hyphen/underscore
backreference behavior, synthetic leading-header suffix stripping, truncated
leading-header removal, and non-ASCII Python fallback. `python -m py_compile`,
`cargo fmt`, `cargo fmt --check`, `cargo check`, `python setup.py build_ext
--inplace`, the focused strip selector (`3 passed, 850 deselected`), the nearby
SEC extraction selector (`6 passed, 847 deselected`), and the full Rust
acceleration suite (`853 passed`) passed. Suspicious-boundary diagnostics run
through the explicit migration entrypoint on `full_data_run/year_merged` with
`--max-files 1` processed `229` filings, extracted `1372` items, flagged `225`
items, and reported `2` embedded fail items; a matched forced-Python-strip run
produced identical suspicious CSV and report hashes.

LM2011 Refinitiv document-ownership doc-filing artifact reads now normalize
`KYPERMNO` through the existing Rust-backed batch expression instead of the
remaining row-wise `map_elements(_normalize_kypermno)` callback. The artifact
reader still preserves the eager parquet boundary, accepted lowercase/uppercase
`kypermno` column contract, duplicate-`doc_id` validation, and Python batch
fallback when the Rust extension is unavailable. `python -m py_compile`, `cargo
fmt`, `cargo fmt --check`, `cargo check`, and the focused doc-ownership
artifact/batch selector (`5 passed, 851 deselected`) passed. SEC
suspicious-boundary diagnostics were not applicable because this slice does not
touch SEC extraction or boundary heuristics.

Item-text cleaning scope diagnostics and manual-audit sample preparation now
use Rust-backed batch helpers for activation-status and audit-period
classification instead of dataframe `map_elements` scalar callbacks. The
wrappers preserve scalar helper behavior, Python batch fallback, scope status
labels, audit-period buckets, Polars eager-frame output contracts, and the
existing Rust manual-audit sample reducer. `python -m py_compile`, `cargo fmt`,
`cargo fmt --check`, `cargo check`, `python setup.py build_ext --inplace`, the
focused item-cleaning migration selector (`8 passed, 851 deselected`), and the
standalone item-text cleaning suite (`15 passed`) passed. The full Rust
acceleration suite also passed with `859 passed`. SEC
suspicious-boundary diagnostics were not applicable because this slice does not
touch SEC extraction or boundary heuristics.

FinBERT section-universe and sentence-provenance preparation now avoid the
remaining scalar `map_elements` callbacks for document-type and benchmark
item-code scope normalization. The section loader uses the existing CCM
Rust-backed batch expressions for SEC raw forms and canonical LM2011 forms,
while sentence provenance uses a new Rust-backed benchmark item-code
text-scope batch expression with Python batch fallback. `python -m py_compile`,
`cargo fmt`, `cargo fmt --check`, `cargo check`, `python setup.py build_ext
--inplace`, the focused migration selector (`4 passed, 859 deselected`), and
targeted downstream FinBERT gates under `PYTHONPATH=src_rust_migration` (`2
passed`) passed. The full Rust acceleration suite also passed with `863
passed`. SEC suspicious-boundary diagnostics were not applicable
because this slice does not touch SEC extraction or boundary heuristics.

LM2011 date bucketing now has Rust-backed batch expressions for
previous-month-end and quarter-start derivation. These expressions replace
scalar `map_elements` callbacks in event-panel prior-month price joins, Phase 0
validation SUE announcement prior-month joins, and quarterly Fama-MacBeth
grouping while preserving null propagation and scalar Python fallback behavior.
`python -m py_compile`, `cargo fmt`, `cargo fmt --check`, `cargo check`,
`python setup.py build_ext --inplace`, the focused date-bucket selector (`5 passed, 860
deselected`), and targeted downstream LM2011 validation/regression gates under
`PYTHONPATH=src_rust_migration` (`23 passed, 14 deselected, 1 warning`) passed.
The full Rust acceleration suite also passed with `865 passed`. SEC
suspicious-boundary diagnostics were not applicable because this slice does not
touch SEC extraction or boundary heuristics.

Multi-surface audit case sampling now builds deterministic stable sort keys
through a Rust-backed batch helper and Polars `map_batches`, removing the
scalar `map_elements` callback in `_with_stable_sort_key`. The wrapper keeps
the existing scalar helper behavior for digit-bearing keys, falls back to the
Python hash path for no-digit or non-ASCII keys, preserves null sort keys, and
retains the original eager DataFrame output contract. `python -m py_compile`,
`cargo fmt`, `cargo fmt --check`, `cargo check`, `python setup.py build_ext
--inplace`, the focused stable-sort selector (`6 passed, 863 deselected`), the
standalone multi-surface audit suite under `PYTHONPATH=src_rust_migration` (`4
passed`), and the full Rust acceleration suite (`869 passed`) passed. SEC
suspicious-boundary diagnostics were not applicable because this slice does not
touch SEC extraction or boundary heuristics.

Multi-surface audit item-boundary scoring now computes boundary-snippet risk
and snippet-delta risk with Rust-backed batch helpers and Polars `map_batches`,
removing two scalar struct `map_elements` callbacks from item case assembly.
The wrappers preserve the existing boundary leak/TOC-cluster semantics,
whitespace-normalized snippet comparison, null-as-empty snippet handling,
Python batch fallback, and eager DataFrame output contract. The same audit pass
also replaces control-case peer-group membership with native Polars struct
membership, removing the last scalar `map_elements` call from
`multisurface_audit.py`. `python -m py_compile`, `cargo fmt`, `cargo
fmt --check`, `cargo check`, `python setup.py build_ext --inplace`, the
combined multisurface selector (`13 passed, 861 deselected`), the standalone
multi-surface audit suite under
`PYTHONPATH=src_rust_migration` (`4 passed`), and the full Rust acceleration
suite (`874 passed`) passed. SEC suspicious-boundary diagnostics were not
applicable because this slice does not touch SEC extraction or boundary
heuristics.

LM2011 text-feature base construction now uses the existing Rust-backed batch
LM2011 form expression for both eager `_ensure_normalized_form` and LazyFrame
`_build_text_base_lf` paths. This removes the remaining scalar
`map_elements(normalize_lm2011_form_value)` callbacks from
`core/sec/lm2011_text.py` while preserving normalized-form data contracts,
existing pre-normalized column handling, Python batch fallback, and LazyFrame
semantics. `python -m py_compile`, `cargo fmt --check`, `cargo check`, the
focused form/text selector (`9 passed, 868 deselected`), the broader text
feature selector (`10 passed, 867 deselected`), and the full Rust acceleration
suite (`877 passed`) passed. No new Rust source was needed for this slice; it
reuses the already-built Rust batch form normalizer. SEC suspicious-boundary
diagnostics were not applicable because this slice does not touch SEC
extraction or boundary heuristics.

LM2011 visible-prefix source construction now uses a reusable Rust-backed batch
LM token-count expression for Polars text columns. The expression batches
non-null text through the existing Rust count-only tokenizer, preserves the
prior scalar `map_elements` null behavior, and is used by
`lm2011_sample_post_refinitiv_runner.py` for both original joined sentence text
and retained visible-prefix text. `python -m py_compile`, `cargo fmt`, `cargo
fmt --check`, `cargo check`, the focused token-count selector (`4 passed, 877
deselected`), the targeted LM2011 sample runner gate under
`PYTHONPATH=src_rust_migration` (`18 passed, 69 deselected`), and the full Rust
acceleration suite (`881 passed`) passed. No new Rust source was needed for
this slice; it reuses the existing count-only token batch export. SEC
suspicious-boundary diagnostics were not applicable because this slice does not
touch SEC extraction or boundary heuristics.

Item-text cleaning manual-audit fallback sampling now uses the existing
Rust-backed batch audit-period expression instead of the fallback-only scalar
`map_elements(_audit_period_py)` callback. The output selection contract is
unchanged, and the Python fallback remains available through the batch helper.
The LM2011 post-Refinitiv extension CSV companion writer also now serializes
`signal_inputs`, `left_signal_inputs`, and `right_signal_inputs` columns with a
Rust-backed batch JSON helper and Python fallback, preserving the prior
`json.dumps` output contract for optional list/scalar values. `python -m
py_compile`, `cargo fmt`, `cargo fmt --check`, `cargo check`, `python setup.py
build_ext --inplace`, the focused item-cleaning/CSV selector (`6 passed, 878
deselected`), the targeted extension runner gate under
`PYTHONPATH=src_rust_migration` (`13 passed, 74 deselected`), and the full Rust
acceleration suite (`884 passed`) passed. A repository scan now finds no
remaining `map_elements` calls under `src_rust_migration/thesis_pkg`. SEC
suspicious-boundary diagnostics were not applicable because this slice does not
touch SEC extraction or boundary heuristics.

LM2011 quarterly cross-sectional OLS design preparation now uses a
column-oriented Rust entrypoint before falling back to the older Rust row-dict
helper and then the Python implementation. This removes the normal-path
`DataFrame.to_dicts()` conversion from `_cross_section_design_rows` while
preserving dependent-variable, regressor, industry-dummy, and fallback output
contracts. `python -m py_compile`, `cargo fmt`, `cargo fmt --check`, `cargo
check`, `python setup.py build_ext --inplace`, the focused design selector (`3
passed, 881 deselected`), the targeted LM2011 regression gate under
`PYTHONPATH=src_rust_migration` (`16 passed, 12 deselected, 1 warning`), and
the full Rust acceleration suite (`884 passed`) passed. SEC
suspicious-boundary diagnostics were not applicable because this slice does not
touch SEC extraction or boundary heuristics.

LM2011 monthly strategy factor-loading summary construction now uses a
column-oriented Rust entrypoint before falling back to the older Rust row-dict
helper and then the Python implementation. This removes the normal-path
`DataFrame.to_dicts()` conversion from `_strategy_factor_loading_rows` while
preserving signal grouping order, null-row filtering, FF4 coefficient output
columns, Python fallback behavior, and the monthly strategy summary schema.
`python -m py_compile`, `cargo fmt`, `cargo fmt --check`, `cargo check`,
`python setup.py build_ext --inplace`, the focused strategy factor-loading
selector (`3 passed, 881 deselected`), and the targeted LM2011 pipeline gate
under `PYTHONPATH=src_rust_migration` (`11 passed, 72 deselected, 1 warning`)
passed. The full Rust acceleration suite also passed with `884 passed`. SEC
suspicious-boundary diagnostics were not applicable because this slice does
not touch SEC extraction or boundary heuristics.

LM2011 Table IA.II monthly strategy result-row expansion now uses a
column-oriented Rust entrypoint before falling back to the older Rust row-dict
helper and then the Python implementation. This removes the normal-path
`DataFrame.to_dicts()` conversion from `_table_ia_ii_result_rows` while
preserving signal row order, coefficient expansion, optional standard-error
and t-stat columns, schema contracts, and Python fallback behavior. `python -m
py_compile`, `cargo fmt`, `cargo fmt --check`, `cargo check`, `python setup.py
build_ext --inplace`, the focused Table IA.II selector (`3 passed, 881
deselected`), the targeted LM2011 regression gate under
`PYTHONPATH=src_rust_migration` (`4 passed, 24 deselected`), and the targeted
LM2011 pipeline gate (`11 passed, 72 deselected, 1 warning`) passed. The full
Rust acceleration suite also passed with `884 passed`. SEC suspicious-boundary
diagnostics were not applicable because this slice does not touch SEC
extraction or boundary heuristics.

LM2011 event-window per-document regression metric construction now uses a
column-oriented Rust entrypoint before falling back to the older Rust row-dict
helper and then the Python implementation. This removes the normal-path
`DataFrame.to_dicts()` conversion from `_regression_metrics_from_window_rows`
while preserving `doc_id`/`relative_day` ordering, alpha/RMSE calculation,
`n_obs`, row output contracts, and Python fallback behavior. `python -m
py_compile`, `cargo fmt`, `cargo fmt --check`, `cargo check`, `python setup.py
build_ext --inplace`, the focused regression-window selector (`3 passed, 881
deselected`), and the targeted LM2011 event-window pipeline gate under
`PYTHONPATH=src_rust_migration` (`7 passed, 76 deselected`) passed. The full
Rust acceleration suite also passed with `884 passed`. SEC
suspicious-boundary diagnostics were not applicable because this slice does not
touch SEC extraction or boundary heuristics.

LM2011 streaming text-feature pass-1 row construction now uses a
column-oriented Rust entrypoint before falling back to the older Rust row-dict
helper and then the Python implementation. This removes the normal-path
`DataFrame.to_dicts()` conversion from `_prepare_pass1_rows` by sending
metadata columns and text values separately, while preserving Python-cleaned
text support, metadata projection, compact matched-count JSON, document
frequency counts, row output contracts, and Python fallback behavior. `python
-m py_compile`, `cargo fmt`, `cargo fmt --check`, `cargo check`, `python
setup.py build_ext --inplace`, the focused pass-1 selector (`4 passed, 880
deselected`), and the targeted LM2011 text-feature pipeline gate under
`PYTHONPATH=src_rust_migration` (`21 passed, 62 deselected`) passed. The full
Rust acceleration suite also passed with `884 passed`. SEC
suspicious-boundary diagnostics were not applicable because this slice does not
touch SEC extraction or boundary heuristics.

LM2011 document-stat preparation now uses a column-oriented Rust entrypoint
before falling back to the older Rust row-dict helper and then the Python
implementation. This removes the normal-path batch `DataFrame.to_dicts()`
conversion from `_prepare_document_stats` by sending metadata columns, `doc_id`
values, and text values separately, while preserving Python-cleaned text
support, base-row metadata projection, token-count dictionaries, recognized
word totals, IDF inputs, and Python fallback behavior. `python -m py_compile`,
`cargo fmt`, `cargo fmt --check`, `cargo check`, `python setup.py build_ext
--inplace`, the focused document-stat selector (`4 passed, 880 deselected`),
and the targeted LM2011 text-feature pipeline gate under
`PYTHONPATH=src_rust_migration` (`21 passed, 62 deselected`) passed. The full
Rust acceleration suite also passed with `884 passed`. SEC
suspicious-boundary diagnostics were not applicable because this slice does not
touch SEC extraction or boundary heuristics.

LM2011 streaming text-feature pass-2 feature-row construction now uses a
column-oriented Rust entrypoint before falling back to the older Rust row-dict
helper and then the Python implementation. This removes the normal-path
`pass1_df.to_dicts()` conversion in both streaming writers by sending pass-1
parquet columns directly to Rust, while preserving optional `item_id`, raw
form, normalized form, token-count, matched-count JSON, signal proportion,
TF-IDF, cleaning-policy, and fallback output contracts. `python -m py_compile`,
`cargo fmt`, `cargo fmt --check`, `cargo check`, `python setup.py build_ext
--inplace`, the focused pass-1 feature-row selector (`5 passed, 881
deselected`), and the targeted LM2011 text-feature pipeline gate under
`PYTHONPATH=src_rust_migration` (`21 passed, 62 deselected`) passed. The full
Rust acceleration suite also passed with `886 passed`. SEC
suspicious-boundary diagnostics were not applicable because this slice does not
touch SEC extraction or boundary heuristics.

SEC pipeline no-item stats aggregation now uses a column-oriented Rust
entrypoint before falling back to the existing Rust row-dict helper and then the
Python implementation. `aggregate_no_item_stats_csvs` still owns CSV I/O and
Polars shape handling in Python, while the normal aggregation handoff avoids
materializing `_NO_ITEM_STATS_COLUMNS` through `to_dicts()`. The verification
gates passed: `python -m py_compile`, `cargo fmt`, `cargo fmt --check`,
`cargo check`, and `python setup.py build_ext --inplace`; the focused no-item
migration selector passed with `4 passed, 883 deselected`, and the full Rust
acceleration suite passed with `887 passed`. The extension rebuild emitted the
known nonfatal setuptools logging and Cargo readonly-cache warnings. SEC
suspicious-boundary diagnostics were not applicable because this slice only
changes no-item stats aggregation, not SEC extraction or boundary heuristics.

Refinitiv LSEG provider response-frame fingerprinting now uses a column-oriented
Rust entrypoint before falling back to the existing Rust row-dict helper and
then the Python implementation. The API response metadata contract is unchanged:
empty frames still hash as `sha1(b"empty")`, JSON keys are sorted, unsupported
or non-ASCII fast-path values fall through to the Python `json.dumps(...,
default=str, sort_keys=True)` implementation, and provider CSV/Polars response
normalization remains in Python. The verification gates passed: `python -m
py_compile`, `cargo fmt`, `cargo fmt --check`, `cargo check`, `python setup.py
build_ext --inplace`, the focused LSEG provider fingerprint selector (`4
passed, 885 deselected`), and the full Rust acceleration suite (`889 passed`).
The extension rebuild emitted the known nonfatal setuptools logging and Cargo
readonly-cache warnings.

Refinitiv ownership authority allowlist key normalization now uses a
column-oriented Rust entrypoint before falling back to the existing Rust
row-dict helper and then the Python implementation. The key contract is
unchanged: `KYPERMNO` and `candidate_ric` values are normalized with the same
trim/empty-to-null semantics, duplicate keys still collapse through the Python
`set` return shape, and missing or incompatible column fast paths fall back to
the row/Python implementations. The verification gates passed: `python -m
py_compile`, `cargo fmt`, `cargo fmt --check`, `cargo check`, `python setup.py
build_ext --inplace`, the focused allowlist selector (`4 passed, 885
deselected`), and the full Rust acceleration suite (`889 passed`). The
extension rebuild emitted the known nonfatal setuptools logging and Cargo
readonly-cache warnings.

FinBERT sentence-confusion review majority-bucket metric row aggregation now
uses a column-oriented Rust entrypoint before falling back to the existing Rust
row-dict helper and then the Python implementation. The review-summary contract
is unchanged: null/blank majority buckets are skipped, `confusion_cell` remains
required for retained rows, `sample_weight` keeps the weighted `or 1.0`
semantics, and precision/recall/specificity outputs retain the same optional
float behavior. The verification gates passed: Python bytecode compile, cargo
format/check, cargo check, extension rebuild, the focused bucket-metric
selector (`4 passed, 886 deselected`), and the full Rust acceleration suite
(`890 passed`). The extension rebuild emitted the known nonfatal setuptools
logging and Cargo readonly-cache warnings.

FinBERT sentence-length report JSON-ready frame record serialization now uses a
column-oriented Rust entrypoint before falling back to the existing Rust
row-dict helper and then the Python implementation. The summary-payload
contract is unchanged: records keep DataFrame column order, null values stay
null, and date/datetime-like values are converted through `isoformat()` while
non-date scalars pass through unchanged. The verification gates passed: Python
bytecode compile, cargo format/check, cargo check, extension rebuild, the
focused sentence-length selector (`4 passed, 887 deselected`), and the full
Rust acceleration suite (`891 passed`). The extension rebuild emitted the
known nonfatal setuptools logging and Cargo readonly-cache warnings.

FinBERT tail document-surface eager aggregation now uses a column-oriented Rust
entrypoint before falling back to the existing Rust row-dict helper and then the
Python implementation. The large-data `build_finbert_tail_doc_surface_lf`
LazyFrame path remains unchanged, while the eager materialized wrapper avoids
normal-path `sentence_scores_df.to_dicts()` and preserves text-scope alias
normalization, grouping keys, top-tail metrics, token-weight semantics, and
Python fallback behavior. The verification gates passed: Python bytecode
compile, cargo format/check, cargo check, extension rebuild, the focused
FinBERT tail selector (`4 passed, 888 deselected`), the standalone
tail-feature suite with `PYTHONPATH=src_rust_migration` (`4 passed`), and the
full Rust acceleration suite (`892 passed`). The extension rebuild emitted the
known nonfatal setuptools logging and Cargo readonly-cache warnings.

FinBERT sentence-preprocessing manifest count aggregation now uses a
column-oriented Rust entrypoint before falling back to the existing Rust
row-dict helper and then the Python implementation. The manifest contract is
unchanged: processed/reused file counts and sentence, oversize, chunking,
warning-split, cleaned-scope, dropped-cleaning, and flagged-cleaning row totals
are reduced with the same integer semantics. The verification gates passed:
Python bytecode compile, cargo format/check, cargo check, extension rebuild,
the focused preprocessing manifest-count selector (`4 passed, 889
deselected`), and the full Rust acceleration suite (`893 passed`). The
extension rebuild emitted the known nonfatal setuptools logging and Cargo
readonly-cache warnings.

LM2011 extension result-row, quarterly fit-row, and skipped-quarter diagnostic
row construction now use column-oriented Rust entrypoints before falling back
to the existing Rust row-dict helpers and then the Python implementations. The
contracts are unchanged: result rows retain `n_obs` derivation and
normal-approximation p-values, quarterly fit rows retain common-sample policy
metadata and all regression fit passthrough fields, and skipped-quarter rows
retain skip reasons, regressor diagnostics, and condition-number/null
behavior. The verification gates passed: Python bytecode compile, cargo
format/check, cargo check, extension rebuild, the focused extension-row
selector (`12 passed, 884 deselected`), and the full Rust acceleration suite
(`896 passed`). The extension rebuild emitted the known nonfatal setuptools
logging and Cargo readonly-cache warnings.

FinBERT bucket-length tuning summary row assembly now uses a column-oriented
Rust entrypoint before falling back to the existing Rust row-dict helper and
then the Python implementation. The summary contract is unchanged: output rows
remain ordered as short/medium/long, missing buckets are filled with the same
empty quantile values, and current edge/max-length annotations are retained.
The verification gates passed: Python bytecode compile, cargo format/check,
cargo check, extension rebuild, the focused bucket-length summary selector (`4
passed, 893 deselected`), and the full Rust acceleration suite (`897 passed`).
The extension rebuild emitted the known nonfatal setuptools logging and Cargo
readonly-cache warnings.

FinBERT benchmark dataset audit-share row construction now uses a
column-oriented Rust entrypoint before falling back to the existing Rust
row-dict helper and then the Python implementation. The share-row contract is
unchanged: rows retain the sorted Polars grouping key columns, eligible and
selected counts, per-group share denominators, and `share_diff` calculations.
The verification gates passed: Python bytecode compile, cargo format/check,
cargo check, extension rebuild, the focused dataset share-row selector (`4
passed, 894 deselected`), and the full Rust acceleration suite (`898 passed`).
The extension rebuild emitted the known nonfatal setuptools logging and Cargo
readonly-cache warnings.

FinBERT sentence-confusion review streaming sample target-position generation
now uses a column-oriented Rust entrypoint before falling back to the existing
Rust row-dict helper and then the Python implementation. The sampling contract
is unchanged: deterministic per-stratum seeds, target ordinal ordering,
zero-sample stratum skipping, over-allocation validation, and summary first/last
ordinal values are preserved. The verification gates passed: Python bytecode
compile, cargo format/check, cargo check, extension rebuild, the focused
target-position selector (`5 passed, 894 deselected`), and the full Rust
acceleration suite (`899 passed`). The extension rebuild emitted the known
nonfatal setuptools logging and Cargo readonly-cache warnings.

FinBERT high-confidence sentence-example markdown report rendering now uses a
column-oriented Rust entrypoint before falling back to the existing Rust
row-dict helper and then the Python implementation. The report contract is
unchanged: metadata headers, ordered item/sentiment sections, eligible/listed
counts, sample ordering, probability formatting, and scalar string rendering
are preserved. The verification gates passed: Python bytecode compile, cargo
format/check, cargo check, extension rebuild, the focused sentence-example
markdown selector (`4 passed, 896 deselected`), and the full Rust acceleration
suite (`900 passed`). The extension rebuild emitted the known nonfatal
setuptools logging and Cargo readonly-cache warnings.

FinBERT sentence-confusion review neighbor-target construction now uses a
column-oriented Rust entrypoint from the sampled Polars frame before falling
back to the existing Rust row-dict helper and then the Python implementation.
The neighbor-context contract is unchanged: null and invalid sentence indexes
are skipped, previous/next target sentence indexes are derived with the same
integer conversion semantics, and `review_case_id`/`benchmark_row_id` values
are passed through unchanged. The verification gates passed: Python bytecode
compile, cargo format/check, cargo check, extension rebuild, the focused
neighbor-target selector (`5 passed, 897 deselected`), and the full Rust
acceleration suite (`902 passed`). The extension rebuild emitted the known
nonfatal setuptools logging and Cargo readonly-cache warnings.

FinBERT sentence-confusion review reviewed-case row construction now uses a
column-oriented Rust entrypoint from the sampled Polars frame before falling
back to the existing Rust row-dict helper and then the Python implementation.
The review merge contract is unchanged: review records still avoid overwriting
sample columns, extra review fields retain the `review_` prefix, missing labels
produce `uncertain` outcomes and missing-case diagnostics, and confusion-cell
classification uses the same final gold-negative source precedence. The
verification gates passed: Python bytecode compile, cargo format/check, cargo
check, extension rebuild, the focused reviewed-row selector (`5 passed, 899
deselected`), and the full Rust acceleration suite (`904 passed`). The
extension rebuild emitted the known nonfatal setuptools logging and Cargo
readonly-cache warnings.

Multi-surface audit chunk construction now uses a column-oriented Rust
entrypoint from the audit-case Polars frame before falling back to the existing
Rust row-dict helper and then the Python implementation. The chunking contract
is unchanged: records are bucket-interleaved by case source, text scope, and
full-report-needed status, then assigned round-robin across configured chunks;
chunk CSV rows are still materialized with the existing audit-case schema and
sorting. The verification gates passed: Python bytecode compile, cargo
format/check, cargo check, extension rebuild, the focused multi-surface chunk
selector (`5 passed, 901 deselected`), and the full Rust acceleration suite
(`906 passed`). The extension rebuild emitted the known nonfatal setuptools
logging and Cargo readonly-cache warnings.

LM2011 phase-0 validation Packet A MD&A row construction now uses a
column-oriented Rust entrypoint from the item Polars frame before falling back
to the existing Rust row-dict helper and then the Python implementation. The
row contract is unchanged: legacy/current token counts, appendix token counts,
recognized master-dictionary counts, threshold flags, marker placeholders,
paper-cleaning no-tail-anchor fields, and snippets are preserved. The
verification gates passed: Python bytecode compile, cargo format/check, cargo
check, extension rebuild, the focused Packet A MD&A selector (`4 passed, 903
deselected`), and the full Rust acceleration suite (`907 passed`). The
extension rebuild emitted the known nonfatal setuptools logging and Cargo
readonly-cache warnings.

SEC/CCM run-report markdown table rendering now uses a column-oriented Rust
entrypoint from the clipped Polars frame before falling back to the existing
Rust row-dict helper and then the Python implementation. The rendering contract
is unchanged: column order, scalar formatting, date/datetime handling, pipe
escaping, empty values, and truncation notes are preserved. The verification
gates passed: Python bytecode compile, cargo format/check, cargo check,
extension rebuild, the focused SEC/CCM markdown selector (`7 passed, 901
deselected`), and the full Rust acceleration suite (`908 passed`). The
extension rebuild emitted the known nonfatal setuptools logging and Cargo
readonly-cache warnings.

Refinitiv LSEG lookup request-item construction now uses a column-oriented
Rust entrypoint from the lookup snapshot Polars frame before falling back to
the existing Rust row-dict helper and then the Python implementation. The
request-item contract is unchanged: blank bridge row IDs and blank lookup
inputs are skipped, identifier types keep their configured order, stable item
IDs remain derived from bridge row ID plus identifier type, and request items
retain their existing dataclass conversion behavior. The verification gates
passed: Python bytecode compile, cargo format/check, cargo check, extension
rebuild, the focused LSEG lookup item-builder selector (`4 passed, 905
deselected`), and the full Rust acceleration suite (`909 passed`). The
extension rebuild emitted the known nonfatal setuptools logging and Cargo
readonly-cache warnings.

Refinitiv LSEG lookup batch-response normalization now uses a column-oriented
Rust entrypoint for both request-item fields and the standardized response
frame before falling back to the existing Rust row-dict helper and then the
Python implementation. The response contract is unchanged: provider field
aliases are standardized in Python, duplicate instruments keep last-response
semantics, lookup inputs are normalized identically, unmatched items retain
null returned fields, and output columns stay ordered by
`LOOKUP_RESPONSE_COLUMNS`. The verification gates passed: Python bytecode
compile, cargo format/check, cargo check, extension rebuild, the focused LSEG
lookup selector (`8 passed, 902 deselected`), and the full Rust acceleration
suite (`910 passed`). The extension rebuild emitted the known nonfatal
setuptools logging and Cargo readonly-cache warnings.

Refinitiv analyst LSEG actuals batch-response normalization now uses a
column-oriented Rust entrypoint for request item fields and standardized
actuals response columns before falling back to the existing Rust row-dict
helper and then the Python implementation. The actuals contract is unchanged:
response instruments remain raw-string keyed, item windows still come from
`SDate`/`EDate`, announcement-date filtering preserves per-item response-row
indexes, fiscal period/end-date parsing remains ISO-normalized, and
`row_parse_status` values are unchanged. The verification gates passed:
Python bytecode compile, cargo format/check, cargo check, extension rebuild,
the focused analyst response selector (`4 passed, 907 deselected`), and the
full Rust acceleration suite (`911 passed`). The extension rebuild emitted the
known nonfatal setuptools logging and Cargo readonly-cache warnings.

Refinitiv analyst LSEG estimates batch-response normalization now uses a
column-oriented Rust entrypoint for request item fields and standardized
estimate response columns before falling back to the existing Rust row-dict
helper and then the Python implementation. The estimates contract is
unchanged: response instruments remain raw-string keyed, request windows still
come from `SDate`/`EDate`, calc-date filtering preserves per-item response-row
indexes, fiscal period/end-date and estimate-count normalization remain
unchanged, and `row_parse_status` values are preserved. The verification gates
passed: Python bytecode compile, cargo format/check, cargo check, extension
rebuild, the focused analyst response selector (`5 passed, 907 deselected`),
and the full Rust acceleration suite (`912 passed`). The extension rebuild
emitted the known nonfatal setuptools logging and Cargo readonly-cache
warnings.

Refinitiv ownership LSEG universe batch-response normalization now uses a
column-oriented Rust entrypoint for request item fields and standardized
ownership response columns before falling back to the existing Rust row-dict
helper and then the Python implementation. The universe response contract is
unchanged: lookup text normalization still keys responses by instrument, item
windows still come from `SDate`/`EDate`, returned dates are ISO-normalized and
filtered within the request window, empty value/category rows are skipped, and
the output schema remains the ownership-universe result schema with `item_id`.
The verification gates passed: Python bytecode compile, cargo format/check,
cargo check, extension rebuild, the focused ownership response selector (`4
passed, 909 deselected`), and the full Rust acceleration suite (`913 passed`).
The extension rebuild emitted the known nonfatal setuptools logging and Cargo
readonly-cache warnings.

Refinitiv document-ownership LSEG exact and fallback batch-response
normalization now use a shared column-oriented Rust entrypoint for request item
fields and standardized ownership response columns before falling back to the
existing Rust row-dict helper and then the Python implementations. The
document response contract is unchanged: exact/fallback stage wrappers keep
their request-stage labels, response dates and categories retain the same
normalization, all-empty rows are skipped, institutional-category flags are
preserved, and the output schema remains the document ownership raw schema
with `item_id`. The verification gates passed: Python bytecode compile, cargo
format/check, cargo check, extension rebuild, the focused ownership response
selector (`5 passed, 909 deselected`), and the full Rust acceleration suite
(`914 passed`). The extension rebuild emitted the known nonfatal setuptools
logging and Cargo readonly-cache warnings.

Refinitiv ownership LSEG universe request-item construction now uses a
column-oriented Rust entrypoint for the handoff frame before falling back to
the existing Rust row-dict helper and then the Python implementation. The item
contract is unchanged: retrieval eligibility is truthy-filtered, lookup row
IDs and candidate RICs use the same text normalization, request windows still
normalize to month boundaries, stable item IDs remain derived from
`ownership_lookup_row_id`, and Python still materializes payload row dicts for
`RequestItem.payload["handoff_row"]`. The verification gates passed: Python
bytecode compile, cargo format/check, cargo check, extension rebuild, the
focused ownership item-builder selector (`4 passed, 911 deselected`), and the
full Rust acceleration suite (`915 passed`). The extension rebuild emitted the
known nonfatal setuptools logging and Cargo readonly-cache warnings.

Refinitiv document-ownership LSEG exact and fallback request-item construction
now use column-oriented Rust entrypoints for the request frame before falling
back to the existing Rust row-dict helpers and then the Python implementations.
The item contract is unchanged: retrieval eligibility is truthy-filtered,
document IDs and authoritative RICs use the same text normalization, exact
items duplicate `target_effective_date` into start/end request dates, fallback
items preserve their fallback-window bounds, stable item IDs remain stage/doc
ID based, and Python still materializes payload row dicts for
`RequestItem.payload["request_row"]`. The verification gates passed: Python
bytecode compile, cargo format/check, cargo check, extension rebuild, the
focused ownership item-builder selector (`5 passed, 911 deselected`), and the
full Rust acceleration suite (`916 passed`). The extension rebuild emitted the
known nonfatal setuptools logging and Cargo readonly-cache warnings.

Refinitiv analyst LSEG actuals and estimates request-item construction now
uses column-oriented Rust entrypoints for the request frame before falling back
to the existing Rust row-dict helpers and then the Python implementations. The
item contract is unchanged: retrieval eligibility is truthy-filtered,
request-group IDs keep the stable-ID encoding path, effective collection RICs
are stringified as before, actuals keep the `FI0` quarterly request parameters,
estimates keep the configured `FQ1`/`FQ2` period expansion, and Python still
materializes payload row dicts for `RequestItem.payload["request_row"]`. The
verification gates passed: Python bytecode compile, cargo format/check, cargo
check, extension rebuild, the focused analyst item-builder selector (`4
passed, 913 deselected`), and the full Rust acceleration suite (`917 passed`).
The extension rebuild emitted the known nonfatal setuptools logging and Cargo
readonly-cache warnings.

Refinitiv document-ownership exact/fallback institutional hit selection and
fallback-request row selection now use column-oriented Rust entrypoints before
falling back to the existing Rust row-dict helpers and then the Python
implementations. The selection contracts are unchanged: document IDs retain
lookup-text normalization, institutional ownership values retain the same
negative/null/drop and 100-percent cap rules, exact hits still reject
conflicting per-document values, fallback hits still choose the latest
in-window response date, and fallback requests still exclude exact hits,
conflicts, missing document IDs, and ineligible requests. The verification
gates passed: Python bytecode compile, cargo format/check, cargo check,
extension rebuild, the focused doc-ownership selector (`12 passed, 908
deselected`), and the full Rust acceleration suite (`920 passed`). The
extension rebuild emitted the known nonfatal setuptools logging and Cargo
readonly-cache warnings.

Refinitiv document-ownership request-row construction now uses a
column-oriented Rust entrypoint over the doc-filing, authority-decision, and
authority-exception frames before falling back to the existing Rust row-dict
helper and then the Python implementation. The request contract is unchanged:
document IDs and KYPERMNOs retain the same normalization, date-varying
authority exception windows still resolve against the target quarter end,
review/missing-authority exclusion reasons are preserved, min/max request-date
clipping still applies to fallback windows, and the output schema remains
`DOC_OWNERSHIP_REQUEST_COLUMNS`. The verification gates passed: Python
bytecode compile, cargo format/check, cargo check, extension rebuild, the
focused doc-ownership request selector (`4 passed, 917 deselected`), and the
full Rust acceleration suite (`921 passed`). The extension rebuild emitted the
known nonfatal setuptools logging and Cargo readonly-cache warnings.

Refinitiv document-ownership universe-diagnostics summary reduction now uses a
column-oriented Rust entrypoint over the normalized detail frame before falling
back to the existing Rust row-dict helper and then the Python implementation.
The summary contract is unchanged: backbone/request/final overlap counts,
doc-set equality flags, final retrieval-status counts, non-null ownership
counts, mismatch CIK top counts, and overlap rates remain identical to the
Python reducer. The verification gates passed: Python bytecode compile, cargo
format/check, cargo check, extension rebuild, the focused doc-ownership
universe-diagnostics selector (`6 passed, 920 deselected`), and the full Rust
acceleration suite (`926 passed`). The extension rebuild emitted the known
nonfatal setuptools logging and Cargo readonly-cache warnings.

Refinitiv ownership authority candidate-metric construction now uses a
column-oriented Rust entrypoint over the normalized ownership row-summary and
ownership result frames before falling back to the existing Rust row-dict
helper and then the Python implementation. The metric contract is unchanged:
candidate keys still use normalized `KYPERMNO`/RIC pairs, conventional/effective
/ticker role flags keep the same source-family semantics, bridge and ownership
date spans are preserved, same-date/category value conflicts still use the
existing distinct-value rules, and downstream authority decisions still receive
the same observation value sets and unique row sets. The newer column-native
candidate-metric path no longer returns reconstructed source `request_rows` or
`result_rows`; those compatibility payloads remain only on the older row-record
Rust/Python fallback path. The verification gates passed: Python bytecode
compile, cargo format/check, cargo check, extension rebuild, the focused
authority candidate-metric selector (`5 passed, 918 deselected`), and the full
Rust acceleration suite (`923 passed`). The extension rebuild emitted the known
nonfatal setuptools logging and Cargo readonly-cache warnings.

Refinitiv ownership authority conventional-component grouping now uses a
metadata-oriented Rust entrypoint over `candidate_meta` and `pair_meta` before
falling back to the existing Rust row-dict helper and then the Python
implementation. This removes Python-side construction of reducer row payloads
on the fast path while preserving the same component IDs, benign-alias
union-find grouping, canonical member selection, bridge-window merging,
ownership union maps, and review flags. The verification gates passed: Python
bytecode compile, cargo format/check, cargo check, extension rebuild, the
focused conventional-components selector (`4 passed, 920 deselected`), and the
full Rust acceleration suite (`924 passed`). The extension rebuild emitted the
known nonfatal setuptools logging and Cargo readonly-cache warnings.

FinBERT sentence confusion-review sample ID/order assignment now uses a
column-oriented Rust entrypoint over the sorted sample frame before falling
back to the existing Rust row-dict helper and then the Python implementation.
The output contract is unchanged: source sample columns are preserved,
`sample_order` is assigned from one-based sorted position, and
`review_case_id` is overwritten with the stable `finbert_review_######`
identifier before neighbor-context attachment and labeling chunk export. The
verification gates passed: Python bytecode compile, cargo format/check, cargo
check, extension rebuild, the focused sample-ID selector (`5 passed, 923
deselected`), and the full Rust acceleration suite (`928 passed`). The
extension rebuild emitted the known nonfatal setuptools logging and Cargo
readonly-cache warnings.

The Rust extension source has been refactored so
`rust/lm2011_rust/src/lib.rs` is no longer the monolithic PyO3 registration
file. `lib.rs` now delegates `#[pymodule]` registration to focused
`py_exports` domain modules while preserving the same Python-visible
`_lm2011_rust` exports and the existing Rust fast paths and Python fallback
contracts. The latest verification gates passed after the export split:
`cargo fmt`, `cargo fmt --check`, `cargo check`, Python export smoke
(`435` source pyfunctions, `435` unique source pyfunctions, no missing module
exports, `Lm2011TokenCounter` present), `python setup.py build_ext --inplace`,
focused cross-domain real-extension tests (`6 passed, 932 deselected`), and
the full migration parity suite (`938 passed`). The extension rebuild emitted
the known nonfatal setuptools logging and Cargo readonly-cache warnings.

The Refinitiv Step 1 and ownership-authority column fast paths have been
tightened so the targeted reducers are genuinely column-native internally.
`refinitiv_bridge_resolution_frame_rows_from_columns` now derives resolution
rows directly from typed column vectors instead of first rebuilding source
`PyDict` rows. `refinitiv_bridge_ownership_universe_handoff_columns` remains
column-native for the ownership-universe handoff. The authority
candidate-metric column reducer now computes request counts, support flags,
date spans, observation value sets, and unique row sets from column vectors and
does not call row-object reconstruction helpers or return reconstructed source
row payloads. Authority final-panel and review-required column reducers retain
their column-oriented inputs and only return final output rows.

The reproducible probe is
`src_rust_migration/refinitiv_column_native_probe.py`. On the local
`full_data_run` artifacts it passed parity for full Step 1 resolution,
full ownership-universe handoff, and a deterministic 750-PERMNO authority
subset against both Python fallback and persisted Step 1 outputs. The guarded
main probe completed under
`src_rust_migration/benchmark_results/watchguard_main_subset750_limit16`
(`peak_private_gb=12.356`); the default 8 GB limit was too low for the combined
full-artifact load. Isolated candidate-metrics watchguard probes on
2,000 PERMNOs / 705,075 ownership rows completed under the 8 GB limit and show
the Rust column path lowering peak private memory from 6.463 GB to 5.346 GB
versus Python fallback, while wall time remained slower for that isolated
candidate reducer. The same main probe showed the full authority subset
slightly faster end-to-end in Rust (15.987s versus 17.157s). The latest gates
passed: `cargo fmt --check`, `cargo check`, `python setup.py build_ext
--inplace`, focused Refinitiv acceleration tests (`12 passed, 926 deselected`),
and the full migration parity suite (`938 passed`).

The migration copy now also exposes repo-internal domain package boundaries:
`thesis_core`, `thesis_sec`, `thesis_refinitiv`, `thesis_lm2011`, and
`thesis_native`. The Rust extension is built at
`thesis_native._lm2011_rust`, while `thesis_pkg.core.sec` keeps a
compatibility alias for legacy extension imports. Reusable LSEG provider,
batching, ledger, stage-audit, and API helper surfaces are grouped under
`thesis_refinitiv.lseg_client`. The Refinitiv bridge implementation is now
owned by `thesis_refinitiv.bridge`, with
`thesis_pkg.pipelines.refinitiv_bridge_pipeline` kept as a `sys.modules`
compatibility alias so legacy public/private imports, monkeypatch identity,
Rust metrics, and fallback behavior are preserved. Other thesis-specific
Refinitiv modules are still being moved incrementally. See
`DOMAIN_PACKAGE_MIGRATION.md` for the current boundary state and future
standalone LSEG API package extraction note.

The final migration-readiness cleanup also confirmed that the narrower
`thesis_pkg.pipelines.refinitiv.bridge` module aliases `thesis_refinitiv.bridge`
directly, internal imports for moved bridge and reusable LSEG owners point at
the new domain boundaries, and no stale Python-loadable `_lm2011_rust*.pyd`
exists outside `thesis_native`. Cargo target `_lm2011_rust.dll` outputs are
build-cache artifacts, and `thesis_pkg.core.sec.extraction_fast` remains a
separate Cython SEC extraction accelerator.

A post-cleanup guarded sample comparison was run with
`tools/run_with_memory_watchguard.ps1` and
`src_rust_migration/refinitiv_column_native_probe.py --skip-full-candidate`
against `full_data_run/refinitiv_step1` plus
`full_data_run/sample_5pct_seed42`. The probe now defers loading the largest
ownership-result parquet until after bridge comparisons, and in
`--skip-full-candidate` mode scans only the selected authority subset. The run
completed under
`src_rust_migration/benchmark_results/watchguard_migration_readiness_sample_20260514_skipfull_streamed`
with `peak_private_gb=4.236`, matching Rust versus Python and persisted
artifacts for full Step 1 resolution, full ownership-universe handoff, and the
750-PERMNO authority subset.

## Known Gaps

- The migration is still isolated to `src_rust_migration`; the original `src`
  package tree is intentionally untouched.
- `thesis_refinitiv.authority`, `thesis_refinitiv.ownership`,
  `thesis_core.status`, `thesis_sec`, and `thesis_lm2011` are intentionally
  deferred facades or placeholders in this readiness snapshot. Moving them to
  primary ownership requires broader runner/package-export work and is tracked
  in `DOMAIN_PACKAGE_MIGRATION.md`.
- Full SEC item-boundary extraction remains Python/Cython-backed. Changes to
  extraction scanners or boundary selection still require the
  suspicious-boundary diagnostics workflow.
- Some non-target Refinitiv bridge reducers still keep row-dict compatibility
  helpers for validation/case diagnostics and other lower-priority surfaces.
- Candidate-metrics wall time is still slower in isolation because the current
  Python wrapper must convert Polars columns into Python lists and convert
  compact observation payloads back into Python metadata; the current win is
  lower peak private memory on the authority candidate-metrics path.

# Rust Migration Audit

## Objective Checklist

- Copy `src` in the repository root first.
  - Evidence: root-level `src_rust_migration/` exists and contains the copied
    `thesis_pkg` package tree.
  - Current state: original `src/` has no tracked edits from this migration.

- Work in the copied folder.
  - Evidence: Rust build files and modified Python modules are under
    `src_rust_migration/`.
  - Evidence: import-only smoke checks confirmed copied runners resolve `SRC`
    to `src_rust_migration`.
  - Current state: the only test file added outside the copy is
    `tests/test_lm2011_rust_accel.py`, which forces imports from
    `src_rust_migration`.

- Migrate Python hot paths to Rust for performance.
  - Implemented:
    - LM2011 tokenization and dictionary counting.
    - LM2011 dictionary-token normalization for dictionary/list inputs.
    - LM2011 dictionary materialization helpers for cell normalization,
      active extended-dictionary membership checks, and order-preserving
      word-list de-duplication.
    - LM2011 document-stat preparation for uncleaned and Python-cleaned text
      feature batches, including column-oriented handoff, metadata row
      projection, token counts, recognized-token counts, matched-token counts,
      and IDF inputs.
    - LM2011 streaming text-feature pass-1 row construction for uncleaned
      and Python-cleaned text batches, including column-oriented handoff,
      compact matched-count JSON payloads, and document frequency inputs.
    - LM2011 streaming text-feature pass-2 feature-row construction from
      pass-1 shards, including column-oriented handoff, signal proportions,
      TF-IDF weights, and cleaning-policy metadata.
    - LM2011 streaming writer incremental `doc_id` duplicate validation for
      staged text-source shards.
    - LM2011 batch token-count expression for Polars text columns, preserving
      null propagation while batching non-null text through Rust.
    - LM2011 count-only token counting for visible-prefix aggregation paths.
    - FinBERT visible-prefix LM2011 token-count batches for original and
      retained sentence text.
    - FinBERT bucket-length tuning summary row assembly for fixed bucket
      ordering and empty-bucket fill rows, including a column-oriented handoff
      before the row-dict compatibility bridge.
    - LM2011 eager text-feature microbatch span planning.
    - LM2011 text feature-row construction for token totals,
      recognized-token totals, signal proportions, and paper-formula TF-IDF
      weights.
    - LM2011 SEC/CCM form normalization.
    - LM2011 SEC/CCM batch form normalization helpers used by LazyFrame
      `map_batches`, replacing row-wise Python callbacks in sample-backbone
      form gates.
    - LM2011 text-feature base `normalized_form` construction now uses the
      Rust-backed batch form expression for eager and LazyFrame full-10-K/MD&A
      feature inputs instead of scalar `map_elements`.
    - LM2011 post-Refinitiv FinBERT-visible prefix source construction now
      uses the Rust-backed LM token-count batch expression instead of scalar
      `map_elements` for original and visible-prefix document text counts.
    - LM2011 post-Refinitiv extension CSV companions now batch JSON
      serialization for signal-input list columns through the Rust extension
      with Python fallback, replacing the final scalar `map_elements`
      callback in the copied package tree.
    - LM2011 FF48 SIC mapping text parsing for cached industry-range lookup.
    - CCM daily/SEC bridge form-match token normalization used before
      filtering filings by keep-form sets.
    - SEC filing identifier utility helpers for accession digit normalization,
      10-digit CIK formatting, and canonical `doc_id` construction.
    - SEC utility newline normalization, narrow date parsing, Roman numeral
      parsing, default item-part inference, and bullet-prefix classification
      used by filing cleanup, regime lookup, extraction date handling, and
      item metadata/boundary heuristics.
    - SEC extraction EDGAR metadata stripping for ASCII filing bodies,
      including complete metadata block removal, synthetic leading-header
      suffix stripping, truncated leading-header guards, and non-ASCII Python
      fallback.
    - SEC extraction Part-marker scanning for 10-K/10-Q part heading
      detection, including sparse-layout midline detection, 10-Q form-header
      rescue, TOC-mask skipping, ordered Part I/II filtering, and fallback
      through the existing native/Python chain.
    - SEC extraction high-confidence item-text truncation for empty-section
      stubs, successor Item headings, and confirmed later-Part restarts while
      preserving cross-reference and TOC-window guards.
    - SEC source filename parsing for raw filing metadata extracted from
      archive member names.
    - SEC pipeline completed-year checkpoint JSON decoding for per-year batch
      merge resume state.
    - SEC pipeline no-item stats weighted aggregation for full-sample
      document-type and total rows, including a column-oriented Rust handoff
      while keeping CSV I/O in Python.
    - Parquet I/O stream-copy and quick magic-byte probing used by verified
      artifact promotion and staged concat validation while keeping PyArrow
      quick/full validation in Python.
    - SEC Parquet filing-text stream batch matching by `doc_id`, used by
      multi-surface and suspicious-boundary diagnostics to fetch escalated
      full filings without materializing unmatched `full_text` values.
    - SEC filing boundary-authority status classification and public boundary
      payload projection used when converting extraction diagnostics into
      stable item-boundary metadata.
    - SEC regime form-type normalization.
    - SEC HTML-audit scalar rendering helpers for safe slugs, boolean/integer
      parsing, item ordering, and quartile bucketing.
    - SEC HTML-audit filing status classification for pass/warning/fail
      report grouping and sampling.
    - SEC HTML-audit deterministic status/stratified filing sampling,
      including weighted pass/warning/fail allocation, item-count quartile
      strata, missing core-item strata, and Python fallback with exact seeded
      parity.
    - Sentence split quality classifiers.
    - Sentence split quality residual-fragment classifiers.
    - Sentence split quality generic-reference continuation classifier.
    - Sentence dataset chunk-row expansion for long cleaned item scopes,
      including split-audit row construction.
    - Sentence split quality batch flag classification for short/very-short,
      one-word, numeric-only, separator-only, citation stubs,
      generic-reference endings/continuations, citation-prefix-only lines,
      header-like lines, table-like, lowercase-fragment, and
      missing-terminal-punctuation flags.
    - Sentence split quality report item-code ordering for metadata payloads.
    - Item-text cleaning layout classifiers.
    - Item-text cleaning table-like numeric row detection.
    - Item-text cleaning table header/context row detection.
    - Item-text cleaning full-text newline normalization.
    - Item-text cleaning blank-run collapse after newline normalization and
      table filtering.
    - Item-text cleaning early TOC-like line detection.
    - Item-text cleaning early TOC-prefix trimming.
    - Item-text cleaning item-aware tail-bleed marker scanning.
    - Item-text cleaning reference-only stub detection.
    - Item-text cleaning drop-reason and manual-audit reason construction for
      cleaned-scope row audit payloads.
    - Item-text cleaning LM2011 token-count batches for cleaned scope audit
      and drop-rule inputs.
    - Item-text cleaning row-audit base payload, review-status, and
      production eligibility helpers used by cleaned-scope audit assembly.
    - Item-text cleaning batch audit/cleaned-row finalization for cleaned
      scope artifacts, including drop/review/production metadata and audit
      snippets.
    - Item 7 LM token floor-sweep threshold summary-row reduction, including
      dropped-row totals, floor/reference-stub counts, and confirmed
      false-positive share metrics.
    - Item-text cleaning benchmark item-code to text-scope mapping, activation
      status, and audit-period labeling.
    - Item-text cleaning layout-only line removal for page markers, running
      report headers, and structural residue lines.
    - Item-text cleaning table-like block removal, including table headers,
      support lines, single-line-with-header handling, and header-context
      drops.
    - Item-text cleaning fused single-text cleaning orchestration for FinBERT
      cleaned scope preprocessing, including newline normalization,
      layout-line removal, TOC-prefix trimming, item-aware tail truncation,
      reference-stub detection, table-like block removal, blank-run collapse,
      and non-body text detection in one Rust call with Python fallback.
    - Item-text cleaning cleaned-scope preparation for FinBERT preprocessing,
      including item-code scope normalization, source newline normalization,
      fused cleaning dispatch, character-count/removal-ratio calculation, and
      prepared-row payload assembly before LM token-count and finalization
      batches.
    - Item-text cleaning manual-audit sample selection, including deterministic
      per-scope/period caps, manual-audit-priority ordering, sample-reason
      projection, and manual-review placeholder columns.
    - Item-text cleaning scope-diagnostics/manual-audit batch classification
      for activation status and calendar-year audit period, replacing
      remaining `map_elements` scalar callbacks in those dataframe assembly
      paths.
    - Item-text cleaning manual-audit fallback sampling now also uses the
      batch audit-period expression, removing the fallback-only scalar
      `map_elements` callback while preserving output parity.
    - FinBERT section-universe document-type normalization through
      Rust-backed SEC raw-form and canonical LM2011 form batch expressions.
    - FinBERT sentence provenance benchmark item-code to text-scope
      normalization through a Rust-backed batch expression.
    - LM2011 Refinitiv document-ownership `KYPERMNO` normalization.
    - LM2011 Refinitiv document-ownership date normalization.
    - LM2011 Refinitiv document-ownership target quarter-end derivation used
      by ownership request construction.
    - LM2011 Refinitiv document-ownership target-effective-date and
      request-bound date-window clipping used by ownership request
      construction.
    - LM2011 Refinitiv document-ownership request row construction, including
      filing-date target windows, authority exception matching, request-bound
      clipping, and exclusion reasons.
    - LM2011 Refinitiv document-ownership universe-diagnostics summary
      reduction, including a column-oriented detail-frame handoff before the
      row-dict compatibility bridge, doc-set overlap counts, final-status
      counts, mismatch CIK top counts, and overlap rates.
    - LM2011 Refinitiv document-ownership membership-prep batch normalization
      for diagnostic `doc_id`, CIK, form/status text, and `KYPERMNO` columns.
    - LM2011 Refinitiv document-ownership doc-filing artifact reader
      `KYPERMNO` normalization, replacing the remaining row-wise
      `map_elements` callback with the existing Rust-backed batch normalizer
      and Python batch fallback.
    - LM2011 Refinitiv document-ownership retrieval-sheet matching used when
      parsing filled exact/fallback workbooks.
    - LM2011 Refinitiv document-ownership exact institutional hit selection
      used by fallback-request construction and final output assembly.
    - LM2011 Refinitiv document-ownership fallback institutional hit
      selection, including latest-date selection, fallback-window cutoff
      filtering, and conflict detection.
    - LM2011 Refinitiv document-ownership fallback-request row selection
      after exact-hit and exact-conflict screening.
    - Refinitiv ownership/analyst numeric response normalization and
      institutional ownership percentage capping.
    - Refinitiv analyst-estimate integer count normalization.
    - Refinitiv analyst actuals/estimates fiscal-period label normalization
      inside LSEG response assembly loops.
    - Refinitiv analyst month-end shift helper used by forecast revision
      cutoff selection.
    - Refinitiv analyst latest-date lookup helper used when selecting
      forecast snapshots on or before cutoff dates.
    - Refinitiv analyst estimate date-index freezing used before forecast
      snapshot lookup and revision-base selection.
    - Refinitiv analyst estimate snapshot canonicalization for
      duplicate-conflict detection and source request-group/RIC
      consolidation.
    - Refinitiv analyst actual event canonicalization for duplicate actual-EPS
      conflict detection and source request-group/RIC consolidation.
    - Refinitiv analyst normalized-event row construction, including selected
      forecast snapshot lookup, missing/nonunique fiscal-period derivation
      rejections, forecast revision calculations, and duplicate
      normalized-event conflict detection.
    - Refinitiv analyst source request-group list normalization used by actual
      event and estimate snapshot canonicalization.
    - Refinitiv analyst LSEG actuals and estimates request-item row
      construction from request-group windows.
    - Refinitiv analyst LSEG actuals and estimates batch response row
      normalization.
    - Refinitiv ownership LSEG universe, document-exact, and
      document-fallback request-item row construction from handoff/request
      frames.
    - Refinitiv ownership LSEG universe, document-exact, and
      document-fallback batch response row normalization.
    - Refinitiv LSEG document-recovery unresolved request masking, including
      duplicate returned `doc_id` handling, null returned-ID skipping, null
      request `doc_id` exclusion, retrieval-eligibility gating, and
      request-row order preservation.
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
    - Refinitiv analyst request-group ID hashing.
    - Refinitiv analyst batched request-group ID hashing for eligible
      authority rows in the request-group builder.
    - Refinitiv ownership authority scalar helpers for date-span counts,
      tolerance-based value matching, distinct ownership-value sorting,
      ownership candidate-key normalization, source-family labels, and
      component IDs.
    - Refinitiv ownership authority allowlist key normalization used by
      reviewed ticker authority decisions, including a column-oriented Rust
      handoff before the row-dict compatibility bridge.
    - Refinitiv ownership authority interval merging for date-window
      consolidation.
    - Refinitiv ownership authority candidate metric construction, including
      request/result grouping, ownership observation sets, per-PERMNO coverage
      shares, and downstream metadata payloads.
    - Refinitiv ownership authority pairwise alias diagnostics for
      candidate-RIC overlap, value-conflict, benign-alias, regime-split, and
      review flags.
    - Refinitiv ownership authority conventional-component grouping and
      metadata reduction, including a direct candidate/pair metadata handoff
      before the row-dict compatibility bridge, benign-alias union-find
      collapse, canonical member selection, bridge-window merging, ownership
      union maps, and review flags.
    - Refinitiv LSEG `stable_hash_id` hashing for scalar and simple nested
      JSON payloads used by request IDs, batch IDs, and plan fingerprints.
    - Refinitiv LSEG request-signature hashing for simple field/parameter
      payloads used by batch grouping and interval planners.
    - Refinitiv LSEG simple batch item grouping, item-id sorting, chunking,
      and unique-instrument-limit batching.
    - Refinitiv LSEG split-batch child metadata construction for retry
      subdivision.
    - Refinitiv LSEG error classification.
    - Refinitiv LSEG session-not-opened message detection used by provider
      retry decisions.
    - Refinitiv LSEG identifier-list parsing for unresolved-identifier
      diagnostics.
    - Refinitiv LSEG unresolved-identifier message parsing for direct provider
      helper calls.
    - Refinitiv LSEG provider integer coercion for HTTP metadata.
    - Refinitiv LSEG provider UTF-8 record coercion used by the
      pandas-to-Polars fallback path.
    - Refinitiv LSEG provider response-header sanitization for metadata
      capture.
    - Refinitiv LSEG provider response-frame fingerprint hashing for API
      response metadata, including a column-oriented Rust handoff before the
      row-dict compatibility bridge.
    - Refinitiv LSEG API execution response row-count aggregation by
      `item_id`.
    - Refinitiv LSEG API execution response row-count aggregation now passes
      plain `item_id` value lists to Rust instead of one-column row
      dictionaries.
    - Refinitiv LSEG API execution item-result detail row construction for
      per-item success/requeue diagnostics.
    - Refinitiv LSEG API execution mixed zero/positive ownership-universe
      requeue decision.
    - Refinitiv LSEG API execution singleton and split child-batch row
      planning for retry subdivision.
    - Refinitiv LSEG batch retry overload classification.
    - Refinitiv LSEG lookup request-item row construction from lookup
      snapshots.
    - Refinitiv LSEG lookup batch response normalization.
    - Refinitiv LSEG stage-audit metadata scalar/list/mapping normalization.
    - Refinitiv LSEG daily-limit header/message detection used by API retry
      and stage-defer policy.
    - Refinitiv LSEG empty-result classification for final-attempt
      unresolved identifier handling.
    - Refinitiv LSEG deterministic batch error-policy classification for
      unresolved identifiers, daily limits, session failures, overloads,
      fatal exceptions, and max-attempt exhaustion.
    - LM2011 event-panel previous-month-end date bucketing and quarterly
      Fama-MacBeth quarter-start bucketing.
    - LM2011 previous-month-end and quarter-start batch date expressions used
      by event-panel price joins, Phase 0 validation SUE diagnostics, and
      quarterly Fama-MacBeth grouping.
    - LM2011 quarterly Fama-MacBeth weighted-mean and Newey-West
      standard-error numeric helpers.
    - LM2011 event-window scalar validation and derived event/post-event day
      offsets used by event-panel builders.
    - LM2011 fixed-shape FF3 OLS alpha/RMSE calculation used by event-window
      regression metric builders.
    - LM2011 event-window per-document regression metric row construction
      with column-oriented handoff.
    - LM2011 event-window daily row expansion for event-screen and
      event-return panel builders, replacing the eager range join with a Rust
      row-index pair selector while preserving Polars schema assembly and
      downstream semantics.
    - LM2011 fixed-shape FF4 OLS coefficient, standard-error, t-statistic,
      and R2 calculation used by monthly strategy factor-loading summaries.
    - LM2011 monthly strategy factor-loading row grouping, column-oriented
      preparation, and summary-row construction.
    - LM2011 regression-output finite-float sanitizer used before writing
      model fit diagnostics.
    - LM2011 Phase 0 validation-audit legacy Appendix-style tokenization.
    - LM2011 Phase 0 validation-audit marker/snippet scanning, whitespace
      truncation, truthy-row counting, and series counting used by packet
      summaries.
    - LM2011 Phase 0 validation-audit raw form normalization/count aggregation
      used by broad year-merged corpus distribution checks.
    - LM2011 Phase 0 validation-audit form-count row finalization used by
      corpus form distribution comparisons.
    - LM2011 Phase 0 validation-audit representative-term document count
      updates used by corpus coverage comparisons.
    - LM2011 Phase 0 validation-audit Packet B event-panel attrition row
      aggregation used by backbone-to-event coverage checks.
    - LM2011 Phase 0 validation-audit Packet C unit-audit row construction for
      non-null counts and mean absolute value checks.
    - LM2011 Phase 0 validation-audit Packet D coverage-reconciliation row
      construction for reported/actual FinBERT backbone denominator
      diagnostics.
    - LM2011 Phase 0 validation-audit Packet A MD&A row payload construction,
      including current/Appendix token counts, recognized-word counts,
      threshold flags, and audit snippets.
    - LM2011 Phase 0 validation-audit Packet A token-delta summary aggregation
      by full-10-K and MD&A scope.
    - LM2011 Phase 0 validation-audit Packet A summary-frame per-year and
      overall aggregation for marker, threshold, and cleaning diagnostics.
    - LM2011 Phase 0 validation-audit Packet A strip-comparison per-year and
      overall aggregation for residual marker and truncation diagnostics.
    - LM2011 Phase 0 validation-audit Packet A example-row filtering and
      priority ordering for manual audit excerpts.
    - LM2011 extension text-scope alias normalization.
    - LM2011 extension normal-approximation two-sided p-value calculation.
    - LM2011 extension result-row conversion for quarterly regression outputs,
      including `n_obs` derivation and p-value calculation, now with a
      column-oriented handoff before the row-dict compatibility bridge.
    - LM2011 extension quarterly fit-row construction for common-sample
      fit-comparison diagnostics, including a column-oriented handoff before
      the row-dict compatibility bridge.
    - LM2011 extension skipped-quarter diagnostic row construction for
      fit-comparison artifacts, including a column-oriented handoff before the
      row-dict compatibility bridge.
    - LM2011 extension quarterly fit-difference row construction for
      common-sample specification comparisons.
    - LM2011 extension common-quarter fit-summary aggregation for
      fit-comparison diagnostics.
    - LM2011 extension common-quarter fit-comparison aggregation, including
      weighted/equal R2 deltas, Newey-West SEs, t-statistics, and p-values.
    - FinBERT robustness-runner fit-comparison row assembly now delegates to
      the Rust-backed LM2011 extension quarterly-fit, skipped-quarter,
      summary, difference, and comparison-row helpers.
    - LM2011 Table IA.II monthly strategy result-row expansion from FF4
      summary rows, including column-oriented handoff.
    - LM2011 quarterly Fama-MacBeth final result-row aggregation from retained
      quarter coefficient time series.
    - LM2011 cross-sectional OLS design-row and column-oriented preparation
      for quarterly Fama-MacBeth regressions.
    - Refinitiv bridge workbook scalar normalization for workbook headers and
      row values.
    - Refinitiv bridge extended-workbook boolean cell normalization.
    - Refinitiv bridge lookup-text normalization.
    - Refinitiv bridge lookup-result filtering, normalized lookup-result
      matching, and bridge-row LIID parsing.
    - Refinitiv bridge ownership-result date normalization for filled
      ownership validation workbook parsing.
    - Refinitiv bridge ownership-universe request-date and date-text
      rendering used by handoff and review-row assembly.
    - Refinitiv bridge identity-candidate agreement checks used by RIC
      resolution conflict and extension policy.
    - Refinitiv bridge material identity-conflict checks for ISIN/CUSIP
      mismatch, RIC-only disagreement, and partial-identity ticker
      diagnostics.
    - Refinitiv Excel ownership-universe retrieval block payload construction,
      including repeated-block input values, cell references, and Workspace
      ownership formula strings while keeping workbook writing in Python.
    - Refinitiv Excel LM2011 document-ownership retrieval block payload
      construction, including exact/fallback formula strings, repeated-block
      cell references, and date-preserving input values while keeping workbook
      writing in Python.
    - Refinitiv Excel resolution-diagnostic retrieval block payload
      construction, including repeated-block input values, `_xll.RDP.Data`
      formula strings, and diagnostic lookup date references while keeping
      workbook writing in Python.
    - Refinitiv Excel ownership-smoke retrieval block payload construction,
      including repeated-block smoke-test input values while keeping workbook
      writing in Python.
    - Refinitiv Excel ownership-validation sheet payload construction,
      including case/sheet grouping, repeated-slot input values, and Workspace
      ownership formula strings while keeping workbook writing in Python.
    - Refinitiv Excel extended RIC lookup summary formula payload
      construction, including lookup-range COUNTIF/SUMPRODUCT formulas while
      keeping workbook writing in Python.
    - Refinitiv Excel extended RIC lookup formula payload construction,
      including per-identifier lookup, success, pairwise identity, and
      all-successful consistency formulas while keeping workbook writing in
      Python.
    - Refinitiv bridge accepted-source matching and adjacent-extension source
      choice used by RIC extension policy.
    - Refinitiv bridge resolution-diagnostic classifiers for target class,
      conventional-source detection, support scope, and block-reason labels.
    - Refinitiv bridge summary reducers for normalized value counts and
      strict `True` record counts used by diagnostic manifests.
    - Refinitiv bridge batched strict-`True` field counting used by
      resolution diagnostic manifests.
    - Refinitiv bridge resolution-diagnostic class-summary reducer for
      grouped target diagnostics and extension support/block counts.
    - Refinitiv bridge ownership-validation retrieval-role sort key used when
      ordering validation workbook blocks and retrieval summaries.
    - Refinitiv bridge ownership-validation pairwise result comparison,
      including overlap counts, returned-RIC/category agreement,
      value-difference summaries, corroboration, same-identity RIC-variant
      support, and conflict flags.
    - Refinitiv bridge ownership-validation case-summary row construction,
      including candidate/adjacent ownership-data flags, pair-support counts,
      and validation-bucket assignment.
    - Refinitiv bridge ownership-universe handoff row construction, including
      effective-RIC, conventional-conflict, ticker-fallback, and
      nonretrievable branches.
    - Refinitiv bridge failed-lookup classification used by null-RIC rescue
      diagnostics.
    - Refinitiv bridge batched failed-lookup classification used by null-RIC
      rescue candidate construction.
    - Refinitiv bridge ownership-universe candidate-key normalization used
      when matching filled ownership retrieval workbooks back to handoff rows.
    - Refinitiv bridge ownership-smoke lookup-input priority resolution used
      when building null-RIC rescue review samples.
    - Refinitiv bridge ownership-smoke sample-row projection, including
      normalized request fields, date text rendering, truthy flags, and count
      coercion for null-RIC rescue review samples.
    - Refinitiv bridge alternative-identifier selection used by null-RIC
      rescue diagnostics.
    - Refinitiv bridge batched alternative-identifier selection used by
      null-RIC rescue candidate construction.
    - Refinitiv bridge ownership-validation result text and numeric value
      normalization.
    - Multi-surface audit whitespace normalization.
    - Multi-surface audit boundary snippet-risk detection.
    - Multi-surface audit stable integer sort-key fast path for digit-bearing
      case and sentence identifiers.
    - Multi-surface audit ASCII normalized-text index construction for
      full-report context matching.
    - Multi-surface audit ASCII normalized match-bound lookup for full-report
      context matching.
    - Multi-surface audit escalation scoring, cap enforcement, stable
      ordering, and full-report review reason projection for audit-pack case
      rows.
    - Multi-surface audit review-record interleaving and chunk index planning
      for audit-pack chunk files.
    - FinBERT sentence confusion-review sample row ID and order assignment
      before neighbor-context attachment and labeling chunk export, including
      a column-oriented handoff from the sorted sample frame before the
      row-dict compatibility bridge.
    - FinBERT sentence confusion-review neighbor-target row construction for
      previous/next sentence context joins.
    - FinBERT sentence confusion-review allocation row finalization, including
      sample weights, population fractions, and allocation-mode labels.
    - FinBERT sentence confusion-review labeling JSONL payload row
      construction for human review and LLM pass chunks.
    - FinBERT sentence confusion-review round-robin chunk index planning for
      review and LLM-pass JSONL shards.
    - FinBERT sentence confusion-review CSV-safe reviewed-row export
      preparation for nested list/dict values.
    - FinBERT sentence confusion-review reviewed-case row construction,
      including review-label joins, final gold-label selection, and
      confusion-cell assignment.
    - FinBERT sentence confusion-review examples-by-cell markdown rendering
      for per-cell reviewed sentence examples.
    - FinBERT high-confidence sentence-example sample-candidate row
      finalization, including item/sentiment ordering, candidate sorting, and
      filing-date export normalization.
    - FinBERT high-confidence sentence-example item/sentiment and
      year/item/sentiment count-row finalization from accumulator
      dictionaries.
    - FinBERT high-confidence sentence-example markdown report rendering for
      ordered item/sentiment sample sections, now with a column-oriented
      handoff before the row-dict compatibility bridge.
    - Manifest-contract stable string fingerprint normalization, including
      iterable handling, null skipping, de-duplication, sorting, and SHA-256
      hashing.
    - Manifest-contract semantic reuse-guard mismatch detection for version,
      payload, and fingerprint drift checks.
    - FinBERT token-bucket batch assignment used after token-length
      annotation.
    - Refinitiv bridge resolution summary scalar counts for accepted,
      extended, effective, unresolved, blocked, ticker-candidate, and
      identity-conflict rows.
    - Refinitiv bridge resolution-diagnostic class-summary reducer for
      grouped target diagnostics and extension support/block counts.
    - Refinitiv bridge batched strict-`True` field counts for resolution
      diagnostic target, candidate-match, and identity-match summaries.
    - Refinitiv bridge ownership-validation case-summary row construction for
      candidate data flags, adjacent support flags, pair counts, and bucket
      labels.
    - Refinitiv bridge ownership-universe handoff row construction for
      universe-effective, ISIN/CUSIP conflict candidate, ticker fallback, and
      nonretrievable cases.
    - Refinitiv bridge batched failed-lookup classification for null-RIC
      rescue candidate construction.
    - Refinitiv bridge ownership-smoke sample-row projection for normalized
      lookup fields, request dates, truthy flags, and count fields.
    - Refinitiv bridge batched alternative-identifier selection for null-RIC
      rescue candidate construction.
    - LM2011 paper-style full-10-K exhibit-tail detection.
    - FinBERT sentence postprocessing classifiers used for reference stitching
      and artifact-line cleanup.
    - FinBERT sentence postprocessing shared sentence-key whitespace
      normalization.
    - FinBERT sentence postprocessing list transformation for artifact cleanup
      and reference-stitch policy application.
    - FinBERT token-bucket assignment used after token length annotation.
    - FinBERT visible-prefix retained-end calculation from fast-tokenizer
      offsets.
    - FinBERT model-label normalization.
    - FinBERT benchmark fallback softmax row and batch probability calculation
      for logits when native Torch softmax is unavailable.
    - FinBERT benchmark per-sentence probability-column and predicted-label
      construction after model logits are converted to probabilities.
    - FinBERT benchmark median timing summary helper used by benchmark and
      staged inference summaries.
    - FinBERT benchmark stage-summary aggregation for run-summary row,
      duration, throughput, and peak-VRAM reporting.
    - FinBERT benchmark bucket max-length selection used by benchmark and
      staged inference batch loops.
    - FinBERT benchmark CUDA device-index parsing used by runtime environment
      summaries.
    - FinBERT benchmark AMP dtype resolution used by runtime/autocast setup.
    - FinBERT staged-inference tokenizer bucket-summary aggregation for
      per-year/per-bucket row counts and token-count mean/median statistics.
    - FinBERT item-analysis coverage-report reduction for
      per-backbone-document item coverage flags and summary counts.
    - FinBERT sentence-preprocessing token-bucket count reduction used by
      yearly preprocessing summary rows.
    - FinBERT sentence-preprocessing manifest count aggregation for processed,
      reused, sentence, oversize, chunking, and cleaning row totals, including
      a column-oriented Rust handoff before the row-dict compatibility bridge.
    - FinBERT sentence-preprocessing split-audit metric reduction for chunked
      section counts, warning-boundary counts, and max original character
      length.
    - FinBERT sentence-preprocessing fallback split-boundary warning payload
      reduction for affected section counts and sorted split-reason counts.
    - FinBERT sentence-example sample-text normalization for ASCII rows.
    - FinBERT sentence-length report item-code ordering and JSON-ready record
      serialization for summary payloads, including a column-oriented
      frame-record Rust handoff before the row-dict compatibility bridge.
    - FinBERT shared contract year-filter normalization for run configuration
      dataclasses.
    - FinBERT bucket-length tuning item-code/year filter normalization and
      sentence-example item-code filter normalization.
    - FinBERT bucket-length tuning short/medium/long scalar selection and
      round-up-to-multiple recommendation helpers.
    - FinBERT sentence-example deterministic BLAKE2b sample-key calculation.
    - FinBERT sentence-example batched BLAKE2b sample-key calculation and
      ASCII sample-text normalization used before per-group candidate
      reduction.
    - FinBERT sentence-example per-group candidate replacement and
      de-duplication during deterministic sample accumulation.
    - FinBERT sentence-example batch accumulator updates for count
      dictionaries, document-id sets, and deterministic sample selections.
    - FinBERT sentence-example item-code ordering for candidate count/sample
      report metadata.
    - FinBERT sentence-confusion review binary-label normalization.
    - FinBERT sentence-confusion review deterministic sample seed derivation
      from SHA-256 prefixes.
    - FinBERT sentence-confusion review candidate-threshold and confusion-cell
      helpers used by exact sampling and reviewed-label metrics.
    - FinBERT sentence-confusion review metric-payload calculation for
      reviewed-label accuracy, precision, recall, specificity, and error
      rates.
    - FinBERT sentence-confusion review confusion-cell weighted/unweighted
      count reduction and uncertain-row metric-bound calculation used in
      review summaries.
    - FinBERT sentence-confusion review balanced and proportional sample-count
      allocation for deterministic stratum review sampling.
    - FinBERT sentence-confusion review streaming sample target-position
      generation, including a column-oriented Rust handoff before the row-dict
      compatibility bridge.
    - FinBERT sentence-confusion review majority-bucket metric row aggregation
      used by review summary CSVs, including a column-oriented Rust handoff
      before the row-dict compatibility bridge.
    - FinBERT benchmark dataset SHA-256 selection-key and selected-text
      hashing.
    - FinBERT benchmark constrained Hamilton apportionment used by year and
      year-item sample allocation.
    - FinBERT benchmark dataset year-level allocation row construction,
      including capacity caps and selected/eligible share columns.
    - FinBERT benchmark dataset year-item allocation orchestration,
      capacity-map reducers, and positive allocation-target extraction used by
      sample planning/selection.
    - FinBERT benchmark dataset audit-share row construction for selected
      versus eligible universe summaries after Polars grouping and sorting,
      now with a column-oriented handoff before the row-dict compatibility
      bridge.
    - FinBERT benchmark dataset ranked section selection by per-stratum quotas
      after deterministic selection-key sorting.
    - Sample item-cleaning sentence diagnostics doc-id sampler, preserving
      item-scope coverage seeding and deterministic fallback fill ordering.
    - Manifest canonical JSON SHA-256 and stable string-set fingerprint
      hashing.
    - Manifest semantic file SHA-256 fingerprinting with a Rust streaming
      hasher.
    - FinBERT tail-feature text-scope alias normalization.
    - FinBERT tail document-surface eager row reduction for materialized
      sentence-score frames, with the Polars LazyFrame builder retained for
      large-data pipeline use, now with a column-oriented Rust handoff before
      the row-dict compatibility bridge.
    - Multi-surface audit boundary-snippet risk and snippet-delta risk batch
      construction used by item-boundary case scoring, replacing scalar struct
      `map_elements` callbacks while preserving Python fallback behavior.
    - Multi-surface audit control-case peer-group membership now uses native
      Polars struct membership instead of a scalar `map_elements` callback.
    - Multi-surface audit deterministic stable sort-key batch construction
      used by case sampling, replacing the scalar `map_elements` callback
      while preserving nullable sort-key behavior and Python fallback.
    - FinBERT sentence chunk-end selection for oversized section chunking.
    - Copied runner bootstrap paths now prefer the `src_rust_migration`
      package tree when scripts are executed from the migration copy.
    - SEC/CCM pre-merge markdown scalar value formatting and table rendering
      for bounded run reports, preserving float/date formatting, pipe
      escaping, and truncation notes.
    - Refinitiv ownership LSEG request-log event counting for fetch manifests
      and daily-limit diagnostics.
    - Refinitiv LSEG ledger item-ID JSON decoding for completed-batch state
      updates.
    - Refinitiv LSEG ledger string-array JSON decoding for item and batch
      field lists.
    - Refinitiv LSEG stage-audit item-ID JSON decoding for legacy backfill
      tolerance checks.
  - Evidence:
    - Rust source: `rust/lm2011_rust/src/lib.rs`.
    - PyO3 extension module: `thesis_native._lm2011_rust`.
    - Legacy extension compatibility shim:
      `thesis_pkg.core.sec._lm2011_rust`.
    - Python fallback wrappers in the corresponding `thesis_pkg` modules.
  - Gap: this is an incremental migration, not a full Python-codebase rewrite.

- Preserve Python fallback behavior.
  - Evidence: each migrated Python surface imports `_lm2011_rust`
    opportunistically and falls back to its prior Python implementation.
  - Evidence: `tests/test_lm2011_rust_accel.py` covers real-extension paths,
    Python fallback paths, and parity samples.

- Avoid large-sample tests due to memory concerns.
  - Evidence: verification used targeted unit tests and synthetic fixtures only.
  - Not run: full pipeline runners, large sample diagnostics, or broad
    end-to-end sample jobs.

## Verification Performed

From the repository root:

```powershell
& "$env:USERPROFILE\.cargo\bin\cargo.exe" fmt --manifest-path src_rust_migration\rust\lm2011_rust\Cargo.toml -- --check
& "$env:USERPROFILE\.cargo\bin\cargo.exe" check --manifest-path src_rust_migration\rust\lm2011_rust\Cargo.toml
```

From `src_rust_migration/`:

```powershell
$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
python setup.py build_ext --inplace
```

From the repository root:

```powershell
$env:PYTHONPATH = "src_rust_migration"
pytest tests\test_lm2011_rust_accel.py tests\test_lm2011_cleaning.py tests\test_lm2011_pipeline.py::test_tokenize_lm2011_text_matches_appendix_contract tests\test_lm2011_pipeline.py::test_lm2011_text_features_use_total_token_denominators_and_match_hyphenated_dictionary_entries tests\test_lm2011_pipeline.py::test_write_lm2011_text_features_full_10k_parquet_matches_eager_builder tests\test_lm2011_pipeline.py::test_write_lm2011_text_features_mda_parquet_matches_eager_builder
```

Latest result for the final command before the sentence-postprocessing slice:
`31 passed`.

Additional sentence-postprocessing verification:

```powershell
$env:PYTHONPATH = "src_rust_migration"
pytest tests\test_lm2011_rust_accel.py tests\test_finbert_sentence_dataset.py::test_postprocess_sentence_texts_merges_item7_reference_stub_chain tests\test_finbert_sentence_dataset.py::test_postprocess_sentence_texts_strips_leading_item7_artifact_lines tests\test_finbert_sentence_dataset.py::test_postprocess_sentence_texts_drops_pure_item7_header_artifacts tests\test_finbert_sentence_dataset.py::test_postprocess_sentence_texts_item7_v2_preserves_citation_prefix_lines tests\test_finbert_sentence_dataset.py::test_postprocess_sentence_texts_item7_v2_stitches_generic_statement_no_chain tests\test_finbert_sentence_dataset.py::test_postprocess_sentence_texts_item7_v2_skips_blank_artifacts_between_stitches tests\test_finbert_sentence_dataset.py::test_postprocess_sentence_texts_v3_stitches_item1_reference_continuation tests\test_finbert_sentence_dataset.py::test_postprocess_sentence_texts_v3_stitches_item1a_reference_continuation tests\test_finbert_sentence_dataset.py::test_postprocess_sentence_texts_v3_retains_item7_v2_behavior tests\test_finbert_sentence_dataset.py::test_postprocess_sentence_texts_v3_does_not_strip_non_item7_heading_lines tests\test_finbert_sentence_dataset.py::test_postprocess_sentence_texts_v3_leaves_numeric_enumerator_fragments_unchanged
```

Latest result for that command: `33 passed`.

Additional small checks run during the migration covered:

- `tests/test_item_text_cleaning.py`
- `tests/test_refinitiv_doc_ownership_pipeline.py`
- `tests/test_sentence_split_quality_assessment.py`

Final combined small regression command:

```powershell
$env:PYTHONPATH = "src_rust_migration"
pytest tests\test_lm2011_rust_accel.py tests\test_lm2011_cleaning.py tests\test_item_text_cleaning.py tests\test_refinitiv_doc_ownership_pipeline.py tests\test_sentence_split_quality_assessment.py tests\test_lm2011_pipeline.py::test_tokenize_lm2011_text_matches_appendix_contract tests\test_lm2011_pipeline.py::test_lm2011_text_features_use_total_token_denominators_and_match_hyphenated_dictionary_entries tests\test_lm2011_pipeline.py::test_write_lm2011_text_features_full_10k_parquet_matches_eager_builder tests\test_lm2011_pipeline.py::test_write_lm2011_text_features_mda_parquet_matches_eager_builder tests\test_finbert_sentence_dataset.py::test_postprocess_sentence_texts_merges_item7_reference_stub_chain tests\test_finbert_sentence_dataset.py::test_postprocess_sentence_texts_strips_leading_item7_artifact_lines tests\test_finbert_sentence_dataset.py::test_postprocess_sentence_texts_drops_pure_item7_header_artifacts tests\test_finbert_sentence_dataset.py::test_postprocess_sentence_texts_item7_v2_preserves_citation_prefix_lines tests\test_finbert_sentence_dataset.py::test_postprocess_sentence_texts_item7_v2_stitches_generic_statement_no_chain tests\test_finbert_sentence_dataset.py::test_postprocess_sentence_texts_item7_v2_skips_blank_artifacts_between_stitches tests\test_finbert_sentence_dataset.py::test_postprocess_sentence_texts_v3_stitches_item1_reference_continuation tests\test_finbert_sentence_dataset.py::test_postprocess_sentence_texts_v3_stitches_item1a_reference_continuation tests\test_finbert_sentence_dataset.py::test_postprocess_sentence_texts_v3_retains_item7_v2_behavior tests\test_finbert_sentence_dataset.py::test_postprocess_sentence_texts_v3_does_not_strip_non_item7_heading_lines tests\test_finbert_sentence_dataset.py::test_postprocess_sentence_texts_v3_leaves_numeric_enumerator_fragments_unchanged
```

Latest result for the combined command: `74 passed`.

Additional FinBERT token-bucket verification after the final Rust slice:

```powershell
$env:PYTHONPATH = "src_rust_migration"
pytest tests\test_lm2011_rust_accel.py
pytest tests\test_finbert_tokenization.py::test_annotate_finbert_token_lengths_uses_fixed_512_schema tests\test_finbert_tokenization.py::test_annotate_finbert_token_lengths_accepts_custom_bucket_edges tests\test_finbert_tokenization.py::test_annotate_finbert_token_lengths_in_batches_preserves_row_order tests\test_finbert_tokenization.py::test_visible_prefix_fast_offsets_preserve_uncapped_and_slice_at_cap_rows tests\test_finbert_tokenization.py::test_visible_prefix_decoded_fallback_records_policy
```

Latest results: `25 passed` and `5 passed`.

Additional item-cleaning table-like, table-context, and TOC-like row
verification:

```powershell
$env:PYTHONPATH = "src_rust_migration"
pytest tests\test_lm2011_rust_accel.py tests\test_item_text_cleaning.py
```

Latest result after adding table-like, table-context, TOC-like, item-aware
tail-truncation, reference-only stub, and FinBERT visible-prefix retained-end
classifiers plus Refinitiv document-ownership category helpers:
`67 passed`.

Additional SEC regime form-normalizer verification after the latest Rust slice:

```powershell
$env:PYTHONPATH = "src_rust_migration"
pytest tests\test_lm2011_rust_accel.py tests\test_regime_loader.py::test_normalize_form_type tests\test_item_text_cleaning.py tests\test_finbert_tokenization.py::test_visible_prefix_fast_offsets_preserve_uncapped_and_slice_at_cap_rows tests\test_finbert_tokenization.py::test_visible_prefix_decoded_fallback_records_policy tests\test_refinitiv_doc_ownership_pipeline.py::test_clean_institutional_value_handles_negative_and_cap tests\test_finbert_item_analysis.py::test_resolve_finbert_label_mapping_normalizes_model_config
```

Latest result after adding Refinitiv LSEG error classification, Refinitiv LSEG
batch retry overload classification, LM2011 date-bucketing helpers, and the
multi-surface stable sort-key helper, plus Refinitiv document-ownership date
normalization, sentence split residual-fragment classifiers, and FinBERT
sentence-example sample-text normalization, plus Refinitiv analyst integer
count normalization, multi-surface normalized-index construction, FinBERT
sentence-confusion review binary-label normalization, FinBERT benchmark
dataset hashing, LM2011 extension text-scope alias normalization, manifest
fingerprint hashing, FinBERT sentence-confusion deterministic seed hashing, and
FinBERT tail-feature text-scope alias normalization, plus item-cleaning
full-text newline normalization and sentence-split generic-reference
continuation classification, plus Refinitiv LSEG stage-audit metadata
normalization and LSEG provider integer coercion:
`154 passed`.

Latest lightweight gate after adding FinBERT bucket-length tuning item/year
filter normalization and sentence-example item-code filter normalization:
`165 passed`.

Latest lightweight gate after adding LM2011 Phase 0 validation-audit legacy
Appendix-style tokenization:
`169 passed`.

Focused Rust acceleration suite after adding Refinitiv LSEG identifier-list
parsing:
`126 passed`.

Expanded lightweight gate after adding Refinitiv LSEG identifier-list parsing:
`172 passed`.

Focused Rust acceleration suite after adding streaming manifest file SHA-256:
`128 passed`.

Expanded lightweight gate after adding streaming manifest file SHA-256:
`174 passed`.

Focused Rust acceleration plus sentence-example suite after adding the
FinBERT sentence-example BLAKE2b sample-key hasher:
`133 passed`.

Expanded lightweight gate after adding the FinBERT sentence-example BLAKE2b
sample-key hasher:
`177 passed`.

Focused Rust acceleration suite after adding Refinitiv analyst fiscal-period
label normalization:
`134 passed`.

Refinitiv analyst LSEG API synthetic suite after the same change:
`6 passed`.

Expanded lightweight gate after adding Refinitiv analyst fiscal-period label
normalization and including the analyst LSEG API synthetic suite:
`186 passed`.

Focused Rust acceleration suite after adding Refinitiv bridge workbook scalar
normalization:
`137 passed`.

Expanded lightweight gate after adding Refinitiv bridge workbook scalar
normalization:
`189 passed`.

Focused Rust acceleration suite after adding Refinitiv bridge extended-workbook
boolean cell normalization:
`140 passed`.

Expanded lightweight gate after adding Refinitiv bridge extended-workbook
boolean cell normalization:
`192 passed`.

Focused Rust acceleration suite after adding item-text blank-run collapse:
`143 passed`.

Expanded lightweight gate after adding item-text blank-run collapse:
`195 passed`.

Focused Rust acceleration suite after adding LM2011 text microbatch span
planning:
`146 passed`.

Expanded lightweight gate after adding LM2011 text microbatch span planning:
`198 passed`.

Focused Rust acceleration suite after adding FinBERT sentence chunk-end
selection:
`149 passed`.

Expanded lightweight gate after adding FinBERT sentence chunk-end selection and
including two sentence-chunking caller tests:
`203 passed`.

Focused Rust acceleration suite after adding item-text early TOC-prefix
trimming:
`152 passed`.

Expanded lightweight gate after adding item-text early TOC-prefix trimming:
`206 passed`.

Focused Rust acceleration suite after adding SEC filing identifier utility
helpers:
`155 passed`.

Expanded lightweight gate after adding SEC filing identifier utility helpers:
`209 passed`.

Focused Rust acceleration suite after adding Refinitiv ownership authority
scalar helpers:
`158 passed`.

Expanded lightweight gate after adding Refinitiv ownership authority scalar
helpers and including an authority-table assembly test:
`213 passed`.

Focused Rust acceleration suite after adding Refinitiv ownership authority
interval merging:
`161 passed`.

Expanded lightweight gate after adding Refinitiv ownership authority interval
merging:
`216 passed`.

Focused Rust acceleration suite after adding FinBERT benchmark fallback
softmax helpers:
`164 passed`.

Expanded lightweight gate after adding FinBERT benchmark fallback softmax
helpers:
`219 passed`.

Focused Rust acceleration suite after adding FinBERT sentence-confusion
sampling helpers:
`167 passed`.

Expanded lightweight gate after adding FinBERT sentence-confusion sampling
helpers:
`222 passed`.

Focused Rust acceleration suite after adding FinBERT sentence-confusion
metric-payload calculation:
`170 passed`.

Expanded lightweight gate after adding FinBERT sentence-confusion
metric-payload calculation:
`225 passed`.

Focused Rust acceleration suite after adding CCM transform form-match token
normalization:
`173 passed`.

Expanded lightweight gate after adding CCM transform form-match token
normalization:
`228 passed`.

Focused Rust acceleration suite after adding FinBERT shared contract
year-filter normalization:
`177 passed`.

Expanded lightweight gate after adding FinBERT shared contract year-filter
normalization:
`232 passed`.

Focused Rust acceleration suite after adding LM2011 event-window scalar
helpers:
`181 passed`.

Expanded lightweight gate after adding LM2011 event-window scalar helpers:
`236 passed`.

Focused Rust acceleration suite after adding LM2011 regression weighted-mean
and Newey-West helpers:
`185 passed`.

Expanded lightweight gate after adding LM2011 regression weighted-mean and
Newey-West helpers:
`240 passed`.

Focused Rust acceleration suite after adding the LM2011 extension
normal-approximation p-value helper:
`189 passed`.

Expanded lightweight gate after adding the LM2011 extension
normal-approximation p-value helper:
`244 passed`.

Focused Rust acceleration suite after adding LM2011 dictionary materialization
helpers:
`192 passed`.

LM2011 dictionary materialization fixture suite after the same change:
`2 passed`.

Expanded lightweight gate after adding LM2011 dictionary materialization
helpers and including the dictionary fixture suite:
`249 passed`.

Focused Rust acceleration suite after adding the LM2011 regression-output
finite-float sanitizer:
`195 passed`.

Expanded lightweight gate after adding the LM2011 regression-output
finite-float sanitizer:
`252 passed`.

Focused Rust acceleration suite after adding the FinBERT benchmark median
timing helper:
`198 passed`.

Expanded lightweight gate after adding the FinBERT benchmark median timing
helper:
`255 passed`.

Focused Rust acceleration suite after adding FinBERT bucket-length scalar
recommendation helpers:
`202 passed`.

Expanded lightweight gate after adding FinBERT bucket-length scalar
recommendation helpers:
`259 passed`.

Focused Rust acceleration suite after adding FinBERT benchmark bucket
max-length selection:
`206 passed`.

Expanded lightweight gate after adding FinBERT benchmark bucket max-length
selection:
`263 passed`.

Focused Rust acceleration suite after adding FinBERT benchmark CUDA
device-index parsing:
`209 passed`.

Expanded lightweight gate after adding FinBERT benchmark CUDA device-index
parsing:
`266 passed`.

Focused Rust acceleration suite after adding FinBERT benchmark AMP dtype
resolution:
`213 passed`.

Expanded lightweight gate after adding FinBERT benchmark AMP dtype resolution:
`270 passed`.

Focused Rust acceleration suite after adding SEC HTML-audit scalar helper
acceleration:
`216 passed`.

Small SEC HTML-audit fixture suite after the same change:
`3 passed`.

Expanded lightweight gate after adding SEC HTML-audit scalar helper
acceleration:
`273 passed`.

Focused Rust acceleration suite after adding Refinitiv LSEG request-signature
hashing:
`219 passed`.

Standalone Refinitiv LSEG batching fixture suite after the same change:
`8 passed`.

Expanded lightweight gate after adding Refinitiv LSEG request-signature
hashing:
`276 passed`.

Focused Rust acceleration suite after extending Refinitiv LSEG stable hashing
to simple nested JSON payloads:
`219 passed`.

Standalone Refinitiv LSEG batching fixture suite after the same nested-hash
change:
`8 passed`.

Expanded lightweight gate after the same nested-hash change:
`276 passed`.

Focused Rust acceleration suite after adding LM2011 Refinitiv
document-ownership target quarter-end derivation:
`222 passed`.

Standalone Refinitiv document-ownership fixture suite after the same change:
`13 passed`.

Expanded lightweight gate after adding LM2011 Refinitiv document-ownership
target quarter-end derivation:
`279 passed`.

Focused Rust acceleration suite after adding the Refinitiv analyst month-end
shift helper:
`225 passed`.

Standalone Refinitiv analyst fixture suite after the same change:
`13 passed`.

Expanded lightweight gate after adding the Refinitiv analyst month-end shift
helper:
`282 passed`.

Focused Rust acceleration suite after adding the Refinitiv analyst latest-date
lookup helper:
`228 passed`.

Standalone Refinitiv analyst fixture suite after the same change:
`13 passed`.

Expanded lightweight gate after adding the Refinitiv analyst latest-date lookup
helper:
`285 passed`.

Focused Rust acceleration suite after adding Refinitiv LSEG item-window date
parsing for analyst and ownership interval batching:
`234 passed`.

Targeted widened-interval consumer fixtures after the same change:
`1 passed` for `test_analyst_actuals_pipeline_filters_widened_batch_rows_per_item_window`
and `1 passed` for
`test_ownership_universe_api_pipeline_filters_widened_batch_rows_per_item_window`.

Expanded lightweight gate after adding Refinitiv LSEG item-window date parsing:
`291 passed`.

Focused Rust acceleration suite after adding Refinitiv LSEG interval span-day
calculation:
`237 passed`.

Standalone Refinitiv LSEG batching fixture suite after the same change:
`8 passed`.

Expanded lightweight gate after adding Refinitiv LSEG interval span-day
calculation:
`294 passed`.

Focused Rust acceleration suite after adding item-cleaning drop-reason and
manual-audit reason construction:
`240 passed`.

Standalone item-cleaning fixture suite after the same change:
`15 passed`.

Expanded lightweight gate after adding item-cleaning drop-reason and
manual-audit reason construction:
`297 passed`.

Focused Rust acceleration suite after adding Refinitiv bridge
identity-candidate agreement checks:
`243 passed`.

Targeted Refinitiv bridge resolution policy fixture after the same change:
`1 passed`.

Expanded lightweight gate after adding Refinitiv bridge identity-candidate
agreement checks:
`300 passed`.

Focused Rust acceleration suite after adding Refinitiv bridge failed-lookup
classification:
`246 passed`.

Standalone Refinitiv bridge pipeline fixture suite after the same change:
`13 passed`.

Expanded lightweight gate after adding Refinitiv bridge failed-lookup
classification:
`303 passed`.

Focused Rust acceleration suite after adding Refinitiv bridge
alternative-identifier selection:
`249 passed`.

Standalone Refinitiv bridge pipeline fixture suite after the same change:
`13 passed`.

Expanded lightweight gate after adding Refinitiv bridge alternative-identifier
selection:
`306 passed`.

Focused Rust acceleration suite after adding Refinitiv ownership authority
distinct ownership-value sorting:
`249 passed`.

Standalone Refinitiv ownership authority fixture suite after the same change:
`2 passed`.

Expanded lightweight gate after adding Refinitiv ownership authority distinct
ownership-value sorting:
`306 passed`.

Focused Rust acceleration plus Refinitiv ownership authority fixture suites
after adding authority candidate-key normalization:
`251 passed`.

Expanded lightweight gate after adding Refinitiv ownership authority
candidate-key normalization:
`306 passed`.

Focused Rust acceleration plus Refinitiv bridge fixture suites after adding
ownership-universe candidate-key normalization:
`265 passed`.

Expanded lightweight gate after adding Refinitiv bridge ownership-universe
candidate-key normalization:
`309 passed`.

Focused Rust acceleration plus Refinitiv bridge fixture suites after adding
ownership-smoke lookup-input priority resolution:
`268 passed`.

Expanded lightweight gate after adding Refinitiv bridge ownership-smoke
lookup-input priority resolution:
`312 passed`.

Focused Rust acceleration plus Refinitiv bridge fixture suites after adding
resolution-diagnostic classifier helpers:
`271 passed`.

Expanded lightweight gate after adding Refinitiv bridge resolution-diagnostic
classifier helpers:
`315 passed`.

Focused Rust acceleration plus Refinitiv bridge fixture suites after adding
normalized lookup-result matching:
`274 passed`.

Expanded lightweight gate after adding Refinitiv bridge normalized
lookup-result matching:
`318 passed`.

Focused Rust acceleration plus Refinitiv bridge fixture suites after adding
summary reducer helpers:
`277 passed`.

Expanded lightweight gate after adding Refinitiv bridge summary reducer
helpers:
`321 passed`.

Focused Rust acceleration plus Refinitiv bridge fixture suites after adding
ownership-validation retrieval-role sort key:
`280 passed`.

Expanded lightweight gate after adding Refinitiv bridge ownership-validation
retrieval-role sort key:
`324 passed`.

Focused Rust acceleration plus Refinitiv bridge fixture suites after adding
accepted-source matching and adjacent-extension source choice:
`283 passed`.

Expanded lightweight gate after adding Refinitiv bridge accepted-source
matching and adjacent-extension source choice:
`327 passed`.

Focused Rust acceleration suite after adding LM2011 Phase 0 validation-audit
packet-scalar helpers:
`273 passed`.

Targeted Phase 0 packet-A audit smoke test after the same change:
`1 passed`.

Focused Rust acceleration suite after adding FinBERT constrained Hamilton
apportionment:
`276 passed`.

Targeted FinBERT benchmark allocation capacity fixture after the same change:
`1 passed`.

Focused Rust acceleration suite after adding multi-surface normalized
match-bound lookup:
`278 passed`.

Targeted multi-surface audit-pack smoke fixture after the same change:
`1 passed`.

Focused Rust acceleration suite after adding FinBERT sentence postprocessing
list transformation:
`281 passed`.

Targeted FinBERT sentence-postprocessing policy fixtures after the same change:
`12 passed`.

Focused Rust acceleration suite after adding item-text cleaning layout-only
line removal:
`284 passed`.

Standalone item-cleaning fixture suite after the same change:
`15 passed`.

Focused Rust acceleration suite after adding item-text cleaning table-block
line removal:
`287 passed`.

Standalone item-cleaning fixture suite after the same change:
`15 passed`.

Focused Rust acceleration suite after adding SEC source filename parsing:
`290 passed`.

Targeted filing-text parser/raw ZIP smoke tests after the same change:
`2 passed`.

Focused Rust acceleration suite after adding Refinitiv LSEG lookup batch
response normalization:
`293 passed`.

Targeted lookup API pipeline fixture after the same change:
`1 passed`.

Focused Rust acceleration suite after adding Refinitiv analyst source
request-group list normalization:
`296 passed`.

Targeted analyst normalized-output fixture after the same change:
`1 passed`.

Focused Rust acceleration suite after adding Refinitiv LSEG lookup request-item
row construction:
`299 passed`.

Targeted lookup API pipeline fixture after the same change:
`1 passed`.

Focused Rust acceleration suite after adding Refinitiv analyst LSEG actuals and
estimates request-item row construction:
`302 passed`.

Targeted analyst LSEG actuals and estimates API batching fixtures after the
same change:
`1 passed` each.

Focused Rust acceleration suite after adding Refinitiv ownership LSEG
universe/document request-item row construction:
`305 passed`.

Targeted ownership LSEG API batching/finalization fixtures after the same
change:
`2 passed`.

Focused Rust acceleration suite after adding Refinitiv ownership LSEG
universe/document batch response row normalization:
`308 passed`.

Targeted ownership response/API fixtures after the same change:
`3 passed`.

Focused Rust acceleration suite after adding Refinitiv analyst LSEG actuals
and estimates batch response row normalization:
`311 passed`.

Targeted analyst API fixtures after the same change:
`3 passed`.

Focused Rust acceleration plus standalone batching suites after adding
Refinitiv LSEG simple `batch_items` grouping/chunking:
`322 passed`.

Targeted lookup API smoke fixture after the same change:
`1 passed`.

Focused Rust acceleration suite after adding Refinitiv LSEG provider
response-frame fingerprint hashing:
`317 passed`.

Focused Rust acceleration suite after adding Refinitiv ownership authority
allowlist key normalization:
`320 passed`.

Targeted Refinitiv ownership authority allowlist pipeline fixture after the
same change:
`1 passed`.

Focused Rust acceleration suite after adding Refinitiv document-ownership
retrieval-sheet matching:
`323 passed`.

Targeted Refinitiv document-ownership finalize fixture after the same change:
`1 passed`.

Focused Rust acceleration suite after adding Refinitiv document-ownership
exact-hit selection:
`326 passed`.

Targeted Refinitiv document-ownership finalize fixture after the same change:
`1 passed`.

Focused Rust acceleration suite after adding Refinitiv document-ownership
fallback-hit selection:
`329 passed`.

Targeted Refinitiv document-ownership finalize fixture after the same change:
`1 passed`.

Focused Rust acceleration suite after adding Refinitiv document-ownership
fallback-request row selection:
`332 passed`.

Targeted Refinitiv document-ownership finalize fixture after the same change:
`1 passed`.

Focused Rust acceleration suite after adding and expanding sentence split
quality batch flag classification:
`334 passed`.

Targeted sentence split quality analysis fixture after the same change:
`1 passed`.

Focused Rust acceleration suite after adding Refinitiv LSEG API execution
row-count aggregation:
`337 passed`.

Targeted Refinitiv LSEG API batch hardening fixture after the same change:
`1 passed`.

Focused Rust acceleration suite after adding Refinitiv LSEG API execution
mixed zero/positive requeue decision:
`340 passed`.

Targeted Refinitiv LSEG API batch hardening fixture after the same change:
`1 passed`.

Focused Rust acceleration suite after adding Refinitiv analyst estimate
date-index freezing plus actual-event and estimate-snapshot canonicalization:
`349 passed`.

Targeted analyst normalized-output fixtures after the same change:
`6 passed`.

Focused Rust acceleration suite after adding Refinitiv analyst batched
request-group ID hashing:
`352 passed`.

Targeted analyst request-group builder fixtures after the same change:
`4 passed`.

Focused row-count tests after switching Refinitiv LSEG API execution
row-count aggregation from one-column row dictionaries to plain `item_id`
value lists:
`3 passed`.

Targeted LSEG API batch hardening fixture after the same row-count transport
change:
`1 passed`.

Focused Rust acceleration suite after the same row-count transport change:
`352 passed`.

Focused allocation tests after adding FinBERT sentence-confusion balanced and
proportional sample-count allocation:
`6 passed`.

Standalone FinBERT sentence-confusion review suite after the same change:
`4 passed`.

Focused Rust acceleration suite after the same allocation change:
`358 passed`.

Focused batch helper tests after adding FinBERT sentence-example batched
sample-key and normalized-text helpers:
`6 passed`.

Standalone FinBERT sentence-example suite after the same change:
`2 passed`.

Focused Rust acceleration suite after the same sentence-example batch helper
change:
`364 passed`.

Focused stage-summary tests after adding FinBERT benchmark stage-summary
aggregation:
`3 passed`.

Copied FinBERT benchmark runner and sweep smoke tests after the same
stage-summary change:
`4 passed`.

Focused Rust acceleration suite after the same stage-summary change:
`367 passed`.

Focused staged bucket-summary tests after adding FinBERT staged-inference
tokenizer bucket-summary aggregation:
`3 passed`.

Standalone FinBERT staged-inference fixture suite after the same
bucket-summary aggregation change:
`5 passed`.

Focused Rust acceleration suite after the same staged bucket-summary change:
`370 passed`.

Focused FinBERT item-analysis coverage-report tests:
`3 passed, 679 deselected`.

Standalone FinBERT item-analysis suite after the same coverage-report change:
`14 passed`.

Focused Rust acceleration suite after the same item-analysis coverage-report
change:
`682 passed`.

Focused preprocessing bucket-count tests after adding FinBERT
sentence-preprocessing token-bucket count reduction:
`3 passed`.

Targeted sentence-preprocessing fixture subset after the same bucket-count
change:
`8 passed`.

Focused Rust acceleration suite after the same preprocessing bucket-count
change:
`373 passed`.

Focused preprocessing manifest-count tests after adding FinBERT
sentence-preprocessing manifest count aggregation:
`3 passed, 645 deselected`.

Focused Rust acceleration suite after the same preprocessing manifest-count
change:
`648 passed`.

Targeted sentence-preprocessing by-year artifact fixture after the same
manifest-count change:
`1 passed, 22 deselected`.

Focused preprocessing split-metric tests after adding FinBERT
sentence-preprocessing split-audit metric reduction:
`3 passed, 676 deselected`.

Standalone synthetic FinBERT sentence dataset suite after the same split-metric
change:
`23 passed`.

Focused Rust acceleration suite after the same preprocessing split-metric
change:
`679 passed`.

Focused fallback split-warning tests after adding FinBERT sentence-preprocessing
fallback split-boundary warning payload reduction:
`3 passed, 682 deselected`.

Standalone synthetic FinBERT sentence dataset suite after the same fallback
split-warning change:
`23 passed`.

Focused Rust acceleration suite after the same preprocessing fallback
split-warning change:
`685 passed`.

Focused capacity-and-target reducer tests after adding FinBERT benchmark
dataset capacity/target reducers:
`3 passed`.

Standalone synthetic FinBERT benchmark dataset suite after the same
capacity/target reducer change:
`8 passed`.

Focused Rust acceleration suite after the same capacity/target reducer change:
`376 passed`.

Focused year-item allocation tests after adding FinBERT benchmark dataset
year-item allocation orchestration:
`3 passed`.

Standalone synthetic FinBERT benchmark dataset suite after the same year-item
allocation change:
`8 passed`.

Focused Rust acceleration suite after the same year-item allocation change:
`379 passed`.

Focused reducer tests after adding FinBERT sentence-confusion review count and
uncertain-bound reducers:
`3 passed`.

Standalone FinBERT sentence-confusion review suite after the same reducer
change:
`4 passed`.

Focused Rust acceleration suite after the same confusion-review reducer change:
`382 passed`.

Focused bucket-metric tests after adding FinBERT sentence-confusion review
majority-bucket metric row aggregation:
`3 passed`.

Standalone FinBERT sentence-confusion review suite after the same bucket-metric
change:
`4 passed`.

Focused Rust acceleration suite after the same bucket-metric aggregation change:
`385 passed`.

Focused Packet A delta tests after adding LM2011 Phase 0 validation-audit
Packet A token-delta summary aggregation:
`3 passed`.

Targeted Phase 0 Packet A audit fixture after the same delta-summary change:
`1 passed`.

Focused Rust acceleration suite after the same Packet A delta-summary change:
`388 passed`.

Focused Packet A summary-frame tests after adding LM2011 Phase 0
validation-audit Packet A summary-frame aggregation:
`3 passed`.

Targeted Phase 0 Packet A audit fixture after the same summary-frame change:
`1 passed`.

Focused Rust acceleration suite after the same Packet A summary-frame change:
`391 passed`.

Focused Packet A strip-comparison tests after adding LM2011 Phase 0
validation-audit Packet A strip-comparison aggregation:
`3 passed`.

Targeted Phase 0 Packet A audit fixture after the same strip-comparison change:
`1 passed`.

Focused Rust acceleration suite after the same Packet A strip-comparison
change:
`394 passed`.

Focused Packet A examples-frame tests after adding LM2011 Phase 0
validation-audit Packet A example-row selection and ordering:
`3 passed`.

Targeted Phase 0 Packet A audit fixture after the same examples-frame change:
`1 passed`.

Focused Rust acceleration suite after the same Packet A examples-frame change:
`397 passed`.

Focused extension result-row converter tests after adding LM2011 extension
result-row conversion:
`3 passed`.

Targeted synthetic LM2011 extension estimation scaffold after the same
result-row conversion change:
`2 passed`.

Focused Rust acceleration suite after the same LM2011 extension result-row
conversion change:
`400 passed`.

Focused quarterly-fit row tests after adding LM2011 extension quarterly fit-row
construction:
`3 passed`.

Targeted synthetic LM2011 extension fit-comparison scaffold after the same
quarterly fit-row change:
`3 passed`.

Focused Rust acceleration suite after the same quarterly fit-row change:
`403 passed`.

Focused skipped-quarter row tests after adding LM2011 extension skipped-quarter
diagnostic row construction:
`3 passed`.

Targeted synthetic LM2011 extension fit-comparison scaffold after the same
skipped-quarter row change:
`3 passed`.

Focused Rust acceleration suite after the same skipped-quarter row change:
`406 passed`.

Focused quarterly-difference row tests after adding LM2011 extension quarterly
fit-difference row construction:
`3 passed`.

Targeted synthetic LM2011 extension fit-comparison scaffold after the same
quarterly fit-difference row change:
`3 passed`.

Focused Rust acceleration suite after the same quarterly fit-difference row
change:
`409 passed`.

Focused fit-summary row tests after adding LM2011 extension common-quarter
fit-summary aggregation:
`3 passed`.

Targeted synthetic LM2011 extension fit-comparison scaffold after the same
fit-summary aggregation change:
`3 passed`.

Focused Rust acceleration suite after the same fit-summary aggregation change:
`412 passed`.

Focused fit-comparison row tests after adding LM2011 extension common-quarter
fit-comparison aggregation:
`3 passed`.

Targeted synthetic LM2011 extension fit-comparison scaffold after the same
fit-comparison aggregation change:
`3 passed`.

Focused Rust acceleration suite after the same fit-comparison aggregation
change:
`415 passed`.

Focused IA.II result-row tests after adding LM2011 Table IA.II monthly
strategy result-row expansion:
`3 passed`.

Targeted synthetic LM2011 IA.II regression fixtures after the same row
expansion change:
`2 passed`.

Focused Rust acceleration suite after the same IA.II row expansion change:
`418 passed`.

Focused Fama-MacBeth result-row tests after adding LM2011 quarterly
Fama-MacBeth final result-row aggregation:
`3 passed`.

Targeted synthetic LM2011 quarterly regression fixtures after the same
Fama-MacBeth row aggregation change:
`3 passed`.

Focused Rust acceleration suite after the same Fama-MacBeth row aggregation
change:
`421 passed`.

Focused cross-sectional OLS design-row tests after adding LM2011 quarterly
regression design-row preparation:
`3 passed`.

Targeted synthetic LM2011 quarterly regression fixtures after the same
design-row preparation change:
`3 passed`.

Focused Rust acceleration suite after the same design-row preparation change:
`424 passed`.

Focused feature-row tests after adding LM2011 text feature-row construction:
`3 passed`.

Targeted LM2011 text-feature contract tests after the same feature-row
construction change:
`3 passed`.

Focused Rust acceleration suite after the same feature-row construction change:
`427 passed`.

Focused FF3 OLS alpha/RMSE tests after adding LM2011 fixed-shape OLS
calculation:
`3 passed`.

Targeted deterministic and rank-deficient FF3 fixture tests after the same OLS
change:
`2 passed`.

Focused Rust acceleration suite after the same OLS change:
`430 passed`.

Focused dictionary-token tests plus the text-feature builder fixture after
adding LM2011 dictionary-token normalization:
`4 passed`.

Targeted LM2011 text-feature contract tests after the same dictionary-token
normalization change:
`3 passed`.

Focused Rust acceleration suite after the same dictionary-token normalization
change:
`433 passed`.

Focused regression-window row tests after adding LM2011 event-window
per-document regression metric row construction:
`3 passed`.

Targeted small event-window fixtures after the same regression-window row
change:
`3 passed`.

Focused Rust acceleration suite after the same regression-window row change:
`436 passed`.

Focused FF4 coefficient tests after adding LM2011 fixed-shape FF4 OLS
calculation:
`3 passed`.

Targeted deterministic FF4 pipeline fixture after the same FF4 OLS change:
`1 passed`.

Focused Rust acceleration suite after the same FF4 OLS change:
`439 passed`.

Focused strategy factor-loading row tests after adding LM2011 monthly strategy
factor-loading row construction:
`3 passed`.

Targeted trading-strategy FF4 fixture after the same factor-loading row change:
`1 passed`.

Focused Rust acceleration suite after the same factor-loading row change:
`442 passed`.

Focused document-stat tests plus the text-feature builder fixture after adding
LM2011 document-stat preparation:
`4 passed`.

Targeted LM2011 text-feature contract tests after the same document-stat
preparation change:
`3 passed`.

Focused Rust acceleration suite after the same document-stat preparation
change:
`445 passed`.

Focused pass-1 row tests plus the text-feature builder fixture after adding
LM2011 streaming pass-1 row construction:
`4 passed`.

Targeted streaming text-feature writer fixtures after the same pass-1 row
change:
`2 passed`.

Focused Rust acceleration suite after the same pass-1 row change:
`448 passed`.

Focused pass-1 feature-row tests after adding LM2011 streaming pass-2
feature-row construction from pass-1 shards:
`3 passed`.

Focused text-feature builder fixture after the same pass-2 feature-row change:
`1 passed`.

Targeted streaming text-feature writer fixtures after the same pass-2
feature-row change:
`2 passed`.

Focused Rust acceleration suite after the same pass-2 feature-row change:
`451 passed`.

Focused cleaned-text document-stat/pass-1 tests after extending those paths to
use batch Rust counting after Python text cleaning:
`2 passed`.

Focused document-stat/pass-1 tests after the same cleaned-text change:
`8 passed`.

Targeted streaming text-feature writer fixtures after the same cleaned-text
change:
`2 passed`.

Focused Rust acceleration suite after the same cleaned-text change:
`453 passed`.

Focused visible-prefix Rust tests after adding FinBERT visible-prefix LM2011
token-count batching:
`6 passed`.

Targeted visible-prefix tokenization fixtures after the same batch token-count
change:
`2 passed`.

Focused Rust acceleration suite after the same batch token-count change:
`456 passed`.

Focused bucket-length summary-row tests after adding FinBERT bucket-length
tuning summary row assembly:
`3 passed`.

Focused bucket-length helper selection after the same summary-row change:
`10 passed`.

Focused Rust acceleration suite after the same summary-row change:
`459 passed`.

Focused sentence chunk tests after adding sentence dataset chunk-row expansion:
`6 passed`.

Targeted synthetic sentence-dataset chunk fixture after the same chunk-row
change:
`1 passed`.

Focused Rust acceleration suite after the same chunk-row change:
`462 passed`.

Focused item-cleaning token-count tests after adding item-text cleaning
LM2011 token-count batching:
`4 passed`.

Focused item-cleaning helper selection after the same token-count change:
`26 passed`.

Focused Rust acceleration suite after the same token-count change:
`466 passed`.

Focused multi-surface chunk tests after adding audit-record chunk planning:
`3 passed`.

Focused Rust acceleration suite after the same chunk-planning change:
`469 passed`.

Focused FinBERT sentence confusion-review sample-ID tests:
`3 passed`.

Focused Rust acceleration suite after the same sample-ID row change:
`472 passed`.

Focused FinBERT sentence confusion-review neighbor-target tests:
`3 passed`.

Focused Rust acceleration suite after the same neighbor-target row change:
`475 passed`.

Focused FinBERT sentence confusion-review allocation-row tests:
`3 passed`.

Focused Rust acceleration suite after the same allocation-row finalization
change:
`478 passed`.

Focused FinBERT sentence confusion-review labeling-record tests:
`3 passed`.

Focused Rust acceleration suite after the same labeling-record change:
`481 passed`.

Focused FinBERT sentence confusion-review chunk-row tests:
`3 passed`.

Focused Rust acceleration suite after the same chunk-row planning change:
`484 passed`.

Focused FinBERT sentence confusion-review CSV-safe row tests:
`3 passed`.

Focused Rust acceleration suite after the same CSV-safe row change:
`487 passed`.

Focused FinBERT sentence confusion-review reviewed-row tests:
`3 passed`.

Focused Rust acceleration suite after the same reviewed-row change:
`490 passed`.

Focused FinBERT sentence confusion-review examples-by-cell tests:
`3 passed, 688 deselected`.

Standalone FinBERT sentence-confusion review suite after the same
examples-by-cell change:
`4 passed`.

Focused Rust acceleration suite after the same examples-by-cell change:
`691 passed`.

Focused FinBERT high-confidence sentence-example sample-candidate row tests:
`3 passed`.

Focused Rust acceleration suite after the same sample-candidate row change:
`493 passed`.

Focused FinBERT high-confidence sentence-example count-row tests:
`3 passed`.

Focused Rust acceleration suite after the same sentence-example count-row
change:
`496 passed`.

Focused FinBERT high-confidence sentence-example markdown-renderer tests:
`3 passed, 670 deselected`.

Standalone FinBERT sentence-example fixture suite after the same markdown
change:
`2 passed`.

Focused Rust acceleration suite after the same sentence-example markdown
change:
`673 passed`.

Focused manifest-contract hash helper tests after moving stable string
fingerprints into Rust:
`5 passed`.

Focused Rust acceleration suite after the same manifest-contract fingerprint
change:
`496 passed`.

Focused FinBERT token-bucket tests after adding batch assignment:
`7 passed`.

Focused Rust acceleration suite after the same token-bucket batch-assignment
change:
`500 passed`.

Focused Refinitiv bridge resolution summary-count tests:
`3 passed`.

Focused Rust acceleration suite after the same bridge resolution summary-count
change:
`503 passed`.

Focused Refinitiv bridge resolution class-summary tests:
`3 passed`.

Focused Rust acceleration suite after the same bridge resolution class-summary
change:
`506 passed`.

Focused Refinitiv bridge summary-reducer tests after adding batched strict-True
field counts:
`3 passed`.

Focused Rust acceleration suite after the same batched field-count change:
`506 passed`.

Focused Refinitiv bridge ownership-validation case-row tests:
`3 passed`.

Focused Rust acceleration suite after the same ownership-validation case-row
change:
`509 passed`.

Focused Refinitiv bridge ownership-universe handoff-row tests:
`3 passed`.

Focused Rust acceleration suite after the same ownership-universe handoff-row
change:
`512 passed`.

Focused Refinitiv bridge ownership-smoke sample-row tests:
`3 passed`.

Focused Rust acceleration suite after the same ownership-smoke sample-row
change:
`515 passed`.

Focused Refinitiv bridge batched failed-lookup record tests:
`3 passed`.

Focused Rust acceleration suite after the same batched failed-lookup change:
`518 passed`.

Focused Refinitiv bridge batched alternative-identifier tests:
`3 passed`.

Focused Rust acceleration suite after the same batched alternative-identifier
change:
`521 passed`.

Focused Refinitiv LSEG provider identifier-parser tests after exposing direct
unresolved-identifier message parsing:
`3 passed`.

Focused Rust acceleration suite after the same LSEG parser change:
`521 passed`.

Focused Refinitiv LSEG provider UTF-8 record coercion tests:
`3 passed`.

Focused Rust acceleration suite after the same UTF-8 record coercion change:
`524 passed`.

Focused Refinitiv LSEG provider response-header sanitization tests:
`3 passed`.

Focused Rust acceleration suite after the same response-header sanitization
change:
`527 passed`.

Focused Refinitiv LSEG session-not-opened message tests:
`3 passed`.

Focused Rust acceleration suite after the same session-message change:
`530 passed`.

Focused Refinitiv LSEG stage-audit optional mapping normalizer tests:
`3 passed`.

Focused Rust acceleration suite after the same optional mapping change:
`530 passed`.

Focused SEC utility tests after adding newline normalization, date parsing,
Roman numeral parsing, default item-part inference, and bullet-prefix
classification:
`3 passed`.

Focused Rust acceleration suite after the same SEC utility changes:
`530 passed`.

Focused SEC boundary-authority/filename tests after adding boundary-status
classification and public boundary payload projection:
`9 passed`.

Focused Rust acceleration suite after the same boundary-status/payload change:
`536 passed`.

Focused sample item-cleaning sentence diagnostics doc-id sampler tests:
`3 passed`.

Existing deterministic sample item-cleaning diagnostics sampler test:
`1 passed`.

Focused Rust acceleration suite after the same diagnostics sampler change:
`539 passed`.

Focused FinBERT sentence-length report helper tests:
`3 passed`.

Standalone FinBERT sentence-length visualization fixture:
`1 passed`.

Focused Rust acceleration suite after the same sentence-length helper change:
`542 passed`.

Focused sentence split-quality report ordering tests:
`3 passed`.

Standalone sentence split-quality fixture:
`1 passed`.

Focused Rust acceleration suite after the same split-quality ordering change:
`545 passed`.

Focused FinBERT sentence-example ordering tests:
`3 passed`.

Existing FinBERT sentence-example sample/count Rust tests:
`6 passed`.

Focused Rust acceleration suite after the same sentence-example ordering
change:
`548 passed`.

Focused SEC/CCM markdown table tests:
`3 passed`.

Synthetic SEC/CCM end-to-end artifact fixture:
`1 passed`.

Focused Rust acceleration suite after the same SEC/CCM markdown change:
`551 passed`.

Focused SEC/CCM markdown scalar/table tests after exposing the scalar formatter
through Rust:
`6 passed, 688 deselected`.

Synthetic Rust acceleration suite after the same scalar formatter change:
`694 passed`.

Focused Refinitiv ownership request-log event-count tests:
`3 passed`.

Focused Rust acceleration suite after the same request-log counter change:
`554 passed`.

Focused Refinitiv LSEG ledger item-ID JSON tests:
`4 passed`.

Focused Rust acceleration suite after the same ledger item-ID JSON change:
`558 passed`.

Existing request-ledger claim/stale-requeue fixture after the same change:
`1 passed`.

Focused Refinitiv LSEG stage-audit item-ID JSON tests:
`3 passed`.

Focused Rust acceleration suite after the same stage-audit item-ID JSON change:
`561 passed`.

Focused Refinitiv LSEG ledger string-array plus item-ID JSON tests:
`8 passed`.

Focused Rust acceleration suite after the same ledger string-array JSON change:
`565 passed`.

Focused LM2011 FF48 SIC mapping parser tests:
`3 passed`.

Focused Rust acceleration suite after the same FF48 parser change:
`568 passed`.

Focused SEC pipeline completed-year checkpoint tests:
`3 passed`.

Focused Rust acceleration suite after the same SEC checkpoint change:
`571 passed`.

Focused SEC pipeline no-item aggregation tests:
`3 passed, 685 deselected`.

Existing filing-text no-item aggregation fixture after the same SEC pipeline
change:
`1 passed, 54 deselected`.

Focused Rust acceleration suite after the same SEC no-item aggregation change:
`688 passed`.

Focused Refinitiv bridge ownership-result helper tests after adding date
normalization:
`3 passed, 568 deselected`.

Focused Rust acceleration suite after the same ownership-result date change:
`571 passed`.

Focused Refinitiv bridge date-text helper tests after adding ownership-universe
request-date and date-text rendering:
`3 passed, 571 deselected`.

Focused Rust acceleration suite after the same bridge date-text change:
`574 passed`.

Focused item-text cleaning row-audit helper tests:
`3 passed, 574 deselected`.

Focused Rust acceleration suite after the same item-cleaning row-audit helper
change:
`577 passed`.

Focused FinBERT sentence-example sample-selection tests:
`3 passed, 577 deselected`.

Focused Rust acceleration suite after the same FinBERT sample-selection helper
change:
`580 passed`.

Focused Refinitiv LSEG API execution child-batch tests:
`3 passed, 580 deselected`.

Focused Rust acceleration suite after the same child-batch helper change:
`583 passed`.

Focused Refinitiv LSEG API execution item-result detail tests:
`3 passed, 583 deselected`.

Focused Rust acceleration suite after the same item-result detail helper
change:
`586 passed`.

Focused Refinitiv LSEG interval-batch candidate evaluation tests:
`3 passed, 586 deselected`.

Focused Rust acceleration suite after the same interval-candidate helper
change:
`589 passed`.

Focused FinBERT sentence-example accumulator update tests:
`3 passed, 589 deselected`.

Focused Rust acceleration suite after the same accumulator update helper
change:
`592 passed`.

Focused Refinitiv LSEG split-batch child metadata tests:
`3 passed, 592 deselected`.

Focused Rust acceleration suite after the same split-batch helper change:
`595 passed`.

Focused Refinitiv LSEG API common daily-limit helper tests:
`3 passed, 595 deselected`.

Focused Rust acceleration suite after the same daily-limit helper change:
`598 passed`.

Focused Refinitiv LSEG API common empty-result classifier tests:
`3 passed, 598 deselected`.

Focused Rust acceleration suite after the same empty-result classifier change:
`601 passed`.

Focused Refinitiv LSEG API common batch error-policy tests:
`3 passed, 601 deselected`.

Focused Rust acceleration suite after the same batch error-policy change:
`604 passed`.

Focused manifest semantic reuse-guard mismatch tests:
`2 passed, 604 deselected`.

Focused Rust acceleration suite after the same semantic reuse-guard helper
change:
`606 passed`.

Focused Refinitiv bridge accepted-resolution derivation tests:
`3 passed, 606 deselected`.

Focused Rust acceleration suite after the same accepted-resolution helper
change:
`609 passed`.

Focused Refinitiv bridge candidate-extractor plus accepted-resolution tests:
`6 passed, 606 deselected`.

Focused Rust acceleration suite after the same candidate-extractor helper
change:
`612 passed`.

Focused Refinitiv bridge resolver tests after adding effective-resolution
field derivation:
`9 passed, 606 deselected`.

Focused Rust acceleration suite after the same effective-resolution helper
change:
`615 passed`.

Targeted Refinitiv bridge resolver fixtures after the same helper change:
`2 passed, 11 deselected`.

Focused Refinitiv bridge resolution-diagnostic handoff row tests:
`3 passed, 615 deselected`.

Focused Rust acceleration suite after the same handoff row helper change:
`618 passed`.

Targeted Refinitiv bridge resolution diagnostic fixture after the same helper
change:
`1 passed, 12 deselected`.

Focused Refinitiv bridge ownership-validation handoff row tests:
`3 passed, 618 deselected`.

Focused Rust acceleration suite after the same ownership-validation handoff
helper change:
`621 passed`.

Targeted Refinitiv bridge handoff/diagnostic fixture after the same helper
change:
`1 passed, 12 deselected`.

Focused Refinitiv document-ownership final-row tests:
`3 passed, 621 deselected`.

Focused Rust acceleration suite after the same document-ownership final-row
helper change:
`624 passed`.

Standalone Refinitiv document-ownership pipeline fixture suite after the same
helper change:
`13 passed`.

Focused FinBERT benchmark label-mapping tests:
`3 passed, 624 deselected`.

Focused Rust acceleration suite after the same FinBERT label-mapping helper
change:
`627 passed`.

Targeted FinBERT item-analysis mapping fixture after the same helper change:
`1 passed, 13 deselected`.

Focused FinBERT benchmark probability-label tests:
`4 passed, 651 deselected`.

Focused Rust acceleration suite after the same FinBERT probability-label
helper change:
`655 passed`.

Targeted FinBERT item-analysis score fixture after the same helper change:
`1 passed, 13 deselected`.

Targeted FinBERT staged-inference fake-model fixture after the same helper
change:
`1 passed, 4 deselected`.

Focused item-text cleaning finalize-row tests:
`3 passed, 655 deselected`.

Focused Rust acceleration suite after the same item-cleaning finalize-row
helper change:
`658 passed`.

Standalone item-text cleaning fixture suite after the same helper change:
`15 passed`.

Focused SEC HTML-audit scalar helper tests after adding filing status
classification:
`3 passed, 655 deselected`.

Small SEC HTML-audit output fixture after the same filing status change:
`1 passed`.

Focused Rust acceleration suite after the same HTML-audit filing status
change:
`658 passed`.

Focused LM2011 streaming writer doc-id validation tests:
`3 passed, 658 deselected`.

Targeted LM2011 duplicate-writer fixtures after the same doc-id validation
change:
`2 passed, 81 deselected`.

Focused Rust acceleration suite after the same doc-id validation change:
`661 passed`.

Focused Refinitiv document-ownership date-window helper tests:
`3 passed, 661 deselected`.

Targeted Refinitiv document-ownership request-bound fixtures after the same
date-window change:
`6 passed, 7 deselected`.

Focused Rust acceleration suite after the same document-ownership date-window
change:
`664 passed`.

Focused FinBERT benchmark dataset share-row tests:
`3 passed, 664 deselected`.

Standalone synthetic FinBERT benchmark dataset suite after the same share-row
change:
`8 passed`.

Focused Rust acceleration suite after the same FinBERT benchmark dataset
share-row change:
`667 passed`.

Focused FinBERT benchmark dataset year-allocation tests:
`3 passed, 667 deselected`.

Standalone synthetic FinBERT benchmark dataset suite after the same
year-allocation change:
`8 passed`.

Focused Rust acceleration suite after the same FinBERT benchmark dataset
year-allocation change:
`670 passed`.

Focused FinBERT benchmark dataset ranked-selection tests:
`3 passed, 673 deselected`.

Standalone synthetic FinBERT benchmark dataset suite after the same
ranked-selection change:
`8 passed`.

Focused Rust acceleration suite after the same FinBERT benchmark dataset
ranked-selection change:
`676 passed`.

Focused LM2011 Phase 0 validation-audit form-count row tests:
`3 passed, 627 deselected`.

Focused Rust acceleration suite after the same validation-audit form-count
helper change:
`630 passed`.

Targeted LM2011 Phase 0 validation-audit subset after the same helper change:
`2 passed, 7 deselected`.

Focused LM2011 Phase 0 validation-audit representative-term count tests:
`3 passed, 630 deselected`.

Focused Rust acceleration suite after the same validation-audit term-count
helper change:
`633 passed`.

Targeted LM2011 Phase 0 validation-audit subset after the same helper change:
`2 passed, 7 deselected`.

Focused LM2011 Phase 0 validation-audit normalized-form count tests:
`3 passed, 633 deselected`.

Focused Rust acceleration suite after the same validation-audit normalized-form
helper change:
`636 passed`.

Targeted LM2011 Phase 0 validation-audit subset after the same helper change:
`2 passed, 7 deselected`.

Focused LM2011 Phase 0 validation-audit Packet B event-attrition row tests:
`3 passed, 636 deselected`.

Focused Rust acceleration suite after the same validation-audit event-attrition
helper change:
`639 passed`.

Targeted LM2011 Phase 0 validation-audit subset after the same helper change:
`2 passed, 7 deselected`.

Focused LM2011 Phase 0 validation-audit Packet C units-row tests:
`3 passed, 639 deselected`.

Focused Rust acceleration suite after the same validation-audit units-row
helper change:
`642 passed`.

Targeted LM2011 Phase 0 validation-audit Packet C fixture after the same helper
change:
`1 passed, 8 deselected`.

Focused LM2011 Phase 0 validation-audit Packet A MD&A row tests:
`3 passed, 642 deselected`.

Focused Rust acceleration suite after the same validation-audit MD&A row helper
change:
`645 passed`.

Targeted LM2011 Phase 0 validation-audit Packet A fixture after the same helper
change:
`1 passed, 8 deselected`.

Focused LM2011 Phase 0 validation-audit Packet D coverage-row tests:
`3 passed, 648 deselected`.

Focused Rust acceleration suite after the same validation-audit Packet D
coverage-row helper change:
`651 passed`.

Targeted LM2011 Phase 0 validation-audit Packet D fixture after the same helper
change:
`1 passed, 8 deselected`.

Focused Refinitiv bridge ownership-validation pairwise comparison tests:
`4 passed, 694 deselected`.

Focused Rust acceleration suite after the same ownership-comparison change:
`698 passed`.

Focused Refinitiv ownership authority pairwise alias-diagnostics tests:
`4 passed, 698 deselected`.

Focused Rust acceleration suite after the same alias-diagnostics change:
`702 passed`.

Standalone Refinitiv ownership authority fixture suite after the same change:
`2 passed`.

Focused Refinitiv ownership authority candidate-metrics tests:
`3 passed, 702 deselected`.

Focused Rust acceleration suite after the same candidate-metrics change:
`705 passed`.

Standalone Refinitiv ownership authority fixture suite after the same
candidate-metrics change:
`2 passed`.

Focused Refinitiv analyst normalized-event row tests:
`3 passed, 705 deselected`.

Focused Rust acceleration suite after the same normalized-event change:
`708 passed`.

Standalone Refinitiv analyst fixture suite after the same normalized-event
change:
`13 passed`.

Focused Refinitiv bridge material-conflict tests:
`3 passed, 708 deselected`.

Focused Refinitiv bridge identity/material-conflict/accepted-resolution tests:
`9 passed, 702 deselected`.

Focused Rust acceleration suite after the same material-conflict change:
`711 passed`.

Standalone Refinitiv bridge fixture suite after the same material-conflict
change:
`13 passed`.

Focused Item 7 LM floor-sweep threshold-summary tests:
`3 passed, 711 deselected`.

Standalone Item 7 LM floor-sweep fixture after the same threshold-summary
change:
`1 passed`.

Focused Rust acceleration suite after the same threshold-summary change:
`714 passed`.

Focused Refinitiv Excel ownership-universe payload/workbook tests:
`4 passed, 714 deselected`.

Focused Rust acceleration suite after the same Excel ownership-universe
payload change:
`718 passed`.

Focused Refinitiv Excel LM2011 document-ownership payload/workbook tests:
`4 passed, 718 deselected`.

Combined focused Refinitiv Excel payload/workbook tests:
`8 passed, 714 deselected`.

Standalone Refinitiv document-ownership workbook formula fixture after the
same doc-ownership Excel payload change:
`1 passed, 12 deselected`.

Focused Rust acceleration suite after the same doc-ownership Excel payload
change:
`722 passed`.

Focused Refinitiv Excel resolution-diagnostic payload/workbook tests:
`4 passed, 722 deselected`.

Combined focused Refinitiv Excel payload/workbook tests after the same
resolution-diagnostic Excel payload change:
`12 passed, 714 deselected`.

Focused Rust acceleration suite after the same resolution-diagnostic Excel
payload change:
`726 passed`.

Focused Refinitiv Excel ownership-smoke payload/workbook tests:
`4 passed, 726 deselected`.

Combined focused Refinitiv Excel payload/workbook tests after the same
ownership-smoke Excel payload change:
`16 passed, 714 deselected`.

Focused Rust acceleration suite after the same ownership-smoke Excel payload
change:
`730 passed`.

Focused Refinitiv Excel ownership-validation payload/workbook tests:
`4 passed, 730 deselected`.

Combined focused Refinitiv Excel payload/workbook tests after the same
ownership-validation Excel payload change:
`20 passed, 714 deselected`.

Focused Rust acceleration suite after the same ownership-validation Excel
payload change:
`734 passed`.

Focused Refinitiv Excel extended-summary formula payload/workbook tests:
`4 passed, 734 deselected`.

Combined focused Refinitiv Excel payload/workbook tests after the same
extended-summary Excel payload change:
`24 passed, 714 deselected`.

Focused Rust acceleration suite after the same extended-summary Excel payload
change:
`738 passed`.

Focused Refinitiv Excel extended-lookup formula payload/workbook tests:
`4 passed, 738 deselected`.

Combined focused Refinitiv Excel payload/workbook tests after the same
extended-lookup Excel payload change:
`28 passed, 714 deselected`.

Focused Rust acceleration suite after the same extended-lookup Excel payload
change:
`742 passed`.

Focused LM2011 document-ownership request-row tests:
`3 passed, 742 deselected`.

Focused LM2011 document-ownership Rust tests after the same request-row change:
`40 passed, 705 deselected`.

Focused Rust acceleration suite after the same document-ownership request-row
change:
`745 passed`.

Focused LM2011 document-ownership universe-summary tests:
`3 passed, 745 deselected`.

Focused LM2011 document-ownership Rust tests after the same universe-summary
change:
`43 passed, 705 deselected`.

Standalone LM2011 document-ownership universe-diagnostics pipeline tests after
the same universe-summary change:
`2 passed, 11 deselected`.

Focused Rust acceleration suite after the same document-ownership
universe-summary change:
`748 passed`.

Focused LM2011 SEC/CCM form batch-normalization tests:
`6 passed, 744 deselected`.

Small LM2011 CCM suite after the same batch-normalization change:
`7 passed`.

Focused Rust acceleration suite after the same SEC/CCM batch form-normalization
change:
`750 passed`.

Focused LM2011 document-ownership batch/membership diagnostics tests:
`6 passed, 747 deselected`.

Focused LM2011 document-ownership Rust tests after the same membership-prep
batch-normalization change:
`46 passed, 707 deselected`.

Standalone LM2011 document-ownership universe-diagnostics pipeline tests after
the same membership-prep batch-normalization change:
`2 passed, 11 deselected`.

Focused Rust acceleration suite after the same document-ownership membership
batch-normalization change:
`753 passed`.

Copied-runner import-only smoke checks:

```powershell
$env:PYTHONPATH = "src_rust_migration"
python -c "from thesis_pkg.notebooks_and_scripts import finbert_item_analysis_runner as m; print(m.SRC)"
python -c "from thesis_pkg.notebooks_and_scripts import lm2011_phase0_validation_audit as m; print(m.SRC)"
python -c "from thesis_pkg.notebooks_and_scripts import lm2011_finbert_robustness_runner as m; print(m.SRC)"
python -c "from thesis_pkg.notebooks_and_scripts import lm2011_sample_post_refinitiv_runner as m; print(m.SRC)"
python -c "from thesis_pkg.notebooks_and_scripts import sec_ccm_unified_runner as m; print(m.SRC)"
```

Each command printed the root-level `src_rust_migration` path.

Additional targeted probe:

```powershell
pytest tests\test_finbert_tail_features.py
pytest tests\test_lm2011_finbert_robustness_runner.py
pytest tests\test_lm2011_rust_accel.py -k "finbert_tail"
```

Latest results after adding `top_5_sentences_neg_mean` to the copied FinBERT
tail document-surface contract and making the tail-feature suite explicitly
import from `src_rust_migration`: `4 passed` and `5 passed`.
After adding the eager Rust tail document-surface reducer, the targeted
FinBERT tail Rust selector passed with `6 passed, 750 deselected`; the combined
standalone tail/robustness probe passed with `9 passed`; and the focused Rust
acceleration suite passed with `756 passed`.

After routing the FinBERT robustness runner's fit-comparison row assembly
through the Rust-backed LM2011 extension helpers, the standalone robustness
runner suite explicitly imported from `src_rust_migration` and passed with
`5 passed`. `cargo fmt --check`, `cargo check`, and the focused Rust
acceleration suite also passed for this slice, with `756 passed` in
`tests/test_lm2011_rust_accel.py`.

After adding the Parquet I/O stream-copy and magic-byte probe fast paths,
`cargo fmt --check` and `cargo check` passed for
`src_rust_migration/rust/lm2011_rust/Cargo.toml`; `python setup.py build_ext
--inplace` rebuilt the copied extension successfully; the focused Parquet
selector passed with `7 passed, 756 deselected`; and the full Rust
acceleration suite passed with `763 passed`.

After adding the SEC Parquet filing-text stream `doc_id` index selector,
`cargo fmt --check` and `cargo check` passed for
`src_rust_migration/rust/lm2011_rust/Cargo.toml`; `python setup.py build_ext
--inplace` rebuilt the copied extension successfully; and the focused Parquet
stream selector passed with `6 passed, 762 deselected`. The full Rust
acceleration suite also passed after this slice with `768 passed`.

After adding the Refinitiv LSEG document-recovery unresolved mask selector,
`cargo fmt --check` and `cargo check` passed for
`src_rust_migration/rust/lm2011_rust/Cargo.toml`; `python setup.py build_ext
--inplace` rebuilt the copied extension successfully; and the focused recovery
selector passed with `4 passed, 768 deselected`. The full Rust acceleration
suite also passed after this slice with `772 passed`.

After adding the LM2011 event-window row-index selector,
`cargo fmt --check` and `cargo check` passed for
`src_rust_migration/rust/lm2011_rust/Cargo.toml`; `python setup.py build_ext
--inplace` rebuilt the copied extension successfully; and the focused
event-window row-expansion selector passed with `4 passed, 772 deselected`.
The migration-forced LM2011 pipeline event-window gate passed with
`3 passed, 80 deselected`, and the full Rust acceleration suite passed after
this slice with `776 passed`.

After adding the fused Item-text cleaning `clean_item_text` fast path,
`cargo fmt --check` and `cargo check` passed for
`src_rust_migration/rust/lm2011_rust/Cargo.toml`; `python setup.py build_ext
--inplace` rebuilt the copied extension successfully; the focused clean-text
selector passed with `3 passed, 776 deselected`; the broader item-cleaning
Rust selector passed with `35 passed, 744 deselected`; the migration-forced
standalone `tests/test_item_text_cleaning.py` suite passed with `15 passed`;
and the full Rust acceleration suite passed after this slice with `779
passed`.

After adding the Item-text cleaning cleaned-scope preparation fast path,
`cargo fmt --check` and `cargo check` passed for
`src_rust_migration/rust/lm2011_rust/Cargo.toml`; `python setup.py build_ext
--inplace` rebuilt the copied extension successfully; the focused prepare-row
selector passed with `3 passed, 779 deselected`; the broader item-cleaning
Rust selector passed with `38 passed, 744 deselected`; the migration-forced
standalone `tests/test_item_text_cleaning.py` suite passed with `15 passed`;
and the full Rust acceleration suite passed after this slice with `782
passed`.

After adding SEC HTML-audit deterministic status/stratified sampling,
`cargo fmt --check` and `cargo check` passed for
`src_rust_migration/rust/lm2011_rust/Cargo.toml`; `python setup.py build_ext
--inplace` rebuilt the copied extension successfully; the focused
`html_audit_sampling` selector passed with `3 passed, 782 deselected`; the
broader `html_audit` selector passed with `6 passed, 779 deselected`; the full
Rust acceleration suite passed with `785 passed`; and the small HTML-audit
output fixture suite passed with `3 passed`.

After adding Item-text cleaning manual-audit sample construction,
`cargo fmt --check` and `cargo check` passed for
`src_rust_migration/rust/lm2011_rust/Cargo.toml`; `python setup.py build_ext
--inplace` rebuilt the copied extension successfully; the focused
manual-audit sample selector passed with `3 passed, 785 deselected`; the
broader item-cleaning Rust selector passed with `41 passed, 747 deselected`;
the migration-forced standalone `tests/test_item_text_cleaning.py` suite
passed with `15 passed`; and the full Rust acceleration suite passed with
`788 passed`.

After adding SEC extraction Part-marker scanning,
`python -m py_compile` passed for the modified extraction wrapper and migration
test file; `cargo fmt --check` and `cargo check` passed for
`src_rust_migration/rust/lm2011_rust/Cargo.toml`; `python setup.py build_ext
--inplace` rebuilt the copied extension successfully; the focused extraction
Part-marker selector passed with `3 passed, 788 deselected`; the combined SEC
utility/extraction selector passed with `6 passed, 785 deselected`; the full
Rust acceleration suite passed with `791 passed`; focused extraction dispatch
fixtures passed with `9 passed`; focused 10-Q/extraction logic fixtures passed
with `10 passed`; the filing-text suite passed with `55 passed`; and
suspicious-boundary diagnostics on
`.tmp/submission_pipeline_validation/data/sec/year_merged` with `--max-files 1`
reported `6` filings, `18` items, and `0` flagged items. The Hypothesis-based
extraction fuzz selector could not run because `hypothesis` is not installed in
this environment.

After adding Multi-surface audit escalation marking,
`python -m py_compile` passed for the modified audit wrapper and migration
test file; `cargo fmt --check` and `cargo check` passed for
`src_rust_migration/rust/lm2011_rust/Cargo.toml`; `python setup.py build_ext
--inplace` rebuilt the copied extension successfully; the focused escalation
selector passed with `3 passed, 791 deselected`; the broader multi-surface Rust
selector passed with `17 passed, 777 deselected`; the audit-pack fixture
`tests/test_multisurface_finbert_audit.py::test_build_multisurface_audit_pack_creates_cases_and_chunks`
passed with `1 passed`; and the full Rust acceleration suite passed with `794
passed`.

After adding SEC extraction high-confidence truncation,
`python -m py_compile` passed for the modified heuristic wrapper and migration
test file; `cargo fmt --check` and `cargo check` passed for
`src_rust_migration/rust/lm2011_rust/Cargo.toml`; `python setup.py build_ext
--inplace` rebuilt the copied extension successfully; the focused truncation
selector passed with `3 passed, 794 deselected`; the combined extraction
selector passed with `6 passed, 791 deselected`; and the full Rust
acceleration suite passed with `797 passed`. Suspicious-boundary diagnostics
run through the explicit migration entrypoint on `full_data_run/year_merged`
with `--max-files 1` processed `229` filings, extracted `1372` items, flagged
`225` items, and reported `2` embedded fail items.

After adding Refinitiv ownership authority conventional-component grouping,
`python -m py_compile` passed for the modified authority wrapper and migration
test file; `cargo fmt --check` and `cargo check` passed for
`src_rust_migration/rust/lm2011_rust/Cargo.toml`; `python setup.py build_ext
--inplace` rebuilt the copied extension successfully; the focused conventional
component selector passed with `3 passed, 797 deselected`; the broader
Refinitiv authority Rust selector passed with `19 passed, 781 deselected`; the
standalone authority pipeline fixture passed with `2 passed`; and the full Rust
acceleration suite passed with `800 passed`.

After adding SEC embedded-heading GIJ context detection, `python -m py_compile`
passed for the modified embedded-heading wrapper and migration test file;
`cargo fmt --check` and `cargo check` passed for
`src_rust_migration/rust/lm2011_rust/Cargo.toml`; `python setup.py build_ext
--inplace` rebuilt the copied extension successfully; the focused GIJ selector
passed with `3 passed, 800 deselected`; the migration-forced
`tests/test_gij_and_stubs.py::test_detect_gij_context_parses_omit_block`
fixture passed with `1 passed`; and the full Rust acceleration suite passed
with `803 passed`. Suspicious-boundary diagnostics run through the explicit
migration entrypoint on `full_data_run/year_merged` with `--max-files 1`
processed `229` filings, extracted `1372` items, flagged `225` items, and
reported `2` embedded fail items.

After adding SEC embedded-heading hit classification, `python -m py_compile`
passed for the modified embedded-heading wrapper and migration test file;
`cargo fmt --check` and `cargo check` passed for
`src_rust_migration/rust/lm2011_rust/Cargo.toml`; `python setup.py build_ext
--inplace` rebuilt the copied extension successfully; the focused
embedded-heading selector passed with `3 passed, 803 deselected`; the
migration-forced `tests/test_embedded_heading_verifier.py` suite passed with
`13 passed`; and the full Rust acceleration suite passed with `806 passed`.
Suspicious-boundary diagnostics run through the explicit migration entrypoint
on `full_data_run/year_merged` with `--max-files 1` processed `229` filings,
extracted `1372` items, flagged `225` items, and reported `2` embedded fail
items.

After adding FinBERT sentence-confusion review target-position generation,
`python -m py_compile` passed for the modified confusion-review wrapper and
migration test file; `cargo fmt --check` and `cargo check` passed for
`src_rust_migration/rust/lm2011_rust/Cargo.toml`; `python setup.py build_ext
--inplace` rebuilt the copied extension successfully; the focused
target-position selector passed with `4 passed, 806 deselected`; the
migration-forced `tests/test_finbert_sentence_confusion_review.py` suite
passed with `4 passed`; and the full Rust acceleration suite passed with
`810 passed`.

After adding SEC heuristic TOC line-range detection, `python -m py_compile`
passed for the modified heuristic wrapper and migration test file; `cargo
fmt --check` and `cargo check` passed for
`src_rust_migration/rust/lm2011_rust/Cargo.toml`; `python setup.py build_ext
--inplace` rebuilt the copied extension successfully; the focused TOC selector
passed with `3 passed, 810 deselected`; the nearby SEC extraction/heuristics
selector passed with `9 passed, 804 deselected`; and the full Rust acceleration
suite passed with `813 passed`. Suspicious-boundary diagnostics run through the
explicit migration entrypoint on `full_data_run/year_merged` with
`--max-files 1` processed `229` filings, extracted `1372` items, flagged `225`
items, and reported `2` embedded fail items.

After adding SEC heuristic character-level TOC end-position inference,
`python -m py_compile` passed for the modified heuristic wrapper and migration
test file; `cargo fmt --check` and `cargo check` passed for
`src_rust_migration/rust/lm2011_rust/Cargo.toml`; `python setup.py build_ext
--inplace` rebuilt the copied extension successfully; the focused TOC
end-position selector passed with `3 passed, 813 deselected`; the nearby SEC
extraction/heuristics selector passed with `12 passed, 804 deselected`; and the
full Rust acceleration suite passed with `816 passed`. Suspicious-boundary
diagnostics run through the explicit migration entrypoint on
`full_data_run/year_merged` with `--max-files 1` processed `229` filings,
extracted `1372` items, flagged `225` items, and reported `2` embedded fail
items.

After adding SEC pagination artifact removal, `python -m py_compile` passed for
the modified heuristic wrapper and migration test file; `cargo fmt --check` and
`cargo check` passed for `src_rust_migration/rust/lm2011_rust/Cargo.toml`;
`python setup.py build_ext --inplace` rebuilt the copied extension
successfully; the focused pagination selector passed with `3 passed, 816
deselected`; the nearby SEC extraction/heuristics selector passed with `15
passed, 804 deselected`; and the full Rust acceleration suite passed with `819
passed`. A real-shard parity probe over
`full_data_run/year_merged/1995.parquet` checked `274` ASCII filings with zero
Rust/Python pagination mismatches. Suspicious-boundary diagnostics run through
the explicit migration entrypoint on `full_data_run/year_merged` with
`--max-files 1` processed `229` filings, extracted `1372` items, flagged `225`
items, and reported `2` embedded fail items.

After adding SEC trailing Part-marker trimming, `python -m py_compile` passed
for the modified heuristic wrapper and migration test file; `cargo fmt --check`
and `cargo check` passed for `src_rust_migration/rust/lm2011_rust/Cargo.toml`;
`python setup.py build_ext --inplace` rebuilt the copied extension
successfully; the focused trim selector passed with `3 passed, 819 deselected`;
the nearby SEC extraction/heuristics selector passed with `18 passed, 804
deselected`; and the full Rust acceleration suite passed with `822 passed`. A
real-shard parity probe over `full_data_run/year_merged/1995.parquet` checked
`274` ASCII filings with zero Rust/Python trim mismatches. Suspicious-boundary
diagnostics run through the explicit migration entrypoint on
`full_data_run/year_merged` with `--max-files 1` processed `229` filings,
extracted `1372` items, flagged `225` items, and reported `2` embedded fail
items.

After adding SEC reserved-stub end detection, `python -m py_compile` passed for
the modified heuristic wrapper and migration test file; `cargo fmt --check` and
`cargo check` passed for `src_rust_migration/rust/lm2011_rust/Cargo.toml`;
`python setup.py build_ext --inplace` rebuilt the copied extension
successfully; the focused reserved-stub selector passed with `3 passed, 822
deselected`; the nearby SEC extraction/heuristics selector passed with `21
passed, 804 deselected`; and the full Rust acceleration suite passed with `825
passed`. A real-shard parity probe over
`full_data_run/year_merged/1995.parquet` checked `274` ASCII filings with zero
Rust/Python reserved-stub mismatches. Suspicious-boundary diagnostics run
through the explicit migration entrypoint on `full_data_run/year_merged` with
`--max-files 1` processed `229` filings, extracted `1372` items, flagged `225`
items, and reported `2` embedded fail items.

After adding SEC line-start Item heading matching, `python -m py_compile`
passed for the modified heuristic wrapper and migration test file; `cargo
fmt --check` and `cargo check` passed for
`src_rust_migration/rust/lm2011_rust/Cargo.toml`; `python setup.py build_ext
--inplace` rebuilt the copied extension successfully; the focused line-start
selector passed with `3 passed, 825 deselected`; the nearby SEC
extraction/heuristics selector passed with `24 passed, 804 deselected`; and the
full Rust acceleration suite passed with `828 passed`. A real-shard parity
probe over `full_data_run/year_merged/1995.parquet` checked `105410` ASCII
lines with zero Rust/Python line-start mismatches. Suspicious-boundary
diagnostics run through the explicit migration entrypoint on
`full_data_run/year_merged` with `--max-files 1` processed `229` filings,
extracted `1372` items, flagged `225` items, and reported `2` embedded fail
items.

After adding SEC cross-reference prefix detection, `python -m py_compile`
passed for the modified heuristic wrapper and migration test file; `cargo
fmt --check` and `cargo check` passed for
`src_rust_migration/rust/lm2011_rust/Cargo.toml`; `python setup.py build_ext
--inplace` rebuilt the copied extension successfully; the focused prefix
selector passed with `3 passed, 828 deselected`; the nearby SEC
extraction/heuristics selector passed with `6 passed, 825 deselected`; and the
full Rust acceleration suite passed with `831 passed`. The Rust helper is used
for ASCII prefixes and preserves the Python bool contract, 80-character tail
window, cross-reference lead phrases, flexible `in\s+part ...` whitespace,
ASCII word-boundary behavior, and non-ASCII fallback. A real-shard parity probe
over `full_data_run/year_merged/1995.parquet` checked `648` ASCII prefixes with
zero Rust/Python cross-reference prefix mismatches. Suspicious-boundary
diagnostics run through the explicit migration entrypoint on
`full_data_run/year_merged` with `--max-files 1` processed `229` filings,
extracted `1372` items, flagged `225` items, and reported `2` embedded fail
items.

After adding SEC compound item-line detection, `python -m py_compile` passed
for the modified heuristic/diagnostics wrappers and migration test file; `cargo
fmt --check` and `cargo check` passed for
`src_rust_migration/rust/lm2011_rust/Cargo.toml`; `python setup.py build_ext
--inplace` rebuilt the copied extension successfully; the focused compound-item
selector passed with `3 passed, 831 deselected`; the nearby SEC
extraction/heuristics selector passed with `9 passed, 825 deselected`; and the
full Rust acceleration suite passed with `834 passed`. The Rust helper is used
for ASCII heading lines from extraction and suspicious-boundary diagnostics, and
preserves the Python bool contract, `ITEM` word boundaries, flexible
whitespace, digit or roman item tokens with an optional single-letter suffix,
final word-boundary rejection, and non-ASCII fallback. A real-shard parity probe
over `full_data_run/year_merged/1995.parquet` checked `495031` ASCII lines with
zero Rust/Python compound-item mismatches. Suspicious-boundary diagnostics run
through the explicit migration entrypoint on `full_data_run/year_merged` with
`--max-files 1` processed `229` filings, extracted `1372` items, flagged `225`
items, and reported `2` embedded fail items.

After adding SEC heading-suffix prose detection, `python -m py_compile` passed
for the modified heuristic wrapper and migration test file; `cargo fmt --check`
and `cargo check` passed for `src_rust_migration/rust/lm2011_rust/Cargo.toml`;
`python setup.py build_ext --inplace` rebuilt the copied extension
successfully; the focused heading-suffix selector passed with `3 passed, 834
deselected`; the nearby SEC extraction/heuristics selector passed with `12
passed, 825 deselected`; and the full Rust acceleration suite passed with `837
passed`. The Rust helper is used for ASCII item-candidate suffixes and
preserves the Python bool contract, ASCII whitespace trim behavior, leading
`space/tab/colon/hyphen` stripping, the 160-character head window, anchored
cross-reference suffix phrases, `ITEM` and `PART` word-boundary checks,
sentence-break detection, lowercase-initial word ratio, comma/word-count
heuristic, and non-ASCII fallback. A real-shard parity probe over
`full_data_run/year_merged/1995.parquet` checked `2956` ASCII item-candidate
suffixes with zero Rust/Python heading-suffix mismatches. Suspicious-boundary
diagnostics run through the explicit migration entrypoint on
`full_data_run/year_merged` with `--max-files 1` processed `229` filings,
extracted `1372` items, flagged `225` items, and reported `2` embedded fail
items.

After adding SEC Part-marker heading classification, `python -m py_compile`
passed for the modified heuristic wrapper and migration test file; `cargo
fmt --check` and `cargo check` passed for
`src_rust_migration/rust/lm2011_rust/Cargo.toml`; `python setup.py build_ext
--inplace` rebuilt the copied extension successfully; the focused Part-marker
heading selector passed with `3 passed, 837 deselected`; the nearby SEC
extraction/heuristics selector passed with `15 passed, 825 deselected`; and the
full Rust acceleration suite passed with `840 passed`. The Rust helper is used
for ASCII lines and preserves the Python `re.Match`-offset contract, prefix
bullet/non-bullet rejection, combined `PART ... ITEM ...` handling,
comma/alphanumeric separator rejection, 10-character item-start threshold,
punctuation and 80-character suffix rejection, uppercase-letter ratio, short
non-cross-reference suffix acceptance, and non-ASCII fallback. A real-shard
parity probe over `full_data_run/year_merged/1995.parquet` checked `1213` ASCII
Part-marker matches with zero Rust/Python classification mismatches.
Suspicious-boundary diagnostics run through the explicit migration entrypoint on
`full_data_run/year_merged` with `--max-files 1` processed `229` filings,
extracted `1372` items, flagged `225` items, and reported `2` embedded fail
items.

After adding SEC pageish-line detection, `python -m py_compile` passed for the
modified heuristic wrapper and migration test file; `cargo fmt --check` and
`cargo check` passed for `src_rust_migration/rust/lm2011_rust/Cargo.toml`;
`python setup.py build_ext --inplace` rebuilt the copied extension
successfully; the focused pageish selector passed with `3 passed, 840
deselected`; the nearby SEC extraction/heuristics selector passed with `18
passed, 825 deselected`; and the full Rust acceleration suite passed with `843
passed`. The Rust helper is used for ASCII lines and preserves the Python bool
contract, ASCII `strip()`-then-match behavior, 1-4 digit page numbers,
hyphen-wrapped page numbers, 1-6 roman page markers, `page N` / `page N of M`
forms, rejection of `page1` without whitespace after `page`, and non-ASCII
fallback. A real-shard parity probe over `full_data_run/year_merged/1995.parquet`
checked `495031` ASCII lines with zero Rust/Python pageish-line mismatches.
Suspicious-boundary diagnostics run through the explicit migration entrypoint on
`full_data_run/year_merged` with `--max-files 1` processed `229` filings,
extracted `1372` items, flagged `225` items, and reported `2` embedded fail
items.

After adding SEC Part-prefix helper detection, `python -m py_compile` passed
for the modified heuristic wrapper and migration test file; `cargo fmt --check`
and `cargo check` passed for `src_rust_migration/rust/lm2011_rust/Cargo.toml`;
`python setup.py build_ext --inplace` rebuilt the copied extension
successfully; the focused Part-prefix selector passed with `3 passed, 843
deselected`; the nearby SEC extraction/heuristics selector passed with `21
passed, 825 deselected`; and the full Rust acceleration suite passed with `846
passed`. The Rust helpers are used for ASCII candidate prefixes and preserve
the Python `_prefix_is_part_only` bool contract, `_prefix_part_tail` `str |
None` contract with uppercase roman output, leading/trailing ASCII whitespace,
optional colon/hyphen for part-only prefixes, tail-only matching after start or
whitespace/`:;,.'` separators, rejection of invalid roman parts and suffixed
tails, and non-ASCII fallback. A real-shard parity probe over
`full_data_run/year_merged/1995.parquet` checked `4879` ASCII item-candidate
prefixes with zero Rust/Python mismatches across both helpers.
Suspicious-boundary diagnostics run through the explicit migration entrypoint on
`full_data_run/year_merged` with `--max-files 1` processed `229` filings,
extracted `1372` items, flagged `225` items, and reported `2` embedded fail
items.

After adding FinBERT bucket-length tuning edge recommendation, `python -m
py_compile` passed for the modified tuning wrapper and migration test file;
`cargo fmt`, `cargo fmt --check`, and `cargo check` passed for
`src_rust_migration/rust/lm2011_rust/Cargo.toml`; `python setup.py build_ext
--inplace` rebuilt the copied extension successfully with the known nonfatal
setuptools logging and Cargo readonly-cache warnings; the focused
bucket-length selector passed with `14 passed, 836 deselected`; and the full
Rust acceleration suite passed with `850 passed`. The Rust helper preserves the
Python `_recommend_edge` `(edge, reason)` contract, `keep_current` and
`no_rows` policy outcomes, quantile-ceiling/safety-margin rounding, positive
rounding-multiple validation, current-edge caps, and lower-bound enforcement.
SEC suspicious-boundary diagnostics were not run because this slice is limited
to FinBERT bucket-length tuning and does not modify SEC extraction or boundary
heuristics.

After adding SEC EDGAR metadata stripping, `python -m py_compile` passed for
the modified extraction wrapper and migration test file; `cargo fmt`,
`cargo fmt --check`, and `cargo check` passed for
`src_rust_migration/rust/lm2011_rust/Cargo.toml`; `python setup.py build_ext
--inplace` rebuilt the copied extension successfully with the known nonfatal
setuptools logging and Cargo readonly-cache warnings; the focused
strip-edgar-metadata selector passed with `3 passed, 850 deselected`; the
nearby SEC extraction selector passed with `6 passed, 847 deselected`; and the
full Rust acceleration suite passed with `853 passed`. The Rust helper is used
for ASCII filing bodies and preserves the Python regex contract for complete
metadata block removal, mixed-case same-tag closers, exact
hyphen/underscore-tag backreference behavior, leading synthetic-header suffix
stripping, truncated leading-header removal, and non-ASCII fallback.
Suspicious-boundary diagnostics run through the explicit migration entrypoint
on `full_data_run/year_merged` with `--max-files 1` processed `229` filings,
extracted `1372` items, flagged `225` items, and reported `2` embedded fail
items. A matched run with `_strip_edgar_metadata` forced back to the Python
implementation produced identical suspicious CSV and report hashes, so this
slice did not increase bounded diagnostic false positives.

After wiring LM2011 Refinitiv document-ownership doc-filing artifact reads to
the existing Rust-backed batch `KYPERMNO` normalizer, `python -m py_compile`
passed for the modified pipeline module and migration test file; `cargo fmt`,
`cargo fmt --check`, and `cargo check` passed for
`src_rust_migration/rust/lm2011_rust/Cargo.toml`; and the focused
doc-ownership artifact/batch selector passed with `5 passed, 851 deselected`.
The reader still accepts lowercase `kypermno` or uppercase `KYPERMNO` artifact
columns, keeps duplicate-`doc_id` validation in Python, preserves the eager
parquet artifact boundary, and uses the Python batch normalizer when the Rust
extension is unavailable. SEC suspicious-boundary diagnostics were not run
because this slice does not modify SEC extraction or boundary heuristics.

After adding item-text cleaning batch activation-status and audit-period
classifiers, `python -m py_compile` passed for the modified item-cleaning
module and migration test file; `cargo fmt`, `cargo fmt --check`, and `cargo
check` passed for `src_rust_migration/rust/lm2011_rust/Cargo.toml`; `python
setup.py build_ext --inplace` rebuilt the copied extension successfully with
the known nonfatal setuptools logging and Cargo readonly-cache warnings; the
focused item-cleaning migration selector passed with `8 passed, 851
deselected`; and the standalone item-text cleaning suite passed with `15
passed`. The full Rust acceleration suite also passed with `859 passed`. The
new batch expressions replace the remaining `map_elements`
activation-status and audit-period calls in scope diagnostics and manual-audit
sample preparation while preserving Python batch fallback and existing output
schemas. SEC suspicious-boundary diagnostics were not run because this slice
does not modify SEC extraction or boundary heuristics.

After adding FinBERT section-universe and sentence-provenance batch
normalization, `python -m py_compile` passed for the modified item-cleaning,
FinBERT dataset, sentence-provenance modules, and migration test file; `cargo
fmt`, `cargo fmt --check`, and `cargo check` passed for
`src_rust_migration/rust/lm2011_rust/Cargo.toml`; `python setup.py build_ext
--inplace` rebuilt the copied extension successfully with the known nonfatal
setuptools logging and Cargo readonly-cache warnings; the focused migration
selector passed with `4 passed, 859 deselected`; and targeted downstream
FinBERT gates run with `PYTHONPATH=src_rust_migration` passed with `2 passed`.
The full Rust acceleration suite also passed with `863 passed`. The section
loader now uses the CCM batch form expressions for
`document_type_raw` and `document_type_normalized`, and sentence provenance now
uses the new Rust-backed benchmark item-code text-scope batch expression while
preserving Python batch fallback. SEC suspicious-boundary diagnostics were not
run because this slice does not modify SEC extraction or boundary heuristics.

After adding LM2011 previous-month-end and quarter-start batch date
expressions, `python -m py_compile` passed for the modified LM2011 pipeline,
regression, validation-audit modules, and migration test file; `cargo fmt`,
`cargo fmt --check`, and `cargo check` passed for
`src_rust_migration/rust/lm2011_rust/Cargo.toml`; `python setup.py build_ext
--inplace` rebuilt the copied extension successfully with the known nonfatal
setuptools logging and Cargo readonly-cache warnings; the focused date-bucket
selector passed with `5 passed, 860 deselected`; and targeted downstream
LM2011 validation/regression gates run with `PYTHONPATH=src_rust_migration`
passed with `23 passed, 14 deselected, 1 warning`. The full Rust acceleration
suite also passed with `865 passed`. The batch expressions replace scalar
`map_elements` calls in event-panel prior-month price joins, Phase 0 validation
SUE announcement prior-month joins, and quarterly Fama-MacBeth grouping while
preserving null propagation and Python fallback. SEC suspicious-boundary
diagnostics were not run because this slice does not modify SEC extraction or
boundary heuristics.

After adding the multi-surface audit stable sort-key batch helper, `python -m
py_compile` passed for the modified audit module and migration test file;
`cargo fmt`, `cargo fmt --check`, and `cargo check` passed for
`src_rust_migration/rust/lm2011_rust/Cargo.toml`; `python setup.py build_ext
--inplace` rebuilt the copied extension successfully with the known nonfatal
setuptools logging and Cargo readonly-cache warnings; the focused stable-sort
selector passed with `6 passed, 863 deselected`; the standalone multi-surface
audit suite run with `PYTHONPATH=src_rust_migration` passed with `4 passed`;
and the full Rust acceleration suite passed with `869 passed`. The batch path
replaces `_with_stable_sort_key`'s scalar `map_elements` callback with a
Rust-backed `map_batches` expression while preserving digit extraction parity,
no-digit/non-ASCII Python fallback, null sort keys, and the existing eager
DataFrame output contract. SEC suspicious-boundary diagnostics were not run
because this slice does not modify SEC extraction or boundary heuristics.

After adding multi-surface audit boundary-snippet and snippet-delta risk batch
helpers, `python -m py_compile` passed for the modified audit module and
migration test file; `cargo fmt`, `cargo fmt --check`, and `cargo check`
passed for `src_rust_migration/rust/lm2011_rust/Cargo.toml`; `python setup.py
build_ext --inplace` rebuilt the copied extension successfully with the known
nonfatal setuptools logging and Cargo readonly-cache warnings; the combined
multisurface selector passed with `13 passed, 861 deselected`; the standalone
multi-surface audit suite run with `PYTHONPATH=src_rust_migration` passed with
`4 passed`; and the full Rust acceleration suite passed with `874 passed`. The
batch path replaces item-boundary scoring's scalar struct `map_elements`
callbacks while preserving boundary leak/TOC-cluster semantics,
whitespace-normalized snippet-delta comparison, null-as-empty snippet
handling, Python fallback, and the existing eager DataFrame output contract.
The same pass replaces control-case peer-group membership with native Polars
struct membership, removing the last scalar `map_elements` call from
`multisurface_audit.py`. SEC suspicious-boundary diagnostics were not run
because this slice does not modify SEC extraction or boundary heuristics.

After wiring LM2011 text-feature base normalized-form construction to the
existing Rust-backed batch form expression, `python -m py_compile` passed for
the modified text-feature module and migration test file; `cargo fmt --check`
and `cargo check` passed for `src_rust_migration/rust/lm2011_rust/Cargo.toml`;
the focused form/text selector passed with `9 passed, 868 deselected`; the
broader text-feature selector passed with `10 passed, 867 deselected`; and the
full Rust acceleration suite passed with `877 passed`. The eager
`_ensure_normalized_form` and LazyFrame `_build_text_base_lf` paths now reuse
the batch form expression instead of scalar
`map_elements(normalize_lm2011_form_value)`, while preserving
pre-normalized-column handling, normalized-form output contracts, Python batch
fallback, and LazyFrame semantics. No new Rust source was needed for this
slice because it reuses the already-built batch form normalizer. SEC
suspicious-boundary diagnostics were not run because this slice does not
modify SEC extraction or boundary heuristics.

After adding the LM2011 batch token-count expression and wiring it into the
post-Refinitiv FinBERT-visible prefix source builder, `python -m py_compile`
passed for the modified text-feature module, runner module, and migration test
file; `cargo fmt`, `cargo fmt --check`, and `cargo check` passed for
`src_rust_migration/rust/lm2011_rust/Cargo.toml`; the focused token-count
selector passed with `4 passed, 877 deselected`; the targeted LM2011 sample
runner gate run with `PYTHONPATH=src_rust_migration` passed with `18 passed,
69 deselected`; and the full Rust acceleration suite passed with `881 passed`.
The expression batches non-null Polars text through the existing Rust
count-only tokenizer, preserves the previous scalar `map_elements` null
behavior, and replaces original/visible-prefix document text token-count
callbacks in `lm2011_sample_post_refinitiv_runner.py`. No new Rust source was
needed for this slice because it reuses the existing count-only token batch
export. SEC suspicious-boundary diagnostics were not run because this slice
does not modify SEC extraction or boundary heuristics.

After replacing the final scalar callbacks in the copied package tree,
`python -m py_compile` passed for the modified item-cleaning module, runner
module, and migration test file; `cargo fmt`, `cargo fmt --check`, and `cargo
check` passed for `src_rust_migration/rust/lm2011_rust/Cargo.toml`; `python
setup.py build_ext --inplace` rebuilt the copied extension successfully with
the known nonfatal setuptools logging and Cargo readonly-cache warnings; the
focused item-cleaning/CSV selector passed with `6 passed, 878 deselected`; the
targeted extension runner gate run with `PYTHONPATH=src_rust_migration` passed
with `13 passed, 74 deselected`; and the full Rust acceleration suite passed
with `884 passed`. Item-text cleaning manual-audit fallback sampling now uses
the existing batch audit-period expression instead of
`map_elements(_audit_period_py)`, and the LM2011 extension CSV companion writer
serializes optional signal-input list columns through a Rust-backed batch JSON
helper with Python fallback while preserving the prior `json.dumps` output
contract. A repository scan found no remaining `map_elements` calls under
`src_rust_migration/thesis_pkg`. SEC suspicious-boundary diagnostics were not
run because this slice does not modify SEC extraction or boundary heuristics.

After adding a column-oriented LM2011 cross-sectional OLS design entrypoint,
`python -m py_compile` passed for the modified regression module and migration
test file; `cargo fmt`, `cargo fmt --check`, and `cargo check` passed for
`src_rust_migration/rust/lm2011_rust/Cargo.toml`; `python setup.py build_ext
--inplace` rebuilt the copied extension successfully with the known nonfatal
setuptools logging and Cargo readonly-cache warnings; the focused design
selector passed with `3 passed, 881 deselected`; the targeted LM2011
regression gate run with `PYTHONPATH=src_rust_migration` passed with `16
passed, 12 deselected, 1 warning`; and the full Rust acceleration suite passed
with `884 passed`. `_cross_section_design_rows` now feeds dependent,
industry, and regressor columns directly to Rust, preserving the older row-dict
Rust helper as a compatibility fallback and the Python implementation as the
final fallback. SEC suspicious-boundary diagnostics were not run because this
slice does not modify SEC extraction or boundary heuristics.

After adding a column-oriented LM2011 monthly strategy factor-loading
entrypoint, `python -m py_compile` passed for the modified pipeline module and
migration test file; `cargo fmt`, `cargo fmt --check`, and `cargo check`
passed for `src_rust_migration/rust/lm2011_rust/Cargo.toml`; `python setup.py
build_ext --inplace` rebuilt the copied extension successfully with the known
nonfatal setuptools logging and Cargo readonly-cache warnings; the focused
strategy factor-loading selector passed with `3 passed, 881 deselected`; and
the targeted LM2011 pipeline gate run with `PYTHONPATH=src_rust_migration`
passed with `11 passed, 72 deselected, 1 warning`. The full Rust acceleration
suite passed with `884 passed`. `_strategy_factor_loading_rows` now feeds
signal, return, and FF4 factor columns directly to Rust, preserving the older
row-dict Rust helper as a compatibility fallback and the Python implementation
as the final fallback. SEC suspicious-boundary diagnostics were not run because
this slice does not modify SEC extraction or boundary heuristics.

After adding a column-oriented LM2011 Table IA.II monthly strategy result-row
entrypoint, `python -m py_compile` passed for the modified regression module
and migration test file; `cargo fmt`, `cargo fmt --check`, and `cargo check`
passed for `src_rust_migration/rust/lm2011_rust/Cargo.toml`; `python setup.py
build_ext --inplace` rebuilt the copied extension successfully with the known
nonfatal setuptools logging and Cargo readonly-cache warnings; the focused
Table IA.II selector passed with `3 passed, 881 deselected`; the targeted
LM2011 regression gate run with `PYTHONPATH=src_rust_migration` passed with `4
passed, 24 deselected`; and the targeted LM2011 pipeline gate passed with `11
passed, 72 deselected, 1 warning`. The full Rust acceleration suite passed
with `884 passed`. `_table_ia_ii_result_rows` now feeds signal and FF4 summary
value columns directly to Rust, preserving the older row-dict Rust helper as a
compatibility fallback and the Python implementation as the final fallback.
SEC suspicious-boundary diagnostics were not run because this slice does not
modify SEC extraction or boundary heuristics.

After adding a column-oriented LM2011 event-window regression metric entrypoint,
`python -m py_compile` passed for the modified pipeline module and migration
test file; `cargo fmt`, `cargo fmt --check`, and `cargo check` passed for
`src_rust_migration/rust/lm2011_rust/Cargo.toml`; `python setup.py build_ext
--inplace` rebuilt the copied extension successfully with the known nonfatal
setuptools logging and Cargo readonly-cache warnings; the focused
regression-window selector passed with `3 passed, 881 deselected`; and the
targeted LM2011 event-window pipeline gate run with
`PYTHONPATH=src_rust_migration` passed with `7 passed, 76 deselected`. The
full Rust acceleration suite passed with `884 passed`.
`_regression_metrics_from_window_rows` now feeds sorted document and
regression columns directly to Rust, preserving the older row-dict Rust helper
as a compatibility fallback and the Python implementation as the final
fallback. SEC suspicious-boundary diagnostics were not run because this slice
does not modify SEC extraction or boundary heuristics.

After adding a column-oriented LM2011 streaming text-feature pass-1 entrypoint,
`python -m py_compile` passed for the modified text-feature module and
migration test file; `cargo fmt`, `cargo fmt --check`, and `cargo check`
passed for `src_rust_migration/rust/lm2011_rust/Cargo.toml`; `python setup.py
build_ext --inplace` rebuilt the copied extension successfully with the known
nonfatal setuptools logging and Cargo readonly-cache warnings; the focused
pass-1 selector passed with `4 passed, 880 deselected`; and the targeted
LM2011 text-feature pipeline gate run with `PYTHONPATH=src_rust_migration`
passed with `21 passed, 62 deselected`. The full Rust acceleration suite
passed with `884 passed`. `_prepare_pass1_rows` now feeds metadata columns and
text values directly to Rust, preserving the older row-dict Rust helper as a
compatibility fallback and the Python implementation as the final fallback.
SEC suspicious-boundary diagnostics were not run because this slice does not
modify SEC extraction or boundary heuristics.

After adding a column-oriented LM2011 document-stat preparation entrypoint,
`python -m py_compile` passed for the modified text-feature module and
migration test file; `cargo fmt`, `cargo fmt --check`, and `cargo check`
passed for `src_rust_migration/rust/lm2011_rust/Cargo.toml`; `python setup.py
build_ext --inplace` rebuilt the copied extension successfully with the known
nonfatal setuptools logging and Cargo readonly-cache warnings; the focused
document-stat selector passed with `4 passed, 880 deselected`; and the targeted
LM2011 text-feature pipeline gate run with `PYTHONPATH=src_rust_migration`
passed with `21 passed, 62 deselected`. The full Rust acceleration suite
passed with `884 passed`. `_prepare_document_stats` now feeds metadata columns,
doc IDs, and text values directly to Rust, preserving the older row-dict Rust
helper as a compatibility fallback and the Python implementation as the final
fallback. SEC suspicious-boundary diagnostics were not run because this slice
does not modify SEC extraction or boundary heuristics.

After adding a column-oriented LM2011 streaming text-feature pass-2 entrypoint,
`python -m py_compile` passed for the modified text-feature module and
migration test file; `cargo fmt`, `cargo fmt --check`, and `cargo check`
passed for `src_rust_migration/rust/lm2011_rust/Cargo.toml`; `python setup.py
build_ext --inplace` rebuilt the copied extension successfully with the known
nonfatal setuptools logging and Cargo readonly-cache warnings; the focused
pass-1 feature-row selector passed with `5 passed, 881 deselected`; and the
targeted LM2011 text-feature pipeline gate run with
`PYTHONPATH=src_rust_migration` passed with `21 passed, 62 deselected`. The
full Rust acceleration suite passed with `886 passed`.
`_feature_rows_from_pass1_frame` now feeds pass-1 parquet columns directly to
Rust, preserving the older row-dict Rust helper as a compatibility fallback
and the Python implementation as the final fallback. SEC suspicious-boundary
diagnostics were not run because this slice does not modify SEC extraction or
boundary heuristics.

After adding a column-oriented SEC no-item stats aggregation entrypoint,
`python -m py_compile` passed for the modified SEC pipeline module and
migration test file; `cargo fmt`, `cargo fmt --check`, and `cargo check`
passed for `src_rust_migration/rust/lm2011_rust/Cargo.toml`; `python setup.py
build_ext --inplace` rebuilt the copied extension successfully with the known
nonfatal setuptools logging and Cargo readonly-cache warnings; the focused
no-item migration selector passed with `4 passed, 883 deselected`; and the full
Rust acceleration suite passed with `887 passed`.
`_aggregate_no_item_stats_rows` now feeds document-type/count/weighted-length
columns directly to Rust, preserving the older row-dict Rust helper as a
compatibility fallback and the Python implementation as the final fallback.
SEC suspicious-boundary diagnostics were not run because this slice does not
modify SEC extraction or boundary heuristics.

After adding a column-oriented Refinitiv LSEG provider response-frame
fingerprint entrypoint, `python -m py_compile` passed for the modified provider
module and migration test file; `cargo fmt`, `cargo fmt --check`, and
`cargo check` passed for `src_rust_migration/rust/lm2011_rust/Cargo.toml`;
`python setup.py build_ext --inplace` rebuilt the copied extension successfully
with the known nonfatal setuptools logging and Cargo readonly-cache warnings;
the focused LSEG provider fingerprint selector passed with `4 passed, 885
deselected`; and the full Rust acceleration suite passed with `889 passed`.
`_frame_fingerprint` now feeds frame column names and per-column values directly
to Rust, preserving the older row-dict Rust helper as a compatibility fallback
and the Python implementation as the final fallback.

After adding a column-oriented Refinitiv ownership authority allowlist key
entrypoint, `python -m py_compile` passed for the modified authority/provider
modules and migration test file; `cargo fmt`, `cargo fmt --check`, and
`cargo check` passed for `src_rust_migration/rust/lm2011_rust/Cargo.toml`;
`python setup.py build_ext --inplace` rebuilt the copied extension successfully
with the known nonfatal setuptools logging and Cargo readonly-cache warnings;
the focused allowlist selector passed with `4 passed, 885 deselected`; and the
full Rust acceleration suite passed with `889 passed`.
`_allowlist_keys` now feeds `KYPERMNO` and `candidate_ric` columns directly to
Rust, preserving the older row-dict Rust helper as a compatibility fallback and
the Python implementation as the final fallback.

After adding a column-oriented FinBERT sentence-confusion review
majority-bucket metric entrypoint, `python -m py_compile` passed for the
modified confusion-review module and migration test file; cargo format/check
and cargo check passed for `src_rust_migration/rust/lm2011_rust/Cargo.toml`;
the extension rebuild passed with the known nonfatal setuptools logging and
Cargo readonly-cache warnings; the focused bucket-metric selector passed with
`4 passed, 886 deselected`; and the full Rust acceleration suite passed with
`890 passed`.
`_bucket_metric_rows` now feeds majority-bucket, confusion-cell, and optional
sample-weight columns directly to Rust, preserving the older row-dict Rust
helper as a compatibility fallback and the Python implementation as the final
fallback.

After adding a column-oriented FinBERT sentence-length report frame-record
entrypoint, `python -m py_compile` passed for the modified sentence-length
module and migration test file; cargo format/check and cargo check passed for
`src_rust_migration/rust/lm2011_rust/Cargo.toml`; the extension rebuild passed
with the known nonfatal setuptools logging and Cargo readonly-cache warnings;
the focused sentence-length selector passed with `4 passed, 887 deselected`;
and the full Rust acceleration suite passed with `891 passed`.
`_frame_to_records` now feeds frame column names and per-column values directly
to Rust, preserving the older row-dict Rust helper as a compatibility fallback
and the Python implementation as the final fallback.

After adding a column-oriented FinBERT tail document-surface entrypoint,
`python -m py_compile` passed for the modified tail-feature module and
migration test file; cargo format/check and cargo check passed for
`src_rust_migration/rust/lm2011_rust/Cargo.toml`; the extension rebuild passed
with the known nonfatal setuptools logging and Cargo readonly-cache warnings;
the focused FinBERT tail selector passed with `4 passed, 888 deselected`; the
standalone tail-feature suite explicitly run with
`PYTHONPATH=src_rust_migration` passed with `4 passed`; and the full Rust
acceleration suite passed with `892 passed`.
`build_finbert_tail_doc_surface` now feeds sentence-score column names and
per-column values directly to Rust, preserving the older row-dict Rust helper
as a compatibility fallback and the Python implementation as the final
fallback; the Polars LazyFrame tail builder remains unchanged for large-data
pipeline use.

After adding a column-oriented FinBERT sentence-preprocessing manifest-count
entrypoint, `python -m py_compile` passed for the modified preprocessing module
and migration test file; cargo format/check and cargo check passed for
`src_rust_migration/rust/lm2011_rust/Cargo.toml`; the extension rebuild passed
with the known nonfatal setuptools logging and Cargo readonly-cache warnings;
the focused preprocessing manifest-count selector passed with `4 passed, 889
deselected`; and the full Rust acceleration suite passed with `893 passed`.
`_manifest_counts` now feeds summary column names and per-column values
directly to Rust, preserving the older row-dict Rust helper as a compatibility
fallback and the Python implementation as the final fallback.

After adding column-oriented LM2011 extension result-row, quarterly fit-row,
and skipped-quarter diagnostic row entrypoints, `python -m py_compile` passed
for the modified extension module and migration test file; cargo format/check
and cargo check passed for `src_rust_migration/rust/lm2011_rust/Cargo.toml`;
the extension rebuild passed with the known nonfatal setuptools logging and
Cargo readonly-cache warnings; the focused extension-row selector passed with
`12 passed, 884 deselected`; and the full Rust acceleration suite passed with
`896 passed`. `_convert_lm2011_table_results_to_extension_rows`,
`_extension_quarterly_fit_rows`, and `_extension_skipped_quarter_rows` now feed
column names and per-column values directly to Rust, preserving their older
row-dict Rust helpers as compatibility fallbacks and the Python
implementations as final fallbacks.

After adding a column-oriented FinBERT bucket-length tuning summary-row
entrypoint, `python -m py_compile` passed for the modified bucket-length module
and migration test file; cargo format/check and cargo check passed for
`src_rust_migration/rust/lm2011_rust/Cargo.toml`; the extension rebuild passed
with the known nonfatal setuptools logging and Cargo readonly-cache warnings;
the focused bucket-length summary selector passed with `4 passed, 893
deselected`; and the full Rust acceleration suite passed with `897 passed`.
`_bucket_summary_rows` now feeds grouped summary column names and per-column
values directly to Rust, preserving the older row-dict Rust helper as a
compatibility fallback and the Python implementation as the final fallback.

After adding a column-oriented FinBERT benchmark dataset audit-share row
entrypoint, `python -m py_compile` passed for the modified dataset module and
migration test file; cargo format/check and cargo check passed for
`src_rust_migration/rust/lm2011_rust/Cargo.toml`; the extension rebuild passed
with the known nonfatal setuptools logging and Cargo readonly-cache warnings;
the focused dataset share-row selector passed with `4 passed, 894 deselected`;
and the full Rust acceleration suite passed with `898 passed`. `_share_rows`
now feeds sorted grouped frame column names and per-column values directly to
Rust, preserving the older row-dict Rust helper as a compatibility fallback
and the Python implementation as the final fallback.

After adding a column-oriented FinBERT sentence-confusion review
target-position entrypoint, `python -m py_compile` passed for the modified
confusion-review module and migration test file; cargo format/check and cargo
check passed for `src_rust_migration/rust/lm2011_rust/Cargo.toml`; the
extension rebuild passed with the known nonfatal setuptools logging and Cargo
readonly-cache warnings; the focused target-position selector passed with `5
passed, 894 deselected`; and the full Rust acceleration suite passed with `899
passed`. `_target_positions_for_allocation` now feeds allocation column names
and per-column values directly to Rust, preserving the older row-dict Rust
helper as a compatibility fallback and the Python implementation as the final
fallback.

After adding a column-oriented FinBERT high-confidence sentence-example
markdown-renderer entrypoint, `python -m py_compile` passed for the modified
sentence-example module and migration test file; cargo format/check and cargo
check passed for `src_rust_migration/rust/lm2011_rust/Cargo.toml`; the
extension rebuild passed with the known nonfatal setuptools logging and Cargo
readonly-cache warnings; the focused sentence-example markdown selector passed
with `4 passed, 896 deselected`; and the full Rust acceleration suite passed
with `900 passed`. `_render_sample_markdown` now feeds sample and count frame
column names plus per-column values directly to Rust, preserving the older
row-dict Rust helper as a compatibility fallback and the Python implementation
as the final fallback.

After adding a column-oriented FinBERT sentence-confusion review
neighbor-target entrypoint, `python -m py_compile` passed for the modified
confusion-review module and migration test file; cargo format/check and cargo
check passed for `src_rust_migration/rust/lm2011_rust/Cargo.toml`; the
extension rebuild passed with the known nonfatal setuptools logging and Cargo
readonly-cache warnings; the focused neighbor-target selector passed with `5
passed, 897 deselected`; and the full Rust acceleration suite passed with `902
passed`. `_attach_neighbor_context` now feeds sampled frame column names and
per-column values directly to Rust through `_neighbor_target_rows_from_frame`,
preserving the older row-dict Rust helper as a compatibility fallback and the
Python implementation as the final fallback.

After adding a column-oriented FinBERT sentence-confusion review reviewed-case
entrypoint, `python -m py_compile` passed for the modified confusion-review
module and migration test file; cargo format/check and cargo check passed for
`src_rust_migration/rust/lm2011_rust/Cargo.toml`; the extension rebuild passed
with the known nonfatal setuptools logging and Cargo readonly-cache warnings;
the focused reviewed-row selector passed with `5 passed, 899 deselected`; and
the full Rust acceleration suite passed with `904 passed`.
`summarize_finbert_sentence_confusion_review` now feeds sampled frame column
names and per-column values directly to Rust through
`_reviewed_row_dicts_from_frame`, preserving the older row-dict Rust helper as
a compatibility fallback and the Python implementation as the final fallback.

After adding a column-oriented multi-surface audit chunk-record entrypoint,
`python -m py_compile` passed for the modified multi-surface audit module and
migration test file; cargo format/check and cargo check passed for
`src_rust_migration/rust/lm2011_rust/Cargo.toml`; the extension rebuild passed
with the known nonfatal setuptools logging and Cargo readonly-cache warnings;
the focused multi-surface chunk selector passed with `5 passed, 901
deselected`; and the full Rust acceleration suite passed with `906 passed`.
`build_multisurface_audit_pack` now feeds audit-case chunking columns directly
to Rust through `_chunk_records_from_frame`, preserving the older row-dict Rust
helper as a compatibility fallback and the Python implementation as the final
fallback.

After adding a column-oriented LM2011 phase-0 validation Packet A MD&A row
entrypoint, `python -m py_compile` passed for the modified validation-audit
module and migration test file; cargo format/check and cargo check passed for
`src_rust_migration/rust/lm2011_rust/Cargo.toml`; the extension rebuild passed
with the known nonfatal setuptools logging and Cargo readonly-cache warnings;
the focused Packet A MD&A selector passed with `4 passed, 903 deselected`; and
the full Rust acceleration suite passed with `907 passed`.
`_packet_a_mda_rows_from_items` now feeds item `doc_id`/`full_text` columns
directly to Rust, preserving the older row-dict Rust helper as a compatibility
fallback and the Python implementation as the final fallback.

After adding a column-oriented SEC/CCM run-report markdown table entrypoint,
`python -m py_compile` passed for the modified SEC/CCM pipeline module and
migration test file; cargo format/check and cargo check passed for
`src_rust_migration/rust/lm2011_rust/Cargo.toml`; the extension rebuild passed
with the known nonfatal setuptools logging and Cargo readonly-cache warnings;
the focused SEC/CCM markdown selector passed with `7 passed, 901 deselected`;
and the full Rust acceleration suite passed with `908 passed`.
`_to_markdown_table` now feeds clipped frame column names and per-column values
directly to Rust, preserving the older row-dict Rust helper as a compatibility
fallback and the Python implementation as the final fallback.

After adding a column-oriented Refinitiv LSEG lookup request-item entrypoint,
`python -m py_compile` passed for the modified lookup API module and migration
test file; cargo format/check and cargo check passed for
`src_rust_migration/rust/lm2011_rust/Cargo.toml`; the extension rebuild passed
with the known nonfatal setuptools logging and Cargo readonly-cache warnings;
the focused LSEG lookup item-builder selector passed with `4 passed, 905
deselected`; and the full Rust acceleration suite passed with `909 passed`.
`_build_lookup_items` now feeds lookup snapshot column names and per-column
values directly to Rust, preserving the older row-dict Rust helper as a
compatibility fallback and the Python implementation as the final fallback.

After adding a column-oriented Refinitiv LSEG lookup batch-response
normalizer, `python -m py_compile` passed for the modified lookup API module
and migration test file; cargo format/check and cargo check passed for
`src_rust_migration/rust/lm2011_rust/Cargo.toml`; the extension rebuild passed
with the known nonfatal setuptools logging and Cargo readonly-cache warnings;
the focused LSEG lookup selector passed with `8 passed, 902 deselected`; and
the full Rust acceleration suite passed with `910 passed`.
`_normalize_lookup_batch_response` now feeds request-item columns and
standardized response-frame column values directly to Rust, preserving the
older row-dict Rust helper as a compatibility fallback and the Python
implementation as the final fallback.

After adding a column-oriented Refinitiv analyst LSEG actuals batch-response
normalizer, `python -m py_compile` passed for the modified analyst API module
and migration test file; cargo format/check and cargo check passed for
`src_rust_migration/rust/lm2011_rust/Cargo.toml`; the extension rebuild passed
with the known nonfatal setuptools logging and Cargo readonly-cache warnings;
the focused analyst response selector passed with `4 passed, 907 deselected`;
and the full Rust acceleration suite passed with `911 passed`.
`_normalize_analyst_actuals_batch_response` now feeds item instruments,
request-window parameters, and standardized actuals response-frame column
values directly to Rust, preserving the older row-dict Rust helper as a
compatibility fallback and the Python implementation as the final fallback.

After adding a column-oriented Refinitiv analyst LSEG estimates batch-response
normalizer, `python -m py_compile` passed for the modified analyst API module
and migration test file; cargo format/check and cargo check passed for
`src_rust_migration/rust/lm2011_rust/Cargo.toml`; the extension rebuild passed
with the known nonfatal setuptools logging and Cargo readonly-cache warnings;
the focused analyst response selector passed with `5 passed, 907 deselected`;
and the full Rust acceleration suite passed with `912 passed`.
`_normalize_analyst_estimates_batch_response` now feeds item instruments,
request-window parameters, and standardized estimate response-frame column
values directly to Rust, preserving the older row-dict Rust helper as a
compatibility fallback and the Python implementation as the final fallback.

After adding a column-oriented Refinitiv ownership LSEG universe
batch-response normalizer, `python -m py_compile` passed for the modified
ownership API module and migration test file; cargo format/check and cargo
check passed for `src_rust_migration/rust/lm2011_rust/Cargo.toml`; the
extension rebuild passed with the known nonfatal setuptools logging and Cargo
readonly-cache warnings; the focused ownership response selector passed with
`4 passed, 909 deselected`; and the full Rust acceleration suite passed with
`913 passed`. `_normalize_ownership_universe_batch_response` now feeds item
instruments, request-window parameters, and standardized ownership response
columns directly to Rust, preserving the older row-dict Rust helper as a
compatibility fallback and the Python implementation as the final fallback.

After adding a shared column-oriented Refinitiv document-ownership LSEG
batch-response normalizer for exact and fallback stages, `python -m
py_compile` passed for the modified ownership API module and migration test
file; cargo format/check and cargo check passed for
`src_rust_migration/rust/lm2011_rust/Cargo.toml`; the extension rebuild passed
with the known nonfatal setuptools logging and Cargo readonly-cache warnings;
the focused ownership response selector passed with `5 passed, 909
deselected`; and the full Rust acceleration suite passed with `914 passed`.
`_normalize_doc_ownership_response_fast` now feeds item instruments and
standardized ownership response columns directly to Rust for both exact and
fallback wrappers, preserving the older row-dict Rust helper as a
compatibility fallback and the Python implementations as final fallbacks.

After adding a column-oriented Refinitiv ownership LSEG universe request-item
builder, `python -m py_compile` passed for the modified ownership API module
and migration test file; cargo format/check and cargo check passed for
`src_rust_migration/rust/lm2011_rust/Cargo.toml`; the extension rebuild passed
with the known nonfatal setuptools logging and Cargo readonly-cache warnings;
the focused ownership item-builder selector passed with `4 passed, 911
deselected`; and the full Rust acceleration suite passed with `915 passed`.
`_build_ownership_universe_items` now feeds handoff frame columns directly to
Rust for request-field derivation while preserving the older row-dict Rust
helper as a compatibility fallback and the Python implementation as the final
fallback; Python row dict materialization remains for the required
`RequestItem.payload["handoff_row"]` contract.

After adding column-oriented Refinitiv document-ownership LSEG exact and
fallback request-item builders, `python -m py_compile` passed for the modified
ownership API module and migration test file; cargo format/check and cargo
check passed for `src_rust_migration/rust/lm2011_rust/Cargo.toml`; the
extension rebuild passed with the known nonfatal setuptools logging and Cargo
readonly-cache warnings; the focused ownership item-builder selector passed
with `5 passed, 911 deselected`; and the full Rust acceleration suite passed
with `916 passed`. `_build_doc_ownership_exact_items` and
`_build_doc_ownership_fallback_items` now feed request-frame columns directly
to Rust for request-field derivation while preserving the older row-dict Rust
helpers as compatibility fallbacks and the Python implementations as final
fallbacks; Python row dict materialization remains for the required
`RequestItem.payload["request_row"]` contract.

After adding column-oriented Refinitiv analyst LSEG actuals and estimates
request-item builders, `python -m py_compile` passed for the modified analyst
API module and migration test file; cargo format/check and cargo check passed
for `src_rust_migration/rust/lm2011_rust/Cargo.toml`; the extension rebuild
passed with the known nonfatal setuptools logging and Cargo readonly-cache
warnings; the focused analyst item-builder selector passed with `4 passed, 913
deselected`; and the full Rust acceleration suite passed with `917 passed`.
`_build_analyst_actuals_items` and `_build_analyst_estimate_items` now feed
request-frame columns directly to Rust for request-field derivation while
preserving the older row-dict Rust helpers as compatibility fallbacks and the
Python implementations as final fallbacks; Python row dict materialization
remains for the required `RequestItem.payload["request_row"]` contract.

After adding column-oriented Refinitiv document-ownership exact/fallback hit
selection and fallback-request row selection, `python -m py_compile` passed for
the modified doc-ownership module and migration test file; cargo format/check
and cargo check passed for `src_rust_migration/rust/lm2011_rust/Cargo.toml`;
the extension rebuild passed with the known nonfatal setuptools logging and
Cargo readonly-cache warnings; the focused doc-ownership selector passed with
`12 passed, 908 deselected`; and the full Rust acceleration suite passed with
`920 passed`. `_select_exact_hits`, `_select_fallback_hits`, and
`_build_fallback_request_df` now feed frame columns directly to Rust for
selection/reduction while preserving the older row-dict Rust helpers as
compatibility fallbacks and the Python implementations as final fallbacks.

After adding column-oriented Refinitiv document-ownership request-row
construction, `python -m py_compile` passed for the modified doc-ownership
module and migration test file; cargo format/check and cargo check passed for
`src_rust_migration/rust/lm2011_rust/Cargo.toml`; the extension rebuild passed
with the known nonfatal setuptools logging and Cargo readonly-cache warnings;
the focused doc-ownership request selector passed with `4 passed, 917
deselected`; and the full Rust acceleration suite passed with `921 passed`.
`build_refinitiv_lm2011_doc_ownership_requests` now feeds doc-filing,
authority-decision, and authority-exception frame columns directly to Rust
while preserving the older row-dict Rust helper as a compatibility fallback and
the Python implementation as the final fallback.

After adding column-oriented Refinitiv document-ownership
universe-diagnostics summary reduction, `python -m py_compile` passed for the
modified doc-ownership module and migration test file; cargo format/check and
cargo check passed for `src_rust_migration/rust/lm2011_rust/Cargo.toml`; the
extension rebuild passed with the known nonfatal setuptools logging and Cargo
readonly-cache warnings; the focused doc-ownership universe-diagnostics
selector passed with `6 passed, 920 deselected`; and the full Rust
acceleration suite passed with `926 passed`.
`_build_lm2011_doc_ownership_universe_diagnostics` now feeds the normalized
detail frame columns directly to Rust for summary reduction while preserving
the older row-dict Rust helper as a compatibility fallback and the Python
implementation as the final fallback.

After adding column-oriented Refinitiv ownership authority candidate-metric
construction, `python -m py_compile` passed for the modified authority module
and migration test file; cargo format/check and cargo check passed for
`src_rust_migration/rust/lm2011_rust/Cargo.toml`; the extension rebuild passed
with the known nonfatal setuptools logging and Cargo readonly-cache warnings;
the focused authority candidate-metric selector passed with `5 passed, 918
deselected`; and the full Rust acceleration suite passed with `923 passed`.
`build_refinitiv_step1_ownership_authority_tables` now feeds normalized
ownership row-summary and ownership result frame columns directly to Rust for
candidate-metric construction while preserving the older row-dict Rust helper
as a compatibility fallback and the Python implementation as the final
fallback.

After adding the metadata-oriented Refinitiv ownership authority
conventional-component handoff, `python -m py_compile` passed for the modified
authority module and migration test file; cargo format/check and cargo check
passed for `src_rust_migration/rust/lm2011_rust/Cargo.toml`; the extension
rebuild passed with the known nonfatal setuptools logging and Cargo
readonly-cache warnings; the focused conventional-components selector passed
with `4 passed, 920 deselected`; and the full Rust acceleration suite passed
with `924 passed`. `_build_conventional_components` now feeds `candidate_meta`
and `pair_meta` dictionaries directly to Rust for fast-path reducer payload
construction while preserving the older row-dict Rust helper as a compatibility
fallback and the Python implementation as the final fallback.

After adding column-oriented FinBERT sentence confusion-review sample ID/order
assignment, `python -m py_compile` passed for the modified confusion-review
module and migration test file; cargo format/check and cargo check passed for
`src_rust_migration/rust/lm2011_rust/Cargo.toml`; the extension rebuild passed
with the known nonfatal setuptools logging and Cargo readonly-cache warnings;
the focused sample-ID selector passed with `5 passed, 923 deselected`; and the
full Rust acceleration suite passed with `928 passed`. `_finalize_sample_frame`
now feeds the sorted sample frame columns directly to Rust for
`sample_order`/`review_case_id` assignment while preserving the older row-dict
Rust helper as a compatibility fallback and the Python implementation as the
final fallback.

After refactoring the Rust extension source layout, the migration copy keeps
the same Python-visible `_lm2011_rust` module and `#[pymodule]` registration
surface while moving the former monolithic `rust/lm2011_rust/src/lib.rs`
registration block into focused `rust/lm2011_rust/src/py_exports/` modules by
domain. `lib.rs` now contains implementation module declarations,
crate-internal re-exports, and the stable `#[pymodule]` entrypoint that
delegates to `py_exports::register_all`; the export modules cover LM2011 core,
LM2011 analysis, SEC, Refinitiv/LSEG, text processing/item cleaning, FinBERT,
and common I/O/audit helpers. `cargo fmt`, `cargo fmt --check`, and
`cargo check` passed for `src_rust_migration/rust/lm2011_rust/Cargo.toml`; a
Python export smoke confirmed `435` source pyfunctions, `435` unique source
pyfunctions, no missing module exports, and `Lm2011TokenCounter` present; the
extension rebuild passed with the known nonfatal setuptools logging and Cargo
readonly-cache warnings; the focused cross-domain real-extension selector
passed with `6 passed, 932 deselected`; and the full migration parity suite
passed with `938 passed`. No Python fast-path or fallback contract was
intentionally changed.

After migrating the remaining high-volume Refinitiv production
row-materialization reducers, `python -m py_compile` passed for
`refinitiv_bridge_pipeline.py`, `refinitiv/authority.py`, and
`tests/test_lm2011_rust_accel.py`; `cargo fmt --check` and `cargo check`
passed for `src_rust_migration/rust/lm2011_rust/Cargo.toml`; the extension
rebuild passed with the known nonfatal setuptools logging and Cargo
readonly-cache warnings; the focused Refinitiv reducer selector passed with
`10 passed, 928 deselected`; and the full migration parity suite passed with
`938 passed`. The new Rust/Python surface covers:

- `build_refinitiv_step1_resolution_frame` via
  `refinitiv_bridge_resolution_frame_rows_from_columns`, with row-Rust and
  Python fallback paths.
- Ownership-validation handoff and case-summary reducers via
  column-oriented Rust handoffs before existing row-Rust/Python fallbacks.
- Ownership-universe handoff via a direct column reducer for effective,
  conventional-conflict, ticker-fallback, and nonretrievable branches, with
  row-Rust/Python fallbacks.
- Refinitiv ownership authority final-panel reduction over ownership result
  columns and selected assignment rows, plus review-required table assembly
  over decision and candidate-metric columns.

Low-volume diagnostics, workbook/openpyxl parsing, Excel writing, and
error-example paths were left in Python. A realistic local benchmark was run
on `full_data_run/refinitiv_step1/refinitiv_ric_resolution_common_stock.parquet`
with 38,481 source rows: ownership-universe handoff parity matched exactly,
the Rust column path completed in 3.8125s, and the forced Python fallback
completed in 3.6571s on this workstation. The benchmark confirms
contract-level parity on the local production-shaped artifact; this path is
roughly neutral rather than materially faster in the current PyO3/PyDict
output shape.

After making the targeted Refinitiv column fast paths genuinely column-native,
`refinitiv_bridge_resolution_frame_rows_from_columns` processes Step 1
resolution from typed column vectors and no longer calls
`refinitiv_bridge_py_dict_rows_from_column_values`. The ownership-universe
handoff column reducer continues to avoid that helper. The authority
candidate-metric column reducer now processes row-summary and ownership-result
columns directly, no longer calls row reconstruction helpers, and no longer
returns reconstructed source `request_rows` / `result_rows` in the column fast
path. The older row-record Rust/Python fallback path still carries those
compatibility payloads. Authority final-panel and review-required column
reducers remain column-oriented and return only final output rows.

Verification after this tightening:

- `cargo fmt --check` passed for
  `src_rust_migration/rust/lm2011_rust/Cargo.toml`.
- `cargo check` passed for
  `src_rust_migration/rust/lm2011_rust/Cargo.toml`.
- `python setup.py build_ext --inplace` passed from `src_rust_migration`
  with the known nonfatal setuptools logging and Cargo readonly-cache warnings.
- Focused Refinitiv acceleration selector passed with
  `12 passed, 926 deselected`.
- Full migration parity suite passed with `938 passed`.
- Static probe checks in
  `src_rust_migration/refinitiv_column_native_probe.py` confirmed no targeted
  column reducer body contains the old row reconstruction helpers.

Guarded sample/artifact probes were recorded under
`src_rust_migration/benchmark_results/` using
`tools/run_with_memory_watchguard.ps1`. The main probe used
`full_data_run/sample_5pct_seed42` as the sample root and the available Step 1
Refinitiv parquet artifacts under `full_data_run/refinitiv_step1` for persisted
output parity. It passed under
`watchguard_main_subset750_limit16` (`peak_private_gb=12.356`) after the default
8 GB guard killed the combined full-artifact load at 8.859 GB. The successful
main probe matched full Step 1 resolution and full ownership-universe handoff
against Python fallback and persisted artifacts, and matched a 750-PERMNO
authority subset against Python fallback and persisted candidate-metric,
alias-diagnostic, authority-decision, exception, review-required,
ticker-candidate, and final-panel artifacts.

Recorded benchmark highlights:

- Main 750-PERMNO authority subset:
  - Resolution frame: Rust 3.380s, Python fallback 3.481s.
  - Ownership-universe handoff: Rust 3.354s, Python fallback 3.022s.
  - Candidate metrics: Rust 10.062s, Python fallback 6.689s.
  - Authority tables: Rust 15.987s, Python fallback 17.157s.
- Isolated candidate-metrics memory probe on 2,000 PERMNOs /
  705,075 ownership rows:
  - Rust peak private memory: 5.346 GB.
  - Python fallback peak private memory: 6.463 GB.
  - Rust lowered peak private memory by about 1.117 GB (17.3%) while remaining
    slower in wall time on this isolated reducer.

After adding repo-internal domain package boundaries, `thesis_core`,
`thesis_sec`, `thesis_refinitiv`, `thesis_lm2011`, and `thesis_native` are
discoverable packages beside the compatibility `thesis_pkg` tree. The Rust
extension is now the single PyO3 build at `thesis_native._lm2011_rust`; internal
migration-copy fast paths import through that neutral boundary, while
`thesis_pkg.core.sec` aliases `thesis_pkg.core.sec._lm2011_rust` to the same
module for legacy compatibility. `thesis_refinitiv.lseg_client` groups
extraction-ready reusable LSEG client facades for `api_common`, `batching`,
`ledger`, `provider`, and `stage_audit`; a scan of that package found no
thesis-domain identifiers such as `KYPERMNO`, `doc_id`, `LM2011`, ownership
authority, analyst, SEC, CIK, or PERMNO terms. Existing runners still compile
through their legacy `thesis_pkg` imports, and the short future-extraction note
is `src_rust_migration/DOMAIN_PACKAGE_MIGRATION.md`. Verification passed for
package discovery, native/legacy extension identity, LSEG legacy/new module
identity, package bytecode compile, the single extension rebuild, `cargo fmt`,
`cargo fmt --check`, `cargo check`, and the full migration parity suite (`938
passed`).

Final migration-readiness cleanup kept all changes isolated to
`src_rust_migration` and tightened the current package-boundary state:
`thesis_refinitiv.bridge` remains the primary Refinitiv bridge implementation,
both `thesis_pkg.pipelines.refinitiv_bridge_pipeline` and the narrower
`thesis_pkg.pipelines.refinitiv.bridge` legacy module alias that implementation
through `sys.modules`, and moved bridge delegates now import the domain
authority facade instead of reaching directly through the old bridge pipeline
path. The reusable LSEG client modules remain primary under
`thesis_refinitiv.lseg_client` with `thesis_pkg.pipelines.refinitiv.lseg_*`
compatibility shims. A stale-binary audit found the only Python-loadable
`_lm2011_rust*.pyd` package artifact under `thesis_native`; Cargo target
`_lm2011_rust.dll` files are build-cache outputs, and
`thesis_pkg.core.sec.extraction_fast` is a separate Cython SEC extraction
accelerator rather than a stale Rust extension copy. The remaining domain
facades are explicitly deferred in code and in `DOMAIN_PACKAGE_MIGRATION.md`:
`thesis_refinitiv.authority`, `thesis_refinitiv.ownership`,
`thesis_core.status`, `thesis_sec`, and `thesis_lm2011`. Verification for this
readiness pass passed import identity smoke tests for native, bridge, and LSEG
old/new paths, package bytecode compile, `cargo fmt`, `cargo fmt --check`,
`cargo check`, the extension rebuild, and
`pytest tests\test_lm2011_rust_accel.py -q` (`938 passed`).

The post-cleanup sample-data comparison was rerun through
`tools/run_with_memory_watchguard.ps1` using
`src_rust_migration/refinitiv_column_native_probe.py --skip-full-candidate`
against `full_data_run/refinitiv_step1` and
`full_data_run/sample_5pct_seed42`. The probe was adjusted so the full
lookup/resolution/handoff frames are released before authority artifacts are
loaded, and `--skip-full-candidate` now scans only the selected ownership
results subset instead of materializing the full results parquet. The guarded
run
`src_rust_migration/benchmark_results/watchguard_migration_readiness_sample_20260514_skipfull_streamed`
completed with `status=completed`, `exit_code=0`, and
`peak_private_gb=4.236`. Parity matched for full Step 1 resolution
(`rust_vs_python_and_persisted`), full ownership-universe handoff
(`rust_vs_python_and_persisted`), authority candidate metrics on the 750-PERMNO
subset (`rust_vs_python`), and authority tables on the same subset
(`rust_vs_python_and_persisted_subset`). The comparison covered 38,481 lookup
and resolution rows, 38,561 ownership-handoff and row-summary rows, and
204,135 ownership-result rows for the selected authority subset.

## Known Remaining Work

- Full codebase migration is not complete.
- `thesis_refinitiv.authority`, `thesis_refinitiv.ownership`,
  `thesis_core.status`, `thesis_sec`, and `thesis_lm2011` are still deferred
  facades/placeholders. Promote them only with a broader lazy-export and runner
  boundary pass that preserves legacy package initialization behavior.
- Full SEC item-boundary extraction remains Python/Cython-backed. Changing
  extraction scanners or boundary-selection modules requires the repository's
  suspicious-boundary diagnostics workflow.
- Remaining high-value Refinitiv reducer work is now concentrated in broader
  end-to-end handoff stages rather than the analyst normalized-event reducer
  migrated here.
- The Rust extension is integrated in the migration copy only, not promoted into
  the original `src/` package tree.
- Refinitiv authority candidate metrics now meet the memory-pressure benchmark
  criterion, but the isolated Rust candidate reducer remains slower in wall
  time because the Python wrapper still hands PyO3 Python lists across the FFI
  boundary and converts compact observation payloads back into Python metadata.
  Further speed work should focus on a more columnar Python-facing payload for
  downstream alias/component assembly.
- `thesis_pkg.pipelines.refinitiv.lseg_api_execution` is intentionally not in
  the extraction-ready `thesis_refinitiv.lseg_client` subset yet because its
  mixed zero/positive requeue policy is still tied to the ownership-universe
  stage. Decouple that policy before moving execution into a standalone LSEG
  client package.

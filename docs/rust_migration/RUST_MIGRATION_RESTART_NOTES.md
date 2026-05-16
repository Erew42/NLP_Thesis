# Rust Migration Restart Notes

## Current State

The Rust migration is isolated in `src_rust_migration/`. The original `src/`
tree was intentionally left untouched during the latest work.

The migration copy already contains a broad Rust extension at
`src_rust_migration/rust/lm2011_rust/`, Python wrappers with optional Rust
fast paths, and root-level parity tests in `tests/test_lm2011_rust_accel.py`
that force imports from `src_rust_migration`.
The Rust extension source is no longer a monolithic `src/lib.rs`; `lib.rs`
retains the stable `#[pymodule]` entrypoint and delegates export registration
to focused `src/py_exports/` domain modules for LM2011 core, LM2011 analysis,
SEC, Refinitiv/LSEG, text processing/item cleaning, FinBERT, and common
I/O/audit helpers. Implementation code remains split across sibling modules
for shared utilities, LM2011, SEC extraction, sentence processing, FinBERT,
Refinitiv/LSEG, document ownership, parquet I/O, and audit helper domains.

Recent completed slices:

- Repo-internal domain package boundaries now exist beside `thesis_pkg`:
  `thesis_core`, `thesis_sec`, `thesis_refinitiv`, `thesis_lm2011`, and
  `thesis_native`. Existing runners continue to use `thesis_pkg`, while
  selected new domain boundaries now own implementation modules directly.
- `thesis_refinitiv.bridge` now owns the Refinitiv bridge implementation.
  The historical `thesis_pkg.pipelines.refinitiv_bridge_pipeline` path is a
  `sys.modules` compatibility alias to the same module object, preserving
  public/private symbol access, monkeypatch identity, Rust metrics, and
  fallback behavior.
- Reusable LSEG client facades now live under
  `thesis_refinitiv.lseg_client` for `api_common`, `batching`, `ledger`,
  `provider`, and `stage_audit`. The current `lseg_api_execution` module
  remains outside the extraction-ready boundary until its ownership-universe
  mixed-zero policy is injected as a generic callback.
- Targeted Refinitiv Step 1 and ownership-authority column reducers are now
  genuinely column-native internally. Step 1 resolution and authority
  candidate metrics no longer rebuild source `PyDict` rows inside Rust; the
  candidate-metrics column path also stops returning reconstructed source
  `request_rows` / `result_rows` payloads to Python. The older row-record
  compatibility fallback still preserves those payloads.
- High-volume Refinitiv production reducers now avoid Python
  `DataFrame.to_dicts()` materialization for Step 1 RIC resolution,
  ownership-validation handoff/case summaries, ownership-universe handoff,
  and authority final-panel/review-required assembly. Column-oriented Rust
  entrypoints preserve row-Rust and Python fallbacks where applicable.
- Rust extension source layout and PyO3 export registration refactored into
  focused domain modules without changing Python-visible exports or fallback
  contracts.
- Refinitiv authority conventional-component grouping now has a metadata-level
  Rust bridge over `candidate_meta` and `pair_meta`, with row-Rust and Python
  fallbacks.
- Refinitiv document-ownership universe diagnostics now summarize the detail
  frame through a column-oriented Rust bridge, with row-Rust and Python
  fallbacks.
- FinBERT sentence confusion-review sample ID/order assignment now uses a
  column-oriented Rust bridge from the sorted sample frame, with row-Rust and
  Python fallbacks.

Last verified gates:

- `cargo fmt --check`
- `cargo check`
- `python setup.py build_ext --inplace`
- Focused Refinitiv acceleration tests: `12 passed, 926 deselected`
- Full migration parity suite: `938 passed`
- Domain package import/discovery smoke:
  `thesis_core`, `thesis_sec`, `thesis_refinitiv`,
  `thesis_refinitiv.lseg_client`, and `thesis_lm2011` found by
  `setuptools.find_packages`.
- Runner bytecode compile passed for all copied Python script entrypoints under
  `thesis_pkg/notebooks_and_scripts`.
- Guarded probe:
  `src_rust_migration/benchmark_results/watchguard_main_subset750_limit16`
  completed with `peak_private_gb=12.356` after the 8 GB guard killed the
  combined full-artifact load at 8.859 GB.
- Reproducible benchmark/parity script:
  `src_rust_migration/refinitiv_column_native_probe.py`
- Main local benchmark on `full_data_run/refinitiv_step1` artifacts:
  full Step 1 resolution and ownership handoff matched Python fallback and
  persisted artifacts; a 750-PERMNO authority subset matched Python fallback
  and persisted authority artifacts. Authority tables were slightly faster via
  Rust (15.987s vs 17.157s), while isolated candidate metrics were slower
  in wall time.
- Isolated candidate-metrics memory benchmark:
  2,000 PERMNOs / 705,075 ownership rows under the 8 GB watchguard peaked at
  5.346 GB for Rust and 6.463 GB for Python fallback, a 1.117 GB / 17.3%
  private-memory reduction.
- `git diff -- src` was empty

Known nonfatal build warnings:

- `setuptools` logging can emit `ValueError: underlying buffer has been detached`.
- Cargo may warn about a readonly cache database.
- Both occurred while the extension rebuild still exited successfully.

## Suggested Narrower Next Goals

1. Optimize Refinitiv authority candidate metrics for wall time.
   - The current column-native path lowers peak private memory but is slower
     than Python fallback in isolation because Python still sends column lists
     through PyO3 and receives compact observation metadata back as Python
     objects.
   - A next cut should keep alias diagnostics and conventional-component
     assembly in Rust longer or return a more columnar metadata payload.

2. Convert one FinBERT confusion-review frame materialization.
   - Candidate surfaces:
     `_examples_by_cell_markdown`, `_write_examples_by_cell`, or remaining
     report/count payload conversions in
     `finbert_sentence_confusion_review.py`.
   - Prefer a column-oriented Rust bridge only where the input is already a
     Polars frame.

3. Convert one Refinitiv LSEG request/response helper.
   - Candidate files:
     `lseg_analyst_api.py`, `lseg_ownership_api.py`, or `lseg_lookup_api.py`.
   - Keep one request-item builder or one response normalizer per goal.

4. Audit remaining eager row conversion hotspots.
   - Produce a ranked list from `rg "to_dicts\\(|iter_rows\\(" src_rust_migration/thesis_pkg`.
   - Classify each as already migrated fallback, acceptable small diagnostics,
     or next Rust candidate.

5. Benchmark one already migrated hot path.
   - Pick one data-heavy path such as document-ownership hit selection,
     authority candidate metrics, or FinBERT review sample finalization.
   - Compare Python fallback versus Rust fast path on a realistic local sample.

## Recommended Goal Template

Use a narrow restart goal such as:

```text
In src_rust_migration only, migrate one remaining <specific function/path> to a
Rust fast path with Python fallback. Preserve existing Polars/data contracts,
add parity and fallback tests in tests/test_lm2011_rust_accel.py, run
cargo fmt/check, rebuild the extension, run focused pytest and the full
migration parity suite, and update README_RUST_MIGRATION.md plus
RUST_MIGRATION_AUDIT.md.
```

Avoid restarting with the whole broad migration as a single goal unless the
intent is an open-ended multi-session effort.

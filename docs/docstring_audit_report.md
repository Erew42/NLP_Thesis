# Docstring Audit Report (Phase 2 Follow-up)

Date: 2026-02-20

## Scope Reviewed

Primary touched `src/` files:

- `src/thesis_pkg/core/ccm/sec_ccm_contracts.py`
- `src/thesis_pkg/core/ccm/sec_ccm_premerge.py`
- `src/thesis_pkg/core/sec/extraction.py`
- `src/thesis_pkg/core/sec/extraction_utils.py`
- `src/thesis_pkg/core/sec/filing_text.py`
- `src/thesis_pkg/core/sec/html_audit.py`
- `src/thesis_pkg/core/sec/parquet_stream.py`
- `src/thesis_pkg/core/sec/regime.py`
- `src/thesis_pkg/core/sec/suspicious_boundary_diagnostics.py`
- `src/thesis_pkg/pipelines/sec_ccm_pipeline.py`

Direct dependents scanned for stale references:

- `src/thesis_pkg/api.py`
- `src/thesis_pkg/pipeline.py`
- `src/thesis_pkg/filing_text.py`
- `src/thesis_pkg/pipelines/sec_pipeline.py`
- `src/thesis_pkg/core/sec/heuristics.py`
- `src/thesis_pkg/core/sec/embedded_headings.py`

## Key Issues Found

1. Factual mismatches in docstrings versus implementation behavior.
2. Inconsistent docstring style across outward-facing APIs.
3. Public API docstrings missing key contracts (arguments, returns, failure modes, side effects).
4. Missing docstrings on some public symbols/methods.

## Worst Mismatches (Fixed)

1. `src/thesis_pkg/core/sec/regime.py`
- `normalize_form_type` docstring described normalization inconsistent with actual return values (`"10-K"` / `"10-Q"` / `None`).
- `load_regime_spec` docstring claimed local filesystem fallback that is not implemented.

2. `src/thesis_pkg/core/sec/html_audit.py`
- `normalize_extractor_body` docstring claimed HTML escaping, but function only normalizes/repairs text.
- `normalize_sample_weights` docstring claimed exact sum-to-1 normalization, but implementation only merges and clamps weights.
- `classify_filing_status` docstring implied broader metric logic than the actual `any_fail` / `filing_exclusion_reason` / `any_warn` rule.

3. `src/thesis_pkg/core/sec/suspicious_boundary_diagnostics.py`
- `run_boundary_diagnostics` docstring implied always-on artifact generation; implementation is config-conditional.
- `run_boundary_regression` wording was too narrow for actual replay behavior.

4. Outward-facing SEC-CCM APIs lacked usable contracts
- `src/thesis_pkg/core/ccm/sec_ccm_contracts.py`
- `src/thesis_pkg/core/ccm/sec_ccm_premerge.py`
- `src/thesis_pkg/pipelines/sec_ccm_pipeline.py`

## Prioritized Fixes Applied

### P0: Correctness fixes

- Rewrote inaccurate docstrings in `regime.py`, `html_audit.py`, and `suspicious_boundary_diagnostics.py` to match real behavior.
- Explicitly documented behavior of `daily_join_max_forward_lag_days` and lag-gate semantics in SEC-CCM contracts and pipeline stages.

### P1: Public API contract completion

- Normalized public docstrings to Google style (`Args`, `Returns`/`Yields`, `Raises` where applicable) across all scoped files.
- Added side-effect/artifact semantics for I/O-heavy entrypoints (`run_sec_ccm_premerge_pipeline`, `run_boundary_diagnostics`, HTML writers).

### P2: Missing public docs

- Added missing docstrings for:
  - `reset_extraction_fastpath_metrics`
  - `InternalHeadingLeak`
  - `DiagnosticsRow`
  - public serialization helpers such as `to_dict`/`write_json` methods in join-spec dataclasses.

## Direct Dependent Scan Result

- No stale dependent docstrings requiring edits were found in scanned direct dependents.
- Dependent references were import/call sites without conflicting behavior claims.

## Validation

Behavioral regression check:

- `pytest -q tests/test_parquet_stream.py tests/test_regime_switches.py tests/test_sec_ccm_premerge.py`
- Result: `33 passed`

Diff guard outcome:

- No logic refactors were introduced.
- Changes are documentation-focused (docstrings/report content only).

## Residual Risks / Follow-ups

1. A few public re-export modules (for example `api.py`, `pipeline.py`) intentionally remain thin and largely undocumented at symbol level.
2. If future behavior changes occur in diagnostics artifact toggles or SEC-CCM presets, these docstrings should be updated in the same PR to avoid drift.

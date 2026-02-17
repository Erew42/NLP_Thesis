# NLP_THESIS

Python package for my NLP thesis (SEC filings, CRSP/Compustat, etc.).

## SEC-CCM Pre-Merge (Doc Grain)

The repository includes a two-phase SEC-CCM pre-merge pipeline:

1. Phase A (`doc_id -> gvkey/kypermno`) in `src/thesis_pkg/core/ccm/sec_ccm_premerge.py`:
   - Normalizes filing identifiers.
   - Resolves conservative link candidates.
   - Emits `sec_ccm_links_doc.parquet` (one row per `doc_id`).

2. Phase B (alignment + optional daily join):
   - Uses a versioned join spec with explicit modes:
     - Alignment modes:
       - `NEXT_TRADING_DAY_STRICT` (legacy default)
       - `FILING_DATE_EXACT_OR_NEXT_TRADING` (LM2011-style filing-date anchoring)
       - `FILING_DATE_EXACT_ONLY` (diagnostic strict exact match)
     - Daily join modes:
       - `ASOF_FORWARD` (legacy default)
       - `EXACT_ON_ALIGNED_DATE`
   - Legacy V1 inputs are normalized to canonical V2 behavior.
   - `FIRST_CLOSE_AFTER_ACCEPTANCE` in V1 is currently rejected with an explicit error.
   - Emits `final_flagged_data.parquet` (one row per `doc_id`).

Canonical reason codes and join-spec are defined in:
- `src/thesis_pkg/core/ccm/sec_ccm_contracts.py`

Pipeline entrypoint:
- `thesis_pkg.pipeline.run_sec_ccm_premerge_pipeline`

Additional contract details:
- `docs/sec_ccm_premerge.md`

Automatic per-run observability artifacts are produced by default:
- Step performance table: `sec_ccm_run_steps.parquet`
- DAG visuals: `sec_ccm_run_dag.mmd` and `sec_ccm_run_dag.dot`
- Run manifest: `sec_ccm_run_manifest.json`
- Run report: `sec_ccm_run_report.md`

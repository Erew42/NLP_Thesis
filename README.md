# NLP_THESIS

Python package for my NLP thesis (SEC filings, CRSP/Compustat, etc.).

## SEC-CCM Pre-Merge (Doc Grain)

The repository includes a two-phase SEC-CCM pre-merge pipeline:

1. Phase A (`doc_id -> gvkey/kypermno`) in `src/thesis_pkg/core/ccm/sec_ccm_premerge.py`:
   - Normalizes filing identifiers.
   - Resolves conservative link candidates.
   - Emits `sec_ccm_links_doc.parquet` (one row per `doc_id`).

2. Phase B (alignment + optional daily join):
   - Default alignment policy: `NEXT_TRADING_DAY_STRICT`.
   - Computes `aligned_caldt` as first trading day strictly after `filing_date`.
   - Optionally joins CRSP daily (or merged daily panel) keyed by `(kypermno, caldt)`
     using forward asof on `aligned_caldt`.
   - Emits `final_flagged_data.parquet` (one row per `doc_id`).

Canonical reason codes and join-spec are defined in:
- `src/thesis_pkg/core/ccm/sec_ccm_contracts.py`

Pipeline entrypoint:
- `thesis_pkg.pipeline.run_sec_ccm_premerge_pipeline`

Additional contract details:
- `docs/sec_ccm_premerge.md`

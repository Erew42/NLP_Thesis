# Architecture

This tracked page is the stable architecture entrypoint for `thesis_pkg`. The
expanded module-level architecture pages that may appear under this directory
are local/generated artifacts and are intentionally not required for a clean
checkout.

## Source Areas

- `src/thesis_pkg/core/sec/`: SEC filing text normalization, item extraction,
  regime logic, HTML/text boundary diagnostics, and parquet streaming helpers.
- `src/thesis_pkg/core/ccm/`: CRSP/Compustat transforms, canonical links,
  SEC-CCM contracts, and document-grain pre-merge logic.
- `src/thesis_pkg/pipelines/`: reusable pipeline entrypoints for SEC, CCM,
  SEC-CCM, LM2011, and Refinitiv workflows.
- `src/thesis_pkg/pipelines/refinitiv/`: Refinitiv/LSEG authority, request,
  ownership, analyst, ledger, and recovery stages.
- `src/thesis_pkg/benchmarking/`: FinBERT sentence preprocessing/inference,
  item-scope cleaning, benchmark sweeps, and diagnostic audits.
- `src/thesis_pkg/notebooks_and_scripts/`: runnable local and Colab workflow
  entrypoints.
- `src/thesis_pkg/io/` and `src/thesis_pkg/cleaning/`: shared I/O and cleaning
  helpers.

## Pipeline Shape

1. Build and clean CRSP/Compustat market-data inputs.
2. Normalize SEC filing text and extract filing items.
3. Link filings to market data at the document grain and populate
   `data_status`.
4. Add Refinitiv/LSEG ownership and analyst enrichments where required.
5. Build LM2011 dictionary features, FinBERT item features, event panels,
   regressions, and thesis-facing assets.

## Generated Reference

Run `python tools/docs_pipeline.py all` to refresh generated API/reference pages
and the MkDocs navigation for the current source tree.

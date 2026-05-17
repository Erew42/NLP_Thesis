# Master Thesis Code

Python package and workflow repository for a Master thesis project using SEC
filing text, CRSP/Compustat market data, Refinitiv/LSEG enrichment, LM2011-style
dictionary measures, and FinBERT sentiment features.

The installable package is `thesis_pkg`. Most core data work is implemented with
Polars and is organized around analysis-ready document identifiers, conservative
SEC-CCM linking, and reproducible staged outputs.

## Repository Sections

| Path | Purpose |
| --- | --- |
| `src/thesis_pkg/core/sec/` | SEC filing text normalization, 10-K/10-Q item extraction, regime-aware item definitions, HTML/text boundary diagnostics, and parquet streaming helpers. |
| `src/thesis_pkg/core/ccm/` | CRSP/Compustat cleaning, canonical link construction, SEC-CCM contract definitions, and document-grain pre-merge logic. |
| `src/thesis_pkg/pipelines/` | Reusable pipeline entrypoints for SEC processing, CCM processing, SEC-CCM merging, LM2011 replication/extension workflows, and Refinitiv bridge stages. |
| `src/thesis_pkg/pipelines/refinitiv/` | Refinitiv/LSEG authority, ownership, analyst, batching, request ledger, and recovery helpers. |
| `src/thesis_pkg/benchmarking/` | FinBERT sentence preprocessing/inference, item-scope cleaning, benchmark sweeps, diagnostic audits, and sentence-length/tokenization tools. |
| `src/thesis_pkg/notebooks_and_scripts/` | Script and notebook entrypoints for full local/Colab runs, sample reruns, FinBERT analysis, LM2011 validation, and Refinitiv finalization. |
| `src/thesis_pkg/io/` and `src/thesis_pkg/cleaning/` | Thin I/O adapters and shared cleaning utilities. |
| `tests/` | Pytest coverage for extraction, SEC-CCM contracts, Refinitiv stages, LM2011 pipelines, FinBERT workflows, docs tooling, and submission packaging. |
| `docs/` | Tracked documentation entrypoints plus locally generated MkDocs, architecture, reference, behavior, and operational artifacts. Many generated docs paths are intentionally ignored. |
| `thesis_assets/` | Final thesis-facing tables, figures, and reporting adapters generated from retained outputs. |
| `tools/` | Repository utilities for documentation, submission packaging, and retained-output workflows. |

Generated or local-run directories such as `results/`, `output/`,
`full_data_run/`, `reports/`, and scratch/cache folders are not the primary code
surface. Treat them as run artifacts unless a specific workflow documents
otherwise.

## Main Workflows

### 1. Market Data And SEC-CCM Linking

The CCM side cleans and aligns CRSP/Compustat data, then links filings at the
document grain. The canonical SEC filing identifier is:

```text
doc_id = "{cik_10}:{accession_nodash}"
```

The SEC-CCM pre-merge is split into:

1. Phase A (`doc_id -> gvkey/KYPERMNO`) in
   `src/thesis_pkg/core/ccm/sec_ccm_premerge.py`.
2. Phase B, which applies explicit filing-date alignment and daily market-data
   join modes from `src/thesis_pkg/core/ccm/sec_ccm_contracts.py`.

Pipeline entrypoint:

```python
thesis_pkg.pipeline.run_sec_ccm_premerge_pipeline
```

Canonical reason codes and join-spec definitions live in
`src/thesis_pkg/core/ccm/sec_ccm_contracts.py`.

### 2. SEC Filing Text Processing

SEC extraction is a multi-stage heuristic engine, not simple regex scanning. It
normalizes raw filing text, handles sparse HTML layouts, filters hostile tables
of contents, applies filing-date-specific item regimes, and writes item-level
parquet outputs for downstream analysis.

Important modules:

- `src/thesis_pkg/core/sec/extraction.py`
- `src/thesis_pkg/core/sec/embedded_headings.py`
- `src/thesis_pkg/core/sec/regime.py`
- `src/thesis_pkg/pipelines/sec_pipeline.py`

Form labels differ across data sources. SEC text and filenames usually use
hyphenated labels such as `10-K`, `10-Q`, and `10-K/A`, while CCM
`filingdates.SRCTYPE` commonly uses compact labels such as `10K`, `10Q`, and
`10K/A`. Filters or joins that cross those sources should normalize form labels
before comparing them.

### 3. Refinitiv/LSEG Enrichment

Refinitiv and LSEG stages resolve instrument authority, build ownership and
analyst request universes, execute or finalize API outputs, and bridge those
inputs back to document-level thesis panels.

The related code lives primarily in:

- `src/thesis_pkg/pipelines/refinitiv/`
- `src/thesis_pkg/pipelines/refinitiv_bridge_pipeline.py`
- `src/thesis_pkg/notebooks_and_scripts/refinitiv_local_api_runner.py`
- `src/thesis_pkg/notebooks_and_scripts/lm2011_refinitiv_finalize_colab.ipynb`

### 4. LM2011 Replication And Extensions

The LM2011 workflow builds seeded sample backbones, dictionary families,
event-window panels, Fama-MacBeth tables, quarterly sensitivity outputs, and
monthly trading-strategy artifacts.

Primary entrypoint:

```bash
python src/thesis_pkg/notebooks_and_scripts/lm2011_sample_post_refinitiv_runner.py
```

The unified runner can also trigger these stages through environment variables:

```bash
python src/thesis_pkg/notebooks_and_scripts/sec_ccm_unified_runner.py
```

### 5. FinBERT Item Analysis

FinBERT processing cleans item scopes, materializes sentence datasets, runs
staged inference, aggregates item-level features, and writes diagnostic audits.
The workflow supports preprocessing-only and analysis-only passes.

Primary entrypoint:

```bash
python src/thesis_pkg/notebooks_and_scripts/finbert_item_analysis_runner.py --data-profile LOCAL_SAMPLE
```

For explicit non-sample runs:

```bash
python src/thesis_pkg/notebooks_and_scripts/finbert_item_analysis_runner.py --data-profile EXPLICIT --source-items-dir <items_analysis_dir> --output-dir <output_dir>
```

### 6. Thesis Assets And Submission Package

`thesis_assets/` contains thesis-facing tables and figures derived from retained
pipeline outputs. `tools/run_submission_pipeline.py` validates a submission
layout and can rebuild retained thesis assets.

Default readiness check from an extracted submission zip root:

```bash
pip install -e .[benchmark]
python tools/run_submission_pipeline.py --submission-root .
```

Full rerun path from packaged inputs:

```bash
python tools/run_submission_pipeline.py --submission-root . --stage all
```

Use `--stage retained` when retained `items_analysis` and
`sec_ccm_matched_clean` inputs should be consumed directly. See
`SUBMISSION_README.md` for the expected zip layout.

## Installation

Recommended development install with native Cython/Rust accelerators:

```bash
pip install -e .
```

This path expects a working C/Cython build environment and Rust/Cargo.

Pure-Python install without native accelerators:

```powershell
$env:NLP_THESIS_DISABLE_NATIVE_EXTENSIONS = "1"
pip install -e .
```

```bash
NLP_THESIS_DISABLE_NATIVE_EXTENSIONS=1 pip install -e .
```

In constrained environments that also lack native build Python packages, use
the same environment variable with `--no-build-isolation` after ensuring
`setuptools` is installed.

Development and documentation extras:

```bash
pip install -e .[dev,docs]
```

Benchmarking and model-related extras:

```bash
pip install -e .[benchmark]
```

## Documentation

The docs site is built with MkDocs. Most architecture/reference pages are
generated locally and intentionally ignored, so refresh them before checking or
building the site:

```bash
python tools/docs_pipeline.py all
```

Use the individual `extract`, `scaffold`, `check`, and `build` subcommands when
debugging a specific docs stage. The `check` subcommand runs a MkDocs build by
default.

Tracked documentation entrypoints:

- `docs/index.md`
- `docs/architecture/index.md`
- `docs/decisions/index.md`
- `docs/docstring_audit_report.md`

## Testing

Tests use `pytest`:

```bash
pytest
```

Run focused tests when changing a narrow area, for example:

```bash
pytest tests/test_sec_ccm_premerge.py
pytest tests/test_finbert_item_analysis.py
pytest tests/test_lm2011_pipeline.py
```

Changes to SEC item-boundary extraction should also run the tracked
suspicious-boundary diagnostics entrypoint with an explicit input directory:

```bash
python run_diagnostics.py --parquet-dir <sec_parquet_dir>
```

## Core Conventions

- Python 3.10+.
- Core pipeline transformations should use Polars, preferably `LazyFrame`.
- `data_status` is the canonical integer bitmask for data availability and
  merge provenance; avoid duplicate boolean availability columns.
- Key identifiers should use the repository vocabulary: `cik_10`,
  `accession_nodash`, `doc_id`, and `KYPERMNO`.
- Regime-aware filing item logic should come from the JSON regime definitions,
  not hard-coded item titles.

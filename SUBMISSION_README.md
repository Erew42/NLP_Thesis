# Thesis Submission Package

This repository is prepared as a code-plus-input-data submission. Reviewers can
reproduce the generated SEC/CCM, Loughran and McDonald 2011 (LM2011), FinBERT,
extension, and thesis-asset outputs from the packaged inputs, subject to local
compute and dependency availability. Some retained intermediate or analysis
outputs may be included as optional time-saving caches, but they are not the
primary submission contract.

## Supervisor Quick Start

From the extracted zip root:

```bash
pip install -e .[benchmark]
python tools/run_submission_pipeline.py --submission-root . --stage all
```

This submission is organized around the seed-input reproduction path. The
`--stage all` command refreshes the package manifest, validates the packaged
seed inputs, rebuilds the generated SEC/CCM intermediates, and then runs the
analysis stages.

The default command without `--stage all` runs the lightweight `readiness`
sequence, which checks the packaged seed inputs and rebuilds thesis tables and
figures from the retained analysis outputs:

1. write or refresh `submission_package_manifest.json`;
2. validate the packaged seed data paths and schemas;
3. create `submission_lock.json` and rebuild thesis tables/figures from retained
   outputs under `analysis_outputs/`.

For reproduction from the packaged input data, use:

```bash
python tools/run_submission_pipeline.py --submission-root . --stage all
```

Generated thesis assets are written to:

```text
output/thesis_assets/
```

## What the Pipeline Does

The `--stage all` command runs the submission workflow in a fixed order:

- `package-manifest`: records which packaged inputs and optional cached outputs
  are present.
- `validate-seed-inputs`: checks that the submitted SEC, CRSP/Compustat,
  Refinitiv, and LM2011 input files are available and have the expected basic
  structure.
- `sec-ccm-premerge`: links SEC filings to the CRSP/Compustat market and
  accounting data used in the analysis.
- `sec-items-analysis`: extracts the relevant 10-K item text from the submitted
  filing data.
- `validate-derived-inputs`: checks the generated intermediate files before the
  analysis stages start.
- `lm2011`: rebuilds the Loughran-McDonald dictionary measures and main
  event-study results.
- `event-window-sensitivity` and `nw-sensitivity`: rerun the main robustness
  checks for alternative event windows and Newey-West lag choices.
- `finbert-preprocess` and `finbert-analysis`: prepare filing text for FinBERT
  and generate the FinBERT sentiment measures. Included cached FinBERT outputs
  may save time here.
- `extension`, `visible-prefix-extension`, and `finbert-robustness`: build the
  combined LM2011/FinBERT extensions and related robustness checks.
- `thesis-assets`: writes the final thesis tables, figures, and run manifest.

## Running Stages

If no `--stage` is provided, the runner uses `readiness`. That mode validates
the seed-input package and rebuilds thesis assets from retained outputs. For a
complete recomputation of derived SEC/CCM and analysis outputs, use:

```bash
python tools/run_submission_pipeline.py --submission-root . --stage all
```

To preview the stage plan without running any work, add `--dry-run`:

```bash
python tools/run_submission_pipeline.py --submission-root . --stage all --dry-run
```

To run a single stage, pass its name explicitly:

```bash
python tools/run_submission_pipeline.py --submission-root . --stage validate-seed-inputs
python tools/run_submission_pipeline.py --submission-root . --stage sec-ccm-premerge
python tools/run_submission_pipeline.py --submission-root . --stage thesis-assets
```

Single-stage runs assume their prerequisite inputs already exist. Use
`--stage all` for a complete ordered rerun, or use the default `readiness`
mode when the retained analysis outputs under `analysis_outputs/` are already
available and you only need to rebuild thesis assets.

## Expected Zip Layout

Keep these paths in the submitted zip:

```text
pyproject.toml
SUBMISSION_README.md
submission_pipeline_config.json
src/
thesis_assets/
tools/
data/
analysis_outputs/
```

`submission_pipeline_config.json` contains submission-local artifact overrides
needed for retained thesis-asset rebuilds. In this package it locks the retained
three-outcome Table VI no-ownership validation surface under
`analysis_outputs/lm2011_table_vi_validation_second_pass/`.

The `data/` directory must contain:

```text
data/sec/year_merged/*.parquet
data/ccm_crsp_compustat/
data/refinitiv_finalized/
data/LM2011_additional_data/
```

The CCM daily market panel
`final_flagged_data_compdesc_added.parquet` is a generated intermediate. The
`--stage all` path rebuilds it under:

```text
analysis_outputs/ccm_derived/final_flagged_data_compdesc_added.parquet
```

It is therefore not required as a packaged input when the raw CCM daily, link,
header, filing-date, company-history, and company-description parquets are
present under `data/ccm_crsp_compustat/`.

The default `readiness` check does not require retained derived SEC inputs.
If you want to run retained-mode downstream analysis stages directly, include:

```text
data/sec/items_analysis/*.parquet
data/sec/sec_ccm_matched_clean.parquet
```

For a full recompute from seed parquets, `--stage all` can regenerate those SEC
artifacts under `analysis_outputs/items_analysis/` and
`analysis_outputs/sec_ccm_premerge/`, and can rebuild the CCM daily market panel
under `analysis_outputs/ccm_derived/`. Optional raw SEC zip inputs may be placed
under `data/sec/raw_zips/YYYY.zip`, but raw zip parsing is disabled unless
`submission_pipeline_config.json` sets `source_mode` to `from_raw_zips` and
`run_raw_sec_stages` to `true`.

It is fine to omit development-only folders from the zip, including `tests/`,
`.git/`, `.pytest_cache/`, `.ruff_cache/`, `docs/`, `reports/`, `reviews/`,
and ad hoc scratch directories.

## Tools Included

- `tools/run_submission_pipeline.py`: main supported reviewer entrypoint for
  validating the package and rerunning retained or recomputed stages.
- `tools/build_thesis_assets.py`: direct entrypoint for rebuilding thesis
  tables and figures from retained run outputs.
- `tools/Load_convert_Data_final.ipynb`: Colab notebook used to convert the
  institute-provided CRSP/Compustat `.rds` / `.RData` files into parquet inputs
  used by the pipeline. It is included for data-provenance transparency, but
  reviewers do not need to run it unless starting from the original R data
  exports rather than the packaged parquet inputs.

Most other files in `tools/` are auxiliary diagnostics, audit scripts, sampling
utilities, or documentation/report-generation helpers used during development
and validation. They are included for transparency, but the supported submission
workflow is `tools/run_submission_pipeline.py`; `tools/build_thesis_assets.py`
can be used to rebuild thesis tables and figures directly.

## Full Rerun

The full recomputation path is:

```bash
python tools/run_submission_pipeline.py --submission-root . --stage all
```

To use retained analysis outputs instead of recomputing them, use the default
readiness command:

```bash
python tools/run_submission_pipeline.py --submission-root .
```

Use `--force` only when intentionally rebuilding cached text features,
sentence inference, event panels, regression tables, and extension sensitivity
artifacts.

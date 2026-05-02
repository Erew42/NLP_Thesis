# Thesis Submission Package

This repository is prepared so the submitted zip can be checked from retained
data and analysis artifacts without rerunning the full audit/recompute stack.

## Supervisor Quick Start

From the extracted zip root:

```bash
pip install -e .[benchmark]
python tools/run_submission_pipeline.py --submission-root .
```

The default command runs the lightweight `readiness` sequence:

1. write or refresh `submission_package_manifest.json`;
2. validate required packaged data paths and schemas;
3. create `submission_lock.json` and rebuild thesis tables/figures from retained
   outputs under `analysis_outputs/`.

Generated thesis assets are written to:

```text
output/thesis_assets/
```

## Expected Zip Layout

Keep these paths in the submitted zip:

```text
pyproject.toml
setup.py
README.md
SUBMISSION_README.md
src/
thesis_assets/
tools/run_submission_pipeline.py
tools/build_thesis_assets.py
data/
analysis_outputs/
```

The `data/` directory must contain:

```text
data/sec/year_merged/*.parquet
data/sec/items_analysis/*.parquet
data/sec/sec_ccm_matched_clean.parquet
data/ccm_crsp_compustat/
data/refinitiv_finalized/
data/LM2011_additional_data/
```

It is fine to omit development-only folders from the zip, including `tests/`,
`.git/`, `.pytest_cache/`, `.ruff_cache/`, `docs/`, `reports/`, `reviews/`,
and ad hoc scratch directories.

## Optional Full Rerun

The full recomputation path is still available, but it is intentionally not the
default:

```bash
python tools/run_submission_pipeline.py --submission-root . --stage all
```

Use `--force` only when intentionally rebuilding cached text features,
sentence inference, event panels, regression tables, and extension sensitivity
artifacts.

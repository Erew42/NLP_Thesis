# `thesis_assets`

Lightweight source-side scaffolding for generating thesis-ready tables and figures from already-produced run artifacts.

`thesis_assets/` is source code only. Generated outputs are written under:

`output/thesis_assets/<run_id>/`

with subdirectories for `tables/`, `figures/`, `csv/`, `tex/`, `logs/`, plus `manifest.json`.

## Structure

- `config/`: artifact names, output directory names, and run-root conventions.
- `registry/`: thematic asset registries for Chapter 4 and Chapter 5.
- `builders/`: artifact resolution, sample-contract helpers, and asset builders.
- `figures/`: shared Matplotlib plotting helpers.
- `renderers/`: CSV, LaTeX, figure, and manifest writers.
- `templates/`: tiny LaTeX wrapper fragment(s).
- `cli/`: command-line entrypoints.

## Local CLI Usage

Build all registered assets:

```bash
python -m thesis_assets.cli build-all --run-id local_dev
```

Build one chapter:

```bash
python -m thesis_assets.cli build-chapter --chapter chapter5 --run-id local_dev
```

Build one asset with explicit run roots:

```bash
python -m thesis_assets.cli build-asset \
  --asset-id ch5_fit_horserace_item7_c0 \
  --run-id local_dev \
  --lm2011-post-refinitiv-dir /path/to/lm2011_post_refinitiv \
  --lm2011-extension-dir /path/to/lm2011_extension
```

## Notebook / Colab Usage

If the repo root is not already importable, add it to `sys.path` first:

```python
from pathlib import Path
import sys

repo_root = Path("/content/NLP_Thesis").resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
```

Then call the public Python API directly:

```python
from thesis_assets import build_single_asset

result = build_single_asset(
    asset_id="ch5_concordance_item7_common_sample",
    run_id="colab_demo",
    repo_root=repo_root,
    lm2011_extension_dir=Path("/content/drive/MyDrive/Data_LM/results/sec_ccm_unified_runner/lm2011_extension"),
)

result.manifest_path
```

The same build functions back both the notebook API and the CLI.

## Thin Entry Points Under `tools/`

For convenience, the repo also includes:

- [tools/build_thesis_assets.py](../tools/build_thesis_assets.py): thin local/Colab wrapper around the `thesis_assets` API with `LOCAL_REPO`, `COLAB_DRIVE`, and `EXPLICIT` path profiles. The default local and Drive hints follow `sec_ccm_unified_runner.py` result conventions and then fall back to older standalone runner directories when needed.
- [tools/thesis_assets_colab_entrypoint.ipynb](../tools/thesis_assets_colab_entrypoint.ipynb): Colab-friendly notebook with Drive mount, repo-path setup, parameter cells, and direct calls into the same API. Its default Drive root is `Data_LM`, and it resolves thesis-asset inputs from `results/sec_ccm_unified_runner/...` by default.

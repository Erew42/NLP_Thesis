from __future__ import annotations

from pathlib import Path

from thesis_assets.config.constants import OUTPUT_SUBDIRS
from thesis_assets.config.constants import RUN_FAMILY_FINBERT_ROBUSTNESS
from thesis_assets.config.constants import RUN_FAMILY_FINBERT_RUN
from thesis_assets.config.constants import RUN_FAMILY_LM2011_EXTENSION
from thesis_assets.config.constants import RUN_FAMILY_LM2011_POST_REFINITIV


def build_output_root(repo_root: Path, run_id: str) -> Path:
    if not run_id.strip():
        raise ValueError("run_id must be non-empty.")
    return repo_root.resolve() / "output" / "thesis_assets" / run_id.strip()


def prepare_output_dirs(output_root: Path) -> dict[str, Path]:
    dirs = {name: output_root / name for name in OUTPUT_SUBDIRS}
    output_root.mkdir(parents=True, exist_ok=True)
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def candidate_run_roots(repo_root: Path, run_family: str) -> tuple[Path, ...]:
    repo_root = repo_root.resolve()
    full_data_run = repo_root / "full_data_run"
    sample_results = full_data_run / "sample_5pct_seed42" / "results"
    local_sample_unified = sample_results / "sec_ccm_unified_runner" / "local_sample"

    if run_family == RUN_FAMILY_LM2011_POST_REFINITIV:
        candidates = [
            full_data_run / "lm2011_post_refinitiv",
            sample_results / "lm2011_sample_post_refinitiv_runner",
            local_sample_unified / "lm2011_post_refinitiv",
            Path("/content/drive/MyDrive/Data_LM/results/sec_ccm_unified_runner/lm2011_post_refinitiv"),
            Path("/content/drive/MyDrive/Data_LM/results/lm2011_sample_post_refinitiv_runner"),
            Path("/content/drive/My Drive/Data_LM/results/sec_ccm_unified_runner/lm2011_post_refinitiv"),
            Path("/content/drive/My Drive/Data_LM/results/lm2011_sample_post_refinitiv_runner"),
        ]
    elif run_family == RUN_FAMILY_LM2011_EXTENSION:
        globbed = sorted((full_data_run).glob("lm2011_extension-*"))
        candidates = [
            full_data_run / "lm2011_extension",
            *[path / "lm2011_extension" for path in globbed],
            Path("/content/drive/MyDrive/Data_LM/results/sec_ccm_unified_runner/lm2011_extension"),
            Path("/content/drive/My Drive/Data_LM/results/sec_ccm_unified_runner/lm2011_extension"),
        ]
    elif run_family == RUN_FAMILY_FINBERT_RUN:
        finbert_local = sample_results / "finbert_item_analysis_runner"
        unified_local = local_sample_unified / "finbert_item_analysis"
        finbert_colab = Path("/content/drive/MyDrive/Data_LM/results/finbert_item_analysis_runner")
        unified_colab = Path("/content/drive/MyDrive/Data_LM/results/sec_ccm_unified_runner/finbert_item_analysis")
        finbert_colab_alt = Path("/content/drive/My Drive/Data_LM/results/finbert_item_analysis_runner")
        unified_colab_alt = Path("/content/drive/My Drive/Data_LM/results/sec_ccm_unified_runner/finbert_item_analysis")
        candidates = [
            *sorted(finbert_local.glob("*")),
            *sorted(unified_local.glob("*")),
            *sorted(finbert_colab.glob("*")),
            *sorted(unified_colab.glob("*")),
            *sorted(finbert_colab_alt.glob("*")),
            *sorted(unified_colab_alt.glob("*")),
        ]
    elif run_family == RUN_FAMILY_FINBERT_ROBUSTNESS:
        robustness_candidates = [
            candidate
            for candidate in full_data_run.glob("finbert_robustness*")
            if candidate.is_dir()
        ]
        if robustness_candidates:
            candidates = [max(robustness_candidates, key=lambda path: path.stat().st_mtime)]
        else:
            candidates = [
                Path("/content/drive/MyDrive/Data_LM/results/finbert_robustness"),
                Path("/content/drive/My Drive/Data_LM/results/finbert_robustness"),
            ]
    else:
        raise ValueError(f"Unsupported run family: {run_family!r}")

    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve() if candidate.exists() else candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(candidate)
    return tuple(deduped)

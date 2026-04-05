from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT_ENV_VAR = "NLP_THESIS_REPO_ROOT"


def _resolve_repo_root() -> Path:
    candidates: list[Path] = []
    env_root = os.environ.get(REPO_ROOT_ENV_VAR)
    if env_root:
        candidates.append(Path(env_root).expanduser())

    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])

    script_path = Path(__file__).resolve()
    candidates.extend(script_path.parents)

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "src" / "thesis_pkg" / "pipeline.py").exists():
            return candidate
    raise RuntimeError("Could not resolve repository root containing src/thesis_pkg/pipeline.py")


ROOT = _resolve_repo_root()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from thesis_pkg.benchmarking import BucketBatchConfig
from thesis_pkg.benchmarking import FinbertAnalysisRunConfig
from thesis_pkg.benchmarking import FinbertSentencePreprocessingRunConfig
from thesis_pkg.benchmarking import FinbertSectionUniverseConfig
from thesis_pkg.benchmarking import FinbertRuntimeConfig
from thesis_pkg.benchmarking import SentenceDatasetConfig
from thesis_pkg.benchmarking import run_finbert_item_analysis
from thesis_pkg.benchmarking import run_finbert_sentence_preprocessing


DEFAULT_FULL_DATA_ROOT = ROOT / "full_data_run"
DEFAULT_SAMPLE_ROOT = DEFAULT_FULL_DATA_ROOT / "sample_5pct_seed42"
DEFAULT_SAMPLE_UPSTREAM_RUN_ROOT = DEFAULT_SAMPLE_ROOT / "results" / "sec_ccm_unified_runner" / "local_sample"
DEFAULT_SAMPLE_ITEMS_ANALYSIS_DIR = DEFAULT_SAMPLE_UPSTREAM_RUN_ROOT / "items_analysis"
DEFAULT_SAMPLE_BACKBONE_PATH = (
    DEFAULT_SAMPLE_UPSTREAM_RUN_ROOT / "sec_ccm_premerge" / "final_flagged_data.parquet"
)
DEFAULT_SAMPLE_OUTPUT_ROOT = DEFAULT_SAMPLE_ROOT / "results" / "finbert_item_analysis_runner"

BATCH_PRESETS: dict[str, BucketBatchConfig] = {
    "small": BucketBatchConfig(name="small", short_batch_size=32, medium_batch_size=16, long_batch_size=8),
    "baseline": BucketBatchConfig(name="baseline", short_batch_size=64, medium_batch_size=32, long_batch_size=16),
    "large": BucketBatchConfig(name="large", short_batch_size=128, medium_batch_size=64, long_batch_size=32),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run year-sharded FinBERT item analysis over SEC items_analysis shards.")
    parser.add_argument(
        "--data-profile",
        choices=("LOCAL_SAMPLE", "EXPLICIT"),
        default="LOCAL_SAMPLE",
        help="Use the seeded local sample layout or explicit source/output paths.",
    )
    parser.add_argument("--source-items-dir", type=Path, default=None)
    parser.add_argument("--backbone-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--years", type=int, nargs="*", default=None)
    parser.add_argument("--batch-profile", choices=tuple(BATCH_PRESETS), default="baseline")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--write-sentence-scores", action="store_true")
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Stop after sentence splitting and token-length bucketing; do not run FinBERT inference.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--note", type=str, default="")
    return parser.parse_args(argv)


def _resolve_run_config(args: argparse.Namespace) -> FinbertAnalysisRunConfig:
    if args.data_profile == "LOCAL_SAMPLE":
        source_items_dir = DEFAULT_SAMPLE_ITEMS_ANALYSIS_DIR
        backbone_path = DEFAULT_SAMPLE_BACKBONE_PATH
        output_dir = DEFAULT_SAMPLE_OUTPUT_ROOT
        year_filter = tuple(args.years) if args.years else (2006, 2007, 2008)
    else:
        if args.source_items_dir is None:
            raise ValueError("--source-items-dir is required when --data-profile=EXPLICIT.")
        if args.output_dir is None:
            raise ValueError("--output-dir is required when --data-profile=EXPLICIT.")
        source_items_dir = Path(args.source_items_dir).resolve()
        backbone_path = Path(args.backbone_path).resolve() if args.backbone_path is not None else None
        output_dir = Path(args.output_dir).resolve()
        year_filter = tuple(args.years) if args.years else None

    if args.source_items_dir is not None and args.data_profile == "LOCAL_SAMPLE":
        source_items_dir = Path(args.source_items_dir).resolve()
    if args.backbone_path is not None and args.data_profile == "LOCAL_SAMPLE":
        backbone_path = Path(args.backbone_path).resolve()
    if args.output_dir is not None and args.data_profile == "LOCAL_SAMPLE":
        output_dir = Path(args.output_dir).resolve()

    return FinbertAnalysisRunConfig(
        source_items_dir=source_items_dir,
        out_root=output_dir,
        batch_config=BATCH_PRESETS[args.batch_profile],
        section_universe=FinbertSectionUniverseConfig(source_items_dir=source_items_dir),
        runtime=FinbertRuntimeConfig(device=args.device),
        sentence_dataset=SentenceDatasetConfig(),
        backbone_path=backbone_path,
        year_filter=year_filter,
        write_sentence_scores=args.write_sentence_scores,
        overwrite=args.overwrite,
        run_name=args.run_name,
        note=args.note,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.preprocess_only:
        base_cfg = _resolve_run_config(args)
        artifacts = run_finbert_sentence_preprocessing(
            FinbertSentencePreprocessingRunConfig(
                source_items_dir=base_cfg.source_items_dir,
                out_root=base_cfg.out_root,
                section_universe=base_cfg.section_universe,
                sentence_dataset=base_cfg.sentence_dataset,
                year_filter=base_cfg.year_filter,
                overwrite=base_cfg.overwrite,
                run_name=base_cfg.run_name,
                note=base_cfg.note,
            )
        )
        print(f"run_dir={artifacts.run_dir}")
        print(f"run_manifest={artifacts.run_manifest_path}")
        print(f"sentence_dataset_dir={artifacts.sentence_dataset_dir}")
        print(f"yearly_summary={artifacts.yearly_summary_path}")
        return 0

    run_cfg = _resolve_run_config(args)
    artifacts = run_finbert_item_analysis(run_cfg)
    print(f"run_dir={artifacts.run_dir}")
    print(f"run_manifest={artifacts.run_manifest_path}")
    print(f"item_features_long={artifacts.item_features_long_path}")
    print(f"doc_features_wide={artifacts.doc_features_wide_path}")
    if artifacts.coverage_report_path is not None:
        print(f"coverage_report={artifacts.coverage_report_path}")
    if artifacts.sentence_scores_dir is not None:
        print(f"sentence_scores_dir={artifacts.sentence_scores_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

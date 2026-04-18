from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


REPO_ROOT_ENV_VAR = "NLP_THESIS_REPO_ROOT"
IN_COLAB = "google.colab" in sys.modules


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


def _resolve_colab_drive_root() -> Path:
    for candidate in (
        Path("/content/drive/MyDrive"),
        Path("/content/drive/My Drive"),
        Path("/content/drive"),
    ):
        if candidate.exists():
            return candidate
    return Path("/content/drive")


ROOT = _resolve_repo_root()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from thesis_pkg.benchmarking import BucketBatchConfig
from thesis_pkg.benchmarking import FinbertAnalysisRunConfig
from thesis_pkg.benchmarking import FinbertSentenceParquetInferenceRunArtifacts
from thesis_pkg.benchmarking import FinbertSentenceParquetInferenceRunConfig
from thesis_pkg.benchmarking import FinbertSentencePreprocessingRunConfig
from thesis_pkg.benchmarking import FinbertSentencePreprocessingRunArtifacts
from thesis_pkg.benchmarking import FinbertSectionUniverseConfig
from thesis_pkg.benchmarking import FinbertRuntimeConfig
from thesis_pkg.benchmarking import ALLOWED_SENTENCE_POSTPROCESS_POLICIES
from thesis_pkg.benchmarking import BucketEdgeSpec
from thesis_pkg.benchmarking import DEFAULT_RUNNER_SENTENCE_POSTPROCESS_POLICY
from thesis_pkg.benchmarking import SentenceDatasetConfig
from thesis_pkg.benchmarking import run_finbert_sentence_parquet_inference
from thesis_pkg.benchmarking import run_finbert_sentence_preprocessing
from thesis_pkg.benchmarking import resolve_bucket_lengths_for_edges
from thesis_pkg.benchmarking.run_logging import utc_timestamp


DEFAULT_FULL_DATA_ROOT = (
    _resolve_colab_drive_root() / "Data_LM"
    if IN_COLAB
    else ROOT / "full_data_run"
)
DEFAULT_SAMPLE_ROOT = (
    DEFAULT_FULL_DATA_ROOT
    if IN_COLAB
    else DEFAULT_FULL_DATA_ROOT / "sample_5pct_seed42"
)
DEFAULT_SAMPLE_UPSTREAM_RUN_ROOT = (
    DEFAULT_SAMPLE_ROOT / "results" / "sec_ccm_unified_runner"
    if IN_COLAB
    else DEFAULT_SAMPLE_ROOT / "results" / "sec_ccm_unified_runner" / "local_sample"
)
DEFAULT_SAMPLE_ITEMS_ANALYSIS_DIR = DEFAULT_SAMPLE_UPSTREAM_RUN_ROOT / "items_analysis"
DEFAULT_SAMPLE_LM2011_BACKBONE_PATH = (
    DEFAULT_SAMPLE_ROOT / "results" / "lm2011_sample_post_refinitiv_runner" / "lm2011_sample_backbone.parquet"
)
DEFAULT_SAMPLE_UNIFIED_LM2011_BACKBONE_PATH = (
    DEFAULT_SAMPLE_UPSTREAM_RUN_ROOT / "lm2011_post_refinitiv" / "lm2011_sample_backbone.parquet"
)
DEFAULT_SAMPLE_UPSTREAM_LM2011_BACKBONE_PATH = (
    DEFAULT_SAMPLE_UPSTREAM_RUN_ROOT / "sec_ccm_premerge" / "lm2011_sample_backbone.parquet"
)
DEFAULT_SAMPLE_PREMERGE_BACKBONE_PATH = (
    DEFAULT_SAMPLE_UPSTREAM_RUN_ROOT / "sec_ccm_premerge" / "final_flagged_data.parquet"
)
DEFAULT_SAMPLE_OUTPUT_ROOT = DEFAULT_SAMPLE_ROOT / "results" / "finbert_item_analysis_runner"

BATCH_PRESETS: dict[str, BucketBatchConfig] = {
    "small": BucketBatchConfig(name="small", short_batch_size=32, medium_batch_size=16, long_batch_size=8),
    "baseline": BucketBatchConfig(name="baseline", short_batch_size=64, medium_batch_size=32, long_batch_size=16),
    "large": BucketBatchConfig(name="large", short_batch_size=128, medium_batch_size=64, long_batch_size=32),
    "xlarge": BucketBatchConfig(name="xlarge", short_batch_size=256, medium_batch_size=128, long_batch_size=64),
}
ANALYSIS_RUNNER_NAME = "finbert_item_analysis"


def _resolve_bucket_edges(
    *,
    short_edge: int | None,
    medium_edge: int | None,
) -> BucketEdgeSpec:
    base = BucketEdgeSpec()
    if short_edge is None and medium_edge is None:
        return base
    return BucketEdgeSpec(
        short_edge=short_edge if short_edge is not None else base.short_edge,
        medium_edge=medium_edge if medium_edge is not None else base.medium_edge,
    )


def _default_local_sample_backbone_path() -> Path:
    for candidate in (
        DEFAULT_SAMPLE_UNIFIED_LM2011_BACKBONE_PATH,
        DEFAULT_SAMPLE_LM2011_BACKBONE_PATH,
        DEFAULT_SAMPLE_UPSTREAM_LM2011_BACKBONE_PATH,
        DEFAULT_SAMPLE_PREMERGE_BACKBONE_PATH,
    ):
        if candidate.exists():
            return candidate
    return DEFAULT_SAMPLE_LM2011_BACKBONE_PATH


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
    parser.add_argument(
        "--sentence-postprocess-policy",
        choices=ALLOWED_SENTENCE_POSTPROCESS_POLICIES,
        default=DEFAULT_RUNNER_SENTENCE_POSTPROCESS_POLICY,
    )
    parser.add_argument("--short-edge", type=int, default=None)
    parser.add_argument("--medium-edge", type=int, default=None)
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
        backbone_path = _default_local_sample_backbone_path()
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

    bucket_edges = _resolve_bucket_edges(
        short_edge=args.short_edge,
        medium_edge=args.medium_edge,
    )
    return FinbertAnalysisRunConfig(
        source_items_dir=source_items_dir,
        out_root=output_dir,
        batch_config=BATCH_PRESETS[args.batch_profile],
        bucket_lengths=resolve_bucket_lengths_for_edges(
            bucket_edges=bucket_edges,
            short_max_length=None,
            medium_max_length=None,
            long_max_length=None,
        ),
        section_universe=FinbertSectionUniverseConfig(source_items_dir=source_items_dir),
        runtime=FinbertRuntimeConfig(device=args.device),
        sentence_dataset=SentenceDatasetConfig(
            postprocess_policy=args.sentence_postprocess_policy,
            bucket_edges=bucket_edges,
        ),
        backbone_path=backbone_path,
        year_filter=year_filter,
        write_sentence_scores=args.write_sentence_scores,
        overwrite=args.overwrite,
        run_name=args.run_name,
        note=args.note,
    )


@dataclass(frozen=True)
class FinbertPipelineRunArtifacts:
    preprocessing_artifacts: FinbertSentencePreprocessingRunArtifacts | None
    analysis_artifacts: FinbertSentenceParquetInferenceRunArtifacts | None


def _analysis_run_name(run_cfg: FinbertAnalysisRunConfig) -> str:
    return run_cfg.run_name or f"{ANALYSIS_RUNNER_NAME}_{utc_timestamp().replace(':', '')}"


def _analysis_preprocessing_run_config(
    run_cfg: FinbertAnalysisRunConfig,
    *,
    analysis_run_name: str | None,
) -> FinbertSentencePreprocessingRunConfig:
    return FinbertSentencePreprocessingRunConfig(
        source_items_dir=run_cfg.source_items_dir,
        out_root=run_cfg.out_root / "_staged_intermediates",
        section_universe=run_cfg.section_universe,
        sentence_dataset=run_cfg.sentence_dataset,
        cleaning=run_cfg.cleaning,
        target_doc_universe_path=run_cfg.backbone_path,
        year_filter=run_cfg.year_filter,
        overwrite=run_cfg.overwrite,
        run_name=(
            f"{analysis_run_name}_sentence_preprocessing"
            if analysis_run_name is not None
            else None
        ),
        note=run_cfg.note,
    )


def _resolve_existing_preprocessing_artifacts(
    run_cfg: FinbertSentencePreprocessingRunConfig,
) -> FinbertSentencePreprocessingRunArtifacts:
    if run_cfg.run_name is not None:
        run_dir = run_cfg.out_root / run_cfg.run_name
    else:
        candidates = sorted(
            path.parent.parent
            for path in run_cfg.out_root.glob("*/sentence_dataset/by_year")
            if path.is_dir()
        )
        if not candidates:
            raise FileNotFoundError(
                f"No existing FinBERT sentence preprocessing runs found under {run_cfg.out_root}"
            )
        if len(candidates) > 1:
            raise ValueError(
                "Multiple FinBERT sentence preprocessing runs exist; set SEC_CCM_FINBERT_RUN_NAME "
                "or pass an explicit preprocessing run_name."
            )
        run_dir = candidates[0]
    sentence_dataset_dir = run_dir / "sentence_dataset" / "by_year"
    run_manifest_path = run_dir / "run_manifest.json"
    yearly_summary_path = run_dir / "sentence_dataset_yearly_summary.parquet"
    if not sentence_dataset_dir.exists() or not run_manifest_path.exists():
        raise FileNotFoundError(
            f"Expected FinBERT sentence preprocessing artifacts were not found in {run_dir}"
        )
    return FinbertSentencePreprocessingRunArtifacts(
        run_dir=run_dir,
        run_manifest_path=run_manifest_path,
        sentence_dataset_dir=sentence_dataset_dir,
        yearly_summary_path=yearly_summary_path,
        oversize_sections_path=(run_dir / "oversize_sections.parquet"),
        cleaned_item_scopes_dir=(run_dir / "cleaned_item_scopes" / "by_year"),
        cleaning_row_audit_path=(run_dir / "cleaning_row_audit.parquet"),
        cleaning_flagged_rows_path=(run_dir / "cleaning_flagged_rows.parquet"),
        item_scope_cleaning_diagnostics_path=(run_dir / "item_scope_cleaning_diagnostics.parquet"),
        manual_boundary_audit_sample_path=(run_dir / "manual_boundary_audit_sample.parquet"),
    )


def run_finbert_pipeline(
    analysis_cfg: FinbertAnalysisRunConfig,
    *,
    preprocessing_cfg: FinbertSentencePreprocessingRunConfig | None = None,
    run_preprocess: bool = True,
    run_analysis: bool = True,
) -> FinbertPipelineRunArtifacts:
    if not run_preprocess and not run_analysis:
        raise ValueError("At least one of run_preprocess or run_analysis must be True.")

    analysis_run_name = _analysis_run_name(analysis_cfg)
    resolved_preprocessing_cfg = preprocessing_cfg or _analysis_preprocessing_run_config(
        analysis_cfg,
        analysis_run_name=(
            None
            if run_analysis and not run_preprocess and analysis_cfg.run_name is None
            else analysis_run_name
        ),
    )

    preprocessing_artifacts: FinbertSentencePreprocessingRunArtifacts | None = None
    if run_preprocess:
        preprocessing_artifacts = run_finbert_sentence_preprocessing(resolved_preprocessing_cfg)
    elif run_analysis:
        preprocessing_artifacts = _resolve_existing_preprocessing_artifacts(
            resolved_preprocessing_cfg
        )

    analysis_artifacts: FinbertSentenceParquetInferenceRunArtifacts | None = None
    if run_analysis:
        if preprocessing_artifacts is None:
            raise RuntimeError(
                "FinBERT analysis requires preprocessing artifacts, but none were resolved."
            )
        analysis_artifacts = run_finbert_sentence_parquet_inference(
            FinbertSentenceParquetInferenceRunConfig(
                sentence_dataset_dir=preprocessing_artifacts.sentence_dataset_dir,
                out_root=analysis_cfg.out_root,
                batch_config=analysis_cfg.batch_config,
                runtime=analysis_cfg.runtime,
                bucket_lengths=analysis_cfg.bucket_lengths,
                backbone_path=analysis_cfg.backbone_path,
                year_filter=analysis_cfg.year_filter,
                write_sentence_scores=analysis_cfg.write_sentence_scores,
                overwrite=analysis_cfg.overwrite,
                run_name=analysis_run_name,
                note=analysis_cfg.note,
            )
        )

    return FinbertPipelineRunArtifacts(
        preprocessing_artifacts=preprocessing_artifacts,
        analysis_artifacts=analysis_artifacts,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_cfg = _resolve_run_config(args)
    if args.preprocess_only:
        artifacts = run_finbert_pipeline(
            run_cfg,
            preprocessing_cfg=FinbertSentencePreprocessingRunConfig(
                source_items_dir=run_cfg.source_items_dir,
                out_root=run_cfg.out_root,
                section_universe=run_cfg.section_universe,
                sentence_dataset=run_cfg.sentence_dataset,
                cleaning=run_cfg.cleaning,
                target_doc_universe_path=run_cfg.backbone_path,
                year_filter=run_cfg.year_filter,
                overwrite=run_cfg.overwrite,
                run_name=run_cfg.run_name,
                note=run_cfg.note,
            ),
            run_preprocess=True,
            run_analysis=False,
        )
        preprocessing_artifacts = artifacts.preprocessing_artifacts
        if preprocessing_artifacts is None:
            raise RuntimeError("FinBERT preprocessing did not produce artifacts.")
        print(f"run_dir={preprocessing_artifacts.run_dir}")
        print(f"run_manifest={preprocessing_artifacts.run_manifest_path}")
        print(f"sentence_dataset_dir={preprocessing_artifacts.sentence_dataset_dir}")
        print(f"yearly_summary={preprocessing_artifacts.yearly_summary_path}")
        return 0

    artifacts = run_finbert_pipeline(run_cfg, run_preprocess=True, run_analysis=True)
    analysis_artifacts = artifacts.analysis_artifacts
    if analysis_artifacts is None:
        raise RuntimeError("FinBERT analysis did not produce artifacts.")
    print(f"run_dir={analysis_artifacts.run_dir}")
    print(f"run_manifest={analysis_artifacts.run_manifest_path}")
    print(f"item_features_long={analysis_artifacts.item_features_long_path}")
    print(f"doc_features_wide={analysis_artifacts.doc_features_wide_path}")
    if analysis_artifacts.coverage_report_path is not None:
        print(f"coverage_report={analysis_artifacts.coverage_report_path}")
    if analysis_artifacts.sentence_scores_dir is not None:
        print(f"sentence_scores_dir={analysis_artifacts.sentence_scores_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

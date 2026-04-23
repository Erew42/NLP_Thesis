from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Sequence

import polars as pl


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

from thesis_pkg.benchmarking.finbert_tail_features import build_finbert_tail_doc_surface_lf
from thesis_pkg.benchmarking.finbert_tail_features import TAIL_FEATURE_COLUMNS
from thesis_pkg.benchmarking.finbert_tail_features import TAIL_DOC_SURFACE_SCHEMA
from thesis_pkg.benchmarking.manifest_contracts import MANIFEST_PATH_SEMANTICS_RELATIVE
from thesis_pkg.benchmarking.manifest_contracts import resolve_manifest_path
from thesis_pkg.benchmarking.manifest_contracts import write_manifest_path_value
from thesis_pkg.benchmarking.run_logging import utc_timestamp
from thesis_pkg.benchmarking.run_logging import write_frame
from thesis_pkg.benchmarking.run_logging import write_json
from thesis_pkg.pipelines.lm2011_extension import EXTENSION_PRIMARY_OUTCOME
from thesis_pkg.pipelines.lm2011_extension import EXTENSION_PRIMARY_TEXT_SCOPES
from thesis_pkg.pipelines.lm2011_extension import run_lm2011_extension_estimation_scaffold
from thesis_pkg.pipelines.lm2011_extension import run_lm2011_extension_fit_comparison_scaffold


RUNNER_NAME = "lm2011_finbert_robustness_runner"
MANIFEST_FILENAME = "finbert_robustness_run_manifest.json"
EXTENSION_MANIFEST_FILENAME = "lm2011_extension_run_manifest.json"
FINBERT_ANALYSIS_MANIFEST_FILENAME = "run_manifest.json"
EXTENSION_ANALYSIS_PANEL_FILENAME = "lm2011_extension_analysis_panel.parquet"
FINBERT_SENTENCE_SCORES_DIRNAME = "sentence_scores/by_year"
PARQUET_COMPRESSION = "zstd"
PRIMARY_CONTROL_SET_IDS: tuple[str, ...] = ("C0",)
PRIMARY_OUTCOME_NAMES: tuple[str, ...] = (EXTENSION_PRIMARY_OUTCOME,)
PRIMARY_TEXT_SCOPES: tuple[str, ...] = EXTENSION_PRIMARY_TEXT_SCOPES
PRIMARY_SPECIFICATION_NAMES: tuple[str, ...] = (
    "dictionary_only",
    "finbert_only",
    "dictionary_finbert_joint",
)
QUARTER_WEIGHTINGS: tuple[str, ...] = (
    "quarter_observation_count",
    "equal_quarter",
)
REPLICATION_DICTIONARY_FAMILY_SOURCE = "replication"
EXISTING_SCALE_FAMILY = "existing_scale"
TAIL_SIGNAL_FAMILY = "tail_signal"
ARTIFACT_FILENAMES: dict[str, str] = {
    "existing_scale_coefficients": "finbert_robustness_existing_scale_coefficients.parquet",
    "existing_scale_fit_summary": "finbert_robustness_existing_scale_fit_summary.parquet",
    "existing_scale_fit_comparisons": "finbert_robustness_existing_scale_fit_comparisons.parquet",
    "existing_scale_fit_skipped_quarters": "finbert_robustness_existing_scale_fit_skipped_quarters.parquet",
    "tail_doc_surface": "finbert_robustness_tail_doc_surface.parquet",
    "tail_coefficients": "finbert_robustness_tail_coefficients.parquet",
    "tail_fit_summary": "finbert_robustness_tail_fit_summary.parquet",
    "tail_fit_comparisons": "finbert_robustness_tail_fit_comparisons.parquet",
    "tail_fit_skipped_quarters": "finbert_robustness_tail_fit_skipped_quarters.parquet",
    "candidate_summary": "finbert_robustness_candidate_summary.parquet",
}
TAIL_DOC_SURFACE_BY_YEAR_DIRNAME = "finbert_robustness_tail_doc_surface_by_year"
_VARIANT_METADATA_COLUMNS: tuple[str, ...] = (
    "variant_family",
    "variant_id",
    "variant_description",
)


@dataclass(frozen=True)
class FinbertRobustnessVariant:
    variant_family: str
    variant_id: str
    description: str
    source_columns: tuple[str, ...]


@dataclass(frozen=True)
class FinbertRobustnessRunConfig:
    extension_run_dir: Path
    finbert_analysis_run_dir: Path
    output_dir: Path
    extension_analysis_panel_path: Path | None = None
    finbert_sentence_scores_dir: Path | None = None
    run_name: str | None = None
    note: str = ""


@dataclass(frozen=True)
class FinbertRobustnessRunArtifacts:
    run_dir: Path
    manifest_path: Path
    existing_scale_coefficients_path: Path
    existing_scale_fit_summary_path: Path
    existing_scale_fit_comparisons_path: Path
    existing_scale_fit_skipped_quarters_path: Path
    tail_doc_surface_path: Path
    tail_doc_surface_by_year_dir: Path
    tail_coefficients_path: Path
    tail_fit_summary_path: Path
    tail_fit_comparisons_path: Path
    tail_fit_skipped_quarters_path: Path
    candidate_summary_path: Path


@dataclass(frozen=True)
class TailDocSurfaceArtifacts:
    stacked_path: Path
    by_year_dir: Path
    row_count: int


EXISTING_SCALE_VARIANTS: tuple[FinbertRobustnessVariant, ...] = (
    FinbertRobustnessVariant(
        variant_family=EXISTING_SCALE_FAMILY,
        variant_id="baseline_neg_mean",
        description="Existing length-weighted mean negative probability.",
        source_columns=("finbert_neg_prob_lenw_mean",),
    ),
    FinbertRobustnessVariant(
        variant_family=EXISTING_SCALE_FAMILY,
        variant_id="net_negative_mean",
        description="Existing length-weighted net negativity.",
        source_columns=("finbert_net_negative_lenw_mean",),
    ),
    FinbertRobustnessVariant(
        variant_family=EXISTING_SCALE_FAMILY,
        variant_id="neg_dominant_share",
        description="Existing share of segments classified as negative-dominant.",
        source_columns=("finbert_neg_dominant_share",),
    ),
    FinbertRobustnessVariant(
        variant_family=EXISTING_SCALE_FAMILY,
        variant_id="neg_minus_pos_mean",
        description="On-the-fly negative-minus-positive length-weighted mean.",
        source_columns=("finbert_neg_prob_lenw_mean", "finbert_pos_prob_lenw_mean"),
    ),
    FinbertRobustnessVariant(
        variant_family=EXISTING_SCALE_FAMILY,
        variant_id="neg_mean_logit_clipped",
        description="Logit transform of clipped mean negative probability.",
        source_columns=("finbert_neg_prob_lenw_mean",),
    ),
)
TAIL_VARIANTS: tuple[FinbertRobustnessVariant, ...] = tuple(
    FinbertRobustnessVariant(
        variant_family=TAIL_SIGNAL_FAMILY,
        variant_id=column,
        description=f"Tail-signal variant {column}.",
        source_columns=(column,),
    )
    for column in TAIL_FEATURE_COLUMNS
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run C0-first FinBERT robustness analyses from saved LM2011 extension and FinBERT artifacts."
    )
    parser.add_argument("--extension-run-dir", type=Path, required=True)
    parser.add_argument("--finbert-analysis-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--extension-analysis-panel-path", type=Path, default=None)
    parser.add_argument("--finbert-sentence-scores-dir", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--note", type=str, default="")
    return parser.parse_args(argv)


def _read_json_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_required_path(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def _resolve_manifest_artifact_path(
    *,
    manifest_path: Path,
    raw_path: object,
    path_semantics: str | None,
) -> Path | None:
    if raw_path is None:
        return None
    resolved = resolve_manifest_path(
        raw_path,
        manifest_path=manifest_path,
        path_semantics=path_semantics,
    )
    if resolved is None:
        return None
    return resolved.resolve()


def _resolve_extension_inputs(
    cfg: FinbertRobustnessRunConfig,
) -> dict[str, Any]:
    run_dir = _resolve_required_path(cfg.extension_run_dir, label="extension_run_dir")
    manifest_path = run_dir / EXTENSION_MANIFEST_FILENAME
    manifest = _read_json_payload(manifest_path) if manifest_path.exists() else None
    path_semantics = None if manifest is None else manifest.get("path_semantics")

    analysis_panel_path = (
        _resolve_required_path(cfg.extension_analysis_panel_path, label="extension_analysis_panel_path")
        if cfg.extension_analysis_panel_path is not None
        else None
    )
    if analysis_panel_path is None:
        replication_candidate = run_dir / REPLICATION_DICTIONARY_FAMILY_SOURCE / EXTENSION_ANALYSIS_PANEL_FILENAME
        if replication_candidate.exists():
            analysis_panel_path = replication_candidate.resolve()
        elif manifest is not None:
            analysis_panel_path = _resolve_manifest_artifact_path(
                manifest_path=manifest_path,
                raw_path=((manifest.get("stages") or {}).get("extension_analysis_panel") or {}).get("artifact_path"),
                path_semantics=path_semantics,
            )
        if analysis_panel_path is None or not analysis_panel_path.exists():
            conventional_candidate = run_dir / EXTENSION_ANALYSIS_PANEL_FILENAME
            analysis_panel_path = _resolve_required_path(
                conventional_candidate,
                label="lm2011_extension_analysis_panel.parquet",
            )

    return {
        "run_dir": run_dir,
        "manifest_path": manifest_path.resolve() if manifest_path.exists() else None,
        "manifest": manifest,
        "analysis_panel_path": analysis_panel_path.resolve(),
    }


def _resolve_finbert_analysis_inputs(
    cfg: FinbertRobustnessRunConfig,
) -> dict[str, Any]:
    run_dir = _resolve_required_path(cfg.finbert_analysis_run_dir, label="finbert_analysis_run_dir")
    manifest_path = run_dir / FINBERT_ANALYSIS_MANIFEST_FILENAME
    manifest = _read_json_payload(manifest_path) if manifest_path.exists() else None
    path_semantics = None if manifest is None else manifest.get("path_semantics")
    manifest_artifacts = {} if manifest is None else (manifest.get("artifacts") or {})

    sentence_scores_dir = (
        _resolve_required_path(cfg.finbert_sentence_scores_dir, label="finbert_sentence_scores_dir")
        if cfg.finbert_sentence_scores_dir is not None
        else None
    )
    if sentence_scores_dir is None and manifest is not None:
        sentence_scores_dir = _resolve_manifest_artifact_path(
            manifest_path=manifest_path,
            raw_path=manifest_artifacts.get("sentence_scores_dir"),
            path_semantics=path_semantics,
        )
    if sentence_scores_dir is None or not sentence_scores_dir.exists():
        sentence_scores_dir = _resolve_required_path(
            run_dir / FINBERT_SENTENCE_SCORES_DIRNAME,
            label=FINBERT_SENTENCE_SCORES_DIRNAME,
        )

    return {
        "run_dir": run_dir,
        "manifest_path": manifest_path.resolve() if manifest_path.exists() else None,
        "manifest": manifest,
        "sentence_scores_dir": sentence_scores_dir.resolve(),
    }


def _validate_unique_doc_scope(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    label: str,
) -> None:
    duplicate_keys = (
        (df.lazy() if isinstance(df, pl.DataFrame) else df)
        .group_by("doc_id", "text_scope")
        .agg(pl.len().alias("row_count"))
        .filter(pl.col("row_count") > 1)
        .limit(1)
        .collect()
    )
    if duplicate_keys.height > 0:
        raise ValueError(
            f"{label} must be unique on doc_id and text_scope; found duplicates such as "
            f"{duplicate_keys.row(0, named=True)}."
        )


def _effective_dictionary_family_source(panel_lf: pl.LazyFrame) -> tuple[pl.LazyFrame, str | None]:
    schema = panel_lf.collect_schema()
    if "dictionary_family_source" not in schema:
        return panel_lf, None
    available = (
        panel_lf.select(
            pl.col("dictionary_family_source")
            .cast(pl.Utf8, strict=False)
            .drop_nulls()
            .unique()
            .sort()
            .alias("dictionary_family_source")
        )
        .collect()
        .get_column("dictionary_family_source")
        .to_list()
    )
    if not available:
        return panel_lf, None
    if REPLICATION_DICTIONARY_FAMILY_SOURCE in available:
        return (
            panel_lf.filter(
                pl.col("dictionary_family_source") == pl.lit(REPLICATION_DICTIONARY_FAMILY_SOURCE)
            ),
            REPLICATION_DICTIONARY_FAMILY_SOURCE,
        )
    if len(available) == 1:
        return panel_lf, str(available[0])
    raise ValueError(
        "extension analysis panel contains multiple dictionary_family_source values "
        f"{available!r}; this runner only auto-resolves the replication family."
    )


def _existing_scale_variant_expr(variant: FinbertRobustnessVariant) -> pl.Expr:
    if variant.variant_id == "baseline_neg_mean":
        return pl.col("finbert_neg_prob_lenw_mean").cast(pl.Float64, strict=False)
    if variant.variant_id == "net_negative_mean":
        return pl.col("finbert_net_negative_lenw_mean").cast(pl.Float64, strict=False)
    if variant.variant_id == "neg_dominant_share":
        return pl.col("finbert_neg_dominant_share").cast(pl.Float64, strict=False)
    if variant.variant_id == "neg_minus_pos_mean":
        return (
            pl.col("finbert_neg_prob_lenw_mean").cast(pl.Float64, strict=False)
            - pl.col("finbert_pos_prob_lenw_mean").cast(pl.Float64, strict=False)
        )
    if variant.variant_id == "neg_mean_logit_clipped":
        eps = 1e-6
        raw = pl.col("finbert_neg_prob_lenw_mean").cast(pl.Float64, strict=False)
        clipped = (
            pl.when(raw.is_null())
            .then(pl.lit(None, dtype=pl.Float64))
            .when(raw < eps)
            .then(pl.lit(eps, dtype=pl.Float64))
            .when(raw > 1.0 - eps)
            .then(pl.lit(1.0 - eps, dtype=pl.Float64))
            .otherwise(raw)
        )
        return (clipped / (1.0 - clipped)).log()
    raise ValueError(f"Unsupported existing-scale variant: {variant.variant_id}")


def _annotate_variant_df(df: pl.DataFrame, variant: FinbertRobustnessVariant) -> pl.DataFrame:
    return df.with_columns(
        pl.lit(variant.variant_family, dtype=pl.Utf8).alias("variant_family"),
        pl.lit(variant.variant_id, dtype=pl.Utf8).alias("variant_id"),
        pl.lit(variant.description, dtype=pl.Utf8).alias("variant_description"),
    )


def _collect_variant_frames(frames: Sequence[pl.DataFrame]) -> pl.DataFrame:
    if not frames:
        return pl.DataFrame()
    nonempty = [frame for frame in frames if frame.height > 0]
    if not nonempty:
        return frames[0].head(0)
    return pl.concat(nonempty, how="vertical_relaxed")


def _run_variant_suite(
    *,
    base_panel_df: pl.DataFrame,
    variants: Sequence[FinbertRobustnessVariant],
    alias_expr_builder: Any,
    suite_name: str,
) -> dict[str, pl.DataFrame]:
    coefficient_frames: list[pl.DataFrame] = []
    fit_summary_frames: list[pl.DataFrame] = []
    fit_comparison_frames: list[pl.DataFrame] = []
    fit_skipped_frames: list[pl.DataFrame] = []

    for variant in variants:
        panel_variant_lf = base_panel_df.lazy().with_columns(
            alias_expr_builder(variant).alias("finbert_neg_prob_lenw_mean")
        )
        for quarter_weighting in QUARTER_WEIGHTINGS:
            coefficient_frames.append(
                _annotate_variant_df(
                    run_lm2011_extension_estimation_scaffold(
                        panel_variant_lf,
                        run_id=f"{suite_name}:{variant.variant_id}:coefficients:{quarter_weighting}",
                        text_scopes=PRIMARY_TEXT_SCOPES,
                        outcome_names=PRIMARY_OUTCOME_NAMES,
                        control_set_ids=PRIMARY_CONTROL_SET_IDS,
                        specification_names=PRIMARY_SPECIFICATION_NAMES,
                        quarter_weighting=quarter_weighting,
                    ),
                    variant,
                )
            )

        fit_artifacts = run_lm2011_extension_fit_comparison_scaffold(
            panel_variant_lf,
            run_id=f"{suite_name}:{variant.variant_id}:fit_weighted",
            text_scopes=PRIMARY_TEXT_SCOPES,
            outcome_names=PRIMARY_OUTCOME_NAMES,
            control_set_ids=PRIMARY_CONTROL_SET_IDS,
            specification_names=PRIMARY_SPECIFICATION_NAMES,
        )
        fit_summary_frames.append(_annotate_variant_df(fit_artifacts.summary_df, variant))
        fit_comparison_frames.append(_annotate_variant_df(fit_artifacts.comparison_df, variant))
        fit_skipped_frames.append(_annotate_variant_df(fit_artifacts.skipped_quarters_df, variant))

    return {
        "coefficients": _collect_variant_frames(coefficient_frames),
        "fit_summary": _collect_variant_frames(fit_summary_frames),
        "fit_comparisons": _collect_variant_frames(fit_comparison_frames),
        "fit_skipped_quarters": _collect_variant_frames(fit_skipped_frames),
    }


def _resolve_sentence_score_year_paths(sentence_scores_dir: Path) -> list[Path]:
    year_paths = sorted(path.resolve() for path in sentence_scores_dir.glob("*.parquet"))
    if not year_paths:
        raise FileNotFoundError(f"sentence_scores_dir does not contain any yearly parquet shards: {sentence_scores_dir}")
    return year_paths


def _scan_tail_doc_surface_paths(paths: Sequence[Path]) -> pl.LazyFrame:
    if not paths:
        return pl.DataFrame(schema=TAIL_DOC_SURFACE_SCHEMA).lazy()
    scanned = pl.scan_parquet([str(path) for path in paths])
    scanned_schema = scanned.collect_schema()
    return scanned.select(
        [
            (
                pl.col(name).cast(dtype, strict=False).alias(name)
                if name in scanned_schema
                else pl.lit(None, dtype=dtype).alias(name)
            )
            for name, dtype in TAIL_DOC_SURFACE_SCHEMA.items()
        ]
    )


def _build_tail_doc_surfaces(
    *,
    sentence_scores_dir: Path,
    output_dir: Path,
) -> TailDocSurfaceArtifacts:
    by_year_dir = output_dir / TAIL_DOC_SURFACE_BY_YEAR_DIRNAME
    by_year_dir.mkdir(parents=True, exist_ok=True)
    stacked_path = output_dir / ARTIFACT_FILENAMES["tail_doc_surface"]
    written_year_paths: list[Path] = []
    for year_path in _resolve_sentence_score_year_paths(sentence_scores_dir):
        year_df = build_finbert_tail_doc_surface_lf(
            pl.scan_parquet(str(year_path)),
            text_scopes=PRIMARY_TEXT_SCOPES,
        ).collect()
        _validate_unique_doc_scope(year_df, label=f"tail_doc_surface[{year_path.stem}]")
        output_path = by_year_dir / year_path.name
        write_frame(
            year_df if year_df.height > 0 else pl.DataFrame(schema=TAIL_DOC_SURFACE_SCHEMA),
            output_path,
        )
        written_year_paths.append(output_path)
    stacked_lf = _scan_tail_doc_surface_paths(written_year_paths)
    _validate_unique_doc_scope(stacked_lf, label="tail_doc_surface")
    if stacked_path.exists():
        stacked_path.unlink()
    stacked_lf.sink_parquet(stacked_path, compression=PARQUET_COMPRESSION)
    row_count = int(
        pl.scan_parquet(str(stacked_path))
        .select(pl.len().alias("row_count"))
        .collect()
        .item()
    )
    return TailDocSurfaceArtifacts(
        stacked_path=stacked_path,
        by_year_dir=by_year_dir,
        row_count=row_count,
    )


def _tail_variant_expr(variant: FinbertRobustnessVariant) -> pl.Expr:
    return pl.col(variant.variant_id).cast(pl.Float64, strict=False)


def _build_candidate_summary(
    *,
    coefficient_df: pl.DataFrame,
    fit_summary_df: pl.DataFrame,
) -> pl.DataFrame:
    if coefficient_df.width == 0:
        return pl.DataFrame()
    summary_subset = fit_summary_df.select(
        *(_VARIANT_METADATA_COLUMNS),
        "text_scope",
        "outcome_name",
        "control_set_id",
        "specification_name",
        "total_n_obs",
        "weighted_avg_raw_r2",
        "weighted_avg_adj_r2",
        "equal_quarter_avg_raw_r2",
        "equal_quarter_avg_adj_r2",
        "estimator_status",
    ).rename({"estimator_status": "fit_estimator_status"})
    candidate_df = (
        coefficient_df.filter(
            (pl.col("coefficient_name") == "finbert_neg_prob_lenw_mean")
            & pl.col("specification_name").is_in(("finbert_only", "dictionary_finbert_joint"))
            & (pl.col("estimator_status") == "estimated")
        )
        .join(
            summary_subset,
            on=[
                "variant_family",
                "variant_id",
                "variant_description",
                "text_scope",
                "outcome_name",
                "control_set_id",
                "specification_name",
            ],
            how="left",
        )
        .sort("variant_family", "variant_id", "text_scope", "weighting_rule", "specification_name")
    )
    if candidate_df.height > 0:
        return candidate_df
    return candidate_df.head(0)


def _manifest_path_value(path: Path, *, manifest_path: Path) -> str:
    return write_manifest_path_value(
        path,
        manifest_path=manifest_path,
        path_semantics=MANIFEST_PATH_SEMANTICS_RELATIVE,
    )


def run_lm2011_finbert_robustness(
    cfg: FinbertRobustnessRunConfig,
) -> FinbertRobustnessRunArtifacts:
    extension_inputs = _resolve_extension_inputs(cfg)
    finbert_inputs = _resolve_finbert_analysis_inputs(cfg)

    run_dir = cfg.output_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / MANIFEST_FILENAME
    run_name = cfg.run_name or f"finbert_robustness_{utc_timestamp().replace(':', '')}"
    started_at_utc = utc_timestamp()

    base_panel_lf, dictionary_family_source = _effective_dictionary_family_source(
        pl.scan_parquet(str(extension_inputs["analysis_panel_path"]))
    )
    base_panel_df = base_panel_lf.collect()
    _validate_unique_doc_scope(base_panel_df, label="extension_analysis_panel")

    existing_scale_outputs = _run_variant_suite(
        base_panel_df=base_panel_df,
        variants=EXISTING_SCALE_VARIANTS,
        alias_expr_builder=_existing_scale_variant_expr,
        suite_name=f"{run_name}:existing_scale",
    )

    tail_surface_artifacts = _build_tail_doc_surfaces(
        sentence_scores_dir=finbert_inputs["sentence_scores_dir"],
        output_dir=run_dir,
    )
    tail_panel_df = (
        base_panel_df.lazy()
        .join(
            pl.scan_parquet(str(tail_surface_artifacts.stacked_path)).select(
                "doc_id",
                "text_scope",
                *TAIL_FEATURE_COLUMNS,
            ),
            on=["doc_id", "text_scope"],
            how="left",
        )
        .collect()
    )
    tail_outputs = _run_variant_suite(
        base_panel_df=tail_panel_df,
        variants=TAIL_VARIANTS,
        alias_expr_builder=_tail_variant_expr,
        suite_name=f"{run_name}:tail",
    )

    coefficient_df = _collect_variant_frames(
        [
            existing_scale_outputs["coefficients"],
            tail_outputs["coefficients"],
        ]
    )
    fit_summary_df = _collect_variant_frames(
        [
            existing_scale_outputs["fit_summary"],
            tail_outputs["fit_summary"],
        ]
    )
    candidate_summary_df = _build_candidate_summary(
        coefficient_df=coefficient_df,
        fit_summary_df=fit_summary_df,
    )

    artifacts = FinbertRobustnessRunArtifacts(
        run_dir=run_dir,
        manifest_path=manifest_path,
        existing_scale_coefficients_path=run_dir / ARTIFACT_FILENAMES["existing_scale_coefficients"],
        existing_scale_fit_summary_path=run_dir / ARTIFACT_FILENAMES["existing_scale_fit_summary"],
        existing_scale_fit_comparisons_path=run_dir / ARTIFACT_FILENAMES["existing_scale_fit_comparisons"],
        existing_scale_fit_skipped_quarters_path=run_dir / ARTIFACT_FILENAMES["existing_scale_fit_skipped_quarters"],
        tail_doc_surface_path=tail_surface_artifacts.stacked_path,
        tail_doc_surface_by_year_dir=tail_surface_artifacts.by_year_dir,
        tail_coefficients_path=run_dir / ARTIFACT_FILENAMES["tail_coefficients"],
        tail_fit_summary_path=run_dir / ARTIFACT_FILENAMES["tail_fit_summary"],
        tail_fit_comparisons_path=run_dir / ARTIFACT_FILENAMES["tail_fit_comparisons"],
        tail_fit_skipped_quarters_path=run_dir / ARTIFACT_FILENAMES["tail_fit_skipped_quarters"],
        candidate_summary_path=run_dir / ARTIFACT_FILENAMES["candidate_summary"],
    )

    write_frame(existing_scale_outputs["coefficients"], artifacts.existing_scale_coefficients_path)
    write_frame(existing_scale_outputs["fit_summary"], artifacts.existing_scale_fit_summary_path)
    write_frame(existing_scale_outputs["fit_comparisons"], artifacts.existing_scale_fit_comparisons_path)
    write_frame(existing_scale_outputs["fit_skipped_quarters"], artifacts.existing_scale_fit_skipped_quarters_path)
    write_frame(tail_outputs["coefficients"], artifacts.tail_coefficients_path)
    write_frame(tail_outputs["fit_summary"], artifacts.tail_fit_summary_path)
    write_frame(tail_outputs["fit_comparisons"], artifacts.tail_fit_comparisons_path)
    write_frame(tail_outputs["fit_skipped_quarters"], artifacts.tail_fit_skipped_quarters_path)
    write_frame(candidate_summary_df, artifacts.candidate_summary_path)

    completed_at_utc = utc_timestamp()
    manifest_payload = {
        "runner_name": RUNNER_NAME,
        "run_name": run_name,
        "path_semantics": MANIFEST_PATH_SEMANTICS_RELATIVE,
        "created_at_utc": completed_at_utc,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "note": cfg.note,
        "effective_dictionary_family_source": dictionary_family_source,
        "text_scopes": list(PRIMARY_TEXT_SCOPES),
        "control_set_ids": list(PRIMARY_CONTROL_SET_IDS),
        "outcome_names": list(PRIMARY_OUTCOME_NAMES),
        "quarter_weightings": list(QUARTER_WEIGHTINGS),
        "variant_registry": {
            "existing_scale": [
                {
                    "variant_id": variant.variant_id,
                    "description": variant.description,
                    "source_columns": list(variant.source_columns),
                }
                for variant in EXISTING_SCALE_VARIANTS
            ],
            "tail_signal": [
                {
                    "variant_id": variant.variant_id,
                    "description": variant.description,
                    "source_columns": list(variant.source_columns),
                }
                for variant in TAIL_VARIANTS
            ],
        },
        "resolved_inputs": {
            "extension_run_dir": str(extension_inputs["run_dir"]),
            "extension_manifest_path": (
                str(extension_inputs["manifest_path"])
                if extension_inputs["manifest_path"] is not None
                else None
            ),
            "extension_analysis_panel_path": str(extension_inputs["analysis_panel_path"]),
            "finbert_analysis_run_dir": str(finbert_inputs["run_dir"]),
            "finbert_analysis_manifest_path": (
                str(finbert_inputs["manifest_path"])
                if finbert_inputs["manifest_path"] is not None
                else None
            ),
            "finbert_sentence_scores_dir": str(finbert_inputs["sentence_scores_dir"]),
        },
        "artifacts": {
            "existing_scale_coefficients_path": _manifest_path_value(
                artifacts.existing_scale_coefficients_path,
                manifest_path=manifest_path,
            ),
            "existing_scale_fit_summary_path": _manifest_path_value(
                artifacts.existing_scale_fit_summary_path,
                manifest_path=manifest_path,
            ),
            "existing_scale_fit_comparisons_path": _manifest_path_value(
                artifacts.existing_scale_fit_comparisons_path,
                manifest_path=manifest_path,
            ),
            "existing_scale_fit_skipped_quarters_path": _manifest_path_value(
                artifacts.existing_scale_fit_skipped_quarters_path,
                manifest_path=manifest_path,
            ),
            "tail_doc_surface_path": _manifest_path_value(
                artifacts.tail_doc_surface_path,
                manifest_path=manifest_path,
            ),
            "tail_doc_surface_by_year_dir": _manifest_path_value(
                artifacts.tail_doc_surface_by_year_dir,
                manifest_path=manifest_path,
            ),
            "tail_coefficients_path": _manifest_path_value(
                artifacts.tail_coefficients_path,
                manifest_path=manifest_path,
            ),
            "tail_fit_summary_path": _manifest_path_value(
                artifacts.tail_fit_summary_path,
                manifest_path=manifest_path,
            ),
            "tail_fit_comparisons_path": _manifest_path_value(
                artifacts.tail_fit_comparisons_path,
                manifest_path=manifest_path,
            ),
            "tail_fit_skipped_quarters_path": _manifest_path_value(
                artifacts.tail_fit_skipped_quarters_path,
                manifest_path=manifest_path,
            ),
            "candidate_summary_path": _manifest_path_value(
                artifacts.candidate_summary_path,
                manifest_path=manifest_path,
            ),
        },
        "row_counts": {
            "base_panel": base_panel_df.height,
            "tail_doc_surface": tail_surface_artifacts.row_count,
            "existing_scale_coefficients": existing_scale_outputs["coefficients"].height,
            "existing_scale_fit_summary": existing_scale_outputs["fit_summary"].height,
            "existing_scale_fit_comparisons": existing_scale_outputs["fit_comparisons"].height,
            "existing_scale_fit_skipped_quarters": existing_scale_outputs["fit_skipped_quarters"].height,
            "tail_coefficients": tail_outputs["coefficients"].height,
            "tail_fit_summary": tail_outputs["fit_summary"].height,
            "tail_fit_comparisons": tail_outputs["fit_comparisons"].height,
            "tail_fit_skipped_quarters": tail_outputs["fit_skipped_quarters"].height,
            "candidate_summary": candidate_summary_df.height,
        },
    }
    write_json(manifest_path, manifest_payload)
    return artifacts


def _resolved_run_config(args: argparse.Namespace) -> FinbertRobustnessRunConfig:
    return FinbertRobustnessRunConfig(
        extension_run_dir=Path(args.extension_run_dir).resolve(),
        finbert_analysis_run_dir=Path(args.finbert_analysis_run_dir).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        extension_analysis_panel_path=(
            Path(args.extension_analysis_panel_path).resolve()
            if args.extension_analysis_panel_path is not None
            else None
        ),
        finbert_sentence_scores_dir=(
            Path(args.finbert_sentence_scores_dir).resolve()
            if args.finbert_sentence_scores_dir is not None
            else None
        ),
        run_name=args.run_name,
        note=args.note,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    artifacts = run_lm2011_finbert_robustness(_resolved_run_config(args))
    print(f"run_dir={artifacts.run_dir}")
    print(f"manifest_path={artifacts.manifest_path}")
    print(f"candidate_summary_path={artifacts.candidate_summary_path}")
    print(f"tail_doc_surface_path={artifacts.tail_doc_surface_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

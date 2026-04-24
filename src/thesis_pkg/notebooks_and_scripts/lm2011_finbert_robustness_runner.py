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
from thesis_pkg.pipelines.lm2011_extension import _comparison_signal_name
from thesis_pkg.pipelines.lm2011_extension import _control_set_by_id
from thesis_pkg.pipelines.lm2011_extension import _convert_lm2011_table_results_to_extension_rows
from thesis_pkg.pipelines.lm2011_extension import _empty_extension_fit_comparisons_df
from thesis_pkg.pipelines.lm2011_extension import _empty_extension_fit_quarterly_df
from thesis_pkg.pipelines.lm2011_extension import _empty_extension_fit_skipped_quarters_df
from thesis_pkg.pipelines.lm2011_extension import _empty_extension_fit_summary_df
from thesis_pkg.pipelines.lm2011_extension import _empty_extension_results_df
from thesis_pkg.pipelines.lm2011_extension import _EXTENSION_COMMON_ROW_SAMPLE_POLICY
from thesis_pkg.pipelines.lm2011_extension import _EXTENSION_COMMON_SUCCESS_POLICY
from thesis_pkg.pipelines.lm2011_extension import _extension_fit_comparison_pairs
from thesis_pkg.pipelines.lm2011_extension import _extension_fit_comparison_status_row
from thesis_pkg.pipelines.lm2011_extension import _extension_fit_summary_status_row
from thesis_pkg.pipelines.lm2011_extension import _extension_result_status_row
from thesis_pkg.pipelines.lm2011_extension import _normal_approx_two_sided_p_value
from thesis_pkg.pipelines.lm2011_extension import apply_lm2011_extension_control_set
from thesis_pkg.pipelines.lm2011_extension import EXTENSION_DICTIONARY_FAMILY_LM2011
from thesis_pkg.pipelines.lm2011_extension import EXTENSION_FINBERT_MODEL_FAMILY
from thesis_pkg.pipelines.lm2011_extension import EXTENSION_JOINT_FEATURE_FAMILY
from thesis_pkg.pipelines.lm2011_extension import EXTENSION_PRIMARY_OUTCOME
from thesis_pkg.pipelines.lm2011_extension import EXTENSION_PRIMARY_TEXT_SCOPES
from thesis_pkg.pipelines.lm2011_extension import EXTENSION_SAMPLE_WINDOW
from thesis_pkg.pipelines.lm2011_extension import Lm2011ExtensionComparisonSpec
from thesis_pkg.pipelines.lm2011_extension import run_lm2011_extension_estimation_scaffold
from thesis_pkg.pipelines.lm2011_extension import run_lm2011_extension_fit_comparison_scaffold
from thesis_pkg.pipelines.lm2011_regressions import _newey_west_standard_error
from thesis_pkg.pipelines.lm2011_regressions import _weighted_mean
from thesis_pkg.pipelines.lm2011_regressions import run_lm2011_quarterly_fama_macbeth
from thesis_pkg.pipelines.lm2011_regressions import run_lm2011_quarterly_fama_macbeth_with_diagnostics


RUNNER_NAME = "lm2011_finbert_robustness_runner"
MANIFEST_FILENAME = "finbert_robustness_run_manifest.json"
EXTENSION_MANIFEST_FILENAME = "lm2011_extension_run_manifest.json"
FINBERT_ANALYSIS_MANIFEST_FILENAME = "run_manifest.json"
EXTENSION_ANALYSIS_PANEL_FILENAME = "lm2011_extension_analysis_panel.parquet"
FINBERT_SENTENCE_SCORES_DIRNAME = "sentence_scores/by_year"
PARQUET_COMPRESSION = "zstd"
DEFAULT_TAIL_DOC_BATCH_SIZE = 5_000
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
QUANTILE_SIGNAL_FAMILY = "quantile_signal"
QUANTILE_TRANSFORM = "within_quarter_text_scope_quintile"
QUANTILE_K = 5
REGRESSION_QUARTER_COL = "_regression_quarter"
QUANTILE_LABEL_SUFFIX = "__q5_label"
QUANTILE_SCORE_SUFFIX = "__q5_score"
QUANTILE_TOP_BOTTOM_SUFFIX = "__q5_top_bottom"
DICTIONARY_SIGNAL_COLUMN = "lm_negative_tfidf"
FILING_DATE_COL = "filing_date"
INDUSTRY_COL = "ff48_industry_id"
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
    "quantile_coefficients": "finbert_robustness_quantile_coefficients.parquet",
    "quantile_fit_summary": "finbert_robustness_quantile_fit_summary.parquet",
    "quantile_fit_comparisons": "finbert_robustness_quantile_fit_comparisons.parquet",
    "quantile_fit_skipped_quarters": "finbert_robustness_quantile_fit_skipped_quarters.parquet",
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
class FinbertQuantileVariant:
    variant_family: str
    variant_id: str
    description: str
    source_columns: tuple[str, ...]
    base_variant_id: str
    base_signal_col: str
    transform: str = QUANTILE_TRANSFORM
    quantile_k: int = QUANTILE_K


@dataclass(frozen=True)
class FinbertRobustnessRunConfig:
    extension_run_dir: Path
    finbert_analysis_run_dir: Path
    output_dir: Path
    extension_analysis_panel_path: Path | None = None
    finbert_sentence_scores_dir: Path | None = None
    tail_doc_batch_size: int = DEFAULT_TAIL_DOC_BATCH_SIZE
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
    quantile_coefficients_path: Path
    quantile_fit_summary_path: Path
    quantile_fit_comparisons_path: Path
    quantile_fit_skipped_quarters_path: Path
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
_QUANTILE_BASE_SIGNALS: tuple[tuple[str, str, str], ...] = (
    (
        "baseline_neg_mean",
        "finbert_neg_prob_lenw_mean",
        "Length-weighted mean negative probability",
    ),
    (
        "net_negative_mean",
        "finbert_net_negative_lenw_mean",
        "Length-weighted net negativity",
    ),
    (
        "top_20pct_neg_mean",
        "top_20pct_neg_mean",
        "Top-20 percent sentence negative-probability mean",
    ),
)
QUANTILE_VARIANTS: tuple[FinbertQuantileVariant, ...] = tuple(
    FinbertQuantileVariant(
        variant_family=QUANTILE_SIGNAL_FAMILY,
        variant_id=f"{base_variant_id}{suffix}",
        description=f"{description} transformed to {label}.",
        source_columns=(base_signal_col,),
        base_variant_id=base_variant_id,
        base_signal_col=base_signal_col,
    )
    for base_variant_id, base_signal_col, description in _QUANTILE_BASE_SIGNALS
    for suffix, label in (
        (QUANTILE_SCORE_SUFFIX, "within-quarter text-scope quintile score"),
        (QUANTILE_TOP_BOTTOM_SUFFIX, "top-minus-bottom quintile contrast"),
    )
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
    parser.add_argument(
        "--tail-doc-batch-size",
        type=int,
        default=DEFAULT_TAIL_DOC_BATCH_SIZE,
        help=(
            "Number of unique doc_id values per bounded tail-surface aggregation batch. "
            f"Default: {DEFAULT_TAIL_DOC_BATCH_SIZE}."
        ),
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--note", type=str, default="")
    args = parser.parse_args(argv)
    if args.tail_doc_batch_size < 1:
        parser.error("--tail-doc-batch-size must be >= 1")
    return args


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


def _regression_quarter_expr(filing_date_col: str = FILING_DATE_COL) -> pl.Expr:
    filing_date = pl.col(filing_date_col).cast(pl.Date, strict=False)
    quarter_month = (((filing_date.dt.month() - 1) // 3) * 3 + 1).cast(pl.Int8)
    return pl.date(filing_date.dt.year(), quarter_month, 1).alias(REGRESSION_QUARTER_COL)


def add_within_cell_quantile_variants_lf(
    lf: pl.LazyFrame,
    *,
    signal_col: str,
    group_cols: Sequence[str],
    variant_prefix: str,
    k: int = QUANTILE_K,
) -> pl.LazyFrame:
    label_col = f"{variant_prefix}__q{k}_label"
    score_col = f"{variant_prefix}__q{k}_score"
    top_bottom_col = f"{variant_prefix}__q{k}_top_bottom"
    rank_pct_col = f"__{variant_prefix}__q{k}_rank_pct"
    signal = pl.col(signal_col).cast(pl.Float64, strict=False)
    nonmissing_count = signal.count().over(group_cols).cast(pl.Float64)
    rank_pct = (
        pl.when(signal.is_not_null() & (nonmissing_count > 0.0))
        .then(signal.rank(method="average").over(group_cols) / nonmissing_count)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    label = (
        pl.when(pl.col(rank_pct_col).is_not_null())
        .then((pl.col(rank_pct_col) * pl.lit(float(k))).ceil().clip(1, k))
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .cast(pl.Int8, strict=False)
    )
    return (
        lf.with_columns(rank_pct.alias(rank_pct_col))
        .with_columns(label.alias(label_col))
        .with_columns(
            (
                (pl.col(label_col).cast(pl.Float64) - 1.0) / float(k - 1)
            ).alias(score_col),
            (
                pl.when(pl.col(label_col) == k)
                .then(pl.lit(0.5, dtype=pl.Float64))
                .when(pl.col(label_col) == 1)
                .then(pl.lit(-0.5, dtype=pl.Float64))
                .when(pl.col(label_col).is_not_null())
                .then(pl.lit(0.0, dtype=pl.Float64))
                .otherwise(pl.lit(None, dtype=pl.Float64))
            ).alias(top_bottom_col),
        )
        .drop(rank_pct_col)
    )


def _unique_preserving_order(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _quantile_comparison_specs(
    variant: FinbertQuantileVariant,
) -> tuple[Lm2011ExtensionComparisonSpec, ...]:
    return (
        Lm2011ExtensionComparisonSpec(
            specification_name="dictionary_only",
            feature_family=EXTENSION_DICTIONARY_FAMILY_LM2011,
            signal_inputs=(DICTIONARY_SIGNAL_COLUMN,),
        ),
        Lm2011ExtensionComparisonSpec(
            specification_name="finbert_only",
            feature_family=EXTENSION_FINBERT_MODEL_FAMILY,
            signal_inputs=(variant.variant_id,),
        ),
        Lm2011ExtensionComparisonSpec(
            specification_name="dictionary_finbert_joint",
            feature_family=EXTENSION_JOINT_FEATURE_FAMILY,
            signal_inputs=(DICTIONARY_SIGNAL_COLUMN, variant.variant_id),
        ),
    )


def _quantile_source_signal_columns(
    variant: FinbertQuantileVariant,
    comparison_spec: Lm2011ExtensionComparisonSpec,
) -> tuple[str, ...]:
    return _unique_preserving_order(
        tuple(
            variant.base_signal_col if column == variant.variant_id else column
            for column in comparison_spec.signal_inputs
        )
    )


def _quantile_signal_and_controls(
    comparison_spec: Lm2011ExtensionComparisonSpec,
    control_columns: Sequence[str],
) -> tuple[str, tuple[str, ...]]:
    signal_column = comparison_spec.signal_inputs[0]
    return signal_column, (*comparison_spec.signal_inputs[1:], *control_columns)


def _with_cast_nonmissing_quantile_base(
    panel_lf: pl.LazyFrame,
    *,
    text_scope: str,
    control_set_id: str,
    outcome_name: str,
    source_signal_columns: Sequence[str],
    control_columns: Sequence[str],
) -> pl.LazyFrame:
    float_columns = _unique_preserving_order(
        (*source_signal_columns, *control_columns, outcome_name)
    )
    return (
        apply_lm2011_extension_control_set(panel_lf, control_set_id)
        .filter(pl.col("text_scope") == pl.lit(text_scope))
        .with_columns(
            pl.col(FILING_DATE_COL).cast(pl.Date, strict=False).alias(FILING_DATE_COL),
            pl.col(INDUSTRY_COL).cast(pl.Int32, strict=False).alias(INDUSTRY_COL),
            *[
                pl.col(column).cast(pl.Float64, strict=False).alias(column)
                for column in float_columns
            ],
        )
        .select(
            "text_scope",
            FILING_DATE_COL,
            INDUSTRY_COL,
            *float_columns,
        )
        .drop_nulls(subset=[FILING_DATE_COL, INDUSTRY_COL, *float_columns])
    )


def _build_quantile_spec_panel_lf(
    panel_lf: pl.LazyFrame,
    *,
    variant: FinbertQuantileVariant,
    comparison_spec: Lm2011ExtensionComparisonSpec,
    control_set_id: str,
    control_columns: Sequence[str],
    text_scope: str,
    outcome_name: str,
) -> pl.LazyFrame:
    source_signal_columns = _quantile_source_signal_columns(variant, comparison_spec)
    spec_panel_lf = _with_cast_nonmissing_quantile_base(
        panel_lf,
        text_scope=text_scope,
        control_set_id=control_set_id,
        outcome_name=outcome_name,
        source_signal_columns=source_signal_columns,
        control_columns=control_columns,
    )
    if variant.variant_id not in comparison_spec.signal_inputs:
        return spec_panel_lf
    return add_within_cell_quantile_variants_lf(
        spec_panel_lf.with_columns(_regression_quarter_expr()),
        signal_col=variant.base_signal_col,
        group_cols=[REGRESSION_QUARTER_COL, "text_scope"],
        variant_prefix=variant.base_variant_id,
        k=variant.quantile_k,
    )


def _build_quantile_common_comparison_panel_lf(
    panel_lf: pl.LazyFrame,
    *,
    variant: FinbertQuantileVariant,
    comparison_specs: Sequence[Lm2011ExtensionComparisonSpec],
    control_set_id: str,
    control_columns: Sequence[str],
    text_scope: str,
    outcome_name: str,
) -> pl.LazyFrame:
    source_signal_columns = _unique_preserving_order(
        tuple(
            source_column
            for comparison_spec in comparison_specs
            for source_column in _quantile_source_signal_columns(variant, comparison_spec)
        )
    )
    return add_within_cell_quantile_variants_lf(
        _with_cast_nonmissing_quantile_base(
            panel_lf,
            text_scope=text_scope,
            control_set_id=control_set_id,
            outcome_name=outcome_name,
            source_signal_columns=source_signal_columns,
            control_columns=control_columns,
        ).with_columns(_regression_quarter_expr()),
        signal_col=variant.base_signal_col,
        group_cols=[REGRESSION_QUARTER_COL, "text_scope"],
        variant_prefix=variant.base_variant_id,
        k=variant.quantile_k,
    )


def _run_quantile_coefficients_for_variant(
    panel_lf: pl.LazyFrame,
    *,
    variant: FinbertQuantileVariant,
    suite_name: str,
    nw_lags: int = 1,
) -> pl.DataFrame:
    output_rows: list[dict[str, object]] = []
    comparison_specs = _quantile_comparison_specs(variant)
    for quarter_weighting in QUARTER_WEIGHTINGS:
        run_id = f"{suite_name}:{variant.variant_id}:coefficients:{quarter_weighting}"
        for text_scope in PRIMARY_TEXT_SCOPES:
            for outcome_name in PRIMARY_OUTCOME_NAMES:
                for comparison_spec in comparison_specs:
                    for control_set_id in PRIMARY_CONTROL_SET_IDS:
                        control_set = _control_set_by_id(control_set_id)
                        spec_panel_lf = _build_quantile_spec_panel_lf(
                            panel_lf,
                            variant=variant,
                            comparison_spec=comparison_spec,
                            control_set_id=control_set.control_set_id,
                            control_columns=control_set.controls,
                            text_scope=text_scope,
                            outcome_name=outcome_name,
                        )
                        signal_column, control_columns = _quantile_signal_and_controls(
                            comparison_spec,
                            control_set.controls,
                        )
                        try:
                            result_df = run_lm2011_quarterly_fama_macbeth(
                                spec_panel_lf,
                                table_id="lm2011_extension_results",
                                text_scope=text_scope,
                                dependent_variable=outcome_name,
                                signal_column=signal_column,
                                control_columns=control_columns,
                                specification_id=(
                                    f"{comparison_spec.specification_name}:"
                                    f"{control_set.control_set_id}:{outcome_name}"
                                ),
                                nw_lags=nw_lags,
                                quarter_weighting=quarter_weighting,
                            )
                        except ValueError as exc:
                            output_rows.append(
                                _extension_result_status_row(
                                    run_id=run_id,
                                    sample_window=EXTENSION_SAMPLE_WINDOW,
                                    text_scope=text_scope,
                                    outcome_name=outcome_name,
                                    comparison_spec=comparison_spec,
                                    control_set=control_set,
                                    estimator_status="failed",
                                    failure_reason=str(exc),
                                    nw_lags=nw_lags,
                                )
                            )
                            continue
                        if result_df.height == 0:
                            output_rows.append(
                                _extension_result_status_row(
                                    run_id=run_id,
                                    sample_window=EXTENSION_SAMPLE_WINDOW,
                                    text_scope=text_scope,
                                    outcome_name=outcome_name,
                                    comparison_spec=comparison_spec,
                                    control_set=control_set,
                                    estimator_status="insufficient_sample",
                                    failure_reason="no estimable quarterly Fama-MacBeth cross-sections",
                                    nw_lags=nw_lags,
                                )
                            )
                            continue
                        output_rows.extend(
                            _convert_lm2011_table_results_to_extension_rows(
                                result_df,
                                run_id=run_id,
                                sample_window=EXTENSION_SAMPLE_WINDOW,
                                outcome_name=outcome_name,
                                comparison_spec=comparison_spec,
                                control_set=control_set,
                            )
                        )
    if not output_rows:
        return _empty_extension_results_df()
    return pl.DataFrame(output_rows, schema_overrides=_empty_extension_results_df().schema).select(
        _empty_extension_results_df().columns
    )


def _run_quantile_fit_comparisons_for_variant(
    panel_lf: pl.LazyFrame,
    *,
    variant: FinbertQuantileVariant,
    suite_name: str,
    nw_lags: int = 1,
) -> dict[str, pl.DataFrame]:
    comparison_specs = _quantile_comparison_specs(variant)
    comparison_spec_by_name = {
        comparison_spec.specification_name: comparison_spec
        for comparison_spec in comparison_specs
    }
    comparison_pairs = _extension_fit_comparison_pairs(PRIMARY_SPECIFICATION_NAMES)
    summary_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    skipped_quarter_rows: list[dict[str, object]] = []

    for text_scope in PRIMARY_TEXT_SCOPES:
        for outcome_name in PRIMARY_OUTCOME_NAMES:
            for control_set_id in PRIMARY_CONTROL_SET_IDS:
                control_set = _control_set_by_id(control_set_id)
                run_id = f"{suite_name}:{variant.variant_id}:fit_weighted"
                try:
                    common_panel_lf = _build_quantile_common_comparison_panel_lf(
                        panel_lf,
                        variant=variant,
                        comparison_specs=comparison_specs,
                        control_set_id=control_set.control_set_id,
                        control_columns=control_set.controls,
                        text_scope=text_scope,
                        outcome_name=outcome_name,
                    )
                except ValueError as exc:
                    failure_reason = str(exc)
                    for comparison_spec in comparison_specs:
                        summary_rows.append(
                            _extension_fit_summary_status_row(
                                run_id=run_id,
                                sample_window=EXTENSION_SAMPLE_WINDOW,
                                text_scope=text_scope,
                                outcome_name=outcome_name,
                                comparison_spec=comparison_spec,
                                control_set=control_set,
                                estimator_status="failed",
                                failure_reason=failure_reason,
                            )
                        )
                    for comparison_name, left_name, right_name in comparison_pairs:
                        comparison_rows.append(
                            _extension_fit_comparison_status_row(
                                run_id=run_id,
                                sample_window=EXTENSION_SAMPLE_WINDOW,
                                text_scope=text_scope,
                                outcome_name=outcome_name,
                                control_set=control_set,
                                comparison_name=comparison_name,
                                left_spec=comparison_spec_by_name[left_name],
                                right_spec=comparison_spec_by_name[right_name],
                                estimator_status="failed",
                                failure_reason=failure_reason,
                            )
                        )
                    continue

                spec_quarter_fit_map: dict[str, pl.DataFrame] = {}
                spec_failure_reasons: dict[str, str] = {}
                for comparison_spec in comparison_specs:
                    signal_column, control_columns = _quantile_signal_and_controls(
                        comparison_spec,
                        control_set.controls,
                    )
                    try:
                        bundle = run_lm2011_quarterly_fama_macbeth_with_diagnostics(
                            common_panel_lf,
                            table_id="lm2011_extension_fit",
                            text_scope=text_scope,
                            dependent_variable=outcome_name,
                            signal_column=signal_column,
                            control_columns=control_columns,
                            specification_id=(
                                f"{comparison_spec.specification_name}:"
                                f"{control_set.control_set_id}:{outcome_name}"
                            ),
                            nw_lags=nw_lags,
                            signal_inputs=comparison_spec.signal_inputs,
                            on_rank_deficient="skip",
                        )
                    except ValueError as exc:
                        spec_quarter_fit_map[
                            comparison_spec.specification_name
                        ] = _empty_extension_fit_quarterly_df()
                        spec_failure_reasons[comparison_spec.specification_name] = str(exc)
                        continue

                    fit_rows = [
                        {
                            "run_id": run_id,
                            "sample_window": EXTENSION_SAMPLE_WINDOW,
                            "text_scope": text_scope,
                            "outcome_name": outcome_name,
                            "feature_family": comparison_spec.feature_family,
                            "control_set_id": control_set.control_set_id,
                            "control_set_alias": control_set.spec_alias,
                            "specification_name": comparison_spec.specification_name,
                            "signal_name": _comparison_signal_name(comparison_spec.signal_inputs),
                            "signal_inputs": list(comparison_spec.signal_inputs),
                            "quarter_start": row["quarter_start"],
                            "n_obs": row["n_obs"],
                            "industry_count": row["industry_count"],
                            "industry_dummy_count": row["industry_dummy_count"],
                            "visible_regressor_count": row["visible_regressor_count"],
                            "full_regressor_count": row["full_regressor_count"],
                            "rank": row["rank"],
                            "df_model": row["df_model"],
                            "df_resid": row["df_resid"],
                            "condition_number": row["condition_number"],
                            "raw_r2": row["raw_r2"],
                            "adj_r2": row["adj_r2"],
                            "ssr": row["ssr"],
                            "centered_tss": row["centered_tss"],
                            "weight": row["weight"],
                            "weighting_rule": row["weighting_rule"],
                            "common_row_sample_policy": _EXTENSION_COMMON_ROW_SAMPLE_POLICY,
                        }
                        for row in bundle.quarter_fit_df.to_dicts()
                    ]
                    spec_quarter_fit_df = (
                        pl.DataFrame(
                            fit_rows,
                            schema_overrides=_empty_extension_fit_quarterly_df().schema,
                        )
                        if fit_rows
                        else _empty_extension_fit_quarterly_df()
                    )
                    spec_quarter_fit_map[comparison_spec.specification_name] = spec_quarter_fit_df

                    skipped_quarter_rows.extend(
                        {
                            "run_id": run_id,
                            "sample_window": EXTENSION_SAMPLE_WINDOW,
                            "text_scope": text_scope,
                            "outcome_name": outcome_name,
                            "feature_family": comparison_spec.feature_family,
                            "control_set_id": control_set.control_set_id,
                            "control_set_alias": control_set.spec_alias,
                            "specification_name": comparison_spec.specification_name,
                            "signal_name": _comparison_signal_name(comparison_spec.signal_inputs),
                            "signal_inputs": list(comparison_spec.signal_inputs),
                            "quarter_start": row["quarter_start"],
                            "skip_reason": row["skip_reason"],
                            "n_obs": row["n_obs"],
                            "industry_count": row["industry_count"],
                            "rank": row["rank"],
                            "column_count": row["column_count"],
                            "condition_number": row["condition_number"],
                            "regressors": row["regressors"],
                            "duplicate_regressor_pairs": row["duplicate_regressor_pairs"],
                            "restoring_drop_candidates": row["restoring_drop_candidates"],
                        }
                        for row in bundle.skipped_quarters_df.to_dicts()
                    )

                common_quarters: list[object] = []
                if comparison_specs and all(
                    spec_quarter_fit_map[comparison_spec.specification_name].height > 0
                    for comparison_spec in comparison_specs
                ):
                    common_quarters = sorted(
                        set.intersection(
                            *[
                                set(
                                    spec_quarter_fit_map[
                                        comparison_spec.specification_name
                                    ].get_column("quarter_start").to_list()
                                )
                                for comparison_spec in comparison_specs
                            ]
                        )
                    )

                common_quarter_reason = "no common successful quarters across selected specifications"
                n_obs_mismatch_reason = None
                quarter_row_maps: dict[str, dict[object, dict[str, object]]] = {}
                if common_quarters:
                    quarter_row_maps = {
                        comparison_spec.specification_name: {
                            row["quarter_start"]: row
                            for row in spec_quarter_fit_map[
                                comparison_spec.specification_name
                            ].to_dicts()
                        }
                        for comparison_spec in comparison_specs
                    }
                    for quarter_start in common_quarters:
                        n_obs_values = {
                            int(
                                quarter_row_maps[comparison_spec.specification_name][
                                    quarter_start
                                ]["n_obs"]
                            )
                            for comparison_spec in comparison_specs
                        }
                        if len(n_obs_values) != 1:
                            n_obs_mismatch_reason = (
                                "common successful quarter n_obs mismatch across specifications"
                            )
                            break

                if n_obs_mismatch_reason is not None:
                    for comparison_spec in comparison_specs:
                        summary_rows.append(
                            _extension_fit_summary_status_row(
                                run_id=run_id,
                                sample_window=EXTENSION_SAMPLE_WINDOW,
                                text_scope=text_scope,
                                outcome_name=outcome_name,
                                comparison_spec=comparison_spec,
                                control_set=control_set,
                                estimator_status="failed",
                                failure_reason=n_obs_mismatch_reason,
                            )
                        )
                    for comparison_name, left_name, right_name in comparison_pairs:
                        comparison_rows.append(
                            _extension_fit_comparison_status_row(
                                run_id=run_id,
                                sample_window=EXTENSION_SAMPLE_WINDOW,
                                text_scope=text_scope,
                                outcome_name=outcome_name,
                                control_set=control_set,
                                comparison_name=comparison_name,
                                left_spec=comparison_spec_by_name[left_name],
                                right_spec=comparison_spec_by_name[right_name],
                                estimator_status="failed",
                                failure_reason=n_obs_mismatch_reason,
                            )
                        )
                    continue

                if not common_quarters:
                    for comparison_spec in comparison_specs:
                        failure_reason = spec_failure_reasons.get(
                            comparison_spec.specification_name,
                            common_quarter_reason,
                        )
                        estimator_status = (
                            "failed"
                            if comparison_spec.specification_name in spec_failure_reasons
                            else "insufficient_sample"
                        )
                        summary_rows.append(
                            _extension_fit_summary_status_row(
                                run_id=run_id,
                                sample_window=EXTENSION_SAMPLE_WINDOW,
                                text_scope=text_scope,
                                outcome_name=outcome_name,
                                comparison_spec=comparison_spec,
                                control_set=control_set,
                                estimator_status=estimator_status,
                                failure_reason=failure_reason,
                            )
                        )
                    pair_failure_reason = (
                        "; ".join(sorted(set(spec_failure_reasons.values())))
                        if spec_failure_reasons
                        else common_quarter_reason
                    )
                    pair_status = "failed" if spec_failure_reasons else "insufficient_sample"
                    for comparison_name, left_name, right_name in comparison_pairs:
                        comparison_rows.append(
                            _extension_fit_comparison_status_row(
                                run_id=run_id,
                                sample_window=EXTENSION_SAMPLE_WINDOW,
                                text_scope=text_scope,
                                outcome_name=outcome_name,
                                control_set=control_set,
                                comparison_name=comparison_name,
                                left_spec=comparison_spec_by_name[left_name],
                                right_spec=comparison_spec_by_name[right_name],
                                estimator_status=pair_status,
                                failure_reason=pair_failure_reason,
                            )
                        )
                    continue

                common_weights = [
                    float(
                        quarter_row_maps[comparison_specs[0].specification_name][
                            quarter_start
                        ]["n_obs"]
                    )
                    for quarter_start in common_quarters
                ]
                total_n_obs = int(sum(common_weights))
                mean_quarter_n = sum(common_weights) / float(len(common_weights))

                for comparison_spec in comparison_specs:
                    common_rows = [
                        quarter_row_maps[comparison_spec.specification_name][quarter_start]
                        for quarter_start in common_quarters
                    ]
                    raw_values = [float(row["raw_r2"]) for row in common_rows]
                    adj_values = [float(row["adj_r2"]) for row in common_rows]
                    summary_rows.append(
                        {
                            "run_id": run_id,
                            "sample_window": EXTENSION_SAMPLE_WINDOW,
                            "text_scope": text_scope,
                            "outcome_name": outcome_name,
                            "feature_family": comparison_spec.feature_family,
                            "control_set_id": control_set.control_set_id,
                            "control_set_alias": control_set.spec_alias,
                            "specification_name": comparison_spec.specification_name,
                            "signal_name": _comparison_signal_name(comparison_spec.signal_inputs),
                            "signal_inputs": list(comparison_spec.signal_inputs),
                            "n_quarters": len(common_quarters),
                            "total_n_obs": total_n_obs,
                            "mean_quarter_n": mean_quarter_n,
                            "weighted_avg_raw_r2": _weighted_mean(raw_values, common_weights),
                            "weighted_avg_adj_r2": _weighted_mean(adj_values, common_weights),
                            "equal_quarter_avg_raw_r2": sum(raw_values) / float(len(raw_values)),
                            "equal_quarter_avg_adj_r2": sum(adj_values) / float(len(adj_values)),
                            "weighting_rule": "quarter_observation_count",
                            "common_success_policy": _EXTENSION_COMMON_SUCCESS_POLICY,
                            "estimator_status": "estimated",
                            "failure_reason": None,
                        }
                    )

                for comparison_name, left_name, right_name in comparison_pairs:
                    left_spec = comparison_spec_by_name[left_name]
                    right_spec = comparison_spec_by_name[right_name]
                    delta_raw_values: list[float] = []
                    delta_adj_values: list[float] = []
                    for quarter_start in common_quarters:
                        left_row = quarter_row_maps[left_name][quarter_start]
                        right_row = quarter_row_maps[right_name][quarter_start]
                        delta_raw_values.append(float(left_row["raw_r2"]) - float(right_row["raw_r2"]))
                        delta_adj_values.append(float(left_row["adj_r2"]) - float(right_row["adj_r2"]))
                    nw_se = None
                    nw_t_stat = None
                    nw_p_value = None
                    if len(delta_adj_values) >= 3:
                        nw_se = _newey_west_standard_error(delta_adj_values, common_weights, nw_lags=nw_lags)
                        weighted_delta_adj = _weighted_mean(delta_adj_values, common_weights)
                        if weighted_delta_adj is not None and nw_se is not None and nw_se > 0:
                            nw_t_stat = weighted_delta_adj / nw_se
                            nw_p_value = _normal_approx_two_sided_p_value(nw_t_stat)
                    comparison_rows.append(
                        {
                            "run_id": run_id,
                            "sample_window": EXTENSION_SAMPLE_WINDOW,
                            "text_scope": text_scope,
                            "outcome_name": outcome_name,
                            "control_set_id": control_set.control_set_id,
                            "control_set_alias": control_set.spec_alias,
                            "comparison_name": comparison_name,
                            "left_specification_name": left_name,
                            "left_signal_name": _comparison_signal_name(left_spec.signal_inputs),
                            "left_signal_inputs": list(left_spec.signal_inputs),
                            "right_specification_name": right_name,
                            "right_signal_name": _comparison_signal_name(right_spec.signal_inputs),
                            "right_signal_inputs": list(right_spec.signal_inputs),
                            "n_quarters": len(common_quarters),
                            "total_n_obs": total_n_obs,
                            "mean_quarter_n": mean_quarter_n,
                            "weighted_avg_delta_raw_r2": _weighted_mean(delta_raw_values, common_weights),
                            "weighted_avg_delta_adj_r2": _weighted_mean(delta_adj_values, common_weights),
                            "equal_quarter_avg_delta_raw_r2": sum(delta_raw_values)
                            / float(len(delta_raw_values)),
                            "equal_quarter_avg_delta_adj_r2": sum(delta_adj_values)
                            / float(len(delta_adj_values)),
                            "nw_lags": nw_lags,
                            "nw_se_delta_adj_r2": nw_se,
                            "nw_t_stat_delta_adj_r2": nw_t_stat,
                            "nw_p_value_delta_adj_r2": nw_p_value,
                            "weighting_rule": "quarter_observation_count",
                            "common_success_policy": _EXTENSION_COMMON_SUCCESS_POLICY,
                            "estimator_status": "estimated",
                            "failure_reason": None,
                        }
                    )

    return {
        "fit_summary": (
            pl.DataFrame(summary_rows, schema_overrides=_empty_extension_fit_summary_df().schema).select(
                _empty_extension_fit_summary_df().columns
            )
            if summary_rows
            else _empty_extension_fit_summary_df()
        ),
        "fit_comparisons": (
            pl.DataFrame(
                comparison_rows,
                schema_overrides=_empty_extension_fit_comparisons_df().schema,
            ).select(_empty_extension_fit_comparisons_df().columns)
            if comparison_rows
            else _empty_extension_fit_comparisons_df()
        ),
        "fit_skipped_quarters": (
            pl.DataFrame(
                skipped_quarter_rows,
                schema_overrides=_empty_extension_fit_skipped_quarters_df().schema,
            ).select(_empty_extension_fit_skipped_quarters_df().columns)
            if skipped_quarter_rows
            else _empty_extension_fit_skipped_quarters_df()
        ),
    }


def _run_quantile_variant_suite(
    *,
    base_panel_df: pl.DataFrame,
    suite_name: str,
) -> dict[str, pl.DataFrame]:
    coefficient_frames: list[pl.DataFrame] = []
    fit_summary_frames: list[pl.DataFrame] = []
    fit_comparison_frames: list[pl.DataFrame] = []
    fit_skipped_frames: list[pl.DataFrame] = []
    panel_lf = base_panel_df.lazy()

    for variant in QUANTILE_VARIANTS:
        coefficient_frames.append(
            _annotate_variant_df(
                _run_quantile_coefficients_for_variant(
                    panel_lf,
                    variant=variant,
                    suite_name=suite_name,
                ),
                variant,
            )
        )
        fit_artifacts = _run_quantile_fit_comparisons_for_variant(
            panel_lf,
            variant=variant,
            suite_name=suite_name,
        )
        fit_summary_frames.append(_annotate_variant_df(fit_artifacts["fit_summary"], variant))
        fit_comparison_frames.append(_annotate_variant_df(fit_artifacts["fit_comparisons"], variant))
        fit_skipped_frames.append(_annotate_variant_df(fit_artifacts["fit_skipped_quarters"], variant))

    return {
        "coefficients": _collect_variant_frames(coefficient_frames),
        "fit_summary": _collect_variant_frames(fit_summary_frames),
        "fit_comparisons": _collect_variant_frames(fit_comparison_frames),
        "fit_skipped_quarters": _collect_variant_frames(fit_skipped_frames),
    }


def _annotate_variant_df(df: pl.DataFrame, variant: Any) -> pl.DataFrame:
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


def _collect_tail_doc_batch_ids(year_path: Path) -> pl.DataFrame:
    return (
        pl.scan_parquet(str(year_path))
        .select(pl.col("doc_id").cast(pl.Utf8, strict=False).alias("doc_id"))
        .drop_nulls(subset=["doc_id"])
        .unique(subset=["doc_id"], keep="first", maintain_order=False)
        .collect()
    )


def _tail_doc_surface_batch_paths(
    *,
    year_path: Path,
    year_doc_ids_df: pl.DataFrame,
    by_year_dir: Path,
    tail_doc_batch_size: int,
) -> list[Path]:
    batch_dir = by_year_dir / "_batches" / year_path.stem
    batch_dir.mkdir(parents=True, exist_ok=True)
    for stale_batch_path in batch_dir.glob("*.parquet"):
        stale_batch_path.unlink()

    batch_paths: list[Path] = []
    for batch_index, offset in enumerate(range(0, year_doc_ids_df.height, tail_doc_batch_size)):
        doc_batch_df = year_doc_ids_df.slice(offset, tail_doc_batch_size)
        if doc_batch_df.height <= 0:
            continue
        batch_df = build_finbert_tail_doc_surface_lf(
            pl.scan_parquet(str(year_path)).join(doc_batch_df.lazy(), on="doc_id", how="semi"),
            text_scopes=PRIMARY_TEXT_SCOPES,
        ).collect()
        _validate_unique_doc_scope(
            batch_df,
            label=f"tail_doc_surface[{year_path.stem}:batch_{batch_index:05d}]",
        )
        batch_path = batch_dir / f"batch_{batch_index:05d}.parquet"
        write_frame(
            batch_df if batch_df.height > 0 else pl.DataFrame(schema=TAIL_DOC_SURFACE_SCHEMA),
            batch_path,
        )
        batch_paths.append(batch_path)
    return batch_paths


def _build_tail_doc_surfaces(
    *,
    sentence_scores_dir: Path,
    output_dir: Path,
    tail_doc_batch_size: int,
) -> TailDocSurfaceArtifacts:
    if tail_doc_batch_size < 1:
        raise ValueError("tail_doc_batch_size must be >= 1.")
    by_year_dir = output_dir / TAIL_DOC_SURFACE_BY_YEAR_DIRNAME
    by_year_dir.mkdir(parents=True, exist_ok=True)
    stacked_path = output_dir / ARTIFACT_FILENAMES["tail_doc_surface"]
    written_year_paths: list[Path] = []
    for year_path in _resolve_sentence_score_year_paths(sentence_scores_dir):
        year_doc_ids_df = _collect_tail_doc_batch_ids(year_path)
        batch_paths = _tail_doc_surface_batch_paths(
            year_path=year_path,
            year_doc_ids_df=year_doc_ids_df,
            by_year_dir=by_year_dir,
            tail_doc_batch_size=tail_doc_batch_size,
        )
        year_lf = _scan_tail_doc_surface_paths(batch_paths)
        _validate_unique_doc_scope(year_lf, label=f"tail_doc_surface[{year_path.stem}]")
        output_path = by_year_dir / year_path.name
        if output_path.exists():
            output_path.unlink()
        year_lf.sink_parquet(output_path, compression=PARQUET_COMPRESSION)
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
        tail_doc_batch_size=cfg.tail_doc_batch_size,
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
    quantile_outputs = _run_quantile_variant_suite(
        base_panel_df=tail_panel_df,
        suite_name=f"{run_name}:quantile",
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
        quantile_coefficients_path=run_dir / ARTIFACT_FILENAMES["quantile_coefficients"],
        quantile_fit_summary_path=run_dir / ARTIFACT_FILENAMES["quantile_fit_summary"],
        quantile_fit_comparisons_path=run_dir / ARTIFACT_FILENAMES["quantile_fit_comparisons"],
        quantile_fit_skipped_quarters_path=run_dir / ARTIFACT_FILENAMES["quantile_fit_skipped_quarters"],
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
    write_frame(quantile_outputs["coefficients"], artifacts.quantile_coefficients_path)
    write_frame(quantile_outputs["fit_summary"], artifacts.quantile_fit_summary_path)
    write_frame(quantile_outputs["fit_comparisons"], artifacts.quantile_fit_comparisons_path)
    write_frame(quantile_outputs["fit_skipped_quarters"], artifacts.quantile_fit_skipped_quarters_path)
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
            "quantile_signal": [
                {
                    "variant_id": variant.variant_id,
                    "description": variant.description,
                    "source_columns": list(variant.source_columns),
                    "base_variant_id": variant.base_variant_id,
                    "base_signal_col": variant.base_signal_col,
                    "transform": variant.transform,
                    "quantile_k": variant.quantile_k,
                }
                for variant in QUANTILE_VARIANTS
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
            "tail_doc_batch_size": cfg.tail_doc_batch_size,
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
            "quantile_coefficients_path": _manifest_path_value(
                artifacts.quantile_coefficients_path,
                manifest_path=manifest_path,
            ),
            "quantile_fit_summary_path": _manifest_path_value(
                artifacts.quantile_fit_summary_path,
                manifest_path=manifest_path,
            ),
            "quantile_fit_comparisons_path": _manifest_path_value(
                artifacts.quantile_fit_comparisons_path,
                manifest_path=manifest_path,
            ),
            "quantile_fit_skipped_quarters_path": _manifest_path_value(
                artifacts.quantile_fit_skipped_quarters_path,
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
            "quantile_coefficients": quantile_outputs["coefficients"].height,
            "quantile_fit_summary": quantile_outputs["fit_summary"].height,
            "quantile_fit_comparisons": quantile_outputs["fit_comparisons"].height,
            "quantile_fit_skipped_quarters": quantile_outputs["fit_skipped_quarters"].height,
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
        tail_doc_batch_size=int(args.tail_doc_batch_size),
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

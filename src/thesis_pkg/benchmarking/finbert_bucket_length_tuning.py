from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import polars as pl

from thesis_pkg.benchmarking.contracts import BucketEdgeSpec
from thesis_pkg.benchmarking.contracts import BucketLengthSpec
from thesis_pkg.benchmarking.contracts import DEFAULT_BUCKET_EDGE_SPEC
from thesis_pkg.benchmarking.contracts import resolve_bucket_lengths_for_edges
from thesis_pkg.benchmarking.sentence_length_visualization import DOC_ID_COLUMN
from thesis_pkg.benchmarking.sentence_length_visualization import ITEM_CODE_COLUMN
from thesis_pkg.benchmarking.sentence_length_visualization import TOKEN_BUCKET_COLUMN
from thesis_pkg.benchmarking.sentence_length_visualization import TOKEN_COUNT_COLUMN
from thesis_pkg.benchmarking.sentence_length_visualization import YEAR_COLUMN
from thesis_pkg.benchmarking.sentence_length_visualization import normalize_sentence_dataset_dir
from thesis_pkg.benchmarking.sentence_length_visualization import sentence_dataset_paths


BUCKET_ORDER: tuple[str, ...] = ("short", "medium", "long")
RECOMMENDED_BUCKET_COLUMN = "recommended_finbert_token_bucket"
SUMMARY_QUANTILES: tuple[float, ...] = (0.90, 0.95, 0.99, 0.995, 0.999)
REQUIRED_COLUMNS: tuple[str, ...] = (
    DOC_ID_COLUMN,
    YEAR_COLUMN,
    ITEM_CODE_COLUMN,
    TOKEN_COUNT_COLUMN,
    TOKEN_BUCKET_COLUMN,
)


@dataclass(frozen=True)
class BucketEdgeRecommendation:
    summary_by_bucket: pl.DataFrame
    recommendation_summary: pl.DataFrame
    recommended_edges: BucketEdgeSpec
    effective_bucket_lengths: BucketLengthSpec
    env_overrides: dict[str, int]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class BucketEdgeRecommendationArtifacts:
    output_dir: Path
    summary_by_bucket_parquet_path: Path
    summary_by_bucket_csv_path: Path
    recommendation_summary_parquet_path: Path
    recommendation_summary_csv_path: Path
    metadata_path: Path
    env_overrides_path: Path


def _quantile_alias(quantile: float) -> str:
    if math.isclose(quantile, 0.90):
        return "token_p90"
    if math.isclose(quantile, 0.95):
        return "token_p95"
    if math.isclose(quantile, 0.99):
        return "token_p99"
    if math.isclose(quantile, 0.995):
        return "token_p995"
    if math.isclose(quantile, 0.999):
        return "token_p999"
    raise ValueError(f"Unsupported summary quantile: {quantile!r}")


def _normalize_item_codes(item_codes: tuple[str, ...] | None) -> tuple[str, ...] | None:
    if item_codes is None:
        return None
    normalized = tuple(
        dict.fromkeys(str(code).strip().lower() for code in item_codes if str(code).strip())
    )
    return normalized or None


def _normalize_years(years: tuple[int, ...] | None) -> tuple[int, ...] | None:
    if years is None:
        return None
    normalized = tuple(sorted({int(year) for year in years}))
    return normalized or None


def _required_schema(path: Path) -> None:
    schema = pl.scan_parquet(path).collect_schema()
    missing = [column for column in REQUIRED_COLUMNS if column not in schema]
    if missing:
        raise ValueError(f"Sentence parquet {path} is missing required columns: {missing}")


def _scan_sentence_tokens(
    sentence_dataset_dir: Path,
    *,
    item_codes: tuple[str, ...] | None,
    years: tuple[int, ...] | None,
) -> tuple[pl.LazyFrame, tuple[Path, ...], Path]:
    by_year_dir = normalize_sentence_dataset_dir(sentence_dataset_dir)
    parquet_paths = sentence_dataset_paths(by_year_dir)
    _required_schema(parquet_paths[0])
    lf = pl.scan_parquet([str(path) for path in parquet_paths]).select(REQUIRED_COLUMNS)
    if item_codes is not None:
        lf = lf.filter(pl.col(ITEM_CODE_COLUMN).is_in(item_codes))
    if years is not None:
        lf = lf.filter(pl.col(YEAR_COLUMN).is_in(years))
    return lf, parquet_paths, by_year_dir


def _current_bucket_upper_edge(bucket: str, current_edges: BucketEdgeSpec) -> int:
    if bucket == "short":
        return current_edges.short_edge
    if bucket == "medium":
        return current_edges.medium_edge
    if bucket == "long":
        return BucketLengthSpec().long_max_length
    raise ValueError(f"Unknown bucket: {bucket!r}")


def _bucket_current_length(bucket: str, current_lengths: BucketLengthSpec) -> int:
    if bucket == "short":
        return current_lengths.short_max_length
    if bucket == "medium":
        return current_lengths.medium_max_length
    if bucket == "long":
        return current_lengths.long_max_length
    raise ValueError(f"Unknown bucket: {bucket!r}")


def _recommended_bucket_upper_edge(bucket: str, recommended_edges: BucketEdgeSpec) -> int:
    if bucket == "short":
        return recommended_edges.short_edge
    if bucket == "medium":
        return recommended_edges.medium_edge
    if bucket == "long":
        return BucketLengthSpec().long_max_length
    raise ValueError(f"Unknown bucket: {bucket!r}")


def _recommended_bucket_length(bucket: str, bucket_lengths: BucketLengthSpec) -> int:
    if bucket == "short":
        return bucket_lengths.short_max_length
    if bucket == "medium":
        return bucket_lengths.medium_max_length
    if bucket == "long":
        return bucket_lengths.long_max_length
    raise ValueError(f"Unknown bucket: {bucket!r}")


def _round_up_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError("multiple must be a positive integer.")
    if value <= 0:
        return multiple
    return int(math.ceil(value / multiple) * multiple)


def _quantile_exprs(target_quantile: float) -> list[pl.Expr]:
    exprs = [
        pl.len().alias("sentence_rows"),
        pl.col(DOC_ID_COLUMN).n_unique().alias("doc_count"),
        pl.col(TOKEN_COUNT_COLUMN).min().alias("token_min"),
        pl.col(TOKEN_COUNT_COLUMN).median().alias("token_median"),
        pl.col(TOKEN_COUNT_COLUMN).max().alias("token_max"),
        pl.col(TOKEN_COUNT_COLUMN).quantile(target_quantile).alias("token_target_quantile"),
    ]
    for quantile in SUMMARY_QUANTILES:
        exprs.append(pl.col(TOKEN_COUNT_COLUMN).quantile(quantile).alias(_quantile_alias(quantile)))
    return exprs


def _empty_summary_row(
    bucket: str,
    *,
    current_edges: BucketEdgeSpec,
    current_lengths: BucketLengthSpec,
) -> dict[str, Any]:
    return {
        TOKEN_BUCKET_COLUMN: bucket,
        "sentence_rows": 0,
        "doc_count": 0,
        "token_min": None,
        "token_median": None,
        "token_max": None,
        "token_target_quantile": None,
        "token_p90": None,
        "token_p95": None,
        "token_p99": None,
        "token_p995": None,
        "token_p999": None,
        "current_edge_upper_bound": _current_bucket_upper_edge(bucket, current_edges),
        "current_max_length": _bucket_current_length(bucket, current_lengths),
    }


def _bucket_summary(
    token_lf: pl.LazyFrame,
    *,
    current_edges: BucketEdgeSpec,
    current_lengths: BucketLengthSpec,
    target_quantile: float,
) -> pl.DataFrame:
    grouped = token_lf.group_by(TOKEN_BUCKET_COLUMN).agg(_quantile_exprs(target_quantile)).collect()
    grouped_rows = {
        str(row[TOKEN_BUCKET_COLUMN]): row
        for row in grouped.to_dicts()
    }
    ordered_rows: list[dict[str, Any]] = []
    for bucket in BUCKET_ORDER:
        row = dict(
            grouped_rows.get(
                bucket,
                _empty_summary_row(
                    bucket,
                    current_edges=current_edges,
                    current_lengths=current_lengths,
                ),
            )
        )
        row["current_edge_upper_bound"] = _current_bucket_upper_edge(bucket, current_edges)
        row["current_max_length"] = _bucket_current_length(bucket, current_lengths)
        ordered_rows.append(row)
    return pl.DataFrame(ordered_rows)


def _recommend_edge(
    *,
    bucket: str,
    current_edge: int,
    lower_bound: int,
    token_target_quantile: float | int | None,
    round_to: int,
    safety_margin_tokens: int,
    policy: str,
) -> tuple[int, str]:
    if policy == "keep_current":
        return current_edge, "kept_current"
    if token_target_quantile is None:
        return current_edge, "no_rows"
    candidate = int(math.ceil(float(token_target_quantile))) + safety_margin_tokens
    candidate = _round_up_to_multiple(candidate, round_to)
    candidate = min(current_edge, candidate)
    candidate = max(lower_bound, candidate)
    if bucket == "medium" and candidate < lower_bound:
        candidate = lower_bound
    return candidate, "target_quantile"


def _rebucket_expr(recommended_edges: BucketEdgeSpec) -> pl.Expr:
    return (
        pl.when(pl.col(TOKEN_COUNT_COLUMN) <= recommended_edges.short_edge)
        .then(pl.lit("short"))
        .when(pl.col(TOKEN_COUNT_COLUMN) <= recommended_edges.medium_edge)
        .then(pl.lit("medium"))
        .otherwise(pl.lit("long"))
        .alias(RECOMMENDED_BUCKET_COLUMN)
    )


def _bucket_length_expr(column: str, bucket_lengths: BucketLengthSpec) -> pl.Expr:
    return (
        pl.when(pl.col(column) == "short")
        .then(pl.lit(bucket_lengths.short_max_length))
        .when(pl.col(column) == "medium")
        .then(pl.lit(bucket_lengths.medium_max_length))
        .otherwise(pl.lit(bucket_lengths.long_max_length))
    )


def recommend_conservative_bucket_edges(
    sentence_dataset_dir: Path,
    *,
    item_codes: tuple[str, ...] | None = None,
    years: tuple[int, ...] | None = None,
    current_edges: BucketEdgeSpec | None = None,
    current_lengths: BucketLengthSpec | None = None,
    target_quantile: float = 0.999,
    round_to: int = 8,
    safety_margin_tokens: int = 8,
    medium_edge_policy: str = "keep_current",
) -> BucketEdgeRecommendation:
    if not (0.0 < target_quantile <= 1.0):
        raise ValueError("target_quantile must satisfy 0 < target_quantile <= 1.")
    if round_to <= 0:
        raise ValueError("round_to must be a positive integer.")
    if safety_margin_tokens < 0:
        raise ValueError("safety_margin_tokens must be non-negative.")
    if medium_edge_policy not in {"keep_current", "target_quantile"}:
        raise ValueError("medium_edge_policy must be 'keep_current' or 'target_quantile'.")

    resolved_current_edges = current_edges or DEFAULT_BUCKET_EDGE_SPEC
    resolved_current_lengths = current_lengths or BucketLengthSpec()
    normalized_item_codes = _normalize_item_codes(item_codes)
    normalized_years = _normalize_years(years)

    token_lf, parquet_paths, by_year_dir = _scan_sentence_tokens(
        sentence_dataset_dir,
        item_codes=normalized_item_codes,
        years=normalized_years,
    )
    summary_by_bucket = _bucket_summary(
        token_lf,
        current_edges=resolved_current_edges,
        current_lengths=resolved_current_lengths,
        target_quantile=target_quantile,
    )
    if not summary_by_bucket.height or int(summary_by_bucket["sentence_rows"].sum()) <= 0:
        raise ValueError("No sentence rows matched the requested filters.")

    summary_rows = {
        str(row[TOKEN_BUCKET_COLUMN]): row
        for row in summary_by_bucket.to_dicts()
    }
    short_edge, short_policy = _recommend_edge(
        bucket="short",
        current_edge=resolved_current_edges.short_edge,
        lower_bound=round_to,
        token_target_quantile=summary_rows["short"]["token_target_quantile"],
        round_to=round_to,
        safety_margin_tokens=safety_margin_tokens,
        policy="target_quantile",
    )
    medium_edge, medium_policy = _recommend_edge(
        bucket="medium",
        current_edge=resolved_current_edges.medium_edge,
        lower_bound=short_edge,
        token_target_quantile=summary_rows["medium"]["token_target_quantile"],
        round_to=round_to,
        safety_margin_tokens=safety_margin_tokens,
        policy=medium_edge_policy,
    )
    recommended_edges = BucketEdgeSpec(short_edge=short_edge, medium_edge=medium_edge)
    effective_bucket_lengths = resolve_bucket_lengths_for_edges(
        bucket_edges=recommended_edges,
        short_max_length=None,
        medium_max_length=None,
        long_max_length=None,
    )

    current_length_expr = _bucket_length_expr(TOKEN_BUCKET_COLUMN, resolved_current_lengths)
    recommended_length_expr = _bucket_length_expr(RECOMMENDED_BUCKET_COLUMN, effective_bucket_lengths)
    rebucketed_df = (
        token_lf.with_columns(
            [
                _rebucket_expr(recommended_edges),
                current_length_expr.alias("current_effective_max_length"),
            ]
        )
        .with_columns(recommended_length_expr.alias("recommended_effective_max_length"))
        .collect()
    )

    rebucket_counts = {
        str(row[RECOMMENDED_BUCKET_COLUMN]): int(row["sentence_rows"])
        for row in rebucketed_df.group_by(RECOMMENDED_BUCKET_COLUMN)
        .agg(pl.len().alias("sentence_rows"))
        .to_dicts()
    }
    total_sentence_rows = int(rebucketed_df.height)
    total_doc_count = int(rebucketed_df[DOC_ID_COLUMN].n_unique())
    current_estimated_padded_tokens_total = int(rebucketed_df["current_effective_max_length"].sum())
    recommended_estimated_padded_tokens_total = int(
        rebucketed_df["recommended_effective_max_length"].sum()
    )

    recommendation_rows: list[dict[str, Any]] = []
    policy_by_bucket = {
        "short": short_policy,
        "medium": medium_policy,
        "long": "fixed_512",
    }
    for bucket in BUCKET_ORDER:
        current_sentence_rows = int(summary_rows[bucket]["sentence_rows"])
        rebucketed_sentence_rows = int(rebucket_counts.get(bucket, 0))
        current_max_length = _bucket_current_length(bucket, resolved_current_lengths)
        effective_max_length = _recommended_bucket_length(bucket, effective_bucket_lengths)
        recommendation_rows.append(
            {
                TOKEN_BUCKET_COLUMN: bucket,
                "current_edge_upper_bound": _current_bucket_upper_edge(bucket, resolved_current_edges),
                "recommended_edge_upper_bound": _recommended_bucket_upper_edge(bucket, recommended_edges),
                "current_sentence_rows": current_sentence_rows,
                "rebucketed_sentence_rows": rebucketed_sentence_rows,
                "current_sentence_share": float(current_sentence_rows / total_sentence_rows),
                "rebucketed_sentence_share": float(rebucketed_sentence_rows / total_sentence_rows),
                "current_max_length": current_max_length,
                "effective_max_length": effective_max_length,
                "estimated_padded_tokens_current": current_sentence_rows * current_max_length,
                "estimated_padded_tokens_rebucketed": rebucketed_sentence_rows * effective_max_length,
                "estimated_padded_tokens_delta": (
                    rebucketed_sentence_rows * effective_max_length
                    - current_sentence_rows * current_max_length
                ),
                "target_quantile": (
                    target_quantile if bucket in {"short", "medium"} else None
                ),
                "target_quantile_token_count": summary_rows[bucket]["token_target_quantile"],
                "safety_margin_tokens": safety_margin_tokens if bucket in {"short", "medium"} else None,
                "round_to_multiple": round_to if bucket in {"short", "medium"} else None,
                "policy_applied": policy_by_bucket[bucket],
            }
        )

    env_overrides = {
        "SEC_CCM_FINBERT_SHORT_EDGE": recommended_edges.short_edge,
        "SEC_CCM_FINBERT_MEDIUM_EDGE": recommended_edges.medium_edge,
    }
    recommendation_summary = pl.DataFrame(recommendation_rows)
    metadata: dict[str, Any] = {
        "sentence_dataset_dir": str(by_year_dir),
        "parquet_file_count": len(parquet_paths),
        "parquet_files": [str(path) for path in parquet_paths],
        "filters": {
            "item_codes": list(normalized_item_codes) if normalized_item_codes is not None else None,
            "years": list(normalized_years) if normalized_years is not None else None,
        },
        "target_quantile": target_quantile,
        "round_to": round_to,
        "safety_margin_tokens": safety_margin_tokens,
        "medium_edge_policy": medium_edge_policy,
        "current_edges": asdict(resolved_current_edges),
        "recommended_edges": asdict(recommended_edges),
        "effective_bucket_lengths": asdict(effective_bucket_lengths),
        "env_overrides": env_overrides,
        "rebucketing_note": (
            "These recommendations target FinBERT sentence-bucket edge overrides for a fresh "
            "sentence preprocessing run. With no explicit SEC_CCM_FINBERT_*_MAX_LENGTH overrides, "
            "effective short and medium max_length values auto-match the recommended edges while "
            "the long bucket remains at 512."
        ),
        "adds_extra_truncation_beyond_512": False,
        "total_sentence_rows": total_sentence_rows,
        "total_doc_count": total_doc_count,
        "estimated_padded_tokens_current": current_estimated_padded_tokens_total,
        "estimated_padded_tokens_rebucketed": recommended_estimated_padded_tokens_total,
        "estimated_padded_tokens_delta": (
            recommended_estimated_padded_tokens_total - current_estimated_padded_tokens_total
        ),
        "estimated_padded_tokens_ratio": (
            float(recommended_estimated_padded_tokens_total / current_estimated_padded_tokens_total)
            if current_estimated_padded_tokens_total > 0
            else None
        ),
    }
    return BucketEdgeRecommendation(
        summary_by_bucket=summary_by_bucket,
        recommendation_summary=recommendation_summary,
        recommended_edges=recommended_edges,
        effective_bucket_lengths=effective_bucket_lengths,
        env_overrides=env_overrides,
        metadata=metadata,
    )


def write_bucket_edge_recommendation_report(
    recommendation: BucketEdgeRecommendation,
    output_dir: Path,
) -> BucketEdgeRecommendationArtifacts:
    resolved_output_dir = output_dir.resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    summary_parquet_path = resolved_output_dir / "bucket_edge_summary.parquet"
    summary_csv_path = resolved_output_dir / "bucket_edge_summary.csv"
    recommendation_parquet_path = resolved_output_dir / "bucket_edge_recommendations.parquet"
    recommendation_csv_path = resolved_output_dir / "bucket_edge_recommendations.csv"
    metadata_path = resolved_output_dir / "bucket_edge_recommendation.json"
    env_overrides_path = resolved_output_dir / "finbert_bucket_edge_overrides.env"

    recommendation.summary_by_bucket.write_parquet(summary_parquet_path, compression="zstd")
    recommendation.summary_by_bucket.write_csv(summary_csv_path)
    recommendation.recommendation_summary.write_parquet(
        recommendation_parquet_path,
        compression="zstd",
    )
    recommendation.recommendation_summary.write_csv(recommendation_csv_path)
    metadata_path.write_text(
        json.dumps(recommendation.metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    env_overrides_path.write_text(
        "\n".join(f"{key}={value}" for key, value in recommendation.env_overrides.items()) + "\n",
        encoding="utf-8",
    )
    return BucketEdgeRecommendationArtifacts(
        output_dir=resolved_output_dir,
        summary_by_bucket_parquet_path=summary_parquet_path,
        summary_by_bucket_csv_path=summary_csv_path,
        recommendation_summary_parquet_path=recommendation_parquet_path,
        recommendation_summary_csv_path=recommendation_csv_path,
        metadata_path=metadata_path,
        env_overrides_path=env_overrides_path,
    )

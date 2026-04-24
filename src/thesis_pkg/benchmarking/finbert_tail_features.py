from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from thesis_pkg.pipelines.lm2011_extension import normalize_lm2011_extension_text_scope_expr


TAIL_FEATURE_COLUMNS: tuple[str, ...] = (
    "tail_exposure_tau_0_60",
    "tail_exposure_tau_0_70",
    "tail_exposure_tau_0_80",
    "tail_share_tau_0_70",
    "top_10pct_neg_mean",
    "top_20pct_neg_mean",
    "neg_prob_dispersion",
)
TAIL_DOC_SURFACE_SCHEMA: dict[str, pl.DataType] = {
    "doc_id": pl.Utf8,
    "filing_date": pl.Date,
    "text_scope": pl.Utf8,
    "cleaning_policy_id": pl.Utf8,
    "model_name": pl.Utf8,
    "model_version": pl.Utf8,
    "segment_policy_id": pl.Utf8,
    "tail_exposure_tau_0_60": pl.Float64,
    "tail_exposure_tau_0_70": pl.Float64,
    "tail_exposure_tau_0_80": pl.Float64,
    "tail_share_tau_0_70": pl.Float64,
    "top_10pct_neg_mean": pl.Float64,
    "top_20pct_neg_mean": pl.Float64,
    "neg_prob_dispersion": pl.Float64,
}
_TAIL_METADATA_COLUMNS: tuple[str, ...] = (
    "doc_id",
    "filing_date",
    "text_scope",
    "cleaning_policy_id",
    "model_name",
    "model_version",
    "segment_policy_id",
)
_TAIL_TOKEN_WEIGHT_COLUMN = "finbert_token_count_512"


def _empty_tail_doc_surface() -> pl.DataFrame:
    return pl.DataFrame(schema=TAIL_DOC_SURFACE_SCHEMA)


def _require_columns(lf: pl.LazyFrame, columns: Sequence[str], *, label: str) -> None:
    schema = lf.collect_schema()
    missing = [column for column in columns if column not in schema]
    if missing:
        raise ValueError(f"{label} missing required columns: {', '.join(missing)}")


def _resolve_tail_text_scope_expr(schema: pl.Schema) -> pl.Expr:
    if "text_scope" in schema:
        return normalize_lm2011_extension_text_scope_expr(pl.col("text_scope"))
    if "benchmark_item_code" in schema:
        return normalize_lm2011_extension_text_scope_expr(pl.col("benchmark_item_code"))
    if "item_id" in schema:
        return normalize_lm2011_extension_text_scope_expr(pl.col("item_id"))
    raise ValueError(
        "sentence_scores must contain text_scope, benchmark_item_code, or item_id for tail aggregation."
    )


def _optional_utf8_expr(schema: pl.Schema, column: str) -> pl.Expr:
    if column in schema:
        return pl.col(column).cast(pl.Utf8, strict=False)
    return pl.lit(None, dtype=pl.Utf8)


def _optional_int64_expr(schema: pl.Schema, column: str) -> pl.Expr:
    if column in schema:
        return pl.col(column).cast(pl.Int64, strict=False)
    return pl.lit(None, dtype=pl.Int64)


def _normalize_text_scope_value(value: str) -> str:
    raw = value.strip().casefold().replace("-", "_")
    if raw in {"7", "item_7", "mda_item_7", "item_7_mda"}:
        return "item_7_mda"
    if raw in {"1a", "item_1a", "item_1a_risk_factors"}:
        return "item_1a_risk_factors"
    if raw in {"1", "item_1", "item_1_business"}:
        return "item_1_business"
    if raw in {"items_1_1a_7_concat", "item_1_item_1a_item_7_concat"}:
        return "items_1_1a_7_concat"
    return raw


def _top_fraction_cutoff(sentence_count_expr: pl.Expr, fraction: float) -> pl.Expr:
    scaled = (sentence_count_expr.cast(pl.Float64) * pl.lit(fraction)).ceil().cast(pl.Int64)
    return (
        pl.when(sentence_count_expr <= 1)
        .then(pl.lit(1, dtype=pl.Int64))
        .otherwise(scaled)
    )


def _weighted_group_mean(
    value_expr: pl.Expr,
    *,
    weight_expr: pl.Expr,
    denominator_expr: pl.Expr,
    group_keys: Sequence[str],
    alias: str,
) -> pl.Expr:
    numerator = (value_expr * weight_expr).sum().over(group_keys)
    return (
        pl.when(denominator_expr > 0.0)
        .then(numerator / denominator_expr)
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias(alias)
    )


def build_finbert_tail_doc_surface_lf(
    sentence_scores_lf: pl.LazyFrame,
    *,
    text_scopes: Sequence[str],
) -> pl.LazyFrame:
    _require_columns(
        sentence_scores_lf,
        ("doc_id", "negative_prob", "sentence_index", _TAIL_TOKEN_WEIGHT_COLUMN),
        label="sentence_scores",
    )
    schema = sentence_scores_lf.collect_schema()
    normalized_scopes = list(
        dict.fromkeys(_normalize_text_scope_value(str(scope)) for scope in text_scopes)
    )
    if not normalized_scopes:
        raise ValueError("text_scopes must be non-empty for FinBERT tail aggregation.")

    group_keys = list(_TAIL_METADATA_COLUMNS)
    sort_columns = [
        *group_keys,
        "negative_prob",
        "_sentence_index",
        "_benchmark_sentence_id",
    ]
    descending = [False] * len(group_keys) + [True, False, False]
    selected = (
        sentence_scores_lf.select(
            pl.col("doc_id").cast(pl.Utf8, strict=False),
            (
                pl.col("filing_date").cast(pl.Date, strict=False)
                if "filing_date" in schema
                else pl.lit(None, dtype=pl.Date)
            ).alias("filing_date"),
            _resolve_tail_text_scope_expr(schema).alias("text_scope"),
            _optional_utf8_expr(schema, "cleaning_policy_id").alias("cleaning_policy_id"),
            _optional_utf8_expr(schema, "model_name").alias("model_name"),
            _optional_utf8_expr(schema, "model_version").alias("model_version"),
            _optional_utf8_expr(schema, "segment_policy_id").alias("segment_policy_id"),
            pl.col("negative_prob").cast(pl.Float64, strict=False).alias("negative_prob"),
            _optional_int64_expr(schema, "sentence_index").alias("_sentence_index"),
            _optional_utf8_expr(schema, "benchmark_sentence_id").alias("_benchmark_sentence_id"),
            pl.when(pl.col(_TAIL_TOKEN_WEIGHT_COLUMN).cast(pl.Float64, strict=False) > 0.0)
            .then(pl.col(_TAIL_TOKEN_WEIGHT_COLUMN).cast(pl.Float64, strict=False))
            .otherwise(pl.lit(0.0))
            .alias("_token_weight"),
        )
        .filter(pl.col("text_scope").is_in(normalized_scopes))
        .sort(sort_columns, descending=descending, nulls_last=True)
    )
    denominator_expr = pl.col("_token_weight").sum().over(group_keys)
    sentence_count_expr = pl.len().over(group_keys).cast(pl.Int64)
    negative_rank_expr = pl.int_range(1, pl.len() + 1).over(group_keys).cast(pl.Int64)
    with_group_stats = (
        selected.with_columns(
            sentence_count_expr.alias("_sentence_count"),
            negative_rank_expr.alias("_negative_rank"),
        )
        .with_columns(
            _weighted_group_mean(
                pl.col("negative_prob"),
                weight_expr=pl.col("_token_weight"),
                denominator_expr=denominator_expr,
                group_keys=group_keys,
                alias="_negative_prob_weighted_mean",
            )
        )
        .with_columns(
            _weighted_group_mean(
                pl.when(pl.col("negative_prob") >= 0.60)
                .then(pl.col("negative_prob"))
                .otherwise(pl.lit(0.0)),
                weight_expr=pl.col("_token_weight"),
                denominator_expr=denominator_expr,
                group_keys=group_keys,
                alias="tail_exposure_tau_0_60",
            ),
            _weighted_group_mean(
                pl.when(pl.col("negative_prob") >= 0.70)
                .then(pl.col("negative_prob"))
                .otherwise(pl.lit(0.0)),
                weight_expr=pl.col("_token_weight"),
                denominator_expr=denominator_expr,
                group_keys=group_keys,
                alias="tail_exposure_tau_0_70",
            ),
            _weighted_group_mean(
                pl.when(pl.col("negative_prob") >= 0.80)
                .then(pl.col("negative_prob"))
                .otherwise(pl.lit(0.0)),
                weight_expr=pl.col("_token_weight"),
                denominator_expr=denominator_expr,
                group_keys=group_keys,
                alias="tail_exposure_tau_0_80",
            ),
            _weighted_group_mean(
                (pl.col("negative_prob") >= 0.70).cast(pl.Float64),
                weight_expr=pl.col("_token_weight"),
                denominator_expr=denominator_expr,
                group_keys=group_keys,
                alias="tail_share_tau_0_70",
            ),
            _weighted_group_mean(
                pl.when(pl.col("_negative_rank") <= _top_fraction_cutoff(pl.col("_sentence_count"), 0.10))
                .then(pl.col("negative_prob"))
                .otherwise(pl.lit(0.0)),
                weight_expr=pl.when(
                    pl.col("_negative_rank") <= _top_fraction_cutoff(pl.col("_sentence_count"), 0.10)
                )
                .then(pl.col("_token_weight"))
                .otherwise(pl.lit(0.0)),
                denominator_expr=(
                    pl.when(
                        pl.col("_negative_rank") <= _top_fraction_cutoff(pl.col("_sentence_count"), 0.10)
                    )
                    .then(pl.col("_token_weight"))
                    .otherwise(pl.lit(0.0))
                    .sum()
                    .over(group_keys)
                ),
                group_keys=group_keys,
                alias="top_10pct_neg_mean",
            ),
            _weighted_group_mean(
                pl.when(pl.col("_negative_rank") <= _top_fraction_cutoff(pl.col("_sentence_count"), 0.20))
                .then(pl.col("negative_prob"))
                .otherwise(pl.lit(0.0)),
                weight_expr=pl.when(
                    pl.col("_negative_rank") <= _top_fraction_cutoff(pl.col("_sentence_count"), 0.20)
                )
                .then(pl.col("_token_weight"))
                .otherwise(pl.lit(0.0)),
                denominator_expr=(
                    pl.when(
                        pl.col("_negative_rank") <= _top_fraction_cutoff(pl.col("_sentence_count"), 0.20)
                    )
                    .then(pl.col("_token_weight"))
                    .otherwise(pl.lit(0.0))
                    .sum()
                    .over(group_keys)
                ),
                group_keys=group_keys,
                alias="top_20pct_neg_mean",
            ),
            (
                pl.when(denominator_expr > 0.0)
                .then(
                    (
                        (
                            (pl.col("negative_prob") - pl.col("_negative_prob_weighted_mean")) ** 2
                            * pl.col("_token_weight")
                        )
                        .sum()
                        .over(group_keys)
                        / denominator_expr
                    )
                    .sqrt()
                )
                .otherwise(pl.lit(None, dtype=pl.Float64))
                .alias("neg_prob_dispersion")
            ),
        )
    )
    return (
        with_group_stats.select(*_TAIL_METADATA_COLUMNS, *TAIL_FEATURE_COLUMNS)
        .unique(subset=list(_TAIL_METADATA_COLUMNS), keep="first", maintain_order=True)
        .select(_empty_tail_doc_surface().columns)
    )


def build_finbert_tail_doc_surface(
    sentence_scores_df: pl.DataFrame,
    *,
    text_scopes: Sequence[str],
) -> pl.DataFrame:
    if sentence_scores_df.is_empty():
        return _empty_tail_doc_surface()
    return build_finbert_tail_doc_surface_lf(
        sentence_scores_df.lazy(),
        text_scopes=text_scopes,
    ).collect()


__all__ = [
    "TAIL_DOC_SURFACE_SCHEMA",
    "TAIL_FEATURE_COLUMNS",
    "build_finbert_tail_doc_surface",
    "build_finbert_tail_doc_surface_lf",
]

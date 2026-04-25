from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import datetime as dt
import json
import math
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

from thesis_pkg.core.ccm.lm2011 import attach_lm2011_industry_classifications
from thesis_pkg.pipelines.lm2011_pipeline import (
    _apply_lm2011_regression_transforms,
    _fit_checked_ols,
    _require_columns,
    build_lm2011_trading_strategy_ff4_summary,
    build_lm2011_trading_strategy_monthly_returns,
)


_RETURN_CONTROL_COLUMNS: tuple[str, ...] = (
    "log_size",
    "log_book_to_market",
    "pre_ffalpha",
    "log_share_turnover",
    "nasdaq_dummy",
    "institutional_ownership",
)
_SUE_CONTROL_COLUMNS: tuple[str, ...] = (
    *_RETURN_CONTROL_COLUMNS,
    "analyst_dispersion",
    "analyst_revisions",
)
_RETURN_CONTROL_COLUMNS_NO_OWNERSHIP: tuple[str, ...] = tuple(
    column for column in _RETURN_CONTROL_COLUMNS if column != "institutional_ownership"
)
_SUE_CONTROL_COLUMNS_NO_OWNERSHIP: tuple[str, ...] = tuple(
    column for column in _SUE_CONTROL_COLUMNS if column != "institutional_ownership"
)
_TABLE_VI_DEPENDENT_VARIABLES: tuple[str, ...] = (
    "filing_period_excess_return",
    "abnormal_volume",
    "postevent_return_volatility",
)
_TABLE_VI_SIGNAL_COLUMNS: tuple[str, ...] = (
    "h4n_inf_prop",
    "lm_negative_prop",
    "lm_positive_prop",
    "lm_uncertainty_prop",
    "lm_litigious_prop",
    "lm_modal_strong_prop",
    "lm_modal_weak_prop",
    "h4n_inf_tfidf",
    "lm_negative_tfidf",
    "lm_positive_tfidf",
    "lm_uncertainty_tfidf",
    "lm_litigious_tfidf",
    "lm_modal_strong_tfidf",
    "lm_modal_weak_tfidf",
)
_TABLE_VI_LEGACY_OWNERSHIP_DEPENDENT_VARIABLES: tuple[str, ...] = ("filing_period_excess_return",)
_TABLE_VI_LEGACY_OWNERSHIP_SIGNAL_COLUMNS: tuple[str, ...] = tuple(
    column for column in _TABLE_VI_SIGNAL_COLUMNS if not column.startswith("h4n_inf_")
)
_TABLE_RESULT_SCHEMA: dict[str, pl.DataType] = {
    "table_id": pl.Utf8,
    "specification_id": pl.Utf8,
    "text_scope": pl.Utf8,
    "signal_name": pl.Utf8,
    "dependent_variable": pl.Utf8,
    "coefficient_name": pl.Utf8,
    "estimate": pl.Float64,
    "standard_error": pl.Float64,
    "t_stat": pl.Float64,
    "n_quarters": pl.Int32,
    "mean_quarter_n": pl.Float64,
    "weighting_rule": pl.Utf8,
    "nw_lags": pl.Int32,
}
_SKIPPED_QUARTER_SCHEMA: dict[str, pl.DataType] = {
    "table_id": pl.Utf8,
    "specification_id": pl.Utf8,
    "text_scope": pl.Utf8,
    "dependent_variable": pl.Utf8,
    "signal_name": pl.Utf8,
    "quarter_start": pl.Date,
    "skip_reason": pl.Utf8,
    "n_obs": pl.Int32,
    "industry_count": pl.Int32,
    "rank": pl.Int32,
    "column_count": pl.Int32,
    "condition_number": pl.Float64,
    "regressors": pl.Utf8,
    "duplicate_regressor_pairs": pl.Utf8,
    "restoring_drop_candidates": pl.Utf8,
}
_QUARTER_FIT_SCHEMA: dict[str, pl.DataType] = {
    "table_id": pl.Utf8,
    "specification_id": pl.Utf8,
    "text_scope": pl.Utf8,
    "signal_name": pl.Utf8,
    "signal_inputs": pl.List(pl.Utf8),
    "dependent_variable": pl.Utf8,
    "quarter_start": pl.Date,
    "n_obs": pl.Int32,
    "industry_count": pl.Int32,
    "industry_dummy_count": pl.Int32,
    "visible_regressor_count": pl.Int32,
    "full_regressor_count": pl.Int32,
    "rank": pl.Int32,
    "df_model": pl.Float64,
    "df_resid": pl.Float64,
    "condition_number": pl.Float64,
    "raw_r2": pl.Float64,
    "adj_r2": pl.Float64,
    "ssr": pl.Float64,
    "centered_tss": pl.Float64,
    "weight": pl.Float64,
    "weighting_rule": pl.Utf8,
}
_QUARTER_WEIGHTING_RULE_OBSERVATION_COUNT = "quarter_observation_count"
_QUARTER_WEIGHTING_RULE_EQUAL = "equal_quarter"
_INTERCEPT_NAME = "intercept"
_RANK_DEFICIENT_SKIP_REASON = "rank_deficient_design"
_INSUFFICIENT_DOF_SKIP_REASON = "insufficient_degrees_of_freedom"
_NONFINITE_FIT_SKIP_REASON = "non_finite_fit_statistics"
QuarterWeighting = Literal["quarter_observation_count", "equal_quarter"]


def _empty_lm2011_table_results_df() -> pl.DataFrame:
    return pl.DataFrame(schema=_TABLE_RESULT_SCHEMA)


def _empty_skipped_quarters_df() -> pl.DataFrame:
    return pl.DataFrame(schema=_SKIPPED_QUARTER_SCHEMA)


def _empty_quarter_fit_df() -> pl.DataFrame:
    return pl.DataFrame(schema=_QUARTER_FIT_SCHEMA)


@dataclass(frozen=True)
class _QuarterlyFamaMacbethBundle:
    results_df: pl.DataFrame
    skipped_quarters_df: pl.DataFrame
    quarter_fit_df: pl.DataFrame = field(default_factory=_empty_quarter_fit_df)


@dataclass(frozen=True)
class _CrossSectionalOlsOutcome:
    coefficients: dict[str, float] | None
    n_obs: int | None
    quarter_fit_row: dict[str, object] | None = None
    skipped_quarter_row: dict[str, object] | None = None


def _quarter_start(value: dt.date) -> dt.date:
    month = ((value.month - 1) // 3) * 3 + 1
    return dt.date(value.year, month, 1)


def _unique_preserving_order(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)


def _select_text_signal_surface(text_features_lf: pl.LazyFrame, *, label: str) -> pl.LazyFrame:
    _require_columns(text_features_lf, ("doc_id",), label)
    schema = text_features_lf.collect_schema()
    signal_columns = [
        name
        for name in schema.names()
        if name != "doc_id"
        and (
            name.startswith("token_count_")
            or name.startswith("total_token_count_")
            or name.endswith("_prop")
            or name.endswith("_tfidf")
        )
    ]
    if not signal_columns:
        raise ValueError(f"{label} missing LM2011 text signal columns")
    return text_features_lf.select(
        pl.col("doc_id").cast(pl.Utf8, strict=False),
        *[pl.col(column) for column in signal_columns],
    ).unique(subset=["doc_id"], keep="first")


def _attach_ff48_industries(
    panel_lf: pl.LazyFrame,
    company_history_lf: pl.LazyFrame,
    company_description_lf: pl.LazyFrame,
    *,
    ff48_siccodes_path: Path | str,
) -> pl.LazyFrame:
    _require_columns(panel_lf, ("doc_id", "gvkey_int", "filing_date"), "panel")
    return attach_lm2011_industry_classifications(
        panel_lf.select(
            pl.col("doc_id").cast(pl.Utf8, strict=False),
            pl.col("gvkey_int").cast(pl.Int32, strict=False),
            pl.col("filing_date").cast(pl.Date, strict=False),
        ),
        company_history_lf,
        company_description_lf,
        ff48_siccodes_path=ff48_siccodes_path,
        filing_gvkey_col="gvkey_int",
    ).select(
        "doc_id",
        "HSIC",
        "SIC_desc",
        "SIC_final",
        "ff48_industry_id",
        "ff48_industry_short",
        "ff48_industry_name",
    )


def _nullify_infinite_float_columns(panel_lf: pl.LazyFrame) -> pl.LazyFrame:
    # Regression outputs should not carry non-finite float values into exported panels.
    schema = panel_lf.collect_schema()
    float_columns = [
        (name, dtype)
        for name, dtype in schema.items()
        if dtype in (pl.Float32, pl.Float64)
    ]
    if not float_columns:
        return panel_lf
    return panel_lf.with_columns(
        [
            pl.when(pl.col(name).is_infinite().fill_null(False))
            .then(pl.lit(None, dtype=dtype))
            .otherwise(pl.col(name))
            .alias(name)
            for name, dtype in float_columns
        ]
    )


def build_lm2011_return_regression_panel(
    event_panel_lf: pl.LazyFrame,
    text_features_lf: pl.LazyFrame,
    company_history_lf: pl.LazyFrame,
    company_description_lf: pl.LazyFrame,
    *,
    ff48_siccodes_path: Path | str,
    text_scope: str,
) -> pl.LazyFrame:
    _require_columns(
        event_panel_lf,
        (
            "doc_id",
            "gvkey_int",
            "filing_date",
            "size_event",
            "bm_event",
            "share_turnover",
            "pre_ffalpha",
            "institutional_ownership",
            "nasdaq_dummy",
            "filing_period_excess_return",
        ),
        "event_panel",
    )
    text_signal_lf = _select_text_signal_surface(text_features_lf, label="text_features")
    industries_lf = _attach_ff48_industries(
        event_panel_lf,
        company_history_lf,
        company_description_lf,
        ff48_siccodes_path=ff48_siccodes_path,
    )
    return _nullify_infinite_float_columns(
        _apply_lm2011_regression_transforms(
            event_panel_lf.join(text_signal_lf, on="doc_id", how="left").join(
                industries_lf,
                on="doc_id",
                how="left",
            )
        ).with_columns(pl.lit(text_scope).alias("text_scope"))
    )


def build_lm2011_sue_regression_panel(
    sue_panel_lf: pl.LazyFrame,
    full_10k_text_features_lf: pl.LazyFrame,
    company_history_lf: pl.LazyFrame,
    company_description_lf: pl.LazyFrame,
    *,
    ff48_siccodes_path: Path | str,
) -> pl.LazyFrame:
    _require_columns(
        sue_panel_lf,
        (
            "doc_id",
            "gvkey_int",
            "filing_date",
            "size_event",
            "bm_event",
            "share_turnover",
            "sue",
            "analyst_dispersion",
            "analyst_revisions",
            "pre_ffalpha",
            "institutional_ownership",
            "nasdaq_dummy",
        ),
        "sue_panel",
    )
    text_signal_lf = _select_text_signal_surface(full_10k_text_features_lf, label="full_10k_text_features")
    industries_lf = _attach_ff48_industries(
        sue_panel_lf,
        company_history_lf,
        company_description_lf,
        ff48_siccodes_path=ff48_siccodes_path,
    )
    return _nullify_infinite_float_columns(
        _apply_lm2011_regression_transforms(
            sue_panel_lf.join(text_signal_lf, on="doc_id", how="left").join(
                industries_lf,
                on="doc_id",
                how="left",
            )
        ).with_columns(pl.lit("full_10k").alias("text_scope"))
    )


def build_lm2011_normalized_difference_panel(
    return_regression_panel_lf: pl.LazyFrame,
) -> pl.LazyFrame:
    _require_columns(
        return_regression_panel_lf,
        (
            "doc_id",
            "filing_date",
            "ff48_industry_id",
            "lm_negative_prop",
            "h4n_inf_prop",
        ),
        "return_regression_panel",
    )
    base = return_regression_panel_lf.with_columns(
        pl.col("filing_date").cast(pl.Date, strict=False),
        pl.col("ff48_industry_id").cast(pl.Int32, strict=False),
        pl.col("lm_negative_prop").cast(pl.Float64, strict=False),
        pl.col("h4n_inf_prop").cast(pl.Float64, strict=False),
        pl.col("filing_date").dt.year().cast(pl.Int32, strict=False).alias("_filing_year"),
    )
    prior_year_stats = (
        base.group_by("ff48_industry_id", "_filing_year")
        .agg(
            pl.col("lm_negative_prop").mean().alias("_prior_negative_mean"),
            pl.col("lm_negative_prop").std().alias("_prior_negative_std"),
            pl.col("h4n_inf_prop").mean().alias("_prior_h4n_inf_mean"),
            pl.col("h4n_inf_prop").std().alias("_prior_h4n_inf_std"),
        )
        .with_columns((pl.col("_filing_year") + pl.lit(1, dtype=pl.Int32)).alias("_target_filing_year"))
    )
    return (
        base.join(
            prior_year_stats,
            left_on=["ff48_industry_id", "_filing_year"],
            right_on=["ff48_industry_id", "_target_filing_year"],
            how="left",
        )
        .with_columns(
            (
                (pl.col("lm_negative_prop") - pl.col("_prior_negative_mean"))
                / pl.col("_prior_negative_std")
            ).alias("normalized_difference_negative"),
            (
                (pl.col("h4n_inf_prop") - pl.col("_prior_h4n_inf_mean"))
                / pl.col("_prior_h4n_inf_std")
            ).alias("normalized_difference_h4n_inf"),
        )
        .filter(
            pl.col("ff48_industry_id").is_not_null()
            & pl.col("_prior_negative_mean").is_not_null()
            & pl.col("_prior_negative_std").is_not_null()
            & (pl.col("_prior_negative_std") != 0)
            & pl.col("_prior_h4n_inf_mean").is_not_null()
            & pl.col("_prior_h4n_inf_std").is_not_null()
            & (pl.col("_prior_h4n_inf_std") != 0)
        )
        .drop(
            "_filing_year",
            "_prior_negative_mean",
            "_prior_negative_std",
            "_prior_h4n_inf_mean",
            "_prior_h4n_inf_std",
            "right._filing_year",
            "_target_filing_year",
            strict=False,
        )
    )


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float | None:
    total_weight = sum(weights)
    if total_weight <= 0:
        return None
    return sum(weight * value for value, weight in zip(values, weights, strict=True)) / total_weight


def _resolve_quarter_weights(
    quarter_sizes: Sequence[float],
    *,
    quarter_weighting: QuarterWeighting,
) -> tuple[list[float], QuarterWeighting]:
    if quarter_weighting == _QUARTER_WEIGHTING_RULE_OBSERVATION_COUNT:
        return [float(size) for size in quarter_sizes], quarter_weighting
    if quarter_weighting == _QUARTER_WEIGHTING_RULE_EQUAL:
        return [1.0 for _ in quarter_sizes], quarter_weighting
    raise ValueError(
        "quarter_weighting must be one of "
        f"{_QUARTER_WEIGHTING_RULE_OBSERVATION_COUNT!r} or {_QUARTER_WEIGHTING_RULE_EQUAL!r}; "
        f"got {quarter_weighting!r}."
    )


def _newey_west_standard_error(
    values: Sequence[float],
    weights: Sequence[float],
    *,
    nw_lags: int,
) -> float | None:
    estimate = _weighted_mean(values, weights)
    total_weight = sum(weights)
    if estimate is None or total_weight <= 0:
        return None
    normalized_weights = [weight / total_weight for weight in weights]
    psi = [omega * (value - estimate) for value, omega in zip(values, normalized_weights, strict=True)]
    variance = sum(value * value for value in psi)
    max_lag = min(nw_lags, len(psi) - 1)
    for lag in range(1, max_lag + 1):
        bartlett_weight = 1.0 - (float(lag) / float(nw_lags + 1))
        covariance = sum(psi[idx] * psi[idx - lag] for idx in range(lag, len(psi)))
        variance += 2.0 * bartlett_weight * covariance
    return math.sqrt(max(variance, 0.0))


def _json_or_none(value: object) -> str | None:
    if not value:
        return None
    return json.dumps(value, separators=(",", ":"))


def _best_effort_condition_number(design: np.ndarray) -> float | None:
    try:
        condition_number = float(np.linalg.cond(design))
    except np.linalg.LinAlgError:
        return None
    return condition_number if math.isfinite(condition_number) else None


def _best_effort_duplicate_regressor_pairs(
    *,
    full_names: Sequence[str],
    design: np.ndarray,
) -> list[list[str]]:
    duplicate_pairs: list[list[str]] = []
    for left_idx in range(1, design.shape[1]):
        for right_idx in range(left_idx + 1, design.shape[1]):
            if np.array_equal(design[:, left_idx], design[:, right_idx]):
                duplicate_pairs.append([full_names[left_idx], full_names[right_idx]])
    return duplicate_pairs


def _best_effort_restoring_drop_candidates(
    *,
    full_names: Sequence[str],
    design: np.ndarray,
) -> list[str]:
    restoring_candidates: list[str] = []
    for idx in range(1, design.shape[1]):
        reduced_design = np.delete(design, idx, axis=1)
        if np.linalg.matrix_rank(reduced_design) == reduced_design.shape[1]:
            restoring_candidates.append(full_names[idx])
    return restoring_candidates


def _fit_cross_sectional_ols(
    df: pl.DataFrame,
    *,
    table_id: str,
    specification_id: str,
    text_scope: str,
    dependent_variable: str,
    signal_column: str,
    regressor_columns: tuple[str, ...],
    industry_col: str,
    label: str,
    quarter_start: dt.date,
    signal_inputs: Sequence[str] | None = None,
    on_rank_deficient: Literal["raise", "skip"] = "raise",
) -> _CrossSectionalOlsOutcome | None:
    industries = sorted(
        {
            int(value)
            for value in df.get_column(industry_col).drop_nulls().to_list()
            if value is not None
        }
    )
    dummy_industries = industries[1:]
    visible_names = (_INTERCEPT_NAME, *regressor_columns)
    full_names = (
        *visible_names,
        *[f"_industry_dummy_{industry_id}" for industry_id in dummy_industries],
    )
    n_obs = df.height
    column_count = len(full_names)
    effective_signal_inputs = tuple(signal_inputs or (signal_column,))
    if n_obs <= column_count:
        if on_rank_deficient != "skip":
            return None
        skipped_quarter_row = {
            "table_id": table_id,
            "specification_id": specification_id,
            "text_scope": text_scope,
            "dependent_variable": dependent_variable,
            "signal_name": ",".join(effective_signal_inputs),
            "quarter_start": quarter_start,
            "skip_reason": _INSUFFICIENT_DOF_SKIP_REASON,
            "n_obs": n_obs,
            "industry_count": len(industries),
            "rank": None,
            "column_count": column_count,
            "condition_number": None,
            "regressors": ", ".join(full_names),
            "duplicate_regressor_pairs": None,
            "restoring_drop_candidates": None,
        }
        return _CrossSectionalOlsOutcome(
            coefficients=None,
            n_obs=None,
            skipped_quarter_row=skipped_quarter_row,
        )

    endog: list[float] = []
    exog_rows: list[list[float]] = []
    for row in df.select(dependent_variable, industry_col, *regressor_columns).iter_rows(named=True):
        design_row = [float(row[column]) for column in regressor_columns]
        industry_value = int(row[industry_col])
        design_row.extend(1.0 if industry_value == industry_id else 0.0 for industry_id in dummy_industries)
        endog.append(float(row[dependent_variable]))
        exog_rows.append(design_row)
    design = np.column_stack(
        (
            np.ones(n_obs, dtype=float),
            np.asarray(exog_rows, dtype=float),
        )
    )
    rank = int(np.linalg.matrix_rank(design))
    try:
        results = _fit_checked_ols(
            endog,
            exog_rows,
            exog_names=full_names[1:],
            label=label,
        )
    except ValueError as exc:
        if (
            on_rank_deficient == "skip"
            and rank < column_count
            and "rank-deficient OLS design" in str(exc)
        ):
            duplicate_regressor_pairs = _best_effort_duplicate_regressor_pairs(
                full_names=full_names,
                design=design,
            )
            restoring_drop_candidates = _best_effort_restoring_drop_candidates(
                full_names=full_names,
                design=design,
            )
            condition_number = _best_effort_condition_number(design)
            skipped_quarter_row = {
                "table_id": table_id,
                "specification_id": specification_id,
                "text_scope": text_scope,
                "dependent_variable": dependent_variable,
                "signal_name": ",".join(effective_signal_inputs),
                "quarter_start": quarter_start,
                "skip_reason": _RANK_DEFICIENT_SKIP_REASON,
                "n_obs": n_obs,
                "industry_count": len(industries),
                "rank": rank,
                "column_count": column_count,
                "condition_number": condition_number,
                "regressors": ", ".join(full_names),
                "duplicate_regressor_pairs": _json_or_none(duplicate_regressor_pairs),
                "restoring_drop_candidates": _json_or_none(restoring_drop_candidates),
            }
            print(
                {
                    "table_id": table_id,
                    "specification_id": specification_id,
                    "text_scope": text_scope,
                    "dependent_variable": dependent_variable,
                    "signal_name": ",".join(effective_signal_inputs),
                    "quarter_start": str(quarter_start),
                    "skip_reason": _RANK_DEFICIENT_SKIP_REASON,
                    "n_obs": n_obs,
                    "rank": rank,
                    "column_count": column_count,
                    "condition_number": condition_number,
                    "duplicate_regressor_pairs": duplicate_regressor_pairs,
                    "restoring_drop_candidates": restoring_drop_candidates,
                }
            )
            return _CrossSectionalOlsOutcome(
                coefficients=None,
                n_obs=None,
                skipped_quarter_row=skipped_quarter_row,
            )
        raise
    df_resid_value = float(results.df_resid)
    if not math.isfinite(df_resid_value) or df_resid_value <= 0:
        if on_rank_deficient != "skip":
            return None
        skipped_quarter_row = {
            "table_id": table_id,
            "specification_id": specification_id,
            "text_scope": text_scope,
            "dependent_variable": dependent_variable,
            "signal_name": ",".join(effective_signal_inputs),
            "quarter_start": quarter_start,
            "skip_reason": _INSUFFICIENT_DOF_SKIP_REASON,
            "n_obs": n_obs,
            "industry_count": len(industries),
            "rank": rank,
            "column_count": column_count,
            "condition_number": _best_effort_condition_number(design),
            "regressors": ", ".join(full_names),
            "duplicate_regressor_pairs": None,
            "restoring_drop_candidates": None,
        }
        return _CrossSectionalOlsOutcome(
            coefficients=None,
            n_obs=None,
            skipped_quarter_row=skipped_quarter_row,
        )
    raw_r2 = float(results.rsquared)
    adj_r2 = float(results.rsquared_adj)
    quarter_fit_row = None
    skipped_quarter_row = None
    if math.isfinite(raw_r2) and math.isfinite(adj_r2):
        df_model_value = float(results.df_model)
        quarter_fit_row = {
            "table_id": table_id,
            "specification_id": specification_id,
            "text_scope": text_scope,
            "signal_name": ",".join(effective_signal_inputs),
            "signal_inputs": list(effective_signal_inputs),
            "dependent_variable": dependent_variable,
            "quarter_start": quarter_start,
            "n_obs": n_obs,
            "industry_count": len(industries),
            "industry_dummy_count": len(dummy_industries),
            "visible_regressor_count": len(visible_names),
            "full_regressor_count": column_count,
            "rank": rank,
            "df_model": df_model_value if math.isfinite(df_model_value) else None,
            "df_resid": df_resid_value,
            "condition_number": _best_effort_condition_number(design),
            "raw_r2": raw_r2,
            "adj_r2": adj_r2,
            "ssr": float(results.ssr) if math.isfinite(float(results.ssr)) else None,
            "centered_tss": (
                float(results.centered_tss)
                if math.isfinite(float(results.centered_tss))
                else None
            ),
            "weight": float(n_obs),
            "weighting_rule": _QUARTER_WEIGHTING_RULE_OBSERVATION_COUNT,
        }
    elif on_rank_deficient == "skip":
        skipped_quarter_row = {
            "table_id": table_id,
            "specification_id": specification_id,
            "text_scope": text_scope,
            "dependent_variable": dependent_variable,
            "signal_name": ",".join(effective_signal_inputs),
            "quarter_start": quarter_start,
            "skip_reason": _NONFINITE_FIT_SKIP_REASON,
            "n_obs": n_obs,
            "industry_count": len(industries),
            "rank": rank,
            "column_count": column_count,
            "condition_number": _best_effort_condition_number(design),
            "regressors": ", ".join(full_names),
            "duplicate_regressor_pairs": None,
            "restoring_drop_candidates": None,
        }
    return _CrossSectionalOlsOutcome(
        coefficients={name: float(value) for name, value in zip(full_names, results.params, strict=True)},
        n_obs=n_obs,
        quarter_fit_row=quarter_fit_row,
        skipped_quarter_row=skipped_quarter_row,
    )


def _run_lm2011_quarterly_fama_macbeth_bundle(
    panel_lf: pl.LazyFrame,
    *,
    table_id: str,
    text_scope: str,
    dependent_variable: str,
    signal_column: str,
    control_columns: Sequence[str],
    specification_id: str | None = None,
    filing_date_col: str = "filing_date",
    industry_col: str = "ff48_industry_id",
    nw_lags: int = 1,
    quarter_weighting: QuarterWeighting = _QUARTER_WEIGHTING_RULE_OBSERVATION_COUNT,
    signal_inputs: Sequence[str] | None = None,
    on_rank_deficient: Literal["raise", "skip"] = "raise",
) -> _QuarterlyFamaMacbethBundle:
    ordered_controls = _unique_preserving_order(tuple(control_columns))
    required_columns = (
        filing_date_col,
        dependent_variable,
        signal_column,
        industry_col,
        *ordered_controls,
    )
    _require_columns(panel_lf, required_columns, "lm2011_regression_panel")

    selected_lf = (
        panel_lf.select(
            pl.col(filing_date_col).cast(pl.Date, strict=False).alias(filing_date_col),
            pl.col(dependent_variable).cast(pl.Float64, strict=False).alias(dependent_variable),
            pl.col(signal_column).cast(pl.Float64, strict=False).alias(signal_column),
            pl.col(industry_col).cast(pl.Int32, strict=False).alias(industry_col),
            *[
                pl.col(column).cast(pl.Float64, strict=False).alias(column)
                for column in ordered_controls
            ],
        )
        .drop_nulls(subset=list(required_columns))
    )
    with_quarters_df = (
        selected_lf.with_columns(
            pl.col(filing_date_col)
            .map_elements(_quarter_start, return_dtype=pl.Date)
            .alias("_quarter_start")
        )
        .sort("_quarter_start", filing_date_col)
        .collect()
    )
    if with_quarters_df.height == 0:
        return _QuarterlyFamaMacbethBundle(
            results_df=_empty_lm2011_table_results_df(),
            skipped_quarters_df=_empty_skipped_quarters_df(),
        )

    regressor_columns = (signal_column, *ordered_controls)
    coefficient_time_series: dict[str, list[float]] = {
        name: [] for name in (_INTERCEPT_NAME, *regressor_columns)
    }
    quarter_sizes: list[float] = []
    retained_quarters = 0
    skipped_quarter_rows: list[dict[str, object]] = []
    quarter_fit_rows: list[dict[str, object]] = []
    resolved_specification_id = specification_id or signal_column

    for quarter_df in with_quarters_df.partition_by("_quarter_start", maintain_order=True):
        quarter_start = quarter_df.item(0, "_quarter_start")
        outcome = _fit_cross_sectional_ols(
            quarter_df,
            table_id=table_id,
            specification_id=resolved_specification_id,
            text_scope=text_scope,
            dependent_variable=dependent_variable,
            signal_column=signal_column,
            regressor_columns=regressor_columns,
            industry_col=industry_col,
            label=(
                f"Quarterly Fama-MacBeth quarter={quarter_start} "
                f"dependent={dependent_variable} signal={signal_column}"
            ),
            quarter_start=quarter_start,
            signal_inputs=signal_inputs,
            on_rank_deficient=on_rank_deficient,
        )
        if outcome is None:
            continue
        if outcome.skipped_quarter_row is not None:
            skipped_quarter_rows.append(outcome.skipped_quarter_row)
        if outcome.quarter_fit_row is not None:
            quarter_fit_rows.append(outcome.quarter_fit_row)
        if outcome.coefficients is None or outcome.n_obs is None:
            continue
        assert outcome.coefficients is not None
        assert outcome.n_obs is not None
        retained_quarters += 1
        quarter_sizes.append(float(outcome.n_obs))
        for name in coefficient_time_series:
            coefficient_time_series[name].append(outcome.coefficients[name])
        del quarter_df

    if retained_quarters == 0:
        return _QuarterlyFamaMacbethBundle(
            results_df=_empty_lm2011_table_results_df(),
            skipped_quarters_df=(
                pl.DataFrame(skipped_quarter_rows, schema_overrides=_SKIPPED_QUARTER_SCHEMA)
                if skipped_quarter_rows
                else _empty_skipped_quarters_df()
            ),
            quarter_fit_df=(
                pl.DataFrame(quarter_fit_rows, schema_overrides=_QUARTER_FIT_SCHEMA)
                if quarter_fit_rows
                else _empty_quarter_fit_df()
            ),
        )

    quarter_weights, weighting_rule = _resolve_quarter_weights(
        quarter_sizes,
        quarter_weighting=quarter_weighting,
    )
    if quarter_fit_rows:
        for fit_row, quarter_weight in zip(quarter_fit_rows, quarter_weights, strict=True):
            fit_row["weight"] = quarter_weight
            fit_row["weighting_rule"] = weighting_rule

    mean_quarter_n = sum(quarter_sizes) / float(retained_quarters)
    rows: list[dict[str, object]] = []
    for coefficient_name, values in coefficient_time_series.items():
        estimate = _weighted_mean(values, quarter_weights)
        standard_error = _newey_west_standard_error(values, quarter_weights, nw_lags=nw_lags)
        t_stat = None
        if estimate is not None and standard_error is not None and standard_error > 0:
            t_stat = estimate / standard_error
        rows.append(
            {
                "table_id": table_id,
                "specification_id": resolved_specification_id,
                "text_scope": text_scope,
                "signal_name": signal_column,
                "dependent_variable": dependent_variable,
                "coefficient_name": coefficient_name,
                "estimate": estimate,
                "standard_error": standard_error,
                "t_stat": t_stat,
                "n_quarters": retained_quarters,
                "mean_quarter_n": mean_quarter_n,
                "weighting_rule": weighting_rule,
                "nw_lags": nw_lags,
            }
        )
    return _QuarterlyFamaMacbethBundle(
        results_df=pl.DataFrame(rows, schema_overrides=_TABLE_RESULT_SCHEMA),
        skipped_quarters_df=(
            pl.DataFrame(skipped_quarter_rows, schema_overrides=_SKIPPED_QUARTER_SCHEMA)
            if skipped_quarter_rows
            else _empty_skipped_quarters_df()
        ),
        quarter_fit_df=(
            pl.DataFrame(quarter_fit_rows, schema_overrides=_QUARTER_FIT_SCHEMA)
            if quarter_fit_rows
            else _empty_quarter_fit_df()
        ),
    )


def run_lm2011_quarterly_fama_macbeth(
    panel_lf: pl.LazyFrame,
    *,
    table_id: str,
    text_scope: str,
    dependent_variable: str,
    signal_column: str,
    control_columns: Sequence[str],
    specification_id: str | None = None,
    filing_date_col: str = "filing_date",
    industry_col: str = "ff48_industry_id",
    nw_lags: int = 1,
    quarter_weighting: QuarterWeighting = _QUARTER_WEIGHTING_RULE_OBSERVATION_COUNT,
    on_rank_deficient: Literal["raise", "skip"] = "raise",
) -> pl.DataFrame:
    return _run_lm2011_quarterly_fama_macbeth_bundle(
        panel_lf,
        table_id=table_id,
        text_scope=text_scope,
        dependent_variable=dependent_variable,
        signal_column=signal_column,
        control_columns=control_columns,
        specification_id=specification_id,
        filing_date_col=filing_date_col,
        industry_col=industry_col,
        nw_lags=nw_lags,
        quarter_weighting=quarter_weighting,
        on_rank_deficient=on_rank_deficient,
    ).results_df


def run_lm2011_quarterly_fama_macbeth_with_diagnostics(
    panel_lf: pl.LazyFrame,
    *,
    table_id: str,
    text_scope: str,
    dependent_variable: str,
    signal_column: str,
    control_columns: Sequence[str],
    specification_id: str | None = None,
    filing_date_col: str = "filing_date",
    industry_col: str = "ff48_industry_id",
    nw_lags: int = 1,
    quarter_weighting: QuarterWeighting = _QUARTER_WEIGHTING_RULE_OBSERVATION_COUNT,
    signal_inputs: Sequence[str] | None = None,
    on_rank_deficient: Literal["raise", "skip"] = "skip",
) -> _QuarterlyFamaMacbethBundle:
    return _run_lm2011_quarterly_fama_macbeth_bundle(
        panel_lf,
        table_id=table_id,
        text_scope=text_scope,
        dependent_variable=dependent_variable,
        signal_column=signal_column,
        control_columns=control_columns,
        specification_id=specification_id,
        filing_date_col=filing_date_col,
        industry_col=industry_col,
        nw_lags=nw_lags,
        quarter_weighting=quarter_weighting,
        signal_inputs=signal_inputs,
        on_rank_deficient=on_rank_deficient,
    )


def _run_signal_family_with_diagnostics(
    panel_lf: pl.LazyFrame,
    *,
    table_id: str,
    text_scope: str,
    dependent_variable: str,
    signal_columns: Sequence[str],
    control_columns: Sequence[str],
    nw_lags: int = 1,
    quarter_weighting: QuarterWeighting = _QUARTER_WEIGHTING_RULE_OBSERVATION_COUNT,
) -> _QuarterlyFamaMacbethBundle:
    outputs = [
        _run_lm2011_quarterly_fama_macbeth_bundle(
            panel_lf,
            table_id=table_id,
            text_scope=text_scope,
            dependent_variable=dependent_variable,
            signal_column=signal_column,
            control_columns=control_columns,
            specification_id=signal_column,
            nw_lags=nw_lags,
            quarter_weighting=quarter_weighting,
            signal_inputs=(signal_column,),
            on_rank_deficient="skip",
        )
        for signal_column in signal_columns
    ]
    nonempty_results = [output.results_df for output in outputs if output.results_df.height > 0]
    nonempty_skips = [
        output.skipped_quarters_df
        for output in outputs
        if output.skipped_quarters_df.height > 0
    ]
    nonempty_quarter_fits = [
        output.quarter_fit_df
        for output in outputs
        if output.quarter_fit_df.height > 0
    ]
    return _QuarterlyFamaMacbethBundle(
        results_df=(
            pl.concat(nonempty_results, how="vertical_relaxed")
            if nonempty_results
            else _empty_lm2011_table_results_df()
        ),
        skipped_quarters_df=(
            pl.concat(nonempty_skips, how="vertical_relaxed")
            if nonempty_skips
            else _empty_skipped_quarters_df()
        ),
        quarter_fit_df=(
            pl.concat(nonempty_quarter_fits, how="vertical_relaxed")
            if nonempty_quarter_fits
            else _empty_quarter_fit_df()
        ),
    )


def _run_signal_family(
    panel_lf: pl.LazyFrame,
    *,
    table_id: str,
    text_scope: str,
    dependent_variable: str,
    signal_columns: Sequence[str],
    control_columns: Sequence[str],
    nw_lags: int = 1,
) -> pl.DataFrame:
    outputs = [
        run_lm2011_quarterly_fama_macbeth(
            panel_lf,
            table_id=table_id,
            text_scope=text_scope,
            dependent_variable=dependent_variable,
            signal_column=signal_column,
            control_columns=control_columns,
            specification_id=signal_column,
            nw_lags=nw_lags,
        )
        for signal_column in signal_columns
    ]
    nonempty_outputs = [output for output in outputs if output.height > 0]
    if not nonempty_outputs:
        return _empty_lm2011_table_results_df()
    return pl.concat(nonempty_outputs, how="vertical_relaxed")


def _concat_lm2011_table_results(frames: Sequence[pl.DataFrame]) -> pl.DataFrame:
    nonempty_frames = [frame for frame in frames if frame.height > 0]
    if not nonempty_frames:
        return _empty_lm2011_table_results_df()
    return pl.concat(nonempty_frames, how="vertical_relaxed")


def _concat_lm2011_quarterly_bundles(
    bundles: Sequence[_QuarterlyFamaMacbethBundle],
) -> _QuarterlyFamaMacbethBundle:
    return _QuarterlyFamaMacbethBundle(
        results_df=_concat_lm2011_table_results([bundle.results_df for bundle in bundles]),
        skipped_quarters_df=(
            pl.concat(
                [bundle.skipped_quarters_df for bundle in bundles if bundle.skipped_quarters_df.height > 0],
                how="vertical_relaxed",
            )
            if any(bundle.skipped_quarters_df.height > 0 for bundle in bundles)
            else _empty_skipped_quarters_df()
        ),
        quarter_fit_df=(
            pl.concat(
                [bundle.quarter_fit_df for bundle in bundles if bundle.quarter_fit_df.height > 0],
                how="vertical_relaxed",
            )
            if any(bundle.quarter_fit_df.height > 0 for bundle in bundles)
            else _empty_quarter_fit_df()
        ),
    )


def _run_lm2011_table_signal_family(
    panel_lf: pl.LazyFrame,
    *,
    table_id: str,
    text_scope: str,
    dependent_variable: str,
    signal_columns: Sequence[str],
    control_columns: Sequence[str],
    with_diagnostics: bool,
    nw_lags: int = 1, # Default to 1 lag for Newey-West standard errors, matching LM2011's quarterly Fama-MacBeth specification.
) -> _QuarterlyFamaMacbethBundle | pl.DataFrame:
    if with_diagnostics:
        return _run_signal_family_with_diagnostics(
            panel_lf,
            table_id=table_id,
            text_scope=text_scope,
            dependent_variable=dependent_variable,
            signal_columns=signal_columns,
            control_columns=control_columns,
            nw_lags=nw_lags,
        )
    return _run_signal_family(
        panel_lf,
        table_id=table_id,
        text_scope=text_scope,
        dependent_variable=dependent_variable,
        signal_columns=signal_columns,
        control_columns=control_columns,
        nw_lags=nw_lags,
    )


def _build_lm2011_table_iv_results_impl(
    event_panel_lf: pl.LazyFrame,
    full_10k_text_features_lf: pl.LazyFrame,
    company_history_lf: pl.LazyFrame,
    company_description_lf: pl.LazyFrame,
    *,
    ff48_siccodes_path: Path | str,
    control_columns: Sequence[str],
    with_diagnostics: bool,
) -> _QuarterlyFamaMacbethBundle | pl.DataFrame:
    panel_lf = build_lm2011_return_regression_panel(
        event_panel_lf,
        full_10k_text_features_lf,
        company_history_lf,
        company_description_lf,
        ff48_siccodes_path=ff48_siccodes_path,
        text_scope="full_10k",
    )
    return _run_lm2011_table_signal_family(
        panel_lf,
        table_id="table_iv_full_10k",
        text_scope="full_10k",
        dependent_variable="filing_period_excess_return",
        signal_columns=("h4n_inf_prop", "lm_negative_prop", "h4n_inf_tfidf", "lm_negative_tfidf"),
        control_columns=control_columns,
        with_diagnostics=with_diagnostics,
    )


def _build_lm2011_table_iv_results_bundle(
    event_panel_lf: pl.LazyFrame,
    full_10k_text_features_lf: pl.LazyFrame,
    company_history_lf: pl.LazyFrame,
    company_description_lf: pl.LazyFrame,
    *,
    ff48_siccodes_path: Path | str,
) -> _QuarterlyFamaMacbethBundle:
    return _build_lm2011_table_iv_results_impl(
        event_panel_lf,
        full_10k_text_features_lf,
        company_history_lf,
        company_description_lf,
        ff48_siccodes_path=ff48_siccodes_path,
        control_columns=_RETURN_CONTROL_COLUMNS,
        with_diagnostics=True,
    )


def build_lm2011_table_iv_results(
    event_panel_lf: pl.LazyFrame,
    full_10k_text_features_lf: pl.LazyFrame,
    company_history_lf: pl.LazyFrame,
    company_description_lf: pl.LazyFrame,
    *,
    ff48_siccodes_path: Path | str,
) -> pl.DataFrame:
    return _build_lm2011_table_iv_results_impl(
        event_panel_lf,
        full_10k_text_features_lf,
        company_history_lf,
        company_description_lf,
        ff48_siccodes_path=ff48_siccodes_path,
        control_columns=_RETURN_CONTROL_COLUMNS,
        with_diagnostics=False,
    )


def _build_lm2011_table_iv_results_no_ownership_bundle(
    event_panel_lf: pl.LazyFrame,
    full_10k_text_features_lf: pl.LazyFrame,
    company_history_lf: pl.LazyFrame,
    company_description_lf: pl.LazyFrame,
    *,
    ff48_siccodes_path: Path | str,
) -> _QuarterlyFamaMacbethBundle:
    return _build_lm2011_table_iv_results_impl(
        event_panel_lf,
        full_10k_text_features_lf,
        company_history_lf,
        company_description_lf,
        ff48_siccodes_path=ff48_siccodes_path,
        control_columns=_RETURN_CONTROL_COLUMNS_NO_OWNERSHIP,
        with_diagnostics=True,
    )


def build_lm2011_table_iv_results_no_ownership(
    event_panel_lf: pl.LazyFrame,
    full_10k_text_features_lf: pl.LazyFrame,
    company_history_lf: pl.LazyFrame,
    company_description_lf: pl.LazyFrame,
    *,
    ff48_siccodes_path: Path | str,
) -> pl.DataFrame:
    return _build_lm2011_table_iv_results_impl(
        event_panel_lf,
        full_10k_text_features_lf,
        company_history_lf,
        company_description_lf,
        ff48_siccodes_path=ff48_siccodes_path,
        control_columns=_RETURN_CONTROL_COLUMNS_NO_OWNERSHIP,
        with_diagnostics=False,
    )


def _build_lm2011_table_v_results_impl(
    event_panel_lf: pl.LazyFrame,
    mda_text_features_lf: pl.LazyFrame,
    company_history_lf: pl.LazyFrame,
    company_description_lf: pl.LazyFrame,
    *,
    ff48_siccodes_path: Path | str,
    control_columns: Sequence[str],
    with_diagnostics: bool,
) -> _QuarterlyFamaMacbethBundle | pl.DataFrame:
    _require_columns(mda_text_features_lf, ("doc_id", "total_token_count_mda"), "mda_text_features")
    panel_lf = build_lm2011_return_regression_panel(
        event_panel_lf,
        mda_text_features_lf,
        company_history_lf,
        company_description_lf,
        ff48_siccodes_path=ff48_siccodes_path,
        text_scope="mda_item_7",
    )
    panel_lf = panel_lf.filter(pl.col("total_token_count_mda").cast(pl.Float64, strict=False) >= 250.0)
    return _run_lm2011_table_signal_family(
        panel_lf,
        table_id="table_v_mda",
        text_scope="mda_item_7",
        dependent_variable="filing_period_excess_return",
        signal_columns=("h4n_inf_prop", "lm_negative_prop", "h4n_inf_tfidf", "lm_negative_tfidf"),
        control_columns=control_columns,
        with_diagnostics=with_diagnostics,
    )


def _build_lm2011_table_v_results_bundle(
    event_panel_lf: pl.LazyFrame,
    mda_text_features_lf: pl.LazyFrame,
    company_history_lf: pl.LazyFrame,
    company_description_lf: pl.LazyFrame,
    *,
    ff48_siccodes_path: Path | str,
) -> _QuarterlyFamaMacbethBundle:
    return _build_lm2011_table_v_results_impl(
        event_panel_lf,
        mda_text_features_lf,
        company_history_lf,
        company_description_lf,
        ff48_siccodes_path=ff48_siccodes_path,
        control_columns=_RETURN_CONTROL_COLUMNS,
        with_diagnostics=True,
    )


def build_lm2011_table_v_results(
    event_panel_lf: pl.LazyFrame,
    mda_text_features_lf: pl.LazyFrame,
    company_history_lf: pl.LazyFrame,
    company_description_lf: pl.LazyFrame,
    *,
    ff48_siccodes_path: Path | str,
) -> pl.DataFrame:
    return _build_lm2011_table_v_results_impl(
        event_panel_lf,
        mda_text_features_lf,
        company_history_lf,
        company_description_lf,
        ff48_siccodes_path=ff48_siccodes_path,
        control_columns=_RETURN_CONTROL_COLUMNS,
        with_diagnostics=False,
    )


def _build_lm2011_table_v_results_no_ownership_bundle(
    event_panel_lf: pl.LazyFrame,
    mda_text_features_lf: pl.LazyFrame,
    company_history_lf: pl.LazyFrame,
    company_description_lf: pl.LazyFrame,
    *,
    ff48_siccodes_path: Path | str,
) -> _QuarterlyFamaMacbethBundle:
    return _build_lm2011_table_v_results_impl(
        event_panel_lf,
        mda_text_features_lf,
        company_history_lf,
        company_description_lf,
        ff48_siccodes_path=ff48_siccodes_path,
        control_columns=_RETURN_CONTROL_COLUMNS_NO_OWNERSHIP,
        with_diagnostics=True,
    )


def build_lm2011_table_v_results_no_ownership(
    event_panel_lf: pl.LazyFrame,
    mda_text_features_lf: pl.LazyFrame,
    company_history_lf: pl.LazyFrame,
    company_description_lf: pl.LazyFrame,
    *,
    ff48_siccodes_path: Path | str,
) -> pl.DataFrame:
    return _build_lm2011_table_v_results_impl(
        event_panel_lf,
        mda_text_features_lf,
        company_history_lf,
        company_description_lf,
        ff48_siccodes_path=ff48_siccodes_path,
        control_columns=_RETURN_CONTROL_COLUMNS_NO_OWNERSHIP,
        with_diagnostics=False,
    )


def _build_lm2011_table_vi_results_impl(
    event_panel_lf: pl.LazyFrame,
    full_10k_text_features_lf: pl.LazyFrame,
    company_history_lf: pl.LazyFrame,
    company_description_lf: pl.LazyFrame,
    *,
    ff48_siccodes_path: Path | str,
    control_columns: Sequence[str],
    dependent_variables: Sequence[str],
    signal_columns: Sequence[str],
    with_diagnostics: bool,
) -> _QuarterlyFamaMacbethBundle | pl.DataFrame:
    panel_lf = build_lm2011_return_regression_panel(
        event_panel_lf,
        full_10k_text_features_lf,
        company_history_lf,
        company_description_lf,
        ff48_siccodes_path=ff48_siccodes_path,
        text_scope="full_10k",
    )
    if with_diagnostics:
        bundles: list[_QuarterlyFamaMacbethBundle] = []
        for dependent_variable in dependent_variables:
            outcome_bundle = _run_lm2011_table_signal_family(
                panel_lf,
                table_id="table_vi_full_10k_dictionary_surface",
                text_scope="full_10k",
                dependent_variable=dependent_variable,
                signal_columns=signal_columns,
                control_columns=control_columns,
                with_diagnostics=True,
            )
            assert isinstance(outcome_bundle, _QuarterlyFamaMacbethBundle)
            bundles.append(outcome_bundle)
        return _concat_lm2011_quarterly_bundles(bundles)

    frames: list[pl.DataFrame] = []
    for dependent_variable in dependent_variables:
        outcome_df = _run_lm2011_table_signal_family(
            panel_lf,
            table_id="table_vi_full_10k_dictionary_surface",
            text_scope="full_10k",
            dependent_variable=dependent_variable,
            signal_columns=signal_columns,
            control_columns=control_columns,
            with_diagnostics=False,
        )
        assert isinstance(outcome_df, pl.DataFrame)
        frames.append(outcome_df)
    return _concat_lm2011_table_results(frames)


def _build_lm2011_table_vi_results_bundle(
    event_panel_lf: pl.LazyFrame,
    full_10k_text_features_lf: pl.LazyFrame,
    company_history_lf: pl.LazyFrame,
    company_description_lf: pl.LazyFrame,
    *,
    ff48_siccodes_path: Path | str,
) -> _QuarterlyFamaMacbethBundle:
    return _build_lm2011_table_vi_results_impl(
        event_panel_lf,
        full_10k_text_features_lf,
        company_history_lf,
        company_description_lf,
        ff48_siccodes_path=ff48_siccodes_path,
        control_columns=_RETURN_CONTROL_COLUMNS,
        dependent_variables=_TABLE_VI_LEGACY_OWNERSHIP_DEPENDENT_VARIABLES,
        signal_columns=_TABLE_VI_LEGACY_OWNERSHIP_SIGNAL_COLUMNS,
        with_diagnostics=True,
    )


def build_lm2011_table_vi_results(
    event_panel_lf: pl.LazyFrame,
    full_10k_text_features_lf: pl.LazyFrame,
    company_history_lf: pl.LazyFrame,
    company_description_lf: pl.LazyFrame,
    *,
    ff48_siccodes_path: Path | str,
) -> pl.DataFrame:
    return _build_lm2011_table_vi_results_impl(
        event_panel_lf,
        full_10k_text_features_lf,
        company_history_lf,
        company_description_lf,
        ff48_siccodes_path=ff48_siccodes_path,
        control_columns=_RETURN_CONTROL_COLUMNS,
        dependent_variables=_TABLE_VI_LEGACY_OWNERSHIP_DEPENDENT_VARIABLES,
        signal_columns=_TABLE_VI_LEGACY_OWNERSHIP_SIGNAL_COLUMNS,
        with_diagnostics=False,
    )


def _build_lm2011_table_vi_results_no_ownership_bundle(
    event_panel_lf: pl.LazyFrame,
    full_10k_text_features_lf: pl.LazyFrame,
    company_history_lf: pl.LazyFrame,
    company_description_lf: pl.LazyFrame,
    *,
    ff48_siccodes_path: Path | str,
) -> _QuarterlyFamaMacbethBundle:
    return _build_lm2011_table_vi_results_impl(
        event_panel_lf,
        full_10k_text_features_lf,
        company_history_lf,
        company_description_lf,
        ff48_siccodes_path=ff48_siccodes_path,
        control_columns=_RETURN_CONTROL_COLUMNS_NO_OWNERSHIP,
        dependent_variables=_TABLE_VI_DEPENDENT_VARIABLES,
        signal_columns=_TABLE_VI_SIGNAL_COLUMNS,
        with_diagnostics=True,
    )


def build_lm2011_table_vi_results_no_ownership(
    event_panel_lf: pl.LazyFrame,
    full_10k_text_features_lf: pl.LazyFrame,
    company_history_lf: pl.LazyFrame,
    company_description_lf: pl.LazyFrame,
    *,
    ff48_siccodes_path: Path | str,
) -> pl.DataFrame:
    return _build_lm2011_table_vi_results_impl(
        event_panel_lf,
        full_10k_text_features_lf,
        company_history_lf,
        company_description_lf,
        ff48_siccodes_path=ff48_siccodes_path,
        control_columns=_RETURN_CONTROL_COLUMNS_NO_OWNERSHIP,
        dependent_variables=_TABLE_VI_DEPENDENT_VARIABLES,
        signal_columns=_TABLE_VI_SIGNAL_COLUMNS,
        with_diagnostics=False,
    )


def _build_lm2011_table_viii_results_impl(
    sue_panel_lf: pl.LazyFrame,
    full_10k_text_features_lf: pl.LazyFrame,
    company_history_lf: pl.LazyFrame,
    company_description_lf: pl.LazyFrame,
    *,
    ff48_siccodes_path: Path | str,
    control_columns: Sequence[str],
    with_diagnostics: bool,
) -> _QuarterlyFamaMacbethBundle | pl.DataFrame:
    panel_lf = build_lm2011_sue_regression_panel(
        sue_panel_lf,
        full_10k_text_features_lf,
        company_history_lf,
        company_description_lf,
        ff48_siccodes_path=ff48_siccodes_path,
    )
    return _run_lm2011_table_signal_family(
        panel_lf,
        table_id="table_viii_sue",
        text_scope="full_10k",
        dependent_variable="sue",
        signal_columns=("h4n_inf_prop", "lm_negative_prop", "h4n_inf_tfidf", "lm_negative_tfidf"),
        control_columns=control_columns,
        with_diagnostics=with_diagnostics,
    )


def _build_lm2011_table_viii_results_bundle(
    sue_panel_lf: pl.LazyFrame,
    full_10k_text_features_lf: pl.LazyFrame,
    company_history_lf: pl.LazyFrame,
    company_description_lf: pl.LazyFrame,
    *,
    ff48_siccodes_path: Path | str,
) -> _QuarterlyFamaMacbethBundle:
    return _build_lm2011_table_viii_results_impl(
        sue_panel_lf,
        full_10k_text_features_lf,
        company_history_lf,
        company_description_lf,
        ff48_siccodes_path=ff48_siccodes_path,
        control_columns=_SUE_CONTROL_COLUMNS,
        with_diagnostics=True,
    )


def build_lm2011_table_viii_results(
    sue_panel_lf: pl.LazyFrame,
    full_10k_text_features_lf: pl.LazyFrame,
    company_history_lf: pl.LazyFrame,
    company_description_lf: pl.LazyFrame,
    *,
    ff48_siccodes_path: Path | str,
) -> pl.DataFrame:
    return _build_lm2011_table_viii_results_impl(
        sue_panel_lf,
        full_10k_text_features_lf,
        company_history_lf,
        company_description_lf,
        ff48_siccodes_path=ff48_siccodes_path,
        control_columns=_SUE_CONTROL_COLUMNS,
        with_diagnostics=False,
    )


def _build_lm2011_table_viii_results_no_ownership_bundle(
    sue_panel_lf: pl.LazyFrame,
    full_10k_text_features_lf: pl.LazyFrame,
    company_history_lf: pl.LazyFrame,
    company_description_lf: pl.LazyFrame,
    *,
    ff48_siccodes_path: Path | str,
) -> _QuarterlyFamaMacbethBundle:
    return _build_lm2011_table_viii_results_impl(
        sue_panel_lf,
        full_10k_text_features_lf,
        company_history_lf,
        company_description_lf,
        ff48_siccodes_path=ff48_siccodes_path,
        control_columns=_SUE_CONTROL_COLUMNS_NO_OWNERSHIP,
        with_diagnostics=True,
    )


def build_lm2011_table_viii_results_no_ownership(
    sue_panel_lf: pl.LazyFrame,
    full_10k_text_features_lf: pl.LazyFrame,
    company_history_lf: pl.LazyFrame,
    company_description_lf: pl.LazyFrame,
    *,
    ff48_siccodes_path: Path | str,
) -> pl.DataFrame:
    return _build_lm2011_table_viii_results_impl(
        sue_panel_lf,
        full_10k_text_features_lf,
        company_history_lf,
        company_description_lf,
        ff48_siccodes_path=ff48_siccodes_path,
        control_columns=_SUE_CONTROL_COLUMNS_NO_OWNERSHIP,
        with_diagnostics=False,
    )


def _build_lm2011_table_ia_i_results_impl(
    event_panel_lf: pl.LazyFrame,
    full_10k_text_features_lf: pl.LazyFrame,
    company_history_lf: pl.LazyFrame,
    company_description_lf: pl.LazyFrame,
    *,
    ff48_siccodes_path: Path | str,
    control_columns: Sequence[str],
    with_diagnostics: bool,
) -> _QuarterlyFamaMacbethBundle | pl.DataFrame:
    return_panel_lf = build_lm2011_return_regression_panel(
        event_panel_lf,
        full_10k_text_features_lf,
        company_history_lf,
        company_description_lf,
        ff48_siccodes_path=ff48_siccodes_path,
        text_scope="full_10k",
    )
    panel_lf = build_lm2011_normalized_difference_panel(return_panel_lf)
    return _run_lm2011_table_signal_family(
        panel_lf,
        table_id="internet_appendix_table_ia_i",
        text_scope="full_10k",
        dependent_variable="filing_period_excess_return",
        signal_columns=("normalized_difference_negative", "normalized_difference_h4n_inf"),
        control_columns=control_columns,
        with_diagnostics=with_diagnostics,
    )


def _build_lm2011_table_ia_i_results_bundle(
    event_panel_lf: pl.LazyFrame,
    full_10k_text_features_lf: pl.LazyFrame,
    company_history_lf: pl.LazyFrame,
    company_description_lf: pl.LazyFrame,
    *,
    ff48_siccodes_path: Path | str,
) -> _QuarterlyFamaMacbethBundle:
    return _build_lm2011_table_ia_i_results_impl(
        event_panel_lf,
        full_10k_text_features_lf,
        company_history_lf,
        company_description_lf,
        ff48_siccodes_path=ff48_siccodes_path,
        control_columns=_RETURN_CONTROL_COLUMNS,
        with_diagnostics=True,
    )


def build_lm2011_table_ia_i_results(
    event_panel_lf: pl.LazyFrame,
    full_10k_text_features_lf: pl.LazyFrame,
    company_history_lf: pl.LazyFrame,
    company_description_lf: pl.LazyFrame,
    *,
    ff48_siccodes_path: Path | str,
) -> pl.DataFrame:
    return _build_lm2011_table_ia_i_results_impl(
        event_panel_lf,
        full_10k_text_features_lf,
        company_history_lf,
        company_description_lf,
        ff48_siccodes_path=ff48_siccodes_path,
        control_columns=_RETURN_CONTROL_COLUMNS,
        with_diagnostics=False,
    )


def _build_lm2011_table_ia_i_results_no_ownership_bundle(
    event_panel_lf: pl.LazyFrame,
    full_10k_text_features_lf: pl.LazyFrame,
    company_history_lf: pl.LazyFrame,
    company_description_lf: pl.LazyFrame,
    *,
    ff48_siccodes_path: Path | str,
) -> _QuarterlyFamaMacbethBundle:
    return _build_lm2011_table_ia_i_results_impl(
        event_panel_lf,
        full_10k_text_features_lf,
        company_history_lf,
        company_description_lf,
        ff48_siccodes_path=ff48_siccodes_path,
        control_columns=_RETURN_CONTROL_COLUMNS_NO_OWNERSHIP,
        with_diagnostics=True,
    )


def build_lm2011_table_ia_i_results_no_ownership(
    event_panel_lf: pl.LazyFrame,
    full_10k_text_features_lf: pl.LazyFrame,
    company_history_lf: pl.LazyFrame,
    company_description_lf: pl.LazyFrame,
    *,
    ff48_siccodes_path: Path | str,
) -> pl.DataFrame:
    return _build_lm2011_table_ia_i_results_impl(
        event_panel_lf,
        full_10k_text_features_lf,
        company_history_lf,
        company_description_lf,
        ff48_siccodes_path=ff48_siccodes_path,
        control_columns=_RETURN_CONTROL_COLUMNS_NO_OWNERSHIP,
        with_diagnostics=False,
    )


def build_lm2011_table_ia_ii_results_from_monthly_returns(
    trading_strategy_monthly_returns_lf: pl.LazyFrame,
    ff_factors_monthly_with_mom_lf: pl.LazyFrame,
) -> pl.DataFrame:
    monthly_returns_df = trading_strategy_monthly_returns_lf.collect()
    if monthly_returns_df.height == 0:
        return _empty_lm2011_table_results_df()

    summary_df = build_lm2011_trading_strategy_ff4_summary(
        trading_strategy_monthly_returns_lf,
        ff_factors_monthly_with_mom_lf,
    ).collect()
    mean_returns_df = monthly_returns_df.group_by("sort_signal_name").agg(
        pl.col("long_short_return").mean().alias("mean_long_short_return")
    )
    combined = mean_returns_df.join(summary_df, on="sort_signal_name", how="left").sort(
        "sort_signal_name"
    )
    rows: list[dict[str, object]] = []
    coefficient_pairs = (
        ("mean_long_short_return", "mean_long_short_return", None, None),
        (
            "alpha_ff3_mom",
            "alpha_ff3_mom",
            "alpha_ff3_mom_standard_error",
            "alpha_ff3_mom_t_stat",
        ),
        ("beta_market", "beta_market", "beta_market_standard_error", "beta_market_t_stat"),
        ("beta_smb", "beta_smb", "beta_smb_standard_error", "beta_smb_t_stat"),
        ("beta_hml", "beta_hml", "beta_hml_standard_error", "beta_hml_t_stat"),
        ("beta_mom", "beta_mom", "beta_mom_standard_error", "beta_mom_t_stat"),
        ("r2", "r2", None, None),
    )
    for row in combined.iter_rows(named=True):
        signal_name = str(row["sort_signal_name"])
        for (
            coefficient_name,
            source_column,
            standard_error_column,
            t_stat_column,
        ) in coefficient_pairs:
            rows.append(
                {
                    "table_id": "internet_appendix_table_ia_ii",
                    "specification_id": signal_name,
                    "text_scope": "full_10k",
                    "signal_name": signal_name,
                    "dependent_variable": "long_short_return",
                    "coefficient_name": coefficient_name,
                    "estimate": (
                        float(row[source_column])
                        if row[source_column] is not None
                        else None
                    ),
                    "standard_error": (
                        float(row[standard_error_column])
                        if standard_error_column is not None and row[standard_error_column] is not None
                        else None
                    ),
                    "t_stat": (
                        float(row[t_stat_column])
                        if t_stat_column is not None and row[t_stat_column] is not None
                        else None
                    ),
                    "n_quarters": None,
                    "mean_quarter_n": None,
                    "weighting_rule": None,
                    "nw_lags": None,
                }
            )
    return pl.DataFrame(rows, schema_overrides=_TABLE_RESULT_SCHEMA)


def build_lm2011_table_ia_ii_results(
    event_panel_lf: pl.LazyFrame,
    sec_parsed_lf: pl.LazyFrame,
    monthly_stock_lf: pl.LazyFrame,
    ff_factors_monthly_with_mom_lf: pl.LazyFrame,
    *,
    lm_dictionary_lists: Mapping[str, Iterable[str]],
    harvard_negative_word_list: Iterable[str] | None,
    master_dictionary_words: Iterable[str],
    portfolio_weighting: str = "equal",
    monthly_return_col: str = "MRET",
) -> pl.DataFrame:
    monthly_returns_lf = build_lm2011_trading_strategy_monthly_returns(
        event_panel_lf,
        sec_parsed_lf,
        monthly_stock_lf,
        lm_dictionary_lists=lm_dictionary_lists,
        harvard_negative_word_list=harvard_negative_word_list,
        master_dictionary_words=master_dictionary_words,
        portfolio_weighting=portfolio_weighting,
        monthly_return_col=monthly_return_col,
    )
    return build_lm2011_table_ia_ii_results_from_monthly_returns(
        monthly_returns_lf,
        ff_factors_monthly_with_mom_lf,
    )


__all__ = [
    "build_lm2011_return_regression_panel",
    "build_lm2011_sue_regression_panel",
    "build_lm2011_normalized_difference_panel",
    "run_lm2011_quarterly_fama_macbeth",
    "run_lm2011_quarterly_fama_macbeth_with_diagnostics",
    "build_lm2011_table_iv_results",
    "build_lm2011_table_v_results",
    "build_lm2011_table_vi_results",
    "build_lm2011_table_viii_results",
    "build_lm2011_table_ia_i_results",
    "build_lm2011_table_ia_ii_results",
    "build_lm2011_table_ia_ii_results_from_monthly_returns",
]

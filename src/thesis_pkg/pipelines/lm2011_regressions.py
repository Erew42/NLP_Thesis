from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import datetime as dt
import math
from pathlib import Path

import polars as pl

from thesis_pkg.core.ccm.lm2011 import attach_lm2011_industry_classifications
from thesis_pkg.pipelines.lm2011_pipeline import (
    _apply_lm2011_regression_transforms,
    _require_columns,
    _solve_linear_system,
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
_QUARTER_WEIGHTING_RULE = "quarter_observation_count"
_INTERCEPT_NAME = "intercept"


def _empty_lm2011_table_results_df() -> pl.DataFrame:
    return pl.DataFrame(schema=_TABLE_RESULT_SCHEMA)


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
        and (name.startswith("token_count_") or name.endswith("_prop") or name.endswith("_tfidf"))
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
    return _apply_lm2011_regression_transforms(
        event_panel_lf.join(text_signal_lf, on="doc_id", how="left").join(
            industries_lf,
            on="doc_id",
            how="left",
        )
    ).with_columns(pl.lit(text_scope).alias("text_scope"))


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
    return _apply_lm2011_regression_transforms(
        sue_panel_lf.join(text_signal_lf, on="doc_id", how="left").join(
            industries_lf,
            on="doc_id",
            how="left",
        )
    ).with_columns(pl.lit("full_10k").alias("text_scope"))


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


def _fit_cross_sectional_ols(
    df: pl.DataFrame,
    *,
    dependent_variable: str,
    regressor_columns: tuple[str, ...],
    industry_col: str,
) -> tuple[dict[str, float], int] | None:
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
    if n_obs < len(full_names):
        return None

    xtx = [[0.0 for _ in range(len(full_names))] for _ in range(len(full_names))]
    xty = [0.0 for _ in range(len(full_names))]
    for row in df.select(dependent_variable, industry_col, *regressor_columns).iter_rows(named=True):
        design_row = [1.0]
        design_row.extend(float(row[column]) for column in regressor_columns)
        industry_value = int(row[industry_col])
        design_row.extend(1.0 if industry_value == industry_id else 0.0 for industry_id in dummy_industries)
        y_value = float(row[dependent_variable])
        for idx in range(len(full_names)):
            xty[idx] += design_row[idx] * y_value
            for jdx in range(len(full_names)):
                xtx[idx][jdx] += design_row[idx] * design_row[jdx]
    beta = _solve_linear_system(xtx, xty)
    if beta is None:
        return None
    return {name: float(value) for name, value in zip(full_names, beta, strict=True)}, n_obs


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
) -> pl.DataFrame:
    ordered_controls = _unique_preserving_order(tuple(control_columns))
    required_columns = (
        filing_date_col,
        dependent_variable,
        signal_column,
        industry_col,
        *ordered_controls,
    )
    _require_columns(panel_lf, required_columns, "lm2011_regression_panel")

    selected = (
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
        .collect()
    )
    if selected.height == 0:
        return _empty_lm2011_table_results_df()

    with_quarters = selected.with_columns(
        pl.col(filing_date_col)
        .map_elements(_quarter_start, return_dtype=pl.Date)
        .alias("_quarter_start")
    ).sort("_quarter_start", filing_date_col)

    regressor_columns = (signal_column, *ordered_controls)
    coefficient_time_series: dict[str, list[float]] = {
        name: [] for name in (_INTERCEPT_NAME, *regressor_columns)
    }
    quarter_sizes: list[float] = []
    retained_quarters = 0

    for quarter_df in with_quarters.partition_by("_quarter_start", maintain_order=True):
        fit = _fit_cross_sectional_ols(
            quarter_df,
            dependent_variable=dependent_variable,
            regressor_columns=regressor_columns,
            industry_col=industry_col,
        )
        if fit is None:
            continue
        coefficients, n_obs = fit
        retained_quarters += 1
        quarter_sizes.append(float(n_obs))
        for name in coefficient_time_series:
            coefficient_time_series[name].append(coefficients[name])

    if retained_quarters == 0:
        return _empty_lm2011_table_results_df()

    mean_quarter_n = sum(quarter_sizes) / float(retained_quarters)
    rows: list[dict[str, object]] = []
    for coefficient_name, values in coefficient_time_series.items():
        estimate = _weighted_mean(values, quarter_sizes)
        standard_error = _newey_west_standard_error(values, quarter_sizes, nw_lags=nw_lags)
        t_stat = None
        if estimate is not None and standard_error is not None and standard_error > 0:
            t_stat = estimate / standard_error
        rows.append(
            {
                "table_id": table_id,
                "specification_id": specification_id or signal_column,
                "text_scope": text_scope,
                "signal_name": signal_column,
                "dependent_variable": dependent_variable,
                "coefficient_name": coefficient_name,
                "estimate": estimate,
                "standard_error": standard_error,
                "t_stat": t_stat,
                "n_quarters": retained_quarters,
                "mean_quarter_n": mean_quarter_n,
                "weighting_rule": _QUARTER_WEIGHTING_RULE,
                "nw_lags": nw_lags,
            }
        )
    return pl.DataFrame(rows, schema_overrides=_TABLE_RESULT_SCHEMA)


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
    nonempty = [output for output in outputs if output.height > 0]
    if not nonempty:
        return _empty_lm2011_table_results_df()
    return pl.concat(nonempty, how="vertical_relaxed")


def build_lm2011_table_iv_results(
    event_panel_lf: pl.LazyFrame,
    full_10k_text_features_lf: pl.LazyFrame,
    company_history_lf: pl.LazyFrame,
    company_description_lf: pl.LazyFrame,
    *,
    ff48_siccodes_path: Path | str,
) -> pl.DataFrame:
    panel_lf = build_lm2011_return_regression_panel(
        event_panel_lf,
        full_10k_text_features_lf,
        company_history_lf,
        company_description_lf,
        ff48_siccodes_path=ff48_siccodes_path,
        text_scope="full_10k",
    )
    return _run_signal_family(
        panel_lf,
        table_id="table_iv_full_10k",
        text_scope="full_10k",
        dependent_variable="filing_period_excess_return",
        signal_columns=("h4n_inf_prop", "lm_negative_prop", "h4n_inf_tfidf", "lm_negative_tfidf"),
        control_columns=_RETURN_CONTROL_COLUMNS,
    )


def build_lm2011_table_v_results(
    event_panel_lf: pl.LazyFrame,
    mda_text_features_lf: pl.LazyFrame,
    company_history_lf: pl.LazyFrame,
    company_description_lf: pl.LazyFrame,
    *,
    ff48_siccodes_path: Path | str,
) -> pl.DataFrame:
    panel_lf = build_lm2011_return_regression_panel(
        event_panel_lf,
        mda_text_features_lf,
        company_history_lf,
        company_description_lf,
        ff48_siccodes_path=ff48_siccodes_path,
        text_scope="mda_item_7",
    )
    panel_lf = panel_lf.filter(pl.col("token_count_mda").cast(pl.Float64, strict=False) >= 250.0)
    return _run_signal_family(
        panel_lf,
        table_id="table_v_mda",
        text_scope="mda_item_7",
        dependent_variable="filing_period_excess_return",
        signal_columns=("h4n_inf_prop", "lm_negative_prop", "h4n_inf_tfidf", "lm_negative_tfidf"),
        control_columns=_RETURN_CONTROL_COLUMNS,
    )


def build_lm2011_table_vi_results(
    event_panel_lf: pl.LazyFrame,
    full_10k_text_features_lf: pl.LazyFrame,
    company_history_lf: pl.LazyFrame,
    company_description_lf: pl.LazyFrame,
    *,
    ff48_siccodes_path: Path | str,
) -> pl.DataFrame:
    panel_lf = build_lm2011_return_regression_panel(
        event_panel_lf,
        full_10k_text_features_lf,
        company_history_lf,
        company_description_lf,
        ff48_siccodes_path=ff48_siccodes_path,
        text_scope="full_10k",
    )
    return _run_signal_family(
        panel_lf,
        table_id="table_vi_full_10k_dictionary_surface",
        text_scope="full_10k",
        dependent_variable="filing_period_excess_return",
        signal_columns=(
            "lm_negative_prop",
            "lm_negative_tfidf",
            "lm_positive_prop",
            "lm_positive_tfidf",
            "lm_uncertainty_prop",
            "lm_uncertainty_tfidf",
            "lm_litigious_prop",
            "lm_litigious_tfidf",
            "lm_modal_strong_prop",
            "lm_modal_strong_tfidf",
            "lm_modal_weak_prop",
            "lm_modal_weak_tfidf",
        ),
        control_columns=_RETURN_CONTROL_COLUMNS,
    )


def build_lm2011_table_viii_results(
    sue_panel_lf: pl.LazyFrame,
    full_10k_text_features_lf: pl.LazyFrame,
    company_history_lf: pl.LazyFrame,
    company_description_lf: pl.LazyFrame,
    *,
    ff48_siccodes_path: Path | str,
) -> pl.DataFrame:
    panel_lf = build_lm2011_sue_regression_panel(
        sue_panel_lf,
        full_10k_text_features_lf,
        company_history_lf,
        company_description_lf,
        ff48_siccodes_path=ff48_siccodes_path,
    )
    return _run_signal_family(
        panel_lf,
        table_id="table_viii_sue",
        text_scope="full_10k",
        dependent_variable="sue",
        signal_columns=("h4n_inf_prop", "lm_negative_prop", "h4n_inf_tfidf", "lm_negative_tfidf"),
        control_columns=_SUE_CONTROL_COLUMNS,
    )


def build_lm2011_table_ia_i_results(
    event_panel_lf: pl.LazyFrame,
    full_10k_text_features_lf: pl.LazyFrame,
    company_history_lf: pl.LazyFrame,
    company_description_lf: pl.LazyFrame,
    *,
    ff48_siccodes_path: Path | str,
) -> pl.DataFrame:
    return_panel_lf = build_lm2011_return_regression_panel(
        event_panel_lf,
        full_10k_text_features_lf,
        company_history_lf,
        company_description_lf,
        ff48_siccodes_path=ff48_siccodes_path,
        text_scope="full_10k",
    )
    panel_lf = build_lm2011_normalized_difference_panel(return_panel_lf)
    return _run_signal_family(
        panel_lf,
        table_id="internet_appendix_table_ia_i",
        text_scope="full_10k",
        dependent_variable="filing_period_excess_return",
        signal_columns=("normalized_difference_negative", "normalized_difference_h4n_inf"),
        control_columns=_RETURN_CONTROL_COLUMNS,
    )


def build_lm2011_table_ia_ii_results(
    event_panel_lf: pl.LazyFrame,
    sec_parsed_lf: pl.LazyFrame,
    monthly_stock_lf: pl.LazyFrame,
    ff_factors_monthly_with_mom_lf: pl.LazyFrame,
    *,
    lm_dictionary_lists: Mapping[str, Iterable[str]],
    harvard_negative_word_list: Iterable[str] | None,
    portfolio_weighting: str = "equal",
    monthly_return_col: str = "MRET",
) -> pl.DataFrame:
    monthly_returns_lf = build_lm2011_trading_strategy_monthly_returns(
        event_panel_lf,
        sec_parsed_lf,
        monthly_stock_lf,
        lm_dictionary_lists=lm_dictionary_lists,
        harvard_negative_word_list=harvard_negative_word_list,
        portfolio_weighting=portfolio_weighting,
        monthly_return_col=monthly_return_col,
    )
    monthly_returns_df = monthly_returns_lf.collect()
    if monthly_returns_df.height == 0:
        return _empty_lm2011_table_results_df()

    summary_df = build_lm2011_trading_strategy_ff4_summary(
        monthly_returns_lf,
        ff_factors_monthly_with_mom_lf,
    ).collect()
    mean_returns_df = monthly_returns_df.group_by("sort_signal_name").agg(
        pl.col("long_short_return").mean().alias("mean_long_short_return")
    )
    combined = mean_returns_df.join(summary_df, on="sort_signal_name", how="left").sort("sort_signal_name")
    rows: list[dict[str, object]] = []
    coefficient_pairs = (
        ("mean_long_short_return", "mean_long_short_return"),
        ("alpha_ff3_mom", "alpha_ff3_mom"),
        ("beta_market", "beta_market"),
        ("beta_smb", "beta_smb"),
        ("beta_hml", "beta_hml"),
        ("beta_mom", "beta_mom"),
        ("r2", "r2"),
    )
    for row in combined.iter_rows(named=True):
        signal_name = str(row["sort_signal_name"])
        for coefficient_name, source_column in coefficient_pairs:
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
                    "standard_error": None,
                    "t_stat": None,
                    "n_quarters": None,
                    "mean_quarter_n": None,
                    "weighting_rule": None,
                    "nw_lags": None,
                }
            )
    return pl.DataFrame(rows, schema_overrides=_TABLE_RESULT_SCHEMA)


__all__ = [
    "build_lm2011_return_regression_panel",
    "build_lm2011_sue_regression_panel",
    "build_lm2011_normalized_difference_panel",
    "run_lm2011_quarterly_fama_macbeth",
    "build_lm2011_table_iv_results",
    "build_lm2011_table_v_results",
    "build_lm2011_table_vi_results",
    "build_lm2011_table_viii_results",
    "build_lm2011_table_ia_i_results",
    "build_lm2011_table_ia_ii_results",
]

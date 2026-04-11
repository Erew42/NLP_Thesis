from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import datetime as dt
from dataclasses import dataclass
import math
from pathlib import Path

import polars as pl

from thesis_pkg.core.sec.lm2011_text import build_lm2011_text_features_mda
from thesis_pkg.pipelines.lm2011_pipeline import _apply_lm2011_regression_transforms
from thesis_pkg.pipelines.lm2011_pipeline import _require_columns
from thesis_pkg.pipelines.lm2011_regressions import _attach_ff48_industries
from thesis_pkg.pipelines.lm2011_regressions import _nullify_infinite_float_columns
from thesis_pkg.pipelines.lm2011_regressions import run_lm2011_quarterly_fama_macbeth


EXTENSION_SAMPLE_START = dt.date(2009, 1, 1)
EXTENSION_SAMPLE_END = dt.date(2024, 12, 31)
EXTENSION_SAMPLE_WINDOW = "2009_2024"
EXTENSION_PRIMARY_OUTCOME = "filing_period_excess_return"
EXTENSION_SECONDARY_OUTCOMES: tuple[str, ...] = (
    "abnormal_volume",
    "postevent_return_volatility",
)
EXTENSION_PRIMARY_TEXT_SCOPES: tuple[str, ...] = (
    "item_7_mda",
    "item_1a_risk_factors",
)
EXTENSION_ITEM_SCOPE_IDS: Mapping[str, str] = {
    "item_7_mda": "7",
    "item_1a_risk_factors": "1A",
    "item_1_business": "1",
}
EXTENSION_DICTIONARY_FAMILY_LM2011 = "lm2011_frozen"
EXTENSION_FINBERT_MODEL_FAMILY = "finbert"
EXTENSION_JOINT_FEATURE_FAMILY = "dictionary_plus_finbert"
EXTENSION_SIGNAL_COLUMNS: tuple[str, ...] = (
    "h4n_inf_prop",
    "h4n_inf_tfidf",
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
)
EXTENSION_BASE_CONTROL_COLUMNS: tuple[str, ...] = (
    "log_size",
    "log_book_to_market",
    "log_share_turnover",
    "pre_ffalpha",
    "nasdaq_dummy",
)
EXTENSION_OWNERSHIP_CONTROL = "institutional_ownership_proxy_refinitiv"


@dataclass(frozen=True)
class Lm2011ExtensionControlSet:
    control_set_id: str
    spec_alias: str
    controls: tuple[str, ...]
    sample_rule: str
    common_support_ownership: bool
    includes_ownership_control: bool


@dataclass(frozen=True)
class Lm2011ExtensionComparisonSpec:
    specification_name: str
    feature_family: str
    signal_inputs: tuple[str, ...]


_CONTROL_SETS: tuple[Lm2011ExtensionControlSet, ...] = (
    Lm2011ExtensionControlSet(
        control_set_id="C0",
        spec_alias="C0_base_no_ownership",
        controls=EXTENSION_BASE_CONTROL_COLUMNS,
        sample_rule="Maximum-N extension sample; does not require Refinitiv proxy ownership.",
        common_support_ownership=False,
        includes_ownership_control=False,
    ),
    Lm2011ExtensionControlSet(
        control_set_id="C1",
        spec_alias="C0_common_support_no_ownership",
        controls=EXTENSION_BASE_CONTROL_COLUMNS,
        sample_rule="Restrict to rows with nonmissing Refinitiv proxy ownership, but do not include ownership as a control.",
        common_support_ownership=True,
        includes_ownership_control=False,
    ),
    Lm2011ExtensionControlSet(
        control_set_id="C2",
        spec_alias="C1_proxy_ownership_common_support",
        controls=(*EXTENSION_BASE_CONTROL_COLUMNS, EXTENSION_OWNERSHIP_CONTROL),
        sample_rule="Use the same common-support rows as C1 and include Refinitiv proxy ownership as a control.",
        common_support_ownership=True,
        includes_ownership_control=True,
    ),
)
_COMPARISON_SPECS: tuple[Lm2011ExtensionComparisonSpec, ...] = (
    Lm2011ExtensionComparisonSpec(
        specification_name="dictionary_only",
        feature_family=EXTENSION_DICTIONARY_FAMILY_LM2011,
        signal_inputs=("lm_negative_tfidf",),
    ),
    Lm2011ExtensionComparisonSpec(
        specification_name="finbert_only",
        feature_family=EXTENSION_FINBERT_MODEL_FAMILY,
        signal_inputs=("finbert_neg_prob_lenw_mean",),
    ),
    Lm2011ExtensionComparisonSpec(
        specification_name="dictionary_finbert_joint",
        feature_family=EXTENSION_JOINT_FEATURE_FAMILY,
        signal_inputs=("lm_negative_tfidf", "finbert_neg_prob_lenw_mean"),
    ),
)
_EXTENSION_RESULTS_SCHEMA: dict[str, pl.DataType] = {
    "run_id": pl.Utf8,
    "sample_window": pl.Utf8,
    "text_scope": pl.Utf8,
    "outcome_name": pl.Utf8,
    "feature_family": pl.Utf8,
    "control_set_id": pl.Utf8,
    "control_set_alias": pl.Utf8,
    "specification_name": pl.Utf8,
    "coefficient_name": pl.Utf8,
    "signal_name": pl.Utf8,
    "estimate": pl.Float64,
    "standard_error": pl.Float64,
    "t_stat": pl.Float64,
    "p_value": pl.Float64,
    "n_obs": pl.Int64,
    "n_quarters": pl.Int32,
    "mean_quarter_n": pl.Float64,
    "average_r2": pl.Float64,
    "weighting_rule": pl.Utf8,
    "nw_lags": pl.Int32,
    "estimator_status": pl.Utf8,
    "failure_reason": pl.Utf8,
}


def _empty_extension_dictionary_features_df() -> pl.DataFrame:
    schema: dict[str, pl.DataType] = {
        "doc_id": pl.Utf8,
        "cik_10": pl.Utf8,
        "filing_date": pl.Date,
        "text_scope": pl.Utf8,
        "dictionary_family": pl.Utf8,
        "total_token_count": pl.Int32,
        "token_count": pl.Int32,
    }
    schema.update({column: pl.Float64 for column in EXTENSION_SIGNAL_COLUMNS})
    return pl.DataFrame(schema=schema)


def _empty_extension_model_features_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "doc_id": pl.Utf8,
            "text_scope": pl.Utf8,
            "model_name": pl.Utf8,
            "model_version": pl.Utf8,
            "segment_policy_id": pl.Utf8,
            "finbert_segment_count": pl.Int32,
            "finbert_token_count_512": pl.Int64,
            "finbert_token_count_512_sum": pl.Int64,
            "finbert_neg_prob_lenw_mean": pl.Float64,
            "finbert_pos_prob_lenw_mean": pl.Float64,
            "finbert_neu_prob_lenw_mean": pl.Float64,
            "finbert_net_negative_lenw_mean": pl.Float64,
            "finbert_neg_dominant_share": pl.Float64,
        }
    )


def _empty_extension_results_df() -> pl.DataFrame:
    return pl.DataFrame(schema=_EXTENSION_RESULTS_SCHEMA)


def _control_set_by_id(control_set_id: str) -> Lm2011ExtensionControlSet:
    for control_set in _CONTROL_SETS:
        if control_set.control_set_id == control_set_id:
            return control_set
    raise ValueError(f"Unknown LM2011 extension control_set_id: {control_set_id!r}")


def _comparison_spec_by_name(specification_name: str) -> Lm2011ExtensionComparisonSpec:
    for comparison_spec in _COMPARISON_SPECS:
        if comparison_spec.specification_name == specification_name:
            return comparison_spec
    raise ValueError(f"Unknown LM2011 extension specification_name: {specification_name!r}")


def _normalize_scope_value(value: str) -> str:
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


def normalize_lm2011_extension_text_scope_expr(expr: pl.Expr) -> pl.Expr:
    raw = expr.cast(pl.Utf8, strict=False).str.strip_chars().str.to_lowercase().str.replace_all("-", "_")
    return (
        pl.when(raw.is_in(["7", "item_7", "mda_item_7", "item_7_mda"]))
        .then(pl.lit("item_7_mda"))
        .when(raw.is_in(["1a", "item_1a", "item_1a_risk_factors"]))
        .then(pl.lit("item_1a_risk_factors"))
        .when(raw.is_in(["1", "item_1", "item_1_business"]))
        .then(pl.lit("item_1_business"))
        .when(raw.is_in(["items_1_1a_7_concat", "item_1_item_1a_item_7_concat"]))
        .then(pl.lit("items_1_1a_7_concat"))
        .otherwise(raw)
    )


def build_lm2011_extension_control_ladder() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "control_set_id": control_set.control_set_id,
                "spec_alias": control_set.spec_alias,
                "control_columns": list(control_set.controls),
                "sample_rule": control_set.sample_rule,
                "common_support_ownership": control_set.common_support_ownership,
                "includes_ownership_control": control_set.includes_ownership_control,
            }
            for control_set in _CONTROL_SETS
        ],
        schema_overrides={
            "control_set_id": pl.Utf8,
            "spec_alias": pl.Utf8,
            "control_columns": pl.List(pl.Utf8),
            "sample_rule": pl.Utf8,
            "common_support_ownership": pl.Boolean,
            "includes_ownership_control": pl.Boolean,
        },
    )


def build_lm2011_extension_specification_grid() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "specification_name": comparison_spec.specification_name,
                "feature_family": comparison_spec.feature_family,
                "signal_inputs": list(comparison_spec.signal_inputs),
            }
            for comparison_spec in _COMPARISON_SPECS
        ],
        schema_overrides={
            "specification_name": pl.Utf8,
            "feature_family": pl.Utf8,
            "signal_inputs": pl.List(pl.Utf8),
        },
    )


def build_lm2011_extension_dictionary_features(
    sec_item_lf: pl.LazyFrame,
    *,
    dictionary_lists: Mapping[str, Iterable[str]],
    harvard_negative_word_list: Iterable[str] | None,
    master_dictionary_words: Iterable[str],
    text_scope_item_ids: Mapping[str, str] = EXTENSION_ITEM_SCOPE_IDS,
    dictionary_family: str = EXTENSION_DICTIONARY_FAMILY_LM2011,
    text_col: str = "full_text",
    raw_form_col: str = "document_type_filename",
) -> pl.LazyFrame:
    frames: list[pl.LazyFrame] = []
    for raw_text_scope, item_id in text_scope_item_ids.items():
        text_scope = _normalize_scope_value(raw_text_scope)
        scored_lf = build_lm2011_text_features_mda(
            sec_item_lf,
            dictionary_lists=dictionary_lists,
            harvard_negative_word_list=harvard_negative_word_list,
            master_dictionary_words=master_dictionary_words,
            text_col=text_col,
            raw_form_col=raw_form_col,
            required_item_id=item_id,
        )
        frames.append(
            scored_lf.with_columns(
                pl.lit(text_scope, dtype=pl.Utf8).alias("text_scope"),
                pl.lit(dictionary_family, dtype=pl.Utf8).alias("dictionary_family"),
                pl.col("total_token_count_mda").cast(pl.Int32, strict=False).alias("total_token_count"),
                pl.col("token_count_mda").cast(pl.Int32, strict=False).alias("token_count"),
            ).select(_empty_extension_dictionary_features_df().columns)
        )
    if not frames:
        return _empty_extension_dictionary_features_df().lazy()
    return pl.concat(frames, how="vertical_relaxed")


def _scope_expr_from_schema(
    schema: pl.Schema,
    *,
    default_text_scope: str | None,
    label: str,
) -> pl.Expr:
    if "text_scope" in schema:
        return normalize_lm2011_extension_text_scope_expr(pl.col("text_scope"))
    if "benchmark_item_code" in schema:
        return normalize_lm2011_extension_text_scope_expr(pl.col("benchmark_item_code"))
    if "item_id" in schema:
        return normalize_lm2011_extension_text_scope_expr(pl.col("item_id"))
    if default_text_scope is not None:
        return pl.lit(_normalize_scope_value(default_text_scope), dtype=pl.Utf8)
    raise ValueError(f"{label} missing text_scope, benchmark_item_code, or item_id")


def _coalesce_existing_float(schema: pl.Schema, candidates: Sequence[str]) -> pl.Expr:
    exprs = [
        pl.col(column).cast(pl.Float64, strict=False)
        for column in candidates
        if column in schema
    ]
    return pl.coalesce(exprs) if exprs else pl.lit(None, dtype=pl.Float64)


def _coalesce_existing_int(schema: pl.Schema, candidates: Sequence[str], dtype: pl.DataType) -> pl.Expr:
    exprs = [
        pl.col(column).cast(dtype, strict=False)
        for column in candidates
        if column in schema
    ]
    return pl.coalesce(exprs) if exprs else pl.lit(None, dtype=dtype)


def _select_extension_dictionary_surface(
    dictionary_features_lf: pl.LazyFrame | None,
    *,
    default_text_scope: str | None,
    default_dictionary_family: str,
) -> pl.LazyFrame:
    if dictionary_features_lf is None:
        return _empty_extension_dictionary_features_df().lazy()
    _require_columns(dictionary_features_lf, ("doc_id",), "extension_dictionary_features")
    schema = dictionary_features_lf.collect_schema()
    scope_expr = _scope_expr_from_schema(
        schema,
        default_text_scope=default_text_scope,
        label="extension_dictionary_features",
    )
    family_expr = (
        pl.col("dictionary_family").cast(pl.Utf8, strict=False)
        if "dictionary_family" in schema
        else pl.lit(default_dictionary_family, dtype=pl.Utf8)
    )
    selected = dictionary_features_lf.select(
        pl.col("doc_id").cast(pl.Utf8, strict=False),
        (
            pl.col("cik_10").cast(pl.Utf8, strict=False)
            if "cik_10" in schema
            else pl.lit(None, dtype=pl.Utf8)
        ).alias("cik_10"),
        (
            pl.col("filing_date").cast(pl.Date, strict=False)
            if "filing_date" in schema
            else pl.lit(None, dtype=pl.Date)
        ).alias("filing_date"),
        scope_expr.alias("text_scope"),
        family_expr.alias("dictionary_family"),
        _coalesce_existing_int(
            schema,
            ("total_token_count", "total_token_count_mda", "total_token_count_full_10k"),
            pl.Int32,
        ).alias("total_token_count"),
        _coalesce_existing_int(
            schema,
            ("token_count", "token_count_mda", "token_count_full_10k"),
            pl.Int32,
        ).alias("token_count"),
        *[
            (
                pl.col(column).cast(pl.Float64, strict=False)
                if column in schema
                else pl.lit(None, dtype=pl.Float64)
            ).alias(column)
            for column in EXTENSION_SIGNAL_COLUMNS
        ],
    )
    return selected.unique(subset=["doc_id", "text_scope", "dictionary_family"], keep="first")


def _select_extension_model_surface(model_features_lf: pl.LazyFrame | None) -> pl.LazyFrame:
    if model_features_lf is None:
        return _empty_extension_model_features_df().lazy()
    _require_columns(model_features_lf, ("doc_id",), "extension_model_features")
    schema = model_features_lf.collect_schema()
    scope_expr = _scope_expr_from_schema(
        schema,
        default_text_scope=None,
        label="extension_model_features",
    )
    selected = model_features_lf.select(
        pl.col("doc_id").cast(pl.Utf8, strict=False),
        scope_expr.alias("text_scope"),
        (
            pl.col("model_name").cast(pl.Utf8, strict=False)
            if "model_name" in schema
            else pl.lit(None, dtype=pl.Utf8)
        ).alias("model_name"),
        (
            pl.col("model_version").cast(pl.Utf8, strict=False)
            if "model_version" in schema
            else pl.lit(None, dtype=pl.Utf8)
        ).alias("model_version"),
        (
            pl.col("segment_policy_id").cast(pl.Utf8, strict=False)
            if "segment_policy_id" in schema
            else pl.lit(None, dtype=pl.Utf8)
        ).alias("segment_policy_id"),
        _coalesce_existing_int(schema, ("finbert_segment_count", "sentence_count"), pl.Int32).alias(
            "finbert_segment_count"
        ),
        _coalesce_existing_int(
            schema,
            ("finbert_token_count_512", "finbert_token_count_512_sum"),
            pl.Int64,
        ).alias("finbert_token_count_512"),
        _coalesce_existing_int(
            schema,
            ("finbert_token_count_512_sum", "finbert_token_count_512"),
            pl.Int64,
        ).alias("finbert_token_count_512_sum"),
        *[
            _coalesce_existing_float(schema, (column,)).alias(column)
            for column in (
                "finbert_neg_prob_lenw_mean",
                "finbert_pos_prob_lenw_mean",
                "finbert_neu_prob_lenw_mean",
                "finbert_net_negative_lenw_mean",
                "finbert_neg_dominant_share",
            )
        ],
    )
    return selected.unique(subset=["doc_id", "text_scope"], keep="first")


def _event_float_expr(schema: pl.Schema, column: str, *, required: bool = False) -> pl.Expr:
    if column in schema:
        return pl.col(column).cast(pl.Float64, strict=False)
    if required:
        raise ValueError(f"event_panel missing required column: {column}")
    return pl.lit(None, dtype=pl.Float64)


def _build_extension_event_base(
    event_panel_lf: pl.LazyFrame,
    *,
    sample_start: dt.date,
    sample_end: dt.date,
    sample_window: str,
) -> pl.LazyFrame:
    _require_columns(
        event_panel_lf,
        (
            "doc_id",
            "gvkey_int",
            "KYPERMNO",
            "filing_date",
            "size_event",
            "bm_event",
            "share_turnover",
            "pre_ffalpha",
            "nasdaq_dummy",
            EXTENSION_PRIMARY_OUTCOME,
        ),
        "extension_event_panel",
    )
    schema = event_panel_lf.collect_schema()
    ownership_expr = _coalesce_existing_float(
        schema,
        ("institutional_ownership_proxy_refinitiv", "institutional_ownership_pct", "institutional_ownership"),
    )
    return _apply_lm2011_regression_transforms(
        event_panel_lf.select(
            pl.col("doc_id").cast(pl.Utf8, strict=False),
            pl.col("gvkey_int").cast(pl.Int32, strict=False),
            pl.col("KYPERMNO").cast(pl.Int32, strict=False),
            pl.col("filing_date").cast(pl.Date, strict=False),
            _event_float_expr(schema, "size_event", required=True).alias("size_event"),
            _event_float_expr(schema, "bm_event", required=True).alias("bm_event"),
            _event_float_expr(schema, "share_turnover", required=True).alias("share_turnover"),
            _event_float_expr(schema, "pre_ffalpha", required=True).alias("pre_ffalpha"),
            pl.col("nasdaq_dummy").cast(pl.Int8, strict=False).alias("nasdaq_dummy"),
            ownership_expr.alias(EXTENSION_OWNERSHIP_CONTROL),
            _event_float_expr(schema, EXTENSION_PRIMARY_OUTCOME, required=True).alias(EXTENSION_PRIMARY_OUTCOME),
            _event_float_expr(schema, "abnormal_volume").alias("abnormal_volume"),
            _event_float_expr(schema, "postevent_return_volatility").alias("postevent_return_volatility"),
            _event_float_expr(schema, "sue").alias("sue"),
        )
        .filter(
            pl.col("filing_date").is_between(
                pl.lit(sample_start),
                pl.lit(sample_end),
                closed="both",
            )
        )
        .with_columns(
            pl.lit(sample_window, dtype=pl.Utf8).alias("sample_window"),
            pl.col(EXTENSION_OWNERSHIP_CONTROL).is_not_null().alias("ownership_proxy_available"),
            pl.col(EXTENSION_OWNERSHIP_CONTROL).is_not_null().alias("common_support_flag_ownership"),
        )
    )


def build_lm2011_extension_analysis_panel(
    event_panel_lf: pl.LazyFrame,
    dictionary_features_lf: pl.LazyFrame | None,
    model_features_lf: pl.LazyFrame | None,
    company_history_lf: pl.LazyFrame,
    company_description_lf: pl.LazyFrame,
    *,
    ff48_siccodes_path: Path | str,
    dictionary_text_scope: str | None = None,
    dictionary_family: str = EXTENSION_DICTIONARY_FAMILY_LM2011,
    sample_start: dt.date = EXTENSION_SAMPLE_START,
    sample_end: dt.date = EXTENSION_SAMPLE_END,
    sample_window: str = EXTENSION_SAMPLE_WINDOW,
) -> pl.LazyFrame:
    dictionary_surface_lf = _select_extension_dictionary_surface(
        dictionary_features_lf,
        default_text_scope=dictionary_text_scope,
        default_dictionary_family=dictionary_family,
    )
    model_surface_lf = _select_extension_model_surface(model_features_lf)

    key_frames: list[pl.LazyFrame] = []
    if dictionary_features_lf is not None:
        key_frames.append(dictionary_surface_lf.select("doc_id", "text_scope"))
    if model_features_lf is not None:
        key_frames.append(model_surface_lf.select("doc_id", "text_scope"))
    if not key_frames:
        raise ValueError("At least one of dictionary_features_lf or model_features_lf is required")
    keys_lf = pl.concat(key_frames, how="vertical_relaxed").unique(subset=["doc_id", "text_scope"])

    event_base_lf = _build_extension_event_base(
        event_panel_lf,
        sample_start=sample_start,
        sample_end=sample_end,
        sample_window=sample_window,
    )
    industries_lf = _attach_ff48_industries(
        event_panel_lf,
        company_history_lf,
        company_description_lf,
        ff48_siccodes_path=ff48_siccodes_path,
    )
    panel_lf = (
        keys_lf.join(event_base_lf, on="doc_id", how="inner")
        .join(
            dictionary_surface_lf.drop("filing_date", "cik_10", strict=False),
            on=["doc_id", "text_scope"],
            how="left",
        )
        .join(model_surface_lf, on=["doc_id", "text_scope"], how="left")
        .join(industries_lf, on="doc_id", how="left")
        .unique(subset=["doc_id", "text_scope"], keep="first")
    )
    return _nullify_infinite_float_columns(panel_lf)


def apply_lm2011_extension_control_set(
    panel_lf: pl.LazyFrame,
    control_set_id: str,
) -> pl.LazyFrame:
    control_set = _control_set_by_id(control_set_id)
    _require_columns(
        panel_lf,
        ("doc_id", "text_scope", "common_support_flag_ownership", *control_set.controls),
        "extension_analysis_panel",
    )
    if not control_set.common_support_ownership:
        return panel_lf
    return panel_lf.filter(pl.col("common_support_flag_ownership").fill_null(False))


def _all_not_null_expr(columns: Sequence[str]) -> pl.Expr:
    if not columns:
        return pl.lit(True)
    return pl.fold(
        acc=pl.lit(True),
        function=lambda acc, expr: acc & expr,
        exprs=[pl.col(column).is_not_null() for column in columns],
    )


def build_lm2011_extension_sample_loss_table(
    panel_lf: pl.LazyFrame,
    *,
    outcome_names: Sequence[str] = (EXTENSION_PRIMARY_OUTCOME,),
    control_set_ids: Sequence[str] = ("C0", "C1", "C2"),
    specification_names: Sequence[str] = ("dictionary_only", "finbert_only", "dictionary_finbert_joint"),
    filing_date_col: str = "filing_date",
    industry_col: str = "ff48_industry_id",
) -> pl.DataFrame:
    rows: list[pl.DataFrame] = []
    for outcome_name in outcome_names:
        for specification_name in specification_names:
            comparison_spec = _comparison_spec_by_name(specification_name)
            for control_set_id in control_set_ids:
                control_set = _control_set_by_id(control_set_id)
                required = (
                    "sample_window",
                    "text_scope",
                    filing_date_col,
                    outcome_name,
                    industry_col,
                    *comparison_spec.signal_inputs,
                    *control_set.controls,
                )
                _require_columns(panel_lf, required, "extension_analysis_panel")
                scoped_lf = apply_lm2011_extension_control_set(panel_lf, control_set_id).with_columns(
                    pl.col(filing_date_col).cast(pl.Date, strict=False).dt.year().cast(pl.Int32).alias("calendar_year"),
                    _all_not_null_expr(comparison_spec.signal_inputs).alias("_signal_available"),
                    _all_not_null_expr(control_set.controls).alias("_controls_available"),
                    pl.col(outcome_name).is_not_null().alias("_outcome_available"),
                    pl.col(industry_col).is_not_null().alias("_industry_available"),
                )
                row_df = (
                    scoped_lf.group_by("sample_window", "calendar_year", "text_scope")
                    .agg(
                        pl.len().cast(pl.Int64).alias("n_control_set_rows"),
                        pl.col("_outcome_available").cast(pl.Int64).sum().alias("n_outcome_available"),
                        pl.col("_signal_available").cast(pl.Int64).sum().alias("n_signal_available"),
                        pl.col("_controls_available").cast(pl.Int64).sum().alias("n_controls_available"),
                        pl.col("_industry_available").cast(pl.Int64).sum().alias("n_industry_available"),
                        (
                            pl.col("_outcome_available")
                            & pl.col("_signal_available")
                            & pl.col("_controls_available")
                            & pl.col("_industry_available")
                        )
                        .cast(pl.Int64)
                        .sum()
                        .alias("n_estimation_rows"),
                    )
                    .with_columns(
                        pl.lit(outcome_name, dtype=pl.Utf8).alias("outcome_name"),
                        pl.lit(comparison_spec.feature_family, dtype=pl.Utf8).alias("feature_family"),
                        pl.lit(control_set.control_set_id, dtype=pl.Utf8).alias("control_set_id"),
                        pl.lit(control_set.spec_alias, dtype=pl.Utf8).alias("control_set_alias"),
                        pl.lit(specification_name, dtype=pl.Utf8).alias("specification_name"),
                        (pl.col("n_control_set_rows") - pl.col("n_outcome_available")).alias("n_missing_outcome"),
                        (pl.col("n_control_set_rows") - pl.col("n_signal_available")).alias("n_missing_signal"),
                        (pl.col("n_control_set_rows") - pl.col("n_controls_available")).alias("n_missing_controls"),
                        (pl.col("n_control_set_rows") - pl.col("n_industry_available")).alias("n_missing_industry"),
                    )
                    .select(
                        "sample_window",
                        "calendar_year",
                        "text_scope",
                        "outcome_name",
                        "feature_family",
                        "control_set_id",
                        "control_set_alias",
                        "specification_name",
                        "n_control_set_rows",
                        "n_outcome_available",
                        "n_signal_available",
                        "n_controls_available",
                        "n_industry_available",
                        "n_estimation_rows",
                        "n_missing_outcome",
                        "n_missing_signal",
                        "n_missing_controls",
                        "n_missing_industry",
                    )
                    .collect()
                )
                rows.append(row_df)
    if not rows:
        return pl.DataFrame(
            schema={
                "sample_window": pl.Utf8,
                "calendar_year": pl.Int32,
                "text_scope": pl.Utf8,
                "outcome_name": pl.Utf8,
                "feature_family": pl.Utf8,
                "control_set_id": pl.Utf8,
                "control_set_alias": pl.Utf8,
                "specification_name": pl.Utf8,
                "n_control_set_rows": pl.Int64,
                "n_outcome_available": pl.Int64,
                "n_signal_available": pl.Int64,
                "n_controls_available": pl.Int64,
                "n_industry_available": pl.Int64,
                "n_estimation_rows": pl.Int64,
                "n_missing_outcome": pl.Int64,
                "n_missing_signal": pl.Int64,
                "n_missing_controls": pl.Int64,
                "n_missing_industry": pl.Int64,
            }
        )
    return pl.concat(rows, how="vertical_relaxed").sort(
        "sample_window",
        "calendar_year",
        "text_scope",
        "outcome_name",
        "specification_name",
        "control_set_id",
    )


def _normal_approx_two_sided_p_value(t_stat: object) -> float | None:
    if t_stat is None:
        return None
    value = float(t_stat)
    if not math.isfinite(value):
        return None
    return math.erfc(abs(value) / math.sqrt(2.0))


def _extension_result_status_row(
    *,
    run_id: str,
    sample_window: str,
    text_scope: str,
    outcome_name: str,
    comparison_spec: Lm2011ExtensionComparisonSpec,
    control_set: Lm2011ExtensionControlSet,
    estimator_status: str,
    failure_reason: str | None,
    nw_lags: int,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "sample_window": sample_window,
        "text_scope": text_scope,
        "outcome_name": outcome_name,
        "feature_family": comparison_spec.feature_family,
        "control_set_id": control_set.control_set_id,
        "control_set_alias": control_set.spec_alias,
        "specification_name": comparison_spec.specification_name,
        "coefficient_name": None,
        "signal_name": ",".join(comparison_spec.signal_inputs),
        "estimate": None,
        "standard_error": None,
        "t_stat": None,
        "p_value": None,
        "n_obs": None,
        "n_quarters": None,
        "mean_quarter_n": None,
        "average_r2": None,
        "weighting_rule": None,
        "nw_lags": nw_lags,
        "estimator_status": estimator_status,
        "failure_reason": failure_reason,
    }


def _convert_lm2011_table_results_to_extension_rows(
    result_df: pl.DataFrame,
    *,
    run_id: str,
    sample_window: str,
    outcome_name: str,
    comparison_spec: Lm2011ExtensionComparisonSpec,
    control_set: Lm2011ExtensionControlSet,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in result_df.iter_rows(named=True):
        n_quarters = row.get("n_quarters")
        mean_quarter_n = row.get("mean_quarter_n")
        n_obs = None
        if n_quarters is not None and mean_quarter_n is not None:
            n_obs = int(round(float(n_quarters) * float(mean_quarter_n)))
        t_stat = row.get("t_stat")
        rows.append(
            {
                "run_id": run_id,
                "sample_window": sample_window,
                "text_scope": row.get("text_scope"),
                "outcome_name": outcome_name,
                "feature_family": comparison_spec.feature_family,
                "control_set_id": control_set.control_set_id,
                "control_set_alias": control_set.spec_alias,
                "specification_name": comparison_spec.specification_name,
                "coefficient_name": row.get("coefficient_name"),
                "signal_name": ",".join(comparison_spec.signal_inputs),
                "estimate": row.get("estimate"),
                "standard_error": row.get("standard_error"),
                "t_stat": t_stat,
                "p_value": _normal_approx_two_sided_p_value(t_stat),
                "n_obs": n_obs,
                "n_quarters": n_quarters,
                "mean_quarter_n": mean_quarter_n,
                "average_r2": None,
                "weighting_rule": row.get("weighting_rule"),
                "nw_lags": row.get("nw_lags"),
                "estimator_status": "estimated",
                "failure_reason": None,
            }
        )
    return rows


def run_lm2011_extension_estimation_scaffold(
    panel_lf: pl.LazyFrame,
    *,
    run_id: str = "lm2011_extension_scaffold",
    sample_window: str = EXTENSION_SAMPLE_WINDOW,
    text_scopes: Sequence[str] = EXTENSION_PRIMARY_TEXT_SCOPES,
    outcome_names: Sequence[str] = (EXTENSION_PRIMARY_OUTCOME,),
    control_set_ids: Sequence[str] = ("C0", "C1", "C2"),
    specification_names: Sequence[str] = ("dictionary_only", "finbert_only", "dictionary_finbert_joint"),
    nw_lags: int = 1,
) -> pl.DataFrame:
    output_rows: list[dict[str, object]] = []
    for text_scope in text_scopes:
        normalized_text_scope = _normalize_scope_value(text_scope)
        scope_panel_lf = panel_lf.filter(pl.col("text_scope") == pl.lit(normalized_text_scope))
        for outcome_name in outcome_names:
            for specification_name in specification_names:
                comparison_spec = _comparison_spec_by_name(specification_name)
                for control_set_id in control_set_ids:
                    control_set = _control_set_by_id(control_set_id)
                    spec_panel_lf = apply_lm2011_extension_control_set(scope_panel_lf, control_set.control_set_id)
                    signal_column = comparison_spec.signal_inputs[0]
                    control_columns = (*comparison_spec.signal_inputs[1:], *control_set.controls)
                    try:
                        result_df = run_lm2011_quarterly_fama_macbeth(
                            spec_panel_lf,
                            table_id="lm2011_extension_results",
                            text_scope=normalized_text_scope,
                            dependent_variable=outcome_name,
                            signal_column=signal_column,
                            control_columns=control_columns,
                            specification_id=(
                                f"{comparison_spec.specification_name}:{control_set.control_set_id}:{outcome_name}"
                            ),
                            nw_lags=nw_lags,
                        )
                    except ValueError as exc:
                        output_rows.append(
                            _extension_result_status_row(
                                run_id=run_id,
                                sample_window=sample_window,
                                text_scope=normalized_text_scope,
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
                                sample_window=sample_window,
                                text_scope=normalized_text_scope,
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
                            sample_window=sample_window,
                            outcome_name=outcome_name,
                            comparison_spec=comparison_spec,
                            control_set=control_set,
                        )
                    )
    if not output_rows:
        return _empty_extension_results_df()
    return pl.DataFrame(output_rows, schema_overrides=_EXTENSION_RESULTS_SCHEMA).select(
        _empty_extension_results_df().columns
    )


__all__ = [
    "EXTENSION_DICTIONARY_FAMILY_LM2011",
    "EXTENSION_FINBERT_MODEL_FAMILY",
    "EXTENSION_ITEM_SCOPE_IDS",
    "EXTENSION_JOINT_FEATURE_FAMILY",
    "EXTENSION_PRIMARY_OUTCOME",
    "EXTENSION_PRIMARY_TEXT_SCOPES",
    "EXTENSION_SAMPLE_END",
    "EXTENSION_SAMPLE_START",
    "EXTENSION_SAMPLE_WINDOW",
    "EXTENSION_SECONDARY_OUTCOMES",
    "Lm2011ExtensionComparisonSpec",
    "Lm2011ExtensionControlSet",
    "apply_lm2011_extension_control_set",
    "build_lm2011_extension_analysis_panel",
    "build_lm2011_extension_control_ladder",
    "build_lm2011_extension_dictionary_features",
    "build_lm2011_extension_sample_loss_table",
    "build_lm2011_extension_specification_grid",
    "normalize_lm2011_extension_text_scope_expr",
    "run_lm2011_extension_estimation_scaffold",
]

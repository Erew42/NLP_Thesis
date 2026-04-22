from __future__ import annotations

from typing import Sequence

import polars as pl

from thesis_assets.errors import SampleContractError


def raw_available(
    lf: pl.LazyFrame,
    *,
    filters: Sequence[pl.Expr] = (),
) -> pl.LazyFrame:
    return _apply_filters(lf, filters)


def regression_eligible(
    lf: pl.LazyFrame,
    *,
    required_columns: Sequence[str],
    filters: Sequence[pl.Expr] = (),
) -> pl.LazyFrame:
    prepared = _apply_filters(lf, filters)
    _require_columns(prepared, required_columns, "regression_eligible")
    return prepared.drop_nulls(subset=list(required_columns))


def common_row_comparison(
    left_lf: pl.LazyFrame,
    right_lf: pl.LazyFrame,
    *,
    join_keys: Sequence[str],
    left_signal_columns: Sequence[str],
    right_signal_columns: Sequence[str],
    left_filters: Sequence[pl.Expr] = (),
    right_filters: Sequence[pl.Expr] = (),
    suffix: str = "_right",
) -> pl.LazyFrame:
    left_prepared = _apply_filters(left_lf, left_filters)
    right_prepared = _apply_filters(right_lf, right_filters)

    _require_columns(left_prepared, [*join_keys, *left_signal_columns], "common_row_comparison left surface")
    _require_columns(right_prepared, [*join_keys, *right_signal_columns], "common_row_comparison right surface")
    _validate_unique_keys(left_prepared, join_keys, "common_row_comparison left surface")
    _validate_unique_keys(right_prepared, join_keys, "common_row_comparison right surface")

    left_clean = left_prepared.drop_nulls(subset=[*join_keys, *left_signal_columns])
    right_clean = right_prepared.drop_nulls(subset=[*join_keys, *right_signal_columns])
    return left_clean.join(right_clean, on=list(join_keys), how="inner", suffix=suffix)


def common_success_comparison(
    lf: pl.LazyFrame,
    *,
    expected_policy: str,
    filters: Sequence[pl.Expr] = (),
    policy_column: str = "common_success_policy",
) -> pl.LazyFrame:
    prepared = _apply_filters(lf, filters)
    _require_columns(prepared, (policy_column,), "common_success_comparison")
    policy_values = (
        prepared.select(pl.col(policy_column).drop_nulls().unique().sort())
        .collect()
        .get_column(policy_column)
        .to_list()
    )
    row_count = int(prepared.select(pl.len()).collect().item())
    if row_count > 0 and not policy_values:
        raise SampleContractError(
            f"Expected non-null values in {policy_column!r} for common_success_comparison, but all rows were null."
        )
    unexpected = sorted({value for value in policy_values if value != expected_policy})
    if unexpected:
        raise SampleContractError(
            f"Expected common-success policy {expected_policy!r}, found unexpected values: {unexpected}"
        )
    return prepared


def ownership_common_support(
    lf: pl.LazyFrame,
    *,
    ownership_flag_column: str,
    filters: Sequence[pl.Expr] = (),
) -> pl.LazyFrame:
    prepared = _apply_filters(lf, filters)
    _require_columns(prepared, (ownership_flag_column,), "ownership_common_support")
    return prepared.filter(pl.col(ownership_flag_column).fill_null(False))


def _apply_filters(lf: pl.LazyFrame, filters: Sequence[pl.Expr]) -> pl.LazyFrame:
    out = lf
    for predicate in filters:
        out = out.filter(predicate)
    return out


def _require_columns(lf: pl.LazyFrame, columns: Sequence[str], label: str) -> None:
    schema = lf.collect_schema()
    missing = [column for column in columns if column not in schema]
    if missing:
        raise SampleContractError(f"{label} is missing required columns: {missing}")


def _validate_unique_keys(lf: pl.LazyFrame, join_keys: Sequence[str], label: str) -> None:
    duplicate_probe = (
        lf.group_by(list(join_keys))
        .len()
        .filter(pl.col("len") > 1)
        .limit(1)
        .collect()
    )
    if duplicate_probe.height:
        raise SampleContractError(f"{label} contains duplicate join keys for {tuple(join_keys)!r}.")

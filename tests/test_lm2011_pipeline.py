from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import pytest
import yaml

from thesis_pkg.core.sec import lm2011_text as lm2011_text_module
from thesis_pkg.pipeline import (
    build_annual_accounting_panel,
    build_lm2011_event_panel,
    build_lm2011_normalized_filing_feeds,
    build_lm2011_sample_backbone,
    build_lm2011_sue_panel,
    build_lm2011_table_i_sample_creation,
    build_lm2011_text_features_full_10k,
    build_lm2011_text_features_mda,
    build_lm2011_trading_strategy_ff4_summary,
    build_lm2011_trading_strategy_monthly_returns,
    build_lm2011_trading_strategy_monthly_returns_from_text_features,
    tokenize_lm2011_text,
)
from thesis_pkg.core.sec.lm2011_text import (
    write_lm2011_text_features_full_10k_parquet,
    write_lm2011_text_features_mda_parquet,
)
from thesis_pkg.core.ccm.lm2011 import _build_lm2011_sample_backbone_stage_frames
from thesis_pkg.pipelines import lm2011_pipeline
from thesis_pkg.pipelines.lm2011_pipeline import (
    _apply_lm2011_regression_transforms,
    _build_lm2011_event_screen_surface_batched,
    _build_lm2011_table_i_market_stage_frames,
    _ensure_factor_scale,
    _ols_alpha_and_rmse,
    _ols_coefficients_and_r2,
    write_lm2011_event_screen_surface_parquet,
    write_lm2011_sue_panel_parquet,
)


SPEC_PATH = Path("replication_plan/LM2011/lm2011_replication_spec.yaml")


def _daily_window_frame(
    *,
    permno: int,
    start: dt.date,
    n_days: int,
    event_day_index: int,
    exchcd: int = 3,
    shrcd: int = 10,
    prc: float = 10.0,
    include_final_prc: bool = True,
    constant_pre_event_volume: bool = False,
) -> pl.DataFrame:
    dates = [start + dt.timedelta(days=offset) for offset in range(n_days)]
    rows = []
    for idx, caldt in enumerate(dates):
        relative_day = idx - event_day_index
        if constant_pre_event_volume and -65 <= relative_day <= -6:
            vol = 100.0
        else:
            vol = 100.0 + float(idx % 5)
        final_ret = 0.01 if 0 <= relative_day <= 3 else 0.001
        row = {
            "KYPERMNO": permno,
            "CALDT": caldt,
            "FINAL_RET": final_ret,
            "RET": final_ret,
            "PRC": prc,
            "TCAP": abs(prc) * 10.0,
            "SHROUT": 10.0,
            "VOL": vol,
            "SHRCD": shrcd,
            "EXCHCD": exchcd,
        }
        if include_final_prc:
            row["FINAL_PRC"] = prc
        rows.append(row)
    return pl.DataFrame(rows)


def _ff_factors_for_dates(dates: list[dt.date]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "trading_date": dates,
            "mkt_rf": [0.0002 + 0.00001 * (idx % 11) for idx in range(len(dates))],
            "smb": [0.0001 + 0.00001 * (idx % 7) for idx in range(len(dates))],
            "hml": [0.00015 + 0.00001 * (idx % 5) for idx in range(len(dates))],
            "rf": [0.0001] * len(dates),
        }
    )


def _base_event_inputs(*, include_final_prc: bool = True, constant_pre_event_volume: bool = False) -> dict[str, pl.DataFrame]:
    filing_date = dt.date(2023, 9, 18)
    daily = _daily_window_frame(
        permno=1,
        start=dt.date(2023, 1, 1),
        n_days=520,
        event_day_index=260,
        include_final_prc=include_final_prc,
        constant_pre_event_volume=constant_pre_event_volume,
    )
    return {
        "sample_backbone": pl.DataFrame(
            {
                "doc_id": ["doc_ok"],
                "gvkey_int": [1000],
                "KYPERMNO": [1],
                "filing_date": [filing_date],
                "normalized_form": ["10-K"],
            }
        ),
        "daily": daily,
        "annual_panel": pl.DataFrame(
            {
                "gvkey_int": [1000],
                "accounting_period_end": [dt.date(2022, 12, 31)],
                "book_equity_be": [50.0],
                "AT": [80.0],
            }
        ),
        "ownership": pl.DataFrame({"doc_id": ["doc_ok"], "institutional_ownership_pct": [44.0]}),
        "text_features": pl.DataFrame(
            {
                "doc_id": ["doc_ok"],
                "token_count_full_10k": [2500],
                "total_token_count_full_10k": [2500],
            }
        ),
        "ff": _ff_factors_for_dates(daily.get_column("CALDT").to_list()),
    }


def _mutate_relative_day_rows(
    daily: pl.DataFrame,
    *,
    event_day_index: int,
    relative_days: set[int],
    updates: dict[str, object],
) -> pl.DataFrame:
    out = daily.with_row_index("_idx", offset=0).with_columns(
        (pl.col("_idx").cast(pl.Int64, strict=False) - event_day_index).alias("_relative_day")
    )
    for column, value in updates.items():
        out = out.with_columns(
            pl.when(pl.col("_relative_day").is_in(sorted(relative_days)))
            .then(pl.lit(value))
            .otherwise(pl.col(column))
            .alias(column)
        )
    return out.drop("_idx", "_relative_day")


def _table_i_market_inputs() -> dict[str, pl.DataFrame]:
    doc_specs = [
        ("doc_fail_shrcd", 1, 1001),
        ("doc_fail_market_cap", 2, 1002),
        ("doc_fail_price", 3, 1003),
        ("doc_fail_event_window", 4, 1004),
        ("doc_fail_exchange", 5, 1005),
        ("doc_fail_coverage", 6, 1006),
        ("doc_fail_book_to_market", 7, 1007),
        ("doc_fail_token_count", 8, 1008),
        ("doc_keep", 9, 1009),
    ]
    daily_frames: list[pl.DataFrame] = []
    for doc_id, permno, _gvkey in doc_specs:
        daily = _daily_window_frame(
            permno=permno,
            start=dt.date(1997, 1, 1),
            n_days=520,
            event_day_index=260,
        )
        if doc_id == "doc_fail_shrcd":
            daily = daily.with_columns(pl.lit(12).alias("SHRCD"))
        elif doc_id == "doc_fail_market_cap":
            daily = _mutate_relative_day_rows(
                daily,
                event_day_index=260,
                relative_days={-1},
                updates={"TCAP": None, "FINAL_PRC": None, "PRC": None},
            )
        elif doc_id == "doc_fail_price":
            daily = _mutate_relative_day_rows(
                daily,
                event_day_index=260,
                relative_days={-1},
                updates={"TCAP": 25.0, "FINAL_PRC": 2.5, "PRC": 2.5},
            )
        elif doc_id == "doc_fail_event_window":
            daily = _mutate_relative_day_rows(
                daily,
                event_day_index=260,
                relative_days={2},
                updates={"VOL": None},
            )
        elif doc_id == "doc_fail_exchange":
            daily = daily.with_columns(pl.lit(4).alias("EXCHCD"))
        elif doc_id == "doc_fail_coverage":
            daily = _mutate_relative_day_rows(
                daily,
                event_day_index=260,
                relative_days=set(range(-252, -5)),
                updates={"VOL": None, "FINAL_RET": None, "RET": None},
            )
        daily_frames.append(daily)
    filing_date = daily_frames[0].item(260, "CALDT")
    return {
        "sample_backbone": pl.DataFrame(
            {
                "doc_id": [doc_id for doc_id, _, _ in doc_specs],
                "cik_10": [f"{idx + 1:010d}" for idx in range(len(doc_specs))],
                "gvkey_int": [gvkey for _, _, gvkey in doc_specs],
                "KYPERMNO": [permno for _, permno, _ in doc_specs],
                "filing_date": [filing_date] * len(doc_specs),
                "normalized_form": ["10-K"] * len(doc_specs),
            }
        ),
        "daily": pl.concat(daily_frames, how="vertical_relaxed"),
        "annual_panel": pl.DataFrame(
            {
                "gvkey_int": [gvkey for _, _, gvkey in doc_specs],
                "accounting_period_end": [dt.date(1996, 12, 31)] * len(doc_specs),
                "book_equity_be": [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 0.0, 50.0, 50.0],
                "AT": [80.0] * len(doc_specs),
            }
        ),
        "text_features": pl.DataFrame(
            {
                "doc_id": [doc_id for doc_id, _, _ in doc_specs],
                "token_count_full_10k": [2500, 2500, 2500, 2500, 2500, 2500, 2500, 1999, 2500],
                "total_token_count_full_10k": [2500, 2500, 2500, 2500, 2500, 2500, 2500, 1999, 2500],
            }
        ),
        "ff": _ff_factors_for_dates(daily_frames[0].get_column("CALDT").to_list()),
    }


def _lm_dictionary_lists() -> dict[str, list[str]]:
    return {
        "negative": ["loss"],
        "positive": ["gain"],
        "uncertainty": ["uncertain"],
        "litigious": ["lawsuit"],
        "modal_strong": ["must"],
        "modal_weak": ["may"],
    }


def _harvard_negative_word_list() -> list[str]:
    return ["bad"]


def _master_dictionary_words() -> list[str]:
    return [
        "loss",
        "gain",
        "uncertain",
        "lawsuit",
        "must",
        "may",
        "bad",
        "neutral",
        "safe",
        "recognized",
        "going-concern",
        "don't",
        "co-op",
        "alphabeta",
    ]


def _strategy_months() -> list[dt.date]:
    return [
        dt.date(1997, 7, 31),
        dt.date(1997, 8, 31),
        dt.date(1997, 9, 30),
        dt.date(1997, 10, 31),
        dt.date(1997, 11, 30),
        dt.date(1997, 12, 31),
        dt.date(1998, 1, 31),
        dt.date(1998, 2, 28),
        dt.date(1998, 3, 31),
        dt.date(1998, 4, 30),
        dt.date(1998, 5, 31),
        dt.date(1998, 6, 30),
    ]


def _build_strategy_inputs() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    event_panel = pl.DataFrame(
        {
            "doc_id": [f"doc_{idx}" for idx in range(10)],
            "KYPERMNO": [100 + idx for idx in range(10)],
            "filing_date": [dt.date(1996, 3, 1) + dt.timedelta(days=idx) for idx in range(10)],
        }
    )
    sec_parsed = pl.DataFrame(
        {
            "doc_id": [f"doc_{idx}" for idx in range(10)],
            "cik_10": [f"{idx:010d}" for idx in range(10)],
            "filing_date": [dt.date(1996, 3, 1) + dt.timedelta(days=idx) for idx in range(10)],
            "document_type_filename": ["10-K"] * 10,
            "full_text": [
                " ".join((["loss"] * (idx + 1)) + (["bad"] * max(0, 9 - idx)) + (["neutral"] * 20))
                for idx in range(10)
            ],
        }
    )
    monthly_rows = []
    stock_months = [dt.date(1997, 6, 30), *_strategy_months()]
    for permno_idx, permno in enumerate(range(100, 110)):
        for month_idx, month_end in enumerate(stock_months):
            monthly_rows.append(
                {
                    "KYPERMNO": permno,
                    "MCALDT": month_end,
                    "MRET": 0.01 + 0.001 * permno_idx + 0.0002 * month_idx,
                    "MTCAP": 100.0 + 10.0 * permno_idx + month_idx,
                }
            )
    monthly_stock = pl.DataFrame(monthly_rows)
    monthly_factors = pl.DataFrame(
        {
            "month_end": _strategy_months(),
            "mkt_rf": [0.01, 0.015, -0.01, 0.02, 0.005, -0.005, 0.012, 0.011, -0.006, 0.008, 0.007, 0.009],
            "smb": [0.002, -0.001, 0.001, 0.0, 0.003, -0.002, 0.002, -0.001, 0.001, 0.002, 0.0, -0.001],
            "hml": [0.001, 0.002, -0.002, 0.001, 0.0, -0.001, 0.001, 0.002, -0.001, 0.0, 0.001, -0.002],
            "rf": [0.0003] * 12,
            "mom": [0.004, 0.005, -0.003, 0.006, 0.002, -0.001, 0.003, 0.004, -0.002, 0.001, 0.002, 0.003],
        }
    )
    return event_panel, sec_parsed, monthly_stock, monthly_factors


def _assert_frames_equal_with_float_tolerance(
    left: pl.DataFrame,
    right: pl.DataFrame,
    *,
    sort_by: list[str],
) -> None:
    left_sorted = left.sort(sort_by)
    right_sorted = right.sort(sort_by)
    assert left_sorted.columns == right_sorted.columns
    assert left_sorted.height == right_sorted.height
    for column in left_sorted.columns:
        left_values = left_sorted.get_column(column).to_list()
        right_values = right_sorted.get_column(column).to_list()
        dtype = left_sorted.schema[column]
        if dtype in {pl.Float32, pl.Float64}:
            assert right_values == pytest.approx(left_values)
        else:
            assert right_values == left_values


def _build_sue_winsorization_inputs() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    row_count = 100
    row_ids = list(range(row_count))
    filing_date = dt.date(2023, 9, 18)
    pre_filing_trade_date = dt.date(2023, 9, 17)
    quarter_report_date = dt.date(2023, 10, 15)
    fiscal_period_end = dt.date(2023, 9, 30)
    doc_ids = [f"doc_{row_id:03d}" for row_id in row_ids]
    gvkeys = [1000 + row_id for row_id in row_ids]
    permnos = [row_id + 1 for row_id in row_ids]

    event_panel = pl.DataFrame(
        {
            "doc_id": pl.Series("doc_id", doc_ids, dtype=pl.Utf8),
            "gvkey_int": pl.Series("gvkey_int", gvkeys, dtype=pl.Int32),
            "KYPERMNO": pl.Series("KYPERMNO", permnos, dtype=pl.Int32),
            "filing_date": pl.Series("filing_date", [filing_date] * row_count, dtype=pl.Date),
            "pre_filing_trade_date": pl.Series(
                "pre_filing_trade_date",
                [pre_filing_trade_date] * row_count,
                dtype=pl.Date,
            ),
            "size_event": pl.Series("size_event", [100.0] * row_count, dtype=pl.Float64),
            "bm_event": pl.Series("bm_event", [1.5] * row_count, dtype=pl.Float64),
            "share_turnover": pl.Series("share_turnover", [10.0] * row_count, dtype=pl.Float64),
            "pre_ffalpha": pl.Series("pre_ffalpha", [0.02] * row_count, dtype=pl.Float64),
            "institutional_ownership": pl.Series(
                "institutional_ownership",
                [55.0] * row_count,
                dtype=pl.Float64,
            ),
            "nasdaq_dummy": pl.Series("nasdaq_dummy", [1] * row_count, dtype=pl.Int8),
        }
    )
    quarterly_panel = pl.DataFrame(
        {
            "gvkey_int": pl.Series("gvkey_int", gvkeys, dtype=pl.Int32),
            "quarter_report_date": pl.Series(
                "quarter_report_date",
                [quarter_report_date] * row_count,
                dtype=pl.Date,
            ),
            "APDEDATEQ": pl.Series("APDEDATEQ", [fiscal_period_end] * row_count, dtype=pl.Date),
            "PDATEQ": pl.Series("PDATEQ", [fiscal_period_end] * row_count, dtype=pl.Date),
        }
    )
    ibes = pl.DataFrame(
        {
            "gvkey_int": pl.Series("gvkey_int", gvkeys, dtype=pl.Int32),
            "announcement_date": pl.Series(
                "announcement_date",
                [quarter_report_date] * row_count,
                dtype=pl.Date,
            ),
            "fiscal_period_end": pl.Series(
                "fiscal_period_end",
                [fiscal_period_end] * row_count,
                dtype=pl.Date,
            ),
            "actual_eps": pl.Series("actual_eps", [float(row_id) for row_id in row_ids], dtype=pl.Float64),
            "forecast_consensus_mean": pl.Series(
                "forecast_consensus_mean",
                [0.0] * row_count,
                dtype=pl.Float64,
            ),
            "forecast_dispersion": pl.Series(
                "forecast_dispersion",
                [float(row_id * 10) for row_id in row_ids],
                dtype=pl.Float64,
            ),
            "forecast_revision_4m": pl.Series(
                "forecast_revision_4m",
                [float(row_id * 100) for row_id in row_ids],
                dtype=pl.Float64,
            ),
        }
    )
    daily_rows: list[dict[str, object]] = []
    for permno in permnos:
        daily_rows.extend(
            [
                {"KYPERMNO": permno, "CALDT": dt.date(2023, 8, 31), "PRC": 1.0},
                {"KYPERMNO": permno, "CALDT": pre_filing_trade_date, "PRC": 1.0},
            ]
        )
    daily = pl.DataFrame(
        daily_rows,
        schema_overrides={"KYPERMNO": pl.Int32, "CALDT": pl.Date, "PRC": pl.Float64},
    )
    return event_panel, quarterly_panel, ibes, daily


def _assert_sue_winsorization_surface(panel: pl.DataFrame, *, check_schema: bool = False) -> None:
    expected_columns = [
        "doc_id",
        "gvkey_int",
        "KYPERMNO",
        "filing_date",
        "quarter_report_date",
        "size_event",
        "bm_event",
        "share_turnover",
        "sue",
        "analyst_dispersion",
        "analyst_revisions",
        "pre_ffalpha",
        "institutional_ownership",
        "nasdaq_dummy",
    ]
    expected_schema = {
        "doc_id": pl.Utf8,
        "gvkey_int": pl.Int32,
        "KYPERMNO": pl.Int32,
        "filing_date": pl.Date,
        "quarter_report_date": pl.Date,
        "size_event": pl.Float64,
        "bm_event": pl.Float64,
        "share_turnover": pl.Float64,
        "sue": pl.Float64,
        "analyst_dispersion": pl.Float64,
        "analyst_revisions": pl.Float64,
        "pre_ffalpha": pl.Float64,
        "institutional_ownership": pl.Float64,
        "nasdaq_dummy": pl.Int8,
    }
    if check_schema:
        assert panel.columns == expected_columns
        assert panel.schema == expected_schema

    bounds = panel.select(
        pl.col("sue").min().alias("sue_min"),
        pl.col("sue").max().alias("sue_max"),
        pl.col("analyst_dispersion").min().alias("analyst_dispersion_min"),
        pl.col("analyst_dispersion").max().alias("analyst_dispersion_max"),
        pl.col("analyst_revisions").min().alias("analyst_revisions_min"),
        pl.col("analyst_revisions").max().alias("analyst_revisions_max"),
    ).row(0, named=True)
    assert bounds == pytest.approx(
        {
            "sue_min": 1.0,
            "sue_max": 98.0,
            "analyst_dispersion_min": 10.0,
            "analyst_dispersion_max": 980.0,
            "analyst_revisions_min": 100.0,
            "analyst_revisions_max": 9800.0,
        }
    )

    expected_rows = {
        "doc_000": (1.0, 10.0, 100.0),
        "doc_050": (50.0, 500.0, 5000.0),
        "doc_099": (98.0, 980.0, 9800.0),
    }
    for doc_id, (expected_sue, expected_dispersion, expected_revisions) in expected_rows.items():
        row = panel.filter(pl.col("doc_id") == doc_id).row(0, named=True)
        assert row["sue"] == pytest.approx(expected_sue)
        assert row["analyst_dispersion"] == pytest.approx(expected_dispersion)
        assert row["analyst_revisions"] == pytest.approx(expected_revisions)


def test_build_lm2011_normalized_filing_feeds_and_sample_backbone_apply_raw_form_rules() -> None:
    sec_parsed = pl.DataFrame(
        {
            "doc_id": ["keep_10k", "keep_405", "drop_amend", "drop_kt", "drop_ccm_amend"],
            "cik_10": ["0001", "0002", "0003", "0004", "0005"],
            "filing_date": [
                dt.date(1995, 1, 10),
                dt.date(1995, 2, 1),
                dt.date(1995, 2, 2),
                dt.date(1995, 2, 3),
                dt.date(1995, 2, 4),
            ],
            "accession_nodash": ["1", "2", "3", "4", "5"],
            "document_type_filename": ["10K", "10K405", "10KA", "10KT", "10-K"],
        }
    )
    ccm_filingdates = pl.DataFrame(
        {
            "LPERMNO": [1, 2, 3, 4, 5],
            "FILEDATE": [
                dt.date(1995, 1, 10),
                dt.date(1995, 2, 1),
                dt.date(1995, 2, 2),
                dt.date(1995, 2, 3),
                dt.date(1995, 2, 4),
            ],
            "SRCTYPE": ["10K", "10K", "10K", "10K", "10K/A"],
        }
    )
    sec_norm, ccm_norm = build_lm2011_normalized_filing_feeds(sec_parsed.lazy(), ccm_filingdates.lazy())
    assert sec_norm.collect().filter(pl.col("doc_id") == "keep_405").select("normalized_form").item() == "10-K"
    assert ccm_norm.collect().select("normalized_form").to_series().to_list() == ["10-K", "10-K", "10-K", "10-K", "10-K/A"]

    matched_clean = pl.DataFrame(
        {
            "doc_id": ["keep_10k", "keep_405", "drop_amend", "drop_kt", "drop_ccm_amend"],
            "KYPERMNO": [1, 2, 3, 4, 5],
            "gvkey": [1001, 1002, 1003, 1004, 1005],
        }
    )
    backbone = build_lm2011_sample_backbone(
        sec_parsed.lazy(),
        matched_clean.lazy(),
        ccm_filingdates_lf=ccm_filingdates.lazy(),
    ).collect().sort("doc_id")

    assert backbone.get_column("doc_id").to_list() == ["keep_10k", "keep_405"]
    assert backbone.get_column("normalized_form").to_list() == ["10-K", "10-K"]


def test_build_lm2011_sample_backbone_requires_ccm_form_source_when_srctype_missing() -> None:
    sec_parsed = pl.DataFrame(
        {
            "doc_id": ["d1"],
            "cik_10": ["0001"],
            "filing_date": [dt.date(1995, 1, 10)],
            "accession_nodash": ["1"],
            "document_type_filename": ["10-K"],
        }
    )
    matched_clean = pl.DataFrame({"doc_id": ["d1"], "KYPERMNO": [1], "gvkey": [1001]})
    with pytest.raises(ValueError, match="requires CCM raw forms"):
        build_lm2011_sample_backbone(sec_parsed.lazy(), matched_clean.lazy()).collect()


def test_build_lm2011_sample_backbone_enforces_179_day_drop_and_180_day_keep() -> None:
    sec_parsed = pl.DataFrame(
        {
            "doc_id": ["drop_179_a", "drop_179_b", "keep_180_a", "keep_180_b"],
            "cik_10": ["0001", "0001", "0002", "0002"],
            "filing_date": [
                dt.date(1997, 12, 31),
                dt.date(1998, 6, 28),
                dt.date(1997, 12, 31),
                dt.date(1998, 6, 29),
            ],
            "accession_nodash": ["1", "2", "3", "4"],
            "document_type_filename": ["10-K", "10-K", "10-K", "10-K"],
        }
    )
    matched_clean = pl.DataFrame(
        {
            "doc_id": ["drop_179_a", "drop_179_b", "keep_180_a", "keep_180_b"],
            "KYPERMNO": [1, 2, 3, 4],
            "gvkey": [1001, 1001, 1002, 1002],
            "SRCTYPE": ["10K", "10K", "10K", "10K"],
        }
    )
    backbone = build_lm2011_sample_backbone(sec_parsed.lazy(), matched_clean.lazy()).collect()
    assert set(backbone.get_column("doc_id").to_list()) == {"drop_179_a", "keep_180_a", "keep_180_b"}


def test_build_lm2011_sample_backbone_uses_accession_nodash_as_same_day_tie_break() -> None:
    sec_parsed = pl.DataFrame(
        {
            "doc_id": ["high_accession", "low_accession"],
            "cik_10": ["0001", "0001"],
            "filing_date": [dt.date(1998, 3, 31), dt.date(1998, 3, 31)],
            "accession_nodash": ["0000000002", "0000000001"],
            "document_type_filename": ["10-K", "10-K"],
        }
    )
    matched_clean = pl.DataFrame(
        {
            "doc_id": ["high_accession", "low_accession"],
            "KYPERMNO": [1, 1],
            "gvkey": [1001, 1001],
            "SRCTYPE": ["10K", "10K"],
        }
    )

    backbone = build_lm2011_sample_backbone(sec_parsed.lazy(), matched_clean.lazy()).collect()

    assert backbone.get_column("doc_id").to_list() == ["low_accession"]
    assert backbone.get_column("accession_nodash").to_list() == ["0000000001"]


def test_build_lm2011_sample_backbone_dedupes_duplicate_doc_id_by_earliest_row() -> None:
    sec_parsed = pl.DataFrame(
        {
            "doc_id": ["dup_doc", "dup_doc"],
            "cik_10": ["0001", "0001"],
            "filing_date": [dt.date(1998, 4, 1), dt.date(1998, 3, 31)],
            "accession_nodash": ["0000000002", "0000000001"],
            "document_type_filename": ["10-K", "10-K"],
        }
    )
    matched_clean = pl.DataFrame(
        {
            "doc_id": ["dup_doc"],
            "KYPERMNO": [1],
            "gvkey": [1001],
            "SRCTYPE": ["10K"],
        }
    )

    sec_base = _build_lm2011_sample_backbone_stage_frames(sec_parsed.lazy(), matched_clean.lazy())[0][1].collect()

    assert sec_base.height == 1
    assert sec_base.item(0, "doc_id") == "dup_doc"
    assert sec_base.item(0, "filing_date") == dt.date(1998, 3, 31)
    assert sec_base.item(0, "accession_nodash") == "0000000001"


def test_build_lm2011_sample_backbone_stage_frames_report_expected_attrition() -> None:
    sec_parsed = pl.DataFrame(
        {
            "doc_id": [
                "keep_first",
                "drop_same_year",
                "keep_spacing",
                "drop_spacing",
                "drop_no_match",
                "keep_final",
                "keep_first",
                "drop_amend",
            ],
            "cik_10": ["0001", "0001", "0002", "0002", "0003", "0004", "0001", "0005"],
            "filing_date": [
                dt.date(1995, 1, 10),
                dt.date(1995, 7, 15),
                dt.date(1997, 12, 31),
                dt.date(1998, 6, 28),
                dt.date(1995, 3, 1),
                dt.date(1995, 4, 1),
                dt.date(1995, 1, 10),
                dt.date(1995, 5, 1),
            ],
            "accession_nodash": ["1", "2", "3", "4", "5", "6", "1", "7"],
            "document_type_filename": ["10-K", "10-K", "10-K", "10-K", "10-K", "10-K", "10-K", "10-K/A"],
        }
    )
    matched_clean = pl.DataFrame(
        {
            "doc_id": ["keep_first", "drop_same_year", "keep_spacing", "drop_spacing", "keep_final"],
            "KYPERMNO": [1, 1, 2, 2, 4],
            "gvkey": [1001, 1001, 1002, 1002, 1004],
            "SRCTYPE": ["10K", "10K", "10K", "10K", "10K"],
        }
    )

    stage_frames = _build_lm2011_sample_backbone_stage_frames(sec_parsed.lazy(), matched_clean.lazy())

    assert [row_id for row_id, _ in stage_frames] == [
        "edgar_complete_nonduplicate_sample",
        "first_filing_per_year",
        "minimum_180_day_spacing",
        "crsp_permno_match",
    ]
    assert [frame.select(pl.len()).collect().item() for _, frame in stage_frames] == [6, 5, 4, 3]


def test_build_annual_accounting_panel_computes_me_fiscal_and_firm_value_v() -> None:
    annual_bs = pl.DataFrame(
        {
            "KYGVKEY": [1000],
            "KEYSET": ["STD"],
            "FYYYY": [2022],
            "fyra": [12],
            "SEQ": [100.0],
            "CEQ": [95.0],
            "AT": [160.0],
            "LT": [70.0],
            "TXDITC": [5.0],
            "PSTKL": [None],
            "PSTKRV": [None],
            "PSTK": [10.0],
        }
    )
    annual_is = pl.DataFrame(
        {
            "KYGVKEY": [1000],
            "KEYSET": ["STD"],
            "FYYYY": [2022],
            "fyra": [12],
            "IB": [20.0],
            "XINT": [2.0],
            "TXDI": [1.0],
            "DVP": [1.5],
        }
    )
    annual_pd = pl.DataFrame(
        {
            "KYGVKEY": [1000],
            "KEYSET": ["STD"],
            "FYYYY": [2022],
            "fyra": [12],
            "FYEAR": [2022],
            "FYR": [12],
            "APDEDATE": [None],
            "FDATE": [dt.date(2023, 1, 31)],
            "PDATE": [dt.date(2022, 12, 31)],
        }
    )
    annual_fiscal_market = pl.DataFrame(
        {
            "KYGVKEY": [1000],
            "DATADATE": [dt.date(2022, 12, 31)],
            "MKVALT": [None],
            "PRCC": [8.0],
            "CSHO": [12.0],
        }
    )

    panel = build_annual_accounting_panel(
        annual_bs.lazy(),
        annual_is.lazy(),
        annual_pd.lazy(),
        annual_fiscal_market_lf=annual_fiscal_market.lazy(),
    ).collect()
    row = panel.row(0, named=True)

    assert row["market_equity_me_fiscal"] == 96.0
    assert row["firm_value_v"] == 161.0


def test_lm2011_text_feature_builders_match_public_spec_contract() -> None:
    dictionary_lists = _lm_dictionary_lists()
    sec_parsed = pl.DataFrame(
        {
            "doc_id": ["d1", "d2"],
            "cik_10": ["0001", "0002"],
            "filing_date": [dt.date(2023, 1, 1), dt.date(2023, 1, 2)],
            "document_type_filename": ["10-K", "10-K405"],
            "full_text": ["gain gain loss may must uncertain", "gain lawsuit"],
        }
    )
    sec_items = pl.DataFrame(
        {
            "doc_id": ["d1", "d1", "d2"],
            "cik_10": ["0001", "0001", "0002"],
            "filing_date": [dt.date(2023, 1, 1)] * 3,
            "document_type_filename": ["10-K", "10-K", "10-K405"],
            "item_id": ["7", "8", "7"],
            "full_text": ["loss may", "ignored text", "gain must lawsuit"],
        }
    )

    full_features = build_lm2011_text_features_full_10k(
        sec_parsed.lazy(),
        dictionary_lists=dictionary_lists,
        harvard_negative_word_list=_harvard_negative_word_list(),
        master_dictionary_words=_master_dictionary_words(),
        batch_size=1,
    ).collect().sort("doc_id")
    mda_features = build_lm2011_text_features_mda(
        sec_items.lazy(),
        dictionary_lists=dictionary_lists,
        harvard_negative_word_list=_harvard_negative_word_list(),
        master_dictionary_words=_master_dictionary_words(),
        batch_size=1,
    ).collect().sort("doc_id")

    assert "h4n_inf_tfidf" in full_features.columns
    assert "lm_negative_tfidf" in full_features.columns
    assert "lm_positive_tfidf" in full_features.columns
    assert "lm_uncertainty_tfidf" in full_features.columns
    assert "lm_litigious_tfidf" in full_features.columns
    assert "lm_modal_strong_tfidf" in full_features.columns
    assert "lm_modal_weak_tfidf" in full_features.columns
    assert "h4n_inf_tfidf" in mda_features.columns
    assert "lm_uncertainty_tfidf" in mda_features.columns

    assert full_features.filter(pl.col("doc_id") == "d1").row(0, named=True)["token_count_full_10k"] == 6
    assert full_features.filter(pl.col("doc_id") == "d1").row(0, named=True)["total_token_count_full_10k"] == 6
    assert mda_features.filter(pl.col("doc_id") == "d1").row(0, named=True)["token_count_mda"] == 2
    assert mda_features.filter(pl.col("doc_id") == "d1").row(0, named=True)["total_token_count_mda"] == 2
    assert mda_features["cleaning_policy_id"].unique().to_list() == ["raw_item_text"]


def test_tokenize_lm2011_text_matches_appendix_contract() -> None:
    assert tokenize_lm2011_text("going-concern don't alpha-\nbeta a i x co-op") == [
        "going-concern",
        "don't",
        "alphabeta",
        "co-op",
    ]


def test_lm2011_text_features_use_total_token_denominators_and_match_hyphenated_dictionary_entries() -> None:
    features = build_lm2011_text_features_full_10k(
        pl.DataFrame(
            {
                "doc_id": ["d1"],
                "cik_10": ["0001"],
                "filing_date": [dt.date(2023, 1, 1)],
                "document_type_filename": ["10-K"],
                "full_text": ["going-concern don't unknown a"],
            }
        ).lazy(),
        dictionary_lists={
            "negative": ["going-concern"],
            "positive": ["gain"],
            "uncertainty": ["uncertain"],
            "litigious": ["lawsuit"],
            "modal_strong": ["must"],
            "modal_weak": ["may"],
        },
        harvard_negative_word_list=["don't"],
        master_dictionary_words=["going-concern", "don't"],
    ).collect()

    row = features.row(0, named=True)
    assert row["total_token_count_full_10k"] == 3
    assert row["token_count_full_10k"] == 2
    assert row["lm_negative_prop"] == pytest.approx(1.0 / 3.0)
    assert row["h4n_inf_prop"] == pytest.approx(1.0 / 3.0)


def test_lm2011_text_feature_builders_use_exact_paper_tfidf_formula() -> None:
    full_features = build_lm2011_text_features_full_10k(
        pl.DataFrame(
            {
                "doc_id": ["d1", "d2", "d3"],
                "cik_10": ["0001", "0002", "0003"],
                "filing_date": [dt.date(2023, 1, 1), dt.date(2023, 1, 2), dt.date(2023, 1, 3)],
                "document_type_filename": ["10-K", "10-K", "10-K"],
                "full_text": ["loss loss gain", "loss bad", "gain safe"],
            }
        ).lazy(),
        dictionary_lists=_lm_dictionary_lists(),
        harvard_negative_word_list=_harvard_negative_word_list(),
        master_dictionary_words=["loss", "gain"],
        batch_size=1,
    ).collect().sort("doc_id")

    idf_loss = math.log(3.0 / 2.0)
    expected_d1_negative = ((1.0 + math.log(2.0)) / (1.0 + math.log(3.0))) * idf_loss
    expected_d2_negative = (1.0 / (1.0 + math.log(2.0))) * idf_loss
    expected_d2_h4n = math.log(3.0) / (1.0 + math.log(2.0))

    d1 = full_features.filter(pl.col("doc_id") == "d1").row(0, named=True)
    d2 = full_features.filter(pl.col("doc_id") == "d2").row(0, named=True)
    d3 = full_features.filter(pl.col("doc_id") == "d3").row(0, named=True)

    assert d2["total_token_count_full_10k"] == 2
    assert d2["token_count_full_10k"] == 1
    assert d1["lm_negative_tfidf"] == pytest.approx(expected_d1_negative)
    assert d2["lm_negative_tfidf"] == pytest.approx(expected_d2_negative)
    assert d2["h4n_inf_tfidf"] == pytest.approx(expected_d2_h4n)
    assert d3["lm_negative_tfidf"] == pytest.approx(0.0)
    assert d1["h4n_inf_tfidf"] == pytest.approx(0.0)


def test_build_lm2011_text_features_full_10k_exports_total_and_recognized_token_counts() -> None:
    features = build_lm2011_text_features_full_10k(
        pl.DataFrame(
            {
                "doc_id": ["d1"],
                "cik_10": ["0001"],
                "filing_date": [dt.date(2023, 1, 1)],
                "document_type_filename": ["10-K"],
                "full_text": ["gain neutral unknown"],
            }
        ).lazy(),
        dictionary_lists=_lm_dictionary_lists(),
        harvard_negative_word_list=_harvard_negative_word_list(),
        master_dictionary_words=["gain", "neutral"],
    ).collect()

    assert features.item(0, "total_token_count_full_10k") == 3
    assert features.item(0, "token_count_full_10k") == 2


def test_write_lm2011_text_features_full_10k_parquet_matches_eager_builder(tmp_path: Path) -> None:
    sec_parsed = pl.DataFrame(
        {
            "doc_id": ["d2", "d1"],
            "cik_10": ["0002", "0001"],
            "filing_date": [dt.date(2023, 1, 2), dt.date(2023, 1, 1)],
            "document_type_filename": ["10-K405", "10-K"],
            "full_text": ["gain lawsuit", "gain gain loss may must uncertain"],
        }
    )
    eager = build_lm2011_text_features_full_10k(
        sec_parsed.lazy(),
        dictionary_lists=_lm_dictionary_lists(),
        harvard_negative_word_list=_harvard_negative_word_list(),
        master_dictionary_words=_master_dictionary_words(),
        batch_size=1,
    ).collect()

    for batch_size in (1, 5, 10):
        output_path = tmp_path / f"full_10k_{batch_size}.parquet"
        row_count = write_lm2011_text_features_full_10k_parquet(
            sec_parsed.lazy(),
            output_path=output_path,
            dictionary_lists=_lm_dictionary_lists(),
            harvard_negative_word_list=_harvard_negative_word_list(),
            master_dictionary_words=_master_dictionary_words(),
            batch_size=batch_size,
        )
        streamed = pl.read_parquet(output_path)

        assert row_count == eager.height
        assert streamed.get_column("doc_id").to_list() == ["d1", "d2"]
        _assert_frames_equal_with_float_tolerance(eager, streamed, sort_by=["doc_id"])


def test_write_lm2011_text_features_mda_parquet_matches_eager_builder(tmp_path: Path) -> None:
    sec_items = pl.DataFrame(
        {
            "doc_id": ["d1", "d1", "d2"],
            "cik_10": ["0001", "0001", "0002"],
            "filing_date": [dt.date(2023, 1, 1)] * 3,
            "document_type_filename": ["10-K", "10-K", "10-K405"],
            "item_id": ["7", "8", "7"],
            "full_text": ["loss may", "ignored text", "gain must lawsuit"],
        }
    )
    eager = build_lm2011_text_features_mda(
        sec_items.lazy(),
        dictionary_lists=_lm_dictionary_lists(),
        harvard_negative_word_list=_harvard_negative_word_list(),
        master_dictionary_words=_master_dictionary_words(),
        batch_size=1,
    ).collect()

    for batch_size in (1, 5, 10):
        output_path = tmp_path / f"mda_{batch_size}.parquet"
        row_count = write_lm2011_text_features_mda_parquet(
            sec_items.lazy(),
            output_path=output_path,
            dictionary_lists=_lm_dictionary_lists(),
            harvard_negative_word_list=_harvard_negative_word_list(),
            master_dictionary_words=_master_dictionary_words(),
            batch_size=batch_size,
        )
        streamed = pl.read_parquet(output_path)

        assert row_count == eager.height
        _assert_frames_equal_with_float_tolerance(eager, streamed, sort_by=["doc_id"])


def test_write_lm2011_text_features_full_10k_parquet_emits_progress_callback(tmp_path: Path) -> None:
    sec_parsed = pl.DataFrame(
        {
            "doc_id": ["d1", "d2", "d3"],
            "cik_10": ["0001", "0002", "0003"],
            "filing_date": [dt.date(2023, 1, 1), dt.date(2023, 1, 2), dt.date(2023, 1, 3)],
            "document_type_filename": ["10-K", "10-K", "10-K"],
            "full_text": ["gain loss", "uncertain may", "lawsuit must"],
        }
    )
    output_path = tmp_path / "full_10k_progress.parquet"
    progress: list[dict[str, object]] = []

    row_count = write_lm2011_text_features_full_10k_parquet(
        sec_parsed.lazy(),
        output_path=output_path,
        dictionary_lists=_lm_dictionary_lists(),
        harvard_negative_word_list=_harvard_negative_word_list(),
        master_dictionary_words=_master_dictionary_words(),
        batch_size=1,
        progress_callback=progress.append,
    )

    assert row_count == 3
    assert progress == [
        {"event": "stage_source_start"},
        {"event": "stage_source_end"},
        {"event": "pass1_start"},
        {"event": "batch", "batch_index": 1, "batch_doc_count": 1, "docs_completed": 1},
        {"event": "batch", "batch_index": 2, "batch_doc_count": 1, "docs_completed": 2},
        {"event": "batch", "batch_index": 3, "batch_doc_count": 1, "docs_completed": 3},
        {"event": "pass2_start"},
        {"event": "pass2_end"},
    ]


def test_write_lm2011_text_features_full_10k_parquet_stages_source_once_before_microbatching(
    tmp_path: Path,
) -> None:
    observed: dict[str, int] = {"rows": 0}

    def _track_text(value: str | None) -> str | None:
        observed["rows"] += 1
        return value

    sec_parsed_lf = pl.DataFrame(
        {
            "doc_id": ["d1", "d2", "d3"],
            "cik_10": ["0001", "0002", "0003"],
            "filing_date": [dt.date(2023, 1, 1), dt.date(2023, 1, 2), dt.date(2023, 1, 3)],
            "document_type_filename": ["10-K", "10-K", "10-K"],
            "full_text": ["gain loss", "uncertain may", "lawsuit must"],
        }
    ).lazy().with_columns(
        pl.col("full_text")
        .map_elements(_track_text, return_dtype=pl.Utf8)
        .alias("full_text")
    )

    output_path = tmp_path / "full_10k_staged_source.parquet"
    row_count = write_lm2011_text_features_full_10k_parquet(
        sec_parsed_lf,
        output_path=output_path,
        dictionary_lists=_lm_dictionary_lists(),
        harvard_negative_word_list=_harvard_negative_word_list(),
        master_dictionary_words=_master_dictionary_words(),
        batch_size=1,
    )

    assert row_count == 3
    assert observed["rows"] == 3


def test_write_lm2011_text_features_full_10k_parquet_staged_source_disables_full_text_statistics(
    tmp_path: Path,
) -> None:
    temp_root = tmp_path / "staged_source_root"
    long_text = " ".join((["loss"] * 4096) + (["uncertain"] * 2048) + (["lawsuit"] * 1024))
    sec_parsed = pl.DataFrame(
        {
            "doc_id": ["d1", "d2", "d3"],
            "cik_10": ["0001", "0002", "0003"],
            "filing_date": [dt.date(2023, 1, 1), dt.date(2023, 1, 2), dt.date(2023, 1, 3)],
            "document_type_filename": ["10-K", "10-K", "10-K"],
            "full_text": [long_text, long_text.replace("loss", "gain"), long_text.replace("uncertain", "must")],
        }
    )
    output_path = tmp_path / "full_10k_staged_source_stats.parquet"

    row_count = write_lm2011_text_features_full_10k_parquet(
        sec_parsed.lazy(),
        output_path=output_path,
        dictionary_lists=_lm_dictionary_lists(),
        harvard_negative_word_list=_harvard_negative_word_list(),
        master_dictionary_words=_master_dictionary_words(),
        batch_size=1,
        temp_root=temp_root,
        cleanup_on_success=False,
    )

    workspaces = sorted(path for path in temp_root.iterdir() if path.is_dir())
    source_path = workspaces[0] / "source.parquet"
    parquet_file = pq.ParquetFile(source_path)
    full_text_index = parquet_file.schema.names.index("full_text")

    assert row_count == 3
    assert len(workspaces) == 1
    assert parquet_file.metadata.num_row_groups == 1
    assert parquet_file.metadata.row_group(0).column(full_text_index).statistics is None


def test_write_lm2011_text_features_full_10k_parquet_staged_source_uses_row_group_floor(
    tmp_path: Path,
) -> None:
    temp_root = tmp_path / "staged_row_groups_root"
    sec_parsed = pl.DataFrame(
        {
            "doc_id": [f"d{idx}" for idx in range(129)],
            "cik_10": [f"{idx:04d}" for idx in range(129)],
            "filing_date": [dt.date(2023, 1, 1)] * 129,
            "document_type_filename": ["10-K"] * 129,
            "full_text": ["gain loss uncertain"] * 129,
        }
    )
    output_path = tmp_path / "full_10k_staged_row_groups.parquet"

    row_count = write_lm2011_text_features_full_10k_parquet(
        sec_parsed.lazy(),
        output_path=output_path,
        dictionary_lists=_lm_dictionary_lists(),
        harvard_negative_word_list=_harvard_negative_word_list(),
        master_dictionary_words=_master_dictionary_words(),
        batch_size=1,
        temp_root=temp_root,
        cleanup_on_success=False,
    )

    workspaces = sorted(path for path in temp_root.iterdir() if path.is_dir())
    source_path = workspaces[0] / "source.parquet"
    parquet_file = pq.ParquetFile(source_path)

    assert row_count == 129
    assert len(workspaces) == 1
    assert parquet_file.metadata.num_rows == 129
    assert parquet_file.metadata.num_row_groups == 2


def test_write_lm2011_text_features_mda_parquet_emits_progress_callback(tmp_path: Path) -> None:
    sec_items = pl.DataFrame(
        {
            "doc_id": ["d1", "d2", "d3"],
            "cik_10": ["0001", "0002", "0003"],
            "filing_date": [dt.date(2023, 1, 1), dt.date(2023, 1, 2), dt.date(2023, 1, 3)],
            "document_type_filename": ["10-K", "10-K", "10-K"],
            "item_id": ["7", "7", "7"],
            "full_text": ["loss may", "gain must", "uncertain lawsuit"],
        }
    )
    output_path = tmp_path / "mda_progress.parquet"
    progress: list[dict[str, object]] = []

    row_count = write_lm2011_text_features_mda_parquet(
        sec_items.lazy(),
        output_path=output_path,
        dictionary_lists=_lm_dictionary_lists(),
        harvard_negative_word_list=_harvard_negative_word_list(),
        master_dictionary_words=_master_dictionary_words(),
        batch_size=1,
        progress_callback=progress.append,
    )

    assert row_count == 3
    assert progress == [
        {"event": "stage_source_start"},
        {"event": "stage_source_end"},
        {"event": "pass1_start"},
        {"event": "batch", "batch_index": 1, "batch_doc_count": 1, "docs_completed": 1},
        {"event": "batch", "batch_index": 2, "batch_doc_count": 1, "docs_completed": 2},
        {"event": "batch", "batch_index": 3, "batch_doc_count": 1, "docs_completed": 3},
        {"event": "pass2_start"},
        {"event": "pass2_end"},
    ]


def test_write_lm2011_text_features_full_10k_parquet_handles_large_docs_with_microbatching(tmp_path: Path) -> None:
    large_text = " ".join((["loss"] * 40_000) + (["gain"] * 20_000) + (["uncertain"] * 10_000))
    sec_parsed = pl.DataFrame(
        {
            "doc_id": ["d1", "d2"],
            "cik_10": ["0001", "0002"],
            "filing_date": [dt.date(2023, 1, 1), dt.date(2023, 1, 2)],
            "document_type_filename": ["10-K", "10-K"],
            "full_text": [large_text, large_text.replace("loss", "lawsuit")],
        }
    )

    output_path = tmp_path / "full_10k_large_docs.parquet"
    row_count = write_lm2011_text_features_full_10k_parquet(
        sec_parsed.lazy(),
        output_path=output_path,
        dictionary_lists=_lm_dictionary_lists(),
        harvard_negative_word_list=_harvard_negative_word_list(),
        master_dictionary_words=_master_dictionary_words(),
        batch_size=1,
    )

    out = pl.read_parquet(output_path).sort("doc_id")
    assert row_count == 2
    assert out.columns[0:5] == [
        "doc_id",
        "cik_10",
        "filing_date",
        "document_type_filename",
        "normalized_form",
    ]
    assert out.get_column("total_token_count_full_10k").to_list() == [70_000, 70_000]


def test_write_lm2011_text_features_full_10k_parquet_uses_unique_system_temp_workspace_when_omitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system_temp = tmp_path / "system_temp"
    system_temp.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(lm2011_text_module.tempfile, "gettempdir", lambda: str(system_temp))

    sec_parsed = pl.DataFrame(
        {
            "doc_id": ["d1"],
            "cik_10": ["0001"],
            "filing_date": [dt.date(2023, 1, 1)],
            "document_type_filename": ["10-K"],
            "full_text": ["gain loss"],
        }
    )
    output_path = tmp_path / "outputs" / "full_10k_system_temp.parquet"

    row_count = write_lm2011_text_features_full_10k_parquet(
        sec_parsed.lazy(),
        output_path=output_path,
        dictionary_lists=_lm_dictionary_lists(),
        harvard_negative_word_list=_harvard_negative_word_list(),
        master_dictionary_words=_master_dictionary_words(),
        batch_size=1,
        cleanup_on_success=False,
    )

    workspaces = sorted(path for path in system_temp.iterdir() if path.is_dir())
    assert row_count == 1
    assert output_path.exists()
    assert len(workspaces) == 1
    assert workspaces[0].parent == system_temp
    assert workspaces[0] != output_path.parent
    assert (workspaces[0] / "final.parquet").exists()


def test_write_streaming_text_features_uses_temp_root_as_workspace_parent(tmp_path: Path) -> None:
    temp_root = tmp_path / "streaming_root"
    output_path = tmp_path / "streamed_features.parquet"

    row_count = lm2011_text_module._write_streaming_text_features(
        [
            pl.DataFrame(
                {
                    "doc_id": ["d1"],
                    "cik_10": ["0001"],
                    "filing_date": [dt.date(2023, 1, 1)],
                    "document_type_filename": ["10-K"],
                    "normalized_form": ["10-K"],
                    "full_text": ["loss gain"],
                }
            )
        ],
        output_path=output_path,
        temp_root=temp_root,
        token_count_col="token_count_streaming",
        total_token_count_col="total_token_count_streaming",
        include_item_id=False,
        raw_form_col="document_type_filename",
        signal_specs=(("loss_signal", frozenset({"loss"}), False),),
        master_dictionary_words=("loss", "gain"),
        text_col="full_text",
        cleanup_on_success=False,
    )

    workspaces = sorted(path for path in temp_root.iterdir() if path.is_dir())
    assert row_count == 1
    assert output_path.exists()
    assert len(workspaces) == 1
    assert workspaces[0].parent == temp_root
    assert (workspaces[0] / "final.parquet").exists()


def test_write_lm2011_text_features_full_10k_parquet_cleans_workspace_on_success(tmp_path: Path) -> None:
    temp_root = tmp_path / "writer_temp_root"
    sec_parsed = pl.DataFrame(
        {
            "doc_id": ["d1"],
            "cik_10": ["0001"],
            "filing_date": [dt.date(2023, 1, 1)],
            "document_type_filename": ["10-K"],
            "full_text": ["gain loss"],
        }
    )
    output_path = tmp_path / "full_10k_cleanup.parquet"

    row_count = write_lm2011_text_features_full_10k_parquet(
        sec_parsed.lazy(),
        output_path=output_path,
        dictionary_lists=_lm_dictionary_lists(),
        harvard_negative_word_list=_harvard_negative_word_list(),
        master_dictionary_words=_master_dictionary_words(),
        batch_size=1,
        temp_root=temp_root,
        cleanup_on_success=True,
    )

    assert row_count == 1
    assert output_path.exists()
    assert temp_root.exists()
    assert not any(temp_root.iterdir())


def test_write_lm2011_text_features_full_10k_parquet_retains_workspace_on_staged_source_validation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temp_root = tmp_path / "failed_source_root"
    sec_parsed = pl.DataFrame(
        {
            "doc_id": ["d1"],
            "cik_10": ["0001"],
            "filing_date": [dt.date(2023, 1, 1)],
            "document_type_filename": ["10-K"],
            "full_text": ["gain loss"],
        }
    )
    output_path = tmp_path / "full_10k_failed_source.parquet"
    real_validate = lm2011_text_module._validate_parquet_quick

    def _fail_source_only(path: Path) -> None:
        if path.name == "source.parquet":
            raise OSError("bad staged source")
        real_validate(path)

    monkeypatch.setattr(lm2011_text_module, "_validate_parquet_quick", _fail_source_only)

    with pytest.raises(OSError, match="staged source parquet validation failed") as excinfo:
        write_lm2011_text_features_full_10k_parquet(
            sec_parsed.lazy(),
            output_path=output_path,
            dictionary_lists=_lm_dictionary_lists(),
            harvard_negative_word_list=_harvard_negative_word_list(),
            master_dictionary_words=_master_dictionary_words(),
            batch_size=1,
            temp_root=temp_root,
        )

    workspaces = sorted(path for path in temp_root.iterdir() if path.is_dir())
    source_path = workspaces[0] / "source.parquet"
    assert len(workspaces) == 1
    assert source_path.exists()
    assert str(source_path) in str(excinfo.value)


def test_write_lm2011_text_features_full_10k_parquet_retains_local_final_when_promotion_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temp_root = tmp_path / "failed_promotion_root"
    sec_parsed = pl.DataFrame(
        {
            "doc_id": ["d1"],
            "cik_10": ["0001"],
            "filing_date": [dt.date(2023, 1, 1)],
            "document_type_filename": ["10-K"],
            "full_text": ["gain loss"],
        }
    )
    output_path = tmp_path / "full_10k_failed_promotion.parquet"

    def _fail_promotion(src: Path, dst: Path, *, retries: int = 3, sleep: float = 1.0, validate: object = True) -> Path:
        raise OSError(f"copy failed from {src} to {dst}")

    monkeypatch.setattr(lm2011_text_module, "_copy_with_verify", _fail_promotion)

    with pytest.raises(OSError, match="final parquet promotion failed") as excinfo:
        write_lm2011_text_features_full_10k_parquet(
            sec_parsed.lazy(),
            output_path=output_path,
            dictionary_lists=_lm_dictionary_lists(),
            harvard_negative_word_list=_harvard_negative_word_list(),
            master_dictionary_words=_master_dictionary_words(),
            batch_size=1,
            temp_root=temp_root,
        )

    workspaces = sorted(path for path in temp_root.iterdir() if path.is_dir())
    local_final_path = workspaces[0] / "final.parquet"
    assert len(workspaces) == 1
    assert local_final_path.exists()
    assert str(local_final_path) in str(excinfo.value)
    assert not output_path.exists()


def test_build_lm2011_event_panel_uses_prc_when_final_prc_is_missing() -> None:
    inputs = _base_event_inputs(include_final_prc=False)
    panel = build_lm2011_event_panel(
        inputs["sample_backbone"].lazy(),
        inputs["daily"].lazy(),
        inputs["annual_panel"].lazy(),
        inputs["ff"].lazy(),
        inputs["ownership"].lazy(),
        inputs["text_features"].lazy(),
    ).collect()
    assert panel.height == 1
    assert panel.columns == [
        "doc_id",
        "gvkey_int",
        "KYPERMNO",
        "filing_date",
        "filing_trade_date",
        "pre_filing_trade_date",
        "size_event",
        "bm_event",
        "share_turnover",
        "pre_ffalpha",
        "institutional_ownership",
        "nasdaq_dummy",
        "filing_period_excess_return",
        "abnormal_volume",
        "postevent_return_volatility",
    ]
    expected_turnover = (
        inputs["daily"]
        .with_row_index("_idx", offset=0)
        .with_columns((pl.col("_idx").cast(pl.Int64, strict=False) - 260).alias("_relative_day"))
        .filter(pl.col("_relative_day").is_between(-252, -6, closed="both"))
        .get_column("VOL")
        .sum()
        / (10.0 * 1_000.0)
    )
    assert panel.item(0, "share_turnover") == pytest.approx(expected_turnover)


def test_build_lm2011_event_panel_requires_filing_day_shares_for_share_turnover() -> None:
    inputs = _base_event_inputs()
    inputs["daily"] = (
        inputs["daily"]
        .with_row_index("_idx", offset=0)
        .with_columns((pl.col("_idx").cast(pl.Int64, strict=False) - 260).alias("_relative_day"))
        .with_columns(
            pl.when(pl.col("_relative_day") == 0)
            .then(pl.lit(None, dtype=pl.Float64))
            .when(pl.col("_relative_day").is_between(1, 3, closed="both"))
            .then(pl.lit(25.0))
            .otherwise(pl.col("SHROUT"))
            .alias("SHROUT")
        )
        .drop("_idx", "_relative_day")
    )

    panel = build_lm2011_event_panel(
        inputs["sample_backbone"].lazy(),
        inputs["daily"].lazy(),
        inputs["annual_panel"].lazy(),
        inputs["ff"].lazy(),
        inputs["ownership"].lazy(),
        inputs["text_features"].lazy(),
    ).collect()

    assert panel.height == 1
    assert panel.row(0, named=True)["share_turnover"] is None


def test_regression_transforms_log_share_turnover_without_mutating_event_panel_contract() -> None:
    transformed = _apply_lm2011_regression_transforms(
        pl.DataFrame(
            {
                "size_event": [100.0],
                "bm_event": [2.5],
                "share_turnover": [12.0],
            }
        ).lazy()
    ).collect()
    row = transformed.row(0, named=True)
    assert row["share_turnover"] == pytest.approx(12.0)
    assert row["log_share_turnover"] == pytest.approx(math.log(12.0))
    assert row["log_size"] == pytest.approx(math.log(100.0))
    assert row["log_book_to_market"] == pytest.approx(math.log(2.5))


@pytest.mark.parametrize(
    ("case_name", "mutator"),
    [
        ("non_common_stock", lambda inputs: inputs.__setitem__("daily", inputs["daily"].with_columns(pl.lit(12).alias("SHRCD")))),
        ("low_price", lambda inputs: inputs.__setitem__("daily", inputs["daily"].with_columns(pl.lit(2.5).alias("FINAL_PRC"), pl.lit(2.5).alias("PRC"), pl.lit(25.0).alias("TCAP")))),
        ("missing_event_volume", lambda inputs: inputs.__setitem__("daily", inputs["daily"].with_columns(pl.when(pl.col("CALDT") == pl.lit(dt.date(2023, 9, 21))).then(None).otherwise(pl.col("VOL")).alias("VOL")))),
        ("bad_exchange", lambda inputs: inputs.__setitem__("daily", inputs["daily"].with_columns(pl.lit(4).alias("EXCHCD")))),
        ("nonpositive_book_equity", lambda inputs: inputs.__setitem__("annual_panel", inputs["annual_panel"].with_columns(pl.lit(0.0).alias("book_equity_be")))),
        (
            "short_token_count",
            lambda inputs: inputs.__setitem__(
                "text_features",
                pl.DataFrame(
                    {
                        "doc_id": ["doc_ok"],
                        "token_count_full_10k": [1999],
                        "total_token_count_full_10k": [1999],
                    }
                ),
            ),
        ),
    ],
)
def test_build_lm2011_event_panel_rejects_paper_filter_failures(case_name: str, mutator) -> None:
    inputs = _base_event_inputs()
    mutator(inputs)
    panel = build_lm2011_event_panel(
        inputs["sample_backbone"].lazy(),
        inputs["daily"].lazy(),
        inputs["annual_panel"].lazy(),
        inputs["ff"].lazy(),
        inputs["ownership"].lazy(),
        inputs["text_features"].lazy(),
    ).collect()
    assert panel.height == 0, case_name


def test_build_lm2011_event_panel_enforces_total_token_boundary_from_builder() -> None:
    inputs = _base_event_inputs()

    def _build_text_features(total_count: int) -> pl.LazyFrame:
        return build_lm2011_text_features_full_10k(
            pl.DataFrame(
                {
                    "doc_id": ["doc_ok"],
                    "cik_10": ["0001"],
                    "filing_date": [inputs["sample_backbone"].item(0, "filing_date")],
                    "document_type_filename": ["10-K"],
                    "full_text": [" ".join(["recognized"] * total_count)],
                }
            ).lazy(),
            dictionary_lists=_lm_dictionary_lists(),
            harvard_negative_word_list=_harvard_negative_word_list(),
            master_dictionary_words=["recognized"],
        )

    short_panel = build_lm2011_event_panel(
        inputs["sample_backbone"].lazy(),
        inputs["daily"].lazy(),
        inputs["annual_panel"].lazy(),
        inputs["ff"].lazy(),
        inputs["ownership"].lazy(),
        _build_text_features(1999),
    ).collect()
    boundary_panel = build_lm2011_event_panel(
        inputs["sample_backbone"].lazy(),
        inputs["daily"].lazy(),
        inputs["annual_panel"].lazy(),
        inputs["ff"].lazy(),
        inputs["ownership"].lazy(),
        _build_text_features(2000),
    ).collect()

    assert short_panel.height == 0
    assert boundary_panel.height == 1


def test_build_lm2011_event_panel_requires_total_token_screen_count_column() -> None:
    inputs = _base_event_inputs()

    with pytest.raises(ValueError, match="missing required columns"):
        build_lm2011_event_panel(
            inputs["sample_backbone"].lazy(),
            inputs["daily"].lazy(),
            inputs["annual_panel"].lazy(),
            inputs["ff"].lazy(),
            inputs["ownership"].lazy(),
            inputs["text_features"].drop("total_token_count_full_10k").lazy(),
        ).collect()


def test_build_lm2011_event_panel_rejects_insufficient_coverage_and_constant_pre_event_volume() -> None:
    short_daily = _daily_window_frame(
        permno=1,
        start=dt.date(2023, 1, 1),
        n_days=150,
        event_day_index=60,
    )
    short_inputs = _base_event_inputs()
    short_inputs["daily"] = short_daily
    short_inputs["ff"] = _ff_factors_for_dates(short_daily.get_column("CALDT").to_list())
    short_panel = build_lm2011_event_panel(
        short_inputs["sample_backbone"].lazy(),
        short_inputs["daily"].lazy(),
        short_inputs["annual_panel"].lazy(),
        short_inputs["ff"].lazy(),
        short_inputs["ownership"].lazy(),
        short_inputs["text_features"].lazy(),
    ).collect()
    assert short_panel.height == 0

    constant_inputs = _base_event_inputs(constant_pre_event_volume=True)
    constant_panel = build_lm2011_event_panel(
        constant_inputs["sample_backbone"].lazy(),
        constant_inputs["daily"].lazy(),
        constant_inputs["annual_panel"].lazy(),
        constant_inputs["ff"].lazy(),
        constant_inputs["ownership"].lazy(),
        constant_inputs["text_features"].lazy(),
    ).collect()
    assert constant_panel.height == 0


def test_build_lm2011_event_panel_postevent_volatility_excludes_days_four_and_five() -> None:
    def _mutate_event_returns(daily: pl.DataFrame, *, relative_days: set[int], value: float) -> pl.DataFrame:
        return (
            daily.with_row_index("_idx", offset=0)
            .with_columns((pl.col("_idx").cast(pl.Int64, strict=False) - 260).alias("_relative_day"))
            .with_columns(
                pl.when(pl.col("_relative_day").is_in(sorted(relative_days)))
                .then(pl.lit(value))
                .otherwise(pl.col("FINAL_RET"))
                .alias("FINAL_RET"),
                pl.when(pl.col("_relative_day").is_in(sorted(relative_days)))
                .then(pl.lit(value))
                .otherwise(pl.col("RET"))
                .alias("RET"),
            )
            .drop("_idx", "_relative_day")
        )

    base_inputs = _base_event_inputs()
    day45_inputs = _base_event_inputs()
    day6_inputs = _base_event_inputs()

    day45_inputs["daily"] = _mutate_event_returns(day45_inputs["daily"], relative_days={4, 5}, value=0.35)
    day6_inputs["daily"] = _mutate_event_returns(day6_inputs["daily"], relative_days={6}, value=0.35)

    def _collect_panel(inputs: dict[str, pl.DataFrame]) -> pl.DataFrame:
        return build_lm2011_event_panel(
            inputs["sample_backbone"].lazy(),
            inputs["daily"].lazy(),
            inputs["annual_panel"].lazy(),
            inputs["ff"].lazy(),
            inputs["ownership"].lazy(),
            inputs["text_features"].lazy(),
        ).collect()

    base_panel = _collect_panel(base_inputs)
    day45_panel = _collect_panel(day45_inputs)
    day6_panel = _collect_panel(day6_inputs)

    base_vol = base_panel.item(0, "postevent_return_volatility")
    day45_vol = day45_panel.item(0, "postevent_return_volatility")
    day6_vol = day6_panel.item(0, "postevent_return_volatility")

    assert day45_vol == pytest.approx(base_vol)
    assert day6_vol > base_vol


def test_build_lm2011_event_panel_rejects_non_unique_ownership_input() -> None:
    inputs = _base_event_inputs()
    duplicate_ownership = pl.DataFrame(
        {
            "doc_id": ["doc_ok", "doc_ok"],
            "institutional_ownership_pct": [44.0, 45.0],
        }
    )
    with pytest.raises(ValueError, match="must be unique on doc_id"):
        build_lm2011_event_panel(
            inputs["sample_backbone"].lazy(),
            inputs["daily"].lazy(),
            inputs["annual_panel"].lazy(),
            inputs["ff"].lazy(),
            duplicate_ownership.lazy(),
            inputs["text_features"].lazy(),
        ).collect()


def test_build_lm2011_table_i_market_stage_frames_report_expected_attrition() -> None:
    inputs = _table_i_market_inputs()

    stage_frames = _build_lm2011_table_i_market_stage_frames(
        inputs["sample_backbone"].lazy(),
        inputs["daily"].lazy(),
        inputs["annual_panel"].lazy(),
        inputs["ff"].lazy(),
        inputs["text_features"].lazy(),
        event_window_doc_batch_size=2,
    )

    assert [row_id for row_id, _ in stage_frames] == [
        "ordinary_common_equity",
        "market_cap_available",
        "price_day_minus_one_ge_3",
        "event_window_returns_and_volume",
        "major_exchange_listing",
        "sixty_day_pre_post_coverage",
        "book_to_market_available_and_book_value_positive",
        "token_count_ge_2000",
    ]
    assert [frame.height for _, frame in stage_frames] == [8, 7, 6, 5, 4, 3, 2, 1]


def test_build_lm2011_event_screen_surface_batched_emits_one_row_per_doc_with_required_columns() -> None:
    inputs = _table_i_market_inputs()

    surface = _build_lm2011_event_screen_surface_batched(
        inputs["sample_backbone"].lazy(),
        inputs["daily"].lazy(),
        inputs["annual_panel"].lazy(),
        inputs["ff"].lazy(),
        inputs["text_features"].lazy(),
        event_window_doc_batch_size=2,
    )

    assert surface.height == inputs["sample_backbone"].height
    assert surface.get_column("doc_id").n_unique() == surface.height
    assert {
        "doc_id",
        "filing_trade_date",
        "pre_filing_trade_date",
        "pre_filing_price",
        "size_event",
        "bm_event",
        "event_return_day_count",
        "event_volume_day_count",
        "pre_turnover_obs",
        "abnormal_volume_pre_obs",
        "event_shares",
        "event_shrcd",
        "event_exchcd",
        "pre_alpha_obs",
        "post_alpha_obs",
        "filing_period_excess_return",
        "share_turnover",
        "abnormal_volume",
        "pre_ffalpha",
        "postevent_return_volatility",
    }.issubset(set(surface.columns))
    assert all(not column.startswith("_") for column in surface.columns)
    assert "full_text" not in surface.columns
    assert "passes_all_filters" not in surface.columns
    assert not any(column.startswith("filter_") for column in surface.columns)


def test_build_lm2011_event_screen_surface_batched_never_exceeds_doc_batch_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _table_i_market_inputs()
    observed_doc_batch_sizes: list[int] = []
    original = lm2011_pipeline._build_window_rows

    def _capture_window_rows(docs_df: pl.DataFrame, daily_df: pl.DataFrame) -> pl.DataFrame:
        observed_doc_batch_sizes.append(int(docs_df.height))
        return original(docs_df, daily_df)

    monkeypatch.setattr(lm2011_pipeline, "_build_window_rows", _capture_window_rows)

    surface = _build_lm2011_event_screen_surface_batched(
        inputs["sample_backbone"].lazy(),
        inputs["daily"].lazy(),
        inputs["annual_panel"].lazy(),
        inputs["ff"].lazy(),
        inputs["text_features"].lazy(),
        event_window_doc_batch_size=2,
    )

    assert surface.height == inputs["sample_backbone"].height
    assert observed_doc_batch_sizes
    assert max(observed_doc_batch_sizes) <= 2


def test_write_lm2011_event_screen_surface_parquet_matches_batched_builder(tmp_path: Path) -> None:
    inputs = _table_i_market_inputs()
    eager = _build_lm2011_event_screen_surface_batched(
        inputs["sample_backbone"].lazy(),
        inputs["daily"].lazy(),
        inputs["annual_panel"].lazy(),
        inputs["ff"].lazy(),
        inputs["text_features"].lazy(),
        event_window_doc_batch_size=2,
    )

    output_path = tmp_path / "event_screen_surface.parquet"
    row_count = write_lm2011_event_screen_surface_parquet(
        inputs["sample_backbone"].lazy(),
        inputs["daily"].lazy(),
        inputs["annual_panel"].lazy(),
        inputs["ff"].lazy(),
        inputs["text_features"].lazy(),
        output_path=output_path,
        event_window_doc_batch_size=2,
    )
    streamed = pl.read_parquet(output_path)

    assert row_count == eager.height
    _assert_frames_equal_with_float_tolerance(eager, streamed, sort_by=["doc_id"])


def test_write_lm2011_event_screen_surface_parquet_never_collects_more_than_doc_batch_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _table_i_market_inputs()
    observed_doc_batch_sizes: list[int] = []
    original = lm2011_pipeline._collect_staged_event_doc_batch

    def _capture_staged_docs_batch(
        staged_docs_lf: pl.LazyFrame,
        *,
        batch_start: int,
        batch_size: int,
    ) -> pl.DataFrame:
        docs_batch = original(
            staged_docs_lf,
            batch_start=batch_start,
            batch_size=batch_size,
        )
        observed_doc_batch_sizes.append(int(docs_batch.height))
        return docs_batch

    monkeypatch.setattr(lm2011_pipeline, "_collect_staged_event_doc_batch", _capture_staged_docs_batch)

    output_path = tmp_path / "event_surface_batched_writer.parquet"
    row_count = write_lm2011_event_screen_surface_parquet(
        inputs["sample_backbone"].lazy(),
        inputs["daily"].lazy(),
        inputs["annual_panel"].lazy(),
        inputs["ff"].lazy(),
        inputs["text_features"].lazy(),
        output_path=output_path,
        event_window_doc_batch_size=2,
    )

    assert row_count == inputs["sample_backbone"].height
    assert observed_doc_batch_sizes
    assert max(observed_doc_batch_sizes) <= 2


def test_write_lm2011_event_screen_surface_parquet_materializes_doc_base_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _table_i_market_inputs()
    docs_df = lm2011_pipeline._prepare_table_i_base_lf(
        inputs["sample_backbone"].lazy(),
        inputs["daily"].lazy(),
        inputs["annual_panel"].lazy(),
        inputs["text_features"].lazy(),
    ).collect()
    observed: dict[str, int] = {"rows": 0}

    def _track_doc_id(value: str | None) -> str | None:
        observed["rows"] += 1
        return value

    tracked_docs_lf = docs_df.lazy().with_columns(
        pl.col("doc_id")
        .map_elements(_track_doc_id, return_dtype=pl.Utf8)
        .alias("doc_id")
    )
    monkeypatch.setattr(lm2011_pipeline, "_prepare_table_i_base_lf", lambda *args, **kwargs: tracked_docs_lf)

    output_path = tmp_path / "event_surface_materialized_once.parquet"
    row_count = write_lm2011_event_screen_surface_parquet(
        inputs["sample_backbone"].lazy(),
        inputs["daily"].lazy(),
        inputs["annual_panel"].lazy(),
        inputs["ff"].lazy(),
        inputs["text_features"].lazy(),
        output_path=output_path,
        event_window_doc_batch_size=2,
    )

    assert row_count == docs_df.height
    assert observed["rows"] == docs_df.height


def test_build_lm2011_event_panel_matches_across_event_window_doc_batch_sizes() -> None:
    inputs = _table_i_market_inputs()
    ownership = pl.DataFrame(
        {
            "doc_id": inputs["sample_backbone"].get_column("doc_id").to_list(),
            "institutional_ownership_pct": [44.0] * inputs["sample_backbone"].height,
        }
    )

    panel_batch_one = build_lm2011_event_panel(
        inputs["sample_backbone"].lazy(),
        inputs["daily"].lazy(),
        inputs["annual_panel"].lazy(),
        inputs["ff"].lazy(),
        ownership.lazy(),
        inputs["text_features"].lazy(),
        event_window_doc_batch_size=1,
    ).collect().sort("doc_id")
    panel_batch_three = build_lm2011_event_panel(
        inputs["sample_backbone"].lazy(),
        inputs["daily"].lazy(),
        inputs["annual_panel"].lazy(),
        inputs["ff"].lazy(),
        ownership.lazy(),
        inputs["text_features"].lazy(),
        event_window_doc_batch_size=3,
    ).collect().sort("doc_id")

    assert panel_batch_one.to_dicts() == panel_batch_three.to_dicts()


def test_build_lm2011_table_i_sample_creation_matches_across_event_window_doc_batch_sizes() -> None:
    inputs = _table_i_market_inputs()
    filing_date = inputs["sample_backbone"].item(0, "filing_date")
    sec_parsed = pl.DataFrame(
        {
            "doc_id": inputs["sample_backbone"].get_column("doc_id").to_list(),
            "cik_10": inputs["sample_backbone"].get_column("cik_10").to_list(),
            "filing_date": [filing_date] * inputs["sample_backbone"].height,
            "accession_nodash": [str(idx + 1) for idx in range(inputs["sample_backbone"].height)],
            "document_type_filename": ["10-K"] * inputs["sample_backbone"].height,
        }
    )
    matched_clean = pl.DataFrame(
        {
            "doc_id": inputs["sample_backbone"].get_column("doc_id").to_list(),
            "KYPERMNO": inputs["sample_backbone"].get_column("KYPERMNO").to_list(),
            "gvkey": inputs["sample_backbone"].get_column("gvkey_int").to_list(),
            "SRCTYPE": ["10K"] * inputs["sample_backbone"].height,
        }
    )

    table_batch_one = build_lm2011_table_i_sample_creation(
        sec_parsed.lazy(),
        matched_clean.lazy(),
        inputs["daily"].lazy(),
        inputs["annual_panel"].lazy(),
        inputs["ff"].lazy(),
        inputs["text_features"].lazy(),
        event_window_doc_batch_size=1,
    ).sort("row_order")
    table_batch_four = build_lm2011_table_i_sample_creation(
        sec_parsed.lazy(),
        matched_clean.lazy(),
        inputs["daily"].lazy(),
        inputs["annual_panel"].lazy(),
        inputs["ff"].lazy(),
        inputs["text_features"].lazy(),
        event_window_doc_batch_size=4,
    ).sort("row_order")

    assert table_batch_one.to_dicts() == table_batch_four.to_dicts()


def test_build_lm2011_table_i_sample_creation_emits_expected_mda_rows() -> None:
    base_daily = _daily_window_frame(permno=1, start=dt.date(1997, 1, 1), n_days=520, event_day_index=260)
    filing_date = base_daily.item(260, "CALDT")
    sec_parsed = pl.DataFrame(
        {
            "doc_id": ["doc_a", "doc_b"],
            "cik_10": ["0000000001", "0000000002"],
            "filing_date": [filing_date, filing_date],
            "accession_nodash": ["1", "2"],
            "document_type_filename": ["10-K", "10-K"],
        }
    )
    matched_clean = pl.DataFrame(
        {
            "doc_id": ["doc_a", "doc_b"],
            "KYPERMNO": [1, 2],
            "gvkey": [1001, 1002],
            "SRCTYPE": ["10K", "10K"],
        }
    )
    daily = pl.concat(
        [
            base_daily,
            _daily_window_frame(permno=2, start=dt.date(1997, 1, 1), n_days=520, event_day_index=260),
        ],
        how="vertical_relaxed",
    )
    annual_panel = pl.DataFrame(
        {
            "gvkey_int": [1001, 1002],
            "accounting_period_end": [dt.date(1996, 12, 31), dt.date(1996, 12, 31)],
            "book_equity_be": [50.0, 50.0],
            "AT": [80.0, 80.0],
        }
    )
    full_text = pl.DataFrame(
        {
            "doc_id": ["doc_a", "doc_b"],
            "token_count_full_10k": [2500, 2500],
            "total_token_count_full_10k": [2500, 2500],
        }
    )
    mda_text = pl.DataFrame(
        {
            "doc_id": ["doc_a", "doc_b"],
            "token_count_mda": [300, 200],
            "total_token_count_mda": [300, 200],
        }
    )

    out = build_lm2011_table_i_sample_creation(
        sec_parsed.lazy(),
        matched_clean.lazy(),
        daily.lazy(),
        annual_panel.lazy(),
        _ff_factors_for_dates(base_daily.get_column("CALDT").to_list()).lazy(),
        full_text.lazy(),
        mda_text_features_lf=mda_text.lazy(),
    ).sort("row_order")

    out_rows = {row["row_id"]: row for row in out.iter_rows(named=True)}
    assert out_rows["token_count_ge_2000"]["sample_size_value"] == pytest.approx(2.0)
    assert out_rows["firm_year_sample"]["sample_size_value"] == pytest.approx(2.0)
    assert out_rows["unique_firms"]["sample_size_value"] == pytest.approx(2.0)
    assert out_rows["average_years_per_firm"]["sample_size_value"] == pytest.approx(1.0)
    assert out_rows["identifiable_mda"]["sample_size_value"] == pytest.approx(2.0)
    assert out_rows["identifiable_mda"]["observations_removed"] == 0
    assert out_rows["mda_token_count_ge_250"]["sample_size_value"] == pytest.approx(1.0)
    assert out_rows["mda_token_count_ge_250"]["observations_removed"] == 1
    assert out_rows["identifiable_mda"]["availability_status"] == "available"


def test_build_lm2011_table_i_sample_creation_marks_missing_mda_inputs_unavailable() -> None:
    daily = _daily_window_frame(permno=1, start=dt.date(1997, 1, 1), n_days=520, event_day_index=260)
    filing_date = daily.item(260, "CALDT")
    sec_parsed = pl.DataFrame(
        {
            "doc_id": ["doc_a"],
            "cik_10": ["0000000001"],
            "filing_date": [filing_date],
            "accession_nodash": ["1"],
            "document_type_filename": ["10-K"],
        }
    )
    matched_clean = pl.DataFrame({"doc_id": ["doc_a"], "KYPERMNO": [1], "gvkey": [1001], "SRCTYPE": ["10K"]})
    annual_panel = pl.DataFrame(
        {
            "gvkey_int": [1001],
            "accounting_period_end": [dt.date(1996, 12, 31)],
            "book_equity_be": [50.0],
            "AT": [80.0],
        }
    )
    out = build_lm2011_table_i_sample_creation(
        sec_parsed.lazy(),
        matched_clean.lazy(),
        daily.lazy(),
        annual_panel.lazy(),
        _ff_factors_for_dates(daily.get_column("CALDT").to_list()).lazy(),
        pl.DataFrame(
            {
                "doc_id": ["doc_a"],
                "token_count_full_10k": [2500],
                "total_token_count_full_10k": [2500],
            }
        ).lazy(),
    )

    mda_rows = out.filter(pl.col("section_id") == pl.lit("mda_subsection")).sort("row_order")
    assert mda_rows.get_column("availability_status").to_list() == ["unavailable", "unavailable"]
    assert mda_rows.get_column("availability_reason").to_list() == [
        "mda_text_features_unavailable",
        "mda_text_features_unavailable",
    ]
    assert out.filter(pl.col("row_id") == pl.lit("token_count_ge_2000")).item(0, "sample_size_value") == pytest.approx(1.0)


def test_build_lm2011_table_i_sample_creation_window_controls_counts_and_label() -> None:
    daily_1997 = _daily_window_frame(permno=1, start=dt.date(1997, 1, 1), n_days=520, event_day_index=260)
    daily_2010 = _daily_window_frame(permno=2, start=dt.date(2010, 1, 1), n_days=520, event_day_index=260)
    sec_parsed = pl.DataFrame(
        {
            "doc_id": ["doc_1997", "doc_2010"],
            "cik_10": ["0000000001", "0000000002"],
            "filing_date": [daily_1997.item(260, "CALDT"), daily_2010.item(260, "CALDT")],
            "accession_nodash": ["1", "2"],
            "document_type_filename": ["10-K", "10-K"],
        }
    )
    matched_clean = pl.DataFrame(
        {
            "doc_id": ["doc_1997", "doc_2010"],
            "KYPERMNO": [1, 2],
            "gvkey": [1001, 1002],
            "SRCTYPE": ["10K", "10K"],
        }
    )
    daily = pl.concat([daily_1997, daily_2010], how="vertical_relaxed")
    annual_panel = pl.DataFrame(
        {
            "gvkey_int": [1001, 1002],
            "accounting_period_end": [dt.date(1996, 12, 31), dt.date(2009, 12, 31)],
            "book_equity_be": [50.0, 50.0],
            "AT": [80.0, 80.0],
        }
    )
    full_text = pl.DataFrame(
        {
            "doc_id": ["doc_1997", "doc_2010"],
            "token_count_full_10k": [2500, 2500],
            "total_token_count_full_10k": [2500, 2500],
        }
    )
    ff = _ff_factors_for_dates(daily.select(pl.col("CALDT").unique().sort()).get_column("CALDT").to_list())

    default_out = build_lm2011_table_i_sample_creation(
        sec_parsed.lazy(),
        matched_clean.lazy(),
        daily.lazy(),
        annual_panel.lazy(),
        ff.lazy(),
        full_text.lazy(),
    )
    extended_out = build_lm2011_table_i_sample_creation(
        sec_parsed.lazy(),
        matched_clean.lazy(),
        daily.lazy(),
        annual_panel.lazy(),
        ff.lazy(),
        full_text.lazy(),
        sample_start=dt.date(1994, 1, 1),
        sample_end=dt.date(2024, 12, 31),
    )

    assert default_out.filter(pl.col("row_id") == pl.lit("token_count_ge_2000")).item(0, "sample_size_value") == pytest.approx(1.0)
    assert extended_out.filter(pl.col("row_id") == pl.lit("token_count_ge_2000")).item(0, "sample_size_value") == pytest.approx(2.0)
    assert "1994-2008" in default_out.item(0, "display_label")
    assert "1994-2024" in extended_out.item(0, "display_label")


def test_factor_scaling_is_uniform_across_columns() -> None:
    percent_scaled = pl.DataFrame({"mkt_rf": [1.2], "smb": [0.5], "hml": [0.2], "rf": [0.1]})
    out = _ensure_factor_scale(percent_scaled, ("mkt_rf", "smb", "hml", "rf"))
    row = out.row(0, named=True)
    assert row["mkt_rf"] == pytest.approx(0.012)
    assert row["smb"] == pytest.approx(0.005)
    assert row["hml"] == pytest.approx(0.002)
    assert row["rf"] == pytest.approx(0.001)

    already_decimal = pl.DataFrame({"mkt_rf": [0.12], "smb": [0.05], "hml": [0.02], "rf": [0.01]})
    unchanged = _ensure_factor_scale(already_decimal, ("mkt_rf", "smb", "hml", "rf"))
    assert unchanged.row(0, named=True)["mkt_rf"] == pytest.approx(0.12)


def test_ols_alpha_and_rmse_matches_deterministic_ff3_fixture_and_uses_n_minus_4() -> None:
    residuals = [0.0, -1.0, 0.0, 1.0, 1.0, -1.0]
    base_rows = [
        {"x1": 0.0, "x2": 0.0, "x3": 0.0},
        {"x1": 1.0, "x2": 0.0, "x3": 0.0},
        {"x1": 0.0, "x2": 1.0, "x3": 0.0},
        {"x1": 0.0, "x2": 0.0, "x3": 1.0},
        {"x1": 1.0, "x2": 1.0, "x3": 0.0},
        {"x1": 0.0, "x2": 1.0, "x3": 1.0},
    ]
    alpha_true = 1.5
    beta_1 = 0.8
    beta_2 = -1.2
    beta_3 = 0.4
    frame = pl.DataFrame(
        {
            "_y": [
                alpha_true + beta_1 * row["x1"] + beta_2 * row["x2"] + beta_3 * row["x3"] + residual
                for row, residual in zip(base_rows, residuals, strict=True)
            ],
            "_x1": [row["x1"] for row in base_rows],
            "_x2": [row["x2"] for row in base_rows],
            "_x3": [row["x3"] for row in base_rows],
        }
    )

    alpha, rmse = _ols_alpha_and_rmse(frame, label="unit_test_ff3")

    assert alpha == pytest.approx(alpha_true)
    assert rmse == pytest.approx(math.sqrt(sum(residual * residual for residual in residuals) / 2.0))


def test_ols_alpha_and_rmse_raises_on_rank_deficient_ff3_design() -> None:
    frame = pl.DataFrame(
        {
            "_y": [1.0, 2.0, 3.0, 4.0, 5.0],
            "_x1": [0.0, 1.0, 2.0, 3.0, 4.0],
            "_x2": [0.0, 2.0, 4.0, 6.0, 8.0],
            "_x3": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )

    with pytest.raises(ValueError, match="rank-deficient OLS design"):
        _ols_alpha_and_rmse(frame, label="rank_deficient_ff3")


def test_ols_coefficients_and_r2_match_deterministic_ff4_fixture() -> None:
    frame = pl.DataFrame(
        {
            "long_short_return": [0.3, 0.5, 0.2, 0.7, 0.1],
            "mkt_rf": [0.0, 1.0, 0.0, 0.0, 0.0],
            "smb": [0.0, 0.0, 1.0, 0.0, 0.0],
            "hml": [0.0, 0.0, 0.0, 1.0, 0.0],
            "mom": [0.0, 0.0, 0.0, 0.0, 1.0],
        }
    )

    coefficients, r2 = _ols_coefficients_and_r2(
        frame,
        y_col="long_short_return",
        x_cols=("mkt_rf", "smb", "hml", "mom"),
    )

    assert coefficients[0] == pytest.approx(0.3)
    assert coefficients[1] == pytest.approx(0.2)
    assert coefficients[2] == pytest.approx(-0.1)
    assert coefficients[3] == pytest.approx(0.4)
    assert coefficients[4] == pytest.approx(-0.2)
    assert r2 == pytest.approx(1.0)


def test_build_lm2011_sue_panel_supports_exact_and_safe_fallback_matching() -> None:
    event_panel = pl.DataFrame(
        {
            "doc_id": ["exact_doc", "fallback_doc"],
            "gvkey_int": [1000, 1001],
            "KYPERMNO": [1, 2],
            "filing_date": [dt.date(2023, 9, 18), dt.date(2023, 9, 18)],
            "pre_filing_trade_date": [dt.date(2023, 9, 17), dt.date(2023, 9, 17)],
            "size_event": [100.0, 120.0],
            "bm_event": [1.5, 1.8],
            "share_turnover": [10.0, 12.0],
            "pre_ffalpha": [0.02, 0.03],
            "institutional_ownership": [55.0, 60.0],
            "nasdaq_dummy": [1, 0],
        }
    )
    quarterly_panel = pl.DataFrame(
        {
            "gvkey_int": [1000, 1001],
            "quarter_report_date": [dt.date(2023, 10, 15), dt.date(2023, 10, 20)],
            "APDEDATEQ": [dt.date(2023, 9, 30), None],
            "PDATEQ": [dt.date(2023, 9, 30), None],
        }
    )
    ibes = pl.DataFrame(
        {
            "gvkey_int": [1000, 1001],
            "announcement_date": [dt.date(2023, 10, 15), dt.date(2023, 10, 20)],
            "fiscal_period_end": [dt.date(2023, 9, 30), dt.date(2023, 9, 25)],
            "actual_eps": [1.5, 2.0],
            "forecast_consensus_mean": [1.0, 1.5],
            "forecast_dispersion": [0.2, 0.3],
            "forecast_revision_4m": [0.24, 0.18],
        }
    )
    daily = pl.DataFrame(
        {
            "KYPERMNO": [1, 1, 2, 2],
            "CALDT": [dt.date(2023, 8, 31), dt.date(2023, 9, 17), dt.date(2023, 8, 31), dt.date(2023, 9, 17)],
            "PRC": [8.0, 10.0, 6.0, 12.0],
        }
    )

    sue_panel = build_lm2011_sue_panel(
        event_panel.lazy(),
        quarterly_panel.lazy(),
        ibes.lazy(),
        daily.lazy(),
    ).collect().sort("doc_id")

    assert sue_panel.get_column("doc_id").to_list() == ["exact_doc", "fallback_doc"]
    assert {"size_event", "bm_event", "share_turnover"}.issubset(set(sue_panel.columns))
    fallback_row = sue_panel.filter(pl.col("doc_id") == "fallback_doc").row(0, named=True)
    assert fallback_row["sue"] == pytest.approx((2.0 - 1.5) / 12.0)
    assert fallback_row["analyst_revisions"] == pytest.approx(0.18 / 6.0)


def test_build_lm2011_sue_panel_drops_ambiguous_announcement_date_fallbacks() -> None:
    event_panel = pl.DataFrame(
        {
            "doc_id": ["ambiguous_doc"],
            "gvkey_int": [1000],
            "KYPERMNO": [1],
            "filing_date": [dt.date(2023, 9, 18)],
            "pre_filing_trade_date": [dt.date(2023, 9, 17)],
            "size_event": [100.0],
            "bm_event": [1.5],
            "share_turnover": [10.0],
            "pre_ffalpha": [0.02],
            "institutional_ownership": [55.0],
            "nasdaq_dummy": [1],
        }
    )
    quarterly_panel = pl.DataFrame(
        {
            "gvkey_int": [1000],
            "quarter_report_date": [dt.date(2023, 10, 15)],
            "APDEDATEQ": [None],
            "PDATEQ": [None],
        }
    )
    ibes = pl.DataFrame(
        {
            "gvkey_int": [1000, 1000],
            "announcement_date": [dt.date(2023, 10, 15), dt.date(2023, 10, 15)],
            "fiscal_period_end": [dt.date(2023, 9, 30), dt.date(2023, 8, 31)],
            "actual_eps": [1.5, 1.6],
            "forecast_consensus_mean": [1.0, 1.0],
            "forecast_dispersion": [0.2, 0.2],
            "forecast_revision_4m": [0.24, 0.24],
        }
    )
    daily = pl.DataFrame(
        {
            "KYPERMNO": [1, 1],
            "CALDT": [dt.date(2023, 8, 31), dt.date(2023, 9, 17)],
            "PRC": [8.0, 10.0],
        }
    )

    sue_panel = build_lm2011_sue_panel(
        event_panel.lazy(),
        quarterly_panel.lazy(),
        ibes.lazy(),
        daily.lazy(),
    ).collect()
    assert sue_panel.height == 0


def test_build_lm2011_sue_panel_keeps_nullable_revision_only_in_upstream_artifact() -> None:
    event_panel = pl.DataFrame(
        {
            "doc_id": ["doc_with_null_revision"],
            "gvkey_int": [1000],
            "KYPERMNO": [1],
            "filing_date": [dt.date(2023, 9, 18)],
            "pre_filing_trade_date": [dt.date(2023, 9, 17)],
            "size_event": [100.0],
            "bm_event": [1.5],
            "share_turnover": [10.0],
            "pre_ffalpha": [0.02],
            "institutional_ownership": [55.0],
            "nasdaq_dummy": [1],
        }
    )
    quarterly_panel = pl.DataFrame(
        {
            "gvkey_int": [1000],
            "quarter_report_date": [dt.date(2023, 10, 15)],
            "APDEDATEQ": [dt.date(2023, 9, 30)],
            "PDATEQ": [dt.date(2023, 9, 30)],
        }
    )
    ibes = pl.DataFrame(
        {
            "gvkey_int": [1000],
            "announcement_date": [dt.date(2023, 10, 15)],
            "fiscal_period_end": [dt.date(2023, 9, 30)],
            "actual_eps": [1.5],
            "forecast_consensus_mean": [1.0],
            "forecast_dispersion": [0.2],
            "forecast_revision_4m": [None],
            "forecast_revision_4m_status": ["MISSING_BASE_SNAPSHOT"],
            "actual_fiscal_period_end_origin": ["API_DIRECT"],
        }
    )
    daily = pl.DataFrame(
        {
            "KYPERMNO": [1, 1],
            "CALDT": [dt.date(2023, 8, 31), dt.date(2023, 9, 17)],
            "PRC": [8.0, 10.0],
        }
    )

    sue_panel = build_lm2011_sue_panel(
        event_panel.lazy(),
        quarterly_panel.lazy(),
        ibes.lazy(),
        daily.lazy(),
    ).collect()

    assert sue_panel.height == 0


def test_build_lm2011_sue_panel_winsorizes_final_panel_globally() -> None:
    event_panel, quarterly_panel, ibes, daily = _build_sue_winsorization_inputs()

    sue_panel = build_lm2011_sue_panel(
        event_panel.lazy(),
        quarterly_panel.lazy(),
        ibes.lazy(),
        daily.lazy(),
    ).collect().sort("doc_id")

    assert sue_panel.height == 100
    _assert_sue_winsorization_surface(sue_panel, check_schema=True)


def test_write_lm2011_sue_panel_parquet_matches_eager_builder(tmp_path: Path) -> None:
    event_panel, quarterly_panel, ibes, daily = _build_sue_winsorization_inputs()

    eager = build_lm2011_sue_panel(
        event_panel.lazy(),
        quarterly_panel.lazy(),
        ibes.lazy(),
        daily.lazy(),
    ).collect().sort("doc_id")
    output_path = tmp_path / "sue_panel.parquet"
    row_count = write_lm2011_sue_panel_parquet(
        event_panel.lazy(),
        quarterly_panel.lazy(),
        ibes.lazy(),
        daily.lazy(),
        output_path=output_path,
        doc_batch_size=10,
    )
    streamed = pl.read_parquet(output_path).sort("doc_id")

    assert row_count == eager.height
    _assert_sue_winsorization_surface(streamed)
    _assert_frames_equal_with_float_tolerance(eager, streamed, sort_by=["doc_id"])


def test_write_lm2011_sue_panel_parquet_avoids_eager_reload_of_final_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_panel, quarterly_panel, ibes, daily = _build_sue_winsorization_inputs()

    eager = build_lm2011_sue_panel(
        event_panel.lazy(),
        quarterly_panel.lazy(),
        ibes.lazy(),
        daily.lazy(),
    ).collect().sort("doc_id")
    output_path = tmp_path / "sue_panel.parquet"
    original_read_parquet = lm2011_pipeline.pl.read_parquet

    def _guard_read_parquet(source: object, *args: object, **kwargs: object) -> pl.DataFrame:
        if isinstance(source, (str, Path)) and Path(source).resolve() == output_path.resolve():
            raise AssertionError("write_lm2011_sue_panel_parquet must not eagerly reload the final output parquet")
        return original_read_parquet(source, *args, **kwargs)

    monkeypatch.setattr(lm2011_pipeline.pl, "read_parquet", _guard_read_parquet)

    row_count = write_lm2011_sue_panel_parquet(
        event_panel.lazy(),
        quarterly_panel.lazy(),
        ibes.lazy(),
        daily.lazy(),
        output_path=output_path,
        doc_batch_size=10,
    )
    streamed = original_read_parquet(output_path).sort("doc_id")

    assert row_count == eager.height
    _assert_sue_winsorization_surface(streamed)
    _assert_frames_equal_with_float_tolerance(eager, streamed, sort_by=["doc_id"])


def test_write_lm2011_sue_panel_parquet_matches_eager_builder_for_mixed_global_fiscal_window_policy(
    tmp_path: Path,
) -> None:
    event_panel = pl.DataFrame(
        {
            "doc_id": ["fallback_nonnull_doc", "null_window_doc"],
            "gvkey_int": [1000, 1001],
            "KYPERMNO": [1, 2],
            "filing_date": [dt.date(2023, 9, 18), dt.date(2023, 9, 18)],
            "pre_filing_trade_date": [dt.date(2023, 9, 17), dt.date(2023, 9, 17)],
            "size_event": [100.0, 120.0],
            "bm_event": [1.5, 1.8],
            "share_turnover": [10.0, 12.0],
            "pre_ffalpha": [0.02, 0.03],
            "institutional_ownership": [55.0, 60.0],
            "nasdaq_dummy": [1, 0],
        }
    )
    quarterly_panel = pl.DataFrame(
        {
            "gvkey_int": [1000, 1001],
            "quarter_report_date": [dt.date(2023, 10, 15), dt.date(2023, 10, 20)],
            "APDEDATEQ": [dt.date(2023, 9, 30), None],
            "PDATEQ": [dt.date(2023, 9, 30), None],
        }
    )
    ibes = pl.DataFrame(
        {
            "gvkey_int": [1000, 1001],
            "announcement_date": [dt.date(2023, 10, 15), dt.date(2023, 10, 20)],
            "fiscal_period_end": [dt.date(2023, 8, 31), dt.date(2023, 9, 25)],
            "actual_eps": [1.5, 2.0],
            "forecast_consensus_mean": [1.0, 1.5],
            "forecast_dispersion": [0.2, 0.3],
            "forecast_revision_4m": [0.24, 0.18],
        }
    )
    daily = pl.DataFrame(
        {
            "KYPERMNO": [1, 1, 2, 2],
            "CALDT": [dt.date(2023, 8, 31), dt.date(2023, 9, 17), dt.date(2023, 8, 31), dt.date(2023, 9, 17)],
            "PRC": [8.0, 10.0, 6.0, 12.0],
        }
    )

    eager = build_lm2011_sue_panel(
        event_panel.lazy(),
        quarterly_panel.lazy(),
        ibes.lazy(),
        daily.lazy(),
    ).collect().sort("doc_id")
    assert eager.get_column("doc_id").to_list() == ["fallback_nonnull_doc", "null_window_doc"]

    output_path_batch_one = tmp_path / "sue_panel_mixed_policy_batch1.parquet"
    row_count_batch_one = write_lm2011_sue_panel_parquet(
        event_panel.lazy(),
        quarterly_panel.lazy(),
        ibes.lazy(),
        daily.lazy(),
        output_path=output_path_batch_one,
        doc_batch_size=1,
    )
    streamed_batch_one = pl.read_parquet(output_path_batch_one).sort("doc_id")

    output_path_batch_two = tmp_path / "sue_panel_mixed_policy_batch2.parquet"
    row_count_batch_two = write_lm2011_sue_panel_parquet(
        event_panel.lazy(),
        quarterly_panel.lazy(),
        ibes.lazy(),
        daily.lazy(),
        output_path=output_path_batch_two,
        doc_batch_size=2,
    )
    streamed_batch_two = pl.read_parquet(output_path_batch_two).sort("doc_id")

    assert row_count_batch_one == eager.height
    assert row_count_batch_two == eager.height
    _assert_frames_equal_with_float_tolerance(eager, streamed_batch_one, sort_by=["doc_id"])
    _assert_frames_equal_with_float_tolerance(eager, streamed_batch_two, sort_by=["doc_id"])
    _assert_frames_equal_with_float_tolerance(streamed_batch_one, streamed_batch_two, sort_by=["doc_id"])


def test_write_lm2011_sue_panel_parquet_materializes_doc_base_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_panel = pl.DataFrame(
        {
            "doc_id": ["exact_doc", "fallback_doc"],
            "gvkey_int": [1000, 1001],
            "KYPERMNO": [1, 2],
            "filing_date": [dt.date(2023, 9, 18), dt.date(2023, 9, 18)],
            "pre_filing_trade_date": [dt.date(2023, 9, 17), dt.date(2023, 9, 17)],
            "size_event": [100.0, 120.0],
            "bm_event": [1.5, 1.8],
            "share_turnover": [10.0, 12.0],
            "pre_ffalpha": [0.02, 0.03],
            "institutional_ownership": [55.0, 60.0],
            "nasdaq_dummy": [1, 0],
        }
    )
    quarterly_panel = pl.DataFrame(
        {
            "gvkey_int": [1000, 1001],
            "quarter_report_date": [dt.date(2023, 10, 15), dt.date(2023, 10, 20)],
            "APDEDATEQ": [dt.date(2023, 9, 30), None],
            "PDATEQ": [dt.date(2023, 9, 30), None],
        }
    )
    ibes = pl.DataFrame(
        {
            "gvkey_int": [1000, 1001],
            "announcement_date": [dt.date(2023, 10, 15), dt.date(2023, 10, 20)],
            "fiscal_period_end": [dt.date(2023, 9, 30), dt.date(2023, 9, 25)],
            "actual_eps": [1.5, 2.0],
            "forecast_consensus_mean": [1.0, 1.5],
            "forecast_dispersion": [0.2, 0.3],
            "forecast_revision_4m": [0.24, 0.18],
        }
    )
    daily = pl.DataFrame(
        {
            "KYPERMNO": [1, 1, 2, 2],
            "CALDT": [dt.date(2023, 8, 31), dt.date(2023, 9, 17), dt.date(2023, 8, 31), dt.date(2023, 9, 17)],
            "PRC": [8.0, 10.0, 6.0, 12.0],
        }
    )
    docs_df = lm2011_pipeline._prepare_lm2011_sue_docs_base_lf(
        event_panel.lazy(),
        quarterly_panel.lazy(),
    ).collect()
    observed: dict[str, int] = {"rows": 0}

    def _track_doc_id(value: str | None) -> str | None:
        observed["rows"] += 1
        return value

    tracked_docs_lf = docs_df.lazy().with_columns(
        pl.col("doc_id")
        .map_elements(_track_doc_id, return_dtype=pl.Utf8)
        .alias("doc_id")
    )
    monkeypatch.setattr(
        lm2011_pipeline,
        "_prepare_lm2011_sue_docs_base_lf",
        lambda *args, **kwargs: tracked_docs_lf,
    )

    output_path = tmp_path / "sue_panel_materialized_once.parquet"
    row_count = write_lm2011_sue_panel_parquet(
        event_panel.lazy(),
        quarterly_panel.lazy(),
        ibes.lazy(),
        daily.lazy(),
        output_path=output_path,
        doc_batch_size=1,
    )

    assert row_count == 2
    assert observed["rows"] == docs_df.height


def test_build_lm2011_trading_strategy_monthly_returns_pin_direction_and_support_weighting_modes() -> None:
    event_panel, sec_parsed, monthly_stock, monthly_factors = _build_strategy_inputs()
    equal_panel = build_lm2011_trading_strategy_monthly_returns(
        event_panel.lazy(),
        sec_parsed.lazy(),
        monthly_stock.lazy(),
        lm_dictionary_lists=_lm_dictionary_lists(),
        harvard_negative_word_list=_harvard_negative_word_list(),
        master_dictionary_words=_master_dictionary_words(),
    ).collect().sort("portfolio_month", "sort_signal_name")

    lagged_value_panel = build_lm2011_trading_strategy_monthly_returns(
        event_panel.lazy(),
        sec_parsed.lazy(),
        monthly_stock.lazy(),
        lm_dictionary_lists=_lm_dictionary_lists(),
        harvard_negative_word_list=_harvard_negative_word_list(),
        master_dictionary_words=_master_dictionary_words(),
        portfolio_weighting="lagged_value",
    ).collect().sort("portfolio_month", "sort_signal_name")

    assert equal_panel.columns == ["portfolio_month", "sort_signal_name", "long_short_return"]
    assert set(equal_panel.get_column("sort_signal_name").unique().to_list()) == {
        "fin_neg_prop",
        "fin_neg_tfidf",
        "h4n_inf_prop",
        "h4n_inf_tfidf",
    }
    assert equal_panel.height == 48
    assert equal_panel.get_column("portfolio_month").unique().sort().to_list() == _strategy_months()
    assert lagged_value_panel.height == equal_panel.height

    fin_neg_returns = equal_panel.filter(pl.col("sort_signal_name").is_in(["fin_neg_prop", "fin_neg_tfidf"]))
    h4n_returns = equal_panel.filter(pl.col("sort_signal_name").is_in(["h4n_inf_prop", "h4n_inf_tfidf"]))
    assert fin_neg_returns.select(pl.col("long_short_return").max()).item() < 0
    assert h4n_returns.select(pl.col("long_short_return").min()).item() > 0

    merged = equal_panel.join(
        lagged_value_panel.rename({"long_short_return": "lagged_long_short_return"}),
        on=["portfolio_month", "sort_signal_name"],
        how="inner",
    )
    assert (
        merged.filter((pl.col("long_short_return") - pl.col("lagged_long_short_return")).abs() > 1e-12).height
        > 0
    )


def test_build_lm2011_trading_strategy_monthly_returns_from_text_features_matches_legacy_builder() -> None:
    event_panel, sec_parsed, monthly_stock, _ = _build_strategy_inputs()
    full_10k_text_features = build_lm2011_text_features_full_10k(
        sec_parsed.lazy(),
        dictionary_lists=_lm_dictionary_lists(),
        harvard_negative_word_list=_harvard_negative_word_list(),
        master_dictionary_words=_master_dictionary_words(),
        batch_size=2,
    ).collect()

    legacy_equal = build_lm2011_trading_strategy_monthly_returns(
        event_panel.lazy(),
        sec_parsed.lazy(),
        monthly_stock.lazy(),
        lm_dictionary_lists=_lm_dictionary_lists(),
        harvard_negative_word_list=_harvard_negative_word_list(),
        master_dictionary_words=_master_dictionary_words(),
    ).collect()
    from_text_features_equal = build_lm2011_trading_strategy_monthly_returns_from_text_features(
        event_panel.lazy(),
        full_10k_text_features.lazy(),
        monthly_stock.lazy(),
    ).collect()
    legacy_lagged_value = build_lm2011_trading_strategy_monthly_returns(
        event_panel.lazy(),
        sec_parsed.lazy(),
        monthly_stock.lazy(),
        lm_dictionary_lists=_lm_dictionary_lists(),
        harvard_negative_word_list=_harvard_negative_word_list(),
        master_dictionary_words=_master_dictionary_words(),
        portfolio_weighting="lagged_value",
    ).collect()
    from_text_features_lagged_value = build_lm2011_trading_strategy_monthly_returns_from_text_features(
        event_panel.lazy(),
        full_10k_text_features.lazy(),
        monthly_stock.lazy(),
        portfolio_weighting="lagged_value",
    ).collect()

    _assert_frames_equal_with_float_tolerance(
        legacy_equal,
        from_text_features_equal,
        sort_by=["portfolio_month", "sort_signal_name"],
    )
    _assert_frames_equal_with_float_tolerance(
        legacy_lagged_value,
        from_text_features_lagged_value,
        sort_by=["portfolio_month", "sort_signal_name"],
    )
    assert set(from_text_features_equal.get_column("sort_signal_name").unique().to_list()) == {
        "fin_neg_prop",
        "fin_neg_tfidf",
        "h4n_inf_prop",
        "h4n_inf_tfidf",
    }
    assert from_text_features_equal.get_column("portfolio_month").unique().sort().to_list() == _strategy_months()


def test_build_lm2011_trading_strategy_monthly_returns_use_prior_year_signal_year() -> None:
    event_panel, sec_parsed, monthly_stock, monthly_factors = _build_strategy_inputs()
    prior_year_required = build_lm2011_trading_strategy_monthly_returns(
        event_panel.with_columns(pl.lit(dt.date(1997, 3, 1)).alias("filing_date")).lazy(),
        sec_parsed.lazy(),
        monthly_stock.lazy(),
        lm_dictionary_lists=_lm_dictionary_lists(),
        harvard_negative_word_list=_harvard_negative_word_list(),
        master_dictionary_words=_master_dictionary_words(),
    ).collect()
    assert prior_year_required.height == 0


def test_build_lm2011_trading_strategy_monthly_returns_propagates_cleaning_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_panel, sec_parsed, monthly_stock, _ = _build_strategy_inputs()
    captured: dict[str, object] = {}

    def _capture_signal_frame(*_: object, **kwargs: object) -> pl.LazyFrame:
        captured["cleaning_contract"] = kwargs.get("cleaning_contract")
        return pl.DataFrame(
            {
                "doc_id": [f"doc_{idx}" for idx in range(10)],
                "cik_10": [f"{idx:010d}" for idx in range(10)],
                "filing_date": [dt.date(1996, 3, 1) + dt.timedelta(days=idx) for idx in range(10)],
                "normalized_form": ["10-K"] * 10,
                "token_count_full_10k": [100] * 10,
                "total_token_count_full_10k": [100] * 10,
                "fin_neg_prop": [0.1 + 0.01 * idx for idx in range(10)],
                "fin_neg_tfidf": [0.2 + 0.01 * idx for idx in range(10)],
                "h4n_inf_prop": [0.3 + 0.01 * idx for idx in range(10)],
                "h4n_inf_tfidf": [0.4 + 0.01 * idx for idx in range(10)],
            }
        ).lazy()

    monkeypatch.setattr(
        "thesis_pkg.pipelines.lm2011_pipeline.build_lm2011_trading_strategy_signal_frame",
        _capture_signal_frame,
    )

    build_lm2011_trading_strategy_monthly_returns(
        event_panel.lazy(),
        sec_parsed.lazy(),
        monthly_stock.lazy(),
        lm_dictionary_lists=_lm_dictionary_lists(),
        harvard_negative_word_list=_harvard_negative_word_list(),
        master_dictionary_words=_master_dictionary_words(),
        cleaning_contract="lm2011_paper",
    ).collect()

    assert captured["cleaning_contract"] == "lm2011_paper"


def test_build_lm2011_trading_strategy_ff4_summary_is_separate_artifact_with_r2() -> None:
    event_panel, sec_parsed, monthly_stock, monthly_factors = _build_strategy_inputs()
    monthly_returns = build_lm2011_trading_strategy_monthly_returns(
        event_panel.lazy(),
        sec_parsed.lazy(),
        monthly_stock.lazy(),
        lm_dictionary_lists=_lm_dictionary_lists(),
        harvard_negative_word_list=_harvard_negative_word_list(),
        master_dictionary_words=_master_dictionary_words(),
    ).collect()
    summary = build_lm2011_trading_strategy_ff4_summary(
        monthly_returns.lazy(),
        monthly_factors.lazy(),
    ).collect().sort("sort_signal_name")

    assert monthly_returns.columns == ["portfolio_month", "sort_signal_name", "long_short_return"]
    assert "r2" not in monthly_returns.columns
    assert summary.columns == [
        "sort_signal_name",
        "alpha_ff3_mom",
        "beta_market",
        "beta_smb",
        "beta_hml",
        "beta_mom",
        "r2",
    ]
    assert summary.height == 4
    assert summary.select(pl.col("r2").is_not_null().all()).item() is True


def test_build_lm2011_trading_strategy_builders_fail_closed_without_required_external_inputs() -> None:
    event_panel, sec_parsed, monthly_stock, monthly_factors = _build_strategy_inputs()
    with pytest.raises(ValueError, match="harvard_negative_word_list is required"):
        build_lm2011_trading_strategy_monthly_returns(
            event_panel.lazy(),
            sec_parsed.lazy(),
            monthly_stock.lazy(),
            lm_dictionary_lists=_lm_dictionary_lists(),
            harvard_negative_word_list=None,
            master_dictionary_words=_master_dictionary_words(),
        ).collect()
    with pytest.raises(ValueError, match="ff_factors_monthly_with_mom_lf is required"):
        build_lm2011_trading_strategy_ff4_summary(
            build_lm2011_trading_strategy_monthly_returns(
                event_panel.lazy(),
                sec_parsed.lazy(),
                monthly_stock.lazy(),
                lm_dictionary_lists=_lm_dictionary_lists(),
                harvard_negative_word_list=_harvard_negative_word_list(),
                master_dictionary_words=_master_dictionary_words(),
            ),
            None,
        ).collect()


def test_sample_backed_daily_units_lock_tcap_matches_price_times_shrout() -> None:
    spec = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    sample_path = Path(
        spec["datasets"]["crsp_ccm_daily"]["derived_artifacts"]["final_daily_panel_parquet"]["sample_file"]
    )
    if not sample_path.exists():
        pytest.skip(f"Sample file not present: {sample_path}")
    schema = pl.scan_parquet(sample_path).collect_schema()
    if "SHROUT" not in schema:
        pytest.skip(f"Sample file does not expose SHROUT: {sample_path}")
    price_col = "FINAL_PRC" if "FINAL_PRC" in schema else "PRC" if "PRC" in schema else None
    market_cap_col = "TCAP" if "TCAP" in schema else "market_equity_me_event" if "market_equity_me_event" in schema else None
    if price_col is None or market_cap_col is None:
        pytest.skip(f"Sample file missing price/market-cap columns needed for unit lock: {sample_path}")
    sample_df = (
        pl.scan_parquet(sample_path)
        .filter(
            pl.col(market_cap_col).is_not_null()
            & pl.col(price_col).is_not_null()
            & pl.col("SHROUT").is_not_null()
            & (pl.col("SHROUT") > 0)
        )
        .select(
            (pl.col(market_cap_col) / (pl.col(price_col).abs() * pl.col("SHROUT"))).alias("ratio")
        )
        .head(100)
        .collect()
    )
    assert sample_df.height > 0
    assert sample_df.select((pl.col("ratio") - 1.0).abs().max()).item() < 1e-9

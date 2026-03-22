from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import polars as pl
import pytest
import yaml

from thesis_pkg.pipeline import (
    build_annual_accounting_panel,
    build_lm2011_event_panel,
    build_lm2011_normalized_filing_feeds,
    build_lm2011_sample_backbone,
    build_lm2011_sue_panel,
    build_lm2011_text_features_full_10k,
    build_lm2011_text_features_mda,
    build_lm2011_trading_strategy_ff4_summary,
    build_lm2011_trading_strategy_monthly_returns,
)
from thesis_pkg.pipelines.lm2011_pipeline import (
    _apply_lm2011_regression_transforms,
    _ensure_factor_scale,
    _ols_alpha_and_rmse,
    _solve_linear_system,
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
        "text_features": pl.DataFrame({"doc_id": ["doc_ok"], "token_count_full_10k": [2500]}),
        "ff": _ff_factors_for_dates(daily.get_column("CALDT").to_list()),
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
    ).collect().sort("doc_id")
    mda_features = build_lm2011_text_features_mda(
        sec_items.lazy(),
        dictionary_lists=dictionary_lists,
        harvard_negative_word_list=_harvard_negative_word_list(),
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
    assert mda_features.filter(pl.col("doc_id") == "d1").row(0, named=True)["token_count_mda"] == 2


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
    ).collect().sort("doc_id")

    idf_loss = math.log(3.0 / 2.0)
    expected_d1_negative = ((1.0 + math.log(2.0)) / (1.0 + math.log(3.0))) * idf_loss
    expected_d2_negative = (1.0 / (1.0 + math.log(2.0))) * idf_loss
    expected_d2_h4n = (1.0 / (1.0 + math.log(2.0))) * math.log(3.0 / 1.0)

    d1 = full_features.filter(pl.col("doc_id") == "d1").row(0, named=True)
    d2 = full_features.filter(pl.col("doc_id") == "d2").row(0, named=True)
    d3 = full_features.filter(pl.col("doc_id") == "d3").row(0, named=True)

    assert d1["lm_negative_tfidf"] == pytest.approx(expected_d1_negative)
    assert d2["lm_negative_tfidf"] == pytest.approx(expected_d2_negative)
    assert d2["h4n_inf_tfidf"] == pytest.approx(expected_d2_h4n)
    assert d3["lm_negative_tfidf"] == pytest.approx(0.0)
    assert d1["h4n_inf_tfidf"] == pytest.approx(0.0)


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
        / 10.0
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
        ("short_token_count", lambda inputs: inputs.__setitem__("text_features", pl.DataFrame({"doc_id": ["doc_ok"], "token_count_full_10k": [1999]}))),
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


def test_ols_alpha_and_rmse_uses_n_minus_4_denominator() -> None:
    rows = [
        {"y": 1.0, "x1": 0.0, "x2": 0.0, "x3": 0.0},
        {"y": 2.2, "x1": 1.0, "x2": 0.0, "x3": 0.0},
        {"y": 4.1, "x1": 0.0, "x2": 1.0, "x3": 0.0},
        {"y": 4.8, "x1": 0.0, "x2": 0.0, "x3": 1.0},
        {"y": 1.3, "x1": 1.0, "x2": 1.0, "x3": 0.0},
        {"y": 0.7, "x1": 0.0, "x2": 1.0, "x3": 1.0},
    ]
    xtx = [[0.0] * 4 for _ in range(4)]
    xty = [0.0] * 4
    syy = 0.0
    for row in rows:
        xs = [1.0, row["x1"], row["x2"], row["x3"]]
        y_value = row["y"]
        syy += y_value * y_value
        for idx in range(4):
            xty[idx] += xs[idx] * y_value
            for jdx in range(4):
                xtx[idx][jdx] += xs[idx] * xs[jdx]
    summary = {
        "n_obs": len(rows),
        "sx1": xtx[0][1],
        "sx2": xtx[0][2],
        "sx3": xtx[0][3],
        "sxx11": xtx[1][1],
        "sxx12": xtx[1][2],
        "sxx13": xtx[1][3],
        "sxx22": xtx[2][2],
        "sxx23": xtx[2][3],
        "sxx33": xtx[3][3],
        "sy": xty[0],
        "syy": syy,
        "sxy1": xty[1],
        "sxy2": xty[2],
        "sxy3": xty[3],
    }
    _, rmse = _ols_alpha_and_rmse(summary)
    beta = _solve_linear_system(xtx, xty)
    assert beta is not None
    sse = syy - sum(beta[idx] * xty[idx] for idx in range(len(beta)))
    expected_rmse = math.sqrt(sse / float(len(rows) - 4))
    assert rmse == pytest.approx(expected_rmse)


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


def test_build_lm2011_trading_strategy_monthly_returns_pin_direction_and_support_weighting_modes() -> None:
    event_panel, sec_parsed, monthly_stock, monthly_factors = _build_strategy_inputs()
    equal_panel = build_lm2011_trading_strategy_monthly_returns(
        event_panel.lazy(),
        sec_parsed.lazy(),
        monthly_stock.lazy(),
        lm_dictionary_lists=_lm_dictionary_lists(),
        harvard_negative_word_list=_harvard_negative_word_list(),
    ).collect().sort("portfolio_month", "sort_signal_name")

    lagged_value_panel = build_lm2011_trading_strategy_monthly_returns(
        event_panel.lazy(),
        sec_parsed.lazy(),
        monthly_stock.lazy(),
        lm_dictionary_lists=_lm_dictionary_lists(),
        harvard_negative_word_list=_harvard_negative_word_list(),
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


def test_build_lm2011_trading_strategy_monthly_returns_use_prior_year_signal_year() -> None:
    event_panel, sec_parsed, monthly_stock, monthly_factors = _build_strategy_inputs()
    prior_year_required = build_lm2011_trading_strategy_monthly_returns(
        event_panel.with_columns(pl.lit(dt.date(1997, 3, 1)).alias("filing_date")).lazy(),
        sec_parsed.lazy(),
        monthly_stock.lazy(),
        lm_dictionary_lists=_lm_dictionary_lists(),
        harvard_negative_word_list=_harvard_negative_word_list(),
    ).collect()
    assert prior_year_required.height == 0


def test_build_lm2011_trading_strategy_ff4_summary_is_separate_artifact_with_r2() -> None:
    event_panel, sec_parsed, monthly_stock, monthly_factors = _build_strategy_inputs()
    monthly_returns = build_lm2011_trading_strategy_monthly_returns(
        event_panel.lazy(),
        sec_parsed.lazy(),
        monthly_stock.lazy(),
        lm_dictionary_lists=_lm_dictionary_lists(),
        harvard_negative_word_list=_harvard_negative_word_list(),
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
        ).collect()
    with pytest.raises(ValueError, match="ff_factors_monthly_with_mom_lf is required"):
        build_lm2011_trading_strategy_ff4_summary(
            build_lm2011_trading_strategy_monthly_returns(
                event_panel.lazy(),
                sec_parsed.lazy(),
                monthly_stock.lazy(),
                lm_dictionary_lists=_lm_dictionary_lists(),
                harvard_negative_word_list=_harvard_negative_word_list(),
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

from __future__ import annotations

import datetime as dt

import polars as pl

from thesis_pkg.pipeline import (
    attach_eligible_quarterly_accounting,
    attach_lm2011_industry_classifications,
    attach_latest_annual_accounting,
    attach_pre_filing_market_data,
    build_annual_accounting_panel,
    build_quarterly_accounting_panel,
    derive_filing_trade_anchors,
)


def test_derive_filing_trade_anchors_handles_trading_and_nonstrading_filing_dates() -> None:
    filings = pl.DataFrame(
        {
            "doc_id": ["d1", "d2"],
            "filing_date": [dt.date(2024, 1, 3), dt.date(2024, 1, 4)],
        }
    )
    trading_calendar = pl.DataFrame(
        {
            "CALDT": [dt.date(2024, 1, 2), dt.date(2024, 1, 3), dt.date(2024, 1, 5)],
        }
    )

    anchored = derive_filing_trade_anchors(filings.lazy(), trading_calendar.lazy()).collect().sort("doc_id")

    assert anchored.select("filing_trade_date").to_series().to_list() == [
        dt.date(2024, 1, 3),
        dt.date(2024, 1, 5),
    ]
    assert anchored.select("pre_filing_trade_date").to_series().to_list() == [
        dt.date(2024, 1, 2),
        dt.date(2024, 1, 3),
    ]


def test_build_and_attach_annual_accounting_panel_applies_ff2001_formulas_and_age_gate() -> None:
    annual_bs = pl.DataFrame(
        {
            "KYGVKEY": [1000],
            "KEYSET": ["STD"],
            "FYYYY": [2022],
            "fyra": [12],
            "SEQ": [100.0],
            "CEQ": [90.0],
            "AT": [150.0],
            "LT": [60.0],
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
            "XINT": [3.0],
            "TXDI": [1.0],
            "DVP": [2.0],
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

    annual_panel = build_annual_accounting_panel(
        annual_bs.lazy(),
        annual_is.lazy(),
        annual_pd.lazy(),
    ).collect()

    row = annual_panel.row(0, named=True)
    assert row["accounting_period_end"] == dt.date(2022, 12, 31)
    assert row["preferred_stock_ps"] == 10.0
    assert row["book_equity_be"] == 95.0
    assert row["ebit_like_e"] == 24.0
    assert row["earnings_available_for_common_y"] == 19.0

    filings = pl.DataFrame(
        {
            "doc_id": ["fresh", "stale"],
            "gvkey": [1000, 1000],
            "filing_date": [dt.date(2023, 6, 1), dt.date(2024, 2, 1)],
        }
    )
    attached = attach_latest_annual_accounting(filings.lazy(), annual_panel.lazy()).collect().sort("doc_id")

    assert attached.filter(pl.col("doc_id") == "fresh").select("book_equity_be").item() == 95.0
    assert attached.filter(pl.col("doc_id") == "fresh").select("accounting_period_end").item() == dt.date(2022, 12, 31)
    assert attached.filter(pl.col("doc_id") == "stale").select("book_equity_be").item() is None
    assert attached.filter(pl.col("doc_id") == "stale").select("accounting_period_end").item() is None


def test_build_and_attach_quarterly_accounting_panel_uses_rdq_fallback_and_90_day_gate() -> None:
    quarterly_bs = pl.DataFrame(
        {
            "KYGVKEY": [1000],
            "KEYSET": ["STD"],
            "FYYYYQ": [20234],
            "fyrq": [12],
            "SEQQ": [100.0],
            "CEQQ": [95.0],
            "ATQ": [160.0],
            "LTQ": [70.0],
            "TXDITCQ": [4.0],
            "PSTKQ": [8.0],
        }
    )
    quarterly_is = pl.DataFrame(
        {
            "KYGVKEY": [1000],
            "KEYSET": ["STD"],
            "FYYYYQ": [20234],
            "fyrq": [12],
            "IBQ": [10.0],
            "XINTQ": [1.0],
            "TXDIQ": [0.5],
            "DVPQ": [0.25],
        }
    )
    quarterly_pd = pl.DataFrame(
        {
            "KYGVKEY": [1000],
            "KEYSET": ["STD"],
            "FYYYYQ": [20234],
            "fyrq": [12],
            "FYEARQ": [2023],
            "FQTR": [4],
            "APDEDATEQ": [dt.date(2023, 12, 31)],
            "FDATEQ": [dt.date(2024, 2, 20)],
            "PDATEQ": [dt.date(2024, 1, 31)],
            "RDQ": [0],
        }
    )

    quarterly_panel = build_quarterly_accounting_panel(
        quarterly_bs.lazy(),
        quarterly_is.lazy(),
        quarterly_pd.lazy(),
    ).collect()
    assert quarterly_panel.select("quarter_report_date").item() == dt.date(2024, 2, 20)

    filings = pl.DataFrame(
        {
            "doc_id": ["eligible", "late"],
            "gvkey": [1000, 1000],
            "filing_date": [dt.date(2024, 1, 10), dt.date(2023, 11, 1)],
        }
    )
    attached = attach_eligible_quarterly_accounting(filings.lazy(), quarterly_panel.lazy()).collect().sort("doc_id")

    assert attached.filter(pl.col("doc_id") == "eligible").select("quarter_report_date").item() == dt.date(2024, 2, 20)
    assert attached.filter(pl.col("doc_id") == "late").select("quarter_report_date").item() is None


def test_attach_pre_filing_market_data_prefers_tcap_and_falls_back_to_price_times_shares() -> None:
    filings = pl.DataFrame(
        {
            "doc_id": ["fallback", "preferred"],
            "kypermno": [1, 2],
            "pre_filing_trade_date": [dt.date(2024, 1, 2), dt.date(2024, 1, 2)],
            "book_equity_be": [50.0, 40.0],
        }
    )
    daily = pl.DataFrame(
        {
            "KYPERMNO": [1, 2],
            "CALDT": [dt.date(2024, 1, 2), dt.date(2024, 1, 2)],
            "TCAP": [None, 80.0],
            "PRC": [-5.0, -5.0],
            "SHROUT": [10.0, 20.0],
            "VOL": [100.0, 200.0],
            "SHRCD": [10, 10],
            "EXCHCD": [1, 3],
        }
    )

    attached = attach_pre_filing_market_data(filings.lazy(), daily.lazy()).collect().sort("doc_id")

    assert attached.filter(pl.col("doc_id") == "fallback").select("market_equity_me_event").item() == 50.0
    assert attached.filter(pl.col("doc_id") == "fallback").select("bm_event").item() == 1.0
    assert attached.filter(pl.col("doc_id") == "preferred").select("market_equity_me_event").item() == 80.0
    assert attached.filter(pl.col("doc_id") == "preferred").select("bm_event").item() == 0.5


def test_attach_lm2011_industry_classifications_uses_historical_sic_then_description_fallback(
    tmp_path,
) -> None:
    ff48_path = tmp_path / "ff48.txt"
    ff48_path.write_text(
        "\n".join(
            [
                " 1 Agric  Agriculture",
                "          0100-0199 Agricultural production - crops",
                "12 MedEq  Medical Equipment",
                "          3840-3849 Surgical, medical, and dental instruments and supplies",
            ]
        ),
        encoding="utf-8",
    )

    filings = pl.DataFrame(
        {
            "doc_id": ["hist", "fallback", "missing"],
            "gvkey": ["1000", "2000", "3000"],
            "filing_date": [dt.date(2024, 1, 10), dt.date(2024, 1, 10), dt.date(2024, 1, 10)],
        }
    )
    company_history = pl.DataFrame(
        {
            "KYGVKEY": ["1000", "2000"],
            "HCHGDT": [dt.date(2020, 1, 1), dt.date(2020, 1, 1)],
            "HCHGENDDT": [None, None],
            "HSIC": [3845, None],
        }
    )
    company_description = pl.DataFrame(
        {
            "KYGVKEY": ["1000", "2000", "3000"],
            "SIC": ["111", "0115", None],
        }
    )

    attached = attach_lm2011_industry_classifications(
        filings.lazy(),
        company_history.lazy(),
        company_description.lazy(),
        ff48_siccodes_path=ff48_path,
    ).collect().sort("doc_id")

    by_doc = {row["doc_id"]: row for row in attached.to_dicts()}
    assert by_doc["hist"]["HSIC"] == 3845
    assert by_doc["hist"]["SIC_desc"] == 111
    assert by_doc["hist"]["SIC_final"] == 3845
    assert by_doc["hist"]["ff48_industry_id"] == 12
    assert by_doc["hist"]["ff48_industry_short"] == "MedEq"

    assert by_doc["fallback"]["HSIC"] is None
    assert by_doc["fallback"]["SIC_desc"] == 115
    assert by_doc["fallback"]["SIC_final"] == 115
    assert by_doc["fallback"]["ff48_industry_id"] == 1
    assert by_doc["fallback"]["ff48_industry_short"] == "Agric"

    assert by_doc["missing"]["HSIC"] is None
    assert by_doc["missing"]["SIC_desc"] is None
    assert by_doc["missing"]["SIC_final"] is None
    assert by_doc["missing"]["ff48_industry_id"] is None

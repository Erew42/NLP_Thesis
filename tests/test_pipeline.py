import datetime as dt
from pathlib import Path

import polars as pl

from thesis_pkg.pipeline import (
    add_final_returns,
    attach_ccm_links,
    attach_filings,
    build_price_panel,
    DataStatus,
    merge_histories,
)


def test_build_price_panel_merges_delistings_and_flags():
    ds = pl.DataFrame(
        {
            "KYPERMNO": [1, 1],
            "CALDT": [dt.date(2024, 1, 2), dt.date(2024, 1, 3)],
            "BIDLO": [10.0, 11.0],
            "ASKHI": [10.5, 11.5],
        }
    )

    dp = pl.DataFrame(
        {
            "KYPERMNO": [1, 1],
            "CALDT": [dt.date(2024, 1, 2), dt.date(2024, 1, 3)],
            "RET": [0.01, 0.02],
            "RETX": [0.009, 0.018],
            "PRC": [-10.2, -11.3],
        }
    )

    delist = pl.DataFrame(
        {
            "KYPERMNO": [1],
            "DLSTDT": [dt.date(2024, 1, 4)],
            "DLRET": [-0.5],
            "DLRETX": [-0.4],
            "DLPRC": [5.0],
            "DLAMT": [None],
            "DLSTCD": [500],
        }
    )

    panel = build_price_panel(
        ds.lazy(),
        dp.lazy(),
        delist.lazy(),
        start_date=dt.date(2024, 1, 2),
    ).collect()

    flags = panel.select("CALDT", "price_status").to_dict(as_series=False)
    assert flags == {
        "CALDT": [dt.date(2024, 1, 2), dt.date(2024, 1, 3)],
        "price_status": [
            int(DataStatus.FULL_PANEL_DATA),
            int(DataStatus.FULL_PANEL_DATA),
        ],
    }

    assert panel.filter(pl.col("CALDT") == dt.date(2024, 1, 2)).select("DLRET").item() is None
    assert panel.filter(pl.col("CALDT") == dt.date(2024, 1, 3)).select("DLRET").item() == -0.5


def test_add_final_returns_combines_sources():
    base = pl.DataFrame(
        {
            "RET": [0.10, None, 0.05],
            "DLRET": [0.05, -0.2, None],
            "RETX": [0.08, None, 0.04],
            "DLRETX": [0.02, None, -0.01],
            "PRC": [-10.0, -5.0, -3.0],
            "DLPRC": [None, 1.5, None],
        }
    )

    enriched = add_final_returns(base.lazy()).collect()

    assert enriched.select("FINAL_RET").to_series().round(4).to_list() == [0.155, -0.2, 0.05]
    assert enriched.select("FINAL_RETX").to_series().round(4).to_list() == [0.1016, None, 0.0296]
    assert enriched.select("FINAL_PRC").to_series().to_list() == [10.0, 1.5, 3.0]
    assert enriched.select("prov_flags").to_series().to_list() == [61, 88, 45]


def test_attach_filings_aligns_to_trading_days_and_orders():
    price = pl.DataFrame(
        {
            "KYPERMNO": [1, 1, 2],
            "CALDT": [dt.date(2024, 1, 2), dt.date(2024, 1, 3), dt.date(2024, 1, 2)],
        }
    )

    filings = pl.DataFrame(
        {
            "LPERMNO": [1, 1, 1, 2, 1],
            "SRCTYPE": ["10-K", "10-K", "10-K", "8-K", "10-K"],
            "FILEDATE": [
                dt.date(2024, 1, 1),
                dt.date(2024, 1, 2),
                dt.date(2024, 1, 2),
                dt.date(2024, 1, 3),
                dt.date(2024, 1, 3),
            ],
            "FILEDATETIME": ["09:30:00", "10:45:00", "08:30:00", None, "07:00:00"],
        }
    )

    attached = attach_filings(price.lazy(), filings.lazy(), ["10-K", "8-K"]).collect()
    subset = attached.select(
        "KYPERMNO",
        "CALDT",
        "SRCTYPE_all",
        "FILEDATE_all",
        "FILEDATETIME_all",
        "n_filings",
    )

    row = subset.filter(
        (pl.col("CALDT") == dt.date(2024, 1, 2)) & (pl.col("KYPERMNO") == 1)
    ).row(0)
    filedates = row[subset.columns.index("FILEDATE_all")]
    srctypes = row[subset.columns.index("SRCTYPE_all")]
    datetimes = row[subset.columns.index("FILEDATETIME_all")]

    assert filedates == [dt.date(2024, 1, 2), dt.date(2024, 1, 2)]
    assert srctypes == ["10-K", "10-K"]
    assert [dtm.time() for dtm in datetimes] == [dt.time(8, 30), dt.time(10, 45)]
    assert (
        subset
        .filter((pl.col("CALDT") == dt.date(2024, 1, 2)) & (pl.col("KYPERMNO") == 1))
        .select("n_filings")
        .item()
        == 2
    )
    assert (
        subset
        .filter((pl.col("CALDT") == dt.date(2024, 1, 3)) & (pl.col("KYPERMNO") == 1))
        .select("n_filings")
        .item()
        == 1
    )


def test_attach_ccm_links_prefers_canonical_primary_link():
    price = pl.DataFrame(
        {
            "KYPERMNO": [1, 2],
            "CALDT": [dt.date(2024, 1, 2), dt.date(2024, 1, 2)],
            "KYGVKEY_final": [None, None],
        }
    )

    links = pl.DataFrame(
        {
            "KYGVKEY": ["1000", "2000"],
            "LPERMNO": [1, 1],
            "LIID": ["A", "B"],
            "LINKDT": [dt.date(2023, 12, 31), dt.date(2023, 12, 31)],
            "LINKENDDT": [dt.date(2024, 12, 31), dt.date(2024, 12, 31)],
            "LINKTYPE": ["LC", "LU"],
            "LINKPRIM": ["P", "P"],
        }
    )

    attached = attach_ccm_links(price.lazy(), links.lazy()).collect()

    chosen = attached.filter(pl.col("KYPERMNO") == 1).row(0)
    link_flag = attached.select("link_quality_flag").to_series().to_list()

    assert chosen[attached.columns.index("KYGVKEY_final")] == "1000"
    assert link_flag == ["canonical_primary_LC", "no_ccm_link"]


def test_merge_histories_combines_good_and_bad_rows(tmp_path: Path):
    price = pl.DataFrame(
        {
            "KYPERMNO": [1, 2],
            "CALDT": [dt.date(2024, 1, 2), dt.date(2024, 1, 2)],
            "KYGVKEY_final": ["1000", None],
            "LIID": ["A", None],
        }
    )

    sec_hist = pl.DataFrame(
        {
            "KYGVKEY": ["1000"],
            "KYIID": ["A"],
            "HSCHGDT": [dt.date(2023, 12, 31)],
            "HSCHGENDDT": [dt.date(2025, 1, 1)],
            "HTPCI": ["USA"],
            "HEXCNTRY": ["US"],
        }
    )

    comp_hist = pl.DataFrame(
        {
            "KYGVKEY": ["1000"],
            "HCHGDT": [dt.date(2023, 1, 1)],
            "HCHGENDDT": [dt.date(2025, 1, 1)],
            "HCIK": ["CIK1"],
            "HSIC": [100],
            "HNAICS": ["1111"],
            "HGSUBIND": ["Sub"],
        }
    )

    output_path = merge_histories(
        price.lazy(),
        sec_hist.lazy(),
        comp_hist.lazy(),
        tmp_path,
    )

    merged = pl.read_parquet(output_path)
    join1 = merged.select("join1_status").to_series().to_list()
    join2 = merged.select("join2_status").to_series().to_list()

    assert "CompHist OK" in join1
    assert "No GVKEY_final" in join1
    assert "SecHist OK" in join2
    assert "NA (Join 1 Failed)" in join2

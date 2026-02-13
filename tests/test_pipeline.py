import datetime as dt
from pathlib import Path
import warnings

import polars as pl
import pytest

from thesis_pkg.pipeline import (
    add_final_returns,
    attach_company_description,
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
            "TCAP": [100_000_000.0, 101_000_000.0],
            "VOL": [1000.0, 1100.0],
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
    nam = pl.DataFrame(
        {
            "KYPERMNO": [1],
            "NAMEDT": [dt.date(2023, 1, 1)],
            "NAMEENDDT": [dt.date(2024, 12, 31)],
            "SHRCD": [10],
            "EXCHCD": [1],
            "PRIMEXCH": ["N"],
            "TRDSTAT": ["A"],
            "SECSTAT": ["A"],
        }
    )
    hdr = pl.DataFrame(
        {
            "KYPERMNO": [1],
            "BEGDT": [dt.date(2020, 1, 1)],
            "ENDDT": [dt.date(2025, 1, 1)],
            "HSHRCD": [11],
            "HEXCD": [3],
            "HPRIMEXCH": ["Q"],
            "HTRDSTAT": ["A"],
            "HSECSTAT": ["A"],
        }
    )

    panel = build_price_panel(
        ds.lazy(),
        dp.lazy(),
        delist.lazy(),
        nam.lazy(),
        hdr.lazy(),
        start_date=dt.date(2024, 1, 2),
    ).collect()

    flags = panel.select("CALDT", "data_status").to_dict(as_series=False)
    assert flags == {
        "CALDT": [dt.date(2024, 1, 2), dt.date(2024, 1, 3)],
        "data_status": [
            int(DataStatus.FULL_PANEL_DATA),
            int(DataStatus.FULL_PANEL_DATA),
        ],
    }

    assert "price_status" not in panel.columns
    assert panel.filter(pl.col("CALDT") == dt.date(2024, 1, 2)).select("DLRET").item() is None
    assert panel.filter(pl.col("CALDT") == dt.date(2024, 1, 3)).select("DLRET").item() == -0.5
    assert panel.select("SHRCD").to_series().to_list() == [10, 10]
    assert panel.select("EXCHCD").to_series().to_list() == [1, 1]


def test_build_price_panel_uses_hdr_fallback_when_nam_out_of_range():
    ds = pl.DataFrame(
        {
            "KYPERMNO": [1],
            "CALDT": [dt.date(2024, 1, 2)],
            "BIDLO": [10.0],
            "ASKHI": [10.5],
        }
    )
    dp = pl.DataFrame(
        {
            "KYPERMNO": [1],
            "CALDT": [dt.date(2024, 1, 2)],
            "RET": [0.01],
            "RETX": [0.009],
            "PRC": [-10.2],
            "TCAP": [100_000_000.0],
            "VOL": [1000.0],
        }
    )
    delist = pl.DataFrame(
        {
            "KYPERMNO": [999],
            "DLSTDT": [dt.date(2024, 1, 2)],
            "DLRET": [None],
            "DLRETX": [None],
            "DLPRC": [None],
            "DLAMT": [None],
            "DLSTCD": [None],
        }
    )
    nam = pl.DataFrame(
        {
            "KYPERMNO": [1],
            "NAMEDT": [dt.date(2020, 1, 1)],
            "NAMEENDDT": [dt.date(2023, 12, 31)],
            "SHRCD": [10],
            "EXCHCD": [1],
            "PRIMEXCH": ["N"],
            "TRDSTAT": ["A"],
            "SECSTAT": ["A"],
        }
    )
    hdr = pl.DataFrame(
        {
            "KYPERMNO": [1],
            "BEGDT": [dt.date(2020, 1, 1)],
            "ENDDT": [dt.date(2024, 12, 31)],
            "HSHRCD": [11],
            "HEXCD": [3],
            "HPRIMEXCH": ["Q"],
            "HTRDSTAT": ["A"],
            "HSECSTAT": ["A"],
        }
    )

    panel = build_price_panel(
        ds.lazy(),
        dp.lazy(),
        delist.lazy(),
        nam.lazy(),
        hdr.lazy(),
        start_date=dt.date(2024, 1, 2),
    ).collect()

    assert panel.select("SHRCD").item() == 11
    assert panel.select("EXCHCD").item() == 3
    assert panel.select("PRIMEXCH").item() == "Q"
    assert panel.select("TRDSTAT").item() == "A"
    assert panel.select("SECSTAT").item() == "A"


def test_build_price_panel_sets_null_metadata_when_nam_and_hdr_out_of_range():
    ds = pl.DataFrame(
        {
            "KYPERMNO": [1],
            "CALDT": [dt.date(2024, 1, 2)],
            "BIDLO": [10.0],
            "ASKHI": [10.5],
        }
    )
    dp = pl.DataFrame(
        {
            "KYPERMNO": [1],
            "CALDT": [dt.date(2024, 1, 2)],
            "RET": [0.01],
            "RETX": [0.009],
            "PRC": [-10.2],
            "TCAP": [100_000_000.0],
            "VOL": [1000.0],
        }
    )
    delist = pl.DataFrame(
        {
            "KYPERMNO": [999],
            "DLSTDT": [dt.date(2024, 1, 2)],
            "DLRET": [None],
            "DLRETX": [None],
            "DLPRC": [None],
            "DLAMT": [None],
            "DLSTCD": [None],
        }
    )
    nam = pl.DataFrame(
        {
            "KYPERMNO": [1],
            "NAMEDT": [dt.date(2020, 1, 1)],
            "NAMEENDDT": [dt.date(2023, 12, 31)],
            "SHRCD": [10],
            "EXCHCD": [1],
            "PRIMEXCH": ["N"],
            "TRDSTAT": ["A"],
            "SECSTAT": ["A"],
        }
    )
    hdr = pl.DataFrame(
        {
            "KYPERMNO": [1],
            "BEGDT": [dt.date(2020, 1, 1)],
            "ENDDT": [dt.date(2023, 12, 31)],
            "HSHRCD": [11],
            "HEXCD": [3],
            "HPRIMEXCH": ["Q"],
            "HTRDSTAT": ["A"],
            "HSECSTAT": ["A"],
        }
    )

    panel = build_price_panel(
        ds.lazy(),
        dp.lazy(),
        delist.lazy(),
        nam.lazy(),
        hdr.lazy(),
        start_date=dt.date(2024, 1, 2),
    ).collect()

    assert panel.select("SHRCD").item() is None
    assert panel.select("EXCHCD").item() is None
    assert panel.select("PRIMEXCH").item() is None
    assert panel.select("TRDSTAT").item() is None
    assert panel.select("SECSTAT").item() is None


def test_build_price_panel_requires_nam_and_hdr_columns():
    ds = pl.DataFrame(
        {
            "KYPERMNO": [1],
            "CALDT": [dt.date(2024, 1, 2)],
            "BIDLO": [10.0],
            "ASKHI": [10.5],
        }
    )
    dp = pl.DataFrame(
        {
            "KYPERMNO": [1],
            "CALDT": [dt.date(2024, 1, 2)],
            "RET": [0.01],
            "RETX": [0.009],
            "PRC": [-10.2],
            "TCAP": [100_000_000.0],
            "VOL": [1000.0],
        }
    )
    delist = pl.DataFrame(
        {
            "KYPERMNO": [999],
            "DLSTDT": [dt.date(2024, 1, 2)],
            "DLRET": [None],
            "DLRETX": [None],
            "DLPRC": [None],
            "DLAMT": [None],
            "DLSTCD": [None],
        }
    )
    nam_missing = pl.DataFrame(
        {
            "KYPERMNO": [1],
            "NAMEDT": [dt.date(2020, 1, 1)],
            "NAMEENDDT": [dt.date(2024, 12, 31)],
        }
    )
    hdr_missing = pl.DataFrame(
        {
            "KYPERMNO": [1],
            "BEGDT": [dt.date(2020, 1, 1)],
            "ENDDT": [dt.date(2024, 12, 31)],
        }
    )

    with pytest.raises(ValueError, match="sfz_nam missing required columns"):
        build_price_panel(
            ds.lazy(),
            dp.lazy(),
            delist.lazy(),
            nam_missing.lazy(),
            hdr_missing.lazy(),
            start_date=dt.date(2024, 1, 2),
        ).collect()


def test_add_final_returns_combines_sources():
    base = pl.DataFrame(
        {
            "RET": [0.10, None, 0.05],
            "DLRET": [0.05, -0.2, None],
            "RETX": [0.08, None, 0.04],
            "DLRETX": [0.02, None, -0.01],
            "PRC": [-10.0, -5.0, -3.0],
            "DLPRC": [None, 1.5, None],
            "data_status": [int(DataStatus.NONE)] * 3,
        }
    )

    enriched = add_final_returns(base.lazy()).collect()

    assert enriched.select("FINAL_RET").to_series().round(4).to_list() == [0.155, -0.2, 0.05]
    assert enriched.select("FINAL_RETX").to_series().round(4).to_list() == [0.1016, None, 0.0296]
    assert enriched.select("FINAL_PRC").to_series().to_list() == [10.0, 1.5, 3.0]
    assert enriched.select("data_status").to_series().to_list() == [61, 88, 45]
    assert "prov_flags" not in enriched.columns


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


def test_attach_company_description_coalesces_and_flags():
    price = pl.DataFrame(
        {
            "KYGVKEY_final": ["1000", "2000", "3000"],
            "HCIK": [" 123456789 ", "", None],
            "CIK_final": ["7777777777", "8888888888", None],
        }
    )

    comp_desc = pl.DataFrame(
        {
            "KYGVKEY": ["1000", "2000", "3000"],
            "CIK": ["123456789", " 000012345 ", "54321.0"],
        }
    )

    attached = attach_company_description(price.lazy(), comp_desc.lazy()).collect().sort("KYGVKEY_final")

    assert all(v & int(DataStatus.HAS_COMP_DESC) for v in attached.select("data_status").to_series().to_list())
    assert attached.select("CIK_final").to_series().to_list() == [
        "0123456789",
        "8888888888",
        "0000054321",
    ]
    assert (
        attached.filter(pl.col("KYGVKEY_final") == "2000")
        .select("HCIK_10")
        .item()
        is None
    )


def test_merge_histories_keeps_rows_and_sets_attempt_bits(tmp_path: Path):
    price = pl.DataFrame(
        {
            "KYPERMNO": [1, 2],
            "CALDT": [dt.date(2024, 1, 2), dt.date(2024, 1, 2)],
            "KYGVKEY_final": ["1000", None],
            "LIID": ["A", None],
            "data_status": [0, 0],
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

    output_path = merge_histories(price.lazy(), sec_hist.lazy(), comp_hist.lazy(), tmp_path)
    merged = pl.read_parquet(output_path).sort("KYPERMNO")

    assert "join1_status" not in merged.columns
    assert "join2_status" not in merged.columns

    with_keys = merged.filter(pl.col("KYGVKEY_final").is_not_null()).select("data_status").item()
    without_keys = merged.filter(pl.col("KYGVKEY_final").is_null()).select("data_status").item()

    assert with_keys & int(DataStatus.SECHIST_VALID)
    assert with_keys & int(DataStatus.COMPHIST_VALID)
    assert with_keys & int(DataStatus.SECHIST_ATTEMPTED)
    assert with_keys & int(DataStatus.COMPHIST_ATTEMPTED)
    assert without_keys == int(DataStatus.NONE)


def test_merge_histories_open_ended_segments_are_valid(tmp_path: Path):
    price = pl.DataFrame(
        {
            "KYPERMNO": [1],
            "CALDT": [dt.date(2024, 2, 1)],
            "KYGVKEY_final": ["1000"],
            "LIID": ["A"],
        }
    )

    sec_hist = pl.DataFrame(
        {
            "KYGVKEY": ["1000"],
            "KYIID": ["A"],
            "HSCHGDT": [dt.date(2024, 1, 1)],
            "HSCHGENDDT": [None],
            "HTPCI": ["USA"],
            "HEXCNTRY": ["US"],
        }
    )

    comp_hist = pl.DataFrame(
        {
            "KYGVKEY": ["1000"],
            "HCHGDT": [dt.date(2023, 1, 1)],
            "HCHGENDDT": [None],
            "HCIK": ["CIK1"],
            "HSIC": [100],
            "HNAICS": ["1111"],
            "HGSUBIND": ["Sub"],
        }
    )

    output_path = merge_histories(price.lazy(), sec_hist.lazy(), comp_hist.lazy(), tmp_path)
    merged = pl.read_parquet(output_path)
    status = merged.select("data_status").item()

    assert status & int(DataStatus.COMPHIST_VALID)
    assert status & int(DataStatus.SECHIST_VALID)
    assert merged.select("HCIK").item() == "CIK1"
    assert merged.select("HTPCI").item() == "USA"


def test_merge_histories_masks_stale_segments(tmp_path: Path):
    price = pl.DataFrame(
        {
            "KYPERMNO": [1],
            "CALDT": [dt.date(2024, 2, 1)],
            "KYGVKEY_final": ["1000"],
            "LIID": ["A"],
        }
    )

    sec_hist = pl.DataFrame(
        {
            "KYGVKEY": ["1000"],
            "KYIID": ["A"],
            "HSCHGDT": [dt.date(2024, 1, 1)],
            "HSCHGENDDT": [dt.date(2024, 1, 15)],
            "HTPCI": ["USA"],
            "HEXCNTRY": ["US"],
        }
    )

    comp_hist = pl.DataFrame(
        {
            "KYGVKEY": ["1000"],
            "HCHGDT": [dt.date(2023, 1, 1)],
            "HCHGENDDT": [dt.date(2023, 12, 31)],
            "HCIK": ["CIK1"],
            "HSIC": [100],
            "HNAICS": ["1111"],
            "HGSUBIND": ["Sub"],
        }
    )

    output_path = merge_histories(price.lazy(), sec_hist.lazy(), comp_hist.lazy(), tmp_path)
    merged = pl.read_parquet(output_path)
    status = merged.select("data_status").item()

    assert status & int(DataStatus.SECHIST_MATCHED)
    assert status & int(DataStatus.SECHIST_STALE)
    assert status & int(DataStatus.COMPHIST_MATCHED)
    assert status & int(DataStatus.COMPHIST_STALE)
    assert status & int(DataStatus.SECHIST_VALID) == 0
    assert status & int(DataStatus.COMPHIST_VALID) == 0
    assert merged.select("HCIK").item() is None
    assert merged.select("HTPCI").item() is None
    assert merged.select("HEXCNTRY").item() is None


def test_merge_histories_still_joins_comp_history_without_liid(tmp_path: Path):
    price = pl.DataFrame(
        {
            "KYPERMNO": [1],
            "CALDT": [dt.date(2024, 2, 1)],
            "KYGVKEY_final": ["1000"],
            "LIID": [None],
        }
    )

    sec_hist = pl.DataFrame(
        {
            "KYGVKEY": ["1000"],
            "KYIID": ["A"],
            "HSCHGDT": [dt.date(2024, 1, 1)],
            "HSCHGENDDT": [dt.date(2024, 12, 31)],
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

    output_path = merge_histories(price.lazy(), sec_hist.lazy(), comp_hist.lazy(), tmp_path)
    merged = pl.read_parquet(output_path)
    status = merged.select("data_status").item()

    assert status & int(DataStatus.COMPHIST_VALID)
    assert status & int(DataStatus.SECHIST_ATTEMPTED) == 0
    assert status & int(DataStatus.SECHIST_CAN_ATTEMPT) == 0
    assert merged.select("HCIK").item() == "CIK1"
    assert merged.select("HNAICS").item() == "1111"
    assert merged.select("HTPCI").item() is None


def test_merge_histories_preserves_uint64_roundtrip(tmp_path: Path):
    price = pl.DataFrame(
        {
            "KYPERMNO": [1],
            "CALDT": [dt.date(2024, 1, 2)],
            "KYGVKEY_final": ["1000"],
            "LIID": ["A"],
            "data_status": [int(DataStatus.HAS_RET)],
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

    output_path = merge_histories(price.lazy(), sec_hist.lazy(), comp_hist.lazy(), tmp_path)

    loaded_schema = pl.scan_parquet(output_path).collect_schema()
    assert loaded_schema["data_status"] == pl.UInt64


def test_merge_histories_grouped_asof_paths_do_not_emit_sortedness_warning(tmp_path: Path):
    price = pl.DataFrame(
        {
            "KYPERMNO": [2, 1],
            "CALDT": [dt.date(2024, 1, 2), dt.date(2024, 1, 2)],
            "KYGVKEY_final": ["2000", "1000"],
            "LIID": ["B", "A"],
            "data_status": [0, 0],
        }
    )

    # Intentionally unsorted source frames to verify internal sorting + grouped asof behavior.
    sec_hist = pl.DataFrame(
        {
            "KYGVKEY": ["2000", "1000"],
            "KYIID": ["B", "A"],
            "HSCHGDT": [dt.date(2023, 12, 31), dt.date(2023, 12, 31)],
            "HSCHGENDDT": [dt.date(2025, 1, 1), dt.date(2025, 1, 1)],
            "HTPCI": ["US", "US"],
            "HEXCNTRY": ["US", "US"],
        }
    )

    comp_hist = pl.DataFrame(
        {
            "KYGVKEY": ["2000", "1000"],
            "HCHGDT": [dt.date(2023, 1, 1), dt.date(2023, 1, 1)],
            "HCHGENDDT": [dt.date(2025, 1, 1), dt.date(2025, 1, 1)],
            "HCIK": ["CIK2", "CIK1"],
            "HSIC": [200, 100],
            "HNAICS": ["2222", "1111"],
            "HGSUBIND": ["Sub2", "Sub1"],
        }
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        merge_histories(price.lazy(), sec_hist.lazy(), comp_hist.lazy(), tmp_path)

    assert not any(
        "Sortedness of columns cannot be checked when 'by' groups provided" in str(w.message)
        for w in caught
    )

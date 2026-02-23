from __future__ import annotations

import datetime as dt

import polars as pl

from thesis_pkg.core.ccm.canonical_links import (
    build_canonical_link_table,
    canonical_link_coverage_metrics,
)
from thesis_pkg.core.ccm.sec_ccm_contracts import MatchReasonCode
from thesis_pkg.core.ccm.sec_ccm_premerge import resolve_links_phase_a


def _empty_linkfiscalperiodall() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "KYGVKEY": pl.Utf8,
            "lpermno": pl.Int32,
            "lpermco": pl.Int32,
            "liid": pl.Utf8,
            "linktype": pl.Utf8,
            "linkprim": pl.Utf8,
            "linkrank": pl.Int32,
            "linkdt": pl.Date,
            "linkenddt": pl.Date,
            "FiscalPeriodCRSPStartDt": pl.Date,
            "FiscalPeriodCRSPEndDt": pl.Date,
        }
    )


def _empty_companydescription() -> pl.DataFrame:
    return pl.DataFrame(schema={"KYGVKEY": pl.Utf8, "CIK": pl.Utf8})


def _canonical_links(rows: list[dict[str, object]]) -> pl.DataFrame:
    base = {
        "cik_10": "0000000000",
        "gvkey": "0",
        "kypermno": 0,
        "lpermco": None,
        "liid": "01",
        "valid_start": dt.date(1900, 1, 1),
        "valid_end": None,
        "link_start": dt.date(1900, 1, 1),
        "link_end": None,
        "cik_start": None,
        "cik_end": None,
        "linktype": "LC",
        "linkprim": "P",
        "link_rank_raw": None,
        "link_rank_effective": 90,
        "link_quality": 4.0,
        "link_source": "linkhistory",
        "source_priority": 1,
        "row_quality_tier": 10,
        "has_window": True,
        "is_sparse_fallback": False,
    }
    return pl.DataFrame([{**base, **row} for row in rows])


def test_missing_rank_history_rows_never_get_zero_effective_rank():
    linkhistory = pl.DataFrame(
        {
            "KYGVKEY": ["1000"],
            "LPERMNO": [1],
            "LPERMCO": [10],
            "LIID": ["01"],
            "LINKDT": [dt.date(2010, 1, 1)],
            "LINKENDDT": [None],
            "LINKTYPE": ["LC"],
            "LINKPRIM": ["P"],
        }
    )
    companyhistory = pl.DataFrame(
        {
            "KYGVKEY": ["1000"],
            "HCHGDT": [dt.date(2010, 1, 1)],
            "HCHGENDDT": [None],
            "HCIK": ["123456789"],
        }
    )

    canonical = build_canonical_link_table(
        linkhistory.lazy(),
        _empty_linkfiscalperiodall().lazy(),
        companyhistory.lazy(),
        _empty_companydescription().lazy(),
    ).collect()

    rank = canonical.select(pl.col("link_rank_effective").min()).item()
    assert rank >= 90
    assert rank != 0


def test_agreement_rows_from_both_sources_are_order_invariant():
    sec = pl.DataFrame(
        {
            "doc_id": ["d1"],
            "cik_10": ["0000000100"],
            "filing_date": [dt.date(2024, 1, 2)],
        }
    )
    rows = [
        {
            "cik_10": "0000000100",
            "gvkey": "1000",
            "kypermno": 7,
            "link_source": "linkhistory",
            "source_priority": 1,
            "link_rank_effective": 90,
        },
        {
            "cik_10": "0000000100",
            "gvkey": "1000",
            "kypermno": 7,
            "link_source": "linkfiscalperiodall",
            "source_priority": 2,
            "link_rank_effective": 1,
            "link_rank_raw": 1,
        },
    ]

    out1 = resolve_links_phase_a(sec.lazy(), _canonical_links(rows).lazy()).collect().row(0, named=True)
    out2 = resolve_links_phase_a(sec.lazy(), _canonical_links(list(reversed(rows))).lazy()).collect().row(0, named=True)

    assert out1["phase_a_reason_code"] == MatchReasonCode.OK.value
    assert out2["phase_a_reason_code"] == MatchReasonCode.OK.value
    assert (out1["gvkey"], out1["kypermno"]) == (out2["gvkey"], out2["kypermno"])


def test_history_wins_over_fiscal_when_both_valid_and_fiscal_only_when_history_invalid():
    sec = pl.DataFrame(
        {
            "doc_id": ["both_valid", "history_invalid"],
            "cik_10": ["0000000200", "0000000200"],
            "filing_date": [dt.date(2024, 6, 1), dt.date(2025, 6, 1)],
        }
    )
    links = _canonical_links(
        [
            {
                "cik_10": "0000000200",
                "gvkey": "2000",
                "kypermno": 20,
                "valid_start": dt.date(2020, 1, 1),
                "valid_end": dt.date(2024, 12, 31),
                "link_start": dt.date(2020, 1, 1),
                "link_end": dt.date(2024, 12, 31),
                "link_source": "linkhistory",
                "source_priority": 1,
                "link_rank_effective": 90,
            },
            {
                "cik_10": "0000000200",
                "gvkey": "2999",
                "kypermno": 29,
                "valid_start": dt.date(2020, 1, 1),
                "valid_end": None,
                "link_start": dt.date(2020, 1, 1),
                "link_end": None,
                "link_source": "linkfiscalperiodall",
                "source_priority": 2,
                "link_rank_effective": 1,
                "link_rank_raw": 1,
            },
        ]
    )

    out = resolve_links_phase_a(sec.lazy(), links.lazy()).collect().sort("doc_id")
    by_doc = {row["doc_id"]: row for row in out.to_dicts()}

    assert by_doc["both_valid"]["phase_a_reason_code"] == MatchReasonCode.OK.value
    assert by_doc["both_valid"]["gvkey"] == "2000"
    assert by_doc["history_invalid"]["phase_a_reason_code"] == MatchReasonCode.OK.value
    assert by_doc["history_invalid"]["gvkey"] == "2999"


def test_cik_window_mismatch_blocks_out_of_window_filings():
    linkhistory = pl.DataFrame(
        {
            "KYGVKEY": ["5000"],
            "LPERMNO": [55],
            "LPERMCO": [555],
            "LIID": ["01"],
            "LINKDT": [dt.date(2000, 1, 1)],
            "LINKENDDT": [None],
            "LINKTYPE": ["LC"],
            "LINKPRIM": ["P"],
        }
    )
    companyhistory = pl.DataFrame(
        {
            "KYGVKEY": ["5000", "5000"],
            "HCHGDT": [dt.date(2000, 1, 1), dt.date(2021, 1, 1)],
            "HCHGENDDT": [dt.date(2020, 12, 31), None],
            "HCIK": ["111", "222"],
        }
    )
    canonical = build_canonical_link_table(
        linkhistory.lazy(),
        _empty_linkfiscalperiodall().lazy(),
        companyhistory.lazy(),
        _empty_companydescription().lazy(),
    )

    sec = pl.DataFrame(
        {
            "doc_id": ["old_ok", "old_bad", "new_ok"],
            "cik_10": ["0000000111", "0000000111", "0000000222"],
            "filing_date": [dt.date(2019, 6, 1), dt.date(2022, 6, 1), dt.date(2022, 6, 1)],
        }
    )
    out = resolve_links_phase_a(sec.lazy(), canonical).collect().sort("doc_id")
    by_doc = {row["doc_id"]: row for row in out.to_dicts()}

    assert by_doc["old_ok"]["phase_a_reason_code"] == MatchReasonCode.OK.value
    assert by_doc["old_ok"]["kypermno"] == 55
    assert by_doc["old_bad"]["phase_a_reason_code"] == MatchReasonCode.CIK_NOT_IN_LINK_UNIVERSE.value
    assert by_doc["new_ok"]["phase_a_reason_code"] == MatchReasonCode.OK.value
    assert by_doc["new_ok"]["kypermno"] == 55


def test_sparse_fallback_rows_only_win_when_no_stage1_candidate_exists():
    sec = pl.DataFrame(
        {
            "doc_id": ["has_complete", "only_sparse"],
            "cik_10": ["0000000300", "0000000400"],
            "filing_date": [dt.date(2024, 1, 2), dt.date(2024, 1, 2)],
        }
    )
    links = _canonical_links(
        [
            {
                "cik_10": "0000000300",
                "gvkey": "3000",
                "kypermno": 30,
                "has_window": True,
                "is_sparse_fallback": False,
                "row_quality_tier": 10,
            },
            {
                "cik_10": "0000000300",
                "gvkey": "3001",
                "kypermno": 31,
                "valid_start": None,
                "valid_end": None,
                "link_start": None,
                "link_end": None,
                "has_window": False,
                "is_sparse_fallback": True,
                "row_quality_tier": 90,
                "link_quality": 0.1,
                "link_rank_effective": 99,
            },
            {
                "cik_10": "0000000400",
                "gvkey": "4001",
                "kypermno": 41,
                "valid_start": None,
                "valid_end": None,
                "link_start": None,
                "link_end": None,
                "has_window": False,
                "is_sparse_fallback": True,
                "row_quality_tier": 90,
                "link_quality": 0.1,
                "link_rank_effective": 99,
            },
        ]
    )

    out = resolve_links_phase_a(sec.lazy(), links.lazy()).collect().sort("doc_id")
    by_doc = {row["doc_id"]: row for row in out.to_dicts()}

    assert by_doc["has_complete"]["phase_a_reason_code"] == MatchReasonCode.OK.value
    assert by_doc["has_complete"]["kypermno"] == 30
    assert by_doc["only_sparse"]["phase_a_reason_code"] == MatchReasonCode.OK.value
    assert by_doc["only_sparse"]["kypermno"] == 41


def test_kypermno_zero_rows_never_win_phase_a():
    sec = pl.DataFrame(
        {
            "doc_id": ["d0"],
            "cik_10": ["0000000500"],
            "filing_date": [dt.date(2024, 1, 2)],
        }
    )
    links = _canonical_links(
        [
            {
                "cik_10": "0000000500",
                "gvkey": "5000",
                "kypermno": 0,
                "row_quality_tier": 90,
            }
        ]
    )

    out = resolve_links_phase_a(sec.lazy(), links.lazy()).collect().row(0, named=True)
    assert out["phase_a_reason_code"] == MatchReasonCode.CIK_NOT_IN_LINK_UNIVERSE.value
    assert out["kypermno"] is None


def test_canonical_retains_rows_without_cik_and_uses_link_window_for_validity():
    linkhistory = pl.DataFrame(
        {
            "KYGVKEY": ["1000", "9999"],
            "LPERMNO": [1, 2],
            "LPERMCO": [10, 20],
            "LIID": ["01", "01"],
            "LINKDT": [dt.date(2015, 1, 1), dt.date(2012, 6, 1)],
            "LINKENDDT": [dt.date(2020, 12, 31), dt.date(2014, 6, 30)],
            "LINKTYPE": ["LC", "LC"],
            "LINKPRIM": ["P", "P"],
        }
    )
    companyhistory = pl.DataFrame(
        {
            "KYGVKEY": ["1000"],
            "HCHGDT": [dt.date(2010, 1, 1)],
            "HCHGENDDT": [None],
            "HCIK": ["123456789"],
        }
    )

    canonical = build_canonical_link_table(
        linkhistory.lazy(),
        _empty_linkfiscalperiodall().lazy(),
        companyhistory.lazy(),
        _empty_companydescription().lazy(),
    ).collect()

    missing_cik_row = canonical.filter(pl.col("gvkey") == "9999").row(0, named=True)
    assert missing_cik_row["cik_10"] is None
    assert missing_cik_row["valid_start"] == dt.date(2012, 6, 1)
    assert missing_cik_row["valid_end"] == dt.date(2014, 6, 30)
    assert missing_cik_row["link_start"] == dt.date(2012, 6, 1)
    assert missing_cik_row["link_end"] == dt.date(2014, 6, 30)


def test_canonical_coverage_metrics_include_missing_cik_counts():
    links = _canonical_links(
        [
            {"cik_10": "0000000100", "gvkey": "1000", "kypermno": 1},
            {"cik_10": None, "gvkey": "2000", "kypermno": 2},
            {"cik_10": None, "gvkey": "3000", "kypermno": 3},
        ]
    )
    metrics = canonical_link_coverage_metrics(links.lazy()).collect().row(0, named=True)

    assert metrics["rows_total"] == 3
    assert metrics["rows_missing_cik"] == 2
    assert metrics["rows_with_cik"] == 1
    assert metrics["distinct_gvkey_missing_cik"] == 2
    assert metrics["distinct_cik_10"] == 1


def test_phase_a_for_cik_present_rows_unchanged_when_cik_missing_rows_exist():
    sec = pl.DataFrame(
        {
            "doc_id": ["d1"],
            "cik_10": ["0000000100"],
            "filing_date": [dt.date(2024, 1, 2)],
        }
    )
    links = _canonical_links(
        [
            {"cik_10": "0000000100", "gvkey": "1000", "kypermno": 1},
            {
                "cik_10": None,
                "gvkey": "9999",
                "kypermno": 9,
                "valid_start": dt.date(2020, 1, 1),
                "valid_end": None,
                "link_start": dt.date(2020, 1, 1),
                "link_end": None,
            },
        ]
    )

    out = resolve_links_phase_a(sec.lazy(), links.lazy()).collect().row(0, named=True)
    assert out["phase_a_reason_code"] == MatchReasonCode.OK.value
    assert out["gvkey"] == "1000"
    assert out["kypermno"] == 1

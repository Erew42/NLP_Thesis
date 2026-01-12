import zipfile
from pathlib import Path

import polars as pl
import pytest

from thesis_pkg.filing_text import (
    parse_filename_minimal,
    extract_filing_items,
    process_zip_year_raw_text,
    process_zip_year,
    concat_parquets_arrow,
    merge_yearly_batches,
    process_year_parquet_extract_items,
    compute_no_item_diagnostics,
    aggregate_no_item_stats_csvs,
    ParsedFilingSchema,
    RawTextSchema,
)


def test_parse_filename_minimal_extracts_fields():
    filename = "20240131_10-K_edgar_data_1234_0001234-24-000001.txt"
    meta = parse_filename_minimal(filename)

    assert meta["filename_parse_ok"] is True
    assert meta["file_date_filename"] == "20240131"
    assert meta["document_type_filename"] == "10-K"
    assert meta["cik"] == 1234
    assert meta["cik_10"] == "0000001234"
    assert meta["accession_nodash"] == "000123424000001"
    assert meta["doc_id"] == "0000001234:0001234-24-000001"


def test_process_zip_year_raw_text_writes_batches(tmp_path: Path):
    zip_path = tmp_path / "sample.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("20240101_10-K_edgar_data_1234_0001234-24-000001.txt", "alpha")
        zf.writestr("20240102_8-K_edgar_data_1234_0001234-24-000002.txt", "bravo")

    out_dir = tmp_path / "out"
    batches = process_zip_year_raw_text(
        zip_path,
        out_dir,
        batch_max_rows=1,
        batch_max_text_bytes=10**9,
        tmp_dir=tmp_path,
    )

    assert len(batches) == 2
    assert all(p.exists() for p in batches)

    first = pl.read_parquet(batches[0])
    assert set(first.columns) == set(RawTextSchema.schema)
    assert first["file_date_filename"].dtype == pl.Date
    assert first["filename_parse_ok"].item() is True
    assert first["full_text"].item() == "alpha"
    assert first["doc_id"].item().endswith(":0001234-24-000001")


def test_process_zip_year_parses_headers_and_conflicts(tmp_path: Path):
    zip_path = tmp_path / "headers.zip"
    text_ok = """ACCESSION NUMBER: 0001234-24-000003
CENTRAL INDEX KEY: 00001234
FILED AS OF DATE: 20240103
"""
    text_cik_conflict = """ACCESSION NUMBER: 0009999-24-000009
CENTRAL INDEX KEY: 00009999
FILED AS OF DATE: 20240105
"""

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("20240103_10-K_edgar_data_1234_0001234-24-000003.txt", text_ok)
        zf.writestr("20240105_10-K_edgar_data_1234_0009999-24-000009.txt", text_cik_conflict)

    out_dir = tmp_path / "out_headers"
    batches = process_zip_year(
        zip_path=zip_path,
        out_dir=out_dir,
        batch_max_rows=10,
        batch_max_text_bytes=10**9,
        tmp_dir=tmp_path,
        header_search_limit=200,
    )

    assert len(batches) == 1
    df = pl.read_parquet(batches[0]).sort("filename")
    assert set(df.columns) == set(ParsedFilingSchema.schema)
    assert df["filing_date"].dtype == pl.Date
    assert df["file_date_filename"].dtype == pl.Date

    rows = {row["filename"]: row for row in df.to_dicts()}

    ok = rows["20240103_10-K_edgar_data_1234_0001234-24-000003.txt"]
    assert ok["cik_conflict"] is False
    assert ok["accession_conflict"] is False
    assert ok["doc_id"] == "0000001234:0001234-24-000003"
    assert ok["ciks_header_secondary"] == []

    conflict = rows["20240105_10-K_edgar_data_1234_0009999-24-000009.txt"]
    assert conflict["cik_conflict"] is True
    assert conflict["accession_conflict"] is False
    assert conflict["filing_date"].isoformat() == "2024-01-05"


def test_merge_yearly_batches_stages_and_merges(tmp_path: Path):
    batch_root = tmp_path / "batches"
    year_dir = batch_root / "2012"
    year_dir.mkdir(parents=True, exist_ok=True)

    df1 = pl.DataFrame({"doc_id": ["a", "b"], "full_text": ["alpha", "bravo"]})
    df2 = pl.DataFrame({"doc_id": ["c"], "full_text": ["charlie"]})

    df1.write_parquet(year_dir / "2012_batch_0001.parquet")
    df2.write_parquet(year_dir / "2012_batch_0002.parquet")

    out_dir = tmp_path / "year_merged"
    work_dir = tmp_path / "work"
    merged = merge_yearly_batches(
        batch_dir=batch_root,
        out_dir=out_dir,
        local_work_dir=work_dir,
        validate_inputs=False,
        stage_inputs_locally=True,
    )

    assert merged == [out_dir / "2012.parquet"]
    assert merged[0].exists()
    assert pl.read_parquet(merged[0]).height == 3


def test_concat_parquets_arrow_reports_bad_file(tmp_path: Path):
    good = tmp_path / "good.parquet"
    bad = tmp_path / "bad.parquet"
    out = tmp_path / "out.parquet"

    df = pl.DataFrame({"doc_id": ["a", "b"], "full_text": ["alpha", "bravo"]})
    df.write_parquet(good)
    df.write_parquet(bad)

    data = bytearray(bad.read_bytes())
    assert data[:4] == b"PAR1"
    assert data[-4:] == b"PAR1"
    mid = len(data) // 2
    for i in range(mid, min(mid + 32, len(data) - 4)):
        data[i] ^= 0xFF
    data[:4] = b"PAR1"
    data[-4:] = b"PAR1"
    bad.write_bytes(data)

    try:
        concat_parquets_arrow([good, bad], out)
    except OSError as exc:
        assert bad.name in str(exc)
    else:
        raise AssertionError("Expected concat_parquets_arrow to fail on a corrupted parquet.")


def test_extract_filing_items_handles_toc_and_part_collisions():
    text = """<Header>
<SEC-Header>
ACCESSION NUMBER: 0000000-00-000000
</SEC-Header>
</Header>

TABLE OF CONTENTS
PART I
Item 1. Financial Statements
Item 2. Management's Discussion and Analysis
Item 3. Quantitative and Qualitative Disclosures About Market Risk
Item 4. Controls and Procedures
PART II
Item 1. Legal Proceedings
Item 1A. Risk Factors
Item 2. Unregistered Sales of Equity Securities and Use of Proceeds
Item 3. Defaults on Senior Securities
Item 4. Mine Safety Disclosures
Item 5. Other Information
Item 6. Exhibits

FILLER
FILLER
FILLER
FILLER
FILLER
FILLER
FILLER
FILLER

PART I
ITEM 1. Financial Statements.
Alpha
3

ITEM 2. Management's Discussion and Analysis.
Bravo

ITEM 3. Quantitative and Qualitative Disclosures About Market Risk.
Charlie

ITEM 4. Controls and Procedures.
Delta

PART II: OTHER INFORMATION
ITEM 1. Legal Proceedings.
None.

ITEM 1A. Risk Factors.
Not applicable.

ITEM 2. Unregistered Sales of Equity Securities and Use of Proceeds.
Echo

ITEM 3. Defaults on Senior Securities.
None

ITEM 4. Mine Safety Disclosures.
None.

ITEM 5. Other Information.
Foxtrot

ITEM 6. Exhibits.
Golf
"""
    items = extract_filing_items(text)
    keys = [it["item"] for it in items]

    assert keys == [
        "I:1",
        "I:2",
        "I:3",
        "I:4",
        "II:1",
        "II:1A",
        "II:2",
        "II:3",
        "II:4",
        "II:5",
        "II:6",
    ]
    # Pagination cleanup: the standalone page number line should be removed from the item text.
    first = next(it for it in items if it["item"] == "I:1")["full_text"] or ""
    assert "\n3\n" not in f"\n{first}\n"


def test_extract_filing_items_handles_inline_toc_on_one_line():
    pad = "X" * 9000
    text = (
        "<Header></Header>\n"
        "TABLE OF CONTENTS PAGE PART I ITEM 1. BUSINESS 3 ITEM 2. PROPERTIES 4 ITEM 3. LEGAL PROCEEDINGS 5 ITEM 4. CONTROLS 6 "
        "PART I ITEM 1. BUSINESS Actual business text. "
        "ITEM 2. PROPERTIES Actual properties text."
        + pad
    )
    items = extract_filing_items(text)
    keys = [it["item"] for it in items]
    assert keys == ["I:1", "I:2"]
    assert "Actual business text" in (items[0]["full_text"] or "")


def test_extract_filing_items_treats_moderately_long_lines_as_sparse_layout():
    # Regression: some filings have very long lines (e.g., ~6-7k chars) but still below the 8k sparse cutoff.
    # Headings can appear mid-line (e.g., "... PART I ITEM 1 ..."), so we need the sparse-layout heuristics.
    pad = "X" * 6000
    text = (
        "<Header></Header>\n"
        "CONTENTS PAGE PART I ITEM 1. BUSINESS 3 ITEM 2. PROPERTIES 4 "
        "PART I ITEM 1. BUSINESS Actual business text. "
        "ITEM 2. PROPERTIES Actual properties text. "
        + pad
    )
    items = extract_filing_items(text)
    assert [it["item"] for it in items] == ["I:1", "I:2"]
    assert "Actual business text" in (items[0]["full_text"] or "")


def test_extract_filing_items_does_not_mask_real_items_on_toc_page_header():
    # Regression: repeated "Table of Contents" page headers can cause TOC masking to accidentally
    # suppress real item headings (and yield zero extracted items).
    text = """<Header></Header>
Item 6. Selected Financial Data.
Not applicable.
Table of Contents
Item 7. Management's Discussion and Analysis.
Alpha
Item 7A. Quantitative and Qualitative Disclosures About Market Risk.
Bravo
Item 8. Financial Statements and Supplementary Data.
Charlie
Item 9. Changes in and Disagreements with Accountants.
Delta
"""
    items = extract_filing_items(text)
    assert [it["item"] for it in items] == ["6", "7", "7A", "8", "9"]


def test_extract_filing_items_continued_filter_is_local_to_heading():
    # Regression: in very long lines, the word "continued" can appear later in the narrative;
    # it should not cause all headings on that line to be rejected.
    text = (
        "<Header></Header>\n"
        "PART I ITEM 1. Business Alpha. "
        + ("X" * 2500)
        + ". ITEM 2. Properties Bravo. "
        + ("Y" * 2500)
        + "We continued to grow."
    )
    items = extract_filing_items(text)
    assert [it["item"] for it in items] == ["I:1", "I:2"]


def test_extract_filing_items_normalizes_roman_item_numbers():
    text = """<Header></Header>
PART I
ITEM I. BUSINESS
Alpha
-7-

ITEM 2. PROPERTIES
Bravo
"""
    items = extract_filing_items(text)
    assert [it["item"] for it in items] == ["I:1", "I:2"]
    assert "-7-" not in (items[0]["full_text"] or "")


def test_extract_filing_items_handles_wrapped_part_and_item_headings():
    # Simulate common PDF/HTML-to-text wrapping where headings are split across lines.
    text = """<Header></Header>
PART
I
ITEM
1.
Business
Alpha

ITEM
2.
Properties
Bravo
"""
    items = extract_filing_items(text)
    assert [it["item"] for it in items] == ["I:1", "I:2"]
    assert "Alpha" in (items[0]["full_text"] or "")
    assert "Bravo" in (items[1]["full_text"] or "")


def test_extract_filing_items_does_not_treat_medicare_part_d_as_filing_part():
    text = """<Header></Header>
PART II
ITEM 1. Legal Proceedings
This section discusses Medicare Part D coverage and other matters.
ITEM 2. Other Information
More text.
"""
    items = extract_filing_items(text, form_type="10-Q")
    assert [it["item"] for it in items] == ["II:1", "II:2"]


def _build_filings_df():
    text_with_items = "ITEM 1. Business\nAlpha\nITEM 2. Properties\nBravo"
    text_no_items = "no relevant items here"
    df = pl.DataFrame(
        [
            {
                "doc_id": "doc1",
                "cik": 1,
                "cik_10": "0000000001",
                "accession_number": "0001-01-000001",
                "accession_nodash": "000101000001",
                "file_date_filename": "20240101",
                "document_type_filename": "10-K",
                "filename": "f1.txt",
                "full_text": text_with_items,
            },
            {
                "doc_id": "doc2",
                "cik": 1,
                "cik_10": "0000000001",
                "accession_number": "0001-01-000002",
                "accession_nodash": "000101000002",
                "file_date_filename": "20240101",
                "document_type_filename": "10-K",
                "filename": "f2.txt",
                "full_text": "",
            },
            {
                "doc_id": "doc3",
                "cik": 1,
                "cik_10": "0000000001",
                "accession_number": "0001-01-000003",
                "accession_nodash": "000101000003",
                "file_date_filename": "20240101",
                "document_type_filename": "8-K",
                "filename": "f3.txt",
                "full_text": text_no_items,
            },
            {
                "doc_id": "doc4",
                "cik": 1,
                "cik_10": "0000000001",
                "accession_number": None,
                "accession_nodash": None,
                "file_date_filename": "20240101",
                "document_type_filename": "10-K",
                "filename": "f4.txt",
                "full_text": "",
            },
        ]
    )
    return df, text_with_items, text_no_items


def test_process_year_parquet_extract_items_no_item_diagnostics(tmp_path: Path):
    df, text_with_items, text_no_items = _build_filings_df()
    year_path = tmp_path / "2024.parquet"
    df.write_parquet(year_path)

    out_dir = tmp_path / "out"
    process_year_parquet_extract_items(
        year_parquet_path=year_path,
        out_dir=out_dir,
        parquet_batch_rows=2,
        out_batch_max_rows=10,
        out_batch_max_text_bytes=10**6,
        tmp_dir=tmp_path,
        local_work_dir=tmp_path / "work",
        non_item_diagnostic=True,
        include_full_text=False,
    )

    no_item_path = out_dir / "2024_no_item_filings.parquet"
    stats_path = out_dir / "2024_no_item_stats.csv"
    assert no_item_path.exists()
    assert stats_path.exists()

    no_item_df = pl.read_parquet(no_item_path)
    assert "full_text" not in no_item_df.columns
    assert set(no_item_df["accession_number"].to_list()) == {
        "0001-01-000002",
        "0001-01-000003",
    }

    stats = pl.read_csv(stats_path)
    row_10k = stats.filter(pl.col("document_type_filename") == "10-K").to_dicts()[0]
    row_8k = stats.filter(pl.col("document_type_filename") == "8-K").to_dicts()[0]
    row_total = stats.filter(pl.col("document_type_filename") == "TOTAL").to_dicts()[0]

    assert row_10k["n_filings_eligible"] == 2
    assert row_10k["n_with_items"] == 1
    assert row_10k["n_no_items"] == 1
    assert row_10k["share_no_item"] == pytest.approx(0.5)
    assert row_10k["avg_text_len_with_items"] == pytest.approx(len(text_with_items))
    assert row_10k["avg_text_len_no_items"] == pytest.approx(0.0)

    assert row_8k["n_filings_eligible"] == 1
    assert row_8k["n_with_items"] == 0
    assert row_8k["n_no_items"] == 1
    assert row_8k["share_no_item"] == pytest.approx(1.0)
    assert row_8k["avg_text_len_with_items"] == pytest.approx(0.0)
    assert row_8k["avg_text_len_no_items"] == pytest.approx(len(text_no_items))

    assert row_total["n_filings_eligible"] == 3
    assert row_total["n_with_items"] == 1
    assert row_total["n_no_items"] == 2
    assert row_total["share_no_item"] == pytest.approx(2 / 3)
    assert row_total["avg_text_len_with_items"] == pytest.approx(len(text_with_items))
    assert row_total["avg_text_len_no_items"] == pytest.approx(len(text_no_items) / 2)


def test_compute_no_item_diagnostics_anti_join(tmp_path: Path):
    df, text_with_items, text_no_items = _build_filings_df()
    filings_path = tmp_path / "2024.parquet"
    df.write_parquet(filings_path)

    items_path = tmp_path / "2024_items.parquet"
    pl.DataFrame(
        {
            "accession_number": ["0001-01-000001"],
            "doc_id": ["doc1"],
            "item": ["1"],
        }
    ).write_parquet(items_path)

    out_no_item = tmp_path / "no_item_filings.parquet"
    out_stats = tmp_path / "no_item_stats.csv"
    compute_no_item_diagnostics(
        filings_path,
        items_path,
        out_no_item,
        out_stats,
        include_full_text=False,
    )

    no_item_df = pl.read_parquet(out_no_item)
    assert "full_text" not in no_item_df.columns
    assert set(no_item_df["accession_number"].to_list()) == {
        "0001-01-000002",
        "0001-01-000003",
    }

    stats = pl.read_csv(out_stats)
    row_10k = stats.filter(pl.col("document_type_filename") == "10-K").to_dicts()[0]
    row_total = stats.filter(pl.col("document_type_filename") == "TOTAL").to_dicts()[0]

    assert row_10k["n_filings_eligible"] == 2
    assert row_10k["n_with_items"] == 1
    assert row_10k["n_no_items"] == 1
    assert row_10k["avg_text_len_with_items"] == pytest.approx(len(text_with_items))
    assert row_10k["avg_text_len_no_items"] == pytest.approx(0.0)

    assert row_total["n_filings_eligible"] == 3
    assert row_total["n_no_items"] == 2
    assert row_total["avg_text_len_no_items"] == pytest.approx(len(text_no_items) / 2)


def test_aggregate_no_item_stats_csvs_weighted(tmp_path: Path):
    stats_2020 = tmp_path / "2020_no_item_stats.csv"
    stats_2021 = tmp_path / "2021_no_item_stats.csv"

    rows_2020 = [
        {
            "year": "2020",
            "document_type_filename": "10-K",
            "n_filings_eligible": 3,
            "n_with_items": 2,
            "n_no_items": 1,
            "share_no_item": 1 / 3,
            "avg_text_len_with_items": 100.0,
            "avg_text_len_no_items": 50.0,
        },
        {
            "year": "2020",
            "document_type_filename": "8-K",
            "n_filings_eligible": 2,
            "n_with_items": 0,
            "n_no_items": 2,
            "share_no_item": 1.0,
            "avg_text_len_with_items": 0.0,
            "avg_text_len_no_items": 30.0,
        },
        {
            "year": "2020",
            "document_type_filename": "TOTAL",
            "n_filings_eligible": 5,
            "n_with_items": 2,
            "n_no_items": 3,
            "share_no_item": 0.6,
            "avg_text_len_with_items": 100.0,
            "avg_text_len_no_items": (50.0 + 60.0) / 3,
        },
    ]
    rows_2021 = [
        {
            "year": "2021",
            "document_type_filename": "10-K",
            "n_filings_eligible": 3,
            "n_with_items": 1,
            "n_no_items": 2,
            "share_no_item": 2 / 3,
            "avg_text_len_with_items": 200.0,
            "avg_text_len_no_items": 150.0,
        },
        {
            "year": "2021",
            "document_type_filename": "TOTAL",
            "n_filings_eligible": 3,
            "n_with_items": 1,
            "n_no_items": 2,
            "share_no_item": 2 / 3,
            "avg_text_len_with_items": 200.0,
            "avg_text_len_no_items": 150.0,
        },
    ]

    pl.DataFrame(rows_2020).write_csv(stats_2020)
    pl.DataFrame(rows_2021).write_csv(stats_2021)

    out_stats = tmp_path / "all_no_item_stats.csv"
    aggregate_no_item_stats_csvs([stats_2020, stats_2021], out_stats)

    df = pl.read_csv(out_stats)
    row_10k = df.filter(pl.col("document_type_filename") == "10-K").to_dicts()[0]
    row_8k = df.filter(pl.col("document_type_filename") == "8-K").to_dicts()[0]
    row_total = df.filter(pl.col("document_type_filename") == "TOTAL").to_dicts()[0]

    assert row_10k["year"] == "ALL"
    assert row_10k["n_filings_eligible"] == 6
    assert row_10k["n_with_items"] == 3
    assert row_10k["n_no_items"] == 3
    assert row_10k["share_no_item"] == pytest.approx(0.5)
    assert row_10k["avg_text_len_with_items"] == pytest.approx((2 * 100 + 200) / 3)
    assert row_10k["avg_text_len_no_items"] == pytest.approx((50 + 300) / 3)

    assert row_8k["n_filings_eligible"] == 2
    assert row_8k["n_with_items"] == 0
    assert row_8k["n_no_items"] == 2
    assert row_8k["share_no_item"] == pytest.approx(1.0)
    assert row_8k["avg_text_len_no_items"] == pytest.approx(30.0)

    assert row_total["n_filings_eligible"] == 8
    assert row_total["n_with_items"] == 3
    assert row_total["n_no_items"] == 5
    assert row_total["share_no_item"] == pytest.approx(5 / 8)
    assert row_total["avg_text_len_no_items"] == pytest.approx(82.0)

import datetime as dt
import zipfile
from pathlib import Path

import polars as pl
import pytest

import thesis_pkg.pipelines.sec_pipeline as sec_pipeline
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
CONFORMED PERIOD OF REPORT: 20231231
"""
    text_cik_conflict = """ACCESSION NUMBER: 0009999-24-000009
CENTRAL INDEX KEY: 00009999
FILED AS OF DATE: 20240105
CONFORMED PERIOD OF REPORT: 20231230
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
    assert df["period_end"].dtype == pl.Date

    rows = {row["filename"]: row for row in df.to_dicts()}

    ok = rows["20240103_10-K_edgar_data_1234_0001234-24-000003.txt"]
    assert ok["cik_conflict"] is False
    assert ok["accession_conflict"] is False
    assert ok["doc_id"] == "0000001234:0001234-24-000003"
    assert ok["ciks_header_secondary"] == []
    assert ok["period_end"].isoformat() == "2023-12-31"

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


def test_extract_filing_items_numeric_dot_headings_fallback_10k():
    text = """<Header></Header>
1. BUSINESS
Alpha

1A. RISK FACTORS
Bravo

2. PROPERTIES
Charlie
"""
    items = extract_filing_items(text, form_type="10-K")
    assert [it["item_id"] for it in items] == ["1", "1A", "2"]
    assert "Alpha" in (items[0]["full_text"] or "")


def test_extract_filing_items_title_only_headings_fallback_10k():
    text = """<Header></Header>
RISK FACTORS
Alpha

MANAGEMENT'S DISCUSSION AND ANALYSIS
Bravo

FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA
Charlie

SIGNATURES
Delta
"""
    items = extract_filing_items(text, form_type="10-K")
    assert [it["item_id"] for it in items] == ["1A", "7", "8", "SIGNATURES"]
    assert "Bravo" in (items[1]["full_text"] or "")


def test_extract_filing_items_regime_blocks_pre_2005_title_fallback():
    text = """<Header></Header>
BUSINESS
Alpha

RISK FACTORS
Bravo

UNRESOLVED STAFF COMMENTS
Charlie
"""
    items = extract_filing_items(text, form_type="10-K", filing_date="20040101")
    assert [it["item_id"] for it in items] == ["1"]


def test_extract_filing_items_regime_allows_modern_title_fallback():
    text = """<Header></Header>
RISK FACTORS
Alpha

UNRESOLVED STAFF COMMENTS
Bravo

CYBERSECURITY
Charlie
"""
    items = extract_filing_items(
        text,
        form_type="10-K",
        filing_date="20240201",
        period_end="20231231",
    )
    assert [it["item_id"] for it in items] == ["1A", "1B", "1C"]


def test_extract_filing_items_item4_reserved_regime():
    text = """<Header></Header>
ITEM 4. Reserved.
Alpha
"""
    items = extract_filing_items(text, form_type="10-K", filing_date="20110630")
    assert items[0]["canonical_item"] == "I:4_RESERVED"
    assert items[0]["item_status"] == "reserved"
    assert items[0]["exists_by_regime"] is True


def test_extract_filing_items_item4_mine_safety_regime():
    text = """<Header></Header>
ITEM 4. Mine Safety Disclosures.
Alpha
"""
    items = extract_filing_items(text, form_type="10-K", filing_date="20130201")
    assert items[0]["canonical_item"] == "I:4_MINE_SAFETY"
    assert items[0]["exists_by_regime"] is True


def test_extract_filing_items_item1c_period_end_gate():
    text = """<Header></Header>
ITEM 1C. Cybersecurity.
Alpha
"""
    items = extract_filing_items(
        text,
        form_type="10-K",
        filing_date="20240120",
        period_end="20221231",
    )
    assert items[0]["item_id"] == "1C"
    assert items[0]["exists_by_regime"] is False


def test_extract_filing_items_drop_impossible():
    text = """<Header></Header>
ITEM 1C. Cybersecurity.
Alpha
"""
    items = extract_filing_items(
        text,
        form_type="10-K",
        filing_date="20240120",
        period_end="20221231",
        drop_impossible=True,
    )
    assert items == []


def test_extract_filing_items_skips_amended_forms():
    text = """<Header></Header>
ITEM 1. Business
Alpha
"""
    assert extract_filing_items(text, form_type="10-K-A") == []
    assert extract_filing_items(text, form_type="10-Q-A") == []
    assert extract_filing_items(text, form_type="10-K/A") == []


def test_extract_filing_items_item14_legacy_exhibits_pre_2002():
    text = """<Header></Header>
PART IV
ITEM 14. EXHIBITS, FINANCIAL STATEMENT SCHEDULES, AND REPORTS ON FORM 8-K.
Alpha
"""
    items = extract_filing_items(text, form_type="10-K", filing_date="20010115")
    assert items[0]["item_part"] == "IV"
    assert items[0]["canonical_item"] == "IV:14_EXHIBITS_SCHEDULES_REPORTS"
    assert items[0]["exists_by_regime"] is True


def test_extract_filing_items_item14_fees_modern():
    text = """<Header></Header>
PART III
ITEM 14. Principal Accountant Fees and Services.
Alpha
"""
    items = extract_filing_items(text, form_type="10-K", filing_date="20180201")
    assert items[0]["item_part"] == "III"
    assert items[0]["canonical_item"] == "III:14_PRINCIPAL_ACCOUNTANT_FEES"
    assert items[0]["exists_by_regime"] is True


def test_extract_filing_items_item16_fees_legacy_2003():
    text = """<Header></Header>
PART III
ITEM 16. Principal Accountant Fees and Services.
Alpha
"""
    items = extract_filing_items(text, form_type="10-K", filing_date="20030630")
    assert items == []


def test_extract_filing_items_strips_edgar_headers():
    text = """<SEC-HEADER>
ITEM 2. PROPERTIES
</SEC-HEADER>
<FileStats><XML_Chars>123</XML_Chars></FileStats>
ITEM 1. BUSINESS
Alpha
"""
    items = extract_filing_items(text, form_type="10-K")
    assert [it["item_id"] for it in items] == ["1"]
    assert "SEC-HEADER" not in (items[0]["full_text"] or "")


def test_extract_filing_items_rejects_toc_item_lines():
    text = """<Header></Header>
TABLE OF CONTENTS
ITEM 1. BUSINESS..................................3
ITEM 2. PROPERTIES................................4

ITEM 1. BUSINESS
Alpha

ITEM 2. PROPERTIES
Bravo
"""
    items = extract_filing_items(text, form_type="10-K")
    assert [it["item_id"] for it in items] == ["1", "2"]


def test_extract_filing_items_late_item_rejects_toc_restart():
    text = """<Header></Header>
TABLE OF CONTENTS
ITEM 13. CERTAIN RELATIONSHIPS AND RELATED TRANSACTIONS
PART I
ITEM 1. BUSINESS
Alpha

PART III
ITEM 13. CERTAIN RELATIONSHIPS AND RELATED TRANSACTIONS
Real item 13 content.
"""
    items = extract_filing_items(text, form_type="10-K", diagnostics=True)
    item_13 = next(it for it in items if it["item_id"] == "13")
    first_line = (item_13["full_text"] or "").lstrip().splitlines()[0]
    assert "REAL ITEM 13 CONTENT" in (item_13["full_text"] or "").upper()
    assert "PART I" not in first_line.upper()
    assert "ITEM 1" not in first_line.upper()


def test_extract_filing_items_summary_block_skips_toc_start():
    text = """<Header></Header>
FORM 10-K SUMMARY
ITEM 1. BUSINESS
ITEM 1A. RISK FACTORS
ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS
ITEM 16. FORM 10-K SUMMARY
PART I
ITEM 1. BUSINESS
Real business content.

PART IV
ITEM 16. FORM 10-K SUMMARY
Actual summary content.
"""
    items = extract_filing_items(text, form_type="10-K", diagnostics=True)
    assert "16" not in {it["item_id"] for it in items}
    item_1 = next(it for it in items if it["item_id"] == "1")
    item_1_text = item_1["full_text"] or ""
    assert "REAL BUSINESS CONTENT" in item_1_text.upper()
    assert "FORM 10-K SUMMARY" not in item_1_text.upper()


def test_extract_filing_items_forward_looking_toc_blocked():
    text = """<Header></Header>
FORWARD-LOOKING STATEMENTS
TABLE OF CONTENTS
ITEM 1. BUSINESS
ITEM 1A. RISK FACTORS
ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS

PART I
ITEM 1. BUSINESS
Real business content.

ITEM 1A. RISK FACTORS
Risk content.
"""
    items = extract_filing_items(text, form_type="10-K", diagnostics=True)
    assert [it["item_id"] for it in items] == ["1", "1A"]
    item_1_text = next(it for it in items if it["item_id"] == "1")["full_text"] or ""
    assert "REAL BUSINESS CONTENT" in item_1_text.upper()
    assert "FORWARD-LOOKING STATEMENTS" not in item_1_text.upper()


def test_extract_filing_items_skips_cross_ref_prefix():
    text = """<Header></Header>
See Part II Item 8. Financial Statements and Supplementary Data.
ITEM 1. Business
Alpha
ITEM 2. Properties
Bravo
"""
    items = extract_filing_items(text, form_type="10-K")
    assert [it["item_id"] for it in items] == ["1", "2"]


def test_extract_filing_items_mid_sentence_cross_ref_does_not_cut():
    text = """<Header></Header>
PART I
ITEM 1. Business
Alpha
We discuss results; see Part II, Item 7 for more detail.
More Alpha.

PART II
ITEM 7. Management's Discussion and Analysis
Bravo
"""
    items = extract_filing_items(text, form_type="10-K")
    assert [it["item"] for it in items] == ["I:1", "II:7"]
    item1_text = next(it for it in items if it["item_id"] == "1")["full_text"] or ""
    assert "see Part II, Item 7" in item1_text
    assert "Bravo" not in item1_text


def test_extract_filing_items_cross_ref_heading_line_does_not_truncate():
    text = """<Header></Header>
ITEM 1. Business
Alpha
ITEM 7. See Part II, Item 7 for more detail.
Tail text.
"""
    items = extract_filing_items(text, form_type="10-K")
    item1_text = next(it for it in items if it["item_id"] == "1")["full_text"] or ""
    assert "Tail text." in item1_text


def test_heading_hygiene_truncates_after_title_on_prose():
    text = """<Header></Header>
ITEM 8. Financial Statements and Supplementary Data. The consolidated financial statements are included here.
Alpha
"""
    items = extract_filing_items(text, form_type="10-K", diagnostics=True)
    heading_clean = items[0].get("_heading_line") or ""
    heading_raw = items[0].get("_heading_line_raw") or ""
    assert "The consolidated financial statements" in heading_raw
    assert "The consolidated financial statements" not in heading_clean
    assert heading_clean.strip().startswith("ITEM 8.")


def test_heading_hygiene_truncates_incorporated_reference_tail():
    text = """<Header></Header>
ITEM 7A. Quantitative and Qualitative Disclosures About Market Risk is incorporated herein by reference.
Alpha
"""
    items = extract_filing_items(text, form_type="10-K", diagnostics=True)
    heading_clean = items[0].get("_heading_line") or ""
    assert "incorporated herein by reference" not in heading_clean.lower()
    assert "Quantitative and Qualitative Disclosures" in heading_clean


def test_heading_hygiene_keeps_title_only_line():
    text = """<Header></Header>
ITEM 3. Legal Proceedings
Alpha
"""
    items = extract_filing_items(text, form_type="10-K", diagnostics=True)
    heading_clean = items[0].get("_heading_line") or ""
    assert heading_clean.strip() == "ITEM 3. Legal Proceedings"


def test_extract_filing_items_infers_part_when_missing():
    text = """<Header></Header>
ITEM 14. Principal Accountant Fees and Services.
Alpha
"""
    items = extract_filing_items(text, form_type="10-K", filing_date="20180201")
    assert items[0]["item_part"] == "III"
    assert items[0]["item"] == "III:14"


def test_extract_filing_items_glued_letter_suffix():
    text = """<Header></Header>
ITEM 1ARISK FACTORS
Alpha
"""
    items = extract_filing_items(text, form_type="10-K")
    assert [it["item_id"] for it in items] == ["1A"]


def test_extract_filing_items_normalizes_9a_t_suffix():
    text = """<Header></Header>
ITEM 9A(T). Controls and Procedures.
Alpha

ITEM 9B. Other Information.
Bravo
"""
    items = extract_filing_items(text, form_type="10-K")
    assert [it["item_id"] for it in items] == ["9A", "9B"]
    assert (items[0]["full_text"] or "").lstrip().startswith("Controls")


def test_extract_filing_items_repairs_split_letter_line():
    text = """<Header></Header>
ITEM 9
A. Controls and Procedures.
Alpha
"""
    items = extract_filing_items(text, form_type="10-K")
    assert [it["item_id"] for it in items] == ["9A"]


def test_extract_filing_items_repairs_punctuated_split_letter_line():
    text = """<Header></Header>
ITEM 1.
A.
Risk Factors
Alpha
"""
    items = extract_filing_items(text, form_type="10-K")
    assert [it["item_id"] for it in items] == ["1A"]


def test_extract_filing_items_glued_heading_titles_use_base_number():
    text = """<Header></Header>
ITEM 1Business
Alpha

ITEM 2Properties
Bravo

ITEM 9Changes in and Disagreements with Accountants.
Charlie
"""
    items = extract_filing_items(text, form_type="10-K")
    assert [it["item_id"] for it in items] == ["1", "2", "9"]


def test_extract_filing_items_glued_heading_prefix_match():
    text = """<Header></Header>
ITEM 9CHANGES IN AND DISAGREEMENTS WITH ACCOUNTANTS ON ACCOUNTING AND
FINANCIAL DISCLOSURE
Alpha
"""
    items = extract_filing_items(text, form_type="10-K")
    assert [it["item_id"] for it in items] == ["9"]


def test_extract_filing_items_glued_heading_short_prefix():
    text = """<Header></Header>
ITEM 9CHANGES
IN AND DISAGREEMENTS WITH ACCOUNTANTS ON ACCOUNTING AND FINANCIAL DISCLOSURE
Alpha
"""
    items = extract_filing_items(text, form_type="10-K")
    assert [it["item_id"] for it in items] == ["9"]


def test_extract_filing_items_skips_item_is_sentence_starts():
    text = """<Header></Header>
Item is included under the caption Quarterly Data.
ITEM 1. Business
Alpha
"""
    items = extract_filing_items(text, form_type="10-K")
    assert [it["item_id"] for it in items] == ["1"]


def test_extract_filing_items_skips_item_lowercase_sentence_starts():
    text = """<Header></Header>
Item 1 was completed by the contractor on time.
ITEM 1. Business
Alpha
"""
    items = extract_filing_items(text, form_type="10-K")
    assert [it["item_id"] for it in items] == ["1"]


def test_extract_filing_items_skips_item_on_proxy_card():
    text = """<Header></Header>
Item 1 on Proxy Card
ITEM 1. Business
Alpha
"""
    items = extract_filing_items(text, form_type="10-K")
    assert [it["item_id"] for it in items] == ["1"]


def test_extract_filing_items_skips_subitem_parentheses():
    text = """<Header></Header>
Item 16(a), Exhibits, of the Registrant s.
Alpha
"""
    items = extract_filing_items(text, form_type="10-K")
    assert items == []


def test_extract_filing_items_ignores_part_cross_ref_lines():
    text = """<Header></Header>
PART I
ITEM 1. Business
Alpha

Part III, Item 13.

ITEM 2. Properties
Bravo
"""
    items = extract_filing_items(text, form_type="10-K")
    assert [it["item"] for it in items][:2] == ["I:1", "I:2"]


def test_extract_filing_items_skips_toc_entry_with_page_number_next_line():
    text = """<Header></Header>
Item 1C.
19

ITEM 1. Business
Alpha
"""
    items = extract_filing_items(text, form_type="10-K")
    assert [it["item_id"] for it in items] == ["1"]


def test_extract_filing_items_numeric_dot_skips_toc_entries_10k():
    text = """<Header></Header>
TABLE OF CONTENTS
1. BUSINESS..................................3
1A. RISK FACTORS............................5
2. PROPERTIES...............................7

1. BUSINESS
Alpha

1A. RISK FACTORS
Bravo

2. PROPERTIES
Charlie
"""
    items = extract_filing_items(text, form_type="10-K")
    assert [it["item_id"] for it in items] == ["1", "1A", "2"]
    assert "Alpha" in (items[0]["full_text"] or "")


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
                "file_date_filename": dt.date(2024, 1, 1),
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
                "file_date_filename": dt.date(2024, 1, 1),
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
                "file_date_filename": dt.date(2024, 1, 1),
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
                "file_date_filename": dt.date(2024, 1, 1),
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


def test_process_year_parquet_extract_items_fast_alias_uses_v2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    df = pl.DataFrame(
        [
            {
                "doc_id": "doc1",
                "cik": 1,
                "cik_10": "0000000001",
                "accession_number": "0001-01-000001",
                "accession_nodash": "000101000001",
                "file_date_filename": dt.date(2024, 1, 1),
                "filing_date": dt.date(2024, 1, 1),
                "period_end": dt.date(2023, 12, 31),
                "document_type_filename": "10-K",
                "filename": "f1.txt",
                "full_text": "ITEM 1. BUSINESS\\nAlpha",
            }
        ]
    )
    year_path = tmp_path / "2024.parquet"
    df.write_parquet(year_path)

    seen_regimes: list[str | None] = []

    def _fake_extract(*_args, **kwargs):
        seen_regimes.append(kwargs.get("extraction_regime"))
        return [
            {
                "item_part": "I",
                "item_id": "1",
                "item": "I:1",
                "canonical_item": "I:1_BUSINESS",
                "exists_by_regime": True,
                "item_status": "active",
                "full_text": "Alpha",
            }
        ]

    monkeypatch.setattr(sec_pipeline, "extract_filing_items", _fake_extract)

    out_dir = tmp_path / "out"
    out_path = process_year_parquet_extract_items(
        year_parquet_path=year_path,
        out_dir=out_dir,
        parquet_batch_rows=1,
        out_batch_max_rows=10,
        out_batch_max_text_bytes=10**6,
        tmp_dir=tmp_path,
        local_work_dir=tmp_path / "work",
        extraction_regime="fast",
    )

    assert seen_regimes == ["v2"]
    out_df = pl.read_parquet(out_path)
    assert out_df.height == 1
    assert out_df["item"].to_list() == ["I:1"]


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

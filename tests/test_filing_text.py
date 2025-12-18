import zipfile
from pathlib import Path

import polars as pl

from thesis_pkg.filing_text import (
    parse_filename_minimal,
    extract_filing_items,
    process_zip_year_raw_text,
    process_zip_year,
    concat_parquets_arrow,
    merge_yearly_batches,
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

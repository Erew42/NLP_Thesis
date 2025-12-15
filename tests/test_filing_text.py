import zipfile
from pathlib import Path

import polars as pl

from thesis_pkg.filing_text import (
    parse_filename_minimal,
    process_zip_year_raw_text,
    process_zip_year,
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

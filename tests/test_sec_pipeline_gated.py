from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl

from thesis_pkg.pipelines.sec_pipeline import process_year_dir_extract_items_gated


def test_process_year_dir_extract_items_gated_filters_by_doc_id(tmp_path: Path):
    year_dir = tmp_path / "year_dir"
    out_dir = tmp_path / "out_dir"
    tmp_dir = tmp_path / "tmp"
    local_work_dir = tmp_path / "work"
    year_dir.mkdir(parents=True, exist_ok=True)

    text = (
        "ITEM 1. Business\n"
        "We build software.\n\n"
        "ITEM 1A. Risk Factors\n"
        "Some risks.\n\n"
        "ITEM 2. Properties\n"
        "Headquarters details.\n"
    )
    year_df = pl.DataFrame(
        {
            "doc_id": ["d_allow", "d_block"],
            "cik": [1, 2],
            "cik_10": ["0000000001", "0000000002"],
            "accession_number": ["0000000001-24-000001", "0000000002-24-000002"],
            "accession_nodash": ["000000000124000001", "000000000224000002"],
            "file_date_filename": [dt.date(2024, 1, 2), dt.date(2024, 1, 2)],
            "filing_date": [dt.date(2024, 1, 2), dt.date(2024, 1, 2)],
            "period_end": [dt.date(2023, 12, 31), dt.date(2023, 12, 31)],
            "document_type_filename": ["10-K", "10-K"],
            "filename": ["a.txt", "b.txt"],
            "full_text": [text, text],
        }
    )
    year_path = year_dir / "2024.parquet"
    year_df.write_parquet(year_path)

    allowlist = pl.DataFrame({"doc_id": ["d_allow"]})
    out_paths = process_year_dir_extract_items_gated(
        year_dir=year_dir,
        out_dir=out_dir,
        doc_id_allowlist=allowlist,
        tmp_dir=tmp_dir,
        local_work_dir=local_work_dir,
        parquet_batch_rows=2,
        out_batch_max_rows=10_000,
        out_batch_max_text_bytes=16 * 1024 * 1024,
    )

    assert out_paths == [out_dir / "2024.parquet"]
    items = pl.read_parquet(out_paths[0])
    assert items.height > 0
    assert items.select(pl.col("doc_id").n_unique()).item() == 1
    assert items.select(pl.col("doc_id").first()).item() == "d_allow"

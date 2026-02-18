from __future__ import annotations

from pathlib import Path

import polars as pl

from thesis_pkg.core.sec.parquet_stream import (
    discover_input_parquet_files,
    iter_parquet_filing_texts,
)


def _write_parquet(path: Path, rows: list[dict[str, str]]) -> None:
    pl.DataFrame(rows).write_parquet(path)


def test_iter_parquet_filing_texts_accepts_yearly_parquet(tmp_path: Path) -> None:
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    _write_parquet(
        parquet_dir / "2020.parquet",
        [
            {
                "doc_id": "d_2020",
                "accession_number": "0000000000-00-000001",
                "full_text": "Annual report text",
            }
        ],
    )

    rows = list(iter_parquet_filing_texts(parquet_dir, {"d_2020"}))
    assert len(rows) == 1
    assert rows[0]["doc_id"] == "d_2020"
    assert rows[0]["full_text"] == "Annual report text"


def test_discover_input_parquet_files_prefers_batch_files(tmp_path: Path) -> None:
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    _write_parquet(
        parquet_dir / "sample_batch_0001.parquet",
        [
            {
                "doc_id": "d_batch",
                "accession_number": "0000000000-00-000002",
                "full_text": "Batch text",
            }
        ],
    )
    _write_parquet(
        parquet_dir / "2020.parquet",
        [
            {
                "doc_id": "d_year",
                "accession_number": "0000000000-00-000003",
                "full_text": "Year text",
            }
        ],
    )

    discovered = discover_input_parquet_files(parquet_dir)
    assert discovered == [parquet_dir / "sample_batch_0001.parquet"]

    rows = list(iter_parquet_filing_texts(parquet_dir, {"d_year"}))
    assert rows == []


def test_discover_input_parquet_files_ignores_non_year_fallback_names(
    tmp_path: Path,
) -> None:
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    _write_parquet(
        parquet_dir / "sample.parquet",
        [
            {
                "doc_id": "d_sample",
                "accession_number": "0000000000-00-000004",
                "full_text": "Sample text",
            }
        ],
    )
    _write_parquet(
        parquet_dir / "2021.parquet",
        [
            {
                "doc_id": "d_year",
                "accession_number": "0000000000-00-000005",
                "full_text": "Year text",
            }
        ],
    )

    discovered = discover_input_parquet_files(parquet_dir)
    assert discovered == [parquet_dir / "2021.parquet"]

    rows = list(iter_parquet_filing_texts(parquet_dir, {"d_sample", "d_year"}))
    assert len(rows) == 1
    assert rows[0]["doc_id"] == "d_year"

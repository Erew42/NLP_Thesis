from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

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


def test_iter_parquet_filing_texts_closes_parquet_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    _write_parquet(
        parquet_dir / "2021.parquet",
        [
            {
                "doc_id": "d_2021",
                "accession_number": "0000000000-00-000006",
                "full_text": "Year 2021 text",
            }
        ],
    )
    _write_parquet(
        parquet_dir / "2022.parquet",
        [
            {
                "doc_id": "d_2022",
                "accession_number": "0000000000-00-000007",
                "full_text": "Year 2022 text",
            }
        ],
    )

    import thesis_pkg.core.sec.parquet_stream as parquet_stream_mod

    real_parquet_file = parquet_stream_mod.pq.ParquetFile
    closed: list[Path] = []

    class _ParquetFileSpy:
        def __init__(self, path: str | Path):
            self._path = Path(path)
            self._pf = real_parquet_file(path)

        @property
        def schema(self):
            return self._pf.schema

        def iter_batches(self, *args, **kwargs):
            return self._pf.iter_batches(*args, **kwargs)

        def close(self) -> None:
            closed.append(self._path)
            self._pf.close()

    monkeypatch.setattr(parquet_stream_mod.pq, "ParquetFile", _ParquetFileSpy)

    rows = list(iter_parquet_filing_texts(parquet_dir, {"missing_doc"}))
    assert rows == []
    assert set(closed) == {parquet_dir / "2021.parquet", parquet_dir / "2022.parquet"}
    assert len(closed) == 2

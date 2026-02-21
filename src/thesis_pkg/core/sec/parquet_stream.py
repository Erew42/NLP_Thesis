from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
import re

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq


def discover_input_parquet_files(parquet_dir: Path) -> list[Path]:
    """
    Discover SEC input Parquet files from a directory.

    Preference order:
    1. Any ``*_batch_*.parquet`` files (if present).
    2. Otherwise, fallback to year-named files matching ``YYYY.parquet``.

    Args:
        parquet_dir: Directory containing candidate parquet datasets.

    Returns:
        list[Path]: Sorted file paths matching the discovery rules.
    """
    files_batch = sorted(parquet_dir.glob("*_batch_*.parquet"))
    if files_batch:
        return files_batch

    files_year = sorted(
        p
        for p in parquet_dir.glob("*.parquet")
        if p.is_file() and re.fullmatch(r"\d{4}", p.stem)
    )
    return files_year


def iter_parquet_filing_texts(
    parquet_dir: Path,
    doc_ids: set[str],
    *,
    batch_size: int = 8,
) -> Iterator[dict[str, str]]:
    """
    Lazily stream selected filing texts from discovered Parquet files.

    Args:
        parquet_dir: Directory containing parquet datasets.
        doc_ids: Target ``doc_id`` values to retrieve.
        batch_size: Batch size passed to ``ParquetFile.iter_batches``.

    Yields:
        dict[str, str]: Dictionaries with keys ``doc_id``, ``accession``,
        and ``full_text``.

    Notes:
        Files missing ``doc_id`` or ``full_text`` columns are skipped.
        Parquet handles are explicitly closed after each file scan.
    """
    if not doc_ids:
        return iter(())

    files = discover_input_parquet_files(parquet_dir)
    remaining = {str(doc_id) for doc_id in doc_ids if doc_id}
    if not remaining:
        return iter(())

    want_cols = ["doc_id", "accession_number", "full_text"]

    def _iterator() -> Iterator[dict[str, str]]:
        for file_path in files:
            if not remaining:
                break
            pf: pq.ParquetFile | None = None
            try:
                pf = pq.ParquetFile(file_path)
                available = set(pf.schema.names)
                columns = [c for c in want_cols if c in available]
                if "doc_id" not in columns or "full_text" not in columns:
                    continue
                for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
                    if not remaining:
                        break
                    tbl = pa.Table.from_batches([batch])
                    df = pl.from_arrow(tbl)
                    if "doc_id" not in df.columns:
                        continue
                    filtered = df.filter(pl.col("doc_id").cast(pl.Utf8).is_in(remaining))
                    if filtered.is_empty():
                        continue
                    for row in filtered.iter_rows(named=True):
                        doc_id = str(row.get("doc_id") or "")
                        if doc_id and doc_id in remaining:
                            remaining.remove(doc_id)
                            yield {
                                "doc_id": doc_id,
                                "accession": str(row.get("accession_number") or ""),
                                "full_text": str(row.get("full_text") or ""),
                            }
            finally:
                if pf is not None:
                    try:
                        pf.close()
                    except Exception:
                        pass

    return _iterator()


__all__ = ["discover_input_parquet_files", "iter_parquet_filing_texts"]

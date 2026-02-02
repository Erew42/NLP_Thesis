from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq


def iter_parquet_filing_texts(
    parquet_dir: Path,
    doc_ids: set[str],
    *,
    batch_size: int = 8,
) -> Iterator[dict[str, str]]:
    if not doc_ids:
        return iter(())

    files = sorted(parquet_dir.glob("*_batch_*.parquet"))
    remaining = {str(doc_id) for doc_id in doc_ids if doc_id}
    if not remaining:
        return iter(())

    want_cols = ["doc_id", "accession_number", "full_text"]

    def _iterator() -> Iterator[dict[str, str]]:
        for file_path in files:
            if not remaining:
                break
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

    return _iterator()


__all__ = ["iter_parquet_filing_texts"]

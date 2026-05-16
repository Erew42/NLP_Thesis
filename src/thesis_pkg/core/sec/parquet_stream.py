from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
import re
from typing import Any

import polars as pl
import pyarrow.parquet as pq

try:
    from thesis_native import _lm2011_rust
except Exception as exc:  # pragma: no cover - optional native extension
    _lm2011_rust = None
    _PARQUET_STREAM_RUST_IMPORT_ERROR: str | None = f"{type(exc).__name__}: {exc}"
else:
    _PARQUET_STREAM_RUST_IMPORT_ERROR = None

_PARQUET_STREAM_RUST_METRICS: dict[str, int] = {
    "selected_indices_fast_success": 0,
    "selected_indices_fast_failures": 0,
    "selected_indices_fallbacks": 0,
}


def get_parquet_stream_rust_accel_metrics() -> dict[str, int | str | bool | None]:
    metrics: dict[str, int | str | bool | None] = dict(_PARQUET_STREAM_RUST_METRICS)
    metrics["rust_accel_available"] = _lm2011_rust is not None
    metrics["rust_accel_import_error"] = _PARQUET_STREAM_RUST_IMPORT_ERROR
    return metrics


def reset_parquet_stream_rust_accel_metrics() -> None:
    for key in _PARQUET_STREAM_RUST_METRICS:
        _PARQUET_STREAM_RUST_METRICS[key] = 0


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


def _normalize_doc_id_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _select_filing_text_indices_py(
    remaining_doc_ids: set[str],
    doc_id_values: list[Any],
) -> list[tuple[int, str]]:
    wanted = {str(doc_id) for doc_id in remaining_doc_ids if doc_id}
    selected: list[tuple[int, str]] = []
    for index, value in enumerate(doc_id_values):
        if not wanted:
            break
        doc_id = _normalize_doc_id_value(value)
        if doc_id and doc_id in wanted:
            selected.append((index, doc_id))
            wanted.remove(doc_id)
    return selected


def _select_filing_text_indices(
    remaining_doc_ids: set[str],
    doc_id_values: list[Any],
) -> list[tuple[int, str]]:
    if _lm2011_rust is not None:
        try:
            raw_indices = _lm2011_rust.parquet_stream_selected_doc_indices(
                [str(doc_id) for doc_id in remaining_doc_ids if doc_id],
                [_normalize_doc_id_value(value) or None for value in doc_id_values],
            )
            _PARQUET_STREAM_RUST_METRICS["selected_indices_fast_success"] += 1
            return [(int(index), str(doc_id)) for index, doc_id in raw_indices]
        except Exception:
            _PARQUET_STREAM_RUST_METRICS["selected_indices_fast_failures"] += 1
    _PARQUET_STREAM_RUST_METRICS["selected_indices_fallbacks"] += 1
    return _select_filing_text_indices_py(remaining_doc_ids, doc_id_values)


def _array_value_to_text(array: Any | None, index: int) -> str:
    if array is None:
        return ""
    value = array[index].as_py()
    return str(value or "")


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
                    batch_columns = list(batch.schema.names)
                    if "doc_id" not in batch_columns or "full_text" not in batch_columns:
                        continue
                    doc_id_values = batch.column(batch_columns.index("doc_id")).to_pylist()
                    selected_indices = _select_filing_text_indices(remaining, doc_id_values)
                    if not selected_indices:
                        continue
                    accession_array = (
                        batch.column(batch_columns.index("accession_number"))
                        if "accession_number" in batch_columns
                        else None
                    )
                    full_text_array = batch.column(batch_columns.index("full_text"))
                    for row_index, doc_id in selected_indices:
                        if doc_id and doc_id in remaining:
                            remaining.remove(doc_id)
                            yield {
                                "doc_id": doc_id,
                                "accession": _array_value_to_text(accession_array, row_index),
                                "full_text": _array_value_to_text(full_text_array, row_index),
                            }
            finally:
                if pf is not None:
                    try:
                        pf.close()
                    except Exception:
                        pass

    return _iterator()


__all__ = [
    "discover_input_parquet_files",
    "get_parquet_stream_rust_accel_metrics",
    "iter_parquet_filing_texts",
    "reset_parquet_stream_rust_accel_metrics",
]

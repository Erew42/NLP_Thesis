from __future__ import annotations

import gc
import json
import os
import re
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Literal

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from thesis_pkg.core.sec.filing_text import (
    FilingItemSchema,
    HEADER_SEARCH_LIMIT_DEFAULT,
    ParsedFilingSchema,
    RawTextSchema,
    _digits_only,
    _is_empty_item_text,
    _make_doc_id,
    extract_filing_items,
    parse_filename_minimal,
    parse_header,
)
from thesis_pkg.io.parquet import (
    _assert_parquet_magic,
    _copy_with_verify,
    _validate_parquet_full,
    _validate_parquet_quick,
    concat_parquets_arrow,
)


def _flush_batch(records: list[dict], out_path: Path, compression: str) -> None:
    if not records:
        return
    df = pl.DataFrame(records, schema=RawTextSchema.schema)
    df = df.with_columns(
        pl.col("file_date_filename").str.strptime(pl.Date, "%Y%m%d", strict=False)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    try:
        tmp_path.unlink(missing_ok=True)
        df.write_parquet(tmp_path, compression=compression)
        _assert_parquet_magic(tmp_path)
        os.replace(tmp_path, out_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _flush_parsed_batch(records: list[dict], out_path: Path, compression: str) -> None:
    if not records:
        return
    df = pl.DataFrame(records, schema=ParsedFilingSchema.schema)
    df = df.with_columns(
        pl.col("filing_date").str.strptime(pl.Date, "%Y%m%d", strict=False),
        pl.col("period_end").str.strptime(pl.Date, "%Y%m%d", strict=False),
        pl.col("file_date_filename").str.strptime(pl.Date, "%Y%m%d", strict=False),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    try:
        tmp_path.unlink(missing_ok=True)
        df.write_parquet(tmp_path, compression=compression)
        _assert_parquet_magic(tmp_path)
        os.replace(tmp_path, out_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _flush_item_batch(records: list[dict], out_path: Path, compression: str) -> None:
    if not records:
        return
    df = pl.DataFrame(records, schema=FilingItemSchema.schema)
    df = df.with_columns(
        pl.col("file_date_filename").cast(pl.Date, strict=False),
        pl.col("filing_date").cast(pl.Date, strict=False),
        pl.col("period_end").cast(pl.Date, strict=False),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    try:
        tmp_path.unlink(missing_ok=True)
        df.write_parquet(tmp_path, compression=compression)
        _assert_parquet_magic(tmp_path)
        os.replace(tmp_path, out_path)
    finally:
        tmp_path.unlink(missing_ok=True)


_NO_ITEM_STATS_COLUMNS = (
    "year",
    "document_type_filename",
    "n_filings_eligible",
    "n_with_items",
    "n_no_items",
    "share_no_item",
    "avg_text_len_with_items",
    "avg_text_len_no_items",
)


def _flush_non_item_batch(records: list[dict], out_path: Path, compression: str) -> None:
    if not records:
        return
    df = pl.DataFrame(records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    try:
        tmp_path.unlink(missing_ok=True)
        df.write_parquet(tmp_path, compression=compression)
        _assert_parquet_magic(tmp_path)
        os.replace(tmp_path, out_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _update_no_item_stats(
    stats: dict[str, dict[str, int]],
    doc_type: str | None,
    has_items: bool,
    text_len: int,
) -> None:
    key = doc_type or "UNKNOWN"
    cur = stats.setdefault(
        key,
        {
            "with_items": 0,
            "with_len": 0,
            "no_items": 0,
            "no_len": 0,
        },
    )
    if has_items:
        cur["with_items"] += 1
        cur["with_len"] += text_len
    else:
        cur["no_items"] += 1
        cur["no_len"] += text_len


def _merge_no_item_stats(
    target: dict[str, dict[str, int]],
    source: dict[str, dict[str, int]],
) -> None:
    for doc_type, stats in source.items():
        cur = target.setdefault(
            doc_type,
            {
                "with_items": 0,
                "with_len": 0,
                "no_items": 0,
                "no_len": 0,
            },
        )
        cur["with_items"] += stats["with_items"]
        cur["with_len"] += stats["with_len"]
        cur["no_items"] += stats["no_items"]
        cur["no_len"] += stats["no_len"]


def _no_item_stats_row(
    year: str,
    doc_type: str,
    counts: dict[str, int],
) -> dict:
    with_items = counts["with_items"]
    no_items = counts["no_items"]
    total = with_items + no_items
    return {
        "year": year,
        "document_type_filename": doc_type,
        "n_filings_eligible": total,
        "n_with_items": with_items,
        "n_no_items": no_items,
        "share_no_item": (no_items / total) if total else 0.0,
        "avg_text_len_with_items": (counts["with_len"] / with_items) if with_items else 0.0,
        "avg_text_len_no_items": (counts["no_len"] / no_items) if no_items else 0.0,
    }


def _build_no_item_stats_rows(
    stats: dict[str, dict[str, int]],
    year: str,
) -> list[dict]:
    rows: list[dict] = []
    totals = {"with_items": 0, "with_len": 0, "no_items": 0, "no_len": 0}
    for doc_type in sorted(stats):
        cur = stats[doc_type]
        totals["with_items"] += cur["with_items"]
        totals["with_len"] += cur["with_len"]
        totals["no_items"] += cur["no_items"]
        totals["no_len"] += cur["no_len"]
        rows.append(_no_item_stats_row(year, doc_type, cur))
    if rows:
        rows.append(_no_item_stats_row(year, "TOTAL", totals))
    return rows


def _write_no_item_stats_csv(stats_path: Path, rows: list[dict]) -> Path:
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        df = pl.DataFrame(rows).select(_NO_ITEM_STATS_COLUMNS, strict=False)
    else:
        df = pl.DataFrame({col: [] for col in _NO_ITEM_STATS_COLUMNS})
    df.write_csv(stats_path)
    return stats_path


def aggregate_no_item_stats_csvs(
    stats_csvs: Path | list[Path],
    out_stats_csv: Path,
    out_doc_type_csv: Path | None = None,
) -> Path:
    """
    Aggregate per-year no-item stats CSVs into a full-sample stats CSV using weighted means.
    Optionally write a compact per-doc-type totals table.
    """
    if isinstance(stats_csvs, list):
        files = [Path(p) for p in stats_csvs]
    else:
        stats_csvs = Path(stats_csvs)
        if stats_csvs.is_dir():
            files = sorted(stats_csvs.glob("*_no_item_stats.csv"))
            if not files:
                files = sorted(stats_csvs.glob("*.csv"))
        else:
            files = [stats_csvs]

    if not files:
        raise ValueError("No stats CSVs provided for aggregation.")

    schema_overrides = {col: pl.Utf8 for col in _NO_ITEM_STATS_COLUMNS}
    dfs: list[pl.DataFrame] = []
    for p in files:
        df = pl.read_csv(p, schema_overrides=schema_overrides)
        for col in _NO_ITEM_STATS_COLUMNS:
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).alias(col))
        dfs.append(df.select(_NO_ITEM_STATS_COLUMNS))
    df = pl.concat(dfs, how="vertical")
    if df.height == 0:
        return _write_no_item_stats_csv(out_stats_csv, [])

    df = df.with_columns(
        pl.col("year").cast(pl.Utf8),
        pl.col("document_type_filename").cast(pl.Utf8),
    ).filter(
        (pl.col("document_type_filename") != "TOTAL") & (pl.col("year") != "ALL")
    ).with_columns(
        pl.col("n_filings_eligible").fill_null(0).cast(pl.Int64),
        pl.col("n_with_items").fill_null(0).cast(pl.Int64),
        pl.col("n_no_items").fill_null(0).cast(pl.Int64),
        pl.col("share_no_item").fill_null(0.0).cast(pl.Float64),
        pl.col("avg_text_len_with_items").fill_null(0.0).cast(pl.Float64),
        pl.col("avg_text_len_no_items").fill_null(0.0).cast(pl.Float64),
    )

    if df.height == 0:
        return _write_no_item_stats_csv(out_stats_csv, [])

    grouped = df.group_by("document_type_filename").agg(
        pl.sum("n_with_items").alias("n_with_items"),
        pl.sum("n_no_items").alias("n_no_items"),
        (pl.col("avg_text_len_with_items") * pl.col("n_with_items"))
        .sum()
        .alias("with_len_sum"),
        (pl.col("avg_text_len_no_items") * pl.col("n_no_items"))
        .sum()
        .alias("no_len_sum"),
    )

    eligible_expr = pl.col("n_with_items") + pl.col("n_no_items")
    totals = grouped.select(
        pl.sum("n_with_items").alias("n_with_items"),
        pl.sum("n_no_items").alias("n_no_items"),
        pl.sum("with_len_sum").alias("with_len_sum"),
        pl.sum("no_len_sum").alias("no_len_sum"),
    ).with_columns(
        eligible_expr.alias("n_filings_eligible"),
        pl.when(eligible_expr > 0)
        .then(pl.col("n_no_items") / eligible_expr)
        .otherwise(0.0)
        .alias("share_no_item"),
        pl.when(pl.col("n_with_items") > 0)
        .then(pl.col("with_len_sum") / pl.col("n_with_items"))
        .otherwise(0.0)
        .alias("avg_text_len_with_items"),
        pl.when(pl.col("n_no_items") > 0)
        .then(pl.col("no_len_sum") / pl.col("n_no_items"))
        .otherwise(0.0)
        .alias("avg_text_len_no_items"),
        pl.lit("TOTAL").alias("document_type_filename"),
        pl.lit("ALL").alias("year"),
    ).drop(["with_len_sum", "no_len_sum"])

    grouped = grouped.with_columns(
        eligible_expr.alias("n_filings_eligible"),
        pl.when(eligible_expr > 0).then(pl.col("n_no_items") / eligible_expr).otherwise(0.0).alias(
            "share_no_item"
        ),
        pl.when(pl.col("n_with_items") > 0)
        .then(pl.col("with_len_sum") / pl.col("n_with_items"))
        .otherwise(0.0)
        .alias("avg_text_len_with_items"),
        pl.when(pl.col("n_no_items") > 0)
        .then(pl.col("no_len_sum") / pl.col("n_no_items"))
        .otherwise(0.0)
        .alias("avg_text_len_no_items"),
        pl.lit("ALL").alias("year"),
    ).drop(["with_len_sum", "no_len_sum"])

    grouped = grouped.select(_NO_ITEM_STATS_COLUMNS)
    totals = totals.select(_NO_ITEM_STATS_COLUMNS)
    result = pl.concat([grouped, totals], how="vertical")
    if out_doc_type_csv is not None:
        out_doc_type_csv.parent.mkdir(parents=True, exist_ok=True)
        compact = result.filter(
            (pl.col("year") == "ALL") & (pl.col("document_type_filename") != "TOTAL")
        ).select(
            [
                "document_type_filename",
                "n_filings_eligible",
                "n_with_items",
                "n_no_items",
                "share_no_item",
            ]
        )
        compact.write_csv(out_doc_type_csv)
    return _write_no_item_stats_csv(out_stats_csv, result.to_dicts())


def _load_item_accession_set(
    item_parquet_path: Path,
    *,
    batch_size: int = 200_000,
) -> set[str]:
    accessions: set[str] = set()
    pf = pq.ParquetFile(item_parquet_path)
    try:
        if "accession_number" not in pf.schema.names:
            raise ValueError(f"accession_number column not found in {item_parquet_path}")
        for batch in pf.iter_batches(batch_size=batch_size, columns=["accession_number"]):
            for val in batch.column(0).to_pylist():
                if val:
                    accessions.add(val)
    finally:
        pf.close()
    return accessions

def process_zip_year_raw_text(
    zip_path: Path,
    out_dir: Path,
    batch_max_rows: int = 1000,
    batch_max_text_bytes: int = 250 * 1024 * 1024,  # 250 MB
    tmp_dir: Path | None = None,
    encoding: str = "utf-8",
    compression: Literal["zstd", "snappy", "gzip", "uncompressed"] = "zstd",
    local_work_dir: Path | None = None,
    copy_retries: int = 3,
    copy_sleep: float = 1.0,
    validate_on_copy: bool | Literal["quick", "full"] = True,
) -> list[Path]:
    """
    Step 1: Read yearly ZIP, parse filename fields, store full_text + filename-derived metadata in parquet batches.
    Writes batches to a local work dir first, then copies them to out_dir with validation/retries and returns the final batch paths.

    Notes:
      - Streams from ZIP without unpacking to disk.
      - Copies ZIP to local tmp_dir first (important when source is a network mount like Drive).
      - Each batch copy is verified (size + quick parquet read) to reduce Drive corruption.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = tmp_dir or Path(tempfile.gettempdir())
    tmp_dir.mkdir(parents=True, exist_ok=True)
    local_root = local_work_dir or (Path(tempfile.gettempdir()) / "_batch_work")
    local_root.mkdir(parents=True, exist_ok=True)
    local_out_dir = local_root / zip_path.stem
    local_out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Copy ZIP local for speed/stability
    src_zip = zip_path
    local_zip = tmp_dir / f"{zip_path.stem}.zip"
    cleanup_local = False
    if src_zip.resolve() == local_zip.resolve():
        local_zip = src_zip
    else:
        _copy_with_verify(
            src_zip,
            local_zip,
            retries=copy_retries,
            sleep=copy_sleep,
            validate=False,
        )
        cleanup_local = True

    written: list[Path] = []
    records: list[dict] = []
    batch_idx = 1
    text_bytes = 0

    try:
        with zipfile.ZipFile(local_zip, "r") as zf:
            members = [m for m in zf.namelist() if m.lower().endswith(".txt")]
            members.sort()  # deterministic order

            for member in members:
                filename = Path(member).name
                meta = parse_filename_minimal(filename)

                # read and decode filing once; avoid re-encoding just to count bytes
                with zf.open(member, "r") as bf:
                    raw = bf.read()
                txt = raw.decode(encoding, errors="replace")

                rec = {
                    **meta,
                    "filename": filename,
                    "zip_member_path": member,
                    "full_text": txt,
                }
                records.append(rec)
                text_bytes += len(raw)

                if len(records) >= batch_max_rows or text_bytes >= batch_max_text_bytes:
                    local_file = local_out_dir / f"{zip_path.stem}_batch_{batch_idx:04d}.parquet"
                    if local_file.exists():
                        local_file.unlink()
                    _flush_batch(records, local_file, compression)
                    final_out_file = out_dir / local_file.name
                    _copy_with_verify(
                        local_file,
                        final_out_file,
                        retries=copy_retries,
                        sleep=copy_sleep,
                        validate=validate_on_copy,
                    )
                    written.append(final_out_file)
                    local_file.unlink(missing_ok=True)

                    records = []
                    batch_idx += 1
                    text_bytes = 0
                    gc.collect()

            # final batch
            if records:
                local_file = local_out_dir / f"{zip_path.stem}_batch_{batch_idx:04d}.parquet"
                if local_file.exists():
                    local_file.unlink()
                _flush_batch(records, local_file, compression)
                final_out_file = out_dir / local_file.name
                _copy_with_verify(
                    local_file,
                    final_out_file,
                    retries=copy_retries,
                    sleep=copy_sleep,
                    validate=validate_on_copy,
                )
                written.append(final_out_file)
                local_file.unlink(missing_ok=True)

    finally:
        if cleanup_local:
            local_zip.unlink(missing_ok=True)
        gc.collect()

    return written


def process_zip_year(
    zip_path: Path,
    out_dir: Path,
    batch_max_rows: int = 2000,
    batch_max_text_bytes: int = 250 * 1024 * 1024,  # 250 MB
    header_search_limit: int = HEADER_SEARCH_LIMIT_DEFAULT,
    tmp_dir: Path | None = None,
    encoding: str = "utf-8",
    compression: Literal["zstd", "snappy", "gzip", "uncompressed"] = "zstd",
    local_work_dir: Path | None = None,
    copy_retries: int = 3,
    copy_sleep: float = 1.0,
    validate_on_copy: bool | Literal["quick", "full"] = True,
) -> list[Path]:
    """
    Parse a yearly ZIP of filings into parquet batches with header metadata and conflict flags.

    Batches are written to a local work directory first, then copied to out_dir with validation/retries (helps when out_dir is remote).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = tmp_dir or Path(tempfile.gettempdir())
    tmp_dir.mkdir(parents=True, exist_ok=True)
    local_root = local_work_dir or (Path(tempfile.gettempdir()) / "_batch_work")
    local_root.mkdir(parents=True, exist_ok=True)
    local_out_dir = local_root / zip_path.stem
    local_out_dir.mkdir(parents=True, exist_ok=True)

    src_zip = zip_path
    local_zip = tmp_dir / f"{zip_path.stem}.zip"
    cleanup_local = False
    if src_zip.resolve() == local_zip.resolve():
        local_zip = src_zip
    else:
        _copy_with_verify(
            src_zip,
            local_zip,
            retries=copy_retries,
            sleep=copy_sleep,
            validate=False,
        )
        cleanup_local = True

    written: list[Path] = []
    records: list[dict] = []
    batch_idx = 1
    text_bytes = 0

    try:
        with zipfile.ZipFile(local_zip, "r") as zf:
            members = [m for m in zf.namelist() if m.lower().endswith(".txt")]
            members.sort()

            for member in members:
                filename = Path(member).name
                meta = parse_filename_minimal(filename)

                with zf.open(member, "r") as bf:
                    raw = bf.read()
                txt = raw.decode(encoding, errors="replace")

                hdr = parse_header(txt, header_search_limit=header_search_limit)

                filename_cik = meta.get("cik")
                filename_acc = meta.get("accession_number")

                cik_conflict = False
                if filename_cik is not None and hdr["header_ciks_int_set"]:
                    if filename_cik not in hdr["header_ciks_int_set"]:
                        cik_conflict = True

                accession_conflict = False
                if filename_acc and hdr["header_accession_str"]:
                    accession_conflict = filename_acc != hdr["header_accession_str"]

                final_accession = hdr["header_accession_str"] or filename_acc
                final_accession_nodash = _digits_only(final_accession)
                final_date = hdr["header_filing_date_str"] or meta.get("file_date_filename")
                period_end_header = hdr.get("header_period_end_str")

                rec = {
                    **meta,
                    "filename": filename,
                    "zip_member_path": member,
                    "full_text": txt,
                    "filing_date": final_date,
                    "filing_date_header": hdr["header_filing_date_str"],
                    "period_end": period_end_header,
                    "period_end_header": period_end_header,
                    "cik_header_primary": hdr["primary_header_cik"],
                    "ciks_header_secondary": hdr["secondary_ciks"],
                    "accession_header": hdr["header_accession_str"],
                    "cik_conflict": cik_conflict,
                    "accession_conflict": accession_conflict,
                    "accession_number": final_accession,
                    "accession_nodash": final_accession_nodash,
                    "doc_id": _make_doc_id(meta.get("cik_10"), final_accession),
                }

                records.append(rec)
                text_bytes += len(raw)

                if len(records) >= batch_max_rows or text_bytes >= batch_max_text_bytes:
                    local_file = local_out_dir / f"{zip_path.stem}_batch_{batch_idx:04d}.parquet"
                    if local_file.exists():
                        local_file.unlink()
                    _flush_parsed_batch(records, local_file, compression)
                    final_out_file = out_dir / local_file.name
                    _copy_with_verify(
                        local_file,
                        final_out_file,
                        retries=copy_retries,
                        sleep=copy_sleep,
                        validate=validate_on_copy,
                    )
                    written.append(final_out_file)
                    local_file.unlink(missing_ok=True)
                    records = []
                    batch_idx += 1
                    text_bytes = 0
                    gc.collect()

            if records:
                local_file = local_out_dir / f"{zip_path.stem}_batch_{batch_idx:04d}.parquet"
                if local_file.exists():
                    local_file.unlink()
                _flush_parsed_batch(records, local_file, compression)
                final_out_file = out_dir / local_file.name
                _copy_with_verify(
                    local_file,
                    final_out_file,
                    retries=copy_retries,
                    sleep=copy_sleep,
                    validate=validate_on_copy,
                )
                written.append(final_out_file)
                local_file.unlink(missing_ok=True)

    finally:
        if cleanup_local:
            local_zip.unlink(missing_ok=True)
        gc.collect()

    return written


def merge_yearly_batches(
    batch_dir: Path,
    out_dir: Path,
    checkpoint_path: Path | None = None,
    local_work_dir: Path | None = None,
    batch_size: int = 32_000,
    compression: str = "zstd",
    compression_level: int | None = 1,
    sleep_between_years: float | None = None,
    validate_inputs: bool | Literal["quick", "full"] = True,
    copy_retries: int = 3,
    copy_sleep: float = 1.0,
    stage_inputs_locally: bool | None = None,
    years: list[str] | None = None,
) -> list[Path]:
    """
    Merge per-year parquet batches (e.g., 2020_batch_0001.parquet) into one parquet per year.
    Returns paths of merged parquet files.

    Designed for slow/remote filesystems (e.g., Drive): writes to a local work dir
    first, optionally stages/validates inputs, then copies the result to `out_dir` with validation/retries.
    When `years` is provided, only those YYYY folders/files are merged.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    local_work_dir = local_work_dir or (Path(tempfile.gettempdir()) / "_merge_work")
    local_work_dir.mkdir(parents=True, exist_ok=True)
    stage_inputs = stage_inputs_locally
    if stage_inputs is None:
        try:
            stage_inputs = batch_dir.stat().st_dev != local_work_dir.stat().st_dev
        except Exception:
            stage_inputs = False

    done: set[str] = set()
    if checkpoint_path and checkpoint_path.exists():
        try:
            done = set(json.loads(checkpoint_path.read_text()))
        except Exception:
            done = set()

    files_by_year: dict[str, list[Path]] = {}

    # Prefer nested layout: batch_dir/<YYYY>/<YYYY>_batch_*.parquet
    for year_dir in batch_dir.iterdir():
        if not (year_dir.is_dir() and year_dir.name.isdigit() and len(year_dir.name) == 4):
            continue
        files = sorted(year_dir.glob(f"{year_dir.name}_batch_*.parquet"))
        if files:
            files_by_year[year_dir.name] = files

    # Fallback to flat layout: batch_dir/<YYYY>_batch_*.parquet
    if not files_by_year:
        for p in batch_dir.glob("*_batch_*.parquet"):
            m = re.match(r"(?P<year>\d{4})_batch_", p.name)
            if not m:
                continue
            year = m.group("year")
            files_by_year.setdefault(year, []).append(p)

    if not files_by_year:
        raise ValueError(
            f"No batch files found under {batch_dir} (expected <YYYY>/<YYYY>_batch_*.parquet or <YYYY>_batch_*.parquet)"
        )

    if years:
        wanted = {str(y) for y in years}
        files_by_year = {year: files for year, files in files_by_year.items() if year in wanted}
        if not files_by_year:
            raise ValueError(f"No batch files found for years={sorted(wanted)} under {batch_dir}")

    merged: list[Path] = []
    for year in sorted(files_by_year):
        out_path = out_dir / f"{year}.parquet"
        if year in done and out_path.exists():
            continue

        files = sorted(files_by_year[year])
        validate_mode: Literal["none", "quick", "full"]
        if validate_inputs is True:
            validate_mode = "quick"
        elif validate_inputs is False:
            validate_mode = "none"
        else:
            validate_mode = validate_inputs

        if validate_mode != "none":
            if stage_inputs:
                for f in files:
                    _assert_parquet_magic(f)
            else:
                for f in files:
                    if validate_mode == "quick":
                        _validate_parquet_quick(f)
                    else:
                        _validate_parquet_full(f, batch_size=min(batch_size, 32_000))

        tmp_path = local_work_dir / f"{year}.parquet"
        if tmp_path.exists():
            tmp_path.unlink()

        try:
            concat_parquets_arrow(
                files,
                tmp_path,
                batch_size=batch_size,
                compression=compression,
                compression_level=compression_level,
                stage_dir=(local_work_dir / "_concat_stage" / year) if stage_inputs else None,
                stage_copy_retries=copy_retries,
                stage_copy_sleep=copy_sleep,
                stage_validate=validate_mode if stage_inputs else False,
            )
            _assert_parquet_magic(tmp_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

        _copy_with_verify(
            tmp_path,
            out_path,
            retries=copy_retries,
            sleep=copy_sleep,
            validate=True,
        )
        tmp_path.unlink(missing_ok=True)
        merged.append(out_path)

        if checkpoint_path:
            done.add(year)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_tmp = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
            checkpoint_tmp.write_text(json.dumps(sorted(done)))
            os.replace(checkpoint_tmp, checkpoint_path)

        if sleep_between_years:
            time.sleep(sleep_between_years)

    return merged


def summarize_year_parquets(parquet_dir: Path) -> list[dict]:
    """
    Inspect yearly parquet files and return a summary per file.
    Each summary includes: path, year, rows, size_bytes, columns, and status.
    """
    summaries: list[dict] = []
    for path in sorted(parquet_dir.glob("*.parquet")):
        year = path.stem
        if not (year.isdigit() and len(year) == 4):
            continue
        size_bytes = path.stat().st_size
        try:
            lf = pl.scan_parquet(path)
            columns = lf.collect_schema().names()
            rows = lf.select(pl.len()).collect().item()

            if "full_text" not in columns:
                status = "WARN_NO_TEXT"
            elif "cik" not in columns:
                status = "ERR_NO_CIK"
            elif rows == 0:
                status = "EMPTY"
            else:
                status = "OK"
        except Exception as exc:  # keep going if a file is corrupt
            columns = []
            rows = None
            status = f"ERROR: {exc}"

        summaries.append(
            {
                "path": path,
                "year": year,
                "rows": rows,
                "size_bytes": size_bytes,
                "columns": columns,
                "status": status,
            }
        )

    return summaries


def build_light_metadata_dataset(
    parquet_dir: Path | list[Path],
    out_path: Path,
    drop_columns: tuple[str, ...] = ("full_text",),
    sort_columns: tuple[str, ...] = ("file_date_filename", "cik"),
    compression: str = "zstd",
) -> Path:
    """
    Merge yearly parquet files into a single metadata-only parquet (drops full_text).

    Accepts either a directory containing per-year parquets or an explicit list of paths.
    """
    files: list[Path]
    if isinstance(parquet_dir, list):
        files = parquet_dir
    else:
        files = sorted(Path(parquet_dir).glob("*.parquet"))

    if not files:
        raise ValueError(f"No parquet files found in {parquet_dir}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    lf = pl.scan_parquet(files)
    available_cols = lf.collect_schema().names()
    cols_to_keep = [c for c in available_cols if c not in set(drop_columns)]
    lf_light = lf.select(cols_to_keep)

    sort_cols = [c for c in sort_columns if c in cols_to_keep]
    if sort_cols:
        lf_light = lf_light.sort(sort_cols)

    lf_light.sink_parquet(out_path, compression=compression)
    return out_path


def process_year_parquet_extract_items(
    year_parquet_path: Path,
    out_dir: Path,
    *,
    parquet_batch_rows: int = 16,
    out_batch_max_rows: int = 50_000,
    out_batch_max_text_bytes: int = 250 * 1024 * 1024,  # 250 MB
    tmp_dir: Path | None = None,
    compression: Literal["zstd", "snappy", "gzip", "uncompressed"] = "zstd",
    local_work_dir: Path | None = None,
    copy_retries: int = 3,
    copy_sleep: float = 1.0,
    validate_on_copy: bool | Literal["quick", "full"] = True,
    non_item_diagnostic: bool = False,
    include_full_text: bool = False,
    regime: bool = True,
) -> Path:
    """
    Expand a merged yearly filing parquet (one row per filing with `full_text`)
    into a yearly item parquet (one row per extracted item).

    Output schema:
      - filing identifiers (doc_id, cik, accession_number, ...)
      - filing_date, period_end (when available)
      - item_part, item_id, item
      - canonical_item, exists_by_regime, item_status
      - full_text contains the extracted item text (pagination artifacts removed)

    Writes intermediate batches to a local work directory and then copies the final
    yearly parquet to `out_dir` with retries/validation (helps on remote mounts).

    When non_item_diagnostic is True, also writes per-year non-item filings to
    `{out_dir}/{year}_no_item_filings.parquet` and a stats CSV to
    `{out_dir}/{year}_no_item_stats.csv`. Set include_full_text=True to keep full_text
    in the no-item parquet output.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = tmp_dir or Path(tempfile.gettempdir())
    tmp_dir.mkdir(parents=True, exist_ok=True)

    local_work_dir = local_work_dir or (Path(tempfile.gettempdir()) / "_item_work")
    local_work_dir.mkdir(parents=True, exist_ok=True)
    local_year_dir = local_work_dir / year_parquet_path.stem
    local_year_dir.mkdir(parents=True, exist_ok=True)

    # Stage input locally for stability/speed.
    src = year_parquet_path
    local_in = tmp_dir / year_parquet_path.name
    cleanup_local = False
    if src.resolve() == local_in.resolve():
        local_in = src
    else:
        _copy_with_verify(
            src,
            local_in,
            retries=copy_retries,
            sleep=copy_sleep,
            validate=True,
        )
        cleanup_local = True

    year = year_parquet_path.stem
    out_path = out_dir / f"{year}.parquet"
    non_item_out_path = (
        out_dir / f"{year}_no_item_filings.parquet" if non_item_diagnostic else None
    )
    non_item_stats_path = out_dir / f"{year}_no_item_stats.csv" if non_item_diagnostic else None
    local_non_item_dir: Path | None = None
    if non_item_diagnostic:
        local_non_item_dir = local_work_dir / f"{year}_no_item_filings"
        local_non_item_dir.mkdir(parents=True, exist_ok=True)

    default_columns = [
        "doc_id",
        "cik",
        "cik_10",
        "accession_number",
        "accession_nodash",
        "file_date_filename",
        "filing_date",
        "period_end",
        "document_type_filename",
        "filename",
        "full_text",
    ]

    written_batches: list[Path] = []
    records: list[dict] = []
    text_bytes = 0
    batch_idx = 1

    filing_rows = 0
    item_rows = 0
    empty_items = 0
    no_item_filings = 0
    non_item_batches: list[Path] = []
    non_item_records: list[dict] = []
    non_item_text_bytes = 0
    non_item_batch_idx = 1
    non_item_stats: dict[str, dict[str, int]] | None = {} if non_item_diagnostic else None

    pf: pq.ParquetFile | None = None
    input_schema: pa.Schema | None = None
    try:
        pf = pq.ParquetFile(local_in)
        input_schema = pf.schema_arrow
        available = set(pf.schema.names)
        if non_item_diagnostic:
            columns = pf.schema.names
        else:
            columns = [c for c in default_columns if c in available]
        for batch in pf.iter_batches(batch_size=parquet_batch_rows, columns=columns):
            tbl = pa.Table.from_batches([batch])
            df = pl.from_arrow(tbl)

            for row in df.iter_rows(named=True):
                filing_rows += 1
                filing_text = row.get("full_text")
                text_len = len(filing_text) if filing_text else 0
                acc = row.get("accession_number")
                eligible = acc is not None
                items: list[dict] = []
                if not filing_text:
                    if eligible:
                        no_item_filings += 1
                        if non_item_stats is not None:
                            _update_no_item_stats(
                                non_item_stats,
                                row.get("document_type_filename"),
                                False,
                                text_len,
                            )
                            row_out = (
                                row
                                if include_full_text
                                else {k: v for k, v in row.items() if k != "full_text"}
                            )
                            non_item_records.append(row_out)
                            non_item_text_bytes += text_len
                            if (
                                len(non_item_records) >= out_batch_max_rows
                                or non_item_text_bytes >= out_batch_max_text_bytes
                            ):
                                assert local_non_item_dir is not None
                                local_batch = (
                                    local_non_item_dir
                                    / f"{year}_no_items_batch_{non_item_batch_idx:04d}.parquet"
                                )
                                if local_batch.exists():
                                    local_batch.unlink()
                                _flush_non_item_batch(non_item_records, local_batch, compression)
                                non_item_batches.append(local_batch)
                                non_item_records = []
                                non_item_text_bytes = 0
                                non_item_batch_idx += 1
                                gc.collect()
                    continue

                try:
                    items = extract_filing_items(
                        filing_text,
                        form_type=row.get("document_type_filename"),
                        filing_date=row.get("filing_date"),
                        period_end=row.get("period_end"),
                        regime=regime,
                    )
                except Exception as exc:
                    print(
                        "[items][error] "
                        f"year={year} row={filing_rows} "
                        f"doc_id={row.get('doc_id')} cik={row.get('cik')} "
                        f"accession={row.get('accession_number')} "
                        f"form={row.get('document_type_filename')} "
                        f"file_date={row.get('file_date_filename')} "
                        f"filename={row.get('filename')} "
                        f"error={type(exc).__name__}: {exc} "
                        f"text_len={len(filing_text)}"
                    )
                    raise
                if not items:
                    if eligible:
                        no_item_filings += 1
                        if non_item_stats is not None:
                            _update_no_item_stats(
                                non_item_stats,
                                row.get("document_type_filename"),
                                False,
                                text_len,
                            )
                            row_out = (
                                row
                                if include_full_text
                                else {k: v for k, v in row.items() if k != "full_text"}
                            )
                            non_item_records.append(row_out)
                            non_item_text_bytes += text_len
                            if (
                                len(non_item_records) >= out_batch_max_rows
                                or non_item_text_bytes >= out_batch_max_text_bytes
                            ):
                                assert local_non_item_dir is not None
                                local_batch = (
                                    local_non_item_dir
                                    / f"{year}_no_items_batch_{non_item_batch_idx:04d}.parquet"
                                )
                                if local_batch.exists():
                                    local_batch.unlink()
                                _flush_non_item_batch(non_item_records, local_batch, compression)
                                non_item_batches.append(local_batch)
                                non_item_records = []
                                non_item_text_bytes = 0
                                non_item_batch_idx += 1
                                gc.collect()
                    continue
                if non_item_stats is not None and eligible:
                    _update_no_item_stats(
                        non_item_stats,
                        row.get("document_type_filename"),
                        True,
                        text_len,
                    )

                for item in items:
                    txt = item.get("full_text") or ""
                    if _is_empty_item_text(txt):
                        empty_items += 1
                    rec = {
                        "doc_id": row.get("doc_id"),
                        "cik": row.get("cik"),
                        "cik_10": row.get("cik_10"),
                        "accession_number": row.get("accession_number"),
                        "accession_nodash": row.get("accession_nodash"),
                        "file_date_filename": row.get("file_date_filename"),
                        "filing_date": row.get("filing_date"),
                        "period_end": row.get("period_end"),
                        "document_type_filename": row.get("document_type_filename"),
                        "filename": row.get("filename"),
                        "item_part": item.get("item_part"),
                        "item_id": item.get("item_id"),
                        "item": item.get("item"),
                        "canonical_item": item.get("canonical_item"),
                        "exists_by_regime": item.get("exists_by_regime"),
                        "item_status": item.get("item_status"),
                        "full_text": txt,
                    }
                    records.append(rec)
                    item_rows += 1
                    text_bytes += len(txt)

                    if len(records) >= out_batch_max_rows or text_bytes >= out_batch_max_text_bytes:
                        local_batch = local_year_dir / f"{year}_items_batch_{batch_idx:04d}.parquet"
                        if local_batch.exists():
                            local_batch.unlink()
                        _flush_item_batch(records, local_batch, compression)
                        written_batches.append(local_batch)
                        records = []
                        text_bytes = 0
                        batch_idx += 1
                        gc.collect()

        if records:
            local_batch = local_year_dir / f"{year}_items_batch_{batch_idx:04d}.parquet"
            if local_batch.exists():
                local_batch.unlink()
            _flush_item_batch(records, local_batch, compression)
            written_batches.append(local_batch)
        if non_item_records:
            assert local_non_item_dir is not None
            local_batch = (
                local_non_item_dir / f"{year}_no_items_batch_{non_item_batch_idx:04d}.parquet"
            )
            if local_batch.exists():
                local_batch.unlink()
            _flush_non_item_batch(non_item_records, local_batch, compression)
            non_item_batches.append(local_batch)

    finally:
        if pf is not None:
            try:
                pf.close()
            except Exception:
                pass
        if cleanup_local:
            local_in.unlink(missing_ok=True)
        gc.collect()

    if not written_batches:
        # Write an empty file with the expected schema.
        empty_df = pl.DataFrame(schema=FilingItemSchema.schema)
        tmp_out = local_year_dir / f"{year}.parquet"
        if tmp_out.exists():
            tmp_out.unlink()
        empty_df.write_parquet(tmp_out, compression=compression)
    else:
        tmp_out = local_year_dir / f"{year}.parquet"
        if tmp_out.exists():
            tmp_out.unlink()
        concat_parquets_arrow(
            in_files=written_batches,
            out_path=tmp_out,
            batch_size=32_000,
            compression=compression,
            compression_level=1,
            stage_dir=None,
        )
        _assert_parquet_magic(tmp_out)

    _copy_with_verify(
        tmp_out,
        out_path,
        retries=copy_retries,
        sleep=copy_sleep,
        validate=validate_on_copy,
    )

    # Keep local output for debugging if needed, but remove intermediate batches.
    for b in written_batches:
        b.unlink(missing_ok=True)

    if non_item_diagnostic:
        assert non_item_out_path is not None
        assert local_non_item_dir is not None
        tmp_non_item_out = local_non_item_dir / f"{year}_no_item_filings.parquet"
        if tmp_non_item_out.exists():
            tmp_non_item_out.unlink()
        if not non_item_batches:
            if input_schema is None:
                raise ValueError(f"Missing input schema for non-item output of {year_parquet_path}")
            output_schema = input_schema
            if not include_full_text and "full_text" in output_schema.names:
                output_schema = pa.schema(
                    [field for field in output_schema if field.name != "full_text"]
                )
            arrays = [pa.array([], type=field.type) for field in output_schema]
            empty_tbl = pa.Table.from_arrays(arrays, schema=output_schema)
            pq.write_table(empty_tbl, tmp_non_item_out, compression=compression)
            _assert_parquet_magic(tmp_non_item_out)
        else:
            concat_parquets_arrow(
                in_files=non_item_batches,
                out_path=tmp_non_item_out,
                batch_size=32_000,
                compression=compression,
                compression_level=1,
                stage_dir=None,
            )
            _assert_parquet_magic(tmp_non_item_out)

        _copy_with_verify(
            tmp_non_item_out,
            non_item_out_path,
            retries=copy_retries,
            sleep=copy_sleep,
            validate=validate_on_copy,
        )

        for b in non_item_batches:
            b.unlink(missing_ok=True)

        if non_item_stats is not None and non_item_stats_path is not None:
            stats_rows = _build_no_item_stats_rows(non_item_stats, year)
            _write_no_item_stats_csv(non_item_stats_path, stats_rows)

    print(
        f"[items] {year} filings={filing_rows} items={item_rows} "
        f"items_per_filing={item_rows / filing_rows if filing_rows else 0:.2f} "
        f"no_item_filings={no_item_filings} empty_items={empty_items} -> {out_path}"
    )

    return out_path


def compute_no_item_diagnostics(
    year_filings_parquet: Path | pl.LazyFrame,
    year_items_parquet: Path | pl.LazyFrame,
    out_no_item_parquet: Path,
    out_stats_csv: Path,
    include_full_text: bool = False,
) -> Path:
    """
    Compute no-item filings by accession_number anti-join against extracted items.
    """
    tmp_dir = Path(tempfile.gettempdir())
    tmp_dir.mkdir(parents=True, exist_ok=True)
    local_work_dir = tmp_dir / "_no_item_diag"
    local_work_dir.mkdir(parents=True, exist_ok=True)

    cleanup_paths: list[Path] = []

    def _stage_parquet(path: Path, label: str) -> Path:
        local_path = tmp_dir / f"{label}{path.suffix}"
        if path.resolve() != local_path.resolve():
            _copy_with_verify(path, local_path, retries=3, sleep=1.0, validate=True)
            cleanup_paths.append(local_path)
            return local_path
        return path

    filings_path: Path | None = None
    items_path: Path | None = None
    if isinstance(year_filings_parquet, pl.LazyFrame):
        filings_lf = year_filings_parquet
    else:
        filings_path = _stage_parquet(
            Path(year_filings_parquet),
            f"{Path(year_filings_parquet).stem}_full",
        )
        filings_lf = pl.scan_parquet(filings_path)

    if isinstance(year_items_parquet, pl.LazyFrame):
        items_lf = year_items_parquet
    else:
        items_path = _stage_parquet(
            Path(year_items_parquet),
            f"{Path(year_items_parquet).stem}_items",
        )
        items_lf = pl.scan_parquet(items_path)

    try:
        filings_cols = filings_lf.collect_schema().names()
        items_cols = items_lf.collect_schema().names()
        if "accession_number" not in filings_cols:
            raise ValueError("accession_number column not found in year_filings_parquet")
        if "accession_number" not in items_cols:
            raise ValueError("accession_number column not found in year_items_parquet")
        if "full_text" not in filings_cols:
            raise ValueError("full_text column not found in year_filings_parquet")

        year_label = "UNKNOWN"
        if isinstance(year_filings_parquet, Path):
            stem = Path(year_filings_parquet).stem
            if stem.isdigit() and len(stem) == 4:
                year_label = stem

        items_accessions = (
            items_lf.select("accession_number")
            .unique()
            .filter(pl.col("accession_number").is_not_null())
        )
        eligible_filings = filings_lf.filter(pl.col("accession_number").is_not_null())

        doc_type_expr = (
            pl.col("document_type_filename").fill_null("UNKNOWN").alias("document_type_filename")
        )
        text_len_expr = pl.col("full_text").fill_null("").str.len_chars()

        with_items_lf = eligible_filings.join(
            items_accessions, on="accession_number", how="semi"
        )
        no_items_lf = eligible_filings.join(items_accessions, on="accession_number", how="anti")

        with_stats = with_items_lf.group_by(doc_type_expr).agg(
            pl.len().alias("n_with_items"),
            text_len_expr.mean().alias("avg_text_len_with_items"),
        )
        no_stats = no_items_lf.group_by(doc_type_expr).agg(
            pl.len().alias("n_no_items"),
            text_len_expr.mean().alias("avg_text_len_no_items"),
        )

        stats_df = (
            with_stats.join(no_stats, on="document_type_filename", how="outer")
            .with_columns(
                pl.col("n_with_items").fill_null(0).cast(pl.Int64),
                pl.col("n_no_items").fill_null(0).cast(pl.Int64),
                pl.col("avg_text_len_with_items").fill_null(0.0).cast(pl.Float64),
                pl.col("avg_text_len_no_items").fill_null(0.0).cast(pl.Float64),
            )
            .with_columns(
                (pl.col("n_with_items") + pl.col("n_no_items")).alias("n_filings_eligible"),
                pl.when((pl.col("n_with_items") + pl.col("n_no_items")) > 0)
                .then(
                    pl.col("n_no_items")
                    / (pl.col("n_with_items") + pl.col("n_no_items"))
                )
                .otherwise(0.0)
                .alias("share_no_item"),
                pl.lit(year_label).alias("year"),
            )
            .select(_NO_ITEM_STATS_COLUMNS)
            .collect()
        )

        stats_rows = stats_df.to_dicts()
        if stats_rows:
            total_with = sum(row["n_with_items"] for row in stats_rows)
            total_no = sum(row["n_no_items"] for row in stats_rows)
            total_len_with = sum(
                row["avg_text_len_with_items"] * row["n_with_items"] for row in stats_rows
            )
            total_len_no = sum(
                row["avg_text_len_no_items"] * row["n_no_items"] for row in stats_rows
            )
            total = total_with + total_no
            stats_rows.append(
                {
                    "year": year_label,
                    "document_type_filename": "TOTAL",
                    "n_filings_eligible": total,
                    "n_with_items": total_with,
                    "n_no_items": total_no,
                    "share_no_item": (total_no / total) if total else 0.0,
                    "avg_text_len_with_items": (
                        total_len_with / total_with if total_with else 0.0
                    ),
                    "avg_text_len_no_items": total_len_no / total_no if total_no else 0.0,
                }
            )

        out_no_item_parquet = Path(out_no_item_parquet)
        out_stats_csv = Path(out_stats_csv)
        out_no_item_parquet.parent.mkdir(parents=True, exist_ok=True)
        out_stats_csv.parent.mkdir(parents=True, exist_ok=True)

        no_items_out = no_items_lf
        if not include_full_text and "full_text" in filings_cols:
            no_items_out = no_items_out.drop("full_text")

        tmp_no_item = local_work_dir / out_no_item_parquet.name
        if tmp_no_item.exists():
            tmp_no_item.unlink()
        no_items_out.sink_parquet(tmp_no_item, compression="zstd")
        _assert_parquet_magic(tmp_no_item)
        _copy_with_verify(tmp_no_item, out_no_item_parquet, retries=3, sleep=1.0, validate=True)

        tmp_stats = local_work_dir / out_stats_csv.name
        if tmp_stats.exists():
            tmp_stats.unlink()
        _write_no_item_stats_csv(tmp_stats, stats_rows)
        _copy_with_verify(tmp_stats, out_stats_csv, retries=3, sleep=1.0, validate=False)
    finally:
        for p in cleanup_paths:
            p.unlink(missing_ok=True)

    return out_no_item_parquet


def process_year_dir_extract_items(
    year_dir: Path,
    out_dir: Path,
    *,
    years: list[str] | None = None,
    parquet_batch_rows: int = 16,
    out_batch_max_rows: int = 50_000,
    out_batch_max_text_bytes: int = 250 * 1024 * 1024,
    tmp_dir: Path | None = None,
    compression: Literal["zstd", "snappy", "gzip", "uncompressed"] = "zstd",
    local_work_dir: Path | None = None,
    copy_retries: int = 3,
    copy_sleep: float = 1.0,
    validate_on_copy: bool | Literal["quick", "full"] = True,
    non_item_diagnostic: bool = False,
    include_full_text: bool = False,
    regime: bool = True,
) -> list[Path]:
    """
    Process a directory of per-year merged filing parquets (e.g., 2024.parquet) into
    per-year item parquets in `out_dir`.
    """
    year_dir = Path(year_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted = set(years) if years else None
    out_paths: list[Path] = []
    for p in sorted(year_dir.glob("*.parquet")):
        if wanted is not None and p.stem not in wanted:
            continue
        out_paths.append(
            process_year_parquet_extract_items(
                year_parquet_path=p,
                out_dir=out_dir,
                parquet_batch_rows=parquet_batch_rows,
                out_batch_max_rows=out_batch_max_rows,
                out_batch_max_text_bytes=out_batch_max_text_bytes,
                tmp_dir=tmp_dir,
                compression=compression,
                local_work_dir=local_work_dir,
                copy_retries=copy_retries,
                copy_sleep=copy_sleep,
                validate_on_copy=validate_on_copy,
                non_item_diagnostic=non_item_diagnostic,
                include_full_text=include_full_text,
                regime=regime,
            )
        )

    return out_paths


def _process_year_parquet_extract_non_items(
    year_parquet_path: Path,
    item_parquet_path: Path,
    out_dir: Path,
    *,
    parquet_batch_rows: int = 16,
    out_batch_max_rows: int = 50_000,
    out_batch_max_text_bytes: int = 250 * 1024 * 1024,
    tmp_dir: Path | None = None,
    compression: Literal["zstd", "snappy", "gzip", "uncompressed"] = "zstd",
    local_work_dir: Path | None = None,
    copy_retries: int = 3,
    copy_sleep: float = 1.0,
    validate_on_copy: bool | Literal["quick", "full"] = True,
) -> tuple[Path, dict[str, dict[str, int]]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = tmp_dir or Path(tempfile.gettempdir())
    tmp_dir.mkdir(parents=True, exist_ok=True)

    local_work_dir = local_work_dir or (Path(tempfile.gettempdir()) / "_non_item_work")
    local_work_dir.mkdir(parents=True, exist_ok=True)
    year = year_parquet_path.stem
    local_year_dir = local_work_dir / f"{year}_no_items"
    local_year_dir.mkdir(parents=True, exist_ok=True)

    local_year = tmp_dir / f"{year}_full{year_parquet_path.suffix}"
    local_items = tmp_dir / f"{year}_items{item_parquet_path.suffix}"
    cleanup_year = False
    cleanup_items = False

    if year_parquet_path.resolve() != local_year.resolve():
        _copy_with_verify(
            year_parquet_path,
            local_year,
            retries=copy_retries,
            sleep=copy_sleep,
            validate=True,
        )
        cleanup_year = True
    else:
        local_year = year_parquet_path

    if item_parquet_path.resolve() != local_items.resolve():
        _copy_with_verify(
            item_parquet_path,
            local_items,
            retries=copy_retries,
            sleep=copy_sleep,
            validate=True,
        )
        cleanup_items = True
    else:
        local_items = item_parquet_path

    out_path = out_dir / f"{year}.parquet"
    item_accessions = _load_item_accession_set(local_items)

    stats: dict[str, dict[str, int]] = {}
    non_item_batches: list[Path] = []
    non_item_records: list[dict] = []
    non_item_text_bytes = 0
    non_item_batch_idx = 1
    input_schema: pa.Schema | None = None

    pf: pq.ParquetFile | None = None
    try:
        pf = pq.ParquetFile(local_year)
        input_schema = pf.schema_arrow
        columns = pf.schema.names
        for batch in pf.iter_batches(batch_size=parquet_batch_rows, columns=columns):
            tbl = pa.Table.from_batches([batch])
            df = pl.from_arrow(tbl)

            for row in df.iter_rows(named=True):
                filing_text = row.get("full_text")
                text_len = len(filing_text) if filing_text else 0
                acc = row.get("accession_number")
                eligible = acc is not None
                if not eligible:
                    continue
                has_items = acc in item_accessions
                _update_no_item_stats(
                    stats,
                    row.get("document_type_filename"),
                    has_items,
                    text_len,
                )
                if has_items:
                    continue

                non_item_records.append(row)
                non_item_text_bytes += text_len
                if (
                    len(non_item_records) >= out_batch_max_rows
                    or non_item_text_bytes >= out_batch_max_text_bytes
                ):
                    local_batch = (
                        local_year_dir / f"{year}_no_items_batch_{non_item_batch_idx:04d}.parquet"
                    )
                    if local_batch.exists():
                        local_batch.unlink()
                    _flush_non_item_batch(non_item_records, local_batch, compression)
                    non_item_batches.append(local_batch)
                    non_item_records = []
                    non_item_text_bytes = 0
                    non_item_batch_idx += 1
                    gc.collect()

        if non_item_records:
            local_batch = (
                local_year_dir / f"{year}_no_items_batch_{non_item_batch_idx:04d}.parquet"
            )
            if local_batch.exists():
                local_batch.unlink()
            _flush_non_item_batch(non_item_records, local_batch, compression)
            non_item_batches.append(local_batch)
    finally:
        if pf is not None:
            try:
                pf.close()
            except Exception:
                pass
        if cleanup_year:
            local_year.unlink(missing_ok=True)
        if cleanup_items:
            local_items.unlink(missing_ok=True)
        gc.collect()

    tmp_out = local_year_dir / f"{year}.parquet"
    if tmp_out.exists():
        tmp_out.unlink()
    if not non_item_batches:
        if input_schema is None:
            raise ValueError(f"Missing input schema for non-item output of {year_parquet_path}")
        arrays = [pa.array([], type=field.type) for field in input_schema]
        empty_tbl = pa.Table.from_arrays(arrays, schema=input_schema)
        pq.write_table(empty_tbl, tmp_out, compression=compression)
        _assert_parquet_magic(tmp_out)
    else:
        concat_parquets_arrow(
            in_files=non_item_batches,
            out_path=tmp_out,
            batch_size=32_000,
            compression=compression,
            compression_level=1,
            stage_dir=None,
        )
        _assert_parquet_magic(tmp_out)

    _copy_with_verify(
        tmp_out,
        out_path,
        retries=copy_retries,
        sleep=copy_sleep,
        validate=validate_on_copy,
    )

    for b in non_item_batches:
        b.unlink(missing_ok=True)

    return out_path, stats


def process_year_parquet_extract_non_items(
    year_parquet_path: Path,
    item_parquet_path: Path,
    out_dir: Path,
    *,
    parquet_batch_rows: int = 16,
    out_batch_max_rows: int = 50_000,
    out_batch_max_text_bytes: int = 250 * 1024 * 1024,
    tmp_dir: Path | None = None,
    compression: Literal["zstd", "snappy", "gzip", "uncompressed"] = "zstd",
    local_work_dir: Path | None = None,
    copy_retries: int = 3,
    copy_sleep: float = 1.0,
    validate_on_copy: bool | Literal["quick", "full"] = True,
    stats_out_path: Path | None = None,
) -> Path:
    """
    Build a per-year parquet of filings with no extracted items by comparing
    a full-text yearly parquet to its item parquet (via accession_number).
    """
    out_path, stats = _process_year_parquet_extract_non_items(
        year_parquet_path=year_parquet_path,
        item_parquet_path=item_parquet_path,
        out_dir=out_dir,
        parquet_batch_rows=parquet_batch_rows,
        out_batch_max_rows=out_batch_max_rows,
        out_batch_max_text_bytes=out_batch_max_text_bytes,
        tmp_dir=tmp_dir,
        compression=compression,
        local_work_dir=local_work_dir,
        copy_retries=copy_retries,
        copy_sleep=copy_sleep,
        validate_on_copy=validate_on_copy,
    )

    year = year_parquet_path.stem
    stats_path = stats_out_path or (out_dir / f"{year}_no_item_stats.csv")
    stats_rows = _build_no_item_stats_rows(stats, year)
    stats_rows.extend(_build_no_item_stats_rows(stats, "ALL"))
    _write_no_item_stats_csv(stats_path, stats_rows)

    return out_path


def process_year_dir_extract_non_items(
    year_dir: Path,
    item_dir: Path,
    out_dir: Path,
    *,
    years: list[str] | None = None,
    parquet_batch_rows: int = 16,
    out_batch_max_rows: int = 50_000,
    out_batch_max_text_bytes: int = 250 * 1024 * 1024,
    tmp_dir: Path | None = None,
    compression: Literal["zstd", "snappy", "gzip", "uncompressed"] = "zstd",
    local_work_dir: Path | None = None,
    copy_retries: int = 3,
    copy_sleep: float = 1.0,
    validate_on_copy: bool | Literal["quick", "full"] = True,
    stats_out_path: Path | None = None,
) -> list[Path]:
    """
    Build per-year non-item filing parquets by comparing year parquets to item parquets.
    Writes a combined stats CSV (per-year + full sample) to stats_out_path.
    """
    year_dir = Path(year_dir)
    item_dir = Path(item_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted = set(years) if years else None
    out_paths: list[Path] = []
    all_stats: dict[str, dict[str, int]] = {}
    stats_rows: list[dict] = []

    for p in sorted(year_dir.glob("*.parquet")):
        if wanted is not None and p.stem not in wanted:
            continue
        item_path = item_dir / f"{p.stem}.parquet"
        if not item_path.exists():
            raise FileNotFoundError(f"Missing item parquet for {p.stem}: {item_path}")

        out_path, year_stats = _process_year_parquet_extract_non_items(
            year_parquet_path=p,
            item_parquet_path=item_path,
            out_dir=out_dir,
            parquet_batch_rows=parquet_batch_rows,
            out_batch_max_rows=out_batch_max_rows,
            out_batch_max_text_bytes=out_batch_max_text_bytes,
            tmp_dir=tmp_dir,
            compression=compression,
            local_work_dir=local_work_dir,
            copy_retries=copy_retries,
            copy_sleep=copy_sleep,
            validate_on_copy=validate_on_copy,
        )
        out_paths.append(out_path)
        stats_rows.extend(_build_no_item_stats_rows(year_stats, p.stem))
        _merge_no_item_stats(all_stats, year_stats)

    if out_paths:
        stats_rows.extend(_build_no_item_stats_rows(all_stats, "ALL"))
        stats_path = stats_out_path or (out_dir / "non_item_stats.csv")
        _write_no_item_stats_csv(stats_path, stats_rows)

    return out_paths


def summarize_item_year_parquets(parquet_dir: Path) -> list[dict]:
    """
    Inspect yearly item parquet files and return a summary per file.
    Each summary includes: path, year, rows, unique_doc_ids (if present), and status.
    """
    summaries: list[dict] = []
    for path in sorted(Path(parquet_dir).glob("*.parquet")):
        year = path.stem
        if not (year.isdigit() and len(year) == 4):
            continue
        size_bytes = path.stat().st_size
        try:
            lf = pl.scan_parquet(path)
            columns = lf.collect_schema().names()
            rows = lf.select(pl.len()).collect().item()
            doc_ids = None
            if "doc_id" in columns:
                doc_ids = lf.select(pl.col("doc_id").n_unique()).collect().item()

            status = "OK"
            if rows == 0:
                status = "EMPTY"
            elif "item" not in columns or "full_text" not in columns:
                status = "WARN_SCHEMA"
        except Exception as exc:
            columns = []
            rows = None
            doc_ids = None
            status = f"ERROR: {exc}"

        summaries.append(
            {
                "path": path,
                "year": year,
                "rows": rows,
                "unique_doc_ids": doc_ids,
                "size_bytes": size_bytes,
                "columns": columns,
                "status": status,
            }
        )

    return summaries


def merge_parquet_files_arrow(
    in_files: list[Path],
    out_path: Path,
    batch_size: int = 32_000,
    compression: str = "zstd",
    compression_level: int | None = 1,
) -> Path:
    """
    Compatibility wrapper around concat_parquets_arrow (streaming append).
    """
    return concat_parquets_arrow(
        in_files=in_files,
        out_path=out_path,
        batch_size=batch_size,
        compression=compression,
        compression_level=compression_level,
    )


def build_light_metadata(
    year_files: list[Path],
    out_path: Path,
    sort_columns: tuple[str, ...] = ("filing_date", "cik"),
    compression: str = "zstd",
) -> Path:
    """
    Compatibility wrapper that drops `full_text` and sorts by filing_date, cik when present.
    """
    return build_light_metadata_dataset(
        parquet_dir=year_files,
        out_path=out_path,
        drop_columns=("full_text",),
        sort_columns=sort_columns,
        compression=compression,
    )

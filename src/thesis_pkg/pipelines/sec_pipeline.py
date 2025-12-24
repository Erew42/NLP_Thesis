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
    df = df.with_columns(pl.col("file_date_filename").cast(pl.Date, strict=False))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    try:
        tmp_path.unlink(missing_ok=True)
        df.write_parquet(tmp_path, compression=compression)
        _assert_parquet_magic(tmp_path)
        os.replace(tmp_path, out_path)
    finally:
        tmp_path.unlink(missing_ok=True)


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

                rec = {
                    **meta,
                    "filename": filename,
                    "zip_member_path": member,
                    "full_text": txt,
                    "filing_date": final_date,
                    "filing_date_header": hdr["header_filing_date_str"],
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
) -> list[Path]:
    """
    Merge per-year parquet batches (e.g., 2020_batch_0001.parquet) into one parquet per year.
    Returns paths of merged parquet files.

    Designed for slow/remote filesystems (e.g., Drive): writes to a local work dir
    first, optionally stages/validates inputs, then copies the result to `out_dir` with validation/retries.
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
) -> Path:
    """
    Expand a merged yearly filing parquet (one row per filing with `full_text`)
    into a yearly item parquet (one row per extracted item).

    Output schema:
      - filing identifiers (doc_id, cik, accession_number, ...)
      - item_part, item_id, item
      - full_text contains the extracted item text (pagination artifacts removed)

    Writes intermediate batches to a local work directory and then copies the final
    yearly parquet to `out_dir` with retries/validation (helps on remote mounts).
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

    columns = [
        "doc_id",
        "cik",
        "cik_10",
        "accession_number",
        "accession_nodash",
        "file_date_filename",
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

    pf: pq.ParquetFile | None = None
    try:
        pf = pq.ParquetFile(local_in)
        for batch in pf.iter_batches(batch_size=parquet_batch_rows, columns=columns):
            tbl = pa.Table.from_batches([batch])
            df = pl.from_arrow(tbl)

            for row in df.iter_rows(named=True):
                filing_rows += 1
                filing_text = row.get("full_text")
                if not filing_text:
                    no_item_filings += 1
                    continue

                try:
                    items = extract_filing_items(
                        filing_text,
                        form_type=row.get("document_type_filename"),
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
                    no_item_filings += 1
                    continue

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
                        "document_type_filename": row.get("document_type_filename"),
                        "filename": row.get("filename"),
                        "item_part": item.get("item_part"),
                        "item_id": item.get("item_id"),
                        "item": item.get("item"),
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

    print(
        f"[items] {year} filings={filing_rows} items={item_rows} "
        f"items_per_filing={item_rows / filing_rows if filing_rows else 0:.2f} "
        f"no_item_filings={no_item_filings} empty_items={empty_items} -> {out_path}"
    )

    return out_path


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
            )
        )

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

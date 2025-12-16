from __future__ import annotations

import gc
import io
import json
import re
import shutil
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq


FILENAME_PATTERN = re.compile(
    r"(\d{8})_"          # 1: Date (YYYYMMDD)
    r"([^_]+)_"          # 2: Type
    r"edgar_data_"
    r"(\d+)_"            # 3: CIK
    r"([\d-]+)"          # 4: Accession Number
    r"\.txt$",
    re.IGNORECASE,
)

HEADER_SEARCH_LIMIT_DEFAULT = 5000

CIK_HEADER_PATTERN = re.compile(r"CENTRAL INDEX KEY:\s*(\d+)")
DATE_HEADER_PATTERN = re.compile(r"FILED AS OF DATE:\s*(\d{8})")
ACC_HEADER_PATTERN = re.compile(r"ACCESSION NUMBER:\s*([\d-]+)")


def _digits_only(s: str | None) -> str | None:
    if not s:
        return None
    d = re.sub(r"\D", "", s)
    return d or None


def _cik_10(cik: int | None) -> str | None:
    if cik is None:
        return None
    return str(int(cik)).zfill(10)


def _make_doc_id(cik10: str | None, acc: str | None) -> str | None:
    # Keep this stable and readable; you can later switch to a hash if desired.
    if acc is None:
        return None
    return f"{cik10}:{acc}" if cik10 else f"UNK:{acc}"


def parse_filename_minimal(filename: str) -> dict:
    """
    Parse all usable info from filename only.
    Returns dict with parse_ok plus components.
    """
    m = FILENAME_PATTERN.match(filename)
    if not m:
        return {
            "filename_parse_ok": False,
            "file_date_filename": None,
            "document_type_filename": None,
            "cik": None,
            "cik_10": None,
            "accession_number": None,
            "accession_nodash": None,
            "doc_id": None,
        }

    date_str = m.group(1)
    doc_type = m.group(2)
    try:
        cik_int = int(m.group(3))
    except Exception:
        cik_int = None
    acc = m.group(4)

    cik10 = _cik_10(cik_int)
    acc_nodash = _digits_only(acc)
    doc_id = _make_doc_id(cik10, acc)

    return {
        "filename_parse_ok": True,
        "file_date_filename": date_str,          # parse to Date at write time
        "document_type_filename": doc_type,
        "cik": cik_int,
        "cik_10": cik10,
        "accession_number": acc,
        "accession_nodash": acc_nodash,
        "doc_id": doc_id,
    }


@dataclass
class RawTextSchema:
    schema = {
        "doc_id": pl.Utf8,
        "cik": pl.Int64,
        "cik_10": pl.Utf8,
        "accession_number": pl.Utf8,
        "accession_nodash": pl.Utf8,
        "file_date_filename": pl.Utf8,            # will cast to Date on write
        "document_type_filename": pl.Utf8,
        "filename": pl.Utf8,
        "zip_member_path": pl.Utf8,
        "filename_parse_ok": pl.Boolean,
        "full_text": pl.Utf8,
    }


@dataclass
class ParsedFilingSchema:
    schema = {
        "doc_id": pl.Utf8,
        "cik": pl.Int64,
        "cik_10": pl.Utf8,
        "accession_number": pl.Utf8,
        "accession_nodash": pl.Utf8,
        "filing_date": pl.Utf8,  # will cast to Date on write
        "filing_date_header": pl.Utf8,
        "file_date_filename": pl.Utf8,  # will cast to Date on write
        "document_type_filename": pl.Utf8,
        "filename": pl.Utf8,
        "zip_member_path": pl.Utf8,
        "filename_parse_ok": pl.Boolean,
        "cik_header_primary": pl.Utf8,
        "ciks_header_secondary": pl.List(pl.Utf8),
        "accession_header": pl.Utf8,
        "cik_conflict": pl.Boolean,
        "accession_conflict": pl.Boolean,
        "full_text": pl.Utf8,
    }


def parse_header(full_text: str, header_search_limit: int = HEADER_SEARCH_LIMIT_DEFAULT) -> dict:
    """
    Extract header metadata (CIKs, accession, filing date) from the top of a filing.
    """
    header = full_text[:header_search_limit]

    header_ciks = CIK_HEADER_PATTERN.findall(header)
    header_ciks_int_set = {int(c) for c in header_ciks if c.isdigit()}

    date_match = DATE_HEADER_PATTERN.search(header)
    header_filing_date_str = date_match.group(1) if date_match else None

    acc_match = ACC_HEADER_PATTERN.search(header)
    header_accession_str = acc_match.group(1) if acc_match else None

    primary_header_cik = header_ciks[0] if header_ciks else None
    secondary_ciks = header_ciks[1:] if len(header_ciks) > 1 else []

    return {
        "header_ciks_int_set": header_ciks_int_set,
        "header_filing_date_str": header_filing_date_str,
        "header_accession_str": header_accession_str,
        "primary_header_cik": primary_header_cik,
        "secondary_ciks": secondary_ciks,
    }


def _flush_batch(records: list[dict], out_path: Path, compression: str) -> None:
    if not records:
        return
    df = pl.DataFrame(records, schema=RawTextSchema.schema)
    df = df.with_columns(
        pl.col("file_date_filename").str.strptime(pl.Date, "%Y%m%d", strict=False)
    )
    df.write_parquet(out_path, compression=compression)


def _flush_parsed_batch(records: list[dict], out_path: Path, compression: str) -> None:
    if not records:
        return
    df = pl.DataFrame(records, schema=ParsedFilingSchema.schema)
    df = df.with_columns(
        pl.col("filing_date").str.strptime(pl.Date, "%Y%m%d", strict=False),
        pl.col("file_date_filename").str.strptime(pl.Date, "%Y%m%d", strict=False),
    )
    df.write_parquet(out_path, compression=compression)


def process_zip_year_raw_text(
    zip_path: Path,
    out_dir: Path,
    batch_max_rows: int = 1000,
    batch_max_text_bytes: int = 250 * 1024 * 1024,  # 250 MB
    tmp_dir: Path | None = None,
    encoding: str = "utf-8",
    compression: Literal["zstd", "snappy", "gzip", "uncompressed"] = "zstd",
    local_work_dir: Path | None = None,
) -> list[Path]:
    """
    Step 1: Read yearly ZIP, parse filename fields, store full_text + filename-derived metadata in parquet batches.
    Writes batches to a local work dir first, then moves them to out_dir and returns list of final batch paths.

    Notes:
      - Streams from ZIP without unpacking to disk.
      - Copies ZIP to local tmp_dir first (important when source is a network mount like Drive).
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
        if local_zip.exists():
            local_zip.unlink()
        shutil.copyfile(src_zip, local_zip)
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
                    if final_out_file.exists():
                        final_out_file.unlink()
                    shutil.move(str(local_file), str(final_out_file))
                    written.append(final_out_file)

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
                if final_out_file.exists():
                    final_out_file.unlink()
                shutil.move(str(local_file), str(final_out_file))
                written.append(final_out_file)

    finally:
        if cleanup_local:
            local_zip.unlink(missing_ok=True)
        gc.collect()

    return written


def concat_parquets_arrow(
    in_files: list[Path],
    out_path: Path,
    batch_size: int = 32_000,
    compression: str = "zstd",
    compression_level: int | None = 1,
) -> Path:
    """
    Stream-concatenate parquet files without loading everything into memory.
    """
    if not in_files:
        raise ValueError("in_files is empty; nothing to concatenate.")

    writer: pq.ParquetWriter | None = None
    try:
        for f in in_files:
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=batch_size):
                tbl = pa.Table.from_batches([batch])
                if writer is None:
                    writer = pq.ParquetWriter(
                        out_path,
                        tbl.schema,
                        compression=compression,
                        compression_level=compression_level if compression == "zstd" else None,
                    )
                writer.write_table(tbl)
    finally:
        if writer is not None:
            writer.close()

    return out_path


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
) -> list[Path]:
    """
    Parse a yearly ZIP of filings into parquet batches with header metadata and conflict flags.

    Batches are written to a local work directory first, then moved to out_dir (helps when out_dir is remote).
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
        if local_zip.exists():
            local_zip.unlink()
        shutil.copyfile(src_zip, local_zip)
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
                    if final_out_file.exists():
                        final_out_file.unlink()
                    shutil.move(str(local_file), str(final_out_file))
                    written.append(final_out_file)
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
                if final_out_file.exists():
                    final_out_file.unlink()
                shutil.move(str(local_file), str(final_out_file))
                written.append(final_out_file)

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
) -> list[Path]:
    """
    Merge per-year parquet batches (e.g., 2020_batch_0001.parquet) into one parquet per year.
    Returns paths of merged parquet files.

    Designed for slow/remote filesystems (e.g., Drive): writes to a local work dir
    first, then moves the result to `out_dir`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    local_work_dir = local_work_dir or (Path(tempfile.gettempdir()) / "_merge_work")
    local_work_dir.mkdir(parents=True, exist_ok=True)

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
        tmp_path = local_work_dir / f"{year}.parquet"
        if tmp_path.exists():
            tmp_path.unlink()

        concat_parquets_arrow(
            files,
            tmp_path,
            batch_size=batch_size,
            compression=compression,
            compression_level=compression_level,
        )
        shutil.move(str(tmp_path), str(out_path))
        merged.append(out_path)

        if checkpoint_path:
            done.add(year)
            checkpoint_path.write_text(json.dumps(sorted(done)))

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

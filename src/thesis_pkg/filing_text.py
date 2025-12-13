from __future__ import annotations

import gc
import io
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import polars as pl


FILENAME_PATTERN = re.compile(
    r"(\d{8})_"          # 1: Date (YYYYMMDD)
    r"([^_]+)_"          # 2: Type
    r"edgar_data_"
    r"(\d+)_"            # 3: CIK
    r"([\d-]+)"          # 4: Accession Number
    r"\.txt$",
    re.IGNORECASE,
)


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


def _flush_batch(records: list[dict], out_path: Path, compression: str) -> None:
    if not records:
        return
    df = pl.DataFrame(records, schema=RawTextSchema.schema)
    df = df.with_columns(
        pl.col("file_date_filename").str.strptime(pl.Date, "%Y%m%d", strict=False)
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
) -> list[Path]:
    """
    Step 1: Read yearly ZIP, parse filename fields, store full_text + filename-derived metadata in parquet batches.
    Writes batches to out_dir and returns list of batch paths.

    Notes:
      - Streams from ZIP without unpacking to disk.
      - Copies ZIP to local tmp_dir first (important when source is a network mount like Drive).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = tmp_dir or Path(tempfile.gettempdir())
    tmp_dir.mkdir(parents=True, exist_ok=True)

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

                # read and decode filing
                with zf.open(member, "r") as bf:
                    txt = io.TextIOWrapper(bf, encoding=encoding, errors="replace").read()

                rec = {
                    **meta,
                    "filename": filename,
                    "zip_member_path": member,
                    "full_text": txt,
                }
                records.append(rec)
                text_bytes += len(txt.encode("utf-8", errors="ignore"))

                if len(records) >= batch_max_rows or text_bytes >= batch_max_text_bytes:
                    out_file = out_dir / f"{zip_path.stem}_batch_{batch_idx:04d}.parquet"
                    _flush_batch(records, out_file, compression)
                    written.append(out_file)

                    records = []
                    batch_idx += 1
                    text_bytes = 0
                    gc.collect()

            # final batch
            if records:
                out_file = out_dir / f"{zip_path.stem}_batch_{batch_idx:04d}.parquet"
                _flush_batch(records, out_file, compression)
                written.append(out_file)

    finally:
        if cleanup_local:
            local_zip.unlink(missing_ok=True)
        gc.collect()

    return written

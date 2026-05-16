from __future__ import annotations

import io
import math
import os
import shutil
import time
from pathlib import Path
from typing import Iterable, Literal

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from thesis_native import _lm2011_rust
except Exception as exc:  # pragma: no cover - optional native extension
    _lm2011_rust = None
    _PARQUET_RUST_IMPORT_ERROR: str | None = f"{type(exc).__name__}: {exc}"
else:
    _PARQUET_RUST_IMPORT_ERROR = None


PARQUET_MAGIC = b"PAR1"

_PARQUET_RUST_METRICS: dict[str, int] = {
    "magic_probe_fast_success": 0,
    "magic_probe_fast_failures": 0,
    "magic_probe_fallbacks": 0,
    "copy_file_stream_fast_success": 0,
    "copy_file_stream_fast_failures": 0,
    "copy_file_stream_fallbacks": 0,
}


def get_parquet_rust_accel_metrics() -> dict[str, int | str | bool | None]:
    metrics: dict[str, int | str | bool | None] = dict(_PARQUET_RUST_METRICS)
    metrics["rust_accel_available"] = _lm2011_rust is not None
    metrics["rust_accel_import_error"] = _PARQUET_RUST_IMPORT_ERROR
    return metrics


def reset_parquet_rust_accel_metrics() -> None:
    for key in _PARQUET_RUST_METRICS:
        _PARQUET_RUST_METRICS[key] = 0


def load_tables(base_dirs: Iterable[Path], wanted: Iterable[str]) -> dict[str, pl.LazyFrame]:
    """Load parquet files into LazyFrames keyed by filename stem."""
    wanted_set = set(wanted)
    tables: dict[str, pl.LazyFrame] = {}
    for base in base_dirs:
        if not base.exists():
            continue
        for path in base.rglob("*.parquet"):
            key = path.stem
            if key in tables or key not in wanted_set:
                continue
            tables[key] = pl.scan_parquet(path)
    return tables


def sink_exact_firm_sample_from_parquet(
    full_source: Path | pl.LazyFrame,
    sample_path: Path | None = None,
    firm_col: str = "KYPERMNO",
    frac: float = 0.01,
    seed: int = 42,
    compression: Literal["lz4", "uncompressed", "snappy", "gzip", "lzo", "brotli", "zstd"] = "zstd",
    save_firm_list_path: Path | None = None,
) -> Path | pl.LazyFrame:
    """
    Sample an exact fraction of firms; optionally persist the result.

    Returns a Path when `sample_path` is provided, otherwise the filtered LazyFrame.
    """
    if not (0 < frac <= 1):
        raise ValueError("frac must be in (0, 1].")

    lf = full_source if isinstance(full_source, pl.LazyFrame) else pl.scan_parquet(full_source)

    firms = (
        lf.select(pl.col(firm_col).unique().drop_nulls())
        .collect()
        .get_column(firm_col)
    )

    k = max(1, math.ceil(len(firms) * frac))
    chosen_df = pl.DataFrame({firm_col: firms.sample(n=k, with_replacement=False, seed=seed)})

    if save_firm_list_path is not None:
        chosen_df.write_parquet(save_firm_list_path, compression=compression)

    sampled = lf.join(chosen_df.lazy(), on=firm_col, how="inner")

    if sample_path is not None:
        sampled.sink_parquet(sample_path, compression=compression)
        return sample_path

    return sampled


def _assert_parquet_magic_parts(path: Path, *, size: int, start: bytes, end: bytes) -> None:
    if size < 8:
        raise OSError(f"Parquet file too small to be valid ({size} bytes): {path}")

    if start != PARQUET_MAGIC:
        raise OSError(f"Parquet magic header missing for {path} (got {start!r})")
    if end != PARQUET_MAGIC:
        raise OSError(f"Parquet magic footer missing for {path} (got {end!r})")


def _assert_parquet_magic_py(path: Path) -> None:
    """
    Quick integrity check for common truncation/corruption cases:
    parquet files must start and end with the magic bytes PAR1.
    """
    try:
        size = path.stat().st_size
    except FileNotFoundError as exc:
        raise OSError(f"Parquet file not found: {path}") from exc

    if size < 8:
        _assert_parquet_magic_parts(path, size=size, start=b"", end=b"")

    with path.open("rb") as f:
        start = f.read(4)
        f.seek(-4, io.SEEK_END)
        end = f.read(4)

    _assert_parquet_magic_parts(path, size=size, start=start, end=end)


def _assert_parquet_magic(path: Path) -> None:
    """
    Quick integrity check for common truncation/corruption cases:
    parquet files must start and end with the magic bytes PAR1.
    """
    if _lm2011_rust is not None:
        try:
            size, start_raw, end_raw = _lm2011_rust.parquet_magic_probe(str(path))
        except Exception:
            _PARQUET_RUST_METRICS["magic_probe_fast_failures"] += 1
        else:
            _PARQUET_RUST_METRICS["magic_probe_fast_success"] += 1
            _assert_parquet_magic_parts(
                path,
                size=int(size),
                start=bytes(start_raw),
                end=bytes(end_raw),
            )
            return

    _PARQUET_RUST_METRICS["magic_probe_fallbacks"] += 1
    _assert_parquet_magic_py(path)


def _copy_file_stream_py(src: Path, dst: Path, buffer_size: int = 16 * 1024 * 1024) -> None:
    """
    Stream copy with explicit flush/fsync for stability on network/remote mounts.
    """
    with src.open("rb") as r, dst.open("wb") as w:
        shutil.copyfileobj(r, w, length=buffer_size)
        w.flush()
        try:
            os.fsync(w.fileno())
        except OSError:
            # Some filesystems (e.g., FUSE mounts) may not support fsync reliably.
            pass


def _copy_file_stream(src: Path, dst: Path, buffer_size: int = 16 * 1024 * 1024) -> None:
    """
    Stream copy with explicit flush/fsync for stability on network/remote mounts.
    """
    if _lm2011_rust is not None and buffer_size > 0:
        try:
            _lm2011_rust.copy_file_stream(str(src), str(dst), int(buffer_size))
        except Exception:
            _PARQUET_RUST_METRICS["copy_file_stream_fast_failures"] += 1
        else:
            _PARQUET_RUST_METRICS["copy_file_stream_fast_success"] += 1
            return

    _PARQUET_RUST_METRICS["copy_file_stream_fallbacks"] += 1
    _copy_file_stream_py(src, dst, buffer_size=buffer_size)


def _validate_parquet_quick(path: Path) -> None:
    """
    Minimal integrity check: ensure footer/row-group headers are readable.
    """
    _assert_parquet_magic(path)
    pf = pq.ParquetFile(path)
    try:
        _ = pf.metadata
        # Touch at least one data batch to catch obvious page/header corruption early.
        for _ in pf.iter_batches(batch_size=1):
            break
    finally:
        pf.close()


def _validate_parquet_full(path: Path, *, batch_size: int = 32_000) -> None:
    """
    Strong integrity check: fully stream-read the file with pyarrow.
    This is slower but catches corruption that only appears later in the file.
    """
    _assert_parquet_magic(path)
    pf = pq.ParquetFile(path)
    try:
        for _ in pf.iter_batches(batch_size=batch_size):
            pass
    finally:
        pf.close()


def _copy_with_verify(
    src: Path,
    dst: Path,
    *,
    retries: int = 3,
    sleep: float = 1.0,
    validate: bool | Literal["quick", "full"] = True,
) -> Path:
    """
    Copy src -> dst with size check, optional parquet validation, and retries.
    Useful for flaky mounts (e.g., Drive).
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    expected_size = src.stat().st_size
    validate_mode: Literal["none", "quick", "full"]
    if validate is True:
        validate_mode = "quick"
    elif validate is False:
        validate_mode = "none"
    else:
        validate_mode = validate

    for attempt in range(1, retries + 1):
        tmp = dst.with_suffix(dst.suffix + ".partial")
        if tmp.exists():
            tmp.unlink()

        _copy_file_stream(src, tmp)

        ok = tmp.stat().st_size == expected_size
        if ok and validate_mode != "none":
            try:
                if validate_mode == "quick":
                    _validate_parquet_quick(tmp)
                else:
                    _validate_parquet_full(tmp)
            except Exception as exc:
                last_error = exc
                ok = False

        if ok:
            os.replace(tmp, dst)
            return dst

        tmp.unlink(missing_ok=True)
        if attempt < retries:
            time.sleep(sleep * attempt)

    raise OSError(f"Failed to copy {src} to {dst} after {retries} attempts") from last_error


def concat_parquets_arrow(
    in_files: list[Path],
    out_path: Path,
    batch_size: int = 32_000,
    compression: str = "zstd",
    compression_level: int | None = 1,
    *,
    stage_dir: Path | None = None,
    stage_copy_retries: int = 3,
    stage_copy_sleep: float = 1.0,
    stage_validate: bool | Literal["none", "quick", "full"] = False,
) -> Path:
    """
    Stream-concatenate parquet files without loading everything into memory.
    """
    if not in_files:
        raise ValueError("in_files is empty; nothing to concatenate.")

    validate_mode: Literal["none", "quick", "full"]
    if stage_validate is True:
        validate_mode = "quick"
    elif stage_validate is False:
        validate_mode = "none"
    else:
        validate_mode = stage_validate

    writer: pq.ParquetWriter | None = None
    try:
        for f in in_files:
            read_path = f
            staged_path: Path | None = None
            try:
                if stage_dir is not None:
                    stage_dir.mkdir(parents=True, exist_ok=True)
                    staged_path = stage_dir / f.name
                    validate_copy: bool | Literal["quick", "full"] = False
                    if validate_mode != "none":
                        validate_copy = validate_mode
                    _copy_with_verify(
                        f,
                        staged_path,
                        retries=stage_copy_retries,
                        sleep=stage_copy_sleep,
                        validate=validate_copy,
                    )
                    read_path = staged_path

                attempts = 0
                while True:
                    attempts += 1
                    pf: pq.ParquetFile | None = None
                    try:
                        pf = pq.ParquetFile(read_path)
                    except Exception as exc:
                        raise OSError(f"Failed to open parquet file {f} (read from {read_path})") from exc

                    try:
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
                        break
                    except Exception as exc:
                        if stage_dir is not None and attempts == 1 and validate_mode != "full":
                            if staged_path is not None:
                                staged_path.unlink(missing_ok=True)
                            _copy_with_verify(
                                f,
                                staged_path or (stage_dir / f.name),
                                retries=stage_copy_retries,
                                sleep=stage_copy_sleep,
                                validate="full",
                            )
                            read_path = staged_path or (stage_dir / f.name)
                            continue
                        raise OSError(f"Failed while reading batches from {f} (read from {read_path})") from exc
                    finally:
                        if pf is not None:
                            try:
                                pf.close()
                            except Exception:
                                pass
            finally:
                if staged_path is not None:
                    staged_path.unlink(missing_ok=True)
    finally:
        if writer is not None:
            writer.close()

    return out_path

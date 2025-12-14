from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import polars as pl

# Make local `src` importable when running from repo checkout
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

from thesis_pkg.filing_text import (  # noqa: E402
    build_light_metadata_dataset,
    merge_yearly_batches,
    process_zip_year_raw_text,
    summarize_year_parquets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process SEC filing zip archives into parquet batches, merge yearly, and build a metadata-only file.",
    )
    parser.add_argument(
        "--zip-dir",
        type=Path,
        default=os.environ.get("SEC_FILINGS_ZIP_DIR"),
        help="Directory containing yearly .zip archives (can set SEC_FILINGS_ZIP_DIR env var).",
    )
    parser.add_argument(
        "--batch-dir",
        type=Path,
        help="Output directory for parquet batches (default: <zip-dir>/parquet_batches).",
    )
    parser.add_argument(
        "--merge-dir",
        type=Path,
        help="Output directory for merged yearly parquets (default: <zip-dir>/year_merged).",
    )
    parser.add_argument(
        "--light-path",
        type=Path,
        help="Path for metadata-only parquet (default: <zip-dir>/filings_metadata_LIGHT.parquet).",
    )
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        help="Temporary directory for staging zip copies (default: <batch-dir>/_tmp).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="JSON file to track merged years (default: <merge-dir>/done_years.json).",
    )
    parser.add_argument("--batch-max-rows", type=int, default=1000)
    parser.add_argument("--batch-max-text-bytes", type=int, default=250 * 1024 * 1024)
    parser.add_argument("--compression", type=str, default="zstd")
    parser.add_argument("--compression-level", type=int, default=1)
    parser.add_argument("--sleep-between-years", type=float, default=0.0)
    parser.add_argument(
        "--overwrite-batches",
        action="store_true",
        help="Rebuild batches even if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.zip_dir:
        raise SystemExit("zip-dir is required (arg or SEC_FILINGS_ZIP_DIR env var).")

    zip_dir = Path(args.zip_dir)
    if not zip_dir.exists():
        raise SystemExit(f"zip-dir not found: {zip_dir}")

    batch_dir = args.batch_dir or (zip_dir / "parquet_batches")
    merge_dir = args.merge_dir or (zip_dir / "year_merged")
    tmp_dir = args.tmp_dir or (batch_dir / "_tmp")
    light_path = args.light_path or (zip_dir / "filings_metadata_LIGHT.parquet")
    checkpoint = args.checkpoint or (merge_dir / "done_years.json")

    batch_dir.mkdir(parents=True, exist_ok=True)
    merge_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: process zip archives into parquet batches
    for zip_path in sorted(zip_dir.glob("*.zip")):
        existing = list(batch_dir.glob(f"{zip_path.stem}_batch_*.parquet"))
        if existing and not args.overwrite_batches:
            print(f"[skip] batches exist for {zip_path.name}")
            continue
        if existing:
            for p in existing:
                p.unlink()

        print(f"[batch] {zip_path.name}")
        process_zip_year_raw_text(
            zip_path=zip_path,
            out_dir=batch_dir,
            batch_max_rows=args.batch_max_rows,
            batch_max_text_bytes=args.batch_max_text_bytes,
            tmp_dir=tmp_dir,
            compression=args.compression,
        )

    # Step 2: merge yearly batches
    merged_paths = merge_yearly_batches(
        batch_dir=batch_dir,
        out_dir=merge_dir,
        checkpoint_path=checkpoint,
        batch_size=32_000,
        compression=args.compression,
        compression_level=args.compression_level,
        sleep_between_years=args.sleep_between_years,
    )
    print(f"[merge] merged {len(merged_paths)} yearly files into {merge_dir}")

    # Step 3: summarize merged files
    summary = summarize_year_parquets(merge_dir)
    print("[summary] year rows status")
    for item in summary:
        print(f"  {item['year']}: {item['rows']} rows ({item['status']})")

    # Step 4: build metadata-only file
    build_light_metadata_dataset(
        parquet_dir=merge_dir,
        out_path=light_path,
        drop_columns=("full_text",),
        sort_columns=("file_date_filename", "cik"),
        compression=args.compression,
    )
    total_rows = pl.scan_parquet(light_path).select(pl.len()).collect().item()
    print(f"[light] {total_rows} rows -> {light_path}")


if __name__ == "__main__":
    main()

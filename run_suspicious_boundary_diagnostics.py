from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make local `src` importable when running from repo checkout
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

from thesis_pkg.core.sec.suspicious_boundary_diagnostics import (  # noqa: E402
    DiagnosticsConfig,
    run_boundary_diagnostics,
)


DEFAULT_PARQUET_DIR = Path(
    r"C:\Users\erik9\Documents\SEC_Data\Data\Sample_Filings\parquet_batches"
)
DEFAULT_OUT_PATH = Path("results/suspicious_boundaries_v5.csv")
DEFAULT_REPORT_PATH = Path("results/suspicious_boundaries_report_v5.txt")
DEFAULT_SAMPLES_DIR = Path("results/Suspicious_Filings_Demo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run suspicious boundary diagnostics on extracted 10-K items."
    )
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=DEFAULT_PARQUET_DIR,
        help="Directory containing parquet batch files.",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=DEFAULT_OUT_PATH,
        help="CSV output path for suspicious boundary rows.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Text report output path.",
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=DEFAULT_SAMPLES_DIR,
        help="Directory for per-filing sample text files.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Parquet batch size for scanning filings.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional cap on number of parquet files to scan (0 = no cap).",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=25,
        help="Max examples to include in the report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DiagnosticsConfig(
        parquet_dir=args.parquet_dir,
        out_path=args.out_path,
        report_path=args.report_path,
        samples_dir=args.samples_dir,
        batch_size=args.batch_size,
        max_files=args.max_files,
        max_examples=args.max_examples,
    )
    run_boundary_diagnostics(config)


if __name__ == "__main__":
    main()

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
DEFAULT_OUT_PATH = Path("results/suspicious_boundaries_v9.csv")
DEFAULT_REPORT_PATH = Path("results/suspicious_boundaries_report_v9.txt")
DEFAULT_SAMPLES_DIR = Path("results/Suspicious_Filings_Demo")
DEFAULT_HTML_OUT_DIR = Path("results/html_audit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run suspicious boundary diagnostics with optional HTML audit output."
        )
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
    parser.add_argument(
        "--disable-embedded-verifier",
        action="store_true",
        help="Disable embedded heading verifier during scan.",
    )
    parser.add_argument(
        "--emit-manifest",
        action="store_true",
        help="Emit extraction manifest CSVs for items and filings.",
    )
    parser.add_argument(
        "--sample-pass",
        type=int,
        default=0,
        help="Sample N pass filings for CSV review outputs (requires manifests).",
    )
    parser.add_argument(
        "--emit-html",
        action="store_true",
        help="Emit offline HTML audit pages (requires manifests).",
    )
    parser.add_argument(
        "--html-out",
        type=Path,
        default=DEFAULT_HTML_OUT_DIR,
        help="Output directory for HTML audit pages.",
    )
    parser.add_argument(
        "--html-scope",
        type=str,
        default="sample",
        choices=("sample", "all"),
        help="HTML scope: mixed-status sample or all not-failed filings.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (CSV samples + HTML mixed-status sample).",
    )
    parser.add_argument(
        "--core-items",
        type=str,
        default="1,1A,7,7A,8",
        help="Comma-separated core item IDs for missing_core_items.",
    )
    return parser.parse_args()


def _parse_core_items(value: str | None) -> tuple[str, ...]:
    if not value:
        return ("1", "1A", "7", "7A", "8")
    parts = [part.strip().upper() for part in value.replace(";", ",").split(",")]
    cleaned = tuple(sorted({part for part in parts if part}))
    return cleaned or ("1", "1A", "7", "7A", "8")


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
        enable_embedded_verifier=not args.disable_embedded_verifier,
        emit_manifest=args.emit_manifest,
        sample_pass=args.sample_pass,
        sample_seed=args.seed,
        core_items=_parse_core_items(args.core_items),
        emit_html=args.emit_html,
        html_out=args.html_out,
        html_scope=args.html_scope,
    )
    run_boundary_diagnostics(config)


if __name__ == "__main__":
    main()

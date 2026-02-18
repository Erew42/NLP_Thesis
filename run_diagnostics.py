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
    parse_focus_items,
    run_boundary_diagnostics,
)
from thesis_pkg.core.sec.extraction import get_extraction_fastpath_metrics  # noqa: E402


DEFAULT_PARQUET_DIR = Path(
    r"C:\Users\erik9\Documents\SEC_Data\Data\Sample_Filings\parquet_batches"
)
DEFAULT_OUT_PATH = Path("results/suspicious_boundaries_v9.csv")
DEFAULT_REPORT_PATH = Path("results/suspicious_boundaries_report_v9.txt")
DEFAULT_SAMPLES_DIR = Path("results/Suspicious_Filings_Demo")
DEFAULT_HTML_OUT_DIR = Path("results/html_audit")
REGIME_CHOICES = ("legacy", "v2", "fast")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run suspicious boundary diagnostics on extracted 10-K/10-Q items with "
            "optional HTML audit output."
        )
    )
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=DEFAULT_PARQUET_DIR,
        help="Directory containing parquet input files (*_batch_*.parquet or YYYY.parquet).",
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
        "--no-manifest",
        action="store_false",
        dest="emit_manifest",
        help="Disable manifest CSV outputs.",
    )
    parser.add_argument(
        "--sample-pass",
        type=int,
        default=None,
        help="Sample N pass filings for CSV review outputs (requires manifests).",
    )
    parser.add_argument(
        "--no-pass-sample",
        action="store_const",
        const=0,
        dest="sample_pass",
        help="Disable pass-sample CSV outputs.",
    )
    parser.add_argument(
        "--emit-html",
        action="store_true",
        help="Emit offline HTML audit pages (requires manifests).",
    )
    parser.add_argument(
        "--no-html",
        action="store_false",
        dest="emit_html",
        help="Disable HTML audit outputs.",
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
    parser.add_argument(
        "--target-set",
        type=str,
        default=None,
        choices=("cohen2020_common", "cohen2020_all_items"),
        help=(
            "Restrict WARN/FAIL and missing-items to a target set "
            "(cohen2020_common or cohen2020_all_items)."
        ),
    )
    parser.add_argument(
        "--focus-items",
        type=str,
        default=None,
        help=(
            "Optional per-item focus set (e.g., "
            "\"10K:1A,7,7A,8,15;10Q:I:1,I:2,I:3,II:1,II:2\")."
        ),
    )
    parser.add_argument(
        "--report-item-scope",
        type=str,
        default=None,
        choices=("all", "target", "focus"),
        help=(
            "Scope for per-item breakdown sections: all items, target-set items, "
            "or focus-items only. Default is target if --target-set is provided, "
            "otherwise all."
        ),
    )
    parser.add_argument(
        "--html-min-total-chars",
        type=int,
        default=None,
        help="Minimum total extracted chars for filings included in HTML audit.",
    )
    parser.add_argument(
        "--html-min-largest-item-chars",
        type=int,
        default=None,
        help="Minimum largest-item length for filings included in HTML audit.",
    )
    parser.add_argument(
        "--html-min-largest-item-chars-pct-total",
        type=float,
        default=None,
        help="Minimum largest-item share of total chars for HTML audit.",
    )
    parser.add_argument(
        "--dump-missing-part-samples",
        type=int,
        default=0,
        help="Write a CSV of N sampled missing-part 10-Q items (0 = disabled).",
    )
    parser.add_argument(
        "--extraction-regime",
        type=str,
        default="legacy",
        choices=REGIME_CHOICES,
        help=(
            "Extraction regime to use: legacy, v2, or fast "
            "(fast = v2 with native fast-path enabled when available)."
        ),
    )
    parser.add_argument(
        "--diagnostics-regime",
        type=str,
        default="legacy",
        choices=REGIME_CHOICES,
        help=(
            "Diagnostics regime to use: legacy, v2, or fast "
            "(fast is treated as v2 for diagnostics logic)."
        ),
    )
    parser.set_defaults(
        emit_manifest=True,
        emit_html=True,
        sample_pass=100,
    )
    args = parser.parse_args()
    if not args.emit_manifest and args.emit_html:
        parser.error(
            "HTML output requires manifests. Remove --no-manifest or add --no-html."
        )
    if not args.emit_manifest and args.sample_pass and args.sample_pass > 0:
        parser.error(
            "Pass-sample outputs require manifests. Remove --no-manifest or set "
            "--sample-pass 0 / use --no-pass-sample."
        )
    if args.report_item_scope == "focus" and not args.focus_items:
        parser.error("--report-item-scope focus requires --focus-items.")
    if args.report_item_scope == "target" and not args.target_set:
        parser.error("--report-item-scope target requires --target-set.")
    return args


def _parse_core_items(value: str | None) -> tuple[str, ...]:
    if not value:
        return ("1", "1A", "7", "7A", "8")
    parts = [part.strip().upper() for part in value.replace(";", ",").split(",")]
    cleaned = tuple(sorted({part for part in parts if part}))
    return cleaned or ("1", "1A", "7", "7A", "8")


def _normalize_regime(value: str) -> str:
    normalized = (value or "legacy").strip().lower()
    if normalized == "fast":
        return "v2"
    if normalized == "v2":
        return "v2"
    return "legacy"


def main() -> None:
    args = parse_args()
    extraction_regime = _normalize_regime(args.extraction_regime)
    diagnostics_regime = _normalize_regime(args.diagnostics_regime)
    if args.extraction_regime == "fast":
        fastpath = get_extraction_fastpath_metrics()
        if not bool(fastpath.get("fastpath_extension_available")):
            print(
                "Warning: --extraction-regime fast requested, but native fast-path "
                "extension is unavailable; running v2 with Python fallback."
            )
    if args.report_item_scope is None:
        report_item_scope = "target" if args.target_set else "all"
    else:
        report_item_scope = args.report_item_scope
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
        target_set=args.target_set,
        focus_items=parse_focus_items(args.focus_items),
        report_item_scope=report_item_scope,
        emit_html=args.emit_html,
        html_out=args.html_out,
        html_scope=args.html_scope,
        html_min_total_chars=args.html_min_total_chars,
        html_min_largest_item_chars=args.html_min_largest_item_chars,
        html_min_largest_item_chars_pct_total=args.html_min_largest_item_chars_pct_total,
        dump_missing_part_samples=args.dump_missing_part_samples,
        extraction_regime=extraction_regime,
        diagnostics_regime=diagnostics_regime,
    )
    run_boundary_diagnostics(config)


if __name__ == "__main__":
    main()

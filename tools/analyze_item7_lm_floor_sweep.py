from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


def _resolve_repo_root() -> Path:
    candidates = [Path.cwd().resolve(), *Path.cwd().resolve().parents, *Path(__file__).resolve().parents]
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "src" / "thesis_pkg").exists():
            return candidate
    raise RuntimeError("Could not resolve repository root containing src/thesis_pkg.")


ROOT = _resolve_repo_root()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from thesis_pkg.benchmarking.contracts import ItemTextCleaningConfig
from thesis_pkg.benchmarking.item7_lm_floor_sweep import DEFAULT_ITEM7_FLOOR_THRESHOLDS
from thesis_pkg.benchmarking.item7_lm_floor_sweep import write_item7_lm_floor_sweep_report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rerun the same sampled item rows over an Item 7 LM-token floor grid and report how "
            "dropped-row counts and confirmed false-positive ratios change."
        )
    )
    parser.add_argument(
        "--sampled-sections-path",
        type=Path,
        required=True,
        help="Parquet/csv from sample_item_cleaning_sentence_diagnostics containing the fixed sampled item rows.",
    )
    parser.add_argument(
        "--review-cases-path",
        type=Path,
        default=None,
        help="Optional reviewed case file from the larger review pack to tag confirmed false-positive removals.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "tmp" / Path(__file__).stem,
        help="Directory for the threshold sweep tables and figures.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="*",
        type=int,
        default=list(DEFAULT_ITEM7_FLOOR_THRESHOLDS),
        help="Grid of Item 7 minimum LM-token thresholds to evaluate. Use 0 to disable the floor.",
    )
    parser.add_argument(
        "--confirmed-fp-benchmark-row-id",
        nargs="*",
        default=None,
        help="Optional confirmed false-positive benchmark_row_id values to force into the ratio calculation.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    artifacts = write_item7_lm_floor_sweep_report(
        args.sampled_sections_path.resolve(),
        args.output_dir.resolve(),
        thresholds=tuple(args.thresholds),
        base_cleaning_cfg=ItemTextCleaningConfig(),
        review_cases_path=args.review_cases_path.resolve() if args.review_cases_path is not None else None,
        confirmed_false_positive_ids=set(args.confirmed_fp_benchmark_row_id or []),
    )
    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    payload = {
        "output_dir": str(artifacts.output_dir),
        "results_path": str(artifacts.results_path),
        "reviewed_case_status_path": (
            str(artifacts.reviewed_case_status_path) if artifacts.reviewed_case_status_path is not None else None
        ),
        "summary_path": str(artifacts.summary_path),
        "thresholds": summary["thresholds"],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

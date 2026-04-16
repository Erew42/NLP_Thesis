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

from thesis_pkg.benchmarking.sentence_split_quality_assessment import analyze_sentence_split_quality
from thesis_pkg.benchmarking.sentence_split_quality_assessment import write_sentence_split_quality_report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assess residual sentence-splitting quality on a FinBERT sentence dataset and write "
            "stable summary tables, example CSVs, and a concise markdown report under tmp/."
        )
    )
    parser.add_argument(
        "--sentence-dataset-dir",
        type=Path,
        required=True,
        help="Path to the sentence dataset root or its by_year directory.",
    )
    parser.add_argument(
        "--cleaning-row-audit-path",
        type=Path,
        default=None,
        help="Optional cleaning_row_audit.parquet path for kept-item context in the report.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "tmp" / Path(__file__).stem,
        help="Directory for summary tables, examples, and the markdown report.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    analysis = analyze_sentence_split_quality(
        args.sentence_dataset_dir.resolve(),
        cleaning_row_audit_path=(
            args.cleaning_row_audit_path.resolve() if args.cleaning_row_audit_path is not None else None
        ),
    )
    artifacts = write_sentence_split_quality_report(analysis, args.output_dir.resolve())
    payload = {
        "output_dir": str(artifacts["output_dir"]),
        "report_path": str(artifacts["report_path"]),
        "summary_by_scope_path": str(artifacts["summary_by_scope_path"]),
        "split_audit_summary_path": str(artifacts["split_audit_summary_path"]),
        "summary": analysis.metadata,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

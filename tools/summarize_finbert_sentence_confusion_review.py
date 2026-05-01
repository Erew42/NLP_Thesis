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

from thesis_pkg.benchmarking.finbert_sentence_confusion_review import (  # noqa: E402
    summarize_finbert_sentence_confusion_review,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize a completed FinBERT negative/adverse sentence review into confusion "
            "matrices, weighted metrics, bucket summaries, and example rows."
        )
    )
    parser.add_argument(
        "--review-dir",
        type=Path,
        required=True,
        help="Review-pack directory containing sample.parquet.",
    )
    parser.add_argument(
        "--human-review-path",
        type=Path,
        required=True,
        help="JSON or JSONL review export from review.html.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to <review-dir>/review_summary.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    artifacts = summarize_finbert_sentence_confusion_review(
        args.review_dir,
        human_review_path=args.human_review_path,
        output_dir=args.output_dir,
    )
    payload = {
        "output_dir": str(artifacts.output_dir),
        "reviewed_cases_path": str(artifacts.reviewed_cases_path),
        "reviewed_cases_csv_path": str(artifacts.reviewed_cases_csv_path),
        "confusion_matrix_path": str(artifacts.confusion_matrix_path),
        "metrics_json_path": str(artifacts.metrics_json_path),
        "metrics_markdown_path": str(artifacts.metrics_markdown_path),
        "majority_bucket_metrics_path": str(artifacts.majority_bucket_metrics_path),
        "examples_by_cell_path": str(artifacts.examples_by_cell_path),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

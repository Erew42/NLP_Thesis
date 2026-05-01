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
    ALLOCATION_MODE_VALUES,
)
from thesis_pkg.benchmarking.finbert_sentence_confusion_review import (  # noqa: E402
    FinbertSentenceConfusionReviewConfig,
)
from thesis_pkg.benchmarking.finbert_sentence_confusion_review import (  # noqa: E402
    THESIS_REVIEW_TEXT_SCOPES,
)
from thesis_pkg.benchmarking.finbert_sentence_confusion_review import (  # noqa: E402
    build_finbert_sentence_confusion_review_pack,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a reproducible FinBERT negative/adverse sentence confusion review pack. "
            "The builder first writes population counts, then optionally draws an OOM-safe "
            "stratified sample and static HTML review page."
        )
    )
    parser.add_argument(
        "--sentence-scores-dir",
        type=Path,
        required=True,
        help="Path to sentence_scores or sentence_scores/by_year.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for counts, sample, chunks, manifest, and review.html.",
    )
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--text-scopes",
        nargs="+",
        default=list(THESIS_REVIEW_TEXT_SCOPES),
        help="Text scopes to include in the review universe.",
    )
    parser.add_argument("--chunk-count", type=int, default=10)
    parser.add_argument(
        "--allocation-mode",
        choices=ALLOCATION_MODE_VALUES,
        default="balanced",
        help="How to allocate sample rows across eligible strata.",
    )
    parser.add_argument("--counts-only", action="store_true")
    parser.add_argument(
        "--include-no-majority",
        action="store_true",
        help="Include rows where no FinBERT class probability exceeds 0.5 in the sampling universe.",
    )
    parser.add_argument(
        "--stream-batch-size",
        type=int,
        default=25_000,
        help="Maximum parquet rows materialized at once during sample/context streaming.",
    )
    parser.add_argument("--initial-oversampling-factor", type=float, default=8.0)
    parser.add_argument("--max-oversampling-factor", type=float, default=256.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = FinbertSentenceConfusionReviewConfig(
        sample_size=args.sample_size,
        seed=args.seed,
        text_scopes=tuple(args.text_scopes),
        chunk_count=args.chunk_count,
        include_no_majority=args.include_no_majority,
        allocation_mode=args.allocation_mode,
        stream_batch_size=args.stream_batch_size,
        initial_oversampling_factor=args.initial_oversampling_factor,
        max_oversampling_factor=args.max_oversampling_factor,
    )
    artifacts = build_finbert_sentence_confusion_review_pack(
        args.sentence_scores_dir,
        output_dir=args.output_dir,
        cfg=cfg,
        counts_only=args.counts_only,
    )
    payload = {
        "output_dir": str(artifacts.output_dir),
        "manifest_path": str(artifacts.manifest_path),
        "population_counts_by_majority_bucket_path": str(
            artifacts.population_counts_by_majority_bucket_path
        ),
        "population_counts_by_stratum_path": str(artifacts.population_counts_by_stratum_path),
        "population_counts_summary_path": str(artifacts.population_counts_summary_path),
        "sample_path": str(artifacts.sample_path) if artifacts.sample_path is not None else None,
        "sample_csv_path": str(artifacts.sample_csv_path)
        if artifacts.sample_csv_path is not None
        else None,
        "review_html_path": str(artifacts.review_html_path)
        if artifacts.review_html_path is not None
        else None,
        "labeling_prompt_path": str(artifacts.labeling_prompt_path)
        if artifacts.labeling_prompt_path is not None
        else None,
        "chunk_dir": str(artifacts.chunk_dir) if artifacts.chunk_dir is not None else None,
        "sample_row_count": artifacts.sample_row_count,
        "counts_only": artifacts.counts_only,
        "oversampling_factor": artifacts.oversampling_factor,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

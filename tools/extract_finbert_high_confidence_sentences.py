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

from thesis_pkg.benchmarking.finbert_sentence_examples import (
    build_high_confidence_sentence_example_pack,
)


def _discover_sentence_scores_dir(root: Path) -> Path:
    candidates: list[Path] = []
    for search_root in (root / "full_data_run", root / "results"):
        if not search_root.exists():
            continue
        for sentence_scores_dir in search_root.rglob("sentence_scores"):
            by_year_dir = sentence_scores_dir / "by_year"
            if by_year_dir.is_dir() and any(by_year_dir.glob("*.parquet")):
                candidates.append(sentence_scores_dir)
    if not candidates:
        raise FileNotFoundError(
            "Could not auto-discover a sentence_scores directory. Pass --sentence-scores-dir explicitly."
        )
    return max(
        candidates,
        key=lambda path: max(candidate.stat().st_mtime for candidate in (path / "by_year").glob("*.parquet")),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract thesis-friendly high-confidence FinBERT sentence examples from sentence_scores. "
            "Defaults to Item 1A and Item 7, keeps only >=95% positive/negative sentences with at least 6 words, "
            "uses bounded batch processing, and emits a smaller sample list."
        )
    )
    parser.add_argument(
        "--sentence-scores-dir",
        type=Path,
        default=None,
        help="Path to sentence_scores or sentence_scores/by_year. Defaults to the latest discovered run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "tmp" / Path(__file__).stem,
        help="Directory for filtered candidates, summaries, and the Markdown sample list.",
    )
    parser.add_argument(
        "--item-codes",
        nargs="+",
        default=["item_1a", "item_7"],
        help="Benchmark item codes to keep.",
    )
    parser.add_argument(
        "--min-probability",
        type=float,
        default=0.95,
        help="Minimum positive or negative FinBERT probability required to keep a sentence.",
    )
    parser.add_argument(
        "--min-word-count",
        type=int,
        default=6,
        help="Minimum whitespace-delimited word count required to keep a sentence.",
    )
    parser.add_argument(
        "--sample-size-per-group",
        type=int,
        default=50,
        help="Number of sample sentences to keep for each item x sentiment bucket.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic sampling seed for the shortlist.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50_000,
        help="Maximum number of source rows to process in memory at a time.",
    )
    parser.add_argument(
        "--write-candidate-shards",
        action="store_true",
        help="Also write all qualifying candidates back out as yearly parquet shards under the output directory.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    sentence_scores_dir = (
        args.sentence_scores_dir.resolve()
        if args.sentence_scores_dir is not None
        else _discover_sentence_scores_dir(ROOT)
    )
    pack = build_high_confidence_sentence_example_pack(
        sentence_scores_dir,
        output_dir=args.output_dir.resolve(),
        item_codes=tuple(args.item_codes) if args.item_codes else None,
        min_probability=args.min_probability,
        min_word_count=args.min_word_count,
        sample_size_per_group=args.sample_size_per_group,
        seed=args.seed,
        batch_size=args.batch_size,
        write_candidate_shards=args.write_candidate_shards,
    )
    payload = {
        "sentence_scores_dir": str(sentence_scores_dir),
        "output_dir": str(pack.artifacts.output_dir),
        "data_dir": str(pack.artifacts.data_dir),
        "candidate_shards_dir": (
            str(pack.artifacts.candidate_shards_dir)
            if pack.artifacts.candidate_shards_dir is not None
            else None
        ),
        "sample_markdown_path": str(pack.artifacts.sample_markdown_path),
        "summary_json_path": str(pack.artifacts.summary_json_path),
        "filters": pack.metadata["filters"],
        "candidate_rows": pack.metadata["candidate_rows"],
        "candidate_doc_count": pack.metadata["candidate_doc_count"],
        "counts_by_item_sentiment": pack.counts_by_item_sentiment.to_dicts(),
        "sample_counts_by_item_sentiment": (
            pack.sample_candidates.group_by(["benchmark_item_code", "sentiment"])
            .len()
            .rename({"len": "sample_rows"})
            .sort(["benchmark_item_code", "sentiment"])
            .to_dicts()
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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

from thesis_pkg.benchmarking.sentence_length_visualization import analyze_sentence_lengths
from thesis_pkg.benchmarking.sentence_length_visualization import write_sentence_length_report


def _discover_sentence_dataset_dir(root: Path) -> Path:
    candidates: list[Path] = []
    for search_root in (root / "full_data_run", root / "results"):
        if not search_root.exists():
            continue
        for sentence_dataset_dir in search_root.rglob("sentence_dataset"):
            by_year_dir = sentence_dataset_dir / "by_year"
            if by_year_dir.is_dir() and any(by_year_dir.glob("*.parquet")):
                candidates.append(by_year_dir)
    if not candidates:
        raise FileNotFoundError(
            "Could not auto-discover a sentence dataset. Pass --sentence-dataset-dir explicitly."
        )
    return max(
        candidates,
        key=lambda path: max(candidate.stat().st_mtime for candidate in path.glob("*.parquet")),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze FinBERT sentence-length distributions and save summary tables plus plots. "
            "Defaults to Item 1A and Item 7."
        )
    )
    parser.add_argument(
        "--sentence-dataset-dir",
        type=Path,
        default=None,
        help="Path to sentence_dataset or sentence_dataset/by_year. Defaults to the latest discovered dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "tmp" / Path(__file__).stem,
        help="Directory for plots and summary tables.",
    )
    parser.add_argument(
        "--item-codes",
        nargs="+",
        default=["item_1a", "item_7"],
        help="Benchmark item codes to include.",
    )
    parser.add_argument(
        "--years",
        nargs="*",
        type=int,
        default=None,
        help="Optional filing years to include.",
    )
    parser.add_argument(
        "--char-bin-width",
        type=int,
        default=25,
        help="Histogram bin width for sentence character counts.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=25,
        help="Number of longest sentences to persist for inspection.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    sentence_dataset_dir = (
        args.sentence_dataset_dir.resolve()
        if args.sentence_dataset_dir is not None
        else _discover_sentence_dataset_dir(ROOT)
    )
    analysis = analyze_sentence_lengths(
        sentence_dataset_dir,
        item_codes=tuple(args.item_codes) if args.item_codes else None,
        years=tuple(args.years) if args.years else None,
        char_bin_width=args.char_bin_width,
        top_n=args.top_n,
    )
    artifacts = write_sentence_length_report(analysis, args.output_dir)
    payload = {
        "sentence_dataset_dir": str(sentence_dataset_dir),
        "output_dir": str(Path(artifacts["output_dir"]).resolve()),
        "figures_dir": str(Path(artifacts["figures_dir"]).resolve()),
        "data_dir": str(Path(artifacts["data_dir"]).resolve()),
        "metadata": analysis.metadata,
        "summary_overall": analysis.summary_overall.to_dicts(),
        "summary_by_item": analysis.summary_by_item.to_dicts(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

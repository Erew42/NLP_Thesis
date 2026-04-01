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

from thesis_pkg.benchmarking import BenchmarkSampleSpec
from thesis_pkg.benchmarking import FinbertBenchmarkSuiteConfig
from thesis_pkg.benchmarking import SentenceDatasetConfig
from thesis_pkg.benchmarking import build_finbert_benchmark_suite


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic FinBERT 10-K benchmark datasets.")
    parser.add_argument("--source-items-dir", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compression", type=str, default="zstd")
    parser.add_argument(
        "--enable-sentences",
        action="store_true",
        help="Materialize an additional derived sentence-level artifact.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = FinbertBenchmarkSuiteConfig(
        source_items_dir=args.source_items_dir.resolve(),
        out_root=args.out_root.resolve(),
        sample_specs=(
            BenchmarkSampleSpec(sample_name="1pct", sample_fraction=0.01),
            BenchmarkSampleSpec(sample_name="5pct", sample_fraction=0.05),
        ),
        seed=args.seed,
        compression=args.compression,
        sentence_dataset=SentenceDatasetConfig(enabled=args.enable_sentences),
    )
    artifacts = build_finbert_benchmark_suite(cfg)
    payload = {
        sample_name: {
            "dataset_tag": artifact.dataset_tag,
            "sections_path": str(artifact.sections_path.resolve()),
            "sentences_path": str(artifact.sentences_path.resolve()) if artifact.sentences_path else None,
            "manifest_path": str(artifact.manifest_path.resolve()),
            "selected_row_count": artifact.selected_row_count,
            "selected_doc_count": artifact.selected_doc_count,
        }
        for sample_name, artifact in artifacts.items()
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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

from thesis_pkg.benchmarking.multisurface_audit import summarize_reviewed_audit_pack


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate reviewed multi-surface FinBERT audit chunks and generate synthesis artifacts "
            "such as pattern summaries, document hotspots, and rule candidate tables."
        )
    )
    parser.add_argument(
        "--audit-pack-dir",
        type=Path,
        required=True,
        help="Path to the built audit-pack directory containing manifest.json.",
    )
    parser.add_argument(
        "--reviewed-chunks-dir",
        type=Path,
        default=None,
        help="Optional override for the directory containing completed reviewed chunk CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory for the synthesis artifacts. Defaults to <audit-pack-dir>/review_summary.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    artifacts = summarize_reviewed_audit_pack(
        args.audit_pack_dir,
        reviewed_chunks_dir=args.reviewed_chunks_dir,
        output_dir=args.output_dir,
    )
    payload = {
        "output_dir": str(artifacts.output_dir),
        "reviewed_cases_path": str(artifacts.reviewed_cases_path),
        "review_summary_path": str(artifacts.review_summary_path),
        "pattern_summary_path": str(artifacts.pattern_summary_path),
        "doc_hotspots_path": str(artifacts.doc_hotspots_path),
        "rule_candidates_path": str(artifacts.rule_candidates_path),
        "do_not_touch_patterns_path": str(artifacts.do_not_touch_patterns_path),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

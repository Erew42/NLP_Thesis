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

from thesis_pkg.benchmarking.multisurface_audit import MultiSurfaceAuditPackConfig
from thesis_pkg.benchmarking.multisurface_audit import build_multisurface_audit_pack
from thesis_pkg.benchmarking.multisurface_audit import resolve_multisurface_audit_sources


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a multi-surface FinBERT audit pack that joins sentence cases, cleaned item cases, "
            "and escalated full 10-K snippets from SEC year_merged parquet files."
        )
    )
    parser.add_argument(
        "--run-manifest-path",
        type=Path,
        required=True,
        help="Path to the staged FinBERT sentence preprocessing run_manifest.json.",
    )
    parser.add_argument(
        "--sec-year-merged-dir",
        type=Path,
        default=None,
        help="Optional override for the SEC sample year_merged parquet directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "tmp" / Path(__file__).stem,
        help="Output directory for the audit pack.",
    )
    parser.add_argument("--sentence-short-target", type=int, default=60)
    parser.add_argument("--sentence-long-target", type=int, default=60)
    parser.add_argument("--sentence-control-target", type=int, default=60)
    parser.add_argument("--item-cleaning-target", type=int, default=60)
    parser.add_argument("--item-hotspot-target", type=int, default=60)
    parser.add_argument("--item-boundary-target", type=int, default=60)
    parser.add_argument("--escalation-cap", type=int, default=120)
    parser.add_argument(
        "--chunk-count",
        type=int,
        default=18,
        help="Number of reviewer chunk CSVs to emit. Recommended range is 18-20.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--full-report-window-chars", type=int, default=1500)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not 18 <= args.chunk_count <= 20:
        raise ValueError("--chunk-count must stay in the 18-20 range for production review packs.")

    cfg = MultiSurfaceAuditPackConfig(
        sentence_short_case_target=args.sentence_short_target,
        sentence_long_case_target=args.sentence_long_target,
        sentence_control_case_target=args.sentence_control_target,
        item_cleaning_case_target=args.item_cleaning_target,
        item_hotspot_case_target=args.item_hotspot_target,
        item_boundary_case_target=args.item_boundary_target,
        escalation_cap=args.escalation_cap,
        chunk_count=args.chunk_count,
        random_seed=args.random_seed,
        full_report_window_chars=args.full_report_window_chars,
    )
    sources = resolve_multisurface_audit_sources(
        args.run_manifest_path,
        sec_year_merged_dir=args.sec_year_merged_dir,
    )
    artifacts = build_multisurface_audit_pack(
        sources,
        output_dir=args.output_dir.resolve(),
        cfg=cfg,
    )
    payload = {
        "output_dir": str(artifacts.output_dir),
        "audit_cases_path": str(artifacts.audit_cases_path),
        "chunk_dir": str(artifacts.chunk_dir),
        "summary_path": str(artifacts.summary_path),
        "manifest_path": str(artifacts.manifest_path),
        "review_instructions_path": str(artifacts.review_instructions_path),
        "primary_case_count": artifacts.primary_case_count,
        "escalated_case_count": artifacts.escalated_case_count,
        "requested_full_report_doc_count": artifacts.requested_full_report_doc_count,
        "fetched_full_report_doc_count": artifacts.fetched_full_report_doc_count,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

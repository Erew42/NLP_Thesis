from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT_ENV_VAR = "NLP_THESIS_REPO_ROOT"


def _resolve_repo_root() -> Path:
    candidates: list[Path] = []
    env_root = os.environ.get(REPO_ROOT_ENV_VAR)
    if env_root:
        candidates.append(Path(env_root).expanduser())

    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])

    script_path = Path(__file__).resolve()
    candidates.extend(script_path.parents)

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "src" / "thesis_pkg" / "pipeline.py").exists():
            return candidate
    raise RuntimeError("Could not resolve repository root containing src/thesis_pkg/pipeline.py")


ROOT = _resolve_repo_root()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from thesis_pkg.pipelines.lm2011_validation_audit import PACKET_CHOICES
from thesis_pkg.pipelines.lm2011_validation_audit import Phase0ValidationAuditConfig
from thesis_pkg.pipelines.lm2011_validation_audit import run_phase0_validation_audit


DEFAULT_SAMPLE_ROOT = ROOT / "full_data_run" / "sample_5pct_seed42"
DEFAULT_UPSTREAM_RUN_ROOT = DEFAULT_SAMPLE_ROOT / "results" / "sec_ccm_unified_runner" / "local_sample"
DEFAULT_LM2011_OUTPUT_DIR = DEFAULT_SAMPLE_ROOT / "results" / "lm2011_sample_post_refinitiv_runner"
DEFAULT_FINBERT_OUTPUT_ROOT = DEFAULT_SAMPLE_ROOT / "results" / "finbert_item_analysis_runner"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / f"lm2011_phase0_validation_sample_{dt.date.today():%Y%m%d}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the sample-first LM2011 Phase 0 validation audit against existing artifacts."
    )
    parser.add_argument("--sample-root", type=Path, default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument("--upstream-run-root", type=Path, default=DEFAULT_UPSTREAM_RUN_ROOT)
    parser.add_argument("--lm2011-output-dir", type=Path, default=DEFAULT_LM2011_OUTPUT_DIR)
    parser.add_argument("--finbert-output-root", type=Path, default=DEFAULT_FINBERT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--packets",
        nargs="+",
        default=list(PACKET_CHOICES),
        help=f"Packets to run. Choices: {', '.join(PACKET_CHOICES)}",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=None,
        help="Optional filing years to audit.",
    )
    parser.add_argument("--finbert-run-dir", type=Path, default=None)
    parser.add_argument("--max-example-rows", type=int, default=25)
    parser.add_argument("--snippet-char-limit", type=int, default=400)
    parser.add_argument("--regime-compare-doc-limit", type=int, default=100)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = Phase0ValidationAuditConfig(
        sample_root=args.sample_root,
        upstream_run_root=args.upstream_run_root,
        lm2011_output_dir=args.lm2011_output_dir,
        finbert_output_root=args.finbert_output_root,
        output_root=args.output_dir,
        packets=tuple(args.packets),
        year_filter=tuple(args.years) if args.years is not None else None,
        finbert_run_dir=args.finbert_run_dir,
        max_example_rows=args.max_example_rows,
        snippet_char_limit=args.snippet_char_limit,
        regime_compare_doc_limit=args.regime_compare_doc_limit,
    )
    artifacts = run_phase0_validation_audit(cfg)
    print(f"report_path={artifacts.report_path}")
    print(f"manifest_path={artifacts.manifest_path}")
    for packet in sorted(artifacts.packet_statuses):
        print(f"packet_{packet}={artifacts.packet_statuses[packet]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

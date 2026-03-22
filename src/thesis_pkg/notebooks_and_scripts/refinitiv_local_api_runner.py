from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Sequence


SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import polars as pl

from thesis_pkg.pipelines.refinitiv import (
    is_lseg_available,
    run_refinitiv_lm2011_doc_ownership_exact_api_pipeline,
    run_refinitiv_lm2011_doc_ownership_fallback_api_pipeline,
    run_refinitiv_lm2011_doc_ownership_finalize_pipeline,
    run_refinitiv_step1_lookup_api_pipeline,
    run_refinitiv_step1_ownership_authority_pipeline,
    run_refinitiv_step1_ownership_universe_api_pipeline,
    run_refinitiv_step1_ownership_universe_handoff_pipeline,
    run_refinitiv_step1_resolution_pipeline,
)


STAGE_ORDER: tuple[str, ...] = (
    "lookup_api",
    "resolution",
    "ownership_handoff",
    "ownership_api",
    "authority",
    "doc_exact_api",
    "doc_fallback_api",
    "doc_finalize",
)


@dataclass(frozen=True)
class RunPaths:
    run_root: Path
    sec_ccm_premerge_dir: Path
    refinitiv_step1_dir: Path
    ownership_universe_dir: Path
    ownership_authority_dir: Path
    doc_ownership_dir: Path
    manifest_path: Path
    doc_filing_artifact_path: Path
    lookup_snapshot_parquet: Path
    lookup_extended_parquet: Path
    resolution_parquet: Path
    ownership_handoff_parquet: Path
    ownership_results_parquet: Path
    ownership_row_summary_parquet: Path
    authority_decisions_parquet: Path
    authority_exceptions_parquet: Path
    reviewed_ticker_allowlist_path: Path
    exact_requests_parquet: Path
    exact_raw_parquet: Path
    fallback_requests_parquet: Path
    fallback_raw_parquet: Path
    doc_ownership_raw_parquet: Path
    doc_ownership_final_parquet: Path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the local API-backed Refinitiv workflow against a copied "
            "sec_ccm_unified_runner run-root."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Copied local run-root matching the sec_ccm_unified_runner output layout.",
    )
    parser.add_argument(
        "--stage-start",
        choices=STAGE_ORDER,
        default=STAGE_ORDER[0],
        help="First stage to execute.",
    )
    parser.add_argument(
        "--stage-stop",
        choices=STAGE_ORDER,
        default=STAGE_ORDER[-1],
        help="Last stage to execute.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reuse existing stage ledgers and staging artifacts. "
            "Use --no-resume to fail fast when selected stage outputs or API state already exist."
        ),
    )
    parser.add_argument("--lookup-batch-size", type=int, default=25)
    parser.add_argument("--ownership-batch-size", type=int, default=10)
    parser.add_argument("--doc-exact-batch-size", type=int, default=15)
    parser.add_argument("--doc-fallback-batch-size", type=int, default=5)
    parser.add_argument("--min-seconds-between-requests", type=float, default=2.0)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument(
        "--reviewed-ticker-allowlist-path",
        type=Path,
        default=None,
        help="Optional override for the reviewed ticker allowlist parquet used by the authority stage.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if STAGE_ORDER.index(args.stage_start) > STAGE_ORDER.index(args.stage_stop):
        parser.error("--stage-start must be earlier than or equal to --stage-stop")
    args.run_root = args.run_root.expanduser().resolve()
    if args.reviewed_ticker_allowlist_path is not None:
        args.reviewed_ticker_allowlist_path = args.reviewed_ticker_allowlist_path.expanduser().resolve()
    return args


def _resolve_run_paths(
    run_root: Path,
    *,
    reviewed_ticker_allowlist_path: Path | None,
) -> RunPaths:
    sec_ccm_premerge_dir = run_root / "sec_ccm_premerge"
    refinitiv_step1_dir = run_root / "refinitiv_step1"
    ownership_universe_dir = refinitiv_step1_dir / "ownership_universe_common_stock"
    ownership_authority_dir = refinitiv_step1_dir / "ownership_authority_common_stock"
    doc_ownership_dir = run_root / "refinitiv_doc_ownership_lm2011"
    return RunPaths(
        run_root=run_root,
        sec_ccm_premerge_dir=sec_ccm_premerge_dir,
        refinitiv_step1_dir=refinitiv_step1_dir,
        ownership_universe_dir=ownership_universe_dir,
        ownership_authority_dir=ownership_authority_dir,
        doc_ownership_dir=doc_ownership_dir,
        manifest_path=run_root / "refinitiv_local_api_runner_manifest.json",
        doc_filing_artifact_path=sec_ccm_premerge_dir / "sec_ccm_matched_clean_filtered.parquet",
        lookup_snapshot_parquet=refinitiv_step1_dir / "refinitiv_ric_lookup_handoff_common_stock_extended_snapshot.parquet",
        lookup_extended_parquet=refinitiv_step1_dir / "refinitiv_ric_lookup_handoff_common_stock_extended.parquet",
        resolution_parquet=refinitiv_step1_dir / "refinitiv_ric_resolution_common_stock.parquet",
        ownership_handoff_parquet=ownership_universe_dir / "refinitiv_ownership_universe_handoff_common_stock.parquet",
        ownership_results_parquet=ownership_universe_dir / "refinitiv_ownership_universe_results.parquet",
        ownership_row_summary_parquet=ownership_universe_dir / "refinitiv_ownership_universe_row_summary.parquet",
        authority_decisions_parquet=ownership_authority_dir / "refinitiv_permno_ownership_authority_decisions.parquet",
        authority_exceptions_parquet=ownership_authority_dir / "refinitiv_permno_ownership_authority_exceptions.parquet",
        reviewed_ticker_allowlist_path=(
            reviewed_ticker_allowlist_path
            if reviewed_ticker_allowlist_path is not None
            else ownership_authority_dir / "refinitiv_permno_ownership_ticker_allowlist.parquet"
        ),
        exact_requests_parquet=doc_ownership_dir / "refinitiv_lm2011_doc_ownership_exact_requests.parquet",
        exact_raw_parquet=doc_ownership_dir / "refinitiv_lm2011_doc_ownership_exact_raw.parquet",
        fallback_requests_parquet=doc_ownership_dir / "refinitiv_lm2011_doc_ownership_fallback_requests.parquet",
        fallback_raw_parquet=doc_ownership_dir / "refinitiv_lm2011_doc_ownership_fallback_raw.parquet",
        doc_ownership_raw_parquet=doc_ownership_dir / "refinitiv_lm2011_doc_ownership_raw.parquet",
        doc_ownership_final_parquet=doc_ownership_dir / "refinitiv_lm2011_doc_ownership.parquet",
    )


def _selected_stages(stage_start: str, stage_stop: str) -> tuple[str, ...]:
    start_idx = STAGE_ORDER.index(stage_start)
    stop_idx = STAGE_ORDER.index(stage_stop)
    return STAGE_ORDER[start_idx : stop_idx + 1]


def _stage_input_paths(paths: RunPaths, stage: str) -> dict[str, Path]:
    if stage == "lookup_api":
        return {"lookup_snapshot_parquet": paths.lookup_snapshot_parquet}
    if stage == "resolution":
        return {"lookup_extended_parquet": paths.lookup_extended_parquet}
    if stage == "ownership_handoff":
        return {"resolution_parquet": paths.resolution_parquet}
    if stage == "ownership_api":
        return {"ownership_handoff_parquet": paths.ownership_handoff_parquet}
    if stage == "authority":
        return {
            "resolution_parquet": paths.resolution_parquet,
            "ownership_results_parquet": paths.ownership_results_parquet,
            "ownership_row_summary_parquet": paths.ownership_row_summary_parquet,
        }
    if stage == "doc_exact_api":
        return {
            "doc_filing_artifact_parquet": paths.doc_filing_artifact_path,
            "authority_decisions_parquet": paths.authority_decisions_parquet,
            "authority_exceptions_parquet": paths.authority_exceptions_parquet,
        }
    if stage == "doc_fallback_api":
        return {
            "exact_requests_parquet": paths.exact_requests_parquet,
            "exact_raw_parquet": paths.exact_raw_parquet,
        }
    if stage == "doc_finalize":
        return {
            "exact_requests_parquet": paths.exact_requests_parquet,
            "exact_raw_parquet": paths.exact_raw_parquet,
            "fallback_requests_parquet": paths.fallback_requests_parquet,
        }
    raise ValueError(f"Unsupported stage: {stage}")


def _stage_output_paths(paths: RunPaths, stage: str) -> dict[str, Path]:
    if stage == "lookup_api":
        return {"lookup_extended_parquet": paths.lookup_extended_parquet}
    if stage == "resolution":
        return {"resolution_parquet": paths.resolution_parquet}
    if stage == "ownership_handoff":
        return {"ownership_handoff_parquet": paths.ownership_handoff_parquet}
    if stage == "ownership_api":
        return {
            "ownership_results_parquet": paths.ownership_results_parquet,
            "ownership_row_summary_parquet": paths.ownership_row_summary_parquet,
        }
    if stage == "authority":
        return {
            "authority_decisions_parquet": paths.authority_decisions_parquet,
            "authority_exceptions_parquet": paths.authority_exceptions_parquet,
        }
    if stage == "doc_exact_api":
        return {
            "exact_requests_parquet": paths.exact_requests_parquet,
            "exact_raw_parquet": paths.exact_raw_parquet,
        }
    if stage == "doc_fallback_api":
        return {
            "fallback_requests_parquet": paths.fallback_requests_parquet,
            "fallback_raw_parquet": paths.fallback_raw_parquet,
        }
    if stage == "doc_finalize":
        return {
            "doc_ownership_raw_parquet": paths.doc_ownership_raw_parquet,
            "doc_ownership_final_parquet": paths.doc_ownership_final_parquet,
        }
    raise ValueError(f"Unsupported stage: {stage}")


def _stage_resume_sentinels(paths: RunPaths, stage: str) -> tuple[Path, ...]:
    if stage == "lookup_api":
        return (
            paths.refinitiv_step1_dir / "staging" / "lookup",
            paths.refinitiv_step1_dir / "refinitiv_lookup_api_ledger.sqlite3",
            paths.refinitiv_step1_dir / "refinitiv_lookup_api_requests.jsonl",
        )
    if stage == "ownership_api":
        return (
            paths.ownership_universe_dir / "staging" / "ownership_universe",
            paths.ownership_universe_dir / "refinitiv_ownership_universe_api_ledger.sqlite3",
            paths.ownership_universe_dir / "refinitiv_ownership_universe_api_requests.jsonl",
        )
    if stage == "doc_exact_api":
        return (
            paths.doc_ownership_dir / "staging" / "doc_ownership_exact",
            paths.doc_ownership_dir / "refinitiv_doc_ownership_exact_api_ledger.sqlite3",
            paths.doc_ownership_dir / "refinitiv_doc_ownership_exact_api_requests.jsonl",
        )
    if stage == "doc_fallback_api":
        return (
            paths.doc_ownership_dir / "staging" / "doc_ownership_fallback",
            paths.doc_ownership_dir / "refinitiv_doc_ownership_fallback_api_ledger.sqlite3",
            paths.doc_ownership_dir / "refinitiv_doc_ownership_fallback_api_requests.jsonl",
        )
    return ()


def _artifact_map(paths: RunPaths, selected_stages: Sequence[str]) -> dict[str, Any]:
    return {
        "run_root": str(paths.run_root),
        "sec_ccm_premerge_dir": str(paths.sec_ccm_premerge_dir),
        "refinitiv_step1_dir": str(paths.refinitiv_step1_dir),
        "ownership_universe_dir": str(paths.ownership_universe_dir),
        "ownership_authority_dir": str(paths.ownership_authority_dir),
        "doc_ownership_dir": str(paths.doc_ownership_dir),
        "selected_stages": list(selected_stages),
        "stage_contracts": {
            stage: {
                "inputs": {name: str(path) for name, path in _stage_input_paths(paths, stage).items()},
                "outputs": {name: str(path) for name, path in _stage_output_paths(paths, stage).items()},
            }
            for stage in selected_stages
        },
    }


def _ensure_lseg_available() -> None:
    if not is_lseg_available():
        raise RuntimeError("lseg.data is not installed in this runtime")


def _ensure_required_inputs(paths: RunPaths, stage: str) -> None:
    for label, path in _stage_input_paths(paths, stage).items():
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")


def _ensure_clean_start(paths: RunPaths, selected_stages: Sequence[str]) -> None:
    existing_paths: list[Path] = []
    for stage in selected_stages:
        existing_paths.extend(path for path in _stage_output_paths(paths, stage).values() if path.exists())
        existing_paths.extend(path for path in _stage_resume_sentinels(paths, stage) if path.exists())
    if existing_paths:
        formatted = ", ".join(str(path) for path in sorted(existing_paths))
        raise FileExistsError(
            "Existing stage outputs or resume state found. Use --resume or remove them first: "
            f"{formatted}"
        )


def _coerce_stage_result(result: dict[str, Any]) -> dict[str, Path]:
    return {key: Path(value) for key, value in result.items()}


def _run_stage(stage: str, args: argparse.Namespace, paths: RunPaths) -> dict[str, Path]:
    if stage == "lookup_api":
        return _coerce_stage_result(
            run_refinitiv_step1_lookup_api_pipeline(
                snapshot_parquet_path=paths.lookup_snapshot_parquet,
                output_dir=paths.refinitiv_step1_dir,
                max_batch_size=args.lookup_batch_size,
                min_seconds_between_requests=args.min_seconds_between_requests,
                max_attempts=args.max_attempts,
            )
        )
    if stage == "resolution":
        return _coerce_stage_result(
            run_refinitiv_step1_resolution_pipeline(
                filled_lookup_workbook_path=paths.lookup_extended_parquet,
                output_dir=paths.refinitiv_step1_dir,
            )
        )
    if stage == "ownership_handoff":
        return _coerce_stage_result(
            run_refinitiv_step1_ownership_universe_handoff_pipeline(
                resolution_artifact_path=paths.resolution_parquet,
                output_dir=paths.ownership_universe_dir,
            )
        )
    if stage == "ownership_api":
        return _coerce_stage_result(
            run_refinitiv_step1_ownership_universe_api_pipeline(
                handoff_parquet_path=paths.ownership_handoff_parquet,
                output_dir=paths.ownership_universe_dir,
                max_batch_size=args.ownership_batch_size,
                min_seconds_between_requests=args.min_seconds_between_requests,
                max_attempts=args.max_attempts,
            )
        )
    if stage == "authority":
        return _coerce_stage_result(
            run_refinitiv_step1_ownership_authority_pipeline(
                resolution_artifact_path=paths.resolution_parquet,
                ownership_results_artifact_path=paths.ownership_results_parquet,
                ownership_row_summary_artifact_path=paths.ownership_row_summary_parquet,
                output_dir=paths.ownership_authority_dir,
                reviewed_ticker_allowlist_path=paths.reviewed_ticker_allowlist_path,
            )
        )
    if stage == "doc_exact_api":
        return _coerce_stage_result(
            run_refinitiv_lm2011_doc_ownership_exact_api_pipeline(
                doc_filing_artifact_path=paths.doc_filing_artifact_path,
                authority_decisions_artifact_path=paths.authority_decisions_parquet,
                authority_exceptions_artifact_path=paths.authority_exceptions_parquet,
                output_dir=paths.doc_ownership_dir,
                max_batch_size=args.doc_exact_batch_size,
                min_seconds_between_requests=args.min_seconds_between_requests,
                max_attempts=args.max_attempts,
            )
        )
    if stage == "doc_fallback_api":
        return _coerce_stage_result(
            run_refinitiv_lm2011_doc_ownership_fallback_api_pipeline(
                output_dir=paths.doc_ownership_dir,
                max_batch_size=args.doc_fallback_batch_size,
                min_seconds_between_requests=args.min_seconds_between_requests,
                max_attempts=args.max_attempts,
            )
        )
    if stage == "doc_finalize":
        return _coerce_stage_result(
            run_refinitiv_lm2011_doc_ownership_finalize_pipeline(
                output_dir=paths.doc_ownership_dir,
            )
        )
    raise ValueError(f"Unsupported stage: {stage}")


def _row_count(path: Path) -> int | None:
    if not path.exists() or path.suffix.lower() != ".parquet":
        return None
    return int(pl.scan_parquet(path).select(pl.len()).collect().item())


def _write_manifest(
    *,
    args: argparse.Namespace,
    paths: RunPaths,
    selected_stages: Sequence[str],
    generated_artifacts: dict[str, dict[str, Path]],
) -> Path:
    row_counts: dict[str, int] = {}
    for stage, stage_artifacts in generated_artifacts.items():
        for label, path in stage_artifacts.items():
            count = _row_count(path)
            if count is not None:
                row_counts[f"{stage}.{label}"] = count
    payload = {
        "pipeline_name": "refinitiv_local_api_runner",
        "artifact_version": "v1",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "selected_stage_start": args.stage_start,
        "selected_stage_stop": args.stage_stop,
        "selected_stages": list(selected_stages),
        "resume": args.resume,
        "run_root": str(paths.run_root),
        "reviewed_ticker_allowlist_path": str(paths.reviewed_ticker_allowlist_path),
        "generated_artifacts": {
            stage: {label: str(path) for label, path in stage_artifacts.items()}
            for stage, stage_artifacts in generated_artifacts.items()
        },
        "row_counts": row_counts,
    }
    paths.manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return paths.manifest_path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = _resolve_run_paths(
        args.run_root,
        reviewed_ticker_allowlist_path=args.reviewed_ticker_allowlist_path,
    )
    selected_stages = _selected_stages(args.stage_start, args.stage_stop)
    _ensure_lseg_available()
    _ensure_required_inputs(paths, args.stage_start)
    if not args.resume:
        _ensure_clean_start(paths, selected_stages)

    print(json.dumps(_artifact_map(paths, selected_stages), indent=2))

    generated_artifacts: dict[str, dict[str, Path]] = {}
    for stage in selected_stages:
        _ensure_required_inputs(paths, stage)
        print(f"running_stage: {stage}")
        result = _run_stage(stage, args, paths)
        generated_artifacts[stage] = result
        print(
            json.dumps(
                {
                    "stage": stage,
                    "artifacts": {label: str(path) for label, path in result.items()},
                },
                indent=2,
            )
        )

    manifest_path = _write_manifest(
        args=args,
        paths=paths,
        selected_stages=selected_stages,
        generated_artifacts=generated_artifacts,
    )
    print(f"manifest_path: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

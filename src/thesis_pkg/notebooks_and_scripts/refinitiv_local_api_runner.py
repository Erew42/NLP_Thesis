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
    run_refinitiv_step1_analyst_actuals_api_pipeline,
    run_refinitiv_step1_analyst_estimates_monthly_api_pipeline,
    run_refinitiv_step1_analyst_normalize_pipeline,
    run_refinitiv_step1_analyst_request_groups_pipeline,
    run_refinitiv_step1_instrument_authority_pipeline,
    run_refinitiv_step1_lookup_api_pipeline,
    run_refinitiv_step1_ownership_authority_pipeline,
    run_refinitiv_step1_ownership_universe_api_pipeline,
    run_refinitiv_step1_ownership_universe_handoff_pipeline,
    run_refinitiv_step1_resolution_pipeline,
)
from thesis_pkg.pipelines.refinitiv.lseg_analyst_api import (
    ANALYST_ACTUALS_STAGE,
    ANALYST_ESTIMATES_STAGE,
    _assemble_analyst_actuals_raw,
    _assemble_analyst_estimates_raw,
)
from thesis_pkg.pipelines.refinitiv.lseg_lookup_api import (
    LOOKUP_STAGE,
    _assemble_lookup_output,
    _read_lookup_snapshot,
)
from thesis_pkg.pipelines.refinitiv.lseg_ownership_api import (
    DOC_EXACT_STAGE,
    DOC_FALLBACK_STAGE,
    OWNERSHIP_UNIVERSE_STAGE,
    _assemble_doc_raw,
    _assemble_ownership_universe_results,
)
from thesis_pkg.pipelines.refinitiv.lseg_recovery import (
    build_doc_unresolved_recovery_artifact,
    build_lookup_unresolved_recovery_artifact,
    build_ownership_unresolved_recovery_artifact,
    write_recovery_manifest,
)
from thesis_pkg.pipelines.refinitiv.lseg_stage_audit import (
    AuditIssue,
    StageAuditResult,
    audit_api_stage,
    default_stage_fetch_manifest_path,
    default_stage_manifest_path,
    write_stage_completion_manifest,
)
from thesis_pkg.pipelines.refinitiv_bridge_pipeline import build_refinitiv_ownership_universe_row_summary


STAGE_ORDER: tuple[str, ...] = (
    "lookup_api",
    "resolution",
    "instrument_authority",
    "ownership_handoff",
    "ownership_api",
    "authority",
    "analyst_request_groups",
    "analyst_actuals_api",
    "analyst_estimates_api",
    "analyst_normalize",
    "doc_exact_api",
    "doc_fallback_api",
    "doc_finalize",
)

STAGE_MANIFEST_RECOVERY_MODES: dict[str, str] = {
    "lookup_stage_manifest_only": "lookup_api",
    "ownership_stage_manifest_only": "ownership_api",
    "analyst_actuals_stage_manifest_only": "analyst_actuals_api",
    "analyst_estimates_stage_manifest_only": "analyst_estimates_api",
    "doc_exact_stage_manifest_only": "doc_exact_api",
    "doc_fallback_stage_manifest_only": "doc_fallback_api",
}
API_STAGES: frozenset[str] = frozenset(
    {
        "lookup_api",
        "ownership_api",
        "analyst_actuals_api",
        "analyst_estimates_api",
        "doc_exact_api",
        "doc_fallback_api",
    }
)

# Editable local-run defaults. CLI flags still override these values.
LOCAL_LOOKUP_BATCH_SIZE_DEFAULT = 50
LOCAL_OWNERSHIP_BATCH_SIZE_DEFAULT = 50
LOCAL_OWNERSHIP_MAX_BATCH_ITEMS_DEFAULT: int | None = 50
LOCAL_OWNERSHIP_MAX_EXTRA_ROWS_ABS_DEFAULT: float | None = 120.0
LOCAL_OWNERSHIP_MAX_EXTRA_ROWS_RATIO_DEFAULT: float | None = 1
LOCAL_OWNERSHIP_MAX_UNION_SPAN_DAYS_DEFAULT: int | None = None
LOCAL_OWNERSHIP_ROW_DENSITY_ROWS_PER_DAY_DEFAULT: float | None = 1
LOCAL_INCLUDE_TICKER_FALLBACK_DEFAULT = False
LOCAL_ANALYST_ACTUALS_BATCH_SIZE_DEFAULT = 50
LOCAL_ANALYST_ACTUALS_MAX_BATCH_ITEMS_DEFAULT: int | None = 50
LOCAL_ANALYST_ACTUALS_MAX_EXTRA_ROWS_ABS_DEFAULT: float | None = 120.0
LOCAL_ANALYST_ACTUALS_MAX_EXTRA_ROWS_RATIO_DEFAULT: float | None = 1
LOCAL_ANALYST_ACTUALS_MAX_UNION_SPAN_DAYS_DEFAULT: int | None = None
LOCAL_ANALYST_ACTUALS_ROW_DENSITY_ROWS_PER_DAY_DEFAULT: float | None = None
LOCAL_ANALYST_ESTIMATES_BATCH_SIZE_DEFAULT = 50
LOCAL_ANALYST_ESTIMATES_MAX_BATCH_ITEMS_DEFAULT: int | None = 50
LOCAL_ANALYST_ESTIMATES_MAX_EXTRA_ROWS_ABS_DEFAULT: float | None = 240.0
LOCAL_ANALYST_ESTIMATES_MAX_EXTRA_ROWS_RATIO_DEFAULT: float | None = 1
LOCAL_ANALYST_ESTIMATES_MAX_UNION_SPAN_DAYS_DEFAULT: int | None = None
LOCAL_ANALYST_ESTIMATES_ROW_DENSITY_ROWS_PER_DAY_DEFAULT: float | None = None
LOCAL_SAFE_BATCH_PROFILE_DEFAULTS: dict[str, int | float | None] = {
    "lookup_batch_size": 25,
    "ownership_batch_size": 10,
    "ownership_max_batch_items": 10,
    "ownership_max_extra_rows_abs": 120.0,
    "ownership_max_extra_rows_ratio": 0.25,
    "ownership_max_union_span_days": 3650,
    "ownership_row_density_rows_per_day": 1.0 / 91.0,
    "analyst_actuals_batch_size": 25,
    "analyst_actuals_max_batch_items": 25,
    "analyst_actuals_max_extra_rows_abs": 120.0,
    "analyst_actuals_max_extra_rows_ratio": 0.25,
    "analyst_actuals_max_union_span_days": None,
    "analyst_actuals_row_density_rows_per_day": 1.0 / 91.0,
    "analyst_estimates_batch_size": 10,
    "analyst_estimates_max_batch_items": 10,
    "analyst_estimates_max_extra_rows_abs": 240.0,
    "analyst_estimates_max_extra_rows_ratio": 0.15,
    "analyst_estimates_max_union_span_days": None,
    "analyst_estimates_row_density_rows_per_day": 1.0 / 30.5,
    "doc_exact_batch_size": 15,
    "doc_fallback_batch_size": 5,
}
CURRENT_BATCH_PROFILE_DEFAULTS: dict[str, int | float | None] = {
    "lookup_batch_size": LOCAL_LOOKUP_BATCH_SIZE_DEFAULT,
    "ownership_batch_size": LOCAL_OWNERSHIP_BATCH_SIZE_DEFAULT,
    "ownership_max_batch_items": LOCAL_OWNERSHIP_MAX_BATCH_ITEMS_DEFAULT,
    "ownership_max_extra_rows_abs": LOCAL_OWNERSHIP_MAX_EXTRA_ROWS_ABS_DEFAULT,
    "ownership_max_extra_rows_ratio": LOCAL_OWNERSHIP_MAX_EXTRA_ROWS_RATIO_DEFAULT,
    "ownership_max_union_span_days": LOCAL_OWNERSHIP_MAX_UNION_SPAN_DAYS_DEFAULT,
    "ownership_row_density_rows_per_day": LOCAL_OWNERSHIP_ROW_DENSITY_ROWS_PER_DAY_DEFAULT,
    "analyst_actuals_batch_size": LOCAL_ANALYST_ACTUALS_BATCH_SIZE_DEFAULT,
    "analyst_actuals_max_batch_items": LOCAL_ANALYST_ACTUALS_MAX_BATCH_ITEMS_DEFAULT,
    "analyst_actuals_max_extra_rows_abs": LOCAL_ANALYST_ACTUALS_MAX_EXTRA_ROWS_ABS_DEFAULT,
    "analyst_actuals_max_extra_rows_ratio": LOCAL_ANALYST_ACTUALS_MAX_EXTRA_ROWS_RATIO_DEFAULT,
    "analyst_actuals_max_union_span_days": LOCAL_ANALYST_ACTUALS_MAX_UNION_SPAN_DAYS_DEFAULT,
    "analyst_actuals_row_density_rows_per_day": LOCAL_ANALYST_ACTUALS_ROW_DENSITY_ROWS_PER_DAY_DEFAULT,
    "analyst_estimates_batch_size": LOCAL_ANALYST_ESTIMATES_BATCH_SIZE_DEFAULT,
    "analyst_estimates_max_batch_items": LOCAL_ANALYST_ESTIMATES_MAX_BATCH_ITEMS_DEFAULT,
    "analyst_estimates_max_extra_rows_abs": LOCAL_ANALYST_ESTIMATES_MAX_EXTRA_ROWS_ABS_DEFAULT,
    "analyst_estimates_max_extra_rows_ratio": LOCAL_ANALYST_ESTIMATES_MAX_EXTRA_ROWS_RATIO_DEFAULT,
    "analyst_estimates_max_union_span_days": LOCAL_ANALYST_ESTIMATES_MAX_UNION_SPAN_DAYS_DEFAULT,
    "analyst_estimates_row_density_rows_per_day": LOCAL_ANALYST_ESTIMATES_ROW_DENSITY_ROWS_PER_DAY_DEFAULT,
    "doc_exact_batch_size": 15,
    "doc_fallback_batch_size": 5,
}
BATCH_PROFILES: dict[str, dict[str, int | float | None]] = {
    "current": CURRENT_BATCH_PROFILE_DEFAULTS,
    "local_safe": LOCAL_SAFE_BATCH_PROFILE_DEFAULTS,
}


@dataclass(frozen=True)
class RunPaths:
    run_root: Path
    sec_ccm_premerge_dir: Path
    refinitiv_step1_dir: Path
    analyst_dir: Path
    ownership_universe_dir: Path
    ownership_authority_dir: Path
    doc_ownership_dir: Path
    manifest_path: Path
    doc_filing_artifact_path: Path
    lookup_snapshot_parquet: Path
    lookup_extended_parquet: Path
    resolution_parquet: Path
    bridge_parquet: Path
    instrument_authority_parquet: Path
    ownership_handoff_parquet: Path
    ownership_results_parquet: Path
    ownership_row_summary_parquet: Path
    analyst_request_group_membership_parquet: Path
    analyst_request_universe_parquet: Path
    analyst_actuals_raw_parquet: Path
    analyst_estimates_raw_parquet: Path
    analyst_normalized_panel_parquet: Path
    analyst_normalization_rejections_parquet: Path
    authority_decisions_parquet: Path
    authority_exceptions_parquet: Path
    reviewed_ticker_allowlist_path: Path
    lookup_stage_manifest_path: Path
    lookup_fetch_manifest_path: Path
    ownership_stage_manifest_path: Path
    ownership_fetch_manifest_path: Path
    analyst_actuals_stage_manifest_path: Path
    analyst_actuals_fetch_manifest_path: Path
    analyst_estimates_stage_manifest_path: Path
    analyst_estimates_fetch_manifest_path: Path
    doc_exact_stage_manifest_path: Path
    doc_exact_fetch_manifest_path: Path
    doc_fallback_stage_manifest_path: Path
    doc_fallback_fetch_manifest_path: Path
    exact_requests_parquet: Path
    exact_raw_parquet: Path
    fallback_requests_parquet: Path
    fallback_raw_parquet: Path
    doc_ownership_raw_parquet: Path
    doc_ownership_final_parquet: Path
    recovery_dir: Path


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
        default=None,
        help="First stage to execute.",
    )
    parser.add_argument(
        "--stage-stop",
        choices=STAGE_ORDER,
        default=None,
        help="Last stage to execute.",
    )
    parser.add_argument(
        "--stage-list",
        type=str,
        default=None,
        help="Comma-separated explicit stage selection. Mutually exclusive with --stage-start/--stage-stop.",
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
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Validate existing artifacts for the selected stages without running API requests.",
    )
    parser.add_argument(
        "--recover-mode",
        choices=(
            "lookup_unresolved",
            "ownership_unresolved",
            "doc_exact_unresolved",
            "doc_fallback_unresolved",
            "resolution_only",
            "doc_finalize_only",
            *STAGE_MANIFEST_RECOVERY_MODES.keys(),
        ),
        default=None,
        help="Run an explicit recovery workflow instead of the normal stage range.",
    )
    parser.add_argument(
        "--recovery-output-path",
        type=Path,
        default=None,
        help="Optional override for the recovery parquet output path used by unresolved-row recovery modes.",
    )
    parser.add_argument(
        "--preflight-probe",
        action="store_true",
        help="Issue a single live probe request before starting each selected API stage.",
    )
    parser.add_argument(
        "--stage-manifest-required",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require canonical upstream stage manifests before consuming prior API outputs.",
    )
    parser.add_argument(
        "--api-stage-mode",
        choices=("full", "fetch_only", "finalize_only"),
        default="full",
        help="Execution mode for API-backed stages.",
    )
    parser.add_argument(
        "--batch-profile",
        choices=("current", "local_safe"),
        default="current",
        help="Batching preset applied to any stage batch settings you do not set explicitly.",
    )
    parser.add_argument("--provider-session-name", type=str, default="desktop.workspace")
    parser.add_argument("--provider-config-name", type=str, default=None)
    parser.add_argument("--provider-timeout-seconds", type=float, default=None)
    parser.add_argument("--lookup-batch-size", type=int, default=None)
    parser.add_argument("--ownership-batch-size", type=int, default=None)
    parser.add_argument("--ownership-max-batch-items", type=int, default=None)
    parser.add_argument(
        "--ownership-max-extra-rows-abs",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--ownership-max-extra-rows-ratio",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--ownership-max-union-span-days",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--ownership-row-density-rows-per-day",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--ticker-fallback",
        dest="include_ticker_fallback",
        action=argparse.BooleanOptionalAction,
        default=LOCAL_INCLUDE_TICKER_FALLBACK_DEFAULT,
        help="Allow ticker-derived fallback rows in the ownership-universe handoff.",
    )
    parser.add_argument("--analyst-actuals-batch-size", type=int, default=None)
    parser.add_argument(
        "--analyst-estimates-batch-size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--analyst-actuals-max-batch-items",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--analyst-actuals-max-extra-rows-abs",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--analyst-actuals-max-extra-rows-ratio",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--analyst-actuals-max-union-span-days",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--analyst-actuals-row-density-rows-per-day",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--analyst-estimates-max-batch-items",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--analyst-estimates-max-extra-rows-abs",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--analyst-estimates-max-extra-rows-ratio",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--analyst-estimates-max-union-span-days",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--analyst-estimates-row-density-rows-per-day",
        type=float,
        default=None,
    )
    parser.add_argument("--doc-exact-batch-size", type=int, default=None)
    parser.add_argument("--doc-fallback-batch-size", type=int, default=None)
    parser.add_argument("--min-seconds-between-requests", type=float, default=2.0)
    parser.add_argument(
        "--min-seconds-between-request-starts-total",
        type=float,
        default=None,
        help=(
            "Aggregate minimum gap between request starts across all workers when --max-workers > 1. "
            "Required for concurrent mode."
        ),
    )
    parser.add_argument("--max-attempts", type=int, default=4)
    #The concurrent multi worker approach does currently not work well with the Refinitiv APIs, so I default to 1 worker to preserve the expected behavior. Adjust with caution.
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of worker processes for API-backed stages. Use 1 to preserve sequential execution.",
    )
    # Useful when also having implemented a ticker filtering approach, as ticker data is messy (Not done currently).
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
    explicit_stage_list = _parse_stage_list(args.stage_list, parser=parser)
    if explicit_stage_list:
        if args.stage_start is not None or args.stage_stop is not None:
            parser.error("--stage-list is mutually exclusive with --stage-start/--stage-stop")
        args.selected_stages = explicit_stage_list
        args.stage_start = explicit_stage_list[0]
        args.stage_stop = explicit_stage_list[-1]
    else:
        args.stage_start = STAGE_ORDER[0] if args.stage_start is None else args.stage_start
        args.stage_stop = STAGE_ORDER[-1] if args.stage_stop is None else args.stage_stop
        if args.recover_mode is None and STAGE_ORDER.index(args.stage_start) > STAGE_ORDER.index(args.stage_stop):
            parser.error("--stage-start must be earlier than or equal to --stage-stop")
        args.selected_stages = _selected_stages(args.stage_start, args.stage_stop)
    if args.recover_mode is None and args.api_stage_mode != "full":
        if len(args.selected_stages) != 1 or args.selected_stages[0] not in API_STAGES:
            parser.error("--api-stage-mode fetch_only/finalize_only requires selecting exactly one API stage")
    args.run_root = args.run_root.expanduser().resolve()
    if args.reviewed_ticker_allowlist_path is not None:
        args.reviewed_ticker_allowlist_path = args.reviewed_ticker_allowlist_path.expanduser().resolve()
    if args.recovery_output_path is not None:
        args.recovery_output_path = args.recovery_output_path.expanduser().resolve()
    if args.max_workers > 1 and args.min_seconds_between_request_starts_total is None:
        parser.error(
            "--min-seconds-between-request-starts-total is required when --max-workers is greater than 1"
        )
    _apply_batch_profile_defaults(args)
    return args


def _parse_stage_list(stage_list_text: str | None, *, parser: argparse.ArgumentParser) -> tuple[str, ...] | None:
    if stage_list_text is None:
        return None
    stages = tuple(part.strip() for part in stage_list_text.split(",") if part.strip())
    if not stages:
        parser.error("--stage-list must contain at least one stage")
    invalid = [stage for stage in stages if stage not in STAGE_ORDER]
    if invalid:
        parser.error(f"unknown stages in --stage-list: {', '.join(invalid)}")
    if len(set(stages)) != len(stages):
        parser.error("--stage-list must not contain duplicate stages")
    stage_indexes = [STAGE_ORDER.index(stage) for stage in stages]
    if stage_indexes != sorted(stage_indexes):
        parser.error("--stage-list must follow the pipeline stage order")
    return stages


def _apply_batch_profile_defaults(args: argparse.Namespace) -> None:
    profile_defaults = BATCH_PROFILES[args.batch_profile]
    for attribute, value in profile_defaults.items():
        if getattr(args, attribute) is None:
            setattr(args, attribute, value)


def _resolve_run_paths(
    run_root: Path,
    *,
    reviewed_ticker_allowlist_path: Path | None,
) -> RunPaths:
    sec_ccm_premerge_dir = run_root / "sec_ccm_premerge"
    refinitiv_step1_dir = run_root / "refinitiv_step1"
    analyst_dir = refinitiv_step1_dir / "analyst_common_stock"
    ownership_universe_dir = refinitiv_step1_dir / "ownership_universe_common_stock"
    ownership_authority_dir = refinitiv_step1_dir / "ownership_authority_common_stock"
    doc_ownership_dir = run_root / "refinitiv_doc_ownership_lm2011"
    return RunPaths(
        run_root=run_root,
        sec_ccm_premerge_dir=sec_ccm_premerge_dir,
        refinitiv_step1_dir=refinitiv_step1_dir,
        analyst_dir=analyst_dir,
        ownership_universe_dir=ownership_universe_dir,
        ownership_authority_dir=ownership_authority_dir,
        doc_ownership_dir=doc_ownership_dir,
        manifest_path=run_root / "refinitiv_local_api_runner_manifest.json",
        doc_filing_artifact_path=sec_ccm_premerge_dir / "sec_ccm_matched_clean_filtered.parquet",
        lookup_snapshot_parquet=refinitiv_step1_dir / "refinitiv_ric_lookup_handoff_common_stock_extended_snapshot.parquet",
        lookup_extended_parquet=refinitiv_step1_dir / "refinitiv_ric_lookup_handoff_common_stock_extended.parquet",
        resolution_parquet=refinitiv_step1_dir / "refinitiv_ric_resolution_common_stock.parquet",
        bridge_parquet=refinitiv_step1_dir / "refinitiv_bridge_universe.parquet",
        instrument_authority_parquet=refinitiv_step1_dir / "refinitiv_instrument_authority_common_stock.parquet",
        ownership_handoff_parquet=ownership_universe_dir / "refinitiv_ownership_universe_handoff_common_stock.parquet",
        ownership_results_parquet=ownership_universe_dir / "refinitiv_ownership_universe_results.parquet",
        ownership_row_summary_parquet=ownership_universe_dir / "refinitiv_ownership_universe_row_summary.parquet",
        analyst_request_group_membership_parquet=analyst_dir / "refinitiv_analyst_request_group_membership_common_stock.parquet",
        analyst_request_universe_parquet=analyst_dir / "refinitiv_analyst_request_universe_common_stock.parquet",
        analyst_actuals_raw_parquet=analyst_dir / "refinitiv_analyst_actuals_raw.parquet",
        analyst_estimates_raw_parquet=analyst_dir / "refinitiv_analyst_estimates_monthly_raw.parquet",
        analyst_normalized_panel_parquet=analyst_dir / "refinitiv_analyst_normalized_panel.parquet",
        analyst_normalization_rejections_parquet=analyst_dir / "refinitiv_analyst_normalization_rejections.parquet",
        authority_decisions_parquet=ownership_authority_dir / "refinitiv_permno_ownership_authority_decisions.parquet",
        authority_exceptions_parquet=ownership_authority_dir / "refinitiv_permno_ownership_authority_exceptions.parquet",
        reviewed_ticker_allowlist_path=(
            reviewed_ticker_allowlist_path
            if reviewed_ticker_allowlist_path is not None
            else ownership_authority_dir / "refinitiv_permno_ownership_ticker_allowlist.parquet"
        ),
        lookup_stage_manifest_path=default_stage_manifest_path(refinitiv_step1_dir, LOOKUP_STAGE),
        lookup_fetch_manifest_path=default_stage_fetch_manifest_path(refinitiv_step1_dir, LOOKUP_STAGE),
        ownership_stage_manifest_path=default_stage_manifest_path(ownership_universe_dir, OWNERSHIP_UNIVERSE_STAGE),
        ownership_fetch_manifest_path=default_stage_fetch_manifest_path(ownership_universe_dir, OWNERSHIP_UNIVERSE_STAGE),
        analyst_actuals_stage_manifest_path=default_stage_manifest_path(analyst_dir, ANALYST_ACTUALS_STAGE),
        analyst_actuals_fetch_manifest_path=default_stage_fetch_manifest_path(analyst_dir, ANALYST_ACTUALS_STAGE),
        analyst_estimates_stage_manifest_path=default_stage_manifest_path(analyst_dir, ANALYST_ESTIMATES_STAGE),
        analyst_estimates_fetch_manifest_path=default_stage_fetch_manifest_path(analyst_dir, ANALYST_ESTIMATES_STAGE),
        doc_exact_stage_manifest_path=default_stage_manifest_path(doc_ownership_dir, DOC_EXACT_STAGE),
        doc_exact_fetch_manifest_path=default_stage_fetch_manifest_path(doc_ownership_dir, DOC_EXACT_STAGE),
        doc_fallback_stage_manifest_path=default_stage_manifest_path(doc_ownership_dir, DOC_FALLBACK_STAGE),
        doc_fallback_fetch_manifest_path=default_stage_fetch_manifest_path(doc_ownership_dir, DOC_FALLBACK_STAGE),
        exact_requests_parquet=doc_ownership_dir / "refinitiv_lm2011_doc_ownership_exact_requests.parquet",
        exact_raw_parquet=doc_ownership_dir / "refinitiv_lm2011_doc_ownership_exact_raw.parquet",
        fallback_requests_parquet=doc_ownership_dir / "refinitiv_lm2011_doc_ownership_fallback_requests.parquet",
        fallback_raw_parquet=doc_ownership_dir / "refinitiv_lm2011_doc_ownership_fallback_raw.parquet",
        doc_ownership_raw_parquet=doc_ownership_dir / "refinitiv_lm2011_doc_ownership_raw.parquet",
        doc_ownership_final_parquet=doc_ownership_dir / "refinitiv_lm2011_doc_ownership.parquet",
        recovery_dir=run_root / "refinitiv_recovery",
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
    if stage == "instrument_authority":
        return {
            "bridge_parquet": paths.bridge_parquet,
            "resolution_parquet": paths.resolution_parquet,
        }
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
    if stage == "analyst_request_groups":
        return {"instrument_authority_parquet": paths.instrument_authority_parquet}
    if stage == "analyst_actuals_api":
        return {"analyst_request_universe_parquet": paths.analyst_request_universe_parquet}
    if stage == "analyst_estimates_api":
        return {"analyst_request_universe_parquet": paths.analyst_request_universe_parquet}
    if stage == "analyst_normalize":
        return {
            "analyst_actuals_raw_parquet": paths.analyst_actuals_raw_parquet,
            "analyst_estimates_raw_parquet": paths.analyst_estimates_raw_parquet,
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
    if stage == "instrument_authority":
        return {"instrument_authority_parquet": paths.instrument_authority_parquet}
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
    if stage == "analyst_request_groups":
        return {
            "analyst_request_group_membership_parquet": paths.analyst_request_group_membership_parquet,
            "analyst_request_universe_parquet": paths.analyst_request_universe_parquet,
        }
    if stage == "analyst_actuals_api":
        return {"analyst_actuals_raw_parquet": paths.analyst_actuals_raw_parquet}
    if stage == "analyst_estimates_api":
        return {"analyst_estimates_raw_parquet": paths.analyst_estimates_raw_parquet}
    if stage == "analyst_normalize":
        return {
            "analyst_normalized_panel_parquet": paths.analyst_normalized_panel_parquet,
            "analyst_normalization_rejections_parquet": paths.analyst_normalization_rejections_parquet,
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
    if stage == "analyst_actuals_api":
        return (
            paths.analyst_dir / "staging" / "analyst_actuals",
            paths.analyst_dir / "refinitiv_analyst_actuals_api_ledger.sqlite3",
            paths.analyst_dir / "refinitiv_analyst_actuals_api_requests.jsonl",
        )
    if stage == "analyst_estimates_api":
        return (
            paths.analyst_dir / "staging" / "analyst_estimates_monthly",
            paths.analyst_dir / "refinitiv_analyst_estimates_api_ledger.sqlite3",
            paths.analyst_dir / "refinitiv_analyst_estimates_api_requests.jsonl",
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


def _stage_manifest_outputs(paths: RunPaths, stage: str) -> tuple[Path, ...]:
    if stage == "lookup_api":
        return (paths.lookup_stage_manifest_path,)
    if stage == "ownership_api":
        return (paths.ownership_stage_manifest_path,)
    if stage == "analyst_actuals_api":
        return (paths.analyst_actuals_stage_manifest_path,)
    if stage == "analyst_estimates_api":
        return (paths.analyst_estimates_stage_manifest_path,)
    if stage == "doc_exact_api":
        return (paths.doc_exact_stage_manifest_path,)
    if stage == "doc_fallback_api":
        return (paths.doc_fallback_stage_manifest_path,)
    return ()


def _stage_fetch_manifest_outputs(paths: RunPaths, stage: str) -> tuple[Path, ...]:
    if stage == "lookup_api":
        return (paths.lookup_fetch_manifest_path,)
    if stage == "ownership_api":
        return (paths.ownership_fetch_manifest_path,)
    if stage == "analyst_actuals_api":
        return (paths.analyst_actuals_fetch_manifest_path,)
    if stage == "analyst_estimates_api":
        return (paths.analyst_estimates_fetch_manifest_path,)
    if stage == "doc_exact_api":
        return (paths.doc_exact_fetch_manifest_path,)
    if stage == "doc_fallback_api":
        return (paths.doc_fallback_fetch_manifest_path,)
    return ()


def _required_stage_manifests(paths: RunPaths, stage: str) -> tuple[Path, ...]:
    if stage in {"resolution", "instrument_authority", "ownership_handoff", "analyst_request_groups"}:
        return (paths.lookup_stage_manifest_path,)
    if stage in {"authority", "doc_exact_api"}:
        return (paths.ownership_stage_manifest_path,)
    if stage == "analyst_normalize":
        return (
            paths.analyst_actuals_stage_manifest_path,
            paths.analyst_estimates_stage_manifest_path,
        )
    if stage == "doc_fallback_api":
        return (paths.doc_exact_stage_manifest_path,)
    if stage == "doc_finalize":
        return (
            paths.doc_exact_stage_manifest_path,
            paths.doc_fallback_stage_manifest_path,
        )
    return ()


def _artifact_map(paths: RunPaths, selected_stages: Sequence[str]) -> dict[str, Any]:
    return {
        "run_root": str(paths.run_root),
        "sec_ccm_premerge_dir": str(paths.sec_ccm_premerge_dir),
        "refinitiv_step1_dir": str(paths.refinitiv_step1_dir),
        "analyst_dir": str(paths.analyst_dir),
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


def _ensure_required_stage_manifests(
    paths: RunPaths,
    stage: str,
    *,
    completed_stages: set[str] | None = None,
) -> None:
    completed_stages = completed_stages or set()
    for manifest_path in _required_stage_manifests(paths, stage):
        producer_stage = _manifest_producer_stage(paths, manifest_path)
        if producer_stage in completed_stages:
            continue
        if not manifest_path.exists():
            raise FileNotFoundError(f"required_stage_manifest not found: {manifest_path}")


def _manifest_producer_stage(paths: RunPaths, manifest_path: Path) -> str | None:
    mapping = {
        paths.lookup_stage_manifest_path: "lookup_api",
        paths.ownership_stage_manifest_path: "ownership_api",
        paths.analyst_actuals_stage_manifest_path: "analyst_actuals_api",
        paths.analyst_estimates_stage_manifest_path: "analyst_estimates_api",
        paths.doc_exact_stage_manifest_path: "doc_exact_api",
        paths.doc_fallback_stage_manifest_path: "doc_fallback_api",
    }
    return mapping.get(manifest_path)


def _ensure_clean_start(paths: RunPaths, selected_stages: Sequence[str]) -> None:
    existing_paths: list[Path] = []
    for stage in selected_stages:
        existing_paths.extend(path for path in _stage_output_paths(paths, stage).values() if path.exists())
        existing_paths.extend(path for path in _stage_resume_sentinels(paths, stage) if path.exists())
        existing_paths.extend(path for path in _stage_manifest_outputs(paths, stage) if path.exists())
        existing_paths.extend(path for path in _stage_fetch_manifest_outputs(paths, stage) if path.exists())
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
                min_seconds_between_request_starts_total=args.min_seconds_between_request_starts_total,
                max_attempts=args.max_attempts,
                max_workers=args.max_workers,
                provider_session_name=args.provider_session_name,
                provider_config_name=args.provider_config_name,
                provider_timeout_seconds=args.provider_timeout_seconds,
                preflight_probe=args.preflight_probe,
                stage_manifest_path=paths.lookup_stage_manifest_path,
                fetch_manifest_path=paths.lookup_fetch_manifest_path,
                api_stage_mode=args.api_stage_mode,
            )
        )
    if stage == "resolution":
        return _coerce_stage_result(
            run_refinitiv_step1_resolution_pipeline(
                filled_lookup_workbook_path=paths.lookup_extended_parquet,
                output_dir=paths.refinitiv_step1_dir,
            )
        )
    if stage == "instrument_authority":
        return _coerce_stage_result(
            run_refinitiv_step1_instrument_authority_pipeline(
                bridge_artifact_path=paths.bridge_parquet,
                resolution_artifact_path=paths.resolution_parquet,
                output_dir=paths.refinitiv_step1_dir,
            )
        )
    if stage == "ownership_handoff":
        return _coerce_stage_result(
            run_refinitiv_step1_ownership_universe_handoff_pipeline(
                resolution_artifact_path=paths.resolution_parquet,
                output_dir=paths.ownership_universe_dir,
                include_ticker_fallback=args.include_ticker_fallback,
            )
        )
    if stage == "ownership_api":
        return _coerce_stage_result(
            run_refinitiv_step1_ownership_universe_api_pipeline(
                handoff_parquet_path=paths.ownership_handoff_parquet,
                output_dir=paths.ownership_universe_dir,
                max_batch_size=args.ownership_batch_size,
                max_batch_items=args.ownership_max_batch_items,
                max_extra_rows_abs=args.ownership_max_extra_rows_abs,
                max_extra_rows_ratio=args.ownership_max_extra_rows_ratio,
                max_union_span_days=args.ownership_max_union_span_days,
                row_density_rows_per_day=args.ownership_row_density_rows_per_day,
                min_seconds_between_requests=args.min_seconds_between_requests,
                min_seconds_between_request_starts_total=args.min_seconds_between_request_starts_total,
                max_attempts=args.max_attempts,
                max_workers=args.max_workers,
                provider_session_name=args.provider_session_name,
                provider_config_name=args.provider_config_name,
                provider_timeout_seconds=args.provider_timeout_seconds,
                preflight_probe=args.preflight_probe,
                stage_manifest_path=paths.ownership_stage_manifest_path,
                fetch_manifest_path=paths.ownership_fetch_manifest_path,
                api_stage_mode=args.api_stage_mode,
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
    if stage == "analyst_request_groups":
        return _coerce_stage_result(
            run_refinitiv_step1_analyst_request_groups_pipeline(
                instrument_authority_artifact_path=paths.instrument_authority_parquet,
                output_dir=paths.analyst_dir,
            )
        )
    if stage == "analyst_actuals_api":
        return _coerce_stage_result(
            run_refinitiv_step1_analyst_actuals_api_pipeline(
                request_universe_parquet_path=paths.analyst_request_universe_parquet,
                output_dir=paths.analyst_dir,
                max_batch_size=args.analyst_actuals_batch_size,
                max_batch_items=args.analyst_actuals_max_batch_items,
                max_extra_rows_abs=args.analyst_actuals_max_extra_rows_abs,
                max_extra_rows_ratio=args.analyst_actuals_max_extra_rows_ratio,
                max_union_span_days=args.analyst_actuals_max_union_span_days,
                row_density_rows_per_day=args.analyst_actuals_row_density_rows_per_day,
                min_seconds_between_requests=args.min_seconds_between_requests,
                min_seconds_between_request_starts_total=args.min_seconds_between_request_starts_total,
                max_attempts=args.max_attempts,
                max_workers=args.max_workers,
                provider_session_name=args.provider_session_name,
                provider_config_name=args.provider_config_name,
                provider_timeout_seconds=args.provider_timeout_seconds,
                preflight_probe=args.preflight_probe,
                stage_manifest_path=paths.analyst_actuals_stage_manifest_path,
                fetch_manifest_path=paths.analyst_actuals_fetch_manifest_path,
                api_stage_mode=args.api_stage_mode,
            )
        )
    if stage == "analyst_estimates_api":
        return _coerce_stage_result(
            run_refinitiv_step1_analyst_estimates_monthly_api_pipeline(
                request_universe_parquet_path=paths.analyst_request_universe_parquet,
                output_dir=paths.analyst_dir,
                max_batch_size=args.analyst_estimates_batch_size,
                max_batch_items=args.analyst_estimates_max_batch_items,
                max_extra_rows_abs=args.analyst_estimates_max_extra_rows_abs,
                max_extra_rows_ratio=args.analyst_estimates_max_extra_rows_ratio,
                max_union_span_days=args.analyst_estimates_max_union_span_days,
                row_density_rows_per_day=args.analyst_estimates_row_density_rows_per_day,
                min_seconds_between_requests=args.min_seconds_between_requests,
                min_seconds_between_request_starts_total=args.min_seconds_between_request_starts_total,
                max_attempts=args.max_attempts,
                max_workers=args.max_workers,
                provider_session_name=args.provider_session_name,
                provider_config_name=args.provider_config_name,
                provider_timeout_seconds=args.provider_timeout_seconds,
                preflight_probe=args.preflight_probe,
                stage_manifest_path=paths.analyst_estimates_stage_manifest_path,
                fetch_manifest_path=paths.analyst_estimates_fetch_manifest_path,
                api_stage_mode=args.api_stage_mode,
            )
        )
    if stage == "analyst_normalize":
        return _coerce_stage_result(
            run_refinitiv_step1_analyst_normalize_pipeline(
                actuals_raw_artifact_path=paths.analyst_actuals_raw_parquet,
                estimates_raw_artifact_path=paths.analyst_estimates_raw_parquet,
                output_dir=paths.analyst_dir,
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
                min_seconds_between_request_starts_total=args.min_seconds_between_request_starts_total,
                max_attempts=args.max_attempts,
                max_workers=args.max_workers,
                provider_session_name=args.provider_session_name,
                provider_config_name=args.provider_config_name,
                provider_timeout_seconds=args.provider_timeout_seconds,
                preflight_probe=args.preflight_probe,
                stage_manifest_path=paths.doc_exact_stage_manifest_path,
                fetch_manifest_path=paths.doc_exact_fetch_manifest_path,
                api_stage_mode=args.api_stage_mode,
            )
        )
    if stage == "doc_fallback_api":
        return _coerce_stage_result(
            run_refinitiv_lm2011_doc_ownership_fallback_api_pipeline(
                output_dir=paths.doc_ownership_dir,
                max_batch_size=args.doc_fallback_batch_size,
                min_seconds_between_requests=args.min_seconds_between_requests,
                min_seconds_between_request_starts_total=args.min_seconds_between_request_starts_total,
                max_attempts=args.max_attempts,
                max_workers=args.max_workers,
                provider_session_name=args.provider_session_name,
                provider_config_name=args.provider_config_name,
                provider_timeout_seconds=args.provider_timeout_seconds,
                preflight_probe=args.preflight_probe,
                stage_manifest_path=paths.doc_fallback_stage_manifest_path,
                fetch_manifest_path=paths.doc_fallback_fetch_manifest_path,
                api_stage_mode=args.api_stage_mode,
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
    mode: str,
    args: argparse.Namespace,
    paths: RunPaths,
    selected_stages: Sequence[str],
    generated_artifacts: dict[str, dict[str, Path]] | None = None,
    audit_results: dict[str, StageAuditResult] | None = None,
    recovery_result: dict[str, Any] | None = None,
) -> Path:
    row_counts: dict[str, int] = {}
    generated_artifacts = generated_artifacts or {}
    audit_results = audit_results or {}
    for stage, stage_artifacts in generated_artifacts.items():
        for label, path in stage_artifacts.items():
            count = _row_count(path)
            if count is not None:
                row_counts[f"{stage}.{label}"] = count
    payload = {
        "pipeline_name": "refinitiv_local_api_runner",
        "artifact_version": "v2",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": mode,
        "selected_stage_start": args.stage_start,
        "selected_stage_stop": args.stage_stop,
        "selected_stages": list(selected_stages),
        "selected_stage_list_explicit": None if args.stage_list is None else list(selected_stages),
        "resume": args.resume,
        "audit_only": args.audit_only,
        "recover_mode": args.recover_mode,
        "api_stage_mode": args.api_stage_mode,
        "batch_profile": args.batch_profile,
        "run_root": str(paths.run_root),
        "reviewed_ticker_allowlist_path": str(paths.reviewed_ticker_allowlist_path),
        "provider_session_name": args.provider_session_name,
        "provider_config_name": args.provider_config_name,
        "provider_timeout_seconds": args.provider_timeout_seconds,
        "batching": {
            "lookup_batch_size": args.lookup_batch_size,
            "ownership_batch_size": args.ownership_batch_size,
            "ownership_max_batch_items": args.ownership_max_batch_items,
            "ownership_max_extra_rows_abs": args.ownership_max_extra_rows_abs,
            "ownership_max_extra_rows_ratio": args.ownership_max_extra_rows_ratio,
            "ownership_max_union_span_days": args.ownership_max_union_span_days,
            "ownership_row_density_rows_per_day": args.ownership_row_density_rows_per_day,
            "include_ticker_fallback": args.include_ticker_fallback,
            "analyst_actuals_batch_size": args.analyst_actuals_batch_size,
            "analyst_actuals_max_batch_items": args.analyst_actuals_max_batch_items,
            "analyst_actuals_max_extra_rows_abs": args.analyst_actuals_max_extra_rows_abs,
            "analyst_actuals_max_extra_rows_ratio": args.analyst_actuals_max_extra_rows_ratio,
            "analyst_actuals_max_union_span_days": args.analyst_actuals_max_union_span_days,
            "analyst_actuals_row_density_rows_per_day": args.analyst_actuals_row_density_rows_per_day,
            "analyst_estimates_batch_size": args.analyst_estimates_batch_size,
            "analyst_estimates_max_batch_items": args.analyst_estimates_max_batch_items,
            "analyst_estimates_max_extra_rows_abs": args.analyst_estimates_max_extra_rows_abs,
            "analyst_estimates_max_extra_rows_ratio": args.analyst_estimates_max_extra_rows_ratio,
            "analyst_estimates_max_union_span_days": args.analyst_estimates_max_union_span_days,
            "analyst_estimates_row_density_rows_per_day": args.analyst_estimates_row_density_rows_per_day,
            "doc_exact_batch_size": args.doc_exact_batch_size,
            "doc_fallback_batch_size": args.doc_fallback_batch_size,
        },
        "generated_artifacts": {
            stage: {label: str(path) for label, path in stage_artifacts.items()}
            for stage, stage_artifacts in generated_artifacts.items()
        },
        "audit_results": {stage: result.to_dict() for stage, result in audit_results.items()},
        "recovery_result": recovery_result or {},
        "row_counts": row_counts,
    }
    paths.manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return paths.manifest_path


def _audit_stage(stage: str, paths: RunPaths) -> StageAuditResult:
    if stage == "lookup_api":
        snapshot_df = _read_lookup_snapshot(paths.lookup_snapshot_parquet)
        return audit_api_stage(
            stage_name=LOOKUP_STAGE,
            ledger_path=paths.refinitiv_step1_dir / "refinitiv_lookup_api_ledger.sqlite3",
            staging_dir=paths.refinitiv_step1_dir / "staging" / LOOKUP_STAGE,
            output_artifacts={"lookup_extended_parquet": paths.lookup_extended_parquet},
            rebuilders={
                "lookup_extended_parquet": lambda: _assemble_lookup_output(
                    snapshot_df,
                    paths.refinitiv_step1_dir / "staging" / LOOKUP_STAGE,
                    ledger_path=paths.refinitiv_step1_dir / "refinitiv_lookup_api_ledger.sqlite3",
                )
            },
            expected_stage_manifest_path=paths.lookup_stage_manifest_path,
        )
    if stage == "ownership_api":
        handoff_df = pl.read_parquet(paths.ownership_handoff_parquet)
        return audit_api_stage(
            stage_name=OWNERSHIP_UNIVERSE_STAGE,
            ledger_path=paths.ownership_universe_dir / "refinitiv_ownership_universe_api_ledger.sqlite3",
            staging_dir=paths.ownership_universe_dir / "staging" / OWNERSHIP_UNIVERSE_STAGE,
            output_artifacts={
                "ownership_results_parquet": paths.ownership_results_parquet,
                "ownership_row_summary_parquet": paths.ownership_row_summary_parquet,
            },
            rebuilders={
                "ownership_results_parquet": lambda: _assemble_ownership_universe_results(
                    paths.ownership_universe_dir / "staging" / OWNERSHIP_UNIVERSE_STAGE,
                    ledger_path=paths.ownership_universe_dir / "refinitiv_ownership_universe_api_ledger.sqlite3",
                ),
                "ownership_row_summary_parquet": lambda: build_refinitiv_ownership_universe_row_summary(
                    handoff_df,
                    _assemble_ownership_universe_results(
                        paths.ownership_universe_dir / "staging" / OWNERSHIP_UNIVERSE_STAGE,
                        ledger_path=paths.ownership_universe_dir / "refinitiv_ownership_universe_api_ledger.sqlite3",
                    ),
                ),
            },
            expected_stage_manifest_path=paths.ownership_stage_manifest_path,
        )
    if stage == "analyst_actuals_api":
        return audit_api_stage(
            stage_name=ANALYST_ACTUALS_STAGE,
            ledger_path=paths.analyst_dir / "refinitiv_analyst_actuals_api_ledger.sqlite3",
            staging_dir=paths.analyst_dir / "staging" / ANALYST_ACTUALS_STAGE,
            output_artifacts={"analyst_actuals_raw_parquet": paths.analyst_actuals_raw_parquet},
            rebuilders={
                "analyst_actuals_raw_parquet": lambda: _assemble_analyst_actuals_raw(
                    paths.analyst_dir / "staging" / ANALYST_ACTUALS_STAGE,
                    ledger_path=paths.analyst_dir / "refinitiv_analyst_actuals_api_ledger.sqlite3",
                )
            },
            expected_stage_manifest_path=paths.analyst_actuals_stage_manifest_path,
        )
    if stage == "analyst_estimates_api":
        return audit_api_stage(
            stage_name=ANALYST_ESTIMATES_STAGE,
            ledger_path=paths.analyst_dir / "refinitiv_analyst_estimates_api_ledger.sqlite3",
            staging_dir=paths.analyst_dir / "staging" / ANALYST_ESTIMATES_STAGE,
            output_artifacts={"analyst_estimates_raw_parquet": paths.analyst_estimates_raw_parquet},
            rebuilders={
                "analyst_estimates_raw_parquet": lambda: _assemble_analyst_estimates_raw(
                    paths.analyst_dir / "staging" / ANALYST_ESTIMATES_STAGE,
                    ledger_path=paths.analyst_dir / "refinitiv_analyst_estimates_api_ledger.sqlite3",
                )
            },
            expected_stage_manifest_path=paths.analyst_estimates_stage_manifest_path,
        )
    if stage == "doc_exact_api":
        return audit_api_stage(
            stage_name=DOC_EXACT_STAGE,
            ledger_path=paths.doc_ownership_dir / "refinitiv_doc_ownership_exact_api_ledger.sqlite3",
            staging_dir=paths.doc_ownership_dir / "staging" / DOC_EXACT_STAGE,
            output_artifacts={
                "exact_requests_parquet": paths.exact_requests_parquet,
                "exact_raw_parquet": paths.exact_raw_parquet,
            },
            rebuilders={
                "exact_raw_parquet": lambda: _assemble_doc_raw(
                    paths.doc_ownership_dir / "staging" / DOC_EXACT_STAGE,
                    ledger_path=paths.doc_ownership_dir / "refinitiv_doc_ownership_exact_api_ledger.sqlite3",
                    stage=DOC_EXACT_STAGE,
                )
            },
            expected_stage_manifest_path=paths.doc_exact_stage_manifest_path,
        )
    if stage == "doc_fallback_api":
        return audit_api_stage(
            stage_name=DOC_FALLBACK_STAGE,
            ledger_path=paths.doc_ownership_dir / "refinitiv_doc_ownership_fallback_api_ledger.sqlite3",
            staging_dir=paths.doc_ownership_dir / "staging" / DOC_FALLBACK_STAGE,
            output_artifacts={
                "fallback_requests_parquet": paths.fallback_requests_parquet,
                "fallback_raw_parquet": paths.fallback_raw_parquet,
            },
            rebuilders={
                "fallback_raw_parquet": lambda: _assemble_doc_raw(
                    paths.doc_ownership_dir / "staging" / DOC_FALLBACK_STAGE,
                    ledger_path=paths.doc_ownership_dir / "refinitiv_doc_ownership_fallback_api_ledger.sqlite3",
                    stage=DOC_FALLBACK_STAGE,
                )
            },
            expected_stage_manifest_path=paths.doc_fallback_stage_manifest_path,
        )

    issues = []
    for label, path in _stage_output_paths(paths, stage).items():
        if not path.exists():
            issues.append(
                AuditIssue(
                    severity="high",
                    code="missing_output_artifact",
                    message=f"{label} not found: {path}",
                    details=None,
                )
            )
    return StageAuditResult(
        stage_name=stage,
        passed=not issues,
        issues=tuple(issues),
        metrics={"output_artifacts": {key: str(value) for key, value in _stage_output_paths(paths, stage).items()}},
    )


def _stage_manifest_stage_name(stage: str) -> str:
    mapping = {
        "lookup_api": LOOKUP_STAGE,
        "ownership_api": OWNERSHIP_UNIVERSE_STAGE,
        "analyst_actuals_api": ANALYST_ACTUALS_STAGE,
        "analyst_estimates_api": ANALYST_ESTIMATES_STAGE,
        "doc_exact_api": DOC_EXACT_STAGE,
        "doc_fallback_api": DOC_FALLBACK_STAGE,
    }
    try:
        return mapping[stage]
    except KeyError as exc:
        raise ValueError(f"Stage does not have a canonical API stage manifest: {stage}") from exc


def _write_stage_manifest_from_existing_artifacts(
    stage: str,
    paths: RunPaths,
    *,
    recover_mode: str,
) -> dict[str, Any]:
    _ensure_required_inputs(paths, stage)
    audit_result = _audit_stage(stage, paths)
    if not audit_result.passed:
        raise RuntimeError(
            "stage manifest recovery audit failed for "
            f"{stage}: {audit_result.to_dict()}"
        )

    sentinels = _stage_resume_sentinels(paths, stage)
    if len(sentinels) != 3:
        raise ValueError(f"Stage does not define API sentinels for manifest recovery: {stage}")
    staging_dir, ledger_path, request_log_path = sentinels
    if not request_log_path.exists():
        raise FileNotFoundError(f"request_log_path not found: {request_log_path}")

    manifest_outputs = _stage_manifest_outputs(paths, stage)
    if len(manifest_outputs) != 1:
        raise ValueError(f"Stage does not define exactly one manifest output path: {stage}")
    manifest_path = manifest_outputs[0]

    write_stage_completion_manifest(
        stage_name=_stage_manifest_stage_name(stage),
        manifest_path=manifest_path,
        input_artifacts=_stage_input_paths(paths, stage),
        output_artifacts=_stage_output_paths(paths, stage),
        ledger_path=ledger_path,
        request_log_path=request_log_path,
        staging_dir=staging_dir,
        audit_result=audit_result,
        summary={
            "recovered_from_existing_artifacts": True,
            "recovered_via": "recover_mode_stage_manifest_only",
            "source_stage": stage,
            "run_session_ids": audit_result.metrics.get("run_session_ids", []),
            "rebuild_row_counts": audit_result.metrics.get("rebuild_row_counts", {}),
        },
    )
    return {
        "mode": recover_mode,
        "stage": stage,
        "stage_manifest_path": str(manifest_path),
        "audit": audit_result.to_dict(),
    }


def _run_recovery(args: argparse.Namespace, paths: RunPaths) -> dict[str, Any]:
    paths.recovery_dir.mkdir(parents=True, exist_ok=True)
    if args.recover_mode == "lookup_unresolved":
        output_path = args.recovery_output_path or paths.recovery_dir / "lookup_unresolved.parquet"
        artifact = build_lookup_unresolved_recovery_artifact(
            resolution_parquet_path=paths.resolution_parquet,
            output_path=output_path,
        )
    elif args.recover_mode == "ownership_unresolved":
        output_path = args.recovery_output_path or paths.recovery_dir / "ownership_unresolved.parquet"
        artifact = build_ownership_unresolved_recovery_artifact(
            row_summary_parquet_path=paths.ownership_row_summary_parquet,
            output_path=output_path,
        )
    elif args.recover_mode == "doc_exact_unresolved":
        output_path = args.recovery_output_path or paths.recovery_dir / "doc_exact_unresolved.parquet"
        artifact = build_doc_unresolved_recovery_artifact(
            recovery_mode="doc_exact_unresolved",
            requests_parquet_path=paths.exact_requests_parquet,
            raw_parquet_path=paths.exact_raw_parquet,
            output_path=output_path,
        )
    elif args.recover_mode == "doc_fallback_unresolved":
        output_path = args.recovery_output_path or paths.recovery_dir / "doc_fallback_unresolved.parquet"
        artifact = build_doc_unresolved_recovery_artifact(
            recovery_mode="doc_fallback_unresolved",
            requests_parquet_path=paths.fallback_requests_parquet,
            raw_parquet_path=paths.fallback_raw_parquet,
            output_path=output_path,
        )
    elif args.recover_mode == "resolution_only":
        _ensure_required_inputs(paths, "resolution")
        if args.stage_manifest_required:
            _ensure_required_stage_manifests(paths, "resolution")
        artifacts = _run_stage("resolution", args, paths)
        return {"mode": "resolution_only", "artifacts": {key: str(value) for key, value in artifacts.items()}}
    elif args.recover_mode == "doc_finalize_only":
        _ensure_required_inputs(paths, "doc_finalize")
        if args.stage_manifest_required:
            _ensure_required_stage_manifests(paths, "doc_finalize")
        artifacts = _run_stage("doc_finalize", args, paths)
        return {"mode": "doc_finalize_only", "artifacts": {key: str(value) for key, value in artifacts.items()}}
    elif args.recover_mode in STAGE_MANIFEST_RECOVERY_MODES:
        return _write_stage_manifest_from_existing_artifacts(
            STAGE_MANIFEST_RECOVERY_MODES[args.recover_mode],
            paths,
            recover_mode=args.recover_mode,
        )
    else:
        raise ValueError(f"Unsupported recover mode: {args.recover_mode}")

    recovery_manifest_path = output_path.with_suffix(".json")
    write_recovery_manifest(manifest_path=recovery_manifest_path, artifact=artifact)
    return {
        "mode": args.recover_mode,
        "artifact": artifact.to_dict(),
        "recovery_manifest_path": str(recovery_manifest_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = _resolve_run_paths(
        args.run_root,
        reviewed_ticker_allowlist_path=args.reviewed_ticker_allowlist_path,
    )
    selected_stages = tuple(args.selected_stages)

    if args.recover_mode is not None:
        recovery_result = _run_recovery(args, paths)
        manifest_path = _write_manifest(
            mode="recover",
            args=args,
            paths=paths,
            selected_stages=selected_stages,
            recovery_result=recovery_result,
        )
        print(json.dumps(recovery_result, indent=2))
        print(f"manifest_path: {manifest_path}")
        return 0

    if not args.audit_only and args.api_stage_mode != "finalize_only":
        _ensure_lseg_available()
    _ensure_required_inputs(paths, args.stage_start)
    if not args.resume and not args.audit_only and args.api_stage_mode != "finalize_only":
        _ensure_clean_start(paths, selected_stages)

    print(json.dumps(_artifact_map(paths, selected_stages), indent=2))

    if args.audit_only:
        audit_results: dict[str, StageAuditResult] = {}
        for stage in selected_stages:
            _ensure_required_inputs(paths, stage)
            if args.stage_manifest_required:
                _ensure_required_stage_manifests(paths, stage)
            result = _audit_stage(stage, paths)
            audit_results[stage] = result
            print(json.dumps({"stage": stage, "audit": result.to_dict()}, indent=2))
            if not result.passed:
                raise RuntimeError(f"audit failed for stage {stage}: {result.to_dict()}")
        manifest_path = _write_manifest(
            mode="audit",
            args=args,
            paths=paths,
            selected_stages=selected_stages,
            audit_results=audit_results,
        )
        print(f"manifest_path: {manifest_path}")
        return 0

    generated_artifacts: dict[str, dict[str, Path]] = {}
    completed_stages: set[str] = set()
    for stage in selected_stages:
        _ensure_required_inputs(paths, stage)
        if args.stage_manifest_required:
            _ensure_required_stage_manifests(paths, stage, completed_stages=completed_stages)
        print(f"running_stage: {stage}")
        result = _run_stage(stage, args, paths)
        generated_artifacts[stage] = result
        completed_stages.add(stage)
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
        mode="run",
        args=args,
        paths=paths,
        selected_stages=selected_stages,
        generated_artifacts=generated_artifacts,
    )
    print(f"manifest_path: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

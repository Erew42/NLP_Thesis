from __future__ import annotations

import calendar
import datetime as dt
import json
from pathlib import Path
from typing import Any, Callable

import polars as pl

from thesis_pkg.pipelines.refinitiv.lseg_api_common import (
    candidate_output_path as _candidate_output_path,
    promote_candidate_output as _promote_candidate_output,
    retry_delay_seconds as _retry_delay_seconds,
    standardize_field_frame as _standardize_field_frame,
    write_parquet_atomic as _write_parquet_atomic,
)
from thesis_pkg.pipelines.refinitiv.lseg_api_execution import run_api_batches
from thesis_pkg.pipelines.refinitiv.lseg_batching import (
    IntervalBatchPlan,
    IntervalBatchPlannerConfig,
    RequestItem,
    batch_items,
    batch_plan_fingerprint,
    build_batch_definition,
    plan_interval_batches,
    request_signature,
    stable_hash_id,
)
from thesis_pkg.pipelines.refinitiv.lseg_ledger import LsegResumeCompatibilityError, RequestLedger
from thesis_pkg.pipelines.refinitiv.lseg_stage_audit import (
    audit_api_stage,
    default_stage_fetch_manifest_path,
    default_stage_manifest_path,
    resolve_stage_fetch_metadata,
    write_stage_completion_manifest,
    write_stage_fetch_manifest,
)
from thesis_pkg.pipelines.refinitiv.doc_ownership import (
    DOC_OWNERSHIP_EXACT_STAGE,
    DOC_OWNERSHIP_FALLBACK_STAGE,
    DOC_OWNERSHIP_RAW_COLUMNS,
    DOC_OWNERSHIP_REQUEST_COLUMNS,
    _build_fallback_request_df,
    _cast_df_to_schema as _cast_doc_df_to_schema,
    _doc_ownership_raw_schema,
    _doc_ownership_request_schema,
    _empty_df,
    _is_institutional_category,
    _normalize_category,
    _normalize_date_value,
    _normalize_float_value,
    build_refinitiv_lm2011_doc_ownership_requests,
)
from thesis_pkg.pipelines.refinitiv_bridge_pipeline import (
    OWNERSHIP_UNIVERSE_HANDOFF_COLUMNS,
    OWNERSHIP_UNIVERSE_RESULTS_COLUMNS,
    OWNERSHIP_UNIVERSE_ROW_SUMMARY_COLUMNS,
    _cast_df_to_schema,
    _normalize_lookup_text,
    _ownership_universe_handoff_schema,
    _ownership_universe_results_schema,
    _ownership_universe_row_summary_schema,
    build_refinitiv_ownership_universe_row_summary,
)


OWNERSHIP_UNIVERSE_STAGE = "ownership_universe"
DOC_EXACT_STAGE = "doc_ownership_exact"
DOC_FALLBACK_STAGE = "doc_ownership_fallback"

OWNERSHIP_UNIVERSE_INTERVAL_BATCH_PLANNER_VERSION = "ownership_universe_interval_batching_v1"
_OWNERSHIP_INTERVAL_SIGNATURE_EXCLUDED_PARAMETER_KEYS = ("SDate", "EDate")

OWNERSHIP_UNIVERSE_DEFAULT_ROW_DENSITY_ROWS_PER_DAY = 1.0 / 91.0
OWNERSHIP_UNIVERSE_DEFAULT_MAX_EXTRA_ROWS_ABS = 120.0
OWNERSHIP_UNIVERSE_DEFAULT_MAX_EXTRA_ROWS_RATIO = 0.25

OWNERSHIP_UNIVERSE_FIELDS: tuple[str, ...] = (
    "TR.CategoryOwnershipPct.Date",
    "TR.CategoryOwnershipPct",
    "TR.InstrStatTypeValue",
)
DOC_EXACT_FIELDS: tuple[str, ...] = (
    "TR.CategoryOwnershipPct.Date",
    "TR.CategoryOwnershipPct",
    "TR.InstrStatTypeValue",
)
DOC_FALLBACK_FIELDS: tuple[str, ...] = (
    "TR.CategoryOwnershipPct.Date",
    "TR.CategoryOwnershipPct",
    "TR.InstrStatTypeValue",
)
OWNERSHIP_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "TR.CategoryOwnershipPct.Date": ("Date",),
    "TR.CategoryOwnershipPct": ("Category Percent Of Tr", "Category Percent Of Traded Shares"),
    "TR.InstrStatTypeValue": ("Investor Statistics Category Value",),
}
API_STAGE_MODES: frozenset[str] = frozenset({"full", "fetch_only", "finalize_only"})
DOC_EXACT_BATCH_PLANNER_VERSION = "doc_ownership_exact_batching_v1"
DOC_FALLBACK_BATCH_PLANNER_VERSION = "doc_ownership_fallback_batching_v1"


def run_refinitiv_step1_ownership_universe_api_pipeline(
    *,
    handoff_parquet_path: Path | str,
    output_dir: Path | str,
    provider: Any | None = None,
    ledger_path: Path | str | None = None,
    request_log_path: Path | str | None = None,
    max_batch_size: int = 10,
    max_batch_items: int | None = None,
    max_extra_rows_abs: float | None = None,
    max_extra_rows_ratio: float | None = None,
    max_union_span_days: int | None = None,
    row_density_rows_per_day: float | None = None,
    min_seconds_between_requests: float = 2.0,
    min_seconds_between_request_starts_total: float | None = None,
    max_attempts: int = 4,
    max_workers: int = 1,
    provider_session_name: str = "desktop.workspace",
    provider_config_name: str | None = None,
    provider_timeout_seconds: float | None = None,
    preflight_probe: bool = False,
    stage_manifest_path: Path | str | None = None,
    fetch_manifest_path: Path | str | None = None,
    api_stage_mode: str = "full",
) -> dict[str, Path]:
    api_stage_mode = _normalize_api_stage_mode(api_stage_mode)
    handoff_parquet_path = Path(handoff_parquet_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = (
        Path(ledger_path)
        if ledger_path is not None
        else output_dir / "refinitiv_ownership_universe_api_ledger.sqlite3"
    )
    request_log_path = (
        Path(request_log_path)
        if request_log_path is not None
        else output_dir / "refinitiv_ownership_universe_api_requests.jsonl"
    )
    manifest_path = (
        Path(stage_manifest_path)
        if stage_manifest_path is not None
        else default_stage_manifest_path(output_dir, OWNERSHIP_UNIVERSE_STAGE)
    )
    fetch_manifest_path = (
        Path(fetch_manifest_path)
        if fetch_manifest_path is not None
        else default_stage_fetch_manifest_path(output_dir, OWNERSHIP_UNIVERSE_STAGE)
    )
    handoff_df = _cast_df_to_schema(
        pl.read_parquet(handoff_parquet_path).select(OWNERSHIP_UNIVERSE_HANDOFF_COLUMNS),
        _ownership_universe_handoff_schema(),
    ).select(OWNERSHIP_UNIVERSE_HANDOFF_COLUMNS)

    items = _build_ownership_universe_items(handoff_df)
    planner_config = _resolve_interval_batch_config(
        max_batch_size=max_batch_size,
        max_batch_items=max_batch_items,
        max_extra_rows_abs=max_extra_rows_abs,
        max_extra_rows_ratio=max_extra_rows_ratio,
        max_union_span_days=max_union_span_days,
        row_density_rows_per_day=row_density_rows_per_day,
        default_row_density_rows_per_day=OWNERSHIP_UNIVERSE_DEFAULT_ROW_DENSITY_ROWS_PER_DAY,
        default_max_extra_rows_abs=OWNERSHIP_UNIVERSE_DEFAULT_MAX_EXTRA_ROWS_ABS,
        default_max_extra_rows_ratio=OWNERSHIP_UNIVERSE_DEFAULT_MAX_EXTRA_ROWS_RATIO,
    )
    interval_plan = _plan_ownership_universe_batches(items, config=planner_config)
    result = {
        "refinitiv_ownership_universe_api_ledger_sqlite3": ledger_path,
        "refinitiv_ownership_universe_api_requests_jsonl": request_log_path,
        "refinitiv_ownership_universe_fetch_manifest_json": fetch_manifest_path,
    }
    stage_run = None
    if api_stage_mode != "finalize_only":
        try:
            stage_run = run_api_batches(
                stage=OWNERSHIP_UNIVERSE_STAGE,
                items=items,
                output_dir=output_dir,
                ledger_path=ledger_path,
                request_log_path=request_log_path,
                provider=provider,
                provider_session_name=provider_session_name,
                provider_config_name=provider_config_name,
                provider_timeout_seconds=provider_timeout_seconds,
                preflight_probe=preflight_probe,
                max_batch_size=max_batch_size,
                min_seconds_between_requests=min_seconds_between_requests,
                min_seconds_between_request_starts_total=min_seconds_between_request_starts_total,
                max_attempts=max_attempts,
                response_normalizer=_normalize_ownership_universe_batch_response,
                lookup_normalizer=_normalize_lookup_text,
                split_after_attempt=2,
                retry_delay_seconds_fn=_retry_delay_seconds,
                max_workers=max_workers,
                planned_batches=list(interval_plan.batches),
                batch_plan_fingerprint=interval_plan.fingerprint,
                planned_batch_metrics=interval_plan.batch_metrics_by_id,
                resume_compatibility_metadata=_interval_resume_metadata(
                    planner_version=OWNERSHIP_UNIVERSE_INTERVAL_BATCH_PLANNER_VERSION,
                    interval_plan=interval_plan,
                ),
            )
        except LsegResumeCompatibilityError as exc:
            _raise_stage_resume_compatibility_error(
                stage_name=OWNERSHIP_UNIVERSE_STAGE,
                output_dir=output_dir,
                fetch_manifest_path=fetch_manifest_path,
                manifest_path=manifest_path,
                exc=exc,
            )
    else:
        _ensure_finalize_state(
            ledger_path=ledger_path,
            request_log_path=request_log_path,
            stage_name=OWNERSHIP_UNIVERSE_STAGE,
        )

    staging_dir = output_dir / "staging" / OWNERSHIP_UNIVERSE_STAGE if stage_run is None else stage_run.staging_dir
    metadata = (
        resolve_stage_fetch_metadata(
            stage_name=OWNERSHIP_UNIVERSE_STAGE,
            ledger_path=ledger_path,
            fetch_manifest_path=fetch_manifest_path,
            stage_manifest_path=manifest_path,
            current_batching_config=planner_config.to_serializable_dict(),
            current_batch_plan_fingerprint=interval_plan.fingerprint,
        )
        if api_stage_mode == "finalize_only"
        else None
    )
    _write_api_fetch_manifest(
        stage_name=OWNERSHIP_UNIVERSE_STAGE,
        manifest_path=fetch_manifest_path,
        staging_dir=staging_dir,
        ledger_path=ledger_path,
        request_log_path=request_log_path,
        request_item_count=len(items) if metadata is None else metadata.request_item_count,
        batch_count=len(interval_plan.batches) if metadata is None else metadata.batch_count,
        batch_plan_fingerprint=interval_plan.fingerprint if metadata is None else metadata.batch_plan_fingerprint,
        batching_config=planner_config.to_serializable_dict() if metadata is None else metadata.batching_config,
        metadata_source="current_run" if metadata is None else metadata.metadata_source,
        cli_batching_args_ignored=False if metadata is None else metadata.cli_batching_args_ignored,
        run_session_ids=_stage_run_session_ids(ledger_path) if metadata is None else metadata.run_session_ids,
    )
    if api_stage_mode == "fetch_only":
        return result

    results_path = output_dir / "refinitiv_ownership_universe_results.parquet"
    row_summary_path = output_dir / "refinitiv_ownership_universe_row_summary.parquet"
    result.update(
        _finalize_ownership_universe_stage(
            handoff_df=handoff_df,
            handoff_parquet_path=handoff_parquet_path,
            ledger_path=ledger_path,
            request_log_path=request_log_path,
            staging_dir=staging_dir,
            results_path=results_path,
            row_summary_path=row_summary_path,
            manifest_path=manifest_path,
            request_item_count=len(items) if metadata is None else _coalesce_count(metadata.request_item_count, len(items)),
            planned_batch_count=len(interval_plan.batches) if metadata is None else metadata.batch_count,
            batch_plan_fingerprint=interval_plan.fingerprint if metadata is None else metadata.batch_plan_fingerprint,
            batching_config=planner_config.to_serializable_dict() if metadata is None else metadata.batching_config,
            metadata_source="current_run" if metadata is None else metadata.metadata_source,
            cli_batching_args_ignored=False if metadata is None else metadata.cli_batching_args_ignored,
            run_session_id=None if stage_run is None else stage_run.run_session_id,
            verify_rebuilders=False,
        )
    )
    return result


def run_refinitiv_lm2011_doc_ownership_exact_api_pipeline(
    *,
    doc_filing_artifact_path: Path | str,
    authority_decisions_artifact_path: Path | str,
    authority_exceptions_artifact_path: Path | str,
    output_dir: Path | str,
    request_min_date: dt.date | None = None,
    request_max_date: dt.date | None = None,
    provider: Any | None = None,
    ledger_path: Path | str | None = None,
    request_log_path: Path | str | None = None,
    max_batch_size: int = 15,
    min_seconds_between_requests: float = 2.0,
    min_seconds_between_request_starts_total: float | None = None,
    max_attempts: int = 4,
    max_workers: int = 1,
    provider_session_name: str = "desktop.workspace",
    provider_config_name: str | None = None,
    provider_timeout_seconds: float | None = None,
    preflight_probe: bool = False,
    stage_manifest_path: Path | str | None = None,
    fetch_manifest_path: Path | str | None = None,
    api_stage_mode: str = "full",
) -> dict[str, Path]:
    api_stage_mode = _normalize_api_stage_mode(api_stage_mode)
    from thesis_pkg.pipelines.refinitiv.doc_ownership import (
        _read_authority_decisions_artifact,
        _read_authority_exceptions_artifact,
        _read_doc_filing_artifact,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = (
        Path(ledger_path) if ledger_path is not None else output_dir / "refinitiv_doc_ownership_exact_api_ledger.sqlite3"
    )
    request_log_path = (
        Path(request_log_path)
        if request_log_path is not None
        else output_dir / "refinitiv_doc_ownership_exact_api_requests.jsonl"
    )
    manifest_path = (
        Path(stage_manifest_path)
        if stage_manifest_path is not None
        else default_stage_manifest_path(output_dir, DOC_EXACT_STAGE)
    )
    fetch_manifest_path = (
        Path(fetch_manifest_path)
        if fetch_manifest_path is not None
        else default_stage_fetch_manifest_path(output_dir, DOC_EXACT_STAGE)
    )

    request_df = build_refinitiv_lm2011_doc_ownership_requests(
        _read_doc_filing_artifact(doc_filing_artifact_path),
        _read_authority_decisions_artifact(authority_decisions_artifact_path),
        _read_authority_exceptions_artifact(authority_exceptions_artifact_path),
        request_min_date=request_min_date,
        request_max_date=request_max_date,
    )
    items = _build_doc_ownership_exact_items(request_df)
    batching_config = {"max_batch_size": max_batch_size}
    planned_batches = batch_items(items, max_batch_size=max_batch_size, unique_instrument_limit=True)
    current_batch_plan_fingerprint = batch_plan_fingerprint(
        planner_version=DOC_EXACT_BATCH_PLANNER_VERSION,
        batching_config=batching_config,
        planned_batches=planned_batches,
    )
    result = {
        "refinitiv_doc_ownership_exact_api_ledger_sqlite3": ledger_path,
        "refinitiv_doc_ownership_exact_api_requests_jsonl": request_log_path,
        "refinitiv_doc_ownership_exact_fetch_manifest_json": fetch_manifest_path,
    }
    stage_run = None
    if api_stage_mode != "finalize_only":
        try:
            stage_run = run_api_batches(
                stage=DOC_EXACT_STAGE,
                items=items,
                output_dir=output_dir,
                ledger_path=ledger_path,
                request_log_path=request_log_path,
                provider=provider,
                provider_session_name=provider_session_name,
                provider_config_name=provider_config_name,
                provider_timeout_seconds=provider_timeout_seconds,
                preflight_probe=preflight_probe,
                max_batch_size=max_batch_size,
                min_seconds_between_requests=min_seconds_between_requests,
                min_seconds_between_request_starts_total=min_seconds_between_request_starts_total,
                max_attempts=max_attempts,
                response_normalizer=_normalize_doc_exact_batch_response,
                lookup_normalizer=_normalize_lookup_text,
                split_after_attempt=2,
                retry_delay_seconds_fn=_retry_delay_seconds,
                max_workers=max_workers,
                planned_batches=planned_batches,
                batch_plan_fingerprint=current_batch_plan_fingerprint,
                resume_compatibility_metadata=_simple_resume_metadata(
                    planner_version=DOC_EXACT_BATCH_PLANNER_VERSION,
                    batching_config=batching_config,
                    planned_batch_count=len(planned_batches),
                ),
            )
        except LsegResumeCompatibilityError as exc:
            _raise_stage_resume_compatibility_error(
                stage_name=DOC_EXACT_STAGE,
                output_dir=output_dir,
                fetch_manifest_path=fetch_manifest_path,
                manifest_path=manifest_path,
                exc=exc,
            )
    else:
        _ensure_finalize_state(
            ledger_path=ledger_path,
            request_log_path=request_log_path,
            stage_name=DOC_EXACT_STAGE,
        )

    staging_dir = output_dir / "staging" / DOC_EXACT_STAGE if stage_run is None else stage_run.staging_dir
    metadata = (
        resolve_stage_fetch_metadata(
            stage_name=DOC_EXACT_STAGE,
            ledger_path=ledger_path,
            fetch_manifest_path=fetch_manifest_path,
            stage_manifest_path=manifest_path,
            current_batching_config=batching_config,
            current_batch_plan_fingerprint=current_batch_plan_fingerprint,
        )
        if api_stage_mode == "finalize_only"
        else None
    )
    _write_api_fetch_manifest(
        stage_name=DOC_EXACT_STAGE,
        manifest_path=fetch_manifest_path,
        staging_dir=staging_dir,
        ledger_path=ledger_path,
        request_log_path=request_log_path,
        request_item_count=len(items) if metadata is None else metadata.request_item_count,
        batch_count=len(planned_batches) if metadata is None else metadata.batch_count,
        batch_plan_fingerprint=current_batch_plan_fingerprint if metadata is None else metadata.batch_plan_fingerprint,
        batching_config=batching_config if metadata is None else metadata.batching_config,
        metadata_source="current_run" if metadata is None else metadata.metadata_source,
        cli_batching_args_ignored=False if metadata is None else metadata.cli_batching_args_ignored,
        run_session_ids=_stage_run_session_ids(ledger_path) if metadata is None else metadata.run_session_ids,
        summary={
            "request_row_count": int(request_df.height),
            "eligible_request_row_count": int(request_df.filter(pl.col("retrieval_eligible").fill_null(False)).height),
        },
    )
    if api_stage_mode == "fetch_only":
        return result

    requests_path = output_dir / "refinitiv_lm2011_doc_ownership_exact_requests.parquet"
    exact_raw_path = output_dir / "refinitiv_lm2011_doc_ownership_exact_raw.parquet"
    result.update(
        _finalize_doc_stage(
            stage_name=DOC_EXACT_STAGE,
            input_artifacts={
                "doc_filing_artifact_parquet": Path(doc_filing_artifact_path),
                "authority_decisions_parquet": Path(authority_decisions_artifact_path),
                "authority_exceptions_parquet": Path(authority_exceptions_artifact_path),
            },
            request_df=request_df,
            requests_path=requests_path,
            raw_path=exact_raw_path,
            ledger_path=ledger_path,
            request_log_path=request_log_path,
            staging_dir=staging_dir,
            manifest_path=manifest_path,
            raw_df=_assemble_doc_raw(
                staging_dir,
                ledger_path=ledger_path,
                stage=DOC_EXACT_STAGE,
            ),
            raw_rebuilder=lambda: _assemble_doc_raw(
                staging_dir,
                ledger_path=ledger_path,
                stage=DOC_EXACT_STAGE,
            ),
            request_row_label="request_row_count",
            request_item_count=len(items) if metadata is None else _coalesce_count(metadata.request_item_count, len(items)),
            planned_batch_count=len(planned_batches) if metadata is None else metadata.batch_count,
            batch_plan_fingerprint=current_batch_plan_fingerprint if metadata is None else metadata.batch_plan_fingerprint,
            batching_config=batching_config if metadata is None else metadata.batching_config,
            metadata_source="current_run" if metadata is None else metadata.metadata_source,
            cli_batching_args_ignored=False if metadata is None else metadata.cli_batching_args_ignored,
            run_session_id=None if stage_run is None else stage_run.run_session_id,
            verify_rebuilders=False,
        )
    )
    return result


def run_refinitiv_lm2011_doc_ownership_fallback_api_pipeline(
    *,
    output_dir: Path | str,
    provider: Any | None = None,
    ledger_path: Path | str | None = None,
    request_log_path: Path | str | None = None,
    max_batch_size: int = 5,
    min_seconds_between_requests: float = 2.0,
    min_seconds_between_request_starts_total: float | None = None,
    max_attempts: int = 4,
    max_workers: int = 1,
    provider_session_name: str = "desktop.workspace",
    provider_config_name: str | None = None,
    provider_timeout_seconds: float | None = None,
    preflight_probe: bool = False,
    stage_manifest_path: Path | str | None = None,
    fetch_manifest_path: Path | str | None = None,
    api_stage_mode: str = "full",
) -> dict[str, Path]:
    api_stage_mode = _normalize_api_stage_mode(api_stage_mode)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = (
        Path(ledger_path)
        if ledger_path is not None
        else output_dir / "refinitiv_doc_ownership_fallback_api_ledger.sqlite3"
    )
    request_log_path = (
        Path(request_log_path)
        if request_log_path is not None
        else output_dir / "refinitiv_doc_ownership_fallback_api_requests.jsonl"
    )
    manifest_path = (
        Path(stage_manifest_path)
        if stage_manifest_path is not None
        else default_stage_manifest_path(output_dir, DOC_FALLBACK_STAGE)
    )
    fetch_manifest_path = (
        Path(fetch_manifest_path)
        if fetch_manifest_path is not None
        else default_stage_fetch_manifest_path(output_dir, DOC_FALLBACK_STAGE)
    )

    exact_requests_path = output_dir / "refinitiv_lm2011_doc_ownership_exact_requests.parquet"
    exact_raw_path = output_dir / "refinitiv_lm2011_doc_ownership_exact_raw.parquet"
    if not exact_requests_path.exists():
        raise FileNotFoundError(f"exact requests parquet not found: {exact_requests_path}")
    if not exact_raw_path.exists():
        raise FileNotFoundError(f"exact raw parquet not found: {exact_raw_path}")

    request_df = _cast_doc_df_to_schema(
        pl.read_parquet(exact_requests_path).select(DOC_OWNERSHIP_REQUEST_COLUMNS),
        _doc_ownership_request_schema(),
    ).select(DOC_OWNERSHIP_REQUEST_COLUMNS)
    exact_raw_df = _cast_doc_df_to_schema(
        pl.read_parquet(exact_raw_path).select(DOC_OWNERSHIP_RAW_COLUMNS),
        _doc_ownership_raw_schema(),
    ).select(DOC_OWNERSHIP_RAW_COLUMNS)
    fallback_request_df = _build_fallback_request_df(request_df, exact_raw_df)
    items = _build_doc_ownership_fallback_items(fallback_request_df)
    batching_config = {"max_batch_size": max_batch_size}
    planned_batches = batch_items(items, max_batch_size=max_batch_size, unique_instrument_limit=True)
    current_batch_plan_fingerprint = batch_plan_fingerprint(
        planner_version=DOC_FALLBACK_BATCH_PLANNER_VERSION,
        batching_config=batching_config,
        planned_batches=planned_batches,
    )
    result = {
        "refinitiv_lm2011_doc_ownership_exact_raw_parquet": exact_raw_path,
        "refinitiv_doc_ownership_fallback_api_ledger_sqlite3": ledger_path,
        "refinitiv_doc_ownership_fallback_api_requests_jsonl": request_log_path,
        "refinitiv_doc_ownership_fallback_fetch_manifest_json": fetch_manifest_path,
    }
    stage_run = None
    if api_stage_mode != "finalize_only":
        try:
            stage_run = run_api_batches(
                stage=DOC_FALLBACK_STAGE,
                items=items,
                output_dir=output_dir,
                ledger_path=ledger_path,
                request_log_path=request_log_path,
                provider=provider,
                provider_session_name=provider_session_name,
                provider_config_name=provider_config_name,
                provider_timeout_seconds=provider_timeout_seconds,
                preflight_probe=preflight_probe,
                max_batch_size=max_batch_size,
                min_seconds_between_requests=min_seconds_between_requests,
                min_seconds_between_request_starts_total=min_seconds_between_request_starts_total,
                max_attempts=max_attempts,
                response_normalizer=_normalize_doc_fallback_batch_response,
                lookup_normalizer=_normalize_lookup_text,
                split_after_attempt=1,
                retry_delay_seconds_fn=_retry_delay_seconds,
                max_workers=max_workers,
                planned_batches=planned_batches,
                batch_plan_fingerprint=current_batch_plan_fingerprint,
                resume_compatibility_metadata=_simple_resume_metadata(
                    planner_version=DOC_FALLBACK_BATCH_PLANNER_VERSION,
                    batching_config=batching_config,
                    planned_batch_count=len(planned_batches),
                ),
            )
        except LsegResumeCompatibilityError as exc:
            _raise_stage_resume_compatibility_error(
                stage_name=DOC_FALLBACK_STAGE,
                output_dir=output_dir,
                fetch_manifest_path=fetch_manifest_path,
                manifest_path=manifest_path,
                exc=exc,
            )
    else:
        _ensure_finalize_state(
            ledger_path=ledger_path,
            request_log_path=request_log_path,
            stage_name=DOC_FALLBACK_STAGE,
        )

    staging_dir = output_dir / "staging" / DOC_FALLBACK_STAGE if stage_run is None else stage_run.staging_dir
    metadata = (
        resolve_stage_fetch_metadata(
            stage_name=DOC_FALLBACK_STAGE,
            ledger_path=ledger_path,
            fetch_manifest_path=fetch_manifest_path,
            stage_manifest_path=manifest_path,
            current_batching_config=batching_config,
            current_batch_plan_fingerprint=current_batch_plan_fingerprint,
        )
        if api_stage_mode == "finalize_only"
        else None
    )
    _write_api_fetch_manifest(
        stage_name=DOC_FALLBACK_STAGE,
        manifest_path=fetch_manifest_path,
        staging_dir=staging_dir,
        ledger_path=ledger_path,
        request_log_path=request_log_path,
        request_item_count=len(items) if metadata is None else metadata.request_item_count,
        batch_count=len(planned_batches) if metadata is None else metadata.batch_count,
        batch_plan_fingerprint=current_batch_plan_fingerprint if metadata is None else metadata.batch_plan_fingerprint,
        batching_config=batching_config if metadata is None else metadata.batching_config,
        metadata_source="current_run" if metadata is None else metadata.metadata_source,
        cli_batching_args_ignored=False if metadata is None else metadata.cli_batching_args_ignored,
        run_session_ids=_stage_run_session_ids(ledger_path) if metadata is None else metadata.run_session_ids,
        summary={
            "fallback_request_row_count": int(fallback_request_df.height),
            "eligible_request_row_count": int(
                fallback_request_df.filter(pl.col("retrieval_eligible").fill_null(False)).height
            ),
        },
    )
    if api_stage_mode == "fetch_only":
        return result

    fallback_requests_path = output_dir / "refinitiv_lm2011_doc_ownership_fallback_requests.parquet"
    fallback_raw_path = output_dir / "refinitiv_lm2011_doc_ownership_fallback_raw.parquet"
    result.update(
        _finalize_doc_stage(
            stage_name=DOC_FALLBACK_STAGE,
            input_artifacts={
                "exact_requests_parquet": exact_requests_path,
                "exact_raw_parquet": exact_raw_path,
            },
            request_df=fallback_request_df,
            requests_path=fallback_requests_path,
            raw_path=fallback_raw_path,
            ledger_path=ledger_path,
            request_log_path=request_log_path,
            staging_dir=staging_dir,
            manifest_path=manifest_path,
            raw_df=_assemble_doc_raw(
                staging_dir,
                ledger_path=ledger_path,
                stage=DOC_FALLBACK_STAGE,
            ),
            raw_rebuilder=lambda: _assemble_doc_raw(
                staging_dir,
                ledger_path=ledger_path,
                stage=DOC_FALLBACK_STAGE,
            ),
            request_row_label="fallback_request_row_count",
            request_item_count=len(items) if metadata is None else _coalesce_count(metadata.request_item_count, len(items)),
            planned_batch_count=len(planned_batches) if metadata is None else metadata.batch_count,
            batch_plan_fingerprint=current_batch_plan_fingerprint if metadata is None else metadata.batch_plan_fingerprint,
            batching_config=batching_config if metadata is None else metadata.batching_config,
            metadata_source="current_run" if metadata is None else metadata.metadata_source,
            cli_batching_args_ignored=False if metadata is None else metadata.cli_batching_args_ignored,
            run_session_id=None if stage_run is None else stage_run.run_session_id,
            verify_rebuilders=False,
        )
    )
    return result


def _assemble_ownership_universe_results(
    staging_dir: Path,
    *,
    ledger_path: Path | str | None = None,
) -> pl.DataFrame:
    results_df = _scan_stage_outputs(
        stage=OWNERSHIP_UNIVERSE_STAGE,
        staging_dir=staging_dir,
        ledger_path=ledger_path,
        schema={name: dtype for name, dtype in _ownership_universe_results_schema().items()},
    ).collect()
    return (
        _cast_df_to_schema(results_df.select(OWNERSHIP_UNIVERSE_RESULTS_COLUMNS), _ownership_universe_results_schema())
        .select(OWNERSHIP_UNIVERSE_RESULTS_COLUMNS)
        .unique(
            subset=[
                "ownership_lookup_row_id",
                "returned_date",
                "returned_category",
                "returned_value",
                "returned_ric",
            ],
            maintain_order=True,
        )
    )


def _assemble_doc_raw(
    staging_dir: Path,
    *,
    ledger_path: Path | str | None = None,
    stage: str | None = None,
) -> pl.DataFrame:
    staging_paths = (
        _resolved_stage_output_paths(
            stage=stage,
            staging_dir=staging_dir,
            ledger_path=ledger_path,
        )
        if ledger_path is not None and stage is not None
        else sorted(staging_dir.glob("*.parquet"))
    )
    raw_df = (
        pl.concat([pl.scan_parquet(path) for path in staging_paths], how="vertical_relaxed").collect()
        if staging_paths
        else _empty_df(DOC_OWNERSHIP_RAW_COLUMNS, _doc_ownership_raw_schema())
    )
    return (
        _cast_doc_df_to_schema(raw_df.select(DOC_OWNERSHIP_RAW_COLUMNS), _doc_ownership_raw_schema())
        .select(DOC_OWNERSHIP_RAW_COLUMNS)
        .unique(
            subset=["doc_id", "request_stage", "response_date", "returned_category", "returned_value"],
            maintain_order=True,
        )
    )


def _finalize_ownership_universe_stage(
    *,
    handoff_df: pl.DataFrame,
    handoff_parquet_path: Path,
    ledger_path: Path,
    request_log_path: Path,
    staging_dir: Path,
    results_path: Path,
    row_summary_path: Path,
    manifest_path: Path,
    request_item_count: int,
    planned_batch_count: int | None,
    batch_plan_fingerprint: str | None,
    batching_config: dict[str, Any] | None,
    metadata_source: str,
    cli_batching_args_ignored: bool,
    run_session_id: str | None,
    verify_rebuilders: bool,
) -> dict[str, Path]:
    results_df = _assemble_ownership_universe_results(staging_dir, ledger_path=ledger_path)
    row_summary_df = _build_ownership_row_summary_df(handoff_df, results_df)
    results_candidate_path = _candidate_output_path(results_path)
    row_summary_candidate_path = _candidate_output_path(row_summary_path)
    _write_parquet_atomic(results_df, results_candidate_path)
    _write_parquet_atomic(row_summary_df, row_summary_candidate_path)
    audit_result = audit_api_stage(
        stage_name=OWNERSHIP_UNIVERSE_STAGE,
        ledger_path=ledger_path,
        staging_dir=staging_dir,
        output_artifacts={
            "ownership_results_parquet": results_candidate_path,
            "ownership_row_summary_parquet": row_summary_candidate_path,
        },
        declared_output_artifacts={
            "ownership_results_parquet": results_path,
            "ownership_row_summary_parquet": row_summary_path,
        },
        rebuilders={
            "ownership_results_parquet": lambda: _assemble_ownership_universe_results(
                staging_dir,
                ledger_path=ledger_path,
            ),
            "ownership_row_summary_parquet": lambda: _build_ownership_row_summary_df(
                handoff_df,
                _assemble_ownership_universe_results(
                    staging_dir,
                    ledger_path=ledger_path,
                ),
            ),
        },
        expected_stage_manifest_path=manifest_path,
        verify_rebuilders=verify_rebuilders,
    )
    if not audit_result.passed:
        raise RuntimeError(f"ownership universe stage audit failed: {audit_result.to_dict()}")
    _promote_candidate_output(results_candidate_path, results_path)
    _promote_candidate_output(row_summary_candidate_path, row_summary_path)

    write_stage_completion_manifest(
        stage_name=OWNERSHIP_UNIVERSE_STAGE,
        manifest_path=manifest_path,
        input_artifacts={"ownership_handoff_parquet": handoff_parquet_path},
        output_artifacts={
            "ownership_results_parquet": results_path,
            "ownership_row_summary_parquet": row_summary_path,
        },
        ledger_path=ledger_path,
        request_log_path=request_log_path,
        staging_dir=staging_dir,
        audit_result=audit_result,
        summary={
            "handoff_row_count": int(handoff_df.height),
            "request_item_count": request_item_count,
            "planned_batch_count": planned_batch_count,
            "batch_plan_fingerprint": batch_plan_fingerprint,
            "batching_config": batching_config,
            "results_row_count": int(results_df.height),
            "rows_with_results": int(row_summary_df.filter(pl.col("ownership_rows_returned").fill_null(0) > 0).height),
            "request_succeeded_count": _count_request_log_events(request_log_path, "request_succeeded"),
            "mixed_zero_positive_success_requeued_count": _count_request_log_events(
                request_log_path,
                "request_succeeded_mixed_zero_items_requeued",
            ),
            "request_unresolved_identifiers_treated_as_empty_count": _count_request_log_events(
                request_log_path,
                "request_unresolved_identifiers_treated_as_empty",
            ),
            "run_session_id": run_session_id,
            "run_session_ids": _stage_run_session_ids(ledger_path),
        },
        metadata_source=metadata_source,
        cli_batching_args_ignored=cli_batching_args_ignored,
    )
    return {
        "refinitiv_ownership_universe_results_parquet": results_path,
        "refinitiv_ownership_universe_row_summary_parquet": row_summary_path,
        "refinitiv_ownership_universe_stage_manifest_json": manifest_path,
    }


def _finalize_doc_stage(
    *,
    stage_name: str,
    input_artifacts: dict[str, Path],
    request_df: pl.DataFrame,
    requests_path: Path,
    raw_path: Path,
    ledger_path: Path,
    request_log_path: Path,
    staging_dir: Path,
    manifest_path: Path,
    raw_df: pl.DataFrame,
    raw_rebuilder: Callable[[], pl.DataFrame],
    request_row_label: str,
    request_item_count: int,
    planned_batch_count: int | None,
    batch_plan_fingerprint: str | None,
    batching_config: dict[str, Any] | None,
    metadata_source: str,
    cli_batching_args_ignored: bool,
    run_session_id: str | None,
    verify_rebuilders: bool,
) -> dict[str, Path]:
    requests_candidate_path = _candidate_output_path(requests_path)
    raw_candidate_path = _candidate_output_path(raw_path)
    _write_parquet_atomic(request_df, requests_candidate_path)
    _write_parquet_atomic(raw_df, raw_candidate_path)
    output_artifacts = (
        {"exact_requests_parquet": requests_candidate_path, "exact_raw_parquet": raw_candidate_path}
        if stage_name == DOC_EXACT_STAGE
        else {"fallback_requests_parquet": requests_candidate_path, "fallback_raw_parquet": raw_candidate_path}
    )
    declared_output_artifacts = (
        {"exact_requests_parquet": requests_path, "exact_raw_parquet": raw_path}
        if stage_name == DOC_EXACT_STAGE
        else {"fallback_requests_parquet": requests_path, "fallback_raw_parquet": raw_path}
    )
    raw_label = "exact_raw_parquet" if stage_name == DOC_EXACT_STAGE else "fallback_raw_parquet"
    audit_result = audit_api_stage(
        stage_name=stage_name,
        ledger_path=ledger_path,
        staging_dir=staging_dir,
        output_artifacts=output_artifacts,
        declared_output_artifacts=declared_output_artifacts,
        rebuilders={raw_label: raw_rebuilder},
        expected_stage_manifest_path=manifest_path,
        verify_rebuilders=verify_rebuilders,
    )
    if not audit_result.passed:
        raise RuntimeError(f"{stage_name} stage audit failed: {audit_result.to_dict()}")
    _promote_candidate_output(requests_candidate_path, requests_path)
    _promote_candidate_output(raw_candidate_path, raw_path)

    write_stage_completion_manifest(
        stage_name=stage_name,
        manifest_path=manifest_path,
        input_artifacts=input_artifacts,
        output_artifacts=declared_output_artifacts,
        ledger_path=ledger_path,
        request_log_path=request_log_path,
        staging_dir=staging_dir,
        audit_result=audit_result,
        summary={
            request_row_label: int(request_df.height),
            "eligible_request_row_count": int(request_df.filter(pl.col("retrieval_eligible").fill_null(False)).height),
            "request_item_count": request_item_count,
            "planned_batch_count": planned_batch_count,
            "batch_plan_fingerprint": batch_plan_fingerprint,
            "batching_config": batching_config,
            "raw_row_count": int(raw_df.height),
            "docs_with_raw_rows": int(raw_df.select(pl.col("doc_id").drop_nulls().n_unique()).item())
            if raw_df.height
            else 0,
            "run_session_id": run_session_id,
            "run_session_ids": _stage_run_session_ids(ledger_path),
        },
        metadata_source=metadata_source,
        cli_batching_args_ignored=cli_batching_args_ignored,
    )
    return (
        {
            "refinitiv_lm2011_doc_ownership_exact_requests_parquet": requests_path,
            "refinitiv_lm2011_doc_ownership_exact_raw_parquet": raw_path,
            "refinitiv_doc_ownership_exact_stage_manifest_json": manifest_path,
        }
        if stage_name == DOC_EXACT_STAGE
        else {
            "refinitiv_lm2011_doc_ownership_fallback_requests_parquet": requests_path,
            "refinitiv_lm2011_doc_ownership_fallback_raw_parquet": raw_path,
            "refinitiv_doc_ownership_fallback_stage_manifest_json": manifest_path,
        }
    )


def _build_ownership_row_summary_df(
    handoff_df: pl.DataFrame,
    results_df: pl.DataFrame,
) -> pl.DataFrame:
    if handoff_df.height == 0:
        return pl.DataFrame(schema=_ownership_universe_row_summary_schema()).select(
            OWNERSHIP_UNIVERSE_ROW_SUMMARY_COLUMNS
        )
    if results_df.height == 0:
        return build_refinitiv_ownership_universe_row_summary(handoff_df, results_df)
    key_columns = list(OWNERSHIP_UNIVERSE_HANDOFF_COLUMNS)
    base_lf = handoff_df.select(key_columns).lazy()
    agg_lf = results_df.lazy().group_by(key_columns).agg(
        pl.len().alias("ownership_rows_returned"),
        pl.col("returned_date").drop_nulls().min().alias("ownership_first_date"),
        pl.col("returned_date").drop_nulls().max().alias("ownership_last_date"),
        pl.col("returned_category").drop_nulls().n_unique().cast(pl.Int64).alias("ownership_distinct_categories"),
        pl.col("returned_value").drop_nulls().len().cast(pl.Int64).alias("ownership_nonnull_value_count"),
        pl.col("returned_ric").drop_nulls().n_unique().cast(pl.Int64).alias("ownership_returned_ric_nunique"),
    )
    summary_df = (
        base_lf.join(agg_lf, on=key_columns, how="left")
        .with_columns(
            pl.col("retrieval_eligible").fill_null(False).alias("retrieval_row_present"),
            pl.col("ownership_rows_returned").fill_null(0).cast(pl.Int64),
            pl.col("ownership_distinct_categories").fill_null(0).cast(pl.Int64),
            pl.col("ownership_nonnull_value_count").fill_null(0).cast(pl.Int64),
            pl.col("ownership_returned_ric_nunique").fill_null(0).cast(pl.Int64),
        )
        .with_columns(
            (pl.col("ownership_returned_ric_nunique") == 1).fill_null(False).alias("ownership_single_returned_ric")
        )
        .select(OWNERSHIP_UNIVERSE_ROW_SUMMARY_COLUMNS)
        .collect()
    )
    return _cast_df_to_schema(summary_df, _ownership_universe_row_summary_schema()).select(
        OWNERSHIP_UNIVERSE_ROW_SUMMARY_COLUMNS
    )


def _write_api_fetch_manifest(
    *,
    stage_name: str,
    manifest_path: Path,
    staging_dir: Path,
    ledger_path: Path,
    request_log_path: Path,
    request_item_count: int,
    batch_count: int,
    batch_plan_fingerprint: str | None,
    batching_config: dict[str, Any] | None,
    metadata_source: str,
    cli_batching_args_ignored: bool,
    run_session_ids: list[str],
    summary: dict[str, Any] | None = None,
) -> Path:
    manifest_summary = dict(summary or {})
    manifest_summary["batch_plan_fingerprint"] = batch_plan_fingerprint
    return write_stage_fetch_manifest(
        stage_name=stage_name,
        manifest_path=manifest_path,
        staging_dir=staging_dir,
        ledger_path=ledger_path,
        request_log_path=request_log_path,
        batching_config=batching_config,
        request_item_count=request_item_count,
        batch_count=batch_count,
        run_session_ids=run_session_ids,
        summary=manifest_summary,
        metadata_source=metadata_source,
        cli_batching_args_ignored=cli_batching_args_ignored,
    )


def _interval_resume_metadata(
    *,
    planner_version: str,
    interval_plan: IntervalBatchPlan,
) -> dict[str, str]:
    return {
        "batch_plan_fingerprint": interval_plan.fingerprint,
        "batch_plan_planner_version": planner_version,
        "batching_config_json": json.dumps(interval_plan.config.to_serializable_dict(), sort_keys=True),
        "planned_batch_count": str(len(interval_plan.batches)),
    }


def _simple_resume_metadata(
    *,
    planner_version: str,
    batching_config: dict[str, Any],
    planned_batch_count: int,
) -> dict[str, str]:
    return {
        "batch_plan_planner_version": planner_version,
        "batching_config_json": json.dumps(batching_config, sort_keys=True),
        "planned_batch_count": str(planned_batch_count),
    }


def _raise_stage_resume_compatibility_error(
    *,
    stage_name: str,
    output_dir: Path,
    fetch_manifest_path: Path,
    manifest_path: Path,
    exc: LsegResumeCompatibilityError,
) -> None:
    output_map = {
        OWNERSHIP_UNIVERSE_STAGE: [output_dir / "refinitiv_ownership_universe_results.parquet", output_dir / "refinitiv_ownership_universe_row_summary.parquet"],
        DOC_EXACT_STAGE: [output_dir / "refinitiv_lm2011_doc_ownership_exact_requests.parquet", output_dir / "refinitiv_lm2011_doc_ownership_exact_raw.parquet"],
        DOC_FALLBACK_STAGE: [output_dir / "refinitiv_lm2011_doc_ownership_fallback_requests.parquet", output_dir / "refinitiv_lm2011_doc_ownership_fallback_raw.parquet"],
    }
    guidance = [str(fetch_manifest_path), str(exc.ledger_path), str(output_dir / f"staging\\{stage_name}"), str(manifest_path)]
    guidance.extend(str(path) for path in output_map.get(stage_name, []))
    explanation = (
        f"The {stage_name} stage already has persisted resume state from a different request universe or batching configuration."
    )
    if fetch_manifest_path.exists():
        explanation += " Review the stored fetch manifest for the original batching metadata before rerunning."
    else:
        explanation += " No stored fetch manifest was found; the ledger metadata is the only available resume record."
        if "batching_config_json" not in exc.existing_stage_meta:
            explanation += " The full historical batching configuration is not available in the stored metadata."
    raise LsegResumeCompatibilityError(
        stage=exc.stage,
        meta_key=exc.meta_key,
        existing_value=exc.existing_value,
        current_value=exc.current_value,
        ledger_path=exc.ledger_path,
        existing_stage_meta=exc.existing_stage_meta,
        current_stage_meta=exc.current_stage_meta,
        explanation=explanation,
        guidance=guidance,
    ) from exc


def _ensure_finalize_state(
    *,
    ledger_path: Path,
    request_log_path: Path,
    stage_name: str,
) -> None:
    if not ledger_path.exists():
        raise FileNotFoundError(f"{stage_name} ledger not found for finalize_only: {ledger_path}")
    if not request_log_path.exists():
        raise FileNotFoundError(f"{stage_name} request log not found for finalize_only: {request_log_path}")


def _stage_run_session_ids(ledger_path: Path) -> list[str]:
    return RequestLedger(ledger_path).run_session_ids()


def _coalesce_count(value: int | None, fallback: int) -> int:
    return fallback if value is None else value


def _scan_stage_outputs(
    *,
    stage: str,
    staging_dir: Path,
    ledger_path: Path | str | None,
    schema: dict[str, pl.DataType],
) -> pl.LazyFrame:
    staging_paths = _resolved_stage_output_paths(
        stage=stage,
        staging_dir=staging_dir,
        ledger_path=ledger_path,
    )
    if not staging_paths:
        return pl.DataFrame(schema=schema).lazy()
    return pl.concat([pl.scan_parquet(path) for path in staging_paths], how="vertical_relaxed")


def _count_request_log_events(request_log_path: Path, event_name: str) -> int:
    if not request_log_path.exists():
        return 0
    count = 0
    for line in request_log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("event") == event_name:
            count += 1
    return count


def _build_ownership_universe_items(handoff_df: pl.DataFrame) -> list[RequestItem]:
    items: list[RequestItem] = []
    for row in handoff_df.filter(pl.col("retrieval_eligible").fill_null(False)).to_dicts():
        ownership_lookup_row_id = _normalize_lookup_text(row.get("ownership_lookup_row_id"))
        candidate_ric = _normalize_lookup_text(row.get("candidate_ric"))
        request_start_date = _normalize_lookup_text(row.get("request_start_date"))
        request_end_date = _normalize_lookup_text(row.get("request_end_date"))
        if ownership_lookup_row_id is None or candidate_ric is None or request_start_date is None or request_end_date is None:
            continue
        request_start_date, request_end_date = _normalize_to_month_boundaries(
            request_start_date,
            request_end_date,
        )
        items.append(
            RequestItem(
                item_id=stable_hash_id(OWNERSHIP_UNIVERSE_STAGE, ownership_lookup_row_id, prefix="item"),
                stage=OWNERSHIP_UNIVERSE_STAGE,
                instrument=candidate_ric,
                batch_key=f"{request_start_date}|{request_end_date}",
                fields=OWNERSHIP_UNIVERSE_FIELDS,
                parameters={
                    "StatType": 7,
                    "SDate": request_start_date,
                    "EDate": request_end_date,
                },
                payload={"handoff_row": row},
            )
        )
    return items


def _resolve_interval_batch_config(
    *,
    max_batch_size: int,
    max_batch_items: int | None,
    max_extra_rows_abs: float | None,
    max_extra_rows_ratio: float | None,
    max_union_span_days: int | None,
    row_density_rows_per_day: float | None,
    default_row_density_rows_per_day: float,
    default_max_extra_rows_abs: float,
    default_max_extra_rows_ratio: float,
) -> IntervalBatchPlannerConfig:
    return IntervalBatchPlannerConfig(
        max_batch_size=max_batch_size,
        max_batch_items=max_batch_size if max_batch_items is None else max_batch_items,
        max_extra_rows_abs=(
            default_max_extra_rows_abs
            if max_extra_rows_abs is None
            else max_extra_rows_abs
        ),
        max_extra_rows_ratio=(
            default_max_extra_rows_ratio
            if max_extra_rows_ratio is None
            else max_extra_rows_ratio
        ),
        max_union_span_days=max_union_span_days,
        row_density_rows_per_day=(
            default_row_density_rows_per_day
            if row_density_rows_per_day is None
            else row_density_rows_per_day
        ),
    )


def _plan_ownership_universe_batches(
    items: list[RequestItem],
    *,
    config: IntervalBatchPlannerConfig,
) -> IntervalBatchPlan:
    return plan_interval_batches(
        items,
        config=config,
        planner_version=OWNERSHIP_UNIVERSE_INTERVAL_BATCH_PLANNER_VERSION,
        signature_fn=_ownership_universe_interval_signature,
        interval_fn=_item_window,
        batch_builder=_build_ownership_universe_interval_batch,
    )


def _ownership_universe_interval_signature(item: RequestItem) -> str:
    return request_signature(
        stage=item.stage,
        fields=item.fields,
        parameters=item.parameters,
        excluded_parameter_keys=_OWNERSHIP_INTERVAL_SIGNATURE_EXCLUDED_PARAMETER_KEYS,
    )


def _item_window(item: Any) -> tuple[dt.date, dt.date]:
    start_text = item.parameters.get("SDate")
    end_text = item.parameters.get("EDate")
    if not start_text or not end_text:
        raise ValueError(f"missing interval parameters for item_id={item.item_id}")
    return (dt.date.fromisoformat(str(start_text)), dt.date.fromisoformat(str(end_text)))


def _normalize_to_month_boundaries(start_text: str, end_text: str) -> tuple[str, str]:
    start_date = dt.date.fromisoformat(start_text)
    end_date = dt.date.fromisoformat(end_text)
    floored_start = start_date.replace(day=1)
    _, last_day = calendar.monthrange(end_date.year, end_date.month)
    ceiled_end = end_date.replace(day=last_day)
    return floored_start.isoformat(), ceiled_end.isoformat()


def _build_ownership_universe_interval_batch(
    items: list[RequestItem],
    start_date: dt.date,
    end_date: dt.date,
) -> Any:
    start_text = start_date.isoformat()
    end_text = end_date.isoformat()
    return build_batch_definition(
        items,
        batch_key=f"{start_text}|{end_text}",
        parameters={"StatType": 7, "SDate": start_text, "EDate": end_text},
    )


def _build_doc_ownership_exact_items(request_df: pl.DataFrame) -> list[RequestItem]:
    items: list[RequestItem] = []
    for row in request_df.filter(pl.col("retrieval_eligible").fill_null(False)).to_dicts():
        doc_id = _normalize_lookup_text(row.get("doc_id"))
        authoritative_ric = _normalize_lookup_text(row.get("authoritative_ric"))
        target_effective_date = _normalize_date_value(row.get("target_effective_date"))
        if doc_id is None or authoritative_ric is None or target_effective_date is None:
            continue
        date_text = target_effective_date.isoformat()
        items.append(
            RequestItem(
                item_id=stable_hash_id(DOC_EXACT_STAGE, doc_id, prefix="item"),
                stage=DOC_EXACT_STAGE,
                instrument=authoritative_ric,
                batch_key=date_text,
                fields=DOC_EXACT_FIELDS,
                parameters={"StatType": 7, "SDate": date_text, "EDate": date_text},
                payload={"request_row": row},
            )
        )
    return items


def _build_doc_ownership_fallback_items(request_df: pl.DataFrame) -> list[RequestItem]:
    items: list[RequestItem] = []
    for row in request_df.filter(pl.col("retrieval_eligible").fill_null(False)).to_dicts():
        doc_id = _normalize_lookup_text(row.get("doc_id"))
        authoritative_ric = _normalize_lookup_text(row.get("authoritative_ric"))
        fallback_window_start = _normalize_date_value(row.get("fallback_window_start"))
        fallback_window_end = _normalize_date_value(row.get("fallback_window_end"))
        if doc_id is None or authoritative_ric is None or fallback_window_start is None or fallback_window_end is None:
            continue
        start_text = fallback_window_start.isoformat()
        end_text = fallback_window_end.isoformat()
        items.append(
            RequestItem(
                item_id=stable_hash_id(DOC_FALLBACK_STAGE, doc_id, prefix="item"),
                stage=DOC_FALLBACK_STAGE,
                instrument=authoritative_ric,
                batch_key=f"{start_text}|{end_text}",
                fields=DOC_FALLBACK_FIELDS,
                parameters={"StatType": 7, "SDate": start_text, "EDate": end_text},
                payload={"request_row": row},
            )
        )
    return items


def _normalize_ownership_universe_batch_response(items: list[Any], frame: pl.DataFrame) -> pl.DataFrame:
    normalized_frame = _standardize_field_frame(
        frame,
        expected_fields=OWNERSHIP_UNIVERSE_FIELDS,
        field_aliases=OWNERSHIP_FIELD_ALIASES,
    )
    rows_by_instrument: dict[str, list[dict[str, Any]]] = {}
    for row in normalized_frame.to_dicts():
        instrument = _normalize_lookup_text(row.get("instrument"))
        if instrument is None:
            continue
        rows_by_instrument.setdefault(instrument, []).append(row)

    rows: list[dict[str, Any]] = []
    for item in items:
        handoff_row = dict(item.payload["handoff_row"])
        item_start_date, item_end_date = _item_window(item)
        matched_rows = rows_by_instrument.get(_normalize_lookup_text(item.instrument) or "", [])
        for matched_row in matched_rows:
            returned_ric = _normalize_lookup_text(matched_row.get("instrument"))
            returned_date = _normalize_date_value(matched_row.get("TR.CategoryOwnershipPct.Date"))
            returned_value = _normalize_float_value(matched_row.get("TR.CategoryOwnershipPct"))
            returned_category = _normalize_lookup_text(matched_row.get("TR.InstrStatTypeValue"))
            if returned_date is None:
                continue
            if not (item_start_date <= returned_date <= item_end_date):
                continue
            if returned_value is None and returned_category is None:
                continue
            rows.append(
                {
                    "item_id": item.item_id,
                    **{name: handoff_row.get(name) for name in OWNERSHIP_UNIVERSE_HANDOFF_COLUMNS},
                    "returned_ric": returned_ric,
                    "returned_date": returned_date,
                    "returned_category": returned_category,
                    "returned_value": returned_value,
                }
            )
    if not rows:
        return pl.DataFrame(schema={"item_id": pl.Utf8, **_ownership_universe_results_schema()})
    return _build_ownership_results_with_item_id_df(rows)


def _normalize_doc_exact_batch_response(items: list[Any], frame: pl.DataFrame) -> pl.DataFrame:
    normalized_frame = _standardize_field_frame(
        frame,
        expected_fields=DOC_EXACT_FIELDS,
        field_aliases=OWNERSHIP_FIELD_ALIASES,
    )
    rows_by_instrument: dict[str, list[dict[str, Any]]] = {}
    for row in normalized_frame.to_dicts():
        instrument = _normalize_lookup_text(row.get("instrument"))
        if instrument is None:
            continue
        rows_by_instrument.setdefault(instrument, []).append(row)

    rows: list[dict[str, Any]] = []
    for item in items:
        request_row = dict(item.payload["request_row"])
        matched_rows = rows_by_instrument.get(_normalize_lookup_text(item.instrument) or "", [])
        for matched_row in matched_rows:
            response_date = _normalize_date_value(matched_row.get("TR.CategoryOwnershipPct.Date"))
            returned_value = _normalize_float_value(matched_row.get("TR.CategoryOwnershipPct"))
            returned_category = _normalize_category(matched_row.get("TR.InstrStatTypeValue"))
            if response_date is None and returned_value is None and returned_category is None:
                continue
            rows.append(
                {
                    "item_id": item.item_id,
                    **{name: request_row.get(name) for name in DOC_OWNERSHIP_REQUEST_COLUMNS},
                    "request_stage": DOC_OWNERSHIP_EXACT_STAGE,
                    "response_date": response_date,
                    "response_date_is_imputed": False,
                    "returned_category": returned_category,
                    "returned_category_normalized": returned_category,
                    "returned_value": returned_value,
                    "is_institutional_category": _is_institutional_category(returned_category),
                }
            )
    if not rows:
        return pl.DataFrame(schema={"item_id": pl.Utf8, **_doc_ownership_raw_schema()})
    return _build_doc_raw_with_item_id_df(rows)


def _normalize_doc_fallback_batch_response(items: list[Any], frame: pl.DataFrame) -> pl.DataFrame:
    normalized_frame = _standardize_field_frame(
        frame,
        expected_fields=DOC_FALLBACK_FIELDS,
        field_aliases=OWNERSHIP_FIELD_ALIASES,
    )
    rows_by_instrument: dict[str, list[dict[str, Any]]] = {}
    for row in normalized_frame.to_dicts():
        instrument = _normalize_lookup_text(row.get("instrument"))
        if instrument is None:
            continue
        rows_by_instrument.setdefault(instrument, []).append(row)

    rows: list[dict[str, Any]] = []
    for item in items:
        request_row = dict(item.payload["request_row"])
        matched_rows = rows_by_instrument.get(_normalize_lookup_text(item.instrument) or "", [])
        for matched_row in matched_rows:
            response_date = _normalize_date_value(matched_row.get("TR.CategoryOwnershipPct.Date"))
            returned_value = _normalize_float_value(matched_row.get("TR.CategoryOwnershipPct"))
            returned_category = _normalize_category(matched_row.get("TR.InstrStatTypeValue"))
            if response_date is None and returned_value is None and returned_category is None:
                continue
            rows.append(
                {
                    "item_id": item.item_id,
                    **{name: request_row.get(name) for name in DOC_OWNERSHIP_REQUEST_COLUMNS},
                    "request_stage": DOC_OWNERSHIP_FALLBACK_STAGE,
                    "response_date": response_date,
                    "response_date_is_imputed": False,
                    "returned_category": returned_category,
                    "returned_category_normalized": returned_category,
                    "returned_value": returned_value,
                    "is_institutional_category": _is_institutional_category(returned_category),
                }
            )
    if not rows:
        return pl.DataFrame(schema={"item_id": pl.Utf8, **_doc_ownership_raw_schema()})
    return _build_doc_raw_with_item_id_df(rows)


def _empty_ownership_universe_results_df() -> pl.DataFrame:
    return _cast_df_to_schema(
        pl.DataFrame(schema={name: dtype for name, dtype in _ownership_universe_results_schema().items()}),
        _ownership_universe_results_schema(),
    ).select(OWNERSHIP_UNIVERSE_RESULTS_COLUMNS)


def _build_doc_raw_with_item_id_df(rows: list[dict[str, Any]]) -> pl.DataFrame:
    schema = {"item_id": pl.Utf8, **_doc_ownership_raw_schema()}
    if not rows:
        return _cast_doc_df_to_schema(pl.DataFrame(schema=schema), schema).select(["item_id", *DOC_OWNERSHIP_RAW_COLUMNS])
    return _cast_doc_df_to_schema(
        pl.DataFrame(rows, infer_schema_length=None),
        schema,
    ).select(["item_id", *DOC_OWNERSHIP_RAW_COLUMNS])


def _build_ownership_results_with_item_id_df(rows: list[dict[str, Any]]) -> pl.DataFrame:
    schema = {"item_id": pl.Utf8, **_ownership_universe_results_schema()}
    if not rows:
        return _cast_df_to_schema(pl.DataFrame(schema=schema), schema).select(["item_id", *OWNERSHIP_UNIVERSE_RESULTS_COLUMNS])
    return _cast_df_to_schema(
        pl.DataFrame(rows, infer_schema_length=None),
        schema,
    ).select(["item_id", *OWNERSHIP_UNIVERSE_RESULTS_COLUMNS])


def _resolved_stage_output_paths(
    *,
    stage: str,
    staging_dir: Path,
    ledger_path: Path | str | None,
) -> list[Path]:
    if ledger_path is None:
        return sorted(staging_dir.glob("*.parquet"))
    ledger = RequestLedger(ledger_path)
    return [
        path
        for path in ledger.succeeded_stage_output_paths(stage=stage)
        if path.exists()
    ]


def _normalize_api_stage_mode(api_stage_mode: str) -> str:
    if api_stage_mode not in API_STAGE_MODES:
        raise ValueError(f"Unsupported api_stage_mode: {api_stage_mode}")
    return api_stage_mode

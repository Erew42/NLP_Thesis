from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import polars as pl

from thesis_pkg.pipelines.refinitiv.analyst import (
    ANALYST_ACTUALS_RAW_COLUMNS,
    ANALYST_ESTIMATES_MONTHLY_RAW_COLUMNS,
    ANALYST_REQUEST_GROUP_COLUMNS,
    _actuals_raw_schema,
    _estimates_raw_schema,
    _request_group_schema,
)
from thesis_pkg.pipelines.refinitiv.doc_ownership import (
    _normalize_date_value,
    _normalize_float_value,
)
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
    build_batch_definition,
    plan_interval_batches,
    request_signature,
    stable_hash_id,
)
from thesis_pkg.pipelines.refinitiv.lseg_ledger import RequestLedger
from thesis_pkg.pipelines.refinitiv.lseg_ledger import LsegResumeCompatibilityError
from thesis_pkg.pipelines.refinitiv.lseg_stage_audit import (
    audit_api_stage,
    default_stage_manifest_path,
    write_stage_completion_manifest,
)
from thesis_pkg.pipelines.refinitiv_bridge_pipeline import (
    _cast_df_to_schema,
)


ANALYST_ACTUALS_STAGE = "analyst_actuals"
ANALYST_ESTIMATES_STAGE = "analyst_estimates_monthly"

ANALYST_INTERVAL_BATCH_PLANNER_VERSION = "analyst_interval_batching_v1"
_ANALYST_INTERVAL_SIGNATURE_EXCLUDED_PARAMETER_KEYS = ("SDate", "EDate")

ANALYST_ACTUALS_DEFAULT_ROW_DENSITY_ROWS_PER_DAY = 1.0 / 91.0
ANALYST_ACTUALS_DEFAULT_MAX_EXTRA_ROWS_ABS = 120.0
ANALYST_ACTUALS_DEFAULT_MAX_EXTRA_ROWS_RATIO = 0.25

ANALYST_ESTIMATES_DEFAULT_ROW_DENSITY_ROWS_PER_DAY = 1.0 / 30.5
ANALYST_ESTIMATES_DEFAULT_MAX_EXTRA_ROWS_ABS = 240.0
ANALYST_ESTIMATES_DEFAULT_MAX_EXTRA_ROWS_RATIO = 0.15

ANALYST_ACTUALS_FIELDS: tuple[str, ...] = (
    "TR.EPSActValue",
    "TR.EPSActValue.date",
    "TR.EPSActValue.periodenddate",
    "TR.EPSActValue.fperiod",
)

ANALYST_ESTIMATES_FIELDS: tuple[str, ...] = (
    "TR.EPSMean",
    "TR.EPSMean.calcdate",
    "TR.EPSMean.periodenddate",
    "TR.EPSMean.fperiod",
    "TR.EPSStdDev",
    "TR.EPSNumberofEstimates",
)

ANALYST_ESTIMATE_REQUEST_PERIODS: tuple[str, ...] = ("FQ1", "FQ2")

ANALYST_ACTUALS_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "TR.EPSActValue": ("EPS Actual Value", "Earnings Per Share - Actual"),
    "TR.EPSActValue.date": ("Date", "TR.EPSActValue Date"),
    "TR.EPSActValue.periodenddate": ("Period End Date",),
    "TR.EPSActValue.fperiod": ("Financial Period Absolute", "Fiscal Period", "FPeriod"),
}

ANALYST_ESTIMATES_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "TR.EPSMean": ("EPS Mean", "Earnings Per Share - Mean"),
    "TR.EPSMean.calcdate": ("Calc Date", "Calculation Date"),
    "TR.EPSMean.periodenddate": ("Period End Date",),
    "TR.EPSMean.fperiod": ("Financial Period Absolute", "Fiscal Period", "FPeriod"),
    "TR.EPSStdDev": ("EPS Standard Deviation", "Standard Deviation", "Earnings Per Share - Standard Deviation"),
    "TR.EPSNumberofEstimates": ("Number Of Estimates", "Earnings Per Share - Number of Estimates"),
}


def _analyst_resume_metadata(
    *,
    interval_plan: IntervalBatchPlan,
) -> dict[str, str]:
    return {
        "batch_plan_fingerprint": interval_plan.fingerprint,
        "batch_plan_planner_version": interval_plan.planner_version,
        "batching_config_json": json.dumps(interval_plan.config.to_serializable_dict(), sort_keys=True),
        "planned_batch_count": str(len(interval_plan.batches)),
    }


def _raise_analyst_resume_compatibility_error(
    *,
    stage: str,
    output_dir: Path,
    exc: LsegResumeCompatibilityError,
) -> None:
    if stage == ANALYST_ACTUALS_STAGE:
        guidance = [
            str(output_dir / "refinitiv_analyst_actuals_api_ledger.sqlite3"),
            str(output_dir / "refinitiv_analyst_actuals_api_requests.jsonl"),
            str(output_dir / "staging" / ANALYST_ACTUALS_STAGE),
            str(output_dir / "refinitiv_analyst_actuals_stage_manifest.json"),
            str(output_dir / "refinitiv_analyst_actuals_raw.parquet"),
            "After rebuilding analyst actuals, rerun downstream analyst normalization.",
        ]
        stage_label = "analyst actuals"
    elif stage == ANALYST_ESTIMATES_STAGE:
        guidance = [
            str(output_dir / "refinitiv_analyst_estimates_api_ledger.sqlite3"),
            str(output_dir / "refinitiv_analyst_estimates_api_requests.jsonl"),
            str(output_dir / "staging" / ANALYST_ESTIMATES_STAGE),
            str(output_dir / "refinitiv_analyst_estimates_stage_manifest.json"),
            str(output_dir / "refinitiv_analyst_estimates_monthly_raw.parquet"),
            "After rebuilding analyst estimates, rerun downstream analyst normalization.",
        ]
        stage_label = "analyst estimates monthly"
    else:
        guidance = []
        stage_label = stage
    raise LsegResumeCompatibilityError(
        stage=exc.stage,
        meta_key=exc.meta_key,
        existing_value=exc.existing_value,
        current_value=exc.current_value,
        ledger_path=exc.ledger_path,
        existing_stage_meta=exc.existing_stage_meta,
        current_stage_meta=exc.current_stage_meta,
        explanation=(
            f"The {stage_label} stage was requested for this run, but its existing resume state is incompatible "
            "with the current request universe or batching configuration."
        ),
        guidance=guidance,
    ) from exc


def run_refinitiv_step1_analyst_actuals_api_pipeline(
    *,
    request_universe_parquet_path: Path | str,
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
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = Path(ledger_path) if ledger_path is not None else output_dir / "refinitiv_analyst_actuals_api_ledger.sqlite3"
    request_log_path = (
        Path(request_log_path)
        if request_log_path is not None
        else output_dir / "refinitiv_analyst_actuals_api_requests.jsonl"
    )
    request_df = _cast_df_to_schema(
        pl.read_parquet(request_universe_parquet_path).select(ANALYST_REQUEST_GROUP_COLUMNS),
        _request_group_schema(),
    ).select(ANALYST_REQUEST_GROUP_COLUMNS)
    items = _build_analyst_actuals_items(request_df)
    planner_config = _resolve_interval_batch_config(
        max_batch_size=max_batch_size,
        max_batch_items=max_batch_items,
        max_extra_rows_abs=max_extra_rows_abs,
        max_extra_rows_ratio=max_extra_rows_ratio,
        max_union_span_days=max_union_span_days,
        row_density_rows_per_day=row_density_rows_per_day,
        default_row_density_rows_per_day=ANALYST_ACTUALS_DEFAULT_ROW_DENSITY_ROWS_PER_DAY,
        default_max_extra_rows_abs=ANALYST_ACTUALS_DEFAULT_MAX_EXTRA_ROWS_ABS,
        default_max_extra_rows_ratio=ANALYST_ACTUALS_DEFAULT_MAX_EXTRA_ROWS_RATIO,
    )
    interval_plan = _plan_analyst_actuals_batches(items, config=planner_config)
    try:
        stage_run = run_api_batches(
            stage=ANALYST_ACTUALS_STAGE,
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
            response_normalizer=_normalize_analyst_actuals_batch_response,
            lookup_normalizer=str,
            split_after_attempt=2,
            retry_delay_seconds_fn=_retry_delay_seconds,
            max_workers=max_workers,
            planned_batches=list(interval_plan.batches),
            batch_plan_fingerprint=interval_plan.fingerprint,
            planned_batch_metrics=interval_plan.batch_metrics_by_id,
            resume_compatibility_metadata=_analyst_resume_metadata(interval_plan=interval_plan),
        )
    except LsegResumeCompatibilityError as exc:
        _raise_analyst_resume_compatibility_error(
            stage=ANALYST_ACTUALS_STAGE,
            output_dir=output_dir,
            exc=exc,
        )
    stage_run_ledger_path = getattr(stage_run, "ledger_path", ledger_path)
    raw_df = _assemble_analyst_actuals_raw(stage_run.staging_dir, ledger_path=stage_run_ledger_path)
    raw_path = output_dir / "refinitiv_analyst_actuals_raw.parquet"
    raw_candidate_path = _candidate_output_path(raw_path)
    _write_parquet_atomic(raw_df, raw_candidate_path)

    manifest_path = (
        Path(stage_manifest_path)
        if stage_manifest_path is not None
        else default_stage_manifest_path(output_dir, ANALYST_ACTUALS_STAGE)
    )
    audit_result = audit_api_stage(
        stage_name=ANALYST_ACTUALS_STAGE,
        ledger_path=ledger_path,
        staging_dir=stage_run.staging_dir,
        output_artifacts={"analyst_actuals_raw_parquet": raw_candidate_path},
        declared_output_artifacts={"analyst_actuals_raw_parquet": raw_path},
        rebuilders={
            "analyst_actuals_raw_parquet": lambda: _assemble_analyst_actuals_raw(
                stage_run.staging_dir,
                ledger_path=stage_run_ledger_path,
            )
        },
        expected_stage_manifest_path=manifest_path,
    )
    if not audit_result.passed:
        raise RuntimeError(f"analyst actuals stage audit failed: {audit_result.to_dict()}")
    _promote_candidate_output(raw_candidate_path, raw_path)

    write_stage_completion_manifest(
        stage_name=ANALYST_ACTUALS_STAGE,
        manifest_path=manifest_path,
        input_artifacts={"analyst_request_universe_parquet": Path(request_universe_parquet_path)},
        output_artifacts={"analyst_actuals_raw_parquet": raw_path},
        ledger_path=ledger_path,
        request_log_path=request_log_path,
        staging_dir=stage_run.staging_dir,
        audit_result=audit_result,
        summary={
            "request_group_row_count": int(request_df.height),
            "request_item_count": len(items),
            "planned_batch_count": len(interval_plan.batches),
            "batch_plan_fingerprint": interval_plan.fingerprint,
            "batching_config": planner_config.to_serializable_dict(),
            "raw_row_count": int(raw_df.height),
            "run_session_id": stage_run.run_session_id,
        },
    )
    result = {
        "refinitiv_analyst_actuals_raw_parquet": raw_path,
        "refinitiv_analyst_actuals_api_ledger_sqlite3": ledger_path,
        "refinitiv_analyst_actuals_api_requests_jsonl": request_log_path,
        "refinitiv_analyst_actuals_stage_manifest_json": manifest_path,
    }
    return result


def run_refinitiv_step1_analyst_estimates_monthly_api_pipeline(
    *,
    request_universe_parquet_path: Path | str,
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
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = Path(ledger_path) if ledger_path is not None else output_dir / "refinitiv_analyst_estimates_api_ledger.sqlite3"
    request_log_path = (
        Path(request_log_path)
        if request_log_path is not None
        else output_dir / "refinitiv_analyst_estimates_api_requests.jsonl"
    )
    request_df = _cast_df_to_schema(
        pl.read_parquet(request_universe_parquet_path).select(ANALYST_REQUEST_GROUP_COLUMNS),
        _request_group_schema(),
    ).select(ANALYST_REQUEST_GROUP_COLUMNS)
    items = _build_analyst_estimate_items(request_df)
    planner_config = _resolve_interval_batch_config(
        max_batch_size=max_batch_size,
        max_batch_items=max_batch_items,
        max_extra_rows_abs=max_extra_rows_abs,
        max_extra_rows_ratio=max_extra_rows_ratio,
        max_union_span_days=max_union_span_days,
        row_density_rows_per_day=row_density_rows_per_day,
        default_row_density_rows_per_day=ANALYST_ESTIMATES_DEFAULT_ROW_DENSITY_ROWS_PER_DAY,
        default_max_extra_rows_abs=ANALYST_ESTIMATES_DEFAULT_MAX_EXTRA_ROWS_ABS,
        default_max_extra_rows_ratio=ANALYST_ESTIMATES_DEFAULT_MAX_EXTRA_ROWS_RATIO,
    )
    interval_plan = _plan_analyst_estimate_batches(items, config=planner_config)
    try:
        stage_run = run_api_batches(
            stage=ANALYST_ESTIMATES_STAGE,
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
            response_normalizer=_normalize_analyst_estimates_batch_response,
            lookup_normalizer=str,
            split_after_attempt=2,
            retry_delay_seconds_fn=_retry_delay_seconds,
            max_workers=max_workers,
            planned_batches=list(interval_plan.batches),
            batch_plan_fingerprint=interval_plan.fingerprint,
            planned_batch_metrics=interval_plan.batch_metrics_by_id,
            resume_compatibility_metadata=_analyst_resume_metadata(interval_plan=interval_plan),
        )
    except LsegResumeCompatibilityError as exc:
        _raise_analyst_resume_compatibility_error(
            stage=ANALYST_ESTIMATES_STAGE,
            output_dir=output_dir,
            exc=exc,
        )
    stage_run_ledger_path = getattr(stage_run, "ledger_path", ledger_path)
    raw_df = _assemble_analyst_estimates_raw(stage_run.staging_dir, ledger_path=stage_run_ledger_path)
    raw_path = output_dir / "refinitiv_analyst_estimates_monthly_raw.parquet"
    raw_candidate_path = _candidate_output_path(raw_path)
    _write_parquet_atomic(raw_df, raw_candidate_path)

    manifest_path = (
        Path(stage_manifest_path)
        if stage_manifest_path is not None
        else default_stage_manifest_path(output_dir, ANALYST_ESTIMATES_STAGE)
    )
    audit_result = audit_api_stage(
        stage_name=ANALYST_ESTIMATES_STAGE,
        ledger_path=ledger_path,
        staging_dir=stage_run.staging_dir,
        output_artifacts={"analyst_estimates_raw_parquet": raw_candidate_path},
        declared_output_artifacts={"analyst_estimates_raw_parquet": raw_path},
        rebuilders={
            "analyst_estimates_raw_parquet": lambda: _assemble_analyst_estimates_raw(
                stage_run.staging_dir,
                ledger_path=stage_run_ledger_path,
            )
        },
        expected_stage_manifest_path=manifest_path,
    )
    if not audit_result.passed:
        raise RuntimeError(f"analyst estimates stage audit failed: {audit_result.to_dict()}")
    _promote_candidate_output(raw_candidate_path, raw_path)

    write_stage_completion_manifest(
        stage_name=ANALYST_ESTIMATES_STAGE,
        manifest_path=manifest_path,
        input_artifacts={"analyst_request_universe_parquet": Path(request_universe_parquet_path)},
        output_artifacts={"analyst_estimates_raw_parquet": raw_path},
        ledger_path=ledger_path,
        request_log_path=request_log_path,
        staging_dir=stage_run.staging_dir,
        audit_result=audit_result,
        summary={
            "request_group_row_count": int(request_df.height),
            "request_item_count": len(items),
            "planned_batch_count": len(interval_plan.batches),
            "batch_plan_fingerprint": interval_plan.fingerprint,
            "batching_config": planner_config.to_serializable_dict(),
            "raw_row_count": int(raw_df.height),
            "run_session_id": stage_run.run_session_id,
        },
    )
    result = {
        "refinitiv_analyst_estimates_monthly_raw_parquet": raw_path,
        "refinitiv_analyst_estimates_api_ledger_sqlite3": ledger_path,
        "refinitiv_analyst_estimates_api_requests_jsonl": request_log_path,
        "refinitiv_analyst_estimates_stage_manifest_json": manifest_path,
    }
    return result


def _assemble_analyst_actuals_raw(
    staging_dir: Path,
    *,
    ledger_path: Path | str | None = None,
) -> pl.DataFrame:
    staging_paths = _resolved_stage_output_paths(
        stage=ANALYST_ACTUALS_STAGE,
        staging_dir=staging_dir,
        ledger_path=ledger_path,
    )
    raw_df = (
        pl.concat([pl.read_parquet(path) for path in staging_paths], how="vertical_relaxed")
        if staging_paths
        else pl.DataFrame(schema=_actuals_raw_schema())
    )
    return (
        _cast_df_to_schema(raw_df.select(ANALYST_ACTUALS_RAW_COLUMNS), _actuals_raw_schema())
        .select(ANALYST_ACTUALS_RAW_COLUMNS)
        .unique(subset=["item_id", "response_row_index"], maintain_order=True)
    )


def _assemble_analyst_estimates_raw(
    staging_dir: Path,
    *,
    ledger_path: Path | str | None = None,
) -> pl.DataFrame:
    staging_paths = _resolved_stage_output_paths(
        stage=ANALYST_ESTIMATES_STAGE,
        staging_dir=staging_dir,
        ledger_path=ledger_path,
    )
    raw_df = (
        pl.concat([pl.read_parquet(path) for path in staging_paths], how="vertical_relaxed")
        if staging_paths
        else pl.DataFrame(schema=_estimates_raw_schema())
    )
    return (
        _cast_df_to_schema(raw_df.select(ANALYST_ESTIMATES_MONTHLY_RAW_COLUMNS), _estimates_raw_schema())
        .select(ANALYST_ESTIMATES_MONTHLY_RAW_COLUMNS)
        .unique(subset=["item_id", "response_row_index"], maintain_order=True)
    )


def _build_analyst_actuals_items(request_df: pl.DataFrame) -> list[RequestItem]:
    items: list[RequestItem] = []
    for row in request_df.filter(pl.col("retrieval_eligible").fill_null(False)).to_dicts():
        request_group_id = row.get("request_group_id")
        ric = row.get("effective_collection_ric")
        start_date = row.get("actuals_request_start_date")
        end_date = row.get("actuals_request_end_date")
        if request_group_id is None or ric is None or start_date is None or end_date is None:
            continue
        start_text = start_date.isoformat()
        end_text = end_date.isoformat()
        items.append(
            RequestItem(
                item_id=stable_hash_id(ANALYST_ACTUALS_STAGE, request_group_id, prefix="item"),
                stage=ANALYST_ACTUALS_STAGE,
                instrument=str(ric),
                batch_key=f"{start_text}|{end_text}",
                fields=ANALYST_ACTUALS_FIELDS,
                parameters={"Frq": "FQ", "Period": "FI0", "SDate": start_text, "EDate": end_text},
                payload={"request_row": row},
            )
        )
    return items


def _build_analyst_estimate_items(request_df: pl.DataFrame) -> list[RequestItem]:
    items: list[RequestItem] = []
    for row in request_df.filter(pl.col("retrieval_eligible").fill_null(False)).to_dicts():
        request_group_id = row.get("request_group_id")
        ric = row.get("effective_collection_ric")
        start_date = row.get("estimates_request_start_date")
        end_date = row.get("estimates_request_end_date")
        if request_group_id is None or ric is None or start_date is None or end_date is None:
            continue
        start_text = start_date.isoformat()
        end_text = end_date.isoformat()
        for request_period in ANALYST_ESTIMATE_REQUEST_PERIODS:
            items.append(
                RequestItem(
                    item_id=stable_hash_id(ANALYST_ESTIMATES_STAGE, request_group_id, request_period, prefix="item"),
                    stage=ANALYST_ESTIMATES_STAGE,
                    instrument=str(ric),
                    batch_key=f"{start_text}|{end_text}|{request_period}",
                    fields=ANALYST_ESTIMATES_FIELDS,
                    parameters={"Frq": "M", "Period": request_period, "SDate": start_text, "EDate": end_text},
                    payload={"request_row": row},
                )
            )
    return items


def _normalize_int_value(value: Any) -> int | None:
    numeric = _normalize_float_value(value)
    if numeric is None:
        return None
    return int(numeric)


def _normalize_analyst_actuals_batch_response(items: list[Any], frame: pl.DataFrame) -> pl.DataFrame:
    normalized_frame = _standardize_field_frame(
        frame,
        expected_fields=ANALYST_ACTUALS_FIELDS,
        field_aliases=ANALYST_ACTUALS_FIELD_ALIASES,
    )
    rows_by_instrument: dict[str, list[dict[str, Any]]] = {}
    for row in normalized_frame.to_dicts():
        instrument = str(row.get("instrument")) if row.get("instrument") is not None else None
        if instrument is None:
            continue
        rows_by_instrument.setdefault(instrument, []).append(row)

    rows: list[dict[str, Any]] = []
    for item in items:
        request_row = dict(item.payload["request_row"])
        item_start_date, item_end_date = _item_window(item)
        matched_rows = rows_by_instrument.get(item.instrument, [])
        response_row_index = 0
        for matched_row in matched_rows:
            announcement_date = _normalize_date_value(matched_row.get("TR.EPSActValue.date"))
            if announcement_date is None or not (item_start_date <= announcement_date <= item_end_date):
                continue
            fiscal_period_end = _normalize_date_value(matched_row.get("TR.EPSActValue.periodenddate"))
            actual_eps = _normalize_float_value(matched_row.get("TR.EPSActValue"))
            raw_fperiod = (
                None
                if matched_row.get("TR.EPSActValue.fperiod") is None
                else str(matched_row.get("TR.EPSActValue.fperiod")).strip() or None
            )
            if announcement_date is None and actual_eps is None:
                continue
            rows.append(
                {
                    "item_id": item.item_id,
                    "response_row_index": response_row_index,
                    "request_group_id": request_row.get("request_group_id"),
                    "gvkey_int": request_row.get("gvkey_int"),
                    "effective_collection_ric": request_row.get("effective_collection_ric"),
                    "announcement_date": announcement_date,
                    "fiscal_period_end": fiscal_period_end,
                    "actual_eps": actual_eps,
                    "raw_fperiod": raw_fperiod,
                    "row_parse_status": (
                        "MISSING_ACTUAL_EPS"
                        if actual_eps is None
                        else "MISSING_FISCAL_PERIOD_END"
                        if fiscal_period_end is None
                        else "OK"
                    ),
                }
            )
            response_row_index += 1
    if not rows:
        return pl.DataFrame(schema=_actuals_raw_schema())
    return _cast_df_to_schema(
        pl.DataFrame(rows, schema=_actuals_raw_schema()),
        _actuals_raw_schema(),
    )


def _normalize_analyst_estimates_batch_response(items: list[Any], frame: pl.DataFrame) -> pl.DataFrame:
    normalized_frame = _standardize_field_frame(
        frame,
        expected_fields=ANALYST_ESTIMATES_FIELDS,
        field_aliases=ANALYST_ESTIMATES_FIELD_ALIASES,
    )
    rows_by_instrument: dict[str, list[dict[str, Any]]] = {}
    for row in normalized_frame.to_dicts():
        instrument = str(row.get("instrument")) if row.get("instrument") is not None else None
        if instrument is None:
            continue
        rows_by_instrument.setdefault(instrument, []).append(row)

    rows: list[dict[str, Any]] = []
    for item in items:
        request_row = dict(item.payload["request_row"])
        item_start_date, item_end_date = _item_window(item)
        matched_rows = rows_by_instrument.get(item.instrument, [])
        response_row_index = 0
        for matched_row in matched_rows:
            calc_date = _normalize_date_value(matched_row.get("TR.EPSMean.calcdate"))
            if calc_date is None or not (item_start_date <= calc_date <= item_end_date):
                continue
            fiscal_period_end = _normalize_date_value(matched_row.get("TR.EPSMean.periodenddate"))
            forecast_consensus_mean = _normalize_float_value(matched_row.get("TR.EPSMean"))
            forecast_dispersion = _normalize_float_value(matched_row.get("TR.EPSStdDev"))
            estimate_count = _normalize_int_value(matched_row.get("TR.EPSNumberofEstimates"))
            raw_fperiod = (
                None
                if matched_row.get("TR.EPSMean.fperiod") is None
                else str(matched_row.get("TR.EPSMean.fperiod")).strip() or None
            )
            if (
                calc_date is None
                and fiscal_period_end is None
                and forecast_consensus_mean is None
                and forecast_dispersion is None
                and estimate_count is None
            ):
                continue
            row_parse_status = (
                "MISSING_FISCAL_PERIOD_END"
                if fiscal_period_end is None
                else "MISSING_CONSENSUS_MEAN"
                if forecast_consensus_mean is None
                else "OK"
            )
            rows.append(
                {
                    "item_id": item.item_id,
                    "response_row_index": response_row_index,
                    "request_group_id": request_row.get("request_group_id"),
                    "request_period": item.parameters.get("Period"),
                    "gvkey_int": request_row.get("gvkey_int"),
                    "effective_collection_ric": request_row.get("effective_collection_ric"),
                    "calc_date": calc_date,
                    "fiscal_period_end": fiscal_period_end,
                    "raw_fperiod": raw_fperiod,
                    "forecast_consensus_mean": forecast_consensus_mean,
                    "forecast_dispersion": forecast_dispersion,
                    "estimate_count": estimate_count,
                    "row_parse_status": row_parse_status,
                }
            )
            response_row_index += 1
    if not rows:
        return pl.DataFrame(schema=_estimates_raw_schema())
    return _cast_df_to_schema(
        pl.DataFrame(rows, schema=_estimates_raw_schema()),
        _estimates_raw_schema(),
    )


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


def _plan_analyst_actuals_batches(
    items: list[RequestItem],
    *,
    config: IntervalBatchPlannerConfig,
) -> IntervalBatchPlan:
    return plan_interval_batches(
        items,
        config=config,
        planner_version=ANALYST_INTERVAL_BATCH_PLANNER_VERSION,
        signature_fn=_analyst_interval_signature,
        interval_fn=_item_window,
        batch_builder=_build_analyst_actuals_interval_batch,
    )


def _plan_analyst_estimate_batches(
    items: list[RequestItem],
    *,
    config: IntervalBatchPlannerConfig,
) -> IntervalBatchPlan:
    return plan_interval_batches(
        items,
        config=config,
        planner_version=ANALYST_INTERVAL_BATCH_PLANNER_VERSION,
        signature_fn=_analyst_interval_signature,
        interval_fn=_item_window,
        batch_builder=_build_analyst_estimates_interval_batch,
    )


def _analyst_interval_signature(item: RequestItem) -> str:
    return request_signature(
        stage=item.stage,
        fields=item.fields,
        parameters=item.parameters,
        excluded_parameter_keys=_ANALYST_INTERVAL_SIGNATURE_EXCLUDED_PARAMETER_KEYS,
    )


def _item_window(item: Any) -> tuple[dt.date, dt.date]:
    start_text = item.parameters.get("SDate")
    end_text = item.parameters.get("EDate")
    if not start_text or not end_text:
        raise ValueError(f"missing interval parameters for item_id={item.item_id}")
    return (dt.date.fromisoformat(str(start_text)), dt.date.fromisoformat(str(end_text)))


def _build_analyst_actuals_interval_batch(
    items: list[RequestItem],
    start_date: dt.date,
    end_date: dt.date,
) -> Any:
    start_text = start_date.isoformat()
    end_text = end_date.isoformat()
    return build_batch_definition(
        items,
        batch_key=f"{start_text}|{end_text}",
        parameters={"Frq": "FQ", "Period": "FI0", "SDate": start_text, "EDate": end_text},
    )


def _build_analyst_estimates_interval_batch(
    items: list[RequestItem],
    start_date: dt.date,
    end_date: dt.date,
) -> Any:
    periods = {str(item.parameters.get("Period")) for item in items}
    if len(periods) != 1:
        raise ValueError(f"analyst estimate batch contains mixed request periods: {sorted(periods)}")
    request_period = next(iter(periods))
    start_text = start_date.isoformat()
    end_text = end_date.isoformat()
    return build_batch_definition(
        items,
        batch_key=f"{start_text}|{end_text}|{request_period}",
        parameters={"Frq": "M", "Period": request_period, "SDate": start_text, "EDate": end_text},
    )


def _resolved_stage_output_paths(
    *,
    stage: str,
    staging_dir: Path,
    ledger_path: Path | str | None,
) -> list[Path]:
    if ledger_path is None:
        return sorted(staging_dir.glob("*.parquet"))
    ledger = RequestLedger(ledger_path)
    paths = []
    for path in ledger.succeeded_stage_output_paths(stage=stage):
        if path.exists():
            paths.append(path)
    return paths


__all__ = [
    "ANALYST_ACTUALS_FIELDS",
    "ANALYST_ACTUALS_STAGE",
    "ANALYST_ESTIMATES_FIELDS",
    "ANALYST_ESTIMATES_STAGE",
    "run_refinitiv_step1_analyst_actuals_api_pipeline",
    "run_refinitiv_step1_analyst_estimates_monthly_api_pipeline",
]

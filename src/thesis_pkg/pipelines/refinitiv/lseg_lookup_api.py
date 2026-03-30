from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from thesis_pkg.pipelines.refinitiv.lseg_api_common import (
    append_json_log as _append_json_log,
    candidate_output_path as _candidate_output_path,
    classify_error as _classify_error,
    daily_limit_likely_exhausted as _daily_limit_likely_exhausted,
    error_details_for_exception as _error_details,
    next_daily_resume_utc as _next_daily_resume_utc,
    promote_candidate_output as _promote_candidate_output,
    retry_delay_seconds as _retry_delay_seconds,
    should_treat_as_empty_result as _should_treat_as_empty_result,
    standardize_field_frame as _standardize_field_frame,
    write_parquet_atomic as _write_parquet_atomic,
)
from thesis_pkg.pipelines.refinitiv.lseg_api_execution import run_api_batches
from thesis_pkg.pipelines.refinitiv.lseg_batching import (
    RequestItem,
    batch_items,
    batch_plan_fingerprint,
    stable_hash_id,
)
from thesis_pkg.pipelines.refinitiv.lseg_ledger import LsegResumeCompatibilityError, RequestLedger
from thesis_pkg.pipelines.refinitiv.lseg_stage_audit import (
    audit_api_stage,
    default_stage_fetch_manifest_path,
    default_stage_manifest_path,
    resolve_stage_output_path,
    resolve_stage_fetch_metadata,
    write_stage_completion_manifest,
    write_stage_fetch_manifest,
)
from thesis_pkg.pipelines.refinitiv_bridge_pipeline import (
    LOOKUP_IDENTIFIER_TYPES,
    RIC_LOOKUP_EXTENDED_BASE_COLUMNS,
    RIC_LOOKUP_EXTENDED_COLUMNS,
    _cast_df_to_schema,
    _extended_lookup_schema,
    _normalize_lookup_text,
    _normalize_resolution_input_df,
)


LOOKUP_STAGE = "lookup"
LOOKUP_FIELDS: tuple[str, ...] = (
    "TR.RIC",
    "TR.CommonName",
    "TR.ISIN",
    "TR.CUSIP",
)
LOOKUP_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "TR.RIC": ("RIC",),
    "TR.CommonName": ("Company Common Name", "Common Name"),
    "TR.ISIN": ("ISIN",),
    "TR.CUSIP": ("CUSIP",),
}
API_STAGE_MODES: frozenset[str] = frozenset({"full", "fetch_only", "finalize_only"})
LOOKUP_BATCH_PLANNER_VERSION = "lookup_batching_v1"


def run_refinitiv_step1_lookup_api_pipeline(
    *,
    snapshot_parquet_path: Path | str,
    output_dir: Path | str,
    provider: Any | None = None,
    ledger_path: Path | str | None = None,
    request_log_path: Path | str | None = None,
    max_batch_size: int = 25,
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
    snapshot_parquet_path = Path(snapshot_parquet_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = Path(ledger_path) if ledger_path is not None else output_dir / "refinitiv_lookup_api_ledger.sqlite3"
    request_log_path = (
        Path(request_log_path)
        if request_log_path is not None
        else output_dir / "refinitiv_lookup_api_requests.jsonl"
    )
    manifest_path = (
        Path(stage_manifest_path)
        if stage_manifest_path is not None
        else default_stage_manifest_path(output_dir, LOOKUP_STAGE)
    )
    fetch_manifest_path = (
        Path(fetch_manifest_path)
        if fetch_manifest_path is not None
        else default_stage_fetch_manifest_path(output_dir, LOOKUP_STAGE)
    )

    snapshot_df = _read_lookup_snapshot(snapshot_parquet_path)
    items = _build_lookup_items(snapshot_df)
    batching_config = {"max_batch_size": max_batch_size}
    planned_batches = batch_items(items, max_batch_size=max_batch_size, unique_instrument_limit=True)
    current_batch_plan_fingerprint = batch_plan_fingerprint(
        planner_version=LOOKUP_BATCH_PLANNER_VERSION,
        batching_config=batching_config,
        planned_batches=planned_batches,
    )
    stage_run = None
    if api_stage_mode != "finalize_only":
        try:
            stage_run = run_api_batches(
                stage=LOOKUP_STAGE,
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
                response_normalizer=_normalize_lookup_batch_response,
                lookup_normalizer=_normalize_lookup_text,
                retry_delay_seconds_fn=_retry_delay_seconds,
                max_workers=max_workers,
                planned_batches=planned_batches,
                batch_plan_fingerprint=current_batch_plan_fingerprint,
                resume_compatibility_metadata={
                    "batch_plan_planner_version": LOOKUP_BATCH_PLANNER_VERSION,
                    "batching_config_json": json.dumps(batching_config, sort_keys=True),
                    "planned_batch_count": str(len(planned_batches)),
                },
            )
        except LsegResumeCompatibilityError as exc:
            _raise_lookup_resume_compatibility_error(
                output_dir=output_dir,
                fetch_manifest_path=fetch_manifest_path,
                manifest_path=manifest_path,
                exc=exc,
            )

    if api_stage_mode == "finalize_only":
        if not ledger_path.exists():
            raise FileNotFoundError(f"lookup ledger not found for finalize_only: {ledger_path}")
        if not request_log_path.exists():
            raise FileNotFoundError(f"lookup request log not found for finalize_only: {request_log_path}")

    staging_dir = output_dir / "staging" / LOOKUP_STAGE if stage_run is None else stage_run.staging_dir
    metadata = (
        resolve_stage_fetch_metadata(
            stage_name=LOOKUP_STAGE,
            ledger_path=ledger_path,
            fetch_manifest_path=fetch_manifest_path,
            stage_manifest_path=manifest_path,
            current_batching_config=batching_config,
            current_batch_plan_fingerprint=current_batch_plan_fingerprint,
        )
        if api_stage_mode == "finalize_only"
        else None
    )
    _write_lookup_fetch_manifest(
        manifest_path=fetch_manifest_path,
        ledger_path=ledger_path,
        request_log_path=request_log_path,
        staging_dir=staging_dir,
        metadata_source="current_run" if metadata is None else metadata.metadata_source,
        cli_batching_args_ignored=False if metadata is None else metadata.cli_batching_args_ignored,
        batch_plan_fingerprint=current_batch_plan_fingerprint if metadata is None else metadata.batch_plan_fingerprint,
        batching_config=batching_config if metadata is None else metadata.batching_config,
        request_item_count=len(items) if metadata is None else metadata.request_item_count,
        batch_count=len(planned_batches) if metadata is None else metadata.batch_count,
        run_session_ids=_stage_run_session_ids(ledger_path) if metadata is None else metadata.run_session_ids,
        items=items,
    )

    result = {
        "refinitiv_lookup_api_ledger_sqlite3": ledger_path,
        "refinitiv_lookup_api_requests_jsonl": request_log_path,
        "refinitiv_lookup_fetch_manifest_json": fetch_manifest_path,
    }
    if api_stage_mode == "fetch_only":
        return result

    output_path = output_dir / "refinitiv_ric_lookup_handoff_common_stock_extended.parquet"
    result.update(
        _finalize_lookup_stage(
            snapshot_df=snapshot_df,
            snapshot_parquet_path=snapshot_parquet_path,
            ledger_path=ledger_path,
            request_log_path=request_log_path,
            staging_dir=staging_dir,
            output_path=output_path,
            manifest_path=manifest_path,
            item_count=len(items),
            planned_batch_count=len(planned_batches) if metadata is None else metadata.batch_count,
            batch_plan_fingerprint=current_batch_plan_fingerprint if metadata is None else metadata.batch_plan_fingerprint,
            batching_config=batching_config if metadata is None else metadata.batching_config,
            metadata_source="current_run" if metadata is None else metadata.metadata_source,
            cli_batching_args_ignored=False if metadata is None else metadata.cli_batching_args_ignored,
            verify_rebuilders=False,
            run_session_id=None if stage_run is None else stage_run.run_session_id,
        )
    )
    return result


def _read_lookup_snapshot(snapshot_parquet_path: Path) -> pl.DataFrame:
    if not snapshot_parquet_path.exists():
        raise FileNotFoundError(f"lookup snapshot parquet not found: {snapshot_parquet_path}")
    return _cast_df_to_schema(
        pl.read_parquet(snapshot_parquet_path).select(RIC_LOOKUP_EXTENDED_COLUMNS),
        _extended_lookup_schema(),
    ).select(RIC_LOOKUP_EXTENDED_COLUMNS)


def _build_lookup_items(snapshot_df: pl.DataFrame) -> list[RequestItem]:
    items: list[RequestItem] = []
    for row in snapshot_df.to_dicts():
        bridge_row_id = _normalize_lookup_text(row.get("bridge_row_id"))
        if bridge_row_id is None:
            continue
        for identifier_type in LOOKUP_IDENTIFIER_TYPES:
            lookup_input = _normalize_lookup_text(row.get(f"{identifier_type}_lookup_input"))
            if lookup_input is None:
                continue
            item_id = stable_hash_id(LOOKUP_STAGE, bridge_row_id, identifier_type, prefix="item")
            items.append(
                RequestItem(
                    item_id=item_id,
                    stage=LOOKUP_STAGE,
                    instrument=lookup_input,
                    batch_key=identifier_type,
                    fields=LOOKUP_FIELDS,
                    parameters={},
                    payload={
                        "bridge_row_id": bridge_row_id,
                        "identifier_type": identifier_type,
                        "lookup_input": lookup_input,
                    },
                )
            )
    return items


def _normalize_lookup_batch_response(
    items: list[Any],
    frame: pl.DataFrame,
) -> pl.DataFrame:
    normalized_frame = _standardize_field_frame(
        frame,
        expected_fields=LOOKUP_FIELDS,
        field_aliases=LOOKUP_FIELD_ALIASES,
    )
    response_by_instrument: dict[str, dict[str, Any]] = {}
    for row in normalized_frame.to_dicts():
        instrument = _normalize_lookup_text(row.get("instrument"))
        if instrument is None:
            continue
        response_by_instrument[instrument] = {
            "returned_ric": _normalize_lookup_text(row.get("TR.RIC")),
            "returned_name": _normalize_lookup_text(row.get("TR.CommonName")),
            "returned_isin": _normalize_lookup_text(row.get("TR.ISIN")),
            "returned_cusip": _normalize_lookup_text(row.get("TR.CUSIP")),
        }

    rows: list[dict[str, Any]] = []
    for item in items:
        payload = item.payload
        instrument = _normalize_lookup_text(item.instrument)
        matched = {} if instrument is None else response_by_instrument.get(instrument, {})
        rows.append(
            {
                "item_id": item.item_id,
                "bridge_row_id": payload["bridge_row_id"],
                "identifier_type": payload["identifier_type"],
                "lookup_input": instrument,
                "returned_ric": matched.get("returned_ric"),
                "returned_name": matched.get("returned_name"),
                "returned_isin": matched.get("returned_isin"),
                "returned_cusip": matched.get("returned_cusip"),
            }
        )
    return pl.DataFrame(rows)


def _assemble_lookup_output(
    snapshot_df: pl.DataFrame,
    staging_dir: Path,
    *,
    ledger_path: Path | str | None = None,
) -> pl.DataFrame:
    results_lf = _lookup_results_lazy_frame(staging_dir, ledger_path=ledger_path).unique(
        subset=["item_id"],
        keep="last",
        maintain_order=True,
    )
    base_columns = list(RIC_LOOKUP_EXTENDED_BASE_COLUMNS) + [
        f"{identifier_type}_lookup_input" for identifier_type in LOOKUP_IDENTIFIER_TYPES
    ]
    assembled_lf = snapshot_df.select(base_columns).lazy()
    for identifier_type in LOOKUP_IDENTIFIER_TYPES:
        identifier_lf = (
            results_lf.filter(pl.col("identifier_type") == identifier_type)
            .select(
                "bridge_row_id",
                pl.col("returned_ric").alias(f"{identifier_type}_returned_ric"),
                pl.col("returned_name").alias(f"{identifier_type}_returned_name"),
                pl.col("returned_isin").alias(f"{identifier_type}_returned_isin"),
                pl.col("returned_cusip").alias(f"{identifier_type}_returned_cusip"),
            )
            .unique(subset=["bridge_row_id"], keep="last", maintain_order=True)
        )
        assembled_lf = assembled_lf.join(identifier_lf, on="bridge_row_id", how="left")

    assembled_df = assembled_lf.collect()
    for identifier_type in LOOKUP_IDENTIFIER_TYPES:
        for suffix in ("returned_ric", "returned_name", "returned_isin", "returned_cusip"):
            column_name = f"{identifier_type}_{suffix}"
            if column_name not in assembled_df.columns:
                assembled_df = assembled_df.with_columns(pl.lit(None, dtype=pl.Utf8).alias(column_name))
    return _normalize_resolution_input_df(assembled_df)


def _finalize_lookup_stage(
    *,
    snapshot_df: pl.DataFrame,
    snapshot_parquet_path: Path,
    ledger_path: Path,
    request_log_path: Path,
    staging_dir: Path,
    output_path: Path,
    manifest_path: Path,
    item_count: int,
    planned_batch_count: int | None,
    batch_plan_fingerprint: str | None,
    batching_config: dict[str, Any] | None,
    metadata_source: str,
    cli_batching_args_ignored: bool,
    verify_rebuilders: bool,
    run_session_id: str | None,
) -> dict[str, Path]:
    final_df = _assemble_lookup_output(snapshot_df, staging_dir, ledger_path=ledger_path)
    candidate_path = _candidate_output_path(output_path)
    _write_parquet_atomic(final_df, candidate_path)

    audit_result = audit_api_stage(
        stage_name=LOOKUP_STAGE,
        ledger_path=ledger_path,
        staging_dir=staging_dir,
        output_artifacts={"lookup_extended_parquet": candidate_path},
        declared_output_artifacts={"lookup_extended_parquet": output_path},
        rebuilders={
            "lookup_extended_parquet": lambda: _assemble_lookup_output(
                snapshot_df,
                staging_dir,
                ledger_path=ledger_path,
            ),
        },
        expected_stage_manifest_path=manifest_path,
        verify_rebuilders=verify_rebuilders,
    )
    if not audit_result.passed:
        raise RuntimeError(f"lookup stage audit failed: {audit_result.to_dict()}")
    _promote_candidate_output(candidate_path, output_path)

    returned_ric_cols = [f"{identifier_type}_returned_ric" for identifier_type in LOOKUP_IDENTIFIER_TYPES]
    rows_with_any_returned_ric = int(
        final_df.select(
            pl.any_horizontal([pl.col(column).is_not_null() for column in returned_ric_cols]).sum()
        ).item()
    )
    run_session_ids = _stage_run_session_ids(ledger_path)
    write_stage_completion_manifest(
        stage_name=LOOKUP_STAGE,
        manifest_path=manifest_path,
        input_artifacts={"lookup_snapshot_parquet": snapshot_parquet_path},
        output_artifacts={"lookup_extended_parquet": output_path},
        ledger_path=ledger_path,
        request_log_path=request_log_path,
        staging_dir=staging_dir,
        audit_result=audit_result,
        summary={
            "snapshot_row_count": int(snapshot_df.height),
            "lookup_item_count": item_count,
            "request_item_count": item_count,
            "planned_batch_count": planned_batch_count,
            "batch_plan_fingerprint": batch_plan_fingerprint,
            "batching_config": batching_config,
            "rows_with_any_returned_ric": rows_with_any_returned_ric,
            "run_session_id": run_session_id,
            "run_session_ids": run_session_ids,
        },
        metadata_source=metadata_source,
        cli_batching_args_ignored=cli_batching_args_ignored,
    )
    return {
        "refinitiv_ric_lookup_handoff_common_stock_extended_parquet": output_path,
        "refinitiv_lookup_stage_manifest_json": manifest_path,
    }


def _lookup_results_lazy_frame(
    staging_dir: Path,
    *,
    ledger_path: Path | str | None = None,
) -> pl.LazyFrame:
    staging_paths = _resolved_stage_output_paths(staging_dir=staging_dir, ledger_path=ledger_path)
    if not staging_paths:
        return pl.DataFrame(
            schema={
                "item_id": pl.Utf8,
                "bridge_row_id": pl.Utf8,
                "identifier_type": pl.Utf8,
                "lookup_input": pl.Utf8,
                "returned_ric": pl.Utf8,
                "returned_name": pl.Utf8,
                "returned_isin": pl.Utf8,
                "returned_cusip": pl.Utf8,
            }
        ).lazy()
    return pl.concat([pl.scan_parquet(path) for path in staging_paths], how="vertical_relaxed")


def _write_lookup_fetch_manifest(
    *,
    manifest_path: Path,
    ledger_path: Path,
    request_log_path: Path,
    staging_dir: Path,
    metadata_source: str,
    cli_batching_args_ignored: bool,
    batch_plan_fingerprint: str | None,
    batching_config: dict[str, Any] | None,
    request_item_count: int | None,
    batch_count: int | None,
    run_session_ids: list[str],
    items: list[RequestItem],
) -> Path:
    return write_stage_fetch_manifest(
        stage_name=LOOKUP_STAGE,
        manifest_path=manifest_path,
        staging_dir=staging_dir,
        ledger_path=ledger_path,
        request_log_path=request_log_path,
        batching_config=batching_config,
        request_item_count=request_item_count,
        batch_count=batch_count,
        run_session_ids=run_session_ids,
        summary={
            "batch_plan_fingerprint": batch_plan_fingerprint,
            "lookup_item_count": len(items),
        },
        metadata_source=metadata_source,
        cli_batching_args_ignored=cli_batching_args_ignored,
    )


def _raise_lookup_resume_compatibility_error(
    *,
    output_dir: Path,
    fetch_manifest_path: Path,
    manifest_path: Path,
    exc: LsegResumeCompatibilityError,
) -> None:
    fetch_manifest_exists = fetch_manifest_path.exists()
    guidance = [
        str(output_dir / "refinitiv_lookup_api_ledger.sqlite3"),
        str(output_dir / "refinitiv_lookup_api_requests.jsonl"),
        str(output_dir / "staging" / LOOKUP_STAGE),
        str(manifest_path),
        str(output_dir / "refinitiv_ric_lookup_handoff_common_stock_extended.parquet"),
    ]
    if fetch_manifest_exists:
        guidance.insert(0, str(fetch_manifest_path))
    explanation = (
        "The lookup stage already has persisted resume state from a different request universe or batching plan."
    )
    if fetch_manifest_exists:
        explanation += " Review the stored fetch manifest for the original batching metadata before rerunning."
    elif "batching_config_json" not in exc.existing_stage_meta:
        explanation += " No stored fetch manifest was found, and the ledger does not retain the full historical batching configuration."
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


def _stage_run_session_ids(ledger_path: Path) -> list[str]:
    return RequestLedger(ledger_path).run_session_ids()


def _resolved_stage_output_paths(
    *,
    staging_dir: Path,
    ledger_path: Path | str | None,
) -> list[Path]:
    if ledger_path is None:
        return sorted(staging_dir.glob("*.parquet"))
    ledger = RequestLedger(ledger_path)
    return [
        resolved_path
        for path in ledger.succeeded_stage_output_paths(stage=LOOKUP_STAGE)
        if (resolved_path := resolve_stage_output_path(path, staging_dir)).exists()
    ]


def _normalize_api_stage_mode(api_stage_mode: str) -> str:
    if api_stage_mode not in API_STAGE_MODES:
        raise ValueError(f"Unsupported api_stage_mode: {api_stage_mode}")
    return api_stage_mode

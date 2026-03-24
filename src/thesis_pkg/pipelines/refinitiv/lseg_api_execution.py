from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable

import polars as pl

from thesis_pkg.pipelines.refinitiv.lseg_api_common import (
    append_json_log,
    classify_error,
    daily_limit_likely_exhausted,
    error_details_for_exception,
    next_daily_resume_utc,
    retry_delay_seconds,
    should_treat_as_empty_result,
    write_parquet_atomic,
)
from thesis_pkg.pipelines.refinitiv.lseg_batching import (
    BatchDefinition,
    RequestItem,
    batch_items,
    split_batch,
)
from thesis_pkg.pipelines.refinitiv.lseg_ledger import (
    LEDGER_DEFERRED_DAILY_LIMIT,
    LEDGER_RETRYABLE_ERROR,
    RequestLedger,
    utc_now,
)
from thesis_pkg.pipelines.refinitiv.lseg_provider import LsegDataProvider


@dataclass(frozen=True)
class ApiStageRunResult:
    staging_dir: Path
    ledger_path: Path
    request_log_path: Path
    run_session_id: str


def run_api_batches(
    *,
    stage: str,
    items: list[RequestItem],
    output_dir: Path,
    ledger_path: Path,
    request_log_path: Path,
    provider: Any | None,
    provider_session_name: str,
    provider_config_name: str | None,
    provider_timeout_seconds: float | None,
    preflight_probe: bool,
    max_batch_size: int,
    min_seconds_between_requests: float,
    max_attempts: int,
    response_normalizer: Callable[[list[Any], pl.DataFrame], pl.DataFrame],
    lookup_normalizer: Callable[[Any], str | None],
    split_after_attempt: int = 2,
    retry_delay_seconds_fn: Callable[[int], float] = retry_delay_seconds,
) -> ApiStageRunResult:
    staging_dir = output_dir / "staging" / stage
    staging_dir.mkdir(parents=True, exist_ok=True)

    planned_batches = batch_items(items, max_batch_size=max_batch_size, unique_instrument_limit=True)
    ledger = RequestLedger(ledger_path)
    ledger.enqueue(items, planned_batches)

    requeued_stale = ledger.requeue_stale_running()
    if requeued_stale:
        append_json_log(
            request_log_path,
            {
                "event": "requeued_stale_running_batches",
                "stage": stage,
                "batch_count": requeued_stale,
            },
        )
    requeued_fatal_batches = ledger.requeue_known_fixable_fatal_batches(stage=stage)
    if requeued_fatal_batches:
        append_json_log(
            request_log_path,
            {
                "event": "requeued_known_fixable_fatal_batches",
                "stage": stage,
                "batch_count": requeued_fatal_batches,
            },
        )

    if not items:
        return ApiStageRunResult(
            staging_dir=staging_dir,
            ledger_path=ledger_path,
            request_log_path=request_log_path,
            run_session_id=ledger.run_session_id,
        )

    owns_provider = provider is None
    provider = provider or LsegDataProvider(
        session_name=provider_session_name,
        config_name=provider_config_name,
        request_timeout=provider_timeout_seconds,
    )
    if hasattr(provider, "open"):
        provider.open()

    if preflight_probe and items:
        probe_item = items[0]
        try:
            if hasattr(provider, "probe"):
                provider.probe(
                    universe=[probe_item.instrument],
                    fields=list(probe_item.fields),
                    parameters=probe_item.parameters,
                )
            else:
                provider.get_data(
                    universe=[probe_item.instrument],
                    fields=list(probe_item.fields),
                    parameters=probe_item.parameters,
                )
            append_json_log(
                request_log_path,
                {
                    "event": "preflight_probe_succeeded",
                    "stage": stage,
                    "instrument": probe_item.instrument,
                    "fields": list(probe_item.fields),
                    "parameters": probe_item.parameters,
                },
            )
        except Exception as exc:
            append_json_log(
                request_log_path,
                {
                    "event": "preflight_probe_failed",
                    "stage": stage,
                    "instrument": probe_item.instrument,
                    "fields": list(probe_item.fields),
                    "parameters": probe_item.parameters,
                    "exception_class": exc.__class__.__name__,
                    "exception_message": str(exc),
                },
            )
            if owns_provider and hasattr(provider, "close"):
                provider.close()
            raise

    last_request_completed_at = 0.0
    try:
        while True:
            batch = ledger.claim_next_batch(stage=stage)
            if batch is None:
                break
            batch_items_rows = ledger.fetch_items(batch)
            if not batch_items_rows:
                continue

            now_monotonic = time.monotonic()
            sleep_seconds = min_seconds_between_requests - (now_monotonic - last_request_completed_at)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

            universe = list(dict.fromkeys(item.instrument for item in batch_items_rows))
            try:
                response = provider.get_data(
                    universe=universe,
                    fields=list(batch.fields),
                    parameters=batch.parameters,
                )
                normalized_batch_df = response_normalizer(batch_items_rows, response.frame)
                staging_path = staging_dir / f"{batch.batch_id}.parquet"
                write_parquet_atomic(normalized_batch_df, staging_path)
                row_count_by_item_id = _row_count_by_item_id(normalized_batch_df)
                ledger.record_success(
                    batch_id=batch.batch_id,
                    row_count_by_item_id=row_count_by_item_id,
                    stage_output_path=staging_path,
                    response_fingerprint=response.metadata.fingerprint,
                    headers=response.metadata.headers,
                    status_code=response.metadata.status_code,
                    latency_ms=response.metadata.latency_ms,
                    rows_returned=int(normalized_batch_df.height),
                    response_bytes=response.metadata.response_bytes,
                )
                append_json_log(
                    request_log_path,
                    {
                        "event": "request_succeeded",
                        "stage": stage,
                        "batch_id": batch.batch_id,
                        "attempt_no": batch.attempt_count,
                        "item_count": len(batch_items_rows),
                        "unique_instrument_count": len(universe),
                        "fields": list(batch.fields),
                        "parameters": batch.parameters,
                        "status_code": response.metadata.status_code,
                        "latency_ms": response.metadata.latency_ms,
                        "rows_returned": int(normalized_batch_df.height),
                        "response_bytes": response.metadata.response_bytes,
                        "headers": response.metadata.headers,
                    },
                )
                if daily_limit_likely_exhausted(response.metadata.headers):
                    resume_at = next_daily_resume_utc(utc_now())
                    ledger.defer_pending_stage_items(
                        stage=stage,
                        next_eligible_at_utc=resume_at,
                        reason="soft_stop_daily_limit_threshold",
                    )
                    break
            except Exception as exc:
                error_details = error_details_for_exception(exc)
                status_code = error_details["status_code"]
                headers = error_details["headers"]
                response_bytes = error_details["response_bytes"]
                if should_treat_as_empty_result(
                    error_kind=error_details["error_kind"],
                    unresolved_identifiers=error_details["unresolved_identifiers"],
                    universe=universe,
                    attempt_no=batch.attempt_count,
                    max_attempts=max_attempts,
                    normalizer=lookup_normalizer,
                ):
                    normalized_batch_df = response_normalizer(batch_items_rows, pl.DataFrame())
                    staging_path = staging_dir / f"{batch.batch_id}.parquet"
                    write_parquet_atomic(normalized_batch_df, staging_path)
                    row_count_by_item_id = _row_count_by_item_id(normalized_batch_df)
                    ledger.record_success(
                        batch_id=batch.batch_id,
                        row_count_by_item_id=row_count_by_item_id,
                        stage_output_path=staging_path,
                        response_fingerprint=None,
                        headers=headers,
                        status_code=status_code,
                        latency_ms=None,
                        rows_returned=int(normalized_batch_df.height),
                        response_bytes=response_bytes,
                    )
                    append_json_log(
                        request_log_path,
                        {
                            "event": "request_unresolved_identifiers_treated_as_empty",
                            "stage": stage,
                            "batch_id": batch.batch_id,
                            "attempt_no": batch.attempt_count,
                            "item_count": len(batch_items_rows),
                            "unique_instrument_count": len(universe),
                            "fields": list(batch.fields),
                            "parameters": batch.parameters,
                            "status_code": status_code,
                            "rows_returned": int(normalized_batch_df.height),
                            "response_bytes": response_bytes,
                            "headers": headers,
                            "exception_class": exc.__class__.__name__,
                            "exception_message": str(exc),
                            "error_kind": error_details["error_kind"],
                            "unresolved_identifiers": error_details["unresolved_identifiers"],
                        },
                    )
                    continue

                policy = classify_error(
                    exc,
                    batch_size=len(universe),
                    attempt_no=batch.attempt_count,
                    max_attempts=max_attempts,
                    split_after_attempt=split_after_attempt,
                )
                append_json_log(
                    request_log_path,
                    {
                        "event": "request_failed",
                        "stage": stage,
                        "batch_id": batch.batch_id,
                        "attempt_no": batch.attempt_count,
                        "item_count": len(batch_items_rows),
                        "unique_instrument_count": len(universe),
                        "fields": list(batch.fields),
                        "parameters": batch.parameters,
                        "status_code": status_code,
                        "response_bytes": response_bytes,
                        "headers": headers,
                        "exception_class": exc.__class__.__name__,
                        "exception_message": str(exc),
                        "error_kind": error_details["error_kind"],
                        "unresolved_identifiers": error_details["unresolved_identifiers"],
                        "policy": policy["state"],
                        "split_batch": policy["split_batch"],
                    },
                )
                if policy["defer_stage"]:
                    resume_at = next_daily_resume_utc(utc_now())
                    ledger.record_error(
                        batch_id=batch.batch_id,
                        next_state=LEDGER_DEFERRED_DAILY_LIMIT,
                        error_message=str(exc),
                        headers=headers,
                        status_code=status_code,
                        latency_ms=None,
                        response_bytes=response_bytes,
                        next_eligible_at_utc=resume_at,
                        exception_class=exc.__class__.__name__,
                    )
                    ledger.defer_pending_stage_items(
                        stage=stage,
                        next_eligible_at_utc=resume_at,
                        reason="daily_limit_exhausted",
                    )
                    break
                if policy["split_batch"] and len(batch.item_ids) > 1:
                    child_batches = _build_child_batches(batch, batch_items_rows)
                    ledger.split_batch(
                        parent_batch_id=batch.batch_id,
                        child_batches=child_batches,
                        reason=str(exc),
                    )
                    append_json_log(
                        request_log_path,
                        {
                            "event": "request_batch_split",
                            "stage": stage,
                            "batch_id": batch.batch_id,
                            "attempt_no": batch.attempt_count,
                            "child_batch_ids": [child.batch_id for child in child_batches],
                            "exception_class": exc.__class__.__name__,
                            "exception_message": str(exc),
                        },
                    )
                    continue
                next_eligible_at_utc = (
                    utc_now() + dt.timedelta(seconds=retry_delay_seconds_fn(batch.attempt_count))
                    if policy["state"] == LEDGER_RETRYABLE_ERROR
                    else None
                )
                ledger.record_error(
                    batch_id=batch.batch_id,
                    next_state=policy["state"],
                    error_message=str(exc),
                    headers=headers,
                    status_code=status_code,
                    latency_ms=None,
                    response_bytes=response_bytes,
                    next_eligible_at_utc=next_eligible_at_utc,
                    exception_class=exc.__class__.__name__,
                )
                if policy["stop_stage"]:
                    raise
            finally:
                last_request_completed_at = time.monotonic()
    finally:
        if owns_provider and hasattr(provider, "close"):
            provider.close()

    return ApiStageRunResult(
        staging_dir=staging_dir,
        ledger_path=ledger_path,
        request_log_path=request_log_path,
        run_session_id=ledger.run_session_id,
    )


def _row_count_by_item_id(frame: pl.DataFrame) -> dict[str, int]:
    if frame.height == 0 or "item_id" not in frame.columns:
        return {}
    return {
        row["item_id"]: int(row["len"])
        for row in frame.group_by("item_id").len().to_dicts()
    }


def _build_child_batches(batch: Any, batch_items_rows: list[Any]) -> list[BatchDefinition]:
    parent = BatchDefinition(
        batch_id=batch.batch_id,
        stage=batch.stage,
        batch_key=batch.batch_key,
        fields=batch.fields,
        parameters=dict(batch.parameters),
        item_ids=tuple(item.item_id for item in batch_items_rows),
        instruments=tuple(item.instrument for item in batch_items_rows),
    )
    return split_batch(parent)

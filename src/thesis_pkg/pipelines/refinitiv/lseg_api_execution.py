from __future__ import annotations

import datetime as dt
import multiprocessing as mp
import pickle
import queue
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


ProviderFactory = Callable[[], Any]


@dataclass(frozen=True)
class ApiStageRunResult:
    staging_dir: Path
    ledger_path: Path
    request_log_path: Path
    run_session_id: str


@dataclass(frozen=True)
class _WorkerConfig:
    stage: str
    staging_dir: Path
    ledger_path: Path
    request_log_path: Path
    run_session_id: str
    provider_session_name: str
    provider_config_name: str | None
    provider_timeout_seconds: float | None
    max_attempts: int
    response_normalizer: Callable[[list[Any], pl.DataFrame], pl.DataFrame]
    lookup_normalizer: Callable[[Any], str | None]
    split_after_attempt: int
    retry_delay_seconds_fn: Callable[[int], float]
    min_seconds_between_requests: float
    min_seconds_between_request_starts_total: float | None
    provider_factory: ProviderFactory | None
    planned_batch_metrics: dict[str, dict[str, Any]] | None = None
    idle_poll_seconds: float = 0.1
    claim_lead_seconds: float = 0.2


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
    max_workers: int = 1,
    min_seconds_between_request_starts_total: float | None = None,
    provider_factory: ProviderFactory | None = None,
    planned_batches: list[BatchDefinition] | None = None,
    batch_plan_fingerprint: str | None = None,
    planned_batch_metrics: dict[str, dict[str, Any]] | None = None,
) -> ApiStageRunResult:
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    if max_workers > 1 and min_seconds_between_request_starts_total is None:
        raise ValueError(
            "min_seconds_between_request_starts_total must be set explicitly when max_workers > 1"
        )
    if max_workers > 1 and provider is not None:
        raise ValueError("provider= is supported only when max_workers=1")

    staging_dir = output_dir / "staging" / stage
    staging_dir.mkdir(parents=True, exist_ok=True)

    ledger = RequestLedger(ledger_path)
    if batch_plan_fingerprint is not None:
        ledger.ensure_stage_meta_value(
            stage=stage,
            key="batch_plan_fingerprint",
            value=batch_plan_fingerprint,
        )
    item_by_id = {item.item_id: item for item in items}
    planned_batches = (
        planned_batches
        if planned_batches is not None
        else batch_items(items, max_batch_size=max_batch_size, unique_instrument_limit=True)
    )
    _validate_planned_batches(
        stage=stage,
        items_by_id=item_by_id,
        planned_batches=planned_batches,
        max_batch_size=max_batch_size,
    )
    ledger.enqueue(items, planned_batches)
    if batch_plan_fingerprint is not None:
        _append_log(
            request_log_path,
            ledger,
            {
                "event": "batch_plan_created",
                "stage": stage,
                "batch_plan_fingerprint": batch_plan_fingerprint,
                "planned_batch_count": len(planned_batches),
                "planned_item_count": len(items),
            },
        )

    requeued_stale = ledger.requeue_stale_running()
    if requeued_stale:
        _append_log(
            request_log_path,
            ledger,
            {
                "event": "requeued_stale_running_batches",
                "stage": stage,
                "batch_count": requeued_stale,
            },
        )
    requeued_fatal_batches = ledger.requeue_known_fixable_fatal_batches(stage=stage)
    if requeued_fatal_batches:
        _append_log(
            request_log_path,
            ledger,
            {
                "event": "requeued_known_fixable_fatal_batches",
                "stage": stage,
                "batch_count": requeued_fatal_batches,
            },
        )

    worker_config = _WorkerConfig(
        stage=stage,
        staging_dir=staging_dir,
        ledger_path=ledger_path,
        request_log_path=request_log_path,
        run_session_id=ledger.run_session_id,
        provider_session_name=provider_session_name,
        provider_config_name=provider_config_name,
        provider_timeout_seconds=provider_timeout_seconds,
        max_attempts=max_attempts,
        response_normalizer=response_normalizer,
        lookup_normalizer=lookup_normalizer,
        split_after_attempt=split_after_attempt,
        retry_delay_seconds_fn=retry_delay_seconds_fn,
        min_seconds_between_requests=min_seconds_between_requests,
        min_seconds_between_request_starts_total=min_seconds_between_request_starts_total,
        provider_factory=provider_factory,
        planned_batch_metrics=planned_batch_metrics,
    )

    ledger.clear_runtime_halt(stage=stage)
    ledger.reset_request_start_scheduler(
        stage=stage,
        initial_delay_seconds=worker_config.claim_lead_seconds if max_workers > 1 else 0.0,
    )

    if not items:
        return ApiStageRunResult(
            staging_dir=staging_dir,
            ledger_path=ledger_path,
            request_log_path=request_log_path,
            run_session_id=ledger.run_session_id,
        )

    if max_workers > 1:
        _validate_concurrent_worker_inputs(worker_config)

    if preflight_probe and items:
        _run_preflight_probe(
            stage=stage,
            request_log_path=request_log_path,
            ledger=ledger,
            probe_item=items[0],
            worker_config=worker_config,
            provider=provider,
        )

    if max_workers == 1:
        sequential_ledger = RequestLedger(
            ledger_path,
            run_session_id=ledger.run_session_id,
            worker_id=ledger.worker_id,
        )
        owns_provider = provider is None
        provider_instance = provider or _build_provider(worker_config)
        if hasattr(provider_instance, "open"):
            provider_instance.open()
        try:
            _run_worker_loop(
                worker_config=worker_config,
                ledger=sequential_ledger,
                provider=provider_instance,
                concurrent_mode=False,
            )
        finally:
            if owns_provider and hasattr(provider_instance, "close"):
                provider_instance.close()
    else:
        ctx = mp.get_context("spawn")
        error_queue: Any = ctx.Queue()
        processes: list[mp.Process] = []
        for worker_index in range(max_workers):
            worker_name = f"worker-{worker_index + 1}"
            process = ctx.Process(
                target=_run_api_batch_worker_process,
                args=(worker_config, worker_name, error_queue),
                name=f"lseg-{stage}-{worker_name}",
            )
            process.start()
            processes.append(process)

        worker_errors: list[dict[str, str]] = []
        try:
            for process in processes:
                process.join()
        finally:
            for process in processes:
                if process.is_alive():
                    process.join(timeout=0.1)

        while True:
            try:
                worker_errors.append(error_queue.get_nowait())
            except queue.Empty:
                break

        for process in processes:
            if process.exitcode not in (0, None):
                if not any(error.get("process_name") == process.name for error in worker_errors):
                    worker_errors.append(
                        {
                            "process_name": process.name,
                            "worker_id": process.name,
                            "exception_class": "WorkerProcessError",
                            "exception_message": f"worker exited with code {process.exitcode}",
                        }
                    )

        if worker_errors:
            first_error = worker_errors[0]
            raise RuntimeError(
                f"concurrent LSEG worker failed ({first_error['worker_id']}): "
                f"{first_error['exception_class']}: {first_error['exception_message']}"
            )

    return ApiStageRunResult(
        staging_dir=staging_dir,
        ledger_path=ledger_path,
        request_log_path=request_log_path,
        run_session_id=ledger.run_session_id,
    )


def _run_worker_loop(
    *,
    worker_config: _WorkerConfig,
    ledger: RequestLedger,
    provider: Any,
    concurrent_mode: bool,
) -> None:
    last_request_completed_at = 0.0
    while True:
        scheduled_start_utc: dt.datetime | None = None
        batch_items_rows: list[Any] | None = None
        universe: list[str] | None = None
        if concurrent_mode:
            reservation = ledger.reserve_next_request_start(
                stage=worker_config.stage,
                min_seconds_between_request_starts_total=float(
                    worker_config.min_seconds_between_request_starts_total or 0.0
                ),
                idle_poll_seconds=worker_config.idle_poll_seconds,
            )
            if reservation.status in {"halted", "drained"}:
                break
            if reservation.status == "wait":
                if reservation.wait_seconds > 0:
                    time.sleep(reservation.wait_seconds)
                continue
            if reservation.status != "reserved":
                raise RuntimeError(f"unsupported request-start reservation status: {reservation.status}")
            assert reservation.scheduled_start_epoch_utc is not None
            claim_ready_epoch = max(
                time.time(),
                reservation.scheduled_start_epoch_utc - worker_config.claim_lead_seconds,
            )
            sleep_seconds = claim_ready_epoch - time.time()
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            scheduled_start_utc = dt.datetime.fromtimestamp(
                reservation.scheduled_start_epoch_utc,
                tz=dt.timezone.utc,
            )
            if ledger.is_runtime_halted(stage=worker_config.stage):
                break
            batch = ledger.claim_next_batch(stage=worker_config.stage)
            if batch is None:
                continue
            batch_items_rows = ledger.fetch_items(batch)
            if not batch_items_rows:
                continue
            universe = list(dict.fromkeys(item.instrument for item in batch_items_rows))
            final_wait_seconds = reservation.scheduled_start_epoch_utc - time.time()
            if final_wait_seconds > 0:
                time.sleep(final_wait_seconds)
        else:
            batch = ledger.claim_next_batch(stage=worker_config.stage)
            if batch is None:
                break
            now_monotonic = time.monotonic()
            sleep_seconds = worker_config.min_seconds_between_requests - (now_monotonic - last_request_completed_at)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            scheduled_start_utc = utc_now()

        try:
            _execute_claimed_batch(
                worker_config=worker_config,
                ledger=ledger,
                provider=provider,
                batch=batch,
                scheduled_start_utc=scheduled_start_utc,
                batch_items_rows=batch_items_rows,
                universe=universe,
            )
        finally:
            if not concurrent_mode:
                last_request_completed_at = time.monotonic()


def _execute_claimed_batch(
    *,
    worker_config: _WorkerConfig,
    ledger: RequestLedger,
    provider: Any,
    batch: Any,
    scheduled_start_utc: dt.datetime | None,
    batch_items_rows: list[Any] | None = None,
    universe: list[str] | None = None,
) -> None:
    batch_items_rows = ledger.fetch_items(batch) if batch_items_rows is None else batch_items_rows
    if not batch_items_rows:
        return

    universe = list(dict.fromkeys(item.instrument for item in batch_items_rows)) if universe is None else universe
    actual_request_started_utc = utc_now()
    request_started_recorded = False
    try:
        response = provider.get_data(
            universe=universe,
            fields=list(batch.fields),
            parameters=batch.parameters,
        )
        ledger.record_request_started(batch_id=batch.batch_id)
        request_started_recorded = True
        normalized_batch_df = worker_config.response_normalizer(batch_items_rows, response.frame)
        staging_path = worker_config.staging_dir / f"{batch.batch_id}.parquet"
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
        _append_log(
            worker_config.request_log_path,
            ledger,
            {
                "event": "request_succeeded",
                "stage": worker_config.stage,
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
                "scheduled_start_utc": _format_utc_timestamp(scheduled_start_utc),
                "actual_request_started_utc": _format_utc_timestamp(actual_request_started_utc),
                **_planned_metrics_payload(worker_config, batch.batch_id),
            },
        )
        if daily_limit_likely_exhausted(response.metadata.headers):
            resume_at = next_daily_resume_utc(utc_now())
            ledger.defer_pending_stage_items(
                stage=worker_config.stage,
                next_eligible_at_utc=resume_at,
                reason="soft_stop_daily_limit_threshold",
            )
            ledger.halt_runtime_work(stage=worker_config.stage, reason="soft_stop_daily_limit_threshold")
            _append_log(
                worker_config.request_log_path,
                ledger,
                {
                    "event": "runtime_halt_set",
                    "stage": worker_config.stage,
                    "reason": "soft_stop_daily_limit_threshold",
                    "scheduled_start_utc": _format_utc_timestamp(scheduled_start_utc),
                },
            )
            return
    except Exception as exc:
        if not request_started_recorded:
            ledger.record_request_started(batch_id=batch.batch_id)
            request_started_recorded = True
        error_details = error_details_for_exception(exc)
        status_code = error_details["status_code"]
        headers = error_details["headers"]
        response_bytes = error_details["response_bytes"]
        if should_treat_as_empty_result(
            error_kind=error_details["error_kind"],
            unresolved_identifiers=error_details["unresolved_identifiers"],
            universe=universe,
            attempt_no=batch.attempt_count,
            max_attempts=worker_config.max_attempts,
            normalizer=worker_config.lookup_normalizer,
        ):
            normalized_batch_df = worker_config.response_normalizer(batch_items_rows, pl.DataFrame())
            staging_path = worker_config.staging_dir / f"{batch.batch_id}.parquet"
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
            _append_log(
                worker_config.request_log_path,
                ledger,
                {
                    "event": "request_unresolved_identifiers_treated_as_empty",
                    "stage": worker_config.stage,
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
                    "scheduled_start_utc": _format_utc_timestamp(scheduled_start_utc),
                    "actual_request_started_utc": _format_utc_timestamp(actual_request_started_utc),
                    **_planned_metrics_payload(worker_config, batch.batch_id),
                },
            )
            return

        policy = classify_error(
            exc,
            batch_size=len(universe),
            attempt_no=batch.attempt_count,
            max_attempts=worker_config.max_attempts,
            split_after_attempt=worker_config.split_after_attempt,
        )
        _append_log(
            worker_config.request_log_path,
            ledger,
            {
                "event": "request_failed",
                "stage": worker_config.stage,
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
                "scheduled_start_utc": _format_utc_timestamp(scheduled_start_utc),
                "actual_request_started_utc": _format_utc_timestamp(actual_request_started_utc),
                **_planned_metrics_payload(worker_config, batch.batch_id),
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
                stage=worker_config.stage,
                next_eligible_at_utc=resume_at,
                reason="daily_limit_exhausted",
            )
            ledger.halt_runtime_work(stage=worker_config.stage, reason="daily_limit_exhausted")
            return
        if policy["split_batch"] and len(batch.item_ids) > 1:
            child_batches = _build_child_batches(batch, batch_items_rows)
            ledger.split_batch(
                parent_batch_id=batch.batch_id,
                child_batches=child_batches,
                reason=str(exc),
            )
            _append_log(
                worker_config.request_log_path,
                ledger,
                {
                    "event": "request_batch_split",
                    "stage": worker_config.stage,
                    "batch_id": batch.batch_id,
                    "attempt_no": batch.attempt_count,
                    "child_batch_ids": [child.batch_id for child in child_batches],
                    "exception_class": exc.__class__.__name__,
                    "exception_message": str(exc),
                    "scheduled_start_utc": _format_utc_timestamp(scheduled_start_utc),
                    "actual_request_started_utc": _format_utc_timestamp(actual_request_started_utc),
                    **_planned_metrics_payload(worker_config, batch.batch_id),
                },
            )
            return
        next_eligible_at_utc = (
            utc_now() + dt.timedelta(seconds=worker_config.retry_delay_seconds_fn(batch.attempt_count))
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
            ledger.halt_runtime_work(stage=worker_config.stage, reason=str(exc))
            raise


def _run_api_batch_worker_process(
    worker_config: _WorkerConfig,
    worker_name: str,
    error_queue: Any,
) -> None:
    worker_id = f"{worker_name}:{mp.current_process().pid}"
    ledger = RequestLedger(
        worker_config.ledger_path,
        run_session_id=worker_config.run_session_id,
        worker_id=worker_id,
    )
    provider = _build_provider(worker_config)
    try:
        if hasattr(provider, "open"):
            provider.open()
        _run_worker_loop(
            worker_config=worker_config,
            ledger=ledger,
            provider=provider,
            concurrent_mode=True,
        )
    except Exception as exc:
        _append_log(
            worker_config.request_log_path,
            ledger,
            {
                "event": "worker_process_failed",
                "stage": worker_config.stage,
                "worker_process_name": worker_name,
                "exception_class": exc.__class__.__name__,
                "exception_message": str(exc),
            },
        )
        error_queue.put(
            {
                "process_name": mp.current_process().name,
                "worker_id": worker_id,
                "exception_class": exc.__class__.__name__,
                "exception_message": str(exc),
            }
        )
        raise
    finally:
        if hasattr(provider, "close"):
            provider.close()


def _run_preflight_probe(
    *,
    stage: str,
    request_log_path: Path,
    ledger: RequestLedger,
    probe_item: RequestItem,
    worker_config: _WorkerConfig,
    provider: Any | None,
) -> None:
    owns_provider = provider is None
    provider_instance = provider or _build_provider(worker_config)
    if hasattr(provider_instance, "open"):
        provider_instance.open()
    try:
        if hasattr(provider_instance, "probe"):
            provider_instance.probe(
                universe=[probe_item.instrument],
                fields=list(probe_item.fields),
                parameters=probe_item.parameters,
            )
        else:
            provider_instance.get_data(
                universe=[probe_item.instrument],
                fields=list(probe_item.fields),
                parameters=probe_item.parameters,
            )
        _append_log(
            request_log_path,
            ledger,
            {
                "event": "preflight_probe_succeeded",
                "stage": stage,
                "instrument": probe_item.instrument,
                "fields": list(probe_item.fields),
                "parameters": probe_item.parameters,
            },
        )
    except Exception as exc:
        _append_log(
            request_log_path,
            ledger,
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
        raise
    finally:
        if owns_provider and hasattr(provider_instance, "close"):
            provider_instance.close()


def _build_provider(worker_config: _WorkerConfig) -> Any:
    if worker_config.provider_factory is not None:
        return worker_config.provider_factory()
    return LsegDataProvider(
        session_name=worker_config.provider_session_name,
        config_name=worker_config.provider_config_name,
        request_timeout=worker_config.provider_timeout_seconds,
    )


def _validate_concurrent_worker_inputs(worker_config: _WorkerConfig) -> None:
    for name, value in (
        ("response_normalizer", worker_config.response_normalizer),
        ("lookup_normalizer", worker_config.lookup_normalizer),
        ("retry_delay_seconds_fn", worker_config.retry_delay_seconds_fn),
        ("provider_factory", worker_config.provider_factory),
    ):
        if value is None:
            continue
        try:
            pickle.dumps(value)
        except Exception as exc:
            raise TypeError(f"{name} must be picklable when max_workers > 1") from exc


def _append_log(request_log_path: Path, ledger: RequestLedger, payload: dict[str, Any]) -> None:
    append_json_log(
        request_log_path,
        {
            **payload,
            "run_session_id": ledger.run_session_id,
            "worker_id": ledger.worker_id,
        },
    )


def _planned_metrics_payload(worker_config: _WorkerConfig, batch_id: str) -> dict[str, Any]:
    if worker_config.planned_batch_metrics is None:
        return {}
    return dict(worker_config.planned_batch_metrics.get(batch_id, {}))


def _format_utc_timestamp(value: dt.datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


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


def _validate_planned_batches(
    *,
    stage: str,
    items_by_id: dict[str, RequestItem],
    planned_batches: list[BatchDefinition],
    max_batch_size: int,
) -> None:
    seen_item_ids: set[str] = set()
    for batch in planned_batches:
        if batch.stage != stage:
            raise ValueError(f"planned batch stage mismatch: expected {stage!r}, got {batch.stage!r}")
        if not batch.item_ids:
            raise ValueError(f"planned batch {batch.batch_id} is empty")
        unknown_item_ids = [item_id for item_id in batch.item_ids if item_id not in items_by_id]
        if unknown_item_ids:
            raise ValueError(
                f"planned batch {batch.batch_id} references unknown item_ids: {unknown_item_ids}"
            )
        unique_instruments = {
            items_by_id[item_id].instrument
            for item_id in batch.item_ids
        }
        if len(unique_instruments) > max_batch_size:
            raise ValueError(
                f"planned batch {batch.batch_id} exceeds max_batch_size={max_batch_size} "
                f"with {len(unique_instruments)} unique instruments"
            )
        for item_id in batch.item_ids:
            if item_id in seen_item_ids:
                raise ValueError(f"planned item_id {item_id!r} appears in multiple batches")
            seen_item_ids.add(item_id)
    missing_item_ids = sorted(set(items_by_id) - seen_item_ids)
    if missing_item_ids:
        raise ValueError(f"planned batches did not cover all items: {missing_item_ids}")

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import random
import time
from typing import Any

import polars as pl

from thesis_pkg.pipelines.refinitiv.lseg_batching import (
    RequestItem,
    batch_items,
    split_batch,
    stable_hash_id,
)
from thesis_pkg.pipelines.refinitiv.lseg_ledger import (
    LEDGER_DEFERRED_DAILY_LIMIT,
    LEDGER_FATAL_ERROR,
    LEDGER_RETRYABLE_ERROR,
    RequestLedger,
    utc_now,
)
from thesis_pkg.pipelines.refinitiv.lseg_provider import (
    LsegRequestError,
    LsegDataProvider,
    classify_lseg_error_message,
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


def run_refinitiv_step1_lookup_api_pipeline(
    *,
    snapshot_parquet_path: Path | str,
    output_dir: Path | str,
    provider: Any | None = None,
    ledger_path: Path | str | None = None,
    request_log_path: Path | str | None = None,
    max_batch_size: int = 25,
    min_seconds_between_requests: float = 2.0,
    max_attempts: int = 4,
) -> dict[str, Path]:
    snapshot_parquet_path = Path(snapshot_parquet_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir / "staging" / LOOKUP_STAGE
    staging_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = Path(ledger_path) if ledger_path is not None else output_dir / "refinitiv_lookup_api_ledger.sqlite3"
    request_log_path = (
        Path(request_log_path)
        if request_log_path is not None
        else output_dir / "refinitiv_lookup_api_requests.jsonl"
    )

    snapshot_df = _read_lookup_snapshot(snapshot_parquet_path)
    items = _build_lookup_items(snapshot_df)
    ledger = RequestLedger(ledger_path)
    ledger.enqueue(
        items,
        batch_items(items, max_batch_size=max_batch_size, unique_instrument_limit=True),
    )
    ledger.requeue_stale_running()
    requeued_fatal_batches = ledger.requeue_known_fixable_fatal_batches(stage=LOOKUP_STAGE)
    if requeued_fatal_batches:
        _append_json_log(
            request_log_path,
            {
                "event": "requeued_known_fixable_fatal_batches",
                "stage": LOOKUP_STAGE,
                "batch_count": requeued_fatal_batches,
            },
        )

    owns_provider = provider is None
    provider = provider or LsegDataProvider()
    if hasattr(provider, "open"):
        provider.open()

    last_request_completed_at = 0.0
    try:
        while True:
            batch = ledger.claim_next_batch(stage=LOOKUP_STAGE)
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
                    fields=list(LOOKUP_FIELDS),
                    parameters={},
                )
                normalized_batch_df = _normalize_lookup_batch_response(
                    batch_items_rows,
                    response.frame,
                )
                staging_path = staging_dir / f"{batch.batch_id}.parquet"
                _write_parquet_atomic(normalized_batch_df, staging_path)
                row_count_by_item_id = {
                    row["item_id"]: 1
                    for row in normalized_batch_df.select("item_id").to_dicts()
                }
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
                _append_json_log(
                    request_log_path,
                    {
                        "event": "request_succeeded",
                        "stage": LOOKUP_STAGE,
                        "batch_id": batch.batch_id,
                        "attempt_no": batch.attempt_count,
                        "item_count": len(batch_items_rows),
                        "unique_instrument_count": len(universe),
                        "fields": list(LOOKUP_FIELDS),
                        "parameters": {},
                        "status_code": response.metadata.status_code,
                        "latency_ms": response.metadata.latency_ms,
                        "rows_returned": int(normalized_batch_df.height),
                        "response_bytes": response.metadata.response_bytes,
                        "headers": response.metadata.headers,
                    },
                )
                if _daily_limit_likely_exhausted(response.metadata.headers):
                    resume_at = _next_daily_resume_utc(utc_now())
                    ledger.defer_pending_stage_items(
                        stage=LOOKUP_STAGE,
                        next_eligible_at_utc=resume_at,
                        reason="soft_stop_daily_limit_threshold",
                    )
                    break
            except Exception as exc:
                error_details = _error_details(exc)
                status_code = error_details["status_code"]
                headers = error_details["headers"]
                response_bytes = error_details["response_bytes"]
                if _should_treat_as_empty_result(
                    error_kind=error_details["error_kind"],
                    unresolved_identifiers=error_details["unresolved_identifiers"],
                    universe=universe,
                    attempt_no=batch.attempt_count,
                    max_attempts=max_attempts,
                ):
                    normalized_batch_df = _normalize_lookup_batch_response(batch_items_rows, pl.DataFrame())
                    staging_path = staging_dir / f"{batch.batch_id}.parquet"
                    _write_parquet_atomic(normalized_batch_df, staging_path)
                    row_count_by_item_id = {
                        row["item_id"]: 1
                        for row in normalized_batch_df.select("item_id").to_dicts()
                    }
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
                    _append_json_log(
                        request_log_path,
                        {
                            "event": "request_unresolved_identifiers_treated_as_empty",
                            "stage": LOOKUP_STAGE,
                            "batch_id": batch.batch_id,
                            "attempt_no": batch.attempt_count,
                            "item_count": len(batch_items_rows),
                            "unique_instrument_count": len(universe),
                            "fields": list(LOOKUP_FIELDS),
                            "parameters": {},
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
                policy = _classify_error(
                    exc,
                    batch_size=len(universe),
                    attempt_no=batch.attempt_count,
                    max_attempts=max_attempts,
                )
                _append_json_log(
                    request_log_path,
                    {
                        "event": "request_failed",
                        "stage": LOOKUP_STAGE,
                        "batch_id": batch.batch_id,
                        "attempt_no": batch.attempt_count,
                        "item_count": len(batch_items_rows),
                        "unique_instrument_count": len(universe),
                        "fields": list(LOOKUP_FIELDS),
                        "parameters": {},
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
                    resume_at = _next_daily_resume_utc(utc_now())
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
                        stage=LOOKUP_STAGE,
                        next_eligible_at_utc=resume_at,
                        reason="daily_limit_exhausted",
                    )
                    break
                if policy["split_batch"] and len(batch.item_ids) > 1:
                    ledger.split_batch(
                        parent_batch_id=batch.batch_id,
                        child_batches=split_batch(batch),
                        reason=str(exc),
                    )
                    continue
                next_eligible_at_utc = (
                    utc_now() + dt.timedelta(seconds=_retry_delay_seconds(batch.attempt_count))
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

    final_df = _assemble_lookup_output(snapshot_df, staging_dir)
    output_path = output_dir / "refinitiv_ric_lookup_handoff_common_stock_extended.parquet"
    final_df.write_parquet(output_path, compression="zstd")
    return {
        "refinitiv_ric_lookup_handoff_common_stock_extended_parquet": output_path,
        "refinitiv_lookup_api_ledger_sqlite3": ledger_path,
        "refinitiv_lookup_api_requests_jsonl": request_log_path,
    }


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


def _assemble_lookup_output(snapshot_df: pl.DataFrame, staging_dir: Path) -> pl.DataFrame:
    staging_paths = sorted(staging_dir.glob("*.parquet"))
    results_df = (
        pl.concat([pl.read_parquet(path) for path in staging_paths], how="vertical_relaxed")
        if staging_paths
        else pl.DataFrame(
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
        )
    )
    results_df = results_df.unique(subset=["item_id"], keep="last", maintain_order=True)

    base_columns = list(RIC_LOOKUP_EXTENDED_BASE_COLUMNS) + [
        f"{identifier_type}_lookup_input" for identifier_type in LOOKUP_IDENTIFIER_TYPES
    ]
    assembled_df = snapshot_df.select(base_columns)
    for identifier_type in LOOKUP_IDENTIFIER_TYPES:
        identifier_df = (
            results_df.filter(pl.col("identifier_type") == identifier_type)
            .select(
                "bridge_row_id",
                pl.col("returned_ric").alias(f"{identifier_type}_returned_ric"),
                pl.col("returned_name").alias(f"{identifier_type}_returned_name"),
                pl.col("returned_isin").alias(f"{identifier_type}_returned_isin"),
                pl.col("returned_cusip").alias(f"{identifier_type}_returned_cusip"),
            )
            .unique(subset=["bridge_row_id"], keep="last", maintain_order=True)
        )
        assembled_df = assembled_df.join(identifier_df, on="bridge_row_id", how="left")

    for identifier_type in LOOKUP_IDENTIFIER_TYPES:
        for suffix in ("returned_ric", "returned_name", "returned_isin", "returned_cusip"):
            column_name = f"{identifier_type}_{suffix}"
            if column_name not in assembled_df.columns:
                assembled_df = assembled_df.with_columns(pl.lit(None, dtype=pl.Utf8).alias(column_name))
    return _normalize_resolution_input_df(assembled_df)


def _standardize_field_frame(
    frame: pl.DataFrame,
    *,
    expected_fields: tuple[str, ...],
    field_aliases: dict[str, tuple[str, ...]] | None = None,
) -> pl.DataFrame:
    if frame.height == 0:
        return pl.DataFrame(schema={"instrument": pl.Utf8, **{field: pl.Utf8 for field in expected_fields}})

    field_aliases = field_aliases or {}
    columns = list(frame.columns)
    rename_map: dict[str, str] = {}
    instrument_column = next(
        (name for name in ("instrument", "Instrument") if name in columns),
        None,
    )
    if instrument_column is None and columns:
        instrument_column = columns[0]
    if instrument_column is not None and instrument_column != "instrument":
        rename_map[instrument_column] = "instrument"
    renamed = frame.rename(rename_map) if rename_map else frame
    current_columns = set(renamed.columns)

    if all(field in current_columns for field in expected_fields):
        return renamed.select(["instrument", *expected_fields])

    exprs = [pl.col("instrument").cast(pl.Utf8, strict=False).alias("instrument")]
    for idx, field in enumerate(expected_fields):
        candidate_columns = (field, *field_aliases.get(field, ()))
        matched_column = next((name for name in candidate_columns if name in current_columns), None)
        if matched_column is not None:
            exprs.append(pl.col(matched_column).cast(pl.Utf8, strict=False).alias(field))
            continue
        available_value_columns = [name for name in renamed.columns if name != "instrument"]
        if idx < len(available_value_columns):
            exprs.append(pl.col(available_value_columns[idx]).cast(pl.Utf8, strict=False).alias(field))
            continue
        exprs.append(pl.lit(None, dtype=pl.Utf8).alias(field))
    return renamed.select(exprs)


def _write_parquet_atomic(df: pl.DataFrame, output_path: Path) -> None:
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    df.write_parquet(temp_path, compression="zstd")
    temp_path.replace(output_path)


def _append_json_log(path: Path, payload: dict[str, Any]) -> None:
    payload = {
        "logged_at_utc": utc_now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        **payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str))
        handle.write("\n")


def _classify_error(
    exc: Exception,
    *,
    batch_size: int,
    attempt_no: int,
    max_attempts: int,
) -> dict[str, Any]:
    error_details = _error_details(exc)
    message = str(exc).lower()
    status_code = error_details["status_code"]
    headers = error_details["headers"]
    if error_details["error_kind"] == "unresolved_identifiers":
        if batch_size == 1 and attempt_no < max_attempts:
            return {
                "state": LEDGER_RETRYABLE_ERROR,
                "split_batch": False,
                "stop_stage": False,
                "defer_stage": False,
            }
        return {
            "state": LEDGER_RETRYABLE_ERROR if batch_size > 1 else LEDGER_FATAL_ERROR,
            "split_batch": batch_size > 1,
            "stop_stage": False,
            "defer_stage": False,
        }
    if _indicates_daily_limit(status_code, headers, message):
        return {"state": LEDGER_DEFERRED_DAILY_LIMIT, "split_batch": False, "stop_stage": False, "defer_stage": True}
    if status_code == 403:
        return {"state": LEDGER_FATAL_ERROR, "split_batch": False, "stop_stage": True, "defer_stage": False}
    if status_code in {429, 503}:
        if attempt_no >= max_attempts:
            return {"state": LEDGER_FATAL_ERROR, "split_batch": False, "stop_stage": True, "defer_stage": False}
        return {
            "state": LEDGER_RETRYABLE_ERROR,
            "split_batch": status_code == 503 and batch_size > 1 and attempt_no >= 2,
            "stop_stage": False,
            "defer_stage": False,
        }
    if status_code == 400 and any(token in message for token in ("overload", "timeout", "backend")):
        if attempt_no >= max_attempts:
            return {"state": LEDGER_FATAL_ERROR, "split_batch": False, "stop_stage": True, "defer_stage": False}
        return {
            "state": LEDGER_RETRYABLE_ERROR,
            "split_batch": batch_size > 1 and attempt_no >= 2,
            "stop_stage": False,
            "defer_stage": False,
        }
    if isinstance(exc, (ValueError, TypeError)):
        return {"state": LEDGER_FATAL_ERROR, "split_batch": False, "stop_stage": True, "defer_stage": False}
    if attempt_no >= max_attempts:
        return {"state": LEDGER_FATAL_ERROR, "split_batch": False, "stop_stage": True, "defer_stage": False}
    return {"state": LEDGER_RETRYABLE_ERROR, "split_batch": False, "stop_stage": False, "defer_stage": False}


def _error_metadata(exc: Exception) -> tuple[int | None, dict[str, Any], int | None]:
    if isinstance(exc, LsegRequestError):
        return exc.status_code, exc.headers, exc.response_bytes
    return None, {}, None


def _error_details(exc: Exception) -> dict[str, Any]:
    status_code, headers, response_bytes = _error_metadata(exc)
    if isinstance(exc, LsegRequestError):
        error_kind = exc.error_kind
        unresolved_identifiers = list(exc.unresolved_identifiers)
    else:
        error_kind, parsed_identifiers = classify_lseg_error_message(str(exc))
        unresolved_identifiers = list(parsed_identifiers)
    return {
        "status_code": status_code,
        "headers": headers,
        "response_bytes": response_bytes,
        "error_kind": error_kind,
        "unresolved_identifiers": unresolved_identifiers,
    }


def _should_treat_as_empty_result(
    *,
    error_kind: str | None,
    unresolved_identifiers: list[str],
    universe: list[str],
    attempt_no: int,
    max_attempts: int,
) -> bool:
    if error_kind != "unresolved_identifiers":
        return False
    if attempt_no < max_attempts:
        return False
    unresolved_set = {
        normalized
        for identifier in unresolved_identifiers
        if (normalized := _normalize_lookup_text(identifier)) is not None
    }
    requested_set = {
        normalized
        for identifier in universe
        if (normalized := _normalize_lookup_text(identifier)) is not None
    }
    return bool(requested_set) and requested_set.issubset(unresolved_set)


def _retry_delay_seconds(attempt_no: int) -> float:
    ladder = [5.0, 15.0, 45.0, 120.0, 300.0]
    base = ladder[min(max(attempt_no - 1, 0), len(ladder) - 1)]
    return min(600.0, random.uniform(0.5 * base, 1.5 * base))


def _daily_limit_likely_exhausted(headers: dict[str, Any]) -> bool:
    numeric_values: list[int] = []
    for key, value in headers.items():
        lowered = key.lower()
        if "remaining" not in lowered and "daily" not in lowered and "limit" not in lowered and "usage" not in lowered:
            continue
        try:
            numeric_values.append(int(value))
        except (TypeError, ValueError):
            continue
    return bool(numeric_values) and min(numeric_values) <= 25


def _indicates_daily_limit(status_code: int | None, headers: dict[str, Any], message: str) -> bool:
    if "daily" in message and "limit" in message:
        return True
    if status_code == 429 and _daily_limit_likely_exhausted(headers):
        return True
    return False


def _next_daily_resume_utc(now: dt.datetime) -> dt.datetime:
    tomorrow = (now + dt.timedelta(days=1)).astimezone(dt.timezone.utc).date()
    return dt.datetime.combine(tomorrow, dt.time(hour=0, minute=5), tzinfo=dt.timezone.utc)

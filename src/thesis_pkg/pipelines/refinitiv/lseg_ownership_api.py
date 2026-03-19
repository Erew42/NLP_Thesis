from __future__ import annotations

import datetime as dt
from pathlib import Path
import time
from typing import Any, Callable

import polars as pl

from thesis_pkg.pipelines.refinitiv.lseg_batching import RequestItem, batch_items, split_batch, stable_hash_id
from thesis_pkg.pipelines.refinitiv.lseg_ledger import (
    LEDGER_DEFERRED_DAILY_LIMIT,
    LEDGER_RETRYABLE_ERROR,
    RequestLedger,
    utc_now,
)
from thesis_pkg.pipelines.refinitiv.lseg_lookup_api import (
    _append_json_log,
    _classify_error,
    _daily_limit_likely_exhausted,
    _error_details,
    _next_daily_resume_utc,
    _retry_delay_seconds,
    _should_treat_as_empty_result,
    _standardize_field_frame,
    _write_parquet_atomic,
)
from thesis_pkg.pipelines.refinitiv.lseg_provider import LsegDataProvider
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
    _cast_df_to_schema,
    _normalize_lookup_text,
    _ownership_universe_handoff_schema,
    _ownership_universe_results_schema,
    build_refinitiv_ownership_universe_row_summary,
)


OWNERSHIP_UNIVERSE_STAGE = "ownership_universe"
DOC_EXACT_STAGE = "doc_ownership_exact"
DOC_FALLBACK_STAGE = "doc_ownership_fallback"

OWNERSHIP_UNIVERSE_FIELDS: tuple[str, ...] = (
    "TR.CategoryOwnershipPct.Date",
    "TR.CategoryOwnershipPct",
    "TR.InstrStatTypeValue",
)
DOC_EXACT_FIELDS: tuple[str, ...] = (
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


def run_refinitiv_step1_ownership_universe_api_pipeline(
    *,
    handoff_parquet_path: Path | str,
    output_dir: Path | str,
    provider: Any | None = None,
    ledger_path: Path | str | None = None,
    request_log_path: Path | str | None = None,
    max_batch_size: int = 10,
    min_seconds_between_requests: float = 2.0,
    max_attempts: int = 4,
) -> dict[str, Path]:
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
    handoff_df = _cast_df_to_schema(
        pl.read_parquet(handoff_parquet_path).select(OWNERSHIP_UNIVERSE_HANDOFF_COLUMNS),
        _ownership_universe_handoff_schema(),
    ).select(OWNERSHIP_UNIVERSE_HANDOFF_COLUMNS)

    items = _build_ownership_universe_items(handoff_df)
    staging_dir, _ = _run_api_batches(
        stage=OWNERSHIP_UNIVERSE_STAGE,
        items=items,
        output_dir=output_dir,
        ledger_path=ledger_path,
        request_log_path=request_log_path,
        provider=provider,
        max_batch_size=max_batch_size,
        min_seconds_between_requests=min_seconds_between_requests,
        max_attempts=max_attempts,
        response_normalizer=_normalize_ownership_universe_batch_response,
    )

    staging_paths = sorted(staging_dir.glob("*.parquet"))
    results_df = (
        pl.concat([pl.read_parquet(path) for path in staging_paths], how="vertical_relaxed")
        if staging_paths
        else _empty_ownership_universe_results_df()
    )
    results_df = (
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
    row_summary_df = build_refinitiv_ownership_universe_row_summary(handoff_df, results_df)

    results_path = output_dir / "refinitiv_ownership_universe_results.parquet"
    row_summary_path = output_dir / "refinitiv_ownership_universe_row_summary.parquet"
    results_df.write_parquet(results_path, compression="zstd")
    row_summary_df.write_parquet(row_summary_path, compression="zstd")
    return {
        "refinitiv_ownership_universe_results_parquet": results_path,
        "refinitiv_ownership_universe_row_summary_parquet": row_summary_path,
        "refinitiv_ownership_universe_api_ledger_sqlite3": ledger_path,
        "refinitiv_ownership_universe_api_requests_jsonl": request_log_path,
    }


def run_refinitiv_lm2011_doc_ownership_exact_api_pipeline(
    *,
    doc_filing_artifact_path: Path | str,
    authority_decisions_artifact_path: Path | str,
    authority_exceptions_artifact_path: Path | str,
    output_dir: Path | str,
    provider: Any | None = None,
    ledger_path: Path | str | None = None,
    request_log_path: Path | str | None = None,
    max_batch_size: int = 15,
    min_seconds_between_requests: float = 2.0,
    max_attempts: int = 4,
) -> dict[str, Path]:
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

    request_df = build_refinitiv_lm2011_doc_ownership_requests(
        _read_doc_filing_artifact(doc_filing_artifact_path),
        _read_authority_decisions_artifact(authority_decisions_artifact_path),
        _read_authority_exceptions_artifact(authority_exceptions_artifact_path),
    )
    requests_path = output_dir / "refinitiv_lm2011_doc_ownership_exact_requests.parquet"
    request_df.write_parquet(requests_path, compression="zstd")

    items = _build_doc_ownership_exact_items(request_df)
    staging_dir, _ = _run_api_batches(
        stage=DOC_EXACT_STAGE,
        items=items,
        output_dir=output_dir,
        ledger_path=ledger_path,
        request_log_path=request_log_path,
        provider=provider,
        max_batch_size=max_batch_size,
        min_seconds_between_requests=min_seconds_between_requests,
        max_attempts=max_attempts,
        response_normalizer=_normalize_doc_exact_batch_response,
    )

    staging_paths = sorted(staging_dir.glob("*.parquet"))
    exact_raw_df = (
        pl.concat([pl.read_parquet(path) for path in staging_paths], how="vertical_relaxed")
        if staging_paths
        else _empty_df(DOC_OWNERSHIP_RAW_COLUMNS, _doc_ownership_raw_schema())
    )
    exact_raw_df = (
        _cast_doc_df_to_schema(exact_raw_df.select(DOC_OWNERSHIP_RAW_COLUMNS), _doc_ownership_raw_schema())
        .select(DOC_OWNERSHIP_RAW_COLUMNS)
        .unique(
            subset=["doc_id", "request_stage", "response_date", "returned_category", "returned_value"],
            maintain_order=True,
        )
    )
    exact_raw_path = output_dir / "refinitiv_lm2011_doc_ownership_exact_raw.parquet"
    exact_raw_df.write_parquet(exact_raw_path, compression="zstd")
    return {
        "refinitiv_lm2011_doc_ownership_exact_requests_parquet": requests_path,
        "refinitiv_lm2011_doc_ownership_exact_raw_parquet": exact_raw_path,
        "refinitiv_doc_ownership_exact_api_ledger_sqlite3": ledger_path,
        "refinitiv_doc_ownership_exact_api_requests_jsonl": request_log_path,
    }


def run_refinitiv_lm2011_doc_ownership_fallback_api_pipeline(
    *,
    output_dir: Path | str,
    provider: Any | None = None,
    ledger_path: Path | str | None = None,
    request_log_path: Path | str | None = None,
    max_batch_size: int = 5,
    min_seconds_between_requests: float = 2.0,
    max_attempts: int = 4,
) -> dict[str, Path]:
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
    fallback_requests_path = output_dir / "refinitiv_lm2011_doc_ownership_fallback_requests.parquet"
    fallback_request_df.write_parquet(fallback_requests_path, compression="zstd")

    items = _build_doc_ownership_fallback_items(fallback_request_df)
    if items:
        staging_dir, _ = _run_api_batches(
            stage=DOC_FALLBACK_STAGE,
            items=items,
            output_dir=output_dir,
            ledger_path=ledger_path,
            request_log_path=request_log_path,
            provider=provider,
            max_batch_size=max_batch_size,
            min_seconds_between_requests=min_seconds_between_requests,
            max_attempts=max_attempts,
            response_normalizer=_normalize_doc_fallback_batch_response,
        )
        staging_paths = sorted(staging_dir.glob("*.parquet"))
        fallback_raw_df = (
            pl.concat([pl.read_parquet(path) for path in staging_paths], how="vertical_relaxed")
            if staging_paths
            else _empty_df(DOC_OWNERSHIP_RAW_COLUMNS, _doc_ownership_raw_schema())
        )
    else:
        fallback_raw_df = _empty_df(DOC_OWNERSHIP_RAW_COLUMNS, _doc_ownership_raw_schema())
    fallback_raw_df = (
        _cast_doc_df_to_schema(fallback_raw_df.select(DOC_OWNERSHIP_RAW_COLUMNS), _doc_ownership_raw_schema())
        .select(DOC_OWNERSHIP_RAW_COLUMNS)
        .unique(
            subset=["doc_id", "request_stage", "response_date", "returned_category", "returned_value"],
            maintain_order=True,
        )
    )
    fallback_raw_path = output_dir / "refinitiv_lm2011_doc_ownership_fallback_raw.parquet"
    fallback_raw_df.write_parquet(fallback_raw_path, compression="zstd")
    return {
        "refinitiv_lm2011_doc_ownership_exact_raw_parquet": exact_raw_path,
        "refinitiv_lm2011_doc_ownership_fallback_requests_parquet": fallback_requests_path,
        "refinitiv_lm2011_doc_ownership_fallback_raw_parquet": fallback_raw_path,
        "refinitiv_doc_ownership_fallback_api_ledger_sqlite3": ledger_path,
        "refinitiv_doc_ownership_fallback_api_requests_jsonl": request_log_path,
    }


def _run_api_batches(
    *,
    stage: str,
    items: list[RequestItem],
    output_dir: Path,
    ledger_path: Path,
    request_log_path: Path,
    provider: Any | None,
    max_batch_size: int,
    min_seconds_between_requests: float,
    max_attempts: int,
    response_normalizer: Callable[[list[Any], pl.DataFrame], pl.DataFrame],
) -> tuple[Path, RequestLedger]:
    staging_dir = output_dir / "staging" / stage
    staging_dir.mkdir(parents=True, exist_ok=True)
    ledger = RequestLedger(ledger_path)
    ledger.enqueue(items, batch_items(items, max_batch_size=max_batch_size, unique_instrument_limit=True))
    ledger.requeue_stale_running()
    requeued_fatal_batches = ledger.requeue_known_fixable_fatal_batches(stage=stage)
    if requeued_fatal_batches:
        _append_json_log(
            request_log_path,
            {
                "event": "requeued_known_fixable_fatal_batches",
                "stage": stage,
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
                _write_parquet_atomic(normalized_batch_df, staging_path)
                row_count_by_item_id = {}
                if normalized_batch_df.height:
                    row_count_by_item_id = {
                        row["item_id"]: int(row["len"])
                        for row in normalized_batch_df.group_by("item_id").len().to_dicts()
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
                if _daily_limit_likely_exhausted(response.metadata.headers):
                    resume_at = _next_daily_resume_utc(utc_now())
                    ledger.defer_pending_stage_items(
                        stage=stage,
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
                ):
                    normalized_batch_df = response_normalizer(batch_items_rows, pl.DataFrame())
                    staging_path = staging_dir / f"{batch.batch_id}.parquet"
                    _write_parquet_atomic(normalized_batch_df, staging_path)
                    row_count_by_item_id = {}
                    if normalized_batch_df.height:
                        row_count_by_item_id = {
                            row["item_id"]: int(row["len"])
                            for row in normalized_batch_df.group_by("item_id").len().to_dicts()
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
                        stage=stage,
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
    return staging_dir, ledger


def _build_ownership_universe_items(handoff_df: pl.DataFrame) -> list[RequestItem]:
    items: list[RequestItem] = []
    for row in handoff_df.filter(pl.col("retrieval_eligible").fill_null(False)).to_dicts():
        ownership_lookup_row_id = _normalize_lookup_text(row.get("ownership_lookup_row_id"))
        candidate_ric = _normalize_lookup_text(row.get("candidate_ric"))
        request_start_date = _normalize_lookup_text(row.get("request_start_date"))
        request_end_date = _normalize_lookup_text(row.get("request_end_date"))
        if ownership_lookup_row_id is None or candidate_ric is None or request_start_date is None or request_end_date is None:
            continue
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


def _build_doc_ownership_exact_items(request_df: pl.DataFrame) -> list[RequestItem]:
    items: list[RequestItem] = []
    for row in request_df.filter(pl.col("retrieval_eligible").fill_null(False)).to_dicts():
        doc_id = _normalize_lookup_text(row.get("doc_id"))
        authoritative_ric = _normalize_lookup_text(row.get("authoritative_ric"))
        target_quarter_end = _normalize_date_value(row.get("target_quarter_end"))
        if doc_id is None or authoritative_ric is None or target_quarter_end is None:
            continue
        date_text = target_quarter_end.isoformat()
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
        matched_rows = rows_by_instrument.get(item.instrument, [])
        for matched_row in matched_rows:
            returned_ric = _normalize_lookup_text(matched_row.get("instrument"))
            returned_date = _normalize_date_value(matched_row.get("TR.CategoryOwnershipPct.Date"))
            returned_value = _normalize_float_value(matched_row.get("TR.CategoryOwnershipPct"))
            returned_category = _normalize_lookup_text(matched_row.get("TR.InstrStatTypeValue"))
            if returned_date is None and returned_value is None and returned_category is None:
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
        target_quarter_end = _normalize_date_value(request_row.get("target_quarter_end"))
        matched_rows = rows_by_instrument.get(item.instrument, [])
        for matched_row in matched_rows:
            returned_value = _normalize_float_value(matched_row.get("TR.CategoryOwnershipPct"))
            returned_category = _normalize_category(matched_row.get("TR.InstrStatTypeValue"))
            if returned_value is None and returned_category is None:
                continue
            rows.append(
                {
                    "item_id": item.item_id,
                    **{name: request_row.get(name) for name in DOC_OWNERSHIP_REQUEST_COLUMNS},
                    "request_stage": DOC_OWNERSHIP_EXACT_STAGE,
                    "response_date": target_quarter_end,
                    "response_date_is_imputed": True,
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
        matched_rows = rows_by_instrument.get(item.instrument, [])
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

from __future__ import annotations

import contextlib
import datetime as dt
import json
from dataclasses import dataclass
import os
from pathlib import Path
import socket
import sqlite3
import time
from typing import Any
import uuid

from thesis_pkg.pipelines.refinitiv.lseg_batching import (
    BatchDefinition,
    RequestItem,
    stable_hash_id,
    stable_json_dumps,
)


LEDGER_PENDING = "pending"
LEDGER_RUNNING = "running"
LEDGER_SUCCEEDED = "succeeded"
LEDGER_RETRYABLE_ERROR = "retryable_error"
LEDGER_DEFERRED_DAILY_LIMIT = "deferred_daily_limit"
LEDGER_FATAL_ERROR = "fatal_error"
LEDGER_TERMINAL_STATES = (LEDGER_SUCCEEDED, LEDGER_DEFERRED_DAILY_LIMIT, LEDGER_FATAL_ERROR)

ATTEMPT_PHASE_CLAIMED = "claimed"
ATTEMPT_PHASE_REQUEST_STARTED = "request_started"
ATTEMPT_PHASE_REQUEST_FINISHED_SUCCESS = "request_finished_success"
ATTEMPT_PHASE_REQUEST_FINISHED_ERROR = "request_finished_error"
ATTEMPT_PHASE_REQUEUED_STALE_RUNNING = "requeued_stale_running"
ATTEMPT_PHASE_SPLIT_INTO_CHILDREN = "split_into_children"
ATTEMPT_PHASE_LEGACY_FINISH_BACKFILL = "legacy_finish_backfill"

LEDGER_SCHEMA_VERSION = 3


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def to_utc_text(value: dt.datetime | None) -> str | None:
    if value is None:
        return None
    normalized = value.astimezone(dt.timezone.utc).replace(microsecond=0)
    return normalized.strftime("%Y-%m-%dT%H:%M:%SZ")


def from_utc_text(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    return dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)


def default_run_session_id() -> str:
    timestamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    return f"run_{timestamp}_{uuid.uuid4().hex[:8]}"


def default_worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


@dataclass(frozen=True)
class LedgerItem:
    item_id: str
    stage: str
    instrument: str
    batch_key: str
    fields: tuple[str, ...]
    parameters: dict[str, Any]
    payload: dict[str, Any]
    state: str
    attempt_count: int
    result_row_count: int | None
    stage_output_path: str | None
    response_fingerprint: str | None
    last_error: str | None
    next_eligible_at_utc: dt.datetime | None


@dataclass(frozen=True)
class LedgerBatch:
    batch_id: str
    stage: str
    batch_key: str
    fields: tuple[str, ...]
    parameters: dict[str, Any]
    item_ids: tuple[str, ...]
    state: str
    attempt_count: int
    last_status_code: int | None
    last_latency_ms: int | None
    rows_returned: int | None
    response_bytes: int | None
    result_file_path: str | None
    header_json: dict[str, Any] | None
    last_error: str | None
    next_eligible_at_utc: dt.datetime | None
    claim_token: str | None
    last_run_session_id: str | None
    last_worker_id: str | None
    running_started_at_utc: dt.datetime | None


@dataclass(frozen=True)
class RequestStartReservation:
    status: str
    scheduled_start_epoch_utc: float | None
    wait_seconds: float


class RequestLedger:
    def __init__(
        self,
        db_path: Path | str,
        *,
        run_session_id: str | None = None,
        worker_id: str | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.run_session_id = run_session_id or default_run_session_id()
        self.worker_id = worker_id or default_worker_id()
        self._initialize()

    def enqueue(self, items: list[RequestItem], batches: list[BatchDefinition]) -> None:
        now_text = to_utc_text(utc_now())
        with self._connect() as conn:
            for item in items:
                conn.execute(
                    """
                    INSERT INTO request_items (
                        item_id, stage, instrument, batch_key, fields_json, parameters_json,
                        payload_json, state, attempt_count, created_at_utc, updated_at_utc
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                    ON CONFLICT(item_id) DO UPDATE SET
                        stage=excluded.stage,
                        instrument=excluded.instrument,
                        batch_key=excluded.batch_key,
                        fields_json=excluded.fields_json,
                        parameters_json=excluded.parameters_json,
                        payload_json=excluded.payload_json,
                        updated_at_utc=excluded.updated_at_utc
                    """,
                    (
                        item.item_id,
                        item.stage,
                        item.instrument,
                        item.batch_key,
                        stable_json_dumps(item.fields),
                        stable_json_dumps(item.parameters),
                        stable_json_dumps(item.payload),
                        LEDGER_PENDING,
                        now_text,
                        now_text,
                    ),
                )
            for batch in batches:
                conn.execute(
                    """
                    INSERT INTO batches (
                        batch_id, stage, batch_key, fields_json, parameters_json, item_ids_json,
                        item_count, state, attempt_count, created_at_utc, updated_at_utc
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                    ON CONFLICT(batch_id) DO NOTHING
                    """,
                    (
                        batch.batch_id,
                        batch.stage,
                        batch.batch_key,
                        stable_json_dumps(batch.fields),
                        stable_json_dumps(batch.parameters),
                        stable_json_dumps(batch.item_ids),
                        len(batch.item_ids),
                        LEDGER_PENDING,
                        now_text,
                        now_text,
                    ),
                )

    def get_meta(self, key: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM ledger_meta WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return str(row["value"])

    def set_meta(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ledger_meta (key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                (key, value),
            )

    def stage_batch_count(self, *, stage: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS batch_count
                FROM batches
                WHERE stage = ?
                """,
                (stage,),
            ).fetchone()
        return 0 if row is None else int(row["batch_count"] or 0)

    def ensure_stage_meta_value(self, *, stage: str, key: str, value: str) -> None:
        meta_key = f"stage:{stage}:{key}"
        existing = self.get_meta(meta_key)
        if existing is None:
            if self.stage_batch_count(stage=stage) > 0:
                raise RuntimeError(
                    f"ledger already contains stage={stage!r} batches but is missing compatibility metadata "
                    f"for {meta_key!r}; remove the stage ledger/staging artifacts before resuming"
                )
            self.set_meta(meta_key, value)
            return
        if existing != value:
            raise RuntimeError(
                f"incompatible resume state for stage={stage!r}: {meta_key!r} changed "
                f"from {existing!r} to {value!r}"
            )

    def requeue_stale_running(self, *, older_than_seconds: int = 900) -> int:
        cutoff = utc_now() - dt.timedelta(seconds=older_than_seconds)
        cutoff_text = to_utc_text(cutoff)
        now_text = to_utc_text(utc_now())
        requeued = 0
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM batches
                WHERE state = ? AND updated_at_utc < ?
                ORDER BY updated_at_utc, batch_id
                """,
                (LEDGER_RUNNING, cutoff_text),
            ).fetchall()
            for row in rows:
                batch_id = str(row["batch_id"])
                item_ids = self._item_ids_from_row(row)
                conn.execute(
                    """
                    UPDATE batches
                    SET state = ?, updated_at_utc = ?, claim_token = NULL, running_started_at_utc = NULL
                    WHERE batch_id = ?
                    """,
                    (
                        LEDGER_PENDING,
                        now_text,
                        batch_id,
                    ),
                )
                self._update_items_for_ids(
                    conn,
                    item_ids=item_ids,
                    state=LEDGER_PENDING,
                    last_error="requeued_stale_running",
                    next_eligible_at_utc=None,
                    updated_at_utc=now_text,
                )
                attempt_no = int(row["attempt_count"] or 0)
                if attempt_no > 0:
                    self._insert_attempt_event(
                        conn,
                        batch_id=batch_id,
                        attempt_no=attempt_no,
                        phase=ATTEMPT_PHASE_REQUEUED_STALE_RUNNING,
                        claim_token=None if row["claim_token"] is None else str(row["claim_token"]),
                        headers=None,
                        status_code=None,
                        latency_ms=None,
                        rows_returned=None,
                        response_bytes=None,
                        exception_class=None,
                        exception_message="requeued stale running batch",
                    )
                requeued += 1
        return requeued

    def requeue_known_fixable_fatal_batches(self, *, stage: str | None = None) -> int:
        predicates = (
            "last_error LIKE 'Unable to resolve all requested identifiers%'",
            "last_error LIKE 'could not append value:%'",
            "last_error LIKE 'failed to convert LSEG response to Polars:%'",
        )
        where_clause = " OR ".join(predicates)
        now_text = to_utc_text(utc_now())
        requeued = 0
        with self._connect() as conn:
            params: list[Any] = [LEDGER_FATAL_ERROR]
            query = f"""
                SELECT *
                FROM batches
                WHERE state = ?
                  AND ({where_clause})
            """
            if stage is not None:
                query += " AND stage = ?"
                params.append(stage)
            rows = conn.execute(query, params).fetchall()
            for row in rows:
                batch_id = str(row["batch_id"])
                item_ids = self._item_ids_from_row(row)
                conn.execute(
                    """
                    UPDATE batches
                    SET state = ?, last_error = NULL, next_eligible_at_utc = NULL, updated_at_utc = ?
                    WHERE batch_id = ?
                    """,
                    (
                        LEDGER_PENDING,
                        now_text,
                        batch_id,
                    ),
                )
                self._update_items_for_ids(
                    conn,
                    item_ids=item_ids,
                    state=LEDGER_PENDING,
                    last_error=None,
                    next_eligible_at_utc=None,
                    updated_at_utc=now_text,
                )
                requeued += 1
        return requeued

    def clear_runtime_halt(self, *, stage: str) -> None:
        now_text = to_utc_text(utc_now())
        with self._connect(begin_immediate=True) as conn:
            conn.execute(
                """
                INSERT INTO stage_runtime_control (
                    stage, halt_new_work, halt_reason, updated_at_utc
                )
                VALUES (?, 0, NULL, ?)
                ON CONFLICT(stage) DO UPDATE SET
                    halt_new_work = 0,
                    halt_reason = NULL,
                    updated_at_utc = excluded.updated_at_utc
                """,
                (stage, now_text),
            )

    def halt_runtime_work(self, *, stage: str, reason: str) -> None:
        now_text = to_utc_text(utc_now())
        with self._connect(begin_immediate=True) as conn:
            conn.execute(
                """
                INSERT INTO stage_runtime_control (
                    stage, halt_new_work, halt_reason, updated_at_utc
                )
                VALUES (?, 1, ?, ?)
                ON CONFLICT(stage) DO UPDATE SET
                    halt_new_work = 1,
                    halt_reason = excluded.halt_reason,
                    updated_at_utc = excluded.updated_at_utc
                """,
                (stage, reason, now_text),
            )

    def is_runtime_halted(self, *, stage: str) -> bool:
        with self._connect() as conn:
            return self._runtime_halt_active(conn, stage=stage)

    def reset_request_start_scheduler(self, *, stage: str, initial_delay_seconds: float = 0.0) -> None:
        now_text = to_utc_text(utc_now())
        now_epoch = time.time() + max(initial_delay_seconds, 0.0)
        with self._connect(begin_immediate=True) as conn:
            conn.execute(
                """
                INSERT INTO request_start_scheduler (
                    stage, next_request_start_epoch_utc, updated_at_utc
                )
                VALUES (?, ?, ?)
                ON CONFLICT(stage) DO UPDATE SET
                    next_request_start_epoch_utc = excluded.next_request_start_epoch_utc,
                    updated_at_utc = excluded.updated_at_utc
                """,
                (stage, now_epoch, now_text),
            )

    def reserve_next_request_start(
        self,
        *,
        stage: str,
        min_seconds_between_request_starts_total: float,
        idle_poll_seconds: float = 0.1,
    ) -> RequestStartReservation:
        if min_seconds_between_request_starts_total < 0:
            raise ValueError("min_seconds_between_request_starts_total must be >= 0")

        with self._connect(begin_immediate=True) as conn:
            now_dt = utc_now()
            now_text = to_utc_text(now_dt)
            now_epoch = time.time()
            if self._runtime_halt_active(conn, stage=stage):
                return RequestStartReservation(status="halted", scheduled_start_epoch_utc=None, wait_seconds=0.0)
            if self._has_claimable_batch(conn, stage=stage, now_text=now_text):
                row = conn.execute(
                    """
                    SELECT next_request_start_epoch_utc
                    FROM request_start_scheduler
                    WHERE stage = ?
                    """,
                    (stage,),
                ).fetchone()
                current_next_epoch = now_epoch if row is None else float(row["next_request_start_epoch_utc"])
                scheduled_start_epoch = max(now_epoch, current_next_epoch)
                next_request_start_epoch = scheduled_start_epoch + min_seconds_between_request_starts_total
                conn.execute(
                    """
                    INSERT INTO request_start_scheduler (
                        stage, next_request_start_epoch_utc, updated_at_utc
                    )
                    VALUES (?, ?, ?)
                    ON CONFLICT(stage) DO UPDATE SET
                        next_request_start_epoch_utc = excluded.next_request_start_epoch_utc,
                        updated_at_utc = excluded.updated_at_utc
                    """,
                    (stage, next_request_start_epoch, now_text),
                )
                return RequestStartReservation(
                    status="reserved",
                    scheduled_start_epoch_utc=scheduled_start_epoch,
                    wait_seconds=max(0.0, scheduled_start_epoch - now_epoch),
                )

            if self._has_outstanding_batch(conn, stage=stage):
                wait_seconds = self._runtime_idle_wait_seconds(
                    conn,
                    stage=stage,
                    now_dt=now_dt,
                    idle_poll_seconds=idle_poll_seconds,
                )
                return RequestStartReservation(status="wait", scheduled_start_epoch_utc=None, wait_seconds=wait_seconds)

            return RequestStartReservation(status="drained", scheduled_start_epoch_utc=None, wait_seconds=0.0)

    def claim_next_batch(self, *, stage: str) -> LedgerBatch | None:
        now = utc_now()
        now_text = to_utc_text(now)
        with self._connect(begin_immediate=True) as conn:
            if self._runtime_halt_active(conn, stage=stage):
                return None
            row = conn.execute(
                """
                SELECT *
                FROM batches
                WHERE stage = ?
                  AND state IN (?, ?)
                  AND (next_eligible_at_utc IS NULL OR next_eligible_at_utc <= ?)
                ORDER BY
                    CASE state
                        WHEN ? THEN 0
                        WHEN ? THEN 1
                        ELSE 2
                    END,
                    created_at_utc,
                    batch_id
                LIMIT 1
                """,
                (
                    stage,
                    LEDGER_PENDING,
                    LEDGER_RETRYABLE_ERROR,
                    now_text,
                    LEDGER_PENDING,
                    LEDGER_RETRYABLE_ERROR,
                ),
            ).fetchone()
            if row is None:
                return None
            attempt_no = int(row["attempt_count"]) + 1
            claim_token = stable_hash_id(row["batch_id"], attempt_no, now_text, prefix="claim")
            batch_id = str(row["batch_id"])
            item_ids = self._item_ids_from_row(row)
            cursor = conn.execute(
                """
                UPDATE batches
                SET state = ?,
                    updated_at_utc = ?,
                    attempt_count = attempt_count + 1,
                    claim_token = ?,
                    running_started_at_utc = NULL,
                    last_run_session_id = ?,
                    last_worker_id = ?,
                    next_eligible_at_utc = NULL
                WHERE batch_id = ?
                  AND state IN (?, ?)
                  AND (next_eligible_at_utc IS NULL OR next_eligible_at_utc <= ?)
                """,
                (
                    LEDGER_RUNNING,
                    now_text,
                    claim_token,
                    self.run_session_id,
                    self.worker_id,
                    batch_id,
                    LEDGER_PENDING,
                    LEDGER_RETRYABLE_ERROR,
                    now_text,
                ),
            )
            if int(cursor.rowcount or 0) != 1:
                return None
            self._increment_item_attempts(
                conn,
                item_ids=item_ids,
                updated_at_utc=now_text,
            )
            self._insert_attempt_event(
                conn,
                batch_id=batch_id,
                attempt_no=attempt_no,
                phase=ATTEMPT_PHASE_CLAIMED,
                claim_token=claim_token,
                headers=None,
                status_code=None,
                latency_ms=None,
                rows_returned=None,
                response_bytes=None,
                exception_class=None,
                exception_message=None,
            )
            updated_row = conn.execute(
                "SELECT * FROM batches WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()
            if updated_row is None:
                return None
            return self._batch_from_row(updated_row)

    def record_request_started(self, *, batch_id: str) -> None:
        now = utc_now()
        now_text = to_utc_text(now)
        with self._connect() as conn:
            batch_row = conn.execute("SELECT * FROM batches WHERE batch_id = ?", (batch_id,)).fetchone()
            if batch_row is None:
                raise KeyError(f"unknown batch_id: {batch_id}")
            attempt_no = int(batch_row["attempt_count"])
            conn.execute(
                """
                UPDATE batches
                SET updated_at_utc = ?, running_started_at_utc = COALESCE(running_started_at_utc, ?)
                WHERE batch_id = ?
                """,
                (now_text, now_text, batch_id),
            )
            self._insert_attempt_event(
                conn,
                batch_id=batch_id,
                attempt_no=attempt_no,
                phase=ATTEMPT_PHASE_REQUEST_STARTED,
                claim_token=None if batch_row["claim_token"] is None else str(batch_row["claim_token"]),
                headers=None,
                status_code=None,
                latency_ms=None,
                rows_returned=None,
                response_bytes=None,
                exception_class=None,
                exception_message=None,
            )

    def fetch_items(self, batch: LedgerBatch) -> list[LedgerItem]:
        placeholders = ", ".join("?" for _ in batch.item_ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM request_items WHERE item_id IN ({placeholders}) ORDER BY item_id",
                batch.item_ids,
            ).fetchall()
        return [self._item_from_row(row) for row in rows]

    def record_success(
        self,
        *,
        batch_id: str,
        row_count_by_item_id: dict[str, int],
        stage_output_path: Path | str,
        response_fingerprint: str | None,
        headers: dict[str, Any] | None,
        status_code: int | None,
        latency_ms: int | None,
        rows_returned: int,
        response_bytes: int | None,
    ) -> None:
        now_text = to_utc_text(utc_now())
        stage_output_text = str(stage_output_path)
        with self._connect() as conn:
            batch_row = conn.execute("SELECT * FROM batches WHERE batch_id = ?", (batch_id,)).fetchone()
            if batch_row is None:
                raise KeyError(f"unknown batch_id: {batch_id}")
            attempt_no = int(batch_row["attempt_count"])
            item_ids = self._item_ids_from_row(batch_row)
            for item_id in item_ids:
                conn.execute(
                    """
                    UPDATE request_items
                    SET state = ?, result_row_count = ?, stage_output_path = ?, response_fingerprint = ?,
                        updated_at_utc = ?, last_error = NULL, next_eligible_at_utc = NULL
                    WHERE item_id = ?
                    """,
                    (
                        LEDGER_SUCCEEDED,
                        int(row_count_by_item_id.get(item_id, 0)),
                        stage_output_text,
                        response_fingerprint,
                        now_text,
                        item_id,
                    ),
                )
            conn.execute(
                """
                UPDATE batches
                SET state = ?, result_file_path = ?, header_json = ?, last_status_code = ?,
                    last_latency_ms = ?, rows_returned = ?, response_bytes = ?, last_error = NULL,
                    next_eligible_at_utc = NULL, updated_at_utc = ?, claim_token = NULL, running_started_at_utc = NULL
                WHERE batch_id = ?
                """,
                (
                    LEDGER_SUCCEEDED,
                    stage_output_text,
                    stable_json_dumps(headers or {}),
                    status_code,
                    latency_ms,
                    rows_returned,
                    response_bytes,
                    now_text,
                    batch_id,
                ),
            )
            self._insert_attempt_row(
                conn,
                batch_id=batch_id,
                attempt_no=attempt_no,
                status_code=status_code,
                latency_ms=latency_ms,
                rows_returned=rows_returned,
                response_bytes=response_bytes,
                headers=headers,
                exception_class=None,
                exception_message=None,
            )
            self._insert_attempt_event(
                conn,
                batch_id=batch_id,
                attempt_no=attempt_no,
                phase=ATTEMPT_PHASE_REQUEST_FINISHED_SUCCESS,
                claim_token=None if batch_row["claim_token"] is None else str(batch_row["claim_token"]),
                headers=headers,
                status_code=status_code,
                latency_ms=latency_ms,
                rows_returned=rows_returned,
                response_bytes=response_bytes,
                exception_class=None,
                exception_message=None,
            )

    def record_error(
        self,
        *,
        batch_id: str,
        next_state: str,
        error_message: str,
        headers: dict[str, Any] | None,
        status_code: int | None,
        latency_ms: int | None,
        response_bytes: int | None,
        next_eligible_at_utc: dt.datetime | None,
        exception_class: str | None,
    ) -> None:
        now_text = to_utc_text(utc_now())
        next_eligible_text = to_utc_text(next_eligible_at_utc)
        with self._connect() as conn:
            batch_row = conn.execute("SELECT * FROM batches WHERE batch_id = ?", (batch_id,)).fetchone()
            if batch_row is None:
                raise KeyError(f"unknown batch_id: {batch_id}")
            attempt_no = int(batch_row["attempt_count"])
            item_ids = self._item_ids_from_row(batch_row)
            conn.execute(
                """
                UPDATE batches
                SET state = ?, last_error = ?, header_json = ?, last_status_code = ?,
                    last_latency_ms = ?, response_bytes = ?, next_eligible_at_utc = ?, updated_at_utc = ?,
                    claim_token = NULL, running_started_at_utc = NULL
                WHERE batch_id = ?
                """,
                (
                    next_state,
                    error_message,
                    stable_json_dumps(headers or {}),
                    status_code,
                    latency_ms,
                    response_bytes,
                    next_eligible_text,
                    now_text,
                    batch_id,
                ),
            )
            self._update_items_for_ids(
                conn,
                item_ids=item_ids,
                state=next_state,
                last_error=error_message,
                next_eligible_at_utc=next_eligible_text,
                updated_at_utc=now_text,
            )
            self._insert_attempt_row(
                conn,
                batch_id=batch_id,
                attempt_no=attempt_no,
                status_code=status_code,
                latency_ms=latency_ms,
                rows_returned=None,
                response_bytes=response_bytes,
                headers=headers,
                exception_class=exception_class,
                exception_message=error_message,
            )
            self._insert_attempt_event(
                conn,
                batch_id=batch_id,
                attempt_no=attempt_no,
                phase=ATTEMPT_PHASE_REQUEST_FINISHED_ERROR,
                claim_token=None if batch_row["claim_token"] is None else str(batch_row["claim_token"]),
                headers=headers,
                status_code=status_code,
                latency_ms=latency_ms,
                rows_returned=None,
                response_bytes=response_bytes,
                exception_class=exception_class,
                exception_message=error_message,
            )

    def split_batch(self, *, parent_batch_id: str, child_batches: list[BatchDefinition], reason: str) -> None:
        now_text = to_utc_text(utc_now())
        with self._connect() as conn:
            parent_row = conn.execute("SELECT * FROM batches WHERE batch_id = ?", (parent_batch_id,)).fetchone()
            if parent_row is None:
                raise KeyError(f"unknown batch_id: {parent_batch_id}")
            item_ids = self._item_ids_from_row(parent_row)
            conn.execute(
                """
                UPDATE batches
                SET state = ?, last_error = ?, updated_at_utc = ?, claim_token = NULL, running_started_at_utc = NULL
                WHERE batch_id = ?
                """,
                (
                    LEDGER_FATAL_ERROR,
                    f"split_into_children:{reason}",
                    now_text,
                    parent_batch_id,
                ),
            )
            self._update_items_for_ids(
                conn,
                item_ids=item_ids,
                state=LEDGER_PENDING,
                last_error="split_into_children",
                next_eligible_at_utc=None,
                updated_at_utc=now_text,
            )
            attempt_no = int(parent_row["attempt_count"] or 0)
            if attempt_no > 0:
                self._insert_attempt_event(
                    conn,
                    batch_id=parent_batch_id,
                    attempt_no=attempt_no,
                    phase=ATTEMPT_PHASE_SPLIT_INTO_CHILDREN,
                    claim_token=None if parent_row["claim_token"] is None else str(parent_row["claim_token"]),
                    headers=None,
                    status_code=None,
                    latency_ms=None,
                    rows_returned=None,
                    response_bytes=None,
                    exception_class=None,
                    exception_message=reason,
                )
            for batch in child_batches:
                conn.execute(
                    """
                    INSERT INTO batches (
                        batch_id, stage, batch_key, fields_json, parameters_json, item_ids_json,
                        item_count, state, attempt_count, split_parent_batch_id, created_at_utc, updated_at_utc
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)
                    ON CONFLICT(batch_id) DO NOTHING
                    """,
                    (
                        batch.batch_id,
                        batch.stage,
                        batch.batch_key,
                        stable_json_dumps(batch.fields),
                        stable_json_dumps(batch.parameters),
                        stable_json_dumps(batch.item_ids),
                        len(batch.item_ids),
                        LEDGER_PENDING,
                        parent_batch_id,
                        now_text,
                        now_text,
                    ),
                )

    def defer_pending_stage_items(self, *, stage: str, next_eligible_at_utc: dt.datetime, reason: str) -> int:
        now_text = to_utc_text(utc_now())
        next_text = to_utc_text(next_eligible_at_utc)
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE batches
                SET state = ?, next_eligible_at_utc = ?, last_error = ?, updated_at_utc = ?
                WHERE stage = ? AND state IN (?, ?)
                """,
                (
                    LEDGER_DEFERRED_DAILY_LIMIT,
                    next_text,
                    reason,
                    now_text,
                    stage,
                    LEDGER_PENDING,
                    LEDGER_RETRYABLE_ERROR,
                ),
            )
            conn.execute(
                """
                UPDATE request_items
                SET state = ?, next_eligible_at_utc = ?, last_error = ?, updated_at_utc = ?
                WHERE stage = ? AND state IN (?, ?)
                """,
                (
                    LEDGER_DEFERRED_DAILY_LIMIT,
                    next_text,
                    reason,
                    now_text,
                    stage,
                    LEDGER_PENDING,
                    LEDGER_RETRYABLE_ERROR,
                ),
            )
            return int(cursor.rowcount or 0)

    def state_counts(self, *, table: str) -> dict[str, int]:
        if table not in {"batches", "request_items"}:
            raise ValueError(f"unsupported table: {table}")
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT state, COUNT(*) AS row_count FROM {table} GROUP BY state ORDER BY state"
            ).fetchall()
        return {str(row["state"]): int(row["row_count"]) for row in rows}

    def attempt_mismatches(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    b.batch_id,
                    b.attempt_count,
                    COALESCE(a.finished_attempt_count, 0) AS finished_attempt_count,
                    COALESCE(e.event_attempt_count, 0) AS event_attempt_count
                FROM batches AS b
                LEFT JOIN (
                    SELECT batch_id, COUNT(*) AS finished_attempt_count
                    FROM batch_attempts
                    GROUP BY batch_id
                ) AS a
                  ON a.batch_id = b.batch_id
                LEFT JOIN (
                    SELECT batch_id, COUNT(DISTINCT attempt_no) AS event_attempt_count
                    FROM batch_attempt_events
                    GROUP BY batch_id
                ) AS e
                  ON e.batch_id = b.batch_id
                WHERE b.attempt_count != COALESCE(a.finished_attempt_count, 0)
                   OR b.attempt_count != COALESCE(e.event_attempt_count, 0)
                ORDER BY b.batch_id
                """
            ).fetchall()
        return [
            {
                "batch_id": str(row["batch_id"]),
                "attempt_count": int(row["attempt_count"]),
                "finished_attempt_count": int(row["finished_attempt_count"]),
                "event_attempt_count": int(row["event_attempt_count"]),
            }
            for row in rows
        ]

    def run_session_ids(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT run_session_id
                FROM batch_attempt_events
                WHERE run_session_id IS NOT NULL
                ORDER BY run_session_id
                """
            ).fetchall()
        return [str(row["run_session_id"]) for row in rows]

    def succeeded_stage_output_paths(self, *, stage: str | None = None) -> list[Path]:
        params: list[Any] = [LEDGER_SUCCEEDED]
        query = """
            SELECT DISTINCT result_file_path
            FROM batches
            WHERE state = ?
              AND result_file_path IS NOT NULL
        """
        if stage is not None:
            query += " AND stage = ?"
            params.append(stage)
        query += " ORDER BY result_file_path"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [Path(str(row["result_file_path"])) for row in rows]

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ledger_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS request_items (
                    item_id TEXT PRIMARY KEY,
                    stage TEXT NOT NULL,
                    instrument TEXT NOT NULL,
                    batch_key TEXT NOT NULL,
                    fields_json TEXT NOT NULL,
                    parameters_json TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    result_row_count INTEGER,
                    stage_output_path TEXT,
                    response_fingerprint TEXT,
                    last_error TEXT,
                    next_eligible_at_utc TEXT,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS batches (
                    batch_id TEXT PRIMARY KEY,
                    stage TEXT NOT NULL,
                    batch_key TEXT NOT NULL,
                    fields_json TEXT NOT NULL,
                    parameters_json TEXT NOT NULL,
                    item_ids_json TEXT NOT NULL,
                    item_count INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    split_parent_batch_id TEXT,
                    last_status_code INTEGER,
                    last_latency_ms INTEGER,
                    rows_returned INTEGER,
                    response_bytes INTEGER,
                    result_file_path TEXT,
                    header_json TEXT,
                    last_error TEXT,
                    next_eligible_at_utc TEXT,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS batch_attempts (
                    batch_id TEXT NOT NULL,
                    attempt_no INTEGER NOT NULL,
                    status_code INTEGER,
                    latency_ms INTEGER,
                    rows_returned INTEGER,
                    response_bytes INTEGER,
                    headers_json TEXT,
                    exception_class TEXT,
                    exception_message TEXT,
                    created_at_utc TEXT NOT NULL,
                    PRIMARY KEY (batch_id, attempt_no)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS stage_runtime_control (
                    stage TEXT PRIMARY KEY,
                    halt_new_work INTEGER NOT NULL DEFAULT 0,
                    halt_reason TEXT,
                    updated_at_utc TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS request_start_scheduler (
                    stage TEXT PRIMARY KEY,
                    next_request_start_epoch_utc REAL NOT NULL,
                    updated_at_utc TEXT NOT NULL
                )
                """
            )
            self._ensure_column(conn, table="batches", column="claim_token", column_type="TEXT")
            self._ensure_column(conn, table="batches", column="running_started_at_utc", column_type="TEXT")
            self._ensure_column(conn, table="batches", column="last_run_session_id", column_type="TEXT")
            self._ensure_column(conn, table="batches", column="last_worker_id", column_type="TEXT")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS batch_attempt_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_id TEXT NOT NULL,
                    attempt_no INTEGER NOT NULL,
                    phase TEXT NOT NULL,
                    run_session_id TEXT,
                    worker_id TEXT,
                    claim_token TEXT,
                    status_code INTEGER,
                    latency_ms INTEGER,
                    rows_returned INTEGER,
                    response_bytes INTEGER,
                    headers_json TEXT,
                    exception_class TEXT,
                    exception_message TEXT,
                    created_at_utc TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_batch_attempt_events_batch_attempt
                ON batch_attempt_events (batch_id, attempt_no, phase)
                """
            )
            conn.execute(
                """
                INSERT INTO ledger_meta (key, value)
                VALUES ('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                (str(LEDGER_SCHEMA_VERSION),),
            )
            self._backfill_missing_attempt_events(conn)

    @contextlib.contextmanager
    def _connect(self, *, begin_immediate: bool = False) -> Any:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        try:
            if begin_immediate:
                conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _item_from_row(self, row: sqlite3.Row) -> LedgerItem:
        return LedgerItem(
            item_id=str(row["item_id"]),
            stage=str(row["stage"]),
            instrument=str(row["instrument"]),
            batch_key=str(row["batch_key"]),
            fields=tuple(json.loads(row["fields_json"])),
            parameters=dict(json.loads(row["parameters_json"])),
            payload=dict(json.loads(row["payload_json"])),
            state=str(row["state"]),
            attempt_count=int(row["attempt_count"]),
            result_row_count=None if row["result_row_count"] is None else int(row["result_row_count"]),
            stage_output_path=None if row["stage_output_path"] is None else str(row["stage_output_path"]),
            response_fingerprint=None if row["response_fingerprint"] is None else str(row["response_fingerprint"]),
            last_error=None if row["last_error"] is None else str(row["last_error"]),
            next_eligible_at_utc=from_utc_text(row["next_eligible_at_utc"]),
        )

    def _batch_from_row(self, row: sqlite3.Row) -> LedgerBatch:
        header_json = None if row["header_json"] is None else dict(json.loads(row["header_json"]))
        return LedgerBatch(
            batch_id=str(row["batch_id"]),
            stage=str(row["stage"]),
            batch_key=str(row["batch_key"]),
            fields=tuple(json.loads(row["fields_json"])),
            parameters=dict(json.loads(row["parameters_json"])),
            item_ids=tuple(json.loads(row["item_ids_json"])),
            state=str(row["state"]),
            attempt_count=int(row["attempt_count"]),
            last_status_code=None if row["last_status_code"] is None else int(row["last_status_code"]),
            last_latency_ms=None if row["last_latency_ms"] is None else int(row["last_latency_ms"]),
            rows_returned=None if row["rows_returned"] is None else int(row["rows_returned"]),
            response_bytes=None if row["response_bytes"] is None else int(row["response_bytes"]),
            result_file_path=None if row["result_file_path"] is None else str(row["result_file_path"]),
            header_json=header_json,
            last_error=None if row["last_error"] is None else str(row["last_error"]),
            next_eligible_at_utc=from_utc_text(row["next_eligible_at_utc"]),
            claim_token=None if row["claim_token"] is None else str(row["claim_token"]),
            last_run_session_id=None if row["last_run_session_id"] is None else str(row["last_run_session_id"]),
            last_worker_id=None if row["last_worker_id"] is None else str(row["last_worker_id"]),
            running_started_at_utc=from_utc_text(row["running_started_at_utc"]),
        )

    def _insert_attempt_row(
        self,
        conn: sqlite3.Connection,
        *,
        batch_id: str,
        attempt_no: int,
        status_code: int | None,
        latency_ms: int | None,
        rows_returned: int | None,
        response_bytes: int | None,
        headers: dict[str, Any] | None,
        exception_class: str | None,
        exception_message: str | None,
    ) -> None:
        conn.execute(
            """
            INSERT OR REPLACE INTO batch_attempts (
                batch_id, attempt_no, status_code, latency_ms, rows_returned,
                response_bytes, headers_json, exception_class, exception_message, created_at_utc
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                batch_id,
                attempt_no,
                status_code,
                latency_ms,
                rows_returned,
                response_bytes,
                stable_json_dumps(headers or {}),
                exception_class,
                exception_message,
                to_utc_text(utc_now()),
            ),
        )

    def _insert_attempt_event(
        self,
        conn: sqlite3.Connection,
        *,
        batch_id: str,
        attempt_no: int,
        phase: str,
        claim_token: str | None,
        headers: dict[str, Any] | None,
        status_code: int | None,
        latency_ms: int | None,
        rows_returned: int | None,
        response_bytes: int | None,
        exception_class: str | None,
        exception_message: str | None,
    ) -> None:
        conn.execute(
            """
            INSERT INTO batch_attempt_events (
                batch_id, attempt_no, phase, run_session_id, worker_id, claim_token,
                status_code, latency_ms, rows_returned, response_bytes,
                headers_json, exception_class, exception_message, created_at_utc
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                batch_id,
                attempt_no,
                phase,
                self.run_session_id,
                self.worker_id,
                claim_token,
                status_code,
                latency_ms,
                rows_returned,
                response_bytes,
                stable_json_dumps(headers or {}),
                exception_class,
                exception_message,
                to_utc_text(utc_now()),
            ),
        )

    def _backfill_missing_attempt_events(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            """
            SELECT
                a.batch_id,
                a.attempt_no,
                a.status_code,
                a.latency_ms,
                a.rows_returned,
                a.response_bytes,
                a.headers_json,
                a.exception_class,
                a.exception_message,
                a.created_at_utc
            FROM batch_attempts AS a
            LEFT JOIN (
                SELECT DISTINCT batch_id, attempt_no
                FROM batch_attempt_events
            ) AS e
              ON e.batch_id = a.batch_id
             AND e.attempt_no = a.attempt_no
            WHERE e.batch_id IS NULL
            ORDER BY a.batch_id, a.attempt_no
            """
        ).fetchall()
        for row in rows:
            headers = {}
            if row["headers_json"] is not None:
                try:
                    parsed_headers = json.loads(row["headers_json"])
                    headers = parsed_headers if isinstance(parsed_headers, dict) else {}
                except Exception:
                    headers = {}
            self._insert_attempt_event(
                conn,
                batch_id=str(row["batch_id"]),
                attempt_no=int(row["attempt_no"]),
                phase=ATTEMPT_PHASE_LEGACY_FINISH_BACKFILL,
                claim_token=None,
                headers=headers,
                status_code=None if row["status_code"] is None else int(row["status_code"]),
                latency_ms=None if row["latency_ms"] is None else int(row["latency_ms"]),
                rows_returned=None if row["rows_returned"] is None else int(row["rows_returned"]),
                response_bytes=None if row["response_bytes"] is None else int(row["response_bytes"]),
                exception_class=None if row["exception_class"] is None else str(row["exception_class"]),
                exception_message=None if row["exception_message"] is None else str(row["exception_message"]),
            )
            conn.execute(
                """
                UPDATE batch_attempt_events
                SET created_at_utc = ?
                WHERE event_id = last_insert_rowid()
                """,
                (str(row["created_at_utc"]),),
            )

    def _increment_item_attempts(
        self,
        conn: sqlite3.Connection,
        *,
        item_ids: tuple[str, ...],
        updated_at_utc: str,
    ) -> None:
        for item_id in item_ids:
            conn.execute(
                """
                UPDATE request_items
                SET state = ?, attempt_count = attempt_count + 1, updated_at_utc = ?, next_eligible_at_utc = NULL
                WHERE item_id = ?
                """,
                (
                    LEDGER_RUNNING,
                    updated_at_utc,
                    item_id,
                ),
            )

    def _update_items_for_ids(
        self,
        conn: sqlite3.Connection,
        *,
        item_ids: tuple[str, ...],
        state: str,
        last_error: str | None,
        next_eligible_at_utc: str | None,
        updated_at_utc: str,
    ) -> None:
        for item_id in item_ids:
            conn.execute(
                """
                UPDATE request_items
                SET state = ?, last_error = ?, next_eligible_at_utc = ?, updated_at_utc = ?
                WHERE item_id = ?
                """,
                (
                    state,
                    last_error,
                    next_eligible_at_utc,
                    updated_at_utc,
                    item_id,
                ),
            )

    def _runtime_halt_active(self, conn: sqlite3.Connection, *, stage: str) -> bool:
        row = conn.execute(
            """
            SELECT halt_new_work
            FROM stage_runtime_control
            WHERE stage = ?
            """,
            (stage,),
        ).fetchone()
        if row is None:
            return False
        return bool(int(row["halt_new_work"] or 0))

    def _has_claimable_batch(self, conn: sqlite3.Connection, *, stage: str, now_text: str) -> bool:
        row = conn.execute(
            """
            SELECT 1
            FROM batches
            WHERE stage = ?
              AND state IN (?, ?)
              AND (next_eligible_at_utc IS NULL OR next_eligible_at_utc <= ?)
            LIMIT 1
            """,
            (
                stage,
                LEDGER_PENDING,
                LEDGER_RETRYABLE_ERROR,
                now_text,
            ),
        ).fetchone()
        return row is not None

    def _has_outstanding_batch(self, conn: sqlite3.Connection, *, stage: str) -> bool:
        row = conn.execute(
            """
            SELECT 1
            FROM batches
            WHERE stage = ?
              AND state IN (?, ?, ?)
            LIMIT 1
            """,
            (
                stage,
                LEDGER_PENDING,
                LEDGER_RETRYABLE_ERROR,
                LEDGER_RUNNING,
            ),
        ).fetchone()
        return row is not None

    def _runtime_idle_wait_seconds(
        self,
        conn: sqlite3.Connection,
        *,
        stage: str,
        now_dt: dt.datetime,
        idle_poll_seconds: float,
    ) -> float:
        running_row = conn.execute(
            """
            SELECT 1
            FROM batches
            WHERE stage = ? AND state = ?
            LIMIT 1
            """,
            (stage, LEDGER_RUNNING),
        ).fetchone()
        if running_row is not None:
            return idle_poll_seconds

        next_row = conn.execute(
            """
            SELECT next_eligible_at_utc
            FROM batches
            WHERE stage = ?
              AND state IN (?, ?)
              AND next_eligible_at_utc IS NOT NULL
            ORDER BY next_eligible_at_utc
            LIMIT 1
            """,
            (
                stage,
                LEDGER_PENDING,
                LEDGER_RETRYABLE_ERROR,
            ),
        ).fetchone()
        if next_row is None or next_row["next_eligible_at_utc"] is None:
            return idle_poll_seconds

        next_eligible = from_utc_text(str(next_row["next_eligible_at_utc"]))
        if next_eligible is None:
            return idle_poll_seconds
        return max(0.0, (next_eligible - now_dt).total_seconds())

    def _item_ids_from_row(self, row: sqlite3.Row) -> tuple[str, ...]:
        return tuple(str(item_id) for item_id in json.loads(row["item_ids_json"]))

    def _ensure_column(
        self,
        conn: sqlite3.Connection,
        *,
        table: str,
        column: str,
        column_type: str,
    ) -> None:
        columns = {
            str(row["name"])
            for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column in columns:
            return
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")

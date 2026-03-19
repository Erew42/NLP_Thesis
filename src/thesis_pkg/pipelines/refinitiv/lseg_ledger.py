from __future__ import annotations

import contextlib
import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Any

from thesis_pkg.pipelines.refinitiv.lseg_batching import BatchDefinition, RequestItem, stable_json_dumps


LEDGER_PENDING = "pending"
LEDGER_RUNNING = "running"
LEDGER_SUCCEEDED = "succeeded"
LEDGER_RETRYABLE_ERROR = "retryable_error"
LEDGER_DEFERRED_DAILY_LIMIT = "deferred_daily_limit"
LEDGER_FATAL_ERROR = "fatal_error"
LEDGER_TERMINAL_STATES = (LEDGER_SUCCEEDED, LEDGER_DEFERRED_DAILY_LIMIT, LEDGER_FATAL_ERROR)


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


class RequestLedger:
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
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

    def requeue_stale_running(self, *, older_than_seconds: int = 900) -> int:
        cutoff = utc_now() - dt.timedelta(seconds=older_than_seconds)
        cutoff_text = to_utc_text(cutoff)
        now_text = to_utc_text(utc_now())
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE batches
                SET state = ?, updated_at_utc = ?
                WHERE state = ? AND updated_at_utc < ?
                """,
                (
                    LEDGER_PENDING,
                    now_text,
                    LEDGER_RUNNING,
                    cutoff_text,
                ),
            )
            return int(cursor.rowcount or 0)

    def requeue_known_fixable_fatal_batches(self, *, stage: str | None = None) -> int:
        now_text = to_utc_text(utc_now())
        predicates = (
            "last_error LIKE 'Unable to resolve all requested identifiers%'",
            "last_error LIKE 'could not append value:%'",
            "last_error LIKE 'failed to convert LSEG response to Polars:%'",
        )
        where_clause = " OR ".join(predicates)
        with self._connect() as conn:
            if stage is None:
                cursor = conn.execute(
                    f"""
                    UPDATE batches
                    SET state = ?, last_error = NULL, next_eligible_at_utc = NULL, updated_at_utc = ?
                    WHERE state = ? AND ({where_clause})
                    """,
                    (
                        LEDGER_PENDING,
                        now_text,
                        LEDGER_FATAL_ERROR,
                    ),
                )
            else:
                cursor = conn.execute(
                    f"""
                    UPDATE batches
                    SET state = ?, last_error = NULL, next_eligible_at_utc = NULL, updated_at_utc = ?
                    WHERE stage = ? AND state = ? AND ({where_clause})
                    """,
                    (
                        LEDGER_PENDING,
                        now_text,
                        stage,
                        LEDGER_FATAL_ERROR,
                    ),
                )
            return int(cursor.rowcount or 0)

    def claim_next_batch(self, *, stage: str) -> LedgerBatch | None:
        now_text = to_utc_text(utc_now())
        with self._connect() as conn:
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
            conn.execute(
                """
                UPDATE batches
                SET state = ?, updated_at_utc = ?, attempt_count = attempt_count + 1
                WHERE batch_id = ?
                """,
                (
                    LEDGER_RUNNING,
                    now_text,
                    row["batch_id"],
                ),
            )
            updated_row = conn.execute(
                "SELECT * FROM batches WHERE batch_id = ?",
                (row["batch_id"],),
            ).fetchone()
            if updated_row is None:
                return None
            return self._batch_from_row(updated_row)

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
            item_ids = tuple(json.loads(batch_row["item_ids_json"]))
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
                    next_eligible_at_utc = NULL, updated_at_utc = ?
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
                attempt_no=int(batch_row["attempt_count"]),
                status_code=status_code,
                latency_ms=latency_ms,
                rows_returned=rows_returned,
                response_bytes=response_bytes,
                headers=headers,
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
            conn.execute(
                """
                UPDATE batches
                SET state = ?, last_error = ?, header_json = ?, last_status_code = ?,
                    last_latency_ms = ?, response_bytes = ?, next_eligible_at_utc = ?, updated_at_utc = ?
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
            self._insert_attempt_row(
                conn,
                batch_id=batch_id,
                attempt_no=int(batch_row["attempt_count"]),
                status_code=status_code,
                latency_ms=latency_ms,
                rows_returned=None,
                response_bytes=response_bytes,
                headers=headers,
                exception_class=exception_class,
                exception_message=error_message,
            )

    def split_batch(self, *, parent_batch_id: str, child_batches: list[BatchDefinition], reason: str) -> None:
        now_text = to_utc_text(utc_now())
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE batches
                SET state = ?, last_error = ?, updated_at_utc = ?
                WHERE batch_id = ?
                """,
                (
                    LEDGER_FATAL_ERROR,
                    f"split_into_children:{reason}",
                    now_text,
                    parent_batch_id,
                ),
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
            return int(cursor.rowcount or 0)

    def _initialize(self) -> None:
        with self._connect() as conn:
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

    @contextlib.contextmanager
    def _connect(self) -> Any:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        try:
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

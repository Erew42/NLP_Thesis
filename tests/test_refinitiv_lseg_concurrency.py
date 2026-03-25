from __future__ import annotations

import datetime as dt
import json
import multiprocessing as mp
import os
from pathlib import Path
import sqlite3
import time

import polars as pl
import pytest

from thesis_pkg.pipelines.refinitiv.lseg_api_common import append_json_log
from thesis_pkg.pipelines.refinitiv.lseg_api_execution import run_api_batches
from thesis_pkg.pipelines.refinitiv.lseg_batching import RequestItem, batch_items
from thesis_pkg.pipelines.refinitiv.lseg_ledger import RequestLedger
from thesis_pkg.pipelines.refinitiv.lseg_provider import (
    LsegDataResponse,
    LsegRequestError,
    LsegResponseMetadata,
)


def _zero_retry_delay(_: int) -> float:
    return 0.0


def _identity_lookup_normalizer(value: object) -> str | None:
    return None if value is None else str(value)


def _simple_response_normalizer(batch_items_rows: list[RequestItem], frame: pl.DataFrame) -> pl.DataFrame:
    value_by_instrument = {
        str(row["Instrument"]): float(row["Value"])
        for row in frame.to_dicts()
    }
    return pl.DataFrame(
        {
            "item_id": [item.item_id for item in batch_items_rows],
            "instrument": [item.instrument for item in batch_items_rows],
            "value": [value_by_instrument.get(item.instrument) for item in batch_items_rows],
        }
    )


def _make_items(*, count: int, batch_key: str = "shared", fields: tuple[str, ...] = ("TR.Field",)) -> list[RequestItem]:
    return [
        RequestItem(
            item_id=f"item-{idx}",
            stage="test_stage",
            instrument=f"RIC{idx}.N",
            batch_key=batch_key,
            fields=fields,
            parameters={},
            payload={"instrument": f"RIC{idx}.N"},
        )
        for idx in range(1, count + 1)
    ]


def _read_staging_dir(staging_dir: Path) -> pl.DataFrame:
    frames = [pl.read_parquet(path) for path in sorted(staging_dir.glob("*.parquet"))]
    if not frames:
        return pl.DataFrame({"item_id": [], "instrument": [], "value": []})
    return pl.concat(frames, how="vertical").sort("item_id")


def _init_provider_state_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS counters (
                name TEXT PRIMARY KEY,
                value INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS call_counts (
                call_key TEXT PRIMARY KEY,
                value INTEGER NOT NULL
            )
            """
        )
        conn.execute("INSERT OR REPLACE INTO counters (name, value) VALUES ('active_count', 0)")
        conn.execute("INSERT OR REPLACE INTO counters (name, value) VALUES ('max_active_count', 0)")
        conn.execute("INSERT OR REPLACE INTO counters (name, value) VALUES ('global_call_count', 0)")
        conn.commit()
    finally:
        conn.close()


def _counter_value(path: Path, name: str) -> int:
    conn = sqlite3.connect(path)
    try:
        row = conn.execute("SELECT value FROM counters WHERE name = ?", (name,)).fetchone()
    finally:
        conn.close()
    if row is None:
        raise KeyError(name)
    return int(row[0])


def _provider_start_call(path: Path, *, call_key: str) -> tuple[int, int, int]:
    conn = sqlite3.connect(path, timeout=30.0)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("BEGIN IMMEDIATE")
        global_call_count = int(
            conn.execute("SELECT value FROM counters WHERE name = 'global_call_count'").fetchone()[0]
        ) + 1
        active_count = int(
            conn.execute("SELECT value FROM counters WHERE name = 'active_count'").fetchone()[0]
        ) + 1
        max_active_count = int(
            conn.execute("SELECT value FROM counters WHERE name = 'max_active_count'").fetchone()[0]
        )
        row = conn.execute("SELECT value FROM call_counts WHERE call_key = ?", (call_key,)).fetchone()
        call_key_count = 1 if row is None else int(row[0]) + 1
        conn.execute(
            "INSERT OR REPLACE INTO counters (name, value) VALUES ('global_call_count', ?)",
            (global_call_count,),
        )
        conn.execute(
            "INSERT OR REPLACE INTO counters (name, value) VALUES ('active_count', ?)",
            (active_count,),
        )
        conn.execute(
            "INSERT OR REPLACE INTO counters (name, value) VALUES ('max_active_count', ?)",
            (max(max_active_count, active_count),),
        )
        conn.execute(
            "INSERT OR REPLACE INTO call_counts (call_key, value) VALUES (?, ?)",
            (call_key, call_key_count),
        )
        conn.commit()
        return global_call_count, call_key_count, active_count
    finally:
        conn.close()


def _provider_finish_call(path: Path) -> int:
    conn = sqlite3.connect(path, timeout=30.0)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("BEGIN IMMEDIATE")
        active_count = int(
            conn.execute("SELECT value FROM counters WHERE name = 'active_count'").fetchone()[0]
        ) - 1
        conn.execute(
            "INSERT OR REPLACE INTO counters (name, value) VALUES ('active_count', ?)",
            (max(active_count, 0),),
        )
        conn.commit()
        return max(active_count, 0)
    finally:
        conn.close()


class _ScriptedConcurrentProvider:
    def __init__(
        self,
        *,
        state_db_path: str,
        events_path: str,
        sleep_seconds: float,
        fail_once_for_multi_item: bool,
        daily_limit_after_first_success: bool,
        fatal_on_first_call: bool,
    ) -> None:
        self._state_db_path = Path(state_db_path)
        self._events_path = Path(events_path)
        self._sleep_seconds = sleep_seconds
        self._fail_once_for_multi_item = fail_once_for_multi_item
        self._daily_limit_after_first_success = daily_limit_after_first_success
        self._fatal_on_first_call = fatal_on_first_call

    def open(self) -> None:
        append_json_log(
            self._events_path,
            {"event": "provider_opened", "pid": os.getpid()},
        )

    def close(self) -> None:
        append_json_log(
            self._events_path,
            {"event": "provider_closed", "pid": os.getpid()},
        )

    def get_data(
        self,
        *,
        universe: list[str],
        fields: list[str],
        parameters: dict[str, object] | None = None,
    ) -> LsegDataResponse:
        call_key = "|".join(universe)
        global_call_no, call_key_count, active_count = _provider_start_call(
            self._state_db_path,
            call_key=call_key,
        )
        started_at_epoch = time.time()
        append_json_log(
            self._events_path,
            {
                "event": "provider_call_started",
                "pid": os.getpid(),
                "global_call_no": global_call_no,
                "call_key_count": call_key_count,
                "universe": list(universe),
                "fields": list(fields),
                "parameters": dict(parameters or {}),
                "started_at_epoch": started_at_epoch,
                "active_count": active_count,
            },
        )
        try:
            if self._sleep_seconds > 0:
                time.sleep(self._sleep_seconds)
            if self._fatal_on_first_call and global_call_no == 1:
                raise ValueError("forced fatal stop-stage error")
            if self._fail_once_for_multi_item and len(universe) > 1 and call_key_count == 1:
                raise LsegRequestError("request timed out", error_kind="transport_timeout")

            headers = {"X-Request-Limit-Remaining": "100"}
            if self._daily_limit_after_first_success and global_call_no == 1:
                headers = {"X-Request-Limit-Remaining": "0"}
            frame = pl.DataFrame(
                {
                    "Instrument": list(universe),
                    "Value": [float(index) for index, _ in enumerate(universe, start=1)],
                }
            )
            return LsegDataResponse(
                frame=frame,
                metadata=LsegResponseMetadata(
                    status_code=200,
                    headers=headers,
                    latency_ms=int(self._sleep_seconds * 1000),
                    response_bytes=128,
                    fingerprint=f"fp-{global_call_no}",
                ),
            )
        finally:
            finished_at_epoch = time.time()
            remaining_active = _provider_finish_call(self._state_db_path)
            append_json_log(
                self._events_path,
                {
                    "event": "provider_call_finished",
                    "pid": os.getpid(),
                    "global_call_no": global_call_no,
                    "finished_at_epoch": finished_at_epoch,
                    "active_count": remaining_active,
                },
            )


class _ConcurrentProviderFactory:
    def __init__(
        self,
        *,
        state_db_path: str,
        events_path: str,
        sleep_seconds: float = 0.0,
        fail_once_for_multi_item: bool = False,
        daily_limit_after_first_success: bool = False,
        fatal_on_first_call: bool = False,
    ) -> None:
        self.state_db_path = state_db_path
        self.events_path = events_path
        self.sleep_seconds = sleep_seconds
        self.fail_once_for_multi_item = fail_once_for_multi_item
        self.daily_limit_after_first_success = daily_limit_after_first_success
        self.fatal_on_first_call = fatal_on_first_call

    def __call__(self) -> _ScriptedConcurrentProvider:
        return _ScriptedConcurrentProvider(
            state_db_path=self.state_db_path,
            events_path=self.events_path,
            sleep_seconds=self.sleep_seconds,
            fail_once_for_multi_item=self.fail_once_for_multi_item,
            daily_limit_after_first_success=self.daily_limit_after_first_success,
            fatal_on_first_call=self.fatal_on_first_call,
        )


def _claim_once_worker(ledger_path: str, start_event: Any, result_queue: Any) -> None:
    ledger = RequestLedger(Path(ledger_path), run_session_id="claim-race")
    start_event.wait()
    claimed = ledger.claim_next_batch(stage="test_stage")
    result_queue.put(None if claimed is None else claimed.batch_id)


def _provider_events(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _iso_to_epoch(value: str) -> float:
    normalized = value.replace("Z", "+00:00")
    return float(dt.datetime.fromisoformat(normalized).timestamp())


def test_request_ledger_atomic_claim_under_process_race(tmp_path: Path) -> None:
    item = RequestItem(
        item_id="item-1",
        stage="test_stage",
        instrument="AAA.N",
        batch_key="single",
        fields=("TR.Field",),
        parameters={},
        payload={"row": 1},
    )
    batch = batch_items([item], max_batch_size=10, unique_instrument_limit=True)[0]
    ledger_path = tmp_path / "ledger.sqlite3"
    ledger = RequestLedger(ledger_path)
    ledger.enqueue([item], [batch])

    ctx = mp.get_context("spawn")
    start_event = ctx.Event()
    result_queue: Any = ctx.Queue()
    processes = [
        ctx.Process(target=_claim_once_worker, args=(str(ledger_path), start_event, result_queue))
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    start_event.set()
    results = [result_queue.get(timeout=10) for _ in processes]
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    claimed_batch_ids = [result for result in results if result is not None]
    assert claimed_batch_ids == [batch.batch_id]
    assert ledger.state_counts(table="batches") == {"running": 1}


def test_run_api_batches_concurrent_scheduler_enforces_start_gap_and_overlap(tmp_path: Path) -> None:
    items = _make_items(count=4)
    state_db = tmp_path / "provider_state.sqlite3"
    events_path = tmp_path / "provider_events.jsonl"
    _init_provider_state_db(state_db)
    provider_factory = _ConcurrentProviderFactory(
        state_db_path=str(state_db),
        events_path=str(events_path),
        sleep_seconds=0.25,
    )

    result = run_api_batches(
        stage="test_stage",
        items=items,
        output_dir=tmp_path,
        ledger_path=tmp_path / "ledger.sqlite3",
        request_log_path=tmp_path / "requests.jsonl",
        provider=None,
        provider_factory=provider_factory,
        provider_session_name="desktop.workspace",
        provider_config_name=None,
        provider_timeout_seconds=None,
        preflight_probe=False,
        max_batch_size=1,
        min_seconds_between_requests=0.0,
        min_seconds_between_request_starts_total=0.1,
        max_attempts=2,
        max_workers=2,
        response_normalizer=_simple_response_normalizer,
        lookup_normalizer=_identity_lookup_normalizer,
        retry_delay_seconds_fn=_zero_retry_delay,
    )

    events = _provider_events(events_path)
    started = sorted(
        (event for event in events if event["event"] == "provider_call_started"),
        key=lambda event: float(event["started_at_epoch"]),
    )
    finished = {
        int(event["global_call_no"]): float(event["finished_at_epoch"])
        for event in events
        if event["event"] == "provider_call_finished"
    }
    assert len(started) == 4
    assert float(started[1]["started_at_epoch"]) < finished[int(started[0]["global_call_no"])]
    assert _counter_value(state_db, "max_active_count") == 2

    parsed_request_log = [
        json.loads(line)
        for line in Path(result.request_log_path).read_text(encoding="utf-8").splitlines()
    ]
    succeeded = sorted(
        (entry for entry in parsed_request_log if entry["event"] == "request_succeeded"),
        key=lambda entry: _iso_to_epoch(str(entry["scheduled_start_utc"])),
    )
    assert len(succeeded) == 4
    scheduled_gaps = [
        _iso_to_epoch(str(succeeded[index]["scheduled_start_utc"]))
        - _iso_to_epoch(str(succeeded[index - 1]["scheduled_start_utc"]))
        for index in range(1, len(succeeded))
    ]
    assert all(gap >= 0.09 for gap in scheduled_gaps)

    provider_opened_pids = {int(event["pid"]) for event in events if event["event"] == "provider_opened"}
    provider_closed_pids = {int(event["pid"]) for event in events if event["event"] == "provider_closed"}
    assert len(provider_opened_pids) == 2
    assert provider_opened_pids == provider_closed_pids


def test_run_api_batches_single_worker_keeps_completion_gap_semantics(tmp_path: Path) -> None:
    items = _make_items(count=2)
    state_db = tmp_path / "provider_state.sqlite3"
    events_path = tmp_path / "provider_events.jsonl"
    _init_provider_state_db(state_db)
    provider = _ConcurrentProviderFactory(
        state_db_path=str(state_db),
        events_path=str(events_path),
        sleep_seconds=0.05,
    )()

    run_api_batches(
        stage="test_stage",
        items=items,
        output_dir=tmp_path,
        ledger_path=tmp_path / "ledger.sqlite3",
        request_log_path=tmp_path / "requests.jsonl",
        provider=provider,
        provider_session_name="desktop.workspace",
        provider_config_name=None,
        provider_timeout_seconds=None,
        preflight_probe=False,
        max_batch_size=1,
        min_seconds_between_requests=0.15,
        max_attempts=2,
        response_normalizer=_simple_response_normalizer,
        lookup_normalizer=_identity_lookup_normalizer,
        retry_delay_seconds_fn=_zero_retry_delay,
    )

    started = sorted(
        (
            event
            for event in _provider_events(events_path)
            if event["event"] == "provider_call_started"
        ),
        key=lambda event: float(event["started_at_epoch"]),
    )
    assert len(started) == 2
    start_gap = float(started[1]["started_at_epoch"]) - float(started[0]["started_at_epoch"])
    assert start_gap >= 0.18


def test_run_api_batches_parallel_matches_sequential_output(tmp_path: Path) -> None:
    items = _make_items(count=4)

    sequential_state_db = tmp_path / "seq_provider_state.sqlite3"
    sequential_events_path = tmp_path / "seq_provider_events.jsonl"
    _init_provider_state_db(sequential_state_db)
    sequential_provider = _ConcurrentProviderFactory(
        state_db_path=str(sequential_state_db),
        events_path=str(sequential_events_path),
        sleep_seconds=0.01,
    )()
    sequential_result = run_api_batches(
        stage="test_stage",
        items=items,
        output_dir=tmp_path / "sequential",
        ledger_path=tmp_path / "sequential" / "ledger.sqlite3",
        request_log_path=tmp_path / "sequential" / "requests.jsonl",
        provider=sequential_provider,
        provider_session_name="desktop.workspace",
        provider_config_name=None,
        provider_timeout_seconds=None,
        preflight_probe=False,
        max_batch_size=1,
        min_seconds_between_requests=0.0,
        max_attempts=2,
        response_normalizer=_simple_response_normalizer,
        lookup_normalizer=_identity_lookup_normalizer,
        retry_delay_seconds_fn=_zero_retry_delay,
    )

    concurrent_state_db = tmp_path / "con_provider_state.sqlite3"
    concurrent_events_path = tmp_path / "con_provider_events.jsonl"
    _init_provider_state_db(concurrent_state_db)
    concurrent_provider_factory = _ConcurrentProviderFactory(
        state_db_path=str(concurrent_state_db),
        events_path=str(concurrent_events_path),
        sleep_seconds=0.01,
    )
    concurrent_result = run_api_batches(
        stage="test_stage",
        items=items,
        output_dir=tmp_path / "concurrent",
        ledger_path=tmp_path / "concurrent" / "ledger.sqlite3",
        request_log_path=tmp_path / "concurrent" / "requests.jsonl",
        provider=None,
        provider_factory=concurrent_provider_factory,
        provider_session_name="desktop.workspace",
        provider_config_name=None,
        provider_timeout_seconds=None,
        preflight_probe=False,
        max_batch_size=1,
        min_seconds_between_requests=0.0,
        min_seconds_between_request_starts_total=0.01,
        max_attempts=2,
        max_workers=2,
        response_normalizer=_simple_response_normalizer,
        lookup_normalizer=_identity_lookup_normalizer,
        retry_delay_seconds_fn=_zero_retry_delay,
    )

    assert _read_staging_dir(sequential_result.staging_dir).to_dicts() == _read_staging_dir(
        concurrent_result.staging_dir
    ).to_dicts()


def test_run_api_batches_parallel_preserves_split_children_processing(tmp_path: Path) -> None:
    items = _make_items(count=2)
    state_db = tmp_path / "provider_state.sqlite3"
    events_path = tmp_path / "provider_events.jsonl"
    _init_provider_state_db(state_db)
    provider_factory = _ConcurrentProviderFactory(
        state_db_path=str(state_db),
        events_path=str(events_path),
        sleep_seconds=0.01,
        fail_once_for_multi_item=True,
    )

    result = run_api_batches(
        stage="test_stage",
        items=items,
        output_dir=tmp_path,
        ledger_path=tmp_path / "ledger.sqlite3",
        request_log_path=tmp_path / "requests.jsonl",
        provider=None,
        provider_factory=provider_factory,
        provider_session_name="desktop.workspace",
        provider_config_name=None,
        provider_timeout_seconds=None,
        preflight_probe=False,
        max_batch_size=2,
        min_seconds_between_requests=0.0,
        min_seconds_between_request_starts_total=0.01,
        max_attempts=3,
        max_workers=2,
        response_normalizer=_simple_response_normalizer,
        lookup_normalizer=_identity_lookup_normalizer,
        split_after_attempt=1,
        retry_delay_seconds_fn=_zero_retry_delay,
    )

    ledger = RequestLedger(result.ledger_path)
    assert ledger.state_counts(table="batches") == {"fatal_error": 1, "succeeded": 2}
    assert len(list(result.staging_dir.glob("*.parquet"))) == 2


def test_run_api_batches_parallel_daily_limit_halt_stops_new_claims(tmp_path: Path) -> None:
    items = _make_items(count=4)
    state_db = tmp_path / "provider_state.sqlite3"
    events_path = tmp_path / "provider_events.jsonl"
    _init_provider_state_db(state_db)
    provider_factory = _ConcurrentProviderFactory(
        state_db_path=str(state_db),
        events_path=str(events_path),
        sleep_seconds=0.2,
        daily_limit_after_first_success=True,
    )

    result = run_api_batches(
        stage="test_stage",
        items=items,
        output_dir=tmp_path,
        ledger_path=tmp_path / "ledger.sqlite3",
        request_log_path=tmp_path / "requests.jsonl",
        provider=None,
        provider_factory=provider_factory,
        provider_session_name="desktop.workspace",
        provider_config_name=None,
        provider_timeout_seconds=None,
        preflight_probe=False,
        max_batch_size=1,
        min_seconds_between_requests=0.0,
        min_seconds_between_request_starts_total=0.05,
        max_attempts=2,
        max_workers=2,
        response_normalizer=_simple_response_normalizer,
        lookup_normalizer=_identity_lookup_normalizer,
        retry_delay_seconds_fn=_zero_retry_delay,
    )

    ledger = RequestLedger(result.ledger_path)
    assert ledger.state_counts(table="batches") == {"deferred_daily_limit": 2, "succeeded": 2}
    request_log = [
        json.loads(line)
        for line in Path(result.request_log_path).read_text(encoding="utf-8").splitlines()
    ]
    succeeded_events = [entry for entry in request_log if entry["event"] == "request_succeeded"]
    assert len(succeeded_events) == 2


def test_run_api_batches_parallel_recovers_stale_running_batches(tmp_path: Path) -> None:
    items = _make_items(count=2)
    ledger_path = tmp_path / "ledger.sqlite3"
    request_log_path = tmp_path / "requests.jsonl"
    output_dir = tmp_path / "output"
    staging_dir = output_dir / "staging" / "test_stage"
    staging_dir.mkdir(parents=True, exist_ok=True)
    planned_batches = batch_items(items, max_batch_size=1, unique_instrument_limit=True)
    ledger = RequestLedger(ledger_path)
    ledger.enqueue(items, planned_batches)
    claimed = ledger.claim_next_batch(stage="test_stage")
    assert claimed is not None

    conn = sqlite3.connect(ledger_path)
    try:
        conn.execute(
            "UPDATE batches SET updated_at_utc = '2000-01-01T00:00:00Z' WHERE batch_id = ?",
            (claimed.batch_id,),
        )
        conn.commit()
    finally:
        conn.close()

    state_db = tmp_path / "provider_state.sqlite3"
    events_path = tmp_path / "provider_events.jsonl"
    _init_provider_state_db(state_db)
    provider_factory = _ConcurrentProviderFactory(
        state_db_path=str(state_db),
        events_path=str(events_path),
        sleep_seconds=0.01,
    )

    result = run_api_batches(
        stage="test_stage",
        items=items,
        output_dir=output_dir,
        ledger_path=ledger_path,
        request_log_path=request_log_path,
        provider=None,
        provider_factory=provider_factory,
        provider_session_name="desktop.workspace",
        provider_config_name=None,
        provider_timeout_seconds=None,
        preflight_probe=False,
        max_batch_size=1,
        min_seconds_between_requests=0.0,
        min_seconds_between_request_starts_total=0.01,
        max_attempts=2,
        max_workers=2,
        response_normalizer=_simple_response_normalizer,
        lookup_normalizer=_identity_lookup_normalizer,
        retry_delay_seconds_fn=_zero_retry_delay,
    )

    final_ledger = RequestLedger(result.ledger_path)
    assert final_ledger.state_counts(table="batches") == {"succeeded": 2}

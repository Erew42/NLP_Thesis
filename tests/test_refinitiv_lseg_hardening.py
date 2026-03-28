from __future__ import annotations

from datetime import date
import sqlite3
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from thesis_pkg.pipelines.refinitiv import lseg_lookup_api
from thesis_pkg.pipelines.refinitiv.doc_ownership import DOC_OWNERSHIP_RAW_COLUMNS
from thesis_pkg.pipelines.refinitiv.lseg_analyst_api import _assemble_analyst_actuals_raw
from thesis_pkg.pipelines.refinitiv.lseg_ownership_api import (
    _assemble_doc_raw,
    _assemble_ownership_universe_results,
    DOC_EXACT_STAGE,
    run_refinitiv_step1_ownership_universe_api_pipeline,
)
from thesis_pkg.pipelines.refinitiv.lseg_api_execution import run_api_batches
from thesis_pkg.pipelines.refinitiv.lseg_batching import RequestItem, batch_items, build_batch_definition
from thesis_pkg.pipelines.refinitiv.lseg_ledger import RequestLedger
from thesis_pkg.pipelines.refinitiv.lseg_provider import (
    LsegDataResponse,
    LsegRequestError,
    LsegResponseMetadata,
)
from thesis_pkg.pipelines.refinitiv.lseg_ledger import LsegResumeCompatibilityError
from thesis_pkg.pipelines.refinitiv.lseg_recovery import (
    build_doc_unresolved_recovery_artifact,
    build_lookup_unresolved_recovery_artifact,
    build_ownership_unresolved_recovery_artifact,
)
from thesis_pkg.pipelines.refinitiv.lseg_stage_audit import AuditIssue, StageAuditResult, audit_api_stage
from thesis_pkg.pipeline import run_refinitiv_step1_lookup_api_pipeline
from thesis_pkg.pipelines.refinitiv_bridge_pipeline import (
    OWNERSHIP_UNIVERSE_HANDOFF_COLUMNS,
    RIC_LOOKUP_COLUMNS,
    build_refinitiv_lookup_extended_diagnostic_artifact,
)


class TimeoutThenSuccessProvider:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def get_data(
        self,
        *,
        universe: list[str],
        fields: list[str],
        parameters: dict[str, Any] | None = None,
    ) -> LsegDataResponse:
        self.calls.append({"universe": list(universe), "fields": list(fields), "parameters": dict(parameters or {})})
        if len(self.calls) == 1:
            raise LsegRequestError("request timed out", error_kind="transport_timeout")
        return LsegDataResponse(
            frame=pl.DataFrame({"Instrument": list(universe), "Value": [1.0 for _ in universe]}),
            metadata=LsegResponseMetadata(
                status_code=200,
                headers={"X-Request-Limit-Remaining": "100"},
                latency_ms=10,
                response_bytes=64,
                fingerprint="ok",
            ),
        )


class EmptySuccessProvider:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def get_data(
        self,
        *,
        universe: list[str],
        fields: list[str],
        parameters: dict[str, Any] | None = None,
    ) -> LsegDataResponse:
        self.calls.append({"universe": list(universe), "fields": list(fields), "parameters": dict(parameters or {})})
        return LsegDataResponse(
            frame=pl.DataFrame({"Instrument": []}),
            metadata=LsegResponseMetadata(
                status_code=200,
                headers={"X-Request-Limit-Remaining": "100"},
                latency_ms=10,
                response_bytes=64,
                fingerprint="ok",
            ),
        )


def _ownership_handoff_row(
    *,
    lookup_row_id: str,
    candidate_ric: str,
    request_start_date: str,
    request_end_date: str,
) -> dict[str, Any]:
    handoff_row = {name: None for name in OWNERSHIP_UNIVERSE_HANDOFF_COLUMNS}
    handoff_row.update({
        "bridge_row_id": lookup_row_id.replace("|", ":"),
        "KYPERMNO": "1",
        "CUSIP": "111111111",
        "ISIN": "US1111111111",
        "TICKER": "ALFA",
        "first_seen_caldt": date.fromisoformat(request_start_date),
        "last_seen_caldt": date.fromisoformat(request_end_date),
        "accepted_ric": candidate_ric,
        "accepted_ric_source": "ISIN",
        "accepted_resolution_status": "resolved_from_isin",
        "conventional_identity_conflict": False,
        "ticker_candidate_available": False,
        "effective_collection_ric": candidate_ric,
        "effective_collection_ric_source": "ISIN",
        "effective_resolution_status": "effective_from_accepted_ric",
        "diagnostic_case_id": lookup_row_id.replace("|", ":"),
        "candidate_slot": "UNIVERSE_EFFECTIVE",
        "candidate_ric": candidate_ric,
        "ownership_lookup_row_id": lookup_row_id,
        "ownership_lookup_role": "UNIVERSE_EFFECTIVE",
        "lookup_input": candidate_ric,
        "lookup_input_source": "effective_collection_ric",
        "request_start_date": request_start_date,
        "request_end_date": request_end_date,
        "retrieval_eligible": True,
    })
    return handoff_row


def test_request_ledger_records_claim_and_stale_requeue_history(tmp_path: Path) -> None:
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
    ledger = RequestLedger(ledger_path, run_session_id="run-test", worker_id="worker-test")
    ledger.enqueue([item], [batch])

    claimed = ledger.claim_next_batch(stage="test_stage")
    assert claimed is not None

    conn = sqlite3.connect(ledger_path)
    conn.row_factory = sqlite3.Row
    try:
        phases = [
            row["phase"]
            for row in conn.execute(
                "SELECT phase FROM batch_attempt_events WHERE batch_id = ? ORDER BY event_id",
                (claimed.batch_id,),
            ).fetchall()
        ]
        assert phases == ["claimed"]
        conn.execute(
            "UPDATE batches SET updated_at_utc = '2000-01-01T00:00:00Z' WHERE batch_id = ?",
            (claimed.batch_id,),
        )
        conn.commit()
    finally:
        conn.close()

    assert ledger.requeue_stale_running(older_than_seconds=0) == 1

    conn = sqlite3.connect(ledger_path)
    conn.row_factory = sqlite3.Row
    try:
        phases = [
            row["phase"]
            for row in conn.execute(
                "SELECT phase FROM batch_attempt_events WHERE batch_id = ? ORDER BY event_id",
                (claimed.batch_id,),
            ).fetchall()
        ]
        item_state = conn.execute("SELECT state FROM request_items WHERE item_id = 'item-1'").fetchone()["state"]
        assert phases[-1] == "requeued_stale_running"
        assert item_state == "pending"
    finally:
        conn.close()


def test_request_ledger_backfills_missing_attempt_events_for_legacy_attempt_rows(tmp_path: Path) -> None:
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
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    stage_path = staging_dir / f"{batch.batch_id}.parquet"
    pl.DataFrame({"item_id": ["item-1"], "value": [1.0]}).write_parquet(stage_path)

    ledger = RequestLedger(ledger_path, run_session_id="run-test", worker_id="worker-test")
    ledger.enqueue([item], [batch])
    claimed = ledger.claim_next_batch(stage="test_stage")
    assert claimed is not None
    ledger.record_success(
        batch_id=claimed.batch_id,
        row_count_by_item_id={"item-1": 1},
        stage_output_path=stage_path,
        response_fingerprint="ok",
        headers={"X-Test": "1"},
        status_code=200,
        latency_ms=1,
        rows_returned=1,
        response_bytes=10,
    )

    conn = sqlite3.connect(ledger_path)
    try:
        conn.execute("DELETE FROM batch_attempt_events")
        conn.commit()
    finally:
        conn.close()

    repaired_ledger = RequestLedger(ledger_path)
    mismatches = repaired_ledger.attempt_mismatches()
    assert mismatches == []

    conn = sqlite3.connect(ledger_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = [
            dict(row)
            for row in conn.execute(
                "SELECT attempt_no, phase, status_code, rows_returned FROM batch_attempt_events ORDER BY event_id"
            ).fetchall()
        ]
    finally:
        conn.close()

    assert rows == [
        {
            "attempt_no": 1,
            "phase": "legacy_finish_backfill",
            "status_code": 200,
            "rows_returned": 1,
        }
    ]


def test_run_api_batches_splits_timeout_prone_batch(tmp_path: Path) -> None:
    items = [
        RequestItem(
            item_id=f"item-{idx}",
            stage="test_stage",
            instrument=instrument,
            batch_key="shared",
            fields=("TR.Field",),
            parameters={},
            payload={"instrument": instrument},
        )
        for idx, instrument in enumerate(("AAA.N", "BBB.N"), start=1)
    ]
    provider = TimeoutThenSuccessProvider()
    result = run_api_batches(
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
        max_batch_size=2,
        min_seconds_between_requests=0.0,
        max_attempts=3,
        response_normalizer=lambda batch_items_rows, frame: pl.DataFrame(
            {
                "item_id": [item.item_id for item in batch_items_rows],
                "instrument": [item.instrument for item in batch_items_rows],
                "value": [
                    next(
                        (
                            row["Value"]
                            for row in frame.to_dicts()
                            if row.get("Instrument") == item.instrument
                        ),
                        None,
                    )
                    for item in batch_items_rows
                ],
            }
        ),
        lookup_normalizer=lambda value: None if value is None else str(value),
        split_after_attempt=1,
        retry_delay_seconds_fn=lambda attempt_no: 0.0,
    )

    ledger = RequestLedger(result.ledger_path)
    assert ledger.state_counts(table="batches") == {"fatal_error": 1, "succeeded": 2}
    assert len(list(result.staging_dir.glob("*.parquet"))) == 2
    assert len(provider.calls) == 3


def test_run_api_batches_rejects_mismatched_batch_plan_fingerprint(tmp_path: Path) -> None:
    items = [
        RequestItem(
            item_id="item-1",
            stage="test_stage",
            instrument="AAA.N",
            batch_key="2020-01-01|2020-01-31",
            fields=("TR.Field",),
            parameters={"SDate": "2020-01-01", "EDate": "2020-01-31"},
            payload={"row": 1},
        )
    ]
    planned_batch = build_batch_definition(
        items,
        batch_key="2020-01-01|2020-01-31",
        parameters={"SDate": "2020-01-01", "EDate": "2020-01-31"},
    )
    provider = EmptySuccessProvider()

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
        min_seconds_between_requests=0.0,
        max_attempts=2,
        response_normalizer=lambda _items, _frame: pl.DataFrame({"item_id": [], "value": []}),
        lookup_normalizer=lambda value: None if value is None else str(value),
        planned_batches=[planned_batch],
        batch_plan_fingerprint="plan-alpha",
    )

    with pytest.raises(LsegResumeCompatibilityError) as exc_info:
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
            min_seconds_between_requests=0.0,
            max_attempts=2,
            response_normalizer=lambda _items, _frame: pl.DataFrame({"item_id": [], "value": []}),
            lookup_normalizer=lambda value: None if value is None else str(value),
            planned_batches=[planned_batch],
            batch_plan_fingerprint="plan-beta",
        )
    exc = exc_info.value
    assert exc.stage == "test_stage"
    assert exc.meta_key == "stage:test_stage:batch_plan_fingerprint"
    assert exc.existing_value == "plan-alpha"
    assert exc.current_value == "plan-beta"
    assert exc.current_stage_meta["batch_plan_fingerprint"] == "plan-beta"


def test_run_api_batches_accepts_matching_batch_plan_fingerprint_on_resume(tmp_path: Path) -> None:
    items = [
        RequestItem(
            item_id="item-1",
            stage="test_stage",
            instrument="AAA.N",
            batch_key="2020-01-01|2020-01-31",
            fields=("TR.Field",),
            parameters={"SDate": "2020-01-01", "EDate": "2020-01-31"},
            payload={"row": 1},
        )
    ]
    planned_batch = build_batch_definition(
        items,
        batch_key="2020-01-01|2020-01-31",
        parameters={"SDate": "2020-01-01", "EDate": "2020-01-31"},
    )
    provider = EmptySuccessProvider()

    first = run_api_batches(
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
        min_seconds_between_requests=0.0,
        max_attempts=2,
        response_normalizer=lambda _items, _frame: pl.DataFrame({"item_id": [], "value": []}),
        lookup_normalizer=lambda value: None if value is None else str(value),
        planned_batches=[planned_batch],
        batch_plan_fingerprint="plan-alpha",
    )
    initial_call_count = len(provider.calls)

    second = run_api_batches(
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
        min_seconds_between_requests=0.0,
        max_attempts=2,
        response_normalizer=lambda _items, _frame: pl.DataFrame({"item_id": [], "value": []}),
        lookup_normalizer=lambda value: None if value is None else str(value),
        planned_batches=[planned_batch],
        batch_plan_fingerprint="plan-alpha",
    )

    assert first.ledger_path == second.ledger_path
    assert len(provider.calls) == initial_call_count


def test_assemble_analyst_actuals_raw_ignores_orphan_staging_when_ledger_is_provided(tmp_path: Path) -> None:
    item = RequestItem(
        item_id="item-1",
        stage="analyst_actuals",
        instrument="AAA.N",
        batch_key="2020-01-01|2020-01-31",
        fields=("TR.Field",),
        parameters={"SDate": "2020-01-01", "EDate": "2020-01-31"},
        payload={"request_row": {"request_group_id": "group-1"}},
    )
    batch = build_batch_definition(
        [item],
        batch_key=item.batch_key,
        parameters=item.parameters,
    )
    ledger_path = tmp_path / "ledger.sqlite3"
    staging_dir = tmp_path / "staging" / "analyst_actuals"
    staging_dir.mkdir(parents=True, exist_ok=True)
    ledger = RequestLedger(ledger_path)
    ledger.enqueue([item], [batch])
    claimed = ledger.claim_next_batch(stage="analyst_actuals")
    assert claimed is not None
    succeeded_path = staging_dir / f"{claimed.batch_id}.parquet"
    pl.DataFrame(
        {
            "item_id": ["item-1"],
            "response_row_index": [0],
            "request_group_id": ["group-1"],
            "gvkey_int": [1000],
            "effective_collection_ric": ["AAA.N"],
            "announcement_date": [date(2020, 1, 15)],
            "fiscal_period_end": [date(2019, 12, 31)],
            "actual_eps": [1.0],
            "raw_fperiod": ["FY2019Q4"],
            "row_parse_status": ["OK"],
        }
    ).write_parquet(succeeded_path)
    pl.DataFrame(
        {
            "item_id": ["orphan-item"],
            "response_row_index": [0],
            "request_group_id": ["orphan-group"],
            "gvkey_int": [9999],
            "effective_collection_ric": ["ZZZ.N"],
            "announcement_date": [date(2020, 1, 20)],
            "fiscal_period_end": [date(2019, 12, 31)],
            "actual_eps": [9.9],
            "raw_fperiod": ["FY2019Q4"],
            "row_parse_status": ["OK"],
        }
    ).write_parquet(staging_dir / "orphan.parquet")
    ledger.record_success(
        batch_id=claimed.batch_id,
        row_count_by_item_id={"item-1": 1},
        stage_output_path=succeeded_path,
        response_fingerprint="ok",
        headers={},
        status_code=200,
        latency_ms=1,
        rows_returned=1,
        response_bytes=10,
    )

    rebuilt = _assemble_analyst_actuals_raw(staging_dir, ledger_path=ledger_path)

    assert rebuilt.height == 1
    assert rebuilt.item(0, "request_group_id") == "group-1"


def test_ownership_universe_stage_fails_on_mismatched_batch_plan_resume(tmp_path: Path) -> None:
    handoff_path = tmp_path / "handoff.parquet"
    pl.DataFrame(
        [
            _ownership_handoff_row(
                lookup_row_id="row-1|UNIVERSE_EFFECTIVE",
                candidate_ric="AAA.N",
                request_start_date="2024-01-01",
                request_end_date="2024-01-31",
            ),
            _ownership_handoff_row(
                lookup_row_id="row-2|UNIVERSE_EFFECTIVE",
                candidate_ric="BBB.N",
                request_start_date="2024-01-15",
                request_end_date="2024-01-31",
            ),
        ]
    ).write_parquet(handoff_path)
    provider = EmptySuccessProvider()
    ledger_path = tmp_path / "ownership.sqlite3"

    run_refinitiv_step1_ownership_universe_api_pipeline(
        handoff_parquet_path=handoff_path,
        output_dir=tmp_path,
        provider=provider,
        ledger_path=ledger_path,
        min_seconds_between_requests=0.0,
    )

    with pytest.raises(LsegResumeCompatibilityError, match="batch_plan_fingerprint"):
        run_refinitiv_step1_ownership_universe_api_pipeline(
            handoff_parquet_path=handoff_path,
            output_dir=tmp_path,
            provider=provider,
            ledger_path=ledger_path,
            max_batch_items=1,
            min_seconds_between_requests=0.0,
        )


def test_ownership_universe_stage_resumes_cleanly_with_matching_batch_plan(tmp_path: Path) -> None:
    handoff_path = tmp_path / "handoff.parquet"
    pl.DataFrame(
        [
            _ownership_handoff_row(
                lookup_row_id="row-1|UNIVERSE_EFFECTIVE",
                candidate_ric="AAA.N",
                request_start_date="2024-01-01",
                request_end_date="2024-01-31",
            )
        ]
    ).write_parquet(handoff_path)
    provider = EmptySuccessProvider()
    ledger_path = tmp_path / "ownership.sqlite3"

    run_refinitiv_step1_ownership_universe_api_pipeline(
        handoff_parquet_path=handoff_path,
        output_dir=tmp_path,
        provider=provider,
        ledger_path=ledger_path,
        min_seconds_between_requests=0.0,
    )
    initial_call_count = len(provider.calls)

    run_refinitiv_step1_ownership_universe_api_pipeline(
        handoff_parquet_path=handoff_path,
        output_dir=tmp_path,
        provider=provider,
        ledger_path=ledger_path,
        min_seconds_between_requests=0.0,
    )

    assert len(provider.calls) == initial_call_count


def test_assemble_ownership_universe_results_ignores_orphan_staging_when_ledger_is_provided(tmp_path: Path) -> None:
    item = RequestItem(
        item_id="item-1",
        stage="ownership_universe",
        instrument="AAA.N",
        batch_key="2024-01-01|2024-01-31",
        fields=(
            "TR.CategoryOwnershipPct.Date",
            "TR.CategoryOwnershipPct",
            "TR.InstrStatTypeValue",
        ),
        parameters={"StatType": 7, "SDate": "2024-01-01", "EDate": "2024-01-31"},
        payload={"handoff_row": {"ownership_lookup_row_id": "row-1|UNIVERSE_EFFECTIVE"}},
    )
    batch = build_batch_definition(
        [item],
        batch_key=item.batch_key,
        parameters=item.parameters,
    )
    ledger_path = tmp_path / "ledger.sqlite3"
    staging_dir = tmp_path / "staging" / "ownership_universe"
    staging_dir.mkdir(parents=True, exist_ok=True)
    ledger = RequestLedger(ledger_path)
    ledger.enqueue([item], [batch])
    claimed = ledger.claim_next_batch(stage="ownership_universe")
    assert claimed is not None
    succeeded_path = staging_dir / f"{claimed.batch_id}.parquet"
    succeeded_row = _ownership_handoff_row(
        lookup_row_id="row-1|UNIVERSE_EFFECTIVE",
        candidate_ric="AAA.N",
        request_start_date="2024-01-01",
        request_end_date="2024-01-31",
    )
    orphan_row = _ownership_handoff_row(
        lookup_row_id="orphan|UNIVERSE_EFFECTIVE",
        candidate_ric="ZZZ.N",
        request_start_date="2024-01-01",
        request_end_date="2024-01-31",
    )
    pl.DataFrame(
        {
            "item_id": ["item-1"],
            **{name: [succeeded_row.get(name)] for name in OWNERSHIP_UNIVERSE_HANDOFF_COLUMNS},
            "returned_ric": ["AAA.N"],
            "returned_date": [date(2024, 1, 15)],
            "returned_category": ["Holdings by Institutions"],
            "returned_value": [55.0],
        }
    ).write_parquet(succeeded_path)
    pl.DataFrame(
        {
            "item_id": ["orphan-item"],
            **{name: [orphan_row.get(name)] for name in OWNERSHIP_UNIVERSE_HANDOFF_COLUMNS},
            "returned_ric": ["ZZZ.N"],
            "returned_date": [date(2024, 1, 20)],
            "returned_category": ["Holdings by Institutions"],
            "returned_value": [99.0],
        }
    ).write_parquet(staging_dir / "orphan.parquet")
    ledger.record_success(
        batch_id=claimed.batch_id,
        row_count_by_item_id={"item-1": 1},
        stage_output_path=succeeded_path,
        response_fingerprint="ok",
        headers={},
        status_code=200,
        latency_ms=1,
        rows_returned=1,
        response_bytes=10,
    )

    rebuilt = _assemble_ownership_universe_results(staging_dir, ledger_path=ledger_path)

    assert rebuilt.height == 1
    assert rebuilt.item(0, "ownership_lookup_row_id") == "row-1|UNIVERSE_EFFECTIVE"


def test_assemble_doc_raw_ignores_orphan_staging_when_ledger_is_provided(tmp_path: Path) -> None:
    item = RequestItem(
        item_id="item-1",
        stage=DOC_EXACT_STAGE,
        instrument="AAA.N",
        batch_key="2024-04-01",
        fields=(
            "TR.CategoryOwnershipPct.Date",
            "TR.CategoryOwnershipPct",
            "TR.InstrStatTypeValue",
        ),
        parameters={"StatType": 7, "SDate": "2024-04-01", "EDate": "2024-04-01"},
        payload={"request_row": {"doc_id": "doc-1"}},
    )
    batch = build_batch_definition(
        [item],
        batch_key=item.batch_key,
        parameters=item.parameters,
    )
    ledger_path = tmp_path / "ledger.sqlite3"
    staging_dir = tmp_path / "staging" / DOC_EXACT_STAGE
    staging_dir.mkdir(parents=True, exist_ok=True)
    ledger = RequestLedger(ledger_path)
    ledger.enqueue([item], [batch])
    claimed = ledger.claim_next_batch(stage=DOC_EXACT_STAGE)
    assert claimed is not None
    succeeded_path = staging_dir / f"{claimed.batch_id}.parquet"
    succeeded_row = {
        "item_id": "item-1",
        "doc_id": "doc-1",
        "filing_date": date(2024, 4, 15),
        "KYPERMNO": "100",
        "authoritative_ric": "AAA.N",
        "authority_decision_status": "STATIC_CONVENTIONAL",
        "target_quarter_end": date(2024, 3, 31),
        "target_effective_date": date(2024, 4, 1),
        "request_stage": "EXACT",
        "response_date": date(2024, 4, 1),
        "response_date_is_imputed": False,
        "returned_category": "Holdings by Institutions",
        "returned_category_normalized": "Holdings by Institutions",
        "returned_value": 55.0,
        "is_institutional_category": True,
    }
    orphan_row = {
        "item_id": "orphan-item",
        "doc_id": "orphan-doc",
        "filing_date": date(2024, 4, 15),
        "KYPERMNO": "999",
        "authoritative_ric": "ZZZ.N",
        "authority_decision_status": "STATIC_CONVENTIONAL",
        "target_quarter_end": date(2024, 3, 31),
        "target_effective_date": date(2024, 4, 1),
        "request_stage": "EXACT",
        "response_date": date(2024, 4, 1),
        "response_date_is_imputed": False,
        "returned_category": "Holdings by Institutions",
        "returned_category_normalized": "Holdings by Institutions",
        "returned_value": 99.0,
        "is_institutional_category": True,
    }
    pl.DataFrame([succeeded_row]).select(["item_id", *DOC_OWNERSHIP_RAW_COLUMNS]).write_parquet(succeeded_path)
    pl.DataFrame([orphan_row]).select(["item_id", *DOC_OWNERSHIP_RAW_COLUMNS]).write_parquet(
        staging_dir / "orphan.parquet"
    )
    ledger.record_success(
        batch_id=claimed.batch_id,
        row_count_by_item_id={"item-1": 1},
        stage_output_path=succeeded_path,
        response_fingerprint="ok",
        headers={},
        status_code=200,
        latency_ms=1,
        rows_returned=1,
        response_bytes=10,
    )

    rebuilt = _assemble_doc_raw(
        staging_dir,
        ledger_path=ledger_path,
        stage=DOC_EXACT_STAGE,
    )

    assert rebuilt.height == 1
    assert rebuilt.item(0, "doc_id") == "doc-1"


def test_audit_api_stage_flags_orphan_staging_files(tmp_path: Path) -> None:
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
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    output_path = tmp_path / "final.parquet"
    ledger = RequestLedger(ledger_path)
    ledger.enqueue([item], [batch])
    claimed = ledger.claim_next_batch(stage="test_stage")
    assert claimed is not None
    stage_df = pl.DataFrame({"item_id": ["item-1"], "value": [1.0]})
    stage_path = staging_dir / f"{claimed.batch_id}.parquet"
    stage_df.write_parquet(stage_path)
    ledger.record_success(
        batch_id=claimed.batch_id,
        row_count_by_item_id={"item-1": 1},
        stage_output_path=stage_path,
        response_fingerprint="ok",
        headers={},
        status_code=200,
        latency_ms=1,
        rows_returned=1,
        response_bytes=10,
    )
    stage_df.write_parquet(output_path)
    pl.DataFrame({"item_id": ["orphan"], "value": [2.0]}).write_parquet(staging_dir / "orphan.parquet")

    audit_result = audit_api_stage(
        stage_name="test_stage",
        ledger_path=ledger_path,
        staging_dir=staging_dir,
        output_artifacts={"final_parquet": output_path},
        rebuilders={"final_parquet": lambda: stage_df},
    )

    assert any(issue.code == "orphan_staging_files" for issue in audit_result.issues)
    assert audit_result.passed is True


def test_audit_api_stage_fails_when_declared_output_artifact_is_missing(tmp_path: Path) -> None:
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
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    output_path = tmp_path / "final.parquet"
    missing_output_path = tmp_path / "missing.parquet"
    ledger = RequestLedger(ledger_path)
    ledger.enqueue([item], [batch])
    claimed = ledger.claim_next_batch(stage="test_stage")
    assert claimed is not None
    stage_df = pl.DataFrame({"item_id": ["item-1"], "value": [1.0]})
    stage_path = staging_dir / f"{claimed.batch_id}.parquet"
    stage_df.write_parquet(stage_path)
    ledger.record_success(
        batch_id=claimed.batch_id,
        row_count_by_item_id={"item-1": 1},
        stage_output_path=stage_path,
        response_fingerprint="ok",
        headers={},
        status_code=200,
        latency_ms=1,
        rows_returned=1,
        response_bytes=10,
    )
    stage_df.write_parquet(output_path)

    audit_result = audit_api_stage(
        stage_name="test_stage",
        ledger_path=ledger_path,
        staging_dir=staging_dir,
        output_artifacts={
            "final_parquet": output_path,
            "missing_parquet": missing_output_path,
        },
        rebuilders={"final_parquet": lambda: stage_df},
    )

    assert audit_result.passed is False
    assert any(
        issue.code == "missing_output_artifact"
        and issue.details is not None
        and issue.details.get("label") == "missing_parquet"
        for issue in audit_result.issues
    )


def test_lookup_stage_keeps_last_good_canonical_output_when_audit_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_path = tmp_path / "lookup_snapshot.parquet"
    output_path = tmp_path / "refinitiv_ric_lookup_handoff_common_stock_extended.parquet"
    sentinel_df = pl.DataFrame({"sentinel": ["keep"]})
    sentinel_df.write_parquet(output_path)
    lookup_df = pl.DataFrame(
        {
            "bridge_row_id": [
                "1:A:111111111:US1111111111:ALFA-1",
                "1:A:111111111:US1111111111:ALFA-2",
            ],
            "KYPERMNO": ["1", "1"],
            "CUSIP": ["111111111", "111111111"],
            "ISIN": ["US1111111111", "US1111111111"],
            "TICKER": ["ALFA", "ALFA"],
            "first_seen_caldt": [date(2024, 1, 1), date(2024, 1, 2)],
            "last_seen_caldt": [date(2024, 1, 10), date(2024, 1, 11)],
            "preferred_lookup_id": ["US1111111111", "US1111111111"],
            "preferred_lookup_type": ["ISIN", "ISIN"],
            "vendor_primary_ric": [None, None],
            "vendor_returned_name": [None, None],
            "vendor_returned_cusip": [None, None],
            "vendor_returned_isin": [None, None],
            "vendor_match_status": [None, None],
            "vendor_notes": [None, None],
        }
    ).select(RIC_LOOKUP_COLUMNS)
    snapshot_df, _, _ = build_refinitiv_lookup_extended_diagnostic_artifact(lookup_df)
    snapshot_df.write_parquet(snapshot_path)
    provider = TimeoutThenSuccessProvider()

    def fake_audit_api_stage(**_: Any) -> StageAuditResult:
        return StageAuditResult(
            stage_name="lookup",
            passed=False,
            issues=(AuditIssue("high", "forced_failure", "forced audit failure"),),
            metrics={},
        )

    monkeypatch.setattr(lseg_lookup_api, "audit_api_stage", fake_audit_api_stage)

    with pytest.raises(RuntimeError, match="lookup stage audit failed"):
        run_refinitiv_step1_lookup_api_pipeline(
            snapshot_parquet_path=snapshot_path,
            output_dir=tmp_path,
            provider=provider,
            min_seconds_between_requests=0.0,
        )

    current_df = pl.read_parquet(output_path)
    assert current_df.to_dicts() == sentinel_df.to_dicts()
    candidate_path = output_path.with_suffix(".parquet.candidate")
    assert candidate_path.exists()


def test_recovery_builders_select_only_unresolved_rows(tmp_path: Path) -> None:
    resolution_path = tmp_path / "resolution.parquet"
    row_summary_path = tmp_path / "row_summary.parquet"
    exact_requests_path = tmp_path / "exact_requests.parquet"
    exact_raw_path = tmp_path / "exact_raw.parquet"

    pl.DataFrame(
        {
            "bridge_row_id": ["row-1", "row-2"],
            "effective_collection_ric": [None, "AAA.N"],
        }
    ).write_parquet(resolution_path)
    pl.DataFrame(
        {
            "ownership_lookup_row_id": ["own-1", "own-2"],
            "retrieval_eligible": [True, True],
            "ownership_rows_returned": [0, 3],
        }
    ).write_parquet(row_summary_path)
    pl.DataFrame(
        {
            "doc_id": ["doc-1", "doc-2"],
            "retrieval_eligible": [True, True],
        }
    ).write_parquet(exact_requests_path)
    pl.DataFrame(
        {
            "doc_id": ["doc-2"],
            "request_stage": ["EXACT"],
        }
    ).write_parquet(exact_raw_path)

    lookup_artifact = build_lookup_unresolved_recovery_artifact(
        resolution_parquet_path=resolution_path,
        output_path=tmp_path / "lookup_unresolved.parquet",
    )
    ownership_artifact = build_ownership_unresolved_recovery_artifact(
        row_summary_parquet_path=row_summary_path,
        output_path=tmp_path / "ownership_unresolved.parquet",
    )
    doc_artifact = build_doc_unresolved_recovery_artifact(
        recovery_mode="doc_exact_unresolved",
        requests_parquet_path=exact_requests_path,
        raw_parquet_path=exact_raw_path,
        output_path=tmp_path / "doc_exact_unresolved.parquet",
    )

    assert lookup_artifact.row_count == 1
    assert ownership_artifact.row_count == 1
    assert doc_artifact.row_count == 1
    assert pl.read_parquet(lookup_artifact.output_path).item(0, "bridge_row_id") == "row-1"
    assert pl.read_parquet(ownership_artifact.output_path).item(0, "ownership_lookup_row_id") == "own-1"
    assert pl.read_parquet(doc_artifact.output_path).item(0, "doc_id") == "doc-1"

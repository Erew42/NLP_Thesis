from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import polars as pl

from thesis_pkg.pipelines.refinitiv.lseg_api_execution import run_api_batches
from thesis_pkg.pipelines.refinitiv.lseg_batching import RequestItem, batch_items
from thesis_pkg.pipelines.refinitiv.lseg_ledger import RequestLedger
from thesis_pkg.pipelines.refinitiv.lseg_provider import (
    LsegDataResponse,
    LsegRequestError,
    LsegResponseMetadata,
)
from thesis_pkg.pipelines.refinitiv.lseg_recovery import (
    build_doc_unresolved_recovery_artifact,
    build_lookup_unresolved_recovery_artifact,
    build_ownership_unresolved_recovery_artifact,
)
from thesis_pkg.pipelines.refinitiv.lseg_stage_audit import audit_api_stage


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
        assert phases == ["claimed", "request_started"]
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

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Any, Callable, Literal

import polars as pl

from thesis_pkg.pipelines.refinitiv.lseg_ledger import (
    ATTEMPT_PHASE_LEGACY_FINISH_BACKFILL,
    LEDGER_TERMINAL_STATES,
    LEDGER_SUCCEEDED,
    RequestLedger,
)


@dataclass(frozen=True)
class AuditIssue:
    severity: str
    code: str
    message: str
    details: dict[str, Any] | None = None


@dataclass(frozen=True)
class StageAuditResult:
    stage_name: str
    passed: bool
    issues: tuple[AuditIssue, ...]
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "passed": self.passed,
            "issues": [
                {
                    "severity": issue.severity,
                    "code": issue.code,
                    "message": issue.message,
                    "details": issue.details or {},
                }
                for issue in self.issues
            ],
            "metrics": self.metrics,
        }


StageStoredMetadataSource = Literal["fetch_manifest", "stage_manifest", "ledger_meta", "unknown"]


@dataclass(frozen=True)
class ResolvedStageFetchMetadata:
    batch_plan_fingerprint: str | None
    batching_config: dict[str, Any] | None
    request_item_count: int | None
    batch_count: int | None
    run_session_ids: list[str]
    metadata_source: StageStoredMetadataSource
    cli_batching_args_ignored: bool


def default_stage_manifest_path(output_dir: Path | str, stage_name: str) -> Path:
    return Path(output_dir) / f"refinitiv_{stage_name}_stage_manifest.json"


def default_stage_fetch_manifest_path(output_dir: Path | str, stage_name: str) -> Path:
    return Path(output_dir) / f"refinitiv_{stage_name}_fetch_manifest.json"


def audit_api_stage(
    *,
    stage_name: str,
    ledger_path: Path | str,
    staging_dir: Path | str,
    output_artifacts: dict[str, Path],
    declared_output_artifacts: dict[str, Path] | None = None,
    rebuilders: dict[str, Callable[[], pl.DataFrame]] | None = None,
    expected_stage_manifest_path: Path | str | None = None,
    verify_rebuilders: bool = True,
) -> StageAuditResult:
    ledger_path = Path(ledger_path)
    staging_dir = Path(staging_dir)
    declared_output_artifacts = declared_output_artifacts or output_artifacts
    rebuilders = rebuilders or {}
    issues: list[AuditIssue] = []
    metrics: dict[str, Any] = {
        "ledger_path": str(ledger_path),
        "staging_dir": str(staging_dir),
        "output_artifacts": {key: str(value) for key, value in output_artifacts.items()},
        "declared_output_artifacts": {key: str(value) for key, value in declared_output_artifacts.items()},
        "stage_manifest_path": None if expected_stage_manifest_path is None else str(expected_stage_manifest_path),
        "verify_rebuilders": verify_rebuilders,
    }

    if not ledger_path.exists():
        issues.append(AuditIssue("high", "missing_ledger", f"ledger not found: {ledger_path}"))
        return StageAuditResult(stage_name=stage_name, passed=False, issues=tuple(issues), metrics=metrics)

    ledger = RequestLedger(ledger_path)
    batch_state_counts = ledger.state_counts(table="batches")
    item_state_counts = ledger.state_counts(table="request_items")
    attempt_mismatches = ledger.attempt_mismatches()
    metrics["batch_state_counts"] = batch_state_counts
    metrics["item_state_counts"] = item_state_counts
    metrics["attempt_mismatches"] = attempt_mismatches
    metrics["run_session_ids"] = ledger.run_session_ids()

    nonterminal_batch_states = {
        state: count
        for state, count in batch_state_counts.items()
        if state not in LEDGER_TERMINAL_STATES
    }
    if nonterminal_batch_states:
        issues.append(
            AuditIssue(
                "high",
                "nonterminal_batches",
                "batch ledger still contains non-terminal states",
                {"state_counts": nonterminal_batch_states},
            )
        )

    nonterminal_item_states = {
        state: count
        for state, count in item_state_counts.items()
        if state not in LEDGER_TERMINAL_STATES
    }
    if nonterminal_item_states:
        issues.append(
            AuditIssue(
                "high",
                "nonterminal_items",
                "item ledger still contains non-terminal states",
                {"state_counts": nonterminal_item_states},
            )
        )

    unexplained_attempt_mismatches: list[dict[str, Any]] = []
    tolerated_attempt_event_gaps: list[dict[str, Any]] = []
    for row in attempt_mismatches:
        if row["attempt_count"] <= row["event_attempt_count"]:
            continue
        tolerated, details = _is_tolerable_legacy_finish_backfill_attempt_gap(
            ledger_path=ledger_path,
            batch_id=str(row["batch_id"]),
            attempt_count=int(row["attempt_count"]),
        )
        if tolerated:
            tolerated_attempt_event_gaps.append({**row, **details})
        else:
            unexplained_attempt_mismatches.append(row)
    metrics["tolerated_attempt_event_gaps"] = tolerated_attempt_event_gaps
    if unexplained_attempt_mismatches:
        issues.append(
            AuditIssue(
                "high",
                "attempt_event_gap",
                "batch attempt count exceeds durable attempt event history",
                {"rows": unexplained_attempt_mismatches},
            )
        )
    explainable_attempt_mismatches = [
        row
        for row in attempt_mismatches
        if row["attempt_count"] == row["event_attempt_count"] and row["attempt_count"] != row["finished_attempt_count"]
    ]
    if explainable_attempt_mismatches:
        issues.append(
            AuditIssue(
                "medium",
                "finished_attempt_gap",
                "attempt summary rows do not cover all attempt events",
                {"rows": explainable_attempt_mismatches},
            )
        )

    succeeded_batch_rows = _query_rows(
        ledger_path,
        """
        SELECT batch_id, result_file_path
        FROM batches
        WHERE state = 'succeeded'
        ORDER BY batch_id
        """,
    )
    missing_stage_files = [
        row for row in succeeded_batch_rows if not row["result_file_path"] or not Path(row["result_file_path"]).exists()
    ]
    if missing_stage_files:
        issues.append(
            AuditIssue(
                "high",
                "missing_succeeded_staging_files",
                "one or more succeeded batches are missing staging parquet outputs",
                {"rows": missing_stage_files},
            )
        )

    staging_paths = sorted(staging_dir.glob("*.parquet")) if staging_dir.exists() else []
    referenced_paths = {
        str(Path(row["result_file_path"]).resolve())
        for row in succeeded_batch_rows
        if row["result_file_path"] is not None
    }
    orphan_staging_paths = [
        str(path)
        for path in staging_paths
        if str(path.resolve()) not in referenced_paths
    ]
    if orphan_staging_paths:
        issues.append(
            AuditIssue(
                "medium",
                "orphan_staging_files",
                "staging parquet files exist without matching succeeded ledger rows",
                {"paths": orphan_staging_paths},
            )
        )
    metrics["staging_file_count"] = len(staging_paths)
    metrics["succeeded_batch_count"] = len(succeeded_batch_rows)

    missing_output_labels: set[str] = set()
    output_row_counts: dict[str, int] = {}
    for label, output_path in output_artifacts.items():
        if output_path.exists():
            if _is_parquetish_path(output_path):
                output_row_counts[label] = int(pl.scan_parquet(output_path).select(pl.len()).collect().item())
            continue
        missing_output_labels.add(label)
        issues.append(
            AuditIssue(
                "high",
                "missing_output_artifact",
                f"output artifact not found: {output_path}",
                {"label": label},
            )
        )
    metrics["output_row_counts"] = output_row_counts

    rebuild_row_counts: dict[str, int] = {}
    if verify_rebuilders:
        for label, rebuilder in rebuilders.items():
            output_path = output_artifacts.get(label)
            if output_path is None:
                issues.append(
                    AuditIssue(
                        "high",
                        "missing_output_artifact_mapping",
                        f"no canonical output artifact registered for rebuild label {label!r}",
                    )
                )
                continue
            if label in missing_output_labels:
                continue
            rebuilt_df = rebuilder()
            actual_df = pl.read_parquet(output_path)
            rebuild_row_counts[label] = int(rebuilt_df.height)
            if not _frames_equal(rebuilt_df, actual_df):
                issues.append(
                    AuditIssue(
                        "high",
                        "rebuild_mismatch",
                        f"rebuilt output did not match canonical artifact: {label}",
                        {"output_path": str(output_path)},
                    )
                )
    metrics["rebuild_row_counts"] = rebuild_row_counts

    if expected_stage_manifest_path is not None:
        expected_stage_manifest_path = Path(expected_stage_manifest_path)
        if expected_stage_manifest_path.exists():
            manifest_payload = json.loads(expected_stage_manifest_path.read_text(encoding="utf-8"))
            manifest_outputs = manifest_payload.get("output_artifacts", {})
            actual_outputs = {key: str(value) for key, value in declared_output_artifacts.items()}
            if manifest_outputs != actual_outputs:
                issues.append(
                    AuditIssue(
                        "medium",
                        "manifest_output_mismatch",
                        "stage manifest output artifact map does not match current paths",
                        {"manifest_outputs": manifest_outputs, "actual_outputs": actual_outputs},
                    )
                )

    passed = not any(issue.severity == "high" for issue in issues)
    return StageAuditResult(stage_name=stage_name, passed=passed, issues=tuple(issues), metrics=metrics)


def write_stage_fetch_manifest(
    *,
    stage_name: str,
    manifest_path: Path | str,
    staging_dir: Path | str,
    ledger_path: Path | str,
    request_log_path: Path | str,
    batching_config: dict[str, Any] | None,
    request_item_count: int | None,
    batch_count: int | None,
    run_session_ids: list[str],
    summary: dict[str, Any] | None = None,
    metadata_source: str | None = None,
    cli_batching_args_ignored: bool = False,
) -> Path:
    manifest_path = Path(manifest_path)
    payload = {
        "manifest_role": "stage_fetch",
        "stage_name": stage_name,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "staging_dir": str(staging_dir),
        "ledger_path": str(ledger_path),
        "request_log_path": str(request_log_path),
        "run_session_ids": run_session_ids,
        "batching_config": batching_config,
        "request_item_count": request_item_count,
        "batch_count": batch_count,
        "metadata_source": metadata_source,
        "cli_batching_args_ignored": cli_batching_args_ignored,
        "summary": summary or {},
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def write_stage_completion_manifest(
    *,
    stage_name: str,
    manifest_path: Path | str,
    input_artifacts: dict[str, Path],
    output_artifacts: dict[str, Path],
    ledger_path: Path | str,
    request_log_path: Path | str,
    staging_dir: Path | str,
    audit_result: StageAuditResult,
    summary: dict[str, Any] | None = None,
    metadata_source: str | None = None,
    cli_batching_args_ignored: bool = False,
) -> Path:
    manifest_path = Path(manifest_path)
    payload = {
        "manifest_role": "stage_completion",
        "stage_name": stage_name,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input_artifacts": {key: str(value) for key, value in input_artifacts.items()},
        "output_artifacts": {key: str(value) for key, value in output_artifacts.items()},
        "ledger_path": str(ledger_path),
        "request_log_path": str(request_log_path),
        "staging_dir": str(staging_dir),
        "metadata_source": metadata_source,
        "cli_batching_args_ignored": cli_batching_args_ignored,
        "summary": summary or {},
        "audit": audit_result.to_dict(),
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def resolve_stage_fetch_metadata(
    *,
    stage_name: str,
    ledger_path: Path | str,
    fetch_manifest_path: Path | str | None = None,
    stage_manifest_path: Path | str | None = None,
    current_batching_config: dict[str, Any] | None = None,
    current_batch_plan_fingerprint: str | None = None,
) -> ResolvedStageFetchMetadata:
    ledger = RequestLedger(ledger_path)
    fetch_payload = _read_json_payload(None if fetch_manifest_path is None else Path(fetch_manifest_path))
    if fetch_payload is not None and fetch_payload.get("stage_name") == stage_name:
        resolved = ResolvedStageFetchMetadata(
            batch_plan_fingerprint=_normalize_optional_text((fetch_payload.get("summary") or {}).get("batch_plan_fingerprint")),
            batching_config=_normalize_optional_mapping(fetch_payload.get("batching_config")),
            request_item_count=_normalize_optional_int(fetch_payload.get("request_item_count")),
            batch_count=_normalize_optional_int(fetch_payload.get("batch_count")),
            run_session_ids=_normalize_string_list(fetch_payload.get("run_session_ids")) or ledger.run_session_ids(),
            metadata_source="fetch_manifest",
            cli_batching_args_ignored=False,
        )
        return _finalize_resolved_stage_fetch_metadata(
            resolved,
            ledger=ledger,
            stage_name=stage_name,
            current_batching_config=current_batching_config,
            current_batch_plan_fingerprint=current_batch_plan_fingerprint,
        )

    stage_payload = _read_json_payload(None if stage_manifest_path is None else Path(stage_manifest_path))
    if stage_payload is not None and stage_payload.get("stage_name") == stage_name:
        summary = stage_payload.get("summary") or {}
        resolved = ResolvedStageFetchMetadata(
            batch_plan_fingerprint=_normalize_optional_text(summary.get("batch_plan_fingerprint")),
            batching_config=_normalize_optional_mapping(summary.get("batching_config")),
            request_item_count=_normalize_optional_int(summary.get("request_item_count")),
            batch_count=_normalize_optional_int(summary.get("planned_batch_count")),
            run_session_ids=_normalize_string_list(summary.get("run_session_ids")) or ledger.run_session_ids(),
            metadata_source="stage_manifest",
            cli_batching_args_ignored=False,
        )
        return _finalize_resolved_stage_fetch_metadata(
            resolved,
            ledger=ledger,
            stage_name=stage_name,
            current_batching_config=current_batching_config,
            current_batch_plan_fingerprint=current_batch_plan_fingerprint,
        )

    stage_meta = ledger.stage_meta(stage=stage_name)
    if stage_meta:
        batching_config: dict[str, Any] | None = None
        raw_batching_config_json = stage_meta.get("batching_config_json")
        if raw_batching_config_json:
            try:
                loaded_batching_config = json.loads(raw_batching_config_json)
            except json.JSONDecodeError:
                loaded_batching_config = None
            batching_config = _normalize_optional_mapping(loaded_batching_config)
        resolved = ResolvedStageFetchMetadata(
            batch_plan_fingerprint=_normalize_optional_text(stage_meta.get("batch_plan_fingerprint")),
            batching_config=batching_config,
            request_item_count=ledger.stage_item_count(stage=stage_name),
            batch_count=_normalize_optional_int(stage_meta.get("planned_batch_count")) or ledger.stage_batch_count(stage=stage_name),
            run_session_ids=ledger.run_session_ids(),
            metadata_source="ledger_meta",
            cli_batching_args_ignored=False,
        )
        return _finalize_resolved_stage_fetch_metadata(
            resolved,
            ledger=ledger,
            stage_name=stage_name,
            current_batching_config=current_batching_config,
            current_batch_plan_fingerprint=current_batch_plan_fingerprint,
        )

    return _finalize_resolved_stage_fetch_metadata(
        ResolvedStageFetchMetadata(
            batch_plan_fingerprint=None,
            batching_config=None,
            request_item_count=ledger.stage_item_count(stage=stage_name),
            batch_count=ledger.stage_batch_count(stage=stage_name),
            run_session_ids=ledger.run_session_ids(),
            metadata_source="unknown",
            cli_batching_args_ignored=False,
        ),
        ledger=ledger,
        stage_name=stage_name,
        current_batching_config=current_batching_config,
        current_batch_plan_fingerprint=current_batch_plan_fingerprint,
    )


def _is_parquetish_path(path: Path) -> bool:
    return any(suffix.lower() == ".parquet" for suffix in path.suffixes)


def _read_json_payload(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _normalize_optional_mapping(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, dict) else None


def _normalize_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for entry in value:
        text = _normalize_optional_text(entry)
        if text is not None:
            result.append(text)
    return result


def _finalize_resolved_stage_fetch_metadata(
    resolved: ResolvedStageFetchMetadata,
    *,
    ledger: RequestLedger,
    stage_name: str,
    current_batching_config: dict[str, Any] | None,
    current_batch_plan_fingerprint: str | None,
) -> ResolvedStageFetchMetadata:
    request_item_count = resolved.request_item_count
    if request_item_count is None:
        request_item_count = ledger.stage_item_count(stage=stage_name)
    batch_count = resolved.batch_count
    if batch_count is None:
        batch_count = ledger.stage_batch_count(stage=stage_name)
    run_session_ids = resolved.run_session_ids or ledger.run_session_ids()
    cli_batching_args_ignored = False
    if resolved.batching_config is not None and current_batching_config is not None:
        cli_batching_args_ignored = resolved.batching_config != current_batching_config
    if resolved.batch_plan_fingerprint is not None and current_batch_plan_fingerprint is not None:
        cli_batching_args_ignored = cli_batching_args_ignored or (
            resolved.batch_plan_fingerprint != current_batch_plan_fingerprint
        )
    return ResolvedStageFetchMetadata(
        batch_plan_fingerprint=resolved.batch_plan_fingerprint,
        batching_config=resolved.batching_config,
        request_item_count=request_item_count,
        batch_count=batch_count,
        run_session_ids=run_session_ids,
        metadata_source=resolved.metadata_source,
        cli_batching_args_ignored=cli_batching_args_ignored,
    )


def _frames_equal(left: pl.DataFrame, right: pl.DataFrame) -> bool:
    if left.columns != right.columns:
        return False
    if left.schema != right.schema:
        return False
    if left.height != right.height:
        return False
    try:
        return bool(left.equals(right, null_equal=True))
    except TypeError:
        return bool(left.equals(right))


def _is_tolerable_legacy_finish_backfill_attempt_gap(
    *,
    ledger_path: Path,
    batch_id: str,
    attempt_count: int,
) -> tuple[bool, dict[str, Any]]:
    conn = sqlite3.connect(ledger_path)
    conn.row_factory = sqlite3.Row
    try:
        batch_row = conn.execute(
            """
            SELECT state, item_count, rows_returned, result_file_path, item_ids_json
            FROM batches
            WHERE batch_id = ?
            """,
            (batch_id,),
        ).fetchone()
        if batch_row is None or batch_row["state"] != LEDGER_SUCCEEDED:
            return False, {}

        attempt_rows = conn.execute(
            """
            SELECT attempt_no
            FROM batch_attempts
            WHERE batch_id = ?
            ORDER BY attempt_no
            """,
            (batch_id,),
        ).fetchall()
        event_rows = conn.execute(
            """
            SELECT attempt_no, phase
            FROM batch_attempt_events
            WHERE batch_id = ?
            ORDER BY event_id
            """,
            (batch_id,),
        ).fetchall()
        if len(attempt_rows) != 1 or len(event_rows) != 1:
            return False, {}
        if int(attempt_rows[0]["attempt_no"]) != attempt_count:
            return False, {}
        if int(event_rows[0]["attempt_no"]) != attempt_count:
            return False, {}
        if str(event_rows[0]["phase"]) != ATTEMPT_PHASE_LEGACY_FINISH_BACKFILL:
            return False, {}

        stage_path = Path(str(batch_row["result_file_path"] or ""))
        if not stage_path.exists():
            return False, {}
        stage_row_count = int(pl.scan_parquet(stage_path).select(pl.len()).collect().item())
        ledger_rows_returned = batch_row["rows_returned"]
        if ledger_rows_returned is None or stage_row_count != int(ledger_rows_returned):
            return False, {}

        item_ids = json.loads(str(batch_row["item_ids_json"]))
        if not isinstance(item_ids, list) or len(item_ids) != int(batch_row["item_count"]):
            return False, {}
        placeholders = ",".join("?" for _ in item_ids)
        item_state_rows = conn.execute(
            f"""
            SELECT state, COUNT(*) AS row_count
            FROM request_items
            WHERE item_id IN ({placeholders})
            GROUP BY state
            ORDER BY state
            """,
            item_ids,
        ).fetchall()
        item_state_counts = {str(row["state"]): int(row["row_count"]) for row in item_state_rows}
        if item_state_counts != {LEDGER_SUCCEEDED: len(item_ids)}:
            return False, {}

        return True, {
            "tolerance_reason": "legacy_finish_backfill_final_attempt_only",
            "stage_row_count": stage_row_count,
            "event_phase": str(event_rows[0]["phase"]),
        }
    finally:
        conn.close()


def _query_rows(db_path: Path, query: str) -> list[dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(query).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()

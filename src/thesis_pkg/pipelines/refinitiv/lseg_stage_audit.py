from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Any, Callable

import polars as pl

from thesis_pkg.pipelines.refinitiv.lseg_ledger import (
    LEDGER_TERMINAL_STATES,
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


def default_stage_manifest_path(output_dir: Path | str, stage_name: str) -> Path:
    return Path(output_dir) / f"refinitiv_{stage_name}_stage_manifest.json"


def audit_api_stage(
    *,
    stage_name: str,
    ledger_path: Path | str,
    staging_dir: Path | str,
    output_artifacts: dict[str, Path],
    rebuilders: dict[str, Callable[[], pl.DataFrame]] | None = None,
    expected_stage_manifest_path: Path | str | None = None,
) -> StageAuditResult:
    ledger_path = Path(ledger_path)
    staging_dir = Path(staging_dir)
    rebuilders = rebuilders or {}
    issues: list[AuditIssue] = []
    metrics: dict[str, Any] = {
        "ledger_path": str(ledger_path),
        "staging_dir": str(staging_dir),
        "output_artifacts": {key: str(value) for key, value in output_artifacts.items()},
        "stage_manifest_path": None if expected_stage_manifest_path is None else str(expected_stage_manifest_path),
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

    unexplained_attempt_mismatches = [
        row
        for row in attempt_mismatches
        if row["attempt_count"] > row["event_attempt_count"]
    ]
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

    rebuild_row_counts: dict[str, int] = {}
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
        if not output_path.exists():
            issues.append(
                AuditIssue("high", "missing_output_artifact", f"output artifact not found: {output_path}")
            )
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
            actual_outputs = {key: str(value) for key, value in output_artifacts.items()}
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
        "summary": summary or {},
        "audit": audit_result.to_dict(),
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


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


def _query_rows(db_path: Path, query: str) -> list[dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(query).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()

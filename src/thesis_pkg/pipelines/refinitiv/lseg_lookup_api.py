from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from thesis_pkg.pipelines.refinitiv.lseg_api_common import (
    append_json_log as _append_json_log,
    candidate_output_path as _candidate_output_path,
    classify_error as _classify_error,
    daily_limit_likely_exhausted as _daily_limit_likely_exhausted,
    error_details_for_exception as _error_details,
    next_daily_resume_utc as _next_daily_resume_utc,
    promote_candidate_output as _promote_candidate_output,
    retry_delay_seconds as _retry_delay_seconds,
    should_treat_as_empty_result as _should_treat_as_empty_result,
    standardize_field_frame as _standardize_field_frame,
    write_parquet_atomic as _write_parquet_atomic,
)
from thesis_pkg.pipelines.refinitiv.lseg_api_execution import run_api_batches
from thesis_pkg.pipelines.refinitiv.lseg_batching import (
    RequestItem,
    stable_hash_id,
)
from thesis_pkg.pipelines.refinitiv.lseg_stage_audit import (
    audit_api_stage,
    default_stage_manifest_path,
    write_stage_completion_manifest,
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
    min_seconds_between_request_starts_total: float | None = None,
    max_attempts: int = 4,
    max_workers: int = 1,
    provider_session_name: str = "desktop.workspace",
    provider_config_name: str | None = None,
    provider_timeout_seconds: float | None = None,
    preflight_probe: bool = False,
    stage_manifest_path: Path | str | None = None,
) -> dict[str, Path]:
    snapshot_parquet_path = Path(snapshot_parquet_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = Path(ledger_path) if ledger_path is not None else output_dir / "refinitiv_lookup_api_ledger.sqlite3"
    request_log_path = (
        Path(request_log_path)
        if request_log_path is not None
        else output_dir / "refinitiv_lookup_api_requests.jsonl"
    )

    snapshot_df = _read_lookup_snapshot(snapshot_parquet_path)
    items = _build_lookup_items(snapshot_df)
    stage_run = run_api_batches(
        stage=LOOKUP_STAGE,
        items=items,
        output_dir=output_dir,
        ledger_path=ledger_path,
        request_log_path=request_log_path,
        provider=provider,
        provider_session_name=provider_session_name,
        provider_config_name=provider_config_name,
        provider_timeout_seconds=provider_timeout_seconds,
        preflight_probe=preflight_probe,
        max_batch_size=max_batch_size,
        min_seconds_between_requests=min_seconds_between_requests,
        min_seconds_between_request_starts_total=min_seconds_between_request_starts_total,
        max_attempts=max_attempts,
        response_normalizer=_normalize_lookup_batch_response,
        lookup_normalizer=_normalize_lookup_text,
        retry_delay_seconds_fn=_retry_delay_seconds,
        max_workers=max_workers,
    )

    final_df = _assemble_lookup_output(snapshot_df, stage_run.staging_dir)
    output_path = output_dir / "refinitiv_ric_lookup_handoff_common_stock_extended.parquet"
    candidate_path = _candidate_output_path(output_path)
    _write_parquet_atomic(final_df, candidate_path)

    manifest_path = (
        Path(stage_manifest_path)
        if stage_manifest_path is not None
        else default_stage_manifest_path(output_dir, LOOKUP_STAGE)
    )
    audit_result = audit_api_stage(
        stage_name=LOOKUP_STAGE,
        ledger_path=ledger_path,
        staging_dir=stage_run.staging_dir,
        output_artifacts={"lookup_extended_parquet": candidate_path},
        declared_output_artifacts={"lookup_extended_parquet": output_path},
        rebuilders={
            "lookup_extended_parquet": lambda: _assemble_lookup_output(snapshot_df, stage_run.staging_dir),
        },
        expected_stage_manifest_path=manifest_path,
    )
    if not audit_result.passed:
        raise RuntimeError(f"lookup stage audit failed: {audit_result.to_dict()}")
    _promote_candidate_output(candidate_path, output_path)

    returned_ric_cols = [f"{identifier_type}_returned_ric" for identifier_type in LOOKUP_IDENTIFIER_TYPES]
    rows_with_any_returned_ric = int(
        final_df.select(
            pl.any_horizontal([pl.col(column).is_not_null() for column in returned_ric_cols]).sum()
        ).item()
    )
    write_stage_completion_manifest(
        stage_name=LOOKUP_STAGE,
        manifest_path=manifest_path,
        input_artifacts={"lookup_snapshot_parquet": snapshot_parquet_path},
        output_artifacts={"lookup_extended_parquet": output_path},
        ledger_path=ledger_path,
        request_log_path=request_log_path,
        staging_dir=stage_run.staging_dir,
        audit_result=audit_result,
        summary={
            "snapshot_row_count": int(snapshot_df.height),
            "lookup_item_count": len(items),
            "rows_with_any_returned_ric": rows_with_any_returned_ric,
            "run_session_id": stage_run.run_session_id,
        },
    )
    result = {
        "refinitiv_ric_lookup_handoff_common_stock_extended_parquet": output_path,
        "refinitiv_lookup_api_ledger_sqlite3": ledger_path,
        "refinitiv_lookup_api_requests_jsonl": request_log_path,
        "refinitiv_lookup_stage_manifest_json": manifest_path,
    }
    return result


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

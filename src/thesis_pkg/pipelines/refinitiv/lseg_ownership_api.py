from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from thesis_pkg.pipelines.refinitiv.lseg_api_common import (
    retry_delay_seconds as _retry_delay_seconds,
    standardize_field_frame as _standardize_field_frame,
    write_parquet_atomic as _write_parquet_atomic,
)
from thesis_pkg.pipelines.refinitiv.lseg_api_execution import run_api_batches
from thesis_pkg.pipelines.refinitiv.lseg_batching import RequestItem, stable_hash_id
from thesis_pkg.pipelines.refinitiv.lseg_stage_audit import (
    audit_api_stage,
    default_stage_manifest_path,
    write_stage_completion_manifest,
)
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
    "TR.CategoryOwnershipPct.Date",
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
    provider_session_name: str = "desktop.workspace",
    provider_config_name: str | None = None,
    provider_timeout_seconds: float | None = None,
    preflight_probe: bool = False,
    stage_manifest_path: Path | str | None = None,
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
    stage_run = run_api_batches(
        stage=OWNERSHIP_UNIVERSE_STAGE,
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
        max_attempts=max_attempts,
        response_normalizer=_normalize_ownership_universe_batch_response,
        lookup_normalizer=_normalize_lookup_text,
        split_after_attempt=2,
        retry_delay_seconds_fn=_retry_delay_seconds,
    )

    results_df = _assemble_ownership_universe_results(stage_run.staging_dir)
    row_summary_df = build_refinitiv_ownership_universe_row_summary(handoff_df, results_df)

    results_path = output_dir / "refinitiv_ownership_universe_results.parquet"
    row_summary_path = output_dir / "refinitiv_ownership_universe_row_summary.parquet"
    _write_parquet_atomic(results_df, results_path)
    _write_parquet_atomic(row_summary_df, row_summary_path)

    manifest_path = (
        Path(stage_manifest_path)
        if stage_manifest_path is not None
        else default_stage_manifest_path(output_dir, OWNERSHIP_UNIVERSE_STAGE)
    )
    audit_result = audit_api_stage(
        stage_name=OWNERSHIP_UNIVERSE_STAGE,
        ledger_path=ledger_path,
        staging_dir=stage_run.staging_dir,
        output_artifacts={
            "ownership_results_parquet": results_path,
            "ownership_row_summary_parquet": row_summary_path,
        },
        rebuilders={
            "ownership_results_parquet": lambda: _assemble_ownership_universe_results(stage_run.staging_dir),
            "ownership_row_summary_parquet": lambda: build_refinitiv_ownership_universe_row_summary(
                handoff_df,
                _assemble_ownership_universe_results(stage_run.staging_dir),
            ),
        },
        expected_stage_manifest_path=manifest_path,
    )
    if not audit_result.passed:
        raise RuntimeError(f"ownership universe stage audit failed: {audit_result.to_dict()}")

    write_stage_completion_manifest(
        stage_name=OWNERSHIP_UNIVERSE_STAGE,
        manifest_path=manifest_path,
        input_artifacts={"ownership_handoff_parquet": handoff_parquet_path},
        output_artifacts={
            "ownership_results_parquet": results_path,
            "ownership_row_summary_parquet": row_summary_path,
        },
        ledger_path=ledger_path,
        request_log_path=request_log_path,
        staging_dir=stage_run.staging_dir,
        audit_result=audit_result,
        summary={
            "handoff_row_count": int(handoff_df.height),
            "request_item_count": len(items),
            "results_row_count": int(results_df.height),
            "rows_with_results": int(
                row_summary_df.filter(pl.col("ownership_rows_returned").fill_null(0) > 0).height
            ),
            "run_session_id": stage_run.run_session_id,
        },
    )
    result = {
        "refinitiv_ownership_universe_results_parquet": results_path,
        "refinitiv_ownership_universe_row_summary_parquet": row_summary_path,
        "refinitiv_ownership_universe_api_ledger_sqlite3": ledger_path,
        "refinitiv_ownership_universe_api_requests_jsonl": request_log_path,
        "refinitiv_ownership_universe_stage_manifest_json": manifest_path,
    }
    return result


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
    provider_session_name: str = "desktop.workspace",
    provider_config_name: str | None = None,
    provider_timeout_seconds: float | None = None,
    preflight_probe: bool = False,
    stage_manifest_path: Path | str | None = None,
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
    _write_parquet_atomic(request_df, requests_path)

    items = _build_doc_ownership_exact_items(request_df)
    stage_run = run_api_batches(
        stage=DOC_EXACT_STAGE,
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
        max_attempts=max_attempts,
        response_normalizer=_normalize_doc_exact_batch_response,
        lookup_normalizer=_normalize_lookup_text,
        split_after_attempt=2,
        retry_delay_seconds_fn=_retry_delay_seconds,
    )

    exact_raw_df = _assemble_doc_raw(stage_run.staging_dir)
    exact_raw_path = output_dir / "refinitiv_lm2011_doc_ownership_exact_raw.parquet"
    _write_parquet_atomic(exact_raw_df, exact_raw_path)

    manifest_path = (
        Path(stage_manifest_path)
        if stage_manifest_path is not None
        else default_stage_manifest_path(output_dir, DOC_EXACT_STAGE)
    )
    audit_result = audit_api_stage(
        stage_name=DOC_EXACT_STAGE,
        ledger_path=ledger_path,
        staging_dir=stage_run.staging_dir,
        output_artifacts={
            "exact_requests_parquet": requests_path,
            "exact_raw_parquet": exact_raw_path,
        },
        rebuilders={"exact_raw_parquet": lambda: _assemble_doc_raw(stage_run.staging_dir)},
        expected_stage_manifest_path=manifest_path,
    )
    if not audit_result.passed:
        raise RuntimeError(f"doc exact stage audit failed: {audit_result.to_dict()}")

    write_stage_completion_manifest(
        stage_name=DOC_EXACT_STAGE,
        manifest_path=manifest_path,
        input_artifacts={
            "doc_filing_artifact_parquet": Path(doc_filing_artifact_path),
            "authority_decisions_parquet": Path(authority_decisions_artifact_path),
            "authority_exceptions_parquet": Path(authority_exceptions_artifact_path),
        },
        output_artifacts={
            "exact_requests_parquet": requests_path,
            "exact_raw_parquet": exact_raw_path,
        },
        ledger_path=ledger_path,
        request_log_path=request_log_path,
        staging_dir=stage_run.staging_dir,
        audit_result=audit_result,
        summary={
            "request_row_count": int(request_df.height),
            "eligible_request_row_count": int(
                request_df.filter(pl.col("retrieval_eligible").fill_null(False)).height
            ),
            "raw_row_count": int(exact_raw_df.height),
            "docs_with_raw_rows": int(exact_raw_df.select(pl.col("doc_id").drop_nulls().n_unique()).item())
            if exact_raw_df.height
            else 0,
            "run_session_id": stage_run.run_session_id,
        },
    )
    result = {
        "refinitiv_lm2011_doc_ownership_exact_requests_parquet": requests_path,
        "refinitiv_lm2011_doc_ownership_exact_raw_parquet": exact_raw_path,
        "refinitiv_doc_ownership_exact_api_ledger_sqlite3": ledger_path,
        "refinitiv_doc_ownership_exact_api_requests_jsonl": request_log_path,
        "refinitiv_doc_ownership_exact_stage_manifest_json": manifest_path,
    }
    return result


def run_refinitiv_lm2011_doc_ownership_fallback_api_pipeline(
    *,
    output_dir: Path | str,
    provider: Any | None = None,
    ledger_path: Path | str | None = None,
    request_log_path: Path | str | None = None,
    max_batch_size: int = 5,
    min_seconds_between_requests: float = 2.0,
    max_attempts: int = 4,
    provider_session_name: str = "desktop.workspace",
    provider_config_name: str | None = None,
    provider_timeout_seconds: float | None = None,
    preflight_probe: bool = False,
    stage_manifest_path: Path | str | None = None,
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
    _write_parquet_atomic(fallback_request_df, fallback_requests_path)

    items = _build_doc_ownership_fallback_items(fallback_request_df)
    stage_run = run_api_batches(
        stage=DOC_FALLBACK_STAGE,
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
        max_attempts=max_attempts,
        response_normalizer=_normalize_doc_fallback_batch_response,
        lookup_normalizer=_normalize_lookup_text,
        split_after_attempt=1,
        retry_delay_seconds_fn=_retry_delay_seconds,
    )
    fallback_raw_df = _assemble_doc_raw(stage_run.staging_dir)
    fallback_raw_path = output_dir / "refinitiv_lm2011_doc_ownership_fallback_raw.parquet"
    _write_parquet_atomic(fallback_raw_df, fallback_raw_path)

    manifest_path = (
        Path(stage_manifest_path)
        if stage_manifest_path is not None
        else default_stage_manifest_path(output_dir, DOC_FALLBACK_STAGE)
    )
    audit_result = audit_api_stage(
        stage_name=DOC_FALLBACK_STAGE,
        ledger_path=ledger_path,
        staging_dir=stage_run.staging_dir,
        output_artifacts={
            "fallback_requests_parquet": fallback_requests_path,
            "fallback_raw_parquet": fallback_raw_path,
        },
        rebuilders={"fallback_raw_parquet": lambda: _assemble_doc_raw(stage_run.staging_dir)},
        expected_stage_manifest_path=manifest_path,
    )
    if not audit_result.passed:
        raise RuntimeError(f"doc fallback stage audit failed: {audit_result.to_dict()}")

    write_stage_completion_manifest(
        stage_name=DOC_FALLBACK_STAGE,
        manifest_path=manifest_path,
        input_artifacts={
            "exact_requests_parquet": exact_requests_path,
            "exact_raw_parquet": exact_raw_path,
        },
        output_artifacts={
            "fallback_requests_parquet": fallback_requests_path,
            "fallback_raw_parquet": fallback_raw_path,
        },
        ledger_path=ledger_path,
        request_log_path=request_log_path,
        staging_dir=stage_run.staging_dir,
        audit_result=audit_result,
        summary={
            "fallback_request_row_count": int(fallback_request_df.height),
            "eligible_request_row_count": int(
                fallback_request_df.filter(pl.col("retrieval_eligible").fill_null(False)).height
            ),
            "raw_row_count": int(fallback_raw_df.height),
            "docs_with_raw_rows": int(fallback_raw_df.select(pl.col("doc_id").drop_nulls().n_unique()).item())
            if fallback_raw_df.height
            else 0,
            "run_session_id": stage_run.run_session_id,
        },
    )
    result = {
        "refinitiv_lm2011_doc_ownership_exact_raw_parquet": exact_raw_path,
        "refinitiv_lm2011_doc_ownership_fallback_requests_parquet": fallback_requests_path,
        "refinitiv_lm2011_doc_ownership_fallback_raw_parquet": fallback_raw_path,
        "refinitiv_doc_ownership_fallback_api_ledger_sqlite3": ledger_path,
        "refinitiv_doc_ownership_fallback_api_requests_jsonl": request_log_path,
        "refinitiv_doc_ownership_fallback_stage_manifest_json": manifest_path,
    }
    return result


def _assemble_ownership_universe_results(staging_dir: Path) -> pl.DataFrame:
    staging_paths = sorted(staging_dir.glob("*.parquet"))
    results_df = (
        pl.concat([pl.read_parquet(path) for path in staging_paths], how="vertical_relaxed")
        if staging_paths
        else _empty_ownership_universe_results_df()
    )
    return (
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


def _assemble_doc_raw(staging_dir: Path) -> pl.DataFrame:
    staging_paths = sorted(staging_dir.glob("*.parquet"))
    raw_df = (
        pl.concat([pl.read_parquet(path) for path in staging_paths], how="vertical_relaxed")
        if staging_paths
        else _empty_df(DOC_OWNERSHIP_RAW_COLUMNS, _doc_ownership_raw_schema())
    )
    return (
        _cast_doc_df_to_schema(raw_df.select(DOC_OWNERSHIP_RAW_COLUMNS), _doc_ownership_raw_schema())
        .select(DOC_OWNERSHIP_RAW_COLUMNS)
        .unique(
            subset=["doc_id", "request_stage", "response_date", "returned_category", "returned_value"],
            maintain_order=True,
        )
    )


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
        target_effective_date = _normalize_date_value(row.get("target_effective_date"))
        if doc_id is None or authoritative_ric is None or target_effective_date is None:
            continue
        date_text = target_effective_date.isoformat()
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
                    "request_stage": DOC_OWNERSHIP_EXACT_STAGE,
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

from __future__ import annotations

import json
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any

import polars as pl

from thesis_pkg.benchmarking.contracts import DEFAULT_FINBERT_AUTHORITY
from thesis_pkg.benchmarking.contracts import FinbertAuthoritySpec
from thesis_pkg.benchmarking.contracts import FinbertSentencePreprocessingRunArtifacts
from thesis_pkg.benchmarking.contracts import FinbertSentencePreprocessingRunConfig
from thesis_pkg.benchmarking.finbert_dataset import _resolve_year_paths
from thesis_pkg.benchmarking.finbert_dataset import load_eligible_section_universe
from thesis_pkg.benchmarking.finbert_dataset import section_universe_contract_payload
from thesis_pkg.benchmarking.item_text_cleaning import CLEANING_ROW_AUDIT_SCHEMA
from thesis_pkg.benchmarking.item_text_cleaning import MANUAL_AUDIT_SAMPLE_SCHEMA
from thesis_pkg.benchmarking.item_text_cleaning import SCOPE_DIAGNOSTICS_SCHEMA
from thesis_pkg.benchmarking.item_text_cleaning import build_segment_policy_id
from thesis_pkg.benchmarking.item_text_cleaning import clean_item_scopes_with_audit
from thesis_pkg.benchmarking.item_text_cleaning import cleaned_scopes_for_sentence_materialization
from thesis_pkg.benchmarking.run_logging import utc_timestamp
from thesis_pkg.benchmarking.sentences import SENTENCE_CHUNK_CHAR_LIMIT
from thesis_pkg.benchmarking.sentences import SENTENCE_SPLIT_AUDIT_SCHEMA
from thesis_pkg.benchmarking.sentences import _derive_sentence_frame_with_split_audit
from thesis_pkg.benchmarking.token_lengths import FINBERT_TOKEN_BUCKET_COLUMN


RUNNER_NAME = "finbert_sentence_preprocessing"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _annotate_analysis_sections(sections_df: pl.DataFrame) -> pl.DataFrame:
    if sections_df.is_empty():
        return sections_df.with_columns(pl.lit(None, dtype=pl.Utf8).alias("benchmark_row_id"))
    return sections_df.with_columns(
        pl.concat_str([pl.col("doc_id"), pl.lit(":"), pl.col("benchmark_item_code")]).alias("benchmark_row_id")
    )


def _empty_split_audit_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=SENTENCE_SPLIT_AUDIT_SCHEMA)


def _empty_frame(schema: dict[str, pl.DataType]) -> pl.DataFrame:
    return pl.DataFrame(schema=schema)


def _align_frame_to_schema(df: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    return df.select(
        [
            (
                pl.col(name).cast(dtype, strict=False).alias(name)
                if name in df.columns
                else pl.lit(None, dtype=dtype).alias(name)
            )
            for name, dtype in schema.items()
        ]
    )


def _concat_frames(frames: list[pl.DataFrame], schema: dict[str, pl.DataType]) -> pl.DataFrame:
    non_empty = [frame for frame in frames if not frame.is_empty()]
    if not non_empty:
        return _empty_frame(schema)
    return pl.concat(
        [_align_frame_to_schema(frame, schema) for frame in non_empty],
        how="vertical_relaxed",
    )


def _load_existing_summary_rows(path: Path, *, overwrite: bool) -> dict[int, dict[str, Any]]:
    if overwrite or not path.exists():
        return {}

    summary_df = pl.read_parquet(path)
    if summary_df.is_empty() or "filing_year" not in summary_df.columns:
        return {}
    return {int(row["filing_year"]): row for row in summary_df.to_dicts()}


def _load_existing_frame(path: Path, *, schema: dict[str, pl.DataType], overwrite: bool) -> pl.DataFrame:
    if overwrite or not path.exists():
        return _empty_frame(schema)
    return _align_frame_to_schema(pl.read_parquet(path), schema)


def _filter_existing_years(df: pl.DataFrame, years: set[int], *, year_col: str = "calendar_year") -> pl.DataFrame:
    if df.is_empty() or not years or year_col not in df.columns:
        return df.head(0)
    return df.filter(pl.col(year_col).is_in(sorted(years)))


def _load_existing_split_audit(path: Path, *, overwrite: bool) -> pl.DataFrame:
    if overwrite or not path.exists():
        return _empty_split_audit_frame()

    split_audit_df = pl.read_parquet(path)
    if split_audit_df.is_empty():
        return _empty_split_audit_frame()
    return split_audit_df


def _split_metrics(split_audit_df: pl.DataFrame) -> dict[str, Any]:
    if split_audit_df.is_empty():
        return {
            "chunked_section_rows": 0,
            "warning_split_rows": 0,
            "max_original_char_count": None,
        }

    warning_split_df = split_audit_df.filter(pl.col("warning_boundary_used"))
    return {
        "chunked_section_rows": int(split_audit_df["benchmark_row_id"].n_unique()),
        "warning_split_rows": int(warning_split_df["benchmark_row_id"].n_unique()) if warning_split_df.height else 0,
        "max_original_char_count": int(split_audit_df["original_char_count"].max()),
    }


def _warn_on_fallback_split_boundaries(filing_year: int, split_audit_df: pl.DataFrame) -> None:
    warning_split_df = split_audit_df.filter(pl.col("warning_boundary_used"))
    if warning_split_df.is_empty():
        return

    affected_row_count = int(warning_split_df["benchmark_row_id"].n_unique())
    reason_counts = warning_split_df.group_by("split_reason").agg(pl.len().alias("split_count")).sort("split_reason")
    counts_text = ", ".join(
        f"{row['split_reason']}={int(row['split_count'])}"
        for row in reason_counts.to_dicts()
    )
    warnings.warn(
        (
            f"Sentence preprocessing for filing year {filing_year} chunked {affected_row_count} oversized "
            f"sections using fallback split boundaries ({counts_text})."
        ),
        RuntimeWarning,
        stacklevel=2,
    )


def _resolve_year_paths_for_run(run_cfg: FinbertSentencePreprocessingRunConfig) -> list[Path]:
    year_paths = _resolve_year_paths(run_cfg.source_items_dir)
    if run_cfg.year_filter is None:
        return year_paths

    target_years = set(run_cfg.year_filter)
    selected = [path for path in year_paths if int(path.stem) in target_years]
    missing_years = sorted(target_years - {int(path.stem) for path in selected})
    if missing_years:
        raise FileNotFoundError(
            f"Requested filing years were not found in {run_cfg.source_items_dir}: {missing_years}"
        )
    return selected


def _semantic_preprocessing_payload(
    run_cfg: FinbertSentencePreprocessingRunConfig,
    authority: FinbertAuthoritySpec,
    segment_policy_id: str,
) -> dict[str, Any]:
    payload = {
        "authority": asdict(authority),
        "sentence_dataset": asdict(run_cfg.sentence_dataset),
        "cleaning": asdict(run_cfg.cleaning),
        "cleaning_policy_id": run_cfg.cleaning.cleaning_policy_id if run_cfg.cleaning.enabled else "raw_item_text",
        "segment_policy_id": segment_policy_id,
        "source_items_dir": str(run_cfg.source_items_dir.resolve()),
        "target_doc_universe_path": (
            str(run_cfg.target_doc_universe_path.resolve())
            if run_cfg.target_doc_universe_path is not None
            else None
        ),
        "year_filter": list(run_cfg.year_filter) if run_cfg.year_filter is not None else None,
    }
    return json.loads(json.dumps(payload))


def _assert_existing_run_compatible(
    run_manifest_path: Path,
    run_cfg: FinbertSentencePreprocessingRunConfig,
    authority: FinbertAuthoritySpec,
    segment_policy_id: str,
) -> None:
    if run_cfg.overwrite or not run_manifest_path.exists():
        return
    manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    expected = _semantic_preprocessing_payload(run_cfg, authority, segment_policy_id)
    existing = manifest.get("semantic_reuse_guard", {})
    mismatched = [
        key
        for key, expected_value in expected.items()
        if existing.get(key) != expected_value
    ]
    if mismatched:
        raise ValueError(
            "Existing FinBERT sentence preprocessing artifacts were created with incompatible "
            f"semantic settings for {mismatched}. Use a new run_name or set overwrite=True."
        )


def _summary_row(
    *,
    filing_year: int,
    status: str,
    source_path: Path,
    sentence_path: Path,
    sections_df: pl.DataFrame | None,
    sentence_df: pl.DataFrame,
    split_audit_df: pl.DataFrame | None,
    cleaning_row_audit_df: pl.DataFrame | None = None,
    flagged_rows_df: pl.DataFrame | None = None,
    existing_summary_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bucket_counts = (
        {
            row[FINBERT_TOKEN_BUCKET_COLUMN]: int(row["sentence_rows"])
            for row in sentence_df.group_by(FINBERT_TOKEN_BUCKET_COLUMN)
            .agg(pl.len().alias("sentence_rows"))
            .to_dicts()
        }
        if not sentence_df.is_empty()
        else {}
    )
    split_metrics = _split_metrics(split_audit_df if split_audit_df is not None else _empty_split_audit_frame())
    if sections_df is not None and sections_df.height:
        max_section_char_count = int(sections_df["char_count"].max()) if "char_count" in sections_df.columns else None
        oversize_section_rows = int((sections_df["char_count"] > SENTENCE_CHUNK_CHAR_LIMIT).sum()) if "char_count" in sections_df.columns else 0
    else:
        max_section_char_count = (
            existing_summary_row.get("max_section_char_count")
            if existing_summary_row is not None
            else split_metrics["max_original_char_count"]
        )
        oversize_section_rows = (
            int(existing_summary_row.get("oversize_section_rows") or 0)
            if existing_summary_row is not None
            else int(split_metrics["chunked_section_rows"])
        )

    return {
        "filing_year": filing_year,
        "status": status,
        "source_path": str(source_path.resolve()),
        "sentence_dataset_path": str(sentence_path.resolve()),
        "section_rows": (
            int(sections_df.height)
            if sections_df is not None
            else existing_summary_row.get("section_rows") if existing_summary_row is not None else None
        ),
        "doc_count": (
            int(sections_df["doc_id"].n_unique())
            if sections_df is not None and sections_df.height
            else int(sentence_df["doc_id"].n_unique()) if sentence_df.height else 0
        ),
        "max_section_char_count": max_section_char_count,
        "oversize_section_rows": oversize_section_rows,
        "chunked_section_rows": int(split_metrics["chunked_section_rows"]),
        "warning_split_rows": int(split_metrics["warning_split_rows"]),
        "sentence_rows": int(sentence_df.height),
        "short_sentence_rows": int(bucket_counts.get("short", 0)),
        "medium_sentence_rows": int(bucket_counts.get("medium", 0)),
        "long_sentence_rows": int(bucket_counts.get("long", 0)),
        "cleaned_scope_rows": (
            int((~cleaning_row_audit_df["dropped_after_cleaning"]).sum())
            if cleaning_row_audit_df is not None and not cleaning_row_audit_df.is_empty()
            else int(existing_summary_row.get("cleaned_scope_rows") or 0)
            if existing_summary_row is not None
            else 0
        ),
        "cleaning_dropped_rows": (
            int(cleaning_row_audit_df["dropped_after_cleaning"].sum())
            if cleaning_row_audit_df is not None and not cleaning_row_audit_df.is_empty()
            else int(existing_summary_row.get("cleaning_dropped_rows") or 0)
            if existing_summary_row is not None
            else 0
        ),
        "cleaning_flagged_rows": (
            int(flagged_rows_df.height)
            if flagged_rows_df is not None
            else int(existing_summary_row.get("cleaning_flagged_rows") or 0)
            if existing_summary_row is not None
            else 0
        ),
    }


def run_finbert_sentence_preprocessing(
    run_cfg: FinbertSentencePreprocessingRunConfig,
    *,
    authority: FinbertAuthoritySpec = DEFAULT_FINBERT_AUTHORITY,
) -> FinbertSentencePreprocessingRunArtifacts:
    year_paths = _resolve_year_paths_for_run(run_cfg)
    run_name = run_cfg.run_name or f"{RUNNER_NAME}_{utc_timestamp().replace(':', '')}"
    run_dir = run_cfg.out_root / run_name
    sentence_dataset_dir = run_dir / "sentence_dataset" / "by_year"
    cleaned_item_scopes_dir = run_dir / "cleaned_item_scopes" / "by_year"
    oversize_sections_path = run_dir / "oversize_sections.parquet"
    yearly_summary_path = run_dir / "sentence_dataset_yearly_summary.parquet"
    yearly_summary_csv_path = run_dir / "sentence_dataset_yearly_summary.csv"
    cleaning_row_audit_path = run_dir / "cleaning_row_audit.parquet"
    cleaning_flagged_rows_path = run_dir / "cleaning_flagged_rows.parquet"
    item_scope_cleaning_diagnostics_path = run_dir / "item_scope_cleaning_diagnostics.parquet"
    item_scope_cleaning_diagnostics_csv_path = run_dir / "item_scope_cleaning_diagnostics.csv"
    manual_boundary_audit_sample_path = run_dir / "manual_boundary_audit_sample.parquet"
    run_manifest_path = run_dir / "run_manifest.json"
    segment_policy_id = build_segment_policy_id(
        run_cfg.sentence_dataset,
        run_cfg.cleaning,
        authority,
        chunk_char_limit=SENTENCE_CHUNK_CHAR_LIMIT,
    )

    _assert_existing_run_compatible(run_manifest_path, run_cfg, authority, segment_policy_id)

    existing_summary_rows = _load_existing_summary_rows(yearly_summary_path, overwrite=run_cfg.overwrite)
    existing_split_audit_df = _load_existing_split_audit(oversize_sections_path, overwrite=run_cfg.overwrite)
    existing_cleaning_row_audit_df = _load_existing_frame(
        cleaning_row_audit_path,
        schema=CLEANING_ROW_AUDIT_SCHEMA,
        overwrite=run_cfg.overwrite,
    )
    existing_flagged_rows_df = _load_existing_frame(
        cleaning_flagged_rows_path,
        schema=CLEANING_ROW_AUDIT_SCHEMA,
        overwrite=run_cfg.overwrite,
    )
    existing_scope_diagnostics_df = _load_existing_frame(
        item_scope_cleaning_diagnostics_path,
        schema=SCOPE_DIAGNOSTICS_SCHEMA,
        overwrite=run_cfg.overwrite,
    )
    existing_manual_audit_df = _load_existing_frame(
        manual_boundary_audit_sample_path,
        schema=MANUAL_AUDIT_SAMPLE_SCHEMA,
        overwrite=run_cfg.overwrite,
    )
    summary_rows: list[dict[str, Any]] = []
    processed_split_audit_chunks: list[pl.DataFrame] = []
    processed_cleaning_row_audit_chunks: list[pl.DataFrame] = []
    processed_flagged_rows_chunks: list[pl.DataFrame] = []
    processed_scope_diagnostics_chunks: list[pl.DataFrame] = []
    processed_manual_audit_chunks: list[pl.DataFrame] = []
    reused_years: set[int] = set()
    for year_path in year_paths:
        filing_year = int(year_path.stem)
        sentence_path = sentence_dataset_dir / f"{filing_year}.parquet"
        existing_summary_row = existing_summary_rows.get(filing_year)
        if sentence_path.exists() and not run_cfg.overwrite:
            sentence_df = pl.read_parquet(sentence_path)
            reused_years.add(filing_year)
            split_audit_df = (
                existing_split_audit_df.filter(pl.col("filing_year") == filing_year)
                if not existing_split_audit_df.is_empty()
                else _empty_split_audit_frame()
            )
            cleaning_year_df = _filter_existing_years(
                existing_cleaning_row_audit_df,
                {filing_year},
                year_col="calendar_year",
            )
            flagged_year_df = _filter_existing_years(
                existing_flagged_rows_df,
                {filing_year},
                year_col="calendar_year",
            )
            summary_rows.append(
                _summary_row(
                    filing_year=filing_year,
                    status="reused_existing",
                    source_path=year_path,
                    sentence_path=sentence_path,
                    sections_df=None,
                    sentence_df=sentence_df,
                    split_audit_df=split_audit_df,
                    cleaning_row_audit_df=cleaning_year_df,
                    flagged_rows_df=flagged_year_df,
                    existing_summary_row=existing_summary_row,
                )
            )
            continue

        sections_df = (
            load_eligible_section_universe(
                run_cfg.section_universe,
                year_paths=[year_path],
                target_doc_universe_path=run_cfg.target_doc_universe_path,
            ).collect()
        )
        sections_df = _annotate_analysis_sections(sections_df)
        cleaning_result = clean_item_scopes_with_audit(
            sections_df,
            run_cfg.cleaning,
            segment_policy_id=segment_policy_id,
        )
        cleaned_scope_path = cleaned_item_scopes_dir / f"{filing_year}.parquet"
        cleaned_scope_path.parent.mkdir(parents=True, exist_ok=True)
        cleaning_result.cleaned_scope_df.write_parquet(
            cleaned_scope_path,
            compression=run_cfg.sentence_dataset.compression,
        )
        sentence_sections_df = cleaned_scopes_for_sentence_materialization(
            cleaning_result.cleaned_scope_df
        )
        sentence_df, split_audit_df = _derive_sentence_frame_with_split_audit(
            sentence_sections_df,
            run_cfg.sentence_dataset,
            authority=authority,
        )
        processed_split_audit_chunks.append(split_audit_df)
        processed_cleaning_row_audit_chunks.append(cleaning_result.row_audit_df)
        processed_flagged_rows_chunks.append(cleaning_result.flagged_rows_df)
        processed_scope_diagnostics_chunks.append(cleaning_result.scope_diagnostics_df)
        processed_manual_audit_chunks.append(cleaning_result.manual_audit_sample_df)
        sentence_path.parent.mkdir(parents=True, exist_ok=True)
        sentence_df.write_parquet(sentence_path, compression=run_cfg.sentence_dataset.compression)
        _warn_on_fallback_split_boundaries(filing_year, split_audit_df)
        summary_rows.append(
            _summary_row(
                filing_year=filing_year,
                status="processed",
                source_path=year_path,
                sentence_path=sentence_path,
                sections_df=sections_df,
                sentence_df=sentence_df,
                split_audit_df=split_audit_df,
                cleaning_row_audit_df=cleaning_result.row_audit_df,
                flagged_rows_df=cleaning_result.flagged_rows_df,
            )
        )

    summary_df = pl.DataFrame(summary_rows).sort("filing_year") if summary_rows else pl.DataFrame()
    run_dir.mkdir(parents=True, exist_ok=True)
    final_split_audit_chunks: list[pl.DataFrame] = []
    final_cleaning_row_audit_chunks: list[pl.DataFrame] = []
    final_flagged_rows_chunks: list[pl.DataFrame] = []
    final_scope_diagnostics_chunks: list[pl.DataFrame] = []
    final_manual_audit_chunks: list[pl.DataFrame] = []
    if reused_years and not existing_split_audit_df.is_empty():
        final_split_audit_chunks.append(
            existing_split_audit_df.filter(pl.col("filing_year").is_in(sorted(reused_years)))
        )
    if reused_years:
        final_cleaning_row_audit_chunks.append(
            _filter_existing_years(existing_cleaning_row_audit_df, reused_years)
        )
        final_flagged_rows_chunks.append(
            _filter_existing_years(existing_flagged_rows_df, reused_years)
        )
        final_scope_diagnostics_chunks.append(
            _filter_existing_years(existing_scope_diagnostics_df, reused_years)
        )
        final_manual_audit_chunks.append(
            _filter_existing_years(existing_manual_audit_df, reused_years)
        )
    final_split_audit_chunks.extend(processed_split_audit_chunks)
    final_cleaning_row_audit_chunks.extend(processed_cleaning_row_audit_chunks)
    final_flagged_rows_chunks.extend(processed_flagged_rows_chunks)
    final_scope_diagnostics_chunks.extend(processed_scope_diagnostics_chunks)
    final_manual_audit_chunks.extend(processed_manual_audit_chunks)
    non_empty_split_audit_chunks = [chunk for chunk in final_split_audit_chunks if not chunk.is_empty()]
    final_split_audit_df = (
        pl.concat(non_empty_split_audit_chunks, how="vertical_relaxed")
        if non_empty_split_audit_chunks
        else _empty_split_audit_frame()
    )
    final_cleaning_row_audit_df = _concat_frames(final_cleaning_row_audit_chunks, CLEANING_ROW_AUDIT_SCHEMA)
    final_flagged_rows_df = _concat_frames(final_flagged_rows_chunks, CLEANING_ROW_AUDIT_SCHEMA)
    final_scope_diagnostics_df = _concat_frames(final_scope_diagnostics_chunks, SCOPE_DIAGNOSTICS_SCHEMA)
    final_manual_audit_df = _concat_frames(final_manual_audit_chunks, MANUAL_AUDIT_SAMPLE_SCHEMA)
    final_split_audit_df.write_parquet(oversize_sections_path, compression="zstd")
    final_cleaning_row_audit_df.write_parquet(cleaning_row_audit_path, compression="zstd")
    final_flagged_rows_df.write_parquet(cleaning_flagged_rows_path, compression="zstd")
    final_scope_diagnostics_df.write_parquet(item_scope_cleaning_diagnostics_path, compression="zstd")
    final_scope_diagnostics_df.write_csv(item_scope_cleaning_diagnostics_csv_path)
    final_manual_audit_df.write_parquet(manual_boundary_audit_sample_path, compression="zstd")
    summary_df.write_parquet(yearly_summary_path, compression="zstd")
    summary_df.write_csv(yearly_summary_csv_path)

    manifest = {
        "runner_name": RUNNER_NAME,
        "run_name": run_name,
        "created_at_utc": utc_timestamp(),
        "authority": asdict(authority),
        "sentence_dataset": asdict(run_cfg.sentence_dataset),
        "cleaning": asdict(run_cfg.cleaning),
        "cleaning_policy_id": run_cfg.cleaning.cleaning_policy_id if run_cfg.cleaning.enabled else "raw_item_text",
        "segment_policy_id": segment_policy_id,
        "semantic_reuse_guard": _semantic_preprocessing_payload(run_cfg, authority, segment_policy_id),
        "section_universe": {
            "source_items_dir": str(run_cfg.section_universe.source_items_dir.resolve()),
            "form_types": list(run_cfg.section_universe.form_types),
            "target_items": [asdict(item) for item in run_cfg.section_universe.target_items],
            "require_active_items": run_cfg.section_universe.require_active_items,
            "require_exists_by_regime": run_cfg.section_universe.require_exists_by_regime,
            "min_char_count": run_cfg.section_universe.min_char_count,
        },
        "accepted_universe_contract": section_universe_contract_payload(
            run_cfg.section_universe,
            target_doc_universe_path=run_cfg.target_doc_universe_path,
        ),
        "target_doc_universe_path": (
            str(run_cfg.target_doc_universe_path.resolve())
            if run_cfg.target_doc_universe_path is not None
            else None
        ),
        "year_filter": list(run_cfg.year_filter) if run_cfg.year_filter is not None else None,
        "overwrite": run_cfg.overwrite,
        "note": run_cfg.note,
        "counts": {
            "year_count": len(summary_rows),
            "processed_year_count": sum(1 for row in summary_rows if row["status"] == "processed"),
            "reused_year_count": sum(1 for row in summary_rows if row["status"] == "reused_existing"),
            "sentence_rows": int(summary_df["sentence_rows"].sum()) if summary_df.height else 0,
            "oversize_section_rows": int(summary_df["oversize_section_rows"].sum()) if summary_df.height else 0,
            "chunked_section_rows": int(summary_df["chunked_section_rows"].sum()) if summary_df.height else 0,
            "warning_split_rows": int(summary_df["warning_split_rows"].sum()) if summary_df.height else 0,
            "cleaned_scope_rows": int(summary_df["cleaned_scope_rows"].sum()) if summary_df.height else 0,
            "cleaning_dropped_rows": int(summary_df["cleaning_dropped_rows"].sum()) if summary_df.height else 0,
            "cleaning_flagged_rows": int(summary_df["cleaning_flagged_rows"].sum()) if summary_df.height else 0,
        },
        "artifacts": {
            "run_dir": str(run_dir.resolve()),
            "sentence_dataset_dir": str(sentence_dataset_dir.resolve()),
            "cleaned_item_scopes_dir": str(cleaned_item_scopes_dir.resolve()),
            "oversize_sections_path": str(oversize_sections_path.resolve()),
            "cleaning_row_audit_path": str(cleaning_row_audit_path.resolve()),
            "cleaning_flagged_rows_path": str(cleaning_flagged_rows_path.resolve()),
            "item_scope_cleaning_diagnostics_path": str(item_scope_cleaning_diagnostics_path.resolve()),
            "item_scope_cleaning_diagnostics_csv_path": str(item_scope_cleaning_diagnostics_csv_path.resolve()),
            "manual_boundary_audit_sample_path": str(manual_boundary_audit_sample_path.resolve()),
            "yearly_summary_path": str(yearly_summary_path.resolve()),
            "yearly_summary_csv_path": str(yearly_summary_csv_path.resolve()),
        },
    }
    _write_json(run_manifest_path, manifest)

    return FinbertSentencePreprocessingRunArtifacts(
        run_dir=run_dir,
        run_manifest_path=run_manifest_path,
        sentence_dataset_dir=sentence_dataset_dir,
        yearly_summary_path=yearly_summary_path,
        oversize_sections_path=oversize_sections_path,
        cleaned_item_scopes_dir=cleaned_item_scopes_dir,
        cleaning_row_audit_path=cleaning_row_audit_path,
        cleaning_flagged_rows_path=cleaning_flagged_rows_path,
        item_scope_cleaning_diagnostics_path=item_scope_cleaning_diagnostics_path,
        manual_boundary_audit_sample_path=manual_boundary_audit_sample_path,
    )

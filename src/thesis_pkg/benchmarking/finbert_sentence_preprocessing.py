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
from thesis_pkg.benchmarking.manifest_contracts import MANIFEST_PATH_SEMANTICS_RELATIVE
from thesis_pkg.benchmarking.manifest_contracts import json_sha256
from thesis_pkg.benchmarking.manifest_contracts import make_semantic_reuse_guard
from thesis_pkg.benchmarking.manifest_contracts import parquet_doc_universe_fingerprint
from thesis_pkg.benchmarking.manifest_contracts import relative_artifact_path
from thesis_pkg.benchmarking.manifest_contracts import semantic_file_fingerprint
from thesis_pkg.benchmarking.manifest_contracts import semantic_guard_mismatches
from thesis_pkg.benchmarking.manifest_contracts import write_manifest_path_value
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
        "year_filter": list(run_cfg.year_filter) if run_cfg.year_filter is not None else None,
    }
    return json.loads(json.dumps(payload))


def _accepted_universe_contract_fingerprint(run_cfg: FinbertSentencePreprocessingRunConfig) -> str:
    return json_sha256(
        section_universe_contract_payload(
            run_cfg.section_universe,
            target_doc_universe_path=run_cfg.target_doc_universe_path,
        )
    )


def _target_doc_universe_fingerprint(run_cfg: FinbertSentencePreprocessingRunConfig) -> dict[str, Any] | None:
    if run_cfg.target_doc_universe_path is None:
        return None
    return parquet_doc_universe_fingerprint(run_cfg.target_doc_universe_path)


def _source_items_fingerprints(year_paths: list[Path]) -> dict[str, dict[str, Any]]:
    return {
        path.stem: semantic_file_fingerprint(path)
        for path in year_paths
    }


def _semantic_preprocessing_guard(
    run_cfg: FinbertSentencePreprocessingRunConfig,
    authority: FinbertAuthoritySpec,
    segment_policy_id: str,
    *,
    year_paths: list[Path],
) -> dict[str, Any]:
    return make_semantic_reuse_guard(
        version="sentence_preprocessing_v3",
        payload=_semantic_preprocessing_payload(run_cfg, authority, segment_policy_id),
        fingerprints={
            "accepted_universe_contract_fingerprint": _accepted_universe_contract_fingerprint(run_cfg),
            "target_doc_universe_fingerprint": _target_doc_universe_fingerprint(run_cfg),
            "source_items_by_year": _source_items_fingerprints(year_paths),
        },
    )


def _assert_existing_run_compatible(
    run_manifest_path: Path,
    run_cfg: FinbertSentencePreprocessingRunConfig,
    authority: FinbertAuthoritySpec,
    segment_policy_id: str,
    *,
    year_paths: list[Path],
) -> None:
    if run_cfg.overwrite or not run_manifest_path.exists():
        return
    manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    expected = _semantic_preprocessing_guard(
        run_cfg,
        authority,
        segment_policy_id,
        year_paths=year_paths,
    )
    existing = manifest.get("semantic_reuse_guard", {})
    mismatched = semantic_guard_mismatches(existing, expected)
    if mismatched:
        raise ValueError(
            "Existing FinBERT sentence preprocessing artifacts were created with incompatible "
            f"semantic settings for {mismatched}. Use a new run_name or set overwrite=True."
        )


def _summary_row(
    *,
    filing_year: int,
    status: str,
    run_dir: Path,
    source_path: Path,
    sentence_path: Path,
    sections_df: pl.DataFrame | None,
    sentence_df: pl.DataFrame,
    split_audit_df: pl.DataFrame | None,
    cleaning_row_audit_df: pl.DataFrame | None = None,
    flagged_rows_df: pl.DataFrame | None = None,
    scope_diagnostics_df: pl.DataFrame | None = None,
    manual_audit_df: pl.DataFrame | None = None,
    existing_summary_row: dict[str, Any] | None = None,
    source_items_shard_fingerprint: dict[str, Any] | None = None,
    accepted_universe_contract_fingerprint: str | None = None,
    target_doc_universe_fingerprint: dict[str, Any] | None = None,
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
        "source_path": relative_artifact_path(source_path, base_path=run_dir),
        "sentence_dataset_path": relative_artifact_path(sentence_path, base_path=run_dir),
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
        "split_audit_rows": (
            int(split_audit_df.height)
            if split_audit_df is not None
            else int(existing_summary_row.get("split_audit_rows") or 0)
            if existing_summary_row is not None
            else 0
        ),
        "cleaning_row_audit_rows": (
            int(cleaning_row_audit_df.height)
            if cleaning_row_audit_df is not None
            else int(existing_summary_row.get("cleaning_row_audit_rows") or 0)
            if existing_summary_row is not None
            else 0
        ),
        "scope_diagnostics_rows": (
            int(scope_diagnostics_df.height)
            if scope_diagnostics_df is not None
            else int(existing_summary_row.get("scope_diagnostics_rows") or 0)
            if existing_summary_row is not None
            else 0
        ),
        "manual_audit_rows": (
            int(manual_audit_df.height)
            if manual_audit_df is not None
            else int(existing_summary_row.get("manual_audit_rows") or 0)
            if existing_summary_row is not None
            else 0
        ),
        "source_items_shard_fingerprint": json.dumps(source_items_shard_fingerprint, sort_keys=True)
        if source_items_shard_fingerprint is not None
        else existing_summary_row.get("source_items_shard_fingerprint")
        if existing_summary_row is not None
        else None,
        "accepted_universe_contract_fingerprint": (
            accepted_universe_contract_fingerprint
            if accepted_universe_contract_fingerprint is not None
            else existing_summary_row.get("accepted_universe_contract_fingerprint")
            if existing_summary_row is not None
            else None
        ),
        "target_doc_universe_fingerprint": json.dumps(target_doc_universe_fingerprint, sort_keys=True)
        if target_doc_universe_fingerprint is not None
        else existing_summary_row.get("target_doc_universe_fingerprint")
        if existing_summary_row is not None
        else None,
    }


def _can_reuse_existing_year(
    *,
    filing_year: int,
    year_path: Path,
    sentence_path: Path,
    cleaned_scope_path: Path,
    existing_summary_row: dict[str, Any] | None,
    split_audit_df: pl.DataFrame,
    cleaning_year_df: pl.DataFrame,
    flagged_year_df: pl.DataFrame,
    scope_diagnostics_year_df: pl.DataFrame,
    manual_audit_year_df: pl.DataFrame,
    required_artifact_paths: tuple[Path, ...],
    accepted_universe_contract_fingerprint: str,
    target_doc_universe_fingerprint: dict[str, Any] | None,
) -> bool:
    if existing_summary_row is None:
        return False
    if not sentence_path.exists() or not cleaned_scope_path.exists():
        return False
    if any(not path.exists() for path in required_artifact_paths):
        return False

    expected_source_fingerprint = json.dumps(
        semantic_file_fingerprint(year_path),
        sort_keys=True,
    )
    if existing_summary_row.get("source_items_shard_fingerprint") != expected_source_fingerprint:
        return False
    if (
        existing_summary_row.get("accepted_universe_contract_fingerprint")
        != accepted_universe_contract_fingerprint
    ):
        return False
    expected_target_fingerprint = (
        json.dumps(target_doc_universe_fingerprint, sort_keys=True)
        if target_doc_universe_fingerprint is not None
        else None
    )
    if existing_summary_row.get("target_doc_universe_fingerprint") != expected_target_fingerprint:
        return False

    expected_counts = {
        "split_audit_rows": int(split_audit_df.height),
        "cleaning_row_audit_rows": int(cleaning_year_df.height),
        "cleaning_flagged_rows": int(flagged_year_df.height),
        "scope_diagnostics_rows": int(scope_diagnostics_year_df.height),
        "manual_audit_rows": int(manual_audit_year_df.height),
    }
    for key, expected_count in expected_counts.items():
        if int(existing_summary_row.get(key) or 0) != expected_count:
            return False
    return True


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

    _assert_existing_run_compatible(
        run_manifest_path,
        run_cfg,
        authority,
        segment_policy_id,
        year_paths=year_paths,
    )

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
    accepted_universe_contract_fingerprint = _accepted_universe_contract_fingerprint(run_cfg)
    target_doc_universe_fingerprint = _target_doc_universe_fingerprint(run_cfg)
    for year_path in year_paths:
        filing_year = int(year_path.stem)
        sentence_path = sentence_dataset_dir / f"{filing_year}.parquet"
        cleaned_scope_path = cleaned_item_scopes_dir / f"{filing_year}.parquet"
        existing_summary_row = existing_summary_rows.get(filing_year)
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
        scope_diagnostics_year_df = _filter_existing_years(
            existing_scope_diagnostics_df,
            {filing_year},
        )
        manual_audit_year_df = _filter_existing_years(
            existing_manual_audit_df,
            {filing_year},
        )
        if not run_cfg.overwrite and _can_reuse_existing_year(
            filing_year=filing_year,
            year_path=year_path,
            sentence_path=sentence_path,
            cleaned_scope_path=cleaned_scope_path,
            existing_summary_row=existing_summary_row,
            split_audit_df=split_audit_df,
            cleaning_year_df=cleaning_year_df,
            flagged_year_df=flagged_year_df,
            scope_diagnostics_year_df=scope_diagnostics_year_df,
            manual_audit_year_df=manual_audit_year_df,
            required_artifact_paths=(
                run_manifest_path,
                cleaning_row_audit_path,
                cleaning_flagged_rows_path,
                item_scope_cleaning_diagnostics_path,
                manual_boundary_audit_sample_path,
                oversize_sections_path,
            ),
            accepted_universe_contract_fingerprint=accepted_universe_contract_fingerprint,
            target_doc_universe_fingerprint=target_doc_universe_fingerprint,
        ):
            sentence_df = pl.read_parquet(sentence_path)
            reused_years.add(filing_year)
            summary_rows.append(
                _summary_row(
                    filing_year=filing_year,
                    status="reused_existing",
                    run_dir=run_dir,
                    source_path=year_path,
                    sentence_path=sentence_path,
                    sections_df=None,
                    sentence_df=sentence_df,
                    split_audit_df=split_audit_df,
                    cleaning_row_audit_df=cleaning_year_df,
                    flagged_rows_df=flagged_year_df,
                    scope_diagnostics_df=scope_diagnostics_year_df,
                    manual_audit_df=manual_audit_year_df,
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
                    run_dir=run_dir,
                    source_path=year_path,
                    sentence_path=sentence_path,
                sections_df=sections_df,
                sentence_df=sentence_df,
                split_audit_df=split_audit_df,
                cleaning_row_audit_df=cleaning_result.row_audit_df,
                flagged_rows_df=cleaning_result.flagged_rows_df,
                scope_diagnostics_df=cleaning_result.scope_diagnostics_df,
                manual_audit_df=cleaning_result.manual_audit_sample_df,
                source_items_shard_fingerprint=semantic_file_fingerprint(year_path),
                accepted_universe_contract_fingerprint=accepted_universe_contract_fingerprint,
                target_doc_universe_fingerprint=target_doc_universe_fingerprint,
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
        "path_semantics": MANIFEST_PATH_SEMANTICS_RELATIVE,
        "runner_name": RUNNER_NAME,
        "run_name": run_name,
        "created_at_utc": utc_timestamp(),
        "authority": asdict(authority),
        "sentence_dataset": asdict(run_cfg.sentence_dataset),
        "cleaning": asdict(run_cfg.cleaning),
        "cleaning_policy_id": run_cfg.cleaning.cleaning_policy_id if run_cfg.cleaning.enabled else "raw_item_text",
        "segment_policy_id": segment_policy_id,
        "semantic_reuse_guard": _semantic_preprocessing_guard(
            run_cfg,
            authority,
            segment_policy_id,
            year_paths=year_paths,
        ),
        "section_universe": {
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
        "year_filter": list(run_cfg.year_filter) if run_cfg.year_filter is not None else None,
        "overwrite": run_cfg.overwrite,
        "note": run_cfg.note,
        "nonportable_diagnostics": {
            "source_items_dir": str(run_cfg.source_items_dir.resolve()),
            "target_doc_universe_path": (
                str(run_cfg.target_doc_universe_path.resolve())
                if run_cfg.target_doc_universe_path is not None
                else None
            ),
        },
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
            "run_dir": write_manifest_path_value(
                run_dir,
                manifest_path=run_manifest_path,
                path_semantics=MANIFEST_PATH_SEMANTICS_RELATIVE,
            ),
            "sentence_dataset_dir": write_manifest_path_value(
                sentence_dataset_dir,
                manifest_path=run_manifest_path,
                path_semantics=MANIFEST_PATH_SEMANTICS_RELATIVE,
            ),
            "cleaned_item_scopes_dir": write_manifest_path_value(
                cleaned_item_scopes_dir,
                manifest_path=run_manifest_path,
                path_semantics=MANIFEST_PATH_SEMANTICS_RELATIVE,
            ),
            "oversize_sections_path": write_manifest_path_value(
                oversize_sections_path,
                manifest_path=run_manifest_path,
                path_semantics=MANIFEST_PATH_SEMANTICS_RELATIVE,
            ),
            "cleaning_row_audit_path": write_manifest_path_value(
                cleaning_row_audit_path,
                manifest_path=run_manifest_path,
                path_semantics=MANIFEST_PATH_SEMANTICS_RELATIVE,
            ),
            "cleaning_flagged_rows_path": write_manifest_path_value(
                cleaning_flagged_rows_path,
                manifest_path=run_manifest_path,
                path_semantics=MANIFEST_PATH_SEMANTICS_RELATIVE,
            ),
            "item_scope_cleaning_diagnostics_path": write_manifest_path_value(
                item_scope_cleaning_diagnostics_path,
                manifest_path=run_manifest_path,
                path_semantics=MANIFEST_PATH_SEMANTICS_RELATIVE,
            ),
            "item_scope_cleaning_diagnostics_csv_path": write_manifest_path_value(
                item_scope_cleaning_diagnostics_csv_path,
                manifest_path=run_manifest_path,
                path_semantics=MANIFEST_PATH_SEMANTICS_RELATIVE,
            ),
            "manual_boundary_audit_sample_path": write_manifest_path_value(
                manual_boundary_audit_sample_path,
                manifest_path=run_manifest_path,
                path_semantics=MANIFEST_PATH_SEMANTICS_RELATIVE,
            ),
            "yearly_summary_path": write_manifest_path_value(
                yearly_summary_path,
                manifest_path=run_manifest_path,
                path_semantics=MANIFEST_PATH_SEMANTICS_RELATIVE,
            ),
            "yearly_summary_csv_path": write_manifest_path_value(
                yearly_summary_csv_path,
                manifest_path=run_manifest_path,
                path_semantics=MANIFEST_PATH_SEMANTICS_RELATIVE,
            ),
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

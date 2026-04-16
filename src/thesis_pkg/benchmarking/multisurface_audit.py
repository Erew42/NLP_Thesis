from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import polars as pl

from thesis_pkg.benchmarking.item_text_cleaning import CLEANED_ITEM_SCOPE_SCHEMA
from thesis_pkg.benchmarking.item_text_cleaning import CLEANING_ROW_AUDIT_SCHEMA
from thesis_pkg.benchmarking.item_text_cleaning import MANUAL_AUDIT_SAMPLE_SCHEMA
from thesis_pkg.benchmarking.item_text_cleaning import SCOPE_DIAGNOSTICS_SCHEMA
from thesis_pkg.benchmarking.manifest_contracts import resolve_manifest_path
from thesis_pkg.benchmarking.manifest_contracts import write_manifest_path_value
from thesis_pkg.core.sec.parquet_stream import iter_parquet_filing_texts


CASE_SOURCE_VALUES: tuple[str, ...] = (
    "sentence_short_fragment",
    "sentence_long_table_like",
    "sentence_control",
    "item_cleaning_flagged",
    "item_doc_hotspot",
    "item_boundary_risk",
)
PRIORITY_BAND_VALUES: tuple[str, ...] = ("high", "medium", "control")
REVIEW_LABEL_VALUES: tuple[str, ...] = (
    "artifact_table_row",
    "artifact_header",
    "artifact_separator",
    "artifact_page_marker",
    "artifact_reference_stub",
    "valid_prose",
    "valid_prose_intro",
    "valid_heading",
    "split_prose_fragment",
    "boundary_error_item",
    "uncertain",
)
ROOT_CAUSE_VALUES: tuple[str, ...] = (
    "item_cleaning",
    "sentence_segmentation",
    "item_boundary_extraction",
    "report_layout_loss",
    "mixed_or_unknown",
)
RECOMMENDED_ACTION_VALUES: tuple[str, ...] = (
    "cleaner_rule",
    "sentence_protect_rule",
    "sentence_stitch_rule",
    "boundary_extraction_followup",
    "no_change",
)
FULL_REPORT_MATCH_STATUS_VALUES: tuple[str, ...] = (
    "not_requested",
    "exact_primary",
    "normalized_primary",
    "item_start_snippet",
    "item_end_snippet",
    "heading_anchor",
    "anchor_fallback",
    "doc_missing",
)

AUDIT_CASE_SCHEMA: dict[str, pl.DataType] = {
    "case_id": pl.Utf8,
    "case_source": pl.Utf8,
    "priority_band": pl.Utf8,
    "doc_id": pl.Utf8,
    "filing_year": pl.Int32,
    "benchmark_row_id": pl.Utf8,
    "benchmark_sentence_id": pl.Utf8,
    "text_scope": pl.Utf8,
    "benchmark_item_code": pl.Utf8,
    "primary_text": pl.Utf8,
    "prev_text": pl.Utf8,
    "next_text": pl.Utf8,
    "item_original_start_snippet": pl.Utf8,
    "item_cleaned_start_snippet": pl.Utf8,
    "item_original_end_snippet": pl.Utf8,
    "item_cleaned_end_snippet": pl.Utf8,
    "full_report_snippet": pl.Utf8,
    "full_report_match_status": pl.Utf8,
    "full_report_needed": pl.Boolean,
    "suspect_reason": pl.Utf8,
    "escalation_reason": pl.Utf8,
    "review_label": pl.Utf8,
    "root_cause": pl.Utf8,
    "recommended_action": pl.Utf8,
    "full_report_changed_decision": pl.Boolean,
    "notes": pl.Utf8,
}

REVIEW_REQUIRED_COLUMNS: tuple[str, ...] = (
    "review_label",
    "root_cause",
    "recommended_action",
)

_SHORT_REFERENCE_STUB_RE = re.compile(r"^\s*(?:SFAS|SAB|FIN|EITF|FASB)\s+No\.?\s*$", re.IGNORECASE)
_SHORT_MARKER_RE = re.compile(
    r"^\s*(?:item\s+\d+[a-z]?\.?|risk\s+factors?\.?|business\.?|index\s+to\s+financial\s+statements)\s*$",
    re.IGNORECASE,
)
_SHORT_NUMERIC_RE = re.compile(r'^\s*[\d().,"%\'`\-]+\s*$')
_TABLE_PHRASE_RE = re.compile(
    r"\b(?:the\s+following\s+table|below\s+table|consolidated\s+statements?|balance\s+sheets?|cash\s+flows?|"
    r"years?\s+ended\s+december|in\s+thousands|in\s+millions|payments\s+due\s+by\s+period)\b",
    re.IGNORECASE,
)
_BOUNDARY_LEAK_RE = re.compile(
    r"\b(?:item\s+1a\b|item\s+1b\b|item\s+2\b|item\s+7a\b|item\s+8\b|signatures?\b|"
    r"index\s+to\s+financial\s+statements\b|legal\s+proceedings\b)\b",
    re.IGNORECASE,
)
_TOC_CLUSTER_RE = re.compile(
    r"(?:item\s+\d+[a-z]?\b.*?){2,}",
    re.IGNORECASE | re.DOTALL,
)
_ITEM_ANCHOR_PATTERNS: dict[str, re.Pattern[str]] = {
    "item_1": re.compile(r"\bitem\s+1\b", re.IGNORECASE),
    "item_1a": re.compile(r"\bitem\s+1a\b", re.IGNORECASE),
    "item_7": re.compile(r"\bitem\s+7\b", re.IGNORECASE),
}


@dataclass(frozen=True)
class MultiSurfaceAuditSources:
    run_manifest_path: Path
    run_dir: Path
    sentence_dataset_dir: Path
    cleaned_item_scopes_dir: Path
    cleaning_row_audit_path: Path
    item_scope_cleaning_diagnostics_path: Path
    manual_boundary_audit_sample_path: Path
    sec_year_merged_dir: Path


@dataclass(frozen=True)
class MultiSurfaceAuditPackConfig:
    sentence_short_case_target: int = 60
    sentence_long_case_target: int = 60
    sentence_control_case_target: int = 60
    item_cleaning_case_target: int = 60
    item_hotspot_case_target: int = 60
    item_boundary_case_target: int = 60
    escalation_cap: int = 120
    chunk_count: int = 18
    random_seed: int = 42
    full_report_window_chars: int = 1500

    @property
    def primary_case_target(self) -> int:
        return (
            self.sentence_short_case_target
            + self.sentence_long_case_target
            + self.sentence_control_case_target
            + self.item_cleaning_case_target
            + self.item_hotspot_case_target
            + self.item_boundary_case_target
        )


@dataclass(frozen=True)
class MultiSurfaceAuditPackArtifacts:
    output_dir: Path
    audit_cases_path: Path
    chunk_dir: Path
    summary_path: Path
    manifest_path: Path
    review_instructions_path: Path
    primary_case_count: int
    escalated_case_count: int
    fetched_full_report_doc_count: int
    requested_full_report_doc_count: int


@dataclass(frozen=True)
class ReviewedAuditSummaryArtifacts:
    output_dir: Path
    reviewed_cases_path: Path
    review_summary_path: Path
    pattern_summary_path: Path
    doc_hotspots_path: Path
    rule_candidates_path: Path
    do_not_touch_patterns_path: Path


@dataclass(frozen=True)
class _NormalizedTextIndex:
    normalized_text: str
    normalized_to_original: tuple[int, ...]


def normalize_by_year_dir(path: Path) -> Path:
    candidate = path.resolve()
    if candidate.name == "by_year":
        by_year_dir = candidate
    else:
        by_year_dir = candidate / "by_year"
    if not by_year_dir.exists():
        raise FileNotFoundError(f"Expected by_year parquet directory at {by_year_dir}")
    parquet_paths = tuple(sorted(by_year_dir.glob("*.parquet")))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found under {by_year_dir}")
    return by_year_dir


def _required_schema(path: Path, required: dict[str, pl.DataType]) -> None:
    schema = pl.scan_parquet(path).collect_schema()
    missing = [name for name in required if name not in schema]
    if missing:
        raise ValueError(f"Parquet artifact {path} is missing required columns: {missing}")


def _read_yearly_parquet_dir(path: Path, required: dict[str, pl.DataType]) -> pl.DataFrame:
    by_year_dir = normalize_by_year_dir(path)
    parquet_paths = tuple(sorted(by_year_dir.glob("*.parquet")))
    _required_schema(parquet_paths[0], required)
    return pl.read_parquet([str(parquet_path) for parquet_path in parquet_paths]).select(list(required))


def _resolve_manifest_artifact_path(
    raw_path: str | Path | None,
    *,
    run_manifest_path: Path,
    path_semantics: str | None,
    label: str,
) -> Path:
    resolved = resolve_manifest_path(
        raw_path,
        manifest_path=run_manifest_path,
        path_semantics=path_semantics,
    )
    if resolved is None:
        raise ValueError(f"Run manifest is missing required artifact path for {label}")
    if not resolved.exists():
        raise FileNotFoundError(f"Resolved artifact path for {label} does not exist: {resolved}")
    return resolved


def _infer_sec_year_merged_dir(run_manifest_path: Path, manifest: dict[str, Any]) -> Path:
    path_semantics = manifest.get("path_semantics")
    candidates: list[Path] = []

    source_items_dir = manifest.get("nonportable_diagnostics", {}).get("source_items_dir")
    if source_items_dir:
        resolved_source_items = resolve_manifest_path(
            source_items_dir,
            manifest_path=run_manifest_path,
            path_semantics=path_semantics,
        )
        if resolved_source_items is not None:
            for ancestor in (resolved_source_items.resolve(), *resolved_source_items.resolve().parents):
                if ancestor.name == "results":
                    candidates.append((ancestor.parent / "year_merged").resolve())

    for ancestor in (run_manifest_path.resolve().parent, *run_manifest_path.resolve().parents):
        if ancestor.name == "results":
            candidates.append((ancestor.parent / "year_merged").resolve())

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not infer the SEC year_merged directory from the preprocessing run manifest. "
        "Pass --sec-year-merged-dir explicitly."
    )


def resolve_multisurface_audit_sources(
    run_manifest_path: Path,
    *,
    sec_year_merged_dir: Path | None = None,
) -> MultiSurfaceAuditSources:
    resolved_manifest = run_manifest_path.resolve()
    manifest = json.loads(resolved_manifest.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts") or {}
    path_semantics = manifest.get("path_semantics")

    sentence_dataset_dir = _resolve_manifest_artifact_path(
        artifacts.get("sentence_dataset_dir"),
        run_manifest_path=resolved_manifest,
        path_semantics=path_semantics,
        label="sentence_dataset_dir",
    )
    cleaned_item_scopes_dir = _resolve_manifest_artifact_path(
        artifacts.get("cleaned_item_scopes_dir"),
        run_manifest_path=resolved_manifest,
        path_semantics=path_semantics,
        label="cleaned_item_scopes_dir",
    )
    cleaning_row_audit_path = _resolve_manifest_artifact_path(
        artifacts.get("cleaning_row_audit_path"),
        run_manifest_path=resolved_manifest,
        path_semantics=path_semantics,
        label="cleaning_row_audit_path",
    )
    item_scope_cleaning_diagnostics_path = _resolve_manifest_artifact_path(
        artifacts.get("item_scope_cleaning_diagnostics_path"),
        run_manifest_path=resolved_manifest,
        path_semantics=path_semantics,
        label="item_scope_cleaning_diagnostics_path",
    )
    manual_boundary_audit_sample_path = _resolve_manifest_artifact_path(
        artifacts.get("manual_boundary_audit_sample_path"),
        run_manifest_path=resolved_manifest,
        path_semantics=path_semantics,
        label="manual_boundary_audit_sample_path",
    )

    resolved_sec_year_merged_dir = (
        sec_year_merged_dir.resolve()
        if sec_year_merged_dir is not None
        else _infer_sec_year_merged_dir(resolved_manifest, manifest)
    )
    if not resolved_sec_year_merged_dir.exists():
        raise FileNotFoundError(f"SEC year_merged directory does not exist: {resolved_sec_year_merged_dir}")

    return MultiSurfaceAuditSources(
        run_manifest_path=resolved_manifest,
        run_dir=resolved_manifest.parent,
        sentence_dataset_dir=sentence_dataset_dir,
        cleaned_item_scopes_dir=cleaned_item_scopes_dir,
        cleaning_row_audit_path=cleaning_row_audit_path,
        item_scope_cleaning_diagnostics_path=item_scope_cleaning_diagnostics_path,
        manual_boundary_audit_sample_path=manual_boundary_audit_sample_path,
        sec_year_merged_dir=resolved_sec_year_merged_dir,
    )


def _align_frame_to_schema(df: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    def _cast_expr(name: str, dtype: pl.DataType) -> pl.Expr:
        if name not in df.columns:
            return pl.lit(None, dtype=dtype).alias(name)
        if dtype == pl.Boolean:
            normalized = pl.col(name).cast(pl.Utf8, strict=False).str.strip_chars().str.to_lowercase()
            return (
                pl.when(normalized.is_in(["true", "1", "yes"]))
                .then(pl.lit(True))
                .when(normalized.is_in(["false", "0", "no"]))
                .then(pl.lit(False))
                .otherwise(pl.lit(None))
                .cast(pl.Boolean, strict=False)
                .alias(name)
            )
        return pl.col(name).cast(dtype, strict=False).alias(name)

    return df.select(
        [
            _cast_expr(name, dtype)
            for name, dtype in schema.items()
        ]
    )


def _stable_int(value: str, seed: int) -> int:
    digits = re.sub(r"\D+", "", value)
    if digits:
        return int(digits[-15:])
    return abs(hash((seed, value)))


def _sanitize_text(value: str | None) -> str:
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def _normalize_spaces(value: str | None) -> str:
    return re.sub(r"\s+", " ", _sanitize_text(value)).strip()


def _sentence_suspect_reason(text: str) -> str:
    normalized = _normalize_spaces(text)
    if _SHORT_REFERENCE_STUB_RE.fullmatch(normalized):
        return "reference_stub"
    if _SHORT_MARKER_RE.fullmatch(normalized):
        return "heading_fragment"
    if _SHORT_NUMERIC_RE.fullmatch(normalized):
        return "numeric_fragment"
    if _TABLE_PHRASE_RE.search(normalized):
        return "table_or_statement_phrase"
    if "\n" in text and ("---" in text or "===" in text):
        return "separator_block"
    if len(normalized) <= 30:
        return "short_fragment"
    return "control_peer"


def _boundary_snippet_risk(*snippets: str | None) -> bool:
    joined = "\n".join(_sanitize_text(snippet) for snippet in snippets if snippet)
    if not joined:
        return False
    return bool(_BOUNDARY_LEAK_RE.search(joined) or _TOC_CLUSTER_RE.search(joined))


def _item_primary_text(row: dict[str, Any]) -> str:
    start_snippet = _sanitize_text(row.get("cleaned_start_snippet"))
    end_snippet = _sanitize_text(row.get("cleaned_end_snippet"))
    if start_snippet and end_snippet and start_snippet != end_snippet:
        return f"[START]\n{start_snippet}\n\n[END]\n{end_snippet}"
    if start_snippet:
        return start_snippet
    if end_snippet:
        return end_snippet
    return _sanitize_text(row.get("cleaned_text"))[:1200]


def _normalize_with_positions(text: str) -> _NormalizedTextIndex:
    normalized_chars: list[str] = []
    normalized_to_original: list[int] = []
    pending_space = False
    started = False

    for index, character in enumerate(text):
        if character.isspace():
            if started:
                pending_space = True
            continue
        if pending_space and normalized_chars:
            normalized_chars.append(" ")
            normalized_to_original.append(index)
        normalized_chars.append(character.casefold())
        normalized_to_original.append(index)
        pending_space = False
        started = True

    return _NormalizedTextIndex(
        normalized_text="".join(normalized_chars),
        normalized_to_original=tuple(normalized_to_original),
    )


def _normalized_match_bounds(index: _NormalizedTextIndex, query: str) -> tuple[int, int] | None:
    normalized_query = _normalize_spaces(query).casefold()
    if len(normalized_query) < 40:
        return None
    start = index.normalized_text.find(normalized_query)
    if start < 0:
        return None
    end = start + len(normalized_query) - 1
    if end >= len(index.normalized_to_original):
        return None
    original_start = index.normalized_to_original[start]
    original_end = index.normalized_to_original[end] + 1
    return original_start, original_end


def _extract_window(text: str, *, start: int, end: int, window_chars: int) -> str:
    bounded_start = max(0, start - window_chars)
    bounded_end = min(len(text), end + window_chars)
    return text[bounded_start:bounded_end].strip()


def _heading_anchor_bounds(full_text: str, benchmark_item_code: str | None) -> tuple[int, int] | None:
    if benchmark_item_code is None:
        return None
    pattern = _ITEM_ANCHOR_PATTERNS.get(str(benchmark_item_code))
    if pattern is None:
        return None
    match = pattern.search(full_text)
    if match is None:
        return None
    return match.start(), match.end()


def _fallback_full_report_snippet(case_row: dict[str, Any]) -> str:
    parts: list[str] = []
    start_snippet = _sanitize_text(case_row.get("item_original_start_snippet") or case_row.get("item_cleaned_start_snippet"))
    end_snippet = _sanitize_text(case_row.get("item_original_end_snippet") or case_row.get("item_cleaned_end_snippet"))
    if start_snippet:
        parts.append(f"[ITEM START]\n{start_snippet}")
    if end_snippet:
        parts.append(f"[ITEM END]\n{end_snippet}")
    if not parts:
        primary = _sanitize_text(case_row.get("primary_text"))
        if primary:
            parts.append(f"[PRIMARY TEXT]\n{primary}")
    return "\n\n".join(parts).strip()


def _resolve_full_report_context(
    case_row: dict[str, Any],
    *,
    full_text: str,
    window_chars: int,
) -> tuple[str, str]:
    primary_text = _sanitize_text(case_row.get("primary_text"))
    if len(primary_text) >= 20:
        raw_index = full_text.find(primary_text)
        if raw_index >= 0:
            return (
                _extract_window(
                    full_text,
                    start=raw_index,
                    end=raw_index + len(primary_text),
                    window_chars=window_chars,
                ),
                "exact_primary",
            )

    normalized_index = _normalize_with_positions(full_text)
    normalized_primary_bounds = _normalized_match_bounds(normalized_index, primary_text)
    if normalized_primary_bounds is not None:
        return (
            _extract_window(
                full_text,
                start=normalized_primary_bounds[0],
                end=normalized_primary_bounds[1],
                window_chars=window_chars,
            ),
            "normalized_primary",
        )

    start_candidates = (
        case_row.get("item_original_start_snippet"),
        case_row.get("item_cleaned_start_snippet"),
    )
    for candidate in start_candidates:
        candidate_text = _sanitize_text(candidate)
        if not candidate_text:
            continue
        raw_index = full_text.find(candidate_text)
        if raw_index >= 0:
            return (
                _extract_window(
                    full_text,
                    start=raw_index,
                    end=raw_index + len(candidate_text),
                    window_chars=window_chars,
                ),
                "item_start_snippet",
            )
        bounds = _normalized_match_bounds(normalized_index, candidate_text)
        if bounds is not None:
            return (
                _extract_window(full_text, start=bounds[0], end=bounds[1], window_chars=window_chars),
                "item_start_snippet",
            )

    end_candidates = (
        case_row.get("item_original_end_snippet"),
        case_row.get("item_cleaned_end_snippet"),
    )
    for candidate in end_candidates:
        candidate_text = _sanitize_text(candidate)
        if not candidate_text:
            continue
        raw_index = full_text.find(candidate_text)
        if raw_index >= 0:
            return (
                _extract_window(
                    full_text,
                    start=raw_index,
                    end=raw_index + len(candidate_text),
                    window_chars=window_chars,
                ),
                "item_end_snippet",
            )
        bounds = _normalized_match_bounds(normalized_index, candidate_text)
        if bounds is not None:
            return (
                _extract_window(full_text, start=bounds[0], end=bounds[1], window_chars=window_chars),
                "item_end_snippet",
            )

    anchor_bounds = _heading_anchor_bounds(full_text, case_row.get("benchmark_item_code"))
    if anchor_bounds is not None:
        return (
            _extract_window(full_text, start=anchor_bounds[0], end=anchor_bounds[1], window_chars=window_chars),
            "heading_anchor",
        )

    return _fallback_full_report_snippet(case_row), "anchor_fallback"


def _sentence_surface(sources: MultiSurfaceAuditSources) -> tuple[pl.DataFrame, pl.DataFrame]:
    sentence_required = {
        "benchmark_sentence_id": pl.Utf8,
        "benchmark_row_id": pl.Utf8,
        "doc_id": pl.Utf8,
        "filing_year": pl.Int32,
        "benchmark_item_code": pl.Utf8,
        "text_scope": pl.Utf8,
        "sentence_index": pl.Int64,
        "sentence_text": pl.Utf8,
        "sentence_char_count": pl.Int64,
        "finbert_token_count_512": pl.Int32,
        "finbert_token_bucket_512": pl.Utf8,
    }
    sentence_df = _read_yearly_parquet_dir(sources.sentence_dataset_dir, sentence_required)
    row_audit_df = pl.read_parquet(sources.cleaning_row_audit_path).select(
        [
            "benchmark_row_id",
            "boundary_authority_status",
            "production_eligible",
            "manual_audit_candidate",
            "manual_audit_reason",
            "tail_truncated",
            "warning_large_removal",
            "original_start_snippet",
            "cleaned_start_snippet",
            "original_end_snippet",
            "cleaned_end_snippet",
        ]
    )
    return sentence_df, row_audit_df


def _sentence_features(sentence_df: pl.DataFrame, row_audit_df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    joined = (
        sentence_df.sort(["benchmark_row_id", "sentence_index"])
        .with_columns(
            [
                pl.col("sentence_text").shift(1).over("benchmark_row_id").alias("prev_text"),
                pl.col("sentence_text").shift(-1).over("benchmark_row_id").alias("next_text"),
            ]
        )
        .join(row_audit_df, on="benchmark_row_id", how="left")
        .with_columns(
            [
                pl.col("sentence_text").str.replace_all(r"\s+", " ").str.strip_chars().alias("_sentence_norm"),
                pl.col("sentence_text").str.count_matches(r"\n").alias("_newline_count"),
                pl.col("sentence_text").str.count_matches(r"\d").alias("_digit_count"),
                pl.col("sentence_text").str.count_matches(r"[-=]{3,}").alias("_separator_run_count"),
                pl.col("sentence_text").str.count_matches(r"\b(?:19|20)\d{2}\b").alias("_year_count"),
            ]
        )
        .with_columns(
            [
                (
                    (pl.col("finbert_token_count_512") <= 6)
                    | (pl.col("sentence_char_count") <= 24)
                    | pl.col("_sentence_norm").str.contains(_SHORT_REFERENCE_STUB_RE.pattern)
                    | pl.col("_sentence_norm").str.contains(_SHORT_MARKER_RE.pattern)
                    | pl.col("_sentence_norm").str.contains(_SHORT_NUMERIC_RE.pattern)
                ).alias("short_candidate"),
                (
                    (pl.col("finbert_token_count_512") >= 160)
                    | (pl.col("sentence_char_count") >= 900)
                    | (
                        pl.col("sentence_text").str.contains(_TABLE_PHRASE_RE.pattern)
                        & (pl.col("_newline_count") >= 2)
                    )
                    | (
                        (pl.col("_separator_run_count") >= 1)
                        & (
                            (pl.col("_digit_count") >= 10)
                            | (pl.col("_year_count") >= 3)
                            | (pl.col("finbert_token_count_512") >= 80)
                        )
                    )
                ).alias("long_candidate"),
                (
                    (pl.col("finbert_token_count_512") <= 6).cast(pl.Int32)
                    + (pl.col("sentence_char_count") <= 24).cast(pl.Int32)
                    + pl.col("_sentence_norm").str.contains(_SHORT_REFERENCE_STUB_RE.pattern).cast(pl.Int32) * 2
                    + pl.col("_sentence_norm").str.contains(_SHORT_MARKER_RE.pattern).cast(pl.Int32)
                    + pl.col("_sentence_norm").str.contains(_SHORT_NUMERIC_RE.pattern).cast(pl.Int32)
                ).alias("short_score"),
                (
                    (pl.col("finbert_token_count_512") >= 160).cast(pl.Int32) * 2
                    + (pl.col("sentence_char_count") >= 900).cast(pl.Int32) * 2
                    + pl.col("sentence_text").str.contains(_TABLE_PHRASE_RE.pattern).cast(pl.Int32)
                    + (pl.col("_separator_run_count") >= 1).cast(pl.Int32)
                    + (pl.col("_year_count") >= 3).cast(pl.Int32)
                    + (pl.col("_digit_count") >= 10).cast(pl.Int32)
                ).alias("long_score"),
            ]
        )
        .drop(["_sentence_norm", "_newline_count", "_digit_count", "_separator_run_count", "_year_count"])
    )

    doc_stats = (
        joined.group_by("doc_id")
        .agg(
            [
                pl.len().alias("sentence_rows"),
                pl.col("short_candidate").sum().alias("short_candidate_rows"),
                pl.col("long_candidate").sum().alias("long_candidate_rows"),
            ]
        )
        .with_columns(
            [
                (pl.col("short_candidate_rows") + pl.col("long_candidate_rows")).alias("suspicious_sentence_rows"),
                (
                    (pl.col("short_candidate_rows") + pl.col("long_candidate_rows"))
                    / pl.col("sentence_rows").clip(lower_bound=1)
                ).alias("doc_suspicious_density"),
            ]
        )
    )

    return joined.join(doc_stats, on="doc_id", how="left"), doc_stats


def _item_surface(
    sources: MultiSurfaceAuditSources,
    doc_stats: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    _required_schema(sources.cleaning_row_audit_path, CLEANING_ROW_AUDIT_SCHEMA)
    _required_schema(sources.manual_boundary_audit_sample_path, MANUAL_AUDIT_SAMPLE_SCHEMA)
    _required_schema(sources.item_scope_cleaning_diagnostics_path, SCOPE_DIAGNOSTICS_SCHEMA)

    row_audit_df = pl.read_parquet(sources.cleaning_row_audit_path)
    manual_boundary_df = pl.read_parquet(sources.manual_boundary_audit_sample_path)
    scope_diag_df = pl.read_parquet(sources.item_scope_cleaning_diagnostics_path)
    cleaned_scopes_df = _read_yearly_parquet_dir(
        sources.cleaned_item_scopes_dir,
        {
            "benchmark_row_id": CLEANED_ITEM_SCOPE_SCHEMA["benchmark_row_id"],
            "cleaned_text": CLEANED_ITEM_SCOPE_SCHEMA["cleaned_text"],
            "cleaned_char_count": CLEANED_ITEM_SCOPE_SCHEMA["cleaned_char_count"],
        },
    )

    manual_lookup = manual_boundary_df.select(
        [
            "benchmark_row_id",
            "audit_period",
            "sample_reason",
            "start_boundary_correct",
            "end_boundary_correct",
            "wrong_item_capture_absent",
            "toc_capture_absent",
            "body_text_nonempty",
        ]
    )
    manual_boundary_ids = manual_boundary_df["benchmark_row_id"].to_list()

    item_df = (
        row_audit_df.join(cleaned_scopes_df, on="benchmark_row_id", how="left")
        .join(doc_stats.select(["doc_id", "doc_suspicious_density", "suspicious_sentence_rows"]), on="doc_id", how="left")
        .join(manual_lookup, on="benchmark_row_id", how="left")
        .with_columns(
            [
                pl.col("benchmark_row_id").is_in(manual_boundary_ids).alias("manual_boundary_sample"),
                pl.struct(
                    [
                        "original_start_snippet",
                        "cleaned_start_snippet",
                        "original_end_snippet",
                        "cleaned_end_snippet",
                    ]
                )
                .map_elements(
                    lambda row: _boundary_snippet_risk(
                        row.get("original_start_snippet"),
                        row.get("cleaned_start_snippet"),
                        row.get("original_end_snippet"),
                        row.get("cleaned_end_snippet"),
                    ),
                    return_dtype=pl.Boolean,
                )
                .alias("boundary_snippet_risk"),
                pl.struct(
                    [
                        "original_start_snippet",
                        "cleaned_start_snippet",
                        "original_end_snippet",
                        "cleaned_end_snippet",
                    ]
                )
                .map_elements(
                    lambda row: (
                        _normalize_spaces(row.get("original_start_snippet"))
                        != _normalize_spaces(row.get("cleaned_start_snippet"))
                    )
                    or (
                        _normalize_spaces(row.get("original_end_snippet"))
                        != _normalize_spaces(row.get("cleaned_end_snippet"))
                    ),
                    return_dtype=pl.Boolean,
                )
                .alias("snippet_delta_risk"),
            ]
        )
        .with_columns(
            [
                (
                    pl.col("manual_audit_candidate").cast(pl.Int32) * 3
                    + (pl.col("removed_char_count") >= 500).cast(pl.Int32) * 2
                    + (pl.col("removal_ratio") >= 0.05).cast(pl.Int32) * 2
                    + pl.col("tail_truncated").cast(pl.Int32) * 2
                    + (pl.col("boundary_authority_status") == "review_needed").cast(pl.Int32)
                ).alias("cleaning_score"),
                (
                    (pl.col("doc_suspicious_density") * 100).fill_null(0.0)
                    + pl.col("suspicious_sentence_rows").fill_null(0)
                ).alias("hotspot_score"),
                (
                    pl.col("manual_boundary_sample").cast(pl.Int32) * 3
                    + (pl.col("boundary_authority_status") == "review_needed").cast(pl.Int32) * 2
                    + pl.col("boundary_snippet_risk").cast(pl.Int32) * 2
                    + pl.col("snippet_delta_risk").cast(pl.Int32)
                    + (pl.col("review_status") == "required_unreviewed").cast(pl.Int32)
                    + pl.col("tail_truncated").cast(pl.Int32)
                ).alias("boundary_score"),
            ]
        )
    )

    return item_df, manual_boundary_df, scope_diag_df


def _with_stable_sort_key(df: pl.DataFrame, *, key_col: str, sort_key_name: str, seed: int) -> pl.DataFrame:
    return df.with_columns(
        pl.col(key_col)
        .map_elements(lambda value: _stable_int(str(value), seed), return_dtype=pl.Int64)
        .alias(sort_key_name)
    )


def _case_base() -> dict[str, Any]:
    return {
        "full_report_snippet": "",
        "full_report_match_status": "not_requested",
        "full_report_needed": False,
        "review_label": None,
        "root_cause": None,
        "recommended_action": None,
        "full_report_changed_decision": None,
        "notes": None,
    }


def _sentence_case(row: dict[str, Any], *, case_source: str, priority_band: str) -> dict[str, Any]:
    suspect_reason = _sentence_suspect_reason(str(row.get("sentence_text") or ""))
    if row.get("boundary_authority_status") == "review_needed":
        suspect_reason = f"{suspect_reason}|row_boundary_review_needed"
    return {
        **_case_base(),
        "case_id": f"{case_source}:{row['benchmark_sentence_id']}",
        "case_source": case_source,
        "priority_band": priority_band,
        "doc_id": row.get("doc_id"),
        "filing_year": row.get("filing_year"),
        "benchmark_row_id": row.get("benchmark_row_id"),
        "benchmark_sentence_id": row.get("benchmark_sentence_id"),
        "text_scope": row.get("text_scope"),
        "benchmark_item_code": row.get("benchmark_item_code"),
        "primary_text": _sanitize_text(row.get("sentence_text")),
        "prev_text": _sanitize_text(row.get("prev_text")),
        "next_text": _sanitize_text(row.get("next_text")),
        "item_original_start_snippet": _sanitize_text(row.get("original_start_snippet")),
        "item_cleaned_start_snippet": _sanitize_text(row.get("cleaned_start_snippet")),
        "item_original_end_snippet": _sanitize_text(row.get("original_end_snippet")),
        "item_cleaned_end_snippet": _sanitize_text(row.get("cleaned_end_snippet")),
        "suspect_reason": suspect_reason,
        "escalation_reason": "",
    }


def _item_case(row: dict[str, Any], *, case_source: str, priority_band: str, suspect_reason: str) -> dict[str, Any]:
    return {
        **_case_base(),
        "case_id": f"{case_source}:{row['benchmark_row_id']}",
        "case_source": case_source,
        "priority_band": priority_band,
        "doc_id": row.get("doc_id"),
        "filing_year": row.get("filing_year"),
        "benchmark_row_id": row.get("benchmark_row_id"),
        "benchmark_sentence_id": None,
        "text_scope": row.get("text_scope"),
        "benchmark_item_code": row.get("benchmark_item_code"),
        "primary_text": _item_primary_text(row),
        "prev_text": "",
        "next_text": "",
        "item_original_start_snippet": _sanitize_text(row.get("original_start_snippet")),
        "item_cleaned_start_snippet": _sanitize_text(row.get("cleaned_start_snippet")),
        "item_original_end_snippet": _sanitize_text(row.get("original_end_snippet")),
        "item_cleaned_end_snippet": _sanitize_text(row.get("cleaned_end_snippet")),
        "suspect_reason": suspect_reason,
        "escalation_reason": "",
    }


def _select_sentence_cases(sentence_df: pl.DataFrame, cfg: MultiSurfaceAuditPackConfig) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    selected_sentence_ids: set[str] = set()

    short_pool = _with_stable_sort_key(
        sentence_df.filter(pl.col("short_candidate")),
        key_col="benchmark_sentence_id",
        sort_key_name="_stable_key",
        seed=cfg.random_seed,
    ).sort(["short_score", "doc_suspicious_density", "_stable_key"], descending=[True, True, False])
    short_selected = short_pool.head(cfg.sentence_short_case_target)
    for row in short_selected.to_dicts():
        selected_sentence_ids.add(str(row["benchmark_sentence_id"]))
        cases.append(_sentence_case(row, case_source="sentence_short_fragment", priority_band="high"))

    long_pool = _with_stable_sort_key(
        sentence_df.filter(~pl.col("benchmark_sentence_id").is_in(list(selected_sentence_ids))).filter(pl.col("long_candidate")),
        key_col="benchmark_sentence_id",
        sort_key_name="_stable_key",
        seed=cfg.random_seed + 1,
    ).sort(["long_score", "doc_suspicious_density", "_stable_key"], descending=[True, True, False])
    long_selected = long_pool.head(cfg.sentence_long_case_target)
    for row in long_selected.to_dicts():
        selected_sentence_ids.add(str(row["benchmark_sentence_id"]))
        cases.append(_sentence_case(row, case_source="sentence_long_table_like", priority_band="high"))

    suspicious_doc_scope_pairs = {
        (str(row["doc_id"]), str(row["text_scope"]))
        for row in pl.concat([short_selected, long_selected], how="vertical_relaxed")
        .select(["doc_id", "text_scope"])
        .to_dicts()
    }

    control_pool = (
        sentence_df.filter(~pl.col("benchmark_sentence_id").is_in(list(selected_sentence_ids)))
        .filter(~pl.col("short_candidate"))
        .filter(~pl.col("long_candidate"))
        .with_columns(
            [
                pl.struct(["doc_id", "text_scope"])
                .map_elements(
                    lambda row: (str(row.get("doc_id")), str(row.get("text_scope"))) in suspicious_doc_scope_pairs,
                    return_dtype=pl.Boolean,
                )
                .alias("_peer_group_hit"),
                (pl.col("finbert_token_count_512") - pl.lit(40)).abs().alias("_median_token_gap"),
                (pl.col("sentence_char_count") - pl.lit(180)).abs().alias("_median_char_gap"),
            ]
        )
    )
    control_pool = _with_stable_sort_key(
        control_pool,
        key_col="benchmark_sentence_id",
        sort_key_name="_stable_key",
        seed=cfg.random_seed + 2,
    ).sort(
        ["_peer_group_hit", "doc_suspicious_density", "_median_token_gap", "_median_char_gap", "_stable_key"],
        descending=[True, True, False, False, False],
    )
    control_selected = control_pool.head(cfg.sentence_control_case_target)
    for row in control_selected.to_dicts():
        selected_sentence_ids.add(str(row["benchmark_sentence_id"]))
        cases.append(_sentence_case(row, case_source="sentence_control", priority_band="control"))

    return cases


def _select_item_cases(
    item_df: pl.DataFrame,
    manual_boundary_df: pl.DataFrame,
    cfg: MultiSurfaceAuditPackConfig,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    selected_row_ids: set[str] = set()
    manual_boundary_ids = set(manual_boundary_df["benchmark_row_id"].to_list())

    cleaning_pool = item_df.filter(
        pl.col("manual_audit_candidate")
        | (pl.col("removed_char_count") >= 500)
        | (pl.col("removal_ratio") >= 0.05)
        | pl.col("tail_truncated")
        | (pl.col("boundary_authority_status") == "review_needed")
    )
    cleaning_pool = _with_stable_sort_key(
        cleaning_pool,
        key_col="benchmark_row_id",
        sort_key_name="_stable_key",
        seed=cfg.random_seed + 10,
    ).sort(["cleaning_score", "hotspot_score", "_stable_key"], descending=[True, True, False])
    cleaning_selected = cleaning_pool.head(cfg.item_cleaning_case_target)
    for row in cleaning_selected.to_dicts():
        selected_row_ids.add(str(row["benchmark_row_id"]))
        reasons: list[str] = ["cleaning_signal"]
        if row.get("manual_audit_candidate"):
            reasons.append("manual_audit_candidate")
        if row.get("removed_char_count", 0) >= 500:
            reasons.append("large_removed_block")
        if row.get("boundary_authority_status") == "review_needed":
            reasons.append("row_boundary_review_needed")
        cases.append(
            _item_case(
                row,
                case_source="item_cleaning_flagged",
                priority_band="high",
                suspect_reason="|".join(reasons),
            )
        )

    hotspot_pool = (
        item_df.filter(~pl.col("benchmark_row_id").is_in(list(selected_row_ids)))
        .filter(pl.col("suspicious_sentence_rows").fill_null(0) > 0)
    )
    hotspot_pool = _with_stable_sort_key(
        hotspot_pool,
        key_col="benchmark_row_id",
        sort_key_name="_stable_key",
        seed=cfg.random_seed + 11,
    ).sort(["hotspot_score", "cleaning_score", "_stable_key"], descending=[True, True, False])
    hotspot_selected = hotspot_pool.head(cfg.item_hotspot_case_target)
    for row in hotspot_selected.to_dicts():
        selected_row_ids.add(str(row["benchmark_row_id"]))
        reasons = ["doc_hotspot", f"suspicious_sentence_rows={int(row.get('suspicious_sentence_rows') or 0)}"]
        cases.append(
            _item_case(
                row,
                case_source="item_doc_hotspot",
                priority_band="medium",
                suspect_reason="|".join(reasons),
            )
        )

    boundary_pool = item_df.filter(
        ~pl.col("benchmark_row_id").is_in(list(selected_row_ids))
    ).filter(
        pl.col("manual_boundary_sample")
        | pl.col("boundary_snippet_risk")
        | pl.col("snippet_delta_risk")
        | pl.col("tail_truncated")
        | (pl.col("boundary_authority_status") == "review_needed")
        | (pl.col("review_status") == "required_unreviewed")
        | pl.col("benchmark_row_id").is_in(list(manual_boundary_ids))
    )
    boundary_pool = _with_stable_sort_key(
        boundary_pool,
        key_col="benchmark_row_id",
        sort_key_name="_stable_key",
        seed=cfg.random_seed + 12,
    ).sort(["boundary_score", "cleaning_score", "_stable_key"], descending=[True, True, False])
    boundary_selected = boundary_pool.head(cfg.item_boundary_case_target)
    for row in boundary_selected.to_dicts():
        selected_row_ids.add(str(row["benchmark_row_id"]))
        reasons = ["boundary_risk"]
        if row.get("manual_boundary_sample"):
            reasons.append("manual_boundary_sample")
        if row.get("boundary_snippet_risk"):
            reasons.append("boundary_snippet_leak")
        if row.get("snippet_delta_risk"):
            reasons.append("snippet_delta")
        if row.get("review_status") == "required_unreviewed":
            reasons.append("required_unreviewed")
        if row.get("tail_truncated"):
            reasons.append("tail_truncated")
        cases.append(
            _item_case(
                row,
                case_source="item_boundary_risk",
                priority_band="high",
                suspect_reason="|".join(reasons),
            )
        )

    return cases


def _validate_cfg(cfg: MultiSurfaceAuditPackConfig) -> None:
    integer_fields = (
        cfg.sentence_short_case_target,
        cfg.sentence_long_case_target,
        cfg.sentence_control_case_target,
        cfg.item_cleaning_case_target,
        cfg.item_hotspot_case_target,
        cfg.item_boundary_case_target,
        cfg.escalation_cap,
        cfg.chunk_count,
        cfg.full_report_window_chars,
    )
    if any(value <= 0 for value in integer_fields):
        raise ValueError("All audit-pack counts and window sizes must be positive.")


def _mark_escalated_cases(cases: list[dict[str, Any]], *, cap: int) -> list[dict[str, Any]]:
    scored_cases: list[tuple[int, str, dict[str, Any], list[str]]] = []
    for case in cases:
        score = 0
        reasons: list[str] = []
        case_source = str(case["case_source"])
        suspect_reason = str(case["suspect_reason"] or "")
        text_scope = str(case["text_scope"] or "")

        if case_source == "item_boundary_risk":
            score += 5
            reasons.append("boundary_context_required")
        if "row_boundary_review_needed" in suspect_reason or "manual_boundary_sample" in suspect_reason:
            score += 4
            reasons.append("item_sentence_disagreement")
        if case_source == "sentence_long_table_like":
            score += 4
            reasons.append("table_vs_prose_ambiguity")
        if case_source == "sentence_short_fragment" and (
            "reference_stub" in suspect_reason or "heading_fragment" in suspect_reason
        ):
            score += 3
            reasons.append("split_or_heading_context")
        if case_source == "item_doc_hotspot":
            score += 3
            reasons.append("doc_hotspot_context")
        if case_source == "item_cleaning_flagged":
            score += 2
            reasons.append("cleaning_context")
        if text_scope == "item_7_mda":
            score += 1
            reasons.append("item7_context")

        scored_cases.append((score, str(case["case_id"]), case, reasons))

    scored_cases.sort(key=lambda item: (-item[0], item[1]))
    escalate_case_ids = {
        case_id
        for score, case_id, _, _ in scored_cases
        if score > 0
    }
    if len(escalate_case_ids) > cap:
        escalate_case_ids = {case_id for _, case_id, _, _ in scored_cases[:cap]}

    enriched: list[dict[str, Any]] = []
    for _, case_id, case, reasons in scored_cases:
        updated = dict(case)
        if case_id in escalate_case_ids:
            updated["full_report_needed"] = True
            updated["escalation_reason"] = "|".join(dict.fromkeys(reasons)) if reasons else "context_review"
        enriched.append(updated)
    return enriched


def _apply_full_report_context(
    cases: list[dict[str, Any]],
    *,
    sec_year_merged_dir: Path,
    window_chars: int,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    escalated_doc_ids = {
        str(case["doc_id"])
        for case in cases
        if case.get("full_report_needed") and case.get("doc_id")
    }
    full_reports_by_doc = {
        row["doc_id"]: row["full_text"]
        for row in iter_parquet_filing_texts(sec_year_merged_dir, escalated_doc_ids)
    }

    enriched_cases: list[dict[str, Any]] = []
    for case in cases:
        updated = dict(case)
        if not updated.get("full_report_needed"):
            enriched_cases.append(updated)
            continue
        doc_id = str(updated["doc_id"])
        full_text = full_reports_by_doc.get(doc_id)
        if full_text is None:
            updated["full_report_snippet"] = ""
            updated["full_report_match_status"] = "doc_missing"
        else:
            snippet, match_status = _resolve_full_report_context(
                updated,
                full_text=full_text,
                window_chars=window_chars,
            )
            updated["full_report_snippet"] = snippet
            updated["full_report_match_status"] = match_status
        enriched_cases.append(updated)

    return enriched_cases, full_reports_by_doc


def _cases_to_frame(cases: list[dict[str, Any]]) -> pl.DataFrame:
    if not cases:
        return pl.DataFrame(schema=AUDIT_CASE_SCHEMA)
    return _align_frame_to_schema(pl.DataFrame(cases), AUDIT_CASE_SCHEMA)


def validate_audit_cases(audit_cases_df: pl.DataFrame) -> None:
    if audit_cases_df.is_empty():
        raise ValueError("Audit case frame is empty.")
    duplicate_case_ids = (
        audit_cases_df.group_by("case_id")
        .len()
        .filter(pl.col("len") > 1)
        .get_column("case_id")
        .to_list()
    )
    if duplicate_case_ids:
        raise ValueError(f"Audit cases contain duplicate case_id values: {duplicate_case_ids[:5]}")

    required_non_null = ("case_id", "case_source", "doc_id", "primary_text")
    for column in required_non_null:
        null_count = audit_cases_df.filter(pl.col(column).is_null() | (pl.col(column) == "")).height
        if null_count:
            raise ValueError(f"Audit cases contain {null_count} rows with null or empty {column}.")

    invalid_case_source = sorted(set(audit_cases_df["case_source"].drop_nulls().to_list()) - set(CASE_SOURCE_VALUES))
    if invalid_case_source:
        raise ValueError(f"Audit cases contain invalid case_source values: {invalid_case_source}")

    invalid_priority = sorted(set(audit_cases_df["priority_band"].drop_nulls().to_list()) - set(PRIORITY_BAND_VALUES))
    if invalid_priority:
        raise ValueError(f"Audit cases contain invalid priority_band values: {invalid_priority}")

    escalated_missing_status = audit_cases_df.filter(
        pl.col("full_report_needed") & pl.col("full_report_match_status").is_null()
    ).height
    if escalated_missing_status:
        raise ValueError("Escalated audit cases are missing full_report_match_status values.")


def _interleave_bucketed_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, bool], list[dict[str, Any]]] = {}
    for record in records:
        key = (
            str(record["case_source"]),
            str(record["text_scope"] or ""),
            bool(record["full_report_needed"]),
        )
        buckets.setdefault(key, []).append(record)

    ordered_keys = sorted(buckets)
    interleaved: list[dict[str, Any]] = []
    while any(buckets.values()):
        for key in ordered_keys:
            bucket = buckets[key]
            if bucket:
                interleaved.append(bucket.pop(0))
    return interleaved


def _write_review_instructions(path: Path) -> None:
    text = (
        "# Multi-Surface FinBERT Audit Review Instructions\n\n"
        "1. Review `primary_text`, `prev_text`, `next_text`, and the item start/end snippets first.\n"
        "2. Fill `review_label`, `root_cause`, and `recommended_action` using only the allowed categorical values.\n"
        "3. When `full_report_needed=true`, inspect `full_report_snippet` and revise the final label if needed.\n"
        "4. For escalated rows, set `full_report_changed_decision` to `true` or `false` explicitly.\n"
        "5. Use `notes` for concise evidence and always cite the local `case_id` if you mention neighboring rows.\n\n"
        f"Allowed `review_label` values: {', '.join(REVIEW_LABEL_VALUES)}\n\n"
        f"Allowed `root_cause` values: {', '.join(ROOT_CAUSE_VALUES)}\n\n"
        f"Allowed `recommended_action` values: {', '.join(RECOMMENDED_ACTION_VALUES)}\n"
    )
    path.write_text(text, encoding="utf-8")


def _chunk_records(records: list[dict[str, Any]], *, chunk_count: int) -> list[list[dict[str, Any]]]:
    ordered_records = _interleave_bucketed_records(records)
    chunks: list[list[dict[str, Any]]] = [[] for _ in range(chunk_count)]
    for index, record in enumerate(ordered_records):
        chunks[index % chunk_count].append(record)
    return chunks


def build_multisurface_audit_pack(
    sources: MultiSurfaceAuditSources,
    *,
    output_dir: Path,
    cfg: MultiSurfaceAuditPackConfig | None = None,
) -> MultiSurfaceAuditPackArtifacts:
    cfg = cfg or MultiSurfaceAuditPackConfig()
    _validate_cfg(cfg)

    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir = output_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    review_instructions_path = output_dir / "review_instructions.md"
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "manifest.json"
    audit_cases_path = output_dir / "audit_cases.parquet"

    sentence_df_raw, row_audit_df = _sentence_surface(sources)
    sentence_df, doc_stats = _sentence_features(sentence_df_raw, row_audit_df)
    item_df, manual_boundary_df, scope_diag_df = _item_surface(sources, doc_stats)

    sentence_cases = _select_sentence_cases(sentence_df, cfg)
    item_cases = _select_item_cases(item_df, manual_boundary_df, cfg)
    cases = _mark_escalated_cases(sentence_cases + item_cases, cap=cfg.escalation_cap)
    cases, full_reports_by_doc = _apply_full_report_context(
        cases,
        sec_year_merged_dir=sources.sec_year_merged_dir,
        window_chars=cfg.full_report_window_chars,
    )
    cases_df = _cases_to_frame(cases).sort(["case_source", "priority_band", "case_id"])
    validate_audit_cases(cases_df)

    cases_df.write_parquet(audit_cases_path, compression="zstd")
    cases_df.write_csv(output_dir / "audit_cases.csv")

    _write_review_instructions(review_instructions_path)
    chunk_records = _chunk_records(cases_df.to_dicts(), chunk_count=cfg.chunk_count)
    chunk_manifest_rows: list[dict[str, Any]] = []
    for chunk_index, records in enumerate(chunk_records, start=1):
        chunk_id = f"chunk_{chunk_index:02d}"
        chunk_path = chunk_dir / f"{chunk_id}.csv"
        chunk_df = _cases_to_frame(records).sort(["case_source", "case_id"])
        chunk_df.write_csv(chunk_path)
        chunk_manifest_rows.append(
            {
                "chunk_id": chunk_id,
                "path": write_manifest_path_value(
                    chunk_path,
                    manifest_path=manifest_path,
                    path_semantics="manifest_relative_v1",
                ),
                "expected_row_count": int(chunk_df.height),
                "case_ids": chunk_df["case_id"].to_list(),
                "case_sources": sorted(set(chunk_df["case_source"].drop_nulls().to_list())),
                "text_scopes": sorted(set(chunk_df["text_scope"].drop_nulls().to_list())),
                "escalated_case_count": int(chunk_df.filter(pl.col("full_report_needed")).height),
                "priority_bands": sorted(set(chunk_df["priority_band"].drop_nulls().to_list())),
            }
        )

    requested_full_report_doc_count = len(
        {
            doc_id
            for doc_id in cases_df.filter(pl.col("full_report_needed")).get_column("doc_id").to_list()
            if doc_id is not None
        }
    )
    fetched_full_report_doc_count = len(full_reports_by_doc)
    build_summary = {
        "sources": {
            "run_manifest_path": str(sources.run_manifest_path),
            "sentence_dataset_dir": str(sources.sentence_dataset_dir),
            "cleaned_item_scopes_dir": str(sources.cleaned_item_scopes_dir),
            "cleaning_row_audit_path": str(sources.cleaning_row_audit_path),
            "item_scope_cleaning_diagnostics_path": str(sources.item_scope_cleaning_diagnostics_path),
            "manual_boundary_audit_sample_path": str(sources.manual_boundary_audit_sample_path),
            "sec_year_merged_dir": str(sources.sec_year_merged_dir),
        },
        "config": asdict(cfg),
        "counts": {
            "primary_case_count": int(cases_df.height),
            "sentence_case_count": int(cases_df.filter(pl.col("case_source").str.starts_with("sentence_")).height),
            "item_case_count": int(cases_df.filter(pl.col("case_source").str.starts_with("item_")).height),
            "control_case_count": int(cases_df.filter(pl.col("case_source") == "sentence_control").height),
            "escalated_case_count": int(cases_df.filter(pl.col("full_report_needed")).height),
            "doc_count": int(cases_df["doc_id"].n_unique()),
            "requested_full_report_doc_count": requested_full_report_doc_count,
            "fetched_full_report_doc_count": fetched_full_report_doc_count,
            "full_report_fetch_coverage": (
                fetched_full_report_doc_count / requested_full_report_doc_count
                if requested_full_report_doc_count
                else 1.0
            ),
        },
        "by_case_source": cases_df.group_by("case_source").len().sort("case_source").to_dicts(),
        "by_text_scope": cases_df.group_by("text_scope").len().sort("text_scope").to_dicts(),
        "scope_diagnostics_rows": scope_diag_df.height,
        "manual_boundary_rows": manual_boundary_df.height,
        "chunk_count": len(chunk_manifest_rows),
    }
    summary_path.write_text(json.dumps(build_summary, indent=2, sort_keys=True), encoding="utf-8")

    manifest_payload = {
        "schema_columns": list(AUDIT_CASE_SCHEMA),
        "allowed_values": {
            "case_source": list(CASE_SOURCE_VALUES),
            "priority_band": list(PRIORITY_BAND_VALUES),
            "review_label": list(REVIEW_LABEL_VALUES),
            "root_cause": list(ROOT_CAUSE_VALUES),
            "recommended_action": list(RECOMMENDED_ACTION_VALUES),
            "full_report_match_status": list(FULL_REPORT_MATCH_STATUS_VALUES),
        },
        "summary_path": write_manifest_path_value(
            summary_path,
            manifest_path=manifest_path,
            path_semantics="manifest_relative_v1",
        ),
        "audit_cases_path": write_manifest_path_value(
            audit_cases_path,
            manifest_path=manifest_path,
            path_semantics="manifest_relative_v1",
        ),
        "review_instructions_path": write_manifest_path_value(
            review_instructions_path,
            manifest_path=manifest_path,
            path_semantics="manifest_relative_v1",
        ),
        "chunks": chunk_manifest_rows,
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8")

    return MultiSurfaceAuditPackArtifacts(
        output_dir=output_dir,
        audit_cases_path=audit_cases_path,
        chunk_dir=chunk_dir,
        summary_path=summary_path,
        manifest_path=manifest_path,
        review_instructions_path=review_instructions_path,
        primary_case_count=int(cases_df.height),
        escalated_case_count=int(cases_df.filter(pl.col("full_report_needed")).height),
        fetched_full_report_doc_count=fetched_full_report_doc_count,
        requested_full_report_doc_count=requested_full_report_doc_count,
    )


def _reviewed_chunk_paths(
    audit_pack_dir: Path,
    *,
    reviewed_chunks_dir: Path | None = None,
) -> tuple[Path, ...]:
    if reviewed_chunks_dir is not None:
        base_dir = reviewed_chunks_dir.resolve()
    else:
        preferred = audit_pack_dir / "reviewed_chunks"
        base_dir = preferred if preferred.exists() else (audit_pack_dir / "chunks")
    if not base_dir.exists():
        raise FileNotFoundError(f"Reviewed chunk directory does not exist: {base_dir}")
    chunk_paths = tuple(sorted(base_dir.glob("chunk_*.csv")))
    if not chunk_paths:
        raise FileNotFoundError(f"No reviewed chunk CSVs found under {base_dir}")
    return chunk_paths


def _manifest_chunk_lookup(audit_pack_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    manifest_path = audit_pack_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunk_lookup = {str(chunk["chunk_id"]): chunk for chunk in manifest.get("chunks", [])}
    if not chunk_lookup:
        raise ValueError(f"Audit pack manifest does not define any chunk metadata: {manifest_path}")
    return chunk_lookup, manifest


def _validate_reviewed_frame(df: pl.DataFrame, *, chunk_id: str) -> None:
    missing_columns = [column for column in AUDIT_CASE_SCHEMA if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Reviewed chunk {chunk_id} is missing required columns: {missing_columns}")

    invalid_review_labels = sorted(set(df["review_label"].drop_nulls().to_list()) - set(REVIEW_LABEL_VALUES))
    if invalid_review_labels:
        raise ValueError(f"Reviewed chunk {chunk_id} has invalid review_label values: {invalid_review_labels}")

    invalid_root_causes = sorted(set(df["root_cause"].drop_nulls().to_list()) - set(ROOT_CAUSE_VALUES))
    if invalid_root_causes:
        raise ValueError(f"Reviewed chunk {chunk_id} has invalid root_cause values: {invalid_root_causes}")

    invalid_actions = sorted(set(df["recommended_action"].drop_nulls().to_list()) - set(RECOMMENDED_ACTION_VALUES))
    if invalid_actions:
        raise ValueError(f"Reviewed chunk {chunk_id} has invalid recommended_action values: {invalid_actions}")

    missing_required_labels = df.filter(
        pl.any_horizontal([pl.col(column).is_null() | (pl.col(column) == "") for column in REVIEW_REQUIRED_COLUMNS])
    ).height
    if missing_required_labels:
        raise ValueError(
            f"Reviewed chunk {chunk_id} has {missing_required_labels} rows with incomplete reviewer labels."
        )

    missing_change_decision = df.filter(
        pl.col("full_report_needed") & pl.col("full_report_changed_decision").is_null()
    ).height
    if missing_change_decision:
        raise ValueError(
            f"Reviewed chunk {chunk_id} has {missing_change_decision} escalated rows without full_report_changed_decision."
        )


def summarize_reviewed_audit_pack(
    audit_pack_dir: Path,
    *,
    reviewed_chunks_dir: Path | None = None,
    output_dir: Path | None = None,
) -> ReviewedAuditSummaryArtifacts:
    resolved_audit_pack_dir = audit_pack_dir.resolve()
    out_dir = (output_dir.resolve() if output_dir is not None else resolved_audit_pack_dir / "review_summary")
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_lookup, manifest = _manifest_chunk_lookup(resolved_audit_pack_dir)
    reviewed_paths = _reviewed_chunk_paths(resolved_audit_pack_dir, reviewed_chunks_dir=reviewed_chunks_dir)

    reviewed_frames: list[pl.DataFrame] = []
    seen_case_ids: set[str] = set()
    completed_chunk_ids: set[str] = set()
    for reviewed_path in reviewed_paths:
        chunk_id = reviewed_path.stem
        if chunk_id not in chunk_lookup:
            raise ValueError(f"Reviewed chunk {chunk_id} is not present in the audit-pack manifest.")
        chunk_manifest = chunk_lookup[chunk_id]
        reviewed_df = _align_frame_to_schema(pl.read_csv(reviewed_path), AUDIT_CASE_SCHEMA)
        if reviewed_df.height != int(chunk_manifest["expected_row_count"]):
            raise ValueError(
                f"Reviewed chunk {chunk_id} row count {reviewed_df.height} does not match "
                f"manifest expectation {chunk_manifest['expected_row_count']}."
            )
        expected_case_ids = set(chunk_manifest["case_ids"])
        actual_case_ids = set(reviewed_df["case_id"].to_list())
        if actual_case_ids != expected_case_ids:
            raise ValueError(f"Reviewed chunk {chunk_id} case_id set does not match manifest.")
        overlap = seen_case_ids & actual_case_ids
        if overlap:
            raise ValueError(f"Reviewed chunks contain duplicate case_id values: {sorted(overlap)[:5]}")
        seen_case_ids.update(actual_case_ids)
        _validate_reviewed_frame(reviewed_df, chunk_id=chunk_id)
        reviewed_frames.append(reviewed_df)
        completed_chunk_ids.add(chunk_id)

    missing_chunk_ids = sorted(set(chunk_lookup) - completed_chunk_ids)
    if missing_chunk_ids:
        raise ValueError(f"Reviewed chunk set is incomplete. Missing chunks: {missing_chunk_ids}")

    reviewed_df = pl.concat(reviewed_frames, how="vertical_relaxed").sort("case_id")
    reviewed_cases_path = out_dir / "reviewed_audit_cases.parquet"
    reviewed_df.write_parquet(reviewed_cases_path, compression="zstd")
    reviewed_df.write_csv(out_dir / "reviewed_audit_cases.csv")

    pattern_summary = (
        reviewed_df.group_by(
            ["review_label", "root_cause", "recommended_action", "case_source", "suspect_reason", "text_scope"]
        )
        .agg(
            [
                pl.len().alias("case_count"),
                pl.col("doc_id").n_unique().alias("doc_count"),
                pl.col("full_report_needed").sum().alias("escalated_case_count"),
                pl.col("full_report_changed_decision").cast(pl.Float64).mean().alias("changed_decision_rate"),
            ]
        )
        .sort(["case_count", "doc_count", "review_label"], descending=[True, True, False])
    )
    pattern_summary_path = out_dir / "pattern_summary.csv"
    pattern_summary.write_csv(pattern_summary_path)

    doc_hotspots = (
        reviewed_df.group_by("doc_id")
        .agg(
            [
                pl.len().alias("case_count"),
                pl.col("review_label").n_unique().alias("review_label_count"),
                pl.col("full_report_needed").sum().alias("escalated_case_count"),
                pl.col("full_report_changed_decision").cast(pl.Float64).mean().alias("changed_decision_rate"),
                pl.col("suspect_reason").sort().first().alias("example_suspect_reason"),
            ]
        )
        .sort(["changed_decision_rate", "case_count", "review_label_count"], descending=[True, True, True])
    )
    doc_hotspots_path = out_dir / "doc_hotspots.csv"
    doc_hotspots.write_csv(doc_hotspots_path)

    rule_candidates = (
        pattern_summary.filter(pl.col("recommended_action") != "no_change")
        .sort(["case_count", "doc_count", "changed_decision_rate"], descending=[True, True, False])
        .head(10)
    )
    rule_candidates_path = out_dir / "rule_candidates.csv"
    rule_candidates.write_csv(rule_candidates_path)

    do_not_touch_patterns = (
        pattern_summary.filter(
            (pl.col("recommended_action") == "no_change")
            | pl.col("review_label").is_in(["valid_prose", "valid_prose_intro", "valid_heading"])
        )
        .sort(["case_count", "doc_count", "changed_decision_rate"], descending=[True, True, False])
        .head(10)
    )
    do_not_touch_patterns_path = out_dir / "do_not_touch_patterns.csv"
    do_not_touch_patterns.write_csv(do_not_touch_patterns_path)

    counts_by_review_label = reviewed_df.group_by("review_label").len().sort("review_label").to_dicts()
    counts_by_root_cause = reviewed_df.group_by("root_cause").len().sort("root_cause").to_dicts()
    counts_by_action = reviewed_df.group_by("recommended_action").len().sort("recommended_action").to_dicts()
    escalated_df = reviewed_df.filter(pl.col("full_report_needed"))

    review_summary = {
        "audit_pack_dir": str(resolved_audit_pack_dir),
        "reviewed_chunk_dir": str((reviewed_chunks_dir or (resolved_audit_pack_dir / "reviewed_chunks")).resolve())
        if reviewed_chunks_dir is not None or (resolved_audit_pack_dir / "reviewed_chunks").exists()
        else str((resolved_audit_pack_dir / "chunks").resolve()),
        "counts": {
            "reviewed_case_count": int(reviewed_df.height),
            "reviewed_doc_count": int(reviewed_df["doc_id"].n_unique()),
            "escalated_case_count": int(escalated_df.height),
            "full_report_changed_decision_rate": (
                float(escalated_df["full_report_changed_decision"].cast(pl.Float64).mean())
                if escalated_df.height
                else 0.0
            ),
        },
        "counts_by_review_label": counts_by_review_label,
        "counts_by_root_cause": counts_by_root_cause,
        "counts_by_recommended_action": counts_by_action,
        "top_rule_candidates": rule_candidates.to_dicts(),
        "top_do_not_touch_patterns": do_not_touch_patterns.to_dicts(),
        "manifest_allowed_values": manifest.get("allowed_values", {}),
    }
    review_summary_path = out_dir / "review_summary.json"
    review_summary_path.write_text(json.dumps(review_summary, indent=2, sort_keys=True, default=str), encoding="utf-8")

    return ReviewedAuditSummaryArtifacts(
        output_dir=out_dir,
        reviewed_cases_path=reviewed_cases_path,
        review_summary_path=review_summary_path,
        pattern_summary_path=pattern_summary_path,
        doc_hotspots_path=doc_hotspots_path,
        rule_candidates_path=rule_candidates_path,
        do_not_touch_patterns_path=do_not_touch_patterns_path,
    )


__all__ = [
    "AUDIT_CASE_SCHEMA",
    "CASE_SOURCE_VALUES",
    "FULL_REPORT_MATCH_STATUS_VALUES",
    "MultiSurfaceAuditPackArtifacts",
    "MultiSurfaceAuditPackConfig",
    "MultiSurfaceAuditSources",
    "RECOMMENDED_ACTION_VALUES",
    "REVIEW_LABEL_VALUES",
    "ROOT_CAUSE_VALUES",
    "ReviewedAuditSummaryArtifacts",
    "build_multisurface_audit_pack",
    "resolve_multisurface_audit_sources",
    "summarize_reviewed_audit_pack",
    "validate_audit_cases",
]

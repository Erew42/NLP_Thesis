from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Any

import polars as pl

from thesis_pkg.benchmarking.contracts import DEFAULT_FINBERT_AUTHORITY
from thesis_pkg.benchmarking.contracts import FinbertAuthoritySpec
from thesis_pkg.benchmarking.contracts import ItemTextCleaningConfig
from thesis_pkg.benchmarking.contracts import SentenceDatasetConfig
from thesis_pkg.core.sec.lm2011_text import tokenize_lm2011_text


ITEM_SCOPE_BY_BENCHMARK_ITEM_CODE: dict[str, str] = {
    "item_7": "item_7_mda",
    "item_1a": "item_1a_risk_factors",
    "item_1": "item_1_business",
}
PRIMARY_ACTIVATION_TEXT_SCOPES = frozenset({"item_7_mda", "item_1a_risk_factors"})
ROBUSTNESS_ONLY_TEXT_SCOPES = frozenset({"item_1_business"})
DEFAULT_MANUAL_AUDIT_SAMPLE_ROWS_PER_SCOPE_PERIOD = 5

_PAGE_MARKER_RE = re.compile(
    r"""^\s*
    (?:
        (?:page\s+)?[-(\[]?\d{1,4}(?:\s+of\s+\d{1,4})?[)\]-]?
        |
        -\s*\d{1,4}\s*-
    )
    \s*$""",
    re.IGNORECASE | re.VERBOSE,
)
_ITEM_HEADING_RE = re.compile(r"^\s*item\s+\d+[a-z]?\b", re.IGNORECASE)
_RUNNING_HEADER_RE = re.compile(
    r"^\s*(?:annual\s+)?report\s+on\s+form\s+10\s*[- ]?k\s*$",
    re.IGNORECASE,
)
_STRUCTURAL_TAG_RE = re.compile(r"^\s*<[^>]{1,120}>\s*$")
_ATTACHMENT_FILENAME_RE = re.compile(r"^\s*[A-Za-z0-9_.-]+\.(?:xml|xsd|xbrl|htm|html)\s*$", re.IGNORECASE)
_TOC_LIKE_LINE_RE = re.compile(
    r"^\s*(?:part\s+[ivxlcdm]+\s*)?item\s+\d+[a-z]?\b.*(?:\.{2,}|\t|\s{2,}|\s+\d{1,4}\s*)$",
    re.IGNORECASE,
)
_NUMERIC_TOKEN_RE = re.compile(r"(?<![A-Za-z])(?:\$?\(?\d[\d,]*\.?\d*%?\)?)(?![A-Za-z])")
_WORD_TOKEN_RE = re.compile(r"[A-Za-z]{2,}")
_REFERENCE_ONLY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bincorporated\s+by\s+reference\b", re.IGNORECASE),
    re.compile(r"^\s*reference\s+is\s+made\s+to\b", re.IGNORECASE),
    re.compile(r"\bthe\s+information\s+required\s+by\s+(?:this\s+)?item\b", re.IGNORECASE),
)
_TAIL_MARKERS_BY_SCOPE: dict[str, tuple[str, ...]] = {
    "item_7_mda": (
        r"item\s+7a\b",
        r"item\s+8\b",
        r"part\s+iii\b",
        r"signatures?\b",
        r"index\s+to\s+exhibits\b",
        r"exhibit\s+index\b",
    ),
    "item_1a_risk_factors": (
        r"item\s+1b\b",
        r"item\s+2\b",
        r"part\s+ii\b",
        r"signatures?\b",
        r"index\s+to\s+exhibits\b",
        r"exhibit\s+index\b",
    ),
    "item_1_business": (
        r"item\s+1a\b",
        r"item\s+1b\b",
        r"item\s+2\b",
        r"part\s+ii\b",
        r"signatures?\b",
        r"index\s+to\s+exhibits\b",
        r"exhibit\s+index\b",
    ),
}

CLEANED_ITEM_SCOPE_SCHEMA: dict[str, pl.DataType] = {
    "doc_id": pl.Utf8,
    "cik_10": pl.Utf8,
    "accession_nodash": pl.Utf8,
    "filing_date": pl.Date,
    "filing_year": pl.Int32,
    "calendar_year": pl.Int32,
    "benchmark_row_id": pl.Utf8,
    "benchmark_item_code": pl.Utf8,
    "benchmark_item_label": pl.Utf8,
    "text_scope": pl.Utf8,
    "item_id": pl.Utf8,
    "canonical_item": pl.Utf8,
    "document_type": pl.Utf8,
    "document_type_raw": pl.Utf8,
    "document_type_normalized": pl.Utf8,
    "source_year_file": pl.Int32,
    "source_record_id": pl.Utf8,
    "source_file_row_nr": pl.Int64,
    "cleaning_policy_id": pl.Utf8,
    "original_char_count": pl.Int32,
    "cleaned_char_count": pl.Int32,
    "removed_char_count": pl.Int32,
    "removal_ratio": pl.Float64,
    "cleaned_lm_total_token_count": pl.Int32,
    "cleaned_text": pl.Utf8,
    "dropped_after_cleaning": pl.Boolean,
    "drop_reason": pl.Utf8,
    "segment_policy_id": pl.Utf8,
}

CLEANING_ROW_AUDIT_SCHEMA: dict[str, pl.DataType] = {
    **{name: dtype for name, dtype in CLEANED_ITEM_SCOPE_SCHEMA.items() if name != "cleaned_text"},
    "page_marker_lines_removed": pl.Int32,
    "report_header_footer_lines_removed": pl.Int32,
    "structural_tag_lines_removed": pl.Int32,
    "table_like_lines_removed": pl.Int32,
    "toc_prefix_trimmed": pl.Boolean,
    "toc_prefix_trimmed_char_count": pl.Int32,
    "tail_truncated": pl.Boolean,
    "tail_truncated_char_count": pl.Int32,
    "reference_only_stub": pl.Boolean,
    "effectively_non_body_text": pl.Boolean,
    "warning_large_removal": pl.Boolean,
    "warning_below_clean_char_count": pl.Boolean,
    "item7_lm_token_floor_failed": pl.Boolean,
    "manual_audit_candidate": pl.Boolean,
    "manual_audit_reason": pl.Utf8,
    "original_start_snippet": pl.Utf8,
    "cleaned_start_snippet": pl.Utf8,
    "original_end_snippet": pl.Utf8,
    "cleaned_end_snippet": pl.Utf8,
}

SCOPE_DIAGNOSTICS_SCHEMA: dict[str, pl.DataType] = {
    "calendar_year": pl.Int32,
    "text_scope": pl.Utf8,
    "n_filings_candidate": pl.Int64,
    "n_filings_extracted": pl.Int64,
    "extraction_rate": pl.Float64,
    "n_rows_after_cleaning": pl.Int64,
    "token_count_mean": pl.Float64,
    "token_count_median": pl.Float64,
    "token_count_p05": pl.Float64,
    "toc_trimmed_rows": pl.Int64,
    "toc_leakage_rate_proxy": pl.Float64,
    "tail_truncated_rows": pl.Int64,
    "reference_stub_rows": pl.Int64,
    "empty_after_cleaning_rows": pl.Int64,
    "large_removal_warning_rows": pl.Int64,
    "manual_audit_queue_n": pl.Int64,
    "manual_audit_pass_rate": pl.Float64,
    "activation_status": pl.Utf8,
}

MANUAL_AUDIT_SAMPLE_SCHEMA: dict[str, pl.DataType] = {
    "doc_id": pl.Utf8,
    "filing_date": pl.Date,
    "calendar_year": pl.Int32,
    "audit_period": pl.Utf8,
    "text_scope": pl.Utf8,
    "benchmark_row_id": pl.Utf8,
    "cleaning_policy_id": pl.Utf8,
    "original_start_snippet": pl.Utf8,
    "cleaned_start_snippet": pl.Utf8,
    "original_end_snippet": pl.Utf8,
    "cleaned_end_snippet": pl.Utf8,
    "sample_reason": pl.Utf8,
    "dropped_after_cleaning": pl.Boolean,
    "warning_large_removal": pl.Boolean,
    "toc_prefix_trimmed": pl.Boolean,
    "tail_truncated": pl.Boolean,
    "reference_only_stub": pl.Boolean,
    "item7_lm_token_floor_failed": pl.Boolean,
    "start_boundary_correct": pl.Boolean,
    "end_boundary_correct": pl.Boolean,
    "wrong_item_capture_absent": pl.Boolean,
    "toc_capture_absent": pl.Boolean,
    "body_text_nonempty": pl.Boolean,
}


@dataclass(frozen=True)
class CleanedTextResult:
    cleaned_text: str
    page_marker_lines_removed: int
    report_header_footer_lines_removed: int
    structural_tag_lines_removed: int
    table_like_lines_removed: int
    toc_prefix_trimmed: bool
    toc_prefix_trimmed_char_count: int
    tail_truncated: bool
    tail_truncated_char_count: int
    reference_only_stub: bool
    effectively_non_body_text: bool


@dataclass(frozen=True)
class ItemTextCleaningResult:
    cleaned_scope_df: pl.DataFrame
    row_audit_df: pl.DataFrame
    flagged_rows_df: pl.DataFrame
    scope_diagnostics_df: pl.DataFrame
    manual_audit_sample_df: pl.DataFrame


def benchmark_item_code_to_text_scope(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().casefold().replace("-", "_")
    return ITEM_SCOPE_BY_BENCHMARK_ITEM_CODE.get(normalized, normalized or None)


def build_segment_policy_id(
    sentence_cfg: SentenceDatasetConfig,
    cleaning_cfg: ItemTextCleaningConfig,
    authority: FinbertAuthoritySpec = DEFAULT_FINBERT_AUTHORITY,
    *,
    chunk_char_limit: int = 250_000,
) -> str:
    cleaning_policy = cleaning_cfg.cleaning_policy_id if cleaning_cfg.enabled else "raw_item_text"
    return (
        f"{sentence_cfg.sentencizer_backend}"
        f"__clean_{cleaning_policy}"
        "__heading_none"
        f"__maxpos{authority.token_count_max_length}"
        f"__chunk{chunk_char_limit}"
        "__lenw_mean_v1"
    )


def _empty_frame(schema: dict[str, pl.DataType]) -> pl.DataFrame:
    return pl.DataFrame(schema=schema)


def _align_to_schema(df: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
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


def _normalize_newlines(text: str | None) -> str:
    if text is None:
        return ""
    return str(text).replace("\r\n", "\n").replace("\r", "\n")


def _collapse_blank_runs(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _snippet_start(text: str, *, length: int = 500) -> str:
    return text[:length]


def _snippet_end(text: str, *, length: int = 500) -> str:
    return text[-length:] if len(text) > length else text


def _is_page_marker_line(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped and _PAGE_MARKER_RE.match(stripped) and not _ITEM_HEADING_RE.match(stripped))


def _is_report_header_footer_line(line: str) -> bool:
    return bool(_RUNNING_HEADER_RE.match(line.strip()))


def _is_structural_residue_line(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped and (_STRUCTURAL_TAG_RE.match(stripped) or _ATTACHMENT_FILENAME_RE.match(stripped)))


def _is_table_like_line(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) < 25:
        return False
    numeric_count = len(_NUMERIC_TOKEN_RE.findall(stripped))
    word_count = len(_WORD_TOKEN_RE.findall(stripped))
    return numeric_count >= 4 and (word_count <= 3 or numeric_count >= max(4, word_count))


def _is_toc_like_line(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped and _TOC_LIKE_LINE_RE.match(stripped))


def _scope_body_hint_re(text_scope: str) -> re.Pattern[str]:
    if text_scope == "item_7_mda":
        return re.compile(r"\b(?:item\s+7\b|management\W*s?\s+discussion|results\s+of\s+operations)\b", re.I)
    if text_scope == "item_1a_risk_factors":
        return re.compile(r"\b(?:item\s+1a\b|risk\s+factors?)\b", re.I)
    if text_scope == "item_1_business":
        return re.compile(r"\b(?:item\s+1\b|business)\b", re.I)
    return re.compile(r"\w+")


def _trim_early_toc_prefix(
    text: str,
    text_scope: str,
    cfg: ItemTextCleaningConfig,
) -> tuple[str, bool, int]:
    if not cfg.trim_early_toc_prefix or not text or cfg.toc_scan_char_window <= 0:
        return text, False, 0

    window_end = min(len(text), cfg.toc_scan_char_window)
    offset = 0
    toc_offsets: list[int] = []
    for raw_line in text[:window_end].splitlines(keepends=True):
        line_end = offset + len(raw_line)
        offset = line_end
        if _is_toc_like_line(raw_line.strip()):
            toc_offsets.append(line_end)

    if len(toc_offsets) < cfg.toc_min_matching_lines:
        return text, False, 0

    trim_end = max(toc_offsets)
    remainder = text[trim_end:].lstrip()
    if len(tokenize_lm2011_text(remainder)) < 5:
        return text, False, 0
    if not _scope_body_hint_re(text_scope).search(remainder[: min(len(remainder), cfg.toc_scan_char_window)]):
        return text, False, 0
    return remainder, True, len(text) - len(remainder)


def _line_remove(
    text: str,
    cfg: ItemTextCleaningConfig,
    *,
    remove_layout_lines: bool,
    remove_table_like_lines: bool,
) -> tuple[str, Counter[str]]:
    counts: Counter[str] = Counter()
    kept_lines: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if remove_layout_lines and cfg.drop_page_markers and _is_page_marker_line(stripped):
            counts["page_marker"] += 1
            continue
        if remove_layout_lines and cfg.drop_report_headers and _is_report_header_footer_line(stripped):
            counts["report_header_footer"] += 1
            continue
        if remove_layout_lines and cfg.drop_structural_tags and _is_structural_residue_line(stripped):
            counts["structural_tag"] += 1
            continue
        if remove_table_like_lines and cfg.drop_table_like_lines and _is_table_like_line(stripped):
            counts["table_like"] += 1
            continue
        kept_lines.append(line.rstrip())
    return "\n".join(kept_lines), counts


def _tail_marker_re(text_scope: str) -> re.Pattern[str] | None:
    markers = _TAIL_MARKERS_BY_SCOPE.get(text_scope)
    if not markers:
        return None
    marker = "|".join(f"(?:{pattern})" for pattern in markers)
    return re.compile(rf"(?im)^\s*(?:{marker})[.\s:-]*")


def _truncate_tail_bleed(
    text: str,
    text_scope: str,
    cfg: ItemTextCleaningConfig,
) -> tuple[str, bool, int]:
    if not cfg.truncate_item_aware_tail_bleed or not text:
        return text, False, 0
    marker_re = _tail_marker_re(text_scope)
    if marker_re is None:
        return text, False, 0

    ordinary_tail_start = int(len(text) * max(0.0, 1.0 - cfg.tail_scan_fraction))
    strong_tail_start = int(len(text) * 0.50)
    best_start: int | None = None
    for match in marker_re.finditer(text):
        marker_text = match.group(0).casefold()
        is_strong_terminal = (
            "signature" in marker_text
            or "index to exhibits" in marker_text
            or "exhibit index" in marker_text
        )
        min_start = strong_tail_start if is_strong_terminal else ordinary_tail_start
        if match.start() < min_start:
            continue
        if best_start is None or match.start() < best_start:
            best_start = match.start()

    if best_start is None:
        return text, False, 0
    truncated = text[:best_start].rstrip()
    return truncated, True, len(text) - len(truncated)


def _is_reference_only_stub(text: str, cfg: ItemTextCleaningConfig) -> bool:
    if not cfg.drop_reference_only_stubs:
        return False
    stripped = text.strip()
    if not stripped or len(stripped) > cfg.reference_stub_max_char_count:
        return False
    return any(pattern.search(stripped) for pattern in _REFERENCE_ONLY_PATTERNS)


def _is_effectively_non_body_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return len(tokenize_lm2011_text(stripped)) == 0


def clean_item_text(
    text: str | None,
    text_scope: str,
    cfg: ItemTextCleaningConfig = ItemTextCleaningConfig(),
) -> CleanedTextResult:
    normalized = _normalize_newlines(text)
    if not cfg.enabled:
        cleaned = _collapse_blank_runs(normalized)
        return CleanedTextResult(
            cleaned_text=cleaned,
            page_marker_lines_removed=0,
            report_header_footer_lines_removed=0,
            structural_tag_lines_removed=0,
            table_like_lines_removed=0,
            toc_prefix_trimmed=False,
            toc_prefix_trimmed_char_count=0,
            tail_truncated=False,
            tail_truncated_char_count=0,
            reference_only_stub=False,
            effectively_non_body_text=_is_effectively_non_body_text(cleaned),
        )

    without_lines, line_counts = _line_remove(
        normalized,
        cfg,
        remove_layout_lines=True,
        remove_table_like_lines=False,
    )
    without_toc, toc_trimmed, toc_removed = _trim_early_toc_prefix(without_lines, text_scope, cfg)
    without_tail, tail_truncated, tail_removed = _truncate_tail_bleed(without_toc, text_scope, cfg)
    reference_only_stub = _is_reference_only_stub(_collapse_blank_runs(without_tail), cfg)
    without_tables, table_counts = _line_remove(
        without_tail,
        cfg,
        remove_layout_lines=False,
        remove_table_like_lines=True,
    )
    cleaned = _collapse_blank_runs(without_tables)
    return CleanedTextResult(
        cleaned_text=cleaned,
        page_marker_lines_removed=int(line_counts["page_marker"]),
        report_header_footer_lines_removed=int(line_counts["report_header_footer"]),
        structural_tag_lines_removed=int(line_counts["structural_tag"]),
        table_like_lines_removed=int(table_counts["table_like"]),
        toc_prefix_trimmed=toc_trimmed,
        toc_prefix_trimmed_char_count=int(toc_removed),
        tail_truncated=tail_truncated,
        tail_truncated_char_count=int(tail_removed),
        reference_only_stub=reference_only_stub,
        effectively_non_body_text=_is_effectively_non_body_text(cleaned),
    )


def _drop_reason(
    *,
    cleaned_text: str,
    text_scope: str,
    cleaning_result: CleanedTextResult,
    cleaned_lm_total_token_count: int,
    cfg: ItemTextCleaningConfig,
) -> str | None:
    if not cfg.enabled:
        return None
    if cfg.drop_blank_after_cleaning and not cleaned_text.strip():
        return "blank_after_cleaning"
    if cleaning_result.reference_only_stub:
        return "reference_only_stub"
    if cleaning_result.effectively_non_body_text:
        return "non_body_text"
    if cfg.hard_drop_min_clean_char_count is not None and len(cleaned_text) < cfg.hard_drop_min_clean_char_count:
        return "below_min_clean_char_count"
    if (
        text_scope == "item_7_mda"
        and cfg.enforce_item7_lm_token_floor
        and cleaned_lm_total_token_count < cfg.item7_min_lm_tokens
    ):
        return "item7_below_lm_token_floor"
    return None


def _manual_audit_reason(row: dict[str, Any]) -> str | None:
    reasons: list[str] = []
    if row["dropped_after_cleaning"]:
        reasons.append(str(row["drop_reason"] or "dropped"))
    if row["warning_large_removal"]:
        reasons.append("large_removal")
    if row["toc_prefix_trimmed"]:
        reasons.append("toc_prefix_trimmed")
    if row["tail_truncated"]:
        reasons.append("tail_truncated")
    if row["reference_only_stub"]:
        reasons.append("reference_only_stub")
    if row["item7_lm_token_floor_failed"]:
        reasons.append("item7_lm_token_floor_failed")
    if row["warning_below_clean_char_count"]:
        reasons.append("below_clean_char_warning")
    return "|".join(dict.fromkeys(reasons)) if reasons else None


def _base_row_payload(row: dict[str, Any], *, text_scope: str) -> dict[str, Any]:
    return {
        "doc_id": row.get("doc_id"),
        "cik_10": row.get("cik_10"),
        "accession_nodash": row.get("accession_nodash"),
        "filing_date": row.get("filing_date"),
        "filing_year": row.get("filing_year"),
        "calendar_year": row.get("filing_year"),
        "benchmark_row_id": row.get("benchmark_row_id"),
        "benchmark_item_code": row.get("benchmark_item_code"),
        "benchmark_item_label": row.get("benchmark_item_label"),
        "text_scope": text_scope,
        "item_id": row.get("item_id"),
        "canonical_item": row.get("canonical_item"),
        "document_type": row.get("document_type"),
        "document_type_raw": row.get("document_type_raw"),
        "document_type_normalized": row.get("document_type_normalized"),
        "source_year_file": row.get("source_year_file"),
        "source_record_id": row.get("source_record_id"),
        "source_file_row_nr": row.get("source_file_row_nr"),
    }


def clean_item_scopes_with_audit(
    sections_df: pl.DataFrame,
    cfg: ItemTextCleaningConfig = ItemTextCleaningConfig(),
    *,
    segment_policy_id: str | None = None,
) -> ItemTextCleaningResult:
    if sections_df.is_empty():
        empty_audit = _empty_frame(CLEANING_ROW_AUDIT_SCHEMA)
        return ItemTextCleaningResult(
            cleaned_scope_df=_empty_frame(CLEANED_ITEM_SCOPE_SCHEMA),
            row_audit_df=empty_audit,
            flagged_rows_df=empty_audit,
            scope_diagnostics_df=_empty_frame(SCOPE_DIAGNOSTICS_SCHEMA),
            manual_audit_sample_df=_empty_frame(MANUAL_AUDIT_SAMPLE_SCHEMA),
        )

    segment_policy = segment_policy_id or "unknown_segment_policy"
    cleaned_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for row in sections_df.iter_rows(named=True):
        benchmark_item_code = row.get("benchmark_item_code")
        text_scope = benchmark_item_code_to_text_scope(benchmark_item_code) or str(benchmark_item_code or "")
        original_text = _normalize_newlines(row.get("full_text"))
        cleaning_result = clean_item_text(original_text, text_scope, cfg)
        cleaned_text = cleaning_result.cleaned_text
        original_char_count = len(original_text)
        cleaned_char_count = len(cleaned_text)
        removed_char_count = max(original_char_count - cleaned_char_count, 0)
        removal_ratio = (removed_char_count / original_char_count) if original_char_count else 0.0
        cleaned_lm_total_token_count = len(tokenize_lm2011_text(cleaned_text))
        reason = _drop_reason(
            cleaned_text=cleaned_text,
            text_scope=text_scope,
            cleaning_result=cleaning_result,
            cleaned_lm_total_token_count=cleaned_lm_total_token_count,
            cfg=cfg,
        )
        dropped = reason is not None
        warning_large_removal = bool(
            original_char_count
            and removal_ratio >= cfg.large_removal_warning_threshold
            and cleaned_char_count < original_char_count
        )
        warning_below_clean_char_count = bool(
            not dropped
            and cleaned_char_count < cfg.warn_below_clean_char_count
        )
        item7_lm_token_floor_failed = bool(
            text_scope == "item_7_mda"
            and cfg.enforce_item7_lm_token_floor
            and cleaned_lm_total_token_count < cfg.item7_min_lm_tokens
        )

        common_payload = {
            **_base_row_payload(row, text_scope=text_scope),
            "cleaning_policy_id": cfg.cleaning_policy_id if cfg.enabled else "raw_item_text",
            "original_char_count": original_char_count,
            "cleaned_char_count": cleaned_char_count,
            "removed_char_count": removed_char_count,
            "removal_ratio": removal_ratio,
            "cleaned_lm_total_token_count": cleaned_lm_total_token_count,
            "dropped_after_cleaning": dropped,
            "drop_reason": reason,
            "segment_policy_id": segment_policy,
        }
        audit_payload = {
            **common_payload,
            "page_marker_lines_removed": cleaning_result.page_marker_lines_removed,
            "report_header_footer_lines_removed": cleaning_result.report_header_footer_lines_removed,
            "structural_tag_lines_removed": cleaning_result.structural_tag_lines_removed,
            "table_like_lines_removed": cleaning_result.table_like_lines_removed,
            "toc_prefix_trimmed": cleaning_result.toc_prefix_trimmed,
            "toc_prefix_trimmed_char_count": cleaning_result.toc_prefix_trimmed_char_count,
            "tail_truncated": cleaning_result.tail_truncated,
            "tail_truncated_char_count": cleaning_result.tail_truncated_char_count,
            "reference_only_stub": cleaning_result.reference_only_stub,
            "effectively_non_body_text": cleaning_result.effectively_non_body_text,
            "warning_large_removal": warning_large_removal,
            "warning_below_clean_char_count": warning_below_clean_char_count,
            "item7_lm_token_floor_failed": item7_lm_token_floor_failed,
            "manual_audit_candidate": False,
            "manual_audit_reason": None,
            "original_start_snippet": _snippet_start(original_text),
            "cleaned_start_snippet": _snippet_start(cleaned_text),
            "original_end_snippet": _snippet_end(original_text),
            "cleaned_end_snippet": _snippet_end(cleaned_text),
        }
        audit_payload["manual_audit_reason"] = _manual_audit_reason(audit_payload)
        audit_payload["manual_audit_candidate"] = audit_payload["manual_audit_reason"] is not None
        audit_rows.append(audit_payload)
        if not dropped:
            cleaned_rows.append({**common_payload, "cleaned_text": cleaned_text})

    row_audit_df = _align_to_schema(pl.DataFrame(audit_rows), CLEANING_ROW_AUDIT_SCHEMA)
    cleaned_scope_df = (
        _align_to_schema(pl.DataFrame(cleaned_rows), CLEANED_ITEM_SCOPE_SCHEMA)
        if cleaned_rows
        else _empty_frame(CLEANED_ITEM_SCOPE_SCHEMA)
    )
    flagged_rows_df = _flagged_rows(row_audit_df)
    diagnostics_df = _build_scope_diagnostics(row_audit_df)
    manual_audit_df = _build_manual_audit_sample(row_audit_df)
    return ItemTextCleaningResult(
        cleaned_scope_df=cleaned_scope_df,
        row_audit_df=row_audit_df,
        flagged_rows_df=flagged_rows_df,
        scope_diagnostics_df=diagnostics_df,
        manual_audit_sample_df=manual_audit_df,
    )


def cleaned_scopes_for_sentence_materialization(cleaned_scope_df: pl.DataFrame) -> pl.DataFrame:
    if cleaned_scope_df.is_empty():
        return cleaned_scope_df.with_columns(
            pl.lit(None, dtype=pl.Utf8).alias("full_text"),
            pl.lit(None, dtype=pl.Int32).alias("char_count"),
        )
    return cleaned_scope_df.with_columns(
        pl.col("cleaned_text").alias("full_text"),
        pl.col("cleaned_char_count").cast(pl.Int32, strict=False).alias("char_count"),
    )


def _flagged_rows(row_audit_df: pl.DataFrame) -> pl.DataFrame:
    if row_audit_df.is_empty():
        return _empty_frame(CLEANING_ROW_AUDIT_SCHEMA)
    return row_audit_df.filter(
        pl.any_horizontal(
            [
                pl.col("dropped_after_cleaning"),
                pl.col("warning_large_removal"),
                pl.col("toc_prefix_trimmed"),
                pl.col("tail_truncated"),
                pl.col("reference_only_stub"),
                pl.col("item7_lm_token_floor_failed"),
                pl.col("warning_below_clean_char_count"),
            ]
        )
    )


def _activation_status(text_scope: str) -> str:
    if text_scope in PRIMARY_ACTIVATION_TEXT_SCOPES:
        return "blocked_pending_manual_audit"
    if text_scope in ROBUSTNESS_ONLY_TEXT_SCOPES:
        return "robustness_only_pending_manual_audit"
    return "diagnostic_only"


def _build_scope_diagnostics(row_audit_df: pl.DataFrame) -> pl.DataFrame:
    if row_audit_df.is_empty():
        return _empty_frame(SCOPE_DIAGNOSTICS_SCHEMA)

    diagnostics = (
        row_audit_df.group_by(["calendar_year", "text_scope"])
        .agg(
            [
                pl.col("doc_id").n_unique().cast(pl.Int64).alias("n_filings_candidate"),
                pl.col("doc_id").n_unique().cast(pl.Int64).alias("n_filings_extracted"),
                (~pl.col("dropped_after_cleaning")).cast(pl.Int64).sum().alias("n_rows_after_cleaning"),
                pl.col("cleaned_lm_total_token_count")
                .filter(~pl.col("dropped_after_cleaning"))
                .mean()
                .alias("token_count_mean"),
                pl.col("cleaned_lm_total_token_count")
                .filter(~pl.col("dropped_after_cleaning"))
                .median()
                .alias("token_count_median"),
                pl.col("cleaned_lm_total_token_count")
                .filter(~pl.col("dropped_after_cleaning"))
                .quantile(0.05)
                .alias("token_count_p05"),
                pl.col("toc_prefix_trimmed").cast(pl.Int64).sum().alias("toc_trimmed_rows"),
                pl.col("tail_truncated").cast(pl.Int64).sum().alias("tail_truncated_rows"),
                pl.col("reference_only_stub").cast(pl.Int64).sum().alias("reference_stub_rows"),
                (pl.col("drop_reason") == "blank_after_cleaning").cast(pl.Int64).sum().alias("empty_after_cleaning_rows"),
                pl.col("warning_large_removal").cast(pl.Int64).sum().alias("large_removal_warning_rows"),
                pl.col("manual_audit_candidate").cast(pl.Int64).sum().alias("manual_audit_queue_n"),
            ]
        )
        .with_columns(
            [
                (
                    pl.col("toc_trimmed_rows").cast(pl.Float64)
                    / pl.max_horizontal(pl.col("n_filings_candidate").cast(pl.Float64), pl.lit(1.0))
                ).alias("toc_leakage_rate_proxy"),
                (
                    pl.col("n_filings_extracted").cast(pl.Float64)
                    / pl.max_horizontal(pl.col("n_filings_candidate").cast(pl.Float64), pl.lit(1.0))
                ).alias("extraction_rate"),
                pl.lit(None, dtype=pl.Float64).alias("manual_audit_pass_rate"),
                pl.col("text_scope")
                .map_elements(_activation_status, return_dtype=pl.Utf8)
                .alias("activation_status"),
            ]
        )
    )
    return _align_to_schema(diagnostics, SCOPE_DIAGNOSTICS_SCHEMA).sort(["calendar_year", "text_scope"])


def _audit_period(calendar_year: int | None) -> str:
    if calendar_year is None:
        return "unknown"
    if calendar_year <= 2008:
        return "pre_2009"
    if calendar_year <= 2016:
        return "2009_2016"
    return "2017_2024"


def _sample_reason(row: dict[str, Any]) -> str:
    reason = row.get("manual_audit_reason")
    if reason:
        return str(reason)
    return "background_scope_period_sample"


def _build_manual_audit_sample(row_audit_df: pl.DataFrame) -> pl.DataFrame:
    if row_audit_df.is_empty():
        return _empty_frame(MANUAL_AUDIT_SAMPLE_SCHEMA)

    rows = (
        row_audit_df.with_columns(
            [
                pl.col("calendar_year")
                .map_elements(_audit_period, return_dtype=pl.Utf8)
                .alias("audit_period"),
                pl.when(pl.col("manual_audit_candidate"))
                .then(pl.lit(0))
                .otherwise(pl.lit(1))
                .alias("_sample_priority"),
            ]
        )
        .sort(["text_scope", "audit_period", "_sample_priority", "calendar_year", "doc_id", "benchmark_row_id"])
        .to_dicts()
    )

    selected: list[dict[str, Any]] = []
    counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        key = (str(row["text_scope"]), str(row["audit_period"]))
        if counts[key] >= DEFAULT_MANUAL_AUDIT_SAMPLE_ROWS_PER_SCOPE_PERIOD:
            continue
        counts[key] += 1
        selected.append(
            {
                "doc_id": row["doc_id"],
                "filing_date": row["filing_date"],
                "calendar_year": row["calendar_year"],
                "audit_period": row["audit_period"],
                "text_scope": row["text_scope"],
                "benchmark_row_id": row["benchmark_row_id"],
                "cleaning_policy_id": row["cleaning_policy_id"],
                "original_start_snippet": row["original_start_snippet"],
                "cleaned_start_snippet": row["cleaned_start_snippet"],
                "original_end_snippet": row["original_end_snippet"],
                "cleaned_end_snippet": row["cleaned_end_snippet"],
                "sample_reason": _sample_reason(row),
                "dropped_after_cleaning": row["dropped_after_cleaning"],
                "warning_large_removal": row["warning_large_removal"],
                "toc_prefix_trimmed": row["toc_prefix_trimmed"],
                "tail_truncated": row["tail_truncated"],
                "reference_only_stub": row["reference_only_stub"],
                "item7_lm_token_floor_failed": row["item7_lm_token_floor_failed"],
                "start_boundary_correct": None,
                "end_boundary_correct": None,
                "wrong_item_capture_absent": None,
                "toc_capture_absent": None,
                "body_text_nonempty": None,
            }
        )

    if not selected:
        return _empty_frame(MANUAL_AUDIT_SAMPLE_SCHEMA)
    return _align_to_schema(pl.DataFrame(selected), MANUAL_AUDIT_SAMPLE_SCHEMA)

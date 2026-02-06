from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from .extraction import (
    extract_filing_items,
    parse_header,
)
from .extraction_utils import EmbeddedHeadingHit
from .heuristics import (
    _looks_like_toc_heading_line,
    _prefix_is_part_only,
    _prefix_looks_like_cross_ref,
    _evaluate_regime_validity,
)
from .patterns import (
    EMBEDDED_CONTINUATION_PATTERN,
    EMBEDDED_CROSS_REF_PATTERN,
    EMBEDDED_ITEM_PATTERN,
    EMBEDDED_ITEM_ROMAN_PATTERN,
    EMBEDDED_PART_PATTERN,
    EMBEDDED_RESERVED_PATTERN,
    EMBEDDED_SEPARATOR_PATTERN,
    EMBEDDED_TOC_DOT_LEADER_PATTERN,
    EMBEDDED_TOC_HEADER_PATTERN,
    EMBEDDED_TOC_ITEM_ONLY_PATTERN,
    EMBEDDED_TOC_PART_ITEM_PATTERN,
    EMBEDDED_TOC_TRAILING_PAGE_PATTERN,
    EMBEDDED_TOC_WINDOW_HEADER_PATTERN,
    ITEM_LINESTART_PATTERN,
    TOC_DOT_LEADER_PATTERN,
)
from .utilities import (
    _default_part_for_item_id,
    _normalize_newlines,
    _prefix_is_bullet,
)
from .embedded_headings import (
    EMBEDDED_FAIL_CLASSIFICATIONS,
    EMBEDDED_IGNORE_CLASSIFICATIONS,
    EMBEDDED_MAX_HITS,
    EMBEDDED_NEARBY_ITEM_WINDOW,
    EMBEDDED_SELF_HIT_MAX_CHAR,
    EMBEDDED_TOC_CLUSTER_LOOKAHEAD,
    EMBEDDED_TOC_START_EARLY_MAX_CHAR,
    EMBEDDED_TOC_START_MISFIRE_MAX_CHAR,
    EMBEDDED_TOC_WINDOW_LINES,
    EMBEDDED_WARN_CLASSIFICATIONS,
    _confirm_part_restart,
    _confirm_prose_after,
    _cross_ref_like,
    _find_embedded_heading_hits,
    _glued_title_marker_item_id,
    _is_late_item,
    _item_id_from_match,
    _item_id_to_int,
    _leading_ws_len,
    _line_snippet,
    _line_starts_item,
    _non_empty_line,
    _normalize_item_id,
    _normalize_part,
    _prose_like_line,
    _sentence_like_line,
    _summarize_embedded_hits,
    _toc_candidate_line,
    _toc_cluster_after,
    _toc_clustered,
    _toc_entry_like,
    _toc_header_nearby,
    _toc_index_style_line,
    _toc_like_line,
    _toc_window_flags,
)
from .regime import get_regime_index, normalize_form_type
from .html_audit import (
    _parse_bool,
    _parse_int,
    _safe_slug,
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_WARNING,
    classify_filing_status,
    normalize_extractor_body,
    normalize_sample_weights,
    sample_filings_by_status,
    write_html_audit,
    write_html_audit_root_index,
)
from .parquet_stream import iter_parquet_filing_texts


ITEM_MENTION_PATTERN = re.compile(r"\bITEM\s+(?:\d+|[IVXLCDM]+)[A-Z]?\b", re.IGNORECASE)
INTERNAL_PART_PATTERN = re.compile(r"(?m)^[ \t]*PART[ \t]+[IVX]+\b", re.IGNORECASE)
INTERNAL_ITEM_PATTERN = re.compile(r"(?m)^[ \t]*ITEM[ \t]+\d+[A-Z]?\b", re.IGNORECASE)
INTERNAL_HEADING_IGNORE_CHARS = 200
INTERNAL_HEADING_CONTEXT_CHARS = 150
PREFIX_KIND_BLANK = "blank"
PREFIX_KIND_PART_ONLY = "part_only"
PREFIX_KIND_BULLET = "bullet"
PREFIX_KIND_TEXTUAL = "textual"
EMBEDDED_WARN_CLUSTER_MIN = 2
MISSING_PART_BUCKETS = (
    "combined_heading_part_item",
    "form_header_midline_part",
    "part_marker_line_only",
    "item_only_no_part",
    "other_unclear",
)
MISSING_PART_PART_PATTERN = re.compile(r"\bPART\s+(I|II)\b", re.IGNORECASE)
MISSING_PART_ITEM_PATTERN = re.compile(r"\bITEM\b", re.IGNORECASE)
MISSING_PART_FORM_HEADER_PATTERN = re.compile(
    r"\bFORM\s+10-Q\b|\bQUARTERLY\s+REPORT\b", re.IGNORECASE
)
MISSING_PART_PART_ITEM_PUNCT_PATTERN = re.compile(
    r"\bPART\b[^\n]{0,80}[,\-][^\n]{0,80}\bITEM\b"
    r"|\bITEM\b[^\n]{0,80}[,\-][^\n]{0,80}\bPART\b",
    re.IGNORECASE,
)

DEFAULT_PARQUET_DIR = Path(
    r"C:\Users\erik9\Documents\SEC_Data\Data\Sample_Filings\parquet_batches"
)
DEFAULT_OUT_PATH = Path("results/suspicious_boundaries_v3_pre.csv")
DEFAULT_REPORT_PATH = Path("results/suspicious_boundaries_report_v3_pre.txt")
DEFAULT_SAMPLES_DIR = Path("results/Suspicious_Filings_Demo")
DEFAULT_CSV_PATH = Path("results/suspicious_boundaries_v5.csv")
DEFAULT_ROOT_CAUSES_PATH = Path("results/suspicious_root_causes_examples.txt")
DEFAULT_MANIFEST_ITEMS_PATH = Path("results/extraction_manifest_items.csv")
DEFAULT_MANIFEST_FILINGS_PATH = Path("results/extraction_manifest_filings.csv")
DEFAULT_SAMPLE_FILINGS_PATH = Path("results/review_sample_pass_filings.csv")
DEFAULT_SAMPLE_ITEMS_PATH = Path("results/review_sample_pass_items.csv")
DEFAULT_MISSING_PART_SAMPLES_PATH = Path("results/missing_part_10q_samples.csv")
DEFAULT_CORE_ITEMS = ("1", "1A", "7", "7A", "8")
OFFSET_BASIS_EXTRACTOR_BODY = "extractor_body"
DEFAULT_HTML_OUT_DIR = Path("results/html_audit")
DEFAULT_HTML_SCOPE = "sample"
DEFAULT_HTML_SAMPLE_SIZE = 100
DEFAULT_HTML_FILING_PREVIEW_CHARS = 1200
DEFAULT_HTML_ITEM_PREVIEW_CHARS = 800
COHEN2020_COMMON_CANONICAL: dict[str, set[str]] = {
    "10-K": {
        "II:7_MDA",
        "I:3_LEGAL_PROCEEDINGS",
        "II:7A_MARKET_RISK",
        "I:1A_RISK_FACTORS",
        "II:9B_OTHER_INFORMATION",
    },
    "10-Q": {
        "I:2_MDA",
        "II:1_LEGAL_PROCEEDINGS",
        "I:3_MARKET_RISK",
        "II:1A_RISK_FACTORS",
        "II:4_OTHER_INFORMATION_REDESIGNATED",
        "II:5_OTHER_INFORMATION",
    },
}
COHEN2020_ALL_ITEMS_10K_CANONICAL = {
    *COHEN2020_COMMON_CANONICAL.get("10-K", set())
}
# TODO: Replace with a dedicated Cohen 2020 10-K item list if/when defined.
COHEN2020_ALL_ITEMS_10Q_KEYS = {
    "I:1",
    "I:2",
    "I:3",
    "I:4",
    "II:1",
    "II:1A",
    "II:2",
    "II:3",
    "II:4",
    "II:5",
}

PER_ITEM_REPORT_LIMIT = 30

PROVENANCE_FIELDS = [
    "prov_python",
    "prov_module",
    "prov_cwd",
    "prov_sys_path_hash",
    "prov_enable_embedded_verifier",
]

DIAGNOSTICS_ROW_FIELDS = [
    "doc_id",
    "cik",
    "accession",
    "form_type",
    "filing_date",
    "period_end",
    "item_part",
    "item_id",
    "item",
    "item_missing_part",
    "canonical_item",
    "exists_by_regime",
    "item_status",
    "counts_for_target",
    "filing_exclusion_reason",
    "gij_omitted_items",
    "heading_line",
    "heading_index",
    "heading_offset",
    "prefix_text",
    "prefix_kind",
    "is_part_only_prefix",
    "is_crossref_prefix",
    "prev_line",
    "next_line",
    "flags",
    "embedded_heading_warn",
    "embedded_heading_fail",
    "first_hit_kind",
    "first_hit_classification",
    "first_hit_item_id",
    "first_hit_part",
    "first_hit_line_idx",
    "first_hit_char_pos",
    "first_hit_snippet",
    "first_fail_kind",
    "first_fail_classification",
    "first_fail_item_id",
    "first_fail_part",
    "first_fail_line_idx",
    "first_fail_char_pos",
    "first_fail_snippet",
    "first_embedded_kind",
    "first_embedded_classification",
    "first_embedded_class",
    "first_embedded_item_id",
    "first_embedded_part",
    "first_embedded_line_idx",
    "first_embedded_char_pos",
    "first_embedded_snippet",
    "heading_line_raw",
    "internal_heading_leak",
    "leak_pos",
    "leak_match",
    "leak_context",
    "leak_next_item_id",
    "leak_next_heading",
]

CSV_FIELDS = PROVENANCE_FIELDS + DIAGNOSTICS_ROW_FIELDS

# Manifest outputs include all extracted items/filings for manual audit.
MANIFEST_ITEM_FIELDS = [
    "doc_id",
    "accession",
    "cik",
    "filing_date",
    "period_end",
    "form",
    "item_part",
    "item_id",
    "item",
    "item_missing_part",
    "canonical_item",
    "item_status",
    "heading_start",
    "heading_end",
    "content_start",
    "content_end",
    "length_chars",
    "heading_line_raw",
    "heading_line_clean",
    "doc_head_200",
    "doc_tail_200",
    "embedded_heading_warn",
    "embedded_heading_fail",
    "first_embedded_kind",
    "first_embedded_classification",
    "first_embedded_item_id",
    "first_embedded_part",
    "first_embedded_line_idx",
    "first_embedded_char_pos",
    "first_embedded_snippet",
    "first_fail_kind",
    "first_fail_classification",
    "first_fail_item_id",
    "first_fail_part",
    "first_fail_line_idx",
    "first_fail_char_pos",
    "first_fail_snippet",
    "filing_exclusion_reason",
    "gij_omitted_items",
    "offset_basis",
]

MISSING_PART_SAMPLE_FIELDS = [
    "doc_id",
    "accession",
    "filing_date",
    "period_end",
    "form_type",
    "item_id",
    "canonical_item",
    "heading_line_raw",
    "heading_index",
    "heading_offset",
    "prefix_text",
    "prefix_kind",
    "next_line",
    "bucket_name",
]

MANIFEST_FILING_FIELDS = [
    "doc_id",
    "accession",
    "cik",
    "filing_date",
    "period_end",
    "form",
    "n_items_extracted",
    "items_extracted",
    "missing_core_items",
    "missing_expected_canonicals",
    "any_warn",
    "any_fail",
    "filing_exclusion_reason",
    "start_candidates_total",
    "start_candidates_toc_rejected",
    "start_selection_unverified",
    "truncated_successor_total",
    "truncated_part_total",
]


@dataclass(frozen=True)
class DiagnosticsConfig:
    parquet_dir: Path = DEFAULT_PARQUET_DIR
    out_path: Path = DEFAULT_OUT_PATH
    report_path: Path = DEFAULT_REPORT_PATH
    samples_dir: Path = DEFAULT_SAMPLES_DIR
    batch_size: int = 8
    max_files: int = 0
    max_examples: int = 25
    enable_embedded_verifier: bool = True
    emit_manifest: bool = True
    manifest_items_path: Path = DEFAULT_MANIFEST_ITEMS_PATH
    manifest_filings_path: Path = DEFAULT_MANIFEST_FILINGS_PATH
    sample_pass: int = 100
    sample_seed: int = 42
    sample_filings_path: Path = DEFAULT_SAMPLE_FILINGS_PATH
    sample_items_path: Path = DEFAULT_SAMPLE_ITEMS_PATH
    dump_missing_part_samples: int = 0
    missing_part_samples_path: Path = DEFAULT_MISSING_PART_SAMPLES_PATH
    core_items: tuple[str, ...] = DEFAULT_CORE_ITEMS
    target_set: str | None = None
    emit_html: bool = True
    html_out: Path = DEFAULT_HTML_OUT_DIR
    html_scope: str = DEFAULT_HTML_SCOPE
    html_sample_weights: dict[str, float] | None = None
    html_min_total_chars: int | None = None
    html_min_largest_item_chars: int | None = None
    html_min_largest_item_chars_pct_total: float | None = None
    extraction_regime: Literal["legacy", "v2"] = "legacy"
    diagnostics_regime: Literal["legacy", "v2"] = "legacy"
    focus_items: dict[str, set[str]] | None = None
    report_item_scope: Literal["all", "target", "focus"] | None = None


@dataclass(frozen=True)
class RegressionConfig:
    csv_path: Path = DEFAULT_CSV_PATH
    parquet_dir: Path = DEFAULT_PARQUET_DIR
    sample_per_flag: int = 3
    max_files: int = 0


@dataclass(frozen=True)
class InternalHeadingLeak:
    position: int
    match_text: str
    context: str


@dataclass(frozen=True)
class DiagnosticsRow:
    doc_id: str
    cik: str
    accession: str
    form_type: str
    filing_date: str
    period_end: str
    item_part: str
    item_id: str
    item: str
    item_missing_part: bool
    canonical_item: str
    exists_by_regime: str | bool | None
    item_status: str
    counts_for_target: bool
    filing_exclusion_reason: str
    gij_omitted_items: str
    heading_line: str
    heading_index: int | None
    heading_offset: int | None
    prefix_text: str
    prefix_kind: str
    is_part_only_prefix: bool
    is_crossref_prefix: bool
    prev_line: str
    next_line: str
    flags: tuple[str, ...]
    embedded_heading_warn: bool
    embedded_heading_fail: bool
    first_hit_kind: str
    first_hit_classification: str
    first_hit_item_id: str
    first_hit_part: str
    first_hit_line_idx: int | None
    first_hit_char_pos: int | None
    first_hit_snippet: str
    first_fail_kind: str
    first_fail_classification: str
    first_fail_item_id: str
    first_fail_part: str
    first_fail_line_idx: int | None
    first_fail_char_pos: int | None
    first_fail_snippet: str
    first_embedded_kind: str
    first_embedded_classification: str
    first_embedded_class: str
    first_embedded_item_id: str
    first_embedded_part: str
    first_embedded_line_idx: int | None
    first_embedded_char_pos: int | None
    first_embedded_snippet: str
    heading_line_raw: str
    internal_heading_leak: bool
    leak_pos: int | str
    leak_match: str
    leak_context: str
    leak_next_item_id: str
    leak_next_heading: str
    embedded_hits: tuple[EmbeddedHeadingHit, ...]
    item_full_text: str

    def to_dict(self, provenance: dict[str, str]) -> dict[str, object]:
        # Keep schema synchronized across CSV/report/sample outputs.
        def _csv_value(value: int | str | None) -> int | str:
            if value is None:
                return ""
            return value

        row = {
            "doc_id": self.doc_id,
            "cik": self.cik,
            "accession": self.accession,
            "form_type": self.form_type,
            "filing_date": self.filing_date,
            "period_end": self.period_end,
            "item_part": self.item_part,
            "item_id": self.item_id,
            "item": self.item,
            "item_missing_part": self.item_missing_part,
            "canonical_item": self.canonical_item,
            "exists_by_regime": self.exists_by_regime,
            "item_status": self.item_status,
            "counts_for_target": self.counts_for_target,
            "filing_exclusion_reason": self.filing_exclusion_reason,
            "gij_omitted_items": self.gij_omitted_items,
            "heading_line": self.heading_line,
            "heading_index": _csv_value(self.heading_index),
            "heading_offset": _csv_value(self.heading_offset),
            "prefix_text": self.prefix_text,
            "prefix_kind": self.prefix_kind,
            "is_part_only_prefix": self.is_part_only_prefix,
            "is_crossref_prefix": self.is_crossref_prefix,
            "prev_line": self.prev_line,
            "next_line": self.next_line,
            "flags": ";".join(self.flags),
            "embedded_heading_warn": self.embedded_heading_warn,
            "embedded_heading_fail": self.embedded_heading_fail,
            "first_hit_kind": self.first_hit_kind,
            "first_hit_classification": self.first_hit_classification,
            "first_hit_item_id": self.first_hit_item_id,
            "first_hit_part": self.first_hit_part,
            "first_hit_line_idx": _csv_value(self.first_hit_line_idx),
            "first_hit_char_pos": _csv_value(self.first_hit_char_pos),
            "first_hit_snippet": self.first_hit_snippet,
            "first_fail_kind": self.first_fail_kind,
            "first_fail_classification": self.first_fail_classification,
            "first_fail_item_id": self.first_fail_item_id,
            "first_fail_part": self.first_fail_part,
            "first_fail_line_idx": _csv_value(self.first_fail_line_idx),
            "first_fail_char_pos": _csv_value(self.first_fail_char_pos),
            "first_fail_snippet": self.first_fail_snippet,
            "first_embedded_kind": self.first_embedded_kind,
            "first_embedded_classification": self.first_embedded_classification,
            "first_embedded_class": self.first_embedded_class,
            "first_embedded_item_id": self.first_embedded_item_id,
            "first_embedded_part": self.first_embedded_part,
            "first_embedded_line_idx": _csv_value(self.first_embedded_line_idx),
            "first_embedded_char_pos": _csv_value(self.first_embedded_char_pos),
            "first_embedded_snippet": self.first_embedded_snippet,
            "heading_line_raw": self.heading_line_raw,
            "internal_heading_leak": "1" if self.internal_heading_leak else "",
            "leak_pos": _csv_value(self.leak_pos),
            "leak_match": self.leak_match,
            "leak_context": self.leak_context,
            "leak_next_item_id": self.leak_next_item_id,
            "leak_next_heading": self.leak_next_heading,
        }
        return {**provenance, **row}


@dataclass
class _ItemBreakdownStats:
    n_items: int = 0
    warn_items: int = 0
    fail_items: int = 0
    internal_heading_leak: int = 0
    warn_class_counts: Counter[str] = field(default_factory=Counter)
    warn_flag_counts: Counter[str] = field(default_factory=Counter)
    fail_class_counts: Counter[str] = field(default_factory=Counter)
    truncated_successor: int = 0
    truncated_part: int = 0
    missing_part: int = 0
    embedded_pos_pcts: list[float] = field(default_factory=list)


@dataclass
class _MissingPartDiagnostics:
    total_10q_items: int = 0
    missing_part_count: int = 0
    missing_by_item_id: Counter[str] = field(default_factory=Counter)
    missing_by_canonical: Counter[str] = field(default_factory=Counter)
    missing_by_bucket: Counter[str] = field(default_factory=Counter)
    missing_by_bucket_nonbullet_prefix: Counter[str] = field(default_factory=Counter)
    examples_by_bucket: dict[str, list[dict[str, str]]] = field(
        default_factory=lambda: defaultdict(list)
    )

def _parse_date(value: str | date | datetime | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        if len(s) == 8 and s.isdigit():
            try:
                return datetime.strptime(s, "%Y%m%d").date()
            except Exception:
                return None
        if len(s) == 10 and s[4] == "-" and s[7] == "-":
            try:
                return datetime.strptime(s, "%Y-%m-%d").date()
            except Exception:
                return None
    return None


def _normalized_form_type(form_type: str | None) -> str | None:
    return normalize_form_type(form_type)


def _is_10k(form_type: str | None) -> bool:
    return _normalized_form_type(form_type) == "10-K"

def _lettered_item(item_id: str) -> bool:
    return bool(re.search(r"\d+[A-Z]$", item_id))


def _weak_heading_letter(item_id: str, heading_line: str) -> bool:
    if not _lettered_item(item_id):
        return False
    m = ITEM_LINESTART_PATTERN.match(heading_line or "")
    if not m:
        return True
    detected = f"{m.group('num')}{m.group('let') or ''}".upper()
    return not detected.endswith(item_id[-1])


def _expected_part_for_item(item: dict, *, normalized_form: str | None) -> str | None:
    if normalized_form == "10-Q":
        allowed_parts = {"I", "II"}
    else:
        allowed_parts = {"I", "II", "III", "IV"}

    canonical = item.get("canonical_item")
    if isinstance(canonical, str) and ":" in canonical:
        part = canonical.split(":", 1)[0].upper()
        if part in allowed_parts:
            return part
    item_key = item.get("item")
    if isinstance(item_key, str) and item_key.startswith("?:"):
        return None
    if isinstance(item_key, str) and ":" in item_key:
        part = item_key.split(":", 1)[0].upper()
        if part in allowed_parts:
            return part
    item_id = item.get("item_id")
    if normalized_form == "10-K" and isinstance(item_id, str):
        return _default_part_for_item_id(item_id)
    return None


def _normalize_target_set(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = str(value).strip().lower()
    if cleaned == "cohen2020":
        return "cohen2020_common"
    return cleaned or None


def _target_set_for_form(
    normalized_form: str | None,
    target_set: str | None,
) -> set[str]:
    if not normalized_form or not target_set:
        return set()
    if target_set == "cohen2020_common":
        return COHEN2020_COMMON_CANONICAL.get(normalized_form, set())
    if target_set == "cohen2020_all_items":
        if normalized_form == "10-Q":
            canonicals = _canonicals_for_item_keys(
                normalized_form,
                COHEN2020_ALL_ITEMS_10Q_KEYS,
            )
            canonicals.update(
                {
                    "II:4_OTHER_INFORMATION_REDESIGNATED",
                    "II:5_OTHER_INFORMATION",
                }
            )
            return canonicals
        if normalized_form == "10-K":
            return set(COHEN2020_ALL_ITEMS_10K_CANONICAL)
    return set()


def _item_counts_for_target(
    canonical_item: str | None,
    *,
    normalized_form: str | None,
    target_set: str | None,
) -> bool:
    if not target_set:
        return True
    if not canonical_item:
        return False
    targets = _target_set_for_form(normalized_form, target_set)
    return canonical_item in targets


def _expected_canonical_items(
    *,
    normalized_form: str | None,
    filing_date: date | None,
    period_end: date | None,
    target_set: str | None,
) -> set[str]:
    if not normalized_form:
        return set()
    index = get_regime_index(normalized_form)
    if not index:
        return set()
    expected: set[str] = set()
    for key, entry in index.items_by_key.items():
        status = str(entry.get("status") or "").lower()
        if status in {"reserved", "optional"}:
            continue
        match, decidable = _evaluate_regime_validity(
            entry.get("validity", []),
            filing_date=filing_date,
            period_end=period_end,
        )
        if not decidable or match is None:
            continue
        canonical = match.get("canonical") or key
        if isinstance(canonical, str) and "NOT_IN_FORM" in canonical:
            continue
        if canonical:
            expected.add(str(canonical))
    if target_set:
        expected &= _target_set_for_form(normalized_form, target_set)
    return expected


def _expected_item_tokens_from_canonicals(
    canonicals: set[str],
) -> tuple[set[str], set[str]]:
    item_ids: set[str] = set()
    parts: set[str] = set()
    for canonical in canonicals:
        if not canonical:
            continue
        head = canonical.split("_", 1)[0]
        if ":" in head:
            part, item_id = head.split(":", 1)
            if part:
                parts.add(part.upper())
            if item_id:
                item_ids.add(item_id.upper())
        else:
            item_ids.add(head.upper())
    return item_ids, parts


def _parse_internal_leak_token(match_text: str) -> tuple[str | None, str | None]:
    if not match_text:
        return None, None
    item_match = re.search(
        r"(?i)\bITEM\s+(?P<num>\d{1,2})(?P<let>[A-Z])?\b",
        match_text,
    )
    if item_match:
        return (
            f"{item_match.group('num')}{item_match.group('let') or ''}".upper(),
            None,
        )
    part_match = re.search(r"(?i)\bPART\s+(?P<roman>[IVX]+)\b", match_text)
    if part_match:
        return None, part_match.group("roman").upper()
    return None, None


def _internal_leak_prose_confirmed(text: str, leak_pos: int) -> bool:
    if not text:
        return False
    if leak_pos < 0:
        return False
    normalized = _normalize_newlines(text)
    if leak_pos >= len(normalized):
        return False
    lines = normalized.splitlines(keepends=True)
    lines_noeol = [line.rstrip("\r\n") for line in lines]
    offset = 0
    line_idx = None
    for idx, line in enumerate(lines):
        next_offset = offset + len(line)
        if leak_pos < next_offset:
            line_idx = idx
            break
        offset = next_offset
    if line_idx is None:
        return False
    toc_window_flags = _toc_window_flags(lines_noeol)
    toc_cache: dict[tuple[int, bool], bool] = {}
    return _confirm_prose_after(lines_noeol, line_idx, toc_cache, toc_window_flags)


def _should_escalate_internal_leak_v2(
    *,
    leak_info: InternalHeadingLeak,
    item_full_text: str,
    next_item_id: str | None,
    next_part: str | None,
    expected_item_ids: set[str],
    expected_parts: set[str],
) -> bool:
    leak_item_id, leak_part = _parse_internal_leak_token(leak_info.match_text)
    next_item_norm = _normalize_item_id(next_item_id)
    next_part_norm = _normalize_part(next_part)
    if leak_item_id and next_item_norm and leak_item_id == next_item_norm:
        return True
    if leak_part and next_part_norm and leak_part == next_part_norm:
        return True
    if leak_item_id and leak_item_id in expected_item_ids:
        return True
    if leak_part and leak_part in expected_parts:
        return True
    return _internal_leak_prose_confirmed(item_full_text, leak_info.position)


def _embedded_warn_v2(hits: list[EmbeddedHeadingHit]) -> bool:
    if not hits:
        return False
    warn_hits = [hit for hit in hits if hit.classification in EMBEDDED_WARN_CLASSIFICATIONS]
    if not warn_hits:
        return False
    non_toc_cross = [
        hit
        for hit in warn_hits
        if hit.classification not in {"toc_row", "cross_ref_line"}
    ]
    if non_toc_cross:
        return True
    toc_cross = [
        hit
        for hit in warn_hits
        if hit.classification in {"toc_row", "cross_ref_line"}
    ]
    if not toc_cross:
        return False
    early = any(
        hit.char_pos <= EMBEDDED_TOC_START_EARLY_MAX_CHAR for hit in toc_cross
    )
    clustered = len(toc_cross) >= EMBEDDED_WARN_CLUSTER_MIN
    return early or clustered


def _canonicals_for_item_keys(
    normalized_form: str | None,
    item_keys: set[str],
) -> set[str]:
    if not normalized_form or not item_keys:
        return set()
    index = get_regime_index(normalized_form)
    if not index:
        return set()
    results: set[str] = set()
    for key in item_keys:
        entry = index.items_by_key.get(key)
        if not entry:
            continue
        for validity in entry.get("validity", []) or []:
            canonical = validity.get("canonical") or key
            if canonical:
                results.add(str(canonical))
    return results


def _all_canonicals_for_form(normalized_form: str | None) -> set[str]:
    if not normalized_form:
        return set()
    index = get_regime_index(normalized_form)
    if not index:
        return set()
    results: set[str] = set()
    for key, entry in index.items_by_key.items():
        for validity in entry.get("validity", []) or []:
            canonical = validity.get("canonical") or key
            if canonical:
                results.add(str(canonical))
    return results


def _compute_html_metrics(
    items_by_doc: dict[str, list[dict[str, object]]]
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for doc_id, items in items_by_doc.items():
        total = 0
        largest = 0
        for item in items:
            length = _parse_int(item.get("length_chars"), default=0)
            total += max(length, 0)
            if length > largest:
                largest = length
        pct = (largest / total) if total > 0 else 0.0
        metrics[doc_id] = {
            "total_chars": float(total),
            "largest_item_chars": float(largest),
            "largest_item_chars_pct_total": float(pct),
        }
    return metrics


def _passes_html_filters(
    row: dict[str, object],
    *,
    metrics_by_doc: dict[str, dict[str, float]],
    min_total_chars: int | None,
    min_largest_item_chars: int | None,
    min_largest_item_chars_pct_total: float | None,
) -> bool:
    if min_total_chars is None and min_largest_item_chars is None and min_largest_item_chars_pct_total is None:
        return True
    doc_id = str(row.get("doc_id") or "")
    metrics = metrics_by_doc.get(doc_id, {})
    if min_total_chars is not None and metrics.get("total_chars", 0.0) < min_total_chars:
        return False
    if (
        min_largest_item_chars is not None
        and metrics.get("largest_item_chars", 0.0) < min_largest_item_chars
    ):
        return False
    if (
        min_largest_item_chars_pct_total is not None
        and metrics.get("largest_item_chars_pct_total", 0.0)
        < min_largest_item_chars_pct_total
    ):
        return False
    return True


def _prefix_text(heading_line: str, heading_offset: int | None) -> str:
    if heading_offset is None or heading_offset <= 0:
        return ""
    return heading_line[:heading_offset]


def _prefix_kind(prefix: str) -> str:
    if not prefix or not prefix.strip():
        return PREFIX_KIND_BLANK
    if _prefix_is_part_only(prefix):
        return PREFIX_KIND_PART_ONLY
    if _prefix_is_bullet(prefix):
        return PREFIX_KIND_BULLET
    return PREFIX_KIND_TEXTUAL


def _prefix_has_textual_content(prefix: str) -> bool:
    if re.search(r"[A-Za-z]", prefix):
        return True
    trimmed = prefix.rstrip()
    if not trimmed:
        return False
    if trimmed[-1] in ".:;!?":
        before = trimmed[:-1]
        if re.search(r"[A-Za-z]", before):
            return True
        if re.search(r"\u2026|\.{2,}", before):
            return True
    return False


def _is_midline_heading_prefix(prefix: str, heading_offset: int | None) -> bool:
    if heading_offset is None or heading_offset <= 0:
        return False
    if _prefix_kind(prefix) != PREFIX_KIND_TEXTUAL:
        return False
    return _prefix_has_textual_content(prefix)


def _prefix_metadata(
    heading_line: str, heading_offset: int | None
) -> tuple[str, str, bool, bool, bool]:
    prefix_text = _prefix_text(heading_line, heading_offset)
    prefix_kind = _prefix_kind(prefix_text)
    is_part_only_prefix = prefix_kind == PREFIX_KIND_PART_ONLY
    is_crossref_prefix = _prefix_looks_like_cross_ref(prefix_text)
    is_midline_heading = _is_midline_heading_prefix(prefix_text, heading_offset)
    return (
        prefix_text,
        prefix_kind,
        is_part_only_prefix,
        is_crossref_prefix,
        is_midline_heading,
    )


def _compound_item_heading(heading_line: str) -> bool:
    if not heading_line:
        return False
    return len(ITEM_MENTION_PATTERN.findall(heading_line)) >= 2


def _split_flags(flag_field: str | None) -> list[str]:
    if not flag_field:
        return []
    return [flag.strip() for flag in flag_field.split(";") if flag.strip()]


def _normalize_body_lines(text: str) -> list[str]:
    return normalize_extractor_body(text).splitlines()


def _parse_core_items_arg(value: str | None) -> tuple[str, ...]:
    if not value:
        return DEFAULT_CORE_ITEMS
    parts = re.split(r"[,\s]+", value.strip())
    cleaned = tuple(sorted({p.strip().upper() for p in parts if p.strip()}))
    return cleaned or DEFAULT_CORE_ITEMS


def _normalize_focus_form(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = re.sub(r"[^A-Z0-9]", "", str(value).upper())
    if cleaned == "10K":
        return "10-K"
    if cleaned == "10Q":
        return "10-Q"
    return None


def _normalize_focus_item_key(form_type: str, token: str) -> str | None:
    cleaned = str(token).strip().upper()
    if not cleaned:
        return None
    normalized_form = _normalized_form_type(form_type) or form_type
    if normalized_form == "10-Q":
        if ":" in cleaned:
            part, item_id = cleaned.split(":", 1)
            part = part.strip().upper() or "?"
            item_id = item_id.strip().upper()
            return f"{part}:{item_id}" if item_id else f"{part}:?"
        return f"?:{cleaned}"
    if ":" in cleaned:
        part, item_id = cleaned.split(":", 1)
        part = part.strip().upper()
        item_id = item_id.strip().upper()
        if part and item_id:
            return f"{part}:{item_id}"
        return item_id or part or None
    return cleaned


def parse_focus_items(value: str | None) -> dict[str, set[str]] | None:
    if not value or not str(value).strip():
        return None
    focus: dict[str, set[str]] = {}
    for raw_section in str(value).split(";"):
        section = raw_section.strip()
        if not section or ":" not in section:
            continue
        form_raw, items_raw = section.split(":", 1)
        form = _normalize_focus_form(form_raw)
        if not form:
            continue
        items: set[str] = set()
        for token in items_raw.split(","):
            item_key = _normalize_focus_item_key(form, token)
            if item_key:
                items.add(item_key)
        if items:
            focus[form] = items
    return focus or None


def _item_key_for_report(
    form_type: str | None,
    item_part: str | None,
    item_id: str | None,
) -> str:
    normalized_form = _normalized_form_type(form_type) or str(form_type or "")
    item_id_clean = str(item_id or "").strip().upper()
    part_clean = _normalize_part(item_part) or str(item_part or "").strip().upper()
    if normalized_form == "10-Q":
        if part_clean:
            if item_id_clean:
                return f"{part_clean}:{item_id_clean}"
            return f"{part_clean}:?"
        if item_id_clean:
            return f"?:{item_id_clean}"
        return "?:?"
    if part_clean:
        if item_id_clean:
            return f"{part_clean}:{item_id_clean}"
        return part_clean
    return item_id_clean or "UNKNOWN"


def _report_focus_items_from_target_set(target_set: str) -> dict[str, set[str]]:
    focus: dict[str, set[str]] = {}
    for form in ("10-K", "10-Q"):
        canonicals = _target_set_for_form(form, target_set)
        if not canonicals:
            continue
        keys: set[str] = set()
        for canonical in canonicals:
            if not canonical:
                continue
            prefix = str(canonical).split("_", 1)[0].strip()
            item_key = _normalize_focus_item_key(form, prefix)
            if item_key:
                keys.add(item_key)
        if form == "10-Q":
            missing_part_keys: set[str] = set()
            for item_key in keys:
                if ":" in item_key:
                    _, item_id = item_key.split(":", 1)
                else:
                    item_id = item_key
                item_id = item_id.strip().upper()
                if item_id:
                    missing_part_keys.add(f"?:{item_id}")
            keys |= missing_part_keys
        if keys:
            focus[form] = keys
    return focus


def _resolve_report_item_scope(
    scope: str | None,
    target_set: str | None,
) -> Literal["all", "target", "focus"]:
    if scope in {"all", "target", "focus"}:
        return scope
    return "target" if target_set else "all"


def _quartile_edges(values: list[int]) -> tuple[int, int, int]:
    if not values:
        return (0, 0, 0)
    ordered = sorted(values)
    n = len(ordered)
    q1 = ordered[int(0.25 * (n - 1))]
    q2 = ordered[int(0.50 * (n - 1))]
    q3 = ordered[int(0.75 * (n - 1))]
    return (q1, q2, q3)


def _quartile_bucket(value: int, edges: tuple[int, int, int]) -> str:
    q1, q2, q3 = edges
    if value <= q1:
        return "Q1"
    if value <= q2:
        return "Q2"
    if value <= q3:
        return "Q3"
    return "Q4"


def _stratified_sample(
    strata: dict[tuple[str, bool], list[dict[str, object]]],
    *,
    sample_size: int,
    seed: int,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    total = sum(len(rows) for rows in strata.values())
    if total == 0 or sample_size <= 0:
        return []
    if sample_size > total:
        sample_size = total

    targets: dict[tuple[str, bool], int] = {}
    fractional: list[tuple[float, tuple[str, bool]]] = []
    for key, rows in strata.items():
        raw = (sample_size * len(rows)) / total
        base = int(raw)
        targets[key] = min(base, len(rows))
        fractional.append((raw - base, key))

    remaining = sample_size - sum(targets.values())
    fractional.sort(reverse=True, key=lambda t: t[0])
    for _, key in fractional:
        if remaining <= 0:
            break
        if targets[key] < len(strata[key]):
            targets[key] += 1
            remaining -= 1

    while remaining > 0:
        available = [key for key in strata if targets[key] < len(strata[key])]
        if not available:
            break
        key = rng.choice(available)
        targets[key] += 1
        remaining -= 1

    sampled: list[dict[str, object]] = []
    for key in sorted(strata.keys()):
        rows = strata[key]
        k = targets.get(key, 0)
        if k <= 0:
            continue
        sampled.extend(rng.sample(rows, k))
    return sampled


def _find_internal_heading_leak(
    text: str,
    *,
    ignore_chars: int = INTERNAL_HEADING_IGNORE_CHARS,
    context_chars: int = INTERNAL_HEADING_CONTEXT_CHARS,
) -> InternalHeadingLeak | None:
    if not text:
        return None
    normalized = _normalize_newlines(text)
    if len(normalized) <= ignore_chars:
        return None

    candidates: list[re.Match[str]] = []
    for pattern in (INTERNAL_PART_PATTERN, INTERNAL_ITEM_PATTERN):
        for match in pattern.finditer(normalized):
            if match.start() < ignore_chars:
                continue
            candidates.append(match)
            break

    if not candidates:
        return None

    earliest = min(candidates, key=lambda m: m.start())
    pos = earliest.start()
    start = max(0, pos - context_chars)
    end = min(len(normalized), pos + context_chars)
    context = normalized[start:end].replace("\n", "\\n")
    return InternalHeadingLeak(
        position=pos,
        match_text=earliest.group(0).strip(),
        context=context,
    )


def _char_pos_pct(char_pos: int, full_text_len: int) -> float:
    if full_text_len <= 0:
        return 0.0
    pct = char_pos / full_text_len
    if pct < 0.0:
        return 0.0
    if pct > 1.0:
        return 1.0
    return pct


def _char_pos_bucket(char_pos_pct: float) -> str:
    if char_pos_pct <= 0.05:
        return "0-0.05"
    if char_pos_pct <= 0.2:
        return "0.05-0.2"
    if char_pos_pct <= 0.8:
        return "0.2-0.8"
    return "0.8-1.0"


def _format_embedded_fail_entry(
    entry: dict[str, str | int],
    *,
    include_accession: bool,
    include_form: bool,
    prefix: str,
) -> str:
    part = entry.get("item_part") or ""
    item_id = entry.get("item_id") or ""
    item_key = f"{part}:{item_id}".strip(":") if part or item_id else "UNKNOWN"
    snippet = str(entry.get("snippet") or "").replace("\"", "'")
    char_pos = int(entry.get("char_pos") or 0)
    full_len = int(entry.get("full_text_len") or 0)
    pos_pct = float(entry.get("char_pos_pct") or _char_pos_pct(char_pos, full_len))
    to_end = int(entry.get("to_end") or max(full_len - char_pos, 0))
    fields = [f"doc_id={entry.get('doc_id','')}"]
    if include_accession:
        fields.append(f"accession={entry.get('accession','')}")
    if include_form:
        fields.append(f"form={entry.get('form_type','')}")
    fields.extend(
        [
            f"item={item_key}",
            f"kind={entry.get('kind','')}",
            f"class={entry.get('classification','')}",
            f"char_pos={char_pos}",
            f"full_len={full_len}",
            f"pos_pct={pos_pct:.3f}",
            f"to_end={to_end}",
            f"snippet=\"{snippet}\"",
        ]
    )
    return f"{prefix}{' '.join(fields)}"


def _rate(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return count / total


def _format_counter_top(counter: Counter[str], *, top_n: int = 3) -> str:
    if not counter:
        return "(none)"
    return ", ".join(
        f"{label}={count}" for label, count in counter.most_common(top_n)
    )


def _summarize_pos_pcts(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return (None, None)
    ordered = sorted(values)
    median = statistics.median(ordered)
    p90_index = max(0, min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1))
    return (median, ordered[p90_index])


def _missing_part_bucket(
    heading_line_raw: str,
    prefix_text: str,
    next_line: str,
) -> str:
    combined = " ".join(
        part for part in (heading_line_raw, prefix_text, next_line) if part
    )
    if not combined:
        return "other_unclear"
    if MISSING_PART_PART_PATTERN.search(combined) and MISSING_PART_ITEM_PATTERN.search(
        combined
    ):
        if MISSING_PART_PART_ITEM_PUNCT_PATTERN.search(combined):
            return "combined_heading_part_item"
    if MISSING_PART_FORM_HEADER_PATTERN.search(combined) and MISSING_PART_PART_PATTERN.search(
        combined
    ):
        return "form_header_midline_part"
    if MISSING_PART_PART_PATTERN.search(combined) and not MISSING_PART_ITEM_PATTERN.search(
        combined
    ):
        return "part_marker_line_only"
    if MISSING_PART_ITEM_PATTERN.search(combined) and not MISSING_PART_PART_PATTERN.search(
        combined
    ):
        return "item_only_no_part"
    return "other_unclear"


def _prefix_non_empty_non_bullet(prefix_kind: str) -> bool:
    return prefix_kind not in {"", PREFIX_KIND_BLANK, PREFIX_KIND_BULLET}


def _update_missing_part_diagnostics(
    stats: _MissingPartDiagnostics,
    *,
    doc_id: str,
    accession: str,
    item_key: str,
    item_id: str,
    canonical_item: str,
    heading_line_raw: str,
    prefix_text: str,
    prefix_kind: str,
    next_line: str,
) -> str:
    bucket = _missing_part_bucket(heading_line_raw, prefix_text, next_line)
    stats.missing_part_count += 1
    stats.missing_by_item_id[item_id or ""] += 1
    if canonical_item:
        stats.missing_by_canonical[canonical_item] += 1
    stats.missing_by_bucket[bucket] += 1
    if _prefix_non_empty_non_bullet(prefix_kind):
        stats.missing_by_bucket_nonbullet_prefix[bucket] += 1
    examples = stats.examples_by_bucket[bucket]
    if len(examples) < 3:
        examples.append(
            {
                "doc_id": doc_id,
                "accession": accession,
                "item": item_key,
                "heading": heading_line_raw,
            }
        )
    return bucket


def _reservoir_sample_update(
    sample: list[dict[str, object]],
    entry: dict[str, object],
    *,
    seen: int,
    sample_size: int,
    rng: random.Random,
) -> int:
    seen += 1
    if sample_size <= 0:
        return seen
    if len(sample) < sample_size:
        sample.append(entry)
        return seen
    pick = rng.randint(1, seen)
    if pick <= sample_size:
        sample[pick - 1] = entry
    return seen


def _update_item_breakdown(
    item_breakdown: dict[tuple[str, str], _ItemBreakdownStats],
    *,
    form_type: str | None,
    item_part: str | None,
    item_id: str | None,
    flags: list[str],
    item_warn: bool,
    item_fail: bool,
    internal_heading_leak: bool,
    embedded_hits: list[EmbeddedHeadingHit],
    embedded_first_flagged: EmbeddedHeadingHit | None,
    truncated_successor: bool,
    truncated_part: bool,
    item_missing_part: bool,
    item_full_text: str,
) -> None:
    form_label = _normalized_form_type(form_type) or str(form_type or "UNKNOWN")
    item_key = _item_key_for_report(form_label, item_part, item_id)
    stats = item_breakdown.setdefault((form_label, item_key), _ItemBreakdownStats())

    stats.n_items += 1
    if item_warn:
        stats.warn_items += 1
    if item_fail:
        stats.fail_items += 1

    if item_warn and internal_heading_leak:
        stats.internal_heading_leak += 1

    if item_warn and flags:
        for flag in flags:
            if flag in {"embedded_heading_warn", "embedded_heading_fail"}:
                continue
            stats.warn_flag_counts[flag] += 1

    if item_warn and embedded_hits:
        for hit in embedded_hits:
            if hit.classification in EMBEDDED_WARN_CLASSIFICATIONS:
                stats.warn_class_counts[hit.classification] += 1

    if item_fail and embedded_hits:
        for hit in embedded_hits:
            if hit.classification in EMBEDDED_FAIL_CLASSIFICATIONS:
                stats.fail_class_counts[hit.classification] += 1

    if truncated_successor:
        stats.truncated_successor += 1
    if truncated_part:
        stats.truncated_part += 1

    if _normalized_form_type(form_type) == "10-Q" and item_missing_part:
        stats.missing_part += 1

    if embedded_first_flagged is not None:
        full_len = embedded_first_flagged.full_text_len
        if full_len <= 0:
            full_len = len(item_full_text or "")
        stats.embedded_pos_pcts.append(
            _char_pos_pct(embedded_first_flagged.char_pos, full_len)
        )


def _build_item_breakdown_rows(
    item_breakdown: dict[tuple[str, str], _ItemBreakdownStats],
    *,
    focus_items: dict[str, set[str]] | None,
) -> dict[str, list[dict[str, object]]]:
    rows_by_form: dict[str, list[dict[str, object]]] = defaultdict(list)
    for (form, item_key), stats in item_breakdown.items():
        if focus_items is not None:
            allowed = focus_items.get(form)
            if not allowed or item_key not in allowed:
                continue
        n_items = stats.n_items
        warn_rate = _rate(stats.warn_items, n_items)
        fail_rate = _rate(stats.fail_items, n_items)
        median_pct, p90_pct = _summarize_pos_pcts(stats.embedded_pos_pcts)
        rows_by_form[form].append(
            {
                "form": form,
                "item_key": item_key,
                "n_items": n_items,
                "warn_items": stats.warn_items,
                "warn_rate": warn_rate,
                "fail_items": stats.fail_items,
                "fail_rate": fail_rate,
                "top_warn_drivers": (
                    "internal_heading_leak="
                    f"{stats.internal_heading_leak}; embedded_warn: "
                    f"{_format_counter_top(stats.warn_class_counts)}; flags: "
                    f"{_format_counter_top(stats.warn_flag_counts)}"
                ),
                "top_fail_drivers": (
                    "embedded_fail: "
                    f"{_format_counter_top(stats.fail_class_counts)}"
                ),
                "pct_truncated_successor": _rate(stats.truncated_successor, n_items),
                "pct_truncated_part": _rate(stats.truncated_part, n_items),
                "pct_missing_part": _rate(stats.missing_part, n_items)
                if form == "10-Q"
                else None,
                "median_first_embedded_pct": median_pct,
                "p90_first_embedded_pct": p90_pct,
            }
        )

    for form, rows in rows_by_form.items():
        rows.sort(
            key=lambda row: (
                float(row["fail_rate"]),
                float(row["warn_rate"]),
                int(row["n_items"]),
            ),
            reverse=True,
        )
    return rows_by_form


def _format_per_item_breakdown_table(
    rows: list[dict[str, object]],
    *,
    limit: int,
) -> list[str]:
    if not rows:
        return ["  (no items matched)"]
    header = (
        "  item_key  n_items  warn_items  warn_rate  fail_items  fail_rate  "
        "top_warn_drivers | top_fail_drivers"
    )
    lines = [header]
    for row in rows[:limit]:
        lines.append(
            "  "
            f"{row['item_key']:<8} "
            f"{int(row['n_items']):>7} "
            f"{int(row['warn_items']):>10} "
            f"{float(row['warn_rate']):>8.1%} "
            f"{int(row['fail_items']):>10} "
            f"{float(row['fail_rate']):>8.1%} "
            f"{row['top_warn_drivers']} | {row['top_fail_drivers']}"
        )
    return lines


def _format_boundary_health_table(
    rows: list[dict[str, object]],
    *,
    limit: int,
) -> list[str]:
    if not rows:
        return ["  (no items matched)"]
    header = (
        "  item_key  n_items  pct_truncated_successor  pct_truncated_part  "
        "pct_missing_part  median_first_embedded_pct  p90_first_embedded_pct"
    )
    lines = [header]
    for row in rows[:limit]:
        missing_part = row["pct_missing_part"]
        median_pct = row["median_first_embedded_pct"]
        p90_pct = row["p90_first_embedded_pct"]
        lines.append(
            "  "
            f"{row['item_key']:<8} "
            f"{int(row['n_items']):>7} "
            f"{float(row['pct_truncated_successor']):>23.1%} "
            f"{float(row['pct_truncated_part']):>18.1%} "
            f"{(f'{missing_part:.1%}' if isinstance(missing_part, float) else 'n/a'):>16} "
            f"{(f'{median_pct:.1%}' if isinstance(median_pct, float) else 'n/a'):>24} "
            f"{(f'{p90_pct:.1%}' if isinstance(p90_pct, float) else 'n/a'):>21}"
        )
    return lines


def _sorted_forms(forms: list[str]) -> list[str]:
    order = {"10-K": 0, "10-Q": 1}
    return sorted(forms, key=lambda form: (order.get(form, 99), form))


def _append_item_breakdown_sections(
    lines_out: list[str],
    *,
    title: str,
    rows_by_form: dict[str, list[dict[str, object]]],
) -> None:
    lines_out.append("")
    lines_out.append(title)
    lines_out.append("Rates shown as % of n_items.")
    lines_out.append(
        "Showing top "
        f"{PER_ITEM_REPORT_LIMIT} rows per form. Increase PER_ITEM_REPORT_LIMIT in "
        "suspicious_boundary_diagnostics.py to show more."
    )
    if not rows_by_form:
        lines_out.append("  (no items matched)")
    else:
        for form in _sorted_forms(list(rows_by_form)):
            lines_out.append(f"Form: {form}")
            lines_out.extend(
                _format_per_item_breakdown_table(
                    rows_by_form[form], limit=PER_ITEM_REPORT_LIMIT
                )
            )

    lines_out.append("")
    lines_out.append("Boundary health signals by item")
    lines_out.append(
        "Rates shown as % of n_items. Embedded position stats use first_embedded "
        "hit positions (char_pos / item_len)."
    )
    lines_out.append(
        "Showing top "
        f"{PER_ITEM_REPORT_LIMIT} rows per form. Increase PER_ITEM_REPORT_LIMIT in "
        "suspicious_boundary_diagnostics.py to show more."
    )
    if not rows_by_form:
        lines_out.append("  (no items matched)")
    else:
        for form in _sorted_forms(list(rows_by_form)):
            lines_out.append(f"Form: {form}")
            lines_out.extend(
                _format_boundary_health_table(
                    rows_by_form[form], limit=PER_ITEM_REPORT_LIMIT
                )
            )


def load_root_causes_text(path: Path = DEFAULT_ROOT_CAUSES_PATH) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    archive_path = path.parent / "archive" / path.name
    if archive_path.exists():
        return archive_path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Root causes file not found at {path} or {archive_path}")


def _short_sys_path_hash() -> str:
    head = sys.path[:5]
    joined = "|".join(str(entry) for entry in head)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:10]


def _build_provenance(config: DiagnosticsConfig) -> dict[str, str]:
    return {
        "prov_python": sys.executable or "",
        "prov_module": str(Path(__file__).resolve()),
        "prov_cwd": str(Path.cwd()),
        "prov_sys_path_hash": _short_sys_path_hash(),
        "prov_enable_embedded_verifier": str(config.enable_embedded_verifier),
    }


def _build_flagged_row(
    *,
    doc_id: str,
    cik: str,
    accession: str,
    form_type: str,
    filing_date: date | None,
    period_end: date | None,
    item: dict,
    item_id: str,
    item_part: str,
    item_missing_part: bool,
    heading_line_clean: str,
    heading_line_raw: str,
    heading_idx: int | None,
    heading_offset: int | None,
    prefix_text: str,
    prefix_kind: str,
    is_part_only_prefix: bool,
    is_crossref_prefix: bool,
    prev_line: str,
    next_line: str,
    flags: list[str],
    embedded_hits: list[EmbeddedHeadingHit],
    embedded_warn: bool,
    embedded_fail: bool,
    embedded_first_hit: EmbeddedHeadingHit | None,
    embedded_first_flagged: EmbeddedHeadingHit | None,
    embedded_first_fail: EmbeddedHeadingHit | None,
    leak_info: InternalHeadingLeak | None,
    leak_pos: int | str,
    leak_match: str,
    leak_context: str,
    leak_next_item_id: str,
    leak_next_heading: str,
    item_full_text: str,
    counts_for_target: bool,
) -> DiagnosticsRow:
    filing_exclusion_reason = str(item.get("_filing_exclusion_reason") or "")
    gij_omitted_items = item.get("_gij_omitted_items")
    if isinstance(gij_omitted_items, (list, tuple, set)):
        gij_omitted_items_str = ",".join(str(entry) for entry in gij_omitted_items)
    else:
        gij_omitted_items_str = str(gij_omitted_items or "")
    return DiagnosticsRow(
        doc_id=doc_id,
        cik=cik,
        accession=accession,
        form_type=form_type,
        filing_date=filing_date.isoformat() if filing_date else "",
        period_end=period_end.isoformat() if period_end else "",
        item_part=item_part or "",
        item_id=item_id,
        item=item.get("item") or "",
        item_missing_part=item_missing_part,
        canonical_item=item.get("canonical_item") or "",
        exists_by_regime=item.get("exists_by_regime"),
        item_status=item.get("item_status") or "",
        counts_for_target=counts_for_target,
        filing_exclusion_reason=filing_exclusion_reason,
        gij_omitted_items=gij_omitted_items_str,
        heading_line=heading_line_clean.strip(),
        heading_index=heading_idx,
        heading_offset=heading_offset,
        prefix_text=prefix_text,
        prefix_kind=prefix_kind,
        is_part_only_prefix=is_part_only_prefix,
        is_crossref_prefix=is_crossref_prefix,
        prev_line=prev_line,
        next_line=next_line,
        flags=tuple(flags),
        embedded_heading_warn=embedded_warn,
        embedded_heading_fail=embedded_fail,
        first_hit_kind=embedded_first_hit.kind if embedded_first_hit else "",
        first_hit_classification=(
            embedded_first_hit.classification if embedded_first_hit else ""
        ),
        first_hit_item_id=embedded_first_hit.item_id if embedded_first_hit else "",
        first_hit_part=embedded_first_hit.part if embedded_first_hit else "",
        first_hit_line_idx=embedded_first_hit.line_idx if embedded_first_hit else None,
        first_hit_char_pos=embedded_first_hit.char_pos if embedded_first_hit else None,
        first_hit_snippet=embedded_first_hit.snippet if embedded_first_hit else "",
        first_fail_kind=embedded_first_fail.kind if embedded_first_fail else "",
        first_fail_classification=(
            embedded_first_fail.classification if embedded_first_fail else ""
        ),
        first_fail_item_id=embedded_first_fail.item_id if embedded_first_fail else "",
        first_fail_part=embedded_first_fail.part if embedded_first_fail else "",
        first_fail_line_idx=embedded_first_fail.line_idx if embedded_first_fail else None,
        first_fail_char_pos=embedded_first_fail.char_pos if embedded_first_fail else None,
        first_fail_snippet=embedded_first_fail.snippet if embedded_first_fail else "",
        first_embedded_kind=embedded_first_flagged.kind if embedded_first_flagged else "",
        first_embedded_classification=(
            embedded_first_flagged.classification if embedded_first_flagged else ""
        ),
        first_embedded_class=(
            embedded_first_flagged.classification if embedded_first_flagged else ""
        ),
        first_embedded_item_id=embedded_first_flagged.item_id if embedded_first_flagged else "",
        first_embedded_part=embedded_first_flagged.part if embedded_first_flagged else "",
        first_embedded_line_idx=(
            embedded_first_flagged.line_idx if embedded_first_flagged else None
        ),
        first_embedded_char_pos=(
            embedded_first_flagged.char_pos if embedded_first_flagged else None
        ),
        first_embedded_snippet=embedded_first_flagged.snippet if embedded_first_flagged else "",
        heading_line_raw=heading_line_raw.strip(),
        internal_heading_leak=leak_info is not None,
        leak_pos=leak_pos,
        leak_match=leak_match,
        leak_context=leak_context,
        leak_next_item_id=leak_next_item_id,
        leak_next_heading=leak_next_heading,
        embedded_hits=tuple(embedded_hits),
        item_full_text=item_full_text,
    )


def _csv_value(value: int | str | None) -> int | str:
    if value is None:
        return ""
    return value


def _build_diagnostics_report(
    *,
    rows: list[DiagnosticsRow],
    total_filings: int,
    total_items: int,
    total_part_only_prefix: int,
    start_candidates_total: int,
    start_candidates_toc_rejected_total: int,
    start_selection_unverified_total: int,
    truncated_successor_total: int,
    truncated_part_total: int,
    parquet_dir: Path,
    max_examples: int,
    provenance: dict[str, str],
    extraction_regime: str,
    diagnostics_regime: str,
    item_breakdown: dict[tuple[str, str], _ItemBreakdownStats],
    missing_part_diagnostics: _MissingPartDiagnostics,
    focus_items: dict[str, set[str]] | None,
    report_item_scope: Literal["all", "target", "focus"],
    target_set: str | None,
) -> str:
    flags_count: Counter[str] = Counter()
    examples_by_flag: dict[str, list[dict[str, str]]] = defaultdict(list)
    prefix_examples: dict[str, list[str]] = defaultdict(list)
    internal_leak_total = 0
    internal_leak_by_form: Counter[str] = Counter()
    internal_leak_by_item: Counter[str] = Counter()
    internal_leak_examples: list[dict[str, str | int]] = []
    embedded_warn_total = 0
    embedded_fail_total = 0
    embedded_classification_counts: Counter[str] = Counter()
    embedded_fail_examples: list[dict[str, str | int]] = []
    embedded_fail_pos_buckets: Counter[str] = Counter()

    for row in rows:
        for flag in row.flags:
            flags_count[flag] += 1
            if len(examples_by_flag[flag]) < 3:
                examples_by_flag[flag].append(
                    {
                        "doc_id": row.doc_id,
                        "cik": row.cik,
                        "accession": row.accession,
                        "item_id": row.item_id,
                        "item_part": row.item_part,
                        "heading": row.heading_line,
                        "prev": row.prev_line,
                        "next": row.next_line,
                    }
                )
            if flag in {"midline_heading", "cross_ref_prefix"}:
                if row.prefix_text and len(prefix_examples[flag]) < 2:
                    prefix_examples[flag].append(row.prefix_text.strip())

        if row.internal_heading_leak:
            internal_leak_total += 1
            internal_leak_by_form[row.form_type] += 1
            internal_leak_by_item[row.item_id] += 1
            internal_leak_examples.append(
                {
                    "doc_id": row.doc_id,
                    "accession": row.accession,
                    "form_type": row.form_type,
                    "item_id": row.item_id,
                    "item_part": row.item_part,
                    "leak_pos": row.leak_pos,
                    "leak_match": row.leak_match,
                    "leak_context": row.leak_context,
                    "next_item_id": row.leak_next_item_id,
                    "next_heading": row.leak_next_heading,
                }
            )

        if row.embedded_heading_warn:
            embedded_warn_total += 1
        if row.embedded_heading_fail:
            embedded_fail_total += 1

        if row.embedded_hits:
            for hit in row.embedded_hits:
                if hit.classification in EMBEDDED_WARN_CLASSIFICATIONS | EMBEDDED_FAIL_CLASSIFICATIONS:
                    embedded_classification_counts[hit.classification] += 1
                if hit.classification in EMBEDDED_FAIL_CLASSIFICATIONS:
                    pos_pct = _char_pos_pct(hit.char_pos, hit.full_text_len)
                    embedded_fail_pos_buckets[_char_pos_bucket(pos_pct)] += 1
                    embedded_fail_examples.append(
                        {
                            "doc_id": row.doc_id,
                            "accession": row.accession,
                            "form_type": row.form_type,
                            "item_id": row.item_id,
                            "item_part": row.item_part,
                            "classification": hit.classification,
                            "kind": hit.kind,
                            "char_pos": hit.char_pos,
                            "full_text_len": hit.full_text_len,
                            "char_pos_pct": pos_pct,
                            "to_end": max(hit.full_text_len - hit.char_pos, 0),
                            "snippet": hit.snippet,
                        }
                    )

    lines_out: list[str] = []
    lines_out.append("Suspicious boundary diagnostics (unified)")
    lines_out.append("Provenance:")
    lines_out.append(f"  python_executable: {provenance.get('prov_python', '')}")
    lines_out.append(f"  module_file: {provenance.get('prov_module', '')}")
    lines_out.append(f"  cwd: {provenance.get('prov_cwd', '')}")
    lines_out.append(f"  sys_path_hash: {provenance.get('prov_sys_path_hash', '')}")
    lines_out.append(
        "  enable_embedded_verifier: "
        f"{provenance.get('prov_enable_embedded_verifier', '')}"
    )
    lines_out.append(f"  extraction_regime: {extraction_regime}")
    lines_out.append(f"  diagnostics_regime: {diagnostics_regime}")
    lines_out.append("")
    lines_out.append(f"Parquet dir: {parquet_dir}")
    lines_out.append(f"Total filings processed: {total_filings}")
    lines_out.append(f"Total items extracted: {total_items}")
    lines_out.append(f"Total flagged items: {len(rows)}")
    lines_out.append(f"Part-only prefixes (informational): {total_part_only_prefix}")
    lines_out.append(f"Start candidates (total): {start_candidates_total}")
    lines_out.append(
        f"Start candidates rejected by TOC/SUMMARY: {start_candidates_toc_rejected_total}"
    )
    lines_out.append(f"Start selections unverified: {start_selection_unverified_total}")
    lines_out.append(f"Truncated by successor heading: {truncated_successor_total}")
    lines_out.append(f"Truncated by PART boundary: {truncated_part_total}")
    lines_out.append("")
    lines_out.append("10-Q PART completeness diagnostics")
    total_10q_items = missing_part_diagnostics.total_10q_items
    missing_part_count = missing_part_diagnostics.missing_part_count
    missing_part_rate = _rate(missing_part_count, total_10q_items)
    lines_out.append(f"  total_10q_items: {total_10q_items}")
    lines_out.append(f"  missing_part_count: {missing_part_count}")
    lines_out.append(f"  missing_part_rate: {missing_part_rate:.2%}")
    if missing_part_diagnostics.missing_by_item_id:
        top_items = missing_part_diagnostics.missing_by_item_id.most_common(15)
        lines_out.append("  missing_part_by_item_id (top 15):")
        for item_id, count in top_items:
            lines_out.append(f"    {item_id or 'UNKNOWN'}: {count}")
    else:
        lines_out.append("  missing_part_by_item_id (top 15): (none)")
    if missing_part_diagnostics.missing_by_canonical:
        top_canonicals = missing_part_diagnostics.missing_by_canonical.most_common(15)
        lines_out.append("  missing_part_by_canonical_item (top 15):")
        for canonical, count in top_canonicals:
            lines_out.append(f"    {canonical}: {count}")
    else:
        lines_out.append("  missing_part_by_canonical_item (top 15): (none)")
    lines_out.append("  missing_part_by_heading_bucket:")
    if missing_part_count > 0 and missing_part_diagnostics.missing_by_bucket:
        for bucket in MISSING_PART_BUCKETS:
            count = missing_part_diagnostics.missing_by_bucket.get(bucket, 0)
            if count <= 0:
                continue
            rate = _rate(count, missing_part_count)
            prefix_nonbullet = missing_part_diagnostics.missing_by_bucket_nonbullet_prefix.get(
                bucket, 0
            )
            prefix_pct = _rate(prefix_nonbullet, count)
            lines_out.append(
                f"    {bucket}: count={count} rate={rate:.2%} "
                f"prefix_nonbullet_pct={prefix_pct:.2%}"
            )
    else:
        lines_out.append("    (none)")
    top_bucket_counts = sorted(
        missing_part_diagnostics.missing_by_bucket.items(),
        key=lambda entry: entry[1],
        reverse=True,
    )[:5]
    if top_bucket_counts:
        lines_out.append("  missing_part_bucket_examples:")
        for bucket, _count in top_bucket_counts:
            examples = missing_part_diagnostics.examples_by_bucket.get(bucket, [])
            if not examples:
                continue
            lines_out.append(f"    {bucket}:")
            for example in examples[:3]:
                heading = str(example.get("heading", "")).replace("\"", "'")
                lines_out.append(
                    "      "
                    f"doc_id={example.get('doc_id','')} "
                    f"accession={example.get('accession','')} "
                    f"item={example.get('item','')} "
                    f"heading=\"{heading}\""
                )
    else:
        lines_out.append("  missing_part_bucket_examples: (none)")
    lines_out.append("")
    lines_out.append("Flags (counts):")
    for flag, count in flags_count.most_common():
        lines_out.append(f"  {flag}: {count}")
    if not flags_count:
        lines_out.append("  (no flags recorded)")
    lines_out.append("")
    lines_out.append("Embedded heading summary:")
    lines_out.append(f"  embedded_heading_warn: {embedded_warn_total}")
    lines_out.append(f"  embedded_heading_fail: {embedded_fail_total}")
    lines_out.append(
        "  toc_start_misfire hits: "
        f"{embedded_classification_counts.get('toc_start_misfire', 0)}"
    )
    lines_out.append(
        "  toc_start_misfire_early hits: "
        f"{embedded_classification_counts.get('toc_start_misfire_early', 0)}"
    )
    if embedded_classification_counts:
        lines_out.append("  By classification (hits):")
        for classification, count in embedded_classification_counts.most_common():
            lines_out.append(f"    {classification}: {count}")
    else:
        lines_out.append("  By classification (hits): (none)")
    lines_out.append("")
    lines_out.append("Embedded fail position buckets (char_pos_pct):")
    for bucket in ("0-0.05", "0.05-0.2", "0.2-0.8", "0.8-1.0"):
        lines_out.append(f"  {bucket}: {embedded_fail_pos_buckets.get(bucket, 0)}")
    lines_out.append("")
    lines_out.append("Earliest embedded heading fails:")
    earliest_embedded_fails = sorted(
        embedded_fail_examples,
        key=lambda entry: entry.get("char_pos", 0),
    )[:max_examples]
    if earliest_embedded_fails:
        for entry in earliest_embedded_fails:
            lines_out.append(
                _format_embedded_fail_entry(
                    entry, include_accession=True, include_form=True, prefix="  "
                )
            )
    else:
        lines_out.append("  (no embedded heading fails detected)")
    lines_out.append("")
    lines_out.append("Latest embedded heading fails:")
    latest_embedded_fails = sorted(
        embedded_fail_examples,
        key=lambda entry: entry.get("char_pos", 0),
        reverse=True,
    )[:max_examples]
    if latest_embedded_fails:
        for entry in latest_embedded_fails:
            lines_out.append(
                _format_embedded_fail_entry(
                    entry, include_accession=True, include_form=True, prefix="  "
                )
            )
    else:
        lines_out.append("  (no embedded heading fails detected)")
    lines_out.append("")
    lines_out.append("Closest-to-end embedded heading fails:")
    closest_to_end_fails = sorted(
        embedded_fail_examples,
        key=lambda entry: entry.get("to_end", 0),
    )[:max_examples]
    if closest_to_end_fails:
        for entry in closest_to_end_fails:
            lines_out.append(
                _format_embedded_fail_entry(
                    entry, include_accession=True, include_form=True, prefix="  "
                )
            )
    else:
        lines_out.append("  (no embedded heading fails detected)")
    lines_out.append("")
    lines_out.append("Examples by flag:")
    if examples_by_flag:
        for flag, examples in examples_by_flag.items():
            lines_out.append(f"  {flag}:")
            for ex in examples:
                lines_out.append(
                    "    "
                    f"doc_id={ex['doc_id']} cik={ex['cik']} accession={ex['accession']} "
                    f"part={ex['item_part']} item_id={ex['item_id']} "
                    f"heading=\"{ex['heading']}\""
                )
                if ex["prev"]:
                    lines_out.append(f"      prev=\"{ex['prev']}\"")
                if ex["next"]:
                    lines_out.append(f"      next=\"{ex['next']}\"")
    else:
        lines_out.append("  (no flagged examples captured)")
    lines_out.append("")
    lines_out.append("Representative prefixes:")
    for flag in ("midline_heading", "cross_ref_prefix"):
        reps = prefix_examples.get(flag, [])
        if reps:
            lines_out.append(f"  {flag}:")
            for rep in reps:
                lines_out.append(f"    prefix=\"{rep}\"")
        else:
            lines_out.append(f"  {flag}: (none captured)")

    lines_out.append("")
    lines_out.append("Internal heading leak audit:")
    lines_out.append(f"Total internal heading leaks: {internal_leak_total}")
    if internal_leak_by_form:
        lines_out.append("By form:")
        for form, count in internal_leak_by_form.most_common():
            form_label = form or "UNKNOWN"
            lines_out.append(f"  {form_label}: {count}")
    else:
        lines_out.append("By form: (none detected)")
    if internal_leak_by_item:
        lines_out.append("By item_id:")
        for item_id, count in internal_leak_by_item.most_common():
            lines_out.append(f"  {item_id}: {count}")
    else:
        lines_out.append("By item_id: (none detected)")

    lines_out.append("Worst cases (earliest leak position):")
    worst_cases = sorted(
        internal_leak_examples,
        key=lambda entry: entry.get("leak_pos", 0),
    )[:max_examples]
    if worst_cases:
        for entry in worst_cases:
            part = entry.get("item_part") or ""
            item_id = entry.get("item_id") or ""
            item_key = f"{part}:{item_id}".strip(":") if part or item_id else "UNKNOWN"
            lines_out.append(
                "  "
                f"doc_id={entry.get('doc_id','')} accession={entry.get('accession','')} "
                f"form={entry.get('form_type','')} item={item_key} "
                f"leak_pos={entry.get('leak_pos','')} match=\"{entry.get('leak_match','')}\""
            )
            if entry.get("next_item_id") or entry.get("next_heading"):
                lines_out.append(
                    "    "
                    f"next_item_id={entry.get('next_item_id','')} "
                    f"next_heading=\"{entry.get('next_heading','')}\""
                )
            if entry.get("leak_context"):
                lines_out.append(f"    context=\"{entry.get('leak_context','')}\"")
    else:
        lines_out.append("  (no internal heading leaks detected)")

    if report_item_scope == "all":
        rows_all = _build_item_breakdown_rows(item_breakdown, focus_items=None)
        _append_item_breakdown_sections(
            lines_out,
            title="Per-item breakdown (All extracted items)",
            rows_by_form=rows_all,
        )
    elif report_item_scope == "focus":
        rows_focus = _build_item_breakdown_rows(item_breakdown, focus_items=focus_items)
        _append_item_breakdown_sections(
            lines_out,
            title="Per-item breakdown (Focus set: custom)",
            rows_by_form=rows_focus,
        )
    elif report_item_scope == "target":
        target_items = _report_focus_items_from_target_set(target_set or "")
        rows_target = _build_item_breakdown_rows(
            item_breakdown, focus_items=target_items
        )
        _append_item_breakdown_sections(
            lines_out,
            title=f"Per-item breakdown (Target set: {target_set})",
            rows_by_form=rows_target,
        )

    return "\n".join(lines_out)


def run_boundary_diagnostics(config: DiagnosticsConfig) -> dict[str, int]:
    parquet_dir = config.parquet_dir
    if not parquet_dir.exists():
        raise SystemExit(f"parquet-dir not found: {parquet_dir}")

    files = sorted(parquet_dir.glob("*_batch_*.parquet"))
    if config.max_files and config.max_files > 0:
        files = files[: config.max_files]
    if not files:
        raise SystemExit(f"No parquet batch files found in {parquet_dir}")

    config.out_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.samples_dir.mkdir(parents=True, exist_ok=True)

    emit_manifest = config.emit_manifest or config.sample_pass > 0 or config.emit_html
    if emit_manifest:
        config.manifest_items_path.parent.mkdir(parents=True, exist_ok=True)
        config.manifest_filings_path.parent.mkdir(parents=True, exist_ok=True)
    if config.sample_pass and config.sample_pass > 0:
        config.sample_filings_path.parent.mkdir(parents=True, exist_ok=True)
        config.sample_items_path.parent.mkdir(parents=True, exist_ok=True)

    provenance = _build_provenance(config)
    out_headers = CSV_FIELDS

    core_items = {item.strip().upper() for item in config.core_items if item}
    if not core_items:
        core_items = set(DEFAULT_CORE_ITEMS)
    target_set = _normalize_target_set(config.target_set)
    report_item_scope = _resolve_report_item_scope(
        config.report_item_scope, target_set
    )
    if report_item_scope == "focus" and not config.focus_items:
        raise ValueError("report_item_scope='focus' requires focus_items.")
    if report_item_scope == "target" and not target_set:
        raise ValueError("report_item_scope='target' requires target_set.")

    total_filings = 0
    total_items = 0
    total_part_only_prefix = 0
    start_candidates_total = 0
    start_candidates_toc_rejected_total = 0
    start_selection_unverified_total = 0
    truncated_successor_total = 0
    truncated_part_total = 0
    missing_part_diagnostics = _MissingPartDiagnostics()
    missing_part_samples: list[dict[str, object]] = []
    missing_part_seen = 0
    missing_part_rng = random.Random(config.sample_seed)
    # Keep a single row representation to avoid CSV/report/sample schema drift.
    flagged_rows: list[DiagnosticsRow] = []
    item_breakdown: dict[tuple[str, str], _ItemBreakdownStats] = {}
    filing_rows_for_sampling: list[dict[str, object]] = []

    want_cols = [
        "doc_id",
        "cik",
        "accession_number",
        "document_type_filename",
        "file_date_filename",
        "full_text",
    ]

    with ExitStack() as stack:
        flagged_csv = stack.enter_context(
            config.out_path.open("w", newline="", encoding="utf-8")
        )
        writer = csv.DictWriter(flagged_csv, fieldnames=out_headers)
        writer.writeheader()

        manifest_items_writer: csv.DictWriter | None = None
        manifest_filings_writer: csv.DictWriter | None = None
        if emit_manifest:
            manifest_items_file = stack.enter_context(
                config.manifest_items_path.open("w", newline="", encoding="utf-8")
            )
            manifest_items_writer = csv.DictWriter(
                manifest_items_file, fieldnames=MANIFEST_ITEM_FIELDS
            )
            manifest_items_writer.writeheader()
            manifest_filings_file = stack.enter_context(
                config.manifest_filings_path.open("w", newline="", encoding="utf-8")
            )
            manifest_filings_writer = csv.DictWriter(
                manifest_filings_file, fieldnames=MANIFEST_FILING_FIELDS
            )
            manifest_filings_writer.writeheader()

        for file_path in files:
            pf = pq.ParquetFile(file_path)
            available = set(pf.schema.names)
            columns = [c for c in want_cols if c in available]
            for batch in pf.iter_batches(batch_size=config.batch_size, columns=columns):
                tbl = pa.Table.from_batches([batch])
                df = pl.from_arrow(tbl)
                for row in df.iter_rows(named=True):
                    form_type = row.get("document_type_filename")
                    normalized_form = _normalized_form_type(form_type)
                    if normalized_form not in {"10-K", "10-Q"}:
                        continue

                    total_filings += 1
                    text = row.get("full_text") or ""
                    if not text:
                        continue

                    hdr = parse_header(text)
                    filing_date = _parse_date(hdr.get("header_filing_date_str")) or _parse_date(
                        row.get("file_date_filename")
                    )
                    period_end = _parse_date(hdr.get("header_period_end_str"))

                    doc_id = str(row.get("doc_id") or "")
                    accession = str(row.get("accession_number") or "")
                    cik = str(row.get("cik") or "")
                    form_label = normalized_form or str(form_type or "")
                    filing_date_str = filing_date.isoformat() if filing_date else ""
                    period_end_str = period_end.isoformat() if period_end else ""
                    expected_canonicals = _expected_canonical_items(
                        normalized_form=normalized_form,
                        filing_date=filing_date,
                        period_end=period_end,
                        target_set=target_set,
                    )
                    expected_item_ids, expected_parts = _expected_item_tokens_from_canonicals(
                        expected_canonicals
                    )

                    items = extract_filing_items(
                        text,
                        form_type=form_type,
                        filing_date=filing_date,
                        period_end=period_end,
                        regime=True,
                        diagnostics=True,
                        extraction_regime=config.extraction_regime,
                    )
                    if not items:
                        if emit_manifest and manifest_filings_writer:
                            missing_core_items = ",".join(sorted(core_items))
                            missing_expected = ",".join(sorted(expected_canonicals))
                            filing_row = {
                                "doc_id": doc_id,
                                "accession": accession,
                                "cik": cik,
                                "filing_date": filing_date_str,
                                "period_end": period_end_str,
                                "form": form_label,
                                "n_items_extracted": 0,
                                "items_extracted": "",
                                "missing_core_items": missing_core_items,
                                "missing_expected_canonicals": missing_expected,
                                "any_warn": False,
                                "any_fail": False,
                                "filing_exclusion_reason": "",
                                "start_candidates_total": 0,
                                "start_candidates_toc_rejected": 0,
                                "start_selection_unverified": 0,
                                "truncated_successor_total": 0,
                                "truncated_part_total": 0,
                            }
                            manifest_filings_writer.writerow(filing_row)
                            if config.sample_pass and config.sample_pass > 0:
                                filing_rows_for_sampling.append(
                                    {
                                        "row": filing_row,
                                        "missing_core": bool(missing_core_items),
                                        "any_fail": False,
                                        "filing_exclusion_reason": "",
                                    }
                                )
                        continue

                    body = normalize_extractor_body(text)
                    lines = body.splitlines()
                    total_items += len(items)
                    item_id_sequence = [
                        _normalize_item_id(str(entry.get("item_id") or "")) for entry in items
                    ]
                    flagged_items_for_doc: list[DiagnosticsRow] = []
                    item_ids_ordered: list[str] = []
                    item_ids_set: set[str] = set()
                    extracted_canonicals: set[str] = set()
                    filing_exclusion_reason = ""
                    filing_any_warn = False
                    filing_any_fail = False
                    start_candidates_total_filing = 0
                    start_candidates_toc_rejected_filing = 0
                    start_selection_unverified_filing = 0
                    truncated_successor_total_filing = 0
                    truncated_part_total_filing = 0

                    for item_idx, item in enumerate(items):
                        next_item_id = (
                            item_id_sequence[item_idx + 1]
                            if item_idx + 1 < len(item_id_sequence)
                            else None
                        )
                        next_part = (
                            items[item_idx + 1].get("item_part")
                            if item_idx + 1 < len(items)
                            else None
                        )
                        nearby_slice = item_id_sequence[
                            item_idx + 1 : item_idx + 1 + EMBEDDED_NEARBY_ITEM_WINDOW
                        ]
                        nearby_item_ids = {item_id for item_id in nearby_slice if item_id}
                        heading_line_raw = item.get("_heading_line_raw")
                        if heading_line_raw is None:
                            heading_line_raw = item.get("_heading_line") or ""
                        heading_line_raw = str(heading_line_raw or "")

                        heading_line_clean = item.get("_heading_line")
                        if heading_line_clean is None or heading_line_clean == "":
                            heading_line_clean = item.get("_heading_line_clean") or heading_line_raw
                        heading_line_clean = str(heading_line_clean or "")

                        heading_idx = item.get("_heading_line_index")
                        heading_offset = item.get("_heading_offset")
                        next_line_non_empty = ""
                        if heading_idx is not None and heading_idx >= 0:
                            _, next_line_non_empty = _non_empty_line(
                                lines, heading_idx + 1
                            )
                        item_id = str(item.get("item_id") or "")
                        item_id_norm = item_id.strip().upper()
                        item_part = item.get("item_part")
                        item_key = str(item.get("item") or "")
                        item_full_text = str(item.get("full_text") or "")
                        canonical_item = str(item.get("canonical_item") or "")
                        if canonical_item:
                            extracted_canonicals.add(canonical_item)
                        current_part_norm = _normalize_part(item_part) or _normalize_part(
                            _expected_part_for_item(item, normalized_form=normalized_form)
                        )
                        (
                            prefix_text,
                            prefix_kind,
                            is_part_only_prefix,
                            is_crossref_prefix,
                            is_midline_heading,
                        ) = _prefix_metadata(heading_line_raw, heading_offset)
                        if is_part_only_prefix and heading_offset and heading_offset > 0:
                            total_part_only_prefix += 1

                        leak_info = _find_internal_heading_leak(item_full_text)
                        leak_pos: int | str = ""
                        leak_match = ""
                        leak_context = ""
                        leak_next_item_id = ""
                        leak_next_heading = ""
                        leak_escalate = True
                        if leak_info is not None and config.diagnostics_regime == "v2":
                            leak_escalate = _should_escalate_internal_leak_v2(
                                leak_info=leak_info,
                                item_full_text=item_full_text,
                                next_item_id=next_item_id,
                                next_part=str(next_part or ""),
                                expected_item_ids=expected_item_ids,
                                expected_parts=expected_parts,
                            )
                        embedded_hits = (
                            _find_embedded_heading_hits(
                                item_full_text,
                                current_item_id=item_id,
                                current_part=current_part_norm,
                                next_item_id=next_item_id,
                                nearby_item_ids=nearby_item_ids,
                            )
                            if config.enable_embedded_verifier
                            else []
                        )
                        (
                            embedded_warn,
                            embedded_fail,
                            embedded_first_hit,
                            embedded_first_flagged,
                            embedded_first_fail,
                            _embedded_counts,
                        ) = _summarize_embedded_hits(embedded_hits)
                        if config.diagnostics_regime == "v2":
                            embedded_warn = _embedded_warn_v2(embedded_hits)
                        start_candidates_total += int(item.get("_start_candidates_total") or 0)
                        start_candidates_toc_rejected_total += int(
                            item.get("_start_candidates_toc_rejected") or 0
                        )
                        if item.get("_start_selection_verified") is False:
                            start_selection_unverified_total += 1
                        if item.get("_truncated_successor_heading"):
                            truncated_successor_total += 1
                        if item.get("_truncated_part_boundary"):
                            truncated_part_total += 1

                        start_candidates_total_filing += int(
                            item.get("_start_candidates_total") or 0
                        )
                        start_candidates_toc_rejected_filing += int(
                            item.get("_start_candidates_toc_rejected") or 0
                        )
                        if item.get("_start_selection_verified") is False:
                            start_selection_unverified_filing += 1
                        if item.get("_truncated_successor_heading"):
                            truncated_successor_total_filing += 1
                        if item.get("_truncated_part_boundary"):
                            truncated_part_total_filing += 1

                        flags: list[str] = []
                        if heading_idx is None or heading_idx < 0:
                            flags.append("missing_heading_line")
                        else:
                            if _looks_like_toc_heading_line(lines, heading_idx):
                                flags.append("toc_like_heading")
                            if heading_offset and is_crossref_prefix:
                                flags.append("cross_ref_prefix")
                            if _compound_item_heading(heading_line_clean):
                                flags.append("compound_item_heading")

                        if is_midline_heading:
                            flags.append("midline_heading")

                        if _weak_heading_letter(item_id, heading_line_raw):
                            flags.append("weak_letter_heading")

                        if heading_idx is not None and heading_idx >= 0:
                            if next_line_non_empty:
                                if re.match(
                                    r"^\s*[ABC]\s*[\.\):\-]", next_line_non_empty
                                ):
                                    if item_id in {"1", "7", "9"} or _lettered_item(item_id):
                                        flags.append("split_letter_line")

                        expected_part = _expected_part_for_item(
                            item, normalized_form=normalized_form
                        )
                        if expected_part and not item_part:
                            flags.append("missing_part")
                        if item_part and expected_part and item_part != expected_part:
                            flags.append("part_mismatch")

                        if TOC_DOT_LEADER_PATTERN.search(heading_line_raw):
                            flags.append("dot_leader_heading")

                        if leak_info is not None:
                            if leak_escalate:
                                flags.append("internal_heading_leak")
                            leak_pos = leak_info.position
                            leak_match = leak_info.match_text
                            leak_context = leak_info.context
                            if item_idx + 1 < len(items):
                                next_item = items[item_idx + 1]
                                leak_next_item_id = str(next_item.get("item_id") or "")
                                leak_next_heading = str(
                                    next_item.get("_heading_line")
                                    or next_item.get("_heading_line_clean")
                                    or next_item.get("_heading_line_raw")
                                    or ""
                                )

                        if embedded_warn:
                            flags.append("embedded_heading_warn")
                        if embedded_fail:
                            flags.append("embedded_heading_fail")

                        flags = sorted(set(flags))
                        first_fail_classification = (
                            embedded_first_fail.classification if embedded_first_fail else ""
                        )
                        item_fail = embedded_fail or bool(first_fail_classification)
                        item_warn = bool(flags) and not item_fail
                        item_missing_part = bool(item.get("item_missing_part")) or item_key.startswith(
                            "?:"
                        )
                        if normalized_form == "10-Q":
                            missing_part_diagnostics.total_10q_items += 1
                            if item_missing_part:
                                bucket = _update_missing_part_diagnostics(
                                    missing_part_diagnostics,
                                    doc_id=doc_id,
                                    accession=accession,
                                    item_key=item_key or f"{item_part or ''}:{item_id}".strip(":"),
                                    item_id=item_id,
                                    canonical_item=canonical_item,
                                    heading_line_raw=heading_line_raw,
                                    prefix_text=prefix_text,
                                    prefix_kind=prefix_kind,
                                    next_line=next_line_non_empty,
                                )
                                if config.dump_missing_part_samples > 0:
                                    entry = {
                                        "doc_id": doc_id,
                                        "accession": accession,
                                        "filing_date": filing_date_str,
                                        "period_end": period_end_str,
                                        "form_type": form_label,
                                        "item_id": item_id,
                                        "canonical_item": canonical_item,
                                        "heading_line_raw": heading_line_raw,
                                        "heading_index": heading_idx,
                                        "heading_offset": heading_offset,
                                        "prefix_text": prefix_text,
                                        "prefix_kind": prefix_kind,
                                        "next_line": next_line_non_empty,
                                        "bucket_name": bucket,
                                    }
                                    missing_part_seen = _reservoir_sample_update(
                                        missing_part_samples,
                                        entry,
                                        seen=missing_part_seen,
                                        sample_size=config.dump_missing_part_samples,
                                        rng=missing_part_rng,
                                    )
                        _update_item_breakdown(
                            item_breakdown,
                            form_type=form_label,
                            item_part=str(item_part or ""),
                            item_id=item_id,
                            flags=flags,
                            item_warn=item_warn,
                            item_fail=item_fail,
                            internal_heading_leak=leak_info is not None,
                            embedded_hits=embedded_hits,
                            embedded_first_flagged=embedded_first_flagged,
                            truncated_successor=bool(item.get("_truncated_successor_heading")),
                            truncated_part=bool(item.get("_truncated_part_boundary")),
                            item_missing_part=item_missing_part,
                            item_full_text=item_full_text,
                        )
                        counts_for_target = _item_counts_for_target(
                            canonical_item,
                            normalized_form=normalized_form,
                            target_set=target_set,
                        )
                        if counts_for_target:
                            filing_any_fail = filing_any_fail or item_fail
                            filing_any_warn = filing_any_warn or item_warn

                        if item_id_norm:
                            item_ids_ordered.append(item_id_norm)
                            item_ids_set.add(item_id_norm)

                        item_filing_exclusion_reason = str(
                            item.get("_filing_exclusion_reason") or ""
                        )
                        if not filing_exclusion_reason and item_filing_exclusion_reason:
                            filing_exclusion_reason = item_filing_exclusion_reason

                        if emit_manifest and manifest_items_writer:
                            heading_start = item.get("_heading_start")
                            heading_end = item.get("_heading_end")
                            content_start = item.get("_content_start")
                            content_end = item.get("_content_end")
                            length_chars = ""
                            doc_head_200 = ""
                            doc_tail_200 = ""
                            if isinstance(content_start, int) and isinstance(content_end, int):
                                length_chars = max(content_end - content_start, 0)
                                body_len = len(body)
                                start = max(0, min(content_start, body_len))
                                end = max(0, min(content_end, body_len))
                                if end < start:
                                    end = start
                                # Snippets are slices from extractor body offsets.
                                doc_head_200 = body[start : min(start + 200, body_len)]
                                tail_start = max(end - 200, start)
                                doc_tail_200 = body[tail_start:end]

                            gij_omitted_items = item.get("_gij_omitted_items")
                            if isinstance(gij_omitted_items, (list, tuple, set)):
                                gij_omitted_items_str = ",".join(
                                    str(entry) for entry in gij_omitted_items
                                )
                            else:
                                gij_omitted_items_str = str(gij_omitted_items or "")

                            item_status = str(item.get("item_status") or "")
                            if item_id_norm == "16":
                                item_status = "excluded"

                            manifest_items_writer.writerow(
                                {
                                    "doc_id": doc_id,
                                    "accession": accession,
                                    "cik": cik,
                                    "filing_date": filing_date_str,
                                    "period_end": period_end_str,
                                    "form": form_label,
                                    "item_part": str(item_part or ""),
                                    "item_id": item_id,
                                    "item": item_key,
                                    "item_missing_part": item_missing_part,
                                    "canonical_item": str(item.get("canonical_item") or ""),
                                    "item_status": item_status,
                                    "heading_start": _csv_value(heading_start),
                                    "heading_end": _csv_value(heading_end),
                                    "content_start": _csv_value(content_start),
                                    "content_end": _csv_value(content_end),
                                    "length_chars": _csv_value(length_chars),
                                    "heading_line_raw": heading_line_raw,
                                    "heading_line_clean": heading_line_clean,
                                    "doc_head_200": doc_head_200,
                                    "doc_tail_200": doc_tail_200,
                                    "embedded_heading_warn": embedded_warn,
                                    "embedded_heading_fail": embedded_fail,
                                    "first_embedded_kind": (
                                        embedded_first_flagged.kind
                                        if embedded_first_flagged
                                        else ""
                                    ),
                                    "first_embedded_classification": (
                                        embedded_first_flagged.classification
                                        if embedded_first_flagged
                                        else ""
                                    ),
                                    "first_embedded_item_id": (
                                        embedded_first_flagged.item_id
                                        if embedded_first_flagged
                                        else ""
                                    ),
                                    "first_embedded_part": (
                                        embedded_first_flagged.part
                                        if embedded_first_flagged
                                        else ""
                                    ),
                                    "first_embedded_line_idx": _csv_value(
                                        embedded_first_flagged.line_idx
                                        if embedded_first_flagged
                                        else None
                                    ),
                                    "first_embedded_char_pos": _csv_value(
                                        embedded_first_flagged.char_pos
                                        if embedded_first_flagged
                                        else None
                                    ),
                                    "first_embedded_snippet": (
                                        embedded_first_flagged.snippet
                                        if embedded_first_flagged
                                        else ""
                                    ),
                                    "first_fail_kind": (
                                        embedded_first_fail.kind if embedded_first_fail else ""
                                    ),
                                    "first_fail_classification": first_fail_classification,
                                    "first_fail_item_id": (
                                        embedded_first_fail.item_id
                                        if embedded_first_fail
                                        else ""
                                    ),
                                    "first_fail_part": (
                                        embedded_first_fail.part if embedded_first_fail else ""
                                    ),
                                    "first_fail_line_idx": _csv_value(
                                        embedded_first_fail.line_idx
                                        if embedded_first_fail
                                        else None
                                    ),
                                    "first_fail_char_pos": _csv_value(
                                        embedded_first_fail.char_pos
                                        if embedded_first_fail
                                        else None
                                    ),
                                    "first_fail_snippet": (
                                        embedded_first_fail.snippet
                                        if embedded_first_fail
                                        else ""
                                    ),
                                    "filing_exclusion_reason": item_filing_exclusion_reason,
                                    "gij_omitted_items": gij_omitted_items_str,
                                    "offset_basis": OFFSET_BASIS_EXTRACTOR_BODY,
                                }
                            )

                        is_flagged = bool(flags)
                        if not is_flagged:
                            continue

                        prev_line = ""
                        next_line = ""
                        if heading_idx is not None and heading_idx >= 0:
                            if heading_idx > 0:
                                prev_line = lines[heading_idx - 1].strip()
                            if heading_idx + 1 < len(lines):
                                next_line = lines[heading_idx + 1].strip()

                        item_full_text = str(item.get("full_text") or "")
                        row_entry = _build_flagged_row(
                            doc_id=doc_id,
                            cik=str(row.get("cik") or ""),
                            accession=accession,
                            form_type=form_label,
                            filing_date=filing_date,
                            period_end=period_end,
                            item=item,
                            item_id=item_id,
                            item_part=str(item_part or ""),
                            item_missing_part=item_missing_part,
                            heading_line_clean=heading_line_clean,
                            heading_line_raw=heading_line_raw,
                            heading_idx=heading_idx,
                            heading_offset=heading_offset,
                            prefix_text=prefix_text,
                            prefix_kind=prefix_kind,
                            is_part_only_prefix=is_part_only_prefix,
                            is_crossref_prefix=is_crossref_prefix,
                            prev_line=prev_line,
                            next_line=next_line,
                            flags=flags,
                            embedded_hits=embedded_hits,
                            embedded_warn=embedded_warn,
                            embedded_fail=embedded_fail,
                            embedded_first_hit=embedded_first_hit,
                            embedded_first_flagged=embedded_first_flagged,
                            embedded_first_fail=embedded_first_fail,
                            leak_info=leak_info,
                            leak_pos=leak_pos,
                            leak_match=leak_match,
                            leak_context=leak_context,
                            leak_next_item_id=leak_next_item_id,
                            leak_next_heading=leak_next_heading,
                            item_full_text=item_full_text,
                            counts_for_target=counts_for_target,
                        )
                        flagged_rows.append(row_entry)
                        flagged_items_for_doc.append(row_entry)
                        writer.writerow(row_entry.to_dict(provenance))

                    if emit_manifest and manifest_filings_writer:
                        missing_core = sorted(core_items - item_ids_set)
                        missing_core_items_str = ",".join(missing_core)
                        missing_expected = sorted(expected_canonicals - extracted_canonicals)
                        missing_expected_str = ",".join(missing_expected)
                        filing_row = {
                            "doc_id": doc_id,
                            "accession": accession,
                            "cik": cik,
                            "filing_date": filing_date_str,
                            "period_end": period_end_str,
                            "form": form_label,
                            "n_items_extracted": len(items),
                            "items_extracted": ",".join(item_ids_ordered),
                            "missing_core_items": missing_core_items_str,
                            "missing_expected_canonicals": missing_expected_str,
                            "any_warn": filing_any_warn,
                            "any_fail": filing_any_fail,
                            "filing_exclusion_reason": filing_exclusion_reason,
                            "start_candidates_total": start_candidates_total_filing,
                            "start_candidates_toc_rejected": start_candidates_toc_rejected_filing,
                            "start_selection_unverified": start_selection_unverified_filing,
                            "truncated_successor_total": truncated_successor_total_filing,
                            "truncated_part_total": truncated_part_total_filing,
                        }
                        manifest_filings_writer.writerow(filing_row)
                        if config.sample_pass and config.sample_pass > 0:
                            # Sampling strata use quartiles x missing-core flag.
                            filing_rows_for_sampling.append(
                                {
                                    "row": filing_row,
                                    "missing_core": bool(missing_core),
                                    "any_fail": filing_any_fail,
                                    "filing_exclusion_reason": filing_exclusion_reason,
                                }
                            )

                    if flagged_items_for_doc:
                        safe_id = _safe_slug(doc_id or accession)
                        sample_path = config.samples_dir / f"{safe_id}.txt"
                        _write_sample_file(
                            sample_path,
                            filing_meta={
                                "doc_id": doc_id,
                                "cik": cik,
                                "accession": accession,
                                "form_type": form_label,
                                "filing_date": filing_date_str,
                                "period_end": period_end_str,
                            },
                            flagged_items=flagged_items_for_doc,
                            full_text=text,
                        )

    report = _build_diagnostics_report(
        rows=flagged_rows,
        total_filings=total_filings,
        total_items=total_items,
        total_part_only_prefix=total_part_only_prefix,
        start_candidates_total=start_candidates_total,
        start_candidates_toc_rejected_total=start_candidates_toc_rejected_total,
        start_selection_unverified_total=start_selection_unverified_total,
        truncated_successor_total=truncated_successor_total,
        truncated_part_total=truncated_part_total,
        parquet_dir=parquet_dir,
        max_examples=config.max_examples,
        provenance=provenance,
        extraction_regime=config.extraction_regime,
        diagnostics_regime=config.diagnostics_regime,
        item_breakdown=item_breakdown,
        missing_part_diagnostics=missing_part_diagnostics,
        focus_items=config.focus_items,
        report_item_scope=report_item_scope,
        target_set=target_set,
    )
    print(report)
    config.report_path.write_text(report, encoding="utf-8")
    print(f"\nReport written to {config.report_path}")

    if config.dump_missing_part_samples > 0:
        config.missing_part_samples_path.parent.mkdir(parents=True, exist_ok=True)
        with config.missing_part_samples_path.open(
            "w", newline="", encoding="utf-8"
        ) as samples_handle:
            writer = csv.DictWriter(
                samples_handle, fieldnames=MISSING_PART_SAMPLE_FIELDS
            )
            writer.writeheader()
            for entry in missing_part_samples:
                writer.writerow(
                    {
                        **entry,
                        "heading_index": _csv_value(entry.get("heading_index")),
                        "heading_offset": _csv_value(entry.get("heading_offset")),
                    }
                )
        print(
            f"Missing-part sample CSV written to {config.missing_part_samples_path}"
        )

    if config.sample_pass and config.sample_pass > 0:
        pass_rows = [
            entry
            for entry in filing_rows_for_sampling
            if not entry["any_fail"] and not entry["filing_exclusion_reason"]
        ]
        edges = _quartile_edges(
            [int(entry["row"]["n_items_extracted"]) for entry in pass_rows]
        )
        strata: dict[tuple[str, bool], list[dict[str, object]]] = defaultdict(list)
        for entry in pass_rows:
            row = entry["row"]
            bucket = _quartile_bucket(int(row["n_items_extracted"]), edges)
            strata[(bucket, entry["missing_core"])].append(row)

        sampled_rows = _stratified_sample(
            strata, sample_size=config.sample_pass, seed=config.sample_seed
        )
        sample_doc_ids = {row.get("doc_id", "") for row in sampled_rows}

        with config.sample_filings_path.open(
            "w", newline="", encoding="utf-8"
        ) as sample_filings:
            sample_writer = csv.DictWriter(sample_filings, fieldnames=MANIFEST_FILING_FIELDS)
            sample_writer.writeheader()
            for row in sampled_rows:
                sample_writer.writerow(row)

        if emit_manifest:
            csv.field_size_limit(10**7)
            with config.manifest_items_path.open(
                "r", newline="", encoding="utf-8"
            ) as manifest_items:
                reader = csv.DictReader(manifest_items)
                with config.sample_items_path.open(
                    "w", newline="", encoding="utf-8"
                ) as sample_items:
                    item_fields = reader.fieldnames or MANIFEST_ITEM_FIELDS
                    item_writer = csv.DictWriter(sample_items, fieldnames=item_fields)
                    item_writer.writeheader()
                    for row in reader:
                        if row.get("doc_id") in sample_doc_ids:
                            item_writer.writerow(row)

    if config.emit_html:
        if config.html_scope not in {"sample", "all"}:
            raise SystemExit(f"html-scope must be 'sample' or 'all' (got {config.html_scope})")
        if not emit_manifest:
            raise SystemExit("--emit-html requires --emit-manifest")

        csv.field_size_limit(10**7)
        manifest_rows = _read_csv_rows(config.manifest_filings_path)
        items_by_doc_id: dict[str, list[dict[str, object]]] = defaultdict(list)
        with config.manifest_items_path.open(
            "r", newline="", encoding="utf-8"
        ) as manifest_items:
            reader = csv.DictReader(manifest_items)
            for row in reader:
                doc_id = str(row.get("doc_id") or "")
                if doc_id:
                    items_by_doc_id[doc_id].append(row)
        metrics_by_doc = _compute_html_metrics(items_by_doc_id)

        rows_by_form: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in manifest_rows:
            normalized_form = _normalized_form_type(row.get("form"))
            if normalized_form not in {"10-K", "10-Q"}:
                continue
            row["form"] = normalized_form
            rows_by_form[normalized_form].append(row)

        form_entries: list[dict[str, object]] = []
        weights = normalize_sample_weights(config.html_sample_weights)
        weights_label = (
            f"{STATUS_PASS}={weights.get(STATUS_PASS, 0):.2f}, "
            f"{STATUS_WARNING}={weights.get(STATUS_WARNING, 0):.2f}, "
            f"{STATUS_FAIL}={weights.get(STATUS_FAIL, 0):.2f}"
        )
        for normalized_form in sorted(rows_by_form):
            form_rows = rows_by_form[normalized_form]
            eligible_rows = [
                row
                for row in form_rows
                if _passes_html_filters(
                    row,
                    metrics_by_doc=metrics_by_doc,
                    min_total_chars=config.html_min_total_chars,
                    min_largest_item_chars=config.html_min_largest_item_chars,
                    min_largest_item_chars_pct_total=config.html_min_largest_item_chars_pct_total,
                )
            ]
            pass_rows = [row for row in eligible_rows if classify_filing_status(row) == STATUS_PASS]
            warning_rows = [
                row for row in eligible_rows if classify_filing_status(row) == STATUS_WARNING
            ]
            fail_rows = [row for row in eligible_rows if classify_filing_status(row) == STATUS_FAIL]
            not_failed_rows = [
                row for row in eligible_rows if classify_filing_status(row) != STATUS_FAIL
            ]

            if config.html_scope == "all":
                index_rows = not_failed_rows
                scope_label = f"all (not failed: {len(index_rows)})"
            else:
                index_rows = sample_filings_by_status(
                    eligible_rows,
                    sample_size=DEFAULT_HTML_SAMPLE_SIZE,
                    seed=config.sample_seed,
                    weights=config.html_sample_weights,
                )
                scope_label = f"sample ({len(index_rows)})"

            selected_doc_ids = {str(row.get("doc_id") or "") for row in index_rows}
            items_by_filing: dict[str, list[dict[str, object]]] = defaultdict(list)
            if selected_doc_ids:
                for doc_id in selected_doc_ids:
                    items_by_filing[doc_id] = list(items_by_doc_id.get(doc_id, []))

            index_rows_by_doc_id = {
                str(row.get("doc_id") or ""): row
                for row in index_rows
                if row.get("doc_id")
            }
            normalized_by_doc_id: dict[str, str] = {}
            form_out_dir = config.html_out / normalized_form
            assets_root = form_out_dir / "assets"
            filing_assets_dir = assets_root / "filings"
            item_assets_dir = assets_root / "items"

            if selected_doc_ids:
                filing_assets_dir.mkdir(parents=True, exist_ok=True)
                item_assets_dir.mkdir(parents=True, exist_ok=True)
                accession_lookup = {
                    str(row.get("doc_id") or ""): str(row.get("accession") or "")
                    for row in index_rows
                }
                for entry in iter_parquet_filing_texts(
                    config.parquet_dir, selected_doc_ids, batch_size=config.batch_size
                ):
                    doc_id = entry.get("doc_id") or ""
                    if not doc_id:
                        continue
                    full_text = entry.get("full_text") or ""
                    if not full_text:
                        continue
                    accession = accession_lookup.get(doc_id) or entry.get("accession", "")
                    normalized = normalize_extractor_body(full_text)
                    normalized_by_doc_id[doc_id] = normalized
                    asset_name = f"{_safe_slug(doc_id)}_{_safe_slug(accession)}.txt"
                    asset_rel = f"assets/filings/{asset_name}"
                    (filing_assets_dir / asset_name).write_text(normalized, encoding="utf-8")
                    row = index_rows_by_doc_id.get(doc_id)
                    if row is not None:
                        row["filing_text_asset"] = asset_rel
                        row["filing_text_preview"] = _truncate_text(
                            normalized, limit=DEFAULT_HTML_FILING_PREVIEW_CHARS
                        )

                for doc_id, items in items_by_filing.items():
                    normalized = normalized_by_doc_id.get(doc_id)
                    if not normalized:
                        continue
                    row_accession = str(
                        index_rows_by_doc_id.get(doc_id, {}).get("accession") or ""
                    )
                    for item in items:
                        accession = str(item.get("accession") or row_accession)
                        item_part = str(item.get("item_part") or "")
                        item_id = str(item.get("item_id") or "")
                        content_start = _parse_int(item.get("content_start"), default=-1)
                        content_end = _parse_int(item.get("content_end"), default=-1)
                        if content_start < 0 or content_end <= content_start:
                            continue
                        item_text = normalized[content_start:content_end]
                        item_name = (
                            f"{_safe_slug(doc_id)}_{_safe_slug(accession)}_"
                            f"{_safe_slug(item_part)}_{_safe_slug(item_id)}_"
                            f"{content_start}_{content_end}.txt"
                        ).strip("_")
                        item_rel = f"assets/items/{item_name}"
                        (item_assets_dir / item_name).write_text(item_text, encoding="utf-8")
                        item["item_text_asset"] = item_rel
                        item["item_text_preview"] = _truncate_text(
                            item_text, limit=DEFAULT_HTML_ITEM_PREVIEW_CHARS
                        )

            metadata = {
                "pass_definition": "any_fail == False and filing_exclusion_reason is empty",
                "total_filings": len(eligible_rows),
                "total_items": sum(
                    _parse_int(row.get("n_items_extracted"), default=0)
                    for row in eligible_rows
                ),
                "total_pass_filings": len(pass_rows),
                "total_warning_filings": len(warning_rows),
                "total_fail_filings": len(fail_rows),
                "sample_size": len(index_rows),
                "sample_weights": weights_label,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "offset_basis": OFFSET_BASIS_EXTRACTOR_BODY,
            }
            write_html_audit(
                index_rows=index_rows,
                items_by_filing=items_by_filing,
                out_dir=form_out_dir,
                scope_label=scope_label,
                metadata=metadata,
            )
            form_entries.append(
                {
                    "form": normalized_form,
                    "index_path": f"{normalized_form}/index.html",
                    "total_filings": len(eligible_rows),
                    "pass_count": len(pass_rows),
                    "warn_count": len(warning_rows),
                    "fail_count": len(fail_rows),
                    "scope": scope_label,
                }
            )

        root_metadata = {
            "pass_definition": "any_fail == False and filing_exclusion_reason is empty",
            "total_filings": total_filings,
            "total_items": total_items,
            "sample_weights": weights_label,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "offset_basis": OFFSET_BASIS_EXTRACTOR_BODY,
        }
        write_html_audit_root_index(
            form_entries=form_entries,
            out_dir=config.html_out,
            metadata=root_metadata,
        )
        print(f"HTML audit written to {config.html_out / 'index.html'}")

    return {
        "total_filings": total_filings,
        "total_items": total_items,
        "total_flagged": len(flagged_rows),
        "total_internal_heading_leaks": sum(
            1 for row in flagged_rows if row.internal_heading_leak
        ),
    }


def run_boundary_regression(config: RegressionConfig) -> dict[str, int]:
    csv_path = config.csv_path
    if not csv_path.exists():
        archive_path = csv_path.parent / "archive" / csv_path.name
        if archive_path.exists():
            csv_path = archive_path
    parquet_dir = config.parquet_dir

    if not csv_path.exists():
        raise SystemExit(f"csv-path not found: {csv_path}")
    if not parquet_dir.exists():
        raise SystemExit(f"parquet-dir not found: {parquet_dir}")

    baseline = pl.read_csv(
        csv_path,
        schema_overrides={
            "item_id": pl.Utf8,
            "item_part": pl.Utf8,
            "item": pl.Utf8,
            "canonical_item": pl.Utf8,
            "exists_by_regime": pl.Utf8,
            "item_status": pl.Utf8,
            "heading_index": pl.Utf8,
            "heading_offset": pl.Utf8,
            "flags": pl.Utf8,
        },
        infer_schema_length=50_000,
    )
    doc_ids = set(baseline.get_column("doc_id").to_list())
    before_counts: Counter[str] = Counter()
    for flag_field in baseline.get_column("flags").to_list():
        for flag in _split_flags(flag_field):
            before_counts[flag] += 1
    before_first_embedded_counts: Counter[str] = Counter()
    if "first_embedded_classification" in baseline.columns:
        for value in baseline.get_column("first_embedded_classification").to_list():
            if value:
                before_first_embedded_counts[str(value)] += 1

    files = sorted(parquet_dir.glob("*_batch_*.parquet"))
    if config.max_files and config.max_files > 0:
        files = files[: config.max_files]
    if not files:
        raise SystemExit(f"No parquet batch files found in {parquet_dir}")

    after_counts: Counter[str] = Counter()
    examples_by_flag: dict[str, list[dict[str, str]]] = defaultdict(list)
    prefix_examples: dict[str, list[str]] = defaultdict(list)
    total_filings = 0
    total_items = 0
    total_flagged = 0
    total_part_only_prefix = 0
    start_candidates_total = 0
    start_candidates_toc_rejected_total = 0
    start_selection_unverified_total = 0
    truncated_successor_total = 0
    truncated_part_total = 0
    seen_doc_ids: set[str] = set()
    embedded_warn_total = 0
    embedded_fail_total = 0
    embedded_classification_counts: Counter[str] = Counter()
    after_first_embedded_counts: Counter[str] = Counter()
    embedded_fail_examples: list[dict[str, str | int]] = []

    want_cols = [
        "doc_id",
        "cik",
        "accession_number",
        "document_type_filename",
        "file_date_filename",
        "full_text",
    ]

    for file_path in files:
        pf = pq.ParquetFile(file_path)
        available = set(pf.schema.names)
        columns = [c for c in want_cols if c in available]
        for batch in pf.iter_batches(batch_size=8, columns=columns):
            tbl = pa.Table.from_batches([batch])
            df = pl.from_arrow(tbl)
            for row in df.iter_rows(named=True):
                doc_id = str(row.get("doc_id") or "")
                if doc_id not in doc_ids:
                    continue
                seen_doc_ids.add(doc_id)
                form_type = row.get("document_type_filename")
                normalized_form = _normalized_form_type(form_type)
                if normalized_form not in {"10-K", "10-Q"}:
                    continue

                total_filings += 1
                text = row.get("full_text") or ""
                if not text:
                    continue

                hdr = parse_header(text)
                filing_date = _parse_date(hdr.get("header_filing_date_str")) or _parse_date(
                    row.get("file_date_filename")
                )
                period_end = _parse_date(hdr.get("header_period_end_str"))

                items = extract_filing_items(
                    text,
                    form_type=form_type,
                    filing_date=filing_date,
                    period_end=period_end,
                    regime=True,
                    diagnostics=True,
                )
                if not items:
                    continue

                lines = _normalize_body_lines(text)
                total_items += len(items)
                item_id_sequence = [
                    _normalize_item_id(str(entry.get("item_id") or "")) for entry in items
                ]

                for item_idx, item in enumerate(items):
                    next_item_id = (
                        item_id_sequence[item_idx + 1]
                        if item_idx + 1 < len(item_id_sequence)
                        else None
                    )
                    nearby_slice = item_id_sequence[
                        item_idx + 1 : item_idx + 1 + EMBEDDED_NEARBY_ITEM_WINDOW
                    ]
                    nearby_item_ids = {item_id for item_id in nearby_slice if item_id}
                    heading_line_raw = item.get("_heading_line_raw")
                    if heading_line_raw is None:
                        heading_line_raw = item.get("_heading_line") or ""
                    heading_line_raw = str(heading_line_raw or "")

                    heading_line_clean = item.get("_heading_line")
                    if heading_line_clean is None or heading_line_clean == "":
                        heading_line_clean = item.get("_heading_line_clean") or heading_line_raw
                    heading_line_clean = str(heading_line_clean or "")

                    heading_idx = item.get("_heading_line_index")
                    heading_offset = item.get("_heading_offset")
                    item_id = str(item.get("item_id") or "")
                    item_part = item.get("item_part")
                    current_part_norm = _normalize_part(item_part) or _normalize_part(
                        _expected_part_for_item(item, normalized_form=normalized_form)
                    )
                    (
                        prefix_text,
                        prefix_kind,
                        is_part_only_prefix,
                        is_crossref_prefix,
                        is_midline_heading,
                    ) = _prefix_metadata(heading_line_raw, heading_offset)
                    if is_part_only_prefix and heading_offset and heading_offset > 0:
                        total_part_only_prefix += 1

                    leak_info = _find_internal_heading_leak(item.get("full_text") or "")
                    embedded_hits = _find_embedded_heading_hits(
                        item.get("full_text") or "",
                        current_item_id=item_id,
                        current_part=current_part_norm,
                        next_item_id=next_item_id,
                        nearby_item_ids=nearby_item_ids,
                    )
                    (
                        embedded_warn,
                        embedded_fail,
                        _embedded_first_hit,
                        _embedded_first_flagged,
                        _embedded_first_fail,
                        embedded_counts,
                    ) = _summarize_embedded_hits(embedded_hits)
                    if _embedded_first_flagged:
                        after_first_embedded_counts[_embedded_first_flagged.classification] += 1
                    start_candidates_total += int(item.get("_start_candidates_total") or 0)
                    start_candidates_toc_rejected_total += int(
                        item.get("_start_candidates_toc_rejected") or 0
                    )
                    if item.get("_start_selection_verified") is False:
                        start_selection_unverified_total += 1
                    if item.get("_truncated_successor_heading"):
                        truncated_successor_total += 1
                    if item.get("_truncated_part_boundary"):
                        truncated_part_total += 1
                    if embedded_warn:
                        embedded_warn_total += 1
                    if embedded_fail:
                        embedded_fail_total += 1
                    embedded_classification_counts.update(embedded_counts)
                    for hit in embedded_hits:
                        if hit.classification in EMBEDDED_FAIL_CLASSIFICATIONS:
                            pos_pct = _char_pos_pct(hit.char_pos, hit.full_text_len)
                            embedded_fail_pos_buckets[_char_pos_bucket(pos_pct)] += 1
                            embedded_fail_examples.append(
                                {
                                    "doc_id": doc_id,
                                    "item_id": item_id,
                                    "item_part": str(item_part or ""),
                                    "classification": hit.classification,
                                    "kind": hit.kind,
                                    "char_pos": hit.char_pos,
                                    "full_text_len": hit.full_text_len,
                                    "char_pos_pct": pos_pct,
                                    "to_end": max(hit.full_text_len - hit.char_pos, 0),
                                    "snippet": hit.snippet,
                                }
                            )

                    flags: list[str] = []
                    if heading_idx is None or heading_idx < 0:
                        flags.append("missing_heading_line")
                    else:
                        if _looks_like_toc_heading_line(lines, heading_idx):
                            flags.append("toc_like_heading")
                        if heading_offset and is_crossref_prefix:
                            flags.append("cross_ref_prefix")
                        if _compound_item_heading(heading_line_clean):
                            flags.append("compound_item_heading")

                    if is_midline_heading:
                        flags.append("midline_heading")

                    if _weak_heading_letter(item_id, heading_line_raw):
                        flags.append("weak_letter_heading")

                    if heading_idx is not None and heading_idx >= 0:
                        _, next_line = _non_empty_line(lines, heading_idx + 1)
                        if next_line:
                            if re.match(r"^\s*[ABC]\s*[\.\):\-]", next_line):
                                if item_id in {"1", "7", "9"} or _lettered_item(item_id):
                                    flags.append("split_letter_line")

                    expected_part = _expected_part_for_item(
                        item, normalized_form=normalized_form
                    )
                    if expected_part and not item_part:
                        flags.append("missing_part")
                    if item_part and expected_part and item_part != expected_part:
                        flags.append("part_mismatch")

                    if TOC_DOT_LEADER_PATTERN.search(heading_line_raw):
                        flags.append("dot_leader_heading")

                    if leak_info is not None:
                        flags.append("internal_heading_leak")

                    if embedded_warn:
                        flags.append("embedded_heading_warn")
                    if embedded_fail:
                        flags.append("embedded_heading_fail")

                    if not flags:
                        continue

                    flags = sorted(set(flags))
                    total_flagged += 1
                    for flag in flags:
                        after_counts[flag] += 1
                        if len(examples_by_flag[flag]) < config.sample_per_flag:
                            examples_by_flag[flag].append(
                                {
                                    "doc_id": doc_id,
                                    "item": str(item.get("item") or ""),
                                    "heading": heading_line_clean.strip(),
                                }
                            )
                        if flag in {"midline_heading", "cross_ref_prefix"}:
                            if len(prefix_examples[flag]) < 2:
                                prefix_examples[flag].append(prefix_text.strip())

    missing_doc_ids = doc_ids - seen_doc_ids

    lines_out: list[str] = []
    lines_out.append("Suspicious boundary regression (unified)")
    lines_out.append(f"CSV path: {csv_path}")
    lines_out.append(f"Parquet dir: {parquet_dir}")
    lines_out.append(f"Total filings processed: {total_filings}")
    lines_out.append(f"Total items extracted: {total_items}")
    lines_out.append(f"Total flagged items: {total_flagged}")
    lines_out.append(f"Part-only prefixes (informational): {total_part_only_prefix}")
    lines_out.append(f"Start candidates (total): {start_candidates_total}")
    lines_out.append(
        f"Start candidates rejected by TOC/SUMMARY: {start_candidates_toc_rejected_total}"
    )
    lines_out.append(f"Start selections unverified: {start_selection_unverified_total}")
    lines_out.append(f"Truncated by successor heading: {truncated_successor_total}")
    lines_out.append(f"Truncated by PART boundary: {truncated_part_total}")
    if missing_doc_ids:
        lines_out.append(f"Doc IDs missing in parquet: {len(missing_doc_ids)}")
    lines_out.append("")
    lines_out.append("Flag counts (before -> after):")
    all_flags = set(before_counts) | set(after_counts)
    for flag in sorted(all_flags, key=lambda f: (-after_counts.get(f, 0), f)):
        lines_out.append(
            f"  {flag}: {before_counts.get(flag, 0)} -> {after_counts.get(flag, 0)}"
        )
    lines_out.append("")
    lines_out.append("toc_start_misfire (first_embedded_classification, before -> after):")
    lines_out.append(
        "  toc_start_misfire: "
        f"{before_first_embedded_counts.get('toc_start_misfire', 0)} -> "
        f"{after_first_embedded_counts.get('toc_start_misfire', 0)}"
    )
    lines_out.append(
        "  toc_start_misfire_early: "
        f"{before_first_embedded_counts.get('toc_start_misfire_early', 0)} -> "
        f"{after_first_embedded_counts.get('toc_start_misfire_early', 0)}"
    )
    lines_out.append("")
    lines_out.append("Embedded heading summary (after scan):")
    lines_out.append(f"  embedded_heading_warn: {embedded_warn_total}")
    lines_out.append(f"  embedded_heading_fail: {embedded_fail_total}")
    lines_out.append(
        "  toc_start_misfire hits: "
        f"{embedded_classification_counts.get('toc_start_misfire', 0)}"
    )
    lines_out.append(
        "  toc_start_misfire_early hits: "
        f"{embedded_classification_counts.get('toc_start_misfire_early', 0)}"
    )
    if embedded_classification_counts:
        lines_out.append("  By classification (hits):")
        for classification, count in embedded_classification_counts.most_common():
            lines_out.append(f"    {classification}: {count}")
    else:
        lines_out.append("  By classification (hits): (none)")
    lines_out.append("  Embedded fail position buckets (char_pos_pct):")
    for bucket in ("0-0.05", "0.05-0.2", "0.2-0.8", "0.8-1.0"):
        lines_out.append(f"    {bucket}: {embedded_fail_pos_buckets.get(bucket, 0)}")
    earliest_embedded_fails = sorted(
        embedded_fail_examples,
        key=lambda entry: entry.get("char_pos", 0),
    )[: config.sample_per_flag]
    if earliest_embedded_fails:
        lines_out.append("  Earliest embedded heading fails:")
        for entry in earliest_embedded_fails:
            lines_out.append(
                _format_embedded_fail_entry(
                    entry, include_accession=False, include_form=False, prefix="    "
                )
            )
    else:
        lines_out.append("  Earliest embedded heading fails: (none)")
    latest_embedded_fails = sorted(
        embedded_fail_examples,
        key=lambda entry: entry.get("char_pos", 0),
        reverse=True,
    )[: config.sample_per_flag]
    if latest_embedded_fails:
        lines_out.append("  Latest embedded heading fails:")
        for entry in latest_embedded_fails:
            lines_out.append(
                _format_embedded_fail_entry(
                    entry, include_accession=False, include_form=False, prefix="    "
                )
            )
    else:
        lines_out.append("  Latest embedded heading fails: (none)")
    closest_to_end_fails = sorted(
        embedded_fail_examples,
        key=lambda entry: entry.get("to_end", 0),
    )[: config.sample_per_flag]
    if closest_to_end_fails:
        lines_out.append("  Closest-to-end embedded heading fails:")
        for entry in closest_to_end_fails:
            lines_out.append(
                _format_embedded_fail_entry(
                    entry, include_accession=False, include_form=False, prefix="    "
                )
            )
    else:
        lines_out.append("  Closest-to-end embedded heading fails: (none)")
    lines_out.append("")
    lines_out.append("Remaining examples by flag:")
    if examples_by_flag:
        for flag, examples in examples_by_flag.items():
            lines_out.append(f"  {flag}:")
            for ex in examples:
                lines_out.append(
                    f"    doc_id={ex['doc_id']} item={ex['item']} heading=\"{ex['heading']}\""
                )
    else:
        lines_out.append("  (no remaining flagged examples captured)")
    lines_out.append("")
    lines_out.append("Representative prefixes:")
    for flag in ("midline_heading", "cross_ref_prefix"):
        reps = prefix_examples.get(flag, [])
        if reps:
            lines_out.append(f"  {flag}:")
            for rep in reps:
                lines_out.append(f"    prefix=\"{rep}\"")
        else:
            lines_out.append(f"  {flag}: (none captured)")

    print("\n".join(lines_out))

    return {
        "total_filings": total_filings,
        "total_items": total_items,
        "total_flagged": total_flagged,
    }


def _read_csv_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [row for row in reader]


def _summary_stats(values: list[int]) -> dict[str, int]:
    if not values:
        return {"min": 0, "p25": 0, "median": 0, "p75": 0, "max": 0}
    ordered = sorted(values)
    q1, q2, q3 = _quartile_edges(ordered)
    return {
        "min": ordered[0],
        "p25": q1,
        "median": q2,
        "p75": q3,
        "max": ordered[-1],
    }


def _summarize_compare_metrics(
    *,
    manifest_filings: list[dict[str, object]],
    manifest_items: list[dict[str, object]],
    flagged_rows: list[dict[str, object]],
) -> dict[str, object]:
    status_counts: Counter[str] = Counter()
    n_items_values: list[int] = []
    truncated_successor_total = 0
    truncated_part_total = 0
    for row in manifest_filings:
        status_counts[classify_filing_status(row)] += 1
        n_items_values.append(_parse_int(row.get("n_items_extracted"), default=0))
        truncated_successor_total += _parse_int(
            row.get("truncated_successor_total"), default=0
        )
        truncated_part_total += _parse_int(row.get("truncated_part_total"), default=0)

    flag_counts: Counter[str] = Counter()
    for row in flagged_rows:
        for flag in _split_flags(str(row.get("flags") or "")):
            flag_counts[flag] += 1

    embedded_fail_counts: Counter[str] = Counter()
    missing_part_10q = 0
    for row in manifest_items:
        if _parse_bool(row.get("embedded_heading_fail")):
            classification = str(row.get("first_fail_classification") or "")
            if classification:
                embedded_fail_counts[classification] += 1
        if _normalized_form_type(row.get("form")) == "10-Q":
            if _parse_bool(row.get("item_missing_part")):
                missing_part_10q += 1

    return {
        "status_counts": dict(status_counts),
        "flag_counts": dict(flag_counts),
        "embedded_fail_counts": dict(embedded_fail_counts),
        "truncated_successor_total": truncated_successor_total,
        "truncated_part_total": truncated_part_total,
        "missing_part_10q": missing_part_10q,
        "n_items_extracted_summary": _summary_stats(n_items_values),
    }


def _diff_counter(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    keys = set(a) | set(b)
    return {key: int(b.get(key, 0)) - int(a.get(key, 0)) for key in sorted(keys)}


def run_boundary_comparison(
    base_config: DiagnosticsConfig,
    *,
    extraction_regime_b: Literal["legacy", "v2"] = "v2",
    diagnostics_regime_b: Literal["legacy", "v2"] = "v2",
    out_dir: Path | None = None,
) -> dict[str, dict[str, object]]:
    """
    Run diagnostics twice (legacy vs v2) on the same inputs and return summary deltas.
    """
    base_dir = out_dir or (base_config.out_path.parent / "boundary_compare")
    base_dir.mkdir(parents=True, exist_ok=True)

    def _build_config(label: str, extraction_regime: str, diagnostics_regime: str) -> DiagnosticsConfig:
        out_path = base_dir / f"suspicious_{label}.csv"
        report_path = base_dir / f"suspicious_report_{label}.txt"
        samples_dir = base_dir / f"samples_{label}"
        manifest_items_path = base_dir / f"manifest_items_{label}.csv"
        manifest_filings_path = base_dir / f"manifest_filings_{label}.csv"
        sample_filings_path = base_dir / f"sample_filings_{label}.csv"
        sample_items_path = base_dir / f"sample_items_{label}.csv"
        html_out = base_dir / f"html_{label}"
        return replace(
            base_config,
            out_path=out_path,
            report_path=report_path,
            samples_dir=samples_dir,
            manifest_items_path=manifest_items_path,
            manifest_filings_path=manifest_filings_path,
            sample_filings_path=sample_filings_path,
            sample_items_path=sample_items_path,
            html_out=html_out,
            emit_html=False,
            emit_manifest=True,
            sample_pass=0,
            extraction_regime=extraction_regime,
            diagnostics_regime=diagnostics_regime,
        )

    legacy_config = _build_config("legacy", "legacy", "legacy")
    v2_config = _build_config("v2", extraction_regime_b, diagnostics_regime_b)

    run_boundary_diagnostics(legacy_config)
    run_boundary_diagnostics(v2_config)

    legacy_metrics = _summarize_compare_metrics(
        manifest_filings=_read_csv_rows(legacy_config.manifest_filings_path),
        manifest_items=_read_csv_rows(legacy_config.manifest_items_path),
        flagged_rows=_read_csv_rows(legacy_config.out_path),
    )
    v2_metrics = _summarize_compare_metrics(
        manifest_filings=_read_csv_rows(v2_config.manifest_filings_path),
        manifest_items=_read_csv_rows(v2_config.manifest_items_path),
        flagged_rows=_read_csv_rows(v2_config.out_path),
    )

    delta = {
        "status_counts": _diff_counter(
            legacy_metrics["status_counts"], v2_metrics["status_counts"]
        ),
        "flag_counts": _diff_counter(
            legacy_metrics["flag_counts"], v2_metrics["flag_counts"]
        ),
        "embedded_fail_counts": _diff_counter(
            legacy_metrics["embedded_fail_counts"], v2_metrics["embedded_fail_counts"]
        ),
        "truncated_successor_total": (
            v2_metrics["truncated_successor_total"]
            - legacy_metrics["truncated_successor_total"]
        ),
        "truncated_part_total": (
            v2_metrics["truncated_part_total"] - legacy_metrics["truncated_part_total"]
        ),
        "missing_part_10q": v2_metrics["missing_part_10q"]
        - legacy_metrics["missing_part_10q"],
        "n_items_extracted_summary": {
            key: v2_metrics["n_items_extracted_summary"][key]
            - legacy_metrics["n_items_extracted_summary"][key]
            for key in v2_metrics["n_items_extracted_summary"]
        },
    }

    return {
        "legacy": legacy_metrics,
        "v2": v2_metrics,
        "delta": delta,
    }


def _is_pass_filing(row: dict[str, object]) -> bool:
    any_fail = _parse_bool(row.get("any_fail"))
    exclusion = str(row.get("filing_exclusion_reason") or "").strip()
    return not any_fail and not exclusion


def _sample_pass_rows(
    pass_rows: list[dict[str, object]],
    *,
    sample_size: int,
    seed: int,
) -> list[dict[str, object]]:
    if not pass_rows or sample_size <= 0:
        return []
    edges = _quartile_edges(
        [_parse_int(row.get("n_items_extracted"), default=0) for row in pass_rows]
    )
    strata: dict[tuple[str, bool], list[dict[str, object]]] = defaultdict(list)
    for row in pass_rows:
        bucket = _quartile_bucket(
            _parse_int(row.get("n_items_extracted"), default=0), edges
        )
        missing_core = bool(str(row.get("missing_core_items") or "").strip())
        strata[(bucket, missing_core)].append(row)
    return _stratified_sample(strata, sample_size=sample_size, seed=seed)


def _format_suspicious_items_line(items: list[DiagnosticsRow], *, max_items: int = 12) -> str:
    parts: list[str] = []
    for entry in items:
        flags = entry.flags
        item_key = entry.item or "UNKNOWN"
        if flags:
            parts.append(f"{item_key} ({','.join(flags)})")
        else:
            parts.append(item_key)
    trimmed = parts[:max_items]
    remainder = len(parts) - len(trimmed)
    line = "SUSPICIOUS_ITEMS: " + "; ".join(trimmed)
    if remainder > 0:
        line += f"; ... +{remainder} more"
    return line


def _truncate_text(text: str, *, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...[truncated]..."


def _write_sample_file(
    path: Path,
    *,
    filing_meta: dict[str, str],
    flagged_items: list[DiagnosticsRow],
    full_text: str,
) -> None:
    lines: list[str] = []
    lines.append(_format_suspicious_items_line(flagged_items))
    lines.append("")
    lines.append(
        "FILING_META: "
        f"doc_id={filing_meta.get('doc_id','')} "
        f"cik={filing_meta.get('cik','')} "
        f"accession={filing_meta.get('accession','')} "
        f"form={filing_meta.get('form_type','')} "
        f"filing_date={filing_meta.get('filing_date','')} "
        f"period_end={filing_meta.get('period_end','')}"
    )
    lines.append("")
    lines.append("EXTRACTED_ITEMS:")
    lines.append("")
    for entry in flagged_items:
        item_key = entry.item or "UNKNOWN"
        flags = entry.flags
        lines.append(f"---- ITEM {item_key} ({','.join(flags)}) ----")
        heading = entry.heading_line or ""
        lines.append(f"heading_line: {heading}")
        heading_raw = entry.heading_line_raw or ""
        if heading_raw:
            lines.append(f"heading_line_raw: {heading_raw}")
        lines.append(f"heading_index: {entry.heading_index}")
        lines.append(f"heading_offset: {entry.heading_offset}")
        lines.append(f"prefix_kind: {entry.prefix_kind}")
        lines.append(f"prefix_text: {entry.prefix_text}")
        lines.append(f"is_part_only_prefix: {entry.is_part_only_prefix}")
        lines.append(f"is_crossref_prefix: {entry.is_crossref_prefix}")
        if entry.filing_exclusion_reason:
            lines.append(f"filing_exclusion_reason: {entry.filing_exclusion_reason}")
        if entry.gij_omitted_items:
            lines.append(f"gij_omitted_items: {entry.gij_omitted_items}")
        if entry.leak_match:
            lines.append(
                "internal_heading_leak: "
                f"pos={entry.leak_pos} match=\"{entry.leak_match}\""
            )
            if entry.leak_next_item_id or entry.leak_next_heading:
                lines.append(
                    "leak_next_item: "
                    f"id={entry.leak_next_item_id} "
                    f"heading=\"{entry.leak_next_heading}\""
                )
            if entry.leak_context:
                lines.append(f"leak_context: {entry.leak_context}")
        lines.append(f"embedded_heading_warn: {entry.embedded_heading_warn}")
        lines.append(f"embedded_heading_fail: {entry.embedded_heading_fail}")
        if entry.first_hit_kind or entry.first_hit_classification:
            snippet = str(entry.first_hit_snippet).replace("\"", "'")
            lines.append(
                "first_hit: "
                f"kind={entry.first_hit_kind} class={entry.first_hit_classification} "
                f"item_id={entry.first_hit_item_id} part={entry.first_hit_part} "
                f"line_idx={entry.first_hit_line_idx} char_pos={entry.first_hit_char_pos} "
                f"snippet=\"{snippet}\""
            )
        if entry.first_embedded_kind or entry.first_embedded_classification:
            snippet = str(entry.first_embedded_snippet).replace("\"", "'")
            lines.append(
                "first_embedded: "
                f"kind={entry.first_embedded_kind} class={entry.first_embedded_classification} "
                f"item_id={entry.first_embedded_item_id} part={entry.first_embedded_part} "
                f"line_idx={entry.first_embedded_line_idx} char_pos={entry.first_embedded_char_pos} "
                f"snippet=\"{snippet}\""
            )
        if entry.first_fail_kind or entry.first_fail_classification:
            snippet = str(entry.first_fail_snippet).replace("\"", "'")
            lines.append(
                "first_fail: "
                f"kind={entry.first_fail_kind} class={entry.first_fail_classification} "
                f"item_id={entry.first_fail_item_id} part={entry.first_fail_part} "
                f"line_idx={entry.first_fail_line_idx} char_pos={entry.first_fail_char_pos} "
                f"snippet=\"{snippet}\""
            )
        if entry.embedded_hits:
            lines.append("embedded_hits:")
            for hit in entry.embedded_hits:
                snippet = hit.snippet.replace("\"", "'")
                lines.append(
                    "  "
                    f"kind={hit.kind} class={hit.classification} item_id={hit.item_id or ''} "
                    f"part={hit.part or ''} line_idx={hit.line_idx} char_pos={hit.char_pos} "
                    f"snippet=\"{snippet}\""
                )
        lines.append("")
        item_text = entry.item_full_text or ""
        lines.append(item_text.strip())
        lines.append("")
    lines.append("FULL_FILING_TEXT:")
    lines.append("")
    lines.append(full_text or "")
    path.write_text("\n".join(lines), encoding="utf-8")


def _build_parser() -> tuple[argparse.ArgumentParser, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(
        description="Unified diagnostics for suspicious item boundaries."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser("scan", help="Scan parquet batches and write flagged CSV/report.")
    scan.add_argument(
        "--parquet-dir",
        type=Path,
        default=DEFAULT_PARQUET_DIR,
        help="Directory containing parquet batch files.",
    )
    scan.add_argument(
        "--out-path",
        type=Path,
        default=DEFAULT_OUT_PATH,
        help="CSV output path for suspicious boundary rows.",
    )
    scan.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Text report output path.",
    )
    scan.add_argument(
        "--samples-dir",
        type=Path,
        default=DEFAULT_SAMPLES_DIR,
        help="Directory for per-filing sample text files.",
    )
    scan.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Parquet batch size for scanning filings.",
    )
    scan.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional cap on number of parquet files to scan (0 = no cap).",
    )
    scan.add_argument(
        "--max-examples",
        type=int,
        default=25,
        help="Max examples to include in the report.",
    )
    scan.add_argument(
        "--disable-embedded-verifier",
        action="store_true",
        help="Disable embedded heading verifier during scan.",
    )
    scan.add_argument(
        "--emit-manifest",
        action="store_true",
        help="Emit extraction manifest CSVs for items and filings.",
    )
    scan.add_argument(
        "--no-manifest",
        action="store_false",
        dest="emit_manifest",
        help="Disable manifest CSV outputs.",
    )
    scan.add_argument(
        "--sample-pass",
        type=int,
        default=None,
        help="Sample N pass filings for CSV review outputs (requires manifests).",
    )
    scan.add_argument(
        "--no-pass-sample",
        action="store_const",
        const=0,
        dest="sample_pass",
        help="Disable pass-sample CSV outputs.",
    )
    scan.add_argument(
        "--emit-html",
        action="store_true",
        help="Emit offline HTML audit pages (requires manifests).",
    )
    scan.add_argument(
        "--no-html",
        action="store_false",
        dest="emit_html",
        help="Disable HTML audit outputs.",
    )
    scan.add_argument(
        "--html-out",
        type=Path,
        default=DEFAULT_HTML_OUT_DIR,
        help="Output directory for HTML audit pages.",
    )
    scan.add_argument(
        "--html-scope",
        type=str,
        default=DEFAULT_HTML_SCOPE,
        choices=("sample", "all"),
        help="HTML scope: mixed-status sample or all not-failed filings.",
    )
    scan.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (CSV samples + HTML mixed-status sample).",
    )
    scan.add_argument(
        "--core-items",
        type=str,
        default=",".join(DEFAULT_CORE_ITEMS),
        help="Comma-separated core item IDs for missing_core_items.",
    )
    scan.add_argument(
        "--target-set",
        type=str,
        default=None,
        help="Restrict WARN/FAIL and missing-items to a target set (e.g., cohen2020).",
    )
    scan.add_argument(
        "--focus-items",
        type=str,
        default=None,
        help=(
            "Optional per-item focus set (e.g., "
            "\"10K:1A,7,7A,8,15;10Q:I:1,I:2,I:3,II:1,II:2\")."
        ),
    )
    scan.add_argument(
        "--report-item-scope",
        type=str,
        default=None,
        choices=("all", "target", "focus"),
        help=(
            "Scope for per-item breakdown sections: all items, target-set items, "
            "or focus-items only. Default is target if --target-set is provided, "
            "otherwise all."
        ),
    )
    scan.add_argument(
        "--html-min-total-chars",
        type=int,
        default=None,
        help="Minimum total extracted chars for filings included in HTML audit.",
    )
    scan.add_argument(
        "--html-min-largest-item-chars",
        type=int,
        default=None,
        help="Minimum largest-item length for filings included in HTML audit.",
    )
    scan.add_argument(
        "--html-min-largest-item-chars-pct-total",
        type=float,
        default=None,
        help="Minimum largest-item share of total chars for HTML audit.",
    )
    scan.add_argument(
        "--dump-missing-part-samples",
        type=int,
        default=0,
        help=(
            "Write a CSV of N sampled missing-part 10-Q items "
            f"(0 = disabled, default path: {DEFAULT_MISSING_PART_SAMPLES_PATH})."
        ),
    )
    scan.add_argument(
        "--extraction-regime",
        type=str,
        default="legacy",
        choices=("legacy", "v2"),
        help="Extraction regime to use (default: legacy).",
    )
    scan.add_argument(
        "--diagnostics-regime",
        type=str,
        default="legacy",
        choices=("legacy", "v2"),
        help="Diagnostics regime to use (default: legacy).",
    )
    scan.set_defaults(
        emit_manifest=True,
        emit_html=True,
        sample_pass=100,
    )

    regress = subparsers.add_parser(
        "regress", help="Re-run diagnostics for suspicious_boundaries_v5.csv."
    )
    regress.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="Path to suspicious_boundaries_v5.csv.",
    )
    regress.add_argument(
        "--parquet-dir",
        type=Path,
        default=DEFAULT_PARQUET_DIR,
        help="Directory containing parquet batch files.",
    )
    regress.add_argument(
        "--sample-per-flag",
        type=int,
        default=3,
        help="Number of remaining examples to print per flag.",
    )
    regress.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional cap on parquet files scanned (0 = no cap).",
    )

    return parser, scan


def main(argv: list[str] | None = None) -> None:
    parser, scan_parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "scan":
        if not args.emit_manifest and args.emit_html:
            scan_parser.error(
                "HTML output requires manifests. Remove --no-manifest or add --no-html."
            )
        if not args.emit_manifest and args.sample_pass and args.sample_pass > 0:
            scan_parser.error(
                "Pass-sample outputs require manifests. Remove --no-manifest or set "
                "--sample-pass 0 / use --no-pass-sample."
            )
        if args.report_item_scope == "focus" and not args.focus_items:
            scan_parser.error("--report-item-scope focus requires --focus-items.")
        if args.report_item_scope == "target" and not args.target_set:
            scan_parser.error("--report-item-scope target requires --target-set.")
        if args.report_item_scope is None:
            report_item_scope = "target" if args.target_set else "all"
        else:
            report_item_scope = args.report_item_scope
        config = DiagnosticsConfig(
            parquet_dir=args.parquet_dir,
            out_path=args.out_path,
            report_path=args.report_path,
            samples_dir=args.samples_dir,
            batch_size=args.batch_size,
            max_files=args.max_files,
            max_examples=args.max_examples,
            enable_embedded_verifier=not args.disable_embedded_verifier,
            emit_manifest=args.emit_manifest,
            sample_pass=args.sample_pass,
            sample_seed=args.seed,
            core_items=_parse_core_items_arg(args.core_items),
            target_set=_normalize_target_set(args.target_set),
            focus_items=parse_focus_items(args.focus_items),
            report_item_scope=report_item_scope,
            emit_html=args.emit_html,
            html_out=args.html_out,
            html_scope=args.html_scope,
            html_min_total_chars=args.html_min_total_chars,
            html_min_largest_item_chars=args.html_min_largest_item_chars,
            html_min_largest_item_chars_pct_total=args.html_min_largest_item_chars_pct_total,
            dump_missing_part_samples=args.dump_missing_part_samples,
            extraction_regime=args.extraction_regime,
            diagnostics_regime=args.diagnostics_regime,
        )
        run_boundary_diagnostics(config)
        return

    if args.command == "regress":
        config = RegressionConfig(
            csv_path=args.csv_path,
            parquet_dir=args.parquet_dir,
            sample_per_flag=args.sample_per_flag,
            max_files=args.max_files,
        )
        run_boundary_regression(config)
        return

    raise SystemExit(f"Unknown command: {args.command}")


__all__ = [
    "DiagnosticsConfig",
    "RegressionConfig",
    "DEFAULT_ROOT_CAUSES_PATH",
    "load_root_causes_text",
    "parse_focus_items",
    "run_boundary_comparison",
    "run_boundary_diagnostics",
    "run_boundary_regression",
    "main",
]

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from thesis_pkg.core.sec.filing_text import (
    ITEM_LINESTART_PATTERN,
    _default_part_for_item_id,
    _looks_like_toc_heading_line,
    _normalize_newlines,
    _prefix_is_bullet,
    _prefix_is_part_only,
    _prefix_looks_like_cross_ref,
    _repair_wrapped_headings,
    _strip_edgar_metadata,
    extract_filing_items,
    parse_header,
)
from thesis_pkg.core.sec.embedded_headings import (
    TOC_DOT_LEADER_PATTERN,
    EMBEDDED_CONTINUATION_PATTERN,
    EMBEDDED_CROSS_REF_PATTERN,
    EMBEDDED_FAIL_CLASSIFICATIONS,
    EMBEDDED_IGNORE_CLASSIFICATIONS,
    EMBEDDED_ITEM_PATTERN,
    EMBEDDED_ITEM_ROMAN_PATTERN,
    EMBEDDED_MAX_HITS,
    EMBEDDED_NEARBY_ITEM_WINDOW,
    EMBEDDED_PART_PATTERN,
    EMBEDDED_RESERVED_PATTERN,
    EMBEDDED_SELF_HIT_MAX_CHAR,
    EMBEDDED_SEPARATOR_PATTERN,
    EMBEDDED_TOC_CLUSTER_LOOKAHEAD,
    EMBEDDED_TOC_DOT_LEADER_PATTERN,
    EMBEDDED_TOC_HEADER_PATTERN,
    EMBEDDED_TOC_ITEM_ONLY_PATTERN,
    EMBEDDED_TOC_PART_ITEM_PATTERN,
    EMBEDDED_TOC_START_EARLY_MAX_CHAR,
    EMBEDDED_TOC_START_MISFIRE_MAX_CHAR,
    EMBEDDED_TOC_TRAILING_PAGE_PATTERN,
    EMBEDDED_TOC_WINDOW_HEADER_PATTERN,
    EMBEDDED_TOC_WINDOW_LINES,
    EMBEDDED_WARN_CLASSIFICATIONS,
    EmbeddedHeadingHit,
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


ITEM_MENTION_PATTERN = re.compile(r"\bITEM\s+(?:\d+|[IVXLCDM]+)[A-Z]?\b", re.IGNORECASE)
INTERNAL_PART_PATTERN = re.compile(r"(?m)^[ \t]*PART[ \t]+[IVX]+\b", re.IGNORECASE)
INTERNAL_ITEM_PATTERN = re.compile(r"(?m)^[ \t]*ITEM[ \t]+\d+[A-Z]?\b", re.IGNORECASE)
INTERNAL_HEADING_IGNORE_CHARS = 200
INTERNAL_HEADING_CONTEXT_CHARS = 150
PREFIX_KIND_BLANK = "blank"
PREFIX_KIND_PART_ONLY = "part_only"
PREFIX_KIND_BULLET = "bullet"
PREFIX_KIND_TEXTUAL = "textual"

DEFAULT_PARQUET_DIR = Path(
    r"C:\Users\erik9\Documents\SEC_Data\Data\Sample_Filings\parquet_batches"
)
DEFAULT_OUT_PATH = Path("results/suspicious_boundaries_v3_pre.csv")
DEFAULT_REPORT_PATH = Path("results/suspicious_boundaries_report_v3_pre.txt")
DEFAULT_SAMPLES_DIR = Path("results/Suspicious_Filings_Demo")
DEFAULT_CSV_PATH = Path("results/suspicious_boundaries_v5.csv")
DEFAULT_ROOT_CAUSES_PATH = Path("results/suspicious_root_causes_examples.txt")

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
    "filing_date",
    "period_end",
    "item_part",
    "item_id",
    "item",
    "canonical_item",
    "exists_by_regime",
    "item_status",
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
    canonical_item: str
    exists_by_regime: str | bool | None
    item_status: str
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
            "filing_date": self.filing_date,
            "period_end": self.period_end,
            "item_part": self.item_part,
            "item_id": self.item_id,
            "item": self.item,
            "canonical_item": self.canonical_item,
            "exists_by_regime": self.exists_by_regime,
            "item_status": self.item_status,
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


def _is_10k(form_type: str | None) -> bool:
    form = (form_type or "").upper().strip()
    return form.startswith("10-K") or form.startswith("10K")

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


def _expected_part_for_item(item: dict) -> str | None:
    canonical = item.get("canonical_item")
    if isinstance(canonical, str) and ":" in canonical:
        part = canonical.split(":", 1)[0].upper()
        if part in {"I", "II", "III", "IV"}:
            return part
    item_key = item.get("item")
    if isinstance(item_key, str) and ":" in item_key:
        part = item_key.split(":", 1)[0].upper()
        if part in {"I", "II", "III", "IV"}:
            return part
    item_id = item.get("item_id")
    if isinstance(item_id, str):
        return _default_part_for_item_id(item_id)
    return None


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
    body = _repair_wrapped_headings(_strip_edgar_metadata(_normalize_newlines(text)))
    return body.splitlines()


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
        canonical_item=item.get("canonical_item") or "",
        exists_by_regime=item.get("exists_by_regime"),
        item_status=item.get("item_status") or "",
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

    provenance = _build_provenance(config)
    out_headers = CSV_FIELDS

    total_filings = 0
    total_items = 0
    total_part_only_prefix = 0
    start_candidates_total = 0
    start_candidates_toc_rejected_total = 0
    start_selection_unverified_total = 0
    truncated_successor_total = 0
    truncated_part_total = 0
    # Keep a single row representation to avoid CSV/report/sample schema drift.
    flagged_rows: list[DiagnosticsRow] = []

    want_cols = [
        "doc_id",
        "cik",
        "accession_number",
        "document_type_filename",
        "file_date_filename",
        "full_text",
    ]

    with config.out_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=out_headers)
        writer.writeheader()

        for file_path in files:
            pf = pq.ParquetFile(file_path)
            available = set(pf.schema.names)
            columns = [c for c in want_cols if c in available]
            for batch in pf.iter_batches(batch_size=config.batch_size, columns=columns):
                tbl = pa.Table.from_batches([batch])
                df = pl.from_arrow(tbl)
                for row in df.iter_rows(named=True):
                    form_type = row.get("document_type_filename")
                    if not _is_10k(form_type):
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
                    flagged_items_for_doc: list[DiagnosticsRow] = []

                    doc_id = str(row.get("doc_id") or "")
                    accession = str(row.get("accession_number") or "")
                    form_label = str(form_type or "")

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
                            _expected_part_for_item(item)
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
                        leak_pos: int | str = ""
                        leak_match = ""
                        leak_context = ""
                        leak_next_item_id = ""
                        leak_next_heading = ""
                        embedded_hits = (
                            _find_embedded_heading_hits(
                                item.get("full_text") or "",
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

                        expected_part = _expected_part_for_item(item)
                        if expected_part and not item_part:
                            flags.append("missing_part")
                        if item_part and expected_part and item_part != expected_part:
                            flags.append("part_mismatch")

                        if TOC_DOT_LEADER_PATTERN.search(heading_line_raw):
                            flags.append("dot_leader_heading")

                        if leak_info is not None:
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

                        if not flags:
                            continue

                        flags = sorted(set(flags))

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
                        )
                        flagged_rows.append(row_entry)
                        flagged_items_for_doc.append(row_entry)
                        writer.writerow(row_entry.to_dict(provenance))

                    if flagged_items_for_doc:
                        doc_id = str(row.get("doc_id") or "")
                        accession = str(row.get("accession_number") or "")
                        safe_id = _safe_slug(doc_id or accession)
                        sample_path = config.samples_dir / f"{safe_id}.txt"
                        _write_sample_file(
                            sample_path,
                            filing_meta={
                                "doc_id": doc_id,
                                "cik": str(row.get("cik") or ""),
                                "accession": accession,
                                "form_type": str(form_type or ""),
                                "filing_date": filing_date.isoformat() if filing_date else "",
                                "period_end": period_end.isoformat() if period_end else "",
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
    )
    print(report)
    config.report_path.write_text(report, encoding="utf-8")
    print(f"\nReport written to {config.report_path}")

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
                if not _is_10k(form_type):
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
                        _expected_part_for_item(item)
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

                    expected_part = _expected_part_for_item(item)
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


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_") or "unknown"


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


def _build_parser() -> argparse.ArgumentParser:
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

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "scan":
        config = DiagnosticsConfig(
            parquet_dir=args.parquet_dir,
            out_path=args.out_path,
            report_path=args.report_path,
            samples_dir=args.samples_dir,
            batch_size=args.batch_size,
            max_files=args.max_files,
            max_examples=args.max_examples,
            enable_embedded_verifier=not args.disable_embedded_verifier,
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
    "run_boundary_diagnostics",
    "run_boundary_regression",
    "main",
]

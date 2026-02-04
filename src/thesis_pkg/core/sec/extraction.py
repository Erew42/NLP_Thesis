from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass, replace
from datetime import date, datetime
from typing import Literal

from . import embedded_headings
from .extraction_utils import _ItemBoundary
from .heuristics import (
    HEADING_CONF_HIGH,
    HEADING_CONF_LOW,
    HEADING_CONF_MED,
    _annotate_items_with_regime,
    _apply_heading_hygiene,
    _apply_high_confidence_truncation,
    _build_regime_item_titles_10k,
    _build_title_lookup,
    _detect_heading_style_10k,
    _detect_toc_line_ranges,
    _has_content_after,
    _heading_suffix_looks_like_prose,
    _heading_title_matches_item,
    _infer_front_matter_end_pos,
    _infer_part_for_item_id,
    _infer_toc_end_pos,
    _is_plausible_successor,
    _is_toc_entry_line,
    _line_has_compound_items,
    _line_ranges_to_mask,
    _looks_like_toc_heading_line,
    _match_numeric_dot_heading,
    _match_title_only_heading,
    _normalize_item_match,
    _pageish_line,
    _part_marker_is_heading,
    _prefix_is_part_only,
    _prefix_looks_like_cross_ref,
    _prefix_part_tail,
    _remove_pagination,
    _repair_wrapped_headings,
    _reserved_stub_end,
    _select_best_boundaries,
    _starts_with_lowercase_title,
    _trim_trailing_part_marker,
)
from .patterns import (
    ACC_HEADER_PATTERN,
    CIK_HEADER_PATTERN,
    DATE_HEADER_PATTERN,
    PERIOD_END_HEADER_PATTERN,
    EDGAR_BLOCK_TAG_PATTERN,
    EDGAR_TRAILING_TAG_PATTERN,
    EMPTY_ITEM_PATTERN,
    TOC_HEADER_LINE_PATTERN,
    TOC_MARKER_PATTERN,
    ITEM_WORD_PATTERN,
    PART_MARKER_PATTERN,
    PART_LINESTART_PATTERN,
    ITEM_CANDIDATE_PATTERN,
    COMBINED_PART_ITEM_PATTERN,
)
from .utilities import (
    _normalize_newlines,
    _parse_date,
    _prefix_is_bullet,
)

HEADER_SEARCH_LIMIT_DEFAULT = 5000


@dataclass(frozen=True)
class _PartMarker:
    start: int
    part: str
    line_index: int
    high_confidence: bool


def _strip_leading_header_block(full_text: str) -> str:
    """
    Remove the synthetic <Header>...</Header> block when present.
    Keeps all remaining text unchanged.
    """
    m = re.search(r"</\s*(sec-header|header)\s*>", full_text, flags=re.IGNORECASE)
    if not m:
        return full_text
    return full_text[m.end() :]

def _strip_edgar_metadata(full_text: str) -> str:
    """
    Remove EDGAR metadata blocks like <SEC-Header>, <Header>, <FileStats>, and <XML_Chars>.
    """
    if not full_text:
        return full_text
    text = EDGAR_BLOCK_TAG_PATTERN.sub("", full_text)
    text = _strip_leading_header_block(text)
    # Guard against truncated header tags that lack closing markers.
    text = EDGAR_TRAILING_TAG_PATTERN.sub("", text)
    return text

def _extract_fallback_items_10k(
    lines: list[str],
    line_starts: list[int],
    lookup: dict[str, str],
    *,
    front_matter_end_pos: int | None,
    max_item: int,
    allow_numeric: bool,
    allow_titles: bool,
) -> list["_ItemBoundary"]:
    boundaries: list[_ItemBoundary] = []
    current_part: str | None = None

    for i, line in enumerate(lines):
        if front_matter_end_pos is not None and line_starts[i] < front_matter_end_pos:
            continue
        if TOC_HEADER_LINE_PATTERN.match(line):
            continue
        if _is_toc_entry_line(line, lookup, max_item=max_item):
            continue

        m_part = PART_LINESTART_PATTERN.match(line)
        if m_part is not None and _part_marker_is_heading(line, m_part):
            current_part = m_part.group("part").upper()
            continue

        item_id: str | None = None
        if allow_numeric:
            item_id = _match_numeric_dot_heading(line, lookup, max_item=max_item)
        if item_id is None and allow_titles:
            item_id = _match_title_only_heading(line, lookup)
        if item_id is None:
            continue
        if item_id == "16":
            continue

        start_abs = line_starts[i]
        content_start_abs = line_starts[i] + len(line)
        boundaries.append(
            _ItemBoundary(
                start=start_abs,
                content_start=content_start_abs,
                item_part=current_part,
                item_id=item_id,
                line_index=i,
                confidence=HEADING_CONF_HIGH,
            )
        )

    return boundaries


def _scan_part_markers_v2(
    lines: list[str],
    line_starts: list[int],
    *,
    allowed_parts: set[str],
    scan_sparse_layout: bool,
    toc_mask: set[int],
) -> list[_PartMarker]:
    markers: list[_PartMarker] = []
    for i, line in enumerate(lines):
        if not line:
            continue
        if i in toc_mask:
            continue

        matches: list[re.Match[str]] = []
        if scan_sparse_layout:
            matches = list(PART_MARKER_PATTERN.finditer(line))
        else:
            match = PART_LINESTART_PATTERN.match(line)
            if match is not None:
                matches = [match]

        for match in matches:
            part = match.group("part").upper()
            if part not in allowed_parts:
                continue
            prefix = line[: match.start()]
            if prefix.strip() and not _prefix_is_bullet(prefix):
                continue
            high_confidence = _part_marker_is_heading(line, match)
            if not high_confidence and scan_sparse_layout and match.start() == 0:
                if ITEM_WORD_PATTERN.search(line) and not _looks_like_toc_heading_line(lines, i):
                    high_confidence = True
            if not high_confidence:
                continue
            markers.append(
                _PartMarker(
                    start=line_starts[i] + match.start(),
                    part=part,
                    line_index=i,
                    high_confidence=high_confidence,
                )
            )

    return markers


def _apply_part_by_position_v2(
    boundaries: list[_ItemBoundary],
    part_markers: list[_PartMarker],
) -> list[_ItemBoundary]:
    if not boundaries or not part_markers:
        return boundaries
    markers = sorted(
        [marker for marker in part_markers if marker.high_confidence],
        key=lambda marker: marker.start,
    )
    if not markers:
        return boundaries
    marker_starts = [marker.start for marker in markers]
    updated: list[_ItemBoundary] = []
    for boundary in boundaries:
        if boundary.item_part is not None:
            updated.append(boundary)
            continue
        idx = bisect_right(marker_starts, boundary.start) - 1
        if idx >= 0:
            updated.append(replace(boundary, item_part=markers[idx].part))
        else:
            updated.append(boundary)
    return updated


def extract_filing_items(
    full_text: str,
    *,
    form_type: str | None = None,
    filing_date: str | date | datetime | None = None,
    period_end: str | date | datetime | None = None,
    regime: bool = True,
    drop_impossible: bool = False,
    diagnostics: bool = False,
    repair_boundaries: bool = True,
    max_item_number: int = 20,
    extraction_regime: Literal["legacy", "v2"] = "legacy",
) -> list[dict[str, str | bool | None]]:
    """
    Extract filing item sections from `full_text`.

    Returns a list of dicts with:
      - item_part: roman numeral part when detected (e.g., 'I', 'II'); may be None
      - item_id: normalized item id (e.g., '1', '1A')
      - item: combined key '<part>:<id>' when part exists; for 10-Q without part uses '?:<id>' placeholder
      - full_text: extracted text for the item (pagination artifacts removed)
      - canonical_item: regime-stable meaning when available
      - exists_by_regime: True/False when regime rules can be evaluated, else None
      - item_status: active/reserved/optional/unknown
      - _heading_line (clean) / _heading_line_raw / _heading_line_index / _heading_offset when diagnostics=True
      - _heading_start/_heading_end/_content_start/_content_end offsets in extractor body when diagnostics=True
      - when drop_impossible=True, items with exists_by_regime == False are dropped
      - when repair_boundaries=True, high-confidence end-boundary truncation is applied
      - extraction_regime controls legacy vs v2-only behaviors (default legacy)

    The function does not emit TOC rows; TOC is only used internally to avoid false starts.
    """
    if not full_text:
        return []

    form = (form_type or "").strip().upper()
    if re.match(r"^10-?K[/-]A", form) or re.match(r"^10-?Q[/-]A", form):
        # Skip amended filings (10-K-A/10-Q-A) to avoid duplicate extraction.
        return []
    is_10k = form.startswith("10K") or form.startswith("10-K")
    is_10q = form.startswith("10Q") or form.startswith("10-Q")
    if is_10q:
        allowed_parts = {"I", "II"}
    else:
        # Default to 10-K style parts (I-IV). This also prevents accidental matches like "Part D".
        allowed_parts = {"I", "II", "III", "IV"}

    filing_date_parsed = _parse_date(filing_date)
    period_end_parsed = _parse_date(period_end)
    title_map: dict[str, list[str]] = {}
    title_lookup: dict[str, str] = {}
    if is_10k:
        title_map = _build_regime_item_titles_10k(
            filing_date=filing_date_parsed,
            period_end=period_end_parsed,
            enable_regime=bool(regime),
        )
        title_lookup = _build_title_lookup(title_map)
    # Regime-aware lookup only gates fallback/title-only matching; explicit ITEM headings still pass through.

    body = _normalize_newlines(full_text)
    body = _strip_edgar_metadata(body)
    body = _repair_wrapped_headings(body)
    lines = body.split("\n")
    gij_context = embedded_headings.detect_gij_context(lines)
    head = lines[: min(len(lines), 800)]
    head_nonempty_lens = [len(l) for l in head if l]
    max_line_len = max(head_nonempty_lens, default=0)
    avg_line_len = (sum(head_nonempty_lens) / len(head_nonempty_lens)) if head_nonempty_lens else 0.0
    # Some filings have relatively few lines but many long lines (e.g., HTML/PDF conversions).
    # Those often place headings mid-line; treat them like "sparse layout" even when max_line_len < 8000.
    sparse_layout = (
        max_line_len >= 8_000
        or (len(lines) <= 200 and max_line_len >= 2_000)
        or (len(lines) <= 300 and avg_line_len >= 1_000 and max_line_len >= 4_000)
    )
    toc_window_flags = embedded_headings._toc_window_flags(lines)
    toc_cache: dict[tuple[int, bool], bool] = {}

    # Precompute line start offsets (for slicing via absolute positions)
    line_starts: list[int] = []
    pos = 0
    for line in lines:
        line_starts.append(pos)
        pos += len(line) + 1  # '\n'

    front_matter_end_pos: int | None = None
    heading_style: str | None = None
    if is_10k:
        front_matter_end_pos = _infer_front_matter_end_pos(
            body,
            lines,
            line_starts,
            title_lookup,
            max_item=max_item_number,
        )
        heading_style = _detect_heading_style_10k(
            lines,
            line_starts,
            front_matter_end_pos,
            title_lookup,
            max_item=max_item_number,
        )

    # First pass: normal TOC masking + TOC cutoff. If that yields no items, do a relaxed pass
    # that disables TOC suppression (some filings include spurious TOC headers) and forces sparse
    # scanning to recover mid-line headings.
    attempt = 0
    boundaries: list[_ItemBoundary] = []
    while True:
        toc_ranges = [] if attempt else _detect_toc_line_ranges(lines)
        toc_mask = set() if attempt else _line_ranges_to_mask(toc_ranges)

        toc_end_pos = None if attempt else _infer_toc_end_pos(body)
        if toc_end_pos is None and toc_ranges and not attempt:
            max_end_line = max(e for _, e in toc_ranges)
            if max_end_line + 1 < len(line_starts):
                toc_end_pos = line_starts[max_end_line + 1]
            else:
                toc_end_pos = len(body)

        skip_until_pos: int | None = None
        if not attempt:
            candidates = [pos for pos in (front_matter_end_pos, toc_end_pos) if pos is not None]
            if candidates:
                skip_until_pos = max(candidates)

        scan_sparse_layout = True if attempt else sparse_layout
        boundaries = []
        current_part: str | None = None
        toc_part: str | None = None
        part_markers: list[_PartMarker] = []
        if extraction_regime == "v2" and is_10q:
            part_markers = _scan_part_markers_v2(
                lines,
                line_starts,
                allowed_parts=allowed_parts,
                scan_sparse_layout=scan_sparse_layout,
                toc_mask=toc_mask,
            )

        # Build candidates by scanning lines in order while tracking PART markers within each line.
        for i, line in enumerate(lines):
            line_in_toc_range = i in toc_mask
            if not line_in_toc_range:
                toc_part = None

            # Skip obvious page-header TOC repeats inside items (keep content otherwise).
            if TOC_HEADER_LINE_PATTERN.match(line):
                continue

            events: list[tuple[int, str, str | None, re.Match[str] | None]] = []
            line_has_item_word = ITEM_WORD_PATTERN.search(line) is not None

            combined_match = (
                COMBINED_PART_ITEM_PATTERN.match(line) if extraction_regime == "v2" else None
            )
            if combined_match:
                part = combined_match.group("part").upper()
                if part in allowed_parts:
                    item_match = ITEM_CANDIDATE_PATTERN.search(
                        line,
                        combined_match.start("item"),
                    )
                    if item_match and item_match.start() == combined_match.start("item"):
                        item_id, content_adjust = _normalize_item_match(
                            line,
                            item_match,
                            is_10k=is_10k,
                            max_item=max_item_number,
                        )
                        if item_id is not None and item_id != "16":
                            prefix = line[: item_match.start()]
                            if not _prefix_looks_like_cross_ref(prefix):
                                suffix = line[item_match.end() : item_match.end() + 64]
                                if not re.search(r"(?i)^\s*[\(\[]?\s*continued\b", suffix):
                                    if not re.match(
                                        r"(?i)^\s*[\(\[]\s*[a-z0-9]",
                                        suffix,
                                    ):
                                        is_toc_marker = TOC_MARKER_PATTERN.search(line) is not None
                                        item_word_count = 0
                                        if i < 800 or is_toc_marker or line_in_toc_range:
                                            item_word_count = len(
                                                ITEM_WORD_PATTERN.findall(line)
                                            )
                                        line_toc_like = _looks_like_toc_heading_line(lines, i)
                                        if (
                                            item_word_count >= 3
                                            and len(line) <= 5_000
                                            and (i < 800 or is_toc_marker or line_in_toc_range)
                                        ):
                                            line_toc_like = True
                                        if (
                                            is_toc_marker
                                            and item_word_count >= 1
                                            and len(line) <= 8_000
                                        ):
                                            line_toc_like = True
                                        content_after = False
                                        if line_in_toc_range or toc_window_flags[i]:
                                            content_after = _has_content_after(
                                                lines,
                                                i,
                                                toc_cache=toc_cache,
                                                toc_window_flags=toc_window_flags,
                                            )
                                        toc_window_like = False
                                        if not line_toc_like:
                                            if embedded_headings._toc_candidate_line(line):
                                                line_toc_like = True
                                            elif toc_window_flags[i] and not content_after:
                                                if not (
                                                    scan_sparse_layout and len(line) > 2_000
                                                ):
                                                    line_toc_like = True
                                                    toc_window_like = True
                                        compound_line = _line_has_compound_items(line)
                                        title_match = is_10k and _heading_title_matches_item(
                                            item_id, line, item_match
                                        )
                                        confidence = HEADING_CONF_HIGH
                                        if is_10k and not title_match and _heading_suffix_looks_like_prose(
                                            line[item_match.end() :]
                                        ):
                                            confidence = min(confidence, HEADING_CONF_MED)
                                        if compound_line:
                                            confidence = min(confidence, HEADING_CONF_LOW)
                                        if _pageish_line(line):
                                            confidence = min(confidence, HEADING_CONF_LOW)
                                        candidate_toc_like = line_toc_like
                                        if (
                                            candidate_toc_like
                                            and toc_window_like
                                            and scan_sparse_layout
                                            and item_match.start()
                                            > embedded_headings.EMBEDDED_TOC_START_EARLY_MAX_CHAR
                                        ):
                                            candidate_toc_like = False
                                        boundaries.append(
                                            _ItemBoundary(
                                                start=line_starts[i] + item_match.start(),
                                                content_start=(
                                                    line_starts[i]
                                                    + item_match.end()
                                                    + content_adjust
                                                ),
                                                item_part=part,
                                                item_id=item_id,
                                                line_index=i,
                                                confidence=confidence,
                                                in_toc_range=(
                                                    line_in_toc_range and not content_after
                                                ),
                                                toc_like_line=candidate_toc_like,
                                            )
                                        )
                                        if not line_in_toc_range:
                                            current_part = part
                                        continue

            if scan_sparse_layout:
                for m in PART_MARKER_PATTERN.finditer(line):
                    if not _part_marker_is_heading(line, m):
                        continue
                    part = m.group("part").upper()
                    if part in allowed_parts:
                        events.append((m.start(), "part", part, m))
            else:
                m = PART_LINESTART_PATTERN.match(line)
                if m is not None:
                    if not _part_marker_is_heading(line, m):
                        continue
                    part = m.group("part").upper()
                    if part in allowed_parts:
                        events.append((m.start(), "part", part, m))

            for m in ITEM_CANDIDATE_PATTERN.finditer(line):
                events.append((m.start(), "item", None, m))

            if not events:
                continue

            events.sort(key=lambda t: t[0])
            line_part: str | None = None
            line_part_end: int | None = None
            is_toc_marker = TOC_MARKER_PATTERN.search(line) is not None
            item_word_count = 0
            if i < 800 or is_toc_marker or line_in_toc_range:
                item_word_count = len(ITEM_WORD_PATTERN.findall(line))
            line_toc_like = _looks_like_toc_heading_line(lines, i)
            if item_word_count >= 3 and len(line) <= 5_000 and (i < 800 or is_toc_marker or line_in_toc_range):
                line_toc_like = True
            if is_toc_marker and item_word_count >= 1 and len(line) <= 8_000:
                line_toc_like = True
            content_after = False
            if line_in_toc_range or toc_window_flags[i]:
                content_after = _has_content_after(
                    lines, i, toc_cache=toc_cache, toc_window_flags=toc_window_flags
                )
            toc_window_like = False
            if not line_toc_like:
                if embedded_headings._toc_candidate_line(line):
                    line_toc_like = True
                elif toc_window_flags[i] and not content_after:
                    if not (scan_sparse_layout and len(line) > 2_000):
                        line_toc_like = True
                        toc_window_like = True
            compound_line = _line_has_compound_items(line)
            for _, kind, part, m in events:
                if kind == "part":
                    assert m is not None
                    line_part = part
                    line_part_end = m.end()
                    if line_in_toc_range:
                        if not line_has_item_word:
                            toc_part = part
                    elif not line_has_item_word:
                        current_part = part
                    continue

                assert m is not None
                item_id, content_adjust = _normalize_item_match(
                    line,
                    m,
                    is_10k=is_10k,
                    max_item=max_item_number,
                )
                if item_id is None:
                    continue
                if item_id == "16":
                    continue

                abs_start = line_starts[i] + m.start()

                # Heuristics to avoid cross-references:
                prefix = line[: m.start()]
                is_line_start = (
                    prefix.strip() == "" or _prefix_is_bullet(prefix) or _prefix_is_part_only(prefix)
                )
                prefix_part = _prefix_part_tail(prefix)
                part_near_item = (
                    line_part_end is not None and (m.start() - line_part_end) <= 60
                )
                if prefix_part:
                    part_near_item = True
                if _prefix_looks_like_cross_ref(prefix):
                    continue

                # Only accept headings that look like real section starts (line-start or 'PART .. ITEM ..').
                # This intentionally rejects mid-sentence cross-references like "Part II, Item 7 ...".
                accept = is_line_start or part_near_item
                midline = False
                if not accept and scan_sparse_layout:
                    # For filings with very few line breaks, headings often follow sentence punctuation.
                    # Keep this strict elsewhere to avoid mid-sentence cross-reference matches.
                    k = abs_start - 1
                    while k >= 0 and body[k].isspace():
                        k -= 1
                    prev_char = body[k] if k >= 0 else ""
                    if prev_char in ".:;!?":
                        if not _prefix_looks_like_cross_ref(prefix):
                            accept = True
                            midline = True

                if not accept:
                    continue

                if _starts_with_lowercase_title(line, m):
                    continue

                # Ignore "(continued)" headings; they are typically page-header repeats.
                # Only treat it as a continuation marker when it appears immediately after the heading,
                # not elsewhere later in the same (potentially very long) line.
                suffix = line[m.end() : m.end() + 64]
                if re.search(r"(?i)^\s*[\(\[]?\s*continued\b", suffix):
                    continue
                if re.match(r"(?i)^\s*[\(\[]\s*[a-z0-9]", suffix):
                    probe = suffix.strip().lstrip(" \t:-.([")
                    probe = probe.rstrip("])")
                    if not (EMPTY_ITEM_PATTERN.match(probe) and "reserved" in probe.lower()):
                        continue

                title_match = is_10k and _heading_title_matches_item(item_id, line, m)
                confidence = HEADING_CONF_HIGH if (is_line_start or part_near_item) else HEADING_CONF_MED
                if midline:
                    confidence = HEADING_CONF_MED if title_match else HEADING_CONF_LOW
                elif is_10k and (is_line_start or part_near_item):
                    if not title_match and _heading_suffix_looks_like_prose(line[m.end() :]):
                        confidence = min(confidence, HEADING_CONF_MED)
                if compound_line:
                    confidence = min(confidence, HEADING_CONF_LOW)
                if _pageish_line(line):
                    confidence = min(confidence, HEADING_CONF_LOW)

                item_part = toc_part if line_in_toc_range else current_part
                part_hint = line_part or prefix_part
                if part_near_item and part_hint:
                    item_part = part_hint
                    if confidence >= HEADING_CONF_HIGH or prefix_part:
                        current_part = part_hint

                start_abs = line_starts[i] + m.start()
                content_start_abs = line_starts[i] + m.end() + content_adjust
                candidate_toc_like = line_toc_like
                if (
                    candidate_toc_like
                    and toc_window_like
                    and scan_sparse_layout
                    and m.start() > embedded_headings.EMBEDDED_TOC_START_EARLY_MAX_CHAR
                ):
                    candidate_toc_like = False
                boundaries.append(
                    _ItemBoundary(
                        start=start_abs,
                        content_start=content_start_abs,
                        item_part=item_part,
                        item_id=item_id,
                        line_index=i,
                        confidence=confidence,
                        in_toc_range=line_in_toc_range and not content_after,
                        toc_like_line=candidate_toc_like,
                    )
                )

        if boundaries or attempt:
            break
        attempt += 1

    if not boundaries and is_10k:
        allow_numeric = heading_style == "B"
        allow_titles = heading_style in {"A", "B", "C", "UNKNOWN", None}
        boundaries = _extract_fallback_items_10k(
            lines,
            line_starts,
            title_lookup,
            front_matter_end_pos=front_matter_end_pos,
            max_item=max_item_number,
            allow_numeric=allow_numeric,
            allow_titles=allow_titles,
        )

    if not boundaries:
        return []

    if extraction_regime == "v2" and is_10q:
        boundaries = _apply_part_by_position_v2(boundaries, part_markers)

    boundaries, selection_meta = _select_best_boundaries(
        boundaries,
        lines=lines,
        body=body,
        skip_until_pos=skip_until_pos,
        filing_date=filing_date_parsed,
        period_end=period_end_parsed,
        is_10k=is_10k,
        max_item=max_item_number,
        gij_context=gij_context,
    )
    if not boundaries:
        return []

    def _boundary_end(idx: int, pool: list[_ItemBoundary]) -> int:
        end_pos = len(body)
        for j in range(idx + 1, len(pool)):
            cand = pool[j]
            if cand.confidence < HEADING_CONF_HIGH:
                continue
            if not _is_plausible_successor(
                pool[idx],
                cand,
                filing_date=filing_date_parsed,
                period_end=period_end_parsed,
                is_10k=is_10k,
            ):
                continue
            end_pos = cand.start
            break
        return end_pos

    boundary_ends = [_boundary_end(idx, boundaries) for idx in range(len(boundaries))]

    out_items: list[dict[str, str | None]] = []
    for idx, b in enumerate(boundaries):
        end = boundary_ends[idx]
        raw_chunk = body[b.content_start:end]
        lstripped_chunk = raw_chunk.lstrip(" \t:-")
        leading_trim = len(raw_chunk) - len(lstripped_chunk)
        content_start = b.content_start + leading_trim
        content_end = end
        chunk = lstripped_chunk
        chunk = _remove_pagination(chunk)
        chunk = _trim_trailing_part_marker(chunk)
        truncated_successor = False
        truncated_part = False
        item_id = b.item_id
        part = b.item_part
        inferred_part = (
            _infer_part_for_item_id(
                item_id,
                filing_date=filing_date_parsed,
                period_end=period_end_parsed,
                is_10k=is_10k,
            )
            if is_10k
            else None
        )
        part_for_detection = inferred_part or part
        raw_chunk: str | None = None
        raw_chunk_candidate = chunk
        if repair_boundaries:
            next_item_id = boundaries[idx + 1].item_id if idx + 1 < len(boundaries) else None
            next_part = boundaries[idx + 1].item_part if idx + 1 < len(boundaries) else None
            if next_item_id and is_10k and not next_part:
                next_part = _infer_part_for_item_id(
                    next_item_id,
                    filing_date=filing_date_parsed,
                    period_end=period_end_parsed,
                    is_10k=is_10k,
                )
            chunk, truncated_successor, truncated_part = _apply_high_confidence_truncation(
                chunk,
                next_item_id=next_item_id,
                current_part=part_for_detection,
                max_item=max_item_number,
            )
            if (truncated_successor or truncated_part) and raw_chunk is None:
                raw_chunk = raw_chunk_candidate
            leak_hit = embedded_headings.find_strong_leak(
                chunk,
                current_item_id=item_id,
                current_part=part_for_detection,
                next_item_id=next_item_id,
                next_part=next_part,
            )
            if leak_hit and 0 <= leak_hit.char_pos < len(chunk):
                if raw_chunk is None:
                    raw_chunk = raw_chunk_candidate
                chunk = chunk[: leak_hit.char_pos].rstrip()

        reserved_end = _reserved_stub_end(chunk)
        if reserved_end is not None:
            if raw_chunk is None:
                raw_chunk = raw_chunk_candidate
            chunk = chunk[:reserved_end].rstrip()

        item_missing_part = False
        if is_10k:
            if inferred_part and (part is None or part != inferred_part):
                part = inferred_part
            item_key = f"{part}:{item_id}" if part else item_id
        elif is_10q:
            if part:
                item_key = f"{part}:{item_id}"
            else:
                item_key = f"?:{item_id}" if item_id else "?:"
                item_missing_part = True
        else:
            item_key = f"{part}:{item_id}" if part else item_id

        record = {
            "item_part": part,
            "item_id": item_id,
            "item": item_key,
            "full_text": chunk,
        }
        if is_10q:
            record["item_missing_part"] = item_missing_part
        if gij_context.get("gij_asset_backed"):
            record["_filing_exclusion_reason"] = str(gij_context.get("gij_reason") or "")
            omitted_items = gij_context.get("gij_omitted_items")
            if isinstance(omitted_items, set) and omitted_items:
                record["_gij_omitted_items"] = sorted(omitted_items)
        if raw_chunk is not None:
            record["_raw_text"] = raw_chunk
        if diagnostics:
            dedup_part = b.item_part
            if is_10k:
                inferred = _infer_part_for_item_id(
                    b.item_id,
                    filing_date=filing_date_parsed,
                    period_end=period_end_parsed,
                    is_10k=is_10k,
                )
                if inferred:
                    dedup_part = inferred
            meta_key = (dedup_part, b.item_id)
            selection_meta_entry = selection_meta.get(meta_key, {})
            record["_start_candidates_total"] = selection_meta_entry.get(
                "start_candidates_total"
            )
            record["_start_candidates_toc_rejected"] = selection_meta_entry.get(
                "start_candidates_toc_rejected"
            )
            record["_start_selection_verified"] = selection_meta_entry.get(
                "start_selection_verified"
            )
            record["_truncated_successor_heading"] = truncated_successor
            record["_truncated_part_boundary"] = truncated_part
            idx = bisect_right(line_starts, b.start) - 1
            if 0 <= idx < len(lines):
                raw_line = lines[idx]
                record["_heading_line_raw"] = raw_line
                record["_heading_line"] = raw_line
                record["_heading_line_index"] = idx
                record["_heading_offset"] = b.start - line_starts[idx]
                record["_heading_start"] = b.start
                record["_heading_end"] = b.content_start
                record["_content_start"] = content_start
                record["_content_end"] = content_end
            else:
                record["_heading_line_raw"] = ""
                record["_heading_line"] = ""
                record["_heading_line_index"] = None
                record["_heading_offset"] = None
                record["_heading_start"] = None
                record["_heading_end"] = None
                record["_content_start"] = None
                record["_content_end"] = None
        out_items.append(record)

    _annotate_items_with_regime(
        out_items,
        form_type=form_type,
        filing_date=filing_date_parsed,
        period_end=period_end_parsed,
        enable_regime=bool(regime),
    )
    if diagnostics:
        # Heading hygiene only updates metadata fields; item boundaries are unchanged.
        _apply_heading_hygiene(out_items, title_map=title_map, is_10k=is_10k)
    if drop_impossible:
        out_items = [
            item for item in out_items if item.get("exists_by_regime") is not False
        ]
    return out_items

def parse_header(full_text: str, header_search_limit: int = HEADER_SEARCH_LIMIT_DEFAULT) -> dict:
    """
    Extract header metadata (CIKs, accession, filing date, period end) from the top of a filing.
    """
    header = full_text[:header_search_limit]

    header_ciks = CIK_HEADER_PATTERN.findall(header)
    header_ciks_int_set = {int(c) for c in header_ciks if c.isdigit()}

    date_match = DATE_HEADER_PATTERN.search(header)
    header_filing_date_str = date_match.group(1) if date_match else None

    period_match = PERIOD_END_HEADER_PATTERN.search(header)
    header_period_end_str = period_match.group(1) if period_match else None

    acc_match = ACC_HEADER_PATTERN.search(header)
    header_accession_str = acc_match.group(1) if acc_match else None

    primary_header_cik = header_ciks[0] if header_ciks else None
    secondary_ciks = header_ciks[1:] if len(header_ciks) > 1 else []

    return {
        "header_ciks_int_set": header_ciks_int_set,
        "header_filing_date_str": header_filing_date_str,
        "header_period_end_str": header_period_end_str,
        "header_accession_str": header_accession_str,
        "primary_header_cik": primary_header_cik,
        "secondary_ciks": secondary_ciks,
    }

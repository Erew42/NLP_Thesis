from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass


TOC_DOT_LEADER_PATTERN = re.compile(r"(?:\.{4,}|(?:\.\s*){4,})\s*\d{1,4}\s*$")
EMBEDDED_ITEM_PATTERN = re.compile(
    r"(?i)^[ \t]*ITEM[ \t]+(?P<num>\d{1,2})(?P<let>[A-Z])?\b"
)
# Corpus scan shows occasional ITEM I-style tokens; accept roman numerals for embedded checks.
EMBEDDED_ITEM_ROMAN_PATTERN = re.compile(r"(?i)^[ \t]*ITEM[ \t]+(?P<roman>[IVXLCDM]+)\b")
EMBEDDED_PART_PATTERN = re.compile(
    r"(?i)^[ \t]*PART[ \t]+(?P<roman>[IVX]+)[ \t]*[.\-\u2013\u2014:]?[ \t]*$"
)
EMBEDDED_CONTINUATION_PATTERN = re.compile(r"(?i)\b(?:continued|cont\.|concluded)\b")
EMBEDDED_CROSS_REF_PATTERN = re.compile(
    r"(?i)\b(?:see|refer to|refer back to|as discussed|as described|as set forth|as noted|"
    r"included in|incorporated by reference|incorporated herein by reference|of this form|"
    r"in this form|set forth in|pursuant to|under|in accordance with)\b"
)
EMBEDDED_RESERVED_PATTERN = re.compile(r"(?i)\[\s*reserved\s*\]")
EMBEDDED_TOC_DOT_LEADER_PATTERN = re.compile(r"(?:\.{8,}|(?:\.\s*){8,})")
EMBEDDED_TOC_TRAILING_PAGE_PATTERN = re.compile(r"\s+\d{1,4}\s*$")
EMBEDDED_TOC_HEADER_PATTERN = re.compile(
    r"(?i)\b(?:table\s+of\s+contents?|index|(?:form\s+)?10-?k\s+summary)\b"
)
EMBEDDED_TOC_WINDOW_HEADER_PATTERN = re.compile(
    r"(?i)^\s*(?:table\s+of\s+contents?|index|(?:form\s+)?10-?k\s+summary)\b"
)
EMBEDDED_TOC_WINDOW_LINES = 150
EMBEDDED_TOC_PART_ITEM_PATTERN = re.compile(
    r"(?i)^\s*PART\s+(?P<part>[IVX]+)\s*(?:[,/\-:;]\s*|\s+)?ITEM\s+"
    r"(?P<num>\d{1,2})(?P<let>[A-Z])?\b"
)
EMBEDDED_TOC_ITEM_ONLY_PATTERN = re.compile(r"(?i)^\s*ITEM\s+\d{1,2}[A-Z]?\b")
EMBEDDED_SEPARATOR_PATTERN = re.compile(r"^[\s\-=*]{3,}$")
EMBEDDED_SELF_HIT_MAX_CHAR = 10
EMBEDDED_MAX_HITS = 3
EMBEDDED_TOC_START_EARLY_MAX_CHAR = 500
EMBEDDED_TOC_START_MISFIRE_MAX_CHAR = 3000
EMBEDDED_TOC_CLUSTER_LOOKAHEAD = 8
EMBEDDED_NEARBY_ITEM_WINDOW = 3
EMBEDDED_IGNORE_CLASSIFICATIONS = {"same_item_continuation"}
# Warn-only to avoid false positives from TOC rows and cross-refs inside items.
EMBEDDED_WARN_CLASSIFICATIONS = {
    "same_item_duplicate",
    "toc_row",
    "cross_ref_line",
    "part_restart_unconfirmed",
    "glued_title_marker",
    "toc_start_misfire_early",
    "overlap_unconfirmed",
}
EMBEDDED_FAIL_CLASSIFICATIONS = {
    "true_overlap",
    "true_overlap_next_item",
    "part_restart",
    "toc_start_misfire",
}
EMBEDDED_STRONG_LEAK_CLASSIFICATIONS = {
    "true_overlap",
    "true_overlap_next_item",
    "part_restart",
}
ROMAN_ITEM_ID_MAP = {
    "I": "1",
    "II": "2",
    "III": "3",
    "IV": "4",
    "V": "5",
    "VI": "6",
    "VII": "7",
    "VIII": "8",
    "IX": "9",
    "X": "10",
    "XI": "11",
    "XII": "12",
    "XIII": "13",
    "XIV": "14",
    "XV": "15",
    "XVI": "16",
    "XVII": "17",
    "XVIII": "18",
    "XIX": "19",
    "XX": "20",
}


@dataclass(frozen=True)
class EmbeddedHeadingHit:
    kind: str
    classification: str
    item_id: str | None
    part: str | None
    line_idx: int
    char_pos: int
    full_text_len: int
    snippet: str


def _non_empty_line(lines: list[str], start: int, *, max_scan: int = 4) -> tuple[int | None, str | None]:
    idx = start
    while idx < len(lines) and idx < start + max_scan:
        if lines[idx].strip():
            return idx, lines[idx]
        idx += 1
    return None, None


def _normalize_item_id(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = re.sub(r"\s+", "", value).upper()
    return cleaned or None


def _normalize_part(value: str | None) -> str | None:
    if not value:
        return None
    part = value.strip().upper()
    if part in {"I", "II", "III", "IV"}:
        return part
    return None


def _item_id_to_int(item_id: str | None) -> int | None:
    if not item_id:
        return None
    cleaned = re.sub(r"\s+", "", item_id).upper()
    if cleaned in ROMAN_ITEM_ID_MAP:
        return int(ROMAN_ITEM_ID_MAP[cleaned])
    match = re.match(r"(\d+)", cleaned)
    if match:
        return int(match.group(1))
    return None


def _is_late_item(current_item_id: str | None, current_part: str | None) -> bool:
    if current_part and current_part.upper() == "IV":
        return True
    item_num = _item_id_to_int(current_item_id)
    return item_num is not None and item_num >= 10


def _item_id_from_match(match: re.Match[str]) -> str | None:
    num = match.groupdict().get("num")
    let = match.groupdict().get("let")
    roman = match.groupdict().get("roman")
    if num:
        return f"{num}{(let or '')}".upper()
    if roman:
        mapped = ROMAN_ITEM_ID_MAP.get(roman.upper())
        if mapped:
            return mapped
    return None


def _glued_title_marker_item_id(match: re.Match[str], line: str) -> str | None:
    num = match.groupdict().get("num")
    let = match.groupdict().get("let")
    if not num or not let:
        return None
    suffix = line[match.end() :]
    if re.match(r"\.[a-z]", suffix):
        return num
    return None


def _leading_ws_len(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def _line_snippet(line: str, *, max_len: int = 200) -> str:
    snippet = line.strip()
    if len(snippet) > max_len:
        return snippet[: max_len - 3].rstrip() + "..."
    return snippet


# Heuristics below guard against TOC/index artifacts, glued title markers, and line-start cross-references:
# - TOC windows keep index-style PART/ITEM rows from being treated as true overlaps.
# - Index-style detection catches comma/column TOC rows without dot leaders.
# - Line-start cross-ref checks prevent "SEE/REFER TO" headings from passing prose tests.
# - toc_start_misfire flags early PART I / ITEM 1-2 hits inside late items.
# - toc_start_misfire_early flags very-early TOC-like hits inside early items.
# - glued_title_marker catches "ITEM 4M.ine" artifacts before overlap decisions.


def _title_like_suffix(suffix: str) -> bool:
    if not suffix:
        return False
    if re.search(r"[.!?]\s*$", suffix):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'&-]*", suffix)
    if len(words) < 2:
        return False
    if suffix.isupper():
        return True
    uppercase = sum(1 for word in words if word[0].isupper())
    return uppercase / max(1, len(words)) >= 0.6


def _toc_index_style_line(line: str) -> bool:
    if not line or not line.strip():
        return False
    match = EMBEDDED_TOC_PART_ITEM_PATTERN.match(line) or EMBEDDED_TOC_ITEM_ONLY_PATTERN.match(line)
    if not match:
        return False
    suffix = line[match.end() :].strip()
    if not suffix:
        return False
    if EMBEDDED_TOC_DOT_LEADER_PATTERN.search(line) or EMBEDDED_TOC_TRAILING_PAGE_PATTERN.search(line):
        return True
    if re.search(r"(?i)\bPART\s+[IVX]+\s*,\s*ITEM\b", line):
        return True
    if re.search(r"(?i)\bITEM\s+\d{1,2}[A-Z]?\s*,", line):
        return True
    if re.search(r"\s{2,}", line[match.end() :]):
        return True
    return _title_like_suffix(suffix)


def _toc_candidate_line(line: str) -> bool:
    if not line or not line.strip():
        return False
    if TOC_DOT_LEADER_PATTERN.search(line) or EMBEDDED_TOC_DOT_LEADER_PATTERN.search(line):
        return True
    if EMBEDDED_TOC_TRAILING_PAGE_PATTERN.search(line):
        return bool(re.search(r"\s{2,}\d{1,4}\s*$", line))
    return False


def _toc_header_nearby(lines: list[str], idx: int, *, window: int = 5) -> bool:
    start = max(0, idx - window)
    end = min(len(lines), idx + window + 1)
    for j in range(start, end):
        if EMBEDDED_TOC_HEADER_PATTERN.search(lines[j]):
            return True
    return False


def _toc_clustered(lines: list[str], idx: int) -> bool:
    start = max(0, idx - 3)
    end = min(len(lines), idx + 4)
    candidate_count = 0
    for j in range(start, end):
        if _toc_entry_like(lines[j]):
            candidate_count += 1
            if candidate_count >= 2:
                return True
    return False


def _toc_entry_like(line: str) -> bool:
    return _toc_candidate_line(line) or _toc_index_style_line(line)


def _toc_like_line(
    lines: list[str],
    idx: int,
    cache: dict[tuple[int, bool], bool],
    toc_window_flags: list[bool] | None = None,
) -> bool:
    cache_key = (idx, toc_window_flags is not None)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    line = lines[idx]
    if not line or not line.strip():
        cache[cache_key] = False
        return False
    if _toc_entry_like(line):
        cache[cache_key] = True
        return True
    if toc_window_flags and toc_window_flags[idx]:
        if _line_starts_item(line) or EMBEDDED_PART_PATTERN.match(line) or EMBEDDED_TOC_PART_ITEM_PATTERN.match(line):
            cache[cache_key] = True
            return True
    if EMBEDDED_TOC_TRAILING_PAGE_PATTERN.search(line) and (
        _toc_header_nearby(lines, idx) or _toc_clustered(lines, idx)
    ):
        cache[cache_key] = True
        return True
    cache[cache_key] = False
    return False


def _cross_ref_like(suffix: str, *, next_line: str | None = None) -> bool:
    if suffix:
        probe = suffix.strip()
        if probe:
            probe = probe[:120]
            if EMBEDDED_CROSS_REF_PATTERN.search(probe) or EMBEDDED_TOC_HEADER_PATTERN.search(probe):
                return True
    if next_line:
        next_trim = next_line.strip()
        if next_trim and (
            EMBEDDED_CROSS_REF_PATTERN.search(next_trim)
            or EMBEDDED_TOC_HEADER_PATTERN.search(next_trim)
        ):
            return True
    return False


def _toc_window_flags(lines: list[str], *, window: int = EMBEDDED_TOC_WINDOW_LINES) -> list[bool]:
    flags = [False] * len(lines)
    remaining = 0
    for idx, line in enumerate(lines):
        if EMBEDDED_TOC_WINDOW_HEADER_PATTERN.search(line):
            remaining = window
        if remaining > 0:
            flags[idx] = True
            remaining -= 1
    return flags


def _prose_like_line(line: str, *, toc_like: bool) -> bool:
    if toc_like:
        return False
    stripped = line.strip()
    if not stripped:
        return False
    if EMBEDDED_SEPARATOR_PATTERN.match(stripped):
        return False
    letters = re.findall(r"[A-Za-z]", stripped)
    alpha_count = len(letters)
    if alpha_count == 0:
        return False
    has_lower = any(ch.islower() for ch in letters)
    if not has_lower and alpha_count < 30:
        return False
    if alpha_count >= 20 and has_lower:
        return True
    return bool(re.search(r"[.!?]\s*$", stripped))


def _sentence_like_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if not re.search(r"[.;]\s*$", stripped):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'&-]*", stripped)
    if len(words) < 8:
        return False
    if not any(ch.islower() for ch in stripped if ch.isalpha()):
        return False
    titlecase = sum(
        1 for word in words if len(word) > 1 and word[0].isupper() and word[1:].islower()
    )
    return (titlecase / len(words)) <= 0.6


def _confirm_prose_after(
    lines: list[str],
    start_idx: int,
    toc_cache: dict[tuple[int, bool], bool],
    toc_window_flags: list[bool] | None = None,
) -> bool:
    non_empty = 0
    for j in range(start_idx + 1, len(lines)):
        line = lines[j]
        if not line.strip():
            continue
        non_empty += 1
        if non_empty > 10:
            break
        toc_like = _toc_like_line(lines, j, toc_cache, toc_window_flags)
        if _line_starts_item(line) or EMBEDDED_PART_PATTERN.match(line) or toc_like:
            continue
        if _sentence_like_line(line):
            return True
    return False


def _line_starts_item(line: str) -> bool:
    return bool(
        EMBEDDED_ITEM_PATTERN.match(line)
        or EMBEDDED_ITEM_ROMAN_PATTERN.match(line)
        or EMBEDDED_TOC_PART_ITEM_PATTERN.match(line)
    )


def _toc_cluster_after(
    lines: list[str], idx: int, *, max_non_empty: int = EMBEDDED_TOC_CLUSTER_LOOKAHEAD
) -> bool:
    non_empty = 0
    heading_hits = 0
    for j in range(idx + 1, len(lines)):
        line = lines[j]
        if not line.strip():
            continue
        non_empty += 1
        if _line_starts_item(line) or EMBEDDED_PART_PATTERN.match(line):
            heading_hits += 1
            if heading_hits >= 2:
                return True
        if non_empty >= max_non_empty:
            break
    return False


def _confirm_part_restart(lines: list[str], start_idx: int) -> bool:
    non_empty = 0
    for j in range(start_idx + 1, len(lines)):
        if not lines[j].strip():
            continue
        non_empty += 1
        if non_empty > 8:
            break
        if _line_starts_item(lines[j]):
            return True
    return False


def _summarize_embedded_hits(
    hits: list[EmbeddedHeadingHit],
) -> tuple[
    bool,
    bool,
    EmbeddedHeadingHit | None,
    EmbeddedHeadingHit | None,
    EmbeddedHeadingHit | None,
    Counter[str],
]:
    warn = False
    fail = False
    first_hit: EmbeddedHeadingHit | None = hits[0] if hits else None
    first_flagged: EmbeddedHeadingHit | None = None
    first_fail: EmbeddedHeadingHit | None = None
    counts: Counter[str] = Counter()
    for hit in hits:
        if hit.classification in EMBEDDED_WARN_CLASSIFICATIONS:
            warn = True
        if hit.classification in EMBEDDED_FAIL_CLASSIFICATIONS:
            fail = True
        if hit.classification in EMBEDDED_WARN_CLASSIFICATIONS | EMBEDDED_FAIL_CLASSIFICATIONS:
            counts[hit.classification] += 1
            if first_flagged is None:
                first_flagged = hit
        if hit.classification in EMBEDDED_FAIL_CLASSIFICATIONS and first_fail is None:
            first_fail = hit
    return warn, fail, first_hit, first_flagged, first_fail, counts


def _find_embedded_heading_hits(
    full_text: str,
    *,
    current_item_id: str,
    current_part: str | None,
    next_item_id: str | None = None,
    nearby_item_ids: set[str] | None = None,
    max_hits: int = EMBEDDED_MAX_HITS,
) -> list[EmbeddedHeadingHit]:
    if not full_text:
        return []
    full_text_len = len(full_text)
    lines = full_text.splitlines(keepends=True)
    lines_noeol = [line.rstrip("\r\n") for line in lines]
    current_item = _normalize_item_id(current_item_id)
    current_part_norm = _normalize_part(current_part)
    next_item_norm = _normalize_item_id(next_item_id)
    toc_window_flags = _toc_window_flags(lines_noeol)
    is_late_item = _is_late_item(current_item, current_part_norm)
    hits: list[EmbeddedHeadingHit] = []
    non_ignored_hits = 0
    toc_cache: dict[tuple[int, bool], bool] = {}

    offset = 0
    for idx, (line, line_noeol) in enumerate(zip(lines, lines_noeol)):
        if not line_noeol.strip():
            offset += len(line)
            continue

        part_item_match = EMBEDDED_TOC_PART_ITEM_PATTERN.match(line_noeol)
        item_match = EMBEDDED_ITEM_PATTERN.match(line_noeol)
        roman_match = None
        part_match = None
        if not part_item_match and not item_match:
            roman_match = EMBEDDED_ITEM_ROMAN_PATTERN.match(line_noeol)
            if not roman_match:
                part_match = EMBEDDED_PART_PATTERN.match(line_noeol)

        if not part_item_match and not item_match and not roman_match and not part_match:
            offset += len(line)
            continue

        # Char offsets are computed against raw full_text (splitlines keepends), no normalization.
        char_pos = offset + _leading_ws_len(line_noeol)
        snippet = _line_snippet(line_noeol)
        toc_cluster_detected = _toc_cluster_after(lines_noeol, idx)

        if part_item_match or item_match or roman_match:
            match = part_item_match or item_match or roman_match
            embedded_item_id = _item_id_from_match(match)
            if not embedded_item_id:
                offset += len(line)
                continue
            glued_title_item_id = _glued_title_marker_item_id(match, line_noeol)
            glued_title_marker = False
            if glued_title_item_id:
                embedded_item_id = glued_title_item_id
                glued_title_marker = True
            embedded_item_id = embedded_item_id.upper()
            embedded_item_norm = _normalize_item_id(embedded_item_id)
            successor_match = bool(
                next_item_norm and embedded_item_norm and embedded_item_norm == next_item_norm
            )
            nearby_match = bool(
                embedded_item_norm and nearby_item_ids and embedded_item_norm in nearby_item_ids
            )
            embedded_part = None
            if part_item_match:
                embedded_part = part_item_match.group("part").upper()
            if (
                current_item
                and embedded_item_norm == current_item
                and char_pos <= EMBEDDED_SELF_HIT_MAX_CHAR
            ):
                offset += len(line)
                continue

            if current_item and embedded_item_norm == current_item:
                if EMBEDDED_CONTINUATION_PATTERN.search(line_noeol):
                    classification = "same_item_continuation"
                else:
                    classification = "same_item_duplicate"
            else:
                toc_like = _toc_like_line(lines_noeol, idx, toc_cache)
                toc_window_hit = toc_window_flags[idx]
                suffix = line_noeol[match.end() :].strip(" \t:-.")
                _, next_line = _non_empty_line(lines_noeol, idx + 1, max_scan=3)
                strong_prose = _confirm_prose_after(lines_noeol, idx, toc_cache, toc_window_flags)
                cross_ref = (not toc_like) and _cross_ref_like(suffix, next_line=next_line)
                strong_overlap_evidence = bool(
                    EMBEDDED_RESERVED_PATTERN.search(line_noeol)
                    or (strong_prose and _title_like_suffix(suffix))
                )

                if toc_window_hit and not strong_prose:
                    classification = "toc_row"
                elif toc_like:
                    classification = "toc_row"
                elif cross_ref:
                    classification = "cross_ref_line"
                elif EMBEDDED_RESERVED_PATTERN.search(line_noeol):
                    classification = "true_overlap"
                elif strong_prose:
                    classification = "true_overlap"
                else:
                    classification = "toc_row" if _toc_entry_like(line_noeol) else "cross_ref_line"

            if (
                not hits
                and char_pos <= EMBEDDED_TOC_START_EARLY_MAX_CHAR
                and (
                    toc_window_flags[idx]
                    or EMBEDDED_TOC_DOT_LEADER_PATTERN.search(line_noeol)
                    or EMBEDDED_TOC_TRAILING_PAGE_PATTERN.search(line_noeol)
                    or toc_cluster_detected
                )
            ):
                classification = "toc_start_misfire_early"
            elif char_pos <= EMBEDDED_TOC_START_EARLY_MAX_CHAR and toc_cluster_detected:
                classification = "toc_row"

            if (
                not hits
                and is_late_item
                and char_pos <= EMBEDDED_TOC_START_MISFIRE_MAX_CHAR
                and (
                    (embedded_part or "").upper() == "I"
                    or (_item_id_to_int(embedded_item_id) in {1, 2})
                )
            ):
                classification = "toc_start_misfire"

            if glued_title_marker:
                if classification in {"toc_start_misfire", "toc_start_misfire_early"}:
                    pass
                elif successor_match or EMBEDDED_RESERVED_PATTERN.search(line_noeol):
                    pass
                else:
                    classification = "glued_title_marker"

            if classification == "true_overlap":
                if not nearby_match and not strong_overlap_evidence:
                    classification = "overlap_unconfirmed"

            if (
                successor_match
                and classification in {"true_overlap", "overlap_unconfirmed"}
                and char_pos > EMBEDDED_TOC_START_EARLY_MAX_CHAR
                and not toc_cluster_detected
            ):
                classification = "true_overlap_next_item"

            hit = EmbeddedHeadingHit(
                kind="item",
                classification=classification,
                item_id=embedded_item_id,
                part=embedded_part,
                line_idx=idx,
                char_pos=char_pos,
                full_text_len=full_text_len,
                snippet=snippet,
            )
        else:
            embedded_part = part_match.group("roman").upper()
            toc_like = _toc_like_line(lines_noeol, idx, toc_cache)
            toc_window_hit = toc_window_flags[idx]
            strong_prose = _confirm_prose_after(lines_noeol, idx, toc_cache, toc_window_flags)
            if toc_window_hit and not strong_prose:
                classification = "toc_row"
            elif toc_like:
                classification = "toc_row"
            elif _confirm_part_restart(lines_noeol, idx):
                if current_part_norm and embedded_part != current_part_norm:
                    classification = "part_restart"
                else:
                    classification = "part_restart_unconfirmed"
            else:
                classification = "toc_row" if _toc_entry_like(line_noeol) else "part_restart_unconfirmed"

            if (
                not hits
                and is_late_item
                and char_pos <= EMBEDDED_TOC_START_MISFIRE_MAX_CHAR
                and embedded_part == "I"
            ):
                classification = "toc_start_misfire"
            if (
                not hits
                and char_pos <= EMBEDDED_TOC_START_EARLY_MAX_CHAR
                and toc_cluster_detected
            ):
                classification = "toc_start_misfire_early"
            elif char_pos <= EMBEDDED_TOC_START_EARLY_MAX_CHAR and toc_cluster_detected:
                classification = "toc_row"

            hit = EmbeddedHeadingHit(
                kind="part",
                classification=classification,
                item_id=None,
                part=embedded_part,
                line_idx=idx,
                char_pos=char_pos,
                full_text_len=full_text_len,
                snippet=snippet,
            )

        hits.append(hit)
        if hit.classification not in EMBEDDED_IGNORE_CLASSIFICATIONS:
            non_ignored_hits += 1
            if non_ignored_hits >= max_hits:
                break

        offset += len(line)

    return hits


def find_strong_leak(
    full_text: str,
    *,
    current_item_id: str,
    current_part: str | None,
    next_item_id: str | None,
    next_part: str | None,
    max_hits: int = EMBEDDED_MAX_HITS,
) -> EmbeddedHeadingHit | None:
    if not full_text:
        return None
    hits = _find_embedded_heading_hits(
        full_text,
        current_item_id=current_item_id,
        current_part=current_part,
        next_item_id=next_item_id,
        nearby_item_ids=None,
        max_hits=max_hits,
    )
    next_item_norm = _normalize_item_id(next_item_id)
    next_part_norm = _normalize_part(next_part)
    for hit in hits:
        if hit.classification not in EMBEDDED_STRONG_LEAK_CLASSIFICATIONS:
            continue
        if hit.kind == "item":
            if next_item_norm and _normalize_item_id(hit.item_id) == next_item_norm:
                return hit
        elif hit.kind == "part":
            if next_part_norm and _normalize_part(hit.part) == next_part_norm:
                return hit
    return None

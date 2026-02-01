from __future__ import annotations

import re
from datetime import date, datetime

from . import embedded_headings
from .extraction_utils import _ItemBoundary
from .patterns import (
    TOC_MARKER_PATTERN,
    TOC_HEADER_LINE_PATTERN,
    TOC_INDEX_MARKER_PATTERN,
    SUMMARY_MARKER_PATTERN,
    SUMMARY_HEADER_LINE_PATTERN,
    INDEX_HEADER_LINE_PATTERN,
    START_TOC_SUMMARY_MARKER_PATTERN,
    ITEM_WORD_PATTERN,
    PART_MARKER_PATTERN,
    PART_LINESTART_PATTERN,
    ITEM_CANDIDATE_PATTERN,
    ITEM_LINESTART_PATTERN,
    NUMERIC_DOT_HEADING_PATTERN,
    DOT_LEADER_PATTERN,
    TOC_DOT_LEADER_PATTERN,
    CONTINUED_PATTERN,
    ITEM_MENTION_PATTERN,
    PART_ONLY_PREFIX_PATTERN,
    PART_PREFIX_TAIL_PATTERN,
    CROSS_REF_PREFIX_PATTERN,
    CROSS_REF_PART_PATTERN,
    COVER_PAGE_MARKER_PATTERN,
    FORM_10K_PATTERN,
    PAGE_HYPHEN_PATTERN,
    PAGE_NUMBER_PATTERN,
    PAGE_ROMAN_PATTERN,
    PAGE_OF_PATTERN,
    EMPTY_ITEM_PATTERN,
    START_TRAILING_PAGE_NUMBER_PATTERN,
    PROSE_SENTENCE_BREAK_PATTERN,
    CROSS_REF_SUFFIX_START_PATTERN,
    PART_REF_PATTERN,
    HEADING_TAIL_STRONG_PATTERN,
    HEADING_TAIL_SOFT_PATTERN,
    COVER_PAGE_ANCHOR_PATTERN,
    _WRAPPED_TABLE_OF_CONTENTS_PATTERN,
    _WRAPPED_PART_PATTERN,
    _WRAPPED_ITEM_PATTERN,
    _WRAPPED_ITEM_LETTER_PATTERN,
    _ITEM_SUFFIX_PAREN_PATTERN,
    TOC_ENTRY_PATTERN,
)
from .regime import (
    ITEM_TITLES_10K,
    ITEM_TITLES_10K_BY_CANONICAL,
    _ALLOWED_ITEM_LETTERS_10K,
    _ITEM_REGIME_BY_ID,
    _ITEM_REGIME_ITEMS,
    _ITEM_REGIME_LEGACY,
    _ITEM_REGIME_SPEC,
)
from .utilities import (
    _default_part_for_item_id,
    _normalize_newlines,
    _parse_date,
    _prefix_is_bullet,
    _roman_to_int,
)

HEADING_CONF_LOW = 0
HEADING_CONF_MED = 1
HEADING_CONF_HIGH = 2
START_CANDIDATE_MAX = 8
START_SCORE_LOOKAHEAD_LINES = 20
START_SCORE_TOC_MARKER_WINDOW = 40
START_LOOKAHEAD_GUARD_NONEMPTY = 20
START_LOOKAHEAD_GUARD_MAX_CHARS = 500
START_SCORE_STRUCTURAL_HIGH_RATIO = 0.35
START_SCORE_STRUCTURAL_LOW_RATIO = 0.15
START_SCORE_TRAILING_PAGE_MIN = 3

def _normalize_item_id(num_raw: str, let_raw: str | None, *, max_item: int = 20) -> str | None:
    """
    Normalize item number + optional letter to a canonical ID like '1', '1A', ...
    Filters out non-filing items (e.g., Item 405) via max_item.
    """
    num_raw = (num_raw or "").strip()
    if not num_raw:
        return None

    n: int | None
    if num_raw.isdigit():
        n = int(num_raw)
    else:
        n = _roman_to_int(num_raw)

    if n is None or n <= 0 or n > max_item:
        return None

    letter = (let_raw or "").strip().upper()
    if letter and not re.fullmatch(r"[A-Z]", letter):
        letter = ""
    return f"{n}{letter}" if letter else str(n)

def _normalize_heading_text(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    text = text.replace("\u2019", "'")
    text = DOT_LEADER_PATTERN.sub(" ", text)
    text = re.sub(r"\s+\d{1,4}\s*$", "", text)
    text = text.upper()
    text = text.replace("&", " AND ")
    text = re.sub(r"[^A-Z0-9' ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _build_title_lookup(mapping: dict[str, list[str]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for item_id, titles in mapping.items():
        for title in titles:
            norm = _normalize_heading_text(title)
            if norm:
                lookup[norm] = item_id
    return lookup

_ITEM_TITLE_LOOKUP_10K = _build_title_lookup(ITEM_TITLES_10K)
_ITEM_TITLES_10K_NORM: dict[str, set[str]] = {}
for _item_id, _titles in ITEM_TITLES_10K.items():
    normed = {_normalize_heading_text(t) for t in _titles}
    _ITEM_TITLES_10K_NORM[_item_id] = {t for t in normed if t}

def _resolve_item_key(item_part: str | None, item_id: str | None) -> str | None:
    if not item_id:
        return None
    part = item_part or _default_part_for_item_id(item_id)
    if not part:
        return None
    return f"{part}:{item_id}"

def _item_letter_allowed_10k(num: int | None, let: str | None) -> bool:
    if not let:
        return True
    if num is None:
        return False
    allowed = _ALLOWED_ITEM_LETTERS_10K.get(num)
    return allowed is not None and let in allowed

def _normalize_item_match(
    line: str,
    match: re.Match[str],
    *,
    is_10k: bool,
    max_item: int,
) -> tuple[str | None, int]:
    item_id = _normalize_item_id(
        match.group("num"),
        match.group("let"),
        max_item=max_item,
    )
    if item_id is None:
        return None, 0

    content_adjust = 0
    if is_10k and match.group("let"):
        base_id = _normalize_item_id(
            match.group("num"),
            None,
            max_item=max_item,
        )
        base_num = int(base_id) if base_id and base_id.isdigit() else None
        let = match.group("let").upper()
        glued = match.end() < len(line) and line[match.end()].isalpha()
        if not _item_letter_allowed_10k(base_num, let):
            if glued and base_id:
                item_id = base_id
                content_adjust = -1
            else:
                return None, 0
        elif glued and _glued_title_matches_base(base_id, let, line, match):
            item_id = base_id or item_id
            content_adjust = -1

    suffix = line[match.end() :]
    suffix_match = re.match(r"\s*\(\s*[A-Z]\s*\)\s*", suffix, flags=re.IGNORECASE)
    if suffix_match:
        content_adjust += suffix_match.end()

    return item_id, content_adjust

def _glued_title_from_line(line: str, match: re.Match[str]) -> str:
    rest = line[match.end() :]
    rest = re.split(r"\bITEM\b|\bPART\b", rest, maxsplit=1, flags=re.IGNORECASE)[0]
    return rest

def _glued_title_matches_base(
    base_item_id: str | None,
    let: str,
    line: str,
    match: re.Match[str],
) -> bool:
    if not base_item_id:
        return False
    titles = _ITEM_TITLES_10K_NORM.get(base_item_id)
    if not titles:
        return False
    rest = _glued_title_from_line(line, match)
    candidate = f"{let}{rest}".strip()
    if not candidate:
        return False
    norm = _normalize_heading_text(candidate)
    if norm in titles:
        return True
    # Allow truncated wrapped headings to match the base title prefix.
    if len(norm) >= 6:
        return any(title.startswith(norm) for title in titles)
    return False

def _starts_with_lowercase_title(line: str, match: re.Match[str]) -> bool:
    suffix = line[match.end() :]
    if suffix and suffix[0].islower():
        if match.end() > 0 and line[match.end() - 1].isspace():
            return True
        return False
    for ch in suffix:
        if ch.isalpha():
            return ch.islower()
    return False

def _part_marker_is_heading(line: str, match: re.Match[str]) -> bool:
    """
    Accept PART markers that look like true headings, not cross-references.
    """
    prefix = line[: match.start()]
    if prefix.strip() and not _prefix_is_bullet(prefix):
        return False

    suffix = line[match.end() :]
    item_match = ITEM_CANDIDATE_PATTERN.search(suffix)
    if item_match is not None:
        between = suffix[: item_match.start()]
        if "," in between:
            return False
        if re.search(r"[A-Za-z0-9]", between):
            return False
        return item_match.start() <= 10

    trimmed = suffix.strip()
    if not trimmed:
        return True
    if re.search(r"[\.!?]", trimmed):
        return False
    if len(trimmed) > 80:
        return False
    letters = [ch for ch in trimmed if ch.isalpha()]
    if not letters:
        return True
    upper = sum(1 for ch in letters if ch.isupper())
    if upper / len(letters) >= 0.8:
        return True
    if len(trimmed.split()) <= 4 and not re.search(r"\bsee\b|\brefer\b", trimmed, re.IGNORECASE):
        return True
    return False

def _pageish_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    return bool(
        PAGE_NUMBER_PATTERN.match(s)
        or PAGE_HYPHEN_PATTERN.match(s)
        or PAGE_ROMAN_PATTERN.match(s)
        or PAGE_OF_PATTERN.match(s)
    )

def _prefix_is_part_only(prefix: str) -> bool:
    if not prefix:
        return False
    return bool(PART_ONLY_PREFIX_PATTERN.match(prefix))

def _prefix_part_tail(prefix: str) -> str | None:
    if not prefix:
        return None
    m = PART_PREFIX_TAIL_PATTERN.search(prefix)
    if not m:
        return None
    return m.group("part").upper()

def _prefix_looks_like_cross_ref(prefix: str) -> bool:
    if not prefix or not prefix.strip():
        return False
    tail = prefix.strip()[-80:]
    if CROSS_REF_PREFIX_PATTERN.search(tail) or CROSS_REF_PART_PATTERN.search(tail):
        return True
    return False

def _line_has_compound_items(line: str) -> bool:
    if not line:
        return False
    return len(ITEM_MENTION_PATTERN.findall(line)) >= 2

def _heading_title_matches_item(
    item_id: str | None,
    line: str,
    match: re.Match[str],
) -> bool:
    if not item_id:
        return False
    titles = _ITEM_TITLES_10K_NORM.get(item_id)
    if not titles:
        return False
    rest = line[match.end() :].strip(" \t:-.")
    rest = re.sub(r"^\(\s*[A-Z]\s*\)\s*", "", rest)
    if not rest:
        return False
    norm = _normalize_heading_text(rest)
    if not norm:
        return False
    if norm in titles:
        return True
    if len(norm) >= 6:
        return any(title.startswith(norm) for title in titles)
    return False

def _heading_suffix_looks_like_prose(suffix: str) -> bool:
    if not suffix:
        return False
    head = suffix.strip().lstrip(" \t:-")
    if not head:
        return False
    head = head[:160]
    if CROSS_REF_SUFFIX_START_PATTERN.search(head):
        return True
    if ITEM_WORD_PATTERN.search(head):
        return True
    if PART_REF_PATTERN.search(head):
        return True
    if PROSE_SENTENCE_BREAK_PATTERN.search(head):
        return True
    words = re.findall(r"[A-Za-z]+", head)
    if len(words) >= 10:
        lower_initial = sum(1 for word in words if word and word[0].islower())
        if lower_initial / len(words) >= 0.6:
            return True
    if head.count(",") >= 2 and len(words) >= 8:
        return True
    return False

def _heading_title_candidates(
    item_id: str | None,
    canonical_item: str | None,
    *,
    title_map: dict[str, list[str]],
    is_10k: bool,
) -> list[str]:
    if not item_id and not canonical_item:
        return []
    candidates: list[str] = []
    if is_10k and canonical_item:
        candidates.extend(ITEM_TITLES_10K_BY_CANONICAL.get(canonical_item, []))
    if is_10k and item_id:
        candidates.extend(title_map.get(item_id, []))
        candidates.extend(ITEM_TITLES_10K.get(item_id, []))
    seen: set[str] = set()
    ordered: list[str] = []
    for title in candidates:
        if title not in seen:
            ordered.append(title)
            seen.add(title)
    return ordered

def _find_heading_title_span(line: str, title: str) -> tuple[int, int] | None:
    norm = _normalize_heading_text(title)
    if not norm:
        return None
    tokens = norm.split()
    if not tokens:
        return None
    parts: list[str] = []
    for token in tokens:
        if token == "AND":
            parts.append(r"(?:AND|&)")
            continue
        token_re = re.escape(token).replace("'", "['\u2019]")
        parts.append(token_re)
    pattern = r"[^A-Za-z0-9]+".join(parts)
    match = re.search(pattern, line, flags=re.IGNORECASE)
    if not match:
        return None
    return match.start(), match.end()

def _truncate_heading_to_title(segment: str, titles: list[str]) -> str | None:
    if not segment or not titles:
        return None
    for title in sorted(titles, key=len, reverse=True):
        span = _find_heading_title_span(segment, title)
        if span is None:
            continue
        return segment[: span[1]].rstrip()
    return None

def _truncate_heading_tail_fallback(segment: str) -> str:
    if not segment:
        return segment
    trimmed = segment.rstrip()
    words = re.findall(r"[A-Za-z]+", trimmed)
    short_title = len(trimmed) <= 80 and len(words) <= 12

    strong = HEADING_TAIL_STRONG_PATTERN.search(trimmed)
    if strong:
        return trimmed[: strong.start()].rstrip(" \t:-.")

    if not short_title:
        for match in HEADING_TAIL_SOFT_PATTERN.finditer(trimmed):
            token = match.group(0)
            if not any(ch.isupper() for ch in token):
                continue
            prefix = trimmed[: match.start()].rstrip()
            if len(prefix) < 25 or len(re.findall(r"[A-Za-z]+", prefix)) < 3:
                continue
            return prefix.rstrip(" \t:-.")

    return trimmed

def _clean_heading_line(
    raw_line: str,
    heading_offset: int | None,
    *,
    item_id: str | None,
    canonical_item: str | None,
    title_map: dict[str, list[str]],
    is_10k: bool,
) -> str:
    if not raw_line:
        return ""
    start = heading_offset if isinstance(heading_offset, int) and heading_offset >= 0 else 0
    if start > len(raw_line):
        start = 0
    segment = raw_line[start:].lstrip()
    if not segment:
        return ""

    titles = _heading_title_candidates(
        item_id,
        canonical_item,
        title_map=title_map,
        is_10k=is_10k,
    )
    truncated = _truncate_heading_to_title(segment, titles)
    if truncated is not None:
        return truncated.strip()

    return _truncate_heading_tail_fallback(segment).strip()

def _looks_like_toc_heading_line(
    lines: list[str],
    idx: int,
    *,
    max_early: int = 400,
    max_line_len: int = 240,
) -> bool:
    line = lines[idx]
    if not line:
        return False
    line_trim = line.strip()
    if not line_trim:
        return False
    # Avoid masking long, sparse-layout lines where TOC entries and real headings can coexist.
    if len(line_trim) > 2000:
        return False

    if DOT_LEADER_PATTERN.search(line):
        return True

    if len(line_trim) <= max_line_len and re.search(r"\s+\d{1,4}\s*$", line):
        if idx <= max_early:
            return True
        start = max(0, idx - 3)
        end = min(len(lines), idx + 4)
        for j in range(start, end):
            if TOC_INDEX_MARKER_PATTERN.search(lines[j]):
                return True
        return False

    if idx > max_early:
        return False

    j = idx + 1
    max_scan = min(len(lines), idx + 5)
    while j < max_scan and lines[j].strip() == "":
        j += 1
    if j < len(lines) and _pageish_line(lines[j]):
        return len(line_trim) <= max_line_len
    return False

def _has_content_after(
    lines: list[str],
    start_idx: int,
    *,
    toc_cache: dict[tuple[int, bool], bool],
    toc_window_flags: list[bool] | None,
    max_non_empty: int = 4,
) -> bool:
    non_empty = 0
    for j in range(start_idx + 1, len(lines)):
        line = lines[j]
        if not line.strip():
            continue
        non_empty += 1
        if non_empty > max_non_empty:
            break
        toc_like = embedded_headings._toc_like_line(lines, j, toc_cache, toc_window_flags)
        if embedded_headings._line_starts_item(line) or embedded_headings.EMBEDDED_PART_PATTERN.match(line) or toc_like:
            return False
        if re.search(r"[A-Za-z]", line):
            return True
    return False

def _evaluate_regime_validity(
    validity: list[dict],
    *,
    filing_date: date | None,
    period_end: date | None,
) -> tuple[dict | None, bool]:
    if not validity:
        return None, True

    missing_date = False
    for entry in validity:
        trigger = entry.get("trigger")
        if trigger == "effective_date":
            date_value = filing_date
        elif trigger == "fiscal_year_end_ge":
            # Use filing_date as a proxy when period_end is missing.
            date_value = period_end or filing_date
        else:
            continue

        if date_value is None:
            missing_date = True
            continue

        start = _parse_date(entry.get("start"))
        end = _parse_date(entry.get("end"))
        if start is not None and date_value < start:
            continue
        if end is not None and date_value >= end:
            continue
        return entry, True

    if missing_date:
        return None, False
    return None, True

def _canonical_for_entry(entry: dict, fallback: str | None) -> str | None:
    validity = entry.get("validity") or []
    canonicals = {v.get("canonical") for v in validity if v.get("canonical")}
    if len(canonicals) == 1:
        return next(iter(canonicals))
    return fallback

def _status_from_entry(entry: dict, canonical: str | None) -> str:
    if canonical and canonical.endswith("_RESERVED"):
        return "reserved"
    status = entry.get("status")
    if status == "optional":
        return "optional"
    if status == "active":
        return "active"
    if status == "time_varying":
        return "active"
    return "unknown"

def _find_regime_matches_by_item_id(
    item_id: str | None,
    *,
    filing_date: date | None,
    period_end: date | None,
    exclude_key: str | None = None,
) -> tuple[list[tuple[str, dict, dict]], bool]:
    if not item_id:
        return [], False
    matches: list[tuple[str, dict, dict]] = []
    missing_date = False
    for key, entry in _ITEM_REGIME_BY_ID.get(item_id, []):
        if exclude_key and key == exclude_key:
            continue
        match, decidable = _evaluate_regime_validity(
            entry.get("validity", []),
            filing_date=filing_date,
            period_end=period_end,
        )
        if not decidable:
            missing_date = True
            continue
        if match is not None:
            matches.append((key, entry, match))
    return matches, missing_date

def _infer_part_for_item_id(
    item_id: str | None,
    *,
    filing_date: date | None,
    period_end: date | None,
    is_10k: bool,
) -> str | None:
    if not is_10k or not item_id:
        return None
    matches, missing_date = _find_regime_matches_by_item_id(
        item_id,
        filing_date=filing_date,
        period_end=period_end,
    )
    if matches:
        parts = {
            (entry.get("part") or key.split(":", 1)[0]).upper()
            for key, entry, _ in matches
            if isinstance(entry.get("part") or key, str)
        }
        if len(parts) == 1:
            return next(iter(parts))
    if missing_date:
        return _default_part_for_item_id(item_id)
    return _default_part_for_item_id(item_id)

def _item_order_key(
    item_part: str | None,
    item_id: str | None,
    *,
    filing_date: date | None,
    period_end: date | None,
    is_10k: bool,
) -> tuple[int, int, int, str] | None:
    if not item_id:
        return None
    part = item_part
    if part is None and is_10k:
        part = _infer_part_for_item_id(
            item_id,
            filing_date=filing_date,
            period_end=period_end,
            is_10k=is_10k,
        )
    part = part or _default_part_for_item_id(item_id)
    part_order = {"I": 1, "II": 2, "III": 3, "IV": 4}.get(part or "", 99)

    m = re.match(r"^(?P<num>\d{1,2})(?P<let>[A-Z])?$", item_id)
    if not m:
        return None
    num = int(m.group("num"))
    let = m.group("let")
    let_val = (ord(let.upper()) - ord("A") + 1) if let else 0
    return (part_order, num, let_val, item_id)

def _is_plausible_successor(
    current: "_ItemBoundary",
    candidate: "_ItemBoundary",
    *,
    filing_date: date | None,
    period_end: date | None,
    is_10k: bool,
) -> bool:
    cur_key = _item_order_key(
        current.item_part,
        current.item_id,
        filing_date=filing_date,
        period_end=period_end,
        is_10k=is_10k,
    )
    cand_key = _item_order_key(
        candidate.item_part,
        candidate.item_id,
        filing_date=filing_date,
        period_end=period_end,
        is_10k=is_10k,
    )
    if cur_key is None or cand_key is None:
        return True
    return cand_key > cur_key

def _annotation_from_match(
    entry: dict,
    match: dict,
    *,
    fallback: str | None,
) -> dict[str, str | bool | None]:
    canonical = match.get("canonical") or fallback
    return {
        "canonical_item": canonical,
        "exists_by_regime": True,
        "item_status": _status_from_entry(entry, canonical),
    }

def _regime_annotation_for_item(
    item_part: str | None,
    item_id: str | None,
    *,
    filing_date: date | None,
    period_end: date | None,
) -> dict[str, str | bool | None]:
    item_key = _resolve_item_key(item_part, item_id)
    fallback = item_key or item_id
    if not _ITEM_REGIME_SPEC or not item_key:
        return {
            "canonical_item": fallback,
            "exists_by_regime": None,
            "item_status": "unknown",
        }

    entry = _ITEM_REGIME_ITEMS.get(item_key) or _ITEM_REGIME_LEGACY.get(item_key)
    if not entry:
        matches, missing_date = _find_regime_matches_by_item_id(
            item_id,
            filing_date=filing_date,
            period_end=period_end,
        )
        if matches:
            if len(matches) == 1:
                alt_key, alt_entry, alt_match = matches[0]
                return _annotation_from_match(
                    alt_entry,
                    alt_match,
                    fallback=alt_key,
                )
            return {
                "canonical_item": fallback,
                "exists_by_regime": None,
                "item_status": "unknown",
            }
        if missing_date:
            return {
                "canonical_item": fallback,
                "exists_by_regime": None,
                "item_status": "unknown",
            }
        return {
            "canonical_item": fallback,
            "exists_by_regime": None,
            "item_status": "unknown",
        }

    match, decidable = _evaluate_regime_validity(
        entry.get("validity", []),
        filing_date=filing_date,
        period_end=period_end,
    )
    if not decidable:
        return {
            "canonical_item": _canonical_for_entry(entry, fallback),
            "exists_by_regime": None,
            "item_status": "unknown",
        }
    if match is None:
        matches, missing_date = _find_regime_matches_by_item_id(
            item_id,
            filing_date=filing_date,
            period_end=period_end,
            exclude_key=item_key,
        )
        if matches:
            if len(matches) == 1:
                alt_key, alt_entry, alt_match = matches[0]
                return _annotation_from_match(
                    alt_entry,
                    alt_match,
                    fallback=alt_key,
                )
            return {
                "canonical_item": _canonical_for_entry(entry, fallback),
                "exists_by_regime": None,
                "item_status": "unknown",
            }
        if missing_date:
            return {
                "canonical_item": _canonical_for_entry(entry, fallback),
                "exists_by_regime": None,
                "item_status": "unknown",
            }
        return {
            "canonical_item": _canonical_for_entry(entry, fallback),
            "exists_by_regime": False,
            "item_status": "unknown",
        }

    return _annotation_from_match(entry, match, fallback=fallback)

def _build_regime_item_titles_10k(
    *,
    filing_date: date | None,
    period_end: date | None,
    enable_regime: bool,
) -> dict[str, list[str]]:
    if not enable_regime or not _ITEM_REGIME_SPEC:
        return ITEM_TITLES_10K

    mapping: dict[str, list[str]] = {}
    for item_id, titles in ITEM_TITLES_10K.items():
        item_key = _resolve_item_key(None, item_id)
        entry = _ITEM_REGIME_ITEMS.get(item_key) if item_key else None
        if not entry:
            mapping[item_id] = titles
            continue

        match, decidable = _evaluate_regime_validity(
            entry.get("validity", []),
            filing_date=filing_date,
            period_end=period_end,
        )
        if not decidable:
            mapping[item_id] = titles
            continue
        if match is None:
            matches, missing_date = _find_regime_matches_by_item_id(
                item_id,
                filing_date=filing_date,
                period_end=period_end,
                exclude_key=item_key,
            )
            if matches:
                if len(matches) == 1:
                    _, alt_entry, alt_match = matches[0]
                    canonical = alt_match.get("canonical")
                    override = ITEM_TITLES_10K_BY_CANONICAL.get(canonical or "")
                    if override:
                        mapping[item_id] = override
                        continue
                mapping[item_id] = titles
                continue
            if missing_date:
                mapping[item_id] = titles
                continue
            continue

        canonical = match.get("canonical")
        override = ITEM_TITLES_10K_BY_CANONICAL.get(canonical or "")
        mapping[item_id] = override or titles

    return mapping

def _build_title_lookup_10k(
    *,
    filing_date: date | None,
    period_end: date | None,
    enable_regime: bool,
) -> dict[str, str]:
    mapping = _build_regime_item_titles_10k(
        filing_date=filing_date,
        period_end=period_end,
        enable_regime=enable_regime,
    )
    return _build_title_lookup(mapping)

def _match_title_only_heading(
    line: str,
    lookup: dict[str, str],
    *,
    allow_reserved: bool = False,
) -> str | None:
    if not line:
        return None
    if ITEM_WORD_PATTERN.search(line):
        return None
    norm = _normalize_heading_text(line)
    if not allow_reserved and norm == "RESERVED":
        return None
    return lookup.get(norm)

def _match_numeric_dot_heading(
    line: str,
    lookup: dict[str, str],
    *,
    max_item: int,
) -> str | None:
    if not line:
        return None
    if ITEM_WORD_PATTERN.search(line):
        return None
    m = NUMERIC_DOT_HEADING_PATTERN.match(line)
    if not m:
        return None
    item_id = _normalize_item_id(m.group("num"), m.group("let"), max_item=max_item)
    if item_id is None:
        return None
    title_norm = _normalize_heading_text(m.group("title"))
    if not title_norm:
        return None
    mapped = lookup.get(title_norm)
    if mapped != item_id:
        return None
    return item_id

def _is_toc_entry_line(line: str, lookup: dict[str, str], *, max_item: int) -> bool:
    if not line:
        return False
    if not (DOT_LEADER_PATTERN.search(line) or re.search(r"\s+\d{1,4}\s*$", line)):
        return False
    m = ITEM_LINESTART_PATTERN.match(line)
    if m and _normalize_item_id(m.group("num"), m.group("let"), max_item=max_item) is not None:
        return True
    if _match_numeric_dot_heading(line, lookup, max_item=max_item) is not None:
        return True
    if _match_title_only_heading(line, lookup) is not None:
        return True
    return False

def _infer_toc_end_line_by_titles(
    lines: list[str],
    lookup: dict[str, str],
    *,
    max_item: int,
    max_lines: int = 400,
    max_scan: int = 300,
) -> int | None:
    n = min(len(lines), max_lines)
    if n == 0:
        return None
    toc_start: int | None = None
    for i in range(n):
        if TOC_MARKER_PATTERN.search(lines[i]) or TOC_HEADER_LINE_PATTERN.match(lines[i]):
            toc_start = i
            break
    if toc_start is None:
        return None
    last_entry: int | None = None
    end = min(len(lines), toc_start + max_scan)
    for i in range(toc_start, end):
        if _is_toc_entry_line(lines[i], lookup, max_item=max_item):
            last_entry = i
    return last_entry

def _infer_front_matter_end_pos(
    body: str,
    lines: list[str],
    line_starts: list[int],
    lookup: dict[str, str],
    *,
    max_item: int,
) -> int | None:
    toc_end_pos = _infer_toc_end_pos(body)
    cover_end_pos = _infer_cover_page_end_pos(body)
    end_pos = None
    if toc_end_pos is not None:
        end_pos = toc_end_pos
    if cover_end_pos is not None:
        end_pos = max(end_pos or 0, cover_end_pos)

    if end_pos is not None:
        return end_pos

    toc_end_line = _infer_toc_end_line_by_titles(lines, lookup, max_item=max_item)
    if toc_end_line is None:
        return None
    idx = toc_end_line + 1
    if idx < len(line_starts):
        return line_starts[idx]
    return None

def _infer_cover_page_end_pos(body: str, *, max_chars: int = 20_000) -> int | None:
    window = body[:max_chars]
    if not COVER_PAGE_MARKER_PATTERN.search(window):
        return None
    if not FORM_10K_PATTERN.search(window):
        return None

    cover_matches = [
        m for m in (COVER_PAGE_MARKER_PATTERN.search(window), FORM_10K_PATTERN.search(window)) if m
    ]
    if not cover_matches:
        return None
    cover_end = max(m.end() for m in cover_matches)

    m = COVER_PAGE_ANCHOR_PATTERN.search(body, pos=cover_end)
    if m is None:
        return None
    return m.start()

def _repair_wrapped_headings(body: str) -> str:
    """
    Many filings come from PDF/HTML conversions where headings are wrapped like:
      - 'TABLE\\nOF CONTENTS'
      - 'PART\\nII'
      - 'Item\\n 1.'

    This helper repairs these cases so line-based heuristics can work reliably.
    """
    if not body:
        return body

    body = _WRAPPED_TABLE_OF_CONTENTS_PATTERN.sub("TABLE OF CONTENTS", body)

    def _fix_part(m: re.Match[str]) -> str:
        lead = m.group("lead")
        part = m.group("part").upper()
        return f"{lead} {part}"

    body = _WRAPPED_PART_PATTERN.sub(_fix_part, body)

    def _fix_item(m: re.Match[str]) -> str:
        lead = m.group("lead")
        num = m.group("num").upper()
        let = (m.group("let") or "").upper()
        punc = m.group("punc") or ""
        return f"{lead} {num}{let}{punc}"

    body = _WRAPPED_ITEM_PATTERN.sub(_fix_item, body)

    def _fix_split_letter(m: re.Match[str]) -> str:
        lead = m.group("lead")
        let = m.group("let").upper()
        punc = m.group("punc") or m.group("num_punc") or ""
        return f"{lead}{let}{punc}"

    body = _WRAPPED_ITEM_LETTER_PATTERN.sub(_fix_split_letter, body)

    def _fix_suffix_paren(m: re.Match[str]) -> str:
        lead = m.group("lead")
        punc = m.group("punc") or ""
        return f"{lead}{punc}"

    body = _ITEM_SUFFIX_PAREN_PATTERN.sub(_fix_suffix_paren, body)
    return body

def _detect_toc_line_ranges(
    lines: list[str], *, max_lines: int | None = None
) -> list[tuple[int, int]]:
    """
    Best-effort detection of Table of Contents / Summary blocks.
    Returns inclusive (start_line, end_line) ranges.
    """
    n = len(lines) if max_lines is None else min(len(lines), max_lines)
    if n == 0:
        return []

    ranges: list[tuple[int, int]] = []

    def _pageish(line: str) -> bool:
        return bool(
            PAGE_NUMBER_PATTERN.match(line)
            or PAGE_HYPHEN_PATTERN.match(line)
            or PAGE_ROMAN_PATTERN.match(line)
            or PAGE_OF_PATTERN.match(line)
        )

    def _prose_like(line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        if re.fullmatch(r"[\s\-=*]{3,}", stripped):
            return False
        letters = re.findall(r"[A-Za-z]", stripped)
        if not letters:
            return False
        has_lower = any(ch.islower() for ch in letters)
        if not has_lower and len(letters) < 30:
            return False
        if len(letters) >= 20 and has_lower:
            return True
        return bool(re.search(r"[.!?]\s*$", stripped))

    def _gap_text_line(line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        if _pageish(stripped):
            return False
        if TOC_INDEX_MARKER_PATTERN.search(stripped):
            return False
        letters = [ch for ch in stripped if ch.isalpha()]
        if not letters:
            return False
        return any(ch.islower() for ch in letters)

    # 1) Inline TOC lines (many ITEM tokens on the same line) and explicit marker lines.
    inline_lines: list[int] = []
    toc_markers: set[int] = set()
    heading_lines: list[int] = []
    for i in range(n):
        line = lines[i]
        line_len = len(line)
        stripped = line.strip()
        if not stripped:
            continue
        item_words = len(ITEM_WORD_PATTERN.findall(line))
        has_marker = TOC_INDEX_MARKER_PATTERN.search(line) is not None
        has_summary_marker = SUMMARY_MARKER_PATTERN.search(line) is not None
        if has_marker:
            toc_markers.add(i)
        if item_words >= 3 and line_len <= 5_000:
            inline_lines.append(i)
            continue
        if has_marker and item_words >= 1 and line_len <= 8_000:
            if not has_summary_marker or DOT_LEADER_PATTERN.search(line) or re.search(
                r"\s+\d{1,4}\s*$", line
            ):
                inline_lines.append(i)

        m_item = ITEM_LINESTART_PATTERN.match(line)
        if m_item and _normalize_item_id(m_item.group("num"), m_item.group("let")) is not None:
            heading_lines.append(i)
            continue
        m_part = PART_LINESTART_PATTERN.match(line)
        if m_part is not None and _part_marker_is_heading(line, m_part):
            heading_lines.append(i)
            continue
        if DOT_LEADER_PATTERN.search(line) and NUMERIC_DOT_HEADING_PATTERN.match(line):
            heading_lines.append(i)

    for i in inline_lines:
        ranges.append((i, i))

    # Handle split markers like:
    #   TABLE
    #   OF CONTENTS
    for i in range(n - 1):
        a = lines[i].strip().lower()
        b = lines[i + 1].strip().lower()
        if a == "table" and b.startswith("of contents"):
            toc_markers.add(i)
            toc_markers.add(i + 1)
        if a == "table of" and b.startswith("contents"):
            toc_markers.add(i)
            toc_markers.add(i + 1)
        if a == "form 10-k" and b.startswith("summary"):
            toc_markers.add(i)
            toc_markers.add(i + 1)

    # 2) Multi-line TOC clusters: dense heading blocks with sparse prose.
    clusters: list[list[int]] = []
    if heading_lines:
        cur = [heading_lines[0]]
        for idx in heading_lines[1:]:
            if idx - cur[-1] <= 6:
                has_prose_gap = False
                for j in range(cur[-1] + 1, idx):
                    if _prose_like(lines[j]) or _gap_text_line(lines[j]):
                        has_prose_gap = True
                        break
                if has_prose_gap:
                    clusters.append(cur)
                    cur = [idx]
                else:
                    cur.append(idx)
            else:
                clusters.append(cur)
                cur = [idx]
        clusters.append(cur)

    chosen: list[tuple[int, int]] = []
    for cl in clusters:
        has_marker_near = any((cl[0] - 10) <= mi <= (cl[-1] + 10) for mi in toc_markers)
        min_headings = 3 if has_marker_near else 4
        if len(cl) < min_headings:
            continue

        nonempty_block = [lines[i].strip() for i in range(cl[0], cl[-1] + 1) if lines[i].strip()]
        if not nonempty_block:
            continue

        headingish = 0
        prose_like = 0
        dot_hits = 0
        page_hits = 0
        last_toc_like_idx: int | None = None
        for i in range(cl[0], cl[-1] + 1):
            line = lines[i].strip()
            if not line:
                continue
            if DOT_LEADER_PATTERN.search(line) or re.search(r"\s+\d{1,4}\s*$", line) or _pageish(line):
                last_toc_like_idx = i
        for s in nonempty_block:
            if DOT_LEADER_PATTERN.search(s):
                dot_hits += 1
            if re.search(r"\s+\d{1,4}\s*$", s) or _pageish(s):
                page_hits += 1
            m = ITEM_LINESTART_PATTERN.match(s)
            if m and _normalize_item_id(m.group("num"), m.group("let")) is not None:
                headingish += 1
                continue
            m_part = PART_LINESTART_PATTERN.match(s)
            if m_part is not None and _part_marker_is_heading(s, m_part):
                headingish += 1
                continue
            if TOC_HEADER_LINE_PATTERN.match(s) or SUMMARY_HEADER_LINE_PATTERN.match(s):
                headingish += 1
                continue
            if INDEX_HEADER_LINE_PATTERN.match(s):
                headingish += 1
                continue
            if _pageish(s):
                headingish += 1
                continue
            if _prose_like(s):
                prose_like += 1

        headingish_ratio = headingish / len(nonempty_block)
        prose_ratio = prose_like / len(nonempty_block)
        if headingish_ratio < 0.75 or prose_ratio > 0.2:
            continue

        late_cluster = cl[0] > 300
        strong_page = dot_hits >= 2 or page_hits >= 2
        if late_cluster:
            if not (has_marker_near or dot_hits >= 3 or page_hits >= 3):
                continue
        elif not (has_marker_near or strong_page):
            continue
        end_idx = cl[-1]
        if (dot_hits >= 2 or page_hits >= 2) and last_toc_like_idx is not None:
            end_idx = min(end_idx, last_toc_like_idx)
        chosen.append((cl[0], end_idx))

    ranges.extend(chosen)

    # Merge overlapping/adjacent ranges.
    if not ranges:
        return []
    ranges.sort()
    merged: list[tuple[int, int]] = [ranges[0]]
    for s, e in ranges[1:]:
        ps, pe = merged[-1]
        if s <= pe + 1:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged

def _infer_toc_end_pos(body: str, *, max_chars: int = 20_000) -> int | None:
    """
    Character-based TOC cutoff for filings with very few line breaks.
    Returns the end position (in `body`) of a likely TOC region, or None.
    """
    m = re.search(r"table\s+of\s+contents?", body, flags=re.IGNORECASE)
    if not m:
        return None

    start = m.start()
    end = min(len(body), start + max_chars)
    window = body[start:end]

    last_end: int | None = None
    count = 0
    for mm in TOC_ENTRY_PATTERN.finditer(window):
        item_id = _normalize_item_id(mm.group("num"), mm.group("let"))
        if item_id is None:
            continue
        try:
            page = int(mm.group("page"))
        except Exception:
            continue
        if page <= 0 or page > 500:
            continue
        count += 1
        last_end = start + mm.end()

    if count >= 4 and last_end is not None:
        return last_end
    return None

def _line_ranges_to_mask(ranges: list[tuple[int, int]]) -> set[int]:
    mask: set[int] = set()
    for s, e in ranges:
        for i in range(s, e + 1):
            mask.add(i)
    return mask

def _remove_pagination(text: str) -> str:
    """
    Remove common page-number and page-header artifacts while preserving table rows.
    """
    text = _normalize_newlines(text)
    lines = text.split("\n")
    if not lines:
        return text

    out: list[str] = []
    for i, line in enumerate(lines):
        s = line.strip()
        prev_blank = i > 0 and lines[i - 1].strip() == ""
        next_blank = i + 1 < len(lines) and lines[i + 1].strip() == ""

        if not s:
            out.append("")
            continue

        if TOC_HEADER_LINE_PATTERN.match(s):
            continue
        if PAGE_OF_PATTERN.match(s):
            continue
        if PAGE_HYPHEN_PATTERN.match(s):
            continue
        if PAGE_ROMAN_PATTERN.match(s) and (prev_blank or next_blank):
            continue
        if PAGE_NUMBER_PATTERN.match(s):
            try:
                v = int(s)
            except Exception:
                v = None
            if v is not None and v <= 500 and (prev_blank or next_blank):
                continue

        out.append(line)

    cleaned = "\n".join(out)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned

def _trim_trailing_part_marker(text: str) -> str:
    if not text:
        return text
    lines = text.split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return text
    last = lines[-1].strip()
    m = PART_LINESTART_PATTERN.match(last)
    if m:
        remainder = re.sub(
            r"(?i)^\s*PART\s+(?:IV|III|II|I)\b", "", last
        ).strip(" \t:-")
        if not remainder:
            lines.pop()
    return "\n".join(lines).rstrip()

def _reserved_stub_end(text: str) -> int | None:
    if not text:
        return None
    offset = 0
    for line in text.splitlines(keepends=True):
        offset += len(line)
        stripped = line.strip()
        if not stripped:
            continue
        probe = stripped.lstrip(" \t:-.")
        if not probe:
            continue
        if EMPTY_ITEM_PATTERN.match(probe) and "reserved" in probe.lower():
            return offset
        return None
    return None

def _item_id_to_int_simple(item_id: str | None) -> int | None:
    if not item_id:
        return None
    m = re.match(r"^(\d+)", item_id)
    if not m:
        return None
    return int(m.group(1))

def _leading_ws_len(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))

def _line_start_item_match(
    line: str,
    *,
    max_item: int,
) -> tuple[str | None, int | None]:
    if not line:
        return None, None
    m = ITEM_LINESTART_PATTERN.match(line)
    if m:
        item_id = _normalize_item_id(m.group("num"), m.group("let"), max_item=max_item)
        if item_id:
            return item_id, m.end()

    part_match = PART_LINESTART_PATTERN.match(line)
    if part_match:
        rest = line[part_match.end() :]
        rest_stripped = rest.lstrip(" \t:/-")
        offset = len(rest) - len(rest_stripped)
        m_item = ITEM_CANDIDATE_PATTERN.match(rest_stripped)
        if m_item:
            item_id = _normalize_item_id(m_item.group("num"), m_item.group("let"), max_item=max_item)
            if item_id:
                return item_id, part_match.end() + offset + m_item.end()

    return None, None

def _is_late_item_start(item_id: str | None, item_part: str | None) -> bool:
    if item_part and item_part.upper() in {"III", "IV"}:
        return True
    num = _item_id_to_int_simple(item_id)
    return num is not None and num >= 10

def _late_item_restart_after(
    lines: list[str],
    start_idx: int,
    *,
    max_non_empty: int = 20,
    max_item: int,
) -> bool:
    non_empty = 0
    for j in range(start_idx + 1, len(lines)):
        line = lines[j]
        if not line.strip():
            continue
        non_empty += 1
        if non_empty > max_non_empty:
            break
        part_match = embedded_headings.EMBEDDED_PART_PATTERN.match(line)
        if part_match and part_match.group("roman").upper() == "I":
            return True
        part_item_match = embedded_headings.EMBEDDED_TOC_PART_ITEM_PATTERN.match(line)
        if part_item_match:
            item_id = embedded_headings._item_id_from_match(part_item_match)
            if item_id == "1":
                return True
        item_match = embedded_headings.EMBEDDED_ITEM_PATTERN.match(line) or (
            embedded_headings.EMBEDDED_ITEM_ROMAN_PATTERN.match(line)
        )
        if item_match:
            item_id = embedded_headings._item_id_from_match(item_match)
            if item_id == "1":
                return True
    return False

def _title_like_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if re.search(r"[.!?]\s*$", stripped):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'&-]*", stripped)
    if len(words) < 2 or len(words) > 14:
        return False
    if stripped.isupper():
        return True
    titlecase = sum(1 for word in words if word[0].isupper())
    return titlecase / len(words) >= 0.7

def _part_boundary_header_like(
    lines: list[str],
    idx: int,
    *,
    max_item: int,
    lookahead: int = 6,
) -> bool:
    if not PART_ONLY_PREFIX_PATTERN.match(lines[idx]):
        return False
    if idx > 0:
        prev = lines[idx - 1].strip()
        if prev and not re.fullmatch(r"[\s\-=*]{3,}", prev):
            return False
    non_empty = 0
    for j in range(idx + 1, len(lines)):
        if not lines[j].strip():
            continue
        non_empty += 1
        if non_empty > lookahead:
            break
        item_id, _ = _line_start_item_match(lines[j], max_item=max_item)
        if item_id:
            return True
        if _title_like_line(lines[j]):
            return True
    return False

def _apply_high_confidence_truncation(
    text: str,
    *,
    next_item_id: str | None,
    current_part: str | None,
    max_item: int,
) -> tuple[str, bool, bool]:
    if not text:
        return text, False, False

    lines = text.splitlines(keepends=True)
    lines_noeol = [line.rstrip("\r\n") for line in lines]
    non_empty = 0
    offset = 0
    for idx, line_noeol in enumerate(lines_noeol):
        line = lines[idx]
        if not line_noeol.strip():
            offset += len(line)
            continue
        non_empty += 1
        if embedded_headings.is_empty_section_line(line_noeol):
            cut = offset + len(line)
            return text[:cut].rstrip(), False, False
        offset += len(line)
        if non_empty >= 3:
            break

    if next_item_id:
        offset = 0
        for idx, line_noeol in enumerate(lines_noeol):
            if not line_noeol.strip():
                offset += len(lines[idx])
                continue
            item_id, match_end = _line_start_item_match(line_noeol, max_item=max_item)
            if item_id and item_id == next_item_id:
                suffix = ""
                if match_end is not None:
                    suffix = line_noeol[match_end:]
                suffix = suffix.lstrip(" \t:-.")
                if not CROSS_REF_SUFFIX_START_PATTERN.match(suffix):
                    cut = offset + _leading_ws_len(line_noeol)
                    return text[:cut].rstrip(), True, False
            offset += len(lines[idx])

    part_order = {"I": 1, "II": 2, "III": 3, "IV": 4}
    current_part_norm = (current_part or "").upper()
    current_part_order = part_order.get(current_part_norm, 0)
    if current_part_order:
        toc_window_flags = embedded_headings._toc_window_flags(lines_noeol)
        offset = 0
        for idx, line_noeol in enumerate(lines_noeol):
            if not line_noeol.strip():
                offset += len(lines[idx])
                continue
            part_match = embedded_headings.STRICT_PART_PATTERN.match(line_noeol)
            if not part_match:
                offset += len(lines[idx])
                continue
            if toc_window_flags[idx]:
                offset += len(lines[idx])
                continue
            next_part = part_match.group("part").upper()
            next_part_order = part_order.get(next_part, 0)
            if next_part_order <= current_part_order:
                offset += len(lines[idx])
                continue
            if not embedded_headings._confirm_strict_part_heading(lines_noeol, idx):
                offset += len(lines[idx])
                continue
            cut = offset + _leading_ws_len(line_noeol)
            return text[:cut].rstrip(), False, True
        # end for

    return text, False, False

def _select_best_boundaries(
    candidates: list["_ItemBoundary"],
    *,
    lines: list[str],
    body: str,
    skip_until_pos: int | None,
    filing_date: date | None,
    period_end: date | None,
    is_10k: bool,
    max_item: int,
    gij_context: dict[str, bool | list[tuple[int, int]] | set[str] | str] | None = None,
) -> tuple[list["_ItemBoundary"], dict[tuple[str | None, str], dict[str, int | bool]]]:
    if not candidates:
        return [], {}

    candidates = sorted(candidates, key=lambda b: b.start)
    line_starts: list[int] = []
    pos = 0
    for line in lines:
        line_starts.append(pos)
        pos += len(line) + 1  # '\n'

    toc_window_flags = embedded_headings._toc_window_flags(lines)
    toc_cache: dict[tuple[int, bool], bool] = {}
    gij_omit_ranges = gij_context.get("gij_omit_ranges", []) if gij_context else []
    gij_substitute_ranges = gij_context.get("gij_substitute_ranges", []) if gij_context else []

    def _line_in_ranges(idx: int, ranges: list[tuple[int, int]]) -> bool:
        return any(start <= idx < end for start, end in ranges)

    by_key: dict[tuple[str | None, str], list[tuple[int, "_ItemBoundary"]]] = {}
    for idx, b in enumerate(candidates):
        part = b.item_part
        if is_10k:
            inferred = _infer_part_for_item_id(
                b.item_id,
                filing_date=filing_date,
                period_end=period_end,
                is_10k=is_10k,
            )
            if inferred:
                part = inferred
        key = (part, b.item_id)
        by_key.setdefault(key, []).append((idx, b))

    orderable: list[tuple[tuple[int, int, int, str], tuple[str | None, str]]] = []
    unordered: list[tuple[str | None, str]] = []
    for key in by_key:
        part, item_id = key
        if part is None and not is_10k:
            order_key = None
        else:
            order_key = _item_order_key(
                part,
                item_id,
                filing_date=filing_date,
                period_end=period_end,
                is_10k=is_10k,
            )
        if order_key is None:
            unordered.append(key)
        else:
            orderable.append((order_key, key))

    orderable.sort(key=lambda t: t[0])
    unordered.sort(key=lambda k: str(k[1]))
    ordered_keys = [k for _, k in orderable] + unordered

    next_map: dict[tuple[str | None, str], tuple[str | None, str]] = {}
    for idx, key in enumerate(ordered_keys):
        if idx + 1 < len(ordered_keys):
            next_map[key] = ordered_keys[idx + 1]

    def _guard_window_text(boundary: _ItemBoundary) -> str:
        if boundary.line_index < 0 or boundary.line_index >= len(lines):
            return ""
        start_offset = boundary.content_start - line_starts[boundary.line_index]
        if start_offset < 0:
            start_offset = 0
        line = lines[boundary.line_index]
        if start_offset > len(line):
            start_offset = len(line)
        window_lines = [line[start_offset:]]
        non_empty = 1 if window_lines[0].strip() else 0
        char_count = len(window_lines[0])
        if (
            char_count >= START_LOOKAHEAD_GUARD_MAX_CHARS
            or non_empty >= START_LOOKAHEAD_GUARD_NONEMPTY
        ):
            return "\n".join(window_lines)
        for j in range(boundary.line_index + 1, len(lines)):
            next_line = lines[j]
            window_lines.append(next_line)
            char_count += len(next_line)
            if next_line.strip():
                non_empty += 1
            if (
                char_count >= START_LOOKAHEAD_GUARD_MAX_CHARS
                or non_empty >= START_LOOKAHEAD_GUARD_NONEMPTY
            ):
                break
        return "\n".join(window_lines)

    def _guard_has_empty_stub(guard_lines: list[str]) -> bool:
        non_empty = 0
        for line in guard_lines:
            if not line.strip():
                continue
            non_empty += 1
            if embedded_headings.is_empty_section_line(line):
                return True
            if non_empty >= 3:
                break
        return False

    def _toc_marker_near(idx: int) -> bool:
        start = max(0, idx - START_SCORE_TOC_MARKER_WINDOW)
        end = min(len(lines), idx + START_SCORE_TOC_MARKER_WINDOW + 1)
        for j in range(start, end):
            if START_TOC_SUMMARY_MARKER_PATTERN.search(lines[j]):
                return True
        return False

    def _score_candidate(boundary: _ItemBoundary) -> int:
        score = 0
        line = lines[boundary.line_index] if 0 <= boundary.line_index < len(lines) else ""
        line_idx = boundary.line_index
        if embedded_headings.is_item_run_line(line):
            score -= 30
        if _line_in_ranges(line_idx, gij_omit_ranges):
            score -= 30
        if _line_in_ranges(line_idx, gij_substitute_ranges):
            score -= 20
        if TOC_DOT_LEADER_PATTERN.search(line):
            score -= 10
        if START_TRAILING_PAGE_NUMBER_PATTERN.search(line):
            score -= 8
        if _toc_marker_near(boundary.line_index):
            score -= 8
        if embedded_headings._toc_cluster_after(lines, boundary.line_index):
            score -= 6

        window_end = min(len(lines), boundary.line_index + 1 + START_SCORE_LOOKAHEAD_LINES)
        non_empty = 0
        prose_hits = 0
        structural_hits = 0
        trailing_page_hits = 0
        for j in range(boundary.line_index + 1, window_end):
            line_j = lines[j]
            if not line_j.strip():
                continue
            non_empty += 1
            if START_TRAILING_PAGE_NUMBER_PATTERN.search(line_j):
                trailing_page_hits += 1
            toc_like = embedded_headings._toc_like_line(lines, j, toc_cache, toc_window_flags)
            if embedded_headings._prose_like_line(line_j, toc_like=toc_like):
                prose_hits += 1
            if embedded_headings._line_starts_item(line_j) or embedded_headings.EMBEDDED_PART_PATTERN.match(
                line_j
            ):
                structural_hits += 1

        if trailing_page_hits >= START_SCORE_TRAILING_PAGE_MIN:
            score -= 6
        if non_empty:
            prose_ratio = prose_hits / non_empty
            structural_ratio = structural_hits / non_empty
            if prose_ratio > 0.30:
                score += 6
            if structural_ratio <= START_SCORE_STRUCTURAL_LOW_RATIO:
                score += 4
            elif structural_ratio >= START_SCORE_STRUCTURAL_HIGH_RATIO:
                score -= 4
        return score

    def _guard_has_prose_before_hit(guard_lines: list[str], hit_line_idx: int) -> bool:
        if hit_line_idx <= 0:
            return False
        guard_toc_flags = embedded_headings._toc_window_flags(guard_lines)
        guard_cache: dict[tuple[int, bool], bool] = {}
        for j in range(min(hit_line_idx, len(guard_lines))):
            line = guard_lines[j]
            if not line.strip():
                continue
            toc_like = embedded_headings._toc_like_line(guard_lines, j, guard_cache, guard_toc_flags)
            if embedded_headings._prose_like_line(line, toc_like=toc_like):
                return True
        return False

    def _restart_marker_index(guard_lines: list[str]) -> int | None:
        for idx, line in enumerate(guard_lines):
            if not line.strip():
                continue
            part_match = embedded_headings.STRICT_PART_PATTERN.match(line)
            if part_match and part_match.group("part").upper() == "I":
                return idx
            item_match = (
                embedded_headings.EMBEDDED_ITEM_PATTERN.match(line)
                or embedded_headings.EMBEDDED_ITEM_ROMAN_PATTERN.match(line)
                or embedded_headings.EMBEDDED_TOC_PART_ITEM_PATTERN.match(line)
            )
            if item_match:
                item_id = embedded_headings._item_id_from_match(item_match)
                if item_id == "1":
                    return idx
        return None

    selected: list["_ItemBoundary"] = []
    meta: dict[tuple[str | None, str], dict[str, int | bool]] = {}
    for key, entries in by_key.items():
        entries = sorted(entries, key=lambda t: t[1].start)
        if len(entries) > START_CANDIDATE_MAX:
            entries = entries[:START_CANDIDATE_MAX]
        total = len(entries)
        toc_rejected = 0
        scored_entries: list[tuple[_ItemBoundary, int, bool]] = []
        for _, b in entries:
            part_for_checks = b.item_part
            if is_10k:
                inferred_part = _infer_part_for_item_id(
                    b.item_id,
                    filing_date=filing_date,
                    period_end=period_end,
                    is_10k=is_10k,
                )
                if inferred_part:
                    part_for_checks = inferred_part

            guard_text = _guard_window_text(b)
            guard_lines = guard_text.splitlines() if guard_text else []
            reserved_stub = _reserved_stub_end(guard_text) is not None
            empty_stub = _guard_has_empty_stub(guard_lines)
            guard_reject = False
            if guard_text and not reserved_stub and not empty_stub:
                restart_hit = embedded_headings.find_strong_leak(
                    guard_text,
                    current_item_id=b.item_id,
                    current_part=part_for_checks,
                    next_item_id="1",
                    next_part="I",
                )
                if restart_hit is not None:
                    guard_reject = True
                else:
                    restart_idx = _restart_marker_index(guard_lines)
                    if restart_idx is not None and not _guard_has_prose_before_hit(
                        guard_lines, restart_idx
                    ):
                        guard_reject = True
                if not guard_reject:
                    next_key = next_map.get(key)
                    if next_key:
                        next_part, next_item_id = next_key
                        successor_hit = embedded_headings.find_strong_leak(
                            guard_text,
                            current_item_id=b.item_id,
                            current_part=part_for_checks,
                            next_item_id=next_item_id,
                            next_part=next_part,
                        )
                        if successor_hit is not None and not _guard_has_prose_before_hit(
                            guard_lines, successor_hit.line_idx
                        ):
                            guard_reject = True
            if guard_reject:
                toc_rejected += 1

            score = _score_candidate(b)
            scored_entries.append((b, score, guard_reject))

        if not scored_entries:
            continue
        ok_entries = [entry for entry in scored_entries if not entry[2]]
        if ok_entries:
            chosen = max(ok_entries, key=lambda entry: (entry[1], entry[0].start))[0]
            verified = True
        else:
            continue

        selected.append(chosen)
        meta[key] = {
            "start_candidates_total": total,
            "start_candidates_toc_rejected": toc_rejected,
            "start_selection_verified": verified,
        }

    selected.sort(key=lambda b: b.start)
    return selected, meta

def _detect_heading_style_10k(
    lines: list[str],
    line_starts: list[int],
    front_matter_end_pos: int | None,
    lookup: dict[str, str],
    *,
    max_item: int,
) -> str:
    """
    Classify 10-K heading style for fallback routing.
    A: ITEM 1/1A headings present
    B: numeric-dot headings with mapped titles (e.g., 1A. RISK FACTORS)
    C: title-only headings with mapped titles (e.g., RISK FACTORS)
    """
    seen_numeric = False
    seen_title = False
    for i, line in enumerate(lines):
        if front_matter_end_pos is not None and line_starts[i] < front_matter_end_pos:
            continue
        if TOC_HEADER_LINE_PATTERN.match(line):
            continue
        m = ITEM_LINESTART_PATTERN.match(line)
        if m and _normalize_item_id(m.group("num"), m.group("let"), max_item=max_item) is not None:
            return "A"
        if _match_numeric_dot_heading(line, lookup, max_item=max_item):
            seen_numeric = True
            continue
        if _match_title_only_heading(line, lookup):
            seen_title = True
    if seen_numeric:
        return "B"
    if seen_title:
        return "C"
    return "UNKNOWN"

def _is_empty_item_text(text: str | None) -> bool:
    if not text:
        return True
    return bool(EMPTY_ITEM_PATTERN.match(text.strip()))

def _annotate_items_with_regime(
    items: list[dict[str, str | bool | None]],
    *,
    form_type: str | None,
    filing_date: date | None,
    period_end: date | None,
    enable_regime: bool,
) -> None:
    if not items:
        return
    form = (form_type or "").upper().strip()
    apply_regime = enable_regime and (form.startswith("10K") or form.startswith("10-K"))
    for item in items:
        if apply_regime:
            item.update(
                _regime_annotation_for_item(
                    item.get("item_part"),
                    item.get("item_id"),
                    filing_date=filing_date,
                    period_end=period_end,
                )
            )
            continue

        fallback = item.get("item") or item.get("item_id")
        item.setdefault("canonical_item", fallback)
        item.setdefault("exists_by_regime", None)
        item.setdefault("item_status", "unknown")

def _apply_heading_hygiene(
    items: list[dict[str, str | bool | None]],
    *,
    title_map: dict[str, list[str]],
    is_10k: bool,
) -> None:
    """
    Clean heading metadata without altering any boundary indices or content offsets.
    """
    for item in items:
        raw_line = item.get("_heading_line_raw")
        if raw_line is None:
            continue
        heading_offset = item.get("_heading_offset")
        heading_offset_val = heading_offset if isinstance(heading_offset, int) else None
        item["_heading_line"] = _clean_heading_line(
            str(raw_line),
            heading_offset_val,
            item_id=item.get("item_id") if isinstance(item.get("item_id"), str) else None,
            canonical_item=(
                item.get("canonical_item") if isinstance(item.get("canonical_item"), str) else None
            ),
            title_map=title_map,
            is_10k=is_10k,
        )

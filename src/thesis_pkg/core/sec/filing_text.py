from __future__ import annotations

import json
import re
from bisect import bisect_right
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import polars as pl

from thesis_pkg.core.sec import embedded_headings


FILENAME_PATTERN = re.compile(
    r"(\d{8})_"          # 1: Date (YYYYMMDD)
    r"([^_]+)_"          # 2: Type
    r"edgar_data_"
    r"(\d+)_"            # 3: CIK
    r"([\d-]+)"          # 4: Accession Number
    r"\.txt$",
    re.IGNORECASE,
)

HEADER_SEARCH_LIMIT_DEFAULT = 5000

CIK_HEADER_PATTERN = re.compile(r"CENTRAL INDEX KEY:\s*(\d+)")
DATE_HEADER_PATTERN = re.compile(r"FILED AS OF DATE:\s*(\d{8})")
PERIOD_END_HEADER_PATTERN = re.compile(r"CONFORMED PERIOD OF REPORT:\s*(\d{8})")
ACC_HEADER_PATTERN = re.compile(r"ACCESSION NUMBER:\s*([\d-]+)")

EDGAR_BLOCK_TAG_PATTERN = re.compile(
    r"(?is)<\s*(?P<tag>sec-header|header|filestats|file-stats|xml_chars|xml-chars)\b[^>]*>"
    r".*?<\s*/\s*(?P=tag)\s*>"
)
EDGAR_TRAILING_TAG_PATTERN = re.compile(
    r"(?is)^\s*<\s*(sec-header|header)\b[^>]*>.*",
)

TOC_MARKER_PATTERN = re.compile(
    # Be conservative: "INDEX" appears in many non-TOC contexts (e.g., "Exhibit Index"),
    # and false TOC masking can suppress real item headings.
    r"\btable\s+of\s+contents?\b|\btable\s+of\s+content\b",
    re.IGNORECASE,
)
TOC_HEADER_LINE_PATTERN = re.compile(r"^\s*table\s+of\s+contents?\s*$", re.IGNORECASE)
TOC_INDEX_MARKER_PATTERN = re.compile(
    r"\btable\s+of\s+contents?\b|\btable\s+of\s+content\b|\bindex\b|\b(?:form\s+)?10-?k\s+summary\b",
    re.IGNORECASE,
)
SUMMARY_MARKER_PATTERN = re.compile(r"\b(?:form\s+)?10-?k\s+summary\b", re.IGNORECASE)
SUMMARY_HEADER_LINE_PATTERN = re.compile(
    r"^\s*(?:form\s+)?10-?k\s+summary\s*$", re.IGNORECASE
)
INDEX_HEADER_LINE_PATTERN = re.compile(r"^\s*index(?:\b|$)", re.IGNORECASE)
ITEM_WORD_PATTERN = re.compile(r"\bITEM\b", re.IGNORECASE)
PART_MARKER_PATTERN = re.compile(r"\bPART\s+(?P<part>IV|III|II|I)\b(?!\s*,)", re.IGNORECASE)
PART_LINESTART_PATTERN = re.compile(r"^\s*PART\s+(?P<part>IV|III|II|I)\b", re.IGNORECASE)
ITEM_CANDIDATE_PATTERN = re.compile(
    r"\bITEM\s+(?P<num>\d+|[IVXLCDM]+)(?P<let>[A-Z])?\s*[\.:]?",
    re.IGNORECASE,
)
ITEM_LINESTART_PATTERN = re.compile(
    r"^\s*(?:PART\s+[IVXLCDM]+\s*[:\-]?\s*)?ITEM\s+(?P<num>\d+|[IVXLCDM]+)"
    r"(?P<let>[A-Z])?(?=\b|(?-i:[A-Z]))",
    re.IGNORECASE,
)
NUMERIC_DOT_HEADING_PATTERN = re.compile(
    r"^\s*(?P<num>\d{1,2})(?P<let>[A-Z])?\s*[\.\):\-]?\s+(?P<title>.+?)\s*$",
    re.IGNORECASE,
)
DOT_LEADER_PATTERN = re.compile(r"(?:\.{4,}|(?:\.\s*){4,})")
TOC_DOT_LEADER_PATTERN = embedded_headings.TOC_DOT_LEADER_PATTERN
CONTINUED_PATTERN = re.compile(r"\bcontinued\b", re.IGNORECASE)
ITEM_MENTION_PATTERN = re.compile(r"\bITEM\s+(?:\d+|[IVXLCDM]+)[A-Z]?\b", re.IGNORECASE)
PART_ONLY_PREFIX_PATTERN = re.compile(
    r"^\s*PART\s+(?:IV|III|II|I)\s*[:\-]?\s*$",
    re.IGNORECASE,
)
PART_PREFIX_TAIL_PATTERN = re.compile(
    r"(?:^|[\s:;,.])PART\s+(?P<part>IV|III|II|I)\s*$",
    re.IGNORECASE,
)
CROSS_REF_PREFIX_PATTERN = re.compile(
    r"(?i)\bsee\b|\brefer\b|\bas discussed\b|\bas described\b|\bas set forth\b|\bas noted\b"
    r"|\bpursuant to\b|\bunder\b|\bin accordance with\b",
)
CROSS_REF_PART_PATTERN = re.compile(
    r"(?i)\bin\s+part\s+(?:IV|III|II|I)\b",
)
COVER_PAGE_MARKER_PATTERN = re.compile(
    r"UNITED STATES\s+SECURITIES\s+AND\s+EXCHANGE\s+COMMISSION",
    re.IGNORECASE,
)
FORM_10K_PATTERN = re.compile(r"\bFORM\s+10-?K\b", re.IGNORECASE)

PAGE_HYPHEN_PATTERN = re.compile(r"^\s*-\d{1,4}-\s*$")
PAGE_NUMBER_PATTERN = re.compile(r"^\s*\d{1,4}\s*$")
PAGE_ROMAN_PATTERN = re.compile(r"^\s*[ivxlcdm]{1,6}\s*$", re.IGNORECASE)
PAGE_OF_PATTERN = re.compile(r"^\s*page\s+\d+\s*(?:of\s+\d+)?\s*$", re.IGNORECASE)

EMPTY_ITEM_PATTERN = re.compile(
    r"^\s*(?:\(?[a-z]\)?\s*)?(?:none\.?|n/?a\.?|not applicable\.?|not required\.?|\[reserved\]|reserved)\s*$",
    re.IGNORECASE,
)
PROSE_SENTENCE_BREAK_PATTERN = re.compile(r"[.!?]\s+[A-Za-z]")
CROSS_REF_SUFFIX_START_PATTERN = re.compile(
    r"(?i)^(?:see|refer|as discussed|as described|as set forth|as noted|pursuant to|under|"
    r"in accordance with|in part\s+(?:IV|III|II|I))\b",
)
PART_REF_PATTERN = re.compile(r"(?i)\bpart\s+(?:IV|III|II|I)\b")
HEADING_TAIL_STRONG_PATTERN = re.compile(
    r"(?i)\b(?:is incorporated(?: herein)? by reference|herein by reference|set forth)\b"
)
HEADING_TAIL_SOFT_PATTERN = re.compile(
    r"\b(?:THE|SEE|REFER|PAGES|INCLUDED|BEGIN)\b",
    re.IGNORECASE,
)

HEADING_CONF_LOW = 0
HEADING_CONF_MED = 1
HEADING_CONF_HIGH = 2

ITEM_TITLES_10K = {
    "1": ["BUSINESS"],
    "1A": ["RISK FACTORS"],
    "1B": ["UNRESOLVED STAFF COMMENTS"],
    "1C": ["CYBERSECURITY"],
    "2": ["PROPERTIES", "PROPERTY"],
    "3": ["LEGAL PROCEEDINGS"],
    "4": [
        "MINE SAFETY DISCLOSURES",
        "SUBMISSION OF MATTERS TO A VOTE OF SECURITY HOLDERS",
        "SUBMISSION OF MATTERS TO A VOTE OF SHAREHOLDERS",
        "RESERVED",
    ],
    "5": [
        "MARKET FOR REGISTRANT'S COMMON EQUITY",
        "MARKET FOR REGISTRANT S COMMON EQUITY",
        "MARKET FOR REGISTRANTS COMMON EQUITY",
        "MARKET FOR REGISTRANT'S COMMON EQUITY, RELATED STOCKHOLDER MATTERS AND ISSUER PURCHASES OF EQUITY SECURITIES",
        "MARKET FOR REGISTRANT S COMMON EQUITY, RELATED STOCKHOLDER MATTERS AND ISSUER PURCHASES OF EQUITY SECURITIES",
    ],
    "6": ["SELECTED FINANCIAL DATA", "RESERVED"],
    "7": [
        "MANAGEMENT'S DISCUSSION AND ANALYSIS",
        "MANAGEMENT S DISCUSSION AND ANALYSIS",
        "MANAGEMENTS DISCUSSION AND ANALYSIS",
        "MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS",
        "MANAGEMENT S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS",
    ],
    "7A": [
        "QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK",
        "QUANTITATIVE AND QUALITATIVE DISCLOSURES",
    ],
    "8": [
        "FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA",
        "FINANCIAL STATEMENTS",
        "CONSOLIDATED FINANCIAL STATEMENTS",
        "CONSOLIDATED FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA",
        "NOTES TO CONSOLIDATED FINANCIAL STATEMENTS",
        "NOTES TO THE CONSOLIDATED FINANCIAL STATEMENTS",
    ],
    "9": [
        "CHANGES IN AND DISAGREEMENTS WITH ACCOUNTANTS",
        "CHANGES IN AND DISAGREEMENTS WITH ACCOUNTANTS ON ACCOUNTING AND FINANCIAL DISCLOSURE",
        "CHANGES IN AND DISAGREEMENTS WITH ACCOUNTANTS ON ACCOUNTING AND FINANCIAL DISCLOSURES",
    ],
    "9A": ["CONTROLS AND PROCEDURES"],
    "9B": ["OTHER INFORMATION"],
    "9C": ["DISCLOSURE REGARDING FOREIGN JURISDICTIONS THAT PREVENT INSPECTIONS"],
    "10": [
        "DIRECTORS, EXECUTIVE OFFICERS AND CORPORATE GOVERNANCE",
        "DIRECTORS AND EXECUTIVE OFFICERS",
    ],
    "11": ["EXECUTIVE COMPENSATION"],
    "12": [
        "SECURITY OWNERSHIP OF CERTAIN BENEFICIAL OWNERS AND MANAGEMENT",
        "SECURITY OWNERSHIP OF CERTAIN BENEFICIAL OWNERS AND MANAGEMENT AND RELATED STOCKHOLDER MATTERS",
    ],
    "13": [
        "CERTAIN RELATIONSHIPS AND RELATED TRANSACTIONS",
        "CERTAIN RELATIONSHIPS AND RELATED TRANSACTIONS AND DIRECTOR INDEPENDENCE",
    ],
    "14": ["PRINCIPAL ACCOUNTANT FEES AND SERVICES"],
    "15": [
        "EXHIBITS",
        "EXHIBITS AND FINANCIAL STATEMENT SCHEDULES",
        "EXHIBITS AND FINANCIAL STATEMENTS",
        "EXHIBITS FINANCIAL STATEMENT SCHEDULES",
        "INDEX TO EXHIBITS",
    ],
    "SIGNATURES": ["SIGNATURES"],
}

ITEM_TITLES_10K_BY_CANONICAL = {
    "I:4_VOTING_RESULTS_LEGACY": [
        "SUBMISSION OF MATTERS TO A VOTE OF SECURITY HOLDERS",
        "SUBMISSION OF MATTERS TO A VOTE OF SHAREHOLDERS",
    ],
    "I:4_RESERVED": ["RESERVED"],
    "I:4_MINE_SAFETY": ["MINE SAFETY DISCLOSURES"],
    "II:6_SELECTED_FINANCIAL_DATA": ["SELECTED FINANCIAL DATA"],
    "II:6_RESERVED": ["RESERVED"],
    "III:14_CONTROLS_AND_PROCEDURES_LEGACY": ["CONTROLS AND PROCEDURES"],
    "III:14_PRINCIPAL_ACCOUNTANT_FEES": ["PRINCIPAL ACCOUNTANT FEES AND SERVICES"],
    "III:16_PRINCIPAL_ACCOUNTANT_FEES_LEGACY": ["PRINCIPAL ACCOUNTANT FEES AND SERVICES"],
    "IV:14_EXHIBITS_SCHEDULES_REPORTS": [
        "EXHIBITS",
        "EXHIBITS AND FINANCIAL STATEMENT SCHEDULES",
        "EXHIBITS AND FINANCIAL STATEMENTS",
        "EXHIBITS FINANCIAL STATEMENT SCHEDULES",
        "INDEX TO EXHIBITS",
    ],
}


def _digits_only(s: str | None) -> str | None:
    if not s:
        return None
    d = re.sub(r"\D", "", s)
    return d or None


def _cik_10(cik: int | None) -> str | None:
    if cik is None:
        return None
    return str(int(cik)).zfill(10)


def _make_doc_id(cik10: str | None, acc: str | None) -> str | None:
    # Keep this stable and readable; you can later switch to a hash if desired.
    if acc is None:
        return None
    return f"{cik10}:{acc}" if cik10 else f"UNK:{acc}"


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


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


def _roman_to_int(s: str) -> int | None:
    """
    Convert a roman numeral (I, II, IV, ...) to int.
    Returns None if the string is not a valid roman numeral.
    """
    s = s.strip().upper()
    if not s or not re.fullmatch(r"[IVXLCDM]+", s):
        return None

    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    prev = 0
    for ch in reversed(s):
        v = values[ch]
        if v < prev:
            total -= v
        else:
            total += v
            prev = v
    # Validate by round-trip to reduce false positives.
    if total <= 0:
        return None
    return total


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
        if re.fullmatch(r"\d{8}", s):
            try:
                return datetime.strptime(s, "%Y%m%d").date()
            except Exception:
                return None
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            try:
                return datetime.strptime(s, "%Y-%m-%d").date()
            except Exception:
                return None
    return None


def _load_item_regime_spec() -> dict | None:
    path = Path(__file__).resolve().parent / "item_regime_10k.json"
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


_ITEM_TITLE_LOOKUP_10K = _build_title_lookup(ITEM_TITLES_10K)
_ITEM_TITLES_10K_NORM: dict[str, set[str]] = {}
for _item_id, _titles in ITEM_TITLES_10K.items():
    normed = {_normalize_heading_text(t) for t in _titles}
    _ITEM_TITLES_10K_NORM[_item_id] = {t for t in normed if t}

_ALLOWED_ITEM_LETTERS_10K: dict[int, set[str]] = {
    1: {"A", "B", "C"},
    7: {"A"},
    9: {"A", "B", "C"},
}
_ITEM_REGIME_SPEC = _load_item_regime_spec()
# Regime spec is optional; missing or unreadable specs fall back to permissive behavior.
_ITEM_REGIME_ITEMS = _ITEM_REGIME_SPEC.get("items", {}) if _ITEM_REGIME_SPEC else {}
_ITEM_REGIME_LEGACY = (
    {entry["slot"]: entry for entry in _ITEM_REGIME_SPEC.get("legacy_slots", [])}
    if _ITEM_REGIME_SPEC
    else {}
)
_ITEM_REGIME_BY_ID: dict[str, list[tuple[str, dict]]] = {}
if _ITEM_REGIME_SPEC:
    combined = dict(_ITEM_REGIME_ITEMS)
    combined.update(_ITEM_REGIME_LEGACY)
    for key, entry in combined.items():
        item_id = entry.get("item_id")
        if not item_id and ":" in key:
            item_id = key.split(":", 1)[1]
        if not item_id:
            continue
        _ITEM_REGIME_BY_ID.setdefault(item_id, []).append((key, entry))


def _default_part_for_item_id(item_id: str | None) -> str | None:
    if not item_id or not item_id[0].isdigit():
        return None
    m = re.match(r"^(?P<num>\d{1,2})", item_id)
    if not m:
        return None
    n = int(m.group("num"))
    if 1 <= n <= 4:
        return "I"
    if 5 <= n <= 9:
        return "II"
    if 10 <= n <= 14:
        return "III"
    if 15 <= n <= 16:
        return "IV"
    return None


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


def _prefix_is_bullet(prefix: str) -> bool:
    if not prefix:
        return False
    return bool(re.fullmatch(r"[\s\-\*\u2022\u00b7\u2013\u2014]+", prefix))


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

    anchor = re.compile(
        r"(?im)^\s*(?:table\s+of\s+contents?\b|part\s+(?:iv|iii|ii|i)\b|item\s+\d+)",
    )
    m = anchor.search(body, pos=cover_end)
    if m is None:
        return None
    return m.start()


_WRAPPED_TABLE_OF_CONTENTS_PATTERN = re.compile(
    r"(?im)^\s*table(?:\s*\n+\s*|\s+)of(?:\s*\n+\s*|\s+)contents?\b"
)
_WRAPPED_PART_PATTERN = re.compile(r"(?im)^(?P<lead>\s*part)\s*\n+\s*(?P<part>iv|iii|ii|i)\b")
_WRAPPED_ITEM_PATTERN = re.compile(
    r"(?im)^(?P<lead>\s*item)\s*\n+\s*(?P<num>\d+|[ivxlcdm]+)\s*(?P<let>[a-z])?\s*(?P<punc>[\.:])?"
)
_WRAPPED_ITEM_LETTER_PATTERN = re.compile(
    r"(?im)^(?P<lead>\s*item\s+(?P<num>\d+|[ivxlcdm]+))\s*(?P<num_punc>[\.\):\-])?"
    r"\s*\n+\s*(?P<let>[a-z])\s*(?P<punc>[\.\):\-])?(?=\s|$)"
)
_ITEM_SUFFIX_PAREN_PATTERN = re.compile(
    r"(?im)^(?P<lead>\s*item\s+\d+[a-z])\s*\((?P<suffix>[a-z])\)\s*(?P<punc>[\.\):\-])?"
)


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

    toc_entry = re.compile(
        r"\bITEM\s+(?P<num>\d+|[IVXLCDM]+)(?P<let>[A-Z])?\s*[\.:]?\s+.{0,120}?\b(?P<page>\d{1,3})\b(?=\s+(?:ITEM|PART)\b|\s*$)",
        re.IGNORECASE | re.DOTALL,
    )

    last_end: int | None = None
    count = 0
    for mm in toc_entry.finditer(window):
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
    max_item: int,
) -> tuple[str, bool, bool]:
    if not text:
        return text, False, False

    lines = text.splitlines(keepends=True)
    lines_noeol = [line.rstrip("\r\n") for line in lines]

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

    offset = 0
    for idx, line_noeol in enumerate(lines_noeol):
        if PART_ONLY_PREFIX_PATTERN.match(line_noeol):
            if _part_boundary_header_like(lines_noeol, idx, max_item=max_item):
                cut = offset + _leading_ws_len(line_noeol)
                return text[:cut].rstrip(), False, True
        offset += len(lines[idx])

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
) -> tuple[list["_ItemBoundary"], dict[tuple[str | None, str], dict[str, int | bool]]]:
    if not candidates:
        return [], {}

    candidates = sorted(candidates, key=lambda b: b.start)

    def _candidate_end(idx: int, pool: list["_ItemBoundary"]) -> int:
        end_pos = len(body)
        for j in range(idx + 1, len(pool)):
            cand = pool[j]
            if cand.confidence < HEADING_CONF_HIGH:
                continue
            if not _is_plausible_successor(
                pool[idx],
                cand,
                filing_date=filing_date,
                period_end=period_end,
                is_10k=is_10k,
            ):
                continue
            end_pos = cand.start
            break
        return end_pos

    candidate_ends = [_candidate_end(idx, candidates) for idx in range(len(candidates))]
    candidate_approx_len = [
        max(candidate_ends[idx] - b.content_start, 0) for idx, b in enumerate(candidates)
    ]

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

    prev_map: dict[tuple[str | None, str], tuple[str | None, str]] = {}
    next_map: dict[tuple[str | None, str], tuple[str | None, str]] = {}
    for idx, key in enumerate(ordered_keys):
        if idx > 0:
            prev_map[key] = ordered_keys[idx - 1]
        if idx + 1 < len(ordered_keys):
            next_map[key] = ordered_keys[idx + 1]

    min_start_by_key = {key: min(b.start for _, b in entries) for key, entries in by_key.items()}
    max_start_by_key = {key: max(b.start for _, b in entries) for key, entries in by_key.items()}

    def _fallback_select(entries: list[tuple[int, "_ItemBoundary"]]) -> "_ItemBoundary":
        best: "_ItemBoundary" | None = None
        best_score: tuple[int, int, int] | None = None
        for idx, b in entries:
            end = candidate_ends[idx]
            chunk = body[b.content_start : end]
            chunk = chunk.lstrip(" \t:-")
            chunk = _remove_pagination(chunk)
            score = len(chunk.strip())
            score_key = (b.confidence, score, b.start)
            if best_score is None or score_key > best_score:
                best = b
                best_score = score_key
        assert best is not None
        return best

    selected: list["_ItemBoundary"] = []
    meta: dict[tuple[str | None, str], dict[str, int | bool]] = {}
    for key, entries in by_key.items():
        total = len(entries)
        toc_rejected = 0
        ok_entries: list[tuple[int, "_ItemBoundary", str | None, str | None]] = []
        for idx, b in entries:
            part_for_checks = b.item_part
            inferred_part = None
            if is_10k:
                inferred_part = _infer_part_for_item_id(
                    b.item_id,
                    filing_date=filing_date,
                    period_end=period_end,
                    is_10k=is_10k,
                )
                if inferred_part:
                    part_for_checks = inferred_part

            late_restart = False
            if _is_late_item_start(b.item_id, part_for_checks):
                late_restart = _late_item_restart_after(
                    lines,
                    b.line_index,
                    max_non_empty=20,
                    max_item=max_item,
                )

            if b.in_toc_range or late_restart or b.toc_like_line:
                toc_rejected += 1
                continue
            ok_entries.append((idx, b, part_for_checks, inferred_part))

        chosen: "_ItemBoundary"
        verified = True
        if ok_entries:
            prev_key = prev_map.get(key)
            next_key = next_map.get(key)
            prev_start = min_start_by_key.get(prev_key)
            next_start = max_start_by_key.get(next_key)

            def _score(entry: tuple[int, "_ItemBoundary", str | None, str | None]) -> tuple[int, int, int, int, int, int]:
                idx, b, part_for_checks, inferred_part = entry
                ordering_bonus = 0
                if prev_start is not None and b.start > prev_start:
                    ordering_bonus += 1
                if next_start is not None and b.start < next_start:
                    ordering_bonus += 1
                if ordering_bonus == 2:
                    ordering_bonus += 1

                part_bonus = 0
                if is_10k and inferred_part and b.item_part:
                    part_bonus = 1 if b.item_part == inferred_part else -1

                score = {HEADING_CONF_HIGH: 4, HEADING_CONF_MED: 2, HEADING_CONF_LOW: 0}.get(
                    b.confidence, 0
                )
                score += ordering_bonus
                score += part_bonus
                if skip_until_pos is not None and b.start < skip_until_pos:
                    score -= 2
                if b.toc_like_line:
                    score -= 3

                approx_len = candidate_approx_len[idx]
                return (score, b.confidence, ordering_bonus, part_bonus, approx_len, b.start)

            chosen = max(ok_entries, key=_score)[1]
        else:
            only_toc = all(
                b.in_toc_range or b.toc_like_line for _, b in entries
            )
            if only_toc:
                continue
            chosen = _fallback_select(entries)
            verified = False

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


@dataclass(frozen=True)
class _ItemBoundary:
    start: int
    content_start: int
    item_part: str | None
    item_id: str
    line_index: int
    confidence: int
    in_toc_range: bool = False
    toc_like_line: bool = False


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
) -> list[dict[str, str | bool | None]]:
    """
    Extract filing item sections from `full_text`.

    Returns a list of dicts with:
      - item_part: roman numeral part when detected (e.g., 'I', 'II'); may be None
      - item_id: normalized item id (e.g., '1', '1A')
      - item: combined key '<part>:<id>' when part exists, else '<id>'
      - full_text: extracted text for the item (pagination artifacts removed)
      - canonical_item: regime-stable meaning when available
      - exists_by_regime: True/False when regime rules can be evaluated, else None
      - item_status: active/reserved/optional/unknown
      - _heading_line (clean) / _heading_line_raw / _heading_line_index / _heading_offset when diagnostics=True
      - when drop_impossible=True, items with exists_by_regime == False are dropped
      - when repair_boundaries=True, high-confidence end-boundary truncation is applied

    The function does not emit TOC rows; TOC is only used internally to avoid false starts.
    """
    if not full_text:
        return []

    form = (form_type or "").strip().upper()
    if re.match(r"^10-?K[/-]A", form) or re.match(r"^10-?Q[/-]A", form):
        # Skip amended filings (10-K-A/10-Q-A) to avoid duplicate extraction.
        return []
    is_10k = form.startswith("10K") or form.startswith("10-K")
    if form.startswith("10Q") or form.startswith("10-Q"):
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

    boundaries, selection_meta = _select_best_boundaries(
        boundaries,
        lines=lines,
        body=body,
        skip_until_pos=skip_until_pos,
        filing_date=filing_date_parsed,
        period_end=period_end_parsed,
        is_10k=is_10k,
        max_item=max_item_number,
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
        chunk = body[b.content_start : end]
        chunk = chunk.lstrip(" \t:-")
        chunk = _remove_pagination(chunk)
        chunk = _trim_trailing_part_marker(chunk)
        truncated_successor = False
        truncated_part = False
        item_id = b.item_id
        part = b.item_part
        inferred_part = _infer_part_for_item_id(
            item_id,
            filing_date=filing_date_parsed,
            period_end=period_end_parsed,
            is_10k=is_10k,
        )
        part_for_detection = inferred_part or part
        raw_chunk: str | None = None
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
                max_item=max_item_number,
            )
            leak_hit = embedded_headings.find_strong_leak(
                chunk,
                current_item_id=item_id,
                current_part=part_for_detection,
                next_item_id=next_item_id,
                next_part=next_part,
            )
            if leak_hit and 0 <= leak_hit.char_pos < len(chunk):
                raw_chunk = chunk
                chunk = chunk[: leak_hit.char_pos].rstrip()

        if inferred_part and (part is None or part != inferred_part):
            part = inferred_part
        item_key = f"{part}:{item_id}" if part else item_id

        record = {
            "item_part": part,
            "item_id": item_id,
            "item": item_key,
            "full_text": chunk,
        }
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
            else:
                record["_heading_line_raw"] = ""
                record["_heading_line"] = ""
                record["_heading_line_index"] = None
                record["_heading_offset"] = None
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


def parse_filename_minimal(filename: str) -> dict:
    """
    Parse all usable info from filename only.
    Returns dict with parse_ok plus components.
    """
    m = FILENAME_PATTERN.match(filename)
    if not m:
        return {
            "filename_parse_ok": False,
            "file_date_filename": None,
            "document_type_filename": None,
            "cik": None,
            "cik_10": None,
            "accession_number": None,
            "accession_nodash": None,
            "doc_id": None,
        }

    date_str = m.group(1)
    doc_type = m.group(2)
    try:
        cik_int = int(m.group(3))
    except Exception:
        cik_int = None
    acc = m.group(4)

    cik10 = _cik_10(cik_int)
    acc_nodash = _digits_only(acc)
    doc_id = _make_doc_id(cik10, acc)

    return {
        "filename_parse_ok": True,
        "file_date_filename": date_str,          # parse to Date at write time
        "document_type_filename": doc_type,
        "cik": cik_int,
        "cik_10": cik10,
        "accession_number": acc,
        "accession_nodash": acc_nodash,
        "doc_id": doc_id,
    }


@dataclass
class RawTextSchema:
    schema = {
        "doc_id": pl.Utf8,
        "cik": pl.Int64,
        "cik_10": pl.Utf8,
        "accession_number": pl.Utf8,
        "accession_nodash": pl.Utf8,
        "file_date_filename": pl.Utf8,            # will cast to Date on write
        "document_type_filename": pl.Utf8,
        "filename": pl.Utf8,
        "zip_member_path": pl.Utf8,
        "filename_parse_ok": pl.Boolean,
        "full_text": pl.Utf8,
    }


@dataclass
class ParsedFilingSchema:
    schema = {
        "doc_id": pl.Utf8,
        "cik": pl.Int64,
        "cik_10": pl.Utf8,
        "accession_number": pl.Utf8,
        "accession_nodash": pl.Utf8,
        "filing_date": pl.Utf8,  # will cast to Date on write
        "filing_date_header": pl.Utf8,
        "period_end": pl.Utf8,  # will cast to Date on write
        "period_end_header": pl.Utf8,
        "file_date_filename": pl.Utf8,  # will cast to Date on write
        "document_type_filename": pl.Utf8,
        "filename": pl.Utf8,
        "zip_member_path": pl.Utf8,
        "filename_parse_ok": pl.Boolean,
        "cik_header_primary": pl.Utf8,
        "ciks_header_secondary": pl.List(pl.Utf8),
        "accession_header": pl.Utf8,
        "cik_conflict": pl.Boolean,
        "accession_conflict": pl.Boolean,
        "full_text": pl.Utf8,
    }


@dataclass
class FilingItemSchema:
    schema = {
        "doc_id": pl.Utf8,
        "cik": pl.Int64,
        "cik_10": pl.Utf8,
        "accession_number": pl.Utf8,
        "accession_nodash": pl.Utf8,
        "file_date_filename": pl.Date,
        "filing_date": pl.Date,
        "period_end": pl.Date,
        "document_type_filename": pl.Utf8,
        "filename": pl.Utf8,
        "item_part": pl.Utf8,
        "item_id": pl.Utf8,
        "item": pl.Utf8,
        "canonical_item": pl.Utf8,
        "exists_by_regime": pl.Boolean,
        "item_status": pl.Utf8,
        "full_text": pl.Utf8,
    }


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

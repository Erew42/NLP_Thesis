from __future__ import annotations

import json
import re
from bisect import bisect_right
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import polars as pl


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

TOC_MARKER_PATTERN = re.compile(
    # Be conservative: "INDEX" appears in many non-TOC contexts (e.g., "Exhibit Index"),
    # and false TOC masking can suppress real item headings.
    r"\btable\s+of\s+contents?\b|\btable\s+of\s+content\b",
    re.IGNORECASE,
)
TOC_HEADER_LINE_PATTERN = re.compile(r"^\s*table\s+of\s+contents?\s*$", re.IGNORECASE)
ITEM_WORD_PATTERN = re.compile(r"\bITEM\b", re.IGNORECASE)
PART_MARKER_PATTERN = re.compile(r"\bPART\s+(?P<part>IV|III|II|I)\b(?!\s*,)", re.IGNORECASE)
PART_LINESTART_PATTERN = re.compile(r"^\s*PART\s+(?P<part>IV|III|II|I)\b", re.IGNORECASE)
ITEM_CANDIDATE_PATTERN = re.compile(
    r"\bITEM\s+(?P<num>\d+|[IVXLCDM]+)(?P<let>[A-Z])?\s*[\.:]?",
    re.IGNORECASE,
)
ITEM_LINESTART_PATTERN = re.compile(
    r"^\s*(?:PART\s+[IVXLCDM]+\s*[:\-]?\s*)?ITEM\s+(?P<num>\d+|[IVXLCDM]+)(?P<let>[A-Z])?\b",
    re.IGNORECASE,
)
NUMERIC_DOT_HEADING_PATTERN = re.compile(
    r"^\s*(?P<num>\d{1,2})(?P<let>[A-Z])?\s*[\.\):\-]?\s+(?P<title>.+?)\s*$",
    re.IGNORECASE,
)
TOC_DOT_LEADER_PATTERN = re.compile(r"\.{2,}\s*\d{1,4}\s*$")
CONTINUED_PATTERN = re.compile(r"\bcontinued\b", re.IGNORECASE)

PAGE_HYPHEN_PATTERN = re.compile(r"^\s*-\d{1,4}-\s*$")
PAGE_NUMBER_PATTERN = re.compile(r"^\s*\d{1,4}\s*$")
PAGE_ROMAN_PATTERN = re.compile(r"^\s*[ivxlcdm]{1,6}\s*$", re.IGNORECASE)
PAGE_OF_PATTERN = re.compile(r"^\s*page\s+\d+\s*(?:of\s+\d+)?\s*$", re.IGNORECASE)

EMPTY_ITEM_PATTERN = re.compile(
    r"^\s*(?:\(?[a-z]\)?\s*)?(?:none\.?|n/?a\.?|not applicable\.?|not required\.?|\[reserved\]|reserved)\s*$",
    re.IGNORECASE,
)

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
    "16": ["FORM 10-K SUMMARY", "FORM 10K SUMMARY"],
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
    idx = full_text.find("</Header>")
    if idx == -1:
        return full_text
    return full_text[idx + len("</Header>") :]


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
    text = TOC_DOT_LEADER_PATTERN.sub("", text)
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


def _looks_like_toc_heading_line(lines: list[str], idx: int, *, max_early: int = 400) -> bool:
    if idx > max_early:
        return False
    line = lines[idx]
    if not line:
        return False
    if TOC_DOT_LEADER_PATTERN.search(line) or re.search(r"\s+\d{1,4}\s*$", line):
        return True

    j = idx + 1
    max_scan = min(len(lines), idx + 5)
    while j < max_scan and lines[j].strip() == "":
        j += 1
    if j < len(lines) and _pageish_line(lines[j]):
        return len(line.strip()) <= 160
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
            date_value = period_end
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
    if not (TOC_DOT_LEADER_PATTERN.search(line) or re.search(r"\s+\d{1,4}\s*$", line)):
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
    if toc_end_pos is not None:
        return toc_end_pos
    toc_end_line = _infer_toc_end_line_by_titles(lines, lookup, max_item=max_item)
    if toc_end_line is None:
        return None
    idx = toc_end_line + 1
    if idx < len(line_starts):
        return line_starts[idx]
    return None


_WRAPPED_TABLE_OF_CONTENTS_PATTERN = re.compile(
    r"(?im)^\s*table\s*\n+\s*of(?:\s*\n+\s*|\s+)\s*contents?\b"
)
_WRAPPED_PART_PATTERN = re.compile(r"(?im)^(?P<lead>\s*part)\s*\n+\s*(?P<part>iv|iii|ii|i)\b")
_WRAPPED_ITEM_PATTERN = re.compile(
    r"(?im)^(?P<lead>\s*item)\s*\n+\s*(?P<num>\d+|[ivxlcdm]+)\s*(?P<let>[a-z])?\s*(?P<punc>[\.:])?"
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
    return body


def _detect_toc_line_ranges(lines: list[str], *, max_lines: int = 400) -> list[tuple[int, int]]:
    """
    Best-effort detection of a Table of Contents block near the top of a filing.
    Returns inclusive (start_line, end_line) ranges.
    """
    n = min(len(lines), max_lines)
    if n == 0:
        return []

    ranges: list[tuple[int, int]] = []

    # 1) Inline TOC lines (many ITEM tokens on the same line) and lines explicitly marked as TOC/Index.
    inline_lines: list[int] = []
    for i in range(n):
        line = lines[i]
        line_len = len(line)
        item_words = len(ITEM_WORD_PATTERN.findall(line))
        # Avoid flagging extremely long lines (some filings have most content on one line);
        # for those, we rely on the character-based TOC cutoff instead.
        if item_words >= 3 and line_len <= 5_000:
            inline_lines.append(i)
            continue
        if TOC_MARKER_PATTERN.search(line) and item_words >= 1 and line_len <= 8_000:
            inline_lines.append(i)

    for i in inline_lines:
        ranges.append((i, i))

    # 2) Multi-line TOC clusters: many line-start ITEM headings close together near the top.
    heading_lines: list[int] = []
    for i in range(n):
        m = ITEM_LINESTART_PATTERN.match(lines[i])
        if not m:
            continue
        if _normalize_item_id(m.group("num"), m.group("let")) is None:
            continue
        heading_lines.append(i)

    clusters: list[list[int]] = []
    if heading_lines:
        cur = [heading_lines[0]]
        for idx in heading_lines[1:]:
            if idx - cur[-1] <= 6:
                cur.append(idx)
            else:
                clusters.append(cur)
                cur = [idx]
        clusters.append(cur)

    def _pageish(line: str) -> bool:
        return bool(
            PAGE_NUMBER_PATTERN.match(line)
            or PAGE_HYPHEN_PATTERN.match(line)
            or PAGE_ROMAN_PATTERN.match(line)
            or PAGE_OF_PATTERN.match(line)
        )

    toc_markers = {i for i in range(n) if TOC_MARKER_PATTERN.search(lines[i])}

    # Handle split markers like:
    #   TABLE
    #   OF CONTENTS
    # and similar 2-line patterns.
    for i in range(n - 1):
        a = lines[i].strip().lower()
        b = lines[i + 1].strip().lower()
        if a == "table" and b.startswith("of contents"):
            toc_markers.add(i)
            toc_markers.add(i + 1)
        if a == "table of" and b.startswith("contents"):
            toc_markers.add(i)
            toc_markers.add(i + 1)

    chosen: list[tuple[int, int]] = []
    for cl in clusters:
        if len(cl) < 4:
            continue
        if cl[0] > 300:  # TOC is typically early
            continue

        followed = 0
        for li in cl:
            # TOC entries often include a page number (or dot leaders + page) on the same line.
            s = lines[li].strip()
            if re.search(r"\.{3,}\s*\d{1,3}\s*$", s) or re.search(r"\s\d{1,3}\s*$", s):
                followed += 1
                continue
            for j in (li + 1, li + 2, li + 3):
                if j < n and _pageish(lines[j]):
                    followed += 1
                    break

        has_marker_near = any((cl[0] - 10) <= mi <= (cl[-1] + 10) for mi in toc_markers)

        # Consider the whole block: a true TOC block is mostly headings and pagination markers,
        # while real content contains narrative lines that break this density.
        nonempty_block = [lines[i].strip() for i in range(cl[0], cl[-1] + 1) if lines[i].strip()]
        if nonempty_block:
            headingish = 0
            for s in nonempty_block:
                m = ITEM_LINESTART_PATTERN.match(s)
                if m and _normalize_item_id(m.group("num"), m.group("let")) is not None:
                    headingish += 1
                    continue
                if PART_LINESTART_PATTERN.match(s) is not None:
                    headingish += 1
                    continue
                if TOC_HEADER_LINE_PATTERN.match(s):
                    headingish += 1
                    continue
                if _pageish(s):
                    headingish += 1
                    continue
            headingish_ratio = headingish / len(nonempty_block)
        else:
            headingish_ratio = 0.0

        # Require either an explicit marker nearby or strong page-number evidence, plus a high
        # density of headings in the block. This avoids masking real item starts when a TOC header
        # repeats as a page header inside the filing.
        if headingish_ratio >= 0.75 and (has_marker_near or followed >= 2):
            chosen.append((cl[0], cl[-1]))

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
        if m_part is not None:
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
            )
        )

    return boundaries


@dataclass(frozen=True)
class _ItemBoundary:
    start: int
    content_start: int
    item_part: str | None
    item_id: str


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


def extract_filing_items(
    full_text: str,
    *,
    form_type: str | None = None,
    filing_date: str | date | datetime | None = None,
    period_end: str | date | datetime | None = None,
    regime: bool = True,
    diagnostics: bool = False,
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
      - _heading_line/_heading_line_index/_heading_offset when diagnostics=True

    The function does not emit TOC rows; TOC is only used internally to avoid false starts.
    """
    if not full_text:
        return []

    form = (form_type or "").strip().upper()
    is_10k = form.startswith("10K") or form.startswith("10-K")
    if form.startswith("10Q") or form.startswith("10-Q"):
        allowed_parts = {"I", "II"}
    else:
        # Default to 10-K style parts (I-IV). This also prevents accidental matches like "Part D".
        allowed_parts = {"I", "II", "III", "IV"}

    filing_date_parsed = _parse_date(filing_date)
    period_end_parsed = _parse_date(period_end)
    title_lookup = _build_title_lookup_10k(
        filing_date=filing_date_parsed,
        period_end=period_end_parsed,
        enable_regime=bool(regime) and is_10k,
    )
    # Regime-aware lookup only gates fallback/title-only matching; explicit ITEM headings still pass through.

    body = _normalize_newlines(_strip_leading_header_block(full_text))
    body = _repair_wrapped_headings(body)
    lines = body.split("\n")
    head = lines[: min(len(lines), 400)]
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

        scan_sparse_layout = True if attempt else sparse_layout
        boundaries = []
        current_part: str | None = None

        # Build candidates by scanning lines in order while tracking PART markers within each line.
        for i, line in enumerate(lines):
            if i in toc_mask:
                continue

            # Skip obvious page-header TOC repeats inside items (keep content otherwise).
            if TOC_HEADER_LINE_PATTERN.match(line):
                continue

            events: list[tuple[int, str, str | None, re.Match[str] | None]] = []

            if scan_sparse_layout:
                for m in PART_MARKER_PATTERN.finditer(line):
                    part = m.group("part").upper()
                    if part in allowed_parts:
                        events.append((m.start(), "part", part, m))
            else:
                m = PART_LINESTART_PATTERN.match(line)
                if m is not None:
                    part = m.group("part").upper()
                    if part in allowed_parts:
                        events.append((m.start(), "part", part, m))

            for m in ITEM_CANDIDATE_PATTERN.finditer(line):
                events.append((m.start(), "item", None, m))

            if not events:
                continue

            events.sort(key=lambda t: t[0])
            last_part_end: int | None = None
            is_toc_marker = TOC_MARKER_PATTERN.search(line) is not None
            item_word_count = 0
            if i < 400 or is_toc_marker:
                item_word_count = len(ITEM_WORD_PATTERN.findall(line))
            for _, kind, part, m in events:
                if kind == "part":
                    assert m is not None
                    current_part = part
                    last_part_end = m.end()
                    continue

                assert m is not None
                item_id = _normalize_item_id(
                    m.group("num"),
                    m.group("let"),
                    max_item=max_item_number,
                )
                if item_id is None:
                    continue
                content_adjust = 0
                if is_10k and m.group("let"):
                    base_id = _normalize_item_id(
                        m.group("num"),
                        None,
                        max_item=max_item_number,
                    )
                    base_num = int(base_id) if base_id and base_id.isdigit() else None
                    let = m.group("let").upper()
                    glued = m.end() < len(line) and line[m.end()].isalpha()
                    if not _item_letter_allowed_10k(base_num, let):
                        if glued and base_id:
                            item_id = base_id
                            content_adjust = -1
                        else:
                            continue
                    elif glued and _glued_title_matches_base(base_id, let, line, m):
                        item_id = base_id or item_id
                        content_adjust = -1

                # Basic TOC-line filters (only for reasonably short early lines).
                if i < 400:
                    if item_word_count >= 3 and len(line) <= 5_000:
                        continue
                # Only skip TOC-marker lines when they look like true TOC listings (many ITEM tokens).
                # Some filings embed a repeated "Table of Contents" page header alongside real headings.
                if is_toc_marker and item_word_count >= 3 and len(line) <= 8_000:
                    continue
                if _looks_like_toc_heading_line(lines, i):
                    continue

                abs_start = line_starts[i] + m.start()
                if toc_end_pos is not None and abs_start < toc_end_pos:
                    continue

                # Heuristics to avoid cross-references:
                prefix = line[: m.start()]
                is_line_start = prefix.strip() == "" or _prefix_is_bullet(prefix)
                part_near_item = last_part_end is not None and (m.start() - last_part_end) <= 60

                # Only accept headings that look like real section starts (line-start or 'PART .. ITEM ..').
                # This intentionally rejects mid-sentence cross-references like "Part II, Item 7 ...".
                accept = is_line_start or part_near_item
                if not accept and scan_sparse_layout:
                    # For filings with very few line breaks, headings often follow sentence punctuation.
                    # Keep this strict elsewhere to avoid mid-sentence cross-reference matches.
                    k = abs_start - 1
                    while k >= 0 and body[k].isspace():
                        k -= 1
                    prev_char = body[k] if k >= 0 else ""
                    if prev_char in ".:;!?":
                        prev = prefix[-16:].lower()
                        if not re.search(r"(see|in|under|from|to)\s+$", prev):
                            accept = True

                if not accept:
                    continue

                if is_line_start and _starts_with_lowercase_title(line, m):
                    continue

                # Ignore "(continued)" headings; they are typically page-header repeats.
                # Only treat it as a continuation marker when it appears immediately after the heading,
                # not elsewhere later in the same (potentially very long) line.
                suffix = line[m.end() : m.end() + 64]
                if re.search(r"(?i)^\s*[\(\[]?\s*continued\b", suffix):
                    continue
                if re.match(r"(?i)^\s*[\(\[]\s*[a-z0-9]", suffix):
                    continue

                start_abs = line_starts[i] + m.start()
                content_start_abs = line_starts[i] + m.end() + content_adjust
                boundaries.append(
                    _ItemBoundary(
                        start=start_abs,
                        content_start=content_start_abs,
                        item_part=current_part,
                        item_id=item_id,
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

    boundaries.sort(key=lambda b: b.start)

    # De-duplicate per (part, item_id) by preferring the boundary that yields the most content.
    # This helps when TOCs or page-header repeats leak through and appear before the real section start.
    best_by_key: dict[tuple[str | None, str], tuple[int, int, _ItemBoundary]] = {}
    for idx, b in enumerate(boundaries):
        end = boundaries[idx + 1].start if idx + 1 < len(boundaries) else len(body)
        chunk = body[b.content_start : end]
        chunk = chunk.lstrip(" \t:-")
        chunk = _remove_pagination(chunk)
        score = len(chunk.strip())
        key = (b.item_part, b.item_id)
        prev = best_by_key.get(key)
        if prev is None or score > prev[0] or (score == prev[0] and b.start > prev[1]):
            best_by_key[key] = (score, b.start, b)

    boundaries = [t[2] for t in best_by_key.values()]
    boundaries.sort(key=lambda b: b.start)

    out_items: list[dict[str, str | None]] = []
    for idx, b in enumerate(boundaries):
        end = boundaries[idx + 1].start if idx + 1 < len(boundaries) else len(body)
        chunk = body[b.content_start : end]
        chunk = chunk.lstrip(" \t:-")
        chunk = _remove_pagination(chunk)

        part = b.item_part
        item_id = b.item_id
        item_key = f"{part}:{item_id}" if part else item_id

        record = {
            "item_part": part,
            "item_id": item_id,
            "item": item_key,
            "full_text": chunk,
        }
        if diagnostics:
            idx = bisect_right(line_starts, b.start) - 1
            if 0 <= idx < len(lines):
                record["_heading_line"] = lines[idx]
                record["_heading_line_index"] = idx
                record["_heading_offset"] = b.start - line_starts[idx]
            else:
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

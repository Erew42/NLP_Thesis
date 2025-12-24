from __future__ import annotations

import re
from dataclasses import dataclass

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
CONTINUED_PATTERN = re.compile(r"\bcontinued\b", re.IGNORECASE)

PAGE_HYPHEN_PATTERN = re.compile(r"^\s*-\d{1,4}-\s*$")
PAGE_NUMBER_PATTERN = re.compile(r"^\s*\d{1,4}\s*$")
PAGE_ROMAN_PATTERN = re.compile(r"^\s*[ivxlcdm]{1,6}\s*$", re.IGNORECASE)
PAGE_OF_PATTERN = re.compile(r"^\s*page\s+\d+\s*(?:of\s+\d+)?\s*$", re.IGNORECASE)

EMPTY_ITEM_PATTERN = re.compile(
    r"^\s*(?:\(?[a-z]\)?\s*)?(?:none\.?|n/?a\.?|not applicable\.?|not required\.?|\[reserved\]|reserved)\s*$",
    re.IGNORECASE,
)


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


def extract_filing_items(
    full_text: str,
    *,
    form_type: str | None = None,
    max_item_number: int = 20,
) -> list[dict[str, str | None]]:
    """
    Extract filing item sections from `full_text`.

    Returns a list of dicts with:
      - item_part: roman numeral part when detected (e.g., 'I', 'II'); may be None
      - item_id: normalized item id (e.g., '1', '1A')
      - item: combined key '<part>:<id>' when part exists, else '<id>'
      - full_text: extracted text for the item (pagination artifacts removed)

    The function does not emit TOC rows; TOC is only used internally to avoid false starts.
    """
    if not full_text:
        return []

    form = (form_type or "").strip().upper()
    if form.startswith("10Q") or form.startswith("10-Q"):
        allowed_parts = {"I", "II"}
    else:
        # Default to 10-K style parts (I-IV). This also prevents accidental matches like "Part D".
        allowed_parts = {"I", "II", "III", "IV"}

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

                # Basic TOC-line filters (only for reasonably short early lines).
                if i < 400:
                    if item_word_count >= 3 and len(line) <= 5_000:
                        continue
                # Only skip TOC-marker lines when they look like true TOC listings (many ITEM tokens).
                # Some filings embed a repeated "Table of Contents" page header alongside real headings.
                if is_toc_marker and item_word_count >= 3 and len(line) <= 8_000:
                    continue

                abs_start = line_starts[i] + m.start()
                if toc_end_pos is not None and abs_start < toc_end_pos:
                    continue

                # Heuristics to avoid cross-references:
                prefix = line[: m.start()]
                is_line_start = prefix.strip() == ""
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

                # Ignore "(continued)" headings; they are typically page-header repeats.
                # Only treat it as a continuation marker when it appears immediately after the heading,
                # not elsewhere later in the same (potentially very long) line.
                suffix = line[m.end() : m.end() + 64]
                if re.search(r"(?i)^\s*[\(\[]?\s*continued\b", suffix):
                    continue

                start_abs = line_starts[i] + m.start()
                content_start_abs = line_starts[i] + m.end()
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

        out_items.append(
            {
                "item_part": part,
                "item_id": item_id,
                "item": item_key,
                "full_text": chunk,
            }
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
        "document_type_filename": pl.Utf8,
        "filename": pl.Utf8,
        "item_part": pl.Utf8,
        "item_id": pl.Utf8,
        "item": pl.Utf8,
        "full_text": pl.Utf8,
    }


def parse_header(full_text: str, header_search_limit: int = HEADER_SEARCH_LIMIT_DEFAULT) -> dict:
    """
    Extract header metadata (CIKs, accession, filing date) from the top of a filing.
    """
    header = full_text[:header_search_limit]

    header_ciks = CIK_HEADER_PATTERN.findall(header)
    header_ciks_int_set = {int(c) for c in header_ciks if c.isdigit()}

    date_match = DATE_HEADER_PATTERN.search(header)
    header_filing_date_str = date_match.group(1) if date_match else None

    acc_match = ACC_HEADER_PATTERN.search(header)
    header_accession_str = acc_match.group(1) if acc_match else None

    primary_header_cik = header_ciks[0] if header_ciks else None
    secondary_ciks = header_ciks[1:] if len(header_ciks) > 1 else []

    return {
        "header_ciks_int_set": header_ciks_int_set,
        "header_filing_date_str": header_filing_date_str,
        "header_accession_str": header_accession_str,
        "primary_header_cik": primary_header_cik,
        "secondary_ciks": secondary_ciks,
    }

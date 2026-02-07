# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False
from __future__ import annotations

from array import array
import re

from . import embedded_headings
from .heuristics import (
    HEADING_CONF_HIGH,
    HEADING_CONF_LOW,
    HEADING_CONF_MED,
    _has_content_after,
    _heading_suffix_looks_like_prose,
    _heading_title_matches_item,
    _line_has_compound_items,
    _looks_like_toc_heading_line,
    _normalize_item_match,
    _pageish_line,
    _part_marker_is_heading,
    _prefix_is_part_only,
    _prefix_looks_like_cross_ref,
    _prefix_part_tail,
    _starts_with_lowercase_title,
)
from .patterns import (
    COMBINED_PART_ITEM_PATTERN,
    CROSS_REF_PART_PATTERN,
    CROSS_REF_PREFIX_PATTERN,
    EMPTY_ITEM_PATTERN,
    ITEM_CANDIDATE_PATTERN,
    ITEM_WORD_PATTERN,
    PART_ONLY_PREFIX_PATTERN,
    PART_PREFIX_TAIL_PATTERN,
    PART_LINESTART_PATTERN,
    PART_MARKER_PATTERN,
    TOC_HEADER_LINE_PATTERN,
    TOC_MARKER_PATTERN,
)


cdef inline bint _contains_item_hint(str line):
    if not line:
        return False
    return ("ITEM" in line) or ("Item" in line) or ("item" in line)


cdef inline bint _contains_part_hint(str line):
    if not line:
        return False
    return ("PART" in line) or ("Part" in line) or ("part" in line)


cdef inline bint _prefix_is_bullet_fast(str prefix):
    cdef Py_ssize_t i, n
    cdef str ch
    if not prefix:
        return False
    n = len(prefix)
    for i in range(n):
        ch = prefix[i]
        if ch.isspace() or ch == "-" or ch == "*" or ch == "\u2022" or ch == "\u00b7" or ch == "\u2013" or ch == "\u2014":
            continue
        return False
    return True


cdef inline bint _prefix_is_part_only_fast(str prefix):
    if not prefix:
        return False
    return PART_ONLY_PREFIX_PATTERN.match(prefix) is not None


cdef inline object _prefix_part_tail_fast(str prefix):
    cdef object m
    if not prefix:
        return None
    m = PART_PREFIX_TAIL_PATTERN.search(prefix)
    if m is None:
        return None
    return m.group("part").upper()


cdef inline bint _prefix_looks_like_cross_ref_fast(str prefix):
    cdef str tail
    cdef Py_ssize_t n
    if not prefix or not prefix.strip():
        return False
    tail = prefix.strip()
    n = len(tail)
    if n > 80:
        tail = tail[n - 80 :]
    return (
        CROSS_REF_PREFIX_PATTERN.search(tail) is not None
        or CROSS_REF_PART_PATTERN.search(tail) is not None
    )


cdef inline bint _starts_with_lowercase_title_fast(str line, object match):
    cdef str suffix
    cdef Py_ssize_t i, n
    cdef str ch
    suffix = line[match.end() :]
    if suffix and suffix[0].islower():
        if match.end() > 0 and line[match.end() - 1].isspace():
            return True
        return False
    n = len(suffix)
    for i in range(n):
        ch = suffix[i]
        if ch.isalpha():
            return ch.islower()
    return False


cdef object _next_valid_part_match_sparse(object part_iter, str line, set allowed_parts):
    cdef object match
    cdef str part
    while True:
        try:
            match = next(part_iter)
        except StopIteration:
            return None
        if not _part_marker_is_heading(line, match):
            continue
        part = match.group("part").upper()
        if part in allowed_parts:
            return match


cpdef list scan_part_markers_v2_fast(
    list lines,
    object line_starts,
    set allowed_parts,
    bint scan_sparse_layout,
    set toc_mask,
    bint is_10q=False,
):
    cdef Py_ssize_t i, n_lines, match_start
    cdef long long start_abs
    cdef bint allow_form_header
    cdef bint high_confidence
    cdef bint has_part_hint
    cdef str line
    cdef str part
    cdef str prefix
    cdef str suffix
    cdef object match
    cdef object markers = []
    cdef object filtered
    cdef object marker
    cdef object starts_arr = array("q", line_starts)
    cdef long long[:] starts_mv = starts_arr
    cdef bint seen_part_i = False
    cdef bint seen_part_ii = False

    n_lines = len(lines)
    for i in range(n_lines):
        line = lines[i]
        if not line:
            continue
        if i in toc_mask:
            continue
        has_part_hint = _contains_part_hint(line)
        if not has_part_hint:
            continue

        if scan_sparse_layout:
            for match in PART_MARKER_PATTERN.finditer(line):
                part = match.group("part").upper()
                if part not in allowed_parts:
                    continue
                prefix = line[: match.start()]
                allow_form_header = False
                if prefix.strip() and not _prefix_is_bullet_fast(prefix):
                    if is_10q and not _prefix_looks_like_cross_ref_fast(prefix):
                        if re.search(r"(?i)form\s+10-q|quarterly report", prefix):
                            suffix = line[match.end() :].lower()
                            if part == "I" and "financial" in suffix:
                                allow_form_header = True
                            elif part == "II" and "other" in suffix:
                                allow_form_header = True
                    if not allow_form_header:
                        continue

                high_confidence = _part_marker_is_heading(line, match)
                if allow_form_header:
                    high_confidence = True
                if not high_confidence and match.start() == 0:
                    if ITEM_WORD_PATTERN.search(line) and not _looks_like_toc_heading_line(
                        lines, i
                    ):
                        high_confidence = True
                if not high_confidence:
                    continue

                match_start = <Py_ssize_t>match.start()
                start_abs = starts_mv[i] + match_start
                markers.append((start_abs, part, i, high_confidence))
        else:
            match = PART_LINESTART_PATTERN.match(line)
            if match is None:
                continue
            part = match.group("part").upper()
            if part not in allowed_parts:
                continue
            prefix = line[: match.start()]
            allow_form_header = False
            if prefix.strip() and not _prefix_is_bullet_fast(prefix):
                if is_10q and not _prefix_looks_like_cross_ref_fast(prefix):
                    if re.search(r"(?i)form\s+10-q|quarterly report", prefix):
                        suffix = line[match.end() :].lower()
                        if part == "I" and "financial" in suffix:
                            allow_form_header = True
                        elif part == "II" and "other" in suffix:
                            allow_form_header = True
                if not allow_form_header:
                    continue

            high_confidence = _part_marker_is_heading(line, match)
            if allow_form_header:
                high_confidence = True
            if not high_confidence and match.start() == 0:
                if ITEM_WORD_PATTERN.search(line) and not _looks_like_toc_heading_line(
                    lines, i
                ):
                    high_confidence = True
            if not high_confidence:
                continue

            match_start = <Py_ssize_t>match.start()
            start_abs = starts_mv[i] + match_start
            markers.append((start_abs, part, i, high_confidence))

    if not is_10q or not markers:
        return markers

    filtered = []
    for marker in markers:
        part = marker[1]
        if part == "I":
            if not seen_part_i and not seen_part_ii:
                filtered.append(marker)
                seen_part_i = True
            continue
        if part == "II":
            if seen_part_i and not seen_part_ii:
                filtered.append(marker)
                seen_part_ii = True
            continue
    return filtered


cpdef list scan_item_boundaries_fast(
    list lines,
    object line_starts,
    str body,
    bint is_10k,
    int max_item_number,
    set allowed_parts,
    bint scan_sparse_layout,
    set toc_mask,
    list toc_window_flags,
    dict toc_cache,
    bint extraction_regime_v2,
):
    cdef Py_ssize_t i, n_lines, k
    cdef bint line_in_toc_range
    cdef bint line_has_item_word
    cdef bint is_toc_marker
    cdef bint content_after
    cdef bint toc_window_like
    cdef bint line_toc_like
    cdef bint compound_line
    cdef bint title_match
    cdef bint candidate_toc_like
    cdef bint is_line_start
    cdef bint part_near_item
    cdef bint accept
    cdef bint midline
    cdef bint has_item_hint
    cdef bint has_part_hint
    cdef int item_word_count
    cdef int content_adjust
    cdef int confidence
    cdef long long start_abs
    cdef long long content_start_abs
    cdef long long abs_start
    cdef object boundaries = []
    cdef object starts_arr = array("q", line_starts)
    cdef long long[:] starts_mv = starts_arr
    cdef object current_part = None
    cdef object toc_part = None
    cdef object part_iter
    cdef object item_iter
    cdef object next_part
    cdef object next_item
    cdef object m
    cdef object combined_match
    cdef object item_match
    cdef object item_id
    cdef object line_part
    cdef object line_part_end
    cdef object prefix_part
    cdef str line
    cdef str prefix
    cdef str suffix
    cdef str part
    cdef str probe
    cdef str prev_char
    cdef object item_part
    cdef object part_hint

    n_lines = len(lines)
    for i in range(n_lines):
        line = lines[i]
        line_in_toc_range = i in toc_mask
        if not line_in_toc_range:
            toc_part = None

        if TOC_HEADER_LINE_PATTERN.match(line):
            continue

        has_item_hint = _contains_item_hint(line)
        has_part_hint = _contains_part_hint(line)
        if not has_item_hint and not has_part_hint:
            continue

        line_has_item_word = has_item_hint and (ITEM_WORD_PATTERN.search(line) is not None)

        combined_match = (
            COMBINED_PART_ITEM_PATTERN.match(line)
            if (extraction_regime_v2 and has_item_hint and has_part_hint)
            else None
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
                        if not _prefix_looks_like_cross_ref_fast(prefix):
                            suffix = line[item_match.end() : item_match.end() + 64]
                            if not re.search(r"(?i)^\s*[\(\[]?\s*continued\b", suffix):
                                if not re.match(
                                    r"(?i)^\s*[\(\[]\s*[a-z0-9]",
                                    suffix,
                                ):
                                    is_toc_marker = TOC_MARKER_PATTERN.search(line) is not None
                                    item_word_count = 0
                                    if has_item_hint and (i < 800 or is_toc_marker or line_in_toc_range):
                                        item_word_count = len(
                                            ITEM_WORD_PATTERN.findall(line)
                                        )
                                    line_toc_like = _looks_like_toc_heading_line(lines, i)
                                    if (
                                        item_word_count >= 3
                                        and len(line) <= 5000
                                        and (i < 800 or is_toc_marker or line_in_toc_range)
                                    ):
                                        line_toc_like = True
                                    if (
                                        is_toc_marker
                                        and item_word_count >= 1
                                        and len(line) <= 8000
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
                                                scan_sparse_layout and len(line) > 2000
                                            ):
                                                line_toc_like = True
                                                toc_window_like = True
                                    compound_line = _line_has_compound_items(line)
                                    title_match = is_10k and _heading_title_matches_item(
                                        item_id, line, item_match
                                    )
                                    confidence = HEADING_CONF_HIGH
                                    if (
                                        is_10k
                                        and not title_match
                                        and _heading_suffix_looks_like_prose(
                                            line[item_match.end() :]
                                        )
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
                                    start_abs = starts_mv[i] + item_match.start()
                                    content_start_abs = (
                                        starts_mv[i]
                                        + item_match.end()
                                        + content_adjust
                                    )
                                    boundaries.append(
                                        (
                                            start_abs,
                                            content_start_abs,
                                            part,
                                            item_id,
                                            i,
                                            confidence,
                                            line_in_toc_range and not content_after,
                                            candidate_toc_like,
                                        )
                                    )
                                    if not line_in_toc_range:
                                        current_part = part
                                    continue

        next_part = None
        part_iter = None
        if has_part_hint:
            if scan_sparse_layout:
                part_iter = PART_MARKER_PATTERN.finditer(line)
                next_part = _next_valid_part_match_sparse(part_iter, line, allowed_parts)
            else:
                m = PART_LINESTART_PATTERN.match(line)
                if m is not None:
                    if not _part_marker_is_heading(line, m):
                        # Preserve Python-path semantics: invalid line-start PART suppresses this line.
                        continue
                    part = m.group("part").upper()
                    if part in allowed_parts:
                        next_part = m

        next_item = None
        item_iter = None
        if has_item_hint:
            item_iter = ITEM_CANDIDATE_PATTERN.finditer(line)
            try:
                next_item = next(item_iter)
            except StopIteration:
                next_item = None

        if next_part is None and next_item is None:
            continue
        line_part = None
        line_part_end = None
        is_toc_marker = TOC_MARKER_PATTERN.search(line) is not None
        item_word_count = 0
        if has_item_hint and (i < 800 or is_toc_marker or line_in_toc_range):
            item_word_count = len(ITEM_WORD_PATTERN.findall(line))
        line_toc_like = _looks_like_toc_heading_line(lines, i)
        if (
            item_word_count >= 3
            and len(line) <= 5000
            and (i < 800 or is_toc_marker or line_in_toc_range)
        ):
            line_toc_like = True
        if is_toc_marker and item_word_count >= 1 and len(line) <= 8000:
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
                if not (scan_sparse_layout and len(line) > 2000):
                    line_toc_like = True
                    toc_window_like = True
        compound_line = _line_has_compound_items(line)

        while next_part is not None or next_item is not None:
            if (
                next_part is not None
                and (next_item is None or next_part.start() <= next_item.start())
            ):
                m = next_part
                part = m.group("part").upper()
                line_part = part
                line_part_end = m.end()
                if line_in_toc_range:
                    if not line_has_item_word:
                        toc_part = part
                elif not line_has_item_word:
                    current_part = part
                if scan_sparse_layout:
                    next_part = _next_valid_part_match_sparse(
                        part_iter, line, allowed_parts
                    )
                else:
                    next_part = None
                continue

            m = next_item
            if item_iter is not None:
                try:
                    next_item = next(item_iter)
                except StopIteration:
                    next_item = None
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

            abs_start = starts_mv[i] + m.start()
            prefix = line[: m.start()]
            is_line_start = (
                prefix.strip() == ""
                or _prefix_is_bullet_fast(prefix)
                or _prefix_is_part_only_fast(prefix)
            )
            prefix_part = _prefix_part_tail_fast(prefix)
            part_near_item = (
                line_part_end is not None and (m.start() - line_part_end) <= 60
            )
            if prefix_part:
                part_near_item = True
            if _prefix_looks_like_cross_ref_fast(prefix):
                continue

            accept = is_line_start or part_near_item
            midline = False
            if not accept and scan_sparse_layout:
                k = <Py_ssize_t>abs_start - 1
                while k >= 0 and body[k].isspace():
                    k -= 1
                prev_char = body[k] if k >= 0 else ""
                if prev_char in ".:;!?":
                    if not _prefix_looks_like_cross_ref_fast(prefix):
                        accept = True
                        midline = True

            if not accept:
                continue

            if _starts_with_lowercase_title_fast(line, m):
                continue

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

            start_abs = starts_mv[i] + m.start()
            content_start_abs = starts_mv[i] + m.end() + content_adjust
            candidate_toc_like = line_toc_like
            if (
                candidate_toc_like
                and toc_window_like
                and scan_sparse_layout
                and m.start() > embedded_headings.EMBEDDED_TOC_START_EARLY_MAX_CHAR
            ):
                candidate_toc_like = False
            boundaries.append(
                (
                    start_abs,
                    content_start_abs,
                    item_part,
                    item_id,
                    i,
                    confidence,
                    line_in_toc_range and not content_after,
                    candidate_toc_like,
                )
            )

    return boundaries

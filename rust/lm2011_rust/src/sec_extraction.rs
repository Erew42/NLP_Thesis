#![allow(unused_imports)]

use pyo3::exceptions::{PyOSError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyDict, PyFloat, PyInt, PyList, PySet, PyString, PyTuple};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};

use crate::audit_summaries::*;
use crate::common::*;
use crate::doc_ownership::*;
use crate::finbert_allocation::*;
use crate::finbert_confusion::*;
use crate::finbert_sampling::*;
use crate::finbert_tail::*;
use crate::html_audit::*;
use crate::io_parquet::*;
use crate::item_cleaning::*;
use crate::lm2011_extension_tables::*;
use crate::lm2011_features::*;
use crate::lm2011_inputs::*;
use crate::lm2011_regressions::*;
use crate::lm2011_validation::*;
use crate::lseg_api_rows::*;
use crate::lseg_ops::*;
use crate::multisurface_audit::*;
use crate::refinitiv_analyst::*;
use crate::refinitiv_authority::*;
use crate::refinitiv_bridge::*;
use crate::refinitiv_excel::*;
use crate::sentence_cleaning::*;
use crate::sentence_quality_api::*;

#[pyfunction]
#[pyo3(signature = (value=None))]
pub(crate) fn sec_digits_only_value(value: Option<&str>) -> PyResult<Option<String>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_empty() {
        return Ok(None);
    }
    if !value.is_ascii() {
        return Err(PyValueError::new_err("non-ascii input"));
    }
    let digits: String = value.chars().filter(|ch| ch.is_ascii_digit()).collect();
    if digits.is_empty() {
        Ok(None)
    } else {
        Ok(Some(digits))
    }
}

#[pyfunction]
#[pyo3(signature = (value=None))]
pub(crate) fn sec_cik_10_value(value: Option<&Bound<'_, PyAny>>) -> PyResult<Option<String>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    let parsed = py_int_like_to_i64(value)?;
    Ok(Some(python_like_zfill_i64(parsed, 10)))
}

#[pyfunction]
#[pyo3(signature = (cik10=None, accession=None))]
pub(crate) fn sec_make_doc_id_value(
    cik10: Option<&str>,
    accession: Option<&str>,
) -> Option<String> {
    let accession = accession?;
    if let Some(cik10) = cik10 {
        if !cik10.is_empty() {
            return Some(format!("{cik10}:{accession}"));
        }
    }
    Some(format!("UNK:{accession}"))
}

pub(crate) fn parse_sec_utility_date_text(value: &str) -> Option<(i32, u32, u32)> {
    let normalized = value.trim();
    if normalized.is_empty() {
        return None;
    }
    let bytes = normalized.as_bytes();
    if bytes.len() == 8 && bytes.iter().all(u8::is_ascii_digit) {
        let year = normalized[0..4].parse::<i32>().ok()?;
        let month = normalized[4..6].parse::<u32>().ok()?;
        let day = normalized[6..8].parse::<u32>().ok()?;
        if !(1..=12).contains(&month) {
            return None;
        }
        let max_day = days_in_month(year, month)?;
        if day < 1 || day > max_day {
            return None;
        }
        return Some((year, month, day));
    }
    parse_iso_date_parts(normalized)
}

#[pyfunction]
#[pyo3(signature = (value=None))]
pub(crate) fn sec_parse_date_value(
    py: Python<'_>,
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<PyObject>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    let datetime_mod = py.import_bound("datetime")?;
    let datetime_type = datetime_mod.getattr("datetime")?;
    let date_type = datetime_mod.getattr("date")?;
    if value.is_instance(&datetime_type)? || value.is_instance(&date_type)? {
        let (year, month, day) = extract_py_date_parts(value)?;
        return Ok(Some(py_date_object(py, year, month, day)?));
    }
    if !value.is_instance_of::<PyString>() {
        return Ok(None);
    }
    let rendered = value.str()?;
    let Some((year, month, day)) = parse_sec_utility_date_text(rendered.to_str()?) else {
        return Ok(None);
    };
    Ok(Some(py_date_object(py, year, month, day)?))
}

pub(crate) fn sec_roman_to_int_impl(value: &str) -> Option<i64> {
    let normalized = value.trim().to_ascii_uppercase();
    if normalized.is_empty() || !normalized.chars().all(is_roman_char) {
        return None;
    }
    let mut total = 0_i64;
    let mut previous = 0_i64;
    for ch in normalized.chars().rev() {
        let current = match ch {
            'I' => 1,
            'V' => 5,
            'X' => 10,
            'L' => 50,
            'C' => 100,
            'D' => 500,
            'M' => 1000,
            _ => return None,
        };
        if current < previous {
            total -= current;
        } else {
            total += current;
            previous = current;
        }
    }
    if total <= 0 {
        None
    } else {
        Some(total)
    }
}

#[pyfunction]
pub(crate) fn sec_roman_to_int_value(value: &str) -> Option<i64> {
    sec_roman_to_int_impl(value)
}

#[pyfunction]
#[pyo3(signature = (value=None))]
pub(crate) fn sec_default_part_for_item_id_value(
    value: Option<&str>,
) -> PyResult<Option<&'static str>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_empty() {
        return Ok(None);
    }
    if !value.is_ascii() {
        return Err(PyValueError::new_err("non-ascii item id"));
    }
    let bytes = value.as_bytes();
    if !bytes[0].is_ascii_digit() {
        return Ok(None);
    }
    let digit_len = bytes
        .iter()
        .take(2)
        .take_while(|byte| byte.is_ascii_digit())
        .count();
    if digit_len == 0 {
        return Ok(None);
    }
    let number = value[..digit_len]
        .parse::<i64>()
        .map_err(|_| PyValueError::new_err("invalid item number"))?;
    Ok(match number {
        1..=4 => Some("I"),
        5..=9 => Some("II"),
        10..=14 => Some("III"),
        15..=16 => Some("IV"),
        _ => None,
    })
}

#[pyfunction]
pub(crate) fn sec_prefix_is_bullet_value(value: &str) -> bool {
    !value.is_empty()
        && value.chars().all(|ch| {
            ch.is_whitespace()
                || matches!(
                    ch,
                    '-' | '*' | '\u{2022}' | '\u{00b7}' | '\u{2013}' | '\u{2014}'
                )
        })
}

pub(crate) fn extraction_ascii_boundary_before(bytes: &[u8], pos: usize) -> bool {
    pos == 0 || !is_ascii_word_byte(bytes[pos - 1])
}

pub(crate) fn extraction_ascii_boundary_after(bytes: &[u8], pos: usize) -> bool {
    pos >= bytes.len() || !is_ascii_word_byte(bytes[pos])
}

pub(crate) fn extraction_skip_ascii_ws(bytes: &[u8], pos: &mut usize) {
    while *pos < bytes.len() && bytes[*pos].is_ascii_whitespace() {
        *pos += 1;
    }
}

pub(crate) fn is_roman_byte(byte: u8) -> bool {
    matches!(
        byte.to_ascii_uppercase(),
        b'I' | b'V' | b'X' | b'L' | b'C' | b'D' | b'M'
    )
}

pub(crate) fn extraction_match_part_roman(
    bytes: &[u8],
    pos: usize,
) -> Option<(&'static str, usize)> {
    for (part, width) in [
        ("IV", 2usize),
        ("III", 3usize),
        ("II", 2usize),
        ("I", 1usize),
    ] {
        if pos + width > bytes.len() {
            continue;
        }
        if bytes[pos..pos + width]
            .iter()
            .zip(part.as_bytes())
            .all(|(actual, expected)| actual.to_ascii_uppercase() == *expected)
            && extraction_ascii_boundary_after(bytes, pos + width)
        {
            return Some((part, pos + width));
        }
    }
    None
}

pub(crate) fn extraction_negative_part_comma_lookahead(bytes: &[u8], pos: usize) -> bool {
    let mut probe = pos;
    extraction_skip_ascii_ws(bytes, &mut probe);
    probe < bytes.len() && bytes[probe] == b','
}

pub(crate) fn extraction_find_part_matches(
    line: &str,
    sparse: bool,
) -> Vec<(usize, usize, &'static str)> {
    let bytes = line.as_bytes();
    let mut matches = Vec::new();
    if !sparse {
        let mut pos = 0usize;
        extraction_skip_ascii_ws(bytes, &mut pos);
        if starts_with_ascii_at(bytes, pos, b"part") {
            let after_part = pos + 4;
            if after_part < bytes.len() && bytes[after_part].is_ascii_whitespace() {
                let mut roman_pos = after_part;
                extraction_skip_ascii_ws(bytes, &mut roman_pos);
                if let Some((part, end)) = extraction_match_part_roman(bytes, roman_pos) {
                    matches.push((0, end, part));
                }
            }
        }
        return matches;
    }

    let mut pos = 0usize;
    while pos + 4 <= bytes.len() {
        let Some(relative) = line[pos..].to_ascii_lowercase().find("part") else {
            break;
        };
        let start = pos + relative;
        let after_part = start + 4;
        if !extraction_ascii_boundary_before(bytes, start)
            || after_part >= bytes.len()
            || !bytes[after_part].is_ascii_whitespace()
        {
            pos = after_part;
            continue;
        }
        let mut roman_pos = after_part;
        extraction_skip_ascii_ws(bytes, &mut roman_pos);
        let Some((part, end)) = extraction_match_part_roman(bytes, roman_pos) else {
            pos = after_part;
            continue;
        };
        if extraction_negative_part_comma_lookahead(bytes, end) {
            pos = end;
            continue;
        }
        matches.push((start, end, part));
        pos = end;
    }
    matches
}

pub(crate) fn extraction_contains_ascii_word(line_lower: &str, word_lower: &str) -> bool {
    contains_ascii_word_phrase(line_lower, word_lower)
}

pub(crate) fn extraction_find_item_candidate_start(suffix: &str) -> Option<usize> {
    let bytes = suffix.as_bytes();
    let lower = suffix.to_ascii_lowercase();
    let mut pos = 0usize;
    while pos + 4 <= bytes.len() {
        let Some(relative) = lower[pos..].find("item") else {
            break;
        };
        let start = pos + relative;
        let mut probe = start + 4;
        if !extraction_ascii_boundary_before(bytes, start)
            || probe >= bytes.len()
            || !bytes[probe].is_ascii_whitespace()
        {
            pos = start + 4;
            continue;
        }
        extraction_skip_ascii_ws(bytes, &mut probe);
        let token_start = probe;
        if probe < bytes.len() && bytes[probe].is_ascii_digit() {
            while probe < bytes.len() && bytes[probe].is_ascii_digit() {
                probe += 1;
            }
        } else {
            while probe < bytes.len()
                && matches!(
                    bytes[probe].to_ascii_uppercase(),
                    b'I' | b'V' | b'X' | b'L' | b'C' | b'D' | b'M'
                )
            {
                probe += 1;
            }
        }
        if probe == token_start {
            pos = start + 4;
            continue;
        }
        if probe < bytes.len() && bytes[probe].is_ascii_alphabetic() {
            probe += 1;
        }
        while probe < bytes.len() && bytes[probe].is_ascii_whitespace() {
            probe += 1;
        }
        if probe < bytes.len() && matches!(bytes[probe], b'.' | b':') {
            probe += 1;
        }
        let _ = probe;
        return Some(start);
    }
    None
}

pub(crate) fn extraction_part_marker_is_heading(
    line: &str,
    match_start: usize,
    match_end: usize,
) -> bool {
    let prefix = &line[..match_start];
    if !prefix.trim().is_empty() && !sec_prefix_is_bullet_value(prefix) {
        return false;
    }

    let suffix = &line[match_end..];
    if let Some(item_start) = extraction_find_item_candidate_start(suffix) {
        let between = &suffix[..item_start];
        if between.contains(',') {
            return false;
        }
        if between.chars().any(|ch| ch.is_ascii_alphanumeric()) {
            return false;
        }
        return item_start <= 10;
    }

    let trimmed = suffix.trim();
    if trimmed.is_empty() {
        return true;
    }
    if trimmed.chars().any(|ch| matches!(ch, '.' | '!' | '?')) {
        return false;
    }
    if trimmed.len() > 80 {
        return false;
    }
    let letters: Vec<char> = trimmed
        .chars()
        .filter(|ch| ch.is_ascii_alphabetic())
        .collect();
    if letters.is_empty() {
        return true;
    }
    let upper = letters.iter().filter(|ch| ch.is_ascii_uppercase()).count();
    if (upper as f64) / (letters.len() as f64) >= 0.8 {
        return true;
    }
    let lower = trimmed.to_ascii_lowercase();
    trimmed.split_whitespace().count() <= 4
        && !extraction_contains_ascii_word(&lower, "see")
        && !extraction_contains_ascii_word(&lower, "refer")
}

#[pyfunction]
pub(crate) fn sec_part_marker_is_heading(
    line: &str,
    match_start: usize,
    match_end: usize,
) -> PyResult<bool> {
    if !line.is_ascii() {
        return Err(PyValueError::new_err("non-ascii Part-marker line"));
    }
    if match_start > match_end || match_end > line.len() {
        return Err(PyValueError::new_err("invalid Part-marker offsets"));
    }
    Ok(extraction_part_marker_is_heading(
        line,
        match_start,
        match_end,
    ))
}

pub(crate) fn extraction_prefix_looks_like_cross_ref(prefix: &str) -> bool {
    if prefix.trim().is_empty() {
        return false;
    }
    let trimmed = prefix.trim();
    let tail_start = trimmed.len().saturating_sub(80);
    let tail = trimmed[tail_start..].to_ascii_lowercase();
    [
        "see",
        "refer",
        "as discussed",
        "as described",
        "as set forth",
        "as noted",
        "pursuant to",
        "under",
        "in accordance with",
    ]
    .iter()
    .any(|phrase| extraction_contains_ascii_word(&tail, phrase))
        || extraction_contains_in_part_ref(&tail)
}

pub(crate) fn extraction_contains_in_part_ref(tail: &str) -> bool {
    let bytes = tail.as_bytes();
    if bytes.len() < 9 {
        return false;
    }
    for pos in 0..bytes.len().saturating_sub(1) {
        if !starts_with_ascii_at(bytes, pos, b"in")
            || !extraction_ascii_boundary_before(bytes, pos)
            || !extraction_ascii_boundary_after(bytes, pos + 2)
        {
            continue;
        }
        let mut cursor = pos + 2;
        if cursor >= bytes.len() || !bytes[cursor].is_ascii_whitespace() {
            continue;
        }
        extraction_skip_ascii_ws(bytes, &mut cursor);
        if !starts_with_ascii_at(bytes, cursor, b"part")
            || !extraction_ascii_boundary_before(bytes, cursor)
            || !extraction_ascii_boundary_after(bytes, cursor + 4)
        {
            continue;
        }
        cursor += 4;
        if cursor >= bytes.len() || !bytes[cursor].is_ascii_whitespace() {
            continue;
        }
        extraction_skip_ascii_ws(bytes, &mut cursor);
        for roman in [
            b"iv".as_slice(),
            b"iii".as_slice(),
            b"ii".as_slice(),
            b"i".as_slice(),
        ] {
            if starts_with_ascii_at(bytes, cursor, roman)
                && extraction_ascii_boundary_after(bytes, cursor + roman.len())
            {
                return true;
            }
        }
    }
    false
}

pub(crate) fn extraction_item_mention_count_at_least(line: &str, limit: usize) -> bool {
    if limit == 0 {
        return true;
    }
    let bytes = line.as_bytes();
    let mut count = 0usize;
    let mut pos = 0usize;
    while pos + 4 <= bytes.len() {
        if !starts_with_ascii_at(bytes, pos, b"item")
            || !extraction_ascii_boundary_before(bytes, pos)
        {
            pos += 1;
            continue;
        }

        let mut cursor = pos + 4;
        if cursor >= bytes.len() || !bytes[cursor].is_ascii_whitespace() {
            pos += 1;
            continue;
        }
        extraction_skip_ascii_ws(bytes, &mut cursor);

        let token_start = cursor;
        if cursor < bytes.len() && bytes[cursor].is_ascii_digit() {
            while cursor < bytes.len() && bytes[cursor].is_ascii_digit() {
                cursor += 1;
            }
        } else {
            while cursor < bytes.len() && is_roman_byte(bytes[cursor]) {
                cursor += 1;
            }
            if cursor == token_start {
                pos += 1;
                continue;
            }
        }

        if cursor < bytes.len() && bytes[cursor].is_ascii_alphabetic() {
            cursor += 1;
        }
        if !extraction_ascii_boundary_after(bytes, cursor) {
            pos += 1;
            continue;
        }

        count += 1;
        if count >= limit {
            return true;
        }
        pos = cursor.max(pos + 1);
    }
    false
}

pub(crate) fn extraction_trim_ascii_whitespace(value: &str) -> &str {
    let bytes = value.as_bytes();
    let mut start = 0usize;
    let mut end = bytes.len();
    while start < end && bytes[start].is_ascii_whitespace() {
        start += 1;
    }
    while end > start && bytes[end - 1].is_ascii_whitespace() {
        end -= 1;
    }
    &value[start..end]
}

pub(crate) fn extraction_lstrip_heading_suffix_chars(value: &str) -> &str {
    let bytes = value.as_bytes();
    let mut start = 0usize;
    while start < bytes.len() && matches!(bytes[start], b' ' | b'\t' | b':' | b'-') {
        start += 1;
    }
    &value[start..]
}

pub(crate) fn extraction_head_starts_cross_ref_suffix(head_lower: &str) -> bool {
    let bytes = head_lower.as_bytes();
    for phrase in [
        b"see".as_slice(),
        b"refer".as_slice(),
        b"as discussed".as_slice(),
        b"as described".as_slice(),
        b"as set forth".as_slice(),
        b"as noted".as_slice(),
        b"pursuant to".as_slice(),
        b"under".as_slice(),
        b"in accordance with".as_slice(),
    ] {
        if starts_with_ascii_at(bytes, 0, phrase)
            && extraction_ascii_boundary_after(bytes, phrase.len())
        {
            return true;
        }
    }

    if !starts_with_ascii_at(bytes, 0, b"in part") {
        return false;
    }
    let mut cursor = b"in part".len();
    if cursor >= bytes.len() || !bytes[cursor].is_ascii_whitespace() {
        return false;
    }
    extraction_skip_ascii_ws(bytes, &mut cursor);
    for roman in [
        b"iv".as_slice(),
        b"iii".as_slice(),
        b"ii".as_slice(),
        b"i".as_slice(),
    ] {
        if starts_with_ascii_at(bytes, cursor, roman)
            && extraction_ascii_boundary_after(bytes, cursor + roman.len())
        {
            return true;
        }
    }
    false
}

pub(crate) fn extraction_contains_part_ref(head_lower: &str) -> bool {
    let bytes = head_lower.as_bytes();
    let mut pos = 0usize;
    while pos + 4 <= bytes.len() {
        let Some(relative) = head_lower[pos..].find("part") else {
            break;
        };
        let start = pos + relative;
        let mut cursor = start + 4;
        if !extraction_ascii_boundary_before(bytes, start)
            || cursor >= bytes.len()
            || !bytes[cursor].is_ascii_whitespace()
        {
            pos = start + 4;
            continue;
        }
        extraction_skip_ascii_ws(bytes, &mut cursor);
        for roman in [
            b"iv".as_slice(),
            b"iii".as_slice(),
            b"ii".as_slice(),
            b"i".as_slice(),
        ] {
            if starts_with_ascii_at(bytes, cursor, roman)
                && extraction_ascii_boundary_after(bytes, cursor + roman.len())
            {
                return true;
            }
        }
        pos = cursor.max(start + 4);
    }
    false
}

pub(crate) fn extraction_has_prose_sentence_break(head: &str) -> bool {
    let bytes = head.as_bytes();
    let mut pos = 0usize;
    while pos < bytes.len() {
        if !matches!(bytes[pos], b'.' | b'!' | b'?') {
            pos += 1;
            continue;
        }
        let mut cursor = pos + 1;
        if cursor >= bytes.len() || !bytes[cursor].is_ascii_whitespace() {
            pos += 1;
            continue;
        }
        extraction_skip_ascii_ws(bytes, &mut cursor);
        if cursor < bytes.len() && bytes[cursor].is_ascii_alphabetic() {
            return true;
        }
        pos += 1;
    }
    false
}

pub(crate) fn extraction_ascii_word_stats(head: &str) -> (usize, usize) {
    let bytes = head.as_bytes();
    let mut words = 0usize;
    let mut lower_initial = 0usize;
    let mut pos = 0usize;
    while pos < bytes.len() {
        while pos < bytes.len() && !bytes[pos].is_ascii_alphabetic() {
            pos += 1;
        }
        if pos >= bytes.len() {
            break;
        }
        words += 1;
        if bytes[pos].is_ascii_lowercase() {
            lower_initial += 1;
        }
        while pos < bytes.len() && bytes[pos].is_ascii_alphabetic() {
            pos += 1;
        }
    }
    (words, lower_initial)
}

pub(crate) fn extraction_heading_suffix_looks_like_prose(suffix: &str) -> bool {
    if suffix.is_empty() {
        return false;
    }
    let trimmed = extraction_trim_ascii_whitespace(suffix);
    let head = extraction_lstrip_heading_suffix_chars(trimmed);
    if head.is_empty() {
        return false;
    }
    let head = &head[..head.len().min(160)];
    let head_lower = head.to_ascii_lowercase();
    if extraction_head_starts_cross_ref_suffix(&head_lower) {
        return true;
    }
    if extraction_contains_ascii_word(&head_lower, "item") {
        return true;
    }
    if extraction_contains_part_ref(&head_lower) {
        return true;
    }
    if extraction_has_prose_sentence_break(head) {
        return true;
    }
    let (words, lower_initial) = extraction_ascii_word_stats(head);
    if words >= 10 && lower_initial * 10 >= words * 6 {
        return true;
    }
    let comma_count = head.as_bytes().iter().filter(|byte| **byte == b',').count();
    comma_count >= 2 && words >= 8
}

#[pyfunction]
pub(crate) fn sec_line_has_compound_items(line: &str) -> PyResult<bool> {
    if !line.is_ascii() {
        return Err(PyValueError::new_err("non-ascii compound item line"));
    }
    Ok(extraction_item_mention_count_at_least(line, 2))
}

#[pyfunction]
pub(crate) fn sec_heading_suffix_looks_like_prose(suffix: &str) -> PyResult<bool> {
    if !suffix.is_ascii() {
        return Err(PyValueError::new_err("non-ascii heading suffix"));
    }
    Ok(extraction_heading_suffix_looks_like_prose(suffix))
}

#[pyfunction]
pub(crate) fn sec_prefix_looks_like_cross_ref(prefix: &str) -> PyResult<bool> {
    if !prefix.is_ascii() {
        return Err(PyValueError::new_err("non-ascii cross-reference prefix"));
    }
    Ok(extraction_prefix_looks_like_cross_ref(prefix))
}

pub(crate) fn extraction_contains_form_10q_or_quarterly(prefix_lower: &str) -> bool {
    prefix_lower.contains("quarterly report")
        || prefix_lower.contains("form 10-q")
        || prefix_lower.contains("form 10q")
}

pub(crate) fn extraction_has_item_word(line: &str) -> bool {
    extraction_contains_ascii_word(&line.to_ascii_lowercase(), "item")
}

pub(crate) fn extraction_has_dot_leader(line: &str) -> bool {
    let mut plain_dot_run = 0usize;
    let mut spaced_dot_count = 0usize;
    let mut in_spaced_run = false;
    for byte in line.bytes() {
        if byte == b'.' {
            plain_dot_run += 1;
            if plain_dot_run >= 4 {
                return true;
            }
            spaced_dot_count += 1;
            in_spaced_run = true;
            if spaced_dot_count >= 4 {
                return true;
            }
        } else {
            plain_dot_run = 0;
            if !byte.is_ascii_whitespace() {
                spaced_dot_count = 0;
                in_spaced_run = false;
            } else if !in_spaced_run {
                spaced_dot_count = 0;
            }
        }
    }
    false
}

pub(crate) fn extraction_has_trailing_page_number(line: &str) -> bool {
    let bytes = line.as_bytes();
    let mut pos = bytes.len();
    while pos > 0 && bytes[pos - 1].is_ascii_whitespace() {
        pos -= 1;
    }
    let digit_end = pos;
    while pos > 0 && bytes[pos - 1].is_ascii_digit() {
        pos -= 1;
    }
    let digit_count = digit_end.saturating_sub(pos);
    (1..=4).contains(&digit_count) && pos > 0 && bytes[pos - 1].is_ascii_whitespace()
}

pub(crate) fn extraction_toc_index_marker(line: &str) -> bool {
    let lower = line.to_ascii_lowercase();
    lower.contains("table of contents")
        || lower.contains("table of content")
        || extraction_contains_ascii_word(&lower, "index")
        || lower.contains("10-k summary")
        || lower.contains("10k summary")
        || lower.contains("form 10-k summary")
        || lower.contains("form 10k summary")
}

pub(crate) fn extraction_pageish_line(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return false;
    }
    let bytes = trimmed.as_bytes();
    if (1..=4).contains(&bytes.len()) && bytes.iter().all(|byte| byte.is_ascii_digit()) {
        return true;
    }
    if bytes.len() >= 3
        && bytes[0] == b'-'
        && bytes[bytes.len() - 1] == b'-'
        && bytes[1..bytes.len() - 1]
            .iter()
            .all(|byte| byte.is_ascii_digit())
        && bytes.len() - 2 <= 4
    {
        return true;
    }
    if (1..=6).contains(&bytes.len())
        && bytes.iter().all(|byte| {
            matches!(
                byte.to_ascii_lowercase(),
                b'i' | b'v' | b'x' | b'l' | b'c' | b'd' | b'm'
            )
        })
    {
        return true;
    }
    let lower = trimmed.to_ascii_lowercase();
    let lower_bytes = lower.as_bytes();
    if !lower.starts_with("page") || lower_bytes.len() <= 4 || !lower_bytes[4].is_ascii_whitespace()
    {
        return false;
    }
    let rest = lower[4..].trim();
    if rest.is_empty() {
        return false;
    }
    let mut parts = rest.split_whitespace();
    let Some(first) = parts.next() else {
        return false;
    };
    if !first.chars().all(|ch| ch.is_ascii_digit()) {
        return false;
    }
    match (parts.next(), parts.next(), parts.next()) {
        (None, None, None) => true,
        (Some("of"), Some(total), None) => total.chars().all(|ch| ch.is_ascii_digit()),
        _ => false,
    }
}

#[pyfunction]
pub(crate) fn sec_pageish_line(line: &str) -> PyResult<bool> {
    if !line.is_ascii() {
        return Err(PyValueError::new_err("non-ascii pageish line"));
    }
    Ok(extraction_pageish_line(line))
}

pub(crate) const SEC_EDGAR_METADATA_TAGS: [&[u8]; 6] = [
    b"sec-header",
    b"header",
    b"filestats",
    b"file-stats",
    b"xml_chars",
    b"xml-chars",
];
pub(crate) const SEC_EDGAR_HEADER_TAGS: [&[u8]; 2] = [b"sec-header", b"header"];

pub(crate) fn sec_edgar_open_tag_at<'a>(
    bytes: &[u8],
    pos: usize,
    tags: &'a [&'a [u8]],
) -> Option<(&'a [u8], usize)> {
    if bytes.get(pos) != Some(&b'<') {
        return None;
    }
    let mut cursor = pos + 1;
    extraction_skip_ascii_ws(bytes, &mut cursor);
    for tag in tags {
        if !starts_with_ascii_at(bytes, cursor, *tag) {
            continue;
        }
        let after_tag = cursor + tag.len();
        if after_tag < bytes.len() && is_ascii_word_byte(bytes[after_tag]) {
            continue;
        }
        let mut end = after_tag;
        while end < bytes.len() && bytes[end] != b'>' {
            end += 1;
        }
        if end >= bytes.len() {
            return None;
        }
        return Some((*tag, end + 1));
    }
    None
}

pub(crate) fn sec_find_edgar_metadata_close(
    bytes: &[u8],
    tag: &[u8],
    start: usize,
) -> Option<usize> {
    let mut pos = start;
    while pos < bytes.len() {
        if bytes[pos] != b'<' {
            pos += 1;
            continue;
        }
        let mut cursor = pos + 1;
        extraction_skip_ascii_ws(bytes, &mut cursor);
        if cursor >= bytes.len() || bytes[cursor] != b'/' {
            pos += 1;
            continue;
        }
        cursor += 1;
        extraction_skip_ascii_ws(bytes, &mut cursor);
        if !starts_with_ascii_at(bytes, cursor, tag) {
            pos += 1;
            continue;
        }
        cursor += tag.len();
        extraction_skip_ascii_ws(bytes, &mut cursor);
        if cursor < bytes.len() && bytes[cursor] == b'>' {
            return Some(cursor + 1);
        }
        pos += 1;
    }
    None
}

pub(crate) fn sec_strip_edgar_complete_blocks_ascii(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut copy_start = 0usize;
    let mut pos = 0usize;

    while pos < bytes.len() {
        if let Some((tag, open_end)) = sec_edgar_open_tag_at(bytes, pos, &SEC_EDGAR_METADATA_TAGS) {
            if let Some(close_end) = sec_find_edgar_metadata_close(bytes, tag, open_end) {
                out.push_str(&text[copy_start..pos]);
                pos = close_end;
                copy_start = close_end;
                continue;
            }
        }
        pos += 1;
    }

    if copy_start == 0 {
        return text.to_string();
    }
    out.push_str(&text[copy_start..]);
    out
}

pub(crate) fn sec_find_header_close_for_leading_strip(bytes: &[u8]) -> Option<usize> {
    let mut pos = 0usize;
    while pos < bytes.len() {
        if bytes[pos] != b'<' || bytes.get(pos + 1) != Some(&b'/') {
            pos += 1;
            continue;
        }
        let mut cursor = pos + 2;
        extraction_skip_ascii_ws(bytes, &mut cursor);
        for tag in SEC_EDGAR_HEADER_TAGS {
            if !starts_with_ascii_at(bytes, cursor, tag) {
                continue;
            }
            let mut end = cursor + tag.len();
            extraction_skip_ascii_ws(bytes, &mut end);
            if end < bytes.len() && bytes[end] == b'>' {
                return Some(end + 1);
            }
        }
        pos += 1;
    }
    None
}

pub(crate) fn sec_has_truncated_leading_header(bytes: &[u8]) -> bool {
    let mut pos = 0usize;
    extraction_skip_ascii_ws(bytes, &mut pos);
    sec_edgar_open_tag_at(bytes, pos, &SEC_EDGAR_HEADER_TAGS).is_some()
}

pub(crate) fn sec_strip_edgar_metadata_ascii(full_text: &str) -> String {
    if full_text.is_empty() {
        return String::new();
    }
    let mut text = sec_strip_edgar_complete_blocks_ascii(full_text);
    if let Some(end) = sec_find_header_close_for_leading_strip(text.as_bytes()) {
        text = text[end..].to_string();
    }
    if sec_has_truncated_leading_header(text.as_bytes()) {
        return String::new();
    }
    text
}

#[pyfunction]
pub(crate) fn sec_strip_edgar_metadata(full_text: &str) -> PyResult<String> {
    if !full_text.is_ascii() {
        return Err(PyValueError::new_err("non-ascii EDGAR metadata text"));
    }
    Ok(sec_strip_edgar_metadata_ascii(full_text))
}

pub(crate) fn extraction_match_part_only_prefix(prefix: &str) -> bool {
    let bytes = prefix.as_bytes();
    let mut pos = 0usize;
    extraction_skip_ascii_ws(bytes, &mut pos);
    if !starts_with_ascii_at(bytes, pos, b"part") {
        return false;
    }
    pos += 4;
    if pos >= bytes.len() || !bytes[pos].is_ascii_whitespace() {
        return false;
    }
    extraction_skip_ascii_ws(bytes, &mut pos);
    let Some((_part, end)) = extraction_match_part_roman(bytes, pos) else {
        return false;
    };
    pos = end;
    extraction_skip_ascii_ws(bytes, &mut pos);
    if pos < bytes.len() && matches!(bytes[pos], b':' | b'-') {
        pos += 1;
        extraction_skip_ascii_ws(bytes, &mut pos);
    }
    pos == bytes.len()
}

pub(crate) fn extraction_prefix_part_tail(prefix: &str) -> Option<&'static str> {
    let bytes = prefix.as_bytes();
    let mut pos = 0usize;
    let mut found: Option<&'static str> = None;
    while pos + 4 <= bytes.len() {
        let Some(relative) = prefix[pos..].to_ascii_lowercase().find("part") else {
            break;
        };
        let start = pos + relative;
        if start > 0
            && !bytes[start - 1].is_ascii_whitespace()
            && !matches!(bytes[start - 1], b':' | b';' | b',' | b'.')
        {
            pos = start + 4;
            continue;
        }
        let mut cursor = start + 4;
        if cursor >= bytes.len() || !bytes[cursor].is_ascii_whitespace() {
            pos = start + 4;
            continue;
        }
        extraction_skip_ascii_ws(bytes, &mut cursor);
        let Some((part, end)) = extraction_match_part_roman(bytes, cursor) else {
            pos = start + 4;
            continue;
        };
        let mut tail = end;
        extraction_skip_ascii_ws(bytes, &mut tail);
        if tail == bytes.len() {
            found = Some(part);
        }
        pos = start + 4;
    }
    found
}

#[pyfunction]
pub(crate) fn sec_prefix_is_part_only(prefix: &str) -> PyResult<bool> {
    if !prefix.is_ascii() {
        return Err(PyValueError::new_err("non-ascii Part prefix"));
    }
    Ok(extraction_match_part_only_prefix(prefix))
}

#[pyfunction]
pub(crate) fn sec_prefix_part_tail(prefix: &str) -> PyResult<Option<&'static str>> {
    if !prefix.is_ascii() {
        return Err(PyValueError::new_err("non-ascii Part prefix"));
    }
    Ok(extraction_prefix_part_tail(prefix))
}

pub(crate) fn extraction_looks_like_toc_heading_line(lines: &[String], idx: usize) -> bool {
    let Some(line) = lines.get(idx) else {
        return false;
    };
    if line.is_empty() {
        return false;
    }
    let line_trim = line.trim();
    if line_trim.is_empty() || line_trim.len() > 2000 {
        return false;
    }
    if extraction_has_dot_leader(line) {
        return true;
    }
    if line_trim.len() <= 240 && extraction_has_trailing_page_number(line) {
        if idx <= 400 {
            return true;
        }
        let start = idx.saturating_sub(3);
        let end = usize::min(lines.len(), idx + 4);
        return lines[start..end]
            .iter()
            .any(|candidate| extraction_toc_index_marker(candidate));
    }
    if idx > 400 {
        return false;
    }
    let mut j = idx + 1;
    let max_scan = usize::min(lines.len(), idx + 5);
    while j < max_scan && lines[j].trim().is_empty() {
        j += 1;
    }
    j < lines.len() && extraction_pageish_line(&lines[j]) && line_trim.len() <= 240
}

#[pyfunction]
pub(crate) fn sec_scan_part_markers_v2(
    lines: Vec<String>,
    line_starts: Vec<i64>,
    allowed_parts: Vec<String>,
    scan_sparse_layout: bool,
    toc_mask: Vec<usize>,
    is_10q: bool,
) -> PyResult<Option<Vec<(i64, String, i64, bool)>>> {
    if lines.len() != line_starts.len() {
        return Err(PyValueError::new_err(
            "lines and line_starts must have the same length",
        ));
    }
    if lines.iter().any(|line| !line.is_ascii()) {
        return Ok(None);
    }

    let allowed: HashSet<String> = allowed_parts
        .into_iter()
        .map(|part| part.trim().to_ascii_uppercase())
        .filter(|part| !part.is_empty())
        .collect();
    let toc: HashSet<usize> = toc_mask.into_iter().collect();
    let mut markers: Vec<(i64, String, i64, bool)> = Vec::new();

    for (idx, line) in lines.iter().enumerate() {
        if line.is_empty() || toc.contains(&idx) {
            continue;
        }
        for (match_start, match_end, part) in extraction_find_part_matches(line, scan_sparse_layout)
        {
            if !allowed.contains(part) {
                continue;
            }
            let prefix = &line[..match_start];
            let mut allow_form_header = false;
            if !prefix.trim().is_empty() && !sec_prefix_is_bullet_value(prefix) {
                if is_10q && !extraction_prefix_looks_like_cross_ref(prefix) {
                    let prefix_lower = prefix.to_ascii_lowercase();
                    if extraction_contains_form_10q_or_quarterly(&prefix_lower) {
                        let suffix = line[match_end..].to_ascii_lowercase();
                        if (part == "I" && suffix.contains("financial"))
                            || (part == "II" && suffix.contains("other"))
                        {
                            allow_form_header = true;
                        }
                    }
                }
                if !allow_form_header {
                    continue;
                }
            }
            let mut high_confidence =
                extraction_part_marker_is_heading(line, match_start, match_end);
            if allow_form_header {
                high_confidence = true;
            }
            if !high_confidence
                && scan_sparse_layout
                && match_start == 0
                && extraction_has_item_word(line)
                && !extraction_looks_like_toc_heading_line(&lines, idx)
            {
                high_confidence = true;
            }
            if !high_confidence {
                continue;
            }
            markers.push((
                line_starts[idx] + match_start as i64,
                part.to_string(),
                idx as i64,
                high_confidence,
            ));
        }
    }

    if !is_10q || markers.is_empty() {
        return Ok(Some(markers));
    }

    let mut filtered: Vec<(i64, String, i64, bool)> = Vec::new();
    let mut seen_part_i = false;
    let mut seen_part_ii = false;
    for marker in markers {
        if marker.1 == "I" {
            if !seen_part_i && !seen_part_ii {
                filtered.push(marker);
                seen_part_i = true;
            }
            continue;
        }
        if marker.1 == "II" {
            if seen_part_i && !seen_part_ii {
                filtered.push(marker);
                seen_part_ii = true;
            }
            continue;
        }
    }
    Ok(Some(filtered))
}

pub(crate) fn sec_splitlines_keepends(text: &str) -> Vec<&str> {
    let bytes = text.as_bytes();
    let mut lines = Vec::new();
    let mut start = 0usize;
    let mut pos = 0usize;
    while pos < bytes.len() {
        if bytes[pos] == b'\n' {
            pos += 1;
            lines.push(&text[start..pos]);
            start = pos;
            continue;
        }
        if bytes[pos] == b'\r' {
            pos += 1;
            if pos < bytes.len() && bytes[pos] == b'\n' {
                pos += 1;
            }
            lines.push(&text[start..pos]);
            start = pos;
            continue;
        }
        pos += 1;
    }
    if start < text.len() {
        lines.push(&text[start..]);
    }
    lines
}

pub(crate) fn sec_strip_line_eol(line: &str) -> &str {
    line.trim_end_matches(['\r', '\n'])
}

pub(crate) fn sec_leading_space_tab_len(line: &str) -> usize {
    line.as_bytes()
        .iter()
        .take_while(|byte| matches!(byte, b' ' | b'\t'))
        .count()
}

pub(crate) fn sec_empty_section_line(line: &str) -> bool {
    let mut probe = line.trim();
    if probe.is_empty() {
        return false;
    }
    if probe.starts_with('(') || probe.starts_with('[') {
        probe = probe[1..].trim_start();
    }
    if probe.ends_with('.') {
        probe = probe[..probe.len() - 1].trim_end();
    }
    if probe.ends_with(')') || probe.ends_with(']') {
        probe = probe[..probe.len() - 1].trim_end();
    }
    let lower = probe.to_ascii_lowercase();
    matches!(
        lower.as_str(),
        "reserved" | "omitted" | "not applicable" | "nothing to report" | "none"
    )
}

pub(crate) fn sec_trunc_normalize_item_id(
    num_raw: &str,
    let_raw: Option<u8>,
    max_item: i64,
) -> Option<String> {
    let cleaned = num_raw.trim();
    if cleaned.is_empty() {
        return None;
    }
    let number = if cleaned.bytes().all(|byte| byte.is_ascii_digit()) {
        cleaned.parse::<i64>().ok()?
    } else {
        sec_roman_to_int_impl(cleaned)?
    };
    if number <= 0 || number > max_item {
        return None;
    }
    let letter = let_raw.map(|byte| byte.to_ascii_uppercase() as char);
    match letter {
        Some(ch) if ch.is_ascii_uppercase() => Some(format!("{number}{ch}")),
        _ => Some(number.to_string()),
    }
}

pub(crate) fn sec_trunc_consume_item_token(
    line: &str,
    start: usize,
    max_item: i64,
    consume_punctuation: bool,
) -> Option<(String, usize)> {
    let bytes = line.as_bytes();
    let mut pos = start;
    if !starts_with_ascii_at(bytes, pos, b"item") {
        return None;
    }
    pos += 4;
    if pos >= bytes.len() || !bytes[pos].is_ascii_whitespace() {
        return None;
    }
    extraction_skip_ascii_ws(bytes, &mut pos);
    let token_start = pos;
    if pos < bytes.len() && bytes[pos].is_ascii_digit() {
        while pos < bytes.len() && bytes[pos].is_ascii_digit() {
            pos += 1;
        }
    } else {
        while pos < bytes.len()
            && matches!(
                bytes[pos].to_ascii_uppercase(),
                b'I' | b'V' | b'X' | b'L' | b'C' | b'D' | b'M'
            )
        {
            pos += 1;
        }
    }
    if pos == token_start {
        return None;
    }
    let token = &line[token_start..pos];
    let mut letter: Option<u8> = None;
    if pos < bytes.len() && bytes[pos].is_ascii_alphabetic() {
        let candidate = bytes[pos];
        let after_letter = pos + 1;
        let after_letter_ok = after_letter >= bytes.len()
            || !is_ascii_word_byte(bytes[after_letter])
            || bytes[after_letter].is_ascii_uppercase();
        if after_letter_ok {
            letter = Some(candidate);
            pos = after_letter;
        } else if candidate.is_ascii_uppercase() {
            // ITEM_LINESTART_PATTERN may match "Item 7Management" as Item 7
            // because its case-sensitive lookahead sees the uppercase title.
        } else {
            return None;
        }
    } else if pos < bytes.len() && is_ascii_word_byte(bytes[pos]) {
        return None;
    }
    if pos < bytes.len() && is_ascii_word_byte(bytes[pos]) && !bytes[pos].is_ascii_uppercase() {
        return None;
    }
    let item_id = sec_trunc_normalize_item_id(token, letter, max_item)?;
    if consume_punctuation {
        while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        if pos < bytes.len() && matches!(bytes[pos], b'.' | b':') {
            pos += 1;
        }
    }
    Some((item_id, pos))
}

pub(crate) fn sec_trunc_line_start_item_match(
    line: &str,
    max_item: i64,
) -> Option<(String, usize)> {
    let bytes = line.as_bytes();
    let mut pos = 0usize;
    extraction_skip_ascii_ws(bytes, &mut pos);
    let item_start = if starts_with_ascii_at(bytes, pos, b"part") {
        let mut part_pos = pos + 4;
        if part_pos >= bytes.len() || !bytes[part_pos].is_ascii_whitespace() {
            pos
        } else {
            extraction_skip_ascii_ws(bytes, &mut part_pos);
            let roman_start = part_pos;
            while part_pos < bytes.len()
                && matches!(
                    bytes[part_pos].to_ascii_uppercase(),
                    b'I' | b'V' | b'X' | b'L' | b'C' | b'D' | b'M'
                )
            {
                part_pos += 1;
            }
            if part_pos == roman_start {
                pos
            } else {
                while part_pos < bytes.len() && bytes[part_pos].is_ascii_whitespace() {
                    part_pos += 1;
                }
                if part_pos < bytes.len() && matches!(bytes[part_pos], b':' | b'-') {
                    part_pos += 1;
                }
                while part_pos < bytes.len() && bytes[part_pos].is_ascii_whitespace() {
                    part_pos += 1;
                }
                part_pos
            }
        }
    } else {
        pos
    };
    if let Some(result) = sec_trunc_consume_item_token(line, item_start, max_item, false) {
        return Some(result);
    }

    pos = 0;
    extraction_skip_ascii_ws(bytes, &mut pos);
    if !starts_with_ascii_at(bytes, pos, b"part") {
        return None;
    }
    pos += 4;
    if pos >= bytes.len() || !bytes[pos].is_ascii_whitespace() {
        return None;
    }
    extraction_skip_ascii_ws(bytes, &mut pos);
    let Some((_part, part_end)) = extraction_match_part_roman(bytes, pos) else {
        return None;
    };
    let mut rest_pos = part_end;
    while rest_pos < bytes.len() && matches!(bytes[rest_pos], b' ' | b'\t' | b':' | b'/' | b'-') {
        rest_pos += 1;
    }
    sec_trunc_consume_item_token(line, rest_pos, max_item, true)
}

pub(crate) fn sec_cross_ref_suffix_start(suffix: &str) -> bool {
    let lower = suffix.to_ascii_lowercase();
    let phrases = [
        "see",
        "refer",
        "as discussed",
        "as described",
        "as set forth",
        "as noted",
        "pursuant to",
        "under",
        "in accordance with",
    ];
    for phrase in phrases {
        if lower.starts_with(phrase)
            && lower
                .as_bytes()
                .get(phrase.len())
                .is_none_or(|byte| !is_ascii_word_byte(*byte))
        {
            return true;
        }
    }
    for phrase in ["in part iv", "in part iii", "in part ii", "in part i"] {
        if lower.starts_with(phrase)
            && lower
                .as_bytes()
                .get(phrase.len())
                .is_none_or(|byte| !is_ascii_word_byte(*byte))
        {
            return true;
        }
    }
    false
}

pub(crate) fn sec_toc_window_header_line(line: &str) -> bool {
    let bytes = line.as_bytes();
    let mut pos = 0usize;
    extraction_skip_ascii_ws(bytes, &mut pos);
    if starts_with_ascii_at(bytes, pos, b"table") {
        let mut cursor = pos + 5;
        if cursor >= bytes.len() || !bytes[cursor].is_ascii_whitespace() {
            return false;
        }
        extraction_skip_ascii_ws(bytes, &mut cursor);
        if !starts_with_ascii_at(bytes, cursor, b"of") {
            return false;
        }
        cursor += 2;
        if cursor >= bytes.len() || !bytes[cursor].is_ascii_whitespace() {
            return false;
        }
        extraction_skip_ascii_ws(bytes, &mut cursor);
        if !starts_with_ascii_at(bytes, cursor, b"content") {
            return false;
        }
        cursor += 7;
        if cursor < bytes.len() && matches!(bytes[cursor], b's' | b'S') {
            cursor += 1;
        }
        return cursor >= bytes.len() || !is_ascii_word_byte(bytes[cursor]);
    }
    if starts_with_ascii_at(bytes, pos, b"index") {
        let cursor = pos + 5;
        return cursor >= bytes.len() || !is_ascii_word_byte(bytes[cursor]);
    }
    let mut cursor = pos;
    if starts_with_ascii_at(bytes, cursor, b"form") {
        cursor += 4;
        if cursor >= bytes.len() || !bytes[cursor].is_ascii_whitespace() {
            return false;
        }
        extraction_skip_ascii_ws(bytes, &mut cursor);
    }
    if starts_with_ascii_at(bytes, cursor, b"10-k") {
        cursor += 4;
    } else if starts_with_ascii_at(bytes, cursor, b"10k") {
        cursor += 3;
    } else {
        return false;
    }
    if cursor >= bytes.len() || !bytes[cursor].is_ascii_whitespace() {
        return false;
    }
    extraction_skip_ascii_ws(bytes, &mut cursor);
    if !starts_with_ascii_at(bytes, cursor, b"summary") {
        return false;
    }
    cursor += 7;
    cursor >= bytes.len() || !is_ascii_word_byte(bytes[cursor])
}

pub(crate) fn sec_toc_window_flags(lines: &[&str]) -> Vec<bool> {
    let mut flags = vec![false; lines.len()];
    let mut remaining = 0usize;
    for (idx, line) in lines.iter().enumerate() {
        if sec_toc_window_header_line(line) {
            remaining = 150;
        }
        if remaining > 0 {
            flags[idx] = true;
            remaining -= 1;
        }
    }
    flags
}

pub(crate) fn sec_strict_part_line(line: &str) -> Option<&'static str> {
    let bytes = line.as_bytes();
    let mut pos = 0usize;
    extraction_skip_ascii_ws(bytes, &mut pos);
    if !starts_with_ascii_at(bytes, pos, b"part") {
        return None;
    }
    pos += 4;
    if pos >= bytes.len() || !bytes[pos].is_ascii_whitespace() {
        return None;
    }
    extraction_skip_ascii_ws(bytes, &mut pos);
    let (part, mut end) = extraction_match_part_roman(bytes, pos)?;
    while end < bytes.len() && bytes[end].is_ascii_whitespace() {
        end += 1;
    }
    if end < bytes.len() && matches!(bytes[end], b':' | b'.' | b'-') {
        end += 1;
    }
    while end < bytes.len() && bytes[end].is_ascii_whitespace() {
        end += 1;
    }
    if end == bytes.len() {
        Some(part)
    } else {
        None
    }
}

pub(crate) fn sec_embedded_line_starts_item(line: &str) -> bool {
    sec_embedded_item_match(line).is_some()
        || sec_embedded_roman_item_match(line).is_some()
        || sec_embedded_part_item_match(line).is_some()
}

pub(crate) fn sec_confirm_part_restart(lines: &[&str], start_idx: usize) -> bool {
    let mut non_empty = 0usize;
    let mut char_count = 0usize;
    for line in lines.iter().skip(start_idx + 1) {
        char_count += line.len();
        if char_count > 1500 {
            break;
        }
        if line.trim().is_empty() {
            continue;
        }
        non_empty += 1;
        if non_empty > 12 {
            break;
        }
        if sec_embedded_line_starts_item(line) {
            return true;
        }
    }
    false
}

pub(crate) fn sec_confirm_strict_part_heading(lines: &[&str], start_idx: usize) -> bool {
    let mut non_empty = 0usize;
    let mut char_count = 0usize;
    for idx in (start_idx + 1)..lines.len() {
        let line = lines[idx];
        char_count += line.len();
        if char_count > 1500 {
            break;
        }
        if line.trim().is_empty() {
            continue;
        }
        non_empty += 1;
        if non_empty > 12 {
            break;
        }
        if sec_embedded_line_starts_item(line) {
            return true;
        }
        if sec_strict_part_line(line).is_some() && sec_confirm_part_restart(lines, idx) {
            return true;
        }
    }
    false
}

pub(crate) fn sec_part_order(part: Option<&str>) -> i64 {
    match part.unwrap_or("").to_ascii_uppercase().as_str() {
        "I" => 1,
        "II" => 2,
        "III" => 3,
        "IV" => 4,
        _ => 0,
    }
}

pub(crate) fn sec_gij_normalized_line(line: &str) -> String {
    normalize_spaces_impl(line).to_ascii_lowercase()
}

pub(crate) fn sec_contains_gij_general(normalized_line: &str) -> bool {
    contains_ascii_word_phrase(normalized_line, "general instruction j to form 10-k")
        || contains_ascii_word_phrase(normalized_line, "general instruction j to form 10k")
}

pub(crate) fn sec_contains_gij_omit_start(normalized_line: &str) -> bool {
    contains_ascii_word_phrase(
        normalized_line,
        "the following items have been omitted in accordance with general instruction j to form 10-k",
    ) || contains_ascii_word_phrase(
        normalized_line,
        "the following items have been omitted in accordance with general instruction j to form 10k",
    )
}

pub(crate) fn sec_contains_gij_substitute(normalized_line: &str) -> bool {
    contains_ascii_word_phrase(
        normalized_line,
        "substitute information provided in accordance with general instruction j to form 10-k",
    ) || contains_ascii_word_phrase(
        normalized_line,
        "substitute information provided in accordance with general instruction j to form 10k",
    )
}

pub(crate) fn sec_strict_part_pattern_match(line: &str) -> bool {
    let trimmed = line.trim();
    let Some(prefix) = trimmed.get(..4) else {
        return false;
    };
    if !prefix.eq_ignore_ascii_case("part") {
        return false;
    }
    let after_part = &trimmed[4..];
    if !after_part
        .chars()
        .next()
        .is_some_and(|ch| ch.is_whitespace())
    {
        return false;
    }
    let rest = after_part.trim_start();
    for roman in ["III", "II", "IV", "I"] {
        let Some(candidate) = rest.get(..roman.len()) else {
            continue;
        };
        if !candidate.eq_ignore_ascii_case(roman) {
            continue;
        }
        let mut remainder = rest[roman.len()..].trim_start();
        if let Some(ch) = remainder.chars().next() {
            if matches!(ch, ':' | '.' | '-' | '\u{2013}' | '\u{2014}') {
                remainder = remainder[ch.len_utf8()..].trim_start();
            }
        }
        if remainder.is_empty() {
            return true;
        }
    }
    false
}

pub(crate) fn sec_standard_item_tokens(line: &str) -> Vec<String> {
    let bytes = line.as_bytes();
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos + 4 <= bytes.len() {
        if !starts_with_ascii_at(bytes, pos, b"item")
            || !extraction_ascii_boundary_before(bytes, pos)
        {
            pos += 1;
            continue;
        }

        let mut cursor = pos + 4;
        if cursor >= bytes.len() || !bytes[cursor].is_ascii_whitespace() {
            pos += 1;
            continue;
        }
        extraction_skip_ascii_ws(bytes, &mut cursor);

        let digit_start = cursor;
        while cursor < bytes.len()
            && bytes[cursor].is_ascii_digit()
            && cursor.saturating_sub(digit_start) < 2
        {
            cursor += 1;
        }
        if cursor == digit_start {
            pos += 1;
            continue;
        }
        if cursor < bytes.len() && bytes[cursor].is_ascii_alphabetic() {
            cursor += 1;
        }
        if !extraction_ascii_boundary_after(bytes, cursor) {
            pos += 1;
            continue;
        }

        let token_bytes = &bytes[digit_start..cursor];
        let digit_width = token_bytes
            .iter()
            .position(|byte| !byte.is_ascii_digit())
            .unwrap_or(token_bytes.len());
        let item_num = std::str::from_utf8(&token_bytes[..digit_width])
            .ok()
            .and_then(|value| value.parse::<i64>().ok());
        if item_num.is_some_and(|value| (1..=16).contains(&value)) {
            if let Ok(token) = std::str::from_utf8(token_bytes) {
                out.push(token.to_ascii_uppercase());
            }
        }
        pos = cursor.max(pos + 1);
    }
    out
}

pub(crate) fn sec_line_idx_in_ranges(idx: usize, ranges: &[(usize, usize)]) -> bool {
    ranges
        .iter()
        .any(|(start, end)| *start <= idx && idx < *end)
}

pub(crate) fn sec_gij_scan_block_end(
    lines: &[String],
    start_idx: usize,
    stop_on_substitute: bool,
) -> usize {
    let end_idx = lines.len().min(start_idx.saturating_add(200));
    for idx in start_idx..end_idx {
        let normalized = sec_gij_normalized_line(&lines[idx]);
        if stop_on_substitute && sec_contains_gij_substitute(&normalized) {
            return idx;
        }
        if sec_strict_part_pattern_match(&lines[idx]) {
            return idx;
        }
    }
    end_idx
}

pub(crate) type SecGijContextPayload = (
    bool,
    Vec<(usize, usize)>,
    Vec<(usize, usize)>,
    Vec<String>,
    String,
);

#[pyfunction]
pub(crate) fn sec_detect_gij_context(lines: Vec<String>) -> PyResult<SecGijContextPayload> {
    let mut gij_asset_backed = false;
    let mut gij_omit_ranges: Vec<(usize, usize)> = Vec::new();
    let mut gij_substitute_ranges: Vec<(usize, usize)> = Vec::new();
    let mut gij_omitted_items: BTreeSet<String> = BTreeSet::new();

    for (idx, line) in lines.iter().enumerate() {
        let normalized = sec_gij_normalized_line(line);
        if sec_contains_gij_general(&normalized) {
            gij_asset_backed = true;
        }

        if sec_contains_gij_substitute(&normalized) {
            gij_asset_backed = true;
            if !sec_line_idx_in_ranges(idx, &gij_substitute_ranges) {
                let end_idx = sec_gij_scan_block_end(&lines, idx + 1, false);
                gij_substitute_ranges.push((idx, end_idx));
            }
        }

        if sec_contains_gij_omit_start(&normalized) {
            gij_asset_backed = true;
            if sec_line_idx_in_ranges(idx, &gij_omit_ranges) {
                continue;
            }
            let end_idx = sec_gij_scan_block_end(&lines, idx + 1, true);
            gij_omit_ranges.push((idx, end_idx));
            for omit_line in lines.iter().take(end_idx).skip(idx) {
                for token in sec_standard_item_tokens(omit_line) {
                    gij_omitted_items.insert(token);
                }
            }
        }
    }

    let reason = if gij_asset_backed {
        "General Instruction J".to_string()
    } else {
        String::new()
    };
    Ok((
        gij_asset_backed,
        gij_omit_ranges,
        gij_substitute_ranges,
        gij_omitted_items.into_iter().collect(),
        reason,
    ))
}

pub(crate) fn sec_toc_dot_leader4(line: &str) -> bool {
    let mut dotted_run = 0usize;
    let mut spaced_dots = 0usize;
    let mut in_spaced_run = false;
    for byte in line.as_bytes() {
        if *byte == b'.' {
            dotted_run += 1;
            spaced_dots += 1;
            in_spaced_run = true;
            if dotted_run >= 4 || spaced_dots >= 4 {
                return true;
            }
        } else if byte.is_ascii_whitespace() && in_spaced_run {
            dotted_run = 0;
        } else {
            dotted_run = 0;
            spaced_dots = 0;
            in_spaced_run = false;
        }
    }
    false
}

pub(crate) fn sec_toc_trailing_page_number(line: &str) -> bool {
    let trimmed = line.trim_end();
    let bytes = trimmed.as_bytes();
    if bytes.is_empty() {
        return false;
    }
    let mut pos = bytes.len();
    let mut digits = 0usize;
    while pos > 0 && bytes[pos - 1].is_ascii_digit() && digits < 4 {
        pos -= 1;
        digits += 1;
    }
    digits > 0 && pos > 0 && bytes[pos - 1].is_ascii_whitespace()
}

pub(crate) fn sec_toc_pageish(line: &str) -> bool {
    let stripped = line.trim();
    if stripped.is_empty() {
        return false;
    }
    let bytes = stripped.as_bytes();
    if bytes.len() <= 4 && bytes.iter().all(|byte| byte.is_ascii_digit()) {
        return true;
    }
    if bytes.len() >= 3
        && bytes[0] == b'-'
        && bytes[bytes.len() - 1] == b'-'
        && bytes[1..bytes.len() - 1]
            .iter()
            .all(|byte| byte.is_ascii_digit())
        && bytes.len() <= 6
    {
        return true;
    }
    if bytes.len() <= 6
        && bytes.iter().all(|byte| {
            matches!(
                byte.to_ascii_lowercase(),
                b'i' | b'v' | b'x' | b'l' | b'c' | b'd' | b'm'
            )
        })
    {
        return true;
    }
    let lower = stripped.to_ascii_lowercase();
    if let Some(rest) = lower.strip_prefix("page ") {
        let mut parts = rest.split_whitespace();
        let Some(page_num) = parts.next() else {
            return false;
        };
        if !page_num.bytes().all(|byte| byte.is_ascii_digit()) {
            return false;
        }
        match (parts.next(), parts.next(), parts.next()) {
            (None, None, None) => true,
            (Some("of"), Some(total), None) => total.bytes().all(|byte| byte.is_ascii_digit()),
            _ => false,
        }
    } else {
        false
    }
}

pub(crate) fn sec_toc_words(line: &str) -> Vec<&str> {
    sec_embedded_words(line)
}

pub(crate) fn sec_toc_separator_line(stripped: &str) -> bool {
    stripped.len() >= 3
        && stripped
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_whitespace() || matches!(*byte, b'-' | b'=' | b'*'))
}

pub(crate) fn sec_toc_prose_like(line: &str) -> bool {
    let stripped = line.trim();
    if stripped.is_empty() || sec_toc_separator_line(stripped) {
        return false;
    }
    let letters: Vec<u8> = stripped
        .bytes()
        .filter(|byte| byte.is_ascii_alphabetic())
        .collect();
    if letters.is_empty() {
        return false;
    }
    let has_lower = letters.iter().any(|byte| byte.is_ascii_lowercase());
    if !has_lower && letters.len() < 30 {
        return false;
    }
    if letters.len() >= 20 && has_lower {
        return true;
    }
    stripped.ends_with('.') || stripped.ends_with('!') || stripped.ends_with('?')
}

pub(crate) fn sec_toc_contains_index_marker(line: &str) -> bool {
    let lower = sec_gij_normalized_line(line);
    contains_ascii_word_phrase(&lower, "table of contents")
        || contains_ascii_word_phrase(&lower, "table of content")
        || contains_ascii_word_phrase(&lower, "index")
        || contains_ascii_word_phrase(&lower, "10-k summary")
        || contains_ascii_word_phrase(&lower, "10k summary")
        || contains_ascii_word_phrase(&lower, "form 10-k summary")
        || contains_ascii_word_phrase(&lower, "form 10k summary")
}

pub(crate) fn sec_toc_contains_summary_marker(line: &str) -> bool {
    let lower = sec_gij_normalized_line(line);
    contains_ascii_word_phrase(&lower, "10-k summary")
        || contains_ascii_word_phrase(&lower, "10k summary")
        || contains_ascii_word_phrase(&lower, "form 10-k summary")
        || contains_ascii_word_phrase(&lower, "form 10k summary")
}

pub(crate) fn sec_toc_header_line(line: &str) -> bool {
    matches!(
        sec_gij_normalized_line(line).as_str(),
        "table of contents" | "table of content"
    )
}

pub(crate) fn sec_toc_summary_header_line(line: &str) -> bool {
    matches!(
        sec_gij_normalized_line(line).as_str(),
        "10-k summary" | "10k summary" | "form 10-k summary" | "form 10k summary"
    )
}

pub(crate) fn sec_toc_index_header_line(line: &str) -> bool {
    let trimmed = line.trim_start();
    trimmed.len() >= 5
        && trimmed[..5].eq_ignore_ascii_case("index")
        && trimmed
            .as_bytes()
            .get(5)
            .is_none_or(|byte| !is_ascii_word_byte(*byte))
}

pub(crate) fn sec_toc_item_word_count(line: &str) -> usize {
    let lower = line.to_ascii_lowercase();
    let bytes = lower.as_bytes();
    let mut count = 0usize;
    let mut pos = 0usize;
    while let Some(relative) = lower[pos..].find("item") {
        let start = pos + relative;
        let end = start + 4;
        if extraction_ascii_boundary_before(bytes, start)
            && extraction_ascii_boundary_after(bytes, end)
        {
            count += 1;
        }
        pos = end;
    }
    count
}

pub(crate) fn sec_toc_numeric_dot_heading(line: &str) -> bool {
    let bytes = line.as_bytes();
    let mut pos = 0usize;
    extraction_skip_ascii_ws(bytes, &mut pos);
    let digit_start = pos;
    while pos < bytes.len() && bytes[pos].is_ascii_digit() && pos - digit_start < 2 {
        pos += 1;
    }
    if pos == digit_start {
        return false;
    }
    if pos < bytes.len() && bytes[pos].is_ascii_alphabetic() {
        pos += 1;
    }
    extraction_skip_ascii_ws(bytes, &mut pos);
    if pos < bytes.len() && matches!(bytes[pos], b'.' | b')' | b':' | b'-') {
        pos += 1;
    }
    pos < bytes.len() && bytes[pos].is_ascii_whitespace() && line[pos..].trim().len() > 0
}

pub(crate) fn sec_toc_part_line_start(line: &str) -> Option<usize> {
    let bytes = line.as_bytes();
    let mut pos = 0usize;
    extraction_skip_ascii_ws(bytes, &mut pos);
    if !starts_with_ascii_at(bytes, pos, b"part") {
        return None;
    }
    pos += 4;
    if pos >= bytes.len() || !bytes[pos].is_ascii_whitespace() {
        return None;
    }
    extraction_skip_ascii_ws(bytes, &mut pos);
    let (_part, end) = extraction_match_part_roman(bytes, pos)?;
    Some(end)
}

pub(crate) fn sec_toc_item_candidate_start(suffix: &str) -> Option<usize> {
    let lower = suffix.to_ascii_lowercase();
    let bytes = lower.as_bytes();
    let mut pos = 0usize;
    while let Some(relative) = lower[pos..].find("item") {
        let start = pos + relative;
        let end = start + 4;
        if extraction_ascii_boundary_before(bytes, start)
            && end < bytes.len()
            && bytes[end].is_ascii_whitespace()
        {
            let mut cursor = end;
            extraction_skip_ascii_ws(bytes, &mut cursor);
            if cursor < bytes.len()
                && (bytes[cursor].is_ascii_digit()
                    || matches!(
                        bytes[cursor].to_ascii_uppercase(),
                        b'I' | b'V' | b'X' | b'L' | b'C' | b'D' | b'M'
                    ))
            {
                return Some(start);
            }
        }
        pos = end;
    }
    None
}

pub(crate) fn sec_toc_part_marker_is_heading(line: &str, match_end: usize) -> bool {
    let suffix = &line[match_end..];
    if let Some(item_start) = sec_toc_item_candidate_start(suffix) {
        let between = &suffix[..item_start];
        if between.contains(',') {
            return false;
        }
        if between.chars().any(|ch| ch.is_ascii_alphanumeric()) {
            return false;
        }
        return item_start <= 10;
    }

    let trimmed = suffix.trim();
    if trimmed.is_empty() {
        return true;
    }
    if trimmed.contains('.') || trimmed.contains('!') || trimmed.contains('?') {
        return false;
    }
    if trimmed.len() > 80 {
        return false;
    }
    let letters: Vec<u8> = trimmed
        .bytes()
        .filter(|byte| byte.is_ascii_alphabetic())
        .collect();
    if letters.is_empty() {
        return true;
    }
    let upper = letters
        .iter()
        .filter(|byte| byte.is_ascii_uppercase())
        .count();
    if (upper as f64) / (letters.len() as f64) >= 0.8 {
        return true;
    }
    sec_toc_words(trimmed).len() <= 4
        && !contains_ascii_word_phrase(&trimmed.to_ascii_lowercase(), "see")
        && !contains_ascii_word_phrase(&trimmed.to_ascii_lowercase(), "refer")
}

pub(crate) fn sec_toc_heading_line(line: &str) -> bool {
    if sec_trunc_line_start_item_match(line, 20).is_some() {
        return true;
    }
    if let Some(match_end) = sec_toc_part_line_start(line) {
        if sec_toc_part_marker_is_heading(line, match_end) {
            return true;
        }
    }
    sec_toc_dot_leader4(line) && sec_toc_numeric_dot_heading(line)
}

pub(crate) fn sec_toc_gap_text_line(line: &str) -> bool {
    let stripped = line.trim();
    if stripped.is_empty() || sec_toc_pageish(stripped) || sec_toc_contains_index_marker(stripped) {
        return false;
    }
    let letters: Vec<u8> = stripped
        .bytes()
        .filter(|byte| byte.is_ascii_alphabetic())
        .collect();
    !letters.is_empty() && letters.iter().any(|byte| byte.is_ascii_lowercase())
}

pub(crate) fn sec_toc_find_table_of_contents(body: &str) -> Option<usize> {
    let bytes = body.as_bytes();
    let mut pos = 0usize;
    while pos + 5 <= bytes.len() {
        let Some(relative) = body[pos..].to_ascii_lowercase().find("table") else {
            return None;
        };
        let start = pos + relative;
        let mut cursor = start + 5;
        if cursor >= bytes.len() || !bytes[cursor].is_ascii_whitespace() {
            pos = start + 5;
            continue;
        }
        extraction_skip_ascii_ws(bytes, &mut cursor);
        if !starts_with_ascii_at(bytes, cursor, b"of") {
            pos = start + 5;
            continue;
        }
        cursor += 2;
        if cursor >= bytes.len() || !bytes[cursor].is_ascii_whitespace() {
            pos = start + 5;
            continue;
        }
        extraction_skip_ascii_ws(bytes, &mut cursor);
        if !starts_with_ascii_at(bytes, cursor, b"content") {
            pos = start + 5;
            continue;
        }
        return Some(start);
    }
    None
}

pub(crate) fn sec_toc_item_number_valid(num: &str) -> bool {
    let value = if num.as_bytes().iter().all(|byte| byte.is_ascii_digit()) {
        num.parse::<i64>().ok()
    } else {
        sec_roman_to_int_impl(num)
    };
    value.is_some_and(|number| number > 0 && number <= 20)
}

pub(crate) fn sec_toc_lookahead_after_page(window: &str, page_end: usize) -> bool {
    let bytes = window.as_bytes();
    let mut cursor = page_end;
    extraction_skip_ascii_ws(bytes, &mut cursor);
    if cursor == bytes.len() {
        return true;
    }
    if cursor == page_end {
        return false;
    }
    (starts_with_ascii_at(bytes, cursor, b"item")
        && extraction_ascii_boundary_after(bytes, cursor + 4))
        || (starts_with_ascii_at(bytes, cursor, b"part")
            && extraction_ascii_boundary_after(bytes, cursor + 4))
}

pub(crate) fn sec_toc_entry_match_at(window: &str, item_start: usize) -> Option<(usize, bool)> {
    let bytes = window.as_bytes();
    if !extraction_ascii_boundary_before(bytes, item_start)
        || !starts_with_ascii_at(bytes, item_start, b"item")
    {
        return None;
    }
    let mut cursor = item_start + 4;
    if cursor >= bytes.len() || !bytes[cursor].is_ascii_whitespace() {
        return None;
    }
    extraction_skip_ascii_ws(bytes, &mut cursor);

    let num_start = cursor;
    if cursor < bytes.len() && bytes[cursor].is_ascii_digit() {
        while cursor < bytes.len() && bytes[cursor].is_ascii_digit() {
            cursor += 1;
        }
    } else {
        while cursor < bytes.len()
            && matches!(
                bytes[cursor].to_ascii_uppercase(),
                b'I' | b'V' | b'X' | b'L' | b'C' | b'D' | b'M'
            )
        {
            cursor += 1;
        }
    }
    if cursor == num_start {
        return None;
    }
    let num = &window[num_start..cursor];

    if cursor < bytes.len() && bytes[cursor].is_ascii_alphabetic() {
        cursor += 1;
    }
    let after_item_id = cursor;
    let mut punct_probe = cursor;
    extraction_skip_ascii_ws(bytes, &mut punct_probe);
    if punct_probe < bytes.len() && matches!(bytes[punct_probe], b'.' | b':') {
        cursor = punct_probe + 1;
        if cursor >= bytes.len() || !bytes[cursor].is_ascii_whitespace() {
            return None;
        }
        extraction_skip_ascii_ws(bytes, &mut cursor);
    } else {
        if after_item_id >= bytes.len() || !bytes[after_item_id].is_ascii_whitespace() {
            return None;
        }
        cursor = after_item_id;
        extraction_skip_ascii_ws(bytes, &mut cursor);
    }
    let content_start = cursor;
    let scan_end = bytes.len().min(content_start.saturating_add(121));

    let item_id_valid = sec_toc_item_number_valid(num);
    let mut page_start = content_start;
    while page_start < scan_end {
        if !bytes[page_start].is_ascii_digit()
            || !extraction_ascii_boundary_before(bytes, page_start)
        {
            page_start += 1;
            continue;
        }
        let mut page_end = page_start;
        while page_end < bytes.len()
            && page_end - page_start < 3
            && bytes[page_end].is_ascii_digit()
        {
            page_end += 1;
        }
        if !extraction_ascii_boundary_after(bytes, page_end) {
            page_start += 1;
            continue;
        }
        if page_start - content_start <= 120 && sec_toc_lookahead_after_page(window, page_end) {
            let page = window[page_start..page_end].parse::<i64>().ok();
            let page_valid = page.is_some_and(|value| value > 0 && value <= 500);
            return Some((page_end, item_id_valid && page_valid));
        }
        page_start += 1;
    }
    None
}

#[pyfunction]
#[pyo3(signature = (body, max_chars=20000))]
pub(crate) fn sec_infer_toc_end_pos(body: &str, max_chars: i64) -> PyResult<Option<usize>> {
    if !body.is_ascii() {
        return Ok(None);
    }
    let Some(start) = sec_toc_find_table_of_contents(body) else {
        return Ok(None);
    };
    if max_chars <= 0 {
        return Ok(None);
    }
    let end = body.len().min(start.saturating_add(max_chars as usize));
    let window = &body[start..end];
    let mut count = 0usize;
    let mut last_end: Option<usize> = None;
    let mut pos = 0usize;
    while pos + 4 <= window.len() {
        let lower_tail = window[pos..].to_ascii_lowercase();
        let Some(relative) = lower_tail.find("item") else {
            break;
        };
        let item_start = pos + relative;
        if let Some((match_end, is_valid)) = sec_toc_entry_match_at(window, item_start) {
            if is_valid {
                count += 1;
                last_end = Some(start + match_end);
            }
            pos = match_end;
        } else {
            pos = item_start + 4;
        }
    }
    if count >= 4 {
        Ok(last_end)
    } else {
        Ok(None)
    }
}

pub(crate) fn sec_remove_pagination_toc_header(stripped: &str) -> bool {
    matches!(
        normalize_spaces_impl(stripped)
            .to_ascii_lowercase()
            .as_str(),
        "table of contents" | "table of content"
    )
}

pub(crate) fn sec_remove_pagination_page_of(stripped: &str) -> bool {
    let lower = stripped.to_ascii_lowercase();
    let bytes = lower.as_bytes();
    let mut pos = 0usize;
    if !starts_with_ascii_at(bytes, pos, b"page") {
        return false;
    }
    pos += 4;
    if pos >= bytes.len() || !bytes[pos].is_ascii_whitespace() {
        return false;
    }
    extraction_skip_ascii_ws(bytes, &mut pos);
    let number_start = pos;
    while pos < bytes.len() && bytes[pos].is_ascii_digit() {
        pos += 1;
    }
    if pos == number_start {
        return false;
    }
    while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
        pos += 1;
    }
    if pos == bytes.len() {
        return true;
    }
    if !starts_with_ascii_at(bytes, pos, b"of") {
        return false;
    }
    pos += 2;
    if pos >= bytes.len() || !bytes[pos].is_ascii_whitespace() {
        return false;
    }
    extraction_skip_ascii_ws(bytes, &mut pos);
    let total_start = pos;
    while pos < bytes.len() && bytes[pos].is_ascii_digit() {
        pos += 1;
    }
    pos > total_start && pos == bytes.len()
}

pub(crate) fn sec_remove_pagination_hyphen_page(stripped: &str) -> bool {
    let bytes = stripped.as_bytes();
    bytes.len() >= 3
        && bytes.len() <= 6
        && bytes[0] == b'-'
        && bytes[bytes.len() - 1] == b'-'
        && bytes[1..bytes.len() - 1]
            .iter()
            .all(|byte| byte.is_ascii_digit())
}

pub(crate) fn sec_remove_pagination_roman_page(stripped: &str) -> bool {
    let bytes = stripped.as_bytes();
    (1..=6).contains(&bytes.len())
        && bytes.iter().all(|byte| {
            matches!(
                byte.to_ascii_lowercase(),
                b'i' | b'v' | b'x' | b'l' | b'c' | b'd' | b'm'
            )
        })
}

pub(crate) fn sec_remove_pagination_number_page(stripped: &str) -> Option<i64> {
    let bytes = stripped.as_bytes();
    if bytes.is_empty() || bytes.len() > 4 || !bytes.iter().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    stripped.parse::<i64>().ok()
}

#[pyfunction]
pub(crate) fn sec_remove_pagination(text: &str) -> PyResult<String> {
    if !text.is_ascii() {
        return Err(PyValueError::new_err("non-ascii pagination text"));
    }
    let normalized = normalize_newlines_impl(Some(text));
    let lines: Vec<&str> = normalized.split('\n').collect();
    let mut out: Vec<&str> = Vec::with_capacity(lines.len());

    for (idx, line) in lines.iter().enumerate() {
        let stripped = line.trim();
        let prev_blank = idx > 0 && lines[idx - 1].trim().is_empty();
        let next_blank = idx + 1 < lines.len() && lines[idx + 1].trim().is_empty();

        if stripped.is_empty() {
            out.push("");
            continue;
        }
        if sec_remove_pagination_toc_header(stripped)
            || sec_remove_pagination_page_of(stripped)
            || sec_remove_pagination_hyphen_page(stripped)
            || (sec_remove_pagination_roman_page(stripped) && (prev_blank || next_blank))
        {
            continue;
        }
        if let Some(value) = sec_remove_pagination_number_page(stripped) {
            if value <= 500 && (prev_blank || next_blank) {
                continue;
            }
        }
        out.push(line);
    }

    Ok(collapse_blank_runs_impl(&out.join("\n")).trim().to_string())
}

pub(crate) fn sec_trim_part_marker_end(line: &str) -> Option<usize> {
    let bytes = line.as_bytes();
    let mut pos = 0usize;
    extraction_skip_ascii_ws(bytes, &mut pos);
    if !starts_with_ascii_at(bytes, pos, b"part") {
        return None;
    }
    pos += 4;
    if pos >= bytes.len() || !bytes[pos].is_ascii_whitespace() {
        return None;
    }
    extraction_skip_ascii_ws(bytes, &mut pos);
    for width in [2usize, 3, 2, 1] {
        if pos + width > bytes.len() {
            continue;
        }
        let part = &line[pos..pos + width];
        let valid = matches!(
            part.to_ascii_uppercase().as_str(),
            "IV" | "III" | "II" | "I"
        );
        if valid && extraction_ascii_boundary_after(bytes, pos + width) {
            return Some(pos + width);
        }
    }
    None
}

#[pyfunction]
pub(crate) fn sec_trim_trailing_part_marker(text: &str) -> PyResult<String> {
    if !text.is_ascii() {
        return Err(PyValueError::new_err("non-ascii part marker text"));
    }
    if text.is_empty() {
        return Ok(text.to_string());
    }
    let mut lines: Vec<&str> = text.split('\n').collect();
    while lines.last().is_some_and(|line| line.trim().is_empty()) {
        lines.pop();
    }
    if lines.is_empty() {
        return Ok(text.to_string());
    }
    let last = lines[lines.len() - 1].trim();
    if let Some(part_end) = sec_trim_part_marker_end(last) {
        if last[part_end..]
            .trim_matches(|ch| matches!(ch, ' ' | '\t' | ':' | '-'))
            .is_empty()
        {
            lines.pop();
        }
    }
    Ok(lines.join("\n").trim_end().to_string())
}

pub(crate) fn sec_reserved_target_matches(value: &str) -> bool {
    matches!(
        value.to_ascii_lowercase().as_str(),
        "reserved" | "[reserved]"
    )
}

pub(crate) fn sec_reserved_probe_matches(probe: &str) -> bool {
    let value = probe.trim();
    if value.is_empty() || !value.to_ascii_lowercase().contains("reserved") {
        return false;
    }
    if sec_reserved_target_matches(value) {
        return true;
    }

    let bytes = value.as_bytes();
    let mut pos = 0usize;
    if pos < bytes.len() && bytes[pos] == b'(' {
        pos += 1;
    }
    if pos >= bytes.len() || !bytes[pos].is_ascii_alphabetic() {
        return false;
    }
    pos += 1;
    if pos < bytes.len() && bytes[pos] == b')' {
        pos += 1;
    }
    extraction_skip_ascii_ws(bytes, &mut pos);
    pos < value.len() && sec_reserved_target_matches(&value[pos..])
}

pub(crate) fn sec_reserved_line_end(bytes: &[u8], start: usize) -> usize {
    let mut pos = start;
    while pos < bytes.len() {
        match bytes[pos] {
            b'\r' => {
                if pos + 1 < bytes.len() && bytes[pos + 1] == b'\n' {
                    return pos + 2;
                }
                return pos + 1;
            }
            b'\n' | 0x0b | 0x0c | 0x1c | 0x1d | 0x1e => return pos + 1,
            _ => pos += 1,
        }
    }
    bytes.len()
}

#[pyfunction]
pub(crate) fn sec_reserved_stub_end(text: &str) -> PyResult<Option<usize>> {
    if !text.is_ascii() {
        return Err(PyValueError::new_err("non-ascii reserved stub text"));
    }
    if text.is_empty() {
        return Ok(None);
    }
    let bytes = text.as_bytes();
    let mut start = 0usize;
    while start < bytes.len() {
        let end = sec_reserved_line_end(bytes, start);
        let line = &text[start..end];
        let stripped = line.trim();
        if stripped.is_empty() {
            start = end;
            continue;
        }
        let probe = stripped.trim_start_matches(|ch| matches!(ch, ' ' | '\t' | ':' | '-' | '.'));
        if probe.is_empty() {
            start = end;
            continue;
        }
        if sec_reserved_probe_matches(probe) {
            return Ok(Some(end));
        }
        return Ok(None);
    }
    Ok(None)
}

#[pyfunction]
#[pyo3(signature = (line, max_item=20))]
pub(crate) fn sec_line_start_item_match(
    line: &str,
    max_item: i64,
) -> PyResult<Option<(String, usize)>> {
    if !line.is_ascii() {
        return Err(PyValueError::new_err("non-ascii item line"));
    }
    Ok(sec_trunc_line_start_item_match(line, max_item))
}

#[pyfunction]
#[pyo3(signature = (lines, max_lines=None))]
pub(crate) fn sec_detect_toc_line_ranges(
    lines: Vec<String>,
    max_lines: Option<i64>,
) -> PyResult<Option<Vec<(usize, usize)>>> {
    let n = match max_lines {
        Some(value) if value <= 0 => 0usize,
        Some(value) => lines.len().min(value as usize),
        None => lines.len(),
    };
    if n == 0 {
        return Ok(Some(Vec::new()));
    }

    let mut ranges: Vec<(usize, usize)> = Vec::new();
    let mut inline_lines: Vec<usize> = Vec::new();
    let mut toc_markers: BTreeSet<usize> = BTreeSet::new();
    let mut heading_lines: Vec<usize> = Vec::new();

    for (idx, line) in lines.iter().take(n).enumerate() {
        let stripped = line.trim();
        if stripped.is_empty() {
            continue;
        }
        let item_words = sec_toc_item_word_count(line);
        let has_marker = sec_toc_contains_index_marker(line);
        let has_summary_marker = sec_toc_contains_summary_marker(line);
        if has_marker {
            toc_markers.insert(idx);
        }
        if item_words >= 3 && line.len() <= 5_000 {
            inline_lines.push(idx);
            continue;
        }
        if has_marker && item_words >= 1 && line.len() <= 8_000 {
            if !has_summary_marker
                || sec_toc_dot_leader4(line)
                || sec_toc_trailing_page_number(line)
            {
                inline_lines.push(idx);
            }
        }
        if sec_toc_heading_line(line) {
            heading_lines.push(idx);
        }
    }

    for idx in inline_lines {
        ranges.push((idx, idx));
    }

    for idx in 0..n.saturating_sub(1) {
        let a = lines[idx].trim().to_ascii_lowercase();
        let b = lines[idx + 1].trim().to_ascii_lowercase();
        if (a == "table" && b.starts_with("of contents"))
            || (a == "table of" && b.starts_with("contents"))
            || (a == "form 10-k" && b.starts_with("summary"))
        {
            toc_markers.insert(idx);
            toc_markers.insert(idx + 1);
        }
    }

    let mut clusters: Vec<Vec<usize>> = Vec::new();
    if let Some(first) = heading_lines.first().copied() {
        let mut cur = vec![first];
        for idx in heading_lines.iter().skip(1).copied() {
            if idx - cur[cur.len() - 1] <= 6 {
                let mut has_prose_gap = false;
                for gap_idx in (cur[cur.len() - 1] + 1)..idx {
                    if sec_toc_prose_like(&lines[gap_idx]) || sec_toc_gap_text_line(&lines[gap_idx])
                    {
                        has_prose_gap = true;
                        break;
                    }
                }
                if has_prose_gap {
                    clusters.push(cur);
                    cur = vec![idx];
                } else {
                    cur.push(idx);
                }
            } else {
                clusters.push(cur);
                cur = vec![idx];
            }
        }
        clusters.push(cur);
    }

    for cluster in clusters {
        let first = cluster[0];
        let last = cluster[cluster.len() - 1];
        let has_marker_near = toc_markers
            .iter()
            .any(|marker| first.saturating_sub(10) <= *marker && *marker <= last + 10);
        let min_headings = if has_marker_near { 3 } else { 4 };
        if cluster.len() < min_headings {
            continue;
        }

        let nonempty_indices: Vec<usize> = (first..=last)
            .filter(|idx| !lines[*idx].trim().is_empty())
            .collect();
        if nonempty_indices.is_empty() {
            continue;
        }

        let mut headingish = 0usize;
        let mut prose_like = 0usize;
        let mut dot_hits = 0usize;
        let mut page_hits = 0usize;
        let mut last_toc_like_idx: Option<usize> = None;

        for idx in first..=last {
            let line = lines[idx].trim();
            if line.is_empty() {
                continue;
            }
            if sec_toc_dot_leader4(line)
                || sec_toc_trailing_page_number(line)
                || sec_toc_pageish(line)
            {
                last_toc_like_idx = Some(idx);
            }
        }

        for idx in nonempty_indices.iter().copied() {
            let line = lines[idx].trim();
            if sec_toc_dot_leader4(line) {
                dot_hits += 1;
            }
            if sec_toc_trailing_page_number(line) || sec_toc_pageish(line) {
                page_hits += 1;
            }
            if sec_trunc_line_start_item_match(line, 20).is_some() {
                headingish += 1;
                continue;
            }
            if let Some(match_end) = sec_toc_part_line_start(line) {
                if sec_toc_part_marker_is_heading(line, match_end) {
                    headingish += 1;
                    continue;
                }
            }
            if sec_toc_header_line(line) || sec_toc_summary_header_line(line) {
                headingish += 1;
                continue;
            }
            if sec_toc_index_header_line(line) {
                headingish += 1;
                continue;
            }
            if sec_toc_pageish(line) {
                headingish += 1;
                continue;
            }
            if sec_toc_prose_like(line) {
                prose_like += 1;
            }
        }

        let block_len = nonempty_indices.len() as f64;
        let headingish_ratio = headingish as f64 / block_len;
        let prose_ratio = prose_like as f64 / block_len;
        if headingish_ratio < 0.75 || prose_ratio > 0.2 {
            continue;
        }

        let late_cluster = first > 300;
        let strong_page = dot_hits >= 2 || page_hits >= 2;
        if late_cluster {
            if !(has_marker_near || dot_hits >= 3 || page_hits >= 3) {
                continue;
            }
        } else if !(has_marker_near || strong_page) {
            continue;
        }

        let mut end_idx = last;
        if (dot_hits >= 2 || page_hits >= 2) && last_toc_like_idx.is_some() {
            end_idx = end_idx.min(last_toc_like_idx.unwrap());
        }
        ranges.push((first, end_idx));
    }

    if ranges.is_empty() {
        return Ok(Some(Vec::new()));
    }
    ranges.sort_unstable();
    let mut merged: Vec<(usize, usize)> = vec![ranges[0]];
    for (start, end) in ranges.into_iter().skip(1) {
        let last_idx = merged.len() - 1;
        let (prev_start, prev_end) = merged[last_idx];
        if start <= prev_end + 1 {
            merged[last_idx] = (prev_start, prev_end.max(end));
        } else {
            merged.push((start, end));
        }
    }

    Ok(Some(merged))
}

pub(crate) type SecEmbeddedHitRow = (
    String,
    String,
    Option<String>,
    Option<String>,
    usize,
    usize,
    usize,
    String,
);

#[derive(Clone)]
pub(crate) struct SecEmbeddedItemMatch {
    item_id: String,
    part: Option<String>,
    end: usize,
    num: Option<String>,
    letter: Option<String>,
}

pub(crate) fn sec_embedded_normalize_item_id(value: Option<&str>) -> Option<String> {
    let value = value?;
    let cleaned: String = value
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .flat_map(|ch| ch.to_uppercase())
        .collect();
    if cleaned.is_empty() {
        None
    } else {
        Some(cleaned)
    }
}

pub(crate) fn sec_embedded_normalize_part(value: Option<&str>) -> Option<String> {
    let part = value?.trim().to_ascii_uppercase();
    if matches!(part.as_str(), "I" | "II" | "III" | "IV") {
        Some(part)
    } else {
        None
    }
}

pub(crate) fn sec_embedded_roman_item_id(value: &str) -> Option<String> {
    let number = sec_roman_to_int_impl(value)?;
    if (1..=20).contains(&number) {
        Some(number.to_string())
    } else {
        None
    }
}

pub(crate) fn sec_embedded_item_id_to_int(value: Option<&str>) -> Option<i64> {
    let cleaned = sec_embedded_normalize_item_id(value)?;
    if let Some(number) = sec_embedded_roman_item_id(&cleaned) {
        return number.parse::<i64>().ok();
    }
    let digits: String = cleaned
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect();
    if digits.is_empty() {
        None
    } else {
        digits.parse::<i64>().ok()
    }
}

pub(crate) fn sec_embedded_is_late_item(
    current_item_id: Option<&str>,
    current_part: Option<&str>,
) -> bool {
    if current_part.is_some_and(|part| part.eq_ignore_ascii_case("IV")) {
        return true;
    }
    sec_embedded_item_id_to_int(current_item_id).is_some_and(|value| value >= 10)
}

pub(crate) fn sec_skip_space_tab(bytes: &[u8], pos: &mut usize) {
    while *pos < bytes.len() && matches!(bytes[*pos], b' ' | b'\t') {
        *pos += 1;
    }
}

pub(crate) fn sec_parse_embedded_item_after_item_word(
    line: &str,
    mut pos: usize,
) -> Option<SecEmbeddedItemMatch> {
    let bytes = line.as_bytes();
    if pos >= bytes.len() || !matches!(bytes[pos], b' ' | b'\t') {
        return None;
    }
    sec_skip_space_tab(bytes, &mut pos);
    let num_start = pos;
    let mut digit_count = 0usize;
    while pos < bytes.len() && bytes[pos].is_ascii_digit() && digit_count < 2 {
        pos += 1;
        digit_count += 1;
    }
    if digit_count == 0 {
        return None;
    }
    let num = &line[num_start..pos];
    let mut letter: Option<String> = None;
    if pos < bytes.len() && bytes[pos].is_ascii_uppercase() {
        let candidate_end = pos + 1;
        if extraction_ascii_boundary_after(bytes, candidate_end) {
            letter = Some(line[pos..candidate_end].to_string());
            pos = candidate_end;
        }
    }
    if !extraction_ascii_boundary_after(bytes, pos) {
        return None;
    }
    let item_id = format!("{}{}", num, letter.as_deref().unwrap_or("")).to_ascii_uppercase();
    Some(SecEmbeddedItemMatch {
        item_id,
        part: None,
        end: pos,
        num: Some(num.to_string()),
        letter,
    })
}

pub(crate) fn sec_embedded_item_match(line: &str) -> Option<SecEmbeddedItemMatch> {
    let bytes = line.as_bytes();
    let mut pos = 0usize;
    sec_skip_space_tab(bytes, &mut pos);
    if !starts_with_ascii_at(bytes, pos, b"item") {
        return None;
    }
    sec_parse_embedded_item_after_item_word(line, pos + 4)
}

pub(crate) fn sec_embedded_roman_item_match(line: &str) -> Option<SecEmbeddedItemMatch> {
    let bytes = line.as_bytes();
    let mut pos = 0usize;
    sec_skip_space_tab(bytes, &mut pos);
    if !starts_with_ascii_at(bytes, pos, b"item") {
        return None;
    }
    pos += 4;
    if pos >= bytes.len() || !matches!(bytes[pos], b' ' | b'\t') {
        return None;
    }
    sec_skip_space_tab(bytes, &mut pos);
    let roman_start = pos;
    while pos < bytes.len()
        && matches!(
            bytes[pos].to_ascii_uppercase(),
            b'I' | b'V' | b'X' | b'L' | b'C' | b'D' | b'M'
        )
    {
        pos += 1;
    }
    if pos == roman_start || !extraction_ascii_boundary_after(bytes, pos) {
        return None;
    }
    let roman = &line[roman_start..pos];
    let item_id = sec_embedded_roman_item_id(roman)?;
    Some(SecEmbeddedItemMatch {
        item_id,
        part: None,
        end: pos,
        num: None,
        letter: None,
    })
}

pub(crate) fn sec_embedded_part_item_match(line: &str) -> Option<SecEmbeddedItemMatch> {
    let bytes = line.as_bytes();
    let mut pos = 0usize;
    extraction_skip_ascii_ws(bytes, &mut pos);
    if !starts_with_ascii_at(bytes, pos, b"part") {
        return None;
    }
    pos += 4;
    if pos >= bytes.len() || !bytes[pos].is_ascii_whitespace() {
        return None;
    }
    extraction_skip_ascii_ws(bytes, &mut pos);
    let part_start = pos;
    while pos < bytes.len() && matches!(bytes[pos].to_ascii_uppercase(), b'I' | b'V' | b'X') {
        pos += 1;
    }
    if pos == part_start {
        return None;
    }
    let part = line[part_start..pos].to_ascii_uppercase();
    extraction_skip_ascii_ws(bytes, &mut pos);
    if pos < bytes.len() && matches!(bytes[pos], b',' | b'/' | b'-' | b':' | b';') {
        pos += 1;
        extraction_skip_ascii_ws(bytes, &mut pos);
    }
    if !starts_with_ascii_at(bytes, pos, b"item") {
        return None;
    }
    let mut item_match = sec_parse_embedded_item_after_item_word(line, pos + 4)?;
    item_match.part = Some(part);
    Some(item_match)
}

pub(crate) fn sec_embedded_part_line_match(line: &str) -> Option<String> {
    let bytes = line.as_bytes();
    let mut pos = 0usize;
    sec_skip_space_tab(bytes, &mut pos);
    if !starts_with_ascii_at(bytes, pos, b"part") {
        return None;
    }
    pos += 4;
    if pos >= bytes.len() || !matches!(bytes[pos], b' ' | b'\t') {
        return None;
    }
    sec_skip_space_tab(bytes, &mut pos);
    let part_start = pos;
    while pos < bytes.len() && matches!(bytes[pos].to_ascii_uppercase(), b'I' | b'V' | b'X') {
        pos += 1;
    }
    if pos == part_start {
        return None;
    }
    let part = line[part_start..pos].to_ascii_uppercase();
    sec_skip_space_tab(bytes, &mut pos);
    if pos < bytes.len() && matches!(bytes[pos], b'.' | b'-' | b':') {
        pos += 1;
    }
    sec_skip_space_tab(bytes, &mut pos);
    if pos == bytes.len() {
        Some(part)
    } else {
        None
    }
}

pub(crate) fn sec_embedded_line_starts_item_exact(line: &str) -> bool {
    sec_embedded_item_match(line).is_some()
        || sec_embedded_roman_item_match(line).is_some()
        || sec_embedded_part_item_match(line).is_some()
}

pub(crate) fn sec_embedded_dot_leader(line: &str) -> bool {
    let mut plain_dot_run = 0usize;
    let mut spaced_dot_count = 0usize;
    let mut in_spaced_run = false;
    for byte in line.bytes() {
        if byte == b'.' {
            plain_dot_run += 1;
            if plain_dot_run >= 8 {
                return true;
            }
            spaced_dot_count += 1;
            in_spaced_run = true;
            if spaced_dot_count >= 8 {
                return true;
            }
        } else {
            plain_dot_run = 0;
            if !byte.is_ascii_whitespace() {
                spaced_dot_count = 0;
                in_spaced_run = false;
            } else if !in_spaced_run {
                spaced_dot_count = 0;
            }
        }
    }
    false
}

pub(crate) fn sec_embedded_trailing_page(line: &str) -> bool {
    let trimmed = line.trim_end();
    let bytes = trimmed.as_bytes();
    if bytes.is_empty() {
        return false;
    }
    let mut pos = bytes.len();
    let mut digits = 0usize;
    while pos > 0 && bytes[pos - 1].is_ascii_digit() && digits < 4 {
        pos -= 1;
        digits += 1;
    }
    digits > 0 && pos > 0 && bytes[pos - 1].is_ascii_whitespace()
}

pub(crate) fn sec_embedded_trailing_page_gap(line: &str) -> bool {
    let trimmed = line.trim_end();
    let bytes = trimmed.as_bytes();
    let mut pos = bytes.len();
    let mut digits = 0usize;
    while pos > 0 && bytes[pos - 1].is_ascii_digit() && digits < 4 {
        pos -= 1;
        digits += 1;
    }
    if digits == 0 {
        return false;
    }
    let mut spaces = 0usize;
    while pos > 0 && bytes[pos - 1].is_ascii_whitespace() {
        pos -= 1;
        spaces += 1;
    }
    spaces >= 2
}

pub(crate) fn sec_embedded_words(text: &str) -> Vec<&str> {
    let bytes = text.as_bytes();
    let mut words = Vec::new();
    let mut pos = 0usize;
    while pos < bytes.len() {
        if !bytes[pos].is_ascii_alphabetic() {
            pos += 1;
            continue;
        }
        let start = pos;
        pos += 1;
        while pos < bytes.len()
            && (bytes[pos].is_ascii_alphabetic() || matches!(bytes[pos], b'\'' | b'&' | b'-'))
        {
            pos += 1;
        }
        words.push(&text[start..pos]);
    }
    words
}

pub(crate) fn sec_embedded_is_upper(text: &str) -> bool {
    let mut has_alpha = false;
    for byte in text.as_bytes() {
        if byte.is_ascii_alphabetic() {
            has_alpha = true;
            if byte.is_ascii_lowercase() {
                return false;
            }
        }
    }
    has_alpha
}

pub(crate) fn sec_embedded_word_titlecase(word: &str) -> bool {
    let bytes = word.as_bytes();
    if bytes.len() <= 1 || !bytes[0].is_ascii_uppercase() {
        return false;
    }
    let mut has_lower = false;
    for byte in &bytes[1..] {
        if byte.is_ascii_uppercase() {
            return false;
        }
        if byte.is_ascii_lowercase() {
            has_lower = true;
        }
    }
    has_lower
}

pub(crate) fn sec_embedded_title_like_suffix(suffix: &str) -> bool {
    let trimmed = suffix.trim_end();
    if trimmed.is_empty()
        || trimmed.ends_with('.')
        || trimmed.ends_with('!')
        || trimmed.ends_with('?')
    {
        return false;
    }
    let words = sec_embedded_words(suffix);
    if words.len() < 2 {
        return false;
    }
    if sec_embedded_is_upper(suffix) {
        return true;
    }
    let uppercase = words
        .iter()
        .filter(|word| word.as_bytes()[0].is_ascii_uppercase())
        .count();
    (uppercase as f64) / (words.len() as f64) >= 0.6
}

pub(crate) fn sec_embedded_toc_header(line: &str) -> bool {
    let lower = sec_gij_normalized_line(line);
    contains_ascii_word_phrase(&lower, "table of content")
        || contains_ascii_word_phrase(&lower, "table of contents")
        || contains_ascii_word_phrase(&lower, "index")
        || contains_ascii_word_phrase(&lower, "form 10-k summary")
        || contains_ascii_word_phrase(&lower, "form 10k summary")
        || contains_ascii_word_phrase(&lower, "10-k summary")
        || contains_ascii_word_phrase(&lower, "10k summary")
}

pub(crate) fn sec_embedded_toc_window_header(line: &str) -> bool {
    let lower = line.trim_start().to_ascii_lowercase();
    for prefix in [
        "table of contents",
        "table of content",
        "index",
        "form 10-k summary",
        "form 10k summary",
        "10-k summary",
        "10k summary",
    ] {
        if lower.starts_with(prefix) {
            let end = prefix.len();
            return lower
                .as_bytes()
                .get(end)
                .is_none_or(|byte| !is_ascii_word_byte(*byte));
        }
    }
    false
}

pub(crate) fn sec_embedded_toc_candidate_line(line: &str) -> bool {
    if line.trim().is_empty() {
        return false;
    }
    if sec_embedded_dot_leader(line) {
        return true;
    }
    sec_embedded_trailing_page(line) && sec_embedded_trailing_page_gap(line)
}

pub(crate) fn sec_embedded_toc_index_style_line(line: &str) -> bool {
    if line.trim().is_empty() {
        return false;
    }
    let item_match = sec_embedded_part_item_match(line).or_else(|| sec_embedded_item_match(line));
    let Some(item_match) = item_match else {
        return false;
    };
    let suffix = line[item_match.end..].trim();
    if suffix.is_empty() {
        return false;
    }
    if sec_embedded_dot_leader(line) || sec_embedded_trailing_page(line) {
        return true;
    }
    if item_match.part.is_some() {
        return true;
    }
    if suffix.starts_with(',') {
        return true;
    }
    if line[item_match.end..]
        .as_bytes()
        .windows(2)
        .any(|window| window[0].is_ascii_whitespace() && window[1].is_ascii_whitespace())
    {
        return true;
    }
    sec_embedded_title_like_suffix(suffix)
}

pub(crate) fn sec_embedded_toc_entry_like(line: &str) -> bool {
    sec_embedded_toc_candidate_line(line) || sec_embedded_toc_index_style_line(line)
}

pub(crate) fn sec_embedded_toc_header_nearby(lines: &[&str], idx: usize, window: usize) -> bool {
    let start = idx.saturating_sub(window);
    let end = lines.len().min(idx + window + 1);
    lines[start..end]
        .iter()
        .any(|line| sec_embedded_toc_header(line))
}

pub(crate) fn sec_embedded_toc_clustered(lines: &[&str], idx: usize) -> bool {
    let start = idx.saturating_sub(3);
    let end = lines.len().min(idx + 4);
    let mut count = 0usize;
    for line in &lines[start..end] {
        if sec_embedded_toc_entry_like(line) {
            count += 1;
            if count >= 2 {
                return true;
            }
        }
    }
    false
}

pub(crate) fn sec_embedded_toc_like_line(
    lines: &[&str],
    idx: usize,
    toc_window_flags: Option<&[bool]>,
) -> bool {
    let line = lines[idx];
    if line.trim().is_empty() {
        return false;
    }
    if sec_embedded_toc_entry_like(line) {
        return true;
    }
    if toc_window_flags.is_some_and(|flags| flags[idx])
        && (sec_embedded_line_starts_item_exact(line)
            || sec_embedded_part_line_match(line).is_some())
    {
        return true;
    }
    sec_embedded_trailing_page(line)
        && (sec_embedded_toc_header_nearby(lines, idx, 5) || sec_embedded_toc_clustered(lines, idx))
}

pub(crate) fn sec_embedded_toc_window_flags(lines: &[&str]) -> Vec<bool> {
    let mut flags = vec![false; lines.len()];
    let mut remaining = 0usize;
    for (idx, line) in lines.iter().enumerate() {
        if sec_embedded_toc_window_header(line) {
            remaining = 150;
        }
        if remaining > 0 {
            flags[idx] = true;
            remaining -= 1;
        }
    }
    flags
}

pub(crate) fn sec_embedded_sentence_like_line(line: &str) -> bool {
    let stripped = line.trim();
    if stripped.is_empty() || !(stripped.ends_with('.') || stripped.ends_with(';')) {
        return false;
    }
    let words = sec_embedded_words(stripped);
    if words.len() < 8
        || !stripped
            .as_bytes()
            .iter()
            .any(|byte| byte.is_ascii_lowercase())
    {
        return false;
    }
    let titlecase = words
        .iter()
        .filter(|word| sec_embedded_word_titlecase(word))
        .count();
    (titlecase as f64) / (words.len() as f64) <= 0.6
}

pub(crate) fn sec_embedded_confirm_prose_after(
    lines: &[&str],
    start_idx: usize,
    toc_window_flags: &[bool],
) -> bool {
    let mut non_empty = 0usize;
    for idx in (start_idx + 1)..lines.len() {
        let line = lines[idx];
        if line.trim().is_empty() {
            continue;
        }
        non_empty += 1;
        if non_empty > 10 {
            break;
        }
        if sec_embedded_line_starts_item_exact(line)
            || sec_embedded_part_line_match(line).is_some()
            || sec_embedded_toc_like_line(lines, idx, Some(toc_window_flags))
        {
            continue;
        }
        if sec_embedded_sentence_like_line(line) {
            return true;
        }
    }
    false
}

pub(crate) fn sec_embedded_toc_cluster_after(lines: &[&str], idx: usize) -> bool {
    let mut non_empty = 0usize;
    let mut heading_hits = 0usize;
    for line in lines.iter().skip(idx + 1) {
        if line.trim().is_empty() {
            continue;
        }
        non_empty += 1;
        if sec_embedded_line_starts_item_exact(line) || sec_embedded_part_line_match(line).is_some()
        {
            heading_hits += 1;
            if heading_hits >= 2 {
                return true;
            }
        }
        if non_empty >= 8 {
            break;
        }
    }
    false
}

pub(crate) fn sec_embedded_confirm_part_restart(lines: &[&str], start_idx: usize) -> bool {
    let mut non_empty = 0usize;
    let mut char_count = 0usize;
    for line in lines.iter().skip(start_idx + 1) {
        char_count += line.len();
        if char_count > 1500 {
            break;
        }
        if line.trim().is_empty() {
            continue;
        }
        non_empty += 1;
        if non_empty > 12 {
            break;
        }
        if sec_embedded_line_starts_item_exact(line) {
            return true;
        }
    }
    false
}

pub(crate) fn sec_embedded_cross_ref_text(line: &str) -> bool {
    let lower = sec_gij_normalized_line(line);
    for phrase in [
        "see",
        "refer to",
        "refer back to",
        "as discussed",
        "as described",
        "as set forth",
        "as noted",
        "included in",
        "incorporated by reference",
        "incorporated herein by reference",
        "of this form",
        "in this form",
        "set forth in",
        "pursuant to",
        "under",
        "in accordance with",
    ] {
        if contains_ascii_word_phrase(&lower, phrase) {
            return true;
        }
    }
    sec_embedded_toc_header(&lower)
}

pub(crate) fn sec_embedded_cross_ref_like(suffix: &str, next_line: Option<&str>) -> bool {
    let probe = suffix.trim();
    if !probe.is_empty() {
        let probe = &probe[..probe.len().min(120)];
        if sec_embedded_cross_ref_text(probe) {
            return true;
        }
    }
    if let Some(next_line) = next_line {
        let next_trim = next_line.trim();
        if !next_trim.is_empty() && sec_embedded_cross_ref_text(next_trim) {
            return true;
        }
    }
    false
}

pub(crate) fn sec_embedded_non_empty_line<'a>(
    lines: &'a [&'a str],
    start: usize,
    max_scan: usize,
) -> Option<&'a str> {
    let end = lines.len().min(start + max_scan);
    for line in &lines[start..end] {
        if !line.trim().is_empty() {
            return Some(*line);
        }
    }
    None
}

pub(crate) fn sec_embedded_line_snippet(line: &str) -> String {
    let snippet = line.trim();
    if snippet.len() > 200 {
        format!("{}...", snippet[..197].trim_end())
    } else {
        snippet.to_string()
    }
}

pub(crate) fn sec_embedded_has_reserved(line: &str) -> bool {
    let lower = line.to_ascii_lowercase();
    lower.contains("[reserved]") || lower.contains("[ reserved ]")
}

pub(crate) fn sec_embedded_has_continuation(line: &str) -> bool {
    let lower = line.to_ascii_lowercase();
    contains_ascii_word_phrase(&lower, "continued")
        || contains_ascii_word_phrase(&lower, "concluded")
        || lower.contains("cont.")
}

pub(crate) fn sec_embedded_glued_title_marker_item_id(
    item_match: &SecEmbeddedItemMatch,
    line: &str,
) -> Option<String> {
    let num = item_match.num.as_ref()?;
    item_match.letter.as_ref()?;
    let suffix = &line[item_match.end..];
    let bytes = suffix.as_bytes();
    if bytes.len() >= 2 && bytes[0] == b'.' && bytes[1].is_ascii_lowercase() {
        Some(num.to_string())
    } else {
        None
    }
}

#[pyfunction]
#[pyo3(signature = (full_text, current_item_id, current_part=None, next_item_id=None, nearby_item_ids=None, max_hits=3))]
pub(crate) fn sec_find_embedded_heading_hits(
    full_text: &str,
    current_item_id: &str,
    current_part: Option<&str>,
    next_item_id: Option<&str>,
    nearby_item_ids: Option<Vec<String>>,
    max_hits: i64,
) -> PyResult<Option<Vec<SecEmbeddedHitRow>>> {
    if !full_text.is_ascii()
        || !current_item_id.is_ascii()
        || current_part.is_some_and(|value| !value.is_ascii())
        || next_item_id.is_some_and(|value| !value.is_ascii())
        || nearby_item_ids
            .as_ref()
            .is_some_and(|values| values.iter().any(|value| !value.is_ascii()))
    {
        return Ok(None);
    }
    if full_text.is_empty() {
        return Ok(Some(Vec::new()));
    }

    let full_text_len = full_text.len();
    let lines = sec_splitlines_keepends(full_text);
    let lines_noeol: Vec<&str> = lines.iter().map(|line| sec_strip_line_eol(line)).collect();
    let current_item = sec_embedded_normalize_item_id(Some(current_item_id));
    let current_part_norm = sec_embedded_normalize_part(current_part);
    let next_item_norm = sec_embedded_normalize_item_id(next_item_id);
    let nearby_set: HashSet<String> = nearby_item_ids.unwrap_or_default().into_iter().collect();
    let toc_window_flags = sec_embedded_toc_window_flags(&lines_noeol);
    let is_late_item =
        sec_embedded_is_late_item(current_item.as_deref(), current_part_norm.as_deref());
    let mut hits: Vec<SecEmbeddedHitRow> = Vec::new();
    let mut non_ignored_hits = 0i64;
    let mut offset = 0usize;

    for (idx, (line, line_noeol)) in lines.iter().zip(lines_noeol.iter()).enumerate() {
        if line_noeol.trim().is_empty() {
            offset += line.len();
            continue;
        }

        let part_item_match = sec_embedded_part_item_match(line_noeol);
        let item_match = if part_item_match.is_none() {
            sec_embedded_item_match(line_noeol)
        } else {
            None
        };
        let roman_match = if part_item_match.is_none() && item_match.is_none() {
            sec_embedded_roman_item_match(line_noeol)
        } else {
            None
        };
        let part_match =
            if part_item_match.is_none() && item_match.is_none() && roman_match.is_none() {
                sec_embedded_part_line_match(line_noeol)
            } else {
                None
            };

        if part_item_match.is_none()
            && item_match.is_none()
            && roman_match.is_none()
            && part_match.is_none()
        {
            offset += line.len();
            continue;
        }

        let char_pos = offset + sec_leading_space_tab_len(line_noeol);
        let snippet = sec_embedded_line_snippet(line_noeol);
        let toc_cluster_detected = sec_embedded_toc_cluster_after(&lines_noeol, idx);

        let hit = if let Some(mut heading_match) = part_item_match.or(item_match).or(roman_match) {
            let mut embedded_item_id = heading_match.item_id.clone();
            let glued_title_item_id =
                sec_embedded_glued_title_marker_item_id(&heading_match, line_noeol);
            let glued_title_marker = glued_title_item_id.is_some();
            if let Some(item_id) = glued_title_item_id {
                embedded_item_id = item_id;
            }
            embedded_item_id = embedded_item_id.to_ascii_uppercase();
            let embedded_item_norm = sec_embedded_normalize_item_id(Some(&embedded_item_id));
            let successor_match = next_item_norm
                .as_ref()
                .zip(embedded_item_norm.as_ref())
                .is_some_and(|(next_item, embedded_item)| next_item == embedded_item);
            let nearby_match = embedded_item_norm
                .as_ref()
                .is_some_and(|embedded_item| nearby_set.contains(embedded_item));
            let embedded_part = heading_match.part.take();

            if current_item
                .as_ref()
                .zip(embedded_item_norm.as_ref())
                .is_some_and(|(current, embedded)| current == embedded)
                && char_pos <= 10
            {
                offset += line.len();
                continue;
            }

            let mut classification: String;
            if current_item
                .as_ref()
                .zip(embedded_item_norm.as_ref())
                .is_some_and(|(current, embedded)| current == embedded)
            {
                if sec_embedded_has_continuation(line_noeol) {
                    classification = "same_item_continuation".to_string();
                } else {
                    classification = "same_item_duplicate".to_string();
                }
            } else {
                let toc_like = sec_embedded_toc_like_line(&lines_noeol, idx, None);
                let toc_window_hit = toc_window_flags[idx];
                let suffix =
                    line_noeol[heading_match.end..].trim_matches([' ', '\t', ':', '-', '.']);
                let next_line = sec_embedded_non_empty_line(&lines_noeol, idx + 1, 3);
                let strong_prose =
                    sec_embedded_confirm_prose_after(&lines_noeol, idx, &toc_window_flags);
                let reserved = sec_embedded_has_reserved(line_noeol);
                let cross_ref = !toc_like && sec_embedded_cross_ref_like(suffix, next_line);
                let strong_overlap_evidence =
                    reserved || (strong_prose && sec_embedded_title_like_suffix(suffix));

                if toc_window_hit && !strong_prose {
                    classification = "toc_row".to_string();
                } else if toc_like {
                    classification = "toc_row".to_string();
                } else if cross_ref {
                    classification = "cross_ref_line".to_string();
                } else if reserved || strong_prose {
                    classification = "true_overlap".to_string();
                } else if sec_embedded_toc_entry_like(line_noeol) {
                    classification = "toc_row".to_string();
                } else {
                    classification = "cross_ref_line".to_string();
                }

                if classification == "true_overlap" && !nearby_match && !strong_overlap_evidence {
                    classification = "overlap_unconfirmed".to_string();
                }
            }

            if hits.is_empty()
                && char_pos <= 500
                && (toc_window_flags[idx]
                    || sec_embedded_dot_leader(line_noeol)
                    || sec_embedded_trailing_page(line_noeol)
                    || toc_cluster_detected)
            {
                classification = "toc_start_misfire_early".to_string();
            } else if char_pos <= 500 && toc_cluster_detected {
                classification = "toc_row".to_string();
            }

            if hits.is_empty()
                && is_late_item
                && char_pos <= 3000
                && (embedded_part
                    .as_ref()
                    .is_some_and(|part| part.eq_ignore_ascii_case("I"))
                    || sec_embedded_item_id_to_int(Some(&embedded_item_id))
                        .is_some_and(|item_num| item_num == 1 || item_num == 2))
            {
                classification = "toc_start_misfire".to_string();
            }

            if glued_title_marker
                && !matches!(
                    classification.as_str(),
                    "toc_start_misfire" | "toc_start_misfire_early"
                )
                && !successor_match
                && !sec_embedded_has_reserved(line_noeol)
            {
                classification = "glued_title_marker".to_string();
            }

            if successor_match
                && matches!(
                    classification.as_str(),
                    "true_overlap" | "overlap_unconfirmed"
                )
                && char_pos > 500
                && !toc_cluster_detected
            {
                classification = "true_overlap_next_item".to_string();
            }

            (
                "item".to_string(),
                classification,
                Some(embedded_item_id),
                embedded_part,
                idx,
                char_pos,
                full_text_len,
                snippet,
            )
        } else {
            let embedded_part = part_match.expect("part match should exist");
            let toc_like = sec_embedded_toc_like_line(&lines_noeol, idx, None);
            let toc_window_hit = toc_window_flags[idx];
            let strong_prose =
                sec_embedded_confirm_prose_after(&lines_noeol, idx, &toc_window_flags);
            let mut classification = if toc_window_hit && !strong_prose {
                "toc_row".to_string()
            } else if toc_like {
                "toc_row".to_string()
            } else if sec_embedded_confirm_part_restart(&lines_noeol, idx) {
                if current_part_norm
                    .as_ref()
                    .is_some_and(|current_part| current_part != &embedded_part)
                {
                    "part_restart".to_string()
                } else {
                    "part_restart_unconfirmed".to_string()
                }
            } else if sec_embedded_toc_entry_like(line_noeol) {
                "toc_row".to_string()
            } else {
                "part_restart_unconfirmed".to_string()
            };

            if hits.is_empty() && is_late_item && char_pos <= 3000 && embedded_part == "I" {
                classification = "toc_start_misfire".to_string();
            }
            if hits.is_empty() && char_pos <= 500 && toc_cluster_detected {
                classification = "toc_start_misfire_early".to_string();
            } else if char_pos <= 500 && toc_cluster_detected {
                classification = "toc_row".to_string();
            }

            (
                "part".to_string(),
                classification,
                None,
                Some(embedded_part),
                idx,
                char_pos,
                full_text_len,
                snippet,
            )
        };

        let ignored = hit.1 == "same_item_continuation";
        hits.push(hit);
        if !ignored {
            non_ignored_hits += 1;
            if non_ignored_hits >= max_hits {
                break;
            }
        }

        offset += line.len();
    }

    Ok(Some(hits))
}

#[pyfunction]
#[pyo3(signature = (text, next_item_id=None, current_part=None, max_item=20))]
pub(crate) fn sec_apply_high_confidence_truncation(
    text: &str,
    next_item_id: Option<&str>,
    current_part: Option<&str>,
    max_item: i64,
) -> PyResult<Option<(String, bool, bool)>> {
    if !text.is_ascii()
        || next_item_id.is_some_and(|value| !value.is_ascii())
        || current_part.is_some_and(|value| !value.is_ascii())
    {
        return Ok(None);
    }
    if text.is_empty() {
        return Ok(Some((text.to_string(), false, false)));
    }

    let lines = sec_splitlines_keepends(text);
    let lines_noeol: Vec<&str> = lines.iter().map(|line| sec_strip_line_eol(line)).collect();

    let mut non_empty = 0usize;
    let mut offset = 0usize;
    for (idx, line_noeol) in lines_noeol.iter().enumerate() {
        let line = lines[idx];
        if line_noeol.trim().is_empty() {
            offset += line.len();
            continue;
        }
        non_empty += 1;
        if sec_empty_section_line(line_noeol) {
            let cut = offset + line.len();
            return Ok(Some((text[..cut].trim_end().to_string(), false, false)));
        }
        offset += line.len();
        if non_empty >= 3 {
            break;
        }
    }

    if let Some(next_item) = next_item_id {
        offset = 0;
        for (idx, line_noeol) in lines_noeol.iter().enumerate() {
            if line_noeol.trim().is_empty() {
                offset += lines[idx].len();
                continue;
            }
            if let Some((item_id, match_end)) =
                sec_trunc_line_start_item_match(line_noeol, max_item)
            {
                if item_id == next_item {
                    let suffix =
                        line_noeol[match_end..].trim_start_matches([' ', '\t', ':', '-', '.']);
                    if !sec_cross_ref_suffix_start(suffix) {
                        let cut = offset + sec_leading_space_tab_len(line_noeol);
                        return Ok(Some((text[..cut].trim_end().to_string(), true, false)));
                    }
                }
            }
            offset += lines[idx].len();
        }
    }

    let current_order = sec_part_order(current_part);
    if current_order != 0 {
        let toc_window_flags = sec_toc_window_flags(&lines_noeol);
        offset = 0;
        for (idx, line_noeol) in lines_noeol.iter().enumerate() {
            if line_noeol.trim().is_empty() {
                offset += lines[idx].len();
                continue;
            }
            let Some(next_part) = sec_strict_part_line(line_noeol) else {
                offset += lines[idx].len();
                continue;
            };
            if toc_window_flags[idx] {
                offset += lines[idx].len();
                continue;
            }
            let next_order = sec_part_order(Some(next_part));
            if next_order <= current_order {
                offset += lines[idx].len();
                continue;
            }
            if !sec_confirm_strict_part_heading(&lines_noeol, idx) {
                offset += lines[idx].len();
                continue;
            }
            let cut = offset + sec_leading_space_tab_len(line_noeol);
            return Ok(Some((text[..cut].trim_end().to_string(), false, true)));
        }
    }

    Ok(Some((text.to_string(), false, false)))
}

pub(crate) fn sec_cik_10_from_digit_text(cik_text: &str) -> String {
    let stripped = cik_text.trim_start_matches('0');
    let normalized = if stripped.is_empty() { "0" } else { stripped };
    if normalized.len() >= 10 {
        normalized.to_string()
    } else {
        format!("{}{}", "0".repeat(10 - normalized.len()), normalized)
    }
}

pub(crate) type ParsedSecFilename = (
    String,
    String,
    String,
    String,
    String,
    Option<String>,
    String,
);

pub(crate) fn parse_sec_filename_minimal_impl(filename: &str) -> Option<ParsedSecFilename> {
    let suffix_len = ".txt".len();
    if filename.len() <= suffix_len
        || !filename[filename.len() - suffix_len..].eq_ignore_ascii_case(".txt")
    {
        return None;
    }
    let base = &filename[..filename.len() - suffix_len];
    let token = "_edgar_data_";
    let lower_base = base.to_ascii_lowercase();
    let token_pos = lower_base.find(token)?;
    let prefix = &base[..token_pos];
    let suffix = &base[token_pos + token.len()..];
    let (date_str, doc_type) = prefix.split_once('_')?;
    if date_str.len() != 8
        || !date_str.bytes().all(|byte| byte.is_ascii_digit())
        || doc_type.is_empty()
        || doc_type.contains('_')
    {
        return None;
    }
    let (cik_text, accession) = suffix.split_once('_')?;
    if cik_text.is_empty()
        || !cik_text.bytes().all(|byte| byte.is_ascii_digit())
        || accession.is_empty()
        || accession.contains('_')
        || !accession
            .bytes()
            .all(|byte| byte.is_ascii_digit() || byte == b'-')
    {
        return None;
    }
    let cik10 = sec_cik_10_from_digit_text(cik_text);
    let accession_nodash: String = accession.chars().filter(|ch| ch.is_ascii_digit()).collect();
    let accession_nodash = if accession_nodash.is_empty() {
        None
    } else {
        Some(accession_nodash)
    };
    let doc_id = format!("{cik10}:{accession}");
    Some((
        date_str.to_string(),
        doc_type.to_string(),
        cik_text.to_string(),
        accession.to_string(),
        cik10,
        accession_nodash,
        doc_id,
    ))
}

#[pyfunction]
#[pyo3(signature = (filename))]
pub(crate) fn parse_sec_filename_minimal_value(
    filename: &str,
) -> PyResult<Option<ParsedSecFilename>> {
    if !filename.is_ascii() {
        return Err(PyValueError::new_err("non-ascii input"));
    }
    Ok(parse_sec_filename_minimal_impl(filename))
}

pub(crate) fn py_dict_present_non_none(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<bool> {
    Ok(match dict.get_item(key)? {
        Some(value) => !value.is_none(),
        None => false,
    })
}

pub(crate) fn py_dict_exact_true(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<bool> {
    let Some(value) = dict.get_item(key)? else {
        return Ok(false);
    };
    if !value.is_instance_of::<PyBool>() {
        return Ok(false);
    }
    value.extract::<bool>()
}

pub(crate) fn classify_boundary_authority_status_dict(
    dict: &Bound<'_, PyDict>,
) -> PyResult<&'static str> {
    let authority_fields = [
        "boundary_heading_start",
        "boundary_heading_end",
        "boundary_content_start",
        "boundary_content_end",
        "boundary_heading_line_index",
        "boundary_heading_offset",
        "boundary_start_candidates_total",
        "boundary_start_candidates_toc_rejected",
        "boundary_start_selection_verified",
        "boundary_truncated_successor_heading",
        "boundary_truncated_part_boundary",
    ];
    let mut has_any_authority_signal = false;
    for field in authority_fields {
        if py_dict_present_non_none(dict, field)? {
            has_any_authority_signal = true;
            break;
        }
    }
    if !has_any_authority_signal {
        return Ok("unknown");
    }

    for field in [
        "boundary_heading_start",
        "boundary_heading_end",
        "boundary_content_start",
        "boundary_content_end",
    ] {
        if !py_dict_present_non_none(dict, field)? {
            return Ok("review_needed");
        }
    }

    if !py_dict_exact_true(dict, "boundary_start_selection_verified")? {
        return Ok("review_needed");
    }

    if let Some(value) = dict.get_item("boundary_start_candidates_toc_rejected")? {
        if !value.is_none() && py_int_like_to_i64(&value)? > 0 {
            return Ok("review_needed");
        }
    }

    for field in [
        "boundary_truncated_successor_heading",
        "boundary_truncated_part_boundary",
    ] {
        if let Some(value) = dict.get_item(field)? {
            if !value.is_none() && value.is_truthy()? {
                return Ok("review_needed");
            }
        }
    }

    Ok("trusted")
}

#[pyfunction]
pub(crate) fn classify_boundary_authority_status_value(
    item: &Bound<'_, PyAny>,
) -> PyResult<&'static str> {
    let dict = item
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("boundary authority item is not a dict"))?;
    classify_boundary_authority_status_dict(dict)
}

#[pyfunction]
pub(crate) fn public_boundary_payload_value(
    py: Python<'_>,
    item: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let dict = item
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("boundary payload item is not a dict"))?;
    let out = PyDict::new_bound(py);
    for (private_name, public_name) in [
        ("_heading_start", "boundary_heading_start"),
        ("_heading_end", "boundary_heading_end"),
        ("_content_start", "boundary_content_start"),
        ("_content_end", "boundary_content_end"),
        ("_heading_line_index", "boundary_heading_line_index"),
        ("_heading_offset", "boundary_heading_offset"),
        ("_start_candidates_total", "boundary_start_candidates_total"),
        (
            "_start_candidates_toc_rejected",
            "boundary_start_candidates_toc_rejected",
        ),
        (
            "_start_selection_verified",
            "boundary_start_selection_verified",
        ),
        (
            "_truncated_successor_heading",
            "boundary_truncated_successor_heading",
        ),
        (
            "_truncated_part_boundary",
            "boundary_truncated_part_boundary",
        ),
    ] {
        match dict.get_item(private_name)? {
            Some(value) => out.set_item(public_name, value)?,
            None => out.set_item(public_name, py.None())?,
        }
    }
    let status = classify_boundary_authority_status_dict(&out)?;
    out.set_item("boundary_authority_status", status)?;
    Ok(out.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_toc_window_header_accepts_plural_contents() {
        assert!(sec_embedded_toc_window_header("TABLE OF CONTENTS"));
        assert!(sec_embedded_toc_window_header("Table of Content"));
    }

    #[test]
    fn pagination_page_of_matches_python_spacing_contract() {
        assert!(sec_remove_pagination_page_of("Page 1"));
        assert!(sec_remove_pagination_page_of("Page 1 of 10"));
        assert!(sec_remove_pagination_page_of("Page 1of 10"));
        assert!(!sec_remove_pagination_page_of("Page 1of10"));
    }

    #[test]
    fn embedded_line_starts_item_rejects_glued_title_tokens() {
        assert!(sec_embedded_line_starts_item("Item 10. Directors"));
        assert!(sec_embedded_line_starts_item("PART III, Item 10"));
        assert!(!sec_embedded_line_starts_item("Item 10Directors"));
    }

    #[test]
    fn embedded_dot_leader_requires_a_dot_run() {
        assert!(sec_embedded_dot_leader("Item 1 ........ 3"));
        assert!(sec_embedded_dot_leader("Item 1 . . . . . . . . 3"));
        assert!(!sec_embedded_dot_leader(
            "Icahn Enterprises L.P., or Icahn Enterprises, is a master limited partnership formed in Delaware on February 17, 1987. On September 17, 2007, we changed our name from American Real Estate Partners, L.P. to Icahn Enterprises L.P. We are a diversified holding company."
        ));
    }

    #[test]
    fn embedded_prose_after_toc_part_restart_matches_python_guard() {
        let lines = [
            "Principal Accounting Fees and Services ",
            "",
            "172 ",
            "",
            "PART IV ",
            "",
            "Item 15. Exhibits and Financial Statement Schedules ",
            "",
            "173 ",
            "",
            "i",
            "TABLE OF CONTENTS ",
            "",
            " PART I ",
            "",
            " Item 1. Business ",
            "",
            " Introduction ",
            "",
            " Icahn Enterprises L.P., or Icahn Enterprises, is a master limited partnership formed in Delaware on February 17, 1987. On September 17, 2007, we changed our name from American Real Estate Partners, L.P. to Icahn Enterprises L.P. We are a diversified holding company owning subsidiaries engaged in the following continuing operating businesses: Investment Management, Metals, Real Estate and Home Fashion. In addition, as of December 31, 2007, we operated our Gaming segment, which under generally accepted accounting principles is considered discontinued operations as it was in the process of being sold at such date. ",
        ];
        let flags = sec_embedded_toc_window_flags(&lines);
        assert!(sec_embedded_sentence_like_line(lines[19]));
        assert!(sec_embedded_confirm_prose_after(&lines, 13, &flags));
    }
}

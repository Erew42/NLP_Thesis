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
use crate::sec_extraction::*;
use crate::sentence_cleaning::*;

#[pyfunction]
#[pyo3(signature = (text=""))]
pub(crate) fn sentence_is_separator_line(text: &str) -> bool {
    sentence_is_separator_line_impl(text)
}

#[pyfunction]
#[pyo3(signature = (text=""))]
pub(crate) fn normalize_sentence_key_value(text: &str) -> String {
    normalize_sentence_key_impl(text)
}

#[pyfunction]
#[pyo3(signature = (text=""))]
pub(crate) fn sentence_numeric_only_fragment(text: &str) -> bool {
    sentence_numeric_only_fragment_impl(text)
}

#[pyfunction]
#[pyo3(signature = (text="", token_count=0, char_count=0))]
pub(crate) fn sentence_short_fragment(text: &str, token_count: i64, char_count: i64) -> bool {
    sentence_short_fragment_impl(text, token_count, char_count)
}

#[pyfunction]
#[pyo3(signature = (text="", token_count=0, char_count=0))]
pub(crate) fn sentence_very_short_fragment(text: &str, token_count: i64, char_count: i64) -> bool {
    sentence_very_short_fragment_impl(text, token_count, char_count)
}

#[pyfunction]
#[pyo3(signature = (text="", token_count=0, char_count=0))]
pub(crate) fn sentence_lower_fragment(text: &str, token_count: i64, char_count: i64) -> bool {
    sentence_lower_fragment_impl(text, token_count, char_count)
}

#[pyfunction]
#[pyo3(signature = (text=""))]
pub(crate) fn sentence_one_word_fragment(text: &str) -> bool {
    sentence_one_word_fragment_impl(text)
}

#[pyfunction]
#[pyo3(signature = (text=""))]
pub(crate) fn sentence_has_terminal_punct(text: &str) -> bool {
    sentence_has_terminal_punct_impl(text)
}

#[pyfunction]
#[pyo3(signature = (text="", token_count=0))]
pub(crate) fn sentence_table_like(text: &str, token_count: usize) -> bool {
    sentence_table_like_impl(text, token_count)
}

#[pyfunction]
#[pyo3(signature = (text=""))]
pub(crate) fn sentence_ends_with_reference_stub(text: &str) -> bool {
    sentence_ends_with_reference_stub_impl(text)
}

#[pyfunction]
#[pyo3(signature = (text=""))]
pub(crate) fn sentence_ends_with_generic_reference_no(text: &str) -> bool {
    sentence_ends_with_generic_reference_no_impl(text)
}

#[pyfunction]
#[pyo3(signature = (text=""))]
pub(crate) fn sentence_looks_like_citation_continuation(text: &str) -> bool {
    sentence_looks_like_citation_continuation_impl(text, false)
}

#[pyfunction]
#[pyo3(signature = (text=""))]
pub(crate) fn sentence_looks_like_citation_continuation_v3(text: &str) -> bool {
    sentence_looks_like_citation_continuation_impl(text, true)
}

#[pyfunction]
#[pyo3(signature = (text="", next_text=""))]
pub(crate) fn sentence_generic_no_with_continuation(text: &str, next_text: &str) -> bool {
    sentence_generic_no_with_continuation_impl(text, next_text)
}

#[pyfunction]
#[pyo3(signature = (text=""))]
pub(crate) fn sentence_is_citation_prefix_only_line(text: &str) -> bool {
    sentence_is_citation_prefix_only_line_impl(text)
}

#[pyfunction]
#[pyo3(signature = (text=""))]
pub(crate) fn sentence_is_header_like_line(text: &str) -> bool {
    sentence_is_header_like_line_impl(text)
}

pub(crate) type SentenceQualityFlagRow = Vec<bool>;

#[pyfunction]
pub(crate) fn sentence_quality_batch_flags(
    rows: &Bound<'_, PyAny>,
) -> PyResult<Vec<SentenceQualityFlagRow>> {
    let mut out: Vec<SentenceQualityFlagRow> = Vec::new();
    for row in rows.iter()? {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("sentence quality row is not a dict"))?;
        let text = dict_python_or_empty_string(dict, "sentence_text")?;
        let next_text = dict_python_or_empty_string(dict, "next_sentence_text")?;
        let token_count = dict_optional_i64_or_zero(dict, "finbert_token_count_512")?;
        let char_count = dict_optional_i64_or_zero(dict, "sentence_char_count")?;
        let table_token_count = if token_count < 0 {
            0usize
        } else {
            usize::try_from(token_count)
                .map_err(|_| PyValueError::new_err("token count out of range"))?
        };
        out.push(vec![
            sentence_short_fragment_impl(&text, token_count, char_count),
            sentence_very_short_fragment_impl(&text, token_count, char_count),
            sentence_one_word_fragment_impl(&text),
            sentence_numeric_only_fragment_impl(&text),
            sentence_is_separator_line_impl(&text),
            sentence_ends_with_reference_stub_impl(&text),
            sentence_ends_with_generic_reference_no_impl(&text),
            sentence_generic_no_with_continuation_impl(&text, &next_text),
            sentence_is_citation_prefix_only_line_impl(&text),
            sentence_is_header_like_line_impl(&text),
            sentence_table_like_impl(&text, table_token_count),
            sentence_lower_fragment_impl(&text, token_count, char_count),
            !sentence_has_terminal_punct_impl(&text),
        ]);
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (sentence_texts, text_scope=None, policy="none"))]
pub(crate) fn postprocess_sentence_texts_value(
    sentence_texts: Vec<String>,
    text_scope: Option<&str>,
    policy: &str,
) -> Vec<String> {
    postprocess_sentence_texts_impl(sentence_texts, text_scope, policy)
}

#[pyfunction]
#[pyo3(signature = (text=""))]
pub(crate) fn cleaning_is_page_marker_line(text: &str) -> bool {
    cleaning_is_page_marker_line_impl(text)
}

#[pyfunction]
#[pyo3(signature = (text=""))]
pub(crate) fn cleaning_is_report_header_footer_line(text: &str) -> bool {
    cleaning_is_report_header_footer_line_impl(text)
}

#[pyfunction]
#[pyo3(signature = (text=""))]
pub(crate) fn cleaning_is_structural_residue_line(text: &str) -> bool {
    cleaning_is_structural_residue_line_impl(text)
}

#[pyfunction]
#[pyo3(signature = (text, drop_page_markers=true, drop_report_headers=true, drop_structural_tags=true))]
pub(crate) fn cleaning_remove_layout_lines_value(
    text: &str,
    drop_page_markers: bool,
    drop_report_headers: bool,
    drop_structural_tags: bool,
) -> (String, BTreeMap<String, i64>) {
    cleaning_remove_layout_lines_impl(
        text,
        drop_page_markers,
        drop_report_headers,
        drop_structural_tags,
    )
}

#[pyfunction]
#[pyo3(signature = (text, table_like_min_consecutive_lines=1, table_like_allow_single_line_with_header=true, table_like_drop_header_context=true))]
pub(crate) fn cleaning_remove_table_like_lines_value(
    text: &str,
    table_like_min_consecutive_lines: usize,
    table_like_allow_single_line_with_header: bool,
    table_like_drop_header_context: bool,
) -> (String, BTreeMap<String, i64>) {
    cleaning_remove_table_like_lines_impl(
        text,
        table_like_min_consecutive_lines,
        table_like_allow_single_line_with_header,
        table_like_drop_header_context,
    )
}

#[pyfunction]
#[pyo3(signature = (text=""))]
pub(crate) fn cleaning_is_table_like_line(text: &str) -> bool {
    cleaning_is_table_like_line_impl(text)
}

#[pyfunction]
#[pyo3(signature = (text=""))]
pub(crate) fn cleaning_is_table_header_like_line(text: &str) -> bool {
    cleaning_is_table_header_like_line_impl(text)
}

#[pyfunction]
#[pyo3(signature = (text=""))]
pub(crate) fn cleaning_is_strong_table_title_line(text: &str) -> bool {
    cleaning_is_strong_table_title_line_impl(text)
}

#[pyfunction]
#[pyo3(signature = (text=""))]
pub(crate) fn cleaning_is_table_intro_line(text: &str) -> bool {
    cleaning_is_table_intro_line_impl(text)
}

#[pyfunction]
#[pyo3(signature = (text=""))]
pub(crate) fn cleaning_is_table_support_header_line(text: &str) -> bool {
    cleaning_is_table_support_header_line_impl(text)
}

#[pyfunction]
#[pyo3(signature = (text=""))]
pub(crate) fn cleaning_is_toc_like_line(text: &str) -> bool {
    cleaning_toc_like_line_impl(text)
}

#[derive(Clone)]
pub(crate) struct TailCandidate {
    pub(crate) start: usize,
    pub(crate) text: String,
    pub(crate) reason: &'static str,
}

pub(crate) fn ascii_word_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

pub(crate) fn starts_with_ascii_at(bytes: &[u8], pos: usize, needle: &[u8]) -> bool {
    pos + needle.len() <= bytes.len()
        && bytes[pos..pos + needle.len()]
            .iter()
            .zip(needle.iter())
            .all(|(actual, expected)| actual.to_ascii_lowercase() == *expected)
}

pub(crate) fn contains_phrase_with_ascii_boundaries(line: &str, phrase: &[u8]) -> bool {
    let bytes = line.as_bytes();
    if bytes.len() < phrase.len() {
        return false;
    }
    for pos in 0..=bytes.len() - phrase.len() {
        if !starts_with_ascii_at(bytes, pos, phrase) {
            continue;
        }
        let left_ok = pos == 0 || !ascii_word_byte(bytes[pos - 1]);
        let right_pos = pos + phrase.len();
        let right_ok = right_pos == bytes.len() || !ascii_word_byte(bytes[right_pos]);
        if left_ok && right_ok {
            return true;
        }
    }
    false
}

pub(crate) fn contains_exhibit_index_anchor(line: &str) -> bool {
    contains_phrase_with_ascii_boundaries(line, b"exhibit index")
        || contains_phrase_with_ascii_boundaries(line, b"index to exhibits")
}

pub(crate) fn contains_ex_tag_anchor(line: &str) -> bool {
    let bytes = line.as_bytes();
    let mut pos = 0usize;
    while pos < bytes.len() {
        if bytes[pos] != b'<' {
            pos += 1;
            continue;
        }
        let mut cursor = pos + 1;
        if cursor < bytes.len() && bytes[cursor] == b'/' {
            cursor += 1;
        }
        if !starts_with_ascii_at(bytes, cursor, b"ex-") {
            pos += 1;
            continue;
        }
        cursor += 3;
        if cursor >= bytes.len() || !bytes[cursor].is_ascii_alphanumeric() {
            pos += 1;
            continue;
        }
        cursor += 1;
        while cursor < bytes.len()
            && (bytes[cursor].is_ascii_alphanumeric() || matches!(bytes[cursor], b'.' | b'-'))
        {
            cursor += 1;
        }
        if cursor < bytes.len() && bytes[cursor] == b'>' {
            return true;
        }
        pos += 1;
    }
    false
}

pub(crate) fn weak_exhibit_line_anchor(line: &str) -> bool {
    let bytes = line.as_bytes();
    let mut pos = 0usize;
    skip_ascii_ws(bytes, &mut pos);
    if !starts_with_ascii_at(bytes, pos, b"exhibit") {
        return false;
    }
    pos += 7;
    if pos >= bytes.len() || !bytes[pos].is_ascii_whitespace() {
        return false;
    }
    skip_ascii_ws(bytes, &mut pos);
    pos < bytes.len() && bytes[pos].is_ascii_alphanumeric()
}

pub(crate) fn iter_lines_with_char_offsets(text: &str) -> Vec<(usize, String)> {
    let mut lines = Vec::new();
    let mut offset = 0usize;
    for raw_line in text.split_inclusive('\n') {
        let line = raw_line.trim_end_matches(['\r', '\n']).to_string();
        lines.push((offset, line));
        offset += raw_line.chars().count();
    }
    if lines.is_empty() && !text.is_empty() {
        lines.push((0, text.to_string()));
    }
    lines
}

pub(crate) fn line_starts_with_marker(line: &str, marker_words: &[&str]) -> bool {
    let bytes = line.as_bytes();
    let mut pos = 0usize;
    skip_ascii_ws(bytes, &mut pos);
    for (index, word) in marker_words.iter().enumerate() {
        if index > 0 {
            if pos >= bytes.len() || !bytes[pos].is_ascii_whitespace() {
                return false;
            }
            skip_ascii_ws(bytes, &mut pos);
        }
        let word_bytes = word.as_bytes();
        if !starts_with_ascii_at(bytes, pos, word_bytes) {
            return false;
        }
        pos += word_bytes.len();
    }
    pos == bytes.len() || !ascii_word_byte(bytes[pos])
}

pub(crate) fn line_starts_with_signature_marker(line: &str) -> bool {
    let bytes = line.as_bytes();
    let mut pos = 0usize;
    skip_ascii_ws(bytes, &mut pos);
    if !starts_with_ascii_at(bytes, pos, b"signature") {
        return false;
    }
    pos += b"signature".len();
    if pos < bytes.len() && matches!(bytes[pos], b's' | b'S') {
        pos += 1;
    }
    pos == bytes.len() || !ascii_word_byte(bytes[pos])
}

pub(crate) fn cleaning_tail_marker_is_strong(line: &str) -> bool {
    line_starts_with_signature_marker(line)
        || line_starts_with_marker(line, &["index", "to", "exhibits"])
        || line_starts_with_marker(line, &["exhibit", "index"])
}

pub(crate) fn cleaning_tail_marker_match(line: &str, text_scope: &str) -> Option<bool> {
    let matched = match text_scope {
        "item_7_mda" => {
            line_starts_with_marker(line, &["item", "7a"])
                || line_starts_with_marker(line, &["item", "8"])
                || line_starts_with_marker(line, &["part", "iii"])
                || line_starts_with_marker(line, &["index", "to", "exhibits"])
                || line_starts_with_marker(line, &["exhibit", "index"])
        }
        "item_1a_risk_factors" => {
            line_starts_with_marker(line, &["item", "1b"])
                || line_starts_with_marker(line, &["item", "2"])
                || line_starts_with_marker(line, &["part", "ii"])
                || line_starts_with_marker(line, &["index", "to", "exhibits"])
                || line_starts_with_marker(line, &["exhibit", "index"])
        }
        "item_1_business" => {
            line_starts_with_marker(line, &["item", "1a"])
                || line_starts_with_marker(line, &["item", "1b"])
                || line_starts_with_marker(line, &["item", "2"])
                || line_starts_with_marker(line, &["part", "ii"])
                || line_starts_with_marker(line, &["index", "to", "exhibits"])
                || line_starts_with_marker(line, &["exhibit", "index"])
        }
        _ => return None,
    };
    if matched || line_starts_with_signature_marker(line) {
        Some(cleaning_tail_marker_is_strong(line))
    } else {
        None
    }
}

pub(crate) fn cleaning_tail_bleed_start_impl(
    text: &str,
    text_scope: &str,
    tail_scan_fraction: f64,
) -> Option<usize> {
    if text.is_empty() {
        return None;
    }
    let char_len = text.chars().count();
    let ordinary_tail_start = ((char_len as f64) * (1.0 - tail_scan_fraction.max(0.0))) as usize;
    let strong_tail_start = ((char_len as f64) * 0.50) as usize;
    let mut best_start: Option<usize> = None;
    for (start, line) in iter_lines_with_char_offsets(text) {
        let Some(is_strong_terminal) = cleaning_tail_marker_match(&line, text_scope) else {
            continue;
        };
        let min_start = if is_strong_terminal {
            strong_tail_start
        } else {
            ordinary_tail_start
        };
        if start < min_start {
            continue;
        }
        if best_start.is_none_or(|current| start < current) {
            best_start = Some(start);
        }
    }
    best_start
}

#[pyfunction]
pub(crate) fn cleaning_tail_bleed_start(
    text: &str,
    text_scope: &str,
    tail_scan_fraction: f64,
) -> Option<usize> {
    cleaning_tail_bleed_start_impl(text, text_scope, tail_scan_fraction)
}

pub(crate) fn normalized_ascii_words_key(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut pending_space = false;
    for ch in text.trim().chars() {
        if ch.is_whitespace() {
            if !out.is_empty() {
                pending_space = true;
            }
            continue;
        }
        if pending_space && !out.is_empty() {
            out.push(' ');
        }
        out.push(ch.to_ascii_lowercase());
        pending_space = false;
    }
    out
}

pub(crate) fn cleaning_is_reference_only_stub_impl(text: &str, max_char_count: usize) -> bool {
    let stripped = text.trim();
    if stripped.is_empty() || stripped.chars().count() > max_char_count {
        return false;
    }
    let lowered = normalized_ascii_words_key(stripped);
    let reference_prefix = "reference is made to";
    let reference_prefix_match = lowered.starts_with(reference_prefix)
        && lowered
            .as_bytes()
            .get(reference_prefix.len())
            .is_none_or(|byte| !ascii_word_byte(*byte));
    contains_ascii_word_or_phrase(&lowered, "incorporated by reference")
        || reference_prefix_match
        || contains_ascii_word_or_phrase(&lowered, "the information required by item")
        || contains_ascii_word_or_phrase(&lowered, "the information required by this item")
}

#[pyfunction]
pub(crate) fn cleaning_is_reference_only_stub(text: &str, max_char_count: usize) -> bool {
    cleaning_is_reference_only_stub_impl(text, max_char_count)
}

pub(crate) fn rstrip_lines_join_impl(text: &str) -> String {
    text.split('\n')
        .map(|line| line.trim_end().to_string())
        .collect::<Vec<_>>()
        .join("\n")
}

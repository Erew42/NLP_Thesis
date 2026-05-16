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
use crate::refinitiv_analyst::*;
use crate::refinitiv_authority::*;
use crate::refinitiv_bridge::*;
use crate::refinitiv_excel::*;
use crate::sec_extraction::*;
use crate::sentence_cleaning::*;
use crate::sentence_quality_api::*;

pub(crate) struct MultisurfaceEscalationCase {
    score: i32,
    case_id: String,
    original_index: usize,
    reasons: Vec<&'static str>,
    row: PyObject,
}

pub(crate) fn multisurface_required_string(
    dict: &Bound<'_, PyDict>,
    key: &str,
) -> PyResult<String> {
    let Some(value) = dict.get_item(key)? else {
        return Err(PyValueError::new_err(format!(
            "missing required key: {key}"
        )));
    };
    Ok(value.str()?.to_str()?.to_string())
}

pub(crate) fn multisurface_truthy_string(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<String> {
    let Some(value) = dict.get_item(key)? else {
        return Err(PyValueError::new_err(format!(
            "missing required key: {key}"
        )));
    };
    if value.is_truthy()? {
        Ok(value.str()?.to_str()?.to_string())
    } else {
        Ok(String::new())
    }
}

pub(crate) fn multisurface_escalation_reason(reasons: &[&'static str]) -> String {
    if reasons.is_empty() {
        return "context_review".to_string();
    }
    let mut seen: HashSet<&'static str> = HashSet::new();
    let mut ordered: Vec<&'static str> = Vec::new();
    for reason in reasons {
        if seen.insert(*reason) {
            ordered.push(*reason);
        }
    }
    ordered.join("|")
}

pub(crate) fn python_slice_end_len(row_count: usize, cap: i64) -> usize {
    if cap >= 0 {
        return usize::try_from(cap).unwrap_or(usize::MAX).min(row_count);
    }
    let trim = cap
        .checked_neg()
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(usize::MAX);
    row_count.saturating_sub(trim)
}

#[pyfunction]
pub(crate) fn multisurface_mark_escalated_cases(
    py: Python<'_>,
    cases: &Bound<'_, PyAny>,
    cap: i64,
) -> PyResult<Vec<PyObject>> {
    let mut scored_cases: Vec<MultisurfaceEscalationCase> = Vec::new();
    for (original_index, case) in cases.iter()?.enumerate() {
        let case = case?;
        let dict = case
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("multi-surface audit case is not a dict"))?;
        let case_source = multisurface_required_string(dict, "case_source")?;
        let suspect_reason = multisurface_truthy_string(dict, "suspect_reason")?;
        let text_scope = multisurface_truthy_string(dict, "text_scope")?;
        let case_id = multisurface_required_string(dict, "case_id")?;

        let mut score = 0_i32;
        let mut reasons: Vec<&'static str> = Vec::new();
        if case_source == "item_boundary_risk" {
            score += 5;
            reasons.push("boundary_context_required");
        }
        if suspect_reason.contains("row_boundary_review_needed")
            || suspect_reason.contains("manual_boundary_sample")
        {
            score += 4;
            reasons.push("item_sentence_disagreement");
        }
        if case_source == "sentence_long_table_like" {
            score += 4;
            reasons.push("table_vs_prose_ambiguity");
        }
        if case_source == "sentence_short_fragment"
            && (suspect_reason.contains("reference_stub")
                || suspect_reason.contains("heading_fragment"))
        {
            score += 3;
            reasons.push("split_or_heading_context");
        }
        if case_source == "item_doc_hotspot" {
            score += 3;
            reasons.push("doc_hotspot_context");
        }
        if case_source == "item_cleaning_flagged" {
            score += 2;
            reasons.push("cleaning_context");
        }
        if text_scope == "item_7_mda" {
            score += 1;
            reasons.push("item7_context");
        }

        scored_cases.push(MultisurfaceEscalationCase {
            score,
            case_id,
            original_index,
            reasons,
            row: case.clone().into_py(py),
        });
    }

    scored_cases.sort_by(|left, right| {
        right
            .score
            .cmp(&left.score)
            .then_with(|| left.case_id.cmp(&right.case_id))
            .then_with(|| left.original_index.cmp(&right.original_index))
    });

    let mut escalate_case_ids: HashSet<String> = scored_cases
        .iter()
        .filter(|case| case.score > 0)
        .map(|case| case.case_id.clone())
        .collect();
    if (escalate_case_ids.len() as i64) > cap {
        let end = python_slice_end_len(scored_cases.len(), cap);
        escalate_case_ids = scored_cases
            .iter()
            .take(end)
            .map(|case| case.case_id.clone())
            .collect();
    }

    let mut enriched: Vec<PyObject> = Vec::with_capacity(scored_cases.len());
    for case in scored_cases {
        let source = case
            .row
            .bind(py)
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("multi-surface audit case is not a dict"))?;
        let updated = copy_py_dict(py, source)?;
        if escalate_case_ids.contains(&case.case_id) {
            updated.set_item("full_report_needed", true)?;
            updated.set_item(
                "escalation_reason",
                multisurface_escalation_reason(&case.reasons),
            )?;
        }
        enriched.push(updated.into_py(py));
    }
    Ok(enriched)
}

#[pyfunction]
#[pyo3(signature = (text=None))]
pub(crate) fn normalize_newlines_value(text: Option<&str>) -> String {
    normalize_newlines_impl(text)
}

#[pyfunction]
pub(crate) fn collapse_blank_runs_value(text: &str) -> String {
    collapse_blank_runs_impl(text)
}

#[pyfunction]
pub(crate) fn plan_text_microbatch_spans(
    byte_sizes: Vec<i64>,
    max_docs_per_batch: usize,
    max_input_text_bytes_per_microbatch: i64,
) -> Vec<(usize, usize)> {
    if byte_sizes.is_empty() || max_docs_per_batch == 0 {
        return Vec::new();
    }
    let mut spans: Vec<(usize, usize)> = Vec::new();
    let mut current_start = 0usize;
    let mut current_count = 0usize;
    let mut current_bytes = 0i64;
    for (row_index, input_bytes) in byte_sizes.into_iter().enumerate() {
        let should_flush = current_count > 0
            && (current_count >= max_docs_per_batch
                || current_bytes + input_bytes > max_input_text_bytes_per_microbatch);
        if should_flush {
            spans.push((current_start, current_count));
            current_start = row_index;
            current_count = 0;
            current_bytes = 0;
        }
        current_count += 1;
        current_bytes += input_bytes;
        if current_count >= max_docs_per_batch {
            spans.push((current_start, current_count));
            current_start = row_index + 1;
            current_count = 0;
            current_bytes = 0;
        }
    }
    if current_count > 0 {
        spans.push((current_start, current_count));
    }
    spans
}

pub(crate) fn find_last_newline_run_end(chars: &[char], start: usize, end: usize) -> Option<usize> {
    let mut last_end = None;
    let mut index = start;
    while index < end {
        if chars[index] == '\n' {
            let mut run_end = index + 1;
            while run_end < end && chars[run_end] == '\n' {
                run_end += 1;
            }
            if run_end > start {
                last_end = Some(run_end);
            }
            index = run_end;
        } else {
            index += 1;
        }
    }
    last_end
}

pub(crate) fn find_last_double_newline_boundary_end(
    chars: &[char],
    start: usize,
    end: usize,
) -> Option<usize> {
    let mut last_end = None;
    for index in start..end {
        if chars[index] != '\n' {
            continue;
        }
        let mut cursor = index + 1;
        let mut match_end = None;
        while cursor < end && chars[cursor].is_whitespace() {
            if chars[cursor] == '\n' {
                let mut newline_end = cursor + 1;
                while newline_end < end && chars[newline_end] == '\n' {
                    newline_end += 1;
                }
                match_end = Some(newline_end);
                cursor = newline_end;
            } else {
                cursor += 1;
            }
        }
        if let Some(boundary_end) = match_end {
            if boundary_end > start {
                last_end = Some(boundary_end);
            }
        }
    }
    last_end
}

pub(crate) fn find_last_sentence_punct_boundary_end(
    chars: &[char],
    start: usize,
    end: usize,
) -> Option<usize> {
    let mut last_end = None;
    let mut index = start;
    while index < end {
        let ch = chars[index];
        if ch == '.' || ch == '!' || ch == '?' {
            let mut cursor = index + 1;
            while cursor < end && matches!(chars[cursor], ')' | '"' | '\'' | ']') {
                cursor += 1;
            }
            if cursor < end && chars[cursor].is_whitespace() {
                cursor += 1;
                while cursor < end && chars[cursor].is_whitespace() {
                    cursor += 1;
                }
                if cursor > start {
                    last_end = Some(cursor);
                }
                index = cursor;
                continue;
            }
        }
        index += 1;
    }
    last_end
}

pub(crate) fn find_last_whitespace_boundary_end(
    chars: &[char],
    start: usize,
    end: usize,
) -> Option<usize> {
    let mut last_end = None;
    let mut index = start;
    while index < end {
        if chars[index].is_whitespace() {
            let mut run_end = index + 1;
            while run_end < end && chars[run_end].is_whitespace() {
                run_end += 1;
            }
            if run_end > start {
                last_end = Some(run_end);
            }
            index = run_end;
        } else {
            index += 1;
        }
    }
    last_end
}

pub(crate) fn choose_sentence_chunk_end_impl(
    chars: &[char],
    start: usize,
    chunk_char_limit: usize,
) -> (usize, &'static str) {
    let text_len = chars.len();
    let max_end = start.saturating_add(chunk_char_limit).min(text_len);
    if max_end >= text_len {
        return (text_len, "end_of_text");
    }

    if let Some(split_end) = find_last_double_newline_boundary_end(chars, start, max_end) {
        return (split_end, "double_newline");
    }
    if let Some(split_end) = find_last_newline_run_end(chars, start, max_end) {
        return (split_end, "newline");
    }
    if let Some(split_end) = find_last_sentence_punct_boundary_end(chars, start, max_end) {
        return (split_end, "sentence_punct");
    }
    if let Some(split_end) = find_last_whitespace_boundary_end(chars, start, max_end) {
        return (split_end, "whitespace");
    }
    (max_end, "hard_limit_250k")
}

#[pyfunction]
pub(crate) fn choose_sentence_chunk_end(
    text: &str,
    start: usize,
    chunk_char_limit: usize,
) -> (usize, String) {
    let chars: Vec<char> = text.chars().collect();
    let (end, reason) = choose_sentence_chunk_end_impl(&chars, start, chunk_char_limit);
    (end, reason.to_string())
}

pub(crate) fn sentence_chunk_audit_warning(reason: &str) -> bool {
    matches!(reason, "whitespace" | "hard_limit_250k")
}

#[pyfunction]
pub(crate) fn expand_sentence_chunk_rows(
    py: Python<'_>,
    rows: &Bound<'_, PyAny>,
    chunk_char_limit: usize,
) -> PyResult<(Vec<PyObject>, Vec<PyObject>)> {
    let mut expanded_rows = Vec::new();
    let mut audit_rows = Vec::new();
    for row in rows.iter()? {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("sentence chunk row is not a dict"))?;
        let full_text_value = dict_required_py_object(py, dict, "full_text")?;
        let full_text = full_text_value.bind(py).str()?.to_str()?.to_string();
        let chars: Vec<char> = full_text.chars().collect();
        let original_char_count = chars.len();
        if original_char_count <= chunk_char_limit {
            let out = PyDict::new_bound(py);
            for (key, value) in dict.iter() {
                out.set_item(key, value)?;
            }
            expanded_rows.push(out.into_py(py));
            continue;
        }

        let mut row_chunk_count = 0usize;
        let audit_start = audit_rows.len();
        let mut start = 0usize;
        while start < original_char_count {
            let (mut end, mut split_reason) =
                choose_sentence_chunk_end_impl(&chars, start, chunk_char_limit);
            if end <= start {
                end = start
                    .saturating_add(chunk_char_limit)
                    .min(original_char_count);
                split_reason = "hard_limit_250k";
            }
            let chunk_text: String = chars[start..end].iter().collect();
            let chunk_row = PyDict::new_bound(py);
            for (key, value) in dict.iter() {
                chunk_row.set_item(key, value)?;
            }
            chunk_row.set_item("full_text", &chunk_text)?;
            expanded_rows.push(chunk_row.into_py(py));

            let audit = PyDict::new_bound(py);
            audit.set_item(
                "benchmark_row_id",
                dict_required_py_object(py, dict, "benchmark_row_id")?,
            )?;
            audit.set_item("doc_id", dict_required_py_object(py, dict, "doc_id")?)?;
            audit.set_item(
                "benchmark_item_code",
                dict_required_py_object(py, dict, "benchmark_item_code")?,
            )?;
            audit.set_item(
                "filing_year",
                dict_required_py_object(py, dict, "filing_year")?,
            )?;
            audit.set_item("original_char_count", original_char_count as i64)?;
            audit.set_item("chunk_index", row_chunk_count as i64)?;
            audit.set_item("chunk_char_count", (end - start) as i64)?;
            audit.set_item("split_start_char", start as i64)?;
            audit.set_item("split_end_char", end as i64)?;
            audit.set_item("split_reason", split_reason)?;
            audit.set_item("total_chunk_count", 0i64)?;
            audit.set_item(
                "warning_boundary_used",
                sentence_chunk_audit_warning(split_reason),
            )?;
            audit_rows.push(audit.into_py(py));

            start = end;
            row_chunk_count += 1;
        }
        for audit in audit_rows.iter().skip(audit_start) {
            let audit = audit
                .bind(py)
                .downcast::<PyDict>()
                .map_err(|_| PyValueError::new_err("sentence chunk audit row is not a dict"))?;
            audit.set_item("total_chunk_count", row_chunk_count as i64)?;
        }
    }
    Ok((expanded_rows, audit_rows))
}

pub(crate) fn cleaning_scope_body_hint(scope: &str, text: &str) -> bool {
    let lowered = text.to_ascii_lowercase();
    match scope {
        "item_7_mda" => {
            find_ascii_word(&lowered, "item", 0).is_some_and(|pos| {
                let rest = lowered[pos + 4..].trim_start();
                rest.strip_prefix('7')
                    .is_some_and(|tail| tail.chars().next().is_none_or(|ch| !ascii_word_char(ch)))
            }) || lowered.contains("management") && lowered.contains("discussion")
                || contains_ascii_word_or_phrase(&lowered, "results of operations")
        }
        "item_1a_risk_factors" => {
            contains_ascii_word_or_phrase(&lowered, "risk factor")
                || contains_ascii_word_or_phrase(&lowered, "risk factors")
                || lowered.contains("item 1a")
                || lowered.contains("item 1 a")
        }
        "item_1_business" => {
            contains_ascii_word_or_phrase(&lowered, "business")
                || lowered.contains("item 1 ")
                || lowered.ends_with("item 1")
        }
        _ => lowered.chars().any(|ch| ch.is_alphanumeric() || ch == '_'),
    }
}

#[pyfunction]
pub(crate) fn trim_early_toc_prefix_value(
    text: &str,
    text_scope: &str,
    toc_scan_char_window: usize,
    toc_min_matching_lines: usize,
) -> Option<(String, bool, usize)> {
    if !text.is_ascii() {
        return None;
    }
    if text.is_empty() || toc_scan_char_window == 0 {
        return Some((text.to_string(), false, 0));
    }
    let window_end = text.len().min(toc_scan_char_window);
    let mut offset = 0usize;
    let mut toc_offsets: Vec<usize> = Vec::new();
    for raw_line in text[..window_end].split_inclusive('\n') {
        let line_end = offset + raw_line.len();
        offset = line_end;
        if cleaning_toc_like_line_impl(raw_line.trim()) {
            toc_offsets.push(line_end);
        }
    }
    if offset < window_end {
        let raw_line = &text[offset..window_end];
        let line_end = offset + raw_line.len();
        if cleaning_toc_like_line_impl(raw_line.trim()) {
            toc_offsets.push(line_end);
        }
    }
    if toc_offsets.len() < toc_min_matching_lines {
        return Some((text.to_string(), false, 0));
    }
    let trim_end = *toc_offsets.iter().max().unwrap_or(&0);
    let remainder = text[trim_end..].trim_start();
    if count_tokens_impl(Some(remainder)) < 5 {
        return Some((text.to_string(), false, 0));
    }
    let hint_end = remainder.len().min(toc_scan_char_window);
    if !cleaning_scope_body_hint(text_scope, &remainder[..hint_end]) {
        return Some((text.to_string(), false, 0));
    }
    Some((remainder.to_string(), true, text.len() - remainder.len()))
}

#[pyfunction]
pub(crate) fn stable_digits_int_value(value: &str) -> Option<i64> {
    stable_digits_int_impl(value)
}

#[pyfunction]
pub(crate) fn stable_digits_int_values(values: &Bound<'_, PyAny>) -> PyResult<Vec<Option<i64>>> {
    let mut out = Vec::new();
    for value in values.iter()? {
        let value = value?;
        out.push(stable_digits_int_impl(value.str()?.to_str()?));
    }
    Ok(out)
}

#[pyfunction]
pub(crate) fn normalize_ascii_with_positions_value(text: &str) -> Option<(String, Vec<usize>)> {
    normalize_ascii_with_positions_impl(text)
}

#[pyfunction]
pub(crate) fn normalized_ascii_match_bounds_value(
    normalized_text: &str,
    normalized_to_original: Vec<i64>,
    query: &str,
) -> Option<(i64, i64)> {
    normalized_match_bounds_impl(normalized_text, &normalized_to_original, query)
}

pub(crate) fn normalize_finbert_label_name_impl(label: &str) -> Option<String> {
    let cleaned: String = label
        .trim()
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .map(|ch| ch.to_ascii_lowercase())
        .collect();
    match cleaned.as_str() {
        "negative" | "neg" | "bearish" => Some("negative".to_string()),
        "neutral" | "neu" => Some("neutral".to_string()),
        "positive" | "pos" | "bullish" => Some("positive".to_string()),
        _ => None,
    }
}

#[pyfunction]
pub(crate) fn normalize_finbert_label_name(label: &str) -> Option<String> {
    normalize_finbert_label_name_impl(label)
}

pub(crate) fn finbert_label_pairs_from_dict(
    dict: &Bound<'_, PyDict>,
    key_is_label: bool,
) -> Vec<(i64, String)> {
    let mut pairs: Vec<(i64, String)> = Vec::new();
    for (key, value) in dict.iter() {
        let parsed = if key_is_label {
            py_int_like_to_i64(&value).map(|index| (index, key.str()))
        } else {
            py_int_like_to_i64(&key).map(|index| (index, value.str()))
        };
        let Ok((index, label_result)) = parsed else {
            continue;
        };
        let Ok(label_obj) = label_result else {
            continue;
        };
        let Ok(label) = label_obj.to_str() else {
            continue;
        };
        pairs.push((index, label.to_string()));
    }
    pairs
}

#[pyfunction]
pub(crate) fn resolve_finbert_label_mapping(
    raw_id2label: &Bound<'_, PyAny>,
    raw_label2id: &Bound<'_, PyAny>,
) -> PyResult<BTreeMap<i64, String>> {
    let pairs = if let Ok(dict) = raw_id2label.downcast::<PyDict>() {
        finbert_label_pairs_from_dict(dict, false)
    } else if let Ok(dict) = raw_label2id.downcast::<PyDict>() {
        finbert_label_pairs_from_dict(dict, true)
    } else {
        Vec::new()
    };

    let mut sorted_pairs = pairs;
    sorted_pairs.sort_by(|left, right| left.0.cmp(&right.0));

    let mut normalized: BTreeMap<i64, String> = BTreeMap::new();
    let mut seen_labels: BTreeSet<String> = BTreeSet::new();
    for (index, label) in sorted_pairs {
        let Some(normalized_label) = normalize_finbert_label_name_impl(&label) else {
            continue;
        };
        if seen_labels.contains(&normalized_label) {
            return Err(PyValueError::new_err(format!(
                "Duplicate normalized FinBERT label in model config: {normalized_label:?}"
            )));
        }
        seen_labels.insert(normalized_label.clone());
        normalized.insert(index, normalized_label);
    }

    let expected: BTreeSet<String> = ["negative", "neutral", "positive"]
        .into_iter()
        .map(String::from)
        .collect();
    let observed: BTreeSet<String> = normalized.values().cloned().collect();
    if observed != expected {
        let rendered_observed = observed.into_iter().collect::<Vec<_>>();
        return Err(PyValueError::new_err(format!(
            "Could not normalize FinBERT labels from model config. Observed labels: {rendered_observed:?}"
        )));
    }
    Ok(normalized)
}

#[pyfunction]
pub(crate) fn finbert_median_value(values: Vec<f64>) -> PyResult<f64> {
    if values.is_empty() {
        return Ok(0.0);
    }
    if values.iter().any(|value| value.is_nan()) {
        return Err(PyValueError::new_err("median does not accept NaN"));
    }
    let mut ordered = values;
    ordered.sort_by(|left, right| left.total_cmp(right));
    let len = ordered.len();
    let mid = len / 2;
    if len % 2 == 1 {
        Ok(ordered[mid])
    } else {
        Ok((ordered[mid - 1] + ordered[mid]) / 2.0)
    }
}

#[pyfunction]
#[pyo3(signature = (rows, median_seconds, peak_vram_gb=None))]
pub(crate) fn finbert_stage_summary(
    rows: Vec<i64>,
    median_seconds: Vec<f64>,
    peak_vram_gb: Option<Vec<Option<f64>>>,
) -> PyResult<(i64, f64, Option<f64>, Option<f64>)> {
    let mut total_rows = 0_i64;
    for row_count in rows {
        total_rows = total_rows
            .checked_add(row_count)
            .ok_or_else(|| PyValueError::new_err("stage row count overflow"))?;
    }
    let total_seconds: f64 = median_seconds.iter().sum();
    let rows_per_second = if total_seconds != 0.0 {
        Some(total_rows as f64 / total_seconds)
    } else {
        None
    };
    let mut peak_vram: Option<f64> = None;
    if let Some(values) = peak_vram_gb {
        for value in values.into_iter().flatten() {
            match peak_vram {
                Some(current) if value > current => peak_vram = Some(value),
                None => peak_vram = Some(value),
                _ => {}
            }
        }
    }
    Ok((total_rows, total_seconds, rows_per_second, peak_vram))
}

pub(crate) fn median_finite_values(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|left, right| left.total_cmp(right));
    let len = values.len();
    let mid = len / 2;
    if len % 2 == 1 {
        Some(values[mid])
    } else {
        Some((values[mid - 1] + values[mid]) / 2.0)
    }
}

#[pyfunction]
pub(crate) fn finbert_staged_bucket_summary_rows(
    buckets: Vec<Option<String>>,
    token_counts: Vec<Option<f64>>,
    filing_year: i64,
) -> PyResult<Vec<(i64, String, i64, Option<f64>, Option<f64>)>> {
    if buckets.len() != token_counts.len() {
        return Err(PyValueError::new_err(
            "bucket and token-count inputs must have equal length",
        ));
    }
    let mut rows_by_bucket: [i64; 3] = [0, 0, 0];
    let mut token_values_by_bucket: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for (bucket, token_count) in buckets.into_iter().zip(token_counts.into_iter()) {
        let Some(bucket) = bucket else {
            continue;
        };
        let bucket_index = match bucket.as_str() {
            "short" => 0,
            "medium" => 1,
            "long" => 2,
            _ => continue,
        };
        rows_by_bucket[bucket_index] = rows_by_bucket[bucket_index]
            .checked_add(1)
            .ok_or_else(|| PyValueError::new_err("bucket row count overflow"))?;
        if let Some(value) = token_count {
            if !value.is_finite() {
                return Err(PyValueError::new_err("non-finite token count"));
            }
            token_values_by_bucket[bucket_index].push(value);
        }
    }
    let bucket_names = ["short", "medium", "long"];
    let mut rows: Vec<(i64, String, i64, Option<f64>, Option<f64>)> = Vec::with_capacity(3);
    for (index, bucket_name) in bucket_names.iter().enumerate() {
        let values = &mut token_values_by_bucket[index];
        let token_count_mean = if values.is_empty() {
            None
        } else {
            Some(values.iter().sum::<f64>() / values.len() as f64)
        };
        let token_count_median = median_finite_values(values);
        rows.push((
            filing_year,
            (*bucket_name).to_string(),
            rows_by_bucket[index],
            token_count_mean,
            token_count_median,
        ));
    }
    Ok(rows)
}

#[pyfunction]
pub(crate) fn finbert_coverage_report(
    backbone_doc_ids: Vec<String>,
    feature_doc_ids: Vec<Option<String>>,
    feature_item_codes: Vec<Option<String>>,
) -> PyResult<(
    Vec<(String, bool, bool, bool, bool)>,
    (i64, i64, i64, i64, i64),
)> {
    if feature_doc_ids.len() != feature_item_codes.len() {
        return Err(PyValueError::new_err(
            "feature doc-id and item-code vectors must have the same length",
        ));
    }

    let mut flags_by_doc: HashMap<String, (bool, bool, bool, bool)> = HashMap::new();
    for (doc_id, item_code) in feature_doc_ids
        .into_iter()
        .zip(feature_item_codes.into_iter())
    {
        let Some(doc_id) = doc_id else {
            continue;
        };
        let flags = flags_by_doc
            .entry(doc_id)
            .or_insert((false, false, false, false));
        flags.0 = true;
        match item_code.as_deref() {
            Some("item_1") => flags.1 = true,
            Some("item_1a") => flags.2 = true,
            Some("item_7") => flags.3 = true,
            _ => {}
        }
    }

    let mut covered_doc_count = 0_i64;
    let mut covered_item_1_doc_count = 0_i64;
    let mut covered_item_1a_doc_count = 0_i64;
    let mut covered_item_7_doc_count = 0_i64;
    let mut rows = Vec::with_capacity(backbone_doc_ids.len());
    for doc_id in backbone_doc_ids {
        let (has_finbert_features, has_item_1, has_item_1a, has_item_7) = flags_by_doc
            .get(&doc_id)
            .copied()
            .unwrap_or((false, false, false, false));
        if has_finbert_features {
            covered_doc_count += 1;
        }
        if has_item_1 {
            covered_item_1_doc_count += 1;
        }
        if has_item_1a {
            covered_item_1a_doc_count += 1;
        }
        if has_item_7 {
            covered_item_7_doc_count += 1;
        }
        rows.push((
            doc_id,
            has_finbert_features,
            has_item_1,
            has_item_1a,
            has_item_7,
        ));
    }
    let backbone_doc_count = i64::try_from(rows.len())
        .map_err(|_| PyValueError::new_err("backbone doc count overflow"))?;
    Ok((
        rows,
        (
            backbone_doc_count,
            covered_doc_count,
            covered_item_1_doc_count,
            covered_item_1a_doc_count,
            covered_item_7_doc_count,
        ),
    ))
}

#[pyfunction]
pub(crate) fn finbert_split_metrics(
    benchmark_row_ids: Vec<Option<String>>,
    warning_boundary_used: Vec<Option<bool>>,
    original_char_counts: Vec<Option<i64>>,
) -> PyResult<(i64, i64, Option<i64>)> {
    let row_count = benchmark_row_ids.len();
    if warning_boundary_used.len() != row_count || original_char_counts.len() != row_count {
        return Err(PyValueError::new_err(
            "split metric input vectors must have the same length",
        ));
    }

    let mut chunked_ids: HashSet<Option<String>> = HashSet::new();
    let mut warning_ids: HashSet<Option<String>> = HashSet::new();
    let mut max_original_char_count: Option<i64> = None;

    for ((benchmark_row_id, warning_used), original_char_count) in benchmark_row_ids
        .into_iter()
        .zip(warning_boundary_used.into_iter())
        .zip(original_char_counts.into_iter())
    {
        chunked_ids.insert(benchmark_row_id.clone());
        if warning_used.unwrap_or(false) {
            warning_ids.insert(benchmark_row_id);
        }
        if let Some(value) = original_char_count {
            max_original_char_count = Some(match max_original_char_count {
                Some(current) => current.max(value),
                None => value,
            });
        }
    }

    let chunked_section_rows = i64::try_from(chunked_ids.len())
        .map_err(|_| PyValueError::new_err("chunked section row count overflow"))?;
    let warning_split_rows = i64::try_from(warning_ids.len())
        .map_err(|_| PyValueError::new_err("warning split row count overflow"))?;
    Ok((
        chunked_section_rows,
        warning_split_rows,
        max_original_char_count,
    ))
}

#[pyfunction]
pub(crate) fn finbert_fallback_split_warning_payload(
    benchmark_row_ids: Vec<Option<String>>,
    warning_boundary_used: Vec<Option<bool>>,
    split_reasons: Vec<Option<String>>,
) -> PyResult<(i64, Vec<(Option<String>, i64)>)> {
    let row_count = benchmark_row_ids.len();
    if warning_boundary_used.len() != row_count || split_reasons.len() != row_count {
        return Err(PyValueError::new_err(
            "fallback split warning input vectors must have the same length",
        ));
    }

    let mut affected_ids: HashSet<Option<String>> = HashSet::new();
    let mut reason_counts: BTreeMap<Option<String>, i64> = BTreeMap::new();
    for ((benchmark_row_id, warning_used), split_reason) in benchmark_row_ids
        .into_iter()
        .zip(warning_boundary_used.into_iter())
        .zip(split_reasons.into_iter())
    {
        if !warning_used.unwrap_or(false) {
            continue;
        }
        affected_ids.insert(benchmark_row_id);
        let count = reason_counts.entry(split_reason).or_insert(0);
        *count = count
            .checked_add(1)
            .ok_or_else(|| PyValueError::new_err("fallback split reason count overflow"))?;
    }

    let affected_row_count = i64::try_from(affected_ids.len())
        .map_err(|_| PyValueError::new_err("affected split row count overflow"))?;
    Ok((affected_row_count, reason_counts.into_iter().collect()))
}

#[pyfunction]
pub(crate) fn finbert_token_bucket_counts(
    buckets: Vec<Option<String>>,
) -> PyResult<(i64, i64, i64)> {
    let mut short_count = 0_i64;
    let mut medium_count = 0_i64;
    let mut long_count = 0_i64;
    for bucket in buckets.into_iter().flatten() {
        let target = match bucket.as_str() {
            "short" => &mut short_count,
            "medium" => &mut medium_count,
            "long" => &mut long_count,
            _ => continue,
        };
        *target = target
            .checked_add(1)
            .ok_or_else(|| PyValueError::new_err("token bucket count overflow"))?;
    }
    Ok((short_count, medium_count, long_count))
}

#[pyfunction]
pub(crate) fn finbert_preprocessing_manifest_counts(
    py: Python<'_>,
    summary_rows: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let mut year_count = 0_i64;
    let mut processed_year_count = 0_i64;
    let mut reused_year_count = 0_i64;
    let mut sentence_rows = 0_i64;
    let mut oversize_section_rows = 0_i64;
    let mut chunked_section_rows = 0_i64;
    let mut warning_split_rows = 0_i64;
    let mut cleaned_scope_rows = 0_i64;
    let mut cleaning_dropped_rows = 0_i64;
    let mut cleaning_flagged_rows = 0_i64;

    for row in summary_rows.iter()? {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("summary row is not a dict"))?;
        year_count += 1;
        if let Some(status) = dict.get_item("status")? {
            let status_text = status.str()?;
            let status = status_text.to_str()?;
            if status == "processed" {
                processed_year_count += 1;
            } else if status == "reused_existing" {
                reused_year_count += 1;
            }
        }
        for (key, total) in [
            ("sentence_rows", &mut sentence_rows),
            ("oversize_section_rows", &mut oversize_section_rows),
            ("chunked_section_rows", &mut chunked_section_rows),
            ("warning_split_rows", &mut warning_split_rows),
            ("cleaned_scope_rows", &mut cleaned_scope_rows),
            ("cleaning_dropped_rows", &mut cleaning_dropped_rows),
            ("cleaning_flagged_rows", &mut cleaning_flagged_rows),
        ] {
            if let Some(value) = dict.get_item(key)? {
                if !value.is_none() {
                    *total += py_int_like_to_i64(&value)?;
                }
            }
        }
    }

    let out = PyDict::new_bound(py);
    out.set_item("year_count", year_count)?;
    out.set_item("processed_year_count", processed_year_count)?;
    out.set_item("reused_year_count", reused_year_count)?;
    out.set_item("sentence_rows", sentence_rows)?;
    out.set_item("oversize_section_rows", oversize_section_rows)?;
    out.set_item("chunked_section_rows", chunked_section_rows)?;
    out.set_item("warning_split_rows", warning_split_rows)?;
    out.set_item("cleaned_scope_rows", cleaned_scope_rows)?;
    out.set_item("cleaning_dropped_rows", cleaning_dropped_rows)?;
    out.set_item("cleaning_flagged_rows", cleaning_flagged_rows)?;
    Ok(out.into_py(py))
}

#[pyfunction]
pub(crate) fn finbert_preprocessing_manifest_count_columns(
    py: Python<'_>,
    column_names: Vec<String>,
    column_values: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let mut columns: Vec<Vec<PyObject>> = Vec::with_capacity(column_names.len());
    let mut row_count: Option<usize> = None;
    for values in column_values.iter()? {
        let values = values?;
        let mut column: Vec<PyObject> = Vec::new();
        for value in values.iter()? {
            column.push(value?.clone().into_py(py));
        }
        match row_count {
            Some(expected) if column.len() != expected => {
                return Err(PyValueError::new_err(
                    "FinBERT preprocessing manifest count column lengths must match",
                ))
            }
            None => row_count = Some(column.len()),
            _ => {}
        }
        columns.push(column);
    }
    if columns.len() != column_names.len() {
        return Err(PyValueError::new_err(
            "FinBERT preprocessing manifest count column name/value count mismatch",
        ));
    }
    let row_count = row_count.unwrap_or(0);
    let column_index = column_index_by_name(&column_names);
    let mut processed_year_count = 0_i64;
    let mut reused_year_count = 0_i64;
    let mut sentence_rows = 0_i64;
    let mut oversize_section_rows = 0_i64;
    let mut chunked_section_rows = 0_i64;
    let mut warning_split_rows = 0_i64;
    let mut cleaned_scope_rows = 0_i64;
    let mut cleaning_dropped_rows = 0_i64;
    let mut cleaning_flagged_rows = 0_i64;

    let sum_column = |column_name: &str, total: &mut i64| -> PyResult<()> {
        let Some(&column_idx) = column_index.get(column_name) else {
            return Ok(());
        };
        for value in columns[column_idx].iter() {
            let value = value.bind(py);
            if !value.is_none() {
                *total += py_int_like_to_i64(value)?;
            }
        }
        Ok(())
    };

    if let Some(&status_idx) = column_index.get("status") {
        for value in columns[status_idx].iter() {
            let value = value.bind(py);
            if value.is_none() {
                continue;
            }
            let status = value.str()?;
            match status.to_str()? {
                "processed" => processed_year_count += 1,
                "reused_existing" => reused_year_count += 1,
                _ => {}
            }
        }
    }
    sum_column("sentence_rows", &mut sentence_rows)?;
    sum_column("oversize_section_rows", &mut oversize_section_rows)?;
    sum_column("chunked_section_rows", &mut chunked_section_rows)?;
    sum_column("warning_split_rows", &mut warning_split_rows)?;
    sum_column("cleaned_scope_rows", &mut cleaned_scope_rows)?;
    sum_column("cleaning_dropped_rows", &mut cleaning_dropped_rows)?;
    sum_column("cleaning_flagged_rows", &mut cleaning_flagged_rows)?;

    let out = PyDict::new_bound(py);
    out.set_item(
        "year_count",
        i64::try_from(row_count)
            .map_err(|_| PyValueError::new_err("manifest row count overflow"))?,
    )?;
    out.set_item("processed_year_count", processed_year_count)?;
    out.set_item("reused_year_count", reused_year_count)?;
    out.set_item("sentence_rows", sentence_rows)?;
    out.set_item("oversize_section_rows", oversize_section_rows)?;
    out.set_item("chunked_section_rows", chunked_section_rows)?;
    out.set_item("warning_split_rows", warning_split_rows)?;
    out.set_item("cleaned_scope_rows", cleaned_scope_rows)?;
    out.set_item("cleaning_dropped_rows", cleaning_dropped_rows)?;
    out.set_item("cleaning_flagged_rows", cleaning_flagged_rows)?;
    Ok(out.into_py(py))
}

#[pyfunction]
pub(crate) fn finbert_bucket_value_for_name(
    bucket: &str,
    short_value: i64,
    medium_value: i64,
    long_value: i64,
) -> PyResult<i64> {
    match bucket {
        "short" => Ok(short_value),
        "medium" => Ok(medium_value),
        "long" => Ok(long_value),
        _ => Err(PyValueError::new_err(format!("Unknown bucket: {bucket:?}"))),
    }
}

#[pyfunction]
pub(crate) fn round_up_to_multiple_value(value: i64, multiple: i64) -> PyResult<i64> {
    if multiple <= 0 {
        return Err(PyValueError::new_err(
            "multiple must be a positive integer.",
        ));
    }
    if value <= 0 {
        return Ok(multiple);
    }
    Ok(((value + multiple - 1) / multiple) * multiple)
}

#[pyfunction]
#[pyo3(signature = (
    bucket,
    current_edge,
    lower_bound,
    token_target_quantile,
    round_to,
    safety_margin_tokens,
    policy
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn finbert_recommend_bucket_edge(
    bucket: &str,
    current_edge: i64,
    lower_bound: i64,
    token_target_quantile: Option<f64>,
    round_to: i64,
    safety_margin_tokens: i64,
    policy: &str,
) -> PyResult<(i64, &'static str)> {
    if policy == "keep_current" {
        return Ok((current_edge, "kept_current"));
    }
    let Some(target_quantile) = token_target_quantile else {
        return Ok((current_edge, "no_rows"));
    };
    if !target_quantile.is_finite() {
        return Err(PyValueError::new_err(
            "cannot convert non-finite token target quantile to integer",
        ));
    }
    let rounded_target = target_quantile.ceil() as i64 + safety_margin_tokens;
    let mut candidate = round_up_to_multiple_value(rounded_target, round_to)?;
    candidate = candidate.min(current_edge).max(lower_bound);
    if bucket == "medium" && candidate < lower_bound {
        candidate = lower_bound;
    }
    Ok((candidate, "target_quantile"))
}

pub(crate) fn set_finbert_bucket_length_summary_empty_values(
    out: &Bound<'_, PyDict>,
    bucket_col: &str,
    bucket: &str,
) -> PyResult<()> {
    out.set_item(bucket_col, bucket)?;
    out.set_item("sentence_rows", 0i64)?;
    out.set_item("doc_count", 0i64)?;
    out.set_item("token_min", Option::<i64>::None)?;
    out.set_item("token_median", Option::<f64>::None)?;
    out.set_item("token_max", Option::<i64>::None)?;
    out.set_item("token_target_quantile", Option::<f64>::None)?;
    out.set_item("token_p90", Option::<f64>::None)?;
    out.set_item("token_p95", Option::<f64>::None)?;
    out.set_item("token_p99", Option::<f64>::None)?;
    out.set_item("token_p995", Option::<f64>::None)?;
    out.set_item("token_p999", Option::<f64>::None)?;
    Ok(())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn finbert_bucket_length_summary_rows(
    py: Python<'_>,
    grouped_rows: &Bound<'_, PyAny>,
    bucket_col: &str,
    short_edge: i64,
    medium_edge: i64,
    long_edge: i64,
    short_length: i64,
    medium_length: i64,
    long_length: i64,
) -> PyResult<Vec<PyObject>> {
    let mut by_bucket: HashMap<String, PyObject> = HashMap::new();
    for row in grouped_rows.iter()? {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("FinBERT bucket summary row is not a dict"))?;
        let bucket = dict_required_string(dict, bucket_col)?;
        by_bucket.insert(bucket, row.into_py(py));
    }

    let mut out_rows = Vec::new();
    for (bucket, edge, length) in [
        ("short", short_edge, short_length),
        ("medium", medium_edge, medium_length),
        ("long", long_edge, long_length),
    ] {
        let out = PyDict::new_bound(py);
        if let Some(row) = by_bucket.get(bucket) {
            let dict = row.bind(py).downcast::<PyDict>().map_err(|_| {
                PyValueError::new_err("FinBERT bucket summary cached row is not a dict")
            })?;
            for (key, value) in dict.iter() {
                out.set_item(key, value)?;
            }
        } else {
            set_finbert_bucket_length_summary_empty_values(&out, bucket_col, bucket)?;
        }
        out.set_item("current_edge_upper_bound", edge)?;
        out.set_item("current_max_length", length)?;
        out_rows.push(out.into_py(py));
    }
    Ok(out_rows)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn finbert_bucket_length_summary_columns(
    py: Python<'_>,
    column_names: Vec<String>,
    column_values: &Bound<'_, PyAny>,
    bucket_col: &str,
    short_edge: i64,
    medium_edge: i64,
    long_edge: i64,
    short_length: i64,
    medium_length: i64,
    long_length: i64,
) -> PyResult<Vec<PyObject>> {
    let label = "FinBERT bucket length summary";
    let (columns, row_count) =
        collect_pyobject_column_values(py, &column_names, column_values, label)?;
    let column_index = column_index_by_name(&column_names);
    let bucket_idx = required_named_column_index(&column_index, label, bucket_col)?;

    let mut by_bucket: HashMap<String, usize> = HashMap::new();
    for row_idx in 0..row_count {
        let value = columns[bucket_idx][row_idx].bind(py);
        if value.is_none() {
            return Err(PyValueError::new_err(format!(
                "null required key: {bucket_col}"
            )));
        }
        by_bucket.insert(value.str()?.to_str()?.to_string(), row_idx);
    }

    let mut out_rows = Vec::new();
    for (bucket, edge, length) in [
        ("short", short_edge, short_length),
        ("medium", medium_edge, medium_length),
        ("long", long_edge, long_length),
    ] {
        let out = PyDict::new_bound(py);
        if let Some(row_idx) = by_bucket.get(bucket) {
            for (column_idx, column_name) in column_names.iter().enumerate() {
                out.set_item(column_name, columns[column_idx][*row_idx].bind(py))?;
            }
        } else {
            set_finbert_bucket_length_summary_empty_values(&out, bucket_col, bucket)?;
        }
        out.set_item("current_edge_upper_bound", edge)?;
        out.set_item("current_max_length", length)?;
        out_rows.push(out.into_py(py));
    }
    Ok(out_rows)
}

#[pyfunction]
pub(crate) fn parse_device_index_value(device: &str) -> i64 {
    let Some((_, index)) = device.split_once(':') else {
        return 0;
    };
    index.parse::<i64>().unwrap_or(0)
}

#[pyfunction]
pub(crate) fn resolve_finbert_amp_dtype_name(
    use_autocast: bool,
    cuda_available: bool,
    device: &str,
    amp_dtype: &str,
) -> PyResult<Option<String>> {
    if !use_autocast || !device.starts_with("cuda") || !cuda_available {
        return Ok(None);
    }
    match amp_dtype {
        "auto" => Ok(Some("float16".to_string())),
        "float16" | "bfloat16" => Ok(Some(amp_dtype.to_string())),
        _ => Err(PyValueError::new_err(format!(
            "Unsupported amp_dtype: {amp_dtype:?}"
        ))),
    }
}

pub(crate) fn html_audit_safe_slug_impl(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    let mut in_replacement = false;
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
            out.push(ch);
            in_replacement = false;
        } else if !in_replacement {
            out.push('_');
            in_replacement = true;
        }
    }
    let trimmed = out.trim_matches('_');
    if trimmed.is_empty() {
        "unknown".to_string()
    } else {
        trimmed.to_string()
    }
}

#[pyfunction]
pub(crate) fn html_audit_safe_slug_value(value: &str) -> String {
    html_audit_safe_slug_impl(value)
}

#[pyfunction]
#[pyo3(signature = (value=None))]
pub(crate) fn html_audit_parse_bool_value(value: Option<&Bound<'_, PyAny>>) -> PyResult<bool> {
    let Some(value) = value else {
        return Ok(false);
    };
    if value.is_none() {
        return Ok(false);
    }
    if value.is_instance_of::<PyBool>() {
        return value.extract::<bool>();
    }
    let rendered = value.str()?;
    let normalized = rendered.to_str()?.trim().to_ascii_lowercase();
    Ok(matches!(
        normalized.as_str(),
        "1" | "true" | "yes" | "y" | "t"
    ))
}

#[pyfunction]
pub(crate) fn html_audit_filing_status_value(row: &Bound<'_, PyDict>) -> PyResult<&'static str> {
    let any_fail = html_audit_parse_bool_value(row.get_item("any_fail")?.as_ref())?;
    let has_exclusion = match row.get_item("filing_exclusion_reason")? {
        Some(value) if !value.is_none() => {
            let rendered = value.str()?;
            !rendered.to_str()?.trim().is_empty()
        }
        _ => false,
    };
    if any_fail || has_exclusion {
        return Ok("fail");
    }
    if html_audit_parse_bool_value(row.get_item("any_warn")?.as_ref())? {
        return Ok("warning");
    }
    Ok("pass")
}

#[pyfunction]
#[pyo3(signature = (value=None, default=0))]
pub(crate) fn html_audit_parse_int_value(
    value: Option<&Bound<'_, PyAny>>,
    default: i64,
) -> PyResult<i64> {
    let Some(value) = value else {
        return Ok(default);
    };
    if value.is_none() {
        return Ok(default);
    }
    let rendered = value.str()?;
    let text = rendered.to_str()?.trim();
    if text.contains('_') {
        return Err(PyValueError::new_err("underscore-containing integer text"));
    }
    Ok(text.parse::<i64>().unwrap_or(default))
}

#[pyfunction]
pub(crate) fn html_audit_part_rank_value(value: &str) -> i64 {
    match value.trim().to_uppercase().as_str() {
        "I" => 1,
        "II" => 2,
        "III" => 3,
        "IV" => 4,
        _ => 99,
    }
}

#[pyfunction]
pub(crate) fn html_audit_item_id_sort_key_value(value: &str) -> PyResult<(i64, String)> {
    let cleaned = value.trim().to_uppercase();
    if !cleaned.is_ascii() {
        return Err(PyValueError::new_err("non-ascii item id"));
    }
    let digit_len = cleaned
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .map(char::len_utf8)
        .sum::<usize>();
    if digit_len == 0 {
        return Ok((999, cleaned));
    }
    let (digits, suffix) = cleaned.split_at(digit_len);
    if suffix.is_empty() || (suffix.len() == 1 && suffix.chars().all(|ch| ch.is_ascii_uppercase()))
    {
        let number = digits
            .parse::<i64>()
            .map_err(|_| PyValueError::new_err("item id number out of range"))?;
        return Ok((number, suffix.to_string()));
    }
    Ok((999, cleaned))
}

#[pyfunction]
pub(crate) fn html_audit_quartile_edges_value(values: Vec<i64>) -> (i64, i64, i64) {
    if values.is_empty() {
        return (0, 0, 0);
    }
    let mut ordered = values;
    ordered.sort_unstable();
    let n = ordered.len();
    let q1 = ordered[(0.25 * (n.saturating_sub(1)) as f64) as usize];
    let q2 = ordered[(0.50 * (n.saturating_sub(1)) as f64) as usize];
    let q3 = ordered[(0.75 * (n.saturating_sub(1)) as f64) as usize];
    (q1, q2, q3)
}

#[pyfunction]
pub(crate) fn html_audit_quartile_bucket_value(value: i64, edges: (i64, i64, i64)) -> &'static str {
    let (q1, q2, q3) = edges;
    if value <= q1 {
        "Q1"
    } else if value <= q2 {
        "Q2"
    } else if value <= q3 {
        "Q3"
    } else {
        "Q4"
    }
}

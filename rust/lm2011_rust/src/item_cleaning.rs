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
use crate::sentence_quality_api::*;

pub(crate) fn item_cleaning_text_scope_enabled(
    table_like_target_text_scopes: &Bound<'_, PyAny>,
    text_scope: &str,
) -> PyResult<bool> {
    if table_like_target_text_scopes.is_none() {
        return Ok(true);
    }
    for entry in table_like_target_text_scopes.iter()? {
        let entry = entry?;
        let rendered = entry.str()?;
        if rendered.to_str()? == text_scope {
            return Ok(true);
        }
    }
    Ok(false)
}

pub(crate) fn item_cleaning_effectively_non_body_text(text: &str) -> bool {
    let stripped = text.trim();
    !stripped.is_empty() && count_tokens_impl(Some(stripped)) == 0
}

#[pyfunction]
pub(crate) fn item_cleaning_clean_text_value(
    py: Python<'_>,
    text: &Bound<'_, PyAny>,
    text_scope: &str,
    enabled: bool,
    drop_page_markers: bool,
    drop_report_headers: bool,
    drop_structural_tags: bool,
    trim_early_toc_prefix: bool,
    truncate_item_aware_tail_bleed: bool,
    drop_reference_only_stubs: bool,
    drop_table_like_lines: bool,
    table_like_min_consecutive_lines: usize,
    table_like_drop_header_context: bool,
    table_like_allow_single_line_with_header: bool,
    table_like_target_text_scopes: &Bound<'_, PyAny>,
    toc_scan_char_window: usize,
    toc_min_matching_lines: usize,
    tail_scan_fraction: f64,
    reference_stub_max_char_count: usize,
) -> PyResult<Option<PyObject>> {
    let normalized = if text.is_none() {
        normalize_newlines_impl(None)
    } else {
        let rendered = text.str()?;
        normalize_newlines_impl(Some(rendered.to_str()?))
    };
    if !normalized.is_ascii() {
        return Ok(None);
    }

    let out = PyDict::new_bound(py);
    if !enabled {
        let cleaned = collapse_blank_runs_impl(&normalized);
        out.set_item("cleaned_text", cleaned.as_str())?;
        out.set_item("page_marker_lines_removed", 0_i64)?;
        out.set_item("report_header_footer_lines_removed", 0_i64)?;
        out.set_item("structural_tag_lines_removed", 0_i64)?;
        out.set_item("table_like_lines_removed", 0_i64)?;
        out.set_item("toc_prefix_trimmed", false)?;
        out.set_item("toc_prefix_trimmed_char_count", 0_i64)?;
        out.set_item("tail_truncated", false)?;
        out.set_item("tail_truncated_char_count", 0_i64)?;
        out.set_item("reference_only_stub", false)?;
        out.set_item(
            "effectively_non_body_text",
            item_cleaning_effectively_non_body_text(&cleaned),
        )?;
        return Ok(Some(out.into_py(py)));
    }

    let (without_lines, line_counts) = cleaning_remove_layout_lines_impl(
        &normalized,
        drop_page_markers,
        drop_report_headers,
        drop_structural_tags,
    );

    let (without_toc, toc_trimmed, toc_removed) =
        if trim_early_toc_prefix && !without_lines.is_empty() && toc_scan_char_window > 0 {
            let Some((cleaned, trimmed, removed)) = trim_early_toc_prefix_value(
                &without_lines,
                text_scope,
                toc_scan_char_window,
                toc_min_matching_lines,
            ) else {
                return Ok(None);
            };
            (cleaned, trimmed, removed)
        } else {
            (without_lines, false, 0)
        };

    let (without_tail, tail_truncated, tail_removed) =
        if truncate_item_aware_tail_bleed && !without_toc.is_empty() {
            if let Some(best_start) =
                cleaning_tail_bleed_start_impl(&without_toc, text_scope, tail_scan_fraction)
            {
                let start_byte = char_to_byte_index(&without_toc, best_start);
                let truncated = without_toc[..start_byte].trim_end().to_string();
                let removed = without_toc
                    .chars()
                    .count()
                    .saturating_sub(truncated.chars().count());
                (truncated, true, removed)
            } else {
                (without_toc, false, 0)
            }
        } else {
            (without_toc, false, 0)
        };

    let reference_only_stub = if drop_reference_only_stubs {
        cleaning_is_reference_only_stub_impl(
            &collapse_blank_runs_impl(&without_tail),
            reference_stub_max_char_count,
        )
    } else {
        false
    };

    let (without_tables, table_counts) = if drop_table_like_lines {
        if item_cleaning_text_scope_enabled(table_like_target_text_scopes, text_scope)? {
            cleaning_remove_table_like_lines_impl(
                &without_tail,
                table_like_min_consecutive_lines,
                table_like_allow_single_line_with_header,
                table_like_drop_header_context,
            )
        } else {
            (rstrip_lines_join_impl(&without_tail), BTreeMap::new())
        }
    } else {
        (rstrip_lines_join_impl(&without_tail), BTreeMap::new())
    };

    let cleaned = collapse_blank_runs_impl(&without_tables);
    out.set_item("cleaned_text", cleaned.as_str())?;
    out.set_item(
        "page_marker_lines_removed",
        *line_counts.get("page_marker").unwrap_or(&0_i64),
    )?;
    out.set_item(
        "report_header_footer_lines_removed",
        *line_counts.get("report_header_footer").unwrap_or(&0_i64),
    )?;
    out.set_item(
        "structural_tag_lines_removed",
        *line_counts.get("structural_tag").unwrap_or(&0_i64),
    )?;
    out.set_item(
        "table_like_lines_removed",
        *table_counts.get("table_like").unwrap_or(&0_i64),
    )?;
    out.set_item("toc_prefix_trimmed", toc_trimmed)?;
    out.set_item("toc_prefix_trimmed_char_count", toc_removed as i64)?;
    out.set_item("tail_truncated", tail_truncated)?;
    out.set_item("tail_truncated_char_count", tail_removed as i64)?;
    out.set_item("reference_only_stub", reference_only_stub)?;
    out.set_item(
        "effectively_non_body_text",
        item_cleaning_effectively_non_body_text(&cleaned),
    )?;
    Ok(Some(out.into_py(py)))
}

#[pyfunction]
pub(crate) fn item_cleaning_prepare_rows_value(
    py: Python<'_>,
    rows: &Bound<'_, PyAny>,
    enabled: bool,
    drop_page_markers: bool,
    drop_report_headers: bool,
    drop_structural_tags: bool,
    trim_early_toc_prefix: bool,
    truncate_item_aware_tail_bleed: bool,
    drop_reference_only_stubs: bool,
    drop_table_like_lines: bool,
    table_like_min_consecutive_lines: usize,
    table_like_drop_header_context: bool,
    table_like_allow_single_line_with_header: bool,
    table_like_target_text_scopes: &Bound<'_, PyAny>,
    toc_scan_char_window: usize,
    toc_min_matching_lines: usize,
    tail_scan_fraction: f64,
    reference_stub_max_char_count: usize,
) -> PyResult<Option<Vec<PyObject>>> {
    let mut prepared_rows: Vec<PyObject> = Vec::new();
    for row in rows.iter()? {
        let row = row?;
        let row_dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("item cleaning section row is not a dict"))?;
        let benchmark_item_code = dict_raw_string(row_dict, "benchmark_item_code")?;
        let text_scope = benchmark_item_code_to_text_scope_impl(benchmark_item_code.as_deref())
            .unwrap_or_else(|| benchmark_item_code.clone().unwrap_or_default());
        let raw_text = dict_raw_string(row_dict, "full_text")?;
        let original_text = normalize_newlines_impl(raw_text.as_deref());
        if !original_text.is_ascii() {
            return Ok(None);
        }
        let text_obj = match raw_text.as_deref() {
            Some(value) => PyString::new_bound(py, value).into_py(py),
            None => py.None(),
        };
        let Some(cleaning_payload_obj) = item_cleaning_clean_text_value(
            py,
            text_obj.bind(py),
            &text_scope,
            enabled,
            drop_page_markers,
            drop_report_headers,
            drop_structural_tags,
            trim_early_toc_prefix,
            truncate_item_aware_tail_bleed,
            drop_reference_only_stubs,
            drop_table_like_lines,
            table_like_min_consecutive_lines,
            table_like_drop_header_context,
            table_like_allow_single_line_with_header,
            table_like_target_text_scopes,
            toc_scan_char_window,
            toc_min_matching_lines,
            tail_scan_fraction,
            reference_stub_max_char_count,
        )?
        else {
            return Ok(None);
        };
        let cleaning_payload = cleaning_payload_obj
            .bind(py)
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("item cleaning payload is not a dict"))?;
        let cleaned_text = dict_required_string(cleaning_payload, "cleaned_text")?;
        let original_char_count = original_text.chars().count() as i64;
        let cleaned_char_count = cleaned_text.chars().count() as i64;
        let removed_char_count = original_char_count.saturating_sub(cleaned_char_count);
        let removal_ratio = if original_char_count > 0 {
            removed_char_count as f64 / original_char_count as f64
        } else {
            0.0
        };

        let out = PyDict::new_bound(py);
        out.set_item("row", row_dict)?;
        out.set_item("text_scope", text_scope.as_str())?;
        out.set_item("original_text", original_text.as_str())?;
        for key in [
            "cleaned_text",
            "page_marker_lines_removed",
            "report_header_footer_lines_removed",
            "structural_tag_lines_removed",
            "table_like_lines_removed",
            "toc_prefix_trimmed",
            "toc_prefix_trimmed_char_count",
            "tail_truncated",
            "tail_truncated_char_count",
            "reference_only_stub",
            "effectively_non_body_text",
        ] {
            out.set_item(key, dict_required_py_object(py, cleaning_payload, key)?)?;
        }
        out.set_item("original_char_count", original_char_count)?;
        out.set_item("cleaned_char_count", cleaned_char_count)?;
        out.set_item("removed_char_count", removed_char_count)?;
        out.set_item("removal_ratio", removal_ratio)?;
        prepared_rows.push(out.into_py(py));
    }
    Ok(Some(prepared_rows))
}

pub(crate) fn detect_lm2011_exhibit_tail_impl(
    stripped_text: &str,
) -> (Option<usize>, String, Option<String>) {
    const TAIL_START_FRACTION: f64 = 0.60;
    const WEAK_CLUSTER_WINDOW_CHARS: usize = 4_000;

    if stripped_text.is_empty() {
        return (None, "no_tail_anchor".to_string(), None);
    }

    let text_len = stripped_text.chars().count();
    let min_start = ((text_len as f64) * TAIL_START_FRACTION) as usize;
    let mut strong_candidates: Vec<TailCandidate> = Vec::new();
    let mut weak_candidates: Vec<TailCandidate> = Vec::new();

    for (start, line) in iter_lines_with_char_offsets(stripped_text) {
        if line.is_empty() {
            continue;
        }
        if contains_exhibit_index_anchor(&line) {
            strong_candidates.push(TailCandidate {
                start,
                text: line,
                reason: "strong_anchor_exhibit_index",
            });
            continue;
        }
        if contains_ex_tag_anchor(&line) {
            strong_candidates.push(TailCandidate {
                start,
                text: line.clone(),
                reason: "strong_anchor_ex_tag",
            });
        }
        if weak_exhibit_line_anchor(&line) {
            weak_candidates.push(TailCandidate {
                start,
                text: line,
                reason: "weak_anchor_cluster",
            });
        }
    }

    let mut accepted_candidates: Vec<TailCandidate> = strong_candidates
        .iter()
        .filter(|candidate| candidate.start >= min_start)
        .cloned()
        .collect();

    for candidate in &weak_candidates {
        if candidate.start < min_start {
            continue;
        }
        let lookahead_end = candidate.start + WEAK_CLUSTER_WINDOW_CHARS;
        let weak_hits = weak_candidates
            .iter()
            .filter(|other| candidate.start <= other.start && other.start <= lookahead_end)
            .count();
        let strong_hit = strong_candidates
            .iter()
            .any(|other| candidate.start <= other.start && other.start <= lookahead_end);
        if weak_hits >= 2 || (weak_hits >= 1 && strong_hit) {
            accepted_candidates.push(candidate.clone());
        }
    }

    let Some(chosen) = accepted_candidates
        .into_iter()
        .min_by_key(|candidate| candidate.start)
    else {
        return (None, "no_tail_anchor".to_string(), None);
    };
    (
        Some(chosen.start),
        chosen.reason.to_string(),
        Some(chosen.text),
    )
}

#[pyfunction]
#[pyo3(signature = (text=""))]
pub(crate) fn detect_lm2011_exhibit_tail(text: &str) -> (Option<usize>, String, Option<String>) {
    detect_lm2011_exhibit_tail_impl(text)
}

#[pyfunction]
pub(crate) fn assign_finbert_token_bucket_value(
    token_count: i64,
    short_edge: i64,
    medium_edge: i64,
    max_length: i64,
) -> PyResult<String> {
    if token_count <= short_edge {
        return Ok("short".to_string());
    }
    if token_count <= medium_edge {
        return Ok("medium".to_string());
    }
    if token_count <= max_length {
        return Ok("long".to_string());
    }
    Err(PyValueError::new_err(format!(
        "Token count {token_count} exceeds the fixed FinBERT authority max length {max_length}."
    )))
}

#[pyfunction]
pub(crate) fn assign_finbert_token_bucket_values(
    token_counts: Vec<i64>,
    short_edge: i64,
    medium_edge: i64,
    max_length: i64,
) -> PyResult<Vec<String>> {
    token_counts
        .into_iter()
        .map(|token_count| {
            assign_finbert_token_bucket_value(token_count, short_edge, medium_edge, max_length)
        })
        .collect()
}

#[pyfunction]
pub(crate) fn finbert_visible_prefix_retained_end(
    offsets: Vec<Option<(i64, i64)>>,
    special_mask: Vec<i64>,
    text_len: usize,
) -> usize {
    let mut retained_end = 0usize;
    for (offset, is_special) in offsets.into_iter().zip(special_mask.into_iter()) {
        if is_special != 0 {
            continue;
        }
        let Some((start, end)) = offset else {
            continue;
        };
        if end <= start || end <= 0 {
            continue;
        }
        retained_end = retained_end.max(end as usize);
    }
    retained_end.min(text_len)
}

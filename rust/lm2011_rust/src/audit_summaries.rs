#![allow(unused_imports)]

use pyo3::exceptions::{PyOSError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyDict, PyFloat, PyInt, PyList, PySet, PyString, PyTuple};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};

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
use crate::sentence_quality_api::*;

#[pyfunction]
pub(crate) fn normalize_ascii_sample_text_value(text: &str) -> Option<String> {
    normalize_ascii_sample_text_impl(text)
}

#[pyfunction]
pub(crate) fn normalize_ascii_sample_text_values(texts: Vec<String>) -> Option<Vec<String>> {
    let mut out: Vec<String> = Vec::with_capacity(texts.len());
    for text in texts {
        let normalized = normalize_ascii_sample_text_impl(&text)?;
        out.push(normalized);
    }
    Some(out)
}

#[pyfunction]
pub(crate) fn sentence_length_ordered_item_codes(
    codes: Vec<String>,
    default_order: Vec<String>,
) -> Vec<String> {
    let seen: HashSet<&str> = codes.iter().map(String::as_str).collect();
    let default_set: HashSet<&str> = default_order.iter().map(String::as_str).collect();
    let mut out: Vec<String> = Vec::new();
    for code in default_order.iter() {
        if seen.contains(code.as_str()) {
            out.push(code.clone());
        }
    }
    for code in codes.iter() {
        if !default_set.contains(code.as_str()) {
            out.push(code.clone());
        }
    }
    out
}

#[pyfunction]
pub(crate) fn sentence_length_frame_records(
    py: Python<'_>,
    rows: &Bound<'_, PyAny>,
) -> PyResult<Vec<PyObject>> {
    let mut out: Vec<PyObject> = Vec::new();
    for row in rows.iter()? {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("sentence length row is not a dict"))?;
        let out_row = PyDict::new_bound(py);
        for (key, value) in dict.iter() {
            if value.is_none() {
                out_row.set_item(key, value)?;
            } else if value.hasattr("isoformat")? {
                out_row.set_item(key, value.call_method0("isoformat")?)?;
            } else {
                out_row.set_item(key, value)?;
            }
        }
        out.push(out_row.into_py(py));
    }
    Ok(out)
}

#[pyfunction]
pub(crate) fn sentence_length_frame_record_columns(
    py: Python<'_>,
    column_names: &Bound<'_, PyAny>,
    columns: &Bound<'_, PyAny>,
) -> PyResult<Vec<PyObject>> {
    let column_names = required_string_sequence(column_names, "column_names")?;
    let mut column_values: Vec<Vec<PyObject>> = Vec::with_capacity(column_names.len());
    let mut row_count: Option<usize> = None;
    for column in columns.iter()? {
        let column = column?;
        let mut values: Vec<PyObject> = Vec::new();
        for value in column.iter()? {
            values.push(value?.clone().into_py(py));
        }
        match row_count {
            Some(expected) if values.len() != expected => {
                return Err(PyValueError::new_err(
                    "all sentence length record columns must have the same length",
                ))
            }
            None => row_count = Some(values.len()),
            _ => {}
        }
        column_values.push(values);
    }
    if column_values.len() != column_names.len() {
        return Err(PyValueError::new_err(
            "sentence length record column count does not match column_names length",
        ));
    }

    let row_count = row_count.unwrap_or(0);
    let mut out: Vec<PyObject> = Vec::with_capacity(row_count);
    for row_idx in 0..row_count {
        let out_row = PyDict::new_bound(py);
        for (column_idx, column_name) in column_names.iter().enumerate() {
            let value = column_values[column_idx][row_idx].bind(py);
            if value.is_none() {
                out_row.set_item(column_name, value)?;
            } else if value.hasattr("isoformat")? {
                out_row.set_item(column_name, value.call_method0("isoformat")?)?;
            } else {
                out_row.set_item(column_name, value)?;
            }
        }
        out.push(out_row.into_py(py));
    }
    Ok(out)
}

pub(crate) fn sec_ccm_markdown_value_impl(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
) -> PyResult<String> {
    if value.is_none() {
        return Ok(String::new());
    }
    if value.is_instance_of::<PyFloat>() {
        let number = value.extract::<f64>()?;
        if number.is_nan() {
            return Ok(String::new());
        }
        let builtins = py.import_bound("builtins")?;
        let rendered = builtins.getattr("format")?.call1((value, ".6g"))?;
        return Ok(rendered.str()?.to_str()?.to_string());
    }
    let datetime_mod = py.import_bound("datetime")?;
    let datetime_type = datetime_mod.getattr("datetime")?;
    let date_type = datetime_mod.getattr("date")?;
    if value.is_instance(&datetime_type)? {
        let timezone_utc = datetime_mod.getattr("timezone")?.getattr("utc")?;
        let utc_value = value.call_method1("astimezone", (timezone_utc,))?;
        let kwargs = PyDict::new_bound(py);
        kwargs.set_item("timespec", "milliseconds")?;
        let rendered = utc_value.call_method("isoformat", (), Some(&kwargs))?;
        return Ok(rendered.str()?.to_str()?.replace("+00:00", "Z"));
    }
    if value.is_instance(&date_type)? {
        return Ok(value
            .call_method0("isoformat")?
            .str()?
            .to_str()?
            .to_string());
    }
    Ok(value.str()?.to_str()?.replace('|', "\\|"))
}

#[pyfunction]
pub(crate) fn sec_ccm_markdown_value(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<String> {
    sec_ccm_markdown_value_impl(py, value)
}

#[pyfunction]
pub(crate) fn sec_ccm_markdown_table(
    py: Python<'_>,
    columns: Vec<String>,
    rows: &Bound<'_, PyAny>,
    total_height: usize,
    max_rows: usize,
) -> PyResult<String> {
    if columns.is_empty() {
        return Ok("|  |\n|  |".to_string());
    }
    let mut lines: Vec<String> = Vec::new();
    lines.push(format!("| {} |", columns.join(" | ")));
    lines.push(format!("| {} |", vec!["---"; columns.len()].join(" | ")));
    for row in rows.iter()? {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("SEC/CCM markdown row is not a dict"))?;
        let mut values: Vec<String> = Vec::with_capacity(columns.len());
        for column in columns.iter() {
            let value = dict.get_item(column)?.ok_or_else(|| {
                PyValueError::new_err(format!("SEC/CCM markdown row missing column {column}"))
            })?;
            values.push(sec_ccm_markdown_value_impl(py, &value)?);
        }
        lines.push(format!("| {} |", values.join(" | ")));
    }
    let mut out = lines.join("\n");
    if total_height > max_rows {
        out.push_str(&format!("\n\n_Truncated to first {max_rows} rows._"));
    }
    Ok(out)
}

#[pyfunction]
pub(crate) fn sec_ccm_markdown_table_columns(
    py: Python<'_>,
    columns: Vec<String>,
    column_values: &Bound<'_, PyAny>,
    total_height: usize,
    max_rows: usize,
) -> PyResult<String> {
    if columns.is_empty() {
        return Ok("|  |\n|  |".to_string());
    }
    let (values_by_column, row_count) =
        collect_pyobject_column_values(py, &columns, column_values, "SEC/CCM markdown table")?;
    let mut lines: Vec<String> = Vec::new();
    lines.push(format!("| {} |", columns.join(" | ")));
    lines.push(format!("| {} |", vec!["---"; columns.len()].join(" | ")));
    for row_idx in 0..row_count {
        let mut values: Vec<String> = Vec::with_capacity(columns.len());
        for column in values_by_column.iter() {
            values.push(sec_ccm_markdown_value_impl(py, column[row_idx].bind(py))?);
        }
        lines.push(format!("| {} |", values.join(" | ")));
    }
    let mut out = lines.join("\n");
    if total_height > max_rows {
        out.push_str(&format!("\n\n_Truncated to first {max_rows} rows._"));
    }
    Ok(out)
}

#[derive(Default, Clone, Copy)]
pub(crate) struct NoItemStatsAccum {
    n_with_items: i64,
    n_no_items: i64,
    with_len_sum: f64,
    no_len_sum: f64,
}

pub(crate) fn sec_no_item_row_dict(
    py: Python<'_>,
    doc_type: &str,
    accum: NoItemStatsAccum,
) -> PyResult<PyObject> {
    let n_filings_eligible = accum
        .n_with_items
        .checked_add(accum.n_no_items)
        .ok_or_else(|| PyValueError::new_err("no-item eligible count overflow"))?;
    let share_no_item = if n_filings_eligible > 0 {
        accum.n_no_items as f64 / n_filings_eligible as f64
    } else {
        0.0
    };
    let avg_text_len_with_items = if accum.n_with_items > 0 {
        accum.with_len_sum / accum.n_with_items as f64
    } else {
        0.0
    };
    let avg_text_len_no_items = if accum.n_no_items > 0 {
        accum.no_len_sum / accum.n_no_items as f64
    } else {
        0.0
    };

    let out = PyDict::new_bound(py);
    out.set_item("year", "ALL")?;
    out.set_item("document_type_filename", doc_type)?;
    out.set_item("n_filings_eligible", n_filings_eligible)?;
    out.set_item("n_with_items", accum.n_with_items)?;
    out.set_item("n_no_items", accum.n_no_items)?;
    out.set_item("share_no_item", share_no_item)?;
    out.set_item("avg_text_len_with_items", avg_text_len_with_items)?;
    out.set_item("avg_text_len_no_items", avg_text_len_no_items)?;
    Ok(out.into_py(py))
}

#[pyfunction]
pub(crate) fn sec_no_item_aggregate_rows(
    py: Python<'_>,
    rows: &Bound<'_, PyAny>,
) -> PyResult<Vec<PyObject>> {
    let mut by_doc_type: BTreeMap<String, NoItemStatsAccum> = BTreeMap::new();
    for row in rows.iter()? {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("no-item stats row is not a dict"))?;
        let doc_type = dict_required_string(dict, "document_type_filename")?;
        let n_with_items = dict_required_i64(dict, "n_with_items")?;
        let n_no_items = dict_required_i64(dict, "n_no_items")?;
        let avg_text_len_with_items = dict_required_float(dict, "avg_text_len_with_items")?;
        let avg_text_len_no_items = dict_required_float(dict, "avg_text_len_no_items")?;
        let accum = by_doc_type.entry(doc_type).or_default();
        accum.n_with_items = accum
            .n_with_items
            .checked_add(n_with_items)
            .ok_or_else(|| PyValueError::new_err("no-item with-items count overflow"))?;
        accum.n_no_items = accum
            .n_no_items
            .checked_add(n_no_items)
            .ok_or_else(|| PyValueError::new_err("no-item no-items count overflow"))?;
        accum.with_len_sum += avg_text_len_with_items * n_with_items as f64;
        accum.no_len_sum += avg_text_len_no_items * n_no_items as f64;
    }

    let mut total = NoItemStatsAccum::default();
    let mut out_rows = Vec::with_capacity(by_doc_type.len() + 1);
    for (doc_type, accum) in by_doc_type {
        total.n_with_items = total
            .n_with_items
            .checked_add(accum.n_with_items)
            .ok_or_else(|| PyValueError::new_err("no-item total with-items count overflow"))?;
        total.n_no_items = total
            .n_no_items
            .checked_add(accum.n_no_items)
            .ok_or_else(|| PyValueError::new_err("no-item total no-items count overflow"))?;
        total.with_len_sum += accum.with_len_sum;
        total.no_len_sum += accum.no_len_sum;
        out_rows.push(sec_no_item_row_dict(py, &doc_type, accum)?);
    }
    out_rows.push(sec_no_item_row_dict(py, "TOTAL", total)?);
    Ok(out_rows)
}

#[pyfunction]
pub(crate) fn sec_no_item_aggregate_columns(
    py: Python<'_>,
    doc_types: &Bound<'_, PyAny>,
    n_with_items_values: &Bound<'_, PyAny>,
    n_no_items_values: &Bound<'_, PyAny>,
    avg_text_len_with_items_values: &Bound<'_, PyAny>,
    avg_text_len_no_items_values: &Bound<'_, PyAny>,
) -> PyResult<Vec<PyObject>> {
    let doc_types = required_string_sequence(doc_types, "document_type_filename")?;
    let n_with_items_values = required_i64_sequence(n_with_items_values, "n_with_items")?;
    let n_no_items_values = required_i64_sequence(n_no_items_values, "n_no_items")?;
    let avg_text_len_with_items_values =
        required_float_sequence(avg_text_len_with_items_values, "avg_text_len_with_items")?;
    let avg_text_len_no_items_values =
        required_float_sequence(avg_text_len_no_items_values, "avg_text_len_no_items")?;
    let row_count = doc_types.len();
    for (label, length) in [
        ("n_with_items", n_with_items_values.len()),
        ("n_no_items", n_no_items_values.len()),
        (
            "avg_text_len_with_items",
            avg_text_len_with_items_values.len(),
        ),
        ("avg_text_len_no_items", avg_text_len_no_items_values.len()),
    ] {
        if length != row_count {
            return Err(PyValueError::new_err(format!(
                "{label} length does not match document_type_filename length"
            )));
        }
    }

    let mut by_doc_type: BTreeMap<String, NoItemStatsAccum> = BTreeMap::new();
    for row_idx in 0..row_count {
        let n_with_items = n_with_items_values[row_idx];
        let n_no_items = n_no_items_values[row_idx];
        let accum = by_doc_type.entry(doc_types[row_idx].clone()).or_default();
        accum.n_with_items = accum
            .n_with_items
            .checked_add(n_with_items)
            .ok_or_else(|| PyValueError::new_err("no-item with-items count overflow"))?;
        accum.n_no_items = accum
            .n_no_items
            .checked_add(n_no_items)
            .ok_or_else(|| PyValueError::new_err("no-item no-items count overflow"))?;
        accum.with_len_sum += avg_text_len_with_items_values[row_idx] * n_with_items as f64;
        accum.no_len_sum += avg_text_len_no_items_values[row_idx] * n_no_items as f64;
    }

    let mut total = NoItemStatsAccum::default();
    let mut out_rows = Vec::with_capacity(by_doc_type.len() + 1);
    for (doc_type, accum) in by_doc_type {
        total.n_with_items = total
            .n_with_items
            .checked_add(accum.n_with_items)
            .ok_or_else(|| PyValueError::new_err("no-item total with-items count overflow"))?;
        total.n_no_items = total
            .n_no_items
            .checked_add(accum.n_no_items)
            .ok_or_else(|| PyValueError::new_err("no-item total no-items count overflow"))?;
        total.with_len_sum += accum.with_len_sum;
        total.no_len_sum += accum.no_len_sum;
        out_rows.push(sec_no_item_row_dict(py, &doc_type, accum)?);
    }
    out_rows.push(sec_no_item_row_dict(py, "TOTAL", total)?);
    Ok(out_rows)
}

pub(crate) fn normalize_binary_label_impl(raw: &str) -> &'static str {
    match raw.trim().to_lowercase().as_str() {
        "yes" | "y" | "true" | "1" | "negative" | "adverse" => "yes",
        "no" | "n" | "false" | "0" | "not_negative" | "non_negative" | "neutral" | "positive" => {
            "no"
        }
        "uncertain" | "unknown" | "unsure" | "maybe" | "ambiguous" => "uncertain",
        _ => "",
    }
}

#[pyfunction]
#[pyo3(signature = (value=None))]
pub(crate) fn normalize_binary_label_value(value: Option<&Bound<'_, PyAny>>) -> PyResult<String> {
    let Some(value) = value else {
        return Ok(String::new());
    };
    if value.is_none() {
        return Ok(String::new());
    }
    let rendered = value.str()?;
    Ok(normalize_binary_label_impl(rendered.to_str()?).to_string())
}

pub(crate) fn regime_is_amendment_form(form: &str) -> bool {
    if form.contains("/A") || form.ends_with("-A") || form.ends_with("/A") {
        return true;
    }
    let trimmed = form.trim_end();
    let chars: Vec<char> = trimmed.chars().collect();
    if chars.last().is_none_or(|ch| *ch != 'A') {
        return false;
    }
    let mut pos = chars.len().saturating_sub(1);
    while pos > 0 && chars[pos - 1].is_whitespace() {
        pos -= 1;
    }
    pos > 0 && matches!(chars[pos - 1], '-' | '/')
}

pub(crate) fn normalize_regime_form_type_impl(form_type: Option<&str>) -> Option<String> {
    let form = form_type?.trim().to_ascii_uppercase();
    if form.is_empty() || regime_is_amendment_form(&form) {
        return None;
    }
    if form.starts_with("10-K") || form.starts_with("10K") {
        return Some("10-K".to_string());
    }
    if form.starts_with("10-Q") || form.starts_with("10Q") {
        return Some("10-Q".to_string());
    }
    None
}

#[pyfunction]
#[pyo3(signature = (form_type=None))]
pub(crate) fn normalize_regime_form_type(form_type: Option<&str>) -> Option<String> {
    normalize_regime_form_type_impl(form_type)
}

pub(crate) fn benchmark_item_code_to_text_scope_impl(value: Option<&str>) -> Option<String> {
    let value = value?;
    let normalized = value.trim().to_lowercase().replace('-', "_");
    if normalized.is_empty() {
        return None;
    }
    match normalized.as_str() {
        "item_7" => Some("item_7_mda".to_string()),
        "item_1a" => Some("item_1a_risk_factors".to_string()),
        "item_1" => Some("item_1_business".to_string()),
        _ => Some(normalized),
    }
}

#[pyfunction]
#[pyo3(signature = (value=None))]
pub(crate) fn benchmark_item_code_to_text_scope_value(value: Option<&str>) -> Option<String> {
    benchmark_item_code_to_text_scope_impl(value)
}

#[pyfunction]
pub(crate) fn benchmark_item_code_to_text_scope_values(
    values: &Bound<'_, PyAny>,
) -> PyResult<Vec<Option<String>>> {
    let mut out = Vec::new();
    for value in values.iter()? {
        let value = value?;
        let scope = if value.is_none() {
            benchmark_item_code_to_text_scope_impl(None)
        } else {
            benchmark_item_code_to_text_scope_impl(Some(value.str()?.to_str()?))
        };
        out.push(scope);
    }
    Ok(out)
}

#[pyfunction]
pub(crate) fn item_cleaning_base_row_payload_value(
    py: Python<'_>,
    row: &Bound<'_, PyDict>,
    text_scope: &str,
) -> PyResult<PyObject> {
    let out = PyDict::new_bound(py);
    for key in [
        "doc_id",
        "cik_10",
        "accession_nodash",
        "filing_date",
        "filing_year",
        "benchmark_row_id",
        "benchmark_item_code",
        "benchmark_item_label",
        "item_id",
        "canonical_item",
        "document_type",
        "document_type_raw",
        "document_type_normalized",
        "source_year_file",
        "source_record_id",
        "source_file_row_nr",
        "boundary_authority_status",
    ] {
        out.set_item(key, dict_py_object_or_none(py, row, key)?)?;
    }
    out.set_item(
        "calendar_year",
        dict_py_object_or_none(py, row, "filing_year")?,
    )?;
    out.set_item("text_scope", text_scope)?;
    Ok(out.into_py(py))
}

#[pyfunction]
#[pyo3(signature = (manual_audit_candidate, boundary_authority_status=None, existing_review_status=None))]
pub(crate) fn item_cleaning_review_status_value(
    manual_audit_candidate: bool,
    boundary_authority_status: Option<&str>,
    existing_review_status: Option<&str>,
) -> &'static str {
    if existing_review_status == Some("approved") {
        return "approved";
    }
    if existing_review_status == Some("rejected") {
        return "rejected";
    }
    if manual_audit_candidate || boundary_authority_status == Some("review_needed") {
        return "required_unreviewed";
    }
    if matches!(boundary_authority_status, None | Some("" | "unknown")) {
        return "required_unreviewed";
    }
    "not_required"
}

#[pyfunction]
pub(crate) fn item_cleaning_production_eligible_value(review_status: &str) -> bool {
    matches!(review_status, "not_required" | "approved")
}

pub(crate) fn item_cleaning_set_base_payload(
    py: Python<'_>,
    out: &Bound<'_, PyDict>,
    row: &Bound<'_, PyDict>,
    text_scope: &str,
) -> PyResult<()> {
    for key in [
        "doc_id",
        "cik_10",
        "accession_nodash",
        "filing_date",
        "filing_year",
        "benchmark_row_id",
        "benchmark_item_code",
        "benchmark_item_label",
        "item_id",
        "canonical_item",
        "document_type",
        "document_type_raw",
        "document_type_normalized",
        "source_year_file",
        "source_record_id",
        "source_file_row_nr",
        "boundary_authority_status",
    ] {
        out.set_item(key, dict_py_object_or_none(py, row, key)?)?;
    }
    out.set_item(
        "calendar_year",
        dict_py_object_or_none(py, row, "filing_year")?,
    )?;
    out.set_item("text_scope", text_scope)?;
    Ok(())
}

pub(crate) fn item_cleaning_snippet_start(text: &str, length: usize) -> String {
    text.chars().take(length).collect()
}

pub(crate) fn item_cleaning_snippet_end(text: &str, length: usize) -> String {
    let char_count = text.chars().count();
    if char_count <= length {
        return text.to_string();
    }
    text.chars().skip(char_count - length).collect()
}

pub(crate) fn dict_required_bool(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<bool> {
    let Some(value) = dict.get_item(key)? else {
        return Err(PyValueError::new_err(format!(
            "missing required key: {key}"
        )));
    };
    value.is_truthy()
}

pub(crate) fn set_optional_string_item(
    dict: &Bound<'_, PyDict>,
    key: &str,
    value: &Option<String>,
) -> PyResult<()> {
    match value {
        Some(value) => dict.set_item(key, value),
        None => dict.set_item(key, Option::<String>::None),
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn item_cleaning_finalize_rows(
    py: Python<'_>,
    records: &Bound<'_, PyAny>,
    cfg_enabled: bool,
    cleaning_policy_id: &str,
    drop_blank_after_cleaning: bool,
    hard_drop_min_clean_char_count: &Bound<'_, PyAny>,
    warn_below_clean_char_count: i64,
    large_removal_warning_threshold: f64,
    enforce_item7_lm_token_floor: bool,
    item7_min_lm_tokens: i64,
    segment_policy: &str,
) -> PyResult<(Vec<PyObject>, Vec<PyObject>)> {
    let hard_drop_min_clean_char_count = if hard_drop_min_clean_char_count.is_none() {
        None
    } else {
        let value = py_int_like_to_i64(hard_drop_min_clean_char_count)?;
        if value < 0 {
            return Err(PyValueError::new_err(
                "hard_drop_min_clean_char_count must be nonnegative",
            ));
        }
        Some(value as usize)
    };
    let mut audit_rows: Vec<PyObject> = Vec::new();
    let mut cleaned_rows: Vec<PyObject> = Vec::new();

    for record in records.iter()? {
        let record = record?;
        let record = record
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("cleaning finalization record is not a dict"))?;
        let row_value = record
            .get_item("row")?
            .ok_or_else(|| PyValueError::new_err("cleaning finalization record is missing row"))?;
        let row = row_value
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("cleaning finalization row is not a dict"))?;
        let text_scope = dict_required_string(record, "text_scope")?;
        let original_text = dict_required_string(record, "original_text")?;
        let cleaned_text = dict_required_string(record, "cleaned_text")?;
        let original_char_count = dict_required_i64(record, "original_char_count")?;
        let cleaned_char_count = dict_required_i64(record, "cleaned_char_count")?;
        let removed_char_count = dict_required_i64(record, "removed_char_count")?;
        let removal_ratio = dict_required_float(record, "removal_ratio")?;
        let cleaned_lm_total_token_count =
            dict_required_i64(record, "cleaned_lm_total_token_count")?;
        let page_marker_lines_removed = dict_required_i64(record, "page_marker_lines_removed")?;
        let report_header_footer_lines_removed =
            dict_required_i64(record, "report_header_footer_lines_removed")?;
        let structural_tag_lines_removed =
            dict_required_i64(record, "structural_tag_lines_removed")?;
        let table_like_lines_removed = dict_required_i64(record, "table_like_lines_removed")?;
        let toc_prefix_trimmed = dict_required_bool(record, "toc_prefix_trimmed")?;
        let toc_prefix_trimmed_char_count =
            dict_required_i64(record, "toc_prefix_trimmed_char_count")?;
        let tail_truncated = dict_required_bool(record, "tail_truncated")?;
        let tail_truncated_char_count = dict_required_i64(record, "tail_truncated_char_count")?;
        let reference_only_stub = dict_required_bool(record, "reference_only_stub")?;
        let effectively_non_body_text = dict_required_bool(record, "effectively_non_body_text")?;

        let reason = item_cleaning_drop_reason_value(
            &cleaned_text,
            &text_scope,
            cfg_enabled,
            drop_blank_after_cleaning,
            reference_only_stub,
            effectively_non_body_text,
            enforce_item7_lm_token_floor,
            cleaned_lm_total_token_count,
            item7_min_lm_tokens,
            hard_drop_min_clean_char_count,
        );
        let dropped = reason.is_some();
        let warning_large_removal = original_char_count != 0
            && removal_ratio >= large_removal_warning_threshold
            && cleaned_char_count < original_char_count;
        let warning_below_clean_char_count =
            !dropped && cleaned_char_count < warn_below_clean_char_count;
        let item7_lm_token_floor_failed = text_scope == "item_7_mda"
            && enforce_item7_lm_token_floor
            && cleaned_lm_total_token_count < item7_min_lm_tokens;

        let audit = PyDict::new_bound(py);
        item_cleaning_set_base_payload(py, &audit, row, &text_scope)?;
        audit.set_item("cleaning_policy_id", cleaning_policy_id)?;
        audit.set_item("original_char_count", original_char_count)?;
        audit.set_item("cleaned_char_count", cleaned_char_count)?;
        audit.set_item("removed_char_count", removed_char_count)?;
        audit.set_item("removal_ratio", removal_ratio)?;
        audit.set_item("cleaned_lm_total_token_count", cleaned_lm_total_token_count)?;
        audit.set_item("dropped_after_cleaning", dropped)?;
        set_optional_string_item(&audit, "drop_reason", &reason)?;
        audit.set_item("segment_policy_id", segment_policy)?;
        audit.set_item("page_marker_lines_removed", page_marker_lines_removed)?;
        audit.set_item(
            "report_header_footer_lines_removed",
            report_header_footer_lines_removed,
        )?;
        audit.set_item("structural_tag_lines_removed", structural_tag_lines_removed)?;
        audit.set_item("table_like_lines_removed", table_like_lines_removed)?;
        audit.set_item("toc_prefix_trimmed", toc_prefix_trimmed)?;
        audit.set_item(
            "toc_prefix_trimmed_char_count",
            toc_prefix_trimmed_char_count,
        )?;
        audit.set_item("tail_truncated", tail_truncated)?;
        audit.set_item("tail_truncated_char_count", tail_truncated_char_count)?;
        audit.set_item("reference_only_stub", reference_only_stub)?;
        audit.set_item("effectively_non_body_text", effectively_non_body_text)?;
        audit.set_item("warning_large_removal", warning_large_removal)?;
        audit.set_item(
            "warning_below_clean_char_count",
            warning_below_clean_char_count,
        )?;
        audit.set_item("item7_lm_token_floor_failed", item7_lm_token_floor_failed)?;
        let manual_audit_reason = item_cleaning_manual_audit_reason_value(
            dropped,
            warning_large_removal,
            toc_prefix_trimmed,
            tail_truncated,
            reference_only_stub,
            item7_lm_token_floor_failed,
            warning_below_clean_char_count,
            reason.as_deref(),
        );
        let manual_audit_candidate = manual_audit_reason.is_some();
        audit.set_item("manual_audit_candidate", manual_audit_candidate)?;
        set_optional_string_item(&audit, "manual_audit_reason", &manual_audit_reason)?;
        audit.set_item(
            "original_start_snippet",
            item_cleaning_snippet_start(&original_text, 500),
        )?;
        audit.set_item(
            "cleaned_start_snippet",
            item_cleaning_snippet_start(&cleaned_text, 500),
        )?;
        audit.set_item(
            "original_end_snippet",
            item_cleaning_snippet_end(&original_text, 500),
        )?;
        audit.set_item(
            "cleaned_end_snippet",
            item_cleaning_snippet_end(&cleaned_text, 500),
        )?;
        let boundary_authority_status = dict_raw_string(row, "boundary_authority_status")?;
        let existing_review_status = dict_raw_string(row, "review_status")?;
        let review_status = item_cleaning_review_status_value(
            manual_audit_candidate,
            boundary_authority_status.as_deref(),
            existing_review_status.as_deref(),
        );
        let production_eligible = item_cleaning_production_eligible_value(review_status);
        audit.set_item("review_status", review_status)?;
        audit.set_item("production_eligible", production_eligible)?;

        if !dropped {
            let cleaned = PyDict::new_bound(py);
            item_cleaning_set_base_payload(py, &cleaned, row, &text_scope)?;
            cleaned.set_item("cleaning_policy_id", cleaning_policy_id)?;
            cleaned.set_item("original_char_count", original_char_count)?;
            cleaned.set_item("cleaned_char_count", cleaned_char_count)?;
            cleaned.set_item("removed_char_count", removed_char_count)?;
            cleaned.set_item("removal_ratio", removal_ratio)?;
            cleaned.set_item("cleaned_lm_total_token_count", cleaned_lm_total_token_count)?;
            cleaned.set_item("cleaned_text", cleaned_text)?;
            cleaned.set_item("dropped_after_cleaning", dropped)?;
            set_optional_string_item(&cleaned, "drop_reason", &reason)?;
            cleaned.set_item("segment_policy_id", segment_policy)?;
            cleaned.set_item("review_status", review_status)?;
            cleaned.set_item("production_eligible", production_eligible)?;
            cleaned_rows.push(cleaned.into_py(py));
        }
        audit_rows.push(audit.into_py(py));
    }

    Ok((audit_rows, cleaned_rows))
}

pub(crate) fn item_cleaning_activation_status_impl(text_scope: &str) -> &'static str {
    match text_scope {
        "item_7_mda" | "item_1a_risk_factors" => "blocked_pending_manual_audit",
        "item_1_business" => "robustness_only_pending_manual_audit",
        _ => "diagnostic_only",
    }
}

#[pyfunction]
pub(crate) fn item_cleaning_activation_status(text_scope: &str) -> &'static str {
    item_cleaning_activation_status_impl(text_scope)
}

#[pyfunction]
pub(crate) fn item_cleaning_activation_status_values(
    values: &Bound<'_, PyAny>,
) -> PyResult<Vec<String>> {
    let mut out = Vec::new();
    for value in values.iter()? {
        let value = value?;
        let status = if value.is_none() {
            item_cleaning_activation_status_impl("")
        } else {
            item_cleaning_activation_status_impl(value.str()?.to_str()?)
        };
        out.push(status.to_string());
    }
    Ok(out)
}

pub(crate) fn item_cleaning_audit_period_impl(calendar_year: Option<i64>) -> &'static str {
    let Some(calendar_year) = calendar_year else {
        return "unknown";
    };
    if calendar_year <= 2008 {
        "pre_2009"
    } else if calendar_year <= 2016 {
        "2009_2016"
    } else {
        "2017_2024"
    }
}

#[pyfunction]
#[pyo3(signature = (calendar_year=None))]
pub(crate) fn item_cleaning_audit_period(calendar_year: Option<i64>) -> &'static str {
    item_cleaning_audit_period_impl(calendar_year)
}

#[pyfunction]
pub(crate) fn item_cleaning_audit_period_values(
    values: &Bound<'_, PyAny>,
) -> PyResult<Vec<String>> {
    let mut out = Vec::new();
    for value in values.iter()? {
        let value = value?;
        let calendar_year = if value.is_none() {
            None
        } else {
            Some(value.extract::<i64>()?)
        };
        out.push(item_cleaning_audit_period_impl(calendar_year).to_string());
    }
    Ok(out)
}

pub(crate) fn item_cleaning_manual_audit_sample_reason(
    dict: &Bound<'_, PyDict>,
) -> PyResult<String> {
    let Some(reason) = dict.get_item("manual_audit_reason")? else {
        return Ok("background_scope_period_sample".to_string());
    };
    if reason.is_truthy()? {
        return Ok(reason.str()?.to_str()?.to_string());
    }
    Ok("background_scope_period_sample".to_string())
}

#[pyfunction]
pub(crate) fn item_cleaning_manual_audit_sample_rows(
    py: Python<'_>,
    rows: &Bound<'_, PyAny>,
    max_rows_per_scope_period: usize,
) -> PyResult<Vec<PyObject>> {
    if max_rows_per_scope_period == 0 {
        return Ok(Vec::new());
    }
    let mut selected_rows = Vec::new();
    let mut counts: HashMap<(String, String), usize> = HashMap::new();
    let required_keys = [
        "doc_id",
        "filing_date",
        "calendar_year",
        "audit_period",
        "text_scope",
        "benchmark_row_id",
        "cleaning_policy_id",
        "original_start_snippet",
        "cleaned_start_snippet",
        "original_end_snippet",
        "cleaned_end_snippet",
        "dropped_after_cleaning",
        "warning_large_removal",
        "toc_prefix_trimmed",
        "tail_truncated",
        "reference_only_stub",
        "item7_lm_token_floor_failed",
    ];

    for row in rows.iter()? {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("manual audit sample row is not a dict"))?;
        let text_scope = dict_required_string(dict, "text_scope")?;
        let audit_period = dict_required_string(dict, "audit_period")?;
        let key = (text_scope, audit_period);
        let count = counts.entry(key).or_insert(0);
        if *count >= max_rows_per_scope_period {
            continue;
        }
        *count += 1;

        let out = PyDict::new_bound(py);
        for key in required_keys {
            copy_required_dict_key(&out, dict, key)?;
        }
        copy_optional_dict_key(&out, dict, "boundary_authority_status")?;
        copy_optional_dict_key(&out, dict, "review_status")?;
        copy_optional_dict_key(&out, dict, "production_eligible")?;
        out.set_item(
            "sample_reason",
            item_cleaning_manual_audit_sample_reason(dict)?,
        )?;
        out.set_item("start_boundary_correct", Option::<bool>::None)?;
        out.set_item("end_boundary_correct", Option::<bool>::None)?;
        out.set_item("wrong_item_capture_absent", Option::<bool>::None)?;
        out.set_item("toc_capture_absent", Option::<bool>::None)?;
        out.set_item("body_text_nonempty", Option::<bool>::None)?;
        selected_rows.push(out.into_py(py));
    }

    Ok(selected_rows)
}

#[pyfunction]
#[pyo3(signature = (
    cleaned_text,
    text_scope,
    cfg_enabled,
    drop_blank_after_cleaning,
    reference_only_stub,
    effectively_non_body_text,
    enforce_item7_lm_token_floor,
    cleaned_lm_total_token_count,
    item7_min_lm_tokens,
    hard_drop_min_clean_char_count=None
))]
pub(crate) fn item_cleaning_drop_reason_value(
    cleaned_text: &str,
    text_scope: &str,
    cfg_enabled: bool,
    drop_blank_after_cleaning: bool,
    reference_only_stub: bool,
    effectively_non_body_text: bool,
    enforce_item7_lm_token_floor: bool,
    cleaned_lm_total_token_count: i64,
    item7_min_lm_tokens: i64,
    hard_drop_min_clean_char_count: Option<usize>,
) -> Option<String> {
    if !cfg_enabled {
        return None;
    }
    if drop_blank_after_cleaning && cleaned_text.trim().is_empty() {
        return Some("blank_after_cleaning".to_string());
    }
    if reference_only_stub {
        return Some("reference_only_stub".to_string());
    }
    if effectively_non_body_text {
        return Some("non_body_text".to_string());
    }
    if hard_drop_min_clean_char_count
        .is_some_and(|min_chars| cleaned_text.chars().count() < min_chars)
    {
        return Some("below_min_clean_char_count".to_string());
    }
    if text_scope == "item_7_mda"
        && enforce_item7_lm_token_floor
        && cleaned_lm_total_token_count < item7_min_lm_tokens
    {
        return Some("item7_below_lm_token_floor".to_string());
    }
    None
}

pub(crate) fn push_unique_reason(reasons: &mut Vec<String>, value: String) {
    if !reasons.iter().any(|existing| existing == &value) {
        reasons.push(value);
    }
}

#[pyfunction]
#[pyo3(signature = (
    dropped_after_cleaning,
    warning_large_removal,
    toc_prefix_trimmed,
    tail_truncated,
    reference_only_stub,
    item7_lm_token_floor_failed,
    warning_below_clean_char_count,
    drop_reason=None
))]
pub(crate) fn item_cleaning_manual_audit_reason_value(
    dropped_after_cleaning: bool,
    warning_large_removal: bool,
    toc_prefix_trimmed: bool,
    tail_truncated: bool,
    reference_only_stub: bool,
    item7_lm_token_floor_failed: bool,
    warning_below_clean_char_count: bool,
    drop_reason: Option<&str>,
) -> Option<String> {
    let mut reasons: Vec<String> = Vec::new();
    if dropped_after_cleaning {
        let reason = match drop_reason {
            Some(value) if !value.is_empty() => value.to_string(),
            _ => "dropped".to_string(),
        };
        push_unique_reason(&mut reasons, reason);
    }
    if warning_large_removal {
        push_unique_reason(&mut reasons, "large_removal".to_string());
    }
    if toc_prefix_trimmed {
        push_unique_reason(&mut reasons, "toc_prefix_trimmed".to_string());
    }
    if tail_truncated {
        push_unique_reason(&mut reasons, "tail_truncated".to_string());
    }
    if reference_only_stub {
        push_unique_reason(&mut reasons, "reference_only_stub".to_string());
    }
    if item7_lm_token_floor_failed {
        push_unique_reason(&mut reasons, "item7_lm_token_floor_failed".to_string());
    }
    if warning_below_clean_char_count {
        push_unique_reason(&mut reasons, "below_clean_char_warning".to_string());
    }
    if reasons.is_empty() {
        None
    } else {
        Some(reasons.join("|"))
    }
}

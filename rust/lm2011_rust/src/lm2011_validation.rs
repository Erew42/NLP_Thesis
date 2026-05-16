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
#[pyo3(signature = (text=None))]
pub(crate) fn legacy_lm2011_tokens_value(text: Option<&str>) -> Vec<String> {
    legacy_lm2011_tokens_impl(text)
}

#[pyfunction]
#[pyo3(signature = (text=None, limit=400))]
pub(crate) fn lm2011_validation_truncate_text(text: Option<&str>, limit: i64) -> Option<String> {
    truncate_text_impl(text, limit)
}

#[pyfunction]
#[pyo3(signature = (text=None, snippet_char_limit=400))]
pub(crate) fn lm2011_validation_marker_flags_and_snippet(
    text: Option<&str>,
    snippet_char_limit: i64,
) -> (BTreeMap<String, bool>, Option<String>) {
    validation_marker_flags_and_snippet_impl(text, snippet_char_limit)
}

#[pyfunction]
pub(crate) fn lm2011_validation_count_true(
    records: &Bound<'_, PyAny>,
    field_name: &str,
) -> PyResult<i64> {
    let mut count = 0_i64;
    for record in records.iter()? {
        let record = record?;
        let dict = record
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("record is not a dict"))?;
        let Some(value) = dict.get_item(field_name)? else {
            continue;
        };
        if value.is_truthy()? {
            count += 1;
        }
    }
    Ok(count)
}

#[pyfunction]
pub(crate) fn lm2011_validation_count_series(values: Vec<String>) -> BTreeMap<String, i64> {
    let mut counts: BTreeMap<String, i64> = BTreeMap::new();
    for value in values {
        *counts.entry(value).or_insert(0) += 1;
    }
    counts
}

#[pyfunction]
pub(crate) fn lm2011_validation_form_count_rows(
    py: Python<'_>,
    corpus_label: &str,
    counts: &Bound<'_, PyAny>,
) -> PyResult<Vec<PyObject>> {
    let count_dict = counts
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("form counts are not a dict"))?;
    let mut normalized_counts: BTreeMap<String, i64> = BTreeMap::new();
    for (key, value) in count_dict.iter() {
        let normalized_key = key.str()?.to_str()?.to_string();
        let count = py_int_like_to_i64(&value)?;
        normalized_counts.insert(normalized_key, count);
    }

    let mut rows: Vec<PyObject> = Vec::with_capacity(normalized_counts.len());
    for (normalized_form, doc_count) in normalized_counts {
        let out = PyDict::new_bound(py);
        out.set_item("corpus_label", corpus_label)?;
        out.set_item("normalized_form", normalized_form)?;
        out.set_item("doc_count", doc_count)?;
        rows.push(out.into_py(py));
    }
    Ok(rows)
}

#[pyfunction]
#[pyo3(signature = (values, other_value=Some("Other")))]
pub(crate) fn lm2011_validation_normalized_form_counts(
    values: &Bound<'_, PyAny>,
    other_value: Option<&str>,
) -> PyResult<BTreeMap<String, i64>> {
    let mut counts: BTreeMap<String, i64> = BTreeMap::new();
    if values.is_none() {
        return Ok(counts);
    }
    for value in values.iter()? {
        let value = value?;
        if value.is_none() {
            continue;
        }
        let rendered = value.str()?;
        if let Some(normalized) = normalize_lm2011_form_value(Some(rendered.to_str()?), other_value)
        {
            *counts.entry(normalized).or_insert(0) += 1;
        }
    }
    Ok(counts)
}

#[pyfunction]
pub(crate) fn lm2011_validation_term_count_updates(
    values: &Bound<'_, PyAny>,
    terms: Vec<String>,
) -> PyResult<(i64, BTreeMap<String, i64>)> {
    let mut doc_count = 0_i64;
    let mut term_counts: BTreeMap<String, i64> = BTreeMap::new();
    for value in values.iter()? {
        let value = value?;
        doc_count += 1;
        if !value.is_instance_of::<PyString>() {
            continue;
        }
        let text = value.extract::<&str>()?;
        let token_set: HashSet<String> = tokenize_impl(Some(text)).into_iter().collect();
        for term in &terms {
            if token_set.contains(term) {
                *term_counts.entry(term.clone()).or_insert(0) += 1;
            }
        }
    }
    Ok((doc_count, term_counts))
}

pub(crate) fn optional_doc_id_text(value: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    if value.is_none() {
        return Ok(None);
    }
    let rendered = value.str()?;
    Ok(Some(rendered.to_str()?.to_string()))
}

pub(crate) fn optional_year_i64(value: &Bound<'_, PyAny>) -> PyResult<Option<i64>> {
    if value.is_none() {
        return Ok(None);
    }
    Ok(Some(py_int_like_to_i64(value)?))
}

#[pyfunction]
pub(crate) fn lm2011_validation_event_attrition_rows(
    py: Python<'_>,
    accepted_doc_ids: &Bound<'_, PyAny>,
    accepted_years: &Bound<'_, PyAny>,
    event_doc_ids: &Bound<'_, PyAny>,
    event_years: &Bound<'_, PyAny>,
) -> PyResult<Vec<PyObject>> {
    let mut accepted_by_year: BTreeMap<i64, Vec<Option<String>>> = BTreeMap::new();
    let mut accepted_all: Vec<Option<String>> = Vec::new();
    let mut accepted_doc_iter = accepted_doc_ids.iter()?;
    let mut accepted_year_iter = accepted_years.iter()?;
    loop {
        match (accepted_doc_iter.next(), accepted_year_iter.next()) {
            (Some(doc_result), Some(year_result)) => {
                let doc_id = optional_doc_id_text(&doc_result?)?;
                let Some(filing_year) = optional_year_i64(&year_result?)? else {
                    return Err(PyValueError::new_err("accepted filing_year is null"));
                };
                accepted_by_year
                    .entry(filing_year)
                    .or_default()
                    .push(doc_id.clone());
                accepted_all.push(doc_id);
            }
            (None, None) => break,
            _ => {
                return Err(PyValueError::new_err(
                    "accepted doc_id and filing_year lengths differ",
                ))
            }
        }
    }

    let mut event_doc_ids_all: HashSet<String> = HashSet::new();
    let mut event_doc_ids_by_year: HashMap<i64, HashSet<String>> = HashMap::new();
    let mut event_counts_by_year: BTreeMap<i64, i64> = BTreeMap::new();
    let mut event_total_count = 0_i64;
    let mut event_doc_iter = event_doc_ids.iter()?;
    let mut event_year_iter = event_years.iter()?;
    loop {
        match (event_doc_iter.next(), event_year_iter.next()) {
            (Some(doc_result), Some(year_result)) => {
                event_total_count += 1;
                let doc_id = optional_doc_id_text(&doc_result?)?;
                let filing_year = optional_year_i64(&year_result?)?;
                if let Some(ref doc_id) = doc_id {
                    event_doc_ids_all.insert(doc_id.clone());
                }
                if let Some(filing_year) = filing_year {
                    *event_counts_by_year.entry(filing_year).or_insert(0) += 1;
                    if let Some(doc_id) = doc_id {
                        event_doc_ids_by_year
                            .entry(filing_year)
                            .or_default()
                            .insert(doc_id);
                    }
                }
            }
            (None, None) => break,
            _ => {
                return Err(PyValueError::new_err(
                    "event doc_id and filing_year lengths differ",
                ))
            }
        }
    }

    let mut rows: Vec<PyObject> = Vec::with_capacity(accepted_by_year.len() + 1);
    for (filing_year, doc_ids) in accepted_by_year {
        let event_doc_ids = event_doc_ids_by_year.get(&filing_year);
        let lost_doc_count = doc_ids
            .iter()
            .filter(|doc_id| match (doc_id.as_ref(), event_doc_ids) {
                (Some(doc_id), Some(event_doc_ids)) => !event_doc_ids.contains(doc_id),
                (Some(_), None) | (None, _) => true,
            })
            .count() as i64;
        let out = PyDict::new_bound(py);
        out.set_item("filing_year", filing_year)?;
        out.set_item("backbone_doc_count", doc_ids.len() as i64)?;
        out.set_item(
            "event_panel_doc_count",
            *event_counts_by_year.get(&filing_year).unwrap_or(&0),
        )?;
        out.set_item("lost_doc_count", lost_doc_count)?;
        rows.push(out.into_py(py));
    }

    let overall_lost_doc_count = accepted_all
        .iter()
        .filter(|doc_id| match doc_id.as_ref() {
            Some(doc_id) => !event_doc_ids_all.contains(doc_id),
            None => true,
        })
        .count() as i64;
    let out = PyDict::new_bound(py);
    out.set_item("filing_year", py.None())?;
    out.set_item("backbone_doc_count", accepted_all.len() as i64)?;
    out.set_item("event_panel_doc_count", event_total_count)?;
    out.set_item("lost_doc_count", overall_lost_doc_count)?;
    rows.push(out.into_py(py));
    Ok(rows)
}

#[pyfunction]
pub(crate) fn lm2011_validation_units_row(
    py: Python<'_>,
    artifact_name: &str,
    field_name: &str,
    values: &Bound<'_, PyAny>,
    paper_multiplier: f64,
) -> PyResult<PyObject> {
    let mut nonnull_row_count = 0_i64;
    let mut abs_sum = 0.0_f64;
    for value in values.iter()? {
        let value = value?;
        if value.is_none() {
            continue;
        }
        let number = value.extract::<f64>()?;
        nonnull_row_count += 1;
        abs_sum += number.abs();
    }
    let mean_abs = if nonnull_row_count > 0 {
        Some(abs_sum / nonnull_row_count as f64)
    } else {
        None
    };

    let out = PyDict::new_bound(py);
    out.set_item("artifact_name", artifact_name)?;
    out.set_item("field_name", field_name)?;
    out.set_item("nonnull_row_count", nonnull_row_count)?;
    match mean_abs {
        Some(mean_abs) => {
            out.set_item("mean_abs_internal", mean_abs)?;
            out.set_item(
                "mean_abs_paper_display_equivalent",
                mean_abs * paper_multiplier,
            )?;
        }
        None => {
            out.set_item("mean_abs_internal", py.None())?;
            out.set_item("mean_abs_paper_display_equivalent", py.None())?;
        }
    }
    out.set_item("paper_display_multiplier", paper_multiplier)?;
    out.set_item("classification", "clearly_internal_unit")?;
    Ok(out.into_py(py))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn lm2011_validation_packet_d_coverage_row(
    py: Python<'_>,
    run_name: &str,
    finbert_backbone_path: &Bound<'_, PyAny>,
    sample_backbone_path: &str,
    finbert_backbone_matches_sample_backbone: bool,
    finbert_backbone_path_matches_sample_backbone: bool,
    reported_backbone_doc_count: i64,
    actual_filtered_doc_count: &Bound<'_, PyAny>,
    sample_backbone_filtered_doc_count: &Bound<'_, PyAny>,
    reported_covered_doc_count: i64,
    actual_covered_doc_count: i64,
) -> PyResult<PyObject> {
    let normalized_finbert_backbone_path = if finbert_backbone_path.is_none() {
        None
    } else {
        Some(finbert_backbone_path.str()?.to_str()?.to_string())
    };
    let actual_filtered_doc_count = if actual_filtered_doc_count.is_none() {
        None
    } else {
        Some(py_int_like_to_i64(actual_filtered_doc_count)?)
    };
    let sample_backbone_filtered_doc_count = if sample_backbone_filtered_doc_count.is_none() {
        None
    } else {
        Some(py_int_like_to_i64(sample_backbone_filtered_doc_count)?)
    };
    let out = PyDict::new_bound(py);
    out.set_item("run_name", run_name)?;
    match normalized_finbert_backbone_path {
        Some(path) => out.set_item("finbert_backbone_path", path)?,
        None => out.set_item("finbert_backbone_path", py.None())?,
    }
    out.set_item("sample_backbone_path", sample_backbone_path)?;
    out.set_item(
        "finbert_backbone_matches_sample_backbone",
        finbert_backbone_matches_sample_backbone,
    )?;
    out.set_item(
        "finbert_backbone_path_matches_sample_backbone",
        finbert_backbone_path_matches_sample_backbone,
    )?;
    out.set_item("reported_backbone_doc_count", reported_backbone_doc_count)?;
    match actual_filtered_doc_count {
        Some(count) => {
            out.set_item("actual_filtered_doc_count", count)?;
            out.set_item("denominator_gap", reported_backbone_doc_count - count)?;
        }
        None => {
            out.set_item("actual_filtered_doc_count", py.None())?;
            out.set_item("denominator_gap", py.None())?;
        }
    }
    match sample_backbone_filtered_doc_count {
        Some(count) => {
            out.set_item("sample_backbone_filtered_doc_count", count)?;
            out.set_item(
                "sample_backbone_denominator_gap",
                reported_backbone_doc_count - count,
            )?;
        }
        None => {
            out.set_item("sample_backbone_filtered_doc_count", py.None())?;
            out.set_item("sample_backbone_denominator_gap", py.None())?;
        }
    }
    out.set_item("reported_covered_doc_count", reported_covered_doc_count)?;
    out.set_item("actual_covered_doc_count", actual_covered_doc_count)?;
    Ok(out.into_py(py))
}

#[pyfunction]
pub(crate) fn lm2011_validation_packet_a_mda_rows(
    py: Python<'_>,
    records: &Bound<'_, PyAny>,
    filing_year: i64,
    master_dictionary: &Bound<'_, PyAny>,
    threshold: i64,
    snippet_char_limit: i64,
) -> PyResult<Vec<PyObject>> {
    let master_words: HashSet<String> = master_dictionary
        .iter()?
        .filter_map(|value| {
            let value = value.ok()?;
            if value.is_none() {
                return None;
            }
            Some(value.str().ok()?.to_str().ok()?.to_string())
        })
        .collect();
    let mut out_rows: Vec<PyObject> = Vec::new();
    for record in records.iter()? {
        let record = record?;
        let dict = record
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("MDA row record is not a dict"))?;
        let doc_id_value = dict
            .get_item("doc_id")?
            .ok_or_else(|| PyValueError::new_err("MDA row record is missing doc_id"))?;
        let doc_id = doc_id_value.str()?.to_str()?.to_string();
        let text_value = dict.get_item("full_text")?;
        let full_text = text_value
            .as_ref()
            .filter(|value| value.is_instance_of::<PyString>())
            .and_then(|value| value.extract::<&str>().ok());

        let current_token_count = legacy_lm2011_tokens_impl(full_text).len() as i64;
        let appendix_tokens = tokenize_impl(full_text);
        let appendix_token_count = appendix_tokens.len() as i64;
        let recognized_word_count = appendix_tokens
            .iter()
            .filter(|token| master_words.contains(token.as_str()))
            .count() as i64;
        let current_threshold_pass = current_token_count >= threshold;
        let appendix_threshold_pass = appendix_token_count >= threshold;
        let recognized_threshold_pass = recognized_word_count >= threshold;
        let threshold_flip = current_threshold_pass != appendix_threshold_pass;
        let snippet = truncate_text_impl(full_text, snippet_char_limit);

        let out = PyDict::new_bound(py);
        out.set_item("text_scope", "mda")?;
        out.set_item("doc_id", doc_id)?;
        out.set_item("filing_year", filing_year)?;
        out.set_item("threshold", threshold)?;
        out.set_item("current_token_count", current_token_count)?;
        out.set_item("appendix_token_count", appendix_token_count)?;
        out.set_item("recognized_word_count", recognized_word_count)?;
        out.set_item("current_threshold_pass", current_threshold_pass)?;
        out.set_item("appendix_threshold_pass", appendix_threshold_pass)?;
        out.set_item("recognized_threshold_pass", recognized_threshold_pass)?;
        out.set_item("threshold_flip", threshold_flip)?;
        out.set_item("has_sec_header", false)?;
        out.set_item("has_html_marker", false)?;
        out.set_item("has_table_marker", false)?;
        out.set_item("has_exhibit_marker", false)?;
        out.set_item("any_marker", false)?;
        out.set_item("edgar_stripped_current_token_count", current_token_count)?;
        out.set_item("edgar_stripped_appendix_token_count", appendix_token_count)?;
        out.set_item(
            "edgar_stripped_recognized_word_count",
            recognized_word_count,
        )?;
        out.set_item("edgar_stripped_has_sec_header", false)?;
        out.set_item("edgar_stripped_has_html_marker", false)?;
        out.set_item("edgar_stripped_has_table_marker", false)?;
        out.set_item("edgar_stripped_has_exhibit_marker", false)?;
        out.set_item("edgar_stripped_any_marker", false)?;
        out.set_item("paper_cleaned_current_token_count", current_token_count)?;
        out.set_item("paper_cleaned_appendix_token_count", appendix_token_count)?;
        out.set_item("paper_cleaned_recognized_word_count", recognized_word_count)?;
        out.set_item(
            "paper_cleaned_current_threshold_pass",
            current_threshold_pass,
        )?;
        out.set_item(
            "paper_cleaned_appendix_threshold_pass",
            appendix_threshold_pass,
        )?;
        out.set_item(
            "paper_cleaned_recognized_threshold_pass",
            recognized_threshold_pass,
        )?;
        out.set_item("paper_cleaned_threshold_flip", threshold_flip)?;
        out.set_item("paper_cleaned_has_sec_header", false)?;
        out.set_item("paper_cleaned_has_html_marker", false)?;
        out.set_item("paper_cleaned_has_table_marker", false)?;
        out.set_item("paper_cleaned_has_exhibit_marker", false)?;
        out.set_item("paper_cleaned_any_marker", false)?;
        out.set_item("paper_cleaned_cut_reason", "no_tail_anchor")?;
        out.set_item("paper_cleaned_cut_start", py.None())?;
        out.set_item("paper_cleaned_cut_share", py.None())?;
        out.set_item("paper_cleaned_anchor_text", py.None())?;
        out.set_item("example_snippet", snippet.clone())?;
        out.set_item("edgar_stripped_example_snippet", snippet.clone())?;
        out.set_item("paper_cleaned_example_snippet", snippet.clone())?;
        out.set_item("paper_cleaned_pre_cut_tail_snippet", py.None())?;
        out.set_item("paper_cleaned_post_cut_tail_snippet", snippet)?;
        out_rows.push(out.into_py(py));
    }
    Ok(out_rows)
}

#[pyfunction]
pub(crate) fn lm2011_validation_packet_a_mda_row_columns(
    py: Python<'_>,
    column_names: Vec<String>,
    column_values: &Bound<'_, PyAny>,
    filing_year: i64,
    master_dictionary: &Bound<'_, PyAny>,
    threshold: i64,
    snippet_char_limit: i64,
) -> PyResult<Vec<PyObject>> {
    let label = "LM2011 validation packet A MDA rows";
    let (columns, row_count) =
        collect_pyobject_column_values(py, &column_names, column_values, label)?;
    let column_index = column_index_by_name(&column_names);
    let doc_id_idx = required_named_column_index(&column_index, label, "doc_id")?;
    let full_text_idx = required_named_column_index(&column_index, label, "full_text")?;
    let master_words: HashSet<String> = master_dictionary
        .iter()?
        .filter_map(|value| {
            let value = value.ok()?;
            if value.is_none() {
                return None;
            }
            Some(value.str().ok()?.to_str().ok()?.to_string())
        })
        .collect();
    let mut out_rows: Vec<PyObject> = Vec::new();

    for row_idx in 0..row_count {
        let doc_id = columns[doc_id_idx][row_idx]
            .bind(py)
            .str()?
            .to_str()?
            .to_string();
        let text_value = columns[full_text_idx][row_idx].bind(py);
        let full_text = if text_value.is_instance_of::<PyString>() {
            text_value.extract::<&str>().ok()
        } else {
            None
        };

        let current_token_count = legacy_lm2011_tokens_impl(full_text).len() as i64;
        let appendix_tokens = tokenize_impl(full_text);
        let appendix_token_count = appendix_tokens.len() as i64;
        let recognized_word_count = appendix_tokens
            .iter()
            .filter(|token| master_words.contains(token.as_str()))
            .count() as i64;
        let current_threshold_pass = current_token_count >= threshold;
        let appendix_threshold_pass = appendix_token_count >= threshold;
        let recognized_threshold_pass = recognized_word_count >= threshold;
        let threshold_flip = current_threshold_pass != appendix_threshold_pass;
        let snippet = truncate_text_impl(full_text, snippet_char_limit);

        let out = PyDict::new_bound(py);
        out.set_item("text_scope", "mda")?;
        out.set_item("doc_id", doc_id)?;
        out.set_item("filing_year", filing_year)?;
        out.set_item("threshold", threshold)?;
        out.set_item("current_token_count", current_token_count)?;
        out.set_item("appendix_token_count", appendix_token_count)?;
        out.set_item("recognized_word_count", recognized_word_count)?;
        out.set_item("current_threshold_pass", current_threshold_pass)?;
        out.set_item("appendix_threshold_pass", appendix_threshold_pass)?;
        out.set_item("recognized_threshold_pass", recognized_threshold_pass)?;
        out.set_item("threshold_flip", threshold_flip)?;
        out.set_item("has_sec_header", false)?;
        out.set_item("has_html_marker", false)?;
        out.set_item("has_table_marker", false)?;
        out.set_item("has_exhibit_marker", false)?;
        out.set_item("any_marker", false)?;
        out.set_item("edgar_stripped_current_token_count", current_token_count)?;
        out.set_item("edgar_stripped_appendix_token_count", appendix_token_count)?;
        out.set_item(
            "edgar_stripped_recognized_word_count",
            recognized_word_count,
        )?;
        out.set_item("edgar_stripped_has_sec_header", false)?;
        out.set_item("edgar_stripped_has_html_marker", false)?;
        out.set_item("edgar_stripped_has_table_marker", false)?;
        out.set_item("edgar_stripped_has_exhibit_marker", false)?;
        out.set_item("edgar_stripped_any_marker", false)?;
        out.set_item("paper_cleaned_current_token_count", current_token_count)?;
        out.set_item("paper_cleaned_appendix_token_count", appendix_token_count)?;
        out.set_item("paper_cleaned_recognized_word_count", recognized_word_count)?;
        out.set_item(
            "paper_cleaned_current_threshold_pass",
            current_threshold_pass,
        )?;
        out.set_item(
            "paper_cleaned_appendix_threshold_pass",
            appendix_threshold_pass,
        )?;
        out.set_item(
            "paper_cleaned_recognized_threshold_pass",
            recognized_threshold_pass,
        )?;
        out.set_item("paper_cleaned_threshold_flip", threshold_flip)?;
        out.set_item("paper_cleaned_has_sec_header", false)?;
        out.set_item("paper_cleaned_has_html_marker", false)?;
        out.set_item("paper_cleaned_has_table_marker", false)?;
        out.set_item("paper_cleaned_has_exhibit_marker", false)?;
        out.set_item("paper_cleaned_any_marker", false)?;
        out.set_item("paper_cleaned_cut_reason", "no_tail_anchor")?;
        out.set_item("paper_cleaned_cut_start", py.None())?;
        out.set_item("paper_cleaned_cut_share", py.None())?;
        out.set_item("paper_cleaned_anchor_text", py.None())?;
        out.set_item("example_snippet", snippet.clone())?;
        out.set_item("edgar_stripped_example_snippet", snippet.clone())?;
        out.set_item("paper_cleaned_example_snippet", snippet.clone())?;
        out.set_item("paper_cleaned_pre_cut_tail_snippet", py.None())?;
        out.set_item("paper_cleaned_post_cut_tail_snippet", snippet)?;
        out_rows.push(out.into_py(py));
    }
    Ok(out_rows)
}

#[derive(Default)]
pub(crate) struct PacketADeltaScopeAccum {
    doc_count: i64,
    current_vs_appendix_doc_count: i64,
    appendix_vs_recognized_doc_count: i64,
    current_minus_appendix_sum: f64,
    appendix_minus_recognized_sum: f64,
}

impl PacketADeltaScopeAccum {
    fn update(
        &mut self,
        current_token_count: i64,
        appendix_token_count: i64,
        recognized_word_count: i64,
    ) {
        self.doc_count += 1;
        if current_token_count != appendix_token_count {
            self.current_vs_appendix_doc_count += 1;
        }
        if appendix_token_count != recognized_word_count {
            self.appendix_vs_recognized_doc_count += 1;
        }
        self.current_minus_appendix_sum += (current_token_count - appendix_token_count) as f64;
        self.appendix_minus_recognized_sum += (appendix_token_count - recognized_word_count) as f64;
    }

    fn mean_current_minus_appendix(&self) -> f64 {
        if self.doc_count == 0 {
            0.0
        } else {
            self.current_minus_appendix_sum / self.doc_count as f64
        }
    }

    fn mean_appendix_minus_recognized(&self) -> f64 {
        if self.doc_count == 0 {
            0.0
        } else {
            self.appendix_minus_recognized_sum / self.doc_count as f64
        }
    }
}

pub(crate) fn dict_required_i64(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<i64> {
    let Some(value) = dict.get_item(key)? else {
        return Err(PyValueError::new_err(format!(
            "missing required key: {key}"
        )));
    };
    if value.is_none() {
        return Err(PyValueError::new_err(format!("null required key: {key}")));
    }
    py_int_like_to_i64(&value)
}

pub(crate) type PacketADeltaSummaryRow = (i64, i64, i64, f64, f64, i64, i64, i64, f64, f64);

#[pyfunction]
pub(crate) fn lm2011_validation_packet_a_delta_summary(
    rows: &Bound<'_, PyAny>,
) -> PyResult<PacketADeltaSummaryRow> {
    let mut full_10k = PacketADeltaScopeAccum::default();
    let mut mda = PacketADeltaScopeAccum::default();
    for row in rows.iter()? {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("packet A delta row is not a dict"))?;
        let scope = dict_required_string(dict, "text_scope")?;
        let target = match scope.as_str() {
            "full_10k" => &mut full_10k,
            "mda" => &mut mda,
            _ => continue,
        };
        target.update(
            dict_required_i64(dict, "current_token_count")?,
            dict_required_i64(dict, "appendix_token_count")?,
            dict_required_i64(dict, "recognized_word_count")?,
        );
    }
    Ok((
        full_10k.doc_count,
        full_10k.current_vs_appendix_doc_count,
        full_10k.appendix_vs_recognized_doc_count,
        full_10k.mean_current_minus_appendix(),
        full_10k.mean_appendix_minus_recognized(),
        mda.doc_count,
        mda.current_vs_appendix_doc_count,
        mda.appendix_vs_recognized_doc_count,
        mda.mean_current_minus_appendix(),
        mda.mean_appendix_minus_recognized(),
    ))
}

#[derive(Default, Clone)]
pub(crate) struct PacketASummaryAccum {
    doc_count: i64,
    docs_with_any_marker: i64,
    docs_with_any_marker_after_edgar_strip: i64,
    docs_with_any_marker_after_paper_cleaning: i64,
    docs_with_exhibit_marker_after_paper_cleaning: i64,
    current_threshold_pass_count: i64,
    appendix_threshold_pass_count: i64,
    recognized_threshold_pass_count: i64,
    threshold_flip_count: i64,
    token_drop_after_edgar_strip_sum: f64,
    token_drop_after_paper_cleaning_sum: f64,
    recognized_word_drop_after_paper_cleaning_sum: f64,
    paper_cleaned_threshold_flip_count: i64,
    paper_cleaned_appendix_threshold_pass_count: i64,
    paper_cleaned_truncated_doc_count: i64,
}

impl PacketASummaryAccum {
    fn update(&mut self, dict: &Bound<'_, PyDict>) -> PyResult<()> {
        self.doc_count += 1;
        self.docs_with_any_marker += i64::from(dict_truthy_bool(dict, "any_marker")?);
        self.docs_with_any_marker_after_edgar_strip +=
            i64::from(dict_truthy_bool(dict, "edgar_stripped_any_marker")?);
        self.docs_with_any_marker_after_paper_cleaning +=
            i64::from(dict_truthy_bool(dict, "paper_cleaned_any_marker")?);
        self.docs_with_exhibit_marker_after_paper_cleaning +=
            i64::from(dict_truthy_bool(dict, "paper_cleaned_has_exhibit_marker")?);
        self.current_threshold_pass_count +=
            i64::from(dict_truthy_bool(dict, "current_threshold_pass")?);
        self.appendix_threshold_pass_count +=
            i64::from(dict_truthy_bool(dict, "appendix_threshold_pass")?);
        self.recognized_threshold_pass_count +=
            i64::from(dict_truthy_bool(dict, "recognized_threshold_pass")?);
        self.threshold_flip_count += i64::from(dict_truthy_bool(dict, "threshold_flip")?);
        self.paper_cleaned_threshold_flip_count +=
            i64::from(dict_truthy_bool(dict, "paper_cleaned_threshold_flip")?);
        self.paper_cleaned_appendix_threshold_pass_count += i64::from(dict_truthy_bool(
            dict,
            "paper_cleaned_appendix_threshold_pass",
        )?);
        let cut_reason = dict_normalized_string(dict, "paper_cleaned_cut_reason")?;
        if cut_reason.as_deref() != Some("no_tail_anchor") {
            self.paper_cleaned_truncated_doc_count += 1;
        }
        self.token_drop_after_edgar_strip_sum += (dict_required_i64(dict, "current_token_count")?
            - dict_required_i64(dict, "edgar_stripped_current_token_count")?)
            as f64;
        self.token_drop_after_paper_cleaning_sum +=
            (dict_required_i64(dict, "edgar_stripped_current_token_count")?
                - dict_required_i64(dict, "paper_cleaned_current_token_count")?) as f64;
        self.recognized_word_drop_after_paper_cleaning_sum +=
            (dict_required_i64(dict, "edgar_stripped_recognized_word_count")?
                - dict_required_i64(dict, "paper_cleaned_recognized_word_count")?)
                as f64;
        Ok(())
    }

    fn mean_token_drop_after_edgar_strip(&self) -> f64 {
        if self.doc_count == 0 {
            0.0
        } else {
            self.token_drop_after_edgar_strip_sum / self.doc_count as f64
        }
    }

    fn mean_token_drop_after_paper_cleaning(&self) -> f64 {
        if self.doc_count == 0 {
            0.0
        } else {
            self.token_drop_after_paper_cleaning_sum / self.doc_count as f64
        }
    }

    fn mean_recognized_word_drop_after_paper_cleaning(&self) -> f64 {
        if self.doc_count == 0 {
            0.0
        } else {
            self.recognized_word_drop_after_paper_cleaning_sum / self.doc_count as f64
        }
    }
}

pub(crate) fn packet_a_summary_py_row(
    py: Python<'_>,
    text_scope: &str,
    filing_year: Option<i64>,
    accum: &PacketASummaryAccum,
) -> PyResult<PyObject> {
    let row = PyDict::new_bound(py);
    row.set_item("text_scope", text_scope)?;
    row.set_item("filing_year", filing_year)?;
    row.set_item("doc_count", accum.doc_count)?;
    row.set_item("docs_with_any_marker", accum.docs_with_any_marker)?;
    row.set_item(
        "docs_with_any_marker_after_edgar_strip",
        accum.docs_with_any_marker_after_edgar_strip,
    )?;
    row.set_item(
        "docs_with_any_marker_after_paper_cleaning",
        accum.docs_with_any_marker_after_paper_cleaning,
    )?;
    row.set_item(
        "docs_with_exhibit_marker_after_paper_cleaning",
        accum.docs_with_exhibit_marker_after_paper_cleaning,
    )?;
    row.set_item(
        "current_threshold_pass_count",
        accum.current_threshold_pass_count,
    )?;
    row.set_item(
        "appendix_threshold_pass_count",
        accum.appendix_threshold_pass_count,
    )?;
    row.set_item(
        "recognized_threshold_pass_count",
        accum.recognized_threshold_pass_count,
    )?;
    row.set_item("threshold_flip_count", accum.threshold_flip_count)?;
    row.set_item(
        "mean_token_drop_after_edgar_strip",
        accum.mean_token_drop_after_edgar_strip(),
    )?;
    row.set_item(
        "mean_token_drop_after_paper_cleaning",
        accum.mean_token_drop_after_paper_cleaning(),
    )?;
    row.set_item(
        "mean_recognized_word_drop_after_paper_cleaning",
        accum.mean_recognized_word_drop_after_paper_cleaning(),
    )?;
    row.set_item(
        "paper_cleaned_threshold_flip_count",
        accum.paper_cleaned_threshold_flip_count,
    )?;
    row.set_item(
        "paper_cleaned_appendix_threshold_pass_count",
        accum.paper_cleaned_appendix_threshold_pass_count,
    )?;
    row.set_item(
        "paper_cleaned_truncated_doc_count",
        accum.paper_cleaned_truncated_doc_count,
    )?;
    Ok(row.into_py(py))
}

#[pyfunction]
pub(crate) fn lm2011_validation_packet_a_summary_rows(
    py: Python<'_>,
    rows: &Bound<'_, PyAny>,
) -> PyResult<Vec<PyObject>> {
    let mut per_year: BTreeMap<(String, i64), PacketASummaryAccum> = BTreeMap::new();
    let mut overall: BTreeMap<String, PacketASummaryAccum> = BTreeMap::new();
    for row in rows.iter()? {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("packet A summary row is not a dict"))?;
        let text_scope = dict_required_string(dict, "text_scope")?;
        let filing_year = dict_required_i64(dict, "filing_year")?;
        per_year
            .entry((text_scope.clone(), filing_year))
            .or_default()
            .update(dict)?;
        overall.entry(text_scope).or_default().update(dict)?;
    }

    let mut output: Vec<PyObject> = Vec::with_capacity(per_year.len() + overall.len());
    for ((text_scope, filing_year), accum) in &per_year {
        output.push(packet_a_summary_py_row(
            py,
            text_scope,
            Some(*filing_year),
            accum,
        )?);
    }
    for (text_scope, accum) in &overall {
        output.push(packet_a_summary_py_row(py, text_scope, None, accum)?);
    }
    Ok(output)
}

#[derive(Default, Clone)]
pub(crate) struct PacketAStripAccum {
    doc_count: i64,
    docs_with_any_marker_before: i64,
    docs_with_any_marker_after_edgar_strip: i64,
    docs_with_any_marker_after_paper_cleaning: i64,
    docs_with_sec_header_after_edgar_strip: i64,
    docs_with_html_marker_after_edgar_strip: i64,
    docs_with_table_marker_after_edgar_strip: i64,
    docs_with_exhibit_marker_after_edgar_strip: i64,
    docs_with_sec_header_after_paper_cleaning: i64,
    docs_with_html_marker_after_paper_cleaning: i64,
    docs_with_table_marker_after_paper_cleaning: i64,
    docs_with_exhibit_marker_after_paper_cleaning: i64,
    truncated_doc_count: i64,
}

impl PacketAStripAccum {
    fn update(&mut self, dict: &Bound<'_, PyDict>) -> PyResult<()> {
        self.doc_count += 1;
        self.docs_with_any_marker_before += i64::from(dict_truthy_bool(dict, "any_marker")?);
        self.docs_with_any_marker_after_edgar_strip +=
            i64::from(dict_truthy_bool(dict, "edgar_stripped_any_marker")?);
        self.docs_with_any_marker_after_paper_cleaning +=
            i64::from(dict_truthy_bool(dict, "paper_cleaned_any_marker")?);
        self.docs_with_sec_header_after_edgar_strip +=
            i64::from(dict_truthy_bool(dict, "edgar_stripped_has_sec_header")?);
        self.docs_with_html_marker_after_edgar_strip +=
            i64::from(dict_truthy_bool(dict, "edgar_stripped_has_html_marker")?);
        self.docs_with_table_marker_after_edgar_strip +=
            i64::from(dict_truthy_bool(dict, "edgar_stripped_has_table_marker")?);
        self.docs_with_exhibit_marker_after_edgar_strip +=
            i64::from(dict_truthy_bool(dict, "edgar_stripped_has_exhibit_marker")?);
        self.docs_with_sec_header_after_paper_cleaning +=
            i64::from(dict_truthy_bool(dict, "paper_cleaned_has_sec_header")?);
        self.docs_with_html_marker_after_paper_cleaning +=
            i64::from(dict_truthy_bool(dict, "paper_cleaned_has_html_marker")?);
        self.docs_with_table_marker_after_paper_cleaning +=
            i64::from(dict_truthy_bool(dict, "paper_cleaned_has_table_marker")?);
        self.docs_with_exhibit_marker_after_paper_cleaning +=
            i64::from(dict_truthy_bool(dict, "paper_cleaned_has_exhibit_marker")?);
        let cut_reason = dict_normalized_string(dict, "paper_cleaned_cut_reason")?;
        if cut_reason.as_deref() != Some("no_tail_anchor") {
            self.truncated_doc_count += 1;
        }
        Ok(())
    }
}

pub(crate) fn packet_a_strip_py_row(
    py: Python<'_>,
    text_scope: &str,
    filing_year: Option<i64>,
    accum: &PacketAStripAccum,
) -> PyResult<PyObject> {
    let row = PyDict::new_bound(py);
    row.set_item("text_scope", text_scope)?;
    row.set_item("filing_year", filing_year)?;
    row.set_item("doc_count", accum.doc_count)?;
    row.set_item(
        "docs_with_any_marker_before",
        accum.docs_with_any_marker_before,
    )?;
    row.set_item(
        "docs_with_any_marker_after_edgar_strip",
        accum.docs_with_any_marker_after_edgar_strip,
    )?;
    row.set_item(
        "docs_with_any_marker_after_paper_cleaning",
        accum.docs_with_any_marker_after_paper_cleaning,
    )?;
    row.set_item(
        "docs_with_sec_header_after_edgar_strip",
        accum.docs_with_sec_header_after_edgar_strip,
    )?;
    row.set_item(
        "docs_with_html_marker_after_edgar_strip",
        accum.docs_with_html_marker_after_edgar_strip,
    )?;
    row.set_item(
        "docs_with_table_marker_after_edgar_strip",
        accum.docs_with_table_marker_after_edgar_strip,
    )?;
    row.set_item(
        "docs_with_exhibit_marker_after_edgar_strip",
        accum.docs_with_exhibit_marker_after_edgar_strip,
    )?;
    row.set_item(
        "docs_with_sec_header_after_paper_cleaning",
        accum.docs_with_sec_header_after_paper_cleaning,
    )?;
    row.set_item(
        "docs_with_html_marker_after_paper_cleaning",
        accum.docs_with_html_marker_after_paper_cleaning,
    )?;
    row.set_item(
        "docs_with_table_marker_after_paper_cleaning",
        accum.docs_with_table_marker_after_paper_cleaning,
    )?;
    row.set_item(
        "docs_with_exhibit_marker_after_paper_cleaning",
        accum.docs_with_exhibit_marker_after_paper_cleaning,
    )?;
    row.set_item("truncated_doc_count", accum.truncated_doc_count)?;
    Ok(row.into_py(py))
}

#[pyfunction]
pub(crate) fn lm2011_validation_packet_a_strip_comparison_rows(
    py: Python<'_>,
    rows: &Bound<'_, PyAny>,
) -> PyResult<Vec<PyObject>> {
    let mut per_year: BTreeMap<(String, i64), PacketAStripAccum> = BTreeMap::new();
    let mut overall: BTreeMap<String, PacketAStripAccum> = BTreeMap::new();
    for row in rows.iter()? {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("packet A strip row is not a dict"))?;
        let text_scope = dict_required_string(dict, "text_scope")?;
        let filing_year = dict_required_i64(dict, "filing_year")?;
        per_year
            .entry((text_scope.clone(), filing_year))
            .or_default()
            .update(dict)?;
        overall.entry(text_scope).or_default().update(dict)?;
    }

    let mut output: Vec<PyObject> = Vec::with_capacity(per_year.len() + overall.len());
    for ((text_scope, filing_year), accum) in &per_year {
        output.push(packet_a_strip_py_row(
            py,
            text_scope,
            Some(*filing_year),
            accum,
        )?);
    }
    for (text_scope, accum) in &overall {
        output.push(packet_a_strip_py_row(py, text_scope, None, accum)?);
    }
    Ok(output)
}

#[derive(Clone)]
pub(crate) struct PacketAExampleRow {
    text_scope: String,
    doc_id: String,
    filing_year: i64,
    threshold_flip: bool,
    any_marker: bool,
    edgar_stripped_any_marker: bool,
    paper_cleaned_any_marker: bool,
    paper_cleaned_threshold_flip: bool,
    paper_cleaned_cut_reason: String,
    paper_cleaned_cut_share: Option<f64>,
    example_snippet: Option<String>,
    edgar_stripped_example_snippet: Option<String>,
    paper_cleaned_example_snippet: Option<String>,
    paper_cleaned_pre_cut_tail_snippet: Option<String>,
    paper_cleaned_post_cut_tail_snippet: Option<String>,
}

impl PacketAExampleRow {
    fn sort_key(&self) -> (i64, i64, i64, i64, i64, i64, &str, &str) {
        (
            if self.paper_cleaned_cut_reason != "no_tail_anchor" {
                0
            } else {
                1
            },
            if self.threshold_flip { 0 } else { 1 },
            if self.paper_cleaned_threshold_flip {
                0
            } else {
                1
            },
            if self.any_marker { 0 } else { 1 },
            if self.edgar_stripped_any_marker { 0 } else { 1 },
            if self.paper_cleaned_any_marker { 0 } else { 1 },
            self.text_scope.as_str(),
            self.doc_id.as_str(),
        )
    }
}

pub(crate) fn packet_a_example_row_from_dict(
    dict: &Bound<'_, PyDict>,
) -> PyResult<Option<PacketAExampleRow>> {
    let threshold_flip = dict_truthy_bool(dict, "threshold_flip")?;
    let paper_cleaned_threshold_flip = dict_truthy_bool(dict, "paper_cleaned_threshold_flip")?;
    let any_marker = dict_truthy_bool(dict, "any_marker")?;
    let edgar_stripped_any_marker = dict_truthy_bool(dict, "edgar_stripped_any_marker")?;
    let paper_cleaned_any_marker = dict_truthy_bool(dict, "paper_cleaned_any_marker")?;
    let paper_cleaned_cut_reason = dict_required_string(dict, "paper_cleaned_cut_reason")?;
    if !threshold_flip
        && !paper_cleaned_threshold_flip
        && !any_marker
        && !edgar_stripped_any_marker
        && !paper_cleaned_any_marker
        && paper_cleaned_cut_reason == "no_tail_anchor"
    {
        return Ok(None);
    }
    Ok(Some(PacketAExampleRow {
        text_scope: dict_required_string(dict, "text_scope")?,
        doc_id: dict_required_string(dict, "doc_id")?,
        filing_year: dict_required_i64(dict, "filing_year")?,
        threshold_flip,
        any_marker,
        edgar_stripped_any_marker,
        paper_cleaned_any_marker,
        paper_cleaned_threshold_flip,
        paper_cleaned_cut_reason,
        paper_cleaned_cut_share: dict_optional_float(dict, "paper_cleaned_cut_share")?,
        example_snippet: dict_raw_string(dict, "example_snippet")?,
        edgar_stripped_example_snippet: dict_raw_string(dict, "edgar_stripped_example_snippet")?,
        paper_cleaned_example_snippet: dict_raw_string(dict, "paper_cleaned_example_snippet")?,
        paper_cleaned_pre_cut_tail_snippet: dict_raw_string(
            dict,
            "paper_cleaned_pre_cut_tail_snippet",
        )?,
        paper_cleaned_post_cut_tail_snippet: dict_raw_string(
            dict,
            "paper_cleaned_post_cut_tail_snippet",
        )?,
    }))
}

pub(crate) fn packet_a_example_py_row(
    py: Python<'_>,
    row: &PacketAExampleRow,
) -> PyResult<PyObject> {
    let out = PyDict::new_bound(py);
    out.set_item("text_scope", &row.text_scope)?;
    out.set_item("doc_id", &row.doc_id)?;
    out.set_item("filing_year", row.filing_year)?;
    out.set_item("threshold_flip", row.threshold_flip)?;
    out.set_item("any_marker", row.any_marker)?;
    out.set_item("edgar_stripped_any_marker", row.edgar_stripped_any_marker)?;
    out.set_item("paper_cleaned_any_marker", row.paper_cleaned_any_marker)?;
    out.set_item(
        "paper_cleaned_threshold_flip",
        row.paper_cleaned_threshold_flip,
    )?;
    out.set_item("paper_cleaned_cut_reason", &row.paper_cleaned_cut_reason)?;
    out.set_item("paper_cleaned_cut_share", row.paper_cleaned_cut_share)?;
    out.set_item("example_snippet", &row.example_snippet)?;
    out.set_item(
        "edgar_stripped_example_snippet",
        &row.edgar_stripped_example_snippet,
    )?;
    out.set_item(
        "paper_cleaned_example_snippet",
        &row.paper_cleaned_example_snippet,
    )?;
    out.set_item(
        "paper_cleaned_pre_cut_tail_snippet",
        &row.paper_cleaned_pre_cut_tail_snippet,
    )?;
    out.set_item(
        "paper_cleaned_post_cut_tail_snippet",
        &row.paper_cleaned_post_cut_tail_snippet,
    )?;
    Ok(out.into_py(py))
}

#[pyfunction]
pub(crate) fn lm2011_validation_packet_a_example_rows(
    py: Python<'_>,
    rows: &Bound<'_, PyAny>,
    max_rows: i64,
) -> PyResult<Vec<PyObject>> {
    let mut examples: Vec<PacketAExampleRow> = Vec::new();
    for row in rows.iter()? {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("packet A example row is not a dict"))?;
        if let Some(example) = packet_a_example_row_from_dict(dict)? {
            examples.push(example);
        }
    }
    examples.sort_by(|left, right| left.sort_key().cmp(&right.sort_key()));
    let retained_len = if max_rows >= 0 {
        usize::try_from(max_rows)
            .unwrap_or(usize::MAX)
            .min(examples.len())
    } else {
        let drop_count = usize::try_from(max_rows.saturating_abs()).unwrap_or(usize::MAX);
        examples.len().saturating_sub(drop_count)
    };
    examples
        .iter()
        .take(retained_len)
        .map(|row| packet_a_example_py_row(py, row))
        .collect()
}

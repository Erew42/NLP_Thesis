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
#[pyo3(signature = (text=None))]
pub(crate) fn tokenize_lm2011_text(text: Option<&str>) -> Vec<String> {
    tokenize_impl(text)
}

#[pyfunction]
#[pyo3(signature = (text=None))]
pub(crate) fn count_lm2011_text_tokens(text: Option<&str>) -> usize {
    count_tokens_impl(text)
}

#[pyfunction]
pub(crate) fn count_lm2011_text_token_values(values: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    if values.is_none() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for value in values.iter()? {
        let value = value?;
        if value.is_none() {
            out.push(count_tokens_impl(None));
        } else {
            let rendered = value.str()?;
            out.push(count_tokens_impl(Some(rendered.to_str()?)));
        }
    }
    Ok(out)
}

pub(crate) fn optional_string_py_object(py: Python<'_>, value: &Option<String>) -> PyObject {
    match value {
        Some(value) => value.clone().into_py(py),
        None => py.None(),
    }
}

#[pyfunction]
pub(crate) fn validate_lm2011_incremental_text_doc_ids(
    py: Python<'_>,
    doc_ids: &Bound<'_, PyAny>,
    seen_doc_ids: &Bound<'_, PyAny>,
    duplicate_limit: i64,
) -> PyResult<(Vec<PyObject>, Vec<PyObject>)> {
    let mut seen: HashSet<Option<String>> = HashSet::new();
    for value in seen_doc_ids.iter()? {
        seen.insert(optional_doc_id_text(&value?)?);
    }

    let mut batch_seen: HashSet<Option<String>> = HashSet::new();
    let mut batch_seen_order: Vec<Option<String>> = Vec::new();
    let mut duplicate_seen: HashSet<Option<String>> = HashSet::new();
    let mut duplicate_doc_ids: Vec<Option<String>> = Vec::new();
    let limit = duplicate_limit.max(0) as usize;

    for value in doc_ids.iter()? {
        let doc_id = optional_doc_id_text(&value?)?;
        if (batch_seen.contains(&doc_id) || seen.contains(&doc_id))
            && !duplicate_seen.contains(&doc_id)
            && duplicate_doc_ids.len() < limit
        {
            duplicate_seen.insert(doc_id.clone());
            duplicate_doc_ids.push(doc_id.clone());
        }
        if batch_seen.insert(doc_id.clone()) {
            batch_seen_order.push(doc_id);
        }
    }

    Ok((
        duplicate_doc_ids
            .iter()
            .map(|value| optional_string_py_object(py, value))
            .collect(),
        batch_seen_order
            .iter()
            .map(|value| optional_string_py_object(py, value))
            .collect(),
    ))
}

#[pyfunction]
pub(crate) fn normalize_lm2011_dictionary_tokens(
    values: &Bound<'_, PyAny>,
) -> PyResult<Vec<String>> {
    if values.is_none() {
        return Ok(Vec::new());
    }
    let mut tokens = BTreeSet::new();
    for value in values.iter()? {
        let value = value?;
        if value.is_none() {
            continue;
        }
        let rendered = value.str()?;
        scan_tokens_impl(Some(rendered.to_str()?), |token| {
            tokens.insert(token);
        });
    }
    Ok(tokens.into_iter().collect())
}

#[pyfunction]
#[pyo3(signature = (value=None, other_value=Some("Other")))]
pub(crate) fn normalize_lm2011_form_value(
    value: Option<&str>,
    other_value: Option<&str>,
) -> Option<String> {
    let cleaned = clean_form_token_impl(value)?;
    canonical_lm2011_form(&cleaned)
        .map(str::to_string)
        .or_else(|| other_value.map(str::to_string))
}

#[pyfunction]
#[pyo3(signature = (values, other_value=Some("Other")))]
pub(crate) fn normalize_lm2011_form_values(
    values: &Bound<'_, PyAny>,
    other_value: Option<&str>,
) -> PyResult<Vec<Option<String>>> {
    let mut out = Vec::new();
    for value in values.iter()? {
        let value = value?;
        if value.is_none() {
            out.push(None);
        } else {
            let rendered = value.str()?;
            out.push(normalize_lm2011_form_value(
                Some(rendered.to_str()?),
                other_value,
            ));
        }
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (value=None))]
pub(crate) fn normalize_sec_raw_form_value(value: Option<&str>) -> Option<String> {
    normalize_sec_raw_form_impl(value)
}

#[pyfunction]
pub(crate) fn normalize_sec_raw_form_values(
    values: &Bound<'_, PyAny>,
) -> PyResult<Vec<Option<String>>> {
    let mut out = Vec::new();
    for value in values.iter()? {
        let value = value?;
        if value.is_none() {
            out.push(None);
        } else {
            let rendered = value.str()?;
            out.push(normalize_sec_raw_form_impl(Some(rendered.to_str()?)));
        }
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (value=None))]
pub(crate) fn normalize_ccm_raw_form_value(value: Option<&str>) -> Option<String> {
    normalize_ccm_raw_form_impl(value)
}

#[pyfunction]
pub(crate) fn normalize_ccm_raw_form_values(
    values: &Bound<'_, PyAny>,
) -> PyResult<Vec<Option<String>>> {
    let mut out = Vec::new();
    for value in values.iter()? {
        let value = value?;
        if value.is_none() {
            out.push(None);
        } else {
            let rendered = value.str()?;
            out.push(normalize_ccm_raw_form_impl(Some(rendered.to_str()?)));
        }
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (value=None))]
pub(crate) fn ccm_form_match_token_value(value: Option<&str>) -> Option<String> {
    ccm_form_match_token_impl(value)
}

#[pyfunction]
#[pyo3(signature = (value=None))]
pub(crate) fn normalize_year_filter_values(
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<Vec<i64>>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    let mut years = BTreeSet::new();
    for item in value.iter()? {
        let year = py_int_like_to_i64(&item?)?;
        if year < 0 {
            return Err(PyValueError::new_err(format!(
                "year_filter values must be positive integers, got {year}."
            )));
        }
        years.insert(year);
    }
    Ok(Some(years.into_iter().collect()))
}

#[pyfunction]
#[pyo3(signature = (value=None))]
pub(crate) fn normalize_lookup_text_value(value: Option<&str>) -> Option<String> {
    normalize_lookup_text_impl(value)
}

#[pyfunction]
#[pyo3(signature = (value=None))]
pub(crate) fn lm2011_dictionary_normalize_cell_value(value: Option<&str>) -> String {
    lm2011_dictionary_normalize_cell_impl(value)
}

#[pyfunction]
#[pyo3(signature = (value=None))]
pub(crate) fn lm2011_dictionary_active_membership_value(value: Option<&str>) -> bool {
    lm2011_dictionary_active_membership_impl(value)
}

#[pyfunction]
pub(crate) fn lm2011_dictionary_unique_preserve_order(values: Vec<String>) -> Vec<String> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut unique_values = Vec::with_capacity(values.len());
    for value in values {
        if seen.insert(value.clone()) {
            unique_values.push(value);
        }
    }
    unique_values
}

#[pyfunction]
#[pyo3(signature = (value=None))]
pub(crate) fn finite_float_or_none_value(
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<f64>> {
    let Some(value) = value else {
        return Ok(None);
    };
    py_float_like_to_finite_option(value)
}

#[pyfunction]
#[pyo3(signature = (value=None))]
pub(crate) fn normalize_lookup_text_any_value(
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<String>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    py_str_normalized(value)
}

#[pyfunction]
pub(crate) fn normalize_lookup_text_any_values(
    values: &Bound<'_, PyAny>,
) -> PyResult<Vec<Option<String>>> {
    let mut out = Vec::new();
    for value in values.iter()? {
        let value = value?;
        out.push(normalize_lookup_text_any_impl(Some(&value))?);
    }
    Ok(out)
}

pub(crate) fn normalize_lookup_text_any_impl(
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<String>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    py_str_normalized(value)
}

#[pyfunction]
#[pyo3(signature = (value=None))]
pub(crate) fn normalize_workbook_scalar_value(
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<String>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    if let Ok(date_method) = value.getattr("date") {
        if let Ok(date_value) = date_method.call0() {
            return py_str_normalized(&date_value);
        }
    }
    py_str_normalized(value)
}

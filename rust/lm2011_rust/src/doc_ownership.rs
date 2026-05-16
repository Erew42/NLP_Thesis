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
#[pyo3(signature = (value=None))]
pub(crate) fn normalize_kypermno_value(
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<String>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    if let Ok(text) = value.extract::<&str>() {
        return Ok(normalize_lookup_text_impl(Some(text)));
    }
    if value.is_instance_of::<PyBool>() || value.is_instance_of::<PyInt>() {
        return py_str_normalized(value);
    }
    if value.is_instance_of::<PyFloat>() {
        let float_value = value.extract::<f64>()?;
        if !float_value.is_finite() {
            return Err(PyValueError::new_err(
                "cannot normalize non-finite KYPERMNO float",
            ));
        }
        if float_value.trunc() == float_value {
            let int_obj = value
                .py()
                .import_bound("builtins")?
                .getattr("int")?
                .call1((value,))?;
            return py_str_normalized(&int_obj);
        }
        return py_str_normalized(value);
    }
    py_str_normalized(value)
}

#[pyfunction]
pub(crate) fn normalize_kypermno_values(
    values: &Bound<'_, PyAny>,
) -> PyResult<Vec<Option<String>>> {
    let mut out = Vec::new();
    for value in values.iter()? {
        let value = value?;
        out.push(normalize_kypermno_value(Some(&value))?);
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (value=None))]
pub(crate) fn normalize_doc_ownership_float_value(
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<f64>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    if value.is_instance_of::<PyBool>() {
        return Ok(Some(if value.extract::<bool>()? { 1.0 } else { 0.0 }));
    }
    if value.is_instance_of::<PyInt>() || value.is_instance_of::<PyFloat>() {
        return Ok(Some(value.extract::<f64>()?));
    }
    let Some(normalized) = py_str_normalized(value)? else {
        return Ok(None);
    };
    Ok(normalized.parse::<f64>().ok())
}

pub(crate) fn py_int_from_f64(py: Python<'_>, value: f64) -> PyResult<PyObject> {
    Ok(py
        .import_bound("builtins")?
        .getattr("int")?
        .call1((value,))?
        .into_py(py))
}

#[pyfunction]
#[pyo3(signature = (value=None))]
pub(crate) fn normalize_doc_ownership_int_value(
    py: Python<'_>,
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<PyObject>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    if value.is_instance_of::<PyBool>() {
        return Ok(Some(py_int_from_f64(
            py,
            if value.extract::<bool>()? { 1.0 } else { 0.0 },
        )?));
    }
    if value.is_instance_of::<PyInt>() || value.is_instance_of::<PyFloat>() {
        return Ok(Some(py_int_from_f64(py, value.extract::<f64>()?)?));
    }
    let Some(normalized) = py_str_normalized(value)? else {
        return Ok(None);
    };
    let Some(float_value) = normalized.parse::<f64>().ok() else {
        return Ok(None);
    };
    Ok(Some(py_int_from_f64(py, float_value)?))
}

#[pyfunction]
#[pyo3(signature = (value=None))]
pub(crate) fn normalize_doc_ownership_date_value(
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
    let rendered = value.str()?;
    let Some((year, month, day)) = parse_doc_ownership_date_text(rendered.to_str()?) else {
        return Ok(None);
    };
    Ok(Some(py_date_object(py, year, month, day)?))
}

#[pyfunction]
pub(crate) fn doc_ownership_most_recent_quarter_end_before(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let (year, month, _) = extract_py_date_parts(value)?;
    let quarter_start_month = ((month - 1) / 3) * 3 + 1;
    if quarter_start_month == 1 {
        return py_date_object(py, year - 1, 12, 31);
    }
    let previous_month = quarter_start_month - 1;
    let Some(previous_day) = days_in_month(year, previous_month) else {
        return Err(PyValueError::new_err("date month out of range"));
    };
    py_date_object(py, year, previous_month, previous_day)
}

#[pyfunction]
pub(crate) fn doc_ownership_target_effective_date_for_quarter_end(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let (mut year, mut month, mut day) = extract_py_date_parts(value)?;
    let Some(max_day) = days_in_month(year, month) else {
        return Err(PyValueError::new_err("date month out of range"));
    };
    if day < max_day {
        day += 1;
    } else {
        day = 1;
        if month == 12 {
            year += 1;
            month = 1;
        } else {
            month += 1;
        }
    }
    py_date_object(py, year, month, day)
}

pub(crate) fn optional_date_parts(
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<(i32, u32, u32)>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    Ok(Some(extract_py_date_parts(value)?))
}

#[pyfunction]
#[pyo3(signature = (value=None, lower_bound=None))]
pub(crate) fn doc_ownership_clip_date_lower(
    py: Python<'_>,
    value: Option<&Bound<'_, PyAny>>,
    lower_bound: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<PyObject>> {
    let Some(value_parts) = optional_date_parts(value)? else {
        return Ok(None);
    };
    let Some(lower_parts) = optional_date_parts(lower_bound)? else {
        return Ok(Some(py_date_object(
            py,
            value_parts.0,
            value_parts.1,
            value_parts.2,
        )?));
    };
    let value_ordinal = date_ordinal(value_parts.0, value_parts.1, value_parts.2)?;
    let lower_ordinal = date_ordinal(lower_parts.0, lower_parts.1, lower_parts.2)?;
    let selected = if value_ordinal < lower_ordinal {
        lower_parts
    } else {
        value_parts
    };
    Ok(Some(py_date_object(
        py, selected.0, selected.1, selected.2,
    )?))
}

#[pyfunction]
#[pyo3(signature = (value=None, upper_bound=None))]
pub(crate) fn doc_ownership_clip_date_upper(
    py: Python<'_>,
    value: Option<&Bound<'_, PyAny>>,
    upper_bound: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<PyObject>> {
    let Some(value_parts) = optional_date_parts(value)? else {
        return Ok(None);
    };
    let Some(upper_parts) = optional_date_parts(upper_bound)? else {
        return Ok(Some(py_date_object(
            py,
            value_parts.0,
            value_parts.1,
            value_parts.2,
        )?));
    };
    let value_ordinal = date_ordinal(value_parts.0, value_parts.1, value_parts.2)?;
    let upper_ordinal = date_ordinal(upper_parts.0, upper_parts.1, upper_parts.2)?;
    let selected = if value_ordinal > upper_ordinal {
        upper_parts
    } else {
        value_parts
    };
    Ok(Some(py_date_object(
        py, selected.0, selected.1, selected.2,
    )?))
}

#[pyfunction]
pub(crate) fn doc_ownership_matching_retrieval_sheets(
    sheet_names: &Bound<'_, PyAny>,
    prefix: &str,
) -> PyResult<Vec<String>> {
    let mut out: Vec<String> = Vec::new();
    let prefix_with_separator = format!("{prefix}_");
    for entry in sheet_names.iter()? {
        let sheet_name = entry?.extract::<String>()?;
        if sheet_name == prefix || sheet_name.starts_with(&prefix_with_separator) {
            out.push(sheet_name);
        }
    }
    out.sort_unstable();
    Ok(out)
}

pub(crate) type DocOwnershipDateParts = (i32, u32, u32);

pub(crate) fn doc_ownership_date_parts_from_dict(
    dict: &Bound<'_, PyDict>,
    key: &str,
) -> PyResult<Option<DocOwnershipDateParts>> {
    let Some(value) = dict.get_item(key)? else {
        return Ok(None);
    };
    let Some(date_text) = py_any_date_iso_string(&value)? else {
        return Ok(None);
    };
    Ok(parse_doc_ownership_date_text(&date_text))
}

pub(crate) fn doc_ownership_date_parts_to_py(
    py: Python<'_>,
    value: Option<DocOwnershipDateParts>,
) -> PyResult<Option<PyObject>> {
    let Some((year, month, day)) = value else {
        return Ok(None);
    };
    Ok(Some(py_date_object(py, year, month, day)?))
}

pub(crate) fn doc_ownership_date_parts_ordinal(value: DocOwnershipDateParts) -> PyResult<i64> {
    date_ordinal(value.0, value.1, value.2)
}

pub(crate) fn doc_ownership_date_parts_leq(
    left: DocOwnershipDateParts,
    right: DocOwnershipDateParts,
) -> PyResult<bool> {
    Ok(doc_ownership_date_parts_ordinal(left)? <= doc_ownership_date_parts_ordinal(right)?)
}

pub(crate) fn doc_ownership_date_parts_lt(
    left: DocOwnershipDateParts,
    right: DocOwnershipDateParts,
) -> PyResult<bool> {
    Ok(doc_ownership_date_parts_ordinal(left)? < doc_ownership_date_parts_ordinal(right)?)
}

pub(crate) fn doc_ownership_date_parts_gt(
    left: DocOwnershipDateParts,
    right: DocOwnershipDateParts,
) -> PyResult<bool> {
    Ok(doc_ownership_date_parts_ordinal(left)? > doc_ownership_date_parts_ordinal(right)?)
}

pub(crate) fn doc_ownership_most_recent_quarter_end_parts(
    value: DocOwnershipDateParts,
) -> PyResult<DocOwnershipDateParts> {
    let (year, month, _) = value;
    let quarter_start_month = ((month - 1) / 3) * 3 + 1;
    if quarter_start_month == 1 {
        return Ok((year - 1, 12, 31));
    }
    let previous_month = quarter_start_month - 1;
    let Some(previous_day) = days_in_month(year, previous_month) else {
        return Err(PyValueError::new_err("date month out of range"));
    };
    Ok((year, previous_month, previous_day))
}

pub(crate) fn doc_ownership_target_effective_date_parts(
    value: DocOwnershipDateParts,
) -> PyResult<DocOwnershipDateParts> {
    let (mut year, mut month, mut day) = value;
    let Some(max_day) = days_in_month(year, month) else {
        return Err(PyValueError::new_err("date month out of range"));
    };
    if day < max_day {
        day += 1;
    } else {
        day = 1;
        if month == 12 {
            year += 1;
            month = 1;
        } else {
            month += 1;
        }
    }
    Ok((year, month, day))
}

pub(crate) fn doc_ownership_add_days(
    mut value: DocOwnershipDateParts,
    days: u32,
) -> PyResult<DocOwnershipDateParts> {
    for _ in 0..days {
        let Some(max_day) = days_in_month(value.0, value.1) else {
            return Err(PyValueError::new_err("date month out of range"));
        };
        if value.2 < max_day {
            value.2 += 1;
        } else {
            value.2 = 1;
            if value.1 == 12 {
                value.0 += 1;
                value.1 = 1;
            } else {
                value.1 += 1;
            }
        }
    }
    Ok(value)
}

pub(crate) fn doc_ownership_min_date_parts(
    left: DocOwnershipDateParts,
    right: DocOwnershipDateParts,
) -> PyResult<DocOwnershipDateParts> {
    if doc_ownership_date_parts_leq(left, right)? {
        Ok(left)
    } else {
        Ok(right)
    }
}

pub(crate) fn doc_ownership_clip_lower_parts(
    value: Option<DocOwnershipDateParts>,
    lower_bound: Option<DocOwnershipDateParts>,
) -> PyResult<Option<DocOwnershipDateParts>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let Some(lower_bound) = lower_bound else {
        return Ok(Some(value));
    };
    if doc_ownership_date_parts_lt(value, lower_bound)? {
        Ok(Some(lower_bound))
    } else {
        Ok(Some(value))
    }
}

pub(crate) fn doc_ownership_clip_upper_parts(
    value: Option<DocOwnershipDateParts>,
    upper_bound: Option<DocOwnershipDateParts>,
) -> PyResult<Option<DocOwnershipDateParts>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let Some(upper_bound) = upper_bound else {
        return Ok(Some(value));
    };
    if doc_ownership_date_parts_gt(value, upper_bound)? {
        Ok(Some(upper_bound))
    } else {
        Ok(Some(value))
    }
}

pub(crate) struct DocOwnershipDecision {
    authoritative_ric: Option<String>,
    status: Option<String>,
}

#[derive(Clone)]
pub(crate) struct DocOwnershipException {
    start_date: Option<DocOwnershipDateParts>,
    end_date: Option<DocOwnershipDateParts>,
    authoritative_ric: Option<String>,
}

#[pyfunction]
#[pyo3(signature = (
    doc_filing_rows,
    authority_decision_rows,
    authority_exception_rows,
    request_min_date=None,
    request_max_date=None,
    fallback_days=45
))]
pub(crate) fn doc_ownership_request_rows(
    py: Python<'_>,
    doc_filing_rows: &Bound<'_, PyAny>,
    authority_decision_rows: &Bound<'_, PyAny>,
    authority_exception_rows: &Bound<'_, PyAny>,
    request_min_date: Option<&Bound<'_, PyAny>>,
    request_max_date: Option<&Bound<'_, PyAny>>,
    fallback_days: i64,
) -> PyResult<Vec<PyObject>> {
    if fallback_days < 0 {
        return Err(PyValueError::new_err("fallback_days must be non-negative"));
    }
    let fallback_days = u32::try_from(fallback_days)
        .map_err(|_| PyValueError::new_err("fallback_days out of range"))?;
    let request_min_parts = optional_date_parts(request_min_date)?;
    let request_max_parts = optional_date_parts(request_max_date)?;

    let mut decisions_by_permno: HashMap<String, DocOwnershipDecision> = HashMap::new();
    for row in authority_decision_rows.iter()? {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("doc ownership decision row is not a dict"))?;
        let Some(kypermno) = normalize_kypermno_value(dict.get_item("KYPERMNO")?.as_ref())? else {
            continue;
        };
        decisions_by_permno.insert(
            kypermno,
            DocOwnershipDecision {
                authoritative_ric: dict_normalized_string(dict, "authoritative_ric")?,
                status: dict_normalized_string(dict, "authority_decision_status")?,
            },
        );
    }

    let mut exceptions_by_permno: HashMap<String, Vec<DocOwnershipException>> = HashMap::new();
    for row in authority_exception_rows.iter()? {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("doc ownership exception row is not a dict"))?;
        let Some(kypermno) = normalize_kypermno_value(dict.get_item("KYPERMNO")?.as_ref())? else {
            continue;
        };
        exceptions_by_permno
            .entry(kypermno)
            .or_default()
            .push(DocOwnershipException {
                start_date: doc_ownership_date_parts_from_dict(
                    dict,
                    "authority_window_start_date",
                )?,
                end_date: doc_ownership_date_parts_from_dict(dict, "authority_window_end_date")?,
                authoritative_ric: dict_normalized_string(dict, "authoritative_ric")?,
            });
    }

    let mut out_rows: Vec<PyObject> = Vec::new();
    for row in doc_filing_rows.iter()? {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("doc ownership filing row is not a dict"))?;
        let doc_id = dict_normalized_string(dict, "doc_id")?;
        let filing_date = doc_ownership_date_parts_from_dict(dict, "filing_date")?;
        let mut kypermno = normalize_kypermno_value(dict.get_item("KYPERMNO")?.as_ref())?;
        if kypermno.is_none() {
            kypermno = normalize_kypermno_value(dict.get_item("kypermno")?.as_ref())?;
        }
        let target_quarter_end = match filing_date {
            Some(parts) => Some(doc_ownership_most_recent_quarter_end_parts(parts)?),
            None => None,
        };
        let target_effective_date = match target_quarter_end {
            Some(parts) => Some(doc_ownership_target_effective_date_parts(parts)?),
            None => None,
        };
        let mut fallback_window_start = target_effective_date;
        let mut fallback_window_end = match (filing_date, target_effective_date) {
            (Some(filing), Some(target)) => Some(doc_ownership_min_date_parts(
                filing,
                doc_ownership_add_days(target, fallback_days)?,
            )?),
            _ => None,
        };

        let mut authoritative_ric: Option<String> = None;
        let mut retrieval_exclusion_reason: Option<&str> = None;
        let mut authority_decision_status: Option<String> = None;
        let decision = kypermno
            .as_ref()
            .and_then(|key| decisions_by_permno.get(key));

        if doc_id.is_none() {
            retrieval_exclusion_reason = Some("missing_doc_id");
        } else if filing_date.is_none() {
            retrieval_exclusion_reason = Some("missing_filing_date");
        } else if kypermno.is_none() {
            retrieval_exclusion_reason = Some("missing_kypermno");
        } else if decision.is_none() {
            retrieval_exclusion_reason = Some("no_authority_decision_for_kypermno");
        } else if let Some(decision) = decision {
            authority_decision_status = decision.status.clone();
            if authority_decision_status.as_deref() == Some("DATE_VARYING_CONVENTIONAL_EXCEPTION") {
                let mut matching_rows: Vec<DocOwnershipException> = Vec::new();
                if let (Some(kypermno), Some(target)) = (kypermno.as_ref(), target_quarter_end) {
                    if let Some(exception_rows) = exceptions_by_permno.get(kypermno) {
                        for exception in exception_rows {
                            let (Some(start_date), Some(end_date)) =
                                (exception.start_date, exception.end_date)
                            else {
                                continue;
                            };
                            if doc_ownership_date_parts_leq(start_date, target)?
                                && doc_ownership_date_parts_leq(target, end_date)?
                            {
                                matching_rows.push(exception.clone());
                            }
                        }
                    }
                }
                if matching_rows.len() == 1 {
                    authoritative_ric = matching_rows[0].authoritative_ric.clone();
                    if authoritative_ric.is_none() {
                        retrieval_exclusion_reason =
                            Some("matched_exception_window_missing_authoritative_ric");
                    }
                } else if matching_rows.len() > 1 {
                    retrieval_exclusion_reason =
                        Some("multiple_exception_windows_for_target_quarter");
                } else {
                    retrieval_exclusion_reason =
                        Some("no_exception_window_match_for_target_quarter");
                }
            } else if authority_decision_status.as_deref() == Some("REVIEW_REQUIRED") {
                retrieval_exclusion_reason = Some("authority_review_required");
            } else if decision.authoritative_ric.is_some() {
                authoritative_ric = decision.authoritative_ric.clone();
            } else {
                retrieval_exclusion_reason = Some("no_authoritative_ric");
            }
        }

        if retrieval_exclusion_reason.is_none() {
            if let Some(target_effective_date) = target_effective_date {
                if let Some(request_min_parts) = request_min_parts {
                    if doc_ownership_date_parts_lt(target_effective_date, request_min_parts)? {
                        retrieval_exclusion_reason =
                            Some("target_effective_date_before_request_min_date");
                    }
                }
                if retrieval_exclusion_reason.is_none() {
                    if let Some(request_max_parts) = request_max_parts {
                        if doc_ownership_date_parts_gt(target_effective_date, request_max_parts)? {
                            retrieval_exclusion_reason =
                                Some("target_effective_date_after_request_max_date");
                        }
                    }
                }
            }
        }

        if retrieval_exclusion_reason.is_none() {
            fallback_window_start =
                doc_ownership_clip_lower_parts(fallback_window_start, request_min_parts)?;
            fallback_window_end =
                doc_ownership_clip_upper_parts(fallback_window_end, request_max_parts)?;
        }

        let retrieval_eligible =
            authoritative_ric.is_some() && retrieval_exclusion_reason.is_none();
        let out = PyDict::new_bound(py);
        out.set_item("doc_id", doc_id)?;
        out.set_item(
            "filing_date",
            doc_ownership_date_parts_to_py(py, filing_date)?,
        )?;
        out.set_item("KYPERMNO", kypermno)?;
        out.set_item("authoritative_ric", authoritative_ric)?;
        out.set_item("authority_decision_status", authority_decision_status)?;
        out.set_item(
            "target_quarter_end",
            doc_ownership_date_parts_to_py(py, target_quarter_end)?,
        )?;
        out.set_item(
            "target_effective_date",
            doc_ownership_date_parts_to_py(py, target_effective_date)?,
        )?;
        out.set_item(
            "fallback_window_start",
            doc_ownership_date_parts_to_py(py, fallback_window_start)?,
        )?;
        out.set_item(
            "fallback_window_end",
            doc_ownership_date_parts_to_py(py, fallback_window_end)?,
        )?;
        out.set_item("retrieval_eligible", retrieval_eligible)?;
        out.set_item("retrieval_exclusion_reason", retrieval_exclusion_reason)?;
        out_rows.push(out.into_py(py));
    }
    Ok(out_rows)
}

#[pyfunction]
#[pyo3(signature = (
    doc_filing_column_names,
    doc_filing_column_values,
    authority_decision_column_names,
    authority_decision_column_values,
    authority_exception_column_names,
    authority_exception_column_values,
    request_min_date=None,
    request_max_date=None,
    fallback_days=45
))]
pub(crate) fn doc_ownership_request_rows_columns(
    py: Python<'_>,
    doc_filing_column_names: Vec<String>,
    doc_filing_column_values: &Bound<'_, PyAny>,
    authority_decision_column_names: Vec<String>,
    authority_decision_column_values: &Bound<'_, PyAny>,
    authority_exception_column_names: Vec<String>,
    authority_exception_column_values: &Bound<'_, PyAny>,
    request_min_date: Option<&Bound<'_, PyAny>>,
    request_max_date: Option<&Bound<'_, PyAny>>,
    fallback_days: i64,
) -> PyResult<Vec<PyObject>> {
    if fallback_days < 0 {
        return Err(PyValueError::new_err("fallback_days must be non-negative"));
    }
    let fallback_days = u32::try_from(fallback_days)
        .map_err(|_| PyValueError::new_err("fallback_days out of range"))?;
    let request_min_parts = optional_date_parts(request_min_date)?;
    let request_max_parts = optional_date_parts(request_max_date)?;

    let decision_label = "doc ownership decision rows";
    let (decision_columns, decision_row_count) = collect_pyobject_column_values(
        py,
        &authority_decision_column_names,
        authority_decision_column_values,
        decision_label,
    )?;
    let decision_column_index = column_index_by_name(&authority_decision_column_names);
    let mut decisions_by_permno: HashMap<String, DocOwnershipDecision> = HashMap::new();
    for row_idx in 0..decision_row_count {
        let Some(kypermno) = column_kypermno_value(
            py,
            &decision_columns,
            &decision_column_index,
            row_idx,
            "KYPERMNO",
        )?
        else {
            continue;
        };
        decisions_by_permno.insert(
            kypermno,
            DocOwnershipDecision {
                authoritative_ric: column_normalized_string(
                    py,
                    &decision_columns,
                    &decision_column_index,
                    row_idx,
                    "authoritative_ric",
                )?,
                status: column_normalized_string(
                    py,
                    &decision_columns,
                    &decision_column_index,
                    row_idx,
                    "authority_decision_status",
                )?,
            },
        );
    }

    let exception_label = "doc ownership exception rows";
    let (exception_columns, exception_row_count) = collect_pyobject_column_values(
        py,
        &authority_exception_column_names,
        authority_exception_column_values,
        exception_label,
    )?;
    let exception_column_index = column_index_by_name(&authority_exception_column_names);
    let mut exceptions_by_permno: HashMap<String, Vec<DocOwnershipException>> = HashMap::new();
    for row_idx in 0..exception_row_count {
        let Some(kypermno) = column_kypermno_value(
            py,
            &exception_columns,
            &exception_column_index,
            row_idx,
            "KYPERMNO",
        )?
        else {
            continue;
        };
        exceptions_by_permno
            .entry(kypermno)
            .or_default()
            .push(DocOwnershipException {
                start_date: column_doc_ownership_date_parts(
                    py,
                    &exception_columns,
                    &exception_column_index,
                    row_idx,
                    "authority_window_start_date",
                )?,
                end_date: column_doc_ownership_date_parts(
                    py,
                    &exception_columns,
                    &exception_column_index,
                    row_idx,
                    "authority_window_end_date",
                )?,
                authoritative_ric: column_normalized_string(
                    py,
                    &exception_columns,
                    &exception_column_index,
                    row_idx,
                    "authoritative_ric",
                )?,
            });
    }

    let filing_label = "doc ownership filing rows";
    let (filing_columns, filing_row_count) = collect_pyobject_column_values(
        py,
        &doc_filing_column_names,
        doc_filing_column_values,
        filing_label,
    )?;
    let filing_column_index = column_index_by_name(&doc_filing_column_names);
    let mut out_rows: Vec<PyObject> = Vec::new();
    for row_idx in 0..filing_row_count {
        let doc_id =
            column_normalized_string(py, &filing_columns, &filing_column_index, row_idx, "doc_id")?;
        let filing_date = column_doc_ownership_date_parts(
            py,
            &filing_columns,
            &filing_column_index,
            row_idx,
            "filing_date",
        )?;
        let mut kypermno = column_kypermno_value(
            py,
            &filing_columns,
            &filing_column_index,
            row_idx,
            "KYPERMNO",
        )?;
        if kypermno.is_none() {
            kypermno = column_kypermno_value(
                py,
                &filing_columns,
                &filing_column_index,
                row_idx,
                "kypermno",
            )?;
        }
        let target_quarter_end = match filing_date {
            Some(parts) => Some(doc_ownership_most_recent_quarter_end_parts(parts)?),
            None => None,
        };
        let target_effective_date = match target_quarter_end {
            Some(parts) => Some(doc_ownership_target_effective_date_parts(parts)?),
            None => None,
        };
        let mut fallback_window_start = target_effective_date;
        let mut fallback_window_end = match (filing_date, target_effective_date) {
            (Some(filing), Some(target)) => Some(doc_ownership_min_date_parts(
                filing,
                doc_ownership_add_days(target, fallback_days)?,
            )?),
            _ => None,
        };

        let mut authoritative_ric: Option<String> = None;
        let mut retrieval_exclusion_reason: Option<&str> = None;
        let mut authority_decision_status: Option<String> = None;
        let decision = kypermno
            .as_ref()
            .and_then(|key| decisions_by_permno.get(key));

        if doc_id.is_none() {
            retrieval_exclusion_reason = Some("missing_doc_id");
        } else if filing_date.is_none() {
            retrieval_exclusion_reason = Some("missing_filing_date");
        } else if kypermno.is_none() {
            retrieval_exclusion_reason = Some("missing_kypermno");
        } else if decision.is_none() {
            retrieval_exclusion_reason = Some("no_authority_decision_for_kypermno");
        } else if let Some(decision) = decision {
            authority_decision_status = decision.status.clone();
            if authority_decision_status.as_deref() == Some("DATE_VARYING_CONVENTIONAL_EXCEPTION") {
                let mut matching_rows: Vec<DocOwnershipException> = Vec::new();
                if let (Some(kypermno), Some(target)) = (kypermno.as_ref(), target_quarter_end) {
                    if let Some(exception_rows) = exceptions_by_permno.get(kypermno) {
                        for exception in exception_rows {
                            let (Some(start_date), Some(end_date)) =
                                (exception.start_date, exception.end_date)
                            else {
                                continue;
                            };
                            if doc_ownership_date_parts_leq(start_date, target)?
                                && doc_ownership_date_parts_leq(target, end_date)?
                            {
                                matching_rows.push(exception.clone());
                            }
                        }
                    }
                }
                if matching_rows.len() == 1 {
                    authoritative_ric = matching_rows[0].authoritative_ric.clone();
                    if authoritative_ric.is_none() {
                        retrieval_exclusion_reason =
                            Some("matched_exception_window_missing_authoritative_ric");
                    }
                } else if matching_rows.len() > 1 {
                    retrieval_exclusion_reason =
                        Some("multiple_exception_windows_for_target_quarter");
                } else {
                    retrieval_exclusion_reason =
                        Some("no_exception_window_match_for_target_quarter");
                }
            } else if authority_decision_status.as_deref() == Some("REVIEW_REQUIRED") {
                retrieval_exclusion_reason = Some("authority_review_required");
            } else if decision.authoritative_ric.is_some() {
                authoritative_ric = decision.authoritative_ric.clone();
            } else {
                retrieval_exclusion_reason = Some("no_authoritative_ric");
            }
        }

        if retrieval_exclusion_reason.is_none() {
            if let Some(target_effective_date) = target_effective_date {
                if let Some(request_min_parts) = request_min_parts {
                    if doc_ownership_date_parts_lt(target_effective_date, request_min_parts)? {
                        retrieval_exclusion_reason =
                            Some("target_effective_date_before_request_min_date");
                    }
                }
                if retrieval_exclusion_reason.is_none() {
                    if let Some(request_max_parts) = request_max_parts {
                        if doc_ownership_date_parts_gt(target_effective_date, request_max_parts)? {
                            retrieval_exclusion_reason =
                                Some("target_effective_date_after_request_max_date");
                        }
                    }
                }
            }
        }

        if retrieval_exclusion_reason.is_none() {
            fallback_window_start =
                doc_ownership_clip_lower_parts(fallback_window_start, request_min_parts)?;
            fallback_window_end =
                doc_ownership_clip_upper_parts(fallback_window_end, request_max_parts)?;
        }

        let retrieval_eligible =
            authoritative_ric.is_some() && retrieval_exclusion_reason.is_none();
        let out = PyDict::new_bound(py);
        out.set_item("doc_id", doc_id)?;
        out.set_item(
            "filing_date",
            doc_ownership_date_parts_to_py(py, filing_date)?,
        )?;
        out.set_item("KYPERMNO", kypermno)?;
        out.set_item("authoritative_ric", authoritative_ric)?;
        out.set_item("authority_decision_status", authority_decision_status)?;
        out.set_item(
            "target_quarter_end",
            doc_ownership_date_parts_to_py(py, target_quarter_end)?,
        )?;
        out.set_item(
            "target_effective_date",
            doc_ownership_date_parts_to_py(py, target_effective_date)?,
        )?;
        out.set_item(
            "fallback_window_start",
            doc_ownership_date_parts_to_py(py, fallback_window_start)?,
        )?;
        out.set_item(
            "fallback_window_end",
            doc_ownership_date_parts_to_py(py, fallback_window_end)?,
        )?;
        out.set_item("retrieval_eligible", retrieval_eligible)?;
        out.set_item("retrieval_exclusion_reason", retrieval_exclusion_reason)?;
        out_rows.push(out.into_py(py));
    }
    Ok(out_rows)
}

pub(crate) fn doc_ownership_safe_rate(numerator: i64, denominator: i64) -> Option<f64> {
    if denominator == 0 {
        None
    } else {
        Some(numerator as f64 / denominator as f64)
    }
}

pub(crate) fn doc_ownership_top_cik_count_rows(
    py: Python<'_>,
    counts: &HashMap<String, i64>,
) -> PyResult<PyObject> {
    let mut entries: Vec<(&String, &i64)> = counts.iter().collect();
    entries.sort_by(|(left_cik, left_count), (right_cik, right_count)| {
        right_count
            .cmp(left_count)
            .then_with(|| left_cik.cmp(right_cik))
    });
    let rows = PyList::empty_bound(py);
    for (cik_10, count) in entries.into_iter().take(20) {
        let row = PyDict::new_bound(py);
        row.set_item("cik_10", cik_10)?;
        row.set_item("doc_count", count)?;
        rows.append(row)?;
    }
    Ok(rows.into_py(py))
}

#[derive(Default)]
pub(crate) struct DocOwnershipUniverseSummaryCounts {
    backbone_doc_count: i64,
    request_doc_count: i64,
    final_doc_count_raw: i64,
    backbone_request_overlap: i64,
    backbone_only_doc_count: i64,
    request_only_doc_count: i64,
    final_backbone_overlap: i64,
    final_request_overlap: i64,
    request_missing_from_final_doc_count: i64,
    final_only_doc_count: i64,
    backbone_missing_from_final_doc_count: i64,
    final_missing_from_backbone_doc_count: i64,
    final_nonnull_ownership_doc_count: i64,
    retrieval_status_counts: BTreeMap<String, i64>,
    backbone_request_mismatch_cik_counts: HashMap<String, i64>,
    request_final_mismatch_cik_counts: HashMap<String, i64>,
}

impl DocOwnershipUniverseSummaryCounts {
    fn push(
        &mut self,
        in_backbone: bool,
        in_request: bool,
        in_final: bool,
        final_has_nonnull_ownership: bool,
        final_retrieval_status: Option<String>,
        cik_10: Option<String>,
        final_present: bool,
    ) {
        if in_backbone {
            self.backbone_doc_count += 1;
        }
        if in_request {
            self.request_doc_count += 1;
        }
        if in_final {
            self.final_doc_count_raw += 1;
        }
        if in_backbone && in_request {
            self.backbone_request_overlap += 1;
        }
        if in_backbone && !in_request {
            self.backbone_only_doc_count += 1;
        }
        if in_request && !in_backbone {
            self.request_only_doc_count += 1;
        }
        if final_present {
            if in_backbone && in_final {
                self.final_backbone_overlap += 1;
            }
            if in_request && in_final {
                self.final_request_overlap += 1;
            }
            if in_request && !in_final {
                self.request_missing_from_final_doc_count += 1;
            }
            if in_final && !in_request {
                self.final_only_doc_count += 1;
            }
            if in_backbone && !in_final {
                self.backbone_missing_from_final_doc_count += 1;
            }
            if in_final && !in_backbone {
                self.final_missing_from_backbone_doc_count += 1;
            }
            if final_has_nonnull_ownership {
                self.final_nonnull_ownership_doc_count += 1;
            }
            if let Some(status) = final_retrieval_status {
                *self.retrieval_status_counts.entry(status).or_insert(0) += 1;
            }
        }

        if let Some(cik_10) = cik_10 {
            if in_backbone != in_request {
                *self
                    .backbone_request_mismatch_cik_counts
                    .entry(cik_10.clone())
                    .or_insert(0) += 1;
            }
            if final_present && in_request != in_final {
                *self
                    .request_final_mismatch_cik_counts
                    .entry(cik_10)
                    .or_insert(0) += 1;
            }
        }
    }

    fn to_py(&self, py: Python<'_>, final_present: bool) -> PyResult<PyObject> {
        let backbone_request_doc_sets_equal =
            self.backbone_only_doc_count == 0 && self.request_only_doc_count == 0;
        let backbone_final_doc_sets_equal = final_present.then_some(
            self.backbone_missing_from_final_doc_count == 0
                && self.final_missing_from_backbone_doc_count == 0,
        );
        let request_final_doc_sets_equal = final_present.then_some(
            self.request_missing_from_final_doc_count == 0 && self.final_only_doc_count == 0,
        );
        let all_doc_sets_equal = final_present.then_some(
            backbone_request_doc_sets_equal
                && backbone_final_doc_sets_equal.unwrap_or(false)
                && request_final_doc_sets_equal.unwrap_or(false),
        );

        let out = PyDict::new_bound(py);
        out.set_item("backbone_doc_count", self.backbone_doc_count)?;
        out.set_item("request_doc_count", self.request_doc_count)?;
        out.set_item(
            "final_doc_count",
            final_present.then_some(self.final_doc_count_raw),
        )?;
        out.set_item(
            "backbone_request_overlap_doc_count",
            self.backbone_request_overlap,
        )?;
        out.set_item(
            "backbone_request_overlap_rate_of_backbone",
            doc_ownership_safe_rate(self.backbone_request_overlap, self.backbone_doc_count),
        )?;
        out.set_item(
            "backbone_request_overlap_rate_of_request",
            doc_ownership_safe_rate(self.backbone_request_overlap, self.request_doc_count),
        )?;
        out.set_item("backbone_only_doc_count", self.backbone_only_doc_count)?;
        out.set_item("request_only_doc_count", self.request_only_doc_count)?;
        out.set_item(
            "backbone_request_doc_sets_equal",
            backbone_request_doc_sets_equal,
        )?;
        out.set_item(
            "final_backbone_overlap_doc_count",
            final_present.then_some(self.final_backbone_overlap),
        )?;
        out.set_item(
            "final_request_overlap_doc_count",
            final_present.then_some(self.final_request_overlap),
        )?;
        out.set_item(
            "request_missing_from_final_doc_count",
            final_present.then_some(self.request_missing_from_final_doc_count),
        )?;
        out.set_item(
            "final_only_doc_count",
            final_present.then_some(self.final_only_doc_count),
        )?;
        out.set_item(
            "backbone_missing_from_final_doc_count",
            final_present.then_some(self.backbone_missing_from_final_doc_count),
        )?;
        out.set_item(
            "final_missing_from_backbone_doc_count",
            final_present.then_some(self.final_missing_from_backbone_doc_count),
        )?;
        out.set_item(
            "final_nonnull_ownership_doc_count",
            final_present.then_some(self.final_nonnull_ownership_doc_count),
        )?;
        out.set_item(
            "backbone_final_doc_sets_equal",
            backbone_final_doc_sets_equal,
        )?;
        out.set_item("request_final_doc_sets_equal", request_final_doc_sets_equal)?;
        out.set_item("all_doc_sets_equal", all_doc_sets_equal)?;
        if final_present {
            let status_dict = PyDict::new_bound(py);
            for (status, count) in &self.retrieval_status_counts {
                status_dict.set_item(status, *count)?;
            }
            out.set_item("retrieval_status_counts", status_dict)?;
        } else {
            out.set_item("retrieval_status_counts", py.None())?;
        }
        out.set_item(
            "backbone_request_mismatch_cik_counts_top",
            doc_ownership_top_cik_count_rows(py, &self.backbone_request_mismatch_cik_counts)?,
        )?;
        if final_present {
            out.set_item(
                "request_final_mismatch_cik_counts_top",
                doc_ownership_top_cik_count_rows(py, &self.request_final_mismatch_cik_counts)?,
            )?;
        } else {
            out.set_item(
                "request_final_mismatch_cik_counts_top",
                PyList::empty_bound(py),
            )?;
        }
        Ok(out.into_py(py))
    }
}

#[pyfunction]
pub(crate) fn doc_ownership_universe_summary_values(
    py: Python<'_>,
    detail_rows: &Bound<'_, PyAny>,
    final_present: bool,
) -> PyResult<PyObject> {
    let mut summary = DocOwnershipUniverseSummaryCounts::default();
    for row in detail_rows.iter()? {
        let row = row?;
        let dict = row.downcast::<PyDict>().map_err(|_| {
            PyValueError::new_err("doc ownership universe detail row is not a dict")
        })?;
        summary.push(
            dict_truthy_bool(dict, "in_backbone")?,
            dict_truthy_bool(dict, "in_request")?,
            dict_truthy_bool(dict, "in_final")?,
            dict_truthy_bool(dict, "final_has_nonnull_ownership")?,
            if final_present {
                dict_normalized_string(dict, "final_retrieval_status")?
            } else {
                None
            },
            dict_normalized_string(dict, "cik_10")?,
            final_present,
        );
    }
    summary.to_py(py, final_present)
}

#[pyfunction]
pub(crate) fn doc_ownership_universe_summary_column_values(
    py: Python<'_>,
    column_names: Vec<String>,
    column_values: &Bound<'_, PyAny>,
    final_present: bool,
) -> PyResult<PyObject> {
    let label = "doc ownership universe detail rows";
    let (columns, row_count) =
        collect_pyobject_column_values(py, &column_names, column_values, label)?;
    let column_index = column_index_by_name(&column_names);
    let mut summary = DocOwnershipUniverseSummaryCounts::default();
    for row_idx in 0..row_count {
        summary.push(
            column_truthy_bool(py, &columns, &column_index, row_idx, "in_backbone")?,
            column_truthy_bool(py, &columns, &column_index, row_idx, "in_request")?,
            column_truthy_bool(py, &columns, &column_index, row_idx, "in_final")?,
            column_truthy_bool(
                py,
                &columns,
                &column_index,
                row_idx,
                "final_has_nonnull_ownership",
            )?,
            if final_present {
                column_normalized_string(
                    py,
                    &columns,
                    &column_index,
                    row_idx,
                    "final_retrieval_status",
                )?
            } else {
                None
            },
            column_normalized_string(py, &columns, &column_index, row_idx, "cik_10")?,
            final_present,
        );
    }
    summary.to_py(py, final_present)
}

#[pyfunction]
pub(crate) fn doc_ownership_select_exact_hit_rows(
    rows: &Bound<'_, PyAny>,
) -> PyResult<(Vec<(String, usize, f64)>, Vec<String>)> {
    let mut rows_by_doc_id: HashMap<String, Vec<(usize, f64)>> = HashMap::new();
    let mut doc_order: Vec<String> = Vec::new();

    for (row_index, row) in rows.iter()?.enumerate() {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("exact ownership raw row is not a dict"))?;
        let Some(doc_id) = dict_normalized_string(dict, "doc_id")? else {
            continue;
        };
        if !dict_truthy_bool(dict, "is_institutional_category")? {
            continue;
        }
        let Some(raw_value) = dict_optional_float(dict, "returned_value")? else {
            continue;
        };
        let Some(cleaned_value) = clean_doc_ownership_institutional_value(Some(raw_value)) else {
            continue;
        };
        if !cleaned_value.is_finite() {
            return Err(PyValueError::new_err(
                "non-finite exact ownership value falls back to Python",
            ));
        }
        if !rows_by_doc_id.contains_key(&doc_id) {
            doc_order.push(doc_id.clone());
        }
        rows_by_doc_id
            .entry(doc_id)
            .or_default()
            .push((row_index, cleaned_value));
    }

    let mut selected: Vec<(String, usize, f64)> = Vec::new();
    let mut conflicts: Vec<String> = Vec::new();
    for doc_id in doc_order {
        let Some(entries) = rows_by_doc_id.get(&doc_id) else {
            continue;
        };
        let mut unique_values: HashSet<u64> = HashSet::new();
        for (_, value) in entries {
            let normalized_zero = if *value == 0.0 { 0.0 } else { *value };
            unique_values.insert(normalized_zero.to_bits());
        }
        if unique_values.len() != 1 {
            conflicts.push(doc_id);
            continue;
        }
        let (row_index, cleaned_value) = entries[0];
        selected.push((doc_id, row_index, cleaned_value));
    }
    Ok((selected, conflicts))
}

#[pyfunction]
pub(crate) fn doc_ownership_select_exact_hit_columns(
    py: Python<'_>,
    column_names: Vec<String>,
    column_values: &Bound<'_, PyAny>,
) -> PyResult<(Vec<(String, usize, f64)>, Vec<String>)> {
    let label = "exact ownership raw rows";
    let (columns, row_count) =
        collect_pyobject_column_values(py, &column_names, column_values, label)?;
    let column_index = column_index_by_name(&column_names);
    let mut rows_by_doc_id: HashMap<String, Vec<(usize, f64)>> = HashMap::new();
    let mut doc_order: Vec<String> = Vec::new();

    for row_index in 0..row_count {
        let Some(doc_id) =
            column_normalized_string(py, &columns, &column_index, row_index, "doc_id")?
        else {
            continue;
        };
        if !column_truthy_bool(
            py,
            &columns,
            &column_index,
            row_index,
            "is_institutional_category",
        )? {
            continue;
        }
        let Some(raw_value) =
            column_optional_float(py, &columns, &column_index, row_index, "returned_value")?
        else {
            continue;
        };
        let Some(cleaned_value) = clean_doc_ownership_institutional_value(Some(raw_value)) else {
            continue;
        };
        if !cleaned_value.is_finite() {
            return Err(PyValueError::new_err(
                "non-finite exact ownership value falls back to Python",
            ));
        }
        if !rows_by_doc_id.contains_key(&doc_id) {
            doc_order.push(doc_id.clone());
        }
        rows_by_doc_id
            .entry(doc_id)
            .or_default()
            .push((row_index, cleaned_value));
    }

    let mut selected: Vec<(String, usize, f64)> = Vec::new();
    let mut conflicts: Vec<String> = Vec::new();
    for doc_id in doc_order {
        let Some(entries) = rows_by_doc_id.get(&doc_id) else {
            continue;
        };
        let mut unique_values: HashSet<u64> = HashSet::new();
        for (_, value) in entries {
            let normalized_zero = if *value == 0.0 { 0.0 } else { *value };
            unique_values.insert(normalized_zero.to_bits());
        }
        if unique_values.len() != 1 {
            conflicts.push(doc_id);
            continue;
        }
        let (row_index, cleaned_value) = entries[0];
        selected.push((doc_id, row_index, cleaned_value));
    }
    Ok((selected, conflicts))
}

#[pyfunction]
pub(crate) fn doc_ownership_select_fallback_hit_rows(
    fallback_rows: &Bound<'_, PyAny>,
    request_rows: &Bound<'_, PyAny>,
) -> PyResult<(Vec<(String, usize, f64)>, Vec<String>)> {
    let mut fallback_window_end_by_doc_id: HashMap<String, Option<String>> = HashMap::new();
    for row in request_rows.iter()? {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("doc ownership request row is not a dict"))?;
        let Some(doc_id) = dict_normalized_string(dict, "doc_id")? else {
            continue;
        };
        fallback_window_end_by_doc_id.insert(doc_id, dict_date_key(dict, "fallback_window_end")?);
    }

    let mut rows_by_doc_id: HashMap<String, Vec<(usize, f64, String)>> = HashMap::new();
    let mut doc_order: Vec<String> = Vec::new();
    for (row_index, row) in fallback_rows.iter()?.enumerate() {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("fallback ownership raw row is not a dict"))?;
        let Some(doc_id) = dict_normalized_string(dict, "doc_id")? else {
            continue;
        };
        if !dict_truthy_bool(dict, "is_institutional_category")? {
            continue;
        }
        let Some(raw_value) = dict_optional_float(dict, "returned_value")? else {
            continue;
        };
        let Some(cleaned_value) = clean_doc_ownership_institutional_value(Some(raw_value)) else {
            continue;
        };
        if !cleaned_value.is_finite() {
            return Err(PyValueError::new_err(
                "non-finite fallback ownership value falls back to Python",
            ));
        }
        let Some(response_date) = dict_date_key(dict, "response_date")? else {
            continue;
        };
        if let Some(Some(fallback_window_end)) = fallback_window_end_by_doc_id.get(&doc_id) {
            if &response_date > fallback_window_end {
                continue;
            }
        }
        if !rows_by_doc_id.contains_key(&doc_id) {
            doc_order.push(doc_id.clone());
        }
        rows_by_doc_id
            .entry(doc_id)
            .or_default()
            .push((row_index, cleaned_value, response_date));
    }

    let mut selected: Vec<(String, usize, f64)> = Vec::new();
    let mut conflicts: Vec<String> = Vec::new();
    for doc_id in doc_order {
        let Some(entries) = rows_by_doc_id.get(&doc_id) else {
            continue;
        };
        let Some(latest_date) = entries.iter().map(|(_, _, date)| date).max() else {
            continue;
        };
        let mut latest_rows: Vec<(usize, f64)> = Vec::new();
        let mut unique_values: HashSet<u64> = HashSet::new();
        for (row_index, cleaned_value, response_date) in entries {
            if response_date != latest_date {
                continue;
            }
            let normalized_zero = if *cleaned_value == 0.0 {
                0.0
            } else {
                *cleaned_value
            };
            unique_values.insert(normalized_zero.to_bits());
            latest_rows.push((*row_index, *cleaned_value));
        }
        if unique_values.len() != 1 {
            conflicts.push(doc_id);
            continue;
        }
        if let Some((row_index, cleaned_value)) = latest_rows.first() {
            selected.push((doc_id, *row_index, *cleaned_value));
        }
    }
    Ok((selected, conflicts))
}

#[pyfunction]
pub(crate) fn doc_ownership_select_fallback_hit_columns(
    py: Python<'_>,
    fallback_column_names: Vec<String>,
    fallback_column_values: &Bound<'_, PyAny>,
    request_column_names: Vec<String>,
    request_column_values: &Bound<'_, PyAny>,
) -> PyResult<(Vec<(String, usize, f64)>, Vec<String>)> {
    let request_label = "doc ownership request rows";
    let (request_columns, request_row_count) = collect_pyobject_column_values(
        py,
        &request_column_names,
        request_column_values,
        request_label,
    )?;
    let request_column_index = column_index_by_name(&request_column_names);
    let mut fallback_window_end_by_doc_id: HashMap<String, Option<String>> = HashMap::new();
    for row_index in 0..request_row_count {
        let Some(doc_id) = column_normalized_string(
            py,
            &request_columns,
            &request_column_index,
            row_index,
            "doc_id",
        )?
        else {
            continue;
        };
        fallback_window_end_by_doc_id.insert(
            doc_id,
            column_date_key(
                py,
                &request_columns,
                &request_column_index,
                row_index,
                "fallback_window_end",
            )?,
        );
    }

    let fallback_label = "fallback ownership raw rows";
    let (fallback_columns, fallback_row_count) = collect_pyobject_column_values(
        py,
        &fallback_column_names,
        fallback_column_values,
        fallback_label,
    )?;
    let fallback_column_index = column_index_by_name(&fallback_column_names);
    let mut rows_by_doc_id: HashMap<String, Vec<(usize, f64, String)>> = HashMap::new();
    let mut doc_order: Vec<String> = Vec::new();
    for row_index in 0..fallback_row_count {
        let Some(doc_id) = column_normalized_string(
            py,
            &fallback_columns,
            &fallback_column_index,
            row_index,
            "doc_id",
        )?
        else {
            continue;
        };
        if !column_truthy_bool(
            py,
            &fallback_columns,
            &fallback_column_index,
            row_index,
            "is_institutional_category",
        )? {
            continue;
        }
        let Some(raw_value) = column_optional_float(
            py,
            &fallback_columns,
            &fallback_column_index,
            row_index,
            "returned_value",
        )?
        else {
            continue;
        };
        let Some(cleaned_value) = clean_doc_ownership_institutional_value(Some(raw_value)) else {
            continue;
        };
        if !cleaned_value.is_finite() {
            return Err(PyValueError::new_err(
                "non-finite fallback ownership value falls back to Python",
            ));
        }
        let Some(response_date) = column_date_key(
            py,
            &fallback_columns,
            &fallback_column_index,
            row_index,
            "response_date",
        )?
        else {
            continue;
        };
        if let Some(Some(fallback_window_end)) = fallback_window_end_by_doc_id.get(&doc_id) {
            if &response_date > fallback_window_end {
                continue;
            }
        }
        if !rows_by_doc_id.contains_key(&doc_id) {
            doc_order.push(doc_id.clone());
        }
        rows_by_doc_id
            .entry(doc_id)
            .or_default()
            .push((row_index, cleaned_value, response_date));
    }

    let mut selected: Vec<(String, usize, f64)> = Vec::new();
    let mut conflicts: Vec<String> = Vec::new();
    for doc_id in doc_order {
        let Some(entries) = rows_by_doc_id.get(&doc_id) else {
            continue;
        };
        let Some(latest_date) = entries.iter().map(|(_, _, date)| date).max() else {
            continue;
        };
        let mut latest_rows: Vec<(usize, f64)> = Vec::new();
        let mut unique_values: HashSet<u64> = HashSet::new();
        for (row_index, cleaned_value, response_date) in entries {
            if response_date != latest_date {
                continue;
            }
            let normalized_zero = if *cleaned_value == 0.0 {
                0.0
            } else {
                *cleaned_value
            };
            unique_values.insert(normalized_zero.to_bits());
            latest_rows.push((*row_index, *cleaned_value));
        }
        if unique_values.len() != 1 {
            conflicts.push(doc_id);
            continue;
        }
        if let Some((row_index, cleaned_value)) = latest_rows.first() {
            selected.push((doc_id, *row_index, *cleaned_value));
        }
    }
    Ok((selected, conflicts))
}

pub(crate) fn normalized_string_hash_set(values: &Bound<'_, PyAny>) -> PyResult<HashSet<String>> {
    let mut out: HashSet<String> = HashSet::new();
    for value in values.iter()? {
        let value = value?;
        if let Some(normalized) = normalize_lookup_text_any_impl(Some(&value))? {
            out.insert(normalized);
        }
    }
    Ok(out)
}

#[pyfunction]
pub(crate) fn doc_ownership_fallback_request_row_indices(
    request_rows: &Bound<'_, PyAny>,
    selected_doc_ids: &Bound<'_, PyAny>,
    conflict_doc_ids: &Bound<'_, PyAny>,
) -> PyResult<Vec<usize>> {
    let mut excluded = normalized_string_hash_set(selected_doc_ids)?;
    excluded.extend(normalized_string_hash_set(conflict_doc_ids)?);

    let mut out: Vec<usize> = Vec::new();
    for (row_index, row) in request_rows.iter()?.enumerate() {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("doc ownership request row is not a dict"))?;
        let Some(doc_id) = dict_normalized_string(dict, "doc_id")? else {
            continue;
        };
        if !dict_truthy_bool(dict, "retrieval_eligible")? {
            continue;
        }
        if excluded.contains(&doc_id) {
            continue;
        }
        out.push(row_index);
    }
    Ok(out)
}

#[pyfunction]
pub(crate) fn doc_ownership_fallback_request_row_indices_columns(
    py: Python<'_>,
    request_column_names: Vec<String>,
    request_column_values: &Bound<'_, PyAny>,
    selected_doc_ids: &Bound<'_, PyAny>,
    conflict_doc_ids: &Bound<'_, PyAny>,
) -> PyResult<Vec<usize>> {
    let mut excluded = normalized_string_hash_set(selected_doc_ids)?;
    excluded.extend(normalized_string_hash_set(conflict_doc_ids)?);

    let label = "doc ownership request rows";
    let (columns, row_count) =
        collect_pyobject_column_values(py, &request_column_names, request_column_values, label)?;
    let column_index = column_index_by_name(&request_column_names);
    let mut out: Vec<usize> = Vec::new();
    for row_index in 0..row_count {
        let Some(doc_id) =
            column_normalized_string(py, &columns, &column_index, row_index, "doc_id")?
        else {
            continue;
        };
        if !column_truthy_bool(py, &columns, &column_index, row_index, "retrieval_eligible")? {
            continue;
        }
        if excluded.contains(&doc_id) {
            continue;
        }
        out.push(row_index);
    }
    Ok(out)
}

#[pyfunction]
pub(crate) fn item7_lm_floor_threshold_summary_row(
    py: Python<'_>,
    row_audit_rows: &Bound<'_, PyAny>,
    threshold: i64,
    confirmed_false_positive_ids: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let mut confirmed_ids: HashSet<String> = HashSet::new();
    for value in confirmed_false_positive_ids.iter()? {
        let value = value?;
        confirmed_ids.insert(value.extract::<String>()?);
    }
    let mut sample_item7_rows = 0_i64;
    let mut total_dropped = 0_i64;
    let mut floor_dropped = 0_i64;
    let mut reference_stub_dropped = 0_i64;
    let mut confirmed_fp_removed = 0_i64;

    for row in row_audit_rows.iter()? {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("item7 floor sweep audit row is not a dict"))?;
        sample_item7_rows += 1;
        if !dict_truthy_bool(dict, "dropped_after_cleaning")? {
            continue;
        }
        total_dropped += 1;
        let drop_reason = dict_normalized_string(dict, "drop_reason")?;
        if drop_reason.as_deref() == Some("item7_below_lm_token_floor") {
            floor_dropped += 1;
        }
        if drop_reason.as_deref() == Some("reference_only_stub") {
            reference_stub_dropped += 1;
        }
        if let Some(value) = dict.get_item("benchmark_row_id")? {
            if value.is_none() {
                continue;
            }
            let benchmark_row_id = value.extract::<String>()?;
            if confirmed_ids.contains(&benchmark_row_id) {
                confirmed_fp_removed += 1;
            }
        }
    }

    let out = PyDict::new_bound(py);
    out.set_item("item7_min_lm_tokens", threshold)?;
    out.set_item("item7_floor_enabled", threshold > 0)?;
    out.set_item("sample_item7_rows", sample_item7_rows)?;
    out.set_item("item7_rows_dropped_total", total_dropped)?;
    out.set_item("item7_rows_dropped_by_floor", floor_dropped)?;
    out.set_item(
        "item7_rows_dropped_by_reference_stub",
        reference_stub_dropped,
    )?;
    out.set_item("item7_rows_kept", sample_item7_rows - total_dropped)?;
    out.set_item(
        "confirmed_false_positive_removed_rows",
        confirmed_fp_removed,
    )?;
    out.set_item(
        "confirmed_false_positive_saved_rows",
        confirmed_ids.len() as i64 - confirmed_fp_removed,
    )?;
    if total_dropped > 0 {
        out.set_item(
            "confirmed_fp_share_of_total_dropped",
            confirmed_fp_removed as f64 / total_dropped as f64,
        )?;
    } else {
        out.set_item("confirmed_fp_share_of_total_dropped", py.None())?;
    }
    if floor_dropped > 0 {
        out.set_item(
            "confirmed_fp_share_of_floor_dropped",
            confirmed_fp_removed as f64 / floor_dropped as f64,
        )?;
    } else {
        out.set_item("confirmed_fp_share_of_floor_dropped", py.None())?;
    }
    Ok(out.into_py(py))
}

pub(crate) fn doc_ownership_selected_row_map(
    py: Python<'_>,
    selected_rows: &Bound<'_, PyAny>,
    label: &str,
) -> PyResult<HashMap<String, PyObject>> {
    let selected_dict = selected_rows
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err(format!("{label} selected rows are not a dict")))?;
    let mut out: HashMap<String, PyObject> = HashMap::new();
    for (key, value) in selected_dict.iter() {
        let Some(doc_id) = normalize_lookup_text_any_impl(Some(&key))? else {
            continue;
        };
        out.insert(doc_id, value.into_py(py));
    }
    Ok(out)
}

pub(crate) fn doc_ownership_selected_fields(
    py: Python<'_>,
    row: &PyObject,
    label: &str,
) -> PyResult<(Option<PyObject>, Option<String>, Option<f64>)> {
    let dict = row
        .bind(py)
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err(format!("{label} selected row is not a dict")))?;
    let selected_response_date =
        normalize_doc_ownership_date_value(py, dict.get_item("response_date")?.as_ref())?;
    let returned_category = dict_doc_category(dict, "returned_category")?;
    let institutional_ownership_pct = clean_doc_ownership_institutional_value(
        normalize_doc_ownership_float_value(dict.get_item("returned_value")?.as_ref())?,
    );
    Ok((
        selected_response_date,
        returned_category,
        institutional_ownership_pct,
    ))
}

#[pyfunction]
pub(crate) fn doc_ownership_final_rows(
    py: Python<'_>,
    request_rows: &Bound<'_, PyAny>,
    exact_selected: &Bound<'_, PyAny>,
    exact_conflict_docs: &Bound<'_, PyAny>,
    fallback_selected: &Bound<'_, PyAny>,
    fallback_conflict_docs: &Bound<'_, PyAny>,
) -> PyResult<Vec<PyObject>> {
    let exact_selected_by_doc_id = doc_ownership_selected_row_map(py, exact_selected, "exact")?;
    let fallback_selected_by_doc_id =
        doc_ownership_selected_row_map(py, fallback_selected, "fallback")?;
    let exact_conflict_doc_ids = normalized_string_hash_set(exact_conflict_docs)?;
    let fallback_conflict_doc_ids = normalized_string_hash_set(fallback_conflict_docs)?;

    let mut out_rows: Vec<PyObject> = Vec::new();
    for row in request_rows.iter()? {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("doc ownership request row is not a dict"))?;
        let Some(doc_id) = dict_normalized_string(dict, "doc_id")? else {
            continue;
        };

        let mut retrieval_status = "NO_USABLE_INSTITUTIONAL_ROW";
        let mut selected_response_date: Option<PyObject> = None;
        let mut returned_category: Option<String> = None;
        let mut institutional_ownership_pct: Option<f64> = None;
        let mut fallback_used = false;

        if !dict_truthy_bool(dict, "retrieval_eligible")? {
            if dict_normalized_string(dict, "authority_decision_status")?.as_deref()
                == Some("REVIEW_REQUIRED")
            {
                retrieval_status = "AUTHORITY_REVIEW_REQUIRED";
            } else {
                retrieval_status = "NO_AUTHORITATIVE_RIC";
            }
        } else if exact_conflict_doc_ids.contains(&doc_id)
            || fallback_conflict_doc_ids.contains(&doc_id)
        {
            retrieval_status = "FALLBACK_CONFLICT_REVIEW";
        } else if let Some(selected_row) = exact_selected_by_doc_id.get(&doc_id) {
            retrieval_status = "EXACT_TARGET_HIT";
            let (date, category, value) = doc_ownership_selected_fields(py, selected_row, "exact")?;
            selected_response_date = date;
            returned_category = category;
            institutional_ownership_pct = value;
        } else if let Some(selected_row) = fallback_selected_by_doc_id.get(&doc_id) {
            retrieval_status = "FALLBACK_WINDOW_HIT";
            fallback_used = true;
            let (date, category, value) =
                doc_ownership_selected_fields(py, selected_row, "fallback")?;
            selected_response_date = date;
            returned_category = category;
            institutional_ownership_pct = value;
        }

        let out = PyDict::new_bound(py);
        out.set_item("doc_id", doc_id)?;
        out.set_item(
            "filing_date",
            normalize_doc_ownership_date_value(py, dict.get_item("filing_date")?.as_ref())?,
        )?;
        out.set_item(
            "KYPERMNO",
            normalize_kypermno_value(dict.get_item("KYPERMNO")?.as_ref())?,
        )?;
        out.set_item(
            "authoritative_ric",
            dict_normalized_string(dict, "authoritative_ric")?,
        )?;
        out.set_item(
            "authority_decision_status",
            dict_normalized_string(dict, "authority_decision_status")?,
        )?;
        out.set_item(
            "target_quarter_end",
            normalize_doc_ownership_date_value(py, dict.get_item("target_quarter_end")?.as_ref())?,
        )?;
        out.set_item(
            "target_effective_date",
            normalize_doc_ownership_date_value(
                py,
                dict.get_item("target_effective_date")?.as_ref(),
            )?,
        )?;
        out.set_item("selected_response_date", selected_response_date)?;
        out.set_item("returned_category", returned_category)?;
        out.set_item("institutional_ownership_pct", institutional_ownership_pct)?;
        out.set_item("retrieval_status", retrieval_status)?;
        out.set_item("fallback_used", fallback_used)?;
        out_rows.push(out.into_py(py));
    }

    Ok(out_rows)
}

#[pyfunction]
pub(crate) fn refinitiv_analyst_shift_months(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    months: i64,
) -> PyResult<PyObject> {
    let (year, month, _) = extract_py_date_parts(value)?;
    let month_index = i64::from(year) * 12 + i64::from(month) - 1 + months;
    let shifted_year = month_index.div_euclid(12);
    let shifted_month = month_index.rem_euclid(12) + 1;
    let shifted_year_i32 = i32::try_from(shifted_year)
        .map_err(|_| PyValueError::new_err("shifted date year out of range"))?;
    let shifted_month_u32 = u32::try_from(shifted_month)
        .map_err(|_| PyValueError::new_err("shifted date month out of range"))?;
    let Some(last_day) = days_in_month(shifted_year_i32, shifted_month_u32) else {
        return Err(PyValueError::new_err("shifted date month out of range"));
    };
    py_date_object(py, shifted_year_i32, shifted_month_u32, last_day)
}

#[pyfunction]
#[pyo3(signature = (sorted_dates=None, cutoff=None))]
pub(crate) fn refinitiv_analyst_latest_date_on_or_before(
    py: Python<'_>,
    sorted_dates: Option<&Bound<'_, PyAny>>,
    cutoff: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<PyObject>> {
    let Some(sorted_dates) = sorted_dates else {
        return Ok(None);
    };
    if sorted_dates.is_none() {
        return Ok(None);
    }
    let Some(cutoff) = cutoff else {
        return Err(PyValueError::new_err("cutoff is required"));
    };
    if cutoff.is_none() {
        return Err(PyValueError::new_err("cutoff is required"));
    }
    let cutoff_key = extract_py_date_parts(cutoff)?;
    let mut entries: Vec<(i32, u32, u32)> = Vec::new();
    for entry in sorted_dates.iter()? {
        entries.push(extract_py_date_parts(&entry?)?);
    }
    if entries.is_empty() {
        return Ok(None);
    }
    let mut low = 0usize;
    let mut high = entries.len();
    while low < high {
        let mid = low + (high - low) / 2;
        if entries[mid] <= cutoff_key {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    if low == 0 {
        return Ok(None);
    }
    let (year, month, day) = entries[low - 1];
    Ok(Some(py_date_object(py, year, month, day)?))
}

#[pyfunction]
#[pyo3(signature = (value=None))]
pub(crate) fn clean_doc_ownership_institutional_value(value: Option<f64>) -> Option<f64> {
    let value = value?;
    if value < 0.0 {
        return None;
    }
    if value > 100.0 {
        Some(100.0)
    } else {
        Some(value)
    }
}

#[pyfunction]
#[pyo3(signature = (value=None))]
pub(crate) fn normalize_doc_ownership_category_value(
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<String>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    let Some(normalized) = py_str_normalized(value)? else {
        return Ok(None);
    };
    let collapsed = collapse_whitespace_to_spaces(&normalized);
    Ok(if collapsed.is_empty() {
        None
    } else {
        Some(collapsed)
    })
}

#[pyfunction]
#[pyo3(signature = (value=None))]
pub(crate) fn is_doc_ownership_institutional_category(value: Option<&str>) -> bool {
    value.is_some_and(|text| text.eq_ignore_ascii_case("Holdings by Institutions"))
}

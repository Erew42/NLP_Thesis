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
use crate::refinitiv_authority::*;
use crate::refinitiv_bridge::*;
use crate::refinitiv_excel::*;
use crate::sec_extraction::*;
use crate::sentence_cleaning::*;
use crate::sentence_quality_api::*;

#[pyfunction]
pub(crate) fn normalize_lseg_month_boundaries(
    start_text: &str,
    end_text: &str,
) -> PyResult<(String, String)> {
    let Some((start_year, start_month, _)) = parse_iso_date_parts(start_text) else {
        return Err(PyValueError::new_err(format!(
            "Invalid isoformat string: {start_text:?}"
        )));
    };
    let Some((end_year, end_month, _)) = parse_iso_date_parts(end_text) else {
        return Err(PyValueError::new_err(format!(
            "Invalid isoformat string: {end_text:?}"
        )));
    };
    let Some(end_day) = days_in_month(end_year, end_month) else {
        return Err(PyValueError::new_err(format!(
            "Invalid isoformat string: {end_text:?}"
        )));
    };
    Ok((
        format!("{start_year:04}-{start_month:02}-01"),
        format!("{end_year:04}-{end_month:02}-{end_day:02}"),
    ))
}

#[pyfunction]
pub(crate) fn lseg_item_window_dates(
    py: Python<'_>,
    start_text: &str,
    end_text: &str,
) -> PyResult<(PyObject, PyObject)> {
    let Some((start_year, start_month, start_day)) = parse_iso_date_parts(start_text) else {
        return Err(PyValueError::new_err(format!(
            "Invalid isoformat string: {start_text:?}"
        )));
    };
    let Some((end_year, end_month, end_day)) = parse_iso_date_parts(end_text) else {
        return Err(PyValueError::new_err(format!(
            "Invalid isoformat string: {end_text:?}"
        )));
    };
    Ok((
        py_date_object(py, start_year, start_month, start_day)?,
        py_date_object(py, end_year, end_month, end_day)?,
    ))
}

#[pyfunction]
pub(crate) fn lseg_interval_span_days(
    start_date: &Bound<'_, PyAny>,
    end_date: &Bound<'_, PyAny>,
) -> PyResult<i64> {
    let (start_year, start_month, start_day) = extract_py_date_parts(start_date)?;
    let (end_year, end_month, end_day) = extract_py_date_parts(end_date)?;
    Ok(date_ordinal(end_year, end_month, end_day)?
        - date_ordinal(start_year, start_month, start_day)?
        + 1)
}

#[pyfunction]
#[pyo3(signature = (
    batch_start_date,
    batch_end_date,
    batch_item_count,
    batch_unique_instruments,
    batch_standalone_rows_estimate,
    batch_batched_rows_estimate,
    candidate_start_date,
    candidate_end_date,
    candidate_instrument,
    max_batch_size,
    row_density_rows_per_day,
    max_batch_items=None,
    max_union_span_days=None,
    max_extra_rows_abs=None,
    max_extra_rows_ratio=None
))]
pub(crate) fn lseg_evaluate_interval_candidate(
    py: Python<'_>,
    batch_start_date: &Bound<'_, PyAny>,
    batch_end_date: &Bound<'_, PyAny>,
    batch_item_count: usize,
    batch_unique_instruments: Vec<String>,
    batch_standalone_rows_estimate: f64,
    batch_batched_rows_estimate: f64,
    candidate_start_date: &Bound<'_, PyAny>,
    candidate_end_date: &Bound<'_, PyAny>,
    candidate_instrument: &str,
    max_batch_size: usize,
    row_density_rows_per_day: f64,
    max_batch_items: Option<usize>,
    max_union_span_days: Option<i64>,
    max_extra_rows_abs: Option<f64>,
    max_extra_rows_ratio: Option<f64>,
) -> PyResult<
    Option<(
        PyObject,
        PyObject,
        Vec<String>,
        f64,
        f64,
        f64,
        f64,
        i64,
        f64,
    )>,
> {
    let batch_start = extract_py_date_parts(batch_start_date)?;
    let batch_end = extract_py_date_parts(batch_end_date)?;
    let candidate_start = extract_py_date_parts(candidate_start_date)?;
    let candidate_end = extract_py_date_parts(candidate_end_date)?;
    let new_start = std::cmp::min(batch_start, candidate_start);
    let new_end = std::cmp::max(batch_end, candidate_end);

    let mut unique_instruments: BTreeSet<String> = batch_unique_instruments.into_iter().collect();
    unique_instruments.insert(candidate_instrument.to_string());
    let item_count = batch_item_count + 1;

    let candidate_span = date_ordinal(candidate_end.0, candidate_end.1, candidate_end.2)?
        - date_ordinal(candidate_start.0, candidate_start.1, candidate_start.2)?
        + 1;
    let union_span_days = date_ordinal(new_end.0, new_end.1, new_end.2)?
        - date_ordinal(new_start.0, new_start.1, new_start.2)?
        + 1;
    let new_standalone_rows =
        batch_standalone_rows_estimate + row_density_rows_per_day * candidate_span as f64;
    let new_batched_rows =
        row_density_rows_per_day * union_span_days as f64 * unique_instruments.len() as f64;
    let extra_rows_abs = new_batched_rows - new_standalone_rows;
    let extra_rows_ratio = if new_standalone_rows <= 0.0 {
        0.0
    } else {
        extra_rows_abs / new_standalone_rows
    };

    if unique_instruments.len() > max_batch_size {
        return Ok(None);
    }
    if max_batch_items.is_some_and(|limit| item_count > limit) {
        return Ok(None);
    }
    if max_union_span_days.is_some_and(|limit| union_span_days > limit) {
        return Ok(None);
    }
    if max_extra_rows_abs.is_some_and(|limit| extra_rows_abs > limit) {
        return Ok(None);
    }
    if max_extra_rows_ratio.is_some_and(|limit| extra_rows_ratio > limit) {
        return Ok(None);
    }

    let candidate_standalone_rows = row_density_rows_per_day * candidate_span as f64;
    let delta_rows = new_batched_rows - batch_batched_rows_estimate - candidate_standalone_rows;
    Ok(Some((
        py_date_object(py, new_start.0, new_start.1, new_start.2)?,
        py_date_object(py, new_end.0, new_end.1, new_end.2)?,
        unique_instruments.into_iter().collect(),
        new_standalone_rows,
        new_batched_rows,
        extra_rows_abs,
        extra_rows_ratio,
        union_span_days,
        delta_rows,
    )))
}

#[pyfunction]
pub(crate) fn normalize_analyst_request_group_id(
    gvkey_int: i64,
    effective_collection_ric: &str,
) -> PyResult<String> {
    let payload = analyst_request_group_payload(gvkey_int, effective_collection_ric)?;
    Ok(stable_hash_id_from_payload("group", &payload))
}

#[pyfunction]
pub(crate) fn normalize_analyst_request_group_ids(
    rows: &Bound<'_, PyAny>,
) -> PyResult<Vec<Option<String>>> {
    let mut out: Vec<Option<String>> = Vec::new();
    for row in rows.iter()? {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("analyst request-group row is not a dict"))?;
        let Some(gvkey_value) = dict.get_item("gvkey_int")? else {
            out.push(None);
            continue;
        };
        let Some(ric_value) = dict.get_item("effective_collection_ric")? else {
            out.push(None);
            continue;
        };
        if gvkey_value.is_none() || ric_value.is_none() {
            out.push(None);
            continue;
        }
        let gvkey_int = py_int_like_to_i64(&gvkey_value)?;
        let effective_collection_ric = ric_value.str()?.to_str()?.to_string();
        out.push(Some(normalize_analyst_request_group_id(
            gvkey_int,
            &effective_collection_ric,
        )?));
    }
    Ok(out)
}

#[pyfunction]
pub(crate) fn normalize_analyst_group_list_values(
    values: &Bound<'_, PyAny>,
) -> PyResult<Vec<String>> {
    let mut normalized: BTreeSet<String> = BTreeSet::new();
    for value in values.iter()? {
        let value = value?;
        if !value.is_truthy()? {
            continue;
        }
        normalized.insert(value.str()?.to_str()?.to_string());
    }
    Ok(normalized.into_iter().collect())
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub(crate) struct AnalystEstimateValueKey {
    forecast_consensus_mean: Option<u64>,
    forecast_dispersion: Option<u64>,
    estimate_count: Option<i64>,
}

pub(crate) type AnalystEstimateSnapshotCanonical = (
    Option<f64>,
    Option<f64>,
    Option<i64>,
    Option<String>,
    Vec<String>,
);

pub(crate) type AnalystActualEventCanonical = (Option<f64>, Option<String>, Vec<String>);

pub(crate) fn analyst_estimate_float_key(value: Option<f64>) -> PyResult<Option<u64>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if !value.is_finite() {
        return Err(PyValueError::new_err(
            "non-finite analyst estimate value falls back to Python",
        ));
    }
    Ok(Some((if value == 0.0 { 0.0 } else { value }).to_bits()))
}

#[pyfunction]
pub(crate) fn refinitiv_analyst_canonicalize_actual_event(
    rows: &Bound<'_, PyAny>,
) -> PyResult<(Option<AnalystActualEventCanonical>, bool, Vec<String>)> {
    let mut seen_key: Option<Option<u64>> = None;
    let mut selected_actual_eps: Option<Option<f64>> = None;
    let mut request_groups: BTreeSet<String> = BTreeSet::new();
    let mut effective_collection_rics: BTreeSet<String> = BTreeSet::new();
    let mut row_count = 0usize;
    for row in rows.iter()? {
        row_count += 1;
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("analyst actual event row is not a dict"))?;
        let Some(actual_eps_value) = dict.get_item("actual_eps")? else {
            return Err(PyValueError::new_err("missing actual_eps"));
        };
        let actual_eps = normalize_doc_ownership_float_value(Some(&actual_eps_value))?;
        let key = analyst_estimate_float_key(actual_eps)?;
        let request_group_id = dict_python_or_empty_string(dict, "request_group_id")?;
        if !request_group_id.is_empty() {
            request_groups.insert(request_group_id);
        }
        let effective_collection_ric =
            dict_python_or_empty_string(dict, "effective_collection_ric")?;
        if !effective_collection_ric.is_empty() {
            effective_collection_rics.insert(effective_collection_ric);
        }
        if let Some(existing_key) = seen_key {
            if existing_key != key {
                return Ok((None, true, request_groups.into_iter().collect()));
            }
        } else {
            seen_key = Some(key);
            selected_actual_eps = Some(actual_eps);
        }
    }
    if row_count == 0 {
        return Err(PyValueError::new_err("empty analyst actual event rows"));
    }
    let effective_collection_ric = if effective_collection_rics.len() == 1 {
        effective_collection_rics.iter().next().cloned()
    } else {
        None
    };
    let source_request_group_ids: Vec<String> = request_groups.into_iter().collect();
    Ok((
        Some((
            selected_actual_eps.unwrap_or(None),
            effective_collection_ric,
            source_request_group_ids.clone(),
        )),
        false,
        source_request_group_ids,
    ))
}

#[pyfunction]
pub(crate) fn refinitiv_analyst_canonicalize_estimate_snapshot(
    py: Python<'_>,
    rows: &Bound<'_, PyAny>,
) -> PyResult<(Option<AnalystEstimateSnapshotCanonical>, bool)> {
    let mut seen_key: Option<AnalystEstimateValueKey> = None;
    let mut selected_values: Option<(Option<f64>, Option<f64>, Option<i64>)> = None;
    let mut request_groups: BTreeSet<String> = BTreeSet::new();
    let mut effective_collection_rics: BTreeSet<String> = BTreeSet::new();
    let mut row_count = 0usize;
    for row in rows.iter()? {
        row_count += 1;
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("analyst estimate snapshot row is not a dict"))?;
        let forecast_consensus_mean = dict_optional_float(dict, "forecast_consensus_mean")?;
        let forecast_dispersion = dict_optional_float(dict, "forecast_dispersion")?;
        let estimate_count = dict_optional_int(py, dict, "estimate_count")?;
        let key = AnalystEstimateValueKey {
            forecast_consensus_mean: analyst_estimate_float_key(forecast_consensus_mean)?,
            forecast_dispersion: analyst_estimate_float_key(forecast_dispersion)?,
            estimate_count,
        };
        if let Some(existing_key) = seen_key {
            if existing_key != key {
                return Ok((None, true));
            }
        } else {
            seen_key = Some(key);
            selected_values = Some((forecast_consensus_mean, forecast_dispersion, estimate_count));
        }
        let request_group_id = dict_python_or_empty_string(dict, "request_group_id")?;
        if !request_group_id.is_empty() {
            request_groups.insert(request_group_id);
        }
        let effective_collection_ric =
            dict_python_or_empty_string(dict, "effective_collection_ric")?;
        if !effective_collection_ric.is_empty() {
            effective_collection_rics.insert(effective_collection_ric);
        }
    }
    if row_count == 0 {
        return Err(PyValueError::new_err(
            "empty analyst estimate snapshot rows",
        ));
    }
    let Some((forecast_consensus_mean, forecast_dispersion, estimate_count)) = selected_values
    else {
        return Ok((None, false));
    };
    let effective_collection_ric = if effective_collection_rics.len() == 1 {
        effective_collection_rics.iter().next().cloned()
    } else {
        None
    };
    Ok((
        Some((
            forecast_consensus_mean,
            forecast_dispersion,
            estimate_count,
            effective_collection_ric,
            request_groups.into_iter().collect(),
        )),
        false,
    ))
}

#[pyfunction]
pub(crate) fn refinitiv_analyst_freeze_sorted_date_index(
    py: Python<'_>,
    index: &Bound<'_, PyAny>,
) -> PyResult<Vec<(PyObject, Vec<PyObject>)>> {
    let dict = index
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("date index is not a dict"))?;
    let mut out: Vec<(PyObject, Vec<PyObject>)> = Vec::new();
    for (key, values) in dict.iter() {
        let mut entries: Vec<(i32, u32, u32)> = Vec::new();
        for entry in values.iter()? {
            entries.push(extract_py_date_parts(&entry?)?);
        }
        if entries.is_empty() {
            continue;
        }
        entries.sort_unstable();
        let mut dates: Vec<PyObject> = Vec::with_capacity(entries.len());
        for (year, month, day) in entries {
            dates.push(py_date_object(py, year, month, day)?);
        }
        out.push((key.into_py(py), dates));
    }
    Ok(out)
}

pub(crate) struct AnalystDateValue {
    ordinal: i64,
    year: i32,
    month: u32,
    day: u32,
    obj: PyObject,
    iso: String,
}

impl AnalystDateValue {
    fn clone_ref(&self, py: Python<'_>) -> Self {
        Self {
            ordinal: self.ordinal,
            year: self.year,
            month: self.month,
            day: self.day,
            obj: self.obj.clone_ref(py),
            iso: self.iso.clone(),
        }
    }
}

pub(crate) struct AnalystActualRecord {
    gvkey_int: i64,
    announcement_date: AnalystDateValue,
    fiscal_period_end: Option<AnalystDateValue>,
    actual_eps: f64,
    effective_collection_ric: Option<String>,
    source_request_group_ids: Vec<String>,
}

pub(crate) struct AnalystEstimateSnapshotRecord {
    forecast_consensus_mean: Option<f64>,
    forecast_dispersion: Option<f64>,
    effective_collection_ric: Option<String>,
    source_request_group_ids: Vec<String>,
}

pub(crate) struct AnalystNormalizedProvisional {
    gvkey_int: i64,
    announcement_date: AnalystDateValue,
    fiscal_period_end: AnalystDateValue,
    actual_fiscal_period_end_origin: &'static str,
    actual_eps: f64,
    forecast_consensus_mean: f64,
    forecast_dispersion: f64,
    forecast_revision_4m: Option<f64>,
    forecast_revision_4m_status: &'static str,
    forecast_revision_1m: Option<f64>,
    selected_forecast_calc_date: AnalystDateValue,
    revision_base_calc_date_4m: Option<AnalystDateValue>,
    revision_base_calc_date_1m: Option<AnalystDateValue>,
    effective_collection_ric: Option<String>,
    source_request_group_ids: Vec<String>,
}

#[derive(Eq, PartialEq)]
pub(crate) struct AnalystNormalizedValueKey {
    actual_fiscal_period_end_origin: &'static str,
    actual_eps: Option<u64>,
    forecast_consensus_mean: Option<u64>,
    forecast_dispersion: Option<u64>,
    forecast_revision_4m: Option<u64>,
    forecast_revision_4m_status: &'static str,
    forecast_revision_1m: Option<u64>,
    selected_forecast_calc_date: i64,
    revision_base_calc_date_4m: Option<i64>,
    revision_base_calc_date_1m: Option<i64>,
}

pub(crate) enum AnalystRejectionHashPart<'a> {
    Text(&'a str),
    Int(i64),
    Date(&'a AnalystDateValue),
}

pub(crate) fn analyst_date_value_from_py(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
) -> PyResult<AnalystDateValue> {
    let (year, month, day) = extract_py_date_parts(value)?;
    let ordinal = date_ordinal(year, month, day)?;
    Ok(AnalystDateValue {
        ordinal,
        year,
        month,
        day,
        obj: value.into_py(py),
        iso: iso_date_string(year, month, day),
    })
}

pub(crate) fn analyst_optional_date_from_dict(
    py: Python<'_>,
    dict: &Bound<'_, PyDict>,
    key: &str,
) -> PyResult<Option<AnalystDateValue>> {
    let Some(value) = dict.get_item(key)? else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    analyst_date_value_from_py(py, &value).map(Some)
}

pub(crate) fn analyst_required_date_from_dict(
    py: Python<'_>,
    dict: &Bound<'_, PyDict>,
    key: &str,
) -> PyResult<AnalystDateValue> {
    let Some(value) = dict.get_item(key)? else {
        return Err(PyValueError::new_err(format!("missing {key}")));
    };
    if value.is_none() {
        return Err(PyValueError::new_err(format!("null {key}")));
    }
    analyst_date_value_from_py(py, &value)
}

pub(crate) fn analyst_optional_string_from_dict(
    dict: &Bound<'_, PyDict>,
    key: &str,
) -> PyResult<Option<String>> {
    let value = dict_python_or_empty_string(dict, key)?;
    if value.is_empty() {
        Ok(None)
    } else {
        Ok(Some(value))
    }
}

pub(crate) fn analyst_required_i64_from_dict(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<i64> {
    let Some(value) = dict.get_item(key)? else {
        return Err(PyValueError::new_err(format!("missing {key}")));
    };
    py_int_like_to_i64(&value)
}

pub(crate) fn analyst_required_f64_from_dict(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<f64> {
    let value = dict_required_float(dict, key)?;
    if !value.is_finite() {
        return Err(PyValueError::new_err(
            "non-finite analyst normalized value falls back to Python",
        ));
    }
    Ok(value)
}

pub(crate) fn analyst_optional_f64_from_dict(
    dict: &Bound<'_, PyDict>,
    key: &str,
) -> PyResult<Option<f64>> {
    let value = dict_optional_float(dict, key)?;
    if value.is_some_and(|number| !number.is_finite()) {
        return Err(PyValueError::new_err(
            "non-finite analyst normalized value falls back to Python",
        ));
    }
    Ok(value)
}

pub(crate) fn analyst_group_ids_from_dict(
    dict: &Bound<'_, PyDict>,
    key: &str,
) -> PyResult<Vec<String>> {
    let Some(values) = dict.get_item(key)? else {
        return Ok(Vec::new());
    };
    if values.is_none() {
        return Ok(Vec::new());
    }
    let mut out: BTreeSet<String> = BTreeSet::new();
    for value in values.iter()? {
        let value = value?;
        if value.is_truthy()? {
            out.insert(value.str()?.to_str()?.to_string());
        }
    }
    Ok(out.into_iter().collect())
}

pub(crate) fn analyst_parse_actual_records(
    py: Python<'_>,
    rows: &Bound<'_, PyAny>,
) -> PyResult<Vec<AnalystActualRecord>> {
    let mut out = Vec::new();
    for row in rows.iter()? {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("analyst canonical actual row is not a dict"))?;
        out.push(AnalystActualRecord {
            gvkey_int: analyst_required_i64_from_dict(dict, "gvkey_int")?,
            announcement_date: analyst_required_date_from_dict(py, dict, "announcement_date")?,
            fiscal_period_end: analyst_optional_date_from_dict(py, dict, "fiscal_period_end")?,
            actual_eps: analyst_required_f64_from_dict(dict, "actual_eps")?,
            effective_collection_ric: analyst_optional_string_from_dict(
                dict,
                "effective_collection_ric",
            )?,
            source_request_group_ids: analyst_group_ids_from_dict(
                dict,
                "source_request_group_ids",
            )?,
        });
    }
    Ok(out)
}

pub(crate) fn analyst_parse_estimate_records(
    rows: &Bound<'_, PyAny>,
) -> PyResult<HashMap<(i64, i64, i64), AnalystEstimateSnapshotRecord>> {
    let mut out = HashMap::new();
    let py = rows.py();
    for row in rows.iter()? {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("analyst canonical estimate row is not a dict"))?;
        let gvkey_int = analyst_required_i64_from_dict(dict, "gvkey_int")?;
        let fiscal_period_end = analyst_required_date_from_dict(py, dict, "fiscal_period_end")?;
        let calc_date = analyst_required_date_from_dict(py, dict, "calc_date")?;
        out.insert(
            (gvkey_int, fiscal_period_end.ordinal, calc_date.ordinal),
            AnalystEstimateSnapshotRecord {
                forecast_consensus_mean: analyst_optional_f64_from_dict(
                    dict,
                    "forecast_consensus_mean",
                )?,
                forecast_dispersion: analyst_optional_f64_from_dict(dict, "forecast_dispersion")?,
                effective_collection_ric: analyst_optional_string_from_dict(
                    dict,
                    "effective_collection_ric",
                )?,
                source_request_group_ids: analyst_group_ids_from_dict(
                    dict,
                    "source_request_group_ids",
                )?,
            },
        );
    }
    Ok(out)
}

pub(crate) fn analyst_parse_conflicting_estimate_keys(
    py: Python<'_>,
    keys: &Bound<'_, PyAny>,
) -> PyResult<HashSet<(i64, i64, i64)>> {
    let mut out = HashSet::new();
    for key in keys.iter()? {
        let key = key?;
        let tuple = key.downcast::<PyTuple>().map_err(|_| {
            PyValueError::new_err("analyst conflicting estimate key is not a tuple")
        })?;
        if tuple.len() != 3 {
            return Err(PyValueError::new_err(
                "analyst conflicting estimate key must have three elements",
            ));
        }
        let gvkey_int = py_int_like_to_i64(&tuple.get_item(0)?)?;
        let fiscal_period_end = analyst_date_value_from_py(py, &tuple.get_item(1)?)?;
        let calc_date = analyst_date_value_from_py(py, &tuple.get_item(2)?)?;
        out.insert((gvkey_int, fiscal_period_end.ordinal, calc_date.ordinal));
    }
    Ok(out)
}

pub(crate) fn analyst_parse_tuple_i64_date_key(
    py: Python<'_>,
    key: &Bound<'_, PyAny>,
) -> PyResult<(i64, AnalystDateValue)> {
    let tuple = key
        .downcast::<PyTuple>()
        .map_err(|_| PyValueError::new_err("analyst date-index key is not a tuple"))?;
    if tuple.len() != 2 {
        return Err(PyValueError::new_err(
            "analyst date-index key must have two elements",
        ));
    }
    Ok((
        py_int_like_to_i64(&tuple.get_item(0)?)?,
        analyst_date_value_from_py(py, &tuple.get_item(1)?)?,
    ))
}

pub(crate) fn analyst_parse_date_list(
    py: Python<'_>,
    values: &Bound<'_, PyAny>,
) -> PyResult<Vec<AnalystDateValue>> {
    let mut out = Vec::new();
    for value in values.iter()? {
        out.push(analyst_date_value_from_py(py, &value?)?);
    }
    out.sort_by_key(|value| value.ordinal);
    Ok(out)
}

pub(crate) fn analyst_parse_snapshot_date_index(
    py: Python<'_>,
    index: &Bound<'_, PyAny>,
) -> PyResult<HashMap<(i64, i64), Vec<AnalystDateValue>>> {
    let dict = index
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("analyst snapshot date index is not a dict"))?;
    let mut out = HashMap::new();
    for (key, values) in dict.iter() {
        let (gvkey_int, fiscal_period_end) = analyst_parse_tuple_i64_date_key(py, &key)?;
        let dates = analyst_parse_date_list(py, &values)?;
        if !dates.is_empty() {
            out.insert((gvkey_int, fiscal_period_end.ordinal), dates);
        }
    }
    Ok(out)
}

pub(crate) fn analyst_parse_gvkey_date_index(
    py: Python<'_>,
    index: &Bound<'_, PyAny>,
) -> PyResult<HashMap<i64, Vec<AnalystDateValue>>> {
    let dict = index
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("analyst gvkey date index is not a dict"))?;
    let mut out = HashMap::new();
    for (key, values) in dict.iter() {
        let gvkey_int = py_int_like_to_i64(&key)?;
        let dates = analyst_parse_date_list(py, &values)?;
        if !dates.is_empty() {
            out.insert(gvkey_int, dates);
        }
    }
    Ok(out)
}

pub(crate) fn analyst_parse_fiscal_period_end_index(
    py: Python<'_>,
    index: &Bound<'_, PyAny>,
) -> PyResult<HashMap<(i64, i64), Vec<AnalystDateValue>>> {
    let dict = index
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("analyst fiscal-period-end index is not a dict"))?;
    let mut out = HashMap::new();
    for (key, values) in dict.iter() {
        let (gvkey_int, calc_date) = analyst_parse_tuple_i64_date_key(py, &key)?;
        let dates = analyst_parse_date_list(py, &values)?;
        if !dates.is_empty() {
            out.insert((gvkey_int, calc_date.ordinal), dates);
        }
    }
    Ok(out)
}

pub(crate) fn analyst_latest_date_on_or_before(
    py: Python<'_>,
    sorted_dates: Option<&Vec<AnalystDateValue>>,
    cutoff: &AnalystDateValue,
) -> Option<AnalystDateValue> {
    let sorted_dates = sorted_dates?;
    if sorted_dates.is_empty() {
        return None;
    }
    let mut low = 0usize;
    let mut high = sorted_dates.len();
    while low < high {
        let mid = low + (high - low) / 2;
        if sorted_dates[mid].ordinal <= cutoff.ordinal {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    if low == 0 {
        None
    } else {
        Some(sorted_dates[low - 1].clone_ref(py))
    }
}

pub(crate) fn analyst_shift_month_date(
    py: Python<'_>,
    value: &AnalystDateValue,
    months: i64,
) -> PyResult<AnalystDateValue> {
    let month_index = i64::from(value.year) * 12 + i64::from(value.month) - 1 + months;
    let shifted_year = month_index.div_euclid(12);
    let shifted_month = month_index.rem_euclid(12) + 1;
    let shifted_year_i32 = i32::try_from(shifted_year)
        .map_err(|_| PyValueError::new_err("shifted date year out of range"))?;
    let shifted_month_u32 = u32::try_from(shifted_month)
        .map_err(|_| PyValueError::new_err("shifted date month out of range"))?;
    let Some(last_day) = days_in_month(shifted_year_i32, shifted_month_u32) else {
        return Err(PyValueError::new_err("shifted date month out of range"));
    };
    let obj = py_date_object(py, shifted_year_i32, shifted_month_u32, last_day)?;
    let ordinal = date_ordinal(shifted_year_i32, shifted_month_u32, last_day)?;
    Ok(AnalystDateValue {
        ordinal,
        year: shifted_year_i32,
        month: shifted_month_u32,
        day: last_day,
        obj,
        iso: iso_date_string(shifted_year_i32, shifted_month_u32, last_day),
    })
}

pub(crate) fn analyst_push_rejection_hash_part(
    out: &mut String,
    part: &AnalystRejectionHashPart<'_>,
) -> PyResult<()> {
    match part {
        AnalystRejectionHashPart::Text(value) => push_json_ascii_string(out, value)?,
        AnalystRejectionHashPart::Int(value) => out.push_str(&value.to_string()),
        AnalystRejectionHashPart::Date(value) => push_json_ascii_string(out, &value.iso)?,
    }
    Ok(())
}

pub(crate) fn analyst_rejection_case_id(
    parts: &[AnalystRejectionHashPart<'_>],
) -> PyResult<String> {
    let mut payload = String::new();
    payload.push('[');
    for (index, part) in parts.iter().enumerate() {
        if index > 0 {
            payload.push(',');
        }
        analyst_push_rejection_hash_part(&mut payload, part)?;
    }
    payload.push(']');
    Ok(stable_hash_id_from_payload("reject", &payload))
}

pub(crate) fn analyst_set_optional_date(
    py: Python<'_>,
    dict: &Bound<'_, PyDict>,
    key: &str,
    value: Option<&AnalystDateValue>,
) -> PyResult<()> {
    match value {
        Some(value) => dict.set_item(key, value.obj.clone_ref(py)),
        None => dict.set_item(key, py.None()),
    }
}

pub(crate) fn analyst_set_optional_float(
    py: Python<'_>,
    dict: &Bound<'_, PyDict>,
    key: &str,
    value: Option<f64>,
) -> PyResult<()> {
    match value {
        Some(value) => dict.set_item(key, value),
        None => dict.set_item(key, py.None()),
    }
}

pub(crate) fn analyst_rejection_row(
    py: Python<'_>,
    rejection_case_id: String,
    gvkey_int: i64,
    announcement_date: &AnalystDateValue,
    fiscal_period_end: Option<&AnalystDateValue>,
    selected_calc_date: Option<&AnalystDateValue>,
    rejection_status: &str,
    rejection_reason: &str,
    source_request_group_ids: Vec<String>,
) -> PyResult<PyObject> {
    let out = PyDict::new_bound(py);
    out.set_item("rejection_case_id", rejection_case_id)?;
    out.set_item("gvkey_int", gvkey_int)?;
    out.set_item("announcement_date", announcement_date.obj.clone_ref(py))?;
    analyst_set_optional_date(py, &out, "fiscal_period_end", fiscal_period_end)?;
    analyst_set_optional_date(py, &out, "selected_calc_date", selected_calc_date)?;
    out.set_item("rejection_status", rejection_status)?;
    out.set_item("rejection_reason", rejection_reason)?;
    out.set_item("source_request_group_ids", source_request_group_ids)?;
    Ok(out.into_py(py))
}

pub(crate) fn analyst_sorted_group_union<'a>(
    groups: impl Iterator<Item = &'a String>,
) -> Vec<String> {
    groups
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

pub(crate) fn analyst_normalized_value_key(
    row: &AnalystNormalizedProvisional,
) -> PyResult<AnalystNormalizedValueKey> {
    Ok(AnalystNormalizedValueKey {
        actual_fiscal_period_end_origin: row.actual_fiscal_period_end_origin,
        actual_eps: analyst_estimate_float_key(Some(row.actual_eps))?,
        forecast_consensus_mean: analyst_estimate_float_key(Some(row.forecast_consensus_mean))?,
        forecast_dispersion: analyst_estimate_float_key(Some(row.forecast_dispersion))?,
        forecast_revision_4m: analyst_estimate_float_key(row.forecast_revision_4m)?,
        forecast_revision_4m_status: row.forecast_revision_4m_status,
        forecast_revision_1m: analyst_estimate_float_key(row.forecast_revision_1m)?,
        selected_forecast_calc_date: row.selected_forecast_calc_date.ordinal,
        revision_base_calc_date_4m: row
            .revision_base_calc_date_4m
            .as_ref()
            .map(|date| date.ordinal),
        revision_base_calc_date_1m: row
            .revision_base_calc_date_1m
            .as_ref()
            .map(|date| date.ordinal),
    })
}

pub(crate) fn analyst_normalized_row_to_dict(
    py: Python<'_>,
    row: &AnalystNormalizedProvisional,
    effective_collection_ric: Option<String>,
    source_request_group_ids: Vec<String>,
) -> PyResult<PyObject> {
    let out = PyDict::new_bound(py);
    out.set_item("gvkey_int", row.gvkey_int)?;
    out.set_item("announcement_date", row.announcement_date.obj.clone_ref(py))?;
    out.set_item("fiscal_period_end", row.fiscal_period_end.obj.clone_ref(py))?;
    out.set_item(
        "actual_fiscal_period_end_origin",
        row.actual_fiscal_period_end_origin,
    )?;
    out.set_item("actual_eps", row.actual_eps)?;
    out.set_item("forecast_consensus_mean", row.forecast_consensus_mean)?;
    out.set_item("forecast_dispersion", row.forecast_dispersion)?;
    analyst_set_optional_float(py, &out, "forecast_revision_4m", row.forecast_revision_4m)?;
    out.set_item(
        "forecast_revision_4m_status",
        row.forecast_revision_4m_status,
    )?;
    analyst_set_optional_float(py, &out, "forecast_revision_1m", row.forecast_revision_1m)?;
    out.set_item(
        "selected_forecast_calc_date",
        row.selected_forecast_calc_date.obj.clone_ref(py),
    )?;
    analyst_set_optional_date(
        py,
        &out,
        "revision_base_calc_date_4m",
        row.revision_base_calc_date_4m.as_ref(),
    )?;
    analyst_set_optional_date(
        py,
        &out,
        "revision_base_calc_date_1m",
        row.revision_base_calc_date_1m.as_ref(),
    )?;
    match effective_collection_ric {
        Some(value) => out.set_item("effective_collection_ric", value)?,
        None => out.set_item("effective_collection_ric", py.None())?,
    };
    out.set_item("source_request_group_ids", source_request_group_ids)?;
    Ok(out.into_py(py))
}

#[pyfunction]
pub(crate) fn refinitiv_analyst_build_normalized_event_rows(
    py: Python<'_>,
    canonical_actuals: &Bound<'_, PyAny>,
    canonical_estimates: &Bound<'_, PyAny>,
    conflicting_estimate_keys: &Bound<'_, PyAny>,
    estimate_calc_dates_by_snapshot_key: &Bound<'_, PyAny>,
    estimate_calc_dates_by_gvkey: &Bound<'_, PyAny>,
    estimate_fiscal_period_ends_by_gvkey_calc_date: &Bound<'_, PyAny>,
) -> PyResult<(Vec<PyObject>, Vec<PyObject>)> {
    let canonical_actuals = analyst_parse_actual_records(py, canonical_actuals)?;
    let canonical_estimates = analyst_parse_estimate_records(canonical_estimates)?;
    let conflicting_estimate_keys =
        analyst_parse_conflicting_estimate_keys(py, conflicting_estimate_keys)?;
    let estimate_calc_dates_by_snapshot_key =
        analyst_parse_snapshot_date_index(py, estimate_calc_dates_by_snapshot_key)?;
    let estimate_calc_dates_by_gvkey =
        analyst_parse_gvkey_date_index(py, estimate_calc_dates_by_gvkey)?;
    let estimate_fiscal_period_ends_by_gvkey_calc_date =
        analyst_parse_fiscal_period_end_index(py, estimate_fiscal_period_ends_by_gvkey_calc_date)?;

    let mut rejection_rows: Vec<PyObject> = Vec::new();
    let mut provisional_rows: Vec<AnalystNormalizedProvisional> = Vec::new();

    for event in canonical_actuals.iter() {
        let gvkey_int = event.gvkey_int;
        let request_groups: Vec<String> =
            analyst_sorted_group_union(event.source_request_group_ids.iter());
        let mut actual_fiscal_period_end_origin = "API_DIRECT";
        let fiscal_period_end: AnalystDateValue;
        let selected_calc_date: AnalystDateValue;

        if let Some(event_fiscal_period_end) = event.fiscal_period_end.as_ref() {
            fiscal_period_end = event_fiscal_period_end.clone_ref(py);
            let snapshot_dates =
                estimate_calc_dates_by_snapshot_key.get(&(gvkey_int, fiscal_period_end.ordinal));
            let Some(selected) =
                analyst_latest_date_on_or_before(py, snapshot_dates, &event.announcement_date)
            else {
                let rejection_case_id = analyst_rejection_case_id(&[
                    AnalystRejectionHashPart::Text("analyst_rejection"),
                    AnalystRejectionHashPart::Int(gvkey_int),
                    AnalystRejectionHashPart::Date(&event.announcement_date),
                    AnalystRejectionHashPart::Date(&fiscal_period_end),
                    AnalystRejectionHashPart::Text("missing_selected_estimate_snapshot"),
                ])?;
                rejection_rows.push(analyst_rejection_row(
                    py,
                    rejection_case_id,
                    gvkey_int,
                    &event.announcement_date,
                    Some(&fiscal_period_end),
                    None,
                    "MISSING_SELECTED_ESTIMATE_SNAPSHOT",
                    "no estimate snapshot existed on or before the announcement date",
                    request_groups,
                )?);
                continue;
            };
            selected_calc_date = selected;
        } else {
            let Some(selected) = analyst_latest_date_on_or_before(
                py,
                estimate_calc_dates_by_gvkey.get(&gvkey_int),
                &event.announcement_date,
            ) else {
                let rejection_case_id = analyst_rejection_case_id(&[
                    AnalystRejectionHashPart::Text("analyst_rejection"),
                    AnalystRejectionHashPart::Int(gvkey_int),
                    AnalystRejectionHashPart::Date(&event.announcement_date),
                    AnalystRejectionHashPart::Text("missing_derived_fiscal_period_end"),
                ])?;
                rejection_rows.push(analyst_rejection_row(
                    py,
                    rejection_case_id,
                    gvkey_int,
                    &event.announcement_date,
                    None,
                    None,
                    "MISSING_DERIVED_FISCAL_PERIOD_END",
                    "actual event is missing fiscal period end and no estimate snapshots were available",
                    request_groups,
                )?);
                continue;
            };
            let derived_fiscal_period_ends =
                estimate_fiscal_period_ends_by_gvkey_calc_date.get(&(gvkey_int, selected.ordinal));
            let derived_count = derived_fiscal_period_ends.map_or(0, Vec::len);
            if derived_count == 0 {
                let rejection_case_id = analyst_rejection_case_id(&[
                    AnalystRejectionHashPart::Text("analyst_rejection"),
                    AnalystRejectionHashPart::Int(gvkey_int),
                    AnalystRejectionHashPart::Date(&event.announcement_date),
                    AnalystRejectionHashPart::Date(&selected),
                    AnalystRejectionHashPart::Text("missing_derived_fiscal_period_end"),
                ])?;
                rejection_rows.push(analyst_rejection_row(
                    py,
                    rejection_case_id,
                    gvkey_int,
                    &event.announcement_date,
                    None,
                    Some(&selected),
                    "MISSING_DERIVED_FISCAL_PERIOD_END",
                    "latest estimate-side calc date did not expose a fiscal period end",
                    request_groups,
                )?);
                continue;
            }
            if derived_count > 1 {
                let rejection_case_id = analyst_rejection_case_id(&[
                    AnalystRejectionHashPart::Text("analyst_rejection"),
                    AnalystRejectionHashPart::Int(gvkey_int),
                    AnalystRejectionHashPart::Date(&event.announcement_date),
                    AnalystRejectionHashPart::Date(&selected),
                    AnalystRejectionHashPart::Text("nonunique_derived_fiscal_period_end"),
                ])?;
                rejection_rows.push(analyst_rejection_row(
                    py,
                    rejection_case_id,
                    gvkey_int,
                    &event.announcement_date,
                    None,
                    Some(&selected),
                    "NONUNIQUE_DERIVED_FISCAL_PERIOD_END",
                    "latest estimate-side calc date implied multiple fiscal period ends",
                    request_groups,
                )?);
                continue;
            }
            let derived = derived_fiscal_period_ends.expect("checked above");
            fiscal_period_end = derived[0].clone_ref(py);
            selected_calc_date = selected;
            actual_fiscal_period_end_origin = "ESTIMATE_FALLBACK";
        }

        let snapshot_calc_dates =
            estimate_calc_dates_by_snapshot_key.get(&(gvkey_int, fiscal_period_end.ordinal));
        let selected_snapshot_key = (
            gvkey_int,
            fiscal_period_end.ordinal,
            selected_calc_date.ordinal,
        );
        if conflicting_estimate_keys.contains(&selected_snapshot_key) {
            let rejection_case_id = analyst_rejection_case_id(&[
                AnalystRejectionHashPart::Text("analyst_rejection"),
                AnalystRejectionHashPart::Int(gvkey_int),
                AnalystRejectionHashPart::Date(&event.announcement_date),
                AnalystRejectionHashPart::Date(&fiscal_period_end),
                AnalystRejectionHashPart::Date(&selected_calc_date),
                AnalystRejectionHashPart::Text("conflicting_selected_estimate_snapshot"),
            ])?;
            rejection_rows.push(analyst_rejection_row(
                py,
                rejection_case_id,
                gvkey_int,
                &event.announcement_date,
                Some(&fiscal_period_end),
                Some(&selected_calc_date),
                "CONFLICTING_SELECTED_ESTIMATE_SNAPSHOT",
                "selected estimate snapshot had conflicting duplicate values",
                request_groups,
            )?);
            continue;
        }

        let selected_snapshot = canonical_estimates.get(&selected_snapshot_key);
        let selected_mean = selected_snapshot.and_then(|snapshot| snapshot.forecast_consensus_mean);
        let selected_dispersion =
            selected_snapshot.and_then(|snapshot| snapshot.forecast_dispersion);
        let (Some(selected_snapshot), Some(selected_mean), Some(selected_dispersion)) =
            (selected_snapshot, selected_mean, selected_dispersion)
        else {
            let rejection_case_id = analyst_rejection_case_id(&[
                AnalystRejectionHashPart::Text("analyst_rejection"),
                AnalystRejectionHashPart::Int(gvkey_int),
                AnalystRejectionHashPart::Date(&event.announcement_date),
                AnalystRejectionHashPart::Date(&fiscal_period_end),
                AnalystRejectionHashPart::Date(&selected_calc_date),
                AnalystRejectionHashPart::Text("missing_selected_estimate_snapshot"),
            ])?;
            rejection_rows.push(analyst_rejection_row(
                py,
                rejection_case_id,
                gvkey_int,
                &event.announcement_date,
                Some(&fiscal_period_end),
                Some(&selected_calc_date),
                "MISSING_SELECTED_ESTIMATE_SNAPSHOT",
                "selected estimate snapshot was missing or did not expose a consensus mean and dispersion",
                request_groups,
            )?);
            continue;
        };

        let revision_4m_cutoff = analyst_shift_month_date(py, &selected_calc_date, -4)?;
        let revision_base_calc_date_4m =
            analyst_latest_date_on_or_before(py, snapshot_calc_dates, &revision_4m_cutoff);
        let mut forecast_revision_4m = None;
        let mut forecast_revision_4m_status = "OK";
        if let Some(revision_base_calc_date_4m_value) = revision_base_calc_date_4m.as_ref() {
            let revision_base_key_4m = (
                gvkey_int,
                fiscal_period_end.ordinal,
                revision_base_calc_date_4m_value.ordinal,
            );
            if conflicting_estimate_keys.contains(&revision_base_key_4m) {
                let rejection_case_id = analyst_rejection_case_id(&[
                    AnalystRejectionHashPart::Text("analyst_rejection"),
                    AnalystRejectionHashPart::Int(gvkey_int),
                    AnalystRejectionHashPart::Date(&event.announcement_date),
                    AnalystRejectionHashPart::Date(&fiscal_period_end),
                    AnalystRejectionHashPart::Date(revision_base_calc_date_4m_value),
                    AnalystRejectionHashPart::Text("conflicting_revision_base_4m"),
                ])?;
                rejection_rows.push(analyst_rejection_row(
                    py,
                    rejection_case_id,
                    gvkey_int,
                    &event.announcement_date,
                    Some(&fiscal_period_end),
                    Some(&selected_calc_date),
                    "CONFLICTING_REVISION_BASE_4M",
                    "four-month revision base snapshot had conflicting duplicate values",
                    request_groups,
                )?);
                continue;
            }
            match canonical_estimates
                .get(&revision_base_key_4m)
                .and_then(|snapshot| snapshot.forecast_consensus_mean)
            {
                Some(base_mean) => forecast_revision_4m = Some(selected_mean - base_mean),
                None => forecast_revision_4m_status = "MISSING_BASE_CONSENSUS_MEAN",
            }
        } else {
            forecast_revision_4m_status = "MISSING_BASE_SNAPSHOT";
        }

        let revision_1m_cutoff = analyst_shift_month_date(py, &selected_calc_date, -1)?;
        let revision_base_calc_date_1m =
            analyst_latest_date_on_or_before(py, snapshot_calc_dates, &revision_1m_cutoff);
        let mut forecast_revision_1m = None;
        if let Some(revision_base_calc_date_1m_value) = revision_base_calc_date_1m.as_ref() {
            let revision_base_key_1m = (
                gvkey_int,
                fiscal_period_end.ordinal,
                revision_base_calc_date_1m_value.ordinal,
            );
            if !conflicting_estimate_keys.contains(&revision_base_key_1m) {
                if let Some(base_mean) = canonical_estimates
                    .get(&revision_base_key_1m)
                    .and_then(|snapshot| snapshot.forecast_consensus_mean)
                {
                    forecast_revision_1m = Some(selected_mean - base_mean);
                }
            }
        }

        let mut source_request_group_ids = BTreeSet::new();
        for value in event.source_request_group_ids.iter() {
            source_request_group_ids.insert(value.clone());
        }
        for value in selected_snapshot.source_request_group_ids.iter() {
            source_request_group_ids.insert(value.clone());
        }
        let effective_collection_ric = event
            .effective_collection_ric
            .clone()
            .or_else(|| selected_snapshot.effective_collection_ric.clone());

        provisional_rows.push(AnalystNormalizedProvisional {
            gvkey_int,
            announcement_date: event.announcement_date.clone_ref(py),
            fiscal_period_end,
            actual_fiscal_period_end_origin,
            actual_eps: event.actual_eps,
            forecast_consensus_mean: selected_mean,
            forecast_dispersion: selected_dispersion,
            forecast_revision_4m,
            forecast_revision_4m_status,
            forecast_revision_1m,
            selected_forecast_calc_date: selected_calc_date,
            revision_base_calc_date_4m,
            revision_base_calc_date_1m,
            effective_collection_ric,
            source_request_group_ids: source_request_group_ids.into_iter().collect(),
        });
    }

    let mut group_index: HashMap<(i64, i64, i64), usize> = HashMap::new();
    let mut groups: Vec<Vec<AnalystNormalizedProvisional>> = Vec::new();
    for row in provisional_rows {
        let key = (
            row.gvkey_int,
            row.announcement_date.ordinal,
            row.fiscal_period_end.ordinal,
        );
        let index = match group_index.get(&key) {
            Some(index) => *index,
            None => {
                let index = groups.len();
                group_index.insert(key, index);
                groups.push(Vec::new());
                index
            }
        };
        groups[index].push(row);
    }

    let mut normalized_rows: Vec<PyObject> = Vec::new();
    for rows in groups.iter() {
        let Some(first_row) = rows.first() else {
            continue;
        };
        let first_key = analyst_normalized_value_key(first_row)?;
        let mut conflicting = false;
        for row in rows.iter().skip(1) {
            if analyst_normalized_value_key(row)? != first_key {
                conflicting = true;
                break;
            }
        }

        if conflicting {
            let source_request_group_ids = rows
                .iter()
                .flat_map(|row| row.source_request_group_ids.iter())
                .cloned()
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            let rejection_case_id = analyst_rejection_case_id(&[
                AnalystRejectionHashPart::Text("analyst_rejection"),
                AnalystRejectionHashPart::Int(first_row.gvkey_int),
                AnalystRejectionHashPart::Date(&first_row.announcement_date),
                AnalystRejectionHashPart::Date(&first_row.fiscal_period_end),
                AnalystRejectionHashPart::Text("conflicting_normalized_event"),
            ])?;
            rejection_rows.push(analyst_rejection_row(
                py,
                rejection_case_id,
                first_row.gvkey_int,
                &first_row.announcement_date,
                Some(&first_row.fiscal_period_end),
                None,
                "CONFLICTING_NORMALIZED_EVENT",
                "multiple request groups produced conflicting normalized values for the same event key",
                source_request_group_ids,
            )?);
            continue;
        }

        let effective_collection_rics = rows
            .iter()
            .filter_map(|row| row.effective_collection_ric.as_ref())
            .cloned()
            .collect::<BTreeSet<_>>();
        let effective_collection_ric = if effective_collection_rics.len() == 1 {
            effective_collection_rics.into_iter().next()
        } else {
            None
        };
        let source_request_group_ids = rows
            .iter()
            .flat_map(|row| row.source_request_group_ids.iter())
            .cloned()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        normalized_rows.push(analyst_normalized_row_to_dict(
            py,
            first_row,
            effective_collection_ric,
            source_request_group_ids,
        )?);
    }

    Ok((normalized_rows, rejection_rows))
}

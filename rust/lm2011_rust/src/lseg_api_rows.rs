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
use crate::lseg_ops::*;
use crate::multisurface_audit::*;
use crate::refinitiv_analyst::*;
use crate::refinitiv_authority::*;
use crate::refinitiv_bridge::*;
use crate::refinitiv_excel::*;
use crate::sec_extraction::*;
use crate::sentence_cleaning::*;
use crate::sentence_quality_api::*;

#[derive(Clone, Default)]
pub(crate) struct LookupBatchReturnedValues {
    returned_ric: Option<String>,
    returned_name: Option<String>,
    returned_isin: Option<String>,
    returned_cusip: Option<String>,
}

pub(crate) type LookupBatchOutputRow = (
    String,
    String,
    String,
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
);

pub(crate) fn dict_required_string(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<String> {
    let Some(value) = dict.get_item(key)? else {
        return Err(PyValueError::new_err(format!(
            "missing required key: {key}"
        )));
    };
    if value.is_none() {
        return Err(PyValueError::new_err(format!("null required key: {key}")));
    }
    Ok(value.str()?.to_str()?.to_string())
}

pub(crate) fn dict_normalized_string(
    dict: &Bound<'_, PyDict>,
    key: &str,
) -> PyResult<Option<String>> {
    let Some(value) = dict.get_item(key)? else {
        return Ok(None);
    };
    normalize_lookup_text_any_impl(Some(&value))
}

#[pyfunction]
pub(crate) fn normalize_lookup_batch_response_rows_value(
    item_rows: &Bound<'_, PyAny>,
    response_rows: &Bound<'_, PyAny>,
) -> PyResult<Vec<LookupBatchOutputRow>> {
    let mut response_by_instrument: HashMap<String, LookupBatchReturnedValues> = HashMap::new();
    for row in response_rows.iter()? {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("response row is not a dict"))?;
        let Some(instrument) = dict_normalized_string(dict, "instrument")? else {
            continue;
        };
        response_by_instrument.insert(
            instrument,
            LookupBatchReturnedValues {
                returned_ric: dict_normalized_string(dict, "TR.RIC")?,
                returned_name: dict_normalized_string(dict, "TR.CommonName")?,
                returned_isin: dict_normalized_string(dict, "TR.ISIN")?,
                returned_cusip: dict_normalized_string(dict, "TR.CUSIP")?,
            },
        );
    }

    let mut rows: Vec<LookupBatchOutputRow> = Vec::new();
    for item in item_rows.iter()? {
        let item = item?;
        let dict = item
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("item row is not a dict"))?;
        let item_id = dict_required_string(dict, "item_id")?;
        let bridge_row_id = dict_required_string(dict, "bridge_row_id")?;
        let identifier_type = dict_required_string(dict, "identifier_type")?;
        let instrument = dict_normalized_string(dict, "instrument")?;
        let matched = instrument
            .as_ref()
            .and_then(|value| response_by_instrument.get(value));
        rows.push((
            item_id,
            bridge_row_id,
            identifier_type,
            instrument,
            matched.and_then(|value| value.returned_ric.clone()),
            matched.and_then(|value| value.returned_name.clone()),
            matched.and_then(|value| value.returned_isin.clone()),
            matched.and_then(|value| value.returned_cusip.clone()),
        ));
    }
    Ok(rows)
}

#[pyfunction]
pub(crate) fn normalize_lookup_batch_response_columns(
    py: Python<'_>,
    item_ids: &Bound<'_, PyAny>,
    bridge_row_ids: &Bound<'_, PyAny>,
    identifier_types: &Bound<'_, PyAny>,
    instruments: &Bound<'_, PyAny>,
    response_column_names: Vec<String>,
    response_column_values: &Bound<'_, PyAny>,
) -> PyResult<Vec<LookupBatchOutputRow>> {
    let item_ids = required_string_sequence(item_ids, "item_id")?;
    let bridge_row_ids = required_string_sequence(bridge_row_ids, "bridge_row_id")?;
    let identifier_types = required_string_sequence(identifier_types, "identifier_type")?;
    let instruments = pyobject_sequence(py, instruments)?;
    let item_count = item_ids.len();
    if bridge_row_ids.len() != item_count
        || identifier_types.len() != item_count
        || instruments.len() != item_count
    {
        return Err(PyValueError::new_err(
            "all LSEG lookup response item columns must have the same length",
        ));
    }

    let label = "LSEG lookup batch response";
    let (response_columns, response_row_count) =
        collect_pyobject_column_values(py, &response_column_names, response_column_values, label)?;
    let response_column_index = column_index_by_name(&response_column_names);
    let instrument_idx = required_named_column_index(&response_column_index, label, "instrument")?;

    let mut response_by_instrument: HashMap<String, LookupBatchReturnedValues> = HashMap::new();
    for row_idx in 0..response_row_count {
        let Some(instrument) = normalize_lookup_text_any_impl(Some(
            response_columns[instrument_idx][row_idx].bind(py),
        ))?
        else {
            continue;
        };
        response_by_instrument.insert(
            instrument,
            LookupBatchReturnedValues {
                returned_ric: optional_column_value(
                    &response_columns,
                    &response_column_index,
                    row_idx,
                    "TR.RIC",
                )
                .map(|value| normalize_lookup_text_any_impl(Some(value.bind(py))))
                .transpose()?
                .flatten(),
                returned_name: optional_column_value(
                    &response_columns,
                    &response_column_index,
                    row_idx,
                    "TR.CommonName",
                )
                .map(|value| normalize_lookup_text_any_impl(Some(value.bind(py))))
                .transpose()?
                .flatten(),
                returned_isin: optional_column_value(
                    &response_columns,
                    &response_column_index,
                    row_idx,
                    "TR.ISIN",
                )
                .map(|value| normalize_lookup_text_any_impl(Some(value.bind(py))))
                .transpose()?
                .flatten(),
                returned_cusip: optional_column_value(
                    &response_columns,
                    &response_column_index,
                    row_idx,
                    "TR.CUSIP",
                )
                .map(|value| normalize_lookup_text_any_impl(Some(value.bind(py))))
                .transpose()?
                .flatten(),
            },
        );
    }

    let mut rows: Vec<LookupBatchOutputRow> = Vec::with_capacity(item_count);
    for row_idx in 0..item_count {
        let instrument = normalize_lookup_text_any_impl(Some(instruments[row_idx].bind(py)))?;
        let matched = instrument
            .as_ref()
            .and_then(|value| response_by_instrument.get(value));
        rows.push((
            item_ids[row_idx].clone(),
            bridge_row_ids[row_idx].clone(),
            identifier_types[row_idx].clone(),
            instrument,
            matched.and_then(|value| value.returned_ric.clone()),
            matched.and_then(|value| value.returned_name.clone()),
            matched.and_then(|value| value.returned_isin.clone()),
            matched.and_then(|value| value.returned_cusip.clone()),
        ));
    }
    Ok(rows)
}

pub(crate) type LookupItemRow = (String, String, String, String);

pub(crate) fn lookup_item_id_for_bridge_identifier(
    bridge_row_id: &str,
    identifier_type: &str,
) -> PyResult<String> {
    let mut payload = String::new();
    payload.push('[');
    push_json_ascii_string(&mut payload, "lookup")?;
    payload.push(',');
    push_json_ascii_string(&mut payload, bridge_row_id)?;
    payload.push(',');
    push_json_ascii_string(&mut payload, identifier_type)?;
    payload.push(']');
    Ok(stable_hash_id_from_payload("item", &payload))
}

#[pyfunction]
pub(crate) fn build_lookup_item_rows_value(
    snapshot_rows: &Bound<'_, PyAny>,
    identifier_types: &Bound<'_, PyAny>,
) -> PyResult<Vec<LookupItemRow>> {
    let mut identifiers: Vec<String> = Vec::new();
    for identifier in identifier_types.iter()? {
        let identifier = identifier?;
        identifiers.push(identifier.str()?.to_str()?.to_string());
    }

    let mut rows: Vec<LookupItemRow> = Vec::new();
    for row in snapshot_rows.iter()? {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("snapshot row is not a dict"))?;
        let Some(bridge_row_id) = dict_normalized_string(dict, "bridge_row_id")? else {
            continue;
        };
        for identifier_type in identifiers.iter() {
            let lookup_column = format!("{identifier_type}_lookup_input");
            let Some(lookup_input) = dict_normalized_string(dict, &lookup_column)? else {
                continue;
            };
            rows.push((
                lookup_item_id_for_bridge_identifier(&bridge_row_id, identifier_type)?,
                bridge_row_id.clone(),
                identifier_type.clone(),
                lookup_input,
            ));
        }
    }
    Ok(rows)
}

#[pyfunction]
pub(crate) fn build_lookup_item_row_columns(
    py: Python<'_>,
    column_names: Vec<String>,
    column_values: &Bound<'_, PyAny>,
    identifier_types: &Bound<'_, PyAny>,
) -> PyResult<Vec<LookupItemRow>> {
    let mut identifiers: Vec<String> = Vec::new();
    for identifier in identifier_types.iter()? {
        let identifier = identifier?;
        identifiers.push(identifier.str()?.to_str()?.to_string());
    }
    let label = "LSEG lookup item rows";
    let (columns, row_count) =
        collect_pyobject_column_values(py, &column_names, column_values, label)?;
    let column_index = column_index_by_name(&column_names);
    let bridge_row_idx = required_named_column_index(&column_index, label, "bridge_row_id")?;

    let mut rows: Vec<LookupItemRow> = Vec::new();
    for row_idx in 0..row_count {
        let Some(bridge_row_id) =
            normalize_lookup_text_any_impl(Some(columns[bridge_row_idx][row_idx].bind(py)))?
        else {
            continue;
        };
        for identifier_type in identifiers.iter() {
            let lookup_column = format!("{identifier_type}_lookup_input");
            let Some(raw_lookup_input) =
                optional_column_value(&columns, &column_index, row_idx, &lookup_column)
            else {
                continue;
            };
            let Some(lookup_input) =
                normalize_lookup_text_any_impl(Some(raw_lookup_input.bind(py)))?
            else {
                continue;
            };
            rows.push((
                lookup_item_id_for_bridge_identifier(&bridge_row_id, identifier_type)?,
                bridge_row_id.clone(),
                identifier_type.clone(),
                lookup_input,
            ));
        }
    }
    Ok(rows)
}

pub(crate) type AnalystItemRow = (usize, String, String, String, String, Option<String>);
pub(crate) type AnalystActualBatchResponseRow = (
    usize,
    i64,
    String,
    Option<String>,
    Option<f64>,
    Option<String>,
    String,
);
pub(crate) type AnalystEstimateBatchResponseRow = (
    usize,
    i64,
    String,
    Option<String>,
    Option<String>,
    Option<f64>,
    Option<f64>,
    Option<i64>,
    String,
);
pub(crate) type OwnershipItemRow = (usize, String, String, String, String);
pub(crate) type OwnershipUniverseBatchResponseRow =
    (usize, String, String, Option<String>, Option<f64>);
pub(crate) type DocOwnershipBatchResponseRow =
    (usize, Option<String>, Option<String>, Option<f64>, bool);

pub(crate) struct ParsedAnalystActualResponseRow {
    announcement_date_text: Option<String>,
    announcement_date_ordinal: Option<i64>,
    fiscal_period_end_text: Option<String>,
    actual_eps: Option<f64>,
    raw_fperiod: Option<String>,
}

pub(crate) struct ParsedAnalystEstimateResponseRow {
    calc_date_text: Option<String>,
    calc_date_ordinal: Option<i64>,
    fiscal_period_end_text: Option<String>,
    raw_fperiod: Option<String>,
    forecast_consensus_mean: Option<f64>,
    forecast_dispersion: Option<f64>,
    estimate_count: Option<i64>,
}

pub(crate) struct ParsedOwnershipResponseRow {
    date_text: Option<String>,
    date_ordinal: Option<i64>,
    value: Option<f64>,
    lookup_category: Option<String>,
    doc_category: Option<String>,
}

pub(crate) fn py_date_iso_string(value: &Bound<'_, PyAny>) -> PyResult<String> {
    let (year, month, day) = extract_py_date_parts(value)?;
    Ok(format!("{year:04}-{month:02}-{day:02}"))
}

pub(crate) fn analyst_item_id(
    stage: &str,
    request_group_id: &Bound<'_, PyAny>,
    period: Option<&str>,
) -> PyResult<String> {
    let mut payload = String::new();
    payload.push('[');
    push_json_ascii_string(&mut payload, stage)?;
    payload.push(',');
    push_stable_json_simple_part(&mut payload, request_group_id)?;
    if let Some(period) = period {
        payload.push(',');
        push_json_ascii_string(&mut payload, period)?;
    }
    payload.push(']');
    Ok(stable_hash_id_from_payload("item", &payload))
}

pub(crate) fn analyst_common_item_fields<'py>(
    dict: &Bound<'py, PyDict>,
    start_field: &str,
    end_field: &str,
) -> PyResult<Option<(Bound<'py, PyAny>, String, String, String)>> {
    let Some(eligible) = dict.get_item("retrieval_eligible")? else {
        return Ok(None);
    };
    if eligible.is_none() || !eligible.is_truthy()? {
        return Ok(None);
    }
    let Some(request_group_id) = dict.get_item("request_group_id")? else {
        return Ok(None);
    };
    if request_group_id.is_none() {
        return Ok(None);
    }
    let Some(ric_value) = dict.get_item("effective_collection_ric")? else {
        return Ok(None);
    };
    if ric_value.is_none() {
        return Ok(None);
    }
    let Some(start_date) = dict.get_item(start_field)? else {
        return Ok(None);
    };
    if start_date.is_none() {
        return Ok(None);
    }
    let Some(end_date) = dict.get_item(end_field)? else {
        return Ok(None);
    };
    if end_date.is_none() {
        return Ok(None);
    }
    Ok(Some((
        request_group_id,
        ric_value.str()?.to_str()?.to_string(),
        py_date_iso_string(&start_date)?,
        py_date_iso_string(&end_date)?,
    )))
}

#[pyfunction]
pub(crate) fn build_analyst_actual_item_rows_value(
    rows: &Bound<'_, PyAny>,
) -> PyResult<Vec<AnalystItemRow>> {
    let mut out: Vec<AnalystItemRow> = Vec::new();
    for (row_index, row) in rows.iter()?.enumerate() {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("analyst request row is not a dict"))?;
        let Some((request_group_id, ric, start_text, end_text)) = analyst_common_item_fields(
            dict,
            "actuals_request_start_date",
            "actuals_request_end_date",
        )?
        else {
            continue;
        };
        out.push((
            row_index,
            analyst_item_id("analyst_actuals", &request_group_id, None)?,
            ric,
            start_text,
            end_text,
            None,
        ));
    }
    Ok(out)
}

pub(crate) fn analyst_common_item_columns<'a>(
    py: Python<'_>,
    columns: &'a [Vec<PyObject>],
    column_index: &HashMap<String, usize>,
    row_idx: usize,
    start_field: &str,
    end_field: &str,
) -> PyResult<Option<(&'a PyObject, String, String, String)>> {
    let retrieval_eligible =
        optional_column_value(columns, column_index, row_idx, "retrieval_eligible")
            .map(|value| {
                let value = value.bind(py);
                if value.is_none() {
                    Ok(false)
                } else {
                    value.is_truthy()
                }
            })
            .transpose()?
            .unwrap_or(false);
    if !retrieval_eligible {
        return Ok(None);
    }
    let Some(request_group_id) =
        optional_column_value(columns, column_index, row_idx, "request_group_id")
    else {
        return Ok(None);
    };
    if request_group_id.bind(py).is_none() {
        return Ok(None);
    }
    let Some(ric_value) =
        optional_column_value(columns, column_index, row_idx, "effective_collection_ric")
    else {
        return Ok(None);
    };
    let ric_value = ric_value.bind(py);
    if ric_value.is_none() {
        return Ok(None);
    }
    let Some(start_date) = optional_column_value(columns, column_index, row_idx, start_field)
    else {
        return Ok(None);
    };
    let start_date = start_date.bind(py);
    if start_date.is_none() {
        return Ok(None);
    }
    let Some(end_date) = optional_column_value(columns, column_index, row_idx, end_field) else {
        return Ok(None);
    };
    let end_date = end_date.bind(py);
    if end_date.is_none() {
        return Ok(None);
    }
    Ok(Some((
        request_group_id,
        ric_value.str()?.to_str()?.to_string(),
        py_date_iso_string(start_date)?,
        py_date_iso_string(end_date)?,
    )))
}

#[pyfunction]
pub(crate) fn build_analyst_actual_item_row_columns(
    py: Python<'_>,
    column_names: Vec<String>,
    column_values: &Bound<'_, PyAny>,
) -> PyResult<Vec<AnalystItemRow>> {
    let label = "analyst actual item rows";
    let (columns, row_count) =
        collect_pyobject_column_values(py, &column_names, column_values, label)?;
    let column_index = column_index_by_name(&column_names);
    let mut out: Vec<AnalystItemRow> = Vec::new();
    for row_idx in 0..row_count {
        let Some((request_group_id, ric, start_text, end_text)) = analyst_common_item_columns(
            py,
            &columns,
            &column_index,
            row_idx,
            "actuals_request_start_date",
            "actuals_request_end_date",
        )?
        else {
            continue;
        };
        out.push((
            row_idx,
            analyst_item_id("analyst_actuals", request_group_id.bind(py), None)?,
            ric,
            start_text,
            end_text,
            None,
        ));
    }
    Ok(out)
}

#[pyfunction]
pub(crate) fn build_analyst_estimate_item_rows_value(
    rows: &Bound<'_, PyAny>,
    request_periods: &Bound<'_, PyAny>,
) -> PyResult<Vec<AnalystItemRow>> {
    let mut periods: Vec<String> = Vec::new();
    for period in request_periods.iter()? {
        periods.push(period?.str()?.to_str()?.to_string());
    }
    let mut out: Vec<AnalystItemRow> = Vec::new();
    for (row_index, row) in rows.iter()?.enumerate() {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("analyst request row is not a dict"))?;
        let Some((request_group_id, ric, start_text, end_text)) = analyst_common_item_fields(
            dict,
            "estimates_request_start_date",
            "estimates_request_end_date",
        )?
        else {
            continue;
        };
        for period in periods.iter() {
            out.push((
                row_index,
                analyst_item_id("analyst_estimates_monthly", &request_group_id, Some(period))?,
                ric.clone(),
                start_text.clone(),
                end_text.clone(),
                Some(period.clone()),
            ));
        }
    }
    Ok(out)
}

#[pyfunction]
pub(crate) fn build_analyst_estimate_item_row_columns(
    py: Python<'_>,
    column_names: Vec<String>,
    column_values: &Bound<'_, PyAny>,
    request_periods: &Bound<'_, PyAny>,
) -> PyResult<Vec<AnalystItemRow>> {
    let mut periods: Vec<String> = Vec::new();
    for period in request_periods.iter()? {
        periods.push(period?.str()?.to_str()?.to_string());
    }
    let label = "analyst estimate item rows";
    let (columns, row_count) =
        collect_pyobject_column_values(py, &column_names, column_values, label)?;
    let column_index = column_index_by_name(&column_names);
    let mut out: Vec<AnalystItemRow> = Vec::new();
    for row_idx in 0..row_count {
        let Some((request_group_id, ric, start_text, end_text)) = analyst_common_item_columns(
            py,
            &columns,
            &column_index,
            row_idx,
            "estimates_request_start_date",
            "estimates_request_end_date",
        )?
        else {
            continue;
        };
        for period in periods.iter() {
            out.push((
                row_idx,
                analyst_item_id(
                    "analyst_estimates_monthly",
                    request_group_id.bind(py),
                    Some(period),
                )?,
                ric.clone(),
                start_text.clone(),
                end_text.clone(),
                Some(period.clone()),
            ));
        }
    }
    Ok(out)
}

pub(crate) fn dict_raw_string(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<String>> {
    let Some(value) = dict.get_item(key)? else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    Ok(Some(value.str()?.to_str()?.to_string()))
}

pub(crate) fn py_raw_string(value: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    if value.is_none() {
        return Ok(None);
    }
    Ok(Some(value.str()?.to_str()?.to_string()))
}

pub(crate) fn dict_required_py_object(
    py: Python<'_>,
    dict: &Bound<'_, PyDict>,
    key: &str,
) -> PyResult<PyObject> {
    let Some(value) = dict.get_item(key)? else {
        return Err(PyValueError::new_err(format!(
            "missing required key: {key}"
        )));
    };
    Ok(value.into_py(py))
}

pub(crate) fn dict_py_object_or_none(
    py: Python<'_>,
    dict: &Bound<'_, PyDict>,
    key: &str,
) -> PyResult<PyObject> {
    match dict.get_item(key)? {
        Some(value) => Ok(value.into_py(py)),
        None => Ok(py.None()),
    }
}

pub(crate) fn copy_py_dict<'py>(
    py: Python<'py>,
    source: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new_bound(py);
    for (key, value) in source.iter() {
        out.set_item(key, value)?;
    }
    Ok(out)
}

pub(crate) fn dict_optional_int(
    py: Python<'_>,
    dict: &Bound<'_, PyDict>,
    key: &str,
) -> PyResult<Option<i64>> {
    let Some(value) = dict.get_item(key)? else {
        return Ok(None);
    };
    py_optional_int(py, &value)
}

pub(crate) fn py_optional_int(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Option<i64>> {
    let Some(int_obj) = normalize_doc_ownership_int_value(py, Some(value))? else {
        return Ok(None);
    };
    int_obj.extract::<i64>(py).map(Some)
}

pub(crate) fn parse_analyst_actual_response_rows_by_instrument(
    rows: &Bound<'_, PyAny>,
) -> PyResult<HashMap<String, Vec<ParsedAnalystActualResponseRow>>> {
    let mut rows_by_instrument: HashMap<String, Vec<ParsedAnalystActualResponseRow>> =
        HashMap::new();
    for row in rows.iter()? {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("analyst actuals response row is not a dict"))?;
        let Some(instrument) = dict_raw_string(dict, "instrument")? else {
            continue;
        };
        let announcement_date_text = dict_iso_date_string(dict, "TR.EPSActValue.date")?;
        let announcement_date_ordinal = match announcement_date_text.as_deref() {
            Some(text) => Some(date_text_ordinal(text)?),
            None => None,
        };
        rows_by_instrument
            .entry(instrument)
            .or_default()
            .push(ParsedAnalystActualResponseRow {
                announcement_date_text,
                announcement_date_ordinal,
                fiscal_period_end_text: dict_iso_date_string(dict, "TR.EPSActValue.periodenddate")?,
                actual_eps: dict_optional_float(dict, "TR.EPSActValue")?,
                raw_fperiod: dict_normalized_string(dict, "TR.EPSActValue.fperiod")?,
            });
    }
    Ok(rows_by_instrument)
}

pub(crate) fn parse_analyst_actual_response_columns_by_instrument(
    py: Python<'_>,
    column_names: Vec<String>,
    column_values: &Bound<'_, PyAny>,
) -> PyResult<HashMap<String, Vec<ParsedAnalystActualResponseRow>>> {
    let label = "analyst actuals response rows";
    let (columns, row_count) =
        collect_pyobject_column_values(py, &column_names, column_values, label)?;
    let column_index = column_index_by_name(&column_names);
    let instrument_idx = required_named_column_index(&column_index, label, "instrument")?;

    let mut rows_by_instrument: HashMap<String, Vec<ParsedAnalystActualResponseRow>> =
        HashMap::new();
    for row_idx in 0..row_count {
        let Some(instrument) = py_raw_string(columns[instrument_idx][row_idx].bind(py))? else {
            continue;
        };
        let announcement_date_text =
            optional_column_value(&columns, &column_index, row_idx, "TR.EPSActValue.date")
                .map(|value| py_any_date_iso_string(value.bind(py)))
                .transpose()?
                .flatten();
        let announcement_date_ordinal = match announcement_date_text.as_deref() {
            Some(text) => Some(date_text_ordinal(text)?),
            None => None,
        };
        rows_by_instrument
            .entry(instrument)
            .or_default()
            .push(ParsedAnalystActualResponseRow {
                announcement_date_text,
                announcement_date_ordinal,
                fiscal_period_end_text: optional_column_value(
                    &columns,
                    &column_index,
                    row_idx,
                    "TR.EPSActValue.periodenddate",
                )
                .map(|value| py_any_date_iso_string(value.bind(py)))
                .transpose()?
                .flatten(),
                actual_eps: optional_column_value(
                    &columns,
                    &column_index,
                    row_idx,
                    "TR.EPSActValue",
                )
                .map(|value| normalize_doc_ownership_float_value(Some(value.bind(py))))
                .transpose()?
                .flatten(),
                raw_fperiod: optional_column_value(
                    &columns,
                    &column_index,
                    row_idx,
                    "TR.EPSActValue.fperiod",
                )
                .map(|value| normalize_lookup_text_any_impl(Some(value.bind(py))))
                .transpose()?
                .flatten(),
            });
    }
    Ok(rows_by_instrument)
}

#[pyfunction]
pub(crate) fn normalize_analyst_actuals_batch_response_rows_value(
    item_rows: &Bound<'_, PyAny>,
    response_rows: &Bound<'_, PyAny>,
) -> PyResult<Vec<AnalystActualBatchResponseRow>> {
    let response_by_instrument = parse_analyst_actual_response_rows_by_instrument(response_rows)?;
    let mut out: Vec<AnalystActualBatchResponseRow> = Vec::new();
    for item_row in item_rows.iter()? {
        let item_row = item_row?;
        let dict = item_row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("analyst actuals item row is not a dict"))?;
        let Some(item_index_value) = dict.get_item("item_index")? else {
            return Err(PyValueError::new_err("missing item_index"));
        };
        let item_index = item_index_value.extract::<usize>()?;
        let Some(instrument) = dict_raw_string(dict, "instrument")? else {
            continue;
        };
        let Some(start_text) = dict_normalized_string(dict, "start_text")? else {
            return Err(PyValueError::new_err("missing start_text"));
        };
        let Some(end_text) = dict_normalized_string(dict, "end_text")? else {
            return Err(PyValueError::new_err("missing end_text"));
        };
        let start_ordinal = date_text_ordinal(&start_text)?;
        let end_ordinal = date_text_ordinal(&end_text)?;
        let Some(matched_rows) = response_by_instrument.get(&instrument) else {
            continue;
        };
        let mut response_row_index = 0_i64;
        for matched_row in matched_rows {
            let Some(announcement_ordinal) = matched_row.announcement_date_ordinal else {
                continue;
            };
            if announcement_ordinal < start_ordinal || announcement_ordinal > end_ordinal {
                continue;
            }
            let Some(announcement_date_text) = matched_row.announcement_date_text.as_ref() else {
                continue;
            };
            let row_parse_status = if matched_row.actual_eps.is_none() {
                "MISSING_ACTUAL_EPS"
            } else if matched_row.fiscal_period_end_text.is_none() {
                "MISSING_FISCAL_PERIOD_END"
            } else {
                "OK"
            };
            out.push((
                item_index,
                response_row_index,
                announcement_date_text.clone(),
                matched_row.fiscal_period_end_text.clone(),
                matched_row.actual_eps,
                matched_row.raw_fperiod.clone(),
                row_parse_status.to_string(),
            ));
            response_row_index += 1;
        }
    }
    Ok(out)
}

#[pyfunction]
pub(crate) fn normalize_analyst_actuals_batch_response_columns(
    py: Python<'_>,
    instruments: &Bound<'_, PyAny>,
    start_texts: &Bound<'_, PyAny>,
    end_texts: &Bound<'_, PyAny>,
    response_column_names: Vec<String>,
    response_column_values: &Bound<'_, PyAny>,
) -> PyResult<Vec<AnalystActualBatchResponseRow>> {
    let instruments = pyobject_sequence(py, instruments)?;
    let start_texts = pyobject_sequence(py, start_texts)?;
    let end_texts = pyobject_sequence(py, end_texts)?;
    let item_count = instruments.len();
    if start_texts.len() != item_count || end_texts.len() != item_count {
        return Err(PyValueError::new_err(
            "all analyst actuals item columns must have the same length",
        ));
    }

    let response_by_instrument = parse_analyst_actual_response_columns_by_instrument(
        py,
        response_column_names,
        response_column_values,
    )?;
    let mut out: Vec<AnalystActualBatchResponseRow> = Vec::new();
    for item_index in 0..item_count {
        let Some(instrument) = py_raw_string(instruments[item_index].bind(py))? else {
            continue;
        };
        let Some(start_text) =
            normalize_lookup_text_any_impl(Some(start_texts[item_index].bind(py)))?
        else {
            return Err(PyValueError::new_err("missing start_text"));
        };
        let Some(end_text) = normalize_lookup_text_any_impl(Some(end_texts[item_index].bind(py)))?
        else {
            return Err(PyValueError::new_err("missing end_text"));
        };
        let start_ordinal = date_text_ordinal(&start_text)?;
        let end_ordinal = date_text_ordinal(&end_text)?;
        let Some(matched_rows) = response_by_instrument.get(&instrument) else {
            continue;
        };
        let mut response_row_index = 0_i64;
        for matched_row in matched_rows {
            let Some(announcement_ordinal) = matched_row.announcement_date_ordinal else {
                continue;
            };
            if announcement_ordinal < start_ordinal || announcement_ordinal > end_ordinal {
                continue;
            }
            let Some(announcement_date_text) = matched_row.announcement_date_text.as_ref() else {
                continue;
            };
            let row_parse_status = if matched_row.actual_eps.is_none() {
                "MISSING_ACTUAL_EPS"
            } else if matched_row.fiscal_period_end_text.is_none() {
                "MISSING_FISCAL_PERIOD_END"
            } else {
                "OK"
            };
            out.push((
                item_index,
                response_row_index,
                announcement_date_text.clone(),
                matched_row.fiscal_period_end_text.clone(),
                matched_row.actual_eps,
                matched_row.raw_fperiod.clone(),
                row_parse_status.to_string(),
            ));
            response_row_index += 1;
        }
    }
    Ok(out)
}

pub(crate) fn parse_analyst_estimate_response_rows_by_instrument(
    py: Python<'_>,
    rows: &Bound<'_, PyAny>,
) -> PyResult<HashMap<String, Vec<ParsedAnalystEstimateResponseRow>>> {
    let mut rows_by_instrument: HashMap<String, Vec<ParsedAnalystEstimateResponseRow>> =
        HashMap::new();
    for row in rows.iter()? {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("analyst estimates response row is not a dict"))?;
        let Some(instrument) = dict_raw_string(dict, "instrument")? else {
            continue;
        };
        let calc_date_text = dict_iso_date_string(dict, "TR.EPSMean.calcdate")?;
        let calc_date_ordinal = match calc_date_text.as_deref() {
            Some(text) => Some(date_text_ordinal(text)?),
            None => None,
        };
        rows_by_instrument
            .entry(instrument)
            .or_default()
            .push(ParsedAnalystEstimateResponseRow {
                calc_date_text,
                calc_date_ordinal,
                fiscal_period_end_text: dict_iso_date_string(dict, "TR.EPSMean.periodenddate")?,
                raw_fperiod: dict_normalized_string(dict, "TR.EPSMean.fperiod")?,
                forecast_consensus_mean: dict_optional_float(dict, "TR.EPSMean")?,
                forecast_dispersion: dict_optional_float(dict, "TR.EPSStdDev")?,
                estimate_count: dict_optional_int(py, dict, "TR.EPSNumberofEstimates")?,
            });
    }
    Ok(rows_by_instrument)
}

pub(crate) fn parse_analyst_estimate_response_columns_by_instrument(
    py: Python<'_>,
    column_names: Vec<String>,
    column_values: &Bound<'_, PyAny>,
) -> PyResult<HashMap<String, Vec<ParsedAnalystEstimateResponseRow>>> {
    let label = "analyst estimates response rows";
    let (columns, row_count) =
        collect_pyobject_column_values(py, &column_names, column_values, label)?;
    let column_index = column_index_by_name(&column_names);
    let instrument_idx = required_named_column_index(&column_index, label, "instrument")?;

    let mut rows_by_instrument: HashMap<String, Vec<ParsedAnalystEstimateResponseRow>> =
        HashMap::new();
    for row_idx in 0..row_count {
        let Some(instrument) = py_raw_string(columns[instrument_idx][row_idx].bind(py))? else {
            continue;
        };
        let calc_date_text =
            optional_column_value(&columns, &column_index, row_idx, "TR.EPSMean.calcdate")
                .map(|value| py_any_date_iso_string(value.bind(py)))
                .transpose()?
                .flatten();
        let calc_date_ordinal = match calc_date_text.as_deref() {
            Some(text) => Some(date_text_ordinal(text)?),
            None => None,
        };
        rows_by_instrument
            .entry(instrument)
            .or_default()
            .push(ParsedAnalystEstimateResponseRow {
                calc_date_text,
                calc_date_ordinal,
                fiscal_period_end_text: optional_column_value(
                    &columns,
                    &column_index,
                    row_idx,
                    "TR.EPSMean.periodenddate",
                )
                .map(|value| py_any_date_iso_string(value.bind(py)))
                .transpose()?
                .flatten(),
                raw_fperiod: optional_column_value(
                    &columns,
                    &column_index,
                    row_idx,
                    "TR.EPSMean.fperiod",
                )
                .map(|value| normalize_lookup_text_any_impl(Some(value.bind(py))))
                .transpose()?
                .flatten(),
                forecast_consensus_mean: optional_column_value(
                    &columns,
                    &column_index,
                    row_idx,
                    "TR.EPSMean",
                )
                .map(|value| normalize_doc_ownership_float_value(Some(value.bind(py))))
                .transpose()?
                .flatten(),
                forecast_dispersion: optional_column_value(
                    &columns,
                    &column_index,
                    row_idx,
                    "TR.EPSStdDev",
                )
                .map(|value| normalize_doc_ownership_float_value(Some(value.bind(py))))
                .transpose()?
                .flatten(),
                estimate_count: optional_column_value(
                    &columns,
                    &column_index,
                    row_idx,
                    "TR.EPSNumberofEstimates",
                )
                .map(|value| py_optional_int(py, value.bind(py)))
                .transpose()?
                .flatten(),
            });
    }
    Ok(rows_by_instrument)
}

#[pyfunction]
pub(crate) fn normalize_analyst_estimates_batch_response_rows_value(
    py: Python<'_>,
    item_rows: &Bound<'_, PyAny>,
    response_rows: &Bound<'_, PyAny>,
) -> PyResult<Vec<AnalystEstimateBatchResponseRow>> {
    let response_by_instrument =
        parse_analyst_estimate_response_rows_by_instrument(py, response_rows)?;
    let mut out: Vec<AnalystEstimateBatchResponseRow> = Vec::new();
    for item_row in item_rows.iter()? {
        let item_row = item_row?;
        let dict = item_row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("analyst estimates item row is not a dict"))?;
        let Some(item_index_value) = dict.get_item("item_index")? else {
            return Err(PyValueError::new_err("missing item_index"));
        };
        let item_index = item_index_value.extract::<usize>()?;
        let Some(instrument) = dict_raw_string(dict, "instrument")? else {
            continue;
        };
        let Some(start_text) = dict_normalized_string(dict, "start_text")? else {
            return Err(PyValueError::new_err("missing start_text"));
        };
        let Some(end_text) = dict_normalized_string(dict, "end_text")? else {
            return Err(PyValueError::new_err("missing end_text"));
        };
        let start_ordinal = date_text_ordinal(&start_text)?;
        let end_ordinal = date_text_ordinal(&end_text)?;
        let Some(matched_rows) = response_by_instrument.get(&instrument) else {
            continue;
        };
        let mut response_row_index = 0_i64;
        for matched_row in matched_rows {
            let Some(calc_ordinal) = matched_row.calc_date_ordinal else {
                continue;
            };
            if calc_ordinal < start_ordinal || calc_ordinal > end_ordinal {
                continue;
            }
            let Some(calc_date_text) = matched_row.calc_date_text.as_ref() else {
                continue;
            };
            let row_parse_status = if matched_row.fiscal_period_end_text.is_none() {
                "MISSING_FISCAL_PERIOD_END"
            } else if matched_row.forecast_consensus_mean.is_none() {
                "MISSING_CONSENSUS_MEAN"
            } else {
                "OK"
            };
            out.push((
                item_index,
                response_row_index,
                calc_date_text.clone(),
                matched_row.fiscal_period_end_text.clone(),
                matched_row.raw_fperiod.clone(),
                matched_row.forecast_consensus_mean,
                matched_row.forecast_dispersion,
                matched_row.estimate_count,
                row_parse_status.to_string(),
            ));
            response_row_index += 1;
        }
    }
    Ok(out)
}

#[pyfunction]
pub(crate) fn normalize_analyst_estimates_batch_response_columns(
    py: Python<'_>,
    instruments: &Bound<'_, PyAny>,
    start_texts: &Bound<'_, PyAny>,
    end_texts: &Bound<'_, PyAny>,
    response_column_names: Vec<String>,
    response_column_values: &Bound<'_, PyAny>,
) -> PyResult<Vec<AnalystEstimateBatchResponseRow>> {
    let instruments = pyobject_sequence(py, instruments)?;
    let start_texts = pyobject_sequence(py, start_texts)?;
    let end_texts = pyobject_sequence(py, end_texts)?;
    let item_count = instruments.len();
    if start_texts.len() != item_count || end_texts.len() != item_count {
        return Err(PyValueError::new_err(
            "all analyst estimates item columns must have the same length",
        ));
    }

    let response_by_instrument = parse_analyst_estimate_response_columns_by_instrument(
        py,
        response_column_names,
        response_column_values,
    )?;
    let mut out: Vec<AnalystEstimateBatchResponseRow> = Vec::new();
    for item_index in 0..item_count {
        let Some(instrument) = py_raw_string(instruments[item_index].bind(py))? else {
            continue;
        };
        let Some(start_text) =
            normalize_lookup_text_any_impl(Some(start_texts[item_index].bind(py)))?
        else {
            return Err(PyValueError::new_err("missing start_text"));
        };
        let Some(end_text) = normalize_lookup_text_any_impl(Some(end_texts[item_index].bind(py)))?
        else {
            return Err(PyValueError::new_err("missing end_text"));
        };
        let start_ordinal = date_text_ordinal(&start_text)?;
        let end_ordinal = date_text_ordinal(&end_text)?;
        let Some(matched_rows) = response_by_instrument.get(&instrument) else {
            continue;
        };
        let mut response_row_index = 0_i64;
        for matched_row in matched_rows {
            let Some(calc_ordinal) = matched_row.calc_date_ordinal else {
                continue;
            };
            if calc_ordinal < start_ordinal || calc_ordinal > end_ordinal {
                continue;
            }
            let Some(calc_date_text) = matched_row.calc_date_text.as_ref() else {
                continue;
            };
            let row_parse_status = if matched_row.fiscal_period_end_text.is_none() {
                "MISSING_FISCAL_PERIOD_END"
            } else if matched_row.forecast_consensus_mean.is_none() {
                "MISSING_CONSENSUS_MEAN"
            } else {
                "OK"
            };
            out.push((
                item_index,
                response_row_index,
                calc_date_text.clone(),
                matched_row.fiscal_period_end_text.clone(),
                matched_row.raw_fperiod.clone(),
                matched_row.forecast_consensus_mean,
                matched_row.forecast_dispersion,
                matched_row.estimate_count,
                row_parse_status.to_string(),
            ));
            response_row_index += 1;
        }
    }
    Ok(out)
}

pub(crate) fn lseg_item_id_one_part(stage: &str, key: &str) -> PyResult<String> {
    let mut payload = String::new();
    payload.push('[');
    push_json_ascii_string(&mut payload, stage)?;
    payload.push(',');
    push_json_ascii_string(&mut payload, key)?;
    payload.push(']');
    Ok(stable_hash_id_from_payload("item", &payload))
}

pub(crate) fn py_any_date_iso_string(value: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    if value.is_none() {
        return Ok(None);
    }
    if let Ok((year, month, day)) = extract_py_date_parts(value) {
        return Ok(Some(format!("{year:04}-{month:02}-{day:02}")));
    }
    let Some(text) = py_str_normalized(value)? else {
        return Ok(None);
    };
    let Some((year, month, day)) = parse_doc_ownership_date_text(&text) else {
        return Ok(None);
    };
    Ok(Some(format!("{year:04}-{month:02}-{day:02}")))
}

pub(crate) fn dict_iso_date_string(
    dict: &Bound<'_, PyDict>,
    key: &str,
) -> PyResult<Option<String>> {
    let Some(value) = dict.get_item(key)? else {
        return Ok(None);
    };
    py_any_date_iso_string(&value)
}

pub(crate) fn dict_truthy_bool(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<bool> {
    let Some(value) = dict.get_item(key)? else {
        return Ok(false);
    };
    if value.is_none() {
        return Ok(false);
    }
    value.is_truthy()
}

pub(crate) fn column_truthy_bool(
    py: Python<'_>,
    columns: &[Vec<PyObject>],
    column_index: &HashMap<String, usize>,
    row_idx: usize,
    column_name: &str,
) -> PyResult<bool> {
    let Some(value) = optional_column_value(columns, column_index, row_idx, column_name) else {
        return Ok(false);
    };
    let value = value.bind(py);
    if value.is_none() {
        return Ok(false);
    }
    value.is_truthy()
}

pub(crate) fn column_normalized_string(
    py: Python<'_>,
    columns: &[Vec<PyObject>],
    column_index: &HashMap<String, usize>,
    row_idx: usize,
    column_name: &str,
) -> PyResult<Option<String>> {
    optional_column_value(columns, column_index, row_idx, column_name)
        .map(|value| normalize_lookup_text_any_impl(Some(value.bind(py))))
        .transpose()
        .map(|value| value.flatten())
}

pub(crate) fn column_optional_float(
    py: Python<'_>,
    columns: &[Vec<PyObject>],
    column_index: &HashMap<String, usize>,
    row_idx: usize,
    column_name: &str,
) -> PyResult<Option<f64>> {
    optional_column_value(columns, column_index, row_idx, column_name)
        .map(|value| normalize_doc_ownership_float_value(Some(value.bind(py))))
        .transpose()
        .map(|value| value.flatten())
}

pub(crate) fn column_date_key(
    py: Python<'_>,
    columns: &[Vec<PyObject>],
    column_index: &HashMap<String, usize>,
    row_idx: usize,
    column_name: &str,
) -> PyResult<Option<String>> {
    optional_column_value(columns, column_index, row_idx, column_name)
        .map(|value| py_any_date_iso_string(value.bind(py)))
        .transpose()
        .map(|value| value.flatten())
}

pub(crate) fn column_kypermno_value(
    py: Python<'_>,
    columns: &[Vec<PyObject>],
    column_index: &HashMap<String, usize>,
    row_idx: usize,
    column_name: &str,
) -> PyResult<Option<String>> {
    optional_column_value(columns, column_index, row_idx, column_name)
        .map(|value| normalize_kypermno_value(Some(value.bind(py))))
        .transpose()
        .map(|value| value.flatten())
}

pub(crate) fn column_doc_ownership_date_parts(
    py: Python<'_>,
    columns: &[Vec<PyObject>],
    column_index: &HashMap<String, usize>,
    row_idx: usize,
    column_name: &str,
) -> PyResult<Option<DocOwnershipDateParts>> {
    let Some(value) = optional_column_value(columns, column_index, row_idx, column_name) else {
        return Ok(None);
    };
    let Some(date_text) = py_any_date_iso_string(value.bind(py))? else {
        return Ok(None);
    };
    Ok(parse_doc_ownership_date_text(&date_text))
}

#[pyfunction]
pub(crate) fn build_ownership_universe_item_rows_value(
    rows: &Bound<'_, PyAny>,
) -> PyResult<Vec<OwnershipItemRow>> {
    let mut out: Vec<OwnershipItemRow> = Vec::new();
    for (row_index, row) in rows.iter()?.enumerate() {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("ownership universe handoff row is not a dict"))?;
        if !dict_truthy_bool(dict, "retrieval_eligible")? {
            continue;
        }
        let Some(ownership_lookup_row_id) =
            dict_normalized_string(dict, "ownership_lookup_row_id")?
        else {
            continue;
        };
        let Some(candidate_ric) = dict_normalized_string(dict, "candidate_ric")? else {
            continue;
        };
        let Some(request_start_date) = dict_normalized_string(dict, "request_start_date")? else {
            continue;
        };
        let Some(request_end_date) = dict_normalized_string(dict, "request_end_date")? else {
            continue;
        };
        let (start_text, end_text) =
            normalize_lseg_month_boundaries(&request_start_date, &request_end_date)?;
        out.push((
            row_index,
            lseg_item_id_one_part("ownership_universe", &ownership_lookup_row_id)?,
            candidate_ric,
            start_text,
            end_text,
        ));
    }
    Ok(out)
}

#[pyfunction]
pub(crate) fn build_ownership_universe_item_row_columns(
    py: Python<'_>,
    column_names: Vec<String>,
    column_values: &Bound<'_, PyAny>,
) -> PyResult<Vec<OwnershipItemRow>> {
    let label = "ownership universe item rows";
    let (columns, row_count) =
        collect_pyobject_column_values(py, &column_names, column_values, label)?;
    let column_index = column_index_by_name(&column_names);
    let mut out: Vec<OwnershipItemRow> = Vec::new();
    for row_idx in 0..row_count {
        let retrieval_eligible =
            optional_column_value(&columns, &column_index, row_idx, "retrieval_eligible")
                .map(|value| {
                    let value = value.bind(py);
                    if value.is_none() {
                        Ok(false)
                    } else {
                        value.is_truthy()
                    }
                })
                .transpose()?
                .unwrap_or(false);
        if !retrieval_eligible {
            continue;
        }
        let Some(ownership_lookup_row_id) =
            optional_column_value(&columns, &column_index, row_idx, "ownership_lookup_row_id")
                .map(|value| normalize_lookup_text_any_impl(Some(value.bind(py))))
                .transpose()?
                .flatten()
        else {
            continue;
        };
        let Some(candidate_ric) =
            optional_column_value(&columns, &column_index, row_idx, "candidate_ric")
                .map(|value| normalize_lookup_text_any_impl(Some(value.bind(py))))
                .transpose()?
                .flatten()
        else {
            continue;
        };
        let Some(request_start_date) =
            optional_column_value(&columns, &column_index, row_idx, "request_start_date")
                .map(|value| normalize_lookup_text_any_impl(Some(value.bind(py))))
                .transpose()?
                .flatten()
        else {
            continue;
        };
        let Some(request_end_date) =
            optional_column_value(&columns, &column_index, row_idx, "request_end_date")
                .map(|value| normalize_lookup_text_any_impl(Some(value.bind(py))))
                .transpose()?
                .flatten()
        else {
            continue;
        };
        let (start_text, end_text) =
            normalize_lseg_month_boundaries(&request_start_date, &request_end_date)?;
        out.push((
            row_idx,
            lseg_item_id_one_part("ownership_universe", &ownership_lookup_row_id)?,
            candidate_ric,
            start_text,
            end_text,
        ));
    }
    Ok(out)
}

#[pyfunction]
pub(crate) fn build_doc_ownership_exact_item_rows_value(
    rows: &Bound<'_, PyAny>,
) -> PyResult<Vec<OwnershipItemRow>> {
    let mut out: Vec<OwnershipItemRow> = Vec::new();
    for (row_index, row) in rows.iter()?.enumerate() {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("doc ownership request row is not a dict"))?;
        if !dict_truthy_bool(dict, "retrieval_eligible")? {
            continue;
        }
        let Some(doc_id) = dict_normalized_string(dict, "doc_id")? else {
            continue;
        };
        let Some(authoritative_ric) = dict_normalized_string(dict, "authoritative_ric")? else {
            continue;
        };
        let Some(date_text) = dict_iso_date_string(dict, "target_effective_date")? else {
            continue;
        };
        out.push((
            row_index,
            lseg_item_id_one_part("doc_ownership_exact", &doc_id)?,
            authoritative_ric,
            date_text.clone(),
            date_text,
        ));
    }
    Ok(out)
}

pub(crate) fn build_doc_ownership_item_row_columns_impl(
    py: Python<'_>,
    column_names: &[String],
    column_values: &Bound<'_, PyAny>,
    label: &str,
    stage: &str,
    start_column: &str,
    end_column: Option<&str>,
) -> PyResult<Vec<OwnershipItemRow>> {
    let (columns, row_count) =
        collect_pyobject_column_values(py, column_names, column_values, label)?;
    let column_index = column_index_by_name(column_names);
    let mut out: Vec<OwnershipItemRow> = Vec::new();
    for row_idx in 0..row_count {
        let retrieval_eligible =
            optional_column_value(&columns, &column_index, row_idx, "retrieval_eligible")
                .map(|value| {
                    let value = value.bind(py);
                    if value.is_none() {
                        Ok(false)
                    } else {
                        value.is_truthy()
                    }
                })
                .transpose()?
                .unwrap_or(false);
        if !retrieval_eligible {
            continue;
        }
        let Some(doc_id) = optional_column_value(&columns, &column_index, row_idx, "doc_id")
            .map(|value| normalize_lookup_text_any_impl(Some(value.bind(py))))
            .transpose()?
            .flatten()
        else {
            continue;
        };
        let Some(authoritative_ric) =
            optional_column_value(&columns, &column_index, row_idx, "authoritative_ric")
                .map(|value| normalize_lookup_text_any_impl(Some(value.bind(py))))
                .transpose()?
                .flatten()
        else {
            continue;
        };
        let Some(start_text) =
            optional_column_value(&columns, &column_index, row_idx, start_column)
                .map(|value| py_any_date_iso_string(value.bind(py)))
                .transpose()?
                .flatten()
        else {
            continue;
        };
        let end_text = match end_column {
            Some(column_name) => {
                let Some(value) =
                    optional_column_value(&columns, &column_index, row_idx, column_name)
                        .map(|value| py_any_date_iso_string(value.bind(py)))
                        .transpose()?
                        .flatten()
                else {
                    continue;
                };
                value
            }
            None => start_text.clone(),
        };
        out.push((
            row_idx,
            lseg_item_id_one_part(stage, &doc_id)?,
            authoritative_ric,
            start_text,
            end_text,
        ));
    }
    Ok(out)
}

#[pyfunction]
pub(crate) fn build_doc_ownership_exact_item_row_columns(
    py: Python<'_>,
    column_names: Vec<String>,
    column_values: &Bound<'_, PyAny>,
) -> PyResult<Vec<OwnershipItemRow>> {
    build_doc_ownership_item_row_columns_impl(
        py,
        &column_names,
        column_values,
        "doc ownership exact item rows",
        "doc_ownership_exact",
        "target_effective_date",
        None,
    )
}

#[pyfunction]
pub(crate) fn build_doc_ownership_fallback_item_rows_value(
    rows: &Bound<'_, PyAny>,
) -> PyResult<Vec<OwnershipItemRow>> {
    let mut out: Vec<OwnershipItemRow> = Vec::new();
    for (row_index, row) in rows.iter()?.enumerate() {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("doc ownership request row is not a dict"))?;
        if !dict_truthy_bool(dict, "retrieval_eligible")? {
            continue;
        }
        let Some(doc_id) = dict_normalized_string(dict, "doc_id")? else {
            continue;
        };
        let Some(authoritative_ric) = dict_normalized_string(dict, "authoritative_ric")? else {
            continue;
        };
        let Some(start_text) = dict_iso_date_string(dict, "fallback_window_start")? else {
            continue;
        };
        let Some(end_text) = dict_iso_date_string(dict, "fallback_window_end")? else {
            continue;
        };
        out.push((
            row_index,
            lseg_item_id_one_part("doc_ownership_fallback", &doc_id)?,
            authoritative_ric,
            start_text,
            end_text,
        ));
    }
    Ok(out)
}

#[pyfunction]
pub(crate) fn build_doc_ownership_fallback_item_row_columns(
    py: Python<'_>,
    column_names: Vec<String>,
    column_values: &Bound<'_, PyAny>,
) -> PyResult<Vec<OwnershipItemRow>> {
    build_doc_ownership_item_row_columns_impl(
        py,
        &column_names,
        column_values,
        "doc ownership fallback item rows",
        "doc_ownership_fallback",
        "fallback_window_start",
        Some("fallback_window_end"),
    )
}

pub(crate) fn date_text_ordinal(text: &str) -> PyResult<i64> {
    let Some((year, month, day)) = parse_doc_ownership_date_text(text) else {
        return Err(PyValueError::new_err(format!(
            "Invalid isoformat string: {text:?}"
        )));
    };
    date_ordinal(year, month, day)
}

pub(crate) fn dict_optional_float(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<f64>> {
    let Some(value) = dict.get_item(key)? else {
        return Ok(None);
    };
    normalize_doc_ownership_float_value(Some(&value))
}

pub(crate) fn dict_required_float(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<f64> {
    let Some(value) = dict.get_item(key)? else {
        return Err(PyValueError::new_err(format!(
            "missing required key: {key}"
        )));
    };
    let Some(number) = normalize_doc_ownership_float_value(Some(&value))? else {
        return Err(PyValueError::new_err(format!("null required key: {key}")));
    };
    Ok(number)
}

pub(crate) fn dict_doc_category(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<String>> {
    let Some(value) = dict.get_item(key)? else {
        return Ok(None);
    };
    normalize_doc_ownership_category_value(Some(&value))
}

pub(crate) fn dict_date_key(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<String>> {
    let Some(value) = dict.get_item(key)? else {
        return Ok(None);
    };
    py_any_date_iso_string(&value)
}

pub(crate) fn dict_python_or_empty_string(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<String> {
    let Some(value) = dict.get_item(key)? else {
        return Ok(String::new());
    };
    if value.is_none() || !value.is_truthy()? {
        return Ok(String::new());
    }
    Ok(value.str()?.to_str()?.to_string())
}

pub(crate) fn dict_optional_i64_or_zero(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<i64> {
    let Some(value) = dict.get_item(key)? else {
        return Ok(0);
    };
    Ok(stage_audit_optional_int_impl(Some(&value))?.unwrap_or(0))
}

pub(crate) fn parse_ownership_response_rows_by_instrument(
    rows: &Bound<'_, PyAny>,
) -> PyResult<HashMap<String, Vec<ParsedOwnershipResponseRow>>> {
    let mut rows_by_instrument: HashMap<String, Vec<ParsedOwnershipResponseRow>> = HashMap::new();
    for row in rows.iter()? {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("ownership response row is not a dict"))?;
        let Some(instrument) = dict_normalized_string(dict, "instrument")? else {
            continue;
        };
        let date_text = dict_iso_date_string(dict, "TR.CategoryOwnershipPct.Date")?;
        let date_ordinal = match date_text.as_deref() {
            Some(text) => Some(date_text_ordinal(text)?),
            None => None,
        };
        let value = dict_optional_float(dict, "TR.CategoryOwnershipPct")?;
        let lookup_category = dict_normalized_string(dict, "TR.InstrStatTypeValue")?;
        let doc_category = dict_doc_category(dict, "TR.InstrStatTypeValue")?;
        rows_by_instrument
            .entry(instrument)
            .or_default()
            .push(ParsedOwnershipResponseRow {
                date_text,
                date_ordinal,
                value,
                lookup_category,
                doc_category,
            });
    }
    Ok(rows_by_instrument)
}

pub(crate) fn parse_ownership_response_columns_by_instrument(
    py: Python<'_>,
    column_names: Vec<String>,
    column_values: &Bound<'_, PyAny>,
) -> PyResult<HashMap<String, Vec<ParsedOwnershipResponseRow>>> {
    let label = "ownership response rows";
    let (columns, row_count) =
        collect_pyobject_column_values(py, &column_names, column_values, label)?;
    let column_index = column_index_by_name(&column_names);
    let instrument_idx = required_named_column_index(&column_index, label, "instrument")?;

    let mut rows_by_instrument: HashMap<String, Vec<ParsedOwnershipResponseRow>> = HashMap::new();
    for row_idx in 0..row_count {
        let Some(instrument) =
            normalize_lookup_text_any_impl(Some(columns[instrument_idx][row_idx].bind(py)))?
        else {
            continue;
        };
        let date_text = optional_column_value(
            &columns,
            &column_index,
            row_idx,
            "TR.CategoryOwnershipPct.Date",
        )
        .map(|value| py_any_date_iso_string(value.bind(py)))
        .transpose()?
        .flatten();
        let date_ordinal = match date_text.as_deref() {
            Some(text) => Some(date_text_ordinal(text)?),
            None => None,
        };
        rows_by_instrument
            .entry(instrument)
            .or_default()
            .push(ParsedOwnershipResponseRow {
                date_text,
                date_ordinal,
                value: optional_column_value(
                    &columns,
                    &column_index,
                    row_idx,
                    "TR.CategoryOwnershipPct",
                )
                .map(|value| normalize_doc_ownership_float_value(Some(value.bind(py))))
                .transpose()?
                .flatten(),
                lookup_category: optional_column_value(
                    &columns,
                    &column_index,
                    row_idx,
                    "TR.InstrStatTypeValue",
                )
                .map(|value| normalize_lookup_text_any_impl(Some(value.bind(py))))
                .transpose()?
                .flatten(),
                doc_category: optional_column_value(
                    &columns,
                    &column_index,
                    row_idx,
                    "TR.InstrStatTypeValue",
                )
                .map(|value| normalize_doc_ownership_category_value(Some(value.bind(py))))
                .transpose()?
                .flatten(),
            });
    }
    Ok(rows_by_instrument)
}

#[pyfunction]
pub(crate) fn normalize_ownership_universe_batch_response_rows_value(
    item_rows: &Bound<'_, PyAny>,
    response_rows: &Bound<'_, PyAny>,
) -> PyResult<Vec<OwnershipUniverseBatchResponseRow>> {
    let response_by_instrument = parse_ownership_response_rows_by_instrument(response_rows)?;
    let mut out: Vec<OwnershipUniverseBatchResponseRow> = Vec::new();
    for item_row in item_rows.iter()? {
        let item_row = item_row?;
        let dict = item_row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("ownership universe item row is not a dict"))?;
        let Some(item_index_value) = dict.get_item("item_index")? else {
            return Err(PyValueError::new_err("missing item_index"));
        };
        let item_index = item_index_value.extract::<usize>()?;
        let Some(instrument) = dict_normalized_string(dict, "instrument")? else {
            continue;
        };
        let Some(start_text) = dict_normalized_string(dict, "start_text")? else {
            return Err(PyValueError::new_err("missing start_text"));
        };
        let Some(end_text) = dict_normalized_string(dict, "end_text")? else {
            return Err(PyValueError::new_err("missing end_text"));
        };
        let start_ordinal = date_text_ordinal(&start_text)?;
        let end_ordinal = date_text_ordinal(&end_text)?;
        let Some(matched_rows) = response_by_instrument.get(&instrument) else {
            continue;
        };
        for matched_row in matched_rows {
            let Some(returned_date_ordinal) = matched_row.date_ordinal else {
                continue;
            };
            if returned_date_ordinal < start_ordinal || returned_date_ordinal > end_ordinal {
                continue;
            }
            if matched_row.value.is_none() && matched_row.lookup_category.is_none() {
                continue;
            }
            let Some(date_text) = matched_row.date_text.as_ref() else {
                continue;
            };
            out.push((
                item_index,
                instrument.clone(),
                date_text.clone(),
                matched_row.lookup_category.clone(),
                matched_row.value,
            ));
        }
    }
    Ok(out)
}

#[pyfunction]
pub(crate) fn normalize_ownership_universe_batch_response_columns(
    py: Python<'_>,
    instruments: &Bound<'_, PyAny>,
    start_texts: &Bound<'_, PyAny>,
    end_texts: &Bound<'_, PyAny>,
    response_column_names: Vec<String>,
    response_column_values: &Bound<'_, PyAny>,
) -> PyResult<Vec<OwnershipUniverseBatchResponseRow>> {
    let instruments = pyobject_sequence(py, instruments)?;
    let start_texts = pyobject_sequence(py, start_texts)?;
    let end_texts = pyobject_sequence(py, end_texts)?;
    let item_count = instruments.len();
    if start_texts.len() != item_count || end_texts.len() != item_count {
        return Err(PyValueError::new_err(
            "all ownership universe item columns must have the same length",
        ));
    }

    let response_by_instrument = parse_ownership_response_columns_by_instrument(
        py,
        response_column_names,
        response_column_values,
    )?;
    let mut out: Vec<OwnershipUniverseBatchResponseRow> = Vec::new();
    for item_index in 0..item_count {
        let Some(instrument) =
            normalize_lookup_text_any_impl(Some(instruments[item_index].bind(py)))?
        else {
            continue;
        };
        let Some(start_text) =
            normalize_lookup_text_any_impl(Some(start_texts[item_index].bind(py)))?
        else {
            return Err(PyValueError::new_err("missing start_text"));
        };
        let Some(end_text) = normalize_lookup_text_any_impl(Some(end_texts[item_index].bind(py)))?
        else {
            return Err(PyValueError::new_err("missing end_text"));
        };
        let start_ordinal = date_text_ordinal(&start_text)?;
        let end_ordinal = date_text_ordinal(&end_text)?;
        let Some(matched_rows) = response_by_instrument.get(&instrument) else {
            continue;
        };
        for matched_row in matched_rows {
            let Some(returned_date_ordinal) = matched_row.date_ordinal else {
                continue;
            };
            if returned_date_ordinal < start_ordinal || returned_date_ordinal > end_ordinal {
                continue;
            }
            if matched_row.value.is_none() && matched_row.lookup_category.is_none() {
                continue;
            }
            let Some(date_text) = matched_row.date_text.as_ref() else {
                continue;
            };
            out.push((
                item_index,
                instrument.clone(),
                date_text.clone(),
                matched_row.lookup_category.clone(),
                matched_row.value,
            ));
        }
    }
    Ok(out)
}

#[pyfunction]
pub(crate) fn normalize_doc_ownership_batch_response_rows_value(
    item_rows: &Bound<'_, PyAny>,
    response_rows: &Bound<'_, PyAny>,
) -> PyResult<Vec<DocOwnershipBatchResponseRow>> {
    let response_by_instrument = parse_ownership_response_rows_by_instrument(response_rows)?;
    let mut out: Vec<DocOwnershipBatchResponseRow> = Vec::new();
    for item_row in item_rows.iter()? {
        let item_row = item_row?;
        let dict = item_row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("doc ownership item row is not a dict"))?;
        let Some(item_index_value) = dict.get_item("item_index")? else {
            return Err(PyValueError::new_err("missing item_index"));
        };
        let item_index = item_index_value.extract::<usize>()?;
        let Some(instrument) = dict_normalized_string(dict, "instrument")? else {
            continue;
        };
        let Some(matched_rows) = response_by_instrument.get(&instrument) else {
            continue;
        };
        for matched_row in matched_rows {
            if matched_row.date_text.is_none()
                && matched_row.value.is_none()
                && matched_row.doc_category.is_none()
            {
                continue;
            }
            out.push((
                item_index,
                matched_row.date_text.clone(),
                matched_row.doc_category.clone(),
                matched_row.value,
                is_doc_ownership_institutional_category(matched_row.doc_category.as_deref()),
            ));
        }
    }
    Ok(out)
}

#[pyfunction]
pub(crate) fn normalize_doc_ownership_batch_response_columns(
    py: Python<'_>,
    instruments: &Bound<'_, PyAny>,
    response_column_names: Vec<String>,
    response_column_values: &Bound<'_, PyAny>,
) -> PyResult<Vec<DocOwnershipBatchResponseRow>> {
    let instruments = pyobject_sequence(py, instruments)?;
    let response_by_instrument = parse_ownership_response_columns_by_instrument(
        py,
        response_column_names,
        response_column_values,
    )?;
    let mut out: Vec<DocOwnershipBatchResponseRow> = Vec::new();
    for (item_index, instrument_value) in instruments.iter().enumerate() {
        let Some(instrument) = normalize_lookup_text_any_impl(Some(instrument_value.bind(py)))?
        else {
            continue;
        };
        let Some(matched_rows) = response_by_instrument.get(&instrument) else {
            continue;
        };
        for matched_row in matched_rows {
            if matched_row.date_text.is_none()
                && matched_row.value.is_none()
                && matched_row.doc_category.is_none()
            {
                continue;
            }
            out.push((
                item_index,
                matched_row.date_text.clone(),
                matched_row.doc_category.clone(),
                matched_row.value,
                is_doc_ownership_institutional_category(matched_row.doc_category.as_deref()),
            ));
        }
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (start_ordinal=None, end_ordinal=None))]
pub(crate) fn refinitiv_authority_date_span_days(
    start_ordinal: Option<i64>,
    end_ordinal: Option<i64>,
) -> i64 {
    let (Some(start), Some(end)) = (start_ordinal, end_ordinal) else {
        return 0;
    };
    if end < start {
        0
    } else {
        end - start + 1
    }
}

#[pyfunction]
#[pyo3(signature = (left=None, right=None, tolerance=0.000001))]
pub(crate) fn refinitiv_authority_values_match(
    left: Option<f64>,
    right: Option<f64>,
    tolerance: f64,
) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(left), Some(right)) => (left - right).abs() <= tolerance,
        _ => false,
    }
}

#[pyfunction]
pub(crate) fn refinitiv_authority_distinct_values(values: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    let mut out: Vec<f64> = Vec::new();
    for entry in values.iter()? {
        let entry = entry?;
        if entry.is_none() {
            continue;
        }
        let number = entry.extract::<f64>()?;
        if !number.is_finite() {
            return Err(PyValueError::new_err(
                "non-finite authority value falls back to Python",
            ));
        }
        out.push(number);
    }
    out.sort_by(|left, right| {
        left.partial_cmp(right)
            .expect("finite floats should be totally comparable")
    });
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (kypermno=None, candidate_ric=None))]
pub(crate) fn refinitiv_authority_candidate_key(
    kypermno: Option<&Bound<'_, PyAny>>,
    candidate_ric: Option<&Bound<'_, PyAny>>,
) -> PyResult<(Option<String>, Option<String>)> {
    Ok((
        normalize_lookup_text_any_impl(kypermno)?,
        normalize_lookup_text_any_impl(candidate_ric)?,
    ))
}

#[pyfunction]
pub(crate) fn refinitiv_authority_allowlist_keys(
    rows: &Bound<'_, PyAny>,
) -> PyResult<Vec<(Option<String>, Option<String>)>> {
    let mut out: Vec<(Option<String>, Option<String>)> = Vec::new();
    for row in rows.iter()? {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("allowlist row is not a dict"))?;
        out.push((
            dict_normalized_string(dict, "KYPERMNO")?,
            dict_normalized_string(dict, "candidate_ric")?,
        ));
    }
    Ok(out)
}

#[pyfunction]
pub(crate) fn refinitiv_authority_allowlist_key_columns(
    kypermno_values: &Bound<'_, PyAny>,
    candidate_ric_values: &Bound<'_, PyAny>,
) -> PyResult<Vec<(Option<String>, Option<String>)>> {
    let mut out: Vec<(Option<String>, Option<String>)> = Vec::new();
    let mut kypermno_iter = kypermno_values.iter()?;
    let mut candidate_ric_iter = candidate_ric_values.iter()?;
    loop {
        match (kypermno_iter.next(), candidate_ric_iter.next()) {
            (None, None) => break,
            (Some(_), None) | (None, Some(_)) => {
                return Err(PyValueError::new_err(
                    "KYPERMNO and candidate_ric columns must have the same length",
                ))
            }
            (Some(kypermno), Some(candidate_ric)) => {
                let kypermno = kypermno?;
                let candidate_ric = candidate_ric?;
                out.push((
                    normalize_lookup_text_any_impl(Some(&kypermno))?,
                    normalize_lookup_text_any_impl(Some(&candidate_ric))?,
                ));
            }
        }
    }
    Ok(out)
}

#[pyfunction]
pub(crate) fn refinitiv_authority_source_family_from_flags(
    has_conventional_support: bool,
    has_ticker_support: bool,
) -> &'static str {
    if has_conventional_support && has_ticker_support {
        "MIXED"
    } else if has_conventional_support {
        "CONVENTIONAL"
    } else {
        "TICKER"
    }
}

#[pyfunction]
pub(crate) fn refinitiv_authority_component_id(kypermno: &str, component_index: i64) -> String {
    format!("{kypermno}|COMPONENT|{component_index:02}")
}

#[pyfunction]
pub(crate) fn refinitiv_authority_merge_intervals(
    mut intervals: Vec<(i64, i64)>,
    max_gap_days: i64,
) -> Vec<(i64, i64)> {
    if intervals.is_empty() {
        return Vec::new();
    }
    intervals.sort_unstable();
    let mut merged: Vec<(i64, i64)> = Vec::with_capacity(intervals.len());
    merged.push(intervals[0]);
    for (start_date, end_date) in intervals.into_iter().skip(1) {
        let last_index = merged.len() - 1;
        let (prior_start, prior_end) = merged[last_index];
        if start_date - prior_end <= max_gap_days + 1 {
            merged[last_index] = (prior_start, prior_end.max(end_date));
        } else {
            merged.push((start_date, end_date));
        }
    }
    merged
}

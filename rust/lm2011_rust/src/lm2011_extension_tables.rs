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
pub(crate) fn lm2011_extension_normal_approx_two_sided_p_value(t_stat: f64) -> Option<f64> {
    if !t_stat.is_finite() {
        return None;
    }
    Some(erfc_approx(t_stat.abs() / 2.0_f64.sqrt()))
}

pub(crate) fn python_round_f64_to_i64(value: f64) -> PyResult<i64> {
    if !value.is_finite() {
        return Err(PyValueError::new_err(
            "cannot convert non-finite float to integer",
        ));
    }
    if value < i64::MIN as f64 || value > i64::MAX as f64 {
        return Err(PyValueError::new_err("rounded value is out of i64 range"));
    }

    let floor = value.floor();
    let frac = value - floor;
    let tie_tolerance = f64::EPSILON * value.abs().max(1.0);
    let rounded = if (frac - 0.5).abs() <= tie_tolerance {
        let floor_i = floor as i64;
        if floor_i % 2 == 0 {
            floor_i
        } else {
            floor_i + 1
        }
    } else if frac < 0.5 {
        floor as i64
    } else {
        floor as i64 + 1
    };
    Ok(rounded)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn lm2011_extension_result_rows(
    py: Python<'_>,
    rows: &Bound<'_, PyAny>,
    run_id: &str,
    sample_window: &str,
    outcome_name: &str,
    feature_family: &str,
    control_set_id: &str,
    control_set_alias: &str,
    specification_name: &str,
    signal_name: &str,
) -> PyResult<Vec<PyObject>> {
    let mut out_rows = Vec::new();
    for row in rows.iter()? {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("LM2011 extension result row is not a dict"))?;

        let n_quarters = dict_optional_int(py, dict, "n_quarters")?;
        let mean_quarter_n = dict_optional_float(dict, "mean_quarter_n")?;
        let n_obs = match (n_quarters, mean_quarter_n) {
            (Some(n_quarters), Some(mean_quarter_n)) => {
                Some(python_round_f64_to_i64(n_quarters as f64 * mean_quarter_n)?)
            }
            _ => None,
        };
        let t_stat = dict_optional_float(dict, "t_stat")?;
        let p_value = t_stat.and_then(lm2011_extension_normal_approx_two_sided_p_value);

        let out = PyDict::new_bound(py);
        out.set_item("run_id", run_id)?;
        out.set_item("sample_window", sample_window)?;
        out.set_item("text_scope", dict_raw_string(dict, "text_scope")?)?;
        out.set_item("outcome_name", outcome_name)?;
        out.set_item("feature_family", feature_family)?;
        out.set_item("control_set_id", control_set_id)?;
        out.set_item("control_set_alias", control_set_alias)?;
        out.set_item("specification_name", specification_name)?;
        out.set_item(
            "coefficient_name",
            dict_raw_string(dict, "coefficient_name")?,
        )?;
        out.set_item("signal_name", signal_name)?;
        out.set_item("estimate", dict_optional_float(dict, "estimate")?)?;
        out.set_item(
            "standard_error",
            dict_optional_float(dict, "standard_error")?,
        )?;
        out.set_item("t_stat", t_stat)?;
        out.set_item("p_value", p_value)?;
        out.set_item("n_obs", n_obs)?;
        out.set_item("n_quarters", n_quarters)?;
        out.set_item("mean_quarter_n", mean_quarter_n)?;
        out.set_item("average_r2", Option::<f64>::None)?;
        out.set_item("weighting_rule", dict_raw_string(dict, "weighting_rule")?)?;
        out.set_item("nw_lags", dict_optional_int(py, dict, "nw_lags")?)?;
        out.set_item("estimator_status", "estimated")?;
        out.set_item("failure_reason", Option::<String>::None)?;
        out_rows.push(out.into_py(py));
    }
    Ok(out_rows)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn lm2011_extension_quarterly_fit_rows(
    py: Python<'_>,
    rows: &Bound<'_, PyAny>,
    run_id: &str,
    sample_window: &str,
    text_scope: &str,
    outcome_name: &str,
    feature_family: &str,
    control_set_id: &str,
    control_set_alias: &str,
    specification_name: &str,
    signal_name: &str,
    signal_inputs: Vec<String>,
    common_row_sample_policy: &str,
) -> PyResult<Vec<PyObject>> {
    let signal_inputs_list = PyList::new_bound(py, signal_inputs);
    let passthrough_keys = [
        "quarter_start",
        "n_obs",
        "industry_count",
        "industry_dummy_count",
        "visible_regressor_count",
        "full_regressor_count",
        "rank",
        "df_model",
        "df_resid",
        "condition_number",
        "raw_r2",
        "adj_r2",
        "ssr",
        "centered_tss",
        "weight",
        "weighting_rule",
    ];
    let mut out_rows = Vec::new();
    for row in rows.iter()? {
        let row = row?;
        let dict = row.downcast::<PyDict>().map_err(|_| {
            PyValueError::new_err("LM2011 extension quarterly fit row is not a dict")
        })?;

        let out = PyDict::new_bound(py);
        out.set_item("run_id", run_id)?;
        out.set_item("sample_window", sample_window)?;
        out.set_item("text_scope", text_scope)?;
        out.set_item("outcome_name", outcome_name)?;
        out.set_item("feature_family", feature_family)?;
        out.set_item("control_set_id", control_set_id)?;
        out.set_item("control_set_alias", control_set_alias)?;
        out.set_item("specification_name", specification_name)?;
        out.set_item("signal_name", signal_name)?;
        out.set_item("signal_inputs", &signal_inputs_list)?;
        for key in passthrough_keys {
            out.set_item(key, dict_required_py_object(py, dict, key)?)?;
        }
        out.set_item("common_row_sample_policy", common_row_sample_policy)?;
        out_rows.push(out.into_py(py));
    }
    Ok(out_rows)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn lm2011_extension_skipped_quarter_rows(
    py: Python<'_>,
    rows: &Bound<'_, PyAny>,
    run_id: &str,
    sample_window: &str,
    text_scope: &str,
    outcome_name: &str,
    feature_family: &str,
    control_set_id: &str,
    control_set_alias: &str,
    specification_name: &str,
    signal_name: &str,
    signal_inputs: Vec<String>,
) -> PyResult<Vec<PyObject>> {
    let signal_inputs_list = PyList::new_bound(py, signal_inputs);
    let passthrough_keys = [
        "quarter_start",
        "skip_reason",
        "n_obs",
        "industry_count",
        "rank",
        "column_count",
        "condition_number",
        "regressors",
        "duplicate_regressor_pairs",
        "restoring_drop_candidates",
    ];
    let mut out_rows = Vec::new();
    for row in rows.iter()? {
        let row = row?;
        let dict = row.downcast::<PyDict>().map_err(|_| {
            PyValueError::new_err("LM2011 extension skipped quarter row is not a dict")
        })?;

        let out = PyDict::new_bound(py);
        out.set_item("run_id", run_id)?;
        out.set_item("sample_window", sample_window)?;
        out.set_item("text_scope", text_scope)?;
        out.set_item("outcome_name", outcome_name)?;
        out.set_item("feature_family", feature_family)?;
        out.set_item("control_set_id", control_set_id)?;
        out.set_item("control_set_alias", control_set_alias)?;
        out.set_item("specification_name", specification_name)?;
        out.set_item("signal_name", signal_name)?;
        out.set_item("signal_inputs", &signal_inputs_list)?;
        for key in passthrough_keys {
            out.set_item(key, dict_required_py_object(py, dict, key)?)?;
        }
        out_rows.push(out.into_py(py));
    }
    Ok(out_rows)
}

pub(crate) fn collect_pyobject_column_values(
    py: Python<'_>,
    column_names: &[String],
    column_values: &Bound<'_, PyAny>,
    label: &str,
) -> PyResult<(Vec<Vec<PyObject>>, usize)> {
    let mut columns: Vec<Vec<PyObject>> = Vec::with_capacity(column_names.len());
    let mut row_count: Option<usize> = None;
    for values in column_values.iter()? {
        let values = values?;
        let column = pyobject_sequence(py, &values)?;
        match row_count {
            Some(expected) if column.len() != expected => {
                return Err(PyValueError::new_err(format!(
                    "all {label} columns must have the same length"
                )))
            }
            None => row_count = Some(column.len()),
            _ => {}
        }
        columns.push(column);
    }
    if columns.len() != column_names.len() {
        return Err(PyValueError::new_err(format!(
            "{label} column count does not match column_names length"
        )));
    }
    Ok((columns, row_count.unwrap_or(0)))
}

pub(crate) fn required_named_column_index(
    column_index: &HashMap<String, usize>,
    label: &str,
    column_name: &str,
) -> PyResult<usize> {
    column_index.get(column_name).copied().ok_or_else(|| {
        PyValueError::new_err(format!("missing required {label} column: {column_name}"))
    })
}

pub(crate) fn optional_column_value<'a>(
    columns: &'a [Vec<PyObject>],
    column_index: &HashMap<String, usize>,
    row_idx: usize,
    column_name: &str,
) -> Option<&'a PyObject> {
    column_index
        .get(column_name)
        .and_then(|column_idx| columns.get(*column_idx))
        .and_then(|column| column.get(row_idx))
}

pub(crate) fn optional_column_pyobject(
    py: Python<'_>,
    columns: &[Vec<PyObject>],
    column_index: &HashMap<String, usize>,
    row_idx: usize,
    column_name: &str,
) -> PyObject {
    optional_column_value(columns, column_index, row_idx, column_name)
        .map(|value| value.clone_ref(py))
        .unwrap_or_else(|| py.None())
}

pub(crate) fn optional_column_float(
    py: Python<'_>,
    columns: &[Vec<PyObject>],
    column_index: &HashMap<String, usize>,
    row_idx: usize,
    column_name: &str,
) -> PyResult<Option<f64>> {
    let Some(value) = optional_column_value(columns, column_index, row_idx, column_name) else {
        return Ok(None);
    };
    normalize_doc_ownership_float_value(Some(value.bind(py)))
}

pub(crate) fn optional_column_int(
    py: Python<'_>,
    columns: &[Vec<PyObject>],
    column_index: &HashMap<String, usize>,
    row_idx: usize,
    column_name: &str,
) -> PyResult<Option<i64>> {
    let Some(value) = optional_column_value(columns, column_index, row_idx, column_name) else {
        return Ok(None);
    };
    let Some(int_obj) = normalize_doc_ownership_int_value(py, Some(value.bind(py)))? else {
        return Ok(None);
    };
    int_obj.extract::<i64>(py).map(Some)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn lm2011_extension_result_columns(
    py: Python<'_>,
    column_names: Vec<String>,
    column_values: &Bound<'_, PyAny>,
    run_id: &str,
    sample_window: &str,
    outcome_name: &str,
    feature_family: &str,
    control_set_id: &str,
    control_set_alias: &str,
    specification_name: &str,
    signal_name: &str,
) -> PyResult<Vec<PyObject>> {
    let label = "LM2011 extension result";
    let (columns, row_count) =
        collect_pyobject_column_values(py, &column_names, column_values, label)?;
    let column_index = column_index_by_name(&column_names);
    let mut out_rows = Vec::with_capacity(row_count);
    for row_idx in 0..row_count {
        let n_quarters = optional_column_int(py, &columns, &column_index, row_idx, "n_quarters")?;
        let mean_quarter_n =
            optional_column_float(py, &columns, &column_index, row_idx, "mean_quarter_n")?;
        let n_obs = match (n_quarters, mean_quarter_n) {
            (Some(n_quarters), Some(mean_quarter_n)) => {
                Some(python_round_f64_to_i64(n_quarters as f64 * mean_quarter_n)?)
            }
            _ => None,
        };
        let t_stat = optional_column_float(py, &columns, &column_index, row_idx, "t_stat")?;
        let p_value = t_stat.and_then(lm2011_extension_normal_approx_two_sided_p_value);

        let out = PyDict::new_bound(py);
        out.set_item("run_id", run_id)?;
        out.set_item("sample_window", sample_window)?;
        out.set_item(
            "text_scope",
            optional_column_pyobject(py, &columns, &column_index, row_idx, "text_scope"),
        )?;
        out.set_item("outcome_name", outcome_name)?;
        out.set_item("feature_family", feature_family)?;
        out.set_item("control_set_id", control_set_id)?;
        out.set_item("control_set_alias", control_set_alias)?;
        out.set_item("specification_name", specification_name)?;
        out.set_item(
            "coefficient_name",
            optional_column_pyobject(py, &columns, &column_index, row_idx, "coefficient_name"),
        )?;
        out.set_item("signal_name", signal_name)?;
        out.set_item(
            "estimate",
            optional_column_float(py, &columns, &column_index, row_idx, "estimate")?,
        )?;
        out.set_item(
            "standard_error",
            optional_column_float(py, &columns, &column_index, row_idx, "standard_error")?,
        )?;
        out.set_item("t_stat", t_stat)?;
        out.set_item("p_value", p_value)?;
        out.set_item("n_obs", n_obs)?;
        out.set_item("n_quarters", n_quarters)?;
        out.set_item("mean_quarter_n", mean_quarter_n)?;
        out.set_item("average_r2", Option::<f64>::None)?;
        out.set_item(
            "weighting_rule",
            optional_column_pyobject(py, &columns, &column_index, row_idx, "weighting_rule"),
        )?;
        out.set_item(
            "nw_lags",
            optional_column_int(py, &columns, &column_index, row_idx, "nw_lags")?,
        )?;
        out.set_item("estimator_status", "estimated")?;
        out.set_item("failure_reason", Option::<String>::None)?;
        out_rows.push(out.into_py(py));
    }
    Ok(out_rows)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn lm2011_extension_quarterly_fit_columns(
    py: Python<'_>,
    column_names: Vec<String>,
    column_values: &Bound<'_, PyAny>,
    run_id: &str,
    sample_window: &str,
    text_scope: &str,
    outcome_name: &str,
    feature_family: &str,
    control_set_id: &str,
    control_set_alias: &str,
    specification_name: &str,
    signal_name: &str,
    signal_inputs: Vec<String>,
    common_row_sample_policy: &str,
) -> PyResult<Vec<PyObject>> {
    let label = "LM2011 extension quarterly fit";
    let (columns, row_count) =
        collect_pyobject_column_values(py, &column_names, column_values, label)?;
    let column_index = column_index_by_name(&column_names);
    let passthrough_keys = [
        "quarter_start",
        "n_obs",
        "industry_count",
        "industry_dummy_count",
        "visible_regressor_count",
        "full_regressor_count",
        "rank",
        "df_model",
        "df_resid",
        "condition_number",
        "raw_r2",
        "adj_r2",
        "ssr",
        "centered_tss",
        "weight",
        "weighting_rule",
    ];
    let passthrough_indices = passthrough_keys
        .iter()
        .map(|key| required_named_column_index(&column_index, label, key).map(|idx| (*key, idx)))
        .collect::<PyResult<Vec<_>>>()?;
    let signal_inputs_list = PyList::new_bound(py, signal_inputs);
    let mut out_rows = Vec::with_capacity(row_count);
    for row_idx in 0..row_count {
        let out = PyDict::new_bound(py);
        out.set_item("run_id", run_id)?;
        out.set_item("sample_window", sample_window)?;
        out.set_item("text_scope", text_scope)?;
        out.set_item("outcome_name", outcome_name)?;
        out.set_item("feature_family", feature_family)?;
        out.set_item("control_set_id", control_set_id)?;
        out.set_item("control_set_alias", control_set_alias)?;
        out.set_item("specification_name", specification_name)?;
        out.set_item("signal_name", signal_name)?;
        out.set_item("signal_inputs", &signal_inputs_list)?;
        for (key, column_idx) in &passthrough_indices {
            out.set_item(*key, columns[*column_idx][row_idx].bind(py))?;
        }
        out.set_item("common_row_sample_policy", common_row_sample_policy)?;
        out_rows.push(out.into_py(py));
    }
    Ok(out_rows)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn lm2011_extension_skipped_quarter_columns(
    py: Python<'_>,
    column_names: Vec<String>,
    column_values: &Bound<'_, PyAny>,
    run_id: &str,
    sample_window: &str,
    text_scope: &str,
    outcome_name: &str,
    feature_family: &str,
    control_set_id: &str,
    control_set_alias: &str,
    specification_name: &str,
    signal_name: &str,
    signal_inputs: Vec<String>,
) -> PyResult<Vec<PyObject>> {
    let label = "LM2011 extension skipped-quarter";
    let (columns, row_count) =
        collect_pyobject_column_values(py, &column_names, column_values, label)?;
    let column_index = column_index_by_name(&column_names);
    let passthrough_keys = [
        "quarter_start",
        "skip_reason",
        "n_obs",
        "industry_count",
        "rank",
        "column_count",
        "condition_number",
        "regressors",
        "duplicate_regressor_pairs",
        "restoring_drop_candidates",
    ];
    let passthrough_indices = passthrough_keys
        .iter()
        .map(|key| required_named_column_index(&column_index, label, key).map(|idx| (*key, idx)))
        .collect::<PyResult<Vec<_>>>()?;
    let signal_inputs_list = PyList::new_bound(py, signal_inputs);
    let mut out_rows = Vec::with_capacity(row_count);
    for row_idx in 0..row_count {
        let out = PyDict::new_bound(py);
        out.set_item("run_id", run_id)?;
        out.set_item("sample_window", sample_window)?;
        out.set_item("text_scope", text_scope)?;
        out.set_item("outcome_name", outcome_name)?;
        out.set_item("feature_family", feature_family)?;
        out.set_item("control_set_id", control_set_id)?;
        out.set_item("control_set_alias", control_set_alias)?;
        out.set_item("specification_name", specification_name)?;
        out.set_item("signal_name", signal_name)?;
        out.set_item("signal_inputs", &signal_inputs_list)?;
        for (key, column_idx) in &passthrough_indices {
            out.set_item(*key, columns[*column_idx][row_idx].bind(py))?;
        }
        out_rows.push(out.into_py(py));
    }
    Ok(out_rows)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn lm2011_extension_quarterly_difference_rows(
    py: Python<'_>,
    left_rows: &Bound<'_, PyAny>,
    right_rows: &Bound<'_, PyAny>,
    run_id: &str,
    sample_window: &str,
    text_scope: &str,
    outcome_name: &str,
    control_set_id: &str,
    control_set_alias: &str,
    comparison_name: &str,
    left_specification_name: &str,
    left_signal_name: &str,
    left_signal_inputs: Vec<String>,
    right_specification_name: &str,
    right_signal_name: &str,
    right_signal_inputs: Vec<String>,
    common_success_policy: &str,
) -> PyResult<Vec<PyObject>> {
    let mut left_dicts = Vec::new();
    for row in left_rows.iter()? {
        let row = row?;
        left_dicts.push(
            row.downcast::<PyDict>()
                .map_err(|_| {
                    PyValueError::new_err(
                        "LM2011 extension left quarterly difference row is not a dict",
                    )
                })?
                .to_owned(),
        );
    }
    let mut right_dicts = Vec::new();
    for row in right_rows.iter()? {
        let row = row?;
        right_dicts.push(
            row.downcast::<PyDict>()
                .map_err(|_| {
                    PyValueError::new_err(
                        "LM2011 extension right quarterly difference row is not a dict",
                    )
                })?
                .to_owned(),
        );
    }
    if left_dicts.len() != right_dicts.len() {
        return Err(PyValueError::new_err(
            "left_rows and right_rows must have the same length",
        ));
    }

    let left_signal_inputs_list = PyList::new_bound(py, left_signal_inputs);
    let right_signal_inputs_list = PyList::new_bound(py, right_signal_inputs);
    let mut out_rows = Vec::with_capacity(left_dicts.len());
    for (left, right) in left_dicts.iter().zip(right_dicts.iter()) {
        let n_obs = dict_required_i64(left, "n_obs")?;
        let left_raw_r2 = dict_required_float(left, "raw_r2")?;
        let right_raw_r2 = dict_required_float(right, "raw_r2")?;
        let left_adj_r2 = dict_required_float(left, "adj_r2")?;
        let right_adj_r2 = dict_required_float(right, "adj_r2")?;
        let delta_raw_r2 = left_raw_r2 - right_raw_r2;
        let delta_adj_r2 = left_adj_r2 - right_adj_r2;

        let out = PyDict::new_bound(py);
        out.set_item("run_id", run_id)?;
        out.set_item("sample_window", sample_window)?;
        out.set_item("text_scope", text_scope)?;
        out.set_item("outcome_name", outcome_name)?;
        out.set_item("control_set_id", control_set_id)?;
        out.set_item("control_set_alias", control_set_alias)?;
        out.set_item("comparison_name", comparison_name)?;
        out.set_item("left_specification_name", left_specification_name)?;
        out.set_item("left_signal_name", left_signal_name)?;
        out.set_item("left_signal_inputs", &left_signal_inputs_list)?;
        out.set_item("right_specification_name", right_specification_name)?;
        out.set_item("right_signal_name", right_signal_name)?;
        out.set_item("right_signal_inputs", &right_signal_inputs_list)?;
        out.set_item(
            "quarter_start",
            dict_required_py_object(py, left, "quarter_start")?,
        )?;
        out.set_item("n_obs", n_obs)?;
        out.set_item("weight", n_obs as f64)?;
        out.set_item("left_raw_r2", left_raw_r2)?;
        out.set_item("right_raw_r2", right_raw_r2)?;
        out.set_item("delta_raw_r2", delta_raw_r2)?;
        out.set_item("left_adj_r2", left_adj_r2)?;
        out.set_item("right_adj_r2", right_adj_r2)?;
        out.set_item("delta_adj_r2", delta_adj_r2)?;
        out.set_item("weighting_rule", "quarter_observation_count")?;
        out.set_item("common_success_policy", common_success_policy)?;
        out_rows.push(out.into_py(py));
    }
    Ok(out_rows)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn lm2011_extension_fit_summary_estimated_row(
    py: Python<'_>,
    rows: &Bound<'_, PyAny>,
    run_id: &str,
    sample_window: &str,
    text_scope: &str,
    outcome_name: &str,
    feature_family: &str,
    control_set_id: &str,
    control_set_alias: &str,
    specification_name: &str,
    signal_name: &str,
    signal_inputs: Vec<String>,
    common_success_policy: &str,
) -> PyResult<PyObject> {
    let mut weights = Vec::new();
    let mut raw_values = Vec::new();
    let mut adj_values = Vec::new();
    for row in rows.iter()? {
        let row = row?;
        let dict = row.downcast::<PyDict>().map_err(|_| {
            PyValueError::new_err("LM2011 extension fit summary row input is not a dict")
        })?;
        let n_obs = dict_required_i64(dict, "n_obs")?;
        weights.push(n_obs as f64);
        raw_values.push(dict_required_float(dict, "raw_r2")?);
        adj_values.push(dict_required_float(dict, "adj_r2")?);
    }
    if weights.is_empty() {
        return Err(PyValueError::new_err(
            "fit summary estimated row requires at least one quarter",
        ));
    }

    let n_quarters = weights.len() as i64;
    let total_n_obs_f64: f64 = weights.iter().sum();
    let total_n_obs = total_n_obs_f64 as i64;
    let mean_quarter_n = total_n_obs_f64 / weights.len() as f64;
    let weighted_avg_raw_r2 = lm2011_weighted_mean_impl(&raw_values, &weights)?;
    let weighted_avg_adj_r2 = lm2011_weighted_mean_impl(&adj_values, &weights)?;
    let equal_quarter_avg_raw_r2 = raw_values.iter().sum::<f64>() / raw_values.len() as f64;
    let equal_quarter_avg_adj_r2 = adj_values.iter().sum::<f64>() / adj_values.len() as f64;

    let out = PyDict::new_bound(py);
    out.set_item("run_id", run_id)?;
    out.set_item("sample_window", sample_window)?;
    out.set_item("text_scope", text_scope)?;
    out.set_item("outcome_name", outcome_name)?;
    out.set_item("feature_family", feature_family)?;
    out.set_item("control_set_id", control_set_id)?;
    out.set_item("control_set_alias", control_set_alias)?;
    out.set_item("specification_name", specification_name)?;
    out.set_item("signal_name", signal_name)?;
    out.set_item("signal_inputs", PyList::new_bound(py, signal_inputs))?;
    out.set_item("n_quarters", n_quarters)?;
    out.set_item("total_n_obs", total_n_obs)?;
    out.set_item("mean_quarter_n", mean_quarter_n)?;
    out.set_item("weighted_avg_raw_r2", weighted_avg_raw_r2)?;
    out.set_item("weighted_avg_adj_r2", weighted_avg_adj_r2)?;
    out.set_item("equal_quarter_avg_raw_r2", equal_quarter_avg_raw_r2)?;
    out.set_item("equal_quarter_avg_adj_r2", equal_quarter_avg_adj_r2)?;
    out.set_item("weighting_rule", "quarter_observation_count")?;
    out.set_item("common_success_policy", common_success_policy)?;
    out.set_item("estimator_status", "estimated")?;
    out.set_item("failure_reason", Option::<String>::None)?;
    Ok(out.into_py(py))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn lm2011_extension_fit_comparison_estimated_row(
    py: Python<'_>,
    rows: &Bound<'_, PyAny>,
    run_id: &str,
    sample_window: &str,
    text_scope: &str,
    outcome_name: &str,
    control_set_id: &str,
    control_set_alias: &str,
    comparison_name: &str,
    left_specification_name: &str,
    left_signal_name: &str,
    left_signal_inputs: Vec<String>,
    right_specification_name: &str,
    right_signal_name: &str,
    right_signal_inputs: Vec<String>,
    nw_lags: i64,
    common_success_policy: &str,
) -> PyResult<PyObject> {
    let mut weights = Vec::new();
    let mut delta_raw_values = Vec::new();
    let mut delta_adj_values = Vec::new();
    for row in rows.iter()? {
        let row = row?;
        let dict = row.downcast::<PyDict>().map_err(|_| {
            PyValueError::new_err("LM2011 extension fit comparison row input is not a dict")
        })?;
        weights.push(dict_required_float(dict, "weight")?);
        delta_raw_values.push(dict_required_float(dict, "delta_raw_r2")?);
        delta_adj_values.push(dict_required_float(dict, "delta_adj_r2")?);
    }
    if weights.is_empty() {
        return Err(PyValueError::new_err(
            "fit comparison estimated row requires at least one quarter",
        ));
    }

    let n_quarters = weights.len() as i64;
    let total_n_obs_f64: f64 = weights.iter().sum();
    let total_n_obs = total_n_obs_f64 as i64;
    let mean_quarter_n = total_n_obs_f64 / weights.len() as f64;
    let weighted_avg_delta_raw_r2 = lm2011_weighted_mean_impl(&delta_raw_values, &weights)?;
    let weighted_avg_delta_adj_r2 = lm2011_weighted_mean_impl(&delta_adj_values, &weights)?;
    let equal_quarter_avg_delta_raw_r2 =
        delta_raw_values.iter().sum::<f64>() / delta_raw_values.len() as f64;
    let equal_quarter_avg_delta_adj_r2 =
        delta_adj_values.iter().sum::<f64>() / delta_adj_values.len() as f64;

    let mut nw_se = None;
    let mut nw_t_stat = None;
    let mut nw_p_value = None;
    if delta_adj_values.len() >= 3 {
        nw_se =
            lm2011_newey_west_standard_error(delta_adj_values.clone(), weights.clone(), nw_lags)?;
        if let (Some(weighted_delta_adj), Some(standard_error)) = (weighted_avg_delta_adj_r2, nw_se)
        {
            if standard_error > 0.0 {
                let t_stat = weighted_delta_adj / standard_error;
                nw_t_stat = Some(t_stat);
                nw_p_value = lm2011_extension_normal_approx_two_sided_p_value(t_stat);
            }
        }
    }

    let out = PyDict::new_bound(py);
    out.set_item("run_id", run_id)?;
    out.set_item("sample_window", sample_window)?;
    out.set_item("text_scope", text_scope)?;
    out.set_item("outcome_name", outcome_name)?;
    out.set_item("control_set_id", control_set_id)?;
    out.set_item("control_set_alias", control_set_alias)?;
    out.set_item("comparison_name", comparison_name)?;
    out.set_item("left_specification_name", left_specification_name)?;
    out.set_item("left_signal_name", left_signal_name)?;
    out.set_item(
        "left_signal_inputs",
        PyList::new_bound(py, left_signal_inputs),
    )?;
    out.set_item("right_specification_name", right_specification_name)?;
    out.set_item("right_signal_name", right_signal_name)?;
    out.set_item(
        "right_signal_inputs",
        PyList::new_bound(py, right_signal_inputs),
    )?;
    out.set_item("n_quarters", n_quarters)?;
    out.set_item("total_n_obs", total_n_obs)?;
    out.set_item("mean_quarter_n", mean_quarter_n)?;
    out.set_item("weighted_avg_delta_raw_r2", weighted_avg_delta_raw_r2)?;
    out.set_item("weighted_avg_delta_adj_r2", weighted_avg_delta_adj_r2)?;
    out.set_item(
        "equal_quarter_avg_delta_raw_r2",
        equal_quarter_avg_delta_raw_r2,
    )?;
    out.set_item(
        "equal_quarter_avg_delta_adj_r2",
        equal_quarter_avg_delta_adj_r2,
    )?;
    out.set_item("nw_lags", nw_lags)?;
    out.set_item("nw_se_delta_adj_r2", nw_se)?;
    out.set_item("nw_t_stat_delta_adj_r2", nw_t_stat)?;
    out.set_item("nw_p_value_delta_adj_r2", nw_p_value)?;
    out.set_item("weighting_rule", "quarter_observation_count")?;
    out.set_item("common_success_policy", common_success_policy)?;
    out.set_item("estimator_status", "estimated")?;
    out.set_item("failure_reason", Option::<String>::None)?;
    Ok(out.into_py(py))
}

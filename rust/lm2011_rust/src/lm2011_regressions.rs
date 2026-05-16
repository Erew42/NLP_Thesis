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
pub(crate) fn lm2011_previous_month_end(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let (year, month, _) = extract_py_date_parts(value)?;
    if month == 1 {
        return py_date_object(py, year - 1, 12, 31);
    }
    let prev_month = month - 1;
    let Some(day) = days_in_month(year, prev_month) else {
        return Err(PyValueError::new_err("date month out of range"));
    };
    py_date_object(py, year, prev_month, day)
}

#[pyfunction]
pub(crate) fn lm2011_previous_month_end_values(
    py: Python<'_>,
    values: &Bound<'_, PyAny>,
) -> PyResult<Vec<Option<PyObject>>> {
    let mut out = Vec::new();
    for value in values.iter()? {
        let value = value?;
        if value.is_none() {
            out.push(None);
        } else {
            out.push(Some(lm2011_previous_month_end(py, &value)?));
        }
    }
    Ok(out)
}

#[pyfunction]
pub(crate) fn lm2011_quarter_start(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    let (year, month, _) = extract_py_date_parts(value)?;
    let quarter_month = ((month - 1) / 3) * 3 + 1;
    py_date_object(py, year, quarter_month, 1)
}

#[pyfunction]
pub(crate) fn lm2011_quarter_start_values(
    py: Python<'_>,
    values: &Bound<'_, PyAny>,
) -> PyResult<Vec<Option<PyObject>>> {
    let mut out = Vec::new();
    for value in values.iter()? {
        let value = value?;
        if value.is_none() {
            out.push(None);
        } else {
            out.push(Some(lm2011_quarter_start(py, &value)?));
        }
    }
    Ok(out)
}

pub(crate) fn validated_positive_int(
    value: &Bound<'_, PyAny>,
    message: &'static str,
) -> PyResult<i64> {
    let resolved = py_int_like_to_i64(value)?;
    if resolved < 1 {
        return Err(PyValueError::new_err(message));
    }
    Ok(resolved)
}

#[pyfunction]
pub(crate) fn lm2011_validated_event_window_doc_batch_size(
    value: &Bound<'_, PyAny>,
) -> PyResult<i64> {
    validated_positive_int(value, "event_window_doc_batch_size must be >= 1")
}

#[pyfunction]
pub(crate) fn lm2011_validated_event_window_days(value: &Bound<'_, PyAny>) -> PyResult<i64> {
    validated_positive_int(value, "event_window_days must be >= 1")
}

#[pyfunction]
pub(crate) fn lm2011_event_window_end_day(value: &Bound<'_, PyAny>) -> PyResult<i64> {
    Ok(lm2011_validated_event_window_days(value)? - 1)
}

#[pyfunction]
pub(crate) fn lm2011_postevent_start_day(value: &Bound<'_, PyAny>) -> PyResult<i64> {
    Ok(lm2011_validated_event_window_days(value)? + 2)
}

pub(crate) fn solve_4x4(mut matrix: [[f64; 4]; 4], mut rhs: [f64; 4]) -> Option<[f64; 4]> {
    const PIVOT_EPSILON: f64 = 1.0e-12;
    for col in 0..4 {
        let mut pivot_row = col;
        let mut pivot_abs = matrix[col][col].abs();
        for row in (col + 1)..4 {
            let candidate = matrix[row][col].abs();
            if candidate > pivot_abs {
                pivot_abs = candidate;
                pivot_row = row;
            }
        }
        if pivot_abs <= PIVOT_EPSILON {
            return None;
        }
        if pivot_row != col {
            matrix.swap(col, pivot_row);
            rhs.swap(col, pivot_row);
        }
        for row in (col + 1)..4 {
            let factor = matrix[row][col] / matrix[col][col];
            matrix[row][col] = 0.0;
            for inner_col in (col + 1)..4 {
                matrix[row][inner_col] -= factor * matrix[col][inner_col];
            }
            rhs[row] -= factor * rhs[col];
        }
    }

    let mut solution = [0.0; 4];
    for row in (0..4).rev() {
        let mut value = rhs[row];
        for col in (row + 1)..4 {
            value -= matrix[row][col] * solution[col];
        }
        if matrix[row][row].abs() <= PIVOT_EPSILON {
            return None;
        }
        solution[row] = value / matrix[row][row];
    }
    Some(solution)
}

#[pyfunction]
pub(crate) fn lm2011_ols_alpha_rmse(
    y: Vec<f64>,
    x1: Vec<f64>,
    x2: Vec<f64>,
    x3: Vec<f64>,
) -> PyResult<(Option<f64>, Option<f64>)> {
    let n = y.len();
    if x1.len() != n || x2.len() != n || x3.len() != n {
        return Err(PyValueError::new_err(
            "LM2011 OLS inputs must have equal lengths",
        ));
    }
    if n <= 4 {
        return Ok((None, None));
    }

    let mut xtx = [[0.0; 4]; 4];
    let mut xty = [0.0; 4];
    for idx in 0..n {
        let row = [1.0, x1[idx], x2[idx], x3[idx]];
        if !y[idx].is_finite() || row.iter().any(|value| !value.is_finite()) {
            return Err(PyValueError::new_err("LM2011 OLS inputs must be finite"));
        }
        for left in 0..4 {
            xty[left] += row[left] * y[idx];
            for right in 0..4 {
                xtx[left][right] += row[left] * row[right];
            }
        }
    }

    let Some(beta) = solve_4x4(xtx, xty) else {
        return Err(PyValueError::new_err("rank-deficient OLS design"));
    };
    let mut rss = 0.0;
    for idx in 0..n {
        let fitted = beta[0] + beta[1] * x1[idx] + beta[2] * x2[idx] + beta[3] * x3[idx];
        let residual = y[idx] - fitted;
        rss += residual * residual;
    }
    let mse_resid = rss / ((n - 4) as f64);
    Ok((Some(beta[0]), Some(mse_resid.max(0.0).sqrt())))
}

pub(crate) fn push_lm2011_regression_metric_row(
    py: Python<'_>,
    rows: &mut Vec<PyObject>,
    doc_id: &str,
    y: &[f64],
    x1: &[f64],
    x2: &[f64],
    x3: &[f64],
    alpha_name: &str,
    rmse_name: &str,
) -> PyResult<()> {
    let (alpha, rmse) = lm2011_ols_alpha_rmse(y.to_vec(), x1.to_vec(), x2.to_vec(), x3.to_vec())?;
    let out = PyDict::new_bound(py);
    out.set_item("doc_id", doc_id)?;
    out.set_item(alpha_name, alpha)?;
    out.set_item(rmse_name, rmse)?;
    out.set_item("n_obs", y.len() as i64)?;
    rows.push(out.into_py(py));
    Ok(())
}

#[pyfunction]
pub(crate) fn lm2011_regression_metrics_from_window_rows(
    py: Python<'_>,
    rows: &Bound<'_, PyAny>,
    alpha_name: &str,
    rmse_name: &str,
) -> PyResult<Vec<PyObject>> {
    let mut out = Vec::new();
    let mut current_doc_id: Option<String> = None;
    let mut y = Vec::new();
    let mut x1 = Vec::new();
    let mut x2 = Vec::new();
    let mut x3 = Vec::new();

    for row in rows.iter()? {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("LM2011 regression window row is not a dict"))?;
        let doc_id = dict_required_string(dict, "doc_id")?;
        if current_doc_id
            .as_ref()
            .is_some_and(|current| current != &doc_id)
        {
            let finished_doc_id = current_doc_id.take().unwrap_or_default();
            push_lm2011_regression_metric_row(
                py,
                &mut out,
                &finished_doc_id,
                &y,
                &x1,
                &x2,
                &x3,
                alpha_name,
                rmse_name,
            )?;
            y.clear();
            x1.clear();
            x2.clear();
            x3.clear();
        }
        if current_doc_id.is_none() {
            current_doc_id = Some(doc_id);
        }
        y.push(dict_required_float(dict, "_y")?);
        x1.push(dict_required_float(dict, "_x1")?);
        x2.push(dict_required_float(dict, "_x2")?);
        x3.push(dict_required_float(dict, "_x3")?);
    }

    if let Some(doc_id) = current_doc_id {
        push_lm2011_regression_metric_row(
            py, &mut out, &doc_id, &y, &x1, &x2, &x3, alpha_name, rmse_name,
        )?;
    }
    Ok(out)
}

#[pyfunction]
pub(crate) fn lm2011_regression_metrics_from_window_columns(
    py: Python<'_>,
    doc_ids: &Bound<'_, PyAny>,
    y_values: &Bound<'_, PyAny>,
    x1_values: &Bound<'_, PyAny>,
    x2_values: &Bound<'_, PyAny>,
    x3_values: &Bound<'_, PyAny>,
    alpha_name: &str,
    rmse_name: &str,
) -> PyResult<Vec<PyObject>> {
    let doc_ids = required_string_sequence(doc_ids, "doc_id")?;
    let y_values = required_float_sequence(y_values, "_y")?;
    let x1_values = required_float_sequence(x1_values, "_x1")?;
    let x2_values = required_float_sequence(x2_values, "_x2")?;
    let x3_values = required_float_sequence(x3_values, "_x3")?;
    let row_count = doc_ids.len();
    for (label, length) in [
        ("_y", y_values.len()),
        ("_x1", x1_values.len()),
        ("_x2", x2_values.len()),
        ("_x3", x3_values.len()),
    ] {
        if length != row_count {
            return Err(PyValueError::new_err(format!(
                "{label} length does not match doc_id length"
            )));
        }
    }

    let mut out = Vec::new();
    let mut current_doc_id: Option<String> = None;
    let mut y = Vec::new();
    let mut x1 = Vec::new();
    let mut x2 = Vec::new();
    let mut x3 = Vec::new();
    for row_idx in 0..row_count {
        let doc_id = &doc_ids[row_idx];
        if current_doc_id
            .as_ref()
            .is_some_and(|current| current != doc_id)
        {
            let finished_doc_id = current_doc_id.take().unwrap_or_default();
            push_lm2011_regression_metric_row(
                py,
                &mut out,
                &finished_doc_id,
                &y,
                &x1,
                &x2,
                &x3,
                alpha_name,
                rmse_name,
            )?;
            y.clear();
            x1.clear();
            x2.clear();
            x3.clear();
        }
        if current_doc_id.is_none() {
            current_doc_id = Some(doc_id.clone());
        }
        y.push(y_values[row_idx]);
        x1.push(x1_values[row_idx]);
        x2.push(x2_values[row_idx]);
        x3.push(x3_values[row_idx]);
    }

    if let Some(doc_id) = current_doc_id {
        push_lm2011_regression_metric_row(
            py, &mut out, &doc_id, &y, &x1, &x2, &x3, alpha_name, rmse_name,
        )?;
    }
    Ok(out)
}

pub(crate) fn invert_5x5(mut matrix: [[f64; 5]; 5]) -> Option<[[f64; 5]; 5]> {
    const PIVOT_EPSILON: f64 = 1.0e-12;
    let mut inverse = [[0.0; 5]; 5];
    for idx in 0..5 {
        inverse[idx][idx] = 1.0;
    }
    for col in 0..5 {
        let mut pivot_row = col;
        let mut pivot_abs = matrix[col][col].abs();
        for row in (col + 1)..5 {
            let candidate = matrix[row][col].abs();
            if candidate > pivot_abs {
                pivot_abs = candidate;
                pivot_row = row;
            }
        }
        if pivot_abs <= PIVOT_EPSILON {
            return None;
        }
        if pivot_row != col {
            matrix.swap(col, pivot_row);
            inverse.swap(col, pivot_row);
        }
        let pivot = matrix[col][col];
        for idx in 0..5 {
            matrix[col][idx] /= pivot;
            inverse[col][idx] /= pivot;
        }
        for row in 0..5 {
            if row == col {
                continue;
            }
            let factor = matrix[row][col];
            if factor == 0.0 {
                continue;
            }
            for idx in 0..5 {
                matrix[row][idx] -= factor * matrix[col][idx];
                inverse[row][idx] -= factor * inverse[col][idx];
            }
        }
    }
    Some(inverse)
}

pub(crate) fn finite_f64_option(value: f64) -> Option<f64> {
    if value.is_finite() {
        Some(value)
    } else {
        None
    }
}

#[pyfunction]
pub(crate) fn lm2011_ols_ff4_coefficients(
    y: Vec<f64>,
    mkt_rf: Vec<f64>,
    smb: Vec<f64>,
    hml: Vec<f64>,
    mom: Vec<f64>,
) -> PyResult<(
    Vec<Option<f64>>,
    Vec<Option<f64>>,
    Vec<Option<f64>>,
    Option<f64>,
)> {
    let n = y.len();
    if mkt_rf.len() != n || smb.len() != n || hml.len() != n || mom.len() != n {
        return Err(PyValueError::new_err(
            "LM2011 FF4 OLS inputs must have equal lengths",
        ));
    }
    if n <= 4 {
        let null_values = vec![None; 5];
        return Ok((null_values.clone(), null_values.clone(), null_values, None));
    }

    let mut xtx = [[0.0; 5]; 5];
    let mut xty = [0.0; 5];
    let y_mean = y.iter().sum::<f64>() / (n as f64);
    for idx in 0..n {
        let row = [1.0, mkt_rf[idx], smb[idx], hml[idx], mom[idx]];
        if !y[idx].is_finite() || row.iter().any(|value| !value.is_finite()) {
            return Err(PyValueError::new_err(
                "LM2011 FF4 OLS inputs must be finite",
            ));
        }
        for left in 0..5 {
            xty[left] += row[left] * y[idx];
            for right in 0..5 {
                xtx[left][right] += row[left] * row[right];
            }
        }
    }

    let Some(inverse) = invert_5x5(xtx) else {
        return Err(PyValueError::new_err("rank-deficient OLS design"));
    };
    let mut beta = [0.0; 5];
    for row in 0..5 {
        beta[row] = (0..5).map(|col| inverse[row][col] * xty[col]).sum();
    }

    let mut rss = 0.0;
    let mut centered_tss = 0.0;
    for idx in 0..n {
        let fitted = beta[0]
            + beta[1] * mkt_rf[idx]
            + beta[2] * smb[idx]
            + beta[3] * hml[idx]
            + beta[4] * mom[idx];
        let residual = y[idx] - fitted;
        rss += residual * residual;
        let centered = y[idx] - y_mean;
        centered_tss += centered * centered;
    }

    let coefficients: Vec<Option<f64>> =
        beta.iter().map(|value| finite_f64_option(*value)).collect();
    let r2 = if centered_tss > 0.0 {
        finite_f64_option(1.0 - (rss / centered_tss))
    } else {
        None
    };

    let df_resid = (n as i64) - 5;
    if df_resid <= 0 {
        let null_values = vec![None; 5];
        return Ok((coefficients, null_values.clone(), null_values, r2));
    }
    let sigma2 = rss / (df_resid as f64);
    let mut standard_errors = Vec::with_capacity(5);
    let mut t_stats = Vec::with_capacity(5);
    for idx in 0..5 {
        let variance = sigma2 * inverse[idx][idx];
        let se = if variance >= 0.0 {
            finite_f64_option(variance.sqrt())
        } else {
            None
        };
        let t_stat = match (coefficients[idx], se) {
            (Some(coef), Some(se_value)) if se_value > 0.0 => finite_f64_option(coef / se_value),
            _ => None,
        };
        standard_errors.push(se);
        t_stats.push(t_stat);
    }
    Ok((coefficients, standard_errors, t_stats, r2))
}

#[derive(Default)]
pub(crate) struct StrategyFactorGroup {
    signal_name: String,
    y: Vec<f64>,
    mkt_rf: Vec<f64>,
    smb: Vec<f64>,
    hml: Vec<f64>,
    mom: Vec<f64>,
}

pub(crate) fn option_vec_value(values: &[Option<f64>], idx: usize) -> Option<f64> {
    values.get(idx).copied().flatten()
}

pub(crate) fn push_strategy_factor_loading_row(
    py: Python<'_>,
    rows: &mut Vec<PyObject>,
    group: &StrategyFactorGroup,
) -> PyResult<()> {
    let (coefficients, standard_errors, t_stats, r2) = lm2011_ols_ff4_coefficients(
        group.y.clone(),
        group.mkt_rf.clone(),
        group.smb.clone(),
        group.hml.clone(),
        group.mom.clone(),
    )?;
    let out = PyDict::new_bound(py);
    out.set_item("sort_signal_name", &group.signal_name)?;
    out.set_item("alpha_ff3_mom", option_vec_value(&coefficients, 0))?;
    out.set_item(
        "alpha_ff3_mom_standard_error",
        option_vec_value(&standard_errors, 0),
    )?;
    out.set_item("alpha_ff3_mom_t_stat", option_vec_value(&t_stats, 0))?;
    out.set_item("beta_market", option_vec_value(&coefficients, 1))?;
    out.set_item(
        "beta_market_standard_error",
        option_vec_value(&standard_errors, 1),
    )?;
    out.set_item("beta_market_t_stat", option_vec_value(&t_stats, 1))?;
    out.set_item("beta_smb", option_vec_value(&coefficients, 2))?;
    out.set_item(
        "beta_smb_standard_error",
        option_vec_value(&standard_errors, 2),
    )?;
    out.set_item("beta_smb_t_stat", option_vec_value(&t_stats, 2))?;
    out.set_item("beta_hml", option_vec_value(&coefficients, 3))?;
    out.set_item(
        "beta_hml_standard_error",
        option_vec_value(&standard_errors, 3),
    )?;
    out.set_item("beta_hml_t_stat", option_vec_value(&t_stats, 3))?;
    out.set_item("beta_mom", option_vec_value(&coefficients, 4))?;
    out.set_item(
        "beta_mom_standard_error",
        option_vec_value(&standard_errors, 4),
    )?;
    out.set_item("beta_mom_t_stat", option_vec_value(&t_stats, 4))?;
    out.set_item("r2", r2)?;
    rows.push(out.into_py(py));
    Ok(())
}

#[pyfunction]
pub(crate) fn lm2011_strategy_factor_loading_rows(
    py: Python<'_>,
    rows: &Bound<'_, PyAny>,
) -> PyResult<Vec<PyObject>> {
    let mut groups: Vec<StrategyFactorGroup> = Vec::new();
    let mut group_index: HashMap<String, usize> = HashMap::new();
    for row in rows.iter()? {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("LM2011 strategy factor row is not a dict"))?;
        let signal_name = dict_required_string(dict, "sort_signal_name")?;
        let idx = match group_index.get(&signal_name) {
            Some(idx) => *idx,
            None => {
                let idx = groups.len();
                group_index.insert(signal_name.clone(), idx);
                groups.push(StrategyFactorGroup {
                    signal_name: signal_name.clone(),
                    ..Default::default()
                });
                idx
            }
        };
        let y = dict_optional_float(dict, "long_short_return")?;
        let mkt_rf = dict_optional_float(dict, "mkt_rf")?;
        let smb = dict_optional_float(dict, "smb")?;
        let hml = dict_optional_float(dict, "hml")?;
        let mom = dict_optional_float(dict, "mom")?;
        if let (Some(y), Some(mkt_rf), Some(smb), Some(hml), Some(mom)) = (y, mkt_rf, smb, hml, mom)
        {
            groups[idx].y.push(y);
            groups[idx].mkt_rf.push(mkt_rf);
            groups[idx].smb.push(smb);
            groups[idx].hml.push(hml);
            groups[idx].mom.push(mom);
        }
    }

    let mut out = Vec::new();
    for group in &groups {
        push_strategy_factor_loading_row(py, &mut out, group)?;
    }
    Ok(out)
}

pub(crate) fn required_string_sequence(
    values: &Bound<'_, PyAny>,
    label: &str,
) -> PyResult<Vec<String>> {
    let mut out = Vec::new();
    for value in values.iter()? {
        let value = value?;
        if value.is_none() {
            return Err(PyValueError::new_err(format!(
                "null required value: {label}"
            )));
        }
        out.push(value.str()?.to_str()?.to_string());
    }
    Ok(out)
}

pub(crate) fn optional_float_sequence(values: &Bound<'_, PyAny>) -> PyResult<Vec<Option<f64>>> {
    let mut out = Vec::new();
    for value in values.iter()? {
        let value = value?;
        out.push(normalize_doc_ownership_float_value(Some(&value))?);
    }
    Ok(out)
}

#[pyfunction]
pub(crate) fn lm2011_strategy_factor_loading_columns(
    py: Python<'_>,
    sort_signal_names: &Bound<'_, PyAny>,
    long_short_returns: &Bound<'_, PyAny>,
    mkt_rf_values: &Bound<'_, PyAny>,
    smb_values: &Bound<'_, PyAny>,
    hml_values: &Bound<'_, PyAny>,
    mom_values: &Bound<'_, PyAny>,
) -> PyResult<Vec<PyObject>> {
    let signal_names = required_string_sequence(sort_signal_names, "sort_signal_name")?;
    let y_values = optional_float_sequence(long_short_returns)?;
    let mkt_rf_values = optional_float_sequence(mkt_rf_values)?;
    let smb_values = optional_float_sequence(smb_values)?;
    let hml_values = optional_float_sequence(hml_values)?;
    let mom_values = optional_float_sequence(mom_values)?;
    let row_count = signal_names.len();
    for (label, length) in [
        ("long_short_return", y_values.len()),
        ("mkt_rf", mkt_rf_values.len()),
        ("smb", smb_values.len()),
        ("hml", hml_values.len()),
        ("mom", mom_values.len()),
    ] {
        if length != row_count {
            return Err(PyValueError::new_err(format!(
                "{label} length does not match sort_signal_name length"
            )));
        }
    }

    let mut groups: Vec<StrategyFactorGroup> = Vec::new();
    let mut group_index: HashMap<String, usize> = HashMap::new();
    for row_idx in 0..row_count {
        let signal_name = &signal_names[row_idx];
        let idx = match group_index.get(signal_name) {
            Some(idx) => *idx,
            None => {
                let idx = groups.len();
                group_index.insert(signal_name.clone(), idx);
                groups.push(StrategyFactorGroup {
                    signal_name: signal_name.clone(),
                    ..Default::default()
                });
                idx
            }
        };
        if let (Some(y), Some(mkt_rf), Some(smb), Some(hml), Some(mom)) = (
            y_values[row_idx],
            mkt_rf_values[row_idx],
            smb_values[row_idx],
            hml_values[row_idx],
            mom_values[row_idx],
        ) {
            groups[idx].y.push(y);
            groups[idx].mkt_rf.push(mkt_rf);
            groups[idx].smb.push(smb);
            groups[idx].hml.push(hml);
            groups[idx].mom.push(mom);
        }
    }

    let mut out = Vec::new();
    for group in &groups {
        push_strategy_factor_loading_row(py, &mut out, group)?;
    }
    Ok(out)
}

pub(crate) fn lm2011_weighted_mean_impl(values: &[f64], weights: &[f64]) -> PyResult<Option<f64>> {
    let total_weight: f64 = weights.iter().sum();
    if total_weight <= 0.0 {
        return Ok(None);
    }
    if values.len() != weights.len() {
        return Err(PyValueError::new_err(
            "zip() argument 2 is shorter/longer than argument 1",
        ));
    }
    let numerator: f64 = values
        .iter()
        .zip(weights.iter())
        .map(|(value, weight)| value * weight)
        .sum();
    Ok(Some(numerator / total_weight))
}

#[pyfunction]
pub(crate) fn lm2011_weighted_mean(values: Vec<f64>, weights: Vec<f64>) -> PyResult<Option<f64>> {
    lm2011_weighted_mean_impl(&values, &weights)
}

#[pyfunction]
pub(crate) fn lm2011_newey_west_standard_error(
    values: Vec<f64>,
    weights: Vec<f64>,
    nw_lags: i64,
) -> PyResult<Option<f64>> {
    let Some(estimate) = lm2011_weighted_mean_impl(&values, &weights)? else {
        return Ok(None);
    };
    let total_weight: f64 = weights.iter().sum();
    if total_weight <= 0.0 {
        return Ok(None);
    }
    if values.len() != weights.len() {
        return Err(PyValueError::new_err(
            "zip() argument 2 is shorter/longer than argument 1",
        ));
    }
    let psi: Vec<f64> = values
        .iter()
        .zip(weights.iter())
        .map(|(value, weight)| (weight / total_weight) * (value - estimate))
        .collect();
    let mut variance: f64 = psi.iter().map(|value| value * value).sum();
    if nw_lags > 0 && psi.len() > 1 {
        let max_lag = nw_lags.min((psi.len() - 1) as i64);
        for lag in 1..=max_lag {
            let bartlett_weight = 1.0 - ((lag as f64) / ((nw_lags + 1) as f64));
            let lag_usize = lag as usize;
            let covariance: f64 = (lag_usize..psi.len())
                .map(|idx| psi[idx] * psi[idx - lag_usize])
                .sum();
            variance += 2.0 * bartlett_weight * covariance;
        }
    }
    Ok(Some(variance.max(0.0).sqrt()))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn lm2011_fama_macbeth_result_rows(
    py: Python<'_>,
    series_items: &Bound<'_, PyAny>,
    quarter_sizes: Vec<f64>,
    quarter_weights: Vec<f64>,
    table_id: &str,
    specification_id: &str,
    text_scope: &str,
    signal_name: &str,
    dependent_variable: &str,
    weighting_rule: &str,
    nw_lags: i64,
) -> PyResult<Vec<PyObject>> {
    if quarter_sizes.is_empty() {
        return Err(PyValueError::new_err(
            "Fama-MacBeth result rows require at least one retained quarter",
        ));
    }
    let retained_quarters = quarter_sizes.len() as i64;
    let mean_quarter_n = quarter_sizes.iter().sum::<f64>() / quarter_sizes.len() as f64;
    let mut rows = Vec::new();
    for item in series_items.iter()? {
        let item = item?;
        let tuple = item.downcast::<PyTuple>().map_err(|_| {
            PyValueError::new_err("Fama-MacBeth coefficient series item is not a tuple")
        })?;
        if tuple.len() != 2 {
            return Err(PyValueError::new_err(
                "Fama-MacBeth coefficient series item must contain name and values",
            ));
        }
        let coefficient_name = tuple.get_item(0)?.str()?.to_str()?.to_string();
        let values: Vec<f64> = tuple.get_item(1)?.extract()?;
        let estimate = lm2011_weighted_mean_impl(&values, &quarter_weights)?;
        let standard_error =
            lm2011_newey_west_standard_error(values.clone(), quarter_weights.clone(), nw_lags)?;
        let mut t_stat = None;
        if let (Some(estimate_value), Some(standard_error_value)) = (estimate, standard_error) {
            if standard_error_value > 0.0 {
                t_stat = Some(estimate_value / standard_error_value);
            }
        }

        let out = PyDict::new_bound(py);
        out.set_item("table_id", table_id)?;
        out.set_item("specification_id", specification_id)?;
        out.set_item("text_scope", text_scope)?;
        out.set_item("signal_name", signal_name)?;
        out.set_item("dependent_variable", dependent_variable)?;
        out.set_item("coefficient_name", coefficient_name)?;
        out.set_item("estimate", estimate)?;
        out.set_item("standard_error", standard_error)?;
        out.set_item("t_stat", t_stat)?;
        out.set_item("n_quarters", retained_quarters)?;
        out.set_item("mean_quarter_n", mean_quarter_n)?;
        out.set_item("weighting_rule", weighting_rule)?;
        out.set_item("nw_lags", nw_lags)?;
        rows.push(out.into_py(py));
    }
    Ok(rows)
}

#[pyfunction]
pub(crate) fn lm2011_cross_section_design_rows(
    rows: &Bound<'_, PyAny>,
    dependent_variable: &str,
    industry_col: &str,
    regressor_columns: Vec<String>,
    dummy_industries: Vec<i64>,
) -> PyResult<(Vec<f64>, Vec<Vec<f64>>)> {
    let mut endog = Vec::new();
    let mut exog_rows = Vec::new();
    for row in rows.iter()? {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("LM2011 cross-section design row is not a dict"))?;
        let mut design_row = Vec::with_capacity(regressor_columns.len() + dummy_industries.len());
        for column in &regressor_columns {
            design_row.push(dict_required_float(dict, column)?);
        }
        let industry_value = dict_required_i64(dict, industry_col)?;
        for industry_id in &dummy_industries {
            design_row.push(if industry_value == *industry_id {
                1.0
            } else {
                0.0
            });
        }
        endog.push(dict_required_float(dict, dependent_variable)?);
        exog_rows.push(design_row);
    }
    Ok((endog, exog_rows))
}

pub(crate) fn required_float_sequence(
    values: &Bound<'_, PyAny>,
    label: &str,
) -> PyResult<Vec<f64>> {
    let mut out = Vec::new();
    for value in values.iter()? {
        let value = value?;
        let Some(number) = normalize_doc_ownership_float_value(Some(&value))? else {
            return Err(PyValueError::new_err(format!(
                "null required value: {label}"
            )));
        };
        out.push(number);
    }
    Ok(out)
}

pub(crate) fn required_i64_sequence(values: &Bound<'_, PyAny>, label: &str) -> PyResult<Vec<i64>> {
    let mut out = Vec::new();
    for value in values.iter()? {
        let value = value?;
        if value.is_none() {
            return Err(PyValueError::new_err(format!(
                "null required value: {label}"
            )));
        }
        out.push(py_int_like_to_i64(&value)?);
    }
    Ok(out)
}

#[pyfunction]
pub(crate) fn lm2011_cross_section_design_columns(
    dependent_values: &Bound<'_, PyAny>,
    industry_values: &Bound<'_, PyAny>,
    regressor_columns: Vec<String>,
    regressor_value_columns: &Bound<'_, PyAny>,
    dummy_industries: Vec<i64>,
) -> PyResult<(Vec<f64>, Vec<Vec<f64>>)> {
    let endog = required_float_sequence(dependent_values, "dependent_variable")?;
    let industries = required_i64_sequence(industry_values, "industry_col")?;
    let row_count = endog.len();
    if industries.len() != row_count {
        return Err(PyValueError::new_err(
            "industry column length does not match dependent column length",
        ));
    }

    let mut regressor_values: Vec<Vec<f64>> = Vec::new();
    for (idx, values) in regressor_value_columns.iter()?.enumerate() {
        let values = values?;
        let label = regressor_columns
            .get(idx)
            .map(String::as_str)
            .unwrap_or("regressor");
        let column_values = required_float_sequence(&values, label)?;
        if column_values.len() != row_count {
            return Err(PyValueError::new_err(format!(
                "regressor column length does not match dependent column length: {label}"
            )));
        }
        regressor_values.push(column_values);
    }
    if regressor_values.len() != regressor_columns.len() {
        return Err(PyValueError::new_err(
            "regressor column name/value count mismatch",
        ));
    }

    let mut exog_rows = Vec::with_capacity(row_count);
    for row_idx in 0..row_count {
        let mut design_row = Vec::with_capacity(regressor_values.len() + dummy_industries.len());
        for values in &regressor_values {
            design_row.push(values[row_idx]);
        }
        let industry_value = industries[row_idx];
        for industry_id in &dummy_industries {
            design_row.push(if industry_value == *industry_id {
                1.0
            } else {
                0.0
            });
        }
        exog_rows.push(design_row);
    }
    Ok((endog, exog_rows))
}

pub(crate) fn table_ia_ii_optional_float(
    dict: &Bound<'_, PyDict>,
    key: Option<&str>,
) -> PyResult<Option<f64>> {
    let Some(key) = key else {
        return Ok(None);
    };
    dict_optional_float(dict, key)
}

#[pyfunction]
pub(crate) fn lm2011_table_ia_ii_result_rows(
    py: Python<'_>,
    rows: &Bound<'_, PyAny>,
) -> PyResult<Vec<PyObject>> {
    let coefficient_pairs = [
        (
            "mean_long_short_return",
            "mean_long_short_return",
            None,
            None,
        ),
        (
            "alpha_ff3_mom",
            "alpha_ff3_mom",
            Some("alpha_ff3_mom_standard_error"),
            Some("alpha_ff3_mom_t_stat"),
        ),
        (
            "beta_market",
            "beta_market",
            Some("beta_market_standard_error"),
            Some("beta_market_t_stat"),
        ),
        (
            "beta_smb",
            "beta_smb",
            Some("beta_smb_standard_error"),
            Some("beta_smb_t_stat"),
        ),
        (
            "beta_hml",
            "beta_hml",
            Some("beta_hml_standard_error"),
            Some("beta_hml_t_stat"),
        ),
        (
            "beta_mom",
            "beta_mom",
            Some("beta_mom_standard_error"),
            Some("beta_mom_t_stat"),
        ),
        ("r2", "r2", None, None),
    ];

    let mut out_rows = Vec::new();
    for row in rows.iter()? {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("LM2011 Table IA.II input row is not a dict"))?;
        let Some(raw_signal_name) = dict.get_item("sort_signal_name")? else {
            return Err(PyValueError::new_err(
                "missing required key: sort_signal_name",
            ));
        };
        let signal_name = raw_signal_name.str()?.to_str()?.to_string();
        for (coefficient_name, source_column, standard_error_column, t_stat_column) in
            coefficient_pairs
        {
            let out = PyDict::new_bound(py);
            out.set_item("table_id", "internet_appendix_table_ia_ii")?;
            out.set_item("specification_id", &signal_name)?;
            out.set_item("text_scope", "full_10k")?;
            out.set_item("signal_name", &signal_name)?;
            out.set_item("dependent_variable", "long_short_return")?;
            out.set_item("coefficient_name", coefficient_name)?;
            out.set_item("estimate", dict_optional_float(dict, source_column)?)?;
            out.set_item(
                "standard_error",
                table_ia_ii_optional_float(dict, standard_error_column)?,
            )?;
            out.set_item("t_stat", table_ia_ii_optional_float(dict, t_stat_column)?)?;
            out.set_item("n_quarters", Option::<i64>::None)?;
            out.set_item("mean_quarter_n", Option::<f64>::None)?;
            out.set_item("weighting_rule", Option::<String>::None)?;
            out.set_item("nw_lags", Option::<i64>::None)?;
            out_rows.push(out.into_py(py));
        }
    }
    Ok(out_rows)
}

pub(crate) fn table_ia_ii_column_value(
    value_columns: &[Vec<Option<f64>>],
    column_idx: Option<usize>,
    row_idx: usize,
) -> Option<f64> {
    column_idx.and_then(|idx| value_columns.get(idx)?.get(row_idx).copied().flatten())
}

#[pyfunction]
pub(crate) fn lm2011_table_ia_ii_result_columns(
    py: Python<'_>,
    sort_signal_names: &Bound<'_, PyAny>,
    value_columns: &Bound<'_, PyAny>,
) -> PyResult<Vec<PyObject>> {
    let signal_names = required_string_sequence(sort_signal_names, "sort_signal_name")?;
    let row_count = signal_names.len();
    let mut columns: Vec<Vec<Option<f64>>> = Vec::new();
    for values in value_columns.iter()? {
        let values = values?;
        let column_values = optional_float_sequence(&values)?;
        if column_values.len() != row_count {
            return Err(PyValueError::new_err(
                "Table IA.II value column length does not match sort_signal_name length",
            ));
        }
        columns.push(column_values);
    }
    if columns.len() != 17 {
        return Err(PyValueError::new_err(
            "Table IA.II column helper expects 17 value columns",
        ));
    }

    let coefficient_pairs = [
        ("mean_long_short_return", 0usize, None, None),
        ("alpha_ff3_mom", 1usize, Some(2usize), Some(3usize)),
        ("beta_market", 4usize, Some(5usize), Some(6usize)),
        ("beta_smb", 7usize, Some(8usize), Some(9usize)),
        ("beta_hml", 10usize, Some(11usize), Some(12usize)),
        ("beta_mom", 13usize, Some(14usize), Some(15usize)),
        ("r2", 16usize, None, None),
    ];

    let mut out_rows = Vec::new();
    for (row_idx, signal_name) in signal_names.iter().enumerate() {
        for (coefficient_name, source_idx, standard_error_idx, t_stat_idx) in coefficient_pairs {
            let out = PyDict::new_bound(py);
            out.set_item("table_id", "internet_appendix_table_ia_ii")?;
            out.set_item("specification_id", signal_name)?;
            out.set_item("text_scope", "full_10k")?;
            out.set_item("signal_name", signal_name)?;
            out.set_item("dependent_variable", "long_short_return")?;
            out.set_item("coefficient_name", coefficient_name)?;
            out.set_item(
                "estimate",
                table_ia_ii_column_value(&columns, Some(source_idx), row_idx),
            )?;
            out.set_item(
                "standard_error",
                table_ia_ii_column_value(&columns, standard_error_idx, row_idx),
            )?;
            out.set_item(
                "t_stat",
                table_ia_ii_column_value(&columns, t_stat_idx, row_idx),
            )?;
            out.set_item("n_quarters", Option::<i64>::None)?;
            out.set_item("mean_quarter_n", Option::<f64>::None)?;
            out.set_item("weighting_rule", Option::<String>::None)?;
            out.set_item("nw_lags", Option::<i64>::None)?;
            out_rows.push(out.into_py(py));
        }
    }
    Ok(out_rows)
}

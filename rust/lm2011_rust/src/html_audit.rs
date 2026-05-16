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

pub(crate) struct HtmlAuditStratum {
    bucket: String,
    missing_core: bool,
    rows: Vec<PyObject>,
    target: usize,
}

pub(crate) struct HtmlAuditStatusRows {
    status: &'static str,
    rows: Vec<PyObject>,
    target: usize,
}

pub(crate) fn html_audit_missing_core_items(dict: &Bound<'_, PyDict>) -> PyResult<bool> {
    let Some(value) = dict.get_item("missing_core_items")? else {
        return Ok(false);
    };
    if value.is_none() || !value.is_truthy()? {
        return Ok(false);
    }
    let rendered = value.str()?;
    Ok(!rendered.to_str()?.trim().is_empty())
}

pub(crate) fn html_audit_sample_with_rng(
    py: Python<'_>,
    rng: &Bound<'_, PyAny>,
    rows: &[PyObject],
    sample_count: usize,
) -> PyResult<Vec<PyObject>> {
    if sample_count == 0 {
        return Ok(Vec::new());
    }
    let row_list = PyList::new_bound(
        py,
        rows.iter().map(|row| row.clone_ref(py)).collect::<Vec<_>>(),
    );
    let sampled = rng.call_method1("sample", (row_list, sample_count))?;
    let mut out = Vec::with_capacity(sample_count);
    for row in sampled.iter()? {
        out.push(row?.clone().into_py(py));
    }
    Ok(out)
}

pub(crate) fn html_audit_allocate_proportional_targets(
    py: Python<'_>,
    rng: &Bound<'_, PyAny>,
    capacities: &[usize],
    sample_size: usize,
    weights: Option<&[f64]>,
) -> PyResult<Vec<usize>> {
    let total: usize = capacities.iter().sum();
    if total == 0 || sample_size == 0 {
        return Ok(vec![0; capacities.len()]);
    }
    let sample_size = sample_size.min(total);
    let denominator = match weights {
        Some(weights) => {
            let sum: f64 = weights
                .iter()
                .zip(capacities.iter())
                .filter_map(
                    |(weight, capacity)| {
                        if *capacity > 0 {
                            Some(*weight)
                        } else {
                            None
                        }
                    },
                )
                .sum();
            if sum <= 0.0 {
                capacities.iter().filter(|capacity| **capacity > 0).count() as f64
            } else {
                sum
            }
        }
        None => total as f64,
    };

    let mut targets = vec![0_usize; capacities.len()];
    let mut fractional: Vec<(f64, usize)> = Vec::new();
    for (idx, capacity) in capacities.iter().enumerate() {
        if *capacity == 0 {
            continue;
        }
        let raw = match weights {
            Some(weights) => sample_size as f64 * (weights[idx] / denominator),
            None => sample_size as f64 * (*capacity as f64 / total as f64),
        };
        let base = raw.trunc() as usize;
        targets[idx] = base.min(*capacity);
        fractional.push((raw - base as f64, idx));
    }

    let mut remaining = sample_size.saturating_sub(targets.iter().sum());
    fractional.sort_by(|left, right| right.0.partial_cmp(&left.0).unwrap_or(Ordering::Equal));
    for (_, idx) in fractional {
        if remaining == 0 {
            break;
        }
        if targets[idx] < capacities[idx] {
            targets[idx] += 1;
            remaining -= 1;
        }
    }

    while remaining > 0 {
        let available: Vec<usize> = capacities
            .iter()
            .enumerate()
            .filter_map(|(idx, capacity)| {
                if targets[idx] < *capacity {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect();
        if available.is_empty() {
            break;
        }
        let available_list = PyList::new_bound(py, available);
        let chosen_idx = rng
            .call_method1("choice", (available_list,))?
            .extract::<usize>()?;
        targets[chosen_idx] += 1;
        remaining -= 1;
    }

    Ok(targets)
}

pub(crate) fn html_audit_sample_stratified_row_objects(
    py: Python<'_>,
    row_objects: &[PyObject],
    sample_size: i64,
    seed: i64,
) -> PyResult<Vec<PyObject>> {
    if row_objects.is_empty() || sample_size <= 0 {
        return Ok(Vec::new());
    }

    let mut item_counts = Vec::with_capacity(row_objects.len());
    for row in row_objects {
        let dict = row
            .bind(py)
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("HTML audit sampling row is not a dict"))?;
        item_counts.push(html_audit_parse_int_value(
            dict.get_item("n_items_extracted")?.as_ref(),
            0,
        )?);
    }
    let edges = html_audit_quartile_edges_value(item_counts.clone());
    let mut strata: Vec<HtmlAuditStratum> = Vec::new();
    for (row, item_count) in row_objects.iter().zip(item_counts.into_iter()) {
        let dict = row
            .bind(py)
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("HTML audit sampling row is not a dict"))?;
        let bucket = html_audit_quartile_bucket_value(item_count, edges).to_string();
        let missing_core = html_audit_missing_core_items(dict)?;
        if let Some(existing) = strata
            .iter_mut()
            .find(|entry| entry.bucket == bucket && entry.missing_core == missing_core)
        {
            existing.rows.push(row.clone_ref(py));
        } else {
            strata.push(HtmlAuditStratum {
                bucket,
                missing_core,
                rows: vec![row.clone_ref(py)],
                target: 0,
            });
        }
    }

    let random_cls = py.import_bound("random")?.getattr("Random")?;
    let rng = random_cls.call1((seed,))?;
    let capacities: Vec<usize> = strata.iter().map(|entry| entry.rows.len()).collect();
    let targets = html_audit_allocate_proportional_targets(
        py,
        &rng,
        &capacities,
        sample_size as usize,
        None,
    )?;
    for (entry, target) in strata.iter_mut().zip(targets.into_iter()) {
        entry.target = target;
    }

    let mut order: Vec<usize> = (0..strata.len()).collect();
    order.sort_by(|left, right| {
        let left_entry = &strata[*left];
        let right_entry = &strata[*right];
        left_entry
            .bucket
            .cmp(&right_entry.bucket)
            .then(left_entry.missing_core.cmp(&right_entry.missing_core))
    });

    let mut sampled = Vec::new();
    for idx in order {
        let target = strata[idx].target;
        if target == 0 {
            continue;
        }
        sampled.extend(html_audit_sample_with_rng(
            py,
            &rng,
            &strata[idx].rows,
            target,
        )?);
    }
    Ok(sampled)
}

#[pyfunction]
pub(crate) fn html_audit_sample_stratified_rows(
    py: Python<'_>,
    rows: &Bound<'_, PyAny>,
    sample_size: i64,
    seed: i64,
) -> PyResult<Vec<PyObject>> {
    let mut row_objects = Vec::new();
    for row in rows.iter()? {
        let row = row?;
        row.downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("HTML audit sampling row is not a dict"))?;
        row_objects.push(row.clone().into_py(py));
    }
    html_audit_sample_stratified_row_objects(py, &row_objects, sample_size, seed)
}

pub(crate) fn html_audit_weight_value(
    weights: Option<&Bound<'_, PyDict>>,
    status: &str,
    default: f64,
) -> PyResult<f64> {
    let Some(weights) = weights else {
        return Ok(default);
    };
    let Some(value) = weights.get_item(status)? else {
        return Ok(default);
    };
    if value.is_none() {
        return Ok(default);
    }
    let weight = value.extract::<f64>()?;
    Ok(weight.max(0.0))
}

#[pyfunction]
#[pyo3(signature = (rows, sample_size, seed, weights=None))]
pub(crate) fn html_audit_sample_filings_by_status(
    py: Python<'_>,
    rows: &Bound<'_, PyAny>,
    sample_size: i64,
    seed: i64,
    weights: Option<&Bound<'_, PyDict>>,
) -> PyResult<Vec<PyObject>> {
    if sample_size <= 0 {
        return Ok(Vec::new());
    }

    let mut status_rows = vec![
        HtmlAuditStatusRows {
            status: "pass",
            rows: Vec::new(),
            target: 0,
        },
        HtmlAuditStatusRows {
            status: "warning",
            rows: Vec::new(),
            target: 0,
        },
        HtmlAuditStatusRows {
            status: "fail",
            rows: Vec::new(),
            target: 0,
        },
    ];

    for row in rows.iter()? {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("HTML audit filing row is not a dict"))?;
        let status = html_audit_filing_status_value(dict)?;
        let status_idx = match status {
            "pass" => 0,
            "warning" => 1,
            "fail" => 2,
            _ => return Err(PyValueError::new_err("unknown HTML audit filing status")),
        };
        status_rows[status_idx].rows.push(row.clone().into_py(py));
    }

    let total_available: usize = status_rows.iter().map(|entry| entry.rows.len()).sum();
    if total_available == 0 {
        return Ok(Vec::new());
    }

    let random_cls = py.import_bound("random")?.getattr("Random")?;
    let rng = random_cls.call1((seed,))?;
    let capacities: Vec<usize> = status_rows.iter().map(|entry| entry.rows.len()).collect();
    let weight_values = vec![
        html_audit_weight_value(weights, "pass", 0.5)?,
        html_audit_weight_value(weights, "warning", 0.3)?,
        html_audit_weight_value(weights, "fail", 0.2)?,
    ];
    let targets = html_audit_allocate_proportional_targets(
        py,
        &rng,
        &capacities,
        sample_size as usize,
        Some(&weight_values),
    )?;
    for (entry, target) in status_rows.iter_mut().zip(targets.into_iter()) {
        entry.target = target;
    }

    let mut sampled = Vec::new();
    for (idx, entry) in status_rows.iter().enumerate() {
        if entry.target == 0 {
            continue;
        }
        let _status = entry.status;
        sampled.extend(html_audit_sample_stratified_row_objects(
            py,
            &entry.rows,
            entry.target as i64,
            seed + (idx as i64 + 1) * 101,
        )?);
    }
    Ok(sampled)
}

pub(crate) fn finbert_softmax_row_impl(values: &[f64]) -> PyResult<Vec<f64>> {
    if values.is_empty() {
        return Ok(Vec::new());
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("non-finite softmax input"));
    }
    let max_value = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exp_values: Vec<f64> = values
        .iter()
        .map(|value| (*value - max_value).exp())
        .collect();
    let denom: f64 = exp_values.iter().sum();
    if denom == 0.0 {
        return Ok(vec![0.0; values.len()]);
    }
    Ok(exp_values.into_iter().map(|value| value / denom).collect())
}

#[pyfunction]
pub(crate) fn finbert_softmax_row(values: Vec<f64>) -> PyResult<Vec<f64>> {
    finbert_softmax_row_impl(&values)
}

#[pyfunction]
pub(crate) fn finbert_softmax_rows(rows: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
    rows.iter()
        .map(|row| finbert_softmax_row_impl(row))
        .collect()
}

#[pyfunction]
pub(crate) fn finbert_probability_columns_and_predicted_labels(
    probability_rows: &Bound<'_, PyAny>,
    label_mapping: &Bound<'_, PyDict>,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<String>)> {
    let mut negative = Vec::new();
    let mut neutral = Vec::new();
    let mut positive = Vec::new();
    let mut predicted = Vec::new();

    for row in probability_rows.iter()? {
        let row = row?;
        let mut negative_value: Option<f64> = None;
        let mut neutral_value: Option<f64> = None;
        let mut positive_value: Option<f64> = None;
        for (index, probability) in row.iter()?.enumerate() {
            let probability = probability?;
            let Some(label) = label_mapping.get_item(index)? else {
                continue;
            };
            let label = label.str()?;
            match label.to_str()? {
                "negative" => negative_value = Some(probability.extract::<f64>()?),
                "neutral" => neutral_value = Some(probability.extract::<f64>()?),
                "positive" => positive_value = Some(probability.extract::<f64>()?),
                _ => {}
            }
        }

        let neg =
            negative_value.ok_or_else(|| PyValueError::new_err("missing negative probability"))?;
        let neu =
            neutral_value.ok_or_else(|| PyValueError::new_err("missing neutral probability"))?;
        let pos =
            positive_value.ok_or_else(|| PyValueError::new_err("missing positive probability"))?;
        let mut best_label = "negative";
        let mut best_value = neg;
        if neu > best_value {
            best_label = "neutral";
            best_value = neu;
        }
        if pos > best_value {
            best_label = "positive";
        }
        negative.push(neg);
        neutral.push(neu);
        positive.push(pos);
        predicted.push(best_label.to_string());
    }

    Ok((negative, neutral, positive, predicted))
}

#[pyfunction]
pub(crate) fn finbert_confusion_candidate_threshold(
    population_count: i64,
    sample_count: i64,
    oversampling_factor: f64,
) -> PyResult<f64> {
    if population_count <= 0 || sample_count <= 0 {
        return Ok(0.0);
    }
    if !oversampling_factor.is_finite() {
        return Err(PyValueError::new_err("non-finite oversampling factor"));
    }
    Ok(((oversampling_factor * sample_count as f64) / population_count as f64).min(1.0))
}

#[pyfunction]
pub(crate) fn finbert_confusion_cell(predicted_label: &str, gold_negative: &str) -> &'static str {
    if !matches!(gold_negative, "yes" | "no") {
        return "uncertain";
    }
    match (predicted_label == "negative", gold_negative == "yes") {
        (true, true) => "TP",
        (true, false) => "FP",
        (false, true) => "FN",
        (false, false) => "TN",
    }
}

#[pyfunction]
pub(crate) fn finbert_confusion_sample_id_rows(
    py: Python<'_>,
    rows: &Bound<'_, PyAny>,
) -> PyResult<Vec<PyObject>> {
    let mut out_rows: Vec<PyObject> = Vec::new();
    for (row_index, row) in rows.iter()?.enumerate() {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("FinBERT confusion sample row is not a dict"))?;
        let out = PyDict::new_bound(py);
        for (key, value) in dict.iter() {
            out.set_item(key, value)?;
        }
        let sample_order = row_index + 1;
        out.set_item("sample_order", sample_order as i64)?;
        out.set_item(
            "review_case_id",
            format!("finbert_review_{sample_order:06}"),
        )?;
        out_rows.push(out.into_py(py));
    }
    Ok(out_rows)
}

#[pyfunction]
pub(crate) fn finbert_confusion_sample_id_columns(
    py: Python<'_>,
    column_names: Vec<String>,
    column_values: &Bound<'_, PyAny>,
) -> PyResult<Vec<PyObject>> {
    let label = "FinBERT confusion sample ID rows";
    let (columns, row_count) =
        collect_pyobject_column_values(py, &column_names, column_values, label)?;
    let mut out_rows: Vec<PyObject> = Vec::with_capacity(row_count);
    for row_idx in 0..row_count {
        let out = PyDict::new_bound(py);
        for (column_idx, column_name) in column_names.iter().enumerate() {
            out.set_item(column_name, columns[column_idx][row_idx].bind(py))?;
        }
        let sample_order = row_idx + 1;
        out.set_item("sample_order", sample_order as i64)?;
        out.set_item(
            "review_case_id",
            format!("finbert_review_{sample_order:06}"),
        )?;
        out_rows.push(out.into_py(py));
    }
    Ok(out_rows)
}

#[pyfunction]
pub(crate) fn finbert_confusion_neighbor_target_rows(
    py: Python<'_>,
    rows: &Bound<'_, PyAny>,
) -> PyResult<Vec<PyObject>> {
    let mut out_rows: Vec<PyObject> = Vec::new();
    for row in rows.iter()? {
        let row = row?;
        let dict = row.downcast::<PyDict>().map_err(|_| {
            PyValueError::new_err("FinBERT confusion neighbor source row is not a dict")
        })?;
        let Some(sentence_index_value) = dict.get_item("sentence_index")? else {
            continue;
        };
        if sentence_index_value.is_none() {
            continue;
        }
        let Ok(current_index) = py_int_like_to_i64(&sentence_index_value) else {
            continue;
        };
        let Some(review_case_id) = dict.get_item("review_case_id")? else {
            return Err(PyValueError::new_err(
                "missing required key: review_case_id",
            ));
        };
        let Some(benchmark_row_id) = dict.get_item("benchmark_row_id")? else {
            return Err(PyValueError::new_err(
                "missing required key: benchmark_row_id",
            ));
        };

        let prev_row = PyDict::new_bound(py);
        prev_row.set_item("review_case_id", &review_case_id)?;
        prev_row.set_item("neighbor_kind", "prev_text")?;
        prev_row.set_item("benchmark_row_id", &benchmark_row_id)?;
        prev_row.set_item("sentence_index", current_index - 1)?;
        out_rows.push(prev_row.into_py(py));

        let next_row = PyDict::new_bound(py);
        next_row.set_item("review_case_id", review_case_id)?;
        next_row.set_item("neighbor_kind", "next_text")?;
        next_row.set_item("benchmark_row_id", benchmark_row_id)?;
        next_row.set_item("sentence_index", current_index + 1)?;
        out_rows.push(next_row.into_py(py));
    }
    Ok(out_rows)
}

#[pyfunction]
pub(crate) fn finbert_confusion_neighbor_target_columns(
    py: Python<'_>,
    column_names: Vec<String>,
    column_values: &Bound<'_, PyAny>,
) -> PyResult<Vec<PyObject>> {
    let label = "FinBERT confusion neighbor targets";
    let (columns, row_count) =
        collect_pyobject_column_values(py, &column_names, column_values, label)?;
    let column_index = column_index_by_name(&column_names);
    let review_case_idx = required_named_column_index(&column_index, label, "review_case_id")?;
    let benchmark_row_idx = required_named_column_index(&column_index, label, "benchmark_row_id")?;
    let sentence_idx = required_named_column_index(&column_index, label, "sentence_index")?;

    let mut out_rows: Vec<PyObject> = Vec::new();
    for row_idx in 0..row_count {
        let sentence_index_value = columns[sentence_idx][row_idx].bind(py);
        if sentence_index_value.is_none() {
            continue;
        }
        let Ok(current_index) = py_int_like_to_i64(sentence_index_value) else {
            continue;
        };
        let review_case_id = columns[review_case_idx][row_idx].bind(py);
        let benchmark_row_id = columns[benchmark_row_idx][row_idx].bind(py);

        let prev_row = PyDict::new_bound(py);
        prev_row.set_item("review_case_id", review_case_id)?;
        prev_row.set_item("neighbor_kind", "prev_text")?;
        prev_row.set_item("benchmark_row_id", benchmark_row_id)?;
        prev_row.set_item("sentence_index", current_index - 1)?;
        out_rows.push(prev_row.into_py(py));

        let next_row = PyDict::new_bound(py);
        next_row.set_item("review_case_id", review_case_id)?;
        next_row.set_item("neighbor_kind", "next_text")?;
        next_row.set_item("benchmark_row_id", benchmark_row_id)?;
        next_row.set_item("sentence_index", current_index + 1)?;
        out_rows.push(next_row.into_py(py));
    }
    Ok(out_rows)
}

#[pyfunction]
pub(crate) fn finbert_confusion_finalize_allocation_rows(
    py: Python<'_>,
    rows: &Bound<'_, PyAny>,
    total_population: i64,
    allocation_mode: &str,
) -> PyResult<Vec<PyObject>> {
    if total_population == 0 {
        return Err(PyValueError::new_err("total_population must be non-zero"));
    }
    let mut out_rows: Vec<PyObject> = Vec::new();
    for row in rows.iter()? {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("FinBERT confusion allocation row is not a dict"))?;
        let sample_count_value = dict
            .get_item("sample_count")?
            .ok_or_else(|| PyValueError::new_err("missing required key: sample_count"))?;
        let population_count_value = dict
            .get_item("population_count")?
            .ok_or_else(|| PyValueError::new_err("missing required key: population_count"))?;
        let sample_count = py_int_like_to_i64(&sample_count_value)?;
        let population_count = py_int_like_to_i64(&population_count_value)?;

        let out = PyDict::new_bound(py);
        for (key, value) in dict.iter() {
            out.set_item(key, value)?;
        }
        out.set_item("sample_count", sample_count)?;
        if sample_count == 0 {
            out.set_item("sample_weight", Option::<f64>::None)?;
        } else {
            out.set_item(
                "sample_weight",
                population_count as f64 / sample_count as f64,
            )?;
        }
        out.set_item(
            "population_fraction",
            population_count as f64 / total_population as f64,
        )?;
        out.set_item("allocation_mode", allocation_mode)?;
        out_rows.push(out.into_py(py));
    }
    Ok(out_rows)
}

#[pyfunction]
pub(crate) fn finbert_confusion_target_position_rows(
    py: Python<'_>,
    rows: &Bound<'_, PyAny>,
    seed: i64,
) -> PyResult<(Vec<PyObject>, Vec<PyObject>)> {
    let random_cls = py.import_bound("random")?.getattr("Random")?;
    let builtins = py.import_bound("builtins")?;
    let range_fn = builtins.getattr("range")?;
    let mut target_rows: Vec<PyObject> = Vec::new();
    let mut summary_rows: Vec<PyObject> = Vec::new();

    for row in rows.iter()? {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("FinBERT confusion allocation row is not a dict"))?;
        let stratum_value = dict
            .get_item("stratum_id")?
            .ok_or_else(|| PyValueError::new_err("missing required key: stratum_id"))?;
        let stratum_id = stratum_value.str()?.to_str()?.to_string();
        let population_count_value = dict
            .get_item("population_count")?
            .ok_or_else(|| PyValueError::new_err("missing required key: population_count"))?;
        let sample_count_value = dict
            .get_item("sample_count")?
            .ok_or_else(|| PyValueError::new_err("missing required key: sample_count"))?;
        let population_count = py_int_like_to_i64(&population_count_value)?;
        let sample_count = py_int_like_to_i64(&sample_count_value)?;
        if sample_count <= 0 {
            continue;
        }
        if sample_count > population_count {
            return Err(PyValueError::new_err(format!(
                "Cannot sample {sample_count} rows from stratum {stratum_id} with population {population_count}."
            )));
        }

        let stable_seed = sha256_first_u64_impl(&format!("{seed}::{stratum_id}"));
        let rng = random_cls.call1((stable_seed,))?;
        let population_range = range_fn.call1((population_count,))?;
        let raw_positions = rng
            .getattr("sample")?
            .call1((population_range, sample_count))?;
        let mut positions: Vec<i64> = raw_positions.extract()?;
        positions.sort_unstable();

        for (rank, position) in positions.iter().enumerate() {
            let out = PyDict::new_bound(py);
            out.set_item("stratum_id", &stratum_id)?;
            out.set_item("_target_ordinal", *position)?;
            out.set_item("_target_rank", rank as i64)?;
            target_rows.push(out.into_py(py));
        }

        let summary = PyDict::new_bound(py);
        summary.set_item("stratum_id", &stratum_id)?;
        summary.set_item("population_count", population_count)?;
        summary.set_item("sample_count", sample_count)?;
        summary.set_item("first_target_ordinal", positions.first().copied())?;
        summary.set_item("last_target_ordinal", positions.last().copied())?;
        summary_rows.push(summary.into_py(py));
    }

    Ok((target_rows, summary_rows))
}

#[pyfunction]
pub(crate) fn finbert_confusion_target_position_columns(
    py: Python<'_>,
    column_names: Vec<String>,
    column_values: &Bound<'_, PyAny>,
    seed: i64,
) -> PyResult<(Vec<PyObject>, Vec<PyObject>)> {
    let label = "FinBERT confusion target positions";
    let (columns, row_count) =
        collect_pyobject_column_values(py, &column_names, column_values, label)?;
    let column_index = column_index_by_name(&column_names);
    let stratum_idx = required_named_column_index(&column_index, label, "stratum_id")?;
    let population_idx = required_named_column_index(&column_index, label, "population_count")?;
    let sample_idx = required_named_column_index(&column_index, label, "sample_count")?;

    let random_cls = py.import_bound("random")?.getattr("Random")?;
    let builtins = py.import_bound("builtins")?;
    let range_fn = builtins.getattr("range")?;
    let mut target_rows: Vec<PyObject> = Vec::new();
    let mut summary_rows: Vec<PyObject> = Vec::new();

    for row_idx in 0..row_count {
        let stratum_value = columns[stratum_idx][row_idx].bind(py);
        if stratum_value.is_none() {
            return Err(PyValueError::new_err("null required key: stratum_id"));
        }
        let stratum_id = stratum_value.str()?.to_str()?.to_string();
        let population_count = py_int_like_to_i64(columns[population_idx][row_idx].bind(py))?;
        let sample_count = py_int_like_to_i64(columns[sample_idx][row_idx].bind(py))?;
        if sample_count <= 0 {
            continue;
        }
        if sample_count > population_count {
            return Err(PyValueError::new_err(format!(
                "Cannot sample {sample_count} rows from stratum {stratum_id} with population {population_count}."
            )));
        }

        let stable_seed = sha256_first_u64_impl(&format!("{seed}::{stratum_id}"));
        let rng = random_cls.call1((stable_seed,))?;
        let population_range = range_fn.call1((population_count,))?;
        let raw_positions = rng
            .getattr("sample")?
            .call1((population_range, sample_count))?;
        let mut positions: Vec<i64> = raw_positions.extract()?;
        positions.sort_unstable();

        for (rank, position) in positions.iter().enumerate() {
            let out = PyDict::new_bound(py);
            out.set_item("stratum_id", &stratum_id)?;
            out.set_item("_target_ordinal", *position)?;
            out.set_item("_target_rank", rank as i64)?;
            target_rows.push(out.into_py(py));
        }

        let summary = PyDict::new_bound(py);
        summary.set_item("stratum_id", &stratum_id)?;
        summary.set_item("population_count", population_count)?;
        summary.set_item("sample_count", sample_count)?;
        summary.set_item("first_target_ordinal", positions.first().copied())?;
        summary.set_item("last_target_ordinal", positions.last().copied())?;
        summary_rows.push(summary.into_py(py));
    }

    Ok((target_rows, summary_rows))
}

pub(crate) fn copy_required_dict_key(
    out: &Bound<'_, PyDict>,
    dict: &Bound<'_, PyDict>,
    key: &str,
) -> PyResult<()> {
    let Some(value) = dict.get_item(key)? else {
        return Err(PyValueError::new_err(format!(
            "missing required key: {key}"
        )));
    };
    out.set_item(key, value)?;
    Ok(())
}

pub(crate) fn copy_optional_dict_key(
    out: &Bound<'_, PyDict>,
    dict: &Bound<'_, PyDict>,
    key: &str,
) -> PyResult<()> {
    match dict.get_item(key)? {
        Some(value) => out.set_item(key, value)?,
        None => out.set_item(key, Option::<String>::None)?,
    }
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (rows, pass_id=None, recommended_model=None, recommended_reasoning_effort=None))]
pub(crate) fn finbert_confusion_labeling_input_records(
    py: Python<'_>,
    rows: &Bound<'_, PyAny>,
    pass_id: Option<&str>,
    recommended_model: Option<&str>,
    recommended_reasoning_effort: Option<&str>,
) -> PyResult<Vec<PyObject>> {
    let mut out_rows: Vec<PyObject> = Vec::new();
    let required_keys = [
        "review_case_id",
        "benchmark_sentence_id",
        "doc_id",
        "filing_year",
        "text_scope",
        "benchmark_item_code",
        "sentence_index",
        "sentence_text",
        "predicted_label",
        "negative_prob",
        "neutral_prob",
        "positive_prob",
        "probability_majority_bucket",
        "finbert_token_bucket_512",
    ];
    for row in rows.iter()? {
        let row = row?;
        let dict = row.downcast::<PyDict>().map_err(|_| {
            PyValueError::new_err("FinBERT confusion labeling source row is not a dict")
        })?;
        let out = PyDict::new_bound(py);
        for key in required_keys {
            copy_required_dict_key(&out, dict, key)?;
        }
        copy_optional_dict_key(&out, dict, "prev_text")?;
        copy_optional_dict_key(&out, dict, "next_text")?;
        if let Some(pass_id) = pass_id {
            out.set_item("labeling_pass_id", pass_id)?;
        }
        if let Some(recommended_model) = recommended_model {
            out.set_item("recommended_model", recommended_model)?;
        }
        if let Some(recommended_reasoning_effort) = recommended_reasoning_effort {
            out.set_item("recommended_reasoning_effort", recommended_reasoning_effort)?;
        }
        out_rows.push(out.into_py(py));
    }
    Ok(out_rows)
}

#[pyfunction]
pub(crate) fn finbert_confusion_chunk_row_indices(
    row_count: usize,
    chunk_count: usize,
) -> PyResult<Vec<Vec<usize>>> {
    if chunk_count == 0 {
        return Err(PyValueError::new_err("chunk_count must be positive"));
    }
    let mut chunks: Vec<Vec<usize>> = vec![Vec::new(); chunk_count];
    for row_index in 0..row_count {
        chunks[row_index % chunk_count].push(row_index);
    }
    Ok(chunks)
}

#[pyfunction]
pub(crate) fn finbert_confusion_csv_safe_rows(
    py: Python<'_>,
    rows: &Bound<'_, PyAny>,
) -> PyResult<Vec<PyObject>> {
    let json_dumps = py.import_bound("json")?.getattr("dumps")?;
    let dumps_kwargs = PyDict::new_bound(py);
    dumps_kwargs.set_item("sort_keys", true)?;

    let mut out_rows: Vec<PyObject> = Vec::new();
    for row in rows.iter()? {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("FinBERT confusion CSV source row is not a dict"))?;
        let out = PyDict::new_bound(py);
        for (key, value) in dict.iter() {
            if value.is_instance_of::<PyList>() || value.is_instance_of::<PyDict>() {
                let rendered = json_dumps.call((value,), Some(&dumps_kwargs))?;
                out.set_item(key, rendered)?;
            } else {
                out.set_item(key, value)?;
            }
        }
        out_rows.push(out.into_py(py));
    }
    Ok(out_rows)
}

pub(crate) fn normalize_binary_label_bound(value: Option<Bound<'_, PyAny>>) -> PyResult<String> {
    let Some(value) = value else {
        return Ok(String::new());
    };
    if value.is_none() {
        return Ok(String::new());
    }
    let rendered = value.str()?;
    Ok(normalize_binary_label_impl(rendered.to_str()?).to_string())
}

pub(crate) fn final_gold_negative_from_record(
    record: Option<&Bound<'_, PyDict>>,
) -> PyResult<(String, String)> {
    let Some(record) = record else {
        return Ok((String::new(), "missing".to_string()));
    };
    for key in [
        "human_gold_negative",
        "final_gold_negative",
        "gold_negative",
    ] {
        let label = normalize_binary_label_bound(record.get_item(key)?)?;
        if !label.is_empty() {
            return Ok((label, key.to_string()));
        }
    }
    let a_label = normalize_binary_label_bound(record.get_item("llm_a_gold_negative")?)?;
    let b_label = normalize_binary_label_bound(record.get_item("llm_b_gold_negative")?)?;
    if !a_label.is_empty() && a_label == b_label && a_label != "uncertain" {
        return Ok((a_label, "llm_consensus".to_string()));
    }
    Ok((String::new(), "missing".to_string()))
}

#[pyfunction]
pub(crate) fn finbert_confusion_reviewed_case_rows(
    py: Python<'_>,
    sample_rows: &Bound<'_, PyAny>,
    labels_by_case_id: &Bound<'_, PyAny>,
) -> PyResult<(Vec<PyObject>, Vec<String>)> {
    let labels = labels_by_case_id
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("labels_by_case_id is not a dict"))?;
    let mut out_rows: Vec<PyObject> = Vec::new();
    let mut missing_review_case_ids: Vec<String> = Vec::new();

    for row in sample_rows.iter()? {
        let row = row?;
        let row_dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("FinBERT confusion sample row is not a dict"))?;
        let Some(review_case_value) = row_dict.get_item("review_case_id")? else {
            return Err(PyValueError::new_err(
                "missing required key: review_case_id",
            ));
        };
        let review_case_id = review_case_value.str()?.to_str()?.to_string();
        let record_value = labels.get_item(review_case_id.as_str())?;
        let record_dict = match record_value.as_ref() {
            Some(value) if !value.is_none() => Some(value.downcast::<PyDict>().map_err(|_| {
                PyValueError::new_err("FinBERT confusion review label record is not a dict")
            })?),
            _ => None,
        };

        let (mut gold_negative, label_source) = final_gold_negative_from_record(record_dict)?;
        if gold_negative.is_empty() {
            missing_review_case_ids.push(review_case_id);
            gold_negative = "uncertain".to_string();
        }

        let out = PyDict::new_bound(py);
        for (key, value) in row_dict.iter() {
            out.set_item(key, value)?;
        }
        if let Some(record_dict) = record_dict {
            for (key, value) in record_dict.iter() {
                if !row_dict.contains(&key)? {
                    let review_key = format!("review_{}", key.str()?.to_str()?);
                    out.set_item(review_key, value)?;
                }
            }
        }
        let Some(predicted_label) = row_dict.get_item("predicted_label")? else {
            return Err(PyValueError::new_err(
                "missing required key: predicted_label",
            ));
        };
        let predicted_label = predicted_label.str()?;
        out.set_item("gold_negative_final", gold_negative.as_str())?;
        out.set_item("gold_negative_source", label_source.as_str())?;
        out.set_item(
            "confusion_cell",
            finbert_confusion_cell(predicted_label.to_str()?, gold_negative.as_str()),
        )?;
        out_rows.push(out.into_py(py));
    }

    Ok((out_rows, missing_review_case_ids))
}

#[pyfunction]
pub(crate) fn finbert_confusion_reviewed_case_columns(
    py: Python<'_>,
    column_names: Vec<String>,
    column_values: &Bound<'_, PyAny>,
    labels_by_case_id: &Bound<'_, PyAny>,
) -> PyResult<(Vec<PyObject>, Vec<String>)> {
    let label = "FinBERT confusion reviewed cases";
    let (columns, row_count) =
        collect_pyobject_column_values(py, &column_names, column_values, label)?;
    let column_index = column_index_by_name(&column_names);
    let review_case_idx = required_named_column_index(&column_index, label, "review_case_id")?;
    let predicted_idx = required_named_column_index(&column_index, label, "predicted_label")?;
    let labels = labels_by_case_id
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("labels_by_case_id is not a dict"))?;
    let mut out_rows: Vec<PyObject> = Vec::new();
    let mut missing_review_case_ids: Vec<String> = Vec::new();

    for row_idx in 0..row_count {
        let review_case_value = columns[review_case_idx][row_idx].bind(py);
        let review_case_id = review_case_value.str()?.to_str()?.to_string();
        let record_value = labels.get_item(review_case_id.as_str())?;
        let record_dict = match record_value.as_ref() {
            Some(value) if !value.is_none() => Some(value.downcast::<PyDict>().map_err(|_| {
                PyValueError::new_err("FinBERT confusion review label record is not a dict")
            })?),
            _ => None,
        };

        let (mut gold_negative, label_source) = final_gold_negative_from_record(record_dict)?;
        if gold_negative.is_empty() {
            missing_review_case_ids.push(review_case_id);
            gold_negative = "uncertain".to_string();
        }

        let out = PyDict::new_bound(py);
        for (column_idx, column_name) in column_names.iter().enumerate() {
            out.set_item(column_name.as_str(), columns[column_idx][row_idx].bind(py))?;
        }
        if let Some(record_dict) = record_dict {
            for (key, value) in record_dict.iter() {
                let key_text = key.str()?;
                let key_name = key_text.to_str()?;
                if !column_index.contains_key(key_name) {
                    out.set_item(format!("review_{key_name}"), value)?;
                }
            }
        }
        let predicted_label = columns[predicted_idx][row_idx].bind(py).str()?;
        out.set_item("gold_negative_final", gold_negative.as_str())?;
        out.set_item("gold_negative_source", label_source.as_str())?;
        out.set_item(
            "confusion_cell",
            finbert_confusion_cell(predicted_label.to_str()?, gold_negative.as_str()),
        )?;
        out_rows.push(out.into_py(py));
    }

    Ok((out_rows, missing_review_case_ids))
}

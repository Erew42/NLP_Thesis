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

#[derive(Clone)]
pub(crate) struct ConfusionAllocationRow {
    original_index: usize,
    stratum_id: String,
    population_count: i64,
}

#[derive(Clone)]
pub(crate) struct ProportionalAllocationRow {
    stratum_id: String,
    population_count: i64,
    exact_quota: f64,
    remainder: f64,
    sample_count: i64,
}

#[pyfunction]
pub(crate) fn finbert_confusion_balanced_sample_count_pairs(
    rows: &Bound<'_, PyAny>,
    target_sample_size: i64,
) -> PyResult<Vec<(usize, i64)>> {
    let mut active_rows: Vec<ConfusionAllocationRow> = Vec::new();
    for (original_index, row) in rows.iter()?.enumerate() {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("confusion allocation row is not a dict"))?;
        let population_count = dict_python_or_zero_i64(dict, "population_count")?;
        if population_count <= 0 {
            continue;
        }
        active_rows.push(ConfusionAllocationRow {
            original_index,
            stratum_id: dict_required_string(dict, "stratum_id")?,
            population_count,
        });
    }
    if active_rows.is_empty() {
        return Err(PyValueError::new_err(
            "No eligible rows found for the requested review universe.",
        ));
    }
    active_rows.sort_by(|left, right| left.stratum_id.cmp(&right.stratum_id));
    let active_count = i64::try_from(active_rows.len())
        .map_err(|_| PyValueError::new_err("too many allocation rows"))?;
    let base = target_sample_size.div_euclid(active_count);
    let remainder = target_sample_size.rem_euclid(active_count);
    let mut allocation_pairs: Vec<(usize, i64)> = Vec::with_capacity(active_rows.len());
    for (active_index, row) in active_rows.iter().enumerate() {
        let bonus = if i64::try_from(active_index).unwrap_or(i64::MAX) < remainder {
            1
        } else {
            0
        };
        allocation_pairs.push((row.original_index, (base + bonus).min(row.population_count)));
    }
    let mut allocated: i64 = allocation_pairs
        .iter()
        .map(|(_, sample_count)| *sample_count)
        .sum();
    while allocated < target_sample_size {
        let mut progressed = false;
        for (pair_index, (_, sample_count)) in allocation_pairs.iter_mut().enumerate() {
            if allocated >= target_sample_size {
                break;
            }
            let population_count = active_rows[pair_index].population_count;
            if *sample_count < population_count {
                *sample_count += 1;
                allocated += 1;
                progressed = true;
            }
        }
        if !progressed {
            break;
        }
    }
    Ok(allocation_pairs)
}

#[pyfunction]
pub(crate) fn finbert_confusion_proportional_sample_counts(
    rows: &Bound<'_, PyAny>,
    target_sample_size: i64,
    total_population: i64,
) -> PyResult<Vec<i64>> {
    if total_population <= 0 {
        return Err(PyValueError::new_err("total_population must be positive"));
    }
    let mut quota_rows: Vec<ProportionalAllocationRow> = Vec::new();
    for row in rows.iter()? {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("confusion allocation row is not a dict"))?;
        let population_count = dict_python_or_zero_i64(dict, "population_count")?;
        quota_rows.push(ProportionalAllocationRow {
            stratum_id: dict_required_string(dict, "stratum_id")?,
            population_count,
            exact_quota: 0.0,
            remainder: 0.0,
            sample_count: 0,
        });
    }
    if target_sample_size == total_population {
        return Ok(quota_rows.iter().map(|row| row.population_count).collect());
    }
    let mut running_floor = 0_i64;
    for row in quota_rows.iter_mut() {
        let exact =
            target_sample_size as f64 * row.population_count as f64 / total_population as f64;
        let floor_count = exact.floor() as i64;
        row.exact_quota = exact;
        row.remainder = exact - floor_count as f64;
        row.sample_count = floor_count.min(row.population_count);
        running_floor += row.sample_count;
    }

    let mut remaining = target_sample_size - running_floor;
    let mut remainder_order: Vec<usize> = (0..quota_rows.len()).collect();
    remainder_order.sort_by(|left, right| {
        quota_rows[*right]
            .remainder
            .partial_cmp(&quota_rows[*left].remainder)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                quota_rows[*left]
                    .stratum_id
                    .cmp(&quota_rows[*right].stratum_id)
            })
    });
    for row_index in remainder_order {
        if remaining <= 0 {
            break;
        }
        let row = &mut quota_rows[row_index];
        if row.sample_count < row.population_count {
            row.sample_count += 1;
            remaining -= 1;
        }
    }

    if target_sample_size >= i64::try_from(quota_rows.len()).unwrap_or(i64::MAX) {
        let zero_indices: Vec<usize> = quota_rows
            .iter()
            .enumerate()
            .filter_map(|(index, row)| {
                if row.sample_count == 0 && row.population_count > 0 {
                    Some(index)
                } else {
                    None
                }
            })
            .collect();
        for zero_index in zero_indices {
            let mut donor_indices: Vec<usize> = quota_rows
                .iter()
                .enumerate()
                .filter_map(|(index, row)| {
                    if row.sample_count > 1 && row.exact_quota < row.sample_count as f64 {
                        Some(index)
                    } else {
                        None
                    }
                })
                .collect();
            if donor_indices.is_empty() {
                donor_indices = quota_rows
                    .iter()
                    .enumerate()
                    .filter_map(|(index, row)| {
                        if row.sample_count > 1 {
                            Some(index)
                        } else {
                            None
                        }
                    })
                    .collect();
            }
            if donor_indices.is_empty() {
                break;
            }
            donor_indices.sort_by(|left, right| {
                let left_diff =
                    quota_rows[*left].sample_count as f64 - quota_rows[*left].exact_quota;
                let right_diff =
                    quota_rows[*right].sample_count as f64 - quota_rows[*right].exact_quota;
                right_diff
                    .partial_cmp(&left_diff)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| {
                        quota_rows[*right]
                            .sample_count
                            .cmp(&quota_rows[*left].sample_count)
                    })
                    .then_with(|| left.cmp(right))
            });
            let donor_index = donor_indices[0];
            quota_rows[donor_index].sample_count -= 1;
            quota_rows[zero_index].sample_count = 1;
        }
    }

    let actual_total: i64 = quota_rows.iter().map(|row| row.sample_count).sum();
    if actual_total != target_sample_size {
        return Err(PyValueError::new_err(format!(
            "Internal allocation error: allocated {actual_total}, expected {target_sample_size}."
        )));
    }
    Ok(quota_rows.iter().map(|row| row.sample_count).collect())
}

pub(crate) fn safe_divide_option(numerator: f64, denominator: f64) -> Option<f64> {
    if denominator == 0.0 {
        None
    } else {
        Some(numerator / denominator)
    }
}

#[pyfunction]
pub(crate) fn finbert_confusion_metric_payload(
    tp: f64,
    fp: f64,
    fn_count: f64,
    tn: f64,
) -> HashMap<String, Option<f64>> {
    let total = tp + fp + fn_count + tn;
    let mut payload = HashMap::with_capacity(7);
    payload.insert("accuracy".to_string(), safe_divide_option(tp + tn, total));
    payload.insert("precision".to_string(), safe_divide_option(tp, tp + fp));
    payload.insert("recall".to_string(), safe_divide_option(tp, tp + fn_count));
    payload.insert("specificity".to_string(), safe_divide_option(tn, tn + fp));
    payload.insert(
        "false_positive_rate".to_string(),
        safe_divide_option(fp, fp + tn),
    );
    payload.insert(
        "false_negative_rate".to_string(),
        safe_divide_option(fn_count, fn_count + tp),
    );
    payload.insert("resolved_count".to_string(), Some(total));
    payload
}

pub(crate) fn confusion_review_weight_value(
    value: Option<&Bound<'_, PyAny>>,
    weighted: bool,
) -> PyResult<f64> {
    if !weighted {
        return Ok(1.0);
    }
    let Some(value) = value else {
        return Ok(1.0);
    };
    if value.is_none() || !value.is_truthy()? {
        return Ok(1.0);
    }
    if value.is_instance_of::<PyBool>() {
        return Ok(if value.extract::<bool>()? { 1.0 } else { 0.0 });
    }
    if value.is_instance_of::<PyInt>() || value.is_instance_of::<PyFloat>() {
        return value.extract::<f64>();
    }
    let rendered = value.str()?;
    rendered
        .to_str()?
        .trim()
        .parse::<f64>()
        .map_err(|_| PyValueError::new_err("sample_weight is not numeric"))
}

pub(crate) fn confusion_review_row_weight(
    dict: &Bound<'_, PyDict>,
    weighted: bool,
) -> PyResult<f64> {
    let value = dict.get_item("sample_weight")?;
    confusion_review_weight_value(value.as_ref(), weighted)
}

pub(crate) fn init_confusion_counts() -> BTreeMap<String, f64> {
    let mut counts = BTreeMap::new();
    for cell in ["TP", "FP", "FN", "TN", "uncertain"] {
        counts.insert(cell.to_string(), 0.0);
    }
    counts
}

#[pyfunction]
pub(crate) fn finbert_confusion_counts_by_cell(
    rows: &Bound<'_, PyAny>,
    weighted: bool,
) -> PyResult<BTreeMap<String, f64>> {
    let mut counts = init_confusion_counts();
    for row in rows.iter()? {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("confusion row is not a dict"))?;
        let cell = dict_required_string(dict, "confusion_cell")?;
        let weight = confusion_review_row_weight(dict, weighted)?;
        *counts.entry(cell).or_insert(0.0) += weight;
    }
    Ok(counts)
}

#[pyfunction]
pub(crate) fn finbert_confusion_uncertain_metric_bounds(
    rows: &Bound<'_, PyAny>,
    weighted: bool,
) -> PyResult<BTreeMap<String, BTreeMap<String, Option<f64>>>> {
    let mut resolved = init_confusion_counts();
    let mut uncertain_pred_pos = 0.0_f64;
    let mut uncertain_pred_neg = 0.0_f64;
    for row in rows.iter()? {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("confusion row is not a dict"))?;
        let cell = dict_required_string(dict, "confusion_cell")?;
        let weight = confusion_review_row_weight(dict, weighted)?;
        if cell == "uncertain" {
            let predicted_label = dict_required_string(dict, "predicted_label")?;
            if predicted_label == "negative" {
                uncertain_pred_pos += weight;
            } else {
                uncertain_pred_neg += weight;
            }
        } else {
            *resolved.entry(cell).or_insert(0.0) += weight;
        }
    }

    let tp = *resolved.get("TP").unwrap_or(&0.0);
    let fp = *resolved.get("FP").unwrap_or(&0.0);
    let fn_count = *resolved.get("FN").unwrap_or(&0.0);
    let tn = *resolved.get("TN").unwrap_or(&0.0);
    let total = tp + fp + fn_count + tn + uncertain_pred_pos + uncertain_pred_neg;
    let mut payload: BTreeMap<String, BTreeMap<String, Option<f64>>> = BTreeMap::new();

    let mut precision = BTreeMap::new();
    precision.insert(
        "lower".to_string(),
        safe_divide_option(tp, tp + fp + uncertain_pred_pos),
    );
    precision.insert(
        "upper".to_string(),
        safe_divide_option(tp + uncertain_pred_pos, tp + uncertain_pred_pos + fp),
    );
    payload.insert("precision".to_string(), precision);

    let mut recall = BTreeMap::new();
    recall.insert(
        "lower".to_string(),
        safe_divide_option(tp, tp + fn_count + uncertain_pred_neg),
    );
    recall.insert(
        "upper".to_string(),
        safe_divide_option(tp + uncertain_pred_pos, tp + uncertain_pred_pos + fn_count),
    );
    payload.insert("recall".to_string(), recall);

    let mut accuracy = BTreeMap::new();
    accuracy.insert("lower".to_string(), safe_divide_option(tp + tn, total));
    accuracy.insert(
        "upper".to_string(),
        safe_divide_option(tp + tn + uncertain_pred_pos + uncertain_pred_neg, total),
    );
    payload.insert("accuracy".to_string(), accuracy);

    Ok(payload)
}

#[derive(Default)]
pub(crate) struct ConfusionBucketMetricAccum {
    row_count: i64,
    tp: f64,
    fp: f64,
    fn_count: f64,
    tn: f64,
    uncertain: f64,
}

impl ConfusionBucketMetricAccum {
    fn add(&mut self, cell: &str, weight: f64) {
        self.row_count += 1;
        match cell {
            "TP" => self.tp += weight,
            "FP" => self.fp += weight,
            "FN" => self.fn_count += weight,
            "TN" => self.tn += weight,
            "uncertain" => self.uncertain += weight,
            _ => {}
        }
    }
}

#[pyfunction]
pub(crate) fn finbert_confusion_bucket_metric_rows(
    rows: &Bound<'_, PyAny>,
) -> PyResult<
    Vec<(
        String,
        i64,
        f64,
        f64,
        f64,
        f64,
        f64,
        Option<f64>,
        Option<f64>,
        Option<f64>,
    )>,
> {
    let mut by_bucket: BTreeMap<String, ConfusionBucketMetricAccum> = BTreeMap::new();
    for row in rows.iter()? {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("confusion row is not a dict"))?;
        let Some(bucket) = dict_normalized_string(dict, "probability_majority_bucket")? else {
            continue;
        };
        let cell = dict_required_string(dict, "confusion_cell")?;
        let weight = confusion_review_row_weight(dict, true)?;
        by_bucket.entry(bucket).or_default().add(&cell, weight);
    }

    let mut out = Vec::with_capacity(by_bucket.len());
    for (bucket, acc) in by_bucket {
        out.push((
            bucket,
            acc.row_count,
            acc.tp,
            acc.fp,
            acc.fn_count,
            acc.tn,
            acc.uncertain,
            safe_divide_option(acc.tp, acc.tp + acc.fp),
            safe_divide_option(acc.tp, acc.tp + acc.fn_count),
            safe_divide_option(acc.tn, acc.tn + acc.fp),
        ));
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (bucket_values, confusion_cell_values, sample_weight_values=None))]
pub(crate) fn finbert_confusion_bucket_metric_columns(
    bucket_values: &Bound<'_, PyAny>,
    confusion_cell_values: &Bound<'_, PyAny>,
    sample_weight_values: Option<&Bound<'_, PyAny>>,
) -> PyResult<
    Vec<(
        String,
        i64,
        f64,
        f64,
        f64,
        f64,
        f64,
        Option<f64>,
        Option<f64>,
        Option<f64>,
    )>,
> {
    let mut by_bucket: BTreeMap<String, ConfusionBucketMetricAccum> = BTreeMap::new();
    let mut bucket_iter = bucket_values.iter()?;
    let mut cell_iter = confusion_cell_values.iter()?;
    let mut weight_iter = match sample_weight_values {
        Some(values) if !values.is_none() => Some(values.iter()?),
        _ => None,
    };

    loop {
        let bucket_next = bucket_iter.next();
        let cell_next = cell_iter.next();
        match (bucket_next, cell_next) {
            (None, None) => break,
            (Some(_), None) | (None, Some(_)) => return Err(PyValueError::new_err(
                "probability_majority_bucket and confusion_cell columns must have the same length",
            )),
            (Some(bucket), Some(cell)) => {
                let bucket = bucket?;
                let cell = cell?;
                let weight_value = if let Some(iter) = weight_iter.as_mut() {
                    match iter.next() {
                        Some(value) => Some(value?),
                        None => {
                            return Err(PyValueError::new_err(
                                "sample_weight column must match bucket column length",
                            ))
                        }
                    }
                } else {
                    None
                };
                let Some(bucket) = normalize_lookup_text_any_impl(Some(&bucket))? else {
                    continue;
                };
                if cell.is_none() {
                    return Err(PyValueError::new_err("null required key: confusion_cell"));
                }
                let cell = cell.str()?.to_str()?.to_string();
                let weight = confusion_review_weight_value(weight_value.as_ref(), true)?;
                by_bucket.entry(bucket).or_default().add(&cell, weight);
            }
        }
    }
    if let Some(iter) = weight_iter.as_mut() {
        if let Some(extra) = iter.next() {
            extra?;
            return Err(PyValueError::new_err(
                "sample_weight column must match bucket column length",
            ));
        }
    }

    let mut out = Vec::with_capacity(by_bucket.len());
    for (bucket, acc) in by_bucket {
        out.push((
            bucket,
            acc.row_count,
            acc.tp,
            acc.fp,
            acc.fn_count,
            acc.tn,
            acc.uncertain,
            safe_divide_option(acc.tp, acc.tp + acc.fp),
            safe_divide_option(acc.tp, acc.tp + acc.fn_count),
            safe_divide_option(acc.tn, acc.tn + acc.fp),
        ));
    }
    Ok(out)
}

pub(crate) fn confusion_review_optional_repr(
    dict: &Bound<'_, PyDict>,
    key: &str,
) -> PyResult<String> {
    let Some(value) = dict.get_item(key)? else {
        return Ok("None".to_string());
    };
    if value.is_none() {
        return Ok("None".to_string());
    }
    Ok(value.str()?.to_str()?.to_string())
}

pub(crate) fn confusion_review_compact_sentence(dict: &Bound<'_, PyDict>) -> PyResult<String> {
    let raw = match dict.get_item("sentence_text")? {
        Some(value) if !value.is_none() => value.str()?.to_str()?.to_string(),
        _ => String::new(),
    };
    let compact = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    if compact.chars().count() > 260 {
        let truncated: String = compact.chars().take(257).collect();
        Ok(format!("{truncated}..."))
    } else {
        Ok(compact)
    }
}

#[pyfunction]
pub(crate) fn finbert_confusion_examples_by_cell_markdown(
    rows: &Bound<'_, PyAny>,
    per_cell: usize,
) -> PyResult<String> {
    let mut dicts = Vec::new();
    for row in rows.iter()? {
        let row = row?;
        dicts.push(
            row.downcast::<PyDict>()
                .map_err(|_| PyValueError::new_err("confusion example row is not a dict"))?
                .to_owned(),
        );
    }

    let cells = ["TP", "FP", "FN", "TN", "uncertain"];
    let mut lines = vec!["# Examples by Confusion Cell".to_string(), String::new()];
    for cell in cells {
        lines.push(format!("## {cell}"));
        lines.push(String::new());
        let mut emitted = 0_usize;
        for row in dicts.iter() {
            if dict_required_string(row, "confusion_cell")? != cell {
                continue;
            }
            if emitted >= per_cell {
                break;
            }
            let sentence = confusion_review_compact_sentence(row)?;
            lines.push(format!(
                "- `{}` `{}` pred={} gold={}: {}",
                dict_required_string(row, "review_case_id")?,
                dict_required_string(row, "text_scope")?,
                dict_required_string(row, "predicted_label")?,
                confusion_review_optional_repr(row, "gold_negative_final")?,
                sentence
            ));
            emitted += 1;
        }
        if emitted == 0 {
            lines.push("No rows.".to_string());
        }
        lines.push(String::new());
    }
    Ok(lines.join("\n"))
}

pub(crate) fn erfc_approx(value: f64) -> f64 {
    let z = value.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let polynomial = -z * z - 1.26551223
        + t * (1.00002368
            + t * (0.37409196
                + t * (0.09678418
                    + t * (-0.18628806
                        + t * (0.27886807
                            + t * (-1.13520398
                                + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))));
    let result = t * polynomial.exp();
    if value >= 0.0 {
        result
    } else {
        2.0 - result
    }
}

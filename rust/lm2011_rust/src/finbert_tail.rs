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

pub(crate) fn lm2011_extension_text_scope_impl(value: &str) -> String {
    let raw = value.trim().to_lowercase().replace('-', "_");
    match raw.as_str() {
        "full_10k" | "full_10_k" | "10k" | "10_k" | "full_filing" => "full_10k".to_string(),
        "items_1a_7_combined"
        | "items_1a_7_concat"
        | "item_1a_item_7_combined"
        | "item_1a_item_7_concat"
        | "item_1a_7_combined"
        | "item_1a_7_concat" => "items_1a_7_combined".to_string(),
        "7" | "item_7" | "mda_item_7" | "item_7_mda" => "item_7_mda".to_string(),
        "1a" | "item_1a" | "item_1a_risk_factors" => "item_1a_risk_factors".to_string(),
        "1" | "item_1" | "item_1_business" => "item_1_business".to_string(),
        "items_1_1a_7_concat" | "item_1_item_1a_item_7_concat" => "items_1_1a_7_concat".to_string(),
        _ => raw,
    }
}

#[pyfunction]
pub(crate) fn lm2011_extension_text_scope_value(value: &str) -> String {
    lm2011_extension_text_scope_impl(value)
}

pub(crate) fn finbert_tail_text_scope_impl(value: &str) -> String {
    let raw = value.trim().to_lowercase().replace('-', "_");
    match raw.as_str() {
        "7" | "item_7" | "mda_item_7" | "item_7_mda" => "item_7_mda".to_string(),
        "1a" | "item_1a" | "item_1a_risk_factors" => "item_1a_risk_factors".to_string(),
        "1" | "item_1" | "item_1_business" => "item_1_business".to_string(),
        "items_1_1a_7_concat" | "item_1_item_1a_item_7_concat" => "items_1_1a_7_concat".to_string(),
        _ => raw,
    }
}

#[pyfunction]
pub(crate) fn finbert_tail_text_scope_value(value: &str) -> String {
    finbert_tail_text_scope_impl(value)
}

#[derive(Clone, Eq, PartialEq)]
pub(crate) struct FinbertTailGroupKey {
    doc_id: Option<String>,
    filing_date_key: Option<String>,
    text_scope: Option<String>,
    cleaning_policy_id: Option<String>,
    model_name: Option<String>,
    model_version: Option<String>,
    segment_policy_id: Option<String>,
}

pub(crate) struct FinbertTailInputRow {
    key: FinbertTailGroupKey,
    filing_date: PyObject,
    negative_prob: Option<f64>,
    sentence_index: Option<i64>,
    benchmark_sentence_id: Option<String>,
    token_weight: f64,
}

pub(crate) fn cmp_option_string_null_last(
    left: &Option<String>,
    right: &Option<String>,
) -> Ordering {
    match (left, right) {
        (Some(left), Some(right)) => left.cmp(right),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

pub(crate) fn cmp_option_i64_null_last(left: &Option<i64>, right: &Option<i64>) -> Ordering {
    match (left, right) {
        (Some(left), Some(right)) => left.cmp(right),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

pub(crate) fn cmp_option_f64_desc_null_last(left: &Option<f64>, right: &Option<f64>) -> Ordering {
    match (left, right) {
        (Some(left), Some(right)) => right.partial_cmp(left).unwrap_or(Ordering::Equal),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

pub(crate) fn finbert_tail_group_key_cmp(
    left: &FinbertTailGroupKey,
    right: &FinbertTailGroupKey,
) -> Ordering {
    cmp_option_string_null_last(&left.doc_id, &right.doc_id)
        .then_with(|| cmp_option_string_null_last(&left.filing_date_key, &right.filing_date_key))
        .then_with(|| cmp_option_string_null_last(&left.text_scope, &right.text_scope))
        .then_with(|| {
            cmp_option_string_null_last(&left.cleaning_policy_id, &right.cleaning_policy_id)
        })
        .then_with(|| cmp_option_string_null_last(&left.model_name, &right.model_name))
        .then_with(|| cmp_option_string_null_last(&left.model_version, &right.model_version))
        .then_with(|| {
            cmp_option_string_null_last(&left.segment_policy_id, &right.segment_policy_id)
        })
}

pub(crate) fn finbert_tail_input_row_cmp(
    left: &FinbertTailInputRow,
    right: &FinbertTailInputRow,
) -> Ordering {
    finbert_tail_group_key_cmp(&left.key, &right.key)
        .then_with(|| cmp_option_f64_desc_null_last(&left.negative_prob, &right.negative_prob))
        .then_with(|| cmp_option_i64_null_last(&left.sentence_index, &right.sentence_index))
        .then_with(|| {
            cmp_option_string_null_last(&left.benchmark_sentence_id, &right.benchmark_sentence_id)
        })
}

pub(crate) fn dict_optional_string_for_tail(
    dict: &Bound<'_, PyDict>,
    key: &str,
) -> PyResult<Option<String>> {
    dict_raw_string(dict, key)
}

pub(crate) fn dict_optional_i64_for_tail(
    dict: &Bound<'_, PyDict>,
    key: &str,
) -> PyResult<Option<i64>> {
    let Some(value) = dict.get_item(key)? else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    match py_int_like_to_i64(&value) {
        Ok(value) => Ok(Some(value)),
        Err(_) => Ok(None),
    }
}

pub(crate) fn dict_optional_f64_for_tail(
    dict: &Bound<'_, PyDict>,
    key: &str,
) -> PyResult<Option<f64>> {
    let Some(value) = dict.get_item(key)? else {
        return Ok(None);
    };
    py_float_like_to_finite_option(&value)
}

pub(crate) fn dict_contains_key(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<bool> {
    Ok(dict.get_item(key)?.is_some())
}

pub(crate) fn finbert_tail_row_text_scope(dict: &Bound<'_, PyDict>) -> PyResult<Option<String>> {
    for key in ["text_scope", "benchmark_item_code", "item_id"] {
        if !dict_contains_key(dict, key)? {
            continue;
        }
        let Some(value) = dict_optional_string_for_tail(dict, key)? else {
            return Ok(None);
        };
        return Ok(Some(lm2011_extension_text_scope_impl(&value)));
    }
    Err(PyValueError::new_err(
        "sentence_scores must contain text_scope, benchmark_item_code, or item_id for tail aggregation.",
    ))
}

pub(crate) fn finbert_tail_safe_divide(numerator: f64, denominator: f64) -> Option<f64> {
    if denominator > 0.0 {
        Some(numerator / denominator)
    } else {
        None
    }
}

pub(crate) fn finbert_tail_top_fraction_cutoff(sentence_count: usize, fraction: f64) -> usize {
    if sentence_count <= 1 {
        1
    } else {
        ((sentence_count as f64) * fraction).ceil() as usize
    }
}

pub(crate) fn finbert_tail_weighted_mean_for_ranks(
    group: &[FinbertTailInputRow],
    max_rank: usize,
) -> Option<f64> {
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for (idx, row) in group.iter().enumerate() {
        if idx + 1 > max_rank {
            continue;
        }
        denominator += row.token_weight;
        if let Some(value) = row.negative_prob {
            numerator += value * row.token_weight;
        }
    }
    finbert_tail_safe_divide(numerator, denominator)
}

#[pyfunction]
pub(crate) fn finbert_tail_doc_surface_rows(
    py: Python<'_>,
    rows: &Bound<'_, PyAny>,
    text_scopes: Vec<String>,
) -> PyResult<Vec<PyObject>> {
    let mut normalized_scopes: Vec<String> = Vec::new();
    let mut seen_scopes: HashSet<String> = HashSet::new();
    for scope in text_scopes {
        let normalized = finbert_tail_text_scope_impl(&scope);
        if seen_scopes.insert(normalized.clone()) {
            normalized_scopes.push(normalized);
        }
    }
    if normalized_scopes.is_empty() {
        return Err(PyValueError::new_err(
            "text_scopes must be non-empty for FinBERT tail aggregation.",
        ));
    }
    let normalized_scope_set: HashSet<String> = normalized_scopes.into_iter().collect();

    let mut parsed_rows: Vec<FinbertTailInputRow> = Vec::new();
    for row in rows.iter()? {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("tail surface row is not a dict"))?;
        let Some(text_scope) = finbert_tail_row_text_scope(dict)? else {
            continue;
        };
        if !normalized_scope_set.contains(&text_scope) {
            continue;
        }
        let token_weight = match dict_optional_f64_for_tail(dict, "finbert_token_count_512")? {
            Some(value) if value > 0.0 => value,
            _ => 0.0,
        };
        let filing_date = dict_py_object_or_none(py, dict, "filing_date")?;
        let filing_date_key = if filing_date.is_none(py) {
            None
        } else {
            Some(filing_date.bind(py).str()?.to_str()?.to_string())
        };
        parsed_rows.push(FinbertTailInputRow {
            key: FinbertTailGroupKey {
                doc_id: dict_optional_string_for_tail(dict, "doc_id")?,
                filing_date_key,
                text_scope: Some(text_scope),
                cleaning_policy_id: dict_optional_string_for_tail(dict, "cleaning_policy_id")?,
                model_name: dict_optional_string_for_tail(dict, "model_name")?,
                model_version: dict_optional_string_for_tail(dict, "model_version")?,
                segment_policy_id: dict_optional_string_for_tail(dict, "segment_policy_id")?,
            },
            filing_date,
            negative_prob: dict_optional_f64_for_tail(dict, "negative_prob")?,
            sentence_index: dict_optional_i64_for_tail(dict, "sentence_index")?,
            benchmark_sentence_id: dict_optional_string_for_tail(dict, "benchmark_sentence_id")?,
            token_weight,
        });
    }
    parsed_rows.sort_by(finbert_tail_input_row_cmp);

    let mut out: Vec<PyObject> = Vec::new();
    let mut start = 0usize;
    while start < parsed_rows.len() {
        let mut end = start + 1;
        while end < parsed_rows.len() && parsed_rows[end].key == parsed_rows[start].key {
            end += 1;
        }
        let group = &parsed_rows[start..end];
        let first = &group[0];
        let denominator: f64 = group.iter().map(|row| row.token_weight).sum();

        let feature_values = if denominator > 0.0 {
            let weighted_mean_numerator: f64 = group
                .iter()
                .filter_map(|row| row.negative_prob.map(|value| value * row.token_weight))
                .sum();
            let weighted_mean = weighted_mean_numerator / denominator;
            let dispersion_numerator: f64 = group
                .iter()
                .filter_map(|row| {
                    row.negative_prob.map(|value| {
                        let delta = value - weighted_mean;
                        delta * delta * row.token_weight
                    })
                })
                .sum();
            let threshold_exposure = |threshold: f64| -> f64 {
                group
                    .iter()
                    .map(|row| match row.negative_prob {
                        Some(value) if value >= threshold => value * row.token_weight,
                        _ => 0.0,
                    })
                    .sum::<f64>()
                    / denominator
            };
            let threshold_share = |threshold: f64| -> f64 {
                group
                    .iter()
                    .map(|row| match row.negative_prob {
                        Some(value) if value >= threshold => row.token_weight,
                        _ => 0.0,
                    })
                    .sum::<f64>()
                    / denominator
            };
            (
                Some(threshold_exposure(0.60)),
                Some(threshold_exposure(0.70)),
                Some(threshold_exposure(0.80)),
                Some(threshold_share(0.70)),
                finbert_tail_weighted_mean_for_ranks(
                    group,
                    finbert_tail_top_fraction_cutoff(group.len(), 0.10),
                ),
                finbert_tail_weighted_mean_for_ranks(
                    group,
                    finbert_tail_top_fraction_cutoff(group.len(), 0.20),
                ),
                finbert_tail_weighted_mean_for_ranks(group, 5),
                Some((dispersion_numerator / denominator).sqrt()),
            )
        } else {
            (None, None, None, None, None, None, None, None)
        };

        let row_out = PyDict::new_bound(py);
        row_out.set_item("doc_id", first.key.doc_id.clone())?;
        row_out.set_item("filing_date", first.filing_date.clone_ref(py))?;
        row_out.set_item("text_scope", first.key.text_scope.clone())?;
        row_out.set_item("cleaning_policy_id", first.key.cleaning_policy_id.clone())?;
        row_out.set_item("model_name", first.key.model_name.clone())?;
        row_out.set_item("model_version", first.key.model_version.clone())?;
        row_out.set_item("segment_policy_id", first.key.segment_policy_id.clone())?;
        row_out.set_item("tail_exposure_tau_0_60", feature_values.0)?;
        row_out.set_item("tail_exposure_tau_0_70", feature_values.1)?;
        row_out.set_item("tail_exposure_tau_0_80", feature_values.2)?;
        row_out.set_item("tail_share_tau_0_70", feature_values.3)?;
        row_out.set_item("top_10pct_neg_mean", feature_values.4)?;
        row_out.set_item("top_20pct_neg_mean", feature_values.5)?;
        row_out.set_item("top_5_sentences_neg_mean", feature_values.6)?;
        row_out.set_item("neg_prob_dispersion", feature_values.7)?;
        out.push(row_out.into_py(py));
        start = end;
    }
    Ok(out)
}

#[pyfunction]
pub(crate) fn finbert_tail_doc_surface_columns(
    py: Python<'_>,
    column_names: Vec<String>,
    column_values: &Bound<'_, PyAny>,
    text_scopes: Vec<String>,
) -> PyResult<Vec<PyObject>> {
    let mut normalized_scopes: Vec<String> = Vec::new();
    let mut seen_scopes: HashSet<String> = HashSet::new();
    for scope in text_scopes {
        let normalized = finbert_tail_text_scope_impl(&scope);
        if seen_scopes.insert(normalized.clone()) {
            normalized_scopes.push(normalized);
        }
    }
    if normalized_scopes.is_empty() {
        return Err(PyValueError::new_err(
            "text_scopes must be non-empty for FinBERT tail aggregation.",
        ));
    }
    let normalized_scope_set: HashSet<String> = normalized_scopes.into_iter().collect();

    let mut columns: Vec<Vec<PyObject>> = Vec::with_capacity(column_names.len());
    let mut row_count: Option<usize> = None;
    for values in column_values.iter()? {
        let values = values?;
        let mut column: Vec<PyObject> = Vec::new();
        for value in values.iter()? {
            column.push(value?.clone().into_py(py));
        }
        match row_count {
            Some(expected) if column.len() != expected => {
                return Err(PyValueError::new_err(
                    "FinBERT tail column lengths must match",
                ))
            }
            None => row_count = Some(column.len()),
            _ => {}
        }
        columns.push(column);
    }
    if columns.len() != column_names.len() {
        return Err(PyValueError::new_err(
            "FinBERT tail column name/value count mismatch",
        ));
    }
    let row_count = row_count.unwrap_or(0);
    let column_index = column_index_by_name(&column_names);

    let mut parsed_rows: Vec<FinbertTailInputRow> = Vec::new();
    for row_idx in 0..row_count {
        let optional_string = |key: &str| -> PyResult<Option<String>> {
            let Some(&column_idx) = column_index.get(key) else {
                return Ok(None);
            };
            let value = columns[column_idx][row_idx].bind(py);
            if value.is_none() {
                Ok(None)
            } else {
                Ok(Some(value.str()?.to_str()?.to_string()))
            }
        };
        let optional_i64 = |key: &str| -> PyResult<Option<i64>> {
            let Some(&column_idx) = column_index.get(key) else {
                return Ok(None);
            };
            let value = columns[column_idx][row_idx].bind(py);
            if value.is_none() {
                return Ok(None);
            }
            match py_int_like_to_i64(value) {
                Ok(value) => Ok(Some(value)),
                Err(_) => Ok(None),
            }
        };
        let optional_f64 = |key: &str| -> PyResult<Option<f64>> {
            let Some(&column_idx) = column_index.get(key) else {
                return Ok(None);
            };
            let value = columns[column_idx][row_idx].bind(py);
            py_float_like_to_finite_option(value)
        };

        let mut text_scope: Option<String> = None;
        let mut text_scope_column_seen = false;
        for key in ["text_scope", "benchmark_item_code", "item_id"] {
            if !column_index.contains_key(key) {
                continue;
            }
            text_scope_column_seen = true;
            let Some(value) = optional_string(key)? else {
                text_scope = None;
                break;
            };
            text_scope = Some(lm2011_extension_text_scope_impl(&value));
            break;
        }
        if !text_scope_column_seen {
            return Err(PyValueError::new_err(
                "sentence_scores must contain text_scope, benchmark_item_code, or item_id for tail aggregation.",
            ));
        }
        let Some(text_scope) = text_scope else {
            continue;
        };
        if !normalized_scope_set.contains(&text_scope) {
            continue;
        }

        let token_weight = match optional_f64("finbert_token_count_512")? {
            Some(value) if value > 0.0 => value,
            _ => 0.0,
        };
        let filing_date = if let Some(&column_idx) = column_index.get("filing_date") {
            columns[column_idx][row_idx].clone_ref(py)
        } else {
            py.None()
        };
        let filing_date_key = if filing_date.is_none(py) {
            None
        } else {
            Some(filing_date.bind(py).str()?.to_str()?.to_string())
        };
        parsed_rows.push(FinbertTailInputRow {
            key: FinbertTailGroupKey {
                doc_id: optional_string("doc_id")?,
                filing_date_key,
                text_scope: Some(text_scope),
                cleaning_policy_id: optional_string("cleaning_policy_id")?,
                model_name: optional_string("model_name")?,
                model_version: optional_string("model_version")?,
                segment_policy_id: optional_string("segment_policy_id")?,
            },
            filing_date,
            negative_prob: optional_f64("negative_prob")?,
            sentence_index: optional_i64("sentence_index")?,
            benchmark_sentence_id: optional_string("benchmark_sentence_id")?,
            token_weight,
        });
    }
    parsed_rows.sort_by(finbert_tail_input_row_cmp);

    let mut out: Vec<PyObject> = Vec::new();
    let mut start = 0usize;
    while start < parsed_rows.len() {
        let mut end = start + 1;
        while end < parsed_rows.len() && parsed_rows[end].key == parsed_rows[start].key {
            end += 1;
        }
        let group = &parsed_rows[start..end];
        let first = &group[0];
        let denominator: f64 = group.iter().map(|row| row.token_weight).sum();

        let feature_values = if denominator > 0.0 {
            let weighted_mean_numerator: f64 = group
                .iter()
                .filter_map(|row| row.negative_prob.map(|value| value * row.token_weight))
                .sum();
            let weighted_mean = weighted_mean_numerator / denominator;
            let dispersion_numerator: f64 = group
                .iter()
                .filter_map(|row| {
                    row.negative_prob.map(|value| {
                        let delta = value - weighted_mean;
                        delta * delta * row.token_weight
                    })
                })
                .sum();
            let threshold_exposure = |threshold: f64| -> f64 {
                group
                    .iter()
                    .map(|row| match row.negative_prob {
                        Some(value) if value >= threshold => value * row.token_weight,
                        _ => 0.0,
                    })
                    .sum::<f64>()
                    / denominator
            };
            let threshold_share = |threshold: f64| -> f64 {
                group
                    .iter()
                    .map(|row| match row.negative_prob {
                        Some(value) if value >= threshold => row.token_weight,
                        _ => 0.0,
                    })
                    .sum::<f64>()
                    / denominator
            };
            (
                Some(threshold_exposure(0.60)),
                Some(threshold_exposure(0.70)),
                Some(threshold_exposure(0.80)),
                Some(threshold_share(0.70)),
                finbert_tail_weighted_mean_for_ranks(
                    group,
                    finbert_tail_top_fraction_cutoff(group.len(), 0.10),
                ),
                finbert_tail_weighted_mean_for_ranks(
                    group,
                    finbert_tail_top_fraction_cutoff(group.len(), 0.20),
                ),
                finbert_tail_weighted_mean_for_ranks(group, 5),
                Some((dispersion_numerator / denominator).sqrt()),
            )
        } else {
            (None, None, None, None, None, None, None, None)
        };

        let row_out = PyDict::new_bound(py);
        row_out.set_item("doc_id", first.key.doc_id.clone())?;
        row_out.set_item("filing_date", first.filing_date.clone_ref(py))?;
        row_out.set_item("text_scope", first.key.text_scope.clone())?;
        row_out.set_item("cleaning_policy_id", first.key.cleaning_policy_id.clone())?;
        row_out.set_item("model_name", first.key.model_name.clone())?;
        row_out.set_item("model_version", first.key.model_version.clone())?;
        row_out.set_item("segment_policy_id", first.key.segment_policy_id.clone())?;
        row_out.set_item("tail_exposure_tau_0_60", feature_values.0)?;
        row_out.set_item("tail_exposure_tau_0_70", feature_values.1)?;
        row_out.set_item("tail_exposure_tau_0_80", feature_values.2)?;
        row_out.set_item("tail_share_tau_0_70", feature_values.3)?;
        row_out.set_item("top_10pct_neg_mean", feature_values.4)?;
        row_out.set_item("top_20pct_neg_mean", feature_values.5)?;
        row_out.set_item("top_5_sentences_neg_mean", feature_values.6)?;
        row_out.set_item("neg_prob_dispersion", feature_values.7)?;
        out.push(row_out.into_py(py));
        start = end;
    }
    Ok(out)
}

#[pyfunction]
pub(crate) fn sha256_hex_value(text: &str) -> String {
    sha256_hex_impl(text)
}

#[pyfunction]
pub(crate) fn sha256_first_u64_value(text: &str) -> u64 {
    sha256_first_u64_impl(text)
}

#[pyfunction]
pub(crate) fn sha256_hex_values(values: Vec<String>) -> Vec<String> {
    values.iter().map(|value| sha256_hex_impl(value)).collect()
}

#[pyfunction]
pub(crate) fn stable_string_fingerprint_values(values: &Bound<'_, PyAny>) -> PyResult<String> {
    let mut unique_values: BTreeSet<String> = BTreeSet::new();
    for value in values.iter()? {
        let value = value?;
        if value.is_none() {
            continue;
        }
        unique_values.insert(value.str()?.to_str()?.to_string());
    }
    let mut text = String::new();
    for value in unique_values {
        text.push_str(&value);
        text.push('\n');
    }
    Ok(sha256_hex_impl(&text))
}

pub(crate) fn manifest_optional_dict<'py>(
    source: &'py Bound<'py, PyDict>,
    key: &str,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let Some(value) = source.get_item(key)? else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    match value.downcast::<PyDict>() {
        Ok(dict) => Ok(Some(dict.clone())),
        Err(_) => Ok(None),
    }
}

pub(crate) fn manifest_py_values_equal(
    py: Python<'_>,
    left: Option<Bound<'_, PyAny>>,
    right: Option<Bound<'_, PyAny>>,
) -> PyResult<bool> {
    match (left, right) {
        (None, None) => Ok(true),
        (Some(value), None) | (None, Some(value)) => Ok(value.is_none()),
        (Some(left), Some(right)) => py
            .import_bound("operator")?
            .getattr("eq")?
            .call1((left, right))?
            .extract::<bool>(),
    }
}

pub(crate) fn manifest_dict_key_strings(
    dict: Option<&Bound<'_, PyDict>>,
) -> PyResult<BTreeSet<String>> {
    let mut keys = BTreeSet::new();
    if let Some(dict) = dict {
        for (key, _) in dict.iter() {
            keys.insert(key.str()?.to_str()?.to_string());
        }
    }
    Ok(keys)
}

#[pyfunction]
pub(crate) fn semantic_guard_mismatches_value(
    py: Python<'_>,
    existing: &Bound<'_, PyAny>,
    expected: &Bound<'_, PyAny>,
) -> PyResult<Vec<String>> {
    let existing = existing
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("existing guard must be a dict"))?;
    let expected = expected
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("expected guard must be a dict"))?;
    let mut mismatches: Vec<String> = Vec::new();
    if !manifest_py_values_equal(
        py,
        existing.get_item("version")?,
        expected.get_item("version")?,
    )? {
        mismatches.push("version".to_string());
    }

    let existing_payload = manifest_optional_dict(existing, "payload")?;
    let expected_payload = manifest_optional_dict(expected, "payload")?;
    let mut payload_keys = manifest_dict_key_strings(existing_payload.as_ref())?;
    payload_keys.extend(manifest_dict_key_strings(expected_payload.as_ref())?);
    for key in payload_keys {
        let left = match existing_payload.as_ref() {
            Some(dict) => dict.get_item(key.as_str())?,
            None => None,
        };
        let right = match expected_payload.as_ref() {
            Some(dict) => dict.get_item(key.as_str())?,
            None => None,
        };
        if !manifest_py_values_equal(py, left, right)? {
            mismatches.push(format!("payload.{key}"));
        }
    }

    let existing_fingerprints = manifest_optional_dict(existing, "fingerprints")?;
    let expected_fingerprints = manifest_optional_dict(expected, "fingerprints")?;
    let mut fingerprint_keys = manifest_dict_key_strings(existing_fingerprints.as_ref())?;
    fingerprint_keys.extend(manifest_dict_key_strings(expected_fingerprints.as_ref())?);
    for key in fingerprint_keys {
        let left = match existing_fingerprints.as_ref() {
            Some(dict) => dict.get_item(key.as_str())?,
            None => None,
        };
        let right = match expected_fingerprints.as_ref() {
            Some(dict) => dict.get_item(key.as_str())?,
            None => None,
        };
        if !manifest_py_values_equal(py, left, right)? {
            mismatches.push(format!("fingerprints.{key}"));
        }
    }

    Ok(mismatches)
}

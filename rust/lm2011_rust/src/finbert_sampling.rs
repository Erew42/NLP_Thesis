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
pub(crate) fn finbert_selection_key_value(
    doc_id: &str,
    benchmark_item_code: &str,
    seed: i64,
) -> String {
    finbert_selection_key_impl(doc_id, benchmark_item_code, seed)
}

#[pyfunction]
pub(crate) fn finbert_sentence_sample_key_value(seed: i64, benchmark_sentence_id: &str) -> u64 {
    blake2b_8_u64_big_endian(&format!("{seed}|{benchmark_sentence_id}"))
}

#[pyfunction]
pub(crate) fn finbert_sentence_sample_key_values(
    seed: i64,
    benchmark_sentence_ids: Vec<String>,
) -> Vec<u64> {
    benchmark_sentence_ids
        .iter()
        .map(|benchmark_sentence_id| {
            blake2b_8_u64_big_endian(&format!("{seed}|{benchmark_sentence_id}"))
        })
        .collect()
}

#[pyfunction]
pub(crate) fn finbert_sentence_consider_sample_rows(
    py: Python<'_>,
    selected_rows: &Bound<'_, PyAny>,
    candidate_row: &Bound<'_, PyDict>,
    sample_size_per_group: usize,
    sample_key: u64,
    normalized_text: &str,
) -> PyResult<Vec<PyObject>> {
    let mut selected: Vec<Bound<'_, PyDict>> = Vec::new();
    for row in selected_rows.iter()? {
        let row = row?;
        selected.push(
            row.downcast::<PyDict>()
                .map_err(|_| PyValueError::new_err("selected sample row is not a dict"))?
                .clone(),
        );
    }

    let candidate = copy_py_dict(py, candidate_row)?;
    candidate.set_item("_sample_key", sample_key)?;
    candidate.set_item("_normalized_text", normalized_text)?;
    let candidate_doc_id = dict_raw_string(&candidate, "doc_id")?;

    let mut conflicting_indexes: Vec<usize> = Vec::new();
    for (index, existing) in selected.iter().enumerate() {
        let same_text =
            dict_raw_string(existing, "_normalized_text")?.as_deref() == Some(normalized_text);
        let same_doc = dict_raw_string(existing, "doc_id")? == candidate_doc_id;
        if same_text || same_doc {
            conflicting_indexes.push(index);
        }
    }

    for index in conflicting_indexes.iter().copied() {
        let Some(existing_key_value) = selected[index].get_item("_sample_key")? else {
            return Err(PyValueError::new_err(
                "selected sample row missing _sample_key",
            ));
        };
        if sample_key >= existing_key_value.extract::<u64>()? {
            return selected
                .into_iter()
                .map(|row| Ok(row.into_py(py)))
                .collect();
        }
    }

    for index in conflicting_indexes.into_iter().rev() {
        selected.remove(index);
    }

    if selected.len() < sample_size_per_group {
        selected.push(candidate);
        return selected
            .into_iter()
            .map(|row| Ok(row.into_py(py)))
            .collect();
    }

    let mut largest_index: Option<usize> = None;
    let mut largest_key: u64 = 0;
    for (index, existing) in selected.iter().enumerate() {
        let Some(existing_key_value) = existing.get_item("_sample_key")? else {
            return Err(PyValueError::new_err(
                "selected sample row missing _sample_key",
            ));
        };
        let existing_key = existing_key_value.extract::<u64>()?;
        if largest_index.is_none() || existing_key > largest_key {
            largest_index = Some(index);
            largest_key = existing_key;
        }
    }
    if sample_key < largest_key {
        if let Some(index) = largest_index {
            selected[index] = candidate;
        }
    }
    selected
        .into_iter()
        .map(|row| Ok(row.into_py(py)))
        .collect()
}

pub(crate) fn sentence_example_increment_accumulator(
    py: Python<'_>,
    counts: &Bound<'_, PyDict>,
    key: &Bound<'_, PyTuple>,
    doc_id: &str,
) -> PyResult<()> {
    let acc = if let Some(acc_value) = counts.get_item(key)? {
        acc_value
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("sentence example accumulator is not a dict"))?
            .clone()
    } else {
        let created = PyDict::new_bound(py);
        created.set_item("candidate_rows", 0i64)?;
        created.set_item("doc_ids", PySet::empty_bound(py)?)?;
        counts.set_item(key, &created)?;
        created
    };
    let Some(candidate_rows_value) = acc.get_item("candidate_rows")? else {
        return Err(PyValueError::new_err(
            "sentence example accumulator missing candidate_rows",
        ));
    };
    acc.set_item(
        "candidate_rows",
        py_int_like_to_i64(&candidate_rows_value)? + 1,
    )?;
    let Some(doc_ids_value) = acc.get_item("doc_ids")? else {
        return Err(PyValueError::new_err(
            "sentence example accumulator missing doc_ids",
        ));
    };
    doc_ids_value.call_method1("add", (doc_id,))?;
    Ok(())
}

pub(crate) struct SentenceAccumulatorUpdateRow<'py> {
    row: Bound<'py, PyDict>,
    item_code: String,
    sentiment: String,
    filing_year: i64,
    doc_id: String,
    normalized_text: String,
    sample_key: u64,
}

#[pyfunction]
pub(crate) fn finbert_sentence_update_accumulators(
    py: Python<'_>,
    rows: &Bound<'_, PyAny>,
    counts_by_item_sentiment: &Bound<'_, PyAny>,
    counts_by_year_item_sentiment: &Bound<'_, PyAny>,
    selected_samples: &Bound<'_, PyAny>,
    sample_size_per_group: usize,
    seed: i64,
    all_doc_ids: &Bound<'_, PyAny>,
) -> PyResult<usize> {
    if sample_size_per_group == 0 {
        return Err(PyValueError::new_err(
            "sample_size_per_group must be positive for Rust accumulator update",
        ));
    }
    let counts_by_item_sentiment = counts_by_item_sentiment
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("counts_by_item_sentiment is not a dict"))?;
    let counts_by_year_item_sentiment = counts_by_year_item_sentiment
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("counts_by_year_item_sentiment is not a dict"))?;
    let selected_samples = selected_samples
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("selected_samples is not a dict"))?;

    let mut prepared_rows: Vec<SentenceAccumulatorUpdateRow<'_>> = Vec::new();
    for row_value in rows.iter()? {
        let row_value = row_value?;
        let row = row_value
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("sentence example row is not a dict"))?;
        let item_code = dict_required_string(row, "benchmark_item_code")?;
        let sentiment = dict_required_string(row, "sentiment")?;
        let Some(filing_year_value) = row.get_item("filing_year")? else {
            return Err(PyValueError::new_err("missing required key: filing_year"));
        };
        let filing_year = py_int_like_to_i64(&filing_year_value)?;
        let doc_id = dict_required_string(row, "doc_id")?;
        let benchmark_sentence_id = dict_required_string(row, "benchmark_sentence_id")?;
        let sentence_text = dict_required_string(row, "sentence_text")?;
        let Some(normalized_text) = normalize_ascii_sample_text_impl(&sentence_text) else {
            return Err(PyValueError::new_err(
                "non-ASCII sentence text requires Python fallback",
            ));
        };
        let sample_key = finbert_sentence_sample_key_value(seed, &benchmark_sentence_id);
        prepared_rows.push(SentenceAccumulatorUpdateRow {
            row: row.clone(),
            item_code,
            sentiment,
            filing_year,
            doc_id,
            normalized_text,
            sample_key,
        });
    }

    for prepared in prepared_rows.iter() {
        let key = PyTuple::new_bound(
            py,
            [prepared.item_code.as_str(), prepared.sentiment.as_str()],
        );
        let year_key = PyTuple::new_bound(
            py,
            [
                prepared.filing_year.into_py(py),
                prepared.item_code.as_str().into_py(py),
                prepared.sentiment.as_str().into_py(py),
            ],
        );
        sentence_example_increment_accumulator(
            py,
            counts_by_item_sentiment,
            &key,
            &prepared.doc_id,
        )?;
        sentence_example_increment_accumulator(
            py,
            counts_by_year_item_sentiment,
            &year_key,
            &prepared.doc_id,
        )?;
        all_doc_ids.call_method1("add", (prepared.doc_id.as_str(),))?;

        let selected_rows_value = if let Some(value) = selected_samples.get_item(&key)? {
            value
        } else {
            let empty = PyList::empty_bound(py);
            selected_samples.set_item(&key, &empty)?;
            empty.into_any()
        };
        let updated_rows = finbert_sentence_consider_sample_rows(
            py,
            &selected_rows_value,
            &prepared.row,
            sample_size_per_group,
            prepared.sample_key,
            &prepared.normalized_text,
        )?;
        selected_samples.set_item(&key, PyList::new_bound(py, updated_rows))?;
    }
    Ok(prepared_rows.len())
}

pub(crate) struct SentenceExampleCandidate<'py> {
    row: Bound<'py, PyDict>,
    sentiment_probability: f64,
    filing_year: i64,
    doc_id: String,
    benchmark_sentence_id: String,
}

#[pyfunction]
pub(crate) fn finbert_sentence_sample_candidate_rows(
    py: Python<'_>,
    selected_samples: &Bound<'_, PyAny>,
    ordered_items: Vec<String>,
    sentiment_order: Vec<String>,
    output_columns: Vec<String>,
    filing_date_column: &str,
) -> PyResult<Vec<PyObject>> {
    let selected = selected_samples
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("selected_samples is not a dict"))?;
    let mut out_rows: Vec<PyObject> = Vec::new();

    for item_code in ordered_items.iter() {
        for sentiment in sentiment_order.iter() {
            let key = PyTuple::new_bound(py, [item_code.as_str(), sentiment.as_str()]);
            let Some(candidates_value) = selected.get_item(&key)? else {
                continue;
            };
            let mut candidates: Vec<SentenceExampleCandidate<'_>> = Vec::new();
            for row in candidates_value.iter()? {
                let row = row?;
                let dict = row.downcast::<PyDict>().map_err(|_| {
                    PyValueError::new_err("sentence example candidate row is not a dict")
                })?;
                let Some(prob_value) = dict.get_item("sentiment_probability")? else {
                    return Err(PyValueError::new_err(
                        "missing required key: sentiment_probability",
                    ));
                };
                let sentiment_probability = py_float_like_to_finite_option(&prob_value)?
                    .ok_or_else(|| PyValueError::new_err("invalid sentiment_probability"))?;
                let Some(year_value) = dict.get_item("filing_year")? else {
                    return Err(PyValueError::new_err("missing required key: filing_year"));
                };
                let filing_year = py_int_like_to_i64(&year_value)?;
                candidates.push(SentenceExampleCandidate {
                    row: dict.clone(),
                    sentiment_probability,
                    filing_year,
                    doc_id: dict_required_string(dict, "doc_id")?,
                    benchmark_sentence_id: dict_required_string(dict, "benchmark_sentence_id")?,
                });
            }

            candidates.sort_by(|left, right| {
                right
                    .sentiment_probability
                    .partial_cmp(&left.sentiment_probability)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| left.filing_year.cmp(&right.filing_year))
                    .then_with(|| left.doc_id.cmp(&right.doc_id))
                    .then_with(|| left.benchmark_sentence_id.cmp(&right.benchmark_sentence_id))
            });

            for candidate in candidates {
                let out = PyDict::new_bound(py);
                for column in output_columns.iter() {
                    let Some(value) = candidate.row.get_item(column.as_str())? else {
                        return Err(PyValueError::new_err(format!(
                            "missing required key: {column}"
                        )));
                    };
                    if column == filing_date_column {
                        if let Ok(isoformat) = value.getattr("isoformat") {
                            out.set_item(column, isoformat.call0()?)?;
                            continue;
                        }
                    }
                    out.set_item(column, value)?;
                }
                out_rows.push(out.into_py(py));
            }
        }
    }

    Ok(out_rows)
}

pub(crate) fn markdown_dict_display(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<String> {
    let Some(value) = dict.get_item(key)? else {
        return Err(PyValueError::new_err(format!(
            "missing required key: {key}"
        )));
    };
    Ok(value.str()?.to_str()?.to_string())
}

pub(crate) fn markdown_value_display(value: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(value.str()?.to_str()?.to_string())
}

pub(crate) fn markdown_metadata_i64(metadata: &Bound<'_, PyDict>, key: &str) -> PyResult<i64> {
    let Some(value) = metadata.get_item(key)? else {
        return Err(PyValueError::new_err(format!(
            "missing required metadata key: {key}"
        )));
    };
    py_int_like_to_i64(&value)
}

pub(crate) fn markdown_metadata_string(
    metadata: &Bound<'_, PyDict>,
    key: &str,
) -> PyResult<String> {
    let Some(value) = metadata.get_item(key)? else {
        return Err(PyValueError::new_err(format!(
            "missing required metadata key: {key}"
        )));
    };
    Ok(value.str()?.to_str()?.to_string())
}

pub(crate) fn markdown_metadata_string_list(
    metadata: &Bound<'_, PyDict>,
    key: &str,
) -> PyResult<Vec<String>> {
    let Some(value) = metadata.get_item(key)? else {
        return Err(PyValueError::new_err(format!(
            "missing required metadata key: {key}"
        )));
    };
    let mut out = Vec::new();
    for item in value.iter()? {
        out.push(item?.str()?.to_str()?.to_string());
    }
    Ok(out)
}

pub(crate) fn markdown_item_label(item_code: &str) -> &str {
    match item_code {
        "item_1a" => "Item 1A",
        "item_7" => "Item 7",
        _ => item_code,
    }
}

#[pyfunction]
pub(crate) fn finbert_sentence_render_sample_markdown(
    sample_rows: &Bound<'_, PyAny>,
    count_rows: &Bound<'_, PyAny>,
    metadata: &Bound<'_, PyDict>,
    sentiment_order: Vec<String>,
) -> PyResult<String> {
    let mut counts_lookup: HashMap<(String, String), i64> = HashMap::new();
    for row in count_rows.iter()? {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("count row is not a dict"))?;
        counts_lookup.insert(
            (
                dict_required_string(dict, "benchmark_item_code")?,
                dict_required_string(dict, "sentiment")?,
            ),
            dict_required_i64(dict, "candidate_rows")?,
        );
    }

    let mut rows: Vec<Bound<'_, PyDict>> = Vec::new();
    for row in sample_rows.iter()? {
        let row = row?;
        rows.push(
            row.downcast::<PyDict>()
                .map_err(|_| PyValueError::new_err("sample row is not a dict"))?
                .clone(),
        );
    }

    let filters_value = metadata
        .get_item("filters")?
        .ok_or_else(|| PyValueError::new_err("missing required metadata key: filters"))?;
    let filters = filters_value
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("metadata filters is not a dict"))?;
    let filter_item_codes = markdown_metadata_string_list(filters, "item_codes")?;
    let min_probability = dict_required_float(filters, "min_probability")?;
    let min_word_count = dict_required_i64(filters, "min_word_count")?;
    let batch_size = dict_required_i64(filters, "batch_size")?;
    let item_codes_present_ordered =
        markdown_metadata_string_list(metadata, "item_codes_present_ordered")?;

    let mut lines = vec![
        "# FinBERT High-Confidence Sentence Samples".to_string(),
        String::new(),
        format!(
            "Source sentence_scores: {}",
            markdown_metadata_string(metadata, "sentence_scores_dir")?
        ),
        format!(
            "Filters: items={}, min_probability={:.2}, min_word_count={}, batch_size={}",
            filter_item_codes.join(", "),
            min_probability,
            min_word_count,
            batch_size
        ),
        format!(
            "Candidate rows={}, candidate docs={}, sample_size_per_group={}, sample_seed={}",
            markdown_metadata_i64(metadata, "candidate_rows")?,
            markdown_metadata_i64(metadata, "candidate_doc_count")?,
            markdown_metadata_i64(metadata, "sample_size_per_group")?,
            markdown_metadata_i64(metadata, "sample_seed")?
        ),
        String::new(),
    ];

    for item_code in item_codes_present_ordered {
        for sentiment in sentiment_order.iter() {
            let mut bucket_rows: Vec<&Bound<'_, PyDict>> = Vec::new();
            for row in rows.iter() {
                if dict_required_string(row, "benchmark_item_code")? == item_code
                    && dict_required_string(row, "sentiment")? == *sentiment
                {
                    bucket_rows.push(row);
                }
            }
            let listed_samples = bucket_rows.len();
            let eligible_candidates = counts_lookup
                .get(&(item_code.clone(), sentiment.clone()))
                .copied()
                .unwrap_or(0);
            let mut title = sentiment.clone();
            if let Some(first) = title.get_mut(0..1) {
                first.make_ascii_uppercase();
            }
            lines.push(format!(
                "## {} | {}",
                markdown_item_label(&item_code),
                title
            ));
            lines.push(format!(
                "Eligible candidates: {} | Listed samples: {}",
                eligible_candidates, listed_samples
            ));
            lines.push(String::new());
            for (index, row) in bucket_rows.into_iter().enumerate() {
                lines.push(format!(
                    "{}. p={:.3} | year={} | date={} | doc_id={} | words={} | {}",
                    index + 1,
                    dict_required_float(row, "sentiment_probability")?,
                    markdown_dict_display(row, "filing_year")?,
                    markdown_dict_display(row, "filing_date")?,
                    markdown_dict_display(row, "doc_id")?,
                    markdown_dict_display(row, "sentence_word_count")?,
                    markdown_dict_display(row, "sentence_text")?
                ));
            }
            lines.push(String::new());
        }
    }
    Ok(lines.join("\n").trim_end().to_string() + "\n")
}

#[pyfunction]
pub(crate) fn finbert_sentence_render_sample_markdown_columns(
    py: Python<'_>,
    sample_column_names: Vec<String>,
    sample_column_values: &Bound<'_, PyAny>,
    count_column_names: Vec<String>,
    count_column_values: &Bound<'_, PyAny>,
    metadata: &Bound<'_, PyDict>,
    sentiment_order: Vec<String>,
) -> PyResult<String> {
    let (sample_columns, sample_row_count) = collect_pyobject_column_values(
        py,
        &sample_column_names,
        sample_column_values,
        "FinBERT sentence sample markdown sample rows",
    )?;
    let sample_index = column_index_by_name(&sample_column_names);
    let sample_item_idx = required_named_column_index(
        &sample_index,
        "FinBERT sentence sample markdown sample rows",
        "benchmark_item_code",
    )?;
    let sample_sentiment_idx = required_named_column_index(
        &sample_index,
        "FinBERT sentence sample markdown sample rows",
        "sentiment",
    )?;
    let sample_probability_idx = required_named_column_index(
        &sample_index,
        "FinBERT sentence sample markdown sample rows",
        "sentiment_probability",
    )?;
    let sample_year_idx = required_named_column_index(
        &sample_index,
        "FinBERT sentence sample markdown sample rows",
        "filing_year",
    )?;
    let sample_date_idx = required_named_column_index(
        &sample_index,
        "FinBERT sentence sample markdown sample rows",
        "filing_date",
    )?;
    let sample_doc_id_idx = required_named_column_index(
        &sample_index,
        "FinBERT sentence sample markdown sample rows",
        "doc_id",
    )?;
    let sample_word_count_idx = required_named_column_index(
        &sample_index,
        "FinBERT sentence sample markdown sample rows",
        "sentence_word_count",
    )?;
    let sample_text_idx = required_named_column_index(
        &sample_index,
        "FinBERT sentence sample markdown sample rows",
        "sentence_text",
    )?;

    let (count_columns, count_row_count) = collect_pyobject_column_values(
        py,
        &count_column_names,
        count_column_values,
        "FinBERT sentence sample markdown count rows",
    )?;
    let count_index = column_index_by_name(&count_column_names);
    let count_item_idx = required_named_column_index(
        &count_index,
        "FinBERT sentence sample markdown count rows",
        "benchmark_item_code",
    )?;
    let count_sentiment_idx = required_named_column_index(
        &count_index,
        "FinBERT sentence sample markdown count rows",
        "sentiment",
    )?;
    let count_candidate_idx = required_named_column_index(
        &count_index,
        "FinBERT sentence sample markdown count rows",
        "candidate_rows",
    )?;
    let mut counts_lookup: HashMap<(String, String), i64> = HashMap::new();
    for row_idx in 0..count_row_count {
        counts_lookup.insert(
            (
                markdown_value_display(count_columns[count_item_idx][row_idx].bind(py))?,
                markdown_value_display(count_columns[count_sentiment_idx][row_idx].bind(py))?,
            ),
            py_int_like_to_i64(count_columns[count_candidate_idx][row_idx].bind(py))?,
        );
    }

    let filters_value = metadata
        .get_item("filters")?
        .ok_or_else(|| PyValueError::new_err("missing required metadata key: filters"))?;
    let filters = filters_value
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("metadata filters is not a dict"))?;
    let filter_item_codes = markdown_metadata_string_list(filters, "item_codes")?;
    let min_probability = dict_required_float(filters, "min_probability")?;
    let min_word_count = dict_required_i64(filters, "min_word_count")?;
    let batch_size = dict_required_i64(filters, "batch_size")?;
    let item_codes_present_ordered =
        markdown_metadata_string_list(metadata, "item_codes_present_ordered")?;

    let mut lines = vec![
        "# FinBERT High-Confidence Sentence Samples".to_string(),
        String::new(),
        format!(
            "Source sentence_scores: {}",
            markdown_metadata_string(metadata, "sentence_scores_dir")?
        ),
        format!(
            "Filters: items={}, min_probability={:.2}, min_word_count={}, batch_size={}",
            filter_item_codes.join(", "),
            min_probability,
            min_word_count,
            batch_size
        ),
        format!(
            "Candidate rows={}, candidate docs={}, sample_size_per_group={}, sample_seed={}",
            markdown_metadata_i64(metadata, "candidate_rows")?,
            markdown_metadata_i64(metadata, "candidate_doc_count")?,
            markdown_metadata_i64(metadata, "sample_size_per_group")?,
            markdown_metadata_i64(metadata, "sample_seed")?
        ),
        String::new(),
    ];

    for item_code in item_codes_present_ordered {
        for sentiment in sentiment_order.iter() {
            let mut bucket_indices: Vec<usize> = Vec::new();
            for row_idx in 0..sample_row_count {
                if markdown_value_display(sample_columns[sample_item_idx][row_idx].bind(py))?
                    == item_code
                    && markdown_value_display(
                        sample_columns[sample_sentiment_idx][row_idx].bind(py),
                    )? == *sentiment
                {
                    bucket_indices.push(row_idx);
                }
            }
            let listed_samples = bucket_indices.len();
            let eligible_candidates = counts_lookup
                .get(&(item_code.clone(), sentiment.clone()))
                .copied()
                .unwrap_or(0);
            let mut title = sentiment.clone();
            if let Some(first) = title.get_mut(0..1) {
                first.make_ascii_uppercase();
            }
            lines.push(format!(
                "## {} | {}",
                markdown_item_label(&item_code),
                title
            ));
            lines.push(format!(
                "Eligible candidates: {} | Listed samples: {}",
                eligible_candidates, listed_samples
            ));
            lines.push(String::new());
            for (index, row_idx) in bucket_indices.into_iter().enumerate() {
                let probability = normalize_doc_ownership_float_value(Some(
                    sample_columns[sample_probability_idx][row_idx].bind(py),
                ))?
                .ok_or_else(|| PyValueError::new_err("null required key: sentiment_probability"))?;
                lines.push(format!(
                    "{}. p={:.3} | year={} | date={} | doc_id={} | words={} | {}",
                    index + 1,
                    probability,
                    markdown_value_display(sample_columns[sample_year_idx][row_idx].bind(py))?,
                    markdown_value_display(sample_columns[sample_date_idx][row_idx].bind(py))?,
                    markdown_value_display(sample_columns[sample_doc_id_idx][row_idx].bind(py))?,
                    markdown_value_display(
                        sample_columns[sample_word_count_idx][row_idx].bind(py)
                    )?,
                    markdown_value_display(sample_columns[sample_text_idx][row_idx].bind(py))?
                ));
            }
            lines.push(String::new());
        }
    }
    Ok(lines.join("\n").trim_end().to_string() + "\n")
}

pub(crate) fn sentence_example_acc_counts(acc: &Bound<'_, PyDict>) -> PyResult<(i64, i64)> {
    let Some(candidate_rows_value) = acc.get_item("candidate_rows")? else {
        return Err(PyValueError::new_err(
            "missing required key: candidate_rows",
        ));
    };
    let Some(doc_ids_value) = acc.get_item("doc_ids")? else {
        return Err(PyValueError::new_err("missing required key: doc_ids"));
    };
    let doc_count = py_int_like_to_i64(&doc_ids_value.call_method0("__len__")?)?;
    Ok((py_int_like_to_i64(&candidate_rows_value)?, doc_count))
}

#[pyfunction]
pub(crate) fn finbert_sentence_item_sentiment_count_rows(
    py: Python<'_>,
    counts_by_item_sentiment: &Bound<'_, PyAny>,
    ordered_items: Vec<String>,
    sentiment_order: Vec<String>,
) -> PyResult<Vec<PyObject>> {
    let counts = counts_by_item_sentiment
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("counts_by_item_sentiment is not a dict"))?;
    let mut out_rows: Vec<PyObject> = Vec::new();

    for item_code in ordered_items.iter() {
        for sentiment in sentiment_order.iter() {
            let key = PyTuple::new_bound(py, [item_code.as_str(), sentiment.as_str()]);
            let Some(acc_value) = counts.get_item(&key)? else {
                continue;
            };
            let acc = acc_value
                .downcast::<PyDict>()
                .map_err(|_| PyValueError::new_err("item/sentiment accumulator is not a dict"))?;
            let (candidate_rows, doc_count) = sentence_example_acc_counts(acc)?;
            let out = PyDict::new_bound(py);
            out.set_item("benchmark_item_code", item_code)?;
            out.set_item("sentiment", sentiment)?;
            out.set_item("candidate_rows", candidate_rows)?;
            out.set_item("doc_count", doc_count)?;
            out_rows.push(out.into_py(py));
        }
    }

    Ok(out_rows)
}

pub(crate) struct SentenceExampleYearCountRow {
    filing_year: i64,
    item_code: String,
    sentiment: String,
    candidate_rows: i64,
    doc_count: i64,
    item_order: usize,
    sentiment_order: usize,
}

#[pyfunction]
pub(crate) fn finbert_sentence_year_item_sentiment_count_rows(
    py: Python<'_>,
    counts_by_year_item_sentiment: &Bound<'_, PyAny>,
    ordered_items: Vec<String>,
    sentiment_order: Vec<String>,
) -> PyResult<Vec<PyObject>> {
    let counts = counts_by_year_item_sentiment
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("counts_by_year_item_sentiment is not a dict"))?;
    let item_order: HashMap<&str, usize> = ordered_items
        .iter()
        .enumerate()
        .map(|(index, value)| (value.as_str(), index))
        .collect();
    let sentiment_order_map: HashMap<&str, usize> = sentiment_order
        .iter()
        .enumerate()
        .map(|(index, value)| (value.as_str(), index))
        .collect();
    let default_item_order = item_order.len();
    let default_sentiment_order = sentiment_order_map.len();
    let mut rows: Vec<SentenceExampleYearCountRow> = Vec::new();

    for (key, acc_value) in counts.iter() {
        let key_tuple = key
            .downcast::<PyTuple>()
            .map_err(|_| PyValueError::new_err("year/item/sentiment key is not a tuple"))?;
        if key_tuple.len() != 3 {
            return Err(PyValueError::new_err(
                "year/item/sentiment key must have length 3",
            ));
        }
        let filing_year = py_int_like_to_i64(&key_tuple.get_item(0)?)?;
        let item_value = key_tuple.get_item(1)?;
        let sentiment_value = key_tuple.get_item(2)?;
        let item_code = item_value.str()?.to_str()?.to_string();
        let sentiment = sentiment_value.str()?.to_str()?.to_string();
        let acc = acc_value
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("year/item/sentiment accumulator is not a dict"))?;
        let (candidate_rows, doc_count) = sentence_example_acc_counts(acc)?;
        rows.push(SentenceExampleYearCountRow {
            filing_year,
            item_order: item_order
                .get(item_code.as_str())
                .copied()
                .unwrap_or(default_item_order),
            sentiment_order: sentiment_order_map
                .get(sentiment.as_str())
                .copied()
                .unwrap_or(default_sentiment_order),
            item_code,
            sentiment,
            candidate_rows,
            doc_count,
        });
    }

    rows.sort_by(|left, right| {
        left.filing_year
            .cmp(&right.filing_year)
            .then_with(|| left.item_order.cmp(&right.item_order))
            .then_with(|| left.sentiment_order.cmp(&right.sentiment_order))
    });

    let mut out_rows: Vec<PyObject> = Vec::with_capacity(rows.len());
    for row in rows {
        let out = PyDict::new_bound(py);
        out.set_item("filing_year", row.filing_year)?;
        out.set_item("benchmark_item_code", row.item_code)?;
        out.set_item("sentiment", row.sentiment)?;
        out.set_item("candidate_rows", row.candidate_rows)?;
        out.set_item("doc_count", row.doc_count)?;
        out_rows.push(out.into_py(py));
    }

    Ok(out_rows)
}

#[pyfunction]
pub(crate) fn finbert_selection_keys(
    doc_ids: Vec<String>,
    benchmark_item_codes: Vec<String>,
    seed: i64,
) -> PyResult<Vec<String>> {
    if doc_ids.len() != benchmark_item_codes.len() {
        return Err(PyValueError::new_err(
            "doc_ids and benchmark_item_codes must have the same length",
        ));
    }
    Ok(doc_ids
        .iter()
        .zip(benchmark_item_codes.iter())
        .map(|(doc_id, benchmark_item_code)| {
            finbert_selection_key_impl(doc_id, benchmark_item_code, seed)
        })
        .collect())
}

#[pyfunction]
pub(crate) fn finbert_ranked_selection_indices(
    filing_years: Vec<i64>,
    benchmark_item_codes: Vec<String>,
    target_years: Vec<i64>,
    target_item_codes: Vec<String>,
    target_rows: Vec<i64>,
) -> PyResult<Vec<usize>> {
    if filing_years.len() != benchmark_item_codes.len() {
        return Err(PyValueError::new_err(
            "filing_years and benchmark_item_codes must have the same length",
        ));
    }
    if target_years.len() != target_item_codes.len() || target_years.len() != target_rows.len() {
        return Err(PyValueError::new_err(
            "target_years, target_item_codes, and target_rows must have the same length",
        ));
    }

    let mut quota_map: HashMap<(i64, String), i64> = HashMap::new();
    for ((year, item_code), target) in target_years
        .into_iter()
        .zip(target_item_codes)
        .zip(target_rows)
    {
        if target > 0 {
            quota_map.insert((year, item_code), target);
        }
    }

    let mut selected_indices = Vec::new();
    let mut start = 0usize;
    while start < filing_years.len() {
        let year = filing_years[start];
        let item_code = benchmark_item_codes[start].clone();
        let mut end = start + 1;
        while end < filing_years.len()
            && filing_years[end] == year
            && benchmark_item_codes[end] == item_code
        {
            end += 1;
        }
        if let Some(target) = quota_map.get(&(year, item_code.clone())) {
            let available = end - start;
            let requested = usize::try_from(*target)
                .map_err(|_| PyValueError::new_err("target_rows must be non-negative"))?;
            if requested > available {
                return Err(PyValueError::new_err(format!(
                    "Requested {} rows for ({}, '{}'), but only {} rows are available.",
                    requested, year, item_code, available
                )));
            }
            selected_indices.extend(start..(start + requested));
        }
        start = end;
    }
    Ok(selected_indices)
}

#[pyfunction]
pub(crate) fn finbert_capacity_rows_by_year(
    filing_years: Vec<i64>,
    target_rows: Vec<i64>,
) -> PyResult<Vec<(i64, i64)>> {
    if filing_years.len() != target_rows.len() {
        return Err(PyValueError::new_err(
            "filing_years and target_rows must have the same length",
        ));
    }
    Ok(filing_years.into_iter().zip(target_rows).collect())
}

#[pyfunction]
pub(crate) fn finbert_capacity_rows_by_year_item(
    filing_years: Vec<i64>,
    benchmark_item_codes: Vec<String>,
    target_rows: Vec<i64>,
) -> PyResult<Vec<(i64, String, i64)>> {
    if filing_years.len() != benchmark_item_codes.len() || filing_years.len() != target_rows.len() {
        return Err(PyValueError::new_err(
            "filing_years, benchmark_item_codes, and target_rows must have the same length",
        ));
    }
    Ok(filing_years
        .into_iter()
        .zip(benchmark_item_codes)
        .zip(target_rows)
        .map(|((filing_year, item_code), target_rows)| (filing_year, item_code, target_rows))
        .collect())
}

#[pyfunction]
pub(crate) fn finbert_allocation_targets(
    filing_years: Vec<i64>,
    benchmark_item_codes: Vec<String>,
    target_rows: Vec<i64>,
) -> PyResult<Vec<(i64, String, i64)>> {
    if filing_years.len() != benchmark_item_codes.len() || filing_years.len() != target_rows.len() {
        return Err(PyValueError::new_err(
            "filing_years, benchmark_item_codes, and target_rows must have the same length",
        ));
    }
    Ok(filing_years
        .into_iter()
        .zip(benchmark_item_codes)
        .zip(target_rows)
        .filter_map(|((filing_year, item_code), target_rows)| {
            if target_rows > 0 {
                Some((filing_year, item_code, target_rows))
            } else {
                None
            }
        })
        .collect())
}

#[pyfunction]
pub(crate) fn finbert_dataset_share_rows(
    py: Python<'_>,
    groups: &Bound<'_, PyAny>,
) -> PyResult<Vec<PyObject>> {
    let mut out_rows: Vec<PyObject> = Vec::new();
    for group in groups.iter()? {
        let group = group?;
        let tuple = group
            .downcast::<PyTuple>()
            .map_err(|_| PyValueError::new_err("share-row group is not a tuple"))?;
        if tuple.len() != 3 {
            return Err(PyValueError::new_err(
                "share-row group must contain rows, eligible_total, and selected_total",
            ));
        }
        let rows = tuple.get_item(0)?;
        let eligible_total_value = tuple.get_item(1)?;
        let selected_total_value = tuple.get_item(2)?;
        let eligible_total = py_int_like_to_i64(&eligible_total_value)?;
        let selected_total = py_int_like_to_i64(&selected_total_value)?;
        let eligible_denominator = eligible_total as f64;
        let selected_denominator = selected_total as f64;
        for row in rows.iter()? {
            let row = row?;
            let dict = row
                .downcast::<PyDict>()
                .map_err(|_| PyValueError::new_err("share-row record is not a dict"))?;
            let eligible_rows = dict_required_float(dict, "eligible_rows")?;
            let selected_rows = dict_required_float(dict, "selected_rows")?;
            let eligible_share = if eligible_total != 0 {
                eligible_rows / eligible_denominator
            } else {
                0.0
            };
            let selected_share = if selected_total != 0 {
                selected_rows / selected_denominator
            } else {
                0.0
            };
            let out = copy_py_dict(py, dict)?;
            out.set_item("eligible_share", eligible_share)?;
            out.set_item("selected_share", selected_share)?;
            out.set_item("share_diff", selected_share - eligible_share)?;
            out_rows.push(out.into_py(py));
        }
    }
    Ok(out_rows)
}

#[pyfunction]
pub(crate) fn finbert_dataset_share_row_columns(
    py: Python<'_>,
    groups: &Bound<'_, PyAny>,
) -> PyResult<Vec<PyObject>> {
    let mut out_rows: Vec<PyObject> = Vec::new();
    for group in groups.iter()? {
        let group = group?;
        let tuple = group
            .downcast::<PyTuple>()
            .map_err(|_| PyValueError::new_err("share-row column group is not a tuple"))?;
        if tuple.len() != 4 {
            return Err(PyValueError::new_err(
                "share-row column group must contain column_names, column_values, eligible_total, and selected_total",
            ));
        }
        let column_names_value = tuple.get_item(0)?;
        let column_values = tuple.get_item(1)?;
        let eligible_total_value = tuple.get_item(2)?;
        let selected_total_value = tuple.get_item(3)?;
        let column_names = required_string_sequence(&column_names_value, "share-row column names")?;
        let (columns, row_count) = collect_pyobject_column_values(
            py,
            &column_names,
            &column_values,
            "FinBERT dataset share rows",
        )?;
        let column_index = column_index_by_name(&column_names);
        required_named_column_index(&column_index, "FinBERT dataset share rows", "eligible_rows")?;
        required_named_column_index(&column_index, "FinBERT dataset share rows", "selected_rows")?;
        let eligible_total = py_int_like_to_i64(&eligible_total_value)?;
        let selected_total = py_int_like_to_i64(&selected_total_value)?;
        let eligible_denominator = eligible_total as f64;
        let selected_denominator = selected_total as f64;
        for row_idx in 0..row_count {
            let eligible_rows =
                optional_column_float(py, &columns, &column_index, row_idx, "eligible_rows")?
                    .ok_or_else(|| PyValueError::new_err("null required key: eligible_rows"))?;
            let selected_rows =
                optional_column_float(py, &columns, &column_index, row_idx, "selected_rows")?
                    .ok_or_else(|| PyValueError::new_err("null required key: selected_rows"))?;
            let eligible_share = if eligible_total != 0 {
                eligible_rows / eligible_denominator
            } else {
                0.0
            };
            let selected_share = if selected_total != 0 {
                selected_rows / selected_denominator
            } else {
                0.0
            };
            let out = PyDict::new_bound(py);
            for (column_idx, column_name) in column_names.iter().enumerate() {
                out.set_item(column_name, columns[column_idx][row_idx].bind(py))?;
            }
            out.set_item("eligible_share", eligible_share)?;
            out.set_item("selected_share", selected_share)?;
            out.set_item("share_diff", selected_share - eligible_share)?;
            out_rows.push(out.into_py(py));
        }
    }
    Ok(out_rows)
}

#[pyfunction]
pub(crate) fn finbert_year_allocation_rows(
    filing_years: Vec<i64>,
    eligible_rows: Vec<i64>,
    target_rows: i64,
    ensure_all_years_present: bool,
    capacity_rows: Vec<i64>,
) -> PyResult<Vec<(i64, i64, i64, i64, f64, f64)>> {
    if filing_years.len() != eligible_rows.len() || filing_years.len() != capacity_rows.len() {
        return Err(PyValueError::new_err(
            "filing_years, eligible_rows, and capacity_rows must have the same length",
        ));
    }
    let allocations = finbert_constrained_hamilton_allocations(
        eligible_rows.clone(),
        capacity_rows.clone(),
        target_rows,
        ensure_all_years_present,
    )?;
    let total_eligible: i64 = eligible_rows.iter().sum();
    let mut rows = Vec::with_capacity(filing_years.len());
    for (((filing_year, eligible), capacity), selected) in filing_years
        .into_iter()
        .zip(eligible_rows)
        .zip(capacity_rows)
        .zip(allocations)
    {
        rows.push((
            filing_year,
            eligible,
            capacity,
            selected,
            if total_eligible != 0 {
                eligible as f64 / total_eligible as f64
            } else {
                0.0
            },
            if target_rows != 0 {
                selected as f64 / target_rows as f64
            } else {
                0.0
            },
        ));
    }
    Ok(rows)
}

#[pyfunction]
#[pyo3(signature = (
    filing_years,
    benchmark_item_codes,
    eligible_rows,
    target_years,
    target_rows,
    capacity_years=None,
    capacity_item_codes=None,
    capacity_rows=None
))]
pub(crate) fn finbert_year_item_allocation_rows(
    filing_years: Vec<i64>,
    benchmark_item_codes: Vec<String>,
    eligible_rows: Vec<i64>,
    target_years: Vec<i64>,
    target_rows: Vec<i64>,
    capacity_years: Option<Vec<i64>>,
    capacity_item_codes: Option<Vec<String>>,
    capacity_rows: Option<Vec<i64>>,
) -> PyResult<Vec<(i64, String, i64, i64, i64, f64, f64)>> {
    if filing_years.len() != benchmark_item_codes.len() || filing_years.len() != eligible_rows.len()
    {
        return Err(PyValueError::new_err(
            "filing_years, benchmark_item_codes, and eligible_rows must have the same length",
        ));
    }
    if target_years.len() != target_rows.len() {
        return Err(PyValueError::new_err(
            "target_years and target_rows must have the same length",
        ));
    }

    let target_map: HashMap<i64, i64> = target_years.into_iter().zip(target_rows).collect();
    let capacity_map: Option<HashMap<(i64, String), i64>> = match (
        capacity_years,
        capacity_item_codes,
        capacity_rows,
    ) {
        (None, None, None) => None,
        (Some(years), Some(items), Some(rows)) => {
            if years.len() != items.len() || years.len() != rows.len() {
                return Err(PyValueError::new_err(
                        "capacity_years, capacity_item_codes, and capacity_rows must have the same length",
                    ));
            }
            Some(
                years
                    .into_iter()
                    .zip(items)
                    .zip(rows)
                    .map(|((year, item), rows)| ((year, item), rows))
                    .collect(),
            )
        }
        _ => {
            return Err(PyValueError::new_err(
                "capacity inputs must be all provided or all omitted",
            ));
        }
    };

    let mut count_rows: Vec<(i64, String, i64)> = filing_years
        .into_iter()
        .zip(benchmark_item_codes)
        .zip(eligible_rows)
        .map(|((filing_year, item_code), eligible_rows)| (filing_year, item_code, eligible_rows))
        .collect();
    count_rows.sort_by(|left, right| match left.0.cmp(&right.0) {
        std::cmp::Ordering::Equal => left.1.cmp(&right.1),
        ordering => ordering,
    });

    let mut output: Vec<(i64, String, i64, i64, i64, f64, f64)> =
        Vec::with_capacity(count_rows.len());
    let mut start = 0_usize;
    while start < count_rows.len() {
        let filing_year = count_rows[start].0;
        let mut end = start + 1;
        while end < count_rows.len() && count_rows[end].0 == filing_year {
            end += 1;
        }
        let group = &count_rows[start..end];
        let target = *target_map.get(&filing_year).unwrap_or(&0);
        let item_codes: Vec<String> = group
            .iter()
            .map(|(_, item_code, _)| item_code.clone())
            .collect();
        let weights: Vec<i64> = group.iter().map(|(_, _, eligible)| *eligible).collect();
        let capacities: Vec<i64> = group
            .iter()
            .map(|(_, item_code, eligible)| {
                capacity_map
                    .as_ref()
                    .and_then(|map| map.get(&(filing_year, item_code.clone())).copied())
                    .unwrap_or(*eligible)
            })
            .collect();
        let allocations = finbert_constrained_hamilton_allocations(
            weights.clone(),
            capacities.clone(),
            target,
            false,
        )?;
        let total_eligible: i64 = weights.iter().sum();
        for (((item_code, eligible), capacity), selected) in item_codes
            .into_iter()
            .zip(weights.into_iter())
            .zip(capacities.into_iter())
            .zip(allocations.into_iter())
        {
            output.push((
                filing_year,
                item_code,
                eligible,
                capacity,
                selected,
                if total_eligible != 0 {
                    eligible as f64 / total_eligible as f64
                } else {
                    0.0
                },
                if target != 0 {
                    selected as f64 / target as f64
                } else {
                    0.0
                },
            ));
        }
        start = end;
    }

    Ok(output)
}

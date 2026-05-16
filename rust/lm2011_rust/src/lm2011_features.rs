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

pub(crate) struct Lm2011FeatureSignalSpec {
    stem: String,
    tokens: Vec<String>,
    include_tfidf: bool,
}

pub(crate) fn lm2011_feature_mapping_i64(
    dict: &Bound<'_, PyDict>,
    key: &str,
    default_value: i64,
) -> PyResult<i64> {
    let Some(value) = dict.get_item(key)? else {
        return Ok(default_value);
    };
    if value.is_none() {
        return Ok(default_value);
    }
    py_int_like_to_i64(&value)
}

pub(crate) fn lm2011_feature_mapping_f64(
    dict: &Bound<'_, PyDict>,
    key: &str,
    default_value: f64,
) -> PyResult<f64> {
    let Some(value) = dict.get_item(key)? else {
        return Ok(default_value);
    };
    let Some(number) = normalize_doc_ownership_float_value(Some(&value))? else {
        return Ok(default_value);
    };
    Ok(number)
}

pub(crate) fn lm2011_term_weight_impl(
    term_frequency: i64,
    document_length: i64,
    inverse_document_frequency: f64,
) -> f64 {
    if term_frequency < 1 || document_length < 1 {
        return 0.0;
    }
    ((1.0 + (term_frequency as f64).ln()) / (1.0 + (document_length as f64).ln()))
        * inverse_document_frequency
}

pub(crate) fn lm2011_feature_signal_specs(
    signal_specs: &Bound<'_, PyAny>,
) -> PyResult<Vec<Lm2011FeatureSignalSpec>> {
    let mut specs = Vec::new();
    for spec in signal_specs.iter()? {
        let spec = spec?;
        let tuple = spec
            .downcast::<PyTuple>()
            .map_err(|_| PyValueError::new_err("LM2011 feature signal spec is not a tuple"))?;
        if tuple.len() != 3 {
            return Err(PyValueError::new_err(
                "LM2011 feature signal spec must contain stem, tokens, and include_tfidf",
            ));
        }
        let stem = tuple.get_item(0)?.str()?.to_str()?.to_string();
        let mut tokens = Vec::new();
        for token in tuple.get_item(1)?.iter()? {
            tokens.push(token?.str()?.to_str()?.to_string());
        }
        let include_tfidf = tuple.get_item(2)?.is_truthy()?;
        specs.push(Lm2011FeatureSignalSpec {
            stem,
            tokens,
            include_tfidf,
        });
    }
    Ok(specs)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_lm2011_feature_rows(
    py: Python<'_>,
    base_rows: &Bound<'_, PyAny>,
    doc_token_counts: &Bound<'_, PyDict>,
    doc_token_totals: &Bound<'_, PyDict>,
    doc_recognized_word_totals: &Bound<'_, PyDict>,
    idf_by_token: &Bound<'_, PyDict>,
    token_count_col: &str,
    total_token_count_col: &str,
    signal_specs: &Bound<'_, PyAny>,
) -> PyResult<Vec<PyObject>> {
    let signal_specs = lm2011_feature_signal_specs(signal_specs)?;
    let mut rows = Vec::new();
    for row in base_rows.iter()? {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("LM2011 feature base row is not a dict"))?;
        let out = PyDict::new_bound(py);
        for (key, value) in dict.iter() {
            out.set_item(key, value)?;
        }

        let doc_id = dict_required_string(dict, "doc_id")?;
        let token_total = lm2011_feature_mapping_i64(doc_token_totals, &doc_id, 0)?;
        let recognized_word_total =
            lm2011_feature_mapping_i64(doc_recognized_word_totals, &doc_id, 0)?;
        let counts_dict = match doc_token_counts.get_item(&doc_id)? {
            Some(value) if !value.is_none() => {
                Some(value.downcast_into::<PyDict>().map_err(|_| {
                    PyValueError::new_err("LM2011 feature token-count entry is not a dict")
                })?)
            }
            _ => None,
        };
        let denominator = if token_total > 0 {
            Some(token_total as f64)
        } else {
            None
        };

        out.set_item(total_token_count_col, token_total)?;
        out.set_item(token_count_col, recognized_word_total)?;
        for spec in &signal_specs {
            let mut matched_count = 0.0;
            let mut tfidf = 0.0;
            for token in &spec.tokens {
                let count = match counts_dict.as_ref() {
                    Some(counts) => lm2011_feature_mapping_i64(counts, token, 0)?,
                    None => 0,
                };
                matched_count += count as f64;
                if spec.include_tfidf && count > 0 {
                    let idf = lm2011_feature_mapping_f64(idf_by_token, token, 0.0)?;
                    tfidf += lm2011_term_weight_impl(count, token_total, idf);
                }
            }
            out.set_item(
                format!("{}_prop", spec.stem),
                denominator.map(|value| matched_count / value),
            )?;
            if spec.include_tfidf {
                out.set_item(format!("{}_tfidf", spec.stem), denominator.map(|_| tfidf))?;
            }
        }
        rows.push(out.into_py(py));
    }
    Ok(rows)
}

pub(crate) fn lm2011_counts_json_skip_ws(bytes: &[u8], pos: &mut usize) {
    while *pos < bytes.len() && matches!(bytes[*pos], b' ' | b'\n' | b'\r' | b'\t') {
        *pos += 1;
    }
}

pub(crate) fn lm2011_counts_json_expect(
    bytes: &[u8],
    pos: &mut usize,
    expected: u8,
) -> PyResult<()> {
    if *pos >= bytes.len() || bytes[*pos] != expected {
        return Err(PyValueError::new_err("invalid LM2011 counts JSON"));
    }
    *pos += 1;
    Ok(())
}

pub(crate) fn lm2011_counts_json_string(bytes: &[u8], pos: &mut usize) -> PyResult<String> {
    lm2011_counts_json_expect(bytes, pos, b'"')?;
    let mut out = String::new();
    while *pos < bytes.len() {
        let byte = bytes[*pos];
        *pos += 1;
        match byte {
            b'"' => return Ok(out),
            b'\\' => {
                if *pos >= bytes.len() {
                    return Err(PyValueError::new_err(
                        "unterminated LM2011 counts JSON escape",
                    ));
                }
                let escaped = bytes[*pos];
                *pos += 1;
                match escaped {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'b' => out.push('\u{0008}'),
                    b'f' => out.push('\u{000C}'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'u' => {
                        return Err(PyValueError::new_err(
                            "unsupported LM2011 counts JSON unicode escape",
                        ))
                    }
                    _ => return Err(PyValueError::new_err("invalid LM2011 counts JSON escape")),
                }
            }
            0x00..=0x1F => {
                return Err(PyValueError::new_err(
                    "control character in LM2011 counts JSON key",
                ))
            }
            0x80..=0xFF => return Err(PyValueError::new_err("non-ascii LM2011 counts JSON key")),
            _ => out.push(byte as char),
        }
    }
    Err(PyValueError::new_err(
        "unterminated LM2011 counts JSON string",
    ))
}

pub(crate) fn lm2011_counts_json_i64(bytes: &[u8], pos: &mut usize) -> PyResult<i64> {
    let start = *pos;
    if *pos < bytes.len() && bytes[*pos] == b'-' {
        *pos += 1;
    }
    while *pos < bytes.len() && bytes[*pos].is_ascii_digit() {
        *pos += 1;
    }
    if *pos == start || (bytes[start] == b'-' && *pos == start + 1) {
        return Err(PyValueError::new_err("invalid LM2011 counts JSON integer"));
    }
    let text = std::str::from_utf8(&bytes[start..*pos])
        .map_err(|_| PyValueError::new_err("invalid LM2011 counts JSON integer"))?;
    text.parse::<i64>()
        .map_err(|_| PyValueError::new_err("invalid LM2011 counts JSON integer"))
}

pub(crate) fn lm2011_counts_from_json(text: &str) -> PyResult<HashMap<String, i64>> {
    let bytes = text.as_bytes();
    let mut pos = 0usize;
    let mut counts = HashMap::new();
    lm2011_counts_json_skip_ws(bytes, &mut pos);
    lm2011_counts_json_expect(bytes, &mut pos, b'{')?;
    lm2011_counts_json_skip_ws(bytes, &mut pos);
    if pos < bytes.len() && bytes[pos] == b'}' {
        pos += 1;
        lm2011_counts_json_skip_ws(bytes, &mut pos);
        if pos == bytes.len() {
            return Ok(counts);
        }
        return Err(PyValueError::new_err("trailing LM2011 counts JSON content"));
    }

    loop {
        lm2011_counts_json_skip_ws(bytes, &mut pos);
        let token = lm2011_counts_json_string(bytes, &mut pos)?;
        lm2011_counts_json_skip_ws(bytes, &mut pos);
        lm2011_counts_json_expect(bytes, &mut pos, b':')?;
        lm2011_counts_json_skip_ws(bytes, &mut pos);
        let count = lm2011_counts_json_i64(bytes, &mut pos)?;
        counts.insert(token, count);
        lm2011_counts_json_skip_ws(bytes, &mut pos);
        if pos < bytes.len() && bytes[pos] == b',' {
            pos += 1;
            continue;
        }
        if pos < bytes.len() && bytes[pos] == b'}' {
            pos += 1;
            break;
        }
        return Err(PyValueError::new_err("invalid LM2011 counts JSON object"));
    }
    lm2011_counts_json_skip_ws(bytes, &mut pos);
    if pos != bytes.len() {
        return Err(PyValueError::new_err("trailing LM2011 counts JSON content"));
    }
    Ok(counts)
}

#[pyfunction]
#[pyo3(signature = (
    pass1_rows,
    raw_form_col,
    token_count_col,
    total_token_count_col,
    signal_specs,
    idf_by_token,
    cleaning_policy_id = None
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_lm2011_feature_rows_from_pass1(
    py: Python<'_>,
    pass1_rows: &Bound<'_, PyAny>,
    raw_form_col: &str,
    token_count_col: &str,
    total_token_count_col: &str,
    signal_specs: &Bound<'_, PyAny>,
    idf_by_token: &Bound<'_, PyDict>,
    cleaning_policy_id: Option<&str>,
) -> PyResult<Vec<PyObject>> {
    let signal_specs = lm2011_feature_signal_specs(signal_specs)?;
    let mut rows = Vec::new();
    for row in pass1_rows.iter()? {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("LM2011 pass1 feature row is not a dict"))?;
        let out = PyDict::new_bound(py);
        out.set_item("doc_id", dict_required_py_object(py, dict, "doc_id")?)?;
        out.set_item("cik_10", dict_required_py_object(py, dict, "cik_10")?)?;
        out.set_item(
            "filing_date",
            dict_required_py_object(py, dict, "filing_date")?,
        )?;
        match dict.get_item(raw_form_col)? {
            Some(value) => out.set_item(raw_form_col, value)?,
            None => out.set_item(raw_form_col, Option::<String>::None)?,
        };
        if let Some(value) = dict.get_item("item_id")? {
            out.set_item("item_id", value)?;
        }
        out.set_item(
            "normalized_form",
            dict_required_py_object(py, dict, "normalized_form")?,
        )?;

        let token_total = lm2011_feature_mapping_i64(dict, total_token_count_col, 0)?;
        let recognized_word_total = lm2011_feature_mapping_i64(dict, token_count_col, 0)?;
        let counts_json = match dict.get_item("_matched_counts_json")? {
            Some(value) if !value.is_none() => {
                let rendered = value.str()?;
                let text = rendered.to_str()?;
                if text.is_empty() {
                    "{}".to_string()
                } else {
                    text.to_string()
                }
            }
            _ => "{}".to_string(),
        };
        let counts = lm2011_counts_from_json(&counts_json)?;
        let denominator = if token_total > 0 {
            Some(token_total as f64)
        } else {
            None
        };

        out.set_item(total_token_count_col, token_total)?;
        out.set_item(token_count_col, recognized_word_total)?;
        for spec in &signal_specs {
            let mut matched_count = 0.0;
            let mut tfidf = 0.0;
            for token in &spec.tokens {
                let count = counts.get(token).copied().unwrap_or(0);
                matched_count += count as f64;
                if spec.include_tfidf && count > 0 {
                    let idf = lm2011_feature_mapping_f64(idf_by_token, token, 0.0)?;
                    tfidf += lm2011_term_weight_impl(count, token_total, idf);
                }
            }
            out.set_item(
                format!("{}_prop", spec.stem),
                denominator.map(|value| matched_count / value),
            )?;
            if spec.include_tfidf {
                out.set_item(format!("{}_tfidf", spec.stem), denominator.map(|_| tfidf))?;
            }
        }
        if let Some(policy) = cleaning_policy_id {
            out.set_item("cleaning_policy_id", policy)?;
        }
        rows.push(out.into_py(py));
    }
    Ok(rows)
}

pub(crate) fn column_index_by_name(column_names: &[String]) -> HashMap<String, usize> {
    column_names
        .iter()
        .enumerate()
        .map(|(idx, name)| (name.clone(), idx))
        .collect()
}

pub(crate) fn required_column_index(
    column_index: &HashMap<String, usize>,
    column_name: &str,
) -> PyResult<usize> {
    column_index.get(column_name).copied().ok_or_else(|| {
        PyValueError::new_err(format!(
            "missing required LM2011 pass1 feature column: {column_name}"
        ))
    })
}

pub(crate) fn pyobject_i64_or_default(
    py: Python<'_>,
    value: Option<&PyObject>,
    default_value: i64,
) -> PyResult<i64> {
    let Some(value) = value else {
        return Ok(default_value);
    };
    let value = value.bind(py);
    if value.is_none() {
        return Ok(default_value);
    }
    py_int_like_to_i64(value)
}

pub(crate) fn pyobject_string_or_default(
    py: Python<'_>,
    value: Option<&PyObject>,
    default_value: &str,
) -> PyResult<String> {
    let Some(value) = value else {
        return Ok(default_value.to_string());
    };
    let value = value.bind(py);
    if value.is_none() {
        return Ok(default_value.to_string());
    }
    let rendered = value.str()?;
    let text = rendered.to_str()?;
    if text.is_empty() {
        Ok(default_value.to_string())
    } else {
        Ok(text.to_string())
    }
}

#[pyfunction]
#[pyo3(signature = (
    pass1_column_names,
    pass1_column_values,
    raw_form_col,
    token_count_col,
    total_token_count_col,
    signal_specs,
    idf_by_token,
    cleaning_policy_id = None
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_lm2011_feature_rows_from_pass1_columns(
    py: Python<'_>,
    pass1_column_names: Vec<String>,
    pass1_column_values: &Bound<'_, PyAny>,
    raw_form_col: &str,
    token_count_col: &str,
    total_token_count_col: &str,
    signal_specs: &Bound<'_, PyAny>,
    idf_by_token: &Bound<'_, PyDict>,
    cleaning_policy_id: Option<&str>,
) -> PyResult<Vec<PyObject>> {
    let signal_specs = lm2011_feature_signal_specs(signal_specs)?;
    let mut columns: Vec<Vec<PyObject>> = Vec::new();
    for values in pass1_column_values.iter()? {
        columns.push(pyobject_sequence(py, &values?)?);
    }
    if columns.len() != pass1_column_names.len() {
        return Err(PyValueError::new_err(
            "LM2011 pass1 feature column name/value count mismatch",
        ));
    }
    let row_count = columns.first().map_or(0, Vec::len);
    for (column_idx, column_values) in columns.iter().enumerate() {
        if column_values.len() != row_count {
            let column_name = pass1_column_names
                .get(column_idx)
                .map(String::as_str)
                .unwrap_or("pass1_column");
            return Err(PyValueError::new_err(format!(
                "LM2011 pass1 feature column length mismatch: {column_name}"
            )));
        }
    }

    let column_index = column_index_by_name(&pass1_column_names);
    let doc_id_idx = required_column_index(&column_index, "doc_id")?;
    let cik_idx = required_column_index(&column_index, "cik_10")?;
    let filing_date_idx = required_column_index(&column_index, "filing_date")?;
    let normalized_form_idx = required_column_index(&column_index, "normalized_form")?;
    let total_token_count_idx = column_index.get(total_token_count_col).copied();
    let token_count_idx = column_index.get(token_count_col).copied();
    let counts_json_idx = column_index.get("_matched_counts_json").copied();
    let raw_form_idx = column_index.get(raw_form_col).copied();
    let item_id_idx = column_index.get("item_id").copied();

    let mut rows = Vec::with_capacity(row_count);
    for row_idx in 0..row_count {
        let out = PyDict::new_bound(py);
        out.set_item("doc_id", columns[doc_id_idx][row_idx].clone_ref(py))?;
        out.set_item("cik_10", columns[cik_idx][row_idx].clone_ref(py))?;
        out.set_item(
            "filing_date",
            columns[filing_date_idx][row_idx].clone_ref(py),
        )?;
        match raw_form_idx {
            Some(idx) => out.set_item(raw_form_col, columns[idx][row_idx].clone_ref(py))?,
            None => out.set_item(raw_form_col, Option::<String>::None)?,
        };
        if let Some(idx) = item_id_idx {
            out.set_item("item_id", columns[idx][row_idx].clone_ref(py))?;
        }
        out.set_item(
            "normalized_form",
            columns[normalized_form_idx][row_idx].clone_ref(py),
        )?;

        let token_total = pyobject_i64_or_default(
            py,
            total_token_count_idx.and_then(|idx| columns[idx].get(row_idx)),
            0,
        )?;
        let recognized_word_total = pyobject_i64_or_default(
            py,
            token_count_idx.and_then(|idx| columns[idx].get(row_idx)),
            0,
        )?;
        let counts_json = pyobject_string_or_default(
            py,
            counts_json_idx.and_then(|idx| columns[idx].get(row_idx)),
            "{}",
        )?;
        let counts = lm2011_counts_from_json(&counts_json)?;
        let denominator = if token_total > 0 {
            Some(token_total as f64)
        } else {
            None
        };

        out.set_item(total_token_count_col, token_total)?;
        out.set_item(token_count_col, recognized_word_total)?;
        for spec in &signal_specs {
            let mut matched_count = 0.0;
            let mut tfidf = 0.0;
            for token in &spec.tokens {
                let count = counts.get(token).copied().unwrap_or(0);
                matched_count += count as f64;
                if spec.include_tfidf && count > 0 {
                    let idf = lm2011_feature_mapping_f64(idf_by_token, token, 0.0)?;
                    tfidf += lm2011_term_weight_impl(count, token_total, idf);
                }
            }
            out.set_item(
                format!("{}_prop", spec.stem),
                denominator.map(|value| matched_count / value),
            )?;
            if spec.include_tfidf {
                out.set_item(format!("{}_tfidf", spec.stem), denominator.map(|_| tfidf))?;
            }
        }
        if let Some(policy) = cleaning_policy_id {
            out.set_item("cleaning_policy_id", policy)?;
        }
        rows.push(out.into_py(py));
    }
    Ok(rows)
}

#[pyclass]
pub(crate) struct Lm2011TokenCounter {
    vocabulary: HashSet<String>,
    master_dictionary_words: HashSet<String>,
}

#[pymethods]
impl Lm2011TokenCounter {
    #[new]
    fn new(vocabulary: Vec<String>, master_dictionary_words: Vec<String>) -> Self {
        Self {
            vocabulary: vocabulary.into_iter().collect(),
            master_dictionary_words: master_dictionary_words.into_iter().collect(),
        }
    }

    #[pyo3(signature = (text=None))]
    fn count_document_tokens(&self, text: Option<&str>) -> (usize, usize, HashMap<String, usize>) {
        let mut token_total = 0usize;
        let mut recognized_word_total = 0usize;
        let mut counts: HashMap<String, usize> = HashMap::new();

        scan_tokens_impl(text, |token| {
            token_total += 1;
            if self.master_dictionary_words.contains(&token) {
                recognized_word_total += 1;
            }
            if self.vocabulary.contains(&token) {
                *counts.entry(token).or_insert(0) += 1;
            }
        });

        (token_total, recognized_word_total, counts)
    }
}

#[pyfunction]
pub(crate) fn prepare_lm2011_document_stats(
    py: Python<'_>,
    batches: &Bound<'_, PyAny>,
    text_col: &str,
    vocabulary: Vec<String>,
    master_dictionary_words: Vec<String>,
) -> PyResult<(Vec<PyObject>, PyObject, PyObject, PyObject, PyObject)> {
    let vocabulary: HashSet<String> = vocabulary.into_iter().collect();
    let master_dictionary_words: HashSet<String> = master_dictionary_words.into_iter().collect();
    let mut base_rows = Vec::new();
    let doc_token_counts = PyDict::new_bound(py);
    let doc_token_totals = PyDict::new_bound(py);
    let doc_recognized_word_totals = PyDict::new_bound(py);
    let mut document_frequency: HashMap<String, usize> = HashMap::new();
    let mut num_docs = 0usize;

    for batch in batches.iter()? {
        let batch = batch?;
        for row in batch.iter()? {
            let row = row?;
            let dict = row
                .downcast::<PyDict>()
                .map_err(|_| PyValueError::new_err("LM2011 document-stat row is not a dict"))?;
            let out = PyDict::new_bound(py);
            for (key, value) in dict.iter() {
                if key.str()?.to_str()? != text_col {
                    out.set_item(key, value)?;
                }
            }
            let doc_id = dict_required_string(dict, "doc_id")?;
            let text_value = dict.get_item(text_col)?;
            let text_input = match text_value {
                Some(value) if value.is_instance_of::<PyString>() => {
                    Some(value.str()?.to_str()?.to_string())
                }
                _ => None,
            };
            let mut token_total = 0usize;
            let mut recognized_word_total = 0usize;
            let mut counts: HashMap<String, usize> = HashMap::new();
            scan_tokens_impl(text_input.as_deref(), |token| {
                token_total += 1;
                if master_dictionary_words.contains(&token) {
                    recognized_word_total += 1;
                }
                if vocabulary.contains(&token) {
                    *counts.entry(token).or_insert(0) += 1;
                }
            });
            for token in counts.keys() {
                *document_frequency.entry(token.clone()).or_insert(0) += 1;
            }

            let count_dict = PyDict::new_bound(py);
            for (token, count) in &counts {
                count_dict.set_item(token, *count as i64)?;
            }
            base_rows.push(out.into_py(py));
            doc_token_counts.set_item(&doc_id, count_dict)?;
            doc_token_totals.set_item(&doc_id, token_total as i64)?;
            doc_recognized_word_totals.set_item(&doc_id, recognized_word_total as i64)?;
            num_docs += 1;
        }
    }

    let idf_by_token = PyDict::new_bound(py);
    let denominator = num_docs.max(1) as f64;
    for (token, doc_freq) in document_frequency {
        if doc_freq > 0 {
            idf_by_token.set_item(token, (denominator / (doc_freq as f64)).ln())?;
        }
    }

    Ok((
        base_rows,
        doc_token_counts.into_py(py),
        doc_token_totals.into_py(py),
        doc_recognized_word_totals.into_py(py),
        idf_by_token.into_py(py),
    ))
}

#[pyfunction]
pub(crate) fn prepare_lm2011_document_stats_columns(
    py: Python<'_>,
    batch_meta_column_names: &Bound<'_, PyAny>,
    batch_meta_column_values: &Bound<'_, PyAny>,
    batch_doc_ids: &Bound<'_, PyAny>,
    batch_text_values: &Bound<'_, PyAny>,
    vocabulary: Vec<String>,
    master_dictionary_words: Vec<String>,
) -> PyResult<(Vec<PyObject>, PyObject, PyObject, PyObject, PyObject)> {
    let vocabulary: HashSet<String> = vocabulary.into_iter().collect();
    let master_dictionary_words: HashSet<String> = master_dictionary_words.into_iter().collect();
    let mut base_rows = Vec::new();
    let doc_token_counts = PyDict::new_bound(py);
    let doc_token_totals = PyDict::new_bound(py);
    let doc_recognized_word_totals = PyDict::new_bound(py);
    let mut document_frequency: HashMap<String, usize> = HashMap::new();
    let mut num_docs = 0usize;

    let mut name_batches = Vec::new();
    for names in batch_meta_column_names.iter()? {
        name_batches.push(names?.extract::<Vec<String>>()?);
    }

    let mut value_batches: Vec<Vec<Vec<PyObject>>> = Vec::new();
    for batch_values in batch_meta_column_values.iter()? {
        let batch_values = batch_values?;
        let mut columns = Vec::new();
        for values in batch_values.iter()? {
            columns.push(pyobject_sequence(py, &values?)?);
        }
        value_batches.push(columns);
    }

    let mut doc_id_batches = Vec::new();
    for values in batch_doc_ids.iter()? {
        doc_id_batches.push(required_string_sequence(&values?, "doc_id")?);
    }

    let mut text_batches = Vec::new();
    for values in batch_text_values.iter()? {
        text_batches.push(optional_pystring_sequence(&values?)?);
    }

    let batch_count = name_batches.len();
    if value_batches.len() != batch_count
        || doc_id_batches.len() != batch_count
        || text_batches.len() != batch_count
    {
        return Err(PyValueError::new_err(
            "LM2011 document-stat batch input lengths do not match",
        ));
    }

    for batch_idx in 0..batch_count {
        let meta_names = &name_batches[batch_idx];
        let meta_values = &value_batches[batch_idx];
        let doc_ids = &doc_id_batches[batch_idx];
        let text_values = &text_batches[batch_idx];
        let row_count = text_values.len();
        if doc_ids.len() != row_count {
            return Err(PyValueError::new_err(
                "LM2011 document-stat doc_id length does not match text length",
            ));
        }
        if meta_values.len() != meta_names.len() {
            return Err(PyValueError::new_err(
                "LM2011 document-stat metadata column name/value count mismatch",
            ));
        }
        for (column_idx, column_values) in meta_values.iter().enumerate() {
            if column_values.len() != row_count {
                let column_name = meta_names
                    .get(column_idx)
                    .map(String::as_str)
                    .unwrap_or("metadata");
                return Err(PyValueError::new_err(format!(
                    "LM2011 document-stat metadata column length does not match text length: {column_name}"
                )));
            }
        }

        for row_idx in 0..row_count {
            let out = PyDict::new_bound(py);
            for (column_idx, column_name) in meta_names.iter().enumerate() {
                out.set_item(column_name, meta_values[column_idx][row_idx].clone_ref(py))?;
            }
            let doc_id = &doc_ids[row_idx];
            let mut token_total = 0usize;
            let mut recognized_word_total = 0usize;
            let mut counts: HashMap<String, usize> = HashMap::new();
            scan_tokens_impl(text_values[row_idx].as_deref(), |token| {
                token_total += 1;
                if master_dictionary_words.contains(&token) {
                    recognized_word_total += 1;
                }
                if vocabulary.contains(&token) {
                    *counts.entry(token).or_insert(0) += 1;
                }
            });
            for token in counts.keys() {
                *document_frequency.entry(token.clone()).or_insert(0) += 1;
            }

            let count_dict = PyDict::new_bound(py);
            for (token, count) in &counts {
                count_dict.set_item(token, *count as i64)?;
            }
            base_rows.push(out.into_py(py));
            doc_token_counts.set_item(doc_id, count_dict)?;
            doc_token_totals.set_item(doc_id, token_total as i64)?;
            doc_recognized_word_totals.set_item(doc_id, recognized_word_total as i64)?;
            num_docs += 1;
        }
    }

    let idf_by_token = PyDict::new_bound(py);
    let denominator = num_docs.max(1) as f64;
    for (token, doc_freq) in document_frequency {
        if doc_freq > 0 {
            idf_by_token.set_item(token, (denominator / (doc_freq as f64)).ln())?;
        }
    }

    Ok((
        base_rows,
        doc_token_counts.into_py(py),
        doc_token_totals.into_py(py),
        doc_recognized_word_totals.into_py(py),
        idf_by_token.into_py(py),
    ))
}

pub(crate) fn lm2011_counts_json(counts: &HashMap<String, usize>) -> String {
    if counts.is_empty() {
        return "{}".to_string();
    }
    let ordered: BTreeMap<&String, &usize> = counts.iter().collect();
    let mut out = String::from("{");
    for (idx, (token, count)) in ordered.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        out.push('"');
        for ch in token.chars() {
            match ch {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                _ => out.push(ch),
            }
        }
        out.push_str("\":");
        out.push_str(&count.to_string());
    }
    out.push('}');
    out
}

pub(crate) fn optional_pystring_sequence(
    values: &Bound<'_, PyAny>,
) -> PyResult<Vec<Option<String>>> {
    let mut out = Vec::new();
    for value in values.iter()? {
        let value = value?;
        if value.is_instance_of::<PyString>() {
            out.push(Some(value.str()?.to_str()?.to_string()));
        } else {
            out.push(None);
        }
    }
    Ok(out)
}

pub(crate) fn pyobject_sequence(
    py: Python<'_>,
    values: &Bound<'_, PyAny>,
) -> PyResult<Vec<PyObject>> {
    let mut out = Vec::new();
    for value in values.iter()? {
        out.push(value?.into_py(py));
    }
    Ok(out)
}

#[pyfunction]
pub(crate) fn prepare_lm2011_pass1_rows(
    py: Python<'_>,
    rows: &Bound<'_, PyAny>,
    text_col: &str,
    vocabulary: Vec<String>,
    master_dictionary_words: Vec<String>,
    token_count_col: &str,
    total_token_count_col: &str,
) -> PyResult<(Vec<PyObject>, PyObject)> {
    let vocabulary: HashSet<String> = vocabulary.into_iter().collect();
    let master_dictionary_words: HashSet<String> = master_dictionary_words.into_iter().collect();
    let mut out_rows = Vec::new();
    let mut document_frequency: HashMap<String, usize> = HashMap::new();

    for row in rows.iter()? {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("LM2011 pass1 row is not a dict"))?;
        let out = PyDict::new_bound(py);
        for (key, value) in dict.iter() {
            if key.str()?.to_str()? != text_col {
                out.set_item(key, value)?;
            }
        }

        let text_value = dict.get_item(text_col)?;
        let text_input = match text_value {
            Some(value) if value.is_instance_of::<PyString>() => {
                Some(value.str()?.to_str()?.to_string())
            }
            _ => None,
        };
        let mut token_total = 0usize;
        let mut recognized_word_total = 0usize;
        let mut counts: HashMap<String, usize> = HashMap::new();
        scan_tokens_impl(text_input.as_deref(), |token| {
            token_total += 1;
            if master_dictionary_words.contains(&token) {
                recognized_word_total += 1;
            }
            if vocabulary.contains(&token) {
                *counts.entry(token).or_insert(0) += 1;
            }
        });
        for token in counts.keys() {
            *document_frequency.entry(token.clone()).or_insert(0) += 1;
        }

        out.set_item(total_token_count_col, token_total as i64)?;
        out.set_item(token_count_col, recognized_word_total as i64)?;
        out.set_item("_matched_counts_json", lm2011_counts_json(&counts))?;
        out_rows.push(out.into_py(py));
    }

    let frequency_dict = PyDict::new_bound(py);
    for (token, count) in document_frequency {
        frequency_dict.set_item(token, count as i64)?;
    }
    Ok((out_rows, frequency_dict.into_py(py)))
}

#[pyfunction]
pub(crate) fn prepare_lm2011_pass1_columns(
    py: Python<'_>,
    meta_column_names: Vec<String>,
    meta_column_values: &Bound<'_, PyAny>,
    text_values: &Bound<'_, PyAny>,
    vocabulary: Vec<String>,
    master_dictionary_words: Vec<String>,
    token_count_col: &str,
    total_token_count_col: &str,
) -> PyResult<(Vec<PyObject>, PyObject)> {
    let text_values = optional_pystring_sequence(text_values)?;
    let row_count = text_values.len();
    let mut meta_values: Vec<Vec<PyObject>> = Vec::new();
    for (idx, values) in meta_column_values.iter()?.enumerate() {
        let values = values?;
        let column_values = pyobject_sequence(py, &values)?;
        if column_values.len() != row_count {
            let column_name = meta_column_names
                .get(idx)
                .map(String::as_str)
                .unwrap_or("metadata");
            return Err(PyValueError::new_err(format!(
                "LM2011 pass1 metadata column length does not match text length: {column_name}"
            )));
        }
        meta_values.push(column_values);
    }
    if meta_values.len() != meta_column_names.len() {
        return Err(PyValueError::new_err(
            "LM2011 pass1 metadata column name/value count mismatch",
        ));
    }

    let vocabulary: HashSet<String> = vocabulary.into_iter().collect();
    let master_dictionary_words: HashSet<String> = master_dictionary_words.into_iter().collect();
    let mut out_rows = Vec::with_capacity(row_count);
    let mut document_frequency: HashMap<String, usize> = HashMap::new();

    for row_idx in 0..row_count {
        let out = PyDict::new_bound(py);
        for (column_idx, column_name) in meta_column_names.iter().enumerate() {
            out.set_item(column_name, meta_values[column_idx][row_idx].clone_ref(py))?;
        }

        let mut token_total = 0usize;
        let mut recognized_word_total = 0usize;
        let mut counts: HashMap<String, usize> = HashMap::new();
        scan_tokens_impl(text_values[row_idx].as_deref(), |token| {
            token_total += 1;
            if master_dictionary_words.contains(&token) {
                recognized_word_total += 1;
            }
            if vocabulary.contains(&token) {
                *counts.entry(token).or_insert(0) += 1;
            }
        });
        for token in counts.keys() {
            *document_frequency.entry(token.clone()).or_insert(0) += 1;
        }

        out.set_item(total_token_count_col, token_total as i64)?;
        out.set_item(token_count_col, recognized_word_total as i64)?;
        out.set_item("_matched_counts_json", lm2011_counts_json(&counts))?;
        out_rows.push(out.into_py(py));
    }

    let frequency_dict = PyDict::new_bound(py);
    for (token, count) in document_frequency {
        frequency_dict.set_item(token, count as i64)?;
    }
    Ok((out_rows, frequency_dict.into_py(py)))
}

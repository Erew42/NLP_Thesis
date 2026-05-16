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
use crate::multisurface_audit::*;
use crate::refinitiv_analyst::*;
use crate::refinitiv_authority::*;
use crate::refinitiv_bridge::*;
use crate::refinitiv_excel::*;
use crate::sec_extraction::*;
use crate::sentence_cleaning::*;
use crate::sentence_quality_api::*;

#[pyfunction]
pub(crate) fn stable_hash_id_simple(prefix: &str, parts: &Bound<'_, PyAny>) -> PyResult<String> {
    let mut payload = String::new();
    payload.push('[');
    let mut first = true;
    for part in parts.iter()? {
        if first {
            first = false;
        } else {
            payload.push(',');
        }
        push_stable_json_simple_part(&mut payload, &part?)?;
    }
    payload.push(']');
    Ok(stable_hash_id_from_payload(prefix, &payload))
}

#[pyfunction]
#[pyo3(signature = (stage, fields, parameters=None, excluded_parameter_keys=None))]
pub(crate) fn lseg_request_signature_value(
    stage: &str,
    fields: &Bound<'_, PyAny>,
    parameters: Option<&Bound<'_, PyAny>>,
    excluded_parameter_keys: Option<&Bound<'_, PyAny>>,
) -> PyResult<String> {
    let mut payload = String::new();
    push_request_signature_payload(
        &mut payload,
        stage,
        fields,
        parameters,
        excluded_parameter_keys,
    )?;
    Ok(stable_hash_id_from_payload("sig", &payload))
}

#[derive(Clone)]
pub(crate) struct BatchItemIndexRow {
    item_index: usize,
    item_id: String,
    instrument: String,
}

pub(crate) struct BatchItemIndexGroup {
    items: Vec<BatchItemIndexRow>,
}

#[pyfunction]
pub(crate) fn lseg_batch_item_index_groups(
    item_rows: &Bound<'_, PyAny>,
    max_batch_size: isize,
    unique_instrument_limit: bool,
) -> PyResult<Vec<Vec<usize>>> {
    if max_batch_size <= 0 {
        return Err(PyValueError::new_err("max_batch_size must be positive"));
    }
    let max_batch_size = max_batch_size as usize;
    let mut group_positions: HashMap<(String, String, String), usize> = HashMap::new();
    let mut groups: Vec<BatchItemIndexGroup> = Vec::new();
    for item_row in item_rows.iter()? {
        let item_row = item_row?;
        let dict = item_row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("batch item row is not a dict"))?;
        let Some(item_index_value) = dict.get_item("item_index")? else {
            return Err(PyValueError::new_err("missing item_index"));
        };
        let item_index = item_index_value.extract::<usize>()?;
        let key = (
            dict_required_string(dict, "stage")?,
            dict_required_string(dict, "batch_key")?,
            dict_required_string(dict, "signature")?,
        );
        let row = BatchItemIndexRow {
            item_index,
            item_id: dict_required_string(dict, "item_id")?,
            instrument: dict_required_string(dict, "instrument")?,
        };
        if let Some(group_index) = group_positions.get(&key) {
            groups[*group_index].items.push(row);
        } else {
            group_positions.insert(key, groups.len());
            groups.push(BatchItemIndexGroup { items: vec![row] });
        }
    }

    let mut out: Vec<Vec<usize>> = Vec::new();
    for group in groups.iter_mut() {
        group
            .items
            .sort_by(|left, right| left.item_id.cmp(&right.item_id));
        if unique_instrument_limit {
            let mut current: Vec<usize> = Vec::new();
            let mut current_instruments: HashSet<String> = HashSet::new();
            for item in group.items.iter() {
                if !current.is_empty()
                    && !current_instruments.contains(&item.instrument)
                    && current_instruments.len() >= max_batch_size
                {
                    out.push(current);
                    current = Vec::new();
                    current_instruments = HashSet::new();
                }
                current.push(item.item_index);
                current_instruments.insert(item.instrument.clone());
            }
            if !current.is_empty() {
                out.push(current);
            }
        } else {
            for chunk in group.items.chunks(max_batch_size) {
                out.push(chunk.iter().map(|item| item.item_index).collect());
            }
        }
    }
    Ok(out)
}

#[pyfunction]
pub(crate) fn lseg_frame_fingerprint_rows(rows: &Bound<'_, PyAny>) -> PyResult<String> {
    let mut iterator = rows.iter()?;
    let Some(first_row) = iterator.next() else {
        return Ok(sha1_hex_from_bytes(b"empty"));
    };
    let mut payload = String::new();
    payload.push('[');
    push_json_default_str_part(&mut payload, &first_row?)?;
    for row in iterator {
        payload.push(',');
        push_json_default_str_part(&mut payload, &row?)?;
    }
    payload.push(']');
    Ok(sha1_hex_from_bytes(payload.as_bytes()))
}

#[pyfunction]
pub(crate) fn lseg_frame_fingerprint_columns(
    py: Python<'_>,
    column_names: &Bound<'_, PyAny>,
    columns: &Bound<'_, PyAny>,
) -> PyResult<String> {
    let column_names = required_string_sequence(column_names, "column_names")?;
    let mut column_values: Vec<Vec<PyObject>> = Vec::with_capacity(column_names.len());
    let mut row_count: Option<usize> = None;
    for column in columns.iter()? {
        let column = column?;
        let mut values: Vec<PyObject> = Vec::new();
        for value in column.iter()? {
            values.push(value?.clone().into_py(py));
        }
        match row_count {
            Some(expected) if values.len() != expected => {
                return Err(PyValueError::new_err(
                    "all frame fingerprint columns must have the same length",
                ))
            }
            None => row_count = Some(values.len()),
            _ => {}
        }
        column_values.push(values);
    }
    if column_values.len() != column_names.len() {
        return Err(PyValueError::new_err(
            "frame fingerprint column count does not match column_names length",
        ));
    }
    let row_count = row_count.unwrap_or(0);
    if row_count == 0 {
        return Ok(sha1_hex_from_bytes(b"empty"));
    }

    let mut sorted_indices: Vec<usize> = (0..column_names.len()).collect();
    sorted_indices.sort_by(|left, right| column_names[*left].cmp(&column_names[*right]));
    let mut payload = String::new();
    payload.push('[');
    for row_idx in 0..row_count {
        if row_idx > 0 {
            payload.push(',');
        }
        payload.push('{');
        for (sorted_idx, column_idx) in sorted_indices.iter().enumerate() {
            if sorted_idx > 0 {
                payload.push(',');
            }
            push_json_ascii_string(&mut payload, &column_names[*column_idx])?;
            payload.push(':');
            push_json_default_str_part(&mut payload, column_values[*column_idx][row_idx].bind(py))?;
        }
        payload.push('}');
    }
    payload.push(']');
    Ok(sha1_hex_from_bytes(payload.as_bytes()))
}

#[pyfunction]
pub(crate) fn sample_scope_coverage_doc_ids(
    py: Python<'_>,
    doc_ids: &Bound<'_, PyAny>,
    item_codes_by_row: &Bound<'_, PyAny>,
    item_codes: Vec<String>,
    sample_doc_count: isize,
    seed: i64,
) -> PyResult<Vec<String>> {
    let mut unique_doc_ids: BTreeSet<String> = BTreeSet::new();
    let mut docs_by_item: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut doc_iter = doc_ids.iter()?;
    let mut item_iter = item_codes_by_row.iter()?;

    loop {
        let doc_next = doc_iter.next();
        let item_next = item_iter.next();
        match (doc_next, item_next) {
            (None, None) => break,
            (Some(_), None) | (None, Some(_)) => {
                return Err(PyValueError::new_err(
                    "doc_ids and item_codes_by_row must have the same length",
                ));
            }
            (Some(doc_value), Some(item_value)) => {
                let doc_value = doc_value?;
                let item_value = item_value?;
                let doc_id = doc_value.str()?.to_str()?.to_string();
                unique_doc_ids.insert(doc_id.clone());
                if !item_value.is_none() {
                    let item_code = item_value.str()?.to_str()?.to_string();
                    docs_by_item.entry(item_code).or_default().insert(doc_id);
                }
            }
        }
    }

    if unique_doc_ids.is_empty() {
        return Ok(Vec::new());
    }
    let all_doc_ids: Vec<String> = unique_doc_ids.iter().cloned().collect();
    if sample_doc_count <= 0 {
        return Ok(Vec::new());
    }
    let target_count = sample_doc_count as usize;
    if target_count >= all_doc_ids.len() {
        return Ok(all_doc_ids);
    }

    let random_cls = py.import_bound("random")?.getattr("Random")?;
    let mut selected: Vec<String> = Vec::new();
    let mut selected_set: HashSet<String> = HashSet::new();
    for (offset, item_code) in item_codes.iter().enumerate() {
        if selected.len() >= target_count {
            break;
        }
        let Some(candidate_set) = docs_by_item.get(item_code) else {
            continue;
        };
        let candidates: Vec<String> = candidate_set
            .iter()
            .filter(|doc_id| !selected_set.contains(*doc_id))
            .cloned()
            .collect();
        if candidates.is_empty() {
            continue;
        }
        let rng = random_cls.call1((seed + offset as i64,))?;
        let candidates_list = PyList::new_bound(py, candidates);
        let chosen = rng.call_method1("choice", (candidates_list,))?;
        let chosen_doc_id = chosen.str()?.to_str()?.to_string();
        selected_set.insert(chosen_doc_id.clone());
        selected.push(chosen_doc_id);
    }

    let remaining_doc_ids: Vec<String> = all_doc_ids
        .into_iter()
        .filter(|doc_id| !selected_set.contains(doc_id))
        .collect();
    let rng = random_cls.call1((seed + 10_003,))?;
    let remaining_list = PyList::new_bound(py, remaining_doc_ids);
    rng.call_method1("shuffle", (&remaining_list,))?;
    let shuffled_remaining: Vec<String> = remaining_list.extract()?;
    let needed = target_count.saturating_sub(selected.len());
    selected.extend(shuffled_remaining.into_iter().take(needed));
    Ok(selected)
}

pub(crate) fn json_skip_ws(bytes: &[u8], pos: &mut usize) {
    while *pos < bytes.len() && matches!(bytes[*pos], b' ' | b'\n' | b'\r' | b'\t') {
        *pos += 1;
    }
}

pub(crate) fn json_parse_hex4(bytes: &[u8], pos: &mut usize) -> PyResult<u16> {
    if *pos + 4 > bytes.len() {
        return Err(PyValueError::new_err("truncated JSON unicode escape"));
    }
    let mut value = 0u16;
    for _ in 0..4 {
        value <<= 4;
        value |= match bytes[*pos] {
            b'0'..=b'9' => u16::from(bytes[*pos] - b'0'),
            b'a'..=b'f' => u16::from(bytes[*pos] - b'a' + 10),
            b'A'..=b'F' => u16::from(bytes[*pos] - b'A' + 10),
            _ => return Err(PyValueError::new_err("invalid JSON unicode escape")),
        };
        *pos += 1;
    }
    Ok(value)
}

pub(crate) fn json_push_codepoint(out: &mut Vec<u8>, codepoint: u32) -> PyResult<()> {
    let Some(ch) = char::from_u32(codepoint) else {
        return Err(PyValueError::new_err("invalid JSON unicode codepoint"));
    };
    let mut buffer = [0u8; 4];
    out.extend_from_slice(ch.encode_utf8(&mut buffer).as_bytes());
    Ok(())
}

pub(crate) fn json_parse_string(bytes: &[u8], pos: &mut usize) -> PyResult<String> {
    if bytes.get(*pos) != Some(&b'"') {
        return Err(PyValueError::new_err("expected JSON string"));
    }
    *pos += 1;
    let mut out: Vec<u8> = Vec::new();
    while *pos < bytes.len() {
        let byte = bytes[*pos];
        *pos += 1;
        match byte {
            b'"' => {
                return String::from_utf8(out)
                    .map_err(|_| PyValueError::new_err("invalid UTF-8 in JSON string"));
            }
            b'\\' => {
                if *pos >= bytes.len() {
                    return Err(PyValueError::new_err("truncated JSON escape"));
                }
                let escaped = bytes[*pos];
                *pos += 1;
                match escaped {
                    b'"' => out.push(b'"'),
                    b'\\' => out.push(b'\\'),
                    b'/' => out.push(b'/'),
                    b'b' => out.push(0x08),
                    b'f' => out.push(0x0c),
                    b'n' => out.push(b'\n'),
                    b'r' => out.push(b'\r'),
                    b't' => out.push(b'\t'),
                    b'u' => {
                        let first = json_parse_hex4(bytes, pos)?;
                        if (0xD800..=0xDBFF).contains(&first) {
                            if *pos + 6 > bytes.len()
                                || bytes[*pos] != b'\\'
                                || bytes[*pos + 1] != b'u'
                            {
                                return Err(PyValueError::new_err(
                                    "invalid JSON unicode surrogate",
                                ));
                            }
                            *pos += 2;
                            let second = json_parse_hex4(bytes, pos)?;
                            if !(0xDC00..=0xDFFF).contains(&second) {
                                return Err(PyValueError::new_err(
                                    "invalid JSON unicode surrogate",
                                ));
                            }
                            let codepoint = 0x10000
                                + (((u32::from(first) - 0xD800) << 10)
                                    | (u32::from(second) - 0xDC00));
                            json_push_codepoint(&mut out, codepoint)?;
                        } else if (0xDC00..=0xDFFF).contains(&first) {
                            return Err(PyValueError::new_err("invalid JSON unicode surrogate"));
                        } else {
                            json_push_codepoint(&mut out, u32::from(first))?;
                        }
                    }
                    _ => return Err(PyValueError::new_err("invalid JSON string escape")),
                }
            }
            0x00..=0x1f => return Err(PyValueError::new_err("invalid JSON control character")),
            _ => out.push(byte),
        }
    }
    Err(PyValueError::new_err("unterminated JSON string"))
}

pub(crate) fn json_consume_literal(bytes: &[u8], pos: &mut usize, literal: &[u8]) -> PyResult<()> {
    if bytes.get(*pos..*pos + literal.len()) != Some(literal) {
        return Err(PyValueError::new_err("invalid JSON literal"));
    }
    *pos += literal.len();
    Ok(())
}

pub(crate) fn json_skip_number(bytes: &[u8], pos: &mut usize) -> PyResult<()> {
    if bytes.get(*pos) == Some(&b'-') {
        *pos += 1;
    }
    match bytes.get(*pos).copied() {
        Some(b'0') => *pos += 1,
        Some(b'1'..=b'9') => {
            *pos += 1;
            while matches!(bytes.get(*pos).copied(), Some(b'0'..=b'9')) {
                *pos += 1;
            }
        }
        _ => return Err(PyValueError::new_err("invalid JSON number")),
    }
    if bytes.get(*pos) == Some(&b'.') {
        *pos += 1;
        let digit_start = *pos;
        while matches!(bytes.get(*pos).copied(), Some(b'0'..=b'9')) {
            *pos += 1;
        }
        if *pos == digit_start {
            return Err(PyValueError::new_err("invalid JSON number"));
        }
    }
    if matches!(bytes.get(*pos).copied(), Some(b'e' | b'E')) {
        *pos += 1;
        if matches!(bytes.get(*pos).copied(), Some(b'+' | b'-')) {
            *pos += 1;
        }
        let digit_start = *pos;
        while matches!(bytes.get(*pos).copied(), Some(b'0'..=b'9')) {
            *pos += 1;
        }
        if *pos == digit_start {
            return Err(PyValueError::new_err("invalid JSON number"));
        }
    }
    Ok(())
}

pub(crate) fn json_skip_value(bytes: &[u8], pos: &mut usize) -> PyResult<()> {
    json_skip_ws(bytes, pos);
    match bytes.get(*pos).copied() {
        Some(b'"') => {
            json_parse_string(bytes, pos)?;
            Ok(())
        }
        Some(b'{') => {
            *pos += 1;
            json_skip_ws(bytes, pos);
            if bytes.get(*pos) == Some(&b'}') {
                *pos += 1;
                return Ok(());
            }
            loop {
                json_skip_ws(bytes, pos);
                json_parse_string(bytes, pos)?;
                json_skip_ws(bytes, pos);
                if bytes.get(*pos) != Some(&b':') {
                    return Err(PyValueError::new_err("expected JSON object colon"));
                }
                *pos += 1;
                json_skip_value(bytes, pos)?;
                json_skip_ws(bytes, pos);
                match bytes.get(*pos).copied() {
                    Some(b',') => *pos += 1,
                    Some(b'}') => {
                        *pos += 1;
                        return Ok(());
                    }
                    _ => return Err(PyValueError::new_err("expected JSON object delimiter")),
                }
            }
        }
        Some(b'[') => {
            *pos += 1;
            json_skip_ws(bytes, pos);
            if bytes.get(*pos) == Some(&b']') {
                *pos += 1;
                return Ok(());
            }
            loop {
                json_skip_value(bytes, pos)?;
                json_skip_ws(bytes, pos);
                match bytes.get(*pos).copied() {
                    Some(b',') => *pos += 1,
                    Some(b']') => {
                        *pos += 1;
                        return Ok(());
                    }
                    _ => return Err(PyValueError::new_err("expected JSON array delimiter")),
                }
            }
        }
        Some(b't') => json_consume_literal(bytes, pos, b"true"),
        Some(b'f') => json_consume_literal(bytes, pos, b"false"),
        Some(b'n') => json_consume_literal(bytes, pos, b"null"),
        Some(b'-' | b'0'..=b'9') => json_skip_number(bytes, pos),
        _ => Err(PyValueError::new_err("unsupported JSON value")),
    }
}

pub(crate) fn json_top_level_event_matches(line: &str, event_name: &str) -> PyResult<bool> {
    let bytes = line.as_bytes();
    let mut pos = 0usize;
    json_skip_ws(bytes, &mut pos);
    if bytes.get(pos) != Some(&b'{') {
        return Err(PyValueError::new_err(
            "request log line is not a JSON object",
        ));
    }
    pos += 1;
    let mut event_matches: Option<bool> = None;
    json_skip_ws(bytes, &mut pos);
    if bytes.get(pos) == Some(&b'}') {
        pos += 1;
    } else {
        loop {
            json_skip_ws(bytes, &mut pos);
            let key = json_parse_string(bytes, &mut pos)?;
            json_skip_ws(bytes, &mut pos);
            if bytes.get(pos) != Some(&b':') {
                return Err(PyValueError::new_err("expected JSON object colon"));
            }
            pos += 1;
            json_skip_ws(bytes, &mut pos);
            if key == "event" {
                if bytes.get(pos) == Some(&b'"') {
                    event_matches = Some(json_parse_string(bytes, &mut pos)? == event_name);
                } else {
                    json_skip_value(bytes, &mut pos)?;
                    event_matches = Some(false);
                }
            } else {
                json_skip_value(bytes, &mut pos)?;
            }
            json_skip_ws(bytes, &mut pos);
            match bytes.get(pos).copied() {
                Some(b',') => pos += 1,
                Some(b'}') => {
                    pos += 1;
                    break;
                }
                _ => return Err(PyValueError::new_err("expected JSON object delimiter")),
            }
        }
    }
    json_skip_ws(bytes, &mut pos);
    if pos != bytes.len() {
        return Err(PyValueError::new_err("trailing content after JSON object"));
    }
    Ok(event_matches.unwrap_or(false))
}

#[pyfunction]
pub(crate) fn lseg_count_request_log_events(
    request_log_path: &str,
    event_name: &str,
) -> PyResult<usize> {
    let file = File::open(request_log_path)
        .map_err(|err| PyOSError::new_err(format!("failed to open request log: {err}")))?;
    let reader = BufReader::new(file);
    let mut count = 0usize;
    for line_result in reader.lines() {
        let line = line_result
            .map_err(|err| PyOSError::new_err(format!("failed to read request log: {err}")))?;
        if line.trim().is_empty() {
            continue;
        }
        if json_top_level_event_matches(&line, event_name)? {
            count += 1;
        }
    }
    Ok(count)
}

pub(crate) fn parse_json_string_array(
    array_json: &str,
    label: &'static str,
) -> PyResult<Vec<String>> {
    let bytes = array_json.as_bytes();
    let mut pos = 0usize;
    json_skip_ws(bytes, &mut pos);
    if bytes.get(pos) != Some(&b'[') {
        return Err(PyValueError::new_err(format!(
            "{label} is not a JSON array"
        )));
    }
    pos += 1;
    let mut values: Vec<String> = Vec::new();
    json_skip_ws(bytes, &mut pos);
    if bytes.get(pos) == Some(&b']') {
        pos += 1;
    } else {
        loop {
            json_skip_ws(bytes, &mut pos);
            if bytes.get(pos) != Some(&b'"') {
                return Err(PyValueError::new_err(
                    "non-string JSON array entry falls back to Python",
                ));
            }
            values.push(json_parse_string(bytes, &mut pos)?);
            json_skip_ws(bytes, &mut pos);
            match bytes.get(pos).copied() {
                Some(b',') => pos += 1,
                Some(b']') => {
                    pos += 1;
                    break;
                }
                _ => return Err(PyValueError::new_err("expected JSON array delimiter")),
            }
        }
    }
    json_skip_ws(bytes, &mut pos);
    if pos != bytes.len() {
        return Err(PyValueError::new_err("trailing content after JSON array"));
    }
    Ok(values)
}

#[pyfunction]
pub(crate) fn lseg_string_array_json_values(array_json: &str) -> PyResult<Vec<String>> {
    parse_json_string_array(array_json, "JSON value")
}

#[pyfunction]
pub(crate) fn lseg_ledger_item_ids_json_values(item_ids_json: &str) -> PyResult<Vec<String>> {
    parse_json_string_array(item_ids_json, "item_ids_json")
}

#[pyfunction]
pub(crate) fn lseg_row_count_by_item_id(
    rows: &Bound<'_, PyAny>,
) -> PyResult<Vec<(Option<String>, i64)>> {
    let mut counts: BTreeMap<Option<String>, i64> = BTreeMap::new();
    for row in rows.iter()? {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("LSEG response row is not a dict"))?;
        let item_id = match dict.get_item("item_id")? {
            None => None,
            Some(value) if value.is_none() => None,
            Some(value) if value.is_instance_of::<PyString>() => Some(value.extract::<String>()?),
            Some(_) => {
                return Err(PyValueError::new_err(
                    "non-string item_id falls back to Python",
                ));
            }
        };
        *counts.entry(item_id).or_insert(0) += 1;
    }
    Ok(counts.into_iter().collect())
}

#[pyfunction]
pub(crate) fn lseg_row_count_by_item_id_values(
    values: &Bound<'_, PyAny>,
) -> PyResult<Vec<(Option<String>, i64)>> {
    let mut counts: BTreeMap<Option<String>, i64> = BTreeMap::new();
    for value in values.iter()? {
        let value = value?;
        let item_id = if value.is_none() {
            None
        } else if value.is_instance_of::<PyString>() {
            Some(value.extract::<String>()?)
        } else {
            return Err(PyValueError::new_err(
                "non-string item_id falls back to Python",
            ));
        };
        *counts.entry(item_id).or_insert(0) += 1;
    }
    Ok(counts.into_iter().collect())
}

pub(crate) fn dict_python_or_zero_i64(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<i64> {
    let Some(value) = dict.get_item(key)? else {
        return Ok(0);
    };
    if value.is_none() || !value.is_truthy()? {
        return Ok(0);
    }
    py_int_like_to_i64(&value)
}

#[pyfunction]
pub(crate) fn lseg_should_requeue_mixed_zero_positive_success(
    stage: &str,
    item_results: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    if stage != "ownership_universe" {
        return Ok(false);
    }
    let mut result_count = 0usize;
    let mut has_positive = false;
    let mut has_zero = false;
    for row in item_results.iter()? {
        result_count += 1;
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("item result row is not a dict"))?;
        let row_count = dict_python_or_zero_i64(dict, "row_count")?;
        if row_count > 0 {
            has_positive = true;
        }
        if row_count == 0 {
            has_zero = true;
        }
    }
    Ok(result_count > 1 && has_positive && has_zero)
}

pub(crate) fn lseg_attr_string(value: &Bound<'_, PyAny>, attr: &str) -> PyResult<String> {
    value.getattr(attr)?.str()?.to_str().map(str::to_string)
}

pub(crate) fn lseg_batch_id_from_items(
    stage: &str,
    batch_key: &str,
    item_ids: &[String],
) -> PyResult<String> {
    let mut payload = String::new();
    payload.push('[');
    push_json_ascii_string(&mut payload, stage)?;
    payload.push(',');
    push_json_ascii_string(&mut payload, batch_key)?;
    payload.push(',');
    payload.push('[');
    for (index, item_id) in item_ids.iter().enumerate() {
        if index > 0 {
            payload.push(',');
        }
        push_json_ascii_string(&mut payload, item_id)?;
    }
    payload.push(']');
    payload.push(']');
    Ok(stable_hash_id_from_payload("batch", &payload))
}

pub(crate) fn lseg_child_batch_id(parent_batch_id: &str, suffix: &str) -> PyResult<String> {
    let mut payload = String::new();
    payload.push('[');
    push_json_ascii_string(&mut payload, parent_batch_id)?;
    payload.push(',');
    push_json_ascii_string(&mut payload, suffix)?;
    payload.push(']');
    Ok(stable_hash_id_from_payload("batch", &payload))
}

pub(crate) fn lseg_child_batch_row(
    py: Python<'_>,
    batch_id: String,
    stage: &str,
    batch_key: &str,
    fields: &Bound<'_, PyAny>,
    parameters: &Bound<'_, PyAny>,
    item_ids: Vec<String>,
    instruments: Vec<String>,
) -> PyResult<PyObject> {
    let out = PyDict::new_bound(py);
    out.set_item("batch_id", batch_id)?;
    out.set_item("stage", stage)?;
    out.set_item("batch_key", batch_key)?;
    out.set_item("fields", fields)?;
    out.set_item("parameters", parameters)?;
    out.set_item("item_ids", item_ids)?;
    out.set_item("instruments", instruments)?;
    Ok(out.into_py(py))
}

#[pyfunction]
pub(crate) fn lseg_singleton_child_batch_rows(
    py: Python<'_>,
    batch_items_rows: &Bound<'_, PyAny>,
) -> PyResult<Vec<PyObject>> {
    let mut out = Vec::new();
    for item in batch_items_rows.iter()? {
        let item = item?;
        let item_id = lseg_attr_string(&item, "item_id")?;
        let stage = lseg_attr_string(&item, "stage")?;
        let instrument = lseg_attr_string(&item, "instrument")?;
        let batch_key = lseg_attr_string(&item, "batch_key")?;
        let fields = item.getattr("fields")?;
        let parameters = item.getattr("parameters")?;
        let item_ids = vec![item_id];
        let instruments = vec![instrument];
        let batch_id = lseg_batch_id_from_items(&stage, &batch_key, &item_ids)?;
        out.push(lseg_child_batch_row(
            py,
            batch_id,
            &stage,
            &batch_key,
            &fields,
            &parameters,
            item_ids,
            instruments,
        )?);
    }
    Ok(out)
}

#[pyfunction]
pub(crate) fn lseg_split_child_batch_rows(
    py: Python<'_>,
    batch: &Bound<'_, PyAny>,
    batch_items_rows: &Bound<'_, PyAny>,
) -> PyResult<Vec<PyObject>> {
    let batch_id = lseg_attr_string(batch, "batch_id")?;
    let stage = lseg_attr_string(batch, "stage")?;
    let batch_key = lseg_attr_string(batch, "batch_key")?;
    let fields = batch.getattr("fields")?;
    let parameters = batch.getattr("parameters")?;
    let mut item_ids: Vec<String> = Vec::new();
    let mut instruments: Vec<String> = Vec::new();
    for item in batch_items_rows.iter()? {
        let item = item?;
        item_ids.push(lseg_attr_string(&item, "item_id")?);
        instruments.push(lseg_attr_string(&item, "instrument")?);
    }
    if item_ids.len() <= 1 {
        return Ok(vec![lseg_child_batch_row(
            py,
            batch_id,
            &stage,
            &batch_key,
            &fields,
            &parameters,
            item_ids,
            instruments,
        )?]);
    }
    let midpoint = std::cmp::max(1usize, item_ids.len() / 2);
    let left_item_ids = item_ids[..midpoint].to_vec();
    let right_item_ids = item_ids[midpoint..].to_vec();
    let left_instruments = instruments[..midpoint].to_vec();
    let right_instruments = instruments[midpoint..].to_vec();
    Ok(vec![
        lseg_child_batch_row(
            py,
            lseg_child_batch_id(&batch_id, "a")?,
            &stage,
            &batch_key,
            &fields,
            &parameters,
            left_item_ids,
            left_instruments,
        )?,
        lseg_child_batch_row(
            py,
            lseg_child_batch_id(&batch_id, "b")?,
            &stage,
            &batch_key,
            &fields,
            &parameters,
            right_item_ids,
            right_instruments,
        )?,
    ])
}

#[pyfunction]
pub(crate) fn lseg_split_batch_rows(
    py: Python<'_>,
    batch: &Bound<'_, PyAny>,
) -> PyResult<Vec<PyObject>> {
    let batch_id = lseg_attr_string(batch, "batch_id")?;
    let stage = lseg_attr_string(batch, "stage")?;
    let batch_key = lseg_attr_string(batch, "batch_key")?;
    let fields = batch.getattr("fields")?;
    let parameters = batch.getattr("parameters")?;
    let mut item_ids: Vec<String> = Vec::new();
    for item_id in batch.getattr("item_ids")?.iter()? {
        item_ids.push(item_id?.str()?.to_str()?.to_string());
    }
    let mut instruments: Vec<String> = Vec::new();
    for instrument in batch.getattr("instruments")?.iter()? {
        instruments.push(instrument?.str()?.to_str()?.to_string());
    }
    if item_ids.len() != instruments.len() {
        return Err(PyValueError::new_err(
            "batch item_ids and instruments must have equal length",
        ));
    }
    if item_ids.len() <= 1 {
        return Ok(vec![lseg_child_batch_row(
            py,
            batch_id,
            &stage,
            &batch_key,
            &fields,
            &parameters,
            item_ids,
            instruments,
        )?]);
    }
    let midpoint = std::cmp::max(1usize, item_ids.len() / 2);
    let left_item_ids = item_ids[..midpoint].to_vec();
    let right_item_ids = item_ids[midpoint..].to_vec();
    let left_instruments = instruments[..midpoint].to_vec();
    let right_instruments = instruments[midpoint..].to_vec();
    Ok(vec![
        lseg_child_batch_row(
            py,
            lseg_child_batch_id(&batch_id, "a")?,
            &stage,
            &batch_key,
            &fields,
            &parameters,
            left_item_ids,
            left_instruments,
        )?,
        lseg_child_batch_row(
            py,
            lseg_child_batch_id(&batch_id, "b")?,
            &stage,
            &batch_key,
            &fields,
            &parameters,
            right_item_ids,
            right_instruments,
        )?,
    ])
}

#[pyfunction]
pub(crate) fn lseg_item_result_detail_rows(
    py: Python<'_>,
    batch_items_rows: &Bound<'_, PyAny>,
    row_count_by_item_id: &Bound<'_, PyDict>,
    normalized_instruments: Vec<Option<String>>,
) -> PyResult<Vec<PyObject>> {
    let mut out = Vec::new();
    let mut row_count = 0usize;
    for (index, item) in batch_items_rows.iter()?.enumerate() {
        row_count += 1;
        if index >= normalized_instruments.len() {
            return Err(PyValueError::new_err(
                "normalized instrument count is shorter than batch item count",
            ));
        }
        let item = item?;
        let item_id = lseg_attr_string(&item, "item_id")?;
        let item_count = match row_count_by_item_id.get_item(&item_id)? {
            Some(value) if !value.is_none() => py_int_like_to_i64(&value)?,
            _ => 0,
        };
        let row = PyDict::new_bound(py);
        row.set_item("item_id", item_id)?;
        row.set_item("instrument", normalized_instruments[index].clone())?;
        row.set_item("row_count", item_count)?;
        out.push(row.into_py(py));
    }
    if normalized_instruments.len() != row_count {
        return Err(PyValueError::new_err(
            "normalized instrument count does not match batch item count",
        ));
    }
    Ok(out)
}

#[pyfunction]
pub(crate) fn classify_lseg_error_message_value(message: &str) -> (Option<String>, Vec<String>) {
    classify_lseg_error_message_impl(message)
}

#[pyfunction]
#[pyo3(signature = (status_code=None, error_kind=None, message=""))]
pub(crate) fn is_lseg_overload_like(
    status_code: Option<i64>,
    error_kind: Option<&str>,
    message: &str,
) -> bool {
    if matches!(status_code, Some(429 | 500 | 502 | 503 | 504)) {
        return true;
    }
    if matches!(
        error_kind,
        Some("transport_timeout" | "workspace_proxy_timeout" | "backend_overload")
    ) {
        return true;
    }
    [
        "timeout",
        "timed out",
        "overload",
        "service unavailable",
        "backend",
    ]
    .iter()
    .any(|token| message.contains(token))
}

pub(crate) fn lseg_daily_limit_likely_exhausted_impl(headers: &Bound<'_, PyAny>) -> PyResult<bool> {
    let headers = headers
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("headers must be a dict"))?;
    let mut min_numeric_value: Option<i64> = None;
    for (key, value) in headers.iter() {
        let lowered_value = key.getattr("lower")?.call0()?;
        let lowered = lowered_value.str()?.to_str()?.to_string();
        if !lowered.contains("remaining")
            && !lowered.contains("daily")
            && !lowered.contains("limit")
            && !lowered.contains("usage")
        {
            continue;
        }
        if let Ok(number) = py_int_like_to_i64(&value) {
            min_numeric_value = Some(match min_numeric_value {
                Some(current) => current.min(number),
                None => number,
            });
        }
    }
    Ok(min_numeric_value.is_some_and(|value| value <= 25))
}

#[pyfunction]
pub(crate) fn lseg_daily_limit_likely_exhausted(headers: &Bound<'_, PyAny>) -> PyResult<bool> {
    lseg_daily_limit_likely_exhausted_impl(headers)
}

#[pyfunction]
#[pyo3(signature = (status_code=None, headers=None, message=""))]
pub(crate) fn lseg_indicates_daily_limit(
    status_code: Option<i64>,
    headers: Option<&Bound<'_, PyAny>>,
    message: &str,
) -> PyResult<bool> {
    if message.contains("daily") && message.contains("limit") {
        return Ok(true);
    }
    if status_code == Some(429) {
        if let Some(headers) = headers {
            return lseg_daily_limit_likely_exhausted_impl(headers);
        }
    }
    Ok(false)
}

#[pyfunction]
#[pyo3(signature = (
    error_kind,
    unresolved_identifiers,
    universe,
    attempt_no,
    max_attempts,
    normalizer
))]
pub(crate) fn lseg_should_treat_as_empty_result(
    py: Python<'_>,
    error_kind: Option<&str>,
    unresolved_identifiers: &Bound<'_, PyAny>,
    universe: &Bound<'_, PyAny>,
    attempt_no: i64,
    max_attempts: i64,
    normalizer: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    if error_kind != Some("unresolved_identifiers") {
        return Ok(false);
    }
    if attempt_no < max_attempts {
        return Ok(false);
    }
    let unresolved_set = PySet::empty_bound(py)?;
    for identifier in unresolved_identifiers.iter()? {
        let normalized = normalizer.call1((identifier?,))?;
        if !normalized.is_none() {
            unresolved_set.call_method1("add", (normalized,))?;
        }
    }
    let requested_set = PySet::empty_bound(py)?;
    for identifier in universe.iter()? {
        let normalized = normalizer.call1((identifier?,))?;
        if !normalized.is_none() {
            requested_set.call_method1("add", (normalized,))?;
        }
    }
    let requested_count = py_int_like_to_i64(&requested_set.call_method0("__len__")?)?;
    if requested_count == 0 {
        return Ok(false);
    }
    requested_set
        .call_method1("issubset", (&unresolved_set,))?
        .extract::<bool>()
}

pub(crate) fn lseg_batch_error_policy_dict(
    py: Python<'_>,
    state: &str,
    split_batch: bool,
    stop_stage: bool,
    defer_stage: bool,
) -> PyResult<PyObject> {
    let out = PyDict::new_bound(py);
    out.set_item("state", state)?;
    out.set_item("split_batch", split_batch)?;
    out.set_item("stop_stage", stop_stage)?;
    out.set_item("defer_stage", defer_stage)?;
    Ok(out.into_py(py))
}

#[pyfunction]
#[pyo3(signature = (
    status_code,
    error_kind,
    message,
    headers,
    batch_size,
    attempt_no,
    max_attempts,
    split_after_attempt,
    fatal_exception
))]
pub(crate) fn lseg_classify_batch_error_policy(
    py: Python<'_>,
    status_code: Option<i64>,
    error_kind: Option<&str>,
    message: &str,
    headers: &Bound<'_, PyAny>,
    batch_size: i64,
    attempt_no: i64,
    max_attempts: i64,
    split_after_attempt: i64,
    fatal_exception: bool,
) -> PyResult<PyObject> {
    if error_kind == Some("unresolved_identifiers") {
        if batch_size == 1 && attempt_no < max_attempts {
            return lseg_batch_error_policy_dict(py, "retryable_error", false, false, false);
        }
        return lseg_batch_error_policy_dict(
            py,
            if batch_size > 1 {
                "retryable_error"
            } else {
                "fatal_error"
            },
            batch_size > 1,
            false,
            false,
        );
    }

    if lseg_indicates_daily_limit(status_code, Some(headers), message)? {
        return lseg_batch_error_policy_dict(py, "deferred_daily_limit", false, false, true);
    }

    if status_code == Some(403) || error_kind == Some("session_open_failed") {
        return lseg_batch_error_policy_dict(py, "fatal_error", false, true, false);
    }

    if is_lseg_overload_like(status_code, error_kind, message) {
        if batch_size > 1 && attempt_no >= split_after_attempt {
            return lseg_batch_error_policy_dict(py, "retryable_error", true, false, false);
        }
        if attempt_no >= max_attempts {
            return lseg_batch_error_policy_dict(py, "fatal_error", false, true, false);
        }
        return lseg_batch_error_policy_dict(py, "retryable_error", false, false, false);
    }

    if fatal_exception || attempt_no >= max_attempts {
        return lseg_batch_error_policy_dict(py, "fatal_error", false, true, false);
    }

    lseg_batch_error_policy_dict(py, "retryable_error", false, false, false)
}

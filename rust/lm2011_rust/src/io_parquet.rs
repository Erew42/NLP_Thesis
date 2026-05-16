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
pub(crate) fn sha256_file_hex_value(path: &str) -> PyResult<String> {
    let file = File::open(path).map_err(|exc| {
        PyOSError::new_err(format!("failed to open file for SHA-256 hashing: {exc}"))
    })?;
    let mut reader = BufReader::new(file);
    let mut state = Sha256State::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let read_len = reader.read(&mut buffer).map_err(|exc| {
            PyOSError::new_err(format!("failed to read file for SHA-256 hashing: {exc}"))
        })?;
        if read_len == 0 {
            break;
        }
        state.update(&buffer[..read_len]);
    }
    Ok(bytes_to_hex(&state.finalize()))
}

#[pyfunction]
pub(crate) fn parquet_magic_probe(path: &str) -> PyResult<(u64, Vec<u8>, Vec<u8>)> {
    let mut file = File::open(path).map_err(|exc| {
        PyOSError::new_err(format!(
            "failed to open file for Parquet magic probe: {exc}"
        ))
    })?;
    let size = file
        .metadata()
        .map_err(|exc| {
            PyOSError::new_err(format!(
                "failed to stat file for Parquet magic probe: {exc}"
            ))
        })?
        .len();

    let start_len = std::cmp::min(4, size as usize);
    let mut start = vec![0u8; start_len];
    if start_len > 0 {
        file.read_exact(&mut start).map_err(|exc| {
            PyOSError::new_err(format!("failed to read Parquet magic header bytes: {exc}"))
        })?;
    }

    let end_len = std::cmp::min(4, size as usize);
    let mut end = vec![0u8; end_len];
    if end_len > 0 {
        file.seek(SeekFrom::End(-(end_len as i64))).map_err(|exc| {
            PyOSError::new_err(format!(
                "failed to seek to Parquet magic footer bytes: {exc}"
            ))
        })?;
        file.read_exact(&mut end).map_err(|exc| {
            PyOSError::new_err(format!("failed to read Parquet magic footer bytes: {exc}"))
        })?;
    }

    Ok((size, start, end))
}

#[pyfunction]
pub(crate) fn copy_file_stream(src: &str, dst: &str, buffer_size: usize) -> PyResult<()> {
    let mut reader = File::open(src)
        .map_err(|exc| PyOSError::new_err(format!("failed to open stream-copy source: {exc}")))?;
    let mut writer = File::create(dst).map_err(|exc| {
        PyOSError::new_err(format!("failed to create stream-copy destination: {exc}"))
    })?;

    if buffer_size > 0 {
        let mut buffer = vec![0u8; buffer_size];
        loop {
            let read_len = reader.read(&mut buffer).map_err(|exc| {
                PyOSError::new_err(format!("failed to read stream-copy source: {exc}"))
            })?;
            if read_len == 0 {
                break;
            }
            writer.write_all(&buffer[..read_len]).map_err(|exc| {
                PyOSError::new_err(format!("failed to write stream-copy destination: {exc}"))
            })?;
        }
    }

    writer.flush().map_err(|exc| {
        PyOSError::new_err(format!("failed to flush stream-copy destination: {exc}"))
    })?;
    let _ = writer.sync_all();

    Ok(())
}

#[pyfunction]
pub(crate) fn parquet_stream_selected_doc_indices(
    remaining_doc_ids: Vec<String>,
    doc_ids: Vec<Option<String>>,
) -> Vec<(usize, String)> {
    let mut remaining: HashSet<String> = remaining_doc_ids
        .into_iter()
        .filter(|doc_id| !doc_id.is_empty())
        .collect();
    let mut selected = Vec::new();
    for (index, doc_id) in doc_ids.into_iter().enumerate() {
        if remaining.is_empty() {
            break;
        }
        let Some(doc_id) = doc_id else {
            continue;
        };
        if doc_id.is_empty() {
            continue;
        }
        if remaining.remove(&doc_id) {
            selected.push((index, doc_id));
        }
    }
    selected
}

#[pyfunction]
pub(crate) fn lm2011_window_row_index_pairs(
    doc_permnos: Vec<Option<i64>>,
    filing_trade_indices: Vec<Option<i64>>,
    daily_permnos: Vec<Option<i64>>,
    daily_trade_indices: Vec<Option<i64>>,
    start_day: i64,
    end_day: i64,
) -> PyResult<Vec<(usize, usize)>> {
    if doc_permnos.len() != filing_trade_indices.len() {
        return Err(PyValueError::new_err(
            "doc_permnos and filing_trade_indices must have the same length",
        ));
    }
    if daily_permnos.len() != daily_trade_indices.len() {
        return Err(PyValueError::new_err(
            "daily_permnos and daily_trade_indices must have the same length",
        ));
    }
    if start_day > end_day {
        return Ok(Vec::new());
    }

    let mut daily_by_permno: HashMap<i64, Vec<(i64, usize)>> = HashMap::new();
    for (daily_idx, (permno, trade_index)) in daily_permnos
        .into_iter()
        .zip(daily_trade_indices.into_iter())
        .enumerate()
    {
        let (Some(permno), Some(trade_index)) = (permno, trade_index) else {
            continue;
        };
        daily_by_permno
            .entry(permno)
            .or_default()
            .push((trade_index, daily_idx));
    }
    for rows in daily_by_permno.values_mut() {
        rows.sort_by(|left, right| match left.0.cmp(&right.0) {
            Ordering::Equal => left.1.cmp(&right.1),
            ordering => ordering,
        });
    }

    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for (doc_idx, (permno, filing_trade_index)) in doc_permnos
        .into_iter()
        .zip(filing_trade_indices.into_iter())
        .enumerate()
    {
        let (Some(permno), Some(filing_trade_index)) = (permno, filing_trade_index) else {
            continue;
        };
        let Some(rows) = daily_by_permno.get(&permno) else {
            continue;
        };
        let lower = filing_trade_index.saturating_add(start_day);
        let upper = filing_trade_index.saturating_add(end_day);
        let start = rows.partition_point(|(trade_index, _)| *trade_index < lower);
        let end = rows.partition_point(|(trade_index, _)| *trade_index <= upper);
        for (_, daily_idx) in &rows[start..end] {
            pairs.push((doc_idx, *daily_idx));
        }
    }

    Ok(pairs)
}

#[pyfunction]
pub(crate) fn lseg_doc_unresolved_mask(
    request_doc_ids: Vec<Option<String>>,
    retrieval_eligible: Vec<Option<bool>>,
    returned_doc_ids: Vec<Option<String>>,
) -> PyResult<Vec<bool>> {
    if request_doc_ids.len() != retrieval_eligible.len() {
        return Err(PyValueError::new_err(
            "request_doc_ids and retrieval_eligible must have the same length",
        ));
    }
    let returned: HashSet<String> = returned_doc_ids.into_iter().flatten().collect();
    let mask = request_doc_ids
        .into_iter()
        .zip(retrieval_eligible)
        .map(|(doc_id, eligible)| {
            eligible.unwrap_or(false)
                && match doc_id {
                    Some(doc_id) => !returned.contains(&doc_id),
                    None => false,
                }
        })
        .collect();
    Ok(mask)
}

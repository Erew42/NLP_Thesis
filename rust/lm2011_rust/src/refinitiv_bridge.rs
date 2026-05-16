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
use crate::lseg_ops::*;
use crate::multisurface_audit::*;
use crate::refinitiv_analyst::*;
use crate::refinitiv_authority::*;
use crate::refinitiv_excel::*;
use crate::sec_extraction::*;
use crate::sentence_cleaning::*;
use crate::sentence_quality_api::*;

#[pyfunction]
#[pyo3(signature = (bridge_row_id=None))]
pub(crate) fn parse_bridge_row_id_liid(
    bridge_row_id: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<String>> {
    let Some(normalized) = normalize_lookup_text_any_impl(bridge_row_id)? else {
        return Ok(None);
    };
    let parts: Vec<&str> = normalized.splitn(5, ':').collect();
    if parts.len() != 5 {
        return Err(PyValueError::new_err(format!(
            "bridge_row_id has unexpected format: {normalized:?}"
        )));
    }
    Ok(if parts[1] == "-" {
        None
    } else {
        Some(parts[1].to_string())
    })
}

#[pyfunction]
#[pyo3(signature = (
    left_ric,
    right_ric,
    left_isin=None,
    right_isin=None,
    left_cusip=None,
    right_cusip=None
))]
pub(crate) fn refinitiv_bridge_identity_candidates_agree(
    left_ric: &str,
    right_ric: &str,
    left_isin: Option<&str>,
    right_isin: Option<&str>,
    left_cusip: Option<&str>,
    right_cusip: Option<&str>,
) -> bool {
    if left_isin.is_some() && right_isin.is_some() && left_isin != right_isin {
        return false;
    }
    if left_cusip.is_some() && right_cusip.is_some() && left_cusip != right_cusip {
        return false;
    }
    if left_isin.is_some() && right_isin.is_some() {
        return true;
    }
    if left_cusip.is_some() && right_cusip.is_some() {
        return true;
    }
    if left_isin.is_none() && right_isin.is_none() && left_cusip.is_none() && right_cusip.is_none()
    {
        return left_ric == right_ric;
    }
    false
}

#[pyfunction]
#[pyo3(signature = (
    left_ric,
    right_ric,
    left_isin=None,
    right_isin=None,
    left_cusip=None,
    right_cusip=None
))]
pub(crate) fn refinitiv_bridge_candidates_materially_conflict(
    left_ric: &str,
    right_ric: &str,
    left_isin: Option<&str>,
    right_isin: Option<&str>,
    left_cusip: Option<&str>,
    right_cusip: Option<&str>,
) -> bool {
    !refinitiv_bridge_identity_candidates_agree(
        left_ric,
        right_ric,
        left_isin,
        right_isin,
        left_cusip,
        right_cusip,
    )
}

#[derive(Clone)]
pub(crate) struct RefinitivBridgeIdentityCandidate {
    returned_ric: String,
    returned_isin: Option<String>,
    returned_cusip: Option<String>,
}

pub(crate) fn dict_refinitiv_lookup_result(
    dict: &Bound<'_, PyDict>,
    key: &str,
) -> PyResult<Option<String>> {
    let Some(value) = dict.get_item(key)? else {
        return Ok(None);
    };
    normalize_refinitiv_lookup_result_value(Some(&value))
}

pub(crate) fn dict_lookup_identity_candidate(
    dict: &Bound<'_, PyDict>,
    source: &str,
) -> PyResult<Option<RefinitivBridgeIdentityCandidate>> {
    let returned_ric_key = format!("{source}_returned_ric");
    let Some(returned_ric) = dict_refinitiv_lookup_result(dict, &returned_ric_key)? else {
        return Ok(None);
    };
    Ok(Some(RefinitivBridgeIdentityCandidate {
        returned_ric,
        returned_isin: dict_refinitiv_lookup_result(dict, &format!("{source}_returned_isin"))?,
        returned_cusip: dict_refinitiv_lookup_result(dict, &format!("{source}_returned_cusip"))?,
    }))
}

pub(crate) type RefinitivBridgeCandidateTuple = (String, String, Option<String>, Option<String>);

#[pyfunction]
pub(crate) fn refinitiv_bridge_lookup_candidate_from_record(
    record: &Bound<'_, PyAny>,
    source: &str,
) -> PyResult<Option<RefinitivBridgeCandidateTuple>> {
    let dict = record
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("lookup candidate record is not a dict"))?;
    Ok(
        dict_lookup_identity_candidate(dict, source)?.map(|candidate| {
            (
                source.to_string(),
                candidate.returned_ric,
                candidate.returned_isin,
                candidate.returned_cusip,
            )
        }),
    )
}

#[pyfunction]
pub(crate) fn refinitiv_bridge_accepted_candidate_from_record(
    record: &Bound<'_, PyAny>,
) -> PyResult<Option<RefinitivBridgeCandidateTuple>> {
    let dict = record
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("accepted candidate record is not a dict"))?;
    let Some(returned_ric) = dict_refinitiv_lookup_result(dict, "accepted_ric")? else {
        return Ok(None);
    };
    let source = dict_normalized_string(dict, "accepted_ric_source")?
        .unwrap_or_else(|| "CONVENTIONAL".to_string());
    Ok(Some((
        source,
        returned_ric,
        dict_refinitiv_lookup_result(dict, "accepted_identity_returned_isin")?,
        dict_refinitiv_lookup_result(dict, "accepted_identity_returned_cusip")?,
    )))
}

pub(crate) fn refinitiv_bridge_identity_candidates_agree_impl(
    left: &RefinitivBridgeIdentityCandidate,
    right: &RefinitivBridgeIdentityCandidate,
) -> bool {
    refinitiv_bridge_identity_candidates_agree(
        &left.returned_ric,
        &right.returned_ric,
        left.returned_isin.as_deref(),
        right.returned_isin.as_deref(),
        left.returned_cusip.as_deref(),
        right.returned_cusip.as_deref(),
    )
}

pub(crate) fn merge_bridge_identity_fields(
    left: &RefinitivBridgeIdentityCandidate,
    right: Option<&RefinitivBridgeIdentityCandidate>,
) -> (Option<String>, Option<String>) {
    match right {
        Some(right) => (
            left.returned_isin
                .clone()
                .or_else(|| right.returned_isin.clone()),
            left.returned_cusip
                .clone()
                .or_else(|| right.returned_cusip.clone()),
        ),
        None => (left.returned_isin.clone(), left.returned_cusip.clone()),
    }
}

pub(crate) fn refinitiv_bridge_py_dict_rows_from_column_values(
    py: Python<'_>,
    column_names: &[String],
    column_values: &Bound<'_, PyAny>,
    label: &str,
) -> PyResult<Vec<PyObject>> {
    let (columns, row_count) =
        collect_pyobject_column_values(py, column_names, column_values, label)?;
    let mut rows = Vec::with_capacity(row_count);
    for row_idx in 0..row_count {
        let row = PyDict::new_bound(py);
        for (column_idx, column_name) in column_names.iter().enumerate() {
            row.set_item(column_name, columns[column_idx][row_idx].clone_ref(py))?;
        }
        rows.push(row.into_py(py));
    }
    Ok(rows)
}

pub(crate) fn refinitiv_bridge_py_dict_rows_from_records(
    py: Python<'_>,
    records: &Bound<'_, PyAny>,
    label: &str,
) -> PyResult<Vec<PyObject>> {
    let mut rows = Vec::new();
    for record in records.iter()? {
        let record = record?;
        let dict = record
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err(format!("{label} row is not a dict")))?;
        rows.push(copy_py_dict(py, dict)?.into_py(py));
    }
    Ok(rows)
}

pub(crate) struct RefinitivBridgeAcceptedCandidateWithSource {
    source: String,
    candidate: RefinitivBridgeIdentityCandidate,
}

pub(crate) fn accepted_candidate_with_source_from_dict(
    dict: &Bound<'_, PyDict>,
) -> PyResult<Option<RefinitivBridgeAcceptedCandidateWithSource>> {
    let Some(returned_ric) = dict_refinitiv_lookup_result(dict, "accepted_ric")? else {
        return Ok(None);
    };
    Ok(Some(RefinitivBridgeAcceptedCandidateWithSource {
        source: dict_normalized_string(dict, "accepted_ric_source")?
            .unwrap_or_else(|| "CONVENTIONAL".to_string()),
        candidate: RefinitivBridgeIdentityCandidate {
            returned_ric,
            returned_isin: dict_refinitiv_lookup_result(dict, "accepted_identity_returned_isin")?,
            returned_cusip: dict_refinitiv_lookup_result(dict, "accepted_identity_returned_cusip")?,
        },
    }))
}

pub(crate) fn accepted_candidate_matches_target(
    target: &Bound<'_, PyDict>,
    source: &RefinitivBridgeIdentityCandidate,
) -> PyResult<bool> {
    if let Some(target_isin) = dict_normalized_string(target, "ISIN")? {
        return Ok(source
            .returned_isin
            .as_ref()
            .is_some_and(|value| value == &target_isin));
    }
    if let Some(target_cusip) = dict_normalized_string(target, "CUSIP")? {
        return Ok(source
            .returned_cusip
            .as_ref()
            .is_some_and(|value| value == &target_cusip));
    }
    Ok(false)
}

pub(crate) fn bridge_resolution_sort_key(
    date_type: &Bound<'_, PyAny>,
    dict: &Bound<'_, PyDict>,
) -> PyResult<(String, String, String)> {
    let first_seen = dict.get_item("first_seen_caldt")?;
    let last_seen = dict.get_item("last_seen_caldt")?;
    Ok((
        refinitiv_bridge_date_to_text(date_type, first_seen.as_ref())?.unwrap_or_default(),
        refinitiv_bridge_date_to_text(date_type, last_seen.as_ref())?.unwrap_or_default(),
        dict_normalized_string(dict, "bridge_row_id")?.unwrap_or_default(),
    ))
}

#[derive(Clone)]
pub(crate) struct RefinitivBridgeResolutionColumnRow {
    row_idx: usize,
    bridge_row_id: Option<String>,
    kypermno: Option<String>,
    liid: Option<String>,
    first_seen_sort: String,
    last_seen_sort: String,
    isin: Option<String>,
    cusip: Option<String>,
    ticker_candidate: Option<RefinitivBridgeIdentityCandidate>,
    accepted_identity_returned_isin: Option<String>,
    accepted_identity_returned_cusip: Option<String>,
    accepted_ric: Option<String>,
    accepted_ric_source: Option<String>,
    accepted_resolution_status: String,
    conventional_identity_conflict: bool,
    ticker_candidate_conflicts_with_conventional: Option<bool>,
    extended_ric: Option<String>,
    extended_from_source_idx: Option<usize>,
    extension_direction: Option<String>,
    extension_status: String,
    chosen_source_source: Option<String>,
    effective_collection_ric: Option<String>,
    effective_collection_ric_source: Option<String>,
    effective_resolution_status: String,
}

pub(crate) fn parse_bridge_row_id_liid_text(
    bridge_row_id: Option<&str>,
) -> PyResult<Option<String>> {
    let Some(normalized) = bridge_row_id else {
        return Ok(None);
    };
    let parts: Vec<&str> = normalized.splitn(5, ':').collect();
    if parts.len() != 5 {
        return Err(PyValueError::new_err(format!(
            "bridge_row_id has unexpected format: {normalized:?}"
        )));
    }
    Ok(if parts[1] == "-" {
        None
    } else {
        Some(parts[1].to_string())
    })
}

pub(crate) fn refinitiv_bridge_column_lookup_result(
    py: Python<'_>,
    columns: &[Vec<PyObject>],
    column_index: &HashMap<String, usize>,
    row_idx: usize,
    column_name: &str,
) -> PyResult<Option<String>> {
    optional_column_value(columns, column_index, row_idx, column_name)
        .map(|value| normalize_refinitiv_lookup_result_value(Some(value.bind(py))))
        .transpose()
        .map(|value| value.flatten())
}

pub(crate) fn refinitiv_bridge_column_identity_candidate(
    py: Python<'_>,
    columns: &[Vec<PyObject>],
    column_index: &HashMap<String, usize>,
    row_idx: usize,
    source: &str,
) -> PyResult<Option<RefinitivBridgeIdentityCandidate>> {
    let returned_ric_key = format!("{source}_returned_ric");
    let Some(returned_ric) = refinitiv_bridge_column_lookup_result(
        py,
        columns,
        column_index,
        row_idx,
        &returned_ric_key,
    )?
    else {
        return Ok(None);
    };
    let returned_isin_key = format!("{source}_returned_isin");
    let returned_cusip_key = format!("{source}_returned_cusip");
    Ok(Some(RefinitivBridgeIdentityCandidate {
        returned_ric,
        returned_isin: refinitiv_bridge_column_lookup_result(
            py,
            columns,
            column_index,
            row_idx,
            &returned_isin_key,
        )?,
        returned_cusip: refinitiv_bridge_column_lookup_result(
            py,
            columns,
            column_index,
            row_idx,
            &returned_cusip_key,
        )?,
    }))
}

pub(crate) fn refinitiv_bridge_resolution_column_row(
    py: Python<'_>,
    date_type: &Bound<'_, PyAny>,
    columns: &[Vec<PyObject>],
    column_index: &HashMap<String, usize>,
    row_idx: usize,
) -> PyResult<RefinitivBridgeResolutionColumnRow> {
    let bridge_row_id =
        column_normalized_string(py, columns, column_index, row_idx, "bridge_row_id")?;
    let kypermno = column_normalized_string(py, columns, column_index, row_idx, "KYPERMNO")?;
    let liid = parse_bridge_row_id_liid_text(bridge_row_id.as_deref())?;
    let first_seen = optional_column_value(columns, column_index, row_idx, "first_seen_caldt")
        .map(|value| value.bind(py));
    let last_seen = optional_column_value(columns, column_index, row_idx, "last_seen_caldt")
        .map(|value| value.bind(py));

    let isin_candidate =
        refinitiv_bridge_column_identity_candidate(py, columns, column_index, row_idx, "ISIN")?;
    let cusip_candidate =
        refinitiv_bridge_column_identity_candidate(py, columns, column_index, row_idx, "CUSIP")?;
    let ticker_candidate =
        refinitiv_bridge_column_identity_candidate(py, columns, column_index, row_idx, "TICKER")?;

    let mut accepted_identity_returned_isin: Option<String> = None;
    let mut accepted_identity_returned_cusip: Option<String> = None;
    let mut accepted_ric: Option<String> = None;
    let mut accepted_ric_source: Option<String> = None;
    let mut accepted_resolution_status = "unresolved_after_isin_cusip".to_string();
    let mut conventional_identity_conflict = false;

    match (isin_candidate.as_ref(), cusip_candidate.as_ref()) {
        (Some(isin), None) => {
            let (merged_isin, merged_cusip) = merge_bridge_identity_fields(isin, None);
            accepted_identity_returned_isin = merged_isin;
            accepted_identity_returned_cusip = merged_cusip;
            accepted_ric = Some(isin.returned_ric.clone());
            accepted_ric_source = Some("ISIN".to_string());
            accepted_resolution_status = "resolved_from_isin".to_string();
        }
        (None, Some(cusip)) => {
            let (merged_isin, merged_cusip) = merge_bridge_identity_fields(cusip, None);
            accepted_identity_returned_isin = merged_isin;
            accepted_identity_returned_cusip = merged_cusip;
            accepted_ric = Some(cusip.returned_ric.clone());
            accepted_ric_source = Some("CUSIP".to_string());
            accepted_resolution_status = "resolved_from_cusip".to_string();
        }
        (Some(isin), Some(cusip)) => {
            if refinitiv_bridge_identity_candidates_agree_impl(isin, cusip) {
                let (merged_isin, merged_cusip) = merge_bridge_identity_fields(isin, Some(cusip));
                accepted_identity_returned_isin = merged_isin;
                accepted_identity_returned_cusip = merged_cusip;
                accepted_ric = Some(isin.returned_ric.clone());
                accepted_ric_source = Some("ISIN".to_string());
                accepted_resolution_status = "resolved_conventional_agree".to_string();
            } else {
                conventional_identity_conflict = true;
                accepted_resolution_status = "unresolved_conventional_conflict".to_string();
            }
        }
        (None, None) => {}
    }

    let accepted_candidate = accepted_ric
        .as_ref()
        .map(|ric| RefinitivBridgeIdentityCandidate {
            returned_ric: ric.clone(),
            returned_isin: accepted_identity_returned_isin.clone(),
            returned_cusip: accepted_identity_returned_cusip.clone(),
        });
    let ticker_candidate_conflicts_with_conventional =
        match (ticker_candidate.as_ref(), accepted_candidate.as_ref()) {
            (Some(ticker), Some(accepted)) => Some(
                !refinitiv_bridge_identity_candidates_agree_impl(accepted, ticker),
            ),
            _ => None,
        };

    Ok(RefinitivBridgeResolutionColumnRow {
        row_idx,
        bridge_row_id,
        kypermno,
        liid,
        first_seen_sort: refinitiv_bridge_date_to_text(date_type, first_seen)?.unwrap_or_default(),
        last_seen_sort: refinitiv_bridge_date_to_text(date_type, last_seen)?.unwrap_or_default(),
        isin: column_normalized_string(py, columns, column_index, row_idx, "ISIN")?,
        cusip: column_normalized_string(py, columns, column_index, row_idx, "CUSIP")?,
        ticker_candidate,
        accepted_identity_returned_isin,
        accepted_identity_returned_cusip,
        accepted_ric,
        accepted_ric_source,
        accepted_resolution_status,
        conventional_identity_conflict,
        ticker_candidate_conflicts_with_conventional,
        extended_ric: None,
        extended_from_source_idx: None,
        extension_direction: None,
        extension_status: "not_extended".to_string(),
        chosen_source_source: None,
        effective_collection_ric: None,
        effective_collection_ric_source: None,
        effective_resolution_status: "unresolved_after_accept_and_extend".to_string(),
    })
}

pub(crate) fn refinitiv_bridge_resolution_accepted_candidate(
    row: &RefinitivBridgeResolutionColumnRow,
) -> Option<RefinitivBridgeAcceptedCandidateWithSource> {
    row.accepted_ric
        .as_ref()
        .map(|returned_ric| RefinitivBridgeAcceptedCandidateWithSource {
            source: row
                .accepted_ric_source
                .clone()
                .unwrap_or_else(|| "CONVENTIONAL".to_string()),
            candidate: RefinitivBridgeIdentityCandidate {
                returned_ric: returned_ric.clone(),
                returned_isin: row.accepted_identity_returned_isin.clone(),
                returned_cusip: row.accepted_identity_returned_cusip.clone(),
            },
        })
}

pub(crate) fn refinitiv_bridge_resolution_candidate_matches_target(
    target: &RefinitivBridgeResolutionColumnRow,
    source: &RefinitivBridgeIdentityCandidate,
) -> bool {
    if let Some(target_isin) = target.isin.as_ref() {
        return source
            .returned_isin
            .as_ref()
            .is_some_and(|value| value == target_isin);
    }
    if let Some(target_cusip) = target.cusip.as_ref() {
        return source
            .returned_cusip
            .as_ref()
            .is_some_and(|value| value == target_cusip);
    }
    false
}

pub(crate) fn refinitiv_bridge_resolution_frame_rows_from_columns_impl(
    py: Python<'_>,
    column_names: &[String],
    column_values: &Bound<'_, PyAny>,
    output_columns: Vec<String>,
) -> PyResult<Vec<PyObject>> {
    let date_type = py.import_bound("datetime")?.getattr("date")?;
    let (columns, row_count) =
        collect_pyobject_column_values(py, column_names, column_values, "resolution frame source")?;
    let column_index = column_index_by_name(column_names);
    let mut rows: Vec<RefinitivBridgeResolutionColumnRow> = Vec::with_capacity(row_count);
    let mut grouped_records: BTreeMap<(Option<String>, Option<String>), Vec<usize>> =
        BTreeMap::new();

    for row_idx in 0..row_count {
        let row = refinitiv_bridge_resolution_column_row(
            py,
            &date_type,
            &columns,
            &column_index,
            row_idx,
        )?;
        grouped_records
            .entry((row.kypermno.clone(), row.liid.clone()))
            .or_default()
            .push(rows.len());
        rows.push(row);
    }

    for group_indices in grouped_records.values() {
        let mut ordered_indices = group_indices.clone();
        ordered_indices.sort_by(|left_idx, right_idx| {
            let left = &rows[*left_idx];
            let right = &rows[*right_idx];
            left.first_seen_sort
                .cmp(&right.first_seen_sort)
                .then(left.last_seen_sort.cmp(&right.last_seen_sort))
                .then(
                    left.bridge_row_id
                        .as_deref()
                        .unwrap_or_default()
                        .cmp(right.bridge_row_id.as_deref().unwrap_or_default()),
                )
                .then(left.row_idx.cmp(&right.row_idx))
        });

        for (ordered_pos, record_idx) in ordered_indices.iter().copied().enumerate() {
            if rows[record_idx].accepted_ric.is_some() {
                continue;
            }
            if rows[record_idx].conventional_identity_conflict {
                rows[record_idx].extension_status = "not_extended_due_to_conflict".to_string();
                continue;
            }

            let prior_idx = if ordered_pos > 0 {
                Some(ordered_indices[ordered_pos - 1])
            } else {
                None
            };
            let next_idx = if ordered_pos + 1 < ordered_indices.len() {
                Some(ordered_indices[ordered_pos + 1])
            } else {
                None
            };
            let prior_candidate = prior_idx
                .and_then(|idx| refinitiv_bridge_resolution_accepted_candidate(&rows[idx]));
            let next_candidate =
                next_idx.and_then(|idx| refinitiv_bridge_resolution_accepted_candidate(&rows[idx]));
            let target_has_raw_conventional =
                rows[record_idx].isin.is_some() || rows[record_idx].cusip.is_some();

            if let (Some(prior), Some(next)) = (&prior_candidate, &next_candidate) {
                if !refinitiv_bridge_identity_candidates_agree_impl(
                    &prior.candidate,
                    &next.candidate,
                ) {
                    rows[record_idx].extension_status = "not_extended_due_to_conflict".to_string();
                    continue;
                }
            }

            let compatible_prior = match (prior_candidate.as_ref(), prior_idx) {
                (Some(prior), Some(_)) if target_has_raw_conventional => {
                    refinitiv_bridge_resolution_candidate_matches_target(
                        &rows[record_idx],
                        &prior.candidate,
                    )
                }
                _ => false,
            };
            let compatible_next = match (next_candidate.as_ref(), next_idx) {
                (Some(next), Some(_)) if target_has_raw_conventional => {
                    refinitiv_bridge_resolution_candidate_matches_target(
                        &rows[record_idx],
                        &next.candidate,
                    )
                }
                _ => false,
            };

            let mut chosen_source_idx: Option<usize> = None;
            let mut extension_direction: Option<&str> = None;
            let mut extension_status = "not_extended_no_adjacent_conventional_source";

            if !target_has_raw_conventional {
                if let (Some(prior), Some(next), Some(prior_idx), Some(next_idx)) = (
                    prior_candidate.as_ref(),
                    next_candidate.as_ref(),
                    prior_idx,
                    next_idx,
                ) {
                    if refinitiv_bridge_identity_candidates_agree_impl(
                        &prior.candidate,
                        &next.candidate,
                    ) {
                        let choice =
                            refinitiv_bridge_adjacent_extension_choice(&prior.source, &next.source);
                        chosen_source_idx = Some(if choice == "NEXT" {
                            next_idx
                        } else {
                            prior_idx
                        });
                        extension_direction = Some("ADJACENT");
                        extension_status = "extended_from_adjacent_conventional_span";
                    }
                }
            } else if compatible_prior && compatible_next {
                if let (Some(prior), Some(next), Some(prior_idx), Some(next_idx)) = (
                    prior_candidate.as_ref(),
                    next_candidate.as_ref(),
                    prior_idx,
                    next_idx,
                ) {
                    let choice =
                        refinitiv_bridge_adjacent_extension_choice(&prior.source, &next.source);
                    chosen_source_idx = Some(if choice == "NEXT" {
                        next_idx
                    } else {
                        prior_idx
                    });
                    extension_direction = Some("ADJACENT");
                    extension_status = "extended_from_adjacent_conventional_span";
                }
            } else if compatible_prior {
                chosen_source_idx = prior_idx;
                extension_direction = Some("PRIOR");
                extension_status = "extended_from_prior_conventional_span";
            } else if compatible_next {
                chosen_source_idx = next_idx;
                extension_direction = Some("NEXT");
                extension_status = "extended_from_next_conventional_span";
            }

            rows[record_idx].extension_status = extension_status.to_string();
            if let Some(source_idx) = chosen_source_idx {
                rows[record_idx].extended_ric = rows[source_idx].accepted_ric.clone();
                rows[record_idx].extended_from_source_idx = Some(source_idx);
                rows[record_idx].extension_direction =
                    extension_direction.map(|value| value.to_string());
                rows[record_idx].chosen_source_source =
                    rows[source_idx].accepted_ric_source.clone();
            }
        }
    }

    for row in rows.iter_mut() {
        if let Some(accepted_ric) = row.accepted_ric.clone() {
            row.effective_collection_ric = Some(accepted_ric);
            row.effective_collection_ric_source = row.accepted_ric_source.clone();
            row.effective_resolution_status = "effective_from_accepted_ric".to_string();
        } else if let Some(extended_ric) = row.extended_ric.clone() {
            row.effective_collection_ric = Some(extended_ric);
            row.effective_collection_ric_source = row.chosen_source_source.as_ref().map(|source| {
                format!(
                    "EXTENDED_FROM_{}_{}",
                    row.extension_direction.as_deref().unwrap_or("ADJACENT"),
                    source
                )
            });
            row.effective_resolution_status = "effective_from_extended_ric".to_string();
        } else {
            row.effective_collection_ric = None;
            row.effective_collection_ric_source = None;
            row.effective_resolution_status = "unresolved_after_accept_and_extend".to_string();
        }
    }

    let mut out_rows = Vec::with_capacity(rows.len());
    for row in &rows {
        let out = PyDict::new_bound(py);
        for column in &output_columns {
            match column.as_str() {
                "accepted_identity_returned_isin" => {
                    out.set_item(column, row.accepted_identity_returned_isin.clone())?
                }
                "accepted_identity_returned_cusip" => {
                    out.set_item(column, row.accepted_identity_returned_cusip.clone())?
                }
                "accepted_ric" => out.set_item(column, row.accepted_ric.clone())?,
                "accepted_ric_source" => out.set_item(column, row.accepted_ric_source.clone())?,
                "accepted_resolution_status" => {
                    out.set_item(column, row.accepted_resolution_status.as_str())?
                }
                "conventional_identity_conflict" => {
                    out.set_item(column, row.conventional_identity_conflict)?
                }
                "ticker_candidate_ric" => out.set_item(
                    column,
                    row.ticker_candidate
                        .as_ref()
                        .map(|candidate| candidate.returned_ric.clone()),
                )?,
                "ticker_candidate_available" => {
                    out.set_item(column, row.ticker_candidate.is_some())?
                }
                "ticker_candidate_conflicts_with_conventional" => {
                    out.set_item(column, row.ticker_candidate_conflicts_with_conventional)?
                }
                "extended_ric" => out.set_item(column, row.extended_ric.clone())?,
                "extended_from_bridge_row_id" => out.set_item(
                    column,
                    row.extended_from_source_idx
                        .and_then(|idx| rows.get(idx))
                        .and_then(|source| source.bridge_row_id.clone()),
                )?,
                "extended_from_span_start" => match row.extended_from_source_idx {
                    Some(source_idx) => out.set_item(
                        column,
                        optional_column_pyobject(
                            py,
                            &columns,
                            &column_index,
                            rows[source_idx].row_idx,
                            "first_seen_caldt",
                        ),
                    )?,
                    None => out.set_item(column, Option::<String>::None)?,
                },
                "extended_from_span_end" => match row.extended_from_source_idx {
                    Some(source_idx) => out.set_item(
                        column,
                        optional_column_pyobject(
                            py,
                            &columns,
                            &column_index,
                            rows[source_idx].row_idx,
                            "last_seen_caldt",
                        ),
                    )?,
                    None => out.set_item(column, Option::<String>::None)?,
                },
                "extension_direction" => out.set_item(column, row.extension_direction.clone())?,
                "extension_status" => out.set_item(column, row.extension_status.as_str())?,
                "effective_collection_ric" => {
                    out.set_item(column, row.effective_collection_ric.clone())?
                }
                "effective_collection_ric_source" => {
                    out.set_item(column, row.effective_collection_ric_source.clone())?
                }
                "effective_resolution_status" => {
                    out.set_item(column, row.effective_resolution_status.as_str())?
                }
                _ => out.set_item(
                    column,
                    optional_column_pyobject(py, &columns, &column_index, row.row_idx, column),
                )?,
            }
        }
        out_rows.push(out.into_py(py));
    }

    Ok(out_rows)
}

#[pyfunction]
pub(crate) fn refinitiv_bridge_derive_accepted_resolution(
    py: Python<'_>,
    record: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let dict = record
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("resolution record is not a dict"))?;
    let isin_candidate = dict_lookup_identity_candidate(dict, "ISIN")?;
    let cusip_candidate = dict_lookup_identity_candidate(dict, "CUSIP")?;
    let ticker_candidate = dict_lookup_identity_candidate(dict, "TICKER")?;

    let mut accepted_identity_returned_isin: Option<String> = None;
    let mut accepted_identity_returned_cusip: Option<String> = None;
    let mut accepted_ric: Option<String> = None;
    let mut accepted_ric_source: Option<String> = None;
    let mut accepted_resolution_status = "unresolved_after_isin_cusip";
    let mut conventional_identity_conflict = false;

    match (isin_candidate.as_ref(), cusip_candidate.as_ref()) {
        (Some(isin), None) => {
            let (merged_isin, merged_cusip) = merge_bridge_identity_fields(isin, None);
            accepted_identity_returned_isin = merged_isin;
            accepted_identity_returned_cusip = merged_cusip;
            accepted_ric = Some(isin.returned_ric.clone());
            accepted_ric_source = Some("ISIN".to_string());
            accepted_resolution_status = "resolved_from_isin";
        }
        (None, Some(cusip)) => {
            let (merged_isin, merged_cusip) = merge_bridge_identity_fields(cusip, None);
            accepted_identity_returned_isin = merged_isin;
            accepted_identity_returned_cusip = merged_cusip;
            accepted_ric = Some(cusip.returned_ric.clone());
            accepted_ric_source = Some("CUSIP".to_string());
            accepted_resolution_status = "resolved_from_cusip";
        }
        (Some(isin), Some(cusip)) => {
            if refinitiv_bridge_identity_candidates_agree_impl(isin, cusip) {
                let (merged_isin, merged_cusip) = merge_bridge_identity_fields(isin, Some(cusip));
                accepted_identity_returned_isin = merged_isin;
                accepted_identity_returned_cusip = merged_cusip;
                accepted_ric = Some(isin.returned_ric.clone());
                accepted_ric_source = Some("ISIN".to_string());
                accepted_resolution_status = "resolved_conventional_agree";
            } else {
                conventional_identity_conflict = true;
                accepted_resolution_status = "unresolved_conventional_conflict";
            }
        }
        (None, None) => {}
    }

    let accepted_candidate = accepted_ric
        .as_ref()
        .map(|ric| RefinitivBridgeIdentityCandidate {
            returned_ric: ric.clone(),
            returned_isin: accepted_identity_returned_isin.clone(),
            returned_cusip: accepted_identity_returned_cusip.clone(),
        });

    let ticker_candidate_conflicts_with_conventional =
        match (ticker_candidate.as_ref(), accepted_candidate.as_ref()) {
            (Some(ticker), Some(accepted)) => Some(
                !refinitiv_bridge_identity_candidates_agree_impl(accepted, ticker),
            ),
            _ => None,
        };

    let out = PyDict::new_bound(py);
    out.set_item(
        "accepted_identity_returned_isin",
        accepted_identity_returned_isin,
    )?;
    out.set_item(
        "accepted_identity_returned_cusip",
        accepted_identity_returned_cusip,
    )?;
    out.set_item("accepted_ric", accepted_ric)?;
    out.set_item("accepted_ric_source", accepted_ric_source)?;
    out.set_item("accepted_resolution_status", accepted_resolution_status)?;
    out.set_item(
        "conventional_identity_conflict",
        conventional_identity_conflict,
    )?;
    out.set_item(
        "ticker_candidate_ric",
        ticker_candidate
            .as_ref()
            .map(|candidate| candidate.returned_ric.clone()),
    )?;
    out.set_item("ticker_candidate_available", ticker_candidate.is_some())?;
    out.set_item(
        "ticker_candidate_conflicts_with_conventional",
        ticker_candidate_conflicts_with_conventional,
    )?;
    Ok(out.into_py(py))
}

#[pyfunction]
#[pyo3(signature = (record, chosen_source_source=None))]
pub(crate) fn refinitiv_bridge_effective_resolution_fields(
    record: &Bound<'_, PyAny>,
    chosen_source_source: Option<&Bound<'_, PyAny>>,
) -> PyResult<(Option<String>, Option<String>, String)> {
    let dict = record
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("effective resolution record is not a dict"))?;
    if let Some(accepted_ric) = dict_normalized_string(dict, "accepted_ric")? {
        return Ok((
            Some(accepted_ric),
            dict_normalized_string(dict, "accepted_ric_source")?,
            "effective_from_accepted_ric".to_string(),
        ));
    }
    if let Some(extended_ric) = dict_normalized_string(dict, "extended_ric")? {
        let extension_direction = dict_normalized_string(dict, "extension_direction")?
            .unwrap_or_else(|| "ADJACENT".to_string());
        let effective_source = normalize_lookup_text_any_impl(chosen_source_source)?
            .map(|source| format!("EXTENDED_FROM_{extension_direction}_{source}"));
        return Ok((
            Some(extended_ric),
            effective_source,
            "effective_from_extended_ric".to_string(),
        ));
    }
    Ok((None, None, "unresolved_after_accept_and_extend".to_string()))
}

pub(crate) fn refinitiv_bridge_resolution_frame_rows_impl(
    py: Python<'_>,
    records: Vec<PyObject>,
    output_columns: Vec<String>,
) -> PyResult<Vec<PyObject>> {
    let date_type = py.import_bound("datetime")?.getattr("date")?;
    let mut grouped_records: BTreeMap<(Option<String>, Option<String>), Vec<usize>> =
        BTreeMap::new();

    for (record_idx, record) in records.iter().enumerate() {
        let dict = record
            .bind(py)
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("resolution frame source row is not a dict"))?;
        let derived = refinitiv_bridge_derive_accepted_resolution(py, dict.as_any())?;
        let derived_dict = derived
            .bind(py)
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("derived resolution payload is not a dict"))?;
        for (key, value) in derived_dict.iter() {
            dict.set_item(key, value)?;
        }

        let liid = parse_bridge_row_id_liid(dict.get_item("bridge_row_id")?.as_ref())?;
        grouped_records
            .entry((dict_normalized_string(dict, "KYPERMNO")?, liid))
            .or_default()
            .push(record_idx);
    }

    let mut chosen_source_sources: Vec<Option<String>> = vec![None; records.len()];
    for group_indices in grouped_records.values() {
        let mut ordered_indices = group_indices.clone();
        ordered_indices.sort_by(|left_idx, right_idx| {
            let left_dict = records[*left_idx].bind(py).downcast::<PyDict>().ok();
            let right_dict = records[*right_idx].bind(py).downcast::<PyDict>().ok();
            match (left_dict, right_dict) {
                (Some(left), Some(right)) => {
                    let left_key = bridge_resolution_sort_key(&date_type, left);
                    let right_key = bridge_resolution_sort_key(&date_type, right);
                    match (left_key, right_key) {
                        (Ok(left_key), Ok(right_key)) => left_key.cmp(&right_key),
                        _ => left_idx.cmp(right_idx),
                    }
                }
                _ => left_idx.cmp(right_idx),
            }
        });

        for (ordered_pos, record_idx) in ordered_indices.iter().enumerate() {
            let record = records[*record_idx]
                .bind(py)
                .downcast::<PyDict>()
                .map_err(|_| PyValueError::new_err("ordered resolution row is not a dict"))?;

            record.set_item("extended_ric", Option::<String>::None)?;
            record.set_item("extended_from_bridge_row_id", Option::<String>::None)?;
            record.set_item("extended_from_span_start", Option::<String>::None)?;
            record.set_item("extended_from_span_end", Option::<String>::None)?;
            record.set_item("extension_direction", Option::<String>::None)?;
            record.set_item("extension_status", "not_extended")?;

            if dict_normalized_string(record, "accepted_ric")?.is_some() {
                continue;
            }
            if dict_truthy_bool(record, "conventional_identity_conflict")? {
                record.set_item("extension_status", "not_extended_due_to_conflict")?;
                continue;
            }

            let prior_idx = if ordered_pos > 0 {
                Some(ordered_indices[ordered_pos - 1])
            } else {
                None
            };
            let next_idx = if ordered_pos + 1 < ordered_indices.len() {
                Some(ordered_indices[ordered_pos + 1])
            } else {
                None
            };
            let prior_candidate = prior_idx
                .map(|idx| {
                    records[idx]
                        .bind(py)
                        .downcast::<PyDict>()
                        .map_err(|_| PyValueError::new_err("prior resolution row is not a dict"))
                        .and_then(accepted_candidate_with_source_from_dict)
                })
                .transpose()?
                .flatten();
            let next_candidate = next_idx
                .map(|idx| {
                    records[idx]
                        .bind(py)
                        .downcast::<PyDict>()
                        .map_err(|_| PyValueError::new_err("next resolution row is not a dict"))
                        .and_then(accepted_candidate_with_source_from_dict)
                })
                .transpose()?
                .flatten();
            let target_has_raw_conventional = dict_normalized_string(record, "ISIN")?.is_some()
                || dict_normalized_string(record, "CUSIP")?.is_some();

            if let (Some(prior), Some(next)) = (&prior_candidate, &next_candidate) {
                if !refinitiv_bridge_identity_candidates_agree_impl(
                    &prior.candidate,
                    &next.candidate,
                ) {
                    record.set_item("extension_status", "not_extended_due_to_conflict")?;
                    continue;
                }
            }

            let compatible_prior = match (prior_idx, prior_candidate.as_ref()) {
                (Some(_), Some(prior)) if target_has_raw_conventional => {
                    accepted_candidate_matches_target(record, &prior.candidate)?
                }
                _ => false,
            };
            let compatible_next = match (next_idx, next_candidate.as_ref()) {
                (Some(_), Some(next)) if target_has_raw_conventional => {
                    accepted_candidate_matches_target(record, &next.candidate)?
                }
                _ => false,
            };

            let mut chosen_source_idx: Option<usize> = None;
            let mut extension_direction: Option<&str> = None;
            let mut extension_status = "not_extended_no_adjacent_conventional_source";

            if !target_has_raw_conventional {
                if let (Some(prior), Some(next), Some(prior_idx), Some(next_idx)) = (
                    prior_candidate.as_ref(),
                    next_candidate.as_ref(),
                    prior_idx,
                    next_idx,
                ) {
                    if refinitiv_bridge_identity_candidates_agree_impl(
                        &prior.candidate,
                        &next.candidate,
                    ) {
                        let choice =
                            refinitiv_bridge_adjacent_extension_choice(&prior.source, &next.source);
                        chosen_source_idx = Some(if choice == "NEXT" {
                            next_idx
                        } else {
                            prior_idx
                        });
                        extension_direction = Some("ADJACENT");
                        extension_status = "extended_from_adjacent_conventional_span";
                    }
                }
            } else if compatible_prior && compatible_next {
                if let (Some(prior), Some(next), Some(prior_idx), Some(next_idx)) = (
                    prior_candidate.as_ref(),
                    next_candidate.as_ref(),
                    prior_idx,
                    next_idx,
                ) {
                    let choice =
                        refinitiv_bridge_adjacent_extension_choice(&prior.source, &next.source);
                    chosen_source_idx = Some(if choice == "NEXT" {
                        next_idx
                    } else {
                        prior_idx
                    });
                    extension_direction = Some("ADJACENT");
                    extension_status = "extended_from_adjacent_conventional_span";
                }
            } else if compatible_prior {
                chosen_source_idx = prior_idx;
                extension_direction = Some("PRIOR");
                extension_status = "extended_from_prior_conventional_span";
            } else if compatible_next {
                chosen_source_idx = next_idx;
                extension_direction = Some("NEXT");
                extension_status = "extended_from_next_conventional_span";
            }

            if let Some(source_idx) = chosen_source_idx {
                let source = records[source_idx]
                    .bind(py)
                    .downcast::<PyDict>()
                    .map_err(|_| PyValueError::new_err("chosen resolution row is not a dict"))?;
                record.set_item(
                    "extended_ric",
                    dict_py_object_or_none(py, source, "accepted_ric")?,
                )?;
                record.set_item(
                    "extended_from_bridge_row_id",
                    dict_py_object_or_none(py, source, "bridge_row_id")?,
                )?;
                record.set_item(
                    "extended_from_span_start",
                    dict_py_object_or_none(py, source, "first_seen_caldt")?,
                )?;
                record.set_item(
                    "extended_from_span_end",
                    dict_py_object_or_none(py, source, "last_seen_caldt")?,
                )?;
                record.set_item("extension_direction", extension_direction)?;
                record.set_item("extension_status", extension_status)?;
                chosen_source_sources[*record_idx] =
                    dict_normalized_string(source, "accepted_ric_source")?;
            } else {
                record.set_item("extension_status", extension_status)?;
            }
        }
    }

    let mut out_rows = Vec::with_capacity(records.len());
    for (record_idx, record) in records.iter().enumerate() {
        let dict = record
            .bind(py)
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("final resolution row is not a dict"))?;
        let chosen_source = chosen_source_sources[record_idx]
            .as_ref()
            .map(|value| PyString::new_bound(py, value).into_any());
        let (effective_ric, effective_source, effective_status) =
            refinitiv_bridge_effective_resolution_fields(
                dict.as_any(),
                chosen_source.as_ref().map(|value| value.as_any()),
            )?;
        dict.set_item("effective_collection_ric", effective_ric)?;
        dict.set_item("effective_collection_ric_source", effective_source)?;
        dict.set_item("effective_resolution_status", effective_status)?;

        let out = PyDict::new_bound(py);
        for column in &output_columns {
            out.set_item(column, dict_py_object_or_none(py, dict, column)?)?;
        }
        out_rows.push(out.into_py(py));
    }

    Ok(out_rows)
}

#[pyfunction]
pub(crate) fn refinitiv_bridge_resolution_frame_rows(
    py: Python<'_>,
    records: &Bound<'_, PyAny>,
    output_columns: Vec<String>,
) -> PyResult<Vec<PyObject>> {
    let rows = refinitiv_bridge_py_dict_rows_from_records(py, records, "resolution frame source")?;
    refinitiv_bridge_resolution_frame_rows_impl(py, rows, output_columns)
}

#[pyfunction]
pub(crate) fn refinitiv_bridge_resolution_frame_rows_from_columns(
    py: Python<'_>,
    column_names: Vec<String>,
    column_values: &Bound<'_, PyAny>,
    output_columns: Vec<String>,
) -> PyResult<Vec<PyObject>> {
    refinitiv_bridge_resolution_frame_rows_from_columns_impl(
        py,
        &column_names,
        column_values,
        output_columns,
    )
}

#[pyfunction]
pub(crate) fn refinitiv_bridge_resolution_diagnostic_handoff_row(
    py: Python<'_>,
    case_record: &Bound<'_, PyAny>,
    source_record: &Bound<'_, PyAny>,
    retrieval_sequence_index: i64,
    retrieval_role: &str,
    diagnostic_role: &str,
    lookup_input: &Bound<'_, PyAny>,
    lookup_input_source: &str,
) -> PyResult<Option<PyObject>> {
    let case_dict = case_record
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("resolution diagnostic case record is not a dict"))?;
    let source_dict = source_record
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("resolution diagnostic source record is not a dict"))?;

    let Some(normalized_lookup_input) =
        normalize_refinitiv_lookup_result_value(Some(lookup_input))?
    else {
        return Ok(None);
    };

    let date_type = py.import_bound("datetime")?.getattr("date")?;
    let first_seen_caldt = source_dict.get_item("first_seen_caldt")?;
    let last_seen_caldt = source_dict.get_item("last_seen_caldt")?;

    let out = PyDict::new_bound(py);
    out.set_item(
        "diagnostic_case_id",
        dict_required_py_object(py, case_dict, "diagnostic_case_id")?,
    )?;
    out.set_item(
        "case_target_bridge_row_id",
        dict_required_py_object(py, case_dict, "case_target_bridge_row_id")?,
    )?;
    out.set_item(
        "target_class",
        dict_required_py_object(py, case_dict, "target_class")?,
    )?;
    out.set_item("retrieval_sequence_index", retrieval_sequence_index)?;
    out.set_item("retrieval_role", retrieval_role)?;
    out.set_item("diagnostic_role", diagnostic_role)?;
    out.set_item(
        "bridge_row_id",
        dict_normalized_string(source_dict, "bridge_row_id")?,
    )?;
    out.set_item("KYPERMNO", dict_normalized_string(source_dict, "KYPERMNO")?)?;
    out.set_item("CUSIP", dict_normalized_string(source_dict, "CUSIP")?)?;
    out.set_item("ISIN", dict_normalized_string(source_dict, "ISIN")?)?;
    out.set_item("TICKER", dict_normalized_string(source_dict, "TICKER")?)?;
    out.set_item("lookup_input", normalized_lookup_input)?;
    out.set_item("lookup_input_source", lookup_input_source)?;
    out.set_item(
        "request_start_date",
        refinitiv_bridge_date_to_text(&date_type, first_seen_caldt.as_ref())?,
    )?;
    out.set_item(
        "request_end_date",
        refinitiv_bridge_date_to_text(&date_type, last_seen_caldt.as_ref())?,
    )?;
    out.set_item(
        "effective_collection_ric",
        dict_refinitiv_lookup_result(source_dict, "effective_collection_ric")?,
    )?;
    out.set_item(
        "effective_collection_ric_source",
        dict_normalized_string(source_dict, "effective_collection_ric_source")?,
    )?;
    out.set_item(
        "accepted_ric",
        dict_refinitiv_lookup_result(source_dict, "accepted_ric")?,
    )?;
    out.set_item(
        "accepted_ric_source",
        dict_normalized_string(source_dict, "accepted_ric_source")?,
    )?;
    out.set_item(
        "ISIN_returned_ric",
        dict_refinitiv_lookup_result(case_dict, "ISIN_returned_ric")?,
    )?;
    out.set_item(
        "CUSIP_returned_ric",
        dict_refinitiv_lookup_result(case_dict, "CUSIP_returned_ric")?,
    )?;
    out.set_item(
        "ticker_candidate_ric",
        dict_refinitiv_lookup_result(case_dict, "ticker_candidate_ric")?,
    )?;
    out.set_item(
        "case_previous_effective_collection_ric",
        dict_refinitiv_lookup_result(case_dict, "case_previous_effective_collection_ric")?,
    )?;
    out.set_item(
        "case_next_effective_collection_ric",
        dict_refinitiv_lookup_result(case_dict, "case_next_effective_collection_ric")?,
    )?;
    Ok(Some(out.into_py(py)))
}

#[pyfunction]
#[pyo3(signature = (
    vendor_primary_ric=None,
    vendor_returned_name=None,
    vendor_returned_cusip=None,
    vendor_returned_isin=None
))]
pub(crate) fn refinitiv_bridge_failed_lookup_record(
    vendor_primary_ric: Option<&Bound<'_, PyAny>>,
    vendor_returned_name: Option<&Bound<'_, PyAny>>,
    vendor_returned_cusip: Option<&Bound<'_, PyAny>>,
    vendor_returned_isin: Option<&Bound<'_, PyAny>>,
) -> PyResult<(bool, String, bool)> {
    let (failed_lookup, failed_reason, invalid_identifier_signal) =
        refinitiv_bridge_failed_lookup_record_from_normalized(
            normalize_lookup_text_any_impl(vendor_primary_ric)?,
            normalize_lookup_text_any_impl(vendor_returned_name)?,
            normalize_lookup_text_any_impl(vendor_returned_cusip)?,
            normalize_lookup_text_any_impl(vendor_returned_isin)?,
        );
    Ok((
        failed_lookup,
        failed_reason.to_string(),
        invalid_identifier_signal,
    ))
}

pub(crate) fn refinitiv_bridge_failed_lookup_record_from_normalized(
    vendor_primary_ric: Option<String>,
    vendor_returned_name: Option<String>,
    vendor_returned_cusip: Option<String>,
    vendor_returned_isin: Option<String>,
) -> (bool, &'static str, bool) {
    let returned_fields = [
        vendor_returned_name,
        vendor_returned_cusip,
        vendor_returned_isin,
    ];
    let ric_missing = vendor_primary_ric
        .as_ref()
        .is_none_or(|value| value.eq_ignore_ascii_case("NULL"));
    let invalid_identifier_signal = returned_fields.iter().any(|value| {
        value
            .as_ref()
            .is_some_and(|text| text.to_ascii_lowercase().contains("invalid identifier"))
    });
    let failed_lookup = ric_missing || invalid_identifier_signal;
    let failed_reason = if ric_missing && invalid_identifier_signal {
        "null_ric_and_invalid_identifier"
    } else if ric_missing {
        "null_ric"
    } else if invalid_identifier_signal {
        "invalid_identifier"
    } else {
        "successful_lookup"
    };
    (failed_lookup, failed_reason, invalid_identifier_signal)
}

#[pyfunction]
pub(crate) fn refinitiv_bridge_failed_lookup_records(
    records: &Bound<'_, PyAny>,
) -> PyResult<Vec<(bool, String, bool)>> {
    let mut out_rows = Vec::new();
    for row in records.iter()? {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("failed lookup record is not a dict"))?;
        let (failed_lookup, failed_reason, invalid_identifier_signal) =
            refinitiv_bridge_failed_lookup_record_from_normalized(
                dict_normalized_string(dict, "vendor_primary_ric")?,
                dict_normalized_string(dict, "vendor_returned_name")?,
                dict_normalized_string(dict, "vendor_returned_cusip")?,
                dict_normalized_string(dict, "vendor_returned_isin")?,
            );
        out_rows.push((
            failed_lookup,
            failed_reason.to_string(),
            invalid_identifier_signal,
        ));
    }
    Ok(out_rows)
}

#[pyfunction]
#[pyo3(signature = (
    diagnostic_case_id=None,
    bridge_row_id=None,
    candidate_slot=None,
    candidate_ric=None
))]
pub(crate) fn refinitiv_bridge_ownership_universe_candidate_key(
    diagnostic_case_id: Option<&Bound<'_, PyAny>>,
    bridge_row_id: Option<&Bound<'_, PyAny>>,
    candidate_slot: Option<&Bound<'_, PyAny>>,
    candidate_ric: Option<&Bound<'_, PyAny>>,
) -> PyResult<(
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
)> {
    Ok((
        normalize_lookup_text_any_impl(diagnostic_case_id)?,
        normalize_lookup_text_any_impl(bridge_row_id)?,
        normalize_lookup_text_any_impl(candidate_slot)?,
        normalize_lookup_text_any_impl(candidate_ric)?,
    ))
}

#[pyfunction]
#[pyo3(signature = (
    candidate_successful_ric=None,
    alternative_identifier=None,
    ticker=None,
    preferred_lookup_id=None
))]
pub(crate) fn refinitiv_bridge_resolve_ownership_lookup_input(
    candidate_successful_ric: Option<&Bound<'_, PyAny>>,
    alternative_identifier: Option<&Bound<'_, PyAny>>,
    ticker: Option<&Bound<'_, PyAny>>,
    preferred_lookup_id: Option<&Bound<'_, PyAny>>,
) -> PyResult<(Option<String>, &'static str)> {
    Ok(
        refinitiv_bridge_resolve_ownership_lookup_input_from_normalized(
            normalize_lookup_text_any_impl(candidate_successful_ric)?,
            normalize_lookup_text_any_impl(alternative_identifier)?,
            normalize_lookup_text_any_impl(ticker)?,
            normalize_lookup_text_any_impl(preferred_lookup_id)?,
        ),
    )
}

pub(crate) fn refinitiv_bridge_resolve_ownership_lookup_input_from_normalized(
    candidate_successful_ric: Option<String>,
    alternative_identifier: Option<String>,
    ticker: Option<String>,
    preferred_lookup_id: Option<String>,
) -> (Option<String>, &'static str) {
    let candidates = [
        (candidate_successful_ric, "candidate_successful_ric"),
        (alternative_identifier, "alternative_identifier"),
        (ticker, "TICKER"),
        (preferred_lookup_id, "preferred_lookup_id"),
    ];
    for (value, source_name) in candidates {
        if value.is_some() {
            return (value, source_name);
        }
    }
    (None, "preferred_lookup_id")
}

pub(crate) fn refinitiv_bridge_has_conventional_source_impl(
    isin: Option<&Bound<'_, PyAny>>,
    cusip: Option<&Bound<'_, PyAny>>,
    isin_returned_ric: Option<&Bound<'_, PyAny>>,
    cusip_returned_ric: Option<&Bound<'_, PyAny>>,
) -> PyResult<bool> {
    for value in [isin, cusip, isin_returned_ric, cusip_returned_ric] {
        if normalize_lookup_text_any_impl(value)?.is_some() {
            return Ok(true);
        }
    }
    Ok(false)
}

#[pyfunction]
#[pyo3(signature = (
    conventional_identity_conflict,
    effective_collection_ric=None,
    ticker_candidate_available=false
))]
pub(crate) fn refinitiv_bridge_resolution_target_class(
    conventional_identity_conflict: bool,
    effective_collection_ric: Option<&Bound<'_, PyAny>>,
    ticker_candidate_available: bool,
) -> PyResult<Option<&'static str>> {
    if conventional_identity_conflict {
        return Ok(Some("conventional_conflict"));
    }
    if normalize_refinitiv_lookup_result_value(effective_collection_ric)?.is_none() {
        if ticker_candidate_available {
            return Ok(Some("unresolved_ticker_only_candidate"));
        }
        return Ok(Some("unresolved_no_effective_collection_ric"));
    }
    Ok(None)
}

#[pyfunction]
#[pyo3(signature = (
    isin=None,
    cusip=None,
    isin_returned_ric=None,
    cusip_returned_ric=None
))]
pub(crate) fn refinitiv_bridge_has_conventional_source(
    isin: Option<&Bound<'_, PyAny>>,
    cusip: Option<&Bound<'_, PyAny>>,
    isin_returned_ric: Option<&Bound<'_, PyAny>>,
    cusip_returned_ric: Option<&Bound<'_, PyAny>>,
) -> PyResult<bool> {
    refinitiv_bridge_has_conventional_source_impl(
        isin,
        cusip,
        isin_returned_ric,
        cusip_returned_ric,
    )
}

#[pyfunction]
pub(crate) fn refinitiv_bridge_resolution_support_scope(
    same_liid_effective_support_available: bool,
    cross_liid_effective_support_available: bool,
) -> &'static str {
    if same_liid_effective_support_available {
        "same_liid_support"
    } else if cross_liid_effective_support_available {
        "cross_liid_only_support"
    } else {
        "no_support"
    }
}

#[pyfunction]
#[pyo3(signature = (
    conventional_identity_conflict,
    extension_support_scope,
    case_any_adjacent_effective_available,
    isin=None,
    cusip=None,
    isin_returned_ric=None,
    cusip_returned_ric=None
))]
pub(crate) fn refinitiv_bridge_resolution_block_reason(
    conventional_identity_conflict: bool,
    extension_support_scope: &str,
    case_any_adjacent_effective_available: bool,
    isin: Option<&Bound<'_, PyAny>>,
    cusip: Option<&Bound<'_, PyAny>>,
    isin_returned_ric: Option<&Bound<'_, PyAny>>,
    cusip_returned_ric: Option<&Bound<'_, PyAny>>,
) -> PyResult<&'static str> {
    if conventional_identity_conflict {
        return Ok("conflicting_neighbors");
    }
    if extension_support_scope == "cross_liid_only_support" {
        return Ok("cross_liid_only");
    }
    if !refinitiv_bridge_has_conventional_source_impl(
        isin,
        cusip,
        isin_returned_ric,
        cusip_returned_ric,
    )? {
        return Ok("missing_conventional_source");
    }
    if extension_support_scope == "same_liid_support" && !case_any_adjacent_effective_available {
        return Ok("no_adjacent_same_liid_support");
    }
    if extension_support_scope == "no_support" {
        return Ok("no_effective_support_on_permno");
    }
    Ok("same_liid_support_but_not_extended")
}

#[pyfunction]
#[pyo3(signature = (left=None, right=None))]
pub(crate) fn refinitiv_bridge_match_lookup_result_values(
    left: Option<&Bound<'_, PyAny>>,
    right: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<bool>> {
    let left_normalized = normalize_refinitiv_lookup_result_value(left)?;
    let right_normalized = normalize_refinitiv_lookup_result_value(right)?;
    let (Some(left_value), Some(right_value)) = (left_normalized, right_normalized) else {
        return Ok(None);
    };
    Ok(Some(left_value == right_value))
}

#[pyfunction]
pub(crate) fn refinitiv_bridge_value_counts(
    records: &Bound<'_, PyAny>,
    field_name: &str,
) -> PyResult<BTreeMap<String, i64>> {
    let mut counts: BTreeMap<String, i64> = BTreeMap::new();
    for record in records.iter()? {
        let record = record?;
        let dict = record
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("record is not a dict"))?;
        let Some(value) = dict.get_item(field_name)? else {
            continue;
        };
        let Some(normalized) = normalize_lookup_text_any_impl(Some(&value))? else {
            continue;
        };
        *counts.entry(normalized).or_insert(0) += 1;
    }
    Ok(counts)
}

#[pyfunction]
pub(crate) fn refinitiv_bridge_count_true_records(
    records: &Bound<'_, PyAny>,
    field_name: &str,
) -> PyResult<i64> {
    let mut count = 0_i64;
    for record in records.iter()? {
        let record = record?;
        let dict = record
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("record is not a dict"))?;
        let Some(value) = dict.get_item(field_name)? else {
            continue;
        };
        if value.is_instance_of::<PyBool>() && value.extract::<bool>()? {
            count += 1;
        }
    }
    Ok(count)
}

#[pyfunction]
pub(crate) fn refinitiv_bridge_true_field_counts(
    records: &Bound<'_, PyAny>,
    field_names: Vec<String>,
) -> PyResult<BTreeMap<String, i64>> {
    let mut counts: BTreeMap<String, i64> = field_names
        .iter()
        .map(|field_name| (field_name.clone(), 0_i64))
        .collect();

    for record in records.iter()? {
        let record = record?;
        let dict = record
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("record is not a dict"))?;
        for field_name in &field_names {
            let Some(value) = dict.get_item(field_name)? else {
                continue;
            };
            if value.is_instance_of::<PyBool>() && value.extract::<bool>()? {
                if let Some(count) = counts.get_mut(field_name) {
                    *count += 1;
                }
            }
        }
    }
    Ok(counts)
}

#[pyfunction]
pub(crate) fn refinitiv_bridge_resolution_summary_counts(
    records: &Bound<'_, PyAny>,
) -> PyResult<BTreeMap<String, i64>> {
    let mut rows_with_accepted_ric = 0_i64;
    let mut rows_with_extended_ric = 0_i64;
    let mut rows_with_effective_collection_ric = 0_i64;
    let mut rows_with_ticker_only_candidates_but_no_effective_collection_ric = 0_i64;
    let mut rows_unresolved_after_accept_and_extend = 0_i64;
    let mut rows_blocked_from_extension_due_conventional_conflicts = 0_i64;
    let mut ticker_candidate_available_rows = 0_i64;
    let mut ticker_candidate_conflicts_with_conventional_rows = 0_i64;
    let mut conventional_identity_conflict_rows = 0_i64;

    for record in records.iter()? {
        let record = record?;
        let dict = record
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("record is not a dict"))?;

        let accepted_ric = match dict.get_item("accepted_ric")? {
            Some(value) => normalize_lookup_text_any_impl(Some(&value))?,
            None => None,
        };
        let extended_ric = match dict.get_item("extended_ric")? {
            Some(value) => normalize_lookup_text_any_impl(Some(&value))?,
            None => None,
        };
        let effective_collection_ric = match dict.get_item("effective_collection_ric")? {
            Some(value) => normalize_lookup_text_any_impl(Some(&value))?,
            None => None,
        };
        let effective_resolution_status = match dict.get_item("effective_resolution_status")? {
            Some(value) => normalize_lookup_text_any_impl(Some(&value))?,
            None => None,
        };
        let extension_status = match dict.get_item("extension_status")? {
            Some(value) => normalize_lookup_text_any_impl(Some(&value))?,
            None => None,
        };

        if accepted_ric.is_some() {
            rows_with_accepted_ric += 1;
        }
        if extended_ric.is_some() {
            rows_with_extended_ric += 1;
        }
        if effective_collection_ric.is_some() {
            rows_with_effective_collection_ric += 1;
        }

        let ticker_candidate_available = match dict.get_item("ticker_candidate_available")? {
            Some(value) => value.is_truthy()?,
            None => false,
        };
        if ticker_candidate_available {
            ticker_candidate_available_rows += 1;
            if accepted_ric.is_none()
                && extended_ric.is_none()
                && effective_collection_ric.is_none()
            {
                rows_with_ticker_only_candidates_but_no_effective_collection_ric += 1;
            }
        }

        if effective_resolution_status.as_deref() == Some("unresolved_after_accept_and_extend") {
            rows_unresolved_after_accept_and_extend += 1;
        }
        if extension_status.as_deref() == Some("not_extended_due_to_conflict") {
            rows_blocked_from_extension_due_conventional_conflicts += 1;
        }

        if let Some(value) = dict.get_item("ticker_candidate_conflicts_with_conventional")? {
            if value.is_instance_of::<PyBool>() && value.extract::<bool>()? {
                ticker_candidate_conflicts_with_conventional_rows += 1;
            }
        }
        if let Some(value) = dict.get_item("conventional_identity_conflict")? {
            if value.is_truthy()? {
                conventional_identity_conflict_rows += 1;
            }
        }
    }

    let mut counts: BTreeMap<String, i64> = BTreeMap::new();
    counts.insert("rows_with_accepted_ric".to_string(), rows_with_accepted_ric);
    counts.insert("rows_with_extended_ric".to_string(), rows_with_extended_ric);
    counts.insert(
        "rows_with_effective_collection_ric".to_string(),
        rows_with_effective_collection_ric,
    );
    counts.insert(
        "rows_with_ticker_only_candidates_but_no_effective_collection_ric".to_string(),
        rows_with_ticker_only_candidates_but_no_effective_collection_ric,
    );
    counts.insert(
        "rows_unresolved_after_accept_and_extend".to_string(),
        rows_unresolved_after_accept_and_extend,
    );
    counts.insert(
        "rows_blocked_from_extension_due_conventional_conflicts".to_string(),
        rows_blocked_from_extension_due_conventional_conflicts,
    );
    counts.insert(
        "ticker_candidate_available_rows".to_string(),
        ticker_candidate_available_rows,
    );
    counts.insert(
        "ticker_candidate_conflicts_with_conventional_rows".to_string(),
        ticker_candidate_conflicts_with_conventional_rows,
    );
    counts.insert(
        "conventional_identity_conflict_rows".to_string(),
        conventional_identity_conflict_rows,
    );
    Ok(counts)
}

#[derive(Default)]
pub(crate) struct ResolutionDiagnosticClassAccumulator {
    targets: i64,
    isolated_targets: i64,
    targets_with_previous_effective: i64,
    targets_with_next_effective: i64,
    targets_with_any_adjacent_effective: i64,
    targets_with_both_adjacent_effective: i64,
    extension_support_scope_counts: BTreeMap<String, i64>,
    extension_block_reason_counts: BTreeMap<String, i64>,
}

pub(crate) fn refinitiv_bridge_bump_normalized_count(
    dict: &Bound<'_, PyDict>,
    key: &str,
    counts: &mut BTreeMap<String, i64>,
) -> PyResult<()> {
    let Some(value) = dict.get_item(key)? else {
        return Ok(());
    };
    let Some(normalized) = normalize_lookup_text_any_impl(Some(&value))? else {
        return Ok(());
    };
    *counts.entry(normalized).or_insert(0) += 1;
    Ok(())
}

#[pyfunction]
pub(crate) fn refinitiv_bridge_resolution_diagnostic_class_summary(
    py: Python<'_>,
    records: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let mut grouped: BTreeMap<String, ResolutionDiagnosticClassAccumulator> = BTreeMap::new();

    for record in records.iter()? {
        let record = record?;
        let dict = record
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("record is not a dict"))?;
        let target_class = match dict.get_item("target_class")? {
            Some(value) => normalize_lookup_text_any_impl(Some(&value))?,
            None => None,
        };
        let Some(target_class) = target_class else {
            continue;
        };

        let acc = grouped.entry(target_class).or_default();
        acc.targets += 1;
        if !dict_truthy_bool(dict, "case_any_adjacent_row_available")? {
            acc.isolated_targets += 1;
        }
        if dict_truthy_bool(dict, "case_previous_effective_available")? {
            acc.targets_with_previous_effective += 1;
        }
        if dict_truthy_bool(dict, "case_next_effective_available")? {
            acc.targets_with_next_effective += 1;
        }
        if dict_truthy_bool(dict, "case_any_adjacent_effective_available")? {
            acc.targets_with_any_adjacent_effective += 1;
        }
        if dict_truthy_bool(dict, "case_both_adjacent_effective_available")? {
            acc.targets_with_both_adjacent_effective += 1;
        }
        refinitiv_bridge_bump_normalized_count(
            dict,
            "extension_support_scope",
            &mut acc.extension_support_scope_counts,
        )?;
        refinitiv_bridge_bump_normalized_count(
            dict,
            "extension_block_reason",
            &mut acc.extension_block_reason_counts,
        )?;
    }

    let out = PyDict::new_bound(py);
    for (target_class, acc) in grouped {
        let summary = PyDict::new_bound(py);
        summary.set_item("targets", acc.targets)?;
        summary.set_item("isolated_targets", acc.isolated_targets)?;
        summary.set_item(
            "targets_with_previous_effective",
            acc.targets_with_previous_effective,
        )?;
        summary.set_item(
            "targets_with_next_effective",
            acc.targets_with_next_effective,
        )?;
        summary.set_item(
            "targets_with_any_adjacent_effective",
            acc.targets_with_any_adjacent_effective,
        )?;
        summary.set_item(
            "targets_with_both_adjacent_effective",
            acc.targets_with_both_adjacent_effective,
        )?;

        let support_counts = PyDict::new_bound(py);
        for (key, value) in acc.extension_support_scope_counts {
            support_counts.set_item(key, value)?;
        }
        summary.set_item("extension_support_scope_counts", support_counts)?;

        let block_counts = PyDict::new_bound(py);
        for (key, value) in acc.extension_block_reason_counts {
            block_counts.set_item(key, value)?;
        }
        summary.set_item("extension_block_reason_counts", block_counts)?;

        out.set_item(target_class, summary)?;
    }
    Ok(out.into_py(py))
}

#[pyfunction]
#[pyo3(signature = (retrieval_role=None))]
pub(crate) fn refinitiv_bridge_ownership_validation_role_sort_key(
    retrieval_role: Option<&Bound<'_, PyAny>>,
) -> PyResult<(i64, String)> {
    let Some(normalized) = normalize_lookup_text_any_impl(retrieval_role)? else {
        return Ok((5, String::new()));
    };
    let rank = match normalized.as_str() {
        "PREVIOUS_EFFECTIVE" => 0,
        "TARGET_ISIN_CANDIDATE" => 1,
        "TARGET_CUSIP_CANDIDATE" => 2,
        "TARGET_TICKER_CANDIDATE" => 3,
        "NEXT_EFFECTIVE" => 4,
        _ => 5,
    };
    Ok((rank, normalized))
}

pub(crate) struct OwnershipValidationHandoffSourceRow {
    original_index: usize,
    record: PyObject,
    sort_rank: i64,
    retrieval_sequence_index: i64,
    bridge_row_id: String,
    retrieval_role: Option<String>,
}

#[pyfunction]
pub(crate) fn refinitiv_bridge_ownership_validation_handoff_rows(
    py: Python<'_>,
    records: &Bound<'_, PyAny>,
    source_columns: Vec<String>,
    cases_per_sheet: i64,
    case_band_height: i64,
) -> PyResult<Vec<PyObject>> {
    if cases_per_sheet <= 0 {
        return Err(PyValueError::new_err("cases_per_sheet must be positive"));
    }
    if case_band_height <= 0 {
        return Err(PyValueError::new_err("case_band_height must be positive"));
    }

    let mut grouped: BTreeMap<String, Vec<OwnershipValidationHandoffSourceRow>> = BTreeMap::new();
    for (record_index, record) in records.iter()?.enumerate() {
        let record = record?;
        let record_obj = record.clone().into_py(py);
        let dict = record.downcast::<PyDict>().map_err(|_| {
            PyValueError::new_err("ownership validation handoff source row is not a dict")
        })?;
        let Some(case_id) = dict_normalized_string(dict, "diagnostic_case_id")? else {
            continue;
        };
        let retrieval_role = dict_normalized_string(dict, "retrieval_role")?;
        let (sort_rank, _) = refinitiv_bridge_ownership_validation_role_sort_key(
            dict.get_item("retrieval_role")?.as_ref(),
        )?;
        grouped
            .entry(case_id)
            .or_default()
            .push(OwnershipValidationHandoffSourceRow {
                original_index: record_index,
                record: record_obj,
                sort_rank,
                retrieval_sequence_index: dict_python_int_or_zero(
                    dict,
                    "retrieval_sequence_index",
                )?,
                bridge_row_id: dict_normalized_string(dict, "bridge_row_id")?.unwrap_or_default(),
                retrieval_role,
            });
    }

    let mut out_rows = Vec::new();
    for (case_index, (_diagnostic_case_id, mut case_records)) in grouped.into_iter().enumerate() {
        case_records.sort_by(|left, right| {
            left.sort_rank
                .cmp(&right.sort_rank)
                .then(
                    left.retrieval_sequence_index
                        .cmp(&right.retrieval_sequence_index),
                )
                .then(left.bridge_row_id.cmp(&right.bridge_row_id))
                .then(left.original_index.cmp(&right.original_index))
        });

        let case_ordinal = case_index as i64 + 1;
        let sheet_number = ((case_ordinal - 1) / cases_per_sheet) + 1;
        let sheet_name = format!("ownership_validation_{sheet_number:03}");
        let sheet_case_index = ((case_ordinal - 1) % cases_per_sheet) + 1;
        let case_band_row_start = 1 + ((sheet_case_index - 1) * case_band_height);
        let case_band_row_end = case_band_row_start + case_band_height - 1;

        for source_row in case_records {
            let dict = source_row
                .record
                .bind(py)
                .downcast::<PyDict>()
                .map_err(|_| {
                    PyValueError::new_err("ownership validation handoff cached row is not a dict")
                })?;
            let out = PyDict::new_bound(py);
            for column in &source_columns {
                out.set_item(column, dict_py_object_or_none(py, dict, column)?)?;
            }
            out.set_item("sheet_name", sheet_name.as_str())?;
            out.set_item("sheet_case_index", sheet_case_index)?;
            out.set_item("case_band_row_start", case_band_row_start)?;
            out.set_item("case_band_row_end", case_band_row_end)?;
            out.set_item("block_slot_index", source_row.sort_rank + 1)?;
            out.set_item("block_slot_role", source_row.retrieval_role)?;
            out_rows.push(out.into_py(py));
        }
    }

    Ok(out_rows)
}

#[pyfunction]
pub(crate) fn refinitiv_bridge_ownership_validation_handoff_columns(
    py: Python<'_>,
    column_names: Vec<String>,
    column_values: &Bound<'_, PyAny>,
    source_columns: Vec<String>,
    cases_per_sheet: i64,
    case_band_height: i64,
) -> PyResult<Vec<PyObject>> {
    let rows = refinitiv_bridge_py_dict_rows_from_column_values(
        py,
        &column_names,
        column_values,
        "ownership validation handoff source",
    )?;
    let row_list = PyList::new_bound(py, rows);
    refinitiv_bridge_ownership_validation_handoff_rows(
        py,
        row_list.as_any(),
        source_columns,
        cases_per_sheet,
        case_band_height,
    )
}

pub(crate) fn ownership_py_raw_key(value: Option<&Bound<'_, PyAny>>) -> PyResult<Option<String>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    let rendered = value.str()?.to_str()?.to_string();
    if value.is_instance_of::<PyString>() {
        return Ok(Some(format!("str:{rendered}")));
    }
    let type_name = value.get_type().name()?.to_string();
    Ok(Some(format!("{type_name}:{rendered}")))
}

pub(crate) fn ownership_date_key(
    date_type: &Bound<'_, PyAny>,
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<String>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_none() || !value.is_instance(date_type)? {
        return Ok(None);
    }
    let rendered = value.getattr("isoformat")?.call0()?;
    Ok(Some(format!("date:{}", rendered.str()?.to_str()?)))
}

pub(crate) fn ownership_pair_date_key(
    date_type: &Bound<'_, PyAny>,
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<String>> {
    if let Some(date_key) = ownership_date_key(date_type, value)? {
        return Ok(Some(date_key));
    }
    ownership_py_raw_key(value)
}

#[derive(Default)]
pub(crate) struct OwnershipComparisonSummary {
    date_set: HashSet<String>,
    category_set: HashSet<String>,
    ric_set: HashSet<String>,
    pair_values: HashMap<(String, String), Option<f64>>,
    categories_by_date: HashMap<String, HashSet<String>>,
}

pub(crate) fn ownership_comparison_summary(
    date_type: &Bound<'_, PyAny>,
    rows: &Bound<'_, PyAny>,
) -> PyResult<OwnershipComparisonSummary> {
    let mut summary = OwnershipComparisonSummary::default();
    for row in rows.iter()? {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("ownership comparison row is not a dict"))?;

        let returned_date = dict.get_item("returned_date")?;
        let returned_category = dict.get_item("returned_category")?;
        let returned_value = dict.get_item("returned_value")?;
        let returned_ric = dict.get_item("returned_ric")?;

        let date_set_key = ownership_date_key(date_type, returned_date.as_ref())?;
        if let Some(key) = date_set_key.as_ref() {
            summary.date_set.insert(key.clone());
        }

        if normalize_lookup_text_any_impl(returned_category.as_ref())?.is_some() {
            if let Some(key) = ownership_py_raw_key(returned_category.as_ref())? {
                summary.category_set.insert(key);
            }
        }

        if normalize_lookup_text_any_impl(returned_ric.as_ref())?.is_some() {
            if let Some(key) = ownership_py_raw_key(returned_ric.as_ref())? {
                summary.ric_set.insert(key);
            }
        }

        let pair_date_key = ownership_pair_date_key(date_type, returned_date.as_ref())?;
        let pair_category_key = ownership_py_raw_key(returned_category.as_ref())?;
        if let (Some(pair_date_key), Some(pair_category_key)) = (pair_date_key, pair_category_key) {
            let value = match returned_value.as_ref() {
                Some(value) if !value.is_none() => py_float_like_to_finite_option(value)?,
                _ => None,
            };
            let entry = summary
                .pair_values
                .entry((pair_date_key, pair_category_key.clone()))
                .or_insert(None);
            if entry.is_none() && value.is_some() {
                *entry = value;
            }
            if let Some(date_key) = date_set_key {
                summary
                    .categories_by_date
                    .entry(date_key)
                    .or_default()
                    .insert(pair_category_key);
            }
        }
    }
    Ok(summary)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn refinitiv_bridge_compare_ownership_result_rows(
    py: Python<'_>,
    target_rows: &Bound<'_, PyAny>,
    adjacent_rows: &Bound<'_, PyAny>,
    useful_min_matched_value_pairs: i64,
    useful_min_dates: i64,
    useful_min_categories: i64,
    support_median_abs_diff_max: f64,
    support_mean_abs_diff_max: f64,
    support_max_abs_diff_max: f64,
    conflict_mean_abs_diff_min: f64,
    conflict_max_abs_diff_min: f64,
) -> PyResult<PyObject> {
    let date_type = py.import_bound("datetime")?.getattr("date")?;
    let target = ownership_comparison_summary(&date_type, target_rows)?;
    let adjacent = ownership_comparison_summary(&date_type, adjacent_rows)?;

    let overlap_dates: HashSet<String> = target
        .date_set
        .intersection(&adjacent.date_set)
        .cloned()
        .collect();
    let overlap_categories: HashSet<String> = target
        .category_set
        .intersection(&adjacent.category_set)
        .cloned()
        .collect();

    let mut overlap_date_category_pair_count = 0_i64;
    let mut matched_value_pair_count = 0_i64;
    let mut abs_diffs = Vec::new();
    let mut sum_abs_value_diff = 0.0_f64;
    let mut max_abs_value_diff: Option<f64> = None;
    for (key, target_value) in target.pair_values.iter() {
        let Some(adjacent_value) = adjacent.pair_values.get(key) else {
            continue;
        };
        overlap_date_category_pair_count += 1;
        if let (Some(target_value), Some(adjacent_value)) = (target_value, adjacent_value) {
            let diff = (target_value - adjacent_value).abs();
            matched_value_pair_count += 1;
            sum_abs_value_diff += diff;
            abs_diffs.push(diff);
            max_abs_value_diff = Some(match max_abs_value_diff {
                Some(current) => current.max(diff),
                None => diff,
            });
        }
    }

    let same_returned_ric = if !target.ric_set.is_empty() && !adjacent.ric_set.is_empty() {
        Some(target.ric_set == adjacent.ric_set)
    } else {
        None
    };

    let same_category_set_on_overlap = if overlap_dates.is_empty() {
        None
    } else {
        let mut target_overlap_categories = HashSet::new();
        let mut adjacent_overlap_categories = HashSet::new();
        for date_key in &overlap_dates {
            if let Some(categories) = target.categories_by_date.get(date_key) {
                target_overlap_categories.extend(categories.iter().cloned());
            }
            if let Some(categories) = adjacent.categories_by_date.get(date_key) {
                adjacent_overlap_categories.extend(categories.iter().cloned());
            }
        }
        Some(target_overlap_categories == adjacent_overlap_categories)
    };

    let mean_abs_value_diff = if matched_value_pair_count > 0 {
        Some(sum_abs_value_diff / matched_value_pair_count as f64)
    } else {
        None
    };
    let median_abs_value_diff = median_finite_values(&mut abs_diffs);

    let pair_has_useful_overlap = matched_value_pair_count >= useful_min_matched_value_pairs
        && overlap_dates.len() as i64 >= useful_min_dates
        && overlap_categories.len() as i64 >= useful_min_categories;
    let support_diffs_are_small = median_abs_value_diff
        .zip(mean_abs_value_diff)
        .zip(max_abs_value_diff)
        .map(|((median_diff, mean_diff), max_diff)| {
            median_diff <= support_median_abs_diff_max
                && mean_diff <= support_mean_abs_diff_max
                && max_diff <= support_max_abs_diff_max
        })
        .unwrap_or(false);
    let pair_supports_corrobation = pair_has_useful_overlap
        && same_returned_ric == Some(true)
        && same_category_set_on_overlap == Some(true)
        && support_diffs_are_small;
    let pair_supports_same_identity_ric_variant = pair_has_useful_overlap
        && same_returned_ric == Some(false)
        && same_category_set_on_overlap == Some(true)
        && support_diffs_are_small;
    let pair_conflicts = pair_has_useful_overlap
        && (same_category_set_on_overlap == Some(false)
            || mean_abs_value_diff
                .map(|value| value > conflict_mean_abs_diff_min)
                .unwrap_or(false)
            || max_abs_value_diff
                .map(|value| value > conflict_max_abs_diff_min)
                .unwrap_or(false));

    let out = PyDict::new_bound(py);
    out.set_item("overlap_date_count", overlap_dates.len() as i64)?;
    out.set_item("overlap_category_count", overlap_categories.len() as i64)?;
    out.set_item(
        "overlap_date_category_pair_count",
        overlap_date_category_pair_count,
    )?;
    out.set_item("same_returned_ric", same_returned_ric)?;
    out.set_item("same_category_set_on_overlap", same_category_set_on_overlap)?;
    out.set_item("matched_value_pair_count", matched_value_pair_count)?;
    out.set_item("mean_abs_value_diff", mean_abs_value_diff)?;
    out.set_item("median_abs_value_diff", median_abs_value_diff)?;
    out.set_item("max_abs_value_diff", max_abs_value_diff)?;
    out.set_item("pair_has_useful_overlap", pair_has_useful_overlap)?;
    out.set_item("pair_supports_corrobation", pair_supports_corrobation)?;
    out.set_item(
        "pair_supports_same_identity_ric_variant",
        pair_supports_same_identity_ric_variant,
    )?;
    out.set_item("pair_conflicts", pair_conflicts)?;
    Ok(out.into_py(py))
}

pub(crate) fn dict_python_int_or_zero(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<i64> {
    let Some(value) = dict.get_item(key)? else {
        return Ok(0);
    };
    if value.is_none() || !value.is_truthy()? {
        return Ok(0);
    }
    py_int_like_to_i64(&value)
}

pub(crate) fn dict_raw_string_matches(
    dict: &Bound<'_, PyDict>,
    key: &str,
    expected: &str,
) -> PyResult<bool> {
    Ok(dict_raw_string(dict, key)?.as_deref() == Some(expected))
}

pub(crate) fn refinitiv_bridge_date_to_text(
    date_type: &Bound<'_, PyAny>,
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<String>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    if value.is_instance(date_type)? {
        let rendered = value.getattr("isoformat")?.call0()?;
        return py_str_normalized(&rendered);
    }
    normalize_lookup_text_any_impl(Some(value))
}

#[pyfunction]
pub(crate) fn refinitiv_bridge_ownership_smoke_sample_rows(
    py: Python<'_>,
    categories: Vec<String>,
    rows: &Bound<'_, PyAny>,
) -> PyResult<Vec<PyObject>> {
    let date_type = py.import_bound("datetime")?.getattr("date")?;
    let mut out_rows = Vec::with_capacity(categories.len());

    for (row_index, row) in rows.iter()?.enumerate() {
        if row_index >= categories.len() {
            return Err(PyValueError::new_err(
                "row count exceeds sample category count",
            ));
        }
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("ownership smoke sample row is not a dict"))?;

        let candidate_successful_ric = dict_normalized_string(dict, "candidate_successful_ric")?;
        let alternative_identifier = dict_normalized_string(dict, "alternative_identifier")?;
        let ticker = dict_normalized_string(dict, "TICKER")?;
        let preferred_lookup_id = dict_normalized_string(dict, "preferred_lookup_id")?;
        let (lookup_input, lookup_input_source) =
            refinitiv_bridge_resolve_ownership_lookup_input_from_normalized(
                candidate_successful_ric.clone(),
                alternative_identifier.clone(),
                ticker.clone(),
                preferred_lookup_id.clone(),
            );

        let first_seen_caldt = dict.get_item("first_seen_caldt")?;
        let last_seen_caldt = dict.get_item("last_seen_caldt")?;
        let out = PyDict::new_bound(py);
        out.set_item("sample_category", categories[row_index].as_str())?;
        out.set_item(
            "bridge_row_id",
            dict_normalized_string(dict, "bridge_row_id")?,
        )?;
        out.set_item("KYPERMNO", dict_normalized_string(dict, "KYPERMNO")?)?;
        out.set_item("TICKER", ticker)?;
        out.set_item("lookup_input", lookup_input)?;
        out.set_item("lookup_input_source", lookup_input_source)?;
        out.set_item(
            "request_start_date",
            refinitiv_bridge_date_to_text(&date_type, first_seen_caldt.as_ref())?,
        )?;
        out.set_item(
            "request_end_date",
            refinitiv_bridge_date_to_text(&date_type, last_seen_caldt.as_ref())?,
        )?;
        out.set_item("preferred_lookup_id", preferred_lookup_id)?;
        out.set_item(
            "preferred_lookup_type",
            dict_normalized_string(dict, "preferred_lookup_type")?,
        )?;
        out.set_item("alternative_identifier", alternative_identifier)?;
        out.set_item(
            "alternative_identifier_type",
            dict_normalized_string(dict, "alternative_identifier_type")?,
        )?;
        out.set_item("candidate_successful_ric", candidate_successful_ric)?;
        out.set_item(
            "successful_row_exists_before_span",
            dict_truthy_bool(dict, "successful_row_exists_before_span")?,
        )?;
        out.set_item(
            "successful_row_exists_after_span",
            dict_truthy_bool(dict, "successful_row_exists_after_span")?,
        )?;
        out.set_item(
            "successful_row_overlap_exists",
            dict_truthy_bool(dict, "successful_row_overlap_exists")?,
        )?;
        out.set_item(
            "alternative_identifier_available",
            dict_truthy_bool(dict, "alternative_identifier_available")?,
        )?;
        out.set_item(
            "candidate_successful_ric_available",
            dict_truthy_bool(dict, "candidate_successful_ric_available")?,
        )?;
        out.set_item(
            "unique_successful_identifier_pair_count",
            dict_python_int_or_zero(dict, "unique_successful_identifier_pair_count")?,
        )?;
        out.set_item(
            "unique_successful_ric_count",
            dict_python_int_or_zero(dict, "unique_successful_ric_count")?,
        )?;
        out_rows.push(out.into_py(py));
    }

    if out_rows.len() != categories.len() {
        return Err(PyValueError::new_err(
            "sample category count exceeds row count",
        ));
    }
    Ok(out_rows)
}

#[pyfunction]
pub(crate) fn refinitiv_bridge_ownership_validation_case_rows(
    py: Python<'_>,
    handoff_records: &Bound<'_, PyAny>,
    retrieval_records: &Bound<'_, PyAny>,
    pair_records: &Bound<'_, PyAny>,
    cases: Vec<String>,
) -> PyResult<Vec<PyObject>> {
    let mut out_rows = Vec::with_capacity(cases.len());

    for diagnostic_case_id in cases {
        let mut target_class = None;
        let mut case_target_bridge_row_id = None;
        for row in handoff_records.iter()? {
            let row = row?;
            let dict = row
                .downcast::<PyDict>()
                .map_err(|_| PyValueError::new_err("handoff record is not a dict"))?;
            if dict_raw_string_matches(dict, "diagnostic_case_id", &diagnostic_case_id)? {
                target_class = dict_normalized_string(dict, "target_class")?;
                case_target_bridge_row_id =
                    dict_normalized_string(dict, "case_target_bridge_row_id")?;
                break;
            }
        }

        let mut candidate_has_ownership_data = false;
        let mut any_adjacent_effective_has_ownership_data = false;
        let mut candidate_retrieval_rows_with_data = 0_i64;
        let mut adjacent_effective_retrieval_rows_with_data = 0_i64;

        for row in retrieval_records.iter()? {
            let row = row?;
            let dict = row
                .downcast::<PyDict>()
                .map_err(|_| PyValueError::new_err("retrieval record is not a dict"))?;
            if !dict_raw_string_matches(dict, "diagnostic_case_id", &diagnostic_case_id)? {
                continue;
            }
            let role = dict_raw_string(dict, "retrieval_role")?.unwrap_or_default();
            let has_data = dict_python_int_or_zero(dict, "ownership_rows_returned")? > 0;
            if role.starts_with("TARGET_") {
                if has_data {
                    candidate_has_ownership_data = true;
                    candidate_retrieval_rows_with_data += 1;
                }
            } else if role == "PREVIOUS_EFFECTIVE" || role == "NEXT_EFFECTIVE" {
                if has_data {
                    any_adjacent_effective_has_ownership_data = true;
                    adjacent_effective_retrieval_rows_with_data += 1;
                }
            }
        }

        let mut candidate_matches_previous_effective_ownership = false;
        let mut candidate_matches_next_effective_ownership = false;
        let mut pair_supports_corrobation_count = 0_i64;
        let mut pair_supports_same_identity_ric_variant_count = 0_i64;
        let mut pair_conflicts_count = 0_i64;

        for row in pair_records.iter()? {
            let row = row?;
            let dict = row
                .downcast::<PyDict>()
                .map_err(|_| PyValueError::new_err("pair record is not a dict"))?;
            if !dict_raw_string_matches(dict, "diagnostic_case_id", &diagnostic_case_id)? {
                continue;
            }
            let supports_corrobation = dict_truthy_bool(dict, "pair_supports_corrobation")?;
            let supports_same_identity =
                dict_truthy_bool(dict, "pair_supports_same_identity_ric_variant")?;
            let support_any = supports_corrobation || supports_same_identity;
            let adjacent_direction = dict_raw_string(dict, "adjacent_direction")?;
            if adjacent_direction.as_deref() == Some("PREVIOUS") && support_any {
                candidate_matches_previous_effective_ownership = true;
            }
            if adjacent_direction.as_deref() == Some("NEXT") && support_any {
                candidate_matches_next_effective_ownership = true;
            }
            if supports_corrobation {
                pair_supports_corrobation_count += 1;
            }
            if supports_same_identity {
                pair_supports_same_identity_ric_variant_count += 1;
            }
            if dict_truthy_bool(dict, "pair_conflicts")? {
                pair_conflicts_count += 1;
            }
        }

        let candidate_matches_any_adjacent_effective_ownership =
            candidate_matches_previous_effective_ownership
                || candidate_matches_next_effective_ownership;
        let ownership_validation_bucket =
            if !candidate_has_ownership_data || !any_adjacent_effective_has_ownership_data {
                "ownership_no_useful_data"
            } else if pair_conflicts_count > 0 {
                "ownership_conflicts_with_adjacent_identity"
            } else if pair_supports_corrobation_count > 0 {
                "ownership_corrobates_candidate"
            } else if pair_supports_same_identity_ric_variant_count > 0 {
                "ownership_supports_same_identity_ric_variant"
            } else {
                "ownership_inconclusive_sparse"
            };

        let out = PyDict::new_bound(py);
        out.set_item("diagnostic_case_id", diagnostic_case_id)?;
        out.set_item("target_class", target_class)?;
        out.set_item("case_target_bridge_row_id", case_target_bridge_row_id)?;
        out.set_item("candidate_has_ownership_data", candidate_has_ownership_data)?;
        out.set_item(
            "any_adjacent_effective_has_ownership_data",
            any_adjacent_effective_has_ownership_data,
        )?;
        out.set_item(
            "candidate_matches_previous_effective_ownership",
            candidate_matches_previous_effective_ownership,
        )?;
        out.set_item(
            "candidate_matches_next_effective_ownership",
            candidate_matches_next_effective_ownership,
        )?;
        out.set_item(
            "candidate_matches_any_adjacent_effective_ownership",
            candidate_matches_any_adjacent_effective_ownership,
        )?;
        out.set_item(
            "candidate_retrieval_rows_with_data",
            candidate_retrieval_rows_with_data,
        )?;
        out.set_item(
            "adjacent_effective_retrieval_rows_with_data",
            adjacent_effective_retrieval_rows_with_data,
        )?;
        out.set_item(
            "pair_supports_corrobation_count",
            pair_supports_corrobation_count,
        )?;
        out.set_item(
            "pair_supports_same_identity_ric_variant_count",
            pair_supports_same_identity_ric_variant_count,
        )?;
        out.set_item("pair_conflicts_count", pair_conflicts_count)?;
        out.set_item("ownership_validation_bucket", ownership_validation_bucket)?;
        out_rows.push(out.into_py(py));
    }

    Ok(out_rows)
}

#[pyfunction]
pub(crate) fn refinitiv_bridge_ownership_validation_case_columns(
    py: Python<'_>,
    handoff_column_names: Vec<String>,
    handoff_column_values: &Bound<'_, PyAny>,
    retrieval_column_names: Vec<String>,
    retrieval_column_values: &Bound<'_, PyAny>,
    pair_column_names: Vec<String>,
    pair_column_values: &Bound<'_, PyAny>,
    cases: Vec<String>,
) -> PyResult<Vec<PyObject>> {
    let handoff_rows = refinitiv_bridge_py_dict_rows_from_column_values(
        py,
        &handoff_column_names,
        handoff_column_values,
        "ownership validation handoff",
    )?;
    let retrieval_rows = refinitiv_bridge_py_dict_rows_from_column_values(
        py,
        &retrieval_column_names,
        retrieval_column_values,
        "ownership validation retrieval summary",
    )?;
    let pair_rows = refinitiv_bridge_py_dict_rows_from_column_values(
        py,
        &pair_column_names,
        pair_column_values,
        "ownership validation pairwise",
    )?;
    let handoff_list = PyList::new_bound(py, handoff_rows);
    let retrieval_list = PyList::new_bound(py, retrieval_rows);
    let pair_list = PyList::new_bound(py, pair_rows);
    refinitiv_bridge_ownership_validation_case_rows(
        py,
        handoff_list.as_any(),
        retrieval_list.as_any(),
        pair_list.as_any(),
        cases,
    )
}

pub(crate) fn refinitiv_bridge_ownership_universe_snapshot_row(
    py: Python<'_>,
    record: &Bound<'_, PyDict>,
    resolution_columns: &[String],
    ownership_lookup_role: &str,
    lookup_input: Option<String>,
    lookup_input_source: Option<&str>,
    retrieval_eligible: bool,
    retrieval_exclusion_reason: Option<&str>,
) -> PyResult<PyObject> {
    let out = PyDict::new_bound(py);
    for column in resolution_columns {
        if let Some(value) = record.get_item(column)? {
            out.set_item(column, value)?;
        } else {
            out.set_item(column, Option::<String>::None)?;
        }
    }

    let bridge_row_id = dict_normalized_string(record, "bridge_row_id")?;
    let ownership_lookup_row_id = match bridge_row_id.as_ref() {
        Some(value) => format!("{value}|{ownership_lookup_role}"),
        None => ownership_lookup_role.to_string(),
    };

    out.set_item("diagnostic_case_id", bridge_row_id)?;
    out.set_item("candidate_slot", ownership_lookup_role)?;
    out.set_item("candidate_ric", lookup_input.clone())?;
    out.set_item("ownership_lookup_row_id", ownership_lookup_row_id)?;
    out.set_item("ownership_lookup_role", ownership_lookup_role)?;
    out.set_item("lookup_input", lookup_input)?;
    out.set_item("lookup_input_source", lookup_input_source)?;
    out.set_item(
        "request_start_date",
        dict_normalized_string(record, "first_seen_caldt")?,
    )?;
    out.set_item(
        "request_end_date",
        dict_normalized_string(record, "last_seen_caldt")?,
    )?;
    out.set_item("retrieval_eligible", retrieval_eligible)?;
    out.set_item("retrieval_exclusion_reason", retrieval_exclusion_reason)?;
    Ok(out.into_py(py))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn refinitiv_bridge_ownership_universe_snapshot_row_from_columns(
    py: Python<'_>,
    columns: &[Vec<PyObject>],
    column_index: &HashMap<String, usize>,
    row_idx: usize,
    resolution_columns: &[String],
    ownership_lookup_role: &str,
    lookup_input: Option<String>,
    lookup_input_source: Option<&str>,
    retrieval_eligible: bool,
    retrieval_exclusion_reason: Option<&str>,
) -> PyResult<PyObject> {
    let out = PyDict::new_bound(py);
    for column in resolution_columns {
        out.set_item(
            column,
            optional_column_pyobject(py, columns, column_index, row_idx, column),
        )?;
    }

    let bridge_row_id =
        column_normalized_string(py, columns, column_index, row_idx, "bridge_row_id")?;
    let ownership_lookup_row_id = match bridge_row_id.as_ref() {
        Some(value) => format!("{value}|{ownership_lookup_role}"),
        None => ownership_lookup_role.to_string(),
    };

    out.set_item("diagnostic_case_id", bridge_row_id)?;
    out.set_item("candidate_slot", ownership_lookup_role)?;
    out.set_item("candidate_ric", lookup_input.clone())?;
    out.set_item("ownership_lookup_row_id", ownership_lookup_row_id)?;
    out.set_item("ownership_lookup_role", ownership_lookup_role)?;
    out.set_item("lookup_input", lookup_input)?;
    out.set_item("lookup_input_source", lookup_input_source)?;
    out.set_item(
        "request_start_date",
        column_normalized_string(py, columns, column_index, row_idx, "first_seen_caldt")?,
    )?;
    out.set_item(
        "request_end_date",
        column_normalized_string(py, columns, column_index, row_idx, "last_seen_caldt")?,
    )?;
    out.set_item("retrieval_eligible", retrieval_eligible)?;
    out.set_item("retrieval_exclusion_reason", retrieval_exclusion_reason)?;
    Ok(out.into_py(py))
}

#[pyfunction]
pub(crate) fn refinitiv_bridge_ownership_universe_handoff_rows(
    py: Python<'_>,
    records: &Bound<'_, PyAny>,
    resolution_columns: Vec<String>,
    include_ticker_fallback: bool,
) -> PyResult<Vec<PyObject>> {
    let mut out_rows = Vec::new();

    for record in records.iter()? {
        let record = record?;
        let dict = record
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("ownership universe source record is not a dict"))?;

        let effective_collection_ric = dict_normalized_string(dict, "effective_collection_ric")?;
        let conventional_identity_conflict =
            dict_truthy_bool(dict, "conventional_identity_conflict")?;
        let ticker_candidate_available = dict_truthy_bool(dict, "ticker_candidate_available")?;
        let ticker_candidate_ric = dict_normalized_string(dict, "ticker_candidate_ric")?;

        if let Some(value) = effective_collection_ric {
            out_rows.push(refinitiv_bridge_ownership_universe_snapshot_row(
                py,
                dict,
                &resolution_columns,
                "UNIVERSE_EFFECTIVE",
                Some(value),
                Some("effective_collection_ric"),
                true,
                None,
            )?);
            continue;
        }

        if conventional_identity_conflict {
            let mut emitted = false;
            if let Some(value) = dict_normalized_string(dict, "ISIN_returned_ric")? {
                out_rows.push(refinitiv_bridge_ownership_universe_snapshot_row(
                    py,
                    dict,
                    &resolution_columns,
                    "UNIVERSE_TARGET_ISIN_CANDIDATE",
                    Some(value),
                    Some("ISIN_returned_ric"),
                    true,
                    None,
                )?);
                emitted = true;
            }
            if let Some(value) = dict_normalized_string(dict, "CUSIP_returned_ric")? {
                out_rows.push(refinitiv_bridge_ownership_universe_snapshot_row(
                    py,
                    dict,
                    &resolution_columns,
                    "UNIVERSE_TARGET_CUSIP_CANDIDATE",
                    Some(value),
                    Some("CUSIP_returned_ric"),
                    true,
                    None,
                )?);
                emitted = true;
            }
            if emitted {
                continue;
            }
        }

        if ticker_candidate_available
            && ticker_candidate_ric.is_some()
            && !conventional_identity_conflict
        {
            out_rows.push(refinitiv_bridge_ownership_universe_snapshot_row(
                py,
                dict,
                &resolution_columns,
                "UNIVERSE_TARGET_TICKER_CANDIDATE",
                ticker_candidate_ric,
                Some("ticker_candidate_ric"),
                include_ticker_fallback,
                if include_ticker_fallback {
                    None
                } else {
                    Some("ticker_fallback_disabled")
                },
            )?);
            continue;
        }

        out_rows.push(refinitiv_bridge_ownership_universe_snapshot_row(
            py,
            dict,
            &resolution_columns,
            "UNIVERSE_NOT_RETRIEVABLE",
            None,
            None,
            false,
            Some("no_usable_lookup_input"),
        )?);
    }

    Ok(out_rows)
}

#[pyfunction]
pub(crate) fn refinitiv_bridge_ownership_universe_handoff_columns(
    py: Python<'_>,
    column_names: Vec<String>,
    column_values: &Bound<'_, PyAny>,
    resolution_columns: Vec<String>,
    include_ticker_fallback: bool,
) -> PyResult<Vec<PyObject>> {
    let (columns, row_count) = collect_pyobject_column_values(
        py,
        &column_names,
        column_values,
        "ownership universe source",
    )?;
    let column_index = column_index_by_name(&column_names);
    let mut out_rows = Vec::new();

    for row_idx in 0..row_count {
        let effective_collection_ric = column_normalized_string(
            py,
            &columns,
            &column_index,
            row_idx,
            "effective_collection_ric",
        )?;
        let conventional_identity_conflict = column_truthy_bool(
            py,
            &columns,
            &column_index,
            row_idx,
            "conventional_identity_conflict",
        )?;
        let ticker_candidate_available = column_truthy_bool(
            py,
            &columns,
            &column_index,
            row_idx,
            "ticker_candidate_available",
        )?;
        let ticker_candidate_ric =
            column_normalized_string(py, &columns, &column_index, row_idx, "ticker_candidate_ric")?;

        if let Some(value) = effective_collection_ric {
            out_rows.push(
                refinitiv_bridge_ownership_universe_snapshot_row_from_columns(
                    py,
                    &columns,
                    &column_index,
                    row_idx,
                    &resolution_columns,
                    "UNIVERSE_EFFECTIVE",
                    Some(value),
                    Some("effective_collection_ric"),
                    true,
                    None,
                )?,
            );
            continue;
        }

        if conventional_identity_conflict {
            let mut emitted = false;
            if let Some(value) =
                column_normalized_string(py, &columns, &column_index, row_idx, "ISIN_returned_ric")?
            {
                out_rows.push(
                    refinitiv_bridge_ownership_universe_snapshot_row_from_columns(
                        py,
                        &columns,
                        &column_index,
                        row_idx,
                        &resolution_columns,
                        "UNIVERSE_TARGET_ISIN_CANDIDATE",
                        Some(value),
                        Some("ISIN_returned_ric"),
                        true,
                        None,
                    )?,
                );
                emitted = true;
            }
            if let Some(value) = column_normalized_string(
                py,
                &columns,
                &column_index,
                row_idx,
                "CUSIP_returned_ric",
            )? {
                out_rows.push(
                    refinitiv_bridge_ownership_universe_snapshot_row_from_columns(
                        py,
                        &columns,
                        &column_index,
                        row_idx,
                        &resolution_columns,
                        "UNIVERSE_TARGET_CUSIP_CANDIDATE",
                        Some(value),
                        Some("CUSIP_returned_ric"),
                        true,
                        None,
                    )?,
                );
                emitted = true;
            }
            if emitted {
                continue;
            }
        }

        if ticker_candidate_available
            && ticker_candidate_ric.is_some()
            && !conventional_identity_conflict
        {
            out_rows.push(
                refinitiv_bridge_ownership_universe_snapshot_row_from_columns(
                    py,
                    &columns,
                    &column_index,
                    row_idx,
                    &resolution_columns,
                    "UNIVERSE_TARGET_TICKER_CANDIDATE",
                    ticker_candidate_ric,
                    Some("ticker_candidate_ric"),
                    include_ticker_fallback,
                    if include_ticker_fallback {
                        None
                    } else {
                        Some("ticker_fallback_disabled")
                    },
                )?,
            );
            continue;
        }

        out_rows.push(
            refinitiv_bridge_ownership_universe_snapshot_row_from_columns(
                py,
                &columns,
                &column_index,
                row_idx,
                &resolution_columns,
                "UNIVERSE_NOT_RETRIEVABLE",
                None,
                None,
                false,
                Some("no_usable_lookup_input"),
            )?,
        );
    }

    Ok(out_rows)
}

#[pyfunction]
#[pyo3(signature = (
    target_isin=None,
    target_cusip=None,
    source_returned_isin=None,
    source_returned_cusip=None
))]
pub(crate) fn refinitiv_bridge_accepted_source_matches_target(
    target_isin: Option<&Bound<'_, PyAny>>,
    target_cusip: Option<&Bound<'_, PyAny>>,
    source_returned_isin: Option<&str>,
    source_returned_cusip: Option<&str>,
) -> PyResult<bool> {
    let target_isin = normalize_lookup_text_any_impl(target_isin)?;
    let target_cusip = normalize_lookup_text_any_impl(target_cusip)?;
    if let Some(target) = target_isin {
        return Ok(source_returned_isin.is_some_and(|value| value == target));
    }
    if let Some(target) = target_cusip {
        return Ok(source_returned_cusip.is_some_and(|value| value == target));
    }
    Ok(false)
}

#[pyfunction]
pub(crate) fn refinitiv_bridge_adjacent_extension_choice(
    prior_source: &str,
    next_source: &str,
) -> &'static str {
    if prior_source != "ISIN" && next_source == "ISIN" {
        "NEXT"
    } else {
        "PRIOR"
    }
}

#[pyfunction]
#[pyo3(signature = (
    preferred_lookup_type=None,
    preferred_lookup_id=None,
    cusip=None,
    isin=None,
    ticker=None
))]
pub(crate) fn refinitiv_bridge_alternative_identifier(
    preferred_lookup_type: Option<&Bound<'_, PyAny>>,
    preferred_lookup_id: Option<&Bound<'_, PyAny>>,
    cusip: Option<&Bound<'_, PyAny>>,
    isin: Option<&Bound<'_, PyAny>>,
    ticker: Option<&Bound<'_, PyAny>>,
) -> PyResult<(Option<String>, Option<String>)> {
    Ok(refinitiv_bridge_alternative_identifier_from_normalized(
        normalize_lookup_text_any_impl(preferred_lookup_type)?,
        normalize_lookup_text_any_impl(preferred_lookup_id)?,
        normalize_lookup_text_any_impl(cusip)?,
        normalize_lookup_text_any_impl(isin)?,
        normalize_lookup_text_any_impl(ticker)?,
    ))
}

pub(crate) fn refinitiv_bridge_alternative_identifier_from_normalized(
    preferred_lookup_type: Option<String>,
    preferred_lookup_id: Option<String>,
    cusip: Option<String>,
    isin: Option<String>,
    ticker: Option<String>,
) -> (Option<String>, Option<String>) {
    let candidates: [(Option<&String>, &str); 2] = match preferred_lookup_type.as_deref() {
        Some("ISIN") => [(cusip.as_ref(), "CUSIP"), (ticker.as_ref(), "TICKER")],
        Some("CUSIP") => [(isin.as_ref(), "ISIN"), (ticker.as_ref(), "TICKER")],
        Some("TICKER") => [(isin.as_ref(), "ISIN"), (cusip.as_ref(), "CUSIP")],
        _ => return (None, None),
    };

    for (candidate, candidate_type) in candidates {
        let Some(candidate) = candidate else {
            continue;
        };
        if preferred_lookup_id.as_ref() != Some(candidate) {
            return (Some(candidate.clone()), Some(candidate_type.to_string()));
        }
    }
    (None, None)
}

#[pyfunction]
pub(crate) fn refinitiv_bridge_alternative_identifiers(
    records: &Bound<'_, PyAny>,
) -> PyResult<Vec<(Option<String>, Option<String>)>> {
    let mut out_rows = Vec::new();
    for row in records.iter()? {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("alternative identifier record is not a dict"))?;
        out_rows.push(refinitiv_bridge_alternative_identifier_from_normalized(
            dict_normalized_string(dict, "preferred_lookup_type")?,
            dict_normalized_string(dict, "preferred_lookup_id")?,
            dict_normalized_string(dict, "CUSIP")?,
            dict_normalized_string(dict, "ISIN")?,
            dict_normalized_string(dict, "TICKER")?,
        ));
    }
    Ok(out_rows)
}

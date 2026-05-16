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
use crate::refinitiv_bridge::*;
use crate::refinitiv_excel::*;
use crate::sec_extraction::*;
use crate::sentence_cleaning::*;
use crate::sentence_quality_api::*;

pub(crate) struct AuthorityObservationValueSet {
    date_obj: PyObject,
    values: Vec<Option<f64>>,
}

pub(crate) struct AuthorityUniqueRow {
    date_ordinal: i64,
    date_obj: PyObject,
    category: String,
    value: Option<f64>,
}

pub(crate) struct RefinitivAuthorityCandidateMetricAccumulator {
    kypermno: String,
    candidate_ric: String,
    request_rows: Vec<PyObject>,
    result_rows: Vec<PyObject>,
    role_set: BTreeSet<String>,
    candidate_has_conventional_support: bool,
    candidate_has_effective_support: bool,
    candidate_has_ticker_support: bool,
    candidate_request_with_data_count: i64,
    candidate_zero_return_request_count: i64,
    candidate_bridge_start_date: Option<(i64, PyObject)>,
    candidate_bridge_end_date: Option<(i64, PyObject)>,
    observation_value_sets: BTreeMap<(i64, String), AuthorityObservationValueSet>,
    unique_row_keys: HashSet<String>,
    unique_row_set: Vec<AuthorityUniqueRow>,
}

impl RefinitivAuthorityCandidateMetricAccumulator {
    fn new(kypermno: String, candidate_ric: String) -> Self {
        Self {
            kypermno,
            candidate_ric,
            request_rows: Vec::new(),
            result_rows: Vec::new(),
            role_set: BTreeSet::new(),
            candidate_has_conventional_support: false,
            candidate_has_effective_support: false,
            candidate_has_ticker_support: false,
            candidate_request_with_data_count: 0,
            candidate_zero_return_request_count: 0,
            candidate_bridge_start_date: None,
            candidate_bridge_end_date: None,
            observation_value_sets: BTreeMap::new(),
            unique_row_keys: HashSet::new(),
            unique_row_set: Vec::new(),
        }
    }
}

pub(crate) fn refinitiv_authority_value_sort_key(value: Option<f64>) -> String {
    match value {
        Some(value) if value == 0.0 => "1:0".to_string(),
        Some(value) => format!("1:{:.17}", value),
        None => "0:".to_string(),
    }
}

pub(crate) fn refinitiv_authority_push_unique_value(
    values: &mut Vec<Option<f64>>,
    value: Option<f64>,
) {
    let exists = values.iter().any(|existing| match (*existing, value) {
        (None, None) => true,
        (Some(left), Some(right)) => left == right,
        _ => false,
    });
    if !exists {
        values.push(value);
        values.sort_by(|left, right| {
            refinitiv_authority_value_sort_key(*left)
                .cmp(&refinitiv_authority_value_sort_key(*right))
        });
    }
}

pub(crate) fn refinitiv_authority_metric_set_date_min(
    target: &mut Option<(i64, PyObject)>,
    ordinal: i64,
    date_obj: PyObject,
    py: Python<'_>,
) {
    match target {
        Some((current, _)) if *current <= ordinal => {}
        _ => *target = Some((ordinal, date_obj.clone_ref(py))),
    }
}

pub(crate) fn refinitiv_authority_metric_set_date_max(
    target: &mut Option<(i64, PyObject)>,
    ordinal: i64,
    date_obj: PyObject,
    py: Python<'_>,
) {
    match target {
        Some((current, _)) if *current >= ordinal => {}
        _ => *target = Some((ordinal, date_obj.clone_ref(py))),
    }
}

pub(crate) fn refinitiv_authority_metric_date_from_dict(
    py: Python<'_>,
    dict: &Bound<'_, PyDict>,
    key: &str,
) -> PyResult<Option<(i64, PyObject)>> {
    let Some(value) = dict.get_item(key)? else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    let (year, month, day) = extract_py_date_parts(&value)?;
    Ok(Some((date_ordinal(year, month, day)?, value.into_py(py))))
}

pub(crate) fn refinitiv_authority_metric_record_key(
    dict: &Bound<'_, PyDict>,
) -> PyResult<Option<(String, String)>> {
    let Some(kypermno) = dict_normalized_string(dict, "KYPERMNO")? else {
        return Ok(None);
    };
    let Some(candidate_ric) = dict_normalized_string(dict, "candidate_ric")? else {
        return Ok(None);
    };
    Ok(Some((kypermno, candidate_ric)))
}

pub(crate) fn refinitiv_authority_column_metric_key(
    py: Python<'_>,
    columns: &[Vec<PyObject>],
    column_index: &HashMap<String, usize>,
    row_idx: usize,
) -> PyResult<Option<(String, String)>> {
    let Some(kypermno) = column_normalized_string(py, columns, column_index, row_idx, "KYPERMNO")?
    else {
        return Ok(None);
    };
    let Some(candidate_ric) =
        column_normalized_string(py, columns, column_index, row_idx, "candidate_ric")?
    else {
        return Ok(None);
    };
    Ok(Some((kypermno, candidate_ric)))
}

pub(crate) fn refinitiv_authority_column_int_or_zero(
    py: Python<'_>,
    columns: &[Vec<PyObject>],
    column_index: &HashMap<String, usize>,
    row_idx: usize,
    column_name: &str,
) -> PyResult<i64> {
    let Some(value) = optional_column_value(columns, column_index, row_idx, column_name) else {
        return Ok(0);
    };
    let value = value.bind(py);
    if value.is_none() {
        return Ok(0);
    }
    py_int_like_to_i64(&value)
}

pub(crate) fn refinitiv_authority_column_metric_date(
    py: Python<'_>,
    columns: &[Vec<PyObject>],
    column_index: &HashMap<String, usize>,
    row_idx: usize,
    column_name: &str,
) -> PyResult<Option<(i64, PyObject)>> {
    let Some(value) = optional_column_value(columns, column_index, row_idx, column_name) else {
        return Ok(None);
    };
    let value = value.bind(py);
    if value.is_none() {
        return Ok(None);
    }
    let (year, month, day) = extract_py_date_parts(&value)?;
    Ok(Some((
        date_ordinal(year, month, day)?,
        value.clone().into_py(py),
    )))
}

pub(crate) struct RefinitivAuthorityCandidateMetricColumnAccumulator {
    kypermno: String,
    candidate_ric: String,
    candidate_request_count: i64,
    role_set: BTreeSet<String>,
    candidate_has_conventional_support: bool,
    candidate_has_effective_support: bool,
    candidate_has_ticker_support: bool,
    candidate_request_with_data_count: i64,
    candidate_zero_return_request_count: i64,
    candidate_bridge_start_date: Option<(i64, PyObject)>,
    candidate_bridge_end_date: Option<(i64, PyObject)>,
    observation_value_sets: BTreeMap<(i64, String), AuthorityObservationValueSet>,
    unique_row_keys: HashSet<String>,
    unique_row_set: Vec<AuthorityUniqueRow>,
}

impl RefinitivAuthorityCandidateMetricColumnAccumulator {
    fn new(kypermno: String, candidate_ric: String) -> Self {
        Self {
            kypermno,
            candidate_ric,
            candidate_request_count: 0,
            role_set: BTreeSet::new(),
            candidate_has_conventional_support: false,
            candidate_has_effective_support: false,
            candidate_has_ticker_support: false,
            candidate_request_with_data_count: 0,
            candidate_zero_return_request_count: 0,
            candidate_bridge_start_date: None,
            candidate_bridge_end_date: None,
            observation_value_sets: BTreeMap::new(),
            unique_row_keys: HashSet::new(),
            unique_row_set: Vec::new(),
        }
    }
}

pub(crate) fn refinitiv_authority_output_optional_py(
    py: Python<'_>,
    out: &Bound<'_, PyDict>,
    key: &str,
    value: &Option<(i64, PyObject)>,
) -> PyResult<()> {
    match value {
        Some((_ordinal, obj)) => out.set_item(key, obj.clone_ref(py)),
        None => out.set_item(key, Option::<String>::None),
    }
}

#[pyfunction]
pub(crate) fn refinitiv_authority_candidate_metric_records(
    py: Python<'_>,
    row_summary_records: &Bound<'_, PyAny>,
    results_records: &Bound<'_, PyAny>,
    conventional_roles: Vec<String>,
    ticker_role: &str,
) -> PyResult<Vec<PyObject>> {
    let conventional_roles: HashSet<String> = conventional_roles.into_iter().collect();
    let mut candidates: BTreeMap<(String, String), RefinitivAuthorityCandidateMetricAccumulator> =
        BTreeMap::new();

    for row in row_summary_records.iter()? {
        let row = row?;
        let row_obj = row.clone().into_py(py);
        let dict = row.downcast::<PyDict>().map_err(|_| {
            PyValueError::new_err("authority candidate row-summary record is not a dict")
        })?;
        let Some((kypermno, candidate_ric)) = refinitiv_authority_metric_record_key(dict)? else {
            continue;
        };
        let acc = candidates
            .entry((kypermno.clone(), candidate_ric.clone()))
            .or_insert_with(|| {
                RefinitivAuthorityCandidateMetricAccumulator::new(kypermno, candidate_ric)
            });
        acc.request_rows.push(row_obj);
        if let Some(role) = dict_normalized_string(dict, "ownership_lookup_role")? {
            if conventional_roles.contains(&role) {
                acc.candidate_has_conventional_support = true;
            }
            if role == "UNIVERSE_EFFECTIVE" {
                acc.candidate_has_effective_support = true;
            }
            if role == ticker_role {
                acc.candidate_has_ticker_support = true;
            }
            acc.role_set.insert(role);
        }
        if dict_python_int_or_zero(dict, "ownership_rows_returned")? > 0 {
            acc.candidate_request_with_data_count += 1;
        }
        if dict_truthy_bool(dict, "retrieval_row_present")?
            && dict_python_int_or_zero(dict, "ownership_rows_returned")? == 0
        {
            acc.candidate_zero_return_request_count += 1;
        }
        if let Some((ordinal, date_obj)) =
            refinitiv_authority_metric_date_from_dict(py, dict, "first_seen_caldt")?
        {
            refinitiv_authority_metric_set_date_min(
                &mut acc.candidate_bridge_start_date,
                ordinal,
                date_obj,
                py,
            );
        }
        if let Some((ordinal, date_obj)) =
            refinitiv_authority_metric_date_from_dict(py, dict, "last_seen_caldt")?
        {
            refinitiv_authority_metric_set_date_max(
                &mut acc.candidate_bridge_end_date,
                ordinal,
                date_obj,
                py,
            );
        }
    }

    for row in results_records.iter()? {
        let row = row?;
        let row_obj = row.clone().into_py(py);
        let dict = row.downcast::<PyDict>().map_err(|_| {
            PyValueError::new_err("authority candidate result record is not a dict")
        })?;
        let Some((kypermno, candidate_ric)) = refinitiv_authority_metric_record_key(dict)? else {
            continue;
        };
        let acc = candidates
            .entry((kypermno.clone(), candidate_ric.clone()))
            .or_insert_with(|| {
                RefinitivAuthorityCandidateMetricAccumulator::new(kypermno, candidate_ric)
            });
        acc.result_rows.push(row_obj);
        let Some(date_value) = dict.get_item("returned_date")? else {
            continue;
        };
        if date_value.is_none() {
            continue;
        }
        let (year, month, day) = extract_py_date_parts(&date_value)?;
        let date_ordinal = date_ordinal(year, month, day)?;
        let Some(returned_category) = dict_normalized_string(dict, "returned_category")? else {
            continue;
        };
        let returned_value = match dict.get_item("returned_value")? {
            Some(value) if !value.is_none() => py_float_like_to_finite_option(&value)?,
            _ => None,
        };
        let observation = acc
            .observation_value_sets
            .entry((date_ordinal, returned_category.clone()))
            .or_insert_with(|| AuthorityObservationValueSet {
                date_obj: date_value.clone().into_py(py),
                values: Vec::new(),
            });
        refinitiv_authority_push_unique_value(&mut observation.values, returned_value);
        let unique_key = format!(
            "{}\u{1f}{}\u{1f}{}",
            date_ordinal,
            returned_category,
            refinitiv_authority_value_sort_key(returned_value)
        );
        if acc.unique_row_keys.insert(unique_key) {
            acc.unique_row_set.push(AuthorityUniqueRow {
                date_ordinal,
                date_obj: date_value.into_py(py),
                category: returned_category,
                value: returned_value,
            });
        }
    }

    let mut permno_date_sets: HashMap<String, HashSet<i64>> = HashMap::new();
    let mut permno_row_sets: HashMap<String, HashSet<String>> = HashMap::new();
    for acc in candidates.values() {
        let date_set = permno_date_sets.entry(acc.kypermno.clone()).or_default();
        for (date_ordinal, _category) in acc.observation_value_sets.keys() {
            date_set.insert(*date_ordinal);
        }
        let row_set = permno_row_sets.entry(acc.kypermno.clone()).or_default();
        for row in &acc.unique_row_set {
            row_set.insert(format!(
                "{}\u{1f}{}\u{1f}{}",
                row.date_ordinal,
                row.category,
                refinitiv_authority_value_sort_key(row.value)
            ));
        }
    }

    let mut out_records = Vec::new();
    for acc in candidates.values_mut() {
        acc.unique_row_set.sort_by(|left, right| {
            left.date_ordinal
                .cmp(&right.date_ordinal)
                .then(left.category.cmp(&right.category))
                .then(
                    refinitiv_authority_value_sort_key(left.value)
                        .cmp(&refinitiv_authority_value_sort_key(right.value)),
                )
        });
        let ownership_dates: BTreeSet<i64> = acc
            .observation_value_sets
            .keys()
            .map(|(date_ordinal, _category)| *date_ordinal)
            .collect();
        let ownership_categories: BTreeSet<String> = acc
            .observation_value_sets
            .keys()
            .map(|(_date_ordinal, category)| category.clone())
            .collect();
        let candidate_first_ownership_date = ownership_dates.first().and_then(|date_ordinal| {
            acc.observation_value_sets
                .iter()
                .find(|((candidate_ordinal, _category), _obs)| candidate_ordinal == date_ordinal)
                .map(|(_key, obs)| (*date_ordinal, obs.date_obj.clone_ref(py)))
        });
        let candidate_last_ownership_date = ownership_dates.last().and_then(|date_ordinal| {
            acc.observation_value_sets
                .iter()
                .find(|((candidate_ordinal, _category), _obs)| candidate_ordinal == date_ordinal)
                .map(|(_key, obs)| (*date_ordinal, obs.date_obj.clone_ref(py)))
        });
        let candidate_bridge_span_day_count = match (
            acc.candidate_bridge_start_date.as_ref(),
            acc.candidate_bridge_end_date.as_ref(),
        ) {
            (Some((start, _)), Some((end, _))) if end >= start => end - start + 1,
            _ => 0,
        };
        let candidate_ownership_span_day_count =
            match (ownership_dates.first(), ownership_dates.last()) {
                (Some(start), Some(end)) if end >= start => end - start + 1,
                _ => 0,
            };
        let candidate_internal_conflicts = acc
            .observation_value_sets
            .values()
            .filter(|obs| obs.values.iter().filter(|value| value.is_some()).count() > 1)
            .count() as i64;
        let total_dates = permno_date_sets
            .get(&acc.kypermno)
            .map(|values| values.len())
            .unwrap_or(0);
        let total_rows = permno_row_sets
            .get(&acc.kypermno)
            .map(|values| values.len())
            .unwrap_or(0);
        let date_share = if total_dates > 0 {
            ownership_dates.len() as f64 / total_dates as f64
        } else {
            0.0
        };
        let row_share = if total_rows > 0 {
            acc.unique_row_set.len() as f64 / total_rows as f64
        } else {
            0.0
        };

        let out = PyDict::new_bound(py);
        out.set_item("KYPERMNO", acc.kypermno.as_str())?;
        out.set_item("candidate_ric", acc.candidate_ric.as_str())?;
        out.set_item(
            "candidate_source_family",
            refinitiv_authority_source_family_from_flags(
                acc.candidate_has_conventional_support,
                acc.candidate_has_ticker_support,
            ),
        )?;
        out.set_item(
            "candidate_has_conventional_support",
            acc.candidate_has_conventional_support,
        )?;
        out.set_item(
            "candidate_has_effective_support",
            acc.candidate_has_effective_support,
        )?;
        out.set_item(
            "candidate_has_ticker_support",
            acc.candidate_has_ticker_support,
        )?;
        out.set_item("candidate_request_count", acc.request_rows.len() as i64)?;
        out.set_item(
            "candidate_request_with_data_count",
            acc.candidate_request_with_data_count,
        )?;
        out.set_item(
            "candidate_zero_return_request_count",
            acc.candidate_zero_return_request_count,
        )?;
        out.set_item(
            "candidate_ownership_row_count",
            acc.unique_row_set.len() as i64,
        )?;
        out.set_item(
            "candidate_ownership_date_count",
            ownership_dates.len() as i64,
        )?;
        out.set_item(
            "candidate_distinct_category_count",
            ownership_categories.len() as i64,
        )?;
        refinitiv_authority_output_optional_py(
            py,
            &out,
            "candidate_first_ownership_date",
            &candidate_first_ownership_date,
        )?;
        refinitiv_authority_output_optional_py(
            py,
            &out,
            "candidate_last_ownership_date",
            &candidate_last_ownership_date,
        )?;
        refinitiv_authority_output_optional_py(
            py,
            &out,
            "candidate_bridge_start_date",
            &acc.candidate_bridge_start_date,
        )?;
        refinitiv_authority_output_optional_py(
            py,
            &out,
            "candidate_bridge_end_date",
            &acc.candidate_bridge_end_date,
        )?;
        out.set_item(
            "candidate_bridge_span_day_count",
            candidate_bridge_span_day_count,
        )?;
        out.set_item(
            "candidate_ownership_span_day_count",
            candidate_ownership_span_day_count,
        )?;
        out.set_item("candidate_ownership_date_share_within_permno", date_share)?;
        out.set_item("candidate_ownership_row_share_within_permno", row_share)?;
        out.set_item(
            "candidate_internal_same_date_same_category_differing_value_count",
            candidate_internal_conflicts,
        )?;
        out.set_item("role_set", acc.role_set.iter().cloned().collect::<Vec<_>>())?;
        let request_rows: Vec<PyObject> = acc
            .request_rows
            .iter()
            .map(|row| row.clone_ref(py))
            .collect();
        out.set_item("request_rows", PyList::new_bound(py, request_rows))?;
        let result_rows: Vec<PyObject> = acc
            .result_rows
            .iter()
            .map(|row| row.clone_ref(py))
            .collect();
        out.set_item("result_rows", PyList::new_bound(py, result_rows))?;

        let mut observation_rows: Vec<PyObject> = Vec::new();
        for ((_date_ordinal, category), obs) in &acc.observation_value_sets {
            let value_objects: Vec<PyObject> = obs
                .values
                .iter()
                .map(|value| match value {
                    Some(value) => value.into_py(py),
                    None => py.None(),
                })
                .collect();
            let values_list = PyList::new_bound(py, value_objects);
            observation_rows.push(
                PyTuple::new_bound(
                    py,
                    [
                        obs.date_obj.clone_ref(py),
                        category.clone().into_py(py),
                        values_list.into_py(py),
                    ],
                )
                .into_py(py),
            );
        }
        out.set_item(
            "observation_value_sets",
            PyList::new_bound(py, observation_rows),
        )?;

        let mut unique_rows: Vec<PyObject> = Vec::new();
        for row in &acc.unique_row_set {
            unique_rows.push(
                PyTuple::new_bound(
                    py,
                    [
                        row.date_obj.clone_ref(py),
                        row.category.clone().into_py(py),
                        match row.value {
                            Some(value) => value.into_py(py),
                            None => py.None(),
                        },
                    ],
                )
                .into_py(py),
            );
        }
        out.set_item("unique_row_set", PyList::new_bound(py, unique_rows))?;
        out_records.push(out.into_py(py));
    }

    Ok(out_records)
}

#[pyfunction]
pub(crate) fn refinitiv_authority_candidate_metric_record_columns(
    py: Python<'_>,
    row_summary_column_names: Vec<String>,
    row_summary_column_values: &Bound<'_, PyAny>,
    results_column_names: Vec<String>,
    results_column_values: &Bound<'_, PyAny>,
    conventional_roles: Vec<String>,
    ticker_role: &str,
) -> PyResult<Vec<PyObject>> {
    let conventional_roles: HashSet<String> = conventional_roles.into_iter().collect();
    let (row_summary_columns, row_summary_row_count) = collect_pyobject_column_values(
        py,
        &row_summary_column_names,
        row_summary_column_values,
        "authority candidate row-summary records",
    )?;
    let row_summary_index = column_index_by_name(&row_summary_column_names);
    let (result_columns, result_row_count) = collect_pyobject_column_values(
        py,
        &results_column_names,
        results_column_values,
        "authority candidate result records",
    )?;
    let result_index = column_index_by_name(&results_column_names);

    let mut candidates: BTreeMap<
        (String, String),
        RefinitivAuthorityCandidateMetricColumnAccumulator,
    > = BTreeMap::new();

    for row_idx in 0..row_summary_row_count {
        let Some((kypermno, candidate_ric)) = refinitiv_authority_column_metric_key(
            py,
            &row_summary_columns,
            &row_summary_index,
            row_idx,
        )?
        else {
            continue;
        };
        let acc = candidates
            .entry((kypermno.clone(), candidate_ric.clone()))
            .or_insert_with(|| {
                RefinitivAuthorityCandidateMetricColumnAccumulator::new(kypermno, candidate_ric)
            });
        acc.candidate_request_count += 1;
        if let Some(role) = column_normalized_string(
            py,
            &row_summary_columns,
            &row_summary_index,
            row_idx,
            "ownership_lookup_role",
        )? {
            if conventional_roles.contains(&role) {
                acc.candidate_has_conventional_support = true;
            }
            if role == "UNIVERSE_EFFECTIVE" {
                acc.candidate_has_effective_support = true;
            }
            if role == ticker_role {
                acc.candidate_has_ticker_support = true;
            }
            acc.role_set.insert(role);
        }
        let ownership_rows_returned = refinitiv_authority_column_int_or_zero(
            py,
            &row_summary_columns,
            &row_summary_index,
            row_idx,
            "ownership_rows_returned",
        )?;
        if ownership_rows_returned > 0 {
            acc.candidate_request_with_data_count += 1;
        }
        if column_truthy_bool(
            py,
            &row_summary_columns,
            &row_summary_index,
            row_idx,
            "retrieval_row_present",
        )? && ownership_rows_returned == 0
        {
            acc.candidate_zero_return_request_count += 1;
        }
        if let Some((ordinal, date_obj)) = refinitiv_authority_column_metric_date(
            py,
            &row_summary_columns,
            &row_summary_index,
            row_idx,
            "first_seen_caldt",
        )? {
            refinitiv_authority_metric_set_date_min(
                &mut acc.candidate_bridge_start_date,
                ordinal,
                date_obj,
                py,
            );
        }
        if let Some((ordinal, date_obj)) = refinitiv_authority_column_metric_date(
            py,
            &row_summary_columns,
            &row_summary_index,
            row_idx,
            "last_seen_caldt",
        )? {
            refinitiv_authority_metric_set_date_max(
                &mut acc.candidate_bridge_end_date,
                ordinal,
                date_obj,
                py,
            );
        }
    }

    for row_idx in 0..result_row_count {
        let Some((kypermno, candidate_ric)) =
            refinitiv_authority_column_metric_key(py, &result_columns, &result_index, row_idx)?
        else {
            continue;
        };
        let acc = candidates
            .entry((kypermno.clone(), candidate_ric.clone()))
            .or_insert_with(|| {
                RefinitivAuthorityCandidateMetricColumnAccumulator::new(kypermno, candidate_ric)
            });
        let Some(date_value) =
            optional_column_value(&result_columns, &result_index, row_idx, "returned_date")
        else {
            continue;
        };
        let date_value = date_value.bind(py);
        if date_value.is_none() {
            continue;
        }
        let (year, month, day) = extract_py_date_parts(&date_value)?;
        let date_ordinal = date_ordinal(year, month, day)?;
        let Some(returned_category) = column_normalized_string(
            py,
            &result_columns,
            &result_index,
            row_idx,
            "returned_category",
        )?
        else {
            continue;
        };
        let returned_value = optional_column_float(
            py,
            &result_columns,
            &result_index,
            row_idx,
            "returned_value",
        )?;
        let observation = acc
            .observation_value_sets
            .entry((date_ordinal, returned_category.clone()))
            .or_insert_with(|| AuthorityObservationValueSet {
                date_obj: date_value.clone().into_py(py),
                values: Vec::new(),
            });
        refinitiv_authority_push_unique_value(&mut observation.values, returned_value);
        let unique_key = format!(
            "{}\u{1f}{}\u{1f}{}",
            date_ordinal,
            returned_category,
            refinitiv_authority_value_sort_key(returned_value)
        );
        if acc.unique_row_keys.insert(unique_key) {
            acc.unique_row_set.push(AuthorityUniqueRow {
                date_ordinal,
                date_obj: date_value.into_py(py),
                category: returned_category,
                value: returned_value,
            });
        }
    }

    let mut permno_date_sets: HashMap<String, HashSet<i64>> = HashMap::new();
    let mut permno_row_sets: HashMap<String, HashSet<String>> = HashMap::new();
    for acc in candidates.values() {
        let date_set = permno_date_sets.entry(acc.kypermno.clone()).or_default();
        for (date_ordinal, _category) in acc.observation_value_sets.keys() {
            date_set.insert(*date_ordinal);
        }
        let row_set = permno_row_sets.entry(acc.kypermno.clone()).or_default();
        for row in &acc.unique_row_set {
            row_set.insert(format!(
                "{}\u{1f}{}\u{1f}{}",
                row.date_ordinal,
                row.category,
                refinitiv_authority_value_sort_key(row.value)
            ));
        }
    }

    let mut out_records = Vec::new();
    for acc in candidates.values_mut() {
        acc.unique_row_set.sort_by(|left, right| {
            left.date_ordinal
                .cmp(&right.date_ordinal)
                .then(left.category.cmp(&right.category))
                .then(
                    refinitiv_authority_value_sort_key(left.value)
                        .cmp(&refinitiv_authority_value_sort_key(right.value)),
                )
        });
        let ownership_dates: BTreeSet<i64> = acc
            .observation_value_sets
            .keys()
            .map(|(date_ordinal, _category)| *date_ordinal)
            .collect();
        let ownership_categories: BTreeSet<String> = acc
            .observation_value_sets
            .keys()
            .map(|(_date_ordinal, category)| category.clone())
            .collect();
        let candidate_first_ownership_date = ownership_dates.first().and_then(|date_ordinal| {
            acc.observation_value_sets
                .iter()
                .find(|((candidate_ordinal, _category), _obs)| candidate_ordinal == date_ordinal)
                .map(|(_key, obs)| (*date_ordinal, obs.date_obj.clone_ref(py)))
        });
        let candidate_last_ownership_date = ownership_dates.last().and_then(|date_ordinal| {
            acc.observation_value_sets
                .iter()
                .find(|((candidate_ordinal, _category), _obs)| candidate_ordinal == date_ordinal)
                .map(|(_key, obs)| (*date_ordinal, obs.date_obj.clone_ref(py)))
        });
        let candidate_bridge_span_day_count = match (
            acc.candidate_bridge_start_date.as_ref(),
            acc.candidate_bridge_end_date.as_ref(),
        ) {
            (Some((start, _)), Some((end, _))) if end >= start => end - start + 1,
            _ => 0,
        };
        let candidate_ownership_span_day_count =
            match (ownership_dates.first(), ownership_dates.last()) {
                (Some(start), Some(end)) if end >= start => end - start + 1,
                _ => 0,
            };
        let candidate_internal_conflicts = acc
            .observation_value_sets
            .values()
            .filter(|obs| obs.values.iter().filter(|value| value.is_some()).count() > 1)
            .count() as i64;
        let total_dates = permno_date_sets
            .get(&acc.kypermno)
            .map(|values| values.len())
            .unwrap_or(0);
        let total_rows = permno_row_sets
            .get(&acc.kypermno)
            .map(|values| values.len())
            .unwrap_or(0);
        let date_share = if total_dates > 0 {
            ownership_dates.len() as f64 / total_dates as f64
        } else {
            0.0
        };
        let row_share = if total_rows > 0 {
            acc.unique_row_set.len() as f64 / total_rows as f64
        } else {
            0.0
        };

        let out = PyDict::new_bound(py);
        out.set_item("KYPERMNO", acc.kypermno.as_str())?;
        out.set_item("candidate_ric", acc.candidate_ric.as_str())?;
        out.set_item(
            "candidate_source_family",
            refinitiv_authority_source_family_from_flags(
                acc.candidate_has_conventional_support,
                acc.candidate_has_ticker_support,
            ),
        )?;
        out.set_item(
            "candidate_has_conventional_support",
            acc.candidate_has_conventional_support,
        )?;
        out.set_item(
            "candidate_has_effective_support",
            acc.candidate_has_effective_support,
        )?;
        out.set_item(
            "candidate_has_ticker_support",
            acc.candidate_has_ticker_support,
        )?;
        out.set_item("candidate_request_count", acc.candidate_request_count)?;
        out.set_item(
            "candidate_request_with_data_count",
            acc.candidate_request_with_data_count,
        )?;
        out.set_item(
            "candidate_zero_return_request_count",
            acc.candidate_zero_return_request_count,
        )?;
        out.set_item(
            "candidate_ownership_row_count",
            acc.unique_row_set.len() as i64,
        )?;
        out.set_item(
            "candidate_ownership_date_count",
            ownership_dates.len() as i64,
        )?;
        out.set_item(
            "candidate_distinct_category_count",
            ownership_categories.len() as i64,
        )?;
        refinitiv_authority_output_optional_py(
            py,
            &out,
            "candidate_first_ownership_date",
            &candidate_first_ownership_date,
        )?;
        refinitiv_authority_output_optional_py(
            py,
            &out,
            "candidate_last_ownership_date",
            &candidate_last_ownership_date,
        )?;
        refinitiv_authority_output_optional_py(
            py,
            &out,
            "candidate_bridge_start_date",
            &acc.candidate_bridge_start_date,
        )?;
        refinitiv_authority_output_optional_py(
            py,
            &out,
            "candidate_bridge_end_date",
            &acc.candidate_bridge_end_date,
        )?;
        out.set_item(
            "candidate_bridge_span_day_count",
            candidate_bridge_span_day_count,
        )?;
        out.set_item(
            "candidate_ownership_span_day_count",
            candidate_ownership_span_day_count,
        )?;
        out.set_item("candidate_ownership_date_share_within_permno", date_share)?;
        out.set_item("candidate_ownership_row_share_within_permno", row_share)?;
        out.set_item(
            "candidate_internal_same_date_same_category_differing_value_count",
            candidate_internal_conflicts,
        )?;
        out.set_item("role_set", acc.role_set.iter().cloned().collect::<Vec<_>>())?;

        let mut observation_rows: Vec<PyObject> = Vec::new();
        for ((_date_ordinal, category), obs) in &acc.observation_value_sets {
            let value_objects: Vec<PyObject> = obs
                .values
                .iter()
                .map(|value| match value {
                    Some(value) => value.into_py(py),
                    None => py.None(),
                })
                .collect();
            let values_list = PyList::new_bound(py, value_objects);
            observation_rows.push(
                PyTuple::new_bound(
                    py,
                    [
                        obs.date_obj.clone_ref(py),
                        category.clone().into_py(py),
                        values_list.into_py(py),
                    ],
                )
                .into_py(py),
            );
        }
        out.set_item(
            "observation_value_sets",
            PyList::new_bound(py, observation_rows),
        )?;

        let mut unique_rows: Vec<PyObject> = Vec::new();
        for row in &acc.unique_row_set {
            unique_rows.push(
                PyTuple::new_bound(
                    py,
                    [
                        row.date_obj.clone_ref(py),
                        row.category.clone().into_py(py),
                        match row.value {
                            Some(value) => value.into_py(py),
                            None => py.None(),
                        },
                    ],
                )
                .into_py(py),
            );
        }
        out.set_item("unique_row_set", PyList::new_bound(py, unique_rows))?;
        out_records.push(out.into_py(py));
    }

    Ok(out_records)
}

#[derive(Clone)]
pub(crate) struct RefinitivAuthorityPanelAssignment {
    authoritative_ric: Option<String>,
    authoritative_source_family: String,
    authoritative_component_id: String,
    authority_decision_status: String,
}

pub(crate) struct RefinitivAuthorityPanelResultRow {
    original_index: usize,
    candidate_ric: String,
    returned_date: PyObject,
    returned_category: String,
    returned_value: Option<f64>,
    ownership_lookup_role: Option<String>,
}

#[derive(Default)]
pub(crate) struct RefinitivAuthorityPanelAccumulator {
    assignments_by_permno: HashMap<String, HashMap<String, RefinitivAuthorityPanelAssignment>>,
    permno_order: Vec<String>,
    groups: HashMap<(String, String, String), Vec<RefinitivAuthorityPanelResultRow>>,
    group_order_by_permno: HashMap<String, Vec<(String, String, String)>>,
}

pub(crate) fn refinitiv_authority_panel_date_key(
    value: &Bound<'_, PyAny>,
) -> PyResult<Option<String>> {
    if value.is_none() {
        return Ok(None);
    }
    if let Ok((year, month, day)) = extract_py_date_parts(value) {
        return Ok(Some(format!("{year:04}-{month:02}-{day:02}")));
    }
    let rendered = value.str()?.to_str()?.to_string();
    if rendered.trim().is_empty() {
        Ok(None)
    } else {
        Ok(Some(rendered))
    }
}

pub(crate) fn refinitiv_authority_values_match_impl(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1e-6
}

pub(crate) fn refinitiv_authority_load_panel_assignments(
    records: &Bound<'_, PyAny>,
) -> PyResult<(
    HashMap<String, HashMap<String, RefinitivAuthorityPanelAssignment>>,
    Vec<String>,
)> {
    let mut assignments_by_permno: HashMap<
        String,
        HashMap<String, RefinitivAuthorityPanelAssignment>,
    > = HashMap::new();
    let mut permno_order = Vec::new();
    let mut seen_permnos = HashSet::new();
    for record in records.iter()? {
        let record = record?;
        let dict = record
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("authority panel assignment row is not a dict"))?;
        let Some(kypermno) = dict_normalized_string(dict, "KYPERMNO")? else {
            continue;
        };
        let Some(source_candidate_ric) = dict_normalized_string(dict, "source_candidate_ric")?
        else {
            continue;
        };
        if seen_permnos.insert(kypermno.clone()) {
            permno_order.push(kypermno.clone());
        }
        assignments_by_permno.entry(kypermno).or_default().insert(
            source_candidate_ric,
            RefinitivAuthorityPanelAssignment {
                authoritative_ric: dict_normalized_string(dict, "authoritative_ric")?,
                authoritative_source_family: dict_normalized_string(
                    dict,
                    "authoritative_source_family",
                )?
                .unwrap_or_default(),
                authoritative_component_id: dict_normalized_string(
                    dict,
                    "authoritative_component_id",
                )?
                .unwrap_or_default(),
                authority_decision_status: dict_normalized_string(
                    dict,
                    "authority_decision_status",
                )?
                .unwrap_or_default(),
            },
        );
    }
    Ok((assignments_by_permno, permno_order))
}

pub(crate) fn refinitiv_authority_panel_push_row(
    acc: &mut RefinitivAuthorityPanelAccumulator,
    py: Python<'_>,
    original_index: usize,
    kypermno: Option<String>,
    candidate_ric: Option<String>,
    returned_date: Option<&PyObject>,
    returned_date_bound: Option<&Bound<'_, PyAny>>,
    returned_category: Option<String>,
    returned_value: Option<f64>,
    ownership_lookup_role: Option<String>,
) -> PyResult<()> {
    let (
        Some(kypermno),
        Some(candidate_ric),
        Some(returned_date),
        Some(returned_date_bound),
        Some(returned_category),
    ) = (
        kypermno,
        candidate_ric,
        returned_date,
        returned_date_bound,
        returned_category,
    )
    else {
        return Ok(());
    };
    let Some(candidate_assignments) = acc.assignments_by_permno.get(&kypermno) else {
        return Ok(());
    };
    if !candidate_assignments.contains_key(&candidate_ric) {
        return Ok(());
    }
    let Some(date_key) = refinitiv_authority_panel_date_key(returned_date_bound)? else {
        return Ok(());
    };
    let group_key = (kypermno.clone(), date_key, returned_category.clone());
    if !acc.groups.contains_key(&group_key) {
        acc.group_order_by_permno
            .entry(kypermno.clone())
            .or_default()
            .push(group_key.clone());
    }
    acc.groups
        .entry(group_key)
        .or_default()
        .push(RefinitivAuthorityPanelResultRow {
            original_index,
            candidate_ric,
            returned_date: returned_date.clone_ref(py),
            returned_category,
            returned_value,
            ownership_lookup_role,
        });
    Ok(())
}

pub(crate) fn refinitiv_authority_panel_outputs(
    py: Python<'_>,
    acc: RefinitivAuthorityPanelAccumulator,
) -> PyResult<(Vec<PyObject>, Vec<String>)> {
    let mut panel_rows = Vec::new();
    let mut conflict_permnos = Vec::new();

    for kypermno in acc.permno_order {
        let Some(candidate_assignments) = acc.assignments_by_permno.get(&kypermno) else {
            continue;
        };
        let mut panel_rows_for_permno = Vec::new();
        let mut panel_conflict_detected = false;
        for group_key in acc
            .group_order_by_permno
            .get(&kypermno)
            .into_iter()
            .flatten()
        {
            let Some(group_rows) = acc.groups.get(group_key) else {
                continue;
            };
            let mut unique_values: Vec<f64> = Vec::new();
            for group_row in group_rows {
                let Some(value) = group_row.returned_value else {
                    continue;
                };
                if !unique_values
                    .iter()
                    .any(|seen| refinitiv_authority_values_match_impl(value, *seen))
                {
                    unique_values.push(value);
                }
            }
            if unique_values.len() > 1 {
                panel_conflict_detected = true;
                break;
            }
            let Some(preferred_row) = group_rows.iter().min_by(|left, right| {
                let left_assignment = candidate_assignments.get(&left.candidate_ric);
                let right_assignment = candidate_assignments.get(&right.candidate_ric);
                let left_family_rank = if left_assignment.is_some_and(|assignment| {
                    assignment.authoritative_source_family == "CONVENTIONAL"
                }) {
                    0
                } else {
                    1
                };
                let right_family_rank = if right_assignment.is_some_and(|assignment| {
                    assignment.authoritative_source_family == "CONVENTIONAL"
                }) {
                    0
                } else {
                    1
                };
                left_family_rank
                    .cmp(&right_family_rank)
                    .then(
                        (if left.ownership_lookup_role.as_deref() == Some("UNIVERSE_EFFECTIVE") {
                            0
                        } else {
                            1
                        })
                        .cmp(
                            &(if right.ownership_lookup_role.as_deref()
                                == Some("UNIVERSE_EFFECTIVE")
                            {
                                0
                            } else {
                                1
                            }),
                        ),
                    )
                    .then(left.candidate_ric.cmp(&right.candidate_ric))
                    .then(left.original_index.cmp(&right.original_index))
            }) else {
                continue;
            };
            let Some(assignment) = candidate_assignments.get(&preferred_row.candidate_ric) else {
                continue;
            };
            let out = PyDict::new_bound(py);
            out.set_item("KYPERMNO", kypermno.as_str())?;
            out.set_item("authoritative_ric", assignment.authoritative_ric.clone())?;
            out.set_item(
                "authoritative_source_family",
                assignment.authoritative_source_family.as_str(),
            )?;
            out.set_item(
                "authoritative_component_id",
                assignment.authoritative_component_id.as_str(),
            )?;
            out.set_item(
                "authority_decision_status",
                assignment.authority_decision_status.as_str(),
            )?;
            out.set_item("source_candidate_ric", preferred_row.candidate_ric.as_str())?;
            out.set_item("returned_date", preferred_row.returned_date.clone_ref(py))?;
            out.set_item(
                "returned_category",
                preferred_row.returned_category.as_str(),
            )?;
            match unique_values.first() {
                Some(value) => out.set_item("returned_value", *value)?,
                None => out.set_item("returned_value", Option::<f64>::None)?,
            }
            panel_rows_for_permno.push(out.into_py(py));
        }
        if panel_conflict_detected {
            conflict_permnos.push(kypermno);
        } else {
            panel_rows.extend(panel_rows_for_permno);
        }
    }

    Ok((panel_rows, conflict_permnos))
}

#[pyfunction]
pub(crate) fn refinitiv_authority_final_panel_rows(
    py: Python<'_>,
    results_records: &Bound<'_, PyAny>,
    assignment_records: &Bound<'_, PyAny>,
) -> PyResult<(Vec<PyObject>, Vec<String>)> {
    let (assignments_by_permno, permno_order) =
        refinitiv_authority_load_panel_assignments(assignment_records)?;
    let mut acc = RefinitivAuthorityPanelAccumulator {
        assignments_by_permno,
        permno_order,
        ..Default::default()
    };
    for (row_idx, record) in results_records.iter()?.enumerate() {
        let record = record?;
        let dict = record
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("authority ownership result row is not a dict"))?;
        let returned_date = dict.get_item("returned_date")?;
        let returned_date_obj = returned_date.as_ref().map(|value| value.into_py(py));
        let returned_value = match dict.get_item("returned_value")? {
            Some(value) => py_float_like_to_finite_option(&value)?,
            None => None,
        };
        refinitiv_authority_panel_push_row(
            &mut acc,
            py,
            row_idx,
            dict_normalized_string(dict, "KYPERMNO")?,
            dict_normalized_string(dict, "candidate_ric")?,
            returned_date_obj.as_ref(),
            returned_date.as_ref(),
            dict_normalized_string(dict, "returned_category")?,
            returned_value,
            dict_normalized_string(dict, "ownership_lookup_role")?,
        )?;
    }
    refinitiv_authority_panel_outputs(py, acc)
}

#[pyfunction]
pub(crate) fn refinitiv_authority_final_panel_rows_columns(
    py: Python<'_>,
    results_column_names: Vec<String>,
    results_column_values: &Bound<'_, PyAny>,
    assignment_records: &Bound<'_, PyAny>,
) -> PyResult<(Vec<PyObject>, Vec<String>)> {
    let (assignments_by_permno, permno_order) =
        refinitiv_authority_load_panel_assignments(assignment_records)?;
    let (columns, row_count) = collect_pyobject_column_values(
        py,
        &results_column_names,
        results_column_values,
        "authority ownership result rows",
    )?;
    let column_index = column_index_by_name(&results_column_names);
    let mut acc = RefinitivAuthorityPanelAccumulator {
        assignments_by_permno,
        permno_order,
        ..Default::default()
    };
    for row_idx in 0..row_count {
        let returned_date =
            optional_column_value(&columns, &column_index, row_idx, "returned_date");
        refinitiv_authority_panel_push_row(
            &mut acc,
            py,
            row_idx,
            column_normalized_string(py, &columns, &column_index, row_idx, "KYPERMNO")?,
            column_normalized_string(py, &columns, &column_index, row_idx, "candidate_ric")?,
            returned_date,
            returned_date.map(|value| value.bind(py)),
            column_normalized_string(py, &columns, &column_index, row_idx, "returned_category")?,
            optional_column_float(py, &columns, &column_index, row_idx, "returned_value")?,
            column_normalized_string(
                py,
                &columns,
                &column_index,
                row_idx,
                "ownership_lookup_role",
            )?,
        )?;
    }
    refinitiv_authority_panel_outputs(py, acc)
}

#[pyfunction]
pub(crate) fn refinitiv_authority_review_required_rows_columns(
    py: Python<'_>,
    decision_column_names: Vec<String>,
    decision_column_values: &Bound<'_, PyAny>,
    candidate_column_names: Vec<String>,
    candidate_column_values: &Bound<'_, PyAny>,
    authority_decision_columns: Vec<String>,
) -> PyResult<Vec<PyObject>> {
    let (decision_columns, decision_row_count) = collect_pyobject_column_values(
        py,
        &decision_column_names,
        decision_column_values,
        "authority decision rows",
    )?;
    let decision_index = column_index_by_name(&decision_column_names);
    let (candidate_columns, candidate_row_count) = collect_pyobject_column_values(
        py,
        &candidate_column_names,
        candidate_column_values,
        "authority candidate metric rows",
    )?;
    let candidate_index = column_index_by_name(&candidate_column_names);
    let mut conventional_by_permno: HashMap<String, BTreeSet<String>> = HashMap::new();
    let mut ticker_by_permno: HashMap<String, BTreeSet<String>> = HashMap::new();
    for row_idx in 0..candidate_row_count {
        let Some(kypermno) = column_normalized_string(
            py,
            &candidate_columns,
            &candidate_index,
            row_idx,
            "KYPERMNO",
        )?
        else {
            continue;
        };
        let Some(candidate_ric) = column_normalized_string(
            py,
            &candidate_columns,
            &candidate_index,
            row_idx,
            "candidate_ric",
        )?
        else {
            continue;
        };
        if column_truthy_bool(
            py,
            &candidate_columns,
            &candidate_index,
            row_idx,
            "candidate_has_conventional_support",
        )? {
            conventional_by_permno
                .entry(kypermno.clone())
                .or_default()
                .insert(candidate_ric.clone());
        }
        if column_truthy_bool(
            py,
            &candidate_columns,
            &candidate_index,
            row_idx,
            "candidate_has_ticker_support",
        )? {
            ticker_by_permno
                .entry(kypermno)
                .or_default()
                .insert(candidate_ric);
        }
    }

    let mut out_rows = Vec::new();
    for row_idx in 0..decision_row_count {
        if !column_truthy_bool(
            py,
            &decision_columns,
            &decision_index,
            row_idx,
            "requires_review",
        )? {
            continue;
        }
        let kypermno =
            column_normalized_string(py, &decision_columns, &decision_index, row_idx, "KYPERMNO")?
                .unwrap_or_default();
        let out = PyDict::new_bound(py);
        for column in &authority_decision_columns {
            out.set_item(
                column,
                optional_column_pyobject(py, &decision_columns, &decision_index, row_idx, column),
            )?;
        }
        let conventional = conventional_by_permno
            .get(&kypermno)
            .map(|values| values.iter().cloned().collect::<Vec<_>>().join("|"))
            .unwrap_or_default();
        let ticker = ticker_by_permno
            .get(&kypermno)
            .map(|values| values.iter().cloned().collect::<Vec<_>>().join("|"))
            .unwrap_or_default();
        out.set_item("conventional_candidate_rics", conventional)?;
        out.set_item("ticker_candidate_rics", ticker)?;
        out_rows.push(out.into_py(py));
    }

    Ok(out_rows)
}

#[derive(Clone)]
pub(crate) struct RefinitivAuthorityAliasCandidateRecord {
    kypermno: String,
    candidate_ric: String,
    candidate_source_family: String,
    candidate_has_conventional_support: bool,
    candidate_has_effective_support: bool,
    candidate_first_ownership_date: Option<i64>,
    candidate_last_ownership_date: Option<i64>,
    candidate_bridge_start_date: Option<i64>,
    candidate_bridge_end_date: Option<i64>,
    candidate_internal_differing_value_count: i64,
    observation_value_sets: HashMap<(i64, String), Vec<f64>>,
}

pub(crate) fn refinitiv_authority_date_ordinal_from_py(
    value: &Bound<'_, PyAny>,
) -> PyResult<Option<i64>> {
    if value.is_none() {
        return Ok(None);
    }
    let (year, month, day) = extract_py_date_parts(value)?;
    Ok(Some(date_ordinal(year, month, day)?))
}

pub(crate) fn refinitiv_authority_dict_date_ordinal(
    dict: &Bound<'_, PyDict>,
    key: &str,
) -> PyResult<Option<i64>> {
    let Some(value) = dict.get_item(key)? else {
        return Ok(None);
    };
    refinitiv_authority_date_ordinal_from_py(&value)
}

pub(crate) fn refinitiv_authority_distinct_float_values(
    values: &Bound<'_, PyAny>,
) -> PyResult<Vec<f64>> {
    let mut out = Vec::new();
    for value in values.iter()? {
        let value = value?;
        if value.is_none() {
            continue;
        }
        let number = value.extract::<f64>()?;
        if !number.is_finite() {
            return Err(PyValueError::new_err(
                "non-finite authority value falls back to Python",
            ));
        }
        out.push(number);
    }
    out.sort_by(|left, right| left.total_cmp(right));
    Ok(out)
}

pub(crate) fn refinitiv_authority_alias_candidate_record(
    dict: &Bound<'_, PyDict>,
) -> PyResult<RefinitivAuthorityAliasCandidateRecord> {
    let observation_value_sets_value = dict
        .get_item("observation_value_sets")?
        .ok_or_else(|| PyValueError::new_err("missing observation_value_sets"))?;
    let observation_value_sets_dict = observation_value_sets_value
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("observation_value_sets is not a dict"))?;
    let mut observation_value_sets: HashMap<(i64, String), Vec<f64>> = HashMap::new();
    for (key, values) in observation_value_sets_dict.iter() {
        let tuple = key
            .downcast::<PyTuple>()
            .map_err(|_| PyValueError::new_err("observation key is not a tuple"))?;
        if tuple.len() != 2 {
            return Err(PyValueError::new_err(
                "observation key must contain date and category",
            ));
        }
        let date_value = tuple.get_item(0)?;
        let Some(date_ordinal) = refinitiv_authority_date_ordinal_from_py(&date_value)? else {
            continue;
        };
        let category = tuple.get_item(1)?.str()?.to_str()?.to_string();
        observation_value_sets.insert(
            (date_ordinal, category),
            refinitiv_authority_distinct_float_values(&values)?,
        );
    }

    Ok(RefinitivAuthorityAliasCandidateRecord {
        kypermno: dict_required_string(dict, "KYPERMNO")?,
        candidate_ric: dict_required_string(dict, "candidate_ric")?,
        candidate_source_family: dict_required_string(dict, "candidate_source_family")?,
        candidate_has_conventional_support: dict_truthy_bool(
            dict,
            "candidate_has_conventional_support",
        )?,
        candidate_has_effective_support: dict_truthy_bool(dict, "candidate_has_effective_support")?,
        candidate_first_ownership_date: refinitiv_authority_dict_date_ordinal(
            dict,
            "candidate_first_ownership_date",
        )?,
        candidate_last_ownership_date: refinitiv_authority_dict_date_ordinal(
            dict,
            "candidate_last_ownership_date",
        )?,
        candidate_bridge_start_date: refinitiv_authority_dict_date_ordinal(
            dict,
            "candidate_bridge_start_date",
        )?,
        candidate_bridge_end_date: refinitiv_authority_dict_date_ordinal(
            dict,
            "candidate_bridge_end_date",
        )?,
        candidate_internal_differing_value_count: dict_python_int_or_zero(
            dict,
            "candidate_internal_same_date_same_category_differing_value_count",
        )?,
        observation_value_sets,
    })
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn refinitiv_authority_pairwise_alias_diagnostic_rows(
    py: Python<'_>,
    permno_order: Vec<String>,
    candidate_records: &Bound<'_, PyAny>,
    value_equality_tolerance: f64,
    benign_min_overlap_date_category_count: i64,
    benign_min_overlap_date_count: i64,
    benign_min_overlap_category_count: i64,
    benign_min_overlap_share_of_smaller_history: f64,
) -> PyResult<Vec<PyObject>> {
    let mut candidates_by_permno: HashMap<String, Vec<RefinitivAuthorityAliasCandidateRecord>> =
        HashMap::new();
    for record in candidate_records.iter()? {
        let record = record?;
        let dict = record
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("authority alias candidate record is not a dict"))?;
        let candidate = refinitiv_authority_alias_candidate_record(dict)?;
        candidates_by_permno
            .entry(candidate.kypermno.clone())
            .or_default()
            .push(candidate);
    }
    for candidates in candidates_by_permno.values_mut() {
        candidates.sort_by(|left, right| left.candidate_ric.cmp(&right.candidate_ric));
    }

    let mut rows = Vec::new();
    for kypermno in permno_order {
        let Some(candidates) = candidates_by_permno.get(&kypermno) else {
            continue;
        };
        for left_index in 0..candidates.len() {
            for right_index in (left_index + 1)..candidates.len() {
                let left = &candidates[left_index];
                let right = &candidates[right_index];
                let left_keys: HashSet<(i64, String)> =
                    left.observation_value_sets.keys().cloned().collect();
                let right_keys: HashSet<(i64, String)> =
                    right.observation_value_sets.keys().cloned().collect();
                let overlap_pairs: HashSet<(i64, String)> =
                    left_keys.intersection(&right_keys).cloned().collect();
                let overlap_dates: HashSet<i64> =
                    overlap_pairs.iter().map(|(date, _)| *date).collect();
                let overlap_categories: HashSet<String> = overlap_pairs
                    .iter()
                    .map(|(_, category)| category.clone())
                    .collect();
                let mut diff_values = Vec::new();
                let mut differing_value_count = 0_i64;
                for overlap_key in &overlap_pairs {
                    let left_values = left
                        .observation_value_sets
                        .get(overlap_key)
                        .expect("overlap key must be in left observations");
                    let right_values = right
                        .observation_value_sets
                        .get(overlap_key)
                        .expect("overlap key must be in right observations");
                    if left_values.len() == 1 && right_values.len() == 1 {
                        let abs_diff = (left_values[0] - right_values[0]).abs();
                        diff_values.push(abs_diff);
                        if abs_diff > value_equality_tolerance {
                            differing_value_count += 1;
                        }
                    } else if left_values != right_values {
                        differing_value_count += 1;
                    }
                }
                let smaller_history = left_keys.len().min(right_keys.len());
                let overlap_share = if smaller_history > 0 {
                    overlap_pairs.len() as f64 / smaller_history as f64
                } else {
                    0.0
                };
                let observed_disjoint = match (
                    left.candidate_first_ownership_date,
                    left.candidate_last_ownership_date,
                    right.candidate_first_ownership_date,
                    right.candidate_last_ownership_date,
                ) {
                    (Some(left_first), Some(left_last), Some(right_first), Some(right_last)) => {
                        left_last < right_first || right_last < left_first
                    }
                    _ => false,
                };
                let bridge_disjoint = match (
                    left.candidate_bridge_start_date,
                    left.candidate_bridge_end_date,
                    right.candidate_bridge_start_date,
                    right.candidate_bridge_end_date,
                ) {
                    (Some(left_start), Some(left_end), Some(right_start), Some(right_end)) => {
                        left_end < right_start || right_end < left_start
                    }
                    _ => false,
                };
                let pair_value_conflict_present = differing_value_count > 0;
                let pair_benign_alias_supported = !pair_value_conflict_present
                    && overlap_pairs.len() as i64 >= benign_min_overlap_date_category_count
                    && overlap_dates.len() as i64 >= benign_min_overlap_date_count
                    && overlap_categories.len() as i64 >= benign_min_overlap_category_count
                    && overlap_share >= benign_min_overlap_share_of_smaller_history;
                let pair_regime_split_supported = left.candidate_has_conventional_support
                    && right.candidate_has_conventional_support
                    && !pair_value_conflict_present
                    && (observed_disjoint || bridge_disjoint);
                let pair_requires_review = pair_value_conflict_present
                    || left.candidate_internal_differing_value_count > 0
                    || right.candidate_internal_differing_value_count > 0
                    || (!overlap_pairs.is_empty()
                        && !pair_benign_alias_supported
                        && !pair_regime_split_supported);

                let mean_abs_value_diff = if diff_values.is_empty() {
                    None
                } else {
                    Some(diff_values.iter().sum::<f64>() / diff_values.len() as f64)
                };
                let max_abs_value_diff = diff_values.iter().copied().reduce(f64::max);
                let mut median_values = diff_values.clone();
                let median_abs_value_diff = median_finite_values(&mut median_values);

                let row = PyDict::new_bound(py);
                row.set_item("KYPERMNO", kypermno.as_str())?;
                row.set_item("left_candidate_ric", left.candidate_ric.as_str())?;
                row.set_item("right_candidate_ric", right.candidate_ric.as_str())?;
                row.set_item(
                    "left_candidate_source_family",
                    left.candidate_source_family.as_str(),
                )?;
                row.set_item(
                    "right_candidate_source_family",
                    right.candidate_source_family.as_str(),
                )?;
                row.set_item(
                    "left_candidate_has_effective_support",
                    left.candidate_has_effective_support,
                )?;
                row.set_item(
                    "right_candidate_has_effective_support",
                    right.candidate_has_effective_support,
                )?;
                row.set_item(
                    "pair_same_returned_ric",
                    left.candidate_ric == right.candidate_ric,
                )?;
                row.set_item("pair_overlap_date_count", overlap_dates.len() as i64)?;
                row.set_item(
                    "pair_overlap_category_count",
                    overlap_categories.len() as i64,
                )?;
                row.set_item(
                    "pair_overlap_date_category_count",
                    overlap_pairs.len() as i64,
                )?;
                row.set_item("pair_overlap_share_of_smaller_history", overlap_share)?;
                row.set_item(
                    "pair_same_date_multi_ric_overlap_count",
                    overlap_dates.len() as i64,
                )?;
                row.set_item(
                    "pair_same_date_same_category_overlap_count",
                    overlap_pairs.len() as i64,
                )?;
                row.set_item(
                    "pair_same_date_same_category_differing_value_count",
                    differing_value_count,
                )?;
                row.set_item("pair_mean_abs_value_diff", mean_abs_value_diff)?;
                row.set_item("pair_median_abs_value_diff", median_abs_value_diff)?;
                row.set_item("pair_max_abs_value_diff", max_abs_value_diff)?;
                row.set_item("pair_value_conflict_present", pair_value_conflict_present)?;
                row.set_item(
                    "pair_disjoint_history",
                    observed_disjoint || bridge_disjoint,
                )?;
                row.set_item("pair_benign_alias_supported", pair_benign_alias_supported)?;
                row.set_item("pair_regime_split_supported", pair_regime_split_supported)?;
                row.set_item("pair_requires_review", pair_requires_review)?;
                rows.push(row.into_py(py));
            }
        }
    }

    Ok(rows)
}

pub(crate) struct RefinitivAuthorityComponentCandidate {
    kypermno: String,
    candidate_ric: String,
    candidate_has_conventional_support: bool,
    candidate_has_effective_support: bool,
    candidate_ownership_date_count: i64,
    candidate_bridge_span_day_count: i64,
    candidate_bridge_start_date: Option<(i64, PyObject)>,
    candidate_bridge_end_date: Option<(i64, PyObject)>,
    observation_value_sets: BTreeMap<(i64, String), (PyObject, Vec<Option<f64>>)>,
    unique_row_set: Vec<(i64, PyObject, String, Option<f64>)>,
}

#[derive(Clone, Copy, Default)]
pub(crate) struct RefinitivAuthorityComponentPair {
    pair_benign_alias_supported: bool,
    pair_value_conflict_present: bool,
    pair_regime_split_supported: bool,
    pair_requires_review: bool,
}

pub(crate) fn refinitiv_authority_sort_optional_values(values: &mut [Option<f64>]) {
    values.sort_by(|left, right| match (left, right) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
        (Some(left), Some(right)) => left.total_cmp(right),
    });
}

pub(crate) fn refinitiv_authority_push_unique_optional_value(
    values: &mut Vec<Option<f64>>,
    value: Option<f64>,
) {
    if values.iter().any(|existing| match (*existing, value) {
        (None, None) => true,
        (Some(left), Some(right)) => left == right,
        _ => false,
    }) {
        return;
    }
    values.push(value);
    refinitiv_authority_sort_optional_values(values);
}

pub(crate) fn refinitiv_authority_distinct_some_count(values: &[Option<f64>]) -> usize {
    let mut distinct: Vec<f64> = Vec::new();
    for value in values.iter().flatten() {
        if !distinct.iter().any(|existing| existing == value) {
            distinct.push(*value);
        }
    }
    distinct.len()
}

pub(crate) fn refinitiv_authority_component_candidate_record(
    py: Python<'_>,
    dict: &Bound<'_, PyDict>,
) -> PyResult<RefinitivAuthorityComponentCandidate> {
    let observation_rows = dict
        .get_item("observation_value_sets")?
        .ok_or_else(|| PyValueError::new_err("missing observation_value_sets"))?;
    let unique_rows = dict
        .get_item("unique_row_set")?
        .ok_or_else(|| PyValueError::new_err("missing unique_row_set"))?;

    let mut observation_value_sets: BTreeMap<(i64, String), (PyObject, Vec<Option<f64>>)> =
        BTreeMap::new();
    for row in observation_rows.iter()? {
        let row = row?;
        let tuple = row
            .downcast::<PyTuple>()
            .map_err(|_| PyValueError::new_err("observation row is not a tuple"))?;
        if tuple.len() != 3 {
            return Err(PyValueError::new_err(
                "observation row must contain date, category, and values",
            ));
        }
        let date_value = tuple.get_item(0)?;
        let Some(date_ordinal) = refinitiv_authority_date_ordinal_from_py(&date_value)? else {
            continue;
        };
        let category = tuple.get_item(1)?.str()?.to_str()?.to_string();
        let values_obj = tuple.get_item(2)?;
        let entry = observation_value_sets
            .entry((date_ordinal, category))
            .or_insert_with(|| (date_value.into_py(py), Vec::new()));
        for value in values_obj.iter()? {
            let value = value?;
            let normalized = if value.is_none() {
                None
            } else {
                py_float_like_to_finite_option(&value)?
            };
            refinitiv_authority_push_unique_optional_value(&mut entry.1, normalized);
        }
    }

    let mut unique_row_set = Vec::new();
    for row in unique_rows.iter()? {
        let row = row?;
        let tuple = row
            .downcast::<PyTuple>()
            .map_err(|_| PyValueError::new_err("unique row is not a tuple"))?;
        if tuple.len() != 3 {
            return Err(PyValueError::new_err(
                "unique row must contain date, category, and value",
            ));
        }
        let date_value = tuple.get_item(0)?;
        let Some(date_ordinal) = refinitiv_authority_date_ordinal_from_py(&date_value)? else {
            continue;
        };
        let category = tuple.get_item(1)?.str()?.to_str()?.to_string();
        let value_obj = tuple.get_item(2)?;
        let value = if value_obj.is_none() {
            None
        } else {
            py_float_like_to_finite_option(&value_obj)?
        };
        unique_row_set.push((date_ordinal, date_value.into_py(py), category, value));
    }
    unique_row_set.sort_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then(left.2.cmp(&right.2))
            .then_with(|| match (left.3, right.3) {
                (None, None) => Ordering::Equal,
                (None, Some(_)) => Ordering::Less,
                (Some(_), None) => Ordering::Greater,
                (Some(left), Some(right)) => left.total_cmp(&right),
            })
    });

    Ok(RefinitivAuthorityComponentCandidate {
        kypermno: dict_required_string(dict, "KYPERMNO")?,
        candidate_ric: dict_required_string(dict, "candidate_ric")?,
        candidate_has_conventional_support: dict_truthy_bool(
            dict,
            "candidate_has_conventional_support",
        )?,
        candidate_has_effective_support: dict_truthy_bool(dict, "candidate_has_effective_support")?,
        candidate_ownership_date_count: dict_python_int_or_zero(
            dict,
            "candidate_ownership_date_count",
        )?,
        candidate_bridge_span_day_count: dict_python_int_or_zero(
            dict,
            "candidate_bridge_span_day_count",
        )?,
        candidate_bridge_start_date: refinitiv_authority_metric_date_from_dict(
            py,
            dict,
            "candidate_bridge_start_date",
        )?,
        candidate_bridge_end_date: refinitiv_authority_metric_date_from_dict(
            py,
            dict,
            "candidate_bridge_end_date",
        )?,
        observation_value_sets,
        unique_row_set,
    })
}

pub(crate) fn refinitiv_authority_component_pair_record(
    dict: &Bound<'_, PyDict>,
) -> PyResult<((String, String, String), RefinitivAuthorityComponentPair)> {
    Ok((
        (
            dict_required_string(dict, "KYPERMNO")?,
            dict_required_string(dict, "left_candidate_ric")?,
            dict_required_string(dict, "right_candidate_ric")?,
        ),
        RefinitivAuthorityComponentPair {
            pair_benign_alias_supported: dict_truthy_bool(dict, "pair_benign_alias_supported")?,
            pair_value_conflict_present: dict_truthy_bool(dict, "pair_value_conflict_present")?,
            pair_regime_split_supported: dict_truthy_bool(dict, "pair_regime_split_supported")?,
            pair_requires_review: dict_truthy_bool(dict, "pair_requires_review")?,
        },
    ))
}

pub(crate) fn refinitiv_authority_component_candidate_meta_records(
    py: Python<'_>,
    candidate_meta: &Bound<'_, PyAny>,
) -> PyResult<Vec<PyObject>> {
    let candidate_dict = candidate_meta
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("authority component candidate_meta is not a dict"))?;
    let mut records: Vec<PyObject> = Vec::with_capacity(candidate_dict.len());
    for (key, value) in candidate_dict.iter() {
        let key_tuple = key
            .downcast::<PyTuple>()
            .map_err(|_| PyValueError::new_err("candidate_meta key is not a tuple"))?;
        if key_tuple.len() != 2 {
            return Err(PyValueError::new_err(
                "candidate_meta key must contain KYPERMNO and candidate_ric",
            ));
        }
        let kypermno = key_tuple.get_item(0)?.str()?.to_str()?.to_string();
        let candidate_ric = key_tuple.get_item(1)?.str()?.to_str()?.to_string();
        let meta = value
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("candidate_meta value is not a dict"))?;
        let row = PyDict::new_bound(py);
        row.set_item("KYPERMNO", kypermno)?;
        row.set_item("candidate_ric", candidate_ric)?;
        for field in [
            "candidate_has_conventional_support",
            "candidate_has_effective_support",
            "candidate_ownership_date_count",
            "candidate_bridge_span_day_count",
            "candidate_bridge_start_date",
            "candidate_bridge_end_date",
        ] {
            row.set_item(
                field,
                meta.get_item(field)?
                    .unwrap_or_else(|| py.None().bind(py).clone()),
            )?;
        }

        let observation_value_sets = meta
            .get_item("observation_value_sets")?
            .ok_or_else(|| PyValueError::new_err("missing observation_value_sets"))?;
        let observation_dict = observation_value_sets
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("observation_value_sets is not a dict"))?;
        let mut observation_rows: Vec<PyObject> = Vec::with_capacity(observation_dict.len());
        for (obs_key, values) in observation_dict.iter() {
            let obs_tuple = obs_key
                .downcast::<PyTuple>()
                .map_err(|_| PyValueError::new_err("observation key is not a tuple"))?;
            if obs_tuple.len() != 2 {
                return Err(PyValueError::new_err(
                    "observation key must contain date and category",
                ));
            }
            let values_list = PyList::empty_bound(py);
            for value in values.iter()? {
                values_list.append(value?)?;
            }
            observation_rows.push(
                PyTuple::new_bound(
                    py,
                    [
                        obs_tuple.get_item(0)?.into_py(py),
                        obs_tuple.get_item(1)?.into_py(py),
                        values_list.into_py(py),
                    ],
                )
                .into_py(py),
            );
        }
        row.set_item(
            "observation_value_sets",
            PyList::new_bound(py, observation_rows),
        )?;

        let unique_row_set = meta
            .get_item("unique_row_set")?
            .ok_or_else(|| PyValueError::new_err("missing unique_row_set"))?;
        let unique_rows = PyList::empty_bound(py);
        for unique_row in unique_row_set.iter()? {
            unique_rows.append(unique_row?)?;
        }
        row.set_item("unique_row_set", unique_rows)?;
        records.push(row.into_py(py));
    }
    Ok(records)
}

pub(crate) fn refinitiv_authority_component_pair_meta_records(
    py: Python<'_>,
    pair_meta: &Bound<'_, PyAny>,
) -> PyResult<Vec<PyObject>> {
    let pair_dict = pair_meta
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("authority component pair_meta is not a dict"))?;
    let mut records: Vec<PyObject> = Vec::with_capacity(pair_dict.len());
    for (key, value) in pair_dict.iter() {
        let key_tuple = key
            .downcast::<PyTuple>()
            .map_err(|_| PyValueError::new_err("pair_meta key is not a tuple"))?;
        if key_tuple.len() != 3 {
            return Err(PyValueError::new_err(
                "pair_meta key must contain KYPERMNO, left RIC, and right RIC",
            ));
        }
        let meta = value
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("pair_meta value is not a dict"))?;
        let row = PyDict::new_bound(py);
        row.set_item("KYPERMNO", key_tuple.get_item(0)?)?;
        row.set_item("left_candidate_ric", key_tuple.get_item(1)?)?;
        row.set_item("right_candidate_ric", key_tuple.get_item(2)?)?;
        for field in [
            "pair_benign_alias_supported",
            "pair_value_conflict_present",
            "pair_regime_split_supported",
            "pair_requires_review",
        ] {
            row.set_item(
                field,
                meta.get_item(field)?
                    .unwrap_or_else(|| py.None().bind(py).clone()),
            )?;
        }
        records.push(row.into_py(py));
    }
    Ok(records)
}

pub(crate) fn authority_component_find(
    parent: &mut HashMap<String, String>,
    item: &str,
) -> PyResult<String> {
    let parent_value = parent
        .get(item)
        .ok_or_else(|| PyValueError::new_err("component union-find item is missing"))?
        .clone();
    if parent_value == item {
        return Ok(parent_value);
    }
    let root = authority_component_find(parent, &parent_value)?;
    parent.insert(item.to_string(), root.clone());
    Ok(root)
}

pub(crate) fn authority_component_union(
    parent: &mut HashMap<String, String>,
    left: &str,
    right: &str,
) -> PyResult<()> {
    let left_root = authority_component_find(parent, left)?;
    let right_root = authority_component_find(parent, right)?;
    if left_root != right_root {
        parent.insert(right_root, left_root);
    }
    Ok(())
}

pub(crate) fn refinitiv_authority_component_id_impl(
    kypermno: &str,
    component_index: i64,
) -> String {
    format!("{kypermno}|COMPONENT|{component_index:02}")
}

pub(crate) fn refinitiv_authority_merge_interval_ordinals(
    mut intervals: Vec<(i64, i64)>,
    max_gap_days: i64,
) -> Vec<(i64, i64)> {
    if intervals.is_empty() {
        return Vec::new();
    }
    intervals.sort_unstable();
    let mut merged = vec![intervals[0]];
    for (start_date, end_date) in intervals.into_iter().skip(1) {
        let last_index = merged.len() - 1;
        let (prior_start, prior_end) = merged[last_index];
        if start_date - prior_end <= max_gap_days + 1 {
            merged[last_index] = (prior_start, prior_end.max(end_date));
        } else {
            merged.push((start_date, end_date));
        }
    }
    merged
}

#[pyfunction]
pub(crate) fn refinitiv_authority_conventional_components(
    py: Python<'_>,
    permno_order: Vec<String>,
    candidate_records: &Bound<'_, PyAny>,
    pair_records: &Bound<'_, PyAny>,
    max_gap_days: i64,
) -> PyResult<(Vec<PyObject>, Vec<PyObject>)> {
    let mut candidates_by_key: HashMap<(String, String), RefinitivAuthorityComponentCandidate> =
        HashMap::new();
    for record in candidate_records.iter()? {
        let record = record?;
        let dict = record.downcast::<PyDict>().map_err(|_| {
            PyValueError::new_err("authority component candidate record is not a dict")
        })?;
        let candidate = refinitiv_authority_component_candidate_record(py, dict)?;
        candidates_by_key.insert(
            (candidate.kypermno.clone(), candidate.candidate_ric.clone()),
            candidate,
        );
    }

    let mut pair_map: HashMap<(String, String, String), RefinitivAuthorityComponentPair> =
        HashMap::new();
    for record in pair_records.iter()? {
        let record = record?;
        let dict = record
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("authority component pair record is not a dict"))?;
        let (key, pair) = refinitiv_authority_component_pair_record(dict)?;
        pair_map.insert(key, pair);
    }

    let mut map_rows = Vec::new();
    let mut component_rows = Vec::new();
    for kypermno in permno_order {
        let mut conventional_rics: Vec<String> = candidates_by_key
            .iter()
            .filter_map(|((permno, ric), candidate)| {
                if permno == &kypermno && candidate.candidate_has_conventional_support {
                    Some(ric.clone())
                } else {
                    None
                }
            })
            .collect();
        conventional_rics.sort();

        let mut parent: HashMap<String, String> = conventional_rics
            .iter()
            .map(|ric| (ric.clone(), ric.clone()))
            .collect();
        for left_index in 0..conventional_rics.len() {
            for right_index in (left_index + 1)..conventional_rics.len() {
                let left_ric = &conventional_rics[left_index];
                let right_ric = &conventional_rics[right_index];
                if pair_map
                    .get(&(kypermno.clone(), left_ric.clone(), right_ric.clone()))
                    .is_some_and(|pair| pair.pair_benign_alias_supported)
                {
                    authority_component_union(&mut parent, left_ric, right_ric)?;
                }
            }
        }

        let mut grouped_members: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for candidate_ric in &conventional_rics {
            let root = authority_component_find(&mut parent, candidate_ric)?;
            grouped_members
                .entry(root)
                .or_default()
                .push(candidate_ric.clone());
        }
        let mut ordered_groups: Vec<Vec<String>> = grouped_members.into_values().collect();
        for members in &mut ordered_groups {
            members.sort();
        }
        ordered_groups.sort_by(|left, right| {
            let left_min_bridge = left
                .iter()
                .filter_map(|member| {
                    candidates_by_key
                        .get(&(kypermno.clone(), member.clone()))
                        .and_then(|candidate| {
                            candidate
                                .candidate_bridge_start_date
                                .as_ref()
                                .map(|(ordinal, _)| *ordinal)
                        })
                })
                .min()
                .unwrap_or(i64::MAX);
            let right_min_bridge = right
                .iter()
                .filter_map(|member| {
                    candidates_by_key
                        .get(&(kypermno.clone(), member.clone()))
                        .and_then(|candidate| {
                            candidate
                                .candidate_bridge_start_date
                                .as_ref()
                                .map(|(ordinal, _)| *ordinal)
                        })
                })
                .min()
                .unwrap_or(i64::MAX);
            left_min_bridge.cmp(&right_min_bridge).then(
                left.first()
                    .map(String::as_str)
                    .unwrap_or("")
                    .cmp(right.first().map(String::as_str).unwrap_or("")),
            )
        });

        for (component_index, members) in ordered_groups.iter().enumerate() {
            let component_id =
                refinitiv_authority_component_id_impl(&kypermno, (component_index + 1) as i64);
            for member in members {
                let row = PyDict::new_bound(py);
                row.set_item("KYPERMNO", kypermno.as_str())?;
                row.set_item("candidate_ric", member.as_str())?;
                row.set_item("component_id", component_id.as_str())?;
                map_rows.push(row.into_py(py));
            }

            let mut member_histories: Vec<&RefinitivAuthorityComponentCandidate> = members
                .iter()
                .filter_map(|member| candidates_by_key.get(&(kypermno.clone(), member.clone())))
                .collect();
            let mut union_value_sets: BTreeMap<(i64, String), (PyObject, Vec<Option<f64>>)> =
                BTreeMap::new();
            let mut unique_row_keys: HashSet<String> = HashSet::new();
            let mut bridge_intervals: Vec<(i64, i64)> = Vec::new();
            let mut date_objects: HashMap<i64, PyObject> = HashMap::new();

            for history in &member_histories {
                for ((date_ordinal, category), (date_obj, values)) in
                    &history.observation_value_sets
                {
                    date_objects
                        .entry(*date_ordinal)
                        .or_insert_with(|| date_obj.clone_ref(py));
                    let entry = union_value_sets
                        .entry((*date_ordinal, category.clone()))
                        .or_insert_with(|| (date_obj.clone_ref(py), Vec::new()));
                    for value in values {
                        refinitiv_authority_push_unique_optional_value(&mut entry.1, *value);
                    }
                }
                for (date_ordinal, date_obj, category, value) in &history.unique_row_set {
                    date_objects
                        .entry(*date_ordinal)
                        .or_insert_with(|| date_obj.clone_ref(py));
                    unique_row_keys.insert(format!(
                        "{}\u{1f}{}\u{1f}{}",
                        date_ordinal,
                        category,
                        refinitiv_authority_value_sort_key(*value)
                    ));
                }
                if let (Some((start, start_obj)), Some((end, end_obj))) = (
                    history.candidate_bridge_start_date.as_ref(),
                    history.candidate_bridge_end_date.as_ref(),
                ) {
                    bridge_intervals.push((*start, *end));
                    date_objects
                        .entry(*start)
                        .or_insert_with(|| start_obj.clone_ref(py));
                    date_objects
                        .entry(*end)
                        .or_insert_with(|| end_obj.clone_ref(py));
                }
            }

            let union_dates: BTreeSet<i64> =
                union_value_sets.keys().map(|(date, _)| *date).collect();
            member_histories.sort_by(|left, right| {
                (!left.candidate_has_effective_support)
                    .cmp(&(!right.candidate_has_effective_support))
                    .then(
                        right
                            .candidate_ownership_date_count
                            .cmp(&left.candidate_ownership_date_count),
                    )
                    .then(
                        right
                            .candidate_bridge_span_day_count
                            .cmp(&left.candidate_bridge_span_day_count),
                    )
                    .then(left.candidate_ric.cmp(&right.candidate_ric))
            });
            let canonical_member = member_histories.first().copied();

            let mut component_requires_review = union_value_sets
                .values()
                .any(|(_date_obj, values)| refinitiv_authority_distinct_some_count(values) > 1);
            if !component_requires_review {
                for left_index in 0..members.len() {
                    for right_index in (left_index + 1)..members.len() {
                        let left_member = &members[left_index];
                        let right_member = &members[right_index];
                        if let Some(pair) = pair_map.get(&(
                            kypermno.clone(),
                            left_member.clone(),
                            right_member.clone(),
                        )) {
                            if pair.pair_value_conflict_present
                                || pair.pair_regime_split_supported
                                || pair.pair_requires_review
                            {
                                component_requires_review = true;
                                break;
                            }
                        }
                    }
                    if component_requires_review {
                        break;
                    }
                }
            }

            let bridge_start = bridge_intervals.iter().map(|(start, _)| *start).min();
            let bridge_end = bridge_intervals.iter().map(|(_, end)| *end).max();
            let merged_bridge_windows =
                refinitiv_authority_merge_interval_ordinals(bridge_intervals, max_gap_days);

            let row = PyDict::new_bound(py);
            row.set_item("KYPERMNO", kypermno.as_str())?;
            row.set_item("component_id", component_id.as_str())?;
            row.set_item("member_rics", PyList::new_bound(py, members.clone()))?;
            row.set_item(
                "canonical_ric",
                canonical_member.map(|candidate| candidate.candidate_ric.clone()),
            )?;
            row.set_item(
                "canonical_best_candidate_date_count",
                canonical_member
                    .map(|candidate| candidate.candidate_ownership_date_count)
                    .unwrap_or(0),
            )?;
            row.set_item("ownership_date_count", union_dates.len() as i64)?;
            row.set_item("ownership_row_count", unique_row_keys.len() as i64)?;
            match bridge_start.and_then(|ordinal| date_objects.get(&ordinal)) {
                Some(value) => row.set_item("bridge_start_date", value.clone_ref(py))?,
                None => row.set_item("bridge_start_date", Option::<String>::None)?,
            }
            match bridge_end.and_then(|ordinal| date_objects.get(&ordinal)) {
                Some(value) => row.set_item("bridge_end_date", value.clone_ref(py))?,
                None => row.set_item("bridge_end_date", Option::<String>::None)?,
            }
            match union_dates
                .first()
                .and_then(|ordinal| date_objects.get(ordinal))
            {
                Some(value) => row.set_item("first_ownership_date", value.clone_ref(py))?,
                None => row.set_item("first_ownership_date", Option::<String>::None)?,
            }
            match union_dates
                .last()
                .and_then(|ordinal| date_objects.get(ordinal))
            {
                Some(value) => row.set_item("last_ownership_date", value.clone_ref(py))?,
                None => row.set_item("last_ownership_date", Option::<String>::None)?,
            }

            let mut merged_windows_out: Vec<PyObject> = Vec::new();
            for (start, end) in merged_bridge_windows {
                let Some(start_obj) = date_objects.get(&start) else {
                    continue;
                };
                let Some(end_obj) = date_objects.get(&end) else {
                    continue;
                };
                merged_windows_out.push(
                    PyTuple::new_bound(py, [start_obj.clone_ref(py), end_obj.clone_ref(py)])
                        .into_py(py),
                );
            }
            row.set_item(
                "merged_bridge_windows",
                PyList::new_bound(py, merged_windows_out),
            )?;

            let mut union_rows: Vec<PyObject> = Vec::new();
            for ((_date_ordinal, category), (date_obj, values)) in &union_value_sets {
                let value_objects: Vec<PyObject> = values
                    .iter()
                    .map(|value| match value {
                        Some(value) => value.into_py(py),
                        None => py.None(),
                    })
                    .collect();
                union_rows.push(
                    PyTuple::new_bound(
                        py,
                        [
                            date_obj.clone_ref(py),
                            category.clone().into_py(py),
                            PyList::new_bound(py, value_objects).into_py(py),
                        ],
                    )
                    .into_py(py),
                );
            }
            row.set_item("union_value_sets", PyList::new_bound(py, union_rows))?;
            row.set_item("component_requires_review", component_requires_review)?;
            let coverage_share_of_best_candidate =
                if let Some(candidate) = canonical_member.filter(|_| !union_dates.is_empty()) {
                    Some(candidate.candidate_ownership_date_count as f64 / union_dates.len() as f64)
                } else {
                    None
                };
            row.set_item(
                "coverage_share_of_best_candidate",
                coverage_share_of_best_candidate,
            )?;
            component_rows.push(row.into_py(py));
        }
    }

    Ok((map_rows, component_rows))
}

#[pyfunction]
pub(crate) fn refinitiv_authority_conventional_components_from_meta(
    py: Python<'_>,
    permno_order: Vec<String>,
    candidate_meta: &Bound<'_, PyAny>,
    pair_meta: &Bound<'_, PyAny>,
    max_gap_days: i64,
) -> PyResult<(Vec<PyObject>, Vec<PyObject>)> {
    let candidate_records = PyList::new_bound(
        py,
        refinitiv_authority_component_candidate_meta_records(py, candidate_meta)?,
    );
    let pair_records = PyList::new_bound(
        py,
        refinitiv_authority_component_pair_meta_records(py, pair_meta)?,
    );
    refinitiv_authority_conventional_components(
        py,
        permno_order,
        candidate_records.as_any(),
        pair_records.as_any(),
        max_gap_days,
    )
}

#[pyfunction]
#[pyo3(signature = (value=None))]
pub(crate) fn normalize_extended_workbook_bool_value(
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<bool>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    if value.is_instance_of::<PyBool>() {
        return Ok(Some(value.extract::<bool>()?));
    }
    let rendered = value.str()?;
    let normalized = rendered.to_str()?.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "true" => Ok(Some(true)),
        "false" => Ok(Some(false)),
        _ => Ok(None),
    }
}

pub(crate) fn push_normalized_item_code(
    out: &mut Vec<String>,
    seen: &mut HashSet<String>,
    entry: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let rendered = entry.str()?;
    let raw = rendered.to_str()?.trim();
    if raw.is_empty() {
        return Ok(());
    }
    let normalized = raw.to_lowercase();
    if seen.insert(normalized.clone()) {
        out.push(normalized);
    }
    Ok(())
}

pub(crate) fn normalize_item_codes_optional_impl(
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<Vec<String>>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    let mut out: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    if let Ok(list) = value.downcast::<PyList>() {
        for entry in list.iter() {
            push_normalized_item_code(&mut out, &mut seen, &entry)?;
        }
    } else if let Ok(tuple) = value.downcast::<PyTuple>() {
        for entry in tuple.iter() {
            push_normalized_item_code(&mut out, &mut seen, &entry)?;
        }
    } else {
        return Err(PyValueError::new_err(
            "unsupported item-code container falls back to Python",
        ));
    }
    if out.is_empty() {
        Ok(None)
    } else {
        Ok(Some(out))
    }
}

pub(crate) fn normalize_year_entry(entry: &Bound<'_, PyAny>) -> PyResult<i64> {
    if entry.is_instance_of::<PyBool>() {
        return Ok(if entry.extract::<bool>()? { 1 } else { 0 });
    }
    if entry.is_instance_of::<PyInt>() {
        return entry.extract::<i64>();
    }
    if entry.is_instance_of::<PyFloat>() {
        let number = entry.extract::<f64>()?;
        if number.is_finite() {
            return Ok(number.trunc() as i64);
        }
        return Err(PyValueError::new_err(
            "non-finite year falls back to Python",
        ));
    }
    let rendered = entry.str()?;
    let text = rendered.to_str()?.trim();
    text.parse::<i64>()
        .map_err(|_| PyValueError::new_err("unsupported year value falls back to Python"))
}

pub(crate) fn normalize_years_optional_impl(
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<Vec<i64>>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    let mut years: BTreeSet<i64> = BTreeSet::new();
    if let Ok(list) = value.downcast::<PyList>() {
        for entry in list.iter() {
            years.insert(normalize_year_entry(&entry)?);
        }
    } else if let Ok(tuple) = value.downcast::<PyTuple>() {
        for entry in tuple.iter() {
            years.insert(normalize_year_entry(&entry)?);
        }
    } else {
        return Err(PyValueError::new_err(
            "unsupported year container falls back to Python",
        ));
    }
    if years.is_empty() {
        Ok(None)
    } else {
        Ok(Some(years.into_iter().collect()))
    }
}

#[pyfunction]
#[pyo3(signature = (value=None))]
pub(crate) fn normalize_item_codes_optional_value(
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<Vec<String>>> {
    normalize_item_codes_optional_impl(value)
}

#[pyfunction]
#[pyo3(signature = (value=None))]
pub(crate) fn normalize_years_optional_value(
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<Vec<i64>>> {
    normalize_years_optional_impl(value)
}

pub(crate) fn stage_audit_optional_int_impl(
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<i64>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    if value.is_instance_of::<PyBool>() {
        return Ok(Some(if value.extract::<bool>()? { 1 } else { 0 }));
    }
    if value.is_instance_of::<PyInt>() {
        return Ok(value.extract::<i64>().ok());
    }
    if value.is_instance_of::<PyFloat>() {
        let number = value.extract::<f64>()?;
        if number.is_finite() {
            return Ok(Some(number.trunc() as i64));
        }
        return Ok(None);
    }
    let rendered = value.str()?;
    let text = rendered.to_str()?.trim();
    if text.is_empty() {
        return Ok(None);
    }
    Ok(text.parse::<i64>().ok())
}

#[pyfunction]
#[pyo3(signature = (value=None))]
pub(crate) fn normalize_stage_audit_optional_text(
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<String>> {
    normalize_lookup_text_any_impl(value)
}

#[pyfunction]
#[pyo3(signature = (value=None))]
pub(crate) fn normalize_stage_audit_optional_int(
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<i64>> {
    stage_audit_optional_int_impl(value)
}

#[pyfunction]
#[pyo3(signature = (value=None))]
pub(crate) fn normalize_stage_audit_optional_mapping(
    py: Python<'_>,
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<PyObject>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_none() || !value.is_instance_of::<PyDict>() {
        return Ok(None);
    }
    let dict = value
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("stage audit optional mapping is not a dict"))?;
    let out = PyDict::new_bound(py);
    for (key, dict_value) in dict.iter() {
        out.set_item(key, dict_value)?;
    }
    Ok(Some(out.into_py(py)))
}

#[pyfunction]
#[pyo3(signature = (value=None))]
pub(crate) fn normalize_optional_int_value(
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<i64>> {
    stage_audit_optional_int_impl(value)
}

#[pyfunction]
#[pyo3(signature = (value=None))]
pub(crate) fn normalize_stage_audit_string_list(
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<Vec<String>> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    if value.is_none() {
        return Ok(Vec::new());
    }
    let Ok(list) = value.downcast::<PyList>() else {
        return Ok(Vec::new());
    };
    let mut out = Vec::with_capacity(list.len());
    for entry in list.iter() {
        if let Some(text) = normalize_lookup_text_any_impl(Some(&entry))? {
            out.push(text);
        }
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (value=None))]
pub(crate) fn normalize_refinitiv_lookup_result_value(
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<String>> {
    let Some(normalized) = normalize_lookup_text_any_impl(value)? else {
        return Ok(None);
    };
    let upper = normalized.to_ascii_uppercase();
    if upper == "NULL" || upper == "NO UNIVERSE DEFINED." {
        return Ok(None);
    }
    if normalized
        .to_ascii_lowercase()
        .contains("invalid identifier")
    {
        return Ok(None);
    }
    Ok(Some(normalized))
}

#[pyfunction]
#[pyo3(signature = (value=None))]
pub(crate) fn normalize_refinitiv_ownership_result_text(
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<String>> {
    let Some(normalized) = normalize_lookup_text_any_impl(value)? else {
        return Ok(None);
    };
    let upper = normalized.to_ascii_uppercase();
    if upper == "NULL" || upper == "NO UNIVERSE DEFINED." {
        return Ok(None);
    }
    Ok(Some(normalized))
}

#[pyfunction]
#[pyo3(signature = (value=None))]
pub(crate) fn normalize_refinitiv_ownership_result_value(
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<f64>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    if value.is_instance_of::<PyBool>() {
        return Ok(Some(if value.extract::<bool>()? { 1.0 } else { 0.0 }));
    }
    if value.is_instance_of::<PyInt>() || value.is_instance_of::<PyFloat>() {
        return Ok(Some(value.extract::<f64>()?));
    }
    let Some(normalized) = py_str_normalized(value)? else {
        return Ok(None);
    };
    Ok(normalized.parse::<f64>().ok())
}

#[pyfunction]
#[pyo3(signature = (value=None))]
pub(crate) fn normalize_refinitiv_ownership_result_date(
    py: Python<'_>,
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<PyObject>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    let datetime_mod = py.import_bound("datetime")?;
    let datetime_type = datetime_mod.getattr("datetime")?;
    let date_type = datetime_mod.getattr("date")?;
    if value.is_instance(&datetime_type)? || value.is_instance(&date_type)? {
        let (year, month, day) = extract_py_date_parts(value)?;
        return Ok(Some(py_date_object(py, year, month, day)?));
    }
    let Some(normalized) = py_str_normalized(value)? else {
        return Ok(None);
    };
    let Some((year, month, day)) = parse_strict_iso_date_parts(&normalized) else {
        return Ok(None);
    };
    Ok(Some(py_date_object(py, year, month, day)?))
}

#[pyfunction]
#[pyo3(signature = (value=None))]
pub(crate) fn refinitiv_bridge_ownership_universe_request_date(
    py: Python<'_>,
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<String>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    let datetime_mod = py.import_bound("datetime")?;
    let datetime_type = datetime_mod.getattr("datetime")?;
    let date_type = datetime_mod.getattr("date")?;
    if value.is_instance(&datetime_type)? || value.is_instance(&date_type)? {
        let (year, month, day) = extract_py_date_parts(value)?;
        return Ok(Some(iso_date_string(year, month, day)));
    }
    normalize_lookup_text_any_impl(Some(value))
}

#[pyfunction]
#[pyo3(signature = (value=None))]
pub(crate) fn refinitiv_bridge_date_to_text_value(
    py: Python<'_>,
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<String>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    let datetime_mod = py.import_bound("datetime")?;
    let date_type = datetime_mod.getattr("date")?;
    if value.is_instance(&date_type)? {
        return value
            .call_method0("isoformat")?
            .extract::<String>()
            .map(Some);
    }
    normalize_lookup_text_any_impl(Some(value))
}

#[pyfunction]
#[pyo3(signature = (value=None))]
pub(crate) fn normalize_spaces_value(value: Option<&Bound<'_, PyAny>>) -> PyResult<String> {
    let Some(value) = value else {
        return Ok(String::new());
    };
    if value.is_none() {
        return Ok(String::new());
    }
    let rendered = value.str()?;
    Ok(normalize_spaces_impl(rendered.to_str()?))
}

#[pyfunction]
pub(crate) fn extension_csv_json_dumps_values(
    py: Python<'_>,
    values: &Bound<'_, PyAny>,
) -> PyResult<Vec<Option<String>>> {
    let json_dumps = py.import_bound("json")?.getattr("dumps")?;
    let mut out = Vec::new();
    for value in values.iter()? {
        let value = value?;
        if value.is_none() {
            out.push(None);
            continue;
        }
        out.push(Some(json_dumps.call1((value,))?.extract::<String>()?));
    }
    Ok(out)
}

#[pyfunction]
pub(crate) fn multisurface_boundary_snippet_risk(snippets: Vec<Option<String>>) -> bool {
    multisurface_boundary_snippet_risk_impl(snippets)
}

#[pyfunction]
pub(crate) fn multisurface_boundary_snippet_risk_values(
    original_start: &Bound<'_, PyAny>,
    cleaned_start: &Bound<'_, PyAny>,
    original_end: &Bound<'_, PyAny>,
    cleaned_end: &Bound<'_, PyAny>,
) -> PyResult<Vec<bool>> {
    let original_start = optional_string_values(original_start)?;
    let cleaned_start = optional_string_values(cleaned_start)?;
    let original_end = optional_string_values(original_end)?;
    let cleaned_end = optional_string_values(cleaned_end)?;
    let len = original_start.len();
    validate_equal_column_len("cleaned_start", len, cleaned_start.len())?;
    validate_equal_column_len("original_end", len, original_end.len())?;
    validate_equal_column_len("cleaned_end", len, cleaned_end.len())?;

    let mut out = Vec::with_capacity(len);
    for idx in 0..len {
        out.push(multisurface_boundary_snippet_risk_impl(vec![
            original_start[idx].clone(),
            cleaned_start[idx].clone(),
            original_end[idx].clone(),
            cleaned_end[idx].clone(),
        ]));
    }
    Ok(out)
}

#[pyfunction]
pub(crate) fn multisurface_snippet_delta_risk_values(
    original_start: &Bound<'_, PyAny>,
    cleaned_start: &Bound<'_, PyAny>,
    original_end: &Bound<'_, PyAny>,
    cleaned_end: &Bound<'_, PyAny>,
) -> PyResult<Vec<bool>> {
    let original_start = optional_string_values(original_start)?;
    let cleaned_start = optional_string_values(cleaned_start)?;
    let original_end = optional_string_values(original_end)?;
    let cleaned_end = optional_string_values(cleaned_end)?;
    let len = original_start.len();
    validate_equal_column_len("cleaned_start", len, cleaned_start.len())?;
    validate_equal_column_len("original_end", len, original_end.len())?;
    validate_equal_column_len("cleaned_end", len, cleaned_end.len())?;

    let mut out = Vec::with_capacity(len);
    for idx in 0..len {
        out.push(
            normalized_optional_spaces(&original_start[idx])
                != normalized_optional_spaces(&cleaned_start[idx])
                || normalized_optional_spaces(&original_end[idx])
                    != normalized_optional_spaces(&cleaned_end[idx]),
        );
    }
    Ok(out)
}

#[pyfunction]
pub(crate) fn multisurface_chunk_record_indices(
    rows: &Bound<'_, PyAny>,
    chunk_count: usize,
) -> PyResult<Vec<Vec<usize>>> {
    let mut buckets: BTreeMap<(String, String, bool), Vec<usize>> = BTreeMap::new();
    let mut row_count = 0usize;
    for (row_index, row) in rows.iter()?.enumerate() {
        row_count += 1;
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("multi-surface audit row is not a dict"))?;

        let case_source = match dict.get_item("case_source")? {
            Some(value) => value.str()?.to_str()?.to_string(),
            None => return Err(PyValueError::new_err("missing required key: case_source")),
        };
        let text_scope = match dict.get_item("text_scope")? {
            Some(value) if value.is_truthy()? => value.str()?.to_str()?.to_string(),
            Some(_) => String::new(),
            None => return Err(PyValueError::new_err("missing required key: text_scope")),
        };
        let full_report_needed = match dict.get_item("full_report_needed")? {
            Some(value) => value.is_truthy()?,
            None => {
                return Err(PyValueError::new_err(
                    "missing required key: full_report_needed",
                ));
            }
        };
        buckets
            .entry((case_source, text_scope, full_report_needed))
            .or_default()
            .push(row_index);
    }

    if chunk_count == 0 {
        if row_count == 0 {
            return Ok(Vec::new());
        }
        return Err(PyValueError::new_err("chunk_count must be positive"));
    }

    let ordered_buckets: Vec<Vec<usize>> = buckets.into_values().collect();
    let mut bucket_offsets = vec![0usize; ordered_buckets.len()];
    let mut ordered_indices: Vec<usize> = Vec::with_capacity(row_count);
    loop {
        let mut progressed = false;
        for (bucket_index, bucket) in ordered_buckets.iter().enumerate() {
            let offset = bucket_offsets[bucket_index];
            if offset < bucket.len() {
                ordered_indices.push(bucket[offset]);
                bucket_offsets[bucket_index] = offset + 1;
                progressed = true;
            }
        }
        if !progressed {
            break;
        }
    }

    let mut chunks: Vec<Vec<usize>> = vec![Vec::new(); chunk_count];
    for (ordered_index, row_index) in ordered_indices.into_iter().enumerate() {
        chunks[ordered_index % chunk_count].push(row_index);
    }
    Ok(chunks)
}

#[pyfunction]
pub(crate) fn multisurface_chunk_record_indices_columns(
    py: Python<'_>,
    column_names: Vec<String>,
    column_values: &Bound<'_, PyAny>,
    chunk_count: usize,
) -> PyResult<Vec<Vec<usize>>> {
    let label = "multi-surface audit chunk records";
    let (columns, row_count) =
        collect_pyobject_column_values(py, &column_names, column_values, label)?;
    let column_index = column_index_by_name(&column_names);
    let case_source_idx = required_named_column_index(&column_index, label, "case_source")?;
    let text_scope_idx = required_named_column_index(&column_index, label, "text_scope")?;
    let full_report_idx = required_named_column_index(&column_index, label, "full_report_needed")?;
    let mut buckets: BTreeMap<(String, String, bool), Vec<usize>> = BTreeMap::new();

    for row_index in 0..row_count {
        let case_source = columns[case_source_idx][row_index]
            .bind(py)
            .str()?
            .to_str()?
            .to_string();
        let text_scope_value = columns[text_scope_idx][row_index].bind(py);
        let text_scope = if text_scope_value.is_truthy()? {
            text_scope_value.str()?.to_str()?.to_string()
        } else {
            String::new()
        };
        let full_report_needed = columns[full_report_idx][row_index].bind(py).is_truthy()?;
        buckets
            .entry((case_source, text_scope, full_report_needed))
            .or_default()
            .push(row_index);
    }

    if chunk_count == 0 {
        if row_count == 0 {
            return Ok(Vec::new());
        }
        return Err(PyValueError::new_err("chunk_count must be positive"));
    }

    let ordered_buckets: Vec<Vec<usize>> = buckets.into_values().collect();
    let mut bucket_offsets = vec![0usize; ordered_buckets.len()];
    let mut ordered_indices: Vec<usize> = Vec::with_capacity(row_count);
    loop {
        let mut progressed = false;
        for (bucket_index, bucket) in ordered_buckets.iter().enumerate() {
            let offset = bucket_offsets[bucket_index];
            if offset < bucket.len() {
                ordered_indices.push(bucket[offset]);
                bucket_offsets[bucket_index] = offset + 1;
                progressed = true;
            }
        }
        if !progressed {
            break;
        }
    }

    let mut chunks: Vec<Vec<usize>> = vec![Vec::new(); chunk_count];
    for (ordered_index, row_index) in ordered_indices.into_iter().enumerate() {
        chunks[ordered_index % chunk_count].push(row_index);
    }
    Ok(chunks)
}

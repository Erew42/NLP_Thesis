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
use crate::refinitiv_bridge::*;
use crate::sec_extraction::*;
use crate::sentence_cleaning::*;
use crate::sentence_quality_api::*;

pub(crate) fn excel_col_label(mut col_idx: usize) -> String {
    let mut chars: Vec<char> = Vec::new();
    loop {
        let rem = col_idx % 26;
        chars.push((b'A' + rem as u8) as char);
        if col_idx < 26 {
            break;
        }
        col_idx = (col_idx / 26) - 1;
    }
    chars.iter().rev().collect()
}

pub(crate) fn excel_cell_ref(row_idx: usize, col_idx: usize) -> String {
    format!("{}{}", excel_col_label(col_idx), row_idx + 1)
}

pub(crate) fn excel_abs_cell_ref(row_idx: usize, col_idx: usize) -> String {
    format!("${}${}", excel_col_label(col_idx), row_idx + 1)
}

pub(crate) fn excel_sheet_range_ref(
    sheet_name: &str,
    first_row: usize,
    last_row: usize,
    col_idx: usize,
) -> String {
    format!(
        "'{}'!{}:{}",
        sheet_name,
        excel_abs_cell_ref(first_row, col_idx),
        excel_abs_cell_ref(last_row, col_idx)
    )
}

pub(crate) fn dict_optional_rendered_value(
    dict: &Bound<'_, PyDict>,
    key: &str,
) -> PyResult<String> {
    match dict.get_item(key)? {
        Some(value) if !value.is_none() => Ok(value.str()?.to_str()?.to_string()),
        _ => Ok(String::new()),
    }
}

#[pyfunction]
pub(crate) fn refinitiv_excel_ownership_universe_block_payloads(
    py: Python<'_>,
    rows: &Bound<'_, PyAny>,
    input_field_order: Vec<String>,
    block_width: usize,
) -> PyResult<Vec<PyObject>> {
    let candidate_ric_offset = input_field_order
        .iter()
        .position(|field| field == "candidate_ric")
        .ok_or_else(|| PyValueError::new_err("input_field_order missing candidate_ric"))?
        + 1;
    let request_start_offset = input_field_order
        .iter()
        .position(|field| field == "request_start_date")
        .ok_or_else(|| PyValueError::new_err("input_field_order missing request_start_date"))?
        + 1;
    let request_end_offset = input_field_order
        .iter()
        .position(|field| field == "request_end_date")
        .ok_or_else(|| PyValueError::new_err("input_field_order missing request_end_date"))?
        + 1;

    let mut payloads: Vec<PyObject> = Vec::new();
    for (block_index, row) in rows.iter()?.enumerate() {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("ownership universe block row is not a dict"))?;
        let base_col = block_index * block_width;
        let mut input_values: Vec<String> = Vec::with_capacity(input_field_order.len());
        for field_name in &input_field_order {
            let value = dict.get_item(field_name)?;
            let rendered = match value {
                Some(value) if !value.is_none() => value.str()?.to_str()?.to_string(),
                _ => String::new(),
            };
            input_values.push(rendered);
        }

        let candidate_ric_ref = excel_cell_ref(candidate_ric_offset, base_col);
        let request_start_ref = excel_cell_ref(request_start_offset, base_col);
        let request_end_ref = excel_cell_ref(request_end_offset, base_col);
        let formula = format!(
            "=@RDP.Data({},\"TR.CategoryOwnershipPct.Date;TR.CategoryOwnershipPct;TR.InstrStatTypeValue\",\"StatType=7 SDate=\"&{}&\" EDate=\"&{}&\" CH=Fd RH=IN\")",
            candidate_ric_ref, request_start_ref, request_end_ref
        );

        let out = PyDict::new_bound(py);
        out.set_item("base_col", base_col as i64)?;
        out.set_item("input_values", PyList::new_bound(py, input_values))?;
        out.set_item("formula", formula)?;
        payloads.push(out.into_py(py));
    }
    Ok(payloads)
}

#[pyfunction]
pub(crate) fn refinitiv_excel_lm2011_doc_ownership_block_payloads(
    py: Python<'_>,
    rows: &Bound<'_, PyAny>,
    input_field_order: Vec<String>,
    block_width: usize,
    request_stage: &str,
) -> PyResult<Vec<PyObject>> {
    let authoritative_ric_offset = input_field_order
        .iter()
        .position(|field| field == "authoritative_ric")
        .ok_or_else(|| PyValueError::new_err("input_field_order missing authoritative_ric"))?
        + 1;
    let target_effective_date_offset = input_field_order
        .iter()
        .position(|field| field == "target_effective_date")
        .ok_or_else(|| PyValueError::new_err("input_field_order missing target_effective_date"))?
        + 1;
    let fallback_window_start_offset = input_field_order
        .iter()
        .position(|field| field == "fallback_window_start")
        .ok_or_else(|| PyValueError::new_err("input_field_order missing fallback_window_start"))?
        + 1;
    let fallback_window_end_offset = input_field_order
        .iter()
        .position(|field| field == "fallback_window_end")
        .ok_or_else(|| PyValueError::new_err("input_field_order missing fallback_window_end"))?
        + 1;

    let mut payloads: Vec<PyObject> = Vec::new();
    for (block_index, row) in rows.iter()?.enumerate() {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("doc ownership block row is not a dict"))?;
        let base_col = block_index * block_width;
        let input_values = PyList::empty_bound(py);
        for field_name in &input_field_order {
            match dict.get_item(field_name)? {
                Some(value) => input_values.append(value)?,
                None => input_values.append(py.None())?,
            }
        }

        let authoritative_ric_ref = excel_cell_ref(authoritative_ric_offset, base_col);
        let target_effective_date_ref = excel_cell_ref(target_effective_date_offset, base_col);
        let fallback_window_start_ref = excel_cell_ref(fallback_window_start_offset, base_col);
        let fallback_window_end_ref = excel_cell_ref(fallback_window_end_offset, base_col);
        let data_formula = match request_stage {
            "EXACT" => format!(
                "=@RDP.Data({},\"TR.CategoryOwnershipPct.Date;TR.CategoryOwnershipPct;TR.InstrStatTypeValue\",\"StatType=7 SDate=\"&TEXT({},\"yyyy-mm-dd\")&\" EDate=\"&TEXT({},\"yyyy-mm-dd\")&\" CH=Fd RH=IN\")",
                authoritative_ric_ref, target_effective_date_ref, target_effective_date_ref
            ),
            "FALLBACK" => format!(
                "=@RDP.Data({},\"TR.CategoryOwnershipPct.Date;TR.CategoryOwnershipPct;TR.InstrStatTypeValue\",\"StatType=7 SDate=\"&TEXT({},\"yyyy-mm-dd\")&\" EDate=\"&TEXT({},\"yyyy-mm-dd\")&\" CH=Fd RH=IN\")",
                authoritative_ric_ref, fallback_window_start_ref, fallback_window_end_ref
            ),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unsupported doc ownership request stage: {request_stage}"
                )));
            }
        };

        let out = PyDict::new_bound(py);
        out.set_item("base_col", base_col as i64)?;
        out.set_item("input_values", input_values)?;
        out.set_item(
            "authoritative_ric_formula",
            format!("={authoritative_ric_ref}"),
        )?;
        out.set_item("data_formula", data_formula)?;
        payloads.push(out.into_py(py));
    }
    Ok(payloads)
}

#[pyfunction]
pub(crate) fn refinitiv_excel_resolution_diagnostic_retrieval_block_payloads(
    py: Python<'_>,
    rows: &Bound<'_, PyAny>,
    input_field_order: Vec<String>,
    block_width: usize,
) -> PyResult<Vec<PyObject>> {
    let lookup_input_row = input_field_order
        .iter()
        .position(|field| field == "lookup_input")
        .ok_or_else(|| PyValueError::new_err("input_field_order missing lookup_input"))?
        + 1;
    let request_start_row = input_field_order
        .iter()
        .position(|field| field == "request_start_date")
        .ok_or_else(|| PyValueError::new_err("input_field_order missing request_start_date"))?
        + 1;
    let request_end_row = input_field_order
        .iter()
        .position(|field| field == "request_end_date")
        .ok_or_else(|| PyValueError::new_err("input_field_order missing request_end_date"))?
        + 1;

    let mut payloads: Vec<PyObject> = Vec::new();
    for (block_index, row) in rows.iter()?.enumerate() {
        let row = row?;
        let dict = row.downcast::<PyDict>().map_err(|_| {
            PyValueError::new_err("resolution diagnostic retrieval row is not a dict")
        })?;
        let base_col = block_index * block_width;
        let mut input_values: Vec<String> = Vec::with_capacity(input_field_order.len());
        for field_name in &input_field_order {
            let value = dict.get_item(field_name)?;
            let rendered = match value {
                Some(value) if !value.is_none() => value.str()?.to_str()?.to_string(),
                _ => String::new(),
            };
            input_values.push(rendered);
        }

        let lookup_input_ref = excel_cell_ref(lookup_input_row, base_col);
        let request_start_ref = excel_cell_ref(request_start_row, base_col);
        let request_end_ref = excel_cell_ref(request_end_row, base_col);
        let formula = format!(
            "=IF({}=\"\",\"\",_xll.RDP.Data({},\"TR.CategoryOwnershipPct.Date;TR.CategoryOwnershipPct;TR.InstrStatTypeValue\",\"StatType=7 SDate=\"&{}&\" EDate=\"&{}&\" CH=Fd RH=IN\"))",
            lookup_input_ref, lookup_input_ref, request_start_ref, request_end_ref
        );

        let out = PyDict::new_bound(py);
        out.set_item("base_col", base_col as i64)?;
        out.set_item("input_values", PyList::new_bound(py, input_values))?;
        out.set_item("formula", formula)?;
        payloads.push(out.into_py(py));
    }
    Ok(payloads)
}

#[pyfunction]
pub(crate) fn refinitiv_excel_ownership_smoke_block_payloads(
    py: Python<'_>,
    rows: &Bound<'_, PyAny>,
    input_field_order: Vec<String>,
    block_width: usize,
) -> PyResult<Vec<PyObject>> {
    let mut payloads: Vec<PyObject> = Vec::new();
    for (block_index, row) in rows.iter()?.enumerate() {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("ownership smoke block row is not a dict"))?;
        let base_col = block_index * block_width;
        let mut input_values: Vec<String> = Vec::with_capacity(input_field_order.len());
        for field_name in &input_field_order {
            let value = dict.get_item(field_name)?;
            let rendered = match value {
                Some(value) if !value.is_none() => value.str()?.to_str()?.to_string(),
                _ => String::new(),
            };
            input_values.push(rendered);
        }
        let out = PyDict::new_bound(py);
        out.set_item("base_col", base_col as i64)?;
        out.set_item("input_values", PyList::new_bound(py, input_values))?;
        payloads.push(out.into_py(py));
    }
    Ok(payloads)
}

#[pyfunction]
pub(crate) fn refinitiv_excel_ownership_validation_sheet_payloads(
    py: Python<'_>,
    rows: &Bound<'_, PyAny>,
    input_field_order: Vec<String>,
    block_slot_roles: Vec<String>,
    block_width: usize,
) -> PyResult<Vec<PyObject>> {
    let lookup_input_offset = input_field_order
        .iter()
        .position(|field| field == "lookup_input")
        .ok_or_else(|| PyValueError::new_err("input_field_order missing lookup_input"))?
        + 1;
    let request_start_offset = input_field_order
        .iter()
        .position(|field| field == "request_start_date")
        .ok_or_else(|| PyValueError::new_err("input_field_order missing request_start_date"))?
        + 1;
    let request_end_offset = input_field_order
        .iter()
        .position(|field| field == "request_end_date")
        .ok_or_else(|| PyValueError::new_err("input_field_order missing request_end_date"))?
        + 1;

    #[derive(Clone, Eq, PartialEq)]
    struct CaseRow {
        sheet_name: String,
        sheet_case_index: i64,
        case_band_row_start: i64,
        diagnostic_case_id: String,
    }

    let mut case_rows: Vec<CaseRow> = Vec::new();
    let mut seen_cases: HashSet<(String, i64, i64, String)> = HashSet::new();
    let mut role_to_input_values: HashMap<(String, String), Vec<String>> = HashMap::new();

    for row in rows.iter()? {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("ownership validation row is not a dict"))?;
        let sheet_name = dict_required_string(dict, "sheet_name")?;
        let sheet_case_index = dict_required_i64(dict, "sheet_case_index")?;
        let case_band_row_start = dict_required_i64(dict, "case_band_row_start")?;
        let diagnostic_case_id = dict_required_string(dict, "diagnostic_case_id")?;
        let block_slot_role = dict_required_string(dict, "block_slot_role")?;

        let case_key = (
            sheet_name.clone(),
            sheet_case_index,
            case_band_row_start,
            diagnostic_case_id.clone(),
        );
        if seen_cases.insert(case_key) {
            case_rows.push(CaseRow {
                sheet_name,
                sheet_case_index,
                case_band_row_start,
                diagnostic_case_id: diagnostic_case_id.clone(),
            });
        }

        let mut input_values: Vec<String> = Vec::with_capacity(input_field_order.len());
        for field_name in &input_field_order {
            input_values.push(dict_optional_rendered_value(dict, field_name)?);
        }
        role_to_input_values.insert((diagnostic_case_id, block_slot_role), input_values);
    }

    case_rows.sort_by(|left, right| {
        left.sheet_name
            .cmp(&right.sheet_name)
            .then(left.sheet_case_index.cmp(&right.sheet_case_index))
            .then(left.case_band_row_start.cmp(&right.case_band_row_start))
            .then(left.diagnostic_case_id.cmp(&right.diagnostic_case_id))
    });

    let mut sheet_payloads: Vec<PyObject> = Vec::new();
    let mut sheet_index = 0usize;
    while sheet_index < case_rows.len() {
        let sheet_name = case_rows[sheet_index].sheet_name.clone();
        let cases = PyList::empty_bound(py);
        while sheet_index < case_rows.len() && case_rows[sheet_index].sheet_name == sheet_name {
            let case = &case_rows[sheet_index];
            let start_row = case.case_band_row_start - 1;
            if start_row < 0 {
                return Err(PyValueError::new_err(
                    "case_band_row_start must be a positive 1-based row",
                ));
            }
            let slots = PyList::empty_bound(py);
            for (slot_idx, slot_role) in block_slot_roles.iter().enumerate() {
                let Some(input_values) =
                    role_to_input_values.get(&(case.diagnostic_case_id.clone(), slot_role.clone()))
                else {
                    continue;
                };
                let base_col = 1 + (slot_idx * block_width);
                let lookup_input_ref =
                    excel_cell_ref((start_row as usize) + lookup_input_offset, base_col);
                let request_start_ref =
                    excel_cell_ref((start_row as usize) + request_start_offset, base_col);
                let request_end_ref =
                    excel_cell_ref((start_row as usize) + request_end_offset, base_col);
                let formula = format!(
                    "=@RDP.Data({},\"TR.CategoryOwnershipPct.Date;TR.CategoryOwnershipPct;TR.InstrStatTypeValue\",\"StatType=7 SDate=\"&{}&\" EDate=\"&{}&\" CH=Fd RH=IN\")",
                    lookup_input_ref, request_start_ref, request_end_ref
                );
                let slot = PyDict::new_bound(py);
                slot.set_item("slot_index", slot_idx as i64)?;
                slot.set_item("base_col", base_col as i64)?;
                slot.set_item("input_values", PyList::new_bound(py, input_values.clone()))?;
                slot.set_item("formula", formula)?;
                slots.append(slot)?;
            }
            let case_out = PyDict::new_bound(py);
            case_out.set_item("diagnostic_case_id", &case.diagnostic_case_id)?;
            case_out.set_item("start_row", start_row)?;
            case_out.set_item("slots", slots)?;
            cases.append(case_out)?;
            sheet_index += 1;
        }
        let sheet = PyDict::new_bound(py);
        sheet.set_item("sheet_name", sheet_name)?;
        sheet.set_item("cases", cases)?;
        sheet_payloads.push(sheet.into_py(py));
    }
    Ok(sheet_payloads)
}

#[pyfunction]
pub(crate) fn refinitiv_excel_extended_summary_formula_payloads(
    py: Python<'_>,
    rows: &Bound<'_, PyAny>,
    summary_columns: Vec<String>,
    lookup_sheet_name: &str,
    lookup_column_names: Vec<String>,
    lookup_row_count: usize,
) -> PyResult<Vec<PyObject>> {
    let value_col_idx = summary_columns
        .iter()
        .position(|field| field == "value")
        .ok_or_else(|| PyValueError::new_err("summary columns missing value"))?;
    let mut lookup_columns: HashMap<String, usize> = HashMap::new();
    for (idx, name) in lookup_column_names.iter().enumerate() {
        lookup_columns.insert(name.clone(), idx);
    }

    let lookup_range = |column_name: &str| -> PyResult<String> {
        let Some(col_idx) = lookup_columns.get(column_name) else {
            return Err(PyValueError::new_err(format!(
                "lookup columns missing {column_name}"
            )));
        };
        Ok(excel_sheet_range_ref(
            lookup_sheet_name,
            1,
            lookup_row_count,
            *col_idx,
        ))
    };

    let identifier_types = ["ISIN", "CUSIP", "TICKER"];
    let mut single_success_summary_rows: HashMap<String, usize> = HashMap::new();
    let mut payloads: Vec<PyObject> = Vec::new();

    for (zero_idx, row) in rows.iter()?.enumerate() {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("summary formula row is not a dict"))?;
        let row_idx = zero_idx + 1;
        let category = dict_required_string(dict, "summary_category")?;
        let key = dict_required_string(dict, "summary_key")?;
        let mut formula: Option<String> = None;

        if category == "attempt_count_by_identifier_type" {
            formula = Some(format!(
                "=COUNTIF({},TRUE)",
                lookup_range(&format!("{key}_attempted"))?
            ));
        } else if category == "success_count_by_identifier_type" {
            formula = Some(format!(
                "=COUNTIF({},TRUE)",
                lookup_range(&format!("{key}_success"))?
            ));
        } else if let Some(field_name) = category.strip_prefix("agreement_count_same_") {
            formula = Some(format!(
                "=COUNTIF({},TRUE)",
                lookup_range(&format!("{key}_same_{field_name}"))?
            ));
        } else if let Some(field_name) = category.strip_prefix("conflict_count_same_") {
            formula = Some(format!(
                "=COUNTIF({},FALSE)",
                lookup_range(&format!("{key}_same_{field_name}"))?
            ));
        } else if category == "agreement_count_all_successful_attempts_consistent" {
            formula = Some(format!(
                "=COUNTIF({},TRUE)",
                lookup_range("all_successful_attempts_consistent")?
            ));
        } else if category == "conflict_count_all_successful_attempts_consistent" {
            formula = Some(format!(
                "=COUNTIF({},FALSE)",
                lookup_range("all_successful_attempts_consistent")?
            ));
        } else if category == "only_one_identifier_type_succeeds"
            && identifier_types.contains(&key.as_str())
        {
            let other_types: Vec<&str> = identifier_types
                .iter()
                .copied()
                .filter(|identifier_type| *identifier_type != key)
                .collect();
            formula = Some(format!(
                "=SUMPRODUCT(--({}=TRUE),--({}<>TRUE),--({}<>TRUE))",
                lookup_range(&format!("{key}_success"))?,
                lookup_range(&format!("{}_success", other_types[0]))?,
                lookup_range(&format!("{}_success", other_types[1]))?
            ));
            single_success_summary_rows.insert(key.clone(), row_idx);
        } else if category == "only_one_identifier_type_succeeds" && key == "total" {
            let mut row_refs: Vec<String> = Vec::new();
            for identifier_type in identifier_types {
                if let Some(summary_row_idx) = single_success_summary_rows.get(identifier_type) {
                    row_refs.push(excel_cell_ref(*summary_row_idx, value_col_idx));
                }
            }
            if !row_refs.is_empty() {
                formula = Some(format!("=SUM({})", row_refs.join(",")));
            }
        }

        if let Some(formula) = formula {
            let out = PyDict::new_bound(py);
            out.set_item("row_idx", row_idx as i64)?;
            out.set_item("formula", formula)?;
            match dict.get_item("value")? {
                Some(value) => out.set_item("value", value)?,
                None => out.set_item("value", py.None())?,
            }
            payloads.push(out.into_py(py));
        }
    }
    Ok(payloads)
}

#[pyfunction]
pub(crate) fn refinitiv_excel_extended_lookup_formula_payloads(
    py: Python<'_>,
    column_names: Vec<String>,
    row_count: usize,
) -> PyResult<Vec<PyObject>> {
    let mut column_map: HashMap<String, usize> = HashMap::new();
    for (idx, name) in column_names.iter().enumerate() {
        column_map.insert(name.clone(), idx);
    }
    for required_column in [
        "all_successful_attempts_consistent",
        "ISIN_lookup_input",
        "CUSIP_lookup_input",
        "TICKER_lookup_input",
    ] {
        if !column_map.contains_key(required_column) {
            return Err(PyValueError::new_err(format!(
                "extended lookup worksheet missing required column: {required_column}"
            )));
        }
    }

    let col = |name: &str| -> PyResult<usize> {
        column_map.get(name).copied().ok_or_else(|| {
            PyValueError::new_err(format!("extended lookup worksheet missing column: {name}"))
        })
    };
    let lookup_fields = [
        ("returned_ric", "TR.RIC"),
        ("returned_name", "TR.CommonName"),
        ("returned_isin", "TR.ISIN"),
        ("returned_cusip", "TR.CUSIP"),
    ];
    let identifier_types = ["ISIN", "CUSIP", "TICKER"];
    let identifier_pairs = [("ISIN", "CUSIP"), ("ISIN", "TICKER"), ("CUSIP", "TICKER")];

    let mut payloads: Vec<PyObject> = Vec::new();
    for row_idx in 1..=row_count {
        for identifier_type in identifier_types {
            let lookup_input_ref =
                excel_cell_ref(row_idx, col(&format!("{identifier_type}_lookup_input"))?);
            for (target_suffix, tr_field) in lookup_fields {
                let target_col = col(&format!("{identifier_type}_{target_suffix}"))?;
                let out = PyDict::new_bound(py);
                out.set_item("row_idx", row_idx as i64)?;
                out.set_item("col_idx", target_col as i64)?;
                out.set_item(
                    "formula",
                    format!("=IF({lookup_input_ref}=\"\",\"\",_xll.TR({lookup_input_ref},\"{tr_field}\"))"),
                )?;
                out.set_item("value", "")?;
                payloads.push(out.into_py(py));
            }

            let returned_ric_ref =
                excel_cell_ref(row_idx, col(&format!("{identifier_type}_returned_ric"))?);
            let attempted_col = col(&format!("{identifier_type}_attempted"))?;
            let attempted = PyDict::new_bound(py);
            attempted.set_item("row_idx", row_idx as i64)?;
            attempted.set_item("col_idx", attempted_col as i64)?;
            attempted.set_item("formula", format!("=LEN({lookup_input_ref})>0"))?;
            attempted.set_item("value", false)?;
            payloads.push(attempted.into_py(py));

            let success_col = col(&format!("{identifier_type}_success"))?;
            let success = PyDict::new_bound(py);
            success.set_item("row_idx", row_idx as i64)?;
            success.set_item("col_idx", success_col as i64)?;
            success.set_item(
                "formula",
                format!("=IFERROR({returned_ric_ref}<>\"\",FALSE)"),
            )?;
            success.set_item("value", false)?;
            payloads.push(success.into_py(py));
        }

        for (left_type, right_type) in identifier_pairs {
            for field_name in ["ric", "isin", "cusip"] {
                let left_ref =
                    excel_cell_ref(row_idx, col(&format!("{left_type}_returned_{field_name}"))?);
                let right_ref = excel_cell_ref(
                    row_idx,
                    col(&format!("{right_type}_returned_{field_name}"))?,
                );
                let target_col = col(&format!("{left_type}_vs_{right_type}_same_{field_name}"))?;
                let out = PyDict::new_bound(py);
                out.set_item("row_idx", row_idx as i64)?;
                out.set_item("col_idx", target_col as i64)?;
                out.set_item(
                    "formula",
                    format!(
                        "=IF(AND(IFERROR({left_ref}<>\"\",FALSE),IFERROR({right_ref}<>\"\",FALSE)),IFERROR({left_ref}={right_ref},FALSE),\"\")"
                    ),
                )?;
                out.set_item("value", "")?;
                payloads.push(out.into_py(py));
            }
        }

        let mut success_refs: Vec<String> = Vec::new();
        for identifier_type in identifier_types {
            success_refs.push(excel_cell_ref(
                row_idx,
                col(&format!("{identifier_type}_success"))?,
            ));
        }
        let mut pairwise_refs: Vec<String> = Vec::new();
        for (left_type, right_type) in identifier_pairs {
            for field_name in ["ric", "isin", "cusip"] {
                pairwise_refs.push(excel_cell_ref(
                    row_idx,
                    col(&format!("{left_type}_vs_{right_type}_same_{field_name}"))?,
                ));
            }
        }
        let success_count_expr = success_refs
            .iter()
            .map(|cell_ref| format!("N({cell_ref})"))
            .collect::<Vec<_>>()
            .join("+");
        let pairwise_and_terms = pairwise_refs
            .iter()
            .map(|cell_ref| format!("IF({cell_ref}=\"\",TRUE,{cell_ref})"))
            .collect::<Vec<_>>()
            .join(",");
        let out = PyDict::new_bound(py);
        out.set_item("row_idx", row_idx as i64)?;
        out.set_item("col_idx", col("all_successful_attempts_consistent")? as i64)?;
        out.set_item(
            "formula",
            format!("=IF(({success_count_expr})<2,\"\",AND({pairwise_and_terms}))"),
        )?;
        out.set_item("value", "")?;
        payloads.push(out.into_py(py));
    }
    Ok(payloads)
}

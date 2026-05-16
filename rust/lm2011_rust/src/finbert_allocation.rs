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
use crate::refinitiv_excel::*;
use crate::sec_extraction::*;
use crate::sentence_cleaning::*;
use crate::sentence_quality_api::*;

#[pyfunction]
#[pyo3(signature = (weights, capacities, target_rows, ensure_min_one=false))]
pub(crate) fn finbert_constrained_hamilton_allocations(
    weights: Vec<i64>,
    capacities: Vec<i64>,
    mut target_rows: i64,
    ensure_min_one: bool,
) -> PyResult<Vec<i64>> {
    if weights.len() != capacities.len() {
        return Err(PyValueError::new_err(
            "weights and capacities must have the same length.",
        ));
    }
    if target_rows < 0 {
        return Err(PyValueError::new_err("target_rows must be non-negative."));
    }
    if weights.iter().any(|weight| *weight < 0) {
        return Err(PyValueError::new_err("weights must be non-negative."));
    }
    if capacities.iter().any(|capacity| *capacity < 0) {
        return Err(PyValueError::new_err("capacities must be non-negative."));
    }
    let total_capacity: i64 = capacities.iter().sum();
    if target_rows > total_capacity {
        return Err(PyValueError::new_err(
            "target_rows cannot exceed total capacity.",
        ));
    }

    let mut allocations = vec![0_i64; weights.len()];
    let mut residual_caps = capacities;
    let positive_capacity_indices: Vec<usize> = residual_caps
        .iter()
        .enumerate()
        .filter_map(|(idx, capacity)| if *capacity > 0 { Some(idx) } else { None })
        .collect();
    if ensure_min_one && target_rows >= positive_capacity_indices.len() as i64 {
        for idx in positive_capacity_indices {
            allocations[idx] += 1;
            residual_caps[idx] -= 1;
        }
        target_rows -= residual_caps
            .iter()
            .zip(allocations.iter())
            .filter(|(_, allocation)| **allocation > 0)
            .count() as i64;
    }

    while target_rows > 0 {
        let active_indices: Vec<usize> = residual_caps
            .iter()
            .enumerate()
            .filter_map(|(idx, capacity)| if *capacity > 0 { Some(idx) } else { None })
            .collect();
        if active_indices.is_empty() {
            return Err(PyValueError::new_err(
                "No capacity remained while target_rows was still positive.",
            ));
        }

        let mut active_weights: Vec<i64> = active_indices.iter().map(|idx| weights[*idx]).collect();
        let mut weight_total: i64 = active_weights.iter().sum();
        if weight_total <= 0 {
            active_weights = active_indices
                .iter()
                .map(|idx| residual_caps[*idx])
                .collect();
            weight_total = active_weights.iter().sum();
        }
        if weight_total <= 0 {
            return Err(PyValueError::new_err(
                "No positive weight or residual capacity remained.",
            ));
        }

        let mut exact_rows: Vec<(usize, i128, i64)> = Vec::with_capacity(active_indices.len());
        let mut floor_total = 0_i64;
        for (idx, weight) in active_indices.iter().zip(active_weights.iter()) {
            let numerator = target_rows as i128 * *weight as i128;
            let denominator = weight_total as i128;
            let floor_count = residual_caps[*idx].min((numerator / denominator) as i64);
            let remainder = numerator % denominator;
            exact_rows.push((*idx, remainder, *weight));
            if floor_count > 0 {
                allocations[*idx] += floor_count;
                residual_caps[*idx] -= floor_count;
                floor_total += floor_count;
            }
        }

        if floor_total > 0 {
            target_rows -= floor_total;
            if target_rows == 0 {
                break;
            }
        }

        exact_rows.sort_by(|left, right| {
            right
                .1
                .cmp(&left.1)
                .then(right.2.cmp(&left.2))
                .then(left.0.cmp(&right.0))
        });
        let mut seats_awarded = 0_i64;
        for (idx, _, _) in exact_rows {
            if target_rows == 0 {
                break;
            }
            if residual_caps[idx] <= 0 {
                continue;
            }
            allocations[idx] += 1;
            residual_caps[idx] -= 1;
            target_rows -= 1;
            seats_awarded += 1;
        }
        if seats_awarded == 0 {
            return Err(PyValueError::new_err(
                "Could not allocate remaining target_rows under the capacity constraints.",
            ));
        }
    }

    Ok(allocations)
}

pub(crate) fn parse_iso_date_parts(value: &str) -> Option<(i32, u32, u32)> {
    let bytes = value.as_bytes();
    if bytes.len() != 10 || bytes[4] != b'-' || bytes[7] != b'-' {
        return None;
    }
    let mut parts = value.split('-');
    let year = parts.next()?.parse::<i32>().ok()?;
    let month = parts.next()?.parse::<u32>().ok()?;
    let day = parts.next()?.parse::<u32>().ok()?;
    if parts.next().is_some() || !(1..=12).contains(&month) {
        return None;
    }
    let max_day = days_in_month(year, month)?;
    if day < 1 || day > max_day {
        return None;
    }
    Some((year, month, day))
}

pub(crate) fn parse_strict_iso_date_parts(value: &str) -> Option<(i32, u32, u32)> {
    let bytes = value.as_bytes();
    if bytes.len() != 10 || bytes[4] != b'-' || bytes[7] != b'-' {
        return None;
    }
    if !bytes[..4].iter().all(|b| b.is_ascii_digit())
        || !bytes[5..7].iter().all(|b| b.is_ascii_digit())
        || !bytes[8..10].iter().all(|b| b.is_ascii_digit())
    {
        return None;
    }
    let year = value[..4].parse::<i32>().ok()?;
    if !(1..=9999).contains(&year) {
        return None;
    }
    let month = value[5..7].parse::<u32>().ok()?;
    let day = value[8..10].parse::<u32>().ok()?;
    let max_day = days_in_month(year, month)?;
    if day < 1 || day > max_day {
        return None;
    }
    Some((year, month, day))
}

pub(crate) fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

pub(crate) fn days_in_month(year: i32, month: u32) -> Option<u32> {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => Some(31),
        4 | 6 | 9 | 11 => Some(30),
        2 => Some(if is_leap_year(year) { 29 } else { 28 }),
        _ => None,
    }
}

pub(crate) fn date_ordinal(year: i32, month: u32, day: u32) -> PyResult<i64> {
    if year < 1 {
        return Err(PyValueError::new_err("date year out of range"));
    }
    let days_before_month_common: [i64; 12] =
        [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    let days_before_month_leap: [i64; 12] = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335];
    let year_index = i64::from(year - 1);
    let month_index =
        usize::try_from(month - 1).map_err(|_| PyValueError::new_err("date month out of range"))?;
    let days_before_year = year_index * 365 + year_index / 4 - year_index / 100 + year_index / 400;
    let days_before_month = if is_leap_year(year) {
        days_before_month_leap[month_index]
    } else {
        days_before_month_common[month_index]
    };
    Ok(days_before_year + days_before_month + i64::from(day))
}

pub(crate) fn parse_doc_ownership_date_text(value: &str) -> Option<(i32, u32, u32)> {
    let normalized = value.trim();
    if normalized.is_empty() {
        return None;
    }
    if let Some(parts) = parse_iso_date_parts(normalized) {
        return Some(parts);
    }
    let bytes = normalized.as_bytes();
    if bytes.len() >= 10
        && bytes[4] == b'-'
        && bytes[7] == b'-'
        && (bytes.len() == 10 || bytes[10] == b' ' || bytes[10] == b'T')
    {
        return parse_iso_date_parts(&normalized[..10]);
    }
    if let Some((date_part, _)) = normalized.split_once(' ') {
        return parse_iso_date_parts(date_part.trim());
    }
    None
}

#![allow(unused_imports)]

use pyo3::exceptions::{PyOSError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyDict, PyFloat, PyInt, PyList, PySet, PyString, PyTuple};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};

use crate::audit_summaries::*;
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
use crate::refinitiv_excel::*;
use crate::sec_extraction::*;
use crate::sentence_cleaning::*;
use crate::sentence_quality_api::*;

pub(crate) fn normalize_lookup_text_impl(value: Option<&str>) -> Option<String> {
    let value = value?;
    let normalized = value.trim();
    if normalized.is_empty() {
        None
    } else {
        Some(normalized.to_string())
    }
}

pub(crate) fn lm2011_dictionary_normalize_cell_impl(value: Option<&str>) -> String {
    value.unwrap_or("").trim().to_string()
}

pub(crate) fn lm2011_dictionary_active_membership_impl(value: Option<&str>) -> bool {
    let normalized = lm2011_dictionary_normalize_cell_impl(value);
    !normalized.is_empty() && normalized != "0" && !normalized.starts_with('-')
}

pub(crate) fn python_like_zfill_i64(value: i64, width: usize) -> String {
    let rendered = value.to_string();
    if rendered.len() >= width {
        return rendered;
    }
    if let Some(rest) = rendered.strip_prefix('-') {
        let zeros = width.saturating_sub(1 + rest.len());
        format!("-{}{}", "0".repeat(zeros), rest)
    } else {
        format!("{}{}", "0".repeat(width - rendered.len()), rendered)
    }
}

pub(crate) fn py_int_like_to_i64(value: &Bound<'_, PyAny>) -> PyResult<i64> {
    if value.is_instance_of::<PyBool>() {
        return Ok(if value.extract::<bool>()? { 1 } else { 0 });
    }
    if value.is_instance_of::<PyInt>() {
        return value.extract::<i64>();
    }
    if value.is_instance_of::<PyFloat>() {
        let number = value.extract::<f64>()?;
        if number.is_finite() {
            return Ok(number.trunc() as i64);
        }
        return Err(PyValueError::new_err("non-finite float"));
    }
    if value.is_instance_of::<PyString>() {
        let rendered = value.str()?;
        let text = rendered.to_str()?.trim();
        return text
            .parse::<i64>()
            .map_err(|_| PyValueError::new_err("invalid integer text"));
    }
    Err(PyValueError::new_err("unsupported integer-like value"))
}

pub(crate) fn py_float_like_to_finite_option(value: &Bound<'_, PyAny>) -> PyResult<Option<f64>> {
    if value.is_none() {
        return Ok(None);
    }
    if value.is_instance_of::<PyString>() {
        let rendered = value.str()?;
        let text = rendered.to_str()?.trim();
        if text.is_empty() {
            return Ok(None);
        }
        if text.contains('_') {
            return Err(PyValueError::new_err("underscore-containing float text"));
        }
        return match text.parse::<f64>() {
            Ok(number) => {
                if number.is_finite() {
                    Ok(Some(number))
                } else {
                    Ok(None)
                }
            }
            Err(_) => Ok(None),
        };
    }
    match value.extract::<f64>() {
        Ok(number) => {
            if number.is_finite() {
                Ok(Some(number))
            } else {
                Ok(None)
            }
        }
        Err(err) => {
            if err.is_instance_of::<PyTypeError>(value.py())
                || err.is_instance_of::<PyValueError>(value.py())
            {
                Ok(None)
            } else {
                Err(err)
            }
        }
    }
}

pub(crate) fn clean_form_token_impl(value: Option<&str>) -> Option<String> {
    let value = value?;
    let cleaned: String = value
        .trim()
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .flat_map(|ch| ch.to_uppercase())
        .collect();
    if cleaned.is_empty() {
        None
    } else {
        Some(cleaned)
    }
}

pub(crate) fn canonical_lm2011_form(cleaned: &str) -> Option<&'static str> {
    match cleaned {
        "10-K" | "10-K405" | "10-KT" | "10K" | "10K405" | "10KT" => Some("10-K"),
        "10-K/A" | "10-K-A" | "10-K405/A" | "10-K405-A" | "10-KT/A" | "10-KT-A" | "10K/A"
        | "10K405/A" | "10KT/A" => Some("10-K/A"),
        "10-Q" | "10-QT" | "10Q" | "10QT" => Some("10-Q"),
        "10-Q/A" | "10-Q-A" | "10-QT/A" | "10-QT-A" | "10Q/A" | "10QT/A" => Some("10-Q/A"),
        "20F" | "20-F" => Some("20-F"),
        "20F/A" | "20-F/A" => Some("20-F/A"),
        "40F" | "40-F" => Some("40-F"),
        "40F/A" | "40-F/A" => Some("40-F/A"),
        _ => None,
    }
}

pub(crate) fn ccm_form_match_token_impl(value: Option<&str>) -> Option<String> {
    let value = value?;
    let normalized: String = value
        .trim()
        .chars()
        .flat_map(|ch| ch.to_uppercase())
        .filter(|ch| *ch != ' ' && *ch != '-')
        .collect();
    if normalized.is_empty() {
        return None;
    }
    match normalized.as_str() {
        "10KA" => Some("10K/A".to_string()),
        "10QA" => Some("10Q/A".to_string()),
        "10KTA" => Some("10KT/A".to_string()),
        "10QTA" => Some("10QT/A".to_string()),
        "10K405A" => Some("10K405/A".to_string()),
        _ => Some(normalized),
    }
}

pub(crate) fn parse_ff48_header_line(line: &str) -> Option<(i64, String, String)> {
    let trimmed = line.trim();
    let first_end = trimmed.find(char::is_whitespace)?;
    let id_text = &trimmed[..first_end];
    if id_text.is_empty() || id_text.len() > 2 || !id_text.as_bytes().iter().all(u8::is_ascii_digit)
    {
        return None;
    }
    let rest = trimmed[first_end..].trim_start();
    let second_end = rest.find(char::is_whitespace)?;
    let short = &rest[..second_end];
    if short.is_empty() || !short.as_bytes().iter().all(u8::is_ascii_alphanumeric) {
        return None;
    }
    let name = rest[second_end..].trim();
    if name.is_empty() {
        return None;
    }
    Some((
        id_text.parse::<i64>().ok()?,
        short.to_string(),
        name.to_string(),
    ))
}

pub(crate) fn parse_ff48_range_line(line: &str) -> Option<(i64, i64)> {
    let trimmed = line.trim();
    let bytes = trimmed.as_bytes();
    if bytes.len() < 10 {
        return None;
    }
    if !bytes[..4].iter().all(u8::is_ascii_digit)
        || bytes[4] != b'-'
        || !bytes[5..9].iter().all(u8::is_ascii_digit)
        || !bytes[9].is_ascii_whitespace()
    {
        return None;
    }
    if trimmed[9..].trim().is_empty() {
        return None;
    }
    Some((
        trimmed[..4].parse::<i64>().ok()?,
        trimmed[5..9].parse::<i64>().ok()?,
    ))
}

#[pyfunction]
pub(crate) fn parse_ff48_sic_mapping_rows(text: &str) -> Vec<(i64, String, String, i64, i64)> {
    let mut rows: Vec<(i64, String, String, i64, i64)> = Vec::new();
    let mut current: Option<(i64, String, String)> = None;
    for raw_line in text.lines() {
        if raw_line.trim().is_empty() {
            continue;
        }
        if let Some(header) = parse_ff48_header_line(raw_line) {
            current = Some(header);
            continue;
        }
        if let (Some((industry_id, industry_short, industry_name)), Some((sic_start, sic_end))) =
            (current.as_ref(), parse_ff48_range_line(raw_line))
        {
            rows.push((
                *industry_id,
                industry_short.clone(),
                industry_name.clone(),
                sic_start,
                sic_end,
            ));
        }
    }
    rows
}

pub(crate) fn normalize_sec_raw_form_impl(value: Option<&str>) -> Option<String> {
    let mut cleaned = clean_form_token_impl(value)?;
    let compact = cleaned.replace(['-', '/'], "");
    let mapped = match compact.as_str() {
        "10K" => Some("10-K"),
        "10K405" => Some("10-K405"),
        "10KT" => Some("10-KT"),
        "10KA" => Some("10-K-A"),
        "10K405A" => Some("10-K405-A"),
        "10KTA" => Some("10-KT-A"),
        "10Q" => Some("10-Q"),
        "10QT" => Some("10-QT"),
        "10QA" => Some("10-Q-A"),
        "10QTA" => Some("10-QT-A"),
        "10KSB" => Some("10KSB"),
        "10KSBA" => Some("10KSB-A"),
        "10KSB40" => Some("10KSB40"),
        "10KSB40A" => Some("10KSB40-A"),
        _ => None,
    };
    if let Some(value) = mapped {
        return Some(value.to_string());
    }

    cleaned = cleaned.replace("/A", "-A");
    if cleaned.ends_with('A') && !cleaned.contains('/') && !cleaned.contains("-A") {
        for base in [
            "10-K405", "10-KT", "10-K", "10KSB40", "10KSB", "10-QT", "10-Q",
        ] {
            if cleaned == format!("{base}A") {
                return Some(format!("{base}-A"));
            }
        }
    }
    Some(cleaned)
}

pub(crate) fn normalize_ccm_raw_form_impl(value: Option<&str>) -> Option<String> {
    let cleaned = clean_form_token_impl(value)?;
    let normalized = cleaned.replace('-', "");
    match normalized.as_str() {
        "10KA" => Some("10K/A".to_string()),
        "10QA" => Some("10Q/A".to_string()),
        "10KTA" => Some("10KT/A".to_string()),
        "10QTA" => Some("10QT/A".to_string()),
        "20FA" => Some("20F/A".to_string()),
        "40FA" => Some("40F/A".to_string()),
        _ => Some(normalized),
    }
}

pub(crate) fn py_str_normalized(value: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    let rendered = value.str()?;
    Ok(normalize_lookup_text_impl(Some(rendered.to_str()?)))
}

pub(crate) fn collapse_whitespace_to_spaces(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut pending_space = false;
    for ch in text.chars() {
        if ch.is_whitespace() {
            if !out.is_empty() {
                pending_space = true;
            }
            continue;
        }
        if pending_space && !out.is_empty() {
            out.push(' ');
        }
        out.push(ch);
        pending_space = false;
    }
    out
}

pub(crate) fn normalize_spaces_impl(text: &str) -> String {
    collapse_whitespace_to_spaces(text.trim())
        .trim()
        .to_string()
}

pub(crate) fn normalize_newlines_impl(text: Option<&str>) -> String {
    let Some(text) = text else {
        return String::new();
    };
    text.replace("\r\n", "\n").replace('\r', "\n")
}

pub(crate) fn collapse_blank_runs_impl(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut newline_run = 0usize;
    for ch in text.chars() {
        if ch == '\n' {
            newline_run += 1;
            if newline_run <= 2 {
                out.push('\n');
            }
        } else {
            newline_run = 0;
            out.push(ch);
        }
    }
    out.trim().to_string()
}

pub(crate) fn normalize_ascii_with_positions_impl(text: &str) -> Option<(String, Vec<usize>)> {
    if !text.is_ascii() {
        return None;
    }
    let mut normalized = String::with_capacity(text.len());
    let mut positions: Vec<usize> = Vec::with_capacity(text.len());
    let mut pending_space = false;
    let mut started = false;
    for (index, byte) in text.bytes().enumerate() {
        if byte.is_ascii_whitespace() {
            if started {
                pending_space = true;
            }
            continue;
        }
        if pending_space && !normalized.is_empty() {
            normalized.push(' ');
            positions.push(index);
        }
        normalized.push((byte as char).to_ascii_lowercase());
        positions.push(index);
        pending_space = false;
        started = true;
    }
    Some((normalized, positions))
}

pub(crate) fn normalized_match_bounds_impl(
    normalized_text: &str,
    normalized_to_original: &[i64],
    query: &str,
) -> Option<(i64, i64)> {
    if !normalized_text.is_ascii() || !query.is_ascii() {
        return None;
    }
    let normalized_query = normalize_spaces_impl(query).to_ascii_lowercase();
    if normalized_query.len() < 40 {
        return None;
    }
    let start = normalized_text.find(&normalized_query)?;
    let end = start + normalized_query.len() - 1;
    if end >= normalized_to_original.len() {
        return None;
    }
    let original_start = *normalized_to_original.get(start)?;
    let original_end = normalized_to_original.get(end)?.saturating_add(1);
    Some((original_start, original_end))
}

pub(crate) fn stable_digits_int_impl(value: &str) -> Option<i64> {
    if !value.is_ascii() {
        return None;
    }
    let mut digits: Vec<u8> = Vec::with_capacity(15);
    for byte in value.bytes() {
        if !byte.is_ascii_digit() {
            continue;
        }
        if digits.len() == 15 {
            digits.remove(0);
        }
        digits.push(byte);
    }
    if digits.is_empty() {
        return None;
    }
    std::str::from_utf8(&digits).ok()?.parse::<i64>().ok()
}

pub(crate) fn is_ascii_word_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

pub(crate) fn contains_ascii_word_phrase(text: &str, phrase: &str) -> bool {
    let mut search_start = 0usize;
    while let Some(relative_pos) = text[search_start..].find(phrase) {
        let start = search_start + relative_pos;
        let end = start + phrase.len();
        let before_ok = start == 0 || !is_ascii_word_byte(text.as_bytes()[start - 1]);
        let after_ok = end >= text.len() || !is_ascii_word_byte(text.as_bytes()[end]);
        if before_ok && after_ok {
            return true;
        }
        search_start = end;
    }
    false
}

pub(crate) fn contains_boundary_leak(normalized_lower: &str) -> bool {
    [
        "item 1a",
        "item 1b",
        "item 2",
        "item 7a",
        "item 8",
        "signature",
        "signatures",
        "index to financial statements",
        "legal proceedings",
    ]
    .iter()
    .any(|phrase| contains_ascii_word_phrase(normalized_lower, phrase))
}

pub(crate) fn item_toc_hit_count(text: &str) -> usize {
    let bytes = text.as_bytes();
    let mut pos = 0usize;
    let mut count = 0usize;
    while pos + 4 <= bytes.len() {
        let Some(relative_pos) = text[pos..].find("item") else {
            break;
        };
        let start = pos + relative_pos;
        let mut probe = start + 4;
        if probe >= bytes.len() || !bytes[probe].is_ascii_whitespace() {
            pos = start + 4;
            continue;
        }
        while probe < bytes.len() && bytes[probe].is_ascii_whitespace() {
            probe += 1;
        }
        let digit_start = probe;
        while probe < bytes.len() && bytes[probe].is_ascii_digit() {
            probe += 1;
        }
        if probe == digit_start {
            pos = start + 4;
            continue;
        }
        if probe < bytes.len() && bytes[probe].is_ascii_alphabetic() {
            probe += 1;
        }
        let boundary_ok = probe >= bytes.len() || !is_ascii_word_byte(bytes[probe]);
        if boundary_ok {
            count += 1;
            pos = probe;
        } else {
            pos = start + 4;
        }
    }
    count
}

pub(crate) fn multisurface_boundary_snippet_risk_impl(snippets: Vec<Option<String>>) -> bool {
    let mut joined = String::new();
    let mut saw_snippet = false;
    for snippet in snippets.into_iter().flatten() {
        if snippet.is_empty() {
            continue;
        }
        if saw_snippet {
            joined.push('\n');
        }
        joined.push_str(snippet.replace("\r\n", "\n").replace('\r', "\n").trim());
        saw_snippet = true;
    }
    if joined.is_empty() {
        return false;
    }
    let normalized_lower = normalize_spaces_impl(&joined).to_ascii_lowercase();
    contains_boundary_leak(&normalized_lower) || item_toc_hit_count(&normalized_lower) >= 2
}

pub(crate) fn optional_string_values(values: &Bound<'_, PyAny>) -> PyResult<Vec<Option<String>>> {
    let mut out = Vec::new();
    for value in values.iter()? {
        let value = value?;
        if value.is_none() {
            out.push(None);
        } else {
            out.push(Some(value.str()?.to_str()?.to_string()));
        }
    }
    Ok(out)
}

pub(crate) fn validate_equal_column_len(
    name: &str,
    expected: usize,
    actual: usize,
) -> PyResult<()> {
    if actual != expected {
        return Err(PyValueError::new_err(format!(
            "{name} length {actual} does not match expected length {expected}"
        )));
    }
    Ok(())
}

pub(crate) fn normalized_optional_spaces(value: &Option<String>) -> String {
    normalize_spaces_impl(value.as_deref().unwrap_or(""))
}

pub(crate) fn sha1_digest(data: &[u8]) -> [u8; 20] {
    let mut h0: u32 = 0x67452301;
    let mut h1: u32 = 0xEFCDAB89;
    let mut h2: u32 = 0x98BADCFE;
    let mut h3: u32 = 0x10325476;
    let mut h4: u32 = 0xC3D2E1F0;

    let bit_len = (data.len() as u64) * 8;
    let mut message = data.to_vec();
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in message.chunks(64) {
        let mut w = [0u32; 80];
        for i in 0..16 {
            let start = i * 4;
            w[i] = u32::from_be_bytes([
                chunk[start],
                chunk[start + 1],
                chunk[start + 2],
                chunk[start + 3],
            ]);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }

        let mut a = h0;
        let mut b = h1;
        let mut c = h2;
        let mut d = h3;
        let mut e = h4;

        for i in 0..80 {
            let (f, k) = match i {
                0..=19 => ((b & c) | ((!b) & d), 0x5A827999),
                20..=39 => (b ^ c ^ d, 0x6ED9EBA1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1BBCDC),
                _ => (b ^ c ^ d, 0xCA62C1D6),
            };
            let temp = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(w[i]);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = temp;
        }

        h0 = h0.wrapping_add(a);
        h1 = h1.wrapping_add(b);
        h2 = h2.wrapping_add(c);
        h3 = h3.wrapping_add(d);
        h4 = h4.wrapping_add(e);
    }

    let mut out = [0u8; 20];
    out[0..4].copy_from_slice(&h0.to_be_bytes());
    out[4..8].copy_from_slice(&h1.to_be_bytes());
    out[8..12].copy_from_slice(&h2.to_be_bytes());
    out[12..16].copy_from_slice(&h3.to_be_bytes());
    out[16..20].copy_from_slice(&h4.to_be_bytes());
    out
}

pub(crate) const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

pub(crate) const BLAKE2B_IV: [u64; 8] = [
    0x6a09e667f3bcc908,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179,
];

pub(crate) const BLAKE2B_SIGMA: [[usize; 16]; 12] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
];

pub(crate) fn sha256_digest(data: &[u8]) -> [u8; 32] {
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    let bit_len = (data.len() as u64) * 8;
    let mut message = data.to_vec();
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in message.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            let start = i * 4;
            w[i] = u32::from_be_bytes([
                chunk[start],
                chunk[start + 1],
                chunk[start + 2],
                chunk[start + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let mut a = h[0];
        let mut b = h[1];
        let mut c = h[2];
        let mut d = h[3];
        let mut e = h[4];
        let mut f = h[5];
        let mut g = h[6];
        let mut hh = h[7];

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(SHA256_K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut out = [0u8; 32];
    for (idx, word) in h.iter().enumerate() {
        out[idx * 4..idx * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

pub(crate) fn sha256_compress(h: &mut [u32; 8], chunk: &[u8]) {
    let mut w = [0u32; 64];
    for i in 0..16 {
        let start = i * 4;
        w[i] = u32::from_be_bytes([
            chunk[start],
            chunk[start + 1],
            chunk[start + 2],
            chunk[start + 3],
        ]);
    }
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }

    let mut a = h[0];
    let mut b = h[1];
    let mut c = h[2];
    let mut d = h[3];
    let mut e = h[4];
    let mut f = h[5];
    let mut g = h[6];
    let mut hh = h[7];

    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ ((!e) & g);
        let temp1 = hh
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(SHA256_K[i])
            .wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let temp2 = s0.wrapping_add(maj);

        hh = g;
        g = f;
        f = e;
        e = d.wrapping_add(temp1);
        d = c;
        c = b;
        b = a;
        a = temp1.wrapping_add(temp2);
    }

    h[0] = h[0].wrapping_add(a);
    h[1] = h[1].wrapping_add(b);
    h[2] = h[2].wrapping_add(c);
    h[3] = h[3].wrapping_add(d);
    h[4] = h[4].wrapping_add(e);
    h[5] = h[5].wrapping_add(f);
    h[6] = h[6].wrapping_add(g);
    h[7] = h[7].wrapping_add(hh);
}

pub(crate) struct Sha256State {
    h: [u32; 8],
    buffer: [u8; 64],
    buffer_len: usize,
    len_bytes: u64,
}

impl Sha256State {
    pub(crate) fn new() -> Self {
        Self {
            h: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            buffer: [0u8; 64],
            buffer_len: 0,
            len_bytes: 0,
        }
    }

    pub(crate) fn update(&mut self, mut data: &[u8]) {
        self.len_bytes = self.len_bytes.wrapping_add(data.len() as u64);
        if self.buffer_len > 0 {
            let needed = 64 - self.buffer_len;
            let take = needed.min(data.len());
            self.buffer[self.buffer_len..self.buffer_len + take].copy_from_slice(&data[..take]);
            self.buffer_len += take;
            data = &data[take..];
            if self.buffer_len == 64 {
                sha256_compress(&mut self.h, &self.buffer);
                self.buffer_len = 0;
            }
        }
        while data.len() >= 64 {
            sha256_compress(&mut self.h, &data[..64]);
            data = &data[64..];
        }
        if !data.is_empty() {
            self.buffer[..data.len()].copy_from_slice(data);
            self.buffer_len = data.len();
        }
    }

    pub(crate) fn finalize(mut self) -> [u8; 32] {
        let bit_len = self.len_bytes.wrapping_mul(8);
        self.buffer[self.buffer_len] = 0x80;
        self.buffer_len += 1;
        if self.buffer_len > 56 {
            for idx in self.buffer_len..64 {
                self.buffer[idx] = 0;
            }
            sha256_compress(&mut self.h, &self.buffer);
            self.buffer_len = 0;
        }
        for idx in self.buffer_len..56 {
            self.buffer[idx] = 0;
        }
        self.buffer[56..64].copy_from_slice(&bit_len.to_be_bytes());
        sha256_compress(&mut self.h, &self.buffer);

        let mut out = [0u8; 32];
        for (idx, word) in self.h.iter().enumerate() {
            out[idx * 4..idx * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }
}

pub(crate) fn bytes_to_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

pub(crate) fn blake2b_g(v: &mut [u64; 16], a: usize, b: usize, c: usize, d: usize, x: u64, y: u64) {
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
    v[d] = (v[d] ^ v[a]).rotate_right(32);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(24);
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
    v[d] = (v[d] ^ v[a]).rotate_right(16);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(63);
}

pub(crate) fn blake2b_compress(
    h: &mut [u64; 8],
    block: &[u8; 128],
    byte_count: u128,
    is_last: bool,
) {
    let mut m = [0u64; 16];
    for i in 0..16 {
        let start = i * 8;
        m[i] = u64::from_le_bytes([
            block[start],
            block[start + 1],
            block[start + 2],
            block[start + 3],
            block[start + 4],
            block[start + 5],
            block[start + 6],
            block[start + 7],
        ]);
    }

    let mut v = [0u64; 16];
    v[..8].copy_from_slice(h);
    v[8..].copy_from_slice(&BLAKE2B_IV);
    v[12] ^= byte_count as u64;
    v[13] ^= (byte_count >> 64) as u64;
    if is_last {
        v[14] = !v[14];
    }

    for sigma in BLAKE2B_SIGMA {
        blake2b_g(&mut v, 0, 4, 8, 12, m[sigma[0]], m[sigma[1]]);
        blake2b_g(&mut v, 1, 5, 9, 13, m[sigma[2]], m[sigma[3]]);
        blake2b_g(&mut v, 2, 6, 10, 14, m[sigma[4]], m[sigma[5]]);
        blake2b_g(&mut v, 3, 7, 11, 15, m[sigma[6]], m[sigma[7]]);
        blake2b_g(&mut v, 0, 5, 10, 15, m[sigma[8]], m[sigma[9]]);
        blake2b_g(&mut v, 1, 6, 11, 12, m[sigma[10]], m[sigma[11]]);
        blake2b_g(&mut v, 2, 7, 8, 13, m[sigma[12]], m[sigma[13]]);
        blake2b_g(&mut v, 3, 4, 9, 14, m[sigma[14]], m[sigma[15]]);
    }

    for i in 0..8 {
        h[i] ^= v[i] ^ v[i + 8];
    }
}

pub(crate) fn blake2b_8_digest(data: &[u8]) -> [u8; 8] {
    let mut h = BLAKE2B_IV;
    h[0] ^= 0x01010000 ^ 8;
    let mut byte_count = 0u128;
    let mut remaining = data;

    while remaining.len() > 128 {
        let mut block = [0u8; 128];
        block.copy_from_slice(&remaining[..128]);
        byte_count += 128;
        blake2b_compress(&mut h, &block, byte_count, false);
        remaining = &remaining[128..];
    }

    let mut final_block = [0u8; 128];
    final_block[..remaining.len()].copy_from_slice(remaining);
    byte_count += remaining.len() as u128;
    blake2b_compress(&mut h, &final_block, byte_count, true);
    h[0].to_le_bytes()
}

pub(crate) fn blake2b_8_u64_big_endian(text: &str) -> u64 {
    u64::from_be_bytes(blake2b_8_digest(text.as_bytes()))
}

pub(crate) fn sha256_hex_impl(text: &str) -> String {
    bytes_to_hex(&sha256_digest(text.as_bytes()))
}

pub(crate) fn sha256_first_u64_impl(text: &str) -> u64 {
    let digest = sha256_digest(text.as_bytes());
    u64::from_be_bytes([
        digest[0], digest[1], digest[2], digest[3], digest[4], digest[5], digest[6], digest[7],
    ])
}

pub(crate) fn finbert_selection_key_impl(
    doc_id: &str,
    benchmark_item_code: &str,
    seed: i64,
) -> String {
    sha256_hex_impl(&format!("{seed}|{doc_id}|{benchmark_item_code}"))
}

pub(crate) fn legacy_lm2011_tokens_impl(text: Option<&str>) -> Vec<String> {
    let Some(text) = text else {
        return Vec::new();
    };
    let bytes = text.as_bytes();
    let mut out: Vec<String> = Vec::new();
    let mut index = 0usize;
    while index < bytes.len() {
        if !bytes[index].is_ascii_alphabetic() {
            index += 1;
            continue;
        }
        let mut token = String::new();
        while index < bytes.len() && bytes[index].is_ascii_alphabetic() {
            token.push((bytes[index] as char).to_ascii_lowercase());
            index += 1;
        }
        if index + 1 < bytes.len()
            && bytes[index] == b'\''
            && bytes[index + 1].is_ascii_alphabetic()
        {
            token.push('\'');
            index += 1;
            while index < bytes.len() && bytes[index].is_ascii_alphabetic() {
                token.push((bytes[index] as char).to_ascii_lowercase());
                index += 1;
            }
        }
        out.push(token);
    }
    out
}

pub(crate) fn truncate_text_impl(text: Option<&str>, limit: i64) -> Option<String> {
    let cleaned = collapse_whitespace_to_spaces(text?);
    let cleaned_len = cleaned.chars().count() as i64;
    if cleaned_len <= limit {
        return Some(cleaned);
    }
    let take_len = limit.saturating_sub(3).max(0) as usize;
    let mut prefix: String = cleaned.chars().take(take_len).collect();
    while prefix.ends_with(char::is_whitespace) {
        prefix.pop();
    }
    Some(format!("{prefix}..."))
}

pub(crate) fn min_index(left: Option<usize>, right: Option<usize>) -> Option<usize> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.min(right)),
        (Some(left), None) => Some(left),
        (None, Some(right)) => Some(right),
        (None, None) => None,
    }
}

pub(crate) fn find_any_literal_start(haystack: &str, needles: &[&str]) -> Option<usize> {
    needles
        .iter()
        .filter_map(|needle| haystack.find(needle))
        .min()
}

pub(crate) fn find_literal_with_ascii_boundary_after(
    haystack: &str,
    needle: &str,
) -> Option<usize> {
    let mut best: Option<usize> = None;
    for (idx, _) in haystack.match_indices(needle) {
        let after = idx + needle.len();
        let has_boundary = match haystack.as_bytes().get(after) {
            Some(byte) => !is_ascii_word_byte(*byte),
            None => true,
        };
        if has_boundary {
            best = min_index(best, Some(idx));
        }
    }
    best
}

pub(crate) fn find_exhibit_tag_start(lower: &str) -> Option<usize> {
    let bytes = lower.as_bytes();
    let mut best: Option<usize> = None;
    for prefix in ["<ex-", "</ex-"] {
        for (idx, _) in lower.match_indices(prefix) {
            let mut pos = idx + prefix.len();
            let Some(first) = bytes.get(pos) else {
                continue;
            };
            if !first.is_ascii_alphanumeric() {
                continue;
            }
            pos += 1;
            while let Some(byte) = bytes.get(pos) {
                if byte.is_ascii_alphanumeric() || matches!(*byte, b'.' | b'-') {
                    pos += 1;
                } else {
                    break;
                }
            }
            if matches!(bytes.get(pos), Some(b'>')) {
                best = min_index(best, Some(idx));
            }
        }
    }
    best
}

pub(crate) fn find_exhibit_number_start(lower: &str) -> Option<usize> {
    let bytes = lower.as_bytes();
    let mut best: Option<usize> = None;
    for (idx, _) in lower.match_indices("exhibit") {
        let mut pos = idx + "exhibit".len();
        let Some(next) = bytes.get(pos) else {
            continue;
        };
        if !next.is_ascii_whitespace() {
            continue;
        }
        while matches!(bytes.get(pos), Some(byte) if byte.is_ascii_whitespace()) {
            pos += 1;
        }
        let mut digits = 0usize;
        while digits < 3 {
            let Some(byte) = bytes.get(pos) else {
                break;
            };
            if !byte.is_ascii_digit() {
                break;
            }
            digits += 1;
            pos += 1;
        }
        if digits > 0 {
            best = min_index(best, Some(idx));
        }
    }
    best
}

pub(crate) fn validation_marker_start(lower: &str, marker_name: &str) -> Option<usize> {
    match marker_name {
        "sec_header" => find_any_literal_start(
            lower,
            &[
                "<sec-header",
                "conformed submission type",
                "public document count",
                "accession number:",
            ],
        ),
        "html" => {
            let mut best = find_any_literal_start(lower, &["</html>", "</div>", "</body>", "</p>"]);
            for token in ["<div", "<body", "<p"] {
                best = min_index(best, find_literal_with_ascii_boundary_after(lower, token));
            }
            best = min_index(best, lower.find("<html"));
            best
        }
        "table" => {
            let mut best = find_any_literal_start(lower, &["</table>", "</tr>", "</td>"]);
            for token in ["<table", "<tr", "<td"] {
                best = min_index(best, find_literal_with_ascii_boundary_after(lower, token));
            }
            best
        }
        "exhibit" => {
            let mut best = find_any_literal_start(lower, &["exhibit index", "index to exhibits"]);
            best = min_index(best, find_exhibit_tag_start(lower));
            best = min_index(best, find_exhibit_number_start(lower));
            best
        }
        _ => None,
    }
}

pub(crate) fn char_to_byte_index(text: &str, char_idx: usize) -> usize {
    if char_idx == 0 {
        return 0;
    }
    text.char_indices()
        .nth(char_idx)
        .map_or(text.len(), |(byte_idx, _)| byte_idx)
}

pub(crate) fn byte_to_char_index(text: &str, byte_idx: usize) -> usize {
    text[..byte_idx].chars().count()
}

pub(crate) fn validation_marker_flags_and_snippet_impl(
    text: Option<&str>,
    snippet_char_limit: i64,
) -> (BTreeMap<String, bool>, Option<String>) {
    let marker_names = ["sec_header", "html", "table", "exhibit"];
    let mut flags: BTreeMap<String, bool> = marker_names
        .iter()
        .map(|name| (name.to_string(), false))
        .collect();
    let Some(text) = text else {
        return (flags, None);
    };
    let lower = text.to_ascii_lowercase();
    let mut first_match_start: Option<usize> = None;
    for name in marker_names {
        let match_start = validation_marker_start(&lower, name);
        flags.insert(name.to_string(), match_start.is_some());
        if first_match_start.is_none() {
            first_match_start = match_start;
        }
    }

    let Some(first_byte_start) = first_match_start else {
        return (flags, None);
    };
    let first_char_start = byte_to_char_index(text, first_byte_start);
    let total_chars = text.chars().count() as i64;
    let snippet_start = first_char_start.saturating_sub(80);
    let raw_end = (first_char_start as i64).saturating_add(snippet_char_limit);
    let snippet_end = raw_end.clamp(0, total_chars) as usize;
    let snippet_text = if snippet_end <= snippet_start {
        ""
    } else {
        let start_byte = char_to_byte_index(text, snippet_start);
        let end_byte = char_to_byte_index(text, snippet_end);
        &text[start_byte..end_byte]
    };
    let snippet = truncate_text_impl(Some(snippet_text), snippet_char_limit);
    (flags, snippet)
}

pub(crate) fn push_json_ascii_string(out: &mut String, value: &str) -> PyResult<()> {
    out.push('"');
    for byte in value.bytes() {
        match byte {
            b'"' => out.push_str("\\\""),
            b'\\' => out.push_str("\\\\"),
            b'\x08' => out.push_str("\\b"),
            b'\x0c' => out.push_str("\\f"),
            b'\n' => out.push_str("\\n"),
            b'\r' => out.push_str("\\r"),
            b'\t' => out.push_str("\\t"),
            0x00..=0x1f => out.push_str(&format!("\\u{byte:04x}")),
            0x20..=0x7e => out.push(byte as char),
            _ => {
                return Err(PyValueError::new_err(
                    "non-ASCII RIC values fall back to Python stable_hash_id",
                ))
            }
        }
    }
    out.push('"');
    Ok(())
}

pub(crate) fn analyst_request_group_payload(
    gvkey_int: i64,
    effective_collection_ric: &str,
) -> PyResult<String> {
    let mut payload = String::new();
    payload.push('[');
    push_json_ascii_string(&mut payload, "analyst_request_group")?;
    payload.push(',');
    payload.push_str(&gvkey_int.to_string());
    payload.push(',');
    push_json_ascii_string(&mut payload, effective_collection_ric)?;
    payload.push(']');
    Ok(payload)
}

pub(crate) fn push_stable_json_simple_part(
    out: &mut String,
    value: &Bound<'_, PyAny>,
) -> PyResult<()> {
    if value.is_none() {
        out.push_str("null");
        return Ok(());
    }
    if value.is_instance_of::<PyBool>() {
        out.push_str(if value.extract::<bool>()? {
            "true"
        } else {
            "false"
        });
        return Ok(());
    }
    if value.is_instance_of::<PyInt>() {
        let number = value.extract::<i64>()?;
        out.push_str(&number.to_string());
        return Ok(());
    }
    if value.is_instance_of::<PyFloat>() {
        let number = value.extract::<f64>()?;
        if number.is_nan() {
            out.push_str("NaN");
        } else if number == f64::INFINITY {
            out.push_str("Infinity");
        } else if number == f64::NEG_INFINITY {
            out.push_str("-Infinity");
        } else {
            let rendered = value.repr()?;
            let text = rendered.to_str()?;
            if !text.is_ascii() {
                return Err(PyValueError::new_err(
                    "non-ASCII float representation falls back to Python",
                ));
            }
            out.push_str(text);
        }
        return Ok(());
    }
    if value.is_instance_of::<PyString>() {
        push_json_ascii_string(out, value.extract::<&str>()?)?;
        return Ok(());
    }
    if let Ok(list) = value.downcast::<PyList>() {
        out.push('[');
        for (index, entry) in list.iter().enumerate() {
            if index > 0 {
                out.push(',');
            }
            push_stable_json_simple_part(out, &entry)?;
        }
        out.push(']');
        return Ok(());
    }
    if let Ok(tuple) = value.downcast::<PyTuple>() {
        out.push('[');
        for (index, entry) in tuple.iter().enumerate() {
            if index > 0 {
                out.push(',');
            }
            push_stable_json_simple_part(out, &entry)?;
        }
        out.push(']');
        return Ok(());
    }
    if let Ok(dict) = value.downcast::<PyDict>() {
        let mut items: BTreeMap<String, Bound<'_, PyAny>> = BTreeMap::new();
        for (key, dict_value) in dict.iter() {
            if !key.is_instance_of::<PyString>() {
                return Err(PyValueError::new_err(
                    "unsupported stable_hash_id dict key falls back to Python",
                ));
            }
            let key_text = key.extract::<&str>()?;
            if !key_text.is_ascii() {
                return Err(PyValueError::new_err(
                    "non-ASCII stable_hash_id dict key falls back to Python",
                ));
            }
            items.insert(key_text.to_string(), dict_value);
        }
        out.push('{');
        for (index, (key, dict_value)) in items.iter().enumerate() {
            if index > 0 {
                out.push(',');
            }
            push_json_ascii_string(out, key)?;
            out.push(':');
            push_stable_json_simple_part(out, dict_value)?;
        }
        out.push('}');
        return Ok(());
    }
    Err(PyValueError::new_err(
        "unsupported stable_hash_id part falls back to Python",
    ))
}

pub(crate) fn stable_hash_id_from_payload(prefix: &str, payload: &str) -> String {
    let digest = sha1_digest(payload.as_bytes());
    let mut out = String::with_capacity(prefix.len() + 17);
    out.push_str(prefix);
    out.push('_');
    for byte in digest.iter().take(8) {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

pub(crate) fn sha1_hex_from_bytes(data: &[u8]) -> String {
    let digest = sha1_digest(data);
    let mut out = String::with_capacity(40);
    for byte in digest.iter() {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

pub(crate) fn push_json_default_str_part(
    out: &mut String,
    value: &Bound<'_, PyAny>,
) -> PyResult<()> {
    if value.is_none() {
        out.push_str("null");
        return Ok(());
    }
    if value.is_instance_of::<PyBool>() {
        out.push_str(if value.extract::<bool>()? {
            "true"
        } else {
            "false"
        });
        return Ok(());
    }
    if value.is_instance_of::<PyInt>() {
        let number = value.extract::<i64>()?;
        out.push_str(&number.to_string());
        return Ok(());
    }
    if value.is_instance_of::<PyFloat>() {
        let number = value.extract::<f64>()?;
        if number.is_nan() {
            out.push_str("NaN");
        } else if number == f64::INFINITY {
            out.push_str("Infinity");
        } else if number == f64::NEG_INFINITY {
            out.push_str("-Infinity");
        } else {
            out.push_str(value.repr()?.to_str()?);
        }
        return Ok(());
    }
    if value.is_instance_of::<PyString>() {
        push_json_ascii_string(out, value.extract::<&str>()?)?;
        return Ok(());
    }
    if let Ok(list) = value.downcast::<PyList>() {
        out.push('[');
        for (index, entry) in list.iter().enumerate() {
            if index > 0 {
                out.push(',');
            }
            push_json_default_str_part(out, &entry)?;
        }
        out.push(']');
        return Ok(());
    }
    if let Ok(tuple) = value.downcast::<PyTuple>() {
        out.push('[');
        for (index, entry) in tuple.iter().enumerate() {
            if index > 0 {
                out.push(',');
            }
            push_json_default_str_part(out, &entry)?;
        }
        out.push(']');
        return Ok(());
    }
    if let Ok(dict) = value.downcast::<PyDict>() {
        let mut items: BTreeMap<String, Bound<'_, PyAny>> = BTreeMap::new();
        for (key, dict_value) in dict.iter() {
            if !key.is_instance_of::<PyString>() {
                return Err(PyValueError::new_err(
                    "unsupported frame fingerprint dict key falls back to Python",
                ));
            }
            let key_text = key.extract::<&str>()?;
            if !key_text.is_ascii() {
                return Err(PyValueError::new_err(
                    "non-ASCII frame fingerprint dict key falls back to Python",
                ));
            }
            items.insert(key_text.to_string(), dict_value);
        }
        out.push('{');
        for (index, (key, dict_value)) in items.iter().enumerate() {
            if index > 0 {
                out.push(',');
            }
            push_json_ascii_string(out, key)?;
            out.push(':');
            push_json_default_str_part(out, dict_value)?;
        }
        out.push('}');
        return Ok(());
    }
    let rendered = value.str()?;
    push_json_ascii_string(out, rendered.to_str()?)
}

pub(crate) fn push_request_signature_json_value(
    out: &mut String,
    value: &Bound<'_, PyAny>,
) -> PyResult<()> {
    if value.is_none() {
        out.push_str("null");
        return Ok(());
    }
    if value.is_instance_of::<PyBool>() {
        out.push_str(if value.extract::<bool>()? {
            "true"
        } else {
            "false"
        });
        return Ok(());
    }
    if value.is_instance_of::<PyInt>() {
        let number = value.extract::<i64>()?;
        out.push_str(&number.to_string());
        return Ok(());
    }
    if value.is_instance_of::<PyFloat>() {
        let number = value.extract::<f64>()?;
        if number.is_nan() {
            out.push_str("NaN");
        } else if number == f64::INFINITY {
            out.push_str("Infinity");
        } else if number == f64::NEG_INFINITY {
            out.push_str("-Infinity");
        } else {
            let rendered = value.repr()?;
            let text = rendered.to_str()?;
            if !text.is_ascii() {
                return Err(PyValueError::new_err(
                    "non-ASCII request-signature float falls back to Python",
                ));
            }
            out.push_str(text);
        }
        return Ok(());
    }
    if value.is_instance_of::<PyString>() {
        push_json_ascii_string(out, value.extract::<&str>()?)?;
        return Ok(());
    }
    Err(PyValueError::new_err(
        "unsupported request-signature value falls back to Python",
    ))
}

pub(crate) fn collect_request_signature_text_iter(
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<BTreeSet<String>> {
    let Some(value) = value else {
        return Ok(BTreeSet::new());
    };
    if value.is_none() {
        return Ok(BTreeSet::new());
    }
    let mut out: BTreeSet<String> = BTreeSet::new();
    for entry in value.iter()? {
        let entry = entry?;
        if !entry.is_instance_of::<PyString>() {
            return Err(PyValueError::new_err(
                "non-string excluded request-signature key falls back to Python",
            ));
        }
        let key = entry.extract::<&str>()?;
        if !key.is_ascii() {
            return Err(PyValueError::new_err(
                "non-ASCII excluded request-signature key falls back to Python",
            ));
        }
        out.insert(key.to_string());
    }
    Ok(out)
}

pub(crate) fn push_request_signature_payload(
    out: &mut String,
    stage: &str,
    fields: &Bound<'_, PyAny>,
    parameters: Option<&Bound<'_, PyAny>>,
    excluded_parameter_keys: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let excluded = collect_request_signature_text_iter(excluded_parameter_keys)?;
    let mut field_values: Vec<String> = Vec::new();
    for field in fields.iter()? {
        let field = field?;
        if !field.is_instance_of::<PyString>() {
            return Err(PyValueError::new_err(
                "non-string request-signature field falls back to Python",
            ));
        }
        let text = field.extract::<&str>()?;
        if !text.is_ascii() {
            return Err(PyValueError::new_err(
                "non-ASCII request-signature field falls back to Python",
            ));
        }
        field_values.push(text.to_string());
    }

    let mut filtered_parameters: BTreeMap<String, Bound<'_, PyAny>> = BTreeMap::new();
    if let Some(parameters) = parameters {
        if !parameters.is_none() {
            let dict = parameters.downcast::<PyDict>().map_err(|_| {
                PyValueError::new_err("unsupported request parameters fall back to Python")
            })?;
            for (key, value) in dict.iter() {
                if !key.is_instance_of::<PyString>() {
                    return Err(PyValueError::new_err(
                        "non-string request parameter key falls back to Python",
                    ));
                }
                let key_text = key.extract::<&str>()?;
                if !key_text.is_ascii() {
                    return Err(PyValueError::new_err(
                        "non-ASCII request parameter key falls back to Python",
                    ));
                }
                if !excluded.contains(key_text) {
                    filtered_parameters.insert(key_text.to_string(), value);
                }
            }
        }
    }

    out.push('[');
    push_json_ascii_string(out, stage)?;
    out.push_str(",[");
    for (index, field) in field_values.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        push_json_ascii_string(out, field)?;
    }
    out.push_str("],{");
    for (index, (key, value)) in filtered_parameters.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        push_json_ascii_string(out, key)?;
        out.push(':');
        push_request_signature_json_value(out, value)?;
    }
    out.push_str("}]");
    Ok(())
}

pub(crate) fn parse_lseg_identifier_list(raw_identifiers: &str) -> Vec<String> {
    raw_identifiers
        .trim()
        .split(',')
        .filter_map(|token| {
            let normalized = token.trim().trim_matches(|ch| ch == '\'' || ch == '"');
            if normalized.is_empty() {
                None
            } else {
                Some(normalized.to_string())
            }
        })
        .collect()
}

#[pyfunction]
pub(crate) fn parse_lseg_identifier_list_value(raw_identifiers: &str) -> Vec<String> {
    parse_lseg_identifier_list(raw_identifiers)
}

#[pyfunction]
pub(crate) fn parse_lseg_unresolved_identifiers_value(message: &str) -> Vec<String> {
    parse_lseg_unresolved_identifiers_impl(message)
}

#[pyfunction]
pub(crate) fn lseg_utf8_record_rows(
    py: Python<'_>,
    records: &Bound<'_, PyAny>,
) -> PyResult<Vec<PyObject>> {
    let mut out_rows = Vec::new();
    for row in records.iter()? {
        let row = row?;
        let dict = row
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("LSEG UTF-8 coercion row is not a dict"))?;
        let out = PyDict::new_bound(py);
        for (key, value) in dict.iter() {
            let key_text = key.str()?.to_str()?.to_string();
            if value.is_none() {
                out.set_item(key_text, Option::<String>::None)?;
            } else {
                out.set_item(key_text, value.str()?.to_str()?)?;
            }
        }
        out_rows.push(out.into_py(py));
    }
    Ok(out_rows)
}

pub(crate) fn is_sensitive_lseg_header_name(name: &str) -> bool {
    matches!(
        name.trim().to_ascii_lowercase().as_str(),
        "authorization"
            | "cookie"
            | "set-cookie"
            | "x-api-key"
            | "x-auth-token"
            | "proxy-authorization"
    )
}

#[pyfunction]
pub(crate) fn lseg_sanitize_headers(
    py: Python<'_>,
    headers: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let dict = headers
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("LSEG headers value is not a dict"))?;
    let out = PyDict::new_bound(py);
    for (key, value) in dict.iter() {
        let key_text = key.str()?.to_str()?.to_string();
        if !is_sensitive_lseg_header_name(&key_text) {
            out.set_item(key_text, value)?;
        }
    }
    Ok(out.into_py(py))
}

pub(crate) fn parse_lseg_unresolved_identifiers_impl(message: &str) -> Vec<String> {
    let lowered = message.to_ascii_lowercase();
    let direct_marker = "unable to resolve all requested identifiers in [";
    if let Some(marker_start) = lowered.find(direct_marker) {
        let start = marker_start + direct_marker.len();
        if let Some(relative_end) = lowered[start..].find("].") {
            return parse_lseg_identifier_list(&message[start..start + relative_end]);
        }
    }

    if lowered.contains("unable to collect data for the field")
        && lowered.contains("some specific identifier(s).")
    {
        let requested_marker = "requested universes:";
        if let Some(marker_start) = lowered.find(requested_marker) {
            let after_marker = marker_start + requested_marker.len();
            if let Some(relative_open) = lowered[after_marker..].find('[') {
                let start = after_marker + relative_open + 1;
                if let Some(relative_end) = lowered[start..].find(']') {
                    return parse_lseg_identifier_list(&message[start..start + relative_end]);
                }
            }
        }
    }

    Vec::new()
}

pub(crate) fn classify_lseg_error_message_impl(message: &str) -> (Option<String>, Vec<String>) {
    let unresolved = parse_lseg_unresolved_identifiers_impl(message);
    if !unresolved.is_empty() {
        return (Some("unresolved_identifiers".to_string()), unresolved);
    }

    let normalized = message.to_ascii_lowercase();
    if is_lseg_session_not_opened_message_impl(&normalized) {
        return (Some("session_open_failed".to_string()), Vec::new());
    }
    if normalized.contains("timed out") || normalized.contains("timeout") {
        if ["localhost:9000", "/api/udf", "refinitivworkspace.exe"]
            .iter()
            .any(|marker| normalized.contains(marker))
        {
            return (Some("workspace_proxy_timeout".to_string()), Vec::new());
        }
        return (Some("transport_timeout".to_string()), Vec::new());
    }
    if [
        "service unavailable",
        "overload",
        "backend",
        "502",
        "503",
        "504",
    ]
    .iter()
    .any(|token| normalized.contains(token))
    {
        return (Some("backend_overload".to_string()), Vec::new());
    }
    (None, Vec::new())
}

pub(crate) fn is_lseg_session_not_opened_message_impl(normalized_message: &str) -> bool {
    normalized_message.contains("session is not opened")
        && normalized_message.contains("can't send any request")
}

#[pyfunction]
pub(crate) fn is_lseg_session_not_opened_message(message: &str) -> bool {
    is_lseg_session_not_opened_message_impl(&message.to_ascii_lowercase())
}

pub(crate) fn extract_py_date_parts(value: &Bound<'_, PyAny>) -> PyResult<(i32, u32, u32)> {
    let year = value.getattr("year")?.extract::<i32>()?;
    let month = value.getattr("month")?.extract::<u32>()?;
    let day = value.getattr("day")?.extract::<u32>()?;
    if !(1..=12).contains(&month) {
        return Err(PyValueError::new_err("date month out of range"));
    }
    let Some(max_day) = days_in_month(year, month) else {
        return Err(PyValueError::new_err("date month out of range"));
    };
    if day < 1 || day > max_day {
        return Err(PyValueError::new_err("date day out of range"));
    }
    Ok((year, month, day))
}

pub(crate) fn py_date_object(
    py: Python<'_>,
    year: i32,
    month: u32,
    day: u32,
) -> PyResult<PyObject> {
    Ok(py
        .import_bound("datetime")?
        .getattr("date")?
        .call1((year, month, day))?
        .into_py(py))
}

pub(crate) fn iso_date_string(year: i32, month: u32, day: u32) -> String {
    format!("{year:04}-{month:02}-{day:02}")
}

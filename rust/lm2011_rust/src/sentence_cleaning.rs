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
use crate::refinitiv_excel::*;
use crate::sec_extraction::*;
use crate::sentence_quality_api::*;

pub(crate) fn normalize_sentence_key_impl(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_whitespace = false;
    for ch in text.chars() {
        if ch.is_whitespace() {
            if !out.is_empty() {
                in_whitespace = true;
            }
            continue;
        }
        if in_whitespace && !out.is_empty() {
            out.push(' ');
        }
        out.push(ch);
        in_whitespace = false;
    }
    out.trim().to_string()
}

pub(crate) fn normalize_ascii_sample_text_impl(text: &str) -> Option<String> {
    if !text.is_ascii() {
        return None;
    }
    let mut out = String::with_capacity(text.len());
    let mut pending_space = false;
    for byte in text.bytes() {
        if byte.is_ascii_whitespace() {
            if !out.is_empty() {
                pending_space = true;
            }
            continue;
        }
        if pending_space && !out.is_empty() {
            out.push(' ');
        }
        out.push((byte as char).to_ascii_lowercase());
        pending_space = false;
    }
    Some(out)
}

pub(crate) fn is_roman_char(ch: char) -> bool {
    matches!(
        ch.to_ascii_lowercase(),
        'i' | 'v' | 'x' | 'l' | 'c' | 'd' | 'm'
    )
}

pub(crate) fn is_numeric_or_roman_run(chars: &[char], pos: &mut usize) -> bool {
    if *pos >= chars.len() {
        return false;
    }
    if chars[*pos].is_ascii_digit() {
        while *pos < chars.len() && chars[*pos].is_ascii_digit() {
            *pos += 1;
        }
        return true;
    }
    if is_roman_char(chars[*pos]) {
        while *pos < chars.len() && is_roman_char(chars[*pos]) {
            *pos += 1;
        }
        return true;
    }
    false
}

pub(crate) fn skip_spaces(chars: &[char], pos: &mut usize) {
    while *pos < chars.len() && chars[*pos].is_whitespace() {
        *pos += 1;
    }
}

pub(crate) fn numeric_only_pattern(normalized: &str) -> bool {
    if normalized == "." {
        return true;
    }
    let chars: Vec<char> = normalized.chars().collect();
    let mut pos = 0usize;
    let mut opened = false;
    if pos < chars.len() && chars[pos] == '(' {
        opened = true;
        pos += 1;
    }
    skip_spaces(&chars, &mut pos);
    if !is_numeric_or_roman_run(&chars, &mut pos) {
        return false;
    }
    loop {
        skip_spaces(&chars, &mut pos);
        if pos >= chars.len() || !matches!(chars[pos], '-' | '.' | '/') {
            break;
        }
        let separator = chars[pos];
        let separator_pos = pos;
        pos += 1;
        skip_spaces(&chars, &mut pos);
        if !is_numeric_or_roman_run(&chars, &mut pos) {
            if separator == '.' {
                pos = separator_pos;
                break;
            }
            return false;
        }
    }
    skip_spaces(&chars, &mut pos);
    if pos < chars.len() && matches!(chars[pos], '.' | ')') {
        pos += 1;
    }
    if opened && pos < chars.len() && chars[pos] == ')' {
        pos += 1;
    }
    skip_spaces(&chars, &mut pos);
    pos == chars.len()
}

pub(crate) fn sentence_is_separator_line_impl(text: &str) -> bool {
    let stripped = text.trim();
    stripped.len() >= 3
        && stripped
            .chars()
            .all(|ch| matches!(ch, '-' | '_' | '=' | '*'))
}

pub(crate) fn sentence_numeric_only_fragment_impl(text: &str) -> bool {
    let stripped = normalize_sentence_key_impl(text);
    if stripped.is_empty() || sentence_is_separator_line_impl(&stripped) {
        return false;
    }
    numeric_only_pattern(&stripped) || matches!(stripped.as_str(), "." | "," | ";" | ":")
}

pub(crate) fn sentence_short_fragment_impl(text: &str, token_count: i64, char_count: i64) -> bool {
    let normalized = normalize_sentence_key_impl(text);
    if normalized.is_empty() {
        return false;
    }
    if sentence_numeric_only_fragment_impl(&normalized)
        || sentence_is_separator_line_impl(&normalized)
    {
        return true;
    }
    token_count <= 10 && char_count <= 48
}

pub(crate) fn sentence_very_short_fragment_impl(
    text: &str,
    token_count: i64,
    char_count: i64,
) -> bool {
    let normalized = normalize_sentence_key_impl(text);
    !normalized.is_empty() && token_count <= 4 && char_count <= 16
}

pub(crate) fn sentence_lower_fragment_impl(text: &str, token_count: i64, char_count: i64) -> bool {
    let normalized = normalize_sentence_key_impl(text);
    if normalized.is_empty() || token_count > 12 || char_count > 64 {
        return false;
    }
    let chars: Vec<char> = normalized.chars().collect();
    let Some(first) = chars.first() else {
        return false;
    };
    if !first.is_ascii_lowercase() {
        return false;
    }
    let mut end = chars.len();
    if end > 1 && matches!(chars[end - 1], '.' | '!' | '?') {
        end -= 1;
    }
    let body_len = end.saturating_sub(1);
    body_len <= 60
        && chars[1..end]
            .iter()
            .all(|ch| !matches!(ch, '.' | '!' | '?'))
}

pub(crate) fn sentence_one_word_fragment_impl(text: &str) -> bool {
    let mut word_count = 0usize;
    let mut in_word = false;
    for ch in normalize_sentence_key_impl(text).chars() {
        if ch.is_ascii_alphabetic() {
            if !in_word {
                word_count += 1;
                if word_count > 1 {
                    return false;
                }
                in_word = true;
            }
        } else {
            in_word = false;
        }
    }
    word_count == 1
}

pub(crate) fn sentence_has_terminal_punct_impl(text: &str) -> bool {
    let normalized = normalize_sentence_key_impl(text);
    let mut chars = normalized.chars().rev();
    for ch in chars.by_ref() {
        if matches!(ch, '"' | '\'' | ')' | ']') {
            continue;
        }
        return matches!(ch, '.' | '!' | '?');
    }
    false
}

pub(crate) fn has_separator_run(text: &str) -> bool {
    let mut run = 0usize;
    for ch in text.chars() {
        if matches!(ch, '-' | '_' | '=') {
            run += 1;
            if run >= 3 {
                return true;
            }
        } else {
            run = 0;
        }
    }
    false
}

fn is_ascii_regex_word(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '_'
}

pub(crate) fn count_table_numeric_tokens(text: &str) -> usize {
    let chars: Vec<char> = text.chars().collect();
    let mut count = 0usize;
    let mut pos = 0usize;
    while pos < chars.len() {
        let ch = chars[pos];
        let previous_is_word = pos > 0 && is_ascii_regex_word(chars[pos - 1]);
        if !ch.is_ascii_digit() || previous_is_word {
            pos += 1;
            continue;
        }

        let mut end = pos + 1;
        while end < chars.len()
            && (chars[end].is_ascii_digit()
                || matches!(chars[end], ',' | '(' | ')' | '.' | '%' | '-'))
        {
            end += 1;
        }

        let mut match_end = None;
        for candidate_end in ((pos + 1)..=end).rev() {
            let left_is_word = is_ascii_regex_word(chars[candidate_end - 1]);
            let right_is_word =
                candidate_end < chars.len() && is_ascii_regex_word(chars[candidate_end]);
            if left_is_word != right_is_word {
                match_end = Some(candidate_end);
                break;
            }
        }

        if let Some(candidate_end) = match_end {
            count += 1;
            pos = candidate_end.max(pos + 1);
        } else {
            pos += 1;
        }
    }
    count
}

pub(crate) fn count_uppercase_tokens(text: &str) -> usize {
    text.split(|ch: char| !(ch.is_ascii_alphanumeric() || matches!(ch, '/' | '&' | '-')))
        .filter(|token| token.len() >= 3)
        .filter(|token| {
            let mut chars = token.chars();
            chars.next().is_some_and(|ch| ch.is_ascii_uppercase())
                && chars.all(|ch| ch.is_ascii_uppercase() || matches!(ch, '/' | '&' | '-'))
        })
        .count()
}

pub(crate) fn count_year_tokens(text: &str) -> usize {
    text.split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter(|token| {
            token.len() == 4
                && token.chars().all(|ch| ch.is_ascii_digit())
                && (token.starts_with("19") || token.starts_with("20"))
        })
        .count()
}

pub(crate) fn sentence_table_like_impl(text: &str, token_count: usize) -> bool {
    let compact = normalize_sentence_key_impl(text);
    let line_count = text.chars().filter(|ch| *ch == '\n').count() + 1;
    let numeric_tokens = count_table_numeric_tokens(text);
    let uppercase_tokens = count_uppercase_tokens(text);
    let year_tokens = count_year_tokens(text);
    let colon_hits = text.matches(':').count();
    if has_separator_run(text) {
        return true;
    }
    if token_count < 24 {
        return false;
    }
    if line_count >= 3 && numeric_tokens >= 4 {
        return true;
    }
    if line_count >= 3 && uppercase_tokens >= 6 {
        return true;
    }
    if numeric_tokens >= 8 && year_tokens >= 2 {
        return true;
    }
    if uppercase_tokens >= 8 && colon_hits >= 2 {
        return true;
    }
    compact.to_uppercase() == compact && line_count >= 3 && compact.len() >= 80
}

pub(crate) fn count_cleaning_numeric_tokens(text: &str) -> usize {
    let bytes = text.as_bytes();
    let mut count = 0usize;
    let mut pos = 0usize;
    while pos < bytes.len() {
        let start = pos;
        if bytes[pos] == b'$' {
            pos += 1;
        }
        if pos < bytes.len() && bytes[pos] == b'(' {
            pos += 1;
        }
        if pos >= bytes.len() || !bytes[pos].is_ascii_digit() {
            pos = start + 1;
            continue;
        }
        pos += 1;
        while pos < bytes.len() && (bytes[pos].is_ascii_digit() || bytes[pos] == b',') {
            pos += 1;
        }
        if pos < bytes.len() && bytes[pos] == b'.' {
            pos += 1;
            while pos < bytes.len() && bytes[pos].is_ascii_digit() {
                pos += 1;
            }
        }
        if pos < bytes.len() && bytes[pos] == b'%' {
            pos += 1;
        }
        if pos < bytes.len() && bytes[pos] == b')' {
            pos += 1;
        }
        let left_ok = start == 0 || !bytes[start - 1].is_ascii_alphabetic();
        let right_ok = pos == bytes.len() || !bytes[pos].is_ascii_alphabetic();
        if left_ok && right_ok {
            count += 1;
        }
    }
    count
}

pub(crate) fn count_cleaning_word_tokens(text: &str) -> usize {
    let mut count = 0usize;
    let mut run = 0usize;
    for byte in text.bytes() {
        if byte.is_ascii_alphabetic() {
            run += 1;
            continue;
        }
        if run >= 2 {
            count += 1;
        }
        run = 0;
    }
    if run >= 2 {
        count += 1;
    }
    count
}

pub(crate) fn contains_cleaning_month_name(text: &str) -> bool {
    let lowered = text.to_ascii_lowercase();
    [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ]
    .iter()
    .any(|month| contains_ascii_word_or_phrase(&lowered, month))
}

pub(crate) fn contains_cleaning_from_to_range(text: &str) -> bool {
    let lowered = text.to_ascii_lowercase();
    let Some(from_pos) = find_ascii_word(&lowered, "from", 0) else {
        return false;
    };
    find_ascii_word(&lowered, "to", from_pos + 4).is_some()
}

pub(crate) fn contains_cleaning_prose_numeric_verb(text: &str) -> bool {
    let lowered = text.to_ascii_lowercase();
    [
        "we",
        "our",
        "was",
        "were",
        "incurred",
        "expensed",
        "increased",
        "decreased",
        "approximately",
        "respectively",
        "totaled",
    ]
    .iter()
    .any(|word| contains_ascii_word_or_phrase(&lowered, word))
}

pub(crate) fn find_ascii_word(text: &str, needle: &str, start_at: usize) -> Option<usize> {
    let haystack = text.as_bytes();
    let needle_bytes = needle.as_bytes();
    if haystack.len() < needle_bytes.len() || start_at > haystack.len() - needle_bytes.len() {
        return None;
    }
    for pos in start_at..=haystack.len() - needle_bytes.len() {
        if &haystack[pos..pos + needle_bytes.len()] != needle_bytes {
            continue;
        }
        let left_ok = pos == 0 || !ascii_word_byte(haystack[pos - 1]);
        let right_pos = pos + needle_bytes.len();
        let right_ok = right_pos == haystack.len() || !ascii_word_byte(haystack[right_pos]);
        if left_ok && right_ok {
            return Some(pos);
        }
    }
    None
}

pub(crate) fn cleaning_is_table_like_line_impl(line: &str) -> bool {
    let stripped = line.trim();
    if stripped.chars().count() < 25 {
        return false;
    }
    let numeric_count = count_cleaning_numeric_tokens(stripped);
    let word_count = count_cleaning_word_tokens(stripped);
    if contains_cleaning_month_name(stripped)
        && (contains_cleaning_from_to_range(stripped)
            || contains_cleaning_prose_numeric_verb(stripped))
    {
        return false;
    }
    numeric_count >= 4 && (word_count <= 3 || numeric_count >= std::cmp::max(4, word_count))
}

pub(crate) fn contains_cleaning_table_header_keyword(stripped: &str) -> bool {
    let lowered = stripped.to_ascii_lowercase();
    [
        "consolidated",
        "statement",
        "statements",
        "balance sheet",
        "balance sheets",
        "cash flow",
        "cash flows",
        "segment",
        "line of business",
        "less than",
        "more than",
        "year ended",
        "years ended",
        "as of",
        "payments due by period",
        "selected financial data",
        "note to consolidated",
        "notes to consolidated",
    ]
    .iter()
    .any(|needle| contains_ascii_word_or_phrase(&lowered, needle))
}

pub(crate) fn count_cleaning_year_tokens(stripped: &str) -> usize {
    stripped
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter(|token| {
            token.len() == 4
                && token.chars().all(|ch| ch.is_ascii_digit())
                && (token.starts_with("19") || token.starts_with("20"))
        })
        .count()
}

pub(crate) fn cleaning_is_table_header_like_line_impl(line: &str) -> bool {
    let stripped = line.trim();
    if stripped.chars().count() < 20 {
        return false;
    }
    if !contains_cleaning_table_header_keyword(stripped) {
        return false;
    }
    let letter_count = stripped.chars().filter(|ch| ch.is_alphabetic()).count();
    let year_count = count_cleaning_year_tokens(stripped);
    letter_count >= 8
        && (stripped.to_uppercase() == stripped || year_count >= 2 || stripped.contains(':'))
}

pub(crate) fn cleaning_is_strong_table_title_line_impl(line: &str) -> bool {
    let stripped = line.trim();
    stripped.chars().count() >= 15
        && contains_cleaning_table_header_keyword(stripped)
        && stripped.to_uppercase() == stripped
}

pub(crate) fn cleaning_is_table_intro_line_impl(line: &str) -> bool {
    let stripped = line.trim();
    if stripped.is_empty() || !(stripped.ends_with(':') || stripped.chars().count() <= 140) {
        return false;
    }
    let lowered = stripped.to_ascii_lowercase();
    if contains_ascii_word_or_phrase(&lowered, "summarized as follows")
        || contains_ascii_word_or_phrase(&lowered, "set forth in the table below")
    {
        return true;
    }
    let table_phrase = contains_ascii_word_or_phrase(&lowered, "following table")
        || contains_ascii_word_or_phrase(&lowered, "the following table")
        || contains_ascii_word_or_phrase(&lowered, "below table")
        || contains_ascii_word_or_phrase(&lowered, "the below table");
    if !table_phrase {
        return false;
    }
    [
        "summary",
        "summaries",
        "summarize",
        "summarized",
        "summarizes",
        "present",
        "presents",
        "presented",
        "show",
        "shows",
        "shown",
        "reflect",
        "reflects",
        "reflected",
        "provide",
        "provides",
        "provided",
    ]
    .iter()
    .any(|word| contains_ascii_word_or_phrase(&lowered, word))
}

pub(crate) fn cleaning_is_table_unit_line_impl(stripped: &str) -> bool {
    let lowered = stripped
        .trim()
        .trim_start_matches('(')
        .trim_end_matches(')')
        .trim()
        .to_ascii_lowercase();
    let tokens: Vec<&str> = lowered.split_whitespace().collect();
    match tokens.as_slice() {
        ["million" | "millions" | "thousand" | "thousands"] => true,
        ["dollar" | "dollars", "in", "million" | "millions" | "thousand" | "thousands"] => true,
        ["million" | "millions" | "thousand" | "thousands", "of", "dollar" | "dollars"] => true,
        ["in", "million" | "millions" | "thousand" | "thousands", "of", "dollar" | "dollars"] => {
            true
        }
        _ => false,
    }
}

pub(crate) fn cleaning_years_only_line_impl(stripped: &str) -> bool {
    let lowered = stripped.trim().to_ascii_lowercase();
    for prefix in [
        "years ended",
        "years ending",
        "years shown",
        "years later",
        "years compared",
        "years through",
        "year ended",
        "year ending",
        "year shown",
        "year later",
        "year compared",
        "year through",
        "as of",
    ] {
        if lowered.starts_with(prefix) {
            return true;
        }
    }
    let years: Vec<&str> = stripped.split_whitespace().collect();
    years.len() >= 2
        && years.iter().all(|token| {
            token.len() == 4
                && token.chars().all(|ch| ch.is_ascii_digit())
                && (token.starts_with("19") || token.starts_with("20"))
        })
}

pub(crate) fn cleaning_is_table_support_header_line_impl(line: &str) -> bool {
    let stripped = line.trim();
    !stripped.is_empty()
        && (cleaning_is_table_unit_line_impl(stripped) || cleaning_years_only_line_impl(stripped))
}

pub(crate) fn cleaning_toc_like_line_impl(line: &str) -> bool {
    let stripped = line.trim();
    if stripped.is_empty() {
        return false;
    }
    let chars: Vec<char> = stripped.chars().collect();
    let mut pos = 0usize;

    if starts_with_ascii_at(stripped.as_bytes(), 0, b"part") {
        pos = 4;
        if pos >= chars.len() || !chars[pos].is_whitespace() {
            return false;
        }
        while pos < chars.len() && chars[pos].is_whitespace() {
            pos += 1;
        }
        let roman_start = pos;
        while pos < chars.len() && is_roman_char(chars[pos]) {
            pos += 1;
        }
        if pos == roman_start {
            return false;
        }
        while pos < chars.len() && chars[pos].is_whitespace() {
            pos += 1;
        }
    }

    let remaining: String = chars[pos..].iter().collect();
    let bytes = remaining.as_bytes();
    if !starts_with_ascii_at(bytes, 0, b"item") {
        return false;
    }
    let mut cursor = 4usize;
    if cursor >= bytes.len() || !bytes[cursor].is_ascii_whitespace() {
        return false;
    }
    while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
        cursor += 1;
    }
    let digit_start = cursor;
    while cursor < bytes.len() && bytes[cursor].is_ascii_digit() {
        cursor += 1;
    }
    if cursor == digit_start {
        return false;
    }
    if cursor < bytes.len() && bytes[cursor].is_ascii_alphabetic() {
        cursor += 1;
    }
    if cursor < bytes.len() && bytes[cursor].is_ascii_alphanumeric() {
        return false;
    }

    if stripped.ends_with("..") {
        return true;
    }
    let trimmed_end = stripped.trim_end();
    let end_bytes = trimmed_end.as_bytes();
    let mut end_pos = end_bytes.len();
    while end_pos > 0 && end_bytes[end_pos - 1].is_ascii_digit() {
        end_pos -= 1;
    }
    let digit_count = end_bytes.len() - end_pos;
    if (1..=4).contains(&digit_count) && end_pos > 0 && end_bytes[end_pos - 1].is_ascii_whitespace()
    {
        return true;
    }
    let trailing_ws_count = stripped
        .chars()
        .rev()
        .take_while(|ch| ch.is_whitespace())
        .count();
    trailing_ws_count >= 2 || stripped.ends_with('\t')
}

pub(crate) fn ascii_word_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '_'
}

pub(crate) fn ends_with_ascii_word_suffix_ci(normalized: &str, suffix_lower: &str) -> bool {
    let lowered = normalized.to_ascii_lowercase();
    if !lowered.ends_with(suffix_lower) {
        return false;
    }
    let start = lowered.len() - suffix_lower.len();
    if start == 0 {
        return true;
    }
    lowered
        .get(..start)
        .and_then(|prefix| prefix.chars().last())
        .is_none_or(|ch| !ascii_word_char(ch))
}

pub(crate) fn sentence_ends_with_reference_stub_impl(text: &str) -> bool {
    let normalized = normalize_sentence_key_impl(text);
    [
        "sfas no.", "sab no.", "fin no.", "fasb no.", "asc no.", "eitf no.",
    ]
    .iter()
    .any(|suffix| ends_with_ascii_word_suffix_ci(&normalized, suffix))
}

pub(crate) fn sentence_ends_with_generic_reference_no_impl(text: &str) -> bool {
    ends_with_ascii_word_suffix_ci(&normalize_sentence_key_impl(text), "no.")
}

pub(crate) fn starts_with_accounting_no_reference(lowered: &str) -> bool {
    let mut start = 0usize;
    if lowered.starts_with("fsp ") {
        start = 4;
    }
    for acronym in ["sfas", "sab", "fin", "fasb", "asc", "eitf"] {
        let end = start + acronym.len();
        if lowered.get(start..end) != Some(acronym) {
            continue;
        }
        let Some(rest) = lowered.get(end..) else {
            continue;
        };
        if rest.starts_with(' ') && rest.trim_start().starts_with("no.") {
            return true;
        }
    }
    false
}

pub(crate) fn starts_with_named_no_reference(lowered: &str) -> bool {
    for prefix in ["statement", "opinion", "interpretation", "position"] {
        let Some(rest) = lowered.strip_prefix(prefix) else {
            continue;
        };
        if rest.starts_with(' ') && rest.trim_start().starts_with("no.") {
            return true;
        }
    }
    false
}

pub(crate) fn digit_citation_continuation_start(normalized: &str, v3: bool) -> bool {
    let chars: Vec<char> = normalized.chars().collect();
    let mut pos = 0usize;
    if pos >= chars.len() || !chars[pos].is_ascii_digit() {
        return false;
    }
    while pos < chars.len() && chars[pos].is_ascii_digit() {
        pos += 1;
    }
    if pos >= chars.len() {
        return false;
    }
    if chars[pos].is_ascii_alphabetic() || matches!(chars[pos], ',' | '.') {
        return true;
    }
    if v3 && (chars[pos] == ')' || chars[pos].is_whitespace()) {
        return true;
    }
    if chars[pos] == '-' {
        return pos + 1 < chars.len() && chars[pos + 1].is_ascii_digit();
    }
    let mut probe = pos;
    skip_spaces(&chars, &mut probe);
    if probe < chars.len() && chars[probe] == '(' {
        probe += 1;
        let start = probe;
        while probe < chars.len() && chars[probe].is_ascii_alphabetic() {
            probe += 1;
        }
        return probe > start && probe < chars.len() && chars[probe] == ')';
    }
    false
}

pub(crate) fn sentence_looks_like_citation_continuation_impl(text: &str, v3: bool) -> bool {
    let normalized = normalize_sentence_key_impl(text);
    let lowered = normalized.to_ascii_lowercase();
    starts_with_accounting_no_reference(&lowered)
        || starts_with_named_no_reference(&lowered)
        || digit_citation_continuation_start(&normalized, v3)
}

pub(crate) fn sentence_generic_no_with_continuation_impl(text: &str, next_text: &str) -> bool {
    sentence_ends_with_generic_reference_no_impl(text)
        && sentence_looks_like_citation_continuation_impl(next_text, true)
}

pub(crate) fn consume_optional_spaces(chars: &[char], pos: &mut usize) {
    while *pos < chars.len() && chars[*pos].is_whitespace() {
        *pos += 1;
    }
}

pub(crate) fn consume_digits(chars: &[char], pos: &mut usize) -> bool {
    let start = *pos;
    while *pos < chars.len() && chars[*pos].is_ascii_digit() {
        *pos += 1;
    }
    *pos > start
}

pub(crate) fn consume_single_letter_parens(chars: &[char], pos: &mut usize) -> bool {
    if *pos >= chars.len() || chars[*pos] != '(' {
        return false;
    }
    *pos += 1;
    if *pos >= chars.len() || !chars[*pos].is_ascii_alphabetic() {
        return false;
    }
    *pos += 1;
    if *pos >= chars.len() || chars[*pos] != ')' {
        return false;
    }
    *pos += 1;
    true
}

pub(crate) fn consume_optional_dash_digits(chars: &[char], pos: &mut usize) -> bool {
    if *pos >= chars.len() || chars[*pos] != '-' {
        return true;
    }
    *pos += 1;
    consume_digits(chars, pos)
}

pub(crate) fn citation_prefix_tail_ok(chars: &[char], mut pos: usize) -> bool {
    consume_optional_spaces(chars, &mut pos);
    while pos < chars.len() && matches!(chars[pos], ')' | ',' | '.') {
        pos += 1;
    }
    consume_optional_spaces(chars, &mut pos);
    if pos < chars.len() && chars[pos] == ')' {
        pos += 1;
    }
    consume_optional_spaces(chars, &mut pos);
    pos == chars.len()
}

pub(crate) fn sentence_is_citation_prefix_only_line_impl(line: &str) -> bool {
    let stripped = line.trim();
    if stripped.is_empty() {
        return false;
    }
    let chars: Vec<char> = stripped.chars().collect();
    let mut pos = 0usize;
    if !consume_digits(&chars, &mut pos) {
        return false;
    }
    let after_digits = pos;

    if pos < chars.len() && chars[pos].is_ascii_alphabetic() {
        pos += 1;
        let mut probe = pos;
        consume_optional_spaces(&chars, &mut probe);
        if probe < chars.len() && chars[probe] == '(' {
            if !consume_single_letter_parens(&chars, &mut probe) {
                return false;
            }
            pos = probe;
        }
        if consume_optional_dash_digits(&chars, &mut pos) && citation_prefix_tail_ok(&chars, pos) {
            return true;
        }
    }

    pos = after_digits;
    consume_optional_spaces(&chars, &mut pos);
    if pos < chars.len() && chars[pos] == '(' {
        let mut probe = pos;
        if consume_single_letter_parens(&chars, &mut probe)
            && consume_optional_dash_digits(&chars, &mut probe)
            && citation_prefix_tail_ok(&chars, probe)
        {
            return true;
        }
    }

    pos = after_digits;
    if pos < chars.len() && chars[pos] == '-' {
        pos += 1;
        if consume_digits(&chars, &mut pos) && citation_prefix_tail_ok(&chars, pos) {
            return true;
        }
    }
    false
}

pub(crate) fn contains_ascii_word_or_phrase(lowered: &str, needle: &str) -> bool {
    let haystack = lowered.as_bytes();
    let needle_bytes = needle.as_bytes();
    if haystack.len() < needle_bytes.len() {
        return false;
    }
    for pos in 0..=haystack.len() - needle_bytes.len() {
        if &haystack[pos..pos + needle_bytes.len()] != needle_bytes {
            continue;
        }
        let left_ok = pos == 0 || !ascii_word_byte(haystack[pos - 1]);
        let right_pos = pos + needle_bytes.len();
        let right_ok = right_pos == haystack.len() || !ascii_word_byte(haystack[right_pos]);
        if left_ok && right_ok {
            return true;
        }
    }
    false
}

pub(crate) fn contains_sentence_header_keyword(stripped: &str) -> bool {
    let lowered = stripped.to_ascii_lowercase();
    [
        "consolidated",
        "statement",
        "statements",
        "balance sheet",
        "balance sheets",
        "cash flow",
        "cash flows",
        "operations",
        "changes in",
        "reportable segments",
        "and subsidiaries",
        "payments due by period",
    ]
    .iter()
    .any(|needle| contains_ascii_word_or_phrase(&lowered, needle))
}

pub(crate) fn sentence_unit_header_line_impl(stripped: &str) -> bool {
    let lowered = stripped.to_ascii_lowercase();
    let inner = lowered
        .strip_prefix('(')
        .and_then(|value| value.strip_suffix(')'))
        .unwrap_or(&lowered);
    matches!(inner, "dollars in thousands" | "dollars in millions")
}

pub(crate) fn note_or_item_continued_line_impl(stripped: &str) -> bool {
    let lowered = stripped.to_ascii_lowercase();
    if !(lowered.starts_with("note") || lowered.starts_with("item")) {
        return false;
    }
    let word_end = if lowered.starts_with("note") { 4 } else { 4 };
    if lowered
        .as_bytes()
        .get(word_end)
        .is_some_and(|byte| ascii_word_byte(*byte))
    {
        return false;
    }
    lowered.ends_with("(continued)")
        || lowered.ends_with("(continued).")
        || lowered.ends_with("(unaudited)")
        || lowered.ends_with("(unaudited).")
}

pub(crate) fn sentence_is_header_like_line_impl(line: &str) -> bool {
    let stripped = line.trim();
    if stripped.is_empty() {
        return true;
    }
    if sentence_is_citation_prefix_only_line_impl(stripped) {
        return false;
    }
    if sentence_is_separator_line_impl(stripped) {
        return true;
    }
    if sentence_unit_header_line_impl(stripped) {
        return true;
    }
    if note_or_item_continued_line_impl(stripped) {
        return true;
    }
    if contains_sentence_header_keyword(stripped) && stripped.to_uppercase() == stripped {
        return true;
    }
    let letters: Vec<char> = stripped.chars().filter(|ch| ch.is_alphabetic()).collect();
    if letters.is_empty() {
        return false;
    }
    if letters.iter().any(|ch| ch.is_lowercase()) {
        return false;
    }
    if stripped.chars().count() > 140 {
        return false;
    }
    true
}

pub(crate) fn strip_leading_sentence_artifact_lines_impl(text: &str) -> String {
    let normalized_text = text.replace("\r\n", "\n").replace('\r', "\n");
    let lines: Vec<String> = normalized_text
        .lines()
        .map(|line| line.trim().to_string())
        .collect();
    if lines.is_empty() {
        return String::new();
    }
    let mut kept_start = 0usize;
    while kept_start < lines.len() && sentence_is_header_like_line_impl(&lines[kept_start]) {
        kept_start += 1;
    }
    if kept_start >= lines.len() {
        return String::new();
    }
    lines[kept_start..]
        .iter()
        .filter(|line| !line.is_empty())
        .cloned()
        .collect::<Vec<_>>()
        .join("\n")
}

pub(crate) fn sentence_is_artifact_only_impl(text: &str) -> bool {
    let normalized_text = text.replace("\r\n", "\n").replace('\r', "\n");
    let lines: Vec<&str> = normalized_text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();
    if lines.is_empty() {
        return true;
    }
    lines
        .iter()
        .all(|line| sentence_is_header_like_line_impl(line))
}

pub(crate) fn join_sentence_fragments_impl(current: &str, next_text: &str) -> String {
    let left = current.trim_end();
    let right = next_text.trim_start();
    if left.is_empty() {
        return right.to_string();
    }
    if right.is_empty() {
        return left.to_string();
    }
    if left.ends_with('-') && right.chars().next().is_some_and(|ch| ch.is_lowercase()) {
        return format!("{left}{right}");
    }
    format!("{left} {right}")
}

pub(crate) fn should_stitch_item7_sentence_impl(current: &str, next_text: &str) -> bool {
    let next_clean = next_text.trim();
    if next_clean.is_empty() {
        return false;
    }
    if sentence_ends_with_reference_stub_impl(current) {
        return true;
    }
    current.trim_end().ends_with('-')
        && next_clean
            .chars()
            .next()
            .is_some_and(|ch| ch.is_lowercase())
}

pub(crate) fn should_stitch_item7_sentence_v2_impl(current: &str, next_text: &str) -> bool {
    let next_clean = next_text.trim();
    if next_clean.is_empty() {
        return false;
    }
    if should_stitch_item7_sentence_impl(current, next_clean) {
        return true;
    }
    sentence_ends_with_generic_reference_no_impl(current)
        && sentence_looks_like_citation_continuation_impl(next_clean, false)
}

pub(crate) fn should_stitch_reference_sentence_v3_impl(current: &str, next_text: &str) -> bool {
    let next_clean = next_text.trim();
    if next_clean.is_empty() {
        return false;
    }
    if should_stitch_item7_sentence_impl(current, next_clean) {
        return true;
    }
    sentence_ends_with_generic_reference_no_impl(current)
        && sentence_looks_like_citation_continuation_impl(next_clean, true)
}

pub(crate) fn sentence_policy_artifact_cleanup(policy: &str, text_scope: Option<&str>) -> bool {
    text_scope == Some("item_7_mda")
        && matches!(
            policy,
            "item7_reference_stitch_protect_v1"
                | "item7_reference_stitch_protect_v2"
                | "reference_stitch_protect_v3"
        )
}

pub(crate) fn sentence_policy_reference_stitching(policy: &str, text_scope: Option<&str>) -> bool {
    match policy {
        "item7_reference_stitch_protect_v1" | "item7_reference_stitch_protect_v2" => {
            text_scope == Some("item_7_mda")
        }
        "reference_stitch_protect_v3" => matches!(
            text_scope,
            Some("item_1_business" | "item_1a_risk_factors" | "item_7_mda")
        ),
        _ => false,
    }
}

pub(crate) fn should_stitch_for_sentence_policy(
    policy: &str,
    current: &str,
    next_text: &str,
) -> bool {
    match policy {
        "item7_reference_stitch_protect_v1" => {
            should_stitch_item7_sentence_impl(current, next_text)
        }
        "item7_reference_stitch_protect_v2" => {
            should_stitch_item7_sentence_v2_impl(current, next_text)
        }
        "reference_stitch_protect_v3" => {
            should_stitch_reference_sentence_v3_impl(current, next_text)
        }
        _ => false,
    }
}

pub(crate) fn postprocess_sentence_texts_impl(
    sentence_texts: Vec<String>,
    text_scope: Option<&str>,
    policy: &str,
) -> Vec<String> {
    if policy == "none" {
        return sentence_texts;
    }
    let artifact_cleanup = sentence_policy_artifact_cleanup(policy, text_scope);
    let reference_stitching = sentence_policy_reference_stitching(policy, text_scope);
    if !artifact_cleanup && !reference_stitching {
        return sentence_texts;
    }

    let normalized_texts: Vec<String> = if artifact_cleanup {
        sentence_texts
            .iter()
            .map(|text| strip_leading_sentence_artifact_lines_impl(text))
            .collect()
    } else {
        sentence_texts
    };
    let mut processed: Vec<String> = Vec::with_capacity(normalized_texts.len());
    let mut idx = 0usize;
    while idx < normalized_texts.len() {
        let mut current = normalized_texts[idx].trim().to_string();
        if current.is_empty() {
            idx += 1;
            continue;
        }
        let mut next_idx = idx + 1;
        while next_idx < normalized_texts.len() {
            while next_idx < normalized_texts.len() && normalized_texts[next_idx].trim().is_empty()
            {
                next_idx += 1;
            }
            if next_idx >= normalized_texts.len() {
                break;
            }
            let next_text = normalized_texts[next_idx].trim();
            if !reference_stitching
                || !should_stitch_for_sentence_policy(policy, &current, next_text)
            {
                break;
            }
            current = join_sentence_fragments_impl(&current, next_text);
            idx = next_idx;
            next_idx = idx + 1;
        }
        if artifact_cleanup && sentence_is_artifact_only_impl(&current) {
            idx += 1;
            continue;
        }
        if !current.is_empty() {
            processed.push(current);
        }
        idx += 1;
    }
    processed
}

pub(crate) fn skip_ascii_ws(bytes: &[u8], pos: &mut usize) {
    while *pos < bytes.len() && bytes[*pos].is_ascii_whitespace() {
        *pos += 1;
    }
}

pub(crate) fn consume_ascii_digits_1_to_4(bytes: &[u8], pos: &mut usize) -> bool {
    let start = *pos;
    while *pos < bytes.len() && bytes[*pos].is_ascii_digit() && *pos - start < 4 {
        *pos += 1;
    }
    *pos > start && *pos - start <= 4
}

pub(crate) fn starts_with_ascii_ci(bytes: &[u8], pos: usize, needle: &[u8]) -> bool {
    pos + needle.len() <= bytes.len()
        && bytes[pos..pos + needle.len()]
            .iter()
            .zip(needle.iter())
            .all(|(actual, expected)| actual.to_ascii_lowercase() == *expected)
}

pub(crate) fn item_heading_start_impl(stripped: &str) -> bool {
    let bytes = stripped.as_bytes();
    let mut pos = 0usize;
    skip_ascii_ws(bytes, &mut pos);
    if !starts_with_ascii_ci(bytes, pos, b"item") {
        return false;
    }
    pos += 4;
    if pos >= bytes.len() || !bytes[pos].is_ascii_whitespace() {
        return false;
    }
    skip_ascii_ws(bytes, &mut pos);
    if !consume_ascii_digits_1_to_4(bytes, &mut pos) {
        return false;
    }
    if pos < bytes.len() && bytes[pos].is_ascii_alphabetic() {
        pos += 1;
    }
    pos >= bytes.len() || !bytes[pos].is_ascii_alphanumeric()
}

pub(crate) fn page_marker_dash_wrapped_impl(stripped: &str) -> bool {
    let Some(inner) = stripped
        .strip_prefix('-')
        .and_then(|value| value.strip_suffix('-'))
    else {
        return false;
    };
    let inner = inner.trim();
    !inner.is_empty()
        && inner.len() <= 4
        && inner.as_bytes().iter().all(|byte| byte.is_ascii_digit())
}

pub(crate) fn page_marker_general_impl(stripped: &str) -> bool {
    let bytes = stripped.as_bytes();
    let mut pos = 0usize;
    if starts_with_ascii_ci(bytes, pos, b"page") {
        pos += 4;
        if pos >= bytes.len() || !bytes[pos].is_ascii_whitespace() {
            return false;
        }
        skip_ascii_ws(bytes, &mut pos);
    }
    if pos < bytes.len() && matches!(bytes[pos], b'-' | b'(' | b'[') {
        pos += 1;
    }
    if !consume_ascii_digits_1_to_4(bytes, &mut pos) {
        return false;
    }

    let after_digits = pos;
    skip_ascii_ws(bytes, &mut pos);
    if pos > after_digits {
        if !starts_with_ascii_ci(bytes, pos, b"of") {
            return false;
        }
        pos += 2;
        if pos >= bytes.len() || !bytes[pos].is_ascii_whitespace() {
            return false;
        }
        skip_ascii_ws(bytes, &mut pos);
        if !consume_ascii_digits_1_to_4(bytes, &mut pos) {
            return false;
        }
    }
    if pos < bytes.len() && matches!(bytes[pos], b')' | b']' | b'-') {
        pos += 1;
    }
    pos == bytes.len()
}

pub(crate) fn cleaning_is_page_marker_line_impl(line: &str) -> bool {
    let stripped = line.trim();
    !stripped.is_empty()
        && !item_heading_start_impl(stripped)
        && (page_marker_dash_wrapped_impl(stripped) || page_marker_general_impl(stripped))
}

pub(crate) fn normalized_ascii_header_tokens(line: &str) -> Vec<String> {
    line.trim()
        .split_whitespace()
        .map(|token| token.to_ascii_lowercase())
        .collect()
}

pub(crate) fn cleaning_is_report_header_footer_line_impl(line: &str) -> bool {
    let tokens = normalized_ascii_header_tokens(line);
    let expected_prefix: &[&str] = if tokens.first().is_some_and(|token| token == "annual") {
        &["annual", "report", "on", "form"]
    } else {
        &["report", "on", "form"]
    };
    if tokens.len() <= expected_prefix.len() {
        return false;
    }
    if !tokens
        .iter()
        .zip(expected_prefix.iter())
        .all(|(actual, expected)| actual == expected)
    {
        return false;
    }
    let rest = tokens[expected_prefix.len()..].join("");
    rest == "10k" || rest == "10-k"
}

pub(crate) fn cleaning_is_structural_residue_line_impl(line: &str) -> bool {
    let stripped = line.trim();
    if stripped.is_empty() {
        return false;
    }
    if stripped.starts_with('<') && stripped.ends_with('>') {
        let inner = &stripped[1..stripped.len() - 1];
        if !inner.is_empty() && inner.len() <= 120 && !inner.contains('>') {
            return true;
        }
    }
    let Some((stem, extension)) = stripped.rsplit_once('.') else {
        return false;
    };
    if stem.is_empty()
        || !stem
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.' | b'-'))
    {
        return false;
    }
    matches!(
        extension.to_ascii_lowercase().as_str(),
        "xml" | "xsd" | "xbrl" | "htm" | "html"
    )
}

pub(crate) fn cleaning_remove_layout_lines_impl(
    text: &str,
    drop_page_markers: bool,
    drop_report_headers: bool,
    drop_structural_tags: bool,
) -> (String, BTreeMap<String, i64>) {
    let mut kept_lines: Vec<String> = Vec::new();
    let mut counts: BTreeMap<String, i64> = BTreeMap::new();
    for line in text.split('\n') {
        let stripped = line.trim();
        if drop_page_markers && cleaning_is_page_marker_line_impl(stripped) {
            *counts.entry("page_marker".to_string()).or_insert(0) += 1;
            continue;
        }
        if drop_report_headers && cleaning_is_report_header_footer_line_impl(stripped) {
            *counts
                .entry("report_header_footer".to_string())
                .or_insert(0) += 1;
            continue;
        }
        if drop_structural_tags && cleaning_is_structural_residue_line_impl(stripped) {
            *counts.entry("structural_tag".to_string()).or_insert(0) += 1;
            continue;
        }
        kept_lines.push(line.trim_end().to_string());
    }
    (kept_lines.join("\n"), counts)
}

pub(crate) struct CleaningTableBlockLine<'a> {
    raw_line: &'a str,
    is_table_like: bool,
    is_header_like: bool,
    is_intro_like: bool,
    is_strong_title: bool,
    is_support_like: bool,
}

pub(crate) fn flush_cleaning_table_block(
    block_lines: &mut Vec<CleaningTableBlockLine<'_>>,
    kept_lines: &mut Vec<String>,
    counts: &mut BTreeMap<String, i64>,
    table_like_min_consecutive_lines: usize,
    table_like_allow_single_line_with_header: bool,
    table_like_drop_header_context: bool,
) {
    if block_lines.is_empty() {
        return;
    }
    let table_like_count = block_lines.iter().filter(|line| line.is_table_like).count();
    let header_like_count = block_lines
        .iter()
        .filter(|line| line.is_header_like)
        .count();
    let strong_title_count = block_lines
        .iter()
        .filter(|line| line.is_strong_title)
        .count();
    let intro_like_count = block_lines.iter().filter(|line| line.is_intro_like).count();
    let support_like_count = block_lines
        .iter()
        .filter(|line| line.is_support_like)
        .count();
    let unit_like_count = block_lines
        .iter()
        .filter(|line| cleaning_is_table_unit_line_impl(line.raw_line.trim()))
        .count();

    let mut drop_block = table_like_count >= table_like_min_consecutive_lines;
    if !drop_block
        && table_like_allow_single_line_with_header
        && header_like_count > 0
        && table_like_count >= 1
    {
        drop_block = true;
    }
    if !drop_block
        && table_like_drop_header_context
        && table_like_count == 0
        && block_lines.len() >= 2
    {
        if strong_title_count > 0 && support_like_count > 0 {
            drop_block = true;
        } else if intro_like_count > 0
            && unit_like_count > 0
            && support_like_count > unit_like_count
        {
            drop_block = true;
        }
    }

    for line in block_lines.iter() {
        if drop_block
            && (line.is_table_like
                || (table_like_drop_header_context
                    && (line.is_header_like
                        || line.is_intro_like
                        || line.is_support_like
                        || line.is_strong_title)))
        {
            *counts.entry("table_like".to_string()).or_insert(0) += 1;
            continue;
        }
        kept_lines.push(line.raw_line.trim_end().to_string());
    }
    block_lines.clear();
}

pub(crate) fn cleaning_remove_table_like_lines_impl(
    text: &str,
    table_like_min_consecutive_lines: usize,
    table_like_allow_single_line_with_header: bool,
    table_like_drop_header_context: bool,
) -> (String, BTreeMap<String, i64>) {
    let mut kept_lines: Vec<String> = Vec::new();
    let mut counts: BTreeMap<String, i64> = BTreeMap::new();
    let mut block_lines: Vec<CleaningTableBlockLine<'_>> = Vec::new();

    for line in text.split('\n') {
        let stripped = line.trim();
        let is_table_like = cleaning_is_table_like_line_impl(stripped);
        let is_header_like = cleaning_is_table_header_like_line_impl(stripped);
        let is_intro_like = cleaning_is_table_intro_line_impl(stripped);
        let is_strong_title = cleaning_is_strong_table_title_line_impl(stripped);
        let is_support_like = cleaning_is_table_support_header_line_impl(stripped);
        if is_table_like || is_header_like || is_intro_like || is_support_like {
            block_lines.push(CleaningTableBlockLine {
                raw_line: line,
                is_table_like,
                is_header_like,
                is_intro_like,
                is_strong_title,
                is_support_like,
            });
            continue;
        }
        flush_cleaning_table_block(
            &mut block_lines,
            &mut kept_lines,
            &mut counts,
            table_like_min_consecutive_lines,
            table_like_allow_single_line_with_header,
            table_like_drop_header_context,
        );
        kept_lines.push(line.trim_end().to_string());
    }
    flush_cleaning_table_block(
        &mut block_lines,
        &mut kept_lines,
        &mut counts,
        table_like_min_consecutive_lines,
        table_like_allow_single_line_with_header,
        table_like_drop_header_context,
    );
    (kept_lines.join("\n"), counts)
}

pub(crate) fn is_ascii_letter(byte: u8) -> bool {
    byte.is_ascii_alphabetic()
}

pub(crate) fn casefolded_token(bytes: &[u8], start: usize, end: usize) -> Option<String> {
    if start >= end {
        return None;
    }
    Some(
        bytes[start..end]
            .iter()
            .map(|byte| byte.to_ascii_lowercase() as char)
            .collect::<String>(),
    )
}

pub(crate) fn next_repaired_byte(bytes: &[u8], pos: usize) -> Option<(u8, usize)> {
    if pos >= bytes.len() {
        return None;
    }
    let current = bytes[pos];
    if current != b'-' || pos == 0 || !is_ascii_letter(bytes[pos - 1]) {
        return Some((current, pos + 1));
    }

    let mut probe = pos + 1;
    while probe < bytes.len() && matches!(bytes[probe], b' ' | b'\t' | 0x0b | 0x0c) {
        probe += 1;
    }
    if probe < bytes.len() && bytes[probe] == b'\r' {
        probe += 1;
    }
    if probe >= bytes.len() || bytes[probe] != b'\n' {
        return Some((current, pos + 1));
    }
    probe += 1;
    while probe < bytes.len() && bytes[probe].is_ascii_whitespace() {
        probe += 1;
    }
    if probe < bytes.len() && is_ascii_letter(bytes[probe]) {
        return next_repaired_byte(bytes, probe);
    }
    Some((current, pos + 1))
}

pub(crate) fn repair_linebreak_hyphenation(text: &str) -> Vec<u8> {
    let source = text.as_bytes();
    let mut repaired = Vec::with_capacity(source.len());
    let mut pos = 0;
    while let Some((byte, next_pos)) = next_repaired_byte(source, pos) {
        let normalized = if byte == b'\r' { b'\n' } else { byte };
        repaired.push(normalized);
        pos = next_pos;
    }
    repaired
}

pub(crate) fn scan_tokens_impl<F>(text: Option<&str>, mut visit: F)
where
    F: FnMut(String),
{
    let Some(text) = text else {
        return;
    };
    let bytes = repair_linebreak_hyphenation(text);
    let mut pos = 0;

    while pos < bytes.len() {
        if !is_ascii_letter(bytes[pos]) {
            pos += 1;
            continue;
        }

        let start = pos;
        let mut letter_count = 0;
        while pos < bytes.len() && is_ascii_letter(bytes[pos]) {
            pos += 1;
            letter_count += 1;
        }
        if letter_count < 2 {
            pos = start + 1;
            continue;
        }

        let mut end = pos;
        loop {
            if pos >= bytes.len() || !matches!(bytes[pos], b'-' | b'\'') {
                break;
            }
            let separator = pos;
            pos += 1;
            let segment_start = pos;
            while pos < bytes.len() && is_ascii_letter(bytes[pos]) {
                pos += 1;
            }
            if pos == segment_start {
                pos = separator;
                break;
            }
            end = pos;
        }

        if let Some(token) = casefolded_token(&bytes, start, end) {
            visit(token);
        }
        if pos == end {
            continue;
        }
        pos += 1;
    }
}

pub(crate) fn tokenize_impl(text: Option<&str>) -> Vec<String> {
    let mut tokens = Vec::new();
    scan_tokens_impl(text, |token| tokens.push(token));
    tokens
}

pub(crate) fn count_tokens_impl(text: Option<&str>) -> usize {
    let Some(text) = text else {
        return 0;
    };
    let bytes = repair_linebreak_hyphenation(text);
    let mut pos = 0;
    let mut count = 0usize;

    while pos < bytes.len() {
        if !is_ascii_letter(bytes[pos]) {
            pos += 1;
            continue;
        }

        let start = pos;
        let mut letter_count = 0;
        while pos < bytes.len() && is_ascii_letter(bytes[pos]) {
            pos += 1;
            letter_count += 1;
        }
        if letter_count < 2 {
            pos = start + 1;
            continue;
        }

        let mut end = pos;
        loop {
            if pos >= bytes.len() || !matches!(bytes[pos], b'-' | b'\'') {
                break;
            }
            let separator = pos;
            pos += 1;
            let segment_start = pos;
            while pos < bytes.len() && is_ascii_letter(bytes[pos]) {
                pos += 1;
            }
            if pos == segment_start {
                pos = separator;
                break;
            }
            end = pos;
        }

        if start < end {
            count += 1;
        }
        if pos == end {
            continue;
        }
        pos += 1;
    }

    count
}

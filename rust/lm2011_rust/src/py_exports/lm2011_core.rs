// LM2011 tokenization, dictionaries, form normalization, and text-feature entrypoints.
use crate::*;
use pyo3::wrap_pyfunction;

pub(crate) fn register(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    m.add_function(wrap_pyfunction!(tokenize_lm2011_text, m)?)?;
    m.add_function(wrap_pyfunction!(count_lm2011_text_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(count_lm2011_text_token_values, m)?)?;
    m.add_function(wrap_pyfunction!(
        validate_lm2011_incremental_text_doc_ids,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(normalize_lm2011_dictionary_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(build_lm2011_feature_rows, m)?)?;
    m.add_function(wrap_pyfunction!(build_lm2011_feature_rows_from_pass1, m)?)?;
    m.add_function(wrap_pyfunction!(
        build_lm2011_feature_rows_from_pass1_columns,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(prepare_lm2011_document_stats, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_lm2011_document_stats_columns, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_lm2011_pass1_rows, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_lm2011_pass1_columns, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_lm2011_form_value, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_lm2011_form_values, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_sec_raw_form_value, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_sec_raw_form_values, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_ccm_raw_form_value, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_ccm_raw_form_values, m)?)?;
    m.add_function(wrap_pyfunction!(ccm_form_match_token_value, m)?)?;
    m.add_function(wrap_pyfunction!(parse_ff48_sic_mapping_rows, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_year_filter_values, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_lookup_text_value, m)?)?;
    m.add_function(wrap_pyfunction!(lm2011_dictionary_normalize_cell_value, m)?)?;
    m.add_function(wrap_pyfunction!(
        lm2011_dictionary_active_membership_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        lm2011_dictionary_unique_preserve_order,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(finite_float_or_none_value, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_lookup_text_any_value, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_lookup_text_any_values, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_workbook_scalar_value, m)?)?;
    m.add_class::<Lm2011TokenCounter>()?;
    Ok(())
}

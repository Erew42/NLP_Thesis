// General text normalization, sentence quality, and item-cleaning entrypoints.
use crate::*;
use pyo3::wrap_pyfunction;

pub(crate) fn register(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    m.add_function(wrap_pyfunction!(refinitiv_bridge_date_to_text_value, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_spaces_value, m)?)?;
    m.add_function(wrap_pyfunction!(extension_csv_json_dumps_values, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_newlines_value, m)?)?;
    m.add_function(wrap_pyfunction!(collapse_blank_runs_value, m)?)?;
    m.add_function(wrap_pyfunction!(plan_text_microbatch_spans, m)?)?;
    m.add_function(wrap_pyfunction!(choose_sentence_chunk_end, m)?)?;
    m.add_function(wrap_pyfunction!(expand_sentence_chunk_rows, m)?)?;
    m.add_function(wrap_pyfunction!(trim_early_toc_prefix_value, m)?)?;
    m.add_function(wrap_pyfunction!(multisurface_boundary_snippet_risk, m)?)?;
    m.add_function(wrap_pyfunction!(
        multisurface_boundary_snippet_risk_values,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(multisurface_snippet_delta_risk_values, m)?)?;
    m.add_function(wrap_pyfunction!(multisurface_chunk_record_indices, m)?)?;
    m.add_function(wrap_pyfunction!(
        multisurface_chunk_record_indices_columns,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(multisurface_mark_escalated_cases, m)?)?;
    m.add_function(wrap_pyfunction!(stable_digits_int_value, m)?)?;
    m.add_function(wrap_pyfunction!(stable_digits_int_values, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_ascii_with_positions_value, m)?)?;
    m.add_function(wrap_pyfunction!(normalized_ascii_match_bounds_value, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_ascii_sample_text_value, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_ascii_sample_text_values, m)?)?;
    m.add_function(wrap_pyfunction!(sentence_length_ordered_item_codes, m)?)?;
    m.add_function(wrap_pyfunction!(sentence_length_frame_records, m)?)?;
    m.add_function(wrap_pyfunction!(sentence_length_frame_record_columns, m)?)?;
    m.add_function(wrap_pyfunction!(item_cleaning_clean_text_value, m)?)?;
    m.add_function(wrap_pyfunction!(item_cleaning_prepare_rows_value, m)?)?;
    m.add_function(wrap_pyfunction!(item_cleaning_base_row_payload_value, m)?)?;
    m.add_function(wrap_pyfunction!(item_cleaning_review_status_value, m)?)?;
    m.add_function(wrap_pyfunction!(
        item_cleaning_production_eligible_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(item_cleaning_finalize_rows, m)?)?;
    m.add_function(wrap_pyfunction!(item_cleaning_activation_status, m)?)?;
    m.add_function(wrap_pyfunction!(item_cleaning_activation_status_values, m)?)?;
    m.add_function(wrap_pyfunction!(item_cleaning_audit_period, m)?)?;
    m.add_function(wrap_pyfunction!(item_cleaning_audit_period_values, m)?)?;
    m.add_function(wrap_pyfunction!(item_cleaning_manual_audit_sample_rows, m)?)?;
    m.add_function(wrap_pyfunction!(item_cleaning_drop_reason_value, m)?)?;
    m.add_function(wrap_pyfunction!(
        item_cleaning_manual_audit_reason_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(lm2011_extension_text_scope_value, m)?)?;
    m.add_function(wrap_pyfunction!(sentence_is_separator_line, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_sentence_key_value, m)?)?;
    m.add_function(wrap_pyfunction!(sentence_numeric_only_fragment, m)?)?;
    m.add_function(wrap_pyfunction!(sentence_short_fragment, m)?)?;
    m.add_function(wrap_pyfunction!(sentence_very_short_fragment, m)?)?;
    m.add_function(wrap_pyfunction!(sentence_lower_fragment, m)?)?;
    m.add_function(wrap_pyfunction!(sentence_one_word_fragment, m)?)?;
    m.add_function(wrap_pyfunction!(sentence_has_terminal_punct, m)?)?;
    m.add_function(wrap_pyfunction!(sentence_table_like, m)?)?;
    m.add_function(wrap_pyfunction!(sentence_ends_with_reference_stub, m)?)?;
    m.add_function(wrap_pyfunction!(
        sentence_ends_with_generic_reference_no,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        sentence_looks_like_citation_continuation,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        sentence_looks_like_citation_continuation_v3,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(sentence_generic_no_with_continuation, m)?)?;
    m.add_function(wrap_pyfunction!(sentence_is_citation_prefix_only_line, m)?)?;
    m.add_function(wrap_pyfunction!(sentence_is_header_like_line, m)?)?;
    m.add_function(wrap_pyfunction!(sentence_quality_batch_flags, m)?)?;
    m.add_function(wrap_pyfunction!(postprocess_sentence_texts_value, m)?)?;
    m.add_function(wrap_pyfunction!(cleaning_is_page_marker_line, m)?)?;
    m.add_function(wrap_pyfunction!(cleaning_is_report_header_footer_line, m)?)?;
    m.add_function(wrap_pyfunction!(cleaning_is_structural_residue_line, m)?)?;
    m.add_function(wrap_pyfunction!(cleaning_remove_layout_lines_value, m)?)?;
    m.add_function(wrap_pyfunction!(cleaning_remove_table_like_lines_value, m)?)?;
    m.add_function(wrap_pyfunction!(cleaning_is_table_like_line, m)?)?;
    m.add_function(wrap_pyfunction!(cleaning_is_table_header_like_line, m)?)?;
    m.add_function(wrap_pyfunction!(cleaning_is_strong_table_title_line, m)?)?;
    m.add_function(wrap_pyfunction!(cleaning_is_table_intro_line, m)?)?;
    m.add_function(wrap_pyfunction!(cleaning_is_table_support_header_line, m)?)?;
    m.add_function(wrap_pyfunction!(cleaning_is_toc_like_line, m)?)?;
    m.add_function(wrap_pyfunction!(cleaning_tail_bleed_start, m)?)?;
    m.add_function(wrap_pyfunction!(cleaning_is_reference_only_stub, m)?)?;
    m.add_function(wrap_pyfunction!(detect_lm2011_exhibit_tail, m)?)?;
    Ok(())
}

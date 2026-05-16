// SEC extraction, filing identifiers, HTML audit, and regime helpers.
use crate::*;
use pyo3::wrap_pyfunction;

pub(crate) fn register(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    m.add_function(wrap_pyfunction!(sec_digits_only_value, m)?)?;
    m.add_function(wrap_pyfunction!(sec_cik_10_value, m)?)?;
    m.add_function(wrap_pyfunction!(sec_make_doc_id_value, m)?)?;
    m.add_function(wrap_pyfunction!(sec_parse_date_value, m)?)?;
    m.add_function(wrap_pyfunction!(sec_roman_to_int_value, m)?)?;
    m.add_function(wrap_pyfunction!(sec_default_part_for_item_id_value, m)?)?;
    m.add_function(wrap_pyfunction!(sec_prefix_is_bullet_value, m)?)?;
    m.add_function(wrap_pyfunction!(sec_prefix_looks_like_cross_ref, m)?)?;
    m.add_function(wrap_pyfunction!(sec_line_has_compound_items, m)?)?;
    m.add_function(wrap_pyfunction!(sec_heading_suffix_looks_like_prose, m)?)?;
    m.add_function(wrap_pyfunction!(sec_part_marker_is_heading, m)?)?;
    m.add_function(wrap_pyfunction!(sec_pageish_line, m)?)?;
    m.add_function(wrap_pyfunction!(sec_strip_edgar_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(sec_prefix_is_part_only, m)?)?;
    m.add_function(wrap_pyfunction!(sec_prefix_part_tail, m)?)?;
    m.add_function(wrap_pyfunction!(sec_scan_part_markers_v2, m)?)?;
    m.add_function(wrap_pyfunction!(sec_infer_toc_end_pos, m)?)?;
    m.add_function(wrap_pyfunction!(sec_remove_pagination, m)?)?;
    m.add_function(wrap_pyfunction!(sec_trim_trailing_part_marker, m)?)?;
    m.add_function(wrap_pyfunction!(sec_reserved_stub_end, m)?)?;
    m.add_function(wrap_pyfunction!(sec_line_start_item_match, m)?)?;
    m.add_function(wrap_pyfunction!(sec_detect_toc_line_ranges, m)?)?;
    m.add_function(wrap_pyfunction!(sec_detect_gij_context, m)?)?;
    m.add_function(wrap_pyfunction!(sec_find_embedded_heading_hits, m)?)?;
    m.add_function(wrap_pyfunction!(sec_apply_high_confidence_truncation, m)?)?;
    m.add_function(wrap_pyfunction!(parse_sec_filename_minimal_value, m)?)?;
    m.add_function(wrap_pyfunction!(
        classify_boundary_authority_status_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(public_boundary_payload_value, m)?)?;
    m.add_function(wrap_pyfunction!(html_audit_safe_slug_value, m)?)?;
    m.add_function(wrap_pyfunction!(html_audit_parse_bool_value, m)?)?;
    m.add_function(wrap_pyfunction!(html_audit_filing_status_value, m)?)?;
    m.add_function(wrap_pyfunction!(html_audit_parse_int_value, m)?)?;
    m.add_function(wrap_pyfunction!(html_audit_part_rank_value, m)?)?;
    m.add_function(wrap_pyfunction!(html_audit_item_id_sort_key_value, m)?)?;
    m.add_function(wrap_pyfunction!(html_audit_quartile_edges_value, m)?)?;
    m.add_function(wrap_pyfunction!(html_audit_quartile_bucket_value, m)?)?;
    m.add_function(wrap_pyfunction!(html_audit_sample_stratified_rows, m)?)?;
    m.add_function(wrap_pyfunction!(html_audit_sample_filings_by_status, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_regime_form_type, m)?)?;
    m.add_function(wrap_pyfunction!(
        benchmark_item_code_to_text_scope_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        benchmark_item_code_to_text_scope_values,
        m
    )?)?;
    Ok(())
}

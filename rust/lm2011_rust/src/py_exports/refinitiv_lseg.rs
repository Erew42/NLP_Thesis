// Refinitiv, LSEG, analyst, ownership, and workbook bridge entrypoints.
use crate::*;
use pyo3::wrap_pyfunction;

pub(crate) fn register(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    m.add_function(wrap_pyfunction!(
        refinitiv_excel_ownership_universe_block_payloads,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        refinitiv_excel_lm2011_doc_ownership_block_payloads,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        refinitiv_excel_resolution_diagnostic_retrieval_block_payloads,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        refinitiv_excel_ownership_smoke_block_payloads,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        refinitiv_excel_ownership_validation_sheet_payloads,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        refinitiv_excel_extended_summary_formula_payloads,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        refinitiv_excel_extended_lookup_formula_payloads,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        normalize_lookup_batch_response_rows_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        normalize_lookup_batch_response_columns,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(build_lookup_item_rows_value, m)?)?;
    m.add_function(wrap_pyfunction!(build_lookup_item_row_columns, m)?)?;
    m.add_function(wrap_pyfunction!(build_analyst_actual_item_rows_value, m)?)?;
    m.add_function(wrap_pyfunction!(build_analyst_actual_item_row_columns, m)?)?;
    m.add_function(wrap_pyfunction!(build_analyst_estimate_item_rows_value, m)?)?;
    m.add_function(wrap_pyfunction!(
        build_analyst_estimate_item_row_columns,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        normalize_analyst_actuals_batch_response_rows_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        normalize_analyst_actuals_batch_response_columns,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        normalize_analyst_estimates_batch_response_rows_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        normalize_analyst_estimates_batch_response_columns,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        build_ownership_universe_item_rows_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        build_ownership_universe_item_row_columns,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        build_doc_ownership_exact_item_rows_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        build_doc_ownership_exact_item_row_columns,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        build_doc_ownership_fallback_item_rows_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        build_doc_ownership_fallback_item_row_columns,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        normalize_ownership_universe_batch_response_rows_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        normalize_ownership_universe_batch_response_columns,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        normalize_doc_ownership_batch_response_rows_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        normalize_doc_ownership_batch_response_columns,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(refinitiv_authority_date_span_days, m)?)?;
    m.add_function(wrap_pyfunction!(refinitiv_authority_values_match, m)?)?;
    m.add_function(wrap_pyfunction!(refinitiv_authority_distinct_values, m)?)?;
    m.add_function(wrap_pyfunction!(refinitiv_authority_candidate_key, m)?)?;
    m.add_function(wrap_pyfunction!(refinitiv_authority_allowlist_keys, m)?)?;
    m.add_function(wrap_pyfunction!(
        refinitiv_authority_allowlist_key_columns,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        refinitiv_authority_candidate_metric_records,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        refinitiv_authority_candidate_metric_record_columns,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(refinitiv_authority_final_panel_rows, m)?)?;
    m.add_function(wrap_pyfunction!(
        refinitiv_authority_final_panel_rows_columns,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        refinitiv_authority_review_required_rows_columns,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        refinitiv_authority_source_family_from_flags,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(refinitiv_authority_component_id, m)?)?;
    m.add_function(wrap_pyfunction!(refinitiv_authority_merge_intervals, m)?)?;
    m.add_function(wrap_pyfunction!(
        refinitiv_authority_pairwise_alias_diagnostic_rows,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        refinitiv_authority_conventional_components,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        refinitiv_authority_conventional_components_from_meta,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(normalize_extended_workbook_bool_value, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_item_codes_optional_value, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_years_optional_value, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_stage_audit_optional_text, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_stage_audit_optional_int, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_stage_audit_optional_mapping, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_optional_int_value, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_stage_audit_string_list, m)?)?;
    m.add_function(wrap_pyfunction!(
        normalize_refinitiv_lookup_result_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        normalize_refinitiv_ownership_result_text,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        normalize_refinitiv_ownership_result_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        normalize_refinitiv_ownership_result_date,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        refinitiv_bridge_ownership_universe_request_date,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(normalize_lseg_month_boundaries, m)?)?;
    m.add_function(wrap_pyfunction!(lseg_item_window_dates, m)?)?;
    m.add_function(wrap_pyfunction!(lseg_interval_span_days, m)?)?;
    m.add_function(wrap_pyfunction!(lseg_evaluate_interval_candidate, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_analyst_request_group_id, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_analyst_request_group_ids, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_analyst_group_list_values, m)?)?;
    m.add_function(wrap_pyfunction!(
        refinitiv_analyst_canonicalize_actual_event,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        refinitiv_analyst_canonicalize_estimate_snapshot,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        refinitiv_analyst_freeze_sorted_date_index,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(stable_hash_id_simple, m)?)?;
    m.add_function(wrap_pyfunction!(lseg_request_signature_value, m)?)?;
    m.add_function(wrap_pyfunction!(lseg_batch_item_index_groups, m)?)?;
    m.add_function(wrap_pyfunction!(lseg_frame_fingerprint_rows, m)?)?;
    m.add_function(wrap_pyfunction!(lseg_frame_fingerprint_columns, m)?)?;
    m.add_function(wrap_pyfunction!(sample_scope_coverage_doc_ids, m)?)?;
    m.add_function(wrap_pyfunction!(lseg_count_request_log_events, m)?)?;
    m.add_function(wrap_pyfunction!(lseg_string_array_json_values, m)?)?;
    m.add_function(wrap_pyfunction!(lseg_ledger_item_ids_json_values, m)?)?;
    m.add_function(wrap_pyfunction!(lseg_row_count_by_item_id, m)?)?;
    m.add_function(wrap_pyfunction!(lseg_row_count_by_item_id_values, m)?)?;
    m.add_function(wrap_pyfunction!(
        lseg_should_requeue_mixed_zero_positive_success,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(lseg_singleton_child_batch_rows, m)?)?;
    m.add_function(wrap_pyfunction!(lseg_split_child_batch_rows, m)?)?;
    m.add_function(wrap_pyfunction!(lseg_split_batch_rows, m)?)?;
    m.add_function(wrap_pyfunction!(lseg_item_result_detail_rows, m)?)?;
    m.add_function(wrap_pyfunction!(classify_lseg_error_message_value, m)?)?;
    m.add_function(wrap_pyfunction!(is_lseg_session_not_opened_message, m)?)?;
    m.add_function(wrap_pyfunction!(parse_lseg_identifier_list_value, m)?)?;
    m.add_function(wrap_pyfunction!(
        parse_lseg_unresolved_identifiers_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(lseg_utf8_record_rows, m)?)?;
    m.add_function(wrap_pyfunction!(lseg_sanitize_headers, m)?)?;
    m.add_function(wrap_pyfunction!(is_lseg_overload_like, m)?)?;
    m.add_function(wrap_pyfunction!(lseg_daily_limit_likely_exhausted, m)?)?;
    m.add_function(wrap_pyfunction!(lseg_indicates_daily_limit, m)?)?;
    m.add_function(wrap_pyfunction!(lseg_should_treat_as_empty_result, m)?)?;
    m.add_function(wrap_pyfunction!(lseg_classify_batch_error_policy, m)?)?;
    m.add_function(wrap_pyfunction!(parse_bridge_row_id_liid, m)?)?;
    m.add_function(wrap_pyfunction!(
        refinitiv_bridge_identity_candidates_agree,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        refinitiv_bridge_candidates_materially_conflict,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        refinitiv_bridge_lookup_candidate_from_record,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        refinitiv_bridge_accepted_candidate_from_record,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        refinitiv_bridge_derive_accepted_resolution,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        refinitiv_bridge_effective_resolution_fields,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(refinitiv_bridge_resolution_frame_rows, m)?)?;
    m.add_function(wrap_pyfunction!(
        refinitiv_bridge_resolution_frame_rows_from_columns,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        refinitiv_bridge_resolution_diagnostic_handoff_row,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(refinitiv_bridge_failed_lookup_record, m)?)?;
    m.add_function(wrap_pyfunction!(refinitiv_bridge_failed_lookup_records, m)?)?;
    m.add_function(wrap_pyfunction!(
        refinitiv_bridge_ownership_universe_candidate_key,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        refinitiv_bridge_resolve_ownership_lookup_input,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        refinitiv_bridge_resolution_target_class,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        refinitiv_bridge_has_conventional_source,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        refinitiv_bridge_resolution_support_scope,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        refinitiv_bridge_resolution_block_reason,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        refinitiv_bridge_match_lookup_result_values,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(refinitiv_bridge_value_counts, m)?)?;
    m.add_function(wrap_pyfunction!(refinitiv_bridge_count_true_records, m)?)?;
    m.add_function(wrap_pyfunction!(refinitiv_bridge_true_field_counts, m)?)?;
    m.add_function(wrap_pyfunction!(
        refinitiv_bridge_resolution_summary_counts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        refinitiv_bridge_resolution_diagnostic_class_summary,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        refinitiv_bridge_ownership_validation_role_sort_key,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        refinitiv_bridge_ownership_validation_handoff_rows,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        refinitiv_bridge_ownership_validation_handoff_columns,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        refinitiv_bridge_compare_ownership_result_rows,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        refinitiv_bridge_ownership_validation_case_rows,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        refinitiv_bridge_ownership_validation_case_columns,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        refinitiv_bridge_ownership_smoke_sample_rows,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        refinitiv_bridge_ownership_universe_handoff_rows,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        refinitiv_bridge_ownership_universe_handoff_columns,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        refinitiv_bridge_accepted_source_matches_target,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        refinitiv_bridge_adjacent_extension_choice,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        refinitiv_bridge_alternative_identifier,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        refinitiv_bridge_alternative_identifiers,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(normalize_kypermno_value, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_kypermno_values, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_doc_ownership_float_value, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_doc_ownership_int_value, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_doc_ownership_date_value, m)?)?;
    m.add_function(wrap_pyfunction!(
        doc_ownership_most_recent_quarter_end_before,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        doc_ownership_target_effective_date_for_quarter_end,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(doc_ownership_clip_date_lower, m)?)?;
    m.add_function(wrap_pyfunction!(doc_ownership_clip_date_upper, m)?)?;
    m.add_function(wrap_pyfunction!(
        doc_ownership_matching_retrieval_sheets,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(doc_ownership_request_rows, m)?)?;
    m.add_function(wrap_pyfunction!(doc_ownership_request_rows_columns, m)?)?;
    m.add_function(wrap_pyfunction!(doc_ownership_universe_summary_values, m)?)?;
    m.add_function(wrap_pyfunction!(
        doc_ownership_universe_summary_column_values,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(doc_ownership_select_exact_hit_rows, m)?)?;
    m.add_function(wrap_pyfunction!(doc_ownership_select_exact_hit_columns, m)?)?;
    m.add_function(wrap_pyfunction!(doc_ownership_select_fallback_hit_rows, m)?)?;
    m.add_function(wrap_pyfunction!(
        doc_ownership_select_fallback_hit_columns,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        doc_ownership_fallback_request_row_indices,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        doc_ownership_fallback_request_row_indices_columns,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(doc_ownership_final_rows, m)?)?;
    m.add_function(wrap_pyfunction!(refinitiv_analyst_shift_months, m)?)?;
    m.add_function(wrap_pyfunction!(
        refinitiv_analyst_latest_date_on_or_before,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        refinitiv_analyst_build_normalized_event_rows,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        clean_doc_ownership_institutional_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(normalize_doc_ownership_category_value, m)?)?;
    m.add_function(wrap_pyfunction!(
        is_doc_ownership_institutional_category,
        m
    )?)?;
    Ok(())
}

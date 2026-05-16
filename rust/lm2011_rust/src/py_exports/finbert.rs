// FinBERT allocation, sampling, tail, and confusion-review entrypoints.
use crate::*;
use pyo3::wrap_pyfunction;

pub(crate) fn register(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    m.add_function(wrap_pyfunction!(normalize_finbert_label_name, m)?)?;
    m.add_function(wrap_pyfunction!(resolve_finbert_label_mapping, m)?)?;
    m.add_function(wrap_pyfunction!(finbert_median_value, m)?)?;
    m.add_function(wrap_pyfunction!(finbert_stage_summary, m)?)?;
    m.add_function(wrap_pyfunction!(finbert_staged_bucket_summary_rows, m)?)?;
    m.add_function(wrap_pyfunction!(finbert_token_bucket_counts, m)?)?;
    m.add_function(wrap_pyfunction!(finbert_bucket_value_for_name, m)?)?;
    m.add_function(wrap_pyfunction!(round_up_to_multiple_value, m)?)?;
    m.add_function(wrap_pyfunction!(finbert_recommend_bucket_edge, m)?)?;
    m.add_function(wrap_pyfunction!(finbert_bucket_length_summary_rows, m)?)?;
    m.add_function(wrap_pyfunction!(finbert_bucket_length_summary_columns, m)?)?;
    m.add_function(wrap_pyfunction!(parse_device_index_value, m)?)?;
    m.add_function(wrap_pyfunction!(resolve_finbert_amp_dtype_name, m)?)?;
    m.add_function(wrap_pyfunction!(finbert_coverage_report, m)?)?;
    m.add_function(wrap_pyfunction!(finbert_split_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(finbert_fallback_split_warning_payload, m)?)?;
    m.add_function(wrap_pyfunction!(finbert_preprocessing_manifest_counts, m)?)?;
    m.add_function(wrap_pyfunction!(
        finbert_preprocessing_manifest_count_columns,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(finbert_softmax_row, m)?)?;
    m.add_function(wrap_pyfunction!(finbert_softmax_rows, m)?)?;
    m.add_function(wrap_pyfunction!(
        finbert_probability_columns_and_predicted_labels,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(finbert_confusion_candidate_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(finbert_confusion_cell, m)?)?;
    m.add_function(wrap_pyfunction!(finbert_confusion_sample_id_rows, m)?)?;
    m.add_function(wrap_pyfunction!(finbert_confusion_sample_id_columns, m)?)?;
    m.add_function(wrap_pyfunction!(finbert_confusion_neighbor_target_rows, m)?)?;
    m.add_function(wrap_pyfunction!(
        finbert_confusion_neighbor_target_columns,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(finbert_confusion_target_position_rows, m)?)?;
    m.add_function(wrap_pyfunction!(
        finbert_confusion_target_position_columns,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        finbert_confusion_finalize_allocation_rows,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        finbert_confusion_labeling_input_records,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(finbert_confusion_chunk_row_indices, m)?)?;
    m.add_function(wrap_pyfunction!(finbert_confusion_csv_safe_rows, m)?)?;
    m.add_function(wrap_pyfunction!(finbert_confusion_reviewed_case_rows, m)?)?;
    m.add_function(wrap_pyfunction!(
        finbert_confusion_reviewed_case_columns,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(finbert_confusion_metric_payload, m)?)?;
    m.add_function(wrap_pyfunction!(finbert_confusion_counts_by_cell, m)?)?;
    m.add_function(wrap_pyfunction!(
        finbert_confusion_uncertain_metric_bounds,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(finbert_confusion_bucket_metric_rows, m)?)?;
    m.add_function(wrap_pyfunction!(
        finbert_confusion_bucket_metric_columns,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        finbert_confusion_examples_by_cell_markdown,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        finbert_confusion_balanced_sample_count_pairs,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        finbert_confusion_proportional_sample_counts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        lm2011_extension_normal_approx_two_sided_p_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(lseg_doc_unresolved_mask, m)?)?;
    m.add_function(wrap_pyfunction!(finbert_selection_key_value, m)?)?;
    m.add_function(wrap_pyfunction!(finbert_sentence_sample_key_value, m)?)?;
    m.add_function(wrap_pyfunction!(finbert_sentence_sample_key_values, m)?)?;
    m.add_function(wrap_pyfunction!(finbert_sentence_consider_sample_rows, m)?)?;
    m.add_function(wrap_pyfunction!(finbert_sentence_update_accumulators, m)?)?;
    m.add_function(wrap_pyfunction!(finbert_sentence_sample_candidate_rows, m)?)?;
    m.add_function(wrap_pyfunction!(
        finbert_sentence_render_sample_markdown,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        finbert_sentence_render_sample_markdown_columns,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        finbert_sentence_item_sentiment_count_rows,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        finbert_sentence_year_item_sentiment_count_rows,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(finbert_selection_keys, m)?)?;
    m.add_function(wrap_pyfunction!(finbert_ranked_selection_indices, m)?)?;
    m.add_function(wrap_pyfunction!(finbert_capacity_rows_by_year, m)?)?;
    m.add_function(wrap_pyfunction!(finbert_capacity_rows_by_year_item, m)?)?;
    m.add_function(wrap_pyfunction!(finbert_allocation_targets, m)?)?;
    m.add_function(wrap_pyfunction!(finbert_dataset_share_rows, m)?)?;
    m.add_function(wrap_pyfunction!(finbert_dataset_share_row_columns, m)?)?;
    m.add_function(wrap_pyfunction!(finbert_year_allocation_rows, m)?)?;
    m.add_function(wrap_pyfunction!(finbert_year_item_allocation_rows, m)?)?;
    m.add_function(wrap_pyfunction!(
        finbert_constrained_hamilton_allocations,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(finbert_tail_text_scope_value, m)?)?;
    m.add_function(wrap_pyfunction!(finbert_tail_doc_surface_rows, m)?)?;
    m.add_function(wrap_pyfunction!(finbert_tail_doc_surface_columns, m)?)?;
    m.add_function(wrap_pyfunction!(assign_finbert_token_bucket_value, m)?)?;
    m.add_function(wrap_pyfunction!(assign_finbert_token_bucket_values, m)?)?;
    m.add_function(wrap_pyfunction!(finbert_visible_prefix_retained_end, m)?)?;
    Ok(())
}

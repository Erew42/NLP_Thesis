// LM2011 validation, extension tables, regression, and event-study entrypoints.
use crate::*;
use pyo3::wrap_pyfunction;

pub(crate) fn register(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    m.add_function(wrap_pyfunction!(lm2011_extension_result_rows, m)?)?;
    m.add_function(wrap_pyfunction!(lm2011_extension_result_columns, m)?)?;
    m.add_function(wrap_pyfunction!(lm2011_extension_quarterly_fit_rows, m)?)?;
    m.add_function(wrap_pyfunction!(lm2011_extension_quarterly_fit_columns, m)?)?;
    m.add_function(wrap_pyfunction!(lm2011_extension_skipped_quarter_rows, m)?)?;
    m.add_function(wrap_pyfunction!(
        lm2011_extension_skipped_quarter_columns,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        lm2011_extension_quarterly_difference_rows,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        lm2011_extension_fit_summary_estimated_row,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        lm2011_extension_fit_comparison_estimated_row,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(legacy_lm2011_tokens_value, m)?)?;
    m.add_function(wrap_pyfunction!(lm2011_validation_truncate_text, m)?)?;
    m.add_function(wrap_pyfunction!(
        lm2011_validation_marker_flags_and_snippet,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(lm2011_validation_count_true, m)?)?;
    m.add_function(wrap_pyfunction!(lm2011_validation_count_series, m)?)?;
    m.add_function(wrap_pyfunction!(lm2011_validation_form_count_rows, m)?)?;
    m.add_function(wrap_pyfunction!(
        lm2011_validation_normalized_form_counts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(lm2011_validation_term_count_updates, m)?)?;
    m.add_function(wrap_pyfunction!(lm2011_validation_event_attrition_rows, m)?)?;
    m.add_function(wrap_pyfunction!(lm2011_validation_units_row, m)?)?;
    m.add_function(wrap_pyfunction!(
        lm2011_validation_packet_d_coverage_row,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(lm2011_validation_packet_a_mda_rows, m)?)?;
    m.add_function(wrap_pyfunction!(
        lm2011_validation_packet_a_mda_row_columns,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        lm2011_validation_packet_a_delta_summary,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        lm2011_validation_packet_a_summary_rows,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        lm2011_validation_packet_a_strip_comparison_rows,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        lm2011_validation_packet_a_example_rows,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(lm2011_previous_month_end, m)?)?;
    m.add_function(wrap_pyfunction!(lm2011_previous_month_end_values, m)?)?;
    m.add_function(wrap_pyfunction!(lm2011_quarter_start, m)?)?;
    m.add_function(wrap_pyfunction!(lm2011_quarter_start_values, m)?)?;
    m.add_function(wrap_pyfunction!(
        lm2011_validated_event_window_doc_batch_size,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(lm2011_validated_event_window_days, m)?)?;
    m.add_function(wrap_pyfunction!(lm2011_event_window_end_day, m)?)?;
    m.add_function(wrap_pyfunction!(lm2011_postevent_start_day, m)?)?;
    m.add_function(wrap_pyfunction!(lm2011_ols_alpha_rmse, m)?)?;
    m.add_function(wrap_pyfunction!(
        lm2011_regression_metrics_from_window_rows,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        lm2011_regression_metrics_from_window_columns,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(lm2011_ols_ff4_coefficients, m)?)?;
    m.add_function(wrap_pyfunction!(lm2011_strategy_factor_loading_rows, m)?)?;
    m.add_function(wrap_pyfunction!(lm2011_strategy_factor_loading_columns, m)?)?;
    m.add_function(wrap_pyfunction!(lm2011_weighted_mean, m)?)?;
    m.add_function(wrap_pyfunction!(lm2011_newey_west_standard_error, m)?)?;
    m.add_function(wrap_pyfunction!(lm2011_fama_macbeth_result_rows, m)?)?;
    m.add_function(wrap_pyfunction!(lm2011_cross_section_design_rows, m)?)?;
    m.add_function(wrap_pyfunction!(lm2011_cross_section_design_columns, m)?)?;
    m.add_function(wrap_pyfunction!(lm2011_table_ia_ii_result_rows, m)?)?;
    m.add_function(wrap_pyfunction!(lm2011_table_ia_ii_result_columns, m)?)?;
    m.add_function(wrap_pyfunction!(item7_lm_floor_threshold_summary_row, m)?)?;
    Ok(())
}

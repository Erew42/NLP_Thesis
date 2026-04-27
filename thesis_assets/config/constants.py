from __future__ import annotations


RUN_FAMILY_LM2011_POST_REFINITIV = "lm2011_post_refinitiv"
RUN_FAMILY_LM2011_EXTENSION = "lm2011_extension"
RUN_FAMILY_LM2011_NW_LAG_SENSITIVITY = "lm2011_nw_lag_sensitivity"
RUN_FAMILY_FINBERT_RUN = "finbert_run"
RUN_FAMILY_FINBERT_ROBUSTNESS = "finbert_robustness"

ARTIFACT_KEY_TABLE_I_SAMPLE_CREATION = "lm2011_table_i_sample_creation"
ARTIFACT_KEY_TABLE_IA_II_RESULTS = "lm2011_table_ia_ii_results"
ARTIFACT_KEY_TABLE_IV_RESULTS_NO_OWNERSHIP = "lm2011_table_iv_results_no_ownership"
ARTIFACT_KEY_TABLE_VI_RESULTS_NO_OWNERSHIP = "lm2011_table_vi_results_no_ownership"
ARTIFACT_KEY_RETURN_REGRESSION_PANEL_FULL_10K = "lm2011_return_regression_panel_full_10k"
ARTIFACT_KEY_RETURN_REGRESSION_PANEL_MDA = "lm2011_return_regression_panel_mda"
ARTIFACT_KEY_TRADING_STRATEGY_MONTHLY_RETURNS = "lm2011_trading_strategy_monthly_returns"
ARTIFACT_KEY_TABLE_IV_SKIPPED_QUARTERS = "lm2011_table_iv_results_skipped_quarters"
ARTIFACT_KEY_TABLE_V_SKIPPED_QUARTERS = "lm2011_table_v_results_skipped_quarters"
ARTIFACT_KEY_TABLE_VI_SKIPPED_QUARTERS = "lm2011_table_vi_results_skipped_quarters"
ARTIFACT_KEY_EXTENSION_ANALYSIS_PANEL = "lm2011_extension_analysis_panel"
ARTIFACT_KEY_EXTENSION_RESULTS = "lm2011_extension_results"
ARTIFACT_KEY_EXTENSION_FIT_SUMMARY = "lm2011_extension_fit_summary"
ARTIFACT_KEY_EXTENSION_FIT_COMPARISONS = "lm2011_extension_fit_comparisons"
ARTIFACT_KEY_EXTENSION_FIT_DIFFERENCE_QUARTERLY = "lm2011_extension_fit_difference_quarterly"
ARTIFACT_KEY_EXTENSION_FIT_SKIPPED_QUARTERS = "lm2011_extension_fit_skipped_quarters"
ARTIFACT_KEY_NW_LAG_CORE_TABLES = "core_tables_nw_lag_sensitivity"
ARTIFACT_KEY_NW_LAG_EXTENSION_RESULTS = "extension_results_nw_lag_sensitivity"
ARTIFACT_KEY_NW_LAG_EXTENSION_FIT_COMPARISONS = "extension_fit_comparisons_nw_lag_sensitivity"
ARTIFACT_KEY_EXTENSION_SAMPLE_LOSS = "lm2011_extension_sample_loss"
ARTIFACT_KEY_EXTENSION_CONTROL_LADDER = "lm2011_extension_control_ladder"
ARTIFACT_KEY_EXTENSION_DICTIONARY_SURFACE = "lm2011_extension_dictionary_surface"
ARTIFACT_KEY_EXTENSION_FINBERT_SURFACE = "lm2011_extension_finbert_surface"
ARTIFACT_KEY_FINBERT_ITEM_FEATURES_LONG = "finbert_item_features_long"
ARTIFACT_KEY_FINBERT_MODEL_INFERENCE_YEARLY_SUMMARY = "finbert_model_inference_yearly_summary"
ARTIFACT_KEY_FINBERT_COVERAGE_REPORT = "finbert_coverage_report"
ARTIFACT_KEY_FINBERT_SENTENCE_SCORES_DIR = "finbert_sentence_scores_dir"
ARTIFACT_KEY_FINBERT_ROBUSTNESS_EXISTING_SCALE_COEFFICIENTS = "finbert_robustness_existing_scale_coefficients"
ARTIFACT_KEY_FINBERT_ROBUSTNESS_TAIL_COEFFICIENTS = "finbert_robustness_tail_coefficients"
ARTIFACT_KEY_FINBERT_ROBUSTNESS_QUANTILE_COEFFICIENTS = "finbert_robustness_quantile_coefficients"
ARTIFACT_KEY_FINBERT_ROBUSTNESS_EXISTING_SCALE_FIT_COMPARISONS = "finbert_robustness_existing_scale_fit_comparisons"
ARTIFACT_KEY_FINBERT_ROBUSTNESS_TAIL_FIT_COMPARISONS = "finbert_robustness_tail_fit_comparisons"
ARTIFACT_KEY_FINBERT_ROBUSTNESS_QUANTILE_FIT_COMPARISONS = "finbert_robustness_quantile_fit_comparisons"
ARTIFACT_KEY_FINBERT_ROBUSTNESS_EXISTING_SCALE_FIT_SKIPPED_QUARTERS = (
    "finbert_robustness_existing_scale_fit_skipped_quarters"
)
ARTIFACT_KEY_FINBERT_ROBUSTNESS_TAIL_FIT_SKIPPED_QUARTERS = "finbert_robustness_tail_fit_skipped_quarters"
ARTIFACT_KEY_FINBERT_ROBUSTNESS_QUANTILE_FIT_SKIPPED_QUARTERS = "finbert_robustness_quantile_fit_skipped_quarters"

ARTIFACT_FILENAMES: dict[str, str] = {
    ARTIFACT_KEY_TABLE_I_SAMPLE_CREATION: "lm2011_table_i_sample_creation.parquet",
    ARTIFACT_KEY_TABLE_IA_II_RESULTS: "lm2011_table_ia_ii_results.parquet",
    ARTIFACT_KEY_TABLE_IV_RESULTS_NO_OWNERSHIP: "lm2011_table_iv_results_no_ownership.parquet",
    ARTIFACT_KEY_TABLE_VI_RESULTS_NO_OWNERSHIP: "lm2011_table_vi_results_no_ownership.parquet",
    ARTIFACT_KEY_RETURN_REGRESSION_PANEL_FULL_10K: "lm2011_return_regression_panel_full_10k.parquet",
    ARTIFACT_KEY_RETURN_REGRESSION_PANEL_MDA: "lm2011_return_regression_panel_mda.parquet",
    ARTIFACT_KEY_TRADING_STRATEGY_MONTHLY_RETURNS: "lm2011_trading_strategy_monthly_returns.parquet",
    ARTIFACT_KEY_TABLE_IV_SKIPPED_QUARTERS: "lm2011_table_iv_results_skipped_quarters.parquet",
    ARTIFACT_KEY_TABLE_V_SKIPPED_QUARTERS: "lm2011_table_v_results_skipped_quarters.parquet",
    ARTIFACT_KEY_TABLE_VI_SKIPPED_QUARTERS: "lm2011_table_vi_results_skipped_quarters.parquet",
    ARTIFACT_KEY_EXTENSION_ANALYSIS_PANEL: "lm2011_extension_analysis_panel.parquet",
    ARTIFACT_KEY_EXTENSION_RESULTS: "lm2011_extension_results.parquet",
    ARTIFACT_KEY_EXTENSION_FIT_SUMMARY: "lm2011_extension_fit_summary.parquet",
    ARTIFACT_KEY_EXTENSION_FIT_COMPARISONS: "lm2011_extension_fit_comparisons.parquet",
    ARTIFACT_KEY_EXTENSION_FIT_DIFFERENCE_QUARTERLY: "lm2011_extension_fit_difference_quarterly.parquet",
    ARTIFACT_KEY_EXTENSION_FIT_SKIPPED_QUARTERS: "lm2011_extension_fit_skipped_quarters.parquet",
    ARTIFACT_KEY_NW_LAG_CORE_TABLES: "core_tables_nw_lag_sensitivity.parquet",
    ARTIFACT_KEY_NW_LAG_EXTENSION_RESULTS: "extension_results_nw_lag_sensitivity.parquet",
    ARTIFACT_KEY_NW_LAG_EXTENSION_FIT_COMPARISONS: "extension_fit_comparisons_nw_lag_sensitivity.parquet",
    ARTIFACT_KEY_EXTENSION_SAMPLE_LOSS: "lm2011_extension_sample_loss.parquet",
    ARTIFACT_KEY_EXTENSION_CONTROL_LADDER: "lm2011_extension_control_ladder.parquet",
    ARTIFACT_KEY_EXTENSION_DICTIONARY_SURFACE: "lm2011_extension_dictionary_surface.parquet",
    ARTIFACT_KEY_EXTENSION_FINBERT_SURFACE: "lm2011_extension_finbert_surface.parquet",
    ARTIFACT_KEY_FINBERT_ITEM_FEATURES_LONG: "item_features_long.parquet",
    ARTIFACT_KEY_FINBERT_MODEL_INFERENCE_YEARLY_SUMMARY: "model_inference_yearly_summary.parquet",
    ARTIFACT_KEY_FINBERT_COVERAGE_REPORT: "coverage_report.parquet",
    ARTIFACT_KEY_FINBERT_SENTENCE_SCORES_DIR: "sentence_scores/by_year",
    ARTIFACT_KEY_FINBERT_ROBUSTNESS_EXISTING_SCALE_COEFFICIENTS: "finbert_robustness_existing_scale_coefficients.parquet",
    ARTIFACT_KEY_FINBERT_ROBUSTNESS_TAIL_COEFFICIENTS: "finbert_robustness_tail_coefficients.parquet",
    ARTIFACT_KEY_FINBERT_ROBUSTNESS_QUANTILE_COEFFICIENTS: "finbert_robustness_quantile_coefficients.parquet",
    ARTIFACT_KEY_FINBERT_ROBUSTNESS_EXISTING_SCALE_FIT_COMPARISONS: "finbert_robustness_existing_scale_fit_comparisons.parquet",
    ARTIFACT_KEY_FINBERT_ROBUSTNESS_TAIL_FIT_COMPARISONS: "finbert_robustness_tail_fit_comparisons.parquet",
    ARTIFACT_KEY_FINBERT_ROBUSTNESS_QUANTILE_FIT_COMPARISONS: "finbert_robustness_quantile_fit_comparisons.parquet",
    ARTIFACT_KEY_FINBERT_ROBUSTNESS_EXISTING_SCALE_FIT_SKIPPED_QUARTERS: (
        "finbert_robustness_existing_scale_fit_skipped_quarters.parquet"
    ),
    ARTIFACT_KEY_FINBERT_ROBUSTNESS_TAIL_FIT_SKIPPED_QUARTERS: (
        "finbert_robustness_tail_fit_skipped_quarters.parquet"
    ),
    ARTIFACT_KEY_FINBERT_ROBUSTNESS_QUANTILE_FIT_SKIPPED_QUARTERS: (
        "finbert_robustness_quantile_fit_skipped_quarters.parquet"
    ),
}

ARTIFACT_ALTERNATE_FILENAMES: dict[str, tuple[str, ...]] = {
    ARTIFACT_KEY_TABLE_IA_II_RESULTS: ("lm2011_table_ia_ii_results.rerun.parquet",),
    ARTIFACT_KEY_TABLE_VI_RESULTS_NO_OWNERSHIP: ("lm2011_table_vi_results_no_ownership_validation.parquet",),
    ARTIFACT_KEY_TRADING_STRATEGY_MONTHLY_RETURNS: ("lm2011_trading_strategy_monthly_returns.rerun.parquet",),
}

ARTIFACT_MANIFEST_KEYS: dict[str, tuple[str, ...]] = {
    ARTIFACT_KEY_TABLE_I_SAMPLE_CREATION: ("table_i_sample_creation",),
    ARTIFACT_KEY_TABLE_IA_II_RESULTS: ("table_ia_ii_results",),
    ARTIFACT_KEY_TABLE_IV_RESULTS_NO_OWNERSHIP: ("table_iv_results_no_ownership",),
    ARTIFACT_KEY_TABLE_VI_RESULTS_NO_OWNERSHIP: ("table_vi_results_no_ownership",),
    ARTIFACT_KEY_RETURN_REGRESSION_PANEL_FULL_10K: ("return_regression_panel_full_10k",),
    ARTIFACT_KEY_RETURN_REGRESSION_PANEL_MDA: ("return_regression_panel_mda",),
    ARTIFACT_KEY_TRADING_STRATEGY_MONTHLY_RETURNS: ("trading_strategy_monthly_returns",),
    ARTIFACT_KEY_EXTENSION_ANALYSIS_PANEL: ("extension_analysis_panel",),
    ARTIFACT_KEY_EXTENSION_RESULTS: ("extension_results",),
    ARTIFACT_KEY_EXTENSION_FIT_SUMMARY: ("extension_fit_summary",),
    ARTIFACT_KEY_EXTENSION_FIT_COMPARISONS: ("extension_fit_comparisons",),
    ARTIFACT_KEY_EXTENSION_FIT_DIFFERENCE_QUARTERLY: ("extension_fit_difference_quarterly",),
    ARTIFACT_KEY_EXTENSION_FIT_SKIPPED_QUARTERS: ("extension_fit_skipped_quarters",),
    ARTIFACT_KEY_NW_LAG_CORE_TABLES: ("core_tables_nw_lag_sensitivity",),
    ARTIFACT_KEY_NW_LAG_EXTENSION_RESULTS: ("extension_results_nw_lag_sensitivity",),
    ARTIFACT_KEY_NW_LAG_EXTENSION_FIT_COMPARISONS: ("extension_fit_comparisons_nw_lag_sensitivity",),
    ARTIFACT_KEY_EXTENSION_SAMPLE_LOSS: ("extension_sample_loss",),
    ARTIFACT_KEY_EXTENSION_CONTROL_LADDER: ("extension_control_ladder",),
    ARTIFACT_KEY_EXTENSION_DICTIONARY_SURFACE: ("extension_dictionary_surface",),
    ARTIFACT_KEY_EXTENSION_FINBERT_SURFACE: ("extension_finbert_surface",),
    ARTIFACT_KEY_FINBERT_ITEM_FEATURES_LONG: ("item_features_long_path", "item_features_long"),
    ARTIFACT_KEY_FINBERT_MODEL_INFERENCE_YEARLY_SUMMARY: ("model_inference_yearly_summary",),
    ARTIFACT_KEY_FINBERT_COVERAGE_REPORT: ("coverage_report",),
    ARTIFACT_KEY_FINBERT_SENTENCE_SCORES_DIR: ("sentence_scores_dir",),
    ARTIFACT_KEY_FINBERT_ROBUSTNESS_EXISTING_SCALE_COEFFICIENTS: ("existing_scale_coefficients_path",),
    ARTIFACT_KEY_FINBERT_ROBUSTNESS_TAIL_COEFFICIENTS: ("tail_coefficients_path",),
    ARTIFACT_KEY_FINBERT_ROBUSTNESS_QUANTILE_COEFFICIENTS: ("quantile_coefficients_path",),
    ARTIFACT_KEY_FINBERT_ROBUSTNESS_EXISTING_SCALE_FIT_COMPARISONS: ("existing_scale_fit_comparisons_path",),
    ARTIFACT_KEY_FINBERT_ROBUSTNESS_TAIL_FIT_COMPARISONS: ("tail_fit_comparisons_path",),
    ARTIFACT_KEY_FINBERT_ROBUSTNESS_QUANTILE_FIT_COMPARISONS: ("quantile_fit_comparisons_path",),
}

RUN_MANIFEST_FILENAMES: dict[str, tuple[str, ...]] = {
    RUN_FAMILY_LM2011_POST_REFINITIV: ("lm2011_sample_run_manifest.json", "run_manifest.json"),
    RUN_FAMILY_LM2011_EXTENSION: ("lm2011_extension_run_manifest.json", "run_manifest.json"),
    RUN_FAMILY_LM2011_NW_LAG_SENSITIVITY: ("lm2011_nw_lag_sensitivity_run_manifest.json", "run_manifest.json"),
    RUN_FAMILY_FINBERT_RUN: ("run_manifest.json",),
    RUN_FAMILY_FINBERT_ROBUSTNESS: ("finbert_robustness_run_manifest.json",),
}

RUN_FAMILY_SENTINEL_ARTIFACTS: dict[str, tuple[str, ...]] = {
    RUN_FAMILY_LM2011_POST_REFINITIV: (ARTIFACT_KEY_TABLE_I_SAMPLE_CREATION,),
    RUN_FAMILY_LM2011_EXTENSION: (
        ARTIFACT_KEY_EXTENSION_FIT_SUMMARY,
        ARTIFACT_KEY_EXTENSION_FIT_COMPARISONS,
    ),
    RUN_FAMILY_LM2011_NW_LAG_SENSITIVITY: (
        ARTIFACT_KEY_NW_LAG_CORE_TABLES,
        ARTIFACT_KEY_NW_LAG_EXTENSION_RESULTS,
        ARTIFACT_KEY_NW_LAG_EXTENSION_FIT_COMPARISONS,
    ),
    RUN_FAMILY_FINBERT_RUN: (ARTIFACT_KEY_FINBERT_ITEM_FEATURES_LONG,),
    RUN_FAMILY_FINBERT_ROBUSTNESS: (
        ARTIFACT_KEY_FINBERT_ROBUSTNESS_EXISTING_SCALE_COEFFICIENTS,
        ARTIFACT_KEY_FINBERT_ROBUSTNESS_TAIL_COEFFICIENTS,
        ARTIFACT_KEY_FINBERT_ROBUSTNESS_QUANTILE_COEFFICIENTS,
    ),
}

OUTPUT_SUBDIRS: tuple[str, ...] = ("tables", "figures", "csv", "tex", "logs")
MANIFEST_FILENAME = "manifest.json"
BUILD_LOG_FILENAME = "build.log"

REGISTRY_MODULES: tuple[str, ...] = (
    "chapter4_descriptives",
    "chapter5_results",
)

DEFAULT_COMMON_SUCCESS_POLICY = "all_selected_models_common_successful_quarters"
DEFAULT_COMPARISON_JOIN_KEYS: tuple[str, ...] = (
    "doc_id",
    "filing_date",
    "text_scope",
    "cleaning_policy_id",
)
EXTENSION_OWNERSHIP_SUPPORT_COLUMN = "common_support_flag_ownership"

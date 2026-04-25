from __future__ import annotations


RUN_FAMILY_LM2011_POST_REFINITIV = "lm2011_post_refinitiv"
RUN_FAMILY_LM2011_EXTENSION = "lm2011_extension"
RUN_FAMILY_FINBERT_RUN = "finbert_run"
RUN_FAMILY_FINBERT_ROBUSTNESS = "finbert_robustness"

ARTIFACT_KEY_TABLE_I_SAMPLE_CREATION = "lm2011_table_i_sample_creation"
ARTIFACT_KEY_TABLE_VI_RESULTS_NO_OWNERSHIP = "lm2011_table_vi_results_no_ownership"
ARTIFACT_KEY_EXTENSION_ANALYSIS_PANEL = "lm2011_extension_analysis_panel"
ARTIFACT_KEY_EXTENSION_FIT_SUMMARY = "lm2011_extension_fit_summary"
ARTIFACT_KEY_EXTENSION_FIT_COMPARISONS = "lm2011_extension_fit_comparisons"
ARTIFACT_KEY_EXTENSION_DICTIONARY_SURFACE = "lm2011_extension_dictionary_surface"
ARTIFACT_KEY_EXTENSION_FINBERT_SURFACE = "lm2011_extension_finbert_surface"
ARTIFACT_KEY_FINBERT_ITEM_FEATURES_LONG = "finbert_item_features_long"
ARTIFACT_KEY_FINBERT_SENTENCE_SCORES_DIR = "finbert_sentence_scores_dir"
ARTIFACT_KEY_FINBERT_ROBUSTNESS_EXISTING_SCALE_COEFFICIENTS = "finbert_robustness_existing_scale_coefficients"
ARTIFACT_KEY_FINBERT_ROBUSTNESS_TAIL_COEFFICIENTS = "finbert_robustness_tail_coefficients"
ARTIFACT_KEY_FINBERT_ROBUSTNESS_QUANTILE_COEFFICIENTS = "finbert_robustness_quantile_coefficients"
ARTIFACT_KEY_FINBERT_ROBUSTNESS_EXISTING_SCALE_FIT_COMPARISONS = "finbert_robustness_existing_scale_fit_comparisons"
ARTIFACT_KEY_FINBERT_ROBUSTNESS_TAIL_FIT_COMPARISONS = "finbert_robustness_tail_fit_comparisons"
ARTIFACT_KEY_FINBERT_ROBUSTNESS_QUANTILE_FIT_COMPARISONS = "finbert_robustness_quantile_fit_comparisons"

ARTIFACT_FILENAMES: dict[str, str] = {
    ARTIFACT_KEY_TABLE_I_SAMPLE_CREATION: "lm2011_table_i_sample_creation.parquet",
    ARTIFACT_KEY_TABLE_VI_RESULTS_NO_OWNERSHIP: "lm2011_table_vi_results_no_ownership.parquet",
    ARTIFACT_KEY_EXTENSION_ANALYSIS_PANEL: "lm2011_extension_analysis_panel.parquet",
    ARTIFACT_KEY_EXTENSION_FIT_SUMMARY: "lm2011_extension_fit_summary.parquet",
    ARTIFACT_KEY_EXTENSION_FIT_COMPARISONS: "lm2011_extension_fit_comparisons.parquet",
    ARTIFACT_KEY_EXTENSION_DICTIONARY_SURFACE: "lm2011_extension_dictionary_surface.parquet",
    ARTIFACT_KEY_EXTENSION_FINBERT_SURFACE: "lm2011_extension_finbert_surface.parquet",
    ARTIFACT_KEY_FINBERT_ITEM_FEATURES_LONG: "item_features_long.parquet",
    ARTIFACT_KEY_FINBERT_SENTENCE_SCORES_DIR: "sentence_scores/by_year",
    ARTIFACT_KEY_FINBERT_ROBUSTNESS_EXISTING_SCALE_COEFFICIENTS: "finbert_robustness_existing_scale_coefficients.parquet",
    ARTIFACT_KEY_FINBERT_ROBUSTNESS_TAIL_COEFFICIENTS: "finbert_robustness_tail_coefficients.parquet",
    ARTIFACT_KEY_FINBERT_ROBUSTNESS_QUANTILE_COEFFICIENTS: "finbert_robustness_quantile_coefficients.parquet",
    ARTIFACT_KEY_FINBERT_ROBUSTNESS_EXISTING_SCALE_FIT_COMPARISONS: "finbert_robustness_existing_scale_fit_comparisons.parquet",
    ARTIFACT_KEY_FINBERT_ROBUSTNESS_TAIL_FIT_COMPARISONS: "finbert_robustness_tail_fit_comparisons.parquet",
    ARTIFACT_KEY_FINBERT_ROBUSTNESS_QUANTILE_FIT_COMPARISONS: "finbert_robustness_quantile_fit_comparisons.parquet",
}

ARTIFACT_ALTERNATE_FILENAMES: dict[str, tuple[str, ...]] = {
    ARTIFACT_KEY_TABLE_VI_RESULTS_NO_OWNERSHIP: ("lm2011_table_vi_results_no_ownership_validation.parquet",),
}

ARTIFACT_MANIFEST_KEYS: dict[str, tuple[str, ...]] = {
    ARTIFACT_KEY_TABLE_I_SAMPLE_CREATION: ("table_i_sample_creation",),
    ARTIFACT_KEY_TABLE_VI_RESULTS_NO_OWNERSHIP: ("table_vi_results_no_ownership",),
    ARTIFACT_KEY_EXTENSION_ANALYSIS_PANEL: ("extension_analysis_panel",),
    ARTIFACT_KEY_EXTENSION_FIT_SUMMARY: ("extension_fit_summary",),
    ARTIFACT_KEY_EXTENSION_FIT_COMPARISONS: ("extension_fit_comparisons",),
    ARTIFACT_KEY_EXTENSION_DICTIONARY_SURFACE: ("extension_dictionary_surface",),
    ARTIFACT_KEY_EXTENSION_FINBERT_SURFACE: ("extension_finbert_surface",),
    ARTIFACT_KEY_FINBERT_ITEM_FEATURES_LONG: ("item_features_long_path", "item_features_long"),
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
    RUN_FAMILY_FINBERT_RUN: ("run_manifest.json",),
    RUN_FAMILY_FINBERT_ROBUSTNESS: ("finbert_robustness_run_manifest.json",),
}

RUN_FAMILY_SENTINEL_ARTIFACTS: dict[str, tuple[str, ...]] = {
    RUN_FAMILY_LM2011_POST_REFINITIV: (ARTIFACT_KEY_TABLE_I_SAMPLE_CREATION,),
    RUN_FAMILY_LM2011_EXTENSION: (
        ARTIFACT_KEY_EXTENSION_FIT_SUMMARY,
        ARTIFACT_KEY_EXTENSION_FIT_COMPARISONS,
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

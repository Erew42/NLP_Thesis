from __future__ import annotations

from thesis_assets.config.constants import ARTIFACT_KEY_EXTENSION_DICTIONARY_SURFACE
from thesis_assets.config.constants import ARTIFACT_KEY_EXTENSION_FINBERT_SURFACE
from thesis_assets.config.constants import ARTIFACT_KEY_EXTENSION_ANALYSIS_PANEL
from thesis_assets.config.constants import ARTIFACT_KEY_EXTENSION_FIT_SUMMARY
from thesis_assets.config.constants import ARTIFACT_KEY_FINBERT_SENTENCE_SCORES_DIR
from thesis_assets.config.constants import ARTIFACT_KEY_FINBERT_ROBUSTNESS_EXISTING_SCALE_COEFFICIENTS
from thesis_assets.config.constants import ARTIFACT_KEY_FINBERT_ROBUSTNESS_EXISTING_SCALE_FIT_COMPARISONS
from thesis_assets.config.constants import ARTIFACT_KEY_FINBERT_ROBUSTNESS_QUANTILE_COEFFICIENTS
from thesis_assets.config.constants import ARTIFACT_KEY_FINBERT_ROBUSTNESS_QUANTILE_FIT_COMPARISONS
from thesis_assets.config.constants import ARTIFACT_KEY_FINBERT_ROBUSTNESS_TAIL_COEFFICIENTS
from thesis_assets.config.constants import ARTIFACT_KEY_FINBERT_ROBUSTNESS_TAIL_FIT_COMPARISONS
from thesis_assets.config.constants import ARTIFACT_KEY_TABLE_VI_RESULTS_NO_OWNERSHIP
from thesis_assets.config.constants import RUN_FAMILY_FINBERT_ROBUSTNESS
from thesis_assets.config.constants import RUN_FAMILY_FINBERT_RUN
from thesis_assets.config.constants import RUN_FAMILY_LM2011_EXTENSION
from thesis_assets.config.constants import RUN_FAMILY_LM2011_POST_REFINITIV
from thesis_assets.specs import ArtifactRequirement
from thesis_assets.specs import AssetSpec


ASSETS: tuple[AssetSpec, ...] = (
    AssetSpec(
        asset_id="ch5_fit_horserace_item7_c0",
        chapter="chapter5",
        asset_kind="table",
        output_stem="ch5_fit_horserace_item7_c0",
        caption_stub="Chapter 5 fit comparison scaffold for Item 7, filing-period excess returns, and control set C0.",
        notes_stub="Notes stub: this scaffold reports stored fit-comparison outputs without adding empirical interpretation.",
        sample_contract_id="common_success_comparison",
        builder_id="chapter5_fit_horserace",
        required_artifacts=(
            ArtifactRequirement(
                logical_name="extension_fit_summary",
                run_family=RUN_FAMILY_LM2011_EXTENSION,
                artifact_key=ARTIFACT_KEY_EXTENSION_FIT_SUMMARY,
                required_columns=(
                    "text_scope",
                    "outcome_name",
                    "control_set_id",
                    "feature_family",
                    "specification_name",
                    "common_success_policy",
                    "estimator_status",
                ),
            ),
        ),
    ),
    AssetSpec(
        asset_id="ch5_lm2011_table_vi_no_ownership_outcomes",
        chapter="chapter5",
        asset_kind="table",
        output_stem="ch5_lm2011_table_vi_no_ownership_outcomes",
        caption_stub=(
            "LM2011 Table VI no-ownership dictionary regressions across filing-period return, "
            "abnormal volume, and postevent return volatility."
        ),
        notes_stub=(
            "Rows report the signal coefficient from the no-ownership Fama-MacBeth specification. "
            "Filing-period excess-return and postevent-volatility coefficients are multiplied by 100; "
            "abnormal-volume coefficients are reported on the stored scale."
        ),
        sample_contract_id="lm2011_table_vi_no_ownership",
        builder_id="chapter5_lm2011_table_vi_no_ownership",
        required_artifacts=(
            ArtifactRequirement(
                logical_name="table_vi_results_no_ownership",
                run_family=RUN_FAMILY_LM2011_POST_REFINITIV,
                artifact_key=ARTIFACT_KEY_TABLE_VI_RESULTS_NO_OWNERSHIP,
                required_columns=(
                    "table_id",
                    "specification_id",
                    "text_scope",
                    "signal_name",
                    "dependent_variable",
                    "coefficient_name",
                    "estimate",
                    "standard_error",
                    "t_stat",
                    "n_quarters",
                    "mean_quarter_n",
                    "weighting_rule",
                    "nw_lags",
                ),
            ),
        ),
    ),
    AssetSpec(
        asset_id="ch5_concordance_item7_common_sample",
        chapter="chapter5",
        asset_kind="figure",
        output_stem="ch5_concordance_item7_common_sample",
        caption_stub="Chapter 5 Item 7 concordance scaffold on the matched common row sample.",
        notes_stub="Notes stub: concordance figure uses the matched cleaned-scope sample only and does not mix surfaces.",
        sample_contract_id="common_row_comparison",
        builder_id="chapter5_concordance",
        required_artifacts=(
            ArtifactRequirement(
                logical_name="extension_dictionary_surface",
                run_family=RUN_FAMILY_LM2011_EXTENSION,
                artifact_key=ARTIFACT_KEY_EXTENSION_DICTIONARY_SURFACE,
                relative_subdir="replication",
                required_columns=(
                    "doc_id",
                    "filing_date",
                    "text_scope",
                    "cleaning_policy_id",
                    "dictionary_family",
                    "lm_negative_tfidf",
                ),
            ),
            ArtifactRequirement(
                logical_name="extension_finbert_surface",
                run_family=RUN_FAMILY_LM2011_EXTENSION,
                artifact_key=ARTIFACT_KEY_EXTENSION_FINBERT_SURFACE,
                relative_subdir="replication",
                required_columns=(
                    "doc_id",
                    "filing_date",
                    "text_scope",
                    "cleaning_policy_id",
                    "finbert_neg_prob_lenw_mean",
                ),
            ),
        ),
    ),
    AssetSpec(
        asset_id="ch5_between_filing_ecdf_lm_negative_doc_scores",
        chapter="chapter5",
        asset_kind="figure",
        output_stem="ch5_between_filing_ecdf_lm_negative_doc_scores",
        caption_stub="Document-level ECDFs of LM2011 negative dictionary scores.",
        notes_stub="Notes stub: ECDFs use replication-family cleaned-scope document scores for Item 7 and Item 1A.",
        sample_contract_id="raw_available",
        builder_id="chapter5_lm_doc_score_ecdf",
        required_artifacts=(
            ArtifactRequirement(
                logical_name="extension_dictionary_surface",
                run_family=RUN_FAMILY_LM2011_EXTENSION,
                artifact_key=ARTIFACT_KEY_EXTENSION_DICTIONARY_SURFACE,
                relative_subdir="replication",
                required_columns=(
                    "doc_id",
                    "filing_date",
                    "text_scope",
                    "dictionary_family",
                    "lm_negative_prop",
                    "lm_negative_tfidf",
                ),
            ),
        ),
    ),
    AssetSpec(
        asset_id="ch5_between_filing_ecdf_finbert_doc_scores",
        chapter="chapter5",
        asset_kind="figure",
        output_stem="ch5_between_filing_ecdf_finbert_doc_scores",
        caption_stub="Document-level ECDFs of FinBERT negative-tone scores.",
        notes_stub="Notes stub: ECDFs use cleaned-scope document scores for Item 7 and Item 1A.",
        sample_contract_id="raw_available",
        builder_id="chapter5_finbert_doc_score_ecdf",
        required_artifacts=(
            ArtifactRequirement(
                logical_name="extension_finbert_surface",
                run_family=RUN_FAMILY_LM2011_EXTENSION,
                artifact_key=ARTIFACT_KEY_EXTENSION_FINBERT_SURFACE,
                relative_subdir="replication",
                required_columns=(
                    "doc_id",
                    "filing_date",
                    "text_scope",
                    "finbert_neg_prob_lenw_mean",
                    "finbert_net_negative_lenw_mean",
                ),
            ),
        ),
    ),
    AssetSpec(
        asset_id="ch5_within_filing_sentence_ecdf_finbert_negative",
        chapter="chapter5",
        asset_kind="figure",
        output_stem="ch5_within_filing_sentence_ecdf_finbert_negative",
        caption_stub="Sentence-level ECDF of FinBERT negative probability within the analysis-panel filing universe.",
        notes_stub="Notes stub: sentence-level ECDFs are binned summaries over the analysis-panel doc-scope universe.",
        sample_contract_id="analysis_panel_sentence_scores",
        builder_id="chapter5_finbert_sentence_ecdf",
        required_artifacts=(
            ArtifactRequirement(
                logical_name="extension_analysis_panel",
                run_family=RUN_FAMILY_LM2011_EXTENSION,
                artifact_key=ARTIFACT_KEY_EXTENSION_ANALYSIS_PANEL,
                relative_subdir="replication",
                required_columns=("doc_id", "text_scope"),
            ),
            ArtifactRequirement(
                logical_name="sentence_scores",
                run_family=RUN_FAMILY_FINBERT_RUN,
                artifact_key=ARTIFACT_KEY_FINBERT_SENTENCE_SCORES_DIR,
                required_columns=("doc_id", "text_scope", "negative_prob"),
            ),
        ),
    ),
    AssetSpec(
        asset_id="ch5_within_filing_sentence_ecdf_lm_negative_share",
        chapter="chapter5",
        asset_kind="figure",
        output_stem="ch5_within_filing_sentence_ecdf_lm_negative_share",
        caption_stub="Sentence-level ECDF of LM2011 negative word share within the analysis-panel filing universe.",
        notes_stub="Notes stub: sentence-level LM2011 scores are computed in batches from sentence text.",
        sample_contract_id="analysis_panel_sentence_scores",
        builder_id="chapter5_lm_sentence_ecdf",
        required_artifacts=(
            ArtifactRequirement(
                logical_name="extension_analysis_panel",
                run_family=RUN_FAMILY_LM2011_EXTENSION,
                artifact_key=ARTIFACT_KEY_EXTENSION_ANALYSIS_PANEL,
                relative_subdir="replication",
                required_columns=("doc_id", "text_scope"),
            ),
            ArtifactRequirement(
                logical_name="sentence_scores",
                run_family=RUN_FAMILY_FINBERT_RUN,
                artifact_key=ARTIFACT_KEY_FINBERT_SENTENCE_SCORES_DIR,
                required_columns=("doc_id", "text_scope", "sentence_text"),
            ),
        ),
    ),
    AssetSpec(
        asset_id="ch5_within_filing_high_negative_sentence_share",
        chapter="chapter5",
        asset_kind="figure",
        output_stem="ch5_within_filing_high_negative_sentence_share",
        caption_stub="Per-filing share of high-negative sentences for FinBERT and LM2011 sentence scores.",
        notes_stub="Notes stub: high-negative shares use fixed thresholds and batch-combinable counts.",
        sample_contract_id="analysis_panel_sentence_scores",
        builder_id="chapter5_high_negative_sentence_share",
        required_artifacts=(
            ArtifactRequirement(
                logical_name="extension_analysis_panel",
                run_family=RUN_FAMILY_LM2011_EXTENSION,
                artifact_key=ARTIFACT_KEY_EXTENSION_ANALYSIS_PANEL,
                relative_subdir="replication",
                required_columns=("doc_id", "text_scope"),
            ),
            ArtifactRequirement(
                logical_name="sentence_scores",
                run_family=RUN_FAMILY_FINBERT_RUN,
                artifact_key=ARTIFACT_KEY_FINBERT_SENTENCE_SCORES_DIR,
                required_columns=("doc_id", "text_scope", "sentence_text", "negative_prob"),
            ),
        ),
    ),
    AssetSpec(
        asset_id="ch5_concordance_negative_scores_by_scope",
        chapter="chapter5",
        asset_kind="figure",
        output_stem="ch5_concordance_negative_scores_by_scope",
        caption_stub="Document-level concordance between LM2011 and FinBERT negative-tone scores by text scope.",
        notes_stub="Notes stub: concordance uses matched replication-family cleaned-scope rows.",
        sample_contract_id="common_row_comparison",
        builder_id="chapter5_concordance_by_scope",
        required_artifacts=(
            ArtifactRequirement(
                logical_name="extension_dictionary_surface",
                run_family=RUN_FAMILY_LM2011_EXTENSION,
                artifact_key=ARTIFACT_KEY_EXTENSION_DICTIONARY_SURFACE,
                relative_subdir="replication",
                required_columns=(
                    "doc_id",
                    "filing_date",
                    "text_scope",
                    "cleaning_policy_id",
                    "dictionary_family",
                    "lm_negative_tfidf",
                ),
            ),
            ArtifactRequirement(
                logical_name="extension_finbert_surface",
                run_family=RUN_FAMILY_LM2011_EXTENSION,
                artifact_key=ARTIFACT_KEY_EXTENSION_FINBERT_SURFACE,
                relative_subdir="replication",
                required_columns=(
                    "doc_id",
                    "filing_date",
                    "text_scope",
                    "cleaning_policy_id",
                    "finbert_neg_prob_lenw_mean",
                ),
            ),
        ),
    ),
    AssetSpec(
        asset_id="ch5_finbert_robustness_coefficients",
        chapter="chapter5",
        asset_kind="table",
        output_stem="ch5_finbert_robustness_coefficients",
        caption_stub=(
            "FinBERT robustness regressions for filing-period excess returns and control set C0."
        ),
        notes_stub=(
            "Rows report Fama-MacBeth coefficient estimates from the FinBERT robustness run. "
            "The fixed-count top-five-sentence tail measure is excluded. "
            "Stars denote two-sided p-values: *** p<0.01, ** p<0.05, * p<0.10."
        ),
        sample_contract_id="finbert_robustness_regression_results",
        builder_id="chapter5_finbert_robustness_coefficients",
        required_artifacts=(
            ArtifactRequirement(
                logical_name="existing_scale_coefficients",
                run_family=RUN_FAMILY_FINBERT_ROBUSTNESS,
                artifact_key=ARTIFACT_KEY_FINBERT_ROBUSTNESS_EXISTING_SCALE_COEFFICIENTS,
                required_columns=(
                    "variant_id",
                    "text_scope",
                    "specification_name",
                    "coefficient_name",
                    "estimate",
                    "standard_error",
                    "t_stat",
                    "p_value",
                    "n_obs",
                    "n_quarters",
                    "weighting_rule",
                    "estimator_status",
                ),
            ),
            ArtifactRequirement(
                logical_name="tail_coefficients",
                run_family=RUN_FAMILY_FINBERT_ROBUSTNESS,
                artifact_key=ARTIFACT_KEY_FINBERT_ROBUSTNESS_TAIL_COEFFICIENTS,
                required_columns=(
                    "variant_id",
                    "text_scope",
                    "specification_name",
                    "coefficient_name",
                    "estimate",
                    "standard_error",
                    "t_stat",
                    "p_value",
                    "n_obs",
                    "n_quarters",
                    "weighting_rule",
                    "estimator_status",
                ),
            ),
            ArtifactRequirement(
                logical_name="quantile_coefficients",
                run_family=RUN_FAMILY_FINBERT_ROBUSTNESS,
                artifact_key=ARTIFACT_KEY_FINBERT_ROBUSTNESS_QUANTILE_COEFFICIENTS,
                required_columns=(
                    "variant_id",
                    "text_scope",
                    "specification_name",
                    "coefficient_name",
                    "estimate",
                    "standard_error",
                    "t_stat",
                    "p_value",
                    "n_obs",
                    "n_quarters",
                    "weighting_rule",
                    "estimator_status",
                ),
            ),
        ),
    ),
    AssetSpec(
        asset_id="ch5_finbert_robustness_fit_comparisons",
        chapter="chapter5",
        asset_kind="table",
        output_stem="ch5_finbert_robustness_fit_comparisons",
        caption_stub=(
            "Adjusted R-squared comparisons for FinBERT robustness specifications."
        ),
        notes_stub=(
            "Rows report within-common-sample adjusted R-squared differences from the FinBERT robustness run. "
            "The fixed-count top-five-sentence tail measure is excluded. "
            "Stars denote Newey-West p-values for the adjusted R-squared difference: "
            "*** p<0.01, ** p<0.05, * p<0.10."
        ),
        sample_contract_id="finbert_robustness_fit_comparisons",
        builder_id="chapter5_finbert_robustness_fit_comparisons",
        required_artifacts=(
            ArtifactRequirement(
                logical_name="existing_scale_fit_comparisons",
                run_family=RUN_FAMILY_FINBERT_ROBUSTNESS,
                artifact_key=ARTIFACT_KEY_FINBERT_ROBUSTNESS_EXISTING_SCALE_FIT_COMPARISONS,
                required_columns=(
                    "variant_id",
                    "text_scope",
                    "comparison_name",
                    "weighted_avg_delta_adj_r2",
                    "equal_quarter_avg_delta_adj_r2",
                    "nw_t_stat_delta_adj_r2",
                    "nw_p_value_delta_adj_r2",
                    "total_n_obs",
                    "n_quarters",
                    "estimator_status",
                ),
            ),
            ArtifactRequirement(
                logical_name="tail_fit_comparisons",
                run_family=RUN_FAMILY_FINBERT_ROBUSTNESS,
                artifact_key=ARTIFACT_KEY_FINBERT_ROBUSTNESS_TAIL_FIT_COMPARISONS,
                required_columns=(
                    "variant_id",
                    "text_scope",
                    "comparison_name",
                    "weighted_avg_delta_adj_r2",
                    "equal_quarter_avg_delta_adj_r2",
                    "nw_t_stat_delta_adj_r2",
                    "nw_p_value_delta_adj_r2",
                    "total_n_obs",
                    "n_quarters",
                    "estimator_status",
                ),
            ),
            ArtifactRequirement(
                logical_name="quantile_fit_comparisons",
                run_family=RUN_FAMILY_FINBERT_ROBUSTNESS,
                artifact_key=ARTIFACT_KEY_FINBERT_ROBUSTNESS_QUANTILE_FIT_COMPARISONS,
                required_columns=(
                    "variant_id",
                    "text_scope",
                    "comparison_name",
                    "weighted_avg_delta_adj_r2",
                    "equal_quarter_avg_delta_adj_r2",
                    "nw_t_stat_delta_adj_r2",
                    "nw_p_value_delta_adj_r2",
                    "total_n_obs",
                    "n_quarters",
                    "estimator_status",
                ),
            ),
        ),
    ),
)

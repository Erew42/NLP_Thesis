from __future__ import annotations

from thesis_assets.config.constants import ARTIFACT_KEY_EXTENSION_DICTIONARY_SURFACE
from thesis_assets.config.constants import ARTIFACT_KEY_EXTENSION_FINBERT_SURFACE
from thesis_assets.config.constants import ARTIFACT_KEY_EXTENSION_ANALYSIS_PANEL
from thesis_assets.config.constants import ARTIFACT_KEY_EXTENSION_RESULTS
from thesis_assets.config.constants import ARTIFACT_KEY_EXTENSION_FIT_SUMMARY
from thesis_assets.config.constants import ARTIFACT_KEY_EXTENSION_FIT_COMPARISONS
from thesis_assets.config.constants import ARTIFACT_KEY_EXTENSION_FIT_DIFFERENCE_QUARTERLY
from thesis_assets.config.constants import ARTIFACT_KEY_EXTENSION_FIT_SKIPPED_QUARTERS
from thesis_assets.config.constants import ARTIFACT_KEY_FINBERT_SENTENCE_SCORES_DIR
from thesis_assets.config.constants import ARTIFACT_KEY_FINBERT_ROBUSTNESS_EXISTING_SCALE_COEFFICIENTS
from thesis_assets.config.constants import ARTIFACT_KEY_FINBERT_ROBUSTNESS_EXISTING_SCALE_FIT_COMPARISONS
from thesis_assets.config.constants import ARTIFACT_KEY_FINBERT_ROBUSTNESS_EXISTING_SCALE_FIT_SKIPPED_QUARTERS
from thesis_assets.config.constants import ARTIFACT_KEY_FINBERT_ROBUSTNESS_QUANTILE_COEFFICIENTS
from thesis_assets.config.constants import ARTIFACT_KEY_FINBERT_ROBUSTNESS_QUANTILE_FIT_COMPARISONS
from thesis_assets.config.constants import ARTIFACT_KEY_FINBERT_ROBUSTNESS_QUANTILE_FIT_SKIPPED_QUARTERS
from thesis_assets.config.constants import ARTIFACT_KEY_FINBERT_ROBUSTNESS_TAIL_COEFFICIENTS
from thesis_assets.config.constants import ARTIFACT_KEY_FINBERT_ROBUSTNESS_TAIL_FIT_COMPARISONS
from thesis_assets.config.constants import ARTIFACT_KEY_FINBERT_ROBUSTNESS_TAIL_FIT_SKIPPED_QUARTERS
from thesis_assets.config.constants import ARTIFACT_KEY_RETURN_REGRESSION_PANEL_FULL_10K
from thesis_assets.config.constants import ARTIFACT_KEY_TABLE_IA_II_RESULTS
from thesis_assets.config.constants import ARTIFACT_KEY_TABLE_IV_RESULTS_NO_OWNERSHIP
from thesis_assets.config.constants import ARTIFACT_KEY_TABLE_IV_SKIPPED_QUARTERS
from thesis_assets.config.constants import ARTIFACT_KEY_TABLE_V_SKIPPED_QUARTERS
from thesis_assets.config.constants import ARTIFACT_KEY_TABLE_VI_RESULTS_NO_OWNERSHIP
from thesis_assets.config.constants import ARTIFACT_KEY_TABLE_VI_SKIPPED_QUARTERS
from thesis_assets.config.constants import ARTIFACT_KEY_TRADING_STRATEGY_MONTHLY_RETURNS
from thesis_assets.config.constants import RUN_FAMILY_FINBERT_ROBUSTNESS
from thesis_assets.config.constants import RUN_FAMILY_FINBERT_RUN
from thesis_assets.config.constants import RUN_FAMILY_LM2011_EXTENSION
from thesis_assets.config.constants import RUN_FAMILY_LM2011_POST_REFINITIV
from thesis_assets.specs import ArtifactRequirement
from thesis_assets.specs import AssetSpec


ASSETS: tuple[AssetSpec, ...] = (
    AssetSpec(
        asset_id="ch5_lm2011_full_10k_return_coefficients",
        chapter="chapter5",
        asset_kind="table",
        output_stem="ch5_lm2011_full_10k_return_coefficients",
        caption_stub="Full-10-K LM2011-style filing-period return regression coefficients.",
        notes_stub="Rows report full-10-K no-ownership Fama-MacBeth signal coefficients from the stored Table IV artifact. Return coefficients and standard errors are multiplied by 100.",
        sample_contract_id="lm2011_full_10k_return_coefficients",
        builder_id="chapter5_lm2011_full_10k_return_coefficients",
        required_artifacts=(
            ArtifactRequirement(
                logical_name="table_iv_results_no_ownership",
                run_family=RUN_FAMILY_LM2011_POST_REFINITIV,
                artifact_key=ARTIFACT_KEY_TABLE_IV_RESULTS_NO_OWNERSHIP,
                required_columns=(
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
        asset_id="ch5_lm2011_portfolio_long_short",
        chapter="chapter5",
        asset_kind="table",
        output_stem="ch5_lm2011_portfolio_long_short",
        caption_stub="Text-sorted portfolio high-minus-low negativity spreads.",
        notes_stub="Rows report available long-short portfolio results. Current stored artifacts do not include separate Q1 and Q5 leg returns.",
        sample_contract_id="portfolio_long_short",
        builder_id="chapter5_lm2011_portfolio_long_short",
        required_artifacts=(
            ArtifactRequirement(
                logical_name="table_ia_ii_results",
                run_family=RUN_FAMILY_LM2011_POST_REFINITIV,
                artifact_key=ARTIFACT_KEY_TABLE_IA_II_RESULTS,
                required=False,
                required_columns=(
                    "signal_name",
                    "dependent_variable",
                    "coefficient_name",
                    "estimate",
                ),
            ),
        ),
    ),
    AssetSpec(
        asset_id="ch5_lm2011_portfolio_formation_diagnostics",
        chapter="chapter5",
        asset_kind="table",
        output_stem="ch5_lm2011_portfolio_formation_diagnostics",
        caption_stub="Portfolio formation sample and timing diagnostics.",
        notes_stub="Diagnostics summarize available monthly high-minus-low strategy returns. Separate Q1/Q5 leg diagnostics are not stored.",
        sample_contract_id="portfolio_diagnostics",
        builder_id="chapter5_lm2011_portfolio_formation_diagnostics",
        required_artifacts=(
            ArtifactRequirement(
                logical_name="trading_strategy_monthly_returns",
                run_family=RUN_FAMILY_LM2011_POST_REFINITIV,
                artifact_key=ARTIFACT_KEY_TRADING_STRATEGY_MONTHLY_RETURNS,
                required=False,
                required_columns=("portfolio_month", "sort_signal_name", "long_short_return"),
            ),
        ),
    ),
    AssetSpec(
        asset_id="ch5_portfolio_cumulative_q5_minus_q1",
        chapter="chapter5",
        asset_kind="figure",
        output_stem="ch5_portfolio_cumulative_q5_minus_q1",
        caption_stub="Cumulative Q5-Q1 high-minus-low negativity portfolio returns by text signal.",
        notes_stub="The stored convention is Q5 - Q1; Q5 is most negative filings and Q1 is least negative filings. Separate Q1 and Q5 legs are not stored.",
        sample_contract_id="portfolio_diagnostics",
        builder_id="chapter5_portfolio_cumulative_q5_minus_q1",
        required_artifacts=(
            ArtifactRequirement(
                logical_name="trading_strategy_monthly_returns",
                run_family=RUN_FAMILY_LM2011_POST_REFINITIV,
                artifact_key=ARTIFACT_KEY_TRADING_STRATEGY_MONTHLY_RETURNS,
                required=False,
                required_columns=("portfolio_month", "sort_signal_name", "long_short_return"),
            ),
        ),
    ),
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
        asset_id="ch5_extension_c0_fit_summary",
        chapter="chapter5",
        asset_kind="table",
        output_stem="ch5_extension_c0_fit_summary",
        caption_stub="C0 extension fit summary for dictionary, FinBERT, and joint models.",
        notes_stub=(
            "Rows report Item 1A and Item 7 filing-period excess-return adjusted R2 values for the "
            "no-ownership C0 specification. The table includes replication and extended dictionary families."
        ),
        sample_contract_id="common_success_comparison",
        builder_id="chapter5_extension_c0_fit_summary",
        required_artifacts=(
            ArtifactRequirement(
                logical_name="extension_fit_summary",
                run_family=RUN_FAMILY_LM2011_EXTENSION,
                artifact_key=ARTIFACT_KEY_EXTENSION_FIT_SUMMARY,
                required_columns=(
                    "text_scope",
                    "outcome_name",
                    "control_set_id",
                    "dictionary_family_source",
                    "specification_name",
                    "signal_name",
                    "n_quarters",
                    "total_n_obs",
                    "mean_quarter_n",
                    "weighted_avg_adj_r2",
                    "equal_quarter_avg_adj_r2",
                    "common_success_policy",
                    "estimator_status",
                ),
            ),
        ),
    ),
    AssetSpec(
        asset_id="ch5_extension_c0_fit_comparisons",
        chapter="chapter5",
        asset_kind="table",
        output_stem="ch5_extension_c0_fit_comparisons",
        caption_stub="C0 extension fit-comparison deltas for dictionary, FinBERT, and joint models.",
        notes_stub=(
            "Rows report adjusted-R2 differences for Item 1A and Item 7 no-ownership C0 models. "
            "P-values are Newey-West tests on quarterly adjusted-R2 deltas."
        ),
        sample_contract_id="common_success_comparison",
        builder_id="chapter5_extension_c0_fit_comparisons",
        required_artifacts=(
            ArtifactRequirement(
                logical_name="extension_fit_comparisons",
                run_family=RUN_FAMILY_LM2011_EXTENSION,
                artifact_key=ARTIFACT_KEY_EXTENSION_FIT_COMPARISONS,
                required_columns=(
                    "text_scope",
                    "outcome_name",
                    "control_set_id",
                    "dictionary_family_source",
                    "comparison_name",
                    "n_quarters",
                    "total_n_obs",
                    "mean_quarter_n",
                    "weighted_avg_delta_adj_r2",
                    "equal_quarter_avg_delta_adj_r2",
                    "nw_t_stat_delta_adj_r2",
                    "nw_p_value_delta_adj_r2",
                    "common_success_policy",
                    "estimator_status",
                ),
            ),
        ),
    ),
    AssetSpec(
        asset_id="ch5_extension_fit_delta_path",
        chapter="chapter5",
        asset_kind="figure",
        output_stem="ch5_extension_fit_delta_path",
        caption_stub="Quarterly C0 adjusted-R2 deltas for extension dictionary and FinBERT fit comparisons.",
        notes_stub="Figure uses replication-family Item 1A and Item 7 filing-period excess-return fit-difference quarters under the common-success policy.",
        sample_contract_id="common_success_comparison",
        builder_id="chapter5_extension_fit_delta_path",
        required_artifacts=(
            ArtifactRequirement(
                logical_name="extension_fit_difference_quarterly",
                run_family=RUN_FAMILY_LM2011_EXTENSION,
                artifact_key=ARTIFACT_KEY_EXTENSION_FIT_DIFFERENCE_QUARTERLY,
                required_columns=(
                    "text_scope",
                    "outcome_name",
                    "control_set_id",
                    "dictionary_family_source",
                    "comparison_name",
                    "quarter_start",
                    "n_obs",
                    "delta_adj_r2",
                    "common_success_policy",
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
        asset_id="ch5_score_drift_by_year",
        chapter="chapter5",
        asset_kind="figure",
        output_stem="ch5_score_drift_by_year",
        caption_stub="Yearly mean and median LM2011 and FinBERT negative-tone scores for Item 1A and Item 7.",
        notes_stub="Scores are aggregated from the replication-family extension analysis panel before plotting.",
        sample_contract_id="score_drift_diagnostics",
        builder_id="chapter5_score_drift_by_year",
        required_artifacts=(
            ArtifactRequirement(
                logical_name="extension_analysis_panel",
                run_family=RUN_FAMILY_LM2011_EXTENSION,
                artifact_key=ARTIFACT_KEY_EXTENSION_ANALYSIS_PANEL,
                relative_subdir="replication",
                required_columns=(
                    "doc_id",
                    "filing_date",
                    "text_scope",
                    "lm_negative_tfidf",
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
    AssetSpec(
        asset_id="ch5_matched_dictionary_finbert_coefficients_full",
        chapter="chapter5",
        asset_kind="table",
        output_stem="ch5_matched_dictionary_finbert_coefficients_full",
        caption_stub="Matched dictionary-versus-FinBERT signal coefficient surface.",
        notes_stub="Rows report dictionary-only, FinBERT-only, and joint signal coefficients for Item 1A and Item 7 on the matched extension sample.",
        sample_contract_id="matched_dictionary_finbert_coefficients",
        builder_id="chapter5_matched_dictionary_finbert_coefficients_full",
        required_artifacts=(
            ArtifactRequirement(
                logical_name="extension_results",
                run_family=RUN_FAMILY_LM2011_EXTENSION,
                artifact_key=ARTIFACT_KEY_EXTENSION_RESULTS,
                relative_subdir="replication",
                required_columns=(
                    "text_scope",
                    "specification_name",
                    "coefficient_name",
                    "signal_name",
                    "estimate",
                    "standard_error",
                    "t_stat",
                    "p_value",
                    "n_obs",
                    "n_quarters",
                    "estimator_status",
                ),
            ),
        ),
    ),
    AssetSpec(
        asset_id="ch5_fama_macbeth_skipped_quarter_diagnostics",
        chapter="chapter5",
        asset_kind="table",
        output_stem="ch5_fama_macbeth_skipped_quarter_diagnostics",
        caption_stub="Fama-MacBeth skipped-quarter diagnostics.",
        notes_stub="Rows aggregate stored skipped-quarter diagnostics across LM2011, extension, and FinBERT robustness outputs.",
        sample_contract_id="skipped_quarter_diagnostics",
        builder_id="chapter5_fama_macbeth_skipped_quarter_diagnostics",
        required_artifacts=(
            ArtifactRequirement(
                logical_name="table_iv_skipped_quarters",
                run_family=RUN_FAMILY_LM2011_POST_REFINITIV,
                artifact_key=ARTIFACT_KEY_TABLE_IV_SKIPPED_QUARTERS,
                required=False,
            ),
            ArtifactRequirement(
                logical_name="table_v_skipped_quarters",
                run_family=RUN_FAMILY_LM2011_POST_REFINITIV,
                artifact_key=ARTIFACT_KEY_TABLE_V_SKIPPED_QUARTERS,
                required=False,
            ),
            ArtifactRequirement(
                logical_name="table_vi_skipped_quarters",
                run_family=RUN_FAMILY_LM2011_POST_REFINITIV,
                artifact_key=ARTIFACT_KEY_TABLE_VI_SKIPPED_QUARTERS,
                required=False,
            ),
            ArtifactRequirement(
                logical_name="extension_fit_skipped_quarters",
                run_family=RUN_FAMILY_LM2011_EXTENSION,
                artifact_key=ARTIFACT_KEY_EXTENSION_FIT_SKIPPED_QUARTERS,
                relative_subdir="replication",
                required=False,
            ),
            ArtifactRequirement(
                logical_name="robustness_existing_scale_fit_skipped_quarters",
                run_family=RUN_FAMILY_FINBERT_ROBUSTNESS,
                artifact_key=ARTIFACT_KEY_FINBERT_ROBUSTNESS_EXISTING_SCALE_FIT_SKIPPED_QUARTERS,
                required=False,
            ),
            ArtifactRequirement(
                logical_name="robustness_tail_fit_skipped_quarters",
                run_family=RUN_FAMILY_FINBERT_ROBUSTNESS,
                artifact_key=ARTIFACT_KEY_FINBERT_ROBUSTNESS_TAIL_FIT_SKIPPED_QUARTERS,
                required=False,
            ),
            ArtifactRequirement(
                logical_name="robustness_quantile_fit_skipped_quarters",
                run_family=RUN_FAMILY_FINBERT_ROBUSTNESS,
                artifact_key=ARTIFACT_KEY_FINBERT_ROBUSTNESS_QUANTILE_FIT_SKIPPED_QUARTERS,
                required=False,
            ),
        ),
    ),
    AssetSpec(
        asset_id="ch5_alternative_signal_robustness_full_grid",
        chapter="chapter5",
        asset_kind="table",
        output_stem="ch5_alternative_signal_robustness_full_grid",
        caption_stub="Alternative FinBERT signal robustness grid.",
        notes_stub="Rows stack stored existing-scale, tail-signal, and quantile-signal robustness coefficients.",
        sample_contract_id="finbert_robustness_full_grid",
        builder_id="chapter5_alternative_signal_robustness_full_grid",
        required_artifacts=(
            ArtifactRequirement(
                logical_name="existing_scale_coefficients",
                run_family=RUN_FAMILY_FINBERT_ROBUSTNESS,
                artifact_key=ARTIFACT_KEY_FINBERT_ROBUSTNESS_EXISTING_SCALE_COEFFICIENTS,
                required_columns=("variant_id", "text_scope", "coefficient_name", "estimate", "t_stat"),
            ),
            ArtifactRequirement(
                logical_name="tail_coefficients",
                run_family=RUN_FAMILY_FINBERT_ROBUSTNESS,
                artifact_key=ARTIFACT_KEY_FINBERT_ROBUSTNESS_TAIL_COEFFICIENTS,
                required_columns=("variant_id", "text_scope", "coefficient_name", "estimate", "t_stat"),
            ),
            ArtifactRequirement(
                logical_name="quantile_coefficients",
                run_family=RUN_FAMILY_FINBERT_ROBUSTNESS,
                artifact_key=ARTIFACT_KEY_FINBERT_ROBUSTNESS_QUANTILE_COEFFICIENTS,
                required_columns=("variant_id", "text_scope", "coefficient_name", "estimate", "t_stat"),
            ),
        ),
    ),
    AssetSpec(
        asset_id="ch5_full_controls_coefficient_appendix",
        chapter="chapter5",
        asset_kind="table",
        output_stem="ch5_full_controls_coefficient_appendix",
        caption_stub="Full controls coefficient appendix for stored regression outputs.",
        notes_stub="Rows report C0 control coefficients from selected LM2011 and FinBERT regression artifacts.",
        sample_contract_id="controls_coefficient_appendix",
        builder_id="chapter5_full_controls_coefficient_appendix",
        required_artifacts=(
            ArtifactRequirement(
                logical_name="table_iv_results_no_ownership",
                run_family=RUN_FAMILY_LM2011_POST_REFINITIV,
                artifact_key=ARTIFACT_KEY_TABLE_IV_RESULTS_NO_OWNERSHIP,
                required_columns=("coefficient_name", "estimate", "t_stat"),
            ),
            ArtifactRequirement(
                logical_name="extension_results",
                run_family=RUN_FAMILY_LM2011_EXTENSION,
                artifact_key=ARTIFACT_KEY_EXTENSION_RESULTS,
                relative_subdir="replication",
                required_columns=("coefficient_name", "estimate", "t_stat"),
            ),
            ArtifactRequirement(
                logical_name="existing_scale_coefficients",
                run_family=RUN_FAMILY_FINBERT_ROBUSTNESS,
                artifact_key=ARTIFACT_KEY_FINBERT_ROBUSTNESS_EXISTING_SCALE_COEFFICIENTS,
                required_columns=("coefficient_name", "estimate", "t_stat"),
            ),
        ),
    ),
    AssetSpec(
        asset_id="ch5_text_score_control_correlation_matrix",
        chapter="chapter5",
        asset_kind="table",
        output_stem="ch5_text_score_control_correlation_matrix",
        caption_stub="Correlation matrix for selected text scores and controls.",
        notes_stub="Correlations are computed from the stored full-10-K benchmark regression panel.",
        sample_contract_id="correlation_matrix",
        builder_id="chapter5_text_score_control_correlation_matrix",
        required_artifacts=(
            ArtifactRequirement(
                logical_name="return_regression_panel_full_10k",
                run_family=RUN_FAMILY_LM2011_POST_REFINITIV,
                artifact_key=ARTIFACT_KEY_RETURN_REGRESSION_PANEL_FULL_10K,
                required_columns=(
                    "lm_negative_prop",
                    "lm_negative_tfidf",
                    "h4n_inf_prop",
                    "h4n_inf_tfidf",
                    "log_size",
                    "log_book_to_market",
                    "log_share_turnover",
                    "pre_ffalpha",
                    "nasdaq_dummy",
                ),
            ),
        ),
    ),
    AssetSpec(
        asset_id="ch5_research_question_evidence_map",
        chapter="chapter5",
        asset_kind="table",
        output_stem="ch5_research_question_evidence_map",
        caption_stub="Research-question evidence map for generated Chapter 5 assets.",
        notes_stub="Rows map research questions to generated evidence and conservative reporting guardrails.",
        sample_contract_id="evidence_map",
        builder_id="chapter5_research_question_evidence_map",
        required_artifacts=(),
    ),
)

from __future__ import annotations

from thesis_assets.config.constants import ARTIFACT_KEY_EXTENSION_ANALYSIS_PANEL
from thesis_assets.config.constants import ARTIFACT_KEY_EXTENSION_CONTROL_LADDER
from thesis_assets.config.constants import ARTIFACT_KEY_EXTENSION_DICTIONARY_SURFACE
from thesis_assets.config.constants import ARTIFACT_KEY_EXTENSION_FINBERT_SURFACE
from thesis_assets.config.constants import ARTIFACT_KEY_EXTENSION_SAMPLE_LOSS
from thesis_assets.config.constants import ARTIFACT_KEY_FINBERT_ITEM_FEATURES_LONG
from thesis_assets.config.constants import ARTIFACT_KEY_FINBERT_MODEL_INFERENCE_YEARLY_SUMMARY
from thesis_assets.config.constants import ARTIFACT_KEY_RETURN_REGRESSION_PANEL_FULL_10K
from thesis_assets.config.constants import ARTIFACT_KEY_RETURN_REGRESSION_PANEL_MDA
from thesis_assets.config.constants import ARTIFACT_KEY_TABLE_I_SAMPLE_CREATION
from thesis_assets.config.constants import RUN_FAMILY_FINBERT_RUN
from thesis_assets.config.constants import RUN_FAMILY_LM2011_EXTENSION
from thesis_assets.config.constants import RUN_FAMILY_LM2011_POST_REFINITIV
from thesis_assets.specs import ArtifactRequirement
from thesis_assets.specs import AssetSpec


ASSETS: tuple[AssetSpec, ...] = (
    AssetSpec(
        asset_id="ch4_sample_attrition_lm2011_1994_2008",
        chapter="chapter4",
        asset_kind="table",
        output_stem="ch4_sample_attrition_lm2011_1994_2008",
        caption_stub="Chapter 4 sample-selection and attrition ladder for the 1994-2008 LM2011 window.",
        notes_stub="Notes stub: keep wording near the asset spec until final thesis placement is decided.",
        sample_contract_id="raw_available",
        builder_id="chapter4_sample_attrition",
        required_artifacts=(
            ArtifactRequirement(
                logical_name="table_i_sample_creation",
                run_family=RUN_FAMILY_LM2011_POST_REFINITIV,
                artifact_key=ARTIFACT_KEY_TABLE_I_SAMPLE_CREATION,
                required_columns=(
                    "section_label",
                    "section_order",
                    "row_order",
                    "display_label",
                    "sample_size_value",
                ),
            ),
        ),
    ),
    AssetSpec(
        asset_id="ch4_sample_funnel_raw_to_final_lm2011",
        chapter="chapter4",
        asset_kind="figure",
        output_stem="ch4_sample_funnel_raw_to_final_lm2011",
        caption_stub="Chapter 4 sample-size funnel from raw 10-K filings to final LM2011 samples.",
        notes_stub="Notes stub: figure is generated from the stored LM2011 sample-creation ladder.",
        sample_contract_id="raw_available",
        builder_id="chapter4_sample_funnel",
        required_artifacts=(
            ArtifactRequirement(
                logical_name="table_i_sample_creation",
                run_family=RUN_FAMILY_LM2011_POST_REFINITIV,
                artifact_key=ARTIFACT_KEY_TABLE_I_SAMPLE_CREATION,
                required_columns=(
                    "section_label",
                    "section_order",
                    "row_order",
                    "display_label",
                    "sample_size_value",
                    "sample_size_kind",
                    "observations_removed",
                ),
            ),
        ),
    ),
    AssetSpec(
        asset_id="ch4_sample_attrition_losses_lm2011",
        chapter="chapter4",
        asset_kind="figure",
        output_stem="ch4_sample_attrition_losses_lm2011",
        caption_stub="Chapter 4 observations removed at each LM2011 sample-selection step.",
        notes_stub="Notes stub: observations removed are taken directly from the stored attrition ladder.",
        sample_contract_id="raw_available",
        builder_id="chapter4_sample_attrition_losses",
        required_artifacts=(
            ArtifactRequirement(
                logical_name="table_i_sample_creation",
                run_family=RUN_FAMILY_LM2011_POST_REFINITIV,
                artifact_key=ARTIFACT_KEY_TABLE_I_SAMPLE_CREATION,
                required_columns=(
                    "section_label",
                    "section_order",
                    "row_order",
                    "display_label",
                    "sample_size_value",
                    "observations_removed",
                ),
            ),
        ),
    ),
    AssetSpec(
        asset_id="ch4_sample_stage_bridge_lm2011",
        chapter="chapter4",
        asset_kind="figure",
        output_stem="ch4_sample_stage_bridge_lm2011",
        caption_stub="Chapter 4 bridge across the main raw filing, firm-year, and MD&A sample stages.",
        notes_stub="Notes stub: bridge stages are selected from canonical LM2011 sample-creation rows.",
        sample_contract_id="raw_available",
        builder_id="chapter4_sample_stage_bridge",
        required_artifacts=(
            ArtifactRequirement(
                logical_name="table_i_sample_creation",
                run_family=RUN_FAMILY_LM2011_POST_REFINITIV,
                artifact_key=ARTIFACT_KEY_TABLE_I_SAMPLE_CREATION,
                required_columns=(
                    "section_label",
                    "section_order",
                    "row_order",
                    "display_label",
                    "sample_size_value",
                    "sample_size_kind",
                    "observations_removed",
                ),
            ),
        ),
    ),
    AssetSpec(
        asset_id="ch4_full_10k_regression_sample_summary",
        chapter="chapter4",
        asset_kind="table",
        output_stem="ch4_full_10k_regression_sample_summary",
        caption_stub="Full-10-K benchmark regression-sample summary statistics.",
        notes_stub="Summary statistics are computed from the stored full-10-K return regression panel.",
        sample_contract_id="regression_sample_summary",
        builder_id="chapter4_full_10k_regression_sample_summary",
        required_artifacts=(
            ArtifactRequirement(
                logical_name="return_regression_panel_full_10k",
                run_family=RUN_FAMILY_LM2011_POST_REFINITIV,
                artifact_key=ARTIFACT_KEY_RETURN_REGRESSION_PANEL_FULL_10K,
                required_columns=(
                    "doc_id",
                    "KYPERMNO",
                    "filing_date",
                    "filing_period_excess_return",
                    "lm_negative_prop",
                    "lm_negative_tfidf",
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
        asset_id="ch4_extension_attrition_ladder",
        chapter="chapter4",
        asset_kind="table",
        output_stem="ch4_extension_attrition_ladder",
        caption_stub="Post-2009 Item 1A and Item 7 extension sample attrition ladder.",
        notes_stub="Rows summarize stored extension sample-loss diagnostics and matched analysis-panel counts.",
        sample_contract_id="extension_attrition",
        builder_id="chapter4_extension_attrition_ladder",
        required_artifacts=(
            ArtifactRequirement(
                logical_name="extension_sample_loss",
                run_family=RUN_FAMILY_LM2011_EXTENSION,
                artifact_key=ARTIFACT_KEY_EXTENSION_SAMPLE_LOSS,
                relative_subdir="replication",
                required_columns=(
                    "calendar_year",
                    "text_scope",
                    "n_control_set_rows",
                    "n_estimation_rows",
                ),
            ),
            ArtifactRequirement(
                logical_name="extension_analysis_panel",
                run_family=RUN_FAMILY_LM2011_EXTENSION,
                artifact_key=ARTIFACT_KEY_EXTENSION_ANALYSIS_PANEL,
                relative_subdir="replication",
                required_columns=("doc_id", "filing_date", "text_scope"),
            ),
        ),
    ),
    AssetSpec(
        asset_id="ch4_no_ownership_c0_specification",
        chapter="chapter4",
        asset_kind="table",
        output_stem="ch4_no_ownership_c0_specification",
        caption_stub="No-ownership C0 control-set specification.",
        notes_stub="Controls are reported from the stored extension control ladder and C0 regression conventions.",
        sample_contract_id="specification_metadata",
        builder_id="chapter4_no_ownership_c0_specification",
        required_artifacts=(
            ArtifactRequirement(
                logical_name="extension_control_ladder",
                run_family=RUN_FAMILY_LM2011_EXTENSION,
                artifact_key=ARTIFACT_KEY_EXTENSION_CONTROL_LADDER,
                relative_subdir="replication",
                required_columns=("control_set_id", "control_columns", "includes_ownership_control"),
            ),
        ),
    ),
    AssetSpec(
        asset_id="ch4_ownership_analyst_coverage_diagnostics",
        chapter="chapter4",
        asset_kind="table",
        output_stem="ch4_ownership_analyst_coverage_diagnostics",
        caption_stub="Ownership and analyst coverage diagnostics for excluded coverage-dependent surfaces.",
        notes_stub="Ownership support is computed from the extension analysis panel; analyst coverage is summarized from available Refinitiv normalized/request artifacts.",
        sample_contract_id="coverage_diagnostics",
        builder_id="chapter4_ownership_analyst_coverage_diagnostics",
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
                    "ownership_proxy_available",
                    "common_support_flag_ownership",
                ),
            ),
        ),
    ),
    AssetSpec(
        asset_id="ch4_ownership_coverage_by_year",
        chapter="chapter4",
        asset_kind="figure",
        output_stem="ch4_ownership_coverage_by_year",
        caption_stub="Yearly ownership-proxy coverage for the unrestricted extension panel.",
        notes_stub="Rates are computed from the Item 1A and Item 7 extension analysis panel and motivate keeping C0 central.",
        sample_contract_id="coverage_diagnostics",
        builder_id="chapter4_ownership_coverage_by_year",
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
                    "ownership_proxy_available",
                    "common_support_flag_ownership",
                ),
            ),
        ),
    ),
    AssetSpec(
        asset_id="ch4_item_cleaning_eligibility_diagnostics",
        chapter="chapter4",
        asset_kind="table",
        output_stem="ch4_item_cleaning_eligibility_diagnostics",
        caption_stub="Item extraction and cleaned-scope eligibility diagnostics.",
        notes_stub="Diagnostics come from the FinBERT preprocessing cleaning audit for Item 1A and Item 7.",
        sample_contract_id="cleaning_diagnostics",
        builder_id="chapter4_item_cleaning_eligibility_diagnostics",
        required_artifacts=(
            ArtifactRequirement(
                logical_name="model_inference_yearly_summary",
                run_family=RUN_FAMILY_FINBERT_RUN,
                artifact_key=ARTIFACT_KEY_FINBERT_MODEL_INFERENCE_YEARLY_SUMMARY,
                required_columns=("filing_year", "sentence_rows", "doc_rows"),
            ),
        ),
    ),
    AssetSpec(
        asset_id="ch4_item_cleaning_quality_by_year",
        chapter="chapter4",
        asset_kind="figure",
        output_stem="ch4_item_cleaning_quality_by_year",
        caption_stub="Yearly Item 1A and Item 7 extraction and cleaning quality diagnostics.",
        notes_stub="Figure uses stored FinBERT preprocessing cleaning diagnostics; no filing text is rescanned.",
        sample_contract_id="cleaning_diagnostics",
        builder_id="chapter4_item_cleaning_quality_by_year",
        required_artifacts=(
            ArtifactRequirement(
                logical_name="model_inference_yearly_summary",
                run_family=RUN_FAMILY_FINBERT_RUN,
                artifact_key=ARTIFACT_KEY_FINBERT_MODEL_INFERENCE_YEARLY_SUMMARY,
                required_columns=("filing_year", "sentence_rows", "doc_rows"),
            ),
        ),
    ),
    AssetSpec(
        asset_id="ch4_dictionary_provenance_summary",
        chapter="chapter4",
        asset_kind="table",
        output_stem="ch4_dictionary_provenance_summary",
        caption_stub="Dictionary provenance and score-family summary.",
        notes_stub="Dictionary-family metadata is summarized from generated LM2011 dictionary-family files and run manifests.",
        sample_contract_id="dictionary_provenance",
        builder_id="chapter4_dictionary_provenance_summary",
        required_artifacts=(
            ArtifactRequirement(
                logical_name="table_i_sample_creation",
                run_family=RUN_FAMILY_LM2011_POST_REFINITIV,
                artifact_key=ARTIFACT_KEY_TABLE_I_SAMPLE_CREATION,
                required_columns=("section_label",),
            ),
            ArtifactRequirement(
                logical_name="extension_dictionary_surface",
                run_family=RUN_FAMILY_LM2011_EXTENSION,
                artifact_key=ARTIFACT_KEY_EXTENSION_DICTIONARY_SURFACE,
                relative_subdir="replication",
                required_columns=("dictionary_family", "text_scope"),
            ),
        ),
    ),
    AssetSpec(
        asset_id="ch4_finbert_inference_manifest_summary",
        chapter="chapter4",
        asset_kind="table",
        output_stem="ch4_finbert_inference_manifest_summary",
        caption_stub="FinBERT inference manifest and yearly processing summary.",
        notes_stub="Rows summarize the stored FinBERT run manifest and yearly inference outputs.",
        sample_contract_id="finbert_manifest",
        builder_id="chapter4_finbert_inference_manifest_summary",
        required_artifacts=(
            ArtifactRequirement(
                logical_name="model_inference_yearly_summary",
                run_family=RUN_FAMILY_FINBERT_RUN,
                artifact_key=ARTIFACT_KEY_FINBERT_MODEL_INFERENCE_YEARLY_SUMMARY,
                required_columns=("filing_year", "status", "sentence_rows", "item_feature_rows", "doc_rows"),
            ),
        ),
    ),
    AssetSpec(
        asset_id="ch4_finbert_segment_token_diagnostics",
        chapter="chapter4",
        asset_kind="figure",
        output_stem="ch4_finbert_segment_token_diagnostics",
        caption_stub="FinBERT segment and token-count diagnostics by filing year and item scope.",
        notes_stub="Summary statistics are aggregated from item_features_long before plotting to keep the build memory-bounded.",
        sample_contract_id="finbert_token_diagnostics",
        builder_id="chapter4_finbert_segment_token_diagnostics",
        required_artifacts=(
            ArtifactRequirement(
                logical_name="item_features_long",
                run_family=RUN_FAMILY_FINBERT_RUN,
                artifact_key=ARTIFACT_KEY_FINBERT_ITEM_FEATURES_LONG,
                required_columns=(
                    "doc_id",
                    "filing_year",
                    "text_scope",
                    "sentence_count",
                    "finbert_segment_count",
                    "finbert_token_count_512_sum",
                ),
            ),
        ),
    ),
    AssetSpec(
        asset_id="ch4_score_family_descriptive_statistics",
        chapter="chapter4",
        asset_kind="table",
        output_stem="ch4_score_family_descriptive_statistics",
        caption_stub="Descriptive statistics by text scope and score family.",
        notes_stub="Statistics summarize stored LM2011 and FinBERT score surfaces for full-10-K, MD&A, Item 1A, and Item 7 scopes.",
        sample_contract_id="score_descriptives",
        builder_id="chapter4_score_family_descriptive_statistics",
        required_artifacts=(
            ArtifactRequirement(
                logical_name="return_regression_panel_full_10k",
                run_family=RUN_FAMILY_LM2011_POST_REFINITIV,
                artifact_key=ARTIFACT_KEY_RETURN_REGRESSION_PANEL_FULL_10K,
                required_columns=("text_scope", "lm_negative_prop", "lm_negative_tfidf"),
            ),
            ArtifactRequirement(
                logical_name="return_regression_panel_mda",
                run_family=RUN_FAMILY_LM2011_POST_REFINITIV,
                artifact_key=ARTIFACT_KEY_RETURN_REGRESSION_PANEL_MDA,
                required_columns=("text_scope", "lm_negative_prop", "lm_negative_tfidf"),
            ),
            ArtifactRequirement(
                logical_name="extension_dictionary_surface",
                run_family=RUN_FAMILY_LM2011_EXTENSION,
                artifact_key=ARTIFACT_KEY_EXTENSION_DICTIONARY_SURFACE,
                relative_subdir="replication",
                required_columns=("text_scope", "lm_negative_prop", "lm_negative_tfidf"),
            ),
            ArtifactRequirement(
                logical_name="extension_finbert_surface",
                run_family=RUN_FAMILY_LM2011_EXTENSION,
                artifact_key=ARTIFACT_KEY_EXTENSION_FINBERT_SURFACE,
                relative_subdir="replication",
                required_columns=("text_scope", "finbert_neg_prob_lenw_mean", "finbert_net_negative_lenw_mean"),
            ),
        ),
    ),
    AssetSpec(
        asset_id="ch4_variable_definitions",
        chapter="chapter4",
        asset_kind="table",
        output_stem="ch4_variable_definitions",
        caption_stub="Variable definitions and reporting conventions for the empirical thesis assets.",
        notes_stub="Definitions summarize stored artifact conventions, including the portfolio Q5-Q1 sign rule.",
        sample_contract_id="methodology_definitions",
        builder_id="chapter4_variable_definitions",
        required_artifacts=(),
    ),
)

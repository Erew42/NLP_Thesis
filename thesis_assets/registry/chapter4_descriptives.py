from __future__ import annotations

from thesis_assets.config.constants import ARTIFACT_KEY_TABLE_I_SAMPLE_CREATION
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
)

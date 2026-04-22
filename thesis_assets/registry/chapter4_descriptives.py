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
)

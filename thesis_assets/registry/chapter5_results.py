from __future__ import annotations

from thesis_assets.config.constants import ARTIFACT_KEY_EXTENSION_DICTIONARY_SURFACE
from thesis_assets.config.constants import ARTIFACT_KEY_EXTENSION_FINBERT_SURFACE
from thesis_assets.config.constants import ARTIFACT_KEY_EXTENSION_FIT_SUMMARY
from thesis_assets.config.constants import RUN_FAMILY_LM2011_EXTENSION
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
)

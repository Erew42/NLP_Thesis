from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Mapping
from typing import Sequence

import polars as pl


def _resolve_repo_root() -> Path:
    script_path = Path(__file__).resolve()
    for candidate in (script_path.parent, *script_path.parents):
        if (candidate / "src" / "thesis_pkg" / "pipeline.py").exists() and (candidate / "thesis_assets").exists():
            return candidate
    raise RuntimeError("Could not resolve repository root from tools/run_submission_pipeline.py")


ROOT = _resolve_repo_root()
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from thesis_assets.config.constants import RUN_FAMILY_FINBERT_ROBUSTNESS
from thesis_assets.config.constants import RUN_FAMILY_FINBERT_RUN
from thesis_assets.config.constants import RUN_FAMILY_LM2011_EXTENSION
from thesis_assets.config.constants import RUN_FAMILY_LM2011_EXTENSION_FINBERT_VISIBLE_PREFIX
from thesis_assets.config.constants import RUN_FAMILY_LM2011_NW_LAG_SENSITIVITY
from thesis_assets.config.constants import RUN_FAMILY_LM2011_POST_REFINITIV
from thesis_assets.submission_lock import DEFAULT_FINBERT_MODEL_NAME
from thesis_assets.submission_lock import DEFAULT_FINBERT_REVISION
from thesis_assets.submission_lock import build_submission_lock_payload
from thesis_assets.submission_lock import validate_submission_lock
from thesis_assets.submission_lock import write_submission_lock
from thesis_pkg.benchmarking.manifest_contracts import semantic_file_fingerprint


SUBMISSION_PROFILE = "SUBMISSION"
DEFAULT_RUN_ID = "submission_rerun"
DEFAULT_FINBERT_RUN_NAME = "submission_finbert"
DEFAULT_NW_LAGS: tuple[int, ...] = (1, 2, 3, 4)
STAGE_VALIDATE = "validate"
STAGE_LM2011 = "lm2011"
STAGE_NW_SENSITIVITY = "nw-sensitivity"
STAGE_FINBERT_PREPROCESS = "finbert-preprocess"
STAGE_FINBERT_ANALYSIS = "finbert-analysis"
STAGE_EXTENSION = "extension"
STAGE_VISIBLE_PREFIX_EXTENSION = "visible-prefix-extension"
STAGE_FINBERT_ROBUSTNESS = "finbert-robustness"
STAGE_THESIS_ASSETS = "thesis-assets"
STAGE_ALL = "all"
ORDERED_STAGES: tuple[str, ...] = (
    STAGE_VALIDATE,
    STAGE_LM2011,
    STAGE_NW_SENSITIVITY,
    STAGE_FINBERT_PREPROCESS,
    STAGE_FINBERT_ANALYSIS,
    STAGE_EXTENSION,
    STAGE_VISIBLE_PREFIX_EXTENSION,
    STAGE_FINBERT_ROBUSTNESS,
    STAGE_THESIS_ASSETS,
)
STAGE_CHOICES: tuple[str, ...] = (*ORDERED_STAGES, STAGE_ALL)

REQUIRED_CCM_PARQUETS: tuple[str, ...] = (
    "filingdates.parquet",
    "balancesheetquarterly.parquet",
    "incomestatementquarterly.parquet",
    "perioddescriptorquarterly.parquet",
    "balancesheetindustrialannual.parquet",
    "incomestatementindustrialannual.parquet",
    "perioddescriptorannual.parquet",
    "fiscalmarketdataannual.parquet",
    "companyhistory.parquet",
    "companydescription.parquet",
    "sfz_mth.parquet",
    "final_flagged_data_compdesc_added.parquet",
)
REQUIRED_ADDITIONAL_DATA_FILES: tuple[str, ...] = (
    "F-F_Research_Data_Factors_daily.csv",
    "F-F_Research_Data_Factors.csv",
    "F-F_Momentum_Factor.csv",
    "FF_Siccodes_48_Industries.txt",
    "Fin-Neg.txt",
    "Fin-Pos.txt",
    "Fin-Unc.txt",
    "Fin-Lit.txt",
    "MW-Strong.txt",
    "MW-Weak.txt",
    "Harvard_IV_NEG_Inf.txt",
)
MASTER_DICTIONARY_CANDIDATES: tuple[str, ...] = (
    "LM2011_MasterDictionary.txt",
    "Loughran-McDonald_MasterDictionary_1993-2024.csv",
)
EXTENDED_MASTER_DICTIONARY_FILENAME = "Loughran-McDonald_MasterDictionary_1993-2024.csv"
PACKAGE_MANIFEST_FILENAME = "submission_package_manifest.json"
PACKAGE_MANIFEST_SCHEMA_VERSION = 1
PACKAGE_MANIFEST_PATH_SEMANTICS = "relative_to_submission_root"
PACKAGE_MANIFEST_REQUIRED_GROUPS: tuple[str, ...] = (
    "sec_year_merged",
    "sec_items_analysis",
    "sec_ccm_matched_clean",
    "ccm_crsp_compustat",
    "refinitiv_finalized",
    "lm2011_additional_data",
)
ITEMS_ANALYSIS_REQUIRED_COLUMNS: tuple[str, ...] = (
    "doc_id",
    "item_id",
    "full_text",
    "item_status",
    "exists_by_regime",
    "boundary_authority_status",
)
ITEMS_ANALYSIS_REQUIRED_COLUMN_GROUPS: Mapping[str, tuple[str, ...]] = {
    "cik": ("cik_10", "cik"),
    "accession": ("accession_nodash", "accession_number"),
    "filing_date": ("filing_date", "file_date_filename"),
    "document_type": ("document_type_filename", "document_type"),
}


class SubmissionPipelineError(RuntimeError):
    pass


@dataclass(frozen=True)
class ArtifactOverrideConfig:
    artifact_key: str
    path: Path
    reason: str


@dataclass(frozen=True)
class FinbertSubmissionConfig:
    batch_profile: str = "baseline"
    device: str | None = None
    model_name: str = DEFAULT_FINBERT_MODEL_NAME
    model_revision: str = DEFAULT_FINBERT_REVISION
    tokenizer_revision: str = DEFAULT_FINBERT_REVISION


@dataclass(frozen=True)
class SubmissionPipelineConfig:
    run_id: str = DEFAULT_RUN_ID
    years: tuple[int, ...] | None = None
    nw_lags: tuple[int, ...] = DEFAULT_NW_LAGS
    finbert: FinbertSubmissionConfig = FinbertSubmissionConfig()
    artifact_overrides: tuple[ArtifactOverrideConfig, ...] = ()


@dataclass(frozen=True)
class SubmissionProfile:
    submission_root: Path
    config: SubmissionPipelineConfig

    @property
    def data_root(self) -> Path:
        return self.submission_root / "data"

    @property
    def sec_root(self) -> Path:
        return self.data_root / "sec"

    @property
    def year_merged_dir(self) -> Path:
        return self.sec_root / "year_merged"

    @property
    def items_analysis_dir(self) -> Path:
        return self.sec_root / "items_analysis"

    @property
    def matched_clean_path(self) -> Path:
        return self.sec_root / "sec_ccm_matched_clean.parquet"

    @property
    def ccm_dir(self) -> Path:
        return self.data_root / "ccm_crsp_compustat"

    @property
    def daily_panel_path(self) -> Path:
        return self.ccm_dir / "final_flagged_data_compdesc_added.parquet"

    @property
    def monthly_stock_path(self) -> Path:
        return self.ccm_dir / "sfz_mth.parquet"

    @property
    def additional_data_dir(self) -> Path:
        return self.data_root / "LM2011_additional_data"

    @property
    def refinitiv_dir(self) -> Path:
        return self.data_root / "refinitiv_finalized"

    @property
    def doc_ownership_path(self) -> Path:
        return self.refinitiv_dir / "refinitiv_lm2011_doc_ownership.parquet"

    @property
    def doc_analyst_selected_path(self) -> Path:
        return self.refinitiv_dir / "refinitiv_doc_analyst_selected.parquet"

    @property
    def analysis_outputs_root(self) -> Path:
        return self.submission_root / "analysis_outputs"

    @property
    def lm2011_output_dir(self) -> Path:
        return self.analysis_outputs_root / "lm2011_post_refinitiv"

    @property
    def nw_sensitivity_dir(self) -> Path:
        return self.analysis_outputs_root / "lm2011_nw_lag_sensitivity"

    @property
    def finbert_output_dir(self) -> Path:
        return self.analysis_outputs_root / "finbert_item_analysis"

    @property
    def finbert_preprocessing_run_dir(self) -> Path:
        return self.finbert_output_dir / "_staged_intermediates" / f"{DEFAULT_FINBERT_RUN_NAME}_sentence_preprocessing"

    @property
    def finbert_analysis_run_dir(self) -> Path:
        return self.finbert_output_dir / DEFAULT_FINBERT_RUN_NAME

    @property
    def extension_output_dir(self) -> Path:
        return self.analysis_outputs_root / "lm2011_extension"

    @property
    def visible_prefix_extension_output_dir(self) -> Path:
        return self.analysis_outputs_root / "lm2011_extension_finbert_visible_prefix"

    @property
    def finbert_robustness_output_dir(self) -> Path:
        return self.analysis_outputs_root / "finbert_robustness"

    @property
    def local_work_root(self) -> Path:
        return self.analysis_outputs_root / "_work"

    @property
    def thesis_assets_output_root(self) -> Path:
        return self.submission_root / "output" / "thesis_assets"

    @property
    def pipeline_manifest_path(self) -> Path:
        return self.submission_root / "submission_pipeline_manifest.json"

    @property
    def package_manifest_path(self) -> Path:
        return self.submission_root / PACKAGE_MANIFEST_FILENAME

    @property
    def submission_lock_path(self) -> Path:
        return self.submission_root / "submission_lock.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict supervisor rerun entrypoint for submitted first-stage parquet packages."
    )
    parser.add_argument("--submission-root", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGE_CHOICES, default=STAGE_ALL)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    profile = resolve_submission_profile(
        submission_root=args.submission_root,
        config_path=args.config,
        run_id_override=args.run_id,
    )
    if args.stage == STAGE_ALL:
        return run_all_stages(profile, force=bool(args.force), dry_run=bool(args.dry_run), config_path=args.config)
    return run_stage(profile, args.stage, force=bool(args.force), dry_run=bool(args.dry_run))


def resolve_submission_profile(
    *,
    submission_root: Path,
    config_path: Path | None = None,
    run_id_override: str | None = None,
) -> SubmissionProfile:
    root = Path(submission_root).expanduser().resolve()
    config = load_submission_pipeline_config(root, config_path=config_path)
    if run_id_override:
        config = SubmissionPipelineConfig(
            run_id=run_id_override,
            years=config.years,
            nw_lags=config.nw_lags,
            finbert=config.finbert,
            artifact_overrides=config.artifact_overrides,
        )
    profile = SubmissionProfile(submission_root=root, config=config)
    _assert_profile_paths_inside_submission_root(profile)
    return profile


def load_submission_pipeline_config(
    submission_root: Path,
    *,
    config_path: Path | None,
) -> SubmissionPipelineConfig:
    if config_path is None:
        default_path = submission_root / "submission_pipeline_config.json"
        if not default_path.exists():
            return SubmissionPipelineConfig()
        resolved_config_path = default_path.resolve()
    else:
        raw_config_path = Path(config_path).expanduser()
        resolved_config_path = (
            raw_config_path.resolve()
            if raw_config_path.is_absolute()
            else (submission_root / raw_config_path).resolve()
        )
    if not resolved_config_path.is_relative_to(submission_root):
        raise SubmissionPipelineError(f"Config path must be inside submission_root: {resolved_config_path}")
    payload = json.loads(resolved_config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SubmissionPipelineError("Submission pipeline config must be a JSON object.")
    unknown = sorted(set(payload) - {"run_id", "years", "nw_lags", "finbert", "artifact_overrides"})
    if unknown:
        raise SubmissionPipelineError(f"Unknown submission pipeline config keys: {unknown}")

    finbert_payload = payload.get("finbert") or {}
    if not isinstance(finbert_payload, dict):
        raise SubmissionPipelineError("finbert config must be a JSON object when provided.")
    unknown_finbert = sorted(
        set(finbert_payload)
        - {"batch_profile", "device", "model_name", "model_revision", "tokenizer_revision"}
    )
    if unknown_finbert:
        raise SubmissionPipelineError(f"Unknown finbert config keys: {unknown_finbert}")

    batch_profile = _str_or_default(
        finbert_payload.get("batch_profile"),
        "baseline",
        label="finbert.batch_profile",
    )
    if batch_profile not in {"small", "baseline", "large", "xlarge"}:
        raise SubmissionPipelineError(
            "finbert.batch_profile must be one of small, baseline, large, xlarge."
        )

    return SubmissionPipelineConfig(
        run_id=_str_or_default(payload.get("run_id"), DEFAULT_RUN_ID, label="run_id"),
        years=_normalize_years(payload.get("years")),
        nw_lags=_normalize_nw_lags(payload.get("nw_lags", list(DEFAULT_NW_LAGS))),
        finbert=FinbertSubmissionConfig(
            batch_profile=batch_profile,
            device=_optional_str(finbert_payload.get("device"), label="finbert.device"),
            model_name=_str_or_default(finbert_payload.get("model_name"), DEFAULT_FINBERT_MODEL_NAME, label="finbert.model_name"),
            model_revision=_str_or_default(
                finbert_payload.get("model_revision"),
                DEFAULT_FINBERT_REVISION,
                label="finbert.model_revision",
            ),
            tokenizer_revision=_str_or_default(
                finbert_payload.get("tokenizer_revision"),
                DEFAULT_FINBERT_REVISION,
                label="finbert.tokenizer_revision",
            ),
        ),
        artifact_overrides=_normalize_artifact_overrides(
            payload.get("artifact_overrides", ()),
            submission_root=submission_root,
        ),
    )


def run_all_stages(
    profile: SubmissionProfile,
    *,
    force: bool,
    dry_run: bool,
    config_path: Path | None,
) -> int:
    plan = stage_plan(profile, STAGE_ALL, force=force, config_path=config_path)
    if dry_run:
        print(json.dumps({"profile": SUBMISSION_PROFILE, "dry_run": True, "plan": plan}, indent=2, sort_keys=True))
        return 0
    for stage in ORDERED_STAGES:
        command = _self_stage_command(profile, stage, force=force, config_path=config_path)
        completed = subprocess.run(command, cwd=str(ROOT), check=False)
        if completed.returncode != 0:
            return int(completed.returncode)
    return 0


def run_stage(profile: SubmissionProfile, stage: str, *, force: bool, dry_run: bool) -> int:
    if dry_run:
        print(
            json.dumps(
                {"profile": SUBMISSION_PROFILE, "dry_run": True, "plan": stage_plan(profile, stage, force=force)},
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if stage == STAGE_VALIDATE:
        validate_submission_profile(profile)
        _record_pipeline_stage(profile, stage, "completed", {"validated": True})
        return 0
    if stage == STAGE_LM2011:
        return _run_command_stage(profile, stage, build_lm2011_command(profile, force=force))
    if stage == STAGE_NW_SENSITIVITY:
        _promote_nw_sensitivity_pack(profile, require_core=True, require_extension=False)
        _record_pipeline_stage(profile, stage, "completed", {"nw_lags": list(profile.config.nw_lags)})
        return 0
    if stage == STAGE_FINBERT_PREPROCESS:
        return _run_command_stage(profile, stage, build_finbert_preprocess_command(profile, force=force))
    if stage == STAGE_FINBERT_ANALYSIS:
        _require_finbert_preprocessing_checkpoint(profile)
        return _run_command_stage(profile, stage, build_finbert_analysis_command(profile, force=force))
    if stage == STAGE_EXTENSION:
        validate_submission_profile(profile, outputs_required=False)
        _run_extension_stage(profile, force=force)
        _promote_nw_sensitivity_pack(profile, require_core=True, require_extension=len(profile.config.nw_lags) > 1)
        _record_pipeline_stage(profile, stage, "completed", {"extension_output_dir": _rel(profile.extension_output_dir, profile)})
        return 0
    if stage == STAGE_VISIBLE_PREFIX_EXTENSION:
        validate_submission_profile(profile, outputs_required=False)
        _run_visible_prefix_extension_stage(profile, force=force)
        _record_pipeline_stage(
            profile,
            stage,
            "completed",
            {
                "visible_prefix_extension_output_dir": _rel(
                    profile.visible_prefix_extension_output_dir,
                    profile,
                )
            },
        )
        return 0
    if stage == STAGE_FINBERT_ROBUSTNESS:
        return _run_command_stage(profile, stage, build_finbert_robustness_command(profile, force=force))
    if stage == STAGE_THESIS_ASSETS:
        _write_submission_lock(profile)
        validate_submission_lock(profile.submission_lock_path)
        return _run_command_stage(profile, stage, build_thesis_assets_command(profile))
    raise SubmissionPipelineError(f"Unsupported stage: {stage}")


def stage_plan(
    profile: SubmissionProfile,
    stage: str,
    *,
    force: bool,
    config_path: Path | None = None,
) -> list[dict[str, Any]]:
    stages = ORDERED_STAGES if stage == STAGE_ALL else (stage,)
    plan: list[dict[str, Any]] = []
    for stage_name in stages:
        if stage_name == STAGE_VALIDATE:
            details = {"action": "validate strict SUBMISSION layout"}
        elif stage_name == STAGE_LM2011:
            details = {"command": _display_command(build_lm2011_command(profile, force=force))}
        elif stage_name == STAGE_NW_SENSITIVITY:
            details = {"action": "promote NW sensitivity artifacts", "output_dir": _rel(profile.nw_sensitivity_dir, profile)}
        elif stage_name == STAGE_FINBERT_PREPROCESS:
            details = {"command": _display_command(build_finbert_preprocess_command(profile, force=force))}
        elif stage_name == STAGE_FINBERT_ANALYSIS:
            details = {"command": _display_command(build_finbert_analysis_command(profile, force=force))}
        elif stage_name == STAGE_EXTENSION:
            details = {"action": "run LM2011 extension dictionary-family comparison", "output_dir": _rel(profile.extension_output_dir, profile)}
        elif stage_name == STAGE_VISIBLE_PREFIX_EXTENSION:
            details = {
                "action": "run FinBERT-visible LM2011 extension sensitivity",
                "output_dir": _rel(profile.visible_prefix_extension_output_dir, profile),
            }
        elif stage_name == STAGE_FINBERT_ROBUSTNESS:
            details = {"command": _display_command(build_finbert_robustness_command(profile, force=force))}
        elif stage_name == STAGE_THESIS_ASSETS:
            details = {"command": _display_command(build_thesis_assets_command(profile))}
        else:
            raise SubmissionPipelineError(f"Unsupported stage in plan: {stage_name}")
        plan.append({"stage": stage_name, **details})
    return plan


def build_lm2011_command(profile: SubmissionProfile, *, force: bool) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "src" / "thesis_pkg" / "notebooks_and_scripts" / "lm2011_sample_post_refinitiv_runner.py"),
        "--sample-root",
        str(profile.submission_root),
        "--upstream-run-root",
        str(profile.analysis_outputs_root / "_strict_no_upstream_auto_resolution"),
        "--additional-data-dir",
        str(profile.additional_data_dir),
        "--output-dir",
        str(profile.lm2011_output_dir),
        "--local-work-root",
        str(profile.local_work_root / "lm2011_post_refinitiv"),
        "--year-merged-dir",
        str(profile.year_merged_dir),
        "--items-analysis-dir",
        str(profile.items_analysis_dir),
        "--ccm-base-dir",
        str(profile.ccm_dir),
        "--daily-panel-path",
        str(profile.daily_panel_path),
        "--matched-clean-path",
        str(profile.matched_clean_path),
        "--doc-ownership-path",
        str(profile.doc_ownership_path),
        "--doc-analyst-selected-path",
        str(profile.doc_analyst_selected_path),
        "--monthly-stock-path",
        str(profile.monthly_stock_path),
        "--nw-lags",
        *[str(lag) for lag in profile.config.nw_lags],
    ]
    if force:
        command.extend(
            [
                "--recompute-text-features-full-10k",
                "--recompute-text-features-mda",
                "--recompute-strategy-text-features-full-10k",
                "--recompute-event-screen-surface",
                "--recompute-event-panel",
                "--recompute-regression-tables",
            ]
        )
    return command


def build_finbert_preprocess_command(profile: SubmissionProfile, *, force: bool) -> list[str]:
    command = _base_finbert_command(profile)
    command.append("--preprocess-only")
    if force:
        command.append("--overwrite")
    return command


def build_finbert_analysis_command(profile: SubmissionProfile, *, force: bool) -> list[str]:
    command = _base_finbert_command(profile)
    command.extend(["--analysis-only", "--write-sentence-scores"])
    if force:
        command.append("--overwrite")
    return command


def build_finbert_robustness_command(profile: SubmissionProfile, *, force: bool) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "src" / "thesis_pkg" / "notebooks_and_scripts" / "lm2011_finbert_robustness_runner.py"),
        "--extension-run-dir",
        str(profile.extension_output_dir),
        "--finbert-analysis-run-dir",
        str(profile.finbert_analysis_run_dir),
        "--output-dir",
        str(profile.finbert_robustness_output_dir),
        "--extension-analysis-panel-path",
        str(profile.extension_output_dir / "lm2011_extension_analysis_panel.parquet"),
        "--finbert-sentence-scores-dir",
        str(profile.finbert_analysis_run_dir / "sentence_scores" / "by_year"),
    ]
    return command


def build_thesis_assets_command(profile: SubmissionProfile) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "tools" / "build_thesis_assets.py"),
        "build-all",
        "--run-id",
        profile.config.run_id,
        "--repo-root",
        str(ROOT),
        "--output-root",
        str(profile.thesis_assets_output_root),
        "--submission-lock",
        str(profile.submission_lock_path),
    ]


def validate_submission_profile(profile: SubmissionProfile, *, outputs_required: bool = False) -> None:
    _assert_profile_paths_inside_submission_root(profile)
    _validate_submission_package_manifest(profile)
    _require_dir_with_parquet(profile.year_merged_dir, "data/sec/year_merged")
    _require_dir_with_parquet(profile.items_analysis_dir, "data/sec/items_analysis")
    _require_file(profile.matched_clean_path, "SEC/CCM matched-clean parquet")
    _require_file(profile.doc_ownership_path, "finalized Refinitiv document ownership parquet")
    _require_file(profile.doc_analyst_selected_path, "finalized Refinitiv analyst-selected parquet")
    _require_file(profile.daily_panel_path, "CCM daily market panel parquet")
    _require_file(profile.monthly_stock_path, "CCM monthly stock returns parquet")
    for filename in REQUIRED_CCM_PARQUETS:
        _require_file(profile.ccm_dir / filename, f"CCM input {filename}")
    for filename in REQUIRED_ADDITIONAL_DATA_FILES:
        _require_file(profile.additional_data_dir / filename, f"LM2011 additional-data input {filename}")
    _require_file(
        profile.additional_data_dir / EXTENDED_MASTER_DICTIONARY_FILENAME,
        "extended LM master dictionary for dictionary-family comparison",
    )
    if not any((profile.additional_data_dir / filename).exists() for filename in MASTER_DICTIONARY_CANDIDATES):
        raise SubmissionPipelineError(
            "LM2011 additional-data directory is missing a master dictionary candidate: "
            + ", ".join(MASTER_DICTIONARY_CANDIDATES)
        )

    _validate_parquet_columns(_first_parquet(profile.year_merged_dir), ("doc_id",), "year_merged shard")
    _validate_items_analysis_shards(profile.items_analysis_dir)
    _validate_parquet_columns(profile.matched_clean_path, ("doc_id",), "sec_ccm_matched_clean")
    _validate_parquet_columns(profile.doc_ownership_path, ("doc_id",), "refinitiv_lm2011_doc_ownership")
    _validate_parquet_columns(profile.doc_analyst_selected_path, ("doc_id",), "refinitiv_doc_analyst_selected")
    _validate_daily_panel_schema(profile.daily_panel_path)

    for output_root in (
        profile.analysis_outputs_root,
        profile.lm2011_output_dir,
        profile.nw_sensitivity_dir,
        profile.finbert_output_dir,
        profile.extension_output_dir,
        profile.visible_prefix_extension_output_dir,
        profile.finbert_robustness_output_dir,
        profile.thesis_assets_output_root,
    ):
        _assert_inside_submission_root(output_root, profile.submission_root)
        if outputs_required and not output_root.exists():
            raise SubmissionPipelineError(f"Expected output root is missing: {output_root}")


def _run_extension_stage(profile: SubmissionProfile, *, force: bool) -> None:
    from thesis_pkg.notebooks_and_scripts.lm2011_sample_post_refinitiv_runner import (
        DEFAULT_LM2011_FULL_10K_CLEANING_CONTRACT,
    )
    from thesis_pkg.notebooks_and_scripts.lm2011_sample_post_refinitiv_runner import (
        DEFAULT_LM2011_FULL_10K_TEXT_FEATURE_BATCH_SIZE,
    )
    from thesis_pkg.notebooks_and_scripts.lm2011_sample_post_refinitiv_runner import (
        DEFAULT_RAM_LOG_INTERVAL_BATCHES,
    )
    from thesis_pkg.notebooks_and_scripts.lm2011_sample_post_refinitiv_runner import (
        LM2011ExtensionRunConfig,
    )
    from thesis_pkg.notebooks_and_scripts.lm2011_sample_post_refinitiv_runner import (
        run_lm2011_extension_dictionary_family_comparison_pipeline,
    )

    cfg = LM2011ExtensionRunConfig(
        output_dir=profile.extension_output_dir,
        additional_data_dir=profile.additional_data_dir,
        items_analysis_dir=profile.items_analysis_dir,
        company_history_path=profile.ccm_dir / "companyhistory.parquet",
        company_description_path=profile.ccm_dir / "companydescription.parquet",
        ff48_siccodes_path=profile.additional_data_dir / "FF_Siccodes_48_Industries.txt",
        event_panel_path=profile.lm2011_output_dir / "lm2011_event_panel.parquet",
        year_merged_dir=profile.year_merged_dir,
        local_work_root=profile.local_work_root / "lm2011_extension",
        full_10k_cleaning_contract=DEFAULT_LM2011_FULL_10K_CLEANING_CONTRACT,
        full_10k_text_feature_batch_size=DEFAULT_LM2011_FULL_10K_TEXT_FEATURE_BATCH_SIZE,
        recompute_extension_text_features_full_10k=force,
        finbert_analysis_run_dir=profile.finbert_analysis_run_dir,
        finbert_analysis_manifest_path=profile.finbert_analysis_run_dir / "run_manifest.json",
        finbert_preprocessing_run_dir=profile.finbert_preprocessing_run_dir,
        finbert_preprocessing_manifest_path=profile.finbert_preprocessing_run_dir / "run_manifest.json",
        finbert_item_features_long_path=profile.finbert_analysis_run_dir / "item_features_long.parquet",
        finbert_cleaned_item_scopes_dir=profile.finbert_preprocessing_run_dir / "cleaned_item_scopes" / "by_year",
        run_id=profile.config.run_id,
        note="submission pipeline strict source-parquet rerun",
        ram_log_interval_batches=DEFAULT_RAM_LOG_INTERVAL_BATCHES,
        nw_lags=profile.config.nw_lags,
    )
    run_lm2011_extension_dictionary_family_comparison_pipeline(cfg)


def _run_visible_prefix_extension_stage(profile: SubmissionProfile, *, force: bool) -> None:
    from thesis_pkg.notebooks_and_scripts.lm2011_sample_post_refinitiv_runner import (
        DEFAULT_FINBERT_VISIBLE_PREFIX_SENTENCE_BATCH_SIZE,
    )
    from thesis_pkg.notebooks_and_scripts.lm2011_sample_post_refinitiv_runner import (
        DEFAULT_LM2011_FULL_10K_CLEANING_CONTRACT,
    )
    from thesis_pkg.notebooks_and_scripts.lm2011_sample_post_refinitiv_runner import (
        DEFAULT_LM2011_FULL_10K_TEXT_FEATURE_BATCH_SIZE,
    )
    from thesis_pkg.notebooks_and_scripts.lm2011_sample_post_refinitiv_runner import (
        DEFAULT_RAM_LOG_INTERVAL_BATCHES,
    )
    from thesis_pkg.notebooks_and_scripts.lm2011_sample_post_refinitiv_runner import (
        EXTENSION_DICTIONARY_SOURCE_FINBERT_VISIBLE_PREFIX,
    )
    from thesis_pkg.notebooks_and_scripts.lm2011_sample_post_refinitiv_runner import (
        LM2011ExtensionRunConfig,
    )
    from thesis_pkg.notebooks_and_scripts.lm2011_sample_post_refinitiv_runner import (
        run_lm2011_extension_dictionary_family_comparison_pipeline,
    )

    cfg = LM2011ExtensionRunConfig(
        output_dir=profile.visible_prefix_extension_output_dir,
        additional_data_dir=profile.additional_data_dir,
        items_analysis_dir=profile.items_analysis_dir,
        company_history_path=profile.ccm_dir / "companyhistory.parquet",
        company_description_path=profile.ccm_dir / "companydescription.parquet",
        ff48_siccodes_path=profile.additional_data_dir / "FF_Siccodes_48_Industries.txt",
        event_panel_path=profile.lm2011_output_dir / "lm2011_event_panel.parquet",
        year_merged_dir=profile.year_merged_dir,
        local_work_root=profile.local_work_root / "lm2011_extension_finbert_visible_prefix",
        full_10k_cleaning_contract=DEFAULT_LM2011_FULL_10K_CLEANING_CONTRACT,
        full_10k_text_feature_batch_size=DEFAULT_LM2011_FULL_10K_TEXT_FEATURE_BATCH_SIZE,
        recompute_extension_text_features_full_10k=force,
        finbert_analysis_run_dir=profile.finbert_analysis_run_dir,
        finbert_analysis_manifest_path=profile.finbert_analysis_run_dir / "run_manifest.json",
        finbert_preprocessing_run_dir=profile.finbert_preprocessing_run_dir,
        finbert_preprocessing_manifest_path=profile.finbert_preprocessing_run_dir / "run_manifest.json",
        finbert_item_features_long_path=profile.finbert_analysis_run_dir / "item_features_long.parquet",
        finbert_cleaned_item_scopes_dir=profile.finbert_preprocessing_run_dir / "cleaned_item_scopes" / "by_year",
        finbert_sentence_scores_dir=profile.finbert_analysis_run_dir / "sentence_scores" / "by_year",
        finbert_visible_prefix_model_name=profile.config.finbert.model_name,
        finbert_visible_prefix_model_revision=profile.config.finbert.model_revision,
        finbert_visible_prefix_tokenizer_revision=profile.config.finbert.tokenizer_revision,
        finbert_visible_prefix_sentence_batch_size=DEFAULT_FINBERT_VISIBLE_PREFIX_SENTENCE_BATCH_SIZE,
        dictionary_source_mode=EXTENSION_DICTIONARY_SOURCE_FINBERT_VISIBLE_PREFIX,
        text_scopes=("item_1a_risk_factors", "item_7_mda", "items_1a_7_combined"),
        run_id=profile.config.run_id,
        note="submission pipeline FinBERT-visible LM2011 extension sensitivity",
        ram_log_interval_batches=DEFAULT_RAM_LOG_INTERVAL_BATCHES,
        nw_lags=profile.config.nw_lags,
    )
    run_lm2011_extension_dictionary_family_comparison_pipeline(cfg)


def _write_submission_lock(profile: SubmissionProfile) -> Path:
    run_roots = {
        RUN_FAMILY_LM2011_POST_REFINITIV: profile.lm2011_output_dir,
        RUN_FAMILY_LM2011_NW_LAG_SENSITIVITY: profile.nw_sensitivity_dir,
        RUN_FAMILY_FINBERT_RUN: profile.finbert_analysis_run_dir,
        RUN_FAMILY_LM2011_EXTENSION: profile.extension_output_dir,
        RUN_FAMILY_LM2011_EXTENSION_FINBERT_VISIBLE_PREFIX: profile.visible_prefix_extension_output_dir,
        RUN_FAMILY_FINBERT_ROBUSTNESS: profile.finbert_robustness_output_dir,
    }
    overrides = [
        (override.artifact_key, override.path, override.reason)
        for override in profile.config.artifact_overrides
    ]
    payload = build_submission_lock_payload(
        submission_root=profile.submission_root,
        run_id=profile.config.run_id,
        run_roots=run_roots,
        artifact_overrides=overrides,
        provenance_disclosures=[],
    )
    return write_submission_lock(profile.submission_lock_path, payload)


def _promote_nw_sensitivity_pack(
    profile: SubmissionProfile,
    *,
    require_core: bool,
    require_extension: bool,
) -> None:
    if len(profile.config.nw_lags) <= 1:
        raise SubmissionPipelineError("NW sensitivity pack requires at least two nw_lags.")
    profile.nw_sensitivity_dir.mkdir(parents=True, exist_ok=True)
    source_map = {
        "core_tables_nw_lag_sensitivity.parquet": profile.lm2011_output_dir / "core_tables_nw_lag_sensitivity.parquet",
        "extension_results_nw_lag_sensitivity.parquet": profile.extension_output_dir / "extension_results_nw_lag_sensitivity.parquet",
        "extension_fit_comparisons_nw_lag_sensitivity.parquet": (
            profile.extension_output_dir / "extension_fit_comparisons_nw_lag_sensitivity.parquet"
        ),
        "nw_lag_sensitivity_summary.csv": profile.extension_output_dir / "nw_lag_sensitivity_summary.csv",
        "nw_lag_sensitivity_summary.md": profile.extension_output_dir / "nw_lag_sensitivity_summary.md",
    }
    required = ["core_tables_nw_lag_sensitivity.parquet"] if require_core else []
    if require_extension:
        required.extend(
            [
                "extension_results_nw_lag_sensitivity.parquet",
                "extension_fit_comparisons_nw_lag_sensitivity.parquet",
            ]
        )
    missing_required = [filename for filename in required if not source_map[filename].exists()]
    if missing_required:
        raise SubmissionPipelineError(f"Missing required NW sensitivity artifacts: {missing_required}")

    promoted: dict[str, Any] = {}
    for filename, source_path in source_map.items():
        if not source_path.exists():
            continue
        target_path = profile.nw_sensitivity_dir / filename
        shutil.copy2(source_path, target_path)
        promoted[filename] = {
            "source_path": _rel(source_path, profile),
            "promoted_path": _rel(target_path, profile),
            "fingerprint": semantic_file_fingerprint(target_path) if target_path.suffix == ".parquet" else _basic_file_fingerprint(target_path),
        }

    manifest = {
        "schema_version": 1,
        "profile": SUBMISSION_PROFILE,
        "run_id": profile.config.run_id,
        "nw_lags": list(profile.config.nw_lags),
        "primary_lag": profile.config.nw_lags[0],
        "source_roots": {
            RUN_FAMILY_LM2011_POST_REFINITIV: _rel(profile.lm2011_output_dir, profile),
            RUN_FAMILY_LM2011_EXTENSION: _rel(profile.extension_output_dir, profile),
        },
        "artifacts": promoted,
        "complete": all(
            filename in promoted
            for filename in (
                "core_tables_nw_lag_sensitivity.parquet",
                "extension_results_nw_lag_sensitivity.parquet",
                "extension_fit_comparisons_nw_lag_sensitivity.parquet",
            )
        ),
    }
    _write_json(profile.nw_sensitivity_dir / "lm2011_nw_lag_sensitivity_run_manifest.json", manifest)


def _base_finbert_command(profile: SubmissionProfile) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "src" / "thesis_pkg" / "notebooks_and_scripts" / "finbert_item_analysis_runner.py"),
        "--data-profile",
        "EXPLICIT",
        "--source-items-dir",
        str(profile.items_analysis_dir),
        "--backbone-path",
        str(profile.lm2011_output_dir / "lm2011_sample_backbone.parquet"),
        "--output-dir",
        str(profile.finbert_output_dir),
        "--run-name",
        DEFAULT_FINBERT_RUN_NAME,
        "--batch-profile",
        profile.config.finbert.batch_profile,
        "--model-name",
        profile.config.finbert.model_name,
        "--model-revision",
        profile.config.finbert.model_revision,
        "--tokenizer-revision",
        profile.config.finbert.tokenizer_revision,
    ]
    if profile.config.finbert.device is not None:
        command.extend(["--device", profile.config.finbert.device])
    if profile.config.years is not None:
        command.append("--years")
        command.extend(str(year) for year in profile.config.years)
    return command


def _run_command_stage(profile: SubmissionProfile, stage: str, command: Sequence[str]) -> int:
    _record_pipeline_stage(profile, stage, "started", {"command": list(command)})
    completed = subprocess.run(list(command), cwd=str(ROOT), check=False)
    status = "completed" if completed.returncode == 0 else "failed"
    _record_pipeline_stage(
        profile,
        stage,
        status,
        {"command": list(command), "returncode": int(completed.returncode)},
    )
    return int(completed.returncode)


def _self_stage_command(
    profile: SubmissionProfile,
    stage: str,
    *,
    force: bool,
    config_path: Path | None,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--submission-root",
        str(profile.submission_root),
        "--stage",
        stage,
    ]
    if config_path is not None:
        command.extend(["--config", str(config_path)])
    if profile.config.run_id != DEFAULT_RUN_ID:
        command.extend(["--run-id", profile.config.run_id])
    if force:
        command.append("--force")
    return command


def _record_pipeline_stage(
    profile: SubmissionProfile,
    stage: str,
    status: str,
    details: Mapping[str, Any],
) -> None:
    manifest = _read_json_or_empty(profile.pipeline_manifest_path)
    if not manifest:
        manifest = {
            "schema_version": 1,
            "profile": SUBMISSION_PROFILE,
            "run_id": profile.config.run_id,
            "submission_root": str(profile.submission_root),
            "stages": {},
            "config": {
                "years": list(profile.config.years) if profile.config.years is not None else None,
                "nw_lags": list(profile.config.nw_lags),
                "finbert": asdict(profile.config.finbert),
            },
        }
    manifest["stages"][stage] = {"status": status, **dict(details)}
    _write_json(profile.pipeline_manifest_path, manifest)


def _require_finbert_preprocessing_checkpoint(profile: SubmissionProfile) -> None:
    run_dir = profile.finbert_preprocessing_run_dir
    required_paths = (
        run_dir / "run_manifest.json",
        run_dir / "sentence_dataset" / "by_year",
    )
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise SubmissionPipelineError(
            "FinBERT analysis requires the named preprocessing checkpoint. Missing: "
            + ", ".join(missing)
        )


def _assert_profile_paths_inside_submission_root(profile: SubmissionProfile) -> None:
    for path in (
        profile.data_root,
        profile.sec_root,
        profile.year_merged_dir,
        profile.items_analysis_dir,
        profile.matched_clean_path,
        profile.ccm_dir,
        profile.daily_panel_path,
        profile.monthly_stock_path,
        profile.additional_data_dir,
        profile.refinitiv_dir,
        profile.doc_ownership_path,
        profile.doc_analyst_selected_path,
        profile.analysis_outputs_root,
        profile.lm2011_output_dir,
        profile.nw_sensitivity_dir,
        profile.finbert_output_dir,
        profile.finbert_preprocessing_run_dir,
        profile.finbert_analysis_run_dir,
        profile.extension_output_dir,
        profile.visible_prefix_extension_output_dir,
        profile.finbert_robustness_output_dir,
        profile.thesis_assets_output_root,
        profile.pipeline_manifest_path,
        profile.package_manifest_path,
        profile.submission_lock_path,
    ):
        _assert_inside_submission_root(path, profile.submission_root)
    for override in profile.config.artifact_overrides:
        _assert_inside_submission_root(override.path, profile.submission_root)


def _assert_inside_submission_root(path: Path, submission_root: Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_relative_to(submission_root.resolve()):
        raise SubmissionPipelineError(f"Resolved path escapes submission_root: {resolved}")
    return resolved


def _normalize_artifact_overrides(raw_value: Any, *, submission_root: Path) -> tuple[ArtifactOverrideConfig, ...]:
    if raw_value in (None, ""):
        return ()
    if not isinstance(raw_value, list):
        raise SubmissionPipelineError("artifact_overrides must be a list when provided.")
    overrides: list[ArtifactOverrideConfig] = []
    for index, raw_override in enumerate(raw_value):
        if not isinstance(raw_override, dict):
            raise SubmissionPipelineError(f"artifact_overrides[{index}] must be a JSON object.")
        artifact_key = _str_or_default(raw_override.get("artifact_key"), "", label=f"artifact_overrides[{index}].artifact_key")
        reason = _str_or_default(raw_override.get("reason"), "", label=f"artifact_overrides[{index}].reason")
        raw_path = raw_override.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise SubmissionPipelineError(f"artifact_overrides[{index}].path must be a non-empty relative string.")
        value = Path(raw_path)
        if value.is_absolute():
            raise SubmissionPipelineError(f"artifact_overrides[{index}].path must be relative to submission_root.")
        resolved = (submission_root / value).resolve()
        if not resolved.is_relative_to(submission_root):
            raise SubmissionPipelineError(f"artifact_overrides[{index}].path escapes submission_root: {raw_path}")
        if not artifact_key.strip():
            raise SubmissionPipelineError(f"artifact_overrides[{index}].artifact_key must be non-empty.")
        if not reason.strip():
            raise SubmissionPipelineError(f"artifact_overrides[{index}].reason must be non-empty.")
        overrides.append(ArtifactOverrideConfig(artifact_key=artifact_key, path=resolved, reason=reason))
    return tuple(overrides)


def _normalize_years(raw_value: Any) -> tuple[int, ...] | None:
    if raw_value in (None, ""):
        return None
    if not isinstance(raw_value, list):
        raise SubmissionPipelineError("years must be a list of integers when provided.")
    years = tuple(int(value) for value in raw_value)
    if any(year < 1900 or year > 2100 for year in years):
        raise SubmissionPipelineError(f"years contains implausible filing years: {years}")
    return tuple(dict.fromkeys(years))


def _normalize_nw_lags(raw_value: Any) -> tuple[int, ...]:
    if not isinstance(raw_value, list) or not raw_value:
        raise SubmissionPipelineError("nw_lags must be a non-empty list of positive integers.")
    lags = tuple(dict.fromkeys(int(value) for value in raw_value))
    if any(lag <= 0 for lag in lags):
        raise SubmissionPipelineError(f"nw_lags must be positive integers, got {lags}")
    if lags[0] != 1:
        raise SubmissionPipelineError("nw_lags must keep lag 1 as the first primary lag.")
    return lags


def _optional_str(value: Any, *, label: str) -> str | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise SubmissionPipelineError(f"{label} must be a string when provided.")
    return value


def _str_or_default(value: Any, default: str, *, label: str) -> str:
    if value in (None, ""):
        return default
    if not isinstance(value, str):
        raise SubmissionPipelineError(f"{label} must be a string.")
    return value


def _require_dir_with_parquet(path: Path, label: str) -> None:
    if not path.exists() or not path.is_dir():
        raise SubmissionPipelineError(f"Required directory is missing: {label} ({path})")
    if not any(path.glob("*.parquet")):
        raise SubmissionPipelineError(f"Required directory contains no parquet shards: {label} ({path})")


def _require_file(path: Path, label: str) -> None:
    if not path.exists() or not path.is_file():
        raise SubmissionPipelineError(f"Required file is missing: {label} ({path})")


def _first_parquet(path: Path) -> Path:
    paths = sorted(path.glob("*.parquet"))
    if not paths:
        raise SubmissionPipelineError(f"No parquet files found in {path}")
    return paths[0]


def _parquet_paths(path: Path) -> list[Path]:
    paths = sorted(path.glob("*.parquet"))
    if not paths:
        raise SubmissionPipelineError(f"No parquet files found in {path}")
    return paths


def _validate_parquet_columns(path: Path, required_columns: Sequence[str], label: str) -> None:
    schema = pl.scan_parquet(path).collect_schema()
    missing = [column for column in required_columns if column not in schema]
    if missing:
        raise SubmissionPipelineError(f"{label} is missing required columns {missing}: {path}")


def _validate_items_analysis_shards(items_analysis_dir: Path) -> None:
    for path in _parquet_paths(items_analysis_dir):
        schema_names = set(pl.scan_parquet(path).collect_schema().names())
        missing_columns = [column for column in ITEMS_ANALYSIS_REQUIRED_COLUMNS if column not in schema_names]
        missing_groups = [
            label
            for label, candidates in ITEMS_ANALYSIS_REQUIRED_COLUMN_GROUPS.items()
            if not any(candidate in schema_names for candidate in candidates)
        ]
        if missing_columns or missing_groups:
            raise SubmissionPipelineError(
                "items_analysis shard schema is invalid. "
                f"Missing columns={missing_columns}; missing column groups={missing_groups}; path={path}"
            )


def _validate_submission_package_manifest(profile: SubmissionProfile) -> None:
    _require_file(profile.package_manifest_path, "submission package manifest")
    try:
        payload = json.loads(profile.package_manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SubmissionPipelineError(f"submission package manifest is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SubmissionPipelineError("submission package manifest must be a JSON object.")
    if payload.get("schema_version") != PACKAGE_MANIFEST_SCHEMA_VERSION:
        raise SubmissionPipelineError(
            "submission package manifest schema_version must be "
            f"{PACKAGE_MANIFEST_SCHEMA_VERSION}."
        )
    if payload.get("profile") != SUBMISSION_PROFILE:
        raise SubmissionPipelineError(f"submission package manifest profile must be {SUBMISSION_PROFILE!r}.")
    if payload.get("path_semantics") != PACKAGE_MANIFEST_PATH_SEMANTICS:
        raise SubmissionPipelineError(
            "submission package manifest path_semantics must be "
            f"{PACKAGE_MANIFEST_PATH_SEMANTICS!r}."
        )
    if "code_version" not in payload:
        raise SubmissionPipelineError("submission package manifest must include code_version.")
    artifact_groups = payload.get("artifact_groups")
    if not isinstance(artifact_groups, dict):
        raise SubmissionPipelineError("submission package manifest artifact_groups must be a JSON object.")
    missing_groups = [group for group in PACKAGE_MANIFEST_REQUIRED_GROUPS if group not in artifact_groups]
    if missing_groups:
        raise SubmissionPipelineError(f"submission package manifest is missing artifact groups: {missing_groups}")
    for group_name in PACKAGE_MANIFEST_REQUIRED_GROUPS:
        group_payload = artifact_groups[group_name]
        if not isinstance(group_payload, dict):
            raise SubmissionPipelineError(
                f"submission package manifest artifact_groups.{group_name} must be a JSON object."
            )
        paths = group_payload.get("paths")
        if not isinstance(paths, list) or not paths:
            raise SubmissionPipelineError(
                f"submission package manifest artifact_groups.{group_name}.paths must be a non-empty list."
            )
        for index, raw_path in enumerate(paths):
            if not isinstance(raw_path, str) or not raw_path.strip():
                raise SubmissionPipelineError(
                    f"submission package manifest artifact_groups.{group_name}.paths[{index}] "
                    "must be a non-empty relative path string."
                )
            value = Path(raw_path)
            if value.is_absolute():
                raise SubmissionPipelineError(
                    f"submission package manifest path must be relative to submission_root: {raw_path}"
                )
            resolved = (profile.submission_root / value).resolve()
            if not resolved.is_relative_to(profile.submission_root):
                raise SubmissionPipelineError(
                    f"submission package manifest path escapes submission_root: {raw_path}"
                )
            if not resolved.exists():
                raise SubmissionPipelineError(
                    f"submission package manifest references a missing path: {raw_path}"
                )
        schema_summary = group_payload.get("schema_summary")
        if schema_summary is not None and not isinstance(schema_summary, dict):
            raise SubmissionPipelineError(
                f"submission package manifest artifact_groups.{group_name}.schema_summary must be a JSON object."
            )


def _validate_daily_panel_schema(path: Path) -> None:
    schema = pl.scan_parquet(path).collect_schema()
    grouped_candidates = {
        "permno": ("KYPERMNO", "kypermno"),
        "trade_date": ("CALDT", "daily_caldt"),
        "return": ("FINAL_RET", "RET"),
        "price": ("FINAL_PRC", "PRC"),
    }
    required_singletons = ("VOL", "SHROUT", "SHRCD", "EXCHCD")
    missing_groups = [
        label
        for label, candidates in grouped_candidates.items()
        if not any(candidate in schema for candidate in candidates)
    ]
    missing_columns = [column for column in required_singletons if column not in schema]
    if missing_groups or missing_columns:
        raise SubmissionPipelineError(
            "CCM daily market panel schema is invalid. "
            f"Missing groups={missing_groups}; missing columns={missing_columns}; path={path}"
        )


def _display_command(command: Sequence[str]) -> list[str]:
    return [str(part) for part in command]


def _rel(path: Path, profile: SubmissionProfile) -> str:
    resolved = Path(path).resolve()
    return resolved.relative_to(profile.submission_root).as_posix() if resolved.is_relative_to(profile.submission_root) else str(resolved)


def _basic_file_fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _read_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

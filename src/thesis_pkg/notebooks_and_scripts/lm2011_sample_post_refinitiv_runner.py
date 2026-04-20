from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT_ENV_VAR = "NLP_THESIS_REPO_ROOT"
IN_COLAB = "google.colab" in sys.modules


def _resolve_repo_root() -> Path:
    candidates: list[Path] = []
    env_root = os.environ.get(REPO_ROOT_ENV_VAR)
    if env_root:
        candidates.append(Path(env_root).expanduser())

    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])

    script_path = Path(__file__).resolve()
    candidates.extend(script_path.parents)

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "src" / "thesis_pkg" / "pipeline.py").exists():
            return candidate
    raise RuntimeError("Could not resolve repository root containing src/thesis_pkg/pipeline.py")


def _resolve_colab_drive_root() -> Path:
    for candidate in (
        Path("/content/drive/MyDrive"),
        Path("/content/drive/My Drive"),
        Path("/content/drive"),
    ):
        if candidate.exists():
            return candidate
    return Path("/content/drive")


ROOT = _resolve_repo_root()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import polars as pl

from thesis_pkg.core.ccm.lm2011 import (
    build_annual_accounting_panel,
    build_lm2011_sample_backbone,
    build_quarterly_accounting_panel,
)
from thesis_pkg.core.sec.lm2011_cleaning import FULL_10K_CLEANING_CONTRACTS
from thesis_pkg.core.sec.lm2011_dictionary import load_lm2011_dictionary_inputs
from thesis_pkg.core.sec.lm2011_dictionary import load_lm2011_master_dictionary_words
from thesis_pkg.core.sec.lm2011_dictionary import load_lm2011_word_list
from thesis_pkg.core.sec.lm2011_text import (
    DEFAULT_PRODUCTION_FULL_10K_MICROBATCH_SIZE,
    DEFAULT_PRODUCTION_MDA_MICROBATCH_SIZE,
    RAW_ITEM_TEXT_CLEANING_POLICY_ID,
    _build_lm2011_signal_specs,
    _feature_schema,
    build_lm2011_text_features_full_10k,
    build_lm2011_text_features_mda,
    normalize_lm2011_dictionary_lists,
    write_lm2011_text_features_full_10k_parquet,
    write_lm2011_text_features_mda_parquet,
)
from thesis_pkg.io.parquet import _copy_with_verify
from thesis_pkg.pipelines.lm2011_pipeline import (
    build_lm2011_event_panel,
    build_lm2011_sue_panel,
    build_lm2011_table_i_sample_creation,
    build_lm2011_trading_strategy_monthly_returns,
    write_lm2011_sue_panel_parquet,
)
from thesis_pkg.pipelines import lm2011_pipeline
from thesis_pkg.pipelines.lm2011_extension import (
    EXTENSION_ITEM_SCOPE_IDS,
    EXTENSION_PRIMARY_TEXT_SCOPES,
    build_lm2011_extension_analysis_panel,
    build_lm2011_extension_control_ladder,
    build_lm2011_extension_dictionary_features,
    build_lm2011_extension_dictionary_features_from_cleaned_scopes,
    build_lm2011_extension_sample_loss_table,
    build_lm2011_extension_specification_grid,
    normalize_lm2011_extension_text_scope_expr,
    run_lm2011_extension_estimation_scaffold,
)
from thesis_pkg.pipelines.lm2011_regressions import (
    _QuarterlyFamaMacbethBundle,
    _build_lm2011_table_ia_i_results_bundle,
    _build_lm2011_table_ia_i_results_no_ownership_bundle,
    _build_lm2011_table_iv_results_bundle,
    _build_lm2011_table_iv_results_no_ownership_bundle,
    _build_lm2011_table_v_results_bundle,
    _build_lm2011_table_v_results_no_ownership_bundle,
    _build_lm2011_table_vi_results_bundle,
    _build_lm2011_table_vi_results_no_ownership_bundle,
    _build_lm2011_table_viii_results_bundle,
    _build_lm2011_table_viii_results_no_ownership_bundle,
    build_lm2011_return_regression_panel,
    build_lm2011_sue_regression_panel,
    build_lm2011_table_ia_ii_results,
)


DEFAULT_SAMPLE_ROOT = (
    _resolve_colab_drive_root() / "Data_LM"
    if IN_COLAB
    else ROOT / "full_data_run" / "sample_5pct_seed42"
)
DEFAULT_UPSTREAM_RUN_ROOT = (
    DEFAULT_SAMPLE_ROOT / "results" / "sec_ccm_unified_runner"
    if IN_COLAB
    else DEFAULT_SAMPLE_ROOT / "results" / "sec_ccm_unified_runner" / "local_sample"
)
DEFAULT_ADDITIONAL_DATA_DIR = (
    DEFAULT_SAMPLE_ROOT / "LM2011_additional_data"
    if IN_COLAB
    else ROOT / "full_data_run" / "LM2011_additional_data"
)
DEFAULT_OUTPUT_DIR = DEFAULT_SAMPLE_ROOT / "results" / "lm2011_sample_post_refinitiv_runner"
DEFAULT_LOCAL_WORK_ROOT = (
    Path("/content/_batch_work") / "lm2011_post_refinitiv"
    if IN_COLAB
    else ROOT / ".tmp" / "lm2011_post_refinitiv"
)


def _env_value(name: str, *, env: Mapping[str, str] | None = None) -> str | None:
    source = os.environ if env is None else env
    return source.get(name)


def _env_bool_from_mapping(
    name: str,
    default: bool,
    *,
    env: Mapping[str, str] | None = None,
) -> bool:
    value = _env_value(name, env=env)
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value for {name}: {value!r}")


def _env_optional_bool_from_mapping(
    name: str,
    default: bool | None,
    *,
    env: Mapping[str, str] | None = None,
) -> bool | None:
    value = _env_value(name, env=env)
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"", "auto", "none", "null"}:
        return None
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid optional boolean value for {name}: {value!r}")


def _resolve_stage_toggle_from_env(
    env_name: str,
    *,
    umbrella_enabled: bool,
    default_when_umbrella: bool,
    env: Mapping[str, str] | None = None,
) -> bool:
    explicit = _env_optional_bool_from_mapping(env_name, None, env=env)
    if explicit is not None:
        return explicit
    return umbrella_enabled and default_when_umbrella

PARQUET_COMPRESSION = "zstd"
YEAR_MERGED_GLOB = "*.parquet"
ITEMS_ANALYSIS_GLOB = "*.parquet"
LM2011_SEC_BACKBONE_COLUMNS: tuple[str, ...] = (
    "doc_id",
    "cik_10",
    "accession_nodash",
    "document_type_filename",
)

ANNUAL_BALANCE_SHEET_COLUMNS: tuple[str, ...] = (
    "KYGVKEY",
    "KEYSET",
    "FYYYY",
    "fyra",
    "SEQ",
    "CEQ",
    "AT",
    "LT",
    "TXDITC",
    "PSTKL",
    "PSTKRV",
    "PSTK",
)
ANNUAL_INCOME_STATEMENT_COLUMNS: tuple[str, ...] = (
    "KYGVKEY",
    "KEYSET",
    "FYYYY",
    "fyra",
    "IB",
    "XINT",
    "TXDI",
    "DVP",
)
ANNUAL_PERIOD_DESCRIPTOR_COLUMNS: tuple[str, ...] = (
    "KYGVKEY",
    "KEYSET",
    "FYYYY",
    "fyra",
    "FYEAR",
    "FYR",
    "APDEDATE",
    "FDATE",
    "PDATE",
)
ANNUAL_FISCAL_MARKET_COLUMNS: tuple[str, ...] = (
    "KYGVKEY",
    "KEYSET",
    "DATADATE",
    "MKVALT",
    "PRCC",
)
FF_DAILY_COLUMNS: tuple[str, ...] = ("raw_date", "mkt_rf", "smb", "hml", "rf")
FF_MONTHLY_COLUMNS: tuple[str, ...] = ("raw_month", "mkt_rf", "smb", "hml", "rf")
MOMENTUM_MONTHLY_COLUMNS: tuple[str, ...] = ("raw_month", "mom")
KEN_FRENCH_MISSING_FACTOR_VALUES: tuple[float, ...] = (-99.99, -999.0)
MONTHLY_STOCK_CANDIDATES: tuple[str, ...] = (
    "sfz_mth.parquet",
    "stockmonthly.parquet",
    "securitymonthly.parquet",
    "monthlystock.parquet",
)
EMPTY_TABLE_REASON = "insufficient_sample_size_for_estimable_quarterly_fama_macbeth"
NO_ESTIMABLE_QUARTERLY_FAMA_MACBETH_QUARTERS = "no_estimable_quarterly_fama_macbeth_quarters"
SKIPPED_MISSING_SEEDED_UPSTREAM = "skipped_missing_seeded_upstream"
SKIPPED_MISSING_OPTIONAL_INPUT = "skipped_missing_optional_input"
DEFAULT_LM2011_FULL_10K_CLEANING_CONTRACT = "lm2011_paper"
DEFAULT_LM2011_FULL_10K_TEXT_FEATURE_BATCH_SIZE = DEFAULT_PRODUCTION_FULL_10K_MICROBATCH_SIZE
DEFAULT_LM2011_MDA_TEXT_FEATURE_BATCH_SIZE = DEFAULT_PRODUCTION_MDA_MICROBATCH_SIZE
DEFAULT_LM2011_TEXT_FEATURE_BATCH_SIZE = DEFAULT_LM2011_FULL_10K_TEXT_FEATURE_BATCH_SIZE
DEFAULT_LM2011_EVENT_WINDOW_DOC_BATCH_SIZE = 50
DEFAULT_RAM_LOG_INTERVAL_BATCHES = 10

STAGE_ARTIFACT_FILENAMES: dict[str, str] = {
    "sample_backbone": "lm2011_sample_backbone.parquet",
    "annual_accounting_panel": "lm2011_annual_accounting_panel.parquet",
    "quarterly_accounting_panel": "lm2011_quarterly_accounting_panel.parquet",
    "ff_factors_daily_normalized": "lm2011_ff_factors_daily_normalized.parquet",
    "ff_factors_monthly_with_mom_normalized": "lm2011_ff_factors_monthly_with_mom_normalized.parquet",
    "text_features_full_10k": "lm2011_text_features_full_10k.parquet",
    "text_features_mda": "lm2011_text_features_mda.parquet",
    "event_screen_surface": "lm2011_event_screen_surface.parquet",
    "table_i_sample_creation": "lm2011_table_i_sample_creation.parquet",
    "table_i_sample_creation_1994_2024": "lm2011_table_i_sample_creation_1994_2024.parquet",
    "event_panel": "lm2011_event_panel.parquet",
    "sue_panel": "lm2011_sue_panel.parquet",
    "return_regression_panel_full_10k": "lm2011_return_regression_panel_full_10k.parquet",
    "return_regression_panel_mda": "lm2011_return_regression_panel_mda.parquet",
    "sue_regression_panel": "lm2011_sue_regression_panel.parquet",
    "table_iv_results": "lm2011_table_iv_results.parquet",
    "table_iv_results_no_ownership": "lm2011_table_iv_results_no_ownership.parquet",
    "table_v_results": "lm2011_table_v_results.parquet",
    "table_v_results_no_ownership": "lm2011_table_v_results_no_ownership.parquet",
    "table_vi_results": "lm2011_table_vi_results.parquet",
    "table_vi_results_no_ownership": "lm2011_table_vi_results_no_ownership.parquet",
    "table_viii_results": "lm2011_table_viii_results.parquet",
    "table_viii_results_no_ownership": "lm2011_table_viii_results_no_ownership.parquet",
    "table_ia_i_results": "lm2011_table_ia_i_results.parquet",
    "table_ia_i_results_no_ownership": "lm2011_table_ia_i_results_no_ownership.parquet",
    "trading_strategy_monthly_returns": "lm2011_trading_strategy_monthly_returns.parquet",
    "table_ia_ii_results": "lm2011_table_ia_ii_results.parquet",
}
FINAL_REGRESSION_TABLE_STAGE_NAMES: frozenset[str] = frozenset(
    {
        "table_iv_results",
        "table_iv_results_no_ownership",
        "table_v_results",
        "table_v_results_no_ownership",
        "table_vi_results",
        "table_vi_results_no_ownership",
        "table_viii_results",
        "table_viii_results_no_ownership",
        "table_ia_i_results",
        "table_ia_i_results_no_ownership",
        "table_ia_ii_results",
    }
)
QUARTERLY_REGRESSION_TABLE_STAGE_NAMES: frozenset[str] = frozenset(
    FINAL_REGRESSION_TABLE_STAGE_NAMES - {"table_ia_ii_results"}
)
MANIFEST_FILENAME = "lm2011_sample_run_manifest.json"
LM2011_ALL_STAGE_NAMES: tuple[str, ...] = tuple(STAGE_ARTIFACT_FILENAMES)
LM2011_OPTIONAL_STAGE_DEFAULTS_FALSE = frozenset(
    {
        "ff_factors_monthly_with_mom_normalized",
        "trading_strategy_monthly_returns",
        "table_ia_ii_results",
    }
)
SKIPPED_STAGE_DISABLED = "disabled_by_run_config"
STAGE_STATUS_REUSED_EXISTING_ARTIFACT = "reused_existing_artifact"
EXTENSION_MANIFEST_FILENAME = "lm2011_extension_run_manifest.json"
EXTENSION_DICTIONARY_SOURCE_PREFER_CLEANED = "prefer_cleaned_scopes"
EXTENSION_DICTIONARY_SOURCE_RAW = "raw_item_text"
EXTENSION_ALLOWED_DICTIONARY_SOURCE_MODES = (
    EXTENSION_DICTIONARY_SOURCE_PREFER_CLEANED,
    EXTENSION_DICTIONARY_SOURCE_RAW,
)
EXTENSION_STAGE_ARTIFACT_FILENAMES: dict[str, str] = {
    "extension_dictionary_surface": "lm2011_extension_dictionary_surface.parquet",
    "extension_finbert_surface": "lm2011_extension_finbert_surface.parquet",
    "extension_control_ladder": "lm2011_extension_control_ladder.parquet",
    "extension_specification_grid": "lm2011_extension_specification_grid.parquet",
    "extension_analysis_panel": "lm2011_extension_analysis_panel.parquet",
    "extension_sample_loss": "lm2011_extension_sample_loss.parquet",
    "extension_results": "lm2011_extension_results.parquet",
}


@dataclass(frozen=True)
class RunnerPaths:
    sample_root: Path
    upstream_run_root: Path
    additional_data_dir: Path
    output_dir: Path
    local_work_root: Path
    year_merged_dir: Path
    sample_backbone_path: Path | None
    daily_panel_path: Path
    text_features_full_10k_path: Path | None
    text_features_full_10k_path_is_explicit: bool
    text_features_mda_path: Path | None
    text_features_mda_path_is_explicit: bool
    ccm_base_dir: Path
    matched_clean_path: Path
    items_analysis_dir: Path
    doc_ownership_path: Path
    doc_analyst_selected_path: Path
    filingdates_path: Path | None
    quarterly_balance_sheet_path: Path | None
    quarterly_income_statement_path: Path | None
    quarterly_period_descriptor_path: Path | None
    annual_balance_sheet_path: Path | None
    annual_income_statement_path: Path | None
    annual_period_descriptor_path: Path | None
    annual_fiscal_market_path: Path | None
    company_history_path: Path | None
    company_description_path: Path | None
    ff_daily_csv_path: Path
    ff_monthly_csv_path: Path
    momentum_monthly_csv_path: Path
    ff48_siccodes_path: Path
    monthly_stock_path: Path | None
    ff_monthly_with_mom_path: Path | None
    full_10k_cleaning_contract: str
    full_10k_text_feature_batch_size: int
    mda_text_feature_batch_size: int
    recompute_event_screen_surface: bool
    recompute_event_panel: bool
    recompute_regression_tables: bool
    event_window_doc_batch_size: int
    print_ram_stats: bool
    ram_log_interval_batches: int


@dataclass(frozen=True)
class _TextFeatureReuseSpec:
    stage_name: str
    token_count_col: str
    total_token_count_col: str
    include_item_id: bool
    include_cleaning_policy_id: bool
    blocking_stage_names: frozenset[str]


TEXT_FEATURE_REUSE_SPECS: dict[str, _TextFeatureReuseSpec] = {
    "text_features_full_10k": _TextFeatureReuseSpec(
        stage_name="text_features_full_10k",
        token_count_col="token_count_full_10k",
        total_token_count_col="total_token_count_full_10k",
        include_item_id=False,
        include_cleaning_policy_id=False,
        blocking_stage_names=frozenset(
            {
                "event_screen_surface",
                "table_i_sample_creation",
                "table_i_sample_creation_1994_2024",
                "return_regression_panel_full_10k",
                "sue_regression_panel",
                "table_iv_results",
                "table_iv_results_no_ownership",
                "table_vi_results",
                "table_vi_results_no_ownership",
                "table_viii_results",
                "table_viii_results_no_ownership",
                "table_ia_i_results",
                "table_ia_i_results_no_ownership",
            }
        ),
    ),
    "text_features_mda": _TextFeatureReuseSpec(
        stage_name="text_features_mda",
        token_count_col="token_count_mda",
        total_token_count_col="total_token_count_mda",
        include_item_id=True,
        include_cleaning_policy_id=True,
        blocking_stage_names=frozenset(
            {
                "return_regression_panel_mda",
                "table_v_results",
                "table_v_results_no_ownership",
            }
        ),
    ),
}


@dataclass(frozen=True)
class LM2011PostRefinitivRunConfig:
    paths: RunnerPaths
    enabled_stages: tuple[str, ...] = LM2011_ALL_STAGE_NAMES
    fail_closed_for_enabled_stages: bool = False

    def __post_init__(self) -> None:
        unknown = sorted(set(self.enabled_stages) - set(LM2011_ALL_STAGE_NAMES))
        if unknown:
            raise ValueError(f"Unknown LM2011 stage names: {unknown}")


def resolve_lm2011_stage_flags_from_env(
    env: Mapping[str, str] | None = None,
) -> dict[str, bool]:
    umbrella_enabled = _env_bool_from_mapping(
        "SEC_CCM_RUN_LM2011_POST_REFINITIV",
        False,
        env=env,
    )
    return {
        stage_name: _resolve_stage_toggle_from_env(
            f"SEC_CCM_RUN_LM2011_{stage_name.upper()}",
            umbrella_enabled=umbrella_enabled,
            default_when_umbrella=stage_name not in LM2011_OPTIONAL_STAGE_DEFAULTS_FALSE,
            env=env,
        )
        for stage_name in LM2011_ALL_STAGE_NAMES
    }


def resolve_enabled_lm2011_stage_names_from_env(
    env: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    stage_flags = resolve_lm2011_stage_flags_from_env(env)
    return tuple(
        stage_name
        for stage_name in LM2011_ALL_STAGE_NAMES
        if stage_flags[stage_name]
    )


@dataclass(frozen=True)
class LM2011ExtensionRunConfig:
    output_dir: Path
    additional_data_dir: Path
    items_analysis_dir: Path
    event_panel_path: Path
    company_history_path: Path
    company_description_path: Path
    ff48_siccodes_path: Path
    finbert_item_features_long_path: Path | None = None
    finbert_analysis_run_dir: Path | None = None
    finbert_analysis_manifest_path: Path | None = None
    finbert_cleaned_item_scopes_dir: Path | None = None
    finbert_preprocessing_run_dir: Path | None = None
    finbert_preprocessing_manifest_path: Path | None = None
    require_cleaned_scope_match: bool = True
    dictionary_source_mode: str = EXTENSION_DICTIONARY_SOURCE_PREFER_CLEANED
    text_scopes: tuple[str, ...] = EXTENSION_PRIMARY_TEXT_SCOPES
    run_id: str = "lm2011_extension"
    note: str = ""

    def __post_init__(self) -> None:
        if self.dictionary_source_mode not in EXTENSION_ALLOWED_DICTIONARY_SOURCE_MODES:
            raise ValueError(
                "Unknown LM2011 extension dictionary_source_mode: "
                f"{self.dictionary_source_mode!r}"
            )
        if not self.text_scopes:
            raise ValueError("LM2011 extension text_scopes must be non-empty.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the sample-only post-Refinitiv LM2011 replication pipeline.")
    parser.add_argument("--sample-root", type=Path, default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument("--upstream-run-root", type=Path, default=DEFAULT_UPSTREAM_RUN_ROOT)
    parser.add_argument("--additional-data-dir", type=Path, default=DEFAULT_ADDITIONAL_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--local-work-root",
        type=Path,
        default=DEFAULT_LOCAL_WORK_ROOT,
        help="Local scratch root for LM2011 temporary parquet staging before final artifact promotion.",
    )
    parser.add_argument("--year-merged-dir", type=Path, default=None)
    parser.add_argument(
        "--sample-backbone-path",
        type=Path,
        default=None,
        help=(
            "Optional prebuilt LM2011 sample backbone parquet. If omitted, the runner "
            "auto-reuses upstream sec_ccm_premerge/lm2011_sample_backbone.parquet when present."
        ),
    )
    parser.add_argument("--matched-clean-path", type=Path, default=None)
    parser.add_argument("--daily-panel-path", type=Path, default=None)
    parser.add_argument(
        "--text-features-full-10k-path",
        type=Path,
        default=None,
        help=(
            "Optional reusable LM2011 full-10-K text-features parquet. If omitted, the runner "
            "auto-reuses <output-dir>/lm2011_text_features_full_10k.parquet when present."
        ),
    )
    parser.add_argument(
        "--text-features-mda-path",
        type=Path,
        default=None,
        help=(
            "Optional reusable LM2011 MD&A text-features parquet. If omitted, the runner "
            "auto-reuses <output-dir>/lm2011_text_features_mda.parquet when present."
        ),
    )
    parser.add_argument(
        "--recompute-text-features-full-10k",
        action="store_true",
        help=(
            "Force recomputation of full-10-K text features by ignoring any reusable "
            "lm2011_text_features_full_10k.parquet artifact."
        ),
    )
    parser.add_argument(
        "--recompute-text-features-mda",
        action="store_true",
        help=(
            "Force recomputation of MD&A text features by ignoring any reusable "
            "lm2011_text_features_mda.parquet artifact."
        ),
    )
    parser.add_argument(
        "--recompute-event-screen-surface",
        action="store_true",
        help=(
            "Force recomputation of the LM2011 event-screen surface by ignoring any reusable "
            "lm2011_event_screen_surface.parquet artifact."
        ),
    )
    parser.add_argument(
        "--recompute-event-panel",
        action="store_true",
        help=(
            "Force recomputation of the LM2011 event panel by ignoring any reusable "
            "lm2011_event_panel.parquet artifact."
        ),
    )
    parser.add_argument(
        "--recompute-regression-tables",
        action="store_true",
        help=(
            "Force recomputation of the LM2011 final regression tables by ignoring any reusable "
            "result-table artifacts in the output directory."
        ),
    )
    parser.add_argument("--items-analysis-dir", type=Path, default=None)
    parser.add_argument("--ccm-base-dir", type=Path, default=None)
    parser.add_argument(
        "--doc-ownership-path",
        type=Path,
        default=None,
        help=(
            "Optional finalized LM2011 document-ownership parquet. If omitted, the runner "
            "uses <upstream-run-root>/refinitiv_doc_ownership_lm2011/refinitiv_lm2011_doc_ownership.parquet."
        ),
    )
    parser.add_argument(
        "--doc-analyst-selected-path",
        type=Path,
        default=None,
        help=(
            "Optional finalized LM2011 document-analyst parquet. If omitted, the runner "
            "uses <upstream-run-root>/refinitiv_doc_analyst_lm2011/refinitiv_doc_analyst_selected.parquet."
        ),
    )
    parser.add_argument(
        "--monthly-stock-path",
        type=Path,
        default=None,
        help="Optional monthly stock returns parquet. If omitted, monthly strategy stages are skipped.",
    )
    parser.add_argument(
        "--ff-monthly-with-mom-path",
        type=Path,
        default=None,
        help="Optional monthly FF+momentum parquet. If omitted, Table IA.II is skipped.",
    )
    parser.add_argument(
        "--full-10k-cleaning-contract",
        default=DEFAULT_LM2011_FULL_10K_CLEANING_CONTRACT,
        choices=FULL_10K_CLEANING_CONTRACTS,
        help="Full-10-K cleaning contract for paper-faithful sample reruns.",
    )
    parser.add_argument(
        "--full-10k-text-feature-batch-size",
        type=int,
        default=None,
        help="Maximum number of documents per microbatch when scoring LM2011 full-10-K text features.",
    )
    parser.add_argument(
        "--mda-text-feature-batch-size",
        type=int,
        default=None,
        help="Maximum number of documents per microbatch when scoring LM2011 MD&A text features.",
    )
    parser.add_argument(
        "--text-feature-batch-size",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--event-window-doc-batch-size",
        type=int,
        default=DEFAULT_LM2011_EVENT_WINDOW_DOC_BATCH_SIZE,
        help="Number of documents per batch when building LM2011 event-window market surfaces.",
    )
    parser.add_argument(
        "--print-ram-stats",
        action="store_true",
        help="Print Linux /proc-based RAM snapshots around memory-heavy LM2011 stages.",
    )
    parser.add_argument(
        "--ram-log-interval-batches",
        type=int,
        default=DEFAULT_RAM_LOG_INTERVAL_BATCHES,
        help="Emit LM2011 batch progress logs every N batches.",
    )
    return parser.parse_args(argv)


def _first_existing_path(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(f"No existing path found among: {[str(path) for path in paths]}")


def _resolve_ccm_parquet_artifact(base_dir: Path, parquet_name: str) -> Path:
    candidates = [base_dir / parquet_name]
    candidates.extend(
        sorted(
            (child / parquet_name for child in base_dir.glob("documents-export*") if child.is_dir()),
            key=lambda path: str(path),
        )
    )
    return _first_existing_path(*candidates)


def _resolve_optional_ccm_parquet_artifact(base_dir: Path, parquet_names: Sequence[str]) -> Path | None:
    for parquet_name in parquet_names:
        try:
            return _resolve_ccm_parquet_artifact(base_dir, parquet_name)
        except FileNotFoundError:
            continue
    return None


def _resolve_optional_existing_path(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _parquet_glob_exists(directory: Path, pattern: str) -> bool:
    return any(directory.glob(pattern))


def _absolute_path_str(path: Path | None) -> str | None:
    if path is None:
        return None
    return str(path.resolve())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _elapsed_seconds(started_at_utc: str, completed_at_utc: str) -> int | None:
    try:
        started_at = dt.datetime.fromisoformat(started_at_utc)
        completed_at = dt.datetime.fromisoformat(completed_at_utc)
    except ValueError:
        return None
    return max(int((completed_at - started_at).total_seconds()), 0)


def _read_proc_kb_map(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    values: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        parts = raw_value.strip().split()
        if not parts:
            continue
        try:
            values[key.strip()] = int(parts[0])
        except ValueError:
            continue
    return values


def _kb_to_gib(value_kb: int | None) -> float | None:
    if value_kb is None:
        return None
    return round(float(value_kb) / 1024.0 / 1024.0, 3)


def _bytes_to_gib(value_bytes: int | None) -> float | None:
    if value_bytes is None:
        return None
    return round(float(value_bytes) / 1024.0 / 1024.0 / 1024.0, 3)


def _read_optional_int(path: Path) -> int | None:
    if not path.exists():
        return None
    raw_value = path.read_text(encoding="utf-8").strip()
    if raw_value == "" or raw_value.lower() == "max":
        return None
    try:
        return int(raw_value)
    except ValueError:
        return None


def _read_cgroup_memory_bytes() -> dict[str, int | None]:
    v2_limit_path = Path("/sys/fs/cgroup/memory.max")
    v2_current_path = Path("/sys/fs/cgroup/memory.current")
    if v2_limit_path.exists() and v2_current_path.exists():
        return {
            "cgroup_limit_bytes": _read_optional_int(v2_limit_path),
            "cgroup_used_bytes": _read_optional_int(v2_current_path),
        }
    v1_limit_path = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    v1_current_path = Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    return {
        "cgroup_limit_bytes": _read_optional_int(v1_limit_path),
        "cgroup_used_bytes": _read_optional_int(v1_current_path),
    }


def _ram_snapshot(label: str) -> dict[str, object]:
    payload: dict[str, object] = {"label": label}
    meminfo = _read_proc_kb_map(Path("/proc/meminfo"))
    status = _read_proc_kb_map(Path("/proc/self/status"))
    cgroup = _read_cgroup_memory_bytes()
    if not meminfo and not status and all(value is None for value in cgroup.values()):
        payload["ram_stats_unavailable"] = True
        return payload

    mem_total_kb = meminfo.get("MemTotal")
    mem_available_kb = meminfo.get("MemAvailable")
    payload["process_rss_gb"] = _kb_to_gib(status.get("VmRSS"))
    payload["process_hwm_gb"] = _kb_to_gib(status.get("VmHWM"))
    payload["system_total_gb"] = _kb_to_gib(mem_total_kb)
    payload["system_available_gb"] = _kb_to_gib(mem_available_kb)
    if mem_total_kb is not None and mem_available_kb is not None:
        payload["system_used_gb"] = _kb_to_gib(max(mem_total_kb - mem_available_kb, 0))
    cgroup_limit_bytes = cgroup.get("cgroup_limit_bytes")
    cgroup_used_bytes = cgroup.get("cgroup_used_bytes")
    payload["cgroup_limit_gb"] = _bytes_to_gib(cgroup_limit_bytes)
    payload["cgroup_used_gb"] = _bytes_to_gib(cgroup_used_bytes)
    if cgroup_limit_bytes is not None and cgroup_used_bytes is not None:
        payload["cgroup_available_gb"] = _bytes_to_gib(max(cgroup_limit_bytes - cgroup_used_bytes, 0))
    return payload


def _print_ram_snapshot(label: str, *, enabled: bool) -> None:
    if not enabled:
        return
    print(_ram_snapshot(label))


def _gc_cleanup(label: str, *, enabled: bool) -> None:
    _print_ram_snapshot(f"{label}_before_gc", enabled=enabled)
    gc.collect()
    _print_ram_snapshot(f"{label}_after_gc", enabled=enabled)


def _should_log_periodic_batch(
    *,
    batch_index: int,
    total_batches: int | None,
    interval: int,
) -> bool:
    if batch_index <= 1:
        return True
    if total_batches is not None and batch_index >= total_batches:
        return True
    return batch_index % max(interval, 1) == 0


def _checkpoint_manifest(manifest: dict[str, Any], manifest_path: Path) -> None:
    manifest["last_checkpoint_at_utc"] = _utc_timestamp()
    _write_json(manifest_path, manifest)


def _normalize_extension_text_scope_value(value: str) -> str:
    raw = value.strip().casefold().replace("-", "_")
    if raw in {"7", "item_7", "mda_item_7", "item_7_mda"}:
        return "item_7_mda"
    if raw in {"1a", "item_1a", "item_1a_risk_factors"}:
        return "item_1a_risk_factors"
    if raw in {"1", "item_1", "item_1_business"}:
        return "item_1_business"
    if raw in {"items_1_1a_7_concat", "item_1_item_1a_item_7_concat"}:
        return "items_1_1a_7_concat"
    return raw


def _normalized_extension_text_scopes(text_scopes: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(_normalize_extension_text_scope_value(scope) for scope in text_scopes))
    if not normalized:
        raise ValueError("LM2011 extension text_scopes must be non-empty.")
    return normalized


def _extension_artifact_output_path(output_dir: Path, stage_name: str) -> Path:
    return output_dir / EXTENSION_STAGE_ARTIFACT_FILENAMES[stage_name]


def _read_json_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_manifest_path_value(
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    value: object,
) -> Path | None:
    if value is None:
        return None
    raw = str(value).strip()
    if raw == "":
        return None
    candidate = Path(raw).expanduser()
    if str(manifest.get("path_semantics") or "") == "manifest_relative_v1" and not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    return candidate.resolve()


def _resolve_finbert_extension_inputs(
    run_cfg: LM2011ExtensionRunConfig,
) -> dict[str, Any]:
    analysis_run_dir = (
        Path(run_cfg.finbert_analysis_run_dir).resolve()
        if run_cfg.finbert_analysis_run_dir is not None
        else None
    )
    analysis_manifest_path = (
        Path(run_cfg.finbert_analysis_manifest_path).resolve()
        if run_cfg.finbert_analysis_manifest_path is not None
        else (
            (analysis_run_dir / "run_manifest.json").resolve()
            if analysis_run_dir is not None
            else None
        )
    )
    analysis_manifest: dict[str, Any] | None = None
    if analysis_manifest_path is not None:
        if not analysis_manifest_path.exists():
            raise FileNotFoundError(
                f"FinBERT analysis manifest not found: {analysis_manifest_path}"
            )
        analysis_manifest = _read_json_payload(analysis_manifest_path)
        analysis_run_dir = analysis_manifest_path.parent.resolve()
    item_features_long_path = (
        Path(run_cfg.finbert_item_features_long_path).resolve()
        if run_cfg.finbert_item_features_long_path is not None
        else None
    )
    if item_features_long_path is None and analysis_manifest is not None:
        item_features_long_path = _resolve_manifest_path_value(
            manifest=analysis_manifest,
            manifest_path=analysis_manifest_path,
            value=((analysis_manifest.get("artifacts") or {}).get("item_features_long_path")),
        )
    if item_features_long_path is None and analysis_run_dir is not None:
        candidate = analysis_run_dir / "item_features_long.parquet"
        if candidate.exists():
            item_features_long_path = candidate.resolve()
    if item_features_long_path is None or not item_features_long_path.exists():
        raise FileNotFoundError(
            "LM2011 extension requires FinBERT item_features_long.parquet, "
            f"but it was not resolved from {analysis_run_dir or run_cfg.finbert_item_features_long_path}."
        )

    preprocessing_run_dir = (
        Path(run_cfg.finbert_preprocessing_run_dir).resolve()
        if run_cfg.finbert_preprocessing_run_dir is not None
        else None
    )
    preprocessing_manifest_path = (
        Path(run_cfg.finbert_preprocessing_manifest_path).resolve()
        if run_cfg.finbert_preprocessing_manifest_path is not None
        else (
            (preprocessing_run_dir / "run_manifest.json").resolve()
            if preprocessing_run_dir is not None
            else None
        )
    )
    preprocessing_manifest: dict[str, Any] | None = None
    if preprocessing_manifest_path is not None:
        if not preprocessing_manifest_path.exists():
            raise FileNotFoundError(
                f"FinBERT preprocessing manifest not found: {preprocessing_manifest_path}"
            )
        preprocessing_manifest = _read_json_payload(preprocessing_manifest_path)
        preprocessing_run_dir = preprocessing_manifest_path.parent.resolve()

    cleaned_item_scopes_dir = (
        Path(run_cfg.finbert_cleaned_item_scopes_dir).resolve()
        if run_cfg.finbert_cleaned_item_scopes_dir is not None
        else None
    )
    if cleaned_item_scopes_dir is None and preprocessing_manifest is not None:
        cleaned_item_scopes_dir = _resolve_manifest_path_value(
            manifest=preprocessing_manifest,
            manifest_path=preprocessing_manifest_path,
            value=((preprocessing_manifest.get("artifacts") or {}).get("cleaned_item_scopes_dir")),
        )
    if cleaned_item_scopes_dir is None and preprocessing_run_dir is not None:
        candidate = preprocessing_run_dir / "cleaned_item_scopes" / "by_year"
        if candidate.exists():
            cleaned_item_scopes_dir = candidate.resolve()
    if cleaned_item_scopes_dir is not None and not cleaned_item_scopes_dir.exists():
        raise FileNotFoundError(
            f"Resolved FinBERT cleaned_item_scopes_dir does not exist: {cleaned_item_scopes_dir}"
        )

    return {
        "analysis_run_dir": analysis_run_dir,
        "analysis_manifest_path": analysis_manifest_path,
        "analysis_manifest": analysis_manifest,
        "item_features_long_path": item_features_long_path,
        "preprocessing_run_dir": preprocessing_run_dir,
        "preprocessing_manifest_path": preprocessing_manifest_path,
        "preprocessing_manifest": preprocessing_manifest,
        "cleaned_item_scopes_dir": cleaned_item_scopes_dir,
    }


def _build_extension_manifest(
    run_cfg: LM2011ExtensionRunConfig,
    *,
    resolved_finbert_inputs: dict[str, Any],
) -> dict[str, Any]:
    started_at_utc = _utc_timestamp()
    return {
        "runner_name": "lm2011_extension_runner",
        "generated_at_utc": started_at_utc,
        "started_at_utc": started_at_utc,
        "completed_at_utc": None,
        "elapsed_seconds": None,
        "failed_stage": None,
        "run_status": "running",
        "roots": {
            "output_dir": _absolute_path_str(run_cfg.output_dir),
            "additional_data_dir": _absolute_path_str(run_cfg.additional_data_dir),
        },
        "config": {
            "run_id": run_cfg.run_id,
            "note": run_cfg.note,
            "require_cleaned_scope_match": run_cfg.require_cleaned_scope_match,
            "dictionary_source_mode": run_cfg.dictionary_source_mode,
            "text_scopes": list(_normalized_extension_text_scopes(run_cfg.text_scopes)),
        },
        "resolved_inputs": {
            "items_analysis_dir": _absolute_path_str(run_cfg.items_analysis_dir),
            "event_panel_path": _absolute_path_str(run_cfg.event_panel_path),
            "company_history_path": _absolute_path_str(run_cfg.company_history_path),
            "company_description_path": _absolute_path_str(run_cfg.company_description_path),
            "ff48_siccodes_path": _absolute_path_str(run_cfg.ff48_siccodes_path),
            "finbert_analysis_run_dir": _absolute_path_str(resolved_finbert_inputs["analysis_run_dir"]),
            "finbert_analysis_manifest_path": _absolute_path_str(
                resolved_finbert_inputs["analysis_manifest_path"]
            ),
            "finbert_item_features_long_path": _absolute_path_str(
                resolved_finbert_inputs["item_features_long_path"]
            ),
            "finbert_preprocessing_run_dir": _absolute_path_str(
                resolved_finbert_inputs["preprocessing_run_dir"]
            ),
            "finbert_preprocessing_manifest_path": _absolute_path_str(
                resolved_finbert_inputs["preprocessing_manifest_path"]
            ),
            "finbert_cleaned_item_scopes_dir": _absolute_path_str(
                resolved_finbert_inputs["cleaned_item_scopes_dir"]
            ),
        },
        "artifacts": {},
        "row_counts": {},
        "stages": {},
        "dictionary_inputs": {},
    }


def _record_extension_stage_failed(
    manifest: dict[str, Any],
    *,
    output_dir: Path,
    stage_name: str,
    exc: Exception,
    manifest_path: Path | None = None,
) -> None:
    artifact_path = (
        _absolute_path_str(_extension_artifact_output_path(output_dir, stage_name))
        if stage_name in EXTENSION_STAGE_ARTIFACT_FILENAMES
        else None
    )
    manifest["stages"][stage_name] = {
        "status": "failed",
        "artifact_path": artifact_path,
        "row_count": None,
        "reason": str(exc),
        "completed_at_utc": _utc_timestamp(),
    }
    if manifest_path is not None:
        _checkpoint_manifest(manifest, manifest_path)


def _write_extension_stage(
    manifest: dict[str, Any],
    *,
    manifest_path: Path | None,
    output_dir: Path,
    stage_name: str,
    frame: pl.LazyFrame | pl.DataFrame,
    empty_reason: str | None = None,
    extra_artifacts: dict[str, Path] | None = None,
    warnings: Sequence[str] | None = None,
) -> pl.LazyFrame:
    artifact_path = _extension_artifact_output_path(output_dir, stage_name)
    written_path, row_count = _write_frame_artifact(frame, artifact_path)
    _record_stage_success(
        manifest,
        stage_name=stage_name,
        artifact_path=written_path,
        row_count=row_count,
        empty_reason=empty_reason,
        extra_artifacts=extra_artifacts,
        warnings=warnings,
    )
    if manifest_path is not None:
        _checkpoint_manifest(manifest, manifest_path)
    return pl.scan_parquet(written_path)


def _scan_year_sharded_parquet_dir(directory: Path, *, label: str) -> pl.LazyFrame:
    if not directory.exists() or not _parquet_glob_exists(directory, "*.parquet"):
        raise FileNotFoundError(f"{label} not found or empty: {directory}")
    return pl.scan_parquet(str(directory / "*.parquet"))


def _coalesce_existing_expr(
    schema: pl.Schema,
    candidates: Sequence[str],
    *,
    dtype: pl.DataType,
) -> pl.Expr:
    exprs = [
        pl.col(column).cast(dtype, strict=False)
        for column in candidates
        if column in schema
    ]
    return pl.coalesce(exprs) if exprs else pl.lit(None, dtype=dtype)


def _build_extension_finbert_surface_lf(
    item_features_lf: pl.LazyFrame,
    event_doc_ids_lf: pl.LazyFrame,
    *,
    text_scopes: Sequence[str],
) -> pl.LazyFrame:
    schema = item_features_lf.collect_schema()
    if "text_scope" in schema:
        raw_scope_expr = pl.col("text_scope")
    elif "benchmark_item_code" in schema:
        raw_scope_expr = pl.col("benchmark_item_code")
    elif "item_id" in schema:
        raw_scope_expr = pl.col("item_id")
    else:
        raise ValueError(
            "FinBERT item features must contain text_scope, benchmark_item_code, or item_id."
        )
    normalized_text_scopes = list(_normalized_extension_text_scopes(text_scopes))
    return (
        item_features_lf.join(event_doc_ids_lf, on="doc_id", how="inner")
        .select(
            pl.col("doc_id").cast(pl.Utf8, strict=False),
            (
                pl.col("filing_date").cast(pl.Date, strict=False)
                if "filing_date" in schema
                else pl.lit(None, dtype=pl.Date)
            ).alias("filing_date"),
            normalize_lm2011_extension_text_scope_expr(raw_scope_expr).alias("text_scope"),
            (
                pl.col("cleaning_policy_id").cast(pl.Utf8, strict=False)
                if "cleaning_policy_id" in schema
                else pl.lit(None, dtype=pl.Utf8)
            ).alias("cleaning_policy_id"),
            (
                pl.col("model_name").cast(pl.Utf8, strict=False)
                if "model_name" in schema
                else pl.lit(None, dtype=pl.Utf8)
            ).alias("model_name"),
            (
                pl.col("model_version").cast(pl.Utf8, strict=False)
                if "model_version" in schema
                else pl.lit(None, dtype=pl.Utf8)
            ).alias("model_version"),
            (
                pl.col("segment_policy_id").cast(pl.Utf8, strict=False)
                if "segment_policy_id" in schema
                else pl.lit(None, dtype=pl.Utf8)
            ).alias("segment_policy_id"),
            _coalesce_existing_expr(
                schema,
                ("finbert_segment_count", "sentence_count"),
                dtype=pl.Int32,
            ).alias("finbert_segment_count"),
            _coalesce_existing_expr(
                schema,
                ("finbert_token_count_512", "finbert_token_count_512_sum"),
                dtype=pl.Int64,
            ).alias("finbert_token_count_512"),
            _coalesce_existing_expr(
                schema,
                ("finbert_token_count_512_sum", "finbert_token_count_512"),
                dtype=pl.Int64,
            ).alias("finbert_token_count_512_sum"),
            *[
                _coalesce_existing_expr(schema, (column,), dtype=pl.Float64).alias(column)
                for column in (
                    "finbert_neg_prob_lenw_mean",
                    "finbert_pos_prob_lenw_mean",
                    "finbert_neu_prob_lenw_mean",
                    "finbert_net_negative_lenw_mean",
                    "finbert_neg_dominant_share",
                )
            ],
        )
        .filter(pl.col("text_scope").is_in(normalized_text_scopes))
        .unique(subset=["doc_id", "text_scope"], keep="first")
    )


def _build_extension_dictionary_surface_lf(
    run_cfg: LM2011ExtensionRunConfig,
    *,
    cleaned_item_scopes_dir: Path | None,
    event_doc_ids_lf: pl.LazyFrame,
    dictionary_inputs: Any,
) -> pl.LazyFrame:
    normalized_text_scopes = _normalized_extension_text_scopes(run_cfg.text_scopes)
    if run_cfg.dictionary_source_mode == EXTENSION_DICTIONARY_SOURCE_PREFER_CLEANED:
        if cleaned_item_scopes_dir is not None:
            cleaned_scopes_lf = _scan_year_sharded_parquet_dir(
                cleaned_item_scopes_dir,
                label="FinBERT cleaned_item_scopes_dir",
            )
            cleaned_scopes_lf = (
                cleaned_scopes_lf.join(event_doc_ids_lf, on="doc_id", how="inner")
                .with_columns(
                    normalize_lm2011_extension_text_scope_expr(pl.col("text_scope")).alias("text_scope")
                )
                .filter(pl.col("text_scope").is_in(list(normalized_text_scopes)))
            )
            return build_lm2011_extension_dictionary_features_from_cleaned_scopes(
                cleaned_scopes_lf,
                dictionary_lists=dictionary_inputs.dictionary_lists,
                harvard_negative_word_list=dictionary_inputs.harvard_negative_word_list,
                master_dictionary_words=dictionary_inputs.master_dictionary_words,
            ).filter(pl.col("text_scope").is_in(list(normalized_text_scopes)))
        if run_cfg.require_cleaned_scope_match:
            raise FileNotFoundError(
                "LM2011 extension requires FinBERT cleaned_item_scopes_dir for strict matched comparison."
            )

    if not _parquet_glob_exists(run_cfg.items_analysis_dir, ITEMS_ANALYSIS_GLOB):
        raise FileNotFoundError(
            f"LM2011 extension raw items_analysis directory not found or empty: {run_cfg.items_analysis_dir}"
        )
    items_analysis_lf = _prepare_lm2011_sec_input_lf(
        pl.scan_parquet(str(run_cfg.items_analysis_dir / ITEMS_ANALYSIS_GLOB))
    )
    items_analysis_lf = items_analysis_lf.join(event_doc_ids_lf, on="doc_id", how="inner")
    text_scope_item_ids = {
        text_scope: item_id
        for text_scope, item_id in EXTENSION_ITEM_SCOPE_IDS.items()
        if text_scope in normalized_text_scopes
    }
    return build_lm2011_extension_dictionary_features(
        items_analysis_lf,
        dictionary_lists=dictionary_inputs.dictionary_lists,
        harvard_negative_word_list=dictionary_inputs.harvard_negative_word_list,
        master_dictionary_words=dictionary_inputs.master_dictionary_words,
        text_scope_item_ids=text_scope_item_ids,
    ).filter(pl.col("text_scope").is_in(list(normalized_text_scopes)))


def _validate_extension_comparison_mode(
    run_cfg: LM2011ExtensionRunConfig,
    *,
    cleaned_item_scopes_dir: Path | None,
) -> None:
    if run_cfg.dictionary_source_mode != EXTENSION_DICTIONARY_SOURCE_PREFER_CLEANED:
        return
    if cleaned_item_scopes_dir is not None:
        return
    if run_cfg.require_cleaned_scope_match:
        raise FileNotFoundError(
            "LM2011 extension requires FinBERT cleaned_item_scopes_dir for strict matched comparison."
        )
    raise ValueError(
        "LM2011 extension relaxed raw-item fallback is not supported in the current "
        "matched dictionary-versus-FinBERT runner. The extension analysis panel requires "
        "both surfaces to carry aligned cleaned-scope cleaning_policy_id metadata. "
        "Provide cleaned_item_scopes_dir or keep require_cleaned_scope_match=True."
    )


def _resolve_paths(args: argparse.Namespace) -> RunnerPaths:
    sample_root = Path(args.sample_root).resolve()
    upstream_run_root = Path(args.upstream_run_root).resolve()
    additional_data_dir = Path(args.additional_data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    local_work_root = Path(args.local_work_root).resolve()
    legacy_text_feature_batch_size = (
        int(args.text_feature_batch_size)
        if args.text_feature_batch_size is not None
        else None
    )
    full_10k_text_feature_batch_size = int(
        args.full_10k_text_feature_batch_size
        if args.full_10k_text_feature_batch_size is not None
        else (
            legacy_text_feature_batch_size
            if legacy_text_feature_batch_size is not None
            else DEFAULT_LM2011_FULL_10K_TEXT_FEATURE_BATCH_SIZE
        )
    )
    mda_text_feature_batch_size = int(
        args.mda_text_feature_batch_size
        if args.mda_text_feature_batch_size is not None
        else (
            legacy_text_feature_batch_size
            if legacy_text_feature_batch_size is not None
            else DEFAULT_LM2011_MDA_TEXT_FEATURE_BATCH_SIZE
        )
    )
    if full_10k_text_feature_batch_size < 1:
        raise ValueError("--full-10k-text-feature-batch-size must be >= 1")
    if mda_text_feature_batch_size < 1:
        raise ValueError("--mda-text-feature-batch-size must be >= 1")
    if int(args.event_window_doc_batch_size) < 1:
        raise ValueError("--event-window-doc-batch-size must be >= 1")
    if int(args.ram_log_interval_batches) < 1:
        raise ValueError("--ram-log-interval-batches must be >= 1")
    if args.recompute_text_features_full_10k and args.text_features_full_10k_path is not None:
        raise ValueError(
            "--recompute-text-features-full-10k cannot be combined with "
            "--text-features-full-10k-path"
        )
    if args.recompute_text_features_mda and args.text_features_mda_path is not None:
        raise ValueError(
            "--recompute-text-features-mda cannot be combined with "
            "--text-features-mda-path"
        )
    year_merged_dir = (
        Path(args.year_merged_dir).resolve()
        if args.year_merged_dir is not None
        else (
            _resolve_optional_existing_path(
                sample_root / "year_merged",
                sample_root / "parquet_data" / "_year_merged",
                sample_root / "Data" / "Sample_Filings" / "parquet_batches" / "_year_merged",
            )
            or sample_root / "year_merged"
        )
    )
    daily_panel_path = (
        Path(args.daily_panel_path).resolve()
        if args.daily_panel_path is not None
        else (
            _resolve_optional_existing_path(
                sample_root / "derived_data" / "final_flagged_data_compdesc_added.sample_5pct_seed42.parquet",
                sample_root / "derived_data" / "final_flagged_data_compdesc_added.parquet",
                sample_root / "CRSP_Compustat_data" / "derived_data" / "final_flagged_data_compdesc_added.parquet",
                sample_root / "Data" / "CRSP_Compustat_data" / "derived_data" / "final_flagged_data_compdesc_added.parquet",
            )
            or sample_root / "derived_data" / "final_flagged_data_compdesc_added.sample_5pct_seed42.parquet"
        )
    )
    ccm_base_dir = (
        Path(args.ccm_base_dir).resolve()
        if args.ccm_base_dir is not None
        else (
            _resolve_optional_existing_path(
                sample_root / "ccm_parquet_data",
                sample_root / "CRSP_Compustat_data" / "parquet_data",
                sample_root / "Data" / "CRSP_Compustat_data" / "parquet_data",
            )
            or sample_root / "ccm_parquet_data"
        )
    )
    auto_sample_backbone_path = upstream_run_root / "sec_ccm_premerge" / "lm2011_sample_backbone.parquet"
    sample_backbone_path = (
        Path(args.sample_backbone_path).resolve()
        if args.sample_backbone_path is not None
        else (auto_sample_backbone_path.resolve() if auto_sample_backbone_path.exists() else None)
    )
    auto_text_features_full_10k_path = output_dir / STAGE_ARTIFACT_FILENAMES["text_features_full_10k"]
    text_features_full_10k_path_is_explicit = (
        (args.text_features_full_10k_path is not None) and not args.recompute_text_features_full_10k
    )
    text_features_full_10k_path = (
        None
        if args.recompute_text_features_full_10k
        else (
            Path(args.text_features_full_10k_path).resolve()
            if args.text_features_full_10k_path is not None
            else (
                auto_text_features_full_10k_path.resolve()
                if auto_text_features_full_10k_path.exists()
                else None
            )
        )
    )
    auto_text_features_mda_path = output_dir / STAGE_ARTIFACT_FILENAMES["text_features_mda"]
    text_features_mda_path_is_explicit = (
        (args.text_features_mda_path is not None) and not args.recompute_text_features_mda
    )
    text_features_mda_path = (
        None
        if args.recompute_text_features_mda
        else (
            Path(args.text_features_mda_path).resolve()
            if args.text_features_mda_path is not None
            else (
                auto_text_features_mda_path.resolve()
                if auto_text_features_mda_path.exists()
                else None
            )
        )
    )
    matched_clean_path = (
        Path(args.matched_clean_path).resolve()
        if args.matched_clean_path is not None
        else upstream_run_root / "sec_ccm_premerge" / "sec_ccm_matched_clean.parquet"
    )
    items_analysis_dir = (
        Path(args.items_analysis_dir).resolve()
        if args.items_analysis_dir is not None
        else upstream_run_root / "items_analysis"
    )
    doc_ownership_path = (
        Path(args.doc_ownership_path).resolve()
        if args.doc_ownership_path is not None
        else (
            upstream_run_root
            / "refinitiv_doc_ownership_lm2011"
            / "refinitiv_lm2011_doc_ownership.parquet"
        )
    )
    doc_analyst_selected_path = (
        Path(args.doc_analyst_selected_path).resolve()
        if args.doc_analyst_selected_path is not None
        else (
            upstream_run_root
            / "refinitiv_doc_analyst_lm2011"
            / "refinitiv_doc_analyst_selected.parquet"
        )
    )
    monthly_stock_path = (
        Path(args.monthly_stock_path).resolve()
        if args.monthly_stock_path is not None
        else _resolve_optional_ccm_parquet_artifact(ccm_base_dir, MONTHLY_STOCK_CANDIDATES)
    )
    ff_monthly_with_mom_path = (
        Path(args.ff_monthly_with_mom_path).resolve()
        if args.ff_monthly_with_mom_path is not None
        else None
    )
    return RunnerPaths(
        sample_root=sample_root,
        upstream_run_root=upstream_run_root,
        additional_data_dir=additional_data_dir,
        output_dir=output_dir,
        local_work_root=local_work_root,
        year_merged_dir=year_merged_dir,
        sample_backbone_path=sample_backbone_path,
        daily_panel_path=daily_panel_path,
        text_features_full_10k_path=text_features_full_10k_path,
        text_features_full_10k_path_is_explicit=text_features_full_10k_path_is_explicit,
        text_features_mda_path=text_features_mda_path,
        text_features_mda_path_is_explicit=text_features_mda_path_is_explicit,
        ccm_base_dir=ccm_base_dir,
        matched_clean_path=matched_clean_path,
        items_analysis_dir=items_analysis_dir,
        doc_ownership_path=doc_ownership_path,
        doc_analyst_selected_path=doc_analyst_selected_path,
        filingdates_path=_resolve_optional_ccm_parquet_artifact(ccm_base_dir, ("filingdates.parquet",)),
        quarterly_balance_sheet_path=_resolve_optional_ccm_parquet_artifact(
            ccm_base_dir,
            ("balancesheetquarterly.parquet",),
        ),
        quarterly_income_statement_path=_resolve_optional_ccm_parquet_artifact(
            ccm_base_dir,
            ("incomestatementquarterly.parquet",),
        ),
        quarterly_period_descriptor_path=_resolve_optional_ccm_parquet_artifact(
            ccm_base_dir,
            ("perioddescriptorquarterly.parquet",),
        ),
        annual_balance_sheet_path=_resolve_optional_ccm_parquet_artifact(
            ccm_base_dir,
            ("balancesheetindustrialannual.parquet",),
        ),
        annual_income_statement_path=_resolve_optional_ccm_parquet_artifact(
            ccm_base_dir,
            ("incomestatementindustrialannual.parquet",),
        ),
        annual_period_descriptor_path=_resolve_optional_ccm_parquet_artifact(
            ccm_base_dir,
            ("perioddescriptorannual.parquet",),
        ),
        annual_fiscal_market_path=_resolve_optional_ccm_parquet_artifact(
            ccm_base_dir,
            ("fiscalmarketdataannual.parquet",),
        ),
        company_history_path=_resolve_optional_ccm_parquet_artifact(
            ccm_base_dir,
            ("companyhistory.parquet",),
        ),
        company_description_path=_resolve_optional_ccm_parquet_artifact(
            ccm_base_dir,
            ("companydescription.parquet",),
        ),
        ff_daily_csv_path=additional_data_dir / "F-F_Research_Data_Factors_daily.csv",
        ff_monthly_csv_path=additional_data_dir / "F-F_Research_Data_Factors.csv",
        momentum_monthly_csv_path=additional_data_dir / "F-F_Momentum_Factor.csv",
        ff48_siccodes_path=additional_data_dir / "FF_Siccodes_48_Industries.txt",
        monthly_stock_path=monthly_stock_path,
        ff_monthly_with_mom_path=ff_monthly_with_mom_path,
        full_10k_cleaning_contract=str(args.full_10k_cleaning_contract),
        full_10k_text_feature_batch_size=full_10k_text_feature_batch_size,
        mda_text_feature_batch_size=mda_text_feature_batch_size,
        recompute_event_screen_surface=bool(args.recompute_event_screen_surface),
        recompute_event_panel=bool(args.recompute_event_panel),
        recompute_regression_tables=bool(args.recompute_regression_tables),
        event_window_doc_batch_size=int(args.event_window_doc_batch_size),
        print_ram_stats=bool(args.print_ram_stats),
        ram_log_interval_batches=int(args.ram_log_interval_batches),
    )


def build_lm2011_post_refinitiv_run_config(
    args: argparse.Namespace,
    *,
    enabled_stages: Sequence[str] | None = None,
    fail_closed_for_enabled_stages: bool = False,
) -> LM2011PostRefinitivRunConfig:
    resolved_enabled_stages = (
        tuple(LM2011_ALL_STAGE_NAMES)
        if enabled_stages is None
        else tuple(enabled_stages)
    )
    return LM2011PostRefinitivRunConfig(
        paths=_resolve_paths(args),
        enabled_stages=resolved_enabled_stages,
        fail_closed_for_enabled_stages=fail_closed_for_enabled_stages,
    )


def _load_word_list(path: Path) -> tuple[str, ...]:
    return load_lm2011_word_list(path)


def _load_dictionary_lists(additional_data_dir: Path) -> tuple[dict[str, tuple[str, ...]], tuple[str, ...]]:
    dictionary_lists = {
        "negative": _load_word_list(additional_data_dir / "Fin-Neg.txt"),
        "positive": _load_word_list(additional_data_dir / "Fin-Pos.txt"),
        "uncertainty": _load_word_list(additional_data_dir / "Fin-Unc.txt"),
        "litigious": _load_word_list(additional_data_dir / "Fin-Lit.txt"),
        "modal_strong": _load_word_list(additional_data_dir / "MW-Strong.txt"),
        "modal_weak": _load_word_list(additional_data_dir / "MW-Weak.txt"),
    }
    harvard_negative_word_list = _load_word_list(additional_data_dir / "Harvard_IV_NEG_Inf.txt")
    return dictionary_lists, harvard_negative_word_list


def _load_master_dictionary_words(additional_data_dir: Path) -> tuple[str, ...]:
    words, _, _ = load_lm2011_master_dictionary_words(additional_data_dir)
    return words


def _normalize_filing_date_expr(schema_names: set[str]) -> pl.Expr:
    if "filing_date" in schema_names and "file_date_filename" in schema_names:
        return pl.coalesce(
            [
                pl.col("filing_date").cast(pl.Date, strict=False),
                pl.col("file_date_filename").cast(pl.Date, strict=False),
            ]
        ).alias("filing_date")
    if "filing_date" in schema_names:
        return pl.col("filing_date").cast(pl.Date, strict=False).alias("filing_date")
    if "file_date_filename" in schema_names:
        return pl.col("file_date_filename").cast(pl.Date, strict=False).alias("filing_date")
    raise ValueError("SEC input files must contain filing_date or file_date_filename.")


def _prepare_lm2011_sec_input_lf(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.with_columns(_normalize_filing_date_expr(set(lf.collect_schema().names())))


def _prepare_lm2011_sec_backbone_input_lf(lf: pl.LazyFrame) -> pl.LazyFrame:
    schema_names = set(lf.collect_schema().names())
    missing = [name for name in LM2011_SEC_BACKBONE_COLUMNS if name not in schema_names]
    if missing:
        raise ValueError(f"SEC input files missing LM2011 backbone columns: {missing}")
    return lf.select(
        *[pl.col(name) for name in LM2011_SEC_BACKBONE_COLUMNS],
        _normalize_filing_date_expr(schema_names),
    )


def _filter_to_sample_doc_ids_lf(lf: pl.LazyFrame, sample_backbone_lf: pl.LazyFrame) -> pl.LazyFrame:
    sample_doc_ids = (
        sample_backbone_lf.select(pl.col("doc_id").cast(pl.Utf8, strict=False).alias("doc_id"))
        .drop_nulls(subset=["doc_id"])
        .unique()
    )
    return (
        lf.with_columns(pl.col("doc_id").cast(pl.Utf8, strict=False).alias("doc_id"))
        .join(sample_doc_ids, on="doc_id", how="semi")
    )


def _filter_valid_annual_period_descriptor_rows(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.filter(
        pl.col("APDEDATE").cast(pl.Date, strict=False).is_not_null()
        | (
            (pl.col("FYEAR").cast(pl.Int32, strict=False) > 0)
            & pl.col("FYR").cast(pl.Int32, strict=False).is_between(1, 12, closed="both")
        )
    )


def _prepare_annual_accounting_inputs(
    annual_balance_sheet_path: Path,
    annual_income_statement_path: Path,
    annual_period_descriptor_path: Path,
    annual_fiscal_market_path: Path,
) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    annual_balance_sheet_lf = pl.scan_parquet(annual_balance_sheet_path).select(
        [pl.col(name) for name in ANNUAL_BALANCE_SHEET_COLUMNS]
    )
    annual_income_statement_lf = pl.scan_parquet(annual_income_statement_path).select(
        [pl.col(name) for name in ANNUAL_INCOME_STATEMENT_COLUMNS]
    )
    annual_period_descriptor_lf = _filter_valid_annual_period_descriptor_rows(
        pl.scan_parquet(annual_period_descriptor_path).select(
            [pl.col(name) for name in ANNUAL_PERIOD_DESCRIPTOR_COLUMNS]
        )
    )
    annual_fiscal_market_lf = pl.scan_parquet(annual_fiscal_market_path).select(
        [pl.col(name) for name in ANNUAL_FISCAL_MARKET_COLUMNS]
    )
    return (
        annual_balance_sheet_lf,
        annual_income_statement_lf,
        annual_period_descriptor_lf,
        annual_fiscal_market_lf,
    )


def _ken_french_data_skip_rows(csv_path: Path, header_columns: tuple[str, ...]) -> int:
    expected = ("", *header_columns)
    for line_number, line in enumerate(csv_path.read_text(encoding="utf-8-sig").splitlines()):
        cells = tuple(cell.strip().lower() for cell in line.split(",")[: len(expected)])
        if cells == tuple(cell.lower() for cell in expected):
            return line_number + 1
    raise ValueError(f"{csv_path} missing Ken French header row for columns: {list(header_columns)}")


def _factor_value_expr(column: str) -> pl.Expr:
    value = pl.col(column).str.strip_chars().cast(pl.Float64, strict=False)
    return (
        pl.when(value.is_in(KEN_FRENCH_MISSING_FACTOR_VALUES))
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise(value)
        .alias(column)
    )


def _load_ken_french_factor_csv_lf(
    csv_path: Path,
    *,
    raw_period_column: str,
    output_date_column: str,
    output_factor_columns: tuple[str, ...],
    header_columns: tuple[str, ...],
    date_format: str,
    date_len: int,
    month_end_dates: bool,
) -> pl.LazyFrame:
    columns = (raw_period_column, *output_factor_columns)
    parsed_date = pl.col(raw_period_column).str.strip_chars().str.strptime(pl.Date, date_format, strict=False)
    if month_end_dates:
        parsed_date = parsed_date.dt.month_end()
    stripped_period = pl.col(raw_period_column).str.strip_chars()
    return (
        pl.scan_csv(
            csv_path,
            skip_rows=_ken_french_data_skip_rows(csv_path, header_columns),
            has_header=False,
            new_columns=list(columns),
            schema_overrides={name: pl.Utf8 for name in columns},
            truncate_ragged_lines=True,
        )
        .filter((stripped_period.str.len_chars() == date_len) & stripped_period.str.contains(r"^\d+$"))
        .with_columns(
            parsed_date.alias(output_date_column),
            *[_factor_value_expr(column) for column in output_factor_columns],
        )
        .drop(raw_period_column)
        .drop_nulls(subset=[output_date_column, *output_factor_columns])
    )


def _load_ff_factors_daily_lf(csv_path: Path) -> pl.LazyFrame:
    return _load_ken_french_factor_csv_lf(
        csv_path,
        raw_period_column=FF_DAILY_COLUMNS[0],
        output_date_column="trading_date",
        output_factor_columns=FF_DAILY_COLUMNS[1:],
        header_columns=("Mkt-RF", "SMB", "HML", "RF"),
        date_format="%Y%m%d",
        date_len=8,
        month_end_dates=False,
    )


def _load_ff_factors_monthly_lf(csv_path: Path) -> pl.LazyFrame:
    return _load_ken_french_factor_csv_lf(
        csv_path,
        raw_period_column=FF_MONTHLY_COLUMNS[0],
        output_date_column="month_end",
        output_factor_columns=FF_MONTHLY_COLUMNS[1:],
        header_columns=("Mkt-RF", "SMB", "HML", "RF"),
        date_format="%Y%m",
        date_len=6,
        month_end_dates=True,
    )


def _load_momentum_factors_monthly_lf(csv_path: Path) -> pl.LazyFrame:
    return _load_ken_french_factor_csv_lf(
        csv_path,
        raw_period_column=MOMENTUM_MONTHLY_COLUMNS[0],
        output_date_column="month_end",
        output_factor_columns=MOMENTUM_MONTHLY_COLUMNS[1:],
        header_columns=("Mom",),
        date_format="%Y%m",
        date_len=6,
        month_end_dates=True,
    )


def _require_unique_month_end(lf: pl.LazyFrame, label: str) -> None:
    duplicate_months = (
        lf.group_by("month_end")
        .agg(pl.len().alias("_n"))
        .filter(pl.col("_n") > 1)
        .select("month_end")
        .head(10)
        .collect()
        .get_column("month_end")
        .to_list()
    )
    if duplicate_months:
        formatted = [
            value.isoformat() if hasattr(value, "isoformat") else str(value)
            for value in duplicate_months
        ]
        raise ValueError(f"{label} contains duplicate month_end values: {formatted}")


def _load_ff_factors_monthly_with_mom_lf(ff_csv_path: Path, mom_csv_path: Path) -> pl.LazyFrame:
    ff_lf = _load_ff_factors_monthly_lf(ff_csv_path)
    mom_lf = _load_momentum_factors_monthly_lf(mom_csv_path)
    _require_unique_month_end(ff_lf, "monthly FF factors")
    _require_unique_month_end(mom_lf, "monthly momentum factors")
    return (
        ff_lf.join(mom_lf, on="month_end", how="inner")
        .select("month_end", "mkt_rf", "smb", "hml", "rf", "mom")
        .sort("month_end")
    )


def _scan_ff_factors_monthly_with_mom_lf(parquet_path: Path) -> pl.LazyFrame:
    return pl.scan_parquet(parquet_path).select(
        pl.col("month_end").cast(pl.Date, strict=False).alias("month_end"),
        pl.col("mkt_rf").cast(pl.Float64, strict=False).alias("mkt_rf"),
        pl.col("smb").cast(pl.Float64, strict=False).alias("smb"),
        pl.col("hml").cast(pl.Float64, strict=False).alias("hml"),
        pl.col("rf").cast(pl.Float64, strict=False).alias("rf"),
        pl.col("mom").cast(pl.Float64, strict=False).alias("mom"),
    )


def _prepare_doc_analyst_sue_input_lf(selected_path: Path) -> pl.LazyFrame:
    return (
        pl.scan_parquet(selected_path)
        .filter(pl.col("analyst_match_status").cast(pl.Utf8, strict=False) == pl.lit("MATCHED"))
        .select(
            pl.col("gvkey_int").cast(pl.Int32, strict=False).alias("gvkey_int"),
            pl.col("matched_announcement_date").cast(pl.Date, strict=False).alias("announcement_date"),
            pl.col("matched_fiscal_period_end").cast(pl.Date, strict=False).alias("fiscal_period_end"),
            pl.col("actual_eps").cast(pl.Float64, strict=False).alias("actual_eps"),
            pl.col("forecast_consensus_mean").cast(pl.Float64, strict=False).alias("forecast_consensus_mean"),
            pl.col("forecast_dispersion").cast(pl.Float64, strict=False).alias("forecast_dispersion"),
            pl.col("forecast_revision_4m").cast(pl.Float64, strict=False).alias("forecast_revision_4m"),
            pl.col("forecast_revision_1m").cast(pl.Float64, strict=False).alias("forecast_revision_1m"),
        )
        .drop_nulls(subset=["gvkey_int", "announcement_date", "fiscal_period_end"])
        .unique(subset=["gvkey_int", "announcement_date", "fiscal_period_end"], keep="first")
    )


def _collect_frame(frame: pl.LazyFrame | pl.DataFrame) -> pl.DataFrame:
    if isinstance(frame, pl.LazyFrame):
        return frame.collect()
    return frame


def _write_frame_artifact(
    frame: pl.LazyFrame | pl.DataFrame,
    output_path: Path,
) -> tuple[Path, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(frame, pl.LazyFrame):
        if not hasattr(frame, "sink_parquet"):
            raise RuntimeError("Polars LazyFrame.sink_parquet is required for memory-safe stage writes.")
        frame.sink_parquet(output_path, compression=PARQUET_COMPRESSION)
        row_count = pl.scan_parquet(output_path).select(pl.len()).collect().item()
        return output_path, int(row_count)
    frame.write_parquet(output_path, compression=PARQUET_COMPRESSION)
    return output_path, int(frame.height)


def _artifact_output_path(output_dir: Path, stage_name: str) -> Path:
    return output_dir / STAGE_ARTIFACT_FILENAMES[stage_name]


def _format_table_i_sample_value(row: dict[str, Any]) -> str:
    if row["availability_status"] != "available" or row["sample_size_value"] is None:
        return "n/a"
    value = float(row["sample_size_value"])
    if row["sample_size_kind"] == "count":
        return f"{int(round(value)):,}"
    return f"{value:.2f}"


def _format_table_i_removed_value(row: dict[str, Any]) -> str:
    removed = row["observations_removed"]
    if removed is None or row["availability_status"] != "available":
        return ""
    return f"{int(removed):,}"


def _format_table_i_window_label(sample_start: dt.date, sample_end: dt.date) -> str:
    return f"{sample_start.year}-{sample_end.year}"


def _table_i_stage_warnings(table_df: pl.DataFrame) -> list[str]:
    unavailable_reasons = (
        table_df.filter(pl.col("availability_status") != "available")
        .select(pl.col("availability_reason").drop_nulls().unique().sort())
        .get_column("availability_reason")
        .to_list()
        if table_df.height > 0 and "availability_reason" in table_df.columns
        else []
    )
    warnings: list[str] = []
    for reason in unavailable_reasons:
        if reason == "mda_text_features_unavailable":
            warnings.append(
                "MD&A subsection rows are unavailable because lm2011_text_features_mda was not provided."
            )
        else:
            warnings.append(str(reason))
    return warnings


def _render_table_i_sample_creation_markdown(
    table_df: pl.DataFrame,
    *,
    sample_start: dt.date,
    sample_end: dt.date,
    warnings: Sequence[str] | None = None,
) -> str:
    lines = [
        f"# LM2011 Table I Sample Creation ({_format_table_i_window_label(sample_start, sample_end)})",
        "",
        "This table reports the impact of various data filters on initial 10-K sample size.",
        "",
    ]
    for section_label in (
        "Full 10-K Document",
        "Firm-Year Sample",
        "Management Discussion and Analysis (MD&A) Subsection",
    ):
        section_df = table_df.filter(pl.col("section_label") == pl.lit(section_label)).sort("row_order")
        lines.extend(
            [
                f"## {section_label}",
                "",
                "| Source/Filter | Sample Size | Observations Removed |",
                "| --- | ---: | ---: |",
            ]
        )
        for row in section_df.iter_rows(named=True):
            lines.append(
                f"| {row['display_label']} | {_format_table_i_sample_value(row)} | {_format_table_i_removed_value(row)} |"
            )
        lines.append("")
    if warnings:
        lines.append("Notes")
        lines.append("")
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _table_i_extra_artifact_paths(output_dir: Path, stage_name: str) -> tuple[Path, Path]:
    stem = Path(STAGE_ARTIFACT_FILENAMES[stage_name]).stem
    return output_dir / f"{stem}.csv", output_dir / f"{stem}.md"


def _write_table_i_stage(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    output_dir: Path,
    stage_name: str,
    table_df: pl.DataFrame,
    sample_start: dt.date,
    sample_end: dt.date,
) -> pl.LazyFrame:
    csv_path, markdown_path = _table_i_extra_artifact_paths(output_dir, stage_name)
    table_df.write_csv(csv_path)
    table_i_warnings = _table_i_stage_warnings(table_df)
    markdown_path.write_text(
        _render_table_i_sample_creation_markdown(
            table_df,
            sample_start=sample_start,
            sample_end=sample_end,
            warnings=table_i_warnings,
        ),
        encoding="utf-8",
    )
    return _write_stage(
        manifest,
        manifest_path=manifest_path,
        output_dir=output_dir,
        stage_name=stage_name,
        frame=table_df,
        extra_artifacts={"csv": csv_path, "markdown": markdown_path},
        warnings=table_i_warnings,
    )


def _quarterly_regression_diag_paths(output_dir: Path, stage_name: str) -> tuple[Path, Path]:
    stem = Path(STAGE_ARTIFACT_FILENAMES[stage_name]).stem
    return output_dir / f"{stem}_skipped_quarters.parquet", output_dir / f"{stem}_skipped_quarters.csv"


def _write_quarterly_regression_table_stage(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    output_dir: Path,
    stage_name: str,
    bundle: _QuarterlyFamaMacbethBundle,
) -> pl.LazyFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    extra_artifacts: dict[str, Path] = {}
    warnings: list[str] = []
    skipped_quarters_df = bundle.skipped_quarters_df
    if skipped_quarters_df.height > 0:
        skipped_parquet_path, skipped_csv_path = _quarterly_regression_diag_paths(output_dir, stage_name)
        skipped_quarters_df.write_parquet(skipped_parquet_path, compression=PARQUET_COMPRESSION)
        skipped_quarters_df.write_csv(skipped_csv_path)
        extra_artifacts = {
            "skipped_quarters_parquet": skipped_parquet_path,
            "skipped_quarters_csv": skipped_csv_path,
        }
        skipped_quarter_count = int(
            skipped_quarters_df.select(pl.col("quarter_start").n_unique()).item()
        )
        skipped_fit_count = skipped_quarters_df.height
        warning = (
            f"Skipped {skipped_quarter_count} rank-deficient quarter"
            f"{'' if skipped_quarter_count == 1 else 's'} across {skipped_fit_count} "
            f"quarter/signal fit{'' if skipped_fit_count == 1 else 's'}; see skipped_quarters_parquet."
        )
        warnings.append(warning)
        print(
            {
                "stage": stage_name,
                "event": "rank_deficient_quarter_skips",
                "skipped_quarter_count": skipped_quarter_count,
                "skipped_fit_count": skipped_fit_count,
                "skipped_quarters_parquet": str(skipped_parquet_path),
            }
        )
    return _write_stage(
        manifest,
        manifest_path=manifest_path,
        output_dir=output_dir,
        stage_name=stage_name,
        frame=bundle.results_df,
        empty_reason=NO_ESTIMABLE_QUARTERLY_FAMA_MACBETH_QUARTERS,
        extra_artifacts=extra_artifacts,
        warnings=warnings,
    )


def _make_event_screen_progress_logger(
    stage_name: str,
    *,
    print_ram_stats: bool,
    ram_log_interval_batches: int,
) -> Callable[[dict[str, int]], None]:
    def _logger(progress: dict[str, int]) -> None:
        batch_index = progress["batch_index"]
        total_batches = progress["total_batches"]
        if not _should_log_periodic_batch(
            batch_index=batch_index,
            total_batches=total_batches,
            interval=ram_log_interval_batches,
        ):
            return
        payload: dict[str, object] = {
            "stage": stage_name,
            "progress": f"{batch_index}/{total_batches}",
            "batch_doc_count": progress["batch_doc_count"],
            "docs_completed": progress["docs_completed"],
            "docs_total": progress["docs_total"],
        }
        if print_ram_stats:
            payload["ram"] = _ram_snapshot(f"{stage_name}_batch_{batch_index}")
        print(payload)

    return _logger


def _make_text_feature_progress_logger(
    stage_name: str,
    *,
    print_ram_stats: bool,
    ram_log_interval_batches: int,
) -> Callable[[dict[str, object]], None]:
    def _logger(progress: dict[str, object]) -> None:
        event = str(progress.get("event") or "batch")
        if event != "batch":
            payload: dict[str, object] = {
                "stage": stage_name,
                "event": event,
            }
            if print_ram_stats:
                payload["ram"] = _ram_snapshot(f"{stage_name}_{event}")
            print(payload)
            return
        batch_index = int(progress["batch_index"])
        if not _should_log_periodic_batch(
            batch_index=batch_index,
            total_batches=None,
            interval=ram_log_interval_batches,
        ):
            return
        payload: dict[str, object] = {
            "stage": stage_name,
            "event": "batch",
            "batch_index": batch_index,
            "batch_doc_count": int(progress["batch_doc_count"]),
            "docs_completed": int(progress["docs_completed"]),
        }
        if print_ram_stats:
            payload["ram"] = _ram_snapshot(f"{stage_name}_batch_{batch_index}")
        print(payload)

    return _logger


def _build_manifest(paths: RunnerPaths) -> dict[str, Any]:
    started_at_utc = _utc_timestamp()
    return {
        "runner_name": "lm2011_sample_post_refinitiv_runner",
        "generated_at_utc": started_at_utc,
        "started_at_utc": started_at_utc,
        "completed_at_utc": None,
        "elapsed_seconds": None,
        "failed_stage": None,
        "run_status": "running",
        "roots": {
            "sample_root": _absolute_path_str(paths.sample_root),
            "upstream_run_root": _absolute_path_str(paths.upstream_run_root),
            "additional_data_dir": _absolute_path_str(paths.additional_data_dir),
            "output_dir": _absolute_path_str(paths.output_dir),
            "local_work_root": _absolute_path_str(paths.local_work_root),
        },
        "config": {
            "full_10k_cleaning_contract": paths.full_10k_cleaning_contract,
            "raw_mda_cleaning_policy_id": RAW_ITEM_TEXT_CLEANING_POLICY_ID,
            "text_feature_batch_size": paths.full_10k_text_feature_batch_size,
            "full_10k_text_feature_batch_size": paths.full_10k_text_feature_batch_size,
            "mda_text_feature_batch_size": paths.mda_text_feature_batch_size,
            "recompute_event_screen_surface": paths.recompute_event_screen_surface,
            "recompute_event_panel": paths.recompute_event_panel,
            "recompute_regression_tables": paths.recompute_regression_tables,
            "event_window_doc_batch_size": paths.event_window_doc_batch_size,
            "print_ram_stats": paths.print_ram_stats,
            "ram_log_interval_batches": paths.ram_log_interval_batches,
        },
        "resolved_inputs": {
            "year_merged_dir": _absolute_path_str(paths.year_merged_dir),
            "sample_backbone_path": _absolute_path_str(paths.sample_backbone_path),
            "daily_panel_path": _absolute_path_str(paths.daily_panel_path),
            "text_features_full_10k_path": _absolute_path_str(paths.text_features_full_10k_path),
            "text_features_mda_path": _absolute_path_str(paths.text_features_mda_path),
            "ccm_base_dir": _absolute_path_str(paths.ccm_base_dir),
            "matched_clean_path": _absolute_path_str(paths.matched_clean_path),
            "items_analysis_dir": _absolute_path_str(paths.items_analysis_dir),
            "doc_ownership_path": _absolute_path_str(paths.doc_ownership_path),
            "doc_analyst_selected_path": _absolute_path_str(paths.doc_analyst_selected_path),
            "filingdates_path": _absolute_path_str(paths.filingdates_path),
            "quarterly_balance_sheet_path": _absolute_path_str(paths.quarterly_balance_sheet_path),
            "quarterly_income_statement_path": _absolute_path_str(paths.quarterly_income_statement_path),
            "quarterly_period_descriptor_path": _absolute_path_str(paths.quarterly_period_descriptor_path),
            "annual_balance_sheet_path": _absolute_path_str(paths.annual_balance_sheet_path),
            "annual_income_statement_path": _absolute_path_str(paths.annual_income_statement_path),
            "annual_period_descriptor_path": _absolute_path_str(paths.annual_period_descriptor_path),
            "annual_fiscal_market_path": _absolute_path_str(paths.annual_fiscal_market_path),
            "company_history_path": _absolute_path_str(paths.company_history_path),
            "company_description_path": _absolute_path_str(paths.company_description_path),
            "ff_daily_csv_path": _absolute_path_str(paths.ff_daily_csv_path),
            "ff_monthly_csv_path": _absolute_path_str(paths.ff_monthly_csv_path),
            "momentum_monthly_csv_path": _absolute_path_str(paths.momentum_monthly_csv_path),
            "ff48_siccodes_path": _absolute_path_str(paths.ff48_siccodes_path),
            "monthly_stock_path": _absolute_path_str(paths.monthly_stock_path),
            "ff_monthly_with_mom_path": _absolute_path_str(paths.ff_monthly_with_mom_path),
        },
        "artifacts": {},
        "row_counts": {},
        "stages": {},
        "dictionary_inputs": {},
    }


def _normalize_dictionary_inputs_manifest_for_compare(
    payload: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if payload is None:
        return None
    resources_out: list[dict[str, Any]] = []
    raw_resources = payload.get("resources")
    if isinstance(raw_resources, Sequence) and not isinstance(raw_resources, (str, bytes)):
        for resource in raw_resources:
            if not isinstance(resource, Mapping):
                continue
            resources_out.append(
                {
                    "name": resource.get("name"),
                    "role": resource.get("role"),
                    "provenance_status": resource.get("provenance_status"),
                    "word_count": resource.get("word_count"),
                }
            )
    resources_out.sort(
        key=lambda resource: (
            str(resource.get("role") or ""),
            str(resource.get("name") or ""),
        )
    )
    return {
        "master_dictionary_provenance_status": payload.get("master_dictionary_provenance_status"),
        "resources": resources_out,
    }


def _expected_text_feature_schema(
    spec: _TextFeatureReuseSpec,
    *,
    dictionary_inputs,
) -> dict[str, pl.DataType]:
    normalized_dict = normalize_lm2011_dictionary_lists(dictionary_inputs.dictionary_lists)
    signal_specs = _build_lm2011_signal_specs(
        normalized_dict=normalized_dict,
        harvard_negative_word_list=dictionary_inputs.harvard_negative_word_list,
    )
    return _feature_schema(
        include_item_id=spec.include_item_id,
        include_cleaning_policy_id=spec.include_cleaning_policy_id,
        raw_form_col="document_type_filename",
        token_count_col=spec.token_count_col,
        total_token_count_col=spec.total_token_count_col,
        signal_specs=signal_specs,
    )


def _validate_text_feature_artifact_schema(
    artifact_path: Path,
    *,
    stage_name: str,
    expected_schema: Mapping[str, pl.DataType],
) -> None:
    schema = pl.scan_parquet(artifact_path).collect_schema()
    missing = [name for name in expected_schema if name not in schema]
    if missing:
        raise ValueError(
            f"{stage_name} reusable artifact missing required columns: {missing}"
        )
    dtype_mismatches = [
        f"{name} expected {dtype!r} got {schema[name]!r}"
        for name, dtype in expected_schema.items()
        if schema[name] != dtype
    ]
    if dtype_mismatches:
        raise ValueError(
            f"{stage_name} reusable artifact has incompatible schema: {dtype_mismatches}"
        )


def _load_reuse_source_manifest(artifact_path: Path) -> dict[str, Any] | None:
    manifest_path = artifact_path.parent / MANIFEST_FILENAME
    if not manifest_path.exists():
        return None
    return _read_json_payload(manifest_path)


def _validate_text_feature_artifact_semantics(
    artifact_path: Path,
    *,
    spec: _TextFeatureReuseSpec,
    current_config: Mapping[str, Any],
    current_dictionary_inputs_manifest: Mapping[str, Any] | None,
    is_explicit_override: bool,
    source_manifest_override: Mapping[str, Any] | None = None,
) -> list[str]:
    source_manifest_path = artifact_path.parent / MANIFEST_FILENAME
    source_manifest = (
        dict(source_manifest_override)
        if source_manifest_override is not None
        else _load_reuse_source_manifest(artifact_path)
    )
    if source_manifest is None:
        if is_explicit_override:
            return [
                "No sibling lm2011_sample_run_manifest.json was found for the explicit reuse artifact; "
                "semantic validation fell back to schema-only checks."
            ]
        raise ValueError(
            f"{spec.stage_name} reusable artifact is missing sibling manifest {source_manifest_path}"
        )

    source_config = source_manifest.get("config")
    if not isinstance(source_config, Mapping):
        raise ValueError(
            f"{spec.stage_name} reusable artifact manifest missing config section: {source_manifest_path}"
        )
    current_value: object | None
    source_value: object | None
    if spec.stage_name == "text_features_full_10k":
        current_value = current_config.get("full_10k_cleaning_contract")
        source_value = source_config.get("full_10k_cleaning_contract")
        if source_value != current_value:
            raise ValueError(
                f"{spec.stage_name} reusable artifact manifest full_10k_cleaning_contract={source_value!r} "
                f"does not match current run {current_value!r}"
            )
    else:
        current_value = current_config.get("raw_mda_cleaning_policy_id")
        source_value = source_config.get("raw_mda_cleaning_policy_id")
        if source_value != current_value:
            raise ValueError(
                f"{spec.stage_name} reusable artifact manifest raw_mda_cleaning_policy_id={source_value!r} "
                f"does not match current run {current_value!r}"
            )

    normalized_source_dictionary_inputs = _normalize_dictionary_inputs_manifest_for_compare(
        source_manifest.get("dictionary_inputs")
        if isinstance(source_manifest.get("dictionary_inputs"), Mapping)
        else None
    )
    normalized_current_dictionary_inputs = _normalize_dictionary_inputs_manifest_for_compare(
        current_dictionary_inputs_manifest
    )
    if normalized_source_dictionary_inputs != normalized_current_dictionary_inputs:
        raise ValueError(
            f"{spec.stage_name} reusable artifact manifest dictionary_inputs do not match the current run"
        )
    return []


def _record_reused_stage_artifact(
    manifest: dict[str, Any],
    *,
    manifest_path: Path | None,
    stage_name: str,
    artifact_path: Path,
    source_path: Path,
    extra_artifacts: dict[str, Path] | None = None,
    warnings: Sequence[str] | None = None,
) -> pl.LazyFrame:
    row_count = int(pl.scan_parquet(artifact_path).select(pl.len()).collect().item())
    completed_at_utc = _utc_timestamp()
    manifest["artifacts"][stage_name] = _absolute_path_str(artifact_path)
    manifest["row_counts"][stage_name] = row_count
    manifest["stages"][stage_name] = {
        "status": STAGE_STATUS_REUSED_EXISTING_ARTIFACT,
        "artifact_path": _absolute_path_str(artifact_path),
        "row_count": row_count,
        "reason": None,
        "extra_artifacts": {
            name: _absolute_path_str(path)
            for name, path in (extra_artifacts or {}).items()
        },
        "source_path": _absolute_path_str(source_path),
        "warnings": list(warnings or []),
        "completed_at_utc": completed_at_utc,
    }
    print(
        {
            "stage": stage_name,
            "status": STAGE_STATUS_REUSED_EXISTING_ARTIFACT,
            "row_count": row_count,
            "artifact_path": str(artifact_path),
            "source_path": str(source_path),
        }
    )
    if manifest_path is not None:
        _checkpoint_manifest(manifest, manifest_path)
    return pl.scan_parquet(artifact_path)


def _record_stage_success(
    manifest: dict[str, Any],
    *,
    stage_name: str,
    artifact_path: Path,
    row_count: int,
    empty_reason: str | None = None,
    source_path: Path | None = None,
    extra_artifacts: dict[str, Path] | None = None,
    warnings: Sequence[str] | None = None,
) -> None:
    status = "generated_empty" if row_count == 0 else "generated"
    completed_at_utc = _utc_timestamp()
    manifest["artifacts"][stage_name] = _absolute_path_str(artifact_path)
    manifest["row_counts"][stage_name] = row_count
    manifest["stages"][stage_name] = {
        "status": status,
        "artifact_path": _absolute_path_str(artifact_path),
        "row_count": row_count,
        "reason": empty_reason if status == "generated_empty" else None,
        "extra_artifacts": {
            name: _absolute_path_str(path)
            for name, path in (extra_artifacts or {}).items()
        },
        "source_path": _absolute_path_str(source_path),
        "warnings": list(warnings or []),
        "completed_at_utc": completed_at_utc,
    }
    print({"stage": stage_name, "status": status, "row_count": row_count, "artifact_path": str(artifact_path)})


def _record_stage_skipped(
    manifest: dict[str, Any],
    *,
    stage_name: str,
    reason: str,
    detail: str,
    manifest_path: Path | None = None,
) -> None:
    artifact_path = _artifact_output_path(Path(manifest["roots"]["output_dir"]), stage_name)
    manifest["stages"][stage_name] = {
        "status": reason,
        "artifact_path": _absolute_path_str(artifact_path),
        "row_count": None,
        "reason": detail,
        "completed_at_utc": _utc_timestamp(),
    }
    print({"stage": stage_name, "status": reason, "reason": detail})
    if manifest_path is not None:
        _checkpoint_manifest(manifest, manifest_path)


def _record_stage_failed(
    manifest: dict[str, Any],
    *,
    stage_name: str,
    exc: Exception,
    manifest_path: Path | None = None,
) -> None:
    output_dir = Path(manifest["roots"]["output_dir"])
    manifest["stages"][stage_name] = {
        "status": "failed",
        "artifact_path": _absolute_path_str(_artifact_output_path(output_dir, stage_name))
        if stage_name in STAGE_ARTIFACT_FILENAMES
        else None,
        "row_count": None,
        "reason": f"{type(exc).__name__}: {exc}",
        "completed_at_utc": _utc_timestamp(),
    }
    if manifest_path is not None:
        _checkpoint_manifest(manifest, manifest_path)


def _write_stage(
    manifest: dict[str, Any],
    *,
    manifest_path: Path | None,
    output_dir: Path,
    stage_name: str,
    frame: pl.LazyFrame | pl.DataFrame,
    empty_reason: str | None = None,
    source_path: Path | None = None,
    extra_artifacts: dict[str, Path] | None = None,
    warnings: Sequence[str] | None = None,
) -> pl.LazyFrame:
    artifact_path = _artifact_output_path(output_dir, stage_name)
    written_path, row_count = _write_frame_artifact(frame, artifact_path)
    _record_stage_success(
        manifest,
        stage_name=stage_name,
        artifact_path=written_path,
        row_count=row_count,
        empty_reason=empty_reason,
        source_path=source_path,
        extra_artifacts=extra_artifacts,
        warnings=warnings,
    )
    if manifest_path is not None:
        _checkpoint_manifest(manifest, manifest_path)
    return pl.scan_parquet(written_path)


def _record_existing_stage_artifact(
    manifest: dict[str, Any],
    *,
    manifest_path: Path | None,
    stage_name: str,
    artifact_path: Path,
    empty_reason: str | None = None,
    source_path: Path | None = None,
    extra_artifacts: dict[str, Path] | None = None,
    warnings: Sequence[str] | None = None,
) -> pl.LazyFrame:
    row_count = int(pl.scan_parquet(artifact_path).select(pl.len()).collect().item())
    _record_stage_success(
        manifest,
        stage_name=stage_name,
        artifact_path=artifact_path,
        row_count=row_count,
        empty_reason=empty_reason,
        source_path=source_path,
        extra_artifacts=extra_artifacts,
        warnings=warnings,
    )
    if manifest_path is not None:
        _checkpoint_manifest(manifest, manifest_path)
    return pl.scan_parquet(artifact_path)


def _text_feature_reuse_candidate(
    paths: RunnerPaths,
    *,
    stage_name: str,
) -> tuple[Path | None, bool]:
    if stage_name == "text_features_full_10k":
        return paths.text_features_full_10k_path, paths.text_features_full_10k_path_is_explicit
    if stage_name == "text_features_mda":
        return paths.text_features_mda_path, paths.text_features_mda_path_is_explicit
    raise ValueError(f"Unsupported text-feature stage for reuse: {stage_name}")


def _text_feature_reuse_override_hint(stage_name: str) -> str:
    if stage_name == "text_features_full_10k":
        return "--text-features-full-10k-path / SEC_CCM_LM2011_TEXT_FEATURES_FULL_10K_PATH"
    if stage_name == "text_features_mda":
        return "--text-features-mda-path / SEC_CCM_LM2011_TEXT_FEATURES_MDA_PATH"
    raise ValueError(f"Unsupported text-feature stage for reuse: {stage_name}")


def _stage_recompute_enabled(paths: RunnerPaths, stage_name: str) -> bool:
    if stage_name == "event_screen_surface":
        return paths.recompute_event_screen_surface
    if stage_name == "event_panel":
        return paths.recompute_event_panel
    if stage_name in FINAL_REGRESSION_TABLE_STAGE_NAMES:
        return paths.recompute_regression_tables
    return False


def _existing_reused_stage_extra_artifacts(
    output_dir: Path,
    stage_name: str,
) -> dict[str, Path]:
    if stage_name not in QUARTERLY_REGRESSION_TABLE_STAGE_NAMES:
        return {}
    skipped_parquet_path, skipped_csv_path = _quarterly_regression_diag_paths(
        output_dir,
        stage_name,
    )
    extra_artifacts: dict[str, Path] = {}
    if skipped_parquet_path.exists():
        extra_artifacts["skipped_quarters_parquet"] = skipped_parquet_path
    if skipped_csv_path.exists():
        extra_artifacts["skipped_quarters_csv"] = skipped_csv_path
    return extra_artifacts


def _resolve_reusable_canonical_stage_artifact(
    run_cfg: LM2011PostRefinitivRunConfig,
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    stage_name: str,
) -> pl.LazyFrame | None:
    if not _stage_enabled(run_cfg, stage_name):
        return None
    if _stage_recompute_enabled(run_cfg.paths, stage_name):
        return None
    artifact_path = _artifact_output_path(run_cfg.paths.output_dir, stage_name)
    if not artifact_path.exists():
        return None
    return _record_reused_stage_artifact(
        manifest,
        manifest_path=manifest_path,
        stage_name=stage_name,
        artifact_path=artifact_path,
        source_path=artifact_path,
        extra_artifacts=_existing_reused_stage_extra_artifacts(
            run_cfg.paths.output_dir,
            stage_name,
        ),
    )


def _enabled_text_feature_blocking_stages(
    run_cfg: LM2011PostRefinitivRunConfig,
    *,
    spec: _TextFeatureReuseSpec,
) -> tuple[str, ...]:
    return tuple(
        stage_name
        for stage_name in run_cfg.enabled_stages
        if stage_name in spec.blocking_stage_names
    )


def _text_feature_stage_needs_preload(
    run_cfg: LM2011PostRefinitivRunConfig,
    *,
    spec: _TextFeatureReuseSpec,
) -> bool:
    return _stage_enabled(run_cfg, spec.stage_name) or bool(
        _enabled_text_feature_blocking_stages(run_cfg, spec=spec)
    )


def _resolve_reusable_text_feature_stage(
    run_cfg: LM2011PostRefinitivRunConfig,
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    paths: RunnerPaths,
    dictionary_inputs,
    spec: _TextFeatureReuseSpec,
    preexisting_output_manifest: Mapping[str, Any] | None = None,
) -> pl.LazyFrame | None:
    candidate_path, is_explicit_override = _text_feature_reuse_candidate(
        paths,
        stage_name=spec.stage_name,
    )
    can_rebuild = _stage_enabled(run_cfg, spec.stage_name)
    blocking_stages = _enabled_text_feature_blocking_stages(run_cfg, spec=spec)
    if candidate_path is None:
        if not can_rebuild and blocking_stages:
            blocked = ", ".join(blocking_stages)
            hint = _text_feature_reuse_override_hint(spec.stage_name)
            raise RuntimeError(
                f"LM2011 stages require {spec.stage_name}, but no compatible reusable artifact was available. "
                f"Blocked stages: {blocked}. Enable {spec.stage_name} to rebuild it or provide {hint}."
            )
        return None

    expected_schema = _expected_text_feature_schema(spec, dictionary_inputs=dictionary_inputs)
    source_manifest_override = (
        preexisting_output_manifest
        if candidate_path.parent.resolve() == paths.output_dir.resolve()
        else None
    )
    try:
        _validate_text_feature_artifact_schema(
            candidate_path,
            stage_name=spec.stage_name,
            expected_schema=expected_schema,
        )
        warnings = _validate_text_feature_artifact_semantics(
            candidate_path,
            spec=spec,
            current_config=manifest["config"],
            current_dictionary_inputs_manifest=manifest.get("dictionary_inputs"),
            is_explicit_override=is_explicit_override,
            source_manifest_override=source_manifest_override,
        )
    except Exception as exc:
        if is_explicit_override:
            raise ValueError(
                f"Explicit reusable artifact for {spec.stage_name} is incompatible: {candidate_path}. {exc}"
            ) from exc
        if can_rebuild:
            print(
                {
                    "stage": spec.stage_name,
                    "status": "ignoring_incompatible_existing_artifact",
                    "artifact_path": str(candidate_path),
                    "reason": str(exc),
                }
            )
            return None
        if blocking_stages:
            blocked = ", ".join(blocking_stages)
            hint = _text_feature_reuse_override_hint(spec.stage_name)
            raise RuntimeError(
                f"LM2011 stages require {spec.stage_name}, but the canonical reusable artifact at "
                f"{candidate_path} is incompatible. Blocked stages: {blocked}. Enable {spec.stage_name} "
                f"to rebuild it or provide {hint}. Root cause: {exc}"
            ) from exc
        return None

    canonical_artifact_path = _artifact_output_path(paths.output_dir, spec.stage_name)
    load_path = canonical_artifact_path
    if candidate_path.resolve() != canonical_artifact_path.resolve():
        _copy_with_verify(candidate_path, canonical_artifact_path, validate="quick")
    else:
        load_path = candidate_path
    return _record_reused_stage_artifact(
        manifest,
        manifest_path=manifest_path,
        stage_name=spec.stage_name,
        artifact_path=load_path,
        source_path=candidate_path,
        warnings=warnings,
    )


def _write_streaming_text_feature_stage(
    manifest: dict[str, Any],
    *,
    manifest_path: Path | None,
    output_dir: Path,
    stage_name: str,
    writer: Callable[[], int],
    warnings: Sequence[str] | None = None,
    print_ram_stats: bool = False,
) -> pl.LazyFrame:
    artifact_path = _artifact_output_path(output_dir, stage_name)
    _print_ram_snapshot(f"{stage_name}_start", enabled=print_ram_stats)
    try:
        writer()
    except Exception:
        _print_ram_snapshot(f"{stage_name}_failed", enabled=print_ram_stats)
        raise
    _print_ram_snapshot(f"{stage_name}_end", enabled=print_ram_stats)
    return _record_existing_stage_artifact(
        manifest,
        manifest_path=manifest_path,
        stage_name=stage_name,
        artifact_path=artifact_path,
        warnings=warnings,
    )


def _write_event_screen_surface_stage(
    manifest: dict[str, Any],
    *,
    manifest_path: Path | None,
    output_dir: Path,
    sample_backbone_lf: pl.LazyFrame,
    annual_accounting_panel_lf: pl.LazyFrame,
    ff_factors_daily_lf: pl.LazyFrame,
    text_features_full_10k_lf: pl.LazyFrame,
    daily_panel_path: Path,
    event_window_doc_batch_size: int,
    progress_callback: Callable[[dict[str, int]], None] | None = None,
    print_ram_stats: bool = False,
) -> pl.LazyFrame:
    artifact_path = _artifact_output_path(output_dir, "event_screen_surface")
    _print_ram_snapshot("event_screen_surface_start", enabled=print_ram_stats)
    try:
        lm2011_pipeline.write_lm2011_event_screen_surface_parquet(
            sample_backbone_lf,
            pl.scan_parquet(daily_panel_path),
            annual_accounting_panel_lf,
            ff_factors_daily_lf,
            text_features_full_10k_lf,
            output_path=artifact_path,
            event_window_doc_batch_size=event_window_doc_batch_size,
            progress_callback=progress_callback,
        )
    except Exception:
        _print_ram_snapshot("event_screen_surface_failed", enabled=print_ram_stats)
        raise
    _print_ram_snapshot("event_screen_surface_end", enabled=print_ram_stats)
    return _record_existing_stage_artifact(
        manifest,
        manifest_path=manifest_path,
        stage_name="event_screen_surface",
        artifact_path=artifact_path,
    )


def _write_sue_panel_stage(
    manifest: dict[str, Any],
    *,
    manifest_path: Path | None,
    output_dir: Path,
    event_panel_lf: pl.LazyFrame,
    quarterly_accounting_panel_lf: pl.LazyFrame,
    ibes_unadjusted_earnings_lf: pl.LazyFrame,
    daily_lf: pl.LazyFrame,
    doc_batch_size: int,
    print_ram_stats: bool = False,
) -> pl.LazyFrame:
    artifact_path = _artifact_output_path(output_dir, "sue_panel")
    _print_ram_snapshot("sue_panel_start", enabled=print_ram_stats)
    try:
        write_lm2011_sue_panel_parquet(
            event_panel_lf,
            quarterly_accounting_panel_lf,
            ibes_unadjusted_earnings_lf,
            daily_lf,
            output_path=artifact_path,
            doc_batch_size=doc_batch_size,
        )
    except Exception:
        _print_ram_snapshot("sue_panel_failed", enabled=print_ram_stats)
        raise
    _print_ram_snapshot("sue_panel_end", enabled=print_ram_stats)
    return _record_existing_stage_artifact(
        manifest,
        manifest_path=manifest_path,
        stage_name="sue_panel",
        artifact_path=artifact_path,
    )


def _missing_required_paths(*paths: Path | None) -> bool:
    return any(path is None or not path.exists() for path in paths)


def _describe_missing_paths(path_map: dict[str, Path | None]) -> str:
    missing = [name for name, path in path_map.items() if path is None or not path.exists()]
    return ", ".join(missing)


def _stage_enabled(run_cfg: LM2011PostRefinitivRunConfig, stage_name: str) -> bool:
    return stage_name in run_cfg.enabled_stages


def _stage_disabled(
    run_cfg: LM2011PostRefinitivRunConfig,
    manifest: dict[str, Any],
    manifest_path: Path,
    stage_name: str,
) -> bool:
    if _stage_enabled(run_cfg, stage_name):
        return False
    _record_stage_skipped(
        manifest,
        stage_name=stage_name,
        reason=SKIPPED_STAGE_DISABLED,
        detail="stage disabled by run configuration",
        manifest_path=manifest_path,
    )
    return True


def _skip_or_raise_stage(
    run_cfg: LM2011PostRefinitivRunConfig,
    manifest: dict[str, Any],
    manifest_path: Path,
    *,
    stage_name: str,
    reason: str,
    detail: str,
) -> None:
    if run_cfg.fail_closed_for_enabled_stages:
        raise RuntimeError(
            f"LM2011 stage {stage_name} is enabled but missing required inputs: {detail}"
        )
    _record_stage_skipped(
        manifest,
        stage_name=stage_name,
        reason=reason,
        detail=detail,
        manifest_path=manifest_path,
    )


def run_lm2011_post_refinitiv_pipeline(run_cfg: LM2011PostRefinitivRunConfig) -> int:
    paths = run_cfg.paths
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = paths.output_dir / MANIFEST_FILENAME
    preexisting_output_manifest = (
        _read_json_payload(manifest_path)
        if manifest_path.exists()
        else None
    )
    manifest = _build_manifest(paths)
    current_stage_name = "initialization"
    _checkpoint_manifest(manifest, manifest_path)
    _print_ram_snapshot("lm2011_post_refinitiv_pipeline_start", enabled=paths.print_ram_stats)

    try:
        dictionary_inputs = load_lm2011_dictionary_inputs(paths.additional_data_dir)
        dictionary_lists = dictionary_inputs.dictionary_lists
        harvard_negative_word_list = dictionary_inputs.harvard_negative_word_list
        master_dictionary_words = dictionary_inputs.master_dictionary_words
        manifest["dictionary_inputs"] = dictionary_inputs.to_manifest_dict()

        sample_backbone_lf: pl.LazyFrame | None = None
        annual_accounting_panel_lf: pl.LazyFrame | None = None
        quarterly_accounting_panel_lf: pl.LazyFrame | None = None
        ff_factors_daily_lf: pl.LazyFrame | None = None
        ff_factors_monthly_with_mom_lf: pl.LazyFrame | None = None
        text_features_full_10k_lf: pl.LazyFrame | None = None
        text_features_mda_lf: pl.LazyFrame | None = None
        event_screen_surface_lf: pl.LazyFrame | None = None
        table_i_sample_creation_lf: pl.LazyFrame | None = None
        event_panel_lf: pl.LazyFrame | None = None
        sue_panel_lf: pl.LazyFrame | None = None
        return_regression_panel_full_10k_lf: pl.LazyFrame | None = None
        return_regression_panel_mda_lf: pl.LazyFrame | None = None
        sue_regression_panel_lf: pl.LazyFrame | None = None
        year_merged_lf = (
            _prepare_lm2011_sec_input_lf(pl.scan_parquet(str(paths.year_merged_dir / YEAR_MERGED_GLOB)))
            if _parquet_glob_exists(paths.year_merged_dir, YEAR_MERGED_GLOB)
            else None
        )
        year_merged_backbone_lf = (
            _prepare_lm2011_sec_backbone_input_lf(pl.scan_parquet(str(paths.year_merged_dir / YEAR_MERGED_GLOB)))
            if _parquet_glob_exists(paths.year_merged_dir, YEAR_MERGED_GLOB)
            else None
        )
        items_analysis_lf = (
            _prepare_lm2011_sec_input_lf(pl.scan_parquet(str(paths.items_analysis_dir / ITEMS_ANALYSIS_GLOB)))
            if _parquet_glob_exists(paths.items_analysis_dir, ITEMS_ANALYSIS_GLOB)
            else None
        )

        for spec in (
            TEXT_FEATURE_REUSE_SPECS["text_features_full_10k"],
            TEXT_FEATURE_REUSE_SPECS["text_features_mda"],
        ):
            if not _text_feature_stage_needs_preload(run_cfg, spec=spec):
                continue
            current_stage_name = spec.stage_name
            resolved_lf = _resolve_reusable_text_feature_stage(
                run_cfg,
                manifest=manifest,
                manifest_path=manifest_path,
                paths=paths,
                dictionary_inputs=dictionary_inputs,
                spec=spec,
                preexisting_output_manifest=preexisting_output_manifest,
            )
            if spec.stage_name == "text_features_full_10k":
                text_features_full_10k_lf = resolved_lf
            else:
                text_features_mda_lf = resolved_lf

        current_stage_name = "sample_backbone"
        if _stage_disabled(run_cfg, manifest, manifest_path, "sample_backbone"):
            pass
        elif paths.sample_backbone_path is not None:
            sample_backbone_lf = _write_stage(
                manifest,
                manifest_path=manifest_path,
                output_dir=paths.output_dir,
                stage_name="sample_backbone",
                frame=pl.scan_parquet(paths.sample_backbone_path),
                source_path=paths.sample_backbone_path,
            )
        elif (
            year_merged_backbone_lf is not None
            and paths.matched_clean_path.exists()
            and paths.filingdates_path is not None
            and paths.filingdates_path.exists()
        ):
            sample_backbone_lf = _write_stage(
                manifest,
                manifest_path=manifest_path,
                output_dir=paths.output_dir,
                stage_name="sample_backbone",
                frame=build_lm2011_sample_backbone(
                    year_merged_backbone_lf,
                    pl.scan_parquet(paths.matched_clean_path),
                    ccm_filingdates_lf=pl.scan_parquet(paths.filingdates_path),
                ),
            )
        else:
            _skip_or_raise_stage(
                run_cfg,
                manifest,
                manifest_path,
                stage_name="sample_backbone",
                reason=SKIPPED_MISSING_SEEDED_UPSTREAM,
                detail=_describe_missing_paths(
                    {
                        "year_merged": paths.year_merged_dir if _parquet_glob_exists(paths.year_merged_dir, YEAR_MERGED_GLOB) else None,
                        "matched_clean": paths.matched_clean_path,
                        "filingdates": paths.filingdates_path,
                    }
                ),
            )

        current_stage_name = "annual_accounting_panel"
        if _stage_disabled(run_cfg, manifest, manifest_path, "annual_accounting_panel"):
            pass
        elif not _missing_required_paths(
            paths.annual_balance_sheet_path,
            paths.annual_income_statement_path,
            paths.annual_period_descriptor_path,
            paths.annual_fiscal_market_path,
        ):
            annual_balance_sheet_lf, annual_income_statement_lf, annual_period_descriptor_lf, annual_fiscal_market_lf = (
                _prepare_annual_accounting_inputs(
                    paths.annual_balance_sheet_path,
                    paths.annual_income_statement_path,
                    paths.annual_period_descriptor_path,
                    paths.annual_fiscal_market_path,
                )
            )
            annual_accounting_panel_lf = _write_stage(
                manifest,
                manifest_path=manifest_path,
                output_dir=paths.output_dir,
                stage_name="annual_accounting_panel",
                frame=build_annual_accounting_panel(
                    annual_balance_sheet_lf,
                    annual_income_statement_lf,
                    annual_period_descriptor_lf,
                    annual_fiscal_market_lf=annual_fiscal_market_lf,
                ),
            )
        else:
            _skip_or_raise_stage(
                run_cfg,
                manifest,
                manifest_path,
                stage_name="annual_accounting_panel",
                reason=SKIPPED_MISSING_OPTIONAL_INPUT,
                detail=_describe_missing_paths(
                    {
                        "annual_balance_sheet": paths.annual_balance_sheet_path,
                        "annual_income_statement": paths.annual_income_statement_path,
                        "annual_period_descriptor": paths.annual_period_descriptor_path,
                        "annual_fiscal_market": paths.annual_fiscal_market_path,
                    }
                ),
            )

        current_stage_name = "quarterly_accounting_panel"
        if _stage_disabled(run_cfg, manifest, manifest_path, "quarterly_accounting_panel"):
            pass
        elif not _missing_required_paths(
            paths.quarterly_balance_sheet_path,
            paths.quarterly_income_statement_path,
            paths.quarterly_period_descriptor_path,
        ):
            quarterly_accounting_panel_lf = _write_stage(
                manifest,
                manifest_path=manifest_path,
                output_dir=paths.output_dir,
                stage_name="quarterly_accounting_panel",
                frame=build_quarterly_accounting_panel(
                    pl.scan_parquet(paths.quarterly_balance_sheet_path),
                    pl.scan_parquet(paths.quarterly_income_statement_path),
                    pl.scan_parquet(paths.quarterly_period_descriptor_path),
                ),
            )
        else:
            _skip_or_raise_stage(
                run_cfg,
                manifest,
                manifest_path,
                stage_name="quarterly_accounting_panel",
                reason=SKIPPED_MISSING_OPTIONAL_INPUT,
                detail=_describe_missing_paths(
                    {
                        "quarterly_balance_sheet": paths.quarterly_balance_sheet_path,
                        "quarterly_income_statement": paths.quarterly_income_statement_path,
                        "quarterly_period_descriptor": paths.quarterly_period_descriptor_path,
                    }
                ),
            )

        current_stage_name = "ff_factors_daily_normalized"
        if _stage_disabled(run_cfg, manifest, manifest_path, "ff_factors_daily_normalized"):
            pass
        elif paths.ff_daily_csv_path.exists():
            ff_factors_daily_lf = _write_stage(
                manifest,
                manifest_path=manifest_path,
                output_dir=paths.output_dir,
                stage_name="ff_factors_daily_normalized",
                frame=_load_ff_factors_daily_lf(paths.ff_daily_csv_path),
            )
        else:
            _skip_or_raise_stage(
                run_cfg,
                manifest,
                manifest_path,
                stage_name="ff_factors_daily_normalized",
                reason=SKIPPED_MISSING_OPTIONAL_INPUT,
                detail="ff_daily_csv_path",
            )

        current_stage_name = "ff_factors_monthly_with_mom_normalized"
        if _stage_disabled(
            run_cfg,
            manifest,
            manifest_path,
            "ff_factors_monthly_with_mom_normalized",
        ):
            pass
        elif paths.ff_monthly_with_mom_path is not None:
            if paths.ff_monthly_with_mom_path.exists():
                monthly_factor_input_lf = _scan_ff_factors_monthly_with_mom_lf(paths.ff_monthly_with_mom_path)
                _require_unique_month_end(monthly_factor_input_lf, "monthly FF+momentum factors")
                ff_factors_monthly_with_mom_lf = _write_stage(
                    manifest,
                    manifest_path=manifest_path,
                    output_dir=paths.output_dir,
                    stage_name="ff_factors_monthly_with_mom_normalized",
                    frame=monthly_factor_input_lf,
                )
            else:
                _skip_or_raise_stage(
                    run_cfg,
                    manifest,
                    manifest_path,
                    stage_name="ff_factors_monthly_with_mom_normalized",
                    reason=SKIPPED_MISSING_OPTIONAL_INPUT,
                    detail="ff_monthly_with_mom_path",
                )
        elif paths.ff_monthly_csv_path.exists() and paths.momentum_monthly_csv_path.exists():
            ff_factors_monthly_with_mom_lf = _write_stage(
                manifest,
                manifest_path=manifest_path,
                output_dir=paths.output_dir,
                stage_name="ff_factors_monthly_with_mom_normalized",
                frame=_load_ff_factors_monthly_with_mom_lf(
                    paths.ff_monthly_csv_path,
                    paths.momentum_monthly_csv_path,
                ),
            )
        else:
            _skip_or_raise_stage(
                run_cfg,
                manifest,
                manifest_path,
                stage_name="ff_factors_monthly_with_mom_normalized",
                reason=SKIPPED_MISSING_OPTIONAL_INPUT,
                detail=_describe_missing_paths(
                    {
                        "ff_monthly_csv_path": paths.ff_monthly_csv_path,
                        "momentum_monthly_csv_path": paths.momentum_monthly_csv_path,
                    }
                ),
            )

        text_year_merged_lf = year_merged_lf
        text_items_analysis_lf = items_analysis_lf
        text_stage_warnings: list[str] = []
        if sample_backbone_lf is not None:
            if text_year_merged_lf is not None:
                text_year_merged_lf = _filter_to_sample_doc_ids_lf(text_year_merged_lf, sample_backbone_lf)
            if text_items_analysis_lf is not None:
                text_items_analysis_lf = _filter_to_sample_doc_ids_lf(text_items_analysis_lf, sample_backbone_lf)
        else:
            text_stage_warnings.append(
                "Text feature scoring was not filtered to the LM2011 sample universe because sample_backbone is unavailable."
            )

        current_stage_name = "text_features_full_10k"
        if text_features_full_10k_lf is not None:
            pass
        elif _stage_disabled(run_cfg, manifest, manifest_path, "text_features_full_10k"):
            pass
        elif text_year_merged_lf is not None:
            _gc_cleanup("text_features_full_10k_pre", enabled=paths.print_ram_stats)
            text_features_full_10k_lf = _write_streaming_text_feature_stage(
                manifest,
                manifest_path=manifest_path,
                output_dir=paths.output_dir,
                stage_name="text_features_full_10k",
                writer=lambda: write_lm2011_text_features_full_10k_parquet(
                    text_year_merged_lf,
                    output_path=_artifact_output_path(paths.output_dir, "text_features_full_10k"),
                    dictionary_lists=dictionary_lists,
                    harvard_negative_word_list=harvard_negative_word_list,
                    master_dictionary_words=master_dictionary_words,
                    cleaning_contract=paths.full_10k_cleaning_contract,
                    batch_size=paths.full_10k_text_feature_batch_size,
                    temp_root=paths.local_work_root / "text_features_full_10k",
                    progress_callback=_make_text_feature_progress_logger(
                        "text_features_full_10k",
                        print_ram_stats=paths.print_ram_stats,
                        ram_log_interval_batches=paths.ram_log_interval_batches,
                    ),
                ),
                warnings=text_stage_warnings,
                print_ram_stats=paths.print_ram_stats,
            )
            _gc_cleanup("text_features_full_10k_post", enabled=paths.print_ram_stats)
        else:
            _skip_or_raise_stage(
                run_cfg,
                manifest,
                manifest_path,
                stage_name="text_features_full_10k",
                reason=SKIPPED_MISSING_OPTIONAL_INPUT,
                detail="year_merged",
            )

        current_stage_name = "text_features_mda"
        if text_features_mda_lf is not None:
            pass
        elif _stage_disabled(run_cfg, manifest, manifest_path, "text_features_mda"):
            pass
        elif text_items_analysis_lf is not None:
            text_features_mda_lf = _write_streaming_text_feature_stage(
                manifest,
                manifest_path=manifest_path,
                output_dir=paths.output_dir,
                stage_name="text_features_mda",
                writer=lambda: write_lm2011_text_features_mda_parquet(
                    text_items_analysis_lf,
                    output_path=_artifact_output_path(paths.output_dir, "text_features_mda"),
                    dictionary_lists=dictionary_lists,
                    harvard_negative_word_list=harvard_negative_word_list,
                    master_dictionary_words=master_dictionary_words,
                    batch_size=paths.mda_text_feature_batch_size,
                    temp_root=paths.local_work_root / "text_features_mda",
                    progress_callback=_make_text_feature_progress_logger(
                        "text_features_mda",
                        print_ram_stats=paths.print_ram_stats,
                        ram_log_interval_batches=paths.ram_log_interval_batches,
                    ),
                ),
                warnings=text_stage_warnings,
                print_ram_stats=paths.print_ram_stats,
            )
            _gc_cleanup("text_features_mda_post", enabled=paths.print_ram_stats)
        else:
            _skip_or_raise_stage(
                run_cfg,
                manifest,
                manifest_path,
                stage_name="text_features_mda",
                reason=SKIPPED_MISSING_SEEDED_UPSTREAM,
                detail="items_analysis",
            )

        current_stage_name = "event_screen_surface"
        if _stage_disabled(run_cfg, manifest, manifest_path, "event_screen_surface"):
            pass
        elif (
            event_screen_surface_lf := _resolve_reusable_canonical_stage_artifact(
                run_cfg,
                manifest=manifest,
                manifest_path=manifest_path,
                stage_name="event_screen_surface",
            )
        ) is not None:
            pass
        elif (
            sample_backbone_lf is not None
            and annual_accounting_panel_lf is not None
            and ff_factors_daily_lf is not None
            and text_features_full_10k_lf is not None
            and paths.daily_panel_path.exists()
        ):
            event_screen_surface_lf = _write_event_screen_surface_stage(
                manifest,
                manifest_path=manifest_path,
                output_dir=paths.output_dir,
                sample_backbone_lf=sample_backbone_lf,
                annual_accounting_panel_lf=annual_accounting_panel_lf,
                ff_factors_daily_lf=ff_factors_daily_lf,
                text_features_full_10k_lf=text_features_full_10k_lf,
                daily_panel_path=paths.daily_panel_path,
                event_window_doc_batch_size=paths.event_window_doc_batch_size,
                progress_callback=_make_event_screen_progress_logger(
                    "event_screen_surface",
                    print_ram_stats=paths.print_ram_stats,
                    ram_log_interval_batches=paths.ram_log_interval_batches,
                ),
                print_ram_stats=paths.print_ram_stats,
            )
            _gc_cleanup("event_screen_surface_post", enabled=paths.print_ram_stats)
        else:
            _skip_or_raise_stage(
                run_cfg,
                manifest,
                manifest_path,
                stage_name="event_screen_surface",
                reason=SKIPPED_MISSING_OPTIONAL_INPUT,
                detail=_describe_missing_paths(
                    {
                        "sample_backbone": Path(manifest["artifacts"]["sample_backbone"]) if "sample_backbone" in manifest["artifacts"] else None,
                        "annual_accounting_panel": Path(manifest["artifacts"]["annual_accounting_panel"]) if "annual_accounting_panel" in manifest["artifacts"] else None,
                        "ff_factors_daily_normalized": Path(manifest["artifacts"]["ff_factors_daily_normalized"]) if "ff_factors_daily_normalized" in manifest["artifacts"] else None,
                        "text_features_full_10k": Path(manifest["artifacts"]["text_features_full_10k"]) if "text_features_full_10k" in manifest["artifacts"] else None,
                        "daily_panel_path": paths.daily_panel_path,
                    }
                ),
            )

        current_stage_name = "table_i_sample_creation"
        if _stage_disabled(run_cfg, manifest, manifest_path, "table_i_sample_creation"):
            pass
        elif (
            event_screen_surface_lf is not None
            and year_merged_backbone_lf is not None
            and paths.matched_clean_path.exists()
            and paths.filingdates_path is not None
            and paths.filingdates_path.exists()
            and annual_accounting_panel_lf is not None
            and ff_factors_daily_lf is not None
            and text_features_full_10k_lf is not None
            and paths.daily_panel_path.exists()
        ):
            table_i_df = build_lm2011_table_i_sample_creation(
                year_merged_backbone_lf,
                pl.scan_parquet(paths.matched_clean_path),
                pl.scan_parquet(paths.daily_panel_path),
                annual_accounting_panel_lf,
                ff_factors_daily_lf,
                text_features_full_10k_lf,
                ccm_filingdates_lf=pl.scan_parquet(paths.filingdates_path),
                mda_text_features_lf=text_features_mda_lf,
                event_window_doc_batch_size=paths.event_window_doc_batch_size,
                _precomputed_event_screen_surface_lf=event_screen_surface_lf,
            )
            table_i_sample_creation_lf = _write_table_i_stage(
                manifest,
                manifest_path=manifest_path,
                output_dir=paths.output_dir,
                stage_name="table_i_sample_creation",
                table_df=table_i_df,
                sample_start=dt.date(1994, 1, 1),
                sample_end=dt.date(2008, 12, 31),
            )
        else:
            _skip_or_raise_stage(
                run_cfg,
                manifest,
                manifest_path,
                stage_name="table_i_sample_creation",
                reason=SKIPPED_MISSING_OPTIONAL_INPUT,
                detail=_describe_missing_paths(
                    {
                        "year_merged": paths.year_merged_dir if _parquet_glob_exists(paths.year_merged_dir, YEAR_MERGED_GLOB) else None,
                        "matched_clean": paths.matched_clean_path,
                        "filingdates": paths.filingdates_path,
                        "event_screen_surface": Path(manifest["artifacts"]["event_screen_surface"]) if "event_screen_surface" in manifest["artifacts"] else None,
                        "annual_accounting_panel": Path(manifest["artifacts"]["annual_accounting_panel"]) if "annual_accounting_panel" in manifest["artifacts"] else None,
                        "ff_factors_daily_normalized": Path(manifest["artifacts"]["ff_factors_daily_normalized"]) if "ff_factors_daily_normalized" in manifest["artifacts"] else None,
                        "text_features_full_10k": Path(manifest["artifacts"]["text_features_full_10k"]) if "text_features_full_10k" in manifest["artifacts"] else None,
                        "daily_panel_path": paths.daily_panel_path,
                    }
                ),
            )

        current_stage_name = "table_i_sample_creation_1994_2024"
        if _stage_disabled(
            run_cfg,
            manifest,
            manifest_path,
            "table_i_sample_creation_1994_2024",
        ):
            pass
        elif (
            year_merged_backbone_lf is not None
            and paths.matched_clean_path.exists()
            and paths.filingdates_path is not None
            and paths.filingdates_path.exists()
            and annual_accounting_panel_lf is not None
            and ff_factors_daily_lf is not None
            and text_features_full_10k_lf is not None
            and paths.daily_panel_path.exists()
        ):
            table_i_1994_2024_df = build_lm2011_table_i_sample_creation(
                year_merged_backbone_lf,
                pl.scan_parquet(paths.matched_clean_path),
                pl.scan_parquet(paths.daily_panel_path),
                annual_accounting_panel_lf,
                ff_factors_daily_lf,
                text_features_full_10k_lf,
                ccm_filingdates_lf=pl.scan_parquet(paths.filingdates_path),
                mda_text_features_lf=text_features_mda_lf,
                sample_start=dt.date(1994, 1, 1),
                sample_end=dt.date(2024, 12, 31),
                event_window_doc_batch_size=paths.event_window_doc_batch_size,
                _precomputed_event_screen_surface_lf=event_screen_surface_lf,
            )
            _write_table_i_stage(
                manifest,
                manifest_path=manifest_path,
                output_dir=paths.output_dir,
                stage_name="table_i_sample_creation_1994_2024",
                table_df=table_i_1994_2024_df,
                sample_start=dt.date(1994, 1, 1),
                sample_end=dt.date(2024, 12, 31),
            )
        else:
            _skip_or_raise_stage(
                run_cfg,
                manifest,
                manifest_path,
                stage_name="table_i_sample_creation_1994_2024",
                reason=SKIPPED_MISSING_OPTIONAL_INPUT,
                detail=_describe_missing_paths(
                    {
                        "year_merged": paths.year_merged_dir if year_merged_lf is not None else None,
                        "matched_clean": paths.matched_clean_path,
                        "filingdates": paths.filingdates_path,
                        "annual_accounting_panel": Path(manifest["artifacts"]["annual_accounting_panel"]) if "annual_accounting_panel" in manifest["artifacts"] else None,
                        "ff_factors_daily_normalized": Path(manifest["artifacts"]["ff_factors_daily_normalized"]) if "ff_factors_daily_normalized" in manifest["artifacts"] else None,
                        "text_features_full_10k": Path(manifest["artifacts"]["text_features_full_10k"]) if "text_features_full_10k" in manifest["artifacts"] else None,
                        "daily_panel_path": paths.daily_panel_path,
                    }
                ),
            )

        current_stage_name = "event_panel"
        if _stage_disabled(run_cfg, manifest, manifest_path, "event_panel"):
            pass
        elif (
            event_panel_lf := _resolve_reusable_canonical_stage_artifact(
                run_cfg,
                manifest=manifest,
                manifest_path=manifest_path,
                stage_name="event_panel",
            )
        ) is not None:
            pass
        elif (
            event_screen_surface_lf is not None
            and paths.doc_ownership_path.exists()
        ):
            event_panel_lf = _write_stage(
                manifest,
                manifest_path=manifest_path,
                output_dir=paths.output_dir,
                stage_name="event_panel",
                frame=build_lm2011_event_panel(
                    sample_backbone_lf=sample_backbone_lf,
                    daily_lf=pl.scan_parquet(paths.daily_panel_path),
                    annual_accounting_panel_lf=annual_accounting_panel_lf,
                    ff_factors_daily_lf=ff_factors_daily_lf,
                    ownership_lf=pl.scan_parquet(paths.doc_ownership_path),
                    full_10k_text_features_lf=text_features_full_10k_lf,
                    event_window_doc_batch_size=paths.event_window_doc_batch_size,
                    _precomputed_event_screen_surface_lf=event_screen_surface_lf,
                ),
            )
        else:
            _skip_or_raise_stage(
                run_cfg,
                manifest,
                manifest_path,
                stage_name="event_panel",
                reason=SKIPPED_MISSING_SEEDED_UPSTREAM,
                detail=_describe_missing_paths(
                    {
                        "event_screen_surface": Path(manifest["artifacts"]["event_screen_surface"]) if "event_screen_surface" in manifest["artifacts"] else None,
                        "sample_backbone": Path(manifest["artifacts"]["sample_backbone"]) if "sample_backbone" in manifest["artifacts"] else None,
                        "annual_accounting_panel": Path(manifest["artifacts"]["annual_accounting_panel"]) if "annual_accounting_panel" in manifest["artifacts"] else None,
                        "ff_factors_daily_normalized": Path(manifest["artifacts"]["ff_factors_daily_normalized"]) if "ff_factors_daily_normalized" in manifest["artifacts"] else None,
                        "text_features_full_10k": Path(manifest["artifacts"]["text_features_full_10k"]) if "text_features_full_10k" in manifest["artifacts"] else None,
                        "daily_panel_path": paths.daily_panel_path,
                        "doc_ownership_path": paths.doc_ownership_path,
                    }
                ),
            )
        if event_screen_surface_lf is not None:
            event_screen_surface_lf = None
        _gc_cleanup("event_panel_post", enabled=paths.print_ram_stats)

        current_stage_name = "sue_panel"
        if _stage_disabled(run_cfg, manifest, manifest_path, "sue_panel"):
            pass
        elif (
            event_panel_lf is not None
            and quarterly_accounting_panel_lf is not None
            and paths.doc_analyst_selected_path.exists()
            and paths.daily_panel_path.exists()
        ):
            sue_panel_lf = _write_sue_panel_stage(
                manifest,
                manifest_path=manifest_path,
                output_dir=paths.output_dir,
                event_panel_lf=event_panel_lf,
                quarterly_accounting_panel_lf=quarterly_accounting_panel_lf,
                ibes_unadjusted_earnings_lf=_prepare_doc_analyst_sue_input_lf(paths.doc_analyst_selected_path),
                daily_lf=pl.scan_parquet(paths.daily_panel_path),
                doc_batch_size=paths.event_window_doc_batch_size,
                print_ram_stats=paths.print_ram_stats,
            )
            _gc_cleanup("sue_panel_post", enabled=paths.print_ram_stats)
        else:
            _skip_or_raise_stage(
                run_cfg,
                manifest,
                manifest_path,
                stage_name="sue_panel",
                reason=SKIPPED_MISSING_SEEDED_UPSTREAM,
                detail=_describe_missing_paths(
                    {
                        "event_panel": Path(manifest["artifacts"]["event_panel"]) if "event_panel" in manifest["artifacts"] else None,
                        "quarterly_accounting_panel": Path(manifest["artifacts"]["quarterly_accounting_panel"]) if "quarterly_accounting_panel" in manifest["artifacts"] else None,
                        "doc_analyst_selected_path": paths.doc_analyst_selected_path,
                        "daily_panel_path": paths.daily_panel_path,
                    }
                ),
            )

        can_run_regressions = not _missing_required_paths(
            paths.company_history_path,
            paths.company_description_path,
            paths.ff48_siccodes_path,
        )

        current_stage_name = "return_regression_panel_full_10k"
        if _stage_disabled(
            run_cfg,
            manifest,
            manifest_path,
            "return_regression_panel_full_10k",
        ):
            pass
        elif event_panel_lf is not None and text_features_full_10k_lf is not None and can_run_regressions:
            return_regression_panel_full_10k_lf = _write_stage(
                manifest,
                manifest_path=manifest_path,
                output_dir=paths.output_dir,
                stage_name="return_regression_panel_full_10k",
                frame=build_lm2011_return_regression_panel(
                    event_panel_lf,
                    text_features_full_10k_lf,
                    pl.scan_parquet(paths.company_history_path),
                    pl.scan_parquet(paths.company_description_path),
                    ff48_siccodes_path=paths.ff48_siccodes_path,
                    text_scope="full_10k",
                ),
            )
        else:
            _skip_or_raise_stage(
                run_cfg,
                manifest,
                manifest_path,
                stage_name="return_regression_panel_full_10k",
                reason=SKIPPED_MISSING_OPTIONAL_INPUT,
                detail=_describe_missing_paths(
                    {
                        "event_panel": Path(manifest["artifacts"]["event_panel"]) if "event_panel" in manifest["artifacts"] else None,
                        "text_features_full_10k": Path(manifest["artifacts"]["text_features_full_10k"]) if "text_features_full_10k" in manifest["artifacts"] else None,
                        "company_history": paths.company_history_path,
                        "company_description": paths.company_description_path,
                        "ff48_siccodes": paths.ff48_siccodes_path,
                    }
                ),
            )

        current_stage_name = "return_regression_panel_mda"
        if _stage_disabled(run_cfg, manifest, manifest_path, "return_regression_panel_mda"):
            pass
        elif event_panel_lf is not None and text_features_mda_lf is not None and can_run_regressions:
            return_regression_panel_mda_lf = _write_stage(
                manifest,
                manifest_path=manifest_path,
                output_dir=paths.output_dir,
                stage_name="return_regression_panel_mda",
                frame=build_lm2011_return_regression_panel(
                    event_panel_lf,
                    text_features_mda_lf,
                    pl.scan_parquet(paths.company_history_path),
                    pl.scan_parquet(paths.company_description_path),
                    ff48_siccodes_path=paths.ff48_siccodes_path,
                    text_scope="mda_item_7",
                ),
            )
        else:
            _skip_or_raise_stage(
                run_cfg,
                manifest,
                manifest_path,
                stage_name="return_regression_panel_mda",
                reason=SKIPPED_MISSING_OPTIONAL_INPUT,
                detail=_describe_missing_paths(
                    {
                        "event_panel": Path(manifest["artifacts"]["event_panel"]) if "event_panel" in manifest["artifacts"] else None,
                        "text_features_mda": Path(manifest["artifacts"]["text_features_mda"]) if "text_features_mda" in manifest["artifacts"] else None,
                        "company_history": paths.company_history_path,
                        "company_description": paths.company_description_path,
                        "ff48_siccodes": paths.ff48_siccodes_path,
                    }
                ),
            )

        current_stage_name = "sue_regression_panel"
        if _stage_disabled(run_cfg, manifest, manifest_path, "sue_regression_panel"):
            pass
        elif sue_panel_lf is not None and text_features_full_10k_lf is not None and can_run_regressions:
            sue_regression_panel_lf = _write_stage(
                manifest,
                manifest_path=manifest_path,
                output_dir=paths.output_dir,
                stage_name="sue_regression_panel",
                frame=build_lm2011_sue_regression_panel(
                    sue_panel_lf,
                    text_features_full_10k_lf,
                    pl.scan_parquet(paths.company_history_path),
                    pl.scan_parquet(paths.company_description_path),
                    ff48_siccodes_path=paths.ff48_siccodes_path,
                ),
            )
        else:
            _skip_or_raise_stage(
                run_cfg,
                manifest,
                manifest_path,
                stage_name="sue_regression_panel",
                reason=SKIPPED_MISSING_OPTIONAL_INPUT,
                detail=_describe_missing_paths(
                    {
                        "sue_panel": Path(manifest["artifacts"]["sue_panel"]) if "sue_panel" in manifest["artifacts"] else None,
                        "text_features_full_10k": Path(manifest["artifacts"]["text_features_full_10k"]) if "text_features_full_10k" in manifest["artifacts"] else None,
                        "company_history": paths.company_history_path,
                        "company_description": paths.company_description_path,
                        "ff48_siccodes": paths.ff48_siccodes_path,
                    }
                ),
            )

        table_stage_specs = (
            (
                "table_iv_results",
                lambda: _build_lm2011_table_iv_results_bundle(
                    event_panel_lf,
                    text_features_full_10k_lf,
                    pl.scan_parquet(paths.company_history_path),
                    pl.scan_parquet(paths.company_description_path),
                    ff48_siccodes_path=paths.ff48_siccodes_path,
                ),
                event_panel_lf is not None and text_features_full_10k_lf is not None and can_run_regressions,
            ),
            (
                "table_iv_results_no_ownership",
                lambda: _build_lm2011_table_iv_results_no_ownership_bundle(
                    event_panel_lf,
                    text_features_full_10k_lf,
                    pl.scan_parquet(paths.company_history_path),
                    pl.scan_parquet(paths.company_description_path),
                    ff48_siccodes_path=paths.ff48_siccodes_path,
                ),
                event_panel_lf is not None and text_features_full_10k_lf is not None and can_run_regressions,
            ),
            (
                "table_v_results",
                lambda: _build_lm2011_table_v_results_bundle(
                    event_panel_lf,
                    text_features_mda_lf,
                    pl.scan_parquet(paths.company_history_path),
                    pl.scan_parquet(paths.company_description_path),
                    ff48_siccodes_path=paths.ff48_siccodes_path,
                ),
                event_panel_lf is not None and text_features_mda_lf is not None and can_run_regressions,
            ),
            (
                "table_v_results_no_ownership",
                lambda: _build_lm2011_table_v_results_no_ownership_bundle(
                    event_panel_lf,
                    text_features_mda_lf,
                    pl.scan_parquet(paths.company_history_path),
                    pl.scan_parquet(paths.company_description_path),
                    ff48_siccodes_path=paths.ff48_siccodes_path,
                ),
                event_panel_lf is not None and text_features_mda_lf is not None and can_run_regressions,
            ),
            (
                "table_vi_results",
                lambda: _build_lm2011_table_vi_results_bundle(
                    event_panel_lf,
                    text_features_full_10k_lf,
                    pl.scan_parquet(paths.company_history_path),
                    pl.scan_parquet(paths.company_description_path),
                    ff48_siccodes_path=paths.ff48_siccodes_path,
                ),
                event_panel_lf is not None and text_features_full_10k_lf is not None and can_run_regressions,
            ),
            (
                "table_vi_results_no_ownership",
                lambda: _build_lm2011_table_vi_results_no_ownership_bundle(
                    event_panel_lf,
                    text_features_full_10k_lf,
                    pl.scan_parquet(paths.company_history_path),
                    pl.scan_parquet(paths.company_description_path),
                    ff48_siccodes_path=paths.ff48_siccodes_path,
                ),
                event_panel_lf is not None and text_features_full_10k_lf is not None and can_run_regressions,
            ),
            (
                "table_viii_results",
                lambda: _build_lm2011_table_viii_results_bundle(
                    sue_panel_lf,
                    text_features_full_10k_lf,
                    pl.scan_parquet(paths.company_history_path),
                    pl.scan_parquet(paths.company_description_path),
                    ff48_siccodes_path=paths.ff48_siccodes_path,
                ),
                sue_panel_lf is not None and text_features_full_10k_lf is not None and can_run_regressions,
            ),
            (
                "table_viii_results_no_ownership",
                lambda: _build_lm2011_table_viii_results_no_ownership_bundle(
                    sue_panel_lf,
                    text_features_full_10k_lf,
                    pl.scan_parquet(paths.company_history_path),
                    pl.scan_parquet(paths.company_description_path),
                    ff48_siccodes_path=paths.ff48_siccodes_path,
                ),
                sue_panel_lf is not None and text_features_full_10k_lf is not None and can_run_regressions,
            ),
            (
                "table_ia_i_results",
                lambda: _build_lm2011_table_ia_i_results_bundle(
                    event_panel_lf,
                    text_features_full_10k_lf,
                    pl.scan_parquet(paths.company_history_path),
                    pl.scan_parquet(paths.company_description_path),
                    ff48_siccodes_path=paths.ff48_siccodes_path,
                ),
                event_panel_lf is not None and text_features_full_10k_lf is not None and can_run_regressions,
            ),
            (
                "table_ia_i_results_no_ownership",
                lambda: _build_lm2011_table_ia_i_results_no_ownership_bundle(
                    event_panel_lf,
                    text_features_full_10k_lf,
                    pl.scan_parquet(paths.company_history_path),
                    pl.scan_parquet(paths.company_description_path),
                    ff48_siccodes_path=paths.ff48_siccodes_path,
                ),
                event_panel_lf is not None and text_features_full_10k_lf is not None and can_run_regressions,
            ),
        )
        for stage_name, builder, should_run in table_stage_specs:
            current_stage_name = stage_name
            if _stage_disabled(run_cfg, manifest, manifest_path, stage_name):
                continue
            if (
                _resolve_reusable_canonical_stage_artifact(
                    run_cfg,
                    manifest=manifest,
                    manifest_path=manifest_path,
                    stage_name=stage_name,
                )
                is not None
            ):
                continue
            if should_run:
                _write_quarterly_regression_table_stage(
                    manifest,
                    manifest_path=manifest_path,
                    output_dir=paths.output_dir,
                    stage_name=stage_name,
                    bundle=builder(),
                )
            else:
                _skip_or_raise_stage(
                    run_cfg,
                    manifest,
                    manifest_path,
                    stage_name=stage_name,
                    reason=SKIPPED_MISSING_OPTIONAL_INPUT,
                    detail="missing upstream regression inputs",
                )

        current_stage_name = "trading_strategy_monthly_returns"
        if _stage_disabled(
            run_cfg,
            manifest,
            manifest_path,
            "trading_strategy_monthly_returns",
        ):
            pass
        elif (
            paths.monthly_stock_path is not None
            and paths.monthly_stock_path.exists()
            and event_panel_lf is not None
            and year_merged_lf is not None
        ):
            _write_stage(
                manifest,
                manifest_path=manifest_path,
                output_dir=paths.output_dir,
                stage_name="trading_strategy_monthly_returns",
                frame=build_lm2011_trading_strategy_monthly_returns(
                    event_panel_lf,
                    year_merged_lf,
                    pl.scan_parquet(paths.monthly_stock_path),
                    lm_dictionary_lists=dictionary_lists,
                    harvard_negative_word_list=harvard_negative_word_list,
                    master_dictionary_words=master_dictionary_words,
                    cleaning_contract=paths.full_10k_cleaning_contract,
                ),
            )
        else:
            _skip_or_raise_stage(
                run_cfg,
                manifest,
                manifest_path,
                stage_name="trading_strategy_monthly_returns",
                reason=SKIPPED_MISSING_OPTIONAL_INPUT,
                detail="monthly_stock_path",
            )

        current_stage_name = "table_ia_ii_results"
        if _stage_disabled(run_cfg, manifest, manifest_path, "table_ia_ii_results"):
            pass
        elif (
            _resolve_reusable_canonical_stage_artifact(
                run_cfg,
                manifest=manifest,
                manifest_path=manifest_path,
                stage_name="table_ia_ii_results",
            )
            is not None
        ):
            pass
        elif (
            paths.monthly_stock_path is not None
            and paths.monthly_stock_path.exists()
            and ff_factors_monthly_with_mom_lf is not None
            and event_panel_lf is not None
            and year_merged_lf is not None
        ):
            _write_stage(
                manifest,
                manifest_path=manifest_path,
                output_dir=paths.output_dir,
                stage_name="table_ia_ii_results",
                frame=build_lm2011_table_ia_ii_results(
                    event_panel_lf,
                    year_merged_lf,
                    pl.scan_parquet(paths.monthly_stock_path),
                    ff_factors_monthly_with_mom_lf,
                    lm_dictionary_lists=dictionary_lists,
                    harvard_negative_word_list=harvard_negative_word_list,
                    master_dictionary_words=master_dictionary_words,
                ),
                empty_reason=EMPTY_TABLE_REASON,
            )
        else:
            _skip_or_raise_stage(
                run_cfg,
                manifest,
                manifest_path,
                stage_name="table_ia_ii_results",
                reason=SKIPPED_MISSING_OPTIONAL_INPUT,
                detail=_describe_missing_paths(
                    {
                        "monthly_stock_path": paths.monthly_stock_path,
                        "ff_factors_monthly_with_mom_normalized": Path(
                            manifest["artifacts"]["ff_factors_monthly_with_mom_normalized"]
                        )
                        if "ff_factors_monthly_with_mom_normalized" in manifest["artifacts"]
                        else None,
                        "event_panel": Path(manifest["artifacts"]["event_panel"]) if "event_panel" in manifest["artifacts"] else None,
                        "year_merged": paths.year_merged_dir if year_merged_lf is not None else None,
                    }
                ),
            )

        completed_at_utc = _utc_timestamp()
        manifest["run_status"] = "completed"
        manifest["completed_at_utc"] = completed_at_utc
        manifest["elapsed_seconds"] = _elapsed_seconds(str(manifest["started_at_utc"]), completed_at_utc)
        manifest["failed_stage"] = None
        _checkpoint_manifest(manifest, manifest_path)
        _print_ram_snapshot("lm2011_post_refinitiv_pipeline_end", enabled=paths.print_ram_stats)
        return 0
    except Exception as exc:
        _print_ram_snapshot(
            f"lm2011_post_refinitiv_pipeline_failed_{current_stage_name}",
            enabled=paths.print_ram_stats,
        )
        _record_stage_failed(
            manifest,
            stage_name=current_stage_name,
            exc=exc,
        )
        completed_at_utc = _utc_timestamp()
        manifest["run_status"] = "failed"
        manifest["completed_at_utc"] = completed_at_utc
        manifest["elapsed_seconds"] = _elapsed_seconds(str(manifest["started_at_utc"]), completed_at_utc)
        manifest["failed_stage"] = current_stage_name
        manifest["error"] = {
            "stage": current_stage_name,
            "type": type(exc).__name__,
            "message": str(exc),
        }
        _checkpoint_manifest(manifest, manifest_path)
        raise


def run_lm2011_extension_pipeline(run_cfg: LM2011ExtensionRunConfig) -> int:
    output_dir = Path(run_cfg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_finbert_inputs = _resolve_finbert_extension_inputs(run_cfg)
    manifest = _build_extension_manifest(
        run_cfg,
        resolved_finbert_inputs=resolved_finbert_inputs,
    )
    manifest_path = output_dir / EXTENSION_MANIFEST_FILENAME
    current_stage_name = "initialization"
    _checkpoint_manifest(manifest, manifest_path)

    try:
        current_stage_name = "configuration_validation"
        _validate_extension_comparison_mode(
            run_cfg,
            cleaned_item_scopes_dir=resolved_finbert_inputs["cleaned_item_scopes_dir"],
        )

        dictionary_inputs = load_lm2011_dictionary_inputs(run_cfg.additional_data_dir)
        manifest["dictionary_inputs"] = dictionary_inputs.to_manifest_dict()

        event_panel_lf = pl.scan_parquet(run_cfg.event_panel_path)
        event_doc_ids_lf = event_panel_lf.select(
            pl.col("doc_id").cast(pl.Utf8, strict=False)
        ).unique()

        manifest["config"]["effective_dictionary_source_mode"] = run_cfg.dictionary_source_mode
        _checkpoint_manifest(manifest, manifest_path)

        current_stage_name = "extension_dictionary_surface"
        dictionary_surface_lf = _write_extension_stage(
            manifest,
            manifest_path=manifest_path,
            output_dir=output_dir,
            stage_name="extension_dictionary_surface",
            frame=_build_extension_dictionary_surface_lf(
                run_cfg,
                cleaned_item_scopes_dir=resolved_finbert_inputs["cleaned_item_scopes_dir"],
                event_doc_ids_lf=event_doc_ids_lf,
                dictionary_inputs=dictionary_inputs,
            ),
        )

        current_stage_name = "extension_finbert_surface"
        finbert_surface_lf = _write_extension_stage(
            manifest,
            manifest_path=manifest_path,
            output_dir=output_dir,
            stage_name="extension_finbert_surface",
            frame=_build_extension_finbert_surface_lf(
                pl.scan_parquet(resolved_finbert_inputs["item_features_long_path"]),
                event_doc_ids_lf,
                text_scopes=run_cfg.text_scopes,
            ),
        )

        current_stage_name = "extension_control_ladder"
        _write_extension_stage(
            manifest,
            manifest_path=manifest_path,
            output_dir=output_dir,
            stage_name="extension_control_ladder",
            frame=build_lm2011_extension_control_ladder(),
        )

        current_stage_name = "extension_specification_grid"
        _write_extension_stage(
            manifest,
            manifest_path=manifest_path,
            output_dir=output_dir,
            stage_name="extension_specification_grid",
            frame=build_lm2011_extension_specification_grid(),
        )

        current_stage_name = "extension_analysis_panel"
        extension_panel_lf = _write_extension_stage(
            manifest,
            manifest_path=manifest_path,
            output_dir=output_dir,
            stage_name="extension_analysis_panel",
            frame=build_lm2011_extension_analysis_panel(
                event_panel_lf,
                dictionary_surface_lf,
                finbert_surface_lf,
                pl.scan_parquet(run_cfg.company_history_path),
                pl.scan_parquet(run_cfg.company_description_path),
                ff48_siccodes_path=run_cfg.ff48_siccodes_path,
            ),
        )

        current_stage_name = "extension_sample_loss"
        _write_extension_stage(
            manifest,
            manifest_path=manifest_path,
            output_dir=output_dir,
            stage_name="extension_sample_loss",
            frame=build_lm2011_extension_sample_loss_table(extension_panel_lf),
        )

        current_stage_name = "extension_results"
        _write_extension_stage(
            manifest,
            manifest_path=manifest_path,
            output_dir=output_dir,
            stage_name="extension_results",
            frame=run_lm2011_extension_estimation_scaffold(
                extension_panel_lf,
                run_id=run_cfg.run_id,
                text_scopes=_normalized_extension_text_scopes(run_cfg.text_scopes),
            ),
        )

        completed_at_utc = _utc_timestamp()
        manifest["run_status"] = "completed"
        manifest["completed_at_utc"] = completed_at_utc
        manifest["elapsed_seconds"] = _elapsed_seconds(
            str(manifest["started_at_utc"]),
            completed_at_utc,
        )
        manifest["failed_stage"] = None
        _checkpoint_manifest(manifest, manifest_path)
        return 0
    except Exception as exc:
        _record_extension_stage_failed(
            manifest,
            output_dir=output_dir,
            stage_name=current_stage_name,
            exc=exc,
            manifest_path=manifest_path,
        )
        completed_at_utc = _utc_timestamp()
        manifest["run_status"] = "failed"
        manifest["completed_at_utc"] = completed_at_utc
        manifest["elapsed_seconds"] = _elapsed_seconds(
            str(manifest["started_at_utc"]),
            completed_at_utc,
        )
        manifest["failed_stage"] = current_stage_name
        manifest["error"] = {
            "stage": current_stage_name,
            "type": type(exc).__name__,
            "message": str(exc),
        }
        _checkpoint_manifest(manifest, manifest_path)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run_lm2011_post_refinitiv_pipeline(
        build_lm2011_post_refinitiv_run_config(args)
    )


if __name__ == "__main__":
    raise SystemExit(main())

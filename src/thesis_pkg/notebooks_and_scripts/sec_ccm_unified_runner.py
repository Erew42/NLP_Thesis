from __future__ import annotations

"""Runnable script translation of ``sec_ccm_unified_runner.ipynb``.

The goal is to preserve the notebook's execution order and behavior while making
it executable as a normal Python script.
"""

import os
import sys
import json
import datetime as dt
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT_ENV_VAR = "NLP_THESIS_REPO_ROOT"
DATA_PROFILE_ENV_VAR = "SEC_CCM_DATA_PROFILE"

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

    return cwd


def _resolve_colab_drive_root() -> Path:
    for candidate in (
        Path("/content/drive/MyDrive"),
        Path("/content/drive/My Drive"),
        Path("/content/drive"),
    ):
        if candidate.exists():
            return candidate
    return Path("/content/drive")


def _first_existing_path(*candidates: Path) -> Path:
    if not candidates:
        raise ValueError("At least one candidate path is required")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None:
        return default
    stripped = value.strip()
    return stripped or default


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if value is None:
        return default
    stripped = value.strip()
    if not stripped:
        return default
    return Path(stripped).expanduser()


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value for {name}: {value!r}")


def _env_optional_bool(name: str, default: bool | None) -> bool | None:
    value = os.environ.get(name)
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


def _env_str_list(name: str, default: list[str]) -> list[str]:
    value = os.environ.get(name)
    if value is None:
        return default
    stripped = value.strip()
    if not stripped:
        return default
    parsed = json.loads(stripped) if stripped.startswith("[") else stripped.split(",")
    return [str(item).strip() for item in parsed if str(item).strip()]


def _env_int_list(name: str, default: list[int]) -> list[int]:
    value = os.environ.get(name)
    if value is None:
        return default
    stripped = value.strip()
    if not stripped:
        return default
    parsed = json.loads(stripped) if stripped.startswith("[") else stripped.split(",")
    return [int(item) for item in parsed]


def _print_rows_table(rows: list[dict[str, object]], *, sort_by: list[str] | None = None, empty_message: str) -> None:
    if not rows:
        print(empty_message)
        return
    frame = pl.DataFrame(rows)
    if sort_by:
        frame = frame.sort(sort_by)
    # Avoid Polars' Unicode box-drawing repr, which can fail on legacy Windows consoles.
    print(frame.write_csv(None, separator="\t"))


def _resolve_ff48_siccodes_path(work_root: Path) -> Path:
    return _env_path(
        "SEC_CCM_FF48_SICCODES_PATH",
        _first_existing_path(
            ROOT / "full_data_run" / "LM2011_additional_data" / "FF_Siccodes_48_Industries.txt",
            ROOT / "LM2011_additional_data" / "FF_Siccodes_48_Industries.txt",
            work_root / "LM2011_additional_data" / "FF_Siccodes_48_Industries.txt",
            work_root / "Data" / "LM2011_additional_data" / "FF_Siccodes_48_Industries.txt",
        ),
    )


def _resolve_ccm_parquet_artifact(base_dir: Path, parquet_name: str) -> Path:
    candidates = [base_dir / parquet_name]
    candidates.extend(
        sorted(
            (child / parquet_name for child in base_dir.glob("documents-export*") if child.is_dir()),
            key=lambda path: str(path),
        )
    )
    return _first_existing_path(*candidates)


ROOT = _resolve_repo_root()
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import polars as pl

from thesis_pkg.core.sec.suspicious_boundary_diagnostics import (
    DiagnosticsConfig,
    parse_focus_items,
    run_boundary_diagnostics,
)
from thesis_pkg.filing_text import (
    build_light_metadata_dataset,
    compute_no_item_diagnostics,
    merge_yearly_batches,
    process_year_dir_extract_items_gated,
    process_zip_year,
    process_zip_year_raw_text,
    summarize_year_parquets,
)
from thesis_pkg.pipeline import (
    CCM_DAILY_BRIDGE_SURFACE_OPTIONAL_COLUMNS,
    CCM_DAILY_BRIDGE_SURFACE_REQUIRED_COLUMNS,
    CCM_DAILY_MARKET_CORE_COLUMNS,
    CCM_DAILY_PHASE_B_SURFACE_COLUMNS,
    SEC_CCM_PHASE_B_DAILY_FEATURE_COLUMNS,
    SecCcmJoinSpecV2,
    attach_lm2011_industry_classifications,
    build_lm2011_sample_backbone,
    build_quarterly_accounting_panel,
    build_or_reuse_ccm_daily_stage,
    is_lseg_available,
    make_sec_ccm_join_spec_preset,
    run_refinitiv_lm2011_doc_analyst_anchor_pipeline,
    run_refinitiv_lm2011_doc_analyst_select_pipeline,
    run_refinitiv_lm2011_doc_ownership_exact_api_pipeline,
    run_refinitiv_lm2011_doc_ownership_exact_handoff_pipeline,
    run_refinitiv_lm2011_doc_ownership_fallback_api_pipeline,
    run_refinitiv_lm2011_doc_ownership_fallback_handoff_pipeline,
    run_refinitiv_lm2011_doc_ownership_finalize_pipeline,
    run_refinitiv_step1_analyst_actuals_api_pipeline,
    run_refinitiv_step1_analyst_estimates_monthly_api_pipeline,
    run_refinitiv_step1_analyst_normalize_pipeline,
    run_refinitiv_step1_analyst_request_groups_pipeline,
    run_refinitiv_step1_instrument_authority_pipeline,
    run_refinitiv_step1_lookup_api_pipeline,
    run_refinitiv_step1_ownership_authority_pipeline,
    run_refinitiv_step1_ownership_universe_api_pipeline,
    run_refinitiv_step1_ownership_universe_handoff_pipeline,
    run_refinitiv_step1_ownership_universe_results_pipeline,
    run_refinitiv_step1_resolution_pipeline,
    run_refinitiv_step1_bridge_pipeline,
    run_sec_ccm_premerge_pipeline,
)

LSEG_API_READY = is_lseg_available()


@dataclass(frozen=True)
class _ArtifactStage:
    name: str
    paths: dict[str, Path]


def _discover_sec_years(zip_dir: Path, merged_dir: Path) -> list[int]:
    years: set[int] = set()
    if zip_dir.exists():
        for path in zip_dir.glob("*.zip"):
            if path.stem.isdigit() and len(path.stem) == 4:
                years.add(int(path.stem))
    if merged_dir.exists():
        for path in merged_dir.glob("*.parquet"):
            if path.stem.isdigit() and len(path.stem) == 4:
                years.add(int(path.stem))
    return sorted(years)


def _has_yearly_outputs(year_dir: Path) -> bool:
    return year_dir.exists() and any(
        path.stem.isdigit() and len(path.stem) == 4 for path in year_dir.glob("*.parquet")
    )


def _first_existing(schema: pl.Schema, candidates: tuple[str, ...], label: str) -> str:
    for candidate in candidates:
        if candidate in schema:
            return candidate
    raise ValueError(f"{label} missing candidates: {list(candidates)}")


def _row_count(path: Path) -> int | None:
    if not path.exists() or path.suffix.lower() != ".parquet":
        return None
    return int(pl.scan_parquet(path).select(pl.len()).collect().item())


def _ownership_rows_with_data(path: Path) -> int | None:
    if not path.exists() or path.suffix.lower() != ".parquet":
        return None
    return int(
        pl.scan_parquet(path)
        .filter(pl.col("ownership_rows_returned") > 0)
        .select(pl.len())
        .collect()
        .item()
    )


def _nonnull_count(path: Path, column: str) -> int | None:
    if not path.exists() or path.suffix.lower() != ".parquet":
        return None
    return int(
        pl.scan_parquet(path)
        .filter(pl.col(column).is_not_null())
        .select(pl.len())
        .collect()
        .item()
    )


def _bool_true_count(path: Path, column: str) -> int | None:
    if not path.exists() or path.suffix.lower() != ".parquet":
        return None
    return int(
        pl.scan_parquet(path)
        .filter(pl.col(column).cast(pl.Boolean, strict=False).fill_null(False))
        .select(pl.len())
        .collect()
        .item()
    )


def _value_counts(path: Path, column: str) -> dict[str, int] | None:
    if not path.exists() or path.suffix.lower() != ".parquet":
        return None
    return {
        str(row[column]): int(row["len"])
        for row in pl.scan_parquet(path)
        .group_by(column)
        .len()
        .collect()
        .to_dicts()
    }


def main() -> None:
    # ## Runtime setup
    if IN_COLAB:
        from google.colab import drive

        drive.mount("/content/drive", force_remount=False)

    print({"IN_COLAB": IN_COLAB, "ROOT": str(ROOT), "SRC_EXISTS": SRC.exists()})

    # ## Config
    default_data_profile = "DRIVE_FULL" if IN_COLAB else "LOCAL_SAMPLE"
    DATA_PROFILE = _env_str(DATA_PROFILE_ENV_VAR, default_data_profile)

    SAMPLE_ROOT = _env_path(
        "SEC_CCM_SAMPLE_ROOT",
        ROOT / "full_data_run" / "sample_5pct_seed42",
    )
    default_work_root = (
        _resolve_colab_drive_root() / "Data_LM"
        if IN_COLAB
        else Path("C:/Users/erik9/Documents/SEC_Data")
    )
    drive_work_root = _env_path("SEC_CCM_WORK_ROOT", default_work_root)
    legacy_sec_root = drive_work_root
    alt_sec_root = drive_work_root / "Data" / "Sample_Filings"
    legacy_ccm_root = drive_work_root / "CRSP_Compustat_data"
    alt_ccm_root = drive_work_root / "Data" / "CRSP_Compustat_data"

    PROFILE_CONFIG = {
        "LOCAL_SAMPLE": {
            "WORK_ROOT": SAMPLE_ROOT,
            "SEC_ZIP_DIR": SAMPLE_ROOT / "10_X_reports",
            "SEC_BATCH_ROOT": SAMPLE_ROOT / "sec_batches",
            "SEC_YEAR_MERGED_DIR": SAMPLE_ROOT / "year_merged",
            "SEC_LIGHT_METADATA_PATH": SAMPLE_ROOT
            / "derived_data"
            / "filings_metadata_LIGHT.sample_5pct_seed42.parquet",
            "CCM_BASE_DIR": SAMPLE_ROOT / "ccm_parquet_data",
            "CCM_DERIVED_DIR": SAMPLE_ROOT / "derived_data",
            "CCM_REUSE_DAILY_PATH": SAMPLE_ROOT
            / "derived_data"
            / "final_flagged_data_compdesc_added.sample_5pct_seed42.parquet",
            "CANONICAL_LINK_NAME": "canonical_link_table_after_startdate_change.sample_5pct_seed42.parquet",
            "CCM_DAILY_NAME": "final_flagged_data_compdesc_added.sample_5pct_seed42.parquet",
            "RUN_ROOT": SAMPLE_ROOT / "results" / "sec_ccm_unified_runner" / "local_sample",
            "RUN_CCM_MODE": "REBUILD",
            "RUN_SEC_PARSE": None,
            "RUN_SEC_YEARLY_MERGE": None,
        },
        "DRIVE_FULL": {
            "WORK_ROOT": drive_work_root,
            "SEC_ZIP_DIR": _first_existing_path(
                legacy_sec_root / "Zip_Files",
                alt_sec_root,
            ),
            "SEC_BATCH_ROOT": _first_existing_path(
                legacy_sec_root / "parquet_data",
                alt_sec_root / "parquet_batches",
            ),
            "SEC_YEAR_MERGED_DIR": _first_existing_path(
                legacy_sec_root / "parquet_data" / "_year_merged",
                alt_sec_root / "parquet_batches" / "_year_merged",
            ),
            "SEC_LIGHT_METADATA_PATH": _first_existing_path(
                legacy_sec_root / "parquet_data" / "filings_metadata_1993_2024_LIGHT.parquet",
                alt_sec_root / "filings_metadata_LIGHT.parquet",
            ),
            "CCM_BASE_DIR": _first_existing_path(
                legacy_ccm_root / "parquet_data",
                alt_ccm_root / "parquet_data",
            ),
            "CCM_DERIVED_DIR": _first_existing_path(
                legacy_ccm_root / "derived_data",
                alt_ccm_root / "derived_data",
            ),
            "CCM_REUSE_DAILY_PATH": _first_existing_path(
                legacy_ccm_root / "derived_data" / "final_flagged_data_compdesc_added.parquet",
                alt_ccm_root / "derived_data" / "final_flagged_data_compdesc_added.parquet",
            ),
            "CANONICAL_LINK_NAME": "canonical_link_table_after_startdate_change.parquet",
            "CCM_DAILY_NAME": "final_flagged_data_compdesc_added.parquet",
            "RUN_ROOT": drive_work_root / "results" / "sec_ccm_unified_runner",
            "RUN_CCM_MODE": "REUSE",
            "RUN_SEC_PARSE": False,
            "RUN_SEC_YEARLY_MERGE": False,
        },
    }

    if DATA_PROFILE not in PROFILE_CONFIG:
        raise ValueError(
            f"Unknown DATA_PROFILE={DATA_PROFILE!r}. Expected one of {list(PROFILE_CONFIG)}"
        )

    profile = PROFILE_CONFIG[DATA_PROFILE]
    WORK_ROOT = _env_path("SEC_CCM_WORK_ROOT", Path(profile["WORK_ROOT"]))
    SEC_ZIP_DIR = _env_path("SEC_CCM_SEC_ZIP_DIR", Path(profile["SEC_ZIP_DIR"]))
    SEC_BATCH_ROOT = _env_path("SEC_CCM_SEC_BATCH_ROOT", Path(profile["SEC_BATCH_ROOT"]))
    SEC_YEAR_MERGED_DIR = _env_path(
        "SEC_CCM_SEC_YEAR_MERGED_DIR",
        Path(profile["SEC_YEAR_MERGED_DIR"]),
    )
    SEC_LIGHT_METADATA_PATH = _env_path(
        "SEC_CCM_SEC_LIGHT_METADATA_PATH",
        Path(profile["SEC_LIGHT_METADATA_PATH"]),
    )
    CCM_BASE_DIR = _env_path("SEC_CCM_CCM_BASE_DIR", Path(profile["CCM_BASE_DIR"]))
    CCM_DERIVED_DIR = _env_path(
        "SEC_CCM_CCM_DERIVED_DIR",
        Path(profile["CCM_DERIVED_DIR"]),
    )
    CCM_REUSE_DAILY_PATH = _env_path(
        "SEC_CCM_CCM_REUSE_DAILY_PATH",
        Path(profile["CCM_REUSE_DAILY_PATH"]),
    )
    CANONICAL_LINK_NAME = _env_str(
        "SEC_CCM_CANONICAL_LINK_NAME",
        str(profile["CANONICAL_LINK_NAME"]),
    )
    CCM_DAILY_NAME = _env_str(
        "SEC_CCM_CCM_DAILY_NAME",
        str(profile["CCM_DAILY_NAME"]),
    )
    RUN_ROOT = _env_path("SEC_CCM_RUN_ROOT", Path(profile["RUN_ROOT"]))
    LM2011_FF48_SICCODES_PATH = _resolve_ff48_siccodes_path(WORK_ROOT)

    available_years = _discover_sec_years(SEC_ZIP_DIR, SEC_YEAR_MERGED_DIR)
    if not available_years:
        available_years = list(range(1995, 2025))

    RUN_CCM_MODE = _env_str("SEC_CCM_RUN_CCM_MODE", str(profile["RUN_CCM_MODE"]))
    existing_year_outputs = _has_yearly_outputs(SEC_YEAR_MERGED_DIR)
    profile_run_sec_parse = _env_optional_bool(
        "SEC_CCM_RUN_SEC_PARSE",
        profile["RUN_SEC_PARSE"],
    )
    if profile_run_sec_parse is None:
        RUN_SEC_PARSE = not existing_year_outputs
    else:
        RUN_SEC_PARSE = profile_run_sec_parse
    profile_run_sec_yearly_merge = _env_optional_bool(
        "SEC_CCM_RUN_SEC_YEARLY_MERGE",
        profile["RUN_SEC_YEARLY_MERGE"],
    )
    if profile_run_sec_yearly_merge is None:
        RUN_SEC_YEARLY_MERGE = RUN_SEC_PARSE or not existing_year_outputs
    else:
        RUN_SEC_YEARLY_MERGE = profile_run_sec_yearly_merge
    RUN_SEC_CCM_PREMERGE = _env_bool("SEC_CCM_RUN_SEC_CCM_PREMERGE", False)
    RUN_REFINITIV_STEP1 = _env_bool("SEC_CCM_RUN_REFINITIV_STEP1", True)
    RUN_REFINITIV_STEP1_RESOLUTION = _env_bool(
        "SEC_CCM_RUN_REFINITIV_STEP1_RESOLUTION",
        True,
    )
    RUN_REFINITIV_OWNERSHIP_UNIVERSE_HANDOFF = _env_bool(
        "SEC_CCM_RUN_REFINITIV_OWNERSHIP_UNIVERSE_HANDOFF",
        True,
    )
    RUN_REFINITIV_OWNERSHIP_UNIVERSE_RESULTS = _env_bool(
        "SEC_CCM_RUN_REFINITIV_OWNERSHIP_UNIVERSE_RESULTS",
        True,
    )
    RUN_REFINITIV_OWNERSHIP_AUTHORITY = _env_bool(
        "SEC_CCM_RUN_REFINITIV_OWNERSHIP_AUTHORITY",
        True,
    )
    RUN_REFINITIV_DOC_OWNERSHIP_LM2011_EXACT_HANDOFF = _env_bool(
        "SEC_CCM_RUN_REFINITIV_DOC_OWNERSHIP_LM2011_EXACT_HANDOFF",
        True,
    )
    RUN_REFINITIV_DOC_OWNERSHIP_LM2011_FALLBACK_HANDOFF = _env_bool(
        "SEC_CCM_RUN_REFINITIV_DOC_OWNERSHIP_LM2011_FALLBACK_HANDOFF",
        True,
    )
    RUN_REFINITIV_DOC_OWNERSHIP_LM2011_FINALIZE = _env_bool(
        "SEC_CCM_RUN_REFINITIV_DOC_OWNERSHIP_LM2011_FINALIZE",
        True,
    )
    RUN_REFINITIV_DOC_ANALYST_LM2011_ANCHORS = _env_bool(
        "SEC_CCM_RUN_REFINITIV_DOC_ANALYST_LM2011_ANCHORS",
        True,
    )
    RUN_REFINITIV_DOC_ANALYST_LM2011_SELECT = _env_bool(
        "SEC_CCM_RUN_REFINITIV_DOC_ANALYST_LM2011_SELECT",
        True,
    )
    RUN_REFINITIV_INSTRUMENT_AUTHORITY = _env_bool(
        "SEC_CCM_RUN_REFINITIV_INSTRUMENT_AUTHORITY",
        True,
    )
    RUN_REFINITIV_ANALYST_REQUEST_GROUPS = _env_bool(
        "SEC_CCM_RUN_REFINITIV_ANALYST_REQUEST_GROUPS",
        True,
    )
    RUN_REFINITIV_ANALYST_ACTUALS = _env_bool(
        "SEC_CCM_RUN_REFINITIV_ANALYST_ACTUALS",
        True,
    )
    RUN_REFINITIV_ANALYST_ESTIMATES_MONTHLY = _env_bool(
        "SEC_CCM_RUN_REFINITIV_ANALYST_ESTIMATES_MONTHLY",
        True,
    )
    RUN_REFINITIV_ANALYST_NORMALIZE = _env_bool(
        "SEC_CCM_RUN_REFINITIV_ANALYST_NORMALIZE",
        True,
    )
    RUN_GATED_ITEM_EXTRACTION = _env_bool(
        "SEC_CCM_RUN_GATED_ITEM_EXTRACTION",
        False,
    )
    RUN_UNMATCHED_DIAGNOSTIC_TRACK = _env_bool(
        "SEC_CCM_RUN_UNMATCHED_DIAGNOSTIC_TRACK",
        False,
    )
    RUN_NO_ITEM_DIAGNOSTICS = _env_bool("SEC_CCM_RUN_NO_ITEM_DIAGNOSTICS", False)
    RUN_BOUNDARY_DIAGNOSTICS = _env_bool("SEC_CCM_RUN_BOUNDARY_DIAGNOSTICS", False)
    RUN_VALIDATION_CHECKS = _env_bool("SEC_CCM_RUN_VALIDATION_CHECKS", False)

    SEC_PARSE_MODE = _env_str("SEC_CCM_SEC_PARSE_MODE", "parsed")
    YEARS = _env_int_list("SEC_CCM_YEARS", available_years)
    ITEM_EXTRACTION_REGIME = _env_str(
        "SEC_CCM_ITEM_EXTRACTION_REGIME",
        "legacy",
    )

    SEC_CCM_OUTPUT_DIR = _env_path("SEC_CCM_OUTPUT_DIR", RUN_ROOT / "sec_ccm_premerge")
    SEC_ITEMS_ANALYSIS_DIR = _env_path(
        "SEC_CCM_ITEMS_ANALYSIS_DIR",
        RUN_ROOT / "items_analysis",
    )
    SEC_ITEMS_DIAGNOSTIC_DIR = _env_path(
        "SEC_CCM_ITEMS_DIAGNOSTIC_DIR",
        RUN_ROOT / "items_diagnostic",
    )
    SEC_NO_ITEM_DIR = _env_path(
        "SEC_CCM_NO_ITEM_DIR",
        RUN_ROOT / "no_item_diagnostics",
    )
    BOUNDARY_OUT_DIR = _env_path(
        "SEC_CCM_BOUNDARY_OUT_DIR",
        RUN_ROOT / "boundary_diagnostics",
    )
    BOUNDARY_INPUT_DIR = _env_path(
        "SEC_CCM_BOUNDARY_INPUT_DIR",
        BOUNDARY_OUT_DIR / "matched_filings_input",
    )
    REFINITIV_STEP1_OUT_DIR = _env_path(
        "SEC_CCM_REFINITIV_STEP1_OUT_DIR",
        RUN_ROOT / "refinitiv_step1",
    )
    REFINITIV_OWNERSHIP_UNIVERSE_DIR = _env_path(
        "SEC_CCM_REFINITIV_OWNERSHIP_UNIVERSE_DIR",
        REFINITIV_STEP1_OUT_DIR / "ownership_universe_common_stock",
    )
    REFINITIV_OWNERSHIP_AUTHORITY_DIR = _env_path(
        "SEC_CCM_REFINITIV_OWNERSHIP_AUTHORITY_DIR",
        REFINITIV_STEP1_OUT_DIR / "ownership_authority_common_stock",
    )
    REFINITIV_ANALYST_COMMON_STOCK_DIR = _env_path(
        "SEC_CCM_REFINITIV_ANALYST_COMMON_STOCK_DIR",
        REFINITIV_STEP1_OUT_DIR / "analyst_common_stock",
    )
    REFINITIV_DOC_OWNERSHIP_LM2011_DIR = _env_path(
        "SEC_CCM_REFINITIV_DOC_OWNERSHIP_LM2011_DIR",
        RUN_ROOT / "refinitiv_doc_ownership_lm2011",
    )
    REFINITIV_DOC_ANALYST_LM2011_DIR = _env_path(
        "SEC_CCM_REFINITIV_DOC_ANALYST_LM2011_DIR",
        RUN_ROOT / "refinitiv_doc_analyst_lm2011",
    )

    if IN_COLAB:
        default_local_tmp = Path("/content/_tmp_zip")
        default_local_work = Path("/content/_batch_work")
        default_local_item_work = Path("/content/_item_work")
        default_local_merge_work = Path("/content/_merge_work")
    else:
        default_local_tmp = ROOT / ".tmp" / "zip"
        default_local_work = ROOT / ".tmp" / "batch_work"
        default_local_item_work = ROOT / ".tmp" / "item_work"
        default_local_merge_work = ROOT / ".tmp" / "merge_work"

    LOCAL_TMP = _env_path("SEC_CCM_LOCAL_TMP", default_local_tmp)
    LOCAL_WORK = _env_path("SEC_CCM_LOCAL_WORK", default_local_work)
    LOCAL_ITEM_WORK = _env_path("SEC_CCM_LOCAL_ITEM_WORK", default_local_item_work)
    LOCAL_MERGE_WORK = _env_path("SEC_CCM_LOCAL_MERGE_WORK", default_local_merge_work)

    for path in [
        SEC_BATCH_ROOT,
        SEC_YEAR_MERGED_DIR,
        RUN_ROOT,
        SEC_CCM_OUTPUT_DIR,
        SEC_ITEMS_ANALYSIS_DIR,
        SEC_ITEMS_DIAGNOSTIC_DIR,
        SEC_NO_ITEM_DIR,
        BOUNDARY_OUT_DIR,
        BOUNDARY_INPUT_DIR,
        REFINITIV_STEP1_OUT_DIR,
        REFINITIV_OWNERSHIP_UNIVERSE_DIR,
        REFINITIV_OWNERSHIP_AUTHORITY_DIR,
        REFINITIV_ANALYST_COMMON_STOCK_DIR,
        REFINITIV_DOC_OWNERSHIP_LM2011_DIR,
        REFINITIV_DOC_ANALYST_LM2011_DIR,
        LOCAL_TMP,
        LOCAL_WORK,
        LOCAL_ITEM_WORK,
        LOCAL_MERGE_WORK,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    default_forms_10k_10q = [
        "10-K",
        "10-K/A",
        "10-KA",
        "10-Q",
        "10-Q/A",
        "10-QA",
        "10-KT",
        "10-KT/A",
        "10-QT",
        "10-QT/A",
        "10-K405",
    ]
    FORMS_10K_10Q = _env_str_list("SEC_CCM_FORMS_10K_10Q", default_forms_10k_10q)
    DAILY_FEATURE_COLUMNS = tuple(
        _env_str_list(
            "SEC_CCM_DAILY_FEATURE_COLUMNS",
            list(SEC_CCM_PHASE_B_DAILY_FEATURE_COLUMNS),
        )
    )
    REQUIRED_DAILY_NON_NULL_FEATURES = tuple(
        _env_str_list("SEC_CCM_REQUIRED_DAILY_NON_NULL_FEATURES", ["RET"])
    )

    print(
        {
            "DATA_PROFILE": DATA_PROFILE,
            "RUN_CCM_MODE": RUN_CCM_MODE,
            "WORK_ROOT": str(WORK_ROOT),
            "SEC_ZIP_DIR": str(SEC_ZIP_DIR),
            "SEC_BATCH_ROOT": str(SEC_BATCH_ROOT),
            "CCM_BASE_DIR": str(CCM_BASE_DIR),
            "RUN_ROOT": str(RUN_ROOT),
            "LM2011_FF48_SICCODES_PATH": str(LM2011_FF48_SICCODES_PATH),
            "RUN_SEC_PARSE": RUN_SEC_PARSE,
            "RUN_SEC_YEARLY_MERGE": RUN_SEC_YEARLY_MERGE,
            "RUN_SEC_CCM_PREMERGE": RUN_SEC_CCM_PREMERGE,
            "RUN_REFINITIV_DOC_ANALYST_LM2011_ANCHORS": RUN_REFINITIV_DOC_ANALYST_LM2011_ANCHORS,
            "RUN_REFINITIV_DOC_ANALYST_LM2011_SELECT": RUN_REFINITIV_DOC_ANALYST_LM2011_SELECT,
            "RUN_REFINITIV_INSTRUMENT_AUTHORITY": RUN_REFINITIV_INSTRUMENT_AUTHORITY,
            "RUN_REFINITIV_ANALYST_REQUEST_GROUPS": RUN_REFINITIV_ANALYST_REQUEST_GROUPS,
            "RUN_REFINITIV_ANALYST_ACTUALS": RUN_REFINITIV_ANALYST_ACTUALS,
            "RUN_REFINITIV_ANALYST_ESTIMATES_MONTHLY": RUN_REFINITIV_ANALYST_ESTIMATES_MONTHLY,
            "RUN_REFINITIV_ANALYST_NORMALIZE": RUN_REFINITIV_ANALYST_NORMALIZE,
            "RUN_GATED_ITEM_EXTRACTION": RUN_GATED_ITEM_EXTRACTION,
            "RUN_VALIDATION_CHECKS": RUN_VALIDATION_CHECKS,
            "SEC_PARSE_MODE": SEC_PARSE_MODE,
            "ITEM_EXTRACTION_REGIME": ITEM_EXTRACTION_REGIME,
            "year_count": len(YEARS),
            "year_range": (YEARS[0], YEARS[-1]) if YEARS else None,
        }
    )

    # ## Preflight
    required_paths: list[tuple[str, Path]] = []

    if RUN_CCM_MODE == "REUSE":
        required_paths.extend(
            [
                ("CCM_REUSE_DAILY_PATH", CCM_REUSE_DAILY_PATH),
                ("CANONICAL_LINK_PATH", CCM_DERIVED_DIR / CANONICAL_LINK_NAME),
            ]
        )
    elif RUN_CCM_MODE == "REBUILD":
        required_paths.append(("CCM_BASE_DIR", CCM_BASE_DIR))
    else:
        raise ValueError(f"Unsupported RUN_CCM_MODE={RUN_CCM_MODE!r}")

    if RUN_SEC_PARSE:
        required_paths.append(("SEC_ZIP_DIR", SEC_ZIP_DIR))
    elif RUN_SEC_YEARLY_MERGE:
        required_paths.append(("SEC_BATCH_ROOT", SEC_BATCH_ROOT))
    else:
        required_paths.append(("SEC_YEAR_MERGED_DIR", SEC_YEAR_MERGED_DIR))

    missing: list[tuple[str, Path]] = []
    for label, path in required_paths:
        exists = path.exists()
        print({"label": label, "path": str(path), "exists": exists})
        if not exists:
            missing.append((label, path))

    if RUN_SEC_PARSE:
        zip_paths = [SEC_ZIP_DIR / f"{year}.zip" for year in YEARS]
        found_zip_count = sum(path.exists() for path in zip_paths)
        print(
            {
                "sec_zip_dir": str(SEC_ZIP_DIR),
                "requested_years": len(YEARS),
                "found_zip_count": found_zip_count,
            }
        )
        if found_zip_count == 0:
            missing.append(("SEC_ZIP_FILES", SEC_ZIP_DIR))
    elif not RUN_SEC_YEARLY_MERGE:
        existing_year_files = [
            path
            for path in SEC_YEAR_MERGED_DIR.glob("*.parquet")
            if path.stem.isdigit() and len(path.stem) == 4
        ]
        print(
            {
                "sec_year_merged_dir": str(SEC_YEAR_MERGED_DIR),
                "existing_year_file_count": len(existing_year_files),
            }
        )
        if not existing_year_files:
            missing.append(("SEC_YEAR_MERGED_FILES", SEC_YEAR_MERGED_DIR))

    if missing:
        details = "\n".join(f"- {label}: {path}" for label, path in missing)
        raise FileNotFoundError(
            f"Profile preflight failed. Missing required inputs:\n{details}"
        )

    # ## 1) CCM stage (build or reuse)
    ccm_stage_paths = build_or_reuse_ccm_daily_stage(
        run_mode=RUN_CCM_MODE,
        ccm_base_dir=CCM_BASE_DIR,
        ccm_derived_dir=CCM_DERIVED_DIR,
        ccm_reuse_daily_path=Path(CCM_REUSE_DAILY_PATH),
        forms_10k_10q=FORMS_10K_10Q,
        start_date="1990-01-01",
        canonical_name=CANONICAL_LINK_NAME,
        daily_name=CCM_DAILY_NAME,
        verbose=1,
    )

    ccm_daily_path = ccm_stage_paths["ccm_daily_path"]
    ccm_daily_market_core_path = ccm_stage_paths["ccm_daily_market_core_path"]
    ccm_daily_phase_b_surface_path = ccm_stage_paths["ccm_daily_phase_b_surface_path"]
    ccm_daily_bridge_surface_path = ccm_stage_paths["ccm_daily_bridge_surface_path"]
    canonical_link_path = ccm_stage_paths["canonical_link_path"]

    ccm_daily_market_core_lf = pl.scan_parquet(ccm_daily_market_core_path)
    ccm_daily_phase_b_lf = pl.scan_parquet(ccm_daily_phase_b_surface_path)
    ccm_daily_bridge_lf = pl.scan_parquet(ccm_daily_bridge_surface_path)
    ccm_daily_legacy_lf = pl.scan_parquet(ccm_daily_path)

    market_core_schema = ccm_daily_market_core_lf.collect_schema()
    phase_b_schema = ccm_daily_phase_b_lf.collect_schema()
    bridge_schema = ccm_daily_bridge_lf.collect_schema()
    legacy_schema = ccm_daily_legacy_lf.collect_schema()

    missing_market_core_cols = [col for col in CCM_DAILY_MARKET_CORE_COLUMNS if col not in market_core_schema]
    missing_phase_b_cols = [col for col in CCM_DAILY_PHASE_B_SURFACE_COLUMNS if col not in phase_b_schema]
    bridge_contract_cols = (
        *CCM_DAILY_BRIDGE_SURFACE_REQUIRED_COLUMNS,
        *CCM_DAILY_BRIDGE_SURFACE_OPTIONAL_COLUMNS,
    )
    missing_bridge_cols = [col for col in bridge_contract_cols if col not in bridge_schema]
    legacy_compat_cols = ("SHROUT", "SRCTYPE_all", "FILEDATE_all", "FILEDATETIME_all", "n_filings")
    missing_legacy_compat_cols = [col for col in legacy_compat_cols if col not in legacy_schema]
    print(
        {
            "ccm_daily_path": str(ccm_daily_path),
            "ccm_daily_market_core_path": str(ccm_daily_market_core_path),
            "ccm_daily_phase_b_surface_path": str(ccm_daily_phase_b_surface_path),
            "ccm_daily_bridge_surface_path": str(ccm_daily_bridge_surface_path),
            "canonical_link_path": str(canonical_link_path),
            "market_core_rows": ccm_daily_market_core_lf.select(pl.len()).collect().item(),
            "phase_b_rows": ccm_daily_phase_b_lf.select(pl.len()).collect().item(),
            "bridge_rows": ccm_daily_bridge_lf.select(pl.len()).collect().item(),
            "legacy_rows": ccm_daily_legacy_lf.select(pl.len()).collect().item(),
            "missing_market_core_cols": missing_market_core_cols,
            "missing_phase_b_cols": missing_phase_b_cols,
            "missing_bridge_cols": missing_bridge_cols,
            "missing_legacy_compat_cols": missing_legacy_compat_cols,
        }
    )
    if RUN_CCM_MODE == "REUSE" and (
        missing_market_core_cols or missing_phase_b_cols or missing_bridge_cols or missing_legacy_compat_cols
    ):
        print(
            {
                "warning": "reused daily artifact stack is missing expected surface columns",
                "missing_market_core_cols": missing_market_core_cols,
                "missing_phase_b_cols": missing_phase_b_cols,
                "missing_bridge_cols": missing_bridge_cols,
                "missing_legacy_compat_cols": missing_legacy_compat_cols,
                "recommended_action": "switch RUN_CCM_MODE to REBUILD to refresh the daily panel",
            }
        )

    # ## 2) Build link universe + trading calendar
    schema = phase_b_schema
    resolved_permno_col = _first_existing(
        schema,
        ("KYPERMNO", "LPERMNO", "PERMNO"),
        "ccm_daily_phase_b_surface",
    )
    resolved_date_col = _first_existing(
        schema,
        ("CALDT", "caldt"),
        "ccm_daily_phase_b_surface",
    )

    link_universe_lf = pl.scan_parquet(canonical_link_path)
    trading_calendar_lf = (
        ccm_daily_market_core_lf.select(
            pl.col(resolved_date_col).cast(pl.Date, strict=False).alias("CALDT")
        )
        .drop_nulls(subset=["CALDT"])
        .unique()
        .sort("CALDT")
    )

    print(
        {
            "permno_col": resolved_permno_col,
            "date_col": resolved_date_col,
            "canonical_link_path": str(canonical_link_path),
        }
    )
    print(
        {
            "link_rows": link_universe_lf.select(pl.len()).collect().item(),
            "trading_days": trading_calendar_lf.select(pl.len()).collect().item(),
        }
    )

    # ## 2b) Refinitiv bridge step 1
    refinitiv_step1_paths: dict[str, Path] | None = None
    refinitiv_resolution_paths: dict[str, Path] | None = None
    refinitiv_ownership_universe_handoff_paths: dict[str, Path] | None = None
    refinitiv_ownership_universe_results_paths: dict[str, Path] | None = None
    refinitiv_ownership_authority_paths: dict[str, Path] | None = None
    refinitiv_instrument_authority_paths: dict[str, Path] | None = None
    refinitiv_analyst_request_group_paths: dict[str, Path] | None = None
    refinitiv_analyst_actuals_paths: dict[str, Path] | None = None
    refinitiv_analyst_estimates_paths: dict[str, Path] | None = None
    refinitiv_analyst_normalize_paths: dict[str, Path] | None = None
    refinitiv_doc_ownership_exact_paths: dict[str, Path] | None = None
    refinitiv_doc_ownership_fallback_paths: dict[str, Path] | None = None
    refinitiv_doc_ownership_finalize_paths: dict[str, Path] | None = None
    refinitiv_doc_analyst_anchor_paths: dict[str, Path] | None = None
    refinitiv_doc_analyst_select_paths: dict[str, Path] | None = None
    refinitiv_artifact_stages: list[_ArtifactStage] = []

    def _record_refinitiv_stage(stage: str, paths: dict[str, Path] | None) -> None:
        if paths is None:
            return
        refinitiv_artifact_stages.append(
            _ArtifactStage(
                name=stage,
                paths={key: Path(value) for key, value in paths.items()},
            )
        )

    if RUN_REFINITIV_STEP1:
        company_description_path = CCM_BASE_DIR / "companydescription.parquet"
        company_description_lf = (
            pl.scan_parquet(company_description_path)
            if company_description_path.exists()
            else None
        )
        refinitiv_step1_paths = run_refinitiv_step1_bridge_pipeline(
            daily_lf=ccm_daily_bridge_lf,
            output_dir=REFINITIV_STEP1_OUT_DIR,
            company_description_lf=company_description_lf,
            source_daily_path=ccm_daily_bridge_surface_path,
        )
        _record_refinitiv_stage("refinitiv_step1", refinitiv_step1_paths)
        for key in sorted(refinitiv_step1_paths):
            print(f"{key}: {refinitiv_step1_paths[key]}")
    if RUN_REFINITIV_STEP1_RESOLUTION:
        filled_extended_lookup_path = (
            REFINITIV_STEP1_OUT_DIR / "refinitiv_ric_lookup_handoff_common_stock_extended_filled_in.xlsx"
        )
        api_lookup_snapshot_path = (
            REFINITIV_STEP1_OUT_DIR / "refinitiv_ric_lookup_handoff_common_stock_extended_snapshot.parquet"
        )
        api_lookup_parquet_path = (
            REFINITIV_STEP1_OUT_DIR / "refinitiv_ric_lookup_handoff_common_stock_extended.parquet"
        )
        if LSEG_API_READY and api_lookup_snapshot_path.exists():
            refinitiv_lookup_api_paths = run_refinitiv_step1_lookup_api_pipeline(
                snapshot_parquet_path=api_lookup_snapshot_path,
                output_dir=REFINITIV_STEP1_OUT_DIR,
            )
            _record_refinitiv_stage("refinitiv_lookup_api", refinitiv_lookup_api_paths)
            for key in sorted(refinitiv_lookup_api_paths):
                print(f"{key}: {refinitiv_lookup_api_paths[key]}")
            refinitiv_resolution_paths = run_refinitiv_step1_resolution_pipeline(
                filled_lookup_workbook_path=api_lookup_parquet_path,
                output_dir=REFINITIV_STEP1_OUT_DIR,
            )
            _record_refinitiv_stage("refinitiv_resolution", refinitiv_resolution_paths)
            for key in sorted(refinitiv_resolution_paths):
                print(f"{key}: {refinitiv_resolution_paths[key]}")
        elif filled_extended_lookup_path.exists():
            refinitiv_resolution_paths = run_refinitiv_step1_resolution_pipeline(
                filled_lookup_workbook_path=filled_extended_lookup_path,
                output_dir=REFINITIV_STEP1_OUT_DIR,
            )
            _record_refinitiv_stage("refinitiv_resolution", refinitiv_resolution_paths)
            for key in sorted(refinitiv_resolution_paths):
                print(f"{key}: {refinitiv_resolution_paths[key]}")
        else:
            print(
                {
                    "warning": "skipping Refinitiv step-1 resolution; filled extended lookup workbook not found",
                    "expected_path": str(filled_extended_lookup_path),
                }
            )
    if RUN_REFINITIV_OWNERSHIP_UNIVERSE_HANDOFF:
        resolution_artifact_path = (
            refinitiv_resolution_paths["refinitiv_ric_resolution_common_stock_parquet"]
            if refinitiv_resolution_paths is not None
            else REFINITIV_STEP1_OUT_DIR / "refinitiv_ric_resolution_common_stock.parquet"
        )
        if resolution_artifact_path.exists():
            refinitiv_ownership_universe_handoff_paths = run_refinitiv_step1_ownership_universe_handoff_pipeline(
                resolution_artifact_path=resolution_artifact_path,
                output_dir=REFINITIV_OWNERSHIP_UNIVERSE_DIR,
            )
            _record_refinitiv_stage(
                "refinitiv_ownership_universe_handoff",
                refinitiv_ownership_universe_handoff_paths,
            )
            for key in sorted(refinitiv_ownership_universe_handoff_paths):
                print(f"{key}: {refinitiv_ownership_universe_handoff_paths[key]}")
        else:
            print(
                {
                    "warning": "skipping Refinitiv ownership universe handoff; resolved parquet not found",
                    "expected_path": str(resolution_artifact_path),
                }
            )
    if RUN_REFINITIV_OWNERSHIP_UNIVERSE_RESULTS:
        ownership_universe_filled_workbook_path = (
            REFINITIV_OWNERSHIP_UNIVERSE_DIR
            / "refinitiv_ownership_universe_handoff_common_stock_filled_in.xlsx"
        )
        ownership_universe_handoff_parquet_path = (
            refinitiv_ownership_universe_handoff_paths["refinitiv_ownership_universe_handoff_common_stock_parquet"]
            if refinitiv_ownership_universe_handoff_paths is not None
            else REFINITIV_OWNERSHIP_UNIVERSE_DIR / "refinitiv_ownership_universe_handoff_common_stock.parquet"
        )
        if LSEG_API_READY and ownership_universe_handoff_parquet_path.exists():
            refinitiv_ownership_universe_results_paths = run_refinitiv_step1_ownership_universe_api_pipeline(
                handoff_parquet_path=ownership_universe_handoff_parquet_path,
                output_dir=REFINITIV_OWNERSHIP_UNIVERSE_DIR,
            )
            _record_refinitiv_stage(
                "refinitiv_ownership_universe_results",
                refinitiv_ownership_universe_results_paths,
            )
            for key in sorted(refinitiv_ownership_universe_results_paths):
                print(f"{key}: {refinitiv_ownership_universe_results_paths[key]}")
        elif ownership_universe_filled_workbook_path.exists():
            refinitiv_ownership_universe_results_paths = run_refinitiv_step1_ownership_universe_results_pipeline(
                filled_workbook_path=ownership_universe_filled_workbook_path,
                output_dir=REFINITIV_OWNERSHIP_UNIVERSE_DIR,
            )
            _record_refinitiv_stage(
                "refinitiv_ownership_universe_results",
                refinitiv_ownership_universe_results_paths,
            )
            for key in sorted(refinitiv_ownership_universe_results_paths):
                print(f"{key}: {refinitiv_ownership_universe_results_paths[key]}")
        else:
            print(
                {
                    "warning": "skipping Refinitiv ownership universe results; filled ownership universe workbook not found",
                    "expected_path": str(ownership_universe_filled_workbook_path),
                }
            )
    if RUN_REFINITIV_OWNERSHIP_AUTHORITY:
        resolution_artifact_path = (
            refinitiv_resolution_paths["refinitiv_ric_resolution_common_stock_parquet"]
            if refinitiv_resolution_paths is not None
            else REFINITIV_STEP1_OUT_DIR / "refinitiv_ric_resolution_common_stock.parquet"
        )
        ownership_results_artifact_path = (
            refinitiv_ownership_universe_results_paths["refinitiv_ownership_universe_results_parquet"]
            if refinitiv_ownership_universe_results_paths is not None
            else REFINITIV_OWNERSHIP_UNIVERSE_DIR / "refinitiv_ownership_universe_results.parquet"
        )
        ownership_row_summary_artifact_path = (
            refinitiv_ownership_universe_results_paths["refinitiv_ownership_universe_row_summary_parquet"]
            if refinitiv_ownership_universe_results_paths is not None
            else REFINITIV_OWNERSHIP_UNIVERSE_DIR / "refinitiv_ownership_universe_row_summary.parquet"
        )
        reviewed_ticker_allowlist_path = (
            REFINITIV_OWNERSHIP_AUTHORITY_DIR / "refinitiv_permno_ownership_ticker_allowlist.parquet"
        )
        if (
            resolution_artifact_path.exists()
            and ownership_results_artifact_path.exists()
            and ownership_row_summary_artifact_path.exists()
        ):
            refinitiv_ownership_authority_paths = run_refinitiv_step1_ownership_authority_pipeline(
                resolution_artifact_path=resolution_artifact_path,
                ownership_results_artifact_path=ownership_results_artifact_path,
                ownership_row_summary_artifact_path=ownership_row_summary_artifact_path,
                output_dir=REFINITIV_OWNERSHIP_AUTHORITY_DIR,
                reviewed_ticker_allowlist_path=reviewed_ticker_allowlist_path,
            )
            _record_refinitiv_stage(
                "refinitiv_ownership_authority",
                refinitiv_ownership_authority_paths,
            )
            for key in sorted(refinitiv_ownership_authority_paths):
                print(f"{key}: {refinitiv_ownership_authority_paths[key]}")
        else:
            print(
                {
                    "warning": "skipping Refinitiv ownership authority; required ownership artifacts not found",
                    "expected_resolution_path": str(resolution_artifact_path),
                    "expected_results_path": str(ownership_results_artifact_path),
                    "expected_row_summary_path": str(ownership_row_summary_artifact_path),
                }
            )
    if RUN_REFINITIV_INSTRUMENT_AUTHORITY:
        bridge_artifact_path = (
            refinitiv_step1_paths["refinitiv_bridge_universe_parquet"]
            if refinitiv_step1_paths is not None
            else REFINITIV_STEP1_OUT_DIR / "refinitiv_bridge_universe.parquet"
        )
        resolution_artifact_path = (
            refinitiv_resolution_paths["refinitiv_ric_resolution_common_stock_parquet"]
            if refinitiv_resolution_paths is not None
            else REFINITIV_STEP1_OUT_DIR / "refinitiv_ric_resolution_common_stock.parquet"
        )
        if bridge_artifact_path.exists() and resolution_artifact_path.exists():
            refinitiv_instrument_authority_paths = run_refinitiv_step1_instrument_authority_pipeline(
                bridge_artifact_path=bridge_artifact_path,
                resolution_artifact_path=resolution_artifact_path,
                output_dir=REFINITIV_STEP1_OUT_DIR,
            )
            _record_refinitiv_stage(
                "refinitiv_instrument_authority",
                refinitiv_instrument_authority_paths,
            )
            for key in sorted(refinitiv_instrument_authority_paths):
                print(f"{key}: {refinitiv_instrument_authority_paths[key]}")
        else:
            print(
                {
                    "warning": "skipping Refinitiv instrument authority; required artifacts not found",
                    "expected_bridge_path": str(bridge_artifact_path),
                    "expected_resolution_path": str(resolution_artifact_path),
                }
            )
    if RUN_REFINITIV_ANALYST_REQUEST_GROUPS:
        instrument_authority_artifact_path = (
            refinitiv_instrument_authority_paths["refinitiv_instrument_authority_common_stock_parquet"]
            if refinitiv_instrument_authority_paths is not None
            else REFINITIV_STEP1_OUT_DIR / "refinitiv_instrument_authority_common_stock.parquet"
        )
        if instrument_authority_artifact_path.exists():
            refinitiv_analyst_request_group_paths = run_refinitiv_step1_analyst_request_groups_pipeline(
                instrument_authority_artifact_path=instrument_authority_artifact_path,
                output_dir=REFINITIV_ANALYST_COMMON_STOCK_DIR,
            )
            _record_refinitiv_stage(
                "refinitiv_analyst_request_groups",
                refinitiv_analyst_request_group_paths,
            )
            for key in sorted(refinitiv_analyst_request_group_paths):
                print(f"{key}: {refinitiv_analyst_request_group_paths[key]}")
        else:
            print(
                {
                    "warning": "skipping Refinitiv analyst request groups; instrument authority artifact not found",
                    "expected_instrument_authority_path": str(instrument_authority_artifact_path),
                }
            )
    if RUN_REFINITIV_ANALYST_ACTUALS:
        analyst_request_universe_path = (
            refinitiv_analyst_request_group_paths["refinitiv_analyst_request_universe_common_stock_parquet"]
            if refinitiv_analyst_request_group_paths is not None
            else REFINITIV_ANALYST_COMMON_STOCK_DIR / "refinitiv_analyst_request_universe_common_stock.parquet"
        )
        if LSEG_API_READY and analyst_request_universe_path.exists():
            refinitiv_analyst_actuals_paths = run_refinitiv_step1_analyst_actuals_api_pipeline(
                request_universe_parquet_path=analyst_request_universe_path,
                output_dir=REFINITIV_ANALYST_COMMON_STOCK_DIR,
            )
            _record_refinitiv_stage(
                "refinitiv_analyst_actuals",
                refinitiv_analyst_actuals_paths,
            )
            for key in sorted(refinitiv_analyst_actuals_paths):
                print(f"{key}: {refinitiv_analyst_actuals_paths[key]}")
        else:
            print(
                {
                    "warning": "skipping Refinitiv analyst actuals; API unavailable or request universe not found",
                    "lseg_api_ready": LSEG_API_READY,
                    "expected_request_universe_path": str(analyst_request_universe_path),
                }
            )
    if RUN_REFINITIV_ANALYST_ESTIMATES_MONTHLY:
        analyst_request_universe_path = (
            refinitiv_analyst_request_group_paths["refinitiv_analyst_request_universe_common_stock_parquet"]
            if refinitiv_analyst_request_group_paths is not None
            else REFINITIV_ANALYST_COMMON_STOCK_DIR / "refinitiv_analyst_request_universe_common_stock.parquet"
        )
        if LSEG_API_READY and analyst_request_universe_path.exists():
            refinitiv_analyst_estimates_paths = run_refinitiv_step1_analyst_estimates_monthly_api_pipeline(
                request_universe_parquet_path=analyst_request_universe_path,
                output_dir=REFINITIV_ANALYST_COMMON_STOCK_DIR,
            )
            _record_refinitiv_stage(
                "refinitiv_analyst_estimates_monthly",
                refinitiv_analyst_estimates_paths,
            )
            for key in sorted(refinitiv_analyst_estimates_paths):
                print(f"{key}: {refinitiv_analyst_estimates_paths[key]}")
        else:
            print(
                {
                    "warning": "skipping Refinitiv analyst estimates monthly; API unavailable or request universe not found",
                    "lseg_api_ready": LSEG_API_READY,
                    "expected_request_universe_path": str(analyst_request_universe_path),
                }
            )
    if RUN_REFINITIV_ANALYST_NORMALIZE:
        analyst_actuals_raw_path = (
            refinitiv_analyst_actuals_paths["refinitiv_analyst_actuals_raw_parquet"]
            if refinitiv_analyst_actuals_paths is not None
            else REFINITIV_ANALYST_COMMON_STOCK_DIR / "refinitiv_analyst_actuals_raw.parquet"
        )
        analyst_estimates_raw_path = (
            refinitiv_analyst_estimates_paths["refinitiv_analyst_estimates_monthly_raw_parquet"]
            if refinitiv_analyst_estimates_paths is not None
            else REFINITIV_ANALYST_COMMON_STOCK_DIR / "refinitiv_analyst_estimates_monthly_raw.parquet"
        )
        if analyst_actuals_raw_path.exists() and analyst_estimates_raw_path.exists():
            refinitiv_analyst_normalize_paths = run_refinitiv_step1_analyst_normalize_pipeline(
                actuals_raw_artifact_path=analyst_actuals_raw_path,
                estimates_raw_artifact_path=analyst_estimates_raw_path,
                output_dir=REFINITIV_ANALYST_COMMON_STOCK_DIR,
            )
            _record_refinitiv_stage(
                "refinitiv_analyst_normalize",
                refinitiv_analyst_normalize_paths,
            )
            for key in sorted(refinitiv_analyst_normalize_paths):
                print(f"{key}: {refinitiv_analyst_normalize_paths[key]}")
        else:
            print(
                {
                    "warning": "skipping Refinitiv analyst normalize; required raw analyst artifacts not found",
                    "expected_actuals_raw_path": str(analyst_actuals_raw_path),
                    "expected_estimates_raw_path": str(analyst_estimates_raw_path),
                }
            )
    # ## 3) SEC parse and yearly merge
    if RUN_SEC_PARSE:
        common = dict(
            tmp_dir=LOCAL_TMP,
            local_work_dir=LOCAL_WORK,
            compression="zstd",
            copy_retries=5,
            copy_sleep=2.0,
            validate_on_copy=True,
        )
        for year in YEARS:
            zip_path = SEC_ZIP_DIR / f"{year}.zip"
            if not zip_path.exists():
                continue
            out_year = SEC_BATCH_ROOT / str(year)
            out_year.mkdir(parents=True, exist_ok=True)
            existing = list(out_year.glob(f"{year}_batch_*.parquet"))
            if existing:
                continue
            if SEC_PARSE_MODE == "raw":
                process_zip_year_raw_text(
                    zip_path=zip_path,
                    out_dir=out_year,
                    batch_max_rows=1000,
                    batch_max_text_bytes=250 * 1024 * 1024,
                    encoding="utf-8",
                    **common,
                )
            else:
                process_zip_year(
                    zip_path=zip_path,
                    out_dir=out_year,
                    batch_max_rows=2000,
                    batch_max_text_bytes=250 * 1024 * 1024,
                    header_search_limit=8000,
                    encoding="utf-8",
                    **common,
                )
    else:
        print("RUN_SEC_PARSE=False; using existing SEC batches.")

    if RUN_SEC_YEARLY_MERGE:
        merge_yearly_batches(
            batch_dir=SEC_BATCH_ROOT,
            out_dir=SEC_YEAR_MERGED_DIR,
            checkpoint_path=SEC_YEAR_MERGED_DIR / "done_years.json",
            local_work_dir=LOCAL_MERGE_WORK,
            batch_size=128_000,
            compression="zstd",
            compression_level=1,
            validate_inputs="full",
            years=[str(year) for year in YEARS],
        )

    sec_summaries = summarize_year_parquets(SEC_YEAR_MERGED_DIR)
    ok_files = [Path(row["path"]) for row in sec_summaries if row.get("status") == "OK"]
    if not ok_files:
        raise ValueError("No OK SEC yearly parquet files found.")
    build_light_metadata_dataset(
        parquet_dir=ok_files,
        out_path=SEC_LIGHT_METADATA_PATH,
        drop_columns=("full_text",),
        sort_columns=("file_date_filename", "cik"),
        compression="zstd",
    )
    print({"ok_year_files": len(ok_files), "light_path": str(SEC_LIGHT_METADATA_PATH)})

    # ## 4) Prepare SEC pre-merge input
    year_files = sorted(
        [
            path
            for path in SEC_YEAR_MERGED_DIR.glob("*.parquet")
            if path.stem.isdigit() and len(path.stem) == 4
        ]
    )
    if not year_files:
        raise ValueError(f"No yearly SEC files found in {SEC_YEAR_MERGED_DIR}")

    sec_raw_lf = pl.scan_parquet(year_files)
    sec_schema = sec_raw_lf.collect_schema()
    for column in ("doc_id", "cik_10"):
        if column not in sec_schema:
            raise ValueError(f"Missing required SEC column: {column}")

    if "filing_date" in sec_schema and "file_date_filename" in sec_schema:
        filing_date_expr = pl.coalesce(
            [
                pl.col("filing_date").cast(pl.Date, strict=False),
                pl.col("file_date_filename").cast(pl.Date, strict=False),
            ]
        ).alias("filing_date")
    elif "filing_date" in sec_schema:
        filing_date_expr = pl.col("filing_date").cast(pl.Date, strict=False).alias(
            "filing_date"
        )
    elif "file_date_filename" in sec_schema:
        filing_date_expr = pl.col("file_date_filename").cast(
            pl.Date,
            strict=False,
        ).alias("filing_date")
    else:
        raise ValueError("Missing both filing_date and file_date_filename.")

    optional_cols = [
        column
        for column in (
            "document_type_filename",
            "form_type",
            "period_end",
            "acceptance_datetime",
            "accession_number",
            "accession_nodash",
        )
        if column in sec_schema
    ]
    sec_premerge_input_lf = (
        sec_raw_lf.with_columns(
            pl.col("doc_id").cast(pl.Utf8, strict=False),
            pl.col("cik_10").cast(pl.Utf8, strict=False),
            filing_date_expr,
        ).select("doc_id", "cik_10", "filing_date", *optional_cols)
    )

    null_dates = (
        sec_premerge_input_lf.select(pl.col("filing_date").is_null().sum()).collect().item()
    )
    if null_dates > 0:
        raise ValueError(f"Null filing_date rows after fallback: {null_dates}")

    print(
        {
            "rows": sec_premerge_input_lf.select(pl.len()).collect().item(),
            "doc_ids": sec_premerge_input_lf.select(pl.col("doc_id").n_unique())
            .collect()
            .item(),
            "optional_cols": optional_cols,
        }
    )

    # ## 5) SEC-CCM pre-merge
    sec_ccm_paths: dict[str, Path] | None = None
    lm2011_industry_enriched_path: Path | None = None

    if RUN_SEC_CCM_PREMERGE:
        join_spec_base = SecCcmJoinSpecV2(
            daily_join_enabled=True,
            daily_join_source="MERGED_DAILY_PANEL",
            daily_permno_col=resolved_permno_col,
            daily_date_col=resolved_date_col,
            daily_feature_columns=tuple(DAILY_FEATURE_COLUMNS),
            required_daily_non_null_features=tuple(REQUIRED_DAILY_NON_NULL_FEATURES),
        )
        join_spec = make_sec_ccm_join_spec_preset("lm2011_filing_date", base=join_spec_base)

        sec_ccm_paths = run_sec_ccm_premerge_pipeline(
            sec_filings_lf=sec_premerge_input_lf,
            link_universe_lf=link_universe_lf,
            trading_calendar_lf=trading_calendar_lf,
            output_dir=SEC_CCM_OUTPUT_DIR,
            daily_lf=ccm_daily_phase_b_lf,
            join_spec=join_spec,
            emit_run_report=True,
        )

        for key in sorted(sec_ccm_paths):
            print(f"{key}: {sec_ccm_paths[key]}")

    if sec_ccm_paths is not None:
        ms = pl.read_parquet(sec_ccm_paths["sec_ccm_match_status"])
        print(
            ms.group_by("match_reason_code")
            .agg(pl.len().alias("n_docs"))
            .sort("n_docs", descending=True)
        )
        total = ms.height
        matched = int(ms.select(pl.col("match_flag").cast(pl.Int64).sum()).item())
        acceptance = int(
            ms.select(pl.col("has_acceptance_datetime").cast(pl.Int64).sum()).item()
        )
        print(
            {
                "total_docs": total,
                "matched_docs": matched,
                "matched_rate": (matched / total) if total else 0.0,
                "acceptance_coverage": (acceptance / total) if total else 0.0,
            }
        )
        print("run_report:", sec_ccm_paths.get("sec_ccm_run_report"))
        print("run_dag_mermaid:", sec_ccm_paths.get("sec_ccm_run_dag_mermaid"))
        print("run_dag_dot:", sec_ccm_paths.get("sec_ccm_run_dag_dot"))

        ff48_siccodes_path = LM2011_FF48_SICCODES_PATH
        company_history_path = CCM_BASE_DIR / "companyhistory.parquet"
        company_description_path = CCM_BASE_DIR / "companydescription.parquet"
        matched_clean_path = Path(sec_ccm_paths["sec_ccm_matched_clean"])
        if matched_clean_path.exists() and company_history_path.exists() and company_description_path.exists() and ff48_siccodes_path.exists():
            lm2011_industry_enriched_path = (
                SEC_CCM_OUTPUT_DIR / "sec_ccm_matched_clean_lm2011_industry.parquet"
            )
            attach_lm2011_industry_classifications(
                pl.scan_parquet(matched_clean_path),
                pl.scan_parquet(company_history_path),
                pl.scan_parquet(company_description_path),
                ff48_siccodes_path=ff48_siccodes_path,
            ).sink_parquet(lm2011_industry_enriched_path, compression="zstd")
            print("lm2011_industry_enriched_path:", lm2011_industry_enriched_path)
        else:
            print(
                {
                    "warning": "skipping LM2011 SIC/FF48 enrichment; required inputs not found",
                    "matched_clean_path": str(matched_clean_path),
                    "company_history_path": str(company_history_path),
                    "company_description_path": str(company_description_path),
                    "ff48_siccodes_path": str(ff48_siccodes_path),
                }
            )

    if RUN_REFINITIV_DOC_OWNERSHIP_LM2011_EXACT_HANDOFF:
        doc_filing_artifact_path = (
            Path(sec_ccm_paths["sec_ccm_matched_clean_filtered"])
            if sec_ccm_paths is not None
            else SEC_CCM_OUTPUT_DIR / "sec_ccm_matched_clean_filtered.parquet"
        )
        authority_decisions_artifact_path = (
            refinitiv_ownership_authority_paths["refinitiv_permno_ownership_authority_decisions_parquet"]
            if refinitiv_ownership_authority_paths is not None
            else REFINITIV_OWNERSHIP_AUTHORITY_DIR / "refinitiv_permno_ownership_authority_decisions.parquet"
        )
        authority_exceptions_artifact_path = (
            refinitiv_ownership_authority_paths["refinitiv_permno_ownership_authority_exceptions_parquet"]
            if refinitiv_ownership_authority_paths is not None
            else REFINITIV_OWNERSHIP_AUTHORITY_DIR / "refinitiv_permno_ownership_authority_exceptions.parquet"
        )
        if (
            doc_filing_artifact_path.exists()
            and authority_decisions_artifact_path.exists()
            and authority_exceptions_artifact_path.exists()
        ):
            if LSEG_API_READY:
                refinitiv_doc_ownership_exact_paths = run_refinitiv_lm2011_doc_ownership_exact_api_pipeline(
                    doc_filing_artifact_path=doc_filing_artifact_path,
                    authority_decisions_artifact_path=authority_decisions_artifact_path,
                    authority_exceptions_artifact_path=authority_exceptions_artifact_path,
                    output_dir=REFINITIV_DOC_OWNERSHIP_LM2011_DIR,
                )
            else:
                refinitiv_doc_ownership_exact_paths = run_refinitiv_lm2011_doc_ownership_exact_handoff_pipeline(
                    doc_filing_artifact_path=doc_filing_artifact_path,
                    authority_decisions_artifact_path=authority_decisions_artifact_path,
                    authority_exceptions_artifact_path=authority_exceptions_artifact_path,
                    output_dir=REFINITIV_DOC_OWNERSHIP_LM2011_DIR,
                )
            _record_refinitiv_stage(
                "refinitiv_doc_ownership_exact",
                refinitiv_doc_ownership_exact_paths,
            )
            for key in sorted(refinitiv_doc_ownership_exact_paths):
                print(f"{key}: {refinitiv_doc_ownership_exact_paths[key]}")
        else:
            print(
                {
                    "warning": "skipping Refinitiv LM2011 doc ownership exact handoff; required artifacts not found",
                    "expected_doc_filing_path": str(doc_filing_artifact_path),
                    "expected_authority_decisions_path": str(authority_decisions_artifact_path),
                    "expected_authority_exceptions_path": str(authority_exceptions_artifact_path),
                }
            )
    if RUN_REFINITIV_DOC_OWNERSHIP_LM2011_FALLBACK_HANDOFF:
        exact_filled_workbook_path = (
            REFINITIV_DOC_OWNERSHIP_LM2011_DIR / "refinitiv_lm2011_doc_ownership_exact_handoff_filled_in.xlsx"
        )
        exact_requests_path = (
            refinitiv_doc_ownership_exact_paths["refinitiv_lm2011_doc_ownership_exact_requests_parquet"]
            if refinitiv_doc_ownership_exact_paths is not None
            else REFINITIV_DOC_OWNERSHIP_LM2011_DIR / "refinitiv_lm2011_doc_ownership_exact_requests.parquet"
        )
        exact_raw_path = (
            refinitiv_doc_ownership_exact_paths["refinitiv_lm2011_doc_ownership_exact_raw_parquet"]
            if refinitiv_doc_ownership_exact_paths is not None
            and "refinitiv_lm2011_doc_ownership_exact_raw_parquet" in refinitiv_doc_ownership_exact_paths
            else REFINITIV_DOC_OWNERSHIP_LM2011_DIR / "refinitiv_lm2011_doc_ownership_exact_raw.parquet"
        )
        if LSEG_API_READY and exact_requests_path.exists() and exact_raw_path.exists():
            refinitiv_doc_ownership_fallback_paths = run_refinitiv_lm2011_doc_ownership_fallback_api_pipeline(
                output_dir=REFINITIV_DOC_OWNERSHIP_LM2011_DIR,
            )
            _record_refinitiv_stage(
                "refinitiv_doc_ownership_fallback",
                refinitiv_doc_ownership_fallback_paths,
            )
            for key in sorted(refinitiv_doc_ownership_fallback_paths):
                print(f"{key}: {refinitiv_doc_ownership_fallback_paths[key]}")
        elif exact_requests_path.exists() and exact_filled_workbook_path.exists():
            refinitiv_doc_ownership_fallback_paths = run_refinitiv_lm2011_doc_ownership_fallback_handoff_pipeline(
                exact_filled_workbook_path=exact_filled_workbook_path,
                output_dir=REFINITIV_DOC_OWNERSHIP_LM2011_DIR,
            )
            _record_refinitiv_stage(
                "refinitiv_doc_ownership_fallback",
                refinitiv_doc_ownership_fallback_paths,
            )
            for key in sorted(refinitiv_doc_ownership_fallback_paths):
                print(f"{key}: {refinitiv_doc_ownership_fallback_paths[key]}")
        else:
            print(
                {
                    "warning": "skipping Refinitiv LM2011 doc ownership fallback handoff; exact requests or filled workbook not found",
                    "expected_exact_requests_path": str(exact_requests_path),
                    "expected_exact_filled_workbook_path": str(exact_filled_workbook_path),
                }
            )
    if RUN_REFINITIV_DOC_OWNERSHIP_LM2011_FINALIZE:
        fallback_requests_path = (
            refinitiv_doc_ownership_fallback_paths["refinitiv_lm2011_doc_ownership_fallback_requests_parquet"]
            if refinitiv_doc_ownership_fallback_paths is not None
            else REFINITIV_DOC_OWNERSHIP_LM2011_DIR / "refinitiv_lm2011_doc_ownership_fallback_requests.parquet"
        )
        exact_raw_path = (
            refinitiv_doc_ownership_fallback_paths["refinitiv_lm2011_doc_ownership_exact_raw_parquet"]
            if refinitiv_doc_ownership_fallback_paths is not None
            else REFINITIV_DOC_OWNERSHIP_LM2011_DIR / "refinitiv_lm2011_doc_ownership_exact_raw.parquet"
        )
        fallback_raw_path = (
            refinitiv_doc_ownership_fallback_paths["refinitiv_lm2011_doc_ownership_fallback_raw_parquet"]
            if refinitiv_doc_ownership_fallback_paths is not None
            and "refinitiv_lm2011_doc_ownership_fallback_raw_parquet" in refinitiv_doc_ownership_fallback_paths
            else REFINITIV_DOC_OWNERSHIP_LM2011_DIR / "refinitiv_lm2011_doc_ownership_fallback_raw.parquet"
        )
        fallback_filled_workbook_path = (
            REFINITIV_DOC_OWNERSHIP_LM2011_DIR / "refinitiv_lm2011_doc_ownership_fallback_handoff_filled_in.xlsx"
        )
        fallback_request_count = _bool_true_count(fallback_requests_path, "retrieval_eligible")
        if fallback_requests_path.exists() and exact_raw_path.exists():
            if fallback_request_count and not fallback_raw_path.exists() and not fallback_filled_workbook_path.exists():
                print(
                    {
                        "warning": "skipping Refinitiv LM2011 doc ownership finalize; fallback requests exist but filled fallback workbook not found",
                        "expected_fallback_filled_workbook_path": str(fallback_filled_workbook_path),
                        "fallback_request_count": fallback_request_count,
                    }
                )
            else:
                refinitiv_doc_ownership_finalize_paths = run_refinitiv_lm2011_doc_ownership_finalize_pipeline(
                    output_dir=REFINITIV_DOC_OWNERSHIP_LM2011_DIR,
                    fallback_filled_workbook_path=(
                        fallback_filled_workbook_path
                        if fallback_request_count and not fallback_raw_path.exists()
                        else None
                    ),
                )
                _record_refinitiv_stage(
                    "refinitiv_doc_ownership_finalize",
                    refinitiv_doc_ownership_finalize_paths,
                )
                for key in sorted(refinitiv_doc_ownership_finalize_paths):
                    print(f"{key}: {refinitiv_doc_ownership_finalize_paths[key]}")
        else:
            print(
                {
                    "warning": "skipping Refinitiv LM2011 doc ownership finalize; fallback requests or exact raw parquet not found",
                    "expected_fallback_requests_path": str(fallback_requests_path),
                    "expected_exact_raw_path": str(exact_raw_path),
                }
            )
    if RUN_REFINITIV_DOC_ANALYST_LM2011_ANCHORS:
        matched_clean_path = (
            Path(sec_ccm_paths["sec_ccm_matched_clean"])
            if sec_ccm_paths is not None
            else SEC_CCM_OUTPUT_DIR / "sec_ccm_matched_clean.parquet"
        )
        quarterly_balance_sheet_path = _resolve_ccm_parquet_artifact(CCM_BASE_DIR, "balancesheetquarterly.parquet")
        quarterly_income_statement_path = _resolve_ccm_parquet_artifact(CCM_BASE_DIR, "incomestatementquarterly.parquet")
        quarterly_period_descriptor_path = _resolve_ccm_parquet_artifact(CCM_BASE_DIR, "perioddescriptorquarterly.parquet")
        filingdates_path = _resolve_ccm_parquet_artifact(CCM_BASE_DIR, "filingdates.parquet")
        if (
            matched_clean_path.exists()
            and quarterly_balance_sheet_path.exists()
            and quarterly_income_statement_path.exists()
            and quarterly_period_descriptor_path.exists()
            and filingdates_path.exists()
        ):
            sample_backbone_lf = build_lm2011_sample_backbone(
                pl.scan_parquet(matched_clean_path),
                pl.scan_parquet(matched_clean_path),
                ccm_filingdates_lf=pl.scan_parquet(filingdates_path),
            )
            quarterly_accounting_panel_lf = build_quarterly_accounting_panel(
                pl.scan_parquet(quarterly_balance_sheet_path),
                pl.scan_parquet(quarterly_income_statement_path),
                pl.scan_parquet(quarterly_period_descriptor_path),
            )
            refinitiv_doc_analyst_anchor_paths = run_refinitiv_lm2011_doc_analyst_anchor_pipeline(
                sample_backbone_lf=sample_backbone_lf,
                quarterly_accounting_panel_lf=quarterly_accounting_panel_lf,
                output_dir=REFINITIV_DOC_ANALYST_LM2011_DIR,
            )
            _record_refinitiv_stage(
                "refinitiv_doc_analyst_anchors",
                refinitiv_doc_analyst_anchor_paths,
            )
            for key in sorted(refinitiv_doc_analyst_anchor_paths):
                print(f"{key}: {refinitiv_doc_analyst_anchor_paths[key]}")
        else:
            print(
                {
                    "warning": "skipping Refinitiv LM2011 doc analyst anchors; required inputs not found",
                    "expected_matched_clean_path": str(matched_clean_path),
                    "expected_quarterly_balance_sheet_path": str(quarterly_balance_sheet_path),
                    "expected_quarterly_income_statement_path": str(quarterly_income_statement_path),
                    "expected_quarterly_period_descriptor_path": str(quarterly_period_descriptor_path),
                    "expected_filingdates_path": str(filingdates_path),
                }
            )
    if RUN_REFINITIV_DOC_ANALYST_LM2011_SELECT:
        doc_analyst_anchors_path = (
            refinitiv_doc_analyst_anchor_paths["refinitiv_doc_analyst_request_anchors_parquet"]
            if refinitiv_doc_analyst_anchor_paths is not None
            else REFINITIV_DOC_ANALYST_LM2011_DIR / "refinitiv_doc_analyst_request_anchors.parquet"
        )
        analyst_normalized_panel_path = (
            REFINITIV_ANALYST_COMMON_STOCK_DIR / "refinitiv_analyst_normalized_panel.parquet"
        )
        if doc_analyst_anchors_path.exists() and analyst_normalized_panel_path.exists():
            refinitiv_doc_analyst_select_paths = run_refinitiv_lm2011_doc_analyst_select_pipeline(
                doc_anchors_artifact_path=doc_analyst_anchors_path,
                analyst_normalized_panel_artifact_path=analyst_normalized_panel_path,
                output_dir=REFINITIV_DOC_ANALYST_LM2011_DIR,
            )
            _record_refinitiv_stage(
                "refinitiv_doc_analyst_select",
                refinitiv_doc_analyst_select_paths,
            )
            for key in sorted(refinitiv_doc_analyst_select_paths):
                print(f"{key}: {refinitiv_doc_analyst_select_paths[key]}")
        else:
            print(
                {
                    "warning": "skipping Refinitiv LM2011 doc analyst select; required artifacts not found",
                    "expected_doc_analyst_anchors_path": str(doc_analyst_anchors_path),
                    "expected_analyst_normalized_panel_path": str(analyst_normalized_panel_path),
                }
            )

    # ## 6) Gated item extraction
    analysis_item_paths: list[Path] = []
    diagnostic_item_paths: list[Path] = []

    if RUN_GATED_ITEM_EXTRACTION:
        if sec_ccm_paths is None:
            raise RuntimeError("Run SEC-CCM pre-merge first.")

        analysis_item_paths = process_year_dir_extract_items_gated(
            year_dir=SEC_YEAR_MERGED_DIR,
            out_dir=SEC_ITEMS_ANALYSIS_DIR,
            doc_id_allowlist=sec_ccm_paths["sec_ccm_analysis_doc_ids"],
            years=[str(year) for year in YEARS],
            parquet_batch_rows=16,
            out_batch_max_rows=50_000,
            out_batch_max_text_bytes=250 * 1024 * 1024,
            tmp_dir=LOCAL_TMP,
            compression="zstd",
            local_work_dir=LOCAL_ITEM_WORK,
            non_item_diagnostic=False,
            include_full_text=False,
            regime=True,
            extraction_regime=ITEM_EXTRACTION_REGIME,
        )
        print({"analysis_year_files": len(analysis_item_paths)})

        if RUN_UNMATCHED_DIAGNOSTIC_TRACK:
            diagnostic_item_paths = process_year_dir_extract_items_gated(
                year_dir=SEC_YEAR_MERGED_DIR,
                out_dir=SEC_ITEMS_DIAGNOSTIC_DIR,
                doc_id_allowlist=sec_ccm_paths["sec_ccm_diagnostic_doc_ids"],
                years=[str(year) for year in YEARS],
                parquet_batch_rows=16,
                out_batch_max_rows=50_000,
                out_batch_max_text_bytes=250 * 1024 * 1024,
                tmp_dir=LOCAL_TMP,
                compression="zstd",
                local_work_dir=LOCAL_ITEM_WORK,
                non_item_diagnostic=False,
                include_full_text=False,
                regime=True,
                extraction_regime=ITEM_EXTRACTION_REGIME,
            )
            print({"diagnostic_year_files": len(diagnostic_item_paths)})

    # ## 7) No-item diagnostics + boundary diagnostics
    analysis_no_item: list[tuple[str, Path, Path]] = []
    if RUN_NO_ITEM_DIAGNOSTICS and RUN_GATED_ITEM_EXTRACTION:
        out_dir = SEC_NO_ITEM_DIR / "analysis"
        out_dir.mkdir(parents=True, exist_ok=True)
        for item_path in analysis_item_paths:
            year = item_path.stem
            filing_path = SEC_YEAR_MERGED_DIR / f"{year}.parquet"
            if not filing_path.exists():
                continue
            out_no_item = out_dir / f"{year}_no_item_filings.parquet"
            out_stats = out_dir / f"{year}_no_item_stats.csv"
            compute_no_item_diagnostics(
                filing_path,
                item_path,
                out_no_item,
                out_stats,
                include_full_text=False,
            )
            analysis_no_item.append((year, out_no_item, out_stats))
    print({"analysis_no_item_years": len(analysis_no_item)})

    boundary_results = None
    if RUN_BOUNDARY_DIAGNOSTICS:
        if sec_ccm_paths is None:
            raise RuntimeError("Run SEC-CCM pre-merge first.")

        allow_lf = (
            pl.scan_parquet(sec_ccm_paths["sec_ccm_analysis_doc_ids"])
            .select(pl.col("doc_id").cast(pl.Utf8))
            .drop_nulls(subset=["doc_id"])
            .unique(subset=["doc_id"])
        )
        staged = 0
        for year in YEARS:
            src = SEC_YEAR_MERGED_DIR / f"{year}.parquet"
            if not src.exists():
                continue
            dst = BOUNDARY_INPUT_DIR / src.name
            pl.scan_parquet(src).join(allow_lf, on="doc_id", how="semi").sink_parquet(
                dst,
                compression="zstd",
            )
            staged += 1

        diag_config = DiagnosticsConfig(
            parquet_dir=BOUNDARY_INPUT_DIR,
            out_path=BOUNDARY_OUT_DIR / "suspicious_boundaries_matched.csv",
            report_path=BOUNDARY_OUT_DIR / "suspicious_boundaries_matched_report.txt",
            samples_dir=BOUNDARY_OUT_DIR / "samples",
            batch_size=8,
            max_files=0,
            max_examples=50,
            emit_manifest=True,
            manifest_items_path=BOUNDARY_OUT_DIR / "manifest_items.csv",
            manifest_filings_path=BOUNDARY_OUT_DIR / "manifest_filings.csv",
            sample_pass=100,
            sample_seed=42,
            sample_filings_path=BOUNDARY_OUT_DIR / "sample_filings.csv",
            sample_items_path=BOUNDARY_OUT_DIR / "sample_items.csv",
            emit_html=True,
            html_out=BOUNDARY_OUT_DIR / "html",
            html_scope="sample",
            extraction_regime="v2",
            diagnostics_regime="v2",
            target_set="cohen2020_common",
            focus_items=parse_focus_items(None),
            report_item_scope="target",
        )
        print({"boundary_staged_year_files": staged})
        boundary_results = run_boundary_diagnostics(diag_config)
        print(boundary_results)

    # ## 8) Validation + artifact index
    validation_rows: list[dict[str, object]] = []

    if RUN_VALIDATION_CHECKS and sec_ccm_paths is not None:
        pre = (
            sec_premerge_input_lf.select(
                pl.len().alias("rows"),
                pl.col("doc_id").n_unique().alias("uniq"),
            )
            .collect()
            .row(0, named=True)
        )
        ms_lf = pl.scan_parquet(sec_ccm_paths["sec_ccm_match_status"])
        ms = (
            ms_lf.select(pl.len().alias("rows"), pl.col("doc_id").n_unique().alias("uniq"))
            .collect()
            .row(0, named=True)
        )
        if pre["rows"] != ms["rows"]:
            raise AssertionError(
                f"premerge rows {pre['rows']} != match_status rows {ms['rows']}"
            )
        if ms["rows"] != ms["uniq"]:
            raise AssertionError("sec_ccm_match_status is not unique on doc_id")

        schema = pl.scan_parquet(sec_ccm_paths["final_flagged_data"]).collect_schema()
        if schema.get("kypermno") != pl.Int32:
            raise AssertionError(f"kypermno dtype not Int32: {schema.get('kypermno')}")
        if schema.get("data_status") != pl.UInt64:
            raise AssertionError(
                f"data_status dtype not UInt64: {schema.get('data_status')}"
            )
        null_status = (
            pl.scan_parquet(sec_ccm_paths["final_flagged_data"])
            .select(pl.col("data_status").is_null().sum())
            .collect()
            .item()
        )
        if null_status != 0:
            raise AssertionError(f"data_status null count: {null_status}")

        validation_rows.append(
            {
                "check": "premerge_vs_match_status_rows",
                "ok": True,
                "details": f"rows={pre['rows']}",
            }
        )
        validation_rows.append(
            {
                "check": "match_status_doc_id_unique",
                "ok": True,
                "details": f"unique={ms['uniq']}",
            }
        )

    if (
        RUN_VALIDATION_CHECKS
        and RUN_GATED_ITEM_EXTRACTION
        and analysis_item_paths
        and sec_ccm_paths is not None
    ):
        allow_lf = (
            pl.scan_parquet(sec_ccm_paths["sec_ccm_analysis_doc_ids"])
            .select(pl.col("doc_id").cast(pl.Utf8))
            .drop_nulls(subset=["doc_id"])
            .unique(subset=["doc_id"])
        )
        extracted_lf = (
            pl.scan_parquet([str(path) for path in analysis_item_paths])
            .select(pl.col("doc_id").cast(pl.Utf8))
            .drop_nulls(subset=["doc_id"])
            .unique(subset=["doc_id"])
        )
        outside = extracted_lf.join(allow_lf, on="doc_id", how="anti").select(
            pl.len()
        ).collect().item()
        if outside != 0:
            raise AssertionError(f"Extracted doc_ids outside analysis allowlist: {outside}")
        validation_rows.append(
            {
                "check": "analysis_items_subset_allowlist",
                "ok": True,
                "details": "outside=0",
            }
        )

    _print_rows_table(validation_rows, empty_message="No validations executed")

    artifact_rows: list[dict[str, object]] = []

    def _add(stage: str, key: str, path: Path) -> None:
        artifact_rows.append(
            {
                "stage": stage,
                "artifact": key,
                "path": str(path),
                "exists": path.exists(),
                "rows": _row_count(path),
            }
        )

    if ccm_daily_path is not None:
        _add("ccm", "ccm_daily_path", ccm_daily_path)
    if ccm_daily_market_core_path is not None:
        _add("ccm", "ccm_daily_market_core_path", ccm_daily_market_core_path)
    if ccm_daily_phase_b_surface_path is not None:
        _add("ccm", "ccm_daily_phase_b_surface_path", ccm_daily_phase_b_surface_path)
    if ccm_daily_bridge_surface_path is not None:
        _add("ccm", "ccm_daily_bridge_surface_path", ccm_daily_bridge_surface_path)
    for stage in refinitiv_artifact_stages:
        for key in sorted(stage.paths):
            _add(stage.name, key, stage.paths[key])
    if sec_ccm_paths is not None:
        for key in sorted(sec_ccm_paths):
            _add("sec_ccm", key, Path(sec_ccm_paths[key]))
    if lm2011_industry_enriched_path is not None:
        _add("lm2011", "sec_ccm_matched_clean_lm2011_industry", lm2011_industry_enriched_path)
    for path in analysis_item_paths:
        _add("items_analysis", path.stem, path)
    for path in diagnostic_item_paths:
        _add("items_diagnostic", path.stem, path)
    for year, no_item_path, csv_path in analysis_no_item:
        _add("no_item_analysis", f"{year}_no_item_filings", no_item_path)
        _add("no_item_analysis", f"{year}_no_item_stats", csv_path)
    for key, path in {
        "boundary_csv": BOUNDARY_OUT_DIR / "suspicious_boundaries_matched.csv",
        "boundary_report": BOUNDARY_OUT_DIR / "suspicious_boundaries_matched_report.txt",
        "boundary_manifest_items": BOUNDARY_OUT_DIR / "manifest_items.csv",
        "boundary_manifest_filings": BOUNDARY_OUT_DIR / "manifest_filings.csv",
        "boundary_html": BOUNDARY_OUT_DIR / "html",
    }.items():
        _add("boundary", key, path)

    _print_rows_table(artifact_rows, sort_by=["stage", "artifact"], empty_message="No artifacts indexed")

    if refinitiv_artifact_stages:
        refinitiv_manifest_path = REFINITIV_STEP1_OUT_DIR / "refinitiv_step1_manifest.json"
        refinitiv_manifest_payload = {
            "pipeline_name": "refinitiv_step1",
            "artifact_version": "v2",
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "manual_inputs": {
                "filled_extended_lookup_workbook": str(
                    REFINITIV_STEP1_OUT_DIR / "refinitiv_ric_lookup_handoff_common_stock_extended_filled_in.xlsx"
                ),
                "filled_ownership_universe_workbook": str(
                    REFINITIV_OWNERSHIP_UNIVERSE_DIR
                    / "refinitiv_ownership_universe_handoff_common_stock_filled_in.xlsx"
                ),
                "reviewed_ticker_allowlist_parquet": str(
                    REFINITIV_OWNERSHIP_AUTHORITY_DIR / "refinitiv_permno_ownership_ticker_allowlist.parquet"
                ),
                "analyst_normalized_panel_parquet": str(
                    REFINITIV_ANALYST_COMMON_STOCK_DIR / "refinitiv_analyst_normalized_panel.parquet"
                ),
                "filled_doc_ownership_exact_workbook": str(
                    REFINITIV_DOC_OWNERSHIP_LM2011_DIR / "refinitiv_lm2011_doc_ownership_exact_handoff_filled_in.xlsx"
                ),
                "filled_doc_ownership_fallback_workbook": str(
                    REFINITIV_DOC_OWNERSHIP_LM2011_DIR
                    / "refinitiv_lm2011_doc_ownership_fallback_handoff_filled_in.xlsx"
                ),
            },
            "counts": {
                "bridge_rows": _row_count(REFINITIV_STEP1_OUT_DIR / "refinitiv_bridge_universe.parquet"),
                "resolution_rows": _row_count(
                    REFINITIV_STEP1_OUT_DIR / "refinitiv_ric_resolution_common_stock.parquet"
                ),
                "instrument_authority_rows": _row_count(
                    REFINITIV_STEP1_OUT_DIR / "refinitiv_instrument_authority_common_stock.parquet"
                ),
                "ownership_universe_handoff_rows": _row_count(
                    REFINITIV_OWNERSHIP_UNIVERSE_DIR
                    / "refinitiv_ownership_universe_handoff_common_stock.parquet"
                ),
                "ownership_universe_results_rows": _row_count(
                    REFINITIV_OWNERSHIP_UNIVERSE_DIR
                    / "refinitiv_ownership_universe_results.parquet"
                ),
                "ownership_universe_row_summary_rows": _row_count(
                    REFINITIV_OWNERSHIP_UNIVERSE_DIR
                    / "refinitiv_ownership_universe_row_summary.parquet"
                ),
                "ownership_universe_requests_with_data": _ownership_rows_with_data(
                    REFINITIV_OWNERSHIP_UNIVERSE_DIR
                    / "refinitiv_ownership_universe_row_summary.parquet"
                ),
                "ownership_authority_decision_rows": _row_count(
                    REFINITIV_OWNERSHIP_AUTHORITY_DIR
                    / "refinitiv_permno_ownership_authority_decisions.parquet"
                ),
                "ownership_authority_exception_rows": _row_count(
                    REFINITIV_OWNERSHIP_AUTHORITY_DIR
                    / "refinitiv_permno_ownership_authority_exceptions.parquet"
                ),
                "ownership_authority_review_rows": _row_count(
                    REFINITIV_OWNERSHIP_AUTHORITY_DIR
                    / "refinitiv_permno_ownership_review_required.parquet"
                ),
                "ownership_authority_panel_rows": _row_count(
                    REFINITIV_OWNERSHIP_AUTHORITY_DIR
                    / "refinitiv_permno_date_ownership_panel.parquet"
                ),
                "analyst_request_group_membership_rows": _row_count(
                    REFINITIV_ANALYST_COMMON_STOCK_DIR
                    / "refinitiv_analyst_request_group_membership_common_stock.parquet"
                ),
                "analyst_request_group_rows": _row_count(
                    REFINITIV_ANALYST_COMMON_STOCK_DIR
                    / "refinitiv_analyst_request_universe_common_stock.parquet"
                ),
                "analyst_actuals_raw_rows": _row_count(
                    REFINITIV_ANALYST_COMMON_STOCK_DIR / "refinitiv_analyst_actuals_raw.parquet"
                ),
                "analyst_estimates_monthly_raw_rows": _row_count(
                    REFINITIV_ANALYST_COMMON_STOCK_DIR / "refinitiv_analyst_estimates_monthly_raw.parquet"
                ),
                "analyst_normalized_panel_rows": _row_count(
                    REFINITIV_ANALYST_COMMON_STOCK_DIR / "refinitiv_analyst_normalized_panel.parquet"
                ),
                "analyst_normalization_rejection_rows": _row_count(
                    REFINITIV_ANALYST_COMMON_STOCK_DIR
                    / "refinitiv_analyst_normalization_rejections.parquet"
                ),
                "doc_ownership_exact_request_rows": _row_count(
                    REFINITIV_DOC_OWNERSHIP_LM2011_DIR
                    / "refinitiv_lm2011_doc_ownership_exact_requests.parquet"
                ),
                "doc_ownership_exact_raw_rows": _row_count(
                    REFINITIV_DOC_OWNERSHIP_LM2011_DIR / "refinitiv_lm2011_doc_ownership_exact_raw.parquet"
                ),
                "doc_ownership_fallback_request_rows": _row_count(
                    REFINITIV_DOC_OWNERSHIP_LM2011_DIR
                    / "refinitiv_lm2011_doc_ownership_fallback_requests.parquet"
                ),
                "doc_ownership_fallback_request_eligible_rows": _bool_true_count(
                    REFINITIV_DOC_OWNERSHIP_LM2011_DIR
                    / "refinitiv_lm2011_doc_ownership_fallback_requests.parquet",
                    "retrieval_eligible",
                ),
                "doc_ownership_fallback_raw_rows": _row_count(
                    REFINITIV_DOC_OWNERSHIP_LM2011_DIR / "refinitiv_lm2011_doc_ownership_fallback_raw.parquet"
                ),
                "doc_ownership_raw_rows": _row_count(
                    REFINITIV_DOC_OWNERSHIP_LM2011_DIR / "refinitiv_lm2011_doc_ownership_raw.parquet"
                ),
                "doc_ownership_final_rows": _row_count(
                    REFINITIV_DOC_OWNERSHIP_LM2011_DIR / "refinitiv_lm2011_doc_ownership.parquet"
                ),
                "doc_ownership_final_nonnull_rows": _nonnull_count(
                    REFINITIV_DOC_OWNERSHIP_LM2011_DIR / "refinitiv_lm2011_doc_ownership.parquet",
                    "institutional_ownership_pct",
                ),
                "doc_ownership_retrieval_status_counts": _value_counts(
                    REFINITIV_DOC_OWNERSHIP_LM2011_DIR / "refinitiv_lm2011_doc_ownership.parquet",
                    "retrieval_status",
                ),
                "doc_analyst_anchor_rows": _row_count(
                    REFINITIV_DOC_ANALYST_LM2011_DIR / "refinitiv_doc_analyst_request_anchors.parquet"
                ),
                "doc_analyst_selected_rows": _row_count(
                    REFINITIV_DOC_ANALYST_LM2011_DIR / "refinitiv_doc_analyst_selected.parquet"
                ),
                "doc_analyst_match_status_counts": _value_counts(
                    REFINITIV_DOC_ANALYST_LM2011_DIR / "refinitiv_doc_analyst_selected.parquet",
                    "analyst_match_status",
                ),
            },
            "artifacts": {
                key: str(path)
                for stage in refinitiv_artifact_stages
                for key, path in sorted(stage.paths.items())
            },
        }
        refinitiv_manifest_path.write_text(
            json.dumps(refinitiv_manifest_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

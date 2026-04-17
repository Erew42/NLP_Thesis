from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from collections.abc import Callable
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
    RAW_ITEM_TEXT_CLEANING_POLICY_ID,
    build_lm2011_text_features_full_10k,
    build_lm2011_text_features_mda,
    write_lm2011_text_features_full_10k_parquet,
    write_lm2011_text_features_mda_parquet,
)
from thesis_pkg.pipelines.lm2011_pipeline import (
    build_lm2011_event_panel,
    build_lm2011_sue_panel,
    build_lm2011_table_i_sample_creation,
    build_lm2011_trading_strategy_monthly_returns,
)
from thesis_pkg.pipelines import lm2011_pipeline
from thesis_pkg.pipelines.lm2011_regressions import (
    build_lm2011_return_regression_panel,
    build_lm2011_sue_regression_panel,
    build_lm2011_table_ia_i_results,
    build_lm2011_table_ia_ii_results,
    build_lm2011_table_iv_results,
    build_lm2011_table_v_results,
    build_lm2011_table_vi_results,
    build_lm2011_table_viii_results,
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
SKIPPED_MISSING_SEEDED_UPSTREAM = "skipped_missing_seeded_upstream"
SKIPPED_MISSING_OPTIONAL_INPUT = "skipped_missing_optional_input"
DEFAULT_LM2011_FULL_10K_CLEANING_CONTRACT = "lm2011_paper"
DEFAULT_LM2011_TEXT_FEATURE_BATCH_SIZE = 10
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
    "table_v_results": "lm2011_table_v_results.parquet",
    "table_vi_results": "lm2011_table_vi_results.parquet",
    "table_viii_results": "lm2011_table_viii_results.parquet",
    "table_ia_i_results": "lm2011_table_ia_i_results.parquet",
    "trading_strategy_monthly_returns": "lm2011_trading_strategy_monthly_returns.parquet",
    "table_ia_ii_results": "lm2011_table_ia_ii_results.parquet",
}
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


@dataclass(frozen=True)
class RunnerPaths:
    sample_root: Path
    upstream_run_root: Path
    additional_data_dir: Path
    output_dir: Path
    year_merged_dir: Path
    sample_backbone_path: Path | None
    daily_panel_path: Path
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
    text_feature_batch_size: int
    event_window_doc_batch_size: int
    print_ram_stats: bool
    ram_log_interval_batches: int


@dataclass(frozen=True)
class LM2011PostRefinitivRunConfig:
    paths: RunnerPaths
    enabled_stages: tuple[str, ...] = LM2011_ALL_STAGE_NAMES
    fail_closed_for_enabled_stages: bool = False

    def __post_init__(self) -> None:
        unknown = sorted(set(self.enabled_stages) - set(LM2011_ALL_STAGE_NAMES))
        if unknown:
            raise ValueError(f"Unknown LM2011 stage names: {unknown}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the sample-only post-Refinitiv LM2011 replication pipeline.")
    parser.add_argument("--sample-root", type=Path, default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument("--upstream-run-root", type=Path, default=DEFAULT_UPSTREAM_RUN_ROOT)
    parser.add_argument("--additional-data-dir", type=Path, default=DEFAULT_ADDITIONAL_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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
    parser.add_argument("--items-analysis-dir", type=Path, default=None)
    parser.add_argument("--ccm-base-dir", type=Path, default=None)
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
        "--text-feature-batch-size",
        type=int,
        default=DEFAULT_LM2011_TEXT_FEATURE_BATCH_SIZE,
        help="Number of rows per batch when scoring LM2011 full-text features.",
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


def _ram_snapshot(label: str) -> dict[str, object]:
    payload: dict[str, object] = {"label": label}
    meminfo = _read_proc_kb_map(Path("/proc/meminfo"))
    status = _read_proc_kb_map(Path("/proc/self/status"))
    if not meminfo and not status:
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
    return payload


def _print_ram_snapshot(label: str, *, enabled: bool) -> None:
    if not enabled:
        return
    print(_ram_snapshot(label))


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


def _resolve_paths(args: argparse.Namespace) -> RunnerPaths:
    sample_root = Path(args.sample_root).resolve()
    upstream_run_root = Path(args.upstream_run_root).resolve()
    additional_data_dir = Path(args.additional_data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if int(args.text_feature_batch_size) < 1:
        raise ValueError("--text-feature-batch-size must be >= 1")
    if int(args.event_window_doc_batch_size) < 1:
        raise ValueError("--event-window-doc-batch-size must be >= 1")
    if int(args.ram_log_interval_batches) < 1:
        raise ValueError("--ram-log-interval-batches must be >= 1")
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
        year_merged_dir=year_merged_dir,
        sample_backbone_path=sample_backbone_path,
        daily_panel_path=daily_panel_path,
        ccm_base_dir=ccm_base_dir,
        matched_clean_path=matched_clean_path,
        items_analysis_dir=items_analysis_dir,
        doc_ownership_path=(
            upstream_run_root
            / "refinitiv_doc_ownership_lm2011"
            / "refinitiv_lm2011_doc_ownership.parquet"
        ),
        doc_analyst_selected_path=(
            upstream_run_root
            / "refinitiv_doc_analyst_lm2011"
            / "refinitiv_doc_analyst_selected.parquet"
        ),
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
        text_feature_batch_size=int(args.text_feature_batch_size),
        event_window_doc_batch_size=int(args.event_window_doc_batch_size),
        print_ram_stats=bool(args.print_ram_stats),
        ram_log_interval_batches=int(args.ram_log_interval_batches),
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
) -> Callable[[dict[str, int]], None]:
    def _logger(progress: dict[str, int]) -> None:
        batch_index = int(progress["batch_index"])
        if not _should_log_periodic_batch(
            batch_index=batch_index,
            total_batches=None,
            interval=ram_log_interval_batches,
        ):
            return
        payload: dict[str, object] = {
            "stage": stage_name,
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
        },
        "config": {
            "full_10k_cleaning_contract": paths.full_10k_cleaning_contract,
            "raw_mda_cleaning_policy_id": RAW_ITEM_TEXT_CLEANING_POLICY_ID,
            "text_feature_batch_size": paths.text_feature_batch_size,
            "event_window_doc_batch_size": paths.event_window_doc_batch_size,
            "print_ram_stats": paths.print_ram_stats,
            "ram_log_interval_batches": paths.ram_log_interval_batches,
        },
        "resolved_inputs": {
            "year_merged_dir": _absolute_path_str(paths.year_merged_dir),
            "sample_backbone_path": _absolute_path_str(paths.sample_backbone_path),
            "daily_panel_path": _absolute_path_str(paths.daily_panel_path),
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
    manifest = _build_manifest(paths)
    manifest_path = paths.output_dir / MANIFEST_FILENAME
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
        if _stage_disabled(run_cfg, manifest, manifest_path, "text_features_full_10k"):
            pass
        elif text_year_merged_lf is not None:
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
                    batch_size=paths.text_feature_batch_size,
                    progress_callback=_make_text_feature_progress_logger(
                        "text_features_full_10k",
                        print_ram_stats=paths.print_ram_stats,
                        ram_log_interval_batches=paths.ram_log_interval_batches,
                    ),
                ),
                warnings=text_stage_warnings,
                print_ram_stats=paths.print_ram_stats,
            )
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
        if _stage_disabled(run_cfg, manifest, manifest_path, "text_features_mda"):
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
                    batch_size=paths.text_feature_batch_size,
                    progress_callback=_make_text_feature_progress_logger(
                        "text_features_mda",
                        print_ram_stats=paths.print_ram_stats,
                        ram_log_interval_batches=paths.ram_log_interval_batches,
                    ),
                ),
                warnings=text_stage_warnings,
                print_ram_stats=paths.print_ram_stats,
            )
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

        current_stage_name = "sue_panel"
        if _stage_disabled(run_cfg, manifest, manifest_path, "sue_panel"):
            pass
        elif (
            event_panel_lf is not None
            and quarterly_accounting_panel_lf is not None
            and paths.doc_analyst_selected_path.exists()
            and paths.daily_panel_path.exists()
        ):
            sue_panel_lf = _write_stage(
                manifest,
                manifest_path=manifest_path,
                output_dir=paths.output_dir,
                stage_name="sue_panel",
                frame=build_lm2011_sue_panel(
                    event_panel_lf=event_panel_lf,
                    quarterly_accounting_panel_lf=quarterly_accounting_panel_lf,
                    ibes_unadjusted_earnings_lf=_prepare_doc_analyst_sue_input_lf(paths.doc_analyst_selected_path),
                    daily_lf=pl.scan_parquet(paths.daily_panel_path),
                ),
            )
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
                lambda: build_lm2011_table_iv_results(
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
                lambda: build_lm2011_table_v_results(
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
                lambda: build_lm2011_table_vi_results(
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
                lambda: build_lm2011_table_viii_results(
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
                lambda: build_lm2011_table_ia_i_results(
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
            if should_run:
                _write_stage(
                    manifest,
                    manifest_path=manifest_path,
                    output_dir=paths.output_dir,
                    stage_name=stage_name,
                    frame=builder(),
                    empty_reason=EMPTY_TABLE_REASON,
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


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run_lm2011_post_refinitiv_pipeline(
        LM2011PostRefinitivRunConfig(paths=_resolve_paths(args))
    )


if __name__ == "__main__":
    raise SystemExit(main())

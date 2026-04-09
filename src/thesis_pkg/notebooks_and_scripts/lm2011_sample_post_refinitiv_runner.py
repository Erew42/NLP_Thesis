from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT_ENV_VAR = "NLP_THESIS_REPO_ROOT"


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
from thesis_pkg.core.sec.lm2011_text import (
    build_lm2011_text_features_full_10k,
    build_lm2011_text_features_mda,
)
from thesis_pkg.pipelines.lm2011_pipeline import (
    build_lm2011_event_panel,
    build_lm2011_sue_panel,
    build_lm2011_table_i_sample_creation,
    build_lm2011_trading_strategy_monthly_returns,
)
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


DEFAULT_SAMPLE_ROOT = ROOT / "full_data_run" / "sample_5pct_seed42"
DEFAULT_UPSTREAM_RUN_ROOT = DEFAULT_SAMPLE_ROOT / "results" / "sec_ccm_unified_runner" / "local_sample"
DEFAULT_ADDITIONAL_DATA_DIR = ROOT / "full_data_run" / "LM2011_additional_data"
DEFAULT_OUTPUT_DIR = DEFAULT_SAMPLE_ROOT / "results" / "lm2011_sample_post_refinitiv_runner"

PARQUET_COMPRESSION = "zstd"
YEAR_MERGED_GLOB = "*.parquet"
ITEMS_ANALYSIS_GLOB = "*.parquet"

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
MONTHLY_STOCK_CANDIDATES: tuple[str, ...] = (
    "stockmonthly.parquet",
    "securitymonthly.parquet",
    "monthlystock.parquet",
)
EMPTY_TABLE_REASON = "insufficient_sample_size_for_estimable_quarterly_fama_macbeth"
SKIPPED_MISSING_SEEDED_UPSTREAM = "skipped_missing_seeded_upstream"
SKIPPED_MISSING_OPTIONAL_INPUT = "skipped_missing_optional_input"

STAGE_ARTIFACT_FILENAMES: dict[str, str] = {
    "sample_backbone": "lm2011_sample_backbone.parquet",
    "annual_accounting_panel": "lm2011_annual_accounting_panel.parquet",
    "quarterly_accounting_panel": "lm2011_quarterly_accounting_panel.parquet",
    "ff_factors_daily_normalized": "lm2011_ff_factors_daily_normalized.parquet",
    "text_features_full_10k": "lm2011_text_features_full_10k.parquet",
    "text_features_mda": "lm2011_text_features_mda.parquet",
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


@dataclass(frozen=True)
class RunnerPaths:
    sample_root: Path
    upstream_run_root: Path
    additional_data_dir: Path
    output_dir: Path
    year_merged_dir: Path
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
    ff48_siccodes_path: Path
    monthly_stock_path: Path | None
    ff_monthly_with_mom_path: Path | None
    full_10k_cleaning_contract: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the sample-only post-Refinitiv LM2011 replication pipeline.")
    parser.add_argument("--sample-root", type=Path, default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument("--upstream-run-root", type=Path, default=DEFAULT_UPSTREAM_RUN_ROOT)
    parser.add_argument("--additional-data-dir", type=Path, default=DEFAULT_ADDITIONAL_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--year-merged-dir", type=Path, default=None)
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
        default="current",
        choices=FULL_10K_CLEANING_CONTRACTS,
        help="Full-10-K cleaning contract for paper-faithful sample reruns.",
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


def _resolve_paths(args: argparse.Namespace) -> RunnerPaths:
    sample_root = Path(args.sample_root).resolve()
    upstream_run_root = Path(args.upstream_run_root).resolve()
    additional_data_dir = Path(args.additional_data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    year_merged_dir = (
        Path(args.year_merged_dir).resolve()
        if args.year_merged_dir is not None
        else sample_root / "year_merged"
    )
    daily_panel_path = (
        Path(args.daily_panel_path).resolve()
        if args.daily_panel_path is not None
        else (
            _resolve_optional_existing_path(
                sample_root / "derived_data" / "final_flagged_data_compdesc_added.sample_5pct_seed42.parquet",
                sample_root / "derived_data" / "final_flagged_data_compdesc_added.parquet",
            )
            or sample_root / "derived_data" / "final_flagged_data_compdesc_added.sample_5pct_seed42.parquet"
        )
    )
    ccm_base_dir = Path(args.ccm_base_dir).resolve() if args.ccm_base_dir is not None else sample_root / "ccm_parquet_data"
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
        ff48_siccodes_path=additional_data_dir / "FF_Siccodes_48_Industries.txt",
        monthly_stock_path=monthly_stock_path,
        ff_monthly_with_mom_path=ff_monthly_with_mom_path,
        full_10k_cleaning_contract=str(args.full_10k_cleaning_contract),
    )


def _load_word_list(path: Path) -> tuple[str, ...]:
    words: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            token = line.strip()
            if token:
                words.append(token.casefold())
    return tuple(words)


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
    txt_path = additional_data_dir / "LM2011_MasterDictionary.txt"
    csv_path = additional_data_dir / "Loughran-McDonald_MasterDictionary_1993-2024.csv"
    if txt_path.exists():
        dictionary_path = txt_path
    elif csv_path.exists():
        dictionary_path = csv_path
    else:
        raise FileNotFoundError(f"No LM master dictionary file found in {additional_data_dir}")
    words_df = pl.read_csv(dictionary_path).select(pl.col("Word").cast(pl.Utf8, strict=False).alias("Word"))
    return tuple(
        word.strip()
        for word in words_df.get_column("Word").drop_nulls().to_list()
        if isinstance(word, str) and word.strip()
    )


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


def _load_ff_factors_daily_lf(csv_path: Path) -> pl.LazyFrame:
    return (
        pl.scan_csv(
            csv_path,
            skip_rows=5,
            has_header=False,
            new_columns=list(FF_DAILY_COLUMNS),
            schema_overrides={name: pl.Utf8 for name in FF_DAILY_COLUMNS},
        )
        .filter(pl.col("raw_date").str.len_chars() == 8)
        .with_columns(
            pl.col("raw_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("trading_date"),
            pl.col("mkt_rf").str.strip_chars().cast(pl.Float64, strict=False).alias("mkt_rf"),
            pl.col("smb").str.strip_chars().cast(pl.Float64, strict=False).alias("smb"),
            pl.col("hml").str.strip_chars().cast(pl.Float64, strict=False).alias("hml"),
            pl.col("rf").str.strip_chars().cast(pl.Float64, strict=False).alias("rf"),
        )
        .drop("raw_date")
        .drop_nulls(subset=["trading_date", "mkt_rf", "smb", "hml", "rf"])
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
    df = _collect_frame(frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path, compression=PARQUET_COMPRESSION)
    return output_path, int(df.height)


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
        output_dir=output_dir,
        stage_name=stage_name,
        frame=table_df,
        extra_artifacts={"csv": csv_path, "markdown": markdown_path},
        warnings=table_i_warnings,
    )


def _build_manifest(paths: RunnerPaths) -> dict[str, Any]:
    return {
        "runner_name": "lm2011_sample_post_refinitiv_runner",
        "generated_at_utc": _utc_timestamp(),
        "run_status": "running",
        "roots": {
            "sample_root": _absolute_path_str(paths.sample_root),
            "upstream_run_root": _absolute_path_str(paths.upstream_run_root),
            "additional_data_dir": _absolute_path_str(paths.additional_data_dir),
            "output_dir": _absolute_path_str(paths.output_dir),
        },
        "config": {
            "full_10k_cleaning_contract": paths.full_10k_cleaning_contract,
        },
        "resolved_inputs": {
            "year_merged_dir": _absolute_path_str(paths.year_merged_dir),
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
            "ff48_siccodes_path": _absolute_path_str(paths.ff48_siccodes_path),
            "monthly_stock_path": _absolute_path_str(paths.monthly_stock_path),
            "ff_monthly_with_mom_path": _absolute_path_str(paths.ff_monthly_with_mom_path),
        },
        "artifacts": {},
        "row_counts": {},
        "stages": {},
    }


def _record_stage_success(
    manifest: dict[str, Any],
    *,
    stage_name: str,
    artifact_path: Path,
    row_count: int,
    empty_reason: str | None = None,
    extra_artifacts: dict[str, Path] | None = None,
    warnings: Sequence[str] | None = None,
) -> None:
    status = "generated_empty" if row_count == 0 else "generated"
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
        "warnings": list(warnings or []),
    }
    print({"stage": stage_name, "status": status, "row_count": row_count, "artifact_path": str(artifact_path)})


def _record_stage_skipped(
    manifest: dict[str, Any],
    *,
    stage_name: str,
    reason: str,
    detail: str,
) -> None:
    artifact_path = _artifact_output_path(Path(manifest["roots"]["output_dir"]), stage_name)
    manifest["stages"][stage_name] = {
        "status": reason,
        "artifact_path": _absolute_path_str(artifact_path),
        "row_count": None,
        "reason": detail,
    }
    print({"stage": stage_name, "status": reason, "reason": detail})


def _record_stage_failed(
    manifest: dict[str, Any],
    *,
    stage_name: str,
    exc: Exception,
) -> None:
    output_dir = Path(manifest["roots"]["output_dir"])
    manifest["stages"][stage_name] = {
        "status": "failed",
        "artifact_path": _absolute_path_str(_artifact_output_path(output_dir, stage_name))
        if stage_name in STAGE_ARTIFACT_FILENAMES
        else None,
        "row_count": None,
        "reason": f"{type(exc).__name__}: {exc}",
    }


def _write_stage(
    manifest: dict[str, Any],
    *,
    output_dir: Path,
    stage_name: str,
    frame: pl.LazyFrame | pl.DataFrame,
    empty_reason: str | None = None,
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
        extra_artifacts=extra_artifacts,
        warnings=warnings,
    )
    return pl.scan_parquet(written_path)


def _missing_required_paths(*paths: Path | None) -> bool:
    return any(path is None or not path.exists() for path in paths)


def _describe_missing_paths(path_map: dict[str, Path | None]) -> str:
    missing = [name for name, path in path_map.items() if path is None or not path.exists()]
    return ", ".join(missing)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = _resolve_paths(args)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _build_manifest(paths)
    manifest_path = paths.output_dir / MANIFEST_FILENAME
    current_stage_name = "initialization"

    try:
        dictionary_lists, harvard_negative_word_list = _load_dictionary_lists(paths.additional_data_dir)
        master_dictionary_words = _load_master_dictionary_words(paths.additional_data_dir)

        sample_backbone_lf: pl.LazyFrame | None = None
        annual_accounting_panel_lf: pl.LazyFrame | None = None
        quarterly_accounting_panel_lf: pl.LazyFrame | None = None
        ff_factors_daily_lf: pl.LazyFrame | None = None
        text_features_full_10k_lf: pl.LazyFrame | None = None
        text_features_mda_lf: pl.LazyFrame | None = None
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
        items_analysis_lf = (
            _prepare_lm2011_sec_input_lf(pl.scan_parquet(str(paths.items_analysis_dir / ITEMS_ANALYSIS_GLOB)))
            if _parquet_glob_exists(paths.items_analysis_dir, ITEMS_ANALYSIS_GLOB)
            else None
        )

        current_stage_name = "sample_backbone"
        if (
            year_merged_lf is not None
            and paths.matched_clean_path.exists()
            and paths.filingdates_path is not None
            and paths.filingdates_path.exists()
        ):
            sample_backbone_lf = _write_stage(
                manifest,
                output_dir=paths.output_dir,
                stage_name="sample_backbone",
                frame=build_lm2011_sample_backbone(
                    year_merged_lf,
                    pl.scan_parquet(paths.matched_clean_path),
                    ccm_filingdates_lf=pl.scan_parquet(paths.filingdates_path),
                ),
            )
        else:
            _record_stage_skipped(
                manifest,
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
        if not _missing_required_paths(
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
            _record_stage_skipped(
                manifest,
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
        if not _missing_required_paths(
            paths.quarterly_balance_sheet_path,
            paths.quarterly_income_statement_path,
            paths.quarterly_period_descriptor_path,
        ):
            quarterly_accounting_panel_lf = _write_stage(
                manifest,
                output_dir=paths.output_dir,
                stage_name="quarterly_accounting_panel",
                frame=build_quarterly_accounting_panel(
                    pl.scan_parquet(paths.quarterly_balance_sheet_path),
                    pl.scan_parquet(paths.quarterly_income_statement_path),
                    pl.scan_parquet(paths.quarterly_period_descriptor_path),
                ),
            )
        else:
            _record_stage_skipped(
                manifest,
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
        if paths.ff_daily_csv_path.exists():
            ff_factors_daily_lf = _write_stage(
                manifest,
                output_dir=paths.output_dir,
                stage_name="ff_factors_daily_normalized",
                frame=_load_ff_factors_daily_lf(paths.ff_daily_csv_path),
            )
        else:
            _record_stage_skipped(
                manifest,
                stage_name="ff_factors_daily_normalized",
                reason=SKIPPED_MISSING_OPTIONAL_INPUT,
                detail="ff_daily_csv_path",
            )

        current_stage_name = "text_features_full_10k"
        if year_merged_lf is not None:
            text_features_full_10k_lf = _write_stage(
                manifest,
                output_dir=paths.output_dir,
                stage_name="text_features_full_10k",
                frame=build_lm2011_text_features_full_10k(
                    year_merged_lf,
                    dictionary_lists=dictionary_lists,
                    harvard_negative_word_list=harvard_negative_word_list,
                    master_dictionary_words=master_dictionary_words,
                    cleaning_contract=paths.full_10k_cleaning_contract,
                ),
            )
        else:
            _record_stage_skipped(
                manifest,
                stage_name="text_features_full_10k",
                reason=SKIPPED_MISSING_OPTIONAL_INPUT,
                detail="year_merged",
            )

        current_stage_name = "text_features_mda"
        if items_analysis_lf is not None:
            text_features_mda_lf = _write_stage(
                manifest,
                output_dir=paths.output_dir,
                stage_name="text_features_mda",
                frame=build_lm2011_text_features_mda(
                    items_analysis_lf,
                    dictionary_lists=dictionary_lists,
                    harvard_negative_word_list=harvard_negative_word_list,
                    master_dictionary_words=master_dictionary_words,
                ),
            )
        else:
            _record_stage_skipped(
                manifest,
                stage_name="text_features_mda",
                reason=SKIPPED_MISSING_SEEDED_UPSTREAM,
                detail="items_analysis",
            )

        current_stage_name = "table_i_sample_creation"
        if (
            year_merged_lf is not None
            and paths.matched_clean_path.exists()
            and paths.filingdates_path is not None
            and paths.filingdates_path.exists()
            and annual_accounting_panel_lf is not None
            and ff_factors_daily_lf is not None
            and text_features_full_10k_lf is not None
            and paths.daily_panel_path.exists()
        ):
            table_i_df = build_lm2011_table_i_sample_creation(
                year_merged_lf,
                pl.scan_parquet(paths.matched_clean_path),
                pl.scan_parquet(paths.daily_panel_path),
                annual_accounting_panel_lf,
                ff_factors_daily_lf,
                text_features_full_10k_lf,
                ccm_filingdates_lf=pl.scan_parquet(paths.filingdates_path),
                mda_text_features_lf=text_features_mda_lf,
            )
            table_i_sample_creation_lf = _write_table_i_stage(
                manifest,
                output_dir=paths.output_dir,
                stage_name="table_i_sample_creation",
                table_df=table_i_df,
                sample_start=dt.date(1994, 1, 1),
                sample_end=dt.date(2008, 12, 31),
            )
        else:
            _record_stage_skipped(
                manifest,
                stage_name="table_i_sample_creation",
                reason=SKIPPED_MISSING_OPTIONAL_INPUT,
                detail=_describe_missing_paths(
                    {
                        "year_merged": paths.year_merged_dir if _parquet_glob_exists(paths.year_merged_dir, YEAR_MERGED_GLOB) else None,
                        "matched_clean": paths.matched_clean_path,
                        "filingdates": paths.filingdates_path,
                        "annual_accounting_panel": Path(manifest["artifacts"]["annual_accounting_panel"]) if "annual_accounting_panel" in manifest["artifacts"] else None,
                        "ff_factors_daily_normalized": Path(manifest["artifacts"]["ff_factors_daily_normalized"]) if "ff_factors_daily_normalized" in manifest["artifacts"] else None,
                        "text_features_full_10k": Path(manifest["artifacts"]["text_features_full_10k"]) if "text_features_full_10k" in manifest["artifacts"] else None,
                        "daily_panel_path": paths.daily_panel_path,
                    }
                ),
            )

        current_stage_name = "table_i_sample_creation_1994_2024"
        if (
            year_merged_lf is not None
            and paths.matched_clean_path.exists()
            and paths.filingdates_path is not None
            and paths.filingdates_path.exists()
            and annual_accounting_panel_lf is not None
            and ff_factors_daily_lf is not None
            and text_features_full_10k_lf is not None
            and paths.daily_panel_path.exists()
        ):
            table_i_1994_2024_df = build_lm2011_table_i_sample_creation(
                year_merged_lf,
                pl.scan_parquet(paths.matched_clean_path),
                pl.scan_parquet(paths.daily_panel_path),
                annual_accounting_panel_lf,
                ff_factors_daily_lf,
                text_features_full_10k_lf,
                ccm_filingdates_lf=pl.scan_parquet(paths.filingdates_path),
                mda_text_features_lf=text_features_mda_lf,
                sample_start=dt.date(1994, 1, 1),
                sample_end=dt.date(2024, 12, 31),
            )
            _write_table_i_stage(
                manifest,
                output_dir=paths.output_dir,
                stage_name="table_i_sample_creation_1994_2024",
                table_df=table_i_1994_2024_df,
                sample_start=dt.date(1994, 1, 1),
                sample_end=dt.date(2024, 12, 31),
            )
        else:
            _record_stage_skipped(
                manifest,
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
        if (
            sample_backbone_lf is not None
            and annual_accounting_panel_lf is not None
            and ff_factors_daily_lf is not None
            and text_features_full_10k_lf is not None
            and paths.daily_panel_path.exists()
            and paths.doc_ownership_path.exists()
        ):
            event_panel_lf = _write_stage(
                manifest,
                output_dir=paths.output_dir,
                stage_name="event_panel",
                frame=build_lm2011_event_panel(
                    sample_backbone_lf=sample_backbone_lf,
                    daily_lf=pl.scan_parquet(paths.daily_panel_path),
                    annual_accounting_panel_lf=annual_accounting_panel_lf,
                    ff_factors_daily_lf=ff_factors_daily_lf,
                    ownership_lf=pl.scan_parquet(paths.doc_ownership_path),
                    full_10k_text_features_lf=text_features_full_10k_lf,
                ),
            )
        else:
            _record_stage_skipped(
                manifest,
                stage_name="event_panel",
                reason=SKIPPED_MISSING_SEEDED_UPSTREAM,
                detail=_describe_missing_paths(
                    {
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
        if (
            event_panel_lf is not None
            and quarterly_accounting_panel_lf is not None
            and paths.doc_analyst_selected_path.exists()
            and paths.daily_panel_path.exists()
        ):
            sue_panel_lf = _write_stage(
                manifest,
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
            _record_stage_skipped(
                manifest,
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
        if event_panel_lf is not None and text_features_full_10k_lf is not None and can_run_regressions:
            return_regression_panel_full_10k_lf = _write_stage(
                manifest,
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
            _record_stage_skipped(
                manifest,
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
        if event_panel_lf is not None and text_features_mda_lf is not None and can_run_regressions:
            return_regression_panel_mda_lf = _write_stage(
                manifest,
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
            _record_stage_skipped(
                manifest,
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
        if sue_panel_lf is not None and text_features_full_10k_lf is not None and can_run_regressions:
            sue_regression_panel_lf = _write_stage(
                manifest,
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
            _record_stage_skipped(
                manifest,
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
            if should_run:
                _write_stage(
                    manifest,
                    output_dir=paths.output_dir,
                    stage_name=stage_name,
                    frame=builder(),
                    empty_reason=EMPTY_TABLE_REASON,
                )
            else:
                _record_stage_skipped(
                    manifest,
                    stage_name=stage_name,
                    reason=SKIPPED_MISSING_OPTIONAL_INPUT,
                    detail="missing upstream regression inputs",
                )

        current_stage_name = "trading_strategy_monthly_returns"
        if (
            paths.monthly_stock_path is not None
            and paths.monthly_stock_path.exists()
            and event_panel_lf is not None
            and year_merged_lf is not None
        ):
            _write_stage(
                manifest,
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
            _record_stage_skipped(
                manifest,
                stage_name="trading_strategy_monthly_returns",
                reason=SKIPPED_MISSING_OPTIONAL_INPUT,
                detail="monthly_stock_path",
            )

        current_stage_name = "table_ia_ii_results"
        if (
            paths.monthly_stock_path is not None
            and paths.monthly_stock_path.exists()
            and paths.ff_monthly_with_mom_path is not None
            and paths.ff_monthly_with_mom_path.exists()
            and event_panel_lf is not None
            and year_merged_lf is not None
        ):
            _write_stage(
                manifest,
                output_dir=paths.output_dir,
                stage_name="table_ia_ii_results",
                frame=build_lm2011_table_ia_ii_results(
                    event_panel_lf,
                    year_merged_lf,
                    pl.scan_parquet(paths.monthly_stock_path),
                    pl.scan_parquet(paths.ff_monthly_with_mom_path),
                    lm_dictionary_lists=dictionary_lists,
                    harvard_negative_word_list=harvard_negative_word_list,
                    master_dictionary_words=master_dictionary_words,
                ),
                empty_reason=EMPTY_TABLE_REASON,
            )
        else:
            _record_stage_skipped(
                manifest,
                stage_name="table_ia_ii_results",
                reason=SKIPPED_MISSING_OPTIONAL_INPUT,
                detail="monthly_stock_path or ff_monthly_with_mom_path",
            )

        manifest["run_status"] = "completed"
        manifest["completed_at_utc"] = _utc_timestamp()
        _write_json(manifest_path, manifest)
        return 0
    except Exception as exc:
        _record_stage_failed(manifest, stage_name=current_stage_name, exc=exc)
        manifest["run_status"] = "failed"
        manifest["completed_at_utc"] = _utc_timestamp()
        manifest["error"] = {
            "stage": current_stage_name,
            "type": type(exc).__name__,
            "message": str(exc),
        }
        _write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

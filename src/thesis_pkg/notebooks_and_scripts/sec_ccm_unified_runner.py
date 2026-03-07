from __future__ import annotations

"""Runnable script translation of ``sec_ccm_unified_runner.ipynb``.

The goal is to preserve the notebook's execution order and behavior while making
it executable as a normal Python script.
"""

import os
import sys
from pathlib import Path


IN_COLAB = "google.colab" in sys.modules

# The notebook assumed execution from repo root. For script execution, prefer
# cwd when it already looks like the repo root; otherwise fall back to the
# script's location.
_CWD_ROOT = Path.cwd().resolve()
_SCRIPT_ROOT = Path(__file__).resolve().parents[3]
ROOT = _CWD_ROOT if (_CWD_ROOT / "src").exists() else _SCRIPT_ROOT
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
    SecCcmJoinSpecV1,
    build_or_reuse_ccm_daily_stage,
    run_sec_ccm_premerge_pipeline,
)


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


def main() -> None:
    # ## Runtime setup
    if IN_COLAB:
        from google.colab import drive

        drive.mount("/content/drive", force_remount=False)

    print({"IN_COLAB": IN_COLAB, "ROOT": str(ROOT), "SRC_EXISTS": SRC.exists()})

    # ## Config
    DATA_PROFILE = "DRIVE_FULL" if IN_COLAB else "LOCAL_SAMPLE"

    SAMPLE_ROOT = ROOT / "full_data_run" / "sample_5pct_seed42"

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
            "RUN_CCM_MODE": "REUSE",
            "RUN_SEC_PARSE": None,
            "RUN_SEC_YEARLY_MERGE": None,
        },
        "DRIVE_FULL": {
            "WORK_ROOT": (
                Path("/content/drive/MyDrive/Data_LM")
                if IN_COLAB
                else Path("C:/Users/erik9/Documents/SEC_Data")
            ),
            "SEC_ZIP_DIR": (
                (
                    Path("/content/drive/MyDrive/Data_LM")
                    if IN_COLAB
                    else Path("C:/Users/erik9/Documents/SEC_Data")
                )
                / "Data"
                / "Sample_Filings"
            ),
            "SEC_BATCH_ROOT": (
                (
                    Path("/content/drive/MyDrive/Data_LM")
                    if IN_COLAB
                    else Path("C:/Users/erik9/Documents/SEC_Data")
                )
                / "Data"
                / "Sample_Filings"
                / "parquet_batches"
            ),
            "SEC_YEAR_MERGED_DIR": (
                (
                    Path("/content/drive/MyDrive/Data_LM")
                    if IN_COLAB
                    else Path("C:/Users/erik9/Documents/SEC_Data")
                )
                / "Data"
                / "Sample_Filings"
                / "parquet_batches"
                / "_year_merged"
            ),
            "SEC_LIGHT_METADATA_PATH": (
                (
                    Path("/content/drive/MyDrive/Data_LM")
                    if IN_COLAB
                    else Path("C:/Users/erik9/Documents/SEC_Data")
                )
                / "Data"
                / "Sample_Filings"
                / "filings_metadata_LIGHT.parquet"
            ),
            "CCM_BASE_DIR": (
                (
                    Path("/content/drive/MyDrive/Data_LM")
                    if IN_COLAB
                    else Path("C:/Users/erik9/Documents/SEC_Data")
                )
                / "Data"
                / "CRSP_Compustat_data"
                / "parquet_data"
            ),
            "CCM_DERIVED_DIR": (
                (
                    Path("/content/drive/MyDrive/Data_LM")
                    if IN_COLAB
                    else Path("C:/Users/erik9/Documents/SEC_Data")
                )
                / "Data"
                / "CRSP_Compustat_data"
                / "derived_data"
            ),
            "CCM_REUSE_DAILY_PATH": (
                (
                    Path("/content/drive/MyDrive/Data_LM")
                    if IN_COLAB
                    else Path("C:/Users/erik9/Documents/SEC_Data")
                )
                / "Data"
                / "CRSP_Compustat_data"
                / "derived_data"
                / "final_flagged_data_compdesc_added.parquet"
            ),
            "CANONICAL_LINK_NAME": "canonical_link_table_after_startdate_change.parquet",
            "CCM_DAILY_NAME": "final_flagged_data_compdesc_added.parquet",
            "RUN_CCM_MODE": "REUSE",
            "RUN_SEC_PARSE": False,
            "RUN_SEC_YEARLY_MERGE": True,
        },
    }

    if DATA_PROFILE not in PROFILE_CONFIG:
        raise ValueError(
            f"Unknown DATA_PROFILE={DATA_PROFILE!r}. Expected one of {list(PROFILE_CONFIG)}"
        )

    profile = PROFILE_CONFIG[DATA_PROFILE]
    WORK_ROOT = Path(profile["WORK_ROOT"])
    SEC_ZIP_DIR = Path(profile["SEC_ZIP_DIR"])
    SEC_BATCH_ROOT = Path(profile["SEC_BATCH_ROOT"])
    SEC_YEAR_MERGED_DIR = Path(profile["SEC_YEAR_MERGED_DIR"])
    SEC_LIGHT_METADATA_PATH = Path(profile["SEC_LIGHT_METADATA_PATH"])
    CCM_BASE_DIR = Path(profile["CCM_BASE_DIR"])
    CCM_DERIVED_DIR = Path(profile["CCM_DERIVED_DIR"])
    CCM_REUSE_DAILY_PATH = Path(profile["CCM_REUSE_DAILY_PATH"])
    CANONICAL_LINK_NAME = str(profile["CANONICAL_LINK_NAME"])
    CCM_DAILY_NAME = str(profile["CCM_DAILY_NAME"])

    available_years = _discover_sec_years(SEC_ZIP_DIR, SEC_YEAR_MERGED_DIR)
    if not available_years:
        available_years = list(range(1995, 2025))

    RUN_CCM_MODE = str(profile["RUN_CCM_MODE"])
    existing_year_outputs = _has_yearly_outputs(SEC_YEAR_MERGED_DIR)
    if profile["RUN_SEC_PARSE"] is None:
        RUN_SEC_PARSE = not existing_year_outputs
    else:
        RUN_SEC_PARSE = bool(profile["RUN_SEC_PARSE"])
    if profile["RUN_SEC_YEARLY_MERGE"] is None:
        RUN_SEC_YEARLY_MERGE = RUN_SEC_PARSE or not existing_year_outputs
    else:
        RUN_SEC_YEARLY_MERGE = bool(profile["RUN_SEC_YEARLY_MERGE"])
    RUN_SEC_CCM_PREMERGE = True
    RUN_GATED_ITEM_EXTRACTION = True
    RUN_UNMATCHED_DIAGNOSTIC_TRACK = False
    RUN_NO_ITEM_DIAGNOSTICS = True
    RUN_BOUNDARY_DIAGNOSTICS = True
    RUN_VALIDATION_CHECKS = True

    SEC_PARSE_MODE = "parsed"
    YEARS = available_years
    ITEM_EXTRACTION_REGIME = "legacy"

    RUN_ROOT = ROOT / "results" / "sec_ccm_unified_runner" / DATA_PROFILE.lower()
    SEC_CCM_OUTPUT_DIR = RUN_ROOT / "sec_ccm_premerge"
    SEC_ITEMS_ANALYSIS_DIR = RUN_ROOT / "items_analysis"
    SEC_ITEMS_DIAGNOSTIC_DIR = RUN_ROOT / "items_diagnostic"
    SEC_NO_ITEM_DIR = RUN_ROOT / "no_item_diagnostics"
    BOUNDARY_OUT_DIR = RUN_ROOT / "boundary_diagnostics"
    BOUNDARY_INPUT_DIR = BOUNDARY_OUT_DIR / "matched_filings_input"

    if IN_COLAB:
        LOCAL_TMP = Path("/content/_tmp_zip")
        LOCAL_WORK = Path("/content/_batch_work")
        LOCAL_ITEM_WORK = Path("/content/_item_work")
        LOCAL_MERGE_WORK = Path("/content/_merge_work")
    else:
        LOCAL_TMP = ROOT / ".tmp" / "zip"
        LOCAL_WORK = ROOT / ".tmp" / "batch_work"
        LOCAL_ITEM_WORK = ROOT / ".tmp" / "item_work"
        LOCAL_MERGE_WORK = ROOT / ".tmp" / "merge_work"

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
        LOCAL_TMP,
        LOCAL_WORK,
        LOCAL_ITEM_WORK,
        LOCAL_MERGE_WORK,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    FORMS_10K_10Q = [
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
    DAILY_FEATURE_COLUMNS = (
        "RET",
        "RETX",
        "PRC",
        "FINAL_PRC",
        "BIDLO",
        "ASKHI",
        "VOL",
        "TCAP",
        "SHRCD",
        "EXCHCD",
    )
    REQUIRED_DAILY_NON_NULL_FEATURES = ("RET",)

    print(
        {
            "DATA_PROFILE": DATA_PROFILE,
            "RUN_CCM_MODE": RUN_CCM_MODE,
            "WORK_ROOT": str(WORK_ROOT),
            "RUN_ROOT": str(RUN_ROOT),
            "RUN_SEC_PARSE": RUN_SEC_PARSE,
            "RUN_SEC_YEARLY_MERGE": RUN_SEC_YEARLY_MERGE,
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
    canonical_link_path = ccm_stage_paths["canonical_link_path"]

    ccm_daily_lf = pl.scan_parquet(ccm_daily_path)
    print(
        {
            "ccm_daily_path": str(ccm_daily_path),
            "canonical_link_path": str(canonical_link_path),
            "rows": ccm_daily_lf.select(pl.len()).collect().item(),
        }
    )

    # ## 2) Build link universe + trading calendar
    schema = ccm_daily_lf.collect_schema()
    resolved_permno_col = _first_existing(
        schema,
        ("KYPERMNO", "LPERMNO", "PERMNO"),
        "ccm_daily",
    )
    resolved_date_col = _first_existing(
        schema,
        ("CALDT", "caldt"),
        "ccm_daily",
    )

    link_universe_lf = pl.scan_parquet(canonical_link_path)
    trading_calendar_lf = (
        ccm_daily_lf.select(
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

    if RUN_SEC_CCM_PREMERGE:
        join_spec = SecCcmJoinSpecV1(
            alignment_policy="NEXT_TRADING_DAY_STRICT",
            daily_join_enabled=True,
            daily_join_source="MERGED_DAILY_PANEL",
            daily_permno_col=resolved_permno_col,
            daily_date_col=resolved_date_col,
            daily_feature_columns=tuple(DAILY_FEATURE_COLUMNS),
            required_daily_non_null_features=tuple(REQUIRED_DAILY_NON_NULL_FEATURES),
        )

        sec_ccm_paths = run_sec_ccm_premerge_pipeline(
            sec_filings_lf=sec_premerge_input_lf,
            link_universe_lf=link_universe_lf,
            trading_calendar_lf=trading_calendar_lf,
            output_dir=SEC_CCM_OUTPUT_DIR,
            daily_lf=ccm_daily_lf,
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

    print(pl.DataFrame(validation_rows) if validation_rows else "No validations executed")

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
    if sec_ccm_paths is not None:
        for key in sorted(sec_ccm_paths):
            _add("sec_ccm", key, Path(sec_ccm_paths[key]))
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

    print(
        pl.DataFrame(artifact_rows).sort(["stage", "artifact"])
        if artifact_rows
        else "No artifacts indexed"
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

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

from thesis_pkg.core.ccm import lm2011 as lm2011_core
from thesis_pkg.pipelines import lm2011_pipeline
from thesis_pkg.pipelines.lm2011_regressions import _RETURN_CONTROL_COLUMNS


DEFAULT_SAMPLE_ROOT = ROOT / "full_data_run" / "sample_5pct_seed42"
DEFAULT_REVIEW_BASENAME = f"lm2011_pre_refinitiv_thinness_audit_{dt.date.today().isoformat()}"
DEFAULT_REPORT_PATH = ROOT / "reviews" / f"{DEFAULT_REVIEW_BASENAME}.md"
DEFAULT_JSON_PATH = ROOT / "reviews" / f"{DEFAULT_REVIEW_BASENAME}.json"


@dataclass(frozen=True)
class AuditPaths:
    sample_root: Path
    sample_manifest_path: Path
    year_merged_dir: Path
    daily_panel_path: Path
    final_flagged_data_path: Path
    match_status_path: Path
    matched_clean_path: Path
    matched_clean_filtered_path: Path
    filingdates_path: Path
    sample_backbone_path: Path
    annual_accounting_panel_path: Path
    ff_factors_daily_path: Path
    text_features_full_10k_path: Path
    ownership_path: Path
    event_panel_path: Path
    return_regression_panel_path: Path
    full_premerge_archive_path: Path | None
    report_path: Path
    json_path: Path


@dataclass(frozen=True)
class VerificationCheck:
    name: str
    expected_rows: int
    actual_rows: int
    expected_permnos: int
    actual_permnos: int
    doc_sets_equal: bool | None = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit pre-Refinitiv LM2011 sample thinness from stored artifacts.")
    parser.add_argument("--sample-root", type=Path, default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--json-path", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--full-premerge-archive-path", type=Path, default=None)
    return parser.parse_args(argv)


def _latest_path(paths: Iterable[Path]) -> Path | None:
    candidates = sorted((path.resolve() for path in paths if path.exists()), key=lambda path: str(path))
    return candidates[-1] if candidates else None


def _discover_filingdates_path(sample_root: Path) -> Path:
    path = _latest_path((sample_root / "ccm_parquet_data").rglob("filingdates.parquet"))
    if path is None:
        raise FileNotFoundError(f"Could not find filingdates.parquet under {sample_root / 'ccm_parquet_data'}")
    return path


def _discover_full_premerge_archive_path(repo_root: Path) -> Path | None:
    return _latest_path(
        (repo_root / "full_data_run" / "archive").glob("sec_ccm_premerge-*/sec_ccm_premerge/sec_ccm_matched_clean.parquet")
    )


def _resolve_paths(args: argparse.Namespace) -> AuditPaths:
    sample_root = Path(args.sample_root).resolve()
    report_path = Path(args.report_path).resolve()
    json_path = Path(args.json_path).resolve()
    full_premerge_archive_path = (
        Path(args.full_premerge_archive_path).resolve()
        if args.full_premerge_archive_path is not None
        else _discover_full_premerge_archive_path(ROOT)
    )
    return AuditPaths(
        sample_root=sample_root,
        sample_manifest_path=sample_root / "sample_manifest.json",
        year_merged_dir=sample_root / "year_merged",
        daily_panel_path=sample_root / "derived_data" / "final_flagged_data_compdesc_added.sample_5pct_seed42.parquet",
        final_flagged_data_path=sample_root / "results" / "sec_ccm_unified_runner" / "local_sample" / "sec_ccm_premerge" / "final_flagged_data.parquet",
        match_status_path=sample_root / "results" / "sec_ccm_unified_runner" / "local_sample" / "sec_ccm_premerge" / "sec_ccm_match_status.parquet",
        matched_clean_path=sample_root / "results" / "sec_ccm_unified_runner" / "local_sample" / "sec_ccm_premerge" / "sec_ccm_matched_clean.parquet",
        matched_clean_filtered_path=sample_root / "results" / "sec_ccm_unified_runner" / "local_sample" / "sec_ccm_premerge" / "sec_ccm_matched_clean_filtered.parquet",
        filingdates_path=_discover_filingdates_path(sample_root),
        sample_backbone_path=sample_root / "results" / "lm2011_sample_post_refinitiv_runner" / "lm2011_sample_backbone.parquet",
        annual_accounting_panel_path=sample_root / "results" / "lm2011_sample_post_refinitiv_runner" / "lm2011_annual_accounting_panel.parquet",
        ff_factors_daily_path=sample_root / "results" / "lm2011_sample_post_refinitiv_runner" / "lm2011_ff_factors_daily_normalized.parquet",
        text_features_full_10k_path=sample_root / "results" / "lm2011_sample_post_refinitiv_runner" / "lm2011_text_features_full_10k.parquet",
        ownership_path=sample_root / "results" / "sec_ccm_unified_runner" / "local_sample" / "refinitiv_doc_ownership_lm2011" / "refinitiv_lm2011_doc_ownership.parquet",
        event_panel_path=sample_root / "results" / "lm2011_sample_post_refinitiv_runner" / "lm2011_event_panel.parquet",
        return_regression_panel_path=sample_root / "results" / "lm2011_sample_post_refinitiv_runner" / "lm2011_return_regression_panel_full_10k.parquet",
        full_premerge_archive_path=full_premerge_archive_path,
        report_path=report_path,
        json_path=json_path,
    )


def _ensure_inputs_exist(paths: AuditPaths) -> None:
    required_paths = {
        "sample_manifest_path": paths.sample_manifest_path,
        "year_merged_dir": paths.year_merged_dir,
        "daily_panel_path": paths.daily_panel_path,
        "final_flagged_data_path": paths.final_flagged_data_path,
        "match_status_path": paths.match_status_path,
        "matched_clean_path": paths.matched_clean_path,
        "matched_clean_filtered_path": paths.matched_clean_filtered_path,
        "filingdates_path": paths.filingdates_path,
        "sample_backbone_path": paths.sample_backbone_path,
        "annual_accounting_panel_path": paths.annual_accounting_panel_path,
        "ff_factors_daily_path": paths.ff_factors_daily_path,
        "text_features_full_10k_path": paths.text_features_full_10k_path,
        "ownership_path": paths.ownership_path,
        "event_panel_path": paths.event_panel_path,
        "return_regression_panel_path": paths.return_regression_panel_path,
    }
    missing = [name for name, path in required_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required audit inputs: {missing}")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=lambda value: value.isoformat() if isinstance(value, (dt.date, dt.datetime)) else str(value),
        ),
        encoding="utf-8",
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _to_python_date(value: Any) -> dt.date | None:
    if value is None:
        return None
    if isinstance(value, dt.date):
        return value
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, str):
        return dt.date.fromisoformat(value)
    raise TypeError(f"Unsupported date value: {value!r}")


def _round_or_none(value: float | None, digits: int = 2) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _format_int(value: int | None) -> str:
    if value is None:
        return "n/a"
    return f"{int(value):,}"


def _format_float(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):,.{digits}f}"


def _format_pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.{digits}f}%"


def _format_date(value: dt.date | None) -> str:
    return value.isoformat() if value is not None else "n/a"


def _scan_parquet(path: Path) -> pl.LazyFrame:
    return pl.scan_parquet(path)


def _collect_year_merged_lf(year_merged_dir: Path) -> pl.LazyFrame:
    year_paths = sorted(year_merged_dir.glob("*.parquet"))
    if not year_paths:
        raise FileNotFoundError(f"No year_merged parquet files found under {year_merged_dir}")
    return pl.scan_parquet([str(path) for path in year_paths])


def _resolve_permno_col(schema: pl.Schema) -> str | None:
    for candidate in ("KYPERMNO", "kypermno"):
        if candidate in schema:
            return candidate
    return None


def _summarize_lazy_frame(
    lf: pl.LazyFrame,
    *,
    doc_col: str = "doc_id",
    cik_col: str | None = "cik_10",
    permno_col: str | None = None,
    date_col: str | None = "filing_date",
) -> dict[str, Any]:
    schema = lf.collect_schema()
    exprs: list[pl.Expr] = [pl.len().alias("rows")]
    if doc_col in schema:
        exprs.append(pl.col(doc_col).n_unique().alias("distinct_doc_id"))
    if cik_col is not None and cik_col in schema:
        exprs.append(pl.col(cik_col).n_unique().alias("distinct_cik"))
    if permno_col is not None and permno_col in schema:
        exprs.append(pl.col(permno_col).cast(pl.Int32, strict=False).n_unique().alias("distinct_permno"))
    if date_col is not None and date_col in schema:
        exprs.extend(
            [
                pl.col(date_col).cast(pl.Date, strict=False).min().alias("min_date"),
                pl.col(date_col).cast(pl.Date, strict=False).max().alias("max_date"),
            ]
        )
    row = lf.select(exprs).collect().to_dicts()[0]
    if "min_date" in row:
        row["min_date"] = _to_python_date(row["min_date"])
    if "max_date" in row:
        row["max_date"] = _to_python_date(row["max_date"])
    return row


def _rows_per_permno_summary(df: pl.DataFrame, permno_col: str) -> dict[str, Any]:
    counts = df.group_by(permno_col).len().rename({"len": "rows_per_permno"})
    count_values = counts.get_column("rows_per_permno").to_list()
    top_rows = (
        counts.with_columns((pl.col("rows_per_permno") / pl.lit(df.height)).alias("row_share"))
        .sort("rows_per_permno", descending=True)
        .head(10)
        .to_dicts()
    )
    return {
        "permnos": counts.height,
        "mean_rows_per_permno": _round_or_none(float(sum(count_values)) / len(count_values) if count_values else None),
        "median_rows_per_permno": _round_or_none(float(statistics.median(count_values)) if count_values else None),
        "max_rows_per_permno": max(count_values) if count_values else None,
        "p90_rows_per_permno": _round_or_none(
            counts.select(pl.col("rows_per_permno").quantile(0.9).alias("p90")).item()
            if count_values
            else None
        ),
        "top_firms": [
            {
                "permno": int(row[permno_col]),
                "rows": int(row["rows_per_permno"]),
                "row_share": float(row["row_share"]),
            }
            for row in top_rows
        ],
    }


def _quarterly_breadth(df: pl.DataFrame, *, permno_col: str, date_col: str = "filing_date") -> dict[str, Any]:
    if df.height == 0:
        return {
            "summary": [],
            "stats": {
                "quarters": 0,
                "min_rows": 0,
                "median_rows": 0.0,
                "max_rows": 0,
                "min_permnos": 0,
                "median_permnos": 0.0,
                "max_permnos": 0,
                "quarters_rows_lt_5": 0,
                "quarters_rows_lt_10": 0,
                "quarters_permnos_lt_5": 0,
                "quarters_permnos_lt_10": 0,
            },
        }

    summary_df = (
        df.with_columns(pl.col(date_col).cast(pl.Date, strict=False).dt.truncate("1q").alias("quarter"))
        .group_by("quarter")
        .agg(
            pl.len().alias("rows"),
            pl.col(permno_col).cast(pl.Int32, strict=False).n_unique().alias("distinct_permno"),
        )
        .sort("quarter")
    )
    rows = summary_df.to_dicts()
    row_counts = [int(row["rows"]) for row in rows]
    permno_counts = [int(row["distinct_permno"]) for row in rows]
    return {
        "summary": [
            {
                "quarter": _format_date(_to_python_date(row["quarter"])),
                "rows": int(row["rows"]),
                "distinct_permno": int(row["distinct_permno"]),
            }
            for row in rows
        ],
        "stats": {
            "quarters": len(rows),
            "min_rows": min(row_counts),
            "median_rows": float(statistics.median(row_counts)),
            "max_rows": max(row_counts),
            "min_permnos": min(permno_counts),
            "median_permnos": float(statistics.median(permno_counts)),
            "max_permnos": max(permno_counts),
            "quarters_rows_lt_5": sum(value < 5 for value in row_counts),
            "quarters_rows_lt_10": sum(value < 10 for value in row_counts),
            "quarters_permnos_lt_5": sum(value < 5 for value in permno_counts),
            "quarters_permnos_lt_10": sum(value < 10 for value in permno_counts),
        },
    }


def _yearly_distinct_permnos(df: pl.DataFrame, *, permno_col: str, date_col: str = "filing_date") -> list[dict[str, Any]]:
    summary_df = (
        df.with_columns(pl.col(date_col).cast(pl.Date, strict=False).dt.year().alias("year"))
        .group_by("year")
        .agg(
            pl.len().alias("rows"),
            pl.col(permno_col).cast(pl.Int32, strict=False).n_unique().alias("distinct_permno"),
        )
        .sort("year")
    )
    return [
        {
            "year": int(row["year"]),
            "rows": int(row["rows"]),
            "distinct_permno": int(row["distinct_permno"]),
        }
        for row in summary_df.to_dicts()
    ]


def _permno_set(df: pl.DataFrame, permno_col: str) -> set[int]:
    return {int(value) for value in df.get_column(permno_col).drop_nulls().unique().to_list()}


def _anchor_overlap_summary(df: pl.DataFrame, permno_col: str, anchors: set[int]) -> dict[str, Any]:
    permnos = _permno_set(df, permno_col)
    overlap = permnos & anchors
    only_non_anchor = sorted(permnos - anchors)
    return {
        "distinct_permnos": len(permnos),
        "anchor_overlap": len(overlap),
        "anchor_overlap_share": (len(overlap) / len(permnos)) if permnos else None,
        "non_anchor_count": len(only_non_anchor),
        "non_anchor_permnos": only_non_anchor,
    }


def _make_stage_row(
    name: str,
    *,
    rows: int,
    distinct_permno: int | None,
    note: str,
    prev_rows: int | None = None,
    prev_permnos: int | None = None,
) -> dict[str, Any]:
    return {
        "stage": name,
        "rows": rows,
        "distinct_permno": distinct_permno,
        "rows_lost": None if prev_rows is None else prev_rows - rows,
        "permnos_lost": None if prev_permnos is None or distinct_permno is None else prev_permnos - distinct_permno,
        "note": note,
    }


def _markdown_table(rows: list[dict[str, Any]], columns: Sequence[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body_lines = ["| " + " | ".join(str(row.get(key, "")) for key, _ in columns) + " |" for row in rows]
    return "\n".join([header, divider, *body_lines])


def _build_backbone_ladder(paths: AuditPaths) -> dict[str, Any]:
    sec_lf = _collect_year_merged_lf(paths.year_merged_dir)
    matched_clean_lf = _scan_parquet(paths.matched_clean_path)
    filingdates_lf = _scan_parquet(paths.filingdates_path)

    sec_schema = sec_lf.collect_schema()
    optional_sec_cols = [
        name
        for name in ("accession_number", "acceptance_datetime", "period_end", "full_text")
        if name in sec_schema
    ]

    sec = (
        sec_lf.drop("normalized_form", strict=False)
        .with_columns(
            pl.col("doc_id").cast(pl.Utf8, strict=False),
            pl.col("cik_10").cast(pl.Utf8, strict=False),
            pl.col("accession_nodash").cast(pl.Utf8, strict=False),
            pl.col("filing_date").cast(pl.Date, strict=False),
            pl.col("document_type_filename").cast(pl.Utf8, strict=False),
            pl.col("document_type_filename")
            .map_elements(lm2011_core._normalize_sec_raw_form_value, return_dtype=pl.Utf8)
            .alias("_sec_raw_form"),
            lm2011_core.normalize_lm2011_form_expr("document_type_filename").alias("normalized_form"),
        )
        .filter(
            pl.col("doc_id").is_not_null()
            & pl.col("cik_10").is_not_null()
            & pl.col("filing_date").is_not_null()
            & pl.col("accession_nodash").is_not_null()
            & pl.col("document_type_filename").is_not_null()
        )
        .sort("doc_id", "filing_date", "accession_nodash")
        .unique(subset=["doc_id"], keep="first")
    )

    step_rows: list[dict[str, Any]] = []
    current = sec
    step_rows.append(
        {
            "stage": "SEC parsed base",
            **_summarize_lazy_frame(current, permno_col=None),
            "note": "All parsed sample filings before LM2011 restrictions.",
        }
    )

    current = current.filter(
        pl.col("filing_date").is_between(
            pl.lit(lm2011_core._LM2011_SAMPLE_START),
            pl.lit(lm2011_core._LM2011_SAMPLE_END),
            closed="both",
        )
    )
    step_rows.append(
        {
            "stage": "SEC date window",
            **_summarize_lazy_frame(current, permno_col=None),
            "note": "Keep filings dated 1994-01-01 through 2008-12-31.",
        }
    )

    current = current.filter(pl.col("_sec_raw_form").is_in(sorted(lm2011_core._LM2011_SEC_INCLUDED_RAW_FORMS)))
    step_rows.append(
        {
            "stage": "SEC included raw forms",
            **_summarize_lazy_frame(current, permno_col=None),
            "note": "Keep SEC raw forms 10-K and 10-K405.",
        }
    )

    current = current.filter(pl.col("_sec_raw_form").is_in(sorted(lm2011_core._LM2011_SEC_EXCLUDED_RAW_FORMS)).not_())
    step_rows.append(
        {
            "stage": "SEC excluded raw forms removed",
            **_summarize_lazy_frame(current, permno_col=None),
            "note": "Drop amended, transition, and small-business SEC forms.",
        }
    )

    current = (
        current.with_columns(pl.col("filing_date").dt.year().alias("_filing_year"))
        .sort("cik_10", "_filing_year", "filing_date", "accession_nodash")
        .unique(subset=["cik_10", "_filing_year"], keep="first")
    )
    step_rows.append(
        {
            "stage": "Per-CIK-year first filing",
            **_summarize_lazy_frame(current, permno_col=None),
            "note": "Keep the first filing per CIK-year.",
        }
    )

    current = (
        current.sort("cik_10", "filing_date", "accession_nodash")
        .with_columns(pl.col("filing_date").shift(1).over("cik_10").alias("_prev_kept_filing_date"))
        .filter(
            pl.col("_prev_kept_filing_date").is_null()
            | ((pl.col("filing_date") - pl.col("_prev_kept_filing_date")).dt.total_days() >= pl.lit(180))
        )
    )
    step_rows.append(
        {
            "stage": "180-day spacing",
            **_summarize_lazy_frame(current, permno_col=None),
            "note": "Require at least 180 days between kept filings for the same CIK.",
        }
    )

    current = current.select(
        "doc_id",
        "cik_10",
        "filing_date",
        "accession_nodash",
        pl.col("document_type_filename"),
        "normalized_form",
        *[pl.col(name) for name in optional_sec_cols],
    )

    matched_schema = matched_clean_lf.collect_schema()
    matched_clean = matched_clean_lf.with_columns(pl.col("doc_id").cast(pl.Utf8, strict=False))
    duplicated_sec_cols = [name for name in ("cik_10", "filing_date", "document_type_filename") if name in matched_schema]
    if duplicated_sec_cols:
        matched_clean = matched_clean.drop(*duplicated_sec_cols, strict=False)

    joined = current.join(matched_clean, on="doc_id", how="inner", suffix="_matched")
    joined_permno_col = _resolve_permno_col(joined.collect_schema())
    if joined_permno_col is None:
        raise ValueError("Backbone reconstruction could not find a permno column after joining matched_clean")
    step_rows.append(
        {
            "stage": "Join to sec_ccm_matched_clean",
            **_summarize_lazy_frame(joined, permno_col=joined_permno_col),
            "note": "Intersect SEC LM2011 candidates with the sampled matched filing universe.",
        }
    )

    ccm_gate = (
        filingdates_lf.select(
            pl.col("LPERMNO").cast(pl.Int32, strict=False).alias("_ccm_permno"),
            pl.col("FILEDATE").cast(pl.Date, strict=False).alias("_ccm_filing_date"),
            pl.col("SRCTYPE")
            .map_elements(lm2011_core._normalize_ccm_raw_form_value, return_dtype=pl.Utf8)
            .alias("_ccm_raw_form"),
        )
        .drop_nulls(subset=["_ccm_permno", "_ccm_filing_date", "_ccm_raw_form"])
        .group_by("_ccm_permno", "_ccm_filing_date")
        .agg(
            pl.col("_ccm_raw_form")
            .is_in(sorted(lm2011_core._LM2011_CCM_INCLUDED_RAW_FORMS))
            .any()
            .alias("_ccm_has_included_form"),
            pl.col("_ccm_raw_form")
            .is_in(sorted(lm2011_core._LM2011_CCM_EXCLUDED_RAW_FORMS))
            .any()
            .alias("_ccm_has_excluded_form"),
        )
    )
    joined_with_ccm = joined.with_columns(
        pl.col(joined_permno_col).cast(pl.Int32, strict=False).alias("_ccm_permno"),
        pl.col("filing_date").cast(pl.Date, strict=False).alias("_ccm_filing_date"),
    ).join(ccm_gate, on=["_ccm_permno", "_ccm_filing_date"], how="left")
    step_rows.append(
        {
            "stage": "CCM raw-form gate diagnostic",
            **_summarize_lazy_frame(joined_with_ccm, permno_col=joined_permno_col),
            "rows_with_included_form": int(
                joined_with_ccm.select(pl.col("_ccm_has_included_form").fill_null(False).sum()).collect().item()
            ),
            "rows_with_excluded_form": int(
                joined_with_ccm.select(pl.col("_ccm_has_excluded_form").fill_null(False).sum()).collect().item()
            ),
            "note": "Diagnostic before applying CCM raw-form inclusion and exclusion.",
        }
    )

    reconstructed_backbone_df = lm2011_core.build_lm2011_sample_backbone(
        sec_lf,
        matched_clean_lf,
        ccm_filingdates_lf=filingdates_lf,
    ).collect()
    step_rows.append(
        {
            "stage": "Final backbone",
            **_summarize_lazy_frame(reconstructed_backbone_df.lazy(), permno_col="KYPERMNO"),
            "note": "Stored LM2011 backbone target.",
        }
    )

    return {
        "steps": step_rows,
        "reconstructed_backbone": reconstructed_backbone_df,
    }


def _build_event_panel_ladder(paths: AuditPaths, backbone_df: pl.DataFrame) -> dict[str, Any]:
    annual_lf = _scan_parquet(paths.annual_accounting_panel_path).with_columns(
        pl.col("gvkey_int").cast(pl.Int32, strict=False).alias("gvkey_int")
    )
    daily_lf = _scan_parquet(paths.daily_panel_path)
    ff_factors_daily_lf = _scan_parquet(paths.ff_factors_daily_path)
    ownership_lf = _scan_parquet(paths.ownership_path)
    text_features_lf = _scan_parquet(paths.text_features_full_10k_path)

    docs_df = lm2011_pipeline._prepare_event_base_frame(
        backbone_df.lazy(),
        daily_lf,
        annual_lf,
        ownership_lf,
        text_features_lf,
    )

    daily_df = lm2011_pipeline._prepare_daily_event_frame(daily_lf, ff_factors_daily_lf, docs_df)
    docs_df, daily_df = lm2011_pipeline._attach_trade_indices(docs_df, daily_df)
    window_df = lm2011_pipeline._build_window_rows(docs_df, daily_df)

    event_summary = window_df.group_by("doc_id").agg(
        pl.when(pl.col("relative_day").is_between(0, 3, closed="both") & pl.col("stock_return").is_not_null())
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .sum()
        .alias("event_return_day_count"),
        pl.when(pl.col("relative_day").is_between(0, 3, closed="both") & pl.col("VOL").is_not_null())
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .sum()
        .alias("event_volume_day_count"),
        pl.when(pl.col("relative_day").is_between(0, 3, closed="both"))
        .then(pl.col("stock_return") + pl.lit(1.0))
        .otherwise(None)
        .product()
        .alias("_event_stock_gross"),
        pl.when(pl.col("relative_day").is_between(0, 3, closed="both"))
        .then(pl.col("market_return") + pl.lit(1.0))
        .otherwise(None)
        .product()
        .alias("_event_market_gross"),
        pl.when(
            pl.col("relative_day").is_between(-252, -6, closed="both")
            & pl.col("stock_return").is_not_null()
            & pl.col("VOL").is_not_null()
        )
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .sum()
        .alias("pre_turnover_obs"),
        pl.when(pl.col("relative_day").is_between(-252, -6, closed="both"))
        .then(pl.col("VOL"))
        .otherwise(None)
        .sum()
        .alias("_turnover_volume_sum"),
        pl.when(pl.col("relative_day").is_between(-65, -6, closed="both") & pl.col("VOL").is_not_null())
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .sum()
        .alias("abnormal_volume_pre_obs"),
        pl.when(pl.col("relative_day") == 0).then(pl.col("SHROUT")).otherwise(None).drop_nulls().first().alias("event_shares"),
        pl.when(pl.col("relative_day") == 0).then(pl.col("SHRCD")).otherwise(None).drop_nulls().first().alias("event_shrcd"),
        pl.when(pl.col("relative_day") == 0).then(pl.col("EXCHCD")).otherwise(None).drop_nulls().first().alias("event_exchcd"),
    )

    abnormal_pre_stats = (
        window_df.filter(pl.col("relative_day").is_between(-65, -6, closed="both") & pl.col("VOL").is_not_null())
        .group_by("doc_id")
        .agg(
            pl.col("VOL").mean().alias("_pre_vol_mean"),
            pl.col("VOL").std().alias("_pre_vol_std"),
        )
    )
    abnormal_event = (
        window_df.filter(pl.col("relative_day").is_between(0, 3, closed="both") & pl.col("VOL").is_not_null())
        .join(abnormal_pre_stats, on="doc_id", how="left")
        .filter(pl.col("_pre_vol_std").is_not_null() & (pl.col("_pre_vol_std") > 0))
        .with_columns(((pl.col("VOL") - pl.col("_pre_vol_mean")) / pl.col("_pre_vol_std")).alias("_std_volume"))
        .group_by("doc_id")
        .agg(pl.col("_std_volume").mean().alias("abnormal_volume"))
        .with_columns(pl.col("doc_id").cast(pl.Utf8, strict=False))
    )

    pre_alpha = lm2011_pipeline._regression_metrics_from_window(
        window_df,
        start_day=-252,
        end_day=-6,
        alpha_name="pre_ffalpha",
        rmse_name="_pre_ffalpha_rmse",
    ).rename({"n_obs": "pre_alpha_obs"})
    post_alpha = lm2011_pipeline._regression_metrics_from_window(
        window_df,
        start_day=6,
        end_day=252,
        alpha_name="_post_ffalpha",
        rmse_name="postevent_return_volatility",
    ).rename({"n_obs": "post_alpha_obs"})
    event_summary = event_summary.with_columns(pl.col("doc_id").cast(pl.Utf8, strict=False))
    pre_alpha = pre_alpha.with_columns(pl.col("doc_id").cast(pl.Utf8, strict=False))
    post_alpha = post_alpha.with_columns(pl.col("doc_id").cast(pl.Utf8, strict=False))

    panel = (
        docs_df.join(event_summary, on="doc_id", how="left")
        .join(abnormal_event, on="doc_id", how="left")
        .join(pre_alpha, on="doc_id", how="left")
        .join(post_alpha, on="doc_id", how="left")
        .with_columns(
            (pl.col("_event_stock_gross") - pl.col("_event_market_gross")).alias("filing_period_excess_return"),
            (
                pl.when(pl.col("event_shares").is_not_null() & (pl.col("event_shares") > 0))
                .then(pl.col("_turnover_volume_sum") / pl.col("event_shares"))
                .otherwise(None)
            ).alias("share_turnover"),
            pl.when(pl.col("event_exchcd") == 3)
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .cast(pl.Int8)
            .alias("nasdaq_dummy"),
        )
    )

    step_rows: list[dict[str, Any]] = []

    def snapshot(name: str, df: pl.DataFrame, note: str) -> None:
        step_rows.append(
            {
                "stage": name,
                "rows": int(df.height),
                "distinct_doc_id": int(df["doc_id"].n_unique()),
                "distinct_permno": int(df["KYPERMNO"].n_unique()),
                "ownership_nonnull_rows": int(df.select(pl.col("institutional_ownership").is_not_null().sum()).item())
                if "institutional_ownership" in df.columns
                else None,
                "ownership_nonnull_share": (
                    float(df.select(pl.col("institutional_ownership").is_not_null().mean()).item())
                    if "institutional_ownership" in df.columns and df.height > 0
                    else None
                ),
                "note": note,
            }
        )

    snapshot(
        "Event base plus metrics",
        panel,
        "Backbone with trade anchors, annual accounting, pre-filing market data, ownership left join, token counts, and event metrics.",
    )

    filters: list[tuple[str, pl.Expr, str]] = [
        ("Common stock", pl.col("event_shrcd").is_in([10, 11]), "Keep SHRCD in {10, 11}."),
        (
            "Positive size",
            pl.col("size_event").cast(pl.Float64, strict=False).is_not_null() & (pl.col("size_event") > 0),
            "Require positive market capitalization at the event date.",
        ),
        (
            "Pre-filing price >= 3",
            pl.col("pre_filing_price").is_not_null() & (pl.col("pre_filing_price") >= 3.0),
            "Require a non-penny-stock price floor.",
        ),
        (
            "Complete 4-day event window",
            (pl.col("event_return_day_count") == 4) & (pl.col("event_volume_day_count") == 4),
            "Require four return and four volume observations in days 0 through 3.",
        ),
        (
            "Major exchange",
            pl.col("event_exchcd").is_in([1, 2, 3]),
            "Require NYSE, AMEX, or NASDAQ.",
        ),
        (
            "History windows >= 60 days",
            (pl.col("pre_turnover_obs") >= 60)
            & (pl.col("abnormal_volume_pre_obs") >= 60)
            & (pl.col("pre_alpha_obs") >= 60)
            & (pl.col("post_alpha_obs") >= 60),
            "Require enough daily history for turnover, abnormal volume, and FF alpha windows.",
        ),
        (
            "Positive book equity",
            pl.col("book_equity_be").cast(pl.Float64, strict=False) > 0,
            "Require annual accounting book equity above zero.",
        ),
        (
            "Positive B/M",
            pl.col("bm_event").cast(pl.Float64, strict=False) > 0,
            "Require positive book-to-market before winsorization.",
        ),
        (
            "Token count >= 2000",
            pl.col("token_count_full_10k").cast(pl.Int32, strict=False) >= 2000,
            "Require full 10-K token count of at least 2,000.",
        ),
        (
            "Abnormal volume non-null",
            pl.col("abnormal_volume").is_not_null(),
            "Require computable abnormal volume.",
        ),
    ]

    current = panel
    for name, expr, note in filters:
        current = current.filter(expr)
        if name == "Positive B/M":
            current = lm2011_pipeline._winsorize_column(current, "bm_event")
        snapshot(name, current, note)

    reconstructed_event_panel_df = current.select(
        "doc_id",
        "gvkey_int",
        "KYPERMNO",
        "filing_date",
        "filing_trade_date",
        "pre_filing_trade_date",
        "size_event",
        "bm_event",
        "share_turnover",
        "pre_ffalpha",
        "institutional_ownership",
        "nasdaq_dummy",
        "filing_period_excess_return",
        "abnormal_volume",
        "postevent_return_volatility",
    )

    return {
        "event_base_df": docs_df,
        "steps": step_rows,
        "reconstructed_event_panel": reconstructed_event_panel_df,
    }


def _build_regression_ready_variants(return_regression_panel_df: pl.DataFrame) -> dict[str, Any]:
    required_base_columns = [
        "filing_period_excess_return",
        "ff48_industry_id",
        *[column for column in _RETURN_CONTROL_COLUMNS if column != "institutional_ownership"],
    ]
    required_owner_columns = ["filing_period_excess_return", "ff48_industry_id", *_RETURN_CONTROL_COLUMNS]

    no_owner_df = return_regression_panel_df.drop_nulls(required_base_columns)
    owner_df = return_regression_panel_df.drop_nulls(required_owner_columns)

    stages = [
        {
            "stage": "Event panel to return regression panel base",
            "rows": int(return_regression_panel_df.height),
            "distinct_permno": int(return_regression_panel_df["KYPERMNO"].n_unique()),
            "note": "Stored full-10K return regression panel before complete-case filtering.",
        },
        {
            "stage": "No-ownership complete case",
            "rows": int(no_owner_df.height),
            "distinct_permno": int(no_owner_df["KYPERMNO"].n_unique()),
            "note": "Drop nulls on dependent variable, FF48 industry, and all non-ownership controls.",
        },
        {
            "stage": "Ownership-required complete case",
            "rows": int(owner_df.height),
            "distinct_permno": int(owner_df["KYPERMNO"].n_unique()),
            "note": "Downstream comparison only: add institutional ownership to the complete-case requirement.",
        },
    ]

    return {
        "stages": stages,
        "no_owner_df": no_owner_df,
        "owner_df": owner_df,
    }


def _verify_counts(
    *,
    stored_backbone_df: pl.DataFrame,
    reconstructed_backbone_df: pl.DataFrame,
    stored_event_panel_df: pl.DataFrame,
    reconstructed_event_panel_df: pl.DataFrame,
    no_owner_df: pl.DataFrame,
    owner_df: pl.DataFrame,
) -> list[VerificationCheck]:
    checks = [
        VerificationCheck(
            name="backbone_reconstruction",
            expected_rows=301,
            actual_rows=reconstructed_backbone_df.height,
            expected_permnos=34,
            actual_permnos=int(reconstructed_backbone_df["KYPERMNO"].n_unique()),
            doc_sets_equal=sorted(stored_backbone_df["doc_id"].to_list()) == sorted(reconstructed_backbone_df["doc_id"].to_list()),
        ),
        VerificationCheck(
            name="event_panel_reconstruction",
            expected_rows=254,
            actual_rows=reconstructed_event_panel_df.height,
            expected_permnos=31,
            actual_permnos=int(reconstructed_event_panel_df["KYPERMNO"].n_unique()),
            doc_sets_equal=sorted(stored_event_panel_df["doc_id"].to_list()) == sorted(reconstructed_event_panel_df["doc_id"].to_list()),
        ),
        VerificationCheck(
            name="no_ownership_complete_case",
            expected_rows=241,
            actual_rows=no_owner_df.height,
            expected_permnos=28,
            actual_permnos=int(no_owner_df["KYPERMNO"].n_unique()),
        ),
        VerificationCheck(
            name="ownership_complete_case",
            expected_rows=24,
            actual_rows=owner_df.height,
            expected_permnos=4,
            actual_permnos=int(owner_df["KYPERMNO"].n_unique()),
        ),
    ]

    failures = [
        check
        for check in checks
        if check.actual_rows != check.expected_rows
        or check.actual_permnos != check.expected_permnos
        or (check.doc_sets_equal is False)
    ]
    if failures:
        failure_text = ", ".join(
            f"{check.name}: rows {check.actual_rows}/{check.expected_rows}, permnos {check.actual_permnos}/{check.expected_permnos}, doc_sets_equal={check.doc_sets_equal}"
            for check in failures
        )
        raise RuntimeError(f"Audit verification failed: {failure_text}")
    return checks


def _rank_bottlenecks(stage_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    prev_row: dict[str, Any] | None = None
    for row in stage_rows:
        if prev_row is not None:
            ranked.append(
                {
                    "transition": f"{prev_row['stage']} -> {row['stage']}",
                    "rows_lost": int(prev_row["rows"]) - int(row["rows"]),
                    "permnos_lost": (
                        None
                        if prev_row.get("distinct_permno") is None or row.get("distinct_permno") is None
                        else int(prev_row["distinct_permno"]) - int(row["distinct_permno"])
                    ),
                    "note": row["note"],
                }
            )
        prev_row = row
    return sorted(
        ranked,
        key=lambda row: (
            row["permnos_lost"] is None,
            -(0 if row["permnos_lost"] is None else int(row["permnos_lost"])),
            -int(row["rows_lost"]),
            row["transition"],
        ),
    )


def _top_share_summary(rows_per_permno_summary: dict[str, Any], top_n: int = 5) -> float:
    return float(sum(item["row_share"] for item in rows_per_permno_summary["top_firms"][:top_n]))


def _build_key_stage_table(
    *,
    matched_clean_df: pl.DataFrame,
    matched_clean_window_df: pl.DataFrame,
    backbone_df: pl.DataFrame,
    event_panel_df: pl.DataFrame,
    no_owner_df: pl.DataFrame,
    owner_df: pl.DataFrame,
) -> list[dict[str, Any]]:
    stages = [
        ("Matched filing universe", matched_clean_df, "True upstream LM2011 matched universe from sec_ccm_premerge across all sample dates."),
        (
            "Matched filing universe (1994-2008 window)",
            matched_clean_window_df,
            "Same matched universe restricted to the LM2011 sample period before backbone-specific LM2011 filters.",
        ),
        ("LM2011 backbone", backbone_df, "After SEC window/forms, per-CIK-year dedup, 180-day spacing, and CCM raw-form gate."),
        ("Pre-Refinitiv event panel", event_panel_df, "After non-ownership event-panel filters. Ownership is only a left-joined context column here."),
        ("No-ownership regression-ready", no_owner_df, "Drop nulls on DV, FF48 industry, and non-ownership controls only."),
        ("Ownership-required comparison", owner_df, "Downstream comparison only. Adds ownership complete-case attrition."),
    ]
    rows: list[dict[str, Any]] = []
    prev_rows: int | None = None
    prev_permnos: int | None = None
    for stage_name, df, note in stages:
        rows.append(
            _make_stage_row(
                stage_name,
                rows=df.height,
                distinct_permno=int(df["KYPERMNO"].n_unique()),
                note=note,
                prev_rows=prev_rows,
                prev_permnos=prev_permnos,
            )
        )
        prev_rows = df.height
        prev_permnos = int(df["KYPERMNO"].n_unique())
    return rows


def _build_sample_baseline(paths: AuditPaths, sample_manifest: dict[str, Any]) -> dict[str, Any]:
    matched_clean_df = pl.read_parquet(paths.matched_clean_path)
    matched_clean_filtered_df = pl.read_parquet(paths.matched_clean_filtered_path)
    backbone_df = pl.read_parquet(paths.sample_backbone_path)
    event_panel_df = pl.read_parquet(paths.event_panel_path)

    matched_permno_col = _resolve_permno_col(matched_clean_df.schema)
    filtered_permno_col = _resolve_permno_col(matched_clean_filtered_df.schema)
    if matched_permno_col is None or filtered_permno_col is None:
        raise ValueError("Could not find permno column in matched universe artifacts")

    anchors = {int(value) for value in sample_manifest["anchors"]["mandatory_anchor_permnos"]}
    counts = sample_manifest["counts"]
    overlap_report = sample_manifest["overlap_report"]
    config = sample_manifest["config"]
    return {
        "sample_manifest_counts": {
            "sampled_permno_count": int(counts["sampled_permno_count"]),
            "covered_filing_cik_count": int(overlap_report["filing_cik_all_covered_count"]),
            "covered_filing_doc_count": int(overlap_report["filing_doc_all_covered_count"]),
            "overlap_target": float(config["overlap_target"]),
            "mandatory_anchor_permno_count": int(counts["mandatory_anchor_permno_count"]),
        },
        "anchor_overlap": {
            "matched_clean": _anchor_overlap_summary(matched_clean_df, matched_permno_col, anchors),
            "backbone": _anchor_overlap_summary(backbone_df, "KYPERMNO", anchors),
            "event_panel": _anchor_overlap_summary(event_panel_df, "KYPERMNO", anchors),
        },
        "matched_clean_filtered_note": {
            "rows": int(matched_clean_filtered_df.height),
            "distinct_permno": int(matched_clean_filtered_df[filtered_permno_col].n_unique()),
            "distinct_cik": int(matched_clean_filtered_df["cik_10"].n_unique()) if "cik_10" in matched_clean_filtered_df.columns else None,
            "note": "Ownership-side artifact only. Not a valid upstream start point for the LM2011 backbone.",
        },
    }


def _build_stage_one_context(paths: AuditPaths) -> dict[str, Any]:
    matched_clean_df = pl.read_parquet(paths.matched_clean_path)
    matched_clean_permno_col = _resolve_permno_col(matched_clean_df.schema)
    if matched_clean_permno_col is None:
        raise ValueError("Could not find permno column in matched_clean artifact")

    stage_one = {
        "sample_matched_clean_summary": _summarize_lazy_frame(
            matched_clean_df.lazy(),
            permno_col=matched_clean_permno_col,
        ),
        "sample_matched_clean_rows_per_permno": _rows_per_permno_summary(matched_clean_df, matched_clean_permno_col),
        "sample_matched_clean_yearly": _yearly_distinct_permnos(matched_clean_df, permno_col=matched_clean_permno_col),
    }

    stage_one["sample_matched_clean_window_summary"] = _summarize_lazy_frame(
        matched_clean_df.lazy().filter(
            pl.col("filing_date").cast(pl.Date, strict=False).is_between(
                dt.date(1994, 1, 1),
                dt.date(2008, 12, 31),
                closed="both",
            )
        ),
        permno_col=matched_clean_permno_col,
    )

    if paths.full_premerge_archive_path is not None and paths.full_premerge_archive_path.exists():
        full_lf = _scan_parquet(paths.full_premerge_archive_path)
        full_permno_col = _resolve_permno_col(full_lf.collect_schema())
        if full_permno_col is None:
            raise ValueError("Could not find permno column in full premerge archive")
        stage_one["full_premerge_context_overall"] = _summarize_lazy_frame(full_lf, permno_col=full_permno_col)
        stage_one["full_premerge_context_1994_2008"] = _summarize_lazy_frame(
            full_lf.filter(
                pl.col("filing_date").cast(pl.Date, strict=False).is_between(
                    dt.date(1994, 1, 1),
                    dt.date(2008, 12, 31),
                    closed="both",
                )
            ),
            permno_col=full_permno_col,
        )
    else:
        stage_one["full_premerge_context_overall"] = None
        stage_one["full_premerge_context_1994_2008"] = None

    return stage_one


def _build_upstream_selection(paths: AuditPaths) -> dict[str, Any]:
    sec_lf = _collect_year_merged_lf(paths.year_merged_dir)
    final_flagged_lf = _scan_parquet(paths.final_flagged_data_path)
    match_status_lf = _scan_parquet(paths.match_status_path)

    final_flagged_schema = final_flagged_lf.collect_schema()
    final_flagged_permno_col = _resolve_permno_col(final_flagged_schema)
    match_status_schema = match_status_lf.collect_schema()
    match_status_permno_col = _resolve_permno_col(match_status_schema)

    if "match_flag" not in match_status_schema:
        raise ValueError("sec_ccm_match_status is missing match_flag")

    phase_a_ok_lf = match_status_lf.filter(pl.col("phase_a_reason_code") == pl.lit("OK"))
    matched_lf = match_status_lf.filter(pl.col("match_flag").cast(pl.Boolean, strict=False).fill_null(False))
    unmatched_lf = match_status_lf.filter(pl.col("match_flag").cast(pl.Boolean, strict=False).fill_null(False).not_())
    phase_a_failed_lf = match_status_lf.filter(pl.col("phase_a_reason_code") != pl.lit("OK"))
    phase_b_failed_lf = phase_a_ok_lf.filter(pl.col("match_flag").cast(pl.Boolean, strict=False).fill_null(False).not_())

    match_reason_counts = (
        match_status_lf.group_by("match_reason_code")
        .agg(pl.len().alias("rows"))
        .sort("rows", descending=True)
        .collect()
        .to_dicts()
    )
    phase_a_reason_counts = (
        match_status_lf.group_by("phase_a_reason_code")
        .agg(pl.len().alias("rows"))
        .sort("rows", descending=True)
        .collect()
        .to_dicts()
    )
    phase_b_reason_counts = (
        match_status_lf.group_by("phase_b_reason_code")
        .agg(pl.len().alias("rows"))
        .sort("rows", descending=True)
        .collect()
        .to_dicts()
    )

    final_flagged_summary = _summarize_lazy_frame(final_flagged_lf, permno_col=final_flagged_permno_col)
    match_status_summary = _summarize_lazy_frame(match_status_lf, permno_col=match_status_permno_col)
    phase_a_ok_summary = _summarize_lazy_frame(phase_a_ok_lf, permno_col=match_status_permno_col)
    matched_summary = _summarize_lazy_frame(matched_lf, permno_col=match_status_permno_col)
    unmatched_summary = _summarize_lazy_frame(unmatched_lf, permno_col=None)
    phase_a_failed_summary = _summarize_lazy_frame(phase_a_failed_lf, permno_col=None)
    phase_b_failed_summary = _summarize_lazy_frame(phase_b_failed_lf, permno_col=match_status_permno_col)

    stage_rows = [
        {
            "stage": "SEC parsed sample universe",
            **_summarize_lazy_frame(sec_lf, permno_col=None),
            "note": "All parsed sample filings before sec_ccm_premerge.",
        },
        {
            "stage": "sec_ccm_premerge final_flagged_data",
            **final_flagged_summary,
            "note": "Premerge output after link resolution, date alignment, daily join, and filter flags. No row loss from the parsed sample universe.",
        },
        {
            "stage": "sec_ccm_premerge phase A OK",
            **phase_a_ok_summary,
            "note": "Docs that survive link-universe membership and date-valid positive link selection.",
        },
        {
            "stage": "sec_ccm_match_status matched=true",
            **matched_summary,
            "note": "Docs whose final match reason is OK after phase B daily-feature and CCM-coverage checks.",
        },
    ]

    match_rate = matched_summary["rows"] / match_status_summary["rows"] if match_status_summary["rows"] else None
    unmatched_rate = unmatched_summary["rows"] / match_status_summary["rows"] if match_status_summary["rows"] else None

    return {
        "parsed_sample_summary": _summarize_lazy_frame(sec_lf, permno_col=None),
        "final_flagged_summary": final_flagged_summary,
        "match_status_summary": match_status_summary,
        "phase_a_ok_summary": phase_a_ok_summary,
        "phase_a_failed_summary": phase_a_failed_summary,
        "phase_b_failed_summary": phase_b_failed_summary,
        "matched_summary": matched_summary,
        "unmatched_summary": unmatched_summary,
        "match_rate": match_rate,
        "unmatched_rate": unmatched_rate,
        "match_reason_counts": match_reason_counts,
        "phase_a_reason_counts": phase_a_reason_counts,
        "phase_b_reason_counts": phase_b_reason_counts,
        "stage_rows": stage_rows,
    }


def _build_panel_breadth_section(
    *,
    matched_clean_df: pl.DataFrame,
    backbone_df: pl.DataFrame,
    event_panel_df: pl.DataFrame,
    no_owner_df: pl.DataFrame,
    owner_df: pl.DataFrame,
) -> dict[str, Any]:
    matched_clean_permno_col = _resolve_permno_col(matched_clean_df.schema)
    if matched_clean_permno_col is None:
        raise ValueError("Could not find permno column in matched_clean for breadth summary")
    panels = {
        "matched_clean": (matched_clean_df, matched_clean_permno_col),
        "backbone": (backbone_df, "KYPERMNO"),
        "event_panel": (event_panel_df, "KYPERMNO"),
        "no_ownership_regression_ready": (no_owner_df, "KYPERMNO"),
        "ownership_required": (owner_df, "KYPERMNO"),
    }
    out: dict[str, Any] = {}
    for key, (df, permno_col) in panels.items():
        rows_per_permno = _rows_per_permno_summary(df, permno_col)
        quarterly = _quarterly_breadth(df, permno_col=permno_col)
        out[key] = {
            "summary": _summarize_lazy_frame(df.lazy(), permno_col=permno_col),
            "rows_per_permno": rows_per_permno,
            "quarterly": quarterly,
            "top5_row_share": _top_share_summary(rows_per_permno, top_n=5),
        }
    return out


def _render_report(audit: dict[str, Any], paths: AuditPaths) -> str:
    sample_baseline = audit["sample_baseline"]
    upstream_selection = audit["upstream_selection"]
    stage_one = audit["stage_one"]
    backbone = audit["backbone"]
    event = audit["event"]
    regression = audit["regression"]
    breadth = audit["breadth"]
    verification = audit["verification"]

    key_stage_rows = [
        {
            "stage": row["stage"],
            "rows": _format_int(row["rows"]),
            "distinct_permno": _format_int(row["distinct_permno"]),
            "rows_lost": _format_int(row["rows_lost"]),
            "permnos_lost": _format_int(row["permnos_lost"]),
            "why_that_loss_matters": row["note"],
        }
        for row in audit["key_stages"]
    ]

    backbone_rows: list[dict[str, Any]] = []
    prev_step: dict[str, Any] | None = None
    for step in backbone["steps"]:
        backbone_rows.append(
            {
                "stage": step["stage"],
                "rows": _format_int(step["rows"]),
                "distinct_cik": _format_int(step.get("distinct_cik")),
                "distinct_permno": _format_int(step.get("distinct_permno")),
                "rows_lost": _format_int(None if prev_step is None else prev_step["rows"] - step["rows"]),
                "ciks_lost": _format_int(
                    None
                    if prev_step is None or step.get("distinct_cik") is None or prev_step.get("distinct_cik") is None
                    else prev_step["distinct_cik"] - step["distinct_cik"]
                ),
                "permnos_lost": _format_int(
                    None
                    if prev_step is None or step.get("distinct_permno") is None or prev_step.get("distinct_permno") is None
                    else prev_step["distinct_permno"] - step["distinct_permno"]
                ),
                "note": step["note"],
            }
        )
        prev_step = step

    event_rows: list[dict[str, Any]] = []
    prev_step = None
    for step in event["steps"]:
        event_rows.append(
            {
                "stage": step["stage"],
                "rows": _format_int(step["rows"]),
                "distinct_permno": _format_int(step["distinct_permno"]),
                "rows_lost": _format_int(None if prev_step is None else prev_step["rows"] - step["rows"]),
                "permnos_lost": _format_int(
                    None
                    if prev_step is None or step.get("distinct_permno") is None or prev_step.get("distinct_permno") is None
                    else prev_step["distinct_permno"] - step["distinct_permno"]
                ),
                "ownership_nonnull_share": _format_pct(step["ownership_nonnull_share"]),
                "note": step["note"],
            }
        )
        prev_step = step

    regression_rows: list[dict[str, Any]] = []
    prev_row: dict[str, Any] | None = None
    for row in regression["stages"]:
        regression_rows.append(
            {
                "stage": row["stage"],
                "rows": _format_int(row["rows"]),
                "distinct_permno": _format_int(row["distinct_permno"]),
                "rows_lost": _format_int(None if prev_row is None else prev_row["rows"] - row["rows"]),
                "permnos_lost": _format_int(
                    None if prev_row is None else prev_row["distinct_permno"] - row["distinct_permno"]
                ),
                "note": row["note"],
            }
        )
        prev_row = row

    upstream_rows = []
    prev_step = None
    for step in upstream_selection["stage_rows"]:
        upstream_rows.append(
            {
                "stage": step["stage"],
                "rows": _format_int(step["rows"]),
                "distinct_cik": _format_int(step.get("distinct_cik")),
                "distinct_permno": _format_int(step.get("distinct_permno")),
                "rows_lost": _format_int(None if prev_step is None else prev_step["rows"] - step["rows"]),
                "ciks_lost": _format_int(
                    None
                    if prev_step is None or step.get("distinct_cik") is None or prev_step.get("distinct_cik") is None
                    else prev_step["distinct_cik"] - step["distinct_cik"]
                ),
                "permnos_lost": _format_int(
                    None
                    if prev_step is None or step.get("distinct_permno") is None or prev_step.get("distinct_permno") is None
                    else prev_step["distinct_permno"] - step["distinct_permno"]
                ),
                "note": step["note"],
            }
        )
        prev_step = step

    breadth_rows: list[dict[str, Any]] = []
    for label, key in [
        ("LM2011 backbone", "backbone"),
        ("Pre-Refinitiv event panel", "event_panel"),
        ("No-ownership regression-ready", "no_ownership_regression_ready"),
    ]:
        item = breadth[key]
        quarterly = item["quarterly"]["stats"]
        rows_per_permno = item["rows_per_permno"]
        breadth_rows.append(
            {
                "universe": label,
                "rows": _format_int(item["summary"]["rows"]),
                "distinct_permno": _format_int(item["summary"]["distinct_permno"]),
                "mean_rows_per_permno": _format_float(rows_per_permno["mean_rows_per_permno"]),
                "median_rows_per_permno": _format_float(rows_per_permno["median_rows_per_permno"]),
                "max_rows_per_permno": _format_int(rows_per_permno["max_rows_per_permno"]),
                "top5_row_share": _format_pct(item["top5_row_share"]),
                "quarters": _format_int(quarterly["quarters"]),
                "median_rows_per_quarter": _format_float(quarterly["median_rows"]),
                "max_rows_per_quarter": _format_int(quarterly["max_rows"]),
                "median_permno_per_quarter": _format_float(quarterly["median_permnos"]),
                "max_permno_per_quarter": _format_int(quarterly["max_permnos"]),
                "quarters_rows_lt_5": _format_int(quarterly["quarters_rows_lt_5"]),
                "quarters_rows_lt_10": _format_int(quarterly["quarters_rows_lt_10"]),
                "quarters_permnos_lt_5": _format_int(quarterly["quarters_permnos_lt_5"]),
                "quarters_permnos_lt_10": _format_int(quarterly["quarters_permnos_lt_10"]),
            }
        )

    bottleneck_rows = [
        {
            "rank": index,
            "transition": row["transition"],
            "permnos_lost": _format_int(row["permnos_lost"]),
            "rows_lost": _format_int(row["rows_lost"]),
            "what_filter_is_doing": row["note"],
        }
        for index, row in enumerate(audit["bottlenecks"], start=1)
    ]

    verification_rows = [
        {
            "check": row["name"],
            "expected_rows": _format_int(row["expected_rows"]),
            "actual_rows": _format_int(row["actual_rows"]),
            "expected_permnos": _format_int(row["expected_permnos"]),
            "actual_permnos": _format_int(row["actual_permnos"]),
            "doc_sets_equal": "yes" if row.get("doc_sets_equal") is True else "n/a" if row.get("doc_sets_equal") is None else "no",
        }
        for row in verification
    ]

    sample_counts = sample_baseline["sample_manifest_counts"]
    anchor_overlap = sample_baseline["anchor_overlap"]
    phase_a_ok = upstream_selection["phase_a_ok_summary"]
    phase_a_failed = upstream_selection["phase_a_failed_summary"]
    phase_b_failed = upstream_selection["phase_b_failed_summary"]
    premerge_matched = upstream_selection["matched_summary"]
    premerge_unmatched = upstream_selection["unmatched_summary"]
    phase_a_reason_map = {row["phase_a_reason_code"]: int(row["rows"]) for row in upstream_selection["phase_a_reason_counts"]}
    phase_b_reason_map = {row["phase_b_reason_code"]: int(row["rows"]) for row in upstream_selection["phase_b_reason_counts"]}
    matched_clean_summary = stage_one["sample_matched_clean_summary"]
    matched_clean_window_summary = stage_one["sample_matched_clean_window_summary"]
    full_context_overall = stage_one["full_premerge_context_overall"]
    full_context_window = stage_one["full_premerge_context_1994_2008"]

    evidence_lines = [
        f"- Hard evidence: the first major contraction happens before the backbone. The sample SEC universe has {_format_int(upstream_selection['parsed_sample_summary']['rows'])} parsed filings across {_format_int(upstream_selection['parsed_sample_summary']['distinct_cik'])} CIKs, phase A of `sec_ccm_premerge` keeps only {_format_int(phase_a_ok['rows'])}, and final `sec_ccm_match_status` keeps {_format_int(premerge_matched['rows'])} matched filings across {_format_int(premerge_matched['distinct_cik'])} CIKs / {_format_int(premerge_matched['distinct_permno'])} PERMNOs.",
        f"- Hard evidence: the completed 5% run enters `sec_ccm_matched_clean` with {_format_int(matched_clean_summary['rows'])} matched filings across {_format_int(matched_clean_summary['distinct_permno'])} PERMNOs, reaches the LM2011 backbone at 301 rows / 34 PERMNOs, reaches the event panel at 254 rows / 31 PERMNOs, and reaches the no-ownership regression-ready panel at 241 rows / 28 PERMNOs.",
        f"- Hard evidence: the sample design itself is concentrated. `sample_manifest.json` reports {sample_counts['sampled_permno_count']:,} sampled PERMNOs overall, but only {sample_counts['covered_filing_cik_count']:,} filing CIKs are covered on the SEC side. The actual matched LM2011 run uses 38 PERMNOs, of which {anchor_overlap['matched_clean']['anchor_overlap']:,} are mandatory anchors.",
        "- Hard evidence: `sec_ccm_matched_clean_filtered` is not the backbone input. It has only 360 rows / 6 PERMNOs in this sample and belongs to the ownership request path.",
        "- Reasonable inference: for this completed 5% overlap-targeted run, the sample is already structurally thin before ownership matters because quarter-level cross sections remain tiny even in the no-ownership regression-ready panel.",
        "- Reasonable inference: within this sample run, ownership is mainly a later aggravating factor rather than the first cause of thinness. It collapses 28 pre-owner regression-ready PERMNOs to 4, but the panel is already too narrow before that step.",
        "- Unresolved uncertainty: there is no full-data LM2011 backbone or full-data pre-ownership event artifact in the repo to prove whether the full workflow is upstream-broken. The latest full premerge archive is only context.",
    ]

    lines = [
        "# Pre-Refinitiv LM2011 Thinness Audit",
        "",
        f"Generated: {dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()}",
        "",
        "## Executive Summary",
        "",
        f"- Yes, the current completed `sample_5pct_seed42` LM2011 run is already structurally thin before ownership becomes the main issue. The first choke point is upstream of the backbone: `sec_ccm_match_status` keeps only {_format_int(premerge_matched['rows'])} matched filings across {_format_int(premerge_matched['distinct_cik'])} CIKs / {_format_int(premerge_matched['distinct_permno'])} PERMNOs out of {_format_int(upstream_selection['parsed_sample_summary']['rows'])} parsed filings across {_format_int(upstream_selection['parsed_sample_summary']['distinct_cik'])} CIKs.",
        f"- The strongest evidence is quarter-level breadth before ownership: the no-ownership regression-ready panel has {_format_int(breadth['no_ownership_regression_ready']['summary']['rows'])} rows across {_format_int(breadth['no_ownership_regression_ready']['summary']['distinct_permno'])} PERMNOs, with median {_format_float(breadth['no_ownership_regression_ready']['quarterly']['stats']['median_permnos'])} PERMNOs per quarter and max {_format_int(breadth['no_ownership_regression_ready']['quarterly']['stats']['max_permnos'])}.",
        f"- The main upstream problem in this stored run starts before the backbone and even before `matched_clean`: `sec_ccm_premerge` drops {_format_int(premerge_unmatched['rows'])} filings before the matched universe, almost all in phase A ({_format_int(phase_a_failed['rows'])} docs) rather than phase B ({_format_int(phase_b_failed['rows'])} docs). The dominant phase-A reason is `CIK_NOT_IN_LINK_UNIVERSE` ({_format_int(phase_a_reason_map.get('CIK_NOT_IN_LINK_UNIVERSE'))} docs). By the time the analysis reaches `sec_ccm_matched_clean`, the sample is already down to {_format_int(matched_clean_summary['distinct_permno'])} PERMNOs, versus {_format_int(full_context_window['distinct_permno']) if full_context_window is not None else 'n/a'} in the latest full premerge archive's 1994-2008 slice.",
        "- Ownership is a downstream aggravating factor here. It reduces the no-ownership regression-ready panel from 241 rows / 28 PERMNOs to 24 rows / 4 PERMNOs, but the panel is already too thin for broad quarterly cross sections before that step.",
        "",
        "## Sample-Design Baseline",
        "",
        f"- Sample manifest counts: sampled PERMNOs = {_format_int(sample_counts['sampled_permno_count'])}, covered filing CIKs = {_format_int(sample_counts['covered_filing_cik_count'])}, covered filing docs = {_format_int(sample_counts['covered_filing_doc_count'])}, overlap target = {_format_pct(sample_counts['overlap_target'])}, mandatory anchors = {_format_int(sample_counts['mandatory_anchor_permno_count'])}.",
        f"- Anchor overlap in the actual LM2011 run: matched universe = {_format_int(anchor_overlap['matched_clean']['distinct_permnos'])} PERMNOs with {_format_int(anchor_overlap['matched_clean']['anchor_overlap'])} anchors; backbone = {_format_int(anchor_overlap['backbone']['distinct_permnos'])} with {_format_int(anchor_overlap['backbone']['anchor_overlap'])} anchors; event panel = {_format_int(anchor_overlap['event_panel']['distinct_permnos'])} with {_format_int(anchor_overlap['event_panel']['anchor_overlap'])} anchors.",
        f"- Ownership-side note: `sec_ccm_matched_clean_filtered` has {_format_int(sample_baseline['matched_clean_filtered_note']['rows'])} rows and {_format_int(sample_baseline['matched_clean_filtered_note']['distinct_permno'])} PERMNOs in this sample. It should not be treated as the LM2011 backbone start point.",
        "",
        "## Upstream Data Selection And sec_ccm_premerge",
        "",
        f"- Parsed sample SEC universe: {_format_int(upstream_selection['parsed_sample_summary']['rows'])} filings across {_format_int(upstream_selection['parsed_sample_summary']['distinct_cik'])} CIKs.",
        f"- `sec_ccm_premerge` phase A keeps {_format_int(phase_a_ok['rows'])} docs and removes {_format_int(phase_a_failed['rows'])}. This is where most row attrition happens, driven by `CIK_NOT_IN_LINK_UNIVERSE` ({_format_int(phase_a_reason_map.get('CIK_NOT_IN_LINK_UNIVERSE'))}) and `NO_DATE_VALID_POSITIVE_LINK` ({_format_int(phase_a_reason_map.get('NO_DATE_VALID_POSITIVE_LINK'))}).",
        f"- `sec_ccm_premerge` phase B then removes only {_format_int(phase_b_failed['rows'])} more docs before `matched_clean`, mainly `REQUIRED_DAILY_FEATURE_MISSING` ({_format_int(phase_b_reason_map.get('REQUIRED_DAILY_FEATURE_MISSING'))}) and `OUT_OF_CCM_COVERAGE` ({_format_int(phase_b_reason_map.get('OUT_OF_CCM_COVERAGE'))}).",
        f"- Final premerge result: {_format_int(premerge_matched['rows'])} matched filings and {_format_int(premerge_unmatched['rows'])} unmatched filings. Match rate = {_format_pct(upstream_selection['match_rate'])}; unmatched rate = {_format_pct(upstream_selection['unmatched_rate'])}.",
        "",
        _markdown_table(
            upstream_rows,
            columns=(
                ("stage", "Stage"),
                ("rows", "Rows"),
                ("distinct_cik", "Distinct CIK"),
                ("distinct_permno", "Distinct PERMNO"),
                ("rows_lost", "Rows Lost"),
                ("ciks_lost", "CIKs Lost"),
                ("permnos_lost", "PERMNOs Lost"),
                ("note", "What The Stage Does"),
            ),
        ),
        "",
        "## Stage 1: Matched Filing Universe",
        "",
        f"- Sample `sec_ccm_matched_clean`: {_format_int(matched_clean_summary['rows'])} filings, {_format_int(matched_clean_summary['distinct_permno'])} PERMNOs, {_format_int(matched_clean_summary['distinct_cik'])} CIKs, filing span {_format_date(matched_clean_summary['min_date'])} to {_format_date(matched_clean_summary['max_date'])}.",
        f"- Sample `sec_ccm_matched_clean` in the LM2011 1994-2008 window: {_format_int(matched_clean_window_summary['rows'])} filings across {_format_int(matched_clean_window_summary['distinct_permno'])} PERMNOs.",
        (
            f"- Latest full premerge archive context: overall `matched_clean` = {_format_int(full_context_overall['rows'])} rows / {_format_int(full_context_overall['distinct_permno'])} PERMNOs; 1994-2008 slice = {_format_int(full_context_window['rows'])} rows / {_format_int(full_context_window['distinct_permno'])} PERMNOs."
            if full_context_overall is not None and full_context_window is not None
            else "- Full premerge archive context was not available."
        ),
        f"- Concentration at this stage is already high: the top 5 PERMNOs account for {_format_pct(_top_share_summary(stage_one['sample_matched_clean_rows_per_permno']))} of matched-clean filings in the sample.",
        "",
        "## Stage 2: LM2011 Backbone Reconstruction",
        "",
        _markdown_table(
            backbone_rows,
            columns=(
                ("stage", "Stage"),
                ("rows", "Rows"),
                ("distinct_cik", "Distinct CIK"),
                ("distinct_permno", "Distinct PERMNO"),
                ("rows_lost", "Rows Lost"),
                ("ciks_lost", "CIKs Lost"),
                ("permnos_lost", "PERMNOs Lost"),
                ("note", "What The Stage Does"),
            ),
        ),
        "",
        "## Stage 3: Pre-Refinitiv Event Eligibility",
        "",
        _markdown_table(
            event_rows,
            columns=(
                ("stage", "Stage"),
                ("rows", "Rows"),
                ("distinct_permno", "Distinct PERMNO"),
                ("rows_lost", "Rows Lost"),
                ("permnos_lost", "PERMNOs Lost"),
                ("ownership_nonnull_share", "Ownership Non-Null Share"),
                ("note", "What The Filter Does"),
            ),
        ),
        "",
        "## Stage 4: Pre-Refinitiv Panel Breadth",
        "",
        _markdown_table(
            key_stage_rows,
            columns=(
                ("stage", "Stage"),
                ("rows", "Rows"),
                ("distinct_permno", "Distinct PERMNO"),
                ("rows_lost", "Rows Lost"),
                ("permnos_lost", "PERMNOs Lost"),
                ("why_that_loss_matters", "Why That Loss Matters"),
            ),
        ),
        "",
        _markdown_table(
            breadth_rows,
            columns=(
                ("universe", "Universe"),
                ("rows", "Rows"),
                ("distinct_permno", "Distinct PERMNO"),
                ("mean_rows_per_permno", "Mean Rows / PERMNO"),
                ("median_rows_per_permno", "Median Rows / PERMNO"),
                ("max_rows_per_permno", "Max Rows / PERMNO"),
                ("top5_row_share", "Top 5 Row Share"),
                ("quarters", "Quarters"),
                ("median_rows_per_quarter", "Median Rows / Quarter"),
                ("max_rows_per_quarter", "Max Rows / Quarter"),
                ("median_permno_per_quarter", "Median PERMNO / Quarter"),
                ("max_permno_per_quarter", "Max PERMNO / Quarter"),
                ("quarters_rows_lt_5", "Quarters Rows < 5"),
                ("quarters_rows_lt_10", "Quarters Rows < 10"),
                ("quarters_permnos_lt_5", "Quarters PERMNO < 5"),
                ("quarters_permnos_lt_10", "Quarters PERMNO < 10"),
            ),
        ),
        "",
        _markdown_table(
            regression_rows,
            columns=(
                ("stage", "Stage"),
                ("rows", "Rows"),
                ("distinct_permno", "Distinct PERMNO"),
                ("rows_lost", "Rows Lost"),
                ("permnos_lost", "PERMNOs Lost"),
                ("note", "What The Stage Does"),
            ),
        ),
        "",
        "## Bottleneck Ranking",
        "",
        _markdown_table(
            bottleneck_rows,
            columns=(
                ("rank", "Rank"),
                ("transition", "Transition"),
                ("permnos_lost", "PERMNOs Lost"),
                ("rows_lost", "Rows Lost"),
                ("what_filter_is_doing", "What The Filter Is Doing"),
            ),
        ),
        "",
        "## Verification",
        "",
        _markdown_table(
            verification_rows,
            columns=(
                ("check", "Check"),
                ("expected_rows", "Expected Rows"),
                ("actual_rows", "Actual Rows"),
                ("expected_permnos", "Expected PERMNOs"),
                ("actual_permnos", "Actual PERMNOs"),
                ("doc_sets_equal", "Doc Sets Equal"),
            ),
        ),
        "",
        "## Final Judgment",
        "",
        f"- The sample is already too thin before ownership in this completed 5% run. The first structural drop happens before the backbone, when `sec_ccm_premerge` reduces the parsed SEC sample from {_format_int(upstream_selection['parsed_sample_summary']['distinct_cik'])} filing CIKs to {_format_int(premerge_matched['distinct_cik'])} matched CIKs / {_format_int(premerge_matched['distinct_permno'])} PERMNOs, with most row loss occurring in phase A.",
        "- Ownership is mainly a later aggravating factor rather than the first root cause in this stored run.",
        "- The first upstream stage to inspect before any major rerun is the sample-construction plus `sec_ccm_premerge` link-universe coverage step, especially why `CIK_NOT_IN_LINK_UNIVERSE` removes 8,614 of 12,066 parsed filings before the matched universe is even formed.",
        "",
        "## Evidence Classification",
        "",
        *evidence_lines,
        "",
        "## Artifact Paths",
        "",
        f"- Report: `{paths.report_path}`",
        f"- JSON metrics: `{paths.json_path}`",
    ]
    return "\n".join(lines) + "\n"


def run_audit(paths: AuditPaths) -> dict[str, Any]:
    sample_manifest = _load_json(paths.sample_manifest_path)

    sample_baseline = _build_sample_baseline(paths, sample_manifest)
    upstream_selection = _build_upstream_selection(paths)
    stage_one = _build_stage_one_context(paths)

    reconstructed_backbone = _build_backbone_ladder(paths)
    stored_backbone_df = pl.read_parquet(paths.sample_backbone_path)
    reconstructed_backbone_df = reconstructed_backbone["reconstructed_backbone"]

    reconstructed_event = _build_event_panel_ladder(paths, reconstructed_backbone_df)
    stored_event_panel_df = pl.read_parquet(paths.event_panel_path)
    reconstructed_event_panel_df = reconstructed_event["reconstructed_event_panel"]

    return_regression_panel_df = pl.read_parquet(paths.return_regression_panel_path)
    regression = _build_regression_ready_variants(return_regression_panel_df)

    verification = _verify_counts(
        stored_backbone_df=stored_backbone_df,
        reconstructed_backbone_df=reconstructed_backbone_df,
        stored_event_panel_df=stored_event_panel_df,
        reconstructed_event_panel_df=reconstructed_event_panel_df,
        no_owner_df=regression["no_owner_df"],
        owner_df=regression["owner_df"],
    )

    matched_clean_df = pl.read_parquet(paths.matched_clean_path)
    matched_clean_permno_col = _resolve_permno_col(matched_clean_df.schema)
    if matched_clean_permno_col is None:
        raise ValueError("Could not find permno column in matched_clean")
    matched_clean_window_df = matched_clean_df.filter(
        pl.col("filing_date").cast(pl.Date, strict=False).is_between(
            pl.lit(lm2011_core._LM2011_SAMPLE_START),
            pl.lit(lm2011_core._LM2011_SAMPLE_END),
            closed="both",
        )
    )
    if matched_clean_permno_col != "KYPERMNO":
        matched_clean_window_df = matched_clean_window_df.rename({matched_clean_permno_col: "KYPERMNO"})
        matched_clean_df_for_keys = matched_clean_df.rename({matched_clean_permno_col: "KYPERMNO"})
    else:
        matched_clean_df_for_keys = matched_clean_df

    breadth = _build_panel_breadth_section(
        matched_clean_df=matched_clean_df,
        backbone_df=stored_backbone_df,
        event_panel_df=stored_event_panel_df,
        no_owner_df=regression["no_owner_df"],
        owner_df=regression["owner_df"],
    )

    key_stages = _build_key_stage_table(
        matched_clean_df=matched_clean_df_for_keys,
        matched_clean_window_df=matched_clean_window_df,
        backbone_df=stored_backbone_df,
        event_panel_df=stored_event_panel_df,
        no_owner_df=regression["no_owner_df"],
        owner_df=regression["owner_df"],
    )

    bottleneck_inputs = [
        {
            "stage": "sec_ccm_premerge final_flagged_data",
            "rows": int(upstream_selection["final_flagged_summary"]["rows"]),
            "distinct_permno": int(upstream_selection["final_flagged_summary"]["distinct_permno"]),
            "note": "Premerge output after link resolution, date alignment, daily joins, and flags.",
        },
        {
            "stage": "sec_ccm_premerge phase A OK",
            "rows": int(upstream_selection["phase_a_ok_summary"]["rows"]),
            "distinct_permno": int(upstream_selection["phase_a_ok_summary"]["distinct_permno"]),
            "note": "Docs that survive link-universe membership and date-valid positive link selection.",
        },
        {
            "stage": "Matched filing universe",
            "rows": int(matched_clean_df.height),
            "distinct_permno": int(matched_clean_df[matched_clean_permno_col].n_unique()),
            "note": "True upstream LM2011 matched universe from sec_ccm_premerge across all sample dates.",
        },
        {
            "stage": "Matched filing universe (1994-2008 window)",
            "rows": int(matched_clean_window_df.height),
            "distinct_permno": int(matched_clean_window_df["KYPERMNO"].n_unique()),
            "note": "Same matched universe restricted to the LM2011 sample period before backbone-specific filters.",
        },
        {
            "stage": "LM2011 backbone",
            "rows": int(stored_backbone_df.height),
            "distinct_permno": int(stored_backbone_df["KYPERMNO"].n_unique()),
            "note": "After SEC window/forms, per-CIK-year dedup, 180-day spacing, and CCM raw-form gate.",
        },
        *[
            {
                "stage": row["stage"],
                "rows": int(row["rows"]),
                "distinct_permno": int(row["distinct_permno"]),
                "note": row["note"],
            }
            for row in reconstructed_event["steps"]
            if row["stage"] != "Event base plus metrics"
        ],
        {
            "stage": "No-ownership regression-ready",
            "rows": int(regression["no_owner_df"].height),
            "distinct_permno": int(regression["no_owner_df"]["KYPERMNO"].n_unique()),
            "note": "Drop nulls on dependent variable, FF48 industry, and non-ownership controls.",
        },
    ]

    return {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "paths": {key: (str(value) if value is not None else None) for key, value in paths.__dict__.items()},
        "sample_baseline": sample_baseline,
        "upstream_selection": upstream_selection,
        "stage_one": stage_one,
        "backbone": {"steps": reconstructed_backbone["steps"]},
        "event": {"steps": reconstructed_event["steps"]},
        "regression": {"stages": regression["stages"]},
        "breadth": breadth,
        "key_stages": key_stages,
        "bottlenecks": _rank_bottlenecks(bottleneck_inputs),
        "verification": [check.__dict__ for check in verification],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = _resolve_paths(args)
    _ensure_inputs_exist(paths)

    audit = run_audit(paths)
    report_text = _render_report(audit, paths)

    _write_json(paths.json_path, audit)
    _write_text(paths.report_path, report_text)

    print(
        json.dumps(
            {
                "report_path": str(paths.report_path),
                "json_path": str(paths.json_path),
                "verification": audit["verification"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

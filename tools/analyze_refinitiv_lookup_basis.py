from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import polars as pl

from thesis_pkg.pipelines.refinitiv import is_lseg_available


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


DEFAULT_RUN_ROOT = (
    ROOT / "full_data_run" / "sample_5pct_seed42" / "results" / "sec_ccm_unified_runner" / "local_sample"
)
DEFAULT_OUTPUT_SUBDIR = Path("diagnostics") / "refinitiv_lookup_basis"
NULL_TEXT_VALUES: frozenset[str] = frozenset({"", "NULL", "None", "none", "null", "nan", "NaN"})
MAX_TOP_ROWS = 10
SAMPLE_BUCKET_ORDER: tuple[str, ...] = (
    "authority_exception_or_review",
    "multi_permno_or_alias_edge",
    "effective_with_successful_ownership",
    "effective_with_failed_or_null_ownership",
    "retrieval_attempt_no_useful_output",
    "ticker_or_nonconventional_candidate",
    "other_retrievable",
)
FAILURE_SAMPLE_BUCKET_ORDER: tuple[str, ...] = (
    "regressed_vs_snapshot",
    "review_required",
    "date_varying_exception",
    "no_conventional_authority",
    "static_conventional_zero_or_null",
    "multi_permno_or_alias_edge",
    "static_conventional_success",
    "other",
)
LM2011_POST_REFINITIV_OUTPUT_DIRNAME = "lm2011_sample_post_refinitiv_runner"


@dataclass(frozen=True)
class RunPaths:
    run_root: Path
    sec_ccm_premerge_dir: Path
    refinitiv_step1_dir: Path
    ownership_universe_dir: Path
    ownership_authority_dir: Path
    basis_path: Path
    lookup_snapshot_path: Path
    lookup_extended_path: Path
    resolution_path: Path
    bridge_path: Path
    instrument_authority_path: Path
    ownership_handoff_path: Path
    ownership_results_path: Path
    ownership_row_summary_path: Path
    authority_decisions_path: Path
    authority_exceptions_path: Path


@dataclass(frozen=True)
class Step1SnapshotPaths:
    label: str
    root: Path
    ownership_handoff_path: Path
    ownership_results_path: Path
    ownership_row_summary_path: Path
    authority_decisions_path: Path
    authority_exceptions_path: Path


@dataclass(frozen=True)
class DocOwnershipPaths:
    root: Path
    exact_requests_path: Path
    exact_raw_path: Path
    fallback_requests_path: Path
    fallback_raw_path: Path
    final_path: Path
    exact_manifest_path: Path
    fallback_manifest_path: Path


@dataclass(frozen=True)
class DownstreamPaths:
    root: Path
    sample_backbone_path: Path
    event_panel_path: Path
    return_regression_panel_full_10k_path: Path
    manifest_path: Path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a copied refinitiv_local_api_runner run-root to diagnose coverage drops "
            "between the filing basis, lookup/resolution, and ownership stages."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=DEFAULT_RUN_ROOT,
        help="Copied local run-root matching the refinitiv_local_api_runner output layout.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Relative paths are resolved under the run-root.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Target number of distinct RICs to sample for ownership retrieval testing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1729,
        help="Deterministic seed used for within-bucket sampling order.",
    )
    parser.add_argument("--year-start", type=int, default=None, help="Optional inclusive filing year lower bound.")
    parser.add_argument("--year-end", type=int, default=None, help="Optional inclusive filing year upper bound.")
    parser.add_argument(
        "--emit-sample-parquet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write the sampled-RIC output as parquet.",
    )
    parser.add_argument(
        "--emit-sample-csv",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write the sampled-RIC output as CSV.",
    )
    parser.add_argument(
        "--emit-failure-sample-parquet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write the failure-oriented sampled-RIC output as parquet.",
    )
    parser.add_argument(
        "--emit-failure-sample-csv",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write the failure-oriented sampled-RIC output as CSV.",
    )
    parser.add_argument(
        "--emit-doc-truth-parquet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write the doc-level ownership truth table as parquet.",
    )
    parser.add_argument(
        "--emit-doc-truth-csv",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write the doc-level ownership truth table as CSV.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print progress messages while running.")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_argument_parser().parse_args(argv)
    args.run_root = args.run_root.expanduser().resolve()
    if args.output_dir is None:
        args.output_dir = (args.run_root / DEFAULT_OUTPUT_SUBDIR).resolve()
    else:
        candidate = args.output_dir.expanduser()
        args.output_dir = (args.run_root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    if args.sample_size < 0:
        raise ValueError("--sample-size must be non-negative")
    if args.year_start is not None and args.year_end is not None and args.year_start > args.year_end:
        raise ValueError("--year-start must be less than or equal to --year-end")
    return args


def _resolve_run_paths(run_root: Path) -> RunPaths:
    sec_ccm_premerge_dir = run_root / "sec_ccm_premerge"
    refinitiv_step1_dir = run_root / "refinitiv_step1"
    ownership_universe_dir = refinitiv_step1_dir / "ownership_universe_common_stock"
    ownership_authority_dir = refinitiv_step1_dir / "ownership_authority_common_stock"
    return RunPaths(
        run_root=run_root,
        sec_ccm_premerge_dir=sec_ccm_premerge_dir,
        refinitiv_step1_dir=refinitiv_step1_dir,
        ownership_universe_dir=ownership_universe_dir,
        ownership_authority_dir=ownership_authority_dir,
        basis_path=sec_ccm_premerge_dir / "sec_ccm_matched_clean_filtered.parquet",
        lookup_snapshot_path=refinitiv_step1_dir / "refinitiv_ric_lookup_handoff_common_stock_extended_snapshot.parquet",
        lookup_extended_path=refinitiv_step1_dir / "refinitiv_ric_lookup_handoff_common_stock_extended.parquet",
        resolution_path=refinitiv_step1_dir / "refinitiv_ric_resolution_common_stock.parquet",
        bridge_path=refinitiv_step1_dir / "refinitiv_bridge_universe.parquet",
        instrument_authority_path=refinitiv_step1_dir / "refinitiv_instrument_authority_common_stock.parquet",
        ownership_handoff_path=ownership_universe_dir / "refinitiv_ownership_universe_handoff_common_stock.parquet",
        ownership_results_path=ownership_universe_dir / "refinitiv_ownership_universe_results.parquet",
        ownership_row_summary_path=ownership_universe_dir / "refinitiv_ownership_universe_row_summary.parquet",
        authority_decisions_path=ownership_authority_dir / "refinitiv_permno_ownership_authority_decisions.parquet",
        authority_exceptions_path=ownership_authority_dir / "refinitiv_permno_ownership_authority_exceptions.parquet",
    )


def _resolve_doc_ownership_paths(run_root: Path) -> DocOwnershipPaths | None:
    root = run_root / "refinitiv_doc_ownership_lm2011"
    paths = DocOwnershipPaths(
        root=root,
        exact_requests_path=root / "refinitiv_lm2011_doc_ownership_exact_requests.parquet",
        exact_raw_path=root / "refinitiv_lm2011_doc_ownership_exact_raw.parquet",
        fallback_requests_path=root / "refinitiv_lm2011_doc_ownership_fallback_requests.parquet",
        fallback_raw_path=root / "refinitiv_lm2011_doc_ownership_fallback_raw.parquet",
        final_path=root / "refinitiv_lm2011_doc_ownership.parquet",
        exact_manifest_path=root / "refinitiv_doc_ownership_exact_stage_manifest.json",
        fallback_manifest_path=root / "refinitiv_doc_ownership_fallback_stage_manifest.json",
    )
    required = (
        paths.exact_requests_path,
        paths.exact_raw_path,
        paths.fallback_requests_path,
        paths.fallback_raw_path,
        paths.final_path,
    )
    return paths if all(path.exists() for path in required) else None


def _resolve_downstream_paths(run_root: Path) -> DownstreamPaths | None:
    if len(run_root.parents) < 3:
        return None
    sample_root = run_root.parents[2]
    root = sample_root / "results" / LM2011_POST_REFINITIV_OUTPUT_DIRNAME
    paths = DownstreamPaths(
        root=root,
        sample_backbone_path=root / "lm2011_sample_backbone.parquet",
        event_panel_path=root / "lm2011_event_panel.parquet",
        return_regression_panel_full_10k_path=root / "lm2011_return_regression_panel_full_10k.parquet",
        manifest_path=root / "lm2011_sample_run_manifest.json",
    )
    required = (
        paths.sample_backbone_path,
        paths.event_panel_path,
        paths.return_regression_panel_full_10k_path,
    )
    return paths if all(path.exists() for path in required) else None


def _discover_step1_snapshot_paths(run_root: Path) -> list[Step1SnapshotPaths]:
    snapshots: list[Step1SnapshotPaths] = []
    for root in sorted(run_root.glob("refinitiv_step1_old*")):
        if not root.is_dir():
            continue
        ownership_universe_dir = root / "ownership_universe_common_stock"
        ownership_authority_dir = root / "ownership_authority_common_stock"
        snapshot = Step1SnapshotPaths(
            label=root.name,
            root=root,
            ownership_handoff_path=ownership_universe_dir / "refinitiv_ownership_universe_handoff_common_stock.parquet",
            ownership_results_path=ownership_universe_dir / "refinitiv_ownership_universe_results.parquet",
            ownership_row_summary_path=ownership_universe_dir / "refinitiv_ownership_universe_row_summary.parquet",
            authority_decisions_path=ownership_authority_dir / "refinitiv_permno_ownership_authority_decisions.parquet",
            authority_exceptions_path=ownership_authority_dir / "refinitiv_permno_ownership_authority_exceptions.parquet",
        )
        required = (
            snapshot.ownership_row_summary_path,
            snapshot.ownership_results_path,
            snapshot.authority_decisions_path,
            snapshot.authority_exceptions_path,
        )
        if all(path.exists() for path in required):
            snapshots.append(snapshot)
    return snapshots


def _ensure_required_inputs(paths: RunPaths) -> None:
    required = {
        "basis_path": paths.basis_path,
        "lookup_snapshot_path": paths.lookup_snapshot_path,
        "lookup_extended_path": paths.lookup_extended_path,
        "resolution_path": paths.resolution_path,
        "bridge_path": paths.bridge_path,
        "instrument_authority_path": paths.instrument_authority_path,
        "ownership_handoff_path": paths.ownership_handoff_path,
        "ownership_results_path": paths.ownership_results_path,
        "ownership_row_summary_path": paths.ownership_row_summary_path,
        "authority_decisions_path": paths.authority_decisions_path,
        "authority_exceptions_path": paths.authority_exceptions_path,
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        details = {name: str(required[name]) for name in missing}
        raise FileNotFoundError(f"Missing required Refinitiv lookup-basis artifacts: {json.dumps(details, indent=2)}")


def _scan_schema(path: Path) -> pl.Schema:
    return pl.scan_parquet(path).collect_schema()


def _ensure_columns(path: Path, label: str, required: Iterable[str]) -> pl.Schema:
    schema = _scan_schema(path)
    missing = [name for name in required if name not in schema]
    if missing:
        raise ValueError(f"{label} missing required columns {missing}: {path}")
    return schema


def _resolve_first_existing(columns: Iterable[str], candidates: Sequence[str], label: str) -> str:
    available = set(columns)
    for candidate in candidates:
        if candidate in available:
            return candidate
    raise ValueError(f"{label} missing any of expected columns: {list(candidates)}")


def _clean_text_expr(column: str, alias: str | None = None) -> pl.Expr:
    expr = pl.col(column).cast(pl.Utf8, strict=False).str.strip_chars()
    return (
        pl.when(expr.is_in(sorted(NULL_TEXT_VALUES)))
        .then(pl.lit(None, dtype=pl.Utf8))
        .otherwise(expr)
        .alias(alias or column)
    )


def _normalize_date_expr(column: str, alias: str | None = None) -> pl.Expr:
    return pl.col(column).cast(pl.Date, strict=False).alias(alias or column)


def _normalize_int_expr(column: str, alias: str | None = None) -> pl.Expr:
    return pl.col(column).cast(pl.Int64, strict=False).alias(alias or column)


def _year_filter_expr(start: int | None, end: int | None, column: str = "filing_date") -> pl.Expr:
    expr = pl.lit(True)
    if start is not None:
        expr = expr & (pl.col(column).dt.year() >= pl.lit(start))
    if end is not None:
        expr = expr & (pl.col(column).dt.year() <= pl.lit(end))
    return expr


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(message)


def _share(numerator: int, denominator: int) -> float | None:
    return (numerator / denominator) if denominator else None


def _semi_join_if_present(lf: pl.LazyFrame, ids_df: pl.DataFrame | None, key: str) -> pl.LazyFrame:
    return lf.join(ids_df.lazy(), on=key, how="semi") if ids_df is not None else lf


def _normalize_category(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if text in NULL_TEXT_VALUES else text


def _is_institutional_category(value: str | None) -> bool:
    return value is not None and value.casefold() == "Holdings by Institutions".casefold()


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_basis_df(paths: RunPaths, *, year_start: int | None, year_end: int | None) -> tuple[pl.DataFrame, dict[str, Any]]:
    schema = _ensure_columns(
        paths.basis_path,
        "basis filing artifact",
        required=("doc_id", "filing_date"),
    )
    permno_col = _resolve_first_existing(schema.keys(), ("KYPERMNO", "kypermno"), "basis filing artifact")
    gvkey_col = next(
        (name for name in ("gvkey_int", "gvkey", "KYGVKEY_final", "KYGVKEY", "KYGVKEY_ccm") if name in schema),
        None,
    )
    ticker_col = next((name for name in ("TICKER", "ticker") if name in schema), None)

    exprs: list[pl.Expr] = [
        _clean_text_expr("doc_id"),
        _normalize_date_expr("filing_date"),
        _normalize_int_expr(permno_col, "KYPERMNO"),
    ]
    if "cik_10" in schema:
        exprs.append(_clean_text_expr("cik_10"))
    if "accession_nodash" in schema:
        exprs.append(_clean_text_expr("accession_nodash"))
    if ticker_col is not None:
        exprs.append(_clean_text_expr(ticker_col, "basis_ticker"))
    else:
        exprs.append(pl.lit(None, dtype=pl.Utf8).alias("basis_ticker"))
    if gvkey_col is not None:
        exprs.append(_normalize_int_expr(gvkey_col, "gvkey_int"))
    else:
        exprs.append(pl.lit(None, dtype=pl.Int64).alias("gvkey_int"))

    basis_df = pl.read_parquet(paths.basis_path).select(exprs)
    if year_start is not None or year_end is not None:
        basis_df = basis_df.filter(_year_filter_expr(year_start, year_end))
    if basis_df.height == 0:
        raise ValueError("No basis filing rows remain after applying the requested year filter.")

    basis_df = basis_df.with_row_index("basis_row_id").with_columns(
        pl.col("KYPERMNO").is_not_null().alias("basis_has_permno"),
        pl.col("filing_date").is_not_null().alias("basis_has_filing_date"),
        pl.col("basis_ticker").is_not_null().alias("basis_has_ticker"),
        pl.col("gvkey_int").is_not_null().alias("basis_has_gvkey_int"),
        (pl.col("KYPERMNO").is_not_null() & pl.col("filing_date").is_not_null()).alias("basis_join_ready"),
    )

    metadata = {
        "basis_permno_source_column": permno_col,
        "basis_gvkey_source_column": gvkey_col,
        "basis_ticker_source_column": ticker_col,
    }
    return basis_df, metadata


def _read_bridge_df(paths: RunPaths) -> pl.DataFrame:
    schema = _ensure_columns(
        paths.bridge_path,
        "bridge universe artifact",
        required=("bridge_row_id", "KYPERMNO", "first_seen_caldt", "last_seen_caldt"),
    )
    return pl.read_parquet(paths.bridge_path).select(
        _clean_text_expr("bridge_row_id"),
        _normalize_int_expr("KYPERMNO"),
        _normalize_date_expr("first_seen_caldt"),
        _normalize_date_expr("last_seen_caldt"),
        _clean_text_expr("CUSIP") if "CUSIP" in schema else pl.lit(None, dtype=pl.Utf8).alias("CUSIP"),
        _clean_text_expr("ISIN") if "ISIN" in schema else pl.lit(None, dtype=pl.Utf8).alias("ISIN"),
        _clean_text_expr("TICKER") if "TICKER" in schema else pl.lit(None, dtype=pl.Utf8).alias("TICKER"),
        _clean_text_expr("vendor_primary_ric")
        if "vendor_primary_ric" in schema
        else pl.lit(None, dtype=pl.Utf8).alias("vendor_primary_ric"),
    ).with_columns(
        pl.any_horizontal(
            pl.col("CUSIP").is_not_null(),
            pl.col("ISIN").is_not_null(),
            pl.col("TICKER").is_not_null(),
        ).alias("bridge_has_lookup_identifier")
    )


def _read_lookup_stage_df(path: Path, *, ids_df: pl.DataFrame, label: str) -> pl.DataFrame:
    required = (
        "bridge_row_id",
        "KYPERMNO",
        "first_seen_caldt",
        "last_seen_caldt",
        "ISIN_success",
        "CUSIP_success",
        "TICKER_success",
    )
    _ensure_columns(path, label, required)
    return (
        pl.scan_parquet(path)
        .join(ids_df.lazy(), on="bridge_row_id", how="semi")
        .select(
            _clean_text_expr("bridge_row_id"),
            _normalize_int_expr("KYPERMNO"),
            _normalize_date_expr("first_seen_caldt"),
            _normalize_date_expr("last_seen_caldt"),
            pl.col("ISIN_success").cast(pl.Boolean, strict=False).alias("ISIN_success"),
            pl.col("CUSIP_success").cast(pl.Boolean, strict=False).alias("CUSIP_success"),
            pl.col("TICKER_success").cast(pl.Boolean, strict=False).alias("TICKER_success"),
        )
        .collect()
    )


def _read_resolution_df(paths: RunPaths, *, ids_df: pl.DataFrame) -> pl.DataFrame:
    schema = _ensure_columns(
        paths.resolution_path,
        "resolution artifact",
        required=(
            "bridge_row_id",
            "KYPERMNO",
            "first_seen_caldt",
            "last_seen_caldt",
            "ISIN_success",
            "CUSIP_success",
            "TICKER_success",
            "accepted_ric",
            "accepted_ric_source",
            "accepted_resolution_status",
            "conventional_identity_conflict",
            "ticker_candidate_ric",
            "ticker_candidate_available",
            "ticker_candidate_conflicts_with_conventional",
            "effective_collection_ric",
            "effective_collection_ric_source",
            "effective_resolution_status",
            "extension_status",
        ),
    )
    return (
        pl.scan_parquet(paths.resolution_path)
        .join(ids_df.lazy(), on="bridge_row_id", how="semi")
        .select(
            _clean_text_expr("bridge_row_id"),
            _normalize_int_expr("KYPERMNO"),
            _normalize_date_expr("first_seen_caldt"),
            _normalize_date_expr("last_seen_caldt"),
            _clean_text_expr("CUSIP") if "CUSIP" in schema else pl.lit(None, dtype=pl.Utf8).alias("CUSIP"),
            _clean_text_expr("ISIN") if "ISIN" in schema else pl.lit(None, dtype=pl.Utf8).alias("ISIN"),
            _clean_text_expr("TICKER") if "TICKER" in schema else pl.lit(None, dtype=pl.Utf8).alias("TICKER"),
            pl.col("ISIN_success").cast(pl.Boolean, strict=False).alias("ISIN_success"),
            pl.col("CUSIP_success").cast(pl.Boolean, strict=False).alias("CUSIP_success"),
            pl.col("TICKER_success").cast(pl.Boolean, strict=False).alias("TICKER_success"),
            _clean_text_expr("accepted_ric"),
            _clean_text_expr("accepted_ric_source"),
            _clean_text_expr("accepted_resolution_status"),
            pl.col("conventional_identity_conflict").cast(pl.Boolean, strict=False).alias("conventional_identity_conflict"),
            _clean_text_expr("ticker_candidate_ric"),
            pl.col("ticker_candidate_available").cast(pl.Boolean, strict=False).alias("ticker_candidate_available"),
            pl.col("ticker_candidate_conflicts_with_conventional")
            .cast(pl.Boolean, strict=False)
            .alias("ticker_candidate_conflicts_with_conventional"),
            _clean_text_expr("effective_collection_ric"),
            _clean_text_expr("effective_collection_ric_source"),
            _clean_text_expr("effective_resolution_status"),
            _clean_text_expr("extension_status"),
        )
        .collect()
    )


def _read_instrument_authority_df(paths: RunPaths, *, ids_df: pl.DataFrame) -> pl.DataFrame:
    schema = _ensure_columns(
        paths.instrument_authority_path,
        "instrument authority artifact",
        required=(
            "bridge_row_id",
            "KYPERMNO",
            "gvkey_int",
            "effective_collection_ric",
            "effective_resolution_status",
            "authority_eligible",
            "authority_exclusion_reason",
        ),
    )
    return (
        pl.scan_parquet(paths.instrument_authority_path)
        .join(ids_df.lazy(), on="bridge_row_id", how="semi")
        .select(
            _clean_text_expr("bridge_row_id"),
            _normalize_int_expr("KYPERMNO"),
            _normalize_int_expr("gvkey_int"),
            _clean_text_expr("gvkey_source_column")
            if "gvkey_source_column" in schema
            else pl.lit(None, dtype=pl.Utf8).alias("gvkey_source_column"),
            _clean_text_expr("effective_collection_ric"),
            _clean_text_expr("effective_resolution_status"),
            pl.col("authority_eligible").cast(pl.Boolean, strict=False).alias("authority_eligible"),
            _clean_text_expr("authority_exclusion_reason"),
        )
        .collect()
    )


def _read_handoff_df(paths: RunPaths, *, ids_df: pl.DataFrame) -> pl.DataFrame:
    schema = _ensure_columns(
        paths.ownership_handoff_path,
        "ownership handoff artifact",
        required=(
            "bridge_row_id",
            "KYPERMNO",
            "candidate_ric",
            "ownership_lookup_row_id",
            "ownership_lookup_role",
            "lookup_input_source",
            "request_start_date",
            "request_end_date",
            "retrieval_eligible",
            "retrieval_exclusion_reason",
            "accepted_ric",
            "accepted_ric_source",
            "accepted_resolution_status",
            "effective_collection_ric",
            "effective_collection_ric_source",
            "effective_resolution_status",
        ),
    )
    return (
        pl.scan_parquet(paths.ownership_handoff_path)
        .join(ids_df.lazy(), on="bridge_row_id", how="semi")
        .select(
            _clean_text_expr("bridge_row_id"),
            _normalize_int_expr("KYPERMNO"),
            _clean_text_expr("CUSIP") if "CUSIP" in schema else pl.lit(None, dtype=pl.Utf8).alias("CUSIP"),
            _clean_text_expr("ISIN") if "ISIN" in schema else pl.lit(None, dtype=pl.Utf8).alias("ISIN"),
            _clean_text_expr("TICKER") if "TICKER" in schema else pl.lit(None, dtype=pl.Utf8).alias("TICKER"),
            _normalize_date_expr("first_seen_caldt") if "first_seen_caldt" in schema else pl.lit(None, dtype=pl.Date).alias("first_seen_caldt"),
            _normalize_date_expr("last_seen_caldt") if "last_seen_caldt" in schema else pl.lit(None, dtype=pl.Date).alias("last_seen_caldt"),
            _clean_text_expr("candidate_ric"),
            _clean_text_expr("ownership_lookup_row_id"),
            _clean_text_expr("ownership_lookup_role"),
            _clean_text_expr("lookup_input_source"),
            _normalize_date_expr("request_start_date"),
            _normalize_date_expr("request_end_date"),
            pl.col("retrieval_eligible").cast(pl.Boolean, strict=False).alias("retrieval_eligible"),
            _clean_text_expr("retrieval_exclusion_reason"),
            _clean_text_expr("accepted_ric"),
            _clean_text_expr("accepted_ric_source"),
            _clean_text_expr("accepted_resolution_status"),
            _clean_text_expr("effective_collection_ric"),
            _clean_text_expr("effective_collection_ric_source"),
            _clean_text_expr("effective_resolution_status"),
        )
        .collect()
    )


def _read_row_summary_df(paths: RunPaths, *, ids_df: pl.DataFrame) -> pl.DataFrame:
    schema = _ensure_columns(
        paths.ownership_row_summary_path,
        "ownership row summary artifact",
        required=(
            "bridge_row_id",
            "KYPERMNO",
            "candidate_ric",
            "ownership_lookup_row_id",
            "ownership_lookup_role",
            "lookup_input_source",
            "request_start_date",
            "request_end_date",
            "retrieval_eligible",
            "retrieval_exclusion_reason",
            "retrieval_row_present",
            "ownership_rows_returned",
            "ownership_nonnull_value_count",
            "ownership_first_date",
            "ownership_last_date",
            "ownership_distinct_categories",
            "ownership_single_returned_ric",
            "ownership_returned_ric_nunique",
            "accepted_ric",
            "accepted_ric_source",
            "accepted_resolution_status",
            "effective_collection_ric",
            "effective_collection_ric_source",
            "effective_resolution_status",
        ),
    )
    return (
        pl.scan_parquet(paths.ownership_row_summary_path)
        .join(ids_df.lazy(), on="bridge_row_id", how="semi")
        .select(
            _clean_text_expr("bridge_row_id"),
            _normalize_int_expr("KYPERMNO"),
            _clean_text_expr("CUSIP") if "CUSIP" in schema else pl.lit(None, dtype=pl.Utf8).alias("CUSIP"),
            _clean_text_expr("ISIN") if "ISIN" in schema else pl.lit(None, dtype=pl.Utf8).alias("ISIN"),
            _clean_text_expr("TICKER") if "TICKER" in schema else pl.lit(None, dtype=pl.Utf8).alias("TICKER"),
            _normalize_date_expr("first_seen_caldt") if "first_seen_caldt" in schema else pl.lit(None, dtype=pl.Date).alias("first_seen_caldt"),
            _normalize_date_expr("last_seen_caldt") if "last_seen_caldt" in schema else pl.lit(None, dtype=pl.Date).alias("last_seen_caldt"),
            _clean_text_expr("candidate_ric"),
            _clean_text_expr("ownership_lookup_row_id"),
            _clean_text_expr("ownership_lookup_role"),
            _clean_text_expr("lookup_input_source"),
            _normalize_date_expr("request_start_date"),
            _normalize_date_expr("request_end_date"),
            pl.col("retrieval_eligible").cast(pl.Boolean, strict=False).alias("retrieval_eligible"),
            _clean_text_expr("retrieval_exclusion_reason"),
            pl.col("retrieval_row_present").cast(pl.Boolean, strict=False).alias("retrieval_row_present"),
            pl.col("ownership_rows_returned").cast(pl.Int64, strict=False).fill_null(0).alias("ownership_rows_returned"),
            pl.col("ownership_nonnull_value_count")
            .cast(pl.Int64, strict=False)
            .fill_null(0)
            .alias("ownership_nonnull_value_count"),
            _normalize_date_expr("ownership_first_date"),
            _normalize_date_expr("ownership_last_date"),
            pl.col("ownership_distinct_categories")
            .cast(pl.Int64, strict=False)
            .fill_null(0)
            .alias("ownership_distinct_categories"),
            pl.col("ownership_single_returned_ric")
            .cast(pl.Boolean, strict=False)
            .fill_null(False)
            .alias("ownership_single_returned_ric"),
            pl.col("ownership_returned_ric_nunique")
            .cast(pl.Int64, strict=False)
            .fill_null(0)
            .alias("ownership_returned_ric_nunique"),
            _clean_text_expr("accepted_ric"),
            _clean_text_expr("accepted_ric_source"),
            _clean_text_expr("accepted_resolution_status"),
            _clean_text_expr("effective_collection_ric"),
            _clean_text_expr("effective_collection_ric_source"),
            _clean_text_expr("effective_resolution_status"),
        )
        .collect()
    )


def _read_results_df(paths: RunPaths, *, ids_df: pl.DataFrame) -> pl.DataFrame:
    _ensure_columns(
        paths.ownership_results_path,
        "ownership results artifact",
        required=(
            "bridge_row_id",
            "KYPERMNO",
            "candidate_ric",
            "ownership_lookup_row_id",
            "ownership_lookup_role",
            "returned_ric",
            "returned_date",
            "returned_category",
            "returned_value",
        ),
    )
    return (
        pl.scan_parquet(paths.ownership_results_path)
        .join(ids_df.lazy(), on="bridge_row_id", how="semi")
        .select(
            _clean_text_expr("bridge_row_id"),
            _normalize_int_expr("KYPERMNO"),
            _clean_text_expr("candidate_ric"),
            _clean_text_expr("ownership_lookup_row_id"),
            _clean_text_expr("ownership_lookup_role"),
            _clean_text_expr("returned_ric"),
            _normalize_date_expr("returned_date"),
            _clean_text_expr("returned_category"),
            pl.col("returned_value").cast(pl.Float64, strict=False).alias("returned_value"),
        )
        .collect()
    )


def _read_authority_decisions_df(paths: RunPaths, *, permnos_df: pl.DataFrame) -> pl.DataFrame:
    schema = _ensure_columns(
        paths.authority_decisions_path,
        "ownership authority decisions artifact",
        required=(
            "KYPERMNO",
            "authoritative_ric",
            "authoritative_source_family",
            "authoritative_component_id",
            "authority_decision_status",
            "authority_decision_reason",
            "requires_review",
        ),
    )
    review_flag_cols = [name for name in schema if name.startswith("review_flag_")]
    return (
        pl.scan_parquet(paths.authority_decisions_path)
        .with_columns(_normalize_int_expr("KYPERMNO"))
        .join(permnos_df.lazy(), on="KYPERMNO", how="semi")
        .select(
            _normalize_int_expr("KYPERMNO"),
            _clean_text_expr("authoritative_ric"),
            _clean_text_expr("authoritative_source_family"),
            _clean_text_expr("authoritative_component_id"),
            _clean_text_expr("authority_decision_status"),
            _clean_text_expr("authority_decision_reason"),
            pl.col("requires_review").cast(pl.Boolean, strict=False).fill_null(False).alias("requires_review"),
            *[
                pl.col(name).cast(pl.Boolean, strict=False).fill_null(False).alias(name)
                for name in review_flag_cols
            ],
        )
        .collect()
    )


def _read_authority_exceptions_df(paths: RunPaths, *, permnos_df: pl.DataFrame) -> pl.DataFrame:
    schema = _ensure_columns(
        paths.authority_exceptions_path,
        "ownership authority exceptions artifact",
        required=(
            "KYPERMNO",
            "authoritative_ric",
            "authoritative_component_id",
            "authority_exception_status",
            "authority_exception_reason",
        ),
    )
    return (
        pl.scan_parquet(paths.authority_exceptions_path)
        .with_columns(_normalize_int_expr("KYPERMNO"))
        .join(permnos_df.lazy(), on="KYPERMNO", how="semi")
        .select(
            _normalize_int_expr("KYPERMNO"),
            _clean_text_expr("authoritative_ric"),
            _clean_text_expr("authoritative_component_id"),
            _normalize_date_expr("authority_window_start_date")
            if "authority_window_start_date" in schema
            else pl.lit(None, dtype=pl.Date).alias("authority_window_start_date"),
            _normalize_date_expr("authority_window_end_date")
            if "authority_window_end_date" in schema
            else pl.lit(None, dtype=pl.Date).alias("authority_window_end_date"),
            _clean_text_expr("authority_exception_status"),
            _clean_text_expr("authority_exception_reason"),
        )
        .collect()
    )


def _read_row_summary_from_path(path: Path, *, ids_df: pl.DataFrame | None = None) -> pl.DataFrame:
    schema = _ensure_columns(
        path,
        "ownership row summary artifact",
        required=(
            "bridge_row_id",
            "KYPERMNO",
            "candidate_ric",
            "ownership_lookup_row_id",
            "ownership_lookup_role",
            "lookup_input_source",
            "request_start_date",
            "request_end_date",
            "retrieval_eligible",
            "retrieval_exclusion_reason",
            "retrieval_row_present",
            "ownership_rows_returned",
            "ownership_nonnull_value_count",
            "ownership_first_date",
            "ownership_last_date",
            "ownership_distinct_categories",
            "ownership_single_returned_ric",
            "ownership_returned_ric_nunique",
            "accepted_ric",
            "accepted_ric_source",
            "accepted_resolution_status",
            "effective_collection_ric",
            "effective_collection_ric_source",
            "effective_resolution_status",
        ),
    )
    return (
        _semi_join_if_present(pl.scan_parquet(path), ids_df, "bridge_row_id")
        .select(
            _clean_text_expr("bridge_row_id"),
            _normalize_int_expr("KYPERMNO"),
            _clean_text_expr("CUSIP") if "CUSIP" in schema else pl.lit(None, dtype=pl.Utf8).alias("CUSIP"),
            _clean_text_expr("ISIN") if "ISIN" in schema else pl.lit(None, dtype=pl.Utf8).alias("ISIN"),
            _clean_text_expr("TICKER") if "TICKER" in schema else pl.lit(None, dtype=pl.Utf8).alias("TICKER"),
            _normalize_date_expr("first_seen_caldt") if "first_seen_caldt" in schema else pl.lit(None, dtype=pl.Date).alias("first_seen_caldt"),
            _normalize_date_expr("last_seen_caldt") if "last_seen_caldt" in schema else pl.lit(None, dtype=pl.Date).alias("last_seen_caldt"),
            _clean_text_expr("candidate_ric"),
            _clean_text_expr("ownership_lookup_row_id"),
            _clean_text_expr("ownership_lookup_role"),
            _clean_text_expr("lookup_input_source"),
            _normalize_date_expr("request_start_date"),
            _normalize_date_expr("request_end_date"),
            pl.col("retrieval_eligible").cast(pl.Boolean, strict=False).alias("retrieval_eligible"),
            _clean_text_expr("retrieval_exclusion_reason"),
            pl.col("retrieval_row_present").cast(pl.Boolean, strict=False).alias("retrieval_row_present"),
            pl.col("ownership_rows_returned").cast(pl.Int64, strict=False).fill_null(0).alias("ownership_rows_returned"),
            pl.col("ownership_nonnull_value_count").cast(pl.Int64, strict=False).fill_null(0).alias("ownership_nonnull_value_count"),
            _normalize_date_expr("ownership_first_date"),
            _normalize_date_expr("ownership_last_date"),
            pl.col("ownership_distinct_categories").cast(pl.Int64, strict=False).fill_null(0).alias("ownership_distinct_categories"),
            pl.col("ownership_single_returned_ric").cast(pl.Boolean, strict=False).fill_null(False).alias("ownership_single_returned_ric"),
            pl.col("ownership_returned_ric_nunique").cast(pl.Int64, strict=False).fill_null(0).alias("ownership_returned_ric_nunique"),
            _clean_text_expr("accepted_ric"),
            _clean_text_expr("accepted_ric_source"),
            _clean_text_expr("accepted_resolution_status"),
            _clean_text_expr("effective_collection_ric"),
            _clean_text_expr("effective_collection_ric_source"),
            _clean_text_expr("effective_resolution_status"),
        )
        .collect()
    )


def _read_results_from_path(path: Path, *, ids_df: pl.DataFrame | None = None) -> pl.DataFrame:
    _ensure_columns(
        path,
        "ownership results artifact",
        required=(
            "bridge_row_id",
            "KYPERMNO",
            "candidate_ric",
            "ownership_lookup_row_id",
            "ownership_lookup_role",
            "returned_ric",
            "returned_date",
            "returned_category",
            "returned_value",
        ),
    )
    return (
        _semi_join_if_present(pl.scan_parquet(path), ids_df, "bridge_row_id")
        .select(
            _clean_text_expr("bridge_row_id"),
            _normalize_int_expr("KYPERMNO"),
            _clean_text_expr("candidate_ric"),
            _clean_text_expr("ownership_lookup_row_id"),
            _clean_text_expr("ownership_lookup_role"),
            _clean_text_expr("returned_ric"),
            _normalize_date_expr("returned_date"),
            _clean_text_expr("returned_category"),
            pl.col("returned_value").cast(pl.Float64, strict=False).alias("returned_value"),
        )
        .collect()
    )


def _read_authority_decisions_from_path(path: Path, *, permnos_df: pl.DataFrame | None = None) -> pl.DataFrame:
    schema = _ensure_columns(
        path,
        "ownership authority decisions artifact",
        required=(
            "KYPERMNO",
            "authoritative_ric",
            "authoritative_source_family",
            "authoritative_component_id",
            "authority_decision_status",
            "authority_decision_reason",
            "requires_review",
        ),
    )
    review_flag_cols = [name for name in schema if name.startswith("review_flag_")]
    return (
        _semi_join_if_present(
            pl.scan_parquet(path).with_columns(_normalize_int_expr("KYPERMNO")),
            permnos_df,
            "KYPERMNO",
        )
        .select(
            _normalize_int_expr("KYPERMNO"),
            _clean_text_expr("authoritative_ric"),
            _clean_text_expr("authoritative_source_family"),
            _clean_text_expr("authoritative_component_id"),
            _clean_text_expr("authority_decision_status"),
            _clean_text_expr("authority_decision_reason"),
            pl.col("requires_review").cast(pl.Boolean, strict=False).fill_null(False).alias("requires_review"),
            *[
                pl.col(name).cast(pl.Boolean, strict=False).fill_null(False).alias(name)
                for name in review_flag_cols
            ],
        )
        .collect()
    )


def _read_authority_exceptions_from_path(path: Path, *, permnos_df: pl.DataFrame | None = None) -> pl.DataFrame:
    schema = _ensure_columns(
        path,
        "ownership authority exceptions artifact",
        required=(
            "KYPERMNO",
            "authoritative_ric",
            "authoritative_component_id",
            "authority_exception_status",
            "authority_exception_reason",
        ),
    )
    return (
        _semi_join_if_present(
            pl.scan_parquet(path).with_columns(_normalize_int_expr("KYPERMNO")),
            permnos_df,
            "KYPERMNO",
        )
        .select(
            _normalize_int_expr("KYPERMNO"),
            _clean_text_expr("authoritative_ric"),
            _clean_text_expr("authoritative_component_id"),
            _normalize_date_expr("authority_window_start_date")
            if "authority_window_start_date" in schema
            else pl.lit(None, dtype=pl.Date).alias("authority_window_start_date"),
            _normalize_date_expr("authority_window_end_date")
            if "authority_window_end_date" in schema
            else pl.lit(None, dtype=pl.Date).alias("authority_window_end_date"),
            _clean_text_expr("authority_exception_status"),
            _clean_text_expr("authority_exception_reason"),
        )
        .collect()
    )


def _read_doc_request_df(path: Path) -> pl.DataFrame:
    _ensure_columns(
        path,
        "doc ownership request artifact",
        required=(
            "doc_id",
            "filing_date",
            "KYPERMNO",
            "authoritative_ric",
            "authority_decision_status",
            "target_quarter_end",
            "target_effective_date",
            "fallback_window_start",
            "fallback_window_end",
            "retrieval_eligible",
            "retrieval_exclusion_reason",
        ),
    )
    return pl.read_parquet(path).select(
        _clean_text_expr("doc_id"),
        _normalize_date_expr("filing_date"),
        _normalize_int_expr("KYPERMNO"),
        _clean_text_expr("authoritative_ric"),
        _clean_text_expr("authority_decision_status"),
        _normalize_date_expr("target_quarter_end"),
        _normalize_date_expr("target_effective_date"),
        _normalize_date_expr("fallback_window_start"),
        _normalize_date_expr("fallback_window_end"),
        pl.col("retrieval_eligible").cast(pl.Boolean, strict=False).fill_null(False).alias("retrieval_eligible"),
        _clean_text_expr("retrieval_exclusion_reason"),
    )


def _read_doc_raw_df(path: Path) -> pl.DataFrame:
    _ensure_columns(
        path,
        "doc ownership raw artifact",
        required=(
            "doc_id",
            "filing_date",
            "KYPERMNO",
            "authoritative_ric",
            "authority_decision_status",
            "target_quarter_end",
            "target_effective_date",
            "request_stage",
            "response_date",
            "returned_category",
            "returned_value",
            "is_institutional_category",
        ),
    )
    return pl.read_parquet(path).select(
        _clean_text_expr("doc_id"),
        _normalize_date_expr("filing_date"),
        _normalize_int_expr("KYPERMNO"),
        _clean_text_expr("authoritative_ric"),
        _clean_text_expr("authority_decision_status"),
        _normalize_date_expr("target_quarter_end"),
        _normalize_date_expr("target_effective_date"),
        _clean_text_expr("request_stage"),
        _normalize_date_expr("response_date"),
        _clean_text_expr("returned_category"),
        _clean_text_expr("returned_category_normalized")
        if "returned_category_normalized" in _scan_schema(path)
        else _clean_text_expr("returned_category", "returned_category_normalized"),
        pl.col("returned_value").cast(pl.Float64, strict=False).alias("returned_value"),
        pl.col("is_institutional_category").cast(pl.Boolean, strict=False).fill_null(False).alias("is_institutional_category"),
    )


def _read_doc_final_df(path: Path) -> pl.DataFrame:
    _ensure_columns(
        path,
        "doc ownership final artifact",
        required=(
            "doc_id",
            "filing_date",
            "KYPERMNO",
            "authoritative_ric",
            "authority_decision_status",
            "target_quarter_end",
            "target_effective_date",
            "institutional_ownership_pct",
            "retrieval_status",
            "fallback_used",
        ),
    )
    return pl.read_parquet(path).select(
        _clean_text_expr("doc_id"),
        _normalize_date_expr("filing_date"),
        _normalize_int_expr("KYPERMNO"),
        _clean_text_expr("authoritative_ric"),
        _clean_text_expr("authority_decision_status"),
        _normalize_date_expr("target_quarter_end"),
        _normalize_date_expr("target_effective_date"),
        _normalize_date_expr("selected_response_date")
        if "selected_response_date" in _scan_schema(path)
        else pl.lit(None, dtype=pl.Date).alias("selected_response_date"),
        pl.col("institutional_ownership_pct").cast(pl.Float64, strict=False).alias("institutional_ownership_pct"),
        _clean_text_expr("retrieval_status"),
        pl.col("fallback_used").cast(pl.Boolean, strict=False).fill_null(False).alias("fallback_used"),
    )


def _build_basis_bridge_matches(basis_df: pl.DataFrame, bridge_df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    matches = (
        basis_df.filter(pl.col("basis_join_ready"))
        .select("basis_row_id", "doc_id", "KYPERMNO", "filing_date")
        .join(
            bridge_df.select(
                "bridge_row_id",
                "KYPERMNO",
                "first_seen_caldt",
                "last_seen_caldt",
                "bridge_has_lookup_identifier",
            ),
            on="KYPERMNO",
            how="inner",
        )
        .filter(
            pl.col("filing_date").is_between(
                pl.col("first_seen_caldt"),
                pl.col("last_seen_caldt"),
                closed="both",
            )
        )
        .select(
            "basis_row_id",
            "doc_id",
            "KYPERMNO",
            "filing_date",
            "bridge_row_id",
            "bridge_has_lookup_identifier",
        )
        .unique(subset=["basis_row_id", "bridge_row_id"], maintain_order=True)
    )
    matched_bridge_ids_df = matches.select("bridge_row_id").unique()
    matched_permnos_df = basis_df.select("KYPERMNO").drop_nulls().unique()
    return matches, matched_bridge_ids_df, matched_permnos_df


def _artifact_stage_stats(
    *,
    name: str,
    df: pl.DataFrame,
    date_min_col: str | None = None,
    date_max_col: str | None = None,
    ric_col: str | None = None,
    note: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "stage": name,
        "rows": int(df.height),
        "distinct_doc_id": int(df.get_column("doc_id").drop_nulls().n_unique()) if "doc_id" in df.columns else None,
        "distinct_bridge_row_id": int(df.get_column("bridge_row_id").drop_nulls().n_unique())
        if "bridge_row_id" in df.columns
        else None,
        "distinct_permno": int(df.get_column("KYPERMNO").drop_nulls().n_unique()) if "KYPERMNO" in df.columns else None,
        "distinct_ric": int(df.get_column(ric_col).drop_nulls().n_unique()) if ric_col and ric_col in df.columns else None,
        "note": note,
    }
    row["date_min"] = df.select(pl.col(date_min_col).min()).item() if date_min_col and date_min_col in df.columns else None
    row["date_max"] = df.select(pl.col(date_max_col).max()).item() if date_max_col and date_max_col in df.columns else None
    return row


def _set_contains_check(*, parent_df: pl.DataFrame, child_df: pl.DataFrame, key: str, check: str) -> dict[str, Any]:
    parent = set(parent_df.get_column(key).drop_nulls().to_list()) if key in parent_df.columns else set()
    child = set(child_df.get_column(key).drop_nulls().to_list()) if key in child_df.columns else set()
    missing = sorted(child - parent)
    return {
        "check": check,
        "key": key,
        "passed": not missing,
        "parent_count": len(parent),
        "child_count": len(child),
        "child_only_count": len(missing),
        "child_only_examples": [str(value) for value in missing[:5]],
    }


def _coverage_row(
    *,
    stage: str,
    basis_df: pl.DataFrame,
    covered_basis_ids_df: pl.DataFrame,
    note: str,
) -> dict[str, Any]:
    covered_df = basis_df.join(covered_basis_ids_df, on="basis_row_id", how="semi")
    total_rows = basis_df.height
    total_permnos = int(basis_df.get_column("KYPERMNO").drop_nulls().n_unique())
    covered_rows = covered_df.height
    covered_permnos = int(covered_df.get_column("KYPERMNO").drop_nulls().n_unique())
    left_df = basis_df.join(covered_basis_ids_df, on="basis_row_id", how="anti")
    return {
        "stage": stage,
        "rows": covered_rows,
        "row_share": (covered_rows / total_rows) if total_rows else None,
        "distinct_permno": covered_permnos,
        "permno_share": (covered_permnos / total_permnos) if total_permnos else None,
        "left_rows": left_df.height,
        "left_distinct_permno": int(left_df.get_column("KYPERMNO").drop_nulls().n_unique()),
        "left_top_filing_years": _top_counts(
            left_df.with_columns(pl.col("filing_date").dt.year().alias("filing_year")),
            "filing_year",
        ),
        "note": note,
    }


def _top_counts(df: pl.DataFrame, column: str, *, max_rows: int = MAX_TOP_ROWS) -> list[dict[str, Any]]:
    if column not in df.columns or df.height == 0:
        return []
    grouped = (
        df.group_by(column)
        .len()
        .rename({"len": "count"})
        .sort("count", descending=True)
        .head(max_rows)
        .to_dicts()
    )
    rows: list[dict[str, Any]] = []
    for row in grouped:
        value = row[column]
        rows.append(
            {
                column: value.isoformat() if isinstance(value, (dt.date, dt.datetime)) else value,
                "count": int(row["count"]),
            }
        )
    return rows


def _group_shares(df: pl.DataFrame, column: str, *, max_rows: int = MAX_TOP_ROWS) -> list[dict[str, Any]]:
    if column not in df.columns or df.height == 0:
        return []
    grouped = (
        df.group_by(column)
        .len()
        .rename({"len": "count"})
        .sort("count", descending=True)
        .with_columns((pl.col("count") / pl.lit(df.height)).alias("share"))
        .head(max_rows)
        .to_dicts()
    )
    rows: list[dict[str, Any]] = []
    for row in grouped:
        value = row[column]
        rows.append(
            {
                column: value.isoformat() if isinstance(value, (dt.date, dt.datetime)) else value,
                "count": int(row["count"]),
                "share": float(row["share"]),
            }
        )
    return rows


def _summarize_request_outcomes(df: pl.DataFrame, group_col: str, *, max_rows: int = MAX_TOP_ROWS) -> list[dict[str, Any]]:
    if group_col not in df.columns or df.height == 0:
        return []
    grouped = (
        df.group_by(group_col)
        .agg(
            pl.len().alias("request_rows"),
            pl.col("KYPERMNO").drop_nulls().n_unique().alias("distinct_permno"),
            pl.col("candidate_ric").drop_nulls().n_unique().alias("distinct_candidate_ric"),
            (pl.col("ownership_rows_returned") > 0).cast(pl.Int64).sum().alias("rows_with_results"),
            (pl.col("ownership_nonnull_value_count") > 0).cast(pl.Int64).sum().alias("rows_with_nonnull"),
            (pl.col("retrieval_row_present").fill_null(False)).cast(pl.Int64).sum().alias("retrieval_row_present_count"),
        )
        .with_columns(
            (pl.col("rows_with_results") / pl.col("request_rows")).alias("result_share"),
            (pl.col("rows_with_nonnull") / pl.col("request_rows")).alias("nonnull_share"),
        )
        .sort("request_rows", descending=True)
        .head(max_rows)
        .to_dicts()
    )
    rows: list[dict[str, Any]] = []
    for row in grouped:
        value = row[group_col]
        rows.append(
            {
                group_col: value.isoformat() if isinstance(value, (dt.date, dt.datetime)) else value,
                "request_rows": int(row["request_rows"]),
                "distinct_permno": int(row["distinct_permno"]),
                "distinct_candidate_ric": int(row["distinct_candidate_ric"]),
                "rows_with_results": int(row["rows_with_results"]),
                "rows_with_nonnull": int(row["rows_with_nonnull"]),
                "retrieval_row_present_count": int(row["retrieval_row_present_count"]),
                "result_share": float(row["result_share"]),
                "nonnull_share": float(row["nonnull_share"]),
            }
        )
    return rows


def _outcome_bucket_expr() -> pl.Expr:
    return (
        pl.when(~pl.col("retrieval_row_present").fill_null(False))
        .then(pl.lit("not_attempted"))
        .when(pl.col("ownership_rows_returned") == 0)
        .then(pl.lit("attempted_no_result_rows"))
        .when(pl.col("ownership_nonnull_value_count") == 0)
        .then(pl.lit("result_rows_but_all_values_null"))
        .otherwise(pl.lit("nonnull_values_returned"))
        .alias("outcome_bucket")
    )


def _build_request_outcomes_df(
    row_summary_df: pl.DataFrame,
    authority_decisions_df: pl.DataFrame,
    authority_exceptions_df: pl.DataFrame,
) -> pl.DataFrame:
    authority_exception_by_permno = authority_exceptions_df.group_by("KYPERMNO").agg(
        pl.len().alias("authority_exception_count"),
        pl.col("authority_exception_status").drop_nulls().first().alias("authority_exception_status"),
        pl.col("authority_exception_reason").drop_nulls().first().alias("authority_exception_reason"),
    )
    return (
        row_summary_df.join(
            authority_decisions_df.select(
                "KYPERMNO",
                "authoritative_ric",
                "authoritative_source_family",
                "authority_decision_status",
                "authority_decision_reason",
                "requires_review",
            ),
            on="KYPERMNO",
            how="left",
        )
        .join(authority_exception_by_permno, on="KYPERMNO", how="left")
        .with_columns(
            pl.col("authority_exception_count").cast(pl.Int64, strict=False).fill_null(0),
            pl.col("requires_review").cast(pl.Boolean, strict=False).fill_null(False),
            (pl.col("request_end_date") - pl.col("request_start_date")).dt.total_days().alias("request_span_days"),
        )
        .with_columns(
            pl.when(pl.col("request_span_days").is_null())
            .then(pl.lit(None, dtype=pl.Utf8))
            .when(pl.col("request_span_days") <= 365)
            .then(pl.lit("<=1y"))
            .when(pl.col("request_span_days") <= (365 * 3))
            .then(pl.lit("1-3y"))
            .when(pl.col("request_span_days") <= (365 * 7))
            .then(pl.lit("3-7y"))
            .otherwise(pl.lit(">7y"))
            .alias("request_span_bucket"),
            pl.col("request_start_date").dt.year().alias("request_start_year"),
            _outcome_bucket_expr(),
        )
    )


def _normalize_results_categories(results_df: pl.DataFrame) -> pl.DataFrame:
    return results_df.with_columns(
        pl.col("returned_category")
        .map_elements(_normalize_category, return_dtype=pl.Utf8)
        .alias("returned_category_normalized")
    ).with_columns(
        pl.col("returned_category_normalized")
        .map_elements(_is_institutional_category, return_dtype=pl.Boolean)
        .fill_null(False)
        .alias("is_institutional_category")
    )


def _multiplicity_summary(df: pl.DataFrame, group_col: str, count_col: str, *, max_rows: int = MAX_TOP_ROWS) -> dict[str, Any]:
    if df.height == 0 or group_col not in df.columns or count_col not in df.columns:
        return {"distribution": [], "top_entities": []}
    distribution = (
        df.group_by(count_col)
        .len()
        .rename({"len": "entity_count"})
        .sort(count_col)
        .to_dicts()
    )
    top_entities = df.sort(count_col, descending=True).head(max_rows).to_dicts()
    for row in top_entities:
        for key, value in list(row.items()):
            if isinstance(value, (dt.date, dt.datetime)):
                row[key] = value.isoformat()
    return {
        "distribution": [
            {count_col: int(row[count_col]), "entity_count": int(row["entity_count"])}
            for row in distribution
        ],
        "top_entities": top_entities,
    }


def _stable_rank(seed: int, *parts: Any) -> str:
    payload = "|".join("" if part is None else str(part) for part in parts)
    return hashlib.sha256(f"{seed}|{payload}".encode("utf-8")).hexdigest()


def _candidate_bucket(row: dict[str, Any]) -> str:
    requires_review = bool(row.get("requires_review"))
    exception_count = int(row.get("authority_exception_count") or 0)
    candidate_ric_permno_count = int(row.get("candidate_ric_permno_count") or 0)
    permno_candidate_ric_count = int(row.get("permno_candidate_ric_count") or 0)
    retrieval_present = bool(row.get("retrieval_row_present"))
    ownership_nonnull = int(row.get("ownership_nonnull_value_count") or 0)
    effective_collection_ric = row.get("effective_collection_ric")
    candidate_ric = row.get("candidate_ric")
    lookup_role = row.get("ownership_lookup_role")
    authoritative_status = row.get("authority_decision_status")
    if requires_review or exception_count > 0 or authoritative_status in {
        "DATE_VARYING_CONVENTIONAL_EXCEPTION",
        "REVIEW_REQUIRED",
    }:
        return "authority_exception_or_review"
    if candidate_ric_permno_count > 1 or permno_candidate_ric_count > 1:
        return "multi_permno_or_alias_edge"
    if effective_collection_ric is not None and candidate_ric == effective_collection_ric and ownership_nonnull > 0:
        return "effective_with_successful_ownership"
    if effective_collection_ric is not None and candidate_ric == effective_collection_ric and retrieval_present:
        return "effective_with_failed_or_null_ownership"
    if retrieval_present and ownership_nonnull == 0:
        return "retrieval_attempt_no_useful_output"
    if lookup_role == "UNIVERSE_TARGET_TICKER_CANDIDATE":
        return "ticker_or_nonconventional_candidate"
    return "other_retrievable"


def _build_sample_table(
    *,
    request_outcomes_df: pl.DataFrame,
    basis_bridge_context_df: pl.DataFrame,
    authority_decisions_df: pl.DataFrame,
    authority_exceptions_df: pl.DataFrame,
    sample_size: int,
    seed: int,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    if request_outcomes_df.height == 0 or sample_size == 0:
        return pl.DataFrame(), {"bucket_counts_available": {}, "bucket_counts_sampled": {}}

    candidate_ric_permno_df = (
        request_outcomes_df.filter(pl.col("candidate_ric").is_not_null())
        .group_by("candidate_ric")
        .agg(
            pl.col("KYPERMNO").drop_nulls().n_unique().alias("candidate_ric_permno_count"),
            pl.col("bridge_row_id").drop_nulls().n_unique().alias("candidate_ric_bridge_row_count"),
        )
    )
    permno_candidate_ric_df = (
        request_outcomes_df.filter(pl.col("candidate_ric").is_not_null())
        .group_by("KYPERMNO")
        .agg(
            pl.col("candidate_ric").drop_nulls().n_unique().alias("permno_candidate_ric_count"),
            pl.col("ownership_lookup_role").drop_nulls().n_unique().alias("permno_lookup_role_count"),
        )
    )
    authority_exception_by_permno = authority_exceptions_df.group_by("KYPERMNO").agg(
        pl.len().alias("authority_exception_count"),
        pl.col("authority_exception_status").drop_nulls().first().alias("authority_exception_status"),
        pl.col("authority_exception_reason").drop_nulls().first().alias("authority_exception_reason"),
    )

    sample_base_df = (
        request_outcomes_df.filter(pl.col("candidate_ric").is_not_null())
        .join(basis_bridge_context_df, on="bridge_row_id", how="left")
        .join(
            authority_decisions_df.select(
                "KYPERMNO",
                "authoritative_ric",
                "authoritative_source_family",
                "authority_decision_status",
                "authority_decision_reason",
                "requires_review",
            ),
            on="KYPERMNO",
            how="left",
        )
        .join(authority_exception_by_permno, on="KYPERMNO", how="left")
        .join(candidate_ric_permno_df, on="candidate_ric", how="left")
        .join(permno_candidate_ric_df, on="KYPERMNO", how="left")
        .with_columns(
            pl.col("authority_exception_count").cast(pl.Int64, strict=False).fill_null(0),
            pl.col("candidate_ric_permno_count").cast(pl.Int64, strict=False).fill_null(0),
            pl.col("candidate_ric_bridge_row_count").cast(pl.Int64, strict=False).fill_null(0),
            pl.col("permno_candidate_ric_count").cast(pl.Int64, strict=False).fill_null(0),
            pl.col("permno_lookup_role_count").cast(pl.Int64, strict=False).fill_null(0),
            pl.col("basis_row_count").cast(pl.Int64, strict=False).fill_null(0),
            pl.col("basis_doc_count").cast(pl.Int64, strict=False).fill_null(0),
            pl.col("requires_review").cast(pl.Boolean, strict=False).fill_null(False),
        )
    )

    bucketed_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in SAMPLE_BUCKET_ORDER}
    for row in sample_base_df.to_dicts():
        row["RIC"] = row.get("candidate_ric") or row.get("effective_collection_ric") or row.get("accepted_ric")
        if row["RIC"] is None:
            continue
        row["reason_bucket"] = _candidate_bucket(row)
        row["_stable_rank"] = _stable_rank(
            seed,
            row.get("RIC"),
            row.get("KYPERMNO"),
            row.get("bridge_row_id"),
            row.get("ownership_lookup_role"),
        )
        bucketed_rows[row["reason_bucket"]].append(row)

    available_counts = {
        bucket: len({row["RIC"] for row in rows})
        for bucket, rows in bucketed_rows.items()
        if rows
    }
    normalized_groups: dict[str, list[dict[str, Any]]] = {}
    for bucket, rows in bucketed_rows.items():
        deduped: dict[str, dict[str, Any]] = {}
        sorted_rows = sorted(
            rows,
            key=lambda row: (
                -int(bool(row.get("requires_review"))),
                -int(row.get("authority_exception_count") or 0),
                -int(row.get("candidate_ric_permno_count") or 0),
                -int(row.get("permno_candidate_ric_count") or 0),
                -int(row.get("basis_doc_count") or 0),
                -int(row.get("ownership_nonnull_value_count") or 0),
                -int(row.get("ownership_rows_returned") or 0),
                row["_stable_rank"],
            ),
        )
        for row in sorted_rows:
            deduped.setdefault(row["RIC"], row)
        normalized_groups[bucket] = list(deduped.values())

    active_buckets = [bucket for bucket in SAMPLE_BUCKET_ORDER if normalized_groups[bucket]]
    if not active_buckets:
        return pl.DataFrame(), {"bucket_counts_available": {}, "bucket_counts_sampled": {}}

    base_target = sample_size // len(active_buckets) if active_buckets else 0
    remainder = sample_size % len(active_buckets) if active_buckets else 0
    targets = {
        bucket: min(len(normalized_groups[bucket]), base_target + (1 if idx < remainder else 0))
        for idx, bucket in enumerate(active_buckets)
    }

    selected_rows: list[dict[str, Any]] = []
    selected_rics: set[str] = set()
    sampled_counts: dict[str, int] = {bucket: 0 for bucket in active_buckets}
    for bucket in active_buckets:
        for row in normalized_groups[bucket]:
            if sampled_counts[bucket] >= targets[bucket]:
                break
            if row["RIC"] in selected_rics:
                continue
            selected_rows.append(row)
            selected_rics.add(row["RIC"])
            sampled_counts[bucket] += 1

    if len(selected_rows) < sample_size:
        for bucket in active_buckets:
            for row in normalized_groups[bucket]:
                if len(selected_rows) >= sample_size:
                    break
                if row["RIC"] in selected_rics:
                    continue
                selected_rows.append(row)
                selected_rics.add(row["RIC"])
                sampled_counts[bucket] += 1
            if len(selected_rows) >= sample_size:
                break

    sample_columns = [
        "reason_bucket",
        "RIC",
        "KYPERMNO",
        "bridge_row_id",
        "request_start_date",
        "request_end_date",
        "basis_doc_count",
        "basis_min_filing_date",
        "basis_max_filing_date",
        "authority_decision_status",
        "authority_decision_reason",
        "authority_exception_status",
        "authority_exception_reason",
        "requires_review",
        "authoritative_ric",
        "authoritative_source_family",
        "candidate_ric",
        "effective_collection_ric",
        "effective_collection_ric_source",
        "accepted_ric",
        "accepted_ric_source",
        "ownership_lookup_role",
        "lookup_input_source",
        "retrieval_row_present",
        "ownership_rows_returned",
        "ownership_nonnull_value_count",
        "candidate_ric_permno_count",
        "permno_candidate_ric_count",
        "CUSIP",
        "ISIN",
        "TICKER",
    ]
    sample_df = pl.DataFrame(selected_rows).select(sample_columns) if selected_rows else pl.DataFrame()
    metadata = {
        "bucket_counts_available": available_counts,
        "bucket_counts_sampled": {bucket: count for bucket, count in sampled_counts.items() if count > 0},
        "active_buckets": active_buckets,
        "requested_sample_size": sample_size,
        "actual_sample_size": int(sample_df.height),
    }
    return sample_df, metadata


def _failure_bucket(row: dict[str, Any]) -> str:
    if bool(row.get("regressed_vs_any_snapshot")):
        return "regressed_vs_snapshot"
    status = row.get("authority_decision_status")
    if status == "REVIEW_REQUIRED" or bool(row.get("requires_review")):
        return "review_required"
    if status == "DATE_VARYING_CONVENTIONAL_EXCEPTION" or int(row.get("authority_exception_count") or 0) > 0:
        return "date_varying_exception"
    if status == "NO_CONVENTIONAL_AUTHORITY":
        return "no_conventional_authority"
    if int(row.get("candidate_ric_permno_count") or 0) > 1 or int(row.get("permno_candidate_ric_count") or 0) > 1:
        return "multi_permno_or_alias_edge"
    if status == "STATIC_CONVENTIONAL" and int(row.get("ownership_nonnull_value_count") or 0) == 0:
        return "static_conventional_zero_or_null"
    if status == "STATIC_CONVENTIONAL" and int(row.get("ownership_nonnull_value_count") or 0) > 0:
        return "static_conventional_success"
    return "other"


def _summarize_snapshot_diffs(
    *,
    current_request_outcomes_df: pl.DataFrame,
    current_authority_decisions_df: pl.DataFrame,
    snapshot_paths: Sequence[Step1SnapshotPaths],
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    current_rows_by_id = {
        str(row["ownership_lookup_row_id"]): row
        for row in current_request_outcomes_df.to_dicts()
        if row.get("ownership_lookup_row_id") is not None
    }
    current_status_by_permno = {
        int(row["KYPERMNO"]): row.get("authority_decision_status")
        for row in current_authority_decisions_df.select("KYPERMNO", "authority_decision_status").to_dicts()
        if row.get("KYPERMNO") is not None
    }
    flags: dict[str, dict[str, Any]] = {
        lookup_id: {
            "ownership_lookup_row_id": lookup_id,
            "regressed_vs_any_snapshot": False,
            "improved_vs_any_snapshot": False,
            "regressed_snapshot_labels": [],
            "improved_snapshot_labels": [],
            "snapshot_prev_max_nonnull_value_count": 0,
            "snapshot_prev_max_rows_returned": 0,
        }
        for lookup_id in current_rows_by_id
    }
    summaries: list[dict[str, Any]] = []

    for snapshot in snapshot_paths:
        snapshot_row_summary_df = _read_row_summary_from_path(snapshot.ownership_row_summary_path)
        snapshot_decisions_df = _read_authority_decisions_from_path(snapshot.authority_decisions_path)
        snapshot_exceptions_df = _read_authority_exceptions_from_path(snapshot.authority_exceptions_path)
        snapshot_request_outcomes_df = _build_request_outcomes_df(
            snapshot_row_summary_df,
            snapshot_decisions_df,
            snapshot_exceptions_df,
        )
        snapshot_rows_by_id = {
            str(row["ownership_lookup_row_id"]): row
            for row in snapshot_request_outcomes_df.to_dicts()
            if row.get("ownership_lookup_row_id") is not None
        }
        snapshot_status_by_permno = {
            int(row["KYPERMNO"]): row.get("authority_decision_status")
            for row in snapshot_decisions_df.select("KYPERMNO", "authority_decision_status").to_dicts()
            if row.get("KYPERMNO") is not None
        }
        current_ids = set(current_rows_by_id)
        snapshot_ids = set(snapshot_rows_by_id)
        overlap_ids = sorted(current_ids & snapshot_ids)
        only_current = sorted(current_ids - snapshot_ids)
        only_snapshot = sorted(snapshot_ids - current_ids)

        candidate_ric_changed = 0
        request_window_changed = 0
        nonnull_regressions: list[dict[str, Any]] = []
        nonnull_improvements = 0
        result_regressions = 0
        result_improvements = 0
        authority_status_changed = 0
        for lookup_id in overlap_ids:
            current_row = current_rows_by_id[lookup_id]
            snapshot_row = snapshot_rows_by_id[lookup_id]
            current_permno = current_row.get("KYPERMNO")
            if current_row.get("candidate_ric") != snapshot_row.get("candidate_ric"):
                candidate_ric_changed += 1
            if (
                current_row.get("request_start_date") != snapshot_row.get("request_start_date")
                or current_row.get("request_end_date") != snapshot_row.get("request_end_date")
            ):
                request_window_changed += 1
            current_nonnull = int(current_row.get("ownership_nonnull_value_count") or 0)
            snapshot_nonnull = int(snapshot_row.get("ownership_nonnull_value_count") or 0)
            current_results = int(current_row.get("ownership_rows_returned") or 0)
            snapshot_results = int(snapshot_row.get("ownership_rows_returned") or 0)
            if snapshot_nonnull > 0 and current_nonnull == 0:
                nonnull_regressions.append(
                    {
                        "ownership_lookup_row_id": lookup_id,
                        "KYPERMNO": current_permno,
                        "candidate_ric": current_row.get("candidate_ric"),
                        "ownership_lookup_role": current_row.get("ownership_lookup_role"),
                        "request_start_date": current_row.get("request_start_date"),
                        "request_end_date": current_row.get("request_end_date"),
                    }
                )
                flags[lookup_id]["regressed_vs_any_snapshot"] = True
                flags[lookup_id]["regressed_snapshot_labels"].append(snapshot.label)
            if snapshot_nonnull == 0 and current_nonnull > 0:
                nonnull_improvements += 1
                flags[lookup_id]["improved_vs_any_snapshot"] = True
                flags[lookup_id]["improved_snapshot_labels"].append(snapshot.label)
            if snapshot_results > 0 and current_results == 0:
                result_regressions += 1
            if snapshot_results == 0 and current_results > 0:
                result_improvements += 1
            flags[lookup_id]["snapshot_prev_max_nonnull_value_count"] = max(
                int(flags[lookup_id]["snapshot_prev_max_nonnull_value_count"]),
                snapshot_nonnull,
            )
            flags[lookup_id]["snapshot_prev_max_rows_returned"] = max(
                int(flags[lookup_id]["snapshot_prev_max_rows_returned"]),
                snapshot_results,
            )
            if current_permno is not None:
                current_status = current_status_by_permno.get(int(current_permno))
                snapshot_status = snapshot_status_by_permno.get(int(current_permno))
                if current_status is not None and snapshot_status is not None and current_status != snapshot_status:
                    authority_status_changed += 1

        summaries.append(
            {
                "snapshot_label": snapshot.label,
                "current_request_rows": int(current_request_outcomes_df.height),
                "snapshot_request_rows": int(snapshot_request_outcomes_df.height),
                "lookup_id_overlap_rows": len(overlap_ids),
                "rows_only_in_current": len(only_current),
                "rows_only_in_snapshot": len(only_snapshot),
                "candidate_ric_changed_rows": candidate_ric_changed,
                "request_window_changed_rows": request_window_changed,
                "nonnull_regression_rows": len(nonnull_regressions),
                "nonnull_improvement_rows": nonnull_improvements,
                "result_regression_rows": result_regressions,
                "result_improvement_rows": result_improvements,
                "authority_status_changed_overlap_rows": authority_status_changed,
                "top_regressed_candidate_rics": _top_counts(pl.DataFrame(nonnull_regressions), "candidate_ric"),
                "top_regressed_permnos": _top_counts(pl.DataFrame(nonnull_regressions), "KYPERMNO"),
            }
        )

    flags_rows = []
    for lookup_id, row in flags.items():
        flags_rows.append(
            {
                **row,
                "regressed_snapshot_labels": ",".join(sorted(set(row["regressed_snapshot_labels"]))),
                "improved_snapshot_labels": ",".join(sorted(set(row["improved_snapshot_labels"]))),
            }
        )
    return pl.DataFrame(flags_rows), summaries


def _build_failure_sample_table(
    *,
    request_outcomes_df: pl.DataFrame,
    authority_exceptions_df: pl.DataFrame,
    diff_flags_df: pl.DataFrame,
    sample_size: int,
    seed: int,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    if request_outcomes_df.height == 0 or sample_size == 0:
        return pl.DataFrame(), {"bucket_counts_available": {}, "bucket_counts_sampled": {}}

    candidate_ric_permno_df = (
        request_outcomes_df.filter(pl.col("candidate_ric").is_not_null())
        .group_by("candidate_ric")
        .agg(
            pl.col("KYPERMNO").drop_nulls().n_unique().alias("candidate_ric_permno_count"),
            pl.col("ownership_lookup_row_id").drop_nulls().n_unique().alias("candidate_ric_lookup_row_count"),
        )
    )
    permno_candidate_ric_df = (
        request_outcomes_df.filter(pl.col("candidate_ric").is_not_null())
        .group_by("KYPERMNO")
        .agg(pl.col("candidate_ric").drop_nulls().n_unique().alias("permno_candidate_ric_count"))
    )
    authority_exception_by_permno = authority_exceptions_df.group_by("KYPERMNO").agg(
        pl.len().alias("authority_exception_count"),
        pl.col("authority_exception_status").drop_nulls().first().alias("authority_exception_status"),
        pl.col("authority_exception_reason").drop_nulls().first().alias("authority_exception_reason"),
    )
    sample_base_df = (
        request_outcomes_df.filter(pl.col("candidate_ric").is_not_null())
        .join(authority_exception_by_permno, on="KYPERMNO", how="left")
        .join(candidate_ric_permno_df, on="candidate_ric", how="left")
        .join(permno_candidate_ric_df, on="KYPERMNO", how="left")
        .join(diff_flags_df, on="ownership_lookup_row_id", how="left")
        .with_columns(
            pl.col("authority_exception_count").cast(pl.Int64, strict=False).fill_null(0),
            pl.col("candidate_ric_permno_count").cast(pl.Int64, strict=False).fill_null(0),
            pl.col("candidate_ric_lookup_row_count").cast(pl.Int64, strict=False).fill_null(0),
            pl.col("permno_candidate_ric_count").cast(pl.Int64, strict=False).fill_null(0),
            pl.col("requires_review").cast(pl.Boolean, strict=False).fill_null(False),
            pl.col("regressed_vs_any_snapshot").cast(pl.Boolean, strict=False).fill_null(False),
            pl.col("improved_vs_any_snapshot").cast(pl.Boolean, strict=False).fill_null(False),
            pl.col("regressed_snapshot_labels").cast(pl.Utf8, strict=False).fill_null(""),
            pl.col("snapshot_prev_max_nonnull_value_count").cast(pl.Int64, strict=False).fill_null(0),
            pl.col("snapshot_prev_max_rows_returned").cast(pl.Int64, strict=False).fill_null(0),
        )
    )

    bucketed_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in FAILURE_SAMPLE_BUCKET_ORDER}
    for row in sample_base_df.to_dicts():
        row["RIC"] = row.get("candidate_ric") or row.get("authoritative_ric")
        if row["RIC"] is None:
            continue
        row["reason_bucket"] = _failure_bucket(row)
        row["_stable_rank"] = _stable_rank(
            seed,
            row.get("RIC"),
            row.get("KYPERMNO"),
            row.get("ownership_lookup_row_id"),
            row.get("regressed_snapshot_labels"),
        )
        bucketed_rows[row["reason_bucket"]].append(row)

    available_counts = {
        bucket: len({row["RIC"] for row in rows})
        for bucket, rows in bucketed_rows.items()
        if rows
    }
    active_buckets = [bucket for bucket in FAILURE_SAMPLE_BUCKET_ORDER if bucketed_rows[bucket]]
    if not active_buckets:
        return pl.DataFrame(), {"bucket_counts_available": {}, "bucket_counts_sampled": {}}

    base_target = sample_size // len(active_buckets)
    remainder = sample_size % len(active_buckets)
    targets = {
        bucket: base_target + (1 if idx < remainder else 0)
        for idx, bucket in enumerate(active_buckets)
    }

    selected_rows: list[dict[str, Any]] = []
    selected_rics: set[str] = set()
    sampled_counts: dict[str, int] = {bucket: 0 for bucket in active_buckets}
    for bucket in active_buckets:
        sorted_rows = sorted(
            bucketed_rows[bucket],
            key=lambda row: (
                -int(bool(row.get("regressed_vs_any_snapshot"))),
                -int(bool(row.get("requires_review"))),
                -int(row.get("authority_exception_count") or 0),
                -int(row.get("candidate_ric_permno_count") or 0),
                -int(row.get("permno_candidate_ric_count") or 0),
                -int(row.get("snapshot_prev_max_nonnull_value_count") or 0),
                -int(row.get("ownership_nonnull_value_count") or 0),
                row["_stable_rank"],
            ),
        )
        for row in sorted_rows:
            if sampled_counts[bucket] >= targets[bucket]:
                break
            if row["RIC"] in selected_rics:
                continue
            selected_rows.append(row)
            selected_rics.add(row["RIC"])
            sampled_counts[bucket] += 1

    if len(selected_rows) < sample_size:
        for bucket in active_buckets:
            for row in sorted(bucketed_rows[bucket], key=lambda row: row["_stable_rank"]):
                if len(selected_rows) >= sample_size:
                    break
                if row["RIC"] in selected_rics:
                    continue
                selected_rows.append(row)
                selected_rics.add(row["RIC"])
                sampled_counts[bucket] += 1

    sample_columns = [
        "reason_bucket",
        "RIC",
        "KYPERMNO",
        "bridge_row_id",
        "ownership_lookup_row_id",
        "authoritative_ric",
        "authority_decision_status",
        "authority_decision_reason",
        "authority_exception_status",
        "authority_exception_reason",
        "requires_review",
        "ownership_lookup_role",
        "lookup_input_source",
        "request_start_date",
        "request_end_date",
        "retrieval_row_present",
        "ownership_rows_returned",
        "ownership_nonnull_value_count",
        "outcome_bucket",
        "regressed_vs_any_snapshot",
        "regressed_snapshot_labels",
        "snapshot_prev_max_rows_returned",
        "snapshot_prev_max_nonnull_value_count",
        "candidate_ric",
        "candidate_ric_permno_count",
        "permno_candidate_ric_count",
        "CUSIP",
        "ISIN",
        "TICKER",
    ]
    sample_df = pl.DataFrame(selected_rows).select(sample_columns) if selected_rows else pl.DataFrame()
    metadata = {
        "bucket_counts_available": available_counts,
        "bucket_counts_sampled": {bucket: count for bucket, count in sampled_counts.items() if count > 0},
        "requested_sample_size": sample_size,
        "actual_sample_size": int(sample_df.height),
    }
    return sample_df, metadata


def _build_doc_ownership_truth_table(
    *,
    exact_requests_df: pl.DataFrame,
    exact_raw_df: pl.DataFrame,
    fallback_requests_df: pl.DataFrame,
    fallback_raw_df: pl.DataFrame,
    final_df: pl.DataFrame,
    institutional_history_by_ric_df: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    exact_summary_df = exact_raw_df.group_by("doc_id").agg(
        pl.len().alias("exact_raw_row_count"),
        pl.col("response_date").drop_nulls().min().alias("exact_first_response_date"),
        pl.col("response_date").drop_nulls().max().alias("exact_last_response_date"),
        pl.when(pl.col("is_institutional_category"))
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .sum()
        .alias("exact_institutional_row_count"),
        pl.when(pl.col("is_institutional_category") & pl.col("returned_value").is_not_null())
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .sum()
        .alias("exact_institutional_nonnull_row_count"),
    )
    fallback_summary_df = fallback_raw_df.group_by("doc_id").agg(
        pl.len().alias("fallback_raw_row_count"),
        pl.col("response_date").drop_nulls().min().alias("fallback_first_response_date"),
        pl.col("response_date").drop_nulls().max().alias("fallback_last_response_date"),
        pl.when(pl.col("is_institutional_category"))
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .sum()
        .alias("fallback_institutional_row_count"),
        pl.when(pl.col("is_institutional_category") & pl.col("returned_value").is_not_null())
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .sum()
        .alias("fallback_institutional_nonnull_row_count"),
    )

    truth_df = (
        exact_requests_df.rename({"retrieval_eligible": "exact_retrieval_eligible", "retrieval_exclusion_reason": "exact_retrieval_exclusion_reason"})
        .join(
            fallback_requests_df.select(
                "doc_id",
                pl.col("retrieval_eligible").alias("fallback_retrieval_eligible"),
                pl.col("retrieval_exclusion_reason").alias("fallback_retrieval_exclusion_reason"),
            ),
            on="doc_id",
            how="left",
        )
        .join(exact_summary_df, on="doc_id", how="left")
        .join(fallback_summary_df, on="doc_id", how="left")
        .join(
            final_df.select(
                "doc_id",
                "selected_response_date",
                "institutional_ownership_pct",
                "retrieval_status",
                "fallback_used",
            ),
            on="doc_id",
            how="left",
        )
        .join(
            institutional_history_by_ric_df.rename({"candidate_ric": "authoritative_ric"}),
            on="authoritative_ric",
            how="left",
        )
        .with_columns(
            [
                pl.col(name).cast(pl.Int64, strict=False).fill_null(0).alias(name)
                for name in (
                    "exact_raw_row_count",
                    "exact_institutional_row_count",
                    "exact_institutional_nonnull_row_count",
                    "fallback_raw_row_count",
                    "fallback_institutional_row_count",
                    "fallback_institutional_nonnull_row_count",
                    "institutional_history_row_count",
                )
            ]
        )
        .with_columns(
            (pl.col("exact_raw_row_count") > 0).alias("exact_raw_present"),
            (pl.col("exact_institutional_nonnull_row_count") > 0).alias("exact_institutional_nonnull_present"),
            pl.col("fallback_retrieval_eligible").is_not_null().alias("fallback_request_present"),
            (pl.col("fallback_raw_row_count") > 0).alias("fallback_raw_present"),
            (pl.col("fallback_institutional_nonnull_row_count") > 0).alias("fallback_institutional_nonnull_present"),
            pl.col("institutional_ownership_pct").is_not_null().alias("final_has_nonnull_ownership"),
            (
                pl.col("first_observed_institutional_date").is_not_null()
                & pl.col("target_effective_date").is_not_null()
                & (pl.col("first_observed_institutional_date") > pl.col("target_effective_date"))
            ).alias("target_date_precedes_first_observed_ownership_date"),
        )
        .with_columns(
            pl.when(~pl.col("exact_retrieval_eligible"))
            .then(pl.lit("not_retrieval_eligible"))
            .when(pl.col("exact_institutional_nonnull_present"))
            .then(pl.lit("available_at_exact_target_date"))
            .when(pl.col("fallback_institutional_nonnull_present"))
            .then(pl.lit("available_only_in_fallback_window"))
            .when(pl.col("first_observed_institutional_date").is_null())
            .then(pl.lit("no_institutional_history_observed_for_ric"))
            .when(pl.col("target_date_precedes_first_observed_ownership_date"))
            .then(pl.lit("available_only_later_in_time_series"))
            .otherwise(pl.lit("no_usable_exact_or_fallback_despite_other_history"))
            .alias("date_coverage_bucket")
        )
        .sort("filing_date", "doc_id")
    )

    summary = {
        "available": True,
        "doc_rows": int(truth_df.height),
        "nonnull_final_rows": int(truth_df.filter(pl.col("final_has_nonnull_ownership")).height),
        "retrieval_status_counts": _group_shares(truth_df, "retrieval_status"),
        "date_coverage_bucket_counts": _group_shares(truth_df, "date_coverage_bucket"),
        "target_date_precedes_first_observed_count": int(
            truth_df.filter(pl.col("target_date_precedes_first_observed_ownership_date")).height
        ),
    }
    return truth_df, summary


def _build_downstream_shrinkage_summary(
    *,
    downstream_paths: DownstreamPaths | None,
    doc_truth_df: pl.DataFrame,
) -> dict[str, Any]:
    if downstream_paths is None:
        return {"available": False, "stage_table": [], "note": "Downstream LM2011 post-Refinitiv artifacts not found."}

    backbone_schema = _scan_schema(downstream_paths.sample_backbone_path)
    backbone_permno_col = _resolve_first_existing(backbone_schema.keys(), ("KYPERMNO", "kypermno"), "LM2011 sample backbone")
    backbone_df = pl.read_parquet(downstream_paths.sample_backbone_path).select(
        _clean_text_expr("doc_id"),
        _normalize_int_expr(backbone_permno_col, "KYPERMNO"),
    ).unique(subset=["doc_id"], maintain_order=True)
    event_df = pl.read_parquet(downstream_paths.event_panel_path).select(
        _clean_text_expr("doc_id"),
        _normalize_int_expr("KYPERMNO"),
    ).unique(subset=["doc_id"], maintain_order=True)
    regression_df = pl.read_parquet(downstream_paths.return_regression_panel_full_10k_path).select(
        _clean_text_expr("doc_id"),
        _normalize_int_expr("KYPERMNO"),
    ).unique(subset=["doc_id"], maintain_order=True)

    doc_truth_subset = doc_truth_df.select("doc_id", "KYPERMNO", "final_has_nonnull_ownership")
    backbone_docs = int(backbone_df.height)
    backbone_permnos = int(backbone_df.get_column("KYPERMNO").drop_nulls().n_unique())

    def _stage_row(stage: str, stage_df: pl.DataFrame, note: str) -> dict[str, Any]:
        joined = stage_df.join(doc_truth_subset, on="doc_id", how="left")
        docs = int(stage_df.height)
        permnos = int(stage_df.get_column("KYPERMNO").drop_nulls().n_unique())
        docs_with_doc_truth = int(joined.filter(pl.col("final_has_nonnull_ownership").is_not_null()).height)
        docs_with_nonnull = int(joined.filter(pl.col("final_has_nonnull_ownership").fill_null(False)).height)
        return {
            "stage": stage,
            "docs": docs,
            "doc_share_vs_backbone": _share(docs, backbone_docs),
            "distinct_permno": permnos,
            "permno_share_vs_backbone": _share(permnos, backbone_permnos),
            "docs_with_doc_truth": docs_with_doc_truth,
            "docs_with_nonnull_ownership": docs_with_nonnull,
            "nonnull_share_within_stage": _share(docs_with_nonnull, docs),
            "note": note,
        }

    stage_table = [
        _stage_row("sample_backbone", backbone_df, "LM2011 post-Refinitiv backbone rows."),
        _stage_row("event_panel", event_df, "Docs surviving event-panel market and text filters."),
        _stage_row(
            "return_regression_panel_full_10k",
            regression_df,
            "Docs surviving the full-10K return-regression panel construction.",
        ),
    ]
    return {
        "available": True,
        "output_dir": str(downstream_paths.root),
        "stage_table": stage_table,
        "backbone_docs": backbone_docs,
        "backbone_permnos": backbone_permnos,
        "note": "Backbone/event/regression counts are anchored to the LM2011 post-Refinitiv output directory.",
    }


def _build_live_api_baseline(
    *,
    paths: RunPaths,
    doc_paths: DocOwnershipPaths | None,
) -> dict[str, Any]:
    ownership_manifest_path = (
        paths.ownership_universe_dir / "refinitiv_ownership_universe_stage_manifest.json"
    )
    ownership_manifest = _read_json_if_exists(ownership_manifest_path)
    exact_manifest = _read_json_if_exists(doc_paths.exact_manifest_path) if doc_paths is not None else None
    fallback_manifest = _read_json_if_exists(doc_paths.fallback_manifest_path) if doc_paths is not None else None
    lseg_available_now = bool(is_lseg_available())
    return {
        "lseg_available_now": lseg_available_now,
        "ownership_universe_manifest_present": ownership_manifest is not None,
        "ownership_universe_manifest_summary": ownership_manifest.get("summary") if ownership_manifest else None,
        "doc_exact_manifest_present": exact_manifest is not None,
        "doc_exact_manifest_summary": exact_manifest.get("summary") if exact_manifest else None,
        "doc_fallback_manifest_present": fallback_manifest is not None,
        "doc_fallback_manifest_summary": fallback_manifest.get("summary") if fallback_manifest else None,
        "note": (
            "Stored stage manifests capture prior live API runs. The current interpreter can import the LSEG SDK."
            if lseg_available_now
            else "Stored stage manifests capture prior live API runs. New direct live calls remain blocked until "
            "is_lseg_available() becomes true in the current environment."
        ),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default, sort_keys=True), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _format_int(value: int | None) -> str:
    return "n/a" if value is None else f"{int(value):,}"


def _format_pct(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{float(value) * 100:.{digits}f}%"


def _format_date(value: dt.date | dt.datetime | None) -> str:
    if value is None:
        return "n/a"
    return value.date().isoformat() if isinstance(value, dt.datetime) else value.isoformat()


def _markdown_table(rows: list[dict[str, Any]], columns: Sequence[tuple[str, str]]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(str(row.get(key, "")) for key, _ in columns) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def _render_report(summary: dict[str, Any]) -> str:
    basis = summary["basis_universe"]
    chain_rows = summary["artifact_chain_table"]
    ladder_rows = summary["join_back_coverage_table"]
    resolution = summary["resolution_diagnostics"]
    ownership = summary["ownership_diagnostics"]
    authority = summary["authority_diagnostics"]
    sampling = summary["sampling"]
    failure_sampling = summary["failure_sampling"]
    integrity = summary["chain_integrity"]
    step1_full = summary["step1_full_universe"]
    step1_snapshot_diffs = summary["step1_snapshot_diffs"]
    doc_diagnostics = summary["doc_ownership_diagnostics"]
    downstream = summary["downstream_diagnostics"]
    live_baseline = summary["live_api_baseline"]

    chain_md_rows = [
        {
            "stage": row["stage"],
            "rows": _format_int(row["rows"]),
            "bridge_rows": _format_int(row["distinct_bridge_row_id"]),
            "permnos": _format_int(row["distinct_permno"]),
            "rics": _format_int(row["distinct_ric"]),
            "date_min": _format_date(row["date_min"]),
            "date_max": _format_date(row["date_max"]),
            "note": row["note"],
        }
        for row in chain_rows
    ]
    ladder_md_rows = [
        {
            "stage": row["stage"],
            "rows": _format_int(row["rows"]),
            "row_share": _format_pct(row["row_share"]),
            "permnos": _format_int(row["distinct_permno"]),
            "permno_share": _format_pct(row["permno_share"]),
            "left_rows": _format_int(row["left_rows"]),
            "note": row["note"],
        }
        for row in ladder_rows
    ]
    integrity_md_rows = [
        {
            "check": row["check"],
            "passed": "yes" if row["passed"] else "no",
            "child_only": _format_int(row["child_only_count"]),
            "examples": ", ".join(row["child_only_examples"]),
        }
        for row in integrity
    ]
    resolution_bucket_rows = [
        {
            "bucket": row["resolution_bucket"],
            "bridge_rows": _format_int(row["bridge_rows"]),
            "share": _format_pct(row["bridge_row_share"]),
            "permnos": _format_int(row["distinct_permno"]),
        }
        for row in resolution["coverage_buckets"]
    ]
    ownership_outcome_rows = [
        {
            "bucket": row["outcome_bucket"],
            "request_rows": _format_int(row["request_rows"]),
            "share": _format_pct(row["request_row_share"]),
            "permnos": _format_int(row["distinct_permno"]),
        }
        for row in ownership["request_outcome_buckets"]
    ]
    step1_full_outcome_rows = [
        {
            "bucket": row["outcome_bucket"],
            "request_rows": _format_int(row["request_rows"]),
            "share": _format_pct(row["request_row_share"]),
            "permnos": _format_int(row["distinct_permno"]),
        }
        for row in step1_full["request_outcome_buckets"]
    ]
    snapshot_diff_rows = [
        {
            "snapshot": row["snapshot_label"],
            "overlap": _format_int(row["lookup_id_overlap_rows"]),
            "regressed": _format_int(row["nonnull_regression_rows"]),
            "improved": _format_int(row["nonnull_improvement_rows"]),
            "result_regressed": _format_int(row["result_regression_rows"]),
            "authority_changed": _format_int(row["authority_status_changed_overlap_rows"]),
        }
        for row in step1_snapshot_diffs
    ]
    downstream_rows = [
        {
            "stage": row["stage"],
            "docs": _format_int(row["docs"]),
            "doc_share": _format_pct(row["doc_share_vs_backbone"]),
            "permnos": _format_int(row["distinct_permno"]),
            "doc_truth_docs": _format_int(row["docs_with_doc_truth"]),
            "nonnull_docs": _format_int(row["docs_with_nonnull_ownership"]),
            "nonnull_share": _format_pct(row["nonnull_share_within_stage"]),
            "note": row["note"],
        }
        for row in downstream.get("stage_table", [])
    ]
    doc_date_bucket_rows = [
        {
            "bucket": row["date_coverage_bucket"],
            "count": _format_int(row["count"]),
            "share": _format_pct(row["share"]),
        }
        for row in doc_diagnostics.get("date_coverage_bucket_counts", [])
    ]
    lines = [
        "# Refinitiv Lookup Basis Analysis",
        "",
        f"Generated: {summary['generated_at_utc']}",
        "",
        "## Basis Universe",
        "",
        f"- Basis rows: {_format_int(basis['rows'])}",
        f"- Distinct doc_id: {_format_int(basis['distinct_doc_id'])}",
        f"- Distinct KYPERMNO: {_format_int(basis['distinct_permno'])}",
        f"- Distinct gvkey_int: {_format_int(basis['distinct_gvkey_int'])}",
        f"- Filing date range: {_format_date(basis['filing_date_min'])} to {_format_date(basis['filing_date_max'])}",
        f"- Join-ready basis rows (`KYPERMNO` + `filing_date`): {_format_int(basis['rows_join_ready'])} ({_format_pct(basis['rows_join_ready_share'])})",
        f"- Basis rows with native ticker: {_format_int(basis['rows_with_native_ticker'])} ({_format_pct(basis['rows_with_native_ticker_share'])})",
        f"- Basis rows with bridge lookup identifiers: {_format_int(basis['rows_with_bridge_lookup_identifier'])} ({_format_pct(basis['rows_with_bridge_lookup_identifier_share'])})",
        "",
        "## Artifact Chain",
        "",
        _markdown_table(
            chain_md_rows,
            columns=(
                ("stage", "Stage"),
                ("rows", "Rows"),
                ("bridge_rows", "Distinct Bridge Rows"),
                ("permnos", "Distinct PERMNO"),
                ("rics", "Distinct RIC"),
                ("date_min", "Date Min"),
                ("date_max", "Date Max"),
                ("note", "Note"),
            ),
        ),
        "",
        "## Chain Integrity",
        "",
        _markdown_table(
            integrity_md_rows,
            columns=(
                ("check", "Check"),
                ("passed", "Passed"),
                ("child_only", "Child-Only Count"),
                ("examples", "Examples"),
            ),
        ),
        "",
        "## Join-Back Coverage",
        "",
        _markdown_table(
            ladder_md_rows,
            columns=(
                ("stage", "Stage"),
                ("rows", "Basis Rows"),
                ("row_share", "Row Share"),
                ("permnos", "Distinct PERMNO"),
                ("permno_share", "PERMNO Share"),
                ("left_rows", "Left Behind"),
                ("note", "Note"),
            ),
        ),
        "",
        "## Resolution Diagnostics",
        "",
        f"- Matched bridge rows in scope: {_format_int(resolution['matched_bridge_rows'])}",
        f"- Bridge rows with any lookup success: {_format_int(resolution['bridge_rows_with_any_lookup_success'])} ({_format_pct(resolution['bridge_rows_with_any_lookup_success_share'])})",
        f"- Bridge rows with final usable RIC: {_format_int(resolution['bridge_rows_with_final_ric'])} ({_format_pct(resolution['bridge_rows_with_final_ric_share'])})",
        f"- Distinct effective RICs: {_format_int(resolution['distinct_effective_ric'])}",
        "",
        _markdown_table(
            resolution_bucket_rows,
            columns=(
                ("bucket", "Resolution Bucket"),
                ("bridge_rows", "Bridge Rows"),
                ("share", "Share"),
                ("permnos", "Distinct PERMNO"),
            ),
        ),
        "",
        "## Authority Diagnostics",
        "",
        f"- Authority decisions in scope: {_format_int(authority['decision_rows'])}",
        f"- Authority exceptions in scope: {_format_int(authority['exception_rows'])}",
        f"- Review-required decisions: {_format_int(authority['review_required_rows'])}",
        "",
        "## Ownership Diagnostics",
        "",
        f"- Handoff rows in scope: {_format_int(ownership['handoff_rows'])}",
        f"- Retrieval-eligible handoff rows: {_format_int(ownership['retrieval_eligible_rows'])} ({_format_pct(ownership['retrieval_eligible_share'])})",
        f"- Attempted request rows: {_format_int(ownership['retrieval_attempted_rows'])} ({_format_pct(ownership['retrieval_attempted_share'])})",
        f"- Ownership result rows: {_format_int(ownership['result_rows'])}",
        f"- Ownership rows with non-null values: {_format_int(ownership['nonnull_result_rows'])}",
        f"- Request rows with non-null ownership values: {_format_int(ownership['request_rows_with_nonnull'])} ({_format_pct(ownership['request_rows_with_nonnull_share'])})",
        "",
        _markdown_table(
            ownership_outcome_rows,
            columns=(
                ("bucket", "Outcome Bucket"),
                ("request_rows", "Request Rows"),
                ("share", "Share"),
                ("permnos", "Distinct PERMNO"),
            ),
        ),
        "",
        "## Full Step1 Universe",
        "",
        f"- Full request rows: {_format_int(step1_full['handoff_rows'])}",
        f"- Distinct PERMNO: {_format_int(step1_full['distinct_permno'])}",
        f"- Distinct candidate RIC: {_format_int(step1_full['distinct_candidate_ric'])}",
        f"- Retrieval-eligible rows: {_format_int(step1_full['retrieval_eligible_rows'])}",
        f"- Attempted rows: {_format_int(step1_full['retrieval_attempted_rows'])}",
        f"- Rows with non-null ownership: {_format_int(step1_full['request_rows_with_nonnull'])}",
        "",
        _markdown_table(
            step1_full_outcome_rows,
            columns=(
                ("bucket", "Outcome Bucket"),
                ("request_rows", "Request Rows"),
                ("share", "Share"),
                ("permnos", "Distinct PERMNO"),
            ),
        ),
        "",
        "## Snapshot Diffs",
        "",
        _markdown_table(
            snapshot_diff_rows,
            columns=(
                ("snapshot", "Snapshot"),
                ("overlap", "Overlap Rows"),
                ("regressed", "Nonnull Regressions"),
                ("improved", "Nonnull Improvements"),
                ("result_regressed", "Result Regressions"),
                ("authority_changed", "Authority Changes"),
            ),
        ),
        "",
        "## Doc Ownership",
        "",
    ]
    if doc_diagnostics.get("available", True):
        lines.extend(
            [
                f"- Doc rows: {_format_int(doc_diagnostics['doc_rows'])}",
                f"- Final non-null ownership rows: {_format_int(doc_diagnostics['nonnull_final_rows'])}",
                f"- Target date precedes first observed ownership date: {_format_int(doc_diagnostics['target_date_precedes_first_observed_count'])}",
                "",
                _markdown_table(
                    doc_date_bucket_rows,
                    columns=(
                        ("bucket", "Date Coverage Bucket"),
                        ("count", "Count"),
                        ("share", "Share"),
                    ),
                ),
                "",
            ]
        )
    else:
        lines.extend([f"- {doc_diagnostics['note']}", ""])

    lines.extend(
        [
            "## Downstream LM2011",
            "",
        ]
    )
    if downstream.get("available"):
        lines.extend(
            [
                _markdown_table(
                    downstream_rows,
                    columns=(
                        ("stage", "Stage"),
                        ("docs", "Docs"),
                        ("doc_share", "Share Vs Backbone"),
                        ("permnos", "Distinct PERMNO"),
                        ("doc_truth_docs", "Docs With Doc Ownership Row"),
                        ("nonnull_docs", "Nonnull Ownership Docs"),
                        ("nonnull_share", "Nonnull Share"),
                        ("note", "Note"),
                    ),
                ),
                "",
            ]
        )
    else:
        lines.extend([f"- {downstream['note']}", ""])

    lines.extend(
        [
            "## Samples",
            "",
            f"- Positive-control requested sample size: {_format_int(sampling['requested_sample_size'])}",
            f"- Positive-control sampled RICs: {_format_int(sampling['actual_sample_size'])}",
            f"- Positive-control buckets represented: {', '.join(sampling['bucket_counts_sampled']) if sampling['bucket_counts_sampled'] else 'none'}",
            f"- Failure-oriented sampled RICs: {_format_int(failure_sampling['actual_sample_size'])}",
            f"- Failure-oriented buckets represented: {', '.join(failure_sampling['bucket_counts_sampled']) if failure_sampling['bucket_counts_sampled'] else 'none'}",
            "",
            "## Live API Baseline",
            "",
            f"- `is_lseg_available()` now: {'true' if live_baseline['lseg_available_now'] else 'false'}",
            f"- Ownership-universe manifest present: {'yes' if live_baseline['ownership_universe_manifest_present'] else 'no'}",
            f"- Doc exact manifest present: {'yes' if live_baseline['doc_exact_manifest_present'] else 'no'}",
            f"- Doc fallback manifest present: {'yes' if live_baseline['doc_fallback_manifest_present'] else 'no'}",
            f"- Note: {live_baseline['note']}",
            "",
            "## Output Paths",
            "",
            f"- Summary markdown: `{summary['output_paths']['summary_markdown']}`",
            f"- Summary JSON: `{summary['output_paths']['summary_json']}`",
            (
                f"- Positive-control sample parquet: `{summary['output_paths']['sample_parquet']}`"
                if summary['output_paths']['sample_parquet'] is not None
                else "- Positive-control sample parquet: not written"
            ),
            (
                f"- Positive-control sample CSV: `{summary['output_paths']['sample_csv']}`"
                if summary['output_paths']['sample_csv'] is not None
                else "- Positive-control sample CSV: not written"
            ),
            (
                f"- Failure-oriented sample parquet: `{summary['output_paths']['failure_sample_parquet']}`"
                if summary['output_paths']['failure_sample_parquet'] is not None
                else "- Failure-oriented sample parquet: not written"
            ),
            (
                f"- Failure-oriented sample CSV: `{summary['output_paths']['failure_sample_csv']}`"
                if summary['output_paths']['failure_sample_csv'] is not None
                else "- Failure-oriented sample CSV: not written"
            ),
            (
                f"- Doc truth parquet: `{summary['output_paths']['doc_truth_parquet']}`"
                if summary['output_paths']['doc_truth_parquet'] is not None
                else "- Doc truth parquet: not written"
            ),
            (
                f"- Doc truth CSV: `{summary['output_paths']['doc_truth_csv']}`"
                if summary['output_paths']['doc_truth_csv'] is not None
                else "- Doc truth CSV: not written"
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def run_analysis(args: argparse.Namespace, paths: RunPaths) -> dict[str, Any]:
    _log(args.verbose, "Loading basis and bridge artifacts...")
    basis_df, basis_metadata = _read_basis_df(paths, year_start=args.year_start, year_end=args.year_end)
    bridge_df = _read_bridge_df(paths)
    basis_bridge_matches_df, matched_bridge_ids_df, matched_permnos_df = _build_basis_bridge_matches(basis_df, bridge_df)
    bridge_matched_df = bridge_df.join(matched_bridge_ids_df, on="bridge_row_id", how="inner")
    basis_bridge_context_df = basis_bridge_matches_df.group_by("bridge_row_id").agg(
        pl.len().alias("basis_row_count"),
        pl.col("doc_id").drop_nulls().n_unique().alias("basis_doc_count"),
        pl.col("filing_date").min().alias("basis_min_filing_date"),
        pl.col("filing_date").max().alias("basis_max_filing_date"),
    )
    doc_paths = _resolve_doc_ownership_paths(paths.run_root)
    downstream_paths = _resolve_downstream_paths(paths.run_root)
    snapshot_paths = _discover_step1_snapshot_paths(paths.run_root)

    _log(args.verbose, "Loading lookup, resolution, authority, and ownership artifacts...")
    lookup_snapshot_df = _read_lookup_stage_df(
        paths.lookup_snapshot_path,
        ids_df=matched_bridge_ids_df,
        label="lookup snapshot artifact",
    )
    lookup_extended_df = _read_lookup_stage_df(
        paths.lookup_extended_path,
        ids_df=matched_bridge_ids_df,
        label="lookup extended artifact",
    )
    resolution_df = _read_resolution_df(paths, ids_df=matched_bridge_ids_df)
    instrument_authority_df = _read_instrument_authority_df(paths, ids_df=matched_bridge_ids_df)
    handoff_df = _read_handoff_df(paths, ids_df=matched_bridge_ids_df)
    row_summary_df = _read_row_summary_df(paths, ids_df=matched_bridge_ids_df)
    results_df = _read_results_df(paths, ids_df=matched_bridge_ids_df)
    authority_decisions_df = _read_authority_decisions_df(paths, permnos_df=matched_permnos_df)
    authority_exceptions_df = _read_authority_exceptions_df(paths, permnos_df=matched_permnos_df)
    _log(args.verbose, "Loading full step1 ownership universe and optional downstream artifacts...")
    full_row_summary_df = _read_row_summary_from_path(paths.ownership_row_summary_path)
    full_results_df = _normalize_results_categories(_read_results_from_path(paths.ownership_results_path))
    full_authority_decisions_df = _read_authority_decisions_from_path(paths.authority_decisions_path)
    full_authority_exceptions_df = _read_authority_exceptions_from_path(paths.authority_exceptions_path)

    doc_exact_requests_df = _read_doc_request_df(doc_paths.exact_requests_path) if doc_paths is not None else None
    doc_exact_raw_df = _read_doc_raw_df(doc_paths.exact_raw_path) if doc_paths is not None else None
    doc_fallback_requests_df = _read_doc_request_df(doc_paths.fallback_requests_path) if doc_paths is not None else None
    doc_fallback_raw_df = _read_doc_raw_df(doc_paths.fallback_raw_path) if doc_paths is not None else None
    doc_final_df = _read_doc_final_df(doc_paths.final_path) if doc_paths is not None else None

    _log(args.verbose, "Building causal-chain and join-back diagnostics...")
    chain_integrity = [
        _set_contains_check(
            parent_df=bridge_matched_df,
            child_df=lookup_snapshot_df,
            key="bridge_row_id",
            check="lookup_snapshot bridge_row_id subset of matched bridge universe",
        ),
        _set_contains_check(
            parent_df=lookup_snapshot_df,
            child_df=lookup_extended_df,
            key="bridge_row_id",
            check="lookup_extended bridge_row_id subset of lookup snapshot",
        ),
        _set_contains_check(
            parent_df=lookup_extended_df,
            child_df=resolution_df,
            key="bridge_row_id",
            check="resolution bridge_row_id subset of lookup extended",
        ),
        _set_contains_check(
            parent_df=resolution_df,
            child_df=instrument_authority_df,
            key="bridge_row_id",
            check="instrument_authority bridge_row_id subset of resolution",
        ),
        _set_contains_check(
            parent_df=resolution_df,
            child_df=handoff_df,
            key="bridge_row_id",
            check="ownership_handoff bridge_row_id subset of resolution",
        ),
        _set_contains_check(
            parent_df=handoff_df,
            child_df=row_summary_df,
            key="ownership_lookup_row_id",
            check="ownership_row_summary ownership_lookup_row_id subset of ownership_handoff",
        ),
        _set_contains_check(
            parent_df=handoff_df,
            child_df=results_df,
            key="ownership_lookup_row_id",
            check="ownership_results ownership_lookup_row_id subset of ownership_handoff",
        ),
        _set_contains_check(
            parent_df=authority_decisions_df,
            child_df=authority_exceptions_df,
            key="KYPERMNO",
            check="authority_exceptions KYPERMNO subset of authority_decisions",
        ),
    ]

    resolution_bridge_df = (
        bridge_matched_df.select("bridge_row_id", "KYPERMNO")
        .join(
            resolution_df.select(
                "bridge_row_id",
                "accepted_resolution_status",
                "effective_resolution_status",
                "accepted_ric",
                "effective_collection_ric",
                "ISIN_success",
                "CUSIP_success",
                "TICKER_success",
            ),
            on="bridge_row_id",
            how="left",
        )
        .with_columns(
            pl.any_horizontal(
                pl.col("ISIN_success").fill_null(False),
                pl.col("CUSIP_success").fill_null(False),
                pl.col("TICKER_success").fill_null(False),
            ).alias("any_lookup_success"),
            (
                pl.col("accepted_resolution_status").is_not_null()
                | pl.col("effective_resolution_status").is_not_null()
            ).alias("resolution_row_present"),
            pl.col("effective_collection_ric").is_not_null().alias("has_final_usable_ric"),
        )
        .with_columns(
            pl.when(~pl.col("resolution_row_present"))
            .then(pl.lit("missing_resolution_row"))
            .when(pl.col("has_final_usable_ric"))
            .then(pl.lit("accepted_or_effective_ric"))
            .when(pl.col("any_lookup_success"))
            .then(pl.lit("lookup_success_but_no_final_ric"))
            .otherwise(pl.lit("no_lookup_success"))
            .alias("resolution_bucket")
        )
    )

    stage_bridge_ids = {
        "bridge_lookup_identifiers": bridge_matched_df.filter(pl.col("bridge_has_lookup_identifier")).select("bridge_row_id").unique(),
        "bridge": matched_bridge_ids_df,
        "resolution": resolution_df.select("bridge_row_id").unique(),
        "effective_ric": resolution_df.filter(pl.col("effective_collection_ric").is_not_null()).select("bridge_row_id").unique(),
        "instrument_authority": instrument_authority_df.select("bridge_row_id").unique(),
        "authority_eligible": instrument_authority_df.filter(pl.col("authority_eligible")).select("bridge_row_id").unique(),
        "ownership_handoff": handoff_df.select("bridge_row_id").unique(),
        "retrieval_eligible": row_summary_df.filter(pl.col("retrieval_eligible")).select("bridge_row_id").unique(),
        "retrieval_attempted": row_summary_df.filter(pl.col("retrieval_row_present")).select("bridge_row_id").unique(),
        "ownership_results": row_summary_df.filter(pl.col("ownership_rows_returned") > 0).select("bridge_row_id").unique(),
        "ownership_nonnull": row_summary_df.filter(pl.col("ownership_nonnull_value_count") > 0).select("bridge_row_id").unique(),
    }
    decision_permnos_df = authority_decisions_df.select("KYPERMNO").drop_nulls().unique()
    exception_permnos_df = authority_exceptions_df.select("KYPERMNO").drop_nulls().unique()

    coverage_ladder = [
        _coverage_row(
            stage="basis_join_ready",
            basis_df=basis_df,
            covered_basis_ids_df=basis_df.filter(pl.col("basis_join_ready")).select("basis_row_id"),
            note="Basis rows with usable `KYPERMNO` and `filing_date` for bridge overlap joins.",
        ),
        _coverage_row(
            stage="bridge_lookup_identifier",
            basis_df=basis_df,
            covered_basis_ids_df=basis_bridge_matches_df.join(
                stage_bridge_ids["bridge_lookup_identifiers"],
                on="bridge_row_id",
                how="semi",
            ).select("basis_row_id").unique(),
            note="Basis rows whose matched bridge spans expose at least one lookup identifier (`CUSIP`/`ISIN`/`TICKER`).",
        ),
        _coverage_row(
            stage="bridge_universe",
            basis_df=basis_df,
            covered_basis_ids_df=basis_bridge_matches_df.select("basis_row_id").unique(),
            note="Basis rows represented in `refinitiv_bridge_universe` by `KYPERMNO` plus filing-date overlap.",
        ),
        _coverage_row(
            stage="resolution_artifact",
            basis_df=basis_df,
            covered_basis_ids_df=basis_bridge_matches_df.join(stage_bridge_ids["resolution"], on="bridge_row_id", how="semi")
            .select("basis_row_id")
            .unique(),
            note="Basis rows whose matched bridge spans have a resolution-stage row.",
        ),
        _coverage_row(
            stage="effective_ric_path",
            basis_df=basis_df,
            covered_basis_ids_df=basis_bridge_matches_df.join(stage_bridge_ids["effective_ric"], on="bridge_row_id", how="semi")
            .select("basis_row_id")
            .unique(),
            note="Basis rows with at least one matched bridge span that resolves to a final usable `effective_collection_ric`.",
        ),
        _coverage_row(
            stage="instrument_authority_artifact",
            basis_df=basis_df,
            covered_basis_ids_df=basis_bridge_matches_df.join(
                stage_bridge_ids["instrument_authority"],
                on="bridge_row_id",
                how="semi",
            ).select("basis_row_id").unique(),
            note="Basis rows represented in the instrument-authority artifact.",
        ),
        _coverage_row(
            stage="instrument_authority_eligible",
            basis_df=basis_df,
            covered_basis_ids_df=basis_bridge_matches_df.join(
                stage_bridge_ids["authority_eligible"],
                on="bridge_row_id",
                how="semi",
            ).select("basis_row_id").unique(),
            note="Basis rows whose matched bridge spans are authority-eligible (`gvkey_int` plus usable effective RIC).",
        ),
        _coverage_row(
            stage="ownership_handoff_artifact",
            basis_df=basis_df,
            covered_basis_ids_df=basis_bridge_matches_df.join(stage_bridge_ids["ownership_handoff"], on="bridge_row_id", how="semi")
            .select("basis_row_id")
            .unique(),
            note="Basis rows represented in `refinitiv_ownership_universe_handoff_common_stock`.",
        ),
        _coverage_row(
            stage="ownership_retrieval_eligible",
            basis_df=basis_df,
            covered_basis_ids_df=basis_bridge_matches_df.join(stage_bridge_ids["retrieval_eligible"], on="bridge_row_id", how="semi")
            .select("basis_row_id")
            .unique(),
            note="Basis rows reaching retrieval-eligible ownership-universe requests.",
        ),
        _coverage_row(
            stage="ownership_retrieval_attempted",
            basis_df=basis_df,
            covered_basis_ids_df=basis_bridge_matches_df.join(stage_bridge_ids["retrieval_attempted"], on="bridge_row_id", how="semi")
            .select("basis_row_id")
            .unique(),
            note="Basis rows whose matched bridge spans have an attempted ownership-universe request row.",
        ),
        _coverage_row(
            stage="ownership_result_rows",
            basis_df=basis_df,
            covered_basis_ids_df=basis_bridge_matches_df.join(stage_bridge_ids["ownership_results"], on="bridge_row_id", how="semi")
            .select("basis_row_id")
            .unique(),
            note="Basis rows whose matched bridge spans produce at least one ownership result row.",
        ),
        _coverage_row(
            stage="ownership_nonnull_values",
            basis_df=basis_df,
            covered_basis_ids_df=basis_bridge_matches_df.join(stage_bridge_ids["ownership_nonnull"], on="bridge_row_id", how="semi")
            .select("basis_row_id")
            .unique(),
            note="Basis rows whose matched bridge spans produce non-null ownership values.",
        ),
        _coverage_row(
            stage="ownership_authority_decision",
            basis_df=basis_df,
            covered_basis_ids_df=basis_df.join(decision_permnos_df, on="KYPERMNO", how="semi").select("basis_row_id"),
            note="Basis rows whose PERMNO receives an ownership-authority decision.",
        ),
        _coverage_row(
            stage="ownership_authority_exception",
            basis_df=basis_df,
            covered_basis_ids_df=basis_df.join(exception_permnos_df, on="KYPERMNO", how="semi").select("basis_row_id"),
            note="Basis rows whose PERMNO falls into an ownership-authority exception table.",
        ),
    ]

    basis_total_permnos = int(basis_df.get_column("KYPERMNO").drop_nulls().n_unique())
    basis_summary = {
        "rows": int(basis_df.height),
        "distinct_doc_id": int(basis_df.get_column("doc_id").drop_nulls().n_unique()),
        "distinct_permno": basis_total_permnos,
        "distinct_gvkey_int": int(basis_df.get_column("gvkey_int").drop_nulls().n_unique()),
        "filing_date_min": basis_df.select(pl.col("filing_date").min()).item(),
        "filing_date_max": basis_df.select(pl.col("filing_date").max()).item(),
        "rows_with_permno": int(basis_df.filter(pl.col("basis_has_permno")).height),
        "rows_with_filing_date": int(basis_df.filter(pl.col("basis_has_filing_date")).height),
        "rows_with_native_ticker": int(basis_df.filter(pl.col("basis_has_ticker")).height),
        "rows_with_native_ticker_share": basis_df.filter(pl.col("basis_has_ticker")).height / basis_df.height,
        "rows_with_gvkey_int": int(basis_df.filter(pl.col("basis_has_gvkey_int")).height),
        "rows_join_ready": int(basis_df.filter(pl.col("basis_join_ready")).height),
        "rows_join_ready_share": basis_df.filter(pl.col("basis_join_ready")).height / basis_df.height,
        "rows_with_bridge_lookup_identifier": coverage_ladder[1]["rows"],
        "rows_with_bridge_lookup_identifier_share": coverage_ladder[1]["row_share"],
        "rows_represented_in_bridge": coverage_ladder[2]["rows"],
        "rows_represented_in_resolution": coverage_ladder[3]["rows"],
        "rows_with_effective_ric_path": coverage_ladder[4]["rows"],
        "rows_represented_in_instrument_authority": coverage_ladder[5]["rows"],
        "rows_authority_eligible": coverage_ladder[6]["rows"],
        "rows_represented_in_ownership_handoff": coverage_ladder[7]["rows"],
        "rows_retrieval_eligible": coverage_ladder[8]["rows"],
        "rows_retrieval_attempted": coverage_ladder[9]["rows"],
        "rows_with_ownership_results": coverage_ladder[10]["rows"],
        "rows_with_ownership_nonnull_values": coverage_ladder[11]["rows"],
        "rows_with_ownership_authority_decision": coverage_ladder[12]["rows"],
        "rows_with_ownership_authority_exception": coverage_ladder[13]["rows"],
        **basis_metadata,
    }

    coverage_bucket_df = resolution_bridge_df.group_by("resolution_bucket").agg(
        pl.len().alias("bridge_rows"),
        pl.col("KYPERMNO").drop_nulls().n_unique().alias("distinct_permno"),
    ).with_columns(
        (pl.col("bridge_rows") / pl.lit(resolution_bridge_df.height)).alias("bridge_row_share")
    ).sort("bridge_rows", descending=True)
    resolution_diagnostics = {
        "matched_bridge_rows": int(resolution_bridge_df.height),
        "matched_bridge_permnos": int(resolution_bridge_df.get_column("KYPERMNO").drop_nulls().n_unique()),
        "bridge_rows_with_any_lookup_success": int(resolution_bridge_df.filter(pl.col("any_lookup_success")).height),
        "bridge_rows_with_any_lookup_success_share": (
            resolution_bridge_df.filter(pl.col("any_lookup_success")).height / resolution_bridge_df.height
            if resolution_bridge_df.height
            else None
        ),
        "bridge_rows_with_final_ric": int(resolution_bridge_df.filter(pl.col("has_final_usable_ric")).height),
        "bridge_rows_with_final_ric_share": (
            resolution_bridge_df.filter(pl.col("has_final_usable_ric")).height / resolution_bridge_df.height
            if resolution_bridge_df.height
            else None
        ),
        "distinct_effective_ric": int(resolution_df.get_column("effective_collection_ric").drop_nulls().n_unique()),
        "distinct_accepted_ric": int(resolution_df.get_column("accepted_ric").drop_nulls().n_unique()),
        "accepted_resolution_status_counts": _group_shares(resolution_df, "accepted_resolution_status"),
        "effective_resolution_status_counts": _group_shares(resolution_df, "effective_resolution_status"),
        "coverage_buckets": coverage_bucket_df.to_dicts(),
    }

    permno_effective_ric_df = (
        resolution_df.filter(pl.col("effective_collection_ric").is_not_null())
        .group_by("KYPERMNO")
        .agg(pl.col("effective_collection_ric").drop_nulls().n_unique().alias("effective_ric_count"))
    )
    effective_ric_permno_df = (
        resolution_df.filter(pl.col("effective_collection_ric").is_not_null())
        .group_by("effective_collection_ric")
        .agg(pl.col("KYPERMNO").drop_nulls().n_unique().alias("permno_count"))
        .rename({"effective_collection_ric": "ric"})
    )
    candidate_ric_permno_df = (
        row_summary_df.filter(pl.col("candidate_ric").is_not_null())
        .group_by("candidate_ric")
        .agg(pl.col("KYPERMNO").drop_nulls().n_unique().alias("permno_count"))
        .rename({"candidate_ric": "ric"})
    )
    resolution_diagnostics["concentration"] = {
        "permno_to_effective_ric": _multiplicity_summary(
            permno_effective_ric_df,
            group_col="KYPERMNO",
            count_col="effective_ric_count",
        ),
        "effective_ric_to_permno": _multiplicity_summary(
            effective_ric_permno_df,
            group_col="ric",
            count_col="permno_count",
        ),
        "candidate_ric_to_permno": _multiplicity_summary(
            candidate_ric_permno_df,
            group_col="ric",
            count_col="permno_count",
        ),
    }

    authority_diagnostics = {
        "decision_rows": int(authority_decisions_df.height),
        "exception_rows": int(authority_exceptions_df.height),
        "review_required_rows": int(authority_decisions_df.filter(pl.col("requires_review")).height),
        "decision_status_counts": _group_shares(authority_decisions_df, "authority_decision_status"),
        "decision_reason_counts": _group_shares(authority_decisions_df, "authority_decision_reason"),
        "exception_status_counts": _group_shares(authority_exceptions_df, "authority_exception_status"),
        "exception_reason_counts": _group_shares(authority_exceptions_df, "authority_exception_reason"),
        "review_flag_counts": {
            name: int(authority_decisions_df.filter(pl.col(name)).height)
            for name in authority_decisions_df.columns
            if name.startswith("review_flag_")
        },
    }

    request_outcomes_df = _build_request_outcomes_df(
        row_summary_df,
        authority_decisions_df,
        authority_exceptions_df,
    )
    full_request_outcomes_df = _build_request_outcomes_df(
        full_row_summary_df,
        full_authority_decisions_df,
        full_authority_exceptions_df,
    )

    retrieval_eligible_rows = int(row_summary_df.filter(pl.col("retrieval_eligible")).height)
    retrieval_attempted_rows = int(row_summary_df.filter(pl.col("retrieval_row_present")).height)
    request_rows_with_results = int(row_summary_df.filter(pl.col("ownership_rows_returned") > 0).height)
    request_rows_with_nonnull = int(row_summary_df.filter(pl.col("ownership_nonnull_value_count") > 0).height)
    ownership_diagnostics = {
        "handoff_rows": int(handoff_df.height),
        "handoff_distinct_permno": int(handoff_df.get_column("KYPERMNO").drop_nulls().n_unique()),
        "handoff_distinct_candidate_ric": int(handoff_df.get_column("candidate_ric").drop_nulls().n_unique()),
        "retrieval_eligible_rows": retrieval_eligible_rows,
        "retrieval_eligible_share": _share(retrieval_eligible_rows, row_summary_df.height),
        "retrieval_attempted_rows": retrieval_attempted_rows,
        "retrieval_attempted_share": _share(retrieval_attempted_rows, row_summary_df.height),
        "result_rows": int(results_df.height),
        "nonnull_result_rows": int(results_df.filter(pl.col("returned_value").is_not_null()).height),
        "request_rows_with_results": request_rows_with_results,
        "request_rows_with_results_share": _share(request_rows_with_results, row_summary_df.height),
        "request_rows_with_nonnull": request_rows_with_nonnull,
        "request_rows_with_nonnull_share": _share(request_rows_with_nonnull, row_summary_df.height),
        "distinct_permno_with_results": int(
            row_summary_df.filter(pl.col("ownership_rows_returned") > 0).get_column("KYPERMNO").drop_nulls().n_unique()
        ),
        "distinct_permno_with_nonnull": int(
            row_summary_df.filter(pl.col("ownership_nonnull_value_count") > 0)
            .get_column("KYPERMNO")
            .drop_nulls()
            .n_unique()
        ),
        "distinct_candidate_ric_with_results": int(
            row_summary_df.filter(pl.col("ownership_rows_returned") > 0).get_column("candidate_ric").drop_nulls().n_unique()
        ),
        "distinct_candidate_ric_with_nonnull": int(
            row_summary_df.filter(pl.col("ownership_nonnull_value_count") > 0)
            .get_column("candidate_ric")
            .drop_nulls()
            .n_unique()
        ),
        "retrieval_exclusion_reason_counts": _group_shares(
            row_summary_df.filter(~pl.col("retrieval_eligible")),
            "retrieval_exclusion_reason",
        ),
        "request_outcome_buckets": (
            request_outcomes_df.group_by("outcome_bucket")
            .agg(
                pl.len().alias("request_rows"),
                pl.col("KYPERMNO").drop_nulls().n_unique().alias("distinct_permno"),
            )
            .with_columns(
                (pl.col("request_rows") / pl.lit(request_outcomes_df.height)).alias("request_row_share")
            )
            .sort("request_rows", descending=True)
            .to_dicts()
        ),
        "outcomes_by_authority_status": _summarize_request_outcomes(request_outcomes_df, "authority_decision_status"),
        "outcomes_by_authority_reason": _summarize_request_outcomes(request_outcomes_df, "authority_decision_reason"),
        "outcomes_by_lookup_role": _summarize_request_outcomes(request_outcomes_df, "ownership_lookup_role"),
        "outcomes_by_lookup_input_source": _summarize_request_outcomes(request_outcomes_df, "lookup_input_source"),
        "outcomes_by_request_span_bucket": _summarize_request_outcomes(request_outcomes_df, "request_span_bucket"),
        "outcomes_by_request_start_year": _summarize_request_outcomes(
            request_outcomes_df.sort("request_start_year"),
            "request_start_year",
            max_rows=50,
        ),
    }

    step1_full_universe = {
        "handoff_rows": int(full_row_summary_df.height),
        "distinct_permno": int(full_row_summary_df.get_column("KYPERMNO").drop_nulls().n_unique()),
        "distinct_candidate_ric": int(full_row_summary_df.get_column("candidate_ric").drop_nulls().n_unique()),
        "retrieval_eligible_rows": int(full_row_summary_df.filter(pl.col("retrieval_eligible")).height),
        "retrieval_attempted_rows": int(full_row_summary_df.filter(pl.col("retrieval_row_present")).height),
        "request_rows_with_results": int(full_row_summary_df.filter(pl.col("ownership_rows_returned") > 0).height),
        "request_rows_with_nonnull": int(full_row_summary_df.filter(pl.col("ownership_nonnull_value_count") > 0).height),
        "distinct_permno_with_nonnull": int(
            full_row_summary_df.filter(pl.col("ownership_nonnull_value_count") > 0)
            .get_column("KYPERMNO")
            .drop_nulls()
            .n_unique()
        ),
        "authority_decision_rows": int(full_authority_decisions_df.height),
        "authority_exception_rows": int(full_authority_exceptions_df.height),
        "authority_decision_status_counts": _group_shares(full_authority_decisions_df, "authority_decision_status"),
        "request_outcome_buckets": (
            full_request_outcomes_df.group_by("outcome_bucket")
            .agg(
                pl.len().alias("request_rows"),
                pl.col("KYPERMNO").drop_nulls().n_unique().alias("distinct_permno"),
            )
            .with_columns(
                (pl.col("request_rows") / pl.lit(full_request_outcomes_df.height)).alias("request_row_share")
            )
            .sort("request_rows", descending=True)
            .to_dicts()
        ),
        "outcomes_by_authority_status": _summarize_request_outcomes(full_request_outcomes_df, "authority_decision_status"),
    }

    diff_flags_df, step1_snapshot_diffs = _summarize_snapshot_diffs(
        current_request_outcomes_df=full_request_outcomes_df,
        current_authority_decisions_df=full_authority_decisions_df,
        snapshot_paths=snapshot_paths,
    )

    _log(args.verbose, "Building deterministic sampled-RIC output...")
    sample_df, sample_metadata = _build_sample_table(
        request_outcomes_df=request_outcomes_df,
        basis_bridge_context_df=basis_bridge_context_df,
        authority_decisions_df=authority_decisions_df,
        authority_exceptions_df=authority_exceptions_df,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    failure_sample_df, failure_sample_metadata = _build_failure_sample_table(
        request_outcomes_df=full_request_outcomes_df,
        authority_exceptions_df=full_authority_exceptions_df,
        diff_flags_df=diff_flags_df,
        sample_size=args.sample_size,
        seed=args.seed,
    )

    institutional_history_by_ric_df = (
        full_results_df.filter(
            pl.col("candidate_ric").is_not_null()
            & pl.col("is_institutional_category")
            & pl.col("returned_value").is_not_null()
        )
        .group_by("candidate_ric")
        .agg(
            pl.col("returned_date").drop_nulls().min().alias("first_observed_institutional_date"),
            pl.col("returned_date").drop_nulls().max().alias("last_observed_institutional_date"),
            pl.len().alias("institutional_history_row_count"),
        )
    )

    if (
        doc_exact_requests_df is not None
        and doc_exact_raw_df is not None
        and doc_fallback_requests_df is not None
        and doc_fallback_raw_df is not None
        and doc_final_df is not None
    ):
        doc_truth_df, doc_ownership_diagnostics = _build_doc_ownership_truth_table(
            exact_requests_df=doc_exact_requests_df,
            exact_raw_df=doc_exact_raw_df,
            fallback_requests_df=doc_fallback_requests_df,
            fallback_raw_df=doc_fallback_raw_df,
            final_df=doc_final_df,
            institutional_history_by_ric_df=institutional_history_by_ric_df,
        )
    else:
        doc_truth_df = pl.DataFrame()
        doc_ownership_diagnostics = {
            "available": False,
            "note": "Doc-level ownership artifacts not found under refinitiv_doc_ownership_lm2011.",
        }

    downstream_diagnostics = _build_downstream_shrinkage_summary(
        downstream_paths=downstream_paths,
        doc_truth_df=(
            doc_truth_df
            if doc_truth_df.height
            else pl.DataFrame(
                schema={
                    "doc_id": pl.Utf8,
                    "KYPERMNO": pl.Int64,
                    "final_has_nonnull_ownership": pl.Boolean,
                }
            )
        ),
    )
    live_api_baseline = _build_live_api_baseline(paths=paths, doc_paths=doc_paths)

    artifact_chain_table = [
        _artifact_stage_stats(
            name="basis_filings",
            df=basis_df,
            date_min_col="filing_date",
            date_max_col="filing_date",
            note="Filtered filing-basis universe used as the diagnostic anchor.",
        ),
        _artifact_stage_stats(
            name="bridge_universe_overlap",
            df=bridge_matched_df,
            date_min_col="first_seen_caldt",
            date_max_col="last_seen_caldt",
            ric_col="vendor_primary_ric",
            note="Bridge rows whose span overlaps at least one basis filing row.",
        ),
        _artifact_stage_stats(
            name="lookup_snapshot_overlap",
            df=lookup_snapshot_df,
            date_min_col="first_seen_caldt",
            date_max_col="last_seen_caldt",
            note="Lookup snapshot rows tied to the matched bridge subset.",
        ),
        _artifact_stage_stats(
            name="lookup_extended_overlap",
            df=lookup_extended_df,
            date_min_col="first_seen_caldt",
            date_max_col="last_seen_caldt",
            note="Extended lookup rows tied to the matched bridge subset.",
        ),
        _artifact_stage_stats(
            name="resolution_overlap",
            df=resolution_df,
            date_min_col="first_seen_caldt",
            date_max_col="last_seen_caldt",
            ric_col="effective_collection_ric",
            note="Resolution rows tied to the matched bridge subset.",
        ),
        _artifact_stage_stats(
            name="instrument_authority_overlap",
            df=instrument_authority_df,
            ric_col="effective_collection_ric",
            note="Instrument-authority rows for the matched bridge subset.",
        ),
        _artifact_stage_stats(
            name="ownership_handoff_overlap",
            df=handoff_df,
            date_min_col="request_start_date",
            date_max_col="request_end_date",
            ric_col="candidate_ric",
            note="Ownership-universe handoff rows for the matched bridge subset.",
        ),
        _artifact_stage_stats(
            name="ownership_results_overlap",
            df=results_df,
            date_min_col="returned_date",
            date_max_col="returned_date",
            ric_col="returned_ric",
            note="Ownership result rows attached to the matched bridge subset.",
        ),
        _artifact_stage_stats(
            name="ownership_row_summary_overlap",
            df=row_summary_df,
            date_min_col="request_start_date",
            date_max_col="request_end_date",
            ric_col="candidate_ric",
            note="Ownership request-level summary rows for the matched bridge subset.",
        ),
        _artifact_stage_stats(
            name="authority_decisions_overlap",
            df=authority_decisions_df,
            ric_col="authoritative_ric",
            note="Ownership authority decisions for basis-relevant PERMNOs.",
        ),
        _artifact_stage_stats(
            name="authority_exceptions_overlap",
            df=authority_exceptions_df,
            date_min_col="authority_window_start_date",
            date_max_col="authority_window_end_date",
            ric_col="authoritative_ric",
            note="Ownership authority exceptions for basis-relevant PERMNOs.",
        ),
    ]

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_markdown_path = output_dir / "refinitiv_lookup_basis_summary.md"
    summary_json_path = output_dir / "refinitiv_lookup_basis_summary.json"
    sample_parquet_path = output_dir / "refinitiv_lookup_basis_sampled_rics.parquet" if args.emit_sample_parquet else None
    sample_csv_path = output_dir / "refinitiv_lookup_basis_sampled_rics.csv" if args.emit_sample_csv else None
    failure_sample_parquet_path = (
        output_dir / "refinitiv_lookup_basis_failure_sampled_rics.parquet"
        if args.emit_failure_sample_parquet
        else None
    )
    failure_sample_csv_path = (
        output_dir / "refinitiv_lookup_basis_failure_sampled_rics.csv"
        if args.emit_failure_sample_csv
        else None
    )
    doc_truth_parquet_path = (
        output_dir / "refinitiv_lookup_basis_doc_ownership_truth.parquet"
        if args.emit_doc_truth_parquet and doc_paths is not None
        else None
    )
    doc_truth_csv_path = (
        output_dir / "refinitiv_lookup_basis_doc_ownership_truth.csv"
        if args.emit_doc_truth_csv and doc_paths is not None
        else None
    )

    summary = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "run_root": str(paths.run_root),
        "filters": {
            "year_start": args.year_start,
            "year_end": args.year_end,
            "sample_size": args.sample_size,
            "seed": args.seed,
        },
        "artifact_paths": {field: str(value) for field, value in paths.__dict__.items()},
        "basis_universe": basis_summary,
        "artifact_chain_table": artifact_chain_table,
        "chain_integrity": chain_integrity,
        "join_back_coverage_table": coverage_ladder,
        "resolution_diagnostics": resolution_diagnostics,
        "authority_diagnostics": authority_diagnostics,
        "ownership_diagnostics": ownership_diagnostics,
        "step1_full_universe": step1_full_universe,
        "step1_snapshot_diffs": step1_snapshot_diffs,
        "doc_ownership_diagnostics": doc_ownership_diagnostics,
        "downstream_diagnostics": downstream_diagnostics,
        "live_api_baseline": live_api_baseline,
        "sampling": {
            **sample_metadata,
            "sample_preview": sample_df.head(10).to_dicts(),
        },
        "failure_sampling": {
            **failure_sample_metadata,
            "sample_preview": failure_sample_df.head(10).to_dicts(),
        },
        "output_paths": {
            "summary_markdown": str(summary_markdown_path),
            "summary_json": str(summary_json_path),
            "sample_parquet": str(sample_parquet_path) if sample_parquet_path is not None else None,
            "sample_csv": str(sample_csv_path) if sample_csv_path is not None else None,
            "failure_sample_parquet": (
                str(failure_sample_parquet_path) if failure_sample_parquet_path is not None else None
            ),
            "failure_sample_csv": str(failure_sample_csv_path) if failure_sample_csv_path is not None else None,
            "doc_truth_parquet": str(doc_truth_parquet_path) if doc_truth_parquet_path is not None else None,
            "doc_truth_csv": str(doc_truth_csv_path) if doc_truth_csv_path is not None else None,
        },
    }
    report_text = _render_report(summary)

    _write_json(summary_json_path, summary)
    _write_text(summary_markdown_path, report_text)
    if sample_parquet_path is not None:
        sample_df.write_parquet(sample_parquet_path, compression="zstd")
    if sample_csv_path is not None:
        sample_df.write_csv(sample_csv_path)
    if failure_sample_parquet_path is not None:
        failure_sample_df.write_parquet(failure_sample_parquet_path, compression="zstd")
    if failure_sample_csv_path is not None:
        failure_sample_df.write_csv(failure_sample_csv_path)
    if doc_truth_parquet_path is not None:
        doc_truth_df.write_parquet(doc_truth_parquet_path, compression="zstd")
    if doc_truth_csv_path is not None:
        doc_truth_df.write_csv(doc_truth_csv_path)

    return {
        "summary": summary,
        "summary_markdown_path": summary_markdown_path,
        "summary_json_path": summary_json_path,
        "sample_parquet_path": sample_parquet_path,
        "sample_csv_path": sample_csv_path,
        "sample_df": sample_df,
        "failure_sample_parquet_path": failure_sample_parquet_path,
        "failure_sample_csv_path": failure_sample_csv_path,
        "failure_sample_df": failure_sample_df,
        "doc_truth_parquet_path": doc_truth_parquet_path,
        "doc_truth_csv_path": doc_truth_csv_path,
        "doc_truth_df": doc_truth_df,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = _resolve_run_paths(args.run_root)
    _ensure_required_inputs(paths)
    result = run_analysis(args, paths)
    print(
        json.dumps(
            {
                "summary_markdown_path": str(result["summary_markdown_path"]),
                "summary_json_path": str(result["summary_json_path"]),
                "sample_parquet_path": (
                    str(result["sample_parquet_path"]) if result["sample_parquet_path"] is not None else None
                ),
                "sample_csv_path": str(result["sample_csv_path"]) if result["sample_csv_path"] is not None else None,
                "sample_rows": int(result["sample_df"].height),
                "failure_sample_parquet_path": (
                    str(result["failure_sample_parquet_path"])
                    if result["failure_sample_parquet_path"] is not None
                    else None
                ),
                "failure_sample_csv_path": (
                    str(result["failure_sample_csv_path"]) if result["failure_sample_csv_path"] is not None else None
                ),
                "failure_sample_rows": int(result["failure_sample_df"].height),
                "doc_truth_parquet_path": (
                    str(result["doc_truth_parquet_path"]) if result["doc_truth_parquet_path"] is not None else None
                ),
                "doc_truth_csv_path": (
                    str(result["doc_truth_csv_path"]) if result["doc_truth_csv_path"] is not None else None
                ),
                "doc_truth_rows": int(result["doc_truth_df"].height),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

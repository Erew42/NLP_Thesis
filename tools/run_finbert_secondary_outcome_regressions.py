from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from typing import Sequence

import polars as pl


REPO_ROOT_ENV_VAR = "NLP_THESIS_REPO_ROOT"
DEFAULT_EXTENSION_PANEL_RELATIVE = Path(
    "full_data_run"
    "/lm2011_extension-20260423T070920Z-3-001"
    "/lm2011_extension"
    "/replication"
    "/lm2011_extension_analysis_panel.parquet"
)
DEFAULT_SECONDARY_OUTCOMES: tuple[str, ...] = (
    "abnormal_volume",
    "postevent_return_volatility",
)
DEFAULT_TEXT_SCOPES: tuple[str, ...] = (
    "item_7_mda",
    "item_1a_risk_factors",
)
DEFAULT_CONTROL_SET_IDS: tuple[str, ...] = ("C0",)
DEFAULT_SPECIFICATION_NAMES: tuple[str, ...] = (
    "dictionary_only",
    "finbert_only",
    "dictionary_finbert_joint",
)
COEFFICIENTS_FILENAME = "finbert_secondary_outcome_coefficients.parquet"
FIT_SUMMARY_FILENAME = "finbert_secondary_outcome_fit_summary.parquet"
FIT_COMPARISONS_FILENAME = "finbert_secondary_outcome_fit_comparisons.parquet"
FIT_SKIPPED_QUARTERS_FILENAME = "finbert_secondary_outcome_fit_skipped_quarters.parquet"
MANIFEST_FILENAME = "finbert_secondary_outcome_run_manifest.json"


def _resolve_repo_root() -> Path:
    candidates: list[Path] = []
    env_root = os.environ.get(REPO_ROOT_ENV_VAR)
    if env_root:
        candidates.append(Path(env_root).expanduser().resolve())

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

from thesis_pkg.pipelines.lm2011_extension import EXTENSION_PRIMARY_OUTCOME
from thesis_pkg.pipelines.lm2011_extension import run_lm2011_extension_estimation_scaffold
from thesis_pkg.pipelines.lm2011_extension import run_lm2011_extension_fit_comparison_scaffold


def _utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _safe_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")


def _unique_preserving_order(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)


def _default_output_dir() -> Path:
    return ROOT / "full_data_run" / f"finbert_secondary_outcomes_{_safe_timestamp()}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run add-on FinBERT secondary-outcome regressions from an existing "
            "LM2011 extension analysis panel. This does not rerun FinBERT or read "
            "sentence-score shards."
        )
    )
    parser.add_argument(
        "--extension-panel-path",
        type=Path,
        default=ROOT / DEFAULT_EXTENSION_PANEL_RELATIVE,
        help="Path to lm2011_extension_analysis_panel.parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to full_data_run/finbert_secondary_outcomes_<timestamp>.",
    )
    parser.add_argument(
        "--outcomes",
        nargs="+",
        default=list(DEFAULT_SECONDARY_OUTCOMES),
        help="Dependent variables to run.",
    )
    parser.add_argument(
        "--text-scopes",
        nargs="+",
        default=list(DEFAULT_TEXT_SCOPES),
        help="Extension text scopes to include.",
    )
    parser.add_argument(
        "--control-set-ids",
        nargs="+",
        default=list(DEFAULT_CONTROL_SET_IDS),
        help="Extension control sets to include.",
    )
    parser.add_argument(
        "--specification-names",
        nargs="+",
        default=list(DEFAULT_SPECIFICATION_NAMES),
        help="Extension specification names to include.",
    )
    parser.add_argument(
        "--nw-lags",
        type=int,
        default=1,
        help="Newey-West lag count for quarterly coefficient time series.",
    )
    parser.add_argument(
        "--include-primary-return",
        action="store_true",
        help="Also include filing_period_excess_return as a same-run benchmark.",
    )
    return parser.parse_args(argv)


def _resolved_output_dir(raw_output_dir: Path | None) -> Path:
    return (raw_output_dir if raw_output_dir is not None else _default_output_dir()).resolve()


def _resolved_outcomes(args: argparse.Namespace) -> tuple[str, ...]:
    outcomes = tuple(str(outcome) for outcome in args.outcomes)
    if args.include_primary_return:
        outcomes = (EXTENSION_PRIMARY_OUTCOME, *outcomes)
    return _unique_preserving_order(outcomes)


def _write_frame_with_csv(df: pl.DataFrame, parquet_path: Path) -> dict[str, str]:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = parquet_path.with_suffix(".csv")
    df.write_parquet(parquet_path, compression="zstd")
    csv_df = df.with_columns(
        [
            pl.col(column).map_elements(_json_stringify_nested_value, return_dtype=pl.Utf8)
            for column, dtype in df.schema.items()
            if dtype.base_type() == pl.List
        ]
    )
    csv_df.write_csv(csv_path)
    return {
        "parquet": str(parquet_path),
        "csv": str(csv_path),
    }


def _json_stringify_nested_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, pl.Series):
        return json.dumps(value.to_list())
    return json.dumps(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _status_counts(df: pl.DataFrame, *, status_column: str = "estimator_status") -> list[dict[str, Any]]:
    if df.height == 0 or status_column not in df.columns:
        return []
    group_columns = [
        column
        for column in ("outcome_name", status_column)
        if column in df.columns
    ]
    if not group_columns:
        return []
    return (
        df.group_by(group_columns)
        .agg(pl.len().alias("rows"))
        .sort(group_columns)
        .to_dicts()
    )


def _input_panel_summary(panel_path: Path, outcomes: Sequence[str]) -> dict[str, Any]:
    lf = pl.scan_parquet(panel_path)
    schema = lf.collect_schema()
    names = set(schema.names())
    required_columns = {
        "filing_date",
        "text_scope",
        "ff48_industry_id",
        "lm_negative_tfidf",
        "finbert_neg_prob_lenw_mean",
        "log_size",
        "log_book_to_market",
        "log_share_turnover",
        "pre_ffalpha",
        "nasdaq_dummy",
        *outcomes,
    }
    missing_columns = sorted(required_columns - names)
    if missing_columns:
        raise ValueError(
            "Extension panel is missing required columns: "
            + ", ".join(missing_columns)
        )

    exprs: list[pl.Expr] = [pl.len().alias("rows")]
    for outcome in outcomes:
        exprs.append(pl.col(outcome).is_not_null().sum().alias(f"{outcome}__non_null"))
    counts = lf.select(exprs).collect().row(0, named=True)
    scope_counts = (
        lf.group_by("text_scope")
        .agg(pl.len().alias("rows"))
        .sort("text_scope")
        .collect()
        .to_dicts()
    )
    return {
        "schema_columns": schema.names(),
        "row_counts": counts,
        "text_scope_counts": scope_counts,
    }


def run_secondary_outcome_regressions(args: argparse.Namespace) -> dict[str, Any]:
    started_at = _utc_timestamp()
    start = time.perf_counter()

    panel_path = args.extension_panel_path.resolve()
    if not panel_path.exists():
        raise FileNotFoundError(f"Extension panel not found: {panel_path}")
    output_dir = _resolved_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outcomes = _resolved_outcomes(args)
    text_scopes = tuple(str(scope) for scope in args.text_scopes)
    control_set_ids = tuple(str(control_set_id) for control_set_id in args.control_set_ids)
    specification_names = tuple(str(name) for name in args.specification_names)

    panel_summary = _input_panel_summary(panel_path, outcomes)
    panel_lf = pl.scan_parquet(panel_path)
    run_id = output_dir.name

    coefficients_df = run_lm2011_extension_estimation_scaffold(
        panel_lf,
        run_id=run_id,
        text_scopes=text_scopes,
        outcome_names=outcomes,
        control_set_ids=control_set_ids,
        specification_names=specification_names,
        nw_lags=args.nw_lags,
    )
    fit_artifacts = run_lm2011_extension_fit_comparison_scaffold(
        pl.scan_parquet(panel_path),
        run_id=run_id,
        text_scopes=text_scopes,
        outcome_names=outcomes,
        control_set_ids=control_set_ids,
        specification_names=specification_names,
        nw_lags=args.nw_lags,
    )

    artifact_paths = {
        "coefficients": _write_frame_with_csv(coefficients_df, output_dir / COEFFICIENTS_FILENAME),
        "fit_summary": _write_frame_with_csv(fit_artifacts.summary_df, output_dir / FIT_SUMMARY_FILENAME),
        "fit_comparisons": _write_frame_with_csv(
            fit_artifacts.comparison_df,
            output_dir / FIT_COMPARISONS_FILENAME,
        ),
        "fit_skipped_quarters": _write_frame_with_csv(
            fit_artifacts.skipped_quarters_df,
            output_dir / FIT_SKIPPED_QUARTERS_FILENAME,
        ),
    }

    completed_at = _utc_timestamp()
    elapsed_seconds = round(time.perf_counter() - start, 3)
    manifest = {
        "runner_name": "finbert_secondary_outcome_regressions",
        "run_id": run_id,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "elapsed_seconds": elapsed_seconds,
        "resolved_inputs": {
            "extension_panel_path": str(panel_path),
        },
        "config": {
            "outcomes": list(outcomes),
            "text_scopes": list(text_scopes),
            "control_set_ids": list(control_set_ids),
            "specification_names": list(specification_names),
            "nw_lags": args.nw_lags,
            "include_primary_return": bool(args.include_primary_return),
        },
        "input_panel": panel_summary,
        "artifacts": artifact_paths,
        "row_counts": {
            "coefficients": coefficients_df.height,
            "fit_summary": fit_artifacts.summary_df.height,
            "fit_comparisons": fit_artifacts.comparison_df.height,
            "fit_skipped_quarters": fit_artifacts.skipped_quarters_df.height,
        },
        "status_counts": {
            "coefficients": _status_counts(coefficients_df),
            "fit_summary": _status_counts(fit_artifacts.summary_df),
            "fit_comparisons": _status_counts(fit_artifacts.comparison_df),
        },
    }
    manifest_path = output_dir / MANIFEST_FILENAME
    _write_json(manifest_path, manifest)

    return {
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "row_counts": manifest["row_counts"],
        "status_counts": manifest["status_counts"],
        "elapsed_seconds": elapsed_seconds,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_secondary_outcome_regressions(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

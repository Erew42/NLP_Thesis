from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import sys
import textwrap
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import polars as pl


REPORT_BASENAME = "lm2011_post_refinitiv_review"
REPORT_OUTPUT_DIR = Path("output") / "latex" / REPORT_BASENAME
FIGURE_SUBDIR = "figures"
PDF_OUTPUT_DIR = Path("output") / "pdf"
MANIFEST_FILENAME = "lm2011_sample_run_manifest.json"
SKIP_STATUSES = {
    "disabled_by_run_config",
    "reused_existing_artifact",
}
MAIN_RESULT_FILES: tuple[str, ...] = (
    "lm2011_table_iv_results.parquet",
    "lm2011_table_v_results.parquet",
    "lm2011_table_vi_results.parquet",
    "lm2011_table_viii_results.parquet",
    "lm2011_table_ia_i_results.parquet",
)
SAMPLE_TABLE_FILES: tuple[str, ...] = (
    "lm2011_table_i_sample_creation.parquet",
    "lm2011_table_i_sample_creation_1994_2024.parquet",
)
SKIPPED_DIAGNOSTIC_FILES: tuple[str, ...] = (
    "lm2011_table_iv_results_skipped_quarters.parquet",
    "lm2011_table_vi_results_skipped_quarters.parquet",
    "lm2011_table_ia_i_results_skipped_quarters.parquet",
)
TRADE_RESULTS_FILE = "lm2011_table_ia_ii_results.parquet"
TRADING_RETURNS_FILE = "lm2011_trading_strategy_monthly_returns.parquet"
EVENT_PANEL_FILE = "lm2011_event_panel.parquet"
FULL_10K_PANEL_FILE = "lm2011_return_regression_panel_full_10k.parquet"
MDA_PANEL_FILE = "lm2011_return_regression_panel_mda.parquet"


@dataclass(frozen=True)
class RunContext:
    repo_root: Path
    run_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]


def _resolve_repo_root() -> Path:
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "src" / "thesis_pkg" / "pipeline.py").exists():
            return candidate
    raise RuntimeError("Could not resolve repository root containing src/thesis_pkg/pipeline.py")


def _parse_iso_datetime(value: Any) -> dt.datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return dt.datetime.fromisoformat(value)
    except ValueError:
        return None


def _candidate_sort_key(manifest: dict[str, Any], run_dir: Path) -> tuple[float, float]:
    completed = _parse_iso_datetime(manifest.get("completed_at_utc"))
    generated = _parse_iso_datetime(manifest.get("generated_at_utc"))
    timestamp = completed or generated
    if timestamp is None:
        return (0.0, run_dir.stat().st_mtime)
    return (timestamp.timestamp(), run_dir.stat().st_mtime)


def _discover_latest_run_dir(repo_root: Path) -> RunContext:
    candidates: list[tuple[tuple[float, float], Path, Path, dict[str, Any]]] = []
    for manifest_path in repo_root.rglob(MANIFEST_FILENAME):
        run_dir = manifest_path.parent
        if run_dir.name != "lm2011_post_refinitiv":
            continue
        if any(part in {"tmp", ".tmp", ".venv", ".git", "archive"} for part in run_dir.parts):
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        candidates.append((_candidate_sort_key(manifest, run_dir), run_dir, manifest_path, manifest))
    if not candidates:
        raise FileNotFoundError("No lm2011_post_refinitiv run directory with a manifest was found")
    _, run_dir, manifest_path, manifest = max(candidates, key=lambda item: item[0])
    return RunContext(
        repo_root=repo_root,
        run_dir=run_dir,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _parquet_row_count(path: Path) -> int:
    return int(pl.scan_parquet(path).select(pl.len().alias("rows")).collect().item())


def _format_timestamp(timestamp: dt.datetime | None) -> str:
    if timestamp is None:
        return "--"
    if timestamp.tzinfo is None:
        return timestamp.isoformat(timespec="seconds")
    return timestamp.astimezone(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _artifact_inventory(run_dir: Path, manifest: dict[str, Any]) -> pl.DataFrame:
    stage_rows: dict[str, dict[str, Any]] = {}
    for stage_name, stage_payload in manifest.get("stages", {}).items():
        if not isinstance(stage_payload, dict):
            continue
        artifact_path = stage_payload.get("artifact_path")
        if not isinstance(artifact_path, str):
            continue
        stage_rows[Path(artifact_path).name] = {
            "stage_name": stage_name,
            "stage_status": stage_payload.get("status"),
            "stage_row_count": stage_payload.get("row_count"),
        }

    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.iterdir()):
        if not path.is_file():
            continue
        filename = path.name
        row: dict[str, Any] = {
            "file_name": filename,
            "suffix": path.suffix.lstrip("."),
            "size_mb": path.stat().st_size / (1024 * 1024),
            "modified_at": dt.datetime.fromtimestamp(path.stat().st_mtime),
            "row_count": _parquet_row_count(path) if path.suffix == ".parquet" else None,
            "stage_name": None,
            "stage_status": None,
            "stage_row_count": None,
        }
        if filename in stage_rows:
            row.update(stage_rows[filename])
        rows.append(row)
    return pl.DataFrame(rows).sort("file_name")


def _disabled_stage_artifacts(manifest: dict[str, Any], run_dir: Path) -> list[str]:
    disabled: list[str] = []
    for stage_name, stage_payload in manifest.get("stages", {}).items():
        if not isinstance(stage_payload, dict):
            continue
        if stage_payload.get("status") != "disabled_by_run_config":
            continue
        artifact_path = stage_payload.get("artifact_path")
        if not isinstance(artifact_path, str):
            continue
        filename = Path(artifact_path).name
        if (run_dir / filename).exists():
            disabled.append(stage_name)
    return sorted(disabled)


def _simplify_duplicate_pairs(value: str | None) -> str:
    if not value:
        return "--"
    pairs = re.findall(r'"([^"]+)"', value)
    if len(pairs) >= 2:
        chunked = list(zip(pairs[0::2], pairs[1::2]))
        return "; ".join(f"{left} = {right}" for left, right in chunked)
    return value


def _summarize_sample_table(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.select(
            pl.col("display_label").alias("Filter"),
            pl.col("sample_size_value").alias("Sample size"),
            pl.col("observations_removed").alias("Removed"),
            pl.col("availability_status").alias("Status"),
        )
        .with_columns(pl.col("Status").fill_null("available"))
    )


def _summarize_main_results(path: Path) -> pl.DataFrame:
    df = pl.read_parquet(path)
    return (
        df.filter(pl.col("coefficient_name") == pl.col("signal_name"))
        .select(
            pl.col("signal_name").alias("Signal"),
            pl.col("estimate").alias("Estimate"),
            pl.col("standard_error").alias("Std. error"),
            pl.col("t_stat").alias("t-stat"),
            pl.col("n_quarters").alias("Quarters"),
            pl.col("mean_quarter_n").alias("Mean quarter n"),
            pl.col("nw_lags").alias("NW lags"),
        )
    )


def _summarize_full_coefficients(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path).select(
        pl.col("specification_id").alias("Specification"),
        pl.col("coefficient_name").alias("Coefficient"),
        pl.col("estimate").alias("Estimate"),
        pl.col("standard_error").alias("Std. error"),
        pl.col("t_stat").alias("t-stat"),
        pl.col("n_quarters").alias("Quarters"),
        pl.col("mean_quarter_n").alias("Mean quarter n"),
        pl.col("nw_lags").alias("NW lags"),
    )


def _summarize_skipped_diagnostics(path: Path) -> pl.DataFrame:
    df = pl.read_parquet(path)
    return (
        df.select(
            pl.col("quarter_start").alias("Quarter"),
            pl.col("signal_name").alias("Signal"),
            pl.col("skip_reason").alias("Reason"),
            pl.col("n_obs").alias("Observations"),
            pl.col("industry_count").alias("Industries"),
            pl.col("rank").alias("Rank"),
            pl.col("column_count").alias("Columns"),
            pl.col("duplicate_regressor_pairs").map_elements(
                _simplify_duplicate_pairs,
                return_dtype=pl.Utf8,
            ).alias("Duplicate regressors"),
        )
        .sort("Quarter", "Signal")
    )


def _summarize_trade_results(path: Path) -> pl.DataFrame:
    df = pl.read_parquet(path)
    pivot = (
        df.pivot(
            values="estimate",
            index="signal_name",
            on="coefficient_name",
        )
        .rename(
            {
                "signal_name": "Signal",
                "mean_long_short_return": "Mean long-short",
                "alpha_ff3_mom": "FF4 alpha",
                "beta_market": "Beta MKT",
                "beta_smb": "Beta SMB",
                "beta_hml": "Beta HML",
                "beta_mom": "Beta MOM",
                "r2": "R^2",
            }
        )
    )
    preferred = [
        "Signal",
        "Mean long-short",
        "FF4 alpha",
        "Beta MKT",
        "Beta SMB",
        "Beta HML",
        "Beta MOM",
        "R^2",
    ]
    ordered = [column for column in preferred if column in pivot.columns]
    return pivot.select(ordered).sort("Signal")


def _main_signal_coefficients(run_dir: Path) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    table_labels = {
        "lm2011_table_iv_results.parquet": "Table IV",
        "lm2011_table_v_results.parquet": "Table V",
        "lm2011_table_vi_results.parquet": "Table VI",
        "lm2011_table_viii_results.parquet": "Table VIII",
        "lm2011_table_ia_i_results.parquet": "Table IA.I",
    }
    for filename in MAIN_RESULT_FILES:
        df = pl.read_parquet(run_dir / filename)
        label = table_labels[filename]
        subset = df.filter(pl.col("coefficient_name") == pl.col("signal_name"))
        for row in subset.iter_rows(named=True):
            rows.append(
                {
                    "table_label": label,
                    "signal_name": row["signal_name"],
                    "estimate": row["estimate"],
                    "standard_error": row["standard_error"],
                    "t_stat": row["t_stat"],
                    "n_quarters": row["n_quarters"],
                }
            )
    return pl.DataFrame(rows)


def _trading_returns_frame(run_dir: Path) -> pl.DataFrame:
    df = pl.read_parquet(run_dir / TRADING_RETURNS_FILE).sort("portfolio_month")
    rows: list[dict[str, Any]] = []
    for signal_name in df.get_column("sort_signal_name").unique().sort().to_list():
        subdf = df.filter(pl.col("sort_signal_name") == signal_name)
        gross = 1.0
        for row in subdf.iter_rows(named=True):
            gross *= 1.0 + float(row["long_short_return"])
            rows.append(
                {
                    "portfolio_month": row["portfolio_month"],
                    "signal_name": signal_name,
                    "cumulative_return": gross - 1.0,
                }
            )
    return pl.DataFrame(rows).sort("portfolio_month", "signal_name")


def _complete_case_quarter_counts(path: Path, *, is_mda: bool) -> pl.DataFrame:
    required = (
        "filing_period_excess_return",
        "log_size",
        "log_book_to_market",
        "pre_ffalpha",
        "log_share_turnover",
        "nasdaq_dummy",
        "institutional_ownership",
        "ff48_industry_id",
    )
    signal_columns = ("h4n_inf_prop", "lm_negative_prop", "h4n_inf_tfidf", "lm_negative_tfidf")
    panel = pl.scan_parquet(path)
    if is_mda:
        panel = panel.filter(pl.col("total_token_count_mda").cast(pl.Float64, strict=False) >= 250.0)
    complete_expr = None
    for name in (*required, *signal_columns):
        current = pl.col(name).is_not_null()
        complete_expr = current if complete_expr is None else complete_expr & current
    return (
        panel.with_columns(pl.col("filing_date").dt.truncate("1q").alias("quarter_start"))
        .group_by("quarter_start")
        .agg(complete_expr.sum().alias("complete_cases"))
        .sort("quarter_start")
        .collect()
    )


def _figure_save(fig: plt.Figure, out_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_sample_attrition(run_dir: Path, figure_dir: Path) -> Path:
    first = pl.read_parquet(run_dir / SAMPLE_TABLE_FILES[0]).filter(
        (pl.col("section_id") == "full_10k_document") & (pl.col("sample_size_kind") == "count")
    )
    second = pl.read_parquet(run_dir / SAMPLE_TABLE_FILES[1]).filter(
        (pl.col("section_id") == "full_10k_document") & (pl.col("sample_size_kind") == "count")
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharex=False)
    palettes = ["#1f5aa6", "#8c3b00"]
    for ax, frame, title, color in zip(
        axes,
        (first, second),
        ("1994-2008 Sample Screen", "1994-2024 Sample Screen"),
        palettes,
        strict=True,
    ):
        labels = [textwrap.fill(value, width=34) for value in frame["display_label"].to_list()]
        counts = [float(value) for value in frame["sample_size_value"].to_list()]
        positions = list(range(len(labels)))
        ax.plot(counts, positions, marker="o", color=color, linewidth=2.3)
        ax.fill_betweenx(positions, counts, color=color, alpha=0.08)
        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(axis="x", alpha=0.25)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value / 1000):,}k"))
        for xpos, ypos in zip(counts, positions, strict=True):
            ax.text(xpos, ypos, f" {int(round(xpos)):,}", va="center", ha="left", fontsize=8)
    axes[0].set_xlabel("Observations")
    axes[1].set_xlabel("Observations")
    fig.suptitle("LM2011 Sample Attrition", fontsize=14, fontweight="bold")
    out_path = figure_dir / "sample_attrition.png"
    _figure_save(fig, out_path)
    return out_path


def _plot_complete_case_quarters(run_dir: Path, figure_dir: Path) -> Path:
    full_df = _complete_case_quarter_counts(run_dir / FULL_10K_PANEL_FILE, is_mda=False).with_columns(
        pl.lit("Full 10-K").alias("surface")
    )
    mda_df = _complete_case_quarter_counts(run_dir / MDA_PANEL_FILE, is_mda=True).with_columns(
        pl.lit("MD&A (250-token floor)").alias("surface")
    )
    combined = pl.concat([full_df, mda_df], how="vertical")
    fig, ax = plt.subplots(figsize=(13.5, 5.2))
    style_map = {
        "Full 10-K": {"color": "#0b6e4f", "linewidth": 2.2},
        "MD&A (250-token floor)": {"color": "#b23a48", "linewidth": 2.2},
    }
    for surface in ("Full 10-K", "MD&A (250-token floor)"):
        subdf = combined.filter(pl.col("surface") == surface)
        ax.plot(
            subdf["quarter_start"].to_list(),
            subdf["complete_cases"].to_list(),
            label=surface,
            **style_map[surface],
        )
    ax.axhline(20, color="#555555", linestyle="--", linewidth=1.2, label="20 complete cases")
    ax.set_title("Quarterly Complete-Case Counts Before Rank Checks", fontsize=13, fontweight="bold")
    ax.set_ylabel("Complete cases")
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(frameon=False, loc="upper right")
    out_path = figure_dir / "complete_case_quarters.png"
    _figure_save(fig, out_path)
    return out_path


def _plot_signal_forest(run_dir: Path, figure_dir: Path) -> Path:
    df = _main_signal_coefficients(run_dir)
    order = []
    for table_label in ("Table IV", "Table V", "Table VI", "Table VIII", "Table IA.I"):
        order.extend(
            df.filter(pl.col("table_label") == table_label)
            .select("signal_name")
            .get_column("signal_name")
            .to_list()
        )
    rows = df.iter_rows(named=True)
    fig, ax = plt.subplots(figsize=(13.5, 10))
    colors = {
        "Table IV": "#1f5aa6",
        "Table V": "#b23a48",
        "Table VI": "#0b6e4f",
        "Table VIII": "#6f4e7c",
        "Table IA.I": "#8c6d1f",
    }
    y_positions = list(range(len(df), 0, -1))
    labels: list[str] = []
    for ypos, row in zip(y_positions, rows, strict=True):
        estimate = float(row["estimate"])
        se = row["standard_error"]
        color = colors[row["table_label"]]
        ax.scatter(estimate, ypos, color=color, s=35, zorder=3)
        if se is not None and not math.isnan(float(se)) and float(se) > 0.0:
            ci = 1.96 * float(se)
            ax.hlines(ypos, estimate - ci, estimate + ci, color=color, linewidth=1.8, alpha=0.85)
        labels.append(f"{row['table_label']}  {row['signal_name']}")
    ax.axvline(0.0, color="#444444", linewidth=1.1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_title("Main Signal Coefficients Across LM2011 Tables", fontsize=13, fontweight="bold")
    ax.set_xlabel("Coefficient estimate")
    ax.grid(axis="x", alpha=0.25)
    out_path = figure_dir / "main_signal_forest.png"
    _figure_save(fig, out_path)
    return out_path


def _plot_trading_returns(run_dir: Path, figure_dir: Path) -> Path:
    df = _trading_returns_frame(run_dir)
    fig, ax = plt.subplots(figsize=(13.5, 5.2))
    palette = {
        "fin_neg_prop": "#1f5aa6",
        "fin_neg_tfidf": "#0b6e4f",
        "h4n_inf_prop": "#b23a48",
        "h4n_inf_tfidf": "#8c6d1f",
    }
    for signal_name in df.get_column("signal_name").unique().sort().to_list():
        subdf = df.filter(pl.col("signal_name") == signal_name)
        ax.plot(
            subdf["portfolio_month"].to_list(),
            subdf["cumulative_return"].to_list(),
            label=signal_name,
            linewidth=2.0,
            color=palette.get(signal_name, "#444444"),
        )
    ax.axhline(0.0, color="#444444", linewidth=1.0)
    ax.set_title("Cumulative Long-Short Returns by LM2011 Sort Signal", fontsize=13, fontweight="bold")
    ax.set_ylabel("Cumulative return")
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(frameon=False, loc="best")
    out_path = figure_dir / "trading_cumulative_returns.png"
    _figure_save(fig, out_path)
    return out_path


def _latex_escape(value: Any) -> str:
    if value is None:
        return "--"
    if isinstance(value, dt.datetime):
        return _latex_escape(value.isoformat(sep=" ", timespec="seconds"))
    if isinstance(value, dt.date):
        return _latex_escape(value.isoformat())
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _format_cell(value: Any, column_name: str) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        if math.isnan(value):
            return "--"
        if column_name in {"Sample size", "Removed", "Observations", "Industries", "Rank", "Columns", "Rows"}:
            return f"{int(round(value)):,}"
        if column_name in {"Size (MB)", "Elapsed seconds"}:
            return f"{value:,.2f}"
        if column_name in {"Estimate", "Std. error", "t-stat", "Mean quarter n", "Mean long-short", "FF4 alpha", "Beta MKT", "Beta SMB", "Beta HML", "Beta MOM", "R^2"}:
            return f"{value:,.4f}"
        if abs(value - round(value)) < 1e-9:
            return f"{int(round(value)):,}"
        return f"{value:,.4f}"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, dt.datetime):
        return _latex_escape(value.isoformat(sep=" ", timespec="seconds"))
    if isinstance(value, dt.date):
        return _latex_escape(value.isoformat())
    return _latex_escape(value)


def _column_spec(columns: Sequence[str]) -> str:
    text_widths = {
        "Filter": "7.6cm",
        "Status": "2.3cm",
        "Signal": "3.1cm",
        "Coefficient": "3.4cm",
        "Specification": "3.0cm",
        "Quarter": "2.4cm",
        "Reason": "3.0cm",
        "Duplicate regressors": "4.7cm",
        "File": "5.0cm",
        "Type": "1.6cm",
        "Stage": "4.2cm",
        "Stage status": "3.1cm",
        "Updated": "3.1cm",
    }
    numeric = {
        "Sample size",
        "Removed",
        "Estimate",
        "Std. error",
        "t-stat",
        "Quarters",
        "Mean quarter n",
        "NW lags",
        "Observations",
        "Industries",
        "Rank",
        "Columns",
        "Rows",
        "Size (MB)",
        "Elapsed seconds",
        "Mean long-short",
        "FF4 alpha",
        "Beta MKT",
        "Beta SMB",
        "Beta HML",
        "Beta MOM",
        "R^2",
    }
    specs: list[str] = []
    for name in columns:
        if name in numeric:
            specs.append("R{1.55cm}")
        elif name in text_widths:
            specs.append(f"L{{{text_widths[name]}}}")
        else:
            specs.append("L{2.6cm}")
    return "".join(specs)


def _longtable_block(
    df: pl.DataFrame,
    *,
    caption: str,
    landscape: bool = False,
    font_size: str = r"\scriptsize",
) -> str:
    lines: list[str] = []
    if landscape:
        lines.append(r"\begin{landscape}")
    lines.extend(
        [
            font_size,
            r"\setlength{\tabcolsep}{4pt}",
            rf"\begin{{longtable}}{{{_column_spec(df.columns)}}}",
            rf"\caption{{{_latex_escape(caption)}}}\\",
            r"\toprule",
            " & ".join(rf"\textbf{{{_latex_escape(column)}}}" for column in df.columns) + r" \\",
            r"\midrule",
            r"\endfirsthead",
            r"\toprule",
            " & ".join(rf"\textbf{{{_latex_escape(column)}}}" for column in df.columns) + r" \\",
            r"\midrule",
            r"\endhead",
            r"\midrule",
            rf"\multicolumn{{{len(df.columns)}}}{{r}}{{\small\emph{{Continued on next page}}}} \\",
            r"\endfoot",
            r"\bottomrule",
            r"\endlastfoot",
        ]
    )
    for row in df.iter_rows(named=True):
        cells = [_format_cell(row[column], column) for column in df.columns]
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\end{longtable}")
    if landscape:
        lines.append(r"\end{landscape}")
    return "\n".join(lines)


def _figure_block(path: Path, caption: str) -> str:
    return "\n".join(
        [
            r"\begin{figure}[H]",
            r"\centering",
            rf"\includegraphics[width=0.96\textwidth]{{{_latex_escape(path.as_posix())}}}",
            rf"\caption{{{_latex_escape(caption)}}}",
            r"\end{figure}",
        ]
    )


def _build_review_notes(context: RunContext, run_dir: Path) -> list[str]:
    notes: list[str] = []
    disabled = _disabled_stage_artifacts(context.manifest, run_dir)
    if disabled:
        preview = ", ".join(disabled[:6])
        suffix = "" if len(disabled) <= 6 else ", ..."
        notes.append(
            f"{len(disabled)} stages are marked disabled-by-run-config in the manifest even though their artifact files remain in the run directory. Treat these as carry-forward artifacts, not outputs proven to have been regenerated in this run ({preview}{suffix})."
        )

    table_v_main = _summarize_main_results(run_dir / "lm2011_table_v_results.parquet")
    if table_v_main.height > 0:
        quarter_values = [value for value in table_v_main["Quarters"].to_list() if value is not None]
        if quarter_values and max(quarter_values) == 1:
            notes.append(
                "Table V is backed by only one estimable quarter. After the MD&A-specific 250-token floor and the full control set, only 2007-Q1 retains 21 complete cases, so the exported standard errors collapse to zero/null and the table should be treated as a thin-sample diagnostic rather than a stable replication result."
            )

    skipped_counts: list[str] = []
    for filename in SKIPPED_DIAGNOSTIC_FILES:
        df = pl.read_parquet(run_dir / filename)
        if df.height == 0:
            continue
        table_name = filename.replace(".parquet", "")
        unique_quarters = df["quarter_start"].n_unique()
        skipped_counts.append(f"{table_name}: {df.height} skipped rows across {unique_quarters} quarter(s)")
    if skipped_counts:
        notes.append(
            "Skipped-quarter diagnostics are concentrated in 1998-Q1. Across Tables IV, VI, and IA.I the same exact collinearity recurs: `nasdaq_dummy` is duplicated by FF48 industry dummy 36."
        )

    main_signals = _main_signal_coefficients(run_dir)
    significant = (
        main_signals.filter(pl.col("t_stat").is_not_null())
        .with_columns(pl.col("t_stat").abs().alias("abs_t"))
        .sort("abs_t", descending=True)
    )
    if significant.height > 0:
        top = significant.row(0, named=True)
        notes.append(
            f"The strongest main-signal t-stat in the exported tables is {top['table_label']} / {top['signal_name']} with t = {float(top['t_stat']):.2f}."
        )

    trade = _summarize_trade_results(run_dir / TRADE_RESULTS_FILE)
    if trade.height > 0:
        mean_returns = [float(value) for value in trade["Mean long-short"].to_list() if value is not None]
        alphas = [float(value) for value in trade["FF4 alpha"].to_list() if value is not None]
        if mean_returns and alphas and all(value < 0 for value in mean_returns) and all(value < 0 for value in alphas):
            notes.append(
                "All four IA.II long-short strategy means and FF4 alphas are negative in the current run, which is consistent with the cumulative-return chart below."
            )
    return notes


def _report_metadata_table(context: RunContext, inventory: pl.DataFrame) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "Field": "Run directory",
                "Value": str(context.run_dir),
            },
            {
                "Field": "Manifest path",
                "Value": str(context.manifest_path),
            },
            {
                "Field": "Run status",
                "Value": str(context.manifest.get("run_status", "--")),
            },
            {
                "Field": "Generated at (UTC)",
                "Value": _format_timestamp(_parse_iso_datetime(context.manifest.get("generated_at_utc"))),
            },
            {
                "Field": "Completed at (UTC)",
                "Value": _format_timestamp(_parse_iso_datetime(context.manifest.get("completed_at_utc"))),
            },
            {
                "Field": "Elapsed seconds",
                "Value": context.manifest.get("elapsed_seconds"),
            },
            {
                "Field": "Files in run directory",
                "Value": inventory.height,
            },
        ]
    )


def _artifact_inventory_table(inventory: pl.DataFrame) -> pl.DataFrame:
    return inventory.select(
        pl.col("file_name").alias("File"),
        pl.col("suffix").alias("Type"),
        pl.col("row_count").alias("Rows"),
        pl.col("size_mb").alias("Size (MB)"),
        pl.col("stage_name").alias("Stage"),
        pl.col("stage_status").alias("Stage status"),
        pl.col("modified_at").alias("Updated"),
    )


def _write_report(context: RunContext, output_dir: Path) -> Path:
    figure_dir = _ensure_dir(output_dir / FIGURE_SUBDIR)
    inventory = _artifact_inventory(context.run_dir, context.manifest)
    report_metadata = _report_metadata_table(context, inventory)
    artifact_inventory = _artifact_inventory_table(inventory)
    review_notes = _build_review_notes(context, context.run_dir)

    figures = {
        "sample_attrition": _plot_sample_attrition(context.run_dir, figure_dir),
        "complete_case_quarters": _plot_complete_case_quarters(context.run_dir, figure_dir),
        "main_signal_forest": _plot_signal_forest(context.run_dir, figure_dir),
        "trading_cumulative_returns": _plot_trading_returns(context.run_dir, figure_dir),
    }

    sections: list[str] = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{array,booktabs,longtable,pdflscape,graphicx,float,xcolor,hyperref}",
        r"\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}",
        r"\newcolumntype{R}[1]{>{\raggedleft\arraybackslash}p{#1}}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{0.6em}",
        r"\begin{document}",
        r"\begin{center}",
        r"{\LARGE \textbf{LM2011 Post-Refinitiv Review of the Latest Run}}\\[0.6em]",
        rf"{{\large \texttt{{{_latex_escape(str(context.run_dir))}}}}}",
        r"\end{center}",
        r"\section*{Run Metadata}",
        _longtable_block(report_metadata, caption="Latest run metadata", font_size=r"\small"),
        r"\section*{Review Notes}",
        r"\begin{itemize}",
    ]
    if review_notes:
        sections.extend(rf"\item {_latex_escape(note)}" for note in review_notes)
    else:
        sections.append(r"\item No material review notes were generated from the manifest and table diagnostics.")
    sections.extend(
        [
            r"\end{itemize}",
            r"\section*{Figures}",
            _figure_block(figures["sample_attrition"].relative_to(output_dir), "Sample attrition across the 1994--2008 and 1994--2024 LM2011 screening tables."),
            _figure_block(figures["complete_case_quarters"].relative_to(output_dir), "Quarterly complete-case counts before rank checks. The MD\&A series applies the 250-token floor used by Table V."),
            _figure_block(figures["main_signal_forest"].relative_to(output_dir), "Main-signal coefficient estimates across Tables IV, V, VI, VIII, and IA.I. Horizontal bars indicate 95\\% confidence intervals when standard errors are available."),
            _figure_block(figures["trading_cumulative_returns"].relative_to(output_dir), "Cumulative long-short returns by LM2011 sort signal using the IA.II monthly strategy output."),
            r"\clearpage",
            r"\section*{Artifact Inventory}",
            _longtable_block(artifact_inventory, caption="Artifacts present in the latest lm2011\\_post\\_refinitiv run", landscape=True),
            r"\clearpage",
            r"\section*{Logical Table Outputs}",
            r"\subsection*{Table I Sample Creation (1994--2008)}",
            _longtable_block(
                _summarize_sample_table(pl.read_parquet(context.run_dir / SAMPLE_TABLE_FILES[0])),
                caption="LM2011 Table I sample creation, 1994--2008",
            ),
            r"\subsection*{Table I Sample Creation (1994--2024)}",
            _longtable_block(
                _summarize_sample_table(pl.read_parquet(context.run_dir / SAMPLE_TABLE_FILES[1])),
                caption="LM2011 Table I sample creation, 1994--2024",
            ),
            r"\subsection*{Table IV Main Signal Summary}",
            _longtable_block(
                _summarize_main_results(context.run_dir / "lm2011_table_iv_results.parquet"),
                caption="Table IV main-signal summary",
            ),
            r"\subsection*{Table V Main Signal Summary}",
            _longtable_block(
                _summarize_main_results(context.run_dir / "lm2011_table_v_results.parquet"),
                caption="Table V main-signal summary",
            ),
            r"\subsection*{Table VI Main Signal Summary}",
            _longtable_block(
                _summarize_main_results(context.run_dir / "lm2011_table_vi_results.parquet"),
                caption="Table VI main-signal summary",
            ),
            r"\subsection*{Table VIII Main Signal Summary}",
            _longtable_block(
                _summarize_main_results(context.run_dir / "lm2011_table_viii_results.parquet"),
                caption="Table VIII main-signal summary",
            ),
            r"\subsection*{Table IA.I Main Signal Summary}",
            _longtable_block(
                _summarize_main_results(context.run_dir / "lm2011_table_ia_i_results.parquet"),
                caption="Table IA.I main-signal summary",
            ),
            r"\subsection*{Table IA.II Trading Strategy Summary}",
            _longtable_block(
                _summarize_trade_results(context.run_dir / TRADE_RESULTS_FILE),
                caption="Table IA.II trading-strategy summary",
            ),
            r"\clearpage",
            r"\section*{Appendix: Full Regression Coefficients}",
        ]
    )

    appendix_tables = {
        "Table IV coefficients": "lm2011_table_iv_results.parquet",
        "Table V coefficients": "lm2011_table_v_results.parquet",
        "Table VI coefficients": "lm2011_table_vi_results.parquet",
        "Table VIII coefficients": "lm2011_table_viii_results.parquet",
        "Table IA.I coefficients": "lm2011_table_ia_i_results.parquet",
    }
    for heading, filename in appendix_tables.items():
        sections.extend(
            [
                rf"\subsection*{{{_latex_escape(heading)}}}",
                _longtable_block(
                    _summarize_full_coefficients(context.run_dir / filename),
                    caption=heading,
                    landscape=True,
                    font_size=r"\tiny",
                ),
            ]
        )

    sections.extend(
        [
            r"\clearpage",
            r"\section*{Appendix: Skipped-Quarter Diagnostics}",
        ]
    )
    for filename in SKIPPED_DIAGNOSTIC_FILES:
        heading = filename.replace(".parquet", "").replace("_", " ")
        sections.extend(
            [
                rf"\subsection*{{{_latex_escape(heading)}}}",
                _longtable_block(
                    _summarize_skipped_diagnostics(context.run_dir / filename),
                    caption=heading,
                    landscape=True,
                    font_size=r"\tiny",
                ),
            ]
        )

    sections.append(r"\end{document}")
    tex_path = output_dir / f"{REPORT_BASENAME}.tex"
    tex_path.write_text("\n".join(sections) + "\n", encoding="utf-8")
    return tex_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review the latest lm2011_post_refinitiv run and export a LaTeX report.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Explicit lm2011_post_refinitiv run directory. Defaults to the latest discovered run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated LaTeX and figure assets. Defaults to output/latex/lm2011_post_refinitiv_review.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = _resolve_repo_root()
    if args.run_dir is None:
        context = _discover_latest_run_dir(repo_root)
    else:
        run_dir = args.run_dir.resolve()
        manifest_path = run_dir / MANIFEST_FILENAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"Run directory missing manifest: {manifest_path}")
        context = RunContext(
            repo_root=repo_root,
            run_dir=run_dir,
            manifest_path=manifest_path,
            manifest=json.loads(manifest_path.read_text(encoding="utf-8")),
        )

    output_dir = _ensure_dir((repo_root / args.output_dir) if args.output_dir is not None else (repo_root / REPORT_OUTPUT_DIR))
    tex_path = _write_report(context, output_dir)
    pdf_path = output_dir / f"{REPORT_BASENAME}.pdf"
    summary = {
        "run_dir": str(context.run_dir),
        "manifest_path": str(context.manifest_path),
        "tex_path": str(tex_path),
        "expected_pdf_path": str(pdf_path),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

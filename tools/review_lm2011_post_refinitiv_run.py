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
SAMPLE_TABLE_FILES: tuple[str, ...] = (
    "lm2011_table_i_sample_creation.parquet",
    "lm2011_table_i_sample_creation_1994_2024.parquet",
)
TRADE_RESULTS_FILE = "lm2011_table_ia_ii_results.parquet"
TRADING_RETURNS_FILE = "lm2011_trading_strategy_monthly_returns.parquet"
EXTENSION_RESULTS_FILE = "lm2011_extension_results.parquet"
EXTENSION_SAMPLE_LOSS_FILE = "lm2011_extension_sample_loss.parquet"
CORE_TABLE_EXPORT_SPECS: tuple[tuple[str, str, str], ...] = (
    ("Table I Sample Creation (1994--2008)", "lm2011_table_i_sample_creation.parquet", "sample"),
    ("Table I Sample Creation (1994--2024)", "lm2011_table_i_sample_creation_1994_2024.parquet", "sample"),
    ("Table IV (With Ownership Control)", "lm2011_table_iv_results.parquet", "quarterly"),
    ("Table IV (No Ownership Control)", "lm2011_table_iv_results_no_ownership.parquet", "quarterly"),
    ("Table V (With Ownership Control)", "lm2011_table_v_results.parquet", "quarterly"),
    ("Table V (No Ownership Control)", "lm2011_table_v_results_no_ownership.parquet", "quarterly"),
    ("Table VI (With Ownership Control)", "lm2011_table_vi_results.parquet", "quarterly"),
    ("Table VI (No Ownership Control)", "lm2011_table_vi_results_no_ownership.parquet", "quarterly"),
    ("Table VIII (With Ownership Control)", "lm2011_table_viii_results.parquet", "quarterly"),
    ("Table VIII (No Ownership Control)", "lm2011_table_viii_results_no_ownership.parquet", "quarterly"),
    ("Table IA.I (With Ownership Control)", "lm2011_table_ia_i_results.parquet", "quarterly"),
    ("Table IA.I (No Ownership Control)", "lm2011_table_ia_i_results_no_ownership.parquet", "quarterly"),
    ("Table IA.II Trading Strategy Summary", TRADE_RESULTS_FILE, "trade"),
)


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


def _figure_save(fig: plt.Figure, out_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_sample_attrition(run_dir: Path, figure_dir: Path) -> Path:
    first = pl.read_parquet(run_dir / SAMPLE_TABLE_FILES[0]).filter(
        (pl.col("section_id") == "full_10k_document") & (pl.col("sample_size_kind") == "count")
    )
    fig, ax = plt.subplots(figsize=(8.8, 7.6))
    color = "#1f5aa6"
    labels = [textwrap.fill(value, width=34) for value in first["display_label"].to_list()]
    counts = [float(value) for value in first["sample_size_value"].to_list()]
    positions = list(range(len(labels)))
    ax.plot(counts, positions, marker="o", color=color, linewidth=2.3)
    ax.fill_betweenx(positions, counts, color=color, alpha=0.08)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_title("1994-2008 Sample Screen", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value / 1000):,}k"))
    for xpos, ypos in zip(counts, positions, strict=True):
        ax.text(xpos, ypos, f" {int(round(xpos)):,}", va="center", ha="left", fontsize=8)
    ax.set_xlabel("Observations")
    fig.suptitle("LM2011 Sample Attrition", fontsize=14, fontweight="bold")
    out_path = figure_dir / "sample_attrition.png"
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
        "Text scope": "2.6cm",
        "Signal": "3.1cm",
        "Coefficient": "3.4cm",
        "Specification": "3.0cm",
        "Control set": "2.1cm",
        "Outcome": "2.9cm",
        "Quarter": "2.4cm",
        "Reason": "3.0cm",
        "Failure reason": "3.8cm",
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
        "Year",
        "Rank",
        "Columns",
        "Rows",
        "Control-set rows",
        "Estimation rows",
        "Missing outcome",
        "Missing signal",
        "Missing controls",
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


def _summarize_quarterly_results(path: Path) -> pl.DataFrame:
    return (
        pl.read_parquet(path)
        .select(
            pl.col("signal_name").alias("Signal"),
            pl.col("coefficient_name").alias("Coefficient"),
            pl.col("estimate").alias("Estimate"),
            pl.col("standard_error").alias("Std. error"),
            pl.col("t_stat").alias("t-stat"),
            pl.col("n_quarters").alias("Quarters"),
            pl.col("mean_quarter_n").alias("Mean quarter n"),
            pl.col("nw_lags").alias("NW lags"),
        )
        .sort("Signal", "Coefficient")
    )


def _summarize_extension_results(path: Path) -> pl.DataFrame:
    return (
        pl.read_parquet(path)
        .select(
            pl.col("text_scope").alias("Text scope"),
            pl.col("specification_name").alias("Specification"),
            pl.col("control_set_id").alias("Control set"),
            pl.col("coefficient_name").alias("Coefficient"),
            pl.col("estimate").alias("Estimate"),
            pl.col("standard_error").alias("Std. error"),
            pl.col("t_stat").alias("t-stat"),
            pl.col("n_obs").alias("Observations"),
            pl.col("n_quarters").alias("Quarters"),
            pl.col("mean_quarter_n").alias("Mean quarter n"),
            pl.col("estimator_status").alias("Status"),
            pl.col("failure_reason").alias("Failure reason"),
        )
        .sort("Text scope", "Specification", "Control set", "Coefficient")
    )


def _summarize_extension_sample_loss(path: Path) -> pl.DataFrame:
    return (
        pl.read_parquet(path)
        .select(
            pl.col("calendar_year").alias("Year"),
            pl.col("text_scope").alias("Text scope"),
            pl.col("specification_name").alias("Specification"),
            pl.col("control_set_id").alias("Control set"),
            pl.col("outcome_name").alias("Outcome"),
            pl.col("n_control_set_rows").alias("Control-set rows"),
            pl.col("n_estimation_rows").alias("Estimation rows"),
            pl.col("n_missing_outcome").alias("Missing outcome"),
            pl.col("n_missing_signal").alias("Missing signal"),
            pl.col("n_missing_controls").alias("Missing controls"),
        )
        .sort("Year", "Text scope", "Specification", "Control set")
    )


def _build_review_notes(
    context: RunContext,
    run_dir: Path,
    *,
    full_export: bool,
    extension_run_dir: Path | None,
) -> list[str]:
    notes = [
        "This export retains the operative 1994-2008 sample-attrition figure and cumulative long-short return figure for the selected lm2011_post_refinitiv run.",
    ]
    extended_stage = (context.manifest.get("stages", {}) or {}).get("table_i_sample_creation_1994_2024")
    if isinstance(extended_stage, dict) and extended_stage.get("status") == "disabled_by_run_config":
        notes.append(
            "The 1994-2024 sample-creation artifact is present in the run directory but its manifest stage is marked disabled_by_run_config, so it should be treated as a carry-forward artifact rather than a confirmed regenerated output."
        )
    trade_path = run_dir / TRADE_RESULTS_FILE
    if trade_path.exists():
        trade = _summarize_trade_results(trade_path)
        if trade.height > 0:
            mean_returns = [float(value) for value in trade["Mean long-short"].to_list() if value is not None]
            alphas = [float(value) for value in trade["FF4 alpha"].to_list() if value is not None]
            if mean_returns and alphas and all(value < 0 for value in mean_returns) and all(value < 0 for value in alphas):
                notes.append(
                    "All four IA.II long-short strategy means and FF4 alphas are negative in the current run, which is consistent with the cumulative-return chart below."
                )
    if full_export and extension_run_dir is not None:
        notes.append(
            f"Full export mode includes extension tables from {extension_run_dir} using lm2011_extension_results.parquet and lm2011_extension_sample_loss.parquet as the authoritative extension artifacts."
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


def _resolve_extension_run_dir(core_run_dir: Path, explicit_extension_run_dir: Path | None) -> Path:
    candidates = (
        [explicit_extension_run_dir.resolve()]
        if explicit_extension_run_dir is not None
        else [(core_run_dir.parent / "lm2011_extension").resolve()]
    )
    for candidate in candidates:
        if (candidate / EXTENSION_RESULTS_FILE).exists() and (candidate / EXTENSION_SAMPLE_LOSS_FILE).exists():
            return candidate
    raise FileNotFoundError(
        "Full export requires extension artifacts, but lm2011_extension_results.parquet and "
        "lm2011_extension_sample_loss.parquet were not resolved. "
        f"Tried: {[str(candidate) for candidate in candidates]}"
    )


def _core_table_sections(run_dir: Path) -> list[tuple[str, pl.DataFrame, str]]:
    sections: list[tuple[str, pl.DataFrame, str]] = []
    for title, filename, kind in CORE_TABLE_EXPORT_SPECS:
        path = run_dir / filename
        if not path.exists():
            continue
        if kind == "sample":
            df = _summarize_sample_table(pl.read_parquet(path))
        elif kind == "trade":
            df = _summarize_trade_results(path)
        else:
            df = _summarize_quarterly_results(path)
        sections.append((title, df, title))
    return sections


def _extension_table_sections(extension_run_dir: Path) -> list[tuple[str, pl.DataFrame, str]]:
    return [
        (
            "LM2011 Extension Sample Loss",
            _summarize_extension_sample_loss(extension_run_dir / EXTENSION_SAMPLE_LOSS_FILE),
            "LM2011 extension sample-loss summary",
        ),
        (
            "LM2011 Extension Results",
            _summarize_extension_results(extension_run_dir / EXTENSION_RESULTS_FILE),
            "LM2011 extension regression results",
        ),
    ]


def _write_report(
    context: RunContext,
    output_dir: Path,
    *,
    full_export: bool,
    extension_run_dir: Path | None,
) -> Path:
    figure_dir = _ensure_dir(output_dir / FIGURE_SUBDIR)
    for stale_png in figure_dir.glob("*.png"):
        stale_png.unlink()
    inventory = _artifact_inventory(context.run_dir, context.manifest)
    report_metadata = _report_metadata_table(context, inventory)
    artifact_inventory = _artifact_inventory_table(inventory)
    review_notes = _build_review_notes(
        context,
        context.run_dir,
        full_export=full_export,
        extension_run_dir=extension_run_dir,
    )
    core_sections = _core_table_sections(context.run_dir)
    extension_sections = _extension_table_sections(extension_run_dir) if extension_run_dir is not None else []

    figures = {
        "sample_attrition": _plot_sample_attrition(context.run_dir, figure_dir),
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
        r"{\LARGE \textbf{LM2011 Post-Refinitiv Review Export}}\\[0.6em]",
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
            _figure_block(figures["sample_attrition"].relative_to(output_dir), "Sample attrition across the 1994--2008 LM2011 screening table used for the selected run."),
            _figure_block(figures["trading_cumulative_returns"].relative_to(output_dir), "Cumulative long-short returns by LM2011 sort signal using the IA.II monthly strategy output."),
            r"\clearpage",
            r"\section*{Artifact Inventory}",
            _longtable_block(artifact_inventory, caption="Run artifacts discovered in the selected lm2011_post_refinitiv directory", landscape=True),
            r"\clearpage",
            r"\section*{Logical Table Outputs}",
        ]
    )
    for title, table_df, caption in core_sections:
        sections.extend(
            [
                rf"\subsection*{{{_latex_escape(title)}}}",
                _longtable_block(table_df, caption=caption),
            ]
        )
    if extension_sections:
        sections.extend([r"\clearpage", r"\section*{Extension Tables}"])
        for title, table_df, caption in extension_sections:
            sections.extend(
                [
                    rf"\subsection*{{{_latex_escape(title)}}}",
                    _longtable_block(table_df, caption=caption),
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
    parser.add_argument(
        "--extension-run-dir",
        type=Path,
        default=None,
        help="Optional lm2011_extension run directory. Required only for --full-export if sibling auto-resolution fails.",
    )
    parser.add_argument(
        "--full-export",
        action="store_true",
        help="Include extension tables in addition to the core lm2011_post_refinitiv outputs.",
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
    extension_run_dir = (
        _resolve_extension_run_dir(context.run_dir, args.extension_run_dir)
        if args.full_export
        else None
    )
    tex_path = _write_report(
        context,
        output_dir,
        full_export=args.full_export,
        extension_run_dir=extension_run_dir,
    )
    pdf_path = output_dir / f"{REPORT_BASENAME}.pdf"
    summary = {
        "run_dir": str(context.run_dir),
        "manifest_path": str(context.manifest_path),
        "extension_run_dir": str(extension_run_dir) if extension_run_dir is not None else None,
        "tex_path": str(tex_path),
        "expected_pdf_path": str(pdf_path),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

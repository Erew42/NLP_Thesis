from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from collections.abc import Callable, Sequence
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
TABLE_SUBDIR = "tables"
PDF_OUTPUT_DIR = Path("output") / "pdf"
MANIFEST_FILENAME = "lm2011_sample_run_manifest.json"
SAMPLE_TABLE_FILE = "lm2011_table_i_sample_creation.parquet"
SAMPLE_TABLE_FILE_EXTENDED = "lm2011_table_i_sample_creation_1994_2024.parquet"
FULL_RETURN_PANEL_FILE = "lm2011_return_regression_panel_full_10k.parquet"
MDA_RETURN_PANEL_FILE = "lm2011_return_regression_panel_mda.parquet"
SUE_PANEL_FILE = "lm2011_sue_regression_panel.parquet"
TRADE_RESULTS_FILE = "lm2011_table_ia_ii_results.parquet"
TRADING_RETURNS_FILE = "lm2011_trading_strategy_monthly_returns.parquet"
EXTENSION_RESULTS_FILE = "lm2011_extension_results.parquet"
EXTENSION_SAMPLE_LOSS_FILE = "lm2011_extension_sample_loss.parquet"
MDA_MIN_TOKEN_COUNT = 250

CORE_QUARTERLY_RESULTS: tuple[tuple[str, str], ...] = (
    ("Table IV (With Ownership Control)", "lm2011_table_iv_results.parquet"),
    ("Table IV (No Ownership Control)", "lm2011_table_iv_results_no_ownership.parquet"),
    ("Table V (With Ownership Control)", "lm2011_table_v_results.parquet"),
    ("Table V (No Ownership Control)", "lm2011_table_v_results_no_ownership.parquet"),
    ("Table VI (With Ownership Control)", "lm2011_table_vi_results.parquet"),
    ("Table VI (No Ownership Control)", "lm2011_table_vi_results_no_ownership.parquet"),
    ("Table VIII (With Ownership Control)", "lm2011_table_viii_results.parquet"),
    ("Table VIII (No Ownership Control)", "lm2011_table_viii_results_no_ownership.parquet"),
    ("Table IA.I (With Ownership Control)", "lm2011_table_ia_i_results.parquet"),
    ("Table IA.I (No Ownership Control)", "lm2011_table_ia_i_results_no_ownership.parquet"),
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


def _format_timestamp(timestamp: dt.datetime | None) -> str:
    if timestamp is None:
        return "--"
    if timestamp.tzinfo is None:
        return timestamp.isoformat(timespec="seconds")
    return timestamp.astimezone(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


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


def _is_missing_number(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _format_float(value: Any, digits: int = 3) -> str:
    if _is_missing_number(value):
        return "--"
    return f"{float(value):.{digits}f}"


def _format_integer(value: Any) -> str:
    if _is_missing_number(value):
        return "--"
    return f"{int(round(float(value))):,}"


def _format_percent(value: Any, digits: int = 2) -> str:
    if _is_missing_number(value):
        return "--"
    return f"{float(value):.{digits}f}\\%"


def _format_currency_billions(value: Any, digits: int = 2) -> str:
    if _is_missing_number(value):
        return "--"
    return f"\\${float(value):,.{digits}f}"


def _figure_save(fig: plt.Figure, out_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _series_summary(df: pl.DataFrame, column_name: str, *, scale: float = 1.0) -> dict[str, Any]:
    if column_name not in df.columns:
        return {"mean": None, "median": None, "std": None, "n": 0}
    expr = pl.col(column_name).cast(pl.Float64, strict=False) * scale
    summary = df.select(
        expr.mean().alias("mean"),
        expr.median().alias("median"),
        expr.std().alias("std"),
        pl.col(column_name).is_not_null().sum().alias("n"),
    ).to_dicts()[0]
    return summary


def _summary_stat_formatter(style: str, value: Any) -> str:
    if style == "pct":
        return _format_percent(value, digits=2)
    if style == "ratio":
        return _format_float(value, digits=3)
    if style == "size_billions":
        return _format_currency_billions(value, digits=2)
    raise ValueError(f"Unsupported summary-stat style: {style}")


def _build_summary_statistics_rows(run_dir: Path) -> tuple[list[dict[str, Any]], list[str]] | None:
    full_path = run_dir / FULL_RETURN_PANEL_FILE
    mda_path = run_dir / MDA_RETURN_PANEL_FILE
    sue_path = run_dir / SUE_PANEL_FILE
    if not (full_path.exists() and mda_path.exists() and sue_path.exists()):
        return None

    full_panel = pl.read_parquet(full_path)
    mda_panel = pl.read_parquet(mda_path).filter(
        pl.col("total_token_count_mda").cast(pl.Float64, strict=False) >= float(MDA_MIN_TOKEN_COUNT)
    )
    sue_panel = pl.read_parquet(sue_path)
    sue_mda_panel = sue_panel.join(mda_panel.select("doc_id").unique(), on="doc_id", how="inner")

    row_specs: list[dict[str, Any]] = [
        {"kind": "section", "label": "Word Lists"},
        {
            "kind": "value",
            "label": "H4N-Inf (H4N w/ inflections)",
            "style": "pct",
            "full": _series_summary(full_panel, "h4n_inf_prop", scale=100.0),
            "mda": _series_summary(mda_panel, "h4n_inf_prop", scale=100.0),
        },
        {
            "kind": "value",
            "label": "Fin-Neg (negative)",
            "style": "pct",
            "full": _series_summary(full_panel, "lm_negative_prop", scale=100.0),
            "mda": _series_summary(mda_panel, "lm_negative_prop", scale=100.0),
        },
        {
            "kind": "value",
            "label": "Fin-Pos (positive)",
            "style": "pct",
            "full": _series_summary(full_panel, "lm_positive_prop", scale=100.0),
            "mda": _series_summary(mda_panel, "lm_positive_prop", scale=100.0),
        },
        {
            "kind": "value",
            "label": "Fin-Unc (uncertainty)",
            "style": "pct",
            "full": _series_summary(full_panel, "lm_uncertainty_prop", scale=100.0),
            "mda": _series_summary(mda_panel, "lm_uncertainty_prop", scale=100.0),
        },
        {
            "kind": "value",
            "label": "Fin-Lit (litigious)",
            "style": "pct",
            "full": _series_summary(full_panel, "lm_litigious_prop", scale=100.0),
            "mda": _series_summary(mda_panel, "lm_litigious_prop", scale=100.0),
        },
        {
            "kind": "value",
            "label": "MW-Strong (strong modal words)",
            "style": "pct",
            "full": _series_summary(full_panel, "lm_modal_strong_prop", scale=100.0),
            "mda": _series_summary(mda_panel, "lm_modal_strong_prop", scale=100.0),
        },
        {
            "kind": "value",
            "label": "MW-Weak (weak modal words)",
            "style": "pct",
            "full": _series_summary(full_panel, "lm_modal_weak_prop", scale=100.0),
            "mda": _series_summary(mda_panel, "lm_modal_weak_prop", scale=100.0),
        },
        {"kind": "section", "label": "Other Variables"},
        {
            "kind": "value",
            "label": "Event period [0,3] excess return",
            "style": "pct",
            "full": _series_summary(full_panel, "filing_period_excess_return", scale=100.0),
            "mda": _series_summary(mda_panel, "filing_period_excess_return", scale=100.0),
        },
        {
            "kind": "value",
            "label": "Size ($billions)",
            "style": "size_billions",
            "full": _series_summary(full_panel, "size_event", scale=1.0 / 1000.0),
            "mda": _series_summary(mda_panel, "size_event", scale=1.0 / 1000.0),
        },
        {
            "kind": "value",
            "label": "Book-to-market",
            "style": "ratio",
            "full": _series_summary(full_panel, "bm_event"),
            "mda": _series_summary(mda_panel, "bm_event"),
        },
        {
            "kind": "value",
            "label": "Turnover",
            "style": "ratio",
            "full": _series_summary(full_panel, "share_turnover"),
            "mda": _series_summary(mda_panel, "share_turnover"),
        },
        {
            "kind": "value",
            "label": "One-year preevent FF alpha",
            "style": "pct",
            "full": _series_summary(full_panel, "pre_ffalpha", scale=100.0),
            "mda": _series_summary(mda_panel, "pre_ffalpha", scale=100.0),
        },
        {
            "kind": "value",
            "label": "Institutional ownership",
            "style": "pct",
            "full": _series_summary(full_panel, "institutional_ownership", scale=1.0),
            "mda": _series_summary(mda_panel, "institutional_ownership", scale=1.0),
        },
        {
            "kind": "value",
            "label": "NASDAQ dummy",
            "style": "pct",
            "full": _series_summary(full_panel, "nasdaq_dummy", scale=100.0),
            "mda": _series_summary(mda_panel, "nasdaq_dummy", scale=100.0),
        },
        {
            "kind": "value",
            "label": "Standardized unexpected earnings",
            "style": "pct",
            "full": _series_summary(sue_panel, "sue", scale=100.0),
            "mda": _series_summary(sue_mda_panel, "sue", scale=100.0),
        },
        {
            "kind": "value",
            "label": "Analysts' earnings forecast dispersion",
            "style": "pct",
            "full": _series_summary(sue_panel, "analyst_dispersion", scale=100.0),
            "mda": _series_summary(sue_mda_panel, "analyst_dispersion", scale=100.0),
        },
        {
            "kind": "value",
            "label": "Analysts' earnings revisions",
            "style": "pct",
            "full": _series_summary(sue_panel, "analyst_revisions", scale=100.0),
            "mda": _series_summary(sue_mda_panel, "analyst_revisions", scale=100.0),
        },
    ]

    ownership_full_n = _series_summary(full_panel, "institutional_ownership")["n"]
    ownership_mda_n = _series_summary(mda_panel, "institutional_ownership")["n"]
    notes = [
        (
            "The first seven variables report word-list proportions relative to total words. "
            f"Full 10-K sample size: {full_panel.height:,}; MD&A sample size: {mda_panel.height:,}."
        ),
        (
            "The three earnings-related variables use the available standardized-unexpected-earnings panel: "
            f"{sue_panel.height:,} full-10-K observations and {sue_mda_panel.height:,} observations "
            "matched to the MD&A sample."
        ),
    ]
    if ownership_full_n or ownership_mda_n:
        notes.append(
            "Institutional ownership is sparse in the staged run: "
            f"{ownership_full_n:,} full-10-K observations and {ownership_mda_n:,} MD&A observations."
        )
    return row_specs, notes


def _summary_statistics_csv_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        if row["kind"] != "value":
            continue
        out_rows.append(
            {
                "Variable": row["label"],
                "Full 10-K Mean": row["full"]["mean"],
                "Full 10-K Median": row["full"]["median"],
                "Full 10-K Std. Dev.": row["full"]["std"],
                "Full 10-K N": row["full"]["n"],
                "MD&A Mean": row["mda"]["mean"],
                "MD&A Median": row["mda"]["median"],
                "MD&A Std. Dev.": row["mda"]["std"],
                "MD&A N": row["mda"]["n"],
                "Format": row["style"],
            }
        )
    return pl.DataFrame(out_rows)


def _render_summary_statistics_table(
    run_dir: Path,
    table_output_dir: Path,
) -> str | None:
    built = _build_summary_statistics_rows(run_dir)
    if built is None:
        return None
    rows, notes = built
    csv_frame = _summary_statistics_csv_frame(rows)
    csv_frame.write_csv(table_output_dir / "lm2011_table_ii_summary_statistics.csv")

    full_panel = pl.read_parquet(run_dir / FULL_RETURN_PANEL_FILE)
    full_n = _format_integer(full_panel.height)
    mda_panel = pl.read_parquet(run_dir / MDA_RETURN_PANEL_FILE).filter(
        pl.col("total_token_count_mda").cast(pl.Float64, strict=False) >= float(MDA_MIN_TOKEN_COUNT)
    )
    mda_n = _format_integer(mda_panel.height)

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"{\bfseries Table II}\\[0.3em]",
        r"{\bfseries Summary Statistics for the 1994 to 2008 10-K Sample}\\[0.7em]",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        rf"& \multicolumn{{3}}{{c}}{{Full 10-K Document ($N = {full_n}$)}} & \multicolumn{{3}}{{c}}{{MD\&A Section ($N = {mda_n}$)}} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Variable & Mean & Median & Std. Dev. & Mean & Median & Std. Dev. \\",
        r"\midrule",
    ]
    for row in rows:
        if row["kind"] == "section":
            lines.append(rf"\multicolumn{{7}}{{l}}{{\textit{{{_latex_escape(row['label'])}}}}} \\")
            continue
        style = row["style"]
        lines.append(
            " & ".join(
                [
                    _latex_escape(row["label"]),
                    _summary_stat_formatter(style, row["full"]["mean"]),
                    _summary_stat_formatter(style, row["full"]["median"]),
                    _summary_stat_formatter(style, row["full"]["std"]),
                    _summary_stat_formatter(style, row["mda"]["mean"]),
                    _summary_stat_formatter(style, row["mda"]["median"]),
                    _summary_stat_formatter(style, row["mda"]["std"]),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            rf"{{\footnotesize\emph{{Note:}} {_latex_escape(' '.join(notes))}}}",
        ]
    )
    return "\n".join(lines)


def _quintile_median_returns_frame(run_dir: Path) -> pl.DataFrame | None:
    panel_path = run_dir / FULL_RETURN_PANEL_FILE
    if not panel_path.exists():
        return None
    panel = pl.read_parquet(panel_path)
    frames: list[pl.DataFrame] = []
    for signal_name, signal_label in (
        ("h4n_inf_prop", "H4N-Inf"),
        ("lm_negative_prop", "Fin-Neg"),
    ):
        if signal_name not in panel.columns:
            continue
        signal_frame = (
            panel.select(
                pl.col(signal_name).cast(pl.Float64, strict=False).alias("signal_value"),
                pl.col("filing_period_excess_return").cast(pl.Float64, strict=False).alias("event_return"),
            )
            .drop_nulls()
            .sort("signal_value")
            .with_row_index("rank_idx")
        )
        if signal_frame.height == 0:
            continue
        ranked = signal_frame.with_columns(
            (
                ((pl.col("rank_idx") * 5) / signal_frame.height)
                .floor()
                .cast(pl.Int32)
                .clip(0, 4)
                + 1
            ).alias("quintile")
        )
        frames.append(
            ranked.group_by("quintile")
            .agg((pl.col("event_return").median() * 100.0).alias("median_excess_return_pct"))
            .sort("quintile")
            .with_columns(pl.lit(signal_label).alias("signal_label"))
            .select("signal_label", "quintile", "median_excess_return_pct")
        )
    if not frames:
        return None
    return pl.concat(frames).sort("signal_label", "quintile")


def _plot_median_excess_return_quintiles(
    run_dir: Path,
    figure_dir: Path,
    table_output_dir: Path,
) -> Path | None:
    quintile_frame = _quintile_median_returns_frame(run_dir)
    if quintile_frame is None or quintile_frame.height == 0:
        return None

    quintile_frame.write_csv(table_output_dir / "lm2011_figure1_quintiles.csv")

    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    styles = {
        "H4N-Inf": {"color": "#bdbdbd", "linestyle": (0, (4, 4)), "marker": "D", "linewidth": 2.2},
        "Fin-Neg": {"color": "#262626", "linestyle": "-", "marker": "o", "linewidth": 2.3},
    }
    for signal_label in ("H4N-Inf", "Fin-Neg"):
        signal_df = quintile_frame.filter(pl.col("signal_label") == signal_label).sort("quintile")
        if signal_df.height == 0:
            continue
        style = styles[signal_label]
        ax.plot(
            signal_df["quintile"].to_list(),
            signal_df["median_excess_return_pct"].to_list(),
            label=signal_label,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=style["linewidth"],
            markersize=6,
        )
    ax.set_xlim(1, 5)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(["Low", "2", "3", "4", "High"])
    ax.set_ylabel("Median Filing Period Excess Return")
    ax.set_xlabel("Quintile (based on proportion of negative words)")
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}%"))
    ax.legend(frameon=True, facecolor="white", edgecolor="black", loc="best")
    out_path = figure_dir / "median_excess_return_quintiles.png"
    _figure_save(fig, out_path)
    return out_path


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


def _plot_trading_returns(run_dir: Path, figure_dir: Path) -> Path | None:
    path = run_dir / TRADING_RETURNS_FILE
    if not path.exists():
        return None
    df = _trading_returns_frame(run_dir)
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
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


def _combined_yearly_sample_sizes_frame(core_run_dir: Path, extension_run_dir: Path) -> pl.DataFrame | None:
    core_full_panel_path = core_run_dir / FULL_RETURN_PANEL_FILE
    core_mda_panel_path = core_run_dir / MDA_RETURN_PANEL_FILE
    extension_event_panel_path = extension_run_dir / "lm2011_extension_event_panel.parquet"
    extension_sample_loss_path = extension_run_dir / EXTENSION_SAMPLE_LOSS_FILE
    if not (
        core_full_panel_path.exists()
        and core_mda_panel_path.exists()
        and extension_event_panel_path.exists()
        and extension_sample_loss_path.exists()
    ):
        return None

    core_full_panel = (
        pl.scan_parquet(core_full_panel_path)
        .select(
            pl.col("filing_date").cast(pl.Date, strict=False),
            pl.col("institutional_ownership"),
        )
        .with_columns(pl.col("filing_date").dt.year().cast(pl.Int32).alias("year"))
    )
    core_mda_panel = (
        pl.scan_parquet(core_mda_panel_path)
        .select(
            pl.col("filing_date").cast(pl.Date, strict=False),
            pl.col("institutional_ownership"),
            pl.col("total_token_count_mda").cast(pl.Float64, strict=False),
        )
        .filter(pl.col("total_token_count_mda") >= float(MDA_MIN_TOKEN_COUNT))
        .with_columns(pl.col("filing_date").dt.year().cast(pl.Int32).alias("year"))
    )
    core_counts = pl.concat(
        [
            core_full_panel.group_by("year")
            .agg(pl.len().cast(pl.Int64).alias("sample_size"))
            .with_columns(
                pl.lit("Full 10-K", dtype=pl.Utf8).alias("series_label"),
                pl.lit("C0", dtype=pl.Utf8).alias("control_group"),
                pl.lit("1994-2008", dtype=pl.Utf8).alias("sample_window"),
                pl.lit("core_full_return_panel", dtype=pl.Utf8).alias("data_source"),
            ),
            core_full_panel.filter(pl.col("institutional_ownership").is_not_null())
            .group_by("year")
            .agg(pl.len().cast(pl.Int64).alias("sample_size"))
            .with_columns(
                pl.lit("Full 10-K", dtype=pl.Utf8).alias("series_label"),
                pl.lit("C1/C2", dtype=pl.Utf8).alias("control_group"),
                pl.lit("1994-2008", dtype=pl.Utf8).alias("sample_window"),
                pl.lit("core_full_return_panel_nonmissing_ownership", dtype=pl.Utf8).alias("data_source"),
            ),
            core_mda_panel.group_by("year")
            .agg(pl.len().cast(pl.Int64).alias("sample_size"))
            .with_columns(
                pl.lit("Item 7 MD&A", dtype=pl.Utf8).alias("series_label"),
                pl.lit("C0", dtype=pl.Utf8).alias("control_group"),
                pl.lit("1994-2008", dtype=pl.Utf8).alias("sample_window"),
                pl.lit("core_mda_return_panel_token_ge_250", dtype=pl.Utf8).alias("data_source"),
            ),
            core_mda_panel.filter(pl.col("institutional_ownership").is_not_null())
            .group_by("year")
            .agg(pl.len().cast(pl.Int64).alias("sample_size"))
            .with_columns(
                pl.lit("Item 7 MD&A", dtype=pl.Utf8).alias("series_label"),
                pl.lit("C1/C2", dtype=pl.Utf8).alias("control_group"),
                pl.lit("1994-2008", dtype=pl.Utf8).alias("sample_window"),
                pl.lit("core_mda_return_panel_token_ge_250_nonmissing_ownership", dtype=pl.Utf8).alias("data_source"),
            ),
        ],
        how="vertical_relaxed",
    ).collect()

    extension_event_panel = (
        pl.scan_parquet(extension_event_panel_path)
        .select(
            pl.col("filing_date").cast(pl.Date, strict=False),
            pl.col("institutional_ownership"),
        )
        .with_columns(pl.col("filing_date").dt.year().cast(pl.Int32).alias("year"))
    )
    extension_full_counts = pl.concat(
        [
            extension_event_panel.group_by("year")
            .agg(pl.len().cast(pl.Int64).alias("sample_size"))
            .with_columns(
                pl.lit("Full 10-K", dtype=pl.Utf8).alias("series_label"),
                pl.lit("C0", dtype=pl.Utf8).alias("control_group"),
                pl.lit("2009-2024", dtype=pl.Utf8).alias("sample_window"),
                pl.lit("extension_event_panel", dtype=pl.Utf8).alias("data_source"),
            ),
            extension_event_panel.filter(pl.col("institutional_ownership").is_not_null())
            .group_by("year")
            .agg(pl.len().cast(pl.Int64).alias("sample_size"))
            .with_columns(
                pl.lit("Full 10-K", dtype=pl.Utf8).alias("series_label"),
                pl.lit("C1/C2", dtype=pl.Utf8).alias("control_group"),
                pl.lit("2009-2024", dtype=pl.Utf8).alias("sample_window"),
                pl.lit("extension_event_panel_nonmissing_ownership", dtype=pl.Utf8).alias("data_source"),
            ),
        ],
        how="vertical_relaxed",
    ).collect()

    extension_counts_raw = (
        pl.scan_parquet(extension_sample_loss_path)
        .filter(
            pl.col("text_scope").is_in(("item_1a_risk_factors", "item_7_mda"))
            & pl.col("control_set_id").is_in(("C0", "C1", "C2"))
            & (pl.col("specification_name") == pl.lit("dictionary_only"))
        )
        .select(
            pl.col("calendar_year").cast(pl.Int32).alias("year"),
            pl.when(pl.col("text_scope") == "item_1a_risk_factors")
            .then(pl.lit("Item 1A risk factors"))
            .when(pl.col("text_scope") == "item_7_mda")
            .then(pl.lit("Item 7 MD&A"))
            .otherwise(pl.col("text_scope"))
            .alias("series_label"),
            pl.when(pl.col("control_set_id") == "C0")
            .then(pl.lit("C0"))
            .otherwise(pl.lit("C1/C2"))
            .alias("control_group"),
            pl.col("sample_window").cast(pl.Utf8),
            pl.col("n_control_set_rows").cast(pl.Int64),
            pl.lit("extension_sample_loss", dtype=pl.Utf8).alias("data_source"),
        )
    )
    extension_counts = (
        extension_counts_raw.group_by("year", "series_label", "control_group", "sample_window", "data_source")
        .agg(
            pl.col("n_control_set_rows").n_unique().alias("_n_unique_row_counts"),
            pl.col("n_control_set_rows").first().cast(pl.Int64).alias("sample_size"),
        )
        .sort("series_label", "control_group", "year")
    ).collect()
    if extension_counts.filter(pl.col("_n_unique_row_counts") > 1).height > 0:
        raise ValueError(
            "Extension sample-loss rows disagree within a (year, series, control-group) block; "
            "cannot collapse C1/C2 into a single yearly sample-size series."
        )
    extension_counts = extension_counts.drop("_n_unique_row_counts")

    output_columns = ["year", "sample_size", "series_label", "control_group", "sample_window", "data_source"]
    combined = pl.concat(
        [
            core_counts.select(output_columns),
            extension_full_counts.select(output_columns),
            extension_counts.select(output_columns),
        ],
        how="vertical_relaxed",
    )
    expected_rows: list[dict[str, Any]] = []
    for control_group in ("C0", "C1/C2"):
        full_source = (
            "core_full_return_panel"
            if control_group == "C0"
            else "core_full_return_panel_nonmissing_ownership"
        )
        item7_source = (
            "core_mda_return_panel_token_ge_250"
            if control_group == "C0"
            else "core_mda_return_panel_token_ge_250_nonmissing_ownership"
        )
        extension_full_source = (
            "extension_event_panel"
            if control_group == "C0"
            else "extension_event_panel_nonmissing_ownership"
        )
        for year in range(1994, 2009):
            expected_rows.extend(
                [
                    {
                        "year": year,
                        "series_label": "Full 10-K",
                        "control_group": control_group,
                        "sample_window": "1994-2008",
                        "data_source": full_source,
                    },
                    {
                        "year": year,
                        "series_label": "Item 7 MD&A",
                        "control_group": control_group,
                        "sample_window": "1994-2008",
                        "data_source": item7_source,
                    },
                ]
            )
        for year in range(2009, 2025):
            expected_rows.extend(
                [
                    {
                        "year": year,
                        "series_label": "Full 10-K",
                        "control_group": control_group,
                        "sample_window": "2009-2024",
                        "data_source": extension_full_source,
                    },
                    {
                        "year": year,
                        "series_label": "Item 7 MD&A",
                        "control_group": control_group,
                        "sample_window": "2009-2024",
                        "data_source": "extension_sample_loss",
                    },
                    {
                        "year": year,
                        "series_label": "Item 1A risk factors",
                        "control_group": control_group,
                        "sample_window": "2009-2024",
                        "data_source": "extension_sample_loss",
                    },
                ]
            )
    expected_frame = pl.DataFrame(expected_rows)
    actual_counts = combined.select("year", "series_label", "control_group", "sample_size")
    return (
        expected_frame.join(
            actual_counts,
            on=["year", "series_label", "control_group"],
            how="left",
        )
        .with_columns(pl.col("sample_size").fill_null(0).cast(pl.Int64))
        .sort("series_label", "control_group", "year")
    )


def _plot_yearly_sample_sizes_by_control_group(
    plot_frame: pl.DataFrame,
    *,
    control_group: str,
    figure_dir: Path,
) -> Path | None:
    group_frame = plot_frame.filter(pl.col("control_group") == control_group).sort("series_label", "year")
    if group_frame.height == 0:
        return None

    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    series_order = ["Full 10-K", "Item 7 MD&A", "Item 1A risk factors"]
    styles = {
        "Full 10-K": {"color": "#1f5aa6", "linestyle": "-", "marker": "o"},
        "Item 7 MD&A": {"color": "#0b6e4f", "linestyle": "-", "marker": "s"},
        "Item 1A risk factors": {"color": "#8c3b2f", "linestyle": "--", "marker": "^"},
    }
    for series_label in series_order:
        series_df = group_frame.filter(pl.col("series_label") == series_label).sort("year")
        if series_df.height == 0:
            continue
        style = styles[series_label]
        ax.plot(
            series_df["year"].to_list(),
            series_df["sample_size"].to_list(),
            label=series_label,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=2.2,
            markersize=5.0,
        )
    tick_years = list(range(1994, 2025, 4))
    if 2024 not in tick_years:
        tick_years.append(2024)
    ax.set_xlim(1994, 2024)
    ax.set_xticks(sorted(set(tick_years)))
    ax.set_ylabel("Sample size")
    ax.set_xlabel("Filing year")
    ax.set_title(f"Yearly Sample Sizes: {control_group}", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.23),
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor="#b0b0b0",
    )
    fig.subplots_adjust(top=0.76, bottom=0.16)
    filename = "yearly_sample_sizes_c0.png" if control_group == "C0" else "yearly_sample_sizes_c1_c2.png"
    out_path = figure_dir / filename
    _figure_save(fig, out_path)
    return out_path


def _render_manual_figure(path: Path, *, number: str, caption: str, width: str = "0.82\\textwidth") -> str:
    return "\n".join(
        [
            r"\begin{figure}[H]",
            r"\centering",
            rf"\includegraphics[width={width}]{{{_latex_escape(path.as_posix())}}}",
            rf"\par\smallskip\parbox{{0.92\textwidth}}{{\small \textbf{{Figure {number}.}} {_latex_escape(caption)}}}",
            r"\end{figure}",
        ]
    )


def _render_sample_creation_table(run_dir: Path) -> str | None:
    sample_path = run_dir / SAMPLE_TABLE_FILE
    if not sample_path.exists():
        return None
    sample = pl.read_parquet(sample_path)
    sort_columns = [column for column in ("section_order", "row_order") if column in sample.columns]
    if sort_columns:
        sample = sample.sort(sort_columns)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"{\bfseries Table I}\\[0.3em]",
        r"{\bfseries Sample Construction for the 1994 to 2008 10-K Sample}\\[0.7em]",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Variable & Sample size & Removed \\",
        r"\midrule",
    ]
    current_section: str | None = None
    for row in sample.iter_rows(named=True):
        section_label = str(row.get("section_label") or "")
        if section_label and section_label != current_section:
            lines.append(rf"\multicolumn{{3}}{{l}}{{\textit{{{_latex_escape(section_label)}}}}} \\")
            current_section = section_label
        kind = str(row.get("sample_size_kind") or "count")
        sample_value = row.get("sample_size_value")
        if kind == "mean":
            sample_text = _format_float(sample_value, digits=2)
        else:
            sample_text = _format_integer(sample_value)
        removed_text = _format_integer(row.get("observations_removed"))
        lines.append(
            " & ".join(
                [
                    _latex_escape(row.get("display_label")),
                    sample_text,
                    "" if removed_text == "--" else removed_text,
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def _result_lookup(df: pl.DataFrame) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, Any]]]:
    lookup: dict[str, dict[str, dict[str, Any]]] = {}
    fit_info: dict[str, dict[str, Any]] = {}
    for row in df.iter_rows(named=True):
        signal_name = str(row["signal_name"])
        coefficient_name = str(row["coefficient_name"])
        lookup.setdefault(signal_name, {})[coefficient_name] = {
            "estimate": row["estimate"],
            "t_stat": row["t_stat"],
            "standard_error": row["standard_error"],
            "signal_name": signal_name,
            "coefficient_name": coefficient_name,
        }
        fit_info.setdefault(
            signal_name,
            {
                "n_quarters": row["n_quarters"],
                "mean_quarter_n": row["mean_quarter_n"],
            },
        )
    return lookup, fit_info


def _scale_table_viii_estimate(row: dict[str, Any]) -> float:
    scale = 100.0
    if row.get("coefficient_name") == row.get("signal_name") and row.get("signal_name") in {
        "h4n_inf_tfidf",
        "lm_negative_tfidf",
    }:
        scale *= 100.0
    return scale


def _coef_cell(
    row: dict[str, Any] | None,
    *,
    digits: int = 3,
    estimate_scale: float | Callable[[dict[str, Any]], float] = 1.0,
) -> str:
    if row is None:
        return ""
    scale = estimate_scale(row) if callable(estimate_scale) else estimate_scale
    estimate_value = row.get("estimate")
    if not _is_missing_number(estimate_value):
        estimate_value = float(estimate_value) * float(scale)
    estimate = _format_float(estimate_value, digits=digits)
    t_stat = _format_float(row.get("t_stat"), digits=2)
    return rf"\shortstack[c]{{{estimate} \\ ({t_stat})}}"


def _fit_summary_note(fit_info: dict[str, dict[str, Any]], signal_order: Sequence[str]) -> str:
    ordered = [fit_info.get(signal_name, {}) for signal_name in signal_order]
    quarters = [row.get("n_quarters") for row in ordered if row.get("n_quarters") is not None]
    mean_ns = [row.get("mean_quarter_n") for row in ordered if row.get("mean_quarter_n") is not None]
    parts = ["Cells report coefficient estimates with Newey-West t-statistics in parentheses."]
    if quarters:
        quarter_set = {int(value) for value in quarters}
        if len(quarter_set) == 1:
            parts.append(f"Each column uses {next(iter(quarter_set))} estimable quarters.")
        else:
            by_column = ", ".join(
                f"({index}) {int(value)}"
                for index, value in enumerate(quarters, start=1)
            )
            parts.append(f"Estimable quarters by column: {by_column}.")
    if mean_ns:
        rounded = {round(float(value), 1) for value in mean_ns}
        if len(rounded) == 1:
            parts.append(f"Mean quarter n is {next(iter(rounded)):.1f}.")
        else:
            by_column = ", ".join(
                f"({index}) {float(value):.1f}"
                for index, value in enumerate(mean_ns, start=1)
            )
            parts.append(f"Mean quarter n by column: {by_column}.")
    if any(
        any(_is_missing_number(coefficients.get("t_stat")) for coefficients in signal_coefficients.values())
        for signal_coefficients in (
            {key: value for key, value in signal_lookup.items()}
            for signal_lookup in []
        )
    ):
        parts.append("Some t-statistics are unavailable in the staged artifact.")
    parts.append("Average R^2 is not stored in the staged output.")
    return " ".join(parts)


def _render_four_column_wordlist_table(
    title_number: str,
    heading: str,
    subtitle: str,
    df: pl.DataFrame,
    *,
    h4n_label: str,
    finneg_label: str,
    extra_controls: Sequence[tuple[str, str]] = (),
    estimate_scale: float | Callable[[dict[str, Any]], float] = 1.0,
    note_suffix: str = "",
) -> str:
    lookup, fit_info = _result_lookup(df)
    signal_order = ["h4n_inf_prop", "lm_negative_prop", "h4n_inf_tfidf", "lm_negative_tfidf"]
    row_lookup = {
        "h4n_inf_prop": lookup.get("h4n_inf_prop", {}).get("h4n_inf_prop"),
        "lm_negative_prop": lookup.get("lm_negative_prop", {}).get("lm_negative_prop"),
        "h4n_inf_tfidf": lookup.get("h4n_inf_tfidf", {}).get("h4n_inf_tfidf"),
        "lm_negative_tfidf": lookup.get("lm_negative_tfidf", {}).get("lm_negative_tfidf"),
    }
    control_rows: list[tuple[str, str]] = [
        ("Log(size)", "log_size"),
        ("Log(book-to-market)", "log_book_to_market"),
        ("Log(share turnover)", "log_share_turnover"),
        ("Pre-FFAlpha", "pre_ffalpha"),
        ("Institutional ownership", "institutional_ownership"),
        ("NASDAQ dummy", "nasdaq_dummy"),
        *extra_controls,
    ]

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        rf"{{\bfseries {title_number}}}\\[0.3em]",
        rf"{{\bfseries {_latex_escape(heading)}}}\\[0.25em]",
        rf"{{\itshape {_latex_escape(subtitle)}}}\\[0.7em]",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Proportional Weights} & \multicolumn{2}{c}{tf.idf Weights} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"& (1) & (2) & (3) & (4) \\",
        r"\midrule",
        r"\multicolumn{5}{l}{\textit{Word Lists}} \\",
        " & ".join(
            [
                _latex_escape(h4n_label),
                _coef_cell(row_lookup["h4n_inf_prop"], estimate_scale=estimate_scale),
                "",
                _coef_cell(row_lookup["h4n_inf_tfidf"], estimate_scale=estimate_scale),
                "",
            ]
        )
        + r" \\",
        " & ".join(
            [
                _latex_escape(finneg_label),
                "",
                _coef_cell(row_lookup["lm_negative_prop"], estimate_scale=estimate_scale),
                "",
                _coef_cell(row_lookup["lm_negative_tfidf"], estimate_scale=estimate_scale),
            ]
        )
        + r" \\",
        r"\addlinespace",
        r"\multicolumn{5}{l}{\textit{Control Variables}} \\",
    ]
    for display_label, coefficient_name in control_rows:
        cells = [
            _coef_cell(lookup.get(signal_name, {}).get(coefficient_name), estimate_scale=estimate_scale)
            for signal_name in signal_order
        ]
        if not any(cells):
            continue
        lines.append(" & ".join([_latex_escape(display_label), *cells]) + r" \\")
    note = _fit_summary_note(fit_info, signal_order)
    if note_suffix:
        note += f" {note_suffix}"
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            rf"{{\footnotesize\emph{{Note:}} {_latex_escape(note)}}}",
        ]
    )
    return "\n".join(lines)


def _render_dictionary_surface_table(
    title_number: str,
    heading: str,
    subtitle: str,
    df: pl.DataFrame,
    *,
    estimate_scale: float | Callable[[dict[str, Any]], float] = 1.0,
    note_suffix: str = "",
) -> str:
    lookup, fit_info = _result_lookup(df)
    panel_a_order = [
        ("Negative", "lm_negative_prop"),
        ("Positive", "lm_positive_prop"),
        ("Uncertainty", "lm_uncertainty_prop"),
        ("Litigious", "lm_litigious_prop"),
        ("Modal Strong", "lm_modal_strong_prop"),
        ("Modal Weak", "lm_modal_weak_prop"),
    ]
    panel_b_order = [
        ("Negative", "lm_negative_tfidf"),
        ("Positive", "lm_positive_tfidf"),
        ("Uncertainty", "lm_uncertainty_tfidf"),
        ("Litigious", "lm_litigious_tfidf"),
        ("Modal Strong", "lm_modal_strong_tfidf"),
        ("Modal Weak", "lm_modal_weak_tfidf"),
    ]

    def _panel_lines(panel_title: str, spec_order: Sequence[tuple[str, str]]) -> list[str]:
        cells = [
            _coef_cell(lookup.get(signal_name, {}).get(signal_name), estimate_scale=estimate_scale)
            for _, signal_name in spec_order
        ]
        return [
            rf"\multicolumn{{7}}{{c}}{{{_latex_escape(panel_title)}}} \\",
            r"\cmidrule(lr){1-7}",
            "Dependent Variable & " + " & ".join(_latex_escape(label) for label, _ in spec_order) + r" \\",
            r"\midrule",
            "Event period excess return & " + " & ".join(cells) + r" \\",
        ]

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\scriptsize",
        rf"{{\bfseries {title_number}}}\\[0.3em]",
        rf"{{\bfseries {_latex_escape(heading)}}}\\[0.25em]",
        rf"{{\itshape {_latex_escape(subtitle)}}}\\[0.7em]",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        *_panel_lines("Panel A: Proportional Weights", panel_a_order),
        r"\addlinespace",
        *_panel_lines("Panel B: tf.idf Weights", panel_b_order),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    note = _fit_summary_note(fit_info, [signal_name for _, signal_name in (*panel_a_order, *panel_b_order)])
    if note_suffix:
        note += f" {note_suffix}"
    note += " The staged Table VI artifact retains the filing-period excess-return surface only; abnormal-volume and postevent-volatility surfaces are not stored in the selected run."
    lines.append(rf"{{\footnotesize\emph{{Note:}} {_latex_escape(note)}}}")
    return "\n".join(lines)


def _render_normalized_difference_table(
    title_number: str,
    heading: str,
    subtitle: str,
    df: pl.DataFrame,
    *,
    estimate_scale: float | Callable[[dict[str, Any]], float] = 1.0,
    note_suffix: str = "",
) -> str:
    lookup, fit_info = _result_lookup(df)
    signal_order = ["normalized_difference_h4n_inf", "normalized_difference_negative"]
    controls = [
        ("Log(size)", "log_size"),
        ("Log(book-to-market)", "log_book_to_market"),
        ("Log(share turnover)", "log_share_turnover"),
        ("Pre-FFAlpha", "pre_ffalpha"),
        ("Institutional ownership", "institutional_ownership"),
        ("NASDAQ dummy", "nasdaq_dummy"),
    ]
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        rf"{{\bfseries {title_number}}}\\[0.3em]",
        rf"{{\bfseries {_latex_escape(heading)}}}\\[0.25em]",
        rf"{{\itshape {_latex_escape(subtitle)}}}\\[0.7em]",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"& H4N-Inf normalized difference & Fin-Neg normalized difference \\",
        r"\midrule",
        r"\multicolumn{3}{l}{\textit{Normalized-Difference Signals}} \\",
        " & ".join(
            [
                r"H4N-Inf normalized difference",
                _coef_cell(
                    lookup.get("normalized_difference_h4n_inf", {}).get("normalized_difference_h4n_inf"),
                    estimate_scale=estimate_scale,
                ),
                "",
            ]
        )
        + r" \\",
        " & ".join(
            [
                r"Fin-Neg normalized difference",
                "",
                _coef_cell(
                    lookup.get("normalized_difference_negative", {}).get("normalized_difference_negative"),
                    estimate_scale=estimate_scale,
                ),
            ]
        )
        + r" \\",
        r"\addlinespace",
        r"\multicolumn{3}{l}{\textit{Control Variables}} \\",
    ]
    for display_label, coefficient_name in controls:
        cells = [
            _coef_cell(lookup.get(signal_name, {}).get(coefficient_name), estimate_scale=estimate_scale)
            for signal_name in signal_order
        ]
        if not any(cells):
            continue
        lines.append(" & ".join([_latex_escape(display_label), *cells]) + r" \\")
    note = _fit_summary_note(fit_info, signal_order)
    if note_suffix:
        note += f" {note_suffix}"
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            rf"{{\footnotesize\emph{{Note:}} {_latex_escape(note)}}}",
        ]
    )
    return "\n".join(lines)


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
    return pivot.select(ordered)


def _render_trade_summary_table(run_dir: Path) -> str | None:
    trade_path = run_dir / TRADE_RESULTS_FILE
    if not trade_path.exists():
        return None
    trade = _summarize_trade_results(trade_path)
    signal_order = ["h4n_inf_prop", "fin_neg_prop", "h4n_inf_tfidf", "fin_neg_tfidf"]
    if "Signal" in trade.columns:
        trade = trade.with_columns(
            pl.col("Signal").replace_strict(
                signal_order,
                [1, 2, 3, 4],
                default=99,
            ).alias("_order")
        ).sort("_order", "Signal").drop("_order")
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"{\bfseries Table IA.II}\\[0.3em]",
        r"{\bfseries Long-Short Trading Strategy Summary}\\[0.7em]",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Signal & Mean long-short & FF4 alpha & Beta MKT & Beta SMB & Beta HML & Beta MOM & $R^2$ \\",
        r"\midrule",
    ]
    for row in trade.iter_rows(named=True):
        lines.append(
            " & ".join(
                [
                    _latex_escape(row.get("Signal")),
                    _format_float(row.get("Mean long-short"), digits=4),
                    _format_float(row.get("FF4 alpha"), digits=4),
                    _format_float(row.get("Beta MKT"), digits=4),
                    _format_float(row.get("Beta SMB"), digits=4),
                    _format_float(row.get("Beta HML"), digits=4),
                    _format_float(row.get("Beta MOM"), digits=4),
                    _format_float(row.get("R^2"), digits=4),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def _column_spec(columns: Sequence[str]) -> str:
    text_widths = {
        "Text scope": "2.6cm",
        "Specification": "3.0cm",
        "Control set": "2.1cm",
        "Coefficient": "3.2cm",
        "Outcome": "2.9cm",
        "Status": "2.2cm",
        "Failure reason": "4.0cm",
    }
    numeric = {
        "Estimate",
        "Std. error",
        "t-stat",
        "Observations",
        "Quarters",
        "Mean quarter n",
        "Year",
        "Control-set rows",
        "Estimation rows",
        "Missing outcome",
        "Missing signal",
        "Missing controls",
    }
    specs: list[str] = []
    for column_name in columns:
        if column_name in numeric:
            specs.append("R{1.55cm}")
        elif column_name in text_widths:
            specs.append(f"L{{{text_widths[column_name]}}}")
        else:
            specs.append("L{2.7cm}")
    return "".join(specs)


def _format_longtable_cell(value: Any, column_name: str) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        if math.isnan(value):
            return "--"
        if column_name in {
            "Estimate",
            "Std. error",
            "t-stat",
            "Mean quarter n",
        }:
            return f"{value:,.4f}"
        return f"{value:,.0f}" if abs(value - round(value)) < 1e-9 else f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return _latex_escape(value)


def _longtable_block(
    df: pl.DataFrame,
    *,
    title: str,
    font_size: str = r"\scriptsize",
) -> str:
    lines = [
        font_size,
        r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{longtable}}{{{_column_spec(df.columns)}}}",
        rf"\multicolumn{{{len(df.columns)}}}{{c}}{{\textbf{{{_latex_escape(title)}}}}} \\",
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
    for row in df.iter_rows(named=True):
        cells = [_format_longtable_cell(row[column], column) for column in df.columns]
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\end{longtable}")
    return "\n".join(lines)


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


def _build_review_notes(
    context: RunContext,
    run_dir: Path,
    *,
    full_export: bool,
    extension_run_dir: Path | None,
) -> list[str]:
    notes = [
        f"Source run completed at {_format_timestamp(_parse_iso_datetime(context.manifest.get('completed_at_utc')))}.",
    ]
    extended_stage = (context.manifest.get("stages", {}) or {}).get("table_i_sample_creation_1994_2024")
    if isinstance(extended_stage, dict) and extended_stage.get("status") == "disabled_by_run_config":
        notes.append(
            "The 1994-2024 sample-creation artifact exists in the run directory but is marked disabled_by_run_config in the manifest, so the report treats the 1994-2008 sample as the operative screening table."
        )
    if not (run_dir / FULL_RETURN_PANEL_FILE).exists():
        notes.append("The full-10-K return-regression panel is missing, so paper-style summary statistics and quintile visualization could not be generated.")
    if not (run_dir / TRADE_RESULTS_FILE).exists():
        notes.append("The IA.II trading-strategy summary is missing from the selected run.")
    if full_export and extension_run_dir is not None:
        notes.append(
            f"Full export mode includes extension tables from {extension_run_dir} using lm2011_extension_results.parquet and lm2011_extension_sample_loss.parquet as the authoritative extension artifacts."
        )
        notes.append(
            "The appendix yearly sample-size figures combine the 1994-2008 core run with the 2009-2024 extension run. The extension sample-loss artifact does not persist a full-document text scope, so the 2009-2024 full-document series is sourced from the extension event panel while item-scope series come from extension sample-loss rows."
        )
    notes.append(
        "Paper Table III cannot be reconstructed from the staged run alone because token-level word-frequency counts are not stored in the exported artifacts."
    )
    return notes


def _paper_table_sections(run_dir: Path) -> list[str]:
    sections: list[str] = []
    sample_block = _render_sample_creation_table(run_dir)
    if sample_block is not None:
        sections.append(sample_block)

    return_note = (
        "Displayed coefficients are multiplied by 100 to express return-regression estimates in percentage-point units, matching LM2011 table conventions."
    )
    sue_note = (
        "Displayed coefficients are multiplied by 100 to express SUE-regression estimates in percentage-point units. Following LM2011 Table VIII, the tf.idf word-list coefficients in columns (3) and (4) are multiplied by an additional 100 for presentation."
    )
    table_map = {filename: title for title, filename in CORE_QUARTERLY_RESULTS}
    for title, filename in CORE_QUARTERLY_RESULTS:
        path = run_dir / filename
        if not path.exists():
            continue
        df = pl.read_parquet(path)
        if filename.startswith("lm2011_table_viii"):
            sections.append(
                _render_four_column_wordlist_table(
                    title,
                    "Comparison of Negative Word Lists Using Standardized Unexpected Earnings Regressions",
                    "Full 10-K document",
                    df,
                    h4n_label="H4N-Inf",
                    finneg_label="Fin-Neg",
                    extra_controls=(
                        ("Analysts' earnings forecast dispersion", "analyst_dispersion"),
                        ("Analysts' earnings revisions", "analyst_revisions"),
                    ),
                    estimate_scale=_scale_table_viii_estimate,
                    note_suffix=sue_note,
                )
            )
        elif filename.startswith("lm2011_table_vi"):
            sections.append(
                _render_dictionary_surface_table(
                    title,
                    "Additional Word Lists and Filing Period Excess Return Regressions",
                    "Finance-dictionary surface reconstructed from staged coefficients",
                    df,
                    estimate_scale=100.0,
                    note_suffix=return_note,
                )
            )
        elif filename.startswith("lm2011_table_v"):
            sections.append(
                _render_four_column_wordlist_table(
                    title,
                    "Comparison of Negative Word Lists Using Filing Period Excess Return Regressions",
                    "MD&A section",
                    df,
                    h4n_label="H4N-Inf (only MD&A)",
                    finneg_label="Fin-Neg (only MD&A)",
                    estimate_scale=100.0,
                    note_suffix=return_note,
                )
            )
        elif filename.startswith("lm2011_table_iv"):
            sections.append(
                _render_four_column_wordlist_table(
                    title,
                    "Comparison of Negative Word Lists Using Filing Period Excess Return Regressions",
                    "Full 10-K document",
                    df,
                    h4n_label="H4N-Inf (Harvard-IV-4-Neg with inflections)",
                    finneg_label="Fin-Neg (negative)",
                    estimate_scale=100.0,
                    note_suffix=return_note,
                )
            )
        elif filename.startswith("lm2011_table_ia_i"):
            sections.append(
                _render_normalized_difference_table(
                    title,
                    "Normalized-Difference Excess Return Regressions",
                    "Full 10-K document",
                    df,
                    estimate_scale=100.0,
                    note_suffix=return_note,
                )
            )
    trade_block = _render_trade_summary_table(run_dir)
    if trade_block is not None:
        sections.append(trade_block)
    return sections


def _extension_sections(extension_run_dir: Path) -> list[str]:
    sample_loss = _summarize_extension_sample_loss(extension_run_dir / EXTENSION_SAMPLE_LOSS_FILE)
    results = _summarize_extension_results(extension_run_dir / EXTENSION_RESULTS_FILE)
    return [
        r"\clearpage",
        r"\section*{Extension Tables}",
        _longtable_block(sample_loss, title="LM2011 Extension Sample Loss", font_size=r"\scriptsize"),
        _longtable_block(results, title="LM2011 Extension Results", font_size=r"\scriptsize"),
    ]


def _write_report(
    context: RunContext,
    output_dir: Path,
    *,
    full_export: bool,
    extension_run_dir: Path | None,
) -> Path:
    figure_dir = _ensure_dir(output_dir / FIGURE_SUBDIR)
    table_output_dir = _ensure_dir(output_dir / TABLE_SUBDIR)
    for stale_png in figure_dir.glob("*.png"):
        stale_png.unlink()

    review_notes = _build_review_notes(
        context,
        context.run_dir,
        full_export=full_export,
        extension_run_dir=extension_run_dir,
    )
    summary_stats_block = _render_summary_statistics_table(context.run_dir, table_output_dir)
    quintile_figure = _plot_median_excess_return_quintiles(context.run_dir, figure_dir, table_output_dir)
    trading_figure = _plot_trading_returns(context.run_dir, figure_dir)
    yearly_sample_sizes_frame = (
        _combined_yearly_sample_sizes_frame(context.run_dir, extension_run_dir)
        if extension_run_dir is not None
        else None
    )
    yearly_sample_sizes_c0_figure: Path | None = None
    yearly_sample_sizes_c1_c2_figure: Path | None = None
    if yearly_sample_sizes_frame is not None and yearly_sample_sizes_frame.height > 0:
        yearly_sample_sizes_frame.write_csv(table_output_dir / "lm2011_yearly_sample_sizes_appendix.csv")
        yearly_sample_sizes_c0_figure = _plot_yearly_sample_sizes_by_control_group(
            yearly_sample_sizes_frame,
            control_group="C0",
            figure_dir=figure_dir,
        )
        yearly_sample_sizes_c1_c2_figure = _plot_yearly_sample_sizes_by_control_group(
            yearly_sample_sizes_frame,
            control_group="C1/C2",
            figure_dir=figure_dir,
        )
    paper_tables = _paper_table_sections(context.run_dir)

    sections: list[str] = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{array,booktabs,longtable,pdflscape,graphicx,float,xcolor,hyperref}",
        r"\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}",
        r"\newcolumntype{R}[1]{>{\raggedleft\arraybackslash}p{#1}}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{0.65em}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{document}",
        r"\begin{center}",
        r"{\LARGE \textbf{LM2011 Replication Tables and Figures}}\\[0.5em]",
        rf"{{\small Source run: \url{{{context.run_dir.resolve().as_posix()}}}}}\\[0.3em]",
        rf"{{\small Manifest: \url{{{context.manifest_path.resolve().as_posix()}}}}}",
        r"\end{center}",
    ]

    if paper_tables:
        sections.extend(paper_tables[:1])
    if summary_stats_block is not None:
        sections.extend([r"\clearpage", summary_stats_block])
    next_figure_number = 1
    if quintile_figure is not None:
        sections.extend(
            [
                r"\clearpage",
                _render_manual_figure(
                    quintile_figure.relative_to(output_dir),
                    number=str(next_figure_number),
                    caption="Median filing period excess return by negative-tone quintile for H4N-Inf and Fin-Neg using the available 1994--2008 full-10-K sample from the selected run. Returns are expressed in percent.",
                    width="0.78\\textwidth",
                ),
            ]
        )
        next_figure_number += 1
    if len(paper_tables) > 1:
        sections.extend([r"\clearpage", *sum(([block, r"\clearpage"] for block in paper_tables[1:]), [])[:-1]])
    if trading_figure is not None:
        sections.extend(
            [
                r"\clearpage",
                _render_manual_figure(
                    trading_figure.relative_to(output_dir),
                    number=str(next_figure_number),
                    caption="Cumulative long-short returns by LM2011 sort signal from the IA.II monthly trading-strategy artifact.",
                    width="0.94\\textwidth",
                ),
            ]
        )
        next_figure_number += 1
    appendix_figures: list[str] = []
    if yearly_sample_sizes_c0_figure is not None:
        appendix_figures.extend(
            [
                _render_manual_figure(
                    yearly_sample_sizes_c0_figure.relative_to(output_dir),
                    number=f"A{1 if not appendix_figures else 2}",
                    caption=(
                        "Appendix sample-coverage figure for C0. Full 10-K and Item 7 counts span 1994--2024, while Item 1A begins in 2009. "
                        "The 1994--2008 Item 7 series uses the core MD&A return panel with the 250-token floor, the 2009--2024 full-document series uses the extension event panel, and the 2009--2024 item-scope series use extension sample-loss counts."
                    ),
                    width="0.92\\textwidth",
                )
            ]
        )
    if yearly_sample_sizes_c1_c2_figure is not None:
        appendix_figures.extend(
            [
                _render_manual_figure(
                    yearly_sample_sizes_c1_c2_figure.relative_to(output_dir),
                    number=f"A{1 if not appendix_figures else 2}",
                    caption=(
                        "Appendix sample-coverage figure for C1/C2. Full 10-K and Item 7 counts span 1994--2024, while Item 1A begins in 2009. "
                        "This figure is included to show the low ownership-conditioned coverage in the common-support specifications."
                    ),
                    width="0.92\\textwidth",
                )
            ]
        )

    if review_notes:
        sections.extend([r"\clearpage", r"\section*{Supplementary Notes}", r"\begin{itemize}"])
        sections.extend(rf"\item {_latex_escape(note)}" for note in review_notes)
        sections.append(r"\end{itemize}")

    if extension_run_dir is not None:
        sections.extend(_extension_sections(extension_run_dir))

    if appendix_figures:
        sections.extend(
            [
                r"\clearpage",
                r"\section*{Appendix Figures}",
                *appendix_figures,
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
    extension_run_dir = _resolve_extension_run_dir(context.run_dir, args.extension_run_dir) if args.full_export else None
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

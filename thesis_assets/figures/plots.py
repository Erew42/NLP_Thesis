from __future__ import annotations

import math
import textwrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import polars as pl


_SCOPE_LABELS = {
    "item_7_mda": "Item 7 MD&A",
    "item_1a_risk_factors": "Item 1A risk factors",
}


def build_sample_funnel_figure(
    df: pl.DataFrame,
    *,
    label_col: str = "display_label",
    value_col: str = "sample_size_value",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.38 * max(df.height, 1))))
    labels = [_wrap_label(value, width=42) for value in df.get_column(label_col).to_list()]
    values = [float(value or 0.0) for value in df.get_column(value_col).to_list()]
    positions = list(range(len(values)))

    ax.barh(positions, values, color="#3d6f8e")
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Filings / observations")
    ax.grid(axis="x", alpha=0.18, linewidth=0.6)
    ax.set_axisbelow(True)
    _annotate_horizontal_bars(ax, positions, values)
    fig.tight_layout()
    return fig


def build_sample_attrition_figure(
    df: pl.DataFrame,
    *,
    label_col: str = "display_label",
    value_col: str = "observations_removed",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.36 * max(df.height, 1))))
    labels = [_wrap_label(value, width=42) for value in df.get_column(label_col).to_list()]
    values = [float(value or 0.0) for value in df.get_column(value_col).to_list()]
    positions = list(range(len(values)))

    ax.barh(positions, values, color="#98633d")
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Observations removed")
    ax.grid(axis="x", alpha=0.18, linewidth=0.6)
    ax.set_axisbelow(True)
    _annotate_horizontal_bars(ax, positions, values)
    fig.tight_layout()
    return fig


def build_sample_bridge_figure(
    df: pl.DataFrame,
    *,
    label_col: str = "bridge_stage",
    value_col: str = "sample_size_value",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    labels = [_wrap_label(value, width=18) for value in df.get_column(label_col).to_list()]
    values = [float(value or 0.0) for value in df.get_column(value_col).to_list()]
    positions = list(range(len(values)))

    ax.plot(positions, values, color="#345c72", marker="o", linewidth=2.0)
    ax.fill_between(positions, values, color="#d7e6ed", alpha=0.75)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=0, fontsize=8)
    ax.set_ylabel("Filings / observations")
    ax.grid(axis="y", alpha=0.18, linewidth=0.6)
    ax.set_axisbelow(True)
    for position, value in zip(positions, values):
        ax.annotate(
            f"{int(round(value)):,}",
            xy=(position, value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    fig.tight_layout()
    return fig


def build_ecdf_lines_figure(
    df: pl.DataFrame,
    *,
    x_col: str,
    y_col: str = "ecdf",
    series_col: str = "series_label",
    x_label: str,
    y_label: str = "ECDF",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for series in df.get_column(series_col).unique().sort().to_list():
        series_df = df.filter(pl.col(series_col) == series).sort(x_col)
        ax.step(
            series_df.get_column(x_col).to_list(),
            series_df.get_column(y_col).to_list(),
            where="post",
            linewidth=1.7,
            label=str(series),
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.16, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def build_concordance_figure(
    df: pl.DataFrame,
    *,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    x_values = df.get_column(x_col).to_list()
    y_values = df.get_column(y_col).to_list()

    hexbin = ax.hexbin(
        x_values,
        y_values,
        gridsize=28,
        mincnt=1,
        cmap="Blues",
        linewidths=0.2,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.15, linewidth=0.5)
    ax.set_axisbelow(True)

    if df.height >= 2:
        correlation = df.select(pl.corr(x_col, y_col)).item()
        if correlation is not None and math.isfinite(float(correlation)):
            ax.text(
                0.02,
                0.98,
                f"corr = {float(correlation):.3f}\nN = {df.height:,}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#d0d7de"},
            )

    colorbar = fig.colorbar(hexbin, ax=ax)
    colorbar.set_label("Observation count")
    fig.tight_layout()
    return fig


def build_concordance_by_scope_figure(
    df: pl.DataFrame,
    *,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
) -> plt.Figure:
    scopes = ["item_7_mda", "item_1a_risk_factors"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharex=False, sharey=False)
    for ax, scope in zip(axes, scopes):
        scope_df = df.filter(pl.col("text_scope") == scope)
        if scope_df.is_empty():
            ax.set_title(_scope_label(scope))
            ax.text(0.5, 0.5, "No matched rows", transform=ax.transAxes, ha="center", va="center")
            continue
        x_values = scope_df.get_column(x_col).to_list()
        y_values = scope_df.get_column(y_col).to_list()
        hexbin = ax.hexbin(
            x_values,
            y_values,
            gridsize=24,
            mincnt=1,
            cmap="Blues",
            linewidths=0.2,
        )
        ax.set_title(_scope_label(scope))
        ax.grid(alpha=0.15, linewidth=0.5)
        ax.set_axisbelow(True)
        if scope_df.height >= 2:
            correlation = scope_df.select(pl.corr(x_col, y_col)).item()
            if correlation is not None and math.isfinite(float(correlation)):
                ax.text(
                    0.02,
                    0.98,
                    f"corr = {float(correlation):.3f}\nN = {scope_df.height:,}",
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=8,
                    bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#d0d7de"},
                )
        fig.colorbar(hexbin, ax=ax, fraction=0.046, pad=0.04).set_label("Observation count")
    axes[0].set_ylabel(y_label)
    for ax in axes:
        ax.set_xlabel(x_label)
    fig.tight_layout()
    return fig


def _annotate_horizontal_bars(ax: plt.Axes, positions: list[int], values: list[float]) -> None:
    max_value = max(values) if values else 0.0
    offset = max_value * 0.01 if max_value > 0 else 1.0
    for position, value in zip(positions, values):
        ax.text(
            value + offset,
            position,
            f"{int(round(value)):,}",
            va="center",
            fontsize=8,
        )


def _wrap_label(value: object, *, width: int) -> str:
    return "\n".join(textwrap.wrap(str(value), width=width, break_long_words=False))


def _scope_label(scope: str) -> str:
    return _SCOPE_LABELS.get(scope, scope)

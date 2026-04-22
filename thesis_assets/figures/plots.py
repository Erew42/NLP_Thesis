from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import polars as pl


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

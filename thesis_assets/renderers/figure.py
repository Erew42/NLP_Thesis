from __future__ import annotations

import gc
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def write_figure_bundle(
    fig: Figure,
    path_stem: Path,
) -> dict[str, Path]:
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = path_stem.with_suffix(".png")
    pdf_path = path_stem.with_suffix(".pdf")
    try:
        fig.savefig(png_path, dpi=220, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
    finally:
        plt.close(fig)
        gc.collect()
    return {
        "png": png_path,
        "pdf": pdf_path,
    }

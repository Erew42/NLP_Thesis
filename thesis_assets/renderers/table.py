from __future__ import annotations

import math
from datetime import date
from datetime import datetime
from pathlib import Path

import polars as pl

from thesis_assets.templates import TABLE_WRAPPER_TEMPLATE_PATH


def write_csv_table(df: pl.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path)
    return path


def write_markdown_table(df: pl.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in df.iter_rows():
        lines.append("| " + " | ".join(_markdown_text(_format_value(value)) for value in row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_latex_table(
    df: pl.DataFrame,
    path: Path,
    *,
    caption: str,
    notes: str,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    template = TABLE_WRAPPER_TEMPLATE_PATH.read_text(encoding="utf-8")
    header_row = " & ".join(_latex_escape(column) for column in df.columns)
    body_lines = [
        " & ".join(_latex_escape(_format_value(value)) for value in row) + r" \\"
        for row in df.iter_rows()
    ]
    rendered = (
        template.replace("{caption}", _latex_escape(caption))
        .replace("{alignment}", "l" * max(len(df.columns), 1))
        .replace("{header_row}", header_row)
        .replace("{body_rows}", "\n".join(body_lines))
        .replace("{notes}", _latex_escape(notes))
    )
    path.write_text(rendered, encoding="utf-8")
    return path


def _format_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        if value.is_integer():
            return f"{int(value):,}"
        return f"{value:.4f}"
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _markdown_text(value: str) -> str:
    return value.replace("|", r"\|").replace("\n", " ")


def _latex_escape(value: str) -> str:
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
    out = value
    for source, replacement in replacements.items():
        out = out.replace(source, replacement)
    return out

from __future__ import annotations

import math
from datetime import date
from datetime import datetime
from pathlib import Path

import polars as pl

MARKDOWN_CELL_WRAP = 36
MARKDOWN_CELL_MAX_CHARS = 280
LATEX_CELL_MAX_CHARS = 180

_DUPLICATE_DISPLAY_COLUMNS = {
    "text_scope": "scope",
    "dictionary_family_source": "dictionary_family",
    "specification_name": "specification",
    "coefficient_name": "coefficient",
    "comparison_name": "comparison",
    "scope_label": "scope",
}

_CONSTANT_CONTEXT_COLUMNS = (
    "canonical_estimator_status",
    "common_success_policy",
    "control_set_id",
    "estimator_status",
    "model_name",
    "model_name_source",
    "model_revision",
    "model_revision_source",
    "outcome_name",
    "reported_scale",
    "sensitivity_assumptions",
    "spread_definition",
    "tokenizer_revision",
    "tokenizer_revision_source",
    "visible_prefix_estimator_status",
    "weighting_rule",
)

_SUPPRESSED_CONSTANT_CONTEXT_COLUMNS = {
    "canonical_estimator_status",
    "estimator_status",
    "visible_prefix_estimator_status",
}

_HEADER_LABELS = {
    "abnormal_volume": "Abn. volume",
    "activation_status": "Status",
    "affected_doc_scope_count": "Affected doc-scopes",
    "analyst_normalized_rows": "Analyst normalized",
    "analyst_request_rows": "Analyst requested",
    "at_512_rows": "At 512",
    "at_512_sentence_share": "At-512 share",
    "average_r2": "Avg. R2",
    "calendar_year": "Year",
    "coefficient": "Coef.",
    "coefficient_name": "Coef.",
    "coefficient_n_obs": "Coef. N",
    "coefficient_n_quarters": "Coef. quarters",
    "coefficient_x100": "Coef. x100",
    "control_set_id": "Controls",
    "coverage_source": "Coverage source",
    "delta_adj_r2_equal_quarter": "Delta adj. R2 equal qtr.",
    "delta_adj_r2_weighted": "Delta adj. R2 weighted",
    "dependent_variable": "Outcome",
    "dictionary_family": "Dictionary",
    "dictionary_family_source": "Dictionary",
    "effect_share_of_mean": "Effect / mean",
    "empty_after_cleaning_rows": "Empty after cleaning",
    "equal_quarter_avg_adj_r2": "Equal-qtr. adj. R2",
    "estimate": "Estimate",
    "estimate_delta": "Delta est.",
    "extraction_rate": "Extraction rate",
    "feature_family": "Feature family",
    "finbert_untruncated_token_count_mass": "Untruncated FinBERT tokens",
    "gained_10pct_by_nw4": "Gained 10% by NW4",
    "iqr_effect_basis_points": "IQR effect, bps",
    "iqr_effect_percentage_points": "IQR effect, pp",
    "iqr_effect_return": "IQR effect",
    "large_removal_warning_rows": "Large removals",
    "lost_5pct_by_nw4": "Lost 5% by NW4",
    "manual_audit_queue_n": "Manual audit queue",
    "mean_filing_period_excess_return": "Mean filing return",
    "mean_postevent_volatility_percentage_points": "Mean postevent vol., pp",
    "mean_quarter_n": "Mean qtr. N",
    "metric_label": "Metric",
    "model_or_comparison": "Model / comparison",
    "n_filings_candidate": "Candidate filings",
    "n_filings_extracted": "Extracted filings",
    "n_obs": "N",
    "n_quarters": "Quarters",
    "nw_lags": "NW lags",
    "nw_p_value": "NW p-value",
    "nw_p_value_delta_adj_r2": "NW p-value",
    "nw_t_stat": "NW t-stat",
    "nw_t_stat_delta_adj_r2": "NW t-stat",
    "original_char_mass": "Original chars",
    "original_lm_token_mass": "Original LM tokens",
    "ownership_available_rate": "Ownership rate",
    "ownership_available_rows": "Ownership rows",
    "ownership_common_support_rate": "Common-support rate",
    "ownership_common_support_rows": "Common-support rows",
    "outcome_name": "Outcome",
    "p_value": "p-value",
    "p_value_delta": "Delta p",
    "postevent_return_volatility": "Postevent volatility",
    "pre_ffalpha": "Pre-FF alpha",
    "quality_metric": "Quality metric",
    "reference_stub_rows": "Reference stubs",
    "reported_scale": "Scale",
    "retained_finbert_token_count_512_mass": "Retained FinBERT tokens",
    "score_iqr": "Score IQR",
    "score_q25": "Score p25",
    "score_q75": "Score p75",
    "se_ratio_nw4_to_nw1": "SE ratio NW4/NW1",
    "sentence_rows": "Sentences",
    "sig5_all_lags": "Sig. 5%, all lags",
    "skipped_quarters_excluded": "Skipped quarters excluded",
    "source_presentation_note": "Presentation note",
    "signal_name": "Signal",
    "specification_name": "Spec.",
    "standard_error": "SE",
    "standard_error_delta": "Delta SE",
    "std_error": "SE",
    "tail_truncated_rows": "Tail truncations",
    "text_scope": "Scope",
    "toc_trimmed_rows": "TOC trims",
    "token_count_mean": "Mean tokens",
    "token_count_median": "Median tokens",
    "token_count_p05": "p05 tokens",
    "total_n_obs": "Total N",
    "t_stat": "t-stat",
    "t_stat_delta": "Delta t",
    "true_over_512_sentence_share": "True >512 share",
    "truly_over_512_rows": "True >512",
    "unique_docs": "Docs",
    "variant_description": "Variant",
    "variant_id": "Variant ID",
    "visible_char_retention": "Visible char retention",
    "visible_lm_token_retention": "Visible LM retention",
    "visible_prefix_char_mass": "Visible chars",
    "visible_prefix_fallback_rows": "Fallback rows",
    "visible_prefix_lm_token_mass": "Visible LM tokens",
    "visible_prefix_policy_ids": "Prefix policy",
    "weighted_avg_adj_r2": "Weighted adj. R2",
    "weighted_avg_delta_adj_r2": "Weighted delta adj. R2",
    "weighting_rule": "Weighting",
}

_VALUE_LABELS = {
    "filing_period_excess_return": "Filing return",
    "full_10k": "Full 10-K",
    "item_1a_risk_factors": "Item 1A",
    "item_7_mda": "Item 7",
    "items_1a_7_combined": "Items 1A+7",
    "Item 1A risk factors": "Item 1A",
    "Item 7 MD&A": "Item 7",
    "quarter_observation_count": "Obs.-weighted",
    "equal_quarter": "Equal-qtr.",
    "postevent_return_volatility": "Postevent volatility",
    "filing_period_excess_return_x100": "Filing return x100",
}


def write_csv_table(df: pl.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path)
    return path


def write_markdown_table(df: pl.DataFrame, path: Path, *, notes: str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    display_df, layout_notes = _prepare_display_table(df)
    headers = [_display_header(column) for column in display_df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in display_df.iter_rows(named=True):
        values = [
            _markdown_text(
                _format_display_value(
                    row[column],
                    column=column,
                    max_chars=MARKDOWN_CELL_MAX_CHARS,
                    wrap_width=MARKDOWN_CELL_WRAP,
                    line_break="<br>",
                )
            )
            for column in display_df.columns
        ]
        lines.append("| " + " | ".join(values) + " |")
    if notes:
        lines.extend(["", f"_Notes:_ {_markdown_text(notes)}"])
    if layout_notes:
        lines.extend(["", f"_Table note:_ {_markdown_text(_layout_note_text(layout_notes))}"])
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
    display_df, layout_notes = _prepare_display_table(df)
    header_row = " & ".join(_latex_escape(_display_header(column)) for column in display_df.columns)
    body_lines = [
        " & ".join(
            _latex_escape(
                _format_display_value(
                    row[column],
                    column=column,
                    max_chars=LATEX_CELL_MAX_CHARS,
                    wrap_width=0,
                    line_break=" ",
                )
            )
            for column in display_df.columns
        )
        + r" \\"
        for row in display_df.iter_rows(named=True)
    ]
    rendered = _render_latex_table(
        display_df,
        caption=caption,
        notes=_combine_notes(notes, layout_notes),
        header_row=header_row,
        body_rows="\n".join(body_lines),
    )
    path.write_text(rendered, encoding="utf-8")
    return path


def _prepare_display_table(df: pl.DataFrame) -> tuple[pl.DataFrame, tuple[str, ...]]:
    drop_columns = {column for column in df.columns if column.startswith("_")}
    for raw_column, display_column in _DUPLICATE_DISPLAY_COLUMNS.items():
        if raw_column in df.columns and display_column in df.columns:
            drop_columns.add(raw_column)

    layout_notes: list[str] = []
    if df.height > 1:
        for column in _CONSTANT_CONTEXT_COLUMNS:
            if column not in df.columns or column in drop_columns:
                continue
            series = df.get_column(column)
            if _is_constant(series):
                drop_columns.add(column)
                if column in _SUPPRESSED_CONSTANT_CONTEXT_COLUMNS:
                    continue
                layout_notes.append(
                    f"{_display_header(column)}={_constant_column_value(series)}"
                )

    display_columns = [column for column in df.columns if column not in drop_columns]
    if not display_columns and df.columns:
        display_columns = [df.columns[0]]
    return df.select(display_columns), tuple(layout_notes)


def _is_constant(series: pl.Series) -> bool:
    non_null = series.drop_nulls()
    return len(non_null) == 0 or non_null.n_unique() <= 1


def _constant_column_value(series: pl.Series) -> str:
    non_null = series.drop_nulls()
    if len(non_null) == 0:
        return ""
    return _format_display_value(
        non_null[0],
        column=series.name,
        max_chars=90,
        wrap_width=0,
        line_break=" ",
    )


def _render_latex_table(
    df: pl.DataFrame,
    *,
    caption: str,
    notes: str,
    header_row: str,
    body_rows: str,
) -> str:
    column_count = max(len(df.columns), 1)
    alignment = _latex_alignment(df)
    long_table = df.height > 35
    wide_table = column_count > 10
    font_size = r"\scriptsize" if wide_table or long_table else r"\small"
    tabcolsep = "2pt" if wide_table else "4pt"
    package_comment = (
        r"% Requires \usepackage{array,longtable,graphicx,pdflscape} in the thesis preamble "
        r"for all generated table layouts."
    )
    caption_text = _latex_escape(caption)
    notes_text = _latex_escape(notes)
    if long_table:
        table_body = "\n".join(
            [
                package_comment,
                _latex_landscape_begin(wide_table),
                "{",
                font_size,
                rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}",
                r"\renewcommand{\arraystretch}{1.12}",
                rf"\begin{{longtable}}{{{alignment}}}",
                rf"\caption{{{caption_text}}}\\",
                r"\hline",
                header_row + r" \\",
                r"\hline",
                r"\endfirsthead",
                r"\hline",
                header_row + r" \\",
                r"\hline",
                r"\endhead",
                body_rows,
                r"\hline",
                rf"\multicolumn{{{column_count}}}{{p{{0.95\textwidth}}}}{{\footnotesize \textit{{Notes:}} {notes_text}}}\\",
                r"\end{longtable}",
                "}",
                _latex_landscape_end(wide_table),
            ]
        )
        return table_body + "\n"

    tabular = "\n".join(
        [
            rf"\begin{{tabular}}{{{alignment}}}",
            r"\hline",
            header_row + r" \\",
            r"\hline",
            body_rows,
            r"\hline",
            r"\end{tabular}",
        ]
    )
    if wide_table:
        tabular = "\n".join([r"\resizebox{\textwidth}{!}{%", tabular, r"}"])
    return (
        "\n".join(
            [
                package_comment,
                r"\begin{table}[htbp]",
                r"\centering",
                rf"\caption{{{caption_text}}}",
                font_size,
                rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}",
                r"\renewcommand{\arraystretch}{1.12}",
                tabular,
                r"\par\smallskip",
                rf"\begin{{minipage}}{{0.95\textwidth}}\footnotesize \textit{{Notes:}} {notes_text}\end{{minipage}}",
                r"\end{table}",
            ]
        )
        + "\n"
    )


def _latex_landscape_begin(wide_table: bool) -> str:
    return r"\begin{landscape}" if wide_table else ""


def _latex_landscape_end(wide_table: bool) -> str:
    return r"\end{landscape}" if wide_table else ""


def _latex_alignment(df: pl.DataFrame) -> str:
    column_count = max(len(df.columns), 1)
    if column_count <= 8 and not _needs_wrapped_latex_columns(df):
        return "".join("r" if _is_numeric_dtype(df.schema[column]) else "l" for column in df.columns)
    width = max(0.032, min(0.12, 0.90 / column_count))
    specs = []
    for column in df.columns:
        direction = r"\raggedleft" if _is_numeric_dtype(df.schema[column]) else r"\raggedright"
        specs.append(rf">{{{direction}\arraybackslash}}p{{{width:.3f}\textwidth}}")
    return "".join(specs)


def _needs_wrapped_latex_columns(df: pl.DataFrame) -> bool:
    for column in df.columns:
        if len(_display_header(column)) > 14:
            return True
        if _is_numeric_dtype(df.schema[column]):
            continue
        for value in df.get_column(column).head(50):
            if len(_format_value(value)) > 36:
                return True
    return False


def _is_numeric_dtype(dtype: pl.DataType) -> bool:
    return dtype.is_integer() or dtype.is_float()


def _combine_notes(notes: str, layout_notes: tuple[str, ...]) -> str:
    if not layout_notes:
        return notes
    return f"{notes} {_layout_note_text(layout_notes)}"


def _layout_note_text(layout_notes: tuple[str, ...]) -> str:
    shown = "; ".join(layout_notes[:8])
    if len(layout_notes) > 8:
        shown = f"{shown}; plus {len(layout_notes) - 8} additional constant columns"
    shown = shown.replace("=", " = ")
    return f"Constant fields omitted from the displayed table: {shown}."


def _display_header(column: str) -> str:
    if column in _HEADER_LABELS:
        return _HEADER_LABELS[column]
    for prefix, label in (
        ("canonical_", "Canon. "),
        ("visible_prefix_", "Visible "),
        ("delta_", "Delta "),
        ("equal_quarter_", "Equal-qtr. "),
        ("weighted_", "Weighted "),
    ):
        if column.startswith(prefix):
            return label + _prefixed_header_detail(_display_header(column.removeprefix(prefix)))
    if column.startswith("t_nw"):
        return "t NW" + column.removeprefix("t_nw")
    if column.startswith("stars_nw"):
        return "Stars NW" + column.removeprefix("stars_nw")
    fallback = column.replace("_", " ").title()
    replacements = {
        "Adj R2": "adj. R2",
        "Cik": "CIK",
        "Doc": "Doc",
        "Finbert": "FinBERT",
        "Id": "ID",
        "Lm": "LM",
        "Lm2011": "LM2011",
        "Mda": "MD&A",
        "Nw": "NW",
        "R2": "R2",
        "Sec": "SEC",
        "Tfidf": "TF-IDF",
    }
    for source, replacement in replacements.items():
        fallback = fallback.replace(source, replacement)
    return fallback


def _prefixed_header_detail(label: str) -> str:
    if label in {"N", "SE"} or label.startswith(("NW", "R2")):
        return label
    return label[:1].lower() + label[1:] if label else label


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


def _format_display_value(
    value: object,
    *,
    column: str,
    max_chars: int,
    wrap_width: int,
    line_break: str,
) -> str:
    formatted = _compact_value(_format_value(value))
    if len(formatted) > max_chars:
        suffix = " ... [truncated; see CSV]"
        formatted = formatted[: max(max_chars - len(suffix), 0)].rstrip() + suffix
    if wrap_width > 0:
        formatted = _wrap_text(formatted, wrap_width=wrap_width, line_break=line_break)
    return formatted


def _compact_value(value: str) -> str:
    stripped = value.strip()
    if stripped in _VALUE_LABELS:
        return _VALUE_LABELS[stripped]
    if ";" in stripped:
        stripped = stripped.replace(";", "; ")
    if "|" in stripped and len(stripped) < 240:
        return " | ".join(part.strip() for part in stripped.split("|") if part.strip())
    return stripped


def _wrap_text(value: str, *, wrap_width: int, line_break: str) -> str:
    if len(value) <= wrap_width:
        return value
    words = value.split()
    if not words:
        return value
    lines: list[str] = []
    current = ""
    for word in words:
        if len(word) > wrap_width:
            if current:
                lines.append(current)
                current = ""
            lines.extend(_split_long_word(word, wrap_width=wrap_width))
            continue
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= wrap_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return line_break.join(lines)


def _split_long_word(word: str, *, wrap_width: int) -> list[str]:
    for separator in ("_", "-", "/"):
        if separator in word:
            parts = word.split(separator)
            chunks: list[str] = []
            current = ""
            for index, part in enumerate(parts):
                token = part if index == 0 else f"{separator}{part}"
                candidate = token if not current else f"{current}{token}"
                if current and len(candidate) > wrap_width:
                    chunks.append(current)
                    current = token.lstrip(separator)
                else:
                    current = candidate
            if current:
                chunks.append(current)
            return chunks
    return [word[index : index + wrap_width] for index in range(0, len(word), wrap_width)]


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

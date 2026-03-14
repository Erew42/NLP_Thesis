from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import xlsxwriter
from xlsxwriter.utility import xl_rowcol_to_cell


def _write_table_sheet(
    workbook: xlsxwriter.Workbook,
    *,
    sheet_name: str,
    df: pl.DataFrame,
    header_fmt: xlsxwriter.format.Format,
    text_columns: tuple[str, ...],
) -> xlsxwriter.worksheet.Worksheet:
    worksheet = workbook.add_worksheet(sheet_name)
    worksheet.freeze_panes(1, 0)

    for col_idx, name in enumerate(df.columns):
        worksheet.write(0, col_idx, name, header_fmt)

    text_col_set = set(text_columns)
    for row_idx, row in enumerate(df.iter_rows(named=True), start=1):
        for col_idx, name in enumerate(df.columns):
            value = row[name]
            if value is None:
                continue
            if name in text_col_set:
                worksheet.write_string(row_idx, col_idx, str(value))
            elif isinstance(value, bool):
                worksheet.write_boolean(row_idx, col_idx, value)
            elif isinstance(value, (int, float)):
                worksheet.write_number(row_idx, col_idx, value)
            else:
                worksheet.write_string(row_idx, col_idx, str(value))

    last_row = max(df.height, 1)
    last_col = max(len(df.columns) - 1, 0)
    worksheet.autofilter(0, 0, last_row, last_col)

    sample_rows = min(df.height, 100)
    for col_idx, name in enumerate(df.columns):
        width = max(len(name), 12)
        if sample_rows > 0:
            sample_values = df[name].head(sample_rows).to_list()
            width = max(width, max(len("" if value is None else str(value)) for value in sample_values))
        worksheet.set_column(col_idx, col_idx, min(width + 2, 40))

    return worksheet


def _sheet_cell_ref(row_idx: int, col_idx: int) -> str:
    return xl_rowcol_to_cell(row_idx, col_idx)


def _sheet_range_ref(sheet_name: str, first_row: int, last_row: int, col_idx: int) -> str:
    first_cell = xl_rowcol_to_cell(first_row, col_idx, row_abs=True, col_abs=True)
    last_cell = xl_rowcol_to_cell(last_row, col_idx, row_abs=True, col_abs=True)
    return f"'{sheet_name}'!{first_cell}:{last_cell}"


def _prefill_refinitiv_extended_lookup_formulas(
    worksheet: xlsxwriter.worksheet.Worksheet,
    *,
    column_names: tuple[str, ...],
    row_count: int,
) -> None:
    if row_count <= 0:
        return

    column_map = {name: idx for idx, name in enumerate(column_names)}
    required_columns = {
        "all_successful_attempts_consistent",
        "ISIN_lookup_input",
        "CUSIP_lookup_input",
        "TICKER_lookup_input",
    }
    missing = sorted(required_columns - set(column_map))
    if missing:
        raise ValueError(f"extended lookup worksheet missing required columns: {missing}")

    lookup_fields = {
        "returned_ric": "TR.RIC",
        "returned_name": "TR.CommonName",
        "returned_isin": "TR.ISIN",
        "returned_cusip": "TR.CUSIP",
    }
    identifier_types = ("ISIN", "CUSIP", "TICKER")
    identifier_pairs = (
        ("ISIN", "CUSIP"),
        ("ISIN", "TICKER"),
        ("CUSIP", "TICKER"),
    )

    for row_idx in range(1, row_count + 1):
        for identifier_type in identifier_types:
            lookup_input_ref = _sheet_cell_ref(row_idx, column_map[f"{identifier_type}_lookup_input"])
            for target_suffix, tr_field in lookup_fields.items():
                target_col = column_map[f"{identifier_type}_{target_suffix}"]
                worksheet.write_formula(
                    row_idx,
                    target_col,
                    f'=IF({lookup_input_ref}="","",_xll.TR({lookup_input_ref},"{tr_field}"))',
                    None,
                    "",
                )

            returned_ric_ref = _sheet_cell_ref(row_idx, column_map[f"{identifier_type}_returned_ric"])
            worksheet.write_formula(
                row_idx,
                column_map[f"{identifier_type}_attempted"],
                f'=LEN({lookup_input_ref})>0',
                None,
                False,
            )
            worksheet.write_formula(
                row_idx,
                column_map[f"{identifier_type}_success"],
                f'=IFERROR({returned_ric_ref}<>"",FALSE)',
                None,
                False,
            )

        for left_type, right_type in identifier_pairs:
            for field_name in ("ric", "isin", "cusip"):
                left_ref = _sheet_cell_ref(row_idx, column_map[f"{left_type}_returned_{field_name}"])
                right_ref = _sheet_cell_ref(row_idx, column_map[f"{right_type}_returned_{field_name}"])
                target_col = column_map[f"{left_type}_vs_{right_type}_same_{field_name}"]
                worksheet.write_formula(
                    row_idx,
                    target_col,
                    (
                        f'=IF(AND(IFERROR({left_ref}<>"",FALSE),IFERROR({right_ref}<>"",FALSE)),'
                        f'IFERROR({left_ref}={right_ref},FALSE),"")'
                    ),
                    None,
                    "",
                )

        success_refs = [
            _sheet_cell_ref(row_idx, column_map[f"{identifier_type}_success"])
            for identifier_type in identifier_types
        ]
        pairwise_refs = [
            _sheet_cell_ref(row_idx, column_map[f"{left_type}_vs_{right_type}_same_{field_name}"])
            for left_type, right_type in identifier_pairs
            for field_name in ("ric", "isin", "cusip")
        ]
        success_count_expr = "+".join(f"N({cell_ref})" for cell_ref in success_refs)
        pairwise_and_terms = ",".join(f'IF({cell_ref}="",TRUE,{cell_ref})' for cell_ref in pairwise_refs)
        worksheet.write_formula(
            row_idx,
            column_map["all_successful_attempts_consistent"],
            f'=IF(({success_count_expr})<2,"",AND({pairwise_and_terms}))',
            None,
            "",
        )


def _prefill_refinitiv_extended_summary_formulas(
    worksheet: xlsxwriter.worksheet.Worksheet,
    *,
    summary_df: pl.DataFrame,
    lookup_sheet_name: str,
    lookup_column_names: tuple[str, ...],
    lookup_row_count: int,
) -> None:
    if summary_df.height <= 0 or lookup_row_count <= 0:
        return

    lookup_columns = {name: idx for idx, name in enumerate(lookup_column_names)}
    value_col_idx = summary_df.columns.index("value")
    identifier_types = ("ISIN", "CUSIP", "TICKER")
    single_success_summary_rows: dict[str, int] = {}

    def _lookup_range(column_name: str) -> str:
        return _sheet_range_ref(lookup_sheet_name, 1, lookup_row_count, lookup_columns[column_name])

    for row_idx, row in enumerate(summary_df.iter_rows(named=True), start=1):
        category = row["summary_category"]
        key = row["summary_key"]
        formula: str | None = None

        if category == "attempt_count_by_identifier_type":
            formula = f'=COUNTIF({_lookup_range(f"{key}_attempted")},TRUE)'
        elif category == "success_count_by_identifier_type":
            formula = f'=COUNTIF({_lookup_range(f"{key}_success")},TRUE)'
        elif category.startswith("agreement_count_same_"):
            field_name = category.removeprefix("agreement_count_same_")
            formula = f'=COUNTIF({_lookup_range(f"{key}_same_{field_name}")},TRUE)'
        elif category.startswith("conflict_count_same_"):
            field_name = category.removeprefix("conflict_count_same_")
            formula = f'=COUNTIF({_lookup_range(f"{key}_same_{field_name}")},FALSE)'
        elif category == "agreement_count_all_successful_attempts_consistent":
            formula = f'=COUNTIF({_lookup_range("all_successful_attempts_consistent")},TRUE)'
        elif category == "conflict_count_all_successful_attempts_consistent":
            formula = f'=COUNTIF({_lookup_range("all_successful_attempts_consistent")},FALSE)'
        elif category == "only_one_identifier_type_succeeds" and key in identifier_types:
            other_types = [identifier_type for identifier_type in identifier_types if identifier_type != key]
            formula = (
                f'=SUMPRODUCT(--({_lookup_range(f"{key}_success")}=TRUE),'
                f'--({_lookup_range(f"{other_types[0]}_success")}<>TRUE),'
                f'--({_lookup_range(f"{other_types[1]}_success")}<>TRUE))'
            )
            single_success_summary_rows[key] = row_idx
        elif category == "only_one_identifier_type_succeeds" and key == "total":
            row_refs = [
                _sheet_cell_ref(single_success_summary_rows[identifier_type], value_col_idx)
                for identifier_type in identifier_types
                if identifier_type in single_success_summary_rows
            ]
            if row_refs:
                formula = f'=SUM({",".join(row_refs)})'

        if formula is not None:
            worksheet.write_formula(row_idx, value_col_idx, formula, None, row["value"])


def _write_readme_sheet(
    workbook: xlsxwriter.Workbook,
    *,
    readme_payload: dict[str, Any],
    header_fmt: xlsxwriter.format.Format,
    wrap_fmt: xlsxwriter.format.Format,
    instructions: tuple[str, ...],
) -> None:
    readme_ws = workbook.add_worksheet("README")
    readme_ws.freeze_panes(1, 0)
    readme_ws.set_column(0, 0, 24)
    readme_ws.set_column(1, 1, 100)
    readme_ws.write(0, 0, "field", header_fmt)
    readme_ws.write(0, 1, "value", header_fmt)

    readme_ws.write(1, 0, "instructions", header_fmt)
    readme_ws.write(1, 1, "\n".join(instructions), wrap_fmt)

    row_idx = 3
    for key, value in readme_payload.items():
        readme_ws.write(row_idx, 0, key, header_fmt)
        if isinstance(value, (dict, list)):
            rendered = json.dumps(value, indent=2, sort_keys=True)
        else:
            rendered = "" if value is None else str(value)
        readme_ws.write(row_idx, 1, rendered, wrap_fmt if "\n" in rendered else None)
        row_idx += 1


def write_refinitiv_bridge_workbook(
    df: pl.DataFrame,
    out_path: Path,
    *,
    readme_payload: dict[str, Any],
    text_columns: tuple[str, ...],
) -> Path:
    """Write the Refinitiv bridge handoff workbook with a fixed two-sheet layout."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    workbook = xlsxwriter.Workbook(
        str(out_path),
        {
            "strings_to_formulas": False,
            "strings_to_numbers": False,
            "strings_to_urls": False,
        },
    )
    try:
        workbook.set_calc_mode("auto")
        header_fmt = workbook.add_format({"bold": True, "bg_color": "#D9E2F3", "border": 1})
        wrap_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})
        _write_table_sheet(
            workbook,
            sheet_name="bridge_universe",
            df=df,
            header_fmt=header_fmt,
            text_columns=text_columns,
        )
        _write_readme_sheet(
            workbook,
            readme_payload=readme_payload,
            header_fmt=header_fmt,
            wrap_fmt=wrap_fmt,
            instructions=(
            "Edit only the vendor_* columns on bridge_universe.",
            "Do not reorder rows, change headers, or overwrite bridge_row_id.",
            "Save the enriched workbook as a new file before re-import.",
            "The authoritative machine-readable artifact is the parquet written alongside this workbook.",
            ),
        )
    finally:
        workbook.close()
    return out_path


def write_refinitiv_ric_lookup_workbook(
    df: pl.DataFrame,
    out_path: Path,
    *,
    readme_payload: dict[str, Any],
    text_columns: tuple[str, ...],
    manual_review_df: pl.DataFrame | None = None,
) -> Path:
    """Write the narrower RIC lookup workbook for Workspace Excel use."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    workbook = xlsxwriter.Workbook(
        str(out_path),
        {
            "strings_to_formulas": False,
            "strings_to_numbers": False,
            "strings_to_urls": False,
        },
    )
    try:
        workbook.set_calc_mode("auto")
        header_fmt = workbook.add_format({"bold": True, "bg_color": "#D9E2F3", "border": 1})
        wrap_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})

        _write_table_sheet(
            workbook,
            sheet_name="ric_lookup",
            df=df,
            header_fmt=header_fmt,
            text_columns=text_columns,
        )
        if manual_review_df is not None and manual_review_df.height > 0:
            _write_table_sheet(
                workbook,
                sheet_name="manual_review",
                df=manual_review_df,
                header_fmt=header_fmt,
                text_columns=text_columns,
            )

        _write_readme_sheet(
            workbook,
            readme_payload=readme_payload,
            header_fmt=header_fmt,
            wrap_fmt=wrap_fmt,
            instructions=(
                "Use ric_lookup for Workspace lookup and edit only vendor_primary_ric, vendor_returned_name, vendor_returned_cusip, vendor_returned_isin, vendor_match_status, and vendor_notes.",
                "Do not reorder rows, change headers, or overwrite bridge_row_id or preferred_lookup_* columns.",
                "Rows without a usable lookup identifier are separated onto manual_review when present.",
                "The authoritative machine-readable artifact remains the parquet written alongside this workbook.",
            ),
        )
    finally:
        workbook.close()
    return out_path


def write_refinitiv_ric_lookup_extended_workbook(
    df: pl.DataFrame,
    out_path: Path,
    *,
    readme_payload: dict[str, Any],
    text_columns: tuple[str, ...],
    summary_df: pl.DataFrame | None = None,
    summary_text_columns: tuple[str, ...] = (),
) -> Path:
    """Write the wide extended RIC lookup diagnostic workbook."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    workbook = xlsxwriter.Workbook(
        str(out_path),
        {
            "strings_to_formulas": False,
            "strings_to_numbers": False,
            "strings_to_urls": False,
        },
    )
    try:
        workbook.set_calc_mode("auto")
        header_fmt = workbook.add_format({"bold": True, "bg_color": "#D9E2F3", "border": 1})
        wrap_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})

        lookup_ws = _write_table_sheet(
            workbook,
            sheet_name="lookup_diagnostics",
            df=df,
            header_fmt=header_fmt,
            text_columns=text_columns,
        )
        _prefill_refinitiv_extended_lookup_formulas(
            lookup_ws,
            column_names=tuple(df.columns),
            row_count=df.height,
        )
        if summary_df is not None:
            summary_ws = _write_table_sheet(
                workbook,
                sheet_name="summary",
                df=summary_df,
                header_fmt=header_fmt,
                text_columns=summary_text_columns,
            )
            _prefill_refinitiv_extended_summary_formulas(
                summary_ws,
                summary_df=summary_df,
                lookup_sheet_name="lookup_diagnostics",
                lookup_column_names=tuple(df.columns),
                lookup_row_count=df.height,
            )

        _write_readme_sheet(
            workbook,
            readme_payload=readme_payload,
            header_fmt=header_fmt,
            wrap_fmt=wrap_fmt,
            instructions=(
                "lookup_diagnostics is the main manual comparison workbook for the sample Refinitiv diagnostic workflow.",
                "Open the workbook in Excel with Workspace enabled and let the prefilled _xll.TR formulas evaluate before reviewing results.",
                "Do not reorder rows, change headers, or overwrite bridge_row_id; save the evaluated workbook as a new file before any re-import step.",
                "preferred_lookup_id and preferred_lookup_type are intentionally omitted here; Python-side final accepted RIC derivation remains separate from this manual comparison workbook.",
            ),
        )
    finally:
        workbook.close()
    return out_path


def write_refinitiv_null_ric_diagnostics_workbook(
    df: pl.DataFrame,
    out_path: Path,
    *,
    readme_payload: dict[str, Any],
    text_columns: tuple[str, ...],
    sheet_name: str = "failed_lookup_review",
    instructions: tuple[str, ...] | None = None,
) -> Path:
    """Write a review workbook for failed/null-RIC diagnostic rows."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    workbook = xlsxwriter.Workbook(
        str(out_path),
        {
            "strings_to_formulas": False,
            "strings_to_numbers": False,
            "strings_to_urls": False,
        },
    )
    try:
        workbook.set_calc_mode("auto")
        header_fmt = workbook.add_format({"bold": True, "bg_color": "#D9E2F3", "border": 1})
        wrap_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})
        _write_table_sheet(
            workbook,
            sheet_name=sheet_name,
            df=df,
            header_fmt=header_fmt,
            text_columns=text_columns,
        )
        _write_readme_sheet(
            workbook,
            readme_payload=readme_payload,
            header_fmt=header_fmt,
            wrap_fmt=wrap_fmt,
            instructions=instructions
            or (
                "Review only failed lookup rows on failed_lookup_review.",
                "Do not overwrite the original filled lookup workbook; this workbook is a diagnostic derivative.",
                "Diagnostic flags and candidate fields are evidence only and do not propagate results automatically.",
                "The authoritative machine-readable artifacts for this step are the parquet/json files written alongside this workbook.",
            ),
        )
    finally:
        workbook.close()
    return out_path


def write_refinitiv_ownership_smoke_testing_workbook(
    df: pl.DataFrame,
    out_path: Path,
    *,
    readme_payload: dict[str, Any],
) -> Path:
    """Write a block-layout ownership smoke-test workbook for manual Excel checks."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    workbook = xlsxwriter.Workbook(
        str(out_path),
        {
            "strings_to_formulas": False,
            "strings_to_numbers": False,
            "strings_to_urls": False,
        },
    )
    try:
        workbook.set_calc_mode("auto")
        header_fmt = workbook.add_format({"bold": True, "bg_color": "#D9E2F3", "border": 1})
        wrap_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})

        worksheet = workbook.add_worksheet("ownership_smoke")
        worksheet.freeze_panes(1, 0)

        block_headers = (
            "input_data",
            "returned_ric",
            "returned_date",
            "returned_value",
            "returned_category",
        )
        input_field_order = (
            "bridge_row_id",
            "lookup_input",
            "request_start_date",
            "request_end_date",
            "sample_category",
        )

        for block_idx, row in enumerate(df.iter_rows(named=True)):
            base_col = block_idx * 5
            for offset, header in enumerate(block_headers):
                worksheet.write(0, base_col + offset, header, header_fmt)

            for row_offset, field_name in enumerate(input_field_order, start=1):
                value = row.get(field_name)
                worksheet.write_string(
                    row_offset,
                    base_col,
                    "" if value is None else str(value),
                )

            worksheet.set_column(base_col, base_col, 28)
            worksheet.set_column(base_col + 1, base_col + 3, 18)
            worksheet.set_column(base_col + 4, base_col + 4, 36)

        _write_readme_sheet(
            workbook,
            readme_payload=readme_payload,
            header_fmt=header_fmt,
            wrap_fmt=wrap_fmt,
            instructions=(
                "Each sampled instrument uses one 5-column block on ownership_smoke.",
                "Rows 2-6 contain the prefilled manual lookup inputs; leave block order and headers unchanged.",
                "Enter Workspace ownership formulas manually and let any returned rows spill below row 6.",
                "This workbook is a smoke-test artifact only and does not overwrite lookup results or propagate rescues.",
            ),
        )
    finally:
        workbook.close()
    return out_path

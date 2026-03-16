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


def _write_refinitiv_resolution_diagnostic_retrieval_sheet(
    workbook: xlsxwriter.Workbook,
    *,
    df: pl.DataFrame,
    header_fmt: xlsxwriter.format.Format,
) -> None:
    worksheet = workbook.add_worksheet("retrieval_handoff")
    worksheet.freeze_panes(1, 0)

    block_headers = (
        "input_data",
        "returned_ric",
        "returned_date",
        "returned_value",
        "returned_category",
    )
    input_field_order = (
        "diagnostic_case_id",
        "target_class",
        "retrieval_role",
        "diagnostic_role",
        "lookup_input_source",
        "case_target_bridge_row_id",
        "bridge_row_id",
        "lookup_input",
        "request_start_date",
        "request_end_date",
        "KYPERMNO",
        "TICKER",
        "CUSIP",
        "ISIN",
    )

    if df.height <= 0:
        for offset, header in enumerate(block_headers):
            worksheet.write(0, offset, header, header_fmt)
        worksheet.write_string(1, 0, "No retrieval rows")
        worksheet.set_column(0, 0, 28)
        worksheet.set_column(1, 3, 18)
        worksheet.set_column(4, 4, 36)
        return

    lookup_input_row = input_field_order.index("lookup_input") + 1
    request_start_row = input_field_order.index("request_start_date") + 1
    request_end_row = input_field_order.index("request_end_date") + 1

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

        lookup_input_ref = _sheet_cell_ref(lookup_input_row, base_col)
        request_start_ref = _sheet_cell_ref(request_start_row, base_col)
        request_end_ref = _sheet_cell_ref(request_end_row, base_col)
        worksheet.write_formula(
            1,
            base_col + 1,
            (
                f'=IF({lookup_input_ref}="","",'
                f'_xll.RDP.Data({lookup_input_ref},'
                '"TR.CategoryOwnershipPct.Date;TR.CategoryOwnershipPct;TR.InstrStatTypeValue",'
                f'"StatType=7 SDate="&{request_start_ref}&" EDate="&{request_end_ref}&" CH=Fd RH=IN"))'
            ),
            None,
            "",
        )

        worksheet.set_column(base_col, base_col, 32)
        worksheet.set_column(base_col + 1, base_col + 3, 18)
        worksheet.set_column(base_col + 4, base_col + 4, 36)


def write_refinitiv_resolution_diagnostic_workbook(
    context_df: pl.DataFrame,
    targets_df: pl.DataFrame,
    handoff_df: pl.DataFrame,
    out_path: Path,
    *,
    readme_payload: dict[str, Any],
    context_text_columns: tuple[str, ...],
    target_text_columns: tuple[str, ...],
) -> Path:
    """Write the resolution diagnostic workbook with context tables and retrieval blocks."""
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
            sheet_name="diagnostic_context",
            df=context_df,
            header_fmt=header_fmt,
            text_columns=context_text_columns,
        )
        _write_table_sheet(
            workbook,
            sheet_name="targets_only",
            df=targets_df,
            header_fmt=header_fmt,
            text_columns=target_text_columns,
        )
        _write_refinitiv_resolution_diagnostic_retrieval_sheet(
            workbook,
            df=handoff_df,
            header_fmt=header_fmt,
        )
        _write_readme_sheet(
            workbook,
            readme_payload=readme_payload,
            header_fmt=header_fmt,
            wrap_fmt=wrap_fmt,
            instructions=(
                "diagnostic_context is the main context sheet; review each TARGET row together with its PREVIOUS and NEXT rows inside the same diagnostic_case_id.",
                "retrieval_handoff contains one 5-column Workspace retrieval block per case-specific lookup input, with formulas prefilled directly into the sheet.",
                "Each retrieval block uses fixed input_data row order: diagnostic_case_id, target_class, retrieval_role, diagnostic_role, lookup_input_source, case_target_bridge_row_id, bridge_row_id, lookup_input, request_start_date, request_end_date, KYPERMNO, TICKER, CUSIP, ISIN.",
                "Open the workbook in Excel with Workspace enabled and let the prefilled _xll.RDP.Data formulas evaluate before reviewing local continuity.",
                "This workbook is diagnostic only and does not overwrite or modify the production resolution artifact.",
            ),
        )
    finally:
        workbook.close()
    return out_path


def _write_refinitiv_ownership_validation_sheet(
    workbook: xlsxwriter.Workbook,
    *,
    df: pl.DataFrame,
    header_fmt: xlsxwriter.format.Format,
    label_fmt: xlsxwriter.format.Format,
    input_field_order: tuple[str, ...],
    block_slot_roles: tuple[str, ...],
) -> None:
    block_headers = (
        "input_data",
        "returned_ric",
        "returned_date",
        "returned_value",
        "returned_category",
    )
    role_to_row = {
        (str(row["diagnostic_case_id"]), str(row["block_slot_role"])): row
        for row in df.iter_rows(named=True)
    }

    if df.height <= 0:
        worksheet = workbook.add_worksheet("ownership_validation_001")
        worksheet.freeze_panes(1, 1)
        worksheet.write(0, 0, "field", header_fmt)
        for offset, header in enumerate(block_headers, start=1):
            worksheet.write(0, offset, header, header_fmt)
        worksheet.write_string(1, 0, "No retrieval rows")
        worksheet.set_column(0, 0, 28)
        worksheet.set_column(1, 1, 28)
        worksheet.set_column(2, 4, 18)
        worksheet.set_column(5, 5, 36)
        return

    case_rows = (
        df.select(
            "sheet_name",
            "sheet_case_index",
            "case_band_row_start",
            "diagnostic_case_id",
        )
        .unique()
        .sort("sheet_name", "sheet_case_index")
        .to_dicts()
    )
    cases_by_sheet: dict[str, list[dict[str, Any]]] = {}
    for row in case_rows:
        cases_by_sheet.setdefault(str(row["sheet_name"]), []).append(row)

    lookup_input_offset = input_field_order.index("lookup_input") + 1
    request_start_offset = input_field_order.index("request_start_date") + 1
    request_end_offset = input_field_order.index("request_end_date") + 1

    for sheet_name, sheet_cases in cases_by_sheet.items():
        worksheet = workbook.add_worksheet(sheet_name)
        worksheet.freeze_panes(1, 1)
        worksheet.set_column(0, 0, 34)
        for slot_idx in range(len(block_slot_roles)):
            base_col = 1 + (slot_idx * 5)
            worksheet.set_column(base_col, base_col, 28)
            worksheet.set_column(base_col + 1, base_col + 3, 18)
            worksheet.set_column(base_col + 4, base_col + 4, 36)

        for case_row in sheet_cases:
            case_id = str(case_row["diagnostic_case_id"])
            start_row = int(case_row["case_band_row_start"]) - 1
            worksheet.write(start_row, 0, "field", header_fmt)
            for slot_idx, _slot_role in enumerate(block_slot_roles):
                base_col = 1 + (slot_idx * 5)
                for offset, header in enumerate(block_headers):
                    worksheet.write(start_row, base_col + offset, header, header_fmt)
            for field_offset, field_name in enumerate(input_field_order, start=1):
                worksheet.write_string(start_row + field_offset, 0, field_name, label_fmt)

            for slot_idx, slot_role in enumerate(block_slot_roles):
                row = role_to_row.get((case_id, slot_role))
                if row is None:
                    continue
                base_col = 1 + (slot_idx * 5)
                for field_offset, field_name in enumerate(input_field_order, start=1):
                    value = row.get(field_name)
                    worksheet.write_string(start_row + field_offset, base_col, "" if value is None else str(value))

                lookup_input_ref = _sheet_cell_ref(start_row + lookup_input_offset, base_col)
                request_start_ref = _sheet_cell_ref(start_row + request_start_offset, base_col)
                request_end_ref = _sheet_cell_ref(start_row + request_end_offset, base_col)
                worksheet.write_formula(
                    start_row + 1,
                    base_col + 1,
                    (
                        f'=@RDP.Data({lookup_input_ref},'
                        '"TR.CategoryOwnershipPct.Date;TR.CategoryOwnershipPct;TR.InstrStatTypeValue",'
                        f'"StatType=7 SDate="&{request_start_ref}&" EDate="&{request_end_ref}&" CH=Fd RH=IN")'
                    ),
                    None,
                    "",
                )


def write_refinitiv_ownership_validation_workbook(
    df: pl.DataFrame,
    out_path: Path,
    *,
    readme_payload: dict[str, Any],
    input_field_order: tuple[str, ...],
    block_slot_roles: tuple[str, ...],
) -> Path:
    """Write the case-banded ownership validation workbook for manual Workspace retrieval."""
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
        label_fmt = workbook.add_format({"bold": True, "bg_color": "#F3F6FA"})
        wrap_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})
        _write_refinitiv_ownership_validation_sheet(
            workbook,
            df=df,
            header_fmt=header_fmt,
            label_fmt=label_fmt,
            input_field_order=input_field_order,
            block_slot_roles=block_slot_roles,
        )
        _write_readme_sheet(
            workbook,
            readme_payload=readme_payload,
            header_fmt=header_fmt,
            wrap_fmt=wrap_fmt,
            instructions=(
                "This workbook is diagnostic only and is driven from refinitiv_ric_resolution_diagnostic_handoff.csv rather than the original bridge universe.",
                "Each ownership_validation_* sheet is case-banded: keep target candidate and adjacent effective blocks together and let the prefilled @RDP.Data formulas evaluate in Excel with Workspace enabled.",
                "Do not reorder sheets, case bands, labels, or block columns; save the evaluated workbook as refinitiv_ownership_validation_handoff_common_stock_filled_in.xlsx before any re-import step.",
                "The machine-readable handoff snapshot is the parquet/csv written alongside this workbook, and later result parsing expects the same fixed workbook geometry.",
            ),
        )
    finally:
        workbook.close()
    return out_path


def _write_refinitiv_ownership_universe_sheet(
    workbook: xlsxwriter.Workbook,
    *,
    df: pl.DataFrame,
    header_fmt: xlsxwriter.format.Format,
    input_field_order: tuple[str, ...],
    block_headers: tuple[str, ...],
) -> None:
    eligible_df = df.filter(pl.col("retrieval_eligible").fill_null(False))
    worksheet = workbook.add_worksheet("ownership_retrieval")
    worksheet.freeze_panes(1, 0)

    if eligible_df.height <= 0:
        for offset, header in enumerate(block_headers):
            worksheet.write(0, offset, header, header_fmt)
        worksheet.write_string(1, 0, "No retrieval rows")
        worksheet.set_column(0, 0, 34)
        worksheet.set_column(1, 3, 18)
        worksheet.set_column(4, 4, 36)
        return

    candidate_ric_offset = input_field_order.index("candidate_ric") + 1
    request_start_offset = input_field_order.index("request_start_date") + 1
    request_end_offset = input_field_order.index("request_end_date") + 1
    block_width = len(block_headers)

    for block_index, row in enumerate(eligible_df.iter_rows(named=True)):
        base_col = block_index * block_width
        worksheet.set_column(base_col, base_col, 34)
        worksheet.set_column(base_col + 1, base_col + 3, 18)
        worksheet.set_column(base_col + 4, base_col + 4, 36)

        for offset, header in enumerate(block_headers):
            worksheet.write(0, base_col + offset, header, header_fmt)

        for field_offset, field_name in enumerate(input_field_order, start=1):
            value = row.get(field_name)
            worksheet.write_string(field_offset, base_col, "" if value is None else str(value))

        candidate_ric_ref = _sheet_cell_ref(candidate_ric_offset, base_col)
        request_start_ref = _sheet_cell_ref(request_start_offset, base_col)
        request_end_ref = _sheet_cell_ref(request_end_offset, base_col)
        worksheet.write_formula(
            1,
            base_col + 1,
            (
                f'=@RDP.Data({candidate_ric_ref},'
                '"TR.CategoryOwnershipPct.Date;TR.CategoryOwnershipPct;TR.InstrStatTypeValue",'
                f'"StatType=7 SDate="&{request_start_ref}&" EDate="&{request_end_ref}&" CH=Fd RH=IN")'
            ),
            None,
            "",
        )


def write_refinitiv_ownership_universe_workbook(
    df: pl.DataFrame,
    out_path: Path,
    *,
    readme_payload: dict[str, Any],
    input_field_order: tuple[str, ...],
    block_headers: tuple[str, ...],
) -> Path:
    """Write the full-universe ownership retrieval workbook as repeated 5-column request blocks."""
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
        _write_refinitiv_ownership_universe_sheet(
            workbook,
            df=df,
            header_fmt=header_fmt,
            input_field_order=input_field_order,
            block_headers=block_headers,
        )
        _write_readme_sheet(
            workbook,
            readme_payload=readme_payload,
            header_fmt=header_fmt,
            wrap_fmt=wrap_fmt,
            instructions=(
                "ownership_retrieval is the only retrieval sheet. Each request is a self-contained 5-column block: input_data, returned_ric, returned_date, returned_value, returned_category.",
                "Within each request block, input_data rows 2-11 are fixed: bridge_row_id, candidate_ric, request_start_date, request_end_date, candidate_slot, lookup_input_source, effective_collection_ric, effective_collection_ric_source, accepted_ric, accepted_ric_source.",
                "The direct @RDP.Data formula is prefilled in the returned_ric cell on row 2 of each block. Open the workbook in Excel with Workspace enabled, let the formulas evaluate, and save the filled workbook before re-import.",
                "The parser reads ownership_retrieval in 5-column steps, skips the spill header row, and reconstructs long-format observations by matching bridge_row_id, candidate_slot, and candidate_ric back to the handoff snapshot.",
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

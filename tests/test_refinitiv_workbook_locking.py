from __future__ import annotations

from pathlib import Path

import polars as pl
from xlsxwriter.exceptions import FileCreateError

from thesis_pkg.pipelines.refinitiv_bridge_pipeline import _write_workbook_or_reuse_locked_output


def test_write_workbook_or_reuse_locked_output_reuses_existing_file(tmp_path: Path) -> None:
    output_path = tmp_path / "locked.xlsx"
    output_path.write_text("existing workbook placeholder", encoding="utf-8")

    def _locked_writer(df: pl.DataFrame, out_path: Path, **_: object) -> Path:
        raise FileCreateError(PermissionError(13, "Permission denied", str(out_path)))

    df = pl.DataFrame({"x": [1]})
    result = _write_workbook_or_reuse_locked_output(_locked_writer, df, output_path)

    assert result == output_path
    assert output_path.exists()


def test_write_workbook_or_reuse_locked_output_raises_when_file_missing(tmp_path: Path) -> None:
    output_path = tmp_path / "missing_locked.xlsx"

    def _locked_writer(df: pl.DataFrame, out_path: Path, **_: object) -> Path:
        raise FileCreateError(PermissionError(13, "Permission denied", str(out_path)))

    df = pl.DataFrame({"x": [1]})
    try:
        _write_workbook_or_reuse_locked_output(_locked_writer, df, output_path)
    except FileCreateError:
        pass
    else:
        raise AssertionError("expected FileCreateError when locked output does not already exist")

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from thesis_pkg.core.ccm.canonical_links import CikHistoryWindowPolicy
from thesis_pkg.pipeline import build_or_reuse_ccm_daily_stage


def _empty_linkfiscalperiodall() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "KYGVKEY": pl.Utf8,
            "lpermno": pl.Int32,
            "lpermco": pl.Int32,
            "liid": pl.Utf8,
            "linktype": pl.Utf8,
            "linkprim": pl.Utf8,
            "linkrank": pl.Int32,
            "linkdt": pl.Date,
            "linkenddt": pl.Date,
            "FiscalPeriodCRSPStartDt": pl.Date,
            "FiscalPeriodCRSPEndDt": pl.Date,
        }
    )


def _write_required_ccm_tables(base_dir: Path, *, include_linkfiscal: bool = True) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    tables: dict[str, pl.DataFrame] = {
        "filingdates": pl.DataFrame(
            {
                "LPERMNO": [1],
                "SRCTYPE": ["10-K"],
                "FILEDATE": [dt.date(2024, 1, 2)],
                "FILEDATETIME": ["08:30:00"],
            }
        ),
        "linkhistory": pl.DataFrame(
            {
                "KYGVKEY": ["1000"],
                "LPERMNO": [1],
                "LPERMCO": [10],
                "LIID": ["A"],
                "LINKTYPE": ["LC"],
                "LINKPRIM": ["P"],
                "LINKDT": [dt.date(2020, 1, 1)],
                "LINKENDDT": [None],
            }
        ),
        "companydescription": pl.DataFrame(
            {
                "KYGVKEY": ["1000"],
                "CIK": ["123456789"],
            }
        ),
        "companyhistory": pl.DataFrame(
            {
                "KYGVKEY": ["1000"],
                "HCHGDT": [dt.date(2020, 1, 1)],
                "HCHGENDDT": [None],
                "HCIK": ["123456789"],
                "HSIC": [100],
                "HNAICS": ["1111"],
                "HGSUBIND": ["Sub"],
            }
        ),
        "securityheaderhistory": pl.DataFrame(
            {
                "KYGVKEY": ["1000"],
                "KYIID": ["A"],
                "HSCHGDT": [dt.date(2020, 1, 1)],
                "HSCHGENDDT": [None],
                "HTPCI": ["USA"],
                "HEXCNTRY": ["US"],
            }
        ),
        "sfz_ds_dly": pl.DataFrame(
            {
                "KYPERMNO": [1],
                "CALDT": [dt.date(2024, 1, 2)],
                "BIDLO": [9.0],
                "ASKHI": [11.0],
            }
        ),
        "sfz_dp_dly": pl.DataFrame(
            {
                "KYPERMNO": [1],
                "CALDT": [dt.date(2024, 1, 2)],
                "PRC": [-10.0],
                "RET": [0.01],
                "RETX": [0.01],
                "TCAP": [100_000_000.0],
                "VOL": [1_000.0],
            }
        ),
        "sfz_del": pl.DataFrame(
            schema={
                "KYPERMNO": pl.Int32,
                "DLSTDT": pl.Date,
                "DLSTCD": pl.Int32,
                "DLPRC": pl.Float64,
                "DLAMT": pl.Float64,
                "DLRET": pl.Float64,
                "DLRETX": pl.Float64,
            }
        ),
        "sfz_nam": pl.DataFrame(
            {
                "KYPERMNO": [1],
                "NAMEDT": [dt.date(2020, 1, 1)],
                "NAMEENDDT": [dt.date(2025, 12, 31)],
                "SHRCD": [10],
                "EXCHCD": [1],
                "PRIMEXCH": ["N"],
                "TRDSTAT": ["A"],
                "SECSTAT": ["A"],
            }
        ),
        "sfz_hdr": pl.DataFrame(
            {
                "KYPERMNO": [1],
                "BEGDT": [dt.date(2020, 1, 1)],
                "ENDDT": [dt.date(2025, 12, 31)],
                "HSHRCD": [10],
                "HEXCD": [1],
                "HPRIMEXCH": ["N"],
                "HTRDSTAT": ["A"],
                "HSECSTAT": ["A"],
            }
        ),
    }
    if include_linkfiscal:
        tables["linkfiscalperiodall"] = _empty_linkfiscalperiodall()

    for name, table in tables.items():
        table.write_parquet(base_dir / f"{name}.parquet", compression="zstd")


def test_build_or_reuse_ccm_daily_stage_rebuild_writes_daily_and_canonical(tmp_path: Path) -> None:
    base_dir = tmp_path / "base"
    derived_dir = tmp_path / "derived"
    reuse_daily_path = tmp_path / "reuse" / "unused.parquet"
    _write_required_ccm_tables(base_dir, include_linkfiscal=True)

    out = build_or_reuse_ccm_daily_stage(
        run_mode="REBUILD",
        ccm_base_dir=base_dir,
        ccm_derived_dir=derived_dir,
        ccm_reuse_daily_path=reuse_daily_path,
        forms_10k_10q=["10-K"],
    )

    assert set(out) == {"ccm_daily_path", "canonical_link_path"}
    assert out["ccm_daily_path"] == derived_dir / "final_flagged_data_compdesc_added.parquet"
    assert out["canonical_link_path"] == derived_dir / "canonical_link_table.parquet"
    assert out["ccm_daily_path"].exists()
    assert out["canonical_link_path"].exists()
    assert pl.scan_parquet(out["canonical_link_path"]).select(pl.len()).collect().item() > 0


def test_build_or_reuse_ccm_daily_stage_rebuild_raises_when_linkfiscalperiodall_missing(tmp_path: Path) -> None:
    base_dir = tmp_path / "base_missing_linkfiscal"
    derived_dir = tmp_path / "derived_missing_linkfiscal"
    reuse_daily_path = tmp_path / "reuse" / "unused.parquet"
    _write_required_ccm_tables(base_dir, include_linkfiscal=False)

    with pytest.raises(ValueError, match="linkfiscalperiodall"):
        build_or_reuse_ccm_daily_stage(
            run_mode="REBUILD",
            ccm_base_dir=base_dir,
            ccm_derived_dir=derived_dir,
            ccm_reuse_daily_path=reuse_daily_path,
            forms_10k_10q=["10-K"],
        )


def test_build_or_reuse_ccm_daily_stage_reuse_succeeds_when_artifacts_exist(tmp_path: Path) -> None:
    derived_dir = tmp_path / "derived_reuse_ok"
    derived_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = derived_dir / "canonical_link_table.parquet"
    reuse_daily_path = tmp_path / "reuse" / "final_flagged_data_compdesc_added.parquet"
    reuse_daily_path.parent.mkdir(parents=True, exist_ok=True)

    pl.DataFrame({"x": [1]}).write_parquet(canonical_path)
    pl.DataFrame({"y": [1]}).write_parquet(reuse_daily_path)

    out = build_or_reuse_ccm_daily_stage(
        run_mode="REUSE",
        ccm_base_dir=tmp_path / "unused_base",
        ccm_derived_dir=derived_dir,
        ccm_reuse_daily_path=reuse_daily_path,
        forms_10k_10q=["10-K"],
    )

    assert out["ccm_daily_path"] == reuse_daily_path
    assert out["canonical_link_path"] == canonical_path


def test_build_or_reuse_ccm_daily_stage_reuse_raises_when_canonical_missing(tmp_path: Path) -> None:
    derived_dir = tmp_path / "derived_reuse_missing"
    derived_dir.mkdir(parents=True, exist_ok=True)
    reuse_daily_path = tmp_path / "reuse" / "final_flagged_data_compdesc_added.parquet"
    reuse_daily_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"y": [1]}).write_parquet(reuse_daily_path)

    with pytest.raises(FileNotFoundError, match="CCM canonical link parquet not found"):
        build_or_reuse_ccm_daily_stage(
            run_mode="REUSE",
            ccm_base_dir=tmp_path / "unused_base",
            ccm_derived_dir=derived_dir,
            ccm_reuse_daily_path=reuse_daily_path,
            forms_10k_10q=["10-K"],
        )


def test_build_or_reuse_ccm_daily_stage_policy_pass_through_controls_cik_start(tmp_path: Path) -> None:
    base_dir = tmp_path / "base_policy"
    _write_required_ccm_tables(base_dir, include_linkfiscal=True)

    default_out = build_or_reuse_ccm_daily_stage(
        run_mode="REBUILD",
        ccm_base_dir=base_dir,
        ccm_derived_dir=tmp_path / "derived_default",
        ccm_reuse_daily_path=tmp_path / "reuse" / "unused_default.parquet",
        forms_10k_10q=["10-K"],
    )
    strict_out = build_or_reuse_ccm_daily_stage(
        run_mode="REBUILD",
        ccm_base_dir=base_dir,
        ccm_derived_dir=tmp_path / "derived_strict",
        ccm_reuse_daily_path=tmp_path / "reuse" / "unused_strict.parquet",
        forms_10k_10q=["10-K"],
        cik_history_window_policy=CikHistoryWindowPolicy.HISTORY_STRICT,
    )

    default_canonical = pl.read_parquet(default_out["canonical_link_path"])
    strict_canonical = pl.read_parquet(strict_out["canonical_link_path"])

    default_row = default_canonical.filter(pl.col("gvkey") == "1000").row(0, named=True)
    strict_row = strict_canonical.filter(pl.col("gvkey") == "1000").row(0, named=True)

    assert default_row["cik_start"] is None
    assert strict_row["cik_start"] == dt.date(2020, 1, 1)

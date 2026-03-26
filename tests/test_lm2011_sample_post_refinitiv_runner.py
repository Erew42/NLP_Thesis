from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

import polars as pl
import pytest

from thesis_pkg.notebooks_and_scripts import lm2011_sample_post_refinitiv_runner as runner


def _write_parquet(path: Path, df: pl.DataFrame | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df is None:
        df = pl.DataFrame({"placeholder": [1]})
    df.write_parquet(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_temp_layout(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    sample_root = tmp_path / "sample"
    upstream_run_root = tmp_path / "upstream"
    additional_data_dir = tmp_path / "additional"
    output_dir = tmp_path / "output"

    _write_parquet(sample_root / "year_merged" / "1995.parquet")
    _write_parquet(sample_root / "derived_data" / "final_flagged_data_compdesc_added.sample_5pct_seed42.parquet")

    ccm_base_dir = sample_root / "ccm_parquet_data" / "documents-export-2025-3-19"
    for name in (
        "filingdates.parquet",
        "balancesheetquarterly.parquet",
        "incomestatementquarterly.parquet",
        "perioddescriptorquarterly.parquet",
        "balancesheetindustrialannual.parquet",
        "incomestatementindustrialannual.parquet",
        "perioddescriptorannual.parquet",
        "fiscalmarketdataannual.parquet",
        "companyhistory.parquet",
        "companydescription.parquet",
    ):
        _write_parquet(ccm_base_dir / name)

    _write_parquet(upstream_run_root / "sec_ccm_premerge" / "sec_ccm_matched_clean.parquet")
    _write_parquet(upstream_run_root / "items_analysis" / "1995.parquet")
    _write_parquet(upstream_run_root / "refinitiv_doc_ownership_lm2011" / "refinitiv_lm2011_doc_ownership.parquet")
    _write_parquet(
        upstream_run_root / "refinitiv_doc_analyst_lm2011" / "refinitiv_doc_analyst_selected.parquet",
        pl.DataFrame(
            {
                "gvkey_int": [1],
                "matched_announcement_date": [dt.date(2000, 1, 10)],
                "matched_fiscal_period_end": [dt.date(1999, 12, 31)],
                "actual_eps": [1.0],
                "forecast_consensus_mean": [0.8],
                "forecast_dispersion": [0.1],
                "forecast_revision_4m": [0.2],
                "forecast_revision_1m": [0.1],
                "analyst_match_status": ["MATCHED"],
            }
        ),
    )

    for name in ("Fin-Neg.txt", "Fin-Pos.txt", "Fin-Unc.txt", "Fin-Lit.txt", "MW-Strong.txt", "MW-Weak.txt"):
        _write_text(additional_data_dir / name, "token\n")
    _write_text(additional_data_dir / "Harvard_IV_NEG_Inf.txt", "harvard\n")
    _write_text(additional_data_dir / "FF_Siccodes_48_Industries.txt", "1 Agric Agriculture\n0100-0199 Range\n")
    _write_text(
        additional_data_dir / "F-F_Research_Data_Factors_daily.csv",
        "\n".join(
            [
                "line1",
                "line2",
                "line3",
                "line4",
                ",Mkt-RF,SMB,HML,RF",
                "19260701,    0.09,   -0.25,   -0.27,    0.01",
            ]
        ),
    )
    return sample_root, upstream_run_root, additional_data_dir, output_dir


def test_resolve_ccm_parquet_artifact_supports_nested_documents_export_layout(tmp_path: Path) -> None:
    base_dir = tmp_path / "ccm"
    nested_dir = base_dir / "documents-export-2025-3-19"
    nested_dir.mkdir(parents=True)
    target = nested_dir / "filingdates.parquet"
    target.write_text("placeholder", encoding="utf-8")

    assert runner._resolve_ccm_parquet_artifact(base_dir, "filingdates.parquet") == target


def test_filter_valid_annual_period_descriptor_rows_excludes_invalid_fallback_rows() -> None:
    lf = pl.DataFrame(
        {
            "KYGVKEY": [1, 2, 3],
            "KEYSET": ["STD", "STD", "STD"],
            "FYYYY": [2000, 2001, 2002],
            "fyra": [12, 12, 12],
            "FYEAR": [0, 2001, 0],
            "FYR": [12, 12, 0],
            "APDEDATE": [None, None, None],
            "FDATE": [None, None, None],
            "PDATE": [None, None, None],
        }
    ).lazy()

    out = runner._filter_valid_annual_period_descriptor_rows(lf).collect()

    assert out.get_column("KYGVKEY").to_list() == [2]


def test_load_ff_factors_daily_lf_uses_skip_rows_5_and_ignores_footer(tmp_path: Path) -> None:
    csv_path = tmp_path / "F-F_Research_Data_Factors_daily.csv"
    csv_path.write_text(
        "\n".join(
            [
                "preamble1",
                "preamble2",
                "preamble3",
                "preamble4",
                ",Mkt-RF,SMB,HML,RF",
                "19260701,    0.09,   -0.25,   -0.27,    0.01",
                "19260702,    0.45,   -0.33,   -0.06,    0.01",
                "Annual Factors: January-December",
            ]
        ),
        encoding="utf-8",
    )

    out = runner._load_ff_factors_daily_lf(csv_path).collect().sort("trading_date")

    assert out.height == 2
    assert out.row(0, named=True)["trading_date"].isoformat() == "1926-07-01"
    assert out.row(1, named=True)["mkt_rf"] == 0.45


def test_prepare_doc_analyst_sue_input_filters_matched_and_deduplicates(tmp_path: Path) -> None:
    selected_path = tmp_path / "selected.parquet"
    _write_parquet(
        selected_path,
        pl.DataFrame(
            {
                "gvkey_int": [1, 1, 1, 2],
                "matched_announcement_date": [
                    dt.date(2000, 1, 10),
                    dt.date(2000, 1, 10),
                    dt.date(2000, 1, 15),
                    dt.date(2001, 1, 10),
                ],
                "matched_fiscal_period_end": [
                    dt.date(1999, 12, 31),
                    dt.date(1999, 12, 31),
                    dt.date(1999, 12, 31),
                    dt.date(2000, 12, 31),
                ],
                "actual_eps": [1.0, 1.1, 1.2, 2.0],
                "forecast_consensus_mean": [0.8, 0.9, 1.0, 1.8],
                "forecast_dispersion": [0.1, 0.1, 0.2, 0.3],
                "forecast_revision_4m": [0.2, 0.2, 0.3, 0.4],
                "forecast_revision_1m": [0.1, 0.1, 0.2, 0.3],
                "analyst_match_status": ["MATCHED", "MATCHED", "NO_MATCH", "MATCHED"],
            }
        ),
    )

    out = runner._prepare_doc_analyst_sue_input_lf(selected_path).collect().sort("gvkey_int", "announcement_date")

    assert out.height == 2
    assert out.columns == [
        "gvkey_int",
        "announcement_date",
        "fiscal_period_end",
        "actual_eps",
        "forecast_consensus_mean",
        "forecast_dispersion",
        "forecast_revision_4m",
        "forecast_revision_1m",
    ]
    assert out.get_column("gvkey_int").to_list() == [1, 2]


def test_main_writes_expected_artifacts_and_manifest_for_stubbed_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_root, upstream_run_root, additional_data_dir, output_dir = _build_temp_layout(tmp_path)

    monkeypatch.setattr(
        runner,
        "_prepare_annual_accounting_inputs",
        lambda *_: (
            pl.DataFrame({"x": [1]}).lazy(),
            pl.DataFrame({"x": [1]}).lazy(),
            pl.DataFrame({"x": [1]}).lazy(),
            pl.DataFrame({"x": [1]}).lazy(),
        ),
    )
    monkeypatch.setattr(runner, "build_lm2011_sample_backbone", lambda *_, **__: pl.DataFrame({"doc_id": ["d1"]}).lazy())
    monkeypatch.setattr(runner, "build_annual_accounting_panel", lambda *_, **__: pl.DataFrame({"gvkey_int": [1]}).lazy())
    monkeypatch.setattr(runner, "build_quarterly_accounting_panel", lambda *_, **__: pl.DataFrame({"gvkey_int": [1]}).lazy())
    monkeypatch.setattr(
        runner,
        "build_lm2011_text_features_full_10k",
        lambda *_, **__: pl.DataFrame({"doc_id": ["d1"], "token_count_full_10k": [2500]}).lazy(),
    )
    monkeypatch.setattr(
        runner,
        "build_lm2011_text_features_mda",
        lambda *_, **__: pl.DataFrame({"doc_id": ["d1"], "token_count_mda": [300]}).lazy(),
    )
    monkeypatch.setattr(runner, "build_lm2011_event_panel", lambda *_, **__: pl.DataFrame({"doc_id": ["d1"]}).lazy())
    monkeypatch.setattr(runner, "build_lm2011_sue_panel", lambda *_, **__: pl.DataFrame({"doc_id": ["d1"]}).lazy())
    monkeypatch.setattr(
        runner,
        "build_lm2011_return_regression_panel",
        lambda *_, **__: pl.DataFrame({"doc_id": ["d1"], "filing_date": [dt.date(2000, 1, 1)]}).lazy(),
    )
    monkeypatch.setattr(
        runner,
        "build_lm2011_sue_regression_panel",
        lambda *_, **__: pl.DataFrame({"doc_id": ["d1"], "filing_date": [dt.date(2000, 1, 1)]}).lazy(),
    )
    empty_table = pl.DataFrame({"table_id": [], "estimate": []}, schema_overrides={"table_id": pl.Utf8, "estimate": pl.Float64})
    monkeypatch.setattr(runner, "build_lm2011_table_iv_results", lambda *_, **__: empty_table)
    monkeypatch.setattr(runner, "build_lm2011_table_v_results", lambda *_, **__: empty_table)
    monkeypatch.setattr(runner, "build_lm2011_table_vi_results", lambda *_, **__: empty_table)
    monkeypatch.setattr(runner, "build_lm2011_table_viii_results", lambda *_, **__: empty_table)
    monkeypatch.setattr(runner, "build_lm2011_table_ia_i_results", lambda *_, **__: empty_table)

    exit_code = runner.main(
        [
            "--sample-root",
            str(sample_root),
            "--upstream-run-root",
            str(upstream_run_root),
            "--additional-data-dir",
            str(additional_data_dir),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert (output_dir / "lm2011_sample_backbone.parquet").exists()
    assert (output_dir / "lm2011_event_panel.parquet").exists()
    assert (output_dir / "lm2011_table_iv_results.parquet").exists()

    manifest = json.loads((output_dir / "lm2011_sample_run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_status"] == "completed"
    assert manifest["stages"]["sample_backbone"]["status"] == "generated"
    assert manifest["stages"]["table_iv_results"]["status"] == "generated_empty"
    assert manifest["stages"]["table_iv_results"]["reason"] == runner.EMPTY_TABLE_REASON
    assert manifest["stages"]["trading_strategy_monthly_returns"]["status"] == runner.SKIPPED_MISSING_OPTIONAL_INPUT
    assert manifest["stages"]["table_ia_ii_results"]["status"] == runner.SKIPPED_MISSING_OPTIONAL_INPUT


@pytest.mark.skipif(
    os.environ.get("RUN_SAMPLE_LM2011_SMOKE") != "1"
    or not runner.DEFAULT_SAMPLE_ROOT.exists()
    or not runner.DEFAULT_UPSTREAM_RUN_ROOT.exists()
    or not runner.DEFAULT_ADDITIONAL_DATA_DIR.exists(),
    reason="Real sample smoke test requires local sample data and RUN_SAMPLE_LM2011_SMOKE=1",
)
def test_real_sample_smoke_builds_nonempty_panels_and_empty_tables(tmp_path: Path) -> None:
    output_dir = tmp_path / "sample_smoke_output"

    exit_code = runner.main(
        [
            "--sample-root",
            str(runner.DEFAULT_SAMPLE_ROOT),
            "--upstream-run-root",
            str(runner.DEFAULT_UPSTREAM_RUN_ROOT),
            "--additional-data-dir",
            str(runner.DEFAULT_ADDITIONAL_DATA_DIR),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0

    manifest = json.loads((output_dir / "lm2011_sample_run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["stages"]["event_panel"]["status"] == "generated"
    assert manifest["stages"]["sue_panel"]["status"] == "generated"
    assert manifest["row_counts"]["event_panel"] > 0
    assert manifest["row_counts"]["sue_panel"] > 0

    for stage_name in (
        "table_iv_results",
        "table_v_results",
        "table_vi_results",
        "table_viii_results",
        "table_ia_i_results",
    ):
        assert manifest["stages"][stage_name]["status"] == "generated_empty"
        assert manifest["stages"][stage_name]["reason"] == runner.EMPTY_TABLE_REASON

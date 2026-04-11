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


def _stub_table_i_sample_creation_df(
    *,
    window_label: str = "1994-2008",
    unavailable_mda: bool = False,
) -> pl.DataFrame:
    rows = [
        {
            "section_id": "full_10k_document",
            "section_label": "Full 10-K Document",
            "section_order": 1,
            "row_order": 1,
            "row_id": "edgar_complete_nonduplicate_sample",
            "display_label": f"EDGAR 10-K/10-K405 {window_label} complete sample (excluding duplicates)",
            "sample_size_kind": "count",
            "sample_size_value": 10.0,
            "observations_removed": None,
            "availability_status": "available",
            "availability_reason": None,
        },
        {
            "section_id": "firm_year_sample",
            "section_label": "Firm-Year Sample",
            "section_order": 2,
            "row_order": 2,
            "row_id": "firm_year_sample",
            "display_label": "Firm-Year Sample",
            "sample_size_kind": "count",
            "sample_size_value": 10.0,
            "observations_removed": None,
            "availability_status": "available",
            "availability_reason": None,
        },
    ]
    if unavailable_mda:
        rows.append(
            {
                "section_id": "mda_subsection",
                "section_label": "Management Discussion and Analysis (MD&A) Subsection",
                "section_order": 3,
                "row_order": 3,
                "row_id": "identifiable_mda",
                "display_label": "Subset of 10-K sample where MD&A section could be identified",
                "sample_size_kind": "count",
                "sample_size_value": None,
                "observations_removed": None,
                "availability_status": "unavailable",
                "availability_reason": "mda_text_features_unavailable",
            }
        )
    return pl.DataFrame(rows)


def _build_temp_layout(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    sample_root = tmp_path / "sample"
    upstream_run_root = tmp_path / "upstream"
    additional_data_dir = tmp_path / "additional"
    output_dir = tmp_path / "output"

    _write_parquet(
        sample_root / "year_merged" / "1995.parquet",
        pl.DataFrame(
            {
                "doc_id": ["d1"],
                "cik_10": ["0000000001"],
                "accession_nodash": ["000000000100000001"],
                "file_date_filename": [dt.date(1995, 1, 1)],
                "document_type_filename": ["10-K"],
                "full_text": ["token"],
            }
        ),
    )
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
    _write_parquet(
        upstream_run_root / "items_analysis" / "1995.parquet",
        pl.DataFrame(
            {
                "doc_id": ["d1"],
                "cik_10": ["0000000001"],
                "filing_date": [dt.date(1995, 1, 1)],
                "document_type_filename": ["10-K"],
                "item_id": ["7"],
                "full_text": ["token"],
            }
        ),
    )
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
    _write_text(additional_data_dir / "LM2011_MasterDictionary.txt", "Word\ntoken\nharvard\nrecognized\n")
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
    captured: dict[str, object] = {}

    def _capture_text_features_full_10k(*_: object, **kwargs: object) -> pl.LazyFrame:
        captured["text_features_full_10k_kwargs"] = kwargs
        return pl.DataFrame(
            {
                "doc_id": ["d1"],
                "token_count_full_10k": [2500],
                "total_token_count_full_10k": [2500],
            }
        ).lazy()

    def _capture_text_features_mda(*_: object, **kwargs: object) -> pl.LazyFrame:
        captured["text_features_mda_kwargs"] = kwargs
        return pl.DataFrame(
            {
                "doc_id": ["d1"],
                "token_count_mda": [300],
                "total_token_count_mda": [300],
            }
        ).lazy()

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
        _capture_text_features_full_10k,
    )
    monkeypatch.setattr(
        runner,
        "build_lm2011_text_features_mda",
        _capture_text_features_mda,
    )
    def _capture_table_i_sample_creation(*_: object, **kwargs: object) -> pl.DataFrame:
        sample_start = kwargs.get("sample_start", dt.date(1994, 1, 1))
        sample_end = kwargs.get("sample_end", dt.date(2008, 12, 31))
        captured.setdefault("table_i_windows", []).append((sample_start, sample_end))
        return _stub_table_i_sample_creation_df(
            window_label=f"{sample_start.year}-{sample_end.year}",
            unavailable_mda=True,
        )

    monkeypatch.setattr(
        runner,
        "build_lm2011_table_i_sample_creation",
        _capture_table_i_sample_creation,
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
            "--full-10k-cleaning-contract",
            "lm2011_paper",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "lm2011_sample_backbone.parquet").exists()
    assert (output_dir / "lm2011_table_i_sample_creation.parquet").exists()
    assert (output_dir / "lm2011_table_i_sample_creation.csv").exists()
    assert (output_dir / "lm2011_table_i_sample_creation.md").exists()
    assert (output_dir / "lm2011_table_i_sample_creation_1994_2024.parquet").exists()
    assert (output_dir / "lm2011_table_i_sample_creation_1994_2024.csv").exists()
    assert (output_dir / "lm2011_table_i_sample_creation_1994_2024.md").exists()
    assert (output_dir / "lm2011_event_panel.parquet").exists()
    assert (output_dir / "lm2011_table_iv_results.parquet").exists()
    assert "1994-2024" in (output_dir / "lm2011_table_i_sample_creation_1994_2024.md").read_text(encoding="utf-8")

    manifest = json.loads((output_dir / "lm2011_sample_run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_status"] == "completed"
    assert manifest["config"]["full_10k_cleaning_contract"] == "lm2011_paper"
    assert manifest["dictionary_inputs"]["resource_scope"] == "repo-local operative LM2011-style lexicon inputs"
    assert "not asserted to be the original LM2011" in manifest["dictionary_inputs"]["historical_provenance_warning"]
    master_resources = [
        resource
        for resource in manifest["dictionary_inputs"]["resources"]
        if resource["role"] == "recognized_word_master_dictionary"
    ]
    assert master_resources[0]["name"] == "LM2011_MasterDictionary.txt"
    assert "not provenance-verified" in master_resources[0]["provenance_status"]
    assert manifest["stages"]["sample_backbone"]["status"] == "generated"
    assert manifest["stages"]["table_i_sample_creation"]["status"] == "generated"
    assert set(manifest["stages"]["table_i_sample_creation"]["extra_artifacts"]) == {"csv", "markdown"}
    assert manifest["stages"]["table_i_sample_creation"]["warnings"] == [
        "MD&A subsection rows are unavailable because lm2011_text_features_mda was not provided."
    ]
    assert manifest["stages"]["table_i_sample_creation_1994_2024"]["status"] == "generated"
    assert set(manifest["stages"]["table_i_sample_creation_1994_2024"]["extra_artifacts"]) == {"csv", "markdown"}
    assert manifest["stages"]["table_i_sample_creation_1994_2024"]["warnings"] == [
        "MD&A subsection rows are unavailable because lm2011_text_features_mda was not provided."
    ]
    assert manifest["stages"]["table_iv_results"]["status"] == "generated_empty"
    assert manifest["stages"]["table_iv_results"]["reason"] == runner.EMPTY_TABLE_REASON
    assert manifest["stages"]["trading_strategy_monthly_returns"]["status"] == runner.SKIPPED_MISSING_OPTIONAL_INPUT
    assert manifest["stages"]["table_ia_ii_results"]["status"] == runner.SKIPPED_MISSING_OPTIONAL_INPUT
    assert captured["text_features_full_10k_kwargs"]["cleaning_contract"] == "lm2011_paper"
    assert captured["text_features_full_10k_kwargs"]["master_dictionary_words"] == ("token", "harvard", "recognized")
    assert captured["text_features_mda_kwargs"]["master_dictionary_words"] == ("token", "harvard", "recognized")
    assert captured["table_i_windows"] == [
        (dt.date(1994, 1, 1), dt.date(2008, 12, 31)),
        (dt.date(1994, 1, 1), dt.date(2024, 12, 31)),
    ]


def test_resolve_paths_honors_colab_style_override_paths(tmp_path: Path) -> None:
    sample_root, upstream_run_root, additional_data_dir, output_dir = _build_temp_layout(tmp_path)
    drive_root = tmp_path / "drive" / "MyDrive" / "sec_full"
    year_merged_dir = drive_root / "year_merged"
    derived_data_dir = drive_root / "derived_data"
    ccm_base_dir = drive_root / "ccm_parquet_data"
    items_analysis_dir = drive_root / "items_analysis"
    matched_clean_path = drive_root / "sec_ccm_premerge" / "sec_ccm_matched_clean.parquet"
    daily_panel_path = derived_data_dir / "final_flagged_data_compdesc_added.parquet"

    _write_parquet(year_merged_dir / "1995.parquet")
    _write_parquet(daily_panel_path)
    _write_parquet(matched_clean_path)
    _write_parquet(items_analysis_dir / "1995.parquet")

    ccm_nested = ccm_base_dir / "documents-export-2025-3-19"
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
        _write_parquet(ccm_nested / name)

    args = runner.parse_args(
        [
            "--sample-root",
            str(sample_root),
            "--upstream-run-root",
            str(upstream_run_root),
            "--additional-data-dir",
            str(additional_data_dir),
            "--output-dir",
            str(output_dir),
            "--year-merged-dir",
            str(year_merged_dir),
            "--matched-clean-path",
            str(matched_clean_path),
            "--daily-panel-path",
            str(daily_panel_path),
            "--items-analysis-dir",
            str(items_analysis_dir),
            "--ccm-base-dir",
            str(ccm_base_dir),
        ]
    )
    paths = runner._resolve_paths(args)

    assert paths.year_merged_dir == year_merged_dir.resolve()
    assert paths.matched_clean_path == matched_clean_path.resolve()
    assert paths.daily_panel_path == daily_panel_path.resolve()
    assert paths.items_analysis_dir == items_analysis_dir.resolve()
    assert paths.ccm_base_dir == ccm_base_dir.resolve()
    assert paths.filingdates_path == (ccm_nested / "filingdates.parquet").resolve()


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
    assert manifest["stages"]["table_i_sample_creation"]["status"] == "generated"
    assert manifest["stages"]["table_i_sample_creation_1994_2024"]["status"] == "generated"
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

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
    _write_parquet(
        ccm_base_dir / "sfz_mth.parquet",
        pl.DataFrame(
            {
                "KYPERMNO": [1],
                "MCALDT": [dt.date(2000, 1, 31)],
                "MRET": [0.01],
                "MTCAP": [100.0],
            }
        ),
    )

    _write_parquet(
        upstream_run_root / "sec_ccm_premerge" / "sec_ccm_matched_clean.parquet",
        pl.DataFrame(
            {
                "doc_id": ["d1"],
                "KYPERMNO": [1],
                "gvkey": [1],
                "SRCTYPE": ["10K"],
            }
        ),
    )
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
    _write_parquet(
        upstream_run_root / "refinitiv_doc_ownership_lm2011" / "refinitiv_lm2011_doc_ownership.parquet",
        pl.DataFrame(
            {
                "doc_id": ["d1"],
                "institutional_ownership_pct": [44.0],
            }
        ),
    )
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
    _write_text(
        additional_data_dir / "F-F_Research_Data_Factors.csv",
        "\n".join(
            [
                "monthly preamble",
                "",
                ",Mkt-RF,SMB,HML,RF",
                "199707,   7.81,  -2.42,   1.00,   0.45",
                "199708,  -5.56,   3.91,   0.80,   0.42",
                " Annual Factors: January-December",
                ",Mkt-RF,SMB,HML,RF",
                "1997,  33.35, -10.00,  15.00,   5.00",
            ]
        ),
    )
    _write_text(
        additional_data_dir / "F-F_Momentum_Factor.csv",
        "\n".join(
            [
                "momentum preamble",
                "more preamble",
                "",
                ",Mom",
                "199707,   1.22",
                "199708,  -0.56",
                "Annual Factors:",
                ",Mom",
                "1997,  -10.00",
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


def test_resolve_paths_auto_resolves_sfz_mth_monthly_stock(tmp_path: Path) -> None:
    sample_root, upstream_run_root, additional_data_dir, output_dir = _build_temp_layout(tmp_path)

    paths = runner._resolve_paths(
        runner.parse_args(
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
    )

    assert paths.monthly_stock_path == (
        sample_root / "ccm_parquet_data" / "documents-export-2025-3-19" / "sfz_mth.parquet"
    ).resolve()


def test_resolve_paths_auto_resolves_prebuilt_sample_backbone(tmp_path: Path) -> None:
    sample_root, upstream_run_root, additional_data_dir, output_dir = _build_temp_layout(tmp_path)
    prebuilt = upstream_run_root / "sec_ccm_premerge" / "lm2011_sample_backbone.parquet"
    _write_parquet(prebuilt, pl.DataFrame({"doc_id": ["d1"]}))

    paths = runner._resolve_paths(
        runner.parse_args(
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
    )

    assert paths.sample_backbone_path == prebuilt.resolve()


def test_parse_args_uses_memory_hardened_defaults() -> None:
    args = runner.parse_args([])

    assert args.full_10k_cleaning_contract == runner.DEFAULT_LM2011_FULL_10K_CLEANING_CONTRACT
    assert args.full_10k_text_feature_batch_size is None
    assert args.mda_text_feature_batch_size is None
    assert args.text_feature_batch_size is None
    assert args.event_window_doc_batch_size == runner.DEFAULT_LM2011_EVENT_WINDOW_DOC_BATCH_SIZE
    assert args.print_ram_stats is False
    assert args.ram_log_interval_batches == runner.DEFAULT_RAM_LOG_INTERVAL_BATCHES


def test_resolve_paths_explicit_sample_backbone_takes_precedence(tmp_path: Path) -> None:
    sample_root, upstream_run_root, additional_data_dir, output_dir = _build_temp_layout(tmp_path)
    auto_prebuilt = upstream_run_root / "sec_ccm_premerge" / "lm2011_sample_backbone.parquet"
    explicit_prebuilt = tmp_path / "explicit" / "lm2011_sample_backbone.parquet"
    _write_parquet(auto_prebuilt, pl.DataFrame({"doc_id": ["auto"]}))
    _write_parquet(explicit_prebuilt, pl.DataFrame({"doc_id": ["explicit"]}))

    paths = runner._resolve_paths(
        runner.parse_args(
            [
                "--sample-root",
                str(sample_root),
                "--upstream-run-root",
                str(upstream_run_root),
                "--additional-data-dir",
                str(additional_data_dir),
                "--output-dir",
                str(output_dir),
                "--sample-backbone-path",
                str(explicit_prebuilt),
            ]
        )
    )

    assert paths.sample_backbone_path == explicit_prebuilt.resolve()


def test_resolve_paths_legacy_text_feature_batch_size_applies_to_both_text_stages(tmp_path: Path) -> None:
    sample_root, upstream_run_root, additional_data_dir, output_dir = _build_temp_layout(tmp_path)

    paths = runner._resolve_paths(
        runner.parse_args(
            [
                "--sample-root",
                str(sample_root),
                "--upstream-run-root",
                str(upstream_run_root),
                "--additional-data-dir",
                str(additional_data_dir),
                "--output-dir",
                str(output_dir),
                "--text-feature-batch-size",
                "7",
            ]
        )
    )

    assert paths.full_10k_text_feature_batch_size == 7
    assert paths.mda_text_feature_batch_size == 7


def test_prepare_lm2011_sec_backbone_input_excludes_full_text() -> None:
    out = runner._prepare_lm2011_sec_backbone_input_lf(
        pl.DataFrame(
            {
                "doc_id": ["d1"],
                "cik_10": ["0000000001"],
                "accession_nodash": ["000000000100000001"],
                "file_date_filename": [dt.date(1995, 1, 1)],
                "document_type_filename": ["10-K"],
                "full_text": ["large text payload"],
            }
        ).lazy()
    ).collect()

    assert "full_text" not in out.columns
    assert out.columns == [
        "doc_id",
        "cik_10",
        "accession_nodash",
        "document_type_filename",
        "filing_date",
    ]


def test_filter_to_sample_doc_ids_limits_text_universe() -> None:
    text_lf = pl.DataFrame({"doc_id": ["d1", "d2"], "full_text": ["keep", "drop"]}).lazy()
    sample_lf = pl.DataFrame({"doc_id": ["d1"]}).lazy()

    out = runner._filter_to_sample_doc_ids_lf(text_lf, sample_lf).collect()

    assert out.get_column("doc_id").to_list() == ["d1"]


def test_write_frame_artifact_streams_lazy_frame_without_collect(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail_collect(_: pl.LazyFrame | pl.DataFrame) -> pl.DataFrame:
        raise AssertionError("Lazy stage writes must not eagerly collect")

    monkeypatch.setattr(runner, "_collect_frame", _fail_collect)
    output_path, row_count = runner._write_frame_artifact(
        pl.DataFrame({"x": [1, 2]}).lazy(),
        tmp_path / "stage.parquet",
    )

    assert output_path.exists()
    assert row_count == 2
    assert pl.read_parquet(output_path).get_column("x").to_list() == [1, 2]


def test_write_stage_records_reused_source_path(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    output_dir = tmp_path / "output"
    manifest_path = tmp_path / "manifest.json"
    _write_parquet(source, pl.DataFrame({"doc_id": ["d1"]}))
    manifest: dict[str, object] = {
        "roots": {"output_dir": str(output_dir)},
        "artifacts": {},
        "row_counts": {},
        "stages": {},
    }

    runner._write_stage(
        manifest,
        output_dir=output_dir,
        stage_name="sample_backbone",
        frame=pl.scan_parquet(source),
        manifest_path=manifest_path,
        source_path=source,
    )

    stage = manifest["stages"]["sample_backbone"]  # type: ignore[index]
    assert stage["source_path"] == str(source.resolve())
    assert stage["row_count"] == 1
    assert manifest_path.exists()


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


def test_load_ff_factors_daily_lf_detects_header_and_ignores_footer(tmp_path: Path) -> None:
    csv_path = tmp_path / "F-F_Research_Data_Factors_daily.csv"
    csv_path.write_text(
        "\n".join(
            [
                "preamble1",
                "preamble2",
                "preamble3",
                "preamble4",
                "preamble5",
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


def test_load_monthly_ff_factors_excludes_annual_rows_and_missing_sentinels(tmp_path: Path) -> None:
    csv_path = tmp_path / "F-F_Research_Data_Factors.csv"
    csv_path.write_text(
        "\n".join(
            [
                "monthly preamble",
                "",
                ",Mkt-RF,SMB,HML,RF",
                "199707,   7.81,  -2.42,   1.00,   0.45",
                "199708, -99.99,   3.91,   0.80,   0.42",
                " Annual Factors: January-December",
                ",Mkt-RF,SMB,HML,RF",
                "1997,  33.35, -10.00,  15.00,   5.00",
            ]
        ),
        encoding="utf-8",
    )

    out = runner._load_ff_factors_monthly_lf(csv_path).collect()

    assert out.height == 1
    assert out.columns == ["mkt_rf", "smb", "hml", "rf", "month_end"]
    assert out.item(0, "month_end") == dt.date(1997, 7, 31)
    assert out.item(0, "mkt_rf") == pytest.approx(7.81)


def test_load_monthly_momentum_factors_excludes_annual_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "F-F_Momentum_Factor.csv"
    csv_path.write_text(
        "\n".join(
            [
                "momentum preamble",
                "more preamble",
                "",
                ",Mom",
                "199707,   1.22",
                "199708,  -0.56",
                "Annual Factors:",
                ",Mom",
                "1997,  -10.00",
            ]
        ),
        encoding="utf-8",
    )

    out = runner._load_momentum_factors_monthly_lf(csv_path).collect().sort("month_end")

    assert out.height == 2
    assert out.get_column("month_end").to_list() == [dt.date(1997, 7, 31), dt.date(1997, 8, 31)]
    assert out.get_column("mom").to_list() == pytest.approx([1.22, -0.56])


def test_load_ff_factors_monthly_with_mom_lf_joins_and_sorts(tmp_path: Path) -> None:
    ff_path = tmp_path / "F-F_Research_Data_Factors.csv"
    mom_path = tmp_path / "F-F_Momentum_Factor.csv"
    ff_path.write_text(
        "\n".join(
            [
                "preamble",
                ",Mkt-RF,SMB,HML,RF",
                "199708,  -5.56,   3.91,   0.80,   0.42",
                "199707,   7.81,  -2.42,   1.00,   0.45",
            ]
        ),
        encoding="utf-8",
    )
    mom_path.write_text(
        "\n".join(
            [
                "preamble",
                ",Mom",
                "199708,  -0.56",
                "199707,   1.22",
                "199709,   9.99",
            ]
        ),
        encoding="utf-8",
    )

    out = runner._load_ff_factors_monthly_with_mom_lf(ff_path, mom_path).collect()

    assert out.columns == ["month_end", "mkt_rf", "smb", "hml", "rf", "mom"]
    assert out.get_column("month_end").to_list() == [dt.date(1997, 7, 31), dt.date(1997, 8, 31)]
    assert out.get_column("mom").to_list() == pytest.approx([1.22, -0.56])


def test_load_ff_factors_monthly_with_mom_lf_rejects_duplicate_months(tmp_path: Path) -> None:
    ff_path = tmp_path / "F-F_Research_Data_Factors.csv"
    mom_path = tmp_path / "F-F_Momentum_Factor.csv"
    ff_path.write_text(
        "\n".join(
            [
                "preamble",
                ",Mkt-RF,SMB,HML,RF",
                "199707,   7.81,  -2.42,   1.00,   0.45",
                "199707,   7.82,  -2.43,   1.01,   0.46",
            ]
        ),
        encoding="utf-8",
    )
    mom_path.write_text(
        "\n".join(
            [
                "preamble",
                ",Mom",
                "199707,   1.22",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="monthly FF factors contains duplicate month_end values"):
        runner._load_ff_factors_monthly_with_mom_lf(ff_path, mom_path)


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
    prebuilt_backbone = upstream_run_root / "sec_ccm_premerge" / "lm2011_sample_backbone.parquet"
    _write_parquet(prebuilt_backbone, pl.DataFrame({"doc_id": ["d1"]}))
    captured: dict[str, object] = {}

    def _capture_text_features_full_10k(*_: object, **kwargs: object) -> int:
        captured["text_features_full_10k_kwargs"] = kwargs
        pl.DataFrame(
            {
                "doc_id": ["d1"],
                "token_count_full_10k": [2500],
                "total_token_count_full_10k": [2500],
            }
        ).write_parquet(Path(kwargs["output_path"]))
        return 1

    def _capture_text_features_mda(*_: object, **kwargs: object) -> int:
        captured["text_features_mda_kwargs"] = kwargs
        pl.DataFrame(
            {
                "doc_id": ["d1"],
                "token_count_mda": [300],
                "total_token_count_mda": [300],
            }
        ).write_parquet(Path(kwargs["output_path"]))
        return 1

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
    monkeypatch.setattr(
        runner,
        "build_lm2011_sample_backbone",
        lambda *_, **__: (_ for _ in ()).throw(AssertionError("prebuilt sample backbone should be reused")),
    )
    monkeypatch.setattr(runner, "build_annual_accounting_panel", lambda *_, **__: pl.DataFrame({"gvkey_int": [1]}).lazy())
    monkeypatch.setattr(runner, "build_quarterly_accounting_panel", lambda *_, **__: pl.DataFrame({"gvkey_int": [1]}).lazy())
    monkeypatch.setattr(
        runner,
        "write_lm2011_text_features_full_10k_parquet",
        _capture_text_features_full_10k,
    )
    monkeypatch.setattr(
        runner,
        "write_lm2011_text_features_mda_parquet",
        _capture_text_features_mda,
    )

    monkeypatch.setattr(
        runner.lm2011_pipeline,
        "write_lm2011_event_screen_surface_parquet",
        lambda *_, **kwargs: pl.DataFrame({"doc_id": ["d1"]}).write_parquet(Path(kwargs["output_path"])) or 1,
    )

    def _capture_table_i_sample_creation(*_: object, **kwargs: object) -> pl.DataFrame:
        sec_input = _[0]
        if isinstance(sec_input, pl.LazyFrame):
            captured.setdefault("table_i_sec_columns", []).append(sec_input.collect_schema().names())
        sample_start = kwargs.get("sample_start", dt.date(1994, 1, 1))
        sample_end = kwargs.get("sample_end", dt.date(2008, 12, 31))
        captured.setdefault("table_i_windows", []).append((sample_start, sample_end))
        captured.setdefault("table_i_has_precomputed_surface", []).append(
            kwargs.get("_precomputed_event_screen_surface_lf") is not None
        )
        captured.setdefault("table_i_has_progress_callback", []).append(
            kwargs.get("_event_screen_progress_callback") is not None
        )
        return _stub_table_i_sample_creation_df(
            window_label=f"{sample_start.year}-{sample_end.year}",
            unavailable_mda=True,
        )

    monkeypatch.setattr(
        runner,
        "build_lm2011_table_i_sample_creation",
        _capture_table_i_sample_creation,
    )
    def _capture_event_panel(*_: object, **kwargs: object) -> pl.LazyFrame:
        captured["event_panel_has_precomputed_surface"] = kwargs.get("_precomputed_event_screen_surface_lf") is not None
        return pl.DataFrame({"doc_id": ["d1"]}).lazy()

    monkeypatch.setattr(runner, "build_lm2011_event_panel", _capture_event_panel)
    monkeypatch.setattr(
        runner,
        "write_lm2011_sue_panel_parquet",
        lambda *_, **kwargs: pl.DataFrame({"doc_id": ["d1"]}).write_parquet(Path(kwargs["output_path"])) or 1,
    )
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
    monkeypatch.setattr(
        runner,
        "build_lm2011_trading_strategy_monthly_returns",
        lambda *_, **__: pl.DataFrame(
            {
                "portfolio_month": [dt.date(2000, 1, 31)],
                "sort_signal_name": ["fin_neg_prop"],
                "long_short_return": [0.01],
            }
        ).lazy(),
    )
    monkeypatch.setattr(runner, "build_lm2011_table_ia_ii_results", lambda *_, **__: empty_table)
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
    assert (output_dir / "lm2011_event_screen_surface.parquet").exists()
    assert (output_dir / "lm2011_table_i_sample_creation.parquet").exists()
    assert (output_dir / "lm2011_table_i_sample_creation.csv").exists()
    assert (output_dir / "lm2011_table_i_sample_creation.md").exists()
    assert (output_dir / "lm2011_table_i_sample_creation_1994_2024.parquet").exists()
    assert (output_dir / "lm2011_table_i_sample_creation_1994_2024.csv").exists()
    assert (output_dir / "lm2011_table_i_sample_creation_1994_2024.md").exists()
    assert (output_dir / "lm2011_event_panel.parquet").exists()
    assert (output_dir / "lm2011_table_iv_results.parquet").exists()
    assert (output_dir / "lm2011_ff_factors_monthly_with_mom_normalized.parquet").exists()
    assert "1994-2024" in (output_dir / "lm2011_table_i_sample_creation_1994_2024.md").read_text(encoding="utf-8")

    manifest = json.loads((output_dir / "lm2011_sample_run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_status"] == "completed"
    assert manifest["started_at_utc"] is not None
    assert manifest["completed_at_utc"] is not None
    assert manifest["elapsed_seconds"] is not None
    assert manifest["failed_stage"] is None
    assert manifest["config"]["full_10k_cleaning_contract"] == "lm2011_paper"
    assert manifest["config"]["raw_mda_cleaning_policy_id"] == "raw_item_text"
    assert manifest["config"]["text_feature_batch_size"] == 4
    assert manifest["config"]["full_10k_text_feature_batch_size"] == 4
    assert manifest["config"]["mda_text_feature_batch_size"] == 20
    assert manifest["config"]["event_window_doc_batch_size"] == 50
    assert manifest["config"]["print_ram_stats"] is False
    assert manifest["config"]["ram_log_interval_batches"] == runner.DEFAULT_RAM_LOG_INTERVAL_BATCHES
    assert manifest["resolved_inputs"]["ff_monthly_csv_path"].endswith("F-F_Research_Data_Factors.csv")
    assert manifest["resolved_inputs"]["momentum_monthly_csv_path"].endswith("F-F_Momentum_Factor.csv")
    assert manifest["resolved_inputs"]["monthly_stock_path"].endswith("sfz_mth.parquet")
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
    assert manifest["stages"]["sample_backbone"]["source_path"] == str(prebuilt_backbone.resolve())
    assert manifest["stages"]["event_screen_surface"]["status"] == "generated"
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
    assert manifest["stages"]["ff_factors_monthly_with_mom_normalized"]["status"] == "generated"
    assert manifest["stages"]["ff_factors_monthly_with_mom_normalized"]["row_count"] == 2
    assert manifest["stages"]["trading_strategy_monthly_returns"]["status"] == "generated"
    assert manifest["stages"]["table_ia_ii_results"]["status"] == "generated_empty"
    assert manifest["stages"]["table_ia_ii_results"]["reason"] == runner.EMPTY_TABLE_REASON
    assert captured["text_features_full_10k_kwargs"]["cleaning_contract"] == "lm2011_paper"
    assert captured["text_features_full_10k_kwargs"]["batch_size"] == 4
    assert callable(captured["text_features_full_10k_kwargs"]["progress_callback"])
    assert captured["text_features_full_10k_kwargs"]["master_dictionary_words"] == ("token", "harvard", "recognized")
    assert captured["text_features_mda_kwargs"]["batch_size"] == 20
    assert callable(captured["text_features_mda_kwargs"]["progress_callback"])
    assert captured["text_features_mda_kwargs"]["master_dictionary_words"] == ("token", "harvard", "recognized")
    assert captured["table_i_windows"] == [
        (dt.date(1994, 1, 1), dt.date(2008, 12, 31)),
        (dt.date(1994, 1, 1), dt.date(2024, 12, 31)),
    ]
    assert captured["table_i_has_precomputed_surface"] == [True, True]
    assert captured["table_i_has_progress_callback"] == [False, False]
    assert captured["event_panel_has_precomputed_surface"] is True
    assert all("full_text" not in columns for columns in captured["table_i_sec_columns"])


def test_runner_builds_event_screen_surface_twice_and_reuses_default_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_root, upstream_run_root, additional_data_dir, output_dir = _build_temp_layout(tmp_path)
    prebuilt_backbone = upstream_run_root / "sec_ccm_premerge" / "lm2011_sample_backbone.parquet"
    _write_parquet(prebuilt_backbone, pl.DataFrame({"doc_id": ["d1"]}))
    captured: dict[str, object] = {"surface_calls": 0}

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
    monkeypatch.setattr(runner, "build_annual_accounting_panel", lambda *_, **__: pl.DataFrame({"gvkey_int": [1]}).lazy())
    monkeypatch.setattr(runner, "build_quarterly_accounting_panel", lambda *_, **__: pl.DataFrame({"gvkey_int": [1]}).lazy())
    monkeypatch.setattr(
        runner,
        "write_lm2011_text_features_full_10k_parquet",
        lambda *_, **kwargs: pl.DataFrame({"doc_id": ["d1"], "total_token_count_full_10k": [2500]}).write_parquet(Path(kwargs["output_path"])) or 1,
    )
    monkeypatch.setattr(
        runner,
        "write_lm2011_text_features_mda_parquet",
        lambda *_, **kwargs: pl.DataFrame({"doc_id": ["d1"], "total_token_count_mda": [300]}).write_parquet(Path(kwargs["output_path"])) or 1,
    )

    def _capture_surface(*_: object, **kwargs: object) -> pl.DataFrame:
        captured["surface_calls"] = int(captured["surface_calls"]) + 1
        captured.setdefault("surface_progress_callbacks", []).append(kwargs.get("progress_callback") is not None)
        return pl.DataFrame({"doc_id": ["d1"]})

    def _capture_surface_write(*_: object, **kwargs: object) -> int:
        captured["surface_calls"] = int(captured["surface_calls"]) + 1
        captured.setdefault("surface_progress_callbacks", []).append(kwargs.get("progress_callback") is not None)
        pl.DataFrame({"doc_id": ["d1"]}).write_parquet(Path(kwargs["output_path"]))
        return 1

    monkeypatch.setattr(runner.lm2011_pipeline, "write_lm2011_event_screen_surface_parquet", _capture_surface_write)
    monkeypatch.setattr(runner.lm2011_pipeline, "_build_lm2011_event_screen_surface_batched", _capture_surface)

    def _capture_table_i(*_: object, **kwargs: object) -> pl.DataFrame:
        captured.setdefault("table_i_has_precomputed_surface", []).append(
            kwargs.get("_precomputed_event_screen_surface_lf") is not None
        )
        if kwargs.get("_precomputed_event_screen_surface_lf") is None:
            runner.lm2011_pipeline._build_lm2011_event_screen_surface_batched(
                pl.DataFrame({"doc_id": ["d1"]}).lazy(),
                pl.DataFrame({"CALDT": [dt.date(1995, 1, 1)], "KYPERMNO": [1], "FINAL_RET": [0.0], "VOL": [1.0], "FINAL_PRC": [1.0], "PRC": [1.0], "SHROUT": [1.0], "SHRCD": [10], "EXCHCD": [1]}).lazy(),
                pl.DataFrame({"gvkey_int": [1]}).lazy(),
                pl.DataFrame({"trading_date": [dt.date(1995, 1, 1)], "mkt_rf": [0.0], "smb": [0.0], "hml": [0.0], "rf": [0.0]}).lazy(),
                pl.DataFrame({"doc_id": ["d1"], "total_token_count_full_10k": [2500]}).lazy(),
                event_window_doc_batch_size=int(kwargs["event_window_doc_batch_size"]),
                progress_callback=kwargs.get("_event_screen_progress_callback"),
            )
        return _stub_table_i_sample_creation_df(
            window_label=f"{kwargs.get('sample_start', dt.date(1994, 1, 1)).year}-{kwargs.get('sample_end', dt.date(2008, 12, 31)).year}",
            unavailable_mda=False,
        )

    monkeypatch.setattr(runner, "build_lm2011_table_i_sample_creation", _capture_table_i)

    def _capture_event_panel(*_: object, **kwargs: object) -> pl.LazyFrame:
        captured["event_panel_has_precomputed_surface"] = kwargs.get("_precomputed_event_screen_surface_lf") is not None
        return pl.DataFrame({"doc_id": ["d1"]}).lazy()

    monkeypatch.setattr(runner, "build_lm2011_event_panel", _capture_event_panel)
    monkeypatch.setattr(
        runner,
        "write_lm2011_sue_panel_parquet",
        lambda *_, **kwargs: pl.DataFrame({"doc_id": ["d1"]}).write_parquet(Path(kwargs["output_path"])) or 1,
    )
    empty_table = pl.DataFrame({"table_id": [], "estimate": []}, schema_overrides={"table_id": pl.Utf8, "estimate": pl.Float64})
    monkeypatch.setattr(runner, "build_lm2011_return_regression_panel", lambda *_, **__: pl.DataFrame({"doc_id": ["d1"]}).lazy())
    monkeypatch.setattr(runner, "build_lm2011_sue_regression_panel", lambda *_, **__: pl.DataFrame({"doc_id": ["d1"]}).lazy())
    monkeypatch.setattr(runner, "build_lm2011_table_iv_results", lambda *_, **__: empty_table)
    monkeypatch.setattr(runner, "build_lm2011_table_v_results", lambda *_, **__: empty_table)
    monkeypatch.setattr(runner, "build_lm2011_table_vi_results", lambda *_, **__: empty_table)
    monkeypatch.setattr(runner, "build_lm2011_table_viii_results", lambda *_, **__: empty_table)
    monkeypatch.setattr(runner, "build_lm2011_table_ia_i_results", lambda *_, **__: empty_table)
    monkeypatch.setattr(runner, "build_lm2011_table_ia_ii_results", lambda *_, **__: empty_table)
    monkeypatch.setattr(
        runner,
        "build_lm2011_trading_strategy_monthly_returns",
        lambda *_, **__: pl.DataFrame({"portfolio_month": [dt.date(2000, 1, 31)], "sort_signal_name": ["fin_neg_prop"], "long_short_return": [0.01]}).lazy(),
    )

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
    assert captured["surface_calls"] == 1
    assert captured["surface_progress_callbacks"] == [True]
    assert captured["table_i_has_precomputed_surface"] == [True, True]
    assert captured["event_panel_has_precomputed_surface"] is True


def test_runner_failure_manifest_preserves_completed_stages_before_extended_table_i_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_root, upstream_run_root, additional_data_dir, output_dir = _build_temp_layout(tmp_path)
    prebuilt_backbone = upstream_run_root / "sec_ccm_premerge" / "lm2011_sample_backbone.parquet"
    _write_parquet(prebuilt_backbone, pl.DataFrame({"doc_id": ["d1"]}))

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
    monkeypatch.setattr(runner, "build_annual_accounting_panel", lambda *_, **__: pl.DataFrame({"gvkey_int": [1]}).lazy())
    monkeypatch.setattr(runner, "build_quarterly_accounting_panel", lambda *_, **__: pl.DataFrame({"gvkey_int": [1]}).lazy())
    monkeypatch.setattr(
        runner,
        "write_lm2011_text_features_full_10k_parquet",
        lambda *_, **kwargs: pl.DataFrame({"doc_id": ["d1"], "total_token_count_full_10k": [2500]}).write_parquet(Path(kwargs["output_path"])) or 1,
    )
    monkeypatch.setattr(
        runner,
        "write_lm2011_text_features_mda_parquet",
        lambda *_, **kwargs: pl.DataFrame({"doc_id": ["d1"], "total_token_count_mda": [300]}).write_parquet(Path(kwargs["output_path"])) or 1,
    )
    monkeypatch.setattr(
        runner.lm2011_pipeline,
        "write_lm2011_event_screen_surface_parquet",
        lambda *_, **kwargs: pl.DataFrame({"doc_id": ["d1"]}).write_parquet(Path(kwargs["output_path"])) or 1,
    )

    def _fail_extended_table_i(*_: object, **kwargs: object) -> pl.DataFrame:
        sample_end = kwargs.get("sample_end", dt.date(2008, 12, 31))
        if sample_end == dt.date(2024, 12, 31):
            raise RuntimeError("extended table i boom")
        return _stub_table_i_sample_creation_df(unavailable_mda=False)

    monkeypatch.setattr(runner, "build_lm2011_table_i_sample_creation", _fail_extended_table_i)
    monkeypatch.setattr(runner, "build_lm2011_event_panel", lambda *_, **__: pl.DataFrame({"doc_id": ["d1"]}).lazy())
    monkeypatch.setattr(
        runner,
        "write_lm2011_sue_panel_parquet",
        lambda *_, **kwargs: pl.DataFrame({"doc_id": ["d1"]}).write_parquet(Path(kwargs["output_path"])) or 1,
    )
    monkeypatch.setattr(runner, "build_lm2011_return_regression_panel", lambda *_, **__: pl.DataFrame({"doc_id": ["d1"]}).lazy())
    monkeypatch.setattr(runner, "build_lm2011_sue_regression_panel", lambda *_, **__: pl.DataFrame({"doc_id": ["d1"]}).lazy())
    empty_table = pl.DataFrame({"table_id": [], "estimate": []}, schema_overrides={"table_id": pl.Utf8, "estimate": pl.Float64})
    monkeypatch.setattr(runner, "build_lm2011_table_iv_results", lambda *_, **__: empty_table)
    monkeypatch.setattr(runner, "build_lm2011_table_v_results", lambda *_, **__: empty_table)
    monkeypatch.setattr(runner, "build_lm2011_table_vi_results", lambda *_, **__: empty_table)
    monkeypatch.setattr(runner, "build_lm2011_table_viii_results", lambda *_, **__: empty_table)
    monkeypatch.setattr(runner, "build_lm2011_table_ia_i_results", lambda *_, **__: empty_table)
    monkeypatch.setattr(runner, "build_lm2011_table_ia_ii_results", lambda *_, **__: empty_table)
    monkeypatch.setattr(
        runner,
        "build_lm2011_trading_strategy_monthly_returns",
        lambda *_, **__: pl.DataFrame({"portfolio_month": [dt.date(2000, 1, 31)], "sort_signal_name": ["fin_neg_prop"], "long_short_return": [0.01]}).lazy(),
    )

    with pytest.raises(RuntimeError, match="extended table i boom"):
        runner.main(
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

    manifest = json.loads((output_dir / "lm2011_sample_run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_status"] == "failed"
    assert manifest["failed_stage"] == "table_i_sample_creation_1994_2024"
    assert manifest["stages"]["sample_backbone"]["status"] == "generated"
    assert manifest["stages"]["event_screen_surface"]["status"] == "generated"
    assert manifest["stages"]["table_i_sample_creation"]["status"] == "generated"
    assert manifest["stages"]["table_i_sample_creation_1994_2024"]["status"] == "failed"
    assert "event_panel" not in manifest["artifacts"]


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


def test_main_delegates_to_shared_lm2011_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sample_root, upstream_run_root, additional_data_dir, output_dir = _build_temp_layout(tmp_path)
    captured: dict[str, object] = {}

    def _capture_run(run_cfg: runner.LM2011PostRefinitivRunConfig) -> int:
        captured["run_cfg"] = run_cfg
        return 0

    monkeypatch.setattr(runner, "run_lm2011_post_refinitiv_pipeline", _capture_run)

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
    run_cfg = captured["run_cfg"]
    assert isinstance(run_cfg, runner.LM2011PostRefinitivRunConfig)
    assert run_cfg.paths.output_dir == output_dir.resolve()
    assert set(run_cfg.enabled_stages) == set(runner.LM2011_ALL_STAGE_NAMES)
    assert run_cfg.fail_closed_for_enabled_stages is False
    assert run_cfg.paths.full_10k_cleaning_contract == runner.DEFAULT_LM2011_FULL_10K_CLEANING_CONTRACT
    assert run_cfg.paths.full_10k_text_feature_batch_size == runner.DEFAULT_LM2011_FULL_10K_TEXT_FEATURE_BATCH_SIZE
    assert run_cfg.paths.mda_text_feature_batch_size == runner.DEFAULT_LM2011_MDA_TEXT_FEATURE_BATCH_SIZE
    assert run_cfg.paths.event_window_doc_batch_size == runner.DEFAULT_LM2011_EVENT_WINDOW_DOC_BATCH_SIZE
    assert run_cfg.paths.print_ram_stats is False
    assert run_cfg.paths.ram_log_interval_batches == runner.DEFAULT_RAM_LOG_INTERVAL_BATCHES


def test_text_feature_progress_logger_respects_interval(capsys) -> None:
    logger = runner._make_text_feature_progress_logger(
        "text_features_full_10k",
        print_ram_stats=False,
        ram_log_interval_batches=3,
    )

    logger({"event": "stage_source_start"})
    logger({"event": "batch", "batch_index": 1, "batch_doc_count": 2, "docs_completed": 2})
    logger({"event": "batch", "batch_index": 2, "batch_doc_count": 2, "docs_completed": 4})
    logger({"event": "batch", "batch_index": 3, "batch_doc_count": 2, "docs_completed": 6})

    output_lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]

    assert len(output_lines) == 3
    assert "event': 'stage_source_start'" in output_lines[0]
    assert "batch_index': 1" in output_lines[1]
    assert "batch_index': 3" in output_lines[2]


def test_shared_lm2011_pipeline_fails_closed_for_enabled_stage(tmp_path: Path) -> None:
    sample_root, upstream_run_root, additional_data_dir, output_dir = _build_temp_layout(tmp_path)
    paths = runner._resolve_paths(
        runner.parse_args(
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
    )

    with pytest.raises(RuntimeError, match="LM2011 stage text_features_full_10k"):
        runner.run_lm2011_post_refinitiv_pipeline(
            runner.LM2011PostRefinitivRunConfig(
                paths=runner.RunnerPaths(
                    **{
                        **paths.__dict__,
                        "year_merged_dir": tmp_path / "missing_year_merged",
                    }
                ),
                enabled_stages=("text_features_full_10k",),
                fail_closed_for_enabled_stages=True,
            )
        )


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
    assert manifest["stages"]["event_screen_surface"]["status"] == "generated"
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

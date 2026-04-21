from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from thesis_pkg.core.sec import lm2011_text
from thesis_pkg.notebooks_and_scripts import lm2011_sample_post_refinitiv_runner as runner


def _write_parquet(path: Path, df: pl.DataFrame | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df is None:
        df = pl.DataFrame({"placeholder": [1]})
    df.write_parquet(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _load_lm2011_notebook_cell_sources() -> list[str]:
    notebook_path = Path("src/thesis_pkg/notebooks_and_scripts/lm2011_sample_post_refinitiv_runner.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    return ["".join(cell.get("source", [])) for cell in notebook["cells"]]


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


def _empty_quarterly_results_df() -> pl.DataFrame:
    return pl.DataFrame(
        {"table_id": [], "estimate": []},
        schema_overrides={"table_id": pl.Utf8, "estimate": pl.Float64},
    )


def _skipped_quarter_diagnostics_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "table_id": ["table_iv_full_10k"],
            "text_scope": ["full_10k"],
            "dependent_variable": ["filing_period_excess_return"],
            "signal_name": ["h4n_inf_prop"],
            "quarter_start": [dt.date(2000, 1, 1)],
            "skip_reason": ["rank_deficient_design"],
            "n_obs": [26],
            "industry_count": [15],
            "rank": [21],
            "column_count": [22],
            "condition_number": [2.13968e18],
            "regressors": ["intercept, h4n_inf_prop, log_size"],
            "duplicate_regressor_pairs": ['[["nasdaq_dummy","_industry_dummy_36"]]'],
            "restoring_drop_candidates": ['["nasdaq_dummy","_industry_dummy_36"]'],
        }
    )


def _quarterly_bundle(
    *,
    results_df: pl.DataFrame | None = None,
    skipped_quarters_df: pl.DataFrame | None = None,
) -> runner._QuarterlyFamaMacbethBundle:
    return runner._QuarterlyFamaMacbethBundle(
        results_df=_empty_quarterly_results_df() if results_df is None else results_df,
        skipped_quarters_df=(
            pl.DataFrame(schema=_skipped_quarter_diagnostics_df().schema)
            if skipped_quarters_df is None
            else skipped_quarters_df
        ),
    )


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

    _write_text(additional_data_dir / "Fin-Neg.txt", "token\nstale_neg\n")
    _write_text(additional_data_dir / "Fin-Pos.txt", "harvard\nstale_pos\n")
    _write_text(additional_data_dir / "Fin-Unc.txt", "recognized\nstale_unc\n")
    _write_text(additional_data_dir / "Fin-Lit.txt", "token\nstale_lit\n")
    _write_text(additional_data_dir / "MW-Strong.txt", "token\n")
    _write_text(additional_data_dir / "MW-Weak.txt", "recognized\n")
    _write_text(additional_data_dir / "Harvard_IV_NEG_Inf.txt", "harvard\n")
    _write_text(
        additional_data_dir / "LM2011_MasterDictionary.txt",
        "\n".join(
            [
                "Word,Negative,Positive,Uncertainty,Litigious,Modal,Source",
                "TOKEN,2009,0,0,2009,1,12of12inf",
                "HARVARD,0,2009,0,0,0,10K_2008",
                "RECOGNIZED,0,0,2009,0,3,10K_2009",
                "STALE_NEG,2011,0,0,0,0,10K_2010",
                "STALE_LIT,0,0,0,2011,0,10K_2010",
            ]
        ),
    )
    _write_text(
        additional_data_dir / "Loughran-McDonald_MasterDictionary_1993-2024.csv",
        "\n".join(
            [
                "Word,Negative,Positive,Uncertainty,Litigious,Strong_Modal,Weak_Modal,Constraining,Complexity,Source",
                "TOKEN,2009,0,0,0,2009,0,0,0,12of12inf",
                "HARVARD,0,2012,0,0,0,0,0,0,10K_2008",
                "RECOGNIZED,0,0,2011,0,0,2011,0,0,10K_2009",
                "STALE_NEG,-2020,0,0,0,0,0,0,0,10K_2010",
                "STALE_LIT,0,0,0,2014,0,0,2014,0,10K_2010",
                "COMPLEXITY_ONLY,0,0,0,0,0,0,0,2024,10K_2010",
            ]
        ),
    )
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


def _effective_replication_dictionary_input_dir(additional_data_dir: Path) -> Path:
    families = runner.materialize_lm2011_dictionary_families(additional_data_dir)
    return families.replication.directory


def _write_valid_text_feature_artifact(
    path: Path,
    *,
    additional_data_dir: Path,
    stage_name: str,
    doc_id: str = "d1",
    dictionary_input_dir: Path | None = None,
) -> None:
    effective_dictionary_input_dir = (
        _effective_replication_dictionary_input_dir(additional_data_dir)
        if dictionary_input_dir is None
        else dictionary_input_dir
    )
    dictionary_inputs = runner.load_lm2011_dictionary_inputs(effective_dictionary_input_dir)
    spec = runner.TEXT_FEATURE_REUSE_SPECS[stage_name]
    normalized_dict = lm2011_text.normalize_lm2011_dictionary_lists(dictionary_inputs.dictionary_lists)
    signal_specs = lm2011_text._build_lm2011_signal_specs(
        normalized_dict=normalized_dict,
        harvard_negative_word_list=dictionary_inputs.harvard_negative_word_list,
    )
    schema = lm2011_text._feature_schema(
        include_item_id=spec.include_item_id,
        include_cleaning_policy_id=spec.include_cleaning_policy_id,
        raw_form_col="document_type_filename",
        token_count_col=spec.token_count_col,
        total_token_count_col=spec.total_token_count_col,
        signal_specs=signal_specs,
    )
    row: dict[str, object] = {}
    for name, dtype in schema.items():
        if name == "doc_id":
            row[name] = doc_id
        elif name == "cik_10":
            row[name] = "0000000001"
        elif name == "filing_date":
            row[name] = dt.date(1995, 1, 1)
        elif name == "document_type_filename":
            row[name] = "10-K"
        elif name == "normalized_form":
            row[name] = "10-K"
        elif name == "item_id":
            row[name] = "7"
        elif name == "cleaning_policy_id":
            row[name] = runner.RAW_ITEM_TEXT_CLEANING_POLICY_ID
        elif dtype == pl.Int32:
            row[name] = 1
        elif dtype == pl.Float64:
            row[name] = 0.1
        else:
            row[name] = "value"
    _write_parquet(path, pl.DataFrame([row], schema_overrides=schema))


def _write_text_feature_reuse_manifest(
    directory: Path,
    *,
    additional_data_dir: Path,
    full_10k_cleaning_contract: str = runner.DEFAULT_LM2011_FULL_10K_CLEANING_CONTRACT,
    raw_mda_cleaning_policy_id: str = runner.RAW_ITEM_TEXT_CLEANING_POLICY_ID,
    dictionary_input_dir: Path | None = None,
) -> None:
    effective_dictionary_input_dir = (
        _effective_replication_dictionary_input_dir(additional_data_dir)
        if dictionary_input_dir is None
        else dictionary_input_dir
    )
    dictionary_inputs = runner.load_lm2011_dictionary_inputs(effective_dictionary_input_dir)
    payload = {
        "runner_name": "lm2011_sample_post_refinitiv_runner",
        "config": {
            "full_10k_cleaning_contract": full_10k_cleaning_contract,
            "raw_mda_cleaning_policy_id": raw_mda_cleaning_policy_id,
        },
        "dictionary_inputs": dictionary_inputs.to_manifest_dict(),
    }
    _write_text(directory / runner.MANIFEST_FILENAME, json.dumps(payload))


def _build_extension_inputs(tmp_path: Path, additional_data_dir: Path) -> dict[str, Path]:
    event_panel_path = tmp_path / "extension_event_panel.parquet"
    company_history_path = tmp_path / "companyhistory.parquet"
    company_description_path = tmp_path / "companydescription.parquet"
    items_analysis_dir = tmp_path / "items_analysis"
    finbert_analysis_run_dir = tmp_path / "finbert_analysis"
    finbert_preprocessing_run_dir = tmp_path / "finbert_preprocess"
    cleaned_item_scopes_dir = finbert_preprocessing_run_dir / "cleaned_item_scopes" / "by_year"
    item_features_long_path = finbert_analysis_run_dir / "item_features_long.parquet"

    _write_parquet(
        event_panel_path,
        pl.DataFrame(
            {
                "doc_id": ["doc1", "doc2"],
                "gvkey_int": [1001, 1002],
                "KYPERMNO": [5001, 5002],
                "filing_date": [dt.date(2009, 3, 2), dt.date(2009, 5, 15)],
                "size_event": [100.0, 125.0],
                "bm_event": [0.8, 0.9],
                "share_turnover": [0.05, 0.08],
                "pre_ffalpha": [0.01, -0.02],
                "institutional_ownership": [None, 45.0],
                "nasdaq_dummy": [1, 0],
                "filing_period_excess_return": [0.02, -0.01],
                "abnormal_volume": [0.3, 0.4],
                "postevent_return_volatility": [0.04, 0.05],
            }
        ),
    )
    _write_parquet(
        company_history_path,
        pl.DataFrame(
            {
                "KYGVKEY": [1001, 1002],
                "HCHGDT": [dt.date(2000, 1, 1), dt.date(2000, 1, 1)],
                "HCHGENDDT": [None, None],
                "HSIC": [111, 3845],
            }
        ),
    )
    _write_parquet(
        company_description_path,
        pl.DataFrame({"KYGVKEY": [1001, 1002], "SIC": ["0111", "3845"]}),
    )
    _write_parquet(
        items_analysis_dir / "2009.parquet",
        pl.DataFrame(
            {
                "doc_id": ["doc1", "doc1", "doc2", "doc2"],
                "cik_10": ["0000000001", "0000000001", "0000000002", "0000000002"],
                "filing_date": [
                    dt.date(2009, 3, 2),
                    dt.date(2009, 3, 2),
                    dt.date(2009, 5, 15),
                    dt.date(2009, 5, 15),
                ],
                "document_type_filename": ["10-K", "10-K", "10-K", "10-K"],
                "item_id": ["7", "1A", "7", "1A"],
                "full_text": [
                    "loss loss gain recognized must",
                    "loss uncertain lawsuit recognized",
                    "loss gain recognized may",
                    "loss uncertain recognized may",
                ],
            }
        ),
    )
    _write_parquet(
        item_features_long_path,
        pl.DataFrame(
            {
                "doc_id": ["doc1", "doc1", "doc2", "doc2"],
                "filing_date": [
                    dt.date(2009, 3, 2),
                    dt.date(2009, 3, 2),
                    dt.date(2009, 5, 15),
                    dt.date(2009, 5, 15),
                ],
                "benchmark_item_code": ["item_7", "item_1a", "item_7", "item_1a"],
                "cleaning_policy_id": [
                    "item_text_clean_v2",
                    "item_text_clean_v2",
                    "item_text_clean_v2",
                    "item_text_clean_v2",
                ],
                "model_name": ["yiyanghkust/finbert-tone"] * 4,
                "model_version": ["rev-a"] * 4,
                "segment_policy_id": ["sentence_dataset_v1_finbert_token_512"] * 4,
                "finbert_segment_count": [3, 2, 4, 2],
                "finbert_token_count_512_sum": [30, 20, 40, 18],
                "finbert_neg_prob_lenw_mean": [0.6, 0.5, 0.2, 0.4],
                "finbert_pos_prob_lenw_mean": [0.1, 0.2, 0.5, 0.3],
                "finbert_neu_prob_lenw_mean": [0.3, 0.3, 0.3, 0.3],
                "finbert_net_negative_lenw_mean": [0.5, 0.3, -0.3, 0.1],
                "finbert_neg_dominant_share": [0.67, 0.5, 0.25, 0.5],
            }
        ),
    )
    _write_parquet(
        cleaned_item_scopes_dir / "2009.parquet",
        pl.DataFrame(
            {
                "doc_id": ["doc1", "doc1", "doc2", "doc2"],
                "cik_10": ["0000000001", "0000000001", "0000000002", "0000000002"],
                "filing_date": [
                    dt.date(2009, 3, 2),
                    dt.date(2009, 3, 2),
                    dt.date(2009, 5, 15),
                    dt.date(2009, 5, 15),
                ],
                "document_type_raw": ["10-K", "10-K", "10-K", "10-K"],
                "item_id": ["7", "1A", "7", "1A"],
                "text_scope": [
                    "item_7_mda",
                    "item_1a_risk_factors",
                    "item_7_mda",
                    "item_1a_risk_factors",
                ],
                "cleaning_policy_id": [
                    "item_text_clean_v2",
                    "item_text_clean_v2",
                    "item_text_clean_v2",
                    "item_text_clean_v2",
                ],
                "cleaned_text": [
                    "loss loss gain recognized must",
                    "loss uncertain lawsuit recognized",
                    "loss gain recognized may",
                    "loss uncertain recognized may",
                ],
            }
        ),
    )
    _write_text(finbert_analysis_run_dir / "run_manifest.json", json.dumps({"runner_name": "analysis"}))
    _write_text(finbert_preprocessing_run_dir / "run_manifest.json", json.dumps({"runner_name": "preprocess"}))

    return {
        "event_panel_path": event_panel_path,
        "company_history_path": company_history_path,
        "company_description_path": company_description_path,
        "items_analysis_dir": items_analysis_dir,
        "finbert_analysis_run_dir": finbert_analysis_run_dir,
        "finbert_preprocessing_run_dir": finbert_preprocessing_run_dir,
        "item_features_long_path": item_features_long_path,
        "cleaned_item_scopes_dir": cleaned_item_scopes_dir,
        "ff48_siccodes_path": additional_data_dir / "FF_Siccodes_48_Industries.txt",
    }


def _write_extension_shared_prereq_artifacts(
    output_dir: Path,
    *,
    event_panel_df: pl.DataFrame | None = None,
) -> runner._ExtensionSharedPrereqArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_backbone_path = output_dir / runner.EXTENSION_SHARED_PREREQ_FILENAMES["sample_backbone"]
    token_counts_path = output_dir / runner.EXTENSION_SHARED_PREREQ_FILENAMES["full_10k_token_counts"]
    event_screen_surface_path = output_dir / runner.EXTENSION_SHARED_PREREQ_FILENAMES["event_screen_surface"]
    event_panel_path = output_dir / runner.EXTENSION_SHARED_PREREQ_FILENAMES["event_panel"]
    if event_panel_df is None:
        event_panel_df = pl.DataFrame(
            {
                "doc_id": ["doc1", "doc2"],
                "gvkey_int": [1001, 1002],
                "KYPERMNO": [5001, 5002],
                "filing_date": [dt.date(2009, 3, 2), dt.date(2009, 5, 15)],
                "filing_trade_date": [dt.date(2009, 3, 2), dt.date(2009, 5, 15)],
                "pre_filing_trade_date": [dt.date(2009, 2, 27), dt.date(2009, 5, 14)],
                "size_event": [100.0, 125.0],
                "bm_event": [0.8, 0.9],
                "share_turnover": [0.05, 0.08],
                "pre_ffalpha": [0.01, -0.02],
                "institutional_ownership": [None, 45.0],
                "nasdaq_dummy": [1, 0],
                "filing_period_excess_return": [0.02, -0.01],
                "abnormal_volume": [0.3, 0.4],
                "postevent_return_volatility": [0.04, 0.05],
            }
        )
    _write_parquet(
        sample_backbone_path,
        pl.DataFrame(
            {
                "doc_id": ["doc1", "doc2"],
                "gvkey_int": [1001, 1002],
                "KYPERMNO": [5001, 5002],
                "filing_date": [dt.date(2009, 3, 2), dt.date(2009, 5, 15)],
                "normalized_form": ["10-K", "10-K"],
            }
        ),
    )
    _write_parquet(
        token_counts_path,
        pl.DataFrame(
            {
                "doc_id": ["doc1", "doc2"],
                "total_token_count_full_10k": [2500, 2500],
            }
        ),
    )
    _write_parquet(
        event_screen_surface_path,
        pl.DataFrame(
            {
                "doc_id": ["doc1", "doc2"],
                "gvkey_int": [1001, 1002],
                "KYPERMNO": [5001, 5002],
                "filing_date": [dt.date(2009, 3, 2), dt.date(2009, 5, 15)],
                "filing_trade_date": [dt.date(2009, 3, 2), dt.date(2009, 5, 15)],
                "pre_filing_trade_date": [dt.date(2009, 2, 27), dt.date(2009, 5, 14)],
                "size_event": [100.0, 125.0],
                "bm_event": [0.8, 0.9],
                "pre_filing_price": [10.0, 11.0],
                "share_turnover": [0.05, 0.08],
                "pre_ffalpha": [0.01, -0.02],
                "event_shares": [10.0, 10.0],
                "event_shrcd": [10, 10],
                "event_exchcd": [3, 1],
                "event_return_day_count": [4, 4],
                "event_volume_day_count": [4, 4],
                "pre_turnover_obs": [100, 100],
                "abnormal_volume_pre_obs": [60, 60],
                "pre_alpha_obs": [100, 100],
                "post_alpha_obs": [100, 100],
                "abnormal_volume": [0.3, 0.4],
                "postevent_return_volatility": [0.04, 0.05],
                "filing_period_excess_return": [0.02, -0.01],
                "nasdaq_dummy": [1, 0],
            }
        ),
    )
    _write_parquet(event_panel_path, event_panel_df)
    return runner._ExtensionSharedPrereqArtifacts(
        sample_backbone_path=sample_backbone_path,
        full_10k_token_counts_path=token_counts_path,
        event_screen_surface_path=event_screen_surface_path,
        event_panel_path=event_panel_path,
        row_counts={
            "sample_backbone": 2,
            "full_10k_token_counts": 2,
            "event_screen_surface": 2,
            "event_panel": int(event_panel_df.height),
        },
    )


def test_extension_shared_prereq_recompute_text_features_full_10k_does_not_force_downstream_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "lm2011_extension_shared_prereq_recompute"
    existing_artifacts = _write_extension_shared_prereq_artifacts(output_dir)
    raw_year_merged_dir = tmp_path / "raw_year_merged"
    raw_year_merged_dir.mkdir(parents=True, exist_ok=True)
    _write_parquet(raw_year_merged_dir / "2009.parquet", pl.DataFrame({"doc_id": ["doc1"]}))

    raw_parquet_paths = {
        "matched_clean_path": tmp_path / "raw_matched_clean.parquet",
        "filingdates_path": tmp_path / "raw_filingdates.parquet",
        "daily_panel_path": tmp_path / "raw_daily.parquet",
        "doc_ownership_path": tmp_path / "raw_doc_ownership.parquet",
        "annual_balance_sheet_path": tmp_path / "raw_annual_bs.parquet",
        "annual_income_statement_path": tmp_path / "raw_annual_is.parquet",
        "annual_period_descriptor_path": tmp_path / "raw_annual_pd.parquet",
        "annual_fiscal_market_path": tmp_path / "raw_annual_fm.parquet",
    }
    for path in raw_parquet_paths.values():
        _write_parquet(path, pl.DataFrame({"dummy": [1]}))
    ff_daily_csv_path = tmp_path / "raw_ff_daily.csv"
    _write_text(ff_daily_csv_path, "raw_date,mkt_rf,smb,hml,rf\n20090102,0.0,0.0,0.0,0.0\n")

    monkeypatch.setattr(
        runner,
        "_resolve_extension_shared_prereq_inputs",
        lambda *_args, **_kwargs: {
            "year_merged_dir": raw_year_merged_dir,
            **raw_parquet_paths,
            "ff_daily_csv_path": ff_daily_csv_path,
            "local_work_root": tmp_path / "local_work",
            "full_10k_cleaning_contract": "lm2011_paper",
            "full_10k_text_feature_batch_size": 4,
            "event_window_doc_batch_size": 50,
        },
    )
    monkeypatch.setattr(runner, "_prepare_lm2011_sec_backbone_input_lf", lambda lf: lf)
    monkeypatch.setattr(runner, "_prepare_lm2011_sec_input_lf", lambda lf: lf)
    monkeypatch.setattr(runner, "_filter_to_sample_doc_ids_lf", lambda lf, *_args, **_kwargs: lf)

    call_counts = {
        "sample_backbone": 0,
        "full_10k": 0,
        "event_screen_surface": 0,
        "event_panel": 0,
    }

    def _fake_build_sample_backbone(*_args: object, **_kwargs: object) -> pl.LazyFrame:
        call_counts["sample_backbone"] += 1
        return pl.DataFrame(
            {
                "doc_id": ["doc1"],
                "gvkey_int": [1001],
                "KYPERMNO": [5001],
                "filing_date": [dt.date(2009, 3, 2)],
                "normalized_form": ["10-K"],
            }
        ).lazy()

    def _fake_write_full_10k(*_args: object, output_path: Path, **_kwargs: object) -> None:
        call_counts["full_10k"] += 1
        _write_parquet(
            output_path,
            pl.DataFrame(
                {
                    "doc_id": ["doc1"],
                    "total_token_count_full_10k": [3000],
                }
            ),
        )

    def _fake_prepare_annual_inputs(*_args: object, **_kwargs: object) -> tuple[pl.LazyFrame, ...]:
        frame = pl.DataFrame({"dummy": [1]}).lazy()
        return frame, frame, frame, frame

    def _fake_build_annual_accounting_panel(*_args: object, **_kwargs: object) -> pl.LazyFrame:
        return pl.DataFrame({"dummy": [1]}).lazy()

    def _fake_load_ff_factors_daily(*_args: object, **_kwargs: object) -> pl.LazyFrame:
        return pl.DataFrame({"dummy": [1]}).lazy()

    def _unexpected_event_screen(*_args: object, **_kwargs: object) -> None:
        call_counts["event_screen_surface"] += 1
        raise AssertionError("event_screen_surface should be reused when recompute_event_screen_surface is false")

    def _unexpected_event_panel(*_args: object, **_kwargs: object) -> pl.LazyFrame:
        call_counts["event_panel"] += 1
        raise AssertionError("event_panel should be reused when recompute_event_panel is false")

    monkeypatch.setattr(runner, "build_lm2011_sample_backbone", _fake_build_sample_backbone)
    monkeypatch.setattr(runner, "write_lm2011_text_features_full_10k_parquet", _fake_write_full_10k)
    monkeypatch.setattr(runner, "_prepare_annual_accounting_inputs", _fake_prepare_annual_inputs)
    monkeypatch.setattr(runner, "build_annual_accounting_panel", _fake_build_annual_accounting_panel)
    monkeypatch.setattr(runner, "_load_ff_factors_daily_lf", _fake_load_ff_factors_daily)
    monkeypatch.setattr(runner.lm2011_pipeline, "write_lm2011_event_screen_surface_parquet", _unexpected_event_screen)
    monkeypatch.setattr(runner, "build_lm2011_event_panel", _unexpected_event_panel)

    artifacts = runner._build_or_reuse_extension_shared_prereqs(
        runner.LM2011ExtensionRunConfig(
            output_dir=output_dir,
            additional_data_dir=tmp_path / "additional_data",
            items_analysis_dir=tmp_path / "items_analysis",
            event_panel_path=None,
            company_history_path=tmp_path / "companyhistory.parquet",
            company_description_path=tmp_path / "companydescription.parquet",
            ff48_siccodes_path=tmp_path / "ff48.txt",
            year_merged_dir=raw_year_merged_dir,
            matched_clean_path=raw_parquet_paths["matched_clean_path"],
            filingdates_path=raw_parquet_paths["filingdates_path"],
            daily_panel_path=raw_parquet_paths["daily_panel_path"],
            doc_ownership_path=raw_parquet_paths["doc_ownership_path"],
            annual_balance_sheet_path=raw_parquet_paths["annual_balance_sheet_path"],
            annual_income_statement_path=raw_parquet_paths["annual_income_statement_path"],
            annual_period_descriptor_path=raw_parquet_paths["annual_period_descriptor_path"],
            annual_fiscal_market_path=raw_parquet_paths["annual_fiscal_market_path"],
            ff_daily_csv_path=ff_daily_csv_path,
            local_work_root=tmp_path / "local_work",
            recompute_text_features_full_10k=True,
            recompute_text_features_mda=True,
            recompute_event_screen_surface=False,
            recompute_event_panel=False,
        ),
        output_dir=output_dir,
        dictionary_inputs=SimpleNamespace(
            dictionary_lists={"negative": ["loss"]},
            harvard_negative_word_list=["decline"],
            master_dictionary_words={"loss", "decline"},
        ),
    )

    assert artifacts is not None
    assert call_counts["sample_backbone"] == 1
    assert call_counts["full_10k"] == 1
    assert call_counts["event_screen_surface"] == 0
    assert call_counts["event_panel"] == 0
    assert pl.read_parquet(artifacts.full_10k_token_counts_path).get_column(
        "total_token_count_full_10k"
    ).to_list() == [3000]
    assert pl.read_parquet(existing_artifacts.event_screen_surface_path).height == 2
    assert pl.read_parquet(existing_artifacts.event_panel_path).height == 2


def test_extension_pipeline_writes_manifest_and_fully_enumerated_results(tmp_path: Path) -> None:
    _, _, additional_data_dir, _ = _build_temp_layout(tmp_path)
    extension_inputs = _build_extension_inputs(tmp_path, additional_data_dir)
    output_dir = tmp_path / "lm2011_extension"

    exit_code = runner.run_lm2011_extension_pipeline(
        runner.LM2011ExtensionRunConfig(
            output_dir=output_dir,
            additional_data_dir=additional_data_dir,
            items_analysis_dir=extension_inputs["items_analysis_dir"],
            event_panel_path=extension_inputs["event_panel_path"],
            company_history_path=extension_inputs["company_history_path"],
            company_description_path=extension_inputs["company_description_path"],
            ff48_siccodes_path=extension_inputs["ff48_siccodes_path"],
            finbert_item_features_long_path=extension_inputs["item_features_long_path"],
            finbert_analysis_run_dir=extension_inputs["finbert_analysis_run_dir"],
            finbert_preprocessing_run_dir=extension_inputs["finbert_preprocessing_run_dir"],
            require_cleaned_scope_match=True,
            run_id="unit_test_extension",
        )
    )

    assert exit_code == 0

    manifest_path = output_dir / runner.EXTENSION_MANIFEST_FILENAME
    results_path = output_dir / runner.EXTENSION_STAGE_ARTIFACT_FILENAMES["extension_results"]
    sample_loss_path = output_dir / runner.EXTENSION_STAGE_ARTIFACT_FILENAMES["extension_sample_loss"]
    fit_quarterly_path = output_dir / runner.EXTENSION_STAGE_ARTIFACT_FILENAMES["extension_fit_quarterly"]
    fit_difference_path = output_dir / runner.EXTENSION_STAGE_ARTIFACT_FILENAMES["extension_fit_difference_quarterly"]
    fit_summary_path = output_dir / runner.EXTENSION_STAGE_ARTIFACT_FILENAMES["extension_fit_summary"]
    fit_comparisons_path = output_dir / runner.EXTENSION_STAGE_ARTIFACT_FILENAMES["extension_fit_comparisons"]
    fit_skips_path = output_dir / runner.EXTENSION_STAGE_ARTIFACT_FILENAMES["extension_fit_skipped_quarters"]
    fit_skips_csv_path = output_dir / "lm2011_extension_fit_skipped_quarters.csv"
    dictionary_surface_path = output_dir / runner.EXTENSION_STAGE_ARTIFACT_FILENAMES["extension_dictionary_surface"]
    finbert_surface_path = output_dir / runner.EXTENSION_STAGE_ARTIFACT_FILENAMES["extension_finbert_surface"]
    analysis_panel_path = output_dir / runner.EXTENSION_STAGE_ARTIFACT_FILENAMES["extension_analysis_panel"]

    for path in (
        manifest_path,
        results_path,
        sample_loss_path,
        fit_quarterly_path,
        fit_difference_path,
        fit_summary_path,
        fit_comparisons_path,
        fit_skips_path,
        fit_skips_csv_path,
        dictionary_surface_path,
        finbert_surface_path,
        analysis_panel_path,
    ):
        assert path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_status"] == "completed"
    assert manifest["failed_stage"] is None
    assert manifest["config"]["dictionary_source_mode"] == runner.EXTENSION_DICTIONARY_SOURCE_PREFER_CLEANED
    assert manifest["config"]["effective_dictionary_source_mode"] == runner.EXTENSION_DICTIONARY_SOURCE_PREFER_CLEANED
    assert manifest["config"]["recompute_text_features_full_10k"] is False
    assert manifest["config"]["recompute_text_features_mda"] is False
    assert manifest["config"]["recompute_event_screen_surface"] is False
    assert manifest["config"]["recompute_event_panel"] is False
    assert manifest["resolved_inputs"]["finbert_item_features_long_path"] == str(
        extension_inputs["item_features_long_path"].resolve()
    )
    assert manifest["resolved_inputs"]["finbert_cleaned_item_scopes_dir"] == str(
        extension_inputs["cleaned_item_scopes_dir"].resolve()
    )
    assert manifest["resolved_inputs"]["effective_event_panel_path"] == str(
        extension_inputs["event_panel_path"].resolve()
    )
    assert manifest["shared_prereq_artifacts"] == {}
    assert manifest["shared_prereq_row_counts"] == {}
    assert manifest["stages"]["extension_dictionary_surface"]["status"] == "generated"
    assert manifest["stages"]["extension_finbert_surface"]["status"] == "generated"
    assert manifest["stages"]["extension_fit_quarterly"]["status"] in {"generated", "generated_empty"}
    assert manifest["stages"]["extension_fit_difference_quarterly"]["status"] in {"generated", "generated_empty"}
    assert manifest["stages"]["extension_fit_summary"]["status"] == "generated"
    assert manifest["stages"]["extension_fit_comparisons"]["status"] == "generated"
    assert manifest["stages"]["extension_fit_skipped_quarters"]["status"] in {"generated", "generated_empty"}
    assert manifest["stages"]["extension_fit_skipped_quarters"]["extra_artifacts"]["csv"] == str(
        fit_skips_csv_path.resolve()
    )
    assert manifest["stages"]["extension_results"]["status"] == "generated"
    assert "extension_fit_summary" in manifest["row_counts"]
    assert "extension_fit_comparisons" in manifest["row_counts"]
    assert manifest["row_counts"]["extension_results"] >= 18

    panel = pl.read_parquet(analysis_panel_path).sort("doc_id", "text_scope")
    assert panel.get_column("sample_window").unique().to_list() == ["2009_2024"]
    assert set(panel.get_column("text_scope").unique().to_list()) == {
        "item_7_mda",
        "item_1a_risk_factors",
    }

    finbert_surface = pl.read_parquet(finbert_surface_path).sort("doc_id", "text_scope")
    assert set(finbert_surface.get_column("text_scope").unique().to_list()) == {
        "item_7_mda",
        "item_1a_risk_factors",
    }
    assert "item_1" not in finbert_surface.get_column("text_scope").to_list()

    results = pl.read_parquet(results_path)
    fit_summary = pl.read_parquet(fit_summary_path)
    fit_comparisons = pl.read_parquet(fit_comparisons_path)
    fit_skips = pl.read_parquet(fit_skips_path)
    assert "estimator_status" in results.columns
    assert results.get_column("estimator_status").null_count() == 0
    assert {
        "equal_quarter_avg_raw_r2",
        "equal_quarter_avg_adj_r2",
        "signal_inputs",
    }.issubset(set(fit_summary.columns))
    assert {
        "left_signal_inputs",
        "right_signal_inputs",
        "equal_quarter_avg_delta_adj_r2",
    }.issubset(set(fit_comparisons.columns))
    assert "signal_inputs" in fit_skips.columns
    observed_grid = {
        (
            row["text_scope"],
            row["specification_name"],
            row["control_set_id"],
            row["outcome_name"],
        )
        for row in results.select(
            "text_scope",
            "specification_name",
            "control_set_id",
            "outcome_name",
        ).unique().to_dicts()
    }
    expected_grid = {
        (text_scope, specification_name, control_set_id, "filing_period_excess_return")
        for text_scope in ("item_7_mda", "item_1a_risk_factors")
        for specification_name in (
            "dictionary_only",
            "finbert_only",
            "dictionary_finbert_joint",
        )
        for control_set_id in ("C0", "C1", "C2")
    }
    assert observed_grid == expected_grid


def test_extension_pipeline_prefers_built_shared_event_panel_when_raw_prereqs_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, additional_data_dir, _ = _build_temp_layout(tmp_path)
    extension_inputs = _build_extension_inputs(tmp_path, additional_data_dir)
    output_dir = tmp_path / "lm2011_extension_shared_prereqs"
    captured: dict[str, object] = {}

    def _fake_shared_prereqs(
        cfg: runner.LM2011ExtensionRunConfig,
        *,
        output_dir: Path,
        dictionary_inputs: object,
    ) -> runner._ExtensionSharedPrereqArtifacts:
        captured["year_merged_dir"] = cfg.year_merged_dir
        captured["matched_clean_path"] = cfg.matched_clean_path
        return _write_extension_shared_prereq_artifacts(output_dir)

    monkeypatch.setattr(
        runner,
        "_build_or_reuse_extension_shared_prereqs",
        _fake_shared_prereqs,
    )

    exit_code = runner.run_lm2011_extension_pipeline(
        runner.LM2011ExtensionRunConfig(
            output_dir=output_dir,
            additional_data_dir=additional_data_dir,
            items_analysis_dir=extension_inputs["items_analysis_dir"],
            event_panel_path=None,
            company_history_path=extension_inputs["company_history_path"],
            company_description_path=extension_inputs["company_description_path"],
            ff48_siccodes_path=extension_inputs["ff48_siccodes_path"],
            year_merged_dir=tmp_path / "raw_year_merged",
            matched_clean_path=tmp_path / "raw_matched_clean.parquet",
            filingdates_path=tmp_path / "raw_filingdates.parquet",
            daily_panel_path=tmp_path / "raw_daily.parquet",
            doc_ownership_path=tmp_path / "raw_doc_ownership.parquet",
            annual_balance_sheet_path=tmp_path / "raw_annual_bs.parquet",
            annual_income_statement_path=tmp_path / "raw_annual_is.parquet",
            annual_period_descriptor_path=tmp_path / "raw_annual_pd.parquet",
            annual_fiscal_market_path=tmp_path / "raw_annual_fm.parquet",
            ff_daily_csv_path=tmp_path / "raw_ff_daily.csv",
            local_work_root=tmp_path / "raw_local_work",
            full_10k_cleaning_contract="lm2011_paper",
            full_10k_text_feature_batch_size=4,
            event_window_doc_batch_size=50,
            finbert_item_features_long_path=extension_inputs["item_features_long_path"],
            finbert_analysis_run_dir=extension_inputs["finbert_analysis_run_dir"],
            finbert_preprocessing_run_dir=extension_inputs["finbert_preprocessing_run_dir"],
            require_cleaned_scope_match=True,
            run_id="unit_test_extension_shared_prereq",
        )
    )

    assert exit_code == 0
    assert captured["year_merged_dir"] == tmp_path / "raw_year_merged"
    assert captured["matched_clean_path"] == tmp_path / "raw_matched_clean.parquet"
    manifest = json.loads((output_dir / runner.EXTENSION_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    expected_effective_path = (
        output_dir / runner.EXTENSION_SHARED_PREREQ_FILENAMES["event_panel"]
    ).resolve()
    assert manifest["resolved_inputs"]["event_panel_path"] is None
    assert manifest["resolved_inputs"]["effective_event_panel_path"] == str(expected_effective_path)
    assert manifest["shared_prereq_artifacts"]["event_panel"] == str(expected_effective_path)
    assert manifest["shared_prereq_row_counts"]["event_panel"] == 2
    analysis_panel = pl.read_parquet(
        output_dir / runner.EXTENSION_STAGE_ARTIFACT_FILENAMES["extension_analysis_panel"]
    )
    assert analysis_panel.height > 0
    assert set(analysis_panel.get_column("doc_id").unique().to_list()) == {"doc1", "doc2"}


def test_extension_dictionary_family_comparison_pipeline_writes_root_and_family_artifacts(
    tmp_path: Path,
) -> None:
    _, _, additional_data_dir, _ = _build_temp_layout(tmp_path)
    extension_inputs = _build_extension_inputs(tmp_path, additional_data_dir)
    output_dir = tmp_path / "lm2011_extension_family_compare"

    exit_code = runner.run_lm2011_extension_dictionary_family_comparison_pipeline(
        runner.LM2011ExtensionRunConfig(
            output_dir=output_dir,
            additional_data_dir=additional_data_dir,
            items_analysis_dir=extension_inputs["items_analysis_dir"],
            event_panel_path=extension_inputs["event_panel_path"],
            company_history_path=extension_inputs["company_history_path"],
            company_description_path=extension_inputs["company_description_path"],
            ff48_siccodes_path=extension_inputs["ff48_siccodes_path"],
            finbert_item_features_long_path=extension_inputs["item_features_long_path"],
            finbert_analysis_run_dir=extension_inputs["finbert_analysis_run_dir"],
            finbert_preprocessing_run_dir=extension_inputs["finbert_preprocessing_run_dir"],
            require_cleaned_scope_match=True,
            run_id="unit_test_extension_family_compare",
        )
    )

    assert exit_code == 0
    manifest_path = output_dir / runner.EXTENSION_MANIFEST_FILENAME
    results_path = output_dir / runner.EXTENSION_STAGE_ARTIFACT_FILENAMES["extension_results"]
    sample_loss_path = output_dir / runner.EXTENSION_STAGE_ARTIFACT_FILENAMES["extension_sample_loss"]
    fit_summary_path = output_dir / runner.EXTENSION_STAGE_ARTIFACT_FILENAMES["extension_fit_summary"]
    fit_skips_csv_path = output_dir / "lm2011_extension_fit_skipped_quarters.csv"

    for path in (
        manifest_path,
        results_path,
        sample_loss_path,
        fit_summary_path,
        fit_skips_csv_path,
        output_dir / "replication" / runner.EXTENSION_MANIFEST_FILENAME,
        output_dir / "extended" / runner.EXTENSION_MANIFEST_FILENAME,
    ):
        assert path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["runner_name"] == "lm2011_extension_dictionary_family_comparison_runner"
    assert manifest["run_status"] == "completed"
    assert manifest["config"]["dictionary_family_comparison"] is True
    assert manifest["config"]["dictionary_families"] == ["replication", "extended"]
    assert set(manifest["family_runs"]) == {"replication", "extended"}
    assert manifest["family_runs"]["replication"]["run_status"] == "completed"
    assert manifest["family_runs"]["extended"]["run_status"] == "completed"

    results = pl.read_parquet(results_path)
    sample_loss = pl.read_parquet(sample_loss_path)
    fit_summary = pl.read_parquet(fit_summary_path)
    assert set(results.get_column("dictionary_family_source").unique().to_list()) == {"replication", "extended"}
    assert set(sample_loss.get_column("dictionary_family_source").unique().to_list()) == {"replication", "extended"}
    assert set(fit_summary.get_column("dictionary_family_source").unique().to_list()) == {"replication", "extended"}
    assert set(
        results.filter(pl.col("specification_name") == "dictionary_only")
        .get_column("dictionary_family_source")
        .unique()
        .to_list()
    ) == {"replication", "extended"}


def test_extension_dictionary_family_comparison_builds_shared_prereqs_once_at_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, additional_data_dir, _ = _build_temp_layout(tmp_path)
    extension_inputs = _build_extension_inputs(tmp_path, additional_data_dir)
    output_dir = tmp_path / "lm2011_extension_family_compare_shared"
    calls: list[tuple[Path, Path | None]] = []

    def _fake_shared_prereqs(
        cfg: runner.LM2011ExtensionRunConfig,
        *,
        output_dir: Path,
        dictionary_inputs: object,
    ) -> runner._ExtensionSharedPrereqArtifacts | None:
        calls.append((output_dir, cfg.year_merged_dir))
        if cfg.year_merged_dir is None:
            return None
        return _write_extension_shared_prereq_artifacts(output_dir)

    monkeypatch.setattr(
        runner,
        "_build_or_reuse_extension_shared_prereqs",
        _fake_shared_prereqs,
    )

    exit_code = runner.run_lm2011_extension_dictionary_family_comparison_pipeline(
        runner.LM2011ExtensionRunConfig(
            output_dir=output_dir,
            additional_data_dir=additional_data_dir,
            items_analysis_dir=extension_inputs["items_analysis_dir"],
            event_panel_path=extension_inputs["event_panel_path"],
            company_history_path=extension_inputs["company_history_path"],
            company_description_path=extension_inputs["company_description_path"],
            ff48_siccodes_path=extension_inputs["ff48_siccodes_path"],
            year_merged_dir=tmp_path / "raw_year_merged",
            matched_clean_path=tmp_path / "raw_matched_clean.parquet",
            filingdates_path=tmp_path / "raw_filingdates.parquet",
            daily_panel_path=tmp_path / "raw_daily.parquet",
            doc_ownership_path=tmp_path / "raw_doc_ownership.parquet",
            annual_balance_sheet_path=tmp_path / "raw_annual_bs.parquet",
            annual_income_statement_path=tmp_path / "raw_annual_is.parquet",
            annual_period_descriptor_path=tmp_path / "raw_annual_pd.parquet",
            annual_fiscal_market_path=tmp_path / "raw_annual_fm.parquet",
            ff_daily_csv_path=tmp_path / "raw_ff_daily.csv",
            local_work_root=tmp_path / "raw_local_work",
            full_10k_cleaning_contract="lm2011_paper",
            full_10k_text_feature_batch_size=4,
            event_window_doc_batch_size=50,
            finbert_item_features_long_path=extension_inputs["item_features_long_path"],
            finbert_analysis_run_dir=extension_inputs["finbert_analysis_run_dir"],
            finbert_preprocessing_run_dir=extension_inputs["finbert_preprocessing_run_dir"],
            require_cleaned_scope_match=True,
            run_id="unit_test_extension_family_shared_prereq",
        )
    )

    assert exit_code == 0
    assert sum(1 for _, year_dir in calls if year_dir is not None) == 1
    root_shared_event_panel_path = (
        output_dir / runner.EXTENSION_SHARED_PREREQ_FILENAMES["event_panel"]
    ).resolve()
    root_manifest = json.loads((output_dir / runner.EXTENSION_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    replication_manifest = json.loads(
        (output_dir / "replication" / runner.EXTENSION_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    extended_manifest = json.loads(
        (output_dir / "extended" / runner.EXTENSION_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert root_manifest["resolved_inputs"]["effective_event_panel_path"] == str(root_shared_event_panel_path)
    assert root_manifest["shared_prereq_artifacts"]["event_panel"] == str(root_shared_event_panel_path)
    assert replication_manifest["resolved_inputs"]["effective_event_panel_path"] == str(root_shared_event_panel_path)
    assert extended_manifest["resolved_inputs"]["effective_event_panel_path"] == str(root_shared_event_panel_path)


def test_extension_pipeline_strict_cleaned_scope_match_fails_closed(tmp_path: Path) -> None:
    _, _, additional_data_dir, _ = _build_temp_layout(tmp_path)
    extension_inputs = _build_extension_inputs(tmp_path, additional_data_dir)
    output_dir = tmp_path / "lm2011_extension"

    with pytest.raises(
        FileNotFoundError,
        match="cleaned_item_scopes_dir",
    ):
        runner.run_lm2011_extension_pipeline(
            runner.LM2011ExtensionRunConfig(
                output_dir=output_dir,
                additional_data_dir=additional_data_dir,
                items_analysis_dir=extension_inputs["items_analysis_dir"],
                event_panel_path=extension_inputs["event_panel_path"],
                company_history_path=extension_inputs["company_history_path"],
                company_description_path=extension_inputs["company_description_path"],
                ff48_siccodes_path=extension_inputs["ff48_siccodes_path"],
                finbert_item_features_long_path=extension_inputs["item_features_long_path"],
                require_cleaned_scope_match=True,
                run_id="unit_test_extension_fail_closed",
            )
        )

    manifest = json.loads((output_dir / runner.EXTENSION_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["run_status"] == "failed"
    assert manifest["failed_stage"] == "configuration_validation"
    assert manifest["stages"]["configuration_validation"]["status"] == "failed"
    assert manifest["stages"]["configuration_validation"]["artifact_path"] is None
    assert "extension_finbert_surface" not in manifest["stages"]


def test_extension_pipeline_relaxed_mode_fails_fast_without_cleaned_scope_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, additional_data_dir, _ = _build_temp_layout(tmp_path)
    extension_inputs = _build_extension_inputs(tmp_path, additional_data_dir)
    output_dir = tmp_path / "lm2011_extension"

    def _unexpected_analysis_panel(*_: object, **__: object) -> pl.LazyFrame:
        raise AssertionError("extension_analysis_panel should not run in relaxed fallback fail-fast mode")

    monkeypatch.setattr(runner, "build_lm2011_extension_analysis_panel", _unexpected_analysis_panel)

    with pytest.raises(
        ValueError,
        match="relaxed raw-item fallback is not supported",
    ):
        runner.run_lm2011_extension_pipeline(
            runner.LM2011ExtensionRunConfig(
                output_dir=output_dir,
                additional_data_dir=additional_data_dir,
                items_analysis_dir=extension_inputs["items_analysis_dir"],
                event_panel_path=extension_inputs["event_panel_path"],
                company_history_path=extension_inputs["company_history_path"],
                company_description_path=extension_inputs["company_description_path"],
                ff48_siccodes_path=extension_inputs["ff48_siccodes_path"],
                finbert_item_features_long_path=extension_inputs["item_features_long_path"],
                require_cleaned_scope_match=False,
                run_id="unit_test_extension_relaxed_fail_fast",
            )
        )

    manifest = json.loads((output_dir / runner.EXTENSION_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["run_status"] == "failed"
    assert manifest["failed_stage"] == "configuration_validation"
    assert manifest["stages"]["configuration_validation"]["status"] == "failed"
    assert manifest["stages"]["configuration_validation"]["artifact_path"] is None
    assert "extension_dictionary_surface" not in manifest["stages"]


def test_extension_dictionary_surface_prefers_cleaned_scopes_when_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, additional_data_dir, _ = _build_temp_layout(tmp_path)
    extension_inputs = _build_extension_inputs(tmp_path, additional_data_dir)
    dictionary_inputs = runner.load_lm2011_dictionary_inputs(additional_data_dir)
    captured: dict[str, object] = {"used_cleaned": False}

    def _cleaned_builder(*args: object, **kwargs: object) -> pl.LazyFrame:
        captured["used_cleaned"] = True
        cleaned_lf = args[0] if args else kwargs["cleaned_scopes_lf"]
        cleaned = cleaned_lf.collect().sort("doc_id", "text_scope")
        assert set(cleaned.get_column("text_scope").to_list()) == {
            "item_7_mda",
            "item_1a_risk_factors",
        }
        return pl.DataFrame(
            {
                "doc_id": ["doc1", "doc1"],
                "cik_10": ["0000000001", "0000000001"],
                "filing_date": [dt.date(2009, 3, 2), dt.date(2009, 3, 2)],
                "text_scope": ["item_7_mda", "item_1a_risk_factors"],
                "cleaning_policy_id": ["item_text_clean_v2", "item_text_clean_v2"],
                "dictionary_family": ["lm2011_frozen", "lm2011_frozen"],
                "total_token_count": [5, 4],
                "token_count": [5, 4],
                "lm_negative_tfidf": [0.1, 0.2],
            }
        ).lazy()

    def _raw_builder(*_: object, **__: object) -> pl.LazyFrame:
        raise AssertionError("raw dictionary rescoring should not run when cleaned scopes are available")

    monkeypatch.setattr(
        runner,
        "build_lm2011_extension_dictionary_features_from_cleaned_scopes",
        _cleaned_builder,
    )
    monkeypatch.setattr(
        runner,
        "build_lm2011_extension_dictionary_features",
        _raw_builder,
    )

    cfg = runner.LM2011ExtensionRunConfig(
        output_dir=tmp_path / "lm2011_extension",
        additional_data_dir=additional_data_dir,
        items_analysis_dir=extension_inputs["items_analysis_dir"],
        event_panel_path=extension_inputs["event_panel_path"],
        company_history_path=extension_inputs["company_history_path"],
        company_description_path=extension_inputs["company_description_path"],
        ff48_siccodes_path=extension_inputs["ff48_siccodes_path"],
        finbert_item_features_long_path=extension_inputs["item_features_long_path"],
    )

    out = runner._build_extension_dictionary_surface_lf(
        cfg,
        cleaned_item_scopes_dir=extension_inputs["cleaned_item_scopes_dir"],
        event_doc_ids_lf=pl.DataFrame({"doc_id": ["doc1", "doc2"]}).lazy(),
        dictionary_inputs=dictionary_inputs,
    ).collect()

    assert captured["used_cleaned"] is True
    assert set(out.get_column("text_scope").to_list()) == {
        "item_7_mda",
        "item_1a_risk_factors",
    }


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
    assert args.text_features_full_10k_path is None
    assert args.text_features_mda_path is None
    assert args.recompute_text_features_full_10k is False
    assert args.recompute_text_features_mda is False
    assert args.recompute_event_screen_surface is False
    assert args.recompute_event_panel is False
    assert args.recompute_regression_tables is False
    assert args.doc_ownership_path is None
    assert args.doc_analyst_selected_path is None
    assert args.event_window_doc_batch_size == runner.DEFAULT_LM2011_EVENT_WINDOW_DOC_BATCH_SIZE
    assert args.local_work_root == runner.DEFAULT_LOCAL_WORK_ROOT
    assert args.print_ram_stats is False
    assert args.ram_log_interval_batches == runner.DEFAULT_RAM_LOG_INTERVAL_BATCHES


def test_no_ownership_stage_names_are_registered_and_block_text_feature_reuse() -> None:
    no_ownership_stages = {
        "table_iv_results_no_ownership",
        "table_v_results_no_ownership",
        "table_vi_results_no_ownership",
        "table_viii_results_no_ownership",
        "table_ia_i_results_no_ownership",
    }

    assert no_ownership_stages.issubset(runner.STAGE_ARTIFACT_FILENAMES)
    assert no_ownership_stages.issubset(runner.LM2011_ALL_STAGE_NAMES)
    assert no_ownership_stages.issubset(runner.FINAL_REGRESSION_TABLE_STAGE_NAMES)
    assert no_ownership_stages.issubset(runner.QUARTERLY_REGRESSION_TABLE_STAGE_NAMES)
    assert {
        "table_iv_results_no_ownership",
        "table_vi_results_no_ownership",
        "table_viii_results_no_ownership",
        "table_ia_i_results_no_ownership",
    }.issubset(runner.TEXT_FEATURE_REUSE_SPECS["text_features_full_10k"].blocking_stage_names)
    assert "table_v_results_no_ownership" in runner.TEXT_FEATURE_REUSE_SPECS["text_features_mda"].blocking_stage_names


def test_resolve_paths_threads_recompute_flags(tmp_path: Path) -> None:
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
                "--recompute-event-screen-surface",
                "--recompute-event-panel",
                "--recompute-regression-tables",
            ]
        )
    )

    assert paths.recompute_event_screen_surface is True
    assert paths.recompute_event_panel is True
    assert paths.recompute_regression_tables is True


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


def test_resolve_paths_explicit_doc_artifact_paths_take_precedence(tmp_path: Path) -> None:
    sample_root, upstream_run_root, additional_data_dir, output_dir = _build_temp_layout(tmp_path)
    explicit_doc_ownership = tmp_path / "explicit" / "doc_ownership.parquet"
    explicit_doc_analyst = tmp_path / "explicit" / "doc_analyst_selected.parquet"
    _write_parquet(explicit_doc_ownership, pl.DataFrame({"doc_id": ["explicit"]}))
    _write_parquet(explicit_doc_analyst, pl.DataFrame({"doc_id": ["explicit"]}))

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
                "--doc-ownership-path",
                str(explicit_doc_ownership),
                "--doc-analyst-selected-path",
                str(explicit_doc_analyst),
            ]
        )
    )

    assert paths.doc_ownership_path == explicit_doc_ownership.resolve()
    assert paths.doc_analyst_selected_path == explicit_doc_analyst.resolve()


def test_resolve_paths_auto_detects_canonical_text_feature_artifacts(tmp_path: Path) -> None:
    sample_root, upstream_run_root, additional_data_dir, output_dir = _build_temp_layout(tmp_path)
    _write_valid_text_feature_artifact(
        output_dir / runner.STAGE_ARTIFACT_FILENAMES["text_features_full_10k"],
        additional_data_dir=additional_data_dir,
        stage_name="text_features_full_10k",
    )
    _write_valid_text_feature_artifact(
        output_dir / runner.STAGE_ARTIFACT_FILENAMES["text_features_mda"],
        additional_data_dir=additional_data_dir,
        stage_name="text_features_mda",
    )

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

    assert paths.text_features_full_10k_path == (
        output_dir / runner.STAGE_ARTIFACT_FILENAMES["text_features_full_10k"]
    ).resolve()
    assert paths.text_features_full_10k_path_is_explicit is False
    assert paths.text_features_mda_path == (
        output_dir / runner.STAGE_ARTIFACT_FILENAMES["text_features_mda"]
    ).resolve()
    assert paths.text_features_mda_path_is_explicit is False


def test_resolve_paths_explicit_text_feature_artifact_paths_take_precedence(tmp_path: Path) -> None:
    sample_root, upstream_run_root, additional_data_dir, output_dir = _build_temp_layout(tmp_path)
    explicit_full_10k = tmp_path / "explicit" / "full_10k.parquet"
    explicit_mda = tmp_path / "explicit" / "mda.parquet"
    _write_valid_text_feature_artifact(
        explicit_full_10k,
        additional_data_dir=additional_data_dir,
        stage_name="text_features_full_10k",
    )
    _write_valid_text_feature_artifact(
        explicit_mda,
        additional_data_dir=additional_data_dir,
        stage_name="text_features_mda",
    )
    _write_valid_text_feature_artifact(
        output_dir / runner.STAGE_ARTIFACT_FILENAMES["text_features_full_10k"],
        additional_data_dir=additional_data_dir,
        stage_name="text_features_full_10k",
    )

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
                "--text-features-full-10k-path",
                str(explicit_full_10k),
                "--text-features-mda-path",
                str(explicit_mda),
            ]
        )
    )

    assert paths.text_features_full_10k_path == explicit_full_10k.resolve()
    assert paths.text_features_full_10k_path_is_explicit is True
    assert paths.text_features_mda_path == explicit_mda.resolve()
    assert paths.text_features_mda_path_is_explicit is True


def test_resolve_paths_recompute_text_feature_flags_disable_auto_reuse(tmp_path: Path) -> None:
    sample_root, upstream_run_root, additional_data_dir, output_dir = _build_temp_layout(tmp_path)
    _write_valid_text_feature_artifact(
        output_dir / runner.STAGE_ARTIFACT_FILENAMES["text_features_full_10k"],
        additional_data_dir=additional_data_dir,
        stage_name="text_features_full_10k",
    )
    _write_valid_text_feature_artifact(
        output_dir / runner.STAGE_ARTIFACT_FILENAMES["text_features_mda"],
        additional_data_dir=additional_data_dir,
        stage_name="text_features_mda",
    )

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
                "--recompute-text-features-full-10k",
                "--recompute-text-features-mda",
            ]
        )
    )

    assert paths.text_features_full_10k_path is None
    assert paths.text_features_full_10k_path_is_explicit is False
    assert paths.text_features_mda_path is None
    assert paths.text_features_mda_path_is_explicit is False


def test_resolve_paths_recompute_text_feature_flags_conflict_with_explicit_paths(tmp_path: Path) -> None:
    sample_root, upstream_run_root, additional_data_dir, output_dir = _build_temp_layout(tmp_path)
    explicit_full_10k = tmp_path / "explicit" / "full_10k.parquet"
    explicit_mda = tmp_path / "explicit" / "mda.parquet"
    _write_valid_text_feature_artifact(
        explicit_full_10k,
        additional_data_dir=additional_data_dir,
        stage_name="text_features_full_10k",
    )
    _write_valid_text_feature_artifact(
        explicit_mda,
        additional_data_dir=additional_data_dir,
        stage_name="text_features_mda",
    )

    with pytest.raises(
        ValueError,
        match="--recompute-text-features-full-10k cannot be combined with --text-features-full-10k-path",
    ):
        runner._resolve_paths(
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
                    "--text-features-full-10k-path",
                    str(explicit_full_10k),
                    "--recompute-text-features-full-10k",
                ]
            )
        )

    with pytest.raises(
        ValueError,
        match="--recompute-text-features-mda cannot be combined with --text-features-mda-path",
    ):
        runner._resolve_paths(
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
                    "--text-features-mda-path",
                    str(explicit_mda),
                    "--recompute-text-features-mda",
                ]
            )
        )


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


def test_write_quarterly_regression_table_stage_records_diagnostics_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    manifest_path = tmp_path / "manifest.json"
    manifest: dict[str, object] = {
        "roots": {"output_dir": str(output_dir)},
        "artifacts": {},
        "row_counts": {},
        "stages": {},
    }
    bundle = _quarterly_bundle(
        results_df=pl.DataFrame({"table_id": ["table_iv_full_10k"], "estimate": [1.0]}),
        skipped_quarters_df=_skipped_quarter_diagnostics_df(),
    )

    runner._write_quarterly_regression_table_stage(
        manifest,
        manifest_path=manifest_path,
        output_dir=output_dir,
        stage_name="table_iv_results",
        bundle=bundle,
    )

    stage = manifest["stages"]["table_iv_results"]  # type: ignore[index]
    skipped_parquet = output_dir / "lm2011_table_iv_results_skipped_quarters.parquet"
    skipped_csv = output_dir / "lm2011_table_iv_results_skipped_quarters.csv"
    assert (output_dir / "lm2011_table_iv_results.parquet").exists()
    assert skipped_parquet.exists()
    assert skipped_csv.exists()
    assert stage["status"] == "generated"
    assert stage["extra_artifacts"] == {
        "skipped_quarters_parquet": str(skipped_parquet.resolve()),
        "skipped_quarters_csv": str(skipped_csv.resolve()),
    }
    assert stage["warnings"] == [
        "Skipped 1 rank-deficient quarter across 1 quarter/signal fit; see skipped_quarters_parquet."
    ]


def test_write_quarterly_regression_table_stage_marks_all_skipped_output_empty(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    manifest_path = tmp_path / "manifest.json"
    manifest: dict[str, object] = {
        "roots": {"output_dir": str(output_dir)},
        "artifacts": {},
        "row_counts": {},
        "stages": {},
    }

    runner._write_quarterly_regression_table_stage(
        manifest,
        manifest_path=manifest_path,
        output_dir=output_dir,
        stage_name="table_iv_results",
        bundle=_quarterly_bundle(skipped_quarters_df=_skipped_quarter_diagnostics_df()),
    )

    stage = manifest["stages"]["table_iv_results"]  # type: ignore[index]
    assert (output_dir / "lm2011_table_iv_results.parquet").exists()
    assert (output_dir / "lm2011_table_iv_results_skipped_quarters.parquet").exists()
    assert (output_dir / "lm2011_table_iv_results_skipped_quarters.csv").exists()
    assert stage["status"] == "generated_empty"
    assert stage["reason"] == runner.NO_ESTIMABLE_QUARTERLY_FAMA_MACBETH_QUARTERS


def test_write_quarterly_regression_table_stage_supports_no_ownership_stage_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    manifest_path = tmp_path / "manifest.json"
    manifest: dict[str, object] = {
        "roots": {"output_dir": str(output_dir)},
        "artifacts": {},
        "row_counts": {},
        "stages": {},
    }

    runner._write_quarterly_regression_table_stage(
        manifest,
        manifest_path=manifest_path,
        output_dir=output_dir,
        stage_name="table_iv_results_no_ownership",
        bundle=_quarterly_bundle(skipped_quarters_df=_skipped_quarter_diagnostics_df()),
    )

    stage = manifest["stages"]["table_iv_results_no_ownership"]  # type: ignore[index]
    assert (output_dir / "lm2011_table_iv_results_no_ownership.parquet").exists()
    assert (output_dir / "lm2011_table_iv_results_no_ownership_skipped_quarters.parquet").exists()
    assert (output_dir / "lm2011_table_iv_results_no_ownership_skipped_quarters.csv").exists()
    assert stage["status"] == "generated_empty"
    assert stage["reason"] == runner.NO_ESTIMABLE_QUARTERLY_FAMA_MACBETH_QUARTERS


def test_pipeline_reuses_existing_text_feature_artifacts_in_output_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_root, upstream_run_root, additional_data_dir, output_dir = _build_temp_layout(tmp_path)
    _write_valid_text_feature_artifact(
        output_dir / runner.STAGE_ARTIFACT_FILENAMES["text_features_full_10k"],
        additional_data_dir=additional_data_dir,
        stage_name="text_features_full_10k",
    )
    _write_valid_text_feature_artifact(
        output_dir / runner.STAGE_ARTIFACT_FILENAMES["text_features_mda"],
        additional_data_dir=additional_data_dir,
        stage_name="text_features_mda",
    )
    _write_text_feature_reuse_manifest(output_dir, additional_data_dir=additional_data_dir)

    run_cfg = runner.build_lm2011_post_refinitiv_run_config(
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
        ),
        enabled_stages=("text_features_full_10k", "text_features_mda"),
    )

    monkeypatch.setattr(
        runner,
        "write_lm2011_text_features_full_10k_parquet",
        lambda *_, **__: (_ for _ in ()).throw(
            AssertionError("full-10-K writer should not run when reuse succeeds")
        ),
    )
    monkeypatch.setattr(
        runner,
        "write_lm2011_text_features_mda_parquet",
        lambda *_, **__: (_ for _ in ()).throw(
            AssertionError("MD&A writer should not run when reuse succeeds")
        ),
    )

    assert runner.run_lm2011_post_refinitiv_pipeline(run_cfg) == 0

    manifest = json.loads((output_dir / runner.MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["stages"]["text_features_full_10k"]["status"] == runner.STAGE_STATUS_REUSED_EXISTING_ARTIFACT
    assert manifest["stages"]["text_features_mda"]["status"] == runner.STAGE_STATUS_REUSED_EXISTING_ARTIFACT


def test_pipeline_reuses_existing_event_screen_surface_artifact_in_output_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_root, upstream_run_root, additional_data_dir, output_dir = _build_temp_layout(tmp_path)
    _write_valid_text_feature_artifact(
        output_dir / runner.STAGE_ARTIFACT_FILENAMES["text_features_full_10k"],
        additional_data_dir=additional_data_dir,
        stage_name="text_features_full_10k",
    )
    _write_text_feature_reuse_manifest(output_dir, additional_data_dir=additional_data_dir)
    _write_parquet(
        output_dir / runner.STAGE_ARTIFACT_FILENAMES["event_screen_surface"],
        pl.DataFrame({"doc_id": ["d1"]}),
    )

    run_cfg = runner.build_lm2011_post_refinitiv_run_config(
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
        ),
        enabled_stages=("event_screen_surface",),
    )

    monkeypatch.setattr(
        runner,
        "_write_event_screen_surface_stage",
        lambda *_, **__: (_ for _ in ()).throw(
            AssertionError("event-screen surface writer should not run when reuse succeeds")
        ),
    )

    assert runner.run_lm2011_post_refinitiv_pipeline(run_cfg) == 0

    manifest = json.loads((output_dir / runner.MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["stages"]["event_screen_surface"]["status"] == runner.STAGE_STATUS_REUSED_EXISTING_ARTIFACT


def test_pipeline_reuses_existing_event_panel_artifact_in_output_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_root, upstream_run_root, additional_data_dir, output_dir = _build_temp_layout(tmp_path)
    _write_parquet(
        output_dir / runner.STAGE_ARTIFACT_FILENAMES["event_panel"],
        pl.DataFrame({"doc_id": ["d1"]}),
    )

    run_cfg = runner.build_lm2011_post_refinitiv_run_config(
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
        ),
        enabled_stages=("event_panel",),
    )

    monkeypatch.setattr(
        runner,
        "build_lm2011_event_panel",
        lambda *_, **__: (_ for _ in ()).throw(
            AssertionError("event-panel builder should not run when reuse succeeds")
        ),
    )

    assert runner.run_lm2011_post_refinitiv_pipeline(run_cfg) == 0

    manifest = json.loads((output_dir / runner.MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["stages"]["event_panel"]["status"] == runner.STAGE_STATUS_REUSED_EXISTING_ARTIFACT


def test_pipeline_reuses_existing_regression_table_artifacts_in_output_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_root, upstream_run_root, additional_data_dir, output_dir = _build_temp_layout(tmp_path)
    _write_valid_text_feature_artifact(
        output_dir / runner.STAGE_ARTIFACT_FILENAMES["text_features_full_10k"],
        additional_data_dir=additional_data_dir,
        stage_name="text_features_full_10k",
    )
    _write_text_feature_reuse_manifest(output_dir, additional_data_dir=additional_data_dir)
    table_iv_path = output_dir / runner.STAGE_ARTIFACT_FILENAMES["table_iv_results"]
    _write_parquet(
        table_iv_path,
        pl.DataFrame({"table_id": ["table_iv_full_10k"], "estimate": [1.0]}),
    )
    skipped_quarters_parquet = output_dir / "lm2011_table_iv_results_skipped_quarters.parquet"
    skipped_quarters_csv = output_dir / "lm2011_table_iv_results_skipped_quarters.csv"
    skipped_quarters_df = _skipped_quarter_diagnostics_df()
    skipped_quarters_df.write_parquet(skipped_quarters_parquet)
    skipped_quarters_df.write_csv(skipped_quarters_csv)
    _write_parquet(
        output_dir / runner.STAGE_ARTIFACT_FILENAMES["table_ia_ii_results"],
        pl.DataFrame({"table_id": ["table_ia_ii"], "estimate": [1.0]}),
    )

    run_cfg = runner.build_lm2011_post_refinitiv_run_config(
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
        ),
        enabled_stages=("table_iv_results", "table_ia_ii_results"),
    )

    monkeypatch.setattr(
        runner,
        "_build_lm2011_table_iv_results_bundle",
        lambda *_, **__: (_ for _ in ()).throw(
            AssertionError("table IV builder should not run when reuse succeeds")
        ),
    )
    monkeypatch.setattr(
        runner,
        "build_lm2011_table_ia_ii_results",
        lambda *_, **__: (_ for _ in ()).throw(
            AssertionError("table IA.II builder should not run when reuse succeeds")
        ),
    )

    assert runner.run_lm2011_post_refinitiv_pipeline(run_cfg) == 0

    manifest = json.loads((output_dir / runner.MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["stages"]["table_iv_results"]["status"] == runner.STAGE_STATUS_REUSED_EXISTING_ARTIFACT
    assert manifest["stages"]["table_iv_results"]["extra_artifacts"] == {
        "skipped_quarters_parquet": str(skipped_quarters_parquet.resolve()),
        "skipped_quarters_csv": str(skipped_quarters_csv.resolve()),
    }
    assert manifest["stages"]["table_ia_ii_results"]["status"] == runner.STAGE_STATUS_REUSED_EXISTING_ARTIFACT


def test_pipeline_reuses_existing_no_ownership_regression_artifacts_in_output_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_root, upstream_run_root, additional_data_dir, output_dir = _build_temp_layout(tmp_path)
    _write_valid_text_feature_artifact(
        output_dir / runner.STAGE_ARTIFACT_FILENAMES["text_features_full_10k"],
        additional_data_dir=additional_data_dir,
        stage_name="text_features_full_10k",
    )
    _write_text_feature_reuse_manifest(output_dir, additional_data_dir=additional_data_dir)
    table_path = output_dir / runner.STAGE_ARTIFACT_FILENAMES["table_iv_results_no_ownership"]
    _write_parquet(
        table_path,
        pl.DataFrame({"table_id": ["table_iv_full_10k"], "estimate": [1.0]}),
    )
    skipped_quarters_parquet = output_dir / "lm2011_table_iv_results_no_ownership_skipped_quarters.parquet"
    skipped_quarters_csv = output_dir / "lm2011_table_iv_results_no_ownership_skipped_quarters.csv"
    skipped_quarters_df = _skipped_quarter_diagnostics_df()
    skipped_quarters_df.write_parquet(skipped_quarters_parquet)
    skipped_quarters_df.write_csv(skipped_quarters_csv)

    run_cfg = runner.build_lm2011_post_refinitiv_run_config(
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
        ),
        enabled_stages=("table_iv_results_no_ownership",),
    )

    monkeypatch.setattr(
        runner,
        "_build_lm2011_table_iv_results_no_ownership_bundle",
        lambda *_, **__: (_ for _ in ()).throw(
            AssertionError("table IV no-ownership builder should not run when reuse succeeds")
        ),
    )

    assert runner.run_lm2011_post_refinitiv_pipeline(run_cfg) == 0

    manifest = json.loads((output_dir / runner.MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert (
        manifest["stages"]["table_iv_results_no_ownership"]["status"]
        == runner.STAGE_STATUS_REUSED_EXISTING_ARTIFACT
    )
    assert manifest["stages"]["table_iv_results_no_ownership"]["extra_artifacts"] == {
        "skipped_quarters_parquet": str(skipped_quarters_parquet.resolve()),
        "skipped_quarters_csv": str(skipped_quarters_csv.resolve()),
    }


def test_pipeline_recompute_event_panel_ignores_reusable_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_root, upstream_run_root, additional_data_dir, output_dir = _build_temp_layout(tmp_path)
    _write_valid_text_feature_artifact(
        output_dir / runner.STAGE_ARTIFACT_FILENAMES["text_features_full_10k"],
        additional_data_dir=additional_data_dir,
        stage_name="text_features_full_10k",
    )
    _write_text_feature_reuse_manifest(output_dir, additional_data_dir=additional_data_dir)
    _write_parquet(
        output_dir / runner.STAGE_ARTIFACT_FILENAMES["event_screen_surface"],
        pl.DataFrame({"doc_id": ["d1"]}),
    )
    _write_parquet(
        output_dir / runner.STAGE_ARTIFACT_FILENAMES["event_panel"],
        pl.DataFrame({"doc_id": ["stale"]}),
    )

    run_cfg = runner.build_lm2011_post_refinitiv_run_config(
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
                "--recompute-event-panel",
            ]
        ),
        enabled_stages=("event_screen_surface", "event_panel"),
    )

    captured: dict[str, bool] = {"called": False}

    def _capture_event_panel(*_: object, **__: object) -> pl.LazyFrame:
        captured["called"] = True
        return pl.DataFrame({"doc_id": ["rebuilt"]}).lazy()

    monkeypatch.setattr(runner, "build_lm2011_event_panel", _capture_event_panel)

    assert runner.run_lm2011_post_refinitiv_pipeline(run_cfg) == 0

    manifest = json.loads((output_dir / runner.MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert captured["called"] is True
    assert manifest["stages"]["event_screen_surface"]["status"] == runner.STAGE_STATUS_REUSED_EXISTING_ARTIFACT
    assert manifest["stages"]["event_panel"]["status"] == "generated"
    assert pl.read_parquet(output_dir / runner.STAGE_ARTIFACT_FILENAMES["event_panel"]).get_column("doc_id").to_list() == [
        "rebuilt"
    ]


def test_pipeline_recompute_regression_tables_ignores_reusable_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_root, upstream_run_root, additional_data_dir, output_dir = _build_temp_layout(tmp_path)
    _write_valid_text_feature_artifact(
        output_dir / runner.STAGE_ARTIFACT_FILENAMES["text_features_full_10k"],
        additional_data_dir=additional_data_dir,
        stage_name="text_features_full_10k",
    )
    _write_text_feature_reuse_manifest(output_dir, additional_data_dir=additional_data_dir)
    _write_parquet(
        output_dir / runner.STAGE_ARTIFACT_FILENAMES["event_panel"],
        pl.DataFrame({"doc_id": ["d1"]}),
    )
    _write_parquet(
        output_dir / runner.STAGE_ARTIFACT_FILENAMES["table_iv_results"],
        pl.DataFrame({"table_id": ["stale"], "estimate": [1.0]}),
    )

    run_cfg = runner.build_lm2011_post_refinitiv_run_config(
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
                "--recompute-regression-tables",
            ]
        ),
        enabled_stages=("event_panel", "table_iv_results"),
    )

    captured: dict[str, bool] = {"called": False}

    def _capture_table_iv(*_: object, **__: object) -> runner._QuarterlyFamaMacbethBundle:
        captured["called"] = True
        return _quarterly_bundle(
            results_df=pl.DataFrame({"table_id": ["table_iv_full_10k"], "estimate": [2.0]}),
        )

    monkeypatch.setattr(runner, "_build_lm2011_table_iv_results_bundle", _capture_table_iv)

    assert runner.run_lm2011_post_refinitiv_pipeline(run_cfg) == 0

    manifest = json.loads((output_dir / runner.MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert captured["called"] is True
    assert manifest["stages"]["event_panel"]["status"] == runner.STAGE_STATUS_REUSED_EXISTING_ARTIFACT
    assert manifest["stages"]["table_iv_results"]["status"] == "generated"
    assert pl.read_parquet(output_dir / runner.STAGE_ARTIFACT_FILENAMES["table_iv_results"]).get_column("estimate").to_list() == [
        2.0
    ]


def test_pipeline_explicit_text_feature_override_is_imported_and_reused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_root, upstream_run_root, additional_data_dir, output_dir = _build_temp_layout(tmp_path)
    external_artifact = tmp_path / "reuse_source" / "lm2011_text_features_full_10k.parquet"
    _write_valid_text_feature_artifact(
        external_artifact,
        additional_data_dir=additional_data_dir,
        stage_name="text_features_full_10k",
    )

    run_cfg = runner.build_lm2011_post_refinitiv_run_config(
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
                "--text-features-full-10k-path",
                str(external_artifact),
            ]
        ),
        enabled_stages=("text_features_full_10k",),
    )

    monkeypatch.setattr(
        runner,
        "write_lm2011_text_features_full_10k_parquet",
        lambda *_, **__: (_ for _ in ()).throw(
            AssertionError("full-10-K writer should not run when explicit reuse succeeds")
        ),
    )

    assert runner.run_lm2011_post_refinitiv_pipeline(run_cfg) == 0

    canonical_artifact = output_dir / runner.STAGE_ARTIFACT_FILENAMES["text_features_full_10k"]
    manifest = json.loads((output_dir / runner.MANIFEST_FILENAME).read_text(encoding="utf-8"))
    stage = manifest["stages"]["text_features_full_10k"]

    assert canonical_artifact.exists()
    assert pl.read_parquet(canonical_artifact).get_column("doc_id").to_list() == ["d1"]
    assert stage["status"] == runner.STAGE_STATUS_REUSED_EXISTING_ARTIFACT
    assert stage["artifact_path"] == str(canonical_artifact.resolve())
    assert stage["source_path"] == str(external_artifact.resolve())
    assert stage["row_count"] == 1
    assert stage["warnings"] == [
        "No sibling lm2011_sample_run_manifest.json was found for the explicit reuse artifact; semantic validation fell back to schema-only checks."
    ]


def test_pipeline_uses_reused_full_10k_text_features_for_downstream_stage_when_generation_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_root, upstream_run_root, additional_data_dir, output_dir = _build_temp_layout(tmp_path)
    sample_backbone_path = tmp_path / "prebuilt_sample_backbone.parquet"
    _write_parquet(
        sample_backbone_path,
        pl.DataFrame(
            {
                "doc_id": ["d1"],
                "cik_10": ["0000000001"],
                "filing_date": [dt.date(1995, 1, 1)],
                "normalized_form": ["10-K"],
                "KYPERMNO": [1],
            }
        ),
    )
    _write_valid_text_feature_artifact(
        output_dir / runner.STAGE_ARTIFACT_FILENAMES["text_features_full_10k"],
        additional_data_dir=additional_data_dir,
        stage_name="text_features_full_10k",
    )
    _write_text_feature_reuse_manifest(output_dir, additional_data_dir=additional_data_dir)

    run_cfg = runner.build_lm2011_post_refinitiv_run_config(
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
                str(sample_backbone_path),
            ]
        ),
        enabled_stages=(
            "sample_backbone",
            "annual_accounting_panel",
            "ff_factors_daily_normalized",
            "event_screen_surface",
        ),
        fail_closed_for_enabled_stages=True,
    )

    monkeypatch.setattr(
        runner,
        "build_annual_accounting_panel",
        lambda *_, **__: pl.DataFrame({"gvkey_int": [1]}).lazy(),
    )

    def _event_screen_surface_stub(
        manifest: dict[str, object],
        *,
        manifest_path: Path | None,
        output_dir: Path,
        text_features_full_10k_lf: pl.LazyFrame,
        **_: object,
    ) -> pl.LazyFrame:
        assert text_features_full_10k_lf.collect().get_column("doc_id").to_list() == ["d1"]
        artifact_path = output_dir / runner.STAGE_ARTIFACT_FILENAMES["event_screen_surface"]
        _write_parquet(artifact_path, pl.DataFrame({"doc_id": ["d1"]}))
        return runner._record_existing_stage_artifact(
            manifest,
            manifest_path=manifest_path,
            stage_name="event_screen_surface",
            artifact_path=artifact_path,
        )

    monkeypatch.setattr(runner, "_write_event_screen_surface_stage", _event_screen_surface_stub)

    assert runner.run_lm2011_post_refinitiv_pipeline(run_cfg) == 0

    manifest = json.loads((output_dir / runner.MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["stages"]["text_features_full_10k"]["status"] == runner.STAGE_STATUS_REUSED_EXISTING_ARTIFACT
    assert manifest["stages"]["event_screen_surface"]["status"] == "generated"


def test_pipeline_invalid_explicit_text_feature_override_fails_early(tmp_path: Path) -> None:
    sample_root, upstream_run_root, additional_data_dir, output_dir = _build_temp_layout(tmp_path)
    invalid_artifact = tmp_path / "invalid_full_10k.parquet"
    _write_parquet(invalid_artifact, pl.DataFrame({"doc_id": ["d1"]}))

    run_cfg = runner.build_lm2011_post_refinitiv_run_config(
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
                "--text-features-full-10k-path",
                str(invalid_artifact),
            ]
        ),
        enabled_stages=("text_features_full_10k",),
    )

    with pytest.raises(ValueError, match="Explicit reusable artifact for text_features_full_10k is incompatible"):
        runner.run_lm2011_post_refinitiv_pipeline(run_cfg)


def test_pipeline_ignores_incompatible_auto_text_feature_artifact_and_rebuilds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_root, upstream_run_root, additional_data_dir, output_dir = _build_temp_layout(tmp_path)
    invalid_artifact = output_dir / runner.STAGE_ARTIFACT_FILENAMES["text_features_full_10k"]
    _write_parquet(invalid_artifact, pl.DataFrame({"doc_id": ["d1"]}))

    run_cfg = runner.build_lm2011_post_refinitiv_run_config(
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
        ),
        enabled_stages=("text_features_full_10k",),
    )

    calls: dict[str, int] = {"count": 0}

    def _full_10k_writer_stub(
        sec_parsed_lf: pl.LazyFrame,
        *,
        output_path: Path,
        **_: object,
    ) -> int:
        calls["count"] += 1
        assert sec_parsed_lf.collect().get_column("doc_id").to_list() == ["d1"]
        _write_valid_text_feature_artifact(
            output_path,
            additional_data_dir=additional_data_dir,
            stage_name="text_features_full_10k",
        )
        return 1

    monkeypatch.setattr(runner, "write_lm2011_text_features_full_10k_parquet", _full_10k_writer_stub)

    assert runner.run_lm2011_post_refinitiv_pipeline(run_cfg) == 0

    manifest = json.loads((output_dir / runner.MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert calls["count"] == 1
    assert manifest["stages"]["text_features_full_10k"]["status"] == "generated"
    assert manifest["row_counts"]["text_features_full_10k"] == 1


def test_pipeline_rejects_root_dictionary_manifest_reuse_after_generated_replication_switch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_root, upstream_run_root, additional_data_dir, output_dir = _build_temp_layout(tmp_path)
    _write_valid_text_feature_artifact(
        output_dir / runner.STAGE_ARTIFACT_FILENAMES["text_features_full_10k"],
        additional_data_dir=additional_data_dir,
        stage_name="text_features_full_10k",
        dictionary_input_dir=additional_data_dir,
    )
    _write_text_feature_reuse_manifest(
        output_dir,
        additional_data_dir=additional_data_dir,
        dictionary_input_dir=additional_data_dir,
    )

    run_cfg = runner.build_lm2011_post_refinitiv_run_config(
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
        ),
        enabled_stages=("text_features_full_10k",),
    )

    calls: dict[str, int] = {"count": 0}

    def _full_10k_writer_stub(
        sec_parsed_lf: pl.LazyFrame,
        *,
        output_path: Path,
        **_: object,
    ) -> int:
        calls["count"] += 1
        assert sec_parsed_lf.collect().get_column("doc_id").to_list() == ["d1"]
        _write_valid_text_feature_artifact(
            output_path,
            additional_data_dir=additional_data_dir,
            stage_name="text_features_full_10k",
        )
        return 1

    monkeypatch.setattr(runner, "write_lm2011_text_features_full_10k_parquet", _full_10k_writer_stub)

    assert runner.run_lm2011_post_refinitiv_pipeline(run_cfg) == 0

    manifest = json.loads((output_dir / runner.MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert calls["count"] == 1
    assert manifest["stages"]["text_features_full_10k"]["status"] == "generated"
    assert manifest["row_counts"]["text_features_full_10k"] == 1


def test_pipeline_invalid_auto_text_feature_artifact_fails_early_when_downstream_requires_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_root, upstream_run_root, additional_data_dir, output_dir = _build_temp_layout(tmp_path)
    sample_backbone_path = tmp_path / "prebuilt_sample_backbone.parquet"
    _write_parquet(
        sample_backbone_path,
        pl.DataFrame(
            {
                "doc_id": ["d1"],
                "cik_10": ["0000000001"],
                "filing_date": [dt.date(1995, 1, 1)],
                "normalized_form": ["10-K"],
                "KYPERMNO": [1],
            }
        ),
    )
    _write_parquet(
        output_dir / runner.STAGE_ARTIFACT_FILENAMES["text_features_full_10k"],
        pl.DataFrame({"doc_id": ["d1"]}),
    )
    _write_text_feature_reuse_manifest(output_dir, additional_data_dir=additional_data_dir)

    run_cfg = runner.build_lm2011_post_refinitiv_run_config(
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
                str(sample_backbone_path),
            ]
        ),
        enabled_stages=(
            "sample_backbone",
            "annual_accounting_panel",
            "ff_factors_daily_normalized",
            "event_screen_surface",
        ),
        fail_closed_for_enabled_stages=True,
    )

    monkeypatch.setattr(
        runner,
        "build_annual_accounting_panel",
        lambda *_, **__: pl.DataFrame({"gvkey_int": [1]}).lazy(),
    )
    monkeypatch.setattr(
        runner,
        "_write_event_screen_surface_stage",
        lambda **_: (_ for _ in ()).throw(
            AssertionError("downstream stage should not run after text-feature preload failure")
        ),
    )

    with pytest.raises(RuntimeError, match="LM2011 stages require text_features_full_10k"):
        runner.run_lm2011_post_refinitiv_pipeline(run_cfg)


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
    local_work_root = tmp_path / "lm2011_local_work"
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
    empty_quarterly_bundle = _quarterly_bundle()
    monkeypatch.setattr(runner, "_build_lm2011_table_iv_results_bundle", lambda *_, **__: empty_quarterly_bundle)
    monkeypatch.setattr(
        runner,
        "_build_lm2011_table_iv_results_no_ownership_bundle",
        lambda *_, **__: empty_quarterly_bundle,
    )
    monkeypatch.setattr(runner, "_build_lm2011_table_v_results_bundle", lambda *_, **__: empty_quarterly_bundle)
    monkeypatch.setattr(
        runner,
        "_build_lm2011_table_v_results_no_ownership_bundle",
        lambda *_, **__: empty_quarterly_bundle,
    )
    monkeypatch.setattr(runner, "_build_lm2011_table_vi_results_bundle", lambda *_, **__: empty_quarterly_bundle)
    monkeypatch.setattr(
        runner,
        "_build_lm2011_table_vi_results_no_ownership_bundle",
        lambda *_, **__: empty_quarterly_bundle,
    )
    monkeypatch.setattr(runner, "_build_lm2011_table_viii_results_bundle", lambda *_, **__: empty_quarterly_bundle)
    monkeypatch.setattr(
        runner,
        "_build_lm2011_table_viii_results_no_ownership_bundle",
        lambda *_, **__: empty_quarterly_bundle,
    )
    monkeypatch.setattr(runner, "_build_lm2011_table_ia_i_results_bundle", lambda *_, **__: empty_quarterly_bundle)
    monkeypatch.setattr(
        runner,
        "_build_lm2011_table_ia_i_results_no_ownership_bundle",
        lambda *_, **__: empty_quarterly_bundle,
    )
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
    monkeypatch.setattr(runner, "build_lm2011_table_ia_ii_results", lambda *_, **__: _empty_quarterly_results_df())
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
            "--local-work-root",
            str(local_work_root),
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
    assert manifest["roots"]["local_work_root"] == str(local_work_root.resolve())
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
    assert manifest["resolved_inputs"]["generated_dictionary_family_root"].endswith(
        "generated_dictionary_families"
    )
    assert manifest["resolved_inputs"]["effective_dictionary_input_dir"].endswith(
        "generated_dictionary_families\\replication"
    )
    assert manifest["dictionary_inputs"]["resource_scope"] == "repo-local operative LM2011-style lexicon inputs"
    assert "not asserted to be the original LM2011" in manifest["dictionary_inputs"]["historical_provenance_warning"]
    assert manifest["dictionary_inputs"]["source_additional_data_dir"] == str(additional_data_dir.resolve())
    assert manifest["dictionary_inputs"]["generated_dictionary_family_root"] == str(
        (additional_data_dir / "generated_dictionary_families").resolve()
    )
    assert manifest["dictionary_inputs"]["effective_dictionary_input_dir"] == str(
        (additional_data_dir / "generated_dictionary_families" / "replication").resolve()
    )
    assert set(manifest["dictionary_inputs"]["generated_dictionary_families"]) == {"replication", "extended"}
    master_resources = [
        resource
        for resource in manifest["dictionary_inputs"]["resources"]
        if resource["role"] == "recognized_word_master_dictionary"
    ]
    assert master_resources[0]["name"] == "LM2011_MasterDictionary.txt"
    assert "not provenance-verified" in master_resources[0]["provenance_status"]
    assert all(
        "generated_dictionary_families\\replication" in resource["path"]
        for resource in manifest["dictionary_inputs"]["resources"]
    )
    assert (additional_data_dir / "generated_dictionary_families" / "replication").exists()
    assert (additional_data_dir / "generated_dictionary_families" / "extended").exists()
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
    assert manifest["stages"]["table_iv_results"]["reason"] == runner.NO_ESTIMABLE_QUARTERLY_FAMA_MACBETH_QUARTERS
    assert manifest["stages"]["ff_factors_monthly_with_mom_normalized"]["status"] == "generated"
    assert manifest["stages"]["ff_factors_monthly_with_mom_normalized"]["row_count"] == 2
    assert manifest["stages"]["trading_strategy_monthly_returns"]["status"] == "generated"
    assert manifest["stages"]["table_ia_ii_results"]["status"] == "generated_empty"
    assert manifest["stages"]["table_ia_ii_results"]["reason"] == runner.EMPTY_TABLE_REASON
    assert captured["text_features_full_10k_kwargs"]["cleaning_contract"] == "lm2011_paper"
    assert captured["text_features_full_10k_kwargs"]["batch_size"] == 4
    assert captured["text_features_full_10k_kwargs"]["temp_root"] == local_work_root.resolve() / "text_features_full_10k"
    assert callable(captured["text_features_full_10k_kwargs"]["progress_callback"])
    assert captured["text_features_full_10k_kwargs"]["master_dictionary_words"] == ("TOKEN", "HARVARD", "RECOGNIZED")
    assert captured["text_features_mda_kwargs"]["batch_size"] == 20
    assert captured["text_features_mda_kwargs"]["temp_root"] == local_work_root.resolve() / "text_features_mda"
    assert callable(captured["text_features_mda_kwargs"]["progress_callback"])
    assert captured["text_features_mda_kwargs"]["master_dictionary_words"] == ("TOKEN", "HARVARD", "RECOGNIZED")
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
    empty_quarterly_bundle = _quarterly_bundle()
    monkeypatch.setattr(runner, "build_lm2011_return_regression_panel", lambda *_, **__: pl.DataFrame({"doc_id": ["d1"]}).lazy())
    monkeypatch.setattr(runner, "build_lm2011_sue_regression_panel", lambda *_, **__: pl.DataFrame({"doc_id": ["d1"]}).lazy())
    monkeypatch.setattr(runner, "_build_lm2011_table_iv_results_bundle", lambda *_, **__: empty_quarterly_bundle)
    monkeypatch.setattr(
        runner,
        "_build_lm2011_table_iv_results_no_ownership_bundle",
        lambda *_, **__: empty_quarterly_bundle,
    )
    monkeypatch.setattr(runner, "_build_lm2011_table_v_results_bundle", lambda *_, **__: empty_quarterly_bundle)
    monkeypatch.setattr(
        runner,
        "_build_lm2011_table_v_results_no_ownership_bundle",
        lambda *_, **__: empty_quarterly_bundle,
    )
    monkeypatch.setattr(runner, "_build_lm2011_table_vi_results_bundle", lambda *_, **__: empty_quarterly_bundle)
    monkeypatch.setattr(
        runner,
        "_build_lm2011_table_vi_results_no_ownership_bundle",
        lambda *_, **__: empty_quarterly_bundle,
    )
    monkeypatch.setattr(runner, "_build_lm2011_table_viii_results_bundle", lambda *_, **__: empty_quarterly_bundle)
    monkeypatch.setattr(
        runner,
        "_build_lm2011_table_viii_results_no_ownership_bundle",
        lambda *_, **__: empty_quarterly_bundle,
    )
    monkeypatch.setattr(runner, "_build_lm2011_table_ia_i_results_bundle", lambda *_, **__: empty_quarterly_bundle)
    monkeypatch.setattr(
        runner,
        "_build_lm2011_table_ia_i_results_no_ownership_bundle",
        lambda *_, **__: empty_quarterly_bundle,
    )
    monkeypatch.setattr(runner, "build_lm2011_table_ia_ii_results", lambda *_, **__: _empty_quarterly_results_df())
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
    empty_quarterly_bundle = _quarterly_bundle()
    monkeypatch.setattr(runner, "_build_lm2011_table_iv_results_bundle", lambda *_, **__: empty_quarterly_bundle)
    monkeypatch.setattr(
        runner,
        "_build_lm2011_table_iv_results_no_ownership_bundle",
        lambda *_, **__: empty_quarterly_bundle,
    )
    monkeypatch.setattr(runner, "_build_lm2011_table_v_results_bundle", lambda *_, **__: empty_quarterly_bundle)
    monkeypatch.setattr(
        runner,
        "_build_lm2011_table_v_results_no_ownership_bundle",
        lambda *_, **__: empty_quarterly_bundle,
    )
    monkeypatch.setattr(runner, "_build_lm2011_table_vi_results_bundle", lambda *_, **__: empty_quarterly_bundle)
    monkeypatch.setattr(
        runner,
        "_build_lm2011_table_vi_results_no_ownership_bundle",
        lambda *_, **__: empty_quarterly_bundle,
    )
    monkeypatch.setattr(runner, "_build_lm2011_table_viii_results_bundle", lambda *_, **__: empty_quarterly_bundle)
    monkeypatch.setattr(
        runner,
        "_build_lm2011_table_viii_results_no_ownership_bundle",
        lambda *_, **__: empty_quarterly_bundle,
    )
    monkeypatch.setattr(runner, "_build_lm2011_table_ia_i_results_bundle", lambda *_, **__: empty_quarterly_bundle)
    monkeypatch.setattr(
        runner,
        "_build_lm2011_table_ia_i_results_no_ownership_bundle",
        lambda *_, **__: empty_quarterly_bundle,
    )
    monkeypatch.setattr(runner, "build_lm2011_table_ia_ii_results", lambda *_, **__: _empty_quarterly_results_df())
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
    local_work_root = tmp_path / "colab_local_work"
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
            "--local-work-root",
            str(local_work_root),
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
    assert paths.local_work_root == local_work_root.resolve()


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
    assert run_cfg.paths.local_work_root == runner.DEFAULT_LOCAL_WORK_ROOT.resolve()
    assert run_cfg.paths.print_ram_stats is False
    assert run_cfg.paths.ram_log_interval_batches == runner.DEFAULT_RAM_LOG_INTERVAL_BATCHES


def test_build_run_config_can_override_stage_selection(
    tmp_path: Path,
) -> None:
    sample_root, upstream_run_root, additional_data_dir, output_dir = _build_temp_layout(tmp_path)
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
        ]
    )

    run_cfg = runner.build_lm2011_post_refinitiv_run_config(
        args,
        enabled_stages=("sample_backbone", "event_panel"),
        fail_closed_for_enabled_stages=True,
    )

    assert run_cfg.paths.output_dir == output_dir.resolve()
    assert run_cfg.enabled_stages == ("sample_backbone", "event_panel")
    assert run_cfg.fail_closed_for_enabled_stages is True


def test_resolve_enabled_lm2011_stage_names_from_env_matches_unified_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from thesis_pkg.notebooks_and_scripts import sec_ccm_unified_runner as unified_runner

    stage_env_names = [
        "SEC_CCM_RUN_LM2011_POST_REFINITIV",
        *[
            f"SEC_CCM_RUN_LM2011_{stage_name.upper()}"
            for stage_name in runner.LM2011_ALL_STAGE_NAMES
        ],
    ]
    for env_name in stage_env_names:
        monkeypatch.delenv(env_name, raising=False)

    monkeypatch.setenv("SEC_CCM_RUN_LM2011_POST_REFINITIV", "true")
    monkeypatch.setenv("SEC_CCM_RUN_LM2011_TEXT_FEATURES_MDA", "false")
    monkeypatch.setenv("SEC_CCM_RUN_LM2011_TABLE_IA_II_RESULTS", "true")

    umbrella_enabled = unified_runner._env_bool("SEC_CCM_RUN_LM2011_POST_REFINITIV", False)
    expected_enabled_stages = tuple(
        stage_name
        for stage_name in unified_runner.LM2011_ALL_STAGE_NAMES
        if unified_runner._resolve_stage_toggle(
            f"SEC_CCM_RUN_LM2011_{stage_name.upper()}",
            umbrella_enabled=umbrella_enabled,
            default_when_umbrella=stage_name
            not in unified_runner.LM2011_OPTIONAL_STAGE_DEFAULTS_FALSE,
        )
    )

    assert runner.resolve_enabled_lm2011_stage_names_from_env() == expected_enabled_stages


def test_notebook_wrapper_honors_unified_env_contract_and_stage_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_root, upstream_run_root, additional_data_dir, _ = _build_temp_layout(tmp_path)
    sec_ccm_output_dir = upstream_run_root / "sec_ccm_premerge"
    sec_ccm_output_dir.mkdir(parents=True, exist_ok=True)
    _write_parquet(
        sec_ccm_output_dir / "sec_ccm_matched_clean.parquet",
        pl.DataFrame({"doc_id": ["d1"], "KYPERMNO": [1]}),
    )
    _write_parquet(
        sec_ccm_output_dir / "lm2011_sample_backbone.parquet",
        pl.DataFrame({"doc_id": ["d1"]}),
    )
    custom_daily_panel = tmp_path / "overrides" / "daily_panel.parquet"
    custom_monthly_stock = tmp_path / "overrides" / "monthly_stock.parquet"
    custom_ff_monthly_with_mom = tmp_path / "overrides" / "ff_monthly_with_mom.parquet"
    _write_parquet(custom_daily_panel)
    _write_parquet(custom_monthly_stock)
    _write_parquet(custom_ff_monthly_with_mom)

    notebook_output_dir = tmp_path / "notebook_output"
    local_work_root = tmp_path / "local_work"

    monkeypatch.setenv("SEC_CCM_WORK_ROOT", str(sample_root))
    monkeypatch.setenv("SEC_CCM_RUN_ROOT", str(upstream_run_root))
    monkeypatch.setenv("SEC_CCM_OUTPUT_DIR", str(sec_ccm_output_dir))
    monkeypatch.setenv("SEC_CCM_ITEMS_ANALYSIS_DIR", str(upstream_run_root / "items_analysis"))
    monkeypatch.setenv("SEC_CCM_CCM_BASE_DIR", str(sample_root / "ccm_parquet_data"))
    monkeypatch.setenv("SEC_CCM_SEC_YEAR_MERGED_DIR", str(sample_root / "year_merged"))
    monkeypatch.setenv(
        "SEC_CCM_REFINITIV_DOC_OWNERSHIP_LM2011_DIR",
        str(upstream_run_root / "refinitiv_doc_ownership_lm2011"),
    )
    monkeypatch.setenv(
        "SEC_CCM_REFINITIV_DOC_ANALYST_LM2011_DIR",
        str(upstream_run_root / "refinitiv_doc_analyst_lm2011"),
    )
    monkeypatch.setenv("SEC_CCM_LM2011_ADDITIONAL_DATA_DIR", str(additional_data_dir))
    monkeypatch.setenv("SEC_CCM_LM2011_OUTPUT_DIR", str(notebook_output_dir))
    monkeypatch.setenv("SEC_CCM_LOCAL_WORK", str(local_work_root))
    monkeypatch.setenv("SEC_CCM_LM2011_DAILY_PANEL_PATH", str(custom_daily_panel))
    monkeypatch.setenv("SEC_CCM_LM2011_MONTHLY_STOCK_PATH", str(custom_monthly_stock))
    monkeypatch.setenv(
        "SEC_CCM_LM2011_FF_MONTHLY_WITH_MOM_PATH",
        str(custom_ff_monthly_with_mom),
    )
    monkeypatch.setenv("SEC_CCM_LM2011_FULL_10K_CLEANING_CONTRACT", "current")
    monkeypatch.setenv("SEC_CCM_LM2011_FULL_10K_TEXT_FEATURE_BATCH_SIZE", "11")
    monkeypatch.setenv("SEC_CCM_LM2011_MDA_TEXT_FEATURE_BATCH_SIZE", "13")
    monkeypatch.setenv("SEC_CCM_LM2011_EVENT_WINDOW_DOC_BATCH_SIZE", "17")
    monkeypatch.setenv("SEC_CCM_LM2011_RECOMPUTE_EVENT_SCREEN_SURFACE", "true")
    monkeypatch.setenv("SEC_CCM_LM2011_RECOMPUTE_EVENT_PANEL", "true")
    monkeypatch.setenv("SEC_CCM_LM2011_RECOMPUTE_REGRESSION_TABLES", "true")
    monkeypatch.setenv("SEC_CCM_PRINT_RAM_STATS", "true")
    monkeypatch.setenv("SEC_CCM_RAM_LOG_INTERVAL_BATCHES", "19")
    monkeypatch.setenv("SEC_CCM_RUN_LM2011_POST_REFINITIV", "true")
    monkeypatch.setenv("SEC_CCM_RUN_LM2011_TEXT_FEATURES_MDA", "false")
    monkeypatch.setenv("SEC_CCM_RUN_LM2011_TABLE_IA_II_RESULTS", "true")

    repo_root = tmp_path / "repo_root"
    repo_root.mkdir(parents=True, exist_ok=True)
    notebook_cells = _load_lm2011_notebook_cell_sources()
    config_source = notebook_cells[2]
    run_args_source = notebook_cells[3]
    execute_source = notebook_cells[4]

    namespace: dict[str, object] = {
        "IN_COLAB": False,
        "Path": Path,
        "os": os,
        "repo_root": repo_root,
    }
    exec(config_source, namespace)

    assert namespace["WORK_ROOT"] == sample_root.resolve()
    assert namespace["UPSTREAM_RUN_ROOT"] == upstream_run_root.resolve()
    assert namespace["YEAR_MERGED_DIR"] == (sample_root / "year_merged").resolve()
    assert namespace["MATCHED_CLEAN_PATH"] == (sec_ccm_output_dir / "sec_ccm_matched_clean.parquet").resolve()
    assert namespace["SAMPLE_BACKBONE_PATH"] == (sec_ccm_output_dir / "lm2011_sample_backbone.parquet").resolve()
    assert namespace["ITEMS_ANALYSIS_DIR"] == (upstream_run_root / "items_analysis").resolve()
    assert namespace["CCM_BASE_DIR"] == (sample_root / "ccm_parquet_data").resolve()
    assert namespace["DAILY_PANEL_PATH"] == custom_daily_panel.resolve()
    assert namespace["MONTHLY_STOCK_PATH"] == custom_monthly_stock.resolve()
    assert namespace["FF_MONTHLY_WITH_MOM_PATH"] == custom_ff_monthly_with_mom.resolve()
    assert namespace["OUTPUT_DIR"] == notebook_output_dir.resolve()
    assert namespace["LOCAL_WORK_ROOT"] == (local_work_root / "lm2011_post_refinitiv").resolve()
    assert namespace["DOC_OWNERSHIP_PATH"] == (
        upstream_run_root
        / "refinitiv_doc_ownership_lm2011"
        / "refinitiv_lm2011_doc_ownership.parquet"
    ).resolve()
    assert namespace["DOC_ANALYST_SELECTED_PATH"] == (
        upstream_run_root
        / "refinitiv_doc_analyst_lm2011"
        / "refinitiv_doc_analyst_selected.parquet"
    ).resolve()
    assert namespace["FULL_10K_CLEANING_CONTRACT"] == "current"
    assert namespace["FULL_10K_TEXT_FEATURE_BATCH_SIZE"] == 11
    assert namespace["MDA_TEXT_FEATURE_BATCH_SIZE"] == 13
    assert namespace["EVENT_WINDOW_DOC_BATCH_SIZE"] == 17
    assert namespace["RECOMPUTE_EVENT_SCREEN_SURFACE"] is True
    assert namespace["RECOMPUTE_EVENT_PANEL"] is True
    assert namespace["RECOMPUTE_REGRESSION_TABLES"] is True
    assert namespace["PRINT_RAM_STATS"] is True
    assert namespace["RAM_LOG_INTERVAL_BATCHES"] == 19

    exec(run_args_source, namespace)
    run_args = namespace["RUN_ARGS"]
    assert "--text-feature-batch-size" not in run_args
    assert "--doc-ownership-path" in run_args
    assert "--doc-analyst-selected-path" in run_args
    assert "--full-10k-text-feature-batch-size" in run_args
    assert "--mda-text-feature-batch-size" in run_args
    assert "--event-window-doc-batch-size" in run_args
    assert "--recompute-event-screen-surface" in run_args
    assert "--recompute-event-panel" in run_args
    assert "--recompute-regression-tables" in run_args
    assert "--print-ram-stats" in run_args

    captured: dict[str, object] = {}

    def _capture_run(run_cfg: runner.LM2011PostRefinitivRunConfig) -> int:
        captured["run_cfg"] = run_cfg
        return 0

    monkeypatch.setattr(namespace["runner"], "run_lm2011_post_refinitiv_pipeline", _capture_run)
    exec(execute_source, namespace)

    run_cfg = captured["run_cfg"]
    assert isinstance(run_cfg, runner.LM2011PostRefinitivRunConfig)
    assert run_cfg.paths.output_dir == notebook_output_dir.resolve()
    assert run_cfg.paths.local_work_root == (local_work_root / "lm2011_post_refinitiv").resolve()
    assert run_cfg.paths.doc_ownership_path == namespace["DOC_OWNERSHIP_PATH"]
    assert run_cfg.paths.doc_analyst_selected_path == namespace["DOC_ANALYST_SELECTED_PATH"]
    assert run_cfg.paths.full_10k_cleaning_contract == "current"
    assert run_cfg.paths.full_10k_text_feature_batch_size == 11
    assert run_cfg.paths.mda_text_feature_batch_size == 13
    assert run_cfg.paths.recompute_event_screen_surface is True
    assert run_cfg.paths.recompute_event_panel is True
    assert run_cfg.paths.recompute_regression_tables is True
    assert run_cfg.paths.event_window_doc_batch_size == 17
    assert run_cfg.fail_closed_for_enabled_stages is True
    assert run_cfg.enabled_stages == runner.resolve_enabled_lm2011_stage_names_from_env()


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
        assert manifest["stages"][stage_name]["reason"] == runner.NO_ESTIMABLE_QUARTERLY_FAMA_MACBETH_QUARTERS

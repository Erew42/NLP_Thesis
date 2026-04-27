from __future__ import annotations

import json
import importlib.util
import sys
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_assets.api import build_single_asset
from thesis_assets.builders.sentence_summaries import SENTENCE_BATCH_SIZE_ENV_VAR
from thesis_assets.builders.sentence_summaries import _lm_negative_sentence_share
from thesis_assets.builders.sample_contracts import common_row_comparison
from thesis_assets.builders.sample_contracts import common_success_comparison
from thesis_assets.builders.sample_contracts import ownership_common_support
from thesis_assets.builders.sample_contracts import raw_available
from thesis_assets.builders.sample_contracts import regression_eligible
from thesis_assets.cli.__main__ import main as cli_main
from thesis_assets.config.constants import DEFAULT_COMMON_SUCCESS_POLICY
from thesis_assets.errors import RegistryError
from thesis_assets.errors import SampleContractError
from thesis_assets.figures import build_metric_panel_ecdf_figure
from thesis_assets.registry import loader
from thesis_assets.specs import BuildResult
from thesis_assets.specs import BuildSessionResult
from thesis_assets.usage import resolve_usage_run_paths


def test_registry_loader_imports_expected_assets() -> None:
    assets = loader.load_registry()
    assert [asset.asset_id for asset in assets] == [
        "ch4_sample_attrition_lm2011_1994_2008",
        "ch4_sample_funnel_raw_to_final_lm2011",
        "ch4_sample_attrition_losses_lm2011",
        "ch4_sample_stage_bridge_lm2011",
        "ch4_full_10k_regression_sample_summary",
        "ch4_extension_attrition_ladder",
        "ch4_no_ownership_c0_specification",
        "ch4_ownership_analyst_coverage_diagnostics",
        "ch4_ownership_coverage_by_year",
        "ch4_item_cleaning_eligibility_diagnostics",
        "ch4_item_cleaning_quality_by_year",
        "ch4_dictionary_provenance_summary",
        "ch4_finbert_inference_manifest_summary",
        "ch4_finbert_segment_token_diagnostics",
        "ch4_score_family_descriptive_statistics",
        "ch4_variable_definitions",
        "ch5_lm2011_full_10k_return_coefficients",
        "ch5_lm2011_portfolio_long_short",
        "ch5_lm2011_portfolio_formation_diagnostics",
        "ch5_portfolio_cumulative_q5_minus_q1",
        "ch5_fit_horserace_item7_c0",
        "ch5_extension_c0_fit_summary",
        "ch5_extension_c0_fit_comparisons",
        "ch5_extension_fit_delta_path",
        "ch5_lm2011_table_vi_no_ownership_outcomes",
        "ch5_nw_lag_baseline_reconciliation",
        "ch5_nw_lag_core_no_ownership_appendix",
        "ch5_nw_lag_extension_coefficients_appendix",
        "ch5_nw_lag_extension_fit_comparisons_appendix",
        "ch5_concordance_item7_common_sample",
        "ch5_between_filing_ecdf_lm_negative_doc_scores",
        "ch5_between_filing_ecdf_finbert_doc_scores",
        "ch5_within_filing_sentence_ecdf_finbert_negative",
        "ch5_within_filing_sentence_ecdf_lm_negative_share",
        "ch5_within_filing_high_negative_sentence_share",
        "ch5_concordance_negative_scores_by_scope",
        "ch5_score_drift_by_year",
        "ch5_finbert_robustness_coefficients",
        "ch5_finbert_robustness_fit_comparisons",
        "ch5_matched_dictionary_finbert_coefficients_full",
        "ch5_fama_macbeth_skipped_quarter_diagnostics",
        "ch5_alternative_signal_robustness_full_grid",
        "ch5_full_controls_coefficient_appendix",
        "ch5_text_score_control_correlation_matrix",
        "ch5_research_question_evidence_map",
    ]


def test_registry_loader_rejects_duplicate_asset_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loader, "REGISTRY_MODULES", ("chapter4_descriptives", "chapter4_descriptives"))
    with pytest.raises(RegistryError, match="Duplicate asset_id"):
        loader.load_registry()


def test_sample_contract_helpers_apply_expected_selection() -> None:
    base_lf = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "value": [10.0, None, 30.0],
            "ownership_flag": [True, False, True],
        }
    ).lazy()
    raw_df = raw_available(base_lf, filters=(pl.col("id") >= 2,)).collect()
    assert raw_df.get_column("id").to_list() == [2, 3]

    eligible_df = regression_eligible(base_lf, required_columns=("value",)).collect()
    assert eligible_df.get_column("id").to_list() == [1, 3]

    ownership_df = ownership_common_support(
        base_lf,
        ownership_flag_column="ownership_flag",
    ).collect()
    assert ownership_df.get_column("id").to_list() == [1, 3]


def test_common_row_comparison_and_common_success_validation() -> None:
    left_lf = pl.DataFrame(
        {
            "doc_id": ["a", "b", "c"],
            "filing_date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
            "text_scope": ["item_7_mda", "item_7_mda", "item_7_mda"],
            "cleaning_policy_id": ["clean", "clean", "clean"],
            "lm_negative_tfidf": [1.0, None, 3.0],
        }
    ).lazy()
    right_lf = pl.DataFrame(
        {
            "doc_id": ["a", "b", "d"],
            "filing_date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 4)],
            "text_scope": ["item_7_mda", "item_7_mda", "item_7_mda"],
            "cleaning_policy_id": ["clean", "clean", "clean"],
            "finbert_neg_prob_lenw_mean": [0.1, 0.2, 0.4],
        }
    ).lazy()

    joined = common_row_comparison(
        left_lf,
        right_lf,
        join_keys=("doc_id", "filing_date", "text_scope", "cleaning_policy_id"),
        left_signal_columns=("lm_negative_tfidf",),
        right_signal_columns=("finbert_neg_prob_lenw_mean",),
    ).collect()
    assert joined.get_column("doc_id").to_list() == ["a"]

    fit_lf = pl.DataFrame(
        {
            "common_success_policy": [DEFAULT_COMMON_SUCCESS_POLICY, DEFAULT_COMMON_SUCCESS_POLICY],
            "specification_name": ["dictionary_only", "finbert_only"],
        }
    ).lazy()
    assert common_success_comparison(
        fit_lf,
        expected_policy=DEFAULT_COMMON_SUCCESS_POLICY,
    ).collect().height == 2

    bad_fit_lf = pl.DataFrame(
        {
            "common_success_policy": ["wrong_policy"],
            "specification_name": ["dictionary_only"],
        }
    ).lazy()
    with pytest.raises(SampleContractError, match="Expected common-success policy"):
        common_success_comparison(
            bad_fit_lf,
            expected_policy=DEFAULT_COMMON_SUCCESS_POLICY,
        ).collect()


def test_manifest_writing_for_single_asset(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    run_root = repo_root / "inputs" / "lm2011_post_refinitiv"
    run_root.mkdir(parents=True)
    _write_sample_attrition_parquet(run_root / "lm2011_table_i_sample_creation.parquet")

    result = build_single_asset(
        asset_id="ch4_sample_attrition_lm2011_1994_2008",
        run_id="unit_manifest",
        repo_root=repo_root,
        lm2011_post_refinitiv_dir=run_root,
    )

    asset_result = result.asset_results["ch4_sample_attrition_lm2011_1994_2008"]
    assert asset_result.status == "completed"
    assert Path(asset_result.output_paths["csv"]).exists()
    assert Path(asset_result.output_paths["tex"]).exists()
    assert Path(asset_result.output_paths["table_preview"]).exists()

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["asset_statuses"]["ch4_sample_attrition_lm2011_1994_2008"] == "completed"
    assert manifest["assets"]["ch4_sample_attrition_lm2011_1994_2008"]["sample_contract_id"] == "raw_available"


def test_chapter4_sample_funnel_figure_outputs(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    run_root = repo_root / "inputs" / "lm2011_post_refinitiv"
    run_root.mkdir(parents=True)
    _write_sample_attrition_parquet(run_root / "lm2011_table_i_sample_creation.parquet")

    result = build_single_asset(
        asset_id="ch4_sample_funnel_raw_to_final_lm2011",
        run_id="unit_ch4_figure",
        repo_root=repo_root,
        lm2011_post_refinitiv_dir=run_root,
    )

    asset_result = result.asset_results["ch4_sample_funnel_raw_to_final_lm2011"]
    assert asset_result.status == "completed"
    assert Path(asset_result.output_paths["csv"]).exists()
    assert Path(asset_result.output_paths["png"]).exists()
    assert Path(asset_result.output_paths["pdf"]).exists()


def test_chapter5_table_vi_no_ownership_asset_uses_validation_alias(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    run_root = repo_root / "inputs" / "lm2011_table_vi_validation_second_pass"
    run_root.mkdir(parents=True)
    _write_table_vi_no_ownership_parquet(run_root / "lm2011_table_vi_results_no_ownership_validation.parquet")

    result = build_single_asset(
        asset_id="ch5_lm2011_table_vi_no_ownership_outcomes",
        run_id="unit_table_vi",
        repo_root=repo_root,
        lm2011_post_refinitiv_dir=run_root,
    )

    asset_result = result.asset_results["ch5_lm2011_table_vi_no_ownership_outcomes"]
    assert asset_result.status == "completed"
    assert asset_result.row_counts == {
        "table_rows": 42,
        "outcome_count": 3,
        "signal_count": 14,
    }
    assert Path(asset_result.output_paths["csv"]).exists()
    assert Path(asset_result.output_paths["tex"]).exists()
    assert Path(asset_result.output_paths["table_preview"]).exists()

    table_df = pl.read_csv(asset_result.output_paths["csv"])
    assert table_df.height == 42
    assert table_df.get_column("outcome").unique().sort().to_list() == [
        "Abnormal volume",
        "Filing-period excess return",
        "Postevent return volatility",
    ]
    assert set(table_df.get_column("weighting").unique().to_list()) == {"Proportional", "tf-idf"}
    assert "H4N-Inf" in table_df.get_column("signal").unique().to_list()
    scales = table_df.group_by("outcome").agg(pl.col("reported_scale").unique().sort()).sort("outcome")
    assert scales.to_dicts() == [
        {"outcome": "Abnormal volume", "reported_scale": ["raw"]},
        {"outcome": "Filing-period excess return", "reported_scale": ["x100"]},
        {"outcome": "Postevent return volatility", "reported_scale": ["x100"]},
    ]


def test_chapter5_table_vi_no_ownership_asset_falls_back_to_validation_pack(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    run_root = repo_root / "full_data_run" / "lm2011_post_refinitiv"
    validation_root = repo_root / "full_data_run" / "lm2011_table_vi_validation_second_pass"
    run_root.mkdir(parents=True)
    validation_root.mkdir(parents=True)
    _write_incomplete_table_vi_no_ownership_parquet(run_root / "lm2011_table_vi_results_no_ownership.parquet")
    _write_table_vi_no_ownership_parquet(validation_root / "lm2011_table_vi_results_no_ownership_validation.parquet")

    result = build_single_asset(
        asset_id="ch5_lm2011_table_vi_no_ownership_outcomes",
        run_id="unit_table_vi_fallback",
        repo_root=repo_root,
        lm2011_post_refinitiv_dir=run_root,
    )

    asset_result = result.asset_results["ch5_lm2011_table_vi_no_ownership_outcomes"]
    assert asset_result.status == "completed"
    assert asset_result.row_counts["table_rows"] == 42
    assert asset_result.warnings
    assert asset_result.resolved_inputs["table_vi_results_no_ownership"] == str(
        (validation_root / "lm2011_table_vi_results_no_ownership_validation.parquet").resolve()
    )


def test_chapter5_full_10k_return_coefficients_asset(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    run_root = repo_root / "inputs" / "lm2011_post_refinitiv"
    run_root.mkdir(parents=True)
    _write_table_iv_return_coefficients_parquet(run_root / "lm2011_table_iv_results_no_ownership.parquet")

    result = build_single_asset(
        asset_id="ch5_lm2011_full_10k_return_coefficients",
        run_id="unit_table_iv_full_10k",
        repo_root=repo_root,
        lm2011_post_refinitiv_dir=run_root,
    )

    asset_result = result.asset_results["ch5_lm2011_full_10k_return_coefficients"]
    assert asset_result.status == "completed"
    assert asset_result.row_counts == {"table_rows": 4}

    table_df = pl.read_csv(asset_result.output_paths["csv"])
    assert table_df.get_column("signal").to_list() == [
        "H4N-Inf proportion",
        "LM negative proportion",
        "H4N-Inf tf-idf",
        "LM negative tf-idf",
    ]
    assert table_df.get_column("estimate_x100").to_list() == pytest.approx([0.1, -0.2, 0.3, -0.4])


def test_chapter5_full_10k_return_coefficients_missing_column_fails(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    run_root = repo_root / "inputs" / "lm2011_post_refinitiv"
    run_root.mkdir(parents=True)
    pl.DataFrame({"text_scope": ["full_10k"]}).write_parquet(
        run_root / "lm2011_table_iv_results_no_ownership.parquet"
    )

    result = build_single_asset(
        asset_id="ch5_lm2011_full_10k_return_coefficients",
        run_id="unit_table_iv_missing_column",
        repo_root=repo_root,
        lm2011_post_refinitiv_dir=run_root,
    )

    asset_result = result.asset_results["ch5_lm2011_full_10k_return_coefficients"]
    assert asset_result.status == "failed"
    assert "missing required columns" in (asset_result.failure_reason or "")


def test_chapter5_portfolio_table_prefers_latest_local_rerun_and_warns(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    run_root = repo_root / "inputs" / "lm2011_post_refinitiv"
    rerun_root = repo_root / "full_data_run" / "lm2011_table_ia_ii_local_rerun_sample_20260425_000000"
    run_root.mkdir(parents=True)
    rerun_root.mkdir(parents=True)
    _write_portfolio_table_ia_ii_parquet(rerun_root / "lm2011_table_ia_ii_results.rerun.parquet")

    result = build_single_asset(
        asset_id="ch5_lm2011_portfolio_long_short",
        run_id="unit_portfolio_rerun",
        repo_root=repo_root,
        lm2011_post_refinitiv_dir=run_root,
    )

    asset_result = result.asset_results["ch5_lm2011_portfolio_long_short"]
    assert asset_result.status == "completed"
    assert asset_result.row_counts == {"table_rows": 3}
    assert any("local Table IA.II rerun artifact" in warning for warning in asset_result.warnings)
    assert asset_result.resolved_inputs["table_ia_ii_results"] == str(
        (rerun_root / "lm2011_table_ia_ii_results.rerun.parquet").resolve()
    )

    table_df = pl.read_csv(asset_result.output_paths["csv"])
    assert set(table_df.get_column("q1_return").unique().to_list()) == {"not stored"}
    assert set(table_df.get_column("q5_return").unique().to_list()) == {"not stored"}
    assert set(table_df.get_column("spread_definition").unique().to_list()) == {
        "Q5 - Q1; Q5 = most negative filings; Q1 = least negative filings"
    }


def test_chapter5_portfolio_cumulative_q5_minus_q1_asset(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    run_root = repo_root / "inputs" / "lm2011_post_refinitiv"
    rerun_root = repo_root / "full_data_run" / "lm2011_table_ia_ii_local_rerun_sample_20260425_010000"
    run_root.mkdir(parents=True)
    rerun_root.mkdir(parents=True)
    _write_portfolio_monthly_returns_parquet(
        rerun_root / "lm2011_trading_strategy_monthly_returns.rerun.parquet"
    )

    result = build_single_asset(
        asset_id="ch5_portfolio_cumulative_q5_minus_q1",
        run_id="unit_portfolio_cumulative",
        repo_root=repo_root,
        lm2011_post_refinitiv_dir=run_root,
    )

    asset_result = result.asset_results["ch5_portfolio_cumulative_q5_minus_q1"]
    assert asset_result.status == "completed"
    assert asset_result.row_counts == {"monthly_rows": 4}
    assert Path(asset_result.output_paths["csv"]).exists()
    assert Path(asset_result.output_paths["png"]).exists()
    assert Path(asset_result.output_paths["pdf"]).exists()

    figure_df = pl.read_csv(asset_result.output_paths["csv"])
    assert set(figure_df.get_column("spread_definition").unique().to_list()) == {
        "Q5 - Q1; Q5 = most negative filings; Q1 = least negative filings"
    }
    fin_neg = figure_df.filter(pl.col("sort_signal_name") == "fin_neg_prop").sort("portfolio_month")
    assert fin_neg.get_column("cumulative_q5_minus_q1_return").to_list() == pytest.approx([0.10, 0.045])


def test_chapter5_extension_c0_fit_summary_asset_filters_to_c0(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    extension_root = repo_root / "inputs" / "lm2011_extension"
    extension_root.mkdir(parents=True)
    _write_extension_fit_summary_parquet(extension_root / "lm2011_extension_fit_summary.parquet")

    result = build_single_asset(
        asset_id="ch5_extension_c0_fit_summary",
        run_id="unit_extension_c0_fit_summary",
        repo_root=repo_root,
        lm2011_extension_dir=extension_root,
    )

    asset_result = result.asset_results["ch5_extension_c0_fit_summary"]
    assert asset_result.status == "completed"
    assert asset_result.row_counts == {"table_rows": 12}

    table_df = pl.read_csv(asset_result.output_paths["csv"])
    assert table_df.height == 12
    assert set(table_df.get_column("scope").unique().to_list()) == {
        "Item 7 MD&A",
        "Item 1A risk factors",
    }
    assert set(table_df.get_column("dictionary_family").unique().to_list()) == {
        "Replication dictionary",
        "Extended dictionary",
    }
    assert set(table_df.get_column("specification").unique().to_list()) == {
        "Dictionary only",
        "FinBERT only",
        "Dictionary + FinBERT",
    }
    assert 999 not in table_df.get_column("total_n_obs").to_list()


def test_chapter5_extension_c0_fit_comparisons_asset_filters_and_stars(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    extension_root = repo_root / "inputs" / "lm2011_extension"
    extension_root.mkdir(parents=True)
    _write_extension_fit_comparisons_parquet(extension_root / "lm2011_extension_fit_comparisons.parquet")

    result = build_single_asset(
        asset_id="ch5_extension_c0_fit_comparisons",
        run_id="unit_extension_c0_fit_comparisons",
        repo_root=repo_root,
        lm2011_extension_dir=extension_root,
    )

    asset_result = result.asset_results["ch5_extension_c0_fit_comparisons"]
    assert asset_result.status == "completed"
    assert asset_result.row_counts == {"table_rows": 12}

    table_df = pl.read_csv(asset_result.output_paths["csv"])
    assert table_df.height == 12
    assert set(table_df.get_column("comparison").unique().to_list()) == {
        "FinBERT - dictionary",
        "Joint - dictionary",
        "Joint - FinBERT",
    }
    assert 999 not in table_df.get_column("total_n_obs").to_list()
    stars_by_comparison = {
        row["comparison"]: row["stars"]
        for row in table_df.select("comparison", "stars").unique().to_dicts()
    }
    assert stars_by_comparison == {
        "FinBERT - dictionary": "",
        "Joint - dictionary": "**",
        "Joint - FinBERT": "*",
    }


def test_chapter5_extension_fit_delta_path_asset_filters_to_c0_replication(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    extension_root = repo_root / "inputs" / "lm2011_extension"
    extension_root.mkdir(parents=True)
    _write_extension_fit_difference_quarterly_parquet(
        extension_root / "lm2011_extension_fit_difference_quarterly.parquet"
    )

    result = build_single_asset(
        asset_id="ch5_extension_fit_delta_path",
        run_id="unit_extension_fit_delta_path",
        repo_root=repo_root,
        lm2011_extension_dir=extension_root,
    )

    asset_result = result.asset_results["ch5_extension_fit_delta_path"]
    assert asset_result.status == "completed"
    assert asset_result.row_counts == {"quarterly_rows": 12}
    assert Path(asset_result.output_paths["csv"]).exists()
    assert Path(asset_result.output_paths["png"]).exists()
    assert Path(asset_result.output_paths["pdf"]).exists()

    figure_df = pl.read_csv(asset_result.output_paths["csv"])
    assert set(figure_df.get_column("comparison").unique().to_list()) == {
        "FinBERT - dictionary",
        "Joint - dictionary",
        "Joint - FinBERT",
    }
    assert set(figure_df.get_column("dictionary_family_source").unique().to_list()) == {"replication"}
    assert 999 not in figure_df.get_column("n_obs").to_list()


def test_chapter5_nw_lag_core_no_ownership_appendix_asset(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    sensitivity_root = repo_root / "inputs" / "lm2011_nw_lag_sensitivity"
    sensitivity_root.mkdir(parents=True)
    _write_nw_core_sensitivity_parquet(sensitivity_root / "core_tables_nw_lag_sensitivity.parquet")

    result = build_single_asset(
        asset_id="ch5_nw_lag_core_no_ownership_appendix",
        run_id="unit_nw_core",
        repo_root=repo_root,
        lm2011_nw_lag_sensitivity_dir=sensitivity_root,
    )

    asset_result = result.asset_results["ch5_nw_lag_core_no_ownership_appendix"]
    assert asset_result.status == "completed"
    table_df = pl.read_csv(asset_result.output_paths["csv"])
    assert {"t_nw1", "t_nw2", "t_nw3", "t_nw4"}.issubset(set(table_df.columns))

    table_iv = table_df.filter(
        (pl.col("table_or_surface") == "Table IV full 10-K returns")
        & (pl.col("coefficient") == "H4N-Inf proportion")
    )
    assert table_iv.select("estimate").item() == pytest.approx(0.1)
    assert table_iv.select("reported_scale").item() == "x100"
    assert table_iv.select("t_nw4").item() == pytest.approx(1.2)

    abnormal_volume = table_df.filter(
        (pl.col("outcome") == "Abnormal volume")
        & (pl.col("coefficient") == "H4N-Inf proportion")
    )
    assert abnormal_volume.select("estimate").item() == pytest.approx(0.002)
    assert abnormal_volume.select("reported_scale").item() == "raw"


def test_chapter5_nw_lag_extension_appendix_assets(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    sensitivity_root = repo_root / "inputs" / "lm2011_nw_lag_sensitivity"
    sensitivity_root.mkdir(parents=True)
    _write_nw_extension_results_sensitivity_parquet(
        sensitivity_root / "extension_results_nw_lag_sensitivity.parquet"
    )
    _write_nw_extension_fit_comparisons_sensitivity_parquet(
        sensitivity_root / "extension_fit_comparisons_nw_lag_sensitivity.parquet"
    )

    coefficient_result = build_single_asset(
        asset_id="ch5_nw_lag_extension_coefficients_appendix",
        run_id="unit_nw_extension_coefficients",
        repo_root=repo_root,
        lm2011_nw_lag_sensitivity_dir=sensitivity_root,
    ).asset_results["ch5_nw_lag_extension_coefficients_appendix"]
    assert coefficient_result.status == "completed"
    coefficient_df = pl.read_csv(coefficient_result.output_paths["csv"])
    assert set(coefficient_df.get_column("scope").unique().to_list()) == {
        "Item 7 MD&A",
        "Item 1A risk factors",
    }
    item7_finbert = coefficient_df.filter(
        (pl.col("scope") == "Item 7 MD&A")
        & (pl.col("specification") == "C0 FinBERT only")
        & (pl.col("coefficient") == "FinBERT negative probability")
    )
    assert item7_finbert.select("estimate").item() == pytest.approx(-2.0)
    assert item7_finbert.select("reported_scale").item() == "x100"
    assert item7_finbert.select("t_nw1").item() == pytest.approx(-3.0)

    fit_result = build_single_asset(
        asset_id="ch5_nw_lag_extension_fit_comparisons_appendix",
        run_id="unit_nw_extension_fit",
        repo_root=repo_root,
        lm2011_nw_lag_sensitivity_dir=sensitivity_root,
    ).asset_results["ch5_nw_lag_extension_fit_comparisons_appendix"]
    assert fit_result.status == "completed"
    fit_df = pl.read_csv(fit_result.output_paths["csv"])
    assert {"C0 Joint - FinBERT", "C1 Joint - FinBERT", "C2 Joint - FinBERT"}.issubset(
        set(fit_df.get_column("specification").to_list())
    )
    assert set(fit_df.get_column("reported_scale").unique().to_list()) == {"raw"}


def test_chapter5_nw_lag_baseline_reconciliation_asset(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    post_root = repo_root / "inputs" / "lm2011_post_refinitiv"
    extension_root = repo_root / "inputs" / "lm2011_extension" / "replication"
    sensitivity_root = repo_root / "inputs" / "lm2011_nw_lag_sensitivity"
    post_root.mkdir(parents=True)
    extension_root.mkdir(parents=True)
    sensitivity_root.mkdir(parents=True)

    _write_table_iv_return_coefficients_parquet(post_root / "lm2011_table_iv_results_no_ownership.parquet")
    _write_table_vi_no_ownership_parquet(post_root / "lm2011_table_vi_results_no_ownership.parquet")
    _write_extension_results_parquet(extension_root / "lm2011_extension_results.parquet")
    _write_extension_fit_comparisons_parquet(extension_root / "lm2011_extension_fit_comparisons.parquet")
    _write_nw_core_sensitivity_parquet(sensitivity_root / "core_tables_nw_lag_sensitivity.parquet")
    _write_nw_extension_results_sensitivity_parquet(
        sensitivity_root / "extension_results_nw_lag_sensitivity.parquet"
    )
    _write_nw_extension_fit_comparisons_sensitivity_parquet(
        sensitivity_root / "extension_fit_comparisons_nw_lag_sensitivity.parquet"
    )

    result = build_single_asset(
        asset_id="ch5_nw_lag_baseline_reconciliation",
        run_id="unit_nw_reconciliation",
        repo_root=repo_root,
        lm2011_post_refinitiv_dir=post_root,
        lm2011_extension_dir=extension_root.parent,
        lm2011_nw_lag_sensitivity_dir=sensitivity_root,
    )

    asset_result = result.asset_results["ch5_nw_lag_baseline_reconciliation"]
    assert asset_result.status == "completed"
    table_df = pl.read_csv(asset_result.output_paths["csv"])
    assert table_df.height == 4
    assert table_df.get_column("max_abs_estimate_diff").max() == pytest.approx(0.0)
    assert table_df.get_column("max_abs_t_diff").max() == pytest.approx(0.0)


def test_chapter5_nw_lag_missing_artifact_failure_is_recorded(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    sensitivity_root = repo_root / "inputs" / "lm2011_nw_lag_sensitivity"
    sensitivity_root.mkdir(parents=True)

    result = build_single_asset(
        asset_id="ch5_nw_lag_core_no_ownership_appendix",
        run_id="unit_nw_missing",
        repo_root=repo_root,
        lm2011_nw_lag_sensitivity_dir=sensitivity_root,
    )

    asset_result = result.asset_results["ch5_nw_lag_core_no_ownership_appendix"]
    assert asset_result.status == "failed"
    assert "core_tables_nw_lag_sensitivity" in (asset_result.failure_reason or "")


def test_chapter4_full_10k_regression_sample_summary_asset(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    run_root = repo_root / "inputs" / "lm2011_post_refinitiv"
    run_root.mkdir(parents=True)
    _write_full_10k_regression_panel(run_root / "lm2011_return_regression_panel_full_10k.parquet")

    result = build_single_asset(
        asset_id="ch4_full_10k_regression_sample_summary",
        run_id="unit_full_10k_summary",
        repo_root=repo_root,
        lm2011_post_refinitiv_dir=run_root,
    )

    asset_result = result.asset_results["ch4_full_10k_regression_sample_summary"]
    assert asset_result.status == "completed"
    table_df = pl.read_csv(asset_result.output_paths["csv"])
    assert "Filing-period excess return" in table_df.get_column("variable").to_list()
    assert table_df.filter(pl.col("variable") == "Log size").select("non_null").item() == 3


def test_chapter5_text_score_control_correlation_matrix_asset(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    run_root = repo_root / "inputs" / "lm2011_post_refinitiv"
    run_root.mkdir(parents=True)
    _write_full_10k_regression_panel(run_root / "lm2011_return_regression_panel_full_10k.parquet")

    result = build_single_asset(
        asset_id="ch5_text_score_control_correlation_matrix",
        run_id="unit_correlation_matrix",
        repo_root=repo_root,
        lm2011_post_refinitiv_dir=run_root,
    )

    asset_result = result.asset_results["ch5_text_score_control_correlation_matrix"]
    assert asset_result.status == "completed"
    assert asset_result.row_counts == {"table_rows": 81}

    table_df = pl.read_csv(asset_result.output_paths["csv"])
    diagonal = table_df.filter(
        (pl.col("row_variable") == "LM negative proportion")
        & (pl.col("column_variable") == "LM negative proportion")
    )
    assert diagonal.select("correlation").item() == pytest.approx(1.0)


def test_chapter4_ownership_analyst_coverage_warns_without_refinitiv_inputs(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    extension_root = repo_root / "inputs" / "lm2011_extension"
    (extension_root / "replication").mkdir(parents=True)
    _write_extension_panel_with_ownership(
        extension_root / "replication" / "lm2011_extension_analysis_panel.parquet"
    )

    result = build_single_asset(
        asset_id="ch4_ownership_analyst_coverage_diagnostics",
        run_id="unit_coverage_diagnostics",
        repo_root=repo_root,
        lm2011_extension_dir=extension_root,
    )

    asset_result = result.asset_results["ch4_ownership_analyst_coverage_diagnostics"]
    assert asset_result.status == "completed"
    assert any("No Refinitiv analyst coverage artifact" in warning for warning in asset_result.warnings)

    table_df = pl.read_csv(asset_result.output_paths["csv"])
    assert set(table_df.get_column("scope").to_list()) == {"Item 7 MD&A", "Item 1A risk factors"}
    assert table_df.get_column("analyst_request_rows").null_count() == table_df.height


def test_chapter4_ownership_coverage_by_year_asset(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    extension_root = repo_root / "inputs" / "lm2011_extension"
    (extension_root / "replication").mkdir(parents=True)
    _write_extension_panel_with_ownership(
        extension_root / "replication" / "lm2011_extension_analysis_panel.parquet"
    )

    result = build_single_asset(
        asset_id="ch4_ownership_coverage_by_year",
        run_id="unit_ownership_coverage_by_year",
        repo_root=repo_root,
        lm2011_extension_dir=extension_root,
    )

    asset_result = result.asset_results["ch4_ownership_coverage_by_year"]
    assert asset_result.status == "completed"
    assert asset_result.row_counts == {"figure_rows": 9}
    assert Path(asset_result.output_paths["csv"]).exists()
    assert Path(asset_result.output_paths["png"]).exists()
    assert Path(asset_result.output_paths["pdf"]).exists()

    figure_df = pl.read_csv(asset_result.output_paths["csv"])
    assert set(figure_df.get_column("coverage_metric").unique().to_list()) == {
        "unrestricted_panel",
        "ownership_available",
        "ownership_common_support",
    }
    unrestricted = figure_df.filter(pl.col("coverage_metric") == "unrestricted_panel")
    assert set(unrestricted.get_column("coverage_rate").unique().to_list()) == {1.0}


def test_chapter4_finbert_manifest_summary_uses_run_manifest_json(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    finbert_root = repo_root / "inputs" / "finbert_run"
    finbert_root.mkdir(parents=True)
    (finbert_root / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_name": "finbert_unit",
                "runtime": {"model_name": "ProsusAI/finbert"},
                "bucket_lengths": {"short": 128, "medium": 256, "long": 512},
            }
        ),
        encoding="utf-8",
    )
    _write_finbert_yearly_summary(finbert_root / "model_inference_yearly_summary.parquet")

    result = build_single_asset(
        asset_id="ch4_finbert_inference_manifest_summary",
        run_id="unit_finbert_manifest",
        repo_root=repo_root,
        finbert_run_dir=finbert_root,
    )

    asset_result = result.asset_results["ch4_finbert_inference_manifest_summary"]
    assert asset_result.status == "completed"
    assert asset_result.row_counts == {"table_rows": 4}

    table_df = pl.read_csv(asset_result.output_paths["csv"])
    model_row = table_df.filter((pl.col("section") == "manifest") & (pl.col("metric") == "model"))
    assert model_row.select("value").item() == "ProsusAI/finbert"


def test_chapter4_item_cleaning_diagnostics_aggregates_shards(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    finbert_root = repo_root / "inputs" / "finbert_run"
    preprocessing_root = repo_root / "inputs" / "_staged_intermediates" / "finbert_run_sentence_preprocessing"
    finbert_root.mkdir(parents=True)
    preprocessing_root.mkdir(parents=True)
    _write_finbert_yearly_summary(finbert_root / "model_inference_yearly_summary.parquet")
    _write_cleaning_diagnostics(preprocessing_root / "item_scope_cleaning_diagnostics.parquet")

    result = build_single_asset(
        asset_id="ch4_item_cleaning_eligibility_diagnostics",
        run_id="unit_cleaning_diagnostics",
        repo_root=repo_root,
        finbert_run_dir=finbert_root,
    )

    asset_result = result.asset_results["ch4_item_cleaning_eligibility_diagnostics"]
    assert asset_result.status == "completed"
    assert asset_result.row_counts == {"table_rows": 2}

    table_df = pl.read_csv(asset_result.output_paths["csv"])
    item7 = table_df.filter((pl.col("calendar_year") == 2020) & (pl.col("scope") == "Item 7 MD&A"))
    assert item7.select("n_filings_candidate").item() == 30
    assert item7.select("n_rows_after_cleaning").item() == 24
    assert item7.select("extraction_rate").item() == pytest.approx(28 / 30)
    assert item7.select("token_count_mean").item() == pytest.approx(((100.0 * 8) + (200.0 * 16)) / 24)


def test_chapter4_item_cleaning_quality_by_year_asset(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    finbert_root = repo_root / "inputs" / "finbert_run"
    preprocessing_root = repo_root / "inputs" / "_staged_intermediates" / "finbert_run_sentence_preprocessing"
    finbert_root.mkdir(parents=True)
    preprocessing_root.mkdir(parents=True)
    _write_finbert_yearly_summary(finbert_root / "model_inference_yearly_summary.parquet")
    _write_cleaning_diagnostics(preprocessing_root / "item_scope_cleaning_diagnostics.parquet")

    result = build_single_asset(
        asset_id="ch4_item_cleaning_quality_by_year",
        run_id="unit_cleaning_quality_by_year",
        repo_root=repo_root,
        finbert_run_dir=finbert_root,
    )

    asset_result = result.asset_results["ch4_item_cleaning_quality_by_year"]
    assert asset_result.status == "completed"
    assert asset_result.row_counts == {"figure_rows": 6}
    assert Path(asset_result.output_paths["csv"]).exists()
    assert Path(asset_result.output_paths["png"]).exists()
    assert Path(asset_result.output_paths["pdf"]).exists()

    figure_df = pl.read_csv(asset_result.output_paths["csv"])
    assert set(figure_df.get_column("quality_metric").unique().to_list()) == {
        "extraction_rate",
        "cleaned_scope_rate",
        "manual_review_share",
    }
    item7_extraction = figure_df.filter(
        (pl.col("scope") == "Item 7 MD&A") & (pl.col("quality_metric") == "extraction_rate")
    )
    assert item7_extraction.select("metric_value").item() == pytest.approx(28 / 30)


def test_chapter4_finbert_segment_token_diagnostics_asset(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    finbert_root = repo_root / "inputs" / "finbert_run"
    finbert_root.mkdir(parents=True)
    _write_finbert_item_features_long_parquet(finbert_root / "item_features_long.parquet")

    result = build_single_asset(
        asset_id="ch4_finbert_segment_token_diagnostics",
        run_id="unit_finbert_segment_tokens",
        repo_root=repo_root,
        finbert_run_dir=finbert_root,
    )

    asset_result = result.asset_results["ch4_finbert_segment_token_diagnostics"]
    assert asset_result.status == "completed"
    assert asset_result.row_counts == {"summary_rows": 3}
    assert Path(asset_result.output_paths["csv"]).exists()
    assert Path(asset_result.output_paths["png"]).exists()
    assert Path(asset_result.output_paths["pdf"]).exists()

    summary_df = pl.read_csv(asset_result.output_paths["csv"])
    item7_2020 = summary_df.filter((pl.col("filing_year") == 2020) & (pl.col("scope") == "Item 7 MD&A"))
    assert item7_2020.select("item_scope_rows").item() == 2
    assert item7_2020.select("token_count_512_median").item() == pytest.approx(250.0)


def test_chapter4_variable_definitions_and_chapter5_evidence_map_include_sign_convention(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    variable_result = build_single_asset(
        asset_id="ch4_variable_definitions",
        run_id="unit_variable_definitions",
        repo_root=repo_root,
    ).asset_results["ch4_variable_definitions"]
    assert variable_result.status == "completed"
    assert variable_result.row_counts == {"table_rows": 6}
    variable_df = pl.read_csv(variable_result.output_paths["csv"])
    assert "Q5 - Q1; Q5 = most negative filings; Q1 = least negative filings" in " ".join(
        variable_df.get_column("definition").to_list()
    )

    evidence_result = build_single_asset(
        asset_id="ch5_research_question_evidence_map",
        run_id="unit_evidence_map",
        repo_root=repo_root,
    ).asset_results["ch5_research_question_evidence_map"]
    assert evidence_result.status == "completed"
    assert evidence_result.row_counts == {"table_rows": 5}
    evidence_df = pl.read_csv(evidence_result.output_paths["csv"])
    assert "ch5_extension_c0_fit_comparisons" in " ".join(evidence_df.get_column("evidence_asset").to_list())
    assert "Q5 - Q1; Q5 = most negative filings; Q1 = least negative filings" in " ".join(
        evidence_df.get_column("claim_guardrail").to_list()
    )


def test_chapter5_score_drift_by_year_asset_filters_item_scopes(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    extension_root = repo_root / "inputs" / "lm2011_extension"
    (extension_root / "replication").mkdir(parents=True)
    _write_extension_analysis_panel(extension_root / "replication" / "lm2011_extension_analysis_panel.parquet")

    result = build_single_asset(
        asset_id="ch5_score_drift_by_year",
        run_id="unit_score_drift_by_year",
        repo_root=repo_root,
        lm2011_extension_dir=extension_root,
    )

    asset_result = result.asset_results["ch5_score_drift_by_year"]
    assert asset_result.status == "completed"
    assert asset_result.row_counts == {"figure_rows": 4}
    assert Path(asset_result.output_paths["csv"]).exists()
    assert Path(asset_result.output_paths["png"]).exists()
    assert Path(asset_result.output_paths["pdf"]).exists()

    figure_df = pl.read_csv(asset_result.output_paths["csv"])
    assert set(figure_df.get_column("scope").unique().to_list()) == {
        "Item 7 MD&A",
        "Item 1A risk factors",
    }
    assert set(figure_df.get_column("score_metric").unique().to_list()) == {
        "lm_negative_tfidf",
        "finbert_neg_prob_lenw_mean",
    }


def test_lm_negative_doc_score_ecdf_uses_separate_metric_panels(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    extension_root = repo_root / "inputs" / "lm2011_extension"
    (extension_root / "replication").mkdir(parents=True)
    _write_extension_dictionary_surface(
        extension_root / "replication" / "lm2011_extension_dictionary_surface.parquet"
    )

    result = build_single_asset(
        asset_id="ch5_between_filing_ecdf_lm_negative_doc_scores",
        run_id="unit_lm_negative_doc_score_ecdf",
        repo_root=repo_root,
        lm2011_extension_dir=extension_root,
    )

    asset_result = result.asset_results["ch5_between_filing_ecdf_lm_negative_doc_scores"]
    assert asset_result.status == "completed"
    assert asset_result.row_counts == {"ecdf_rows": 8, "score_rows": 8}
    assert Path(asset_result.output_paths["csv"]).exists()
    assert Path(asset_result.output_paths["png"]).exists()
    assert Path(asset_result.output_paths["pdf"]).exists()

    ecdf_df = pl.read_csv(asset_result.output_paths["csv"])
    series_counts = (
        ecdf_df.group_by(["metric_id", "text_scope"])
        .agg(
            pl.len().alias("rows"),
            pl.col("score").max().alias("max_score"),
            pl.col("total_count").max().alias("total_count"),
        )
        .sort(["metric_id", "text_scope"])
    )
    assert series_counts.select(["metric_id", "text_scope", "rows", "total_count"]).to_dicts() == [
        {"metric_id": "lm_negative_prop", "text_scope": "item_1a_risk_factors", "rows": 2, "total_count": 2},
        {"metric_id": "lm_negative_prop", "text_scope": "item_7_mda", "rows": 2, "total_count": 2},
        {"metric_id": "lm_negative_tfidf", "text_scope": "item_1a_risk_factors", "rows": 2, "total_count": 2},
        {"metric_id": "lm_negative_tfidf", "text_scope": "item_7_mda", "rows": 2, "total_count": 2},
    ]
    item7_tfidf_max = series_counts.filter(
        (pl.col("metric_id") == "lm_negative_tfidf") & (pl.col("text_scope") == "item_7_mda")
    ).select("max_score").item()
    assert item7_tfidf_max == pytest.approx(100.0)

    figure = build_metric_panel_ecdf_figure(
        ecdf_df,
        metric_panels=(
            ("lm_negative_prop", "Proportion", "LM2011 negative proportion"),
            ("lm_negative_tfidf", "tf-idf", "LM2011 negative tf-idf"),
        ),
        x_col="score",
    )
    try:
        assert len(figure.axes) == 2
        assert [axis.get_xlabel() for axis in figure.axes] == [
            "LM2011 negative proportion",
            "LM2011 negative tf-idf",
        ]
    finally:
        plt.close(figure)


def test_lm_sentence_negative_share_scoring() -> None:
    negative_words = frozenset({"bad", "loss"})
    assert _lm_negative_sentence_share("Bad loss was offset by growth.", negative_words) == pytest.approx(2 / 6)
    assert _lm_negative_sentence_share("Growth improved.", negative_words) == pytest.approx(0.0)
    assert _lm_negative_sentence_share("1234", negative_words) is None


def test_sentence_level_lm_ecdf_uses_shards_and_tiny_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    extension_root = repo_root / "inputs" / "lm2011_extension"
    finbert_root = repo_root / "inputs" / "finbert_run"
    (extension_root / "replication").mkdir(parents=True)
    (finbert_root / "sentence_scores" / "by_year").mkdir(parents=True)
    _write_extension_analysis_panel(extension_root / "replication" / "lm2011_extension_analysis_panel.parquet")
    _write_sentence_score_shard(finbert_root / "sentence_scores" / "by_year" / "2020.parquet")
    _write_replication_negative_word_list(
        repo_root / "full_data_run" / "LM2011_additional_data" / "generated_dictionary_families" / "replication" / "Fin-Neg.txt"
    )
    monkeypatch.setenv(SENTENCE_BATCH_SIZE_ENV_VAR, "1")

    result = build_single_asset(
        asset_id="ch5_within_filing_sentence_ecdf_lm_negative_share",
        run_id="unit_sentence_lm",
        repo_root=repo_root,
        lm2011_extension_dir=extension_root,
        finbert_run_dir=finbert_root,
    )

    asset_result = result.asset_results["ch5_within_filing_sentence_ecdf_lm_negative_share"]
    assert asset_result.status == "completed"
    assert Path(asset_result.output_paths["csv"]).exists()
    assert Path(asset_result.output_paths["png"]).exists()
    assert Path(asset_result.output_paths["pdf"]).exists()

    ecdf_df = pl.read_csv(asset_result.output_paths["csv"])
    totals = (
        ecdf_df.group_by("text_scope")
        .agg(pl.col("total_count").max().alias("total_count"))
        .sort("text_scope")
    )
    assert totals.to_dicts() == [
        {"text_scope": "item_1a_risk_factors", "total_count": 1},
        {"text_scope": "item_7_mda", "total_count": 2},
    ]


def test_missing_artifact_failure_is_recorded_in_manifest(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    run_root = repo_root / "inputs" / "lm2011_post_refinitiv"
    run_root.mkdir(parents=True)

    result = build_single_asset(
        asset_id="ch4_sample_attrition_lm2011_1994_2008",
        run_id="unit_missing",
        repo_root=repo_root,
        lm2011_post_refinitiv_dir=run_root,
    )

    asset_result = result.asset_results["ch4_sample_attrition_lm2011_1994_2008"]
    assert asset_result.status == "failed"
    assert "could not be resolved" in (asset_result.failure_reason or "")

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["asset_statuses"]["ch4_sample_attrition_lm2011_1994_2008"] == "failed"
    assert "could not be resolved" in manifest["assets"]["ch4_sample_attrition_lm2011_1994_2008"]["failure_reason"]


def test_cli_and_api_use_same_build_path(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    run_root = repo_root / "inputs" / "lm2011_post_refinitiv"
    run_root.mkdir(parents=True)
    _write_sample_attrition_parquet(run_root / "lm2011_table_i_sample_creation.parquet")

    api_result = build_single_asset(
        asset_id="ch4_sample_attrition_lm2011_1994_2008",
        run_id="api_run",
        repo_root=repo_root,
        lm2011_post_refinitiv_dir=run_root,
    )
    cli_exit = cli_main(
        [
            "build-asset",
            "--asset-id",
            "ch4_sample_attrition_lm2011_1994_2008",
            "--run-id",
            "cli_run",
            "--repo-root",
            str(repo_root),
            "--lm2011-post-refinitiv-dir",
            str(run_root),
        ]
    )
    assert cli_exit == 0

    api_manifest = json.loads(api_result.manifest_path.read_text(encoding="utf-8"))
    cli_manifest = json.loads((repo_root / "output" / "thesis_assets" / "cli_run" / "manifest.json").read_text(encoding="utf-8"))
    assert api_manifest["asset_statuses"] == cli_manifest["asset_statuses"]
    assert set(api_manifest["assets"]) == set(cli_manifest["assets"])


def test_usage_run_paths_prefer_unified_runner_layout(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    unified_root = (
        repo_root
        / "full_data_run"
        / "sample_5pct_seed42"
        / "results"
        / "sec_ccm_unified_runner"
        / "local_sample"
    )
    (unified_root / "lm2011_post_refinitiv").mkdir(parents=True)
    (unified_root / "lm2011_extension").mkdir(parents=True)
    finbert_run = unified_root / "finbert_item_analysis" / "run_a"
    finbert_run.mkdir(parents=True)
    (finbert_run / "run_manifest.json").write_text("{}", encoding="utf-8")
    (finbert_run / "item_features_long.parquet").touch()

    fallback_post = repo_root / "full_data_run" / "lm2011_post_refinitiv"
    fallback_ext = repo_root / "full_data_run" / "lm2011_extension"
    sensitivity_root = repo_root / "full_data_run" / "lm2011_nw_lag_sensitivity_local_monitored"
    fallback_post.mkdir(parents=True)
    fallback_ext.mkdir(parents=True)
    sensitivity_root.mkdir(parents=True)
    (sensitivity_root / "core_tables_nw_lag_sensitivity.parquet").touch()
    (sensitivity_root / "extension_results_nw_lag_sensitivity.parquet").touch()
    (sensitivity_root / "extension_fit_comparisons_nw_lag_sensitivity.parquet").touch()

    resolved = resolve_usage_run_paths(
        repo_root=repo_root,
        data_profile="LOCAL_REPO",
    )

    assert resolved["lm2011_post_refinitiv_dir"] == (unified_root / "lm2011_post_refinitiv").resolve()
    assert resolved["lm2011_extension_dir"] == (unified_root / "lm2011_extension").resolve()
    assert resolved["lm2011_nw_lag_sensitivity_dir"] == sensitivity_root.resolve()
    assert resolved["finbert_run_dir"] == finbert_run.resolve()


def test_usage_run_paths_support_versioned_snapshot_and_drive_layouts(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    full_data_root = repo_root / "full_data_run"
    versioned_post = full_data_root / "lm2011_post_refinitiv-20260421T173344Z-3-001" / "lm2011_post_refinitiv"
    versioned_ext = full_data_root / "lm2011_extension-20260421T114544Z-3-001" / "lm2011_extension"
    versioned_post.mkdir(parents=True)
    versioned_ext.mkdir(parents=True)

    local_resolved = resolve_usage_run_paths(
        repo_root=repo_root,
        data_profile="LOCAL_REPO",
    )
    assert local_resolved["lm2011_post_refinitiv_dir"] == versioned_post.resolve()
    assert local_resolved["lm2011_extension_dir"] == versioned_ext.resolve()

    drive_data_root = tmp_path / "content" / "drive" / "MyDrive" / "Data_LM"
    unified_drive_root = drive_data_root / "results" / "sec_ccm_unified_runner"
    (unified_drive_root / "lm2011_post_refinitiv").mkdir(parents=True)
    (unified_drive_root / "lm2011_extension").mkdir(parents=True)
    drive_finbert_run = unified_drive_root / "finbert_item_analysis" / "finbert_item_analysis_2026-04-20T105101+0000"
    drive_finbert_run.mkdir(parents=True)
    (drive_finbert_run / "run_manifest.json").write_text("{}", encoding="utf-8")
    (drive_finbert_run / "item_features_long.parquet").touch()

    colab_resolved = resolve_usage_run_paths(
        repo_root=repo_root,
        data_profile="COLAB_DRIVE",
        drive_data_root=drive_data_root,
    )
    assert colab_resolved["lm2011_post_refinitiv_dir"] == (unified_drive_root / "lm2011_post_refinitiv").resolve()
    assert colab_resolved["lm2011_extension_dir"] == (unified_drive_root / "lm2011_extension").resolve()
    assert colab_resolved["finbert_run_dir"] == drive_finbert_run.resolve()


def test_tools_entrypoint_build_asset_emits_json_and_allows_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module_path = REPO_ROOT / "tools" / "build_thesis_assets.py"
    spec = importlib.util.spec_from_file_location("test_build_thesis_assets_tool", module_path)
    assert spec is not None
    assert spec.loader is not None
    tool_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool_module)

    repo_root = tmp_path / "repo"
    output_root = tmp_path / "drive" / "Data_LM" / "results" / "thesis_assets" / "tool_run"
    manifest_path = output_root / "manifest.json"
    resolved_paths = {
        "lm2011_post_refinitiv_dir": tmp_path / "inputs" / "lm2011_post_refinitiv",
        "lm2011_extension_dir": tmp_path / "inputs" / "lm2011_extension",
        "lm2011_nw_lag_sensitivity_dir": tmp_path / "inputs" / "lm2011_nw_lag_sensitivity",
        "finbert_run_dir": tmp_path / "inputs" / "finbert_item_analysis",
        "finbert_robustness_dir": tmp_path / "inputs" / "finbert_robustness",
    }

    monkeypatch.setattr(tool_module, "_resolve_run_paths", lambda **_: resolved_paths)

    captured_kwargs: dict[str, object] = {}

    def _fake_build_single_asset(**kwargs: object) -> BuildSessionResult:
        captured_kwargs.update(kwargs)
        return BuildSessionResult(
            run_id="tool_run",
            output_root=output_root,
            manifest_path=manifest_path,
            asset_results={
                "ch5_fit_horserace_item7_c0": BuildResult(
                    asset_id="ch5_fit_horserace_item7_c0",
                    chapter="chapter5",
                    asset_kind="table",
                    sample_contract_id="common_success_comparison",
                    status="failed",
                    resolved_inputs={"fit_summary": "C:/tmp/lm2011_extension_fit_summary.parquet"},
                    output_paths={},
                    row_counts={},
                    failure_reason="required artifact missing",
                )
            },
        )

    monkeypatch.setattr(tool_module, "build_single_asset", _fake_build_single_asset)

    exit_code = tool_module.main(
        [
            "build-asset",
            "--asset-id",
            "ch5_fit_horserace_item7_c0",
            "--run-id",
            "tool_run",
            "--repo-root",
            str(repo_root),
            "--output-root",
            str(output_root),
            "--allow-failures",
        ]
    )

    assert exit_code == 0
    assert captured_kwargs["repo_root"] == repo_root.resolve()
    assert captured_kwargs["output_root"] == output_root.resolve()
    assert captured_kwargs["run_id"] == "tool_run"

    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "build-asset"
    assert payload["run_id"] == "tool_run"
    assert payload["output_root"] == str(output_root)
    assert payload["manifest_path"] == str(manifest_path)
    assert payload["asset_statuses"] == {"ch5_fit_horserace_item7_c0": "failed"}
    assert payload["resolved_paths"] == {key: str(value) for key, value in resolved_paths.items()}


def _write_sample_attrition_parquet(path: Path) -> None:
    df = pl.DataFrame(
        {
            "section_id": ["full_10k_document", "full_10k_document"],
            "section_label": ["Full 10-K Document", "Full 10-K Document"],
            "section_order": [1, 1],
            "row_order": [1, 2],
            "row_id": ["edgar_complete_nonduplicate_sample", "first_filing_per_year"],
            "display_label": [
                "EDGAR 10-K/10-K405 1994-2008 complete sample (excluding duplicates)",
                "Include only first filing in a given year",
            ],
            "sample_size_kind": ["count", "count"],
            "sample_size_value": [121995.0, 120350.0],
            "observations_removed": [None, 1645],
            "availability_status": ["available", "available"],
            "availability_reason": [None, None],
        }
    )
    df.write_parquet(path)


def _write_table_vi_no_ownership_parquet(path: Path) -> None:
    outcomes = (
        "filing_period_excess_return",
        "abnormal_volume",
        "postevent_return_volatility",
    )
    signals = (
        "h4n_inf_prop",
        "lm_negative_prop",
        "lm_positive_prop",
        "lm_uncertainty_prop",
        "lm_litigious_prop",
        "lm_modal_strong_prop",
        "lm_modal_weak_prop",
        "h4n_inf_tfidf",
        "lm_negative_tfidf",
        "lm_positive_tfidf",
        "lm_uncertainty_tfidf",
        "lm_litigious_tfidf",
        "lm_modal_strong_tfidf",
        "lm_modal_weak_tfidf",
    )
    rows = []
    for outcome_idx, outcome in enumerate(outcomes, start=1):
        for signal_idx, signal in enumerate(signals, start=1):
            rows.append(
                {
                    "table_id": "table_vi_full_10k_dictionary_surface",
                    "specification_id": f"{outcome}__{signal}",
                    "text_scope": "full_10k",
                    "signal_name": signal,
                    "dependent_variable": outcome,
                    "coefficient_name": signal,
                    "estimate": 0.001 * outcome_idx * signal_idx,
                    "standard_error": 0.0001 * signal_idx,
                    "t_stat": 1.5,
                    "n_quarters": 60,
                    "mean_quarter_n": 750.0,
                    "weighting_rule": "quarter_observation_count",
                    "nw_lags": 1,
                }
            )
    pl.DataFrame(rows).write_parquet(path)


def _write_incomplete_table_vi_no_ownership_parquet(path: Path) -> None:
    pl.DataFrame(
        {
            "table_id": ["table_vi_full_10k_dictionary_surface"],
            "specification_id": ["filing_period_excess_return__lm_negative_prop"],
            "text_scope": ["full_10k"],
            "signal_name": ["lm_negative_prop"],
            "dependent_variable": ["filing_period_excess_return"],
            "coefficient_name": ["lm_negative_prop"],
            "estimate": [0.001],
            "standard_error": [0.0001],
            "t_stat": [1.5],
            "n_quarters": [60],
            "mean_quarter_n": [750.0],
            "weighting_rule": ["quarter_observation_count"],
            "nw_lags": [1],
        }
    ).write_parquet(path)


def _write_table_iv_return_coefficients_parquet(path: Path) -> None:
    rows = []
    for signal_name, estimate, standard_error, t_stat in (
        ("h4n_inf_prop", 0.001, 0.0002, 1.5),
        ("lm_negative_prop", -0.002, 0.0003, -2.0),
        ("h4n_inf_tfidf", 0.003, 0.0004, 2.5),
        ("lm_negative_tfidf", -0.004, 0.0005, -3.0),
    ):
        rows.append(
            {
                "table_id": "table_iv_full_10k",
                "specification_id": signal_name,
                "text_scope": "full_10k",
                "signal_name": signal_name,
                "dependent_variable": "filing_period_excess_return",
                "coefficient_name": signal_name,
                "estimate": estimate,
                "standard_error": standard_error,
                "t_stat": t_stat,
                "n_quarters": 60,
                "mean_quarter_n": 100.0,
                "weighting_rule": "quarter_observation_count",
                "nw_lags": 1,
            }
        )
    pl.DataFrame(rows).write_parquet(path)


def _write_portfolio_table_ia_ii_parquet(path: Path) -> None:
    pl.DataFrame(
        {
            "signal_name": ["fin_neg_prop", "fin_neg_prop", "fin_neg_prop"],
            "dependent_variable": ["long_short_return", "long_short_return", "long_short_return"],
            "coefficient_name": ["mean_long_short_return", "alpha_ff3_mom", "r2"],
            "estimate": [0.002, 0.001, 0.12],
            "t_stat": [1.1, 0.9, None],
        }
    ).write_parquet(path)


def _write_portfolio_monthly_returns_parquet(path: Path) -> None:
    pl.DataFrame(
        {
            "portfolio_month": [
                date(2020, 1, 31),
                date(2020, 2, 29),
                date(2020, 1, 31),
                date(2020, 2, 29),
            ],
            "sort_signal_name": ["fin_neg_prop", "fin_neg_prop", "h4n_inf_prop", "h4n_inf_prop"],
            "long_short_return": [0.10, -0.05, 0.00, 0.02],
        }
    ).write_parquet(path)


def _write_extension_fit_summary_parquet(path: Path) -> None:
    rows = []
    spec_rows = (
        ("dictionary_only", "lm2011_frozen", "lm_negative_tfidf", 0.030),
        ("finbert_only", "finbert", "finbert_neg_prob_lenw_mean", 0.031),
        (
            "dictionary_finbert_joint",
            "dictionary_plus_finbert",
            "lm_negative_tfidf,finbert_neg_prob_lenw_mean",
            0.032,
        ),
    )
    for family in ("replication", "extended"):
        for text_scope in ("item_7_mda", "item_1a_risk_factors"):
            for spec_idx, (specification_name, feature_family, signal_name, base_adj_r2) in enumerate(spec_rows):
                rows.append(
                    {
                        "run_id": "unit",
                        "sample_window": "2009_2024",
                        "text_scope": text_scope,
                        "outcome_name": "filing_period_excess_return",
                        "feature_family": feature_family,
                        "control_set_id": "C0",
                        "control_set_alias": "no_ownership",
                        "specification_name": specification_name,
                        "signal_name": signal_name,
                        "signal_inputs": signal_name,
                        "n_quarters": 62,
                        "total_n_obs": 37511 + spec_idx,
                        "mean_quarter_n": 605.0,
                        "weighted_avg_raw_r2": base_adj_r2 + 0.01,
                        "weighted_avg_adj_r2": base_adj_r2,
                        "equal_quarter_avg_raw_r2": base_adj_r2 + 0.02,
                        "equal_quarter_avg_adj_r2": base_adj_r2 + 0.01,
                        "weighting_rule": "quarter_observation_count",
                        "common_success_policy": DEFAULT_COMMON_SUCCESS_POLICY,
                        "estimator_status": "estimated",
                        "failure_reason": None,
                        "dictionary_family_source": family,
                    }
                )
    rows.append({**rows[0], "control_set_id": "C1", "total_n_obs": 999})
    rows.append({**rows[0], "control_set_id": "C2", "total_n_obs": 999})
    pl.DataFrame(rows).write_parquet(path)


def _write_extension_fit_comparisons_parquet(path: Path) -> None:
    rows = []
    comparison_rows = (
        ("finbert_minus_dictionary", 0.001, 1.1, 0.20),
        ("joint_minus_dictionary", 0.002, 2.2, 0.04),
        ("joint_minus_finbert", 0.003, 1.8, 0.08),
    )
    for family in ("replication", "extended"):
        for text_scope in ("item_7_mda", "item_1a_risk_factors"):
            for comparison_name, delta, t_stat, p_value in comparison_rows:
                rows.append(
                    {
                        "run_id": "unit",
                        "sample_window": "2009_2024",
                        "text_scope": text_scope,
                        "outcome_name": "filing_period_excess_return",
                        "control_set_id": "C0",
                        "control_set_alias": "no_ownership",
                        "comparison_name": comparison_name,
                        "left_specification_name": "dictionary_only",
                        "left_signal_name": "lm_negative_tfidf",
                        "left_signal_inputs": "lm_negative_tfidf",
                        "right_specification_name": "finbert_only",
                        "right_signal_name": "finbert_neg_prob_lenw_mean",
                        "right_signal_inputs": "finbert_neg_prob_lenw_mean",
                        "n_quarters": 62,
                        "total_n_obs": 37511,
                        "mean_quarter_n": 605.0,
                        "weighted_avg_delta_raw_r2": delta + 0.01,
                        "weighted_avg_delta_adj_r2": delta,
                        "equal_quarter_avg_delta_raw_r2": delta + 0.02,
                        "equal_quarter_avg_delta_adj_r2": delta + 0.01,
                        "nw_lags": 1,
                        "nw_se_delta_adj_r2": 0.001,
                        "nw_t_stat_delta_adj_r2": t_stat,
                        "nw_p_value_delta_adj_r2": p_value,
                        "weighting_rule": "quarter_observation_count",
                        "common_success_policy": DEFAULT_COMMON_SUCCESS_POLICY,
                        "estimator_status": "estimated",
                        "failure_reason": None,
                        "dictionary_family_source": family,
                    }
                )
    rows.append({**rows[0], "control_set_id": "C1", "total_n_obs": 999})
    rows.append({**rows[0], "control_set_id": "C2", "total_n_obs": 999})
    pl.DataFrame(rows).write_parquet(path)


def _write_extension_results_parquet(path: Path) -> None:
    rows = []
    specs = (
        ("dictionary_only", "lm2011_frozen", "lm_negative_tfidf", "lm_negative_tfidf", -0.010, -2.0),
        ("finbert_only", "finbert", "finbert_neg_prob_lenw_mean", "finbert_neg_prob_lenw_mean", -0.020, -3.0),
        (
            "dictionary_finbert_joint",
            "dictionary_plus_finbert",
            "lm_negative_tfidf",
            "lm_negative_tfidf,finbert_neg_prob_lenw_mean",
            -0.011,
            -2.2,
        ),
        (
            "dictionary_finbert_joint",
            "dictionary_plus_finbert",
            "finbert_neg_prob_lenw_mean",
            "lm_negative_tfidf,finbert_neg_prob_lenw_mean",
            -0.021,
            -3.2,
        ),
    )
    for text_scope in ("item_7_mda", "item_1a_risk_factors"):
        for specification_name, feature_family, coefficient_name, signal_name, estimate, t_stat in specs:
            rows.append(
                {
                    "run_id": "unit",
                    "sample_window": "2009_2024",
                    "text_scope": text_scope,
                    "outcome_name": "filing_period_excess_return",
                    "feature_family": feature_family,
                    "control_set_id": "C0",
                    "control_set_alias": "no_ownership",
                    "specification_name": specification_name,
                    "coefficient_name": coefficient_name,
                    "signal_name": signal_name,
                    "estimate": estimate,
                    "standard_error": abs(estimate / t_stat),
                    "t_stat": t_stat,
                    "p_value": 0.04,
                    "n_obs": 1000,
                    "n_quarters": 62,
                    "mean_quarter_n": 500.0,
                    "average_r2": 0.03,
                    "weighting_rule": "quarter_observation_count",
                    "nw_lags": 1,
                    "estimator_status": "estimated",
                    "failure_reason": None,
                    "dictionary_family_source": "replication",
                }
            )
    pl.DataFrame(rows).write_parquet(path)


def _write_nw_core_sensitivity_parquet(path: Path) -> None:
    base_rows = []
    for signal_name, estimate, t_stat in (
        ("h4n_inf_prop", 0.001, 1.5),
        ("lm_negative_prop", -0.002, -2.0),
        ("h4n_inf_tfidf", 0.003, 2.5),
        ("lm_negative_tfidf", -0.004, -3.0),
    ):
        base_rows.append(
            {
                "table_id": "table_iv_full_10k",
                "specification_id": signal_name,
                "text_scope": "full_10k",
                "signal_name": signal_name,
                "dependent_variable": "filing_period_excess_return",
                "coefficient_name": signal_name,
                "estimate": estimate,
                "base_t_stat": t_stat,
                "n_quarters": 60,
                "mean_quarter_n": 100.0,
                "weighting_rule": "quarter_observation_count",
                "stage_name": "table_iv_results_no_ownership",
            }
        )

    outcomes = (
        "filing_period_excess_return",
        "abnormal_volume",
        "postevent_return_volatility",
    )
    signals = (
        "h4n_inf_prop",
        "lm_negative_prop",
        "lm_positive_prop",
        "lm_uncertainty_prop",
        "lm_litigious_prop",
        "lm_modal_strong_prop",
        "lm_modal_weak_prop",
        "h4n_inf_tfidf",
        "lm_negative_tfidf",
        "lm_positive_tfidf",
        "lm_uncertainty_tfidf",
        "lm_litigious_tfidf",
        "lm_modal_strong_tfidf",
        "lm_modal_weak_tfidf",
    )
    for outcome_idx, outcome in enumerate(outcomes, start=1):
        for signal_idx, signal in enumerate(signals, start=1):
            base_rows.append(
                {
                    "table_id": "table_vi_full_10k_dictionary_surface",
                    "specification_id": f"{outcome}__{signal}",
                    "text_scope": "full_10k",
                    "signal_name": signal,
                    "dependent_variable": outcome,
                    "coefficient_name": signal,
                    "estimate": 0.001 * outcome_idx * signal_idx,
                    "base_t_stat": 1.5,
                    "n_quarters": 60,
                    "mean_quarter_n": 750.0,
                    "weighting_rule": "quarter_observation_count",
                    "stage_name": "table_vi_results_no_ownership",
                }
            )

    rows = []
    for base_row in base_rows:
        base_t_stat = float(base_row.pop("base_t_stat"))
        for lag in (1, 2, 3, 4):
            t_stat = _lagged_unit_t_stat(base_t_stat, lag)
            estimate = float(base_row["estimate"])
            rows.append(
                {
                    **base_row,
                    "standard_error": abs(estimate / t_stat),
                    "t_stat": t_stat,
                    "nw_lags": lag,
                }
            )
    pl.DataFrame(rows).write_parquet(path)


def _write_nw_extension_results_sensitivity_parquet(path: Path) -> None:
    base_rows = []
    specs = (
        ("dictionary_only", "lm2011_frozen", "lm_negative_tfidf", "lm_negative_tfidf", -0.010, -2.0),
        ("finbert_only", "finbert", "finbert_neg_prob_lenw_mean", "finbert_neg_prob_lenw_mean", -0.020, -3.0),
        (
            "dictionary_finbert_joint",
            "dictionary_plus_finbert",
            "lm_negative_tfidf",
            "lm_negative_tfidf,finbert_neg_prob_lenw_mean",
            -0.011,
            -2.2,
        ),
        (
            "dictionary_finbert_joint",
            "dictionary_plus_finbert",
            "finbert_neg_prob_lenw_mean",
            "lm_negative_tfidf,finbert_neg_prob_lenw_mean",
            -0.021,
            -3.2,
        ),
    )
    for text_scope in ("item_7_mda", "item_1a_risk_factors"):
        for specification_name, feature_family, coefficient_name, signal_name, estimate, t_stat in specs:
            base_rows.append(
                {
                    "run_id": "unit",
                    "sample_window": "2009_2024",
                    "text_scope": text_scope,
                    "outcome_name": "filing_period_excess_return",
                    "feature_family": feature_family,
                    "control_set_id": "C0",
                    "control_set_alias": "no_ownership",
                    "specification_name": specification_name,
                    "coefficient_name": coefficient_name,
                    "signal_name": signal_name,
                    "estimate": estimate,
                    "base_t_stat": t_stat,
                    "n_obs": 1000,
                    "n_quarters": 62,
                    "mean_quarter_n": 500.0,
                    "average_r2": 0.03,
                    "weighting_rule": "quarter_observation_count",
                    "estimator_status": "estimated",
                    "failure_reason": None,
                    "dictionary_family_source": "replication",
                }
            )
    rows = []
    for base_row in base_rows:
        base_t_stat = float(base_row.pop("base_t_stat"))
        for lag in (1, 2, 3, 4):
            t_stat = _lagged_unit_t_stat(base_t_stat, lag)
            estimate = float(base_row["estimate"])
            rows.append(
                {
                    **base_row,
                    "standard_error": abs(estimate / t_stat),
                    "t_stat": t_stat,
                    "p_value": 0.04,
                    "nw_lags": lag,
                }
            )
    pl.DataFrame(rows).write_parquet(path)


def _write_nw_extension_fit_comparisons_sensitivity_parquet(path: Path) -> None:
    base_rows = []
    for family in ("replication", "extended"):
        for text_scope in ("item_7_mda", "item_1a_risk_factors"):
            for control_set_id in ("C0", "C1", "C2"):
                for comparison_name, delta, t_stat in (
                    ("finbert_minus_dictionary", 0.001, 1.1),
                    ("joint_minus_dictionary", 0.002, 2.2),
                    ("joint_minus_finbert", 0.003, 1.8),
                ):
                    base_rows.append(
                        {
                            "run_id": "unit",
                            "sample_window": "2009_2024",
                            "text_scope": text_scope,
                            "outcome_name": "filing_period_excess_return",
                            "control_set_id": control_set_id,
                            "control_set_alias": "no_ownership",
                            "comparison_name": comparison_name,
                            "left_specification_name": "dictionary_only",
                            "left_signal_name": "lm_negative_tfidf",
                            "left_signal_inputs": "lm_negative_tfidf",
                            "right_specification_name": "finbert_only",
                            "right_signal_name": "finbert_neg_prob_lenw_mean",
                            "right_signal_inputs": "finbert_neg_prob_lenw_mean",
                            "n_quarters": 62,
                            "total_n_obs": 1000,
                            "mean_quarter_n": 500.0,
                            "weighted_avg_delta_raw_r2": delta + 0.01,
                            "weighted_avg_delta_adj_r2": delta,
                            "equal_quarter_avg_delta_raw_r2": delta + 0.02,
                            "equal_quarter_avg_delta_adj_r2": delta + 0.01,
                            "base_t_stat": t_stat,
                            "weighting_rule": "quarter_observation_count",
                            "common_success_policy": DEFAULT_COMMON_SUCCESS_POLICY,
                            "estimator_status": "estimated",
                            "failure_reason": None,
                            "dictionary_family_source": family,
                        }
                    )
    rows = []
    for base_row in base_rows:
        base_t_stat = float(base_row.pop("base_t_stat"))
        for lag in (1, 2, 3, 4):
            t_stat = _lagged_unit_t_stat(base_t_stat, lag)
            rows.append(
                {
                    **base_row,
                    "nw_lags": lag,
                    "nw_se_delta_adj_r2": abs(float(base_row["weighted_avg_delta_adj_r2"]) / t_stat),
                    "nw_t_stat_delta_adj_r2": t_stat,
                    "nw_p_value_delta_adj_r2": 0.04,
                }
            )
    pl.DataFrame(rows).write_parquet(path)


def _lagged_unit_t_stat(base_t_stat: float, lag: int) -> float:
    sign = -1.0 if base_t_stat < 0 else 1.0
    return sign * max(abs(base_t_stat) - (0.1 * (lag - 1)), 0.1)


def _write_extension_fit_difference_quarterly_parquet(path: Path) -> None:
    rows = []
    comparisons = (
        ("finbert_minus_dictionary", 0.001),
        ("joint_minus_dictionary", 0.002),
        ("joint_minus_finbert", 0.003),
    )
    quarters = (date(2020, 1, 1), date(2020, 4, 1))
    for family in ("replication", "extended"):
        for text_scope in ("item_7_mda", "item_1a_risk_factors"):
            for comparison_name, delta in comparisons:
                for quarter_idx, quarter_start in enumerate(quarters):
                    rows.append(
                        {
                            "run_id": "unit",
                            "sample_window": "2009_2024",
                            "text_scope": text_scope,
                            "outcome_name": "filing_period_excess_return",
                            "control_set_id": "C0",
                            "control_set_alias": "no_ownership",
                            "comparison_name": comparison_name,
                            "quarter_start": quarter_start,
                            "n_obs": 100 + quarter_idx,
                            "weight": 100.0 + quarter_idx,
                            "left_raw_r2": 0.10,
                            "right_raw_r2": 0.11,
                            "delta_raw_r2": delta + 0.01,
                            "left_adj_r2": 0.09,
                            "right_adj_r2": 0.10,
                            "delta_adj_r2": delta + (0.001 * quarter_idx),
                            "weighting_rule": "quarter_observation_count",
                            "common_success_policy": DEFAULT_COMMON_SUCCESS_POLICY,
                            "dictionary_family_source": family,
                        }
                    )
    rows.append({**rows[0], "control_set_id": "C1", "n_obs": 999})
    rows.append({**rows[0], "control_set_id": "C0", "dictionary_family_source": "extended", "n_obs": 999})
    pl.DataFrame(rows).write_parquet(path)


def _write_full_10k_regression_panel(path: Path) -> None:
    pl.DataFrame(
        {
            "doc_id": ["a", "b", "c"],
            "KYPERMNO": [1, 2, 3],
            "filing_date": [date(2020, 1, 1), date(2020, 2, 1), date(2020, 3, 1)],
            "text_scope": ["full_10k", "full_10k", "full_10k"],
            "filing_period_excess_return": [0.01, -0.02, 0.03],
            "lm_negative_prop": [0.1, 0.2, 0.3],
            "lm_negative_tfidf": [0.01, 0.02, 0.03],
            "h4n_inf_prop": [0.4, 0.5, 0.6],
            "h4n_inf_tfidf": [0.04, 0.05, 0.06],
            "log_size": [1.0, 2.0, 3.0],
            "log_book_to_market": [0.7, 0.8, 0.9],
            "log_share_turnover": [0.11, 0.12, 0.13],
            "pre_ffalpha": [0.001, 0.002, 0.003],
            "nasdaq_dummy": [0, 1, 0],
        }
    ).write_parquet(path)


def _write_extension_panel_with_ownership(path: Path) -> None:
    pl.DataFrame(
        {
            "doc_id": ["doc_a", "doc_b", "doc_c"],
            "filing_date": [date(2020, 1, 1), date(2020, 2, 1), date(2021, 1, 1)],
            "text_scope": ["item_7_mda", "item_1a_risk_factors", "item_7_mda"],
            "ownership_proxy_available": [True, False, True],
            "common_support_flag_ownership": [True, False, False],
        }
    ).write_parquet(path)


def _write_finbert_yearly_summary(path: Path) -> None:
    pl.DataFrame(
        {
            "filing_year": [2020, 2021],
            "status": ["completed", "completed"],
            "sentence_rows": [10, 20],
            "item_feature_rows": [2, 4],
            "doc_rows": [1, 2],
        }
    ).write_parquet(path)


def _write_cleaning_diagnostics(path: Path) -> None:
    pl.DataFrame(
        {
            "calendar_year": [2020, 2020, 2020],
            "text_scope": ["item_7_mda", "item_7_mda", "item_1a_risk_factors"],
            "n_filings_candidate": [10, 20, 5],
            "n_filings_extracted": [9, 19, 4],
            "extraction_rate": [0.9, 0.95, 0.8],
            "n_rows_after_cleaning": [8, 16, 4],
            "token_count_mean": [100.0, 200.0, 50.0],
            "token_count_median": [90.0, 180.0, 40.0],
            "token_count_p05": [10.0, 20.0, 5.0],
            "toc_trimmed_rows": [1, 2, 0],
            "tail_truncated_rows": [0, 3, 1],
            "reference_stub_rows": [0, 1, 0],
            "empty_after_cleaning_rows": [0, 0, 1],
            "large_removal_warning_rows": [0, 1, 0],
            "manual_audit_queue_n": [10, 20, 5],
            "activation_status": [
                "blocked_pending_manual_audit",
                "blocked_pending_manual_audit",
                "blocked_pending_manual_audit",
            ],
        }
    ).write_parquet(path)


def _write_finbert_item_features_long_parquet(path: Path) -> None:
    pl.DataFrame(
        {
            "doc_id": ["doc_a", "doc_b", "doc_c", "doc_d"],
            "filing_year": [2020, 2020, 2020, 2021],
            "text_scope": ["item_7_mda", "item_7_mda", "item_1a_risk_factors", "item_7_mda"],
            "sentence_count": [10, 20, 8, 15],
            "finbert_segment_count": [2, 4, 1, 3],
            "finbert_token_count_512_sum": [200, 300, 120, 250],
        }
    ).write_parquet(path)


def _write_extension_analysis_panel(path: Path) -> None:
    df = pl.DataFrame(
        {
            "doc_id": ["doc_a", "doc_b"],
            "filing_date": [date(2020, 1, 1), date(2020, 2, 1)],
            "text_scope": ["item_7_mda", "item_1a_risk_factors"],
            "dictionary_family": ["replication", "replication"],
            "lm_negative_tfidf": [0.020, 0.030],
            "finbert_neg_prob_lenw_mean": [0.400, 0.450],
        }
    )
    df.write_parquet(path)


def _write_extension_dictionary_surface(path: Path) -> None:
    df = pl.DataFrame(
        {
            "doc_id": ["doc_a", "doc_b", "doc_c", "doc_d", "doc_e"],
            "filing_date": [
                date(2020, 1, 1),
                date(2020, 2, 1),
                date(2020, 3, 1),
                date(2020, 4, 1),
                date(2020, 5, 1),
            ],
            "text_scope": [
                "item_7_mda",
                "item_7_mda",
                "item_1a_risk_factors",
                "item_1a_risk_factors",
                "item_7_mda",
            ],
            "dictionary_family": ["replication", "replication", "replication", "replication", "extended"],
            "lm_negative_prop": [0.01, 0.02, 0.04, 0.05, 0.99],
            "lm_negative_tfidf": [10.0, 100.0, 20.0, 30.0, 999.0],
        }
    )
    df.write_parquet(path)


def _write_sentence_score_shard(path: Path) -> None:
    df = pl.DataFrame(
        {
            "doc_id": ["doc_a", "doc_a", "doc_b", "doc_c"],
            "text_scope": ["item_7_mda", "item_7_mda", "item_1a_risk_factors", "item_7_mda"],
            "sentence_text": [
                "Bad loss was offset by growth.",
                "Growth improved.",
                "Bad risk remained.",
                "Bad loss outside the analysis universe.",
            ],
            "negative_prob": [0.98, 0.02, 0.70, 0.99],
        }
    )
    df.write_parquet(path)


def _write_replication_negative_word_list(path: Path) -> None:
    path.parent.mkdir(parents=True)
    path.write_text("bad\nloss\n", encoding="utf-8")

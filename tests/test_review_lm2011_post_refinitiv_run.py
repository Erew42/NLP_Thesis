from __future__ import annotations

import datetime as dt
import importlib.util
import json
from pathlib import Path
import sys

import polars as pl
import pytest


_REVIEW_SPEC = importlib.util.spec_from_file_location(
    "review_lm2011_post_refinitiv_run",
    Path(__file__).resolve().parents[1] / "tools" / "review_lm2011_post_refinitiv_run.py",
)
assert _REVIEW_SPEC is not None and _REVIEW_SPEC.loader is not None
review = importlib.util.module_from_spec(_REVIEW_SPEC)
sys.modules[_REVIEW_SPEC.name] = review
_REVIEW_SPEC.loader.exec_module(review)


def _write_parquet(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)


def _sample_creation_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "section_id": ["full_10k_document"],
            "sample_size_kind": ["count"],
            "display_label": ["EDGAR 10-K sample"],
            "sample_size_value": [12345],
            "observations_removed": [100],
            "availability_status": ["available"],
        }
    )


def _quarterly_results_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "table_id": ["table_iv_full_10k", "table_iv_full_10k"],
            "specification_id": ["lm_negative_prop", "lm_negative_prop"],
            "text_scope": ["full_10k", "full_10k"],
            "signal_name": ["lm_negative_prop", "lm_negative_prop"],
            "dependent_variable": ["filing_period_excess_return", "filing_period_excess_return"],
            "coefficient_name": ["lm_negative_prop", "institutional_ownership"],
            "estimate": [0.1, 0.2],
            "standard_error": [0.01, 0.02],
            "t_stat": [10.0, 10.0],
            "n_quarters": [4, 4],
            "mean_quarter_n": [20.0, 20.0],
            "weighting_rule": ["quarter_observation_count", "quarter_observation_count"],
            "nw_lags": [1, 1],
        }
    )


def _quarterly_results_no_ownership_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "table_id": ["table_iv_full_10k"],
            "specification_id": ["lm_negative_prop"],
            "text_scope": ["full_10k"],
            "signal_name": ["lm_negative_prop"],
            "dependent_variable": ["filing_period_excess_return"],
            "coefficient_name": ["lm_negative_prop"],
            "estimate": [0.15],
            "standard_error": [0.015],
            "t_stat": [10.0],
            "n_quarters": [4],
            "mean_quarter_n": [22.0],
            "weighting_rule": ["quarter_observation_count"],
            "nw_lags": [1],
        }
    )


def _trade_results_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "table_id": ["internet_appendix_table_ia_ii"] * 7,
            "specification_id": ["fin_neg_prop"] * 7,
            "text_scope": ["full_10k"] * 7,
            "signal_name": ["fin_neg_prop"] * 7,
            "dependent_variable": ["long_short_return"] * 7,
            "coefficient_name": [
                "mean_long_short_return",
                "alpha_ff3_mom",
                "beta_market",
                "beta_smb",
                "beta_hml",
                "beta_mom",
                "r2",
            ],
            "estimate": [-0.02, -0.01, 0.3, 0.1, 0.05, 0.04, 0.2],
            "standard_error": [None] * 7,
            "t_stat": [None] * 7,
            "n_quarters": [None] * 7,
            "mean_quarter_n": [None] * 7,
            "weighting_rule": [None] * 7,
            "nw_lags": [None] * 7,
        }
    )


def _trading_returns_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "portfolio_month": [dt.date(1997, 7, 31), dt.date(1997, 8, 31)],
            "sort_signal_name": ["fin_neg_prop", "fin_neg_prop"],
            "long_short_return": [-0.01, 0.02],
        }
    )


def _return_regression_panel_full_10k_df() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for idx in range(10):
        rows.append(
            {
                "doc_id": f"doc-{idx:02d}",
                "filing_period_excess_return": -0.05 + (idx * 0.01),
                "size_event": 1000.0 + (idx * 50.0),
                "bm_event": 0.4 + (idx * 0.02),
                "share_turnover": 0.8 + (idx * 0.1),
                "pre_ffalpha": -0.002 + (idx * 0.0004),
                "institutional_ownership": None if idx >= 7 else 0.20 + (idx * 0.04),
                "nasdaq_dummy": idx % 2,
                "h4n_inf_prop": 0.020 + (idx * 0.002),
                "lm_negative_prop": 0.010 + (idx * 0.001),
                "lm_positive_prop": 0.005 + (idx * 0.0004),
                "lm_uncertainty_prop": 0.008 + (idx * 0.0005),
                "lm_litigious_prop": 0.004 + (idx * 0.0003),
                "lm_modal_strong_prop": 0.001 + (idx * 0.0001),
                "lm_modal_weak_prop": 0.002 + (idx * 0.0002),
            }
        )
    return pl.DataFrame(rows)


def _return_regression_panel_mda_df() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for idx in range(10):
        rows.append(
            {
                "doc_id": f"doc-{idx:02d}",
                "filing_period_excess_return": -0.045 + (idx * 0.009),
                "size_event": 900.0 + (idx * 45.0),
                "bm_event": 0.35 + (idx * 0.018),
                "share_turnover": 0.7 + (idx * 0.09),
                "pre_ffalpha": -0.0018 + (idx * 0.00035),
                "institutional_ownership": None if idx >= 7 else 0.18 + (idx * 0.035),
                "nasdaq_dummy": (idx + 1) % 2,
                "h4n_inf_prop": 0.025 + (idx * 0.0025),
                "lm_negative_prop": 0.012 + (idx * 0.0012),
                "lm_positive_prop": 0.006 + (idx * 0.0004),
                "lm_uncertainty_prop": 0.009 + (idx * 0.0006),
                "lm_litigious_prop": 0.0045 + (idx * 0.0003),
                "lm_modal_strong_prop": 0.0012 + (idx * 0.0001),
                "lm_modal_weak_prop": 0.0022 + (idx * 0.0002),
                "total_token_count_mda": 400 if idx < 8 else 150,
            }
        )
    return pl.DataFrame(rows)


def _sue_panel_df() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for idx in range(7):
        rows.append(
            {
                "doc_id": f"doc-{idx:02d}",
                "sue": -0.001 + (idx * 0.0004),
                "analyst_dispersion": 0.0005 + (idx * 0.0001),
                "analyst_revisions": -0.0007 + (idx * 0.00015),
            }
        )
    return pl.DataFrame(rows)


def _extension_results_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "run_id": ["lm2011_extension"],
            "sample_window": ["2009_2024"],
            "text_scope": ["item_7_mda"],
            "outcome_name": ["filing_period_excess_return"],
            "feature_family": ["dictionary"],
            "control_set_id": ["C1"],
            "control_set_alias": ["C0_common_support_no_ownership"],
            "specification_name": ["dictionary_only"],
            "coefficient_name": ["lm_negative_tfidf"],
            "signal_name": ["lm_negative_tfidf"],
            "estimate": [0.05],
            "standard_error": [0.01],
            "t_stat": [5.0],
            "p_value": [0.001],
            "n_obs": [100],
            "n_quarters": [5],
            "mean_quarter_n": [20.0],
            "average_r2": [None],
            "weighting_rule": ["quarter_observation_count"],
            "nw_lags": [1],
            "estimator_status": ["estimated"],
            "failure_reason": [None],
        }
    )


def _extension_sample_loss_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "sample_window": ["2009_2024"],
            "calendar_year": [2009],
            "text_scope": ["item_7_mda"],
            "outcome_name": ["filing_period_excess_return"],
            "feature_family": ["dictionary"],
            "control_set_id": ["C1"],
            "control_set_alias": ["C0_common_support_no_ownership"],
            "specification_name": ["dictionary_only"],
            "n_control_set_rows": [120],
            "n_outcome_available": [120],
            "n_signal_available": [118],
            "n_controls_available": [117],
            "n_industry_available": [120],
            "n_estimation_rows": [117],
            "n_missing_outcome": [0],
            "n_missing_signal": [2],
            "n_missing_controls": [3],
            "n_missing_industry": [0],
        }
    )


def _build_core_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "lm2011_post_refinitiv"
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_parquet(run_dir / "lm2011_table_i_sample_creation.parquet", _sample_creation_df())
    _write_parquet(run_dir / "lm2011_table_i_sample_creation_1994_2024.parquet", _sample_creation_df())
    _write_parquet(run_dir / "lm2011_table_iv_results.parquet", _quarterly_results_df())
    _write_parquet(
        run_dir / "lm2011_table_iv_results_no_ownership.parquet",
        _quarterly_results_no_ownership_df(),
    )
    _write_parquet(run_dir / review.FULL_RETURN_PANEL_FILE, _return_regression_panel_full_10k_df())
    _write_parquet(run_dir / review.MDA_RETURN_PANEL_FILE, _return_regression_panel_mda_df())
    _write_parquet(run_dir / review.SUE_PANEL_FILE, _sue_panel_df())
    _write_parquet(run_dir / "lm2011_table_ia_ii_results.parquet", _trade_results_df())
    _write_parquet(run_dir / "lm2011_trading_strategy_monthly_returns.parquet", _trading_returns_df())

    manifest = {
        "run_status": "completed",
        "generated_at_utc": "2026-04-20T10:00:00+00:00",
        "completed_at_utc": "2026-04-20T10:05:00+00:00",
        "elapsed_seconds": 300,
        "stages": {
            "table_i_sample_creation": {
                "status": "generated",
                "artifact_path": str((run_dir / "lm2011_table_i_sample_creation.parquet").resolve()),
                "row_count": 1,
            },
            "table_i_sample_creation_1994_2024": {
                "status": "disabled_by_run_config",
                "artifact_path": str((run_dir / "lm2011_table_i_sample_creation_1994_2024.parquet").resolve()),
                "row_count": 1,
            },
            "table_iv_results": {
                "status": "generated",
                "artifact_path": str((run_dir / "lm2011_table_iv_results.parquet").resolve()),
                "row_count": 2,
            },
            "table_iv_results_no_ownership": {
                "status": "generated",
                "artifact_path": str((run_dir / "lm2011_table_iv_results_no_ownership.parquet").resolve()),
                "row_count": 1,
            },
            "table_ia_ii_results": {
                "status": "generated",
                "artifact_path": str((run_dir / "lm2011_table_ia_ii_results.parquet").resolve()),
                "row_count": 4,
            },
        },
    }
    (run_dir / review.MANIFEST_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir


def test_core_export_includes_existing_quarterly_tables_and_no_ownership_siblings(tmp_path: Path) -> None:
    run_dir = _build_core_run(tmp_path)
    output_dir = tmp_path / "report"

    exit_code = review.main(
        [
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    tex = (output_dir / "lm2011_post_refinitiv_review.tex").read_text(encoding="utf-8")
    assert "Table II" in tex
    assert "Summary Statistics for the 1994 to 2008 10-K Sample" in tex
    assert "Table IV (With Ownership Control)" in tex
    assert "Table IV (No Ownership Control)" in tex
    assert r"\shortstack[c]{10.000 \\ (10.00)}" in tex
    assert "Figure 1." in tex
    assert (output_dir / "tables" / "lm2011_table_ii_summary_statistics.csv").exists()


def test_coef_cell_scales_table_viii_tfidf_signal_to_lm2011_display_units() -> None:
    row = {
        "estimate": 1.955687e-07,
        "t_stat": 0.030881,
        "signal_name": "h4n_inf_tfidf",
        "coefficient_name": "h4n_inf_tfidf",
    }

    assert (
        review._coef_cell(row, estimate_scale=review._scale_table_viii_estimate)
        == r"\shortstack[c]{0.002 \\ (0.03)}"
    )


def test_paper_table_sections_render_table_viii_with_sue_heading_and_scaling(tmp_path: Path) -> None:
    run_dir = tmp_path / "lm2011_post_refinitiv"
    run_dir.mkdir(parents=True, exist_ok=True)

    df = pl.DataFrame(
        {
            "table_id": ["table_viii_sue", "table_viii_sue"],
            "specification_id": ["h4n_inf_prop", "h4n_inf_tfidf"],
            "text_scope": ["full_10k", "full_10k"],
            "signal_name": ["h4n_inf_prop", "h4n_inf_tfidf"],
            "dependent_variable": ["sue", "sue"],
            "coefficient_name": ["h4n_inf_prop", "h4n_inf_tfidf"],
            "estimate": [0.03795165, 1.955687e-07],
            "standard_error": [0.01, 1e-08],
            "t_stat": [2.19, 0.03],
            "n_quarters": [58, 58],
            "mean_quarter_n": [395.1, 395.1],
            "weighting_rule": ["quarter_observation_count", "quarter_observation_count"],
            "nw_lags": [1, 1],
        }
    )
    _write_parquet(run_dir / "lm2011_table_viii_results_no_ownership.parquet", df)

    sections = review._paper_table_sections(run_dir)
    joined = "\n".join(sections)

    assert "Standardized Unexpected Earnings Regressions" in joined
    assert "additional 100 for presentation" in joined
    assert r"\shortstack[c]{0.002 \\ (0.03)}" in joined


def test_full_export_fails_when_extension_artifacts_cannot_be_resolved(tmp_path: Path) -> None:
    run_dir = _build_core_run(tmp_path)
    output_dir = tmp_path / "report"

    with pytest.raises(FileNotFoundError, match="Full export requires extension artifacts"):
        review.main(
            [
                "--run-dir",
                str(run_dir),
                "--output-dir",
                str(output_dir),
                "--full-export",
            ]
        )


def test_full_export_succeeds_when_extension_artifacts_are_present(tmp_path: Path) -> None:
    run_dir = _build_core_run(tmp_path)
    extension_run_dir = tmp_path / "lm2011_extension"
    output_dir = tmp_path / "report"

    _write_parquet(extension_run_dir / review.EXTENSION_RESULTS_FILE, _extension_results_df())
    _write_parquet(extension_run_dir / review.EXTENSION_SAMPLE_LOSS_FILE, _extension_sample_loss_df())

    exit_code = review.main(
        [
            "--run-dir",
            str(run_dir),
            "--extension-run-dir",
            str(extension_run_dir),
            "--output-dir",
            str(output_dir),
            "--full-export",
        ]
    )

    assert exit_code == 0
    tex = (output_dir / "lm2011_post_refinitiv_review.tex").read_text(encoding="utf-8")
    assert "LM2011 Extension Sample Loss" in tex
    assert "LM2011 Extension Results" in tex

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path
import random

import polars as pl
import pytest

from thesis_pkg.pipeline import build_lm2011_extension_analysis_panel
from thesis_pkg.pipeline import build_lm2011_extension_control_ladder
from thesis_pkg.pipeline import build_lm2011_extension_dictionary_features
from thesis_pkg.pipeline import build_lm2011_extension_dictionary_features_from_cleaned_scopes
from thesis_pkg.pipeline import build_lm2011_extension_sample_loss_table
from thesis_pkg.pipeline import build_lm2011_extension_specification_grid
from thesis_pkg.pipeline import run_lm2011_extension_estimation_scaffold
from thesis_pkg.pipeline import run_lm2011_extension_fit_comparison_scaffold


def _write_ff48_mapping(tmp_path: Path) -> str:
    ff48_path = tmp_path / "ff48.txt"
    ff48_path.write_text(
        "\n".join(
            [
                " 1 Agric  Agriculture",
                "          0100-0199 Agricultural production - crops",
                "12 MedEq  Medical Equipment",
                "          3840-3849 Surgical, medical, and dental instruments and supplies",
            ]
        ),
        encoding="utf-8",
    )
    return str(ff48_path)


def _dictionary_lists() -> dict[str, list[str]]:
    return {
        "negative": ["loss"],
        "positive": ["gain"],
        "uncertainty": ["uncertain"],
        "litigious": ["lawsuit"],
        "modal_strong": ["must"],
        "modal_weak": ["may"],
    }


def _item_rows() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for doc_id, filing_date in (
        ("doc_missing_owner", dt.date(2009, 3, 2)),
        ("doc_owner", dt.date(2009, 3, 3)),
    ):
        for item_id, text in (
            ("1", "gain business may recognized"),
            ("1A", "loss uncertain lawsuit recognized"),
            ("7", "loss loss gain recognized must"),
        ):
            rows.append(
                {
                    "doc_id": doc_id,
                    "cik_10": "0000000001",
                    "filing_date": filing_date,
                    "document_type_filename": "10-K",
                    "item_id": item_id,
                    "full_text": text,
                }
            )
    return pl.DataFrame(rows)


def _event_panel() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "doc_id": ["doc_missing_owner", "doc_owner"],
            "gvkey_int": [1001, 1002],
            "KYPERMNO": [5001, 5002],
            "filing_date": [dt.date(2009, 3, 2), dt.date(2009, 3, 3)],
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


def _company_frames() -> tuple[pl.DataFrame, pl.DataFrame]:
    history = pl.DataFrame(
        {
            "KYGVKEY": [1001, 1002],
            "HCHGDT": [dt.date(2000, 1, 1), dt.date(2000, 1, 1)],
            "HCHGENDDT": [None, None],
            "HSIC": [111, 3845],
        }
    )
    description = pl.DataFrame({"KYGVKEY": [1001, 1002], "SIC": ["0111", "3845"]})
    return history, description


def test_extension_dictionary_features_score_item_scopes() -> None:
    features = build_lm2011_extension_dictionary_features(
        _item_rows().lazy(),
        dictionary_lists=_dictionary_lists(),
        harvard_negative_word_list=["bad"],
        master_dictionary_words=["loss", "gain", "uncertain", "lawsuit", "must", "may", "recognized", "bad"],
        text_scope_item_ids={"item_7_mda": "7", "item_1a_risk_factors": "1A"},
    ).collect().sort("doc_id", "text_scope")

    assert features.height == 4
    assert set(features.get_column("text_scope").to_list()) == {"item_7_mda", "item_1a_risk_factors"}
    assert features.get_column("dictionary_family").unique().to_list() == ["lm2011_frozen"]
    assert features["cleaning_policy_id"].unique().to_list() == ["raw_item_text"]
    item_7 = features.filter(
        (pl.col("doc_id") == "doc_owner") & (pl.col("text_scope") == "item_7_mda")
    ).row(0, named=True)
    assert item_7["total_token_count"] == 5
    assert item_7["token_count"] == 5
    assert item_7["lm_negative_prop"] == 0.4


def test_extension_dictionary_features_can_score_cleaned_scope_artifact() -> None:
    cleaned_scopes = pl.DataFrame(
        {
            "doc_id": ["doc_owner", "doc_owner"],
            "cik_10": ["0000000001", "0000000001"],
            "filing_date": [dt.date(2009, 3, 3), dt.date(2009, 3, 3)],
            "document_type_raw": ["10-K", "10-K"],
            "item_id": ["7", "1A"],
            "text_scope": ["item_7_mda", "item_1a_risk_factors"],
            "cleaning_policy_id": ["item_text_clean_v2", "item_text_clean_v2"],
            "cleaned_text": ["loss loss gain recognized must", "loss uncertain lawsuit recognized"],
        }
    )

    features = build_lm2011_extension_dictionary_features_from_cleaned_scopes(
        cleaned_scopes.lazy(),
        dictionary_lists=_dictionary_lists(),
        harvard_negative_word_list=["bad"],
        master_dictionary_words=["loss", "gain", "uncertain", "lawsuit", "must", "may", "recognized", "bad"],
    ).collect().sort("text_scope")

    assert features["text_scope"].to_list() == ["item_1a_risk_factors", "item_7_mda"]
    assert features["cleaning_policy_id"].to_list() == ["item_text_clean_v2", "item_text_clean_v2"]
    item_7 = features.filter(pl.col("text_scope") == "item_7_mda").row(0, named=True)
    assert item_7["lm_negative_prop"] == 0.4


def test_extension_panel_rejects_mismatched_cleaned_dictionary_and_model_universes(tmp_path: Path) -> None:
    dictionary_features = pl.DataFrame(
        {
            "doc_id": ["doc_owner"],
            "cik_10": ["0000000001"],
            "filing_date": [dt.date(2009, 3, 3)],
            "text_scope": ["item_7_mda"],
            "cleaning_policy_id": ["item_text_clean_v2"],
            "dictionary_family": ["lm2011_frozen"],
            "total_token_count": [5],
            "token_count": [5],
            "lm_negative_tfidf": [0.2],
        }
    )
    model_features = pl.DataFrame(
        {
            "doc_id": ["doc_owner"],
            "filing_date": [dt.date(2009, 3, 3)],
            "text_scope": ["item_7_mda"],
            "cleaning_policy_id": ["other_policy"],
            "finbert_neg_prob_lenw_mean": [0.3],
        }
    )
    company_history, company_description = _company_frames()

    try:
        build_lm2011_extension_analysis_panel(
            _event_panel().lazy(),
            dictionary_features.lazy(),
            model_features.lazy(),
            company_history.lazy(),
            company_description.lazy(),
            ff48_siccodes_path=_write_ff48_mapping(tmp_path),
        )
    except ValueError as exc:
        assert "identical cleaned" in str(exc)
    else:
        raise AssertionError("Expected mismatched cleaned-scope universes to be rejected")


def test_extension_panel_and_sample_loss_make_ownership_ladder_auditable(tmp_path: Path) -> None:
    dictionary_features = build_lm2011_extension_dictionary_features(
        _item_rows().lazy(),
        dictionary_lists=_dictionary_lists(),
        harvard_negative_word_list=["bad"],
        master_dictionary_words=["loss", "gain", "uncertain", "lawsuit", "must", "may", "recognized", "bad"],
        text_scope_item_ids={"item_7_mda": "7"},
    )
    model_features = pl.DataFrame(
        {
            "doc_id": ["doc_missing_owner", "doc_owner"],
            "benchmark_item_code": ["item_7", "item_7"],
            "model_name": ["yiyanghkust/finbert-tone", "yiyanghkust/finbert-tone"],
            "model_version": ["rev-a", "rev-a"],
            "segment_policy_id": ["sentence_dataset_v1_finbert_token_512"] * 2,
            "finbert_segment_count": [3, 4],
            "finbert_token_count_512_sum": [30, 40],
            "finbert_neg_prob_lenw_mean": [0.6, 0.2],
            "finbert_pos_prob_lenw_mean": [0.1, 0.5],
            "finbert_neu_prob_lenw_mean": [0.3, 0.3],
            "finbert_net_negative_lenw_mean": [0.5, -0.3],
            "finbert_neg_dominant_share": [0.67, 0.25],
        }
    )
    company_history, company_description = _company_frames()

    panel = build_lm2011_extension_analysis_panel(
        _event_panel().lazy(),
        dictionary_features,
        model_features.lazy(),
        company_history.lazy(),
        company_description.lazy(),
        ff48_siccodes_path=_write_ff48_mapping(tmp_path),
    ).collect().sort("doc_id", "text_scope")

    assert panel.select(["doc_id", "text_scope"]).is_unique().all()
    assert panel.get_column("sample_window").unique().to_list() == ["2009_2024"]
    assert panel.get_column("text_scope").unique().to_list() == ["item_7_mda"]
    assert panel.filter(pl.col("doc_id") == "doc_missing_owner").item(0, "ownership_proxy_available") is False
    assert panel.filter(pl.col("doc_id") == "doc_owner").item(0, "ownership_proxy_available") is True
    assert panel.filter(pl.col("doc_id") == "doc_owner").item(0, "finbert_token_count_512") == 40

    ladder = build_lm2011_extension_control_ladder()
    assert ladder.get_column("control_set_id").to_list() == ["C0", "C1", "C2"]
    assert ladder.filter(pl.col("control_set_id") == "C2").item(0, "includes_ownership_control") is True

    loss = build_lm2011_extension_sample_loss_table(panel.lazy()).filter(
        (pl.col("text_scope") == "item_7_mda")
        & (pl.col("specification_name") == "dictionary_only")
        & (pl.col("outcome_name") == "filing_period_excess_return")
    )
    counts = {
        row["control_set_id"]: row["n_control_set_rows"]
        for row in loss.select("control_set_id", "n_control_set_rows").to_dicts()
    }
    assert counts == {"C0": 2, "C1": 1, "C2": 1}


def _estimation_panel() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    rng = random.Random(20260411)
    filing_dates = [dt.date(2009, 2, 16), dt.date(2009, 5, 15)]
    for quarter_idx, filing_date in enumerate(filing_dates):
        for industry_idx, industry_id in enumerate((1, 12)):
            for obs_idx in range(8):
                doc_index = len(rows) + 1
                log_size = 4.0 + rng.random() + 0.05 * quarter_idx
                log_bm = -0.2 + 0.5 * rng.random() + 0.03 * industry_idx
                log_turnover = -3.0 + 0.7 * rng.random() + 0.01 * obs_idx
                pre_ffalpha = -0.02 + 0.04 * rng.random() + 0.002 * quarter_idx
                ownership = 20.0 + 30.0 * rng.random() + 2.0 * industry_idx
                lm_signal = 0.02 + 0.08 * rng.random() + 0.002 * industry_idx
                finbert_signal = 0.25 + 0.35 * rng.random() + 0.01 * quarter_idx
                nasdaq = int((obs_idx + industry_idx) % 2)
                y = (
                    0.01
                    + 0.7 * lm_signal
                    - 0.04 * finbert_signal
                    + 0.003 * log_size
                    - 0.002 * log_bm
                    + 0.001 * log_turnover
                    + 0.02 * pre_ffalpha
                    + 0.0003 * ownership
                    + 0.002 * nasdaq
                    + 0.001 * quarter_idx
                    + 0.002 * industry_idx
                )
                rows.append(
                    {
                        "doc_id": f"doc_{doc_index}",
                        "sample_window": "2009_2024",
                        "text_scope": "item_7_mda",
                        "filing_date": filing_date,
                        "ff48_industry_id": industry_id,
                        "log_size": log_size,
                        "log_book_to_market": log_bm,
                        "log_share_turnover": log_turnover,
                        "pre_ffalpha": pre_ffalpha,
                        "nasdaq_dummy": nasdaq,
                        "institutional_ownership_proxy_refinitiv": ownership,
                        "common_support_flag_ownership": True,
                        "filing_period_excess_return": y,
                        "abnormal_volume": 0.1 + 0.001 * obs_idx,
                        "postevent_return_volatility": 0.04 + 0.001 * obs_idx,
                        "lm_negative_tfidf": lm_signal,
                        "finbert_neg_prob_lenw_mean": finbert_signal,
                    }
                )
    return pl.DataFrame(rows)


def test_extension_estimation_scaffold_runs_primary_comparison_grid() -> None:
    grid = build_lm2011_extension_specification_grid()
    assert grid.get_column("specification_name").to_list() == [
        "dictionary_only",
        "finbert_only",
        "dictionary_finbert_joint",
    ]

    results = run_lm2011_extension_estimation_scaffold(
        _estimation_panel().lazy(),
        run_id="unit_test_extension",
        text_scopes=("item_7_mda",),
    )

    assert results.height > 0
    assert results.get_column("outcome_name").unique().to_list() == ["filing_period_excess_return"]
    assert set(results.get_column("specification_name").unique().to_list()) == {
        "dictionary_only",
        "finbert_only",
        "dictionary_finbert_joint",
    }
    assert set(results.get_column("control_set_id").unique().to_list()) == {"C0", "C1", "C2"}
    assert "estimated" in set(results.get_column("estimator_status").unique().to_list())
    horse_race = results.filter(
        (pl.col("specification_name") == "dictionary_finbert_joint")
        & (pl.col("coefficient_name") == "finbert_neg_prob_lenw_mean")
    )
    assert horse_race.height > 0
    assert all(value is None or math.isfinite(value) for value in results.get_column("p_value").to_list())


def test_extension_fit_comparison_scaffold_uses_common_sample_only_for_fit_artifacts() -> None:
    panel = (
        _estimation_panel()
        .with_row_index("_row")
        .with_columns(
            pl.when((pl.col("_row") % 5) == 0)
            .then(None)
            .otherwise(pl.col("finbert_neg_prob_lenw_mean"))
            .alias("finbert_neg_prob_lenw_mean")
        )
        .drop("_row")
    )

    coefficient_results = run_lm2011_extension_estimation_scaffold(
        panel.lazy(),
        run_id="unit_test_extension",
        text_scopes=("item_7_mda",),
        control_set_ids=("C0",),
    )
    fit_artifacts = run_lm2011_extension_fit_comparison_scaffold(
        panel.lazy(),
        run_id="unit_test_extension",
        text_scopes=("item_7_mda",),
        control_set_ids=("C0",),
    )

    dictionary_coef_n_obs = coefficient_results.filter(
        (pl.col("specification_name") == "dictionary_only")
        & (pl.col("control_set_id") == "C0")
        & (pl.col("coefficient_name") == "lm_negative_tfidf")
    ).item(0, "n_obs")
    dictionary_fit_summary = fit_artifacts.summary_df.filter(
        (pl.col("specification_name") == "dictionary_only")
        & (pl.col("control_set_id") == "C0")
    ).row(0, named=True)
    joint_fit_summary = fit_artifacts.summary_df.filter(
        (pl.col("specification_name") == "dictionary_finbert_joint")
        & (pl.col("control_set_id") == "C0")
    ).row(0, named=True)
    joint_minus_dictionary = fit_artifacts.comparison_df.filter(
        (pl.col("comparison_name") == "joint_minus_dictionary")
        & (pl.col("control_set_id") == "C0")
    ).row(0, named=True)
    difference_rows = fit_artifacts.quarterly_difference_df.filter(
        (pl.col("comparison_name") == "joint_minus_dictionary")
        & (pl.col("control_set_id") == "C0")
    ).sort("quarter_start")
    weights = difference_rows.get_column("weight").to_list()
    deltas = difference_rows.get_column("delta_adj_r2").to_list()
    weighted_delta = sum(weight * delta for weight, delta in zip(weights, deltas, strict=True)) / sum(weights)

    assert dictionary_coef_n_obs > dictionary_fit_summary["total_n_obs"]
    assert joint_fit_summary["signal_name"] == "lm_negative_tfidf,finbert_neg_prob_lenw_mean"
    assert joint_fit_summary["signal_inputs"] == ["lm_negative_tfidf", "finbert_neg_prob_lenw_mean"]
    assert joint_minus_dictionary["weighted_avg_delta_adj_r2"] == pytest.approx(weighted_delta, abs=1e-12)
    assert all(
        value >= -1e-10
        for value in difference_rows.get_column("delta_raw_r2").to_list()
    )
    assert dictionary_fit_summary["equal_quarter_avg_raw_r2"] is not None
    assert joint_minus_dictionary["equal_quarter_avg_delta_adj_r2"] is not None


def test_extension_fit_comparison_scaffold_records_insufficient_dof_skips() -> None:
    panel = (
        _estimation_panel()
        .filter(pl.col("filing_date") == dt.date(2009, 2, 16))
        .with_row_index("_row")
        .with_columns(
            pl.when((pl.col("_row") % 2) == 0)
            .then(None)
            .otherwise(pl.col("finbert_neg_prob_lenw_mean"))
            .alias("finbert_neg_prob_lenw_mean")
        )
        .drop("_row")
    )

    fit_artifacts = run_lm2011_extension_fit_comparison_scaffold(
        panel.lazy(),
        run_id="unit_test_extension",
        text_scopes=("item_7_mda",),
        control_set_ids=("C0",),
    )

    assert fit_artifacts.quarterly_fit_df.height == 0
    assert fit_artifacts.skipped_quarters_df.height > 0
    assert "insufficient_degrees_of_freedom" in fit_artifacts.skipped_quarters_df.get_column("skip_reason").to_list()
    assert fit_artifacts.summary_df.get_column("estimator_status").unique().to_list() == ["insufficient_sample"]

from __future__ import annotations

import datetime as dt
import math

import polars as pl
from polars.testing import assert_frame_equal

from thesis_pkg.benchmarking.finbert_tail_features import TAIL_FEATURE_COLUMNS
from thesis_pkg.benchmarking.finbert_tail_features import build_finbert_tail_doc_surface


def test_build_finbert_tail_doc_surface_computes_threshold_and_top_tail_metrics() -> None:
    sentence_scores = pl.DataFrame(
        {
            "doc_id": ["doc_1"] * 6,
            "filing_date": [dt.date(2010, 3, 1)] * 6,
            "benchmark_item_code": ["item-7"] * 6,
            "cleaning_policy_id": ["item_text_clean_v2"] * 6,
            "model_name": ["yiyanghkust/finbert-tone"] * 6,
            "model_version": ["rev-a"] * 6,
            "segment_policy_id": ["sentence_dataset_v1_finbert_token_512"] * 6,
            "sentence_index": [0, 1, 2, 3, 4, 5],
            "benchmark_sentence_id": [f"sent_{idx}" for idx in range(6)],
            "negative_prob": [0.95, 0.85, 0.75, 0.55, 0.35, 0.15],
            "finbert_token_count_512": [1, 2, 1, 1, 1, 1],
        }
    )

    result = build_finbert_tail_doc_surface(
        sentence_scores,
        text_scopes=("item_7_mda",),
    ).sort("doc_id", "text_scope")

    assert result.height == 1
    row = result.row(0, named=True)
    assert row["doc_id"] == "doc_1"
    assert row["text_scope"] == "item_7_mda"
    assert row["cleaning_policy_id"] == "item_text_clean_v2"
    assert row["model_name"] == "yiyanghkust/finbert-tone"
    assert row["model_version"] == "rev-a"
    assert row["segment_policy_id"] == "sentence_dataset_v1_finbert_token_512"

    weights = [1.0, 2.0, 1.0, 1.0, 1.0, 1.0]
    probs = [0.95, 0.85, 0.75, 0.55, 0.35, 0.15]
    total_weight = sum(weights)
    weighted_mean = sum(weight * prob for weight, prob in zip(weights, probs, strict=True)) / total_weight
    weighted_variance = (
        sum(
            weight * ((prob - weighted_mean) ** 2)
            for weight, prob in zip(weights, probs, strict=True)
        )
        / total_weight
    )

    assert row["tail_exposure_tau_0_60"] == pl.Series([3.4 / 7.0]).item()
    assert row["tail_exposure_tau_0_70"] == pl.Series([3.4 / 7.0]).item()
    assert row["tail_exposure_tau_0_80"] == pl.Series([2.65 / 7.0]).item()
    assert row["tail_share_tau_0_70"] == pl.Series([4.0 / 7.0]).item()
    assert row["top_10pct_neg_mean"] == 0.95
    assert row["top_20pct_neg_mean"] == pl.Series([(0.95 + (2.0 * 0.85)) / 3.0]).item()
    assert row["top_5_sentences_neg_mean"] == pl.Series([4.3 / 6.0]).item()
    assert math.isclose(
        float(row["neg_prob_dispersion"]),
        math.sqrt(weighted_variance),
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def test_build_finbert_tail_doc_surface_handles_zero_weight_groups_and_item_id_scope_fallback() -> None:
    sentence_scores = pl.DataFrame(
        {
            "doc_id": ["doc_2", "doc_2"],
            "filing_date": [dt.date(2011, 3, 1), dt.date(2011, 3, 1)],
            "item_id": ["1A", "1A"],
            "sentence_index": [0, 1],
            "negative_prob": [0.8, 0.4],
            "finbert_token_count_512": [0, 0],
        }
    )

    result = build_finbert_tail_doc_surface(
        sentence_scores,
        text_scopes=("item_1a_risk_factors",),
    )

    assert result.height == 1
    row = result.row(0, named=True)
    assert row["text_scope"] == "item_1a_risk_factors"
    for column in TAIL_FEATURE_COLUMNS:
        assert row[column] is None


def test_build_finbert_tail_doc_surface_is_deterministic_on_tied_scores_and_row_order() -> None:
    ordered_rows = [
        {
            "doc_id": "doc_tie",
            "filing_date": dt.date(2012, 3, 1),
            "benchmark_item_code": "item-7",
            "cleaning_policy_id": "item_text_clean_v2",
            "model_name": "yiyanghkust/finbert-tone",
            "model_version": "rev-a",
            "segment_policy_id": "sentence_dataset_v1_finbert_token_512",
            "sentence_index": sentence_index,
            "benchmark_sentence_id": benchmark_sentence_id,
            "negative_prob": negative_prob,
            "finbert_token_count_512": token_weight,
        }
        for sentence_index, benchmark_sentence_id, negative_prob, token_weight in (
            (10, "sent_top", 0.95, 1),
            (2, "sent_low_index", 0.90, 1),
            (5, "sent_high_index", 0.90, 10),
            (20, "sent_mid_1", 0.40, 1),
            (21, "sent_mid_2", 0.30, 1),
            (22, "sent_mid_3", 0.20, 1),
        )
    ]
    shuffled_rows = [
        ordered_rows[index]
        for index in (2, 5, 0, 4, 1, 3)
    ]

    ordered_result = build_finbert_tail_doc_surface(
        pl.DataFrame(ordered_rows),
        text_scopes=("item_7_mda",),
    ).sort("doc_id", "text_scope")
    shuffled_result = build_finbert_tail_doc_surface(
        pl.DataFrame(shuffled_rows),
        text_scopes=("item_7_mda",),
    ).sort("doc_id", "text_scope")

    assert_frame_equal(ordered_result, shuffled_result)
    assert math.isclose(
        float(ordered_result.get_column("top_20pct_neg_mean").item()),
        0.925,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def test_build_finbert_tail_doc_surface_uses_sentence_index_then_benchmark_sentence_id_as_tiebreakers() -> None:
    sentence_scores = pl.DataFrame(
        {
            "doc_id": ["doc_sentence_index"] * 6 + ["doc_benchmark_id"] * 6,
            "filing_date": [dt.date(2013, 3, 1)] * 12,
            "benchmark_item_code": ["item-7"] * 12,
            "cleaning_policy_id": ["item_text_clean_v2"] * 12,
            "model_name": ["yiyanghkust/finbert-tone"] * 12,
            "model_version": ["rev-a"] * 12,
            "segment_policy_id": ["sentence_dataset_v1_finbert_token_512"] * 12,
            "sentence_index": [
                10,
                2,
                5,
                20,
                21,
                22,
                10,
                2,
                2,
                20,
                21,
                22,
            ],
            "benchmark_sentence_id": [
                "sent_top",
                "sent_low_index",
                "sent_high_index",
                "sent_mid_1",
                "sent_mid_2",
                "sent_mid_3",
                "sent_top",
                "bench_a",
                "bench_b",
                "sent_mid_1",
                "sent_mid_2",
                "sent_mid_3",
            ],
            "negative_prob": [
                0.95,
                0.90,
                0.90,
                0.40,
                0.30,
                0.20,
                0.95,
                0.90,
                0.90,
                0.40,
                0.30,
                0.20,
            ],
            "finbert_token_count_512": [
                1,
                1,
                10,
                1,
                1,
                1,
                1,
                1,
                10,
                1,
                1,
                1,
            ],
        }
    )

    result = build_finbert_tail_doc_surface(
        sentence_scores,
        text_scopes=("item_7_mda",),
    ).sort("doc_id")

    assert result.height == 2
    assert result.get_column("doc_id").to_list() == ["doc_benchmark_id", "doc_sentence_index"]
    for top_20pct_mean in result.get_column("top_20pct_neg_mean").to_list():
        assert math.isclose(float(top_20pct_mean), 0.925, rel_tol=0.0, abs_tol=1e-12)

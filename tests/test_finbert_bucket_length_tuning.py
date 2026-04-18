from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from thesis_pkg.benchmarking.finbert_bucket_length_tuning import (
    recommend_conservative_bucket_edges,
)
from thesis_pkg.benchmarking.finbert_bucket_length_tuning import (
    write_bucket_edge_recommendation_report,
)


def _write_sentence_dataset(sentence_dataset_dir: Path) -> None:
    by_year_dir = sentence_dataset_dir / "by_year"
    by_year_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "benchmark_sentence_id": [
                "s1",
                "s2",
                "s3",
                "s4",
                "s5",
                "s6",
            ],
            "doc_id": [
                "doc1",
                "doc1",
                "doc2",
                "doc2",
                "doc3",
                "doc4",
            ],
            "filing_year": [2006, 2006, 2006, 2006, 2006, 2006],
            "benchmark_item_code": [
                "item_1",
                "item_7",
                "item_1a",
                "item_7",
                "item_1",
                "item_7",
            ],
            "sentence_text": [
                "short a",
                "short b",
                "medium a",
                "medium b",
                "long a",
                "long b",
            ],
            "sentence_char_count": [20, 48, 80, 96, 160, 192],
            "finbert_token_count_512": [40, 104, 140, 208, 300, 420],
            "finbert_token_bucket_512": [
                "short",
                "short",
                "medium",
                "medium",
                "long",
                "long",
            ],
        }
    ).write_parquet(by_year_dir / "2006.parquet")


def test_recommend_conservative_bucket_edges_defaults_keep_medium_edge_current(
    tmp_path: Path,
) -> None:
    sentence_dataset_dir = tmp_path / "sentence_dataset"
    _write_sentence_dataset(sentence_dataset_dir)

    recommendation = recommend_conservative_bucket_edges(
        sentence_dataset_dir,
        target_quantile=1.0,
        round_to=8,
        safety_margin_tokens=0,
    )

    assert recommendation.recommended_edges.short_edge == 104
    assert recommendation.recommended_edges.medium_edge == 256
    assert recommendation.effective_bucket_lengths.short_max_length == 104
    assert recommendation.effective_bucket_lengths.medium_max_length == 256
    assert recommendation.effective_bucket_lengths.long_max_length == 512
    assert recommendation.env_overrides == {
        "SEC_CCM_FINBERT_SHORT_EDGE": 104,
        "SEC_CCM_FINBERT_MEDIUM_EDGE": 256,
    }

    summary = recommendation.summary_by_bucket.sort("finbert_token_bucket_512")
    assert summary["sentence_rows"].to_list() == [2, 2, 2]
    assert summary["current_edge_upper_bound"].to_list() == [512, 256, 128]
    assert summary["current_max_length"].to_list() == [512, 256, 128]
    assert recommendation.metadata["estimated_padded_tokens_current"] == 1792
    assert recommendation.metadata["estimated_padded_tokens_rebucketed"] == 1744
    assert recommendation.metadata["estimated_padded_tokens_delta"] == -48
    assert recommendation.metadata["adds_extra_truncation_beyond_512"] is False


def test_recommend_conservative_bucket_edges_can_reduce_medium_edge(
    tmp_path: Path,
) -> None:
    sentence_dataset_dir = tmp_path / "sentence_dataset"
    _write_sentence_dataset(sentence_dataset_dir)

    recommendation = recommend_conservative_bucket_edges(
        sentence_dataset_dir,
        target_quantile=1.0,
        round_to=8,
        safety_margin_tokens=0,
        medium_edge_policy="target_quantile",
    )

    assert recommendation.recommended_edges.short_edge == 104
    assert recommendation.recommended_edges.medium_edge == 208
    assert recommendation.effective_bucket_lengths.medium_max_length == 208
    assert recommendation.metadata["estimated_padded_tokens_rebucketed"] == 1648
    recommendation_rows = {
        row["finbert_token_bucket_512"]: row
        for row in recommendation.recommendation_summary.to_dicts()
    }
    assert recommendation_rows["medium"]["policy_applied"] == "target_quantile"
    assert recommendation_rows["medium"]["recommended_edge_upper_bound"] == 208
    assert recommendation_rows["medium"]["rebucketed_sentence_rows"] == 2


def test_write_bucket_edge_recommendation_report_persists_env_and_metadata(
    tmp_path: Path,
) -> None:
    sentence_dataset_dir = tmp_path / "sentence_dataset"
    _write_sentence_dataset(sentence_dataset_dir)

    recommendation = recommend_conservative_bucket_edges(
        sentence_dataset_dir,
        target_quantile=1.0,
        round_to=8,
        safety_margin_tokens=0,
    )
    artifacts = write_bucket_edge_recommendation_report(recommendation, tmp_path / "report")

    assert artifacts.summary_by_bucket_parquet_path.exists()
    assert artifacts.recommendation_summary_csv_path.exists()
    assert artifacts.metadata_path.exists()
    assert artifacts.env_overrides_path.exists()

    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    assert metadata["recommended_edges"] == {
        "short_edge": 104,
        "medium_edge": 256,
    }
    assert metadata["effective_bucket_lengths"] == {
        "short_max_length": 104,
        "medium_max_length": 256,
        "long_max_length": 512,
    }
    env_text = artifacts.env_overrides_path.read_text(encoding="utf-8")
    assert "SEC_CCM_FINBERT_SHORT_EDGE=104" in env_text
    assert "SEC_CCM_FINBERT_MEDIUM_EDGE=256" in env_text

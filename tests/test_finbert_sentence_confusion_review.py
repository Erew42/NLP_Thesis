from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from thesis_pkg.benchmarking.finbert_sentence_confusion_review import (
    FinbertSentenceConfusionReviewConfig,
)
from thesis_pkg.benchmarking.finbert_sentence_confusion_review import (
    add_probability_majority_bucket,
)
from thesis_pkg.benchmarking.finbert_sentence_confusion_review import (
    build_finbert_sentence_confusion_review_pack,
)
from thesis_pkg.benchmarking.finbert_sentence_confusion_review import (
    summarize_finbert_sentence_confusion_review,
)


def _score_row(
    index: int,
    *,
    text_scope: str,
    predicted_label: str,
    negative_prob: float,
    neutral_prob: float,
    positive_prob: float,
) -> dict[str, object]:
    item_code = "item_1a" if text_scope == "item_1a_risk_factors" else "item_7"
    doc_id = f"doc_{index:04d}"
    return {
        "benchmark_sentence_id": f"{doc_id}:{item_code}:0",
        "benchmark_row_id": f"{doc_id}:{item_code}",
        "doc_id": doc_id,
        "cik_10": f"{index:010d}",
        "accession_nodash": f"{index:018d}",
        "filing_date": "2020-03-01",
        "filing_year": 2020,
        "benchmark_item_code": item_code,
        "benchmark_item_label": item_code,
        "source_year_file": 2020,
        "document_type": "10-K",
        "document_type_raw": "10-K",
        "document_type_normalized": "10-K",
        "canonical_item": item_code,
        "text_scope": text_scope,
        "cleaning_policy_id": "clean_v1",
        "segment_policy_id": "segment_v1",
        "sentence_index": 0,
        "sentence_text": f"Sentence {index} for {text_scope} with {predicted_label} prediction.",
        "sentence_char_count": 80,
        "sentencizer_backend": "test",
        "sentencizer_version": "1",
        "finbert_token_count_512": 12,
        "finbert_token_bucket_512": "short",
        "negative_prob": negative_prob,
        "neutral_prob": neutral_prob,
        "positive_prob": positive_prob,
        "predicted_label": predicted_label,
    }


def _write_sentence_scores_fixture(tmp_path: Path) -> Path:
    by_year = tmp_path / "sentence_scores" / "by_year"
    by_year.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    index = 1
    specs = [
        ("item_1a_risk_factors", "negative", 0.80, 0.10, 0.10, 8),
        ("item_1a_risk_factors", "neutral", 0.10, 0.80, 0.10, 7),
        ("item_1a_risk_factors", "positive", 0.10, 0.10, 0.80, 6),
        ("item_1a_risk_factors", "negative", 0.40, 0.35, 0.25, 4),
        ("item_7_mda", "negative", 0.75, 0.15, 0.10, 8),
        ("item_7_mda", "neutral", 0.15, 0.70, 0.15, 7),
        ("item_7_mda", "positive", 0.15, 0.15, 0.70, 6),
        ("item_7_mda", "positive", 0.34, 0.33, 0.33, 4),
        ("item_1_business", "negative", 0.90, 0.05, 0.05, 5),
    ]
    for text_scope, predicted_label, neg, neu, pos, count in specs:
        for _ in range(count):
            rows.append(
                _score_row(
                    index,
                    text_scope=text_scope,
                    predicted_label=predicted_label,
                    negative_prob=neg,
                    neutral_prob=neu,
                    positive_prob=pos,
                )
            )
            index += 1
    pl.DataFrame(rows).write_parquet(by_year / "2020.parquet")
    return by_year.parent


def test_probability_majority_bucket_assignment() -> None:
    df = pl.DataFrame(
        {
            "negative_prob": [0.51, 0.10, 0.10, 0.34],
            "neutral_prob": [0.30, 0.60, 0.10, 0.33],
            "positive_prob": [0.19, 0.30, 0.80, 0.33],
        }
    )
    out = add_probability_majority_bucket(df.lazy()).collect()
    assert out["probability_majority_bucket"].to_list() == [
        "negative_majority",
        "neutral_majority",
        "positive_majority",
        "no_majority",
    ]


def test_counts_only_writes_population_artifacts(tmp_path: Path) -> None:
    sentence_scores_dir = _write_sentence_scores_fixture(tmp_path)
    output_dir = tmp_path / "review_counts"

    artifacts = build_finbert_sentence_confusion_review_pack(
        sentence_scores_dir,
        output_dir=output_dir,
        cfg=FinbertSentenceConfusionReviewConfig(sample_size=12, seed=7),
        counts_only=True,
    )

    assert artifacts.counts_only is True
    assert artifacts.sample_path is None
    assert artifacts.population_counts_by_majority_bucket_path.exists()
    assert artifacts.population_counts_by_stratum_path.exists()
    summary = json.loads(artifacts.population_counts_summary_path.read_text(encoding="utf-8"))
    assert summary["total_population_rows"] == 50
    bucket_counts = pl.read_csv(artifacts.population_counts_by_majority_bucket_path)
    assert set(bucket_counts["probability_majority_bucket"]) == {
        "negative_majority",
        "neutral_majority",
        "positive_majority",
        "no_majority",
    }


def test_build_review_pack_samples_exact_rows_and_outputs_review_assets(tmp_path: Path) -> None:
    sentence_scores_dir = _write_sentence_scores_fixture(tmp_path)
    output_dir = tmp_path / "review_pack"

    artifacts = build_finbert_sentence_confusion_review_pack(
        sentence_scores_dir,
        output_dir=output_dir,
        cfg=FinbertSentenceConfusionReviewConfig(sample_size=12, seed=11, chunk_count=10),
    )

    assert artifacts.sample_path is not None
    sample = pl.read_parquet(artifacts.sample_path)
    assert sample.height == 12
    assert sample["sample_order"].to_list() == list(range(1, 13))
    assert set(sample["text_scope"]) <= {"item_1a_risk_factors", "item_7_mda"}
    assert "no_majority" not in set(sample["probability_majority_bucket"])
    assert sample["sample_weight"].min() > 0
    assert artifacts.review_html_path is not None
    html = artifacts.review_html_path.read_text(encoding="utf-8")
    assert "FinBERT Negative/Adverse Sentence Review" in html
    assert "probability_majority_bucket" in html
    assert (output_dir / "chunks" / "chunk_01.jsonl").exists()
    assert len(list((output_dir / "chunks").glob("chunk_*.jsonl"))) == 10
    assert (output_dir / "llm_pass_chunks" / "pass_a_low" / "chunk_01.jsonl").exists()
    assert (output_dir / "llm_pass_chunks" / "pass_b_medium" / "chunk_01.jsonl").exists()

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["sample_row_count"] == 12
    assert sum(row["sample_count"] for row in manifest["allocation"]) == 12
    assert manifest["sampling_attempts"][0]["sampling_method"] == "stream_target_ordinals"
    assert manifest["sampling_attempts"][0]["retained_sample_rows"] == 12


def test_summarize_review_pack_derives_confusion_metrics(tmp_path: Path) -> None:
    sentence_scores_dir = _write_sentence_scores_fixture(tmp_path)
    output_dir = tmp_path / "review_pack"
    build_finbert_sentence_confusion_review_pack(
        sentence_scores_dir,
        output_dir=output_dir,
        cfg=FinbertSentenceConfusionReviewConfig(sample_size=12, seed=13, chunk_count=4),
    )
    sample = pl.read_parquet(output_dir / "sample.parquet")
    review_rows = []
    for row in sample.to_dicts():
        if row["sample_order"] == 1:
            label = "uncertain"
        elif row["predicted_label"] == "negative":
            label = "yes" if row["sample_order"] % 2 == 0 else "no"
        else:
            label = "yes" if row["sample_order"] % 5 == 0 else "no"
        review_rows.append(
            {
                "review_case_id": row["review_case_id"],
                "human_gold_negative": label,
                "human_confidence": "high",
                "human_issue_flags": ["none"],
            }
        )
    human_review_path = output_dir / "human_review.json"
    human_review_path.write_text(json.dumps({"rows": review_rows}), encoding="utf-8")

    artifacts = summarize_finbert_sentence_confusion_review(
        output_dir,
        human_review_path=human_review_path,
    )

    assert artifacts.reviewed_cases_path.exists()
    assert artifacts.confusion_matrix_path.exists()
    assert artifacts.metrics_json_path.exists()
    metrics = json.loads(artifacts.metrics_json_path.read_text(encoding="utf-8"))
    assert metrics["sample_row_count"] == 12
    assert metrics["reviewed_row_count"] == 12
    assert metrics["unweighted_counts_by_cell"]["uncertain"] == 1.0
    confusion = pl.read_csv(artifacts.confusion_matrix_path)
    assert set(confusion["confusion_cell"]) == {"TP", "FP", "FN", "TN", "uncertain"}
    assert artifacts.majority_bucket_metrics_path.exists()
    assert "Thesis Caveat" in artifacts.metrics_markdown_path.read_text(encoding="utf-8")

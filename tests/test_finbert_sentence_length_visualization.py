from __future__ import annotations

from pathlib import Path

import polars as pl

from thesis_pkg.benchmarking.sentence_length_visualization import analyze_sentence_lengths


def test_analyze_sentence_lengths_summarizes_item_distributions(tmp_path: Path) -> None:
    sentence_dir = tmp_path / "sentence_dataset" / "by_year"
    sentence_dir.mkdir(parents=True, exist_ok=True)

    pl.DataFrame(
        {
            "benchmark_sentence_id": [
                "doc1:item_1a:0",
                "doc1:item_1a:1",
                "doc2:item_7:0",
                "doc2:item_7:1",
            ],
            "doc_id": ["doc1", "doc1", "doc2", "doc2"],
            "filing_year": [2006, 2006, 2007, 2007],
            "benchmark_item_code": ["item_1a", "item_1a", "item_7", "item_7"],
            "sentence_text": ["a", "b", "c", "d"],
            "sentence_char_count": [100, 260, 150, 520],
            "finbert_token_count_512": [20, 140, 60, 300],
            "finbert_token_bucket_512": ["short", "medium", "short", "long"],
        }
    ).write_parquet(sentence_dir / "2006.parquet")

    analysis = analyze_sentence_lengths(
        sentence_dir,
        item_codes=("item_1a", "item_7"),
        char_bin_width=50,
        top_n=2,
    )

    overall = analysis.summary_overall.to_dicts()[0]
    assert overall["sentence_rows"] == 4
    assert overall["doc_count"] == 2
    assert overall["token_max"] == 300
    assert overall["char_max"] == 520

    by_item = {row["benchmark_item_code"]: row for row in analysis.summary_by_item.to_dicts()}
    assert by_item["item_1a"]["sentence_rows"] == 2
    assert by_item["item_1a"]["share_token_le_128"] == 0.5
    assert by_item["item_7"]["sentence_rows"] == 2
    assert by_item["item_7"]["share_token_gt_256"] == 0.5

    token_hist = analysis.token_histogram.sort("finbert_token_count_512")
    assert token_hist["finbert_token_count_512"].to_list() == [20, 60, 140, 300]
    assert token_hist["sentence_rows"].to_list() == [1, 1, 1, 1]

    char_hist = analysis.char_histogram.sort("char_bin_start")
    assert char_hist["char_bin_start"].to_list() == [100, 150, 250, 500]
    assert char_hist["sentence_rows"].to_list() == [1, 1, 1, 1]

    longest = analysis.longest_sentences
    assert longest.height == 2
    assert longest["finbert_token_count_512"].to_list() == [300, 140]

    assert analysis.metadata["sentence_rows"] == 4
    assert analysis.metadata["item_codes_present_ordered"] == ["item_1a", "item_7"]

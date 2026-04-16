from __future__ import annotations

import polars as pl

from thesis_pkg.benchmarking.contracts import ItemTextCleaningConfig
from thesis_pkg.benchmarking.item7_lm_floor_sweep import analyze_item7_lm_floor_sweep


def _make_item7_row(doc_id: str, token_count: int) -> dict[str, object]:
    text = " ".join(["discussion"] * token_count)
    return {
        "doc_id": doc_id,
        "cik_10": "0000000001",
        "accession_nodash": doc_id.replace(":", "")[-18:],
        "filing_date": "2010-03-01",
        "filing_year": 2010,
        "benchmark_row_id": f"{doc_id}:item_7",
        "benchmark_item_code": "item_7",
        "benchmark_item_label": "10-K Item 7",
        "item_id": "7",
        "canonical_item": "II:7_MDA",
        "document_type": "10-K",
        "document_type_raw": "10-K",
        "document_type_normalized": "10-K",
        "source_year_file": 2010,
        "boundary_authority_status": "trusted",
        "full_text": text,
    }


def test_analyze_item7_lm_floor_sweep_tracks_confirmed_false_positive_share() -> None:
    sampled_sections_df = pl.DataFrame(
        [
            _make_item7_row("doc_fp", 180),
            _make_item7_row("doc_tp", 120),
            _make_item7_row("doc_keep", 260),
        ]
    )
    reviewed_removed_segment_df = pl.DataFrame(
        {
            "benchmark_row_id": ["doc_fp:item_7", "doc_tp:item_7"],
            "review_label": ["false_positive_removal", "true_positive_removal"],
        }
    )

    results_df, reviewed_case_status_df = analyze_item7_lm_floor_sweep(
        sampled_sections_df,
        thresholds=(150, 200),
        base_cleaning_cfg=ItemTextCleaningConfig(),
        reviewed_removed_segment_df=reviewed_removed_segment_df,
    )

    threshold_150 = results_df.filter(pl.col("item7_min_lm_tokens") == 150).row(0, named=True)
    assert threshold_150["item7_rows_dropped_total"] == 1
    assert threshold_150["confirmed_false_positive_removed_rows"] == 0

    threshold_200 = results_df.filter(pl.col("item7_min_lm_tokens") == 200).row(0, named=True)
    assert threshold_200["item7_rows_dropped_total"] == 2
    assert threshold_200["confirmed_false_positive_removed_rows"] == 1
    assert threshold_200["confirmed_fp_share_of_total_dropped"] == 0.5

    fp_status = reviewed_case_status_df.filter(
        (pl.col("item7_min_lm_tokens") == 200) & (pl.col("benchmark_row_id") == "doc_fp:item_7")
    ).row(0, named=True)
    assert fp_status["dropped_after_cleaning"] is True

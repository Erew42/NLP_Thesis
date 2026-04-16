from __future__ import annotations

from pathlib import Path

import polars as pl

from thesis_pkg.benchmarking.sentence_split_quality_assessment import analyze_sentence_split_quality
from thesis_pkg.benchmarking.sentence_split_quality_assessment import write_sentence_split_quality_report


def test_analyze_sentence_split_quality_summarizes_residual_patterns(tmp_path: Path) -> None:
    sentence_dir = tmp_path / "sentence_dataset" / "by_year"
    sentence_dir.mkdir(parents=True, exist_ok=True)

    pl.DataFrame(
        {
            "benchmark_sentence_id": [
                "doc1:item_1:0",
                "doc1:item_1:1",
                "doc2:item_1a:0",
                "doc3:item_7:0",
                "doc3:item_7:1",
            ],
            "benchmark_row_id": [
                "doc1:item_1",
                "doc1:item_1",
                "doc2:item_1a",
                "doc3:item_7",
                "doc3:item_7",
            ],
            "doc_id": ["doc1", "doc1", "doc2", "doc3", "doc3"],
            "filing_year": [2006, 2006, 2007, 2008, 2008],
            "benchmark_item_code": ["item_1", "item_1", "item_1a", "item_7", "item_7"],
            "sentence_index": [0, 1, 0, 0, 1],
            "finbert_token_count_512": [12, 8, 5, 4, 140],
            "sentence_char_count": [57, 34, 13, 2, 65],
            "sentence_text": [
                "Management believes that the disclosures required by Statement No.",
                "130 are unnecessary under the current presentation.",
                "RISK FACTORS.",
                "3.",
                "CONSOLIDATED STATEMENTS OF CASH FLOWS\n1999 1998 1997\n100 200 300",
            ],
        }
    ).write_parquet(sentence_dir / "2006.parquet")

    pl.DataFrame(
        {
            "benchmark_row_id": ["doc1:item_1", "doc3:item_7"],
            "split_reason": ["double_newline", "end_of_text"],
            "warning_boundary_used": [False, False],
        }
    ).write_parquet(tmp_path / "sentence_split_audit.parquet")

    cleaning_row_audit_path = tmp_path / "item_cleaning" / "cleaning_row_audit.parquet"
    cleaning_row_audit_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "benchmark_row_id": ["doc1:item_1", "doc2:item_1a", "doc3:item_7"],
            "dropped_after_cleaning": [False, False, False],
        }
    ).write_parquet(cleaning_row_audit_path)

    analysis = analyze_sentence_split_quality(
        tmp_path / "sentence_dataset",
        cleaning_row_audit_path=cleaning_row_audit_path,
    )

    by_scope = {row["benchmark_item_code"]: row for row in analysis.summary_by_scope.to_dicts()}
    assert by_scope["item_1"]["generic_no_end_rows"] == 1
    assert by_scope["item_1"]["generic_no_with_continuation_rows"] == 1
    assert by_scope["item_1"]["stitch_candidate_rows"] == 1
    assert by_scope["item_1a"]["header_like_rows"] == 1
    assert by_scope["item_7"]["numeric_only_rows"] == 1
    assert by_scope["item_7"]["table_like_rows"] == 1

    split_summary = {row["split_reason"]: row for row in analysis.split_audit_summary.to_dicts()}
    assert split_summary["double_newline"]["split_rows"] == 1
    assert split_summary["end_of_text"]["chunked_item_rows"] == 1

    artifacts = write_sentence_split_quality_report(analysis, tmp_path / "quality_report")
    assert artifacts["report_path"].exists()
    assert artifacts["summary_by_scope_path"].exists()
    assert (artifacts["examples_dir"] / "generic_no_end.csv").exists()

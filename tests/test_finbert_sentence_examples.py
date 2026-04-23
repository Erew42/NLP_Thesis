from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import polars as pl

from thesis_pkg.benchmarking.finbert_sentence_examples import (
    BENCHMARK_ITEM_CODE_COLUMN,
    SENTENCE_TEXT_COLUMN,
    SENTIMENT_COLUMN,
    build_high_confidence_sentence_example_pack,
)


def _write_sentence_score_year(path: Path, rows: list[dict[str, object]]) -> None:
    pl.DataFrame(rows).write_parquet(path)


def _sample_rows() -> list[dict[str, object]]:
    return [
        {
            "benchmark_sentence_id": "doc_a:item_1a:0",
            "benchmark_row_id": "doc_a:item_1a",
            "doc_id": "doc_a",
            "cik_10": "0000000001",
            "accession_nodash": "000000000120200001",
            "filing_date": "2020-02-01",
            "filing_year": 2020,
            "benchmark_item_code": "item_1a",
            "benchmark_item_label": "10-K Item 1A",
            "source_year_file": 2020,
            "document_type": "10-K",
            "document_type_raw": "10-K",
            "document_type_normalized": "10-K",
            "canonical_item": "1A",
            "text_scope": "item_1a_risk_factors",
            "cleaning_policy_id": "clean_v1",
            "segment_policy_id": "segment_v1",
            "sentence_index": 0,
            "sentence_text": "We expect strong demand and improved operating margins next year.",
            "sentence_char_count": 61,
            "sentencizer_backend": "spacy_blank_en_sentencizer",
            "sentencizer_version": "1",
            "finbert_token_count_512": 13,
            "finbert_token_bucket_512": "short",
            "negative_prob": 0.01,
            "neutral_prob": 0.02,
            "positive_prob": 0.97,
            "predicted_label": "positive",
        },
        {
            "benchmark_sentence_id": "doc_b:item_1a:0",
            "benchmark_row_id": "doc_b:item_1a",
            "doc_id": "doc_b",
            "cik_10": "0000000002",
            "accession_nodash": "000000000220200001",
            "filing_date": "2020-02-02",
            "filing_year": 2020,
            "benchmark_item_code": "item_1a",
            "benchmark_item_label": "10-K Item 1A",
            "source_year_file": 2020,
            "document_type": "10-K",
            "document_type_raw": "10-K",
            "document_type_normalized": "10-K",
            "canonical_item": "1A",
            "text_scope": "item_1a_risk_factors",
            "cleaning_policy_id": "clean_v1",
            "segment_policy_id": "segment_v1",
            "sentence_index": 0,
            "sentence_text": "Demand improved somewhat but visibility remains limited.",
            "sentence_char_count": 55,
            "sentencizer_backend": "spacy_blank_en_sentencizer",
            "sentencizer_version": "1",
            "finbert_token_count_512": 10,
            "finbert_token_bucket_512": "short",
            "negative_prob": 0.02,
            "neutral_prob": 0.04,
            "positive_prob": 0.94,
            "predicted_label": "positive",
        },
        {
            "benchmark_sentence_id": "doc_c:item_1a:0",
            "benchmark_row_id": "doc_c:item_1a",
            "doc_id": "doc_c",
            "cik_10": "0000000003",
            "accession_nodash": "000000000320200001",
            "filing_date": "2020-02-03",
            "filing_year": 2020,
            "benchmark_item_code": "item_1a",
            "benchmark_item_label": "10-K Item 1A",
            "source_year_file": 2020,
            "document_type": "10-K",
            "document_type_raw": "10-K",
            "document_type_normalized": "10-K",
            "canonical_item": "1A",
            "text_scope": "item_1a_risk_factors",
            "cleaning_policy_id": "clean_v1",
            "segment_policy_id": "segment_v1",
            "sentence_index": 0,
            "sentence_text": "Losses may rise quickly.",
            "sentence_char_count": 24,
            "sentencizer_backend": "spacy_blank_en_sentencizer",
            "sentencizer_version": "1",
            "finbert_token_count_512": 5,
            "finbert_token_bucket_512": "short",
            "negative_prob": 0.97,
            "neutral_prob": 0.02,
            "positive_prob": 0.01,
            "predicted_label": "negative",
        },
        {
            "benchmark_sentence_id": "doc_d:item_1:0",
            "benchmark_row_id": "doc_d:item_1",
            "doc_id": "doc_d",
            "cik_10": "0000000004",
            "accession_nodash": "000000000420200001",
            "filing_date": "2020-02-04",
            "filing_year": 2020,
            "benchmark_item_code": "item_1",
            "benchmark_item_label": "10-K Item 1",
            "source_year_file": 2020,
            "document_type": "10-K",
            "document_type_raw": "10-K",
            "document_type_normalized": "10-K",
            "canonical_item": "1",
            "text_scope": "item_1_business",
            "cleaning_policy_id": "clean_v1",
            "segment_policy_id": "segment_v1",
            "sentence_index": 0,
            "sentence_text": "Our business continued to grow across several regions this year.",
            "sentence_char_count": 63,
            "sentencizer_backend": "spacy_blank_en_sentencizer",
            "sentencizer_version": "1",
            "finbert_token_count_512": 12,
            "finbert_token_bucket_512": "short",
            "negative_prob": 0.01,
            "neutral_prob": 0.00,
            "positive_prob": 0.99,
            "predicted_label": "positive",
        },
        {
            "benchmark_sentence_id": "doc_e:item_7:0",
            "benchmark_row_id": "doc_e:item_7",
            "doc_id": "doc_e",
            "cik_10": "0000000005",
            "accession_nodash": "000000000520210001",
            "filing_date": "2021-03-01",
            "filing_year": 2021,
            "benchmark_item_code": "item_7",
            "benchmark_item_label": "10-K Item 7",
            "source_year_file": 2021,
            "document_type": "10-K",
            "document_type_raw": "10-K",
            "document_type_normalized": "10-K",
            "canonical_item": "7",
            "text_scope": "item_7_mda",
            "cleaning_policy_id": "clean_v1",
            "segment_policy_id": "segment_v1",
            "sentence_index": 0,
            "sentence_text": "We may incur significant losses if commodity prices decline further.",
            "sentence_char_count": 68,
            "sentencizer_backend": "spacy_blank_en_sentencizer",
            "sentencizer_version": "1",
            "finbert_token_count_512": 12,
            "finbert_token_bucket_512": "short",
            "negative_prob": 0.98,
            "neutral_prob": 0.01,
            "positive_prob": 0.01,
            "predicted_label": "negative",
        },
        {
            "benchmark_sentence_id": "doc_f:item_7:0",
            "benchmark_row_id": "doc_f:item_7",
            "doc_id": "doc_f",
            "cik_10": "0000000006",
            "accession_nodash": "000000000620210001",
            "filing_date": "2021-03-02",
            "filing_year": 2021,
            "benchmark_item_code": "item_7",
            "benchmark_item_label": "10-K Item 7",
            "source_year_file": 2021,
            "document_type": "10-K",
            "document_type_raw": "10-K",
            "document_type_normalized": "10-K",
            "canonical_item": "7",
            "text_scope": "item_7_mda",
            "cleaning_policy_id": "clean_v1",
            "segment_policy_id": "segment_v1",
            "sentence_index": 0,
            "sentence_text": "Cash flow improved materially and our balance sheet remained resilient.",
            "sentence_char_count": 69,
            "sentencizer_backend": "spacy_blank_en_sentencizer",
            "sentencizer_version": "1",
            "finbert_token_count_512": 12,
            "finbert_token_bucket_512": "short",
            "negative_prob": 0.00,
            "neutral_prob": 0.01,
            "positive_prob": 0.99,
            "predicted_label": "positive",
        },
        {
            "benchmark_sentence_id": "doc_g:item_7:0",
            "benchmark_row_id": "doc_g:item_7",
            "doc_id": "doc_g",
            "cik_10": "0000000007",
            "accession_nodash": "000000000720210001",
            "filing_date": "2021-03-03",
            "filing_year": 2021,
            "benchmark_item_code": "item_7",
            "benchmark_item_label": "10-K Item 7",
            "source_year_file": 2021,
            "document_type": "10-K",
            "document_type_raw": "10-K",
            "document_type_normalized": "10-K",
            "canonical_item": "7",
            "text_scope": "item_7_mda",
            "cleaning_policy_id": "clean_v1",
            "segment_policy_id": "segment_v1",
            "sentence_index": 0,
            "sentence_text": "Cash flow improved materially and our balance sheet remained resilient.",
            "sentence_char_count": 69,
            "sentencizer_backend": "spacy_blank_en_sentencizer",
            "sentencizer_version": "1",
            "finbert_token_count_512": 12,
            "finbert_token_bucket_512": "short",
            "negative_prob": 0.01,
            "neutral_prob": 0.02,
            "positive_prob": 0.97,
            "predicted_label": "positive",
        },
        {
            "benchmark_sentence_id": "doc_h:item_7:0",
            "benchmark_row_id": "doc_h:item_7",
            "doc_id": "doc_h",
            "cik_10": "0000000008",
            "accession_nodash": "000000000820210001",
            "filing_date": "2021-03-04",
            "filing_year": 2021,
            "benchmark_item_code": "item_7",
            "benchmark_item_label": "10-K Item 7",
            "source_year_file": 2021,
            "document_type": "10-K",
            "document_type_raw": "10-K",
            "document_type_normalized": "10-K",
            "canonical_item": "7",
            "text_scope": "item_7_mda",
            "cleaning_policy_id": "clean_v1",
            "segment_policy_id": "segment_v1",
            "sentence_index": 0,
            "sentence_text": "Operating cash inflows remained strong\nand liquidity stayed ample throughout the year.",
            "sentence_char_count": 82,
            "sentencizer_backend": "spacy_blank_en_sentencizer",
            "sentencizer_version": "1",
            "finbert_token_count_512": 14,
            "finbert_token_bucket_512": "short",
            "negative_prob": 0.01,
            "neutral_prob": 0.03,
            "positive_prob": 0.96,
            "predicted_label": "positive",
        },
    ]


def _build_fixture_sentence_scores_dir(tmp_path: Path) -> Path:
    sentence_scores_dir = tmp_path / "sentence_scores" / "by_year"
    sentence_scores_dir.mkdir(parents=True, exist_ok=True)
    rows = _sample_rows()
    _write_sentence_score_year(
        sentence_scores_dir / "2020.parquet",
        [row for row in rows if int(row["filing_year"]) == 2020],
    )
    _write_sentence_score_year(
        sentence_scores_dir / "2021.parquet",
        [row for row in rows if int(row["filing_year"]) == 2021],
    )
    return sentence_scores_dir


def test_build_high_confidence_sentence_example_pack_filters_and_samples(tmp_path: Path) -> None:
    sentence_scores_dir = _build_fixture_sentence_scores_dir(tmp_path)
    output_dir = tmp_path / "examples"

    pack = build_high_confidence_sentence_example_pack(
        sentence_scores_dir,
        output_dir=output_dir,
        sample_size_per_group=2,
    )

    assert pack.metadata["candidate_rows"] == 5
    assert pack.metadata["candidate_doc_count"] == 5
    assert pack.artifacts.candidate_shards_dir is None
    assert pack.artifacts.sample_markdown_path.exists()
    assert pack.artifacts.summary_json_path.exists()

    counts = {
        (row["benchmark_item_code"], row["sentiment"]): row["candidate_rows"]
        for row in pack.counts_by_item_sentiment.to_dicts()
    }
    assert counts == {
        ("item_1a", "positive"): 1,
        ("item_7", "negative"): 1,
        ("item_7", "positive"): 3,
    }

    sample_df = pack.sample_candidates
    assert sample_df.height == 4
    assert set(sample_df[BENCHMARK_ITEM_CODE_COLUMN].unique().to_list()) == {"item_1a", "item_7"}
    assert set(sample_df[SENTIMENT_COLUMN].unique().to_list()) == {"positive", "negative"}
    item7_positive = sample_df.filter(
        (pl.col(BENCHMARK_ITEM_CODE_COLUMN) == "item_7") & (pl.col(SENTIMENT_COLUMN) == "positive")
    )
    assert item7_positive.height == 2
    assert (
        item7_positive.filter(
            pl.col(SENTENCE_TEXT_COLUMN)
            == "Cash flow improved materially and our balance sheet remained resilient."
        ).height
        == 1
    )
    assert (
        item7_positive.filter(pl.col(SENTENCE_TEXT_COLUMN).str.contains(r"\n")).height == 0
    )

    markdown = pack.artifacts.sample_markdown_path.read_text(encoding="utf-8")
    assert "## Item 1A | Positive" in markdown
    assert "## Item 1A | Negative" in markdown
    assert "Eligible candidates: 0 | Listed samples: 0" in markdown
    assert "Operating cash inflows remained strong and liquidity stayed ample throughout the year." in markdown

    summary = json.loads(pack.artifacts.summary_json_path.read_text(encoding="utf-8"))
    assert summary["candidate_rows"] == 5
    assert summary["filters"]["batch_size"] == 50000
    assert summary["filters"]["min_probability"] == 0.95


def test_tool_extract_finbert_high_confidence_sentences_emits_json(
    tmp_path: Path,
    capsys,
) -> None:
    sentence_scores_dir = _build_fixture_sentence_scores_dir(tmp_path)
    module_path = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "extract_finbert_high_confidence_sentences.py"
    )
    spec = importlib.util.spec_from_file_location("test_extract_finbert_high_confidence_sentences", module_path)
    assert spec is not None
    assert spec.loader is not None
    tool_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool_module)

    output_dir = tmp_path / "tool_output"
    exit_code = tool_module.main(
        [
            "--sentence-scores-dir",
            str(sentence_scores_dir),
            "--output-dir",
            str(output_dir),
            "--sample-size-per-group",
            "2",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["candidate_rows"] == 5
    assert payload["candidate_shards_dir"] is None
    assert payload["summary_json_path"] == str(output_dir.resolve() / "summary.json")
    assert payload["sample_markdown_path"] == str(output_dir.resolve() / "sample_candidates.md")

from __future__ import annotations

from pathlib import Path

import polars as pl

from thesis_pkg.benchmarking.contracts import SentenceDatasetConfig
from thesis_pkg.benchmarking.sentences import materialize_sentence_benchmark_dataset


def _fake_annotate_token_lengths(df: pl.DataFrame, _cfg) -> pl.DataFrame:
    counts = [max(1, len(str(text).split())) for text in df["full_text"].to_list()]
    buckets = ["short" for _ in counts]
    return df.with_columns(
        [
            pl.Series("finbert_token_count_512", counts, dtype=pl.Int32),
            pl.Series("finbert_token_bucket_512", buckets, dtype=pl.Utf8),
        ]
    )


class _FakeSpan:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeDoc:
    def __init__(self, text: str) -> None:
        self.sents = [_FakeSpan(part.strip()) for part in text.split(".") if part.strip()]


class _FakeNLP:
    pipe_names = ("sentencizer",)

    def pipe(self, texts, batch_size: int = 0):
        del batch_size
        for text in texts:
            yield _FakeDoc(text)


def test_materialize_sentence_benchmark_dataset(tmp_path: Path, monkeypatch) -> None:
    sections_path = tmp_path / "sections.parquet"
    pl.DataFrame(
        {
            "benchmark_row_id": ["doc1:item_1", "doc2:item_7"],
            "doc_id": ["doc1", "doc2"],
            "filing_date": [None, None],
            "filing_year": [2000, 2001],
            "benchmark_item_code": ["item_1", "item_7"],
            "full_text": ["First sentence. Second sentence.", "Third sentence only."],
        }
    ).write_parquet(sections_path)

    from thesis_pkg.benchmarking import sentences

    monkeypatch.setattr(sentences, "_build_sentencizer", lambda cfg: _FakeNLP())
    monkeypatch.setattr(sentences, "_sentencizer_version", lambda: "test")
    monkeypatch.setattr(sentences, "annotate_token_lengths", _fake_annotate_token_lengths)

    out_path = tmp_path / "sentences.parquet"
    materialize_sentence_benchmark_dataset(
        sections_path,
        SentenceDatasetConfig(enabled=True),
        out_path=out_path,
    )

    result = pl.read_parquet(out_path).sort(["benchmark_row_id", "sentence_index"])
    assert result["benchmark_sentence_id"].to_list() == [
        "doc1:item_1:0",
        "doc1:item_1:1",
        "doc2:item_7:0",
    ]
    assert result["sentencizer_version"].unique().to_list() == ["test"]
    assert "finbert_token_count_512" in result.columns

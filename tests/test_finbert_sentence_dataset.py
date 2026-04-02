from __future__ import annotations

from pathlib import Path

import polars as pl

from thesis_pkg.benchmarking.contracts import DEFAULT_FINBERT_AUTHORITY
from thesis_pkg.benchmarking.contracts import SentenceDatasetConfig
from thesis_pkg.benchmarking.sentences import materialize_sentence_benchmark_dataset


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


def test_materialize_sentence_benchmark_dataset_honors_authority_and_compression(
    tmp_path: Path,
    monkeypatch,
) -> None:
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

    captured: dict[str, object] = {}

    def _fake_annotate(
        df: pl.DataFrame,
        authority,
        *,
        text_col: str = "full_text",
        batch_size: int | None = None,
    ) -> pl.DataFrame:
        del batch_size
        captured["authority"] = authority
        captured["text_col"] = text_col
        return df.with_columns(
            [
                pl.Series("finbert_token_count_512", [5] * df.height, dtype=pl.Int32),
                pl.Series("finbert_token_bucket_512", ["short"] * df.height, dtype=pl.Utf8),
            ]
        )

    def _fake_write_parquet(self, path: Path, *, compression: str = "zstd", **kwargs) -> None:
        del kwargs
        captured["compression"] = compression
        original_write_parquet(self, path, compression=compression)

    original_write_parquet = pl.DataFrame.write_parquet
    monkeypatch.setattr(sentences, "_build_sentencizer", lambda cfg: _FakeNLP())
    monkeypatch.setattr(sentences, "_sentencizer_version", lambda: "test")
    monkeypatch.setattr(sentences, "annotate_finbert_token_lengths_in_batches", _fake_annotate)
    monkeypatch.setattr(pl.DataFrame, "write_parquet", _fake_write_parquet)

    out_path = tmp_path / "sentences.parquet"
    materialize_sentence_benchmark_dataset(
        sections_path,
        SentenceDatasetConfig(enabled=True, compression="lz4"),
        authority=DEFAULT_FINBERT_AUTHORITY,
        compression="lz4",
        out_path=out_path,
    )

    result = pl.read_parquet(out_path).sort(["benchmark_row_id", "sentence_index"])
    assert result["benchmark_sentence_id"].to_list() == [
        "doc1:item_1:0",
        "doc1:item_1:1",
        "doc2:item_7:0",
    ]
    assert captured["authority"] == DEFAULT_FINBERT_AUTHORITY
    assert captured["text_col"] == "full_text"
    assert captured["compression"] == "lz4"

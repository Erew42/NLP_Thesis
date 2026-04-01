from __future__ import annotations

from importlib import metadata
from pathlib import Path
from typing import Any

import polars as pl

from thesis_pkg.benchmarking.contracts import SentenceDatasetConfig
from thesis_pkg.benchmarking.contracts import TokenLengthConfig
from thesis_pkg.benchmarking.token_lengths import annotate_token_lengths


def _build_sentencizer(cfg: SentenceDatasetConfig) -> Any:
    try:
        import spacy
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "spacy is required to materialize the optional sentence benchmark dataset. "
            "Install it before enabling sentence artifact generation."
        ) from exc

    nlp = spacy.blank("en")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


def _sentencizer_version() -> str:
    try:
        return metadata.version("spacy")
    except metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


def _explode_sections_into_sentences(
    sections_df: pl.DataFrame,
    sentence_cfg: SentenceDatasetConfig,
    token_cfg: TokenLengthConfig,
) -> pl.DataFrame:
    nlp = _build_sentencizer(sentence_cfg)
    records: list[dict[str, Any]] = []
    rows = sections_df.select(
        [
            "benchmark_row_id",
            "doc_id",
            "filing_date",
            "filing_year",
            "benchmark_item_code",
            "full_text",
        ]
    ).to_dicts()

    texts = [str(row["full_text"]) for row in rows]
    for row, doc in zip(rows, nlp.pipe(texts, batch_size=sentence_cfg.spacy_batch_size)):
        sentence_index = 0
        for sent in doc.sents:
            sentence_text = sent.text.strip()
            if sentence_cfg.drop_blank_sentences and not sentence_text:
                continue
            records.append(
                {
                    "benchmark_sentence_id": f"{row['benchmark_row_id']}:{sentence_index}",
                    "benchmark_row_id": row["benchmark_row_id"],
                    "doc_id": row["doc_id"],
                    "filing_date": row["filing_date"],
                    "filing_year": row["filing_year"],
                    "benchmark_item_code": row["benchmark_item_code"],
                    "sentence_index": sentence_index,
                    "sentence_text": sentence_text,
                    "sentence_char_count": len(sentence_text),
                    "sentencizer_backend": sentence_cfg.sentencizer_backend,
                    "sentencizer_version": _sentencizer_version(),
                }
            )
            sentence_index += 1

    sentence_df = pl.DataFrame(records)
    if sentence_df.is_empty():
        empty_df = sentence_df.with_columns(
            [
                pl.lit(None, dtype=pl.Int32).alias("finbert_token_count_512"),
                pl.lit(None, dtype=pl.Utf8).alias("finbert_token_bucket_512"),
            ]
        )
        return empty_df.clear()

    sentence_df = sentence_df.rename({"sentence_text": "full_text"})
    sentence_df = annotate_token_lengths(sentence_df, token_cfg)
    return sentence_df.rename({"full_text": "sentence_text"})


def materialize_sentence_benchmark_dataset(
    sections_path: Path,
    cfg: SentenceDatasetConfig,
    *,
    out_path: Path,
) -> Path:
    sections_df = pl.read_parquet(sections_path)
    sentence_df = _explode_sections_into_sentences(sections_df, cfg, TokenLengthConfig())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sentence_df.write_parquet(out_path, compression="zstd")
    return out_path

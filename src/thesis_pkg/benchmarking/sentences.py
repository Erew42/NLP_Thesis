from __future__ import annotations

from importlib import metadata
from pathlib import Path
from typing import Any

import polars as pl

from thesis_pkg.benchmarking.contracts import DEFAULT_FINBERT_AUTHORITY
from thesis_pkg.benchmarking.contracts import FinbertAuthoritySpec
from thesis_pkg.benchmarking.contracts import SentenceDatasetConfig
from thesis_pkg.benchmarking.token_lengths import annotate_finbert_token_lengths


def _build_sentencizer(cfg: SentenceDatasetConfig) -> Any:
    try:
        import spacy
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "spacy is required to materialize the optional sentence benchmark dataset. "
            "Install thesis_pkg[benchmark] before enabling sentence artifact generation."
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


def derive_sentence_frame(
    sections_df: pl.DataFrame,
    cfg: SentenceDatasetConfig,
    *,
    authority: FinbertAuthoritySpec = DEFAULT_FINBERT_AUTHORITY,
) -> pl.DataFrame:
    nlp = _build_sentencizer(cfg)
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

    records: list[dict[str, Any]] = []
    texts = [str(row["full_text"]) for row in rows]
    for row, doc in zip(rows, nlp.pipe(texts, batch_size=cfg.spacy_batch_size)):
        sentence_index = 0
        for sent in doc.sents:
            sentence_text = sent.text.strip()
            if cfg.drop_blank_sentences and not sentence_text:
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
                    "sentencizer_backend": cfg.sentencizer_backend,
                    "sentencizer_version": _sentencizer_version(),
                }
            )
            sentence_index += 1

    if not records:
        return pl.DataFrame(
            schema={
                "benchmark_sentence_id": pl.Utf8,
                "benchmark_row_id": pl.Utf8,
                "doc_id": pl.Utf8,
                "filing_date": pl.Date,
                "filing_year": pl.Int32,
                "benchmark_item_code": pl.Utf8,
                "sentence_index": pl.Int64,
                "sentence_text": pl.Utf8,
                "sentence_char_count": pl.Int64,
                "sentencizer_backend": pl.Utf8,
                "sentencizer_version": pl.Utf8,
                "finbert_token_count_512": pl.Int32,
                "finbert_token_bucket_512": pl.Utf8,
            }
        )

    sentence_df = pl.DataFrame(records)
    sentence_df = sentence_df.rename({"sentence_text": "full_text"})
    sentence_df = annotate_finbert_token_lengths(sentence_df, authority, text_col="full_text")
    return sentence_df.rename({"full_text": "sentence_text"})


def materialize_sentence_benchmark_dataset(
    sections_path: Path,
    cfg: SentenceDatasetConfig,
    *,
    authority: FinbertAuthoritySpec = DEFAULT_FINBERT_AUTHORITY,
    compression: str | None = None,
    out_path: Path,
) -> Path:
    sections_df = pl.read_parquet(sections_path)
    sentence_df = derive_sentence_frame(sections_df, cfg, authority=authority)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sentence_df.write_parquet(out_path, compression=compression or cfg.compression)
    return out_path

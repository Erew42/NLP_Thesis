from __future__ import annotations

from importlib import metadata
from pathlib import Path
from typing import Any

import polars as pl

from thesis_pkg.benchmarking.contracts import DEFAULT_FINBERT_AUTHORITY
from thesis_pkg.benchmarking.contracts import FinbertAuthoritySpec
from thesis_pkg.benchmarking.contracts import SentenceDatasetConfig
from thesis_pkg.benchmarking.token_lengths import annotate_finbert_token_lengths_in_batches


SENTENCE_FRAME_SCHEMA: dict[str, pl.DataType] = {
    "benchmark_sentence_id": pl.Utf8,
    "benchmark_row_id": pl.Utf8,
    "doc_id": pl.Utf8,
    "cik_10": pl.Utf8,
    "accession_nodash": pl.Utf8,
    "filing_date": pl.Date,
    "filing_year": pl.Int32,
    "benchmark_item_code": pl.Utf8,
    "benchmark_item_label": pl.Utf8,
    "source_year_file": pl.Int32,
    "document_type": pl.Utf8,
    "document_type_raw": pl.Utf8,
    "document_type_normalized": pl.Utf8,
    "canonical_item": pl.Utf8,
    "sentence_index": pl.Int64,
    "sentence_text": pl.Utf8,
    "sentence_char_count": pl.Int64,
    "sentencizer_backend": pl.Utf8,
    "sentencizer_version": pl.Utf8,
    "finbert_token_count_512": pl.Int32,
    "finbert_token_bucket_512": pl.Utf8,
}

_SECTION_METADATA_COLUMNS: tuple[str, ...] = (
    "cik_10",
    "accession_nodash",
    "benchmark_item_label",
    "source_year_file",
    "document_type",
    "document_type_raw",
    "document_type_normalized",
    "canonical_item",
)


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


def _empty_sentence_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=SENTENCE_FRAME_SCHEMA)


def _derive_sentence_batch(
    batch_df: pl.DataFrame,
    cfg: SentenceDatasetConfig,
    *,
    authority: FinbertAuthoritySpec,
    nlp: Any,
    sentencizer_version: str,
) -> pl.DataFrame:
    records: list[dict[str, Any]] = []
    select_columns = [
        "benchmark_row_id",
        "doc_id",
        "filing_date",
        "filing_year",
        "benchmark_item_code",
        *[column for column in _SECTION_METADATA_COLUMNS if column in batch_df.columns],
        "full_text",
    ]
    rows = list(
        batch_df.select(select_columns).iter_rows(named=True)
    )
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
                    "cik_10": row.get("cik_10"),
                    "accession_nodash": row.get("accession_nodash"),
                    "filing_date": row["filing_date"],
                    "filing_year": row["filing_year"],
                    "benchmark_item_code": row["benchmark_item_code"],
                    "benchmark_item_label": row.get("benchmark_item_label"),
                    "source_year_file": row.get("source_year_file"),
                    "document_type": row.get("document_type"),
                    "document_type_raw": row.get("document_type_raw"),
                    "document_type_normalized": row.get("document_type_normalized"),
                    "canonical_item": row.get("canonical_item"),
                    "sentence_index": sentence_index,
                    "sentence_text": sentence_text,
                    "sentence_char_count": len(sentence_text),
                    "sentencizer_backend": cfg.sentencizer_backend,
                    "sentencizer_version": sentencizer_version,
                }
            )
            sentence_index += 1

    if not records:
        return _empty_sentence_frame()

    sentence_df = pl.DataFrame(records, schema=SENTENCE_FRAME_SCHEMA)
    sentence_df = sentence_df.rename({"sentence_text": "full_text"})
    sentence_df = annotate_finbert_token_lengths_in_batches(
        sentence_df,
        authority,
        text_col="full_text",
        batch_size=max(cfg.token_length_batch_size, 1),
    )
    return sentence_df.rename({"full_text": "sentence_text"})


def derive_sentence_frame(
    sections_df: pl.DataFrame,
    cfg: SentenceDatasetConfig,
    *,
    authority: FinbertAuthoritySpec = DEFAULT_FINBERT_AUTHORITY,
) -> pl.DataFrame:
    nlp = _build_sentencizer(cfg)
    if sections_df.is_empty():
        return _empty_sentence_frame()

    sentencizer_version = _sentencizer_version()
    chunks: list[pl.DataFrame] = []
    for batch_df in sections_df.iter_slices(n_rows=max(cfg.spacy_batch_size, 1)):
        chunks.append(
            _derive_sentence_batch(
                batch_df,
                cfg,
                authority=authority,
                nlp=nlp,
                sentencizer_version=sentencizer_version,
            )
        )
    non_empty_chunks = [chunk for chunk in chunks if not chunk.is_empty()]
    if not non_empty_chunks:
        return _empty_sentence_frame()
    return pl.concat(non_empty_chunks, how="vertical_relaxed")


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

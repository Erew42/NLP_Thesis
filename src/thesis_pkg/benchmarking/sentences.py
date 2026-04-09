from __future__ import annotations

from importlib import metadata
from pathlib import Path
import re
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

SENTENCE_SPLIT_AUDIT_SCHEMA: dict[str, pl.DataType] = {
    "benchmark_row_id": pl.Utf8,
    "doc_id": pl.Utf8,
    "benchmark_item_code": pl.Utf8,
    "filing_year": pl.Int32,
    "original_char_count": pl.Int64,
    "chunk_index": pl.Int32,
    "chunk_char_count": pl.Int64,
    "split_start_char": pl.Int64,
    "split_end_char": pl.Int64,
    "split_reason": pl.Utf8,
    "total_chunk_count": pl.Int32,
    "warning_boundary_used": pl.Boolean,
}

SENTENCE_CHUNK_CHAR_LIMIT = 250_000
_WARNING_SPLIT_REASONS = frozenset({"whitespace", "hard_limit_250k"})
_DOUBLE_NEWLINE_BOUNDARY_RE = re.compile(r"\n\s*\n+")
_NEWLINE_BOUNDARY_RE = re.compile(r"\n+")
_SENTENCE_PUNCT_BOUNDARY_RE = re.compile(r"""[.!?][)"'\]]*\s+""")
_WHITESPACE_BOUNDARY_RE = re.compile(r"\s+")

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


def _empty_sentence_split_audit_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=SENTENCE_SPLIT_AUDIT_SCHEMA)


def _last_boundary_end(
    text: str,
    *,
    start: int,
    end: int,
    pattern: re.Pattern[str],
) -> int | None:
    last_end: int | None = None
    for match in pattern.finditer(text, start, end):
        if match.end() > start:
            last_end = match.end()
    return last_end


def _choose_chunk_end(text: str, *, start: int) -> tuple[int, str]:
    max_end = min(start + SENTENCE_CHUNK_CHAR_LIMIT, len(text))
    if max_end >= len(text):
        return len(text), "end_of_text"

    for split_reason, pattern in (
        ("double_newline", _DOUBLE_NEWLINE_BOUNDARY_RE),
        ("newline", _NEWLINE_BOUNDARY_RE),
        ("sentence_punct", _SENTENCE_PUNCT_BOUNDARY_RE),
        ("whitespace", _WHITESPACE_BOUNDARY_RE),
    ):
        split_end = _last_boundary_end(text, start=start, end=max_end, pattern=pattern)
        if split_end is not None and split_end > start:
            return split_end, split_reason

    return max_end, "hard_limit_250k"


def _chunk_row_text(row: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    full_text = str(row["full_text"])
    original_char_count = len(full_text)
    if original_char_count <= SENTENCE_CHUNK_CHAR_LIMIT:
        return [row], []

    chunk_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    start = 0
    chunk_index = 0
    while start < original_char_count:
        end, split_reason = _choose_chunk_end(full_text, start=start)
        if end <= start:
            end = min(start + SENTENCE_CHUNK_CHAR_LIMIT, original_char_count)
            split_reason = "hard_limit_250k"

        chunk_text = full_text[start:end]
        chunk_row = dict(row)
        chunk_row["full_text"] = chunk_text
        chunk_rows.append(chunk_row)
        audit_rows.append(
            {
                "benchmark_row_id": row["benchmark_row_id"],
                "doc_id": row["doc_id"],
                "benchmark_item_code": row["benchmark_item_code"],
                "filing_year": row["filing_year"],
                "original_char_count": original_char_count,
                "chunk_index": chunk_index,
                "chunk_char_count": len(chunk_text),
                "split_start_char": start,
                "split_end_char": end,
                "split_reason": split_reason,
                "total_chunk_count": 0,
                "warning_boundary_used": split_reason in _WARNING_SPLIT_REASONS,
            }
        )
        start = end
        chunk_index += 1

    total_chunk_count = len(chunk_rows)
    for audit_row in audit_rows:
        audit_row["total_chunk_count"] = total_chunk_count

    return chunk_rows, audit_rows


def _expand_chunked_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], pl.DataFrame]:
    expanded_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for row in rows:
        row_chunks, row_audit = _chunk_row_text(row)
        expanded_rows.extend(row_chunks)
        audit_rows.extend(row_audit)

    if not audit_rows:
        return expanded_rows, _empty_sentence_split_audit_frame()
    return expanded_rows, pl.DataFrame(audit_rows, schema=SENTENCE_SPLIT_AUDIT_SCHEMA)


def _derive_sentence_batch(
    batch_df: pl.DataFrame,
    cfg: SentenceDatasetConfig,
    *,
    authority: FinbertAuthoritySpec,
    nlp: Any,
    sentencizer_version: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
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
    expanded_rows, split_audit_df = _expand_chunked_rows(rows)
    texts = [str(row["full_text"]) for row in expanded_rows]
    sentence_index_by_row_id: dict[str, int] = {}
    for row, doc in zip(expanded_rows, nlp.pipe(texts, batch_size=cfg.spacy_batch_size)):
        benchmark_row_id = str(row["benchmark_row_id"])
        sentence_index = sentence_index_by_row_id.get(benchmark_row_id, 0)
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
        sentence_index_by_row_id[benchmark_row_id] = sentence_index

    if not records:
        return _empty_sentence_frame(), split_audit_df

    sentence_df = pl.DataFrame(records, schema=SENTENCE_FRAME_SCHEMA)
    sentence_df = sentence_df.rename({"sentence_text": "full_text"})
    sentence_df = annotate_finbert_token_lengths_in_batches(
        sentence_df,
        authority,
        text_col="full_text",
        batch_size=max(cfg.token_length_batch_size, 1),
    )
    return sentence_df.rename({"full_text": "sentence_text"}), split_audit_df


def _derive_sentence_frame_with_split_audit(
    sections_df: pl.DataFrame,
    cfg: SentenceDatasetConfig,
    *,
    authority: FinbertAuthoritySpec = DEFAULT_FINBERT_AUTHORITY,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    nlp = _build_sentencizer(cfg)
    if sections_df.is_empty():
        return _empty_sentence_frame(), _empty_sentence_split_audit_frame()

    sentencizer_version = _sentencizer_version()
    sentence_chunks: list[pl.DataFrame] = []
    split_audit_chunks: list[pl.DataFrame] = []
    for batch_df in sections_df.iter_slices(n_rows=max(cfg.spacy_batch_size, 1)):
        sentence_chunk, split_audit_chunk = _derive_sentence_batch(
            batch_df,
            cfg,
            authority=authority,
            nlp=nlp,
            sentencizer_version=sentencizer_version,
        )
        sentence_chunks.append(sentence_chunk)
        split_audit_chunks.append(split_audit_chunk)

    non_empty_sentence_chunks = [chunk for chunk in sentence_chunks if not chunk.is_empty()]
    sentence_df = (
        pl.concat(non_empty_sentence_chunks, how="vertical_relaxed")
        if non_empty_sentence_chunks
        else _empty_sentence_frame()
    )
    non_empty_audit_chunks = [chunk for chunk in split_audit_chunks if not chunk.is_empty()]
    split_audit_df = (
        pl.concat(non_empty_audit_chunks, how="vertical_relaxed")
        if non_empty_audit_chunks
        else _empty_sentence_split_audit_frame()
    )
    return sentence_df, split_audit_df


def derive_sentence_frame(
    sections_df: pl.DataFrame,
    cfg: SentenceDatasetConfig,
    *,
    authority: FinbertAuthoritySpec = DEFAULT_FINBERT_AUTHORITY,
) -> pl.DataFrame:
    sentence_df, _split_audit_df = _derive_sentence_frame_with_split_audit(
        sections_df,
        cfg,
        authority=authority,
    )
    return sentence_df


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

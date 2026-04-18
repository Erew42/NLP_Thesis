from __future__ import annotations

from importlib import metadata
from pathlib import Path
import re
from typing import Any

import polars as pl

from thesis_pkg.benchmarking.contracts import DEFAULT_FINBERT_AUTHORITY
from thesis_pkg.benchmarking.contracts import FinbertAuthoritySpec
from thesis_pkg.benchmarking.contracts import ItemTextCleaningConfig
from thesis_pkg.benchmarking.contracts import SentenceDatasetConfig
from thesis_pkg.benchmarking.item_text_cleaning import benchmark_item_code_to_text_scope
from thesis_pkg.benchmarking.item_text_cleaning import build_segment_policy_id
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
    "text_scope": pl.Utf8,
    "cleaning_policy_id": pl.Utf8,
    "segment_policy_id": pl.Utf8,
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
_NORMALIZED_WHITESPACE_RE = re.compile(r"\s+")
_REFERENCE_STUB_END_RE = re.compile(r"\b(?:SFAS|SAB|FIN|FASB|ASC|EITF)\s+No\.$", re.IGNORECASE)
_GENERIC_REFERENCE_NO_END_RE = re.compile(r"\bNo\.$", re.IGNORECASE)
_SEPARATOR_ONLY_RE = re.compile(r"^[-_=*]{3,}$")
_UNIT_HEADER_RE = re.compile(r"^\(?DOLLARS IN (?:THOUSANDS|MILLIONS)\)?$", re.IGNORECASE)
_NOTE_CONTINUED_RE = re.compile(r"^(?:NOTE|ITEM)\b.*\((?:CONTINUED|UNAUDITED)\)\.?$", re.IGNORECASE)
_CITATION_PREFIX_ONLY_RE = re.compile(
    r"^(?:\d+[A-Za-z](?:\s*\([A-Za-z]\))?(?:-\d+)?"
    r"|\d+\s*\([A-Za-z]\)(?:-\d+)?"
    r"|\d+-\d+)"
    r"\s*[),.]*(?:\s*\))?\s*$",
    re.IGNORECASE,
)
_CITATION_CONTINUATION_START_RE = re.compile(
    r"^(?:(?:FSP\s+)?(?:SFAS|SAB|FIN|FASB|ASC|EITF)\s+No\."
    r"|(?:statement|opinion|interpretation|position)\s+No\."
    r"|\d+(?:[A-Za-z]|\s*\([A-Za-z]+\)|-\d+|,|\.))",
    re.IGNORECASE,
)
_V3_CITATION_CONTINUATION_START_RE = re.compile(
    r"^(?:(?:FSP\s+)?(?:SFAS|SAB|FIN|FASB|ASC|EITF)\s+No\."
    r"|(?:statement|opinion|interpretation|position)\s+No\."
    r"|\d+(?:[A-Za-z]|\s*\([A-Za-z]+\)|-\d+|,|\.|\)|\s))",
    re.IGNORECASE,
)
_HEADER_KEYWORD_RE = re.compile(
    r"\b(?:CONSOLIDATED|STATEMENTS?|BALANCE SHEETS?|CASH FLOWS?|OPERATIONS|CHANGES IN|"
    r"REPORTABLE SEGMENTS|AND SUBSIDIARIES|PAYMENTS DUE BY PERIOD)\b",
    re.IGNORECASE,
)
_ITEM7_ARTIFACT_CLEANUP_POLICIES = frozenset(
    {
        "item7_reference_stitch_protect_v1",
        "item7_reference_stitch_protect_v2",
        "reference_stitch_protect_v3",
    }
)
_V3_REFERENCE_STITCH_SCOPES = frozenset(
    {
        "item_1_business",
        "item_1a_risk_factors",
        "item_7_mda",
    }
)

_SECTION_METADATA_COLUMNS: tuple[str, ...] = (
    "cik_10",
    "accession_nodash",
    "benchmark_item_label",
    "source_year_file",
    "document_type",
    "document_type_raw",
    "document_type_normalized",
    "canonical_item",
    "text_scope",
    "cleaning_policy_id",
    "segment_policy_id",
)


def _ensure_sentence_provenance(
    sections_df: pl.DataFrame,
    cfg: SentenceDatasetConfig,
    authority: FinbertAuthoritySpec,
) -> pl.DataFrame:
    schema_names = set(sections_df.columns)
    raw_segment_policy_id = build_segment_policy_id(
        cfg,
        ItemTextCleaningConfig(enabled=False, cleaning_policy_id="raw_item_text"),
        authority,
        chunk_char_limit=SENTENCE_CHUNK_CHAR_LIMIT,
    )
    text_scope_expr = (
        pl.when(
            pl.col("text_scope").cast(pl.Utf8, strict=False).is_not_null()
            if "text_scope" in schema_names
            else pl.lit(False)
        )
        .then(
            pl.col("text_scope").cast(pl.Utf8, strict=False)
            if "text_scope" in schema_names
            else pl.lit(None, dtype=pl.Utf8)
        )
        .otherwise(
            pl.col("benchmark_item_code")
            .cast(pl.Utf8, strict=False)
            .map_elements(benchmark_item_code_to_text_scope, return_dtype=pl.Utf8)
        )
    )
    return sections_df.with_columns(
        text_scope_expr.alias("text_scope"),
        pl.coalesce(
            [
                (
                    pl.col("cleaning_policy_id").cast(pl.Utf8, strict=False)
                    if "cleaning_policy_id" in schema_names
                    else pl.lit(None, dtype=pl.Utf8)
                ),
                pl.lit("raw_item_text", dtype=pl.Utf8),
            ]
        ).alias("cleaning_policy_id"),
        pl.coalesce(
            [
                (
                    pl.col("segment_policy_id").cast(pl.Utf8, strict=False)
                    if "segment_policy_id" in schema_names
                    else pl.lit(None, dtype=pl.Utf8)
                ),
                pl.lit(raw_segment_policy_id, dtype=pl.Utf8),
            ]
        ).alias("segment_policy_id"),
    )


def _build_sentencizer(cfg: SentenceDatasetConfig) -> Any:
    try:
        import spacy
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "spacy is required to materialize the optional sentence benchmark dataset. "
            "Install thesis_pkg[benchmark] before enabling sentence artifact generation."
        ) from exc

    # Keep the simple sentencizer backend here. The remaining quality issues in this
    # project are mostly citation-continuation stitching and upstream structure leakage,
    # not punctuation tuning, and the official spaCy Sentencizer config surface does
    # not address those problems directly.
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


def _normalize_sentence_key(text: str) -> str:
    return _NORMALIZED_WHITESPACE_RE.sub(" ", str(text)).strip()


def _ends_with_reference_stub(text: str) -> bool:
    return bool(_REFERENCE_STUB_END_RE.search(_normalize_sentence_key(text)))


def _ends_with_generic_reference_no(text: str) -> bool:
    return bool(_GENERIC_REFERENCE_NO_END_RE.search(_normalize_sentence_key(text)))


def _is_citation_prefix_only_line(line: str) -> bool:
    return bool(_CITATION_PREFIX_ONLY_RE.fullmatch(line.strip()))


def _looks_like_citation_continuation(text: str) -> bool:
    return bool(_CITATION_CONTINUATION_START_RE.search(_normalize_sentence_key(text)))


def _looks_like_citation_continuation_v3(text: str) -> bool:
    return bool(_V3_CITATION_CONTINUATION_START_RE.search(_normalize_sentence_key(text)))


def _is_separator_line(line: str) -> bool:
    return bool(_SEPARATOR_ONLY_RE.fullmatch(line.strip()))


def _is_header_like_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if _is_citation_prefix_only_line(stripped):
        return False
    if _is_separator_line(stripped):
        return True
    if _UNIT_HEADER_RE.fullmatch(stripped):
        return True
    if _NOTE_CONTINUED_RE.fullmatch(stripped):
        return True
    if _HEADER_KEYWORD_RE.search(stripped) and stripped.upper() == stripped:
        return True
    letters = [char for char in stripped if char.isalpha()]
    if not letters:
        return False
    if any(char.islower() for char in letters):
        return False
    if len(stripped) > 140:
        return False
    return True


def _strip_leading_artifact_lines(text: str) -> str:
    lines = [line.strip() for line in str(text).splitlines()]
    if not lines:
        return ""
    kept_start = 0
    while kept_start < len(lines) and _is_header_like_line(lines[kept_start]):
        kept_start += 1
    if kept_start >= len(lines):
        return ""
    return "\n".join(line for line in lines[kept_start:] if line)


def _is_artifact_only_sentence(text: str) -> bool:
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if not lines:
        return True
    return all(_is_header_like_line(line) for line in lines)


def _join_sentence_fragments(current: str, next_text: str) -> str:
    left = current.rstrip()
    right = next_text.lstrip()
    if not left:
        return right
    if not right:
        return left
    if left.endswith("-") and right[:1].islower():
        return f"{left}{right}"
    return f"{left} {right}"


def _should_stitch_item7_sentence(current: str, next_text: str) -> bool:
    next_clean = next_text.strip()
    if not next_clean:
        return False
    if _ends_with_reference_stub(current):
        return True
    return current.rstrip().endswith("-") and next_clean[:1].islower()


def _should_stitch_item7_sentence_v2(current: str, next_text: str) -> bool:
    next_clean = next_text.strip()
    if not next_clean:
        return False
    if _should_stitch_item7_sentence(current, next_clean):
        return True
    return _ends_with_generic_reference_no(current) and _looks_like_citation_continuation(next_clean)


def _should_stitch_reference_sentence_v3(current: str, next_text: str) -> bool:
    next_clean = next_text.strip()
    if not next_clean:
        return False
    if _should_stitch_item7_sentence(current, next_clean):
        return True
    return _ends_with_generic_reference_no(current) and _looks_like_citation_continuation_v3(next_clean)


def _should_apply_item7_artifact_cleanup(*, policy: str, text_scope: str | None) -> bool:
    return text_scope == "item_7_mda" and policy in _ITEM7_ARTIFACT_CLEANUP_POLICIES


def _citation_stitcher_for_policy(policy: str):
    if policy == "item7_reference_stitch_protect_v1":
        return _should_stitch_item7_sentence
    if policy == "item7_reference_stitch_protect_v2":
        return _should_stitch_item7_sentence_v2
    if policy == "reference_stitch_protect_v3":
        return _should_stitch_reference_sentence_v3
    return None


def _should_apply_reference_stitching(*, policy: str, text_scope: str | None) -> bool:
    stitcher = _citation_stitcher_for_policy(policy)
    if stitcher is None or text_scope is None:
        return False
    if policy == "reference_stitch_protect_v3":
        return text_scope in _V3_REFERENCE_STITCH_SCOPES
    return text_scope == "item_7_mda"


def _postprocess_sentence_texts(
    sentence_texts: list[str],
    *,
    text_scope: str | None,
    cfg: SentenceDatasetConfig,
) -> list[str]:
    if cfg.postprocess_policy == "none":
        return sentence_texts

    artifact_cleanup = _should_apply_item7_artifact_cleanup(
        policy=cfg.postprocess_policy,
        text_scope=text_scope,
    )
    reference_stitching = _should_apply_reference_stitching(
        policy=cfg.postprocess_policy,
        text_scope=text_scope,
    )
    if not artifact_cleanup and not reference_stitching:
        return sentence_texts

    normalized_texts = (
        [_strip_leading_artifact_lines(text) for text in sentence_texts]
        if artifact_cleanup
        else list(sentence_texts)
    )
    stitcher = _citation_stitcher_for_policy(cfg.postprocess_policy)
    processed: list[str] = []
    idx = 0
    while idx < len(normalized_texts):
        current = normalized_texts[idx].strip()
        if not current:
            idx += 1
            continue
        next_idx = idx + 1
        while next_idx < len(normalized_texts):
            while next_idx < len(normalized_texts) and not normalized_texts[next_idx].strip():
                next_idx += 1
            if next_idx >= len(normalized_texts):
                break
            next_text = normalized_texts[next_idx].strip()
            if not reference_stitching or stitcher is None or not stitcher(current, next_text):
                break
            current = _join_sentence_fragments(current, next_text)
            idx = next_idx
            next_idx = idx + 1
        if artifact_cleanup and _is_artifact_only_sentence(current):
            idx += 1
            continue
        if current:
            processed.append(current)
        idx += 1
    return processed


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
        raw_sentence_texts = [sent.text.strip() for sent in doc.sents]
        sentence_texts = _postprocess_sentence_texts(
            raw_sentence_texts,
            text_scope=row.get("text_scope"),
            cfg=cfg,
        )
        for sentence_text in sentence_texts:
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
                    "text_scope": row.get("text_scope"),
                    "cleaning_policy_id": row.get("cleaning_policy_id"),
                    "segment_policy_id": row.get("segment_policy_id"),
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
        bucket_edges=cfg.bucket_edges,
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
    sections_df = _ensure_sentence_provenance(sections_df, cfg, authority)

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

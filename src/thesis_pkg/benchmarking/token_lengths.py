from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import polars as pl

from thesis_pkg.benchmarking.contracts import FinbertAuthoritySpec


FINBERT_TOKEN_COUNT_COLUMN = "finbert_token_count_512"
FINBERT_TOKEN_BUCKET_COLUMN = "finbert_token_bucket_512"


def _import_bert_tokenizer():
    try:
        from transformers import BertTokenizer
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "transformers is required for FinBERT token-count annotation. "
            "Install thesis_pkg[benchmark] before running benchmark tooling."
        ) from exc
    return BertTokenizer


@lru_cache(maxsize=None)
def load_finbert_tokenizer(authority: FinbertAuthoritySpec):
    bert_tokenizer = _import_bert_tokenizer()
    kwargs = {"do_lower_case": authority.do_lower_case}
    if authority.tokenizer_revision is not None:
        kwargs["revision"] = authority.tokenizer_revision
    return bert_tokenizer.from_pretrained(
        authority.model_name,
        **kwargs,
    )


def assign_finbert_token_bucket(token_count: int, authority: FinbertAuthoritySpec) -> str:
    short_edge, medium_edge = authority.token_bucket_edges
    if token_count <= short_edge:
        return "short"
    if token_count <= medium_edge:
        return "medium"
    if token_count <= authority.token_count_max_length:
        return "long"
    raise ValueError(
        f"Token count {token_count} exceeds the fixed FinBERT authority max length "
        f"{authority.token_count_max_length}."
    )


def compute_finbert_token_lengths(
    texts: Sequence[str],
    authority: FinbertAuthoritySpec,
) -> list[int]:
    if not texts:
        return []
    tokenizer = load_finbert_tokenizer(authority)
    encoded = tokenizer(
        list(texts),
        add_special_tokens=True,
        truncation=True,
        max_length=authority.token_count_max_length,
    )
    return [len(input_ids) for input_ids in encoded["input_ids"]]


def annotate_finbert_token_lengths(
    df: pl.DataFrame,
    authority: FinbertAuthoritySpec,
    *,
    text_col: str = "full_text",
) -> pl.DataFrame:
    token_counts = compute_finbert_token_lengths(df[text_col].to_list(), authority)
    token_buckets = [assign_finbert_token_bucket(count, authority) for count in token_counts]
    return df.with_columns(
        [
            pl.Series(FINBERT_TOKEN_COUNT_COLUMN, token_counts, dtype=pl.Int32),
            pl.Series(FINBERT_TOKEN_BUCKET_COLUMN, token_buckets, dtype=pl.Utf8),
        ]
    )


def annotate_finbert_token_lengths_in_batches(
    df: pl.DataFrame,
    authority: FinbertAuthoritySpec,
    *,
    text_col: str = "full_text",
    batch_size: int = 1024,
) -> pl.DataFrame:
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if df.is_empty():
        return annotate_finbert_token_lengths(df, authority, text_col=text_col)

    chunks: list[pl.DataFrame] = []
    for offset in range(0, df.height, batch_size):
        chunk = df.slice(offset, batch_size)
        chunks.append(annotate_finbert_token_lengths(chunk, authority, text_col=text_col))
    return pl.concat(chunks, how="vertical_relaxed")

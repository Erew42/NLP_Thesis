from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import polars as pl

from thesis_pkg.benchmarking.contracts import TokenLengthConfig


@lru_cache(maxsize=None)
def _load_tokenizer(tokenizer_name: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "transformers is required to compute FinBERT token lengths. "
            "Install it before building benchmark datasets."
        ) from exc
    return AutoTokenizer.from_pretrained(tokenizer_name)


def assign_token_bucket(token_count: int, cfg: TokenLengthConfig) -> str:
    short_edge, medium_edge, long_edge = cfg.bucket_edges
    if token_count <= short_edge:
        return "short"
    if token_count <= medium_edge:
        return "medium"
    if token_count <= long_edge:
        return "long"
    raise ValueError(
        f"Token count {token_count} exceeds configured max bucket edge {long_edge}. "
        "Token counts should already be truncated to max_length."
    )


def compute_token_lengths(texts: Sequence[str], cfg: TokenLengthConfig) -> list[int]:
    if not texts:
        return []
    tokenizer = _load_tokenizer(cfg.tokenizer_name)
    encoded = tokenizer(
        list(texts),
        add_special_tokens=cfg.add_special_tokens,
        truncation=cfg.truncation,
        max_length=cfg.max_length,
    )
    return [len(input_ids) for input_ids in encoded["input_ids"]]


def annotate_token_lengths(df: pl.DataFrame, cfg: TokenLengthConfig) -> pl.DataFrame:
    token_counts = compute_token_lengths(df["full_text"].to_list(), cfg)
    token_buckets = [assign_token_bucket(count, cfg) for count in token_counts]
    return df.with_columns(
        [
            pl.Series("finbert_token_count_512", token_counts, dtype=pl.Int32),
            pl.Series("finbert_token_bucket_512", token_buckets, dtype=pl.Utf8),
        ]
    )

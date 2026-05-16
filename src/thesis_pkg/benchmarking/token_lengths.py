from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import polars as pl

from thesis_pkg.benchmarking.contracts import BucketEdgeSpec
from thesis_pkg.benchmarking.contracts import FinbertAuthoritySpec

try:
    from thesis_native import _lm2011_rust
except Exception as exc:  # pragma: no cover - optional native extension
    _lm2011_rust = None
    _FINBERT_TOKEN_RUST_IMPORT_ERROR: str | None = f"{type(exc).__name__}: {exc}"
else:
    _FINBERT_TOKEN_RUST_IMPORT_ERROR = None


FINBERT_TOKEN_COUNT_COLUMN = "finbert_token_count_512"
FINBERT_TOKEN_BUCKET_COLUMN = "finbert_token_bucket_512"
_FINBERT_TOKEN_RUST_METRICS: dict[str, int] = {
    "bucket_fast_success": 0,
    "bucket_fast_failures": 0,
    "bucket_fallbacks": 0,
    "bucket_values_fast_success": 0,
    "bucket_values_fast_failures": 0,
    "bucket_values_fallbacks": 0,
}


def get_finbert_token_rust_accel_metrics() -> dict[str, int | str | bool | None]:
    metrics: dict[str, int | str | bool | None] = dict(_FINBERT_TOKEN_RUST_METRICS)
    metrics["rust_accel_available"] = _lm2011_rust is not None
    metrics["rust_accel_import_error"] = _FINBERT_TOKEN_RUST_IMPORT_ERROR
    return metrics


def reset_finbert_token_rust_accel_metrics() -> None:
    for key in _FINBERT_TOKEN_RUST_METRICS:
        _FINBERT_TOKEN_RUST_METRICS[key] = 0


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


def _resolved_bucket_edges(
    authority: FinbertAuthoritySpec,
    bucket_edges: BucketEdgeSpec | None,
) -> BucketEdgeSpec:
    if bucket_edges is not None:
        return bucket_edges
    short_edge, medium_edge = authority.token_bucket_edges
    return BucketEdgeSpec(short_edge=short_edge, medium_edge=medium_edge)


def assign_finbert_token_bucket_py(
    token_count: int,
    authority: FinbertAuthoritySpec,
    *,
    bucket_edges: BucketEdgeSpec | None = None,
) -> str:
    resolved_bucket_edges = _resolved_bucket_edges(authority, bucket_edges)
    short_edge = resolved_bucket_edges.short_edge
    medium_edge = resolved_bucket_edges.medium_edge
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


def assign_finbert_token_bucket(
    token_count: int,
    authority: FinbertAuthoritySpec,
    *,
    bucket_edges: BucketEdgeSpec | None = None,
) -> str:
    resolved_bucket_edges = _resolved_bucket_edges(authority, bucket_edges)
    if _lm2011_rust is not None:
        try:
            out = _lm2011_rust.assign_finbert_token_bucket_value(
                int(token_count),
                int(resolved_bucket_edges.short_edge),
                int(resolved_bucket_edges.medium_edge),
                int(authority.token_count_max_length),
            )
            _FINBERT_TOKEN_RUST_METRICS["bucket_fast_success"] += 1
            return str(out)
        except Exception:
            _FINBERT_TOKEN_RUST_METRICS["bucket_fast_failures"] += 1
    _FINBERT_TOKEN_RUST_METRICS["bucket_fallbacks"] += 1
    return assign_finbert_token_bucket_py(token_count, authority, bucket_edges=resolved_bucket_edges)


def assign_finbert_token_buckets_py(
    token_counts: Sequence[int],
    authority: FinbertAuthoritySpec,
    *,
    bucket_edges: BucketEdgeSpec | None = None,
) -> list[str]:
    resolved_bucket_edges = _resolved_bucket_edges(authority, bucket_edges)
    return [
        assign_finbert_token_bucket_py(
            int(count),
            authority,
            bucket_edges=resolved_bucket_edges,
        )
        for count in token_counts
    ]


def assign_finbert_token_buckets(
    token_counts: Sequence[int],
    authority: FinbertAuthoritySpec,
    *,
    bucket_edges: BucketEdgeSpec | None = None,
) -> list[str]:
    resolved_bucket_edges = _resolved_bucket_edges(authority, bucket_edges)
    if _lm2011_rust is not None:
        try:
            out = _lm2011_rust.assign_finbert_token_bucket_values(
                [int(count) for count in token_counts],
                int(resolved_bucket_edges.short_edge),
                int(resolved_bucket_edges.medium_edge),
                int(authority.token_count_max_length),
            )
            _FINBERT_TOKEN_RUST_METRICS["bucket_values_fast_success"] += 1
            return [str(value) for value in out]
        except Exception:
            _FINBERT_TOKEN_RUST_METRICS["bucket_values_fast_failures"] += 1
    _FINBERT_TOKEN_RUST_METRICS["bucket_values_fallbacks"] += 1
    return assign_finbert_token_buckets_py(
        token_counts,
        authority,
        bucket_edges=resolved_bucket_edges,
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
    bucket_edges: BucketEdgeSpec | None = None,
) -> pl.DataFrame:
    token_counts = compute_finbert_token_lengths(df[text_col].to_list(), authority)
    token_buckets = assign_finbert_token_buckets(token_counts, authority, bucket_edges=bucket_edges)
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
    bucket_edges: BucketEdgeSpec | None = None,
) -> pl.DataFrame:
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if df.is_empty():
        return annotate_finbert_token_lengths(
            df,
            authority,
            text_col=text_col,
            bucket_edges=bucket_edges,
        )

    chunks: list[pl.DataFrame] = []
    for offset in range(0, df.height, batch_size):
        chunk = df.slice(offset, batch_size)
        chunks.append(
            annotate_finbert_token_lengths(
                chunk,
                authority,
                text_col=text_col,
                bucket_edges=bucket_edges,
            )
        )
    return pl.concat(chunks, how="vertical_relaxed")

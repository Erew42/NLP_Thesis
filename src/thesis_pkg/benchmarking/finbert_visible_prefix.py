from __future__ import annotations

from functools import lru_cache
from typing import Any

import polars as pl

from thesis_pkg.benchmarking.contracts import FinbertAuthoritySpec
from thesis_pkg.benchmarking.token_lengths import FINBERT_TOKEN_COUNT_COLUMN
from thesis_pkg.core.sec.lm2011_text import tokenize_lm2011_text

try:
    from thesis_native import _lm2011_rust
except Exception as exc:  # pragma: no cover - optional native extension
    _lm2011_rust = None
    _FINBERT_VISIBLE_PREFIX_RUST_IMPORT_ERROR: str | None = f"{type(exc).__name__}: {exc}"
else:
    _FINBERT_VISIBLE_PREFIX_RUST_IMPORT_ERROR = None


VISIBLE_PREFIX_FAST_OFFSET_POLICY_ID = "bert_fast_offset_prefix_v1"
VISIBLE_PREFIX_DECODE_POLICY_ID = "bert_decode_visible_tokens_v1"
VISIBLE_PREFIX_UNCAPPED_POLICY_ID = "uncapped_sentence_text_v1"
VISIBLE_PREFIX_TEXT_COLUMN = "visible_sentence_text"
VISIBLE_PREFIX_POLICY_COLUMN = "visible_prefix_policy_id"
VISIBLE_PREFIX_FALLBACK_REASON_COLUMN = "visible_prefix_fallback_reason"
VISIBLE_PREFIX_UNTRUNCATED_TOKEN_COUNT_COLUMN = "finbert_untruncated_token_count"
VISIBLE_PREFIX_TRUE_OVER_512_COLUMN = "finbert_true_over_512"
ORIGINAL_SENTENCE_CHAR_COUNT_COLUMN = "original_sentence_char_count"
VISIBLE_SENTENCE_CHAR_COUNT_COLUMN = "visible_sentence_char_count"
ORIGINAL_SENTENCE_LM_TOKEN_COUNT_COLUMN = "original_sentence_lm_token_count"
VISIBLE_SENTENCE_LM_TOKEN_COUNT_COLUMN = "visible_sentence_lm_token_count"

_FINBERT_VISIBLE_PREFIX_RUST_METRICS: dict[str, int] = {
    "retained_end_fast_success": 0,
    "retained_end_fallbacks": 0,
    "lm_token_counts_fast_success": 0,
    "lm_token_counts_fast_failures": 0,
    "lm_token_counts_fallbacks": 0,
}


def get_finbert_visible_prefix_rust_accel_metrics() -> dict[str, int | str | bool | None]:
    metrics: dict[str, int | str | bool | None] = dict(_FINBERT_VISIBLE_PREFIX_RUST_METRICS)
    metrics["rust_accel_available"] = _lm2011_rust is not None
    metrics["rust_accel_import_error"] = _FINBERT_VISIBLE_PREFIX_RUST_IMPORT_ERROR
    return metrics


def reset_finbert_visible_prefix_rust_accel_metrics() -> None:
    for key in _FINBERT_VISIBLE_PREFIX_RUST_METRICS:
        _FINBERT_VISIBLE_PREFIX_RUST_METRICS[key] = 0


def _import_auto_tokenizer():
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "transformers is required for FinBERT visible-prefix reconstruction. "
            "Install thesis_pkg[benchmark] before running benchmark tooling."
        ) from exc
    return AutoTokenizer


def _import_bert_tokenizer():
    try:
        from transformers import BertTokenizer
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "transformers is required for FinBERT visible-prefix reconstruction. "
            "Install thesis_pkg[benchmark] before running benchmark tooling."
        ) from exc
    return BertTokenizer


@lru_cache(maxsize=None)
def load_finbert_visible_prefix_tokenizer(authority: FinbertAuthoritySpec):
    auto_tokenizer = _import_auto_tokenizer()
    kwargs: dict[str, Any] = {
        "use_fast": True,
        "do_lower_case": authority.do_lower_case,
    }
    if authority.tokenizer_revision is not None:
        kwargs["revision"] = authority.tokenizer_revision
    try:
        return auto_tokenizer.from_pretrained(authority.model_name, **kwargs)
    except Exception:
        bert_tokenizer = _import_bert_tokenizer()
        fallback_kwargs = {"do_lower_case": authority.do_lower_case}
        if authority.tokenizer_revision is not None:
            fallback_kwargs["revision"] = authority.tokenizer_revision
        return bert_tokenizer.from_pretrained(authority.model_name, **fallback_kwargs)


def _encoding_value(encoded: Any, key: str) -> Any:
    if isinstance(encoded, dict):
        return encoded.get(key)
    return encoded[key]


def _first_sequence(value: Any) -> Any:
    if (
        isinstance(value, list)
        and value
        and isinstance(value[0], list)
    ):
        return value[0]
    return value


def _untruncated_token_count(tokenizer: Any, text: str, *, max_length: int) -> int | None:
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
        )
        input_ids = _first_sequence(_encoding_value(encoded, "input_ids"))
        return len(input_ids)
    except Exception:
        return None


def _visible_prefix_retained_end_py(offsets: Any, special_mask: Any, text_len: int) -> int:
    retained_end = 0
    for offset, is_special in zip(offsets, special_mask):
        if is_special:
            continue
        if offset is None or len(offset) < 2:
            continue
        start, end = int(offset[0]), int(offset[1])
        if end <= start:
            continue
        retained_end = max(retained_end, end)
    return min(retained_end, text_len)


def _visible_prefix_retained_end(offsets: Any, special_mask: Any, text_len: int) -> int:
    if _lm2011_rust is not None:
        try:
            out = int(
                _lm2011_rust.finbert_visible_prefix_retained_end(
                    offsets,
                    special_mask,
                    int(text_len),
                )
            )
            _FINBERT_VISIBLE_PREFIX_RUST_METRICS["retained_end_fast_success"] += 1
            return out
        except Exception:
            _FINBERT_VISIBLE_PREFIX_RUST_METRICS["retained_end_fallbacks"] += 1
    else:
        _FINBERT_VISIBLE_PREFIX_RUST_METRICS["retained_end_fallbacks"] += 1
    return _visible_prefix_retained_end_py(offsets, special_mask, text_len)


def _fast_offset_visible_prefix(
    tokenizer: Any,
    text: str,
    *,
    max_length: int,
) -> str:
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("FinBERT visible-prefix offset slicing requires a fast tokenizer.")
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
    )
    offsets = _first_sequence(_encoding_value(encoded, "offset_mapping"))
    special_mask = _encoding_value(encoded, "special_tokens_mask")
    if special_mask is None:
        special_mask = [0] * len(offsets)
    else:
        special_mask = _first_sequence(special_mask)

    retained_end = _visible_prefix_retained_end(offsets, special_mask, len(text))
    return text[:retained_end]


def _decoded_visible_prefix(
    tokenizer: Any,
    text: str,
    *,
    max_length: int,
) -> str:
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        return_special_tokens_mask=True,
    )
    input_ids = _first_sequence(_encoding_value(encoded, "input_ids"))
    special_mask = _encoding_value(encoded, "special_tokens_mask")
    if special_mask is not None:
        special_mask = _first_sequence(special_mask)
        input_ids = [
            token_id
            for token_id, is_special in zip(input_ids, special_mask)
            if not is_special
        ]
    return tokenizer.decode(
        input_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def _lm_token_count(text: str | None) -> int:
    return _lm_token_counts([text])[0]


def _lm_token_counts_py(texts: list[str | None]) -> list[int]:
    return [len(tokenize_lm2011_text(text)) for text in texts]


def _lm_token_counts(texts: list[str | None]) -> list[int]:
    if _lm2011_rust is not None:
        try:
            out = _lm2011_rust.count_lm2011_text_token_values(texts)
            _FINBERT_VISIBLE_PREFIX_RUST_METRICS["lm_token_counts_fast_success"] += 1
            return [int(value) for value in out]
        except Exception:
            _FINBERT_VISIBLE_PREFIX_RUST_METRICS["lm_token_counts_fast_failures"] += 1
            _FINBERT_VISIBLE_PREFIX_RUST_METRICS["lm_token_counts_fallbacks"] += 1
    else:
        _FINBERT_VISIBLE_PREFIX_RUST_METRICS["lm_token_counts_fallbacks"] += 1
    return _lm_token_counts_py(texts)


def add_finbert_visible_sentence_prefixes(
    df: pl.DataFrame,
    authority: FinbertAuthoritySpec,
    *,
    tokenizer: Any | None = None,
    text_col: str = "sentence_text",
    token_count_col: str = FINBERT_TOKEN_COUNT_COLUMN,
) -> pl.DataFrame:
    if text_col not in df.columns:
        raise ValueError(f"FinBERT visible-prefix input missing required column: {text_col}")
    if token_count_col not in df.columns:
        raise ValueError(f"FinBERT visible-prefix input missing required column: {token_count_col}")
    if df.is_empty():
        return df.with_columns(
            [
                pl.Series(VISIBLE_PREFIX_TEXT_COLUMN, [], dtype=pl.Utf8),
                pl.Series(VISIBLE_PREFIX_POLICY_COLUMN, [], dtype=pl.Utf8),
                pl.Series(VISIBLE_PREFIX_FALLBACK_REASON_COLUMN, [], dtype=pl.Utf8),
                pl.Series(VISIBLE_PREFIX_UNTRUNCATED_TOKEN_COUNT_COLUMN, [], dtype=pl.Int32),
                pl.Series(VISIBLE_PREFIX_TRUE_OVER_512_COLUMN, [], dtype=pl.Boolean),
                pl.Series(ORIGINAL_SENTENCE_CHAR_COUNT_COLUMN, [], dtype=pl.Int32),
                pl.Series(VISIBLE_SENTENCE_CHAR_COUNT_COLUMN, [], dtype=pl.Int32),
                pl.Series(ORIGINAL_SENTENCE_LM_TOKEN_COUNT_COLUMN, [], dtype=pl.Int32),
                pl.Series(VISIBLE_SENTENCE_LM_TOKEN_COUNT_COLUMN, [], dtype=pl.Int32),
            ]
        )

    resolved_tokenizer = tokenizer or load_finbert_visible_prefix_tokenizer(authority)
    max_length = int(authority.token_count_max_length)
    visible_texts: list[str | None] = []
    policies: list[str] = []
    fallback_reasons: list[str | None] = []
    untruncated_counts: list[int | None] = []
    true_over_512: list[bool | None] = []
    original_char_counts: list[int] = []
    visible_char_counts: list[int] = []
    normalized_texts: list[str] = []

    texts = df[text_col].to_list()
    token_counts = df[token_count_col].to_list()
    for raw_text, raw_token_count in zip(texts, token_counts):
        text = "" if raw_text is None else str(raw_text)
        normalized_texts.append(text)
        token_count = int(raw_token_count) if raw_token_count is not None else None
        is_at_cap = token_count == max_length
        original_char_counts.append(len(text))

        if not is_at_cap:
            visible_text = text
            policy_id = VISIBLE_PREFIX_UNCAPPED_POLICY_ID
            fallback_reason = None
            untruncated_count = token_count
        else:
            untruncated_count = _untruncated_token_count(
                resolved_tokenizer,
                text,
                max_length=max_length,
            )
            try:
                visible_text = _fast_offset_visible_prefix(
                    resolved_tokenizer,
                    text,
                    max_length=max_length,
                )
                policy_id = VISIBLE_PREFIX_FAST_OFFSET_POLICY_ID
                fallback_reason = None
            except Exception as exc:
                visible_text = _decoded_visible_prefix(
                    resolved_tokenizer,
                    text,
                    max_length=max_length,
                )
                policy_id = VISIBLE_PREFIX_DECODE_POLICY_ID
                fallback_reason = f"{type(exc).__name__}: {exc}"

        visible_texts.append(visible_text)
        policies.append(policy_id)
        fallback_reasons.append(fallback_reason)
        untruncated_counts.append(untruncated_count)
        true_over_512.append(
            None if untruncated_count is None else bool(untruncated_count > max_length)
        )
        visible_char_counts.append(len(visible_text or ""))

    original_lm_token_counts = _lm_token_counts(normalized_texts)
    visible_lm_token_counts = _lm_token_counts(visible_texts)

    return df.with_columns(
        [
            pl.Series(VISIBLE_PREFIX_TEXT_COLUMN, visible_texts, dtype=pl.Utf8),
            pl.Series(VISIBLE_PREFIX_POLICY_COLUMN, policies, dtype=pl.Utf8),
            pl.Series(VISIBLE_PREFIX_FALLBACK_REASON_COLUMN, fallback_reasons, dtype=pl.Utf8),
            pl.Series(
                VISIBLE_PREFIX_UNTRUNCATED_TOKEN_COUNT_COLUMN,
                untruncated_counts,
                dtype=pl.Int32,
            ),
            pl.Series(VISIBLE_PREFIX_TRUE_OVER_512_COLUMN, true_over_512, dtype=pl.Boolean),
            pl.Series(
                ORIGINAL_SENTENCE_CHAR_COUNT_COLUMN,
                original_char_counts,
                dtype=pl.Int32,
            ),
            pl.Series(
                VISIBLE_SENTENCE_CHAR_COUNT_COLUMN,
                visible_char_counts,
                dtype=pl.Int32,
            ),
            pl.Series(
                ORIGINAL_SENTENCE_LM_TOKEN_COUNT_COLUMN,
                original_lm_token_counts,
                dtype=pl.Int32,
            ),
            pl.Series(
                VISIBLE_SENTENCE_LM_TOKEN_COUNT_COLUMN,
                visible_lm_token_counts,
                dtype=pl.Int32,
            ),
        ]
    )

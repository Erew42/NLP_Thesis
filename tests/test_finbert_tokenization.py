from __future__ import annotations

from pathlib import Path

import polars as pl

from thesis_pkg.benchmarking.contracts import BucketEdgeSpec
from thesis_pkg.benchmarking.contracts import DEFAULT_FINBERT_AUTHORITY
from thesis_pkg.benchmarking.finbert_visible_prefix import VISIBLE_PREFIX_DECODE_POLICY_ID
from thesis_pkg.benchmarking.finbert_visible_prefix import VISIBLE_PREFIX_FAST_OFFSET_POLICY_ID
from thesis_pkg.benchmarking.finbert_visible_prefix import VISIBLE_PREFIX_TEXT_COLUMN
from thesis_pkg.benchmarking.finbert_visible_prefix import VISIBLE_PREFIX_TRUE_OVER_512_COLUMN
from thesis_pkg.benchmarking.finbert_visible_prefix import add_finbert_visible_sentence_prefixes
from thesis_pkg.benchmarking.token_lengths import annotate_finbert_token_lengths
from thesis_pkg.benchmarking.token_lengths import annotate_finbert_token_lengths_in_batches
from thesis_pkg.benchmarking.token_lengths import load_finbert_tokenizer


def test_load_finbert_tokenizer_uses_explicit_bert_tokenizer(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeBertTokenizer:
        @classmethod
        def from_pretrained(cls, model_name: str, do_lower_case: bool = False):
            captured["model_name"] = model_name
            captured["do_lower_case"] = do_lower_case
            return cls()

        def __call__(self, texts, **kwargs):
            del kwargs
            return {"input_ids": [[1, 2, 3] for _ in texts]}

    from thesis_pkg.benchmarking import token_lengths

    load_finbert_tokenizer.cache_clear()
    monkeypatch.setattr(token_lengths, "_import_bert_tokenizer", lambda: _FakeBertTokenizer)
    tokenizer = load_finbert_tokenizer(DEFAULT_FINBERT_AUTHORITY)

    assert isinstance(tokenizer, _FakeBertTokenizer)
    assert captured["model_name"] == "yiyanghkust/finbert-tone"
    assert captured["do_lower_case"] is True


def test_annotate_finbert_token_lengths_uses_fixed_512_schema(monkeypatch) -> None:
    from thesis_pkg.benchmarking import token_lengths

    monkeypatch.setattr(
        token_lengths,
        "compute_finbert_token_lengths",
        lambda texts, authority: [128 if "short" in text else 300 for text in texts],
    )
    df = pl.DataFrame({"full_text": ["short text", "long text"]})
    result = annotate_finbert_token_lengths(df, DEFAULT_FINBERT_AUTHORITY)

    assert result.columns == ["full_text", "finbert_token_count_512", "finbert_token_bucket_512"]
    assert result["finbert_token_bucket_512"].to_list() == ["short", "long"]


def test_annotate_finbert_token_lengths_accepts_custom_bucket_edges(monkeypatch) -> None:
    from thesis_pkg.benchmarking import token_lengths

    monkeypatch.setattr(
        token_lengths,
        "compute_finbert_token_lengths",
        lambda texts, authority: [64 if "short" in text else 120 for text in texts],
    )
    df = pl.DataFrame({"full_text": ["short text", "medium text"]})
    result = annotate_finbert_token_lengths(
        df,
        DEFAULT_FINBERT_AUTHORITY,
        bucket_edges=BucketEdgeSpec(short_edge=64, medium_edge=128),
    )

    assert result["finbert_token_bucket_512"].to_list() == ["short", "medium"]


def test_annotate_finbert_token_lengths_in_batches_preserves_row_order(monkeypatch) -> None:
    from thesis_pkg.benchmarking import token_lengths

    monkeypatch.setattr(
        token_lengths,
        "compute_finbert_token_lengths",
        lambda texts, authority: [len(text) for text in texts],
    )
    df = pl.DataFrame({"full_text": ["a", "bb", "ccc", "dddd"]})

    result = annotate_finbert_token_lengths_in_batches(
        df,
        DEFAULT_FINBERT_AUTHORITY,
        batch_size=2,
    )

    assert result["full_text"].to_list() == ["a", "bb", "ccc", "dddd"]
    assert result["finbert_token_count_512"].to_list() == [1, 2, 3, 4]


class _WhitespaceFastTokenizer:
    is_fast = True

    def __call__(self, text, **kwargs):
        tokens: list[tuple[int, int]] = []
        cursor = 0
        for token in str(text).split():
            start = str(text).find(token, cursor)
            end = start + len(token)
            tokens.append((start, end))
            cursor = end
        add_special_tokens = kwargs.get("add_special_tokens", True)
        max_length = kwargs.get("max_length")
        if kwargs.get("truncation") and max_length is not None and add_special_tokens:
            tokens = tokens[: max_length - 2]
        input_ids = list(range(1, len(tokens) + 1))
        offsets = tokens
        special_mask = [0] * len(tokens)
        if add_special_tokens:
            input_ids = [101, *input_ids, 102]
            offsets = [(0, 0), *offsets, (0, 0)]
            special_mask = [1, *special_mask, 1]
        payload = {"input_ids": input_ids}
        if kwargs.get("return_offsets_mapping"):
            payload["offset_mapping"] = offsets
        if kwargs.get("return_special_tokens_mask"):
            payload["special_tokens_mask"] = special_mask
        return payload


class _DecodeFallbackTokenizer:
    is_fast = False

    def __call__(self, text, **kwargs):
        token_count = len(str(text).split())
        if kwargs.get("truncation") and kwargs.get("max_length") is not None:
            token_count = min(token_count, int(kwargs["max_length"]) - 2)
        input_ids = [101, *range(1, token_count + 1), 102]
        payload = {"input_ids": input_ids}
        if kwargs.get("return_special_tokens_mask"):
            payload["special_tokens_mask"] = [1, *([0] * token_count), 1]
        return payload

    def decode(self, input_ids, **_kwargs):
        return f"decoded-{len(input_ids)}-tokens"


def test_visible_prefix_fast_offsets_preserve_uncapped_and_slice_at_cap_rows() -> None:
    long_text = " ".join(f"tok{i}" for i in range(515))
    df = pl.DataFrame(
        {
            "sentence_text": ["unchanged punctuation.", long_text],
            "finbert_token_count_512": [5, 512],
        }
    )

    result = add_finbert_visible_sentence_prefixes(
        df,
        DEFAULT_FINBERT_AUTHORITY,
        tokenizer=_WhitespaceFastTokenizer(),
    )

    assert result.item(0, VISIBLE_PREFIX_TEXT_COLUMN) == "unchanged punctuation."
    assert result.item(1, VISIBLE_PREFIX_TEXT_COLUMN) == " ".join(f"tok{i}" for i in range(510))
    assert result.item(1, "visible_prefix_policy_id") == VISIBLE_PREFIX_FAST_OFFSET_POLICY_ID
    assert result.item(1, VISIBLE_PREFIX_TRUE_OVER_512_COLUMN) is True


def test_visible_prefix_decoded_fallback_records_policy() -> None:
    df = pl.DataFrame(
        {
            "sentence_text": [" ".join(f"tok{i}" for i in range(515))],
            "finbert_token_count_512": [512],
        }
    )

    result = add_finbert_visible_sentence_prefixes(
        df,
        DEFAULT_FINBERT_AUTHORITY,
        tokenizer=_DecodeFallbackTokenizer(),
    )

    assert result.item(0, VISIBLE_PREFIX_TEXT_COLUMN) == "decoded-510-tokens"
    assert result.item(0, "visible_prefix_policy_id") == VISIBLE_PREFIX_DECODE_POLICY_ID
    assert result.item(0, VISIBLE_PREFIX_TRUE_OVER_512_COLUMN) is True


def test_pyproject_benchmark_extra_declares_runtime_dependencies() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert 'benchmark = [' in pyproject
    assert '"torch"' in pyproject
    assert '"transformers"' in pyproject
    assert '"spacy"' in pyproject

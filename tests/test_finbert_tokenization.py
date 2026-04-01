from __future__ import annotations

from pathlib import Path

import polars as pl

from thesis_pkg.benchmarking.contracts import DEFAULT_FINBERT_AUTHORITY
from thesis_pkg.benchmarking.token_lengths import annotate_finbert_token_lengths
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


def test_pyproject_benchmark_extra_declares_runtime_dependencies() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert 'benchmark = [' in pyproject
    assert '"torch"' in pyproject
    assert '"transformers"' in pyproject
    assert '"spacy"' in pyproject

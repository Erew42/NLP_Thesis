from __future__ import annotations

import pytest

from thesis_native import _lm2011_rust
from thesis_pkg.benchmarking import finbert_visible_prefix
from thesis_pkg.benchmarking import item_text_cleaning
from thesis_pkg.benchmarking import sentence_split_quality_assessment as sentence_quality


def _require_rust_extension() -> None:
    if _lm2011_rust is None:
        pytest.skip("LM2011 Rust extension is not built in this environment")


def test_visible_prefix_retained_end_matches_python_for_malformed_spans() -> None:
    _require_rust_extension()

    cases = [
        ([(0, 1), (1, 4), (4, 8)], [0], 10),
        ([(-2, -1)], [0], 10),
        ([(-2, 3)], [0], 10),
    ]
    for offsets, special_mask, text_len in cases:
        assert _lm2011_rust.finbert_visible_prefix_retained_end(offsets, special_mask, text_len) == (
            finbert_visible_prefix._visible_prefix_retained_end_py(offsets, special_mask, text_len)
        )


def test_sentence_table_like_matches_python_for_financial_numeric_tokens() -> None:
    _require_rust_extension()

    text = "\n".join(
        [
            "alpha beta gamma delta epsilon zeta eta theta $1 $2",
            "alpha beta gamma delta epsilon zeta eta theta ($3)",
            "alpha beta gamma delta epsilon zeta eta theta $4",
        ]
    )
    token_count = len(text.split())

    assert _lm2011_rust.sentence_table_like(text, token_count) == sentence_quality._table_like_sentence_py(
        text,
        token_count,
    )


def test_table_support_header_matches_python_for_singular_forms() -> None:
    _require_rust_extension()

    cases = [
        "dollar in million",
        "in thousand of dollars",
        "year compared 2020 2021",
        "year through 2020 2021",
        "dollars in millions",
        "years ended 2020 2021",
    ]
    for text in cases:
        assert _lm2011_rust.cleaning_is_table_support_header_line(text) == (
            item_text_cleaning._is_table_support_header_line_py(text)
        )

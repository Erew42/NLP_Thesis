from __future__ import annotations

import pytest

from thesis_pkg.core.sec.regime import (
    build_regime_index,
    get_regime_index,
    load_regime_spec,
    normalize_form_type,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("10K", "10-K"),
        ("10-K", "10-K"),
        ("10-K405", "10-K"),
        (" 10-k ", "10-K"),
        ("10Q", "10-Q"),
        ("10-Q", "10-Q"),
        ("10-Q/A", None),
        ("10-K/A", None),
        ("10-K405/A", None),
        ("", None),
        (None, None),
    ],
)
def test_normalize_form_type(raw: str | None, expected: str | None) -> None:
    assert normalize_form_type(raw) == expected


def test_load_regime_spec_and_index_10k() -> None:
    spec = load_regime_spec("10-K")
    assert spec is not None
    index = build_regime_index(spec)
    assert index.form == "10-K"
    assert index.requires_part is False
    assert "I:1" in index.items_by_key
    assert "effective_date" in index.triggers


def test_load_regime_spec_and_index_10q() -> None:
    spec = load_regime_spec("10-Q")
    assert spec is not None
    index = build_regime_index(spec)
    assert index.form == "10-Q"
    assert index.requires_part is True
    assert "I:1" in index.items_by_key
    assert "II:1A" in index.items_by_key
    assert "1A" not in index.items_by_key
    assert "effective_date" in index.triggers


def test_get_regime_index_unknown() -> None:
    assert get_regime_index("8-K") is None
    assert get_regime_index(None) is None


def test_get_regime_index_cache_normalizes() -> None:
    index_a = get_regime_index("10-K")
    index_b = get_regime_index("10K")
    assert index_a is index_b

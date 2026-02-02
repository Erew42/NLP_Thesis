from __future__ import annotations

from thesis_pkg.core.sec.extraction import extract_filing_items


def test_10q_part_key_collision_avoidance() -> None:
    text = "\n".join(
        [
            "PART I",
            "ITEM 1. FINANCIAL STATEMENTS",
            "Some text here.",
            "PART II",
            "ITEM 1. LEGAL PROCEEDINGS",
            "More text here.",
        ]
    )
    items = extract_filing_items(
        text,
        form_type="10-Q",
        regime=False,
        repair_boundaries=False,
    )
    keys = [item.get("item") for item in items]
    assert keys == ["I:1", "II:1"]


def test_10q_missing_part_placeholder_and_regime_unknown() -> None:
    text = "\n".join(
        [
            "ITEM 1. FINANCIAL STATEMENTS",
            "Some text here.",
        ]
    )
    items = extract_filing_items(
        text,
        form_type="10-Q",
        regime=True,
        repair_boundaries=False,
    )
    assert len(items) == 1
    item = items[0]
    assert item.get("item") == "?:1"
    assert item.get("item_part") is None
    assert item.get("item_missing_part") is True
    assert item.get("exists_by_regime") is None


def test_10k_regression_item_key_unchanged() -> None:
    text = "\n".join(
        [
            "ITEM 1. BUSINESS",
            "Some text here.",
        ]
    )
    items = extract_filing_items(
        text,
        form_type="10-K",
        regime=False,
        repair_boundaries=False,
    )
    assert len(items) == 1
    item = items[0]
    assert item.get("item") == "I:1"
    assert item.get("item_part") == "I"
    assert "item_missing_part" not in item

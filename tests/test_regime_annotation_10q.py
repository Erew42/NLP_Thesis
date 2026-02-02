from __future__ import annotations

from datetime import date

from thesis_pkg.core.sec import heuristics


def _annotate(items: list[dict[str, str | bool | None]], *, form: str, filing: date) -> None:
    heuristics._annotate_items_with_regime(
        items,
        form_type=form,
        filing_date=filing,
        period_end=filing,
        enable_regime=True,
    )


def test_annotate_items_with_regime_10q_2011() -> None:
    items = [
        {"item_part": "II", "item_id": "4", "item": "II:4"},
        {"item_part": "II", "item_id": "5", "item": "II:5"},
        {"item_part": "II", "item_id": "6", "item": "II:6"},
    ]
    _annotate(items, form="10-Q", filing=date(2011, 6, 1))
    assert items[0]["canonical_item"] == "II:4_NOT_IN_FORM_RESERVED"
    assert items[1]["canonical_item"] == "II:5_OTHER_INFORMATION"
    assert items[2]["canonical_item"] == "II:6_EXHIBITS"
    assert items[0]["exists_by_regime"] is True
    assert items[1]["exists_by_regime"] is True
    assert items[2]["exists_by_regime"] is True


def test_annotate_items_with_regime_10q_2013() -> None:
    items = [
        {"item_part": "II", "item_id": "4", "item": "II:4"},
        {"item_part": "II", "item_id": "5", "item": "II:5"},
        {"item_part": "II", "item_id": "6", "item": "II:6"},
    ]
    _annotate(items, form="10-Q", filing=date(2013, 6, 1))
    assert items[0]["canonical_item"] == "II:4_MINE_SAFETY_DISCLOSURES"
    assert items[1]["canonical_item"] == "II:5_OTHER_INFORMATION"
    assert items[2]["canonical_item"] == "II:6_EXHIBITS"
    assert items[0]["exists_by_regime"] is True
    assert items[1]["exists_by_regime"] is True
    assert items[2]["exists_by_regime"] is True


def test_annotate_items_with_regime_10k_regression() -> None:
    items = [{"item_part": "I", "item_id": "1C", "item": "I:1C"}]
    heuristics._annotate_items_with_regime(
        items,
        form_type="10-K",
        filing_date=date(2024, 2, 1),
        period_end=date(2023, 12, 31),
        enable_regime=True,
    )
    assert items[0]["canonical_item"] == "I:1C_CYBERSECURITY"
    assert items[0]["exists_by_regime"] is True

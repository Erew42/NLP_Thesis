from __future__ import annotations

from thesis_pkg.core.sec.suspicious_boundary_diagnostics import _expected_part_for_item


def test_expected_part_for_item_10q_missing_part() -> None:
    item = {"item": "?:1", "item_id": "1", "item_part": None}
    assert _expected_part_for_item(item, normalized_form="10-Q") is None


def test_expected_part_for_item_10q_explicit_part() -> None:
    item = {"item": "II:1", "canonical_item": "II:1_LEGAL_PROCEEDINGS"}
    assert _expected_part_for_item(item, normalized_form="10-Q") == "II"


def test_expected_part_for_item_10k_fallback() -> None:
    item = {"item_id": "1", "item_part": None}
    assert _expected_part_for_item(item, normalized_form="10-K") == "I"

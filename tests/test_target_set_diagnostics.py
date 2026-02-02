from __future__ import annotations

from datetime import date

from thesis_pkg.core.sec.suspicious_boundary_diagnostics import (
    _expected_canonical_items,
    _item_counts_for_target,
)


def test_target_set_excludes_non_target_warn() -> None:
    canonical = "II:2_CHANGES_IN_SECURITIES_AND_USE_OF_PROCEEDS"
    assert (
        _item_counts_for_target(
            canonical,
            normalized_form="10-Q",
            target_set="cohen2020_common",
        )
        is False
    )


def test_missing_items_regime_driven_with_target_set() -> None:
    filing_date = date(2024, 2, 1)
    period_end = date(2023, 12, 31)
    expected_core = _expected_canonical_items(
        normalized_form="10-K",
        filing_date=filing_date,
        period_end=period_end,
        target_set=None,
    )
    expected_target = _expected_canonical_items(
        normalized_form="10-K",
        filing_date=filing_date,
        period_end=period_end,
        target_set="cohen2020_common",
    )
    assert "I:1A_RISK_FACTORS" in expected_core
    assert "II:7_MDA" in expected_target
    missing_core = expected_core - {"I:1A_RISK_FACTORS"}
    missing_target = expected_target - {"I:1A_RISK_FACTORS"}
    assert missing_core != missing_target


def test_expected_canonical_excludes_not_in_form() -> None:
    expected = _expected_canonical_items(
        normalized_form="10-Q",
        filing_date=date(1995, 1, 1),
        period_end=None,
        target_set=None,
    )
    assert "I:3_NOT_IN_FORM_PRE_EFFECTIVE" not in expected

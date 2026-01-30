from __future__ import annotations

from thesis_pkg.filing_text import extract_filing_items


def _get_item(items: list[dict[str, str | bool | None]], item_id: str) -> dict[str, str | bool | None]:
    for item in items:
        if item.get("item_id") == item_id:
            return item
    raise AssertionError(f"Missing item {item_id} in extraction output.")


def test_start_selection_prefers_real_header_over_toc() -> None:
    filler = "\n".join(f"FILLER {i}" for i in range(50))
    text = f"""TABLE OF CONTENTS
ITEM 15.............55
PART I
ITEM 1. Business
ITEM 1A. Risk Factors

{filler}

ITEM 15. Exhibits and Financial Statement Schedules.
This section covers exhibits.
More detail about exhibits and schedules.
Another sentence with exhibits context.
Final line of prose to anchor the section.
"""
    items = extract_filing_items(text, form_type="10-K", diagnostics=True)
    item15 = _get_item(items, "15")

    heading_raw = str(item15.get("_heading_line_raw") or "")
    assert "55" not in heading_raw
    assert item15.get("_start_candidates_total", 0) >= 2
    assert str(item15.get("full_text") or "").lstrip().startswith(
        "Exhibits and Financial Statement Schedules"
    )


def test_part_header_false_positive_avoidance() -> None:
    text = """PART II
ITEM 7. Management's Discussion and Analysis.
We comply with Part IV of the Act when necessary.
Additional discussion continues here.

PART IV
ITEM 15. Exhibits.
Exhibits details follow.
"""
    items = extract_filing_items(text, form_type="10-K")
    item7 = _get_item(items, "7")

    item7_text = str(item7.get("full_text") or "")
    assert "Part IV of the Act" in item7_text
    assert "\nPART IV\n" not in item7_text
    assert "ITEM 15" not in item7_text


def test_reserved_stub_truncates_and_preserves_next_item() -> None:
    text = """PART II
ITEM 6. [Reserved]
ITEM 7. Management's Discussion and Analysis.
More text follows.
"""
    items = extract_filing_items(text, form_type="10-K")
    item6 = _get_item(items, "6")

    item6_text = str(item6.get("full_text") or "")
    assert "reserved" in item6_text.lower()
    assert "ITEM 7" not in item6_text

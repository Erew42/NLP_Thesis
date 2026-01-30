from __future__ import annotations

import json
from pathlib import Path

from thesis_pkg.core.sec import embedded_headings
from thesis_pkg.filing_text import extract_filing_items


def _get_item(items: list[dict[str, str | bool | None]], item_id: str) -> dict[str, str | bool | None]:
    for item in items:
        if item.get("item_id") == item_id:
            return item
    raise AssertionError(f"Missing item {item_id} in extraction output.")


def test_detect_gij_context_parses_omit_block() -> None:
    text = """Intro text.
The following Items have been omitted in accordance with General Instruction J to Form 10-K
Item 1A. Risk Factors
Item 7A. Quantitative and Qualitative Disclosures About Market Risk
Item 9C. Disclosure Regarding Foreign Jurisdictions
Item 1112(b) Asset-Level Disclosures
Substitute information provided in accordance with General Instruction J to Form 10-K
PART I
ITEM 1. Business
"""
    context = embedded_headings.detect_gij_context(text.splitlines())

    assert context["gij_asset_backed"] is True
    assert context["gij_reason"] == "General Instruction J"
    assert context["gij_omit_ranges"]
    assert context["gij_omitted_items"] == {"1A", "7A", "9C"}


def test_item_run_line_rejected_over_real_header() -> None:
    filler = "\n".join(f"Filler prose line {i}." for i in range(1, 22))
    text = f"""ITEM 5. Item 7.
{filler}
ITEM 7. Management's Discussion and Analysis.
ITEM 7A. Quantitative and Qualitative Disclosures.
ITEM 8. Financial Statements and Supplementary Data.
More detail here.
"""
    items = extract_filing_items(text, form_type="10-K", diagnostics=True)
    item7 = _get_item(items, "7")

    heading_raw = str(item7.get("_heading_line_raw") or "")
    assert "Item 5. Item 7." not in heading_raw
    assert heading_raw.strip().startswith("ITEM 7.")


def test_empty_stub_truncates_item_at_stub_line() -> None:
    text = """ITEM 4. Mine Safety Disclosures.
Not applicable.
Additional filler line that should be excluded.
ITEM 5. Market for Registrant's Common Equity.
More text follows.
"""
    items = extract_filing_items(text, form_type="10-K")
    item4 = _get_item(items, "4")

    item_text = str(item4.get("full_text") or "")
    assert "Additional filler line" not in item_text
    assert "ITEM 5" not in item_text
    assert item_text.strip().endswith("Not applicable.")


def test_item_16_regime_present_but_extraction_skips() -> None:
    spec_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "thesis_pkg"
        / "core"
        / "sec"
        / "item_regime_10k.json"
    )
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    items = spec.get("items", {})

    assert any(entry.get("item_id") == "16" for entry in items.values())

    text = """ITEM 16. Form 10-K Summary.
Optional summary content.
ITEM 15. Exhibits.
Exhibits text.
"""
    extracted = extract_filing_items(text, form_type="10-K")
    assert all(item.get("item_id") != "16" for item in extracted)

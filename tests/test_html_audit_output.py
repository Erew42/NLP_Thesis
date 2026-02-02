from __future__ import annotations

from pathlib import Path

from thesis_pkg.core.sec.suspicious_boundary_diagnostics import write_html_audit


def test_write_html_audit_creates_files(tmp_path: Path) -> None:
    doc_id = "0000000001:000000000123456789"
    accession = "000000000123456789"
    index_rows = [
        {
            "doc_id": doc_id,
            "accession": accession,
            "cik": "0000000001",
            "form": "10-K",
            "filing_date": "2023-01-01",
            "period_end": "2022-12-31",
            "n_items_extracted": "1",
            "items_extracted": "1",
            "missing_core_items": "",
            "any_warn": "True",
            "any_fail": "False",
            "filing_exclusion_reason": "",
        }
    ]
    items_by_filing = {
        doc_id: [
            {
                "item_part": "I",
                "item_id": "1",
                "item": "Business",
                "item_status": "ok",
                "length_chars": "100",
                "heading_start": "0",
                "heading_end": "10",
                "content_start": "10",
                "content_end": "110",
                "heading_line_clean": "ITEM 1. BUSINESS",
                "heading_line_raw": "ITEM 1. BUSINESS",
                "embedded_heading_warn": "False",
                "embedded_heading_fail": "False",
                "first_embedded_kind": "",
                "first_embedded_classification": "",
                "first_embedded_item_id": "",
                "first_embedded_part": "",
                "first_embedded_line_idx": "",
                "first_embedded_char_pos": "",
                "first_embedded_snippet": "",
                "first_fail_kind": "",
                "first_fail_classification": "",
                "first_fail_item_id": "",
                "first_fail_part": "",
                "first_fail_line_idx": "",
                "first_fail_char_pos": "",
                "first_fail_snippet": "",
                "doc_head_200": "HEAD",
                "doc_tail_200": "TAIL",
            }
        ]
    }
    metadata = {
        "pass_definition": "any_fail == False and filing_exclusion_reason is empty",
        "total_filings": 1,
        "total_items": 1,
        "total_pass_filings": 1,
        "sample_size": 1,
        "generated_at": "2026-01-30T12:00:00",
        "offset_basis": "extractor_body",
    }

    out_dir = tmp_path / "html_audit"
    write_html_audit(
        index_rows=index_rows,
        items_by_filing=items_by_filing,
        out_dir=out_dir,
        scope_label="sample (1)",
        metadata=metadata,
    )

    index_path = out_dir / "index.html"
    filings_dir = out_dir / "filings"
    assert index_path.exists()
    assert filings_dir.exists()

    filing_files = list(filings_dir.glob("*.html"))
    assert len(filing_files) == 1
    index_text = index_path.read_text(encoding="utf-8")
    assert doc_id in index_text
    assert "SEC Item Extraction Manual Review" in index_text
    assert "10-K Extraction Manual Review" not in index_text
    assert "Form: 10-K" in index_text

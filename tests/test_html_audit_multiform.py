from __future__ import annotations

from pathlib import Path

from thesis_pkg.core.sec.html_audit import (
    write_html_audit,
    write_html_audit_root_index,
)


def _metadata(total_filings: int) -> dict[str, object]:
    return {
        "pass_definition": "any_fail == False and filing_exclusion_reason is empty",
        "total_filings": total_filings,
        "total_items": total_filings,
        "total_pass_filings": total_filings,
        "sample_size": total_filings,
        "generated_at": "2026-02-02T12:00:00",
        "offset_basis": "extractor_body",
    }


def test_html_audit_root_index_and_forms(tmp_path: Path) -> None:
    out_dir = tmp_path / "html_audit"
    doc_k = "0000000001:000000000123456789"
    doc_q = "0000000002:000000000223456789"
    index_rows_k = [
        {
            "doc_id": doc_k,
            "accession": "000000000123456789",
            "cik": "0000000001",
            "form": "10-K",
            "filing_date": "2023-01-01",
            "period_end": "2022-12-31",
            "n_items_extracted": "1",
            "items_extracted": "1",
            "missing_core_items": "",
            "any_warn": "False",
            "any_fail": "False",
            "filing_exclusion_reason": "",
        }
    ]
    index_rows_q = [
        {
            "doc_id": doc_q,
            "accession": "000000000223456789",
            "cik": "0000000002",
            "form": "10-Q",
            "filing_date": "2023-01-01",
            "period_end": "2022-12-31",
            "n_items_extracted": "1",
            "items_extracted": "I:1",
            "missing_core_items": "",
            "any_warn": "False",
            "any_fail": "False",
            "filing_exclusion_reason": "",
        }
    ]
    items_by_filing_k = {doc_k: [{"item_part": "I", "item_id": "1", "item": "I:1"}]}
    items_by_filing_q = {doc_q: [{"item_part": "I", "item_id": "1", "item": "I:1"}]}

    write_html_audit(
        index_rows=index_rows_k,
        items_by_filing=items_by_filing_k,
        out_dir=out_dir / "10-K",
        scope_label="sample (1)",
        metadata=_metadata(total_filings=1),
    )
    write_html_audit(
        index_rows=index_rows_q,
        items_by_filing=items_by_filing_q,
        out_dir=out_dir / "10-Q",
        scope_label="sample (1)",
        metadata=_metadata(total_filings=1),
    )

    write_html_audit_root_index(
        form_entries=[
            {
                "form": "10-K",
                "index_path": "10-K/index.html",
                "total_filings": 1,
                "pass_count": 1,
                "warn_count": 0,
                "fail_count": 0,
                "scope": "sample (1)",
            },
            {
                "form": "10-Q",
                "index_path": "10-Q/index.html",
                "total_filings": 1,
                "pass_count": 1,
                "warn_count": 0,
                "fail_count": 0,
                "scope": "sample (1)",
            },
        ],
        out_dir=out_dir,
        metadata=_metadata(total_filings=2),
    )

    assert (out_dir / "index.html").exists()
    assert (out_dir / "10-K" / "index.html").exists()
    assert (out_dir / "10-Q" / "index.html").exists()
    index_text = (out_dir / "index.html").read_text(encoding="utf-8")
    assert "10-K/index.html" in index_text
    assert "10-Q/index.html" in index_text


def test_non_target_item_still_rendered(tmp_path: Path) -> None:
    out_dir = tmp_path / "html_audit"
    doc_id = "0000000003:000000000323456789"
    index_rows = [
        {
            "doc_id": doc_id,
            "accession": "000000000323456789",
            "cik": "0000000003",
            "form": "10-Q",
            "filing_date": "2023-01-01",
            "period_end": "2022-12-31",
            "n_items_extracted": "1",
            "items_extracted": "II:2",
            "missing_core_items": "",
            "missing_expected_canonicals": "",
            "any_warn": "False",
            "any_fail": "False",
            "filing_exclusion_reason": "",
        }
    ]
    items_by_filing = {
        doc_id: [
            {
                "item_part": "II",
                "item_id": "2",
                "item": "II:2",
                "counts_for_target": False,
            }
        ]
    }
    write_html_audit(
        index_rows=index_rows,
        items_by_filing=items_by_filing,
        out_dir=out_dir / "10-Q",
        scope_label="sample (1)",
        metadata=_metadata(total_filings=1),
    )
    filing_path = out_dir / "10-Q" / "filings"
    html_files = list(filing_path.glob("*.html"))
    assert html_files
    html_text = html_files[0].read_text(encoding="utf-8")
    assert "II:2" in html_text
    assert "PASS" in html_text

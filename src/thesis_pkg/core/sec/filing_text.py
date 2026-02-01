from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from .extraction import HEADER_SEARCH_LIMIT_DEFAULT, extract_filing_items, parse_header
from .patterns import (
    ACC_HEADER_PATTERN,
    CIK_HEADER_PATTERN,
    CONTINUED_PATTERN,
    DATE_HEADER_PATTERN,
    PERIOD_END_HEADER_PATTERN,
    EMPTY_ITEM_PATTERN,
    FILENAME_PATTERN,
    ITEM_CANDIDATE_PATTERN,
    ITEM_LINESTART_PATTERN,
    ITEM_WORD_PATTERN,
    PAGE_HYPHEN_PATTERN,
    PAGE_NUMBER_PATTERN,
    PAGE_OF_PATTERN,
    PAGE_ROMAN_PATTERN,
    PART_LINESTART_PATTERN,
    PART_MARKER_PATTERN,
    TOC_HEADER_LINE_PATTERN,
    TOC_MARKER_PATTERN,
)
from .utilities import _cik_10, _digits_only, _make_doc_id


def parse_filename_minimal(filename: str) -> dict:
    """
    Parse all usable info from filename only.
    Returns dict with parse_ok plus components.
    """
    m = FILENAME_PATTERN.match(filename)
    if not m:
        return {
            "filename_parse_ok": False,
            "file_date_filename": None,
            "document_type_filename": None,
            "cik": None,
            "cik_10": None,
            "accession_number": None,
            "accession_nodash": None,
            "doc_id": None,
        }

    date_str = m.group(1)
    doc_type = m.group(2)
    try:
        cik_int = int(m.group(3))
    except Exception:
        cik_int = None
    acc = m.group(4)

    cik10 = _cik_10(cik_int)
    acc_nodash = _digits_only(acc)
    doc_id = _make_doc_id(cik10, acc)

    return {
        "filename_parse_ok": True,
        "file_date_filename": date_str,          # parse to Date at write time
        "document_type_filename": doc_type,
        "cik": cik_int,
        "cik_10": cik10,
        "accession_number": acc,
        "accession_nodash": acc_nodash,
        "doc_id": doc_id,
    }


@dataclass
class RawTextSchema:
    schema = {
        "doc_id": pl.Utf8,
        "cik": pl.Int64,
        "cik_10": pl.Utf8,
        "accession_number": pl.Utf8,
        "accession_nodash": pl.Utf8,
        "file_date_filename": pl.Utf8,            # will cast to Date on write
        "document_type_filename": pl.Utf8,
        "filename": pl.Utf8,
        "zip_member_path": pl.Utf8,
        "filename_parse_ok": pl.Boolean,
        "full_text": pl.Utf8,
    }


@dataclass
class ParsedFilingSchema:
    schema = {
        "doc_id": pl.Utf8,
        "cik": pl.Int64,
        "cik_10": pl.Utf8,
        "accession_number": pl.Utf8,
        "accession_nodash": pl.Utf8,
        "filing_date": pl.Utf8,  # will cast to Date on write
        "filing_date_header": pl.Utf8,
        "period_end": pl.Utf8,  # will cast to Date on write
        "period_end_header": pl.Utf8,
        "file_date_filename": pl.Utf8,  # will cast to Date on write
        "document_type_filename": pl.Utf8,
        "filename": pl.Utf8,
        "zip_member_path": pl.Utf8,
        "filename_parse_ok": pl.Boolean,
        "cik_header_primary": pl.Utf8,
        "ciks_header_secondary": pl.List(pl.Utf8),
        "accession_header": pl.Utf8,
        "cik_conflict": pl.Boolean,
        "accession_conflict": pl.Boolean,
        "full_text": pl.Utf8,
    }


@dataclass
class FilingItemSchema:
    schema = {
        "doc_id": pl.Utf8,
        "cik": pl.Int64,
        "cik_10": pl.Utf8,
        "accession_number": pl.Utf8,
        "accession_nodash": pl.Utf8,
        "file_date_filename": pl.Date,
        "filing_date": pl.Date,
        "period_end": pl.Date,
        "document_type_filename": pl.Utf8,
        "filename": pl.Utf8,
        "item_part": pl.Utf8,
        "item_id": pl.Utf8,
        "item": pl.Utf8,
        "canonical_item": pl.Utf8,
        "exists_by_regime": pl.Boolean,
        "item_status": pl.Utf8,
        "full_text": pl.Utf8,
    }

from __future__ import annotations

import datetime as dt

import polars as pl

from thesis_pkg.core.sec.lm2011_cleaning import _apply_lm2011_paper_cleaning
from thesis_pkg.core.sec.lm2011_cleaning import clean_full_10k_for_lm2011
from thesis_pkg.core.sec.lm2011_cleaning import detect_exhibit_tail_start
from thesis_pkg.core.sec.lm2011_text import build_lm2011_text_features_full_10k


def _body(word: str = "body", count: int = 800) -> str:
    return "ITEM 1. Business\n" + " ".join([word] * count)


def _dictionary_lists() -> dict[str, list[str]]:
    return {
        "negative": ["loss"],
        "positive": ["gain"],
        "uncertainty": ["uncertain"],
        "litigious": ["lawsuit"],
        "modal_strong": ["must"],
        "modal_weak": ["may"],
    }


def _master_dictionary_words() -> list[str]:
    return ["loss", "gain", "uncertain", "lawsuit", "must", "may", "bad"]


def test_toc_signatures_only_does_not_trigger_truncation() -> None:
    text = "\n".join(
        [
            "TABLE OF CONTENTS",
            "SIGNATURES",
            "Item 14. Exhibits and Reports on Form 8-K",
            "Item 15. Exhibits and Financial Statement Schedules",
            _body(),
        ]
    )

    cleaned_text, decision = _apply_lm2011_paper_cleaning(text)

    assert cleaned_text == text
    assert decision.reason == "no_tail_anchor"
    assert detect_exhibit_tail_start(text) is None


def test_exhibit_index_tail_truncates_from_tail_anchor() -> None:
    text = "\n".join(
        [
            _body(),
            "EXHIBIT INDEX",
            "EXHIBIT 21",
            "Subsidiaries of the Registrant",
        ]
    )

    cleaned_text, decision = _apply_lm2011_paper_cleaning(text)

    assert cleaned_text is not None
    assert "EXHIBIT INDEX" not in cleaned_text
    assert decision.reason == "strong_anchor_exhibit_index"
    assert decision.start is not None


def test_ex_tag_tail_truncates_from_first_tail_ex_tag() -> None:
    text = "\n".join(
        [
            _body(),
            "</EX-13>",
            "<EX-21>",
            "Subsidiaries of the Registrant",
        ]
    )

    cleaned_text, decision = _apply_lm2011_paper_cleaning(text)

    assert cleaned_text is not None
    assert "<EX-21>" not in cleaned_text
    assert decision.reason == "strong_anchor_ex_tag"


def test_weak_exhibit_heading_cluster_truncates_when_tail_cluster_exists() -> None:
    text = "\n".join(
        [
            _body(),
            "EXHIBIT 21",
            "Subsidiaries of the Registrant",
            "EXHIBIT 23.1",
            "Consent of Independent Registered Public Accounting Firm",
        ]
    )

    cleaned_text, decision = _apply_lm2011_paper_cleaning(text)

    assert cleaned_text is not None
    assert "EXHIBIT 21" not in cleaned_text
    assert decision.reason == "weak_anchor_cluster"


def test_mid_body_exhibit_reference_does_not_trigger_truncation() -> None:
    text = "\n".join(
        [
            _body(),
            "See Exhibit 21 for a list of subsidiaries.",
            "Further operating discussion continues here.",
        ]
    )

    cleaned_text, decision = _apply_lm2011_paper_cleaning(text)

    assert cleaned_text == text
    assert decision.reason == "no_tail_anchor"


def test_early_exhibit_index_is_rejected_before_tail_region() -> None:
    text = "\n".join(
        [
            "EXHIBIT INDEX",
            "EXHIBIT 21",
            "Table of subsidiaries",
            _body(count=1400),
        ]
    )

    cleaned_text, decision = _apply_lm2011_paper_cleaning(text)

    assert cleaned_text == text
    assert decision.reason == "no_tail_anchor"


def test_no_anchor_only_strips_edgar_metadata() -> None:
    text = "<SEC-HEADER>header</SEC-HEADER>\n" + _body()

    cleaned_text = clean_full_10k_for_lm2011(text, contract="lm2011_paper")

    assert cleaned_text is not None
    assert "<SEC-HEADER>" not in cleaned_text
    assert cleaned_text.lstrip().startswith("ITEM 1. Business")


def test_full_text_feature_builder_respects_lm2011_paper_cleaning_contract() -> None:
    sec_parsed = pl.DataFrame(
        {
            "doc_id": ["d1"],
            "cik_10": ["0001"],
            "filing_date": [dt.date(2023, 1, 1)],
            "document_type_filename": ["10-K"],
            "full_text": [
                "\n".join(
                    [
                        _body("loss", 900),
                        "EXHIBIT INDEX",
                        "EXHIBIT 21",
                        "loss loss loss loss",
                    ]
                )
            ],
        }
    )

    current_df = build_lm2011_text_features_full_10k(
        sec_parsed.lazy(),
        dictionary_lists=_dictionary_lists(),
        harvard_negative_word_list=["bad"],
        master_dictionary_words=_master_dictionary_words(),
        cleaning_contract="current",
    ).collect()
    paper_df = build_lm2011_text_features_full_10k(
        sec_parsed.lazy(),
        dictionary_lists=_dictionary_lists(),
        harvard_negative_word_list=["bad"],
        master_dictionary_words=_master_dictionary_words(),
        cleaning_contract="lm2011_paper",
    ).collect()

    assert current_df.item(0, "token_count_full_10k") > paper_df.item(0, "token_count_full_10k")

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
import math
import re

import polars as pl

from thesis_pkg.core.ccm.lm2011 import normalize_lm2011_form_value
from thesis_pkg.core.sec.lm2011_cleaning import Full10KCleaningContract
from thesis_pkg.core.sec.lm2011_cleaning import clean_full_10k_for_lm2011


DEFAULT_TEXT_FEATURE_BATCH_SIZE = 1000
LM2011_DICTIONARY_REQUIRED_LISTS: tuple[str, ...] = (
    "negative",
    "positive",
    "uncertainty",
    "litigious",
    "modal_strong",
    "modal_weak",
)
_LINEBREAK_HYPHEN_RE = re.compile(r"([A-Za-z])-\s*(?:\r?\n)\s*([A-Za-z])")
_TOKEN_RE = re.compile(r"[A-Za-z]{2,}(?:[-'][A-Za-z]+)*")
RAW_ITEM_TEXT_CLEANING_POLICY_ID = "raw_item_text"


def _require_columns(lf: pl.LazyFrame, required: tuple[str, ...], label: str) -> None:
    schema = lf.collect_schema()
    missing = [name for name in required if name not in schema]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _normalize_dictionary_tokens(values: Iterable[str] | None) -> frozenset[str]:
    if values is None:
        return frozenset()
    tokens = {
        token
        for value in values
        if value is not None
        for token in tokenize_lm2011_text(str(value))
    }
    return frozenset(tokens)


def normalize_lm2011_dictionary_lists(
    dictionary_lists: Mapping[str, Iterable[str]],
) -> dict[str, frozenset[str]]:
    missing = [name for name in LM2011_DICTIONARY_REQUIRED_LISTS if name not in dictionary_lists]
    if missing:
        raise ValueError(f"dictionary_lists missing required categories: {missing}")
    return {
        name: _normalize_dictionary_tokens(dictionary_lists[name])
        for name in LM2011_DICTIONARY_REQUIRED_LISTS
    }


def tokenize_lm2011_text(text: str | None) -> list[str]:
    if text is None:
        return []
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _LINEBREAK_HYPHEN_RE.sub(r"\1\2", normalized)
    return [token.casefold() for token in _TOKEN_RE.findall(normalized)]


def _normalize_master_dictionary_words(master_dictionary_words: Iterable[str] | None) -> frozenset[str]:
    tokens = _normalize_dictionary_tokens(master_dictionary_words)
    if not tokens:
        raise ValueError("master_dictionary_words is required and must contain at least one token")
    return tokens


def _lm2011_inverse_document_frequency(num_docs: int, document_frequency: int) -> float:
    if num_docs < 1 or document_frequency < 1:
        return 0.0
    return math.log(float(num_docs) / float(document_frequency))


def _lm2011_term_weight(
    *,
    term_frequency: int,
    document_length: int,
    inverse_document_frequency: float,
) -> float:
    if term_frequency < 1 or document_length < 1:
        return 0.0
    return (
        (1.0 + math.log(float(term_frequency)))
        / (1.0 + math.log(float(document_length)))
    ) * inverse_document_frequency


def _ensure_normalized_form(df: pl.DataFrame, *, raw_form_col: str) -> pl.DataFrame:
    if "normalized_form" in df.columns:
        return df.with_columns(pl.col("normalized_form").cast(pl.Utf8, strict=False))
    return df.with_columns(
        pl.col(raw_form_col)
        .map_elements(normalize_lm2011_form_value, return_dtype=pl.Utf8)
        .alias("normalized_form")
    )


def _validated_batch_size(batch_size: int) -> int:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    return batch_size


def _iter_text_base_batches(
    lf: pl.LazyFrame,
    *,
    label: str,
    text_col: str,
    raw_form_col: str,
    include_item_id: bool = False,
    required_item_id: str | None = None,
    batch_size: int = DEFAULT_TEXT_FEATURE_BATCH_SIZE,
) -> Iterable[pl.DataFrame]:
    required = ["doc_id", "cik_10", "filing_date", text_col, raw_form_col]
    if include_item_id:
        required.append("item_id")
    _require_columns(lf, tuple(required), label)
    if not hasattr(lf, "collect_batches"):
        raise RuntimeError(
            "Polars LazyFrame.collect_batches is required for memory-safe LM2011 text scoring. "
            "Upgrade Polars to a version that provides collect_batches."
        )
    batch_size = _validated_batch_size(batch_size)

    base = lf
    if required_item_id is not None:
        base = base.filter(pl.col("item_id").cast(pl.Utf8, strict=False) == pl.lit(required_item_id))

    select_cols = ["doc_id", "cik_10", "filing_date", raw_form_col]
    if include_item_id:
        select_cols.append("item_id")
    select_cols.append(text_col)

    schema = base.collect_schema()
    if "normalized_form" in schema:
        select_cols.append("normalized_form")
    for batch in base.select(*select_cols).collect_batches(chunk_size=batch_size):
        if batch.height:
            yield _ensure_normalized_form(batch, raw_form_col=raw_form_col)


def _prepare_document_stats(
    batches: Iterable[pl.DataFrame],
    *,
    text_col: str,
    vocabulary: frozenset[str],
    master_dictionary_words: frozenset[str],
    text_cleaner: Callable[[str | None], str | None] | None = None,
) -> tuple[list[dict[str, object]], dict[str, Counter[str]], dict[str, int], dict[str, int], dict[str, float]]:
    base_rows: list[dict[str, object]] = []
    doc_token_counts: dict[str, Counter[str]] = {}
    doc_token_totals: dict[str, int] = {}
    doc_recognized_word_totals: dict[str, int] = {}
    document_frequency: Counter[str] = Counter()

    for batch in batches:
        for row in batch.iter_rows(named=True):
            row_dict = dict(row)
            doc_id = str(row_dict["doc_id"])
            text_value = row_dict.pop(text_col, None)
            text_input = text_value if isinstance(text_value, str) else None
            if text_cleaner is not None:
                text_input = text_cleaner(text_input)
            tokens = tokenize_lm2011_text(text_input)
            token_total = len(tokens)
            recognized_word_total = sum(1 for token in tokens if token in master_dictionary_words)
            counts = Counter(token for token in tokens if token in vocabulary)
            base_rows.append(row_dict)
            doc_token_counts[doc_id] = counts
            doc_token_totals[doc_id] = token_total
            doc_recognized_word_totals[doc_id] = recognized_word_total
            document_frequency.update(counts.keys())

    num_docs = max(len(base_rows), 1)
    idf_by_token = {
        token: _lm2011_inverse_document_frequency(num_docs, doc_freq)
        for token, doc_freq in document_frequency.items()
    }
    return base_rows, doc_token_counts, doc_token_totals, doc_recognized_word_totals, idf_by_token


def _build_feature_rows(
    base_rows: list[dict[str, object]],
    *,
    doc_token_counts: Mapping[str, Counter[str]],
    doc_token_totals: Mapping[str, int],
    doc_recognized_word_totals: Mapping[str, int],
    idf_by_token: Mapping[str, float],
    token_count_col: str,
    total_token_count_col: str,
    signal_specs: tuple[tuple[str, frozenset[str], bool], ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in base_rows:
        out = dict(row)
        doc_id = str(out["doc_id"])
        token_total = int(doc_token_totals.get(doc_id, 0))
        recognized_word_total = int(doc_recognized_word_totals.get(doc_id, 0))
        counts = doc_token_counts.get(doc_id, Counter())
        denominator = float(recognized_word_total) if recognized_word_total > 0 else None
        out[total_token_count_col] = token_total
        out[token_count_col] = recognized_word_total
        for signal_stem, signal_tokens, include_tfidf in signal_specs:
            matched_count = float(sum(counts.get(token, 0) for token in signal_tokens))
            out[f"{signal_stem}_prop"] = (matched_count / denominator) if denominator else None
            if include_tfidf:
                out[f"{signal_stem}_tfidf"] = (
                    float(
                        sum(
                            _lm2011_term_weight(
                                term_frequency=counts.get(token, 0),
                                document_length=recognized_word_total,
                                inverse_document_frequency=idf_by_token.get(token, 0.0),
                            )
                            for token in signal_tokens
                            if counts.get(token, 0) > 0
                        )
                    )
                    if denominator
                    else None
                )
        rows.append(out)
    return rows


def _feature_schema(
    *,
    include_item_id: bool,
    include_cleaning_policy_id: bool,
    token_count_col: str,
    total_token_count_col: str,
    signal_specs: tuple[tuple[str, frozenset[str], bool], ...],
) -> dict[str, pl.DataType]:
    schema: dict[str, pl.DataType] = {
        "doc_id": pl.Utf8,
        "cik_10": pl.Utf8,
        "filing_date": pl.Date,
        "normalized_form": pl.Utf8,
        total_token_count_col: pl.Int32,
        token_count_col: pl.Int32,
    }
    if include_item_id:
        schema["item_id"] = pl.Utf8
    if include_cleaning_policy_id:
        schema["cleaning_policy_id"] = pl.Utf8
    for signal_stem, _, include_tfidf in signal_specs:
        schema[f"{signal_stem}_prop"] = pl.Float64
        if include_tfidf:
            schema[f"{signal_stem}_tfidf"] = pl.Float64
    return schema


def _build_scored_text_frame(
    batches: Iterable[pl.DataFrame],
    *,
    text_col: str,
    token_count_col: str,
    total_token_count_col: str,
    include_item_id: bool,
    cleaning_policy_id: str | None = None,
    signal_specs: tuple[tuple[str, frozenset[str], bool], ...],
    master_dictionary_words: Iterable[str],
    text_cleaner: Callable[[str | None], str | None] | None = None,
) -> pl.LazyFrame:
    vocabulary = frozenset().union(*(tokens for _, tokens, _ in signal_specs))
    normalized_master_dictionary_words = _normalize_master_dictionary_words(master_dictionary_words)
    base_rows, doc_token_counts, doc_token_totals, doc_recognized_word_totals, idf_by_token = _prepare_document_stats(
        batches,
        text_col=text_col,
        vocabulary=vocabulary,
        master_dictionary_words=normalized_master_dictionary_words,
        text_cleaner=text_cleaner,
    )
    rows = _build_feature_rows(
        base_rows,
        doc_token_counts=doc_token_counts,
        doc_token_totals=doc_token_totals,
        doc_recognized_word_totals=doc_recognized_word_totals,
        idf_by_token=idf_by_token,
        token_count_col=token_count_col,
        total_token_count_col=total_token_count_col,
        signal_specs=signal_specs,
    )
    schema = _feature_schema(
        include_item_id=include_item_id,
        include_cleaning_policy_id=cleaning_policy_id is not None,
        token_count_col=token_count_col,
        total_token_count_col=total_token_count_col,
        signal_specs=signal_specs,
    )
    df = (
        pl.DataFrame(rows, schema_overrides=schema)
        .with_columns(
            pl.col(total_token_count_col).cast(pl.Int32, strict=False),
            pl.col(token_count_col).cast(pl.Int32, strict=False),
        )
    )
    if cleaning_policy_id is not None:
        df = df.with_columns(pl.lit(cleaning_policy_id, dtype=pl.Utf8).alias("cleaning_policy_id"))
    return df.lazy()


def _build_lm2011_signal_specs(
    *,
    normalized_dict: Mapping[str, frozenset[str]],
    harvard_negative_word_list: Iterable[str] | None,
) -> tuple[tuple[str, frozenset[str], bool], ...]:
    harvard_tokens = _normalize_dictionary_tokens(harvard_negative_word_list)
    if not harvard_tokens:
        raise ValueError("harvard_negative_word_list is required and must contain at least one token")
    return (
        ("h4n_inf", harvard_tokens, True),
        *(
            (
                f"lm_{category}",
                normalized_dict[category],
                True,
            )
            for category in LM2011_DICTIONARY_REQUIRED_LISTS
        ),
    )


def build_lm2011_text_features_full_10k(
    sec_parsed_lf: pl.LazyFrame,
    *,
    dictionary_lists: Mapping[str, Iterable[str]],
    harvard_negative_word_list: Iterable[str] | None,
    master_dictionary_words: Iterable[str],
    text_col: str = "full_text",
    raw_form_col: str = "document_type_filename",
    cleaning_contract: Full10KCleaningContract = "current",
    batch_size: int = DEFAULT_TEXT_FEATURE_BATCH_SIZE,
) -> pl.LazyFrame:
    normalized_dict = normalize_lm2011_dictionary_lists(dictionary_lists)
    signal_specs = _build_lm2011_signal_specs(
        normalized_dict=normalized_dict,
        harvard_negative_word_list=harvard_negative_word_list,
    )
    batches = _iter_text_base_batches(
        sec_parsed_lf,
        label="sec_parsed",
        text_col=text_col,
        raw_form_col=raw_form_col,
        batch_size=batch_size,
    )
    return _build_scored_text_frame(
        batches,
        text_col=text_col,
        token_count_col="token_count_full_10k",
        total_token_count_col="total_token_count_full_10k",
        include_item_id=False,
        signal_specs=signal_specs,
        master_dictionary_words=master_dictionary_words,
        text_cleaner=lambda value: clean_full_10k_for_lm2011(value, contract=cleaning_contract),
    )


def build_lm2011_text_features_mda(
    sec_item_lf: pl.LazyFrame,
    *,
    dictionary_lists: Mapping[str, Iterable[str]],
    harvard_negative_word_list: Iterable[str] | None,
    master_dictionary_words: Iterable[str],
    text_col: str = "full_text",
    raw_form_col: str = "document_type_filename",
    required_item_id: str = "7",
    batch_size: int = DEFAULT_TEXT_FEATURE_BATCH_SIZE,
) -> pl.LazyFrame:
    normalized_dict = normalize_lm2011_dictionary_lists(dictionary_lists)
    signal_specs = _build_lm2011_signal_specs(
        normalized_dict=normalized_dict,
        harvard_negative_word_list=harvard_negative_word_list,
    )
    batches = _iter_text_base_batches(
        sec_item_lf,
        label="sec_items",
        text_col=text_col,
        raw_form_col=raw_form_col,
        include_item_id=True,
        required_item_id=required_item_id,
        batch_size=batch_size,
    )
    return _build_scored_text_frame(
        batches,
        text_col=text_col,
        token_count_col="token_count_mda",
        total_token_count_col="total_token_count_mda",
        include_item_id=True,
        cleaning_policy_id=RAW_ITEM_TEXT_CLEANING_POLICY_ID,
        signal_specs=signal_specs,
        master_dictionary_words=master_dictionary_words,
    )


def build_lm2011_trading_strategy_signal_frame(
    sec_parsed_lf: pl.LazyFrame,
    *,
    lm_dictionary_lists: Mapping[str, Iterable[str]],
    harvard_negative_word_list: Iterable[str] | None,
    master_dictionary_words: Iterable[str],
    text_col: str = "full_text",
    raw_form_col: str = "document_type_filename",
    cleaning_contract: Full10KCleaningContract = "current",
    batch_size: int = DEFAULT_TEXT_FEATURE_BATCH_SIZE,
) -> pl.LazyFrame:
    normalized_lm_dict = normalize_lm2011_dictionary_lists(lm_dictionary_lists)
    harvard_tokens = _normalize_dictionary_tokens(harvard_negative_word_list)
    if not harvard_tokens:
        raise ValueError("harvard_negative_word_list is required and must contain at least one token")
    signal_specs = (
        ("fin_neg", normalized_lm_dict["negative"], True),
        ("h4n_inf", harvard_tokens, True),
    )
    batches = _iter_text_base_batches(
        sec_parsed_lf,
        label="sec_parsed",
        text_col=text_col,
        raw_form_col=raw_form_col,
        batch_size=batch_size,
    )
    return _build_scored_text_frame(
        batches,
        text_col=text_col,
        token_count_col="token_count_full_10k",
        total_token_count_col="total_token_count_full_10k",
        include_item_id=False,
        signal_specs=signal_specs,
        master_dictionary_words=master_dictionary_words,
        text_cleaner=lambda value: clean_full_10k_for_lm2011(value, contract=cleaning_contract),
    )

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
import json
import math
from pathlib import Path
import re
import shutil

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
_STREAMING_PARQUET_COMPRESSION = "zstd"


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
    return list(iter_lm2011_tokens(text))


def iter_lm2011_tokens(text: str | None) -> Iterator[str]:
    if text is None:
        return
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _LINEBREAK_HYPHEN_RE.sub(r"\1\2", normalized)
    for match in _TOKEN_RE.finditer(normalized):
        yield match.group(0).casefold()


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
            token_total, recognized_word_total, counts = _count_document_tokens(
                text_input,
                vocabulary=vocabulary,
                master_dictionary_words=master_dictionary_words,
            )
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


def _count_document_tokens(
    text: str | None,
    *,
    vocabulary: frozenset[str],
    master_dictionary_words: frozenset[str],
) -> tuple[int, int, Counter[str]]:
    token_total = 0
    recognized_word_total = 0
    counts: Counter[str] = Counter()
    for token in iter_lm2011_tokens(text):
        token_total += 1
        if token in master_dictionary_words:
            recognized_word_total += 1
        if token in vocabulary:
            counts[token] += 1
    return token_total, recognized_word_total, counts


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
    raw_form_col: str,
    token_count_col: str,
    total_token_count_col: str,
    signal_specs: tuple[tuple[str, frozenset[str], bool], ...],
) -> dict[str, pl.DataType]:
    schema: dict[str, pl.DataType] = {
        "doc_id": pl.Utf8,
        "cik_10": pl.Utf8,
        "filing_date": pl.Date,
        raw_form_col: pl.Utf8,
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


def _empty_feature_frame(
    *,
    include_item_id: bool,
    cleaning_policy_id: str | None,
    raw_form_col: str,
    token_count_col: str,
    total_token_count_col: str,
    signal_specs: tuple[tuple[str, frozenset[str], bool], ...],
) -> pl.DataFrame:
    return pl.DataFrame(
        schema=_feature_schema(
            include_item_id=include_item_id,
            include_cleaning_policy_id=cleaning_policy_id is not None,
            raw_form_col=raw_form_col,
            token_count_col=token_count_col,
            total_token_count_col=total_token_count_col,
            signal_specs=signal_specs,
        )
    )


def _build_scored_text_frame(
    batches: Iterable[pl.DataFrame],
    *,
    text_col: str,
    token_count_col: str,
    total_token_count_col: str,
    include_item_id: bool,
    raw_form_col: str,
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
        raw_form_col=raw_form_col,
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


def _pass1_schema(
    *,
    include_item_id: bool,
    raw_form_col: str,
    token_count_col: str,
    total_token_count_col: str,
) -> dict[str, pl.DataType]:
    schema: dict[str, pl.DataType] = {
        "doc_id": pl.Utf8,
        "cik_10": pl.Utf8,
        "filing_date": pl.Date,
        raw_form_col: pl.Utf8,
        "normalized_form": pl.Utf8,
        total_token_count_col: pl.Int32,
        token_count_col: pl.Int32,
        "_matched_counts_json": pl.Utf8,
    }
    if include_item_id:
        schema["item_id"] = pl.Utf8
    return schema


def _prepare_pass1_rows(
    batch: pl.DataFrame,
    *,
    text_col: str,
    vocabulary: frozenset[str],
    master_dictionary_words: frozenset[str],
    token_count_col: str,
    total_token_count_col: str,
    text_cleaner: Callable[[str | None], str | None] | None = None,
) -> tuple[list[dict[str, object]], Counter[str]]:
    rows: list[dict[str, object]] = []
    document_frequency: Counter[str] = Counter()
    for row in batch.iter_rows(named=True):
        row_dict = dict(row)
        text_value = row_dict.pop(text_col, None)
        text_input = text_value if isinstance(text_value, str) else None
        if text_cleaner is not None:
            text_input = text_cleaner(text_input)
        token_total, recognized_word_total, counts = _count_document_tokens(
            text_input,
            vocabulary=vocabulary,
            master_dictionary_words=master_dictionary_words,
        )
        rows.append(
            {
                **row_dict,
                total_token_count_col: token_total,
                token_count_col: recognized_word_total,
                "_matched_counts_json": json.dumps(dict(counts), sort_keys=True, separators=(",", ":")),
            }
        )
        document_frequency.update(counts.keys())
    return rows, document_frequency


def _feature_row_from_pass1(
    row: dict[str, object],
    *,
    raw_form_col: str,
    token_count_col: str,
    total_token_count_col: str,
    signal_specs: tuple[tuple[str, frozenset[str], bool], ...],
    idf_by_token: Mapping[str, float],
    cleaning_policy_id: str | None,
) -> dict[str, object]:
    out = {
        "doc_id": row["doc_id"],
        "cik_10": row["cik_10"],
        "filing_date": row["filing_date"],
        raw_form_col: row.get(raw_form_col),
    }
    if "item_id" in row:
        out["item_id"] = row["item_id"]
    out["normalized_form"] = row["normalized_form"]
    out[total_token_count_col] = int(row.get(total_token_count_col) or 0)
    out[token_count_col] = int(row.get(token_count_col) or 0)

    counts = Counter(
        {
            str(token): int(count)
            for token, count in json.loads(str(row.get("_matched_counts_json") or "{}")).items()
        }
    )
    recognized_word_total = int(out[token_count_col])
    denominator = float(recognized_word_total) if recognized_word_total > 0 else None
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
    if cleaning_policy_id is not None:
        out["cleaning_policy_id"] = cleaning_policy_id
    return out


def _prepare_temp_root(output_path: Path, temp_root: Path | None) -> Path:
    resolved = temp_root or output_path.parent / f".{output_path.stem}_tmp"
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _write_streaming_text_features(
    batches: Iterable[pl.DataFrame],
    *,
    output_path: Path,
    temp_root: Path | None,
    token_count_col: str,
    total_token_count_col: str,
    include_item_id: bool,
    raw_form_col: str,
    cleaning_policy_id: str | None = None,
    signal_specs: tuple[tuple[str, frozenset[str], bool], ...],
    master_dictionary_words: Iterable[str],
    text_col: str,
    text_cleaner: Callable[[str | None], str | None] | None = None,
    progress_callback: Callable[[dict[str, int]], None] | None = None,
    cleanup_on_success: bool = True,
) -> int:
    vocabulary = frozenset().union(*(tokens for _, tokens, _ in signal_specs))
    normalized_master_dictionary_words = _normalize_master_dictionary_words(master_dictionary_words)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = _prepare_temp_root(output_path, temp_root)
    pass1_dir = temp_dir / "pass1"
    feature_dir = temp_dir / "features"
    pass1_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)

    pass1_paths: list[Path] = []
    document_frequency: Counter[str] = Counter()
    doc_count = 0
    pass1_schema = _pass1_schema(
        include_item_id=include_item_id,
        raw_form_col=raw_form_col,
        token_count_col=token_count_col,
        total_token_count_col=total_token_count_col,
    )
    feature_schema = _feature_schema(
        include_item_id=include_item_id,
        include_cleaning_policy_id=cleaning_policy_id is not None,
        raw_form_col=raw_form_col,
        token_count_col=token_count_col,
        total_token_count_col=total_token_count_col,
        signal_specs=signal_specs,
    )

    try:
        for batch_index, batch in enumerate(batches, start=1):
            rows, batch_document_frequency = _prepare_pass1_rows(
                batch,
                text_col=text_col,
                vocabulary=vocabulary,
                master_dictionary_words=normalized_master_dictionary_words,
                token_count_col=token_count_col,
                total_token_count_col=total_token_count_col,
                text_cleaner=text_cleaner,
            )
            if not rows:
                continue
            shard_path = pass1_dir / f"{batch_index:06d}.parquet"
            pl.DataFrame(rows, schema_overrides=pass1_schema).write_parquet(
                shard_path,
                compression=_STREAMING_PARQUET_COMPRESSION,
            )
            pass1_paths.append(shard_path)
            document_frequency.update(batch_document_frequency)
            doc_count += len(rows)
            if progress_callback is not None:
                progress_callback(
                    {
                        "batch_index": batch_index,
                        "batch_doc_count": len(rows),
                        "docs_completed": doc_count,
                    }
                )

        if output_path.exists():
            output_path.unlink()
        if doc_count == 0:
            _empty_feature_frame(
                include_item_id=include_item_id,
                cleaning_policy_id=cleaning_policy_id,
                raw_form_col=raw_form_col,
                token_count_col=token_count_col,
                total_token_count_col=total_token_count_col,
                signal_specs=signal_specs,
            ).write_parquet(output_path, compression=_STREAMING_PARQUET_COMPRESSION)
            if cleanup_on_success and temp_dir.exists():
                shutil.rmtree(temp_dir)
            return 0

        idf_by_token = {
            token: _lm2011_inverse_document_frequency(doc_count, doc_freq)
            for token, doc_freq in document_frequency.items()
        }
        feature_shard_paths: list[Path] = []
        for batch_index, shard_path in enumerate(pass1_paths, start=1):
            pass1_df = pl.read_parquet(shard_path)
            rows = [
                _feature_row_from_pass1(
                    row,
                    raw_form_col=raw_form_col,
                    token_count_col=token_count_col,
                    total_token_count_col=total_token_count_col,
                    signal_specs=signal_specs,
                    idf_by_token=idf_by_token,
                    cleaning_policy_id=cleaning_policy_id,
                )
                for row in pass1_df.iter_rows(named=True)
            ]
            if not rows:
                continue
            feature_shard_path = feature_dir / f"{batch_index:06d}.parquet"
            pl.DataFrame(rows, schema_overrides=feature_schema).write_parquet(
                feature_shard_path,
                compression=_STREAMING_PARQUET_COMPRESSION,
            )
            feature_shard_paths.append(feature_shard_path)

        if feature_shard_paths:
            pl.scan_parquet([str(path) for path in feature_shard_paths]).sink_parquet(
                output_path,
                compression=_STREAMING_PARQUET_COMPRESSION,
            )
        else:
            _empty_feature_frame(
                include_item_id=include_item_id,
                cleaning_policy_id=cleaning_policy_id,
                raw_form_col=raw_form_col,
                token_count_col=token_count_col,
                total_token_count_col=total_token_count_col,
                signal_specs=signal_specs,
            ).write_parquet(output_path, compression=_STREAMING_PARQUET_COMPRESSION)
        row_count = int(pl.scan_parquet(output_path).select(pl.len()).collect().item())
        if cleanup_on_success and temp_dir.exists():
            shutil.rmtree(temp_dir)
        return row_count
    except Exception:
        raise


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
        raw_form_col=raw_form_col,
        signal_specs=signal_specs,
        master_dictionary_words=master_dictionary_words,
        text_cleaner=lambda value: clean_full_10k_for_lm2011(value, contract=cleaning_contract),
    )


def write_lm2011_text_features_full_10k_parquet(
    sec_parsed_lf: pl.LazyFrame,
    *,
    output_path: Path,
    dictionary_lists: Mapping[str, Iterable[str]],
    harvard_negative_word_list: Iterable[str] | None,
    master_dictionary_words: Iterable[str],
    text_col: str = "full_text",
    raw_form_col: str = "document_type_filename",
    cleaning_contract: Full10KCleaningContract = "current",
    batch_size: int = DEFAULT_TEXT_FEATURE_BATCH_SIZE,
    progress_callback: Callable[[dict[str, int]], None] | None = None,
    temp_root: Path | None = None,
    cleanup_on_success: bool = True,
) -> int:
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
    return _write_streaming_text_features(
        batches,
        output_path=output_path,
        temp_root=temp_root,
        text_col=text_col,
        token_count_col="token_count_full_10k",
        total_token_count_col="total_token_count_full_10k",
        include_item_id=False,
        raw_form_col=raw_form_col,
        signal_specs=signal_specs,
        master_dictionary_words=master_dictionary_words,
        text_cleaner=lambda value: clean_full_10k_for_lm2011(value, contract=cleaning_contract),
        progress_callback=progress_callback,
        cleanup_on_success=cleanup_on_success,
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
        raw_form_col=raw_form_col,
        cleaning_policy_id=RAW_ITEM_TEXT_CLEANING_POLICY_ID,
        signal_specs=signal_specs,
        master_dictionary_words=master_dictionary_words,
    )


def write_lm2011_text_features_mda_parquet(
    sec_item_lf: pl.LazyFrame,
    *,
    output_path: Path,
    dictionary_lists: Mapping[str, Iterable[str]],
    harvard_negative_word_list: Iterable[str] | None,
    master_dictionary_words: Iterable[str],
    text_col: str = "full_text",
    raw_form_col: str = "document_type_filename",
    required_item_id: str = "7",
    batch_size: int = DEFAULT_TEXT_FEATURE_BATCH_SIZE,
    progress_callback: Callable[[dict[str, int]], None] | None = None,
    temp_root: Path | None = None,
    cleanup_on_success: bool = True,
) -> int:
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
    return _write_streaming_text_features(
        batches,
        output_path=output_path,
        temp_root=temp_root,
        text_col=text_col,
        token_count_col="token_count_mda",
        total_token_count_col="total_token_count_mda",
        include_item_id=True,
        raw_form_col=raw_form_col,
        cleaning_policy_id=RAW_ITEM_TEXT_CLEANING_POLICY_ID,
        signal_specs=signal_specs,
        master_dictionary_words=master_dictionary_words,
        progress_callback=progress_callback,
        cleanup_on_success=cleanup_on_success,
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
        raw_form_col=raw_form_col,
        signal_specs=signal_specs,
        master_dictionary_words=master_dictionary_words,
        text_cleaner=lambda value: clean_full_10k_for_lm2011(value, contract=cleaning_contract),
    )

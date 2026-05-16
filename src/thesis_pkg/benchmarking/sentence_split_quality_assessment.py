from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import polars as pl

from thesis_pkg.benchmarking.sentences import _ends_with_generic_reference_no_py
from thesis_pkg.benchmarking.sentences import _ends_with_reference_stub_py
from thesis_pkg.benchmarking.sentences import _is_citation_prefix_only_line_py
from thesis_pkg.benchmarking.sentences import _is_header_like_line_py
from thesis_pkg.benchmarking.sentences import _is_separator_line as _py_is_separator_line
from thesis_pkg.benchmarking.sentences import _looks_like_citation_continuation_v3_py
from thesis_pkg.benchmarking.sentences import _normalize_sentence_key

try:
    from thesis_native import _lm2011_rust
except Exception as exc:  # pragma: no cover - optional native extension
    _lm2011_rust = None
    _SENTENCE_RUST_IMPORT_ERROR: str | None = f"{type(exc).__name__}: {exc}"
else:
    _SENTENCE_RUST_IMPORT_ERROR = None


SENTENCE_QUALITY_REQUIRED_COLUMNS: tuple[str, ...] = (
    "benchmark_sentence_id",
    "benchmark_row_id",
    "doc_id",
    "filing_year",
    "benchmark_item_code",
    "sentence_index",
    "finbert_token_count_512",
    "sentence_char_count",
    "sentence_text",
)
DEFAULT_ITEM_ORDER: tuple[str, ...] = ("item_1", "item_1a", "item_7")
_LOWER_FRAGMENT_RE = re.compile(r"^[a-z][^.!?]{0,60}[.!?]?$")
_TERMINAL_PUNCT_RE = re.compile(r"""[.!?]["')\]]*$""")
_NUMERIC_ONLY_RE = re.compile(
    r"""^(?:\(?\s*(?:\d+|[ivxlcdm]+)(?:\s*[-./]\s*(?:\d+|[ivxlcdm]+))*\s*[.)]?\)?|\.)$""",
    re.IGNORECASE,
)
_TABLE_NUMERIC_TOKEN_RE = re.compile(r"\b\d[\d,().%-]*\b")
_UPPERCASE_TOKEN_RE = re.compile(r"\b[A-Z][A-Z/&-]{2,}\b")
_YEAR_TOKEN_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_SENTENCE_RUST_METRICS: dict[str, int] = {
    "separator_fast_success": 0,
    "separator_fallbacks": 0,
    "numeric_only_fast_success": 0,
    "numeric_only_fallbacks": 0,
    "short_fragment_fast_success": 0,
    "short_fragment_fallbacks": 0,
    "very_short_fragment_fast_success": 0,
    "very_short_fragment_fallbacks": 0,
    "lower_fragment_fast_success": 0,
    "lower_fragment_fallbacks": 0,
    "one_word_fast_success": 0,
    "one_word_fallbacks": 0,
    "terminal_punct_fast_success": 0,
    "terminal_punct_fallbacks": 0,
    "table_like_fast_success": 0,
    "table_like_fallbacks": 0,
    "generic_no_continuation_fast_success": 0,
    "generic_no_continuation_fallbacks": 0,
    "ordered_item_codes_fast_success": 0,
    "ordered_item_codes_fast_failures": 0,
    "ordered_item_codes_fallbacks": 0,
    "batch_flags_fast_success": 0,
    "batch_flags_fast_failures": 0,
    "batch_flags_fallbacks": 0,
}


def get_sentence_quality_rust_accel_metrics() -> dict[str, int | str | bool | None]:
    metrics: dict[str, int | str | bool | None] = dict(_SENTENCE_RUST_METRICS)
    metrics["rust_accel_available"] = _lm2011_rust is not None
    metrics["rust_accel_import_error"] = _SENTENCE_RUST_IMPORT_ERROR
    return metrics


def reset_sentence_quality_rust_accel_metrics() -> None:
    for key in _SENTENCE_RUST_METRICS:
        _SENTENCE_RUST_METRICS[key] = 0


@dataclass(frozen=True)
class SentenceSplitQualityAssessment:
    summary_by_scope: pl.DataFrame
    split_audit_summary: pl.DataFrame
    example_frames: dict[str, pl.DataFrame]
    metadata: dict[str, Any]


def _normalize_sentence_dataset_dir(sentence_dataset_dir: Path) -> Path:
    candidate = sentence_dataset_dir.resolve()
    by_year_dir = candidate if candidate.name == "by_year" else candidate / "by_year"
    if not by_year_dir.exists():
        raise FileNotFoundError(f"Sentence dataset directory not found: {by_year_dir}")
    parquet_paths = tuple(sorted(path for path in by_year_dir.glob("*.parquet") if path.is_file()))
    if not parquet_paths:
        raise FileNotFoundError(f"No yearly sentence parquet files were found under {by_year_dir}")
    schema = pl.scan_parquet(str(parquet_paths[0])).collect_schema()
    missing = [column for column in SENTENCE_QUALITY_REQUIRED_COLUMNS if column not in schema]
    if missing:
        raise ValueError(f"Sentence parquet {parquet_paths[0]} is missing required columns: {missing}")
    return by_year_dir


def _sentence_dataset_paths(sentence_dataset_dir: Path) -> tuple[Path, ...]:
    by_year_dir = _normalize_sentence_dataset_dir(sentence_dataset_dir)
    return tuple(sorted(path for path in by_year_dir.glob("*.parquet") if path.is_file()))


def _resolve_split_audit_path(sentence_dataset_dir: Path) -> Path | None:
    by_year_dir = _normalize_sentence_dataset_dir(sentence_dataset_dir)
    search_roots = [by_year_dir.parent, by_year_dir.parent.parent]
    candidate_names = ("sentence_split_audit.parquet", "oversize_sections.parquet")
    for root in search_roots:
        for candidate_name in candidate_names:
            candidate = root / candidate_name
            if candidate.exists():
                return candidate
    return None


def _resolve_cleaning_row_audit_path(
    sentence_dataset_dir: Path,
    cleaning_row_audit_path: Path | None,
) -> Path | None:
    if cleaning_row_audit_path is not None:
        resolved = cleaning_row_audit_path.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"cleaning_row_audit_path does not exist: {resolved}")
        return resolved

    by_year_dir = _normalize_sentence_dataset_dir(sentence_dataset_dir)
    search_roots = [by_year_dir.parent, by_year_dir.parent.parent]
    for root in search_roots:
        candidate = root / "item_cleaning" / "cleaning_row_audit.parquet"
        if candidate.exists():
            return candidate
        candidate = root / "cleaning_row_audit.parquet"
        if candidate.exists():
            return candidate
    return None


def _is_separator_line(line: str) -> bool:
    if _lm2011_rust is not None:
        try:
            out = bool(_lm2011_rust.sentence_is_separator_line(str(line or "")))
            _SENTENCE_RUST_METRICS["separator_fast_success"] += 1
            return out
        except Exception:
            _SENTENCE_RUST_METRICS["separator_fallbacks"] += 1
    else:
        _SENTENCE_RUST_METRICS["separator_fallbacks"] += 1
    return _py_is_separator_line(line)


def _numeric_only_fragment_py(text: str) -> bool:
    stripped = _normalize_sentence_key(text)
    if not stripped:
        return False
    if _py_is_separator_line(stripped):
        return False
    if _NUMERIC_ONLY_RE.fullmatch(stripped):
        return True
    return bool(stripped in {".", ",", ";", ":"})


def _numeric_only_fragment(text: str) -> bool:
    if _lm2011_rust is not None:
        try:
            out = bool(_lm2011_rust.sentence_numeric_only_fragment(str(text or "")))
            _SENTENCE_RUST_METRICS["numeric_only_fast_success"] += 1
            return out
        except Exception:
            _SENTENCE_RUST_METRICS["numeric_only_fallbacks"] += 1
    else:
        _SENTENCE_RUST_METRICS["numeric_only_fallbacks"] += 1
    return _numeric_only_fragment_py(text)


def _short_fragment_py(text: str, token_count: int, char_count: int) -> bool:
    normalized = _normalize_sentence_key(text)
    if not normalized:
        return False
    if _numeric_only_fragment_py(normalized) or _py_is_separator_line(normalized):
        return True
    return token_count <= 10 and char_count <= 48


def _short_fragment(text: str, token_count: int, char_count: int) -> bool:
    if _lm2011_rust is not None:
        try:
            out = bool(
                _lm2011_rust.sentence_short_fragment(
                    str(text or ""),
                    int(token_count or 0),
                    int(char_count or 0),
                )
            )
            _SENTENCE_RUST_METRICS["short_fragment_fast_success"] += 1
            return out
        except Exception:
            _SENTENCE_RUST_METRICS["short_fragment_fallbacks"] += 1
    else:
        _SENTENCE_RUST_METRICS["short_fragment_fallbacks"] += 1
    return _short_fragment_py(text, token_count, char_count)


def _very_short_fragment_py(text: str, token_count: int, char_count: int) -> bool:
    normalized = _normalize_sentence_key(text)
    if not normalized:
        return False
    return token_count <= 4 and char_count <= 16


def _very_short_fragment(text: str, token_count: int, char_count: int) -> bool:
    if _lm2011_rust is not None:
        try:
            out = bool(
                _lm2011_rust.sentence_very_short_fragment(
                    str(text or ""),
                    int(token_count or 0),
                    int(char_count or 0),
                )
            )
            _SENTENCE_RUST_METRICS["very_short_fragment_fast_success"] += 1
            return out
        except Exception:
            _SENTENCE_RUST_METRICS["very_short_fragment_fallbacks"] += 1
    else:
        _SENTENCE_RUST_METRICS["very_short_fragment_fallbacks"] += 1
    return _very_short_fragment_py(text, token_count, char_count)


def _one_word_fragment_py(text: str) -> bool:
    words = re.findall(r"[A-Za-z]+", _normalize_sentence_key(text))
    return len(words) == 1


def _one_word_fragment(text: str) -> bool:
    if _lm2011_rust is not None:
        try:
            out = bool(_lm2011_rust.sentence_one_word_fragment(str(text or "")))
            _SENTENCE_RUST_METRICS["one_word_fast_success"] += 1
            return out
        except Exception:
            _SENTENCE_RUST_METRICS["one_word_fallbacks"] += 1
    else:
        _SENTENCE_RUST_METRICS["one_word_fallbacks"] += 1
    return _one_word_fragment_py(text)


def _lower_fragment_py(text: str, token_count: int, char_count: int) -> bool:
    normalized = _normalize_sentence_key(text)
    if not normalized:
        return False
    if token_count > 12 or char_count > 64:
        return False
    return bool(_LOWER_FRAGMENT_RE.fullmatch(normalized))


def _lower_fragment(text: str, token_count: int, char_count: int) -> bool:
    if _lm2011_rust is not None:
        try:
            out = bool(
                _lm2011_rust.sentence_lower_fragment(
                    str(text or ""),
                    int(token_count or 0),
                    int(char_count or 0),
                )
            )
            _SENTENCE_RUST_METRICS["lower_fragment_fast_success"] += 1
            return out
        except Exception:
            _SENTENCE_RUST_METRICS["lower_fragment_fallbacks"] += 1
    else:
        _SENTENCE_RUST_METRICS["lower_fragment_fallbacks"] += 1
    return _lower_fragment_py(text, token_count, char_count)


def _has_terminal_punct_py(text: str) -> bool:
    return bool(_TERMINAL_PUNCT_RE.search(_normalize_sentence_key(text)))


def _has_terminal_punct(text: str) -> bool:
    if _lm2011_rust is not None:
        try:
            out = bool(_lm2011_rust.sentence_has_terminal_punct(str(text or "")))
            _SENTENCE_RUST_METRICS["terminal_punct_fast_success"] += 1
            return out
        except Exception:
            _SENTENCE_RUST_METRICS["terminal_punct_fallbacks"] += 1
    else:
        _SENTENCE_RUST_METRICS["terminal_punct_fallbacks"] += 1
    return _has_terminal_punct_py(text)


def _table_like_sentence_py(text: str, token_count: int) -> bool:
    normalized = str(text)
    compact = _normalize_sentence_key(normalized)
    line_count = max(normalized.count("\n") + 1, 1)
    numeric_tokens = len(_TABLE_NUMERIC_TOKEN_RE.findall(normalized))
    uppercase_tokens = len(_UPPERCASE_TOKEN_RE.findall(normalized))
    year_tokens = len(_YEAR_TOKEN_RE.findall(normalized))
    separator_hits = len(re.findall(r"[-_=]{3,}", normalized))
    colon_hits = normalized.count(":")
    if separator_hits > 0:
        return True
    if token_count < 24:
        return False
    if line_count >= 3 and numeric_tokens >= 4:
        return True
    if line_count >= 3 and uppercase_tokens >= 6:
        return True
    if numeric_tokens >= 8 and year_tokens >= 2:
        return True
    if uppercase_tokens >= 8 and colon_hits >= 2:
        return True
    return compact.upper() == compact and line_count >= 3 and len(compact) >= 80


def _table_like_sentence(text: str, token_count: int) -> bool:
    if _lm2011_rust is not None:
        try:
            out = bool(_lm2011_rust.sentence_table_like(str(text or ""), int(token_count or 0)))
            _SENTENCE_RUST_METRICS["table_like_fast_success"] += 1
            return out
        except Exception:
            _SENTENCE_RUST_METRICS["table_like_fallbacks"] += 1
    else:
        _SENTENCE_RUST_METRICS["table_like_fallbacks"] += 1
    return _table_like_sentence_py(text, token_count)


def _generic_no_with_continuation_py(text: str, next_text: str) -> bool:
    return _ends_with_generic_reference_no_py(str(text or "")) and _looks_like_citation_continuation_v3_py(
        str(next_text or "")
    )


def _generic_no_with_continuation(text: str, next_text: str) -> bool:
    if _lm2011_rust is not None:
        try:
            out = bool(
                _lm2011_rust.sentence_generic_no_with_continuation(
                    str(text or ""),
                    str(next_text or ""),
                )
            )
            _SENTENCE_RUST_METRICS["generic_no_continuation_fast_success"] += 1
            return out
        except Exception:
            _SENTENCE_RUST_METRICS["generic_no_continuation_fallbacks"] += 1
    else:
        _SENTENCE_RUST_METRICS["generic_no_continuation_fallbacks"] += 1
    return _generic_no_with_continuation_py(text, next_text)


SENTENCE_QUALITY_BATCH_FLAG_COLUMNS: tuple[str, ...] = (
    "short_fragment",
    "very_short",
    "one_word",
    "numeric_only",
    "separator_only",
    "citation_stub",
    "generic_no_end",
    "generic_no_with_continuation",
    "citation_prefix_only",
    "header_like",
    "table_like",
    "lower_fragment",
    "no_terminal_punct",
)


def _sentence_quality_flag_rows_py(rows: list[dict[str, Any]]) -> list[dict[str, bool]]:
    out: list[dict[str, bool]] = []
    for row in rows:
        text = str(row.get("sentence_text") or "")
        next_text = str(row.get("next_sentence_text") or "")
        token_count = int(row.get("finbert_token_count_512") or 0)
        char_count = int(row.get("sentence_char_count") or 0)
        out.append(
            {
                "short_fragment": _short_fragment_py(text, token_count, char_count),
                "very_short": _very_short_fragment_py(text, token_count, char_count),
                "one_word": _one_word_fragment_py(text),
                "numeric_only": _numeric_only_fragment_py(text),
                "separator_only": _py_is_separator_line(text),
                "citation_stub": _ends_with_reference_stub_py(text),
                "generic_no_end": _ends_with_generic_reference_no_py(text),
                "generic_no_with_continuation": _generic_no_with_continuation_py(text, next_text),
                "citation_prefix_only": _is_citation_prefix_only_line_py(text),
                "header_like": _is_header_like_line_py(text),
                "table_like": _table_like_sentence_py(text, token_count),
                "lower_fragment": _lower_fragment_py(text, token_count, char_count),
                "no_terminal_punct": not _has_terminal_punct_py(text),
            }
        )
    return out


def _sentence_quality_flag_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if _lm2011_rust is not None:
        try:
            raw_rows = _lm2011_rust.sentence_quality_batch_flags(rows)
            _SENTENCE_RUST_METRICS["batch_flags_fast_success"] += 1
            return pl.DataFrame(
                [
                    dict(zip(SENTENCE_QUALITY_BATCH_FLAG_COLUMNS, flag_row, strict=True))
                    for flag_row in raw_rows
                ],
                schema={name: pl.Boolean for name in SENTENCE_QUALITY_BATCH_FLAG_COLUMNS},
            ).select(SENTENCE_QUALITY_BATCH_FLAG_COLUMNS)
        except Exception:
            _SENTENCE_RUST_METRICS["batch_flags_fast_failures"] += 1
    _SENTENCE_RUST_METRICS["batch_flags_fallbacks"] += 1
    return pl.DataFrame(
        _sentence_quality_flag_rows_py(rows),
        schema={name: pl.Boolean for name in SENTENCE_QUALITY_BATCH_FLAG_COLUMNS},
    ).select(SENTENCE_QUALITY_BATCH_FLAG_COLUMNS)


def _ordered_item_codes_py(codes: list[str]) -> list[str]:
    seen = set(codes)
    ordered = [code for code in DEFAULT_ITEM_ORDER if code in seen]
    ordered.extend(code for code in codes if code not in DEFAULT_ITEM_ORDER)
    return ordered


def _ordered_item_codes(codes: list[str]) -> list[str]:
    if _lm2011_rust is not None:
        try:
            out = list(_lm2011_rust.sentence_length_ordered_item_codes(codes, DEFAULT_ITEM_ORDER))
            _SENTENCE_RUST_METRICS["ordered_item_codes_fast_success"] += 1
            return [str(code) for code in out]
        except Exception:
            _SENTENCE_RUST_METRICS["ordered_item_codes_fast_failures"] += 1
    _SENTENCE_RUST_METRICS["ordered_item_codes_fallbacks"] += 1
    return _ordered_item_codes_py(codes)


def _empty_split_summary() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "split_reason": pl.Utf8,
            "split_rows": pl.Int64,
            "chunked_item_rows": pl.Int64,
            "warning_boundary_rows": pl.Int64,
        }
    )


def analyze_sentence_split_quality(
    sentence_dataset_dir: Path,
    *,
    cleaning_row_audit_path: Path | None = None,
) -> SentenceSplitQualityAssessment:
    sentence_paths = _sentence_dataset_paths(sentence_dataset_dir)
    sentence_df = (
        pl.scan_parquet([str(path) for path in sentence_paths])
        .select(SENTENCE_QUALITY_REQUIRED_COLUMNS)
        .collect()
        .sort(["benchmark_row_id", "sentence_index"])
    )
    if sentence_df.is_empty():
        raise ValueError("No sentence rows were available for quality assessment.")

    sentence_df = sentence_df.with_columns(
        [
            pl.col("sentence_text").shift(1).over("benchmark_row_id").alias("prev_sentence_text"),
            pl.col("sentence_text").shift(-1).over("benchmark_row_id").alias("next_sentence_text"),
        ]
    )
    sentence_df = sentence_df.hstack(
        _sentence_quality_flag_frame(
            sentence_df.select(
                [
                    "sentence_text",
                    "next_sentence_text",
                    "finbert_token_count_512",
                    "sentence_char_count",
                ]
            ).to_dicts()
        )
    )
    sentence_df = sentence_df.with_columns(
        [
            (pl.col("finbert_token_count_512") >= 128).alias("long_sentence"),
            (pl.col("finbert_token_count_512") >= 256).alias("long_very"),
        ]
    ).with_columns(
        [
            (pl.col("generic_no_with_continuation") | pl.col("citation_prefix_only")).alias("stitch_candidate"),
        ]
    )

    summary_by_scope = (
        sentence_df.group_by("benchmark_item_code")
        .agg(
            [
                pl.len().alias("sentence_rows"),
                pl.col("short_fragment").cast(pl.Int64).sum().alias("short_fragment_rows"),
                pl.col("very_short").cast(pl.Int64).sum().alias("very_short_rows"),
                pl.col("one_word").cast(pl.Int64).sum().alias("one_word_rows"),
                pl.col("numeric_only").cast(pl.Int64).sum().alias("numeric_only_rows"),
                pl.col("separator_only").cast(pl.Int64).sum().alias("separator_only_rows"),
                pl.col("citation_stub").cast(pl.Int64).sum().alias("citation_stub_rows"),
                pl.col("generic_no_end").cast(pl.Int64).sum().alias("generic_no_end_rows"),
                pl.col("generic_no_with_continuation")
                .cast(pl.Int64)
                .sum()
                .alias("generic_no_with_continuation_rows"),
                pl.col("citation_prefix_only").cast(pl.Int64).sum().alias("citation_prefix_only_rows"),
                pl.col("stitch_candidate").cast(pl.Int64).sum().alias("stitch_candidate_rows"),
                pl.col("header_like").cast(pl.Int64).sum().alias("header_like_rows"),
                pl.col("table_like").cast(pl.Int64).sum().alias("table_like_rows"),
                pl.col("lower_fragment").cast(pl.Int64).sum().alias("lower_fragment_rows"),
                pl.col("no_terminal_punct").cast(pl.Int64).sum().alias("no_terminal_punct_rows"),
                pl.col("long_sentence").cast(pl.Int64).sum().alias("long_sentence_rows"),
                pl.col("long_very").cast(pl.Int64).sum().alias("long_very_rows"),
            ]
        )
        .with_columns(
            [
                (pl.col("short_fragment_rows") / pl.col("sentence_rows")).alias("short_fragment_share"),
                (pl.col("very_short_rows") / pl.col("sentence_rows")).alias("very_short_share"),
                (pl.col("one_word_rows") / pl.col("sentence_rows")).alias("one_word_share"),
                (pl.col("numeric_only_rows") / pl.col("sentence_rows")).alias("numeric_only_share"),
                (pl.col("separator_only_rows") / pl.col("sentence_rows")).alias("separator_only_share"),
                (pl.col("citation_stub_rows") / pl.col("sentence_rows")).alias("citation_stub_share"),
                (pl.col("generic_no_end_rows") / pl.col("sentence_rows")).alias("generic_no_end_share"),
                (
                    pl.col("generic_no_with_continuation_rows") / pl.col("sentence_rows")
                ).alias("generic_no_with_continuation_share"),
                (pl.col("citation_prefix_only_rows") / pl.col("sentence_rows")).alias(
                    "citation_prefix_only_share"
                ),
                (pl.col("stitch_candidate_rows") / pl.col("sentence_rows")).alias("stitch_candidate_share"),
                (pl.col("header_like_rows") / pl.col("sentence_rows")).alias("header_like_share"),
                (pl.col("table_like_rows") / pl.col("sentence_rows")).alias("table_like_share"),
                (pl.col("lower_fragment_rows") / pl.col("sentence_rows")).alias("lower_fragment_share"),
                (pl.col("no_terminal_punct_rows") / pl.col("sentence_rows")).alias("no_terminal_punct_share"),
                (pl.col("long_sentence_rows") / pl.col("sentence_rows")).alias("long_sentence_share"),
                (pl.col("long_very_rows") / pl.col("sentence_rows")).alias("long_very_share"),
            ]
        )
        .sort("benchmark_item_code")
    )

    split_audit_path = _resolve_split_audit_path(sentence_dataset_dir)
    split_audit_rows = 0
    chunked_item_rows = 0
    warning_boundary_rows = 0
    if split_audit_path is not None:
        split_audit_df = pl.read_parquet(split_audit_path)
        split_audit_rows = int(split_audit_df.height)
        chunked_item_rows = int(split_audit_df["benchmark_row_id"].n_unique())
        warning_boundary_rows = int(split_audit_df["warning_boundary_used"].cast(pl.Int64).sum())
        split_audit_summary = (
            split_audit_df.group_by("split_reason")
            .agg(
                [
                    pl.len().alias("split_rows"),
                    pl.col("benchmark_row_id").n_unique().alias("chunked_item_rows"),
                    pl.col("warning_boundary_used").cast(pl.Int64).sum().alias("warning_boundary_rows"),
                ]
            )
            .sort("split_reason")
        )
    else:
        split_audit_summary = _empty_split_summary()

    example_columns = [
        "benchmark_sentence_id",
        "benchmark_row_id",
        "doc_id",
        "filing_year",
        "benchmark_item_code",
        "sentence_index",
        "finbert_token_count_512",
        "sentence_char_count",
        "sentence_text",
        "prev_sentence_text",
        "next_sentence_text",
    ]
    example_frames = {
        "citation_stub": sentence_df.filter(pl.col("citation_stub")).select(example_columns),
        "generic_no_end": sentence_df.filter(pl.col("generic_no_end")).select(example_columns),
        "citation_prefix_only": sentence_df.filter(pl.col("citation_prefix_only")).select(example_columns),
        "numeric_only": sentence_df.filter(pl.col("numeric_only")).select(example_columns),
        "table_like_long": sentence_df.filter(
            pl.col("table_like") & (pl.col("finbert_token_count_512") >= 64)
        ).select(example_columns),
        "lower_fragment": sentence_df.filter(pl.col("lower_fragment")).select(example_columns),
    }

    resolved_cleaning_row_audit_path = _resolve_cleaning_row_audit_path(
        sentence_dataset_dir,
        cleaning_row_audit_path,
    )
    cleaned_item_rows_kept = None
    if resolved_cleaning_row_audit_path is not None:
        cleaning_audit_df = pl.read_parquet(resolved_cleaning_row_audit_path)
        cleaned_item_rows_kept = int(
            cleaning_audit_df.filter(~pl.col("dropped_after_cleaning")).height
            if "dropped_after_cleaning" in cleaning_audit_df.columns
            else cleaning_audit_df.height
        )

    metadata: dict[str, Any] = {
        "sentence_dataset_dir": str(_normalize_sentence_dataset_dir(sentence_dataset_dir)),
        "sentence_rows": int(sentence_df.height),
        "sentence_doc_count": int(sentence_df["doc_id"].n_unique()),
        "parquet_files": [str(path) for path in sentence_paths],
        "split_audit_path": str(split_audit_path) if split_audit_path is not None else None,
        "split_audit_rows": split_audit_rows,
        "chunked_item_rows": chunked_item_rows,
        "warning_boundary_rows": warning_boundary_rows,
        "cleaning_row_audit_path": (
            str(resolved_cleaning_row_audit_path) if resolved_cleaning_row_audit_path is not None else None
        ),
        "cleaned_item_rows_kept": cleaned_item_rows_kept,
        "item_codes_present": _ordered_item_codes(
            [str(code) for code in summary_by_scope["benchmark_item_code"].to_list()]
        ),
        "example_counts": {
            name: int(frame.height)
            for name, frame in example_frames.items()
        },
    }

    return SentenceSplitQualityAssessment(
        summary_by_scope=summary_by_scope,
        split_audit_summary=split_audit_summary,
        example_frames=example_frames,
        metadata=metadata,
    )


def write_sentence_split_quality_report(
    analysis: SentenceSplitQualityAssessment,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir = output_dir.resolve()
    data_dir = output_dir / "data"
    examples_dir = output_dir / "examples"
    data_dir.mkdir(parents=True, exist_ok=True)
    examples_dir.mkdir(parents=True, exist_ok=True)

    summary_by_scope_path = data_dir / "sentence_issue_summary_by_scope.csv"
    analysis.summary_by_scope.write_csv(summary_by_scope_path)
    analysis.summary_by_scope.write_parquet(data_dir / "sentence_issue_summary_by_scope.parquet")

    split_audit_summary_path = data_dir / "split_audit_summary.csv"
    analysis.split_audit_summary.write_csv(split_audit_summary_path)
    analysis.split_audit_summary.write_parquet(data_dir / "split_audit_summary.parquet")

    example_paths: dict[str, Path] = {}
    for name, frame in analysis.example_frames.items():
        path = examples_dir / f"{name}.csv"
        frame.write_csv(path)
        example_paths[name] = path

    metadata_path = output_dir / "summary.json"
    metadata_path.write_text(json.dumps(analysis.metadata, indent=2, sort_keys=True), encoding="utf-8")

    report_lines = [
        "# Sentence Splitting Quality Assessment",
        "",
        f"- Source run: `{analysis.metadata['sentence_dataset_dir']}`",
        f"- Sentence rows: `{analysis.metadata['sentence_rows']}` across `{analysis.metadata['sentence_doc_count']}` docs",
    ]
    if analysis.metadata["cleaned_item_rows_kept"] is not None:
        report_lines.append(f"- Cleaned item rows kept: `{analysis.metadata['cleaned_item_rows_kept']}`")
    if analysis.metadata["split_audit_path"] is not None and not analysis.split_audit_summary.is_empty():
        report_lines.append(f"- Chunked item rows: `{analysis.metadata['chunked_item_rows']}`")

    report_lines.extend(["", "## Residual Issue Rates by Scope", ""])
    summary_rows = analysis.summary_by_scope.to_dicts()
    order_lookup = {code: idx for idx, code in enumerate(DEFAULT_ITEM_ORDER)}
    for row in sorted(summary_rows, key=lambda item: order_lookup.get(str(item["benchmark_item_code"]), 999)):
        report_lines.extend(
            [
                f"### {row['benchmark_item_code']}",
                f"- rows: `{row['sentence_rows']}`",
                f"- short fragments: `{row['short_fragment_rows']}` (`{row['short_fragment_share']:.2%}`)",
                f"- numeric-only: `{row['numeric_only_rows']}` (`{row['numeric_only_share']:.2%}`)",
                f"- header-like: `{row['header_like_rows']}` (`{row['header_like_share']:.2%}`)",
                f"- table-like: `{row['table_like_rows']}` (`{row['table_like_share']:.2%}`)",
                f"- generic `No.` endings: `{row['generic_no_end_rows']}` (`{row['generic_no_end_share']:.2%}`)",
                f"- direct stitch candidates: `{row['stitch_candidate_rows']}` (`{row['stitch_candidate_share']:.2%}`)",
                f"- citation stubs: `{row['citation_stub_rows']}` (`{row['citation_stub_share']:.2%}`)",
                f"- lowercase fragments: `{row['lower_fragment_rows']}` (`{row['lower_fragment_share']:.2%}`)",
                f"- long sentences >=256 tokens: `{row['long_very_rows']}` (`{row['long_very_share']:.2%}`)",
                "",
            ]
        )

    report_lines.extend(["## Split Audit", ""])
    if analysis.split_audit_summary.is_empty():
        report_lines.append("- No split-audit file was available for this sentence dataset.")
    else:
        for row in analysis.split_audit_summary.to_dicts():
            report_lines.append(f"- `{row['split_reason']}`: `{row['split_rows']}`")

    report_lines.extend(["", "## Example Files", ""])
    for name, frame in analysis.example_frames.items():
        report_lines.append(f"- `{name}`: `{frame.height}` rows, see `examples/{name}.csv`")

    report_lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `citation_stub` should be close to zero when the citation stitcher is catching the obvious `... No.` failures.",
            "- `direct stitch candidates` tracks generic `No.` endings with a continuation-like next sentence plus pure citation-prefix fragments.",
            "- `table_like` is intentionally broad and acts as an upper-bound proxy for structure leakage rather than a precise sentence-boundary error label.",
        ]
    )

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return {
        "output_dir": output_dir,
        "data_dir": data_dir,
        "examples_dir": examples_dir,
        "summary_by_scope_path": summary_by_scope_path,
        "split_audit_summary_path": split_audit_summary_path,
        "summary_json_path": metadata_path,
        "report_path": report_path,
    }

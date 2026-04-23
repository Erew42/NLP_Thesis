from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import polars as pl
import pyarrow.parquet as pq


BENCHMARK_SENTENCE_ID_COLUMN = "benchmark_sentence_id"
BENCHMARK_ITEM_CODE_COLUMN = "benchmark_item_code"
BENCHMARK_ITEM_LABEL_COLUMN = "benchmark_item_label"
CIK_COLUMN = "cik_10"
DOC_ID_COLUMN = "doc_id"
ACCESSION_COLUMN = "accession_nodash"
FILING_DATE_COLUMN = "filing_date"
FILING_YEAR_COLUMN = "filing_year"
SENTENCE_INDEX_COLUMN = "sentence_index"
SENTENCE_TEXT_COLUMN = "sentence_text"
POSITIVE_PROB_COLUMN = "positive_prob"
NEUTRAL_PROB_COLUMN = "neutral_prob"
NEGATIVE_PROB_COLUMN = "negative_prob"
PREDICTED_LABEL_COLUMN = "predicted_label"
WORD_COUNT_COLUMN = "sentence_word_count"
SENTIMENT_COLUMN = "sentiment"
SENTIMENT_PROBABILITY_COLUMN = "sentiment_probability"

DEFAULT_ITEM_CODES: tuple[str, ...] = ("item_1a", "item_7")
DEFAULT_ITEM_ORDER: tuple[str, ...] = DEFAULT_ITEM_CODES
DEFAULT_SENTIMENT_ORDER: tuple[str, ...] = ("positive", "negative")

REQUIRED_COLUMNS: tuple[str, ...] = (
    BENCHMARK_SENTENCE_ID_COLUMN,
    DOC_ID_COLUMN,
    CIK_COLUMN,
    ACCESSION_COLUMN,
    FILING_DATE_COLUMN,
    FILING_YEAR_COLUMN,
    BENCHMARK_ITEM_CODE_COLUMN,
    BENCHMARK_ITEM_LABEL_COLUMN,
    SENTENCE_INDEX_COLUMN,
    SENTENCE_TEXT_COLUMN,
    NEGATIVE_PROB_COLUMN,
    NEUTRAL_PROB_COLUMN,
    POSITIVE_PROB_COLUMN,
    PREDICTED_LABEL_COLUMN,
)

SAMPLE_OUTPUT_COLUMNS: tuple[str, ...] = (
    BENCHMARK_SENTENCE_ID_COLUMN,
    DOC_ID_COLUMN,
    CIK_COLUMN,
    ACCESSION_COLUMN,
    FILING_DATE_COLUMN,
    FILING_YEAR_COLUMN,
    BENCHMARK_ITEM_CODE_COLUMN,
    BENCHMARK_ITEM_LABEL_COLUMN,
    SENTENCE_INDEX_COLUMN,
    SENTENCE_TEXT_COLUMN,
    WORD_COUNT_COLUMN,
    SENTIMENT_COLUMN,
    SENTIMENT_PROBABILITY_COLUMN,
    NEGATIVE_PROB_COLUMN,
    NEUTRAL_PROB_COLUMN,
    POSITIVE_PROB_COLUMN,
    PREDICTED_LABEL_COLUMN,
)

SAMPLE_OUTPUT_SCHEMA: dict[str, pl.DataType] = {
    BENCHMARK_SENTENCE_ID_COLUMN: pl.Utf8,
    DOC_ID_COLUMN: pl.Utf8,
    CIK_COLUMN: pl.Utf8,
    ACCESSION_COLUMN: pl.Utf8,
    FILING_DATE_COLUMN: pl.Utf8,
    FILING_YEAR_COLUMN: pl.Int32,
    BENCHMARK_ITEM_CODE_COLUMN: pl.Utf8,
    BENCHMARK_ITEM_LABEL_COLUMN: pl.Utf8,
    SENTENCE_INDEX_COLUMN: pl.Int64,
    SENTENCE_TEXT_COLUMN: pl.Utf8,
    WORD_COUNT_COLUMN: pl.UInt32,
    SENTIMENT_COLUMN: pl.Utf8,
    SENTIMENT_PROBABILITY_COLUMN: pl.Float64,
    NEGATIVE_PROB_COLUMN: pl.Float64,
    NEUTRAL_PROB_COLUMN: pl.Float64,
    POSITIVE_PROB_COLUMN: pl.Float64,
    PREDICTED_LABEL_COLUMN: pl.Utf8,
}


@dataclass(frozen=True)
class HighConfidenceSentenceExampleArtifacts:
    output_dir: Path
    data_dir: Path
    candidate_shards_dir: Path | None
    counts_by_item_sentiment_parquet_path: Path
    counts_by_item_sentiment_csv_path: Path
    counts_by_year_item_sentiment_parquet_path: Path
    counts_by_year_item_sentiment_csv_path: Path
    sample_candidates_parquet_path: Path
    sample_candidates_csv_path: Path
    sample_markdown_path: Path
    summary_json_path: Path


@dataclass(frozen=True)
class HighConfidenceSentenceExamplePack:
    artifacts: HighConfidenceSentenceExampleArtifacts
    counts_by_item_sentiment: pl.DataFrame
    counts_by_year_item_sentiment: pl.DataFrame
    sample_candidates: pl.DataFrame
    metadata: dict[str, Any]


def normalize_sentence_scores_dir(sentence_scores_dir: Path) -> Path:
    candidate = sentence_scores_dir.resolve()
    if candidate.name == "by_year":
        by_year_dir = candidate
    else:
        by_year_dir = candidate / "by_year"
    if not by_year_dir.exists():
        raise FileNotFoundError(f"Sentence score directory not found: {by_year_dir}")
    parquet_paths = sorted(path for path in by_year_dir.glob("*.parquet") if path.is_file())
    if not parquet_paths:
        raise FileNotFoundError(f"No yearly sentence score parquet files found under {by_year_dir}")
    return by_year_dir


def sentence_score_paths(sentence_scores_dir: Path) -> tuple[Path, ...]:
    by_year_dir = normalize_sentence_scores_dir(sentence_scores_dir)
    return tuple(sorted(path for path in by_year_dir.glob("*.parquet") if path.is_file()))


def _normalize_item_codes(item_codes: tuple[str, ...] | None) -> tuple[str, ...]:
    if item_codes is None:
        return DEFAULT_ITEM_CODES
    normalized = tuple(
        dict.fromkeys(str(code).strip().lower() for code in item_codes if str(code).strip())
    )
    return normalized or DEFAULT_ITEM_CODES


def _ordered_item_codes(item_codes: tuple[str, ...]) -> list[str]:
    seen = set(item_codes)
    ordered = [code for code in DEFAULT_ITEM_ORDER if code in seen]
    ordered.extend(code for code in item_codes if code not in DEFAULT_ITEM_ORDER)
    return ordered


def _item_label(item_code: str) -> str:
    labels = {
        "item_1a": "Item 1A",
        "item_7": "Item 7",
    }
    return labels.get(item_code, item_code)


def _required_schema(path: Path) -> None:
    schema = pl.scan_parquet(path).collect_schema()
    missing = [column for column in REQUIRED_COLUMNS if column not in schema]
    if missing:
        raise ValueError(f"Sentence score parquet {path} is missing required columns: {missing}")


def _prepare_artifacts(
    output_dir: Path,
    *,
    write_candidate_shards: bool,
) -> HighConfidenceSentenceExampleArtifacts:
    output_dir = output_dir.resolve()
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    candidate_shards_dir = output_dir / "candidate_shards" / "by_year" if write_candidate_shards else None
    return HighConfidenceSentenceExampleArtifacts(
        output_dir=output_dir,
        data_dir=data_dir,
        candidate_shards_dir=candidate_shards_dir,
        counts_by_item_sentiment_parquet_path=data_dir / "candidate_counts_by_item_sentiment.parquet",
        counts_by_item_sentiment_csv_path=data_dir / "candidate_counts_by_item_sentiment.csv",
        counts_by_year_item_sentiment_parquet_path=data_dir / "candidate_counts_by_year_item_sentiment.parquet",
        counts_by_year_item_sentiment_csv_path=data_dir / "candidate_counts_by_year_item_sentiment.csv",
        sample_candidates_parquet_path=data_dir / "sample_candidates.parquet",
        sample_candidates_csv_path=data_dir / "sample_candidates.csv",
        sample_markdown_path=output_dir / "sample_candidates.md",
        summary_json_path=output_dir / "summary.json",
    )


def _clear_managed_outputs(artifacts: HighConfidenceSentenceExampleArtifacts) -> None:
    for path in (
        artifacts.counts_by_item_sentiment_parquet_path,
        artifacts.counts_by_item_sentiment_csv_path,
        artifacts.counts_by_year_item_sentiment_parquet_path,
        artifacts.counts_by_year_item_sentiment_csv_path,
        artifacts.sample_candidates_parquet_path,
        artifacts.sample_candidates_csv_path,
        artifacts.sample_markdown_path,
        artifacts.summary_json_path,
        artifacts.data_dir / "high_confidence_sentence_candidates.parquet",
    ):
        if path.exists():
            path.unlink()
    if artifacts.candidate_shards_dir is not None and artifacts.candidate_shards_dir.exists():
        candidate_root = artifacts.candidate_shards_dir.parents[0]
        if candidate_root.parent != artifacts.output_dir:
            raise RuntimeError(
                f"Refusing to remove unexpected candidate shard directory: {artifacts.candidate_shards_dir}"
            )
        shutil.rmtree(candidate_root)


def _filter_candidate_batch(
    batch_df: pl.DataFrame,
    *,
    item_codes: tuple[str, ...],
    min_probability: float,
    min_word_count: int,
) -> pl.DataFrame:
    filtered_df = (
        batch_df.select(REQUIRED_COLUMNS)
        .with_columns(
            [
                pl.col(BENCHMARK_ITEM_CODE_COLUMN)
                .cast(pl.Utf8, strict=False)
                .str.to_lowercase()
                .alias(BENCHMARK_ITEM_CODE_COLUMN),
                pl.col(SENTENCE_TEXT_COLUMN)
                .fill_null("")
                .str.replace_all(r"\s+", " ")
                .str.strip_chars()
                .alias(SENTENCE_TEXT_COLUMN),
            ]
        )
        .filter(pl.col(BENCHMARK_ITEM_CODE_COLUMN).is_in(item_codes))
        .with_columns(
            pl.col(SENTENCE_TEXT_COLUMN)
            .str.count_matches(r"\S+")
            .cast(pl.UInt32)
            .alias(WORD_COUNT_COLUMN)
        )
        .filter(pl.col(WORD_COUNT_COLUMN) >= pl.lit(min_word_count))
        .with_columns(
            [
                pl.when(pl.col(POSITIVE_PROB_COLUMN) >= pl.lit(min_probability))
                .then(pl.lit("positive"))
                .when(pl.col(NEGATIVE_PROB_COLUMN) >= pl.lit(min_probability))
                .then(pl.lit("negative"))
                .otherwise(pl.lit(None, dtype=pl.Utf8))
                .alias(SENTIMENT_COLUMN),
                pl.when(pl.col(POSITIVE_PROB_COLUMN) >= pl.lit(min_probability))
                .then(pl.col(POSITIVE_PROB_COLUMN))
                .when(pl.col(NEGATIVE_PROB_COLUMN) >= pl.lit(min_probability))
                .then(pl.col(NEGATIVE_PROB_COLUMN))
                .otherwise(pl.lit(None, dtype=pl.Float64))
                .alias(SENTIMENT_PROBABILITY_COLUMN),
            ]
        )
        .filter(pl.col(SENTIMENT_COLUMN).is_not_null())
        .select(SAMPLE_OUTPUT_COLUMNS)
    )
    if filtered_df.height == 0:
        return _empty_sample_candidates_frame()
    return filtered_df


def _stable_sample_key(seed: int, benchmark_sentence_id: str) -> int:
    digest = hashlib.blake2b(
        f"{seed}|{benchmark_sentence_id}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _normalized_sample_text(sentence_text: str) -> str:
    return " ".join(sentence_text.split()).casefold()


def _consider_sample(
    selected_rows: list[dict[str, Any]],
    candidate_row: dict[str, Any],
    *,
    sample_size_per_group: int,
    seed: int,
) -> None:
    sample_key = _stable_sample_key(seed, str(candidate_row[BENCHMARK_SENTENCE_ID_COLUMN]))
    normalized_text = _normalized_sample_text(str(candidate_row[SENTENCE_TEXT_COLUMN]))
    candidate = dict(candidate_row)
    candidate["_sample_key"] = sample_key
    candidate["_normalized_text"] = normalized_text

    conflicting_indexes = [
        index
        for index, existing in enumerate(selected_rows)
        if existing["_normalized_text"] == normalized_text
        or existing[DOC_ID_COLUMN] == candidate[DOC_ID_COLUMN]
    ]
    if any(sample_key >= selected_rows[index]["_sample_key"] for index in conflicting_indexes):
        return
    for index in reversed(conflicting_indexes):
        selected_rows.pop(index)

    if len(selected_rows) < sample_size_per_group:
        selected_rows.append(candidate)
        return

    largest_index, largest_entry = max(
        enumerate(selected_rows),
        key=lambda item: item[1]["_sample_key"],
    )
    if sample_key < largest_entry["_sample_key"]:
        selected_rows[largest_index] = candidate


def _update_accumulators(
    filtered_df: pl.DataFrame,
    *,
    counts_by_item_sentiment: dict[tuple[str, str], dict[str, Any]],
    counts_by_year_item_sentiment: dict[tuple[int, str, str], dict[str, Any]],
    selected_samples: dict[tuple[str, str], list[dict[str, Any]]],
    sample_size_per_group: int,
    seed: int,
    all_doc_ids: set[str],
) -> None:
    for row in filtered_df.iter_rows(named=True):
        item_code = str(row[BENCHMARK_ITEM_CODE_COLUMN])
        sentiment = str(row[SENTIMENT_COLUMN])
        filing_year = int(row[FILING_YEAR_COLUMN])
        doc_id = str(row[DOC_ID_COLUMN])
        key = (item_code, sentiment)
        year_key = (filing_year, item_code, sentiment)

        bucket_acc = counts_by_item_sentiment.setdefault(
            key,
            {"candidate_rows": 0, "doc_ids": set()},
        )
        bucket_acc["candidate_rows"] += 1
        bucket_acc["doc_ids"].add(doc_id)

        year_bucket_acc = counts_by_year_item_sentiment.setdefault(
            year_key,
            {"candidate_rows": 0, "doc_ids": set()},
        )
        year_bucket_acc["candidate_rows"] += 1
        year_bucket_acc["doc_ids"].add(doc_id)

        all_doc_ids.add(doc_id)
        selected_rows = selected_samples.setdefault(key, [])
        _consider_sample(
            selected_rows,
            row,
            sample_size_per_group=sample_size_per_group,
            seed=seed,
        )


def _counts_by_item_sentiment_frame(
    counts_by_item_sentiment: dict[tuple[str, str], dict[str, Any]],
    *,
    ordered_items: tuple[str, ...],
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for item_code in ordered_items:
        for sentiment in DEFAULT_SENTIMENT_ORDER:
            acc = counts_by_item_sentiment.get((item_code, sentiment))
            if acc is None:
                continue
            rows.append(
                {
                    BENCHMARK_ITEM_CODE_COLUMN: item_code,
                    SENTIMENT_COLUMN: sentiment,
                    "candidate_rows": int(acc["candidate_rows"]),
                    "doc_count": len(acc["doc_ids"]),
                }
            )
    return pl.DataFrame(rows)


def _counts_by_year_item_sentiment_frame(
    counts_by_year_item_sentiment: dict[tuple[int, str, str], dict[str, Any]],
    *,
    ordered_items: tuple[str, ...],
) -> pl.DataFrame:
    item_order = {item_code: index for index, item_code in enumerate(ordered_items)}
    sentiment_order = {
        sentiment: index for index, sentiment in enumerate(DEFAULT_SENTIMENT_ORDER)
    }
    rows: list[dict[str, Any]] = []
    for (filing_year, item_code, sentiment), acc in counts_by_year_item_sentiment.items():
        rows.append(
            {
                FILING_YEAR_COLUMN: filing_year,
                BENCHMARK_ITEM_CODE_COLUMN: item_code,
                SENTIMENT_COLUMN: sentiment,
                "candidate_rows": int(acc["candidate_rows"]),
                "doc_count": len(acc["doc_ids"]),
                "_item_order": item_order.get(item_code, len(item_order)),
                "_sentiment_order": sentiment_order.get(sentiment, len(sentiment_order)),
            }
        )
    if not rows:
        return pl.DataFrame(rows)
    return (
        pl.DataFrame(rows)
        .sort(
            [FILING_YEAR_COLUMN, "_item_order", "_sentiment_order"],
            descending=[False, False, False],
        )
        .drop(["_item_order", "_sentiment_order"])
    )


def _empty_sample_candidates_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=SAMPLE_OUTPUT_SCHEMA)


def _sample_candidates_frame(
    selected_samples: dict[tuple[str, str], list[dict[str, Any]]],
    *,
    ordered_items: tuple[str, ...],
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for item_code in ordered_items:
        for sentiment in DEFAULT_SENTIMENT_ORDER:
            candidates = selected_samples.get((item_code, sentiment), [])
            for row in sorted(
                candidates,
                key=lambda item: (
                    -float(item[SENTIMENT_PROBABILITY_COLUMN]),
                    int(item[FILING_YEAR_COLUMN]),
                    str(item[DOC_ID_COLUMN]),
                    str(item[BENCHMARK_SENTENCE_ID_COLUMN]),
                ),
            ):
                rows.append(
                    {
                        column: (
                            row[column].isoformat()
                            if column == FILING_DATE_COLUMN and hasattr(row[column], "isoformat")
                            else row[column]
                        )
                        for column in SAMPLE_OUTPUT_COLUMNS
                    }
                )
    if not rows:
        return _empty_sample_candidates_frame()
    return pl.DataFrame(rows, schema_overrides=SAMPLE_OUTPUT_SCHEMA)


def _write_frame_pair(frame: pl.DataFrame, parquet_path: Path, csv_path: Path) -> None:
    frame.write_parquet(parquet_path, compression="zstd")
    frame.write_csv(csv_path)


def _row_to_jsonable(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (value.isoformat() if hasattr(value, "isoformat") else value)
        for key, value in row.items()
    }


def _render_sample_markdown(
    sample_candidates: pl.DataFrame,
    counts_by_item_sentiment: pl.DataFrame,
    metadata: dict[str, Any],
) -> str:
    counts_lookup = {
        (str(row[BENCHMARK_ITEM_CODE_COLUMN]), str(row[SENTIMENT_COLUMN])): int(row["candidate_rows"])
        for row in counts_by_item_sentiment.to_dicts()
    }
    lines = [
        "# FinBERT High-Confidence Sentence Samples",
        "",
        f"Source sentence_scores: {metadata['sentence_scores_dir']}",
        (
            "Filters: items="
            + ", ".join(metadata["filters"]["item_codes"])
            + f", min_probability={metadata['filters']['min_probability']:.2f},"
            + f" min_word_count={metadata['filters']['min_word_count']},"
            + f" batch_size={metadata['filters']['batch_size']}"
        ),
        (
            f"Candidate rows={metadata['candidate_rows']}, candidate docs={metadata['candidate_doc_count']}, "
            f"sample_size_per_group={metadata['sample_size_per_group']}, sample_seed={metadata['sample_seed']}"
        ),
        "",
    ]

    for item_code in metadata["item_codes_present_ordered"]:
        for sentiment in DEFAULT_SENTIMENT_ORDER:
            bucket_df = sample_candidates.filter(
                (pl.col(BENCHMARK_ITEM_CODE_COLUMN) == item_code)
                & (pl.col(SENTIMENT_COLUMN) == sentiment)
            )
            lines.extend(
                [
                    f"## {_item_label(item_code)} | {sentiment.title()}",
                    (
                        f"Eligible candidates: {counts_lookup.get((item_code, sentiment), 0)} | "
                        f"Listed samples: {bucket_df.height}"
                    ),
                    "",
                ]
            )
            for index, row in enumerate(bucket_df.to_dicts(), start=1):
                lines.append(
                    (
                        f"{index}. p={float(row[SENTIMENT_PROBABILITY_COLUMN]):.3f} | "
                        f"year={row[FILING_YEAR_COLUMN]} | date={row[FILING_DATE_COLUMN]} | "
                        f"doc_id={row[DOC_ID_COLUMN]} | words={row[WORD_COUNT_COLUMN]} | "
                        f"{row[SENTENCE_TEXT_COLUMN]}"
                    )
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_high_confidence_sentence_example_pack(
    sentence_scores_dir: Path,
    *,
    output_dir: Path,
    item_codes: tuple[str, ...] | None = None,
    min_probability: float = 0.95,
    min_word_count: int = 6,
    sample_size_per_group: int = 50,
    seed: int = 42,
    batch_size: int = 50_000,
    write_candidate_shards: bool = False,
) -> HighConfidenceSentenceExamplePack:
    if min_probability <= 0.0 or min_probability > 1.0:
        raise ValueError("min_probability must be in the interval (0, 1].")
    if min_word_count <= 0:
        raise ValueError("min_word_count must be positive.")
    if sample_size_per_group <= 0:
        raise ValueError("sample_size_per_group must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    parquet_paths = sentence_score_paths(sentence_scores_dir)
    _required_schema(parquet_paths[0])
    normalized_items = _normalize_item_codes(item_codes)
    artifacts = _prepare_artifacts(output_dir, write_candidate_shards=write_candidate_shards)
    _clear_managed_outputs(artifacts)

    counts_by_item_sentiment: dict[tuple[str, str], dict[str, Any]] = {}
    counts_by_year_item_sentiment: dict[tuple[int, str, str], dict[str, Any]] = {}
    selected_samples: dict[tuple[str, str], list[dict[str, Any]]] = {}
    all_doc_ids: set[str] = set()
    candidate_rows = 0

    if artifacts.candidate_shards_dir is not None:
        artifacts.candidate_shards_dir.mkdir(parents=True, exist_ok=True)

    for parquet_path in parquet_paths:
        year_writer: pq.ParquetWriter | None = None
        year_output_path = (
            artifacts.candidate_shards_dir / parquet_path.name
            if artifacts.candidate_shards_dir is not None
            else None
        )
        parquet_file = pq.ParquetFile(parquet_path)
        try:
            for batch in parquet_file.iter_batches(
                batch_size=batch_size,
                columns=list(REQUIRED_COLUMNS),
            ):
                batch_df = pl.from_arrow(batch)
                filtered_df = _filter_candidate_batch(
                    batch_df,
                    item_codes=normalized_items,
                    min_probability=min_probability,
                    min_word_count=min_word_count,
                )
                if filtered_df.height == 0:
                    continue
                candidate_rows += filtered_df.height
                _update_accumulators(
                    filtered_df,
                    counts_by_item_sentiment=counts_by_item_sentiment,
                    counts_by_year_item_sentiment=counts_by_year_item_sentiment,
                    selected_samples=selected_samples,
                    sample_size_per_group=sample_size_per_group,
                    seed=seed,
                    all_doc_ids=all_doc_ids,
                )
                if year_output_path is not None:
                    table = filtered_df.to_arrow()
                    if year_writer is None:
                        year_writer = pq.ParquetWriter(
                            year_output_path,
                            table.schema,
                            compression="zstd",
                        )
                    year_writer.write_table(table)
        finally:
            parquet_file.close()
            if year_writer is not None:
                year_writer.close()

    if candidate_rows == 0:
        raise ValueError("No sentence rows matched the requested probability and word-count filters.")

    ordered_items = _ordered_item_codes(tuple(item_code for item_code, _ in counts_by_item_sentiment))
    counts_by_item_sentiment_df = _counts_by_item_sentiment_frame(
        counts_by_item_sentiment,
        ordered_items=tuple(ordered_items),
    )
    counts_by_year_item_sentiment_df = _counts_by_year_item_sentiment_frame(
        counts_by_year_item_sentiment,
        ordered_items=tuple(ordered_items),
    )
    sample_candidates_df = _sample_candidates_frame(
        selected_samples,
        ordered_items=tuple(ordered_items),
    )

    _write_frame_pair(
        counts_by_item_sentiment_df,
        artifacts.counts_by_item_sentiment_parquet_path,
        artifacts.counts_by_item_sentiment_csv_path,
    )
    _write_frame_pair(
        counts_by_year_item_sentiment_df,
        artifacts.counts_by_year_item_sentiment_parquet_path,
        artifacts.counts_by_year_item_sentiment_csv_path,
    )
    _write_frame_pair(
        sample_candidates_df,
        artifacts.sample_candidates_parquet_path,
        artifacts.sample_candidates_csv_path,
    )

    metadata: dict[str, Any] = {
        "sentence_scores_dir": str(normalize_sentence_scores_dir(sentence_scores_dir)),
        "parquet_files": [str(path) for path in parquet_paths],
        "parquet_file_count": len(parquet_paths),
        "filters": {
            "item_codes": list(normalized_items),
            "min_probability": min_probability,
            "min_word_count": min_word_count,
            "batch_size": batch_size,
            "write_candidate_shards": write_candidate_shards,
        },
        "candidate_rows": candidate_rows,
        "candidate_doc_count": len(all_doc_ids),
        "sample_size_per_group": sample_size_per_group,
        "sample_seed": seed,
        "item_codes_present": list(ordered_items),
        "item_codes_present_ordered": ordered_items,
        "artifact_paths": {
            "candidate_shards_dir": str(artifacts.candidate_shards_dir)
            if artifacts.candidate_shards_dir is not None
            else None,
            "counts_by_item_sentiment_parquet_path": str(artifacts.counts_by_item_sentiment_parquet_path),
            "counts_by_item_sentiment_csv_path": str(artifacts.counts_by_item_sentiment_csv_path),
            "counts_by_year_item_sentiment_parquet_path": str(
                artifacts.counts_by_year_item_sentiment_parquet_path
            ),
            "counts_by_year_item_sentiment_csv_path": str(
                artifacts.counts_by_year_item_sentiment_csv_path
            ),
            "sample_candidates_parquet_path": str(artifacts.sample_candidates_parquet_path),
            "sample_candidates_csv_path": str(artifacts.sample_candidates_csv_path),
            "sample_markdown_path": str(artifacts.sample_markdown_path),
            "summary_json_path": str(artifacts.summary_json_path),
        },
    }

    artifacts.sample_markdown_path.write_text(
        _render_sample_markdown(sample_candidates_df, counts_by_item_sentiment_df, metadata),
        encoding="utf-8",
    )
    sample_counts_df = (
        sample_candidates_df.group_by([BENCHMARK_ITEM_CODE_COLUMN, SENTIMENT_COLUMN])
        .len()
        .rename({"len": "sample_rows"})
        .sort([BENCHMARK_ITEM_CODE_COLUMN, SENTIMENT_COLUMN])
    )
    artifacts.summary_json_path.write_text(
        json.dumps(
            {
                **metadata,
                "counts_by_item_sentiment": [
                    _row_to_jsonable(row) for row in counts_by_item_sentiment_df.to_dicts()
                ],
                "counts_by_year_item_sentiment": [
                    _row_to_jsonable(row) for row in counts_by_year_item_sentiment_df.to_dicts()
                ],
                "sample_counts_by_item_sentiment": [
                    _row_to_jsonable(row) for row in sample_counts_df.to_dicts()
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return HighConfidenceSentenceExamplePack(
        artifacts=artifacts,
        counts_by_item_sentiment=counts_by_item_sentiment_df,
        counts_by_year_item_sentiment=counts_by_year_item_sentiment_df,
        sample_candidates=sample_candidates_df,
        metadata=metadata,
    )

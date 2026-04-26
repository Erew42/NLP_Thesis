from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Iterable

import polars as pl
import pyarrow.parquet as pq

from thesis_assets.builders.artifacts import parquet_artifact_paths
from thesis_assets.builders.artifacts import scan_parquet_artifact
from thesis_assets.errors import AssetBuildError
from thesis_assets.specs import ResolvedArtifact

DEFAULT_SENTENCE_BATCH_SIZE = 100_000
DEFAULT_ECDF_BINS = 200
FINBERT_HIGH_NEGATIVE_THRESHOLD = 0.95
LM_HIGH_NEGATIVE_SHARE_THRESHOLD = 0.05
SENTENCE_BATCH_SIZE_ENV_VAR = "THESIS_ASSETS_SENTENCE_BATCH_SIZE"
TARGET_TEXT_SCOPES = ("item_7_mda", "item_1a_risk_factors")
LM2011_LINEBREAK_HYPHEN_PATTERN = r"([A-Za-z])-\s*(?:\r?\n)\s*([A-Za-z])"
LM2011_TOKEN_PATTERN = r"[A-Za-z]{2,}(?:[-'][A-Za-z]+)*"
SCOPE_LABELS = {
    "item_7_mda": "Item 7 MD&A",
    "item_1a_risk_factors": "Item 1A risk factors",
}


@dataclass(frozen=True)
class SentenceMetricSummary:
    ecdf: pl.DataFrame
    high_share: pl.DataFrame
    sentence_count: int


def sentence_batch_size_from_env() -> int:
    raw_value = os.environ.get(SENTENCE_BATCH_SIZE_ENV_VAR)
    if raw_value is None or not raw_value.strip():
        return DEFAULT_SENTENCE_BATCH_SIZE
    try:
        batch_size = int(raw_value)
    except ValueError as exc:
        raise AssetBuildError(
            f"{SENTENCE_BATCH_SIZE_ENV_VAR} must be an integer, got {raw_value!r}."
        ) from exc
    if batch_size < 1:
        raise AssetBuildError(f"{SENTENCE_BATCH_SIZE_ENV_VAR} must be >= 1.")
    return batch_size


def build_finbert_sentence_summary(
    *,
    analysis_artifact: ResolvedArtifact,
    sentence_artifact: ResolvedArtifact,
    batch_size: int,
    bins: int = DEFAULT_ECDF_BINS,
) -> SentenceMetricSummary:
    universe_df = _analysis_universe(analysis_artifact)
    histogram_counts = _empty_histogram_counts(bins)
    high_counts: dict[tuple[str, str], list[int]] = {}
    total_sentence_count = 0

    for batch in _iter_sentence_batches(
        sentence_artifact,
        columns=("doc_id", "text_scope", "negative_prob"),
        batch_size=batch_size,
    ):
        filtered = _filter_to_analysis_universe(batch, universe_df)
        if filtered.is_empty():
            continue
        score_df = (
            filtered.lazy()
            .select(
                pl.col("doc_id").cast(pl.Utf8, strict=False),
                pl.col("text_scope").cast(pl.Utf8, strict=False),
                pl.col("negative_prob").cast(pl.Float64, strict=False).alias("score"),
            )
            .drop_nulls(subset=["doc_id", "text_scope", "score"])
            .collect()
        )
        if score_df.is_empty():
            continue
        total_sentence_count += score_df.height
        _accumulate_histogram_counts(
            histogram_counts,
            score_df,
            bins=bins,
            metric_id="finbert_negative_prob",
            metric_label="FinBERT negative probability",
        )
        _accumulate_high_counts(
            high_counts,
            score_df,
            threshold=FINBERT_HIGH_NEGATIVE_THRESHOLD,
        )

    return SentenceMetricSummary(
        ecdf=_histogram_counts_to_ecdf_frame(histogram_counts, bins=bins),
        high_share=_high_counts_to_frame(
            high_counts,
            metric_id="finbert_negative_prob",
            metric_label="FinBERT negative probability",
            threshold=FINBERT_HIGH_NEGATIVE_THRESHOLD,
        ),
        sentence_count=total_sentence_count,
    )


def build_lm_negative_sentence_summary(
    *,
    analysis_artifact: ResolvedArtifact,
    sentence_artifact: ResolvedArtifact,
    negative_words: Iterable[str],
    batch_size: int,
    bins: int = DEFAULT_ECDF_BINS,
) -> SentenceMetricSummary:
    universe_df = _analysis_universe(analysis_artifact)
    normalized_negative_words = frozenset(str(word).casefold() for word in negative_words)
    if not normalized_negative_words:
        raise AssetBuildError("LM2011 negative word list is empty.")

    histogram_counts = _empty_histogram_counts(bins)
    high_counts: dict[tuple[str, str], list[int]] = {}
    total_sentence_count = 0

    for batch in _iter_sentence_batches(
        sentence_artifact,
        columns=("doc_id", "text_scope", "sentence_text"),
        batch_size=batch_size,
    ):
        filtered = _filter_to_analysis_universe(batch, universe_df)
        if filtered.is_empty():
            continue
        score_df = (
            filtered.lazy()
            .with_columns(
                pl.col("sentence_text")
                .cast(pl.Utf8, strict=False)
                .fill_null("")
                .str.replace_all(LM2011_LINEBREAK_HYPHEN_PATTERN, "${1}${2}")
                .str.to_lowercase()
                .str.extract_all(LM2011_TOKEN_PATTERN)
                .alias("_lm_tokens")
            )
            .with_columns(
                pl.col("_lm_tokens").list.len().alias("_lm_token_count"),
                pl.col("_lm_tokens")
                .list.eval(pl.element().is_in(normalized_negative_words))
                .list.sum()
                .alias("_lm_negative_count"),
            )
            .filter(pl.col("_lm_token_count") > 0)
            .select(
                pl.col("doc_id").cast(pl.Utf8, strict=False),
                pl.col("text_scope").cast(pl.Utf8, strict=False),
                (
                    pl.col("_lm_negative_count").cast(pl.Float64)
                    / pl.col("_lm_token_count").cast(pl.Float64)
                ).alias("score"),
            )
            .drop_nulls(subset=["doc_id", "text_scope", "score"])
            .collect()
        )
        if score_df.is_empty():
            continue
        total_sentence_count += score_df.height
        _accumulate_histogram_counts(
            histogram_counts,
            score_df,
            bins=bins,
            metric_id="lm_negative_sentence_share",
            metric_label="LM2011 negative word share",
        )
        _accumulate_high_counts(
            high_counts,
            score_df,
            threshold=LM_HIGH_NEGATIVE_SHARE_THRESHOLD,
        )

    return SentenceMetricSummary(
        ecdf=_histogram_counts_to_ecdf_frame(histogram_counts, bins=bins),
        high_share=_high_counts_to_frame(
            high_counts,
            metric_id="lm_negative_sentence_share",
            metric_label="LM2011 negative word share",
            threshold=LM_HIGH_NEGATIVE_SHARE_THRESHOLD,
        ),
        sentence_count=total_sentence_count,
    )


def _analysis_universe(analysis_artifact: ResolvedArtifact) -> pl.DataFrame:
    lf = scan_parquet_artifact(analysis_artifact)
    schema = lf.collect_schema()
    filters: list[pl.Expr] = [
        pl.col("text_scope").cast(pl.Utf8, strict=False).is_in(TARGET_TEXT_SCOPES),
    ]
    if "dictionary_family_source" in schema:
        filters.append(pl.col("dictionary_family_source").cast(pl.Utf8, strict=False) == pl.lit("replication"))
    if "dictionary_family" in schema:
        filters.append(pl.col("dictionary_family").cast(pl.Utf8, strict=False) == pl.lit("replication"))

    out = lf
    for predicate in filters:
        out = out.filter(predicate)
    universe = (
        out.select(
            pl.col("doc_id").cast(pl.Utf8, strict=False),
            pl.col("text_scope").cast(pl.Utf8, strict=False),
        )
        .drop_nulls(subset=["doc_id", "text_scope"])
        .unique(subset=["doc_id", "text_scope"])
        .collect()
    )
    if universe.is_empty():
        raise AssetBuildError("Analysis-panel universe for sentence figures is empty.")
    return universe


def _iter_sentence_batches(
    artifact: ResolvedArtifact,
    *,
    columns: tuple[str, ...],
    batch_size: int,
):
    for path in parquet_artifact_paths(artifact):
        parquet_file = pq.ParquetFile(path)
        for record_batch in parquet_file.iter_batches(batch_size=batch_size, columns=list(columns)):
            if record_batch.num_rows == 0:
                continue
            yield pl.from_arrow(record_batch)


def _filter_to_analysis_universe(batch: pl.DataFrame, universe_df: pl.DataFrame) -> pl.DataFrame:
    if batch.is_empty():
        return batch
    prepared = batch.with_columns(
        pl.col("doc_id").cast(pl.Utf8, strict=False),
        pl.col("text_scope").cast(pl.Utf8, strict=False),
    ).filter(pl.col("text_scope").is_in(TARGET_TEXT_SCOPES))
    if prepared.is_empty():
        return prepared
    return prepared.join(universe_df, on=["doc_id", "text_scope"], how="semi")


def _lm_negative_sentence_share(text: object, negative_words: frozenset[str]) -> float | None:
    if not isinstance(text, str):
        return None
    normalized = re.sub(LM2011_LINEBREAK_HYPHEN_PATTERN, r"\1\2", text).lower()
    tokens = re.findall(LM2011_TOKEN_PATTERN, normalized)
    if not tokens:
        return None
    negative_count = sum(1 for token in tokens if token in negative_words)
    return float(negative_count) / float(len(tokens))


def _empty_histogram_counts(bins: int) -> defaultdict[tuple[str, str, str], list[int]]:
    return defaultdict(lambda: [0] * bins)


def _accumulate_histogram_counts(
    histogram_counts: defaultdict[tuple[str, str, str], list[int]],
    score_df: pl.DataFrame,
    *,
    bins: int,
    metric_id: str,
    metric_label: str,
) -> None:
    binned = (
        score_df.lazy()
        .with_columns(_score_bin_expr(bins).alias("score_bin"))
        .group_by("text_scope", "score_bin")
        .len()
        .collect()
    )
    for row in binned.iter_rows(named=True):
        scope = str(row["text_scope"])
        score_bin = int(row["score_bin"])
        histogram_counts[(metric_id, metric_label, scope)][score_bin] += int(row["len"])


def _score_bin_expr(bins: int) -> pl.Expr:
    return (
        pl.when(pl.col("score") <= 0.0)
        .then(pl.lit(0))
        .when(pl.col("score") >= 1.0)
        .then(pl.lit(bins - 1))
        .otherwise((pl.col("score") * float(bins)).floor().cast(pl.Int32, strict=False))
    )


def _accumulate_high_counts(
    high_counts: dict[tuple[str, str], list[int]],
    score_df: pl.DataFrame,
    *,
    threshold: float,
) -> None:
    grouped = (
        score_df.lazy()
        .with_columns((pl.col("score") >= threshold).cast(pl.Int32).alias("_is_high_negative"))
        .group_by("doc_id", "text_scope")
        .agg(
            pl.len().alias("sentence_count"),
            pl.col("_is_high_negative").sum().alias("high_sentence_count"),
        )
        .collect()
    )
    for row in grouped.iter_rows(named=True):
        key = (str(row["doc_id"]), str(row["text_scope"]))
        counts = high_counts.setdefault(key, [0, 0])
        counts[0] += int(row["sentence_count"])
        counts[1] += int(row["high_sentence_count"])


def _histogram_counts_to_ecdf_frame(
    histogram_counts: defaultdict[tuple[str, str, str], list[int]],
    *,
    bins: int,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for (metric_id, metric_label, text_scope), counts in sorted(histogram_counts.items()):
        total_count = sum(counts)
        cumulative = 0
        if total_count == 0:
            continue
        for score_bin, count in enumerate(counts):
            cumulative += count
            bin_left = float(score_bin) / float(bins)
            bin_right = float(score_bin + 1) / float(bins)
            rows.append(
                {
                    "metric_id": metric_id,
                    "metric_label": metric_label,
                    "text_scope": text_scope,
                    "scope_label": SCOPE_LABELS.get(text_scope, text_scope),
                    "series_label": f"{metric_label} - {SCOPE_LABELS.get(text_scope, text_scope)}",
                    "score_bin": score_bin,
                    "score_bin_left": bin_left,
                    "score_bin_right": bin_right,
                    "score_bin_midpoint": (bin_left + bin_right) / 2.0,
                    "count": count,
                    "cumulative_count": cumulative,
                    "total_count": total_count,
                    "ecdf": float(cumulative) / float(total_count),
                }
            )
    return pl.DataFrame(rows) if rows else _empty_ecdf_frame()


def _high_counts_to_frame(
    high_counts: dict[tuple[str, str], list[int]],
    *,
    metric_id: str,
    metric_label: str,
    threshold: float,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for (doc_id, text_scope), (sentence_count, high_sentence_count) in sorted(high_counts.items()):
        if sentence_count <= 0:
            continue
        rows.append(
            {
                "metric_id": metric_id,
                "metric_label": metric_label,
                "threshold": threshold,
                "doc_id": doc_id,
                "text_scope": text_scope,
                "scope_label": SCOPE_LABELS.get(text_scope, text_scope),
                "series_label": f"{metric_label} - {SCOPE_LABELS.get(text_scope, text_scope)}",
                "sentence_count": sentence_count,
                "high_sentence_count": high_sentence_count,
                "high_sentence_share": float(high_sentence_count) / float(sentence_count),
            }
        )
    return pl.DataFrame(rows) if rows else _empty_high_share_frame()


def _empty_ecdf_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "metric_id": pl.Utf8,
            "metric_label": pl.Utf8,
            "text_scope": pl.Utf8,
            "scope_label": pl.Utf8,
            "series_label": pl.Utf8,
            "score_bin": pl.Int32,
            "score_bin_left": pl.Float64,
            "score_bin_right": pl.Float64,
            "score_bin_midpoint": pl.Float64,
            "count": pl.Int64,
            "cumulative_count": pl.Int64,
            "total_count": pl.Int64,
            "ecdf": pl.Float64,
        }
    )


def _empty_high_share_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "metric_id": pl.Utf8,
            "metric_label": pl.Utf8,
            "threshold": pl.Float64,
            "doc_id": pl.Utf8,
            "text_scope": pl.Utf8,
            "scope_label": pl.Utf8,
            "series_label": pl.Utf8,
            "sentence_count": pl.Int64,
            "high_sentence_count": pl.Int64,
            "high_sentence_share": pl.Float64,
        }
    )

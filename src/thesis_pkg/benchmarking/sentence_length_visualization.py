from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import polars as pl


TOKEN_COUNT_COLUMN = "finbert_token_count_512"
TOKEN_BUCKET_COLUMN = "finbert_token_bucket_512"
CHAR_COUNT_COLUMN = "sentence_char_count"
ITEM_CODE_COLUMN = "benchmark_item_code"
DOC_ID_COLUMN = "doc_id"
YEAR_COLUMN = "filing_year"
SENTENCE_ID_COLUMN = "benchmark_sentence_id"
SENTENCE_TEXT_COLUMN = "sentence_text"

REQUIRED_COLUMNS: tuple[str, ...] = (
    SENTENCE_ID_COLUMN,
    DOC_ID_COLUMN,
    YEAR_COLUMN,
    ITEM_CODE_COLUMN,
    SENTENCE_TEXT_COLUMN,
    CHAR_COUNT_COLUMN,
    TOKEN_COUNT_COLUMN,
    TOKEN_BUCKET_COLUMN,
)

BUCKET_ORDER: tuple[str, ...] = ("short", "medium", "long")
DEFAULT_ITEM_ORDER: tuple[str, ...] = ("item_1", "item_1a", "item_7")


@dataclass(frozen=True)
class SentenceLengthAnalysis:
    summary_overall: pl.DataFrame
    summary_by_item: pl.DataFrame
    summary_by_year_item: pl.DataFrame
    token_histogram: pl.DataFrame
    token_histogram_by_item: pl.DataFrame
    char_histogram: pl.DataFrame
    char_histogram_by_item: pl.DataFrame
    bucket_counts_by_item: pl.DataFrame
    longest_sentences: pl.DataFrame
    metadata: dict[str, Any]


def normalize_sentence_dataset_dir(sentence_dataset_dir: Path) -> Path:
    candidate = sentence_dataset_dir.resolve()
    if candidate.name == "by_year":
        by_year_dir = candidate
    else:
        by_year_dir = candidate / "by_year"
    if not by_year_dir.exists():
        raise FileNotFoundError(f"Sentence dataset directory not found: {by_year_dir}")
    parquet_paths = sorted(path for path in by_year_dir.glob("*.parquet") if path.is_file())
    if not parquet_paths:
        raise FileNotFoundError(f"No yearly sentence parquet files found under {by_year_dir}")
    return by_year_dir


def sentence_dataset_paths(sentence_dataset_dir: Path) -> tuple[Path, ...]:
    by_year_dir = normalize_sentence_dataset_dir(sentence_dataset_dir)
    return tuple(sorted(path for path in by_year_dir.glob("*.parquet") if path.is_file()))


def _ordered_item_codes(codes: tuple[str, ...]) -> list[str]:
    seen = set(codes)
    ordered = [code for code in DEFAULT_ITEM_ORDER if code in seen]
    ordered.extend(code for code in codes if code not in DEFAULT_ITEM_ORDER)
    return ordered


def _required_schema(path: Path) -> None:
    schema = pl.scan_parquet(path).collect_schema()
    missing = [column for column in REQUIRED_COLUMNS if column not in schema]
    if missing:
        raise ValueError(f"Sentence parquet {path} is missing required columns: {missing}")


def _scan_sentence_lengths(
    sentence_dataset_dir: Path,
    *,
    item_codes: tuple[str, ...] | None = None,
    years: tuple[int, ...] | None = None,
) -> tuple[pl.LazyFrame, tuple[Path, ...], tuple[str, ...] | None]:
    parquet_paths = sentence_dataset_paths(sentence_dataset_dir)
    _required_schema(parquet_paths[0])
    lf = pl.scan_parquet([str(path) for path in parquet_paths]).select(REQUIRED_COLUMNS)
    normalized_items = tuple(dict.fromkeys(item_codes)) if item_codes else None
    if normalized_items:
        lf = lf.filter(pl.col(ITEM_CODE_COLUMN).is_in(normalized_items))
    if years:
        lf = lf.filter(pl.col(YEAR_COLUMN).is_in(years))
    return lf, parquet_paths, normalized_items


def _summary_exprs() -> list[pl.Expr]:
    return [
        pl.len().alias("sentence_rows"),
        pl.col(DOC_ID_COLUMN).n_unique().alias("doc_count"),
        pl.col(TOKEN_COUNT_COLUMN).min().alias("token_min"),
        pl.col(TOKEN_COUNT_COLUMN).median().alias("token_median"),
        pl.col(TOKEN_COUNT_COLUMN).quantile(0.90).alias("token_p90"),
        pl.col(TOKEN_COUNT_COLUMN).quantile(0.95).alias("token_p95"),
        pl.col(TOKEN_COUNT_COLUMN).quantile(0.99).alias("token_p99"),
        pl.col(TOKEN_COUNT_COLUMN).quantile(0.995).alias("token_p995"),
        pl.col(TOKEN_COUNT_COLUMN).max().alias("token_max"),
        pl.col(CHAR_COUNT_COLUMN).min().alias("char_min"),
        pl.col(CHAR_COUNT_COLUMN).median().alias("char_median"),
        pl.col(CHAR_COUNT_COLUMN).quantile(0.90).alias("char_p90"),
        pl.col(CHAR_COUNT_COLUMN).quantile(0.95).alias("char_p95"),
        pl.col(CHAR_COUNT_COLUMN).quantile(0.99).alias("char_p99"),
        pl.col(CHAR_COUNT_COLUMN).max().alias("char_max"),
        (pl.col(TOKEN_COUNT_COLUMN) <= 128).mean().alias("share_token_le_128"),
        (pl.col(TOKEN_COUNT_COLUMN) <= 256).mean().alias("share_token_le_256"),
        (pl.col(TOKEN_COUNT_COLUMN) > 256).mean().alias("share_token_gt_256"),
    ]


def analyze_sentence_lengths(
    sentence_dataset_dir: Path,
    *,
    item_codes: tuple[str, ...] | None = None,
    years: tuple[int, ...] | None = None,
    char_bin_width: int = 25,
    top_n: int = 25,
) -> SentenceLengthAnalysis:
    if char_bin_width <= 0:
        raise ValueError("char_bin_width must be positive.")
    if top_n <= 0:
        raise ValueError("top_n must be positive.")

    lf, parquet_paths, normalized_items = _scan_sentence_lengths(
        sentence_dataset_dir,
        item_codes=item_codes,
        years=years,
    )

    overall = lf.select(_summary_exprs()).collect()
    sentence_rows = int(overall.item(row=0, column="sentence_rows"))
    if sentence_rows == 0:
        raise ValueError("No sentence rows matched the requested filters.")

    summary_by_item = lf.group_by(ITEM_CODE_COLUMN).agg(_summary_exprs()).collect().sort(ITEM_CODE_COLUMN)
    summary_by_year_item = (
        lf.group_by([YEAR_COLUMN, ITEM_CODE_COLUMN])
        .agg(
            [
                pl.len().alias("sentence_rows"),
                pl.col(DOC_ID_COLUMN).n_unique().alias("doc_count"),
                pl.col(TOKEN_COUNT_COLUMN).median().alias("token_median"),
                pl.col(TOKEN_COUNT_COLUMN).quantile(0.95).alias("token_p95"),
                pl.col(CHAR_COUNT_COLUMN).median().alias("char_median"),
                pl.col(CHAR_COUNT_COLUMN).quantile(0.95).alias("char_p95"),
            ]
        )
        .collect()
        .sort([YEAR_COLUMN, ITEM_CODE_COLUMN])
    )
    token_histogram = (
        lf.group_by(TOKEN_COUNT_COLUMN)
        .len()
        .rename({"len": "sentence_rows"})
        .collect()
        .sort(TOKEN_COUNT_COLUMN)
    )
    token_histogram_by_item = (
        lf.group_by([ITEM_CODE_COLUMN, TOKEN_COUNT_COLUMN])
        .len()
        .rename({"len": "sentence_rows"})
        .collect()
        .sort([ITEM_CODE_COLUMN, TOKEN_COUNT_COLUMN])
    )
    char_histogram = (
        lf.with_columns(
            ((pl.col(CHAR_COUNT_COLUMN) // char_bin_width) * char_bin_width).alias("char_bin_start")
        )
        .group_by("char_bin_start")
        .len()
        .rename({"len": "sentence_rows"})
        .collect()
        .sort("char_bin_start")
        .with_columns((pl.col("char_bin_start") + char_bin_width).alias("char_bin_end"))
    )
    char_histogram_by_item = (
        lf.with_columns(
            ((pl.col(CHAR_COUNT_COLUMN) // char_bin_width) * char_bin_width).alias("char_bin_start")
        )
        .group_by([ITEM_CODE_COLUMN, "char_bin_start"])
        .len()
        .rename({"len": "sentence_rows"})
        .collect()
        .sort([ITEM_CODE_COLUMN, "char_bin_start"])
        .with_columns((pl.col("char_bin_start") + char_bin_width).alias("char_bin_end"))
    )
    bucket_counts_by_item = (
        lf.group_by([ITEM_CODE_COLUMN, TOKEN_BUCKET_COLUMN])
        .len()
        .rename({"len": "sentence_rows"})
        .collect()
        .sort([ITEM_CODE_COLUMN, TOKEN_BUCKET_COLUMN])
    )
    longest_sentences = (
        lf.sort(
            [TOKEN_COUNT_COLUMN, CHAR_COUNT_COLUMN, SENTENCE_ID_COLUMN],
            descending=[True, True, False],
        )
        .limit(top_n)
        .collect()
    )

    unique_items = tuple(summary_by_item[ITEM_CODE_COLUMN].to_list())
    metadata: dict[str, Any] = {
        "sentence_dataset_dir": str(normalize_sentence_dataset_dir(sentence_dataset_dir)),
        "parquet_files": [str(path) for path in parquet_paths],
        "parquet_file_count": len(parquet_paths),
        "filters": {
            "item_codes": list(normalized_items) if normalized_items else None,
            "years": list(years) if years else None,
        },
        "char_bin_width": char_bin_width,
        "top_n": top_n,
        "sentence_rows": sentence_rows,
        "doc_count": int(overall.item(row=0, column="doc_count")),
        "item_codes_present": list(unique_items),
        "item_codes_present_ordered": _ordered_item_codes(unique_items),
    }

    return SentenceLengthAnalysis(
        summary_overall=overall,
        summary_by_item=summary_by_item,
        summary_by_year_item=summary_by_year_item,
        token_histogram=token_histogram,
        token_histogram_by_item=token_histogram_by_item,
        char_histogram=char_histogram,
        char_histogram_by_item=char_histogram_by_item,
        bucket_counts_by_item=bucket_counts_by_item,
        longest_sentences=longest_sentences,
        metadata=metadata,
    )


def _frame_to_records(frame: pl.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in frame.to_dicts():
        records.append(
            {
                key: (value.isoformat() if hasattr(value, "isoformat") else value)
                for key, value in row.items()
            }
        )
    return records


def _write_frame_bundle(frame: pl.DataFrame, out_dir: Path, stem: str) -> None:
    frame.write_parquet(out_dir / f"{stem}.parquet", compression="zstd")
    frame.write_csv(out_dir / f"{stem}.csv")


def _save_figure(fig: plt.Figure, out_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _token_axes_title(metadata: dict[str, Any]) -> str:
    filters = metadata["filters"]
    item_codes = filters.get("item_codes")
    years = filters.get("years")
    parts = ["FinBERT Sentence Token Length"]
    if item_codes:
        parts.append("items=" + ", ".join(item_codes))
    if years:
        parts.append("years=" + ", ".join(str(year) for year in years))
    return " | ".join(parts)


def _plot_token_histogram_overall(analysis: SentenceLengthAnalysis, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = analysis.token_histogram[TOKEN_COUNT_COLUMN].to_list()
    y = analysis.token_histogram["sentence_rows"].to_list()
    ax.bar(x, y, width=1.0, color="#0f766e", edgecolor="#115e59", linewidth=0.25)
    ax.set_title(_token_axes_title(analysis.metadata) + " | overall")
    ax.set_xlabel("FinBERT token count (authority 512)")
    ax.set_ylabel("Sentence rows")
    ax.set_xlim(0, 512)
    for edge in (128, 256, 512):
        ax.axvline(edge, color="#9ca3af", linestyle="--", linewidth=1)
    _save_figure(fig, out_path)


def _plot_token_histogram_by_item(analysis: SentenceLengthAnalysis, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    color_map = {"item_1": "#2563eb", "item_1a": "#dc2626", "item_7": "#16a34a"}
    ordered_items = analysis.metadata["item_codes_present_ordered"]
    for item_code in ordered_items:
        item_df = analysis.token_histogram_by_item.filter(pl.col(ITEM_CODE_COLUMN) == item_code)
        ax.step(
            item_df[TOKEN_COUNT_COLUMN].to_list(),
            item_df["sentence_rows"].to_list(),
            where="mid",
            linewidth=1.5,
            label=item_code,
            color=color_map.get(item_code),
        )
    ax.set_title(_token_axes_title(analysis.metadata) + " | by item")
    ax.set_xlabel("FinBERT token count (authority 512)")
    ax.set_ylabel("Sentence rows")
    ax.set_xlim(0, 512)
    ax.legend()
    for edge in (128, 256, 512):
        ax.axvline(edge, color="#d1d5db", linestyle="--", linewidth=1)
    _save_figure(fig, out_path)


def _plot_token_ecdf_by_item(analysis: SentenceLengthAnalysis, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    color_map = {"item_1": "#2563eb", "item_1a": "#dc2626", "item_7": "#16a34a"}
    ordered_items = analysis.metadata["item_codes_present_ordered"]
    for item_code in ordered_items:
        item_df = analysis.token_histogram_by_item.filter(pl.col(ITEM_CODE_COLUMN) == item_code)
        counts = item_df["sentence_rows"].to_list()
        total = sum(counts)
        cumulative: list[float] = []
        running = 0
        for count in counts:
            running += count
            cumulative.append(running / total)
        ax.step(
            item_df[TOKEN_COUNT_COLUMN].to_list(),
            cumulative,
            where="post",
            linewidth=1.8,
            label=item_code,
            color=color_map.get(item_code),
        )
    ax.set_title(_token_axes_title(analysis.metadata) + " | empirical CDF")
    ax.set_xlabel("FinBERT token count (authority 512)")
    ax.set_ylabel("Cumulative share of sentences")
    ax.set_xlim(0, 512)
    ax.set_ylim(0, 1.01)
    ax.legend(loc="lower right")
    for edge in (128, 256, 512):
        ax.axvline(edge, color="#d1d5db", linestyle="--", linewidth=1)
    _save_figure(fig, out_path)


def _plot_char_histogram_overall(analysis: SentenceLengthAnalysis, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = analysis.char_histogram["char_bin_start"].to_list()
    y = analysis.char_histogram["sentence_rows"].to_list()
    width = analysis.metadata["char_bin_width"]
    ax.bar(x, y, width=width, align="edge", color="#7c3aed", edgecolor="#5b21b6", linewidth=0.25)
    ax.set_title(_token_axes_title(analysis.metadata) + " | character count overall")
    ax.set_xlabel(f"Sentence character count (bin width {width})")
    ax.set_ylabel("Sentence rows")
    _save_figure(fig, out_path)


def _plot_char_histogram_by_item(analysis: SentenceLengthAnalysis, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    color_map = {"item_1": "#2563eb", "item_1a": "#dc2626", "item_7": "#16a34a"}
    ordered_items = analysis.metadata["item_codes_present_ordered"]
    for item_code in ordered_items:
        item_df = analysis.char_histogram_by_item.filter(pl.col(ITEM_CODE_COLUMN) == item_code)
        ax.step(
            item_df["char_bin_start"].to_list(),
            item_df["sentence_rows"].to_list(),
            where="post",
            linewidth=1.5,
            label=item_code,
            color=color_map.get(item_code),
        )
    ax.set_title(_token_axes_title(analysis.metadata) + " | character count by item")
    ax.set_xlabel(f"Sentence character count (bin width {analysis.metadata['char_bin_width']})")
    ax.set_ylabel("Sentence rows")
    ax.legend()
    _save_figure(fig, out_path)


def _plot_bucket_share_by_item(analysis: SentenceLengthAnalysis, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ordered_items = analysis.metadata["item_codes_present_ordered"]
    bottom = [0.0] * len(ordered_items)
    color_map = {"short": "#0f766e", "medium": "#d97706", "long": "#b91c1c"}
    for bucket in BUCKET_ORDER:
        heights: list[float] = []
        for item_code in ordered_items:
            item_df = analysis.bucket_counts_by_item.filter(
                (pl.col(ITEM_CODE_COLUMN) == item_code) & (pl.col(TOKEN_BUCKET_COLUMN) == bucket)
            )
            bucket_rows = int(item_df["sentence_rows"][0]) if item_df.height else 0
            total_rows = int(
                analysis.summary_by_item.filter(pl.col(ITEM_CODE_COLUMN) == item_code)["sentence_rows"][0]
            )
            heights.append(bucket_rows / total_rows if total_rows else 0.0)
        ax.bar(ordered_items, heights, bottom=bottom, label=bucket, color=color_map[bucket])
        bottom = [left + right for left, right in zip(bottom, heights)]
    ax.set_title(_token_axes_title(analysis.metadata) + " | token bucket share by item")
    ax.set_ylabel("Share of sentences")
    ax.set_ylim(0, 1.0)
    ax.legend()
    _save_figure(fig, out_path)


def write_sentence_length_report(analysis: SentenceLengthAnalysis, output_dir: Path) -> dict[str, Path]:
    output_dir = output_dir.resolve()
    data_dir = output_dir / "data"
    figures_dir = output_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    _write_frame_bundle(analysis.summary_overall, data_dir, "summary_overall")
    _write_frame_bundle(analysis.summary_by_item, data_dir, "summary_by_item")
    _write_frame_bundle(analysis.summary_by_year_item, data_dir, "summary_by_year_item")
    _write_frame_bundle(analysis.token_histogram, data_dir, "token_histogram_overall")
    _write_frame_bundle(analysis.token_histogram_by_item, data_dir, "token_histogram_by_item")
    _write_frame_bundle(analysis.char_histogram, data_dir, "char_histogram_overall")
    _write_frame_bundle(analysis.char_histogram_by_item, data_dir, "char_histogram_by_item")
    _write_frame_bundle(analysis.bucket_counts_by_item, data_dir, "token_bucket_counts_by_item")
    _write_frame_bundle(analysis.longest_sentences, data_dir, "longest_sentences")

    metadata_path = data_dir / "analysis_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                **analysis.metadata,
                "summary_overall": _frame_to_records(analysis.summary_overall),
                "summary_by_item": _frame_to_records(analysis.summary_by_item),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    token_histogram_overall_path = figures_dir / "token_histogram_overall.png"
    token_histogram_by_item_path = figures_dir / "token_histogram_by_item.png"
    token_ecdf_by_item_path = figures_dir / "token_ecdf_by_item.png"
    char_histogram_overall_path = figures_dir / "char_histogram_overall.png"
    char_histogram_by_item_path = figures_dir / "char_histogram_by_item.png"
    bucket_share_by_item_path = figures_dir / "token_bucket_share_by_item.png"

    _plot_token_histogram_overall(analysis, token_histogram_overall_path)
    _plot_token_histogram_by_item(analysis, token_histogram_by_item_path)
    _plot_token_ecdf_by_item(analysis, token_ecdf_by_item_path)
    _plot_char_histogram_overall(analysis, char_histogram_overall_path)
    _plot_char_histogram_by_item(analysis, char_histogram_by_item_path)
    _plot_bucket_share_by_item(analysis, bucket_share_by_item_path)

    return {
        "output_dir": output_dir,
        "data_dir": data_dir,
        "figures_dir": figures_dir,
        "metadata_json": metadata_path,
        "token_histogram_overall_png": token_histogram_overall_path,
        "token_histogram_by_item_png": token_histogram_by_item_path,
        "token_ecdf_by_item_png": token_ecdf_by_item_path,
        "char_histogram_overall_png": char_histogram_overall_path,
        "char_histogram_by_item_png": char_histogram_by_item_path,
        "token_bucket_share_by_item_png": bucket_share_by_item_path,
    }

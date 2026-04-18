from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import json
from pathlib import Path
import random
from typing import Any

import matplotlib.pyplot as plt
import polars as pl

from thesis_pkg.benchmarking.contracts import DEFAULT_FINBERT_10K_ITEMS
from thesis_pkg.benchmarking.contracts import DEFAULT_FINBERT_AUTHORITY
from thesis_pkg.benchmarking.contracts import BenchmarkItemSpec
from thesis_pkg.benchmarking.contracts import FinbertAuthoritySpec
from thesis_pkg.benchmarking.contracts import FinbertSectionUniverseConfig
from thesis_pkg.benchmarking.contracts import ItemTextCleaningConfig
from thesis_pkg.benchmarking.contracts import SentenceDatasetConfig
from thesis_pkg.benchmarking.finbert_dataset import load_eligible_section_universe
from thesis_pkg.benchmarking.item_text_cleaning import build_segment_policy_id
from thesis_pkg.benchmarking.item_text_cleaning import clean_item_scopes_with_audit
from thesis_pkg.benchmarking.item_text_cleaning import cleaned_scopes_for_sentence_materialization
from thesis_pkg.benchmarking.sentence_length_visualization import analyze_sentence_lengths
from thesis_pkg.benchmarking.sentence_length_visualization import write_sentence_length_report
from thesis_pkg.benchmarking.sentences import _derive_sentence_frame_with_split_audit


DEFAULT_SOURCE_ITEMS_DIR = (
    Path("full_data_run")
    / "sample_5pct_seed42"
    / "results"
    / "sec_ccm_unified_runner"
    / "local_sample"
    / "items_analysis"
)
DEFAULT_ITEM_CODES: tuple[str, ...] = tuple(item.benchmark_item_code for item in DEFAULT_FINBERT_10K_ITEMS)
DEFAULT_POSTPROCESS_POLICY = "item7_reference_stitch_protect_v2"
_ITEM_COLORS = {
    "item_1": "#2563eb",
    "item_1a": "#dc2626",
    "item_7": "#16a34a",
}


@dataclass(frozen=True)
class SampleItemCleaningSentenceDiagnosticsConfig:
    source_items_dir: Path
    output_dir: Path
    sample_doc_count: int = 100
    seed: int = 42
    years: tuple[int, ...] | None = None
    item_codes: tuple[str, ...] = DEFAULT_ITEM_CODES
    selected_doc_ids_path: Path | None = None
    cleaning: ItemTextCleaningConfig = field(default_factory=ItemTextCleaningConfig)
    sentence_dataset: SentenceDatasetConfig = field(
        default_factory=lambda: SentenceDatasetConfig(
            enabled=True,
            postprocess_policy=DEFAULT_POSTPROCESS_POLICY,
        )
    )
    authority: FinbertAuthoritySpec = field(default_factory=lambda: DEFAULT_FINBERT_AUTHORITY)
    char_bin_width: int = 25
    top_n: int = 25

    def __post_init__(self) -> None:
        normalized_years = None
        if self.years is not None:
            normalized_years = tuple(sorted({int(year) for year in self.years}))
            if not normalized_years:
                normalized_years = None
        object.__setattr__(self, "years", normalized_years)

        normalized_item_codes = tuple(dict.fromkeys(code.strip().lower() for code in self.item_codes if code.strip()))
        if not normalized_item_codes:
            raise ValueError("item_codes must contain at least one benchmark item code.")
        invalid_codes = sorted(set(normalized_item_codes) - set(DEFAULT_ITEM_CODES))
        if invalid_codes:
            raise ValueError(
                f"Unsupported item_codes {invalid_codes!r}. Expected a subset of {sorted(DEFAULT_ITEM_CODES)!r}."
            )
        object.__setattr__(self, "item_codes", normalized_item_codes)

        if self.sample_doc_count <= 0:
            raise ValueError("sample_doc_count must be a positive integer.")
        if self.char_bin_width <= 0:
            raise ValueError("char_bin_width must be a positive integer.")
        if self.top_n <= 0:
            raise ValueError("top_n must be a positive integer.")
        if not self.sentence_dataset.enabled:
            raise ValueError("sentence_dataset.enabled must be True for sentence diagnostics.")


@dataclass(frozen=True)
class SampleItemCleaningSentenceDiagnosticsArtifacts:
    output_dir: Path
    selected_doc_ids_path: Path
    sampled_sections_path: Path
    cleaned_item_scopes_path: Path
    cleaning_row_audit_path: Path
    cleaning_flagged_rows_path: Path
    cleaning_diagnostics_path: Path
    manual_audit_sample_path: Path
    sentence_dataset_dir: Path
    sentence_split_audit_path: Path
    item_report_dir: Path
    sentence_report_dir: Path
    summary_path: Path


def _load_selected_doc_ids(path: Path) -> pl.DataFrame:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"selected_doc_ids_path does not exist: {resolved}")
    if resolved.suffix.lower() == ".parquet":
        selected = pl.read_parquet(resolved)
    elif resolved.suffix.lower() == ".csv":
        selected = pl.read_csv(resolved)
    else:
        raise ValueError(
            f"selected_doc_ids_path must be a .parquet or .csv file, got {resolved.name!r}."
        )
    if "doc_id" not in selected.columns:
        raise ValueError("selected_doc_ids_path must contain a doc_id column.")
    return (
        selected.select(pl.col("doc_id").cast(pl.Utf8, strict=False).alias("doc_id"))
        .filter(pl.col("doc_id").is_not_null() & pl.col("doc_id").str.len_chars().gt(0))
        .unique(maintain_order=True)
    )


def _resolve_year_paths(source_items_dir: Path, years: tuple[int, ...] | None) -> list[Path]:
    year_paths = sorted(
        path
        for path in source_items_dir.glob("*.parquet")
        if path.stem.isdigit() and len(path.stem) == 4
    )
    if not year_paths:
        raise FileNotFoundError(f"No year parquet files were found in {source_items_dir}")
    if years is None:
        return year_paths
    selected = [path for path in year_paths if int(path.stem) in set(years)]
    missing = sorted(set(years) - {int(path.stem) for path in selected})
    if missing:
        raise FileNotFoundError(
            f"Requested filing years were not found in {source_items_dir}: {missing}"
        )
    return selected


def _target_items_for_codes(item_codes: tuple[str, ...]) -> tuple[BenchmarkItemSpec, ...]:
    return tuple(item for item in DEFAULT_FINBERT_10K_ITEMS if item.benchmark_item_code in set(item_codes))


def _sample_doc_ids_with_scope_coverage(
    eligible_lf: pl.LazyFrame,
    *,
    item_codes: tuple[str, ...],
    sample_doc_count: int,
    seed: int,
) -> pl.DataFrame:
    doc_item_pairs = (
        eligible_lf.select(["doc_id", "benchmark_item_code"])
        .unique()
        .sort(["doc_id", "benchmark_item_code"])
        .collect()
    )
    unique_doc_ids = sorted(str(doc_id) for doc_id in doc_item_pairs["doc_id"].unique().to_list())
    if not unique_doc_ids:
        return pl.DataFrame({"doc_id": []}, schema={"doc_id": pl.Utf8})
    if sample_doc_count >= len(unique_doc_ids):
        return pl.DataFrame({"doc_id": sorted(unique_doc_ids)}, schema={"doc_id": pl.Utf8})

    selected: list[str] = []
    selected_set: set[str] = set()
    all_doc_ids = unique_doc_ids[:]

    for offset, item_code in enumerate(item_codes):
        if len(selected) >= sample_doc_count:
            break
        candidates = (
            doc_item_pairs.filter(pl.col("benchmark_item_code") == item_code)["doc_id"].unique().to_list()
        )
        candidates = sorted(str(doc_id) for doc_id in candidates if doc_id not in selected_set)
        if not candidates:
            continue
        doc_rng = random.Random(seed + offset)
        chosen_doc_id = doc_rng.choice(candidates)
        selected.append(chosen_doc_id)
        selected_set.add(chosen_doc_id)

    remaining_doc_ids = [doc_id for doc_id in all_doc_ids if doc_id not in selected_set]
    remaining_rng = random.Random(seed + 10_003)
    remaining_rng.shuffle(remaining_doc_ids)
    selected.extend(remaining_doc_ids[: max(sample_doc_count - len(selected), 0)])
    return pl.DataFrame({"doc_id": selected}, schema={"doc_id": pl.Utf8})


def _annotate_sampled_sections(sections_df: pl.DataFrame) -> pl.DataFrame:
    if sections_df.is_empty():
        return sections_df.with_columns(pl.lit(None, dtype=pl.Utf8).alias("benchmark_row_id"))
    return sections_df.with_columns(
        pl.concat_str([pl.col("doc_id"), pl.lit(":"), pl.col("benchmark_item_code")]).alias("benchmark_row_id")
    ).sort(["filing_year", "doc_id", "benchmark_item_code"])


def _write_frame_bundle(df: pl.DataFrame, output_dir: Path, stem: str) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / f"{stem}.parquet"
    csv_path = output_dir / f"{stem}.csv"
    df.write_parquet(parquet_path)
    df.write_csv(csv_path)
    return {"parquet": parquet_path, "csv": csv_path}


def _write_year_shards(sentence_df: pl.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if sentence_df.is_empty():
        return
    years = sorted(int(year) for year in sentence_df["filing_year"].drop_nulls().unique().to_list())
    for year in years:
        sentence_df.filter(pl.col("filing_year") == year).write_parquet(output_dir / f"{year}.parquet")


def _item_summary_by_scope(
    row_audit_df: pl.DataFrame,
    sentence_df: pl.DataFrame,
    split_audit_df: pl.DataFrame,
) -> pl.DataFrame:
    summary = (
        row_audit_df.group_by(["benchmark_item_code", "text_scope"])
        .agg(
            [
                pl.col("doc_id").n_unique().cast(pl.Int64).alias("doc_count"),
                pl.len().cast(pl.Int64).alias("sample_item_rows"),
                (~pl.col("dropped_after_cleaning")).cast(pl.Int64).sum().alias("kept_item_rows"),
                pl.col("dropped_after_cleaning").cast(pl.Int64).sum().alias("dropped_item_rows"),
                pl.col("original_char_count").mean().alias("original_char_mean"),
                pl.col("cleaned_char_count").mean().alias("cleaned_char_mean"),
                pl.col("original_char_count").median().alias("original_char_median"),
                pl.col("cleaned_char_count").median().alias("cleaned_char_median"),
                pl.col("removed_char_count").mean().alias("removed_char_mean"),
                pl.col("removal_ratio").mean().alias("removal_ratio_mean"),
                pl.col("removal_ratio").median().alias("removal_ratio_median"),
                pl.col("warning_large_removal").cast(pl.Int64).sum().alias("large_removal_warning_rows"),
                pl.col("reference_only_stub").cast(pl.Int64).sum().alias("reference_stub_rows"),
                pl.col("toc_prefix_trimmed").cast(pl.Int64).sum().alias("toc_trimmed_rows"),
                pl.col("tail_truncated").cast(pl.Int64).sum().alias("tail_truncated_rows"),
                pl.col("manual_audit_candidate").cast(pl.Int64).sum().alias("manual_audit_candidate_rows"),
                pl.col("item7_lm_token_floor_failed").cast(pl.Int64).sum().alias("item7_lm_token_floor_failed_rows"),
            ]
        )
        .sort("benchmark_item_code")
    )
    if not sentence_df.is_empty():
        sentence_summary = (
            sentence_df.group_by("benchmark_item_code")
            .agg(
                [
                    pl.len().cast(pl.Int64).alias("sentence_rows"),
                    pl.col("doc_id").n_unique().cast(pl.Int64).alias("sentence_doc_count"),
                    pl.col("finbert_token_count_512").median().alias("sentence_token_median"),
                ]
            )
            .sort("benchmark_item_code")
        )
        summary = summary.join(sentence_summary, on="benchmark_item_code", how="left")
    if not split_audit_df.is_empty():
        split_summary = (
            split_audit_df.group_by("benchmark_item_code")
            .agg(
                [
                    pl.len().cast(pl.Int64).alias("split_audit_rows"),
                    pl.col("benchmark_row_id").n_unique().cast(pl.Int64).alias("chunked_item_rows"),
                    pl.col("warning_boundary_used").cast(pl.Int64).sum().alias("warning_split_rows"),
                ]
            )
            .sort("benchmark_item_code")
        )
        summary = summary.join(split_summary, on="benchmark_item_code", how="left")
    return summary.fill_null(0)


def _drop_reason_summary(row_audit_df: pl.DataFrame) -> pl.DataFrame:
    dropped = row_audit_df.filter(pl.col("dropped_after_cleaning"))
    if dropped.is_empty():
        return pl.DataFrame(
            {
                "benchmark_item_code": [],
                "drop_reason": [],
                "item_rows": [],
            },
            schema={
                "benchmark_item_code": pl.Utf8,
                "drop_reason": pl.Utf8,
                "item_rows": pl.Int64,
            },
        )
    return (
        dropped.with_columns(pl.col("drop_reason").fill_null("dropped_unspecified"))
        .group_by(["benchmark_item_code", "drop_reason"])
        .agg(pl.len().cast(pl.Int64).alias("item_rows"))
        .sort(["benchmark_item_code", "drop_reason"])
    )


def _item_rows_by_year(sample_sections_df: pl.DataFrame, sentence_df: pl.DataFrame) -> pl.DataFrame:
    base = (
        sample_sections_df.group_by(["filing_year", "benchmark_item_code"])
        .agg(pl.len().cast(pl.Int64).alias("sample_item_rows"))
        .sort(["filing_year", "benchmark_item_code"])
    )
    if sentence_df.is_empty():
        return base.with_columns(pl.lit(0, dtype=pl.Int64).alias("sentence_rows"))
    sentence_summary = (
        sentence_df.group_by(["filing_year", "benchmark_item_code"])
        .agg(pl.len().cast(pl.Int64).alias("sentence_rows"))
        .sort(["filing_year", "benchmark_item_code"])
    )
    return base.join(sentence_summary, on=["filing_year", "benchmark_item_code"], how="left").fill_null(0)


def _summary_payload(
    cfg: SampleItemCleaningSentenceDiagnosticsConfig,
    *,
    year_paths: list[Path],
    selected_doc_ids_df: pl.DataFrame,
    sample_sections_df: pl.DataFrame,
    cleaning_row_audit_df: pl.DataFrame,
    cleaned_scope_df: pl.DataFrame,
    sentence_df: pl.DataFrame,
    split_audit_df: pl.DataFrame,
    item_summary_by_scope_df: pl.DataFrame,
    sentence_summary_overall: list[dict[str, Any]],
    sentence_summary_by_item: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "source_items_dir": str(cfg.source_items_dir.resolve()),
        "output_dir": str(cfg.output_dir.resolve()),
        "sample_doc_count_requested": cfg.sample_doc_count,
        "sample_doc_count_actual": int(selected_doc_ids_df.height),
        "selected_years": [int(path.stem) for path in year_paths],
        "item_codes": list(cfg.item_codes),
        "seed": cfg.seed,
        "cleaning": {
            "cleaning_policy_id": cfg.cleaning.cleaning_policy_id,
            "enabled": cfg.cleaning.enabled,
        },
        "sentence_dataset": {
            "sentencizer_backend": cfg.sentence_dataset.sentencizer_backend,
            "postprocess_policy": cfg.sentence_dataset.postprocess_policy,
            "bucket_edges": {
                "short_edge": cfg.sentence_dataset.bucket_edges.short_edge,
                "medium_edge": cfg.sentence_dataset.bucket_edges.medium_edge,
            },
            "segment_policy_id": build_segment_policy_id(cfg.sentence_dataset, cfg.cleaning, cfg.authority),
        },
        "authority": {
            "model_name": cfg.authority.model_name,
            "token_count_max_length": cfg.authority.token_count_max_length,
        },
        "counts": {
            "sample_item_rows": int(sample_sections_df.height),
            "cleaned_item_rows_kept": int(cleaned_scope_df.height),
            "cleaning_flagged_rows": int(
                cleaning_row_audit_df.filter(
                    pl.any_horizontal(
                        [
                            pl.col("dropped_after_cleaning"),
                            pl.col("warning_large_removal"),
                            pl.col("toc_prefix_trimmed"),
                            pl.col("tail_truncated"),
                            pl.col("reference_only_stub"),
                            pl.col("item7_lm_token_floor_failed"),
                            pl.col("warning_below_clean_char_count"),
                        ]
                    )
                ).height
            ),
            "manual_audit_candidates": int(
                cleaning_row_audit_df["manual_audit_candidate"].cast(pl.Int64).sum()
                if not cleaning_row_audit_df.is_empty()
                else 0
            ),
            "sentence_rows": int(sentence_df.height),
            "sentence_doc_count": int(sentence_df["doc_id"].n_unique()) if not sentence_df.is_empty() else 0,
            "split_audit_rows": int(split_audit_df.height),
            "chunked_item_rows": int(split_audit_df["benchmark_row_id"].n_unique()) if not split_audit_df.is_empty() else 0,
        },
        "item_summary_by_scope": item_summary_by_scope_df.to_dicts(),
        "sentence_summary_overall": sentence_summary_overall,
        "sentence_summary_by_item": sentence_summary_by_item,
    }


def _save_figure(fig: Any, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _ordered_scope_rows(summary_by_scope_df: pl.DataFrame) -> list[dict[str, Any]]:
    order_lookup = {code: idx for idx, code in enumerate(DEFAULT_ITEM_CODES)}
    return sorted(
        summary_by_scope_df.to_dicts(),
        key=lambda row: order_lookup.get(str(row["benchmark_item_code"]), 999),
    )


def _plot_item_rows_kept_dropped(summary_by_scope_df: pl.DataFrame, out_path: Path) -> None:
    rows = _ordered_scope_rows(summary_by_scope_df)
    labels = [str(row["benchmark_item_code"]) for row in rows]
    kept = [int(row["kept_item_rows"]) for row in rows]
    dropped = [int(row["dropped_item_rows"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.bar(labels, kept, label="kept", color="#2563eb")
    ax.bar(labels, dropped, bottom=kept, label="dropped", color="#dc2626")
    ax.set_title("Sampled item rows kept vs dropped after cleaning")
    ax.set_ylabel("Item rows")
    ax.legend()
    _save_figure(fig, out_path)


def _plot_item_char_medians(summary_by_scope_df: pl.DataFrame, out_path: Path) -> None:
    rows = _ordered_scope_rows(summary_by_scope_df)
    labels = [str(row["benchmark_item_code"]) for row in rows]
    original = [float(row["original_char_median"]) for row in rows]
    cleaned = [float(row["cleaned_char_median"]) for row in rows]
    x_positions = list(range(len(labels)))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.bar([x - width / 2 for x in x_positions], original, width=width, label="original", color="#94a3b8")
    ax.bar([x + width / 2 for x in x_positions], cleaned, width=width, label="cleaned", color="#0f766e")
    ax.set_xticks(x_positions, labels)
    ax.set_title("Median item character count before vs after cleaning")
    ax.set_ylabel("Median character count")
    ax.legend()
    _save_figure(fig, out_path)


def _plot_item_removal_ratio_ecdf(row_audit_df: pl.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ordered_codes = [code for code in DEFAULT_ITEM_CODES if code in set(row_audit_df["benchmark_item_code"].unique().to_list())]
    for item_code in ordered_codes:
        item_df = row_audit_df.filter(pl.col("benchmark_item_code") == item_code).sort("removal_ratio")
        if item_df.is_empty():
            continue
        ratios = [float(value) for value in item_df["removal_ratio"].to_list()]
        y_values = [(idx + 1) / len(ratios) for idx in range(len(ratios))]
        ax.step(
            ratios,
            y_values,
            where="post",
            label=item_code,
            color=_ITEM_COLORS.get(item_code),
            linewidth=1.6,
        )
    ax.set_title("ECDF of item cleaning removal ratio by scope")
    ax.set_xlabel("Removal ratio")
    ax.set_ylabel("Share of sampled items")
    ax.set_xlim(0.0, 1.0)
    ax.legend()
    _save_figure(fig, out_path)


def _plot_item_flag_rates(summary_by_scope_df: pl.DataFrame, out_path: Path) -> None:
    rows = _ordered_scope_rows(summary_by_scope_df)
    labels = [str(row["benchmark_item_code"]) for row in rows]
    flag_columns = [
        ("reference_stub_rows", "reference stubs", "#b91c1c"),
        ("tail_truncated_rows", "tail trunc.", "#d97706"),
        ("toc_trimmed_rows", "toc trimmed", "#0f766e"),
        ("manual_audit_candidate_rows", "manual audit", "#7c3aed"),
    ]
    x_positions = list(range(len(labels)))
    width = 0.18

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    for offset, (column_name, label, color) in enumerate(flag_columns):
        values = []
        for row in rows:
            denom = max(int(row["sample_item_rows"]), 1)
            values.append(float(row[column_name]) / denom)
        ax.bar(
            [x + (offset - 1.5) * width for x in x_positions],
            values,
            width=width,
            label=label,
            color=color,
        )
    ax.set_xticks(x_positions, labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Item cleaning flag rates by scope")
    ax.set_ylabel("Share of sampled item rows")
    ax.legend()
    _save_figure(fig, out_path)


def _plot_item_drop_reasons(drop_reason_df: pl.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    if drop_reason_df.is_empty():
        ax.text(0.5, 0.5, "No dropped item rows in this sample.", ha="center", va="center")
        ax.set_axis_off()
        _save_figure(fig, out_path)
        return

    ordered_codes = [code for code in DEFAULT_ITEM_CODES if code in set(drop_reason_df["benchmark_item_code"].unique().to_list())]
    reasons = sorted(str(value) for value in drop_reason_df["drop_reason"].unique().to_list())
    bottoms = [0 for _ in ordered_codes]
    palette = ["#2563eb", "#dc2626", "#d97706", "#0f766e", "#7c3aed", "#b91c1c"]
    rows = drop_reason_df.to_dicts()
    for idx, reason in enumerate(reasons):
        heights = []
        for item_code in ordered_codes:
            match = next(
                (
                    int(row["item_rows"])
                    for row in rows
                    if row["benchmark_item_code"] == item_code and row["drop_reason"] == reason
                ),
                0,
            )
            heights.append(match)
        ax.bar(
            ordered_codes,
            heights,
            bottom=bottoms,
            label=reason,
            color=palette[idx % len(palette)],
        )
        bottoms = [left + right for left, right in zip(bottoms, heights)]
    ax.set_title("Dropped item rows by drop reason and scope")
    ax.set_ylabel("Dropped item rows")
    ax.legend(fontsize=8)
    _save_figure(fig, out_path)


def write_item_cleaning_report(
    sample_sections_df: pl.DataFrame,
    row_audit_df: pl.DataFrame,
    sentence_df: pl.DataFrame,
    split_audit_df: pl.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir = output_dir.resolve()
    data_dir = output_dir / "data"
    figures_dir = output_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_by_scope_df = _item_summary_by_scope(row_audit_df, sentence_df, split_audit_df)
    drop_reason_df = _drop_reason_summary(row_audit_df)
    rows_by_year_df = _item_rows_by_year(sample_sections_df, sentence_df)

    _write_frame_bundle(summary_by_scope_df, data_dir, "item_summary_by_scope")
    _write_frame_bundle(drop_reason_df, data_dir, "item_drop_reasons_by_scope")
    _write_frame_bundle(rows_by_year_df, data_dir, "item_and_sentence_rows_by_year")

    _plot_item_rows_kept_dropped(summary_by_scope_df, figures_dir / "item_rows_kept_dropped_by_scope.png")
    _plot_item_char_medians(summary_by_scope_df, figures_dir / "item_char_count_median_before_after_by_scope.png")
    _plot_item_removal_ratio_ecdf(row_audit_df, figures_dir / "item_removal_ratio_ecdf_by_scope.png")
    _plot_item_flag_rates(summary_by_scope_df, figures_dir / "item_flag_rates_by_scope.png")
    _plot_item_drop_reasons(drop_reason_df, figures_dir / "item_drop_reasons_by_scope.png")

    return {
        "output_dir": output_dir,
        "data_dir": data_dir,
        "figures_dir": figures_dir,
    }


def run_sample_item_cleaning_sentence_diagnostics(
    cfg: SampleItemCleaningSentenceDiagnosticsConfig,
) -> SampleItemCleaningSentenceDiagnosticsArtifacts:
    source_items_dir = cfg.source_items_dir.resolve()
    output_dir = cfg.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    year_paths = _resolve_year_paths(source_items_dir, cfg.years)
    section_universe = FinbertSectionUniverseConfig(
        source_items_dir=source_items_dir,
        target_items=_target_items_for_codes(cfg.item_codes),
    )
    eligible_lf = load_eligible_section_universe(section_universe, year_paths=year_paths)
    if cfg.selected_doc_ids_path is None:
        selected_doc_ids_df = _sample_doc_ids_with_scope_coverage(
            eligible_lf,
            item_codes=cfg.item_codes,
            sample_doc_count=cfg.sample_doc_count,
            seed=cfg.seed,
        )
    else:
        selected_doc_ids_df = _load_selected_doc_ids(cfg.selected_doc_ids_path)
    if selected_doc_ids_df.is_empty():
        raise ValueError("No eligible documents were available for the requested sampling configuration.")

    sample_sections_df = (
        eligible_lf.join(selected_doc_ids_df.lazy(), on="doc_id", how="inner").collect()
    )
    if sample_sections_df.is_empty():
        raise ValueError("Sampling produced no eligible item rows.")
    sample_sections_df = _annotate_sampled_sections(sample_sections_df)

    sample_dir = output_dir / "sample"
    item_cleaning_dir = output_dir / "item_cleaning"
    sentence_dataset_dir = output_dir / "sentence_dataset" / "by_year"

    selected_doc_ids_paths = _write_frame_bundle(selected_doc_ids_df, sample_dir, "selected_doc_ids")
    sampled_sections_paths = _write_frame_bundle(sample_sections_df, sample_dir, "sampled_sections")

    segment_policy_id = build_segment_policy_id(cfg.sentence_dataset, cfg.cleaning, cfg.authority)
    cleaning_result = clean_item_scopes_with_audit(
        sample_sections_df,
        cfg.cleaning,
        segment_policy_id=segment_policy_id,
    )
    cleaned_item_scopes_paths = _write_frame_bundle(
        cleaning_result.cleaned_scope_df,
        item_cleaning_dir,
        "cleaned_item_scopes",
    )
    cleaning_row_audit_paths = _write_frame_bundle(
        cleaning_result.row_audit_df,
        item_cleaning_dir,
        "cleaning_row_audit",
    )
    cleaning_flagged_rows_paths = _write_frame_bundle(
        cleaning_result.flagged_rows_df,
        item_cleaning_dir,
        "cleaning_flagged_rows",
    )
    cleaning_diagnostics_paths = _write_frame_bundle(
        cleaning_result.scope_diagnostics_df,
        item_cleaning_dir,
        "item_scope_cleaning_diagnostics",
    )
    manual_audit_paths = _write_frame_bundle(
        cleaning_result.manual_audit_sample_df,
        item_cleaning_dir,
        "manual_boundary_audit_sample",
    )

    sentence_sections_df = cleaned_scopes_for_sentence_materialization(cleaning_result.cleaned_scope_df)
    sentence_df, split_audit_df = _derive_sentence_frame_with_split_audit(
        sentence_sections_df,
        cfg.sentence_dataset,
        authority=cfg.authority,
    )
    if sentence_df.is_empty():
        raise ValueError("No sentence rows remained after item cleaning and sentence extraction.")

    _write_year_shards(sentence_df, sentence_dataset_dir)
    sentence_split_audit_paths = _write_frame_bundle(
        split_audit_df,
        output_dir,
        "sentence_split_audit",
    )

    sentence_analysis = analyze_sentence_lengths(
        sentence_dataset_dir,
        item_codes=cfg.item_codes,
        years=cfg.years,
        char_bin_width=cfg.char_bin_width,
        top_n=cfg.top_n,
    )
    sentence_report_artifacts = write_sentence_length_report(
        sentence_analysis,
        output_dir / "sentence_report",
    )
    item_report_artifacts = write_item_cleaning_report(
        sample_sections_df,
        cleaning_result.row_audit_df,
        sentence_df,
        split_audit_df,
        output_dir / "item_report",
    )

    summary = _summary_payload(
        cfg,
        year_paths=year_paths,
        selected_doc_ids_df=selected_doc_ids_df,
        sample_sections_df=sample_sections_df,
        cleaning_row_audit_df=cleaning_result.row_audit_df,
        cleaned_scope_df=cleaning_result.cleaned_scope_df,
        sentence_df=sentence_df,
        split_audit_df=split_audit_df,
        item_summary_by_scope_df=_item_summary_by_scope(cleaning_result.row_audit_df, sentence_df, split_audit_df),
        sentence_summary_overall=sentence_analysis.summary_overall.to_dicts(),
        sentence_summary_by_item=sentence_analysis.summary_by_item.to_dicts(),
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    return SampleItemCleaningSentenceDiagnosticsArtifacts(
        output_dir=output_dir,
        selected_doc_ids_path=selected_doc_ids_paths["parquet"],
        sampled_sections_path=sampled_sections_paths["parquet"],
        cleaned_item_scopes_path=cleaned_item_scopes_paths["parquet"],
        cleaning_row_audit_path=cleaning_row_audit_paths["parquet"],
        cleaning_flagged_rows_path=cleaning_flagged_rows_paths["parquet"],
        cleaning_diagnostics_path=cleaning_diagnostics_paths["parquet"],
        manual_audit_sample_path=manual_audit_paths["parquet"],
        sentence_dataset_dir=sentence_dataset_dir,
        sentence_split_audit_path=sentence_split_audit_paths["parquet"],
        item_report_dir=Path(item_report_artifacts["output_dir"]),
        sentence_report_dir=Path(sentence_report_artifacts["output_dir"]),
        summary_path=summary_path,
    )

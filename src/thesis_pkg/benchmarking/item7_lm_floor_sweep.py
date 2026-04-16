from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import matplotlib
import polars as pl

from thesis_pkg.benchmarking.contracts import ItemTextCleaningConfig
from thesis_pkg.benchmarking.item_text_cleaning import clean_item_scopes_with_audit


matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_ITEM7_FLOOR_THRESHOLDS: tuple[int, ...] = (0, 60, 100, 125, 150, 175, 200, 225, 250, 300)


@dataclass(frozen=True)
class Item7LmFloorSweepArtifacts:
    output_dir: Path
    results_path: Path
    reviewed_case_status_path: Path | None
    summary_path: Path
    figure_paths: tuple[Path, ...]


def _normalize_thresholds(thresholds: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    normalized = tuple(sorted({int(value) for value in thresholds}))
    if not normalized:
        raise ValueError("thresholds must contain at least one value.")
    if normalized[0] < 0:
        raise ValueError("thresholds must be non-negative integers.")
    return normalized


def _load_frame(path: Path) -> pl.DataFrame:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Input path does not exist: {resolved}")
    suffix = resolved.suffix.lower()
    if suffix == ".parquet":
        return pl.read_parquet(resolved)
    if suffix == ".csv":
        return pl.read_csv(resolved, infer_schema_length=10_000)
    raise ValueError(f"Unsupported file type for {resolved.name!r}; expected .parquet or .csv.")


def load_reviewed_removed_segment_cases(review_cases_path: Path) -> pl.DataFrame:
    review_df = _load_frame(review_cases_path)
    required = {"case_type", "benchmark_row_id", "review_label"}
    missing = sorted(required - set(review_df.columns))
    if missing:
        raise ValueError(f"review_cases_path missing required columns: {missing}")
    return (
        review_df.filter(pl.col("case_type") == "removed_segment")
        .select(
            pl.col("benchmark_row_id").cast(pl.Utf8, strict=False),
            pl.col("review_label").cast(pl.Utf8, strict=False),
        )
        .unique(subset=["benchmark_row_id"], keep="first")
    )


def analyze_item7_lm_floor_sweep(
    sampled_sections_df: pl.DataFrame,
    *,
    thresholds: tuple[int, ...] = DEFAULT_ITEM7_FLOOR_THRESHOLDS,
    base_cleaning_cfg: ItemTextCleaningConfig = ItemTextCleaningConfig(),
    reviewed_removed_segment_df: pl.DataFrame | None = None,
    confirmed_false_positive_ids: set[str] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    normalized_thresholds = _normalize_thresholds(thresholds)
    item7_sections = sampled_sections_df.filter(pl.col("benchmark_item_code") == "item_7")
    reviewed_lookup = (
        reviewed_removed_segment_df
        if reviewed_removed_segment_df is not None
        else pl.DataFrame(
            schema={
                "benchmark_row_id": pl.Utf8,
                "review_label": pl.Utf8,
            }
        )
    )
    reviewed_false_positive_ids = set(
        reviewed_lookup.filter(pl.col("review_label") == "false_positive_removal")["benchmark_row_id"].to_list()
        if not reviewed_lookup.is_empty()
        else []
    )
    if confirmed_false_positive_ids is None:
        confirmed_false_positive_ids = reviewed_false_positive_ids
    else:
        confirmed_false_positive_ids = set(confirmed_false_positive_ids) | reviewed_false_positive_ids

    result_rows: list[dict[str, Any]] = []
    reviewed_case_rows: list[dict[str, Any]] = []
    baseline_removed_rows_at_250: int | None = None

    for threshold in normalized_thresholds:
        cleaning_cfg = replace(
            base_cleaning_cfg,
            enforce_item7_lm_token_floor=threshold > 0,
            item7_min_lm_tokens=threshold,
        )
        cleaning_result = clean_item_scopes_with_audit(
            item7_sections,
            cleaning_cfg,
            segment_policy_id="item7_floor_sweep",
        )
        row_audit_df = cleaning_result.row_audit_df.sort("benchmark_row_id")
        dropped_df = row_audit_df.filter(pl.col("dropped_after_cleaning"))
        floor_dropped_df = dropped_df.filter(pl.col("drop_reason") == "item7_below_lm_token_floor")
        reference_stub_dropped_df = dropped_df.filter(pl.col("drop_reason") == "reference_only_stub")
        total_dropped = int(dropped_df.height)
        floor_dropped = int(floor_dropped_df.height)
        if threshold == 250:
            baseline_removed_rows_at_250 = total_dropped

        confirmed_fp_removed = sum(
            1
            for benchmark_row_id in dropped_df["benchmark_row_id"].to_list()
            if benchmark_row_id in confirmed_false_positive_ids
        )

        result_rows.append(
            {
                "item7_min_lm_tokens": threshold,
                "item7_floor_enabled": threshold > 0,
                "sample_item7_rows": int(row_audit_df.height),
                "item7_rows_dropped_total": total_dropped,
                "item7_rows_dropped_by_floor": floor_dropped,
                "item7_rows_dropped_by_reference_stub": int(reference_stub_dropped_df.height),
                "item7_rows_kept": int(row_audit_df.height - total_dropped),
                "confirmed_false_positive_removed_rows": confirmed_fp_removed,
                "confirmed_false_positive_saved_rows": len(confirmed_false_positive_ids) - confirmed_fp_removed,
                "confirmed_fp_share_of_total_dropped": (
                    confirmed_fp_removed / total_dropped if total_dropped else None
                ),
                "confirmed_fp_share_of_floor_dropped": (
                    confirmed_fp_removed / floor_dropped if floor_dropped else None
                ),
            }
        )

        if not reviewed_lookup.is_empty():
            reviewed_status_df = (
                row_audit_df.join(reviewed_lookup, on="benchmark_row_id", how="inner")
                .select(
                    pl.lit(threshold, dtype=pl.Int32).alias("item7_min_lm_tokens"),
                    pl.col("benchmark_row_id"),
                    pl.col("cleaned_lm_total_token_count"),
                    pl.col("dropped_after_cleaning"),
                    pl.col("drop_reason"),
                    pl.col("manual_audit_reason"),
                    pl.col("review_label"),
                )
            )
            reviewed_case_rows.extend(reviewed_status_df.to_dicts())

    results_df = pl.DataFrame(result_rows).sort("item7_min_lm_tokens")
    if baseline_removed_rows_at_250 is None:
        baseline_removed_rows_at_250 = int(
            results_df.filter(
                pl.col("item7_min_lm_tokens") == results_df["item7_min_lm_tokens"].max()
            ).item(0, "item7_rows_dropped_total")
        )
    results_df = results_df.with_columns(
        (pl.lit(baseline_removed_rows_at_250, dtype=pl.Int64) - pl.col("item7_rows_dropped_total")).alias(
            "item7_rows_saved_vs_250"
        )
    )
    reviewed_case_status_df = (
        pl.DataFrame(reviewed_case_rows).sort(["benchmark_row_id", "item7_min_lm_tokens"])
        if reviewed_case_rows
        else pl.DataFrame(
            schema={
                "item7_min_lm_tokens": pl.Int32,
                "benchmark_row_id": pl.Utf8,
                "cleaned_lm_total_token_count": pl.Int32,
                "dropped_after_cleaning": pl.Boolean,
                "drop_reason": pl.Utf8,
                "manual_audit_reason": pl.Utf8,
                "review_label": pl.Utf8,
            }
        )
    )
    return results_df, reviewed_case_status_df


def _save_figure(fig: Any, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_floor_sweep_counts(results_df: pl.DataFrame, out_path: Path) -> None:
    thresholds = results_df["item7_min_lm_tokens"].to_list()
    total_dropped = results_df["item7_rows_dropped_total"].to_list()
    floor_dropped = results_df["item7_rows_dropped_by_floor"].to_list()

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(thresholds, total_dropped, marker="o", linewidth=1.8, color="#2563eb", label="total dropped")
    ax.plot(thresholds, floor_dropped, marker="o", linewidth=1.8, color="#dc2626", label="floor-dropped")
    ax.set_title("Item 7 removed rows vs LM token floor")
    ax.set_xlabel("Item 7 minimum LM tokens")
    ax.set_ylabel("Dropped item rows")
    ax.legend()
    _save_figure(fig, out_path)


def _plot_floor_sweep_false_positive_share(results_df: pl.DataFrame, out_path: Path) -> None:
    thresholds = results_df["item7_min_lm_tokens"].to_list()
    shares = [
        (float(value) * 100.0) if value is not None else 0.0
        for value in results_df["confirmed_fp_share_of_total_dropped"].to_list()
    ]
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(thresholds, shares, marker="o", linewidth=1.8, color="#b91c1c")
    ax.set_title("Confirmed false positives as share of dropped Item 7 rows")
    ax.set_xlabel("Item 7 minimum LM tokens")
    ax.set_ylabel("Confirmed false-positive share (%)")
    _save_figure(fig, out_path)


def write_item7_lm_floor_sweep_report(
    sampled_sections_path: Path,
    output_dir: Path,
    *,
    thresholds: tuple[int, ...] = DEFAULT_ITEM7_FLOOR_THRESHOLDS,
    base_cleaning_cfg: ItemTextCleaningConfig = ItemTextCleaningConfig(),
    review_cases_path: Path | None = None,
    confirmed_false_positive_ids: set[str] | None = None,
) -> Item7LmFloorSweepArtifacts:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sampled_sections_df = _load_frame(sampled_sections_path)
    reviewed_removed_segment_df = (
        load_reviewed_removed_segment_cases(review_cases_path)
        if review_cases_path is not None
        else None
    )
    results_df, reviewed_case_status_df = analyze_item7_lm_floor_sweep(
        sampled_sections_df,
        thresholds=thresholds,
        base_cleaning_cfg=base_cleaning_cfg,
        reviewed_removed_segment_df=reviewed_removed_segment_df,
        confirmed_false_positive_ids=confirmed_false_positive_ids,
    )

    results_path = output_dir / "item7_lm_floor_sweep_results.csv"
    results_df.write_csv(results_path)

    reviewed_case_status_path: Path | None = None
    if not reviewed_case_status_df.is_empty():
        reviewed_case_status_path = output_dir / "item7_lm_floor_sweep_reviewed_case_status.csv"
        reviewed_case_status_df.write_csv(reviewed_case_status_path)

    figure_dir = output_dir / "figures"
    counts_figure_path = figure_dir / "item7_lm_floor_sweep_removed_counts.png"
    fp_figure_path = figure_dir / "item7_lm_floor_sweep_false_positive_share.png"
    _plot_floor_sweep_counts(results_df, counts_figure_path)
    _plot_floor_sweep_false_positive_share(results_df, fp_figure_path)

    summary_payload = {
        "sampled_sections_path": str(sampled_sections_path.resolve()),
        "review_cases_path": str(review_cases_path.resolve()) if review_cases_path is not None else None,
        "confirmed_false_positive_ids": sorted(confirmed_false_positive_ids) if confirmed_false_positive_ids else [],
        "thresholds": list(results_df["item7_min_lm_tokens"].to_list()),
        "base_cleaning_policy_id": base_cleaning_cfg.cleaning_policy_id,
        "floor_enabled_when_threshold_gt_zero": True,
        "results": results_df.to_dicts(),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    return Item7LmFloorSweepArtifacts(
        output_dir=output_dir,
        results_path=results_path,
        reviewed_case_status_path=reviewed_case_status_path,
        summary_path=summary_path,
        figure_paths=(counts_figure_path, fp_figure_path),
    )

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
from dataclasses import asdict
from decimal import Decimal
from decimal import ROUND_HALF_UP
from pathlib import Path
from typing import Any

import polars as pl

from thesis_pkg.benchmarking.contracts import BenchmarkBuildArtifacts
from thesis_pkg.benchmarking.contracts import BenchmarkItemSpec
from thesis_pkg.benchmarking.contracts import BenchmarkSampleSpec
from thesis_pkg.benchmarking.contracts import FinbertBenchmarkSuiteConfig
from thesis_pkg.benchmarking.contracts import SentenceDatasetConfig
from thesis_pkg.benchmarking.contracts import TokenLengthConfig
from thesis_pkg.benchmarking.sentences import _explode_sections_into_sentences
from thesis_pkg.benchmarking.token_lengths import annotate_token_lengths


BENCHMARK_MANIFEST_FILENAME = "benchmark_manifest.json"
SECTION_DATASET_FILENAME = "finbert_10k_item_sections.parquet"
SENTENCE_DATASET_FILENAME = "finbert_10k_item_sentences.parquet"
UNIVERSE_SUMMARY_FILENAME = "universe_summary.json"


def _utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _round_half_up_count(total_rows: int, fraction: float) -> int:
    scaled = Decimal(total_rows) * Decimal(str(fraction))
    return int(scaled.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _sample_dataset_tag(sample_spec: BenchmarkSampleSpec, seed: int) -> str:
    return f"finbert_10k_items_{sample_spec.sample_name}_seed{seed}"


def _item_code_expr(target_items: tuple[BenchmarkItemSpec, ...]) -> pl.Expr:
    expr = pl.lit(None, dtype=pl.Utf8)
    for item in target_items:
        expr = (
            pl.when(pl.col("item_id").cast(pl.Utf8).str.to_uppercase() == item.item_id.upper())
            .then(pl.lit(item.benchmark_item_code))
            .otherwise(expr)
        )
    return expr.alias("benchmark_item_code")


def _item_label_expr(target_items: tuple[BenchmarkItemSpec, ...]) -> pl.Expr:
    expr = pl.lit(None, dtype=pl.Utf8)
    for item in target_items:
        expr = (
            pl.when(pl.col("item_id").cast(pl.Utf8).str.to_uppercase() == item.item_id.upper())
            .then(pl.lit(item.benchmark_item_label))
            .otherwise(expr)
        )
    return expr.alias("benchmark_item_label")


def _validate_sample_specs(sample_specs: tuple[BenchmarkSampleSpec, ...]) -> None:
    if not sample_specs:
        raise ValueError("At least one sample spec is required.")
    seen_names: set[str] = set()
    for spec in sample_specs:
        if spec.sample_name in seen_names:
            raise ValueError(f"Duplicate sample name: {spec.sample_name}")
        if not (0.0 < spec.sample_fraction <= 1.0):
            raise ValueError(f"Sample fraction must be in (0, 1], got {spec.sample_fraction!r}")
        seen_names.add(spec.sample_name)


def _resolve_year_paths(source_items_dir: Path) -> list[Path]:
    year_paths = sorted(
        path for path in source_items_dir.glob("*.parquet") if path.stem.isdigit() and len(path.stem) == 4
    )
    if not year_paths:
        raise FileNotFoundError(f"No year parquet files found in {source_items_dir}")
    return year_paths


def _scan_items_with_year_source(source_items_dir: Path) -> pl.LazyFrame:
    year_paths = _resolve_year_paths(source_items_dir)
    scans = [
        pl.scan_parquet(path).with_columns(pl.lit(int(path.stem)).cast(pl.Int32).alias("source_year_file"))
        for path in year_paths
    ]
    return pl.concat(scans, how="diagonal_relaxed")


def _normalize_cik_expr(schema_names: set[str]) -> pl.Expr:
    if "cik_10" in schema_names:
        return pl.col("cik_10").cast(pl.Utf8, strict=False).str.zfill(10).alias("cik_10")
    if "cik" in schema_names:
        return pl.col("cik").cast(pl.Utf8, strict=False).str.replace_all(r"\D", "").str.zfill(10).alias("cik_10")
    raise ValueError("items_analysis files must contain cik_10 or cik.")


def _normalize_accession_expr(schema_names: set[str]) -> pl.Expr:
    if "accession_nodash" in schema_names:
        return pl.col("accession_nodash").cast(pl.Utf8, strict=False).alias("accession_nodash")
    if "accession_number" in schema_names:
        return (
            pl.col("accession_number")
            .cast(pl.Utf8, strict=False)
            .str.replace_all(r"\D", "")
            .alias("accession_nodash")
        )
    raise ValueError("items_analysis files must contain accession_nodash or accession_number.")


def _normalize_filing_date_expr(schema_names: set[str]) -> pl.Expr:
    if "filing_date" in schema_names and "file_date_filename" in schema_names:
        return pl.coalesce(
            [
                pl.col("filing_date").cast(pl.Date, strict=False),
                pl.col("file_date_filename").cast(pl.Date, strict=False),
            ]
        ).alias("filing_date")
    if "filing_date" in schema_names:
        return pl.col("filing_date").cast(pl.Date, strict=False).alias("filing_date")
    if "file_date_filename" in schema_names:
        return pl.col("file_date_filename").cast(pl.Date, strict=False).alias("filing_date")
    raise ValueError("items_analysis files must contain filing_date or file_date_filename.")


def _normalize_document_type_expr(schema_names: set[str]) -> pl.Expr:
    if "document_type_filename" in schema_names:
        return pl.col("document_type_filename").cast(pl.Utf8, strict=False).alias("document_type")
    if "document_type" in schema_names:
        return pl.col("document_type").cast(pl.Utf8, strict=False).alias("document_type")
    raise ValueError("items_analysis files must contain document_type_filename or document_type.")


def _optional_utf8_expr(column_name: str, schema_names: set[str]) -> pl.Expr:
    if column_name in schema_names:
        return pl.col(column_name).cast(pl.Utf8, strict=False).alias(column_name)
    return pl.lit(None, dtype=pl.Utf8).alias(column_name)


def load_eligible_section_universe(cfg: FinbertBenchmarkSuiteConfig) -> pl.LazyFrame:
    _validate_sample_specs(cfg.sample_specs)
    raw_lf = _scan_items_with_year_source(cfg.source_items_dir)
    schema_names = set(raw_lf.collect_schema().names())
    required_columns = {"doc_id", "item_id", "full_text"}
    missing = sorted(required_columns - schema_names)
    if missing:
        raise ValueError(f"items_analysis files missing required columns: {missing}")
    if cfg.require_active_items and "item_status" not in schema_names:
        raise ValueError("items_analysis files must contain item_status when require_active_items=True.")
    if cfg.require_exists_by_regime and "exists_by_regime" not in schema_names:
        raise ValueError(
            "items_analysis files must contain exists_by_regime when require_exists_by_regime=True."
        )

    lf = raw_lf.with_columns(
        [
            pl.col("doc_id").cast(pl.Utf8, strict=False).alias("doc_id"),
            _normalize_cik_expr(schema_names),
            _normalize_accession_expr(schema_names),
            _normalize_filing_date_expr(schema_names),
            _normalize_document_type_expr(schema_names),
            _optional_utf8_expr("filename", schema_names),
            _optional_utf8_expr("canonical_item", schema_names),
            _optional_utf8_expr("item_part", schema_names),
            _optional_utf8_expr("item_status", schema_names),
            pl.col("item_id").cast(pl.Utf8, strict=False).str.to_uppercase().alias("item_id"),
            pl.col("full_text").cast(pl.Utf8, strict=False).alias("full_text"),
            (
                pl.col("exists_by_regime").cast(pl.Boolean, strict=False)
                if "exists_by_regime" in schema_names
                else pl.lit(True)
            ).alias("exists_by_regime"),
            _item_code_expr(cfg.target_items),
            _item_label_expr(cfg.target_items),
        ]
    )

    filters: list[pl.Expr] = [
        pl.col("benchmark_item_code").is_not_null(),
        pl.col("document_type").is_in(cfg.form_types),
        pl.col("full_text").is_not_null(),
        pl.col("full_text").str.strip_chars().str.len_chars() >= cfg.min_char_count,
    ]
    if cfg.require_active_items:
        filters.append(pl.col("item_status") == "active")
    if cfg.require_exists_by_regime:
        filters.append(pl.col("exists_by_regime"))

    lf = lf.filter(pl.all_horizontal(filters)).with_columns(
        [
            pl.col("filing_date").dt.year().alias("filing_year"),
            pl.col("full_text").str.len_chars().cast(pl.Int32).alias("char_count"),
        ]
    )

    lf = (
        lf.sort(
            by=[
                "doc_id",
                "benchmark_item_code",
                "char_count",
                "canonical_item",
                "filename",
                "accession_nodash",
            ],
            descending=[False, False, True, False, False, False],
            nulls_last=True,
        )
        .unique(subset=["doc_id", "benchmark_item_code"], keep="first", maintain_order=True)
        .select(
            [
                "doc_id",
                "cik_10",
                "accession_nodash",
                "filing_date",
                "filing_year",
                "document_type",
                "benchmark_item_code",
                "benchmark_item_label",
                "item_id",
                "canonical_item",
                "item_part",
                "item_status",
                "exists_by_regime",
                "full_text",
                "char_count",
                "source_year_file",
            ]
        )
    )
    return lf


def _hamilton_apportion(
    keys: list[Any],
    capacities: list[int],
    target_rows: int,
    *,
    ensure_min_one: bool = False,
) -> dict[Any, int]:
    if target_rows < 0:
        raise ValueError("target_rows must be non-negative.")
    if target_rows > sum(capacities):
        raise ValueError("target_rows cannot exceed total capacity.")

    base = [0 for _ in capacities]
    if ensure_min_one and target_rows >= len(capacities):
        for idx, capacity in enumerate(capacities):
            if capacity > 0:
                base[idx] = 1
    assigned = sum(base)
    if assigned > target_rows:
        raise ValueError("Minimum-allocation pass exceeded target_rows.")

    remaining_target = target_rows - assigned
    residual_caps = [capacity - base_count for capacity, base_count in zip(capacities, base)]
    total_residual = sum(residual_caps)
    if remaining_target == 0 or total_residual == 0:
        return {key: base_count for key, base_count in zip(keys, base)}

    exact_rows: list[tuple[Any, Decimal, int, int]] = []
    for idx, (key, residual) in enumerate(zip(keys, residual_caps)):
        exact = Decimal(remaining_target) * Decimal(residual) / Decimal(total_residual)
        exact_rows.append((key, exact, residual, idx))

    floors = {key: int(exact) for key, exact, _residual, _idx in exact_rows}
    allocations = {key: base_count + floors[key] for key, base_count in zip(keys, base)}
    leftover = remaining_target - sum(floors.values())
    ranked = sorted(
        exact_rows,
        key=lambda row: (row[1] - int(row[1]), row[2], -row[3]),
        reverse=True,
    )
    for key, _exact, _residual, _idx in ranked[:leftover]:
        allocations[key] += 1

    return allocations


def _compute_year_allocations_for_target(
    universe_counts_by_year: pl.DataFrame,
    target_rows: int,
    *,
    ensure_all_years_present: bool,
) -> pl.DataFrame:
    counts_df = universe_counts_by_year.sort("filing_year")
    if "eligible_rows" not in counts_df.columns:
        counts_df = counts_df.rename({"len": "eligible_rows", "count": "eligible_rows"})
    years = counts_df["filing_year"].to_list()
    capacities = [int(value) for value in counts_df["eligible_rows"].to_list()]
    allocations = _hamilton_apportion(
        years,
        capacities,
        target_rows,
        ensure_min_one=ensure_all_years_present,
    )
    total_eligible = int(sum(capacities))
    selected = [int(allocations[year]) for year in years]
    return counts_df.with_columns(
        [
            pl.Series("target_rows", selected, dtype=pl.Int32),
            (
                pl.col("eligible_rows").cast(pl.Float64) / pl.lit(total_eligible, dtype=pl.Float64)
                if total_eligible
                else pl.lit(0.0)
            ).alias("eligible_share"),
            (
                pl.Series("target_rows", selected, dtype=pl.Int32).cast(pl.Float64)
                / pl.lit(target_rows, dtype=pl.Float64)
                if target_rows
                else pl.lit(0.0)
            ).alias("target_share"),
        ]
    )


def compute_year_allocations(
    universe_counts_by_year: pl.DataFrame,
    sample_spec: BenchmarkSampleSpec,
    *,
    ensure_all_years_present: bool,
) -> pl.DataFrame:
    total_rows = int(universe_counts_by_year["eligible_rows"].sum())
    target_rows = _round_half_up_count(total_rows, sample_spec.sample_fraction)
    return _compute_year_allocations_for_target(
        universe_counts_by_year,
        target_rows,
        ensure_all_years_present=ensure_all_years_present,
    )


def compute_year_item_allocations(
    universe_counts_by_year_item: pl.DataFrame,
    year_allocations: pl.DataFrame,
) -> pl.DataFrame:
    counts_df = universe_counts_by_year_item.sort(["filing_year", "benchmark_item_code"])
    if "eligible_rows" not in counts_df.columns:
        counts_df = counts_df.rename({"len": "eligible_rows", "count": "eligible_rows"})
    year_targets = {
        int(row["filing_year"]): int(row["target_rows"])
        for row in year_allocations.select(["filing_year", "target_rows"]).to_dicts()
    }

    rows: list[dict[str, Any]] = []
    for group in counts_df.partition_by("filing_year", maintain_order=True):
        filing_year = int(group["filing_year"][0])
        target_rows = int(year_targets.get(filing_year, 0))
        item_codes = group["benchmark_item_code"].to_list()
        capacities = [int(value) for value in group["eligible_rows"].to_list()]
        allocations = _hamilton_apportion(item_codes, capacities, target_rows, ensure_min_one=False)
        total_eligible = int(sum(capacities))
        for item_code, eligible_rows in zip(item_codes, capacities):
            selected_rows = int(allocations[item_code])
            rows.append(
                {
                    "filing_year": filing_year,
                    "benchmark_item_code": item_code,
                    "eligible_rows": eligible_rows,
                    "target_rows": selected_rows,
                    "eligible_share_within_year": (
                        float(eligible_rows / total_eligible) if total_eligible else 0.0
                    ),
                    "target_share_within_year": (
                        float(selected_rows / target_rows) if target_rows else 0.0
                    ),
                }
            )
    return pl.DataFrame(rows).sort(["filing_year", "benchmark_item_code"])


def _selection_key(doc_id: str, benchmark_item_code: str, seed: int) -> str:
    payload = f"{seed}|{doc_id}|{benchmark_item_code}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _annotate_selection_keys(df: pl.DataFrame, seed: int) -> pl.DataFrame:
    keys = [
        _selection_key(doc_id, benchmark_item_code, seed)
        for doc_id, benchmark_item_code in zip(
            df["doc_id"].to_list(),
            df["benchmark_item_code"].to_list(),
        )
    ]
    text_hashes = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in df["full_text"].to_list()]
    benchmark_row_ids = [
        f"{doc_id}:{benchmark_item_code}"
        for doc_id, benchmark_item_code in zip(
            df["doc_id"].to_list(),
            df["benchmark_item_code"].to_list(),
        )
    ]
    return df.with_columns(
        [
            pl.Series("selection_key_hex", keys, dtype=pl.Utf8),
            pl.Series("text_sha256", text_hashes, dtype=pl.Utf8),
            pl.Series("benchmark_row_id", benchmark_row_ids, dtype=pl.Utf8),
        ]
    )


def _select_ranked_from_df(source_df: pl.DataFrame, allocations_df: pl.DataFrame) -> pl.DataFrame:
    if "selection_key_hex" not in source_df.columns:
        raise ValueError("source_df must contain selection_key_hex before selection.")
    if "selection_order" in source_df.columns:
        source_df = source_df.drop("selection_order")

    quota_map = {
        (int(row["filing_year"]), str(row["benchmark_item_code"])): int(row["target_rows"])
        for row in allocations_df.to_dicts()
        if int(row["target_rows"]) > 0
    }
    sorted_source = source_df.sort(
        by=["filing_year", "benchmark_item_code", "selection_key_hex", "doc_id"],
        descending=[False, False, False, False],
    )
    partitions = sorted_source.partition_by(
        ["filing_year", "benchmark_item_code"],
        as_dict=True,
        maintain_order=True,
    )
    selected_frames: list[pl.DataFrame] = []
    for key, target_rows in quota_map.items():
        frame = partitions.get(key)
        if frame is None:
            continue
        if target_rows > frame.height:
            raise ValueError(
                f"Requested {target_rows} rows for {key}, but only {frame.height} rows are available."
            )
        selected_frames.append(frame.head(target_rows))

    if not selected_frames:
        return sorted_source.head(0)

    selected_df = pl.concat(selected_frames, how="vertical_relaxed").sort(
        by=["filing_year", "benchmark_item_code", "selection_key_hex", "doc_id"],
        descending=[False, False, False, False],
    )
    return selected_df.with_row_index(name="selection_order", offset=1)


def select_ranked_section_sample(
    universe_lf: pl.LazyFrame,
    allocations_df: pl.DataFrame,
    *,
    seed: int,
) -> pl.DataFrame:
    source_df = universe_lf.collect()
    source_df = _annotate_selection_keys(source_df, seed)
    return _select_ranked_from_df(source_df, allocations_df)


def _inventory_rows(source_items_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in _resolve_year_paths(source_items_dir):
        row_count = int(pl.scan_parquet(path).select(pl.len()).collect().item())
        rows.append(
            {
                "year_file": int(path.stem),
                "path": str(path.resolve()),
                "row_count": row_count,
            }
        )
    return rows


def _counts_by_year(df: pl.DataFrame) -> pl.DataFrame:
    return df.group_by("filing_year").agg(pl.len().alias("rows")).sort("filing_year")


def _counts_by_year_item(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.group_by(["filing_year", "benchmark_item_code"])
        .agg(pl.len().alias("rows"))
        .sort(["filing_year", "benchmark_item_code"])
    )


def _share_rows(
    eligible_df: pl.DataFrame,
    selected_df: pl.DataFrame,
    keys: list[str],
    *,
    total_keys: list[str] | None = None,
) -> list[dict[str, Any]]:
    eligible_counts = eligible_df.group_by(keys).agg(pl.len().alias("eligible_rows"))
    selected_counts = selected_df.group_by(keys).agg(pl.len().alias("selected_rows"))
    merged = eligible_counts.join(selected_counts, on=keys, how="full").fill_null(0)
    if not total_keys:
        eligible_total = int(eligible_df.height)
        selected_total = int(selected_df.height)
        records: list[dict[str, Any]] = []
        for row in merged.sort(keys).to_dicts():
            eligible_share = (row["eligible_rows"] / eligible_total) if eligible_total else 0.0
            selected_share = (row["selected_rows"] / selected_total) if selected_total else 0.0
            records.append(
                {
                    **row,
                    "eligible_share": eligible_share,
                    "selected_share": selected_share,
                    "share_diff": selected_share - eligible_share,
                }
            )
        return records

    records: list[dict[str, Any]] = []
    for group in merged.partition_by(total_keys, maintain_order=True):
        eligible_total = int(group["eligible_rows"].sum())
        selected_total = int(group["selected_rows"].sum())
        for row in group.sort(keys).to_dicts():
            eligible_share = (row["eligible_rows"] / eligible_total) if eligible_total else 0.0
            selected_share = (row["selected_rows"] / selected_total) if selected_total else 0.0
            records.append(
                {
                    **row,
                    "eligible_share": eligible_share,
                    "selected_share": selected_share,
                    "share_diff": selected_share - eligible_share,
                }
            )
    return records


def _materialize_sentence_dataset(
    sections_df: pl.DataFrame,
    sentence_cfg: SentenceDatasetConfig,
    token_cfg: TokenLengthConfig,
    *,
    out_path: Path,
    compression: str,
) -> Path:
    sentence_df = _explode_sections_into_sentences(sections_df, sentence_cfg, token_cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sentence_df.write_parquet(out_path, compression=compression)
    return out_path


def _serializable_config(cfg: FinbertBenchmarkSuiteConfig, sample_spec: BenchmarkSampleSpec) -> dict[str, Any]:
    return {
        "sample_name": sample_spec.sample_name,
        "sample_fraction": sample_spec.sample_fraction,
        "seed": cfg.seed,
        "compression": cfg.compression,
        "form_types": list(cfg.form_types),
        "target_items": [asdict(item) for item in cfg.target_items],
        "require_active_items": cfg.require_active_items,
        "require_exists_by_regime": cfg.require_exists_by_regime,
        "min_char_count": cfg.min_char_count,
        "ensure_all_years_present": cfg.ensure_all_years_present,
        "nested_samples": cfg.nested_samples,
        "source_items_dir": str(cfg.source_items_dir.resolve()),
        "out_root": str(cfg.out_root.resolve()),
    }


def _write_sample_reports(
    dataset_dir: Path,
    sample_spec: BenchmarkSampleSpec,
    universe_df: pl.DataFrame,
    sample_df: pl.DataFrame,
    year_allocations: pl.DataFrame,
    year_item_allocations: pl.DataFrame,
) -> None:
    reports_dir = dataset_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    eligible_year = _counts_by_year(universe_df)
    selected_year = _counts_by_year(sample_df)
    year_report = (
        eligible_year.rename({"rows": "eligible_rows"})
        .join(selected_year.rename({"rows": "selected_rows"}), on="filing_year", how="left")
        .join(year_allocations.select(["filing_year", "target_rows"]), on="filing_year", how="left")
        .fill_null(0)
        .with_columns(pl.lit(sample_spec.sample_name).alias("sample_name"))
        .with_columns(
            [
                (pl.col("eligible_rows") / pl.lit(int(universe_df.height))).alias("eligible_share"),
                (pl.col("selected_rows") / pl.lit(int(sample_df.height))).alias("selected_share"),
            ]
        )
        .with_columns((pl.col("selected_share") - pl.col("eligible_share")).alias("share_diff"))
        .sort("filing_year")
    )
    year_report.write_csv(reports_dir / "allocation_by_year.csv")

    eligible_year_item = _counts_by_year_item(universe_df)
    selected_year_item = _counts_by_year_item(sample_df)
    year_item_report = (
        eligible_year_item.rename({"rows": "eligible_rows"})
        .join(
            selected_year_item.rename({"rows": "selected_rows"}),
            on=["filing_year", "benchmark_item_code"],
            how="left",
        )
        .join(
            year_item_allocations.select(["filing_year", "benchmark_item_code", "target_rows"]),
            on=["filing_year", "benchmark_item_code"],
            how="left",
        )
        .fill_null(0)
        .with_columns(pl.lit(sample_spec.sample_name).alias("sample_name"))
        .sort(["filing_year", "benchmark_item_code"])
    )
    year_item_report.write_csv(reports_dir / "allocation_by_year_item.csv")

    _write_csv(
        reports_dir / "token_length_audit_overall.csv",
        [
            {"sample_name": sample_spec.sample_name, **row}
            for row in _share_rows(
                universe_df.select(["finbert_token_bucket_512"]),
                sample_df.select(["finbert_token_bucket_512"]),
                ["finbert_token_bucket_512"],
            )
        ],
    )
    _write_csv(
        reports_dir / "token_length_audit_by_item.csv",
        [
            {"sample_name": sample_spec.sample_name, **row}
            for row in _share_rows(
                universe_df.select(["benchmark_item_code", "finbert_token_bucket_512"]),
                sample_df.select(["benchmark_item_code", "finbert_token_bucket_512"]),
                ["benchmark_item_code", "finbert_token_bucket_512"],
                total_keys=["benchmark_item_code"],
            )
        ],
    )
    _write_csv(
        reports_dir / "token_length_audit_by_year.csv",
        [
            {"sample_name": sample_spec.sample_name, **row}
            for row in _share_rows(
                universe_df.select(["filing_year", "finbert_token_bucket_512"]),
                sample_df.select(["filing_year", "finbert_token_bucket_512"]),
                ["filing_year", "finbert_token_bucket_512"],
                total_keys=["filing_year"],
            )
        ],
    )


def _write_universe_summary(dataset_dir: Path, universe_df: pl.DataFrame) -> None:
    payload = {
        "created_at_utc": _utc_timestamp(),
        "eligible_rows": int(universe_df.height),
        "eligible_docs": int(universe_df["doc_id"].n_unique()),
        "year_counts": _counts_by_year(universe_df).to_dicts(),
        "year_item_counts": _counts_by_year_item(universe_df).to_dicts(),
        "item_counts": (
            universe_df.group_by("benchmark_item_code")
            .agg(pl.len().alias("eligible_rows"))
            .sort("benchmark_item_code")
            .to_dicts()
        ),
        "token_bucket_counts": (
            universe_df.group_by("finbert_token_bucket_512")
            .agg(pl.len().alias("eligible_rows"))
            .sort("finbert_token_bucket_512")
            .to_dicts()
        ),
    }
    _write_json(dataset_dir / "reports" / UNIVERSE_SUMMARY_FILENAME, payload)


def _write_inventory(dataset_dir: Path, cfg: FinbertBenchmarkSuiteConfig) -> None:
    rows = _inventory_rows(cfg.source_items_dir)
    inventory_dir = dataset_dir / "inventory"
    _write_json(
        inventory_dir / "source_items_files.json",
        {
            "source_items_dir": str(cfg.source_items_dir.resolve()),
            "year_files": rows,
        },
    )
    _write_csv(inventory_dir / "source_items_files.csv", rows)


def _coerce_output_columns(sample_df: pl.DataFrame) -> pl.DataFrame:
    ordered_columns = [
        "benchmark_row_id",
        "doc_id",
        "cik_10",
        "accession_nodash",
        "filing_date",
        "filing_year",
        "document_type",
        "benchmark_item_code",
        "benchmark_item_label",
        "item_id",
        "canonical_item",
        "item_part",
        "item_status",
        "exists_by_regime",
        "full_text",
        "char_count",
        "finbert_token_count_512",
        "finbert_token_bucket_512",
        "text_sha256",
        "selection_order",
        "source_year_file",
    ]
    return sample_df.select(ordered_columns)


def build_finbert_benchmark_suite(
    cfg: FinbertBenchmarkSuiteConfig,
) -> dict[str, BenchmarkBuildArtifacts]:
    _validate_sample_specs(cfg.sample_specs)
    universe_df = load_eligible_section_universe(cfg).collect()
    universe_df = annotate_token_lengths(universe_df, cfg.token_length)
    universe_df = _annotate_selection_keys(universe_df, cfg.seed)

    if universe_df.is_empty():
        raise ValueError("No eligible benchmark rows found after filtering.")

    spec_by_name = {spec.sample_name: spec for spec in cfg.sample_specs}
    sorted_specs = sorted(cfg.sample_specs, key=lambda spec: spec.sample_fraction, reverse=True)
    max_spec = sorted_specs[0]
    universe_counts_by_year = _counts_by_year(universe_df).rename({"rows": "eligible_rows"})
    universe_counts_by_year_item = _counts_by_year_item(universe_df).rename({"rows": "eligible_rows"})

    selected_frames: dict[str, pl.DataFrame] = {}
    selected_allocations: dict[str, tuple[pl.DataFrame, pl.DataFrame]] = {}

    for spec in sorted_specs:
        target_rows = _round_half_up_count(int(universe_df.height), spec.sample_fraction)
        if spec is max_spec or not cfg.nested_samples:
            source_df = universe_df
            year_counts = universe_counts_by_year
            year_item_counts = universe_counts_by_year_item
        else:
            source_df = selected_frames[max_spec.sample_name]
            year_counts = _counts_by_year(source_df).rename({"rows": "eligible_rows"})
            year_item_counts = _counts_by_year_item(source_df).rename({"rows": "eligible_rows"})

        year_alloc = _compute_year_allocations_for_target(
            year_counts,
            target_rows,
            ensure_all_years_present=cfg.ensure_all_years_present,
        )
        year_item_alloc = compute_year_item_allocations(year_item_counts, year_alloc)
        sample_df = _select_ranked_from_df(source_df, year_item_alloc)
        selected_frames[spec.sample_name] = sample_df
        selected_allocations[spec.sample_name] = (year_alloc, year_item_alloc)

    artifacts: dict[str, BenchmarkBuildArtifacts] = {}
    for sample_name, sample_df in selected_frames.items():
        spec = spec_by_name[sample_name]
        year_alloc, year_item_alloc = selected_allocations[sample_name]
        dataset_tag = _sample_dataset_tag(spec, cfg.seed)
        dataset_dir = cfg.out_root / dataset_tag
        dataset_path = dataset_dir / "dataset" / SECTION_DATASET_FILENAME
        manifest_path = dataset_dir / BENCHMARK_MANIFEST_FILENAME
        sentences_path: Path | None = None

        dataset_dir.mkdir(parents=True, exist_ok=True)
        _write_inventory(dataset_dir, cfg)
        _write_universe_summary(dataset_dir, universe_df)
        _write_sample_reports(dataset_dir, spec, universe_df, sample_df, year_alloc, year_item_alloc)

        output_df = _coerce_output_columns(sample_df)
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.write_parquet(dataset_path, compression=cfg.compression)

        if cfg.sentence_dataset.enabled:
            sentences_path = dataset_dir / "derived" / SENTENCE_DATASET_FILENAME
            _materialize_sentence_dataset(
                output_df,
                cfg.sentence_dataset,
                cfg.token_length,
                out_path=sentences_path,
                compression=cfg.compression,
            )

        manifest = {
            "spec_version": "1.0",
            "dataset_tag": dataset_tag,
            "dataset_name": "finbert_10k_items",
            "created_at_utc": _utc_timestamp(),
            "config": _serializable_config(cfg, spec),
            "eligibility": {
                "document_types": list(cfg.form_types),
                "required_item_status": "active" if cfg.require_active_items else None,
                "require_exists_by_regime": cfg.require_exists_by_regime,
                "min_char_count": cfg.min_char_count,
                "dedupe_key": ["doc_id", "benchmark_item_code"],
                "dedupe_sort": [
                    "doc_id",
                    "benchmark_item_code",
                    "char_count_desc",
                    "canonical_item",
                    "filename",
                    "accession_nodash",
                ],
            },
            "token_length": {
                "tokenizer_name": cfg.token_length.tokenizer_name,
                "max_length": cfg.token_length.max_length,
                "bucket_edges": list(cfg.token_length.bucket_edges),
            },
            "selection": {
                "allocation_policy": "year_then_within_year_item_hamilton",
                "selection_key_fields": ["doc_id", "benchmark_item_code"],
                "selection_hash": "sha256",
                "nested_samples": cfg.nested_samples,
                "nested_from": (
                    _sample_dataset_tag(max_spec, cfg.seed)
                    if cfg.nested_samples and sample_name != max_spec.sample_name
                    else None
                ),
            },
            "counts": {
                "eligible_rows": int(universe_df.height),
                "eligible_docs": int(universe_df["doc_id"].n_unique()),
                "selected_rows": int(output_df.height),
                "selected_docs": int(output_df["doc_id"].n_unique()),
            },
            "artifacts": {
                "sections_path": str(dataset_path.resolve()),
                "sentences_path": str(sentences_path.resolve()) if sentences_path is not None else None,
            },
        }
        _write_json(manifest_path, manifest)

        artifacts[sample_name] = BenchmarkBuildArtifacts(
            dataset_tag=dataset_tag,
            dataset_dir=dataset_dir,
            sections_path=dataset_path,
            sentences_path=sentences_path,
            manifest_path=manifest_path,
            selected_row_count=int(output_df.height),
            selected_doc_count=int(output_df["doc_id"].n_unique()),
        )

    return artifacts

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from thesis_pkg.benchmarking.contracts import BenchmarkSampleSpec
from thesis_pkg.benchmarking.contracts import FinbertBenchmarkSuiteConfig
from thesis_pkg.benchmarking.finbert_dataset import _constrained_hamilton_apportion
from thesis_pkg.benchmarking.finbert_dataset import _tmp_build_root
from thesis_pkg.benchmarking.finbert_dataset import build_finbert_benchmark_suite
from thesis_pkg.benchmarking.finbert_dataset import compute_year_allocations
from thesis_pkg.benchmarking.finbert_dataset import compute_year_item_allocations
from thesis_pkg.benchmarking.finbert_dataset import load_eligible_section_universe
from thesis_pkg.benchmarking.finbert_dataset import select_ranked_section_sample


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(path)


def _long_text(token: str, repeats: int = 80) -> str:
    return (f"{token} " * repeats).strip()


def _fake_annotate_finbert_token_lengths(
    df: pl.DataFrame,
    _authority,
    *,
    text_col: str = "full_text",
    batch_size: int | None = None,
) -> pl.DataFrame:
    del batch_size
    counts = []
    buckets = []
    for text in df[text_col].to_list():
        token_count = min(512, max(1, len(str(text).split())))
        counts.append(token_count)
        if token_count <= 128:
            buckets.append("short")
        elif token_count <= 256:
            buckets.append("medium")
        else:
            buckets.append("long")
    return df.with_columns(
        [
            pl.Series("finbert_token_count_512", counts, dtype=pl.Int32),
            pl.Series("finbert_token_bucket_512", buckets, dtype=pl.Utf8),
        ]
    )


def _sample_rows(
    year: int,
    doc_count: int,
    *,
    include_item_1a: bool,
    include_10k405: bool = False,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx in range(doc_count):
        doc_id = f"{year}:doc:{idx:03d}"
        cik = f"{idx + 1:010d}"
        accession = f"{year}{idx:014d}"
        form = "10-K405" if include_10k405 and idx == 0 else "10-K"
        rows.append(
            {
                "doc_id": doc_id,
                "cik_10": cik,
                "accession_nodash": accession,
                "filing_date": f"{year}-03-01",
                "document_type_filename": form,
                "item_id": "1",
                "canonical_item": "I:1_BUSINESS",
                "item_part": "PART I",
                "item_status": "active",
                "exists_by_regime": True,
                "filename": f"{doc_id}_item1.htm",
                "full_text": _long_text("business"),
            }
        )
        rows.append(
            {
                "doc_id": doc_id,
                "cik_10": cik,
                "accession_nodash": accession,
                "filing_date": f"{year}-03-01",
                "document_type_filename": form,
                "item_id": "7",
                "canonical_item": "II:7_MDA",
                "item_part": "PART II",
                "item_status": "active",
                "exists_by_regime": True,
                "filename": f"{doc_id}_item7.htm",
                "full_text": _long_text("mda"),
            }
        )
        if include_item_1a:
            rows.append(
                {
                    "doc_id": doc_id,
                    "cik_10": cik,
                    "accession_nodash": accession,
                    "filing_date": f"{year}-03-01",
                    "document_type_filename": form,
                    "item_id": "1A",
                    "canonical_item": "I:1A_RISK_FACTORS",
                    "item_part": "PART I",
                    "item_status": "active",
                    "exists_by_regime": True,
                    "filename": f"{doc_id}_item1a.htm",
                    "full_text": _long_text("risk"),
                }
            )

    rows.append(
        {
            "doc_id": f"{year}:invalid:short",
            "cik_10": "0000009999",
            "accession_nodash": f"{year}99999999999999",
            "filing_date": f"{year}-03-01",
            "document_type_filename": "10-K",
            "item_id": "1",
            "canonical_item": "I:1_BUSINESS",
            "item_part": "PART I",
            "item_status": "active",
            "exists_by_regime": True,
            "filename": "short.htm",
            "full_text": "too short for default threshold",
        }
    )
    rows.append(
        {
            "doc_id": f"{year}:invalid:form",
            "cik_10": "0000009998",
            "accession_nodash": f"{year}99999999999998",
            "filing_date": f"{year}-03-01",
            "document_type_filename": "10-Q",
            "item_id": "1",
            "canonical_item": "I:1_BUSINESS",
            "item_part": "PART I",
            "item_status": "active",
            "exists_by_regime": True,
            "filename": "10q.htm",
            "full_text": _long_text("quarter"),
        }
    )
    return rows


def test_compute_year_item_allocations_respects_sparse_early_years() -> None:
    counts = pl.DataFrame(
        {
            "filing_year": [1995, 1995, 2000, 2000, 2000],
            "benchmark_item_code": ["item_1", "item_7", "item_1", "item_1a", "item_7"],
            "eligible_rows": [10, 10, 10, 5, 5],
        }
    )
    year_allocations = pl.DataFrame(
        {
            "filing_year": [1995, 2000],
            "eligible_rows": [20, 20],
            "target_rows": [2, 4],
            "capacity_rows": [20, 20],
        }
    )

    result = compute_year_item_allocations(counts, year_allocations)

    assert result.filter(pl.col("filing_year") == 1995)["benchmark_item_code"].to_list() == ["item_1", "item_7"]
    assert result.filter(pl.col("filing_year") == 1995)["target_rows"].sum() == 2
    assert result.filter(pl.col("filing_year") == 2000)["target_rows"].sum() == 4


def test_constrained_hamilton_respects_parent_capacities() -> None:
    allocations = _constrained_hamilton_apportion(
        ["a", "b", "c"],
        [50, 30, 20],
        [1, 3, 1],
        2,
        ensure_min_one=False,
    )

    assert allocations["a"] <= 1
    assert allocations["b"] <= 3
    assert allocations["c"] <= 1
    assert sum(allocations.values()) == 2


def test_load_eligible_section_universe_uses_default_raw_forms_and_min_char_count(tmp_path: Path) -> None:
    source_dir = tmp_path / "items_analysis"
    _write_parquet(source_dir / "2000.parquet", _sample_rows(2000, 3, include_item_1a=True, include_10k405=True))

    cfg = FinbertBenchmarkSuiteConfig(
        source_items_dir=source_dir,
        out_root=tmp_path / "out",
        sample_specs=(BenchmarkSampleSpec(sample_name="5pct", sample_fraction=0.05),),
    )
    result = load_eligible_section_universe(cfg).collect()

    assert "document_type_raw" in result.columns
    assert "document_type_normalized" in result.columns
    assert result.filter(pl.col("document_type_raw") == "10-K405").height > 0
    assert result.filter(pl.col("char_count") < 250).height == 0
    assert result.filter(pl.col("document_type_raw") == "10-Q").height == 0


def test_build_finbert_benchmark_suite_stages_tokenization_and_keeps_strict_nested_subset(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_dir = tmp_path / "items_analysis"
    _write_parquet(source_dir / "1995.parquet", _sample_rows(1995, 20, include_item_1a=False))
    _write_parquet(source_dir / "1996.parquet", _sample_rows(1996, 20, include_item_1a=False))
    _write_parquet(source_dir / "2000.parquet", _sample_rows(2000, 40, include_item_1a=True, include_10k405=True))

    from thesis_pkg.benchmarking import finbert_dataset

    tokenized_row_counts: list[int] = []

    def _tracking_annotate(
        df: pl.DataFrame,
        authority,
        *,
        text_col: str = "full_text",
        batch_size: int | None = None,
    ) -> pl.DataFrame:
        del authority, text_col, batch_size
        tokenized_row_counts.append(df.height)
        return _fake_annotate_finbert_token_lengths(df, None, text_col="full_text")

    monkeypatch.setattr(finbert_dataset, "annotate_finbert_token_lengths_in_batches", _tracking_annotate)

    cfg = FinbertBenchmarkSuiteConfig(
        source_items_dir=source_dir,
        out_root=tmp_path / "out",
        sample_specs=(
            BenchmarkSampleSpec(sample_name="1pct", sample_fraction=0.01),
            BenchmarkSampleSpec(sample_name="5pct", sample_fraction=0.05),
        ),
        seed=42,
    )
    artifacts = build_finbert_benchmark_suite(cfg)

    sample_5 = pl.read_parquet(artifacts["5pct"].sections_path)
    sample_1 = pl.read_parquet(artifacts["1pct"].sections_path)

    assert sample_5.height == 10
    assert sample_1.height == 2
    assert set(sample_1["benchmark_row_id"].to_list()).issubset(set(sample_5["benchmark_row_id"].to_list()))
    assert sample_5.filter(
        (pl.col("filing_year").is_in([1995, 1996])) & (pl.col("benchmark_item_code") == "item_1a")
    ).height == 0
    assert tokenized_row_counts == [10]

    manifest = json.loads(artifacts["1pct"].manifest_path.read_text(encoding="utf-8"))
    assert manifest["selection"]["nested_policy"] == "strict_nested_constrained_year_then_within_year_item_hamilton"
    assert manifest["selection"]["strict_subset_of_parent"] is True
    assert manifest["token_length_scope"]["selected_rows_tokenized"] is True
    assert manifest["token_length_scope"]["full_universe_token_audit_written"] is False
    assert manifest["eligibility"]["raw_form_allowlist"] == ["10-K", "10-K405"]

    assert (artifacts["5pct"].dataset_dir / "reports" / "sample_token_length_summary.csv").exists()
    assert not (artifacts["5pct"].dataset_dir / "reports" / "token_length_audit_overall.csv").exists()


def test_build_finbert_benchmark_suite_matches_control_selection_for_parent_sample(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_dir = tmp_path / "items_analysis"
    _write_parquet(source_dir / "1995.parquet", _sample_rows(1995, 20, include_item_1a=False))
    _write_parquet(source_dir / "1996.parquet", _sample_rows(1996, 20, include_item_1a=False))
    _write_parquet(source_dir / "2000.parquet", _sample_rows(2000, 40, include_item_1a=True, include_10k405=True))

    from thesis_pkg.benchmarking import finbert_dataset

    monkeypatch.setattr(
        finbert_dataset,
        "annotate_finbert_token_lengths_in_batches",
        _fake_annotate_finbert_token_lengths,
    )

    cfg = FinbertBenchmarkSuiteConfig(
        source_items_dir=source_dir,
        out_root=tmp_path / "out",
        sample_specs=(
            BenchmarkSampleSpec(sample_name="1pct", sample_fraction=0.01),
            BenchmarkSampleSpec(sample_name="5pct", sample_fraction=0.05),
        ),
        seed=42,
    )
    artifacts = build_finbert_benchmark_suite(cfg)

    control_universe = load_eligible_section_universe(cfg)
    control_df = control_universe.collect()
    year_counts = control_df.group_by("filing_year").agg(pl.len().alias("eligible_rows")).sort("filing_year")
    year_item_counts = (
        control_df.group_by(["filing_year", "benchmark_item_code"])
        .agg(pl.len().alias("eligible_rows"))
        .sort(["filing_year", "benchmark_item_code"])
    )
    year_alloc = compute_year_allocations(
        year_counts,
        BenchmarkSampleSpec(sample_name="5pct", sample_fraction=0.05),
        ensure_all_years_present=True,
    )
    year_item_alloc = compute_year_item_allocations(year_item_counts, year_alloc)
    control_selected = (
        select_ranked_section_sample(control_universe, year_item_alloc, seed=42)
        .with_columns(
            pl.concat_str([pl.col("doc_id"), pl.lit(":"), pl.col("benchmark_item_code")]).alias(
                "benchmark_row_id"
            )
        )
        .sort("selection_order")
    )
    built_selected = pl.read_parquet(artifacts["5pct"].sections_path).sort("selection_order")

    assert built_selected["benchmark_row_id"].to_list() == control_selected["benchmark_row_id"].to_list()
    assert built_selected["selection_order"].to_list() == control_selected["selection_order"].to_list()


def test_build_finbert_benchmark_suite_removes_temp_dir_on_success(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_dir = tmp_path / "items_analysis"
    _write_parquet(source_dir / "2000.parquet", _sample_rows(2000, 20, include_item_1a=True, include_10k405=True))

    from thesis_pkg.benchmarking import finbert_dataset

    monkeypatch.setattr(
        finbert_dataset,
        "annotate_finbert_token_lengths_in_batches",
        _fake_annotate_finbert_token_lengths,
    )

    cfg = FinbertBenchmarkSuiteConfig(
        source_items_dir=source_dir,
        out_root=tmp_path / "out",
        sample_specs=(BenchmarkSampleSpec(sample_name="5pct", sample_fraction=0.05),),
        seed=42,
    )
    build_finbert_benchmark_suite(cfg)

    assert not _tmp_build_root(cfg).exists()


def test_build_finbert_benchmark_suite_keeps_temp_dir_on_failure_with_skinny_planning_parquet(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_dir = tmp_path / "items_analysis"
    _write_parquet(source_dir / "2000.parquet", _sample_rows(2000, 20, include_item_1a=True, include_10k405=True))

    from thesis_pkg.benchmarking import finbert_dataset

    monkeypatch.setattr(
        finbert_dataset,
        "annotate_finbert_token_lengths_in_batches",
        _fake_annotate_finbert_token_lengths,
    )
    monkeypatch.setattr(
        finbert_dataset,
        "_annotate_selected_metadata",
        lambda df, authority: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    cfg = FinbertBenchmarkSuiteConfig(
        source_items_dir=source_dir,
        out_root=tmp_path / "out",
        sample_specs=(BenchmarkSampleSpec(sample_name="5pct", sample_fraction=0.05),),
        seed=42,
    )

    with pytest.raises(RuntimeError, match="boom"):
        build_finbert_benchmark_suite(cfg)

    temp_root = _tmp_build_root(cfg)
    planning_path = temp_root / "planning_universe.parquet"
    assert temp_root.exists()
    assert planning_path.exists()
    assert "full_text" not in pl.read_parquet_schema(planning_path).names()

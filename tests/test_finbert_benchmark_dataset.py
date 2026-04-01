from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from thesis_pkg.benchmarking.contracts import BenchmarkSampleSpec
from thesis_pkg.benchmarking.contracts import FinbertBenchmarkSuiteConfig
from thesis_pkg.benchmarking.finbert_dataset import build_finbert_benchmark_suite
from thesis_pkg.benchmarking.finbert_dataset import compute_year_item_allocations


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(path)


def _fake_annotate_token_lengths(df: pl.DataFrame, _cfg) -> pl.DataFrame:
    counts = []
    buckets = []
    for text in df["full_text"].to_list():
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


def _sample_rows(year: int, doc_count: int, *, include_item_1a: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx in range(doc_count):
        doc_id = f"{year}:doc:{idx:03d}"
        cik = f"{idx + 1:010d}"
        accession = f"{year}{idx:014d}"
        rows.append(
            {
                "doc_id": doc_id,
                "cik_10": cik,
                "accession_nodash": accession,
                "filing_date": f"{year}-03-01",
                "document_type_filename": "10-K",
                "item_id": "1",
                "canonical_item": "I:1_BUSINESS",
                "item_part": "PART I",
                "item_status": "active",
                "exists_by_regime": True,
                "filename": f"{doc_id}_item1.htm",
                "full_text": ("business " * 40).strip(),
            }
        )
        rows.append(
            {
                "doc_id": doc_id,
                "cik_10": cik,
                "accession_nodash": accession,
                "filing_date": f"{year}-03-01",
                "document_type_filename": "10-K",
                "item_id": "7",
                "canonical_item": "II:7_MDA",
                "item_part": "PART II",
                "item_status": "active",
                "exists_by_regime": True,
                "filename": f"{doc_id}_item7.htm",
                "full_text": ("mda " * 60).strip(),
            }
        )
        if include_item_1a:
            rows.append(
                {
                    "doc_id": doc_id,
                    "cik_10": cik,
                    "accession_nodash": accession,
                    "filing_date": f"{year}-03-01",
                    "document_type_filename": "10-K",
                    "item_id": "1A",
                    "canonical_item": "I:1A_RISK_FACTORS",
                    "item_part": "PART I",
                    "item_status": "active",
                    "exists_by_regime": True,
                    "filename": f"{doc_id}_item1a.htm",
                    "full_text": ("risk " * 50).strip(),
                }
            )

    rows.append(
        {
            "doc_id": f"{year}:invalid:blank",
            "cik_10": "0000009999",
            "accession_nodash": f"{year}99999999999999",
            "filing_date": f"{year}-03-01",
            "document_type_filename": "10-K",
            "item_id": "1",
            "canonical_item": "I:1_BUSINESS",
            "item_part": "PART I",
            "item_status": "active",
            "exists_by_regime": True,
            "filename": "blank.htm",
            "full_text": "   ",
        }
    )
    rows.append(
        {
            "doc_id": f"{year}:invalid:status",
            "cik_10": "0000009998",
            "accession_nodash": f"{year}99999999999998",
            "filing_date": f"{year}-03-01",
            "document_type_filename": "10-K",
            "item_id": "1",
            "canonical_item": "I:1_BUSINESS",
            "item_part": "PART I",
            "item_status": "reserved",
            "exists_by_regime": True,
            "filename": "reserved.htm",
            "full_text": ("reserved " * 20).strip(),
        }
    )
    rows.append(
        {
            "doc_id": f"{year}:invalid:form",
            "cik_10": "0000009997",
            "accession_nodash": f"{year}99999999999997",
            "filing_date": f"{year}-03-01",
            "document_type_filename": "10-Q",
            "item_id": "1",
            "canonical_item": "I:1_BUSINESS",
            "item_part": "PART I",
            "item_status": "active",
            "exists_by_regime": True,
            "filename": "10q.htm",
            "full_text": ("quarter " * 20).strip(),
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
        }
    )

    result = compute_year_item_allocations(counts, year_allocations)

    assert result.filter(pl.col("filing_year") == 1995)["benchmark_item_code"].to_list() == ["item_1", "item_7"]
    assert result.filter(pl.col("filing_year") == 1995)["target_rows"].sum() == 2
    assert result.filter(pl.col("filing_year") == 2000)["target_rows"].sum() == 4


def test_build_finbert_benchmark_suite_creates_nested_samples_and_reports(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_dir = tmp_path / "items_analysis"
    _write_parquet(source_dir / "1995.parquet", _sample_rows(1995, 10, include_item_1a=False))
    _write_parquet(source_dir / "1996.parquet", _sample_rows(1996, 10, include_item_1a=False))
    _write_parquet(source_dir / "2000.parquet", _sample_rows(2000, 20, include_item_1a=True))

    duplicate_rows = [
        {
            "doc_id": "2000:doc:000",
            "cik_10": "0000000001",
            "accession_nodash": "200000000000000000",
            "filing_date": "2000-03-01",
            "document_type_filename": "10-K",
            "item_id": "1",
            "canonical_item": "I:1_BUSINESS",
            "item_part": "PART I",
            "item_status": "active",
            "exists_by_regime": True,
            "filename": "duplicate_short.htm",
            "full_text": "short duplicate",
        }
    ]
    _write_parquet(source_dir / "2001.parquet", duplicate_rows)

    from thesis_pkg.benchmarking import finbert_dataset

    monkeypatch.setattr(finbert_dataset, "annotate_token_lengths", _fake_annotate_token_lengths)

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

    assert set(artifacts) == {"1pct", "5pct"}

    sample_5 = pl.read_parquet(artifacts["5pct"].sections_path)
    sample_1 = pl.read_parquet(artifacts["1pct"].sections_path)

    assert sample_5.height == 5
    assert sample_1.height == 1
    assert set(sample_1["benchmark_row_id"].to_list()).issubset(set(sample_5["benchmark_row_id"].to_list()))
    assert sample_5["filing_year"].n_unique() == 3
    assert (
        sample_5.filter(
            (pl.col("filing_year").is_in([1995, 1996])) & (pl.col("benchmark_item_code") == "item_1a")
        ).height
        == 0
    )
    assert sample_5.select(pl.col("benchmark_row_id").n_unique()).item() == sample_5.height
    assert "selection_order" in sample_5.columns

    manifest = json.loads(artifacts["5pct"].manifest_path.read_text(encoding="utf-8"))
    assert manifest["selection"]["allocation_policy"] == "year_then_within_year_item_hamilton"
    assert manifest["counts"]["selected_rows"] == 5

    year_item_report = pl.read_csv(artifacts["5pct"].dataset_dir / "reports" / "allocation_by_year_item.csv")
    assert (
        year_item_report.filter(
            (pl.col("filing_year") == 1995) & (pl.col("benchmark_item_code") == "item_1a")
        ).height
        == 0
    )
    assert (artifacts["5pct"].dataset_dir / "reports" / "token_length_audit_overall.csv").exists()
    assert (artifacts["5pct"].dataset_dir / "inventory" / "source_items_files.json").exists()

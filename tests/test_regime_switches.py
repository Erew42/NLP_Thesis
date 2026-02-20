from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl
import pytest

from thesis_pkg.core.sec.extraction import extract_filing_items
from thesis_pkg.core.sec.suspicious_boundary_diagnostics import (
    DiagnosticsConfig,
    InternalHeadingLeak,
    RegressionConfig,
    _embedded_warn_v2,
    _should_escalate_internal_leak_v2,
    run_boundary_comparison,
    run_boundary_diagnostics,
    run_boundary_regression,
)
from thesis_pkg.core.sec.extraction_utils import EmbeddedHeadingHit


FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _load_fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _write_parquet(tmp_path: Path, text: str) -> Path:
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(
        [
            {
                "doc_id": "0000000001:000000000000000001",
                "cik": "0000000001",
                "accession_number": "0000000000-00-000001",
                "document_type_filename": "10-Q",
                "file_date_filename": "20200131",
                "full_text": text,
            }
        ]
    )
    df.write_parquet(parquet_dir / "sample_batch_000.parquet")
    return parquet_dir


def _write_year_parquet(tmp_path: Path, text: str) -> Path:
    parquet_dir = tmp_path / "parquet_year"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(
        [
            {
                "doc_id": "0000000001:000000000000000001",
                "cik": "0000000001",
                "accession_number": "0000000000-00-000001",
                "document_type_filename": "10-Q",
                "file_date_filename": "20200131",
                "full_text": text,
            }
        ]
    )
    df.write_parquet(parquet_dir / "2020.parquet")
    return parquet_dir


def _hash_items(items: list[dict[str, object]]) -> str:
    payload = json.dumps(items, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_legacy_extraction_snapshot() -> None:
    text = _load_fixture("legacy_simple_10q.txt")
    expected = json.loads(
        (FIXTURES / "legacy_simple_10q_expected.json").read_text(encoding="utf-8")
    )
    items = extract_filing_items(
        text,
        form_type="10-Q",
        filing_date="20200131",
        period_end="20191231",
        diagnostics=True,
        extraction_regime="legacy",
    )
    assert items == expected["items"]
    assert _hash_items(items) == expected["hash"]


def test_legacy_diagnostics_snapshot(tmp_path: Path) -> None:
    text = _load_fixture("legacy_simple_10q.txt")
    parquet_dir = _write_parquet(tmp_path, text)

    out_path = tmp_path / "suspicious.csv"
    report_path = tmp_path / "report.txt"
    samples_dir = tmp_path / "samples"
    manifest_items_path = tmp_path / "manifest_items.csv"
    manifest_filings_path = tmp_path / "manifest_filings.csv"
    sample_filings_path = tmp_path / "sample_filings.csv"
    sample_items_path = tmp_path / "sample_items.csv"

    config = DiagnosticsConfig(
        parquet_dir=parquet_dir,
        out_path=out_path,
        report_path=report_path,
        samples_dir=samples_dir,
        batch_size=8,
        max_files=0,
        max_examples=5,
        enable_embedded_verifier=True,
        emit_manifest=True,
        manifest_items_path=manifest_items_path,
        manifest_filings_path=manifest_filings_path,
        sample_pass=0,
        sample_seed=42,
        core_items=("1", "2"),
        target_set=None,
        emit_html=False,
        html_out=tmp_path / "html",
        html_scope="sample",
        extraction_regime="legacy",
        diagnostics_regime="legacy",
    )
    run_boundary_diagnostics(config)

    def _read_csv(path: Path) -> list[dict[str, object]]:
        import csv

        if not path.exists():
            return []
        with path.open("r", newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    actual = {
        "flagged_rows": _read_csv(out_path),
        "manifest_items": _read_csv(manifest_items_path),
        "manifest_filings": _read_csv(manifest_filings_path),
    }
    expected = json.loads(
        (FIXTURES / "legacy_simple_10q_diagnostics_expected.json").read_text(
            encoding="utf-8"
        )
    )
    assert actual == expected


def test_v2_combined_heading_extracts_single_item() -> None:
    text = _load_fixture("v2_combined_heading_10q.txt")
    legacy_items = extract_filing_items(
        text,
        form_type="10-Q",
        filing_date="20200131",
        period_end="20191231",
        diagnostics=True,
        extraction_regime="legacy",
    )
    assert legacy_items == []

    v2_items = extract_filing_items(
        text,
        form_type="10-Q",
        filing_date="20200131",
        period_end="20191231",
        diagnostics=True,
        extraction_regime="v2",
    )
    assert len(v2_items) == 1
    assert v2_items[0]["item_part"] == "I"
    assert v2_items[0]["item_id"] == "1"
    heading_starts = [item.get("_heading_start") for item in v2_items]
    assert len(set(heading_starts)) == len(heading_starts)


def test_v2_part_by_position_separates_10q_items() -> None:
    text = _load_fixture("v2_missing_parts_10q.txt")
    legacy_items = extract_filing_items(
        text,
        form_type="10-Q",
        filing_date="20200131",
        period_end="20191231",
        diagnostics=True,
        extraction_regime="legacy",
    )
    assert len(legacy_items) == 1
    assert legacy_items[0]["item"].startswith("?:")

    v2_items = extract_filing_items(
        text,
        form_type="10-Q",
        filing_date="20200131",
        period_end="20191231",
        diagnostics=True,
        extraction_regime="v2",
    )
    keys = [item["item"] for item in v2_items]
    assert sorted(keys) == ["I:1", "II:1"]
    assert all(not item.get("item_missing_part") for item in v2_items)


def test_v2_embedded_warn_filtering() -> None:
    hits = [
        EmbeddedHeadingHit(
            kind="item",
            classification="toc_row",
            item_id="2",
            part=None,
            line_idx=1,
            char_pos=900,
            full_text_len=2000,
            snippet="ITEM 2 ...",
        )
    ]
    assert not _embedded_warn_v2(hits)
    clustered = hits + [
        EmbeddedHeadingHit(
            kind="item",
            classification="toc_row",
            item_id="3",
            part=None,
            line_idx=2,
            char_pos=950,
            full_text_len=2000,
            snippet="ITEM 3 ...",
        )
    ]
    assert _embedded_warn_v2(clustered)


def test_internal_leak_escalation_v2_successor() -> None:
    leak = InternalHeadingLeak(position=220, match_text="ITEM 2. Properties", context="")
    assert _should_escalate_internal_leak_v2(
        leak_info=leak,
        item_full_text=("A" * 230) + "\nITEM 2. Properties\nMore text.",
        next_item_id="2",
        next_part=None,
        expected_item_ids=set(),
        expected_parts=set(),
    )


def test_internal_leak_escalation_v2_suppresses_non_successor() -> None:
    leak = InternalHeadingLeak(position=220, match_text="ITEM 9. Notes", context="")
    assert not _should_escalate_internal_leak_v2(
        leak_info=leak,
        item_full_text=("A" * 230) + "\nITEM 9. Notes\n12345\n",
        next_item_id="2",
        next_part=None,
        expected_item_ids=set(),
        expected_parts=set(),
    )


def test_run_boundary_comparison_smoke(tmp_path: Path) -> None:
    text = _load_fixture("legacy_simple_10q.txt")
    parquet_dir = _write_parquet(tmp_path, text)

    base_config = DiagnosticsConfig(
        parquet_dir=parquet_dir,
        out_path=tmp_path / "baseline.csv",
        report_path=tmp_path / "baseline.txt",
        samples_dir=tmp_path / "samples",
        batch_size=8,
        max_files=0,
        max_examples=5,
        enable_embedded_verifier=True,
        emit_manifest=True,
        manifest_items_path=tmp_path / "manifest_items.csv",
        manifest_filings_path=tmp_path / "manifest_filings.csv",
        sample_pass=0,
        sample_seed=42,
        core_items=("1", "2"),
        target_set=None,
        emit_html=False,
        html_out=tmp_path / "html",
        html_scope="sample",
        extraction_regime="legacy",
        diagnostics_regime="legacy",
    )
    result = run_boundary_comparison(base_config, out_dir=tmp_path / "compare")
    assert set(result) == {"legacy", "v2", "delta"}
    assert sum(result["legacy"]["status_counts"].values()) == 1
    assert sum(result["v2"]["status_counts"].values()) == 1
    assert result["delta"]["status_counts"].get("PASS", 0) == 0


def test_run_boundary_diagnostics_accepts_yearly_parquet(tmp_path: Path) -> None:
    text = _load_fixture("legacy_simple_10q.txt")
    parquet_dir = _write_year_parquet(tmp_path, text)

    out_path = tmp_path / "suspicious_year.csv"
    report_path = tmp_path / "report_year.txt"
    samples_dir = tmp_path / "samples_year"

    config = DiagnosticsConfig(
        parquet_dir=parquet_dir,
        out_path=out_path,
        report_path=report_path,
        samples_dir=samples_dir,
        batch_size=8,
        max_files=0,
        max_examples=5,
        enable_embedded_verifier=True,
        emit_manifest=False,
        sample_pass=0,
        sample_seed=42,
        core_items=("1", "2"),
        target_set=None,
        emit_html=False,
        html_out=tmp_path / "html_year",
        html_scope="sample",
        extraction_regime="legacy",
        diagnostics_regime="legacy",
    )
    result = run_boundary_diagnostics(config)

    assert out_path.exists()
    assert report_path.exists()
    assert result["total_filings"] >= 1


def test_run_boundary_diagnostics_closes_parquet_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = _load_fixture("legacy_simple_10q.txt")
    parquet_dir = _write_year_parquet(tmp_path, text)
    out_path = tmp_path / "suspicious_year.csv"
    report_path = tmp_path / "report_year.txt"
    samples_dir = tmp_path / "samples_year"

    import thesis_pkg.core.sec.suspicious_boundary_diagnostics as diag_mod

    real_parquet_file = diag_mod.pq.ParquetFile
    closed_count = 0

    class _ParquetFileSpy:
        def __init__(self, path: str | Path):
            self._pf = real_parquet_file(path)

        def iter_batches(self, *args, **kwargs):
            return self._pf.iter_batches(*args, **kwargs)

        def close(self) -> None:
            nonlocal closed_count
            closed_count += 1
            self._pf.close()

    monkeypatch.setattr(diag_mod.pq, "ParquetFile", _ParquetFileSpy)

    config = DiagnosticsConfig(
        parquet_dir=parquet_dir,
        out_path=out_path,
        report_path=report_path,
        samples_dir=samples_dir,
        batch_size=8,
        max_files=0,
        max_examples=5,
        enable_embedded_verifier=True,
        emit_manifest=False,
        sample_pass=0,
        sample_seed=42,
        core_items=("1", "2"),
        target_set=None,
        emit_html=False,
        html_out=tmp_path / "html_year",
        html_scope="sample",
        extraction_regime="legacy",
        diagnostics_regime="legacy",
    )
    run_boundary_diagnostics(config)
    assert closed_count == 1


def test_run_boundary_regression_accepts_yearly_parquet(tmp_path: Path) -> None:
    # Build a baseline CSV from batch-style input.
    baseline_text = _load_fixture("v2_missing_parts_10q.txt")
    baseline_parquet_dir = _write_parquet(tmp_path / "baseline", baseline_text)
    baseline_csv = tmp_path / "baseline_suspicious.csv"
    run_boundary_diagnostics(
        DiagnosticsConfig(
            parquet_dir=baseline_parquet_dir,
            out_path=baseline_csv,
            report_path=tmp_path / "baseline_report.txt",
            samples_dir=tmp_path / "baseline_samples",
            batch_size=8,
            max_files=0,
            max_examples=5,
            enable_embedded_verifier=True,
            emit_manifest=False,
            sample_pass=0,
            sample_seed=42,
            core_items=("1", "2"),
            target_set=None,
            emit_html=False,
            html_out=tmp_path / "baseline_html",
            html_scope="sample",
            extraction_regime="legacy",
            diagnostics_regime="legacy",
        )
    )

    # Regression run uses yearly-style input naming.
    compare_parquet_dir = _write_year_parquet(tmp_path / "compare", baseline_text)
    result = run_boundary_regression(
        RegressionConfig(
            csv_path=baseline_csv,
            parquet_dir=compare_parquet_dir,
            sample_per_flag=1,
            max_files=0,
        )
    )

    assert result["total_filings"] >= 1


def test_run_boundary_regression_closes_parquet_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_text = _load_fixture("v2_missing_parts_10q.txt")
    baseline_parquet_dir = _write_parquet(tmp_path / "baseline", baseline_text)
    baseline_csv = tmp_path / "baseline_suspicious.csv"
    run_boundary_diagnostics(
        DiagnosticsConfig(
            parquet_dir=baseline_parquet_dir,
            out_path=baseline_csv,
            report_path=tmp_path / "baseline_report.txt",
            samples_dir=tmp_path / "baseline_samples",
            batch_size=8,
            max_files=0,
            max_examples=5,
            enable_embedded_verifier=True,
            emit_manifest=False,
            sample_pass=0,
            sample_seed=42,
            core_items=("1", "2"),
            target_set=None,
            emit_html=False,
            html_out=tmp_path / "baseline_html",
            html_scope="sample",
            extraction_regime="legacy",
            diagnostics_regime="legacy",
        )
    )

    compare_parquet_dir = _write_year_parquet(tmp_path / "compare", baseline_text)

    import thesis_pkg.core.sec.suspicious_boundary_diagnostics as diag_mod

    real_parquet_file = diag_mod.pq.ParquetFile
    closed_count = 0

    class _ParquetFileSpy:
        def __init__(self, path: str | Path):
            self._pf = real_parquet_file(path)

        def iter_batches(self, *args, **kwargs):
            return self._pf.iter_batches(*args, **kwargs)

        def close(self) -> None:
            nonlocal closed_count
            closed_count += 1
            self._pf.close()

    monkeypatch.setattr(diag_mod.pq, "ParquetFile", _ParquetFileSpy)

    run_boundary_regression(
        RegressionConfig(
            csv_path=baseline_csv,
            parquet_dir=compare_parquet_dir,
            sample_per_flag=1,
            max_files=0,
        )
    )
    assert closed_count == 1

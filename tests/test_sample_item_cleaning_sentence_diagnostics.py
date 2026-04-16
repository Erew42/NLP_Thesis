from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from thesis_pkg.benchmarking.contracts import SentenceDatasetConfig
from thesis_pkg.benchmarking.finbert_dataset import load_eligible_section_universe
from thesis_pkg.benchmarking.sample_item_cleaning_sentence_diagnostics import _sample_doc_ids_with_scope_coverage
from thesis_pkg.benchmarking.sample_item_cleaning_sentence_diagnostics import (
    SampleItemCleaningSentenceDiagnosticsConfig,
)
from thesis_pkg.benchmarking.sample_item_cleaning_sentence_diagnostics import (
    run_sample_item_cleaning_sentence_diagnostics,
)


class _FakeSpan:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeDoc:
    def __init__(self, text: str) -> None:
        self.sents = [_FakeSpan(part.strip()) for part in text.split(".") if part.strip()]


class _FakeNLP:
    pipe_names = ("sentencizer",)

    def pipe(self, texts, batch_size: int = 0):
        del batch_size
        for text in texts:
            yield _FakeDoc(text)


def _fake_annotate_token_lengths(
    df: pl.DataFrame,
    authority,
    *,
    text_col: str = "sentence_text",
    batch_size: int | None = None,
) -> pl.DataFrame:
    del authority, text_col, batch_size
    return df.with_columns(
        [
            pl.lit(12, dtype=pl.Int32).alias("finbert_token_count_512"),
            pl.lit("short", dtype=pl.Utf8).alias("finbert_token_bucket_512"),
        ]
    )


def _long_text(prefix: str) -> str:
    return (" ".join([prefix, "sentence"]) + ". ") * 60


def _write_items_year(path: Path, *, year: int, doc_suffix: str) -> None:
    doc_id = f"{year}:doc:{doc_suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "doc_id": [doc_id, doc_id, doc_id],
            "cik_10": ["0000000001", "0000000001", "0000000001"],
            "accession_nodash": [f"{year}00000000000001"] * 3,
            "filing_date": [f"{year}-03-01"] * 3,
            "document_type_filename": ["10-K", "10-K", "10-K"],
            "item_id": ["1", "1A", "7"],
            "canonical_item": ["I:1_BUSINESS", "I:1A_RISK_FACTORS", "II:7_MDA"],
            "item_part": ["PART I", "PART I", "PART II"],
            "item_status": ["active", "active", "active"],
            "exists_by_regime": [True, True, True],
            "boundary_authority_status": ["trusted", "trusted", "trusted"],
            "filename": [f"{doc_id}_1.htm", f"{doc_id}_1a.htm", f"{doc_id}_7.htm"],
            "full_text": [
                _long_text("business"),
                _long_text("risk"),
                _long_text("mda"),
            ],
        }
    ).write_parquet(path)


def test_run_sample_item_cleaning_sentence_diagnostics_writes_reports(tmp_path: Path, monkeypatch) -> None:
    from thesis_pkg.benchmarking import sentences

    monkeypatch.setattr(sentences, "_build_sentencizer", lambda cfg: _FakeNLP())
    monkeypatch.setattr(sentences, "annotate_finbert_token_lengths_in_batches", _fake_annotate_token_lengths)

    source_dir = tmp_path / "items_analysis"
    _write_items_year(source_dir / "2006.parquet", year=2006, doc_suffix="001")
    _write_items_year(source_dir / "2007.parquet", year=2007, doc_suffix="002")

    cfg = SampleItemCleaningSentenceDiagnosticsConfig(
        source_items_dir=source_dir,
        output_dir=tmp_path / "out",
        sample_doc_count=2,
        years=(2006, 2007),
        sentence_dataset=SentenceDatasetConfig(
            enabled=True,
            postprocess_policy="item7_reference_stitch_protect_v2",
            spacy_batch_size=1,
            token_length_batch_size=8,
        ),
    )

    artifacts = run_sample_item_cleaning_sentence_diagnostics(cfg)

    assert artifacts.summary_path.exists()
    assert artifacts.cleaned_item_scopes_path.exists()
    assert artifacts.sentence_split_audit_path.exists()
    assert (artifacts.sentence_dataset_dir / "2006.parquet").exists()
    assert (artifacts.item_report_dir / "figures" / "item_rows_kept_dropped_by_scope.png").exists()
    assert (artifacts.sentence_report_dir / "figures" / "token_histogram_overall.png").exists()

    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    assert summary["counts"]["sample_item_rows"] == 6
    assert summary["counts"]["sentence_rows"] > 0
    assert summary["sentence_dataset"]["postprocess_policy"] == "item7_reference_stitch_protect_v2"


def test_sample_doc_ids_with_scope_coverage_is_deterministic(tmp_path: Path) -> None:
    from thesis_pkg.benchmarking.contracts import FinbertSectionUniverseConfig

    source_dir = tmp_path / "items_analysis"
    _write_items_year(source_dir / "2006.parquet", year=2006, doc_suffix="003")
    _write_items_year(source_dir / "2007.parquet", year=2007, doc_suffix="001")
    _write_items_year(source_dir / "2008.parquet", year=2008, doc_suffix="002")

    eligible_lf = load_eligible_section_universe(
        FinbertSectionUniverseConfig(source_items_dir=source_dir),
    )

    first = _sample_doc_ids_with_scope_coverage(
        eligible_lf,
        item_codes=("item_1", "item_1a", "item_7"),
        sample_doc_count=2,
        seed=7,
    )
    second = _sample_doc_ids_with_scope_coverage(
        eligible_lf,
        item_codes=("item_1", "item_1a", "item_7"),
        sample_doc_count=2,
        seed=7,
    )

    assert first["doc_id"].to_list() == second["doc_id"].to_list()


def test_run_sample_item_cleaning_sentence_diagnostics_can_reuse_selected_doc_ids(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from thesis_pkg.benchmarking import sentences

    monkeypatch.setattr(sentences, "_build_sentencizer", lambda cfg: _FakeNLP())
    monkeypatch.setattr(sentences, "annotate_finbert_token_lengths_in_batches", _fake_annotate_token_lengths)

    source_dir = tmp_path / "items_analysis"
    _write_items_year(source_dir / "2006.parquet", year=2006, doc_suffix="001")
    _write_items_year(source_dir / "2007.parquet", year=2007, doc_suffix="002")

    selected_doc_ids_path = tmp_path / "selected_doc_ids.parquet"
    pl.DataFrame({"doc_id": ["2007:doc:002"]}).write_parquet(selected_doc_ids_path)

    cfg = SampleItemCleaningSentenceDiagnosticsConfig(
        source_items_dir=source_dir,
        output_dir=tmp_path / "out_fixed_docs",
        sample_doc_count=2,
        years=(2006, 2007),
        selected_doc_ids_path=selected_doc_ids_path,
        sentence_dataset=SentenceDatasetConfig(
            enabled=True,
            postprocess_policy="item7_reference_stitch_protect_v2",
            spacy_batch_size=1,
            token_length_batch_size=8,
        ),
    )

    artifacts = run_sample_item_cleaning_sentence_diagnostics(cfg)
    selected = pl.read_parquet(artifacts.selected_doc_ids_path)
    assert selected["doc_id"].to_list() == ["2007:doc:002"]

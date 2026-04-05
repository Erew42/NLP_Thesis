from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from thesis_pkg.benchmarking.contracts import BenchmarkSampleSpec
from thesis_pkg.benchmarking.contracts import BucketBatchConfig
from thesis_pkg.benchmarking.contracts import BucketLengthSpec
from thesis_pkg.benchmarking.contracts import FinbertAnalysisRunConfig
from thesis_pkg.benchmarking.contracts import FinbertBenchmarkSuiteConfig
from thesis_pkg.benchmarking.contracts import FinbertSectionUniverseConfig
from thesis_pkg.benchmarking.finbert_analysis import aggregate_sentence_scores_to_item_features
from thesis_pkg.benchmarking.finbert_analysis import build_coverage_report
from thesis_pkg.benchmarking.finbert_analysis import pivot_item_features_to_doc_wide
from thesis_pkg.benchmarking.finbert_analysis import run_finbert_item_analysis
from thesis_pkg.benchmarking.finbert_benchmark import resolve_finbert_label_mapping
from thesis_pkg.benchmarking.finbert_benchmark import score_sentence_frame
from thesis_pkg.benchmarking.finbert_dataset import load_eligible_section_universe


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(path)


def _long_text(token: str, repeats: int = 100) -> str:
    return (f"{token} " * repeats).strip()


def _sample_item_rows(year: int, doc_suffix: str) -> list[dict[str, object]]:
    doc_id = f"{year}:doc:{doc_suffix}"
    return [
        {
            "doc_id": doc_id,
            "cik_10": "0000000001",
            "accession_nodash": f"{year}00000000000001",
            "filing_date": f"{year}-03-01",
            "document_type_filename": "10-K",
            "item_id": "1",
            "canonical_item": "I:1_BUSINESS",
            "item_part": "PART I",
            "item_status": "active",
            "exists_by_regime": True,
            "filename": f"{doc_id}_item1.htm",
            "full_text": _long_text("business"),
        },
        {
            "doc_id": doc_id,
            "cik_10": "0000000001",
            "accession_nodash": f"{year}00000000000001",
            "filing_date": f"{year}-03-01",
            "document_type_filename": "10-K",
            "item_id": "1A",
            "canonical_item": "I:1A_RISK_FACTORS",
            "item_part": "PART I",
            "item_status": "active",
            "exists_by_regime": True,
            "filename": f"{doc_id}_item1a.htm",
            "full_text": _long_text("risk"),
        },
        {
            "doc_id": doc_id,
            "cik_10": "0000000001",
            "accession_nodash": f"{year}00000000000001",
            "filing_date": f"{year}-03-01",
            "document_type_filename": "10-K",
            "item_id": "7",
            "canonical_item": "II:7_MDA",
            "item_part": "PART II",
            "item_status": "active",
            "exists_by_regime": True,
            "filename": f"{doc_id}_item7.htm",
            "full_text": _long_text("mda"),
        },
    ]


class _FakeNoGrad:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        return False


class _FakeAutocast(_FakeNoGrad):
    pass


class _FakeCuda:
    def is_available(self) -> bool:
        return False

    def synchronize(self) -> None:
        return None

    def reset_peak_memory_stats(self) -> None:
        return None

    def max_memory_allocated(self) -> int:
        return 0


class _FakeTorch:
    cuda = _FakeCuda()
    float16 = "float16"
    bfloat16 = "bfloat16"

    @staticmethod
    def no_grad():
        return _FakeNoGrad()

    @staticmethod
    def autocast(*args, **kwargs):
        del args, kwargs
        return _FakeAutocast()

    @staticmethod
    def device(value: str) -> str:
        return value


class _FakeTensor:
    def __init__(self, rows: int) -> None:
        self.rows = rows

    def to(self, device: str):
        del device
        return self


class _FakeTokenizer:
    def __call__(self, texts, **kwargs):
        del kwargs
        return {
            "input_ids": _FakeTensor(len(texts)),
            "attention_mask": _FakeTensor(len(texts)),
        }


class _FakeOutput:
    def __init__(self, logits: list[list[float]]) -> None:
        self.logits = logits


class _FakeModelConfig:
    id2label = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}


class _FakeModel:
    config = _FakeModelConfig()

    def to(self, device: str):
        del device
        return self

    def eval(self):
        return self

    def __call__(self, **kwargs):
        rows = kwargs["input_ids"].rows
        patterns = (
            [3.0, 1.0, 0.5],
            [0.5, 3.0, 1.0],
            [0.5, 1.0, 3.0],
        )
        return _FakeOutput([list(patterns[idx % len(patterns)]) for idx in range(rows)])


def test_shared_section_universe_config_reused_by_benchmark_loader(tmp_path: Path) -> None:
    source_dir = tmp_path / "items_analysis"
    _write_parquet(source_dir / "2006.parquet", _sample_item_rows(2006, "a"))

    universe = FinbertSectionUniverseConfig(source_items_dir=source_dir)
    suite_cfg = FinbertBenchmarkSuiteConfig(
        source_items_dir=tmp_path / "ignored",
        out_root=tmp_path / "out",
        sample_specs=(BenchmarkSampleSpec(sample_name="5pct", sample_fraction=0.05),),
        section_universe=universe,
    )

    shared_rows = load_eligible_section_universe(universe).collect().sort(["doc_id", "benchmark_item_code"])
    suite_rows = load_eligible_section_universe(suite_cfg).collect().sort(["doc_id", "benchmark_item_code"])

    assert suite_cfg.source_items_dir == source_dir
    assert shared_rows.to_dicts() == suite_rows.to_dicts()


def test_resolve_finbert_label_mapping_normalizes_model_config() -> None:
    mapping = resolve_finbert_label_mapping(_FakeModel())

    assert mapping == {0: "negative", 1: "neutral", 2: "positive"}


def test_score_sentence_frame_uses_model_config_labels(tmp_path: Path, monkeypatch) -> None:
    del tmp_path
    from thesis_pkg.benchmarking import finbert_benchmark

    monkeypatch.setattr(finbert_benchmark, "_import_torch", lambda: _FakeTorch())

    sentences_df = pl.DataFrame(
        {
            "benchmark_sentence_id": ["s1", "s2", "s3"],
            "benchmark_row_id": ["r1", "r1", "r2"],
            "doc_id": ["doc1", "doc1", "doc2"],
            "filing_date": [None, None, None],
            "filing_year": [2006, 2006, 2007],
            "benchmark_item_code": ["item_1", "item_1", "item_7"],
            "sentence_index": [0, 1, 0],
            "sentence_text": ["a", "b", "c"],
            "sentence_char_count": [1, 1, 1],
            "sentencizer_backend": ["test", "test", "test"],
            "sentencizer_version": ["test", "test", "test"],
            "finbert_token_count_512": [5, 5, 5],
            "finbert_token_bucket_512": ["short", "short", "short"],
        }
    )

    scored = score_sentence_frame(
        sentences_df,
        _FakeTokenizer(),
        _FakeModel(),
        runtime=FinbertAnalysisRunConfig(
            source_items_dir=Path("unused"),
            out_root=Path("unused"),
            batch_config=BucketBatchConfig(name="test", short_batch_size=2, medium_batch_size=2, long_batch_size=2),
        ).runtime,
        batch_config=BucketBatchConfig(name="test", short_batch_size=2, medium_batch_size=2, long_batch_size=2),
        bucket_lengths=BucketLengthSpec(),
    )

    assert scored["predicted_label"].to_list() == ["negative", "neutral", "negative"]
    assert {"negative_prob", "neutral_prob", "positive_prob"}.issubset(set(scored.columns))


def test_aggregate_sentence_scores_to_item_features() -> None:
    sections_df = pl.DataFrame(
        {
            "benchmark_row_id": ["doc1:item_1", "doc1:item_7"],
            "doc_id": ["doc1", "doc1"],
            "cik_10": ["0000000001", "0000000001"],
            "accession_nodash": ["2006001", "2006001"],
            "filing_date": [None, None],
            "filing_year": [2006, 2006],
            "source_year_file": [2006, 2006],
            "document_type": ["10-K", "10-K"],
            "document_type_raw": ["10-K", "10-K"],
            "document_type_normalized": ["10-K", "10-K"],
            "benchmark_item_code": ["item_1", "item_7"],
            "benchmark_item_label": ["10-K Item 1", "10-K Item 7"],
        }
    )
    sentence_scores_df = pl.DataFrame(
        {
            "benchmark_row_id": ["doc1:item_1", "doc1:item_1", "doc1:item_7"],
            "negative_prob": [0.8, 0.4, 0.2],
            "neutral_prob": [0.1, 0.2, 0.3],
            "positive_prob": [0.1, 0.4, 0.5],
            "predicted_label": ["negative", "positive", "positive"],
        }
    )

    item_features = aggregate_sentence_scores_to_item_features(sentence_scores_df, sections_df).sort(
        "benchmark_item_code"
    )

    item_1 = item_features.filter(pl.col("benchmark_item_code") == "item_1").to_dicts()[0]
    assert item_1["sentence_count"] == 2
    assert round(item_1["negative_prob_mean"], 4) == 0.6
    assert round(item_1["positive_prob_mean"], 4) == 0.25
    assert round(item_1["argmax_share_negative"], 4) == 0.5
    assert round(item_1["sentiment_balance_mean"], 4) == -0.35


def test_pivot_item_features_to_doc_wide() -> None:
    item_features = pl.DataFrame(
        {
            "benchmark_row_id": ["doc1:item_1", "doc1:item_7"],
            "doc_id": ["doc1", "doc1"],
            "cik_10": ["0000000001", "0000000001"],
            "accession_nodash": ["2006001", "2006001"],
            "filing_date": [None, None],
            "filing_year": [2006, 2006],
            "source_year_file": [2006, 2006],
            "document_type": ["10-K", "10-K"],
            "document_type_raw": ["10-K", "10-K"],
            "document_type_normalized": ["10-K", "10-K"],
            "benchmark_item_code": ["item_1", "item_7"],
            "benchmark_item_label": ["10-K Item 1", "10-K Item 7"],
            "sentence_count": [2, 3],
            "negative_prob_mean": [0.6, 0.1],
            "neutral_prob_mean": [0.2, 0.4],
            "positive_prob_mean": [0.2, 0.5],
            "argmax_share_negative": [0.5, 0.0],
            "argmax_share_neutral": [0.0, 0.5],
            "argmax_share_positive": [0.5, 0.5],
            "sentiment_balance_mean": [-0.4, 0.4],
        }
    )

    wide = pivot_item_features_to_doc_wide(item_features)

    assert wide.height == 1
    assert wide["item_1_sentence_count"].to_list() == [2]
    assert wide["item_7_sentiment_balance_mean"].to_list() == [0.4]


def test_build_coverage_report_against_backbone() -> None:
    item_features = pl.DataFrame(
        {
            "doc_id": ["doc1", "doc2"],
            "benchmark_item_code": ["item_1", "item_7"],
        }
    )
    backbone = pl.DataFrame({"doc_id": ["doc1", "doc2", "doc3"]})

    coverage, summary = build_coverage_report(item_features, backbone)

    assert coverage.height == 3
    assert summary["backbone_doc_count"] == 3
    assert summary["covered_doc_count"] == 2
    assert summary["covered_item_1_doc_count"] == 1
    assert summary["covered_item_7_doc_count"] == 1
    assert coverage.filter(pl.col("doc_id") == "doc3")["has_finbert_features"].to_list() == [False]


def test_run_finbert_item_analysis_reuses_existing_year_outputs(tmp_path: Path, monkeypatch) -> None:
    source_dir = tmp_path / "items_analysis"
    _write_parquet(source_dir / "2006.parquet", _sample_item_rows(2006, "a"))
    _write_parquet(source_dir / "2007.parquet", _sample_item_rows(2007, "b"))
    backbone_path = tmp_path / "backbone.parquet"
    pl.DataFrame({"doc_id": ["2006:doc:a", "2007:doc:b", "missing:doc"]}).write_parquet(backbone_path)

    from thesis_pkg.benchmarking import finbert_analysis
    from thesis_pkg.benchmarking import finbert_benchmark

    monkeypatch.setattr(finbert_benchmark, "_import_torch", lambda: _FakeTorch())

    load_counts = {"tokenizer": 0, "model": 0}

    def _fake_load_tokenizer(authority):
        del authority
        load_counts["tokenizer"] += 1
        return _FakeTokenizer()

    def _fake_load_model(authority, runtime):
        del authority, runtime
        load_counts["model"] += 1
        return _FakeModel()

    def _fake_derive_sentence_frame(sections_df: pl.DataFrame, sentence_cfg, *, authority):
        del sentence_cfg, authority
        records: list[dict[str, object]] = []
        for row in sections_df.select(
            [
                "benchmark_row_id",
                "doc_id",
                "filing_date",
                "filing_year",
                "benchmark_item_code",
            ]
        ).iter_rows(named=True):
            for sentence_index in range(2):
                records.append(
                    {
                        "benchmark_sentence_id": f"{row['benchmark_row_id']}:{sentence_index}",
                        "benchmark_row_id": row["benchmark_row_id"],
                        "doc_id": row["doc_id"],
                        "filing_date": row["filing_date"],
                        "filing_year": row["filing_year"],
                        "benchmark_item_code": row["benchmark_item_code"],
                        "sentence_index": sentence_index,
                        "sentence_text": f"sentence {sentence_index}",
                        "sentence_char_count": 10,
                        "sentencizer_backend": "test",
                        "sentencizer_version": "test",
                        "finbert_token_count_512": 5,
                        "finbert_token_bucket_512": "short",
                    }
                )
        return pl.DataFrame(records)

    monkeypatch.setattr(finbert_analysis, "load_finbert_tokenizer", _fake_load_tokenizer)
    monkeypatch.setattr(finbert_analysis, "load_finbert_model", _fake_load_model)
    monkeypatch.setattr(finbert_analysis, "derive_sentence_frame", _fake_derive_sentence_frame)

    run_cfg = FinbertAnalysisRunConfig(
        source_items_dir=source_dir,
        out_root=tmp_path / "runs",
        batch_config=BucketBatchConfig(name="test", short_batch_size=2, medium_batch_size=2, long_batch_size=2),
        section_universe=FinbertSectionUniverseConfig(source_items_dir=source_dir),
        backbone_path=backbone_path,
        year_filter=(2006, 2007),
        write_sentence_scores=True,
        run_name="analysis_smoke",
    )

    artifacts = run_finbert_item_analysis(run_cfg)
    assert artifacts.item_features_long_path.exists()
    assert artifacts.doc_features_wide_path.exists()
    assert artifacts.coverage_report_path is not None and artifacts.coverage_report_path.exists()
    assert load_counts == {"tokenizer": 1, "model": 1}

    second_artifacts = run_finbert_item_analysis(run_cfg)
    assert second_artifacts.run_manifest_path.exists()
    assert load_counts == {"tokenizer": 1, "model": 1}

    manifest = json.loads(second_artifacts.run_manifest_path.read_text(encoding="utf-8"))
    assert manifest["counts"]["reused_year_count"] == 2
    assert all(row["status"] == "reused_existing" for row in manifest["year_results"])

    item_features = pl.read_parquet(second_artifacts.item_features_long_path)
    doc_features = pl.read_parquet(second_artifacts.doc_features_wide_path)
    coverage = pl.read_parquet(second_artifacts.coverage_report_path)

    assert item_features.height == 6
    assert doc_features.height == 2
    assert coverage.height == 3

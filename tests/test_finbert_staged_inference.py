from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from thesis_pkg.benchmarking.contracts import BucketBatchConfig
from thesis_pkg.benchmarking.contracts import FinbertSentenceParquetInferenceRunConfig
from thesis_pkg.benchmarking.contracts import FinbertTokenizerProfileRunConfig
from thesis_pkg.benchmarking.finbert_staged_inference import run_finbert_sentence_parquet_inference
from thesis_pkg.benchmarking.finbert_staged_inference import run_finbert_tokenizer_profile


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


def _write_sentence_year(path: Path, *, year: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc_id = f"{year}:doc:001"
    pl.DataFrame(
        [
            {
                "benchmark_sentence_id": f"{doc_id}:item_1:0",
                "benchmark_row_id": f"{doc_id}:item_1",
                "doc_id": doc_id,
                "cik_10": "0000000001",
                "accession_nodash": f"{year}00000000000001",
                "filing_date": f"{year}-03-01",
                "filing_year": year,
                "benchmark_item_code": "item_1",
                "benchmark_item_label": "10-K Item 1",
                "source_year_file": year,
                "document_type": "10-K",
                "document_type_raw": "10-K",
                "document_type_normalized": "10-K",
                "canonical_item": "I:1_BUSINESS",
                "sentence_index": 0,
                "sentence_text": "business sentence",
                "sentence_char_count": 17,
                "sentencizer_backend": "test",
                "sentencizer_version": "test",
                "finbert_token_count_512": 12,
                "finbert_token_bucket_512": "short",
            },
            {
                "benchmark_sentence_id": f"{doc_id}:item_1:1",
                "benchmark_row_id": f"{doc_id}:item_1",
                "doc_id": doc_id,
                "cik_10": "0000000001",
                "accession_nodash": f"{year}00000000000001",
                "filing_date": f"{year}-03-01",
                "filing_year": year,
                "benchmark_item_code": "item_1",
                "benchmark_item_label": "10-K Item 1",
                "source_year_file": year,
                "document_type": "10-K",
                "document_type_raw": "10-K",
                "document_type_normalized": "10-K",
                "canonical_item": "I:1_BUSINESS",
                "sentence_index": 1,
                "sentence_text": "second business sentence",
                "sentence_char_count": 24,
                "sentencizer_backend": "test",
                "sentencizer_version": "test",
                "finbert_token_count_512": 30,
                "finbert_token_bucket_512": "short",
            },
            {
                "benchmark_sentence_id": f"{doc_id}:item_7:0",
                "benchmark_row_id": f"{doc_id}:item_7",
                "doc_id": doc_id,
                "cik_10": "0000000001",
                "accession_nodash": f"{year}00000000000001",
                "filing_date": f"{year}-03-01",
                "filing_year": year,
                "benchmark_item_code": "item_7",
                "benchmark_item_label": "10-K Item 7",
                "source_year_file": year,
                "document_type": "10-K",
                "document_type_raw": "10-K",
                "document_type_normalized": "10-K",
                "canonical_item": "II:7_MDA",
                "sentence_index": 0,
                "sentence_text": "mda sentence",
                "sentence_char_count": 12,
                "sentencizer_backend": "test",
                "sentencizer_version": "test",
                "finbert_token_count_512": 140,
                "finbert_token_bucket_512": "medium",
            },
        ]
    ).write_parquet(path)


def test_run_finbert_tokenizer_profile_writes_and_reuses_year_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sentence_dataset_dir = tmp_path / "sentence_dataset" / "by_year"
    _write_sentence_year(sentence_dataset_dir / "2006.parquet", year=2006)

    from thesis_pkg.benchmarking import finbert_staged_inference

    monkeypatch.setattr(finbert_staged_inference, "load_finbert_tokenizer", lambda authority: _FakeTokenizer())

    run_cfg = FinbertTokenizerProfileRunConfig(
        sentence_dataset_dir=sentence_dataset_dir,
        out_root=tmp_path / "runs",
        batch_config=BucketBatchConfig(name="baseline", short_batch_size=2, medium_batch_size=2, long_batch_size=1),
        profile_row_cap_per_bucket=2,
        run_name="tokenizer_profile_smoke",
    )

    artifacts = run_finbert_tokenizer_profile(run_cfg)
    assert artifacts.bucket_summary_path.exists()
    assert artifacts.timing_summary_path.exists()

    bucket_summary = pl.read_parquet(artifacts.bucket_summary_path)
    timing_summary = pl.read_parquet(artifacts.timing_summary_path)
    assert bucket_summary["sentence_rows"].to_list() == [2, 1, 0]
    assert timing_summary["sampled_rows"].to_list() == [2, 1, 0]

    second_artifacts = run_finbert_tokenizer_profile(run_cfg)
    manifest = json.loads(second_artifacts.run_manifest_path.read_text(encoding="utf-8"))
    assert manifest["counts"]["reused_year_count"] == 1


def test_run_finbert_sentence_parquet_inference_uses_precomputed_sentence_dataset_only(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sentence_dataset_dir = tmp_path / "sentence_dataset" / "by_year"
    _write_sentence_year(sentence_dataset_dir / "2006.parquet", year=2006)
    backbone_path = tmp_path / "backbone.parquet"
    pl.DataFrame({"doc_id": ["2006:doc:001", "missing:doc"]}).write_parquet(backbone_path)

    from thesis_pkg.benchmarking import finbert_benchmark
    from thesis_pkg.benchmarking import finbert_staged_inference
    from thesis_pkg.benchmarking import sentences

    monkeypatch.setattr(finbert_benchmark, "_import_torch", lambda: _FakeTorch())
    monkeypatch.setattr(finbert_staged_inference, "load_finbert_tokenizer", lambda authority: _FakeTokenizer())
    monkeypatch.setattr(finbert_staged_inference, "load_finbert_model", lambda authority, runtime: _FakeModel())
    monkeypatch.setattr(
        sentences,
        "derive_sentence_frame",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("model stage must not resplit sentences")),
    )

    artifacts = run_finbert_sentence_parquet_inference(
        FinbertSentenceParquetInferenceRunConfig(
            sentence_dataset_dir=sentence_dataset_dir,
            out_root=tmp_path / "runs",
            batch_config=BucketBatchConfig(name="baseline", short_batch_size=2, medium_batch_size=1, long_batch_size=1),
            backbone_path=backbone_path,
            sentence_slice_rows=2,
            write_sentence_scores=True,
            run_name="sentence_parquet_inference_smoke",
        )
    )

    item_features = pl.read_parquet(artifacts.item_features_long_path)
    doc_features = pl.read_parquet(artifacts.doc_features_wide_path)
    coverage = pl.read_parquet(artifacts.coverage_report_path)
    sentence_scores = pl.read_parquet(artifacts.sentence_scores_dir / "2006.parquet")
    manifest = json.loads(artifacts.run_manifest_path.read_text(encoding="utf-8"))

    assert item_features.height == 2
    item_1 = item_features.filter(pl.col("benchmark_item_code") == "item_1").row(0, named=True)
    assert item_1["text_scope"] == "item_1"
    assert item_1["model_name"] == "yiyanghkust/finbert-tone"
    assert item_1["segment_policy_id"] == "sentence_dataset_v1_finbert_token_512"
    assert item_1["finbert_segment_count"] == 2
    assert item_1["finbert_token_count_512_sum"] == 42
    assert "finbert_neg_prob_lenw_mean" in item_features.columns
    assert doc_features.height == 1
    assert "item_1_finbert_neg_prob_lenw_mean" in doc_features.columns
    assert coverage.height == 2
    assert sentence_scores.height == 3
    assert sentence_scores["predicted_label"].to_list() == ["negative", "neutral", "negative"]
    assert manifest["counts"]["processed_year_count"] == 1
    assert manifest["item_feature_contract"]["accepted_unit"] == ["doc_id", "benchmark_item_code"]
    assert manifest["item_feature_contract"]["denominator_contract"]["length_weight_column"] == "finbert_token_count_512"


def test_run_finbert_sentence_parquet_inference_rejects_changed_sentence_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    preprocess_run_dir = tmp_path / "preprocess_run"
    sentence_dataset_dir = preprocess_run_dir / "sentence_dataset" / "by_year"
    _write_sentence_year(sentence_dataset_dir / "2006.parquet", year=2006)
    source_manifest_path = preprocess_run_dir / "run_manifest.json"
    source_manifest_path.write_text(
        json.dumps(
            {
                "cleaning_policy_id": "item_text_clean_v2",
                "cleaning": {"cleaning_policy_id": "item_text_clean_v2"},
                "segment_policy_id": "segment_v1",
                "authority": {},
                "sentence_dataset": {},
            }
        ),
        encoding="utf-8",
    )

    from thesis_pkg.benchmarking import finbert_benchmark
    from thesis_pkg.benchmarking import finbert_staged_inference

    monkeypatch.setattr(finbert_benchmark, "_import_torch", lambda: _FakeTorch())
    monkeypatch.setattr(finbert_staged_inference, "load_finbert_tokenizer", lambda authority: _FakeTokenizer())
    monkeypatch.setattr(finbert_staged_inference, "load_finbert_model", lambda authority, runtime: _FakeModel())

    cfg = FinbertSentenceParquetInferenceRunConfig(
        sentence_dataset_dir=sentence_dataset_dir,
        out_root=tmp_path / "runs",
        batch_config=BucketBatchConfig(name="baseline", short_batch_size=2, medium_batch_size=1, long_batch_size=1),
        run_name="sentence_parquet_inference_smoke",
    )
    artifacts = run_finbert_sentence_parquet_inference(cfg)
    manifest = json.loads(artifacts.run_manifest_path.read_text(encoding="utf-8"))
    assert manifest["item_feature_contract"]["segment_policy_id"] == "segment_v1"
    assert manifest["source_sentence_dataset_manifest"]["cleaning_policy_id"] == "item_text_clean_v2"
    source_manifest_path.write_text(
        json.dumps(
            {
                "cleaning_policy_id": "item_text_clean_v3",
                "cleaning": {"cleaning_policy_id": "item_text_clean_v3"},
                "segment_policy_id": "segment_v2",
                "authority": {},
                "sentence_dataset": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="incompatible semantic settings"):
        run_finbert_sentence_parquet_inference(cfg)

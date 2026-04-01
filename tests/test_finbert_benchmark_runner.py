from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from thesis_pkg.benchmarking.contracts import BucketBatchConfig
from thesis_pkg.benchmarking.contracts import FinbertBenchmarkRunConfig
from thesis_pkg.benchmarking.finbert_benchmark import run_finbert_benchmark


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
        rows = len(texts)
        return {
            "input_ids": _FakeTensor(rows),
            "attention_mask": _FakeTensor(rows),
        }


class _FakeModel:
    def to(self, device: str):
        del device
        return self

    def eval(self):
        return self

    def __call__(self, **kwargs):
        del kwargs
        return {"logits": None}


def test_run_finbert_benchmark_reads_manifest_and_writes_artifacts(tmp_path: Path, monkeypatch) -> None:
    sections_path = tmp_path / "sections.parquet"
    sentences_path = tmp_path / "sentences.parquet"
    pl.DataFrame(
        {
            "benchmark_row_id": ["doc1:item_1", "doc2:item_7"],
            "doc_id": ["doc1", "doc2"],
            "filing_date": [None, None],
            "filing_year": [2000, 2001],
            "benchmark_item_code": ["item_1", "item_7"],
            "full_text": ["First sentence. Second sentence.", "Third sentence only."],
        }
    ).write_parquet(sections_path)
    pl.DataFrame(
        {
            "benchmark_sentence_id": ["doc1:item_1:0", "doc1:item_1:1", "doc2:item_7:0"],
            "benchmark_row_id": ["doc1:item_1", "doc1:item_1", "doc2:item_7"],
            "doc_id": ["doc1", "doc1", "doc2"],
            "filing_date": [None, None, None],
            "filing_year": [2000, 2000, 2001],
            "benchmark_item_code": ["item_1", "item_1", "item_7"],
            "sentence_index": [0, 1, 0],
            "sentence_text": ["First sentence", "Second sentence", "Third sentence only"],
            "sentence_char_count": [14, 15, 19],
            "sentencizer_backend": ["spacy_blank_en_sentencizer"] * 3,
            "sentencizer_version": ["test"] * 3,
            "finbert_token_count_512": [6, 6, 8],
            "finbert_token_bucket_512": ["short", "short", "short"],
        }
    ).write_parquet(sentences_path)

    manifest_path = tmp_path / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_tag": "finbert_10k_items_5pct_seed42",
                "artifacts": {
                    "sections_path": str(sections_path.resolve()),
                    "sentences_path": str(sentences_path.resolve()),
                },
            }
        ),
        encoding="utf-8",
    )

    from thesis_pkg.benchmarking import finbert_benchmark

    monkeypatch.setattr(finbert_benchmark, "_import_torch", lambda: _FakeTorch())
    monkeypatch.setattr(finbert_benchmark, "load_finbert_tokenizer", lambda authority: _FakeTokenizer())
    monkeypatch.setattr(finbert_benchmark, "load_finbert_model", lambda authority, runtime: _FakeModel())
    monkeypatch.setattr(
        finbert_benchmark,
        "benchmark_sentence_splitting",
        lambda sections_df, sentence_cfg, *, runs: {
            "stage": "sentence_split",
            "documents": int(sections_df.height),
            "total_sentences": 3,
            "median_seconds": 0.01,
            "docs_per_second": 200.0,
            "sentences_per_second": 300.0,
            "sentencizer_backend": sentence_cfg.sentencizer_backend,
            "sentencizer_version": "test",
        },
    )

    artifacts = run_finbert_benchmark(
        FinbertBenchmarkRunConfig(
            dataset_manifest_path=manifest_path,
            out_root=tmp_path / "runs",
            batch_config=BucketBatchConfig(name="cpu_smoke", short_batch_size=2, medium_batch_size=2, long_batch_size=2),
            run_name="smoke_run",
        )
    )

    assert artifacts.run_manifest_path.exists()
    assert artifacts.records_path.exists()
    assert artifacts.tokenizer_results_path.exists()
    assert artifacts.model_results_path.exists()
    assert artifacts.full_pipeline_results_path.exists()
    assert artifacts.summary_path.exists()

    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    assert summary["dataset_tag"] == "finbert_10k_items_5pct_seed42"
    assert summary["sentence_source"] == "precomputed"
    assert summary["sentence_rows"] == 3

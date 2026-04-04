from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from thesis_pkg.benchmarking.contracts import BucketBatchConfig
from thesis_pkg.benchmarking.contracts import FinbertBenchmarkRunArtifacts
from thesis_pkg.benchmarking.contracts import FinbertBenchmarkRunConfig
from thesis_pkg.benchmarking.contracts import FinbertBenchmarkSweepConfig
from thesis_pkg.benchmarking.sweep import run_finbert_benchmark_sweep


def test_run_finbert_benchmark_sweep_writes_summary(tmp_path: Path, monkeypatch) -> None:
    manifest_path = tmp_path / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps({"dataset_tag": "dataset", "artifacts": {"sections_path": "unused.parquet", "sentences_path": None}}),
        encoding="utf-8",
    )

    def _fake_run_finbert_benchmark(run_cfg, *, authority, runtime):
        del authority, runtime
        run_dir = run_cfg.out_root / run_cfg.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        summary_path = run_dir / "run_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "run_name": run_cfg.run_name,
                    "dataset_tag": "dataset",
                    "sections_rows": 10,
                    "sentence_rows": 100,
                    "sentence_source": "precomputed",
                    "sentence_source_reason": "registered_precomputed_sentence_artifact",
                    "sentence_split": {"median_seconds": 0.5},
                    "sentence_materialization_seconds": 0.0,
                    "runtime_environment": {"resolved_device": "cpu", "cuda_device_name": None},
                    "benchmark_scope": {"full_pipeline_includes_sentence_splitting": False},
                    "tokenizer": {"seconds": 1.0, "rows_per_second": 100.0, "peak_vram_gb": None},
                    "model": {"seconds": 2.0, "rows_per_second": 50.0, "peak_vram_gb": 1.5},
                    "full_pipeline": {"seconds": 3.0, "rows_per_second": 33.3, "peak_vram_gb": 1.5},
                }
            ),
            encoding="utf-8",
        )
        placeholder = run_dir / "placeholder.parquet"
        pl.DataFrame({"x": [1]}).write_parquet(placeholder)
        return FinbertBenchmarkRunArtifacts(
            run_dir=run_dir,
            run_manifest_path=run_dir / "run_manifest.json",
            records_path=run_dir / "records.jsonl",
            tokenizer_results_path=placeholder,
            model_results_path=placeholder,
            full_pipeline_results_path=placeholder,
            summary_path=summary_path,
            sentence_frame_path=None,
        )

    from thesis_pkg.benchmarking import sweep

    monkeypatch.setattr(sweep, "run_finbert_benchmark", _fake_run_finbert_benchmark)

    summary_df = run_finbert_benchmark_sweep(
        FinbertBenchmarkSweepConfig(
            base_run=FinbertBenchmarkRunConfig(
                dataset_manifest_path=manifest_path,
                out_root=tmp_path / "runs",
                batch_config=BucketBatchConfig(name="base", short_batch_size=2, medium_batch_size=2, long_batch_size=2),
                run_name="base",
            ),
            batch_configs=(
                BucketBatchConfig(name="small", short_batch_size=2, medium_batch_size=2, long_batch_size=2),
                BucketBatchConfig(name="large", short_batch_size=4, medium_batch_size=4, long_batch_size=4),
            ),
        )
    )

    assert summary_df.height == 2
    assert sorted(summary_df["batch_config_name"].to_list()) == ["large", "small"]

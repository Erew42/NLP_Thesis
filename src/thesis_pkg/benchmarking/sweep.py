from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import polars as pl

from thesis_pkg.benchmarking.contracts import DEFAULT_FINBERT_AUTHORITY
from thesis_pkg.benchmarking.contracts import FinbertAuthoritySpec
from thesis_pkg.benchmarking.contracts import FinbertBenchmarkSweepConfig
from thesis_pkg.benchmarking.contracts import FinbertRuntimeConfig
from thesis_pkg.benchmarking.finbert_benchmark import run_finbert_benchmark
from thesis_pkg.benchmarking.manifest_contracts import MANIFEST_PATH_SEMANTICS_RELATIVE
from thesis_pkg.benchmarking.manifest_contracts import write_manifest_path_value
from thesis_pkg.benchmarking.run_logging import append_jsonl_record
from thesis_pkg.benchmarking.run_logging import utc_timestamp
from thesis_pkg.benchmarking.run_logging import write_frame
from thesis_pkg.benchmarking.run_logging import write_json


def run_finbert_benchmark_sweep(
    cfg: FinbertBenchmarkSweepConfig,
    *,
    authority: FinbertAuthoritySpec = DEFAULT_FINBERT_AUTHORITY,
    runtime: FinbertRuntimeConfig = FinbertRuntimeConfig(),
) -> pl.DataFrame:
    sweep_name = f"sweep_{utc_timestamp().replace(':', '')}"
    sweep_dir = cfg.base_run.out_root / sweep_name
    records_path = sweep_dir / "benchmark_sweep_records.jsonl"
    rows: list[dict[str, Any]] = []

    for batch_config in cfg.batch_configs:
        run_cfg = replace(
            cfg.base_run,
            batch_config=batch_config,
            run_name=f"{cfg.base_run.run_name or 'finbert'}_{batch_config.name}",
        )
        artifacts = run_finbert_benchmark(run_cfg, authority=authority, runtime=runtime)
        summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
        validated_contract = summary.get("validated_sentence_universe_contract") or {}
        row = {
            "run_name": summary["run_name"],
            "batch_config_name": batch_config.name,
            "dataset_tag": summary["dataset_tag"],
            "sections_rows": summary.get("sections_rows"),
            "sentence_rows": summary["sentence_rows"],
            "sentence_source": summary["sentence_source"],
            "sentence_source_reason": summary.get("sentence_source_reason"),
            "sentence_split_seconds": summary["sentence_split"]["median_seconds"],
            "sentence_materialization_seconds": summary.get("sentence_materialization_seconds"),
            "resolved_device": summary.get("runtime_environment", {}).get("resolved_device"),
            "cuda_device_name": summary.get("runtime_environment", {}).get("cuda_device_name"),
            "full_pipeline_includes_sentence_splitting": summary.get("benchmark_scope", {}).get(
                "full_pipeline_includes_sentence_splitting"
            ),
            "tokenizer_seconds": summary["tokenizer"]["seconds"],
            "tokenizer_rows_per_second": summary["tokenizer"]["rows_per_second"],
            "model_seconds": summary["model"]["seconds"],
            "model_rows_per_second": summary["model"]["rows_per_second"],
            "model_peak_vram_gb": summary["model"]["peak_vram_gb"],
            "full_pipeline_seconds": summary["full_pipeline"]["seconds"],
            "full_pipeline_rows_per_second": summary["full_pipeline"]["rows_per_second"],
            "full_pipeline_peak_vram_gb": summary["full_pipeline"]["peak_vram_gb"],
            "sentence_universe_contract_fingerprint": validated_contract.get(
                "sentence_universe_contract_fingerprint"
            ),
            "sentence_universe_comparable": validated_contract.get("sentence_universe_comparable"),
            "run_dir": artifacts.run_dir.name,
        }
        rows.append(row)
        append_jsonl_record(records_path, {"created_at_utc": utc_timestamp(), **row})

    summary_df = pl.DataFrame(rows).sort("batch_config_name")
    write_frame(summary_df, sweep_dir / "sweep_summary.parquet")
    summary_df.write_csv(sweep_dir / "sweep_summary.csv")
    write_json(
        sweep_dir / "sweep_manifest.json",
        {
            "manifest_version": 1,
            "path_semantics": MANIFEST_PATH_SEMANTICS_RELATIVE,
            "created_at_utc": utc_timestamp(),
            "dataset_manifest_path": write_manifest_path_value(
                cfg.base_run.dataset_manifest_path,
                manifest_path=sweep_dir / "sweep_manifest.json",
                path_semantics=MANIFEST_PATH_SEMANTICS_RELATIVE,
            ),
            "authority": authority.__dict__,
            "runtime": runtime.__dict__,
            "batch_configs": [batch_config.__dict__ for batch_config in cfg.batch_configs],
            "validated_sentence_universe_contracts": [
                {
                    "run_name": row["run_name"],
                    "sentence_universe_contract_fingerprint": row["sentence_universe_contract_fingerprint"],
                    "sentence_universe_comparable": row["sentence_universe_comparable"],
                }
                for row in rows
            ],
            "artifacts": {
                "records_path": write_manifest_path_value(
                    records_path,
                    manifest_path=sweep_dir / "sweep_manifest.json",
                    path_semantics=MANIFEST_PATH_SEMANTICS_RELATIVE,
                ),
                "summary_parquet_path": write_manifest_path_value(
                    sweep_dir / "sweep_summary.parquet",
                    manifest_path=sweep_dir / "sweep_manifest.json",
                    path_semantics=MANIFEST_PATH_SEMANTICS_RELATIVE,
                ),
                "summary_csv_path": write_manifest_path_value(
                    sweep_dir / "sweep_summary.csv",
                    manifest_path=sweep_dir / "sweep_manifest.json",
                    path_semantics=MANIFEST_PATH_SEMANTICS_RELATIVE,
                ),
            },
        },
    )
    return summary_df

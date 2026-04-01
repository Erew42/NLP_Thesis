from __future__ import annotations

import json
import statistics
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import polars as pl

from thesis_pkg.benchmarking.contracts import DEFAULT_FINBERT_AUTHORITY
from thesis_pkg.benchmarking.contracts import FinbertAuthoritySpec
from thesis_pkg.benchmarking.contracts import FinbertBenchmarkRunArtifacts
from thesis_pkg.benchmarking.contracts import FinbertBenchmarkRunConfig
from thesis_pkg.benchmarking.contracts import FinbertRuntimeConfig
from thesis_pkg.benchmarking.run_logging import append_jsonl_record
from thesis_pkg.benchmarking.run_logging import utc_timestamp
from thesis_pkg.benchmarking.run_logging import write_frame
from thesis_pkg.benchmarking.run_logging import write_json
from thesis_pkg.benchmarking.sentences import _build_sentencizer
from thesis_pkg.benchmarking.sentences import _sentencizer_version
from thesis_pkg.benchmarking.sentences import derive_sentence_frame
from thesis_pkg.benchmarking.token_lengths import FINBERT_TOKEN_BUCKET_COLUMN
from thesis_pkg.benchmarking.token_lengths import load_finbert_tokenizer


def _import_bert_model_class():
    try:
        from transformers import BertForSequenceClassification
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "transformers is required for FinBERT runtime benchmarking. "
            "Install thesis_pkg[benchmark] before running the benchmark runner."
        ) from exc
    return BertForSequenceClassification


def _import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "torch is required for FinBERT runtime benchmarking. "
            "Install thesis_pkg[benchmark] before running the benchmark runner."
        ) from exc
    return torch


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_device(runtime: FinbertRuntimeConfig) -> str:
    torch = _import_torch()
    if runtime.device is not None:
        return runtime.device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_finbert_model(
    authority: FinbertAuthoritySpec,
    runtime: FinbertRuntimeConfig,
):
    torch = _import_torch()
    bert_model = _import_bert_model_class()
    device = _resolve_device(runtime)
    model = bert_model.from_pretrained(authority.model_name)
    model.eval()
    return model.to(torch.device(device))


def _bucket_max_length(bucket: str, run_cfg: FinbertBenchmarkRunConfig) -> int:
    if bucket == "short":
        return run_cfg.bucket_lengths.short_max_length
    if bucket == "medium":
        return run_cfg.bucket_lengths.medium_max_length
    if bucket == "long":
        return run_cfg.bucket_lengths.long_max_length
    raise ValueError(f"Unknown bucket: {bucket!r}")


def _bucket_frame(sentences_df: pl.DataFrame, bucket: str) -> pl.DataFrame:
    return sentences_df.filter(pl.col(FINBERT_TOKEN_BUCKET_COLUMN) == bucket)


def _synchronize(torch_mod, device: str) -> None:
    if device.startswith("cuda") and torch_mod.cuda.is_available():
        torch_mod.cuda.synchronize()


def _reset_peak_vram(torch_mod, device: str) -> None:
    if device.startswith("cuda") and torch_mod.cuda.is_available():
        torch_mod.cuda.reset_peak_memory_stats()


def _peak_vram_gb(torch_mod, device: str) -> float | None:
    if device.startswith("cuda") and torch_mod.cuda.is_available():
        return float(torch_mod.cuda.max_memory_allocated() / (1024**3))
    return None


def _autocast_context(torch_mod, runtime: FinbertRuntimeConfig, device: str):
    if not runtime.use_autocast or not device.startswith("cuda") or not torch_mod.cuda.is_available():
        return nullcontext()
    dtype_name = runtime.amp_dtype
    if dtype_name == "auto":
        dtype = torch_mod.float16
    elif dtype_name == "float16":
        dtype = torch_mod.float16
    elif dtype_name == "bfloat16":
        dtype = torch_mod.bfloat16
    else:
        raise ValueError(f"Unsupported amp_dtype: {dtype_name!r}")
    return torch_mod.autocast(device_type="cuda", dtype=dtype)


def _tokenize_text_batches(
    texts: list[str],
    tokenizer,
    *,
    batch_size: int,
    max_length: int,
    return_tensors: str | None,
) -> list[Any]:
    batches: list[Any] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        batches.append(
            tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors=return_tensors,
            )
        )
    return batches


def _move_batch_to_device(batch: Any, device: str) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def benchmark_sentence_splitting(
    sections_df: pl.DataFrame,
    sentence_cfg,
    *,
    runs: int,
) -> dict[str, Any]:
    nlp = _build_sentencizer(sentence_cfg)
    texts = sections_df["full_text"].to_list()
    list(nlp.pipe(texts, batch_size=sentence_cfg.spacy_batch_size))

    timings: list[float] = []
    sentence_count = 0
    for _ in range(runs):
        start = time.perf_counter()
        docs = list(nlp.pipe(texts, batch_size=sentence_cfg.spacy_batch_size))
        elapsed = time.perf_counter() - start
        sentence_count = sum(sum(1 for _sent in doc.sents) for doc in docs)
        timings.append(elapsed)

    median_seconds = _median(timings)
    return {
        "stage": "sentence_split",
        "documents": int(sections_df.height),
        "total_sentences": int(sentence_count),
        "median_seconds": median_seconds,
        "docs_per_second": (sections_df.height / median_seconds) if median_seconds else None,
        "sentences_per_second": (sentence_count / median_seconds) if median_seconds else None,
        "sentencizer_backend": sentence_cfg.sentencizer_backend,
        "sentencizer_version": _sentencizer_version(),
    }


def benchmark_tokenizer_only(
    sentences_df: pl.DataFrame,
    tokenizer,
    run_cfg: FinbertBenchmarkRunConfig,
    *,
    runs: int,
) -> pl.DataFrame:
    records: list[dict[str, Any]] = []
    for bucket in ("short", "medium", "long"):
        frame = _bucket_frame(sentences_df, bucket)
        texts = frame["sentence_text"].to_list() if frame.height else []
        batch_size = run_cfg.batch_config.batch_size_for_bucket(bucket)
        max_length = _bucket_max_length(bucket, run_cfg)
        if not texts:
            records.append(
                {
                    "bucket": bucket,
                    "rows": 0,
                    "batch_size": batch_size,
                    "max_length": max_length,
                    "median_seconds": 0.0,
                    "rows_per_second": None,
                }
            )
            continue

        tokenizer(texts[: min(len(texts), batch_size)], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        timings: list[float] = []
        for _ in range(runs):
            start = time.perf_counter()
            _tokenize_text_batches(
                texts,
                tokenizer,
                batch_size=batch_size,
                max_length=max_length,
                return_tensors="pt",
            )
            timings.append(time.perf_counter() - start)

        median_seconds = _median(timings)
        records.append(
            {
                "bucket": bucket,
                "rows": int(frame.height),
                "batch_size": batch_size,
                "max_length": max_length,
                "median_seconds": median_seconds,
                "rows_per_second": (frame.height / median_seconds) if median_seconds else None,
            }
        )
    return pl.DataFrame(records)


def benchmark_model_only(
    sentences_df: pl.DataFrame,
    tokenizer,
    model,
    runtime: FinbertRuntimeConfig,
    run_cfg: FinbertBenchmarkRunConfig,
    *,
    runs: int,
) -> pl.DataFrame:
    torch = _import_torch()
    device = _resolve_device(runtime)
    records: list[dict[str, Any]] = []
    for bucket in ("short", "medium", "long"):
        frame = _bucket_frame(sentences_df, bucket)
        texts = frame["sentence_text"].to_list() if frame.height else []
        batch_size = run_cfg.batch_config.batch_size_for_bucket(bucket)
        max_length = _bucket_max_length(bucket, run_cfg)
        if not texts:
            records.append(
                {
                    "bucket": bucket,
                    "rows": 0,
                    "batch_size": batch_size,
                    "max_length": max_length,
                    "median_seconds": 0.0,
                    "rows_per_second": None,
                    "peak_vram_gb": None,
                }
            )
            continue

        encoded_batches = _tokenize_text_batches(
            texts,
            tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            return_tensors="pt",
        )
        peak_vram = 0.0
        timings: list[float] = []
        for _ in range(runs):
            _reset_peak_vram(torch, device)
            _synchronize(torch, device)
            start = time.perf_counter()
            with torch.no_grad():
                with _autocast_context(torch, runtime, device):
                    for batch in encoded_batches:
                        model(**_move_batch_to_device(batch, device))
            _synchronize(torch, device)
            timings.append(time.perf_counter() - start)
            peak = _peak_vram_gb(torch, device)
            if peak is not None:
                peak_vram = max(peak_vram, peak)

        median_seconds = _median(timings)
        records.append(
            {
                "bucket": bucket,
                "rows": int(frame.height),
                "batch_size": batch_size,
                "max_length": max_length,
                "median_seconds": median_seconds,
                "rows_per_second": (frame.height / median_seconds) if median_seconds else None,
                "peak_vram_gb": peak_vram if peak_vram else _peak_vram_gb(torch, device),
            }
        )
    return pl.DataFrame(records)


def benchmark_full_pipeline(
    sentences_df: pl.DataFrame,
    tokenizer,
    model,
    runtime: FinbertRuntimeConfig,
    run_cfg: FinbertBenchmarkRunConfig,
    *,
    runs: int,
) -> pl.DataFrame:
    torch = _import_torch()
    device = _resolve_device(runtime)
    records: list[dict[str, Any]] = []
    for bucket in ("short", "medium", "long"):
        frame = _bucket_frame(sentences_df, bucket)
        texts = frame["sentence_text"].to_list() if frame.height else []
        batch_size = run_cfg.batch_config.batch_size_for_bucket(bucket)
        max_length = _bucket_max_length(bucket, run_cfg)
        if not texts:
            records.append(
                {
                    "bucket": bucket,
                    "rows": 0,
                    "batch_size": batch_size,
                    "max_length": max_length,
                    "median_seconds": 0.0,
                    "rows_per_second": None,
                    "peak_vram_gb": None,
                }
            )
            continue

        warmup_batches = _tokenize_text_batches(
            texts[: min(len(texts), batch_size)],
            tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            for batch in warmup_batches:
                model(**_move_batch_to_device(batch, device))

        peak_vram = 0.0
        timings: list[float] = []
        for _ in range(runs):
            _reset_peak_vram(torch, device)
            _synchronize(torch, device)
            start = time.perf_counter()
            with torch.no_grad():
                with _autocast_context(torch, runtime, device):
                    for batch in _tokenize_text_batches(
                        texts,
                        tokenizer,
                        batch_size=batch_size,
                        max_length=max_length,
                        return_tensors="pt",
                    ):
                        model(**_move_batch_to_device(batch, device))
            _synchronize(torch, device)
            timings.append(time.perf_counter() - start)
            peak = _peak_vram_gb(torch, device)
            if peak is not None:
                peak_vram = max(peak_vram, peak)

        median_seconds = _median(timings)
        records.append(
            {
                "bucket": bucket,
                "rows": int(frame.height),
                "batch_size": batch_size,
                "max_length": max_length,
                "median_seconds": median_seconds,
                "rows_per_second": (frame.height / median_seconds) if median_seconds else None,
                "peak_vram_gb": peak_vram if peak_vram else _peak_vram_gb(torch, device),
            }
        )
    return pl.DataFrame(records)


def _stage_summary(df: pl.DataFrame) -> dict[str, Any]:
    total_rows = int(df["rows"].sum()) if df.height else 0
    total_seconds = float(df["median_seconds"].sum()) if df.height else 0.0
    peak_vram = None
    if "peak_vram_gb" in df.columns:
        peaks = [value for value in df["peak_vram_gb"].to_list() if value is not None]
        peak_vram = max(peaks) if peaks else None
    return {
        "rows": total_rows,
        "seconds": total_seconds,
        "rows_per_second": (total_rows / total_seconds) if total_seconds else None,
        "peak_vram_gb": peak_vram,
    }


def run_finbert_benchmark(
    run_cfg: FinbertBenchmarkRunConfig,
    *,
    authority: FinbertAuthoritySpec = DEFAULT_FINBERT_AUTHORITY,
    runtime: FinbertRuntimeConfig = FinbertRuntimeConfig(),
) -> FinbertBenchmarkRunArtifacts:
    dataset_manifest = _load_manifest(run_cfg.dataset_manifest_path)
    sections_path = Path(dataset_manifest["artifacts"]["sections_path"])
    sections_df = pl.read_parquet(sections_path)

    sentence_stats = benchmark_sentence_splitting(
        sections_df,
        run_cfg.sentence_dataset,
        runs=run_cfg.stage_runs.sentence_split_runs,
    )

    precomputed_sentences = dataset_manifest["artifacts"].get("sentences_path")
    sentence_source = "derived_runtime"
    sentence_frame_path: Path | None = None
    sentence_materialization_seconds = 0.0
    if run_cfg.sentence_policy == "prefer_precomputed" and precomputed_sentences:
        sentence_frame_path = Path(precomputed_sentences)
        if sentence_frame_path.exists():
            sentences_df = pl.read_parquet(sentence_frame_path)
            sentence_source = "precomputed"
        else:
            start = time.perf_counter()
            sentences_df = derive_sentence_frame(
                sections_df,
                run_cfg.sentence_dataset,
                authority=authority,
            )
            sentence_materialization_seconds = time.perf_counter() - start
    else:
        start = time.perf_counter()
        sentences_df = derive_sentence_frame(
            sections_df,
            run_cfg.sentence_dataset,
            authority=authority,
        )
        sentence_materialization_seconds = time.perf_counter() - start

    tokenizer = load_finbert_tokenizer(authority)
    model = load_finbert_model(authority, runtime)
    tokenizer_df = benchmark_tokenizer_only(
        sentences_df,
        tokenizer,
        run_cfg,
        runs=run_cfg.stage_runs.tokenizer_runs,
    )
    model_df = benchmark_model_only(
        sentences_df,
        tokenizer,
        model,
        runtime,
        run_cfg,
        runs=run_cfg.stage_runs.model_runs,
    )
    full_df = benchmark_full_pipeline(
        sentences_df,
        tokenizer,
        model,
        runtime,
        run_cfg,
        runs=run_cfg.stage_runs.full_pipeline_runs,
    )

    dataset_tag = str(dataset_manifest.get("dataset_tag", sections_path.stem))
    run_name = run_cfg.run_name or f"{dataset_tag}_{run_cfg.batch_config.name}_{utc_timestamp().replace(':', '')}"
    run_dir = run_cfg.out_root / run_name
    records_path = run_dir / "benchmark_records.jsonl"
    run_manifest_path = run_dir / "run_manifest.json"
    tokenizer_results_path = run_dir / "tokenizer_bucket_results.parquet"
    model_results_path = run_dir / "model_bucket_results.parquet"
    full_pipeline_results_path = run_dir / "full_pipeline_bucket_results.parquet"
    summary_path = run_dir / "run_summary.json"

    write_frame(tokenizer_df, tokenizer_results_path)
    write_frame(model_df, model_results_path)
    write_frame(full_df, full_pipeline_results_path)

    append_jsonl_record(records_path, {"created_at_utc": utc_timestamp(), "run_name": run_name, **sentence_stats})
    for stage_name, stage_df in (
        ("tokenizer_only", tokenizer_df),
        ("model_only", model_df),
        ("full_pipeline", full_df),
    ):
        for row in stage_df.to_dicts():
            append_jsonl_record(
                records_path,
                {
                    "created_at_utc": utc_timestamp(),
                    "run_name": run_name,
                    "dataset_tag": dataset_tag,
                    "stage": stage_name,
                    **row,
                },
            )

    summary = {
        "run_name": run_name,
        "created_at_utc": utc_timestamp(),
        "dataset_tag": dataset_tag,
        "dataset_manifest_path": str(run_cfg.dataset_manifest_path.resolve()),
        "sections_rows": int(sections_df.height),
        "sentence_rows": int(sentences_df.height),
        "sentence_source": sentence_source,
        "sentence_split": sentence_stats,
        "sentence_materialization_seconds": sentence_materialization_seconds,
        "tokenizer": _stage_summary(tokenizer_df),
        "model": _stage_summary(model_df),
        "full_pipeline": _stage_summary(full_df),
    }
    write_json(summary_path, summary)
    write_json(
        run_manifest_path,
        {
            "run_name": run_name,
            "created_at_utc": utc_timestamp(),
            "dataset_manifest_path": str(run_cfg.dataset_manifest_path.resolve()),
            "dataset_tag": dataset_tag,
            "authority": authority.__dict__,
            "runtime": runtime.__dict__,
            "batch_config": run_cfg.batch_config.__dict__,
            "bucket_lengths": run_cfg.bucket_lengths.__dict__,
            "stage_runs": run_cfg.stage_runs.__dict__,
            "sentence_policy": run_cfg.sentence_policy,
            "sentence_source": sentence_source,
            "note": run_cfg.note,
            "artifacts": {
                "records_path": str(records_path.resolve()),
                "tokenizer_results_path": str(tokenizer_results_path.resolve()),
                "model_results_path": str(model_results_path.resolve()),
                "full_pipeline_results_path": str(full_pipeline_results_path.resolve()),
                "summary_path": str(summary_path.resolve()),
            },
        },
    )

    return FinbertBenchmarkRunArtifacts(
        run_dir=run_dir,
        run_manifest_path=run_manifest_path,
        records_path=records_path,
        tokenizer_results_path=tokenizer_results_path,
        model_results_path=model_results_path,
        full_pipeline_results_path=full_pipeline_results_path,
        summary_path=summary_path,
        sentence_frame_path=sentence_frame_path,
    )

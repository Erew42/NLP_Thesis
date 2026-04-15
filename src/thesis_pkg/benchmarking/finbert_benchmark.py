from __future__ import annotations

import json
import math
import statistics
import re
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import polars as pl

from thesis_pkg.benchmarking.contracts import BucketBatchConfig
from thesis_pkg.benchmarking.contracts import BucketLengthSpec
from thesis_pkg.benchmarking.contracts import DEFAULT_FINBERT_AUTHORITY
from thesis_pkg.benchmarking.contracts import FinbertAuthoritySpec
from thesis_pkg.benchmarking.contracts import FinbertBenchmarkRunArtifacts
from thesis_pkg.benchmarking.contracts import FinbertBenchmarkRunConfig
from thesis_pkg.benchmarking.contracts import FinbertRuntimeConfig
from thesis_pkg.benchmarking.manifest_contracts import json_sha256
from thesis_pkg.benchmarking.manifest_contracts import MANIFEST_PATH_SEMANTICS_RELATIVE
from thesis_pkg.benchmarking.manifest_contracts import resolve_manifest_path
from thesis_pkg.benchmarking.manifest_contracts import semantic_file_fingerprint
from thesis_pkg.benchmarking.manifest_contracts import write_manifest_path_value
from thesis_pkg.benchmarking.run_logging import append_jsonl_record
from thesis_pkg.benchmarking.run_logging import utc_timestamp
from thesis_pkg.benchmarking.run_logging import write_frame
from thesis_pkg.benchmarking.run_logging import write_json
from thesis_pkg.benchmarking.sentences import SENTENCE_FRAME_SCHEMA
from thesis_pkg.benchmarking.sentences import _build_sentencizer
from thesis_pkg.benchmarking.sentences import _sentencizer_version
from thesis_pkg.benchmarking.sentences import derive_sentence_frame
from thesis_pkg.benchmarking.token_lengths import FINBERT_TOKEN_BUCKET_COLUMN
from thesis_pkg.benchmarking.token_lengths import load_finbert_tokenizer

SENTENCE_DATASET_FILENAME = "finbert_10k_item_sentences.parquet"


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


def _inspect_dataset_manifest(
    dataset_manifest: dict[str, Any],
    *,
    manifest_path: Path,
) -> dict[str, Any]:
    artifacts = dataset_manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"Dataset manifest {manifest_path} is missing an 'artifacts' mapping.")

    sections_raw = artifacts.get("sections_path")
    if not sections_raw:
        raise ValueError(f"Dataset manifest {manifest_path} is missing artifacts.sections_path.")

    path_semantics = dataset_manifest.get("path_semantics")
    sections_path = resolve_manifest_path(
        sections_raw,
        manifest_path=manifest_path,
        path_semantics=path_semantics,
    )
    assert sections_path is not None
    if not sections_path.exists():
        raise FileNotFoundError(
            f"Dataset manifest {manifest_path} references missing sections parquet: {sections_path}"
        )

    registered_sentences_path: Path | None = None
    registered_sentences_raw = artifacts.get("sentences_path")
    if registered_sentences_raw:
        registered_sentences_path = resolve_manifest_path(
            registered_sentences_raw,
            manifest_path=manifest_path,
            path_semantics=path_semantics,
        )

    warnings: list[str] = []
    if registered_sentences_path is not None and not registered_sentences_path.exists():
        warnings.append("registered_sentences_path_missing")

    unregistered_sentence_path: Path | None = None
    if registered_sentences_path is None and sections_path.parent.name == "dataset":
        candidate = sections_path.parent.parent / "derived" / SENTENCE_DATASET_FILENAME
        if candidate.exists():
            unregistered_sentence_path = candidate
            warnings.append("unregistered_sentence_artifact_present_but_not_registered")

    return {
        "path_semantics": path_semantics or "absolute_legacy_v0",
        "sections_path": write_manifest_path_value(
            sections_path,
            manifest_path=manifest_path,
            path_semantics=MANIFEST_PATH_SEMANTICS_RELATIVE,
        ),
        "registered_sentences_path": (
            write_manifest_path_value(
                registered_sentences_path,
                manifest_path=manifest_path,
                path_semantics=MANIFEST_PATH_SEMANTICS_RELATIVE,
            )
            if registered_sentences_path is not None
            else None
        ),
        "registered_sentences_available": (
            registered_sentences_path is not None and registered_sentences_path.exists()
        ),
        "unregistered_sentence_artifact_path": (
            write_manifest_path_value(
                unregistered_sentence_path,
                manifest_path=manifest_path,
                path_semantics=MANIFEST_PATH_SEMANTICS_RELATIVE,
            )
            if unregistered_sentence_path is not None
            else None
        ),
        "warnings": warnings,
        "_sections_path_resolved": sections_path,
        "_registered_sentences_path_resolved": registered_sentences_path,
    }


def _portable_manifest_diagnostics(manifest_diagnostics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in manifest_diagnostics.items()
        if not key.startswith("_")
    }


def _validated_sentence_universe_contract(
    dataset_manifest: dict[str, Any],
    *,
    manifest_path: Path,
    manifest_diagnostics: dict[str, Any],
    run_cfg: FinbertBenchmarkRunConfig,
) -> dict[str, Any]:
    declared = dataset_manifest.get("sentence_universe_contract") or {}
    sections_path = manifest_diagnostics["_sections_path_resolved"]
    registered_sentences_path = manifest_diagnostics.get("_registered_sentences_path_resolved")
    declared_contract_version = declared.get("contract_version")
    recomputed = {
        "contract_version": "sentence_universe_contract_v2",
        "sentence_dataset_config": asdict(run_cfg.sentence_dataset),
        "section_universe_contract_fingerprint": declared.get("section_universe_contract_fingerprint"),
        "sections_dataset_fingerprint": semantic_file_fingerprint(sections_path),
        "precomputed_sentences_fingerprint": (
            semantic_file_fingerprint(registered_sentences_path)
            if registered_sentences_path is not None and registered_sentences_path.exists()
            else None
        ),
    }
    mismatches: list[str] = []
    if not declared:
        mismatches.append("missing_contract")
    else:
        if declared.get("sentence_dataset_config") != recomputed["sentence_dataset_config"]:
            mismatches.append("sentence_dataset_config")
        if declared_contract_version == "sentence_universe_contract_v2":
            if declared.get("sections_dataset_fingerprint") != recomputed["sections_dataset_fingerprint"]:
                mismatches.append("sections_dataset_fingerprint")
            if manifest_diagnostics.get("registered_sentences_available"):
                if (
                    declared.get("precomputed_sentences_fingerprint")
                    != recomputed["precomputed_sentences_fingerprint"]
                ):
                    mismatches.append("precomputed_sentences_fingerprint")
    recomputed["sentence_universe_contract_fingerprint"] = json_sha256(recomputed)
    recomputed["declared_contract_version"] = declared_contract_version
    recomputed["sentence_universe_comparable"] = not mismatches
    recomputed["mismatches"] = mismatches
    return recomputed


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
    kwargs = {}
    if authority.model_revision is not None:
        kwargs["revision"] = authority.model_revision
    model = bert_model.from_pretrained(authority.model_name, **kwargs)
    model.eval()
    return model.to(torch.device(device))


def _bucket_max_length(
    bucket: str,
    run_cfg_or_bucket_lengths: FinbertBenchmarkRunConfig | BucketLengthSpec,
) -> int:
    bucket_lengths = (
        run_cfg_or_bucket_lengths.bucket_lengths
        if isinstance(run_cfg_or_bucket_lengths, FinbertBenchmarkRunConfig)
        else run_cfg_or_bucket_lengths
    )
    if bucket == "short":
        return bucket_lengths.short_max_length
    if bucket == "medium":
        return bucket_lengths.medium_max_length
    if bucket == "long":
        return bucket_lengths.long_max_length
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


def _resolved_amp_dtype_name(torch_mod, runtime: FinbertRuntimeConfig, device: str) -> str | None:
    if not runtime.use_autocast or not device.startswith("cuda") or not torch_mod.cuda.is_available():
        return None
    if runtime.amp_dtype == "auto":
        return "float16"
    if runtime.amp_dtype in {"float16", "bfloat16"}:
        return runtime.amp_dtype
    raise ValueError(f"Unsupported amp_dtype: {runtime.amp_dtype!r}")


def _autocast_context(torch_mod, runtime: FinbertRuntimeConfig, device: str):
    resolved_dtype_name = _resolved_amp_dtype_name(torch_mod, runtime, device)
    if resolved_dtype_name is None:
        return nullcontext()
    if resolved_dtype_name == "float16":
        dtype = torch_mod.float16
    elif resolved_dtype_name == "bfloat16":
        dtype = torch_mod.bfloat16
    else:
        raise ValueError(f"Unsupported amp dtype after resolution: {resolved_dtype_name!r}")
    return torch_mod.autocast(device_type="cuda", dtype=dtype)


def _device_index(device: str) -> int:
    if ":" not in device:
        return 0
    _, index = device.split(":", maxsplit=1)
    try:
        return int(index)
    except ValueError:
        return 0


def _runtime_environment(runtime: FinbertRuntimeConfig, device: str) -> dict[str, Any]:
    torch = _import_torch()
    cuda_mod = getattr(torch, "cuda", None)
    cuda_available = bool(cuda_mod is not None and cuda_mod.is_available())

    environment: dict[str, Any] = {
        "requested_device": runtime.device,
        "resolved_device": device,
        "torch_version": getattr(torch, "__version__", None),
        "cuda_available": cuda_available,
        "cuda_device_count": None,
        "cuda_device_name": None,
        "cuda_total_memory_gb": None,
        "autocast_enabled": _resolved_amp_dtype_name(torch, runtime, device) is not None,
        "resolved_amp_dtype": _resolved_amp_dtype_name(torch, runtime, device),
    }
    if not cuda_available:
        return environment

    device_count = getattr(cuda_mod, "device_count", None)
    if callable(device_count):
        environment["cuda_device_count"] = int(device_count())

    if not device.startswith("cuda"):
        return environment

    device_index = _device_index(device)
    get_device_name = getattr(cuda_mod, "get_device_name", None)
    if callable(get_device_name):
        environment["cuda_device_name"] = str(get_device_name(device_index))

    get_device_properties = getattr(cuda_mod, "get_device_properties", None)
    if callable(get_device_properties):
        props = get_device_properties(device_index)
        total_memory = getattr(props, "total_memory", None)
        if total_memory is not None:
            environment["cuda_total_memory_gb"] = float(total_memory / (1024**3))

    return environment


def _benchmark_scope() -> dict[str, Any]:
    return {
        "sentence_split_measured_separately": True,
        "sentence_materialization_includes_token_length_annotation": True,
        "full_pipeline_definition": "tokenizer_and_model_over_sentence_rows",
        "full_pipeline_includes_sentence_splitting": False,
    }


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


def _normalize_finbert_label_name(label: str) -> str | None:
    cleaned = re.sub(r"[^a-z0-9]+", "", str(label).strip().lower())
    if cleaned in {"negative", "neg", "bearish"}:
        return "negative"
    if cleaned in {"neutral", "neu"}:
        return "neutral"
    if cleaned in {"positive", "pos", "bullish"}:
        return "positive"
    return None


def resolve_finbert_label_mapping(model) -> dict[int, str]:
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("Model does not expose a config object with FinBERT labels.")

    raw_id2label = getattr(config, "id2label", None)
    raw_label2id = getattr(config, "label2id", None)

    pairs: list[tuple[int, str]] = []
    if isinstance(raw_id2label, dict):
        for key, value in raw_id2label.items():
            try:
                pairs.append((int(key), str(value)))
            except (TypeError, ValueError):
                continue
    elif isinstance(raw_label2id, dict):
        for key, value in raw_label2id.items():
            try:
                pairs.append((int(value), str(key)))
            except (TypeError, ValueError):
                continue

    normalized: dict[int, str] = {}
    seen_labels: set[str] = set()
    for index, label in sorted(pairs):
        normalized_label = _normalize_finbert_label_name(label)
        if normalized_label is None:
            continue
        if normalized_label in seen_labels:
            raise ValueError(f"Duplicate normalized FinBERT label in model config: {normalized_label!r}")
        normalized[index] = normalized_label
        seen_labels.add(normalized_label)

    expected_labels = {"negative", "neutral", "positive"}
    if set(normalized.values()) != expected_labels:
        raise ValueError(
            "Could not normalize FinBERT labels from model config. "
            f"Observed labels: {sorted(set(normalized.values()))!r}"
        )
    return dict(sorted(normalized.items()))


def _empty_sentence_score_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            **SENTENCE_FRAME_SCHEMA,
            "negative_prob": pl.Float64,
            "neutral_prob": pl.Float64,
            "positive_prob": pl.Float64,
            "predicted_label": pl.Utf8,
        }
    )


def _softmax_row(values: list[float]) -> list[float]:
    if not values:
        return []
    max_value = max(values)
    exp_values = [math.exp(value - max_value) for value in values]
    denom = sum(exp_values)
    if denom == 0.0:
        return [0.0 for _ in values]
    return [value / denom for value in exp_values]


def _to_nested_float_list(values: Any) -> list[list[float]]:
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "tolist"):
        values = values.tolist()
    nested = list(values)
    return [[float(value) for value in row] for row in nested]


def _probability_rows_from_logits(logits: Any, torch_mod) -> list[list[float]]:
    functional = getattr(getattr(torch_mod, "nn", None), "functional", None)
    softmax = getattr(functional, "softmax", None)
    if callable(softmax):
        return _to_nested_float_list(softmax(logits, dim=-1))
    return [_softmax_row(row) for row in _to_nested_float_list(logits)]


def score_sentence_frame(
    sentences_df: pl.DataFrame,
    tokenizer,
    model,
    runtime: FinbertRuntimeConfig,
    *,
    batch_config: BucketBatchConfig,
    bucket_lengths: BucketLengthSpec,
) -> pl.DataFrame:
    if sentences_df.is_empty():
        return _empty_sentence_score_frame()

    torch = _import_torch()
    device = _resolve_device(runtime)
    label_mapping = resolve_finbert_label_mapping(model)
    indexed_sentences = sentences_df.with_row_index("_sentence_row_nr")
    records: list[pl.DataFrame] = []

    for bucket in ("short", "medium", "long"):
        frame = _bucket_frame(indexed_sentences, bucket)
        if frame.is_empty():
            continue

        texts = frame["sentence_text"].to_list()
        batch_size = batch_config.batch_size_for_bucket(bucket)
        max_length = _bucket_max_length(bucket, bucket_lengths)
        probability_columns: dict[str, list[float]] = {
            "negative_prob": [],
            "neutral_prob": [],
            "positive_prob": [],
        }
        predicted_labels: list[str] = []

        with torch.no_grad():
            with _autocast_context(torch, runtime, device):
                for batch in _tokenize_text_batches(
                    texts,
                    tokenizer,
                    batch_size=batch_size,
                    max_length=max_length,
                    return_tensors="pt",
                ):
                    outputs = model(**_move_batch_to_device(batch, device))
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
                    probability_rows = _probability_rows_from_logits(logits, torch)
                    for row in probability_rows:
                        normalized_probs = {
                            label_mapping[index]: float(probability)
                            for index, probability in enumerate(row)
                            if index in label_mapping
                        }
                        for label in ("negative", "neutral", "positive"):
                            probability_columns[f"{label}_prob"].append(normalized_probs[label])
                        predicted_labels.append(
                            max(
                                ("negative", "neutral", "positive"),
                                key=lambda label: normalized_probs[label],
                            )
                        )

        records.append(
            frame.with_columns(
                [
                    pl.Series("negative_prob", probability_columns["negative_prob"], dtype=pl.Float64),
                    pl.Series("neutral_prob", probability_columns["neutral_prob"], dtype=pl.Float64),
                    pl.Series("positive_prob", probability_columns["positive_prob"], dtype=pl.Float64),
                    pl.Series("predicted_label", predicted_labels, dtype=pl.Utf8),
                ]
            )
        )

    if not records:
        return _empty_sentence_score_frame()

    return (
        pl.concat(records, how="vertical_relaxed")
        .sort("_sentence_row_nr")
        .drop("_sentence_row_nr")
    )


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
    manifest_diagnostics = _inspect_dataset_manifest(
        dataset_manifest,
        manifest_path=run_cfg.dataset_manifest_path,
    )
    validated_sentence_universe_contract = _validated_sentence_universe_contract(
        dataset_manifest,
        manifest_path=run_cfg.dataset_manifest_path,
        manifest_diagnostics=manifest_diagnostics,
        run_cfg=run_cfg,
    )
    portable_manifest_diagnostics = _portable_manifest_diagnostics(manifest_diagnostics)
    sections_path = manifest_diagnostics["_sections_path_resolved"]
    sections_df = pl.read_parquet(sections_path)

    sentence_stats = benchmark_sentence_splitting(
        sections_df,
        run_cfg.sentence_dataset,
        runs=run_cfg.stage_runs.sentence_split_runs,
    )

    precomputed_sentences = manifest_diagnostics["_registered_sentences_path_resolved"]
    sentence_source = "derived_runtime"
    sentence_frame_path: Path | None = None
    sentence_materialization_seconds = 0.0
    sentence_source_reason = "policy_forced_runtime_sentence_derivation"
    if run_cfg.sentence_policy == "prefer_precomputed" and precomputed_sentences:
        sentence_frame_path = Path(precomputed_sentences)
        if sentence_frame_path.exists():
            if not validated_sentence_universe_contract["sentence_universe_comparable"]:
                raise ValueError(
                    "Registered precomputed sentence parquet is not semantically comparable to the "
                    "raw sections universe for this benchmark run. "
                    f"Mismatches: {validated_sentence_universe_contract['mismatches']}"
                )
            sentences_df = pl.read_parquet(sentence_frame_path)
            sentence_source = "precomputed"
            sentence_source_reason = "registered_precomputed_sentence_artifact"
        else:
            start = time.perf_counter()
            sentences_df = derive_sentence_frame(
                sections_df,
                run_cfg.sentence_dataset,
                authority=authority,
            )
            sentence_materialization_seconds = time.perf_counter() - start
            sentence_source_reason = "registered_sentence_artifact_missing"
            sentence_frame_path = None
    elif run_cfg.sentence_policy == "prefer_precomputed":
        start = time.perf_counter()
        sentences_df = derive_sentence_frame(
            sections_df,
            run_cfg.sentence_dataset,
            authority=authority,
        )
        sentence_materialization_seconds = time.perf_counter() - start
        sentence_source_reason = "no_registered_precomputed_sentence_artifact"
    else:
        start = time.perf_counter()
        sentences_df = derive_sentence_frame(
            sections_df,
            run_cfg.sentence_dataset,
            authority=authority,
        )
        sentence_materialization_seconds = time.perf_counter() - start
        sentence_source_reason = "policy_forced_runtime_sentence_derivation"

    resolved_device = _resolve_device(runtime)
    runtime_environment = _runtime_environment(runtime, resolved_device)
    benchmark_scope = _benchmark_scope()
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
        "manifest_version": 1,
        "path_semantics": MANIFEST_PATH_SEMANTICS_RELATIVE,
        "run_name": run_name,
        "created_at_utc": utc_timestamp(),
        "dataset_tag": dataset_tag,
        "dataset_manifest_path": write_manifest_path_value(
            run_cfg.dataset_manifest_path,
            manifest_path=summary_path,
            path_semantics=MANIFEST_PATH_SEMANTICS_RELATIVE,
        ),
        "sections_rows": int(sections_df.height),
        "sentence_rows": int(sentences_df.height),
        "sentence_source": sentence_source,
        "sentence_source_reason": sentence_source_reason,
        "sentence_split": sentence_stats,
        "sentence_materialization_seconds": sentence_materialization_seconds,
        "sentence_materialization": {
            "policy": run_cfg.sentence_policy,
            "source": sentence_source,
            "source_reason": sentence_source_reason,
            "registered_sentences_path": manifest_diagnostics["registered_sentences_path"],
            "seconds": sentence_materialization_seconds,
        },
        "dataset_manifest_diagnostics": portable_manifest_diagnostics,
        "validated_sentence_universe_contract": validated_sentence_universe_contract,
        "runtime_environment": runtime_environment,
        "benchmark_scope": benchmark_scope,
        "tokenizer": _stage_summary(tokenizer_df),
        "model": _stage_summary(model_df),
        "full_pipeline": _stage_summary(full_df),
    }
    write_json(summary_path, summary)
    write_json(
        run_manifest_path,
        {
            "manifest_version": 1,
            "path_semantics": MANIFEST_PATH_SEMANTICS_RELATIVE,
            "run_name": run_name,
            "created_at_utc": utc_timestamp(),
            "dataset_manifest_path": write_manifest_path_value(
                run_cfg.dataset_manifest_path,
                manifest_path=run_manifest_path,
                path_semantics=MANIFEST_PATH_SEMANTICS_RELATIVE,
            ),
            "dataset_tag": dataset_tag,
            "authority": authority.__dict__,
            "runtime": runtime.__dict__,
            "batch_config": run_cfg.batch_config.__dict__,
            "bucket_lengths": run_cfg.bucket_lengths.__dict__,
            "stage_runs": run_cfg.stage_runs.__dict__,
            "sentence_policy": run_cfg.sentence_policy,
            "sentence_source": sentence_source,
            "sentence_source_reason": sentence_source_reason,
            "sentence_materialization": summary["sentence_materialization"],
            "dataset_manifest_diagnostics": portable_manifest_diagnostics,
            "validated_sentence_universe_contract": validated_sentence_universe_contract,
            "runtime_environment": runtime_environment,
            "benchmark_scope": benchmark_scope,
            "note": run_cfg.note,
            "artifacts": {
                "records_path": write_manifest_path_value(
                    records_path,
                    manifest_path=run_manifest_path,
                    path_semantics=MANIFEST_PATH_SEMANTICS_RELATIVE,
                ),
                "tokenizer_results_path": write_manifest_path_value(
                    tokenizer_results_path,
                    manifest_path=run_manifest_path,
                    path_semantics=MANIFEST_PATH_SEMANTICS_RELATIVE,
                ),
                "model_results_path": write_manifest_path_value(
                    model_results_path,
                    manifest_path=run_manifest_path,
                    path_semantics=MANIFEST_PATH_SEMANTICS_RELATIVE,
                ),
                "full_pipeline_results_path": write_manifest_path_value(
                    full_pipeline_results_path,
                    manifest_path=run_manifest_path,
                    path_semantics=MANIFEST_PATH_SEMANTICS_RELATIVE,
                ),
                "summary_path": write_manifest_path_value(
                    summary_path,
                    manifest_path=run_manifest_path,
                    path_semantics=MANIFEST_PATH_SEMANTICS_RELATIVE,
                ),
                "sentence_frame_path": (
                    write_manifest_path_value(
                        sentence_frame_path,
                        manifest_path=run_manifest_path,
                        path_semantics=MANIFEST_PATH_SEMANTICS_RELATIVE,
                    )
                    if sentence_frame_path is not None
                    else None
                ),
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

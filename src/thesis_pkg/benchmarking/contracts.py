from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkItemSpec:
    benchmark_item_code: str
    item_id: str
    benchmark_item_label: str


DEFAULT_FINBERT_10K_ITEMS: tuple[BenchmarkItemSpec, ...] = (
    BenchmarkItemSpec(
        benchmark_item_code="item_1",
        item_id="1",
        benchmark_item_label="10-K Item 1",
    ),
    BenchmarkItemSpec(
        benchmark_item_code="item_1a",
        item_id="1A",
        benchmark_item_label="10-K Item 1A",
    ),
    BenchmarkItemSpec(
        benchmark_item_code="item_7",
        item_id="7",
        benchmark_item_label="10-K Item 7",
    ),
)


@dataclass(frozen=True)
class BenchmarkSampleSpec:
    sample_name: str
    sample_fraction: float


@dataclass(frozen=True)
class FinbertAuthoritySpec:
    model_name: str = "yiyanghkust/finbert-tone"
    tokenizer_class_name: str = "BertTokenizer"
    model_class_name: str = "BertForSequenceClassification"
    do_lower_case: bool = True
    token_count_max_length: int = 512
    token_bucket_edges: tuple[int, int] = (128, 256)

    def __post_init__(self) -> None:
        if self.token_count_max_length != 512:
            raise ValueError("The benchmark dataset definition is fixed at the 512-token authority level.")
        if self.token_bucket_edges != (128, 256):
            raise ValueError("The benchmark dataset token buckets are fixed at 128/256/512.")


DEFAULT_FINBERT_AUTHORITY = FinbertAuthoritySpec()


@dataclass(frozen=True)
class SentenceDatasetConfig:
    enabled: bool = False
    sentencizer_backend: str = "spacy_blank_en_sentencizer"
    spacy_batch_size: int = 32
    drop_blank_sentences: bool = True
    compression: str = "zstd"


@dataclass(frozen=True)
class FinbertBenchmarkSuiteConfig:
    source_items_dir: Path
    out_root: Path
    sample_specs: tuple[BenchmarkSampleSpec, ...]
    seed: int = 42
    compression: str = "zstd"
    form_types: tuple[str, ...] = ("10-K", "10-K405")
    target_items: tuple[BenchmarkItemSpec, ...] = DEFAULT_FINBERT_10K_ITEMS
    require_active_items: bool = True
    require_exists_by_regime: bool = True
    min_char_count: int = 250
    ensure_all_years_present: bool = True
    nested_samples: bool = True
    write_full_universe_token_audit: bool = False
    authority: FinbertAuthoritySpec = field(default_factory=FinbertAuthoritySpec)
    sentence_dataset: SentenceDatasetConfig = field(default_factory=SentenceDatasetConfig)


@dataclass(frozen=True)
class BenchmarkBuildArtifacts:
    dataset_tag: str
    dataset_dir: Path
    sections_path: Path
    sentences_path: Path | None
    manifest_path: Path
    selected_row_count: int
    selected_doc_count: int


@dataclass(frozen=True)
class BucketLengthSpec:
    short_max_length: int = 128
    medium_max_length: int = 256
    long_max_length: int = 512

    def __post_init__(self) -> None:
        if not (0 < self.short_max_length <= self.medium_max_length <= self.long_max_length):
            raise ValueError("Bucket max lengths must satisfy 0 < short <= medium <= long.")


@dataclass(frozen=True)
class BucketBatchConfig:
    name: str
    short_batch_size: int
    medium_batch_size: int
    long_batch_size: int

    def __post_init__(self) -> None:
        for value in (self.short_batch_size, self.medium_batch_size, self.long_batch_size):
            if value <= 0:
                raise ValueError("Batch sizes must be positive integers.")

    def batch_size_for_bucket(self, bucket: str) -> int:
        if bucket == "short":
            return self.short_batch_size
        if bucket == "medium":
            return self.medium_batch_size
        if bucket == "long":
            return self.long_batch_size
        raise ValueError(f"Unknown bucket: {bucket!r}")


@dataclass(frozen=True)
class StageRunsConfig:
    sentence_split_runs: int = 5
    tokenizer_runs: int = 3
    model_runs: int = 3
    full_pipeline_runs: int = 3

    def __post_init__(self) -> None:
        for value in (
            self.sentence_split_runs,
            self.tokenizer_runs,
            self.model_runs,
            self.full_pipeline_runs,
        ):
            if value <= 0:
                raise ValueError("Stage run counts must be positive integers.")


@dataclass(frozen=True)
class FinbertRuntimeConfig:
    device: str | None = None
    use_autocast: bool = True
    amp_dtype: str = "auto"


@dataclass(frozen=True)
class FinbertBenchmarkRunConfig:
    dataset_manifest_path: Path
    out_root: Path
    batch_config: BucketBatchConfig
    bucket_lengths: BucketLengthSpec = field(default_factory=BucketLengthSpec)
    stage_runs: StageRunsConfig = field(default_factory=StageRunsConfig)
    sentence_policy: str = "prefer_precomputed"
    sentence_dataset: SentenceDatasetConfig = field(default_factory=SentenceDatasetConfig)
    run_name: str | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.sentence_policy not in {"prefer_precomputed", "derive_runtime"}:
            raise ValueError(
                "sentence_policy must be 'prefer_precomputed' or 'derive_runtime'."
            )


@dataclass(frozen=True)
class FinbertBenchmarkSweepConfig:
    base_run: FinbertBenchmarkRunConfig
    batch_configs: tuple[BucketBatchConfig, ...]


@dataclass(frozen=True)
class FinbertBenchmarkRunArtifacts:
    run_dir: Path
    run_manifest_path: Path
    records_path: Path
    tokenizer_results_path: Path
    model_results_path: Path
    full_pipeline_results_path: Path
    summary_path: Path
    sentence_frame_path: Path | None

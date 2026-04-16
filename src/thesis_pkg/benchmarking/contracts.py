from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path


def _normalize_year_filter(
    year_filter: tuple[int, ...] | None,
) -> tuple[int, ...] | None:
    if year_filter is None:
        return None

    normalized_years = tuple(sorted({int(year) for year in year_filter}))
    for year in normalized_years:
        if year < 0:
            raise ValueError(f"year_filter values must be positive integers, got {year!r}.")
    return normalized_years


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
    model_revision: str | None = None
    tokenizer_revision: str | None = None
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

ALLOWED_SENTENCE_POSTPROCESS_POLICIES: tuple[str, ...] = (
    "none",
    "item7_reference_stitch_protect_v1",
    "item7_reference_stitch_protect_v2",
    "reference_stitch_protect_v3",
)
DEFAULT_RUNNER_SENTENCE_POSTPROCESS_POLICY = "reference_stitch_protect_v3"


@dataclass(frozen=True)
class ItemTextCleaningConfig:
    enabled: bool = True
    cleaning_policy_id: str = "item_text_clean_v2"
    drop_page_markers: bool = True
    drop_report_headers: bool = True
    drop_structural_tags: bool = True
    trim_early_toc_prefix: bool = True
    truncate_item_aware_tail_bleed: bool = True
    drop_reference_only_stubs: bool = True
    drop_table_like_lines: bool = False
    table_like_min_consecutive_lines: int = 1
    table_like_drop_header_context: bool = False
    table_like_allow_single_line_with_header: bool = False
    table_like_target_text_scopes: tuple[str, ...] | None = None
    toc_scan_char_window: int = 2000
    toc_min_matching_lines: int = 2
    tail_scan_fraction: float = 0.35
    reference_stub_max_char_count: int = 1200
    drop_blank_after_cleaning: bool = True
    enforce_item7_lm_token_floor: bool = True
    item7_min_lm_tokens: int = 250
    hard_drop_min_clean_char_count: int | None = None
    warn_below_clean_char_count: int = 500
    large_removal_warning_threshold: float = 0.40
    emit_cleaned_scope_artifact: bool = True
    emit_manual_audit_sample: bool = True

    def __post_init__(self) -> None:
        if not self.cleaning_policy_id:
            raise ValueError("cleaning_policy_id must be non-empty.")
        if self.table_like_min_consecutive_lines < 1:
            raise ValueError("table_like_min_consecutive_lines must be positive.")
        if self.toc_scan_char_window < 0:
            raise ValueError("toc_scan_char_window must be non-negative.")
        if self.toc_min_matching_lines < 1:
            raise ValueError("toc_min_matching_lines must be positive.")
        if not (0.0 <= self.tail_scan_fraction <= 1.0):
            raise ValueError("tail_scan_fraction must be between 0 and 1.")
        if self.reference_stub_max_char_count < 0:
            raise ValueError("reference_stub_max_char_count must be non-negative.")
        if self.item7_min_lm_tokens < 0:
            raise ValueError("item7_min_lm_tokens must be non-negative.")
        if self.hard_drop_min_clean_char_count is not None and self.hard_drop_min_clean_char_count < 0:
            raise ValueError("hard_drop_min_clean_char_count must be non-negative when provided.")
        if self.warn_below_clean_char_count < 0:
            raise ValueError("warn_below_clean_char_count must be non-negative.")
        if not (0.0 <= self.large_removal_warning_threshold <= 1.0):
            raise ValueError("large_removal_warning_threshold must be between 0 and 1.")
        if self.table_like_target_text_scopes is not None:
            normalized_scopes = tuple(
                scope.strip()
                for scope in self.table_like_target_text_scopes
                if str(scope).strip()
            )
            object.__setattr__(
                self,
                "table_like_target_text_scopes",
                normalized_scopes or None,
            )


@dataclass(frozen=True)
class SentenceDatasetConfig:
    enabled: bool = False
    sentencizer_backend: str = "spacy_blank_en_sentencizer"
    postprocess_policy: str = "none"
    spacy_batch_size: int = 32
    token_length_batch_size: int = 1024
    drop_blank_sentences: bool = True
    compression: str = "zstd"

    def __post_init__(self) -> None:
        if self.postprocess_policy not in ALLOWED_SENTENCE_POSTPROCESS_POLICIES:
            raise ValueError(
                f"postprocess_policy must be one of {list(ALLOWED_SENTENCE_POSTPROCESS_POLICIES)!r}, "
                f"got {self.postprocess_policy!r}."
            )


@dataclass(frozen=True)
class FinbertSectionUniverseConfig:
    source_items_dir: Path
    form_types: tuple[str, ...] = ("10-K", "10-K405")
    target_items: tuple[BenchmarkItemSpec, ...] = DEFAULT_FINBERT_10K_ITEMS
    require_active_items: bool = True
    require_exists_by_regime: bool = True
    min_char_count: int = 250


@dataclass(frozen=True)
class FinbertBenchmarkSuiteConfig:
    source_items_dir: Path
    out_root: Path
    sample_specs: tuple[BenchmarkSampleSpec, ...]
    seed: int = 42
    compression: str = "zstd"
    section_universe: FinbertSectionUniverseConfig | None = None
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

    def __post_init__(self) -> None:
        universe = self.section_universe
        if universe is None:
            universe = FinbertSectionUniverseConfig(
                source_items_dir=self.source_items_dir,
                form_types=self.form_types,
                target_items=self.target_items,
                require_active_items=self.require_active_items,
                require_exists_by_regime=self.require_exists_by_regime,
                min_char_count=self.min_char_count,
            )
        object.__setattr__(self, "section_universe", universe)
        object.__setattr__(self, "source_items_dir", universe.source_items_dir)
        object.__setattr__(self, "form_types", universe.form_types)
        object.__setattr__(self, "target_items", universe.target_items)
        object.__setattr__(self, "require_active_items", universe.require_active_items)
        object.__setattr__(self, "require_exists_by_regime", universe.require_exists_by_regime)
        object.__setattr__(self, "min_char_count", universe.min_char_count)


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


@dataclass(frozen=True)
class FinbertAnalysisRunConfig:
    source_items_dir: Path
    out_root: Path
    batch_config: BucketBatchConfig
    section_universe: FinbertSectionUniverseConfig | None = None
    runtime: FinbertRuntimeConfig = field(default_factory=FinbertRuntimeConfig)
    bucket_lengths: BucketLengthSpec = field(default_factory=BucketLengthSpec)
    sentence_dataset: SentenceDatasetConfig = field(default_factory=SentenceDatasetConfig)
    cleaning: ItemTextCleaningConfig = field(default_factory=ItemTextCleaningConfig)
    backbone_path: Path | None = None
    year_filter: tuple[int, ...] | None = None
    write_sentence_scores: bool = False
    overwrite: bool = False
    run_name: str | None = None
    note: str = ""

    def __post_init__(self) -> None:
        universe = self.section_universe
        if universe is None:
            universe = FinbertSectionUniverseConfig(source_items_dir=self.source_items_dir)
        object.__setattr__(self, "section_universe", universe)
        object.__setattr__(self, "source_items_dir", universe.source_items_dir)
        object.__setattr__(self, "year_filter", _normalize_year_filter(self.year_filter))


@dataclass(frozen=True)
class FinbertAnalysisRunArtifacts:
    run_dir: Path
    run_manifest_path: Path
    item_features_long_path: Path
    doc_features_wide_path: Path
    coverage_report_path: Path | None
    sentence_scores_dir: Path | None


@dataclass(frozen=True)
class FinbertSentencePreprocessingRunConfig:
    source_items_dir: Path
    out_root: Path
    section_universe: FinbertSectionUniverseConfig | None = None
    sentence_dataset: SentenceDatasetConfig = field(default_factory=SentenceDatasetConfig)
    cleaning: ItemTextCleaningConfig = field(default_factory=ItemTextCleaningConfig)
    target_doc_universe_path: Path | None = None
    year_filter: tuple[int, ...] | None = None
    overwrite: bool = False
    run_name: str | None = None
    note: str = ""

    def __post_init__(self) -> None:
        universe = self.section_universe
        if universe is None:
            universe = FinbertSectionUniverseConfig(source_items_dir=self.source_items_dir)
        object.__setattr__(self, "section_universe", universe)
        object.__setattr__(self, "source_items_dir", universe.source_items_dir)
        object.__setattr__(self, "year_filter", _normalize_year_filter(self.year_filter))

        if self.target_doc_universe_path is not None:
            object.__setattr__(
                self,
                "target_doc_universe_path",
                Path(self.target_doc_universe_path).resolve(),
            )


@dataclass(frozen=True)
class FinbertSentencePreprocessingRunArtifacts:
    run_dir: Path
    run_manifest_path: Path
    sentence_dataset_dir: Path
    yearly_summary_path: Path
    oversize_sections_path: Path | None = None
    cleaned_item_scopes_dir: Path | None = None
    cleaning_row_audit_path: Path | None = None
    cleaning_flagged_rows_path: Path | None = None
    item_scope_cleaning_diagnostics_path: Path | None = None
    manual_boundary_audit_sample_path: Path | None = None


@dataclass(frozen=True)
class FinbertTokenizerProfileRunConfig:
    sentence_dataset_dir: Path
    out_root: Path
    batch_config: BucketBatchConfig
    bucket_lengths: BucketLengthSpec = field(default_factory=BucketLengthSpec)
    year_filter: tuple[int, ...] | None = None
    profile_row_cap_per_bucket: int = 5000
    sample_seed: int = 42
    overwrite: bool = False
    run_name: str | None = None
    note: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "year_filter", _normalize_year_filter(self.year_filter))
        object.__setattr__(self, "sentence_dataset_dir", Path(self.sentence_dataset_dir).resolve())
        if self.profile_row_cap_per_bucket <= 0:
            raise ValueError("profile_row_cap_per_bucket must be a positive integer.")
        if self.sample_seed < 0:
            raise ValueError("sample_seed must be non-negative.")


@dataclass(frozen=True)
class FinbertTokenizerProfileRunArtifacts:
    run_dir: Path
    run_manifest_path: Path
    bucket_summary_path: Path
    timing_summary_path: Path


@dataclass(frozen=True)
class FinbertSentenceParquetInferenceRunConfig:
    sentence_dataset_dir: Path
    out_root: Path
    batch_config: BucketBatchConfig
    runtime: FinbertRuntimeConfig = field(default_factory=FinbertRuntimeConfig)
    bucket_lengths: BucketLengthSpec = field(default_factory=BucketLengthSpec)
    backbone_path: Path | None = None
    year_filter: tuple[int, ...] | None = None
    sentence_slice_rows: int = 5000
    write_sentence_scores: bool = False
    overwrite: bool = False
    run_name: str | None = None
    note: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "year_filter", _normalize_year_filter(self.year_filter))
        object.__setattr__(self, "sentence_dataset_dir", Path(self.sentence_dataset_dir).resolve())
        if self.backbone_path is not None:
            object.__setattr__(self, "backbone_path", Path(self.backbone_path).resolve())
        if self.sentence_slice_rows <= 0:
            raise ValueError("sentence_slice_rows must be a positive integer.")


@dataclass(frozen=True)
class FinbertSentenceParquetInferenceRunArtifacts:
    run_dir: Path
    run_manifest_path: Path
    item_features_long_path: Path
    doc_features_wide_path: Path
    coverage_report_path: Path | None
    sentence_scores_dir: Path | None
    yearly_summary_path: Path

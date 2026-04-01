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
class TokenLengthConfig:
    tokenizer_name: str = "yiyanghkust/finbert-tone"
    max_length: int = 512
    bucket_edges: tuple[int, int, int] = (128, 256, 512)
    truncation: bool = True
    add_special_tokens: bool = True


@dataclass(frozen=True)
class SentenceDatasetConfig:
    enabled: bool = False
    sentencizer_backend: str = "spacy_blank_en_sentencizer"
    spacy_batch_size: int = 32
    drop_blank_sentences: bool = True


@dataclass(frozen=True)
class FinbertBenchmarkSuiteConfig:
    source_items_dir: Path
    out_root: Path
    sample_specs: tuple[BenchmarkSampleSpec, ...]
    seed: int = 42
    compression: str = "zstd"
    form_types: tuple[str, ...] = ("10-K",)
    target_items: tuple[BenchmarkItemSpec, ...] = DEFAULT_FINBERT_10K_ITEMS
    require_active_items: bool = True
    require_exists_by_regime: bool = True
    min_char_count: int = 1
    ensure_all_years_present: bool = True
    nested_samples: bool = True
    token_length: TokenLengthConfig = field(default_factory=TokenLengthConfig)
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

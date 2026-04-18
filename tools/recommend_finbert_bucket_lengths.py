from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
from pathlib import Path
from typing import Sequence


def _resolve_repo_root() -> Path:
    candidates = [Path.cwd().resolve(), *Path.cwd().resolve().parents, *Path(__file__).resolve().parents]
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "src" / "thesis_pkg").exists():
            return candidate
    raise RuntimeError("Could not resolve repository root containing src/thesis_pkg.")


ROOT = _resolve_repo_root()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from thesis_pkg.benchmarking.contracts import FinbertAuthoritySpec
from thesis_pkg.benchmarking.contracts import ItemTextCleaningConfig
from thesis_pkg.benchmarking.contracts import SentenceDatasetConfig
from thesis_pkg.benchmarking.finbert_bucket_length_tuning import (
    recommend_conservative_bucket_edges,
)
from thesis_pkg.benchmarking.finbert_bucket_length_tuning import (
    write_bucket_edge_recommendation_report,
)
from thesis_pkg.benchmarking.sample_item_cleaning_sentence_diagnostics import DEFAULT_ITEM_CODES
from thesis_pkg.benchmarking.sample_item_cleaning_sentence_diagnostics import DEFAULT_POSTPROCESS_POLICY
from thesis_pkg.benchmarking.sample_item_cleaning_sentence_diagnostics import DEFAULT_SOURCE_ITEMS_DIR
from thesis_pkg.benchmarking.sample_item_cleaning_sentence_diagnostics import (
    SampleItemCleaningSentenceDiagnosticsConfig,
)
from thesis_pkg.benchmarking.sample_item_cleaning_sentence_diagnostics import (
    run_sample_item_cleaning_sentence_diagnostics,
)


def _default_authority_model_name_or_path() -> str:
    snapshot_root = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--yiyanghkust--finbert-tone"
        / "snapshots"
    )
    if snapshot_root.exists():
        snapshots = sorted(
            (path for path in snapshot_root.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if snapshots:
            return str(snapshots[0].resolve())
    return "yiyanghkust/finbert-tone"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample extracted item rows from the local sample items_analysis layout, "
            "materialize FinBERT sentences, and recommend conservative FinBERT "
            "bucket-edge overrides for a fresh rebucketing run."
        )
    )
    parser.add_argument(
        "--source-items-dir",
        type=Path,
        default=ROOT / DEFAULT_SOURCE_ITEMS_DIR,
        help="Path to the items_analysis year shards. Defaults to the sample_5pct_seed42 extracted items.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "tmp" / Path(__file__).stem,
        help="Directory for sampled sentence artifacts and the recommendation report.",
    )
    parser.add_argument(
        "--sample-doc-count",
        type=int,
        default=250,
        help="Number of unique documents to sample before sentence extraction.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for document sampling.",
    )
    parser.add_argument(
        "--years",
        nargs="*",
        type=int,
        default=None,
        help="Optional filing years to include before sampling.",
    )
    parser.add_argument(
        "--item-codes",
        nargs="+",
        default=list(DEFAULT_ITEM_CODES),
        help="Benchmark item codes to keep while building the sampled sentence dataset.",
    )
    parser.add_argument(
        "--selected-doc-ids-path",
        type=Path,
        default=None,
        help="Optional parquet/csv from a prior run to reuse the exact sampled doc_id universe.",
    )
    parser.add_argument(
        "--postprocess-policy",
        type=str,
        default=DEFAULT_POSTPROCESS_POLICY,
        help="Sentence postprocess policy used while materializing the sample sentence dataset.",
    )
    parser.add_argument(
        "--authority-model-name-or-path",
        type=str,
        default=_default_authority_model_name_or_path(),
        help=(
            "Hugging Face model id or a local tokenizer snapshot path for FinBERT token-length "
            "annotation while building the sampled sentence dataset."
        ),
    )
    parser.add_argument(
        "--target-quantile",
        type=float,
        default=0.999,
        help="Quantile to preserve within the current bucket before rounding and safety margin.",
    )
    parser.add_argument(
        "--round-to",
        type=int,
        default=8,
        help="Round suggested max lengths up to this token multiple.",
    )
    parser.add_argument(
        "--safety-margin-tokens",
        type=int,
        default=8,
        help="Add this many tokens above the target quantile before rounding.",
    )
    parser.add_argument(
        "--allow-medium-edge-reduction",
        action="store_true",
        help="Allow the medium/long boundary to move below 256. By default the long bucket boundary stays unchanged.",
    )
    parser.add_argument(
        "--disable-item7-lm-token-floor",
        action="store_true",
        help="Disable the Item 7 LM-token floor while building the sampled sentence dataset.",
    )
    parser.add_argument(
        "--item7-min-lm-tokens",
        type=int,
        default=250,
        help="Minimum LM-token count for the Item 7 floor rule when enabled.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    diagnostic_output_dir = output_dir / "sample_sentence_diagnostics"

    diagnostics_cfg = SampleItemCleaningSentenceDiagnosticsConfig(
        source_items_dir=args.source_items_dir.resolve(),
        output_dir=diagnostic_output_dir,
        sample_doc_count=args.sample_doc_count,
        seed=args.seed,
        years=tuple(args.years) if args.years else None,
        item_codes=tuple(args.item_codes),
        selected_doc_ids_path=(
            args.selected_doc_ids_path.resolve()
            if args.selected_doc_ids_path is not None
            else None
        ),
        cleaning=ItemTextCleaningConfig(
            enforce_item7_lm_token_floor=not args.disable_item7_lm_token_floor,
            item7_min_lm_tokens=args.item7_min_lm_tokens,
        ),
        sentence_dataset=SentenceDatasetConfig(
            enabled=True,
            sentencizer_backend="spacy_blank_en_sentencizer",
            postprocess_policy=args.postprocess_policy,
            spacy_batch_size=32,
            token_length_batch_size=1024,
            drop_blank_sentences=True,
            compression="zstd",
        ),
        authority=FinbertAuthoritySpec(model_name=args.authority_model_name_or_path),
    )
    diagnostics_artifacts = run_sample_item_cleaning_sentence_diagnostics(diagnostics_cfg)
    recommendation = recommend_conservative_bucket_edges(
        diagnostics_artifacts.sentence_dataset_dir,
        item_codes=tuple(args.item_codes),
        years=tuple(args.years) if args.years else None,
        target_quantile=args.target_quantile,
        round_to=args.round_to,
        safety_margin_tokens=args.safety_margin_tokens,
        medium_edge_policy=(
            "target_quantile"
            if args.allow_medium_edge_reduction
            else "keep_current"
        ),
    )
    report_artifacts = write_bucket_edge_recommendation_report(recommendation, output_dir)

    payload = {
        "source_items_dir": str(args.source_items_dir.resolve()),
        "diagnostic_output_dir": str(diagnostics_artifacts.output_dir.resolve()),
        "selected_doc_ids_path": str(diagnostics_artifacts.selected_doc_ids_path.resolve()),
        "sentence_dataset_dir": str(diagnostics_artifacts.sentence_dataset_dir.resolve()),
        "report_output_dir": str(report_artifacts.output_dir.resolve()),
        "summary_by_bucket_parquet_path": str(
            report_artifacts.summary_by_bucket_parquet_path.resolve()
        ),
        "summary_by_bucket_csv_path": str(
            report_artifacts.summary_by_bucket_csv_path.resolve()
        ),
        "recommendation_summary_parquet_path": str(
            report_artifacts.recommendation_summary_parquet_path.resolve()
        ),
        "recommendation_summary_csv_path": str(
            report_artifacts.recommendation_summary_csv_path.resolve()
        ),
        "metadata_path": str(report_artifacts.metadata_path.resolve()),
        "env_overrides_path": str(report_artifacts.env_overrides_path.resolve()),
        "recommended_edges": asdict(recommendation.recommended_edges),
        "effective_bucket_lengths": asdict(recommendation.effective_bucket_lengths),
        "env_overrides": recommendation.env_overrides,
        "filters": recommendation.metadata["filters"],
        "target_quantile": recommendation.metadata["target_quantile"],
        "round_to": recommendation.metadata["round_to"],
        "safety_margin_tokens": recommendation.metadata["safety_margin_tokens"],
        "medium_edge_policy": recommendation.metadata["medium_edge_policy"],
        "estimated_padded_tokens_current": recommendation.metadata["estimated_padded_tokens_current"],
        "estimated_padded_tokens_rebucketed": recommendation.metadata["estimated_padded_tokens_rebucketed"],
        "estimated_padded_tokens_delta": recommendation.metadata["estimated_padded_tokens_delta"],
        "adds_extra_truncation_beyond_512": recommendation.metadata["adds_extra_truncation_beyond_512"],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

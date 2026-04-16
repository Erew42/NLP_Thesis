from __future__ import annotations

import argparse
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

from thesis_pkg.benchmarking.sample_item_cleaning_sentence_diagnostics import DEFAULT_ITEM_CODES
from thesis_pkg.benchmarking.sample_item_cleaning_sentence_diagnostics import DEFAULT_POSTPROCESS_POLICY
from thesis_pkg.benchmarking.sample_item_cleaning_sentence_diagnostics import DEFAULT_SOURCE_ITEMS_DIR
from thesis_pkg.benchmarking.contracts import DEFAULT_FINBERT_AUTHORITY
from thesis_pkg.benchmarking.contracts import FinbertAuthoritySpec
from thesis_pkg.benchmarking.contracts import ItemTextCleaningConfig
from thesis_pkg.benchmarking.contracts import SentenceDatasetConfig
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
    return DEFAULT_FINBERT_AUTHORITY.model_name


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Take a random document-level sample from items_analysis, run item cleaning plus "
            "sentence extraction with the updated Item 7 postprocess policy, and save figures "
            "for both item and sentence data under tmp/."
        )
    )
    parser.add_argument(
        "--source-items-dir",
        type=Path,
        default=ROOT / DEFAULT_SOURCE_ITEMS_DIR,
        help="Path to the items_analysis year shards.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "tmp" / Path(__file__).stem,
        help="Directory for sampled artifacts, reports, and figures.",
    )
    parser.add_argument(
        "--sample-doc-count",
        type=int,
        default=100,
        help="Number of unique documents to sample. The script keeps all requested items from those docs.",
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
        help="Benchmark item codes to keep. Defaults to item_1, item_1a, and item_7.",
    )
    parser.add_argument(
        "--selected-doc-ids-path",
        type=Path,
        default=None,
        help="Optional parquet/csv from a previous run to reuse the exact sampled doc_id universe.",
    )
    parser.add_argument(
        "--char-bin-width",
        type=int,
        default=25,
        help="Character-count histogram bin width for the sentence report.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=25,
        help="Number of longest sampled sentences to persist in the sentence report.",
    )
    parser.add_argument(
        "--postprocess-policy",
        type=str,
        default=DEFAULT_POSTPROCESS_POLICY,
        help=(
            "Sentence postprocess policy. The current updated Item 7 repair policy is "
            f"{DEFAULT_POSTPROCESS_POLICY!r}."
        ),
    )
    parser.add_argument(
        "--authority-model-name-or-path",
        type=str,
        default=_default_authority_model_name_or_path(),
        help=(
            "Hugging Face model id or a local tokenizer/model snapshot path for FinBERT token-length "
            "annotation. Use a local snapshot when network access is unavailable."
        ),
    )
    parser.add_argument(
        "--disable-item7-lm-token-floor",
        action="store_true",
        help="Disable the Item 7 LM-token floor drop rule entirely for this diagnostic run.",
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
    cfg = SampleItemCleaningSentenceDiagnosticsConfig(
        source_items_dir=args.source_items_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        sample_doc_count=args.sample_doc_count,
        seed=args.seed,
        years=tuple(args.years) if args.years else None,
        item_codes=tuple(args.item_codes),
        selected_doc_ids_path=args.selected_doc_ids_path.resolve() if args.selected_doc_ids_path is not None else None,
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
        char_bin_width=args.char_bin_width,
        top_n=args.top_n,
    )
    artifacts = run_sample_item_cleaning_sentence_diagnostics(cfg)
    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    payload = {
        "output_dir": str(artifacts.output_dir),
        "summary_path": str(artifacts.summary_path),
        "selected_doc_ids_path": str(artifacts.selected_doc_ids_path),
        "sampled_sections_path": str(artifacts.sampled_sections_path),
        "cleaned_item_scopes_path": str(artifacts.cleaned_item_scopes_path),
        "sentence_dataset_dir": str(artifacts.sentence_dataset_dir),
        "item_report_dir": str(artifacts.item_report_dir),
        "sentence_report_dir": str(artifacts.sentence_report_dir),
        "counts": summary["counts"],
        "sentence_dataset": summary["sentence_dataset"],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

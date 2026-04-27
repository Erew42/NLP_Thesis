from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from thesis_assets import build_all_assets
from thesis_assets import build_chapter_assets
from thesis_assets import build_single_asset


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build thesis-ready scaffold assets from existing run artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_all_parser = subparsers.add_parser("build-all", help="Build all registered assets.")
    _add_common_arguments(build_all_parser)

    build_chapter_parser = subparsers.add_parser("build-chapter", help="Build all assets for one chapter.")
    _add_common_arguments(build_chapter_parser)
    build_chapter_parser.add_argument("--chapter", choices=("chapter4", "chapter5"), required=True)

    build_asset_parser = subparsers.add_parser("build-asset", help="Build a single asset by asset_id.")
    _add_common_arguments(build_asset_parser)
    build_asset_parser.add_argument("--asset-id", required=True)

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    common_kwargs = {
        "run_id": args.run_id,
        "repo_root": args.repo_root,
        "lm2011_post_refinitiv_dir": args.lm2011_post_refinitiv_dir,
        "lm2011_extension_dir": args.lm2011_extension_dir,
        "lm2011_nw_lag_sensitivity_dir": args.lm2011_nw_lag_sensitivity_dir,
        "finbert_run_dir": args.finbert_run_dir,
        "finbert_robustness_dir": args.finbert_robustness_dir,
    }

    if args.command == "build-all":
        result = build_all_assets(**common_kwargs)
    elif args.command == "build-chapter":
        result = build_chapter_assets(chapter=args.chapter, **common_kwargs)
    elif args.command == "build-asset":
        result = build_single_asset(asset_id=args.asset_id, **common_kwargs)
    else:  # pragma: no cover - argparse enforces valid commands
        raise ValueError(f"Unsupported command {args.command!r}")

    failures = [asset_id for asset_id, build_result in result.asset_results.items() if build_result.status != "completed"]
    print(f"manifest={result.manifest_path}")
    for asset_id, build_result in result.asset_results.items():
        print(f"{asset_id}={build_result.status}")
        if build_result.failure_reason:
            print(f"{asset_id}_failure={build_result.failure_reason}")
    return 1 if failures else 0


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--lm2011-post-refinitiv-dir", type=Path, default=None)
    parser.add_argument("--lm2011-extension-dir", type=Path, default=None)
    parser.add_argument("--lm2011-nw-lag-sensitivity-dir", type=Path, default=None)
    parser.add_argument("--finbert-run-dir", type=Path, default=None)
    parser.add_argument("--finbert-robustness-dir", type=Path, default=None)


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT_ENV_VAR = "NLP_THESIS_REPO_ROOT"


def _resolve_repo_root() -> Path:
    candidates: list[Path] = []
    env_root = os.environ.get(REPO_ROOT_ENV_VAR)
    if env_root:
        candidates.append(Path(env_root).expanduser().resolve())

    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])

    script_path = Path(__file__).resolve()
    candidates.extend(script_path.parents)

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "thesis_assets").exists() and (candidate / "src" / "thesis_pkg" / "pipeline.py").exists():
            return candidate
    raise RuntimeError("Could not resolve repository root containing thesis_assets/ and src/thesis_pkg/pipeline.py")

ROOT = _resolve_repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thesis_assets import build_all_assets
from thesis_assets import build_chapter_assets
from thesis_assets import build_single_asset
from thesis_assets import resolve_colab_drive_root
from thesis_assets import resolve_default_data_profile
from thesis_assets import resolve_usage_run_paths


DEFAULT_PROFILE = resolve_default_data_profile()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Thin local/Colab entrypoint for the repo-root thesis_assets scaffold. "
            "Wraps the thesis_assets Python API with sec_ccm_unified_runner-style local and Drive path resolution."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_all_parser = subparsers.add_parser("build-all", help="Build all registered thesis assets.")
    _add_common_arguments(build_all_parser)

    build_chapter_parser = subparsers.add_parser("build-chapter", help="Build one thesis asset chapter registry.")
    _add_common_arguments(build_chapter_parser)
    build_chapter_parser.add_argument("--chapter", choices=("chapter4", "chapter5"), required=True)

    build_asset_parser = subparsers.add_parser("build-asset", help="Build a single thesis asset.")
    _add_common_arguments(build_asset_parser)
    build_asset_parser.add_argument("--asset-id", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = ROOT if args.repo_root is None else args.repo_root.resolve()

    resolved_paths = _resolve_run_paths(
        repo_root=repo_root,
        data_profile=args.data_profile,
        drive_data_root=args.drive_data_root,
        lm2011_post_refinitiv_dir=args.lm2011_post_refinitiv_dir,
        lm2011_extension_dir=args.lm2011_extension_dir,
        finbert_run_dir=args.finbert_run_dir,
    )

    build_kwargs = {
        "run_id": args.run_id,
        "repo_root": repo_root,
        "lm2011_post_refinitiv_dir": resolved_paths["lm2011_post_refinitiv_dir"],
        "lm2011_extension_dir": resolved_paths["lm2011_extension_dir"],
        "finbert_run_dir": resolved_paths["finbert_run_dir"],
    }
    if args.command == "build-all":
        result = build_all_assets(**build_kwargs)
    elif args.command == "build-chapter":
        result = build_chapter_assets(chapter=args.chapter, **build_kwargs)
    elif args.command == "build-asset":
        result = build_single_asset(asset_id=args.asset_id, **build_kwargs)
    else:  # pragma: no cover - argparse enforces valid commands
        raise ValueError(f"Unsupported command: {args.command!r}")

    asset_statuses = {
        asset_id: build_result.status
        for asset_id, build_result in sorted(result.asset_results.items())
    }
    payload = {
        "command": args.command,
        "run_id": args.run_id,
        "data_profile": args.data_profile,
        "repo_root": str(repo_root),
        "resolved_paths": {
            key: (str(value) if value is not None else None)
            for key, value in resolved_paths.items()
        },
        "manifest_path": str(result.manifest_path),
        "asset_statuses": asset_statuses,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if any(status != "completed" for status in asset_statuses.values()) else 0


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--data-profile",
        choices=("LOCAL_REPO", "COLAB_DRIVE", "EXPLICIT"),
        default=DEFAULT_PROFILE,
        help=(
            "LOCAL_REPO and COLAB_DRIVE follow sec_ccm_unified_runner-style result paths, "
            "while EXPLICIT requires path flags."
        ),
    )
    parser.add_argument("--repo-root", type=Path, default=None, help="Optional repo root override.")
    parser.add_argument(
        "--drive-data-root",
        type=Path,
        default=resolve_colab_drive_root() / "Data_LM",
        help="Drive data root for COLAB_DRIVE profile, usually /content/drive/MyDrive/Data_LM.",
    )
    parser.add_argument("--lm2011-post-refinitiv-dir", type=Path, default=None)
    parser.add_argument("--lm2011-extension-dir", type=Path, default=None)
    parser.add_argument("--finbert-run-dir", type=Path, default=None)


def _resolve_run_paths(
    *,
    repo_root: Path,
    data_profile: str,
    drive_data_root: Path,
    lm2011_post_refinitiv_dir: Path | None,
    lm2011_extension_dir: Path | None,
    finbert_run_dir: Path | None,
) -> dict[str, Path | None]:
    return resolve_usage_run_paths(
        repo_root=repo_root,
        data_profile=data_profile,
        drive_data_root=drive_data_root.resolve(),
        lm2011_post_refinitiv_dir=lm2011_post_refinitiv_dir,
        lm2011_extension_dir=lm2011_extension_dir,
        finbert_run_dir=finbert_run_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())

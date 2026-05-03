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
from thesis_assets.config.constants import RUN_FAMILY_FINBERT_ROBUSTNESS
from thesis_assets.config.constants import RUN_FAMILY_FINBERT_RUN
from thesis_assets.config.constants import RUN_FAMILY_LM2011_EXTENSION
from thesis_assets.config.constants import RUN_FAMILY_LM2011_EXTENSION_FINBERT_VISIBLE_PREFIX
from thesis_assets.config.constants import RUN_FAMILY_LM2011_EVENT_WINDOW_SENSITIVITY
from thesis_assets.config.constants import RUN_FAMILY_LM2011_NW_LAG_SENSITIVITY
from thesis_assets.config.constants import RUN_FAMILY_LM2011_POST_REFINITIV
from thesis_assets.submission_lock import DEFAULT_FINBERT_MODEL_NAME
from thesis_assets.submission_lock import DEFAULT_FINBERT_REVISION
from thesis_assets.submission_lock import FINBERT_INFERRED_REVISION_DISCLOSURE_ID
from thesis_assets.submission_lock import SubmissionLock
from thesis_assets.submission_lock import build_submission_lock_payload
from thesis_assets.submission_lock import load_submission_lock
from thesis_assets.submission_lock import validate_submission_lock
from thesis_assets.submission_lock import write_submission_lock


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

    init_lock_parser = subparsers.add_parser("init-lock", help="Create a submission_lock.json from explicit roots.")
    _add_lock_root_arguments(init_lock_parser)
    init_lock_parser.add_argument("--submission-root", type=Path, required=True)
    init_lock_parser.add_argument("--output", type=Path, required=True)
    init_lock_parser.add_argument("--run-id", required=True)
    init_lock_parser.add_argument(
        "--artifact-override",
        nargs=3,
        action="append",
        metavar=("ARTIFACT_KEY", "PATH", "REASON"),
        default=[],
        help="Explicit locked artifact override. Repeat as needed.",
    )
    init_lock_parser.add_argument(
        "--skip-finbert-inferred-revision-disclosure",
        action="store_true",
        help="Do not add the default warning-level FinBERT inferred-revision disclosure.",
    )

    validate_lock_parser = subparsers.add_parser("validate-lock", help="Validate a submission_lock.json.")
    validate_lock_parser.add_argument("--submission-lock", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "init-lock":
        return _init_lock(args)
    if args.command == "validate-lock":
        return _validate_lock(args)

    repo_root = ROOT if args.repo_root is None else args.repo_root.resolve()

    if args.submission_lock is not None:
        lock = load_submission_lock(args.submission_lock)
        _raise_if_lock_combined_with_run_roots(args)
        resolved_paths = _resolved_paths_from_lock(lock)
    else:
        resolved_paths = _resolve_run_paths(
            repo_root=repo_root,
            data_profile=args.data_profile,
            drive_data_root=args.drive_data_root,
            lm2011_post_refinitiv_dir=args.lm2011_post_refinitiv_dir,
            lm2011_extension_dir=args.lm2011_extension_dir,
            lm2011_extension_finbert_visible_prefix_dir=args.lm2011_extension_finbert_visible_prefix_dir,
            lm2011_nw_lag_sensitivity_dir=args.lm2011_nw_lag_sensitivity_dir,
            lm2011_event_window_sensitivity_dir=args.lm2011_event_window_sensitivity_dir,
            finbert_run_dir=args.finbert_run_dir,
            finbert_robustness_dir=args.finbert_robustness_dir,
        )

    build_kwargs = {
        "run_id": args.run_id,
        "repo_root": repo_root,
        "output_root": args.output_root.resolve() if args.output_root is not None else None,
        "lm2011_post_refinitiv_dir": (
            None if args.submission_lock is not None else resolved_paths["lm2011_post_refinitiv_dir"]
        ),
        "lm2011_extension_dir": (
            None if args.submission_lock is not None else resolved_paths["lm2011_extension_dir"]
        ),
        "lm2011_extension_finbert_visible_prefix_dir": (
            None
            if args.submission_lock is not None
            else resolved_paths["lm2011_extension_finbert_visible_prefix_dir"]
        ),
        "lm2011_nw_lag_sensitivity_dir": (
            None if args.submission_lock is not None else resolved_paths["lm2011_nw_lag_sensitivity_dir"]
        ),
        "lm2011_event_window_sensitivity_dir": (
            None if args.submission_lock is not None else resolved_paths["lm2011_event_window_sensitivity_dir"]
        ),
        "finbert_run_dir": (
            None if args.submission_lock is not None else resolved_paths["finbert_run_dir"]
        ),
        "finbert_robustness_dir": (
            None if args.submission_lock is not None else resolved_paths["finbert_robustness_dir"]
        ),
        "submission_lock_path": args.submission_lock,
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
        "submission_lock": str(args.submission_lock.resolve()) if args.submission_lock is not None else None,
        "repo_root": str(repo_root),
        "output_root": str(result.output_root),
        "resolved_paths": {
            key: (str(value) if value is not None else None)
            for key, value in resolved_paths.items()
        },
        "manifest_path": str(result.manifest_path),
        "asset_statuses": asset_statuses,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    has_failures = any(status != "completed" for status in asset_statuses.values())
    return 0 if (args.allow_failures or not has_failures) else 1


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
        "--output-root",
        type=Path,
        default=None,
        help="Optional override for the final thesis asset output directory, typically .../thesis_assets/<run_id>.",
    )
    parser.add_argument(
        "--drive-data-root",
        type=Path,
        default=resolve_colab_drive_root() / "Data_LM",
        help="Drive data root for COLAB_DRIVE profile, usually /content/drive/MyDrive/Data_LM.",
    )
    parser.add_argument("--lm2011-post-refinitiv-dir", type=Path, default=None)
    parser.add_argument("--lm2011-extension-dir", type=Path, default=None)
    parser.add_argument("--lm2011-extension-finbert-visible-prefix-dir", type=Path, default=None)
    parser.add_argument("--lm2011-nw-lag-sensitivity-dir", type=Path, default=None)
    parser.add_argument("--lm2011-event-window-sensitivity-dir", type=Path, default=None)
    parser.add_argument("--finbert-run-dir", type=Path, default=None)
    parser.add_argument("--finbert-robustness-dir", type=Path, default=None)
    parser.add_argument(
        "--submission-lock",
        type=Path,
        default=None,
        help="Use a submission_lock.json and strict retained-output resolution.",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Exit 0 even when one or more assets fail. The JSON payload still reports failed statuses.",
    )


def _add_lock_root_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lm2011-post-refinitiv-dir", type=Path, default=None)
    parser.add_argument("--lm2011-extension-dir", type=Path, default=None)
    parser.add_argument("--lm2011-extension-finbert-visible-prefix-dir", type=Path, default=None)
    parser.add_argument("--lm2011-nw-lag-sensitivity-dir", type=Path, default=None)
    parser.add_argument("--lm2011-event-window-sensitivity-dir", type=Path, default=None)
    parser.add_argument("--finbert-run-dir", type=Path, default=None)
    parser.add_argument("--finbert-robustness-dir", type=Path, default=None)


def _init_lock(args: argparse.Namespace) -> int:
    run_roots = {
        run_family: path.resolve()
        for run_family, path in _run_root_args(args).items()
        if path is not None
    }
    artifact_overrides = [
        (artifact_key, Path(path).resolve(), reason)
        for artifact_key, path, reason in args.artifact_override
    ]
    disclosures = []
    if not args.skip_finbert_inferred_revision_disclosure:
        disclosures.append(
            {
                "id": FINBERT_INFERRED_REVISION_DISCLOSURE_ID,
                "severity": "warning",
                "run_family": RUN_FAMILY_FINBERT_RUN,
                "model_name": DEFAULT_FINBERT_MODEL_NAME,
                "model_revision": DEFAULT_FINBERT_REVISION,
                "tokenizer_revision": DEFAULT_FINBERT_REVISION,
                "reason": (
                    "Retained FinBERT artifacts have incomplete persisted revision provenance; "
                    "the revision is inferred and disclosed for deterministic retained-output rebuilds."
                ),
            }
        )
    payload = build_submission_lock_payload(
        submission_root=args.submission_root,
        run_id=args.run_id,
        run_roots=run_roots,
        artifact_overrides=artifact_overrides,
        provenance_disclosures=disclosures,
    )
    output_path = write_submission_lock(args.output, payload)
    lock = validate_submission_lock(output_path)
    print(
        json.dumps(
            {
                "submission_lock": str(output_path),
                "submission_root": str(lock.submission_root),
                "run_roots": sorted(lock.run_roots),
                "artifact_overrides": sorted(lock.artifact_overrides),
                "provenance_disclosures": sorted(lock.disclosure_ids()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _validate_lock(args: argparse.Namespace) -> int:
    lock = validate_submission_lock(args.submission_lock)
    print(
        json.dumps(
            {
                "submission_lock": str(lock.path),
                "submission_root": str(lock.submission_root),
                "run_roots": sorted(lock.run_roots),
                "artifact_overrides": sorted(lock.artifact_overrides),
                "provenance_disclosures": sorted(lock.disclosure_ids()),
                "status": "valid",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _run_root_args(args: argparse.Namespace) -> dict[str, Path | None]:
    return {
        RUN_FAMILY_LM2011_POST_REFINITIV: args.lm2011_post_refinitiv_dir,
        RUN_FAMILY_LM2011_EXTENSION: args.lm2011_extension_dir,
        RUN_FAMILY_LM2011_EXTENSION_FINBERT_VISIBLE_PREFIX: (
            args.lm2011_extension_finbert_visible_prefix_dir
        ),
        RUN_FAMILY_LM2011_NW_LAG_SENSITIVITY: args.lm2011_nw_lag_sensitivity_dir,
        RUN_FAMILY_LM2011_EVENT_WINDOW_SENSITIVITY: args.lm2011_event_window_sensitivity_dir,
        RUN_FAMILY_FINBERT_RUN: args.finbert_run_dir,
        RUN_FAMILY_FINBERT_ROBUSTNESS: args.finbert_robustness_dir,
    }


def _raise_if_lock_combined_with_run_roots(args: argparse.Namespace) -> None:
    supplied = [
        run_family
        for run_family, path in _run_root_args(args).items()
        if path is not None
    ]
    if supplied:
        raise ValueError(
            "Do not combine --submission-lock with explicit run-root flags: "
            + ", ".join(sorted(supplied))
        )


def _resolved_paths_from_lock(lock: SubmissionLock) -> dict[str, Path | None]:
    return {
        "lm2011_post_refinitiv_dir": lock.run_roots.get(RUN_FAMILY_LM2011_POST_REFINITIV).path
        if RUN_FAMILY_LM2011_POST_REFINITIV in lock.run_roots
        else None,
        "lm2011_extension_dir": lock.run_roots.get(RUN_FAMILY_LM2011_EXTENSION).path
        if RUN_FAMILY_LM2011_EXTENSION in lock.run_roots
        else None,
        "lm2011_extension_finbert_visible_prefix_dir": (
            lock.run_roots.get(RUN_FAMILY_LM2011_EXTENSION_FINBERT_VISIBLE_PREFIX).path
            if RUN_FAMILY_LM2011_EXTENSION_FINBERT_VISIBLE_PREFIX in lock.run_roots
            else None
        ),
        "lm2011_nw_lag_sensitivity_dir": lock.run_roots.get(RUN_FAMILY_LM2011_NW_LAG_SENSITIVITY).path
        if RUN_FAMILY_LM2011_NW_LAG_SENSITIVITY in lock.run_roots
        else None,
        "lm2011_event_window_sensitivity_dir": (
            lock.run_roots.get(RUN_FAMILY_LM2011_EVENT_WINDOW_SENSITIVITY).path
            if RUN_FAMILY_LM2011_EVENT_WINDOW_SENSITIVITY in lock.run_roots
            else None
        ),
        "finbert_run_dir": lock.run_roots.get(RUN_FAMILY_FINBERT_RUN).path
        if RUN_FAMILY_FINBERT_RUN in lock.run_roots
        else None,
        "finbert_robustness_dir": lock.run_roots.get(RUN_FAMILY_FINBERT_ROBUSTNESS).path
        if RUN_FAMILY_FINBERT_ROBUSTNESS in lock.run_roots
        else None,
    }


def _resolve_run_paths(
    *,
    repo_root: Path,
    data_profile: str,
    drive_data_root: Path,
    lm2011_post_refinitiv_dir: Path | None,
    lm2011_extension_dir: Path | None,
    lm2011_extension_finbert_visible_prefix_dir: Path | None,
    lm2011_nw_lag_sensitivity_dir: Path | None,
    lm2011_event_window_sensitivity_dir: Path | None,
    finbert_run_dir: Path | None,
    finbert_robustness_dir: Path | None,
) -> dict[str, Path | None]:
    return resolve_usage_run_paths(
        repo_root=repo_root,
        data_profile=data_profile,
        drive_data_root=drive_data_root.resolve(),
        lm2011_post_refinitiv_dir=lm2011_post_refinitiv_dir,
        lm2011_extension_dir=lm2011_extension_dir,
        lm2011_extension_finbert_visible_prefix_dir=lm2011_extension_finbert_visible_prefix_dir,
        lm2011_nw_lag_sensitivity_dir=lm2011_nw_lag_sensitivity_dir,
        lm2011_event_window_sensitivity_dir=lm2011_event_window_sensitivity_dir,
        finbert_run_dir=finbert_run_dir,
        finbert_robustness_dir=finbert_robustness_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())

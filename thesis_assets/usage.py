from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence


IN_COLAB = "google.colab" in sys.modules
LOCAL_REPO_PROFILE = "LOCAL_REPO"
COLAB_DRIVE_PROFILE = "COLAB_DRIVE"
EXPLICIT_PROFILE = "EXPLICIT"


def resolve_default_data_profile(*, in_colab: bool | None = None) -> str:
    active_colab = IN_COLAB if in_colab is None else in_colab
    return COLAB_DRIVE_PROFILE if active_colab else LOCAL_REPO_PROFILE


def resolve_colab_drive_root() -> Path:
    for candidate in (
        Path("/content/drive/MyDrive"),
        Path("/content/drive/My Drive"),
        Path("/content/drive"),
    ):
        if candidate.exists():
            return candidate.resolve()
    return Path("/content/drive")


def resolve_usage_run_paths(
    *,
    repo_root: Path,
    data_profile: str,
    drive_data_root: Path | None = None,
    lm2011_post_refinitiv_dir: Path | None = None,
    lm2011_extension_dir: Path | None = None,
    finbert_run_dir: Path | None = None,
) -> dict[str, Path | None]:
    if data_profile == EXPLICIT_PROFILE:
        return {
            "lm2011_post_refinitiv_dir": _resolve_optional_path(lm2011_post_refinitiv_dir),
            "lm2011_extension_dir": _resolve_optional_path(lm2011_extension_dir),
            "finbert_run_dir": _resolve_optional_path(finbert_run_dir),
        }

    if data_profile == LOCAL_REPO_PROFILE:
        defaults = resolve_local_profile_paths(repo_root.resolve())
    elif data_profile == COLAB_DRIVE_PROFILE:
        resolved_drive_root = (
            drive_data_root.resolve()
            if drive_data_root is not None
            else (resolve_colab_drive_root() / "Data_LM").resolve()
        )
        defaults = resolve_colab_profile_paths(resolved_drive_root)
    else:
        raise ValueError(f"Unsupported data_profile: {data_profile!r}")

    return {
        "lm2011_post_refinitiv_dir": _resolve_optional_path(lm2011_post_refinitiv_dir)
        or defaults["lm2011_post_refinitiv_dir"],
        "lm2011_extension_dir": _resolve_optional_path(lm2011_extension_dir)
        or defaults["lm2011_extension_dir"],
        "finbert_run_dir": _resolve_optional_path(finbert_run_dir)
        or defaults["finbert_run_dir"],
    }


def resolve_local_profile_paths(repo_root: Path) -> dict[str, Path | None]:
    full_data_root = repo_root / "full_data_run"
    sample_results_root = full_data_root / "sample_5pct_seed42" / "results"
    unified_root = sample_results_root / "sec_ccm_unified_runner" / "local_sample"
    return {
        "lm2011_post_refinitiv_dir": _first_existing_path(
            (
                unified_root / "lm2011_post_refinitiv",
                _resolve_direct_or_snapshot_dir(full_data_root, "lm2011_post_refinitiv"),
                sample_results_root / "lm2011_sample_post_refinitiv_runner",
                full_data_root / "lm2011_sample_post_refinitiv_runner",
            )
        ),
        "lm2011_extension_dir": _first_existing_path(
            (
                unified_root / "lm2011_extension",
                _resolve_direct_or_snapshot_dir(full_data_root, "lm2011_extension"),
            )
        ),
        "finbert_run_dir": _first_existing_path(
            (
                _resolve_latest_finbert_run(unified_root / "finbert_item_analysis"),
                _resolve_latest_finbert_run(sample_results_root / "finbert_item_analysis_runner"),
            )
        ),
    }


def resolve_colab_profile_paths(drive_data_root: Path) -> dict[str, Path | None]:
    results_root = drive_data_root / "results"
    unified_root = results_root / "sec_ccm_unified_runner"
    return {
        "lm2011_post_refinitiv_dir": _first_existing_path(
            (
                unified_root / "lm2011_post_refinitiv",
                results_root / "lm2011_sample_post_refinitiv_runner",
            )
        ),
        "lm2011_extension_dir": _first_existing_path((unified_root / "lm2011_extension",)),
        "finbert_run_dir": _first_existing_path(
            (
                _resolve_latest_finbert_run(unified_root / "finbert_item_analysis"),
                _resolve_latest_finbert_run(results_root / "finbert_item_analysis_runner"),
            )
        ),
    }


def _resolve_direct_or_snapshot_dir(parent: Path, stem: str) -> Path | None:
    direct = parent / stem
    if direct.exists():
        return direct.resolve()

    snapshots = [
        candidate / stem
        for candidate in parent.glob(f"{stem}-*")
        if candidate.is_dir() and (candidate / stem).exists()
    ]
    if not snapshots:
        return None
    return max(snapshots, key=lambda path: path.stat().st_mtime).resolve()


def _resolve_latest_finbert_run(parent: Path) -> Path | None:
    if not parent.exists() or not parent.is_dir():
        return None

    candidates: list[Path] = []
    for child in parent.iterdir():
        if not child.is_dir() or child.name.startswith("_"):
            continue
        if (child / "run_manifest.json").exists() and (child / "item_features_long.parquet").exists():
            candidates.append(child.resolve())

    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _first_existing_path(candidates: Sequence[Path | None]) -> Path | None:
    for candidate in candidates:
        if candidate is None:
            continue
        if candidate.exists():
            return candidate.resolve()
    return None


def _resolve_optional_path(path: Path | None) -> Path | None:
    return None if path is None else path.resolve()

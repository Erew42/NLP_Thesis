from __future__ import annotations

from thesis_assets.bootstrap import ensure_repo_src_on_path

ensure_repo_src_on_path()

from thesis_assets.api import build_all_assets
from thesis_assets.api import build_chapter_assets
from thesis_assets.api import build_single_asset
from thesis_assets.usage import resolve_colab_drive_root
from thesis_assets.usage import resolve_default_data_profile
from thesis_assets.usage import resolve_usage_run_paths

__all__ = [
    "build_all_assets",
    "build_chapter_assets",
    "build_single_asset",
    "resolve_colab_drive_root",
    "resolve_default_data_profile",
    "resolve_usage_run_paths",
]

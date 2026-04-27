from __future__ import annotations

import logging
from pathlib import Path

from thesis_assets.bootstrap import ensure_repo_src_on_path
from thesis_assets.bootstrap import resolve_repo_root
from thesis_assets.builders import build_asset
from thesis_assets.config import BUILD_LOG_FILENAME
from thesis_assets.config import MANIFEST_FILENAME
from thesis_assets.config import RUN_FAMILY_FINBERT_ROBUSTNESS
from thesis_assets.config import RUN_FAMILY_FINBERT_RUN
from thesis_assets.config import RUN_FAMILY_LM2011_EXTENSION
from thesis_assets.config import RUN_FAMILY_LM2011_NW_LAG_SENSITIVITY
from thesis_assets.config import RUN_FAMILY_LM2011_POST_REFINITIV
from thesis_assets.config import build_output_root
from thesis_assets.config import prepare_output_dirs
from thesis_assets.registry import load_asset_by_id
from thesis_assets.registry import load_assets_by_chapter
from thesis_assets.registry import load_registry
from thesis_assets.renderers import write_build_manifest
from thesis_assets.specs import BuildContext
from thesis_assets.specs import BuildSessionResult


ensure_repo_src_on_path()


def build_all_assets(
    *,
    run_id: str,
    repo_root: Path | None = None,
    output_root: Path | None = None,
    lm2011_post_refinitiv_dir: Path | None = None,
    lm2011_extension_dir: Path | None = None,
    lm2011_nw_lag_sensitivity_dir: Path | None = None,
    finbert_run_dir: Path | None = None,
    finbert_robustness_dir: Path | None = None,
) -> BuildSessionResult:
    return _build_assets(
        load_registry(),
        run_id=run_id,
        repo_root=repo_root,
        output_root=output_root,
        lm2011_post_refinitiv_dir=lm2011_post_refinitiv_dir,
        lm2011_extension_dir=lm2011_extension_dir,
        lm2011_nw_lag_sensitivity_dir=lm2011_nw_lag_sensitivity_dir,
        finbert_run_dir=finbert_run_dir,
        finbert_robustness_dir=finbert_robustness_dir,
    )


def build_chapter_assets(
    *,
    chapter: str,
    run_id: str,
    repo_root: Path | None = None,
    output_root: Path | None = None,
    lm2011_post_refinitiv_dir: Path | None = None,
    lm2011_extension_dir: Path | None = None,
    lm2011_nw_lag_sensitivity_dir: Path | None = None,
    finbert_run_dir: Path | None = None,
    finbert_robustness_dir: Path | None = None,
) -> BuildSessionResult:
    return _build_assets(
        load_assets_by_chapter(chapter),
        run_id=run_id,
        repo_root=repo_root,
        output_root=output_root,
        lm2011_post_refinitiv_dir=lm2011_post_refinitiv_dir,
        lm2011_extension_dir=lm2011_extension_dir,
        lm2011_nw_lag_sensitivity_dir=lm2011_nw_lag_sensitivity_dir,
        finbert_run_dir=finbert_run_dir,
        finbert_robustness_dir=finbert_robustness_dir,
    )


def build_single_asset(
    *,
    asset_id: str,
    run_id: str,
    repo_root: Path | None = None,
    output_root: Path | None = None,
    lm2011_post_refinitiv_dir: Path | None = None,
    lm2011_extension_dir: Path | None = None,
    lm2011_nw_lag_sensitivity_dir: Path | None = None,
    finbert_run_dir: Path | None = None,
    finbert_robustness_dir: Path | None = None,
) -> BuildSessionResult:
    return _build_assets(
        (load_asset_by_id(asset_id),),
        run_id=run_id,
        repo_root=repo_root,
        output_root=output_root,
        lm2011_post_refinitiv_dir=lm2011_post_refinitiv_dir,
        lm2011_extension_dir=lm2011_extension_dir,
        lm2011_nw_lag_sensitivity_dir=lm2011_nw_lag_sensitivity_dir,
        finbert_run_dir=finbert_run_dir,
        finbert_robustness_dir=finbert_robustness_dir,
    )


def _build_assets(
    specs,
    *,
    run_id: str,
    repo_root: Path | None,
    output_root: Path | None,
    lm2011_post_refinitiv_dir: Path | None,
    lm2011_extension_dir: Path | None,
    lm2011_nw_lag_sensitivity_dir: Path | None,
    finbert_run_dir: Path | None,
    finbert_robustness_dir: Path | None,
) -> BuildSessionResult:
    repo_root = resolve_repo_root() if repo_root is None else repo_root.resolve()
    ensure_repo_src_on_path(repo_root)

    output_root = build_output_root(repo_root, run_id) if output_root is None else output_root.resolve()
    _ensure_output_root_is_safe(repo_root, output_root)
    output_dirs = prepare_output_dirs(output_root)
    logger = _configure_logger(output_dirs["logs"] / BUILD_LOG_FILENAME)
    explicit_run_roots = {
        run_family: path.resolve()
        for run_family, path in {
            RUN_FAMILY_LM2011_POST_REFINITIV: lm2011_post_refinitiv_dir,
            RUN_FAMILY_LM2011_EXTENSION: lm2011_extension_dir,
            RUN_FAMILY_LM2011_NW_LAG_SENSITIVITY: lm2011_nw_lag_sensitivity_dir,
            RUN_FAMILY_FINBERT_RUN: finbert_run_dir,
            RUN_FAMILY_FINBERT_ROBUSTNESS: finbert_robustness_dir,
        }.items()
        if path is not None
    }

    context = BuildContext(
        repo_root=repo_root,
        run_id=run_id,
        output_root=output_root,
        output_dirs=output_dirs,
        logger=logger,
        explicit_run_roots=explicit_run_roots,
    )

    asset_results = {}
    for spec in specs:
        asset_results[spec.asset_id] = build_asset(context, spec)

    manifest_path = write_build_manifest(context, asset_results, output_root / MANIFEST_FILENAME)
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)

    return BuildSessionResult(
        run_id=run_id,
        output_root=output_root,
        manifest_path=manifest_path,
        asset_results=asset_results,
    )


def _ensure_output_root_is_safe(repo_root: Path, output_root: Path) -> None:
    thesis_assets_root = (repo_root / "thesis_assets").resolve()
    if output_root.resolve().is_relative_to(thesis_assets_root):
        raise ValueError("Generated outputs must not be written inside thesis_assets/.")


def _configure_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"thesis_assets.{log_path.resolve()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger

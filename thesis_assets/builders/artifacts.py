from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from thesis_assets.config.constants import ARTIFACT_ALTERNATE_FILENAMES
from thesis_assets.config.constants import ARTIFACT_FILENAMES
from thesis_assets.config.constants import ARTIFACT_MANIFEST_KEYS
from thesis_assets.config.constants import RUN_FAMILY_SENTINEL_ARTIFACTS
from thesis_assets.config.constants import RUN_FAMILY_LM2011_NW_LAG_SENSITIVITY
from thesis_assets.config.constants import RUN_MANIFEST_FILENAMES
from thesis_assets.config.runtime import candidate_run_roots
from thesis_assets.errors import MissingArtifactError
from thesis_assets.errors import ResolutionError
from thesis_assets.specs import ArtifactRequirement
from thesis_assets.specs import AssetSpec
from thesis_assets.specs import BuildContext
from thesis_assets.specs import ResolvedArtifact
from thesis_assets.specs import ResolvedRun
from thesis_pkg.benchmarking.manifest_contracts import resolve_manifest_path


def resolve_run(context: BuildContext, run_family: str) -> ResolvedRun:
    cached = context.resolved_runs.get(run_family)
    if cached is not None:
        return cached

    explicit_root = context.explicit_run_roots.get(run_family)
    if explicit_root is not None:
        resolved = _resolve_explicit_run(run_family, explicit_root)
    else:
        resolved = _auto_resolve_run(context.repo_root, run_family)

    context.resolved_runs[run_family] = resolved
    context.logger.info("Resolved run family %s to %s (%s)", run_family, resolved.root, resolved.source)
    return resolved


def resolve_required_artifacts(
    context: BuildContext,
    spec: AssetSpec,
) -> dict[str, ResolvedArtifact]:
    resolved: dict[str, ResolvedArtifact] = {}
    for requirement in spec.required_artifacts:
        try:
            artifact = resolve_artifact(context, requirement)
        except MissingArtifactError:
            if requirement.required:
                raise
            continue
        resolved[requirement.logical_name] = artifact
        context.resolved_artifacts[f"{spec.asset_id}:{requirement.logical_name}"] = artifact
    return resolved


def resolve_artifact(
    context: BuildContext,
    requirement: ArtifactRequirement,
) -> ResolvedArtifact:
    run = resolve_run(context, requirement.run_family)
    candidates = _artifact_candidates(run, requirement)
    for candidate in candidates:
        if candidate.exists():
            _validate_required_columns(candidate, requirement)
            return ResolvedArtifact(requirement=requirement, path=candidate.resolve(), run=run)

    candidate_text = ", ".join(str(path) for path in candidates) or "<none>"
    raise MissingArtifactError(
        f"Required artifact {requirement.logical_name!r} could not be resolved. Checked: {candidate_text}"
    )


def scan_parquet_artifact(artifact: ResolvedArtifact) -> pl.LazyFrame:
    return pl.scan_parquet([str(path) for path in parquet_artifact_paths(artifact)])


def parquet_artifact_paths(artifact: ResolvedArtifact) -> tuple[Path, ...]:
    path = artifact.path
    if path.is_dir():
        paths = tuple(sorted(candidate.resolve() for candidate in path.glob("*.parquet") if candidate.is_file()))
        if not paths:
            raise MissingArtifactError(f"Parquet artifact directory contains no parquet shards: {path}")
        return paths
    return (path,)


def _resolve_explicit_run(run_family: str, root: Path) -> ResolvedRun:
    resolved_root = root.resolve()
    if not resolved_root.exists():
        raise ResolutionError(f"Explicit {run_family} root does not exist: {resolved_root}")
    if not resolved_root.is_dir():
        raise ResolutionError(f"Explicit {run_family} root is not a directory: {resolved_root}")
    manifest_path = _find_manifest_path(resolved_root, run_family)
    manifest = _read_manifest_json(manifest_path) if manifest_path is not None else None
    return ResolvedRun(
        run_family=run_family,
        root=resolved_root,
        source="explicit",
        manifest_path=manifest_path,
        manifest=manifest,
    )


def _auto_resolve_run(repo_root: Path, run_family: str) -> ResolvedRun:
    candidates = [
        candidate.resolve()
        for candidate in candidate_run_roots(repo_root, run_family)
        if _is_compatible_run_root(candidate, run_family)
    ]
    if not candidates:
        raise ResolutionError(
            f"Could not auto-resolve {run_family}. Pass an explicit source directory for this run family."
        )
    if len(candidates) > 1:
        joined = ", ".join(str(candidate) for candidate in candidates)
        raise ResolutionError(
            f"Auto-resolution for {run_family} is ambiguous. Pass an explicit source directory. Candidates: {joined}"
        )
    root = candidates[0]
    manifest_path = _find_manifest_path(root, run_family)
    manifest = _read_manifest_json(manifest_path) if manifest_path is not None else None
    return ResolvedRun(
        run_family=run_family,
        root=root,
        source="auto",
        manifest_path=manifest_path,
        manifest=manifest,
    )


def _is_compatible_run_root(root: Path, run_family: str) -> bool:
    if not root.exists() or not root.is_dir():
        return False
    if _find_manifest_path(root, run_family) is not None:
        return True
    if run_family == RUN_FAMILY_LM2011_NW_LAG_SENSITIVITY:
        return all(
            (root / filename).exists()
            for artifact_key in RUN_FAMILY_SENTINEL_ARTIFACTS[run_family]
            for filename in _artifact_filenames(artifact_key)
        )
    return any(
        (root / filename).exists()
        for artifact_key in RUN_FAMILY_SENTINEL_ARTIFACTS[run_family]
        for filename in _artifact_filenames(artifact_key)
    )


def _find_manifest_path(root: Path, run_family: str) -> Path | None:
    for filename in RUN_MANIFEST_FILENAMES[run_family]:
        candidate = root / filename
        if candidate.exists():
            return candidate.resolve()
    return None


def _read_manifest_json(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_candidates(run: ResolvedRun, requirement: ArtifactRequirement) -> tuple[Path, ...]:
    candidates: list[Path] = []
    seen: set[Path] = set()

    if run.manifest is not None and run.manifest_path is not None:
        manifest = run.manifest
        artifacts = manifest.get("artifacts")
        stages = manifest.get("stages")
        top_level = manifest
        for key in ARTIFACT_MANIFEST_KEYS.get(requirement.artifact_key, ()):
            if isinstance(artifacts, dict):
                raw_value = artifacts.get(key)
                if isinstance(raw_value, str):
                    _append_candidate(
                        candidates,
                        seen,
                        _resolve_manifest_value(raw_value, run.manifest_path, manifest),
                    )
            if isinstance(stages, dict):
                stage_payload = stages.get(key)
                if isinstance(stage_payload, dict):
                    raw_artifact_path = stage_payload.get("artifact_path")
                    if isinstance(raw_artifact_path, str):
                        _append_candidate(
                            candidates,
                            seen,
                            _resolve_manifest_value(raw_artifact_path, run.manifest_path, manifest),
                        )
            raw_top_level = top_level.get(key)
            if isinstance(raw_top_level, str):
                _append_candidate(
                    candidates,
                    seen,
                    _resolve_manifest_value(raw_top_level, run.manifest_path, manifest),
                )

    for filename in _artifact_filenames(requirement.artifact_key):
        if requirement.relative_subdir is not None:
            _append_candidate(candidates, seen, run.root / requirement.relative_subdir / filename)
        _append_candidate(candidates, seen, run.root / filename)
    return tuple(candidates)


def _artifact_filenames(artifact_key: str) -> tuple[str, ...]:
    return (ARTIFACT_FILENAMES[artifact_key], *ARTIFACT_ALTERNATE_FILENAMES.get(artifact_key, ()))


def _resolve_manifest_value(raw_path: str, manifest_path: Path, manifest: dict[str, object]) -> Path:
    resolved = resolve_manifest_path(
        raw_path,
        manifest_path=manifest_path,
        path_semantics=manifest.get("path_semantics"),
    )
    if resolved is None:
        raise MissingArtifactError(f"Manifest path value {raw_path!r} could not be resolved from {manifest_path}")
    return resolved


def _append_candidate(candidates: list[Path], seen: set[Path], candidate: Path) -> None:
    resolved = candidate.resolve() if candidate.exists() else candidate
    if resolved in seen:
        return
    seen.add(resolved)
    candidates.append(candidate)


def _validate_required_columns(path: Path, requirement: ArtifactRequirement) -> None:
    if not requirement.required_columns:
        return
    if requirement.artifact_kind != "parquet":
        return
    schema_path: str | list[str]
    if path.is_dir():
        paths = sorted(candidate for candidate in path.glob("*.parquet") if candidate.is_file())
        if not paths:
            raise MissingArtifactError(
                f"Artifact directory {path} contains no parquet shards for {requirement.logical_name!r}."
            )
        schema_path = [str(candidate) for candidate in paths]
    else:
        schema_path = str(path)
    schema = pl.scan_parquet(schema_path).collect_schema()
    missing = [column for column in requirement.required_columns if column not in schema]
    if missing:
        raise MissingArtifactError(
            f"Artifact {path} is missing required columns for {requirement.logical_name!r}: {missing}"
        )

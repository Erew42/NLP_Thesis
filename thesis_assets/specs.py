from __future__ import annotations

import logging
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArtifactRequirement:
    logical_name: str
    run_family: str
    artifact_key: str
    required: bool = True
    required_columns: tuple[str, ...] = ()
    relative_subdir: str | None = None


@dataclass(frozen=True)
class AssetSpec:
    asset_id: str
    chapter: str
    asset_kind: str
    output_stem: str
    caption_stub: str
    notes_stub: str
    sample_contract_id: str
    builder_id: str
    required_artifacts: tuple[ArtifactRequirement, ...]


@dataclass(frozen=True)
class ResolvedRun:
    run_family: str
    root: Path
    source: str
    manifest_path: Path | None
    manifest: dict[str, Any] | None


@dataclass(frozen=True)
class ResolvedArtifact:
    requirement: ArtifactRequirement
    path: Path
    run: ResolvedRun


@dataclass
class BuildContext:
    repo_root: Path
    run_id: str
    output_root: Path
    output_dirs: dict[str, Path]
    logger: logging.Logger
    explicit_run_roots: dict[str, Path]
    resolved_runs: dict[str, ResolvedRun] = field(default_factory=dict)
    resolved_artifacts: dict[str, ResolvedArtifact] = field(default_factory=dict)


@dataclass(frozen=True)
class BuildResult:
    asset_id: str
    chapter: str
    asset_kind: str
    sample_contract_id: str
    status: str
    resolved_inputs: dict[str, str]
    output_paths: dict[str, str]
    row_counts: dict[str, int]
    warnings: tuple[str, ...] = ()
    failure_reason: str | None = None


@dataclass(frozen=True)
class BuildSessionResult:
    run_id: str
    output_root: Path
    manifest_path: Path
    asset_results: dict[str, BuildResult]

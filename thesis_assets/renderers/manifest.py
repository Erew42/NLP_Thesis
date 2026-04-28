from __future__ import annotations

from pathlib import Path

from thesis_assets.config.constants import BUILD_LOG_FILENAME
from thesis_assets.specs import BuildContext
from thesis_assets.specs import BuildResult
from thesis_pkg.benchmarking.run_logging import utc_timestamp
from thesis_pkg.benchmarking.run_logging import write_json


def write_build_manifest(
    context: BuildContext,
    asset_results: dict[str, BuildResult],
    path: Path,
) -> Path:
    def _display_path(raw_path: str) -> str:
        if context.submission_lock is None:
            return raw_path
        try:
            resolved = Path(raw_path).resolve()
        except OSError:
            return raw_path
        root = context.submission_lock.submission_root
        return resolved.relative_to(root).as_posix() if resolved.is_relative_to(root) else raw_path

    payload = {
        "runner_name": "thesis_assets",
        "generated_at_utc": utc_timestamp(),
        "run_id": context.run_id,
        "output_root": str(context.output_root),
        "strict_submission": context.strict_submission,
        "submission_lock": (
            context.submission_lock.metadata_payload()
            if context.submission_lock is not None
            else None
        ),
        "submission_warnings": sorted(set(context.submission_warnings)),
        "source_roots": {
            run_family: _display_path(str(resolved.root))
            for run_family, resolved in sorted(context.resolved_runs.items())
        },
        "logs": {
            "build_log": str(context.output_dirs["logs"] / BUILD_LOG_FILENAME),
        },
        "asset_statuses": {
            asset_id: result.status
            for asset_id, result in sorted(asset_results.items())
        },
        "assets": {
            asset_id: {
                "asset_id": result.asset_id,
                "chapter": result.chapter,
                "asset_kind": result.asset_kind,
                "sample_contract_id": result.sample_contract_id,
                "status": result.status,
                "resolved_inputs": {
                    key: _display_path(value)
                    for key, value in result.resolved_inputs.items()
                },
                "output_paths": {
                    key: _display_path(value)
                    for key, value in result.output_paths.items()
                },
                "row_counts": result.row_counts,
                "warnings": list(result.warnings),
                "failure_reason": result.failure_reason,
            }
            for asset_id, result in sorted(asset_results.items())
        },
    }
    write_json(path, payload)
    return path

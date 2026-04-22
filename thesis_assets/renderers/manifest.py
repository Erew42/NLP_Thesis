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
    payload = {
        "runner_name": "thesis_assets",
        "generated_at_utc": utc_timestamp(),
        "run_id": context.run_id,
        "output_root": str(context.output_root),
        "source_roots": {
            run_family: str(resolved.root)
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
                "resolved_inputs": result.resolved_inputs,
                "output_paths": result.output_paths,
                "row_counts": result.row_counts,
                "warnings": list(result.warnings),
                "failure_reason": result.failure_reason,
            }
            for asset_id, result in sorted(asset_results.items())
        },
    }
    write_json(path, payload)
    return path

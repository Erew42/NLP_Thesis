from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from thesis_pkg.pipelines.refinitiv.lseg_api_common import write_parquet_atomic


@dataclass(frozen=True)
class RecoveryArtifact:
    recovery_mode: str
    output_path: Path
    row_count: int
    source_artifacts: dict[str, Path]

    def to_dict(self) -> dict[str, object]:
        return {
            "recovery_mode": self.recovery_mode,
            "output_path": str(self.output_path),
            "row_count": self.row_count,
            "source_artifacts": {key: str(value) for key, value in self.source_artifacts.items()},
        }


def build_lookup_unresolved_recovery_artifact(
    *,
    resolution_parquet_path: Path | str,
    output_path: Path | str,
) -> RecoveryArtifact:
    resolution_parquet_path = Path(resolution_parquet_path)
    output_path = Path(output_path)
    resolution_df = pl.read_parquet(resolution_parquet_path)
    unresolved_df = resolution_df.filter(pl.col("effective_collection_ric").is_null())
    write_parquet_atomic(unresolved_df, output_path)
    return RecoveryArtifact(
        recovery_mode="lookup_unresolved",
        output_path=output_path,
        row_count=int(unresolved_df.height),
        source_artifacts={"resolution_parquet": resolution_parquet_path},
    )


def build_ownership_unresolved_recovery_artifact(
    *,
    row_summary_parquet_path: Path | str,
    output_path: Path | str,
) -> RecoveryArtifact:
    row_summary_parquet_path = Path(row_summary_parquet_path)
    output_path = Path(output_path)
    row_summary_df = pl.read_parquet(row_summary_parquet_path)
    unresolved_df = row_summary_df.filter(
        pl.col("retrieval_eligible").fill_null(False)
        & (pl.col("ownership_rows_returned").fill_null(0) == 0)
    )
    write_parquet_atomic(unresolved_df, output_path)
    return RecoveryArtifact(
        recovery_mode="ownership_unresolved",
        output_path=output_path,
        row_count=int(unresolved_df.height),
        source_artifacts={"row_summary_parquet": row_summary_parquet_path},
    )


def build_doc_unresolved_recovery_artifact(
    *,
    recovery_mode: str,
    requests_parquet_path: Path | str,
    raw_parquet_path: Path | str,
    output_path: Path | str,
) -> RecoveryArtifact:
    requests_parquet_path = Path(requests_parquet_path)
    raw_parquet_path = Path(raw_parquet_path)
    output_path = Path(output_path)
    request_df = pl.read_parquet(requests_parquet_path)
    if raw_parquet_path.exists():
        raw_df = pl.read_parquet(raw_parquet_path)
        returned_doc_ids = set(raw_df.select("doc_id").drop_nulls().to_series(0).to_list())
    else:
        returned_doc_ids = set()
    unresolved_df = request_df.filter(
        pl.col("retrieval_eligible").fill_null(False)
        & ~pl.col("doc_id").is_in(sorted(returned_doc_ids))
    )
    write_parquet_atomic(unresolved_df, output_path)
    return RecoveryArtifact(
        recovery_mode=recovery_mode,
        output_path=output_path,
        row_count=int(unresolved_df.height),
        source_artifacts={
            "requests_parquet": requests_parquet_path,
            "raw_parquet": raw_parquet_path,
        },
    )


def write_recovery_manifest(
    *,
    manifest_path: Path | str,
    artifact: RecoveryArtifact,
) -> Path:
    manifest_path = Path(manifest_path)
    payload = {
        "manifest_role": "recovery_artifact",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **artifact.to_dict(),
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path

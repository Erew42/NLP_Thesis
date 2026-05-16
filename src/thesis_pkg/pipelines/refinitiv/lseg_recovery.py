from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

try:
    from thesis_native import _lm2011_rust
except Exception as exc:  # pragma: no cover - optional native extension
    _lm2011_rust = None
    _LSEG_RECOVERY_RUST_IMPORT_ERROR: str | None = f"{type(exc).__name__}: {exc}"
else:
    _LSEG_RECOVERY_RUST_IMPORT_ERROR = None

from thesis_refinitiv.lseg_client.api_common import write_parquet_atomic


_LSEG_RECOVERY_RUST_METRICS: dict[str, int] = {
    "doc_unresolved_mask_fast_success": 0,
    "doc_unresolved_mask_fast_failures": 0,
    "doc_unresolved_mask_fallbacks": 0,
}


def get_lseg_recovery_rust_accel_metrics() -> dict[str, int | str | bool | None]:
    metrics: dict[str, int | str | bool | None] = dict(_LSEG_RECOVERY_RUST_METRICS)
    metrics["rust_accel_available"] = _lm2011_rust is not None
    metrics["rust_accel_import_error"] = _LSEG_RECOVERY_RUST_IMPORT_ERROR
    return metrics


def reset_lseg_recovery_rust_accel_metrics() -> None:
    for key in _LSEG_RECOVERY_RUST_METRICS:
        _LSEG_RECOVERY_RUST_METRICS[key] = 0


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


def _optional_doc_id_text(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _doc_unresolved_mask_py(
    request_doc_ids: list[Any],
    retrieval_eligible_values: list[Any],
    returned_doc_ids: list[Any],
) -> list[bool]:
    returned = {
        returned_doc_id
        for returned_doc_id in (_optional_doc_id_text(value) for value in returned_doc_ids)
        if returned_doc_id is not None
    }
    mask: list[bool] = []
    for doc_id_value, retrieval_eligible in zip(
        request_doc_ids,
        retrieval_eligible_values,
        strict=True,
    ):
        doc_id = _optional_doc_id_text(doc_id_value)
        mask.append(bool(retrieval_eligible) and doc_id is not None and doc_id not in returned)
    return mask


def _doc_unresolved_mask(
    request_doc_ids: list[Any],
    retrieval_eligible_values: list[Any],
    returned_doc_ids: list[Any],
) -> list[bool]:
    if _lm2011_rust is not None:
        try:
            mask = _lm2011_rust.lseg_doc_unresolved_mask(
                [_optional_doc_id_text(value) for value in request_doc_ids],
                [None if value is None else bool(value) for value in retrieval_eligible_values],
                [_optional_doc_id_text(value) for value in returned_doc_ids],
            )
            _LSEG_RECOVERY_RUST_METRICS["doc_unresolved_mask_fast_success"] += 1
            return [bool(value) for value in mask]
        except Exception:
            _LSEG_RECOVERY_RUST_METRICS["doc_unresolved_mask_fast_failures"] += 1
    _LSEG_RECOVERY_RUST_METRICS["doc_unresolved_mask_fallbacks"] += 1
    return _doc_unresolved_mask_py(request_doc_ids, retrieval_eligible_values, returned_doc_ids)


def _filter_doc_unresolved_requests(
    request_df: pl.DataFrame,
    returned_doc_ids: list[Any],
) -> pl.DataFrame:
    mask = _doc_unresolved_mask(
        request_df.get_column("doc_id").to_list(),
        request_df.get_column("retrieval_eligible").to_list(),
        returned_doc_ids,
    )
    return request_df.filter(pl.Series("__doc_unresolved_mask", mask))


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
        returned_doc_ids = raw_df.select("doc_id").drop_nulls().to_series(0).to_list()
    else:
        returned_doc_ids = []
    unresolved_df = _filter_doc_unresolved_requests(request_df, returned_doc_ids)
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

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import polars as pl


MANIFEST_PATH_SEMANTICS_LEGACY = "absolute_legacy_v0"
MANIFEST_PATH_SEMANTICS_RELATIVE = "manifest_relative_v1"


def canonical_json_text(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_json_payload(payload: Any) -> Any:
    return json.loads(canonical_json_text(payload))


def json_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_text(payload).encode("utf-8")).hexdigest()


def stable_string_fingerprint(values: Iterable[str | None]) -> str:
    digest = hashlib.sha256()
    for value in sorted({str(value) for value in values if value is not None}):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def parquet_row_count(path: Path) -> int | None:
    if path.suffix.lower() != ".parquet":
        return None
    return int(pl.scan_parquet(path).select(pl.len()).collect().item())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    relative_path = str(resolved)
    if relative_to is not None:
        relative_path = normalize_contract_path(resolved, base_path=relative_to)
    return {
        "relative_path": relative_path,
        "file_size_bytes": stat.st_size,
        "modified_time_ns": stat.st_mtime_ns,
        "parquet_row_count": parquet_row_count(resolved),
    }


def semantic_file_fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "sha256": _file_sha256(resolved),
        "parquet_row_count": parquet_row_count(resolved),
    }


def parquet_doc_universe_fingerprint(path: Path, *, doc_id_col: str = "doc_id") -> dict[str, Any]:
    resolved = path.resolve()
    lf = pl.scan_parquet(resolved)
    schema = lf.collect_schema()
    if doc_id_col not in schema:
        raise ValueError(f"Parquet artifact {resolved} is missing required doc universe column: {doc_id_col}")
    doc_ids = (
        lf.select(pl.col(doc_id_col).cast(pl.Utf8, strict=False).drop_nulls().unique().sort())
        .collect()
        .get_column(doc_id_col)
        .to_list()
    )
    return {
        "doc_id_column": doc_id_col,
        "doc_count": len(doc_ids),
        "doc_id_fingerprint": stable_string_fingerprint(doc_ids),
    }


def make_semantic_reuse_guard(
    *,
    version: str,
    payload: dict[str, Any],
    fingerprints: dict[str, Any],
) -> dict[str, Any]:
    return {
        "version": version,
        "payload": canonical_json_payload(payload),
        "fingerprints": canonical_json_payload(fingerprints),
    }


def semantic_guard_mismatches(existing: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    mismatches: list[str] = []
    if existing.get("version") != expected.get("version"):
        mismatches.append("version")

    existing_payload = existing.get("payload") or {}
    expected_payload = expected.get("payload") or {}
    for key in sorted(set(existing_payload) | set(expected_payload)):
        if existing_payload.get(key) != expected_payload.get(key):
            mismatches.append(f"payload.{key}")

    existing_fingerprints = existing.get("fingerprints") or {}
    expected_fingerprints = expected.get("fingerprints") or {}
    for key in sorted(set(existing_fingerprints) | set(expected_fingerprints)):
        if existing_fingerprints.get(key) != expected_fingerprints.get(key):
            mismatches.append(f"fingerprints.{key}")

    return mismatches


def normalize_contract_path(path: Path, *, base_path: Path | None = None) -> str:
    resolved = path.resolve()
    if base_path is None:
        return str(resolved)
    base_resolved = base_path.resolve()
    if resolved.is_relative_to(base_resolved):
        return resolved.relative_to(base_resolved).as_posix()
    return str(resolved)


def relative_artifact_path(path: Path, *, base_path: Path) -> str | None:
    resolved = path.resolve()
    base_resolved = base_path.resolve()
    if resolved.is_relative_to(base_resolved):
        return resolved.relative_to(base_resolved).as_posix()
    return None


def write_manifest_path_value(path: Path, *, manifest_path: Path, path_semantics: str) -> str:
    resolved = path.resolve()
    if path_semantics == MANIFEST_PATH_SEMANTICS_RELATIVE:
        return os.path.relpath(resolved, manifest_path.parent.resolve()).replace("\\", "/")
    return str(resolved)


def resolve_manifest_path(
    raw_path: str | Path | None,
    *,
    manifest_path: Path,
    path_semantics: str | None,
) -> Path | None:
    if raw_path in (None, ""):
        return None
    value = Path(str(raw_path))
    semantics = path_semantics or MANIFEST_PATH_SEMANTICS_LEGACY
    if semantics == MANIFEST_PATH_SEMANTICS_RELATIVE and not value.is_absolute():
        return (manifest_path.parent / value).resolve()
    return value.resolve() if value.is_absolute() else value

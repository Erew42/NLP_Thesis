from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from thesis_assets.config.constants import ARTIFACT_FILENAMES
from thesis_assets.config.constants import RUN_FAMILY_SENTINEL_ARTIFACTS
from thesis_assets.config.constants import RUN_MANIFEST_FILENAMES
from thesis_assets.errors import SubmissionLockError
from thesis_pkg.benchmarking.manifest_contracts import json_sha256
from thesis_pkg.benchmarking.manifest_contracts import semantic_file_fingerprint


SUBMISSION_LOCK_SCHEMA_VERSION = 1
SUBMISSION_PATH_SEMANTICS = "relative_to_submission_root"
FINBERT_INFERRED_REVISION_DISCLOSURE_ID = "finbert_inferred_revision"
DEFAULT_FINBERT_MODEL_NAME = "yiyanghkust/finbert-tone"
DEFAULT_FINBERT_REVISION = "4921590d3c0c3832c0efea24c8381ce0bda7844b"


@dataclass(frozen=True)
class LockedRunRoot:
    run_family: str
    path: Path
    relative_path: str
    fingerprint: dict[str, Any]


@dataclass(frozen=True)
class LockedArtifactOverride:
    artifact_key: str
    path: Path
    relative_path: str
    reason: str
    fingerprint: dict[str, Any]
    run_family: str | None = None


@dataclass(frozen=True)
class SubmissionLock:
    path: Path
    submission_root: Path
    run_id: str | None
    run_roots: dict[str, LockedRunRoot]
    artifact_overrides: dict[str, LockedArtifactOverride]
    provenance_disclosures: tuple[dict[str, Any], ...]
    strict_policy: dict[str, Any]
    raw_payload: dict[str, Any]

    def disclosure_ids(self) -> set[str]:
        ids: set[str] = set()
        for disclosure in self.provenance_disclosures:
            raw_id = disclosure.get("id")
            if isinstance(raw_id, str) and raw_id:
                ids.add(raw_id)
        return ids

    def metadata_payload(self) -> dict[str, Any]:
        return {
            "path": _relative_or_absolute(self.path, self.submission_root),
            "submission_root": str(self.submission_root),
            "run_id": self.run_id,
            "schema_version": self.raw_payload.get("schema_version"),
            "path_semantics": self.raw_payload.get("path_semantics"),
            "run_roots": {
                run_family: {
                    "path": locked.relative_path,
                    "fingerprint": locked.fingerprint,
                }
                for run_family, locked in sorted(self.run_roots.items())
            },
            "artifact_overrides": {
                artifact_key: {
                    "path": locked.relative_path,
                    "run_family": locked.run_family,
                    "reason": locked.reason,
                    "fingerprint": locked.fingerprint,
                }
                for artifact_key, locked in sorted(self.artifact_overrides.items())
            },
            "provenance_disclosures": list(self.provenance_disclosures),
            "strict_policy": self.strict_policy,
        }


def load_submission_lock(path: Path, *, validate_fingerprints: bool = True) -> SubmissionLock:
    lock_path = Path(path).expanduser().resolve()
    if not lock_path.exists():
        raise SubmissionLockError(f"Submission lock does not exist: {lock_path}")
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SubmissionLockError("Submission lock must be a JSON object.")

    schema_version = payload.get("schema_version")
    if schema_version != SUBMISSION_LOCK_SCHEMA_VERSION:
        raise SubmissionLockError(
            f"Unsupported submission lock schema_version={schema_version!r}; "
            f"expected {SUBMISSION_LOCK_SCHEMA_VERSION}."
        )
    if payload.get("path_semantics") != SUBMISSION_PATH_SEMANTICS:
        raise SubmissionLockError(
            "Submission lock path_semantics must be "
            f"{SUBMISSION_PATH_SEMANTICS!r}."
        )

    submission_root = lock_path.parent.resolve()
    run_roots = _load_locked_run_roots(
        payload.get("run_roots"),
        submission_root=submission_root,
        validate_fingerprints=validate_fingerprints,
    )
    artifact_overrides = _load_locked_artifact_overrides(
        payload.get("artifact_overrides", {}),
        submission_root=submission_root,
        validate_fingerprints=validate_fingerprints,
    )
    disclosures = _normalize_disclosures(payload.get("provenance_disclosures", ()))
    input_fingerprints = payload.get("input_fingerprints", {})
    if not isinstance(input_fingerprints, dict):
        raise SubmissionLockError("input_fingerprints must be a JSON object when provided.")
    strict_policy = payload.get("strict_policy", {})
    if not isinstance(strict_policy, dict):
        raise SubmissionLockError("strict_policy must be a JSON object when provided.")

    lock = SubmissionLock(
        path=lock_path,
        submission_root=submission_root,
        run_id=payload.get("run_id") if isinstance(payload.get("run_id"), str) else None,
        run_roots=run_roots,
        artifact_overrides=artifact_overrides,
        provenance_disclosures=tuple(disclosures),
        strict_policy=strict_policy,
        raw_payload=payload,
    )
    _validate_critical_provenance(lock)
    return lock


def build_submission_lock_payload(
    *,
    submission_root: Path,
    run_id: str,
    run_roots: dict[str, Path],
    artifact_overrides: list[tuple[str, Path, str]],
    provenance_disclosures: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    root = Path(submission_root).expanduser().resolve()
    locked_roots = {
        run_family: {
            "path": _relative_path(path, root),
            "fingerprint": run_root_fingerprint(Path(path), run_family),
        }
        for run_family, path in sorted(run_roots.items())
        if path is not None
    }
    locked_overrides = {
        artifact_key: {
            "artifact_key": artifact_key,
            "path": _relative_path(path, root),
            "reason": reason,
            "fingerprint": path_fingerprint(path),
        }
        for artifact_key, path, reason in artifact_overrides
    }
    disclosures = list(provenance_disclosures or [])
    return {
        "schema_version": SUBMISSION_LOCK_SCHEMA_VERSION,
        "path_semantics": SUBMISSION_PATH_SEMANTICS,
        "run_id": run_id,
        "run_roots": locked_roots,
        "artifact_overrides": locked_overrides,
        "provenance_disclosures": disclosures,
        "strict_policy": {
            "disable_auto_run_discovery": True,
            "disable_artifact_alternate_filenames": True,
            "disable_table_vi_validation_fallback": True,
            "disable_portfolio_latest_rerun_discovery": True,
            "reject_paths_outside_submission_root": True,
        },
        "input_fingerprints": {},
    }


def write_submission_lock(path: Path, payload: dict[str, Any]) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output


def path_fingerprint(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if resolved.is_file():
        return {"kind": "file", **semantic_file_fingerprint(resolved)}
    if resolved.is_dir():
        entries: list[dict[str, Any]] = []
        for child in sorted(candidate for candidate in resolved.rglob("*") if candidate.is_file()):
            stat = child.stat()
            entries.append(
                {
                    "path": child.relative_to(resolved).as_posix(),
                    "size_bytes": stat.st_size,
                }
            )
        return {
            "kind": "directory",
            "file_count": len(entries),
            "entries_sha256": json_sha256(entries),
        }
    raise SubmissionLockError(f"Cannot fingerprint missing path: {resolved}")


def run_root_fingerprint(path: Path, run_family: str) -> dict[str, Any]:
    root = Path(path).resolve()
    if not root.exists() or not root.is_dir():
        raise SubmissionLockError(f"Run root does not exist or is not a directory: {root}")

    manifest_path = _find_manifest_path(root, run_family)
    if manifest_path is not None:
        return {
            "kind": "run_root",
            "authority": "native_manifest",
            "manifest_path": manifest_path.relative_to(root).as_posix(),
            "manifest": semantic_file_fingerprint(manifest_path),
        }

    sentinel_paths = _sentinel_paths(root, run_family)
    present = [path for path in sentinel_paths if path.exists()]
    if not present:
        raise SubmissionLockError(
            f"Locked {run_family} root has no native manifest and no sentinel artifacts: {root}"
        )
    return {
        "kind": "run_root",
        "authority": "submission_lock_sentinels",
        "sentinels": {
            path.relative_to(root).as_posix(): semantic_file_fingerprint(path)
            for path in present
        },
    }


def validate_submission_lock(path: Path) -> SubmissionLock:
    return load_submission_lock(path, validate_fingerprints=True)


def _load_locked_run_roots(
    raw_roots: object,
    *,
    submission_root: Path,
    validate_fingerprints: bool,
) -> dict[str, LockedRunRoot]:
    if not isinstance(raw_roots, dict) or not raw_roots:
        raise SubmissionLockError("run_roots must be a non-empty JSON object.")
    locked: dict[str, LockedRunRoot] = {}
    for run_family, raw_value in raw_roots.items():
        if run_family not in RUN_MANIFEST_FILENAMES:
            raise SubmissionLockError(f"Unknown run family in submission lock: {run_family!r}")
        raw_entry = raw_value if isinstance(raw_value, dict) else {"path": raw_value}
        raw_path = raw_entry.get("path")
        resolved = _resolve_locked_relative_path(raw_path, submission_root)
        if not resolved.exists() or not resolved.is_dir():
            raise SubmissionLockError(f"Locked {run_family} root is missing or not a directory: {resolved}")
        expected_fingerprint = raw_entry.get("fingerprint")
        if not isinstance(expected_fingerprint, dict):
            raise SubmissionLockError(f"Locked {run_family} root must include a fingerprint.")
        if validate_fingerprints:
            actual_fingerprint = run_root_fingerprint(resolved, run_family)
            if actual_fingerprint != expected_fingerprint:
                raise SubmissionLockError(f"Fingerprint mismatch for locked run root {run_family}: {resolved}")
        locked[run_family] = LockedRunRoot(
            run_family=run_family,
            path=resolved,
            relative_path=str(raw_path),
            fingerprint=expected_fingerprint,
        )
    return locked


def _load_locked_artifact_overrides(
    raw_overrides: object,
    *,
    submission_root: Path,
    validate_fingerprints: bool,
) -> dict[str, LockedArtifactOverride]:
    if raw_overrides in (None, ""):
        return {}
    if not isinstance(raw_overrides, dict):
        raise SubmissionLockError("artifact_overrides must be a JSON object.")
    locked: dict[str, LockedArtifactOverride] = {}
    for key, raw_entry in raw_overrides.items():
        if not isinstance(raw_entry, dict):
            raise SubmissionLockError(f"artifact_overrides[{key!r}] must be a JSON object.")
        artifact_key = raw_entry.get("artifact_key")
        if artifact_key != key:
            raise SubmissionLockError(
                f"artifact_overrides[{key!r}] must include matching artifact_key={key!r}."
            )
        if artifact_key not in ARTIFACT_FILENAMES:
            raise SubmissionLockError(f"Unknown artifact override key: {artifact_key!r}")
        raw_path = raw_entry.get("path")
        resolved = _resolve_locked_relative_path(raw_path, submission_root)
        if not resolved.exists():
            raise SubmissionLockError(f"Locked artifact override is missing: {resolved}")
        expected_fingerprint = raw_entry.get("fingerprint")
        if not isinstance(expected_fingerprint, dict):
            raise SubmissionLockError(f"Locked artifact override {artifact_key!r} must include a fingerprint.")
        if validate_fingerprints:
            actual_fingerprint = path_fingerprint(resolved)
            if actual_fingerprint != expected_fingerprint:
                raise SubmissionLockError(f"Fingerprint mismatch for locked artifact override {artifact_key}: {resolved}")
        reason = raw_entry.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise SubmissionLockError(f"Locked artifact override {artifact_key!r} must include a reason.")
        run_family = raw_entry.get("run_family")
        if run_family is not None and not isinstance(run_family, str):
            raise SubmissionLockError(f"Locked artifact override {artifact_key!r} has invalid run_family.")
        if run_family is not None and run_family not in RUN_MANIFEST_FILENAMES:
            raise SubmissionLockError(f"Locked artifact override {artifact_key!r} has unknown run_family {run_family!r}.")
        locked[artifact_key] = LockedArtifactOverride(
            artifact_key=artifact_key,
            path=resolved,
            relative_path=str(raw_path),
            reason=reason.strip(),
            fingerprint=expected_fingerprint,
            run_family=run_family,
        )
    return locked


def _normalize_disclosures(raw_disclosures: object) -> list[dict[str, Any]]:
    if raw_disclosures in (None, ""):
        return []
    if not isinstance(raw_disclosures, list):
        raise SubmissionLockError("provenance_disclosures must be a JSON array.")
    normalized: list[dict[str, Any]] = []
    for index, disclosure in enumerate(raw_disclosures):
        if not isinstance(disclosure, dict):
            raise SubmissionLockError(f"provenance_disclosures[{index}] must be a JSON object.")
        disclosure_id = disclosure.get("id")
        if not isinstance(disclosure_id, str) or not disclosure_id:
            raise SubmissionLockError(f"provenance_disclosures[{index}] must include a non-empty id.")
        normalized.append(dict(disclosure))
    return normalized


def _validate_critical_provenance(lock: SubmissionLock) -> None:
    finbert_root = lock.run_roots.get("finbert_run")
    if finbert_root is None:
        return
    if not _finbert_has_missing_revision_provenance(finbert_root.path):
        return
    if FINBERT_INFERRED_REVISION_DISCLOSURE_ID not in lock.disclosure_ids():
        raise SubmissionLockError(
            "Locked FinBERT run has missing model/tokenizer revision provenance; "
            f"add provenance_disclosures id={FINBERT_INFERRED_REVISION_DISCLOSURE_ID!r}."
        )


def _finbert_has_missing_revision_provenance(run_root: Path) -> bool:
    missing = False
    manifest_path = run_root / "run_manifest.json"
    if not manifest_path.exists():
        return True
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    authority = manifest.get("authority")
    if not isinstance(authority, dict):
        missing = True
    else:
        missing = not authority.get("model_revision") or not authority.get("tokenizer_revision")

    item_features_path = run_root / "item_features_long.parquet"
    if not item_features_path.exists():
        return True
    schema = pl.scan_parquet(item_features_path).collect_schema()
    if "model_version" not in schema:
        return True
    all_null = bool(
        pl.scan_parquet(item_features_path)
        .select(pl.col("model_version").is_null().all().alias("all_null"))
        .collect()
        .item()
    )
    return missing or all_null


def _find_manifest_path(root: Path, run_family: str) -> Path | None:
    for filename in RUN_MANIFEST_FILENAMES[run_family]:
        candidate = root / filename
        if candidate.exists():
            return candidate.resolve()
    return None


def _sentinel_paths(root: Path, run_family: str) -> tuple[Path, ...]:
    return tuple(
        root / ARTIFACT_FILENAMES[artifact_key]
        for artifact_key in RUN_FAMILY_SENTINEL_ARTIFACTS[run_family]
    )


def _resolve_locked_relative_path(raw_path: object, submission_root: Path) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise SubmissionLockError("Locked paths must be non-empty relative strings.")
    value = Path(raw_path)
    if value.is_absolute():
        raise SubmissionLockError(f"Locked paths must be relative to submission_root, got: {raw_path}")
    resolved = (submission_root / value).resolve()
    if not resolved.is_relative_to(submission_root):
        raise SubmissionLockError(f"Locked path escapes submission_root: {raw_path}")
    return resolved


def _relative_path(path: Path, submission_root: Path) -> str:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_relative_to(submission_root):
        raise SubmissionLockError(f"Path must be inside submission_root: {resolved}")
    return resolved.relative_to(submission_root).as_posix()


def _relative_or_absolute(path: Path, base: Path) -> str:
    resolved = Path(path).resolve()
    return resolved.relative_to(base).as_posix() if resolved.is_relative_to(base) else str(resolved)

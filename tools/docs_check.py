from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


AUTO_NAV_BEGIN = "# BEGIN AUTO-REFERENCE-NAV"
AUTO_NAV_END = "# END AUTO-REFERENCE-NAV"
NOT_IN_NAV_HEADER = 'pages exist in the docs directory, but are not included in the "nav" configuration'
BEHAVIOR_EVIDENCE_REL_PATH = "reference/behavior_evidence.md"
PUBLISH_STATUS_PUBLISHED = "published"
PUBLISH_STATUS_TRUNCATED = "truncated"
PUBLISH_STATUS_OMITTED = "omitted_size_limit"
PUBLISH_NOTE_MISSING_ALLOWED = "MISSING_CANONICAL_ALLOWED"


def canonicalize_json_text(text: str) -> str:
    """Return a deterministic JSON string representation."""
    return json.dumps(json.loads(text), sort_keys=True, separators=(",", ":"))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _module_to_doc_relpath(module_name: str, package_root: str) -> str:
    parts = module_name.split(".")
    if parts and parts[0] == package_root:
        parts = parts[1:]
    return str(Path("reference").joinpath(*parts).with_suffix(".md")).replace("\\", "/")


def _normalize_nav_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_normalize_nav_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_nav_value(item) for key, item in value.items()}
    if isinstance(value, str):
        return value.replace("\\", "/")
    return value


def _collect_markdown_paths(value: Any) -> set[str]:
    paths: set[str] = set()
    if isinstance(value, list):
        for item in value:
            paths.update(_collect_markdown_paths(item))
    elif isinstance(value, dict):
        for item in value.values():
            paths.update(_collect_markdown_paths(item))
    elif isinstance(value, str) and value.replace("\\", "/").endswith(".md"):
        paths.add(value.replace("\\", "/"))
    return paths


def _load_yaml_file(path: Path) -> Any:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if loaded is not None else []


def _extract_managed_nav_block_text(mkdocs_config: Path) -> str:
    lines = mkdocs_config.read_text(encoding="utf-8").splitlines()
    begin_idx = None
    end_idx = None
    for idx, line in enumerate(lines):
        if AUTO_NAV_BEGIN in line:
            begin_idx = idx
        if AUTO_NAV_END in line:
            end_idx = idx
    if begin_idx is None or end_idx is None or begin_idx >= end_idx:
        raise ValueError(
            f"Managed nav markers not found or invalid in {mkdocs_config}: expected {AUTO_NAV_BEGIN} .. {AUTO_NAV_END}"
        )
    return "\n".join(lines[begin_idx + 1 : end_idx]).strip()


def _load_managed_nav_block(mkdocs_config: Path) -> Any:
    block_text = _extract_managed_nav_block_text(mkdocs_config)
    if not block_text:
        return []
    loaded = yaml.safe_load(block_text)
    return loaded if loaded is not None else []


def extract_not_in_nav_pages(log_text: str) -> list[str]:
    """Extract doc paths from MkDocs 'not included in nav' warning blocks."""
    pages: list[str] = []
    lines = log_text.splitlines()
    idx = 0

    while idx < len(lines):
        if NOT_IN_NAV_HEADER not in lines[idx]:
            idx += 1
            continue
        idx += 1
        while idx < len(lines):
            stripped = lines[idx].strip()
            if not stripped:
                idx += 1
                continue
            bullet_match = re.match(r"^- (.+)$", stripped)
            if bullet_match:
                pages.append(bullet_match.group(1).strip())
                idx += 1
                continue
            if re.match(r"^[A-Z]+ - ", stripped):
                break
            break
    return pages


def reference_only_missing_pages(pages: list[str]) -> list[str]:
    offenders: list[str] = []
    for page in pages:
        normalized = page.replace("\\", "/").lstrip("./")
        if normalized.startswith("docs/"):
            normalized = normalized[len("docs/") :]
        if normalized.startswith("reference/"):
            offenders.append(normalized)
    return sorted(set(offenders))


def _build_mkdocs_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(base_env if base_env is not None else os.environ)
    if os.name == "nt":
        env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_mkdocs_build(*, repo_root: Path, mkdocs_config: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "mkdocs", "build", "-f", str(mkdocs_config)],
        cwd=repo_root,
        text=True,
        capture_output=True,
        env=_build_mkdocs_env(),
        check=False,
    )


def _expected_module_pages(inventory: dict[str, Any], package_root: str) -> dict[str, str]:
    return {
        module_name: _module_to_doc_relpath(module_name=module_name, package_root=package_root)
        for module_name in sorted(inventory.keys())
    }


def _resolve_outward_modules(outward_api: dict[str, Any]) -> set[str]:
    modules = set(outward_api.get("all_outward_modules", []))
    module_map = outward_api.get("modules", {})
    if isinstance(module_map, dict):
        modules.update(str(name) for name in module_map.keys())
    return modules


def _latest_python_mtime(src_dir: Path) -> float | None:
    if not src_dir.exists():
        return None
    mtimes = [path.stat().st_mtime for path in src_dir.rglob("*.py") if path.is_file()]
    if not mtimes:
        return None
    return max(mtimes)


def _metadata_stale_files(*, latest_source_mtime: float, metadata_paths: list[Path]) -> list[Path]:
    stale: list[Path] = []
    for path in metadata_paths:
        if not path.exists():
            continue
        if path.stat().st_mtime < latest_source_mtime:
            stale.append(path)
    return stale


def _resolve_artifact_path(repo_root: Path, artifact_path: str) -> Path:
    path = Path(artifact_path)
    return path if path.is_absolute() else (repo_root / path)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _extract_markdown_links(markdown_text: str) -> list[str]:
    links: list[str] = []
    for raw in re.findall(r"\[[^\]]*\]\(([^)]+)\)", markdown_text):
        target = raw.strip().split()[0].strip()
        if target.startswith("<") and target.endswith(">"):
            target = target[1:-1]
        if target:
            links.append(target)
    return links


def _validate_behavior_page_links(*, behavior_page: Path, docs_dir: Path) -> list[str]:
    """Validate behavior page links resolve to published docs assets.

    WHY (TODO by Erik): trace links must remain publishable in MkDocs output,
    so links to docs_metadata/ are rejected even if local files exist.
    """
    if not behavior_page.exists():
        return []
    errors: list[str] = []
    markdown_text = behavior_page.read_text(encoding="utf-8")
    links = _extract_markdown_links(markdown_text)
    for link in links:
        normalized = link.replace("\\", "/")
        if normalized.startswith(("#", "http://", "https://", "mailto:", "tel:")):
            continue
        if "docs_metadata/" in normalized:
            errors.append(
                f"Behavior evidence link targets docs_metadata and will break in site output: {link}"
            )
            continue

        target_path = (behavior_page.parent / link).resolve()
        if not _is_within(target_path, docs_dir):
            errors.append(
                f"Behavior evidence link points outside docs directory: {link}"
            )
            continue
        if not target_path.exists():
            errors.append(
                f"Behavior evidence link target does not exist: {link}"
            )
    return errors


def _load_trace_artifact_rows(manifest_payload: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    raw = manifest_payload.get("artifacts", [])
    if isinstance(raw, dict):
        for value in raw.values():
            rows.append(
                {
                    "canonical_path": str(value),
                    "published_path": "",
                    "published_doc_path": "",
                    "preview_path": "",
                    "preview_doc_path": "",
                    "publish_status": "",
                    "publish_note": "",
                }
            )
        return rows
    if not isinstance(raw, list):
        return rows

    for row in raw:
        if isinstance(row, str):
            rows.append(
                {
                    "canonical_path": row,
                    "published_path": "",
                    "published_doc_path": "",
                    "preview_path": "",
                    "preview_doc_path": "",
                    "publish_status": "",
                    "publish_note": "",
                }
            )
            continue
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "canonical_path": str(row.get("canonical_path") or row.get("path") or ""),
                "published_path": str(row.get("published_path") or ""),
                "published_doc_path": str(row.get("published_doc_path") or ""),
                "preview_path": str(row.get("preview_path") or ""),
                "preview_doc_path": str(row.get("preview_doc_path") or ""),
                "publish_status": str(row.get("publish_status") or ""),
                "publish_note": str(row.get("publish_note") or ""),
            }
        )
    return rows


def _resolve_doc_or_repo_path(
    *,
    repo_root: Path,
    docs_dir: Path,
    repo_rel: str,
    docs_rel: str,
) -> Path | None:
    if repo_rel:
        return _resolve_artifact_path(repo_root, repo_rel)
    if docs_rel:
        return (docs_dir / docs_rel).resolve()
    return None


def check_docs(
    *,
    repo_root: Path,
    inventory_path: Path,
    outward_api_path: Path,
    nav_fragment_path: Path,
    mkdocs_config_path: Path,
    docs_dir: Path,
    package_root: str,
    run_build: bool,
    src_dir: Path,
    import_evidence_path: Path,
    trace_dir: Path,
    require_trace: bool,
) -> list[str]:
    """Validate generated docs metadata, nav integrity, and optional trace outputs.

    WHY (TODO by Erik): docs generation is automation-heavy, so checks fail fast
    when generated contracts drift (nav fragment, trace publish semantics, staleness).
    """
    errors: list[str] = []

    required_files = [
        inventory_path,
        outward_api_path,
        import_evidence_path,
        nav_fragment_path,
        mkdocs_config_path,
    ]
    for path in required_files:
        if not path.exists():
            errors.append(f"Missing required file: {path}")
    if errors:
        return errors

    latest_source_mtime = _latest_python_mtime(src_dir)
    if latest_source_mtime is not None:
        stale_files = _metadata_stale_files(
            latest_source_mtime=latest_source_mtime,
            metadata_paths=[inventory_path, outward_api_path, import_evidence_path, nav_fragment_path],
        )
        if stale_files:
            stale_names = ", ".join(str(path) for path in stale_files)
            errors.append(
                "Docs metadata appears stale versus source files in "
                f"{src_dir}: regenerate extractor/scaffold outputs ({stale_names})."
            )

    inventory_raw = _load_json(inventory_path)
    outward_api_raw = _load_json(outward_api_path)
    inventory = inventory_raw if isinstance(inventory_raw, dict) else {}
    outward_api = outward_api_raw if isinstance(outward_api_raw, dict) else {}

    expected_pages = _expected_module_pages(inventory=inventory, package_root=package_root)
    missing_pages = [
        rel_path
        for rel_path in expected_pages.values()
        if not (docs_dir / Path(rel_path)).exists()
    ]
    if missing_pages:
        errors.append(
            "Missing generated reference pages for inventory modules: "
            + ", ".join(missing_pages[:20])
            + (" ..." if len(missing_pages) > 20 else "")
        )

    missing_outward = [
        module_name
        for module_name in sorted(_resolve_outward_modules(outward_api))
        if not (docs_dir / Path(_module_to_doc_relpath(module_name, package_root))).exists()
    ]
    if missing_outward:
        errors.append(
            "Missing generated reference pages for outward-facing modules: "
            + ", ".join(missing_outward[:20])
            + (" ..." if len(missing_outward) > 20 else "")
        )

    reference_index = docs_dir / "reference" / "index.md"
    if not reference_index.exists():
        errors.append(f"Missing reference index page: {reference_index}")
    behavior_page = docs_dir / Path(BEHAVIOR_EVIDENCE_REL_PATH)
    if not behavior_page.exists():
        errors.append(f"Missing behavior evidence page: {behavior_page}")
    else:
        errors.extend(_validate_behavior_page_links(behavior_page=behavior_page, docs_dir=docs_dir.resolve()))

    fragment_nav = _normalize_nav_value(_load_yaml_file(nav_fragment_path))
    managed_nav = _normalize_nav_value(_load_managed_nav_block(mkdocs_config_path))
    if fragment_nav != managed_nav:
        errors.append(
            f"Managed nav block in {mkdocs_config_path} does not match generated nav fragment {nav_fragment_path}."
        )

    fragment_paths = _collect_markdown_paths(fragment_nav)
    expected_nav_paths = set(expected_pages.values())
    expected_nav_paths.add("reference/index.md")
    expected_nav_paths.add(BEHAVIOR_EVIDENCE_REL_PATH)
    missing_nav_paths = sorted(expected_nav_paths - fragment_paths)
    if missing_nav_paths:
        errors.append(
            "Generated nav fragment missing expected reference paths: "
            + ", ".join(missing_nav_paths[:20])
            + (" ..." if len(missing_nav_paths) > 20 else "")
        )

    if require_trace:
        run_manifest_path = trace_dir / "run_manifest.json"
        if not run_manifest_path.exists():
            errors.append(f"Missing trace manifest: {run_manifest_path}")
        else:
            payload_raw = _load_json(run_manifest_path)
            payload = payload_raw if isinstance(payload_raw, dict) else {}
            trace_generation = payload.get("trace_generation", {})
            allow_missing_canonical = bool(
                trace_generation.get("allow_missing_canonical", False)
                if isinstance(trace_generation, dict)
                else False
            )
            artifact_rows = _load_trace_artifact_rows(payload)
            if not artifact_rows:
                errors.append(
                    f"Trace manifest does not contain artifact references: {run_manifest_path}"
                )
            else:
                missing_trace = []
                missing_published = []
                invalid_published = []
                invalid_status = []
                missing_preview = []
                invalid_preview = []
                for row in artifact_rows:
                    canonical = row.get("canonical_path", "")
                    published = row.get("published_path", "")
                    published_doc = row.get("published_doc_path", "")
                    preview = row.get("preview_path", "")
                    preview_doc = row.get("preview_doc_path", "")
                    publish_status = row.get("publish_status", "")
                    publish_note = row.get("publish_note", "")

                    canonical_abs = _resolve_artifact_path(repo_root, canonical) if canonical else None
                    if canonical_abs is not None and not canonical_abs.exists():
                        if allow_missing_canonical and publish_note == PUBLISH_NOTE_MISSING_ALLOWED:
                            pass
                        else:
                            missing_trace.append(canonical)

                    if publish_status not in {
                        PUBLISH_STATUS_PUBLISHED,
                        PUBLISH_STATUS_TRUNCATED,
                        PUBLISH_STATUS_OMITTED,
                    }:
                        invalid_status.append(str(row))
                        continue

                    published_abs = _resolve_doc_or_repo_path(
                        repo_root=repo_root,
                        docs_dir=docs_dir,
                        repo_rel=published,
                        docs_rel=published_doc,
                    )
                    if publish_status in {PUBLISH_STATUS_PUBLISHED, PUBLISH_STATUS_TRUNCATED}:
                        if published_abs is None:
                            invalid_published.append(str(row))
                        else:
                            if not _is_within(published_abs, docs_dir):
                                invalid_published.append(published_abs.as_posix())
                            elif not published_abs.exists():
                                missing_published.append(published_abs.as_posix())

                    preview_abs = _resolve_doc_or_repo_path(
                        repo_root=repo_root,
                        docs_dir=docs_dir,
                        repo_rel=preview,
                        docs_rel=preview_doc,
                    )
                    if preview_abs is not None:
                        if not _is_within(preview_abs, docs_dir):
                            invalid_preview.append(preview_abs.as_posix())
                        elif not preview_abs.exists():
                            missing_preview.append(preview_abs.as_posix())

                if missing_trace:
                    errors.append(
                        "Trace manifest references missing artifacts: "
                        + ", ".join(missing_trace[:20])
                        + (" ..." if len(missing_trace) > 20 else "")
                    )
                if missing_published:
                    errors.append(
                        "Trace manifest references missing published artifacts under docs/: "
                        + ", ".join(missing_published[:20])
                        + (" ..." if len(missing_published) > 20 else "")
                    )
                if invalid_published:
                    errors.append(
                        "Trace manifest contains invalid published artifact references: "
                        + ", ".join(invalid_published[:20])
                        + (" ..." if len(invalid_published) > 20 else "")
                    )
                if invalid_status:
                    errors.append(
                        "Trace manifest contains invalid publish status values: "
                        + ", ".join(invalid_status[:20])
                        + (" ..." if len(invalid_status) > 20 else "")
                    )
                if missing_preview:
                    errors.append(
                        "Trace manifest references missing preview artifacts under docs/: "
                        + ", ".join(missing_preview[:20])
                        + (" ..." if len(missing_preview) > 20 else "")
                    )
                if invalid_preview:
                    errors.append(
                        "Trace manifest contains invalid preview artifact references: "
                        + ", ".join(invalid_preview[:20])
                        + (" ..." if len(invalid_preview) > 20 else "")
                    )

    if run_build:
        build_result = run_mkdocs_build(repo_root=repo_root, mkdocs_config=mkdocs_config_path)
        build_log = f"{build_result.stdout}\n{build_result.stderr}"
        missing_nav_pages = extract_not_in_nav_pages(build_log)
        reference_missing = reference_only_missing_pages(missing_nav_pages)
        if reference_missing:
            errors.append(
                "MkDocs reported generated reference pages not included in nav: "
                + ", ".join(reference_missing)
            )
        if build_result.returncode != 0:
            errors.append(
                f"MkDocs build failed with exit code {build_result.returncode}. "
                "See build output for details."
            )

    return errors


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check docs metadata/nav completeness and mkdocs build health.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--src-dir", type=Path, default=Path("src/thesis_pkg"))
    parser.add_argument("--inventory", type=Path, default=Path("docs_metadata/inventory.json"))
    parser.add_argument("--outward-api", type=Path, default=Path("docs_metadata/outward_api.json"))
    parser.add_argument("--import-evidence", type=Path, default=Path("docs_metadata/import_evidence.json"))
    parser.add_argument("--nav-fragment", type=Path, default=Path("docs_metadata/reference_nav.yml"))
    parser.add_argument("--mkdocs-config", type=Path, default=Path("mkdocs.yml"))
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--trace-dir", type=Path, default=Path("docs_metadata/behavior"))
    parser.add_argument("--package-root", type=str, default="thesis_pkg")
    parser.add_argument("--require-trace", action="store_true")
    parser.add_argument("--run-build", action="store_true", default=True)
    parser.add_argument("--skip-build", action="store_true")
    return parser


def _resolve_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path)


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    run_build = args.run_build and not args.skip_build

    errors = check_docs(
        repo_root=repo_root,
        src_dir=_resolve_path(repo_root, args.src_dir),
        inventory_path=_resolve_path(repo_root, args.inventory),
        outward_api_path=_resolve_path(repo_root, args.outward_api),
        import_evidence_path=_resolve_path(repo_root, args.import_evidence),
        nav_fragment_path=_resolve_path(repo_root, args.nav_fragment),
        mkdocs_config_path=_resolve_path(repo_root, args.mkdocs_config),
        docs_dir=_resolve_path(repo_root, args.docs_dir),
        trace_dir=_resolve_path(repo_root, args.trace_dir),
        package_root=args.package_root,
        require_trace=bool(args.require_trace),
        run_build=run_build,
    )

    if errors:
        print("Docs checks failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Docs checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

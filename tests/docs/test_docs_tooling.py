from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import docs_check, docs_pipeline


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _module_page_relpath(module_name: str, package_root: str = "thesis_pkg") -> str:
    return docs_check._module_to_doc_relpath(module_name=module_name, package_root=package_root)


def _build_sample_layout(tmp_path: Path) -> dict[str, Path]:
    repo_root = tmp_path
    docs_dir = repo_root / "docs"
    metadata_dir = repo_root / "docs_metadata"
    trace_dir = metadata_dir / "behavior"
    src_dir = repo_root / "src" / "thesis_pkg"
    docs_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "__init__.py").write_text("# sample\n", encoding="utf-8")
    (src_dir / "pipeline.py").write_text("def run() -> None:\n    return None\n", encoding="utf-8")

    modules = [
        "thesis_pkg.__init__",
        "thesis_pkg.pipeline",
        "thesis_pkg.core.sec.patterns",
    ]

    inventory = {
        module: {
            "classes": [],
            "functions": [],
            "constants": [],
        }
        for module in modules
    }
    outward_api = {
        "all_outward_modules": ["thesis_pkg.pipeline"],
        "modules": {"thesis_pkg.pipeline": {"outward_reasons": ["imported_by_module"]}},
    }
    import_evidence = {"modules": {module: {"importer_count": 0} for module in modules}}

    _write_json(metadata_dir / "inventory.json", inventory)
    _write_json(metadata_dir / "outward_api.json", outward_api)
    _write_json(metadata_dir / "import_evidence.json", import_evidence)

    for module_name in modules:
        rel_path = _module_page_relpath(module_name)
        page_path = docs_dir / Path(rel_path)
        page_path.parent.mkdir(parents=True, exist_ok=True)
        page_path.write_text(f"# {module_name}\n", encoding="utf-8")

    (docs_dir / "reference" / "index.md").write_text("# API Reference\n", encoding="utf-8")
    (docs_dir / "reference" / "behavior_evidence.md").write_text("# Behavior Evidence\n", encoding="utf-8")

    nav_fragment = """- Reference:
  - Overview: reference/index.md
  - Behavior Evidence: reference/behavior_evidence.md
  - thesis_pkg: reference/__init__.md
  - thesis_pkg.pipeline: reference/pipeline.md
  - thesis_pkg.core.sec.patterns: reference/core/sec/patterns.md
"""
    (metadata_dir / "reference_nav.yml").write_text(nav_fragment, encoding="utf-8")

    mkdocs_text = """site_name: Test Docs
nav:
  - Home: index.md
  # BEGIN AUTO-REFERENCE-NAV
  - Reference:
      - Overview: reference/index.md
      - Behavior Evidence: reference/behavior_evidence.md
      - thesis_pkg: reference/__init__.md
      - thesis_pkg.pipeline: reference/pipeline.md
      - thesis_pkg.core.sec.patterns: reference/core/sec/patterns.md
  # END AUTO-REFERENCE-NAV
"""
    (repo_root / "mkdocs.yml").write_text(mkdocs_text, encoding="utf-8")

    return {
        "repo_root": repo_root,
        "docs_dir": docs_dir,
        "metadata_dir": metadata_dir,
        "trace_dir": trace_dir,
        "src_dir": src_dir,
        "mkdocs": repo_root / "mkdocs.yml",
    }


def test_nav_structure_check_passes_with_formatting_differences(tmp_path: Path) -> None:
    layout = _build_sample_layout(tmp_path)

    errors = docs_check.check_docs(
        repo_root=layout["repo_root"],
        src_dir=layout["src_dir"],
        inventory_path=layout["metadata_dir"] / "inventory.json",
        outward_api_path=layout["metadata_dir"] / "outward_api.json",
        import_evidence_path=layout["metadata_dir"] / "import_evidence.json",
        nav_fragment_path=layout["metadata_dir"] / "reference_nav.yml",
        mkdocs_config_path=layout["mkdocs"],
        docs_dir=layout["docs_dir"],
        trace_dir=layout["trace_dir"],
        package_root="thesis_pkg",
        run_build=False,
        require_trace=False,
    )

    assert errors == []


def test_nav_structure_check_fails_on_mismatch(tmp_path: Path) -> None:
    layout = _build_sample_layout(tmp_path)
    mkdocs_mismatch = """site_name: Test Docs
nav:
  - Home: index.md
  # BEGIN AUTO-REFERENCE-NAV
  - Reference:
    - Overview: reference/index.md
    - thesis_pkg: reference/__init__.md
    - thesis_pkg.pipeline: reference/pipeline.md
  # END AUTO-REFERENCE-NAV
"""
    layout["mkdocs"].write_text(mkdocs_mismatch, encoding="utf-8")

    errors = docs_check.check_docs(
        repo_root=layout["repo_root"],
        src_dir=layout["src_dir"],
        inventory_path=layout["metadata_dir"] / "inventory.json",
        outward_api_path=layout["metadata_dir"] / "outward_api.json",
        import_evidence_path=layout["metadata_dir"] / "import_evidence.json",
        nav_fragment_path=layout["metadata_dir"] / "reference_nav.yml",
        mkdocs_config_path=layout["mkdocs"],
        docs_dir=layout["docs_dir"],
        trace_dir=layout["trace_dir"],
        package_root="thesis_pkg",
        run_build=False,
        require_trace=False,
    )

    assert any("does not match generated nav fragment" in message for message in errors)


def test_nav_fragment_requires_behavior_evidence_entry(tmp_path: Path) -> None:
    layout = _build_sample_layout(tmp_path)
    nav_fragment_without_behavior = """- Reference:
  - Overview: reference/index.md
  - thesis_pkg: reference/__init__.md
  - thesis_pkg.pipeline: reference/pipeline.md
  - thesis_pkg.core.sec.patterns: reference/core/sec/patterns.md
"""
    (layout["metadata_dir"] / "reference_nav.yml").write_text(nav_fragment_without_behavior, encoding="utf-8")

    errors = docs_check.check_docs(
        repo_root=layout["repo_root"],
        src_dir=layout["src_dir"],
        inventory_path=layout["metadata_dir"] / "inventory.json",
        outward_api_path=layout["metadata_dir"] / "outward_api.json",
        import_evidence_path=layout["metadata_dir"] / "import_evidence.json",
        nav_fragment_path=layout["metadata_dir"] / "reference_nav.yml",
        mkdocs_config_path=layout["mkdocs"],
        docs_dir=layout["docs_dir"],
        trace_dir=layout["trace_dir"],
        package_root="thesis_pkg",
        run_build=False,
        require_trace=False,
    )

    assert any("reference/behavior_evidence.md" in message for message in errors)


def test_freshness_gate_flags_stale_metadata(tmp_path: Path) -> None:
    layout = _build_sample_layout(tmp_path)
    now = time.time()
    stale = now - 200
    fresh = now + 200

    metadata_files = [
        layout["metadata_dir"] / "inventory.json",
        layout["metadata_dir"] / "outward_api.json",
        layout["metadata_dir"] / "import_evidence.json",
        layout["metadata_dir"] / "reference_nav.yml",
    ]
    for path in metadata_files:
        os.utime(path, (stale, stale))

    source_file = layout["src_dir"] / "pipeline.py"
    os.utime(source_file, (fresh, fresh))

    errors = docs_check.check_docs(
        repo_root=layout["repo_root"],
        src_dir=layout["src_dir"],
        inventory_path=layout["metadata_dir"] / "inventory.json",
        outward_api_path=layout["metadata_dir"] / "outward_api.json",
        import_evidence_path=layout["metadata_dir"] / "import_evidence.json",
        nav_fragment_path=layout["metadata_dir"] / "reference_nav.yml",
        mkdocs_config_path=layout["mkdocs"],
        docs_dir=layout["docs_dir"],
        trace_dir=layout["trace_dir"],
        package_root="thesis_pkg",
        run_build=False,
        require_trace=False,
    )

    assert any("Docs metadata appears stale" in message for message in errors)


def test_require_trace_fails_without_manifest(tmp_path: Path) -> None:
    layout = _build_sample_layout(tmp_path)
    errors = docs_check.check_docs(
        repo_root=layout["repo_root"],
        src_dir=layout["src_dir"],
        inventory_path=layout["metadata_dir"] / "inventory.json",
        outward_api_path=layout["metadata_dir"] / "outward_api.json",
        import_evidence_path=layout["metadata_dir"] / "import_evidence.json",
        nav_fragment_path=layout["metadata_dir"] / "reference_nav.yml",
        mkdocs_config_path=layout["mkdocs"],
        docs_dir=layout["docs_dir"],
        trace_dir=layout["trace_dir"],
        package_root="thesis_pkg",
        run_build=False,
        require_trace=True,
    )

    assert any("Missing trace manifest" in message for message in errors)


def test_require_trace_passes_with_manifest_and_artifacts(tmp_path: Path) -> None:
    layout = _build_sample_layout(tmp_path)
    trace_artifact = layout["trace_dir"] / "artifact_manifest.csv"
    trace_artifact.write_text("k,v\n", encoding="utf-8")
    _write_json(
        layout["trace_dir"] / "run_manifest.json",
        {
            "artifacts": [
                {
                    "artifact_key": "artifact_manifest_csv",
                    "path": str(trace_artifact.relative_to(layout["repo_root"])).replace("\\", "/"),
                }
            ]
        },
    )

    errors = docs_check.check_docs(
        repo_root=layout["repo_root"],
        src_dir=layout["src_dir"],
        inventory_path=layout["metadata_dir"] / "inventory.json",
        outward_api_path=layout["metadata_dir"] / "outward_api.json",
        import_evidence_path=layout["metadata_dir"] / "import_evidence.json",
        nav_fragment_path=layout["metadata_dir"] / "reference_nav.yml",
        mkdocs_config_path=layout["mkdocs"],
        docs_dir=layout["docs_dir"],
        trace_dir=layout["trace_dir"],
        package_root="thesis_pkg",
        run_build=False,
        require_trace=True,
    )

    assert errors == []


def test_reference_only_warning_gate_mixed_entries() -> None:
    log_text = """
WARNING -  The following pages exist in the docs directory, but are not included in the "nav" configuration:
  - docstring_audit_report.md
  - other/legacy.md
  - reference/core/sec/patterns.md
"""
    pages = docs_check.extract_not_in_nav_pages(log_text)
    offenders = docs_check.reference_only_missing_pages(pages)
    assert offenders == ["reference/core/sec/patterns.md"]


def test_reference_only_warning_gate_ignores_non_reference() -> None:
    log_text = """
WARNING -  The following pages exist in the docs directory, but are not included in the "nav" configuration:
  - docstring_audit_report.md
  - other/legacy.md
"""
    pages = docs_check.extract_not_in_nav_pages(log_text)
    offenders = docs_check.reference_only_missing_pages(pages)
    assert offenders == []


def test_pipeline_build_env_sets_utf8_on_windows() -> None:
    env = docs_pipeline.build_subprocess_env(
        base_env={"PATH": "X"},
        enforce_mkdocs_utf8=True,
        os_name="nt",
    )
    assert env["PYTHONIOENCODING"] == "utf-8"


def test_json_canonicalization_is_deterministic() -> None:
    first = '{\n  "b": 2,\n  "a": {"y": 2, "x": 1}\n}'
    second = '{"a":{"x":1,"y":2},"b":2}'
    first_canon = docs_check.canonicalize_json_text(first)
    second_canon = docs_check.canonicalize_json_text(second)
    assert first_canon == second_canon

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scaffold_docs


def _build_relpaths(modules: list[str]) -> dict[str, str]:
    return {
        module_name: scaffold_docs._module_to_doc_relpath(module_name=module_name, package_root="thesis_pkg")
        for module_name in modules
    }


def test_build_auto_nav_lines_nested_structure() -> None:
    modules = [
        "thesis_pkg.__init__",
        "thesis_pkg.core.__init__",
        "thesis_pkg.core.sec.__init__",
        "thesis_pkg.core.sec.patterns",
        "thesis_pkg.core.sec.extraction",
        "thesis_pkg.pipeline",
        "thesis_pkg.settings",
    ]
    nav_lines = scaffold_docs._build_auto_nav_lines(
        modules=modules,
        module_to_relpath=_build_relpaths(modules),
        package_root="thesis_pkg",
    )
    assert nav_lines == [
        "  - Reference:",
        "    - Overview: reference/index.md",
        "    - Behavior Evidence: reference/behavior_evidence.md",
        "    - thesis_pkg: reference/__init__.md",
        "    - core:",
        "      - Overview: reference/core/__init__.md",
        "      - sec:",
        "        - Overview: reference/core/sec/__init__.md",
        "        - extraction: reference/core/sec/extraction.md",
        "        - patterns: reference/core/sec/patterns.md",
        "    - pipeline: reference/pipeline.md",
        "    - settings: reference/settings.md",
    ]


def test_build_auto_nav_lines_package_overview_label() -> None:
    modules = [
        "thesis_pkg.__init__",
        "thesis_pkg.core.__init__",
        "thesis_pkg.core.sec.__init__",
        "thesis_pkg.core.sec.patterns",
    ]
    nav_lines = scaffold_docs._build_auto_nav_lines(
        modules=modules,
        module_to_relpath=_build_relpaths(modules),
        package_root="thesis_pkg",
    )
    assert "      - Overview: reference/core/__init__.md" in nav_lines
    assert "        - Overview: reference/core/sec/__init__.md" in nav_lines


def test_build_auto_nav_lines_no_init_literal_labels() -> None:
    modules = [
        "thesis_pkg.__init__",
        "thesis_pkg.core.__init__",
        "thesis_pkg.core.sec.__init__",
        "thesis_pkg.core.sec.patterns",
    ]
    nav_lines = scaffold_docs._build_auto_nav_lines(
        modules=modules,
        module_to_relpath=_build_relpaths(modules),
        package_root="thesis_pkg",
    )
    assert all(not re.match(r"^\s*-\s*__init__:", line) for line in nav_lines)


def test_build_auto_nav_lines_deterministic_order() -> None:
    modules = [
        "thesis_pkg.pipeline",
        "thesis_pkg.__init__",
        "thesis_pkg.core.sec.patterns",
        "thesis_pkg.core.__init__",
        "thesis_pkg.core.sec.__init__",
        "thesis_pkg.core.sec.extraction",
    ]
    module_to_relpath = _build_relpaths(modules)
    nav_lines_a = scaffold_docs._build_auto_nav_lines(
        modules=modules,
        module_to_relpath=module_to_relpath,
        package_root="thesis_pkg",
    )
    nav_lines_b = scaffold_docs._build_auto_nav_lines(
        modules=list(reversed(modules)),
        module_to_relpath=module_to_relpath,
        package_root="thesis_pkg",
    )
    assert nav_lines_a == nav_lines_b
    pipeline_line = next(line for line in nav_lines_a if line.startswith("    - pipeline: "))
    assert nav_lines_a.index("    - core:") < nav_lines_a.index(pipeline_line)

"""Generate FUNCTION_INDEX.md and FUNCTION_INDEX.json from the repo.

Run from the repo root:
    python tools/generate_function_index.py

Outputs:
    FUNCTION_INDEX.md   — markdown table grouped by file
    FUNCTION_INDEX.json — dict keyed by qualified name for programmatic lookup
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

# Directories to skip entirely
SKIP_DIRS = {".venv", "__pycache__", ".git", ".mypy_cache", ".pytest_cache", "node_modules", ".tox"}

# Directories whose files are treated as standalone scripts (no package import path)
SCRIPT_DIRS = {"tools", "tests"}


def _is_package_path(rel: Path) -> bool:
    """True if the path lives under src/ (installable package)."""
    return rel.parts[0] == "src"


def _import_path(rel: Path) -> str | None:
    """Derive dotted import path from a relative .py file path.

    src/thesis_pkg/core/foo.py -> thesis_pkg.core.foo
    anything else               -> None
    """
    if not _is_package_path(rel):
        return None
    # drop src/ prefix and .py suffix
    parts = list(rel.with_suffix("").parts[1:])
    # drop __init__ suffix so the module is the package itself
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else None


def _collect(file: Path, rel: Path) -> list[dict]:
    """Parse a single .py file and return a list of definition records."""
    try:
        source = file.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(file))
    except SyntaxError:
        return []

    import_path = _import_path(rel)
    records: list[dict] = []

    for node in ast.walk(tree):
        # Only module-level and class-level definitions (skip deeply nested)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Determine if this is at module level or directly inside a class
            parent = getattr(node, "_parent", None)
            if parent is None or isinstance(parent, ast.ClassDef):
                kind = "method" if isinstance(parent, ast.ClassDef) else "function"
                parent_class = parent.name if isinstance(parent, ast.ClassDef) else None
                qualified_name = f"{parent_class}.{node.name}" if parent_class else node.name
                import_stmt = (
                    f"from {import_path} import {parent_class or node.name}"
                    if import_path
                    else "N/A (script)"
                )
                records.append({
                    "name": node.name,
                    "qualified_name": qualified_name,
                    "type": kind,
                    "parent_class": parent_class,
                    "import_path": import_path,
                    "import_stmt": import_stmt,
                    "file": str(rel).replace("\\", "/"),
                    "line": node.lineno,
                })
        elif isinstance(node, ast.ClassDef):
            parent = getattr(node, "_parent", None)
            if parent is None or isinstance(parent, ast.Module):
                import_stmt = (
                    f"from {import_path} import {node.name}"
                    if import_path
                    else "N/A (script)"
                )
                records.append({
                    "name": node.name,
                    "qualified_name": node.name,
                    "type": "class",
                    "parent_class": None,
                    "import_path": import_path,
                    "import_stmt": import_stmt,
                    "file": str(rel).replace("\\", "/"),
                    "line": node.lineno,
                })

    return records


def _attach_parents(tree: ast.AST) -> None:
    """Annotate every AST node with a _parent attribute."""
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child._parent = node  # type: ignore[attr-defined]


def collect_file(file: Path, rel: Path) -> list[dict]:
    """Parse with parent annotations."""
    try:
        source = file.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(file))
    except SyntaxError:
        return []
    _attach_parents(tree)
    import_path = _import_path(rel)
    records: list[dict] = []

    for node in ast.walk(tree):
        parent = getattr(node, "_parent", None)

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if parent is None or isinstance(parent, (ast.ClassDef, ast.Module)):
                kind = "method" if isinstance(parent, ast.ClassDef) else "function"
                parent_class = parent.name if isinstance(parent, ast.ClassDef) else None
                qualified_name = f"{parent_class}.{node.name}" if parent_class else node.name
                import_stmt = (
                    f"from {import_path} import {parent_class or node.name}"
                    if import_path
                    else "N/A (script)"
                )
                records.append({
                    "name": node.name,
                    "qualified_name": qualified_name,
                    "type": kind,
                    "parent_class": parent_class,
                    "import_path": import_path,
                    "import_stmt": import_stmt,
                    "file": str(rel).replace("\\", "/"),
                    "line": node.lineno,
                })

        elif isinstance(node, ast.ClassDef):
            if parent is None or isinstance(parent, ast.Module):
                import_stmt = (
                    f"from {import_path} import {node.name}"
                    if import_path
                    else "N/A (script)"
                )
                records.append({
                    "name": node.name,
                    "qualified_name": node.name,
                    "type": "class",
                    "parent_class": None,
                    "import_path": import_path,
                    "import_stmt": import_stmt,
                    "file": str(rel).replace("\\", "/"),
                    "line": node.lineno,
                })

    return sorted(records, key=lambda r: r["line"])


def walk_repo(root: Path) -> list[dict]:
    all_records: list[dict] = []
    for py_file in sorted(root.rglob("*.py")):
        rel = py_file.relative_to(root)
        # Skip excluded dirs
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        all_records.extend(collect_file(py_file, rel))
    return all_records


def write_markdown(records: list[dict], out: Path) -> None:
    from collections import defaultdict
    by_file: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_file[r["file"]].append(r)

    lines = [
        "# Function & Class Index",
        "",
        "Auto-generated by `tools/generate_function_index.py`.",
        "Re-run to refresh after adding or removing definitions.",
        "",
        "---",
        "",
    ]

    for file_path in sorted(by_file.keys()):
        defs = by_file[file_path]
        lines.append(f"## `{file_path}`")
        lines.append("")
        lines.append("| Name | Type | Import / Usage | Line |")
        lines.append("|------|------|----------------|------|")
        for d in defs:
            name = d["qualified_name"]
            kind = d["type"]
            stmt = d["import_stmt"]
            line = d["line"]
            lines.append(f"| `{name}` | {kind} | `{stmt}` | {line} |")
        lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written: {out}  ({len(records)} definitions, {len(by_file)} files)")


def write_json(records: list[dict], out: Path) -> None:
    # Key by "file::qualified_name" to avoid collisions
    index = {}
    for r in records:
        key = f"{r['file']}::{r['qualified_name']}"
        index[key] = r
    out.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Written: {out}")


def main() -> None:
    root = Path(__file__).parent.parent.resolve()
    print(f"Scanning: {root}")
    records = walk_repo(root)
    write_markdown(records, root / "FUNCTION_INDEX.md")
    write_json(records, root / "FUNCTION_INDEX.json")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path


AUTO_NAV_BEGIN = "# BEGIN AUTO-REFERENCE-NAV"
AUTO_NAV_END = "# END AUTO-REFERENCE-NAV"
BEHAVIOR_EVIDENCE_REL_PATH = "reference/behavior_evidence.md"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _module_to_doc_relpath(module_name: str, package_root: str) -> str:
    parts = module_name.split(".")
    if parts and parts[0] == package_root:
        parts = parts[1:]
    return str(Path("reference").joinpath(*parts).with_suffix(".md")).replace("\\", "/")


def _title_for_module(module_name: str) -> str:
    parts = module_name.split(".")
    if parts and parts[-1] == "__init__":
        return f"{'.'.join(parts[:-1])} package"
    return module_name


def _mkdocstrings_target(module_name: str) -> str:
    if module_name.endswith(".__init__"):
        return module_name[: -len(".__init__")]
    return module_name


def _nav_label(module_name: str) -> str:
    if module_name.endswith(".__init__"):
        return module_name[: -len(".__init__")]
    return module_name


def _render_module_page(
    module_name: str,
    module_data: dict,
    outward_data: dict | None,
    import_data: dict | None,
) -> str:
    classes = module_data.get("classes", [])
    functions = module_data.get("functions", [])
    constants = module_data.get("constants", [])
    re_exports = module_data.get("re_exports", [])

    module_doc = module_data.get("docstring") or ""
    module_doc_line = module_doc.splitlines()[0].strip() if module_doc else ""
    has_doc = bool(module_data.get("has_docstring"))
    has_all = bool(module_data.get("has_all"))

    imported_by_modules = []
    imported_by_symbols = []
    if import_data:
        imported_by_modules = import_data.get("imported_by_modules", [])
        imported_by_symbols = import_data.get("imported_by_symbols", [])

    outward_reasons = []
    re_exported_by = []
    is_outward = False
    if outward_data:
        is_outward = True
        outward_reasons = outward_data.get("outward_reasons", [])
        re_exported_by = outward_data.get("re_exported_by", [])

    lines: list[str] = [
        f"# { _title_for_module(module_name) }",
        "",
        f"`{module_name}`",
        "",
        "## Summary",
        "",
        f"- Module docstring present: `{has_doc}`",
        f"- Defines `__all__`: `{has_all}`",
        f"- Classes: `{len(classes)}`",
        f"- Functions: `{len(functions)}`",
        f"- Constants/registry names: `{len(constants)}`",
        f"- Re-exports detected: `{len(re_exports)}`",
        f"- Outward-facing (import evidence): `{is_outward}`",
    ]
    if module_doc_line:
        lines.extend(["", f"Docstring summary: {module_doc_line}"])

    lines.extend(["", "## Import Evidence", ""])
    lines.append(f"- Imported by modules: `{len(imported_by_modules)}`")
    if imported_by_modules:
        sample = ", ".join(f"`{name}`" for name in imported_by_modules[:8])
        lines.append(f"- Importer sample: {sample}")
    lines.append(f"- Imported by symbol edges: `{len(imported_by_symbols)}`")
    if imported_by_symbols:
        symbol_sample = ", ".join(
            f"`{edge.get('importer')}::{edge.get('symbol')}`" for edge in imported_by_symbols[:8]
        )
        lines.append(f"- Symbol import sample: {symbol_sample}")
    if outward_reasons:
        lines.append(f"- Outward reasons: {', '.join(f'`{reason}`' for reason in outward_reasons)}")
    if re_exported_by:
        lines.append(f"- Re-exported by: {', '.join(f'`{name}`' for name in re_exported_by[:8])}")

    lines.extend(["", "## Constants and Re-exports", ""])
    if constants:
        lines.append("### Constants / Registry Names")
        lines.append("")
        for entry in constants[:50]:
            lines.append(f"- `{entry.get('name')}`")
        if len(constants) > 50:
            lines.append(f"- ... ({len(constants) - 50} more)")
        lines.append("")
    else:
        lines.extend(["- No top-level constants detected.", ""])

    if re_exports:
        lines.append("### Re-exports via `__all__`")
        lines.append("")
        for row in re_exports[:50]:
            export = row.get("export", "")
            source_module = row.get("source_module", "")
            source_symbol = row.get("source_symbol", "")
            if source_symbol:
                lines.append(f"- `{export}` <- `{source_module}.{source_symbol}`")
            else:
                lines.append(f"- `{export}` <- `{source_module}`")
        if len(re_exports) > 50:
            lines.append(f"- ... ({len(re_exports) - 50} more)")
        lines.append("")
    else:
        lines.extend(["- No `__all__`-based re-exports detected.", ""])

    lines.extend(["## API Rendering", "", f"::: {_mkdocstrings_target(module_name)}", ""])
    return "\n".join(lines)


def _render_reference_index(
    modules: list[str],
    module_to_relpath: dict[str, str],
    inventory: dict,
    outward_api_modules: dict,
) -> str:
    n_total = len(modules)
    n_outward = len(outward_api_modules)
    n_no_defs = sum(
        1
        for module in modules
        if not inventory[module].get("classes") and not inventory[module].get("functions")
    )
    n_with_all = sum(1 for module in modules if inventory[module].get("has_all"))

    lines: list[str] = [
        "# API Reference",
        "",
        "Automatically generated API/module reference index.",
        "",
        "## Coverage Summary",
        "",
        f"- Total modules: `{n_total}`",
        f"- Outward-facing modules (import evidence): `{n_outward}`",
        f"- Modules with no classes/functions: `{n_no_defs}`",
        f"- Modules defining `__all__`: `{n_with_all}`",
        "",
        "## Module Index",
        "",
    ]

    for module in modules:
        rel_path = module_to_relpath[module]
        cls_count = len(inventory[module].get("classes", []))
        fn_count = len(inventory[module].get("functions", []))
        const_count = len(inventory[module].get("constants", []))
        outward = module in outward_api_modules
        lines.append(
            f"- [`{module}`]({rel_path.replace('reference/', '')}) "
            f"(classes={cls_count}, functions={fn_count}, constants={const_count}, outward={outward})"
        )

    lines.append("")
    return "\n".join(lines)


def _build_auto_nav_lines(modules: list[str], module_to_relpath: dict[str, str]) -> list[str]:
    """Build the managed reference nav block for MkDocs.

    WHY (TODO by Erik): Keep a single generated nav contract so reference pages
    stay discoverable without manual `mkdocs.yml` maintenance.
    """
    nav_lines = [
        "  - Reference:",
        "    - Overview: reference/index.md",
        f"    - Behavior Evidence: {BEHAVIOR_EVIDENCE_REL_PATH}",
    ]
    for module in modules:
        nav_lines.append(f"    - {_nav_label(module)}: {module_to_relpath[module]}")
    return nav_lines


def _update_mkdocs_nav(mkdocs_config: Path, nav_lines: list[str]) -> None:
    text_lines = mkdocs_config.read_text(encoding="utf-8").splitlines()
    begin_idx = None
    end_idx = None
    for idx, line in enumerate(text_lines):
        if AUTO_NAV_BEGIN in line:
            begin_idx = idx
        if AUTO_NAV_END in line:
            end_idx = idx
    if begin_idx is None or end_idx is None or begin_idx >= end_idx:
        raise ValueError(
            f"Managed nav markers not found or invalid in {mkdocs_config}: "
            f"expected {AUTO_NAV_BEGIN} .. {AUTO_NAV_END}"
        )
    updated = text_lines[: begin_idx + 1] + nav_lines + text_lines[end_idx:]
    mkdocs_config.write_text("\n".join(updated) + "\n", encoding="utf-8")


def _bootstrap_top_level_docs(docs_dir: Path) -> None:
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "index.md").write_text(
        "# NLP Thesis Documentation\n\nWelcome to the documentation site.\n",
        encoding="utf-8",
    )
    (docs_dir / "architecture").mkdir(parents=True, exist_ok=True)
    (docs_dir / "architecture" / "index.md").write_text(
        "# Architecture\n\nSystem behavior and pipelines.\n",
        encoding="utf-8",
    )
    (docs_dir / "decisions").mkdir(parents=True, exist_ok=True)
    (docs_dir / "decisions" / "index.md").write_text(
        "# Decisions (ADRs)\n\nArchitecture decision records.\n",
        encoding="utf-8",
    )


def _ensure_behavior_evidence_placeholder(docs_dir: Path) -> None:
    """Ensure the behavior evidence page always exists as a nav-safe target.

    WHY (TODO by Erik): MkDocs nav should be stable even before traces are run.
    """
    path = docs_dir / Path(BEHAVIOR_EVIDENCE_REL_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    path.write_text(
        (
            "# Behavior Evidence\n\n"
            "Dynamic trace artifacts have not been generated yet.\n\n"
            "Run `python tools/docs_trace.py all` to generate behavior evidence artifacts.\n"
        ),
        encoding="utf-8",
    )


def generate_reference_docs(
    *,
    inventory_file: Path,
    outward_api_file: Path,
    import_evidence_file: Path,
    docs_dir: Path,
    package_root: str,
) -> tuple[list[str], dict[str, str], dict]:
    inventory = _load_json(inventory_file)
    outward_api = _load_json(outward_api_file)
    import_evidence = _load_json(import_evidence_file)

    modules = sorted(inventory.keys())
    outward_modules = outward_api.get("modules", {})
    import_modules = import_evidence.get("modules", {})

    docs_ref_dir = docs_dir / "reference"
    docs_ref_dir.mkdir(parents=True, exist_ok=True)

    module_to_relpath: dict[str, str] = {}
    for module_name in modules:
        rel = _module_to_doc_relpath(module_name, package_root)
        module_to_relpath[module_name] = rel
        md_file_path = docs_dir / Path(rel)
        md_file_path.parent.mkdir(parents=True, exist_ok=True)
        content = _render_module_page(
            module_name=module_name,
            module_data=inventory[module_name],
            outward_data=outward_modules.get(module_name),
            import_data=import_modules.get(module_name),
        )
        md_file_path.write_text(content, encoding="utf-8")

    ref_index = _render_reference_index(
        modules=modules,
        module_to_relpath=module_to_relpath,
        inventory=inventory,
        outward_api_modules=outward_modules,
    )
    (docs_ref_dir / "index.md").write_text(ref_index, encoding="utf-8")
    _ensure_behavior_evidence_placeholder(docs_dir)
    return modules, module_to_relpath, outward_api


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate docs/reference pages from metadata.")
    parser.add_argument("--inventory", type=Path, default=Path("docs_metadata/inventory.json"))
    parser.add_argument("--outward-api", type=Path, default=Path("docs_metadata/outward_api.json"))
    parser.add_argument("--import-evidence", type=Path, default=Path("docs_metadata/import_evidence.json"))
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--mkdocs-config", type=Path, default=Path("mkdocs.yml"))
    parser.add_argument("--package-root", type=str, default="thesis_pkg")
    parser.add_argument("--update-nav", action="store_true")
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument(
        "--nav-fragment-out",
        type=Path,
        default=Path("docs_metadata/reference_nav.yml"),
        help="Write generated Reference nav fragment for inspection/debugging.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.bootstrap:
        _bootstrap_top_level_docs(args.docs_dir)

    modules, module_to_relpath, outward_api = generate_reference_docs(
        inventory_file=args.inventory,
        outward_api_file=args.outward_api,
        import_evidence_file=args.import_evidence,
        docs_dir=args.docs_dir,
        package_root=args.package_root,
    )

    nav_lines = _build_auto_nav_lines(modules=modules, module_to_relpath=module_to_relpath)
    args.nav_fragment_out.parent.mkdir(parents=True, exist_ok=True)
    args.nav_fragment_out.write_text("\n".join(nav_lines) + "\n", encoding="utf-8")

    if args.update_nav:
        _update_mkdocs_nav(args.mkdocs_config, nav_lines)
        nav_note = f" and updated {args.mkdocs_config}"
    else:
        nav_note = ""

    n_outward = len(outward_api.get("all_outward_modules", []))
    print(
        f"Generated {len(modules)} module pages under {args.docs_dir / 'reference'} "
        f"(outward={n_outward}); wrote nav fragment to {args.nav_fragment_out}{nav_note}"
    )


if __name__ == "__main__":
    main()

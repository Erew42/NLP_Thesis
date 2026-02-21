from __future__ import annotations

import argparse
import ast
import json
from collections import defaultdict
from pathlib import Path


def _ensure_package_dir(src_dir: Path, package_root: str) -> Path:
    if src_dir.name == package_root and (src_dir / "__init__.py").exists():
        return src_dir
    candidate = src_dir / package_root
    if candidate.exists() and (candidate / "__init__.py").exists():
        return candidate
    raise ValueError(
        f"Could not resolve package directory for package_root={package_root!r} under src_dir={src_dir}"
    )


def _parse_all_exports(tree: ast.Module) -> list[str]:
    exports: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
            continue
        if isinstance(node.value, (ast.List, ast.Tuple, ast.Set)):
            for elt in node.value.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    exports.append(elt.value)
    return sorted(set(exports))


def _resolve_import_from(module_name: str, node: ast.ImportFrom) -> str | None:
    base_parts = module_name.split(".")
    level = int(node.level or 0)
    if level > len(base_parts):
        return None
    anchor_parts = base_parts[:-level] if level else []
    mod_parts = node.module.split(".") if node.module else []
    resolved_parts = [part for part in (*anchor_parts, *mod_parts) if part]
    if not resolved_parts:
        return None
    return ".".join(resolved_parts)


def _collect_metadata(package_dir: Path, package_root: str) -> tuple[dict, dict, dict, dict, dict]:
    inventory: dict[str, dict[str, object]] = {}
    public_api: dict[str, dict[str, object]] = {}
    used_by: dict[str, dict[str, object]] = {}

    reverse_modules: dict[str, set[str]] = defaultdict(set)
    reverse_symbols: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    module_symbol_import_records: dict[str, list[dict[str, str]]] = defaultdict(list)
    reverse_re_exports: dict[str, set[str]] = defaultdict(set)

    package_parent = package_dir.parent

    for py_file in sorted(package_dir.rglob("*.py")):
        module_rel = py_file.relative_to(package_parent).with_suffix("")
        module_name = ".".join(module_rel.parts)

        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
        except SyntaxError:
            continue

        module_doc = ast.get_docstring(tree)
        all_exports = _parse_all_exports(tree)
        has_all = bool(all_exports)

        module_defs: dict[str, list[dict[str, object]]] = {
            "classes": [],
            "functions": [],
            "constants": [],
            "re_exports": [],
        }
        module_imports: list[str] = []
        module_import_symbols: list[dict[str, str]] = []
        imported_bindings: dict[str, dict[str, str]] = {}

        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_imports.append(alias.name)
                    local_name = alias.asname or alias.name.split(".", 1)[0]
                    imported_bindings[local_name] = {
                        "source_module": alias.name,
                        "source_symbol": "",
                    }
            elif isinstance(node, ast.ImportFrom):
                resolved_mod = _resolve_import_from(module_name, node)
                if not resolved_mod:
                    continue
                module_imports.append(resolved_mod)
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    local_name = alias.asname or alias.name
                    module_import_symbols.append(
                        {
                            "module": resolved_mod,
                            "name": alias.name,
                            "asname": local_name,
                        }
                    )
                    imported_bindings[local_name] = {
                        "source_module": resolved_mod,
                        "source_symbol": alias.name,
                    }
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                module_defs["functions"].append(
                    {
                        "name": node.name,
                        "has_docstring": ast.get_docstring(node) is not None,
                        "is_async": isinstance(node, ast.AsyncFunctionDef),
                    }
                )
            elif isinstance(node, ast.ClassDef):
                module_defs["classes"].append(
                    {
                        "name": node.name,
                        "has_docstring": ast.get_docstring(node) is not None,
                    }
                )
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        name = target.id
                        if name == "__all__":
                            continue
                        module_defs["constants"].append(
                            {
                                "name": name,
                            }
                        )
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name):
                    name = node.target.id
                    if name == "__all__":
                        continue
                    module_defs["constants"].append(
                        {
                            "name": name,
                        }
                    )

        # Capture re-export evidence from modules that define __all__.
        if has_all:
            for exported_name in all_exports:
                binding = imported_bindings.get(exported_name)
                if not binding:
                    continue
                module_defs["re_exports"].append(
                    {
                        "export": exported_name,
                        "source_module": binding["source_module"],
                        "source_symbol": binding["source_symbol"],
                    }
                )
                source_module = binding["source_module"]
                if source_module.startswith(package_root):
                    reverse_re_exports[source_module].add(module_name)
                source_symbol = binding["source_symbol"]
                if source_symbol:
                    candidate = f"{source_module}.{source_symbol}"
                    if candidate.startswith(package_root):
                        reverse_re_exports[candidate].add(module_name)

        # Deterministic sort.
        for key in ("classes", "functions", "constants", "re_exports"):
            module_defs[key] = sorted(module_defs[key], key=lambda entry: entry.get("name", entry.get("export", "")))
        module_imports = sorted(set(module_imports))
        module_import_symbols = sorted(
            module_import_symbols,
            key=lambda item: (item["module"], item["name"], item["asname"]),
        )

        inventory[module_name] = {
            "path": py_file.relative_to(package_parent).as_posix(),
            "docstring": module_doc,
            "has_docstring": module_doc is not None,
            "has_all": has_all,
            "all_exports": all_exports,
            "classes": module_defs["classes"],
            "functions": module_defs["functions"],
            "constants": module_defs["constants"],
            "re_exports": module_defs["re_exports"],
            "imports": module_imports,
            "import_symbols": module_import_symbols,
        }

        # Naming/__all__ based public API view.
        if has_all:
            public_names = set(all_exports)
        else:
            public_names = set()
            for key in ("classes", "functions", "constants"):
                for entry in module_defs[key]:
                    name = str(entry["name"])
                    if not name.startswith("_"):
                        public_names.add(name)

        public_api[module_name] = {
            "has_all": has_all,
            "classes": [entry for entry in module_defs["classes"] if entry["name"] in public_names],
            "functions": [entry for entry in module_defs["functions"] if entry["name"] in public_names],
            "constants": [entry for entry in module_defs["constants"] if entry["name"] in public_names],
            "re_exports": module_defs["re_exports"],
        }

        used_by[module_name] = {
            "imports": module_imports,
            "import_symbols": module_import_symbols,
        }

    # Build reverse import evidence.
    module_names = set(inventory)
    for importer, meta in used_by.items():
        if not importer.startswith(package_root):
            continue
        for imported_mod in meta["imports"]:
            if imported_mod in module_names and imported_mod != importer:
                reverse_modules[imported_mod].add(importer)
        for edge in meta["import_symbols"]:
            source_module = edge["module"]
            symbol = edge["name"]
            if source_module in module_names and source_module != importer:
                reverse_modules[source_module].add(importer)
                reverse_symbols[source_module][symbol].add(importer)
                module_symbol_import_records[source_module].append(
                    {
                        "importer": importer,
                        "symbol": symbol,
                        "asname": edge["asname"],
                    }
                )

            # Evidence for `from pkg.subpkg import module` style submodule imports.
            candidate_submodule = f"{source_module}.{symbol}"
            if candidate_submodule in module_names and candidate_submodule != importer:
                reverse_modules[candidate_submodule].add(importer)
                reverse_symbols[candidate_submodule]["__module_import__"].add(importer)
                module_symbol_import_records[candidate_submodule].append(
                    {
                        "importer": importer,
                        "symbol": "__module_import__",
                        "asname": edge["asname"],
                    }
                )

    import_evidence_modules: dict[str, dict[str, object]] = {}
    for module_name in sorted(module_names):
        imported_by_modules = sorted(reverse_modules.get(module_name, set()))
        imported_by_symbols = sorted(
            module_symbol_import_records.get(module_name, []),
            key=lambda row: (row["importer"], row["symbol"], row["asname"]),
        )
        import_evidence_modules[module_name] = {
            "imports": inventory[module_name]["imports"],
            "import_symbols": inventory[module_name]["import_symbols"],
            "imported_by_modules": imported_by_modules,
            "imported_by_symbols": imported_by_symbols,
            "importer_count": len(imported_by_modules),
        }

    import_evidence = {
        "package_root": package_root,
        "modules": import_evidence_modules,
        "reverse_modules": {
            module_name: sorted(importers)
            for module_name, importers in sorted(reverse_modules.items())
        },
        "reverse_symbols": {
            module_name: {
                symbol: sorted(importers)
                for symbol, importers in sorted(symbol_map.items())
            }
            for module_name, symbol_map in sorted(reverse_symbols.items())
        },
    }

    outward_modules: dict[str, dict[str, object]] = {}
    for module_name in sorted(module_names):
        importers = [m for m in reverse_modules.get(module_name, set()) if m.startswith(package_root)]
        re_exporters = [m for m in reverse_re_exports.get(module_name, set()) if m.startswith(package_root)]
        if not importers and not re_exporters:
            continue
        reasons: list[str] = []
        if importers:
            reasons.append("imported_by_module")
        if re_exporters:
            reasons.append("re_exported_via_all")
        outward_modules[module_name] = {
            "imported_by_modules": sorted(set(importers)),
            "re_exported_by": sorted(set(re_exporters)),
            "outward_reasons": reasons,
        }

    outward_api = {
        "package_root": package_root,
        "all_outward_modules": sorted(outward_modules),
        "modules": outward_modules,
    }

    return inventory, public_api, import_evidence, outward_api, used_by


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract docs metadata from Python AST.")
    parser.add_argument("--src-dir", type=Path, default=Path("src/thesis_pkg"))
    parser.add_argument("--package-root", type=str, default="thesis_pkg")
    parser.add_argument("--out-dir", type=Path, default=Path("docs_metadata"))
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=("all", "inventory", "public_api", "import_evidence", "outward_api", "used_by"),
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    package_dir = _ensure_package_dir(args.src_dir, args.package_root)
    inventory, public_api, import_evidence, outward_api, used_by = _collect_metadata(
        package_dir=package_dir,
        package_root=args.package_root,
    )

    outputs = {
        "inventory": ("inventory.json", inventory),
        "public_api": ("public_api.json", public_api),
        "import_evidence": ("import_evidence.json", import_evidence),
        "outward_api": ("outward_api.json", outward_api),
        "used_by": ("used_by.json", used_by),
    }

    selected = outputs.keys() if args.mode == "all" else (args.mode,)
    for key in selected:
        filename, payload = outputs[key]
        _write_json(args.out_dir / filename, payload)

    written = ", ".join(outputs[key][0] for key in selected)
    print(f"Wrote metadata to {args.out_dir.resolve()} ({written})")


if __name__ == "__main__":
    main()

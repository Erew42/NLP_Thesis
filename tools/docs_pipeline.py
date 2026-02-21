from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelinePaths:
    repo_root: Path
    src_dir: Path
    docs_dir: Path
    metadata_dir: Path
    mkdocs_config: Path
    package_root: str

    @property
    def inventory_path(self) -> Path:
        return self.metadata_dir / "inventory.json"

    @property
    def outward_api_path(self) -> Path:
        return self.metadata_dir / "outward_api.json"

    @property
    def import_evidence_path(self) -> Path:
        return self.metadata_dir / "import_evidence.json"

    @property
    def nav_fragment_path(self) -> Path:
        return self.metadata_dir / "reference_nav.yml"

    @property
    def behavior_trace_dir(self) -> Path:
        return self.metadata_dir / "behavior"

    @property
    def behavior_page_path(self) -> Path:
        return self.docs_dir / "reference" / "behavior_evidence.md"


def build_subprocess_env(
    *,
    base_env: dict[str, str] | None = None,
    enforce_mkdocs_utf8: bool = False,
    os_name: str | None = None,
) -> dict[str, str]:
    env = dict(base_env if base_env is not None else os.environ)
    effective_os_name = os_name if os_name is not None else os.name
    if enforce_mkdocs_utf8 and effective_os_name == "nt":
        env["PYTHONIOENCODING"] = "utf-8"
    return env


def _resolve_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path)


def _run_command(
    *,
    repo_root: Path,
    command: list[str],
    env: dict[str, str] | None = None,
) -> int:
    print(f"> {' '.join(command)}")
    result = subprocess.run(
        command,
        cwd=repo_root,
        env=env,
        check=False,
    )
    return int(result.returncode)


def run_extract(paths: PipelinePaths) -> int:
    command = [
        sys.executable,
        "extract_ast.py",
        "--src-dir",
        str(paths.src_dir),
        "--package-root",
        paths.package_root,
        "--out-dir",
        str(paths.metadata_dir),
        "--mode",
        "all",
    ]
    return _run_command(repo_root=paths.repo_root, command=command)


def run_scaffold(paths: PipelinePaths) -> int:
    command = [
        sys.executable,
        "scaffold_docs.py",
        "--inventory",
        str(paths.inventory_path),
        "--outward-api",
        str(paths.outward_api_path),
        "--import-evidence",
        str(paths.import_evidence_path),
        "--docs-dir",
        str(paths.docs_dir),
        "--mkdocs-config",
        str(paths.mkdocs_config),
        "--package-root",
        paths.package_root,
        "--nav-fragment-out",
        str(paths.nav_fragment_path),
        "--update-nav",
    ]
    return _run_command(repo_root=paths.repo_root, command=command)


def run_mkdocs_build(paths: PipelinePaths) -> int:
    command = [sys.executable, "-m", "mkdocs", "build", "-f", str(paths.mkdocs_config)]
    env = build_subprocess_env(enforce_mkdocs_utf8=True)
    return _run_command(repo_root=paths.repo_root, command=command, env=env)


def run_trace(
    paths: PipelinePaths,
    *,
    trace_scope: str,
    seed: int,
    overwrite: bool,
) -> int:
    command = [
        sys.executable,
        "tools/docs_trace.py",
        "all",
        "--repo-root",
        str(paths.repo_root),
        "--out-dir",
        str(paths.behavior_trace_dir),
        "--behavior-page",
        str(paths.behavior_page_path),
        "--trace-scope",
        trace_scope,
        "--seed",
        str(seed),
    ]
    if overwrite:
        command.append("--overwrite")
    return _run_command(repo_root=paths.repo_root, command=command)


def run_check(paths: PipelinePaths, *, skip_build: bool, require_trace: bool) -> int:
    command = [
        sys.executable,
        "tools/docs_check.py",
        "--repo-root",
        str(paths.repo_root),
        "--src-dir",
        str(paths.src_dir),
        "--inventory",
        str(paths.inventory_path),
        "--outward-api",
        str(paths.outward_api_path),
        "--import-evidence",
        str(paths.import_evidence_path),
        "--nav-fragment",
        str(paths.nav_fragment_path),
        "--mkdocs-config",
        str(paths.mkdocs_config),
        "--docs-dir",
        str(paths.docs_dir),
        "--trace-dir",
        str(paths.behavior_trace_dir),
        "--package-root",
        paths.package_root,
    ]
    if require_trace:
        command.append("--require-trace")
    if skip_build:
        command.append("--skip-build")
    else:
        command.append("--run-build")
    return _run_command(repo_root=paths.repo_root, command=command)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run docs metadata/scaffold/build/check pipeline.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--src-dir", type=Path, default=Path("src/thesis_pkg"))
    parser.add_argument("--package-root", type=str, default="thesis_pkg")
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--metadata-dir", type=Path, default=Path("docs_metadata"))
    parser.add_argument("--mkdocs-config", type=Path, default=Path("mkdocs.yml"))

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("extract", help="Run AST metadata extraction.")
    subparsers.add_parser("scaffold", help="Generate reference docs and update managed nav block.")
    subparsers.add_parser("build", help="Run mkdocs build.")
    trace_parser = subparsers.add_parser("trace", help="Generate and render dynamic behavior trace artifacts.")
    trace_parser.add_argument(
        "--trace-scope",
        choices=("sec_ccm", "boundary", "both"),
        default="both",
        help="Trace workload scope.",
    )
    trace_parser.add_argument("--seed", type=int, default=42, help="Deterministic seed for trace sampling.")
    trace_parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing trace outputs.")

    check_parser = subparsers.add_parser("check", help="Run docs checks.")
    check_parser.add_argument("--skip-build", action="store_true", help="Skip mkdocs build execution in docs check.")
    check_parser.add_argument("--require-trace", action="store_true", help="Require trace manifest/artifacts.")

    all_parser = subparsers.add_parser("all", help="Run extract -> scaffold -> check.")
    all_parser.add_argument("--skip-build", action="store_true", help="Skip mkdocs build execution in docs check.")
    all_parser.add_argument("--with-trace", action="store_true", help="Run trace generation before docs checks.")
    all_parser.add_argument(
        "--trace-scope",
        choices=("sec_ccm", "boundary", "both"),
        default="both",
        help="Trace workload scope when --with-trace is enabled.",
    )
    all_parser.add_argument("--seed", type=int, default=42, help="Deterministic seed for trace sampling.")
    all_parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing trace outputs.")
    all_parser.add_argument("--require-trace", action="store_true", help="Require trace artifacts during check.")

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    paths = PipelinePaths(
        repo_root=repo_root,
        src_dir=_resolve_path(repo_root, args.src_dir),
        docs_dir=_resolve_path(repo_root, args.docs_dir),
        metadata_dir=_resolve_path(repo_root, args.metadata_dir),
        mkdocs_config=_resolve_path(repo_root, args.mkdocs_config),
        package_root=args.package_root,
    )

    if args.command == "extract":
        return run_extract(paths)
    if args.command == "scaffold":
        return run_scaffold(paths)
    if args.command == "build":
        return run_mkdocs_build(paths)
    if args.command == "trace":
        return run_trace(
            paths,
            trace_scope=str(args.trace_scope),
            seed=int(args.seed),
            overwrite=bool(args.overwrite),
        )
    if args.command == "check":
        return run_check(
            paths,
            skip_build=bool(args.skip_build),
            require_trace=bool(args.require_trace),
        )

    if args.command == "all":
        extract_rc = run_extract(paths)
        if extract_rc != 0:
            return extract_rc

        scaffold_rc = run_scaffold(paths)
        if scaffold_rc != 0:
            return scaffold_rc

        require_trace = bool(args.require_trace)
        if bool(args.with_trace):
            trace_rc = run_trace(
                paths,
                trace_scope=str(args.trace_scope),
                seed=int(args.seed),
                overwrite=bool(args.overwrite),
            )
            if trace_rc != 0:
                return trace_rc
            require_trace = True

        return run_check(
            paths,
            skip_build=bool(args.skip_build),
            require_trace=require_trace,
        )

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT_ENV_VAR = "NLP_THESIS_REPO_ROOT"


def resolve_repo_root(start: Path | None = None) -> Path:
    candidates: list[Path] = []
    env_root = os.environ.get(REPO_ROOT_ENV_VAR)
    if env_root:
        candidates.append(Path(env_root).expanduser().resolve())

    base = Path.cwd().resolve() if start is None else start.resolve()
    candidates.extend([base, *base.parents])

    package_path = Path(__file__).resolve()
    candidates.extend(package_path.parents)

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "src" / "thesis_pkg" / "pipeline.py").exists():
            return candidate
    raise RuntimeError("Could not resolve repository root containing src/thesis_pkg/pipeline.py")


def ensure_repo_src_on_path(repo_root: Path | None = None) -> Path:
    resolved_repo_root = resolve_repo_root() if repo_root is None else repo_root.resolve()
    src_dir = resolved_repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return resolved_repo_root

from __future__ import annotations

import importlib
import importlib.util
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_thesis_native_import_allows_missing_rust_extension(tmp_path: Path) -> None:
    package_dir = tmp_path / "thesis_native"
    package_dir.mkdir()
    shutil.copyfile(REPO_ROOT / "src" / "thesis_native" / "__init__.py", package_dir / "__init__.py")

    original_module = sys.modules.pop("thesis_native", None)
    original_rust_module = sys.modules.pop("thesis_native._lm2011_rust", None)
    sys.path.insert(0, str(tmp_path))
    try:
        module = importlib.import_module("thesis_native")
        assert module._lm2011_rust is None
        assert module.RUST_IMPORT_ERROR is not None
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("thesis_native", None)
        sys.modules.pop("thesis_native._lm2011_rust", None)
        if original_module is not None:
            sys.modules["thesis_native"] = original_module
        if original_rust_module is not None:
            sys.modules["thesis_native._lm2011_rust"] = original_rust_module


def test_sec_compat_shim_allows_missing_rust_extension(monkeypatch) -> None:
    module_name = "_test_sec_core_without_native"
    alias_name = f"{module_name}._lm2011_rust"
    monkeypatch.setitem(
        sys.modules,
        "thesis_native",
        SimpleNamespace(_lm2011_rust=None),
    )
    sys.modules.pop(alias_name, None)

    spec = importlib.util.spec_from_file_location(
        module_name,
        REPO_ROOT / "src" / "thesis_pkg" / "core" / "sec" / "__init__.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)

    spec.loader.exec_module(module)

    assert module._lm2011_rust is None
    assert alias_name not in sys.modules

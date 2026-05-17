from __future__ import annotations

import os
from pathlib import Path

from setuptools import Extension, find_packages, setup


_DISABLE_NATIVE_ENV = "NLP_THESIS_DISABLE_NATIVE_EXTENSIONS"


def _native_extensions_enabled() -> bool:
    value = os.environ.get(_DISABLE_NATIVE_ENV, "")
    return value.strip().lower() not in {"1", "true", "yes", "on"}


def _build_extensions() -> list[Extension]:
    ext = Extension(
        "thesis_pkg.core.sec.extraction_fast",
        sources=[str(Path("src") / "thesis_pkg" / "core" / "sec" / "extraction_fast.pyx")],
    )
    return [ext]


def _cythonize_extensions() -> list[Extension]:
    try:
        from Cython.Build import cythonize
    except ImportError as exc:  # pragma: no cover - build-time guard
        raise RuntimeError(
            "Cython is required to build extraction_fast. Install with `pip install Cython`."
        ) from exc

    return cythonize(
        _build_extensions(),
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
        },
    )


def _rust_extensions() -> list[object]:
    try:
        from setuptools_rust import Binding, RustExtension, Strip
    except ImportError as exc:  # pragma: no cover - build-time guard
        raise RuntimeError(
            "setuptools-rust is required to build native Rust accelerators. "
            f"Install it or set {_DISABLE_NATIVE_ENV}=1 for a pure-Python build."
        ) from exc

    return [
        RustExtension(
            "thesis_native._lm2011_rust",
            "rust/lm2011_rust/Cargo.toml",
            binding=Binding.PyO3,
            debug=False,
            strip=Strip.Debug,
        )
    ]


setup_kwargs = {
    "packages": find_packages(where="src"),
    "package_dir": {"": "src"},
    "zip_safe": False,
}

if _native_extensions_enabled():
    setup_kwargs["ext_modules"] = _cythonize_extensions()
    setup_kwargs["rust_extensions"] = _rust_extensions()

setup(
    **setup_kwargs,
)

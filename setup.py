from __future__ import annotations

from pathlib import Path

from setuptools import Extension, setup


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


setup(
    ext_modules=_cythonize_extensions(),
    package_dir={"": "src"},
)

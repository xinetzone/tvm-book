"""Utilities to locate and load the tvm_book shared library."""

from __future__ import annotations

import sys
import sysconfig
from pathlib import Path
from typing import Iterable, List, Optional

import tvm_ffi


def _library_name() -> str:
    """Return the platform specific name of the compiled extension."""
    if sys.platform.startswith("win"):
        return "tvm_book.dll"
    if sys.platform.startswith("darwin"):
        return "tvm_book.dylib"
    return "tvm_book.so"


def _normalized_platform_tag() -> str:
    """Normalise the current platform tag so it matches our build-dir layout."""
    return (
        sysconfig.get_platform()
        .replace("-", "_")
        .replace(".", "_")
    )


def _candidate_directories(file_dir: Path) -> Iterable[Path]:
    """Yield directories that may contain the compiled library."""
    build_root = file_dir.parent.parent / "build"
    yield file_dir
    yield build_root
    yield build_root / f"py3-none-{_normalized_platform_tag()}"


def _candidate_paths(lib_name: str, roots: Iterable[Path]) -> Iterable[Path]:
    """Yield possible locations for the compiled library."""
    config_subdirs = ("RelWithDebInfo", "Release", "Debug")
    seen: set[Path] = set()
    for base in roots:
        candidates = [base / lib_name]
        candidates.extend(base / sub / lib_name for sub in config_subdirs)
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            yield candidate

    # As a last resort, search the build tree for the first matching library.
    build_root = (Path(__file__).resolve().parent.parent / "build")
    if build_root.exists():
        for match in build_root.rglob(lib_name):
            if match in seen:
                continue
            yield match
            break


def _load_lib() -> tvm_ffi.Module:
    """Load the compiled tvm_book extension module."""
    file_dir = Path(__file__).resolve().parent
    lib_name = _library_name()
    checked_dirs: List[Path] = []

    for candidate in _candidate_paths(lib_name, _candidate_directories(file_dir)):
        parent = candidate.parent
        if parent not in checked_dirs:
            checked_dirs.append(parent)
        if candidate.exists():
            return tvm_ffi.load_module(str(candidate))

    locations = ", ".join(str(path) for path in checked_dirs) or str(file_dir)
    raise FileNotFoundError(f"Could not find {lib_name} in any of: {locations}")


try:
    _LIB = _load_lib()
    _LOAD_ERROR: Optional[Exception] = None
except FileNotFoundError as err:  # pragma: no cover - exercised on systems without a toolchain
    _LIB = None
    _LOAD_ERROR = err

__all__ = ["_LIB", "_LOAD_ERROR"]

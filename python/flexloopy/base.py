"""Utilities to locate and load the flexloopy shared library."""
import os
import sys
import sysconfig
from pathlib import Path
from typing import Iterable, List, Optional

import tvm_ffi


def _library_name() -> str:
    if sys.platform.startswith("win"):
        return "flexloopy.dll"
    if sys.platform.startswith("darwin"):
        return "flexloopy.dylib"
    return "flexloopy.so"


def _normalized_platform_tag() -> str:
    return sysconfig.get_platform().replace("-", "_").replace(".", "_")


def _candidate_directories(file_dir: Path) -> Iterable[Path]:
    build_root = file_dir.parent.parent / "build"
    yield file_dir
    yield build_root
    yield build_root / f"py3-none-{_normalized_platform_tag()}"


def _candidate_paths(lib_name: str, roots: Iterable[Path]) -> Iterable[Path]:
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


_CACHED_PATH: Optional[Path] = None
_LIB: Optional[tvm_ffi.Module] = None


def _env_lib_path(lib_name: str) -> Optional[Path]:
    p = os.environ.get("TVM_BOOK_LIB_PATH")
    if p:
        path = Path(p)
        if path.is_file():
            return path
    d = os.environ.get("TVM_BOOK_LIB_DIR")
    if d:
        dir_path = Path(d)
        cand = dir_path / lib_name
        if cand.exists():
            return cand
    return None


def _resolve_lib_path() -> Path:
    file_dir = Path(__file__).resolve().parent
    lib_name = _library_name()
    override = _env_lib_path(lib_name)
    if override is not None:
        return override
    for candidate in _candidate_paths(lib_name, _candidate_directories(file_dir)):
        if candidate.exists():
            return candidate
    locations = []
    for p in _candidate_paths(lib_name, _candidate_directories(file_dir)):
        locations.append(str(p.parent))
    hint = "; set TVM_BOOK_LIB_PATH or TVM_BOOK_LIB_DIR to override"
    raise FileNotFoundError(f"Could not find {lib_name} in: {', '.join(dict.fromkeys(locations))}{hint}")


def get_lib() -> tvm_ffi.Module:
    global _LIB, _CACHED_PATH
    if _LIB is not None:
        return _LIB
    if _CACHED_PATH is None:
        _CACHED_PATH = _resolve_lib_path()
    _LIB = tvm_ffi.load_module(str(_CACHED_PATH))
    return _LIB


__all__ = ["get_lib", "_LIB"]

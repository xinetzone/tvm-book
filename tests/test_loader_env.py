import os
import importlib
import pytest


def test_env_override_invalid_path_fallback(monkeypatch):
    monkeypatch.setenv("TVM_BOOK_LIB_PATH", str(os.path.join("nonexistent", "tvm_book.dll")))
    import tvm_book.base as base
    m = base.get_lib()
    assert m is not None


def test_get_lib_returns_module(monkeypatch):
    monkeypatch.delenv("TVM_BOOK_LIB_PATH", raising=False)
    monkeypatch.delenv("TVM_BOOK_LIB_DIR", raising=False)
    import tvm_book.base as base
    m = base.get_lib()
    assert m is not None
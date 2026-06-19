import builtins

import pytest

from pfas_lit_rag.langchain_adapter import _load_langchain_types


def test_langchain_adapter_reports_missing_optional_dependency(monkeypatch) -> None:
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("langchain_core"):
            raise ImportError("missing optional dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="uv sync --extra langchain"):
        _load_langchain_types()

import torch
import pytest


@pytest.fixture(autouse=True)
def patch_torch_compile(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda fn, *args, **kwargs: fn)

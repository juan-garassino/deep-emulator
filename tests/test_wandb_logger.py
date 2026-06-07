"""WandbLogger never crashes the training loop."""
from __future__ import annotations

import builtins
import sys

import pytest

from deepEmulator.utils.wandb_logger import WandbLogger


def test_disabled_mode_is_no_op(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    wb = WandbLogger(project="x", name="y")
    assert wb._run is None
    wb.log({"loss": 0.5}, step=1)
    wb.finish()


def test_missing_wandb_does_not_crash(monkeypatch):
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.delitem(sys.modules, "wandb", raising=False)
    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "wandb":
            raise ImportError("simulated absence")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    wb = WandbLogger(project="x")
    assert wb._run is None
    wb.log({"loss": 0.1})
    wb.finish()


def test_no_api_key_and_no_mode_is_no_op(monkeypatch):
    """Critical: in CI/test envs, wandb.init blocks on interactive auth.
    Without WANDB_API_KEY AND without explicit WANDB_MODE, the helper must skip
    init entirely so test runs do not hang.
    """
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_MODE", raising=False)

    class _ShouldNotBeCalled:
        @staticmethod
        def init(**kwargs):
            raise AssertionError("wandb.init should not have been called")

        @staticmethod
        def finish():
            pass

    monkeypatch.setitem(sys.modules, "wandb", _ShouldNotBeCalled)
    wb = WandbLogger(project="x")
    assert wb._run is None
    wb.log({"loss": 0.1})
    wb.finish()


def test_init_exception_degrades_to_noop(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "fake-key-for-test")
    monkeypatch.delenv("WANDB_MODE", raising=False)

    class _FakeWandb:
        @staticmethod
        def init(**kwargs):
            raise RuntimeError("simulated network failure")

        @staticmethod
        def finish():
            pass

    monkeypatch.setitem(sys.modules, "wandb", _FakeWandb)
    wb = WandbLogger(project="x")
    assert wb._run is None
    wb.log({"step": 1, "loss": 0.5}, step=1)
    wb.finish()


def test_log_exception_swallowed(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "fake-key-for-test")
    monkeypatch.delenv("WANDB_MODE", raising=False)

    class _BrokenRun:
        def log(self, data, step=None):
            raise RuntimeError("simulated wandb backend issue")

    class _FakeWandb:
        @staticmethod
        def init(**kwargs):
            return _BrokenRun()

        @staticmethod
        def finish():
            pass

    monkeypatch.setitem(sys.modules, "wandb", _FakeWandb)
    wb = WandbLogger(project="x")
    assert wb._run is not None
    wb.log({"loss": 0.1})
    wb.finish()

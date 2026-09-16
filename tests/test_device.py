"""get_device() device-agnostic selection — logic tests, no GPU required.

Juan trains on MPS (M-series) locally and CUDA (RunPod) remotely; CI runs on
CPU. These assert the mps -> cuda -> cpu chain and the CUDA-only guard without
needing any accelerator present.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from deepEmulator.utils import device as dev  # noqa: E402


def test_get_device_str_returns_a_valid_backend():
    d = dev.get_device_str()
    assert d in {"mps", "cuda", "cpu"}


def test_get_device_returns_torch_device_and_is_usable():
    d = dev.get_device()
    assert isinstance(d, torch.device)
    # a trivial tensor must actually land on the selected device
    t = torch.zeros(2, 2).to(d)
    assert t.device.type == d.type


def test_preference_chain_mps_beats_cuda_beats_cpu(monkeypatch):
    monkeypatch.setattr(dev, "_mps_ok", lambda: True)
    monkeypatch.setattr(dev, "_cuda_ok", lambda: True)
    assert dev.get_device_str() == "mps"

    monkeypatch.setattr(dev, "_mps_ok", lambda: False)
    assert dev.get_device_str() == "cuda"

    monkeypatch.setattr(dev, "_cuda_ok", lambda: False)
    assert dev.get_device_str() == "cpu"


def test_prefer_override_honored_when_available_else_falls_back(monkeypatch):
    monkeypatch.setattr(dev, "_mps_ok", lambda: False)
    monkeypatch.setattr(dev, "_cuda_ok", lambda: True)
    # prefer cuda, cuda present -> cuda
    assert dev.get_device_str(prefer="cuda") == "cuda"
    # prefer mps, mps absent -> auto-select falls to cuda (never crashes)
    assert dev.get_device_str(prefer="mps") == "cuda"
    # cpu is always honorable
    assert dev.get_device_str(prefer="cpu") == "cpu"


def test_is_cuda_guard():
    assert dev.is_cuda("cuda") is True
    assert dev.is_cuda("cuda:0") is True
    assert dev.is_cuda("mps") is False
    assert dev.is_cuda("cpu") is False
    assert dev.is_cuda(torch.device("cpu")) is False


def test_ddqn_agent_uses_get_device_by_default(monkeypatch):
    """DDQNAgent must route through get_device_str (no hardcoded .cuda())."""
    from deepEmulator.agents.ddqn_torch import DDQNAgent

    monkeypatch.setattr(dev, "_mps_ok", lambda: False)
    monkeypatch.setattr(dev, "_cuda_ok", lambda: False)
    agent = DDQNAgent(obs_shape=(3, 72, 80), n_actions=7)  # no explicit device
    assert agent.device == "cpu"
    assert agent._amp_active is False  # AMP is CUDA-only


def test_ddqn_forward_pass_on_selected_device():
    """DDQN forward pass on random obs, on whatever get_device() picks."""
    import numpy as np

    from deepEmulator.agents.ddqn_torch import DDQNAgent

    d = dev.get_device_str()
    agent = DDQNAgent(obs_shape=(3, 72, 80), n_actions=7, device=d)
    obs = np.random.default_rng(0).integers(0, 256, (3, 72, 80)).astype(np.uint8)
    action = agent.act(obs, explore=False)
    assert 0 <= action < 7
    # explicit forward: batch of random obs -> (B, n_actions) on device
    x = torch.rand(4, 3, 72, 80).to(d)
    q = agent.net(x, mode="online")
    assert q.shape == (4, 7)
    assert q.device.type == torch.device(d).type

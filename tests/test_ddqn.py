"""DDQN agent unit tests — no PyBoy required."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_ddqn_forward_shape():
    from deepEmulator.agents.ddqn_torch import DDQNNet

    net = DDQNNet((3, 72, 80), 7)
    x = torch.zeros(2, 3, 72, 80)
    out = net(x, mode="online")
    assert out.shape == (2, 7)
    out_t = net(x, mode="target")
    assert out_t.shape == (2, 7)


def test_cache_stores_uint8_states_not_float32():
    """F9.7: Replay buffer must keep state tensors at their source dtype (typically uint8)
    so the buffer doesn't 4x-bloat in memory. Conversion to float happens in _recall."""
    from deepEmulator.agents.ddqn_torch import DDQNAgent, DDQNConfig

    cfg = DDQNConfig(burnin=2, batch_size=2, deque_size=10, learn_every=1, sync_every=1000)
    agent = DDQNAgent(obs_shape=(3, 72, 80), n_actions=7, config=cfg, device="cpu")
    obs = np.zeros((3, 72, 80), dtype=np.uint8)
    agent.cache(obs, obs, 0, 0.0, False)
    state_tensor = agent.memory[0][0]
    assert state_tensor.dtype == torch.uint8, (
        f"buffer state must be uint8 for memory efficiency, got {state_tensor.dtype}"
    )
    # And _recall casts to float for the gradient step
    for _ in range(3):
        agent.cache(obs, obs, 0, 0.0, False)
    s, ns, a, r, d = agent._recall()
    assert s.dtype == torch.float32
    assert ns.dtype == torch.float32


def test_ddqn_agent_act_and_cache():
    from deepEmulator.agents.ddqn_torch import DDQNAgent, DDQNConfig

    cfg = DDQNConfig(burnin=2, batch_size=4, deque_size=100, learn_every=1, sync_every=1000)
    agent = DDQNAgent(obs_shape=(3, 72, 80), n_actions=7, config=cfg, device="cpu")
    obs = np.zeros((3, 72, 80), dtype=np.uint8)
    for i in range(8):
        action = agent.act(obs)
        assert 0 <= action < 7
        agent.cache(obs, obs, action, 0.1 * i, False)
    q, loss = agent.learn()
    assert q is not None and loss is not None


def test_ddqn_state_dict_roundtrip(tmp_path):
    from deepEmulator.agents.ddqn_torch import DDQNAgent

    a1 = DDQNAgent(obs_shape=(3, 72, 80), n_actions=7, device="cpu")
    sd = a1.state_dict()
    torch.save(sd, tmp_path / "model.pt")

    a2 = DDQNAgent(obs_shape=(3, 72, 80), n_actions=7, device="cpu")
    a2.load_state_dict(torch.load(tmp_path / "model.pt", weights_only=False))
    obs = np.zeros((3, 72, 80), dtype=np.uint8)

    with torch.no_grad():
        x = torch.from_numpy(obs).float().unsqueeze(0)
        q1 = a1.net(x, mode="online")
        q2 = a2.net(x, mode="online")
    assert torch.allclose(q1, q2)

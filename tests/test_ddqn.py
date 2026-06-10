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


def test_linear_epsilon_anneal_is_pure_function_of_curr_step():
    from deepEmulator.agents.ddqn_torch import DDQNAgent, DDQNConfig

    cfg = DDQNConfig(exploration_anneal_steps=100, exploration_rate_min=0.05)
    agent = DDQNAgent(obs_shape=(3, 72, 80), n_actions=7, config=cfg, device="cpu")
    obs = np.zeros((3, 72, 80), dtype=np.uint8)
    for _ in range(50):
        agent.act(obs)
    assert agent.exploration_rate == pytest.approx(1.0 - 0.95 * 0.5)
    for _ in range(100):
        agent.act(obs)
    assert agent.exploration_rate == pytest.approx(0.05)


def test_epsilon_anneal_resume_correct():
    """Epsilon must be recomputed from curr_step — a restored agent picks up
    the right epsilon on its first act()."""
    from deepEmulator.agents.ddqn_torch import DDQNAgent, DDQNConfig

    cfg = DDQNConfig(exploration_anneal_steps=100)
    agent = DDQNAgent(obs_shape=(3, 72, 80), n_actions=7, config=cfg, device="cpu")
    agent.curr_step = 79  # as if restored from a checkpoint
    agent.act(np.zeros((3, 72, 80), dtype=np.uint8))
    assert agent.exploration_rate == pytest.approx(1.0 - 0.95 * 0.80)


def test_multiplicative_decay_fallback_when_anneal_unset():
    from deepEmulator.agents.ddqn_torch import DDQNAgent, DDQNConfig

    cfg = DDQNConfig(exploration_rate_decay=0.5, exploration_rate_min=0.01)
    agent = DDQNAgent(obs_shape=(3, 72, 80), n_actions=7, config=cfg, device="cpu")
    obs = np.zeros((3, 72, 80), dtype=np.uint8)
    agent.act(obs)
    assert agent.exploration_rate == pytest.approx(0.5)
    agent.act(obs)
    assert agent.exploration_rate == pytest.approx(0.25)


def test_act_eval_mode_does_not_mutate_training_state():
    from deepEmulator.agents.ddqn_torch import DDQNAgent

    agent = DDQNAgent(obs_shape=(3, 72, 80), n_actions=7, device="cpu")
    obs = np.zeros((3, 72, 80), dtype=np.uint8)
    eps_before, step_before = agent.exploration_rate, agent.curr_step
    for _ in range(5):
        a = agent.act(obs, explore=False)
        assert 0 <= a < 7
    assert agent.exploration_rate == eps_before
    assert agent.curr_step == step_before


def test_act_eval_mode_with_zero_epsilon_is_deterministic():
    from deepEmulator.agents.ddqn_torch import DDQNAgent

    agent = DDQNAgent(obs_shape=(3, 72, 80), n_actions=7, device="cpu")
    agent.eval_epsilon = 0.0
    obs = np.random.default_rng(0).integers(0, 256, (3, 72, 80)).astype(np.uint8)
    actions = {agent.act(obs, explore=False) for _ in range(10)}
    assert len(actions) == 1


def test_burnin_gates_on_buffer_fill_not_curr_step():
    """Resume restores curr_step with an empty buffer — learning must wait for
    the buffer to refill, not start on 32 correlated samples."""
    from deepEmulator.agents.ddqn_torch import DDQNAgent, DDQNConfig

    cfg = DDQNConfig(burnin=50, batch_size=4, learn_every=1, sync_every=10_000)
    agent = DDQNAgent(obs_shape=(3, 72, 80), n_actions=7, config=cfg, device="cpu")
    agent.curr_step = 99_999  # as if resumed deep into a run
    obs = np.zeros((3, 72, 80), dtype=np.uint8)
    for i in range(49):
        agent.cache(obs, obs, 0, 0.0, False)
        q, loss = agent.learn()
        assert q is None and loss is None, f"learned with only {i + 1} buffered transitions"
    agent.cache(obs, obs, 0, 0.0, False)
    agent.curr_step += 1  # keep learn_every cadence aligned
    q, loss = agent.learn()
    assert q is not None


def test_normalize_obs_divides_pixels_only():
    from deepEmulator.agents.ddqn_torch import DDQNAgent, DDQNConfig

    obs = np.full((3, 72, 80), 255, dtype=np.uint8)
    cfg_on = DDQNConfig(burnin=1, batch_size=2, learn_every=1, normalize_obs=True)
    a_on = DDQNAgent(obs_shape=(3, 72, 80), n_actions=7, config=cfg_on, device="cpu")
    for _ in range(2):
        a_on.cache(obs, obs, 0, 0.0, False)
    s, *_ = a_on._recall()
    assert float(s.max()) == pytest.approx(1.0)

    cfg_off = DDQNConfig(burnin=1, batch_size=2, learn_every=1, normalize_obs=False)
    a_off = DDQNAgent(obs_shape=(3, 72, 80), n_actions=7, config=cfg_off, device="cpu")
    for _ in range(2):
        a_off.cache(obs, obs, 0, 0.0, False)
    s, *_ = a_off._recall()
    assert float(s.max()) == pytest.approx(255.0)

    # rank-1 latents are never divided, even with the flag on
    lat = np.full((64,), 4.0, dtype=np.float32)
    cfg_lat = DDQNConfig(burnin=1, batch_size=2, learn_every=1, normalize_obs=True)
    a_lat = DDQNAgent(obs_shape=(64,), n_actions=7, config=cfg_lat, device="cpu")
    for _ in range(2):
        a_lat.cache(lat, lat, 0, 0.0, False)
    s, *_ = a_lat._recall()
    assert float(s.max()) == pytest.approx(4.0)


def test_default_gamma_is_099():
    from deepEmulator.agents.ddqn_torch import DDQNConfig

    assert DDQNConfig().gamma == pytest.approx(0.99)

"""Ring replay buffer — golden equivalence against real env semantics.

The buffer reconstructs frame-stack observations by index; the golden test
drives a stack-semantics synthetic env, records every observation the env
actually returned, and asserts the buffer reconstructs them byte-exactly —
including episode boundaries (clip-by-repetition) and ring wraparound.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from deepEmulator.agents.replay_buffer import ReplayBuffer  # noqa: E402


class _StackEnv:
    """Newest-first frame stack, reset repeats one frame — PyBoyEnv semantics."""

    def __init__(self, fs=3, h=4, w=5, seed=0):
        self.fs, self.h, self.w = fs, h, w
        self._rng = np.random.default_rng(seed)
        self._stack = np.zeros((fs, h, w), dtype=np.uint8)
        self._t = 0

    def _new(self):
        return self._rng.integers(0, 256, (self.h, self.w), dtype=np.uint8)

    def reset(self):
        self._stack[:] = self._new()
        self._t = 0
        return self._stack.copy()

    def step(self):
        self._stack[1:] = self._stack[:-1]
        self._stack[0] = self._new()
        self._t += 1
        return self._stack.copy()


def _drive(buffer: ReplayBuffer, env: _StackEnv, episodes, ep_len, rng):
    """Returns ordered list of (state, next_state, action, reward, terminated)."""
    log = []
    for ep in range(episodes):
        obs = env.reset()
        buffer.begin_episode(obs)
        for t in range(ep_len):
            action = int(rng.integers(0, 7))
            reward = float(rng.normal())
            next_obs = env.step()
            terminated = t == ep_len - 1 and ep % 2 == 0  # alternate term/trunc
            truncated = t == ep_len - 1 and not terminated
            buffer.add(next_obs, action, reward, terminated, truncated)
            log.append((obs, next_obs, action, reward, terminated))
            obs = next_obs
    return log


def test_golden_equivalence_one_step():
    rng = np.random.default_rng(1)
    env = _StackEnv()
    buf = ReplayBuffer(capacity=512, obs_shape=(3, 4, 5), n_step=1, gamma=0.9, seed=0)
    log = _drive(buf, env, episodes=3, ep_len=6, rng=rng)
    assert len(buf) == len(log)

    # every sampled transition must match SOME logged transition exactly
    keyed = {}
    for s, ns, a, r, term in log:
        keyed[(s.tobytes(), ns.tobytes())] = (a, r, term)
    s, ns, a, r, d, disc = buf.sample(64)
    for i in range(64):
        key = (s[i].tobytes(), ns[i].tobytes())
        assert key in keyed, f"reconstructed pair not in env history (i={i})"
        ea, er, eterm = keyed[key]
        assert a[i] == ea
        assert r[i] == pytest.approx(er, abs=1e-6)
        assert d[i] == (1.0 if eterm else 0.0)
        assert disc[i] == pytest.approx(0.9)


def test_golden_equivalence_under_wraparound():
    """Small capacity forces multiple full wraps; reconstructions stay exact."""
    rng = np.random.default_rng(2)
    env = _StackEnv()
    buf = ReplayBuffer(capacity=32, obs_shape=(3, 4, 5), n_step=1, gamma=0.9, seed=0)
    log = _drive(buf, env, episodes=12, ep_len=10, rng=rng)

    keyed = {(s.tobytes(), ns.tobytes()): (a, r) for s, ns, a, r, _ in log}
    s, ns, a, r, *_ = buf.sample(32)
    for i in range(32):
        key = (s[i].tobytes(), ns[i].tobytes())
        assert key in keyed
        ea, er = keyed[key]
        assert a[i] == ea and r[i] == pytest.approx(er, abs=1e-6)


def test_nstep_return_math_and_span():
    buf = ReplayBuffer(capacity=128, obs_shape=(4,), n_step=3, gamma=0.5, seed=0)
    obs = [np.full((4,), i, dtype=np.float32) for i in range(6)]
    buf.begin_episode(obs[0])
    buf.add(obs[1], 0, 1.0, False)
    buf.add(obs[2], 1, 2.0, False)
    assert len(buf) == 0  # window not mature
    buf.add(obs[3], 0, 3.0, False)
    assert len(buf) == 1
    s, ns, a, r, d, disc = buf.sample(1)
    # R = 1 + 0.5*2 + 0.25*3 = 2.75 over the window starting at obs[0]
    assert np.allclose(s[0], obs[0])
    assert np.allclose(ns[0], obs[3])
    assert int(a[0]) == 0
    assert float(r[0]) == pytest.approx(2.75)
    assert float(disc[0]) == pytest.approx(0.125)
    assert float(d[0]) == 0.0


def test_nstep_flush_on_terminated_vs_truncated():
    obs = np.zeros((4,), dtype=np.float32)

    buf = ReplayBuffer(capacity=128, obs_shape=(4,), n_step=3, gamma=0.5, seed=0)
    buf.begin_episode(obs)
    buf.add(obs, 0, 1.0, False)
    buf.add(obs, 0, 1.0, True)  # terminal on step 2 -> both partials flush, d=1
    assert len(buf) == 2
    slots = np.flatnonzero(buf.complete)
    assert set(np.round(buf.discount[slots], 4)) == {0.25, 0.5}
    assert all(buf.done[slots] == 1.0)

    buf2 = ReplayBuffer(capacity=128, obs_shape=(4,), n_step=3, gamma=0.5, seed=0)
    buf2.begin_episode(obs)
    buf2.add(obs, 0, 1.0, False)
    buf2.add(obs, 0, 1.0, False, True)  # truncated -> flush but d=0 (bootstrap lives)
    assert len(buf2) == 2
    slots2 = np.flatnonzero(buf2.complete)
    assert all(buf2.done[slots2] == 0.0)
    assert set(np.round(buf2.discount[slots2], 4)) == {0.25, 0.5}


def test_frame_mode_stores_uint8_once_per_step():
    """F9.7 successor: one (H, W) uint8 frame per step, not two full stacks."""
    buf = ReplayBuffer(capacity=100, obs_shape=(3, 72, 80))
    assert buf.frames.dtype == np.uint8
    assert buf.frames.shape[1:] == (72, 80)  # no stack dimension in storage
    s, ns, *_ = (None,) * 6  # noqa: F841
    obs = np.random.default_rng(0).integers(0, 256, (3, 72, 80), dtype=np.uint8)
    buf.begin_episode(obs)
    buf.add(obs, 0, 0.0, False)


def test_sample_torch_returns_float_without_normalizing():
    buf = ReplayBuffer(capacity=64, obs_shape=(3, 4, 5), n_step=1)
    obs = np.full((3, 4, 5), 255, dtype=np.uint8)
    buf.begin_episode(obs)
    for _ in range(8):
        buf.add(obs, 0, 0.0, False)
    s, ns, a, r, d, disc = buf.sample_torch(4, "cpu")
    assert s.dtype == torch.float32 and ns.dtype == torch.float32
    assert float(s.max()) == 255.0  # /255 belongs to the agent's _prep
    assert a.dtype == torch.int64


def test_flat_mode_roundtrip():
    buf = ReplayBuffer(capacity=64, obs_shape=(16,), n_step=1, gamma=0.99)
    rng = np.random.default_rng(0)
    vecs = [rng.normal(size=16).astype(np.float32) for _ in range(6)]
    buf.begin_episode(vecs[0])
    for i in range(5):
        buf.add(vecs[i + 1], i % 3, float(i), False)
    s, ns, a, r, *_ = buf.sample(16)
    for i in range(16):
        j = int(r[i])  # reward encodes the step index
        assert np.allclose(s[i], vecs[j])
        assert np.allclose(ns[i], vecs[j + 1])
        assert int(a[i]) == j % 3

"""Frame corpus tests — uses a fake env to avoid PyBoy/ALE deps."""
from __future__ import annotations

import numpy as np
import pytest

from deepEmulator.core.spaces import Box, Discrete


# --- fake env (mimics our EmulatorEnv contract) -----------------------------
class FakeEnv:
    def __init__(self, h=144, w=160, n_actions=7, seed=0):
        self.h, self.w, self.n_actions = h, w, n_actions
        self.observation_space = Box(0, 255, (3, h // 2, w // 2))
        self.action_space = Discrete(n_actions)
        self._rng = np.random.default_rng(seed)
        self._t = 0

    def reset(self, *, seed=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        return np.zeros(self.observation_space.shape, dtype=np.uint8), {}

    def step(self, action):
        self._t += 1
        obs = self._rng.integers(0, 256, size=self.observation_space.shape, dtype=np.uint8)
        done = self._t >= 50
        return obs, 0.0, False, done, {}

    def render(self):
        return self._rng.integers(0, 256, size=(self.h, self.w, 3), dtype=np.uint8)


def test_normalize_grayscale_input():
    from deepEmulator.data.frame_corpus import normalize_to_96x96

    out = normalize_to_96x96(np.zeros((144, 160), dtype=np.uint8))
    assert out.shape == (1, 96, 96)
    assert out.dtype == np.uint8


def test_normalize_rgb_input():
    from deepEmulator.data.frame_corpus import normalize_to_96x96

    out = normalize_to_96x96(np.full((210, 160, 3), 255, dtype=np.uint8))
    assert out.shape == (1, 96, 96)
    assert out[0, 50, 50] == 255  # white frame stays white


def test_frame_ring_basic():
    from deepEmulator.data.frame_corpus import FrameRing, normalize_to_96x96

    ring = FrameRing(capacity=100, rng=0)
    for _ in range(150):
        ring.push(normalize_to_96x96(np.zeros((144, 160), dtype=np.uint8)))
    assert len(ring) == 100

    batch = ring.sample(8)
    assert batch.shape == (8, 1, 96, 96)


def test_frame_storage_chunks(tmp_path):
    from deepEmulator.data.frame_corpus import FrameStorage, normalize_to_96x96

    store = FrameStorage(tmp_path / "corpus", chunk_size=64)
    for _ in range(140):
        store.push(normalize_to_96x96(np.zeros((144, 160), dtype=np.uint8)))
    store.flush()
    assert store.total_frames == 140
    assert len(list((tmp_path / "corpus").glob("chunk_*.npz"))) == 3  # 64+64+12

    batches = list(store.iter_batches(batch_size=32, seed=0))
    assert all(b.shape == (32, 1, 96, 96) for b in batches)


def test_frame_storage_resumes_index(tmp_path):
    from deepEmulator.data.frame_corpus import FrameStorage, normalize_to_96x96

    root = tmp_path / "corpus"
    s1 = FrameStorage(root, chunk_size=50)
    for _ in range(120):
        s1.push(normalize_to_96x96(np.zeros((144, 160), dtype=np.uint8)))
    s1.flush()

    s2 = FrameStorage(root, chunk_size=50)
    assert s2.total_frames == 120  # picks up from index.json


def test_collector_round_robin_pushes_to_ring():
    from deepEmulator.data.frame_corpus import FrameCollector, FrameRing

    envs = [FakeEnv(144, 160), FakeEnv(210, 160), FakeEnv(320, 224)]
    ring = FrameRing(capacity=500)
    coll = FrameCollector(envs, ring, seed=0)
    coll.run(60)
    assert len(ring) == 60
    batch = ring.sample(16)
    assert batch.shape == (16, 1, 96, 96)


def test_collector_handles_episode_done():
    from deepEmulator.data.frame_corpus import FrameCollector, FrameRing

    envs = [FakeEnv(144, 160, seed=0)]
    ring = FrameRing(capacity=200)
    coll = FrameCollector(envs, ring, seed=0)
    # FakeEnv resets at t=50; run past it
    coll.run(120)
    assert len(ring) == 120

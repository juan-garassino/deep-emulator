"""Tests for deepemu-collect-frames CLI."""
from __future__ import annotations

import sys

import numpy as np
import pytest

from deepEmulator.core.spaces import Box, Discrete


class _FakeEnv:
    """Minimal env mimicking EmulatorEnv contract for collector tests."""

    def __init__(self, h=144, w=160, seed=0, **_kw):
        self.observation_space = Box(0, 255, (3, h // 2, w // 2))
        self.action_space = Discrete(7)
        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._h, self._w = h, w

    def reset(self, *, seed=None):
        self._t = 0
        return np.zeros(self.observation_space.shape, dtype=np.uint8), {}

    def step(self, action):
        self._t += 1
        return (
            self._rng.integers(0, 256, self.observation_space.shape, dtype=np.uint8),
            0.0,
            False,
            self._t >= 100,
            {},
        )

    def render(self):
        return self._rng.integers(0, 256, (self._h, self._w, 3), dtype=np.uint8)

    def close(self):
        pass


def test_module_imports_and_parses_args():
    from deepEmulator.training import collect_frames as cf

    args = cf._parse_args([
        "--cartridges", "POKEMON RED",
        "--rom", "roms/PokemonRed.gb",
        "--frames", "100",
        "--out", "/tmp/x",
    ])
    assert args.cartridges == ["POKEMON RED"]
    assert args.frames == 100


def test_multi_cartridge_arg_parsing():
    from deepEmulator.training import collect_frames as cf

    args = cf._parse_args([
        "--cartridges", "POKEMON RED", "ATARI PONG",
        "--rom", "a.gb", "b.bin",
        "--frames", "10",
        "--out", "/tmp/x",
    ])
    assert args.cartridges == ["POKEMON RED", "ATARI PONG"]
    assert [str(r) for r in args.rom] == ["a.gb", "b.bin"]


def test_end_to_end_cli_with_fake_envs(tmp_path, monkeypatch):
    """Run the CLI with `_build_env` patched to yield FakeEnvs."""
    from deepEmulator.training import collect_frames as cf

    monkeypatch.setattr(cf, "_build_env", lambda cart, rom, init_state: _FakeEnv())

    out = tmp_path / "corpus"
    sys.argv = [
        "deepemu-collect-frames",
        "--cartridges", "POKEMON RED",
        "--rom", "roms/fake.gb",
        "--frames", "200",
        "--out", str(out),
        "--chunk-size", "64",
    ]
    rc = cf.main()
    assert rc == 0
    assert out.exists()
    assert (out / "index.json").exists()
    # 200 frames at chunk_size 64 → 4 chunks (3 full + 1 partial flushed at end)
    chunks = sorted(out.glob("chunk_*.npz"))
    assert len(chunks) >= 3
    # Load a chunk and verify shape
    first = np.load(chunks[0])["frames"]
    assert first.shape[1:] == (1, 96, 96)
    assert first.dtype == np.uint8


def test_multi_cartridge_round_robin(tmp_path, monkeypatch):
    from deepEmulator.training import collect_frames as cf

    monkeypatch.setattr(cf, "_build_env", lambda cart, rom, init_state: _FakeEnv())

    out = tmp_path / "multi"
    sys.argv = [
        "deepemu-collect-frames",
        "--cartridges", "POKEMON RED", "ATARI PONG",
        "--rom", "a.gb", "b.bin",
        "--frames", "60",
        "--out", str(out),
        "--chunk-size", "32",
    ]
    rc = cf.main()
    assert rc == 0
    import json

    idx = json.loads((out / "index.json").read_text())
    assert idx["total_frames"] == 60
    assert idx["shape"] == [1, 96, 96]

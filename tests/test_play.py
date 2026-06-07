"""Tests for `deepemu-play` — exercises bundle loading + env factory + the
human-takeover toggle without opening a real SDL window."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from deepEmulator.core.spaces import Box, Discrete  # noqa: E402


class _FakeInfoEnv:
    """Mimics EmulatorEnv with the trajectory info contract."""

    def __init__(self, episode_length=10):
        self.observation_space = Box(0, 255, (3, 72, 80))
        self.action_space = Discrete(7)
        self._t = 0
        self._eplen = episode_length

        class _Cart:
            cartridge_title = "FAKE"
            platform = "gameboy"
        self.cartridge = _Cart()

    def reset(self, *, seed=None):
        self._t = 0
        return np.zeros(self.observation_space.shape, dtype=np.uint8), {}

    def step(self, action):
        self._t += 1
        obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
        return (
            obs,
            0.1 * self._t,
            False,
            self._t >= self._eplen,
            {"game_state": {}, "trajectory": (self._t, self._t * 2, 40)},
        )

    def render(self):
        return np.zeros((144, 160, 3), dtype=np.uint8)

    def close(self):
        pass


def _make_pixel_bundle(tmp_path: Path) -> Path:
    from deepEmulator.agents.ddqn_torch import DDQNAgent
    from deepEmulator.utils.checkpoints import write_bundle

    agent = DDQNAgent(obs_shape=(3, 72, 80), n_actions=7, device="cpu")
    return write_bundle(
        tmp_path / "run0",
        agent_state=agent.state_dict(),
        cartridge_title="POKEMON RED",
        cartridge_platform="gameboy",
        action_set=list(range(7)),
        obs_shape=(3, 72, 80),
        algo="ddqn",
    )


# --- CLI arg parsing -----------------------------------------------------------
def test_play_arg_parsing():
    from deepEmulator.inference import play

    args = play._parse_args([
        "--cartridge", "POKEMON RED",
        "--rom", "roms/x.gb",
        "--ckpt", "/tmp/bundle",
        "--visible", "--arrows", "--attention",
        "--episodes", "1",
    ])
    assert args.cartridge == "POKEMON RED"
    assert args.visible and args.arrows and args.attention
    assert args.episodes == 1


# --- agent_enabled toggle ------------------------------------------------------
def test_read_agent_enabled_defaults_true_when_no_file(tmp_path):
    from deepEmulator.inference.play import _read_agent_enabled, _toggle_path

    t = _toggle_path(tmp_path)
    assert _read_agent_enabled(t) is True


def test_read_agent_enabled_false_when_zero(tmp_path):
    from deepEmulator.inference.play import _read_agent_enabled, _toggle_path

    t = _toggle_path(tmp_path)
    t.write_text("0")
    assert _read_agent_enabled(t) is False
    t.write_text("1")
    assert _read_agent_enabled(t) is True
    t.write_text("")
    assert _read_agent_enabled(t) is False


# --- end-to-end via injected env_factory --------------------------------------
def test_run_play_pixel_bundle_no_overlays(tmp_path):
    from deepEmulator.inference.play import run_play

    bundle = _make_pixel_bundle(tmp_path)
    rc = run_play(
        cartridge="POKEMON RED",
        rom=tmp_path / "fake.gb",
        init_state=None,
        ckpt=bundle,
        visible=False,
        arrows=False,
        attention=False,
        episodes=2,
        max_episode_steps=5,
        epsilon=0.0,
        env_factory=lambda: _FakeInfoEnv(episode_length=5),
    )
    assert rc == 0


def test_run_play_with_encoder_bundle_no_overlays(tmp_path):
    """If the bundle has an encoder linkage, that path is exercised
    (we still inject a fake env so PyBoy isn't needed)."""
    import json

    from deepEmulator.agents.ddqn_torch import DDQNAgent
    from deepEmulator.encoders.vit import ViTConfig, ViTTiny
    from deepEmulator.inference.play import run_play
    from deepEmulator.utils.checkpoints import write_bundle

    # Build a tiny encoder bundle on disk
    cfg = ViTConfig(image_size=96, patch_size=8, embed_dim=96, depth=2, num_heads=3, out_dim=64)
    enc = ViTTiny(cfg)
    enc_dir = tmp_path / "dino_run"
    enc_dir.mkdir()
    torch.save(enc.state_dict(), enc_dir / "encoder_only.pt")
    (enc_dir / "metadata.json").write_text(
        json.dumps({"algo": "dino", "vit_config": cfg.__dict__})
    )

    # Build a DDQN bundle whose observation_shape is the LATENT shape
    agent = DDQNAgent(obs_shape=(192,), n_actions=7, device="cpu")
    bundle = write_bundle(
        tmp_path / "run1",
        agent_state=agent.state_dict(),
        cartridge_title="POKEMON RED",
        cartridge_platform="gameboy",
        action_set=list(range(7)),
        obs_shape=(192,),
        algo="ddqn",
        extra={"encoder": {
            "path": str(enc_dir.resolve()),
            "frozen": True,
            "latent_dim": 64,
            "preprocessing": "96x96_grayscale",
        }},
    )

    # The injected fake env emits rank-3 obs. Agent expects rank-1 because
    # of the latent-shaped bundle. So we wrap the fake env with the real
    # FrozenEncoderEnv path that play.run_play would build — to keep the
    # test surface honest, exercise that via env_factory ourselves.
    from deepEmulator.encoders.frozen_wrapper import FrozenEncoderEnv, load_frozen_encoder

    encoder, _ = load_frozen_encoder(enc_dir)
    rc = run_play(
        cartridge="POKEMON RED",
        rom=tmp_path / "fake.gb",
        init_state=None,
        ckpt=bundle,
        visible=False,
        arrows=False,
        attention=False,
        episodes=1,
        max_episode_steps=5,
        epsilon=0.0,
        env_factory=lambda: FrozenEncoderEnv(_FakeInfoEnv(episode_length=5), encoder, device="cpu"),
    )
    assert rc == 0


def test_run_play_respects_agent_enabled_toggle(tmp_path, monkeypatch):
    """When agent_enabled.txt contains '0', the agent's act() is bypassed
    (action 0 = NOOP is sent every step)."""
    from deepEmulator.inference import play as play_mod

    bundle = _make_pixel_bundle(tmp_path)
    (bundle / "agent_enabled.txt").write_text("0")

    seen_actions: list[int] = []

    class _SpyEnv(_FakeInfoEnv):
        def step(self, action):
            seen_actions.append(action)
            return super().step(action)

    rc = play_mod.run_play(
        cartridge="POKEMON RED",
        rom=tmp_path / "fake.gb",
        init_state=None,
        ckpt=bundle,
        visible=False,
        arrows=False,
        attention=False,
        episodes=1,
        max_episode_steps=4,
        epsilon=0.0,
        env_factory=lambda: _SpyEnv(episode_length=4),
    )
    assert rc == 0
    # Every step should have been action=0 (NOOP) since the toggle is off
    assert all(a == 0 for a in seen_actions), seen_actions


def test_main_dispatches_run_play(monkeypatch, tmp_path):
    """The CLI entrypoint must call run_play. Inject a fake to assert."""
    from deepEmulator.inference import play as play_mod

    bundle = _make_pixel_bundle(tmp_path)
    called = {}

    def _fake_run_play(**kw):
        called.update(kw)
        return 0

    monkeypatch.setattr(play_mod, "run_play", _fake_run_play)
    sys.argv = [
        "deepemu-play",
        "--cartridge", "POKEMON RED",
        "--rom", "roms/fake.gb",
        "--ckpt", str(bundle),
        "--visible", "--arrows",
        "--episodes", "2",
    ]
    rc = play_mod.main()
    assert rc == 0
    assert called["cartridge"] == "POKEMON RED"
    assert called["visible"] is True
    assert called["arrows"] is True
    assert called["attention"] is False
    assert called["episodes"] == 2

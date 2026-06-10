"""Tests for the frozen-encoder track: rank-1 DDQN + FrozenEncoderEnv + bundle linkage."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")


# --- DDQN dispatching on obs rank -------------------------------------------
def test_ddqn_builds_mlp_for_rank1_obs():
    from deepEmulator.agents.ddqn_torch import DDQNNet

    net = DDQNNet((768,), 7)
    assert net.mode == "mlp"
    x = torch.zeros(2, 768)
    out = net(x, mode="online")
    assert out.shape == (2, 7)
    out_t = net(x, mode="target")
    assert out_t.shape == (2, 7)


def test_ddqn_still_builds_cnn_for_rank3_obs():
    from deepEmulator.agents.ddqn_torch import DDQNNet

    net = DDQNNet((3, 72, 80), 7)
    assert net.mode == "cnn"
    x = torch.zeros(2, 3, 72, 80)
    out = net(x, mode="online")
    assert out.shape == (2, 7)


def test_ddqn_agent_act_with_rank1_obs():
    from deepEmulator.agents.ddqn_torch import DDQNAgent, DDQNConfig

    cfg = DDQNConfig(burnin=2, batch_size=4, deque_size=100, learn_every=1, sync_every=1000)
    agent = DDQNAgent(obs_shape=(256,), n_actions=6, config=cfg, device="cpu")
    obs = np.zeros((256,), dtype=np.float32)
    for i in range(8):
        a = agent.act(obs)
        assert 0 <= a < 6
        agent.cache(obs, obs, a, 0.1 * i, False)
    q, loss = agent.learn()
    assert q is not None and loss is not None


def test_ddqn_rejects_unsupported_obs_rank():
    from deepEmulator.agents.ddqn_torch import DDQNNet

    with pytest.raises(ValueError):
        DDQNNet((4, 4, 4, 4), 7)


# --- FrozenEncoderEnv -------------------------------------------------------
class _FakeInnerEnv:
    """Mimics an EmulatorEnv with rank-3 (C, H, W) frame-stacked obs."""

    def __init__(self, frame_stack=3, h=72, w=80, n_actions=7):
        from deepEmulator.core.spaces import Box, Discrete

        class _Cart:
            cartridge_title = "FAKE"
            platform = "gameboy"

        self.cartridge = _Cart()
        self.observation_space = Box(0, 255, (frame_stack, h, w))
        self.action_space = Discrete(n_actions)
        self._rng = np.random.default_rng(0)
        self._shape = (frame_stack, h, w)
        self._t = 0

    def _obs(self):
        return self._rng.integers(0, 256, size=self._shape, dtype=np.uint8)

    def reset(self, *, seed=None):
        self._t = 0
        return self._obs(), {"game_state": {}}

    def step(self, action):
        self._t += 1
        return self._obs(), 0.1, False, self._t >= 5, {"game_state": {}, "trajectory": (0, 0, 0)}

    def render(self):
        return np.zeros((144, 160, 3), dtype=np.uint8)

    def close(self):
        pass


def test_frozen_encoder_env_emits_latent_obs():
    from deepEmulator.encoders.frozen_wrapper import FrozenEncoderEnv
    from deepEmulator.encoders.vit import ViTConfig, ViTTiny

    enc = ViTTiny(ViTConfig(image_size=96, patch_size=8, embed_dim=96, depth=2, num_heads=3, out_dim=64))
    inner = _FakeInnerEnv(frame_stack=3)
    env = FrozenEncoderEnv(inner, enc, device="cpu")

    assert env.observation_space.shape == (3 * 64,)
    obs, info = env.reset()
    assert obs.shape == (192,)
    assert obs.dtype == np.float32

    obs, r, term, trunc, info = env.step(0)
    assert obs.shape == (192,)
    assert isinstance(r, float)


def test_load_frozen_encoder_roundtrip(tmp_path):
    """Pretend we ran the DINO CLI: write encoder_only.pt + metadata.json, then load."""
    from deepEmulator.encoders.frozen_wrapper import load_frozen_encoder
    from deepEmulator.encoders.vit import ViTConfig, ViTTiny

    cfg = ViTConfig(image_size=96, patch_size=8, embed_dim=96, depth=2, num_heads=3, out_dim=64)
    original = ViTTiny(cfg)
    enc_dir = tmp_path / "dino_run"
    enc_dir.mkdir()
    torch.save(original.state_dict(), enc_dir / "encoder_only.pt")
    (enc_dir / "metadata.json").write_text(
        json.dumps({"algo": "dino", "vit_config": cfg.__dict__})
    )

    loaded, md = load_frozen_encoder(enc_dir, map_location="cpu")
    assert md["algo"] == "dino"
    x = torch.zeros(1, 1, 96, 96)
    with torch.no_grad():
        a = original(x)
        b = loaded(x)
    assert torch.allclose(a, b)


def test_load_frozen_encoder_legacy_bundle_without_algo_key(tmp_path):
    """Bundles written before the `algo` field existed must still load as DINO."""
    from deepEmulator.encoders.frozen_wrapper import load_frozen_encoder
    from deepEmulator.encoders.vit import ViTConfig, ViTTiny

    cfg = ViTConfig(image_size=96, patch_size=8, embed_dim=96, depth=2, num_heads=3, out_dim=64)
    original = ViTTiny(cfg)
    enc_dir = tmp_path / "legacy_run"
    enc_dir.mkdir()
    torch.save(original.state_dict(), enc_dir / "encoder_only.pt")
    (enc_dir / "metadata.json").write_text(
        json.dumps({"vit_config": cfg.__dict__})  # NOTE: no `algo` key
    )

    loaded, md = load_frozen_encoder(enc_dir, map_location="cpu")
    assert "algo" not in md
    x = torch.zeros(1, 1, 96, 96)
    with torch.no_grad():
        assert torch.allclose(original(x), loaded(x))


# --- end-to-end: train CLI with --encoder ----------------------------------
def test_train_cli_passes_encoder_through_to_bundle_metadata(tmp_path, monkeypatch):
    """Exercise the --encoder flag through `train.main` using fake env + tiny encoder."""
    import sys

    from deepEmulator.cartridges import pokemon_red  # registers POKEMON RED  # noqa: F401
    from deepEmulator.encoders.vit import ViTConfig, ViTTiny
    from deepEmulator.training import train as train_mod
    from deepEmulator.utils import checkpoints

    # Build a tiny encoder bundle on disk
    cfg = ViTConfig(image_size=96, patch_size=8, embed_dim=96, depth=2, num_heads=3, out_dim=64)
    encoder = ViTTiny(cfg)
    enc_dir = tmp_path / "dino_run"
    enc_dir.mkdir()
    torch.save(encoder.state_dict(), enc_dir / "encoder_only.pt")
    (enc_dir / "metadata.json").write_text(
        json.dumps({"algo": "dino", "vit_config": cfg.__dict__})
    )

    # Replace PyBoyEnv with the fake inner env in train_mod's namespace
    from deepEmulator.training import train as tm

    orig_pyboy_env = tm.__dict__.get("PyBoyEnv")

    def fake_pyboy_env(adapter, *, rom_path, init_state, headless, max_steps):
        return _FakeInnerEnv(frame_stack=3, h=72, w=80, n_actions=7)

    # The import happens inside main(); monkey-patch by intercepting the symbol after import.
    # Easiest: patch the platforms module.
    import deepEmulator.platforms.gameboy as gb_mod

    real_PyBoyEnv = gb_mod.PyBoyEnv
    gb_mod.PyBoyEnv = lambda *a, **kw: _FakeInnerEnv(frame_stack=3, h=72, w=80, n_actions=7)

    try:
        run_dir = tmp_path / "checkpoints" / "pokemon_red" / "run0"
        monkeypatch.setattr(sys, "argv", [
            "deepemu-train",
            "--cartridge", "POKEMON RED",
            "--rom", str(tmp_path / "fake.gb"),
            "--steps", "5",
            "--max-episode-steps", "5",
            "--save-every", "100",
            "--headless",
            "--run-dir", str(run_dir),
            "--encoder", str(enc_dir),
        ])
        rc = train_mod.main()
        assert rc == 0

        md = json.loads((run_dir / "metadata.json").read_text())
        assert "encoder" in md, md
        assert md["encoder"]["frozen"] is True
        assert md["encoder"]["latent_dim"] == 64
        # the encoder bundle in this test records no preprocessing (legacy);
        # the field now propagates from the ENCODER metadata, not a constant
        assert md["encoder"]["preprocessing"] is None
        # bundle-relative portable copy of the encoder
        assert md["encoder"]["path"] == "encoder"
        assert (run_dir / "encoder" / "encoder_only.pt").exists()
        assert len(md["encoder"]["sha256"]) == 64
        # observation_shape should be 1-D since FrozenEncoderEnv was applied
        assert len(md["observation_shape"]) == 1
        assert md["observation_shape"][0] == 3 * 64
        # metadata contract: env + network + seed blocks (play/eval rebuild from these)
        assert "env" in md and "network" in md and "seed" in md
        assert md["network"]["dueling"] is True  # train.py default for new runs
        assert md["network"]["normalize_obs"] is False  # latent obs are never normalized
        assert md["network"]["n_step"] == 3
        assert md["env"]["max_episode_steps"] == 5
    finally:
        gb_mod.PyBoyEnv = real_PyBoyEnv


def test_load_frozen_encoder_rejects_mismatched_preprocessing(tmp_path):
    """An encoder pretrained on a render-path corpus must refuse to load for
    obs-path RL — silent distribution shift was the encoder-track killer."""
    from deepEmulator.encoders.frozen_wrapper import load_frozen_encoder
    from deepEmulator.encoders.vit import ViTConfig, ViTTiny

    cfg = ViTConfig(image_size=96, patch_size=8, embed_dim=96, depth=2, num_heads=3, out_dim=64)
    enc_dir = tmp_path / "render_run"
    enc_dir.mkdir()
    torch.save(ViTTiny(cfg).state_dict(), enc_dir / "encoder_only.pt")
    (enc_dir / "metadata.json").write_text(
        json.dumps({"algo": "dino", "vit_config": cfg.__dict__,
                    "preprocessing": "render_rec601_to_96x96"})
    )
    with pytest.raises(ValueError, match="preprocessing"):
        load_frozen_encoder(enc_dir)


def test_load_frozen_encoder_accepts_matching_or_missing_preprocessing(tmp_path):
    from deepEmulator.data.frame_corpus import OBS_PREPROCESSING
    from deepEmulator.encoders.frozen_wrapper import load_frozen_encoder
    from deepEmulator.encoders.vit import ViTConfig, ViTTiny

    cfg = ViTConfig(image_size=96, patch_size=8, embed_dim=96, depth=2, num_heads=3, out_dim=64)
    for preprocessing in (OBS_PREPROCESSING, None):
        enc_dir = tmp_path / f"run_{preprocessing}"
        enc_dir.mkdir()
        torch.save(ViTTiny(cfg).state_dict(), enc_dir / "encoder_only.pt")
        md = {"algo": "dino", "vit_config": cfg.__dict__}
        if preprocessing:
            md["preprocessing"] = preprocessing
        (enc_dir / "metadata.json").write_text(json.dumps(md))
        model, _ = load_frozen_encoder(enc_dir)
        assert model is not None

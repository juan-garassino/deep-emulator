"""Eval harness — tests via a fake env factory + minimal bundle."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")


# --- fake env for eval --------------------------------------------------------
class _FakeEvalEnv:
    """A minimal env that yields deterministic per-episode reward = ep_index."""

    def __init__(self, obs_shape=(3, 72, 80), n_actions=7, episode_length=10, ep_reward=1.0):
        from deepEmulator.core.spaces import Box, Discrete

        class _Cart:
            cartridge_title = "FAKE"
            platform = "gameboy"

        self.cartridge = _Cart()
        self.observation_space = Box(0, 255, obs_shape)
        self.action_space = Discrete(n_actions)
        self._obs_shape = obs_shape
        self._episode_length = episode_length
        self._ep_reward = ep_reward
        self._t = 0
        self._ep = -1

    def reset(self, *, seed=None):
        self._t = 0
        self._ep += 1
        return np.zeros(self._obs_shape, dtype=np.uint8), {}

    def step(self, action):
        self._t += 1
        done = self._t >= self._episode_length
        return (
            np.zeros(self._obs_shape, dtype=np.uint8),
            self._ep_reward if done else 0.0,
            done,
            False,
            {},
        )

    def render(self):
        return np.full((144, 160, 3), self._t % 256, dtype=np.uint8)

    def close(self):
        pass


def _make_fake_bundle(tmp_path: Path, *, obs_shape=(3, 72, 80), n_actions=7) -> Path:
    """Write a bundle on disk that `load_bundle` can read."""
    from deepEmulator.agents.ddqn_torch import DDQNAgent
    from deepEmulator.utils.checkpoints import write_bundle

    agent = DDQNAgent(obs_shape=obs_shape, n_actions=n_actions, device="cpu")
    bundle = write_bundle(
        tmp_path / "bundle",
        agent_state=agent.state_dict(),
        cartridge_title="POKEMON RED",
        cartridge_platform="gameboy",
        action_set=list(range(n_actions)),
        obs_shape=obs_shape,
        algo="ddqn",
        extra={"global_steps": 1000},
    )
    # Drop a metrics.tsv so the report curve has data
    (bundle / "metrics.tsv").write_text(
        "Episode\tStep\tEpsilon\tMeanReward\tMeanLength\tMeanLoss\tMeanQValue\tTimeDelta\tTime\n"
        + "\n".join(f"{i}\t{i*100}\t0.5\t{0.1*i}\t100\t0.5\t0.3\t1.0\t{i*1.0}" for i in range(10))
    )
    return bundle


def test_evaluate_bundle_writes_tsv_and_returns_summary(tmp_path):
    from deepEmulator.training.eval import evaluate_bundle

    bundle = _make_fake_bundle(tmp_path)

    def factory(_encoder_path):
        return _FakeEvalEnv(episode_length=5, ep_reward=2.0)

    summary = evaluate_bundle(bundle, env_factory=factory, n_episodes=4, epsilon=0.0)
    assert summary["n_episodes"] == 4
    assert summary["reward_mean"] == pytest.approx(2.0)  # deterministic
    assert summary["length_mean"] == pytest.approx(5.0)
    assert (bundle / "eval_episodes.tsv").exists()

    text = (bundle / "eval_episodes.tsv").read_text().splitlines()
    assert text[0].startswith("episode\treward\tlength")
    assert len(text) == 5  # header + 4 episodes


def test_write_report_single_bundle(tmp_path):
    from deepEmulator.training.eval import write_report

    summary = {
        "bundle_dir": str(tmp_path / "b1"),
        "cartridge": "POKEMON RED",
        "algo": "ddqn",
        "encoder_path": None,
        "global_steps": 50000,
        "n_episodes": 20,
        "reward_mean": 1.5,
        "reward_std": 0.3,
        "reward_min": 1.0,
        "reward_max": 2.0,
        "length_mean": 800.0,
    }
    out = write_report([summary], [tmp_path / "b1"], tmp_path / "report.html")
    assert out.exists()
    html = out.read_text()
    assert "POKEMON RED" in html
    assert "1.500" in html or "1.500</b>" in html


def test_write_report_two_bundles_for_comparison(tmp_path):
    from deepEmulator.training.eval import write_report

    summaries = [
        {
            "bundle_dir": str(tmp_path / "pixel"),
            "cartridge": "POKEMON RED",
            "algo": "ddqn",
            "encoder_path": None,
            "global_steps": 100_000,
            "n_episodes": 20,
            "reward_mean": 1.5, "reward_std": 0.3,
            "reward_min": 1.0, "reward_max": 2.0,
            "length_mean": 800,
        },
        {
            "bundle_dir": str(tmp_path / "dino"),
            "cartridge": "POKEMON RED",
            "algo": "ddqn",
            "encoder_path": "/path/to/dino/encoder",
            "global_steps": 100_000,
            "n_episodes": 20,
            "reward_mean": 2.1, "reward_std": 0.4,
            "reward_min": 1.5, "reward_max": 3.0,
            "length_mean": 900,
        },
    ]
    out = write_report(summaries, [tmp_path / "pixel", tmp_path / "dino"], tmp_path / "report.html")
    html = out.read_text()
    assert "baseline" in html and "treatment" in html
    assert "/path/to/dino/encoder" in html
    # baseline shows "(pixel CNN)" label since encoder_path is None
    assert "pixel CNN" in html


def test_evaluate_bundle_captures_frames_when_render_works(tmp_path):
    from deepEmulator.training.eval import evaluate_bundle

    bundle = _make_fake_bundle(tmp_path)
    factory = lambda _e: _FakeEvalEnv(episode_length=6, ep_reward=1.0)
    _ = evaluate_bundle(bundle, env_factory=factory, n_episodes=2, capture_frames_per_episode=3)
    assert (bundle / "eval_frames.npz").exists()
    arr = np.load(bundle / "eval_frames.npz")["frames"]
    # 2 episodes × min(3, episode_frames=6) = 6 frames, shape (144, 160, 3)
    assert arr.shape == (6, 144, 160, 3)


def test_knn_grid_returns_none_for_pixel_bundle(tmp_path):
    """No encoder → no retrieval section."""
    from deepEmulator.training.eval import _knn_retrieval_grid, evaluate_bundle

    bundle = _make_fake_bundle(tmp_path)
    factory = lambda _e: _FakeEvalEnv(episode_length=6)
    evaluate_bundle(bundle, env_factory=factory, n_episodes=2, capture_frames_per_episode=4)
    assert _knn_retrieval_grid(bundle) is None


def test_knn_grid_renders_for_encoder_bundle(tmp_path):
    """With encoder metadata present, the grid renders as a base64 PNG."""
    import json

    from deepEmulator.encoders.vit import ViTConfig, ViTTiny
    from deepEmulator.training.eval import _knn_retrieval_grid, evaluate_bundle
    from deepEmulator.utils.checkpoints import write_bundle
    from deepEmulator.agents.ddqn_torch import DDQNAgent

    # Tiny encoder bundle on disk
    cfg = ViTConfig(image_size=96, patch_size=8, embed_dim=96, depth=2, num_heads=3, out_dim=64)
    enc = ViTTiny(cfg)
    enc_dir = tmp_path / "dino_run"
    enc_dir.mkdir()
    torch.save(enc.state_dict(), enc_dir / "encoder_only.pt")
    (enc_dir / "metadata.json").write_text(
        json.dumps({"algo": "dino", "vit_config": cfg.__dict__})
    )

    # DDQN bundle (pixel-shaped, but with encoder linkage so k-NN runs)
    agent = DDQNAgent(obs_shape=(3, 72, 80), n_actions=7, device="cpu")
    bundle = write_bundle(
        tmp_path / "ddqn_run",
        agent_state=agent.state_dict(),
        cartridge_title="POKEMON RED",
        cartridge_platform="gameboy",
        action_set=list(range(7)),
        obs_shape=(3, 72, 80),
        algo="ddqn",
        extra={"encoder": {
            "path": str(enc_dir.resolve()),
            "frozen": True,
            "latent_dim": 64,
            "preprocessing": "96x96_grayscale",
        }},
    )
    # Drop in a metrics.tsv stub
    (bundle / "metrics.tsv").write_text("Step\tMeanReward\n0\t0.0\n")

    factory = lambda _e: _FakeEvalEnv(episode_length=10)
    evaluate_bundle(bundle, env_factory=factory, n_episodes=2, capture_frames_per_episode=5)
    grid = _knn_retrieval_grid(bundle)
    assert grid is not None
    assert grid.startswith("data:image/png;base64,")


def test_report_includes_knn_section_when_encoder_present(tmp_path):
    """write_report should add an 'Encoder retrieval' block when the bundle
    has eval_frames.npz + an encoder linkage."""
    import json

    from deepEmulator.encoders.vit import ViTConfig, ViTTiny
    from deepEmulator.training.eval import evaluate_bundle, write_report
    from deepEmulator.utils.checkpoints import write_bundle
    from deepEmulator.agents.ddqn_torch import DDQNAgent

    cfg = ViTConfig(image_size=96, patch_size=8, embed_dim=96, depth=2, num_heads=3, out_dim=64)
    enc = ViTTiny(cfg)
    enc_dir = tmp_path / "enc"
    enc_dir.mkdir()
    torch.save(enc.state_dict(), enc_dir / "encoder_only.pt")
    (enc_dir / "metadata.json").write_text(json.dumps({"algo": "dino", "vit_config": cfg.__dict__}))

    agent = DDQNAgent(obs_shape=(3, 72, 80), n_actions=7, device="cpu")
    bundle = write_bundle(
        tmp_path / "b",
        agent_state=agent.state_dict(),
        cartridge_title="POKEMON RED",
        cartridge_platform="gameboy",
        action_set=list(range(7)),
        obs_shape=(3, 72, 80),
        algo="ddqn",
        extra={"encoder": {"path": str(enc_dir.resolve()), "frozen": True, "latent_dim": 64}},
    )
    summary = evaluate_bundle(
        bundle,
        env_factory=lambda _e: _FakeEvalEnv(episode_length=10),
        n_episodes=2,
        capture_frames_per_episode=5,
    )

    out = write_report([summary], [bundle], tmp_path / "report.html")
    html = out.read_text()
    assert "Encoder retrieval" in html
    assert "k-NN" in html
    assert "data:image/png;base64," in html


def test_end_to_end_cli_eval(tmp_path, monkeypatch):
    """Full CLI smoke: eval one bundle via injected fake env factory."""
    import sys

    from deepEmulator.training import eval as eval_mod

    bundle = _make_fake_bundle(tmp_path)

    monkeypatch.setattr(eval_mod, "make_env", lambda **kw: _FakeEvalEnv(episode_length=4, ep_reward=1.0))

    monkeypatch.setattr(sys, "argv", [
        "deepemu-eval",
        "--baseline", str(bundle),
        "--cartridge", "POKEMON RED",
        "--rom", str(tmp_path / "fake.gb"),
        "--episodes", "3",
        "--out", str(tmp_path / "report.html"),
    ])
    rc = eval_mod.main()
    assert rc == 0
    assert (tmp_path / "report.html").exists()
    assert (bundle / "eval_episodes.tsv").exists()


def test_eval_main_rejects_cartridge_mismatch(tmp_path, monkeypatch):
    import sys

    from deepEmulator.training import eval as eval_mod

    bundle = _make_fake_bundle(tmp_path)
    monkeypatch.setattr(eval_mod, "make_env", lambda **kw: _FakeEvalEnv(episode_length=4))
    monkeypatch.setattr(sys, "argv", [
        "deepemu-eval",
        "--baseline", str(bundle),
        "--cartridge", "POKEMON CRYSTAL",  # bundle says POKEMON RED
        "--rom", str(tmp_path / "fake.gb"),
        "--episodes", "1",
        "--out", str(tmp_path / "report.html"),
    ])
    with pytest.raises(ValueError, match="trained on"):
        eval_mod.main()


def test_eval_per_episode_reseed_makes_epsilon_stream_reproducible(tmp_path):
    """Two evaluations of the same bundle must consume identical epsilon
    coin-flips — the basis of fair baseline-vs-treatment comparison."""
    from deepEmulator.training.eval import evaluate_bundle

    bundle = _make_fake_bundle(tmp_path)

    class _SpyEnv(_FakeEvalEnv):
        def __init__(self):
            super().__init__(episode_length=6)
            self.actions: list[int] = []

        def step(self, action):
            self.actions.append(int(action))
            return super().step(action)

    seen = []
    for _ in range(2):
        env = _SpyEnv()
        evaluate_bundle(
            bundle,
            env_factory=lambda _e, _env=env: _env,
            n_episodes=2,
            epsilon=0.5,
            capture_frames_per_episode=0,
        )
        seen.append(env.actions)
    assert seen[0] == seen[1]


def test_config_from_metadata_legacy_and_new():
    from deepEmulator.agents.ddqn_torch import config_from_metadata

    legacy = config_from_metadata({"cartridge_title": "X"})
    assert legacy.dueling is False
    assert legacy.normalize_obs is False
    assert legacy.n_step == 1

    new = config_from_metadata(
        {"network": {"dueling": True, "normalize_obs": True, "n_step": 3, "gamma": 0.95}}
    )
    assert new.dueling is True
    assert new.normalize_obs is True
    assert new.n_step == 3
    assert new.gamma == pytest.approx(0.95)

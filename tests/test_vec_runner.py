"""VecEnvRunner — spawn workers over the synthetic env (CPU-only, no ROMs)."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from deepEmulator.training.vec_runner import EnvSpec, VecEnvRunner, build_env  # noqa: E402

SPEC = EnvSpec(cartridge="SYNTH BLOB", max_episode_steps=10)


@pytest.mark.timeout(120)
def test_two_workers_step_with_auto_reset():
    with VecEnvRunner(SPEC, num_envs=2, base_seed=0) as runner:
        obs = runner.reset()
        assert obs.shape == (2, 3, 72, 80)
        assert obs.dtype == np.uint8

        saw_reset = False
        for _ in range(30):
            batch = runner.step(np.array([2, 2]))
            assert batch.obs.shape == (2, 3, 72, 80)
            assert batch.rewards.shape == (2,)
            for e in range(2):
                ended = bool(batch.terminated[e] or batch.truncated[e])
                # auto-reset contract: terminal obs delivered separately,
                # batch.obs already holds the next episode's first obs
                assert (batch.pre_reset_obs[e] is not None) == ended
                if ended:
                    saw_reset = True
        assert saw_reset  # episode_length=10, 30 steps -> must have ended


@pytest.mark.timeout(120)
def test_worker_exception_propagates_with_traceback():
    bogus = EnvSpec(cartridge="NO SUCH CARTRIDGE")
    runner = VecEnvRunner(bogus, num_envs=2, base_seed=0)
    with pytest.raises(RuntimeError, match="worker"):
        runner.reset()
    # all workers reaped
    for proc in runner._procs:
        proc.join(timeout=10)
        assert not proc.is_alive()


@pytest.mark.timeout(120)
def test_same_seed_same_transition_stream():
    def run_once():
        rng = np.random.default_rng(7)
        with VecEnvRunner(SPEC, num_envs=2, base_seed=3) as runner:
            obs = runner.reset(base_seed=3)
            stream = [obs.copy()]
            for _ in range(12):
                batch = runner.step(rng.integers(0, 7, size=2))
                stream.append(batch.obs.copy())
                stream.append(batch.rewards.copy())
            return stream

    a = run_once()
    b = run_once()
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


@pytest.mark.timeout(120)
def test_clean_close_no_zombies():
    runner = VecEnvRunner(SPEC, num_envs=2, base_seed=0)
    runner.reset()
    runner.close()
    for proc in runner._procs:
        assert not proc.is_alive()
        assert proc.exitcode == 0


def test_build_env_synthetic_dispatch():
    env = build_env(SPEC)
    obs, _ = env.reset()
    assert obs.shape == (3, 72, 80)
    assert env.cartridge.cartridge_title == "SYNTH BLOB"
    env.close()


def test_act_batch_matches_act_semantics():
    from deepEmulator.agents.ddqn_torch import DDQNAgent, DDQNConfig

    cfg = DDQNConfig(exploration_anneal_steps=100)
    agent = DDQNAgent(obs_shape=(3, 72, 80), n_actions=7, config=cfg, device="cpu")
    obs = np.zeros((4, 3, 72, 80), dtype=np.uint8)
    actions = agent.act_batch(obs)
    assert actions.shape == (4,)
    assert all(0 <= a < 7 for a in actions)
    assert agent.curr_step == 4
    # epsilon recomputed from curr_step (pure function under anneal)
    assert agent.exploration_rate == pytest.approx(1.0 - 0.95 * 0.04)


@pytest.mark.timeout(300)
def test_end_to_end_vec_mini_train(tmp_path, monkeypatch):
    """Full deepemu-train with --num-envs 2 on the synthetic cartridge."""
    import sys

    from deepEmulator.training import train as train_mod

    run_dir = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", [
        "deepemu-train",
        "--cartridge", "SYNTH BLOB",
        "--rom", "unused.gb",
        "--steps", "120",
        "--max-episode-steps", "15",
        "--save-every", "60",
        "--headless",
        "--num-envs", "2",
        "--seed", "0",
        "--run-dir", str(run_dir),
    ])
    rc = train_mod.main()
    assert rc == 0
    assert (run_dir / "model.pt").exists()
    assert (run_dir / "metadata.json").exists()
    rows = (run_dir / "metrics.tsv").read_text().strip().splitlines()
    assert len(rows) >= 2  # header + at least one finished episode
    trajs = list((run_dir / "trajectories").glob("episode_*.csv.gz"))
    assert len(trajs) >= 2  # one per env at minimum

    import torch as _torch

    sd = _torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True)
    assert sd["curr_step"] >= 120

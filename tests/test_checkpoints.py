"""Bundle writer/loader + trajectory CSV round-trip."""
from __future__ import annotations

import csv
import gzip

import pytest

torch = pytest.importorskip("torch")


def test_write_and_load_bundle(tmp_path):
    from deepEmulator.utils.checkpoints import load_bundle, write_bundle

    sd = {"online": torch.zeros(1), "exploration_rate": 0.5, "curr_step": 42}
    run = write_bundle(
        tmp_path / "run1",
        agent_state=sd,
        cartridge_title="POKEMON RED",
        cartridge_platform="gameboy",
        action_set=["down", "left", "right", "up", "a", "b", "start"],
        obs_shape=(3, 72, 80),
        algo="ddqn",
        extra={"global_steps": 42},
    )
    assert (run / "model.pt").exists()
    assert (run / "metadata.json").exists()
    assert (run / "trajectories").is_dir()

    agent_state, metadata = load_bundle(run)
    assert metadata["cartridge_title"] == "POKEMON RED"
    assert metadata["observation_shape"] == [3, 72, 80]
    assert metadata["algo"] == "ddqn"
    assert metadata["global_steps"] == 42
    assert "versions" in metadata
    assert agent_state["curr_step"] == 42


def test_trajectory_writer(tmp_path):
    from deepEmulator.utils.checkpoints import TrajectoryWriter

    tw = TrajectoryWriter(tmp_path)
    tw.start_episode(0)
    for step in range(5):
        tw.write(step, x=step, y=step * 2, map_id=40, action=step % 7, reward=0.1 * step)
    tw.close()

    path = tmp_path / "trajectories" / "episode_000000.csv.gz"
    with gzip.open(path, "rt") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["step", "x", "y", "map_id", "action", "reward"]
    assert len(rows) == 6
    assert rows[3] == ["2", "2", "4", "40", "2", "0.200000"]


def test_write_bundle_updates_latest_marker(tmp_path):
    from deepEmulator.utils.checkpoints import find_latest_run, write_bundle

    parent = tmp_path / "pokemon_red"
    sd = {"online": torch.zeros(1), "exploration_rate": 0.3, "curr_step": 100}
    common = dict(
        agent_state=sd,
        cartridge_title="POKEMON RED",
        cartridge_platform="gameboy",
        action_set=["a"],
        obs_shape=(3, 72, 80),
        algo="ddqn",
    )
    r1 = write_bundle(parent / "20260101_000000", **common)
    r2 = write_bundle(parent / "20260102_000000", **common)
    assert (parent / "latest.txt").exists()
    assert (parent / "latest.txt").read_text().strip() == str(r2.resolve())
    assert find_latest_run(parent) == r2


def test_find_latest_run_fallback_without_marker(tmp_path):
    from deepEmulator.utils.checkpoints import find_latest_run, write_bundle

    parent = tmp_path / "mario_land"
    common = dict(
        agent_state={"online": torch.zeros(1)},
        cartridge_title="X",
        cartridge_platform="gameboy",
        action_set=["a"],
        obs_shape=(1, 1, 1),
        algo="ddqn",
    )
    write_bundle(parent / "20260101_000000", **common)
    r2 = write_bundle(parent / "20260103_000000", **common)
    # delete marker -> falls back to lex-max
    (parent / "latest.txt").unlink()
    assert find_latest_run(parent) == r2


def test_find_latest_run_returns_none_for_empty(tmp_path):
    from deepEmulator.utils.checkpoints import find_latest_run

    assert find_latest_run(tmp_path / "doesnotexist") is None
    (tmp_path / "empty").mkdir()
    assert find_latest_run(tmp_path / "empty") is None


def test_metric_logger_writes_tsv(tmp_path):
    from deepEmulator.utils.logger import MetricLogger

    log = MetricLogger(tmp_path)
    for _ in range(3):
        log.log_step(1.0, 0.5, 0.2)
    row = log.end_episode(episode=0, step=3, epsilon=0.9)
    assert row["Episode"] == 0
    text = (tmp_path / "metrics.tsv").read_text().splitlines()
    assert text[0].startswith("Episode\t")
    assert len(text) == 2

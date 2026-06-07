"""Portable checkpoint bundle — the Colab→local seam.

Layout:
    {run_dir}/
        model.pt         # torch.save(agent.state_dict())
        metadata.json    # cartridge, platform, action_set, obs_shape, algo, versions
        trajectories/episode_{n}.csv.gz
        metrics.tsv
        plots/{reward,loss,q}.jpg  (written by logger, optional)
"""
from __future__ import annotations

import csv
import gzip
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path

import torch


def _versions() -> dict:
    info = {"python": sys.version.split()[0], "torch": torch.__version__}
    try:
        import pyboy  # type: ignore

        info["pyboy"] = pyboy.__version__ if hasattr(pyboy, "__version__") else "unknown"
    except Exception:
        pass
    try:
        import deepEmulator  # noqa

        info["deepEmulator"] = deepEmulator.__version__
    except Exception:
        pass
    info["platform"] = platform.platform()
    return info


def write_bundle(
    run_dir: Path | str,
    *,
    agent_state: dict,
    cartridge_title: str,
    cartridge_platform: str,
    action_set: list,
    obs_shape: tuple,
    algo: str,
    extra: dict | None = None,
    update_latest: bool = True,
) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "trajectories").mkdir(exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)

    torch.save(agent_state, run_dir / "model.pt")

    metadata = {
        "cartridge_title": cartridge_title,
        "platform": cartridge_platform,
        "action_set": list(action_set),
        "observation_shape": list(obs_shape),
        "algo": algo,
        "versions": _versions(),
    }
    if extra:
        metadata.update(extra)
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    if update_latest:
        # Write a "latest" marker as a small text file (works on Drive where
        # symlinks are unreliable). The marker points at the absolute run dir.
        (run_dir.parent / "latest.txt").write_text(str(run_dir.resolve()) + "\n")
    return run_dir


def load_bundle(run_dir: Path | str, map_location: str = "cpu") -> tuple[dict, dict]:
    run_dir = Path(run_dir)
    with open(run_dir / "metadata.json") as f:
        metadata = json.load(f)
    agent_state = torch.load(run_dir / "model.pt", map_location=map_location, weights_only=False)
    return agent_state, metadata


def find_latest_run(parent_dir: Path | str) -> Path | None:
    """Return the latest run dir under `parent_dir`.

    Prefers `latest.txt` if present (most-recent saver wins on Drive even when
    timestamps can be unreliable across sessions). Falls back to lexicographic
    max of timestamp-named subdirs.
    """
    parent = Path(parent_dir)
    if not parent.exists():
        return None
    marker = parent / "latest.txt"
    if marker.exists():
        candidate = Path(marker.read_text().strip())
        if (candidate / "metadata.json").exists():
            return candidate
    candidates = sorted(
        [p for p in parent.iterdir() if p.is_dir() and (p / "metadata.json").exists()]
    )
    return candidates[-1] if candidates else None


class TrajectoryWriter:
    """One gzipped CSV per episode: (step, x, y, map_id, action, reward)."""

    def __init__(self, run_dir: Path | str):
        self.run_dir = Path(run_dir) / "trajectories"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._fp = None
        self._writer = None
        self._episode = -1

    def start_episode(self, episode: int) -> None:
        self.close()
        self._episode = episode
        path = self.run_dir / f"episode_{episode:06d}.csv.gz"
        self._fp = gzip.open(path, "wt", newline="")
        self._writer = csv.writer(self._fp)
        self._writer.writerow(["step", "x", "y", "map_id", "action", "reward"])

    def write(self, step: int, x: int, y: int, map_id: int, action: int, reward: float) -> None:
        if self._writer is None:
            return
        self._writer.writerow([step, x, y, map_id, action, f"{reward:.6f}"])

    def close(self) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None
            self._writer = None

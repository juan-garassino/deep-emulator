"""MetricLogger — TSV per-episode rolling stats. Plots are optional ([viz] extra)."""
from __future__ import annotations

from collections import deque
from pathlib import Path
from time import time


class MetricLogger:
    """Records per-step (reward, loss, q) and emits one TSV row per episode.

    Columns mirror lixado's format: Episode, Step, Epsilon, MeanReward,
    MeanLength, MeanLoss, MeanQValue, TimeDelta, Time.
    """

    HEADER = [
        "Episode",
        "Step",
        "Epsilon",
        "MeanReward",
        "MeanLength",
        "MeanLoss",
        "MeanQValue",
        "TimeDelta",
        "Time",
    ]

    def __init__(self, run_dir: Path | str, window: int = 100):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "metrics.tsv"
        if not self.path.exists():
            with open(self.path, "w") as f:
                f.write("\t".join(self.HEADER) + "\n")

        self._ep_rewards: deque[float] = deque(maxlen=window)
        self._ep_lengths: deque[int] = deque(maxlen=window)
        self._ep_losses: deque[float] = deque(maxlen=window)
        self._ep_qs: deque[float] = deque(maxlen=window)

        self._cur_reward = 0.0
        self._cur_length = 0
        self._cur_losses: list[float] = []
        self._cur_qs: list[float] = []

        self._t0 = time()
        self._t_last = self._t0

    def log_step(self, reward: float, loss: float | None, q: float | None) -> None:
        self._cur_reward += reward
        self._cur_length += 1
        if loss is not None:
            self._cur_losses.append(loss)
        if q is not None:
            self._cur_qs.append(q)

    def end_episode(self, *, episode: int, step: int, epsilon: float) -> dict:
        self._ep_rewards.append(self._cur_reward)
        self._ep_lengths.append(self._cur_length)
        self._ep_losses.append(
            sum(self._cur_losses) / max(len(self._cur_losses), 1) if self._cur_losses else 0.0
        )
        self._ep_qs.append(sum(self._cur_qs) / max(len(self._cur_qs), 1) if self._cur_qs else 0.0)

        now = time()
        row = {
            "Episode": episode,
            "Step": step,
            "Epsilon": f"{epsilon:.4f}",
            "MeanReward": f"{sum(self._ep_rewards) / len(self._ep_rewards):.3f}",
            "MeanLength": f"{sum(self._ep_lengths) / len(self._ep_lengths):.1f}",
            "MeanLoss": f"{sum(self._ep_losses) / len(self._ep_losses):.4f}",
            "MeanQValue": f"{sum(self._ep_qs) / len(self._ep_qs):.4f}",
            "TimeDelta": f"{now - self._t_last:.2f}",
            "Time": f"{now - self._t0:.2f}",
        }
        with open(self.path, "a") as f:
            f.write("\t".join(str(row[c]) for c in self.HEADER) + "\n")

        self._cur_reward = 0.0
        self._cur_length = 0
        self._cur_losses = []
        self._cur_qs = []
        self._t_last = now
        return row

    def plot(self) -> None:  # pragma: no cover - requires viz extra
        """Render reward/length/loss/q JPGs into {run_dir}/plots/. Needs matplotlib."""
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except ImportError:
            return

        plots_dir = self.run_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        for name, series in [
            ("reward", self._ep_rewards),
            ("length", self._ep_lengths),
            ("loss", self._ep_losses),
            ("q", self._ep_qs),
        ]:
            fig, ax = plt.subplots()
            ax.plot(list(series))
            ax.set_title(name)
            fig.savefig(plots_dir / f"{name}.jpg")
            plt.close(fig)

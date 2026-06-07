"""Minimal vendored space primitives (Box, Discrete). No gymnasium dep.

Just enough surface for our envs + DDQN agent. If a future need outgrows this,
revisit before adding gymnasium back.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Box:
    low: float
    high: float
    shape: tuple[int, ...]
    dtype: np.dtype = np.dtype(np.float32)

    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return rng.uniform(self.low, self.high, size=self.shape).astype(self.dtype)

    def contains(self, x: np.ndarray) -> bool:
        return (
            isinstance(x, np.ndarray)
            and x.shape == self.shape
            and bool(np.all(x >= self.low))
            and bool(np.all(x <= self.high))
        )


@dataclass
class Discrete:
    n: int

    def sample(self, rng: np.random.Generator | None = None) -> int:
        rng = rng or np.random.default_rng()
        return int(rng.integers(0, self.n))

    def contains(self, x: int) -> bool:
        return isinstance(x, (int, np.integer)) and 0 <= int(x) < self.n

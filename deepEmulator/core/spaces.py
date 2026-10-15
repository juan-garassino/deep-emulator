"""Minimal vendored space primitives (Box, Discrete). No gymnasium dep.

Just enough surface for our envs + DDQN agent. If a future need outgrows this,
revisit before adding gymnasium back.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Box:
    """Continuous box space: every element lies in `[low, high]` and the array has `shape`."""
    low: float
    high: float
    shape: tuple[int, ...]
    dtype: np.dtype = np.dtype(np.float32)

    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Draw a uniform sample of shape `self.shape` and dtype `self.dtype`."""
        rng = rng or np.random.default_rng()
        return rng.uniform(self.low, self.high, size=self.shape).astype(self.dtype)

    def contains(self, x: np.ndarray) -> bool:
        """True if `x` is an ndarray of exactly `self.shape` with all values within bounds."""
        return (
            isinstance(x, np.ndarray)
            and x.shape == self.shape
            and bool(np.all(x >= self.low))
            and bool(np.all(x <= self.high))
        )


@dataclass
class Discrete:
    """Discrete space of integers `0 .. n-1`."""
    n: int

    def sample(self, rng: np.random.Generator | None = None) -> int:
        """Draw a uniform integer in `[0, n)`."""
        rng = rng or np.random.default_rng()
        return int(rng.integers(0, self.n))

    def contains(self, x: int) -> bool:
        """True if `x` is an integer in `[0, n)`."""
        return isinstance(x, (int, np.integer)) and 0 <= int(x) < self.n

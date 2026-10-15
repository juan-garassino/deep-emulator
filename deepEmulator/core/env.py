"""Minimal Env protocol — no gymnasium dependency.

Matches the standard (obs, reward, terminated, truncated, info) tuple shape
so wrappers and trajectories stay compatible if we ever revisit. Platform
backends (`deepEmulator.platforms.gameboy`, `.sega`) subclass this.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from deepEmulator.core.cartridge import CartridgeAdapter
from deepEmulator.core.spaces import Box, Discrete


class EmulatorEnv:
    """Backend-agnostic emulator env. Concrete subclasses live in `deepEmulator.platforms`."""

    observation_space: Box
    action_space: Discrete

    def __init__(self, cartridge: CartridgeAdapter):
        self.cartridge = cartridge

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict]:  # pragma: no cover - P0 stub
        """Reset the emulator and return `(observation, info)`. Subclasses must override."""
        raise NotImplementedError

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:  # pragma: no cover - P0 stub
        """Apply `action`; return `(obs, reward, terminated, truncated, info)`. Subclasses must override."""
        raise NotImplementedError

    def render(self) -> Any:  # pragma: no cover - P0 stub
        """Return a renderable frame. Subclasses must override."""
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - P0 stub
        """Release emulator resources. The base implementation does nothing."""
        pass

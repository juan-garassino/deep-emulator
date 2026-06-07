"""CartridgeAdapter ABC — per-ROM RAM map + reward + action set + done.

Concrete adapters live in `deepEmulator.cartridges.*`.
Modeled on lixado's AISettingsInterface but Gymnasium-native.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass
class CartridgeAdapter(ABC):
    cartridge_title: str
    platform: Literal["gameboy", "sega"]
    init_state: Path | None = None
    action_set: list[int] = field(default_factory=list)
    observation_shape: tuple[int, ...] = ()

    @abstractmethod
    def compute_reward(self, prev_state: dict, curr_state: dict, emulator: Any) -> float: ...

    @abstractmethod
    def read_game_state(self, emulator: Any) -> dict: ...

    @abstractmethod
    def is_done(self, state: dict) -> bool: ...

    def get_trajectory_coords(self, state: dict) -> tuple[int, int, int]:
        """(x, y, map_id) for arrow visualization. Override if applicable."""
        return (state.get("x", 0), state.get("y", 0), state.get("map_id", 0))

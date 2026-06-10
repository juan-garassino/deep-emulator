"""Zero-reward adapter for any Game Boy / Game Boy Color ROM.

The point: smoke-test the PyBoy plumbing on any ROM without needing a real
RAM map. Reward is always 0, episodes never terminate (env's max_steps bounds
them), and no init.state is required.

Use cases:
- Verify PyBoy 2.4 API works against a cartridge you have lying around
- Collect frames for SSL pretraining from any ROM
- Render `deepemu-play --visible` for any ROM without writing an adapter
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from deepEmulator.core.cartridge import CartridgeAdapter
from deepEmulator.core.registry import register


# Default GB action set, button names matching PyBoy 2.x API. "start" is
# opt-in (include_start=True) — menu spam wastes exploratory actions.
_DEFAULT_ACTIONS = ["down", "left", "right", "up", "a", "b"]


@dataclass
class GenericGameBoyAdapter(CartridgeAdapter):
    cartridge_title: str = "GENERIC GB"
    platform: str = "gameboy"
    init_state: Any = None
    action_set: list = field(default_factory=lambda: list(_DEFAULT_ACTIONS))
    observation_shape: tuple = (3, 72, 80)
    include_start: bool = False

    def __post_init__(self) -> None:
        if self.include_start and "start" not in self.action_set:
            self.action_set = [*self.action_set, "start"]

    def reset_episode(self, emulator: Any) -> None:
        pass

    def read_game_state(self, emulator: Any) -> dict:
        # No RAM reading — zero-cost stub
        return {"x": 0, "y": 0, "map_id": 0}

    def compute_reward(self, prev_state: dict, curr_state: dict, emulator: Any) -> float:
        return 0.0

    def is_done(self, state: dict) -> bool:
        return False


register("GENERIC GB")(GenericGameBoyAdapter)

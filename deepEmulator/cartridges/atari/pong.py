"""Atari Pong cartridge adapter.

ALE Pong has a small action set (NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE
via getMinimalActionSet). Reward is the raw ALE reward, optionally clipped to
{-1, 0, +1}. No life-loss policy — Pong has no extra lives.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from deepEmulator.core.cartridge import CartridgeAdapter
from deepEmulator.core.registry import register


@dataclass
class PongAdapter(CartridgeAdapter):
    cartridge_title: str = "ATARI PONG"
    platform: str = "atari"
    rom_name: str = "pong"
    init_state: Any = None
    action_set: list = None  # set by env from ALE minimal action set
    observation_shape: tuple = (4, 84, 84)

    clip_reward: bool = True
    terminal_on_life_loss: bool = False  # Pong has no lives

    def __post_init__(self) -> None:
        if self.action_set is None:
            self.action_set = []  # env populates after ALE loads ROM

    def reset_episode(self, emulator: Any) -> None:
        pass

    def read_game_state(self, emulator: Any) -> dict:
        return {"game_over": bool(emulator.game_over()), "lives": int(emulator.lives())}

    def compute_reward(self, prev_state: dict, curr_state: dict, emulator: Any) -> float:
        r = float(curr_state.get("raw_reward", 0.0))
        if self.clip_reward:
            if r > 0:
                return 1.0
            if r < 0:
                return -1.0
            return 0.0
        return r

    def is_done(self, state: dict) -> bool:
        return bool(state.get("game_over", False))

    def get_trajectory_coords(self, state: dict) -> tuple[int, int, int]:
        # Atari Pong has no semantic (x, y, map) — return zeros
        return (0, 0, 0)


register("ATARI PONG")(PongAdapter)

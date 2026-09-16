"""CartridgeAdapter ABC — per-ROM RAM map + reward + action set + done.

Concrete adapters live in `deepEmulator.cartridges.*`.
Modeled on lixado's AISettingsInterface but Gymnasium-native.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Literal

# Pairs that can never be pressed together on a d-pad — a game reads them as
# "no direction" and it wastes the combo slot. lixado/PyBoy-RL strips these in
# MarioAISettings.GetActions; we generalize the same guard here.
_CONTRADICTORY: tuple[frozenset[str], ...] = (
    frozenset({"left", "right"}),
    frozenset({"up", "down"}),
)


def combo_action_set(
    buttons: list[str],
    *,
    max_buttons: int = 2,
    include_noop: bool = False,
) -> list[list[str]]:
    """Build a multi-button action space from single buttons.

    Ports lixado/PyBoy-RL's permutation-generated action list (the
    "hold A while running right" fix): every action is a *list* of buttons the
    env presses simultaneously. Combos up to `max_buttons` wide are generated,
    with contradictory d-pad pairs (L+R, U+D) removed.

    Example:
        combo_action_set(["left", "right", "a"], max_buttons=2) ->
        [["left"], ["right"], ["a"], ["left", "a"], ["right", "a"]]
        (["left", "right"] dropped as contradictory)

    Args:
        buttons: single-button vocabulary (PyBoy names, e.g. "a", "left")
        max_buttons: widest simultaneous press (2 = one direction + one action)
        include_noop: prepend the empty action `[]` (idle step) when True

    Returns a list of button-name lists, order-stable (singles first, then
    combos), suitable as an env's action_set where each index maps to a combo.
    """
    actions: list[list[str]] = [[b] for b in buttons]
    for width in range(2, max(2, max_buttons) + 1):
        for combo in combinations(buttons, width):
            combo_set = frozenset(combo)
            if any(bad <= combo_set for bad in _CONTRADICTORY):
                continue
            actions.append(list(combo))
    if include_noop:
        actions.insert(0, [])
    return actions


@dataclass
class CartridgeAdapter(ABC):
    cartridge_title: str
    platform: Literal["gameboy", "atari", "sega"]
    init_state: Path | None = None
    action_set: list[int] = field(default_factory=list)
    observation_shape: tuple[int, ...] = ()

    @abstractmethod
    def compute_reward(self, prev_state: dict, curr_state: dict, emulator: Any) -> float: ...

    @abstractmethod
    def read_game_state(self, emulator: Any) -> dict: ...

    @abstractmethod
    def is_done(self, state: dict) -> bool: ...

    def reset_episode(self, emulator: Any) -> None:
        """Called by the env on every reset(). Stateful adapters (exploration
        bookkeeping, reward baselines) MUST override this; stateless ones get
        the no-op. Part of the env contract — both PyBoyEnv and AtariEnv call it."""

    def get_trajectory_coords(self, state: dict) -> tuple[int, int, int]:
        """(x, y, map_id) for arrow visualization. Override if applicable."""
        return (state.get("x", 0), state.get("y", 0), state.get("map_id", 0))

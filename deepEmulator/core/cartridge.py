"""CartridgeAdapter ABC — per-ROM RAM map + reward + action set + done.

Concrete adapters live in `deepEmulator.cartridges.*`.
Modeled on lixado/PyBoy-RL's AISettings interface, without a gymnasium dependency.
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
    """Per-ROM plugin: RAM reads, reward, done condition and action set.

    Subclass, implement the three abstract methods, and register with
    `deepEmulator.core.registry.register`.

    Attributes:
        cartridge_title: registry key (ROM header title for Game Boy).
        platform: `"gameboy"`, `"atari"` or `"sega"`.
        init_state: optional save-state used as the episode start.
        action_set: platform-specific action vocabulary indexed by action id.
        observation_shape: shape of the observations the env produces.
    """

    cartridge_title: str
    platform: Literal["gameboy", "atari", "sega"]
    init_state: Path | None = None
    action_set: list[int] = field(default_factory=list)
    observation_shape: tuple[int, ...] = ()

    @abstractmethod
    def compute_reward(self, prev_state: dict, curr_state: dict, emulator: Any) -> float:
        """Return the scalar reward for the transition `prev_state -> curr_state`.

        Both dicts come from `read_game_state`; `emulator` is the backend handle.
        """

    @abstractmethod
    def read_game_state(self, emulator: Any) -> dict:
        """Read the cartridge-specific game state (RAM values etc.) into a plain dict."""

    @abstractmethod
    def is_done(self, state: dict) -> bool:
        """True if `state` (from `read_game_state`) is a terminal state of the game."""

    def reset_episode(self, emulator: Any) -> None:
        """Called by the env on every reset(). Stateful adapters (exploration
        bookkeeping, reward baselines) MUST override this; stateless ones get
        the no-op. Part of the env contract — both PyBoyEnv and AtariEnv call it."""

    def get_trajectory_coords(self, state: dict) -> tuple[int, int, int]:
        """(x, y, map_id) for arrow visualization. Override if applicable."""
        return (state.get("x", 0), state.get("y", 0), state.get("map_id", 0))

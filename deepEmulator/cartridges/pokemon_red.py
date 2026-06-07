"""Pokemon Red cartridge adapter.

RAM addresses + reward shaping ported from PWhiddy/PokemonRedExperiments:
- baselines/memory_addresses.py
- v2/red_gym_env_v2.py (~line 514-528 reward, line 459 read_m, line 557 read_hp_fraction)
- v2/global_map.py (local_to_global)

Per-episode state (seen_coords, max rewards) lives on the adapter and is wiped
in `reset_episode()`. The PyBoy env calls `compute_reward` / `read_game_state`
each step.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from deepEmulator.core.cartridge import CartridgeAdapter
from deepEmulator.core.registry import register

# --- RAM map (datacrystal Pokemon Red/Blue RAM map) -----------------------------
PARTY_SIZE = 0xD163
X_POS, Y_POS, MAP_N = 0xD362, 0xD361, 0xD35E
BADGE_COUNT = 0xD356
IN_BATTLE = 0xD057
LEVELS = (0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268)
PARTY_SPECIES = (0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169)
HP = (0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248)
MAX_HP = (0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269)

EVENT_FLAGS_START = 0xD747
EVENT_FLAGS_END = 0xD87E
MUSEUM_TICKET = (0xD754, 0)  # (addr, bit)

# PyBoy action set (7) — matches PWhiddy v2/red_gym_env_v2.py:62-80
ACTIONS = ["down", "left", "right", "up", "a", "b", "start"]


def _bit_count(b: int) -> int:
    return bin(b).count("1")


def _read_bit(pyboy: Any, addr: int, bit: int) -> bool:
    return bin(256 + pyboy.memory[addr])[-bit - 1] == "1"


# --- global map (vendored from v2/global_map.py) --------------------------------
_MAP_DATA_PATH = Path(__file__).parent / "pokemon_red_map_data.json"
_PAD = 20
GLOBAL_MAP_SHAPE = (444 + _PAD * 2, 436 + _PAD * 2)
with open(_MAP_DATA_PATH) as _f:
    _MAP_DATA = {int(e["id"]): e for e in json.load(_f)["regions"]}


def local_to_global(r: int, c: int, map_n: int) -> tuple[int, int]:
    """(y, x, map_id) -> (global_y, global_x). Falls back to center on KeyError."""
    if map_n not in _MAP_DATA:
        return GLOBAL_MAP_SHAPE[0] // 2, GLOBAL_MAP_SHAPE[1] // 2
    map_x, map_y = _MAP_DATA[map_n]["coordinates"]
    gy, gx = r + map_y + _PAD, c + map_x + _PAD
    if 0 <= gy < GLOBAL_MAP_SHAPE[0] and 0 <= gx < GLOBAL_MAP_SHAPE[1]:
        return gy, gx
    return GLOBAL_MAP_SHAPE[0] // 2, GLOBAL_MAP_SHAPE[1] // 2


@dataclass
class PokemonRedAdapter(CartridgeAdapter):
    cartridge_title: str = "POKEMON RED"
    platform: str = "gameboy"
    init_state: Path | None = None
    action_set: list[str] = field(default_factory=lambda: list(ACTIONS))
    observation_shape: tuple[int, ...] = (3, 72, 80)  # (frame_stack, H, W) downscaled 2x

    reward_scale: float = 1.0
    explore_weight: float = 1.0

    # per-episode state
    seen_coords: dict[str, int] = field(default_factory=dict)
    max_event_rew: float = 0.0
    total_healing_rew: float = 0.0
    last_health: float = 1.0
    party_size: int = 0
    died_count: int = 0
    base_event_flags: int = 0
    _total_reward: float = 0.0

    def reset_episode(self, pyboy: Any) -> None:
        self.seen_coords = {}
        self.max_event_rew = 0.0
        self.total_healing_rew = 0.0
        self.last_health = self._hp_fraction(pyboy)
        self.party_size = pyboy.memory[PARTY_SIZE]
        self.died_count = 0
        self.base_event_flags = sum(
            _bit_count(pyboy.memory[i]) for i in range(EVENT_FLAGS_START, EVENT_FLAGS_END)
        )
        self._total_reward = sum(self._reward_components(pyboy).values())

    # --- RAM reads ----------------------------------------------------------
    def _hp_fraction(self, pyboy: Any) -> float:
        hp_sum = sum(256 * pyboy.memory[a] + pyboy.memory[a + 1] for a in HP)
        max_hp_sum = sum(256 * pyboy.memory[a] + pyboy.memory[a + 1] for a in MAX_HP)
        return hp_sum / max(max_hp_sum, 1)

    def _badges(self, pyboy: Any) -> int:
        return _bit_count(pyboy.memory[BADGE_COUNT])

    def _events_reward(self, pyboy: Any) -> int:
        total = sum(
            _bit_count(pyboy.memory[i]) for i in range(EVENT_FLAGS_START, EVENT_FLAGS_END)
        )
        return max(total - self.base_event_flags - int(_read_bit(pyboy, *MUSEUM_TICKET)), 0)

    def _coords(self, pyboy: Any) -> tuple[int, int, int]:
        return pyboy.memory[X_POS], pyboy.memory[Y_POS], pyboy.memory[MAP_N]

    def _update_seen(self, pyboy: Any) -> None:
        if pyboy.memory[IN_BATTLE] == 0:
            x, y, m = self._coords(pyboy)
            key = f"{x}:{y}:{m}"
            self.seen_coords[key] = self.seen_coords.get(key, 0) + 1

    def _stuck_penalty(self, pyboy: Any) -> int:
        x, y, m = self._coords(pyboy)
        key = f"{x}:{y}:{m}"
        return 1 if self.seen_coords.get(key, 0) >= 600 else 0

    def _update_heal(self, pyboy: Any) -> None:
        cur = self._hp_fraction(pyboy)
        if cur > self.last_health and pyboy.memory[PARTY_SIZE] == self.party_size:
            if self.last_health > 0:
                heal = cur - self.last_health
                self.total_healing_rew += heal * heal
            else:
                self.died_count += 1
        self.last_health = cur

    # --- reward components --------------------------------------------------
    def _reward_components(self, pyboy: Any) -> dict[str, float]:
        cur_events = self._events_reward(pyboy)
        self.max_event_rew = max(cur_events, self.max_event_rew)
        return {
            "event": self.reward_scale * self.max_event_rew * 4,
            "heal": self.reward_scale * self.total_healing_rew * 10,
            "badge": self.reward_scale * self._badges(pyboy) * 10,
            "explore": self.reward_scale * self.explore_weight * len(self.seen_coords) * 0.1,
            "stuck": self.reward_scale * self._stuck_penalty(pyboy) * -0.05,
        }

    # --- CartridgeAdapter interface ----------------------------------------
    def read_game_state(self, emulator: Any) -> dict:
        x, y, m = self._coords(emulator)
        return {
            "x": x,
            "y": y,
            "map_id": m,
            "hp_fraction": self._hp_fraction(emulator),
            "badges": self._badges(emulator),
            "party_size": emulator.memory[PARTY_SIZE],
            "levels": [emulator.memory[a] for a in LEVELS],
        }

    def compute_reward(self, prev_state: dict, curr_state: dict, emulator: Any) -> float:
        # update per-step bookkeeping driven by RAM
        self._update_seen(emulator)
        self._update_heal(emulator)
        components = self._reward_components(emulator)
        new_total = sum(components.values())
        delta = new_total - self._total_reward
        self._total_reward = new_total
        return delta

    def is_done(self, state: dict) -> bool:
        # episode termination is step-count-bounded at the env level
        return False

    def get_trajectory_coords(self, state: dict) -> tuple[int, int, int]:
        return state["x"], state["y"], state["map_id"]


register("POKEMON RED")(PokemonRedAdapter)

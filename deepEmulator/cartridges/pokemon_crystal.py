"""Pokemon Crystal (Generation II, Game Boy Color) cartridge adapter.

Target ROM version: Pokemon Crystal v1.1 (US)
    sha1: cd1438c79d6efabd54312c80fb826f4f0eaa3924

RAM addresses sourced from:
- pret/pokecrystal  (commit master @ 8f2162d, fetched this session)
  https://github.com/pret/pokecrystal
- Datacrystal RAM map  https://datacrystal.tcrf.net/wiki/Pok%C3%A9mon_Crystal/RAM_map
- Bulbapedia Gen 2 memory map  https://bulbapedia.bulbagarden.net/

v1.0 vs v1.1: RAM layout is functionally identical for the variables we read.
The two versions differ only in ROM-side bug fixes that don't shift WRAM offsets.

Implementation notes:
- Several addresses (player position, in-battle flag) are widely cited by the
  Crystal speedrun/TAS/RL community but were not authoritatively pinned from a
  pret build artifact in this session. They are marked `# VERIFY` in the
  constants below — call `dump_state(env.pyboy)` after `env.reset()` against a
  real Crystal ROM at a known game state (e.g. standing in Cherrygrove) and
  confirm the values match expectations. Address mistakes here surface as
  obviously wrong reward values (party_count=42, badges=255, etc.).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from deepEmulator.core.cartridge import CartridgeAdapter
from deepEmulator.core.registry import register
from deepEmulator.core.reward import PhasedReward, RewardPhase, constant_per_step


# --- RAM map -----------------------------------------------------------------
# Confirmed via Datacrystal + pret/pokecrystal ram/wram.asm
PLAYER_NAME = 0xD47D                # wPlayerName (11 bytes incl. terminator)
JOHTO_BADGES = 0xD857               # wJohtoBadges (1-byte bitmask, 8 bits)
KANTO_BADGES = 0xD858               # wKantoBadges (1-byte bitmask, 8 bits)
MONEY = 0xD84E                      # wMoney (3 bytes BCD, big-endian)
EVENT_FLAGS_START = 0xDA72          # wEventFlags
NUM_EVENTS = 800                    # from constants/event_flags.asm
EVENT_FLAGS_END = EVENT_FLAGS_START + (NUM_EVENTS // 8)  # 0xDAD4 inclusive

PARTY_COUNT = 0xDCD7                # wPartyCount
PARTY_SPECIES = 0xDCD8              # wPartySpecies (6 bytes + 0xFF terminator)
PARTY_MONS = 0xDCDF                 # wPartyMons (start of struct array)
PARTYMON_STRUCT_LEN = 0x30          # 48 bytes per PartyMon
PARTYMON_LEVEL_OFFSET = 0x1F        # +0x1F → level (1 byte)
PARTYMON_HP_OFFSET = 0x22           # +0x22 → current HP (2 bytes big-endian)
PARTYMON_MAXHP_OFFSET = 0x24        # +0x24 → max HP (2 bytes big-endian)

# Widely cited Crystal community addresses — VERIFY on first ROM load
MAP_GROUP = 0xDCB5                  # wMapGroup  # VERIFY
MAP_NUMBER = 0xDCB6                 # wMapNumber  # VERIFY
Y_COORD = 0xDCB7                    # wYCoord  # VERIFY
X_COORD = 0xDCB8                    # wXCoord  # VERIFY
BATTLE_MODE = 0xD22D                # wBattleMode (0=overworld, 1=wild, 2=trainer)  # VERIFY

# PyBoy 7-action standard GB button set
ACTIONS = ["down", "left", "right", "up", "a", "b", "start"]


def _bit_count(b: int) -> int:
    return bin(b & 0xFF).count("1")


def _read_be16(pyboy: Any, addr: int) -> int:
    """Read 2 bytes big-endian (Gen 2 uses BE for HP, money, etc.)."""
    return (pyboy.memory[addr] << 8) | pyboy.memory[addr + 1]


def dump_state(pyboy: Any) -> dict:
    """Debug helper: print every address this adapter reads, for VERIFY pass.

    Call after a fresh `load_state` to a known game state. Confirms RAM
    interpretations before trusting reward shaping. Useful at first ROM run.
    """
    state = {
        "party_count": pyboy.memory[PARTY_COUNT],
        "party_species": [pyboy.memory[PARTY_SPECIES + i] for i in range(6)],
        "party_levels": [
            pyboy.memory[PARTY_MONS + n * PARTYMON_STRUCT_LEN + PARTYMON_LEVEL_OFFSET]
            for n in range(6)
        ],
        "party_hp_0": _read_be16(pyboy, PARTY_MONS + PARTYMON_HP_OFFSET),
        "party_maxhp_0": _read_be16(pyboy, PARTY_MONS + PARTYMON_MAXHP_OFFSET),
        "johto_badges_byte": hex(pyboy.memory[JOHTO_BADGES]),
        "kanto_badges_byte": hex(pyboy.memory[KANTO_BADGES]),
        "johto_badge_count": _bit_count(pyboy.memory[JOHTO_BADGES]),
        "kanto_badge_count": _bit_count(pyboy.memory[KANTO_BADGES]),
        "map_group": pyboy.memory[MAP_GROUP],
        "map_number": pyboy.memory[MAP_NUMBER],
        "x": pyboy.memory[X_COORD],
        "y": pyboy.memory[Y_COORD],
        "battle_mode": pyboy.memory[BATTLE_MODE],
        "event_flags_set_total": sum(
            _bit_count(pyboy.memory[i]) for i in range(EVENT_FLAGS_START, EVENT_FLAGS_END)
        ),
    }
    return state


# --- adapter -----------------------------------------------------------------
@dataclass
class PokemonCrystalAdapter(CartridgeAdapter):
    cartridge_title: str = "POKEMON CRYSTAL"
    platform: str = "gameboy"
    init_state: Path | None = None
    action_set: list = field(default_factory=lambda: list(ACTIONS))
    observation_shape: tuple = (3, 72, 80)

    # reward weights (CLI-overridable via dataclass replace)
    reward_scale: float = 1.0
    explore_weight: float = 1.0
    badge_weight: float = 10.0
    heal_weight: float = 10.0
    event_weight: float = 4.0
    levels_weight: float = 0.05
    stuck_weight: float = -0.05
    # Boot phase: per-action nudge + bonus when party first appears
    boot_step_reward: float = 0.01
    boot_acquire_bonus: float = 1.0

    # per-episode state (reset_episode wipes these)
    seen_coords: dict[str, int] = field(default_factory=dict)
    max_event_rew: float = 0.0
    total_healing_rew: float = 0.0
    last_health: float = 1.0
    party_size: int = 0
    died_count: int = 0
    base_event_flags: int = 0
    _total_reward: float = 0.0
    # F9.1: track previous step's party_size so the boot phase predicate fires
    # on the 0→1 transition step (when curr.party_size becomes 1). Without this,
    # the boot phase's acquire bonus is unreachable — the transition routes to
    # tutorial phase before constant_per_step's `delta > 0` check sees it.
    _prev_party_size: int = 0
    # Phased reward (lazily built on first reset_episode)
    _phased: PhasedReward | None = None

    # --- RAM reads -----------------------------------------------------------
    def _coords(self, pyboy: Any) -> tuple[int, int, int, int]:
        return (
            pyboy.memory[X_COORD],
            pyboy.memory[Y_COORD],
            pyboy.memory[MAP_GROUP],
            pyboy.memory[MAP_NUMBER],
        )

    def _hp_fraction(self, pyboy: Any) -> float:
        count = min(int(pyboy.memory[PARTY_COUNT]), 6)
        hp_sum = 0
        max_hp_sum = 0
        for n in range(count):
            base = PARTY_MONS + n * PARTYMON_STRUCT_LEN
            hp_sum += _read_be16(pyboy, base + PARTYMON_HP_OFFSET)
            max_hp_sum += _read_be16(pyboy, base + PARTYMON_MAXHP_OFFSET)
        return hp_sum / max(max_hp_sum, 1)

    def _badge_count(self, pyboy: Any) -> int:
        return _bit_count(pyboy.memory[JOHTO_BADGES]) + _bit_count(pyboy.memory[KANTO_BADGES])

    def _levels_sum(self, pyboy: Any) -> int:
        count = min(int(pyboy.memory[PARTY_COUNT]), 6)
        return sum(
            pyboy.memory[PARTY_MONS + n * PARTYMON_STRUCT_LEN + PARTYMON_LEVEL_OFFSET]
            for n in range(count)
        )

    def _event_count(self, pyboy: Any) -> int:
        return sum(
            _bit_count(pyboy.memory[i])
            for i in range(EVENT_FLAGS_START, EVENT_FLAGS_END)
        )

    def _events_progress(self, pyboy: Any) -> int:
        return max(self._event_count(pyboy) - self.base_event_flags, 0)

    def _coord_key(self, pyboy: Any) -> str:
        x, y, g, m = self._coords(pyboy)
        return f"{g}:{m}:{x}:{y}"

    def _update_seen(self, pyboy: Any) -> None:
        # F8.3: skip boot/title-screen states. Real overworld tiles have
        # party_count >= 1; boot has it at 0. Without this, the agent gets
        # spurious "exploration" credit for sitting on the title screen.
        if pyboy.memory[PARTY_COUNT] == 0:
            return
        if pyboy.memory[BATTLE_MODE] == 0:
            key = self._coord_key(pyboy)
            self.seen_coords[key] = self.seen_coords.get(key, 0) + 1

    def _stuck_penalty(self, pyboy: Any) -> int:
        return 1 if self.seen_coords.get(self._coord_key(pyboy), 0) >= 600 else 0

    def _update_heal(self, pyboy: Any) -> None:
        cur = self._hp_fraction(pyboy)
        if cur > self.last_health and pyboy.memory[PARTY_COUNT] == self.party_size:
            if self.last_health > 0:
                heal = cur - self.last_health
                self.total_healing_rew += heal * heal
            else:
                self.died_count += 1
        self.last_health = cur

    # --- reward components ---------------------------------------------------
    def _reward_components(self, pyboy: Any) -> dict[str, float]:
        cur_events = self._events_progress(pyboy)
        self.max_event_rew = max(cur_events, self.max_event_rew)
        return {
            "event": self.reward_scale * self.max_event_rew * self.event_weight,
            "heal": self.reward_scale * self.total_healing_rew * self.heal_weight,
            "badge": self.reward_scale * self._badge_count(pyboy) * self.badge_weight,
            "explore": (
                self.reward_scale * self.explore_weight * len(self.seen_coords) * 0.1
            ),
            "levels": self.reward_scale * self._levels_sum(pyboy) * self.levels_weight,
            "stuck": self.reward_scale * self._stuck_penalty(pyboy) * self.stuck_weight,
        }

    # --- phased reward construction -----------------------------------------
    def _build_phased(self) -> PhasedReward:
        """3-phase reward: boot (no party yet) → tutorial (party, no badges) →
        main (PWhiddy-style full reward).
        """
        # Closure-captured `self` so phases share adapter state
        adapter = self

        def _delta(pyboy):
            components = adapter._reward_components(pyboy)
            new_total = sum(components.values())
            delta = new_total - adapter._total_reward
            adapter._total_reward = new_total
            return delta

        # F9.1: predicate uses `_prev_party_size` (updated each step) so boot
        # also fires on the 0→1 transition step, letting constant_per_step's
        # `bonus = +1.0` reach the reward sum.
        boot_compute = constant_per_step(
            value=self.boot_step_reward,
            first_acquire_bonus=self.boot_acquire_bonus,
            acquire_key="party_size",
        )

        def _boot_compute(prev, curr, pyboy):
            # Wrap the underlying compute so we also update the previous-step
            # tracker for the NEXT step's predicate decision.
            r = boot_compute(prev, curr, pyboy)
            adapter._prev_party_size = curr.get("party_size", 0)
            return r

        boot = RewardPhase(
            name="boot",
            is_active=lambda s, p: adapter._prev_party_size == 0,
            compute=_boot_compute,
        )

        def _tutorial_compute(prev, curr, pyboy):
            adapter._update_seen(pyboy)
            adapter._update_heal(pyboy)
            r = _delta(pyboy)
            adapter._prev_party_size = curr.get("party_size", 0)
            return r

        tutorial = RewardPhase(
            name="tutorial",
            is_active=lambda s, p: s.get("party_size", 0) >= 1 and s.get("badges", 0) == 0,
            compute=_tutorial_compute,
        )

        def _main_compute(prev, curr, pyboy):
            adapter._update_seen(pyboy)
            adapter._update_heal(pyboy)
            r = _delta(pyboy)
            adapter._prev_party_size = curr.get("party_size", 0)
            return r

        main = RewardPhase(
            name="main",
            is_active=lambda s, p: True,  # fallback — fires whenever tutorial doesn't
            compute=_main_compute,
        )

        return PhasedReward(phases=[boot, tutorial, main])

    # --- CartridgeAdapter interface -----------------------------------------
    def reset_episode(self, pyboy: Any) -> None:
        self.seen_coords = {}
        self.max_event_rew = 0.0
        self.total_healing_rew = 0.0
        self.last_health = self._hp_fraction(pyboy)
        self.party_size = pyboy.memory[PARTY_COUNT]
        self.died_count = 0
        self.base_event_flags = self._event_count(pyboy)
        self._total_reward = sum(self._reward_components(pyboy).values())
        # F9.1: initialize the prev-party tracker. If reset places the agent
        # past the intro (init.state present), party_size > 0 → tutorial fires
        # from step 1; otherwise boot fires from step 1.
        self._prev_party_size = int(pyboy.memory[PARTY_COUNT])
        # Lazy phased-reward build (only first time)
        if self._phased is None:
            self._phased = self._build_phased()
        # Reset diagnostic
        self._phased.last_phase = None

    def read_game_state(self, pyboy: Any) -> dict:
        x, y, g, m = self._coords(pyboy)
        return {
            "x": x,
            "y": y,
            "map_group": g,
            "map_number": m,
            # Encode (group, number) into a single int for trajectory CSVs / arrow viz
            "map_id": g * 256 + m,
            "hp_fraction": self._hp_fraction(pyboy),
            "badges": self._badge_count(pyboy),
            "party_size": pyboy.memory[PARTY_COUNT],
            "in_battle": int(pyboy.memory[BATTLE_MODE] != 0),
        }

    def compute_reward(self, prev_state: dict, curr_state: dict, pyboy: Any) -> float:
        # Delegate to phased reward (boot → tutorial → main).
        if self._phased is None:
            self._phased = self._build_phased()
        return self._phased.compute(prev_state, curr_state, pyboy)

    def current_phase(self) -> str | None:
        """Diagnostic: which reward phase fired on the last `compute_reward` call."""
        return self._phased.last_phase if self._phased else None

    def is_done(self, state: dict) -> bool:
        return False  # episode bounded by env max_steps

    def get_trajectory_coords(self, state: dict) -> tuple[int, int, int]:
        return state["x"], state["y"], state["map_id"]


# --- local_to_global (loaded from JSON if present, else identity stub) -------
_MAP_DATA_PATH = Path(__file__).parent / "pokemon_crystal_map_data.json"
_MAP_DATA: dict | None = None
_PAD = 8


def _load_map_data() -> dict | None:
    global _MAP_DATA
    if _MAP_DATA is None and _MAP_DATA_PATH.exists():
        _MAP_DATA = json.loads(_MAP_DATA_PATH.read_text())
    return _MAP_DATA


def local_to_global(r: int, c: int, map_id: int) -> tuple[int, int]:
    """(y, x, encoded_map_id) -> (global_y, global_x).

    `map_id` here is the encoded `group * 256 + number` value from
    `get_trajectory_coords`. If `pokemon_crystal_map_data.json` is not present
    (C4 hasn't shipped yet), falls back to a simple non-overlapping packing
    keyed by map_id.
    """
    md = _load_map_data()
    if md is None:
        # Stub layout: each map gets a 64×64 cell in row-major order on a 4096-wide canvas
        cell = 64
        canvas_cols = 4096 // cell
        cell_row = map_id // canvas_cols
        cell_col = map_id % canvas_cols
        return cell_row * cell + r + _PAD, cell_col * cell + c + _PAD
    group = map_id // 256
    number = map_id % 256
    cell = md.get("groups", {}).get(str(group), {}).get(str(number))
    if cell is None:
        return _PAD, _PAD
    return cell["y_offset"] + r + _PAD, cell["x_offset"] + c + _PAD


register("POKEMON CRYSTAL")(PokemonCrystalAdapter)

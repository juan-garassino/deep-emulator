"""Phased reward shaping.

The cartridge's `compute_reward` typically wants different signals at
different points in the game's lifecycle:

- **Boot phase**: title screen, company logos, intro dialog. No party yet,
  no badges, no map. Standard reward shaping returns 0 because all
  signals are 0 — the agent has nothing to learn from. A tiny positive
  reward per action + a bonus for "first party member acquired" gets the
  agent through the intro on its own.
- **Tutorial phase**: agent has a starter, no badges yet. Explore + level
  + heal rewards apply. Per-cartridge: maybe a bonus for first map
  transition.
- **Main phase**: agent has at least one badge. Full PWhiddy-style
  reward — events, badges, heal, explore, stuck penalty.

`PhasedReward` is an ordered list of `RewardPhase`s — the first one whose
`is_active` predicate fires this step provides the reward. This composes
with the existing `CartridgeAdapter.compute_reward` interface: an adapter
can simply return `self._phased.compute(prev, curr, pyboy)`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


PhasePredicate = Callable[[dict, Any], bool]
PhaseRewardFn = Callable[[dict, dict, Any], float]


@dataclass
class RewardPhase:
    """One regime of the reward function.

    Args:
        name: short label for diagnostics
        is_active: `(curr_state: dict, emulator: Any) -> bool` — phase fires
            this step iff this returns True
        compute: `(prev_state: dict, curr_state: dict, emulator: Any) -> float`
            — the reward delta returned when this phase is active
    """
    name: str
    is_active: PhasePredicate
    compute: PhaseRewardFn


@dataclass
class PhasedReward:
    """Ordered list of `RewardPhase`s. First active phase wins.

    If no phase is active, the reward is 0. (Place a permissive fallback phase
    last with `is_active=lambda s, p: True` to act as a default.)

    Tracks `last_phase` for diagnostics — useful for verifying that an agent
    transitioned out of the boot phase as expected.
    """
    phases: list[RewardPhase] = field(default_factory=list)
    last_phase: str | None = None

    def compute(self, prev_state: dict, curr_state: dict, emulator: Any) -> float:
        for phase in self.phases:
            try:
                active = bool(phase.is_active(curr_state, emulator))
            except Exception:
                active = False
            if active:
                self.last_phase = phase.name
                try:
                    return float(phase.compute(prev_state, curr_state, emulator))
                except Exception:
                    return 0.0
        self.last_phase = None
        return 0.0

    def add(self, phase: RewardPhase) -> "PhasedReward":
        self.phases.append(phase)
        return self

    def phase_names(self) -> list[str]:
        return [p.name for p in self.phases]


# --- helper: a simple constant-per-action boot reward ----------------------
def constant_per_step(value: float = 0.01, *, first_acquire_bonus: float = 1.0,
                      acquire_key: str = "party_size") -> PhaseRewardFn:
    """A reward function suitable for the boot phase.

    Returns `value` for every action, plus `first_acquire_bonus` the step
    when `curr_state[acquire_key] > prev_state[acquire_key]` (e.g. party
    transitions from 0 → 1 when the agent picks a starter).
    """

    def _compute(prev: dict, curr: dict, _emu: Any) -> float:
        delta = curr.get(acquire_key, 0) - prev.get(acquire_key, 0)
        bonus = first_acquire_bonus if delta > 0 else 0.0
        return value + bonus

    return _compute

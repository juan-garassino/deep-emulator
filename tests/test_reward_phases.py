"""RewardPhase + PhasedReward unit tests — no emulator needed."""
from __future__ import annotations

import pytest


def test_phased_reward_picks_first_active_phase():
    from deepEmulator.core.reward import PhasedReward, RewardPhase

    pr = PhasedReward(phases=[
        RewardPhase("boot", lambda s, p: s.get("party_size", 0) == 0, lambda pr_, cu, p: 0.5),
        RewardPhase("main", lambda s, p: s.get("party_size", 0) >= 1, lambda pr_, cu, p: 5.0),
    ])
    assert pr.compute({}, {"party_size": 0}, None) == 0.5
    assert pr.last_phase == "boot"
    assert pr.compute({}, {"party_size": 1}, None) == 5.0
    assert pr.last_phase == "main"


def test_phased_reward_returns_zero_when_no_phase_active():
    from deepEmulator.core.reward import PhasedReward, RewardPhase

    pr = PhasedReward(phases=[
        RewardPhase("never", lambda s, p: False, lambda pr_, cu, p: 999.0),
    ])
    assert pr.compute({}, {}, None) == 0.0
    assert pr.last_phase is None


def test_phased_reward_first_match_wins_with_overlap():
    """Even if multiple phases would fire, only the first listed is used."""
    from deepEmulator.core.reward import PhasedReward, RewardPhase

    pr = PhasedReward(phases=[
        RewardPhase("A", lambda s, p: True, lambda pr_, cu, p: 1.0),
        RewardPhase("B", lambda s, p: True, lambda pr_, cu, p: 2.0),
    ])
    assert pr.compute({}, {}, None) == 1.0


def test_phased_reward_swallows_predicate_exception():
    """A predicate raising shouldn't kill the whole reward chain."""
    from deepEmulator.core.reward import PhasedReward, RewardPhase

    def _boom(s, p):
        raise RuntimeError("boom")

    pr = PhasedReward(phases=[
        RewardPhase("bad", _boom, lambda pr_, cu, p: 99.0),
        RewardPhase("good", lambda s, p: True, lambda pr_, cu, p: 1.0),
    ])
    assert pr.compute({}, {}, None) == 1.0
    assert pr.last_phase == "good"


def test_phased_reward_swallows_compute_exception():
    """A compute raising falls through to 0, doesn't crash the env step."""
    from deepEmulator.core.reward import PhasedReward, RewardPhase

    def _boom(prev, curr, p):
        raise ValueError("nope")

    pr = PhasedReward(phases=[
        RewardPhase("bad", lambda s, p: True, _boom),
    ])
    assert pr.compute({}, {}, None) == 0.0


def test_add_returns_self_for_chaining():
    from deepEmulator.core.reward import PhasedReward, RewardPhase

    pr = (
        PhasedReward()
        .add(RewardPhase("a", lambda s, p: True, lambda pr_, cu, p: 1.0))
        .add(RewardPhase("b", lambda s, p: True, lambda pr_, cu, p: 2.0))
    )
    assert pr.phase_names() == ["a", "b"]


def test_constant_per_step_includes_acquire_bonus():
    from deepEmulator.core.reward import constant_per_step

    fn = constant_per_step(value=0.01, first_acquire_bonus=1.0, acquire_key="party_size")
    # no change → base value
    assert fn({"party_size": 0}, {"party_size": 0}, None) == 0.01
    # acquisition (0 → 1) → bonus
    assert fn({"party_size": 0}, {"party_size": 1}, None) == 0.01 + 1.0
    # no further bonus on subsequent steps with same party size
    assert fn({"party_size": 1}, {"party_size": 1}, None) == 0.01


def test_constant_per_step_handles_missing_prev_key():
    """First step has empty prev_state — should not crash."""
    from deepEmulator.core.reward import constant_per_step

    fn = constant_per_step()
    out = fn({}, {"party_size": 0}, None)
    assert out == 0.01

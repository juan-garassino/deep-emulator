"""Multi-button combo action space + step() — logic tests, no ROM/PyBoy needed.

Ports of lixado/PyBoy-RL's two signature capabilities:
  1. combo_action_set: permutation-generated multi-button action space with
     contradictory d-pad pairs removed (MarioAISettings.GetActions).
  2. PyBoyEnv._send_buttons / step_buttons: simultaneous multi-press with
     stale-release (CustomPyBoyGym.step) — "hold A while running right".

The env's `_send_buttons` is exercised against a stub emulator that records
press/release/tick calls, so no real PyBoy or ROM is constructed.
"""
from __future__ import annotations

from deepEmulator.core.cartridge import combo_action_set


# --- combo_action_set (pure) ------------------------------------------------
def test_combo_singles_then_combos_order_stable():
    actions = combo_action_set(["left", "right", "a"], max_buttons=2)
    # singles first, in input order
    assert actions[:3] == [["left"], ["right"], ["a"]]
    # then the valid 2-wide combos ([left,right] dropped as contradictory)
    assert ["left", "a"] in actions
    assert ["right", "a"] in actions
    assert ["left", "right"] not in actions


def test_combo_drops_contradictory_dpad_pairs():
    actions = combo_action_set(["left", "right", "up", "down"], max_buttons=2)
    assert ["left", "right"] not in actions
    assert ["up", "down"] not in actions
    # non-contradictory cross pairs survive
    assert ["left", "up"] in actions


def test_combo_max_buttons_width():
    a2 = combo_action_set(["up", "a", "b"], max_buttons=2)
    assert all(len(x) <= 2 for x in a2)
    a3 = combo_action_set(["up", "a", "b"], max_buttons=3)
    assert ["up", "a", "b"] in a3


def test_combo_include_noop_prepends_empty():
    actions = combo_action_set(["a", "b"], include_noop=True)
    assert actions[0] == []


# --- env multi-button step (stubbed emulator) --------------------------------
class _StubPyBoy:
    """Records the button/tick call sequence CustomPyBoyGym.step drives."""

    def __init__(self):
        self.calls: list[tuple] = []

    def button_press(self, name):
        self.calls.append(("press", name))

    def button_release(self, name):
        self.calls.append(("release", name))

    def tick(self, n, render):
        self.calls.append(("tick", n))


class _StubCartridge:
    action_set = [["left"], ["right", "a"], ["a"]]

    def read_game_state(self, emu):
        return {"x": 0, "y": 0, "map_id": 0}

    def compute_reward(self, prev, curr, emu):
        return 0.0

    def is_done(self, state):
        return False

    def get_trajectory_coords(self, state):
        return (0, 0, 0)


def _make_env():
    """Build a PyBoyEnv shell without touching real PyBoy/ROM."""
    from deepEmulator.platforms.gameboy import PyBoyEnv

    env = PyBoyEnv.__new__(PyBoyEnv)
    env.pyboy = _StubPyBoy()
    env.cartridge = _StubCartridge()
    env.headless = True
    env.action_freq = 24
    env.press_ticks = 8
    env._held = set()
    return env


def test_resolve_buttons_from_index_and_list():
    env = _make_env()
    assert env._resolve_buttons(1) == ["right", "a"]  # index into combo set
    assert env._resolve_buttons(["up", "b"]) == ["up", "b"]  # direct list


def test_send_buttons_presses_all_then_releases_all():
    env = _make_env()
    env._send_buttons(["right", "a"])
    presses = [n for kind, n in env.pyboy.calls if kind == "press"]
    releases = [n for kind, n in env.pyboy.calls if kind == "release"]
    assert set(presses) == {"right", "a"}  # both pressed together
    assert set(releases) >= {"right", "a"}  # both released after the press window
    assert env._held == set()  # nothing left held after the step


def test_send_buttons_releases_stale_from_previous_combo():
    env = _make_env()
    env._held = {"left"}  # a stale hold from a prior action
    env._send_buttons(["right"])
    # "left" must be released before/around pressing "right"
    assert ("release", "left") in env.pyboy.calls
    assert ("press", "right") in env.pyboy.calls


def test_step_buttons_returns_correct_shape_contract():
    import numpy as np

    env = _make_env()
    # minimal state the step bookkeeping touches
    env.reward_clip = 5.0
    env.max_steps = 100
    env._step_count = 0
    env._prev_state = {}
    env.frame_stack = 3
    env._frame_h, env._frame_w = 72, 80
    env._screen_stack = np.zeros((3, 72, 80), dtype=np.uint8)
    env._grab_frame = lambda: np.zeros((72, 80), dtype=np.uint8)

    obs, reward, terminated, truncated, info = env.step_buttons(1)  # ["right","a"]
    assert obs.shape == (3, 72, 80)
    assert isinstance(reward, float)
    assert terminated is False and truncated is False
    assert "game_state" in info and "trajectory" in info

"""Atari backend — ale-py direct, no gymnasium.

Wraps `ale_py.ALEInterface` and applies canonical DeepMind preprocessing
inside the env: frame-skip 4 with max-pool over t/t-1 (anti-flicker),
grayscale, 84×84 resize, 4-frame stack, NOOP-reset 0-30.

The cartridge adapter only sets reward policy + life-loss policy; the env
owns the DM preprocessing.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np

from deepEmulator.core.cartridge import CartridgeAdapter
from deepEmulator.core.env import EmulatorEnv
from deepEmulator.core.spaces import Box, Discrete


def _resize_grayscale_84(frame: np.ndarray) -> np.ndarray:
    """(H, W) uint8 -> (84, 84) uint8 via block downsampling (no cv2 dep)."""
    h, w = frame.shape
    sh, sw = h // 84, w // 84
    if sh < 1 or sw < 1:
        # input smaller than 84 — pad. Shouldn't happen for Atari (210x160).
        out = np.zeros((84, 84), dtype=np.uint8)
        out[:h, :w] = frame
        return out
    crop = frame[: sh * 84, : sw * 84]
    return crop.reshape(84, sh, 84, sw).mean(axis=(1, 3)).astype(np.uint8)


class AtariEnv(EmulatorEnv):
    """ALE-py backend with DeepMind preprocessing.

    Args:
        cartridge: adapter declaring rom_name, reward policy, life-loss policy
        rom_path: optional override; otherwise resolved via ale_py.roms
        headless: True for offscreen (default); False uses ALE's SDL display if compiled in
        frame_skip: number of frames per env step (max-pool last 2)
        frame_stack: stacked grayscale frames in the observation
        noop_max: random NOOPs on reset [0, noop_max)
        sticky_actions: probability action repeats (0 = deterministic)
    """

    def __init__(
        self,
        cartridge: CartridgeAdapter,
        rom_path: str | Path | None = None,
        headless: bool = True,
        frame_skip: int = 4,
        frame_stack: int = 4,
        noop_max: int = 30,
        sticky_actions: float = 0.0,
        max_steps: int = 27_000,  # 27k * 4 = ~108k ALE frames (DM standard)
        seed: int | None = None,
    ):
        super().__init__(cartridge)
        from ale_py import ALEInterface  # local import — optional dep

        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.noop_max = noop_max
        self.max_steps = max_steps
        self._rng = random.Random(seed)

        self.ale = ALEInterface()
        if seed is not None:
            self.ale.setInt("random_seed", int(seed))
        self.ale.setFloat("repeat_action_probability", float(sticky_actions))
        self.ale.setBool("display_screen", not headless)
        self.ale.setBool("sound", False)

        rom = Path(rom_path) if rom_path else _resolve_rom(getattr(cartridge, "rom_name", ""))
        self.ale.loadROM(str(rom))

        self._minimal_actions = list(self.ale.getMinimalActionSet())
        self.action_space = Discrete(len(self._minimal_actions))
        # write the resolved action set back so bundles record the real linkage
        cartridge.action_set = [int(a) for a in self._minimal_actions]

        self._raw_h, self._raw_w = self.ale.getScreenDims()  # typically (210, 160)
        self._screen_a = np.zeros((self._raw_h, self._raw_w), dtype=np.uint8)
        self._screen_b = np.zeros((self._raw_h, self._raw_w), dtype=np.uint8)
        self._stack = np.zeros((frame_stack, 84, 84), dtype=np.uint8)

        self.observation_space = Box(
            low=0.0, high=255.0, shape=(frame_stack, 84, 84), dtype=np.dtype(np.uint8)
        )

        self._step_count = 0
        self._lives = 0
        self._prev_state: dict = {}

    # --- internal helpers ----------------------------------------------------
    def _grab_grayscale(self, buf: np.ndarray) -> None:
        self.ale.getScreenGrayscale(buf)

    def _push_frame(self, frame84: np.ndarray) -> None:
        self._stack = np.roll(self._stack, 1, axis=0)
        self._stack[0] = frame84

    def _step_with_skip(self, ale_action: int) -> tuple[float, bool]:
        """Frame-skip with max-pool over the last two raw frames."""
        total_reward = 0.0
        terminated = False
        for i in range(self.frame_skip):
            reward = self.ale.act(ale_action)
            total_reward += float(reward)
            if i == self.frame_skip - 2:
                self._grab_grayscale(self._screen_a)
            elif i == self.frame_skip - 1:
                self._grab_grayscale(self._screen_b)
            if self.ale.game_over():
                terminated = True
                break
        if self.frame_skip == 1:
            self._grab_grayscale(self._screen_b)
            self._screen_a[:] = self._screen_b
        max_frame = np.maximum(self._screen_a, self._screen_b)
        self._push_frame(_resize_grayscale_84(max_frame))
        return total_reward, terminated

    # --- EmulatorEnv API -----------------------------------------------------
    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = random.Random(seed)
        self.ale.reset_game()
        # Random NOOPs (action 0 is always NOOP in ALE)
        for _ in range(self._rng.randint(0, self.noop_max)):
            self.ale.act(0)
            if self.ale.game_over():
                self.ale.reset_game()
        self.cartridge.reset_episode(self.ale)
        self._grab_grayscale(self._screen_a)
        first = _resize_grayscale_84(self._screen_a)
        self._stack[:] = 0
        for _ in range(self.frame_stack):
            self._push_frame(first)
        self._step_count = 0
        self._lives = self.ale.lives()
        self._prev_state = self.cartridge.read_game_state(self.ale)
        return self._stack.copy(), {"game_state": self._prev_state, "lives": self._lives}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        ale_action = self._minimal_actions[int(action)]
        raw_reward, terminated = self._step_with_skip(ale_action)

        curr_state = self.cartridge.read_game_state(self.ale)
        curr_state["raw_reward"] = raw_reward
        reward = self.cartridge.compute_reward(self._prev_state, curr_state, self.ale)

        # Life-loss as soft terminal (credit-assignment trick) — adapter opts in
        lives_now = self.ale.lives()
        life_lost = lives_now < self._lives
        self._lives = lives_now
        adapter_done = self.cartridge.is_done(curr_state)
        if getattr(self.cartridge, "terminal_on_life_loss", False) and life_lost:
            terminated = True

        self._step_count += 1
        truncated = self._step_count >= self.max_steps
        self._prev_state = curr_state

        info = {
            "game_state": curr_state,
            "trajectory": self.cartridge.get_trajectory_coords(curr_state),
            "lives": lives_now,
            "raw_reward": raw_reward,
            "ale_terminated": terminated,
            "adapter_done": adapter_done,
        }
        return self._stack.copy(), float(reward), bool(terminated or adapter_done), truncated, info

    def render(self) -> np.ndarray:
        buf = np.zeros((self._raw_h, self._raw_w, 3), dtype=np.uint8)
        self.ale.getScreenRGB(buf)
        return buf

    def close(self) -> None:
        try:
            del self.ale
        except Exception:
            pass


def _resolve_rom(rom_name: str) -> Path:
    """Resolve a ROM by name via ale_py.roms.get_rom_path; falls back to roms/."""
    if not rom_name:
        raise ValueError("cartridge must declare a rom_name (e.g. 'pong')")
    try:
        from ale_py import roms  # type: ignore

        getter = getattr(roms, "get_rom_path", None) or getattr(roms, "rom_path", None)
        if getter is not None:
            return Path(getter(rom_name))
    except Exception:
        pass
    candidate = Path("roms") / f"{rom_name}.bin"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"could not resolve ROM {rom_name!r} via ale_py.roms or roms/{rom_name}.bin"
    )
